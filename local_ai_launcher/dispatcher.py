"""
Smart Agent Dispatcher for local_ai_launcher.
Orchestrates work between the CPU Fast-Path (BGE-Reranker & CPU ONNX Decision Gater on DDR5)
and GPU Deep Reasoning (Qwen 3.8 27B on RTX 5080) with a strict context budget guardrail,
dynamic intent classification, sliding-window circuit breaking, anchored context compaction,
connection pooling, priority scheduling, and SSE streaming.
"""

import re
import json
import time
import random
import logging
import asyncio
import collections
import urllib.request
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

from local_ai_launcher.config import config
from local_ai_launcher.backends.cpu_reranker import rerank
from local_ai_launcher.backends.cpu_decision import (
    evaluate_decision_sync,
    evaluate_decision_async,
    compact_text_sync,
    compact_text_async,
    check_cpu_engine_health
)
from local_ai_launcher.backends.cpu_onnx_gater import (
    evaluate_onnx_decision_sync,
    evaluate_onnx_decision_async
)

logger = logging.getLogger("smart_dispatcher")


from local_ai_launcher.memory_clawback import is_strata_active


# =====================================================================
# Persistent HTTP Connection Pooling (Zero Socket Exhaustion)
# =====================================================================

_SHARED_ASYNC_CLIENT = None
_SHARED_SYNC_CLIENT = None


def get_shared_async_client(timeout: float = 60.0):
    """
    Returns process-singleton httpx.AsyncClient with persistent keep-alive
    connection pooling to prevent socket exhaustion and TIME_WAIT accumulation.
    """
    global _SHARED_ASYNC_CLIENT
    import httpx
    if _SHARED_ASYNC_CLIENT is None or _SHARED_ASYNC_CLIENT.is_closed:
        _SHARED_ASYNC_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=60.0
            ),
            timeout=httpx.Timeout(timeout, connect=3.0)
        )
    return _SHARED_ASYNC_CLIENT


def get_shared_sync_client(timeout: float = 60.0):
    """
    Returns process-singleton httpx.Client with persistent keep-alive connection pooling.
    """
    global _SHARED_SYNC_CLIENT
    import httpx
    if _SHARED_SYNC_CLIENT is None or _SHARED_SYNC_CLIENT.is_closed:
        _SHARED_SYNC_CLIENT = httpx.Client(
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=60.0
            ),
            timeout=httpx.Timeout(timeout, connect=3.0)
        )
    return _SHARED_SYNC_CLIENT


async def close_shared_clients():
    """Gracefully closes persistent pooled HTTP clients."""
    global _SHARED_ASYNC_CLIENT, _SHARED_SYNC_CLIENT
    if _SHARED_ASYNC_CLIENT is not None and not _SHARED_ASYNC_CLIENT.is_closed:
        try:
            await _SHARED_ASYNC_CLIENT.aclose()
        except Exception:
            pass
        _SHARED_ASYNC_CLIENT = None

    if _SHARED_SYNC_CLIENT is not None and not _SHARED_SYNC_CLIENT.is_closed:
        try:
            _SHARED_SYNC_CLIENT.close()
        except Exception:
            pass
        _SHARED_SYNC_CLIENT = None


# =====================================================================
# Sliding-Window EWMA Circuit Breaker with Jitter
# =====================================================================

class SlidingWindowCircuitBreaker:
    """
    Sliding-window circuit breaker with failure rate tracking and jittered backoff.
    Detects flapping backends (e.g. 60% error rate) even if failures are non-consecutive,
    and prevents thundering herd retries via randomized cooldown jitter.
    """
    def __init__(
        self,
        window_size: int = 20,
        failure_rate_threshold: float = 0.35,
        base_cooldown: float = 15.0,
        jitter_range: float = 3.0,
        failure_threshold: Optional[int] = None,
        cooldown_seconds: Optional[float] = None
    ):
        self.window_size = window_size
        self.failure_rate_threshold = failure_rate_threshold
        self.base_cooldown = cooldown_seconds if cooldown_seconds is not None else base_cooldown
        self.jitter_range = jitter_range
        self.history = collections.deque(maxlen=window_size)
        self.last_failure_time = 0.0
        self.current_cooldown = self.base_cooldown
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_threshold = failure_threshold or 3
        self.failure_count = 0

    def record_success(self):
        self.failure_count = 0
        self.history.append(True)
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.history.clear()

    def record_failure(self):
        self.failure_count += 1
        self.history.append(False)
        self.last_failure_time = time.time()

        max_jitter = min(self.jitter_range, self.base_cooldown * 0.2)
        jitter = random.uniform(-max_jitter, max_jitter)
        self.current_cooldown = max(0.01, self.base_cooldown + jitter)

        consecutive_trip = self.failure_count >= self.failure_threshold
        window_trip = False
        if len(self.history) >= 3:
            failures = sum(1 for res in self.history if not res)
            if (failures / len(self.history)) >= self.failure_rate_threshold:
                window_trip = True

        if consecutive_trip or window_trip:
            if self.state != "OPEN":
                logger.warning(
                    f"Circuit breaker TRIPPED to OPEN (consecutive: {self.failure_count}, "
                    f"window error rate: {sum(1 for r in self.history if not r)}/{len(self.history)}). "
                    f"Cooldown: {self.current_cooldown:.1f}s."
                )
            self.state = "OPEN"

    def can_attempt(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.current_cooldown:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state for single probe.")
                return True
            return False
        if self.state == "HALF_OPEN":
            return True
        return True

    def reset(self):
        self.history.clear()
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"


# Alias for backward compatibility
BackendCircuitBreaker = SlidingWindowCircuitBreaker

cpu_circuit_breaker = SlidingWindowCircuitBreaker(
    failure_threshold=getattr(config, 'dispatcher_circuit_breaker_threshold', 3),
    cooldown_seconds=getattr(config, 'dispatcher_circuit_breaker_cooldown', 15.0)
)
gpu_circuit_breaker = SlidingWindowCircuitBreaker(
    failure_threshold=3,
    cooldown_seconds=15.0
)


# =====================================================================
# Priority Tiers & Intent Classification
# =====================================================================

class TaskPriority:
    HIGH = "HIGH"        # Fast-track tool gates, permission checks, invariant assertions
    NORMAL = "NORMAL"    # Code generation, reasoning, analysis
    LOW = "LOW"          # Background context compaction, passive reranking


class RoutingIntent:
    DISCRETE_GATE = "DISCRETE_GATE"
    GENERATIVE_REASONING = "GENERATIVE_REASONING"
    HYBRID_COMPACT = "HYBRID_COMPACT"


def classify_intent_and_tier(
    task_type: str,
    query: str,
    candidate_options: Optional[List[str]],
    tokens: int,
    max_cpu_tokens: int = 4000,
    compaction_threshold: int = 3500,
    policy: str = "auto"
) -> Tuple[str, str]:
    """
    Classifies the user query and determines the optimal execution tier.
    Returns (intent, routing_reason).
    """
    clean_policy = (policy or getattr(config, 'dispatcher_default_policy', 'auto')).lower()

    if clean_policy == "gpu_direct":
        return RoutingIntent.GENERATIVE_REASONING, "POLICY_GPU_DIRECT_ENFORCED"
    if clean_policy == "cpu_only":
        return RoutingIntent.DISCRETE_GATE, "POLICY_CPU_ONLY_ENFORCED"
    if clean_policy == "hybrid_compact":
        return RoutingIntent.HYBRID_COMPACT, "POLICY_HYBRID_COMPACT_ENFORCED"

    is_triage_task = task_type in ("tool_gate", "diff_invariant_check", "syntax_lint", "triage", "classification", "permission")
    has_candidate_options = bool(candidate_options and len(candidate_options) >= 2)

    lower_query = query.lower()
    generative_triggers = ("implement ", "write a function", "explain why", "step by step", "refactor this", "quicksort")
    is_generative_request = any(t in lower_query for t in generative_triggers)

    if (is_triage_task or has_candidate_options) and not is_generative_request:
        if tokens <= max_cpu_tokens:
            return RoutingIntent.DISCRETE_GATE, "BOUNDED_DISCRETE_GATE_CPU"
        else:
            enable_compaction = getattr(config, 'dispatcher_enable_compaction', True)
            if enable_compaction and tokens >= compaction_threshold:
                return RoutingIntent.HYBRID_COMPACT, "DISCRETE_OVER_BUDGET_COMPACTION"
            return RoutingIntent.GENERATIVE_REASONING, "DISCRETE_OVER_CPU_TOKEN_LIMIT"

    enable_compaction = getattr(config, 'dispatcher_enable_compaction', True)
    if enable_compaction and tokens >= compaction_threshold:
        return RoutingIntent.HYBRID_COMPACT, "LARGE_CONTEXT_HYBRID_COMPACTION"

    return RoutingIntent.GENERATIVE_REASONING, "OPEN_ENDED_GENERATIVE_GPU"


# =====================================================================
# Deterministic Prompt Canonicalization & Anchored Context
# =====================================================================

def estimate_tokens(text: str) -> int:
    """
    Approximates token length with sensitivity to code symbols, camelCase, and whitespace.
    BPE tokenizer heuristic: max of word expansion (1.5x) and symbol/char density (chars / 3.4).
    """
    if not text:
        return 0
    word_count = len(text.split())
    char_count = len(text)
    return max(int(word_count * 1.5), int(char_count / 3.4)) + 1


MAX_RERANK_CHUNK_TOKENS = 3000


def adaptive_filter_documents(
    query: str,
    raw_documents: List[str],
    min_relevance: float = 0.65,
    min_chunks: int = 3,
    max_chunks: int = 5,
    max_chunk_tokens: Optional[int] = None
) -> List[str]:
    """
    Reranks documents on CPU and applies an adaptive relevance threshold.
    Avoids arbitrary top_n cutoff by retaining all passages with relevance >= min_relevance,
    bounded between min_chunks and max_chunks.

    Guardrail: Bypasses reranking for whole source files (>3,000 tokens) because cross-encoder
    models (e.g. bge-reranker-v2-m3) truncate input at sequence limits, leading to inaccurate
    relevance scoring and wasted compute on monolithic source code files.
    """
    if not raw_documents:
        return []
    if len(raw_documents) <= min_chunks or not getattr(config, 'use_cpu_reranker', True):
        return raw_documents[:max_chunks]

    effective_max_tokens = max_chunk_tokens or getattr(config, 'cpu_reranker_max_doc_tokens', MAX_RERANK_CHUNK_TOKENS)
    # If candidate documents contain whole source files, bypass CPU cross-encoder reranking
    if any(estimate_tokens(doc) > effective_max_tokens for doc in raw_documents):
        logger.info(
            f"Candidate documents contain whole source files (>{effective_max_tokens} tokens). "
            f"Bypassing CPU cross-encoder reranking to prevent truncation degradation."
        )
        return raw_documents[:max_chunks]

    try:
        ranked = rerank(query=query, documents=raw_documents, top_n=max_chunks, max_doc_tokens=effective_max_tokens)
        selected = [
            item["document"]
            for item in ranked
            if item.get("relevance_score", 0.0) >= min_relevance
        ]
        if len(selected) < min_chunks:
            selected = [item["document"] for item in ranked[:min_chunks]]

        return selected[:max_chunks]
    except Exception as e:
        logger.warning(f"Reranking error: {e}. Falling back to top raw snippets.")
        return raw_documents[:min_chunks]


def canonicalize_prompt_prefix(diff_context: str, documents: List[str]) -> str:
    """
    Guarantees deterministic byte-for-byte context prefix across agent turns
    to maximize RAM KV cache hits on llama-server (reducing prompt eval from ~800ms to <25ms).
    Strips volatile git index timestamps and sorts documents by path/content deterministically.
    """
    cleaned_diff = ""
    if diff_context:
        # Strip volatile git commit/index line e.g. "index abc1234..def5678 100644"
        cleaned_diff = re.sub(r"^index [0-9a-f]+\.\.[0-9a-f]+.*$", "", diff_context, flags=re.MULTILINE)
        cleaned_diff = re.sub(r"\n{3,}", "\n\n", cleaned_diff).strip()

    sorted_docs = sorted([doc.strip() for doc in documents if doc and doc.strip()])

    parts = []
    if cleaned_diff:
        parts.append(f"[Code Diff Context]:\n{cleaned_diff}")
    if sorted_docs:
        retrieval_context = "\n\n---\n\n".join(sorted_docs)
        parts.append(f"[Retrieved Reference Context]:\n{retrieval_context}")

    return "\n\n".join(parts).strip()


def extract_anchored_context(diff_context: str) -> Tuple[str, str]:
    """
    Separates code diff into immutable syntax anchors (+, -, @@, definitions)
    and compressible background noise (verbose unchanged lines, logs).
    Returns (immutable_anchors, compressible_noise).
    """
    if not diff_context:
        return "", ""

    anchor_lines = []
    noise_lines = []

    for line in diff_context.splitlines():
        if (
            line.startswith("diff --git") or
            line.startswith("@@") or
            line.startswith("---") or
            line.startswith("+++") or
            line.startswith("+") or
            line.startswith("-") or
            any(kw in line for kw in ("class ", "def ", "function ", "interface ", "export const "))
        ):
            anchor_lines.append(line)
        else:
            noise_lines.append(line)

    return "\n".join(anchor_lines), "\n".join(noise_lines)


def _build_dispatch_context(
    query: str,
    raw_documents: Optional[List[str]],
    diff_context: str,
    min_relevance: float = 0.65,
    min_chunks: int = 3,
    max_chunks: int = 5
) -> Tuple[str, List[str], float]:
    """
    Builds static prefix-aligned context text and retrieves filtered documents.
    Uses canonical prompt prefixing to lock DDR5 RAM KV cache blocks.
    """
    t0 = time.perf_counter()
    filtered_documents: List[str] = []

    if raw_documents and len(raw_documents) > 0 and getattr(config, 'use_cpu_reranker', True) and not is_strata_active():
        filtered_documents = adaptive_filter_documents(
            query=query,
            raw_documents=raw_documents,
            min_relevance=min_relevance,
            min_chunks=min_chunks,
            max_chunks=max_chunks
        )
    elif raw_documents:
        filtered_documents = raw_documents[:min_chunks]

    context_text = canonicalize_prompt_prefix(diff_context or "", filtered_documents)
    rerank_ms = round((time.perf_counter() - t0) * 1000, 2)
    return context_text, filtered_documents, rerank_ms


def allocate_token_budget(
    query: str,
    diff_context: str,
    raw_documents: Optional[List[str]],
    target_budget: int = 4000,
    min_relevance: float = 0.65,
    min_chunks: int = 3,
    max_chunks: int = 5
) -> Tuple[str, List[str], int]:
    """
    Allocates tokens across diff context, query, and retrieved documents,
    preserving modified hunk lines (+, -) and trimming lower-scoring documents if over budget.
    """
    context_text, filtered_documents, _ = _build_dispatch_context(
        query=query,
        raw_documents=raw_documents,
        diff_context=diff_context,
        min_relevance=min_relevance,
        min_chunks=min_chunks,
        max_chunks=max_chunks
    )
    tokens = estimate_tokens((context_text + " " + query).strip())

    if tokens <= target_budget or not filtered_documents:
        return context_text, filtered_documents, tokens

    while len(filtered_documents) > min_chunks and tokens > target_budget:
        filtered_documents.pop()
        context_text = canonicalize_prompt_prefix(diff_context or "", filtered_documents)
        tokens = estimate_tokens((context_text + " " + query).strip())

    return context_text, filtered_documents, tokens


# =====================================================================
# Main Asynchronous Dispatcher
# =====================================================================

async def smart_agent_dispatch_async(
    task_type: str,
    query: str,
    raw_documents: Optional[List[str]] = None,
    diff_context: str = "",
    candidate_options: Optional[List[str]] = None,
    confidence_threshold: float = 0.85,
    max_cpu_tokens: int = 4000,
    gpu_port: Optional[int] = None,
    cpu_decision_port: Optional[int] = None,
    cpu_onnx_gater_port: Optional[int] = None,
    system_prompt: Optional[str] = None,
    min_relevance: float = 0.65,
    min_chunks: int = 3,
    max_chunks: int = 5,
    temperature: float = 1.45,
    policy: str = "auto",
    compaction_threshold: int = 3500,
    priority: str = TaskPriority.NORMAL,
    slots: Optional[Dict[str, Any]] = None,
    cpu_decider_port: Optional[int] = None,
    slot_confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Asynchronously routes an agent query across CPU and GPU inference tiers:
    1. Prunes retrieval noise using adaptive BGE-Reranker-v2-m3 on CPU.
    2. Dynamic intent classification & canonicalized DDR5 RAM prefix matching.
    3. Dedicated CPU Decider-4B v2.1 multi-slot extraction on port 8084 (~40ms).
    4. Dedicated CPU ONNX Decision/Gater on port 8083 (~15ms zero-generation typed decisions).
    5. Standalone CPU llama-server on port 8081 (if enabled and gater escalated).
    6. Anchored context compaction on large contexts before GPU escalation.
    7. Escalates ambiguous or large-context generative tasks to Qwen 27B on GPU.
    """
    start_time = time.perf_counter()
    timings = {"rerank_ms": 0.0, "cpu_eval_ms": 0.0, "gpu_ms": 0.0, "total_ms": 0.0}

    context_text, filtered_documents, rerank_ms = _build_dispatch_context(
        query=query,
        raw_documents=raw_documents,
        diff_context=diff_context,
        min_relevance=min_relevance,
        min_chunks=min_chunks,
        max_chunks=max_chunks
    )
    timings["rerank_ms"] = rerank_ms

    tokens = estimate_tokens((context_text + " " + query).strip())
    target_options = candidate_options or ["PASS", "FAIL", "ESCALATE"]

    intent, routing_reason = classify_intent_and_tier(
        task_type=task_type,
        query=query,
        candidate_options=candidate_options,
        tokens=tokens,
        max_cpu_tokens=max_cpu_tokens,
        compaction_threshold=compaction_threshold,
        policy=policy
    )

    tokens_saved_gpu = 0

    # --- STAGE 1.8: CPU Multi-Slot Fast-Path (decider-4b on Port 8084) ---
    if slots and getattr(config, 'use_decider_4b', True) and not is_strata_active():
        decider_start = time.perf_counter()
        target_decider_port = cpu_decider_port or getattr(config, 'decider_4b_port', 8084)
        try:
            from local_ai_launcher.backends.cpu_decider import evaluate_decider_slots_async
            context_snippet = context_text if context_text else query
            slot_res = await evaluate_decider_slots_async(
                context=context_snippet,
                slots=slots,
                port=target_decider_port,
                temperature=temperature
            )
            timings["cpu_eval_ms"] = round((time.perf_counter() - decider_start) * 1000, 2)
            eval_slots = slot_res.get("slots") or slot_res.get("results") or {}

            min_required = slot_confidence_threshold if slot_confidence_threshold is not None else 0.20
            all_confident = True
            min_conf = 1.0
            for s_name, s_val in eval_slots.items():
                c = s_val.get("confidence", 0.0) if isinstance(s_val, dict) else 1.0
                min_conf = min(min_conf, c)
                if c < min_required:
                    all_confident = False

            if eval_slots and all_confident:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                timings["total_ms"] = latency_ms
                return {
                    "action": "SLOTS_EVALUATED",
                    "tier": "CPU_DECIDER_4B",
                    "slots": eval_slots,
                    "confidence": min_conf,
                    "token_estimate": tokens,
                    "retrieved_count": len(filtered_documents),
                    "latency_ms": latency_ms,
                    "fallback_triggered": False,
                    "routing_reason": "DECIDER_4B_SLOT_EXTRACTION",
                    "tokens_saved_gpu": tokens,
                    "timing_breakdown": timings,
                    "cache_alignment": "DECIDER_4B_CPU"
                }
            else:
                logger.debug(f"Decider-4b slots low confidence ({min_conf:.2f} < {min_required}). Fallback to subsequent tier.")
        except Exception as e:
            logger.warning(f"CPU decider-4b slots execution error: {e}. Fallback to subsequent tier.")

    # --- STAGE 2: CPU Fast-Path Gating ---
    if intent == RoutingIntent.DISCRETE_GATE and not is_strata_active():
        # Priority 2.1: Dedicated CPU ONNX Decision / Gater (Port 8083)
        if getattr(config, 'use_cpu_onnx_gater', True):
            onnx_start = time.perf_counter()
            prompt_text = f"Context:\n{context_text}\n\nEvaluation Target: {query}\nAllowed Choices: {target_options}"
            try:
                target_onnx_port = cpu_onnx_gater_port or getattr(config, 'cpu_onnx_gater_port', 8083)
                decision = await evaluate_onnx_decision_async(
                    prompt=prompt_text,
                    candidate_options=target_options,
                    port=target_onnx_port,
                    temperature=temperature
                )
                timings["cpu_eval_ms"] = round((time.perf_counter() - onnx_start) * 1000, 2)
                choice = decision.get("choice")
                confidence = decision.get("confidence", 0.0)

                if confidence >= confidence_threshold and choice in target_options and choice != "ESCALATE":
                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    timings["total_ms"] = latency_ms
                    return {
                        "action": choice,
                        "tier": "CPU_ONNX_GATER",
                        "confidence": confidence,
                        "distribution": decision.get("distribution", {}),
                        "token_estimate": tokens,
                        "retrieved_count": len(filtered_documents),
                        "latency_ms": latency_ms,
                        "fallback_triggered": False,
                        "routing_reason": routing_reason,
                        "tokens_saved_gpu": tokens,
                        "timing_breakdown": timings,
                        "cache_alignment": "ONNX_RUNTIME_CPU"
                    }
                else:
                    logger.debug(f"CPU ONNX gater escalated or low confidence ({confidence:.2f} < {confidence_threshold}). Fallback to subsequent tier.")
            except Exception as e:
                logger.warning(f"CPU ONNX gater execution error: {e}. Fallback to subsequent tier.")

        # Priority 2.2: Standalone CPU Decision llama-server (Port 8081)
        if getattr(config, 'use_cpu_decision', True):
            if not cpu_circuit_breaker.can_attempt():
                routing_reason = "CIRCUIT_BREAKER_CPU_OPEN_FAIL_FAST"
                logger.warning("CPU circuit breaker is OPEN. Bypassing CPU directly to GPU.")
            else:
                cpu_start = time.perf_counter()
                prompt_text = f"Context:\n{context_text}\n\nEvaluation Target: {query}\nAllowed Choices: {target_options}"
                
                try:
                    decision = await evaluate_decision_async(
                        prompt=prompt_text,
                        candidate_options=target_options,
                        port=cpu_decision_port or getattr(config, 'cpu_decision_port', 8081),
                        temperature=temperature
                    )
                    timings["cpu_eval_ms"] = round((time.perf_counter() - cpu_start) * 1000, 2)
                    cpu_circuit_breaker.record_success()

                    choice = decision.get("choice")
                    confidence = decision.get("confidence", 0.0)

                    if confidence >= confidence_threshold and choice in target_options and choice != "ESCALATE":
                        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                        timings["total_ms"] = latency_ms
                        return {
                            "action": choice,
                            "tier": "CPU_FAST_PATH",
                            "confidence": confidence,
                            "distribution": decision.get("distribution", {}),
                            "token_estimate": tokens,
                            "retrieved_count": len(filtered_documents),
                            "latency_ms": latency_ms,
                            "fallback_triggered": False,
                            "routing_reason": routing_reason,
                            "tokens_saved_gpu": tokens,
                            "timing_breakdown": timings,
                            "cache_alignment": "CANONICAL_PREFIX"
                        }
                    else:
                        routing_reason = f"LOW_CONFIDENCE_ESCALATION ({confidence:.2f} < {confidence_threshold})"
                except Exception as e:
                    logger.warning(f"CPU fast-path execution failure: {e}")
                    cpu_circuit_breaker.record_failure()
                    routing_reason = "CPU_FAILURE_ESCALATE_GPU"

    # --- STAGE 2.5: Anchored Context Compaction (Hybrid Pipeline) ---
    tier_name = "GPU_QWEN_27B"
    if intent == RoutingIntent.HYBRID_COMPACT and getattr(config, 'use_cpu_decision', True) and not is_strata_active():
        if cpu_circuit_breaker.can_attempt() and context_text:
            try:
                compact_start = time.perf_counter()
                anchors, noise = extract_anchored_context(context_text)
                target_compress = noise if noise else context_text
                compacted = await compact_text_async(
                    text=target_compress,
                    max_summary_tokens=600,
                    port=cpu_decision_port or getattr(config, 'cpu_llama_server_port', 8081)
                )
                if compacted:
                    compact_tokens = estimate_tokens(compacted)
                    tokens_saved_gpu = max(0, tokens - compact_tokens)
                    if anchors:
                        context_text = f"[Immutable Code Anchors]:\n{anchors}\n\n[Compacted Reference Context]:\n{compacted}"
                    else:
                        context_text = f"[Compacted Reference Context]:\n{compacted}"
                    tier_name = "HYBRID_COMPACT_GPU"
                    routing_reason = f"HYBRID_COMPACTION_SAVED_{tokens_saved_gpu}_TOKENS"
                    timings["cpu_eval_ms"] = round((time.perf_counter() - compact_start) * 1000, 2)
                    tokens = estimate_tokens((context_text + " " + query).strip())
            except Exception as e:
                logger.debug(f"Compaction failed: {e}. Falling back to uncompacted GPU prompt.")

    # --- STAGE 3: GPU Deep Reasoning (Persistent Pooled Keep-Alive Client) ---
    target_gpu_port = gpu_port or getattr(config, 'localai_port', 8080)
    gpu_url = f"http://127.0.0.1:{target_gpu_port}/v1/chat/completions"

    default_sys = (
        "You are an expert staff software engineer. Provide a precise, grounded solution."
        if ("code" in task_type or "diff" in task_type)
        else "You are an expert research and synthesis assistant."
    )
    user_content = f"{context_text}\n\nTask:\n{query}".strip() if context_text else f"Task:\n{query}".strip()
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt or default_sys},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 2048,
        "temperature": 0.6
    }

    gpu_start = time.perf_counter()
    try:
        client = get_shared_async_client(timeout=60.0)
        resp = await client.post(gpu_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        timings["gpu_ms"] = round((time.perf_counter() - gpu_start) * 1000, 2)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        timings["total_ms"] = latency_ms

        return {
            "action": content,
            "tier": tier_name,
            "token_estimate": tokens,
            "retrieved_count": len(filtered_documents),
            "latency_ms": latency_ms,
            "fallback_triggered": False,
            "routing_reason": routing_reason,
            "tokens_saved_gpu": tokens_saved_gpu,
            "timing_breakdown": timings,
            "cache_alignment": "CANONICAL_PREFIX"
        }
    except Exception as e:
        logger.error(f"Error querying GPU generative server at {gpu_url}: {e}")
        timings["gpu_ms"] = round((time.perf_counter() - gpu_start) * 1000, 2)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        timings["total_ms"] = latency_ms
        return {
            "action": f"Error during GPU execution: {e}",
            "tier": "ERROR",
            "token_estimate": tokens,
            "retrieved_count": len(filtered_documents),
            "latency_ms": latency_ms,
            "fallback_triggered": True,
            "routing_reason": f"GPU_EXECUTION_FAILURE: {e}",
            "tokens_saved_gpu": 0,
            "timing_breakdown": timings
        }


# =====================================================================
# Synchronous Dispatcher Wrapper
# =====================================================================

def smart_agent_dispatch(
    task_type: str,
    query: str,
    raw_documents: Optional[List[str]] = None,
    diff_context: str = "",
    candidate_options: Optional[List[str]] = None,
    confidence_threshold: float = 0.85,
    max_cpu_tokens: int = 4000,
    gpu_port: Optional[int] = None,
    cpu_decision_port: Optional[int] = None,
    cpu_onnx_gater_port: Optional[int] = None,
    system_prompt: Optional[str] = None,
    min_relevance: float = 0.65,
    min_chunks: int = 3,
    max_chunks: int = 5,
    temperature: float = 1.45,
    policy: str = "auto",
    compaction_threshold: int = 3500,
    priority: str = TaskPriority.NORMAL,
    slots: Optional[Dict[str, Any]] = None,
    cpu_decider_port: Optional[int] = None,
    slot_confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Synchronously routes an agent query across CPU and GPU inference tiers.
    Maintains 100% backward compatibility for synchronous callers.
    """
    start_time = time.perf_counter()
    timings = {"rerank_ms": 0.0, "cpu_eval_ms": 0.0, "gpu_ms": 0.0, "total_ms": 0.0}

    context_text, filtered_documents, rerank_ms = _build_dispatch_context(
        query=query,
        raw_documents=raw_documents,
        diff_context=diff_context,
        min_relevance=min_relevance,
        min_chunks=min_chunks,
        max_chunks=max_chunks
    )
    timings["rerank_ms"] = rerank_ms

    tokens = estimate_tokens((context_text + " " + query).strip())
    target_options = candidate_options or ["PASS", "FAIL", "ESCALATE"]

    intent, routing_reason = classify_intent_and_tier(
        task_type=task_type,
        query=query,
        candidate_options=candidate_options,
        tokens=tokens,
        max_cpu_tokens=max_cpu_tokens,
        compaction_threshold=compaction_threshold,
        policy=policy
    )

    tokens_saved_gpu = 0

    # --- STAGE 1.8: CPU Multi-Slot Fast-Path (decider-4b on Port 8084) ---
    if slots and getattr(config, 'use_decider_4b', True) and not is_strata_active():
        decider_start = time.perf_counter()
        target_decider_port = cpu_decider_port or getattr(config, 'decider_4b_port', 8084)
        try:
            from local_ai_launcher.backends.cpu_decider import evaluate_decider_slots_sync
            context_snippet = context_text if context_text else query
            slot_res = evaluate_decider_slots_sync(
                context=context_snippet,
                slots=slots,
                port=target_decider_port,
                temperature=temperature
            )
            timings["cpu_eval_ms"] = round((time.perf_counter() - decider_start) * 1000, 2)
            eval_slots = slot_res.get("slots") or slot_res.get("results") or {}

            min_required = slot_confidence_threshold if slot_confidence_threshold is not None else 0.20
            all_confident = True
            min_conf = 1.0
            for s_name, s_val in eval_slots.items():
                c = s_val.get("confidence", 0.0) if isinstance(s_val, dict) else 1.0
                min_conf = min(min_conf, c)
                if c < min_required:
                    all_confident = False

            if eval_slots and all_confident:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                timings["total_ms"] = latency_ms
                return {
                    "action": "SLOTS_EVALUATED",
                    "tier": "CPU_DECIDER_4B",
                    "slots": eval_slots,
                    "confidence": min_conf,
                    "token_estimate": tokens,
                    "retrieved_count": len(filtered_documents),
                    "latency_ms": latency_ms,
                    "fallback_triggered": False,
                    "routing_reason": "DECIDER_4B_SLOT_EXTRACTION",
                    "tokens_saved_gpu": tokens,
                    "timing_breakdown": timings,
                    "cache_alignment": "DECIDER_4B_CPU"
                }
            else:
                logger.debug(f"Decider-4b slots low confidence ({min_conf:.2f} < {min_required}). Fallback to subsequent tier.")
        except Exception as e:
            logger.warning(f"CPU decider-4b slots execution error: {e}. Fallback to subsequent tier.")

    # --- STAGE 2: CPU Fast-Path Gating ---
    if intent == RoutingIntent.DISCRETE_GATE and not is_strata_active():
        # Priority 2.1: Dedicated CPU ONNX Decision / Gater (Port 8083)
        if getattr(config, 'use_cpu_onnx_gater', True):
            onnx_start = time.perf_counter()
            prompt_text = f"Context:\n{context_text}\n\nEvaluation Target: {query}\nAllowed Choices: {target_options}"
            try:
                target_onnx_port = cpu_onnx_gater_port or getattr(config, 'cpu_onnx_gater_port', 8083)
                decision = evaluate_onnx_decision_sync(
                    prompt=prompt_text,
                    candidate_options=target_options,
                    port=target_onnx_port,
                    temperature=temperature
                )
                timings["cpu_eval_ms"] = round((time.perf_counter() - onnx_start) * 1000, 2)
                choice = decision.get("choice")
                confidence = decision.get("confidence", 0.0)

                if confidence >= confidence_threshold and choice in target_options and choice != "ESCALATE":
                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    timings["total_ms"] = latency_ms
                    return {
                        "action": choice,
                        "tier": "CPU_ONNX_GATER",
                        "confidence": confidence,
                        "distribution": decision.get("distribution", {}),
                        "token_estimate": tokens,
                        "retrieved_count": len(filtered_documents),
                        "latency_ms": latency_ms,
                        "fallback_triggered": False,
                        "routing_reason": routing_reason,
                        "tokens_saved_gpu": tokens,
                        "timing_breakdown": timings,
                        "cache_alignment": "ONNX_RUNTIME_CPU"
                    }
                else:
                    logger.debug(f"CPU ONNX gater escalated or low confidence ({confidence:.2f} < {confidence_threshold}). Fallback to subsequent tier.")
            except Exception as e:
                logger.warning(f"CPU ONNX gater execution error: {e}. Fallback to subsequent tier.")

        # Priority 2.2: Standalone CPU Decision llama-server (Port 8081)
        if getattr(config, 'use_cpu_decision', True):
            if not cpu_circuit_breaker.can_attempt():
                routing_reason = "CIRCUIT_BREAKER_CPU_OPEN_FAIL_FAST"
            else:
                cpu_start = time.perf_counter()
                prompt_text = f"Context:\n{context_text}\n\nEvaluation Target: {query}\nAllowed Choices: {target_options}"
                
                try:
                    decision = evaluate_decision_sync(
                        prompt=prompt_text,
                        candidate_options=target_options,
                        port=cpu_decision_port or getattr(config, 'cpu_decision_port', 8081),
                        temperature=temperature
                    )
                    timings["cpu_eval_ms"] = round((time.perf_counter() - cpu_start) * 1000, 2)
                    cpu_circuit_breaker.record_success()

                    choice = decision.get("choice")
                    confidence = decision.get("confidence", 0.0)

                    if confidence >= confidence_threshold and choice in target_options and choice != "ESCALATE":
                        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                        timings["total_ms"] = latency_ms
                        return {
                            "action": choice,
                            "tier": "CPU_FAST_PATH",
                            "confidence": confidence,
                            "distribution": decision.get("distribution", {}),
                            "token_estimate": tokens,
                            "retrieved_count": len(filtered_documents),
                            "latency_ms": latency_ms,
                            "fallback_triggered": False,
                            "routing_reason": routing_reason,
                            "tokens_saved_gpu": tokens,
                            "timing_breakdown": timings,
                            "cache_alignment": "CANONICAL_PREFIX"
                        }
                    else:
                        routing_reason = f"LOW_CONFIDENCE_ESCALATION ({confidence:.2f} < {confidence_threshold})"
                except Exception as e:
                    logger.warning(f"CPU fast-path execution failure: {e}")
                    cpu_circuit_breaker.record_failure()
                    routing_reason = "CPU_FAILURE_ESCALATE_GPU"

    # --- STAGE 2.5: Anchored Context Compaction ---
    tier_name = "GPU_QWEN_27B"
    if intent == RoutingIntent.HYBRID_COMPACT and getattr(config, 'use_cpu_decision', True) and not is_strata_active():
        if cpu_circuit_breaker.can_attempt() and context_text:
            try:
                compact_start = time.perf_counter()
                anchors, noise = extract_anchored_context(context_text)
                target_compress = noise if noise else context_text
                compacted = compact_text_sync(
                    text=target_compress,
                    max_summary_tokens=600,
                    port=cpu_decision_port or getattr(config, 'cpu_llama_server_port', 8081)
                )
                if compacted:
                    compact_tokens = estimate_tokens(compacted)
                    tokens_saved_gpu = max(0, tokens - compact_tokens)
                    if anchors:
                        context_text = f"[Immutable Code Anchors]:\n{anchors}\n\n[Compacted Reference Context]:\n{compacted}"
                    else:
                        context_text = f"[Compacted Reference Context]:\n{compacted}"
                    tier_name = "HYBRID_COMPACT_GPU"
                    routing_reason = f"HYBRID_COMPACTION_SAVED_{tokens_saved_gpu}_TOKENS"
                    timings["cpu_eval_ms"] = round((time.perf_counter() - compact_start) * 1000, 2)
                    tokens = estimate_tokens((context_text + " " + query).strip())
            except Exception as e:
                logger.debug(f"Compaction failed: {e}. Falling back to uncompacted GPU prompt.")

    # --- STAGE 3: GPU Deep Reasoning ---
    target_gpu_port = gpu_port or getattr(config, 'localai_port', 8080)
    gpu_url = f"http://127.0.0.1:{target_gpu_port}/v1/chat/completions"

    default_sys = (
        "You are an expert staff software engineer. Provide a precise, grounded solution."
        if ("code" in task_type or "diff" in task_type)
        else "You are an expert research and synthesis assistant."
    )
    user_content = f"{context_text}\n\nTask:\n{query}".strip() if context_text else f"Task:\n{query}".strip()
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt or default_sys},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 2048,
        "temperature": 0.6
    }

    gpu_start = time.perf_counter()
    try:
        client = get_shared_sync_client(timeout=60.0)
        resp = client.post(gpu_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        timings["gpu_ms"] = round((time.perf_counter() - gpu_start) * 1000, 2)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        timings["total_ms"] = latency_ms

        return {
            "action": content,
            "tier": tier_name,
            "token_estimate": tokens,
            "retrieved_count": len(filtered_documents),
            "latency_ms": latency_ms,
            "fallback_triggered": False,
            "routing_reason": routing_reason,
            "tokens_saved_gpu": tokens_saved_gpu,
            "timing_breakdown": timings,
            "cache_alignment": "CANONICAL_PREFIX"
        }
    except Exception as e:
        logger.error(f"Error querying GPU generative server at {gpu_url}: {e}")
        timings["gpu_ms"] = round((time.perf_counter() - gpu_start) * 1000, 2)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        timings["total_ms"] = latency_ms
        return {
            "action": f"Error during GPU execution: {e}",
            "tier": "ERROR",
            "token_estimate": tokens,
            "retrieved_count": len(filtered_documents),
            "latency_ms": latency_ms,
            "fallback_triggered": True,
            "routing_reason": f"GPU_EXECUTION_FAILURE: {e}",
            "tokens_saved_gpu": 0,
            "timing_breakdown": timings
        }


# =====================================================================
# Server-Sent Events (SSE) Streaming Dispatcher
# =====================================================================

async def smart_agent_dispatch_stream(
    task_type: str,
    query: str,
    raw_documents: Optional[List[str]] = None,
    diff_context: str = "",
    candidate_options: Optional[List[str]] = None,
    confidence_threshold: float = 0.85,
    max_cpu_tokens: int = 4000,
    gpu_port: Optional[int] = None,
    cpu_decision_port: Optional[int] = None,
    cpu_onnx_gater_port: Optional[int] = None,
    system_prompt: Optional[str] = None,
    min_relevance: float = 0.65,
    min_chunks: int = 3,
    max_chunks: int = 5,
    temperature: float = 1.45,
    policy: str = "auto",
    slots: Optional[Dict[str, Any]] = None,
    cpu_decider_port: Optional[int] = None,
    slot_confidence_threshold: Optional[float] = None
) -> AsyncGenerator[str, None]:
    """
    Yields Server-Sent Events (SSE) formatted chunks using pooled keep-alive connection.
    If fast-path CPU fires, returns a single discrete event immediately.
    If GPU generative fires, streams tokens in real-time.
    """
    context_text, filtered_documents, _ = _build_dispatch_context(
        query=query,
        raw_documents=raw_documents,
        diff_context=diff_context,
        min_relevance=min_relevance,
        min_chunks=min_chunks,
        max_chunks=max_chunks
    )
    tokens = estimate_tokens((context_text + " " + query).strip())
    target_options = candidate_options or ["PASS", "FAIL", "ESCALATE"]

    # --- STAGE 1.8: CPU Multi-Slot Fast-Path (decider-4b on Port 8084) ---
    if slots and getattr(config, 'use_decider_4b', True):
        target_decider_port = cpu_decider_port or getattr(config, 'decider_4b_port', 8084)
        try:
            from local_ai_launcher.backends.cpu_decider import evaluate_decider_slots_async
            context_snippet = context_text if context_text else query
            slot_res = await evaluate_decider_slots_async(
                context=context_snippet,
                slots=slots,
                port=target_decider_port,
                temperature=temperature
            )
            eval_slots = slot_res.get("slots") or slot_res.get("results") or {}
            min_required = slot_confidence_threshold if slot_confidence_threshold is not None else 0.20
            all_confident = True
            min_conf = 1.0
            for s_name, s_val in eval_slots.items():
                c = s_val.get("confidence", 0.0) if isinstance(s_val, dict) else 1.0
                min_conf = min(min_conf, c)
                if c < min_required:
                    all_confident = False

            if eval_slots and all_confident:
                chunk = {
                    "choices": [
                        {"delta": {"content": json.dumps(eval_slots)}, "index": 0, "finish_reason": "stop"}
                    ],
                    "tier": "CPU_DECIDER_4B",
                    "confidence": min_conf,
                    "slots": eval_slots
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
        except Exception as e:
            logger.warning(f"CPU decider-4b stream slots error: {e}")

    intent, _ = classify_intent_and_tier(
        task_type=task_type,
        query=query,
        candidate_options=candidate_options,
        tokens=tokens,
        max_cpu_tokens=max_cpu_tokens,
        policy=policy
    )

    if intent == RoutingIntent.DISCRETE_GATE:
        # Priority 1: Dedicated CPU ONNX Decision / Gater (Port 8083)
        if getattr(config, 'use_cpu_onnx_gater', True):
            prompt_text = f"Context:\n{context_text}\n\nEvaluation Target: {query}\nAllowed Choices: {target_options}"
            try:
                target_onnx_port = cpu_onnx_gater_port or getattr(config, 'cpu_onnx_gater_port', 8083)
                decision = await evaluate_onnx_decision_async(
                    prompt=prompt_text,
                    candidate_options=target_options,
                    port=target_onnx_port,
                    temperature=temperature
                )
                choice = decision.get("choice")
                confidence = decision.get("confidence", 0.0)

                if confidence >= confidence_threshold and choice in target_options and choice != "ESCALATE":
                    chunk = {
                        "choices": [
                            {"delta": {"content": choice}, "index": 0, "finish_reason": "stop"}
                        ],
                        "tier": "CPU_ONNX_GATER",
                        "confidence": confidence,
                        "distribution": decision.get("distribution", {})
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            except Exception as e:
                logger.warning(f"CPU ONNX gater stream failure: {e}")

        # Priority 2: Standalone CPU Decision llama-server (Port 8081)
        if getattr(config, 'use_cpu_decision', True):
            if cpu_circuit_breaker.can_attempt():
                prompt_text = f"Context:\n{context_text}\n\nEvaluation Target: {query}\nAllowed Choices: {target_options}"
                try:
                    decision = await evaluate_decision_async(
                        prompt=prompt_text,
                        candidate_options=target_options,
                        port=cpu_decision_port or getattr(config, 'cpu_decision_port', 8081),
                        temperature=temperature
                    )
                    choice = decision.get("choice")
                    confidence = decision.get("confidence", 0.0)

                    if confidence >= confidence_threshold and choice in target_options and choice != "ESCALATE":
                        chunk = {
                            "choices": [
                                {"delta": {"content": choice}, "index": 0, "finish_reason": "stop"}
                            ],
                            "tier": "CPU_FAST_PATH",
                            "confidence": confidence,
                            "distribution": decision.get("distribution", {})
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                except Exception as e:
                    logger.warning(f"CPU fast-path stream failure: {e}")

    # Fallback to GPU generative streaming
    target_gpu_port = gpu_port or getattr(config, 'localai_port', 8080)
    gpu_url = f"http://127.0.0.1:{target_gpu_port}/v1/chat/completions"

    default_sys = (
        "You are an expert staff software engineer. Provide a precise, grounded solution."
        if ("code" in task_type or "diff" in task_type)
        else "You are an expert research and synthesis assistant."
    )
    user_content = f"{context_text}\n\nTask:\n{query}".strip() if context_text else f"Task:\n{query}".strip()
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt or default_sys},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 2048,
        "temperature": 0.6,
        "stream": True
    }

    try:
        client = get_shared_async_client(timeout=120.0)
        async with client.stream("POST", gpu_url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n\n"
    except Exception as e:
        err_chunk = {
            "choices": [{"delta": {"content": f"Streaming error: {e}"}, "finish_reason": "error"}],
            "tier": "ERROR"
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# =====================================================================
# Concurrent Priority Batch Dispatcher
# =====================================================================

async def smart_agent_batch_dispatch_async(
    tasks: List[Dict[str, Any]],
    concurrency_limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Executes multiple agent dispatch tasks concurrently bounded by a semaphore
    with high-priority task fast-tracking.
    """
    limit = concurrency_limit or getattr(config, 'dispatcher_batch_concurrency_limit', 4)
    semaphore = asyncio.Semaphore(limit)

    def _prio_key(item: Tuple[int, Dict[str, Any]]) -> int:
        idx, t = item
        p = str(t.get("priority", "NORMAL")).upper()
        if p == "HIGH" or t.get("task_type") in ("tool_gate", "permission", "diff_invariant_check"):
            return 0
        elif p == "LOW":
            return 2
        return 1

    indexed_tasks = list(enumerate(tasks))
    sorted_tasks = sorted(indexed_tasks, key=_prio_key)
    results = [None] * len(tasks)

    async def _worker(orig_idx: int, task_params: Dict[str, Any]):
        async with semaphore:
            res = await smart_agent_dispatch_async(**task_params)
            results[orig_idx] = res

    await asyncio.gather(*[_worker(idx, t) for idx, t in sorted_tasks])
    return results


# =====================================================================
# CLM System 1 Mitigations & High-Speed Gating
# =====================================================================

def validate_powershell_ast(command: str) -> Tuple[bool, str]:
    """
    Native PowerShell Abstract Syntax Tree (AST) validation.
    Mitigation for Fast-Path shell execution: ensures command contains
    no arbitrary script blocks, command expressions, or multi-element pipelines.
    """
    if not command or not command.strip():
        return False, "Empty command"

    stripped = command.strip()
    escaped_cmd = stripped.replace("'", "''")
    ps_code = f"""$tokens = $null; $errors = $null;
$ast = [System.Management.Automation.Language.Parser]::ParseInput('{escaped_cmd}', [ref]$tokens, [ref]$errors);
if ($errors.Count -gt 0) {{ exit 1 }}
$unsafeCmds = @('iex', 'invoke-expression', 'invoke-command', 'start-process', 'set-executionpolicy', 'encodedcommand')
$unsafe = $ast.FindAll({{
    param($n)
    ($n -is [System.Management.Automation.Language.CommandExpressionAst]) -or
    ($n -is [System.Management.Automation.Language.MemberAst]) -or
    ($n -is [System.Management.Automation.Language.InvokeMemberExpressionAst]) -or
    ($n -is [System.Management.Automation.Language.ParenExpressionAst]) -or
    ($n -is [System.Management.Automation.Language.ScriptBlockAst] -and $n -ne $ast) -or
    ($n -is [System.Management.Automation.Language.PipelineAst] -and $n.PipelineElements.Count -gt 1) -or
    ($n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -ne $null -and $unsafeCmds -contains $n.GetCommandName().ToLower())
}}, $true);
if ($unsafe.Count -gt 0) {{ exit 1 }} else {{ exit 0 }}
"""

    try:
        import subprocess
        import base64
        b64 = base64.b64encode(ps_code.encode("utf-16le")).decode("ascii")
        res = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", b64],
            capture_output=True,
            text=True,
            timeout=5.0
        )
        if res.returncode == 0:
            return True, "AST Validation Passed"
        return False, "PowerShell AST identified unsafe expressions or syntax error"
    except Exception as e:
        # Fallback lexical analysis if PowerShell execution fails
        dangerous = (";", "&&", "||", "|", "`", "$", "iex", "invoke-expression", "invoke-webrequest", "curl")
        if any(d in stripped.lower() for d in dangerous):
            return False, f"PowerShell unavailable; lexical check rejected command ({e})"
        return True, "PowerShell unavailable; lexical fallback allowed benign command"


def evaluate_security_triage(
    prompt: str,
    refusal_patterns: Optional[List[str]] = None,
    window_size: int = 64,
    stride: int = 32
) -> Dict[str, Any]:
    """
    Fast-path security triage with sliding-window adversarial camouflage mitigation
    and three-tier score calibration (BLOCK >= 0.88, AUDIT 0.72-0.88, SAFE < 0.72).
    """
    default_refusal_patterns = [
        "ignore previous instructions", "bypass safety guardrails", "jailbreak",
        "reveal system prompt", "malicious payload", "delete production database",
        "format c:", "rmdir /s /q c:\\", "disable antivirus", "exfiltrate private key"
    ]
    patterns = [p.lower() for p in (refusal_patterns or default_refusal_patterns)]

    tokens = prompt.split()
    max_refusal_sim = 0.0
    matched_pattern = ""

    # 1. Sliding Window Adversarial Camouflage Mitigation
    if len(tokens) <= window_size:
        windows = [" ".join(tokens)]
    else:
        windows = [
            " ".join(tokens[i:i + window_size])
            for i in range(0, len(tokens) - window_size + 1, stride)
        ]

    for win in windows:
        win_lower = win.lower()
        for pat in patterns:
            if pat in win_lower:
                sim = 0.95
            else:
                pat_words = set(pat.split())
                overlap = sum(1 for w in pat_words if w in win_lower)
                sim = overlap / len(pat_words) if pat_words else 0.0
            if sim > max_refusal_sim:
                max_refusal_sim = sim
                matched_pattern = pat

    # 2. Three-Tier Score Calibration
    if max_refusal_sim >= 0.88:
        return {
            "action": "BLOCK",
            "reason": f"Security Refusal Policy Triggered (pattern: {matched_pattern})",
            "score": round(max_refusal_sim, 3),
            "audit_required": True
        }

    audit_flag = max_refusal_sim >= 0.72

    # 3. Fast-Path Trivial Shell Command Evaluation
    stripped = prompt.strip()
    benign_starters = ("dir", "ls", "get-childitem", "git status", "git branch", "git log", "echo ", "pwd")
    if any(stripped.lower().startswith(b) for b in benign_starters) and not audit_flag:
        is_safe, reason = validate_powershell_ast(stripped)
        if is_safe:
            return {
                "action": "FAST_PATH_EXECUTE",
                "command": stripped,
                "score": round(max_refusal_sim, 3),
                "audit_required": False
            }
        else:
            return {
                "action": "BLOCK",
                "reason": f"AST Validation Failed for Fast-Path Shell: {reason}",
                "score": round(max_refusal_sim, 3),
                "audit_required": True
            }

    return {
        "action": "ROUTE_TO_QWEN_GPU",
        "score": round(max_refusal_sim, 3),
        "audit_required": audit_flag
    }


def two_stage_patch_verification(
    issue_text: str,
    candidate_patches: List[str],
    repo_dir: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Two-Stage Agentic Code & Trajectory Verification (Best-of-N Ranking).
    Mitigations:
    1. Surface Syntax Bias Gate: Deterministic syntax check (Python ast.parse or diff structure check)
       discards invalid patches immediately before neural ranking.
    2. Semantic Relevance: Reranks surviving valid patches using the CPU cross-encoder.
    """
    if not candidate_patches:
        return {"ranked_candidates": [], "rejected_syntax_count": 0, "best_patch": None}

    import ast
    valid_candidates = []
    rejected_count = 0

    # Stage 1: Deterministic Syntax Validation Gate
    for idx, patch in enumerate(candidate_patches):
        is_valid = True
        # If patch is standalone python script/function
        if "def " in patch or "import " in patch:
            try:
                ast.parse(patch)
            except SyntaxError:
                is_valid = False

        # If patch is a unified diff
        if is_valid and ("diff --git" in patch or "@@ " in patch):
            has_hunk = "@@ " in patch
            has_headers = "--- " in patch and "+++ " in patch
            if not (has_hunk and has_headers):
                is_valid = False

        if is_valid:
            valid_candidates.append({"id": idx, "patch": patch})
        else:
            rejected_count += 1

    if not valid_candidates:
        return {
            "ranked_candidates": [],
            "rejected_syntax_count": rejected_count,
            "error": "All generated patches failed syntax parsing gate",
            "best_patch": None
        }

    # Stage 2: Cross-Encoder Semantic Verification Ranking
    docs = [c["patch"] for c in valid_candidates]
    query = f"Issue: {issue_text}\nGoal: Select correct patch."
    try:
        ranked = rerank(query=query, documents=docs, top_n=len(docs))
        ranked_results = []
        for r in ranked:
            orig_idx = valid_candidates[r["index"]]["id"]
            ranked_results.append({
                "candidate_id": orig_idx,
                "relevance_score": r["relevance_score"],
                "patch": valid_candidates[r["index"]]["patch"]
            })
        best_patch = ranked_results[0]["patch"] if ranked_results else valid_candidates[0]["patch"]
        return {
            "ranked_candidates": ranked_results,
            "rejected_syntax_count": rejected_count,
            "best_patch": best_patch
        }
    except Exception as e:
        logger.warning(f"Reranking error in patch verification: {e}")
        return {
            "ranked_candidates": [{"candidate_id": c["id"], "relevance_score": 0.5, "patch": c["patch"]} for c in valid_candidates],
            "rejected_syntax_count": rejected_count,
            "best_patch": valid_candidates[0]["patch"]
        }


def shortlist_tools_for_turn(
    state_text: str,
    tools: List[Dict[str, Any]],
    top_k: int = 3,
    min_delta: float = 0.15
) -> Dict[str, Any]:
    """
    Transparent CLM Tool Shortlisting for Turn.
    Applies lexical negation masking, CSLS hubness mitigation, and confidence-gated window.
    """
    from local_ai_launcher.backends.cpu_onnx_gater import shortlist_tools_sync
    return shortlist_tools_sync(state=state_text, tools=tools, top_k=top_k, min_delta=min_delta)

