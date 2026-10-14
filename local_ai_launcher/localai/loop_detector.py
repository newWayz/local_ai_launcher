"""
Real-time Local AI Model Loop Detection & Mitigation Module.

Provides:
1. StreamingLoopDetector: Sliding window n-gram and cyclic sentence autocorrelation detector
   for in-flight SSE streams (/v1/chat/completions with stream=True).
2. detect_and_truncate_loop: Non-streaming counterpart for standard JSON completions.

Zero external dependencies (uses standard library: re, collections, typing, logging).
"""

import re
import logging
from collections import deque, Counter
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("localai_loop_detector")


class StreamingLoopDetector:
    """
    Detects autoregressive repetition loops during real-time token streaming.
    
    Handles:
    - Variable token/chunk slicing across SSE delta packets.
    - Variable loop periodicity (period 1: A-A-A; period 2: A-B-A-B; period 3: A-B-C-A-B-C).
    - Variable block/sentence length.
    - Lexical jitter and near-duplicate variations using Jaccard word-set similarity.
    - Code block suppression to avoid false positives on repetitive unit test suites or tables.
    - Automatic <think> tag synthesis to ensure clean rendering in agent harnesses.
    """

    def __init__(
        self,
        ngram_size: int = 8,
        ngram_threshold: int = 3,
        cycle_sim_threshold: float = 0.82,
        min_sentence_len: int = 35,
        max_history_words: int = 1500,
        enabled: bool = True
    ):
        try:
            self.ngram_size = max(1, int(ngram_size))
        except (ValueError, TypeError):
            self.ngram_size = 8

        try:
            self.ngram_threshold = max(1, int(ngram_threshold))
        except (ValueError, TypeError):
            self.ngram_threshold = 3

        try:
            self.cycle_sim_threshold = max(0.01, min(1.0, float(cycle_sim_threshold)))
        except (ValueError, TypeError):
            self.cycle_sim_threshold = 0.82

        try:
            self.min_sentence_len = max(0, int(min_sentence_len))
        except (ValueError, TypeError):
            self.min_sentence_len = 35

        try:
            self.max_history_words = max(50, int(max_history_words))
        except (ValueError, TypeError):
            self.max_history_words = 1500

        # Guard against negative numbers / truthiness traps for enabled
        if isinstance(enabled, (int, float)):
            self.enabled = False if enabled <= 0 else True
        elif isinstance(enabled, str):
            s = enabled.strip().lower()
            self.enabled = False if s in ("false", "0", "-1", "off", "no", "disable", "disabled") else True
        else:
            self.enabled = bool(enabled)

        self.full_text = ""
        self.unprocessed_text = ""
        self.words: List[str] = []
        self.ngram_counts: Counter = Counter()
        self.sentences: List[str] = []
        self.sentence_buffer = ""

        self.in_code_block = False
        self.in_thinking_block = False

        self.loop_detected = False
        self.reason = ""
        self.loop_type = ""
        self.trigger_index = -1

    def _sim(self, s1: str, s2: str) -> float:
        """Compute Jaccard similarity across normalized alphanumeric word sets."""
        w1 = set(re.findall(r"\b[a-zA-Z0-9_]+\b", s1.lower()))
        w2 = set(re.findall(r"\b[a-zA-Z0-9_]+\b", s2.lower()))
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def feed(self, delta: str, is_reasoning: bool = False) -> bool:
        """
        Feed an incoming streaming delta (content or reasoning_content).
        Returns True if a loop is detected, False otherwise.
        """
        if not self.enabled or self.loop_detected or not delta:
            return self.loop_detected

        self.full_text += delta
        self.unprocessed_text += delta
        self.sentence_buffer += delta

        # Track thinking tags state
        if is_reasoning:
            self.in_thinking_block = True
        else:
            last_open = self.full_text.rfind("<think>")
            last_close = self.full_text.rfind("</think>")
            self.in_thinking_block = (last_open != -1 and last_open > last_close)

        # Track markdown code block fence state
        # Using accumulated text count prevents desynchronization from split fences or multiple fences in one delta
        self.in_code_block = (self.full_text.count("```") % 2 == 1)

        # 1. Chunk-Boundary Safe Word Tokenization
        # Only parse complete words up to the last whitespace/delimiter
        last_delim_idx = max(
            self.unprocessed_text.rfind(" "),
            self.unprocessed_text.rfind("\n"),
            self.unprocessed_text.rfind("\t"),
        )
        if last_delim_idx != -1:
            ready_chunk = self.unprocessed_text[:last_delim_idx]
            self.unprocessed_text = self.unprocessed_text[last_delim_idx:]

            new_words = re.findall(r"\b[a-zA-Z0-9_]+\b", ready_chunk.lower())
            for w in new_words:
                self.words.append(w)
                if len(self.words) >= self.ngram_size:
                    ng = tuple(self.words[-self.ngram_size:])
                    self.ngram_counts[ng] += 1
                    
                    # If inside a code block, require higher repeat threshold (5x) to accommodate test tables
                    thresh = self.ngram_threshold if not self.in_code_block else (self.ngram_threshold + 2)
                    if self.ngram_counts[ng] >= thresh:
                        self.loop_detected = True
                        self.loop_type = "ngram"
                        self.trigger_index = len(self.full_text)
                        self.reason = (
                            f"Multi-word {self.ngram_size}-gram repeated {self.ngram_counts[ng]} times: "
                            f"'{' '.join(ng)}'"
                        )
                        logger.warning(f"[LoopDetector] {self.reason}")
                        return True

                # Enforce sliding window: discard oldest word and decrement its n-gram count
                if len(self.words) > self.max_history_words:
                    old_ng = tuple(self.words[:self.ngram_size])
                    if self.ngram_counts[old_ng] > 1:
                        self.ngram_counts[old_ng] -= 1
                    else:
                        self.ngram_counts.pop(old_ng, None)
                    self.words.pop(0)

        # 2. Sentence / Clause Cyclic Autocorrelation
        # Only evaluate sentences outside code blocks to prevent false alarms on parametrized tests/tables
        if not self.in_code_block and any(punct in delta for punct in [".", "\n", "?", "!", ";"]):
            parts = re.split(r"[\.\n\?!;]+", self.sentence_buffer)
            if len(parts) > 1:
                for p in parts[:-1]:
                    s = p.strip()
                    if len(s) >= self.min_sentence_len:
                        self.sentences.append(s)
                        if len(self.sentences) > 50:
                            self.sentences.pop(0)
                        n_s = len(self.sentences)
                        # Check periods p = 1 (A-A-A), p = 2 (A-B-A-B), p = 3 (A-B-C-A-B-C)
                        for period in [1, 2, 3]:
                            if n_s >= period * 2:
                                matches = 0
                                for offset in range(period):
                                    idx_curr = -1 - offset
                                    idx_prev = -1 - offset - period
                                    sim = self._sim(self.sentences[idx_curr], self.sentences[idx_prev])
                                    if sim >= self.cycle_sim_threshold:
                                        matches += 1
                                if matches == period:
                                    self.loop_detected = True
                                    self.loop_type = "cycle"
                                    self.trigger_index = len(self.full_text)
                                    self.reason = (
                                        f"Sentence cycle match (period={period}): "
                                        f"'{self.sentences[-1][:60]}...' matched '{self.sentences[-1-period][:60]}...'"
                                    )
                                    logger.warning(f"[LoopDetector] {self.reason}")
                                    return True
                self.sentence_buffer = parts[-1]

        return False

    def build_halt_chunk(self, req_id: str, model_name: str, created: int) -> dict:
        """
        Builds a safe, compliant OpenAI chat completion chunk signaling completion termination.
        Synthesizes closing </think> tag if interrupted while in reasoning monologue.
        """
        delta_dict: Dict[str, Any] = {}
        
        if self.in_thinking_block:
            delta_dict["reasoning_content"] = (
                "\n\n[Reasoning halted: repetition loop detected by local_ai_launcher gateway.]"
            )
            delta_dict["content"] = (
                "\n</think>\n\n[Output interrupted: The model entered a repetitive reasoning loop. "
                "Halting turn to prevent runaway GPU consumption. Please choose an alternate hypothesis or test.]"
            )
        else:
            delta_dict["content"] = (
                "\n\n[Response halted: repetition loop detected by local_ai_launcher gateway. "
                "Please retry with an alternate approach or prompt.]"
            )

        return {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta_dict,
                "finish_reason": "stop"
            }]
        }


def detect_and_truncate_loop(
    text: str,
    ngram_size: int = 8,
    ngram_threshold: int = 3,
    cycle_sim_threshold: float = 0.82,
    min_sentence_len: int = 35,
    is_reasoning: bool = False
) -> Tuple[str, bool, str]:
    """
    Non-streaming helper: evaluates full string for repetition loops.
    If loop is detected, truncates the string at the onset of repetition and appends a halt notice.
    Returns: (cleaned_text, loop_detected, reason)
    """
    if not text or len(text) < 150:
        return text, False, ""

    detector = StreamingLoopDetector(
        ngram_size=ngram_size,
        ngram_threshold=ngram_threshold,
        cycle_sim_threshold=cycle_sim_threshold,
        min_sentence_len=min_sentence_len
    )

    # Feed in character chunks to simulate stream and catch exact trigger point
    chunk_size = 40
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if detector.feed(chunk, is_reasoning=is_reasoning):
            cut_point = max(0, detector.trigger_index)
            truncated = text[:cut_point].rstrip()
            notice = (
                "\n\n[Response truncated: repetition loop detected by local_ai_launcher gateway.]"
            )
            return truncated + notice, True, detector.reason

    return text, False, ""
