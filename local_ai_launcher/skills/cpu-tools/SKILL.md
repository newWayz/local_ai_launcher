---
name: cpu-tools
description: High-speed CPU cross-encoder reranking (~25ms), von System One discrete gating (~12ms), and Decider-4B multi-slot extraction without GPU VRAM overhead.
---

# CPU Auxiliary Microservices

The local AI engine runs dedicated CPU microservices in DDR5 host RAM for high-throughput evaluations without loading or evicting GPU models:

1. **CPU Reranker** (`BAAI/bge-reranker-v2-m3` on Port 8082 / Gateway 8080):
   - Fast semantic passage & code ranking in ~25ms on DDR5 RAM.
   - Use for scoring and ranking candidate passages, docstrings, and search chunks (<800–1,000 tokens).
   - **Guardrail**: Whole source files (>3,000 tokens) bypass cross-encoder reranking to prevent sequence truncation distortion.

2. **CPU von System One Gater** (`wfzyx/von` on Port 8083 / Gateway 8080):
   - Sub-15ms discrete choice, boolean (noul), and score evaluation in a single forward pass.
   - Use for quick boolean checks (PASS/FAIL, ALLOW/BLOCK), safety tripwires, and blast-radius checks.

3. **CPU Decider-4B v2.1 Engine** (`Mapika/decider-4b-v2.1` on Port 8084 / Gateway 8080):
   - Schema-first multi-slot parallel extraction, DOM accessibility probes (Mind2Web / MiniWoB++), and softened MCTS rollout policy priors.
   - Evaluates complex state distributions in 1 forward pass on CPU with 0 MB GPU VRAM.

## How to Invoke

From your shell or tool execution:

### 1. Rerank Code or Text Snippets
Candidate snippets should be focused passages, function hunks, docstrings, or search hits (<800–1,000 tokens).
> **Note on Whole Files**: Whole source files (>3,000 tokens) automatically bypass CPU cross-encoder reranking to avoid truncation artifacts and wasted compute. Pass extracted passages rather than monolithic source files.

Pass candidate snippets as JSON via stdin or argument:
```bash
echo '["snippet 1: auth handling", "snippet 2: css styling", "snippet 3: jwt token auth"]' | python -m local_ai_launcher.bridges.cpu_tools_cli rerank --query "jwt token verification" --top-n 2
```

Or pass a JSON file containing a list of strings:
```bash
python -m local_ai_launcher.bridges.cpu_tools_cli rerank --query "database connection" --file candidates.json --top-n 3
```

### 2. Fast Discrete Decision / Gate Check
Evaluate a prompt against candidate options in ~14ms:
```bash
python -m local_ai_launcher.bridges.cpu_tools_cli decide --prompt "Assertion failure: expected True got False: FAIL" --options PASS FAIL
```

Or check tool permission:
```bash
python -m local_ai_launcher.bridges.cpu_tools_cli decide --prompt "Agent requested deleting production database: BLOCK" --options ALLOW BLOCK
```

### 3. Safety & Command Blast-Radius Tripwire (von)
Check prompt injection or destructive command risk in sub-10ms:
```bash
python -m local_ai_launcher.bridges.cpu_tools_cli tripwire --prompt "git reset --hard HEAD~1"
```

### 4. Browser DOM Action Selection (decider-4b v2.1)
Resolve optimal browser element and action type via Mind2Web probe with >90% token pruning:
```bash
python -m local_ai_launcher.bridges.cpu_tools_cli dom --goal "Click complete purchase" --html "<button id='btn_checkout'>Complete Purchase</button>"
```

### 5. Multi-Slot State Extraction (decider-4b v2.1)
Extract multiple discrete fields in a single forward pass:
```bash
python -m local_ai_launcher.bridges.cpu_tools_cli slots --context "User wants to deploy hotfix to prod" --slots '{"intent": ["DEPLOY", "ROLLBACK"], "risk": ["LOW", "HIGH"]}'
```

## Output Format

Reranker returns ranked items with relevance scores:
```json
{
  "results": [
    { "index": 2, "relevance_score": 0.94, "document": "snippet 3: jwt token auth" },
    { "index": 0, "relevance_score": 0.78, "document": "snippet 1: auth handling" }
  ],
  "port": 8082
}
```

Decision gater returns winner choice and calibrated confidence:
```json
{
  "choice": "FAIL",
  "confidence": 0.96,
  "distribution": { "PASS": 0.04, "FAIL": 0.96 },
  "latency_ms": 14.2,
  "port": 8083
}
```

Tripwire returns safety detection and blast radius severity (1–5):
```json
{
  "tripwire_tripped": true,
  "is_malicious": false,
  "blast_radius_severity": 5,
  "latency_ms": 0.2
}
```

DOM action selector returns planned action and element ID:
```json
{
  "action": "CLICK",
  "selected_element_id": "btn_checkout",
  "confidence": 0.90,
  "action_summary": "Plan: Action 'CLICK' on element 'btn_checkout' (Confidence: 0.90)",
  "tokens_saved_gpu": 240
}
```

Multi-slot extractor returns resolved distributions:
```json
{
  "slots": {
    "intent": { "choice": "DEPLOY", "confidence": 0.82 },
    "risk": { "choice": "HIGH", "confidence": 0.75 }
  },
  "latency_ms": 0.3
}
```
