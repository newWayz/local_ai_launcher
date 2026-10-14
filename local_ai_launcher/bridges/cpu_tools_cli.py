"""
CPU Auxiliary Tools CLI Bridge & Provisioner.

Provides CLI access to the local CPU Reranker (bge-reranker-v2-m3) and
CPU von System One Gater (wfzyx/von; legacy aliases: open-jev, clm-8b) for agent harnesses
(Hermes, Pi, Orca, and subagents).
"""

import sys
import os
import json
import logging
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional

from local_ai_launcher.config import config
from local_ai_launcher.paths import AI_WORKSPACE_DIR, atomic_write_text

logger = logging.getLogger("cpu_tools_cli")

CPU_TOOLS_SKILL_TEMPLATE = """---
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

__CALLER_LINE__

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
"""


def rerank(query: str, documents: List[str], top_n: Optional[int] = 5) -> Dict[str, Any]:
    """Reranks documents against query, trying direct port 8082, gateway 8080, then in-process fallback."""
    if not documents:
        return {"results": [], "port": "none"}

    max_doc_tokens = getattr(config, "cpu_reranker_max_doc_tokens", 3000)
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(int(len(text.split()) * 1.5), int(len(text) / 3.4)) + 1

    if any(_estimate_tokens(d) > max_doc_tokens for d in documents):
        logger.info(f"Candidate documents contain whole source files (>{max_doc_tokens} tokens). Bypassing CPU reranker.")
        return {
            "model": getattr(config, "cpu_reranker_model", "BAAI/bge-reranker-v2-m3"),
            "results": [
                {
                    "index": idx,
                    "relevance_score": round(max(0.01, 1.0 - (idx * 0.05)), 4),
                    "document": doc,
                    "bypassed": True,
                    "reason": "exceeds_max_rerank_tokens"
                }
                for idx, doc in enumerate(documents[:top_n if top_n else len(documents)])
            ],
            "port": "bypassed",
            "warning": f"Candidate documents exceed {max_doc_tokens} tokens (whole files). Reranking was bypassed."
        }

    ports = [
        getattr(config, "cpu_reranker_port", 8082),
        getattr(config, "localai_port", 8080)
    ]
    payload = json.dumps({
        "query": query,
        "documents": documents,
        "top_n": top_n
    }).encode("utf-8")

    for p in ports:
        url = f"http://127.0.0.1:{p}/v1/rerank"
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    data["port"] = p
                    return data
        except Exception:
            continue

    # In-process CPU fallback
    try:
        from local_ai_launcher.backends.cpu_reranker import rerank as local_rerank
        model_name = getattr(config, "cpu_reranker_model", "BAAI/bge-reranker-v2-m3")
        results = local_rerank(query=query, documents=documents, top_n=top_n, model_name=model_name)
        return {
            "model": model_name,
            "results": results,
            "port": "in-process"
        }
    except Exception as e:
        return {
            "error": f"Failed to execute rerank across ports {ports} and in-process: {e}",
            "results": []
        }


def decide(prompt: str, candidate_options: List[str], temperature: float = 1.45) -> Dict[str, Any]:
    """Evaluates discrete choice, trying direct port 8083, gateway 8080, then in-process fallback."""
    ports = [
        getattr(config, "cpu_onnx_gater_port", 8083),
        getattr(config, "localai_port", 8080)
    ]
    payload = json.dumps({
        "prompt": prompt,
        "candidate_options": candidate_options,
        "temperature": temperature
    }).encode("utf-8")

    for p in ports:
        url = f"http://127.0.0.1:{p}/v1/decide"
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    data["port"] = p
                    return data
        except Exception:
            continue

    # In-process CPU fallback
    try:
        from local_ai_launcher.backends.cpu_onnx_gater import get_or_load_onnx_gater
        gater = get_or_load_onnx_gater()
        result = gater.decide(prompt=prompt, candidate_options=candidate_options, temperature=temperature)
        result["port"] = "in-process"
        return result
    except Exception as e:
        return {
            "error": f"Failed to execute decision gater across ports {ports} and in-process: {e}",
            "choice": candidate_options[0] if candidate_options else "UNKNOWN",
            "confidence": 0.0
        }


def shortlist(state: str, tools: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
    """Shortlists tools against state using von/onnx gater on Port 8083 or Gateway 8080."""
    from local_ai_launcher.backends.cpu_onnx_gater import shortlist_tools_sync
    return shortlist_tools_sync(state=state, tools=tools, top_k=top_k)


def ensure_cpu_tools_skills(target_dirs: Optional[List[Any]] = None) -> List[Path]:

    """
    Ensure cpu-tools/SKILL.md is reliably provisioned for agent harnesses
    (.pi, .hermes, .agents) across workspaces.
    """
    if target_dirs is None:
        target_dirs = [
            Path(AI_WORKSPACE_DIR) / ".agents" / "skills" / "cpu-tools",
            Path(AI_WORKSPACE_DIR) / "local_ai_launcher" / "skills" / "cpu-tools",
            Path(AI_WORKSPACE_DIR) / ".hermes" / "skills" / "cpu-tools",
            Path.home() / ".hermes" / "skills" / "cpu-tools",
        ]

    written_paths: List[Path] = []
    for item in target_dirs:
        try:
            p = Path(item)
            if p.name.lower() == "skill.md":
                skill_file = p
                skill_dir = p.parent
            else:
                skill_dir = p
                skill_file = skill_dir / "SKILL.md"

            path_str = str(skill_dir).replace("\\", "/").lower()
            if "/.pi" in path_str:
                caller_line = "From Pi shell / tool execution:"
            elif "/.hermes" in path_str:
                caller_line = "From Hermes shell / tool execution:"
            else:
                caller_line = "From your shell or tool execution:"

            content = CPU_TOOLS_SKILL_TEMPLATE.replace("__CALLER_LINE__", caller_line)

            if skill_file.exists():
                try:
                    if skill_file.read_text(encoding="utf-8") == content:
                        written_paths.append(skill_file)
                        continue
                except Exception:
                    pass

            atomic_write_text(skill_file, content)
            written_paths.append(skill_file)
            logger.info(f"[cpu-tools] Provisioned skill to {skill_file}")
        except Exception as e:
            logger.warning(f"[cpu-tools] Failed to provision skill in {item}: {e}")

    return written_paths


def main():
    parser = argparse.ArgumentParser(description="CPU Auxiliary Tools CLI Bridge")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to run")

    # 1. rerank subcommand
    p_rerank = subparsers.add_parser("rerank", help="Semantic cross-encoder passage ranking")
    p_rerank.add_argument("--query", required=True, help="Search query or semantic target")
    p_rerank.add_argument("--top-n", type=int, default=5, help="Top N results to return")
    p_rerank.add_argument("--docs", nargs="*", help="List of candidate document strings")
    p_rerank.add_argument("--file", help="Path to JSON file containing list of string documents")

    # 2. decide subcommand
    p_decide = subparsers.add_parser("decide", help="Fast discrete choice evaluation")
    p_decide.add_argument("--prompt", required=True, help="Prompt or context to evaluate")
    p_decide.add_argument("--options", nargs="+", default=["PASS", "FAIL"], help="Candidate options (e.g. PASS FAIL)")
    p_decide.add_argument("--temperature", type=float, default=1.45, help="Softmax sharpness temperature")

    # 3. install-skills subcommand
    p_install = subparsers.add_parser("install-skills", help="Provision cpu-tools SKILL.md to harnesses")

    # 4. shortlist subcommand
    p_shortlist = subparsers.add_parser("shortlist", help="Zero-shot CLM tool shortlisting")
    p_shortlist.add_argument("--state", required=True, help="Conversation state or intent")
    p_shortlist.add_argument("--tools", nargs="*", help="List of tool names")
    p_shortlist.add_argument("--file", help="Path to JSON file containing list of tool schemas")
    p_shortlist.add_argument("--top-k", type=int, default=3, help="Top K tools to shortlist")

    # 5. register subcommand
    p_register = subparsers.add_parser("register", help="Pre-cache candidate tools into DDR5 action matrix")
    p_register.add_argument("--tools", nargs="*", help="List of tool names")
    p_register.add_argument("--file", help="Path to JSON file containing list of tool schemas")

    # 6. info subcommand
    p_info = subparsers.add_parser("info", help="Get engine status, vector dimension, and cache metadata")

    # 7. tripwire subcommand
    p_tripwire = subparsers.add_parser("tripwire", help="Fast safety and command blast-radius check")
    p_tripwire.add_argument("--prompt", required=True, help="Prompt or command to check")

    # 8. dom subcommand (Mind2Web / MiniWoB++ decider-4b probe)
    p_dom = subparsers.add_parser("dom", help="Browser DOM action and element resolution via decider-4b")
    p_dom.add_argument("--goal", required=True, help="User goal or instruction")
    p_dom.add_argument("--html", help="HTML or DOM string")
    p_dom.add_argument("--file", help="Path to file containing HTML or DOM text")

    # 9. slots subcommand (Multi-slot parallel extraction via decider-4b)
    p_slots = subparsers.add_parser("slots", help="Multi-slot parallel state extraction via decider-4b")
    p_slots.add_argument("--context", required=True, help="Context text")
    p_slots.add_argument("--slots", required=True, help="JSON string or file defining slots")

    args = parser.parse_args()

    if args.subcommand == "rerank":
        docs: List[str] = []
        if args.docs:
            docs = args.docs
        elif args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                content = json.load(f)
                docs = content if isinstance(content, list) else [str(content)]
        elif not sys.stdin.isatty():
            raw_in = sys.stdin.read().strip()
            if raw_in:
                try:
                    content = json.loads(raw_in)
                    docs = content if isinstance(content, list) else [str(content)]
                except Exception:
                    docs = [line.strip() for line in raw_in.splitlines() if line.strip()]

        if not docs:
            print(json.dumps({"error": "No documents provided via --docs, --file, or stdin.", "results": []}))
            sys.exit(1)

        result = rerank(query=args.query, documents=docs, top_n=args.top_n)
        print(json.dumps(result, indent=2))

    elif args.subcommand == "decide":
        result = decide(prompt=args.prompt, candidate_options=args.options, temperature=args.temperature)
        print(json.dumps(result, indent=2))

    elif args.subcommand == "shortlist":
        tool_list: List[Dict[str, Any]] = []
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                raw_tools = json.load(f)
                tool_list = raw_tools if isinstance(raw_tools, list) else [raw_tools]
        elif args.tools:
            tool_list = [{"name": t, "description": t} for t in args.tools]
        elif not sys.stdin.isatty():
            raw_in = sys.stdin.read().strip()
            if raw_in:
                try:
                    parsed = json.loads(raw_in)
                    tool_list = parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    tool_list = [{"name": line.strip(), "description": line.strip()} for line in raw_in.splitlines() if line.strip()]

        result = shortlist(state=args.state, tools=tool_list, top_k=args.top_k)
        print(json.dumps(result, indent=2))

    elif args.subcommand == "register":
        tool_list: List[Dict[str, Any]] = []
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                raw_tools = json.load(f)
                tool_list = raw_tools if isinstance(raw_tools, list) else [raw_tools]
        elif args.tools:
            tool_list = [{"name": t, "description": t} for t in args.tools]
        elif not sys.stdin.isatty():
            raw_in = sys.stdin.read().strip()
            if raw_in:
                try:
                    parsed = json.loads(raw_in)
                    tool_list = parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    tool_list = [{"name": line.strip(), "description": line.strip()} for line in raw_in.splitlines() if line.strip()]

        from local_ai_launcher.backends.cpu_onnx_gater import register_tools_sync
        result = register_tools_sync(tools=tool_list)
        print(json.dumps(result, indent=2))

    elif args.subcommand == "info":
        from local_ai_launcher.backends.cpu_onnx_gater import get_clm_info_sync
        result = get_clm_info_sync()
        print(json.dumps(result, indent=2))

    elif args.subcommand == "tripwire":
        from local_ai_launcher.backends.cpu_onnx_gater import get_or_load_onnx_gater
        gater = get_or_load_onnx_gater()
        result = gater.tripwire(prompt=args.prompt)
        print(json.dumps(result, indent=2))

    elif args.subcommand == "dom":
        html_content = ""
        if args.html:
            html_content = args.html
        elif args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                html_content = f.read()
        elif not sys.stdin.isatty():
            html_content = sys.stdin.read()

        from local_ai_launcher.bridges.dom_navigator import evaluate_browser_dom_action_sync
        res = evaluate_browser_dom_action_sync(goal=args.goal, raw_html_or_dom=html_content)
        print(json.dumps(res, indent=2))

    elif args.subcommand == "slots":
        raw_slots = args.slots
        slot_dict = {}
        if os.path.exists(raw_slots):
            with open(raw_slots, "r", encoding="utf-8") as f:
                slot_dict = json.load(f)
        else:
            try:
                slot_dict = json.loads(raw_slots)
            except Exception:
                slot_dict = {s.strip(): ["YES", "NO"] for s in raw_slots.split(",") if s.strip()}

        from local_ai_launcher.backends.cpu_decider import evaluate_decider_slots_sync
        res = evaluate_decider_slots_sync(context=args.context, slots=slot_dict)
        print(json.dumps(res, indent=2))

    elif args.subcommand == "install-skills":
        paths = ensure_cpu_tools_skills()
        print(json.dumps({"status": "ok", "provisioned": [str(p) for p in paths]}, indent=2))

    else:
        parser.print_help()



if __name__ == "__main__":
    main()
