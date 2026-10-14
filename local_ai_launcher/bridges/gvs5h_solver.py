"""
GVS5H Zero-Shot Self-Orchestrator Bridge (deep-solve engine).

Based on the ledger-based self-orchestration architecture from:
  "Zero-Shot Self-Orchestration with Ledger-Based Control Improves Coding in Language Models"
  (Victor Gao, Vida Khosrowshahi, Ali Khosrowshahi, Xihao Sun, Juhyun Lee, Simon Lee / Persis Capital, ICLR 2027)
  https://github.com/slee-persis/GVS5H

This module executes a 5-phase sequential multi-agent loop using a single underlying model
(default: qwen-38_27b-blackwell-256k-think) via LocalAI / llama-server. Every role runs in a fresh,
bounded context communicating through a file-based ledger on disk.
"""

import os
import re
import sys
import time
import json
import uuid
import shutil
import hashlib
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from local_ai_launcher.paths import AI_WORKSPACE_DIR
from local_ai_launcher.process import assign_process_to_job

logger = logging.getLogger("gvs5h_solver")

# --- Constants & Limits (matched to GVS5H paper & Blackwell RTX 5080 constraints) ---
DEFAULT_MODEL = os.environ.get("GVS5H_MODEL", "qwen-38_27b-blackwell-256k-think")
DEFAULT_ENDPOINT = os.environ.get("GVS5H_ENDPOINT", "http://localhost:8080/v1/chat/completions")
FALLBACK_ENDPOINT = "http://127.0.0.1:9931/v1/chat/completions"

MAX_PLAN_CHARS = 4000
MAX_NOTES_CHARS = 8000
MAX_ANSWER_CHARS = 20000
MAX_TASKS = 12
DEFAULT_MAX_ITERS = 6
DEFAULT_MAX_SECONDS = 180
DEFAULT_TEST_TIMEOUT = 10.0
DEFAULT_MAX_LOG_RUNS = 100

WS_ROOT = Path(AI_WORKSPACE_DIR) / "tmp" / "gvs5h"
LOGS_ROOT = Path(AI_WORKSPACE_DIR) / "logs" / "gvs5h"

# --- Skill Provisioning & Auto-Discovery ---

DEEP_SOLVE_SKILL_TEMPLATE = (
    "---\n"
    "name: deep-solve\n"
    "description: Zero-shot ledger-based self-orchestration (GVS5H) for solving complex algorithms, mathematically dense modules, hard debugging, or logic synthesis using local Qwen 3.8 27B without context bloat or runaway reasoning.\n"
    "---\n\n"
    "# Deep Solve (GVS5H Self-Orchestrator)\n\n"
    "The `deep-solve` skill invokes the **GVS5H** (*Gorilla vs. 5 Humans*) ledger-based self-orchestration engine. It executes a multi-turn, 5-stage sequential optimization loop (Planning $\\to$ Ideation $\\to$ Curation $\\to$ Execution $\\to$ Test Verification) using the locally served `qwen-38_27b-blackwell-256k-think` model via LocalAI.\n\n"
    "Use this skill when:\n"
    "* Implementing complex algorithms (dynamic programming, graphs, automata, parsers, state machines).\n"
    "* A single-turn generation repeatedly fails subtle edge cases or gets stuck in repetitive reasoning.\n"
    "* Refactoring or optimizing an isolated module with verifiable assertions or unit tests.\n\n"
    "## Why use `deep-solve` instead of standard single-turn coding?\n"
    "* **Zero Context Bloat**: Every worker invocation runs in a fresh, bounded context ($\\le 8\\text{k}$ tokens), operating at the highest precision band of the local Q4 KV cache.\n"
    "* **Separation of Ideation & Execution**: The ideator explores algorithmic approaches and bounds in pure prose before any code is generated, preventing runaway edge-case interrogations.\n"
    "* **Subprocess Test Gate**: Code is compiled and tested against a real test suite in an isolated sandbox. The manager is programmatically forbidden from signing off on a failing test suite.\n"
    "* **Regression Rollback**: Tracks the best-performing solution across rounds, automatically rolling back if later iterations introduce bugs.\n\n"
    "## How to Invoke\n\n"
    "__CALLER_LINE__\n"
    "```bash\n"
    "python -m local_ai_launcher.bridges.gvs5h_solver \\\n"
    "  --task \"<Clear, detailed task description and algorithmic requirements>\" \\\n"
    "  --target \"<Path to output file, e.g. src/my_app/solver.py>\" \\\n"
    "  --test-cmd \"<Command to verify code, e.g. pytest tests/test_solver.py>\"\n"
    "```\n\n"
    "### Optional Arguments:\n"
    "* `--max-iters <N>`: Maximum manager-worker iteration cycles (default: 6).\n"
    "* `--max-seconds <S>`: Wall-clock watchdog limit before triggering clean finalization (default: 180s).\n"
    "* `--model <model_id>`: Target model (default: `qwen-38_27b-blackwell-256k-think`).\n"
    "* `--endpoint <url>`: API endpoint (default: `http://localhost:8080/v1/chat/completions`).\n\n"
    "## Output\n"
    "The command returns a JSON object summarizing the run:\n"
    "```json\n"
    "{\n"
    "  \"status\": \"success\",\n"
    "  \"delivered\": true,\n"
    "  \"target_file\": \"src/my_app/solver.py\",\n"
    "  \"iterations\": 3,\n"
    "  \"duration_seconds\": 78.4,\n"
    "  \"tests_passed\": \"5/5\",\n"
    "  \"logs_dir\": \"C:\\\\aiWorkspace\\\\logs\\\\gvs5h\\\\...\"\n"
    "}\n"
    "```\n"
    "Existing target files are backed up to `<target_file>.bak` before modification. Full conversation transcripts are archived to `logs/gvs5h/`.\n"
)


def ensure_deep_solve_skills(target_dirs: Optional[List[Any]] = None) -> List[Path]:
    """
    Ensure deep-solve/SKILL.md is reliably provisioned for agent harnesses
    (.pi, .hermes, .agents) even after reinstallations or directory wipes.
    """
    from local_ai_launcher.paths import atomic_write_text, AI_WORKSPACE_DIR

    if target_dirs is None:
        target_dirs = [
            Path(AI_WORKSPACE_DIR) / ".pi" / "skills" / "deep-solve",
            Path(AI_WORKSPACE_DIR) / ".hermes" / "skills" / "deep-solve",
            Path(AI_WORKSPACE_DIR) / ".agents" / "skills" / "deep-solve",
            Path.home() / ".hermes" / "skills" / "deep-solve",
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

            content = DEEP_SOLVE_SKILL_TEMPLATE.replace("__CALLER_LINE__", caller_line)

            if skill_file.exists():
                try:
                    if skill_file.read_text(encoding="utf-8") == content:
                        written_paths.append(skill_file)
                        continue
                except Exception:
                    pass

            atomic_write_text(skill_file, content)
            written_paths.append(skill_file)
            logger.info(f"[deep-solve] Provisioned skill to {skill_file}")
        except Exception as e:
            logger.warning(f"[deep-solve] Failed to provision skill in {item}: {e}")

    return written_paths


# --- Parsing & Formatting Helpers ---

def strip_think(text: str) -> str:
    """Strip DeepSeek/Qwen style <think>...</think> chain-of-thought blocks."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_code(text: str) -> str:
    """Strip markdown code fences to enforce pure prose during ideation."""
    if not text:
        return ""
    return re.sub(r"```.*?```", "[code omitted -- algorithmic approach only]", text, flags=re.DOTALL).strip()


def parse_sections(text: str) -> Dict[str, str]:
    """Parse output divided by ### SECTION headers."""
    clean = strip_think(text)
    out: Dict[str, str] = {}
    cur: Optional[str] = None
    buf: List[str] = []
    
    for line in clean.splitlines():
        s = line.strip()
        m = (re.match(r"^#{1,3}\s*([A-Za-z0-9_]+)\s*$", s)
             or re.match(r"^\*\*([A-Za-z0-9_]+)\*\*\s*:?\s*$", s)
             or re.match(r"^([A-Z_]{3,})\s*:\s*$", s))
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur = m.group(1).upper()
            buf = []
        else:
            buf.append(line)
            
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def parse_bullets(text: str) -> List[str]:
    """Extract bulleted or numbered items."""
    if not text:
        return []
    items = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)", line)
        if m:
            items.append(m.group(1).strip())
    return items


def extract_python_code(text: str) -> str:
    """Extract code from Python markdown blocks, with graceful fallbacks."""
    clean = strip_think(text)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", clean, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback to any code block
    m2 = re.search(r"```\s*\n(.*?)```", clean, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return ""


def parse_tasks(text: str) -> List[Dict[str, Any]]:
    """Parse task lists with [todo] or [done] status indicators."""
    out = []
    for b in parse_bullets(text):
        m = re.match(r"\[\s*([a-z_ ]+?)\s*\]\s*(.*)", b, re.IGNORECASE)
        if m:
            raw = m.group(1).lower().replace(" ", "").replace("_", "")
            desc = m.group(2).strip()
            status = "done" if raw == "done" else ("in_progress" if raw in ("wip", "inprogress") else "pending")
        else:
            desc, status = b, "pending"
        if desc:
            out.append({"id": len(out) + 1, "desc": desc, "status": status, "result": ""})
    return out[:MAX_TASKS]


# --- HTTP Client & Endpoint Dispatcher ---

def dispatch_chat_completion(
    messages: List[Dict[str, str]],
    temperature: float,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    max_tokens: int = 4096,
    meta: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0
) -> str:
    """
    Dispatches an HTTP chat completion request to LocalAI / llama-server.
    Reuses the existing resident model in VRAM and dynamically updates temperature.
    """
    import urllib.request
    # Harden against invalid / negative numbers
    temperature = max(0.0, min(2.0, float(temperature)))
    max_tokens = max(1, int(max_tokens))
    timeout = max(1.0, float(timeout))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "GVS5H-Solver/1.0"}

    # Attempt primary endpoint, fallback if connection refused
    target_urls = [endpoint]
    if endpoint != FALLBACK_ENDPOINT:
        target_urls.append(FALLBACK_ENDPOINT)

    last_err = None
    for url in target_urls:
        req = urllib.request.Request(url, data=data, headers=headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    choice = res_json.get("choices", [{}])[0]
                    msg = choice.get("message", {})
                    content = msg.get("content") or ""
                    
                    if meta is not None:
                        meta["finish_reason"] = choice.get("finish_reason")
                        usage = res_json.get("usage", {})
                        meta["completion_tokens"] = usage.get("completion_tokens", 0)
                        meta["prompt_tokens"] = usage.get("prompt_tokens", 0)
                        meta["model"] = res_json.get("model", model)
                    return content
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")[:300]
                except Exception:
                    pass
                last_err = f"HTTP {e.code}: {err_body}"
                logger.warning(f"[gvs5h] Attempt {attempt+1} on {url} failed: {last_err}")
                time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                last_err = str(e)
                logger.warning(f"[gvs5h] Attempt {attempt+1} error on {url}: {e}")
                time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"GVS5H could not communicate with LLM endpoints {target_urls}: {last_err}")


# --- Workspace Hygiene & Lifecycle Management ---

def cleanup_stale_sandboxes(max_age_hours: float = 24.0) -> None:
    """Purge temporary sandboxes older than max_age_hours."""
    if not WS_ROOT.exists():
        return
    max_age_hours = max(0.1, float(max_age_hours))
    now = time.time()
    for child in WS_ROOT.iterdir():
        if child.is_dir():
            try:
                if now - child.stat().st_mtime > max_age_hours * 3600:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass


def get_default_max_log_runs() -> int:
    """Resolve default log folder retention limit from env or LauncherConfig."""
    env_val = os.environ.get("GVS5H_MAX_LOG_RUNS")
    if env_val:
        try:
            return max(1, int(env_val))
        except (ValueError, TypeError):
            pass
    try:
        from local_ai_launcher.config import LauncherConfig
        cfg = LauncherConfig()
        return max(1, int(getattr(cfg, "gvs5h_max_log_runs", DEFAULT_MAX_LOG_RUNS)))
    except Exception:
        return DEFAULT_MAX_LOG_RUNS


def prune_old_gvs5h_logs(
    max_runs: Optional[int] = None,
    logs_root: Optional[Path] = None
) -> List[Path]:
    """
    Ensure C:\aiWorkspace\logs\gvs5h retains at most `max_runs` recent directories.
    Deletes the oldest folders by modification time (or timestamp prefix)
    when the total number exceeds `max_runs`.
    Returns the list of deleted folder Paths.
    """
    root = Path(logs_root) if logs_root is not None else LOGS_ROOT
    if not root.exists():
        return []

    if max_runs is None:
        target_limit = get_default_max_log_runs()
    else:
        try:
            target_limit = max(1, int(max_runs))
        except (ValueError, TypeError):
            target_limit = get_default_max_log_runs()

    subdirs: List[Tuple[float, Path]] = []
    for child in root.iterdir():
        if child.is_dir():
            try:
                mtime = child.stat().st_mtime
                name_ts = 0.0
                parts = child.name.split("_", 1)
                if parts[0].isdigit():
                    name_ts = float(parts[0])
                sort_key = max(mtime, name_ts)
                subdirs.append((sort_key, child))
            except Exception:
                subdirs.append((0.0, child))

    # Oldest first, newest last
    subdirs.sort(key=lambda item: item[0])

    deleted: List[Path] = []
    excess_count = len(subdirs) - target_limit
    if excess_count > 0:
        for _, old_dir in subdirs[:excess_count]:
            try:
                shutil.rmtree(old_dir, ignore_errors=True)
                deleted.append(old_dir)
                logger.info(f"[GVS5H] Pruned old log folder: {old_dir.name}")
            except Exception as e:
                logger.debug(f"[GVS5H] Failed to prune log folder {old_dir}: {e}")

    return deleted


# --- Core GVS5H Engine Class ---

class GVS5HSolver:
    """
    Manages the GVS5H ledger workspace, multi-role prompts, test execution,
    best-so-far regression rollback, and atomic target file updates.
    """

    def __init__(
        self,
        task: str,
        target_file: str,
        test_cmd: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        max_iters: int = DEFAULT_MAX_ITERS,
        max_seconds: int = DEFAULT_MAX_SECONDS,
        test_timeout: float = DEFAULT_TEST_TIMEOUT,
        verbose: bool = True,
        max_log_runs: Optional[int] = None
    ):
        self.task = task.strip()
        self.target_path = Path(target_file).resolve()
        self.test_cmd = test_cmd.strip() if test_cmd else None
        self.model = model
        self.endpoint = endpoint
        try:
            self.max_iters = max(1, int(max_iters))
        except (ValueError, TypeError):
            self.max_iters = DEFAULT_MAX_ITERS

        try:
            self.max_seconds = max(1, int(max_seconds))
        except (ValueError, TypeError):
            self.max_seconds = DEFAULT_MAX_SECONDS

        try:
            self.test_timeout = max(0.1, float(test_timeout))
        except (ValueError, TypeError):
            self.test_timeout = DEFAULT_TEST_TIMEOUT

        if max_log_runs is not None:
            try:
                self.max_log_runs = max(1, int(max_log_runs))
            except (ValueError, TypeError):
                self.max_log_runs = get_default_max_log_runs()
        else:
            self.max_log_runs = get_default_max_log_runs()

        self.verbose = verbose

        self.job_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.slug = hashlib.md5(self.task.encode()).hexdigest()[:8]
        self.ws = WS_ROOT / f"{self.job_id}_{self.slug}"
        self.logs_dir = LOGS_ROOT / f"{self.job_id}_{self.slug}"

        self.start_time = 0.0
        self.best_passed_count = -1
        self.best_solution_code = ""
        self.consecutive_fails = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
        logger.info(msg)

    # --- File I/O within the Sandbox Ledger ---

    def _read(self, filename: str) -> str:
        p = self.ws / filename
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def _write(self, filename: str, content: str) -> None:
        p = self.ws / filename
        p.write_text(content, encoding="utf-8")

    def _append(self, filename: str, content: str) -> None:
        p = self.ws / filename
        with p.open("a", encoding="utf-8") as f:
            f.write(content)

    def _record(self, entry: Dict[str, Any]) -> None:
        with (self.ws / "transcript.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _chat(self, role: str, messages: List[Dict[str, str]], temperature: float) -> str:
        meta: Dict[str, Any] = {}
        t0 = time.time()
        reply = dispatch_chat_completion(
            messages=messages,
            temperature=temperature,
            model=self.model,
            endpoint=self.endpoint,
            meta=meta
        )
        self._record({
            "t": time.time(),
            "duration": round(time.time() - t0, 3),
            "role": role,
            "request": messages,
            "response": reply,
            "temperature": temperature,
            "finish_reason": meta.get("finish_reason"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens")
        })
        return reply

    # --- Stage 1: Primary Manager Plan (temp=0.3) ---

    def _primary_plan(self) -> List[Dict[str, Any]]:
        sys = (
            "You are the PRIMARY orchestrator and software architect. "
            "Given a programming task or bug, produce a short overarching plan to solve it, "
            "then a concrete task list for specialized workers.\n"
            "Respond with EXACTLY these sections:\n"
            "### PLAN\n<3-6 sentence overarching strategy>\n"
            "### TASKS\n<3-5 bullet tasks, each a concrete, independent unit of work>"
        )
        reply = self._chat("primary_plan", [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"TASK:\n{self.task}"}
        ], temperature=0.3)

        sec = parse_sections(reply)
        plan_text = (sec.get("PLAN") or reply.strip())[:MAX_PLAN_CHARS]
        self._write("plan.md", plan_text)
        
        task_bullets = parse_bullets(sec.get("TASKS", ""))
        tasks = [{"id": i+1, "desc": b, "status": "pending", "result": ""} for i, b in enumerate(task_bullets[:MAX_TASKS])]
        self._write("tasks.json", json.dumps(tasks, indent=2))
        self.log(f"    [primary] plan created with {len(tasks)} initial tasks")
        return tasks

    # --- Stage 2: Ideation Worker (temp=0.4, strictly no code) ---

    def _ideation_worker(self) -> List[str]:
        sys = (
            "You are the FIRST WORKER. Do NOT solve the problem and do NOT write ANY code. "
            "Analyze the problem: identify core difficulties, list 2-3 DISTINCT candidate approaches "
            "(different algorithms, data structures, or reductions), and identify potential pitfalls. "
            "Describe each approach in PROSE ONLY -- absolutely no code blocks; a later worker writes code.\n"
            "Respond with EXACTLY these sections:\n"
            "### NOTES\n<your conceptual analysis and trade-offs>\n"
            "### NEXT\n<bullet list of distinct approaches to try next>"
        )
        reply = self._chat("ideation", [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"PROBLEM:\n{self.task}\n\nPLAN:\n{self._read('plan.md')}"}
        ], temperature=0.4)

        sec = parse_sections(reply)
        notes = strip_code(sec.get("NOTES") or reply.strip())[:MAX_NOTES_CHARS]
        self._write("notes.md", f"## Ideation & Candidate Approaches\n{notes}\n")
        proposals = [strip_code(p) for p in parse_bullets(sec.get("NEXT", ""))]
        self.log(f"    [worker:ideate] brainstormed {len(proposals)} candidate approaches (prose only)")
        return proposals

    # --- Stage 3: Primary Manager Curate & Assign (temp=0.2) ---

    def _primary_manage(
        self,
        tasks: List[Dict[str, Any]],
        proposals: List[str],
        latest_feedback: str
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        current_code = self._read("solution.py").strip()
        task_lines = "\n".join(f"- [{'done' if t['status'] == 'done' else 'todo'}] {t['desc']}" for t in tasks) or "(none)"
        prop_lines = "\n".join(f"- {p}" for p in proposals) or "(none)"

        sys = (
            "You are the PRIMARY orchestrator and manager. You own the task list and decide when the problem is solved.\n"
            "- If the latest test verdict shows all tests PASSED and the code is complete, set STATUS to 'done'.\n"
            "- If tests FAILED, you MUST set STATUS to 'continue' and assign a task to fix the failing case or switch approach.\n"
            "- CURATE the task list: fold in new proposals, mark completed tasks, and choose ONE specific next task.\n"
            "Respond with EXACTLY:\n"
            "### STATUS\n<done|continue>\n"
            "### NEXT\n<exact text of the ONE task to execute next; omit if done>\n"
            "### TASKS\n<curated list, each '- [done] ...' or '- [todo] ...'>"
        )
        reply = self._chat("primary_manage", [
            {"role": "system", "content": sys},
            {"role": "user", "content": (
                f"PROBLEM:\n{self.task}\n\nCURRENT SOLUTION:\n{current_code or '(none yet)'}\n\n"
                f"NOTES:\n{self._read('notes.md')}\n\nTASK LIST:\n{task_lines}\n\n"
                f"LATEST TEST VERDICT & WORKER RESULT:\n{latest_feedback}\n\n"
                f"PROPOSED STEPS:\n{prop_lines}"
            )}
        ], temperature=0.2)

        sec = parse_sections(reply)
        curated = parse_tasks(sec.get("TASKS", "")) or tasks
        status = "done" if sec.get("STATUS", "").strip().lower().startswith("done") else "continue"
        
        # Invariant: cannot be done without code
        if status == "done" and not current_code:
            status = "continue"

        nxt_bullets = parse_bullets(sec.get("NEXT", ""))
        nxt = (nxt_bullets[0] if nxt_bullets else sec.get("NEXT", "").strip())
        nxt = re.sub(r"^[-*+]\s*", "", nxt).strip()
        
        if status == "continue" and not nxt:
            todo = [t for t in curated if t["status"] != "done"]
            nxt = todo[0]["desc"] if todo else "Implement the full solution using the most promising approach in notes."

        self.log(f"    [primary-manage] status: {status}" + (f", next: {nxt[:60]}..." if status == "continue" else ""))
        return status, nxt, curated

    # --- Stage 4: Task Worker (temp=0.2) ---

    def _worker(
        self,
        task_desc: str,
        task_id: int,
        clean_slate: bool = False,
        finalize: bool = False
    ) -> Tuple[str, List[str], str, bool]:
        cur_code = self._read("solution.py")
        plan_context = "" if clean_slate else f"PLAN:\n{self._read('plan.md')}\n\n"
        notes_context = "" if clean_slate else f"NOTES:\n{self._read('notes.md')}\n\n"

        goal = "Produce the DEFINITIVE final self-contained program now." if finalize else f"Complete this task: {task_desc}"
        if clean_slate:
            goal += " (Note: Previous attempts stalled. Provide a fresh, clean-slate implementation from first principles)."

        sys = (
            "You are a WORKER subagent expert in software implementation. "
            "Write robust, self-contained Python code. Build on notes where helpful, but feel free to write "
            "a fresh solution if the current approach is flawed.\n"
            "Respond with EXACTLY:\n"
            "### CODE\n```python\n<full updated code>\n```\n"
            "### NOTES\n<complete updated notes file: retain key facts, remove superseded ideas. Under 500 words>\n"
            "### NEXT\n<bullet list of remaining checks, or 'none'>\n"
            "### STATUS\n<solved|continue>"
        )
        meta: Dict[str, Any] = {}
        reply = self._chat(f"worker_{task_id}", [
            {"role": "system", "content": sys},
            {"role": "user", "content": (
                f"PROBLEM:\n{self.task}\n\n{plan_context}{notes_context}"
                f"CURRENT WORK:\n{cur_code or '(none yet)'}\n\nYOUR TASK: {goal}"
            )}
        ], temperature=0.2)

        sec = parse_sections(reply)
        new_code = extract_python_code(sec.get("CODE", "")) or extract_python_code(reply)
        wrote = False
        if new_code:
            self._write("solution.py", new_code)
            if self.target_path.name and self.target_path.name != "solution.py":
                self._write(self.target_path.name, new_code)
            wrote = True

        # Cut-off Summarizer if token ceiling was hit
        if meta.get("finish_reason") == "length" and not finalize:
            digest = self._summarize_cutoff(reply, task_desc)
            summary = f"Worker HIT TOKEN LIMIT before finishing: {task_desc[:60]}. Thinking summary: {digest[:300]}"
            self.log(f"    [worker] task {task_id} -> CUT OFF by token cap; summarized partial progress")
            return "continue", [], summary, wrote

        if sec.get("NOTES"):
            self._write("notes.md", sec["NOTES"].strip()[:MAX_NOTES_CHARS] + "\n")

        nexts = [b for b in parse_bullets(sec.get("NEXT", "")) if b.lower() != "none"]
        worker_status = sec.get("STATUS", "continue").strip().lower()
        summary = f"Worker reported '{worker_status}' on: {task_desc[:60]}."
        self.log(f"    [worker] task {task_id} -> {worker_status}, wrote_code={wrote}")
        return worker_status, nexts, summary, wrote

    # --- Cut-off Summarizer ---

    def _summarize_cutoff(self, partial_reply: str, task_desc: str) -> str:
        clean = strip_think(partial_reply)
        snippet = clean if len(clean) <= 6000 else clean[:2500] + "\n...[truncated]...\n" + clean[-3500:]
        sys = (
            "A worker hit the token limit mid-generation. Summarize its partial attempt in 3-4 sentences: "
            "what approach was it building, what worked, and what remained unfinished. Do NOT write code."
        )
        summary = self._chat("cutoff_summary", [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"TASK: {task_desc}\n\nPARTIAL ATTEMPT:\n{snippet}"}
        ], temperature=0.2)
        return strip_think(summary).strip()

    # --- Stage 5: Sandboxed Subprocess Verifier ---

    def _run_tests(self) -> Dict[str, Any]:
        """
        Executes candidate solution against the test suite in an isolated subprocess.
        Binds child processes to Windows Job Objects and enforces execution timeouts.
        """
        sol_file = self.ws / "solution.py"
        target_alias = self.ws / self.target_path.name
        if sol_file.exists():
            shutil.copyfile(sol_file, target_alias)
        elif target_alias.exists():
            shutil.copyfile(target_alias, sol_file)

        if not sol_file.exists() or not sol_file.read_text(encoding="utf-8").strip():
            return {"ran": False, "passed": 0, "total": 0, "feedback": "[TESTS: No code written yet]"}

        if not self.test_cmd:
            # Syntax validation if no test command provided
            try:
                compile(sol_file.read_text(encoding="utf-8"), str(sol_file), "exec")
                return {"ran": True, "passed": 1, "total": 1, "feedback": "[SYNTAX CHECK: Valid Python code]"}
            except SyntaxError as se:
                return {"ran": True, "passed": 0, "total": 1, "feedback": f"[SYNTAX ERROR: {se}]"}

        # Run test command within the sandbox cwd
        cmd = self.test_cmd
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.ws) + os.path.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        # Remove pycache in sandbox to prevent stale bytecode execution
        pycache_dir = self.ws / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir, ignore_errors=True)

        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(self.ws),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            try:
                assign_process_to_job(proc)
            except Exception:
                pass

            stdout, stderr = proc.communicate(timeout=self.test_timeout)
            ret = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = "", "Test execution timed out (>10s)"
            ret = -1
        except Exception as e:
            stdout, stderr = "", str(e)
            ret = -1

        combined = (stdout + "\n" + stderr).strip()
        
        # Parse passed count if pytest style (e.g. "5 passed")
        passed = 0
        total = 1
        m = re.search(r"(\d+)\s+passed", combined)
        if m:
            passed = int(m.group(1))
            total = passed
            m_fail = re.search(r"(\d+)\s+failed", combined)
            if m_fail:
                total += int(m_fail.group(1))

        success = (ret == 0)
        if success and passed == 0:
            passed = 1
            total = 1

        feedback = (
            f"[TESTS: PASSED all {total} checks]" if success else
            f"[TESTS: FAILED ({passed}/{total} passed). Error/Output:\n{combined[:500]}]"
        )
        return {
            "ran": True,
            "success": success,
            "passed": passed,
            "total": total,
            "feedback": feedback
        }

    # --- Execution Loop with Mitigations ---

    def solve(self) -> Dict[str, Any]:
        """Runs the complete GVS5H ledger-based optimization loop."""
        cleanup_stale_sandboxes()
        prune_old_gvs5h_logs(max_runs=self.max_log_runs, logs_root=LOGS_ROOT)
        self.start_time = time.time()
        self.ws.mkdir(parents=True, exist_ok=True)

        for fname in ("task.md", "plan.md", "notes.md", "solution.py", "tasks.json", "transcript.jsonl"):
            self._write(fname, "")
        self._write("task.md", self.task)

        # Baseline: If target file already exists, copy it into sandbox
        if self.target_path.exists():
            original_content = self.target_path.read_text(encoding="utf-8")
            self._write("solution.py", original_content)
            if self.target_path.name != "solution.py":
                self._write(self.target_path.name, original_content)
            self._write("best_solution.py", original_content)
            self.best_solution_code = original_content

        self.log(f"\n[GVS5H] Initialized ledger sandbox: {self.ws.name}")
        self.log(f"[GVS5H] Target file: {self.target_path.name} | Model: {self.model}")

        # Step 1: Plan
        tasks = self._primary_plan()

        # Step 2: Ideate
        proposals = self._ideation_worker()

        # Step 3: Manage
        status, next_desc, tasks = self._primary_manage(tasks, proposals, "Ideation complete.")

        iters = 0
        prev_desc = None

        while status == "continue" and next_desc and iters < self.max_iters:
            # Watchdog check: ensure GVS5H returns before parent harness timeout
            elapsed = time.time() - self.start_time
            if elapsed >= (self.max_seconds - 25):
                self.log(f"    [watchdog] Approaching max time limit ({int(elapsed)}s/{self.max_seconds}s); triggering finalizer.")
                break

            # No-progress guard
            if prev_desc and next_desc.strip().lower() == prev_desc.strip().lower():
                self.log(f"    [primary] Reissued identical task '{next_desc[:40]}'; stopping to prevent loop.")
                break
            prev_desc = next_desc

            iters += 1
            self.log(f"\n--- Round {iters}/{self.max_iters} ---")

            # Mitigation 5: Clean-slate worker if stuck 2 rounds in a row
            clean_slate_active = (self.consecutive_fails >= 2)
            if clean_slate_active:
                self.log("    [mitigation] 2 consecutive failures detected; triggering clean-slate worker without prior notes.")

            worker_status, nexts, summary, wrote = self._worker(
                task_desc=next_desc,
                task_id=iters,
                clean_slate=clean_slate_active
            )

            # Verification
            test_res = self._run_tests()
            test_feedback = test_res.get("feedback", "")
            if test_res.get("ran"):
                self.log(f"    [verifier] {test_feedback}")
                summary = f"{test_feedback}\n{summary}"

                # Mitigation 1: Best-so-far tracking & rollback protection
                if test_res.get("passed", 0) > self.best_passed_count:
                    self.best_passed_count = test_res["passed"]
                    cur_code = self._read("solution.py")
                    self.best_solution_code = cur_code
                    self._write("best_solution.py", cur_code)
                    self.log(f"    [checkpoint] New best score: {self.best_passed_count}/{test_res['total']} tests passed.")

                if not test_res.get("success", False):
                    self.consecutive_fails += 1
                else:
                    self.consecutive_fails = 0

            # Manager loop
            status, next_desc, tasks = self._primary_manage(tasks, nexts, summary)

            # Hard Override: If tests failed, manager cannot sign off as done
            if test_res.get("ran") and not test_res.get("success", False) and status == "done":
                status = "continue"
                next_desc = "Fix the failing test cases or try a different algorithmic formulation."
                self.log("    [primary-manage] OVERRIDE 'done' -> Tests are still failing. Continuing.")

        # Finalizer pass if not cleanly done
        final_code = self._read("solution.py").strip()
        final_test = self._run_tests()

        if not final_test.get("success", False) and self.best_solution_code:
            self.log("    [mitigation] Rolling back to best-so-far solution checkpoint to prevent regression.")
            final_code = self.best_solution_code
            self._write("solution.py", final_code)
            final_test = self._run_tests()

        if not final_code:
            self.log("    [finalize] Running emergency finalizer pass...")
            self._worker("Finalize solution", iters + 1, finalize=True)
            final_code = self._read("solution.py").strip()
            final_test = self._run_tests()

        # Mitigation 2: Pre-flight backup and atomic delivery
        delivered = False
        if final_code:
            self.target_path.parent.mkdir(parents=True, exist_ok=True)
            if self.target_path.exists():
                bak_path = self.target_path.with_suffix(self.target_path.suffix + ".bak")
                try:
                    shutil.copyfile(self.target_path, bak_path)
                    self.log(f"[GVS5H] Created backup of target file: {bak_path.name}")
                except Exception:
                    pass

            # Atomic write via temporary file
            temp_target = self.target_path.with_suffix(".tmp_gvs5h")
            temp_target.write_text(final_code, encoding="utf-8")
            temp_target.replace(self.target_path)
            delivered = True
            self.log(f"[GVS5H] Successfully delivered verified code to {self.target_path}")

        # Mitigation 6: Archive transcripts & clean temp sandbox
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for archive_file in ("transcript.jsonl", "plan.md", "notes.md", "solution.py"):
            src = self.ws / archive_file
            if src.exists():
                shutil.copyfile(src, self.logs_dir / archive_file)

        shutil.rmtree(self.ws, ignore_errors=True)
        self.log(f"[GVS5H] Sandbox cleaned up. Audit trail saved to: {self.logs_dir.name}\n")

        # Mitigation 7: Prune older log run folders down to latest retention limit (default: 100)
        prune_old_gvs5h_logs(max_runs=self.max_log_runs, logs_root=LOGS_ROOT)

        return {
            "status": "success" if (final_test.get("success") or delivered) else "failed",
            "delivered": delivered,
            "target_file": str(self.target_path),
            "iterations": iters,
            "duration_seconds": round(time.time() - self.start_time, 2),
            "tests_passed": f"{final_test.get('passed', 0)}/{final_test.get('total', 0)}",
            "logs_dir": str(self.logs_dir)
        }


# --- Programmatic API & CLI Runner ---

def run_gvs5h_solve(**kwargs) -> Dict[str, Any]:
    try:
        ensure_deep_solve_skills()
    except Exception as e:
        logger.debug(f"[deep-solve] Skill auto-provisioning notice: {e}")
    solver = GVS5HSolver(**kwargs)
    return solver.solve()


def main():
    parser = argparse.ArgumentParser(description="GVS5H Zero-Shot Self-Orchestrator Deep-Solve Bridge")
    parser.add_argument("--task", default=None, help="Task description or specification")
    parser.add_argument("--target", default=None, help="Destination source file path (e.g. src/solver.py)")
    parser.add_argument("--test-cmd", default=None, help="Optional test command to verify candidate solution")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"LocalAI / llama-server URL (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS, help="Max manager-worker rounds")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS, help="Wall-clock watchdog timeout in seconds")
    parser.add_argument("--test-timeout", type=float, default=DEFAULT_TEST_TIMEOUT, help="Max seconds per test run")
    parser.add_argument("--max-log-runs", type=int, default=None, help=f"Max recent log run folders to retain (default: {DEFAULT_MAX_LOG_RUNS})")
    parser.add_argument("--prune-logs", action="store_true", help="Immediately prune older log run folders in C:\\aiWorkspace\\logs\\gvs5h and exit")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    parser.add_argument("--ensure-skills", action="store_true", help="Provision deep-solve SKILL.md into Pi, Hermes, and workspace")

    args = parser.parse_args()

    if args.ensure_skills:
        paths = ensure_deep_solve_skills()
        print(json.dumps({"status": "provisioned", "skills": [str(p) for p in paths]}, indent=2))
        sys.exit(0)

    if args.prune_logs:
        pruned = prune_old_gvs5h_logs(max_runs=args.max_log_runs)
        print(json.dumps({
            "status": "pruned",
            "pruned_count": len(pruned),
            "pruned_folders": [p.name for p in pruned],
            "retained_limit": args.max_log_runs or get_default_max_log_runs()
        }, indent=2))
        sys.exit(0)

    if not args.task or not args.target:
        parser.error("--task and --target are required unless --ensure-skills or --prune-logs is passed.")

    res = run_gvs5h_solve(
        task=args.task,
        target_file=args.target,
        test_cmd=args.test_cmd,
        model=args.model,
        endpoint=args.endpoint,
        max_iters=args.max_iters,
        max_seconds=args.max_seconds,
        test_timeout=args.test_timeout,
        max_log_runs=args.max_log_runs,
        verbose=not args.quiet
    )
    print(json.dumps(res, indent=2))
    sys.exit(0 if res.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
