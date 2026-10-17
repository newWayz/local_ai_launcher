"""
Unified Driver Context Preloader for Coding Harnesses (Pi, Hermes, OpenCode on T3 Code).

Pre-evaluates the canonical agent system prompt and function/tool schemas into the
LLM's KV cache (e.g., llama-server / LocalAI / OmniRoute) via a lightweight,
in-process HTTP request with max_tokens=1.

This achieves sub-30ms Time To First Token (TTFT) on initial user turn without:
1. Spawning heavy external PowerShell or shell processes.
2. Generating unconstrained response tokens.
3. Creating throwaway session directories or polluting session SQLite databases.
"""

import json
import time
import logging
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional

from local_ai_launcher.config import config, AI_WORKSPACE_DIR, CONFIG_DIR, LOCAL_AI_LAUNCHER_DIR, logger


# ---------------------------------------------------------------------------
# Canonical Tool Schemas
# ---------------------------------------------------------------------------

OPENCODE_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a command in the shell environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command string to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view",
            "description": "View file contents or list directory entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file or directory."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace occurrences of text within an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "old_string": {"type": "string", "description": "Exact text to replace."},
                    "new_string": {"type": "string", "description": "New replacement content."}
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "patch",
            "description": "Apply a unified diff patch to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "patch": {"type": "string", "description": "Unified diff content."}
                },
                "required": ["path", "patch"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. **/*.py)."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents using regular expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {"type": "string", "description": "Directory or file path to search within."}
                },
                "required": ["pattern"]
            }
        }
    }
]

PI_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file contents with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file to read."},
                    "start_line": {"type": "integer", "description": "Optional start line number."},
                    "end_line": {"type": "integer", "description": "Optional end line number."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or overwrite a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "content": {"type": "string", "description": "Complete file content."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit an existing file by replacing a unique chunk of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "old_chunk": {"type": "string", "description": "Exact text to replace."},
                    "new_chunk": {"type": "string", "description": "Replacement text."}
                },
                "required": ["path", "old_chunk", "new_chunk"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Find files by name or pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Filename or glob pattern."},
                    "path": {"type": "string", "description": "Root path to search."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents by regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to match."},
                    "path": {"type": "string", "description": "Path to search within."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List directory contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."}
                },
                "required": ["command"]
            }
        }
    }
]

HERMES_FALLBACK_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Execute command in the terminal session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_editor",
            "description": "View, create, or modify files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"]},
                    "path": {"type": "string", "description": "File path."},
                    "file_text": {"type": "string", "description": "File content for create."},
                    "old_str": {"type": "string", "description": "Target string for str_replace."},
                    "new_str": {"type": "string", "description": "New string for str_replace."}
                },
                "required": ["command", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deep_solve",
            "description": "Dispatch complex algorithm or math synthesis to GVS5H local solver.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Task description to solve."}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cpu_rerank",
            "description": "Score and rank candidate passages using CPU cross-encoder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Target query."},
                    "documents": {"type": "array", "items": {"type": "string"}, "description": "Candidate passages."},
                    "top_n": {"type": "integer", "description": "Number of top passages to return."}
                },
                "required": ["query", "documents"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cpu_decide",
            "description": "Make low-latency classification or routing decision via CPU ONNX gater.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Decision context."},
                    "candidate_options": {"type": "array", "items": {"type": "string"}, "description": "Allowed candidate choices."}
                },
                "required": ["prompt", "candidate_options"]
            }
        }
    }
]


def _load_canonical_hermes_tools() -> List[Dict[str, Any]]:
    """Loads canonical Hermes function calling tool definitions matching Hermes Agent runtime."""
    tools_paths = [
        LOCAL_AI_LAUNCHER_DIR / "hermes_tools.json",
        AI_WORKSPACE_DIR / "local_ai_launcher" / "hermes_tools.json",
        AI_WORKSPACE_DIR / ".hermes" / "hermes_tools.json",
        Path.home() / ".hermes" / "hermes_tools.json",
    ]
    for p in tools_paths:
        if p and p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    return data
            except Exception as e:
                logger.debug(f"[ContextPreloader] Failed to read {p}: {e}")
    return HERMES_FALLBACK_TOOLS


HERMES_TOOLS: List[Dict[str, Any]] = _load_canonical_hermes_tools()

DEEPSEEK_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a UTF-8 text file and return line-numbered content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to read, resolved by the filesystem backend."},
                    "offset": {"type": "number", "description": "1-based first line to return. Defaults to 1."},
                    "limit": {"type": "number", "description": "Maximum number of lines to return. Defaults to 2000."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or fully replace a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to write, resolved by the filesystem backend."},
                    "content": {"type": "string", "description": "Full UTF-8 text content to write."}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit an existing UTF-8 text file by replacing literal text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to edit, resolved by the filesystem backend."},
                    "old_string": {"type": "string", "description": "Literal text to replace. Must match exactly."},
                    "new_string": {"type": "string", "description": "Literal replacement text. Use an empty string to delete the match."},
                    "replace_all": {"type": "boolean", "description": "Replace all matches. Defaults to false; when false, old_string must appear exactly once."}
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files whose paths match a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match file paths against."},
                    "path": {"type": "string", "description": "Directory to search in. Defaults to the session workspace."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with a ripgrep regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "File or directory to search."},
                    "include": {"type": "string", "description": "One glob filter for which files to search."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pwsh",
            "description": "Execute a PowerShell command (pwsh -Command) and return its stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The PowerShell command to execute."},
                    "description": {"type": "string", "description": "Clear, concise description of what this command does."}
                },
                "required": ["command", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_image",
            "description": "Read a PNG/JPEG/WebP/GIF file and return the image itself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the image file, resolved by the filesystem backend."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_question",
            "description": "Ask the user a concise question when you need confirmation, a choice, or missing information before proceeding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "Questions to ask the user before continuing.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Stable id for this question; echoed in the answer."},
                                "question": {"type": "string", "description": "The specific question to ask the user."}
                            },
                            "required": ["id", "question"]
                        }
                    }
                },
                "required": ["questions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "skill",
            "description": "Load the full instructions for an available skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The exact skill name from the available skills list."}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Record and update a structured task list for the current work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The COMPLETE task list, replacing any previous list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "What the task is — a short imperative line."},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Canonical System Prompt Builders
# ---------------------------------------------------------------------------

def is_local_claude_native_model(model_name: Optional[str]) -> bool:
    """Returns True if the target model or active configuration targets a local Claude native engine
    (such as Strata or any configured local_claude_native engine)."""
    m_str = str(model_name or "").lower().strip()
    try:
        from local_ai_launcher.harness_model import get_local_claude_native_engines
        claude_engines = get_local_claude_native_engines(config)
    except Exception:
        claude_engines = ["strata"]

    if any(eng in m_str for eng in claude_engines):
        return True

    try:
        from local_ai_launcher.presets import load_effective_models_parser
        parser = load_effective_models_parser()
        if m_str in parser:
            eng = parser[m_str].get("engine", "").lower().strip()
            if eng in claude_engines:
                return True
        for sec in parser.sections():
            aliases = [a.strip().lower() for a in parser.get(sec, "alias", fallback="").split(",") if a.strip()]
            if m_str in aliases:
                eng = parser.get(sec, "engine", fallback="").lower().strip()
                if eng in claude_engines:
                    return True
    except Exception:
        pass

    try:
        active_engine = (
            getattr(config, "choose_engine", None)
            or getattr(config, "default_engine", None)
            or getattr(config, "engine", None)
            or getattr(config, "local_claude_native_engine", None)
        )
        if active_engine and any(eng in str(active_engine).lower() for eng in claude_engines):
            return True
    except Exception:
        pass
    return False


def is_strata_model(model_name: Optional[str]) -> bool:
    """Returns True if the target model or active configuration targets the Strata engine specifically."""
    m_str = str(model_name or "").lower().strip()
    if not m_str:
        return False
    if "strata" in m_str:
        return True
    try:
        from local_ai_launcher.backends.strata import load_strata_model_registry
        reg = load_strata_model_registry()
        for k, v in reg.items():
            if m_str == k.lower():
                return True
            raw_aliases = v.get("aliases") or v.get("alias", [])
            if isinstance(raw_aliases, str):
                aliases = [a.strip().lower() for a in raw_aliases.split(",") if a.strip()]
            elif isinstance(raw_aliases, list):
                aliases = [str(a).strip().lower() for a in raw_aliases if str(a).strip()]
            else:
                aliases = []
            if m_str in aliases:
                return True
            m_model = v.get("model") or v.get("model_path", "")
            if m_model:
                clean_m = m_model.strip().lower().replace("\\", "/")
                target_p = m_str.replace("\\", "/")
                if (clean_m == target_p or
                    Path(clean_m).name == Path(target_p).name or
                    target_p.endswith(clean_m) or
                    clean_m.endswith(target_p)):
                    return True
    except Exception:
        pass
    try:
        from local_ai_launcher.presets import load_effective_models_parser
        parser = load_effective_models_parser()
        if m_str in parser:
            eng = parser[m_str].get("engine", "").lower().strip()
            if eng == "strata":
                return True
        for sec in parser.sections():
            aliases = [a.strip().lower() for a in parser.get(sec, "alias", fallback="").split(",") if a.strip()]
            if m_str in aliases:
                eng = parser.get(sec, "engine", fallback="").lower().strip()
                if eng == "strata":
                    return True
    except Exception:
        pass
    try:
        active_engine = getattr(config, "choose_engine", None) or getattr(config, "default_engine", None)
        if active_engine and "strata" in str(active_engine).lower():
            return True
    except Exception:
        pass
    return False


def _load_user_instructions() -> str:
    """Loads USER.md context from config directory if present."""
    user_md = CONFIG_DIR / "USER.md"
    if user_md.exists():
        try:
            return user_md.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def _load_agents_instructions() -> str:
    """Loads AGENTS.md context from workspace root if present."""
    agents_md = AI_WORKSPACE_DIR / "AGENTS.md"
    if agents_md.exists():
        try:
            return agents_md.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def _load_hermes_soul() -> str:
    """Loads SOUL.md from Hermes configuration directory."""
    candidates = [
        AI_WORKSPACE_DIR / ".hermes" / "SOUL.md",
        Path.home() / ".hermes" / "SOUL.md",
    ]
    for p in candidates:
        if p and p.is_file():
            try:
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
            except Exception:
                pass
    return (
        "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
        "You are helpful, knowledgeable, and direct. You assist users with a wide range of tasks "
        "including answering questions, writing and editing code, analyzing information, creative work, "
        "and executing actions via your tools."
    )


def _load_canonical_hermes_prompt(model_name: Optional[str] = None) -> str:
    """Loads canonical Hermes system prompt matching what Hermes Agent builds at session turn 1."""
    is_strata = is_strata_model(model_name)
    candidates = []
    if is_strata:
        candidates.extend([
            LOCAL_AI_LAUNCHER_DIR / "hermes_system_prompt_strata.txt",
            AI_WORKSPACE_DIR / "local_ai_launcher" / "hermes_system_prompt_strata.txt",
            AI_WORKSPACE_DIR / ".hermes" / "hermes_system_prompt_strata.txt",
            Path.home() / ".hermes" / "hermes_system_prompt_strata.txt",
        ])
    candidates.extend([
        LOCAL_AI_LAUNCHER_DIR / "hermes_system_prompt.txt",
        AI_WORKSPACE_DIR / "local_ai_launcher" / "hermes_system_prompt.txt",
        AI_WORKSPACE_DIR / ".hermes" / "hermes_system_prompt.txt",
        Path.home() / ".hermes" / "hermes_system_prompt.txt",
    ])
    for p in candidates:
        if p and p.is_file():
            try:
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
            except Exception:
                pass

    # Dynamic assembly fallback matching Hermes architecture:
    soul = _load_hermes_soul()
    agents_directives = _load_agents_instructions()
    user_directives = _load_user_instructions()

    parts = [soul]
    if agents_directives:
        parts.append(f"Workspace Directives:\n{agents_directives}")
    if user_directives:
        parts.append(f"User Directives:\n{user_directives}")
    if not is_strata:
        parts.append("Use auxiliary CPU microservices for fast discrete gating and document reranking.")
    return "\n\n".join(parts)


def build_harness_prefix(harness_name: str, model_name: str) -> Dict[str, Any]:
    """Constructs the deterministic system prompt and tool definitions for a given harness."""
    name = (harness_name or "").lower().strip()
    user_context = _load_user_instructions()

    if name in ("t3", "opencode", "t3code"):
        system_parts = [
            "You are OpenCode, an AI coding assistant powered by the local workspace runtime.",
            "You have access to tools to read, edit, and explore the workspace repository.",
            "Always inspect relevant files before modifying them.",
        ]
        if user_context:
            system_parts.append(f"\nUser Directives:\n{user_context}")
        system_content = "\n\n".join(system_parts)
        return {
            "messages": [{"role": "system", "content": system_content}],
            "tools": OPENCODE_TOOLS
        }

    elif name == "pi":
        agents_context = _load_agents_instructions()
        system_parts = [
            "You are Pi, a senior autonomous software architect and pair-programming assistant.",
            "Always inspect and verify code changes using tests before concluding tasks.",
        ]
        if agents_context:
            system_parts.append(f"\nWorkspace Directives:\n{agents_context}")
        elif user_context:
            system_parts.append(f"\nUser Directives:\n{user_context}")
        system_content = "\n\n".join(system_parts)
        return {
            "messages": [{"role": "system", "content": system_content}],
            "tools": PI_TOOLS
        }

    elif name in ("omp", "oh-my-pi", "ohmypi"):
        agents_context = _load_agents_instructions()
        system_parts = [
            "You are Oh My Pi (omp), an advanced terminal AI coding agent and software architect.",
            "Always inspect and verify code changes using tests before concluding tasks.",
        ]
        if agents_context:
            system_parts.append(f"\nWorkspace Directives:\n{agents_context}")
        elif user_context:
            system_parts.append(f"\nUser Directives:\n{user_context}")
        system_content = "\n\n".join(system_parts)
        return {
            "messages": [{"role": "system", "content": system_content}],
            "tools": PI_TOOLS
        }

    elif name in ("hermes", "hermes-agent"):
        system_content = _load_canonical_hermes_prompt(model_name)
        return {
            "messages": [{"role": "system", "content": system_content}],
            "tools": _load_canonical_hermes_tools()
        }

    elif name in ("deepseek", "dsh"):
        agents_context = _load_agents_instructions()
        user_context = _load_user_instructions()
        system_parts = [
            "You are an AI agent powered by DeepSeek Harness.",
            f"You are a coding agent powered by the {model_name} model. Your working directory is {AI_WORKSPACE_DIR}.",
            "Use the read, write, edit, glob, grep, and pwsh tools to inspect and update the workspace.",
        ]
        if agents_context:
            system_parts.append(f"\nWorkspace Directives:\n{agents_context}")
        elif user_context:
            system_parts.append(f"\nUser Directives:\n{user_context}")
        system_content = "\n\n".join(system_parts)
        return {
            "messages": [{"role": "system", "content": system_content}],
            "tools": DEEPSEEK_TOOLS
        }

    elif name in ("claude", "claude-code", "claudecode", "claude-desktop", "claudedesktop"):
        agents_context = _load_agents_instructions()
        user_context = _load_user_instructions()
        system_parts = [
            "You are Claude, an AI assistant powered by the local workspace runtime.",
            "Always inspect and verify code changes using tests before concluding tasks.",
        ]
        if agents_context:
            system_parts.append(f"\nWorkspace Directives:\n{agents_context}")
        elif user_context:
            system_parts.append(f"\nUser Directives:\n{user_context}")
        system_content = "\n\n".join(system_parts)
        return {
            "messages": [{"role": "system", "content": system_content}],
            "tools": OPENCODE_TOOLS
        }

    elif name == "herdr":
        agents_context = _load_agents_instructions()
        system_parts = [
            "You are an AI coding assistant operating inside Herdr, the multi-agent terminal runtime.",
            "Always inspect and verify code changes using tests before concluding tasks.",
        ]
        if agents_context:
            system_parts.append(f"\nWorkspace Directives:\n{agents_context}")
        elif user_context:
            system_parts.append(f"\nUser Directives:\n{user_context}")
        system_content = "\n\n".join(system_parts)
        return {
            "messages": [{"role": "system", "content": system_content}],
            "tools": PI_TOOLS
        }

    else:
        # Default fallback prefix
        system_content = "You are an AI coding assistant in a local development environment."
        if user_context:
            system_content += f"\n\n{user_context}"
        return {
            "messages": [{"role": "system", "content": system_content}],
            "tools": OPENCODE_TOOLS
        }


# ---------------------------------------------------------------------------
# Background In-Process Pre-Evaluation
# ---------------------------------------------------------------------------

def is_local_endpoint(url: Optional[str]) -> bool:
    """Returns True if the URL targets a local loopback server."""
    if not url:
        return False
    u = url.lower()
    return any(h in u for h in ("127.0.0.1", "localhost", "0.0.0.0", "::1"))


_preloaded_contexts = set()
_preloaded_harness_by_model = {}
_preloaded_lock = threading.Lock()


def mark_context_preloaded(base_url: str, model_name: str, harness_name: str) -> None:
    """Mark a harness context as pre-evaluated in the KV cache."""
    norm_url = (base_url or "").rstrip("/").lower()
    m_name = str(model_name).lower()
    h_name = str(harness_name).lower()
    with _preloaded_lock:
        _preloaded_contexts.add((norm_url, m_name, h_name))
        _preloaded_harness_by_model[(norm_url, m_name)] = h_name


def is_context_preloaded(base_url: str, model_name: str, harness_name: str) -> bool:
    """Check if a harness context was already pre-evaluated in this session."""
    norm_url = (base_url or "").rstrip("/").lower()
    with _preloaded_lock:
        return (norm_url, str(model_name).lower(), str(harness_name).lower()) in _preloaded_contexts


def get_preloaded_harness_for_model(base_url: str, model_name: str) -> Optional[str]:
    """Returns the harness name whose context is currently preloaded for this model, if any."""
    norm_url = (base_url or "").rstrip("/").lower()
    with _preloaded_lock:
        return _preloaded_harness_by_model.get((norm_url, str(model_name).lower()))


def clear_preloaded_cache() -> None:
    """Clear recorded preloaded contexts (primarily for test isolation)."""
    with _preloaded_lock:
        _preloaded_contexts.clear()
        _preloaded_harness_by_model.clear()


def preload_harness_context(
    llm_base_url: str,
    model_name: str,
    harness_name: str,
    delay_seconds: float = 0.5,
    timeout: float = 20.0,
    force: bool = False
) -> Optional[threading.Thread]:
    """Pre-evaluates the harness's system prompt and tool definitions into the LLM KV cache.

    Sends an asynchronous HTTP request with max_tokens=1 to /v1/chat/completions so
    the local backend fills its prefix cache slot without generating extra tokens or
    spawning external shell processes.
    """
    if not getattr(config, "enable_context_preloading", True):
        logger.debug(f"[ContextPreloader] Context preloading disabled by configuration.")
        return None

    if not is_local_endpoint(llm_base_url):
        logger.debug(f"[ContextPreloader] Skipping preloading for non-local endpoint: {llm_base_url}")
        return None

    effective_model = model_name
    if "1-" in str(model_name) or "combo" in str(model_name).lower():
        effective_model = getattr(config, "default_model", model_name)

    if not force:
        if is_context_preloaded(llm_base_url, effective_model, harness_name):
            logger.debug(f"[ContextPreloader] Context already preloaded for '{harness_name}' ({effective_model}). Skipping redundant prefill.")
            return None
        existing_h = get_preloaded_harness_for_model(llm_base_url, effective_model)
        if existing_h and existing_h != str(harness_name).lower():
            logger.info(
                f"[ContextPreloader] Designated harness '{existing_h}' is already preloaded for model '{effective_model}'. "
                f"Skipping context prefill for '{harness_name}' to preserve designated harness environment."
            )
            return None

    prefix = build_harness_prefix(harness_name, effective_model)

    effective_timeout = timeout
    if is_strata_model(effective_model):
        effective_timeout = max(timeout, 180.0)

    def _worker():
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)

            base = (llm_base_url or "").rstrip("/")
            if not base.endswith("/v1"):
                url = f"{base}/v1/chat/completions"
            else:
                url = f"{base}/chat/completions"

            messages = list(prefix.get("messages", []))
            # Minimal user message to conform to standard chat completion schemas
            messages.append({"role": "user", "content": "ping"})

            payload = {
                "model": effective_model,
                "messages": messages,
                "tools": prefix.get("tools", []),
                "max_tokens": 1,
                "temperature": 0.0
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"HarnessContextPreloader/{harness_name}"
                }
            )
            start_t = time.time()
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                        resp.read()
                    break
                except urllib.error.HTTPError as http_err:
                    # If backend rejects tools parameter (400 Bad Request), gracefully fall back to messages-only
                    if http_err.code == 400 and payload.get("tools"):
                        logger.debug(f"[ContextPreloader] Backend returned 400 with tools for '{harness_name}', falling back to messages-only prefill...")
                        fallback_payload = dict(payload)
                        fallback_payload.pop("tools", None)
                        fallback_body = json.dumps(fallback_payload).encode("utf-8")
                        fallback_req = urllib.request.Request(
                            url,
                            data=fallback_body,
                            headers={
                                "Content-Type": "application/json",
                                "User-Agent": f"HarnessContextPreloader/{harness_name}"
                            }
                        )
                        with urllib.request.urlopen(fallback_req, timeout=effective_timeout) as fallback_resp:
                            fallback_resp.read()
                        break
                    else:
                        raise
                except (urllib.error.URLError, ConnectionError) as conn_err:
                    if attempt < max_retries - 1 and delay_seconds >= 0:
                        time.sleep(1.0)
                        continue
                    raise

            mark_context_preloaded(llm_base_url, effective_model, harness_name)
            duration = time.time() - start_t
            logger.info(
                f"  -> Pre-evaluated context into KV cache for '{harness_name}' "
                f"({len(prefix.get('tools', []))} tools, model: {effective_model}, {duration:.2f}s)"
            )
        except (Exception, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                return
            logger.warning(f"[ContextPreloader] Notice: Pre-evaluation for '{harness_name}' non-critical exit: {e}")

    t = threading.Thread(target=_worker, daemon=True, name=f"ContextPreload_{harness_name}")
    t.start()
    return t
