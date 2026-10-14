"""
Harness and Engine Characteristics Registry and Routing Model.

Defines extensible architectural specifications for all coding agent harnesses
and inference engines supported by LocalAI Launcher.

Manages capabilities including:
- Offline model execution capability (always, strata-only, or never)
- Meta-harness designation (ADE/wrapper harnesses like Orca or T3 Code that can host child harnesses)
- CLI/TUI vs GUI application distinction
- Leftmost priority resolution for LLM server context preloading
- Leftmost offline CLI harness resolution for meta-harness inner agent defaults
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class HarnessCharacteristics:
    """Architectural specification of a harness's runtime capabilities and characteristics."""
    canonical_name: str
    aliases: Tuple[str, ...] = ()
    is_meta_harness: bool = False
    is_cli: bool = True
    offline_support: str = "always"  # "always", "local_claude_native" (or legacy "strata_only"), "never"
    description: str = ""

    def is_offline_capable(
        self,
        model_name: Optional[str] = None,
        engine: Optional[str] = None,
        cfg: Optional[Any] = None
    ) -> bool:
        """Determines whether this harness can execute against local offline models
        under the active engine and runtime configuration."""
        if self.offline_support == "never":
            return False

        if self.offline_support == "always":
            return True

        if self.offline_support in ("local_claude_native", "strata_only"):
            return self._check_local_claude_native_support(model_name=model_name, engine=engine, cfg=cfg)

        return False

    def _check_local_claude_native_support(
        self,
        model_name: Optional[str] = None,
        engine: Optional[str] = None,
        cfg: Optional[Any] = None
    ) -> bool:
        """Verifies if a local Claude native engine (e.g. Strata or registered local_claude_native engines)
        is active and mode is not official-only."""
        from local_ai_launcher.context_preloader import is_local_claude_native_model, is_strata_model

        # 1. Check if model or engine targets a local Claude native engine (Strata is default member)
        is_native = is_local_claude_native_model(model_name) or is_strata_model(model_name)
        if not is_native and engine and is_local_claude_native_engine(str(engine), cfg=cfg):
            is_native = True
        if not is_native and cfg:
            default_m = getattr(cfg, "default_model", None)
            if is_local_claude_native_model(default_m) or is_strata_model(default_m):
                is_native = True
            active_eng = (
                getattr(cfg, "choose_engine", None)
                or getattr(cfg, "engine", None)
                or getattr(cfg, "blackwell_engine", None)
                or getattr(cfg, "local_claude_native_engine", None)
            )
            if active_eng and is_local_claude_native_engine(str(active_eng), cfg=cfg):
                is_native = True

        if not is_native:
            return False

        # 2. Check harness mode constraints (e.g. claude_mode != "official")
        if self.canonical_name == "claude":
            claude_mode = getattr(cfg, "claude_mode", "local") if cfg else "local"
            if str(claude_mode).strip().lower() == "official":
                return False
            return True

        if self.canonical_name == "claude-desktop":
            desktop_mode = getattr(cfg, "claude_desktop_mode", "gateway") if cfg else "gateway"
            if str(desktop_mode).strip().lower() in ("off", "disabled", "none"):
                return False
            return True

        return True

    def _check_strata_offline_support(
        self,
        model_name: Optional[str] = None,
        engine: Optional[str] = None,
        cfg: Optional[Any] = None
    ) -> bool:
        """Backward compatibility alias for _check_local_claude_native_support."""
        return self._check_local_claude_native_support(model_name=model_name, engine=engine, cfg=cfg)


@dataclass(frozen=True)
class EngineCharacteristics:
    """Architectural specification of a local LLM runtime engine."""
    engine_name: str
    aliases: Tuple[str, ...] = ()
    supports_anthropic_messages: bool = False
    supports_openai_completions: bool = True
    supports_speculative: bool = False
    is_local_claude_native: bool = False
    description: str = ""


# -----------------------------------------------------------------------------
# Canonical Harness Specifications Registry
# -----------------------------------------------------------------------------

HARNESS_SPECS: Dict[str, HarnessCharacteristics] = {
    "pi": HarnessCharacteristics(
        canonical_name="pi",
        aliases=("pi", "pie"),
        is_meta_harness=False,
        is_cli=True,
        offline_support="always",
        description="Pi coding agent terminal harness (autonomous software architect)."
    ),
    "hermes": HarnessCharacteristics(
        canonical_name="hermes",
        aliases=("hermes", "hermes-agent"),
        is_meta_harness=False,
        is_cli=True,
        offline_support="always",
        description="Hermes Agent autonomous coding CLI and terminal runtime."
    ),
    "deepseek": HarnessCharacteristics(
        canonical_name="deepseek",
        aliases=("deepseek", "dsh"),
        is_meta_harness=False,
        is_cli=True,
        offline_support="always",
        description="DeepSeek Harness (dsh) coding terminal agent."
    ),
    "claude": HarnessCharacteristics(
        canonical_name="claude",
        aliases=("claude", "claude-code", "claudecode", "claude-local", "claude-code-local"),
        is_meta_harness=False,
        is_cli=True,
        offline_support="local_claude_native",
        description="Claude Code CLI terminal agent (offline with local Claude native engines, e.g. Strata)."
    ),
    "claude-desktop": HarnessCharacteristics(
        canonical_name="claude-desktop",
        aliases=("claude-desktop", "claudedesktop", "claude-desktop-local", "claude-desktop-gateway"),
        is_meta_harness=False,
        is_cli=False,  # Full desktop GUI application, not a CLI
        offline_support="local_claude_native",
        description="Claude Desktop GUI application (offline Developer Gateway with local Claude native engines, e.g. Strata)."
    ),
    "opencode": HarnessCharacteristics(
        canonical_name="opencode",
        aliases=("opencode",),
        is_meta_harness=False,
        is_cli=True,
        offline_support="always",
        description="OpenCode local AI coding terminal driver."
    ),
    "openlumara": HarnessCharacteristics(
        canonical_name="openlumara",
        aliases=("openlumara", "lumara"),
        is_meta_harness=False,
        is_cli=True,
        offline_support="always",
        description="OpenLumara / Trae agent coding terminal harness."
    ),
    "orca": HarnessCharacteristics(
        canonical_name="orca",
        aliases=("orca",),
        is_meta_harness=True,  # Meta-harness capable of hosting Pi, Hermes, Claude, Trae inside
        is_cli=False,          # Full desktop ADE GUI
        offline_support="always",
        description="Stably AI Orca Agent Development Environment (ADE meta-harness)."
    ),
    "t3": HarnessCharacteristics(
        canonical_name="t3",
        aliases=("t3", "t3code"),
        is_meta_harness=True,  # Meta-harness capable of hosting OpenCode, Pi, Hermes inside
        is_cli=False,          # Full desktop ADE / web UI
        offline_support="always",
        description="T3 Code agent control surface (ADE meta-harness)."
    ),
    "antigravity": HarnessCharacteristics(
        canonical_name="antigravity",
        aliases=("antigravity", "agy", "agy-ide", "agy-desktop"),
        is_meta_harness=False,
        is_cli=False,
        offline_support="never",
        description="Google Antigravity Cloud IDE / Desktop (online-only)."
    ),
    "claude-official": HarnessCharacteristics(
        canonical_name="claude-official",
        aliases=("claude-official", "claude-code-official", "claude-desktop-official"),
        is_meta_harness=False,
        is_cli=True,
        offline_support="never",
        description="Official Anthropic Claude subscription mode (online-only)."
    ),
    "ls": HarnessCharacteristics(
        canonical_name="ls",
        aliases=("ls", "llama-server-ui"),
        is_meta_harness=False,
        is_cli=False,
        offline_support="always",
        description="llama-server native web UI."
    ),
}

# Fast alias-to-canonical lookup table
HARNESS_ALIAS_MAP: Dict[str, str] = {}
for _canonical, _spec in HARNESS_SPECS.items():
    HARNESS_ALIAS_MAP[_canonical.lower()] = _canonical
    for _alias in _spec.aliases:
        HARNESS_ALIAS_MAP[_alias.lower()] = _canonical


# -----------------------------------------------------------------------------
# Canonical Engine Specifications Registry
# -----------------------------------------------------------------------------

ENGINE_SPECS: Dict[str, EngineCharacteristics] = {
    "strata": EngineCharacteristics(
        engine_name="strata",
        aliases=("strata", "strata-engine", "strata_engine"),
        supports_anthropic_messages=True,
        supports_openai_completions=True,
        supports_speculative=True,
        is_local_claude_native=True,
        description="Strata 125B MoE speculative decoding engine (member of local_claude_native; native Anthropic messages & OpenAI API)."
    ),
    "huang": EngineCharacteristics(
        engine_name="huang",
        aliases=("huang", "llama-cpp-huang"),
        supports_anthropic_messages=False,
        supports_openai_completions=True,
        supports_speculative=True,
        description="Raymond Huang adaptive KV streaming llama-server engine (fixed arena, draft-mtp)."
    ),
    "bruno": EngineCharacteristics(
        engine_name="bruno",
        aliases=("bruno", "llama-cpp-blackwell"),
        supports_anthropic_messages=False,
        supports_openai_completions=True,
        supports_speculative=False,
        description="Bruno Passini sparse VMM 256k llama-server engine."
    ),
    "official": EngineCharacteristics(
        engine_name="official",
        aliases=("official", "llama-cpp", "llamacpp"),
        supports_anthropic_messages=False,
        supports_openai_completions=True,
        supports_speculative=False,
        description="Official llama.cpp baseline C++ inference engine."
    ),
    "tabbyapi": EngineCharacteristics(
        engine_name="tabbyapi",
        aliases=("tabbyapi", "tabby"),
        supports_anthropic_messages=False,
        supports_openai_completions=True,
        supports_speculative=False,
        description="TabbyAPI EXL3 high-throughput inference engine."
    ),
    "freetoken": EngineCharacteristics(
        engine_name="freetoken",
        aliases=("freetoken",),
        supports_anthropic_messages=False,
        supports_openai_completions=True,
        supports_speculative=False,
        description="FreeToken NVFP4 inference engine."
    ),
}

# -----------------------------------------------------------------------------
# Local Claude Native Engines Registry (Shared across Anthropic-compatible engines)
# -----------------------------------------------------------------------------

DEFAULT_LOCAL_CLAUDE_NATIVE_ENGINES: Tuple[str, ...] = ("strata",)
_REGISTERED_LOCAL_CLAUDE_NATIVE_ENGINES: List[str] = list(DEFAULT_LOCAL_CLAUDE_NATIVE_ENGINES)


def get_local_claude_native_engines(cfg: Optional[Any] = None) -> List[str]:
    """Returns the list of engines that provide local Claude native (Anthropic Messages) support.

    Includes canonical members (defaulting to 'strata') and dynamically registered or
    configured engines from LauncherConfig or ENGINE_SPECS.
    """
    engines = list(_REGISTERED_LOCAL_CLAUDE_NATIVE_ENGINES)
    for e_name, e_spec in ENGINE_SPECS.items():
        if (getattr(e_spec, "is_local_claude_native", False) or getattr(e_spec, "supports_anthropic_messages", False)) and e_name not in engines:
            engines.append(e_name)
    if cfg:
        cfg_engines = getattr(cfg, "local_claude_native_engines", None)
        if cfg_engines:
            for ce in cfg_engines:
                ce_s = str(ce).strip().lower()
                if ce_s and ce_s not in engines:
                    engines.append(ce_s)
    return engines


def is_local_claude_native_engine(engine: Optional[str], cfg: Optional[Any] = None) -> bool:
    """Returns True if the engine name or alias is recognized as a local Claude native engine."""
    if not engine:
        return False
    e_str = str(engine).strip().lower()
    known = get_local_claude_native_engines(cfg)
    if e_str in known:
        return True
    for k in known:
        if k in e_str:
            return True
    for eng_id, spec in ENGINE_SPECS.items():
        if getattr(spec, "is_local_claude_native", False) or getattr(spec, "supports_anthropic_messages", False):
            if e_str == eng_id or e_str in [a.lower() for a in getattr(spec, "aliases", ())]:
                return True
    return False


def register_local_claude_native_engine(
    engine_name: str,
    characteristics: Optional[EngineCharacteristics] = None
) -> None:
    """Registers an engine as a member of local_claude_native, allowing expandability to new LLM engines."""
    clean = str(engine_name).strip().lower()
    if clean not in _REGISTERED_LOCAL_CLAUDE_NATIVE_ENGINES:
        _REGISTERED_LOCAL_CLAUDE_NATIVE_ENGINES.append(clean)
    if characteristics is not None:
        ENGINE_SPECS[clean] = characteristics
    elif clean in ENGINE_SPECS:
        old = ENGINE_SPECS[clean]
        ENGINE_SPECS[clean] = EngineCharacteristics(
            engine_name=old.engine_name,
            aliases=old.aliases,
            supports_anthropic_messages=True,
            supports_openai_completions=old.supports_openai_completions,
            supports_speculative=old.supports_speculative,
            is_local_claude_native=True,
            description=old.description or f"{engine_name} engine with local Claude native support."
        )


# -----------------------------------------------------------------------------
# Helper Resolution Functions
# -----------------------------------------------------------------------------

def get_harness_characteristics(name: str) -> Optional[HarnessCharacteristics]:
    """Retrieves the HarnessCharacteristics for a given harness name or alias."""
    if not name:
        return None
    key = str(name).strip().lower()
    canonical = HARNESS_ALIAS_MAP.get(key)
    if canonical:
        return HARNESS_SPECS.get(canonical)
    return None


def is_meta_harness(name: str) -> bool:
    """Returns True if the harness is an ADE/meta-harness capable of loading other harnesses inside."""
    spec = get_harness_characteristics(name)
    return spec.is_meta_harness if spec else False


def is_cli_harness(name: str) -> bool:
    """Returns True if the harness is a command-line / terminal-based agent (not a GUI app)."""
    spec = get_harness_characteristics(name)
    return spec.is_cli if spec else False


def is_offline_capable(
    name: str,
    model_name: Optional[str] = None,
    engine: Optional[str] = None,
    cfg: Optional[Any] = None
) -> bool:
    """Returns True if the harness can execute offline models under the current engine configuration."""
    spec = get_harness_characteristics(name)
    if not spec:
        return False
    return spec.is_offline_capable(model_name=model_name, engine=engine, cfg=cfg)


def normalize_harness_list(choose_harness: Any) -> List[str]:
    """Normalizes choose_harness (list, tuple, comma-separated string) into a clean list of strings."""
    if not choose_harness:
        return []
    if isinstance(choose_harness, (list, tuple)):
        items = [str(h).strip() for h in choose_harness if h]
    elif isinstance(choose_harness, str):
        # Support space or comma delimited string if passed as single string
        items = [h.strip() for h in choose_harness.replace(",", " ").split() if h.strip()]
    else:
        items = [str(choose_harness).strip()]
    return [h for h in items if h.lower() not in ("none", "off", "disabled", "false", "no", "")]


def resolve_server_preloaded_harness(
    choose_harness: Any,
    model_name: Optional[str] = None,
    engine: Optional[str] = None,
    cfg: Optional[Any] = None
) -> str:
    """Resolves which harness environment should be preloaded into the running local LLM engine server.

    Rule 1:
    If more than 1 offline-capable harness is defined in choose_harness,
    the LEFT-MOST harness that is NOT an offline model capable meta-harness
    (a meta-harness like Orca ADE or T3 Code can load other harnesses inside)
    is the harness environment to be loaded.

    Example:
      choose_harness = ["orca", "t3", "claudedesktop", "pi", "hermes"]
      - "orca": meta-harness -> skipped
      - "t3": meta-harness -> skipped
      - "claudedesktop": non-meta, offline-capable (under Strata) -> SELECTED!
      -> returns "claudedesktop"
      (pi, hermes, orca, t3 contexts are not loaded into the engine server).

    Edge Case:
      If ONLY meta-harnesses are present in choose_harness (e.g. ["orca"] or ["t3code"]),
      the server preloads the leftmost meta-harness's default inner CLI harness (default "pi")
      so time-to-first-token is minimized when the meta-harness opens.
    """
    raw_list = normalize_harness_list(choose_harness)
    if not raw_list:
        return "pi"

    # 1. Filter for harnesses that are offline-capable under active engine & settings
    offline_candidates = []
    for h in raw_list:
        spec = get_harness_characteristics(h)
        if spec and spec.is_offline_capable(model_name=model_name, engine=engine, cfg=cfg):
            offline_candidates.append(h)

    if not offline_candidates:
        # Fallback to pi if no offline harnesses matched
        return "pi"

    # 2. Find leftmost harness that is NOT a meta-harness
    for h in offline_candidates:
        if not is_meta_harness(h):
            spec = get_harness_characteristics(h)
            # Return canonical name or original entry if mapped
            return spec.canonical_name if spec else h

    # 3. If ONLY meta-harnesses were specified (e.g. ["orca"] or ["t3code"]),
    # resolve the default inner CLI harness for the leftmost meta-harness (defaults to "pi").
    return resolve_meta_harness_inner_cli_harness(raw_list, model_name=model_name, engine=engine, cfg=cfg)


def resolve_meta_harness_inner_cli_harness(
    choose_harness: Any,
    model_name: Optional[str] = None,
    engine: Optional[str] = None,
    cfg: Optional[Any] = None
) -> str:
    """Resolves the inner CLI harness context for the leftmost meta-harness (e.g. Orca ADE or T3 Code).

    Rule 2:
    In the leftmost meta-harness load, have the harness context of the
    leftmost non-meta harness that supports offline CLI.

    Example:
      choose_harness = ["orca", "t3", "claudedesktop", "pi", "hermes"]
      - "orca": meta-harness -> skipped
      - "t3": meta-harness -> skipped
      - "claudedesktop": is_cli is False (full GUI app) -> skipped
      - "pi": is_cli is True, offline-capable -> SELECTED!
      -> returns "pi"
      Orca ADE or T3 Code initializes with Pi harness context.

    Fallback:
      If no offline CLI harness is explicitly present in choose_harness
      (e.g. choose_harness = ["orca"] or ["t3code"] or ["orca", "claudedesktop"]),
      defaults to "pi" for both Orca and T3 Code.
    """
    raw_list = normalize_harness_list(choose_harness)

    for h in raw_list:
        spec = get_harness_characteristics(h)
        if not spec:
            continue
        # Must be non-meta, offline-capable, and a CLI/TUI agent
        if not spec.is_meta_harness and spec.is_cli and spec.is_offline_capable(model_name=model_name, engine=engine, cfg=cfg):
            return spec.canonical_name

    # Default inner CLI harness for meta-harnesses:
    # When a local Claude native engine (such as Strata) is active, offline Claude Code CLI is the default ("claude");
    # otherwise, Pi is the default ("pi").
    from local_ai_launcher.context_preloader import is_local_claude_native_model, is_strata_model
    is_native = is_local_claude_native_model(model_name) or is_strata_model(model_name)
    if not is_native and engine and is_local_claude_native_engine(engine, cfg=cfg):
        is_native = True
    if not is_native and cfg:
        default_m = getattr(cfg, "default_model", None)
        if is_local_claude_native_model(default_m) or is_strata_model(default_m):
            is_native = True
        active_eng = (
            getattr(cfg, "choose_engine", None)
            or getattr(cfg, "engine", None)
            or getattr(cfg, "blackwell_engine", None)
            or getattr(cfg, "local_claude_native_engine", None)
        )
        if active_eng and is_local_claude_native_engine(active_eng, cfg=cfg):
            is_native = True

    if is_native:
        claude_mode = getattr(cfg, "claude_mode", "local") if cfg else "local"
        if str(claude_mode).strip().lower() != "official":
            return "claude"

    return "pi"
