"""
Model preset resolution and high-fidelity layered configuration overlays.
Supports merging baseline presets (e.g. models.ini) with user/hardware-specific
local overrides (e.g. models.local.ini) safely, atomically, and without losing tuning fidelity.
"""
import os
import tempfile
import logging
import configparser
from pathlib import Path
from typing import Optional, Tuple

from local_ai_launcher.paths import (
    AI_WORKSPACE_DIR,
    LOCAL_AI_LAUNCHER_DIR,
    TMP_DIR,
    MODELS_PRESET_FILE,
    MODELS_LOCAL_OVERRIDE_FILE,
    RESOLVED_MODELS_PRESET_FILE,
)

logger = logging.getLogger("presets_resolver")


def get_candidate_override_file(override_file: Optional[Path] = None) -> Optional[Path]:
    """Locate local override file if present."""
    if override_file is not None:
        p = Path(override_file)
        return p if p.is_file() else None
    
    for candidate in [
        MODELS_LOCAL_OVERRIDE_FILE,
        LOCAL_AI_LAUNCHER_DIR / "models.local.ini",
    ]:
        if candidate and candidate.is_file():
            return candidate
    return None


def load_effective_models_parser(
    base_file: Optional[Path] = None,
    override_file: Optional[Path] = None,
) -> configparser.ConfigParser:
    """
    Loads a ConfigParser containing the base preset merged with any local override file.
    Gracefully handles any formatting/syntax errors in the override file by logging a warning
    and falling back to the base preset without crashing or losing fidelity.
    """
    base = Path(base_file) if base_file else MODELS_PRESET_FILE
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    
    if base.is_file():
        try:
            parser.read(str(base), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Error reading base models preset {base}: {e}")

    actual_override = get_candidate_override_file(override_file)
    if actual_override:
        try:
            over_parser = configparser.ConfigParser(strict=False, interpolation=None)
            over_parser.read(str(actual_override), encoding="utf-8")
            
            for section in over_parser.sections():
                if not parser.has_section(section):
                    parser.add_section(section)
                for opt, val in over_parser.items(section):
                    parser.set(section, opt, val)
            logger.info(f"[presets] Successfully applied local override from {actual_override.name}")
        except Exception as e:
            logger.warning(
                f"[presets] Failed to parse local override from {actual_override}: {e}. "
                f"Falling back safely to base preset {base.name} without overrides."
            )
            
    return parser


# Keys recognized in models.ini for launcher / shim metadata but NOT recognized by llama-server CLI arguments.
UNRECOGNIZED_LLAMA_SERVER_KEYS = {
    "engine",
    "mtp",
    "kv-stream-arena-mib",
    "kv-stream-auxiliary-layers",
}

NON_LLAMA_ENGINES = {
    "strata",
    "tabbyapi-exl3",
    "freetoken-nvfp4",
}


def needs_llama_sanitization(parser: configparser.ConfigParser) -> bool:
    """Checks if the parser contains any sections or keys unsupported by standard llama-server.exe."""
    for section in parser.sections():
        engine = parser.get(section, "engine", fallback="").lower().strip()
        if engine in NON_LLAMA_ENGINES or (section != "*" and "strata" in section.lower() and engine == "strata"):
            return True
        spec_type = parser.get(section, "spec-type", fallback="").lower().strip()
        model_str = parser.get(section, "model", fallback="").strip()
        if spec_type == "draft-mtp" and "-mtp" not in model_str.lower() and "deepseek" not in model_str.lower():
            return True
        for opt in parser.options(section):
            opt_lower = opt.lower().strip()
            if opt_lower in UNRECOGNIZED_LLAMA_SERVER_KEYS or opt_lower.startswith("kv-stream-"):
                return True
    return False


def sanitize_parser_for_llama_server(parser: configparser.ConfigParser) -> configparser.ConfigParser:
    """
    Produces a sanitized ConfigParser strictly conforming to llama-server.exe argument
    specifications, stripping unsupported metadata keys and non-llama backend sections.
    Auto-resolves -mtp model filenames when draft-mtp is requested, or strips draft-mtp
    if the model lacks native MTP layers to prevent fatal llama-server aborts.
    """
    clean_parser = configparser.ConfigParser(strict=False, interpolation=None)
    for section in parser.sections():
        engine = parser.get(section, "engine", fallback="").lower().strip()
        if engine in NON_LLAMA_ENGINES or (section != "*" and "strata" in section.lower() and engine == "strata"):
            continue

        clean_parser.add_section(section)

        model_str = parser.get(section, "model", fallback="").strip()
        spec_type = parser.get(section, "spec-type", fallback="").strip().lower()
        has_mtp_support = False
        resolved_model_str = model_str

        if "-mtp" in model_str.lower() or "deepseek" in model_str.lower():
            has_mtp_support = True
        elif model_str:
            p = Path(model_str)
            if not p.is_absolute():
                p = AI_WORKSPACE_DIR / model_str
            mtp_candidate = p.with_name(p.stem + "-mtp" + p.suffix)
            if mtp_candidate.is_file():
                has_mtp_support = True
                try:
                    resolved_model_str = str(mtp_candidate.relative_to(AI_WORKSPACE_DIR)).replace("\\", "/")
                except Exception:
                    resolved_model_str = str(mtp_candidate).replace("\\", "/")

        had_streaming_arena = any(opt.lower().strip().startswith("kv-stream-") for opt in parser.options(section))
        for opt, val in parser.items(section):
            opt_lower = opt.lower().strip()
            if opt_lower in UNRECOGNIZED_LLAMA_SERVER_KEYS or opt_lower.startswith("kv-stream-"):
                continue
            if spec_type == "draft-mtp" and not has_mtp_support and (
                opt_lower.startswith("spec-") or opt_lower.startswith("cache-type-k-draft") or opt_lower.startswith("cache-type-v-draft")
            ):
                continue
            if opt_lower == "model" and has_mtp_support:
                val = resolved_model_str
            clean_parser.set(section, opt, val)
    return clean_parser


def resolve_effective_models_preset(
    base_file: Optional[Path] = None,
    override_file: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """
    Resolves the effective INI preset file path for llama-server.
    
    - If NO local override exists and base preset is already clean: returns (base_file, False). Zero overhead.
    - If a local override exists OR base preset contains custom metadata keys unsupported by
      llama-server: merges base + local, sanitizes unrecognized keys (e.g. engine, mtp, kv-stream-*),
      writes an atomic snapshot to TMP_DIR / 'resolved_models.ini', and returns (resolved_path, is_overridden).
    """
    base = Path(base_file) if base_file else MODELS_PRESET_FILE
    actual_override = get_candidate_override_file(override_file)
    is_overridden = actual_override is not None

    parser = load_effective_models_parser(base, actual_override)
    must_sanitize = needs_llama_sanitization(parser)

    if not is_overridden and not must_sanitize:
        return base, False

    try:
        clean_parser = sanitize_parser_for_llama_server(parser) if must_sanitize else parser
        target = RESOLVED_MODELS_PRESET_FILE
        target.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_p = tempfile.mkstemp(
            dir=str(target.parent),
            prefix="resolved_models_",
            suffix=".ini",
            text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            clean_parser.write(f)
        os.replace(tmp_p, target)
        logger.info(f"[presets] Generated consolidated models preset at {target}")
        return target, is_overridden
    except Exception as e:
        logger.warning(
            f"[presets] Failed to generate resolved models preset from {actual_override}: {e}. "
            f"Falling back to base preset {base}."
        )
        return base, is_overridden


__all__ = [
    "get_candidate_override_file",
    "load_effective_models_parser",
    "resolve_effective_models_preset",
    "sanitize_parser_for_llama_server",
    "needs_llama_sanitization",
    "UNRECOGNIZED_LLAMA_SERVER_KEYS",
    "NON_LLAMA_ENGINES",
]
