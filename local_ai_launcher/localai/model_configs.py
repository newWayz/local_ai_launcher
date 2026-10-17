import os
import sys
import re
import yaml
import logging
import hashlib
import configparser
from pathlib import Path
from typing import Dict, List, Any, Optional

from local_ai_launcher.config import (
    AI_WORKSPACE_DIR,
    MODELS_PRESET_FILE,
    FREETOKENS_MODELS_PRESET_FILE,
    STRATA_MODELS_PRESET_FILE,
    LOCALAI_MODELS_DIR,
    LOCAL_AI_LAUNCHER_DIR,
    config
)
from local_ai_launcher.models import scan_model_directories, get_model_specs

logger = logging.getLogger("localai_model_configs")


def sanitize_filename(name: str) -> str:
    """Make string safe for yaml filename."""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def normalize_portable_path(raw_path: str) -> str:
    """
    Normalizes a model path so that it is always a clean relative POSIX path,
    guaranteeing complete portability regardless of workspace drive or directory.
    """
    path_str = str(raw_path).replace("\\", "/")
    try:
        p = Path(path_str)
        if p.is_absolute():
            # If path is within the workspace, make it relative to AI_WORKSPACE_DIR
            try:
                rel = p.resolve().relative_to(AI_WORKSPACE_DIR.resolve())
                return rel.as_posix()
            except ValueError:
                pass
    except Exception:
        pass
    return path_str


def get_source_manifest_hash() -> str:
    """
    Computes a lightweight checksum based on modification timestamps and sizes
    of models.ini, freetokens_models.ini, and models/exl3/ directory tree (including
    all tabby_config.yml and sampler_config.yml files).
    """
    hasher = hashlib.md5()
    # 0. Active hardware profile
    active_profile = getattr(config, "hardware_profile", "blackwell_16gb")
    hasher.update(f"profile:{active_profile}".encode("utf-8"))

    # 1. models.ini
    if MODELS_PRESET_FILE.exists():
        stat = MODELS_PRESET_FILE.stat()
        hasher.update(f"models.ini:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))

    # 1b. models.local.ini (layered override if present)
    from local_ai_launcher.presets import get_candidate_override_file
    override_file = get_candidate_override_file()
    if override_file and override_file.is_file():
        stat = override_file.stat()
        hasher.update(f"models.local.ini:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))

    # 2. freetokens_models.ini
    if FREETOKENS_MODELS_PRESET_FILE.exists():
        stat = FREETOKENS_MODELS_PRESET_FILE.stat()
        hasher.update(f"ft_models.ini:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))

    # 2b. strata_models.json, strata_models.local.json, & strata_settings.json
    if STRATA_MODELS_PRESET_FILE.exists():
        stat = STRATA_MODELS_PRESET_FILE.stat()
        hasher.update(f"strata_models.json:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))
    strata_local = LOCAL_AI_LAUNCHER_DIR / "strata_models.local.json"
    if strata_local.exists():
        stat = strata_local.stat()
        hasher.update(f"strata_local:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))
    strata_settings_file = LOCAL_AI_LAUNCHER_DIR / "strata_settings.json"
    if strata_settings_file.exists():
        stat = strata_settings_file.stat()
        hasher.update(f"strata_settings:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))
    # 3. models/exl3 directories and config files
    exl3_dir = AI_WORKSPACE_DIR / "models" / "exl3"
    if exl3_dir.exists():
        for d in sorted(exl3_dir.rglob("*")):
            try:
                stat = d.stat()
                if d.is_dir():
                    hasher.update(f"exl3_d_{d.name}:{stat.st_mtime}".encode("utf-8"))
                elif d.suffix.lower() in (".yml", ".yaml", ".safetensors"):
                    hasher.update(f"exl3_f_{d.name}:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))
            except Exception:
                pass
    # 4. models/llama-server directories and .gguf files
    llama_dir = AI_WORKSPACE_DIR / "models" / "llama-server"
    if llama_dir.exists():
        for d in sorted(llama_dir.rglob("*")):
            try:
                stat = d.stat()
                if d.is_dir():
                    hasher.update(f"llama_d_{d.name}:{stat.st_mtime}".encode("utf-8"))
                elif d.suffix.lower() == ".gguf":
                    hasher.update(f"llama_f_{d.name}:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))
            except Exception:
                pass
    return hasher.hexdigest()


def generate_localai_yaml(
    name: str,
    backend: str,
    model_path: str,
    context_size: int = 32768,
    gpu_layers: int = -1,
    parameters: Optional[Dict[str, Any]] = None,
    sampling: Optional[Dict[str, Any]] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    engine_args: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates standard LocalAI model configuration dictionary with portable paths and full settings fidelity."""
    params = parameters or {}
    params["model"] = normalize_portable_path(model_path)

    try:
        context_size = max(512, int(context_size))
    except (ValueError, TypeError):
        context_size = 32768

    try:
        gpu_layers = int(gpu_layers)
        if gpu_layers < -1:
            gpu_layers = -1
    except (ValueError, TypeError):
        gpu_layers = -1

    doc = {
        "name": name,
        "backend": backend,
        "context_size": context_size,
        "gpu_layers": gpu_layers,
        "parameters": params,
    }

    if sampling:
        doc["sampling"] = sampling
    if reasoning:
        doc["reasoning"] = reasoning
    if engine_args:
        doc["engine_args"] = engine_args

    if overrides:
        doc.update(overrides)

    return doc


def generate_all_model_configs(target_dir: Optional[Path] = None, force: bool = False) -> List[Path]:
    """
    Parses models.ini, freetokens_models.ini, and models/exl3/ to generate
    clean, unified LocalAI YAML configuration files with 100% settings fidelity.

    Smart Update Strategy:
    - Computes manifest hash of source configs & directories.
    - If unchanged and force=False, returns cached configs instantly.
    - If changed, generates updated configs and automatically prunes stale/deleted model files.
    """
    out_dir = target_dir or LOCALAI_MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    hash_file = out_dir / ".source_manifest.hash"
    current_hash = get_source_manifest_hash()

    if not force and hash_file.exists():
        try:
            stored_hash = hash_file.read_text(encoding="utf-8").strip()
            existing_yamls = list(out_dir.glob("*.yaml"))
            if stored_hash == current_hash and len(existing_yamls) > 0:
                logger.debug("LocalAI model configurations are up to date (manifest hash match).")
                return existing_yamls
        except Exception:
            pass

    generated_files = []

    # 1. Parse models.ini & local overrides (GGUF models - Blackwell & standard llama-cpp)
    from local_ai_launcher.presets import load_effective_models_parser
    parser = load_effective_models_parser()
    if parser.sections():
        try:
            global_defaults = {}
            if parser.has_section("*"):
                global_defaults = dict(parser.items("*"))
            elif parser.has_section("default"):
                global_defaults = dict(parser.items("default"))

            for sec in parser.sections():
                if sec.strip() == "*" or sec.lower() == "default":
                    continue
                # Merge [*] global defaults; per-model section explicitly overwrites [*]
                sec_data = dict(global_defaults)
                sec_data.update(dict(parser.items(sec)))

                model_file = sec_data.get("model", "")
                if not model_file:
                    continue

                try:
                    ctx_size = max(512, int(sec_data.get("ctx-size", "131072")))
                except (ValueError, TypeError):
                    ctx_size = 131072

                try:
                    ngl = int(sec_data.get("ngl", sec_data.get("n-gpu-layers", "-1")))
                    if ngl < -1:
                        ngl = -1
                except (ValueError, TypeError):
                    ngl = -1

                try:
                    threads = max(1, int(sec_data.get("threads", "16")))
                except (ValueError, TypeError):
                    threads = 16

                try:
                    threads_batch = max(1, int(sec_data.get("threads-batch", str(threads))))
                except (ValueError, TypeError):
                    threads_batch = threads

                try:
                    batch_size = max(1, int(sec_data.get("batch-size", "4096")))
                except (ValueError, TypeError):
                    batch_size = 4096

                try:
                    ubatch_size = max(1, int(sec_data.get("ubatch-size", "4096")))
                except (ValueError, TypeError):
                    ubatch_size = 4096

                flash_attn = sec_data.get("flash-attn", "on")
                try:
                    cache_reuse = max(0, int(sec_data.get("cache-reuse", "256")))
                except (ValueError, TypeError):
                    cache_reuse = 256

                ctk = sec_data.get("ctk", "q8_0")
                ctv = sec_data.get("ctv", "q8_0")
                load_mode = sec_data.get("load-mode", "").strip().lower()
                no_mmap_flag = sec_data.get("no-mmap", "False").lower() in ("true", "1", "yes")
                # load-mode = none is the modern llama.cpp global preset flag to disable mmap
                mmap_enabled = False if (load_mode == "none" or no_mmap_flag) else (load_mode == "mmap" or sec_data.get("mmap", "False").lower() in ("true", "1", "yes"))

                is_strata = (
                    "strata" in sec.lower() or
                    sec_data.get("engine", "").lower() == "strata" or
                    sec_data.get("backend", "").lower() == "strata"
                )
                is_blackwell = (
                    "blackwell" in sec.lower() or 
                    "blackwell" in model_file.lower() or
                    any("blackwell" in a.lower() for a in sec_data.get("alias", "").split(",")) or
                    sec_data.get("engine", "").lower() in ("blackwell", "huang", "bruno") or
                    "huang" in sec.lower() or
                    "bruno" in sec.lower()
                )
                if is_strata:
                    backend = "strata"
                elif is_blackwell:
                    engine_opt = sec_data.get("engine", "").lower().strip()
                    if engine_opt == "huang":
                        backend = "huang-llama"
                    elif engine_opt == "bruno":
                        backend = "bruno-llama"
                    else:
                        backend = "blackwell-llama"
                else:
                    backend = "llama-cpp"

                is_bruno_target = (
                    backend == "bruno-llama" or
                    (backend == "blackwell-llama" and getattr(config, "blackwell_engine", "huang") == "bruno" and not ("huang" in sec.lower() or "ascii" in sec.lower() or "gsq" in sec.lower()))
                )
                if is_bruno_target:
                    from local_ai_launcher.backends.bruno_builder import resolve_bruno_kv_cache_types
                    ctk, ctv = resolve_bruno_kv_cache_types(ctk, ctv)

                params: Dict[str, Any] = {
                    "threads": threads,
                    "threads_batch": threads_batch,
                    "batch_size": batch_size,
                    "ubatch_size": ubatch_size,
                    "flash_attn": flash_attn,
                    "cache_reuse": cache_reuse,
                    "ctk": ctk,
                    "ctv": ctv,
                }

                if "mmproj" in sec_data:
                    params["mmproj"] = normalize_portable_path(sec_data["mmproj"])
                if "mtp" in sec_data:
                    params["mtp"] = normalize_portable_path(sec_data["mtp"])
                if is_strata:
                    params["mmap_experts"] = getattr(config, "strata_mmap_experts", True)
                if "family" in sec_data:
                    params["family"] = sec_data["family"]
                if "chat-template-file" in sec_data:
                    params["chat_template_file"] = normalize_portable_path(sec_data["chat-template-file"])
                if "jinja" in sec_data:
                    val = str(sec_data["jinja"]).lower().strip()
                    params["jinja"] = val in ("on", "true", "yes", "1")
                if "model-draft" in sec_data:
                    params["model_draft"] = normalize_portable_path(sec_data["model-draft"])
                if "cache-type-k-draft" in sec_data:
                    params["cache_type_k_draft"] = sec_data["cache-type-k-draft"]
                if "cache-type-v-draft" in sec_data:
                    params["cache_type_v_draft"] = sec_data["cache-type-v-draft"]
                if is_bruno_target and ("cache_type_k_draft" in params or "cache_type_v_draft" in params):
                    from local_ai_launcher.backends.bruno_builder import resolve_bruno_kv_cache_types
                    dk, dv = resolve_bruno_kv_cache_types(
                        params.get("cache_type_k_draft", "q4_0"),
                        params.get("cache_type_v_draft", "q4_0")
                    )
                    params["cache_type_k_draft"] = dk
                    params["cache_type_v_draft"] = dv
                if "ctx-checkpoints" in sec_data:
                    try:
                        params["ctx_checkpoints"] = max(0, int(sec_data["ctx-checkpoints"]))
                    except (ValueError, TypeError):
                        pass
                if "spec-type" in sec_data:
                    params["spec_type"] = sec_data["spec-type"]
                if "spec-draft-n-max" in sec_data:
                    try:
                        params["spec_draft_n_max"] = max(1, int(sec_data["spec-draft-n-max"]))
                    except (ValueError, TypeError):
                        pass
                if "spec-draft-p-min" in sec_data:
                    try:
                        params["spec_draft_p_min"] = max(0.0, min(1.0, float(sec_data["spec-draft-p-min"])))
                    except (ValueError, TypeError):
                        pass
                if "spec-draft-p-split" in sec_data:
                    try:
                        params["spec_draft_p_split"] = max(0.0, min(1.0, float(sec_data["spec-draft-p-split"])))
                    except (ValueError, TypeError):
                        pass
                if "n-cpu-moe" in sec_data:
                    try:
                        params["n_cpu_moe"] = max(0, int(sec_data["n-cpu-moe"]))
                    except (ValueError, TypeError):
                        pass
                if "seed" in sec_data:
                    try:
                        params["seed"] = max(0, int(sec_data["seed"]))
                    except (ValueError, TypeError):
                        pass
                if "parallel" in sec_data:
                    try:
                        params["parallel"] = max(1, int(sec_data["parallel"]))
                    except (ValueError, TypeError):
                        pass
                if "kv-stream-auxiliary-layers" in sec_data:
                    try:
                        params["kv_stream_auxiliary_layers"] = max(0, int(sec_data["kv-stream-auxiliary-layers"]))
                    except (ValueError, TypeError):
                        pass
                if "kv-stream-arena-mib" in sec_data:
                    try:
                        params["kv_stream_arena_mib"] = max(128, int(sec_data["kv-stream-arena-mib"]))
                    except (ValueError, TypeError):
                        pass
                if "no-mmproj" in sec_data:
                    val = str(sec_data["no-mmproj"]).lower().strip()
                    params["no_mmproj"] = val in ("on", "true", "yes", "1")

                # Sampling block
                sampling: Dict[str, Any] = {}
                if "temp" in sec_data or "temperature" in sec_data:
                    try:
                        sampling["temperature"] = max(0.0, min(2.0, float(sec_data.get("temp", sec_data.get("temperature")))))
                    except (ValueError, TypeError):
                        pass
                if "top-p" in sec_data or "top_p" in sec_data:
                    try:
                        sampling["top_p"] = max(0.0, min(1.0, float(sec_data.get("top-p", sec_data.get("top_p")))))
                    except (ValueError, TypeError):
                        pass
                if "top-k" in sec_data or "top_k" in sec_data:
                    try:
                        sampling["top_k"] = max(0, int(sec_data.get("top-k", sec_data.get("top_k"))))
                    except (ValueError, TypeError):
                        pass
                if "min-p" in sec_data or "min_p" in sec_data:
                    try:
                        sampling["min_p"] = max(0.0, min(1.0, float(sec_data.get("min-p", sec_data.get("min_p")))))
                    except (ValueError, TypeError):
                        pass
                if "presence-penalty" in sec_data or "presence_penalty" in sec_data:
                    try:
                        sampling["presence_penalty"] = max(-2.0, min(2.0, float(sec_data.get("presence-penalty", sec_data.get("presence_penalty")))))
                    except (ValueError, TypeError):
                        pass
                if "repeat-penalty" in sec_data or "repetition-penalty" in sec_data or "repeat_penalty" in sec_data or "repetition_penalty" in sec_data:
                    try:
                        sampling["repetition_penalty"] = max(0.0, float(sec_data.get("repeat-penalty", sec_data.get("repetition-penalty", sec_data.get("repeat_penalty", sec_data.get("repetition_penalty"))))))
                    except (ValueError, TypeError):
                        pass
                if "seed" in sec_data:
                    try:
                        sampling["seed"] = max(0, int(sec_data["seed"]))
                    except (ValueError, TypeError):
                        pass

                # Reasoning block
                reasoning: Dict[str, Any] = {}
                if "reasoning" in sec_data:
                    reasoning["enabled"] = sec_data["reasoning"].lower() in ("on", "true", "1", "yes")
                if "reasoning-format" in sec_data:
                    reasoning["format"] = sec_data["reasoning-format"]
                if "reasoning-preserve" in sec_data:
                    reasoning["preserve"] = sec_data["reasoning-preserve"].lower() in ("true", "1", "yes", "on")
                if "reasoning-budget" in sec_data:
                    try:
                        reasoning["budget_tokens"] = max(0, int(sec_data["reasoning-budget"]))
                    except (ValueError, TypeError):
                        pass
                if "reasoning-budget-message" in sec_data:
                    reasoning["budget_message"] = sec_data["reasoning-budget-message"].strip('"\'')
                if "reasoning-effort" in sec_data:
                    reasoning["effort"] = sec_data["reasoning-effort"]

                yaml_doc = generate_localai_yaml(
                    name=sec,
                    backend=backend,
                    model_path=model_file,
                    context_size=ctx_size,
                    gpu_layers=ngl,
                    parameters=params,
                    sampling=sampling,
                    reasoning=reasoning,
                    engine_args=sec_data,
                    overrides={"mmap": mmap_enabled}
                )

                file_path = out_dir / f"{sanitize_filename(sec)}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(yaml_doc, f, default_flow_style=False, sort_keys=False)
                generated_files.append(file_path)

                # Also generate alias files
                aliases = [a.strip() for a in sec_data.get("alias", "").split(",") if a.strip()]
                for alias in aliases:
                    if alias != sec:
                        alias_doc = dict(yaml_doc)
                        alias_doc["name"] = alias
                        alias_path = out_dir / f"{sanitize_filename(alias)}.yaml"
                        with open(alias_path, "w", encoding="utf-8") as f:
                            yaml.dump(alias_doc, f, default_flow_style=False, sort_keys=False)
                        generated_files.append(alias_path)
        except Exception as e:
            logger.warning(f"Failed to generate configs from {MODELS_PRESET_FILE}: {e}")

    # 2. Parse freetokens_models.ini (NVFP4 models -> freetoken-nvfp4)
    if FREETOKENS_MODELS_PRESET_FILE.exists():
        ft_parser = configparser.ConfigParser(strict=False)
        try:
            ft_parser.read(str(FREETOKENS_MODELS_PRESET_FILE), encoding="utf-8")
            for sec in ft_parser.sections():
                if sec.strip() == "*" or sec.lower() == "default":
                    continue
                sec_data = dict(ft_parser.items(sec))
                try:
                    max_seq_len = max(512, int(sec_data.get("max-seq-len-override", "8192")))
                except (ValueError, TypeError):
                    max_seq_len = 8192

                try:
                    kv_reserve = max(0, int(sec_data.get("kv-reserve-tokens", str(max_seq_len))))
                except (ValueError, TypeError):
                    kv_reserve = max_seq_len

                ctx_size = max(512, min(max_seq_len, kv_reserve))

                clean_name = sec
                if clean_name.startswith("freetoken/") or clean_name.startswith("freetoken\\"):
                    clean_name = clean_name[10:].replace("\\", "/")

                try:
                    mem_ratio = max(0.01, min(1.0, float(sec_data.get("memory-ratio", "0.88"))))
                except (ValueError, TypeError):
                    mem_ratio = 0.88

                try:
                    max_prefill = max(1, int(sec_data.get("max-prefill-length", "4096")))
                except (ValueError, TypeError):
                    max_prefill = 4096

                try:
                    cg_max_bs = max(1, int(sec_data.get("cuda-graph-max-bs", "1")))
                except (ValueError, TypeError):
                    cg_max_bs = 1

                params = {
                    "max_seq_len_override": max_seq_len,
                    "kv_reserve_tokens": kv_reserve,
                    "memory_ratio": mem_ratio,
                    "max_prefill_length": max_prefill,
                    "cuda_graph_max_bs": cg_max_bs,
                    "cache_type": sec_data.get("cache-type", "radix"),
                    "enable_cache_report": sec_data.get("enable-cache-report", "true").lower() in ("true", "1", "yes"),
                    "enable_special_token_ckpt": sec_data.get("enable-special-token-ckpt", "true").lower() in ("true", "1", "yes"),
                    "sampling_defaults": sec_data.get("sampling-defaults", "model"),
                    "tool_call_parser": sec_data.get("tool-call-parser", "qwen3_coder"),
                    "moe_backend": sec_data.get("moe-backend", "offload"),
                    "moe_prefill_hit_d2d": sec_data.get("moe-prefill-hit-d2d", "true").lower() in ("true", "1", "yes"),
                    "expert_load": sec_data.get("expert-load", "parallel"),
                    "nvfp4_backend": sec_data.get("nvfp4-backend", "auto"),
                    "moe_cache_auto": sec_data.get("moe-cache-auto", "true").lower() in ("true", "1", "yes"),
                    "reasoning_parser": sec_data.get("reasoning-parser", "qwen3"),
                }

                reasoning = {
                    "parser": sec_data.get("reasoning-parser", "qwen3"),
                    "enabled": sec_data.get("reasoning-parser", "off").lower() != "off"
                }

                yaml_doc = generate_localai_yaml(
                    name=sec,
                    backend="freetoken-nvfp4",
                    model_path=clean_name,
                    context_size=ctx_size,
                    gpu_layers=-1,
                    parameters=params,
                    reasoning=reasoning,
                    engine_args=sec_data
                )

                file_path = out_dir / f"{sanitize_filename(sec)}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(yaml_doc, f, default_flow_style=False, sort_keys=False)
                generated_files.append(file_path)

                aliases = [a.strip() for a in sec_data.get("alias", "").split(",") if a.strip()]
                for alias in aliases:
                    if alias != sec:
                        alias_doc = dict(yaml_doc)
                        alias_doc["name"] = alias
                        alias_path = out_dir / f"{sanitize_filename(alias)}.yaml"
                        with open(alias_path, "w", encoding="utf-8") as f:
                            yaml.dump(alias_doc, f, default_flow_style=False, sort_keys=False)
                        generated_files.append(alias_path)
        except Exception as e:
            logger.warning(f"Failed to generate configs from {FREETOKENS_MODELS_PRESET_FILE}: {e}")

    # 3. Scan models/exl3/ (EXL3 models -> tabbyapi-exl3)
    exl3_dir = AI_WORKSPACE_DIR / "models" / "exl3"
    if exl3_dir.exists():
        try:
            for model_dir in scan_model_directories(exl3_dir):
                if any(model_dir.glob("*.safetensors")):
                    rel_name = model_dir.relative_to(exl3_dir).as_posix()
                    
                    # Search tabby_config.yml in model_dir or parent dir
                    tabby_conf = model_dir / "tabby_config.yml"
                    if not tabby_conf.exists() and model_dir.parent.exists():
                        parent_tc = model_dir.parent / "tabby_config.yml"
                        if parent_tc.exists():
                            tabby_conf = parent_tc

                    t_data = {}
                    if tabby_conf.exists():
                        try:
                            t_data = yaml.safe_load(tabby_conf.read_text(encoding="utf-8")) or {}
                        except Exception as e:
                            logger.warning(f"Failed reading {tabby_conf}: {e}")

                    m_opts = t_data.get("model", {}) if isinstance(t_data, dict) else {}
                    try:
                        max_seq_len = max(512, int(m_opts.get("max_seq_len", 131072)))
                    except (ValueError, TypeError):
                        max_seq_len = 131072

                    try:
                        cache_size = max(512, int(m_opts.get("cache_size", max_seq_len)))
                    except (ValueError, TypeError):
                        cache_size = max_seq_len

                    cache_mode = m_opts.get("cache_mode", "8,8")
                    vision = bool(m_opts.get("vision", False))
                    prompt_tmpl = m_opts.get("prompt_template", "")
                    if prompt_tmpl:
                        prompt_tmpl = normalize_portable_path(prompt_tmpl)
                    jinja_templates = bool(m_opts.get("jinja_templates", True))
                    tool_format = m_opts.get("tool_format", "qwen3_coder")
                    tool_calls_in_reasoning = bool(m_opts.get("tool_calls_in_reasoning", True))
                    draft_model = m_opts.get("draft_model", {})

                    try:
                        b_toks = max(0, int(m_opts.get("reasoning_budget_tokens", 4096)))
                    except (ValueError, TypeError):
                        b_toks = 4096

                    reasoning = {
                        "enabled": bool(m_opts.get("reasoning", False)),
                        "budget_tokens": b_toks,
                        "budget_message": str(m_opts.get("reasoning_budget_message", "")).strip('"\''),
                        "template_vars_default": m_opts.get("template_vars_default", {})
                    }

                    # Search sampler_config.yml in model_dir or parent dir
                    sampler_conf = model_dir / "sampler_config.yml"
                    if not sampler_conf.exists() and model_dir.parent.exists():
                        parent_sc = model_dir.parent / "sampler_config.yml"
                        if parent_sc.exists():
                            sampler_conf = parent_sc

                    sampling: Dict[str, Any] = {}
                    if sampler_conf.exists():
                        try:
                            s_data = yaml.safe_load(sampler_conf.read_text(encoding="utf-8")) or {}
                            for key, val in s_data.items():
                                v = val.get("override", val) if isinstance(val, dict) else val
                                if key in ("temperature", "temp"):
                                    try:
                                        sampling["temperature"] = max(0.0, min(2.0, float(v)))
                                    except (ValueError, TypeError):
                                        pass
                                elif key in ("top_p", "top-p"):
                                    try:
                                        sampling["top_p"] = max(0.0, min(1.0, float(v)))
                                    except (ValueError, TypeError):
                                        pass
                                elif key in ("top_k", "top-k"):
                                    try:
                                        sampling["top_k"] = max(0, int(v))
                                    except (ValueError, TypeError):
                                        pass
                                elif key in ("min_p", "min-p"):
                                    try:
                                        sampling["min_p"] = max(0.0, min(1.0, float(v)))
                                    except (ValueError, TypeError):
                                        pass
                                elif key in ("presence_penalty", "presence-penalty"):
                                    try:
                                        sampling["presence_penalty"] = max(-2.0, min(2.0, float(v)))
                                    except (ValueError, TypeError):
                                        pass
                                elif key in ("repetition_penalty", "repeat_penalty", "repeat-penalty", "repetition-penalty"):
                                    try:
                                        sampling["repetition_penalty"] = max(0.0, float(v))
                                    except (ValueError, TypeError):
                                        pass
                        except Exception as e:
                            logger.warning(f"Failed reading {sampler_conf}: {e}")

                    params = {
                        "model_folder": rel_name,
                        "max_seq_len": max_seq_len,
                        "cache_size": cache_size,
                        "cache_mode": cache_mode,
                        "vision": vision,
                        "prompt_template": prompt_tmpl,
                        "jinja_templates": jinja_templates,
                        "tool_format": tool_format,
                        "tool_calls_in_reasoning": tool_calls_in_reasoning,
                        "draft_model": draft_model,
                    }

                    yaml_doc = generate_localai_yaml(
                        name=model_dir.name,
                        backend="tabbyapi-exl3",
                        model_path=f"models/exl3/{rel_name}",
                        context_size=max_seq_len,
                        gpu_layers=-1,
                        parameters=params,
                        sampling=sampling,
                        reasoning=reasoning,
                        engine_args=m_opts
                    )

                    file_path = out_dir / f"{sanitize_filename(model_dir.name)}.yaml"
                    with open(file_path, "w", encoding="utf-8") as f:
                        yaml.dump(yaml_doc, f, default_flow_style=False, sort_keys=False)
                    generated_files.append(file_path)

                    # Also create alias for relative path (e.g. turboderp/Qwen3.8-27B-exl3-SC-3.0bpw-H4)
                    if rel_name != model_dir.name:
                        rel_alias_doc = dict(yaml_doc)
                        rel_alias_doc["name"] = rel_name
                        rel_alias_path = out_dir / f"{sanitize_filename(rel_name)}.yaml"
                        with open(rel_alias_path, "w", encoding="utf-8") as f:
                            yaml.dump(rel_alias_doc, f, default_flow_style=False, sort_keys=False)
                        generated_files.append(rel_alias_path)

                    # Suffix-stripped alias (e.g. Qwen3.8-27B-exl3-SC-3.0bpw-H4-V4 -> Qwen3.8-27B-exl3-SC-3.0bpw-H4)
                    base_alias = re.sub(r'[-_]v\d+$', '', model_dir.name, flags=re.IGNORECASE)
                    if base_alias != model_dir.name:
                        base_alias_doc = dict(yaml_doc)
                        base_alias_doc["name"] = base_alias
                        base_alias_path = out_dir / f"{sanitize_filename(base_alias)}.yaml"
                        with open(base_alias_path, "w", encoding="utf-8") as f:
                            yaml.dump(base_alias_doc, f, default_flow_style=False, sort_keys=False)
                        generated_files.append(base_alias_path)
                        
                        if "/" in rel_name:
                            rel_base_alias = rel_name.rsplit('/', 1)[0] + '/' + base_alias
                            rel_b_doc = dict(yaml_doc)
                            rel_b_doc["name"] = rel_base_alias
                            rel_b_path = out_dir / f"{sanitize_filename(rel_base_alias)}.yaml"
                            with open(rel_b_path, "w", encoding="utf-8") as f:
                                yaml.dump(rel_b_doc, f, default_flow_style=False, sort_keys=False)
                            generated_files.append(rel_b_path)

                    # Friendly short aliases
                    friendly_aliases = []
                    lower_name = model_dir.name.lower()
                    if "qwen" in lower_name and "exl3" in lower_name:
                        friendly_aliases.append("qwen-38-exl3")
                    if "muse" in lower_name and "exl3" in lower_name:
                        friendly_aliases.append("muse-glimmer-exl3")
                    for fa in friendly_aliases:
                        fa_doc = dict(yaml_doc)
                        fa_doc["name"] = fa
                        fa_path = out_dir / f"{sanitize_filename(fa)}.yaml"
                        with open(fa_path, "w", encoding="utf-8") as f:
                            yaml.dump(fa_doc, f, default_flow_style=False, sort_keys=False)
                        generated_files.append(fa_path)
        except Exception as e:
            logger.warning(f"Failed to generate configs from {exl3_dir}: {e}")

    # 4. Parse strata_models.json (Strata MoE models -> strata)
    try:
        from local_ai_launcher.backends.strata import load_strata_model_registry
        strata_registry = load_strata_model_registry()
        for m_name, m_spec in strata_registry.items():
            model_file = m_spec.get("model") or m_spec.get("model_path")
            if not model_file:
                continue

            try:
                ctx_size = max(512, int(m_spec.get("ctx_size", m_spec.get("ctx-size", m_spec.get("context_size", 262144)))))
            except (ValueError, TypeError):
                ctx_size = 262144

            params: Dict[str, Any] = {
                "mmap_experts": m_spec.get("mmap_experts", getattr(config, "strata_mmap_experts", True)),
            }
            if "mtp" in m_spec or "mtp_dir" in m_spec:
                params["mtp"] = normalize_portable_path(str(m_spec.get("mtp") or m_spec.get("mtp_dir")))
            if "mmproj" in m_spec:
                params["mmproj"] = normalize_portable_path(str(m_spec["mmproj"]))
            if "chat_template_file" in m_spec or "chat-template-file" in m_spec:
                params["chat_template_file"] = normalize_portable_path(str(m_spec.get("chat_template_file") or m_spec.get("chat-template-file")))
            if "jinja" in m_spec:
                val = m_spec["jinja"]
                params["jinja"] = val is True or str(val).lower().strip() in ("on", "true", "yes", "1")
            if "threads" in m_spec:
                params["threads"] = m_spec["threads"]

            # Sampling block
            sampling: Dict[str, Any] = {}
            raw_sampling = m_spec.get("sampling", {})
            if isinstance(raw_sampling, dict):
                for k, v in raw_sampling.items():
                    norm_k = k.replace("-", "_")
                    if norm_k in ("temperature", "temp"):
                        try:
                            sampling["temperature"] = max(0.0, min(2.0, float(v)))
                        except (ValueError, TypeError):
                            pass
                    elif norm_k in ("top_p",):
                        try:
                            sampling["top_p"] = max(0.0, min(1.0, float(v)))
                        except (ValueError, TypeError):
                            pass
                    elif norm_k in ("top_k",):
                        try:
                            sampling["top_k"] = max(0, int(v))
                        except (ValueError, TypeError):
                            pass
                    elif norm_k in ("min_p",):
                        try:
                            sampling["min_p"] = max(0.0, min(1.0, float(v)))
                        except (ValueError, TypeError):
                            pass
                    elif norm_k in ("presence_penalty",):
                        try:
                            sampling["presence_penalty"] = max(-2.0, min(2.0, float(v)))
                        except (ValueError, TypeError):
                            pass
                    elif norm_k in ("repetition_penalty", "repeat_penalty"):
                        try:
                            sampling["repetition_penalty"] = max(0.0, float(v))
                        except (ValueError, TypeError):
                            pass

            # Flat sampling fallback if not nested under "sampling"
            for k in ("temp", "temperature", "top-p", "top_p", "top-k", "top_k", "min-p", "min_p", "presence-penalty", "presence_penalty", "repeat-penalty", "repeat_penalty", "repetition-penalty", "repetition_penalty"):
                norm_k = k.replace("-", "_")
                if k in m_spec and norm_k not in sampling:
                    try:
                        sampling[norm_k] = float(m_spec[k])
                    except (ValueError, TypeError):
                        pass

            # Reasoning block
            reasoning: Dict[str, Any] = {}
            raw_reasoning = m_spec.get("reasoning", {})
            if isinstance(raw_reasoning, dict):
                reasoning["enabled"] = bool(raw_reasoning.get("enabled", True))
                if "effort" in raw_reasoning:
                    reasoning["effort"] = str(raw_reasoning["effort"])
                elif "reasoning_effort" in raw_reasoning:
                    reasoning["effort"] = str(raw_reasoning["reasoning_effort"])
                elif "reasoning_effort" in m_spec:
                    reasoning["effort"] = str(m_spec["reasoning_effort"])
                if "format" in raw_reasoning:
                    reasoning["format"] = str(raw_reasoning["format"])
                if "preserve" in raw_reasoning:
                    reasoning["preserve"] = bool(raw_reasoning["preserve"])
                if "budget_tokens" in raw_reasoning:
                    try:
                        reasoning["budget_tokens"] = max(0, int(raw_reasoning["budget_tokens"]))
                    except (ValueError, TypeError):
                        pass
                if "budget_message" in raw_reasoning:
                    reasoning["budget_message"] = str(raw_reasoning["budget_message"]).strip('"\'')
            elif isinstance(raw_reasoning, bool):
                reasoning["enabled"] = raw_reasoning

            yaml_doc = generate_localai_yaml(
                name=m_name,
                backend="strata",
                model_path=model_file,
                context_size=ctx_size,
                gpu_layers=-1,
                parameters=params,
                sampling=sampling,
                reasoning=reasoning,
                engine_args=m_spec
            )

            file_path = out_dir / f"{sanitize_filename(m_name)}.yaml"
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_doc, f, default_flow_style=False, sort_keys=False)
            generated_files.append(file_path)

            raw_aliases = m_spec.get("aliases") or m_spec.get("alias", [])
            if isinstance(raw_aliases, str):
                aliases = [a.strip() for a in raw_aliases.split(",") if a.strip()]
            elif isinstance(raw_aliases, list):
                aliases = [str(a).strip() for a in raw_aliases if str(a).strip()]
            else:
                aliases = []
            for alias in aliases:
                if alias != m_name:
                    alias_doc = dict(yaml_doc)
                    alias_doc["name"] = alias
                    alias_path = out_dir / f"{sanitize_filename(alias)}.yaml"
                    with open(alias_path, "w", encoding="utf-8") as f:
                        yaml.dump(alias_doc, f, default_flow_style=False, sort_keys=False)
                    generated_files.append(alias_path)
    except Exception as e:
        logger.warning(f"Failed to generate configs from strata_models.json: {e}")

    # 5. Standard Director LLM config (CPU fallback / image_vid_gen Director)
    director_path = "models/llama-server/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
    if not (AI_WORKSPACE_DIR / director_path).exists():
        director_path = "models/llama-server/gguf/Gemma-4-E4B-Q4_K_P.gguf"

    director_doc = generate_localai_yaml(
        name="gemma-4-director",
        backend="blackwell-llama",
        model_path=director_path,
        context_size=32768,
        gpu_layers=0,  # CPU-only by default to leave 16GB free for diffusion
        parameters={"threads": 8}
    )
    dir_path = out_dir / "gemma-4-director.yaml"
    with open(dir_path, "w", encoding="utf-8") as f:
        yaml.dump(director_doc, f, default_flow_style=False, sort_keys=False)
    generated_files.append(dir_path)

    # Prune stale / orphaned yaml files (e.g. models deleted or renamed in models.ini or models/exl3)
    active_names = {p.name for p in generated_files}
    pruned_count = 0
    for existing_file in out_dir.glob("*.yaml"):
        if existing_file.name not in active_names:
            try:
                existing_file.unlink()
                pruned_count += 1
                logger.info(f"  -> Pruned stale LocalAI model config: {existing_file.name}")
            except Exception as e:
                logger.debug(f"Could not prune {existing_file.name}: {e}")

    try:
        hash_file.write_text(current_hash, encoding="utf-8")
    except Exception:
        pass

    logger.info(f"Generated {len(generated_files)} LocalAI model YAML configs in {out_dir}" + (f" (pruned {pruned_count} stale configs)" if pruned_count else ""))
    return generated_files


def get_model_backend_mapping(models_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping of model_id -> dict(backend, yaml_path, config_data).
    Smartly detects on-disk modifications to models.ini or models/exl3/ and
    auto-refreshes the mapping in memory and on disk.
    """
    m_dir = models_dir or LOCALAI_MODELS_DIR
    hash_file = m_dir / ".source_manifest.hash"
    current_hash = get_source_manifest_hash()

    # If source files have changed or configs are missing, auto-regenerate smartly
    if not m_dir.exists() or not hash_file.exists() or not any(m_dir.glob("*.yaml")):
        generate_all_model_configs(target_dir=m_dir)
    else:
        try:
            stored_hash = hash_file.read_text(encoding="utf-8").strip()
            if stored_hash != current_hash:
                generate_all_model_configs(target_dir=m_dir)
        except Exception:
            generate_all_model_configs(target_dir=m_dir)

    mapping = {}
    for yml in m_dir.glob("*.yaml"):
        try:
            with open(yml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and "name" in data and "backend" in data:
                mapping[data["name"]] = {
                    "backend": data["backend"],
                    "yaml_path": yml,
                    "data": data
                }
        except Exception:
            pass
    return mapping
