import os
import sys
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

from local_ai_launcher.config import (
    AI_WORKSPACE_DIR,
    LOCAL_AI_LAUNCHER_DIR,
    STRATA_DIR,
    STRATA_ENGINE_DIR,
    STRATA_EXE,
    config,
    logger,
    prompt_choice
)

logger = logging.getLogger("strata_backend")




STRATA_SERVER_PATCH_MARKER = "# lai: chat_template"


def patch_strata_server() -> bool:
    """Apply custom chat template loading and xhigh reasoning effort support to strata/serve/server.py.
    Idempotent. Returns True when patched or already applied, False on
    upstream drift (anchors not found) - callers should warn, never abort."""
    server_py = STRATA_DIR / "serve" / "server.py"
    if not server_py.is_file():
        return False
    try:
        src = server_py.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[strata-patch] cannot read {server_py}: {e}")
        return False

    changed = False
    out = src

    if STRATA_SERVER_PATCH_MARKER not in out:
        anchor = '    tpl = tpath / "chat_template.jinja"\n'
        if out.count(anchor) == 1:
            repl = (
                '    # lai: chat_template from strata config\n'
                '    tpl_custom = cfg.get("chat_template_file") if cfg.get("use_custom_chat_template", True) else None\n'
                '    if tpl_custom and tpl_custom != "original" and Path(tpl_custom).is_file():\n'
                '        tpl = Path(tpl_custom)\n'
                '    else:\n'
                '        tpl = tpath / "chat_template.jinja"\n'
            )
            out = out.replace(anchor, repl, 1)
            changed = True

    # Permit 'xhigh' in clean_shared_defaults
    effort_anchor = '            if value not in ("none", "low", "medium", "high"):\n'
    if effort_anchor in out:
        out = out.replace(effort_anchor, '            if value not in ("none", "low", "medium", "high", "xhigh"):\n', 1)
        changed = True

    if not changed:
        return True

    try:
        import ast as _ast
        _ast.parse(out)
    except SyntaxError as e:
        logger.warning(f"[strata-patch] patched server source fails syntax check ({e}); patch skipped.")
        return False
    try:
        # Upstream repo uses LF line endings; preserve LF to prevent git diff churn.
        with open(server_py, "w", encoding="utf-8", newline="\n") as f:
            f.write(out.replace("\r\n", "\n"))
        logger.info("[strata-patch] applied custom chat template & xhigh effort support to strata serve server.py")
    except Exception as e:
        logger.warning(f"[strata-patch] cannot write {server_py}: {e}")
        return False
    return True


def check_strata() -> bool:
    """
    Ensures Strata repository and ready-made engine binary (strata.exe) exist
    inside the workspace directory (C:/aiWorkspace/strata).
    """
    if not STRATA_DIR.exists():
        logger.info(f"Strata repository not found at {STRATA_DIR}. Cloning from GitHub...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/Niko1221/Strata.git", str(STRATA_DIR)],
                cwd=str(AI_WORKSPACE_DIR),
                check=True,
                timeout=180
            )
            logger.info("  -> Strata repository cloned successfully.")
        except Exception as e:
            logger.error(f"Failed to clone Strata repository: {e}")
            return False

    # Custom chat template serve patch (portable, idempotent, non-fatal on upstream drift).
    if not patch_strata_server():
        logger.debug("  -> strata serve server patch skipped or not applied; relying on pack tokenizer sync.")

    # Check if strata.exe exists in engine/
    if not STRATA_EXE.exists():
        logger.info(f"Strata engine binary not found at {STRATA_EXE}. Fetching ready-made engine...")
        try:
            sys.path.insert(0, str(STRATA_DIR))
            import setup as strata_setup
            gpus = strata_setup.gpus()
            if not gpus:
                logger.error("No compatible GPU found by Strata setup.")
                return False
            primary_gpu = gpus[0]
            eng = strata_setup.get_prebuilt(strata_setup.PREBUILT_URL, primary_gpu, "none")
            if not eng or not (eng / "strata.exe").exists():
                logger.error("Could not obtain prebuilt Strata engine.")
                return False
            logger.info(f"  -> Prebuilt Strata engine installed at {eng}")
        except Exception as e:
            logger.error(f"Error initializing Strata engine: {e}")
            return False

    return True


def load_strata_model_registry(registry_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads Strata model presets from strata_models.json (or strata_settings.json["models"]).
    Fully decoupled from models.ini, allowing native JSON specification of models,
    shards, MTP directories, and sampling/reasoning configurations.
    Supports comment-tolerant JSON (ignoring lines beginning with # or //).
    """
    candidates = []
    if registry_file:
        candidates.append(Path(registry_file))
    candidates.extend([
        LOCAL_AI_LAUNCHER_DIR / "strata_models.local.json",
        AI_WORKSPACE_DIR / "strata_models.local.json",
        LOCAL_AI_LAUNCHER_DIR / "strata_models.json",
        AI_WORKSPACE_DIR / "strata_models.json",
        LOCAL_AI_LAUNCHER_DIR / "strata_models.json.example",
    ])
    models: Dict[str, Any] = {}

    # 1. Check dedicated strata_models JSON candidates
    for p in candidates:
        if p and p.is_file():
            try:
                raw = p.read_text(encoding="utf-8")
                clean_lines = [l for l in raw.splitlines() if not l.strip().startswith(("//", "#"))]
                data = json.loads("\n".join(clean_lines))
                if isinstance(data, dict):
                    for k, v in data.items():
                        if not k.startswith("_") and isinstance(v, dict):
                            models[k] = v
                    if models:
                        break
            except Exception as e:
                logger.debug(f"[Strata Backend] Notice reading strata models from {p}: {e}")

    # 2. Check strata_settings.json for optional embedded 'models' object
    if not models:
        try:
            settings = load_strata_settings()
            if "models" in settings and isinstance(settings["models"], dict):
                models = {k: v for k, v in settings["models"].items() if isinstance(v, dict)}
        except Exception:
            pass

    return models


def resolve_strata_model_files(model_path_str: str, mtp_path_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolves the GGUF shards, PLE table, and MTP directory from a configured model path or alias.
    First resolves against native Strata model registry (strata_models.json), with a legacy
    fallback to models.ini for backward compatibility.
    """
    res = {
        "shards": [],
        "first_shard": None,
        "ple_shard": None,
        "mtp_dir": None,
        "pack_dir": None,
        "chat_template_file": None
    }

    p = Path(model_path_str)
    if not p.is_absolute():
        p = AI_WORKSPACE_DIR / p

    target_key = model_path_str.strip().lower()

    # 1. Primary lookup: native Strata model registry (strata_models.json)
    registry = load_strata_model_registry()
    matched_entry = None
    for m_name, m_spec in registry.items():
        if m_name.lower() == target_key:
            matched_entry = m_spec
            break
        raw_aliases = m_spec.get("aliases") or m_spec.get("alias", [])
        if isinstance(raw_aliases, str):
            aliases = [a.strip().lower() for a in raw_aliases.split(",") if a.strip()]
        elif isinstance(raw_aliases, list):
            aliases = [str(a).strip().lower() for a in raw_aliases if str(a).strip()]
        else:
            aliases = []
        if target_key in aliases:
            matched_entry = m_spec
            break

        # Match by model path or filename if target_key was supplied as file or subpath
        m_model = m_spec.get("model") or m_spec.get("model_path", "")
        if m_model:
            clean_m = m_model.strip().lower().replace("\\", "/")
            target_norm = target_key.replace("\\", "/")
            if (clean_m == target_norm or
                Path(clean_m).name == Path(target_norm).name or
                target_norm.endswith(clean_m) or
                clean_m.endswith(target_norm)):
                matched_entry = m_spec
                break

    if matched_entry:
        cand_tmpl = matched_entry.get("chat_template_file") or matched_entry.get("chat_template")
        if cand_tmpl:
            res["chat_template_file"] = str(cand_tmpl)
        cand_mmproj = matched_entry.get("mmproj")
        if cand_mmproj:
            res["mmproj"] = str(cand_mmproj)
        cand_model = matched_entry.get("model") or matched_entry.get("model_path")
        if cand_model:
            cand_p = Path(cand_model)
            if not cand_p.is_absolute():
                cand_p = AI_WORKSPACE_DIR / cand_p
            if cand_p.exists():
                p = cand_p
            elif (cand_p.parent / "IQ3_S" / cand_p.name).exists():
                p = cand_p.parent / "IQ3_S" / cand_p.name
            elif cand_p.parent.name.upper() == "IQ3_S" and (cand_p.parent.parent / cand_p.name).exists():
                p = cand_p.parent.parent / cand_p.name
            else:
                p = cand_p
        if not mtp_path_str:
            cand_mtp = matched_entry.get("mtp") or matched_entry.get("mtp_dir")
            if cand_mtp:
                mtp_path_str = str(cand_mtp)

    # 2. Backward compatibility fallback: check models.ini if not found in native registry
    if not matched_entry and not p.exists():
        try:
            from local_ai_launcher.presets import load_effective_models_parser
            cp = load_effective_models_parser()
            for s in cp.sections():
                aliases = [a.strip().lower() for a in cp.get(s, "alias", fallback="").split(",") if a.strip()]
                if s.lower() == target_key or target_key in aliases:
                    cand_tmpl = cp.get(s, "chat_template_file", fallback="") or cp.get(s, "chat_template", fallback="")
                    if cand_tmpl:
                        res["chat_template_file"] = cand_tmpl.strip()
                    if cp.has_option(s, "mmproj"):
                        res["mmproj"] = cp.get(s, "mmproj").strip()
                    if cp.has_option(s, "model"):
                        cand = cp.get(s, "model").strip()
                        cand_p = Path(cand)
                        if not cand_p.is_absolute():
                            cand_p = AI_WORKSPACE_DIR / cand_p
                        if cand_p.exists():
                            p = cand_p
                    if not mtp_path_str and cp.has_option(s, "mtp"):
                        mtp_path_str = cp.get(s, "mtp").strip()
                    break
        except Exception as e:
            logger.debug(f"[strata_backend] Model alias fallback lookup notice: {e}")

    # 3. Fallback resolution for chat_template_file if model was supplied as path directly
    if not res.get("chat_template_file"):
        try:
            registry = load_strata_model_registry()
            for m_name, m_spec in registry.items():
                m_model = m_spec.get("model") or m_spec.get("model_path", "")
                if m_model and Path(m_model).name == p.name:
                    cand_tmpl = m_spec.get("chat_template_file") or m_spec.get("chat_template")
                    if cand_tmpl:
                        res["chat_template_file"] = str(cand_tmpl)
                        break
        except Exception:
            pass

    model_dir = p.parent
    shard_name = p.name

    # Discover related shards (-00001-of-*, -00002-of-*)
    all_shards = []
    if model_dir.exists():
        all_shards = sorted(list(model_dir.glob("*-0000*-of-*.gguf")))
        if not all_shards and p.exists():
            all_shards = [p]
    else:
        all_shards = [p]

    if not all_shards and p:
        all_shards = [p]

    # If first shard filename indicates multipart GGUF, infer all companion shards
    if all_shards and "-00001-of-" in all_shards[0].name:
        s1 = all_shards[0]
        import re
        m_parts = re.search(r"-00001-of-(\d{5})\.gguf$", s1.name, re.IGNORECASE)
        if m_parts:
            total_shards = int(m_parts.group(1))
            for i in range(1, total_shards + 1):
                s_name = s1.name.replace("-00001-of-", f"-{i:05d}-of-")
                expected_p = s1.parent / s_name
                if expected_p not in all_shards:
                    all_shards.append(expected_p)
            all_shards.sort()
        else:
            s2_name = s1.name.replace("-00001-of-", "-00002-of-")
            s2 = s1.parent / s2_name
            if s2 not in all_shards:
                all_shards.append(s2)

    res["shards"] = all_shards
    if all_shards:
        res["first_shard"] = all_shards[0]
        if len(all_shards) > 1:
            res["ple_shard"] = all_shards[1]
        else:
            res["ple_shard"] = all_shards[0]

    # Resolve MTP directory
    if mtp_path_str:
        mtp_p = Path(mtp_path_str)
        if not mtp_p.is_absolute():
            mtp_p = AI_WORKSPACE_DIR / mtp_p
        if mtp_p.exists():
            res["mtp_dir"] = mtp_p

    # Fallback search for strata/rt if not explicitly given or not found
    if not res["mtp_dir"]:
        candidate_rt = model_dir.parent / "strata" / "rt"
        if candidate_rt.exists():
            res["mtp_dir"] = candidate_rt
        else:
            candidate_rt2 = model_dir / "strata" / "rt"
            if candidate_rt2.exists():
                res["mtp_dir"] = candidate_rt2
            else:
                common_rt = AI_WORKSPACE_DIR / "models" / "llama-server" / "SC117" / "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF" / "strata" / "rt"
                if common_rt.exists():
                    res["mtp_dir"] = common_rt

    # Define dedicated pack directory
    pack_name = p.stem.lower().replace("-00001-of-00002", "").replace(".gguf", "")
    res["pack_dir"] = STRATA_DIR / "data" / "packs" / pack_name

    return res


def ensure_strata_pack(first_shard: Path, pack_dir: Path) -> bool:
    """
    Ensures that the Strata native expert pack exists for the given GGUF model.
    If missing, runs tools/iq_pack.py to prepare index.txt, dense.bin, and tokenizer.
    """
    if not first_shard or not first_shard.exists():
        logger.error(f"[Strata Backend] First shard does not exist: {first_shard}")
        return False

    pack_dir.mkdir(parents=True, exist_ok=True)
    native_experts = pack_dir / "native_experts.txt"
    tokenizer_vocab = pack_dir / "tokenizer" / "vocab.json"

    if native_experts.exists() and tokenizer_vocab.exists():
        logger.debug(f"[Strata Backend] Native pack already verified at {pack_dir}")
        return True

    logger.info(f"[Strata Backend] Building native Strata pack at {pack_dir} for {first_shard.name}...")
    llama_dir = STRATA_DIR / "third_party" / "llama.cpp"
    if not llama_dir.exists():
        try:
            sys.path.insert(0, str(STRATA_DIR))
            import setup as strata_setup
            strata_setup.get_llama_cpp()
        except Exception as e:
            logger.warning(f"Failed to fetch llama.cpp source: {e}")

    env = dict(os.environ)
    if llama_dir.exists():
        env["STRATA_GGUF_PY"] = str(llama_dir / "gguf-py")

    cmd = [
        sys.executable,
        str(STRATA_DIR / "tools" / "iq_pack.py"),
        "--gguf", str(first_shard),
        "--out", str(pack_dir)
    ]

    try:
        res = subprocess.run(cmd, env=env, cwd=str(STRATA_DIR), capture_output=True, text=True, check=True)
        logger.info(f"[Strata Backend] Successfully prepared pack at {pack_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[Strata Backend] Failed to build pack: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"[Strata Backend] Unexpected error running iq_pack: {e}")
        return False


def load_strata_settings(settings_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads Strata engine configuration from strata_settings.json located in local_ai_launcher.
    Allows easy maintenance and transparent adjustment of MTP spec, memory, and KV tunables.
    Supports comment-tolerant JSON (ignoring lines beginning with # or //).
    """
    is_test = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI") == "1" or "pytest" in sys.modules)
    if settings_file is None and is_test:
        return {}

    candidates = []
    if settings_file:
        candidates.append(Path(settings_file))
    candidates.extend([
        LOCAL_AI_LAUNCHER_DIR / "strata_settings.json",
        AI_WORKSPACE_DIR / "strata_settings.json"
    ])
    for p in candidates:
        if p and p.is_file():
            try:
                raw = p.read_text(encoding="utf-8")
                clean_lines = []
                for line in raw.splitlines():
                    stripped = line.strip()
                    if not stripped.startswith("//") and not stripped.startswith("#"):
                        clean_lines.append(line)
                data = json.loads("\n".join(clean_lines))
                if isinstance(data, dict):
                    # Filter out documentation helper keys starting with underscore
                    return {k: v for k, v in data.items() if not k.startswith("_")}
            except Exception as e:
                logger.warning(f"[Strata Backend] Failed to load strata settings from {p}: {e}")
                break
    return {}


def build_strata_config(
    model_name: str,
    first_shard: Path,
    ple_shard: Optional[Path],
    pack_dir: Path,
    mtp_dir: Optional[Path],
    port: int,
    max_context: int = 262144,
    kv_mode: Optional[str] = None,
    expert_cache: str = "auto",
    spec: Optional[int] = None,
    mmap_experts: Optional[bool] = None,
    resident_budget_gib: Optional[float] = None,
    settings_file: Optional[Path] = None,
    chat_template_file: Optional[str | Path] = None,
    use_custom_chat_template: Optional[bool] = None,
    kv_resident: Optional[int] = None,
    vram_reserve_mib: Optional[int] = None,
    system_ram_kv_cache: Optional[bool] = None,
    reasoning_effort: Optional[str] = None
) -> Path:
    """
    Assembles strata-<model>.json configuration for Strata's serve/server.py.
    Loads user tunables from local_ai_launcher/strata_settings.json if present.
    """
    user_settings = load_strata_settings(settings_file)

    # Resolve mmap_experts
    if mmap_experts is None:
        if "mmap_experts" in user_settings:
            mmap_experts = bool(user_settings["mmap_experts"])
        else:
            mmap_experts = getattr(config, "strata_mmap_experts", True)

    # Resolve kv_mode with validation
    valid_kv = ("int8", "q4_0", "k8v4", "fp16")
    if kv_mode is None:
        raw_kv = str(user_settings.get("kv_mode", getattr(config, "strata_kv_mode", "int8"))).lower().strip()
        kv_mode = raw_kv if raw_kv in valid_kv else "int8"
    else:
        kv_mode = str(kv_mode).lower().strip()
        if kv_mode not in valid_kv:
            kv_mode = "int8"

    # Resolve resident_budget_gib with non-negative validation
    if resident_budget_gib is None:
        raw_budget = user_settings.get("resident_budget_gib", getattr(config, "strata_resident_budget_gib", 52.0))
        try:
            resident_budget_gib = max(0.0, float(raw_budget))
        except (ValueError, TypeError):
            resident_budget_gib = 52.0
    else:
        try:
            resident_budget_gib = max(0.0, float(resident_budget_gib))
        except (ValueError, TypeError):
            resident_budget_gib = 52.0

    # Resolve spec (verify window size: must be >= 2 for native IQ packs)
    if spec is None:
        raw_spec = user_settings.get("spec", 4)
        try:
            spec = max(2, int(raw_spec))
        except (ValueError, TypeError):
            spec = 4
    else:
        try:
            spec = max(2, int(spec))
        except (ValueError, TypeError):
            spec = 4

    # Resolve spec_min_p (clamped between 0.0 and 1.0)
    raw_p = user_settings.get("spec_min_p", 0.5)
    try:
        spec_min_p = max(0.0, min(1.0, float(raw_p)))
    except (ValueError, TypeError):
        spec_min_p = 0.5

    # Optional drafter & window tunables
    suffix_draft = user_settings.get("suffix_draft")
    if suffix_draft is not None:
        try:
            suffix_draft = max(0, int(suffix_draft))
        except (ValueError, TypeError):
            suffix_draft = None

    coupled_draft = user_settings.get("coupled_draft")
    mtp_window = user_settings.get("mtp_window")
    if mtp_window is not None:
        try:
            mtp_window = max(0, int(mtp_window))
        except (ValueError, TypeError):
            mtp_window = None

    mtp_max_t = user_settings.get("mtp_max_t")
    if mtp_max_t is not None:
        try:
            mtp_max_t = max(0, int(mtp_max_t))
        except (ValueError, TypeError):
            mtp_max_t = None

    if expert_cache == "auto" and "expert_cache" in user_settings:
        expert_cache = str(user_settings["expert_cache"])

    prefill_val = str(user_settings.get("prefill", "auto"))

    try:
        max_context = max(512, int(max_context))
    except (ValueError, TypeError):
        max_context = 262144

    clean_tag = model_name.replace("/", "_").replace("\\", "_").lower()

    # Destination paths: determine if this is an automated test execution
    is_test = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI") == "1" or "pytest" in sys.modules)
    is_dummy_model = clean_tag.startswith("test") or "dummy" in clean_tag

    launcher_cfg_path = LOCAL_AI_LAUNCHER_DIR / f"strata-{clean_tag}.json"
    strata_cfg_path = STRATA_DIR / f"strata-{clean_tag}.json"

    # Avoid polluting LOCAL_AI_LAUNCHER_DIR with throwaway test fixtures
    target_output_path = launcher_cfg_path if not (is_test and is_dummy_model) else strata_cfg_path

    profile_file = STRATA_DIR / "data" / "expert-profile.bin"
    engine_exe = STRATA_EXE.as_posix()
    args = [
        "--pack", pack_dir.as_posix(),
        "--native", first_shard.as_posix(),
        "--expert-profile", profile_file.as_posix(),
        "--expert-cache", expert_cache,
        "--prefill", prefill_val,
        "--spec", str(spec),
        "--spec-min-p", str(spec_min_p),
        "--max-context", str(max_context)
    ]

    if ple_shard and ple_shard.exists():
        args += ["--ple-gguf", ple_shard.as_posix()]

    if mtp_dir and mtp_dir.exists():
        args += ["--mtp", mtp_dir.as_posix()]

    if kv_mode:
        args += ["--kv", kv_mode]

    if suffix_draft is not None:
        args += ["--suffix-draft", str(suffix_draft)]

    if coupled_draft is True:
        args += ["--coupled-draft"]
    elif coupled_draft is False:
        args += ["--no-coupled-draft"]

    if mtp_window:
        args += ["--mtp-window", str(mtp_window)]

    if mtp_max_t:
        args += ["--mtp-max-t", str(mtp_max_t)]

    # Resolve system_ram_kv_cache mode toggle
    if system_ram_kv_cache is None:
        raw_toggle = (
            user_settings.get("system_ram_kv_cache")
            or user_settings.get("ram_only_kv_cache")
            or user_settings.get("system_ram_only_kv_cache")
            or user_settings.get("ram_kv_cache")
            or getattr(config, "strata_system_ram_kv_cache", False)
        )
        system_ram_kv_cache = bool(raw_toggle)
    else:
        system_ram_kv_cache = bool(system_ram_kv_cache)

    # Resolve kv_resident (KV streaming window)
    if kv_resident is None:
        raw_kv_res = user_settings.get("kv_resident", getattr(config, "strata_kv_resident", None))
        if raw_kv_res is not None:
            try:
                kv_resident = int(raw_kv_res)
            except (ValueError, TypeError):
                kv_resident = None

    if system_ram_kv_cache:
        # System RAM only KV cache mode enabled:
        # Stream KV cache with minimal resident window (min 20480 cells per QSA layer)
        active_kv_resident = max(20480, kv_resident if (kv_resident and kv_resident > 0) else 20480)
        args += ["--kv-resident", str(active_kv_resident)]
        logger.info(
            f"[Strata Backend] System RAM KV cache mode ENABLED: "
            f"--kv-resident {active_kv_resident} (streaming in host RAM; max VRAM freed for experts)"
        )
    elif kv_resident is not None and kv_resident > 0:
        clamped_kv_res = max(20480, kv_resident)
        args += ["--kv-resident", str(clamped_kv_res)]

    # Resolve vram_reserve_mib (WDDM headroom buffer)
    if vram_reserve_mib is None:
        raw_vram_res = user_settings.get("vram_reserve_mib", getattr(config, "strata_vram_reserve_mib", None))
        if raw_vram_res is not None:
            try:
                vram_reserve_mib = int(raw_vram_res)
            except (ValueError, TypeError):
                vram_reserve_mib = None

    # When system_ram_kv_cache is enabled on Windows, ensure safe headroom (default 700 MiB)
    if system_ram_kv_cache and vram_reserve_mib is None:
        vram_reserve_mib = 700

    if vram_reserve_mib is not None and vram_reserve_mib >= 0:
        args += ["--vram-reserve-mib", str(vram_reserve_mib)]

    if mmap_experts:
        # Dynamic OS file-page cache mode (safe, avoids locking 50+ GB of committed RAM)
        args += ["--mmap-experts"]
        if resident_budget_gib is not None and float(resident_budget_gib) > 0:
            if profile_file.exists():
                budget_val = int(resident_budget_gib) if float(resident_budget_gib).is_integer() else float(resident_budget_gib)
                args += ["--resident-budget-gib", str(budget_val)]
            else:
                logger.warning(
                    f"[Strata Backend] Expert profile not found at {profile_file}. "
                    "Skipping --resident-budget-gib and falling back to pure dynamic mmap."
                )
    else:
        # Full in-memory resident arena (pure RAM speed, requires brutal RAM clawback)
        logger.warning(
            f"[Strata Backend] mmap_experts=False for {model_name}. "
            "Model will load all experts directly into physical RAM."
        )

    # Resolve use_custom_chat_template setting
    if use_custom_chat_template is None:
        if "use_custom_chat_template" in user_settings:
            use_custom_chat_template = bool(user_settings["use_custom_chat_template"])
        else:
            use_custom_chat_template = getattr(
                config,
                "local_claude_native_use_custom_chat_template",
                getattr(config, "strata_use_custom_chat_template", True)
            )
    else:
        use_custom_chat_template = bool(use_custom_chat_template)

    # Resolve chat_template_file
    if chat_template_file is None:
        chat_template_file = user_settings.get("chat_template_file") or getattr(
            config, "local_claude_native_chat_template_file", None
        )

    resolved_tmpl_path: Optional[Path] = None
    if chat_template_file:
        cand_tmpl = Path(chat_template_file)
        if not cand_tmpl.is_absolute():
            for base_dir in (AI_WORKSPACE_DIR, LOCAL_AI_LAUNCHER_DIR):
                if (base_dir / cand_tmpl).is_file():
                    cand_tmpl = base_dir / cand_tmpl
                    break
            if not cand_tmpl.is_absolute():
                cand_tmpl = AI_WORKSPACE_DIR / cand_tmpl
        if cand_tmpl.is_file():
            resolved_tmpl_path = cand_tmpl

    # Preserve original template and manage active template in pack tokenizer directory
    tokenizer_dir = pack_dir / "tokenizer" if pack_dir else None
    active_tmpl_desc = "original"
    if tokenizer_dir and tokenizer_dir.is_dir():
        orig_tmpl_file = tokenizer_dir / "chat_template.original.jinja"
        active_tmpl_file = tokenizer_dir / "chat_template.jinja"

        # 1. First-touch backup: preserve original GGUF template permanently
        if not orig_tmpl_file.is_file() and active_tmpl_file.is_file():
            try:
                shutil.copy2(active_tmpl_file, orig_tmpl_file)
                logger.info(f"[Strata Backend] Backed up original GGUF chat template to {orig_tmpl_file.name}")
            except Exception as e:
                logger.warning(f"[Strata Backend] Failed to backup original chat template: {e}")

        # 2. Deploy configured or original template based on use_custom_chat_template
        if use_custom_chat_template and resolved_tmpl_path:
            try:
                shutil.copy2(resolved_tmpl_path, active_tmpl_file)
                active_tmpl_desc = resolved_tmpl_path.as_posix()
                logger.info(f"[Strata Backend] Applied custom chat template: {resolved_tmpl_path.as_posix()}")
            except Exception as e:
                logger.warning(f"[Strata Backend] Failed to copy custom chat template to tokenizer dir: {e}")
        else:
            # Revert/preserve original template
            if orig_tmpl_file.is_file():
                try:
                    shutil.copy2(orig_tmpl_file, active_tmpl_file)
                    active_tmpl_desc = orig_tmpl_file.as_posix()
                    logger.info("[Strata Backend] Preserved and restored original GGUF chat template (strata_use_custom_chat_template=False)")
                except Exception as e:
                    logger.warning(f"[Strata Backend] Failed to restore original chat template: {e}")

    # Resolve reasoning_effort (from parameter, user settings, or model preset registry)
    if reasoning_effort is None:
        reasoning_effort = user_settings.get("reasoning_effort") or user_settings.get("effort")
    if reasoning_effort is None:
        try:
            registry = load_strata_model_registry()
            matched_spec = registry.get(model_name)
            if not matched_spec:
                for k, v in registry.items():
                    if k.lower() == model_name.lower():
                        matched_spec = v
                        break
            if matched_spec:
                r_sec = matched_spec.get("reasoning", {}) if isinstance(matched_spec.get("reasoning"), dict) else {}
                reasoning_effort = r_sec.get("effort") or r_sec.get("reasoning_effort") or matched_spec.get("reasoning_effort")
        except Exception:
            pass

    cfg = {
        "exe": engine_exe,
        "args": args,
        "cwd": STRATA_DIR.as_posix(),
        "tokenizer": (pack_dir / "tokenizer").as_posix(),
        "model_name": model_name,
        "log": (STRATA_DIR / f"strata-{clean_tag}.log").as_posix(),
        "port": port,
        "fit_max_tokens": user_settings.get("fit_max_tokens", True),
        "system_ram_kv_cache": system_ram_kv_cache,
        "chat_template_file": active_tmpl_desc,
        "use_custom_chat_template": use_custom_chat_template
    }
    if reasoning_effort:
        cfg["reasoning_effort"] = str(reasoning_effort).lower()

    # Save active config to destination
    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_output_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Also mirror to strata_cfg_path if target was launcher_cfg_path
    if target_output_path != strata_cfg_path and STRATA_DIR.exists():
        try:
            with open(strata_cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    logger.info(f"[Strata Backend] Generated Strata engine configuration: {target_output_path}")
    return target_output_path
