"""
LocalAI Unified Launcher Configuration.
Clean facade re-exporting paths, terminal prompts, and typed runtime configuration dataclasses.
Supports optional user overrides via launcher.toml or launcher.json.
"""
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

# Re-export filesystem paths & topology constants
from local_ai_launcher.paths import (
    AI_WORKSPACE_DIR,
    LOCAL_AI_LAUNCHER_DIR,
    CONFIG_DIR,
    PROJECTS_DIR,
    HERMES_DIR,
    PI_DIR,
    OMP_DIR,
    DSH_DIR,
    T3_DIR,
    ORCA_DIR,
    ORCA_DATA_DIR,
    CLAUDE_DIR,
    CLAUDE_LOCAL_DIR,
    CLAUDE_DESKTOP_DIR,
    GITHUB_DIR,
    AGY_IDE_DIR,
    AGY_DESKTOP_DIR,
    OPENLUMARA_DIR,
    LOCALAI_MODELS_DIR,
    LOCALAI_REPO_DIR,
    LOCALAI_BIN_DIR,
    LOCALAI_EXE,
    STRATA_DIR,
    STRATA_ENGINE_DIR,
    STRATA_EXE,
    SCRIPT_DIR,
    INSTALL_DIR_LLAMA_CPP,
    EXE_PATH_LLAMA_CPP,
    INSTALL_DIR_LLAMA_CPP_BLACKWELL,
    EXE_PATH_LLAMA_CPP_BLACKWELL,
    INSTALL_DIR_LLAMA_CPP_BRUNO,
    EXE_PATH_LLAMA_CPP_BRUNO,
    INSTALL_DIR_LLAMA_CPP_HUANG,
    EXE_PATH_LLAMA_CPP_HUANG,
    MODELS_PRESET_FILE,
    FREETOKENS_MODELS_PRESET_FILE,
    STRATA_MODELS_PRESET_FILE,
    MODELS_LOCAL_OVERRIDE_FILE,
    RESOLVED_MODELS_PRESET_FILE,
    TMP_DIR,
    root_dir,
    OMNIROUTE_PORT,
    LITELLM_PORT,
    LITELLM_CONFIG_FILE,
)

# Re-export preset resolution & layered overlay helpers
from local_ai_launcher.presets import (
    load_effective_models_parser,
    resolve_effective_models_preset,
)

# Re-export terminal prompting & interactive helpers
from local_ai_launcher.prompts import (
    prompt_pause,
    prompt_choice,
    prompt_string,
    fatal_error,
    _resolve_behavior,
)


def _extract_skill_sources(data: dict) -> dict:
    """Helper to extract suite definitions from nested TOML dicts."""
    if not isinstance(data, dict):
        return {}
    # Case: [skills.sources.<name>] or [skills.suites.<name>]
    if "skills" in data and isinstance(data["skills"], dict):
        skills_sec = data["skills"]
        for sub in ("sources", "suites", "external_sources"):
            if sub in skills_sec and isinstance(skills_sec[sub], dict):
                return skills_sec[sub]
        # Or [skills.<name>] where each entry is a suite config dict with git
        if any(isinstance(v, dict) and ("git" in v or "router_path" in v or "router_subpath" in v) for v in skills_sec.values()):
            return {k: v for k, v in skills_sec.items() if isinstance(v, dict)}
    # Case: [sources.<name>] or [suites.<name>] or [external_sources.<name>]
    for sub in ("sources", "suites", "external_sources"):
        if sub in data and isinstance(data[sub], dict):
            return data[sub]
    # Case: [<name>] where top-level tables are suite config dicts
    if any(isinstance(v, dict) and ("git" in v or "router_path" in v or "router_subpath" in v) for v in data.values()):
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _load_user_overrides(config_file: Optional[Path] = None) -> dict:
    """Load optional user overrides from launcher.toml, external_sources.toml, or launcher.json if present."""
    result = {}
    external_skill_suites = {}

    # 1. Base external sources shipped with local_ai_launcher (tracked in git)
    base_ext = LOCAL_AI_LAUNCHER_DIR / "external_sources.toml"
    if base_ext.is_file():
        try:
            import tomllib
            with open(base_ext, "rb") as f:
                base_data = tomllib.load(f)
                extracted = _extract_skill_sources(base_data)
                if extracted:
                    external_skill_suites.update(extracted)
        except Exception as e:
            logging.getLogger("service_launcher").warning(f"Failed to load base external_sources from {base_ext}: {e}")

    # 2. Workspace/user overrides for external sources (e.g. config/external_sources.toml)
    for user_ext in [
        CONFIG_DIR / "external_sources.toml",
        AI_WORKSPACE_DIR / "external_sources.toml",
    ]:
        if user_ext and user_ext.is_file() and user_ext.resolve() != base_ext.resolve():
            try:
                import tomllib
                with open(user_ext, "rb") as f:
                    u_data = tomllib.load(f)
                    extracted = _extract_skill_sources(u_data)
                    if extracted:
                        for k, v in extracted.items():
                            if k in external_skill_suites and isinstance(external_skill_suites[k], dict) and isinstance(v, dict):
                                external_skill_suites[k].update(v)
                            else:
                                external_skill_suites[k] = v
            except Exception as e:
                logging.getLogger("service_launcher").warning(f"Failed to load user external_sources from {user_ext}: {e}")
            break

    if external_skill_suites:
        result["external_sources"] = external_skill_suites

    # 3. Load primary launcher.toml / launcher.json
    candidates = []
    if config_file:
        candidates.append(Path(config_file))
    env_cfg = os.environ.get("LOCALAI_CONFIG_FILE")
    if env_cfg:
        candidates.append(Path(env_cfg))
    candidates.extend([
        LOCAL_AI_LAUNCHER_DIR / "launcher.toml",
        AI_WORKSPACE_DIR / "launcher.toml",
        LOCAL_AI_LAUNCHER_DIR / "launcher.json",
        AI_WORKSPACE_DIR / "launcher.json",
    ])
    for p in candidates:
        if p and p.is_file():
            try:
                if p.suffix.lower() == ".toml":
                    try:
                        import tomllib
                        with open(p, "rb") as f:
                            data = tomllib.load(f)
                            result.update(data)
                    except ImportError:
                        continue
                elif p.suffix.lower() == ".json":
                    import json
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        result.update(data)
            except Exception as e:
                logging.getLogger("service_launcher").warning(f"Failed to load config overrides from {p}: {e}")
            break
    return result


def _flatten_overrides(overrides: dict) -> dict:
    """Flatten nested sections from TOML (e.g. [loop_detection], [loop_recovery], [skills]) into top-level keys."""
    flat = {}
    for k, v in overrides.items():
        if isinstance(v, dict):
            if k in (
                "harness_update_behaviors", "harness_install_behaviors",
                "harness_extensions", "harness_skills", "harness_plugins"
            ):
                existing = flat.get(k, {})
                merged = dict(existing) if isinstance(existing, dict) else {}
                merged.update(v)
                flat[k] = merged
                continue

            if k in ("harness_update_frequency", "harness_update_frequency_hrs"):
                existing = flat.get("harness_update_frequency_hrs", {})
                merged = dict(existing) if isinstance(existing, dict) else {}
                merged.update(v)
                flat["harness_update_frequency_hrs"] = merged
                continue

            if k in ("engine_update_frequency", "engine_update_frequency_hrs"):
                existing = flat.get("engine_update_frequency_hrs", {})
                merged = dict(existing) if isinstance(existing, dict) else {}
                merged.update(v)
                flat["engine_update_frequency_hrs"] = merged
                continue

            if k in ("update_frequency", "update_frequencies", "update_frequency_hrs"):
                for sub_k, sub_v in v.items():
                    if sub_k in ("default", "global", "frequency", "hrs", "hours"):
                        flat["update_frequency_hrs"] = sub_v
                    elif sub_k in ("harness", "harnesses"):
                        if isinstance(sub_v, dict):
                            flat.setdefault("harness_update_frequency_hrs", {}).update(sub_v)
                    elif sub_k in ("engine", "engines"):
                        if isinstance(sub_v, dict):
                            flat.setdefault("engine_update_frequency_hrs", {}).update(sub_v)
                    else:
                        flat.setdefault("harness_update_frequency_hrs", {})[sub_k] = sub_v
                continue

            for sub_k, sub_v in v.items():
                if k not in ("external_sources", "external_skills"):
                    flat[sub_k] = sub_v

                if k == "loop_recovery":
                    flat[f"loop_recovery_{sub_k}"] = sub_v
                    if sub_k in ("temp_boost", "temp_reduce"):
                        flat["loop_recovery_temp_reduce"] = sub_v
                        flat["loop_recovery_temp_boost"] = sub_v
                elif k in ("loop_detection", "loop_detector"):
                    if sub_k == "enable":
                        flat["enable_loop_detection"] = sub_v
                    elif sub_k.startswith("recovery_"):
                        flat[f"loop_{sub_k}"] = sub_v
                    elif sub_k.startswith("detector_"):
                        flat[f"loop_{sub_k}"] = sub_v
                    else:
                        flat[f"loop_detector_{sub_k}"] = sub_v
                        flat[f"loop_recovery_{sub_k}"] = sub_v
                elif k in ("hardware", "profile", "profiles"):
                    if sub_k in ("profile", "name", "active_profile"):
                        flat["hardware_profile"] = sub_v
                    elif sub_k in ("models_preset", "models_preset_file"):
                        flat["models_preset_file"] = sub_v
                    elif sub_k in ("override", "models_override", "models_local_override_file"):
                        flat["models_local_override_file"] = sub_v
                    else:
                        flat[sub_k] = sub_v
                elif k in ("blackwell", "blackwell_engine"):
                    if sub_k in ("engine", "selected", "type", "variant"):
                        flat["blackwell_engine"] = sub_v
                    elif sub_k.startswith("huang_"):
                        flat[sub_k] = sub_v
                    elif sub_k in ("arena_mib", "draft_max", "draft_p_min", "draft_p_split", "spec_type", "auxiliary_layers", "batch_size", "ubatch_size", "parallel", "no_mmproj"):
                        flat[f"huang_{sub_k}"] = sub_v
                    else:
                        flat[f"blackwell_{sub_k}"] = sub_v
                elif k == "strata":
                    if sub_k in ("mmap_experts", "mmap"):
                        flat["strata_mmap_experts"] = sub_v
                    elif sub_k in ("brutal_ram_clawback", "brutal_clawback", "brutal"):
                        flat["strata_brutal_ram_clawback"] = sub_v
                    elif sub_k in ("kv_mode", "kv", "strata_kv_mode"):
                        flat["strata_kv_mode"] = sub_v
                    elif sub_k in ("resident_budget_gib", "resident_budget", "strata_resident_budget_gib"):
                        flat["strata_resident_budget_gib"] = sub_v
                    elif sub_k in ("use_custom_chat_template", "custom_chat_template", "use_custom_template", "strata_use_custom_chat_template"):
                        flat["strata_use_custom_chat_template"] = sub_v
                        flat["local_claude_native_use_custom_chat_template"] = sub_v
                    elif sub_k in ("system_ram_kv_cache", "ram_only_kv_cache", "system_ram_only_kv_cache", "ram_kv_cache", "strata_system_ram_kv_cache", "kv_streaming"):
                        flat["strata_system_ram_kv_cache"] = sub_v
                    elif sub_k in ("kv_resident", "strata_kv_resident"):
                        flat["strata_kv_resident"] = sub_v
                    elif sub_k in ("vram_reserve_mib", "vram_reserve", "strata_vram_reserve_mib"):
                        flat["strata_vram_reserve_mib"] = sub_v
                    else:
                        flat[f"strata_{sub_k}"] = sub_v
                elif k in ("local_claude_native", "claude_native"):
                    if sub_k in ("engines", "engine_list", "supported_engines"):
                        flat["local_claude_native_engines"] = sub_v
                    elif sub_k in ("engine", "default_engine"):
                        flat["local_claude_native_engine"] = sub_v
                        existing = flat.get("local_claude_native_engines", ["strata"])
                        if isinstance(existing, list) and sub_v not in existing:
                            flat["local_claude_native_engines"] = list(existing) + [sub_v]
                    elif sub_k in ("use_custom_chat_template", "custom_chat_template", "use_custom_template", "local_claude_native_use_custom_chat_template"):
                        flat["local_claude_native_use_custom_chat_template"] = sub_v
                        flat["strata_use_custom_chat_template"] = sub_v
                    elif sub_k in ("chat_template_file", "chat_template", "template"):
                        flat["local_claude_native_chat_template_file"] = sub_v
                    else:
                        flat[f"local_claude_native_{sub_k}"] = sub_v
                elif k in ("claude", "claude_code"):
                    if sub_k in ("mode", "claude_mode"):
                        flat["claude_mode"] = sub_v
                        flat["claude_mode_explicit"] = True
                    elif sub_k in ("model", "claude_model"):
                        flat["claude_model"] = sub_v
                    elif sub_k in ("api_key", "claude_api_key", "key"):
                        flat["claude_api_key"] = sub_v
                    elif sub_k in ("base_url", "claude_base_url", "url"):
                        flat["claude_base_url"] = sub_v
                    elif sub_k in ("dangerously_skip_permissions", "skip_permissions"):
                        flat["claude_dangerously_skip_permissions"] = sub_v
                    elif sub_k in ("permission_mode", "permissions_mode", "permission", "permissions"):
                        flat["claude_permission_mode"] = sub_v
                        flat["claude_permission_mode_explicit"] = True
                    elif sub_k in ("thinking_mode", "thinking_modes", "thinking", "initial_thinking_mode"):
                        flat["claude_thinking_mode"] = sub_v
                        flat["claude_thinking_mode_explicit"] = True
                    elif sub_k in ("offline_inject_contract", "inject_contract", "inject_persona_contract"):
                        flat["claude_offline_inject_contract"] = bool(sub_v)
                    elif sub_k in ("disallow_cloud_search_online", "disallow_web_search_online", "claude_disallow_cloud_search_online"):
                        flat["claude_disallow_cloud_search_online"] = sub_v
                    elif sub_k in ("disallow_cloud_search", "disallow_web_search", "claude_disallow_cloud_search"):
                        flat["claude_disallow_cloud_search"] = sub_v
                        flat["claude_disallow_cloud_search_online"] = sub_v
                    elif sub_k in ("auto_compact_ratio", "compact_ratio", "claude_auto_compact_ratio"):
                        flat["claude_auto_compact_ratio"] = sub_v
                    elif sub_k in ("max_context_tokens", "max_context", "context_tokens", "claude_max_context_tokens"):
                        flat["claude_max_context_tokens"] = sub_v
                    elif sub_k in ("auto_compact_window", "compact_window", "claude_auto_compact_window"):
                        flat["claude_auto_compact_window"] = sub_v
                    elif sub_k in ("desktop_mode", "claude_desktop_mode"):
                        flat["claude_desktop_mode"] = sub_v
                        flat["claude_desktop_mode_explicit"] = True
                    else:
                        flat[f"claude_{sub_k}"] = sub_v
                elif k in ("claude_desktop", "claudedesktop"):
                    if sub_k in ("mode", "desktop_mode", "claude_desktop_mode"):
                        flat["claude_desktop_mode"] = sub_v
                        flat["claude_desktop_mode_explicit"] = True
                    elif sub_k in ("permission_mode", "permissions_mode", "permission", "permissions"):
                        flat["claude_desktop_permission_mode"] = sub_v
                        flat["claude_desktop_permission_mode_explicit"] = True
                    elif sub_k in ("thinking_mode", "thinking_modes", "thinking", "initial_thinking_mode"):
                        flat["claude_desktop_thinking_mode"] = sub_v
                        flat["claude_desktop_thinking_mode_explicit"] = True
                    elif sub_k in ("offline_inject_contract", "inject_contract", "inject_persona_contract"):
                        flat["claude_offline_inject_contract"] = bool(sub_v)
                    elif sub_k in ("disallow_cloud_search_online", "disallow_web_search_online", "claude_desktop_disallow_cloud_search_online"):
                        flat["claude_disallow_cloud_search_online"] = sub_v
                    elif sub_k in ("auto_compact_ratio", "compact_ratio", "claude_auto_compact_ratio", "claude_desktop_auto_compact_ratio"):
                        flat["claude_desktop_auto_compact_ratio"] = sub_v
                    elif sub_k in ("max_context_tokens", "max_context", "context_tokens", "claude_max_context_tokens", "claude_desktop_max_context_tokens"):
                        flat["claude_desktop_max_context_tokens"] = sub_v
                    elif sub_k in ("auto_compact_window", "compact_window", "claude_auto_compact_window", "claude_desktop_auto_compact_window"):
                        flat["claude_desktop_auto_compact_window"] = sub_v
                    else:
                        flat[f"claude_desktop_{sub_k}"] = sub_v
                elif k in ("litellm", "lite_llm"):
                    if sub_k in ("enable", "enabled", "use_proxy", "use_litellm_proxy"):
                        flat["use_litellm_proxy"] = sub_v
                    elif sub_k in ("port", "litellm_port"):
                        flat["litellm_port"] = sub_v
                    elif sub_k in ("install", "install_behavior", "litellm_install_behavior"):
                        flat["litellm_install_behavior"] = sub_v
                    elif sub_k in ("update", "update_behavior", "litellm_update_behavior"):
                        flat["litellm_update_behavior"] = sub_v
                    else:
                        flat[f"litellm_{sub_k}"] = sub_v
                elif k == "gvs5h":
                    if sub_k in ("max_log_runs", "max_runs"):
                        flat["gvs5h_max_log_runs"] = sub_v
                    else:
                        flat[f"gvs5h_{sub_k}"] = sub_v
                elif k == "skills":
                    if sub_k in ("auto_update", "auto_update_on_launch"):
                        flat["auto_update_skills_on_launch"] = sub_v
                    elif sub_k in ("enable", "enable_gateway"):
                        flat["enable_skill_gateway"] = sub_v
                    elif sub_k in ("suites", "sources", "external_sources"):
                        existing = flat.get("skill_suites", {})
                        if isinstance(sub_v, dict):
                            merged = dict(existing)
                            for suite_k, suite_v in sub_v.items():
                                if suite_k in merged and isinstance(merged[suite_k], dict) and isinstance(suite_v, dict):
                                    merged[suite_k].update(suite_v)
                                else:
                                    merged[suite_k] = suite_v
                            flat["skill_suites"] = merged
                    else:
                        flat[f"skills_{sub_k}"] = sub_v
                elif k in ("external_sources", "external_skills"):
                    # sub_k is suite name (e.g. gstack), sub_v is suite config dict
                    existing = flat.get("skill_suites", {})
                    merged = dict(existing)
                    if isinstance(sub_v, dict):
                        if sub_k in merged and isinstance(merged[sub_k], dict):
                            merged[sub_k].update(sub_v)
                        else:
                            merged[sub_k] = sub_v
                    flat["skill_suites"] = merged
                elif k in ("gateway", "localai"):
                    if sub_k in ("unknown_model_policy", "gateway_unknown_model_policy"):
                        flat["gateway_unknown_model_policy"] = sub_v
                elif k in ("donsetch", "don_setch"):
                    for sub_k, sub_v in v.items():
                        if sub_k in ("enable", "enabled"):
                            flat["pi_donsetch_enabled"] = bool(sub_v)
                            flat["deepseek_donsetch_enabled"] = bool(sub_v)
                        elif sub_k in ("pi", "pi_enabled"):
                            flat["pi_donsetch_enabled"] = bool(sub_v)
                        elif sub_k in ("deepseek", "dsh", "deepseek_enabled", "dsh_enabled"):
                            flat["deepseek_donsetch_enabled"] = bool(sub_v)
                        elif sub_k in ("tool_prefix", "prefix"):
                            flat["donsetch_tool_prefix"] = str(sub_v)
                        elif sub_k in ("auto_update", "autoupdate"):
                            flat["donsetch_auto_update"] = bool(sub_v)
                        elif sub_k in ("channel", "release_channel"):
                            flat["donsetch_channel"] = str(sub_v)
                        elif sub_k in ("bin", "binary", "path"):
                            flat["donsetch_bin"] = Path(sub_v) if sub_v else None
                        else:
                            flat[f"donsetch_{sub_k}"] = sub_v
                elif k == "pi":
                    for sub_k, sub_v in v.items():
                        if sub_k in ("donsetch", "donsetch_enabled", "enable_donsetch"):
                            flat["pi_donsetch_enabled"] = bool(sub_v)
                        elif sub_k in ("extensions", "extension"):
                            existing_e = flat.get("harness_extensions", {})
                            m_e = dict(existing_e) if isinstance(existing_e, dict) else {}
                            m_e.setdefault("pi", []).extend(sub_v if isinstance(sub_v, list) else [sub_v])
                            flat["harness_extensions"] = m_e
                elif k in ("deepseek", "dsh"):
                    for sub_k, sub_v in v.items():
                        if sub_k in ("donsetch", "donsetch_enabled", "enable_donsetch"):
                            flat["deepseek_donsetch_enabled"] = bool(sub_v)
                        elif sub_k in ("plugins", "plugin"):
                            existing_p = flat.get("harness_plugins", {})
                            m_p = dict(existing_p) if isinstance(existing_p, dict) else {}
                            m_p.setdefault("deepseek", []).extend(sub_v if isinstance(sub_v, list) else [sub_v])
                            flat["harness_plugins"] = m_p
                        elif sub_k in ("tool_prefix", "prefix"):
                            flat["donsetch_tool_prefix"] = str(sub_v)
                elif k in ("searxng", "searx"):
                    for sub_k, sub_v in v.items():
                        if sub_k in ("enable", "enabled"):
                            flat["searxng_enabled"] = bool(sub_v)
                        elif sub_k == "mode":
                            flat["searxng_mode"] = str(sub_v).strip().lower()
                        elif sub_k == "port":
                            flat["searxng_port"] = sub_v
                        elif sub_k == "host":
                            flat["searxng_host"] = str(sub_v)
                        elif sub_k in ("remote_url", "url"):
                            flat["searxng_remote_url"] = str(sub_v)
                        elif sub_k in ("upstream_engines", "engines", "providers"):
                            flat["searxng_upstream_engines"] = list(sub_v) if isinstance(sub_v, list) else [str(sub_v)]
                        elif sub_k in ("max_results", "limit", "default_limit", "max_hits"):
                            flat["searxng_max_results"] = int(sub_v)
                        elif sub_k in ("categories", "category"):
                            flat["searxng_categories"] = str(sub_v)
                        elif sub_k in ("auto_fallback_native", "fallback_native"):
                            flat["searxng_auto_fallback_native"] = bool(sub_v)
                        elif sub_k in ("timeout", "search_timeout"):
                            flat["searxng_timeout"] = float(sub_v)
                        elif sub_k in ("wsl_distro", "distro"):
                            flat["searxng_wsl_distro"] = str(sub_v)
                        elif sub_k in ("wsl_container_name", "container_name"):
                            flat["searxng_wsl_container_name"] = str(sub_v)
                        elif sub_k in ("uwsgi_workers", "workers"):
                            flat["searxng_uwsgi_workers"] = int(sub_v)
                        elif sub_k in ("uwsgi_threads", "threads"):
                            flat["searxng_uwsgi_threads"] = int(sub_v)
                        elif sub_k == "limiter":
                            flat["searxng_limiter"] = bool(sub_v)
                        else:
                            flat[f"searxng_{sub_k}"] = sub_v
                elif k in ("rea", "mcp_rea"):
                    for sub_k, sub_v in v.items():
                        if sub_k in ("enable", "enabled"):
                            flat["rea_enabled"] = bool(sub_v)
                        elif sub_k in ("profile", "schema_profile", "input_schema_profile"):
                            flat["rea_schema_profile"] = str(sub_v).strip().lower()
                        else:
                            flat[f"rea_{sub_k}"] = sub_v
                elif k in ("aside", "aside_browser", "aside_cli"):
                    for sub_k, sub_v in v.items():
                        if sub_k in ("enable", "enabled"):
                            flat["aside_browser_enabled"] = bool(sub_v)
                        elif sub_k in ("model", "model_name"):
                            flat["aside_browser_model"] = str(sub_v)
                        elif sub_k in ("bin", "binary", "path"):
                            flat["aside_browser_bin"] = Path(sub_v) if sub_v else None
                        elif sub_k in ("auto_sync", "sync"):
                            flat["aside_browser_auto_sync"] = bool(sub_v)
                        elif sub_k in ("headless", "run_headless"):
                            flat["aside_browser_headless"] = bool(sub_v)
                        else:
                            flat[f"aside_browser_{sub_k}"] = sub_v
                elif k in ("git", "git_client"):
                    for sub_k, sub_v in v.items():
                        if sub_k in ("auto_update", "autoupdate", "enable", "enabled"):
                            flat["git_auto_update"] = bool(sub_v)
                            if bool(sub_v) and flat.get("git_update_behavior", "never_update") == "never_update":
                                flat["git_update_behavior"] = "always_update"
                        elif sub_k in ("update", "update_behavior", "behavior"):
                            flat["git_update_behavior"] = str(sub_v).strip().lower()
                            if flat["git_update_behavior"] in ("always_update", "always", "existing"):
                                flat["git_auto_update"] = True
                        elif sub_k in ("frequency", "frequency_hrs", "update_frequency_hrs"):
                            flat["git_update_frequency_hrs"] = sub_v
                        else:
                            flat[f"git_{sub_k}"] = sub_v
        else:
            if k == "update_frequency":
                flat["update_frequency_hrs"] = v
            elif k in ("git_auto_update", "git_autoupdate", "auto_update_git"):
                flat["git_auto_update"] = bool(v)
                if bool(v) and flat.get("git_update_behavior", "never_update") == "never_update":
                    flat["git_update_behavior"] = "always_update"
            elif k in ("git_update_behavior", "git_update"):
                flat["git_update_behavior"] = str(v).strip().lower()
                if flat["git_update_behavior"] in ("always_update", "always", "existing"):
                    flat["git_auto_update"] = True
            elif k in ("git_update_frequency_hrs", "git_update_frequency"):
                flat["git_update_frequency_hrs"] = v
            elif k in ("rea_enabled", "enable_rea"):
                flat["rea_enabled"] = bool(v)
            elif k in ("rea_schema_profile", "rea_profile"):
                flat["rea_schema_profile"] = str(v).strip().lower()
            elif k in ("system_ram_kv_cache", "ram_only_kv_cache", "system_ram_only_kv_cache", "ram_kv_cache", "kv_streaming"):
                flat["strata_system_ram_kv_cache"] = v
            elif k in ("kv_resident", "strata_kv_resident"):
                flat["strata_kv_resident"] = v
            elif k in ("vram_reserve_mib", "vram_reserve", "strata_vram_reserve_mib"):
                flat["strata_vram_reserve_mib"] = v
            elif k in ("local_claude_native_engines", "claude_native_engines"):
                flat["local_claude_native_engines"] = v
            elif k in ("local_claude_native_use_custom_chat_template", "claude_native_use_custom_chat_template"):
                flat["local_claude_native_use_custom_chat_template"] = v
                flat["strata_use_custom_chat_template"] = v
            elif k in ("local_claude_native_chat_template_file", "claude_native_chat_template_file"):
                flat["local_claude_native_chat_template_file"] = v
            elif k in ("local_claude_native_engine", "claude_native_engine"):
                flat["local_claude_native_engine"] = v
            elif k in ("claude_thinking_mode", "thinking_mode"):
                flat["claude_thinking_mode"] = v
                flat["claude_thinking_mode_explicit"] = True
            elif k in ("claude_offline_inject_contract", "offline_inject_contract"):
                flat["claude_offline_inject_contract"] = bool(v)
            elif k in ("aside_browser_enabled", "enable_aside_browser", "aside_enabled"):
                flat["aside_browser_enabled"] = bool(v)
            elif k in ("aside_browser_model", "aside_model"):
                flat["aside_browser_model"] = str(v)
            elif k in ("aside_browser_bin", "aside_bin"):
                flat["aside_browser_bin"] = Path(v) if v else None
            elif k in ("aside_browser_auto_sync", "aside_auto_sync"):
                flat["aside_browser_auto_sync"] = bool(v)
            elif k in ("aside_browser_headless", "aside_headless"):
                flat["aside_browser_headless"] = bool(v)
            else:
                flat[k] = v
    return flat


# =============================================================================
# High-Fidelity Hardware Profiles Registry
# Preserves 100% of tested numbers without lossy heuristics or dumb toggles.
# =============================================================================
HARDWARE_PROFILES: Dict[str, Dict[str, Any]] = {
    "blackwell_16gb": {
        "name": "blackwell_16gb",
        "description": "Nvidia Blackwell 16GB VRAM (NVFP4/EXL3) + DDR5 CPU Auxiliary Offloading",
        "engine_binary": "llama-cpp-blackwell/llama-server.exe",
        "vram_ceiling_mb": 16384,
        "vram_evict_threshold_mb": 15000,
        "strict_vram_eviction": True,
        "default_n_cpu_moe": 25,
        "ctk": "q8_0",
        "ctv": "q8_0",
        "batch_size": 4096,
        "ubatch_size": 4096,
        "clm_pcore_affinity": True,
        "clm_pcore_mask": "0x00FF",
        "aux_pcore_affinity": True,
        "aux_pcore_mask": "0x00FF",
    },
    "ada_24gb": {
        "name": "ada_24gb",
        "description": "Nvidia Ada / Ampere 24GB VRAM (Full GPU Offload)",
        "engine_binary": "llama-cpp/llama-server.exe",
        "vram_ceiling_mb": 24576,
        "vram_evict_threshold_mb": 23000,
        "strict_vram_eviction": False,
        "default_n_cpu_moe": 0,
        "ctk": "q8_0",
        "ctv": "q8_0",
        "batch_size": 8192,
        "ubatch_size": 4096,
        "clm_pcore_affinity": False,
        "clm_pcore_mask": "",
        "aux_pcore_affinity": False,
        "aux_pcore_mask": "",
    },
    "standard_16gb": {
        "name": "standard_16gb",
        "description": "Standard CUDA 16GB VRAM (e.g. RTX 4080 / 4070Ti Super)",
        "engine_binary": "llama-cpp/llama-server.exe",
        "vram_ceiling_mb": 16384,
        "vram_evict_threshold_mb": 15000,
        "strict_vram_eviction": True,
        "default_n_cpu_moe": 25,
        "ctk": "q8_0",
        "ctv": "q8_0",
        "batch_size": 4096,
        "ubatch_size": 4096,
        "clm_pcore_affinity": False,
        "clm_pcore_mask": "",
        "aux_pcore_affinity": False,
        "aux_pcore_mask": "",
    },
    "budget_12gb": {
        "name": "budget_12gb",
        "description": "Standard CUDA 12GB VRAM (e.g. RTX 3060 / 4070)",
        "engine_binary": "llama-cpp/llama-server.exe",
        "vram_ceiling_mb": 12288,
        "vram_evict_threshold_mb": 11500,
        "strict_vram_eviction": True,
        "default_n_cpu_moe": 32,
        "ctk": "q4_0",
        "ctv": "q4_0",
        "batch_size": 2048,
        "ubatch_size": 2048,
        "clm_pcore_affinity": False,
        "clm_pcore_mask": "",
        "aux_pcore_affinity": False,
        "aux_pcore_mask": "",
    },
}


def get_active_hardware_profile(cfg: Optional["LauncherConfig"] = None) -> Dict[str, Any]:
    """Returns the dictionary definition for the active hardware profile, falling back to blackwell_16gb."""
    active_name = "blackwell_16gb"
    effective_cfg = cfg if cfg is not None else globals().get("config")
    if effective_cfg is not None and hasattr(effective_cfg, "hardware_profile"):
        active_name = effective_cfg.hardware_profile or "blackwell_16gb"
    profile = HARDWARE_PROFILES.get(active_name)
    if not profile:
        profile = HARDWARE_PROFILES.get("blackwell_16gb", {})
    res = dict(profile)
    if active_name == "blackwell_16gb":
        bw_engine = getattr(effective_cfg, "blackwell_engine", "huang") if effective_cfg else "huang"
        if "bruno" in str(bw_engine).lower():
            res["engine_binary"] = "llama-cpp-blackwell/llama-server.exe"
        else:
            res["engine_binary"] = "llama-cpp-huang/llama-server.exe"
    return res


def _resolve_default_model() -> str:
    """Resolves default model dynamically based on environment or existing on-disk weights."""
    env_m = os.environ.get("LOCALAI_DEFAULT_MODEL") or os.environ.get("DEFAULT_MODEL")
    if env_m:
        return env_m.strip()

    gsq_path = AI_WORKSPACE_DIR / "models/llama-server/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    if gsq_path.exists():
        return "qwen-38_27b-huang-gsq-iq3s-think"

    bruno_path = AI_WORKSPACE_DIR / "models/llama-server/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf"
    if bruno_path.exists():
        return "qwen-38_27b-blackwell-256k-think"

    return "qwen-38_27b-huang-gsq-iq3s-think"


def _get_strata_setting(key: str, fallback: Any) -> Any:
    """Retrieves setting from strata_settings.json (or environment variable), falling back cleanly."""
    env_val = os.environ.get(f"STRATA_{key.upper()}")
    if env_val is not None:
        if isinstance(fallback, bool):
            return env_val.strip().lower() not in ("0", "false", "off", "no")
        elif isinstance(fallback, (int, float)):
            try:
                return type(fallback)(env_val)
            except ValueError:
                pass
        return env_val.strip()
    try:
        from local_ai_launcher.backends.strata import load_strata_settings
        st = load_strata_settings()
        if key in st:
            return st[key]
    except Exception:
        pass
    return fallback


@dataclass
class LauncherConfig:
    """
    Central runtime configuration for local_ai_launcher.
    
    Default model can be overridden by name or alias:
      - GGUF Blackwell: "qwen-38_27b-blackwell-256k-think", "qwen-38_27b-blackwell-256k-nothink"
      - EXL3 (TabbyAPI): "Qwen3.8-27B-exl3-SC-3.0bpw-H4-V4", "Muse-Glimmer-30B-exl3-3.0bpw"
      - NVFP4 (FreeToken): "nvidia_Qwen3.6-35B-A3B-NVFP4"
      - Strata (Speculative MoE): "strata-iq3s", "qwen-38-flash-strata-iq3s"
      - GGUF standard: "gemma-4-12B-it-QAT-audio-think-uncen-vision"
    """
    # Active Hardware Profile & High-Fidelity Presets
    hardware_profile: str = "blackwell_16gb"
    models_preset_file: Optional[str] = None
    models_local_override_file: Optional[str] = None

    default_model: str = field(default_factory=_resolve_default_model)
    # Active harnesses to launch. When multiple offline-capable harnesses are defined,
    # the leftmost non-meta harness is preloaded into the LLM engine server KV cache at boot,
    # and the leftmost meta-harness (e.g. orca, t3) is initialized with the leftmost offline CLI harness.
    choose_harness: List[str] = field(default_factory=lambda: ["agy-desktop", "orca"])
    
    def get_server_preloaded_harness(self) -> str:
        """Returns the leftmost non-meta offline-capable harness to preload into the LLM engine server."""
        from local_ai_launcher.harness_model import resolve_server_preloaded_harness
        return resolve_server_preloaded_harness(self.choose_harness, model_name=self.default_model, cfg=self)

    def get_meta_harness_inner_cli_harness(self) -> str:
        """Returns the leftmost non-meta offline CLI harness to load inside the leftmost meta-harness."""
        from local_ai_launcher.harness_model import resolve_meta_harness_inner_cli_harness
        return resolve_meta_harness_inner_cli_harness(self.choose_harness, model_name=self.default_model, cfg=self)
    
    # LocalAI Core Configuration
    localai_port: int = 8080
    localai_host: str = "127.0.0.1"
    localai_single_active_backend: bool = True
    localai_watchdog_idle: bool = True
    localai_watchdog_idle_timeout: str = "5m"
    localai_execution_mode: str = "native"  # Options: 'auto', 'native', 'docker', 'wsl'
    localai_headless: bool = field(
        default_factory=lambda: os.environ.get("LOCALAI_HEADLESS", "").strip().lower() in ("1", "true", "yes")
    )
    gateway_unknown_model_policy: str = field(
        default_factory=lambda: os.environ.get("LOCALAI_UNKNOWN_MODEL_POLICY", "reject").strip().lower()
    )

    @property
    def open_localai_gui(self) -> bool:
        return not self.localai_headless

    @open_localai_gui.setter
    def open_localai_gui(self, val: bool):
        self.localai_headless = not val
    
    # gRPC Shim Ports (for external backends)
    blackwell_shim_port: int = 9940
    tabbyapi_shim_port: int = 9941
    freetoken_shim_port: int = 9942
    strata_shim_port: int = 9943
    strata_engine_port: int = 9934
    strata_mmap_experts: bool = field(
        default_factory=lambda: bool(_get_strata_setting("mmap_experts", True))
    )
    strata_brutal_ram_clawback: str = field(
        default_factory=lambda: str(_get_strata_setting("brutal_ram_clawback", "light")).strip().lower()
    )
    strata_kv_mode: str = field(
        default_factory=lambda: str(_get_strata_setting("kv_mode", "int8")).strip().lower()
    )
    strata_resident_budget_gib: float = field(
        default_factory=lambda: float(_get_strata_setting("resident_budget_gib", 52.0))
    )
    # Local Claude Native Shared Configuration
    # Engines supporting native Anthropic Messages API (Strata is default member; expandable to new engines)
    local_claude_native_engines: List[str] = field(
        default_factory=lambda: ["strata"]
    )
    local_claude_native_use_custom_chat_template: bool = field(
        default_factory=lambda: bool(_get_strata_setting("use_custom_chat_template", True))
    )
    local_claude_native_chat_template_file: Optional[str] = None
    local_claude_native_engine: Optional[str] = None

    # Strata MoE Engine Specific Settings (Strata is a member of local_claude_native)
    strata_use_custom_chat_template: bool = field(
        default_factory=lambda: bool(_get_strata_setting("use_custom_chat_template", True))
    )
    strata_system_ram_kv_cache: bool = field(
        default_factory=lambda: bool(_get_strata_setting("system_ram_kv_cache", False))
    )
    strata_kv_resident: Optional[int] = field(
        default_factory=lambda: _get_strata_setting("kv_resident", None)
    )
    strata_vram_reserve_mib: Optional[int] = field(
        default_factory=lambda: _get_strata_setting("vram_reserve_mib", None)
    )

    def is_local_claude_native_engine(self, engine: Optional[str]) -> bool:
        """Checks if the given engine name is recognized as a member of local_claude_native engines."""
        if not engine:
            return False
        clean = str(engine).strip().lower()
        if clean in [str(e).strip().lower() for e in self.local_claude_native_engines]:
            return True
        try:
            from local_ai_launcher.harness_model import is_local_claude_native_engine as _check_eng
            return _check_eng(clean, self)
        except Exception:
            return clean == "strata"
    
    # Real-Time LLM Loop Detection Configuration
    enable_loop_detection: bool = True
    loop_detector_ngram_size: int = 8
    loop_detector_ngram_threshold: int = 3
    loop_detector_cycle_similarity: float = 0.82
    loop_detector_min_sentence_len: int = 35

    # Autonomous Loop Recovery & Sampler Perturbation
    loop_recovery_max_retries: int = 3
    loop_recovery_temp_reduce: float = 0.1
    loop_recovery_temp_boost: float = 0.1
    loop_recovery_temp_floor: float = 0.15
    loop_recovery_presence_penalty: float = 0.0
    
    use_omniroute_proxy: bool = False
    omniroute_default_model: str = "1-local_first-brain"
    run_mnemosyne_verification: bool = True
    show_launcher_logging: bool = True
    show_llama_server_console: bool = True
    preload_default_model: bool = True
    enable_context_preloading: bool = True
    skip_hermes_web_ui_build: bool = True
    create_ai_bot_user: bool = False
    ai_bot_username: str = "ai_bot"
    ai_bot_password: str = "ai_bot"
    start_signal_daemon: bool = True
    signal_daemon_port: int = 8085
    start_discord_gateway: bool = True

    localai_repo_url: str = "https://github.com/mudler/LocalAI.git"
    localai_install_behavior: str = "existing"
    localai_update_behavior: str = "never_update"

    llamacpp_fork: str = "official"
    llamacpp_install_behavior: str = "existing"
    llamacpp_blackwell_install_behavior: str = "existing"
    llamacpp_blackwell_update_behavior: str = "never_update"

    # Modular Blackwell Engine ("huang" [default] or "bruno")
    blackwell_engine: str = "huang"
    huang_arena_mib: int = 2240
    huang_auxiliary_layers: int = 1
    huang_draft_max: int = 3
    huang_draft_p_min: float = 0.20
    huang_draft_p_split: float = 0.10
    huang_spec_type: str = "draft-mtp"
    huang_batch_size: int = 512
    huang_ubatch_size: int = 256
    huang_parallel: int = 1
    huang_no_mmproj: bool = True
    llamacpp_huang_install_behavior: str = "existing"
    llamacpp_huang_update_behavior: str = "never_update"
    models_install_behavior: str = "existing"
    omniroute_install_behavior: str = "existing"
    tabbyapi_install_behavior: str = "existing"
    freetoken_install_behavior: str = "existing"
    litellm_install_behavior: str = "always_install"
    harness_install_behaviors: Dict[str, str] = field(default_factory=lambda: {
        'pi': 'always_install', 'pie': 'always_install', 'hermes': 'always_install', 'opencode': 'never_install',
        'omp': 'always_install', 'oh-my-pi': 'always_install',
        'agy': 'always_install', 'deepseek': 'always_install', 'orca': 'always_install', 't3': 'always_install',
        'agy-ide': 'always_install', 'agy-desktop': 'always_install', 'openlumara': 'always_install',
        'antigravity': 'always_install', 'claude': 'always_install', 'claude-code': 'always_install',
        'claude-desktop': 'always_install', 'claudedesktop': 'always_install'
    })
    
    git_update_behavior: str = 'never_update'
    git_auto_update: bool = False
    git_update_frequency_hrs: Union[int, float] = 168
    
    github_sync_mode: str = 'off'
    github_install_behavior: str = 'never_install'
    github_update_behavior: str = 'never_update'
    
    llamacpp_update_behavior: str = 'never_update'
    omniroute_update_behavior: str = 'never_update'
    tabbyapi_update_behavior: str = 'never_update'
    freetoken_update_behavior: str = 'never_update'
    litellm_update_behavior: str = "never_update"
    harness_update_behaviors: Dict[str, str] = field(default_factory=lambda: {
        'pi': 'always_update', 'pie': 'never_update', 'hermes': 'always_update', 'opencode': 'always_update',
        'omp': 'always_update', 'oh-my-pi': 'always_update',
        'agy': 'always_update', 'deepseek': 'always_update', 'orca': 'always_update', 't3': 'always_update',
        'agy-ide': 'always_update', 'agy-desktop': 'always_update', 'openlumara': 'always_update',
        'antigravity': 'always_update', 'claude': 'always_update', 'claude-code': 'always_update',
        'claude-desktop': 'always_update', 'claudedesktop': 'always_update'
    })

    # Update Frequency Controls in hours (e.g. 0 = always, 1 = 1 hour, 24 = 24 hours, 168 = weekly, < 0 = never)
    update_frequency_hrs: Union[int, float] = 168
    force_update: bool = False
    harness_update_frequency_hrs: Dict[str, Union[int, float]] = field(default_factory=dict)
    engine_update_frequency_hrs: Dict[str, Union[int, float]] = field(default_factory=dict)
    
    harness_extensions: Dict[str, List[str]] = field(default_factory=lambda: {
        'pi': ["https://github.com/tmdgusya/roach-pi", "npm:donsetch"], 'hermes': [], 'pie': [], 'opencode': [], 'agy': [],
        'omp': [], 'oh-my-pi': [],
        'deepseek': [], 'orca': [], 't3': [], 'agy-ide': [], 'agy-desktop': [], 'openlumara': [],
        'antigravity': [], 'claude': [], 'claude-desktop': []
    })
    harness_plugins: Dict[str, List[str]] = field(default_factory=lambda: {
        'deepseek': ["github:dondai44423/donsetch-dsh"], 'hermes': [], 'pie': [], 'opencode': [], 'agy': [],
        'pi': [], 'omp': [], 'oh-my-pi': [], 'orca': [], 't3': [], 'agy-ide': [], 'agy-desktop': [], 'openlumara': [],
        'antigravity': [], 'claude': [], 'claude-desktop': []
    })
    harness_skills: Dict[str, List[str]] = field(default_factory=lambda: {
        'pi': ["https://github.com/DietrichGebert/ponytail"], 'hermes': [], 'pie': [], 'opencode': [], 'agy': [],
        'omp': [], 'oh-my-pi': [],
        'deepseek': [], 'orca': [], 't3': [], 'agy-ide': [], 'agy-desktop': [], 'openlumara': [],
        'antigravity': [], 'claude': [], 'claude-desktop': []
    })
    
    # DonSeTch Web Research Configuration (Scoped to Pi and DSH)
    pi_donsetch_enabled: bool = True
    deepseek_donsetch_enabled: bool = True
    donsetch_tool_prefix: str = "donsetch"
    donsetch_auto_update: bool = True
    donsetch_channel: str = "stable"
    donsetch_bin: Optional[Path] = None

    enable_skill_gateway: bool = True
    auto_update_skills_on_launch: bool = False
    skill_suites: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    mcp_server_port: int = 9932
    t3_port: int = 3773

    # REA (Reverse Engineer Anything) MCP Toolkit Configuration
    rea_enabled: bool = True
    rea_schema_profile: str = "compact"

    # Claude Code CLI & Desktop Configuration
    claude_dir: Path = CLAUDE_DIR
    claude_local_dir: Path = CLAUDE_LOCAL_DIR
    claude_mode: str = "local"
    claude_mode_explicit: bool = False
    claude_model: str = "claude-sonnet-4-5"
    claude_api_key: str = "dummy-token"
    claude_base_url: str = ""
    claude_dangerously_skip_permissions: bool = False
    claude_permission_mode: str = "smart"  # "smart" (acceptEdits) | "yolo" (bypassPermissions) | "default" (prompts for edits)
    claude_permission_mode_explicit: bool = False
    claude_thinking_mode: str = "adaptive"  # "adaptive" (equivalent to enabled) | "disabled" | "low" | "medium" | "high"
    claude_thinking_mode_explicit: bool = False
    claude_offline_inject_contract: bool = True  # Injects anti-refusal persona contract into /v1/messages for offline Claude harnesses
    claude_auto_compact_ratio: float = 0.91
    claude_max_context_tokens: Optional[int] = None
    claude_auto_compact_window: Optional[int] = None
    claude_disallow_cloud_search: bool = True
    claude_disallow_cloud_search_online: bool = False

    claude_desktop_dir: Path = CLAUDE_DESKTOP_DIR
    claude_desktop_mode: str = "mcp"  # "mcp" (MCP bridge to local model) | "gateway" (3P Developer mode direct)
    claude_desktop_mode_explicit: bool = False
    claude_desktop_permission_mode: Optional[str] = None  # None (inherits claude_permission_mode) | "smart" | "yolo" | "default"
    claude_desktop_permission_mode_explicit: bool = False
    claude_desktop_thinking_mode: Optional[str] = None  # None (inherits claude_thinking_mode) | "adaptive" | "disabled" | "low" | "medium" | "high"
    claude_desktop_thinking_mode_explicit: bool = False
    claude_desktop_auto_compact_ratio: Optional[float] = None
    claude_desktop_max_context_tokens: Optional[int] = None
    claude_desktop_auto_compact_window: Optional[int] = None
    _initialized: bool = False
    
    # LiteLLM Universal OpenAI <-> Anthropic Proxy Configuration
    use_litellm_proxy: bool = True
    litellm_port: int = LITELLM_PORT
    litellm_config_file: Path = LITELLM_CONFIG_FILE
    
    voice_to_text: str = 'parakeet_tdt_v3'
    parakeet_device: str = 'NPU'
    voice_dictation_hotkey: str = 'f8'
    orca_voice_to_text: bool = False

    @property
    def use_intel_npu_voice_to_text(self) -> bool:
        return str(getattr(self, 'voice_to_text', 'parakeet_tdt_v3')).lower() not in ('none', 'false', 'off', 'disabled')

    @use_intel_npu_voice_to_text.setter
    def use_intel_npu_voice_to_text(self, val: bool):
        if not val:
            self.voice_to_text = 'none'
        elif getattr(self, 'voice_to_text', 'none') == 'none':
            self.voice_to_text = 'parakeet_tdt_v3'

    # 1. CPU-Only llama-server (Port 8081 - Media Director / Gemma 4 Vision GGUF or CPU text models)
    # 1. CPU-Only llama-server (Port 8081 - Media Production Director / Gemma 4 Vision GGUF)
    # Distinct reference for other AI and studio_addon to use with 0 MB GPU VRAM
    enable_cpu_llama_server: bool = True
    cpu_llama_server_model: str = 'gemma-4-12B-it-QAT-audio-think-uncen-vision'
    cpu_llama_server_threads: int = 4
    cpu_llama_server_port: int = 8081
    cpu_llama_server_context_size: int = 16384

    @property
    def use_cpu_llama_server(self) -> bool:
        if not getattr(self, 'enable_cpu_llama_server', True):
            return False
        return str(getattr(self, 'cpu_llama_server_model', '')).lower() not in ('none', 'false', 'off', 'disabled', '')

    # 2. CPU Auxiliary Reranker Engine (Port 8082 - bge-reranker-v2-m3)
    enable_cpu_reranker: bool = True
    cpu_reranker_model: str = 'BAAI/bge-reranker-v2-m3'
    cpu_reranker_device: str = 'CPU'
    cpu_reranker_port: int = 8082
    cpu_reranker_max_doc_tokens: int = 3000

    @property
    def use_cpu_reranker(self) -> bool:
        if not getattr(self, 'enable_cpu_reranker', True):
            return False
        return str(getattr(self, 'cpu_reranker_model', '')).lower() not in ('none', 'false', 'off', 'disabled', '')

    # 3. CPU Auxiliary Model 1: von (Port 8083 - Sub-15ms System One Reflex Gatekeeper)
    # ModernBERT-based non-autoregressive encoder evaluating Choice, Noul (Yes/No), and Score.
    enable_von_gater: bool = True
    von_model: str = "wfzyx/von"
    von_port: int = 8083
    von_threads: int = 4
    von_bypass_trivial_queries: bool = False
    von_safety_tripwire_threshold: float = 0.90

    # Options wired to ONNX Decision Gater (with backward compatibility)
    enable_cpu_onnx_gater: bool = True
    cpu_onnx_gater_model: str = 'wfzyx/von'
    cpu_onnx_gater_port: int = 8083
    cpu_onnx_gater_device: str = 'CPU'
    cpu_onnx_gater_threads: int = 4

    enable_cpu_decision: bool = True
    cpu_decision_model: str = 'wfzyx/von'
    cpu_decision_threads: int = 4
    cpu_decision_port: int = 8083
    cpu_decision_context_size: int = 16384

    # 4. CPU Auxiliary Model 2: decider-4b v2.1 (Port 8084 - Complex Long-Horizon Decision & Multi-Slot Evaluator)
    # Fine-tuned Qwen3.5-4B schema-first decision model (0 MB GPU VRAM, DDR5 CPU AVX-VNNI).
    # Evaluates multi-slot extraction, DOM accessibility probes (Mind2Web / MiniWoB++), and MCTS softened policy priors.
    # Architecture Note (Port 8081 & 8084):
    # - Port 8081: The unified CPU llama-server.exe worker port. Used as the C++ inference backend for Decider-4B
    #   and also hosts the Gemma-4 Director during media production workflows.
    #   These two workloads are strictly mutually exclusive and never overlap.
    # - Port 8084: The dedicated FastAPI Engine providing structured cognitive endpoints (/v1/decider/*).
    # - Port 9935: Retired; low-level C++ worker inference runs directly on Port 8081.
    enable_decider_4b: bool = True
    decider_4b_model: str = "Mapika/decider-4b-v2.1-GGUF"
    decider_4b_port: int = 8084
    decider_4b_worker_port: int = 8081
    decider_4b_threads: int = 4
    decider_4b_device: str = "CPU"
    decider_4b_context_size: int = 32768
    decider_4b_enable_dom_navigation: bool = True
    decider_4b_enable_mcts_prior: bool = True

    # Strata Auxiliary Gating Exemption Rule
    strata_bypass_aux_models: bool = True

    # von tool-shortlisting configuration (legacy CLM-era clm_* key names retained)
    clm_shortlist_threshold: int = 15
    clm_shortlist_top_k: int = 5
    clm_shortlist_min_delta: float = 0.15
    clm_csls_k_neighbors: int = 5
    clm_enable_negation_masking: bool = True
    clm_pcore_affinity: bool = True

    @property
    def use_von_gater(self) -> bool:
        if not getattr(self, 'enable_von_gater', True) and not getattr(self, 'enable_cpu_onnx_gater', True):
            return False
        return str(getattr(self, 'von_model', '')).lower() not in ('none', 'false', 'off', 'disabled', '')

    @property
    def use_decider_4b(self) -> bool:
        if not getattr(self, 'enable_decider_4b', True):
            return False
        return str(getattr(self, 'decider_4b_model', '')).lower() not in ('none', 'false', 'off', 'disabled', '')

    @property
    def use_cpu_onnx_gater(self) -> bool:
        if not getattr(self, 'enable_cpu_onnx_gater', True) or not getattr(self, 'enable_cpu_decision', True):
            return False
        return str(getattr(self, 'cpu_onnx_gater_model', '')).lower() not in ('none', 'false', 'off', 'disabled', '')

    @property
    def use_cpu_decision(self) -> bool:
        if not getattr(self, 'enable_cpu_decision', True) and not getattr(self, 'enable_cpu_onnx_gater', True):
            return False
        return str(getattr(self, 'cpu_decision_model', '')).lower() not in ('none', 'false', 'off', 'disabled', '')

    # Automatic CPU Dispatch & Gating
    enable_auto_cpu_dispatch: bool = True

    # Real-time Local AI Model Loop Detection & Mitigation Module
    enable_loop_detection: bool = True
    loop_detector_ngram_size: int = 8
    loop_detector_ngram_threshold: int = 3
    loop_detector_cycle_similarity: float = 0.82
    loop_detector_min_sentence_len: int = 35
    loop_detector_max_history_words: int = 1500
    loop_recovery_max_retries: int = 3
    loop_recovery_temp_reduce: float = 0.1
    loop_recovery_temp_boost: float = 0.1
    loop_recovery_temp_floor: float = 0.15
    loop_recovery_presence_penalty: float = 0.0


    # Advanced Smart Dispatcher Configuration
    dispatcher_default_policy: str = 'auto'  # 'auto', 'fast_first', 'gpu_direct', 'cpu_only', 'hybrid_compact'
    dispatcher_confidence_threshold: float = 0.85
    dispatcher_max_cpu_tokens: int = 4000
    dispatcher_circuit_breaker_threshold: int = 3
    dispatcher_circuit_breaker_cooldown: float = 15.0
    dispatcher_compaction_threshold: int = 3500
    dispatcher_enable_compaction: bool = True
    dispatcher_batch_concurrency_limit: int = 4

    # GVS5H Autonomous Solver Log Retention
    gvs5h_max_log_runs: int = 100
            
    use_cuda: bool = True
    harness_start_dir_rel: str = "projects"
    harness_start_dir: Optional[Path] = None
    python_version: str = "3.11"

    compaction_enabled: bool = True
    pi_compaction_keep_ratio: float = 0.4
    hermes_compression_threshold_ratio: float = 0.5
    hermes_proactive_prune_ratio: float = 0.4

    pi_web_search_workflow: str = "auto-summary"
    pi_web_search_auto_open_browser: bool = False

    openlumara_repo_url: str = "https://github.com/Rose22/openlumara.git"
    openlumara_mode: str = "cli"

    # SearXNG Portable Metasearch Service (Port 8088 default)
    searxng_enabled: bool = True
    searxng_mode: str = "wsl_docker"  # "wsl_docker" (default, Tier 3), "native" (Tier 2), "remote" (Tier 1), "disabled"
    searxng_port: int = 8088
    searxng_host: str = "127.0.0.1"
    searxng_remote_url: str = "https://searx.be"
    searxng_upstream_engines: List[str] = field(default_factory=lambda: ["duckduckgo", "brave", "wikipedia", "google", "github"])
    searxng_max_results: int = 10
    searxng_categories: str = "general"
    searxng_auto_fallback_native: bool = True
    searxng_timeout: float = 8.0
    searxng_wsl_distro: str = "Ubuntu-26.04"
    searxng_wsl_container_name: str = "searxng"
    searxng_uwsgi_workers: int = 1
    searxng_uwsgi_threads: int = 4
    searxng_limiter: bool = False

    @property
    def searxng_effective_url(self) -> str:
        if str(getattr(self, "searxng_mode", "wsl_docker")).lower() == "remote":
            return str(getattr(self, "searxng_remote_url", "https://searx.be")).rstrip("/")
        host = getattr(self, "searxng_host", "127.0.0.1")
        port = getattr(self, "searxng_port", 8088)
        return f"http://{host}:{port}"

    # Aside Browser Interactive Automation Configuration
    aside_browser_enabled: bool = True
    aside_browser_model: str = "local-engine"
    aside_browser_bin: Optional[Path] = None
    aside_browser_auto_sync: bool = True
    aside_browser_headless: bool = False

    config_file: Optional[Path] = None

    def __post_init__(self):
        # Attach workspace paths
        self.hermes_dir = HERMES_DIR
        self.pi_dir = PI_DIR
        self.omp_dir = OMP_DIR
        self.dsh_dir = DSH_DIR
        self.t3_dir = T3_DIR
        self.orca_dir = ORCA_DIR
        self.orca_data_dir = ORCA_DATA_DIR
        self.github_dir = GITHUB_DIR
        self.agy_ide_dir = AGY_IDE_DIR
        self.agy_desktop_dir = AGY_DESKTOP_DIR
        self.openlumara_dir = OPENLUMARA_DIR
        self.localai_models_dir = LOCALAI_MODELS_DIR
        self.localai_repo_dir = LOCALAI_REPO_DIR
        self.localai_bin_dir = LOCALAI_BIN_DIR
        self.localai_exe = LOCALAI_EXE
        if self.harness_start_dir is None:
            self.harness_start_dir = PROJECTS_DIR

        # Apply user overrides from launcher.toml or launcher.json if available
        # When running under automated tests without an explicit config_file, skip loading workspace overrides
        is_test = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI") == "1" or "pytest" in sys.modules)
        if self.config_file is not None or not is_test:
            overrides = _flatten_overrides(_load_user_overrides(self.config_file))
            if "llamacpp_blackwell_engine" in overrides and "blackwell_engine" not in overrides:
                overrides["blackwell_engine"] = overrides["llamacpp_blackwell_engine"]
            for k, v in overrides.items():
                if hasattr(self, k):
                    setattr(self, k, self._sanitize_field(k, v))

        if hasattr(self, "blackwell_engine") and self.blackwell_engine:
            self.blackwell_engine = "bruno" if "bruno" in str(self.blackwell_engine).lower() else "huang"

        # Environment variables take precedence over config files
        if "LOCALAI_HEADLESS" in os.environ:
            self.localai_headless = os.environ.get("LOCALAI_HEADLESS", "").strip().lower() in ("1", "true", "yes")


        self._initialized = True
        self.claude_mode_explicit = "claude_mode" in overrides if (self.config_file is not None or not is_test) else False
        self.claude_desktop_mode_explicit = "claude_desktop_mode" in overrides if (self.config_file is not None or not is_test) else False

    def load_overrides(self, path: Path) -> None:
        """Explicitly load and apply overrides from a specific TOML or JSON file."""
        overrides = _flatten_overrides(_load_user_overrides(config_file=path))
        if "llamacpp_blackwell_engine" in overrides and "blackwell_engine" not in overrides:
            overrides["blackwell_engine"] = overrides["llamacpp_blackwell_engine"]
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, self._sanitize_field(k, v))
        if "claude_mode" in overrides:
            self.claude_mode_explicit = True
        if "claude_desktop_mode" in overrides:
            self.claude_desktop_mode_explicit = True
        if hasattr(self, "blackwell_engine") and self.blackwell_engine:
            self.blackwell_engine = "bruno" if "bruno" in str(self.blackwell_engine).lower() else "huang"

    def _sanitize_field(self, name: str, value: Any) -> Any:
        """Harden configuration against variables and flags with invalid or negative values."""
        cfg_logger = logging.getLogger("service_launcher")

        # 1. Non-negative integers (must be >= 0)
        if name in ("loop_recovery_max_retries", "loop_detector_min_sentence_len"):
            try:
                ival = int(value)
                if ival < 0:
                    cfg_logger.warning(
                        f"Config '{name}' cannot be negative (got {value}). Clamping to 0."
                    )
                    return 0
                return ival
            except (ValueError, TypeError):
                cfg_logger.warning(f"Config '{name}' received invalid integer ({value}). Keeping default.")
                return getattr(self, name, 0)

        # 2. Non-negative floats (must be >= 0.0)
        if name in ("loop_recovery_temp_reduce", "loop_recovery_temp_boost", "loop_recovery_temp_floor", "loop_recovery_presence_penalty"):
            try:
                fval = float(value)
                if fval < 0.0:
                    cfg_logger.warning(
                        f"Config '{name}' cannot be negative (got {value}). Clamping to 0.0."
                    )
                    return 0.0
                return fval
            except (ValueError, TypeError):
                cfg_logger.warning(f"Config '{name}' received invalid float ({value}). Keeping default.")
                return getattr(self, name, 0.0)

        # 3. Positive integers with safe defaults (must be >= 1)
        positive_int_defaults = {
            "loop_detector_ngram_size": 8,
            "loop_detector_ngram_threshold": 3,
            "loop_detector_max_history_words": 1500,
            "gvs5h_max_log_runs": 100,
        }
        if name in positive_int_defaults:
            default_val = positive_int_defaults[name]
            try:
                ival = int(value)
                if ival < 1:
                    cfg_logger.warning(
                        f"Config '{name}' must be >= 1 (got {value}). Resetting to default {default_val}."
                    )
                    return default_val
                return ival
            except (ValueError, TypeError):
                return getattr(self, name, default_val)

        if name == "loop_detector_min_sentence_len":
            try:
                ival = int(value)
                return max(0, ival)
            except (ValueError, TypeError):
                return getattr(self, name, 35)

        if name == "searxng_max_results":
            try:
                ival = int(value)
                if ival < 1:
                    return 10
                return min(ival, 50)
            except (ValueError, TypeError):
                return getattr(self, name, 10)

        # Huang Adaptive-KV tunables
        if name == "huang_arena_mib":
            try:
                ival = int(value)
                if ival < 128:
                    cfg_logger.warning(f"Config '{name}' ({value} MiB) is below minimum safe threshold. Clamping to 128 MiB.")
                    return 128
                if ival > 65536:
                    cfg_logger.warning(f"Config '{name}' ({value} MiB) exceeds 64 GiB. Clamping to 65536 MiB.")
                    return 65536
                return ival
            except (ValueError, TypeError):
                return getattr(self, name, 2240)

        if name == "huang_draft_max":
            try:
                ival = int(value)
                if ival < 1:
                    cfg_logger.warning(f"Config '{name}' must be >= 1. Clamping to 1.")
                    return 1
                if ival > 16:
                    cfg_logger.warning(f"Config '{name}' exceeds 16 draft tokens. Clamping to 16.")
                    return 16
                return ival
            except (ValueError, TypeError):
                return getattr(self, name, 3)

        if name == "huang_spec_type":
            val_s = str(value).lower().strip()
            if val_s not in ("draft-mtp", "draft", "none", "mtp"):
                cfg_logger.warning(f"Invalid huang_spec_type '{value}'. Falling back to 'draft-mtp'.")
                return "draft-mtp"
            return val_s

        if name in ("huang_draft_p_min", "huang_draft_p_split"):
            default_val = 0.20 if name == "huang_draft_p_min" else 0.10
            try:
                fval = float(value)
                if fval < 0.0:
                    cfg_logger.warning(f"Config '{name}' cannot be negative. Clamping to 0.0.")
                    return 0.0
                if fval > 1.0:
                    cfg_logger.warning(f"Config '{name}' exceeds 1.0. Clamping to 1.0.")
                    return 1.0
                return fval
            except (ValueError, TypeError):
                return getattr(self, name, default_val)

        if name == "huang_auxiliary_layers":
            try:
                ival = int(value)
                if ival < 0:
                    cfg_logger.warning(f"Config '{name}' cannot be negative. Clamping to 0.")
                    return 0
                if ival > 16:
                    cfg_logger.warning(f"Config '{name}' exceeds 16 auxiliary layers. Clamping to 16.")
                    return 16
                return ival
            except (ValueError, TypeError):
                return getattr(self, name, 1)

        if name in ("huang_batch_size", "huang_ubatch_size"):
            default_val = 512 if name == "huang_batch_size" else 256
            try:
                ival = int(value)
                if ival < 1:
                    cfg_logger.warning(f"Config '{name}' must be >= 1. Resetting to {default_val}.")
                    return default_val
                if ival > 16384:
                    cfg_logger.warning(f"Config '{name}' exceeds 16384. Clamping to 16384.")
                    return 16384
                return ival
            except (ValueError, TypeError):
                return getattr(self, name, default_val)

        if name == "huang_parallel":
            try:
                ival = int(value)
                if ival < 1:
                    cfg_logger.warning(f"Config '{name}' must be >= 1. Clamping to 1.")
                    return 1
                if ival > 16:
                    cfg_logger.warning(f"Config '{name}' exceeds 16 parallel slots. Clamping to 16.")
                    return 16
                return ival
            except (ValueError, TypeError):
                return getattr(self, name, 1)

        if name == "huang_no_mmproj":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes", "on")

        # 4. Network ports (1 <= port <= 65535)
        port_defaults = {
            "localai_port": 8080,
            "blackwell_shim_port": 9940,
            "tabbyapi_shim_port": 9941,
            "freetoken_shim_port": 9942,
            "strata_shim_port": 9943,
            "mcp_server_port": 9932,
            "t3_port": 3773,
            "cpu_llama_server_port": 8081,
            "cpu_decision_port": 8083,
            "cpu_reranker_port": 8082,
            "cpu_onnx_gater_port": 8083,
            "signal_daemon_port": 8085,
            "litellm_port": 4000,
            "searxng_port": 8088,
        }
        if name in port_defaults:
            default_port = port_defaults[name]
            try:
                ival = int(value)
                if ival < 1 or ival > 65535:
                    cfg_logger.warning(
                        f"Config port '{name}' must be between 1 and 65535 (got {value}). Resetting to default {default_port}."
                    )
                    return default_port
                return ival
            except (ValueError, TypeError):
                return getattr(self, name, default_port)

        # 5. Unit interval ratios (0.0 <= ratio <= 1.0)
        if name in ("pi_compaction_keep_ratio", "hermes_compression_threshold_ratio", "hermes_proactive_prune_ratio"):
            try:
                fval = float(value)
                if fval < 0.0:
                    cfg_logger.warning(
                        f"Config ratio '{name}' cannot be negative (got {value}). Clamping to 0.0."
                    )
                    return 0.0
                elif fval > 1.0:
                    cfg_logger.warning(
                        f"Config ratio '{name}' cannot exceed 1.0 (got {value}). Clamping to 1.0."
                    )
                    return 1.0
                return fval
            except (ValueError, TypeError):
                return getattr(self, name, 0.5)

        # 6. Cycle similarity (must be between 0.0 and 1.0, strictly positive)
        if name == "loop_detector_cycle_similarity":
            try:
                fval = float(value)
                if fval <= 0.0 or fval > 1.0:
                    cfg_logger.warning(
                        f"Config 'loop_detector_cycle_similarity' must be in range (0.0, 1.0] (got {value}). Resetting to default 0.82."
                    )
                    return 0.82
                return fval
            except (ValueError, TypeError):
                return getattr(self, name, 0.82)

        # 7. Boolean flags against negative numbers & truthiness traps
        boolean_flags = {
            "localai_single_active_backend",
            "localai_watchdog_idle",
            "localai_headless",
            "enable_loop_detection",
            "enable_cpu_reranker",
            "enable_cpu_onnx_gater",
            "enable_cpu_decision",
            "enable_cpu_llama_server",
            "enable_auto_cpu_dispatch",
            "use_omniroute_proxy",
            "run_mnemosyne_verification",
            "show_launcher_logging",
            "show_llama_server_console",
            "preload_default_model",
            "enable_context_preloading",
            "skip_hermes_web_ui_build",
            "create_ai_bot_user",
            "start_signal_daemon",
            "start_discord_gateway",
            "orca_voice_to_text",
            "use_cuda",
            "compaction_enabled",
            "pi_web_search_auto_open_browser",
            "force_update",
            "strata_mmap_experts",
            "strata_use_custom_chat_template",
            "local_claude_native_use_custom_chat_template",
            "strata_system_ram_kv_cache",
            "use_litellm_proxy",
            "claude_disallow_cloud_search",
            "claude_disallow_cloud_search_online",
            "aside_browser_enabled",
            "aside_browser_auto_sync",
            "aside_browser_headless",
        }
        if name in boolean_flags:
            if isinstance(value, (int, float)):
                if value < 0:
                    cfg_logger.warning(
                        f"Config flag '{name}' received negative numeric value ({value}). Coercing to False."
                    )
                    return False
                return bool(value)
            if isinstance(value, str):
                s = value.strip().lower()
                if s in ("false", "0", "-1", "off", "no", "disable", "disabled"):
                    return False
                if s in ("true", "1", "yes", "on", "enable", "enabled"):
                    return True
                try:
                    num = float(s)
                    if num <= 0:
                        cfg_logger.warning(
                            f"Config flag '{name}' received non-positive numeric string '{value}'. Coercing to False."
                        )
                        return False
                    return True
                except ValueError:
                    pass
                return bool(value)

        # 8. Strata KV cache quantization mode validation
        if name == "strata_kv_mode":
            val_s = str(value).lower().strip()
            valid_modes = ("int8", "q4_0", "k8v4", "fp16")
            if val_s not in valid_modes:
                cfg_logger.warning(
                    f"Invalid strata_kv_mode '{value}'. Valid options are {valid_modes}. Falling back to 'int8'."
                )
                return "int8"
            return val_s

        # 8b. Strata resident budget in GiB validation
        if name == "strata_resident_budget_gib":
            if value is None or value == "" or str(value).lower() in ("none", "null", "false"):
                return 0.0
            try:
                val_f = float(value)
                return max(0.0, val_f)
            except (ValueError, TypeError):
                cfg_logger.warning(
                    f"Invalid strata_resident_budget_gib '{value}'. Falling back to 52.0."
                )
                return 52.0

        # 8c. Strata brutal RAM clawback mode validation ('off', 'light', 'max')
        if name == "strata_brutal_ram_clawback":
            if isinstance(value, bool):
                return "max" if value else "off"
            s = str(value).strip().lower()
            if s in ("max", "brutal", "full", "heavy", "true", "1"):
                return "max"
            if s in ("off", "none", "disabled", "disable", "false", "0"):
                return "off"
            if s in ("light", "gentle", "soft"):
                return "light"
            cfg_logger.warning(
                f"Invalid strata_brutal_ram_clawback '{value}'. Valid options are ('off', 'light', 'max'). Falling back to 'light'."
            )
            return "light"

        # 8d. Strata KV resident window validation (clamped to >= 20480 if > 0)
        if name == "strata_kv_resident":
            if value is None or str(value).lower() in ("none", "null", ""):
                return None
            try:
                ival = int(value)
                return max(20480, ival) if ival > 0 else 0
            except (ValueError, TypeError):
                return None

        # 8e. Strata VRAM reserve in MiB validation
        if name == "strata_vram_reserve_mib":
            if value is None or str(value).lower() in ("none", "null", ""):
                return None
            try:
                ival = int(value)
                return max(0, ival)
            except (ValueError, TypeError):
                return None

        # 9. Update frequency configuration sanitization
        if name == "update_frequency_hrs":
            from local_ai_launcher.update_frequency import parse_frequency_hours
            hrs = parse_frequency_hours(value, default=168.0)
            if hrs == float("inf"):
                return float("inf")
            return int(hrs) if hrs == int(hrs) else float(hrs)

        if name in ("harness_update_frequency_hrs", "engine_update_frequency_hrs"):
            from local_ai_launcher.update_frequency import parse_frequency_hours
            if isinstance(value, dict):
                res = {}
                for k, v in value.items():
                    hrs = parse_frequency_hours(v, default=168.0)
                    res[str(k).lower()] = float("inf") if hrs == float("inf") else (int(hrs) if hrs == int(hrs) else float(hrs))
                return res
            return getattr(self, name, {})

        # 10. Claude Code and Claude Desktop mode sanitization
        if name == "claude_desktop_mode":
            s = str(value).strip().lower()
            if s in ("mcp", "bridge", "mcp_bridge"):
                return "mcp"
            if s in ("gateway", "3p", "developer", "dev"):
                return "gateway"
            if s in ("off", "disabled", "none", "false", "0"):
                return "off"
            cfg_logger.warning(
                f"Invalid claude_desktop_mode '{value}'. Valid options are ('mcp', 'gateway', 'off'). Falling back to 'mcp'."
            )
            return "mcp"

        if name == "local_claude_native_engines":
            if isinstance(value, str):
                items = [x.strip().lower() for x in value.replace(",", " ").split() if x.strip()]
            elif isinstance(value, (list, tuple, set)):
                items = [str(x).strip().lower() for x in value if str(x).strip()]
            else:
                items = ["strata"]
            if "strata" not in items:
                items.append("strata")
            return items

        if name == "claude_mode":
            s = str(value).strip().lower()
            if s in ("local", "offline", "zero", "strata", "local_claude_native", "claude_native"):
                return "local"
            if s in ("official", "anthropic", "cloud", "online", "paid"):
                return "official"
            cfg_logger.warning(
                f"Invalid claude_mode '{value}'. Valid options are ('local', 'official'). Falling back to 'local'."
            )
            return "local"

        if name in ("claude_disallow_cloud_search", "claude_disallow_cloud_search_online"):
            return bool(value)

        if name == "claude_auto_compact_ratio":
            try:
                val = float(value)
                return max(0.10, min(val, 0.99))
            except (ValueError, TypeError):
                return 0.91

        if name == "claude_desktop_auto_compact_ratio":
            if value is None or str(value).strip().lower() in ("none", "auto", "null", ""):
                return None
            try:
                val = float(value)
                return max(0.10, min(val, 0.99))
            except (ValueError, TypeError):
                return None

        if name in (
            "claude_max_context_tokens", "claude_auto_compact_window",
            "claude_desktop_max_context_tokens", "claude_desktop_auto_compact_window"
        ):
            if value is None or str(value).strip().lower() in ("none", "auto", "null", ""):
                return None
            try:
                val_i = int(value)
                return max(512, val_i)
            except (ValueError, TypeError):
                return None

        if name == "litellm_install_behavior":
            s = str(value).strip().lower()
            return s if s in ("always_install", "never_install", "existing") else "always_install"

        if name == "litellm_update_behavior":
            s = str(value).strip().lower()
            return s if s in ("always_update", "never_update", "existing") else "never_update"

        return value


    def _normalize_gater_alias(self, value: Any) -> Any:
        """
        Canonical gater name is 'wfzyx/von'. Legacy deprecated model names
        (open-jev-deberta-v3-large, clm-8b) are accepted as aliases and
        normalized to von at ingest so downstream code only ever sees von.
        Disabling sentinels ('none', '', etc.) pass through untouched.
        """
        s = str(value or "").strip().lower()
        if any(k in s for k in ("open-jev", "openjev", "deberta", "clm-8b", "clm-v0.1", "contrastive-lm", "stanford-nvidia")):
            return "wfzyx/von"
        return value

    def __setattr__(self, name: str, value: Any):
        value = self._sanitize_field(name, value)
        super().__setattr__(name, value)
        if name == "local_claude_native_use_custom_chat_template":
            super().__setattr__("strata_use_custom_chat_template", bool(value))
        elif name == "strata_use_custom_chat_template":
            super().__setattr__("local_claude_native_use_custom_chat_template", bool(value))
        elif name == "loop_recovery_temp_reduce":
            super().__setattr__("loop_recovery_temp_boost", value)
        elif name == "loop_recovery_temp_boost":
            super().__setattr__("loop_recovery_temp_reduce", value)
        elif name == "von_model":
            value = self._normalize_gater_alias(value)
            super().__setattr__("von_model", value)
            super().__setattr__("cpu_onnx_gater_model", value)
            super().__setattr__("cpu_decision_model", value)
        elif name in ("cpu_onnx_gater_model", "cpu_decision_model"):
            value = self._normalize_gater_alias(value)
            super().__setattr__(name, value)
            if name == "cpu_decision_model":
                if value in ("none", "false", "off", "disabled", ""):
                    super().__setattr__("cpu_onnx_gater_model", value)
                elif getattr(self, "cpu_onnx_gater_model", "") in ("wfzyx/von", "", "none"):
                    super().__setattr__("cpu_onnx_gater_model", value)
        elif name in ("cpu_decision_threads", "von_threads"):
            super().__setattr__("cpu_onnx_gater_threads", int(value))
            super().__setattr__("von_threads", int(value))
        elif name in ("cpu_decision_port", "von_port"):
            super().__setattr__("cpu_onnx_gater_port", int(value))
            super().__setattr__("von_port", int(value))
        elif name == "enable_cpu_decision":
            super().__setattr__("enable_cpu_onnx_gater", bool(value))
        elif name == "enable_cpu_onnx_gater":
            super().__setattr__("enable_cpu_decision", bool(value))
        elif name == "enable_von_gater":
            super().__setattr__("enable_cpu_onnx_gater", bool(value))


# Singleton launcher configuration instance
config = LauncherConfig()

# Primary API spec ports
OPENAI_API_SPEC_PORT = config.localai_port
LLM_BACKEND_PORT = config.localai_port

# Setup logging configuration for Mnemosyne verifier & launcher
log_level = logging.INFO if config.show_launcher_logging else logging.WARNING
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

mnemosyne_logger = logging.getLogger("mnemosyne_verifier")
logger = logging.getLogger("service_launcher")


@dataclass
class MnemosyneAppConfig:
    master_path: Path = CONFIG_DIR / "USER.md"
    backup_dir: Path = CONFIG_DIR / "backup"
    projects_dir: Path = PROJECTS_DIR
    hermes_config_dir: Path = HERMES_DIR
    pi_config_dir: Path = PI_DIR
    omp_config_dir: Path = OMP_DIR
    t3_config_dir: Path = T3_DIR
    orca_config_dir: Path = ORCA_DIR
    agy_ide_config_dir: Path = AGY_IDE_DIR
    agy_desktop_config_dir: Path = AGY_DESKTOP_DIR
    claude_config_dir: Path = CLAUDE_DIR
    claude_local_config_dir: Path = CLAUDE_LOCAL_DIR
    claude_desktop_config_dir: Path = CLAUDE_DESKTOP_DIR
    mnemosyne_db_path: Path = CONFIG_DIR / "mnemosyne" / "memory.db"
    
    @property
    def hermes_dir(self) -> Path:
        return self.hermes_config_dir

    @property
    def pi_dir(self) -> Path:
        return self.pi_config_dir

    @property
    def omp_dir(self) -> Path:
        return self.omp_config_dir

    @property
    def t3_dir(self) -> Path:
        return self.t3_config_dir

    @property
    def orca_dir(self) -> Path:
        return self.orca_config_dir

    @property
    def orca_data_dir(self) -> Path:
        return AI_WORKSPACE_DIR / "orca" / "data"
        
    @property
    def agy_ide_dir(self) -> Path:
        return self.agy_ide_config_dir
        
    @property
    def agy_desktop_dir(self) -> Path:
        return self.agy_desktop_config_dir

    @property
    def claude_dir(self) -> Path:
        return self.claude_config_dir

    @property
    def claude_local_dir(self) -> Path:
        custom = getattr(self.cfg, "claude_local_dir", None)
        if custom is not None and custom != CLAUDE_LOCAL_DIR:
            return custom
        if self.claude_dir != CLAUDE_DIR:
            if self.claude_dir.name == ".claude":
                return self.claude_dir.parent / ".cbox"
            return self.claude_dir / ".cbox"
        return self.claude_local_config_dir

    @property
    def claude_desktop_dir(self) -> Path:
        return self.claude_desktop_config_dir
    
    backup_interval_days: int = 5
    max_backups_to_keep: int = 10
    install_mnemosyne: str = "install"

    def __setattr__(self, name: str, value: Any):
        if name in ("backup_interval_days", "max_backups_to_keep"):
            default_val = 5 if name == "backup_interval_days" else 10
            try:
                ival = int(value)
                if ival < 1:
                    logging.getLogger("service_launcher").warning(
                        f"Mnemosyne config '{name}' must be >= 1 (got {value}). Clamping to {default_val}."
                    )
                    value = default_val
                else:
                    value = ival
            except (ValueError, TypeError):
                value = default_val
        super().__setattr__(name, value)
