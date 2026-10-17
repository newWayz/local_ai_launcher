import subprocess
import time
import os
import shutil
from pathlib import Path
import json
import webbrowser
from typing import Optional

from local_ai_launcher.config import *
from local_ai_launcher.environment import *
from local_ai_launcher.models import *
from local_ai_launcher.process import *
from local_ai_launcher.harness_base import HarnessAdapter


def _resolve_opencode_executable() -> Path:
    """Find the OpenCode executable used as the T3 Code local agent driver."""
    local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
    candidates = [
        AI_WORKSPACE_DIR / "node_modules" / "opencode-ai" / "bin" / "opencode.exe",
        local_bin / "opencode.cmd",
        AI_WORKSPACE_DIR / "go" / "bin" / "opencode.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    which_path = shutil.which("opencode")
    if which_path:
        return Path(which_path)
    return candidates[0]


def setup_t3_config(t3_home: Path, llm_base_url: str, active_model: Optional[str] = None) -> None:
    """Configure T3 Code settings and local workspace state for local LLM integration."""
    t3_home.mkdir(parents=True, exist_ok=True)
    model_name = resolve_model_name(active_model)
    raw_ctx_size, is_reasoning = get_model_specs(model_name)
    try:
        ctx_size = max(512, int(raw_ctx_size))
    except (ValueError, TypeError):
        ctx_size = 32768
    # Hardware-proportional context output allocation for high-end GPUs (e.g. RTX 5080)
    max_output_tokens = min(ctx_size, max(8192, ctx_size // 8))

    preset_models = [m for m in get_available_preset_models(only_installed=True, active_model=model_name) if m.get("id") != "*"]
    models_list = []
    matched = next((m for m in preset_models if m["id"] == model_name), None)
    if matched:
        models_list.append(matched)
    else:
        models_list.append({
            "id": model_name,
            "name": f"Local Model ({model_name})",
            "contextWindow": ctx_size,
            "maxTokens": max_output_tokens,
            "input": ["text", "image"] if any(k in model_name.lower() for k in ("vision", "-vl", "_vl", "mmproj")) else ["text"]
        })

    for m in preset_models:
        if not any(existing["id"] == m["id"] for existing in models_list):
            models_list.append(m)

    try:
        t3_port = int(getattr(config, 't3_port', 3773))
        if not (1 <= t3_port <= 65535):
            t3_port = 3773
    except (ValueError, TypeError):
        t3_port = 3773

    opencode_exe = _resolve_opencode_executable()

    # 1. Ensure Pi and Hermes harnesses are fully configured and synced for T3 Code sessions
    try:
        from local_ai_launcher.harnesses.pi import setup_pi_models_and_auth
        setup_pi_models_and_auth(
            config.pi_dir,
            llm_base_url,
            active_model=model_name,
            use_omniroute=config.use_omniroute_proxy,
            omniroute_model=config.omniroute_default_model
        )
        logger.info("  -> Synced workspace Pi models and auth for T3 Code integration.")
    except Exception as pi_err:
        logger.warning(f"  -> Warning: Pi models and auth configuration notice for T3: {pi_err}")

    try:
        from local_ai_launcher.harnesses.hermes import HermesAdapter
        hermes_adapter = HermesAdapter(AI_WORKSPACE_DIR, [])
        hermes_adapter._ensure_user_profile_junction()
        from local_ai_launcher.bridges.gvs5h_solver import ensure_deep_solve_skills
        ensure_deep_solve_skills([config.hermes_dir / "skills" / "deep-solve"])
        from local_ai_launcher.bridges.cpu_tools_cli import ensure_cpu_tools_skills
        ensure_cpu_tools_skills([config.hermes_dir / "skills" / "cpu-tools"])
        logger.info("  -> Synced Hermes profile and auxiliary skills for T3 Code integration.")
    except Exception as hermes_err:
        logger.warning(f"  -> Warning: Hermes configuration notice for T3: {hermes_err}")

    try:
        from local_ai_launcher.harnesses.omp import setup_omp_models_and_auth
        setup_omp_models_and_auth(
            getattr(config, "omp_dir", AI_WORKSPACE_DIR / ".omp"),
            llm_base_url,
            active_model=model_name,
            use_omniroute=config.use_omniroute_proxy,
            omniroute_model=config.omniroute_default_model
        )
        logger.info("  -> Synced workspace OMP models and auth for T3 Code integration.")
    except Exception as omp_err:
        logger.warning(f"  -> Warning: OMP models and auth configuration notice for T3: {omp_err}")

    # Synchronize USER.md into T3 directory for global reference
    user_md_src = CONFIG_DIR / "USER.md"
    if user_md_src.exists():
        try:
            shutil.copy2(user_md_src, t3_home / "USER.md")
        except Exception as e:
            logger.debug(f"Could not copy USER.md to t3_home: {e}")

    from local_ai_launcher.harness_model import resolve_meta_harness_inner_cli_harness
    inner_cli = resolve_meta_harness_inner_cli_harness(
        getattr(config, 'choose_harness', []),
        model_name=model_name,
        cfg=config
    )
    # When Strata engine is active and Claude is resolved, default to claude; otherwise hermes/omp if resolved, else pi
    if inner_cli == "claude":
        default_harness = "claude"
    elif inner_cli == "hermes":
        default_harness = "hermes"
    elif inner_cli in ("omp", "oh-my-pi", "ohmypi"):
        default_harness = "omp"
    else:
        default_harness = "pi"

    # 2. Configure OpenCode provider and delegation commands for Pi and Hermes harnesses
    opencode_models_dict = {}
    for m in models_list:
        m_id = m.get("id")
        if m_id:
            has_vision = "image" in m.get("input", [])
            m_ctx = m.get("contextWindow", 131072)
            spec = {
                "name": m.get("name") or m_id,
                "attachment": has_vision,
                "modalities": {
                    "input": ["text", "image"] if has_vision else ["text"],
                    "output": ["text"]
                },
                "limit": {
                    "context": m_ctx,
                    "output": min(m_ctx, max(8192, m_ctx // 8))
                }
            }
            if "think" in m_id.lower() or "reasoning" in m_id.lower():
                spec["reasoning"] = True
            opencode_models_dict[m_id] = spec

    # OpenCode requires custom OpenAI-compatible providers to specify:
    # 1. "npm": "@ai-sdk/openai-compatible" so OpenCode knows the provider protocol
    # 2. "options": { "baseURL": ... } so OpenCode does not try to request "undefined/chat/completions"
    pi_provider_spec = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Pi Coding Harness",
        "options": {
            "baseURL": llm_base_url,
            "apiKey": "dummy"
        },
        "baseURL": llm_base_url,
        "models": opencode_models_dict
    }

    omp_provider_spec = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Oh My Pi",
        "options": {
            "baseURL": llm_base_url,
            "apiKey": "dummy"
        },
        "baseURL": llm_base_url,
        "models": opencode_models_dict
    }

    hermes_provider_spec = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Hermes",
        "options": {
            "baseURL": llm_base_url,
            "apiKey": "dummy"
        },
        "baseURL": llm_base_url,
        "models": opencode_models_dict
    }

    localai_provider_spec = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "LocalAI Server",
        "options": {
            "baseURL": llm_base_url,
            "apiKey": "dummy"
        },
        "baseURL": llm_base_url,
        "models": opencode_models_dict
    }

    claude_provider_spec = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Claude Code",
        "options": {
            "baseURL": llm_base_url,
            "apiKey": "dummy"
        },
        "baseURL": llm_base_url,
        "models": opencode_models_dict
    }

    opencode_config_data = {
        "$schema": "https://opencode.ai/config.json",
        "disabled_providers": ["openai", "opencode", "anthropic", "google"],
        "provider": {
            "pi": pi_provider_spec,
            "omp": omp_provider_spec,
            "hermes": hermes_provider_spec
        },
        "model": f"{default_harness}/{model_name}",
        "command": {
            "pi": {
                "description": "Delegate coding tasks to Pi coding agent",
                "template": f"Use Pi coding agent (installed at {(AI_WORKSPACE_DIR / 'node' / 'pi.ps1').as_posix()}) to complete this task: $ARGUMENTS"
            },
            "omp": {
                "description": "Delegate coding tasks to Oh My Pi coding agent",
                "template": f"Use Oh My Pi coding agent to complete this task: $ARGUMENTS"
            },
            "hermes": {
                "description": "Delegate tasks to Hermes agent stack",
                "template": f"Use Hermes agent (installed at {(AI_WORKSPACE_DIR / '.hermes' / 'bin' / 'hermes.exe').as_posix()}) to complete this task: $ARGUMENTS"
            }
        },
        "agent": {
            "pi": {
                "description": "Pi Coding Agent Harness",
                "mode": "subagent"
            },
            "omp": {
                "description": "Oh My Pi Coding Agent Harness",
                "mode": "subagent"
            },
            "hermes": {
                "description": "Hermes Coding Agent Stack",
                "mode": "subagent"
            }
        }
    }

    opencode_config_data["provider"]["claude"] = claude_provider_spec
    opencode_config_data["command"]["claude"] = {
        "description": "Delegate coding tasks to Claude Code CLI agent",
        "template": f"Use Claude Code CLI agent to complete this task: $ARGUMENTS"
    }
    opencode_config_data["agent"]["claude"] = {
        "description": "Claude Code CLI Agent",
        "mode": "subagent"
    }

    from local_ai_launcher.paths import atomic_write_json

    try:
        user_opencode_dir = Path.home() / ".config" / "opencode"
        user_opencode_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(user_opencode_dir / "opencode.jsonc", opencode_config_data)
        atomic_write_json(AI_WORKSPACE_DIR / "opencode.json", opencode_config_data)
        logger.info(f"  -> Configured OpenCode harness providers in {user_opencode_dir / 'opencode.jsonc'}")
    except Exception as e:
        logger.warning(f"  -> Warning: Could not write OpenCode config: {e}")

    # 2b. Generate harness-isolated OpenCode configs so probes stay strictly partitioned
    # Setting enabled_providers ensures that each harness only registers its own provider.
    opencode_pi_config = {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": ["pi"],
        "disabled_providers": ["openai", "opencode", "anthropic", "google", "localai", "hermes", "claude"],
        "provider": {
            "pi": dict(pi_provider_spec, name=" ")
        },
        "model": f"pi/{model_name}"
    }
    opencode_hermes_config = {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": ["hermes"],
        "disabled_providers": ["openai", "opencode", "anthropic", "google", "localai", "pi", "claude"],
        "provider": {
            "hermes": dict(hermes_provider_spec, name=" ")
        },
        "model": f"hermes/{model_name}"
    }
    pi_config_file = t3_home / "opencode_pi.json"
    omp_config_file = t3_home / "opencode_omp.json"
    hermes_config_file = t3_home / "opencode_hermes.json"
    opencode_omp_config = {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": ["omp"],
        "disabled_providers": ["openai", "opencode", "anthropic", "google", "localai", "pi", "hermes", "claude"],
        "provider": {
            "omp": dict(omp_provider_spec, name=" ")
        },
        "model": f"omp/{model_name}"
    }
    atomic_write_json(pi_config_file, opencode_pi_config)
    atomic_write_json(omp_config_file, opencode_omp_config)
    atomic_write_json(hermes_config_file, opencode_hermes_config)

    claude_config_file = t3_home / "opencode_claude.json"
    opencode_claude_config = {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": ["claude"],
        "disabled_providers": ["openai", "opencode", "anthropic", "google", "localai", "pi", "omp", "hermes"],
        "provider": {
            "claude": dict(claude_provider_spec, name=" ")
        },
        "model": f"claude/{model_name}"
    }
    atomic_write_json(claude_config_file, opencode_claude_config)

    # 3. Global User Data Configuration
    # Uses T3CODE_BASE_DIR if set, otherwise defaults to workspace-contained directory
    t3_base_dir_env = os.environ.get("T3CODE_BASE_DIR")
    if t3_base_dir_env:
        global_t3_dir = Path(t3_base_dir_env)
    else:
        global_t3_dir = AI_WORKSPACE_DIR / ".t3" / "userdata"

    try:
        real_user_t3_dir = Path.home() / ".t3" / "userdata"
    except Exception:
        real_user_t3_dir = AI_WORKSPACE_DIR / ".t3" / "userdata"

    # Separate XDG data/cache/state directories for each harness instance so that concurrent
    # OpenCode processes never collide or lock the shared SQLite database (opencode.db).
    pi_data_dir = t3_home / "opencode_data" / "pi"
    omp_data_dir = t3_home / "opencode_data" / "omp"
    hermes_data_dir = t3_home / "opencode_data" / "hermes"
    pi_cache_dir = t3_home / "opencode_cache" / "pi"
    omp_cache_dir = t3_home / "opencode_cache" / "omp"
    hermes_cache_dir = t3_home / "opencode_cache" / "hermes"
    pi_state_dir = t3_home / "opencode_state" / "pi"
    omp_state_dir = t3_home / "opencode_state" / "omp"
    hermes_state_dir = t3_home / "opencode_state" / "hermes"

    for d in [pi_data_dir, omp_data_dir, hermes_data_dir, pi_cache_dir, omp_cache_dir, hermes_cache_dir, pi_state_dir, omp_state_dir, hermes_state_dir]:
        d.mkdir(parents=True, exist_ok=True)

    local_env = [
        {"name": "OPENAI_BASE_URL", "value": llm_base_url},
        {"name": "OPENAI_API_BASE", "value": llm_base_url},
        {"name": "OPENAI_MODEL", "value": model_name},
        {"name": "MODEL", "value": model_name},
        {"name": "HOME", "value": str(AI_WORKSPACE_DIR)},
        {"name": "USERPROFILE", "value": str(AI_WORKSPACE_DIR)},
        {"name": "PI_HOME", "value": str(AI_WORKSPACE_DIR / ".pi")},
        {"name": "PI_CODING_AGENT_DIR", "value": str(AI_WORKSPACE_DIR / ".pi" / "agent")},
        {"name": "PI_SKIP_VERSION_CHECK", "value": "1"},
        {"name": "HERMES_HOME", "value": str(AI_WORKSPACE_DIR / ".hermes")},
        {"name": "HF_HOME", "value": str(AI_WORKSPACE_DIR / "models" / "huggingface")},
        {"name": "HF_HUB_CACHE", "value": str(AI_WORKSPACE_DIR / "models" / "huggingface")},
        {"name": "npm_config_allow_git", "value": "all"},
    ]

    if getattr(config, "aside_browser_enabled", True):
        try:
            from local_ai_launcher.paths import get_aside_binary_path
            aside_bin = getattr(config, "aside_browser_bin", None) or get_aside_binary_path()
            if aside_bin and Path(aside_bin).is_file():
                local_env.append({"name": "ASIDE_CLI_DIR", "value": str(Path(aside_bin).parent)})
                local_env.append({"name": "ASIDE_BINARY", "value": str(aside_bin)})
        except Exception:
            pass

    pi_env = list(local_env) + [
        {"name": "OPENCODE_CONFIG", "value": pi_config_file.as_posix()},
        {"name": "XDG_DATA_HOME", "value": pi_data_dir.as_posix()},
        {"name": "XDG_CACHE_HOME", "value": pi_cache_dir.as_posix()},
        {"name": "XDG_STATE_HOME", "value": pi_state_dir.as_posix()},
    ]
    omp_env = list(local_env) + [
        {"name": "OMP_HOME", "value": str(AI_WORKSPACE_DIR / ".omp")},
        {"name": "PI_CONFIG_DIR", "value": ".omp"},
        {"name": "PI_CODING_AGENT_DIR", "value": str(AI_WORKSPACE_DIR / ".omp" / "agent")},
        {"name": "OMP_SKIP_SETUP", "value": "1"},
        {"name": "OPENCODE_CONFIG", "value": omp_config_file.as_posix()},
        {"name": "XDG_DATA_HOME", "value": omp_data_dir.as_posix()},
        {"name": "XDG_CACHE_HOME", "value": omp_cache_dir.as_posix()},
        {"name": "XDG_STATE_HOME", "value": omp_state_dir.as_posix()},
    ]
    hermes_env = list(local_env) + [
        {"name": "OPENCODE_CONFIG", "value": hermes_config_file.as_posix()},
        {"name": "XDG_DATA_HOME", "value": hermes_data_dir.as_posix()},
        {"name": "XDG_CACHE_HOME", "value": hermes_cache_dir.as_posix()},
        {"name": "XDG_STATE_HOME", "value": hermes_state_dir.as_posix()},
    ]

    from local_ai_launcher.harnesses.claude import (
        is_claude_offline_mode_active,
        ensure_claude_local_profile,
        get_claude_local_env,
        resolve_claude_permission_mode,
        resolve_claude_context_and_compact,
        ClaudeAdapter,
    )
    claude_offline = is_claude_offline_mode_active(active_model)
    claude_adapter = ClaudeAdapter(AI_WORKSPACE_DIR, [])
    claude_exe_path = claude_adapter._resolve_executable() or shutil.which("claude") or "claude"

    if claude_offline:
        claude_local_dir = ensure_claude_local_profile(llm_base_url, active_model)
        claude_local_env = get_claude_local_env(llm_base_url, active_model)
        eff_perm_mode = resolve_claude_permission_mode(is_desktop=False)
        claude_launch_args = "--dangerously-skip-permissions" if eff_perm_mode == "bypassPermissions" else ""
        _, auto_compact = resolve_claude_context_and_compact(model_name, is_desktop=False)
        auto_compact_str = str(auto_compact) if (auto_compact and 100000 <= auto_compact <= 1000000) else ""
    else:
        claude_local_dir = None
        claude_local_env = {}
        claude_launch_args = ""
        auto_compact_str = ""

    for target_dir in [global_t3_dir, real_user_t3_dir]:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            settings_path = target_dir / "settings.json"
            data = {}
            if settings_path.exists():
                try:
                    data = json.loads(settings_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            if not isinstance(data, dict):
                data = {}

            # Preserve existing user customizations while safely injecting local harness provider configs
            providers = data.setdefault("providers", {})
            opencode_cfg = providers.setdefault("opencode", {})
            # Disable legacy flat opencode provider so T3 Code does not synthesize a duplicate
            # shadow "OpenCode" instance alongside the dedicated "pi" and "hermes" providerInstances.
            opencode_cfg["enabled"] = False
            opencode_cfg["binaryPath"] = str(opencode_exe)
            # CRITICAL: Do NOT set serverUrl for opencode; serverUrl tells T3 Code that this
            # is an external OpenCode server, causing it to probe /global/health on the LLM port
            opencode_cfg.pop("serverUrl", None)

            antigravity_cfg = providers.setdefault("antigravity", {})
            antigravity_cfg.setdefault("enabled", True)

            codex_cfg = providers.setdefault("codex", {})
            codex_cfg.setdefault("enabled", True)

            claude_agent_cfg = providers.setdefault("claudeAgent", {})
            if claude_offline:
                claude_agent_cfg["enabled"] = True
                claude_agent_cfg["binaryPath"] = str(claude_exe_path)
                claude_agent_cfg["homePath"] = str(claude_local_dir)
                claude_agent_cfg["launchArgs"] = claude_launch_args
                claude_agent_cfg["autoCompactWindow"] = auto_compact_str
            else:
                claude_agent_cfg["enabled"] = True
                claude_agent_cfg["binaryPath"] = "claude"
                claude_agent_cfg["homePath"] = ""
                claude_agent_cfg["launchArgs"] = ""
                claude_agent_cfg["autoCompactWindow"] = ""

            provider_instances = data.setdefault("providerInstances", {})
            # Remove server-level entries from providerInstances so only harnesses appear
            provider_instances.pop("localai", None)
            provider_instances.pop("openai", None)

            # Register Pi Coding Harness and Hermes as selectable harnesses in T3 Code
            provider_instances["pi"] = {
                "driver": "opencode",
                "displayName": "Pi Coding Harness",
                "accentColor": "#6366f1",
                "enabled": True,
                "environment": pi_env,
                "models": models_list,
                "config": {
                    "binaryPath": str(opencode_exe),
                    "models": models_list
                }
            }
            provider_instances["omp"] = {
                "driver": "opencode",
                "displayName": "Oh My Pi",
                "accentColor": "#8b5cf6",
                "enabled": True,
                "environment": omp_env,
                "models": models_list,
                "config": {
                    "binaryPath": str(opencode_exe),
                    "models": models_list
                }
            }
            provider_instances["hermes"] = {
                "driver": "opencode",
                "displayName": "Hermes",
                "accentColor": "#10b981",
                "enabled": True,
                "environment": hermes_env,
                "models": models_list,
                "config": {
                    "binaryPath": str(opencode_exe),
                    "models": models_list
                }
            }

            claude_data_dir = t3_home / "opencode_data" / "claude"
            claude_cache_dir = t3_home / "opencode_cache" / "claude"
            claude_state_dir = t3_home / "opencode_state" / "claude"
            for d in [claude_data_dir, claude_cache_dir, claude_state_dir]:
                d.mkdir(parents=True, exist_ok=True)
            claude_env = list(local_env) + [
                {"name": "OPENCODE_CONFIG", "value": (t3_home / "opencode_claude.json").as_posix()},
                {"name": "XDG_DATA_HOME", "value": claude_data_dir.as_posix()},
                {"name": "XDG_CACHE_HOME", "value": claude_cache_dir.as_posix()},
                {"name": "XDG_STATE_HOME", "value": claude_state_dir.as_posix()},
            ]
            if claude_offline:
                for k, v in claude_local_env.items():
                    claude_env.append({"name": k, "value": str(v)})
            else:
                claude_env.append({"name": "CLAUDE_CONFIG_DIR", "value": ""})

            provider_instances["claude"] = {
                "driver": "opencode",
                "displayName": "Claude Code",
                "accentColor": "#d97706",
                "enabled": True,
                "environment": claude_env,
                "models": models_list,
                "config": {
                    "binaryPath": str(opencode_exe),
                    "models": models_list
                }
            }

            # Clean up any legacy or misleading serverUrl in providerInstances
            for inst_id, inst_cfg in list(provider_instances.items()):
                if isinstance(inst_cfg, dict):
                    if inst_cfg.get("driver") == "opencode":
                        if isinstance(inst_cfg.get("config"), dict):
                            inst_cfg["config"].pop("serverUrl", None)
                            inst_cfg["config"]["binaryPath"] = str(opencode_exe)
                        inst_cfg.pop("serverUrl", None)
                        inst_cfg.pop("baseUrl", None)

            data["serverSettings"] = {
                "port": t3_port,
                "host": "127.0.0.1"
            }
            data["defaultModelSelection"] = {
                "instanceId": default_harness,
                "model": f"{default_harness}/{model_name}"
            }

            # Strip deprecated flat keys that trigger @effect/schema validation failures
            for bad_key in ["telemetry", "crashReporting", "defaultWorkspace", "defaultProvider", "defaultModel"]:
                data.pop(bad_key, None)

            atomic_write_json(settings_path, data)
            logger.info(f"  -> Configured T3 Code user settings in {settings_path}")
        except Exception as e:
            logger.warning(f"  -> Warning: Could not write T3 global settings to {target_dir}: {e}")

    # 4. Pre-populate T3 caches for Pi and Hermes while purging stale server caches
    for target_dir in [global_t3_dir, real_user_t3_dir]:
        cache_dir = target_dir.parent / "caches"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            stale_caches = ["localai.json", "openai.json", "opencode.json"]
            if claude_offline:
                stale_caches.append("claudeAgent.json")
            for stale_cache in stale_caches:
                stale_file = cache_dir / stale_cache
                if stale_file.exists():
                    try:
                        stale_file.unlink()
                    except Exception:
                        pass

            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            harness_cache_list = [("pi", "Pi Coding Harness"), ("hermes", "Hermes"), ("claude", "Claude Code")]
            for h_id, h_disp in harness_cache_list:
                h_models = []
                for m in models_list:
                    m_id = m.get("id")
                    if not m_id:
                        continue
                    # Note: Do NOT set subProvider; omitting subProvider ensures T3 Code's
                    # ProviderModelPicker cleanly displays the instance displayName ("Pi Coding Harness"
                    # or "Hermes") without stuttering into "Pi Coding Harness · Pi Coding Harness".
                    h_models.append({
                        "slug": f"{h_id}/{m_id}",
                        "name": m.get("name") or m_id,
                        "isCustom": False,
                        "capabilities": {
                            "optionDescriptors": [
                                {
                                    "id": "variant",
                                    "label": "Reasoning",
                                    "type": "select",
                                    "options": [
                                        {"id": "low", "label": "Low"},
                                        {"id": "medium", "label": "Medium", "isDefault": True},
                                        {"id": "high", "label": "High"},
                                        {"id": "xhigh", "label": "Xhigh"}
                                    ],
                                    "currentValue": "medium"
                                },
                                {
                                    "id": "agent",
                                    "label": "Agent",
                                    "type": "select",
                                    "options": [
                                        {"id": "build", "label": "Build", "isDefault": True},
                                        {"id": "plan", "label": "Plan"}
                                    ],
                                    "currentValue": "build"
                                }
                            ]
                        }
                    })
                atomic_write_json(cache_dir / f"{h_id}.json", {
                    "instanceId": h_id,
                    "driver": "opencode",
                    "displayName": h_disp,
                    "showInteractionModeToggle": False,
                    "enabled": True,
                    "installed": True,
                    "version": "1.18.33",
                    "status": "ready",
                    "auth": {
                        "status": "authenticated",
                        "type": "opencode"
                    },
                    "checkedAt": now_iso,
                    "message": f"Connected to {h_disp} through OpenCode.",
                    "models": h_models
                })
        except Exception as cache_err:
            logger.debug(f"Cache pre-population notice for {cache_dir}: {cache_err}")

    # 5. Local Workspace State (.t3/config.json)
    local_config_file = t3_home / "config.json"
    local_state_data = {
        "activeProject": str(config.harness_start_dir or PROJECTS_DIR),
        "systemPromptPath": str(CONFIG_DIR / "USER.md"),
        "openaiBaseUrl": llm_base_url,
        "defaultModel": f"{default_harness}/{model_name}",
        "defaultHarness": default_harness
    }
    try:
        from local_ai_launcher.paths import atomic_write_json
        atomic_write_json(local_config_file, local_state_data)
        logger.info(f"  -> Configured local T3 state in {local_config_file}")
    except Exception as e:
        logger.warning(f"  -> Warning: Could not write local T3 state: {e}")


class T3Adapter(HarnessAdapter):
    """Adapter for T3 Code harness (pingdotgg/t3code agent control surface)."""

    TRACK_CONSOLE_CLOSE = False

    def __init__(self, root_dir, processes):
        super().__init__("t3", root_dir, processes)
        self.local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
        self.local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
        self.t3_home = config.t3_dir
        self.desktop_exe = self._resolve_desktop_executable()
        self.cli_exe = self._resolve_cli_executable()
        self.t3_exe = self.desktop_exe or self.cli_exe

    def _resolve_desktop_executable(self) -> Optional[Path]:
        """Find the T3 Code standalone desktop GUI application if installed."""
        candidates = [
            self.local_appdata / "Programs" / "t3code" / "T3 Code (Alpha).exe",
            self.local_appdata / "Programs" / "t3code" / "T3 Code.exe",
            self.local_appdata / "Programs" / "t3-code" / "T3 Code.exe",
            self.local_appdata / "Programs" / "t3" / "T3 Code.exe",
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "T3 Code" / "T3 Code.exe",
        ]
        for c in candidates:
            if c.exists():
                return c
        t3code_dir = self.local_appdata / "Programs" / "t3code"
        if t3code_dir.exists():
            exes = [e for e in t3code_dir.glob("T3 Code*.exe") if "unins" not in e.name.lower()]
            if exes:
                return exes[0]
        return None

    def _resolve_cli_executable(self) -> Path:
        """Find the T3 Code CLI/server binary in workspace node_modules, LocalAppData, or PATH."""
        candidates = [
            AI_WORKSPACE_DIR / "node_modules" / "@t3code" / "t3-win32-x64" / "t3.exe",
            self.local_bin / "t3.cmd",
            AI_WORKSPACE_DIR / "node_modules" / ".bin" / "t3.cmd",
        ]
        for c in candidates:
            if c.exists():
                return c
        which_path = shutil.which("t3") or shutil.which("t3-code")
        if which_path:
            return Path(which_path)
        return candidates[0]

    def _resolve_executable(self) -> Path:
        """Find the T3 Code executable, prioritizing the desktop GUI app if installed."""
        desktop = self._resolve_desktop_executable()
        if desktop:
            return desktop
        return self._resolve_cli_executable()

    def _install_t3(self) -> bool:
        """Install latest official T3 Code via winget (desktop GUI app), falling back to npm."""
        logger.info("  -> Installing official T3 Code desktop app via winget (T3Tools.T3Code)...")
        try:
            res = subprocess.run(
                ["winget", "install", "--id", "T3Tools.T3Code", "-e", "--silent",
                 "--accept-package-agreements", "--accept-source-agreements"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if res.returncode == 0:
                logger.info("  -> T3 Code desktop app installed successfully via winget.")
                self.desktop_exe = self._resolve_desktop_executable()
                self.cli_exe = self._resolve_cli_executable()
                self.t3_exe = self.desktop_exe or self.cli_exe
                return True
            logger.warning(f"  -> winget install returned code {res.returncode}. Trying npm...")
        except Exception as wg_err:
            logger.warning(f"  -> winget install failed: {wg_err}. Trying npm...")

        try:
            npm_exe = get_npm_exe()
            subprocess.run(
                [npm_exe, "install", "--prefix", str(AI_WORKSPACE_DIR), "--no-fund", "--no-audit", "t3@latest"],
                shell=NPM_SHELL, check=True
            )
            logger.info("  -> T3 Code installed successfully via npm.")
            self.cli_exe = self._resolve_cli_executable()
            self.t3_exe = self.desktop_exe or self.cli_exe
            return True
        except Exception as npm_err:
            logger.error(f"  -> Failed to install T3 Code via npm: {npm_err}")
            return False

    def _update_t3(self) -> bool:
        """Update T3 Code to latest official release via winget or npm."""
        logger.info("Updating T3 Code to latest official release...")
        if self.desktop_exe and self.desktop_exe.exists():
            try:
                res = subprocess.run(
                    ["winget", "upgrade", "--id", "T3Tools.T3Code", "-e", "--silent",
                     "--accept-package-agreements", "--accept-source-agreements"],
                    capture_output=True, text=True, encoding='utf-8', errors='replace'
                )
                if res.returncode == 0:
                    logger.info("  -> T3 Code desktop app updated successfully via winget.")
                    self.desktop_exe = self._resolve_desktop_executable()
                    self.t3_exe = self.desktop_exe or self.cli_exe
                    return True
            except Exception as wg_err:
                logger.warning(f"  -> winget upgrade failed: {wg_err}. Trying npm...")

        try:
            npm_exe = get_npm_exe()
            subprocess.run(
                [npm_exe, "install", "--prefix", str(AI_WORKSPACE_DIR), "--no-fund", "--no-audit", "t3@latest"],
                shell=NPM_SHELL, check=True
            )
            logger.info("  -> T3 Code updated successfully via npm.")
            self.cli_exe = self._resolve_cli_executable()
            self.t3_exe = self.desktop_exe or self.cli_exe
            return True
        except Exception as e:
            logger.error(f"  -> Failed to update T3 Code: {e}")
            return False

    def _ensure_opencode_installed(self) -> bool:
        """Ensure the opencode-ai package is installed for T3 Code local agent execution."""
        opencode_exe = _resolve_opencode_executable()
        if opencode_exe.exists():
            return True
        logger.info("  -> Installing opencode-ai package for T3 Code local agent driver...")
        try:
            npm_exe = get_npm_exe()
            subprocess.run(
                [npm_exe, "install", "--prefix", str(AI_WORKSPACE_DIR), "--no-fund", "--no-audit", "opencode-ai@latest"],
                shell=NPM_SHELL, check=True
            )
            postinstall_mjs = AI_WORKSPACE_DIR / "node_modules" / "opencode-ai" / "postinstall.mjs"
            if postinstall_mjs.exists():
                subprocess.run(["node", str(postinstall_mjs)], check=True)
            logger.info("  -> opencode-ai installed successfully.")
            return True
        except Exception as e:
            logger.warning(f"  -> Failed to install opencode-ai: {e}")
            return False

    def check(self) -> bool:
        logger.info("\nChecking for T3 Code harness...")
        self.desktop_exe = self._resolve_desktop_executable()
        self.cli_exe = self._resolve_cli_executable()
        self.t3_exe = self.desktop_exe or self.cli_exe
        is_installed = (self.desktop_exe is not None and self.desktop_exe.exists()) or self.cli_exe.exists() or bool(shutil.which("t3"))

        if is_installed:
            target_desc = f"desktop app at: {self.desktop_exe}" if self.desktop_exe else f"CLI at: {self.cli_exe}"
            logger.info(f"  -> T3 Code is installed ({target_desc})")
            if self._should_update():
                if self._update_t3():
                    self.record_update()
            self._ensure_opencode_installed()
            return True
        else:
            logger.info("  -> T3 Code is not installed.")
            if self._should_install():
                installed = self._install_t3()
                if installed:
                    self._ensure_opencode_installed()
                return installed
            else:
                logger.info("  -> Skipping T3 Code installation.")
                return False

    def start(self, llm_base_url: str, active_model: Optional[str] = None) -> None:
        model_name = resolve_model_name(active_model)
        try:
            t3_port = int(getattr(config, 't3_port', 3773))
            if not (1 <= t3_port <= 65535):
                t3_port = 3773
        except (ValueError, TypeError):
            t3_port = 3773

        # Re-resolve executables in case installed or updated during check
        self.desktop_exe = self._resolve_desktop_executable()
        self.cli_exe = self._resolve_cli_executable()
        self.t3_exe = self.desktop_exe or self.cli_exe

        # Pre-configure settings and ensure workspace directories exist
        setup_t3_config(self.t3_home, llm_base_url, active_model)

        if getattr(config, "aside_browser_enabled", True) and getattr(config, "aside_browser_auto_sync", True):
            try:
                from local_ai_launcher.backends.aside import setup_aside_local_engine, is_aside_installed
                if is_aside_installed():
                    setup_aside_local_engine(llm_base_url, active_model=model_name)
            except Exception as e:
                logger.warning(f"Notice: Failed to configure Aside browser for T3: {e}")

        # Pre-evaluate OpenCode agent system prompt and tool definitions into KV cache for sub-second TTFT
        from local_ai_launcher.context_preloader import preload_harness_context
        preload_harness_context(llm_base_url, model_name, harness_name="t3")

        from local_ai_launcher.harnesses.claude import (
            is_claude_offline_mode_active,
            get_claude_local_env,
        )
        claude_offline = is_claude_offline_mode_active(active_model)
        claude_local_env = get_claude_local_env(llm_base_url, active_model) if claude_offline else {}

        # Prepare extra bin directories for Pi, Hermes, and Claude CLI execution inside T3 Code sessions
        extra_bin_dirs = [
            str(AI_WORKSPACE_DIR / "node"),
            str(AI_WORKSPACE_DIR / ".hermes" / "bin"),
            str(AI_WORKSPACE_DIR / "node_modules" / ".bin"),
        ]
        if claude_offline:
            claude_local_dir = getattr(config, "claude_local_dir", None) or (AI_WORKSPACE_DIR / ".cbox")
            extra_bin_dirs.append(str(claude_local_dir))
        if getattr(config, "aside_browser_enabled", True):
            try:
                from local_ai_launcher.paths import get_aside_binary_path
                aside_bin = getattr(config, "aside_browser_bin", None) or get_aside_binary_path()
                if aside_bin and Path(aside_bin).is_file():
                    extra_bin_dirs.append(str(Path(aside_bin).parent))
            except Exception:
                pass

        if self.desktop_exe and self.desktop_exe.exists():
            # STANDALONE DESKTOP GUI MODE:
            # The Electron desktop app spawns and manages its own internal child backend server
            # on t3_port and bootstraps its local bearer auth session. We must ensure the port is free
            # so the internal child server does not crash with EADDRINUSE.
            force_kill_port(t3_port)
            logger.info(f"Starting T3 Code Desktop GUI application ({self.desktop_exe.name}, model: {model_name}, endpoint: {llm_base_url})...")
            try:
                desktop_env = os.environ.copy()
                desktop_env['OPENAI_BASE_URL'] = llm_base_url
                desktop_env['OPENAI_API_BASE'] = llm_base_url
                desktop_env['OPENAI_API_KEY'] = 'dummy'
                desktop_env['OPENAI_MODEL'] = model_name
                desktop_env['MODEL'] = model_name
                desktop_env['HOME'] = str(AI_WORKSPACE_DIR)
                desktop_env['USERPROFILE'] = str(AI_WORKSPACE_DIR)
                desktop_env['PI_HOME'] = str(AI_WORKSPACE_DIR / ".pi")
                desktop_env['OMP_HOME'] = str(AI_WORKSPACE_DIR / ".omp")
                desktop_env['PI_CODING_AGENT_DIR'] = str(AI_WORKSPACE_DIR / ".pi" / "agent")
                desktop_env['PI_SKIP_VERSION_CHECK'] = "1"
                desktop_env['HERMES_HOME'] = str(AI_WORKSPACE_DIR / ".hermes")
                desktop_env['HF_HOME'] = str(AI_WORKSPACE_DIR / "models" / "huggingface")
                desktop_env['HF_HUB_CACHE'] = str(AI_WORKSPACE_DIR / "models" / "huggingface")
                desktop_env['npm_config_allow_git'] = "all"
                desktop_env['T3CODE_PORT'] = str(t3_port)
                desktop_env['T3CODE_BASE_DIR'] = str(self.t3_home)
                if claude_offline:
                    desktop_env.update(claude_local_env)
                else:
                    desktop_env["CLAUDE_CONFIG_DIR"] = ""

                current_path = desktop_env.get("PATH", "")
                new_paths = [p for p in extra_bin_dirs if p.lower() not in current_path.lower()]
                if new_paths:
                    desktop_env["PATH"] = os.pathsep.join(new_paths) + os.pathsep + current_path

                gui_cmd = [str(self.desktop_exe)]
                if config.harness_start_dir and Path(config.harness_start_dir).exists():
                    gui_cmd.append(str(config.harness_start_dir))

                gui_proc = subprocess.Popen(
                    gui_cmd,
                    cwd=str(self.desktop_exe.parent),
                    env=desktop_env
                )
                assign_process_to_job(gui_proc)
                self.processes.append(('T3 Code Desktop GUI', gui_proc, True))
                logger.info("  -> T3 Code Desktop GUI window launched successfully.")
            except Exception as e:
                logger.error(f"Error starting T3 Code Desktop GUI: {e}")
        else:
            # HEADLESS / CLI WEB MODE:
            # Standalone desktop app is not installed. Start the background HTTP server in a
            # Windows Terminal tab and open the user's web browser.
            current_path = os.environ.get("PATH", "")
            new_paths = [p for p in extra_bin_dirs if p.lower() not in current_path.lower()]
            updated_path = os.pathsep.join(new_paths) + os.pathsep + current_path if new_paths else current_path

            cli_env_map = {
                'OPENAI_BASE_URL': llm_base_url,
                'OPENAI_API_BASE': llm_base_url,
                'OPENAI_API_KEY': 'dummy',
                'OPENAI_MODEL': model_name,
                'MODEL': model_name,
                'HOME': str(AI_WORKSPACE_DIR),
                'USERPROFILE': str(AI_WORKSPACE_DIR),
                'PI_HOME': str(AI_WORKSPACE_DIR / ".pi"),
                'OMP_HOME': str(AI_WORKSPACE_DIR / ".omp"),
                'PI_CODING_AGENT_DIR': str(AI_WORKSPACE_DIR / ".pi" / "agent"),
                'PI_SKIP_VERSION_CHECK': '1',
                'HERMES_HOME': str(AI_WORKSPACE_DIR / ".hermes"),
                'HF_HOME': str(AI_WORKSPACE_DIR / "models" / "huggingface"),
                'HF_HUB_CACHE': str(AI_WORKSPACE_DIR / "models" / "huggingface"),
                'npm_config_allow_git': 'all',
                'PATH': updated_path,
                'T3CODE_PORT': str(t3_port),
                'T3CODE_BASE_DIR': str(self.t3_home),
            }
            if claude_offline:
                cli_env_map.update(claude_local_env)
            else:
                cli_env_map["CLAUDE_CONFIG_DIR"] = ""
            env_setup = build_ps_env(cli_env_map)

            encoding_preamble = (
                "chcp 65001 | Out-Null; "
                "[console]::InputEncoding=[console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
            )

            server_target = self.cli_exe if self.cli_exe.exists() else "npx t3@latest"
            if isinstance(server_target, Path) or (isinstance(server_target, str) and not server_target.startswith("npx")):
                server_cmd = (
                    f"& '{server_target}' serve --port {t3_port} "
                    f"--base-dir '{self.t3_home}' "
                    f"--auto-bootstrap-project-from-cwd "
                    f"'{config.harness_start_dir}'"
                )
            else:
                server_cmd = (
                    f"{server_target} serve --port {t3_port} "
                    f"--base-dir '{self.t3_home}' "
                    f"--auto-bootstrap-project-from-cwd "
                    f"'{config.harness_start_dir}'"
                )

            logger.info(f"Starting T3 Code server backend (model: {model_name}, endpoint: {llm_base_url}, port: {t3_port})...")
            try:
                full_command = f'powershell.exe -NoExit -Command "{encoding_preamble}{env_setup}{server_cmd}"'
                server_process = start_in_wt_tab(full_command, cwd=str(config.harness_start_dir), title="T3 Code Server")
                assign_process_to_job(server_process)
                self.processes.append(('T3 Code Server', server_process, self.TRACK_CONSOLE_CLOSE))

                time.sleep(0.5)
                if server_process.poll() is not None:
                    logger.error(f"  -> T3 Code server process exited immediately with code {server_process.returncode}")
                    return

                # Wait for backend server port readiness before opening browser
                if wait_for_port(t3_port, timeout=8.0):
                    if not getattr(config, "localai_headless", False):
                        logger.info(f"  -> T3 Code server responded on port {t3_port}. Opening browser...")
                        webbrowser.open(f"http://localhost:{t3_port}")
                    else:
                        logger.info(f"  -> T3 Code server responded on port {t3_port} (running headless).")
                else:
                    logger.warning(f"  -> T3 Code server did not respond on port {t3_port} within timeout. Skipping browser.")
            except Exception as e:
                logger.error(f"Error starting T3 Code: {e}")
