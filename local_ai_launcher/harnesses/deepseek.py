import subprocess
import time
import sys
import os
import shutil
import json
from pathlib import Path
import logging
from typing import Optional

from local_ai_launcher.config import *
from local_ai_launcher.environment import *
from local_ai_launcher.models import *
from local_ai_launcher.process import *
from local_ai_launcher.harness_base import HarnessAdapter

class DeepseekAdapter(HarnessAdapter):
    """Adapter for Deepseek coding harness."""
    
    def __init__(self, root_dir, processes):
        super().__init__("deepseek", root_dir, processes)
        self.local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
        self.exe_name = "dsh.cmd" if os.name == 'nt' else "dsh"
        self.deepseek_exe = self._resolve_executable()

    def _resolve_executable(self) -> Optional[Path]:
        which_dsh = shutil.which("dsh")
        if (self.local_bin / self.exe_name).exists():
            return self.local_bin / self.exe_name
        elif which_dsh:
            return Path(which_dsh)
        return None

    def ensure_plugins_and_skills(self):
        """Ensure plugins and scoped skills defined for DeepSeek harness are provisioned."""
        dsh_home = Path(os.environ.get("DSH_HOME", AI_WORKSPACE_DIR / ".dsh"))
        
        # 1. Provision scoped skills (e.g. donsetch)
        if getattr(config, "deepseek_donsetch_enabled", True):
            try:
                from local_ai_launcher.skills_manager import SkillManager
                SkillManager().ensure_scoped_harness_skills("deepseek", dsh_home / "skills")
            except Exception as e:
                logger.warning(f"  -> Warning: Failed to provision scoped skills for DeepSeek: {e}")
        else:
            scoped_skill = dsh_home / "skills" / "donsetch"
            if scoped_skill.exists():
                shutil.rmtree(scoped_skill, ignore_errors=True)

        # 2. Check and install first-class DSH plugins
        plugins = list(getattr(config, "harness_plugins", {}).get(self.name, []))
        if getattr(config, "deepseek_donsetch_enabled", True):
            if not any("donsetch-dsh" in p for p in plugins):
                plugins.append("github:dondai44423/donsetch-dsh")
        else:
            plugins = [p for p in plugins if "donsetch-dsh" not in p]

        if not plugins:
            return

        web_profile = dsh_home / "profiles" / "web"
        pkg_json_file = web_profile / "package.json"
        installed_plugins = set()
        if pkg_json_file.exists():
            try:
                with open(pkg_json_file, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                    bundles = pdata.get("dsh", {}).get("profile", {}).get("bundles", [])
                    deps = pdata.get("dependencies", {})
                    installed_plugins = set(bundles).union(deps.keys())
            except Exception:
                pass

        if not self.deepseek_exe or not self.deepseek_exe.exists():
            self.deepseek_exe = self._resolve_executable()

        if not (self.deepseek_exe and self.deepseek_exe.exists()) and not shutil.which("dsh"):
            logger.debug("DeepSeek executable not found; skipping plugin installation for now.")
            return

        dsh_cmd_str = str(self.deepseek_exe) if (self.deepseek_exe and self.deepseek_exe.exists()) else "dsh"
        env = os.environ.copy()
        env["DSH_HOME"] = str(dsh_home)
        
        paths_to_add = []
        if self.local_bin and self.local_bin.exists():
            paths_to_add.append(str(self.local_bin))
        node_bin = AI_WORKSPACE_DIR / "node"
        if node_bin.exists():
            paths_to_add.append(str(node_bin))
        if paths_to_add:
            env["PATH"] = ";".join(paths_to_add + [env.get("PATH", "")])

        for plugin_src in plugins:
            pkg_name = "@dsh-external/donsetch" if "donsetch-dsh" in plugin_src else plugin_src
            if pkg_name in installed_plugins:
                logger.info(f"  -> Plugin '{plugin_src}' is already installed for {self.name}.")
            else:
                logger.info(f"  -> Installing plugin '{plugin_src}' for {self.name}...")
                try:
                    subprocess.run(
                        [dsh_cmd_str, "plugin", "--profile", "web", "add", plugin_src],
                        shell=NPM_SHELL,
                        check=True,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    logger.info(f"  -> Plugin '{plugin_src}' installed successfully.")
                except Exception as e:
                    logger.warning(f"  -> Warning: Failed to install plugin '{plugin_src}': {e}")

    def check(self):
        logger.info(f"\nChecking for Deepseek harness...")
        is_installed = (self.local_bin / self.exe_name).exists() or shutil.which("dsh")
        
        if is_installed:
            logger.info("  -> Deepseek is already installed.")
            if self._should_update():
                logger.info("Updating Deepseek...")
                try:
                    subprocess.run([get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "--no-fund", "--no-audit", "--loglevel=error", "--ignore-scripts", "@deepseek-ai/dsh@latest"], shell=NPM_SHELL, check=True)
                    logger.info("  -> Deepseek updated successfully via npm.")
                    self.deepseek_exe = self._resolve_executable()
                    self.record_update()
                except subprocess.CalledProcessError as e:
                    logger.error(f"  -> Failed to update Deepseek via npm: {e}")
            else:
                self.deepseek_exe = self._resolve_executable()
            self.ensure_plugins_and_skills()
            return True
        else:
            logger.info("  -> Deepseek is not installed.")
            if self._should_install():
                logger.info("Installing Deepseek into workspace...")
                try:
                    subprocess.run([get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "--no-fund", "--no-audit", "--loglevel=error", "--ignore-scripts", "@deepseek-ai/dsh@latest"], shell=NPM_SHELL, check=True)
                    logger.info(f"  -> Deepseek installed successfully via npm in {AI_WORKSPACE_DIR}.")
                    self.deepseek_exe = self._resolve_executable()
                    self.ensure_plugins_and_skills()
                    return True
                except subprocess.CalledProcessError as e:
                    logger.error(f"  -> Failed to install Deepseek: {e}")
                    return False
            else:
                logger.info("  -> Skipping Deepseek installation.")
                return False

    def start(self, llm_base_url, active_model=None):
        self.ensure_plugins_and_skills()
        model_name = resolve_model_name(active_model)

        compaction = compute_compaction_params(model_name)
        try:
            raw_ctx_size = max(512, int(compaction.ctx_size))
        except (ValueError, TypeError):
            raw_ctx_size = 32768

        # Clamp requested context size to hardware VRAM safety bounds
        try:
            from local_ai_launcher.vram_coordinator import calculate_safe_context_limit
            engine = getattr(config, "choose_engine", None) or getattr(config, "default_engine", "official")
            if isinstance(engine, (list, tuple)):
                engine = engine[0] if engine else "official"
            ctx_size = calculate_safe_context_limit(model_name, requested_ctx=raw_ctx_size, engine=str(engine))
        except Exception as e:
            logger.debug(f"DeepSeek context safety calculation notice: {e}")
            ctx_size = raw_ctx_size

        try:
            ctx_size = max(512, int(ctx_size))
        except (ValueError, TypeError):
            ctx_size = 32768

        # Hardware-proportional context output allocation for high-end GPUs (e.g. RTX 5080)
        max_output_tokens = min(ctx_size, max(8192, ctx_size // 8))

        try:
            thresh_ratio = max(0.0, min(1.0, float(config.hermes_compression_threshold_ratio)))
        except (ValueError, TypeError):
            thresh_ratio = 0.85

        try:
            retain_ratio = max(0.0, min(1.0, float(config.pi_compaction_keep_ratio)))
        except (ValueError, TypeError):
            retain_ratio = 0.50

        # Check multimodal / vision capabilities
        is_vision = False
        try:
            models_info = get_available_preset_models(only_installed=False, active_model=model_name)
            matched = next((m for m in models_info if m.get("id") == model_name), None)
            if matched and "image" in matched.get("input", []):
                is_vision = True
        except Exception:
            pass
        if not is_vision:
            is_vision = any(k in model_name.lower() for k in ("vision", "-vl", "_vl", "mmproj"))

        if is_vision:
            model_entry_yaml = f"""      - id: {model_name}
        name: Local Model ({model_name})
        contextWindow: {ctx_size}
        maxTokens: {max_output_tokens}
        inputModalities:
          - text
          - image
        imagePixelBudget: 640000
        imageMaxBytes: 1048576"""
        else:
            model_entry_yaml = f"""      - id: {model_name}
        name: Local Model ({model_name})
        contextWindow: {ctx_size}
        maxTokens: {max_output_tokens}
        inputModalities:
          - text"""

        patch_file = config.harness_start_dir / "dsh_model_patch.yml"
        
        donsetch_patch = ""
        if getattr(config, "deepseek_donsetch_enabled", True):
            tool_prefix = getattr(config, "donsetch_tool_prefix", "donsetch")
            auto_update_str = "true" if getattr(config, "donsetch_auto_update", True) else "false"
            donsetch_patch = f"""- id: donsetch
  config:
    toolPrefix: {tool_prefix}
    autoUpdate: {auto_update_str}
"""

        yaml_content = f"""- id: compaction-basic
  disabled: {'false' if config.compaction_enabled else 'true'}
  config:
    thresholdRatio: {thresh_ratio}
    retainRatio: {retain_ratio}
- id: agent-default-model
  config:
    provider: deepseek-official
    model: {model_name}
- id: session-title-llm
  config:
    targetWords: 5
    targetCjkCharacters: 10
    maxInputBytes: 4096
    maxOutputTokens: 64
    timeoutMs: 60000
    provider: deepseek-official
    model: {model_name}
- id: llm-deepseek
  config:
    baseURL: {llm_base_url}
    apiKeyEnv: DEEPSEEK_API_KEY
    defaultContextWindow: {ctx_size}
    maxTokens: {max_output_tokens}
    models:
{model_entry_yaml}
{donsetch_patch}"""
        from local_ai_launcher.paths import atomic_write_text
        atomic_write_text(patch_file, yaml_content)

        dsh_home = Path(os.environ.get("DSH_HOME", AI_WORKSPACE_DIR / ".dsh"))
        settings_file = dsh_home / "settings.yaml"
        try:
            import yaml
            settings_data = {}
            if settings_file.exists():
                try:
                    loaded = yaml.safe_load(settings_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        settings_data = loaded
                except Exception as parse_err:
                    logger.warning(f"Could not parse existing DSH settings.yaml: {parse_err}")
            if not isinstance(settings_data.get("agent-default-model"), dict):
                settings_data["agent-default-model"] = {}
            settings_data["agent-default-model"]["provider"] = "deepseek-official"

            # Check if model is valid or alias-recognized before syncing to settings.yaml
            preset_ids = {m.get("id") for m in get_available_preset_models(only_installed=False, active_model=model_name)}
            is_valid = (
                model_name in preset_ids
                or model_name.startswith("1-")
                or model_name in ("omniroute", "default")
                or (Path(model_name).exists() if (isinstance(model_name, (str, Path)) and not any(c in str(model_name) for c in "*?<>|")) else False)
                or ((AI_WORKSPACE_DIR / model_name).exists() if (isinstance(model_name, (str, Path)) and not any(c in str(model_name) for c in "*?<>|")) else False)
            )
            target_sync_model = model_name if is_valid else normalize_model_identifier(config.default_model)
            settings_data["agent-default-model"]["model"] = target_sync_model
            atomic_write_text(settings_file, yaml.dump(settings_data, sort_keys=False))
        except Exception as e:
            logger.warning(f"Notice: Could not sync DSH settings.yaml: {e}")

        if not self.deepseek_exe or not self.deepseek_exe.exists():
            self.deepseek_exe = self._resolve_executable()
        deepseek_args = f"web --patch '{patch_file}'"
        deepseek_bin = f"& '{self.deepseek_exe}'" if self.deepseek_exe and self.deepseek_exe.exists() else "dsh"
        deepseek_cmd = f"{deepseek_bin} {deepseek_args}"

        dsh_env = {
            'OPENAI_BASE_URL': llm_base_url,
            'OPENAI_API_BASE': llm_base_url,
            'OPENAI_API_KEY': 'dummy',
            'OPENAI_MODEL': model_name,
            'MODEL': model_name,
            'DEEPSEEK_BASE_URL': llm_base_url,
            'DEEPSEEK_API_KEY': 'dummy',
            'DSH_HOME': str(dsh_home),
            'DSH_AGENTS_HOME': str(AI_WORKSPACE_DIR / ".agents"),
        }
        if getattr(config, "deepseek_donsetch_enabled", True):
            from local_ai_launcher.paths import get_donsetch_binary_path
            don_bin = getattr(config, "donsetch_bin", None) or get_donsetch_binary_path()
            if don_bin:
                dsh_env['DONSETCH_BIN'] = str(don_bin)
                dsh_env['DONSETCH_BINARY'] = str(don_bin)
            dsh_env['DONSETCH_DSH_PREFIX'] = getattr(config, "donsetch_tool_prefix", "donsetch")
            dsh_env['DONSETCH_DSH_CHANNEL'] = getattr(config, "donsetch_channel", "stable")
            if not getattr(config, "donsetch_auto_update", True):
                dsh_env['DONSETCH_DSH_AUTOUPDATE'] = 'off'
        elif getattr(config, "searxng_enabled", True) and str(getattr(config, "searxng_mode", "wsl_docker")).lower() not in ("disabled", "off", "none"):
            searxng_url = getattr(config, "searxng_effective_url", "http://127.0.0.1:8088")
            dsh_env['SEARXNG_BASE_URL'] = searxng_url
            dsh_env['SEARXNG_URL'] = searxng_url

        node_bin = AI_WORKSPACE_DIR / "node"
        if node_bin.exists():
            dsh_env['PATH'] = f"{node_bin};{os.environ.get('PATH', '')}"

        env_setup = build_ps_env(dsh_env, harness_name="deepseek")
        
        # Pre-evaluate DeepSeek agent context and tool definitions into KV cache for sub-second TTFT
        try:
            from local_ai_launcher.context_preloader import preload_harness_context
            preload_harness_context(llm_base_url, model_name, harness_name="deepseek")
        except Exception as e:
            logger.debug(f"DeepSeek context preloading notice: {e}")
            from local_ai_launcher.harness_base import send_background_warmup
            send_background_warmup(llm_base_url, model_name)

        logger.info("Starting Deepseek harness...")
        try:
            deepseek_process = start_in_wt_tab(f'powershell.exe -NoExit -Command "chcp 65001 | Out-Null; [console]::InputEncoding=[console]::OutputEncoding=[System.Text.Encoding]::UTF8; {env_setup}{deepseek_cmd}"', cwd=str(config.harness_start_dir))
            self.processes.append(('Deepseek harness', deepseek_process, self.TRACK_CONSOLE_CLOSE))
        except Exception as e:
            logger.error(f"Error starting Deepseek: {e}")
