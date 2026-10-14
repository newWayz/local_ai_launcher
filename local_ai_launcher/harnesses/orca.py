import subprocess
import os
import shutil
from pathlib import Path
import json
import urllib.request
import tempfile
from typing import Optional

from local_ai_launcher.config import *
from local_ai_launcher.environment import *
from local_ai_launcher.models import *
from local_ai_launcher.process import *
from local_ai_launcher.harness_base import HarnessAdapter


def _run_silent(cmd: List[str] | str, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess silently on Windows without allocating or flashing console windows."""
    if sys.platform == 'win32':
        kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
        startupinfo = kwargs.get('startupinfo')
        if startupinfo is None:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            kwargs['startupinfo'] = startupinfo
    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)
    return subprocess.run(cmd, **kwargs)


def setup_orca_config(orca_home: Path, llm_base_url: str, active_model: Optional[str] = None) -> None:
    """Configure Stably AI Orca ADE worktree environment and ensure model pass-through."""
    orca_home.mkdir(parents=True, exist_ok=True)
    model_name = resolve_model_name(active_model)
    ctx_size, is_reasoning = get_model_specs(model_name)

    start_dir = config.harness_start_dir or PROJECTS_DIR
    start_dir.mkdir(parents=True, exist_ok=True)

    # 0. Self-maintaining workspace storage junctions
    appdata_dir = Path(os.environ.get("APPDATA", ""))
    user_home = Path(os.path.expanduser("~"))
    orca_data_dir = getattr(config, 'orca_data_dir', ORCA_DATA_DIR)

    # Migrate legacy data from in-tree install folder (C:/aiWorkspace/orca/data) to persistent location (.orca/data)
    legacy_data_dir = AI_WORKSPACE_DIR / "orca" / "data"
    if legacy_data_dir.exists() and legacy_data_dir.resolve() != orca_data_dir.resolve() and not orca_data_dir.exists():
        try:
            orca_data_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(legacy_data_dir, orca_data_dir, dirs_exist_ok=True)
            logger.info(f"  -> Successfully migrated Orca state from {legacy_data_dir} to safe persistent path {orca_data_dir}")
        except Exception as mig_err:
            logger.warning(f"  -> Notice migrating legacy Orca data: {mig_err}")

    if appdata_dir.exists():
        ensure_directory_junction(appdata_dir / "Orca", orca_data_dir)
        ensure_directory_junction(appdata_dir / "orca", orca_data_dir)
    if user_home.exists():
        ensure_directory_junction(user_home / ".orca", orca_home)

    # 0b. Pre-flight purge of stale daemon and runtime lockfiles from prior ungraceful exits
    try:
        purge_stale_orca_locks(orca_data_dir)
    except Exception as e:
        logger.debug(f"Orca pre-flight lock purge notice: {e}")

    # 1. Stably AI Orca User Settings (prioritize workspace-contained AppData)
    appdata_str = os.environ.get("APPDATA", "")
    candidates = []
    if os.environ.get("PYTEST_CURRENT_TEST") and appdata_str:
        candidates.append(Path(appdata_str) / "Orca" / "profiles" / "local-default" / "orca-data.json")
        candidates.append(Path(appdata_str) / "orca" / "profiles" / "local-default" / "orca-data.json")

    candidates.extend([
        orca_data_dir / "profiles" / "local-default" / "orca-data.json",
        AI_WORKSPACE_DIR / ".orca" / "data" / "profiles" / "local-default" / "orca-data.json",
        AI_WORKSPACE_DIR / "orca" / "data" / "profiles" / "local-default" / "orca-data.json",
        AI_WORKSPACE_DIR / "AppData" / "Roaming" / "orca" / "profiles" / "local-default" / "orca-data.json",
        AI_WORKSPACE_DIR / "AppData" / "Roaming" / "Orca" / "profiles" / "local-default" / "orca-data.json",
    ])

    if appdata_str:
        candidates.append(Path(appdata_str) / "orca" / "profiles" / "local-default" / "orca-data.json")
        candidates.append(Path(appdata_str) / "Orca" / "profiles" / "local-default" / "orca-data.json")

    if os.environ.get("PYTEST_CURRENT_TEST") and appdata_str:
        orca_settings_file = Path(appdata_str) / "Orca" / "profiles" / "local-default" / "orca-data.json"
    else:
        orca_settings_file = None
        for c in candidates:
            if c.exists():
                orca_settings_file = c
                break
        if not orca_settings_file:
            orca_settings_file = candidates[0]

    try:
        orca_settings_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if orca_settings_file.exists():
            try:
                data = json.loads(orca_settings_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        if not isinstance(data, dict):
            data = {}

        if "schemaVersion" not in data:
            data["schemaVersion"] = 1
        if "settings" not in data:
            data["settings"] = {}

        data["settings"]["workspaceDir"] = str(start_dir)

        default_env = data["settings"].setdefault("agentDefaultEnv", {})
        common_agent_env = {
            "OPENAI_BASE_URL": llm_base_url,
            "OPENAI_API_BASE": llm_base_url,
            "OPENAI_API_KEY": "dummy",
            "OPENAI_MODEL": model_name,
            "MODEL": model_name,
            "HOME": str(AI_WORKSPACE_DIR),
            "PI_HOME": str(AI_WORKSPACE_DIR / ".pi"),
            "PI_CODING_AGENT_DIR": str(AI_WORKSPACE_DIR / ".pi" / "agent"),
            "PI_SKIP_VERSION_CHECK": "1",
            "HF_HOME": str(AI_WORKSPACE_DIR / "models" / "huggingface"),
            "HF_HUB_CACHE": str(AI_WORKSPACE_DIR / "models" / "huggingface"),
            "npm_config_allow_git": "all",
        }
        for agent_key in ["pi", "trae", "openlumara", "hermes", "antigravity", "deepseek", "goose", "t3", "t3code"]:
            agent_dict = default_env.setdefault(agent_key, {})
            for k, v in common_agent_env.items():
                agent_dict[k] = v

        # Claude Code CLI: Local vs Official isolation in Orca ADE
        from local_ai_launcher.harnesses.claude import is_strata_active

        default_env["claude-official"] = {
            "CLAUDE_CONFIG_DIR": "",
        }

        cmd_overrides = data["settings"].setdefault("agentCmdOverrides", {})

        if is_strata_active(active_model):
            claude_local_dir = AI_WORKSPACE_DIR / ".claude" / "local"
            claude_base = llm_base_url.rstrip("/v1").rstrip("/") if llm_base_url else "http://127.0.0.1:8080"
            claude_model = getattr(config, "claude_model", "claude-sonnet-4-5") or "claude-sonnet-4-5"
            dummy_key = getattr(config, "claude_api_key", "dummy-token") or "dummy-token"

            claude_local_env = {
                "CLAUDE_CONFIG_DIR": str(claude_local_dir),
                "ANTHROPIC_BASE_URL": claude_base,
                "ANTHROPIC_API_KEY": dummy_key,
                "ANTHROPIC_MODEL": claude_model,
                "CLAUDE_MODEL": claude_model,
            }
            default_env["claude-local"] = claude_local_env
            claude_runner = str(claude_local_dir / ("claude-local.cmd" if os.name == 'nt' else "claude-local.sh"))
            cmd_overrides["claude-local"] = claude_runner

            active_claude_mode = getattr(config, "claude_mode", "local").strip().lower()
            if active_claude_mode == "local":
                default_env["claude"] = claude_local_env.copy()
            else:
                default_env["claude"] = {"CLAUDE_CONFIG_DIR": ""}
        else:
            default_env.pop("claude-local", None)
            cmd_overrides.pop("claude-local", None)
            default_env["claude"] = {"CLAUDE_CONFIG_DIR": ""}

        from local_ai_launcher.harness_model import resolve_meta_harness_inner_cli_harness
        inner_cli = resolve_meta_harness_inner_cli_harness(
            getattr(config, 'choose_harness', []),
            model_name=model_name,
            cfg=config
        )
        orca_agent_map = {
            "openlumara": "trae",
            "pi": "pi",
            "hermes": "hermes",
            "deepseek": "deepseek",
            "claude": "claude-local" if "claude-local" in default_env else "claude",
        }
        target_orca_agent = orca_agent_map.get(inner_cli, inner_cli)
        data["settings"]["defaultTuiAgent"] = target_orca_agent
        disabled_tui = data["settings"].get("disabledTuiAgents", [])
        if isinstance(disabled_tui, list) and target_orca_agent in disabled_tui:
            disabled_tui.remove(target_orca_agent)
            data["settings"]["disabledTuiAgents"] = disabled_tui

        if getattr(config, 'orca_voice_to_text', False):
            if "voice" not in data["settings"]:
                data["settings"]["voice"] = {}
            data["settings"]["voice"]["enabled"] = True
            data["settings"]["voice"]["openAiApiKeyConfigured"] = True
            data["settings"]["voice"]["sttModel"] = "whisper-1"

        # Remove the bad flat keys we might have added previously
        for bad_key in ["telemetry", "enableTelemetry", "defaultWorkspace", "activeModel", "openaiEndpoint"]:
            data.pop(bad_key, None)

        from local_ai_launcher.paths import atomic_write_json
        atomic_write_json(orca_settings_file, data)
        logger.info(f"  -> Configured Orca profile settings in {orca_settings_file}")
    except Exception as e:
        logger.warning(f"  -> Warning: Could not write Orca profile settings: {e}")

    # 2. Bridge OpenLumara into Orca ADE environment
    try:
        from local_ai_launcher.bridges.orca_openlumara import configure_orca_profile_for_openlumara, install_openlumara_wrappers
        install_openlumara_wrappers(AI_WORKSPACE_DIR)
        set_default = (inner_cli == "openlumara")
        configure_orca_profile_for_openlumara(profile_path=orca_settings_file, llm_base_url=llm_base_url, active_model=model_name, set_default_agent=set_default)
    except Exception as bridge_err:
        logger.warning(f"  -> Warning: OpenLumara bridge configuration notice: {bridge_err}")

    # 2b. Ensure Pi harness configuration in workspace is active for Orca's Pi agent
    try:
        from local_ai_launcher.harnesses.pi import setup_pi_models_and_auth
        setup_pi_models_and_auth(config.pi_dir, llm_base_url, active_model=model_name, use_omniroute=config.use_omniroute_proxy, omniroute_model=config.omniroute_default_model)
        logger.info("  -> Synced workspace Pi models and auth for Orca ADE integration.")
    except Exception as pi_err:
        logger.warning(f"  -> Warning: Pi models and auth configuration notice for Orca: {pi_err}")

    # 3. Worktree File Propagation (.worktreeinclude)
    # Stably AI Orca copies/symlinks files specified in .worktreeinclude to child agent worktrees
    worktree_include_file = start_dir / ".worktreeinclude"
    default_includes = [
        "USER.md",
        ".env",
        ".git/info/exclude"
    ]
    try:
        existing_lines = []
        if worktree_include_file.exists():
            existing_lines = [line.strip() for line in worktree_include_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for inc in default_includes:
            if inc not in existing_lines:
                existing_lines.append(inc)
        worktree_include_file.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")
        logger.info(f"  -> Verified Orca .worktreeinclude file in {start_dir}")
    except Exception as e:
        logger.warning(f"  -> Warning: Could not update .worktreeinclude: {e}")

    # 3. Synchronize USER.md into Orca directory for global reference
    user_md_src = CONFIG_DIR / "USER.md"
    if user_md_src.exists():
        try:
            shutil.copy2(user_md_src, orca_home / "USER.md")
            logger.info(f"  -> Synced USER.md to {orca_home / 'USER.md'}")
        except Exception as e:
            logger.warning(f"  -> Warning: Could not sync USER.md to Orca: {e}")

    # 4. Ensure Git readiness in harness_start_dir for Orca's worktree functionality
    try:
        res = _run_silent(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(start_dir),
        )
        if res.returncode != 0:
            logger.info(f"  -> Initializing Git repository in {start_dir} for Orca worktree management...")
            _run_silent(["git", "init"], cwd=str(start_dir), check=True)
            _run_silent(
                ["git", "-c", "user.name=ai_launcher", "-c", "user.email=ai@launcher.local", "commit", "--allow-empty", "-m", "chore: initial commit for orca worktree management"],
                cwd=str(start_dir), check=True
            )
            logger.info("  -> Git repository initialized successfully for Orca.")
    except Exception as git_err:
        logger.warning(f"  -> Warning: Git readiness verification failed: {git_err}")


class OrcaAdapter(HarnessAdapter):
    """Adapter for Orca coding harness (Stably AI Agent Development Environment)."""

    TRACK_CONSOLE_CLOSE = False

    def __init__(self, root_dir, processes):
        super().__init__("orca", root_dir, processes)
        self.local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
        self.orca_home = config.orca_dir
        self.orca_exe = self._resolve_executable()

    def _resolve_executable(self) -> Path:
        """Find the Orca executable."""
        candidates = [
            AI_WORKSPACE_DIR / "orca" / "Orca.exe",
            AI_WORKSPACE_DIR / "orca" / "orca.exe",
            self.local_appdata / "Programs" / "orca" / "Orca.exe",
            self.local_appdata / "Programs" / "Orca" / "Orca.exe",
            self.local_appdata / "orca" / "orca.exe",
        ]
        for c in candidates:
            if c.exists():
                return c
        which_path = shutil.which("orca") or shutil.which("orca.exe")
        if which_path:
            return Path(which_path)
        return candidates[0]

    def _install_orca(self) -> bool:
        logger.info("  -> Orca is distributed as a standalone app. Skipping npm install.")
        return True

    def _update_orca(self) -> bool:
        logger.info("Checking for Orca ADE updates...")
        orca_cmd = shutil.which("orca") or shutil.which("orca.cmd")
        if not orca_cmd:
            orca_res_bin = AI_WORKSPACE_DIR / "orca" / "resources" / "bin" / "orca.cmd"
            if orca_res_bin.exists():
                orca_cmd = str(orca_res_bin)

        if orca_cmd:
            try:
                logger.info("  -> Updating Orca ADE skills...")
                res = _run_silent([str(orca_cmd), "skills", "update", "--all"], encoding="utf-8", errors="replace", check=False, timeout=60)
                if res.returncode == 0:
                    logger.info("  -> Orca ADE skills updated successfully.")
                else:
                    logger.debug(f"  -> Orca skills update output: {res.stdout or res.stderr}")
            except Exception as e:
                logger.warning(f"  -> Warning: Failed to update Orca skills: {e}")
        return True

    def check(self) -> bool:
        logger.info("\nChecking for Orca harness...")
        is_installed = self.orca_exe.exists() or bool(shutil.which("orca"))

        if is_installed:
            logger.info(f"  -> Orca is installed at: {self.orca_exe}")
            if self._should_update():
                if self._update_orca():
                    self.record_update()
            return True
        else:
            logger.info("  -> Orca is not installed.")
            return False

    def start(self, llm_base_url: str, active_model: Optional[str] = None) -> None:
        model_name = resolve_model_name(active_model)

        # Pre-flight setup: worktree preparation and env profile
        setup_orca_config(self.orca_home, llm_base_url, active_model)
        
        actual_llm_url = llm_base_url
        if getattr(config, 'orca_voice_to_text', False):
            mcp_port = getattr(config, 'mcp_server_port', 9932)
            try:
                mcp_port = int(mcp_port)
                if not (1 <= mcp_port <= 65535):
                    mcp_port = 9932
            except (ValueError, TypeError):
                mcp_port = 9932
            actual_llm_url = f"http://127.0.0.1:{mcp_port}/v1"
            if not is_port_in_use(mcp_port):
                logger.info(f"  -> Starting Proxy/MCP Server on port {mcp_port} for Orca Voice routing...")
                proxy_env = build_ps_env({
                    'OPENAI_BASE_URL': llm_base_url,
                    'OPENAI_MODEL': model_name
                })
                proxy_cmd = f'uvicorn local_ai_launcher.mcp_server:app --port {mcp_port} --host 127.0.0.1'
                try:
                    proxy_proc = subprocess.Popen(
                        f'powershell.exe -WindowStyle Hidden -Command "{proxy_env}{proxy_cmd}"',
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    self.processes.append(('Proxy/MCP Server', proxy_proc, False))
                except Exception as e:
                    logger.error(f"  -> Failed to start proxy server: {e}")

        # Prepare clean environment conforming to Stably ADE contract
        orca_env = os.environ.copy()
        orca_env["OPENAI_BASE_URL"] = actual_llm_url
        orca_env["OPENAI_API_BASE"] = actual_llm_url
        orca_env["OPENAI_API_KEY"] = "dummy"
        orca_env["OPENAI_MODEL"] = model_name
        orca_env["MODEL"] = model_name
        orca_data_dir = getattr(config, 'orca_data_dir', ORCA_DATA_DIR)
        orca_data_dir.mkdir(parents=True, exist_ok=True)
        orca_env["ORCA_USER_DATA_PATH"] = str(orca_data_dir)
        orca_env.pop("ELECTRON_RUN_AS_NODE", None)

        cmd = [str(self.orca_exe)] if self.orca_exe.exists() else ["orca"]
        target_cwd = str(config.harness_start_dir or PROJECTS_DIR)

        logger.info(f"Starting Orca harness (model: {model_name}, endpoint: {llm_base_url})...")
        try:
            creationflags = subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0
            orca_process = subprocess.Popen(
                cmd,
                cwd=target_cwd,
                env=orca_env,
                creationflags=creationflags
            )
            assign_process_to_job(orca_process)
            self.processes.append(('Orca harness', orca_process, self.TRACK_CONSOLE_CLOSE))
            logger.info(f"  -> Orca ADE launched successfully (PID: {orca_process.pid}).")
        except Exception as e:
            logger.error(f"Error starting Orca: {e}")
