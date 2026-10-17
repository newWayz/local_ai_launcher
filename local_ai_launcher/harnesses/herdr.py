import subprocess
import sys
import os
import shutil
from pathlib import Path
import json
from typing import Optional, List, Dict, Any

from local_ai_launcher.config import config, PROJECTS_DIR, AI_WORKSPACE_DIR, CONFIG_DIR
from local_ai_launcher.paths import (
    HERDR_DIR,
    HERDR_DATA_DIR,
    get_herdr_binary_path,
    atomic_write_text,
    atomic_write_json,
)
from local_ai_launcher.environment import ensure_directory_junction, logger
from local_ai_launcher.models import resolve_model_name, get_model_specs
from local_ai_launcher.process import assign_process_to_job, start_in_wt_tab, build_ps_env
from local_ai_launcher.harness_base import HarnessAdapter
from local_ai_launcher.harness_model import resolve_meta_harness_inner_cli_harness


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


def setup_herdr_config(herdr_home: Path, llm_base_url: str, active_model: Optional[str] = None) -> None:
    """Configure Herdr terminal runtime multiplexer (meta-harness) environment portably."""
    herdr_home = Path(herdr_home)
    herdr_home.mkdir(parents=True, exist_ok=True)
    herdr_data_dir = getattr(config, 'herdr_data_dir', HERDR_DATA_DIR)
    herdr_data_dir.mkdir(parents=True, exist_ok=True)

    model_name = resolve_model_name(active_model)
    start_dir = config.harness_start_dir or PROJECTS_DIR
    start_dir.mkdir(parents=True, exist_ok=True)

    # 1. Portable NTFS Directory Junctions
    # Point OS/user roaming config paths directly into the workspace-contained .herdr directory
    appdata_str = os.environ.get("APPDATA", "")
    if appdata_str:
        appdata_dir = Path(appdata_str)
        if appdata_dir.exists():
            ensure_directory_junction(appdata_dir / "herdr", herdr_home)

    user_home = Path(os.path.expanduser("~"))
    if user_home.exists():
        ensure_directory_junction(user_home / ".herdr", herdr_home)
        ensure_directory_junction(user_home / ".config" / "herdr", herdr_home)

    # 2. Resolve Leftmost Non-Meta Inner CLI Harness for Herdr (Rule 2)
    inner_cli = resolve_meta_harness_inner_cli_harness(
        getattr(config, 'choose_harness', []),
        model_name=model_name,
        cfg=config
    )

    # 3. Synchronize Child Harnesses for Meta-Harness Operations
    # Pi Coding Agent sync
    try:
        from local_ai_launcher.harnesses.pi import setup_pi_models_and_auth
        setup_pi_models_and_auth(
            config.pi_dir,
            llm_base_url,
            active_model=model_name,
            use_omniroute=config.use_omniroute_proxy,
            omniroute_model=config.omniroute_default_model
        )
        logger.info("  -> Synced workspace Pi models and auth for Herdr integration.")
    except Exception as pi_err:
        logger.warning(f"  -> Warning: Pi models and auth configuration notice for Herdr: {pi_err}")

    # Oh My Pi (omp) sync
    try:
        from local_ai_launcher.harnesses.omp import setup_omp_models_and_auth
        setup_omp_models_and_auth(
            getattr(config, "omp_dir", AI_WORKSPACE_DIR / ".omp"),
            llm_base_url,
            active_model=model_name,
            use_omniroute=config.use_omniroute_proxy,
            omniroute_model=config.omniroute_default_model
        )
        logger.info("  -> Synced workspace OMP models and auth for Herdr integration.")
    except Exception as omp_err:
        logger.warning(f"  -> Warning: OMP models and auth configuration notice for Herdr: {omp_err}")

    # Claude Code CLI: Local vs Official isolation
    from local_ai_launcher.harnesses.claude import (
        is_claude_offline_mode_active,
        ensure_claude_local_profile,
        get_claude_local_env,
    )
    claude_offline = is_claude_offline_mode_active(active_model)
    claude_runner_cmd = "claude"
    claude_local_env = {}
    if claude_offline:
        claude_local_dir = ensure_claude_local_profile(llm_base_url, active_model)
        runner_file_name = "cbox.cmd" if os.name == 'nt' else "cbox.sh"
        fallback_runner = "claude-local.cmd" if os.name == 'nt' else "claude-local.sh"
        if (claude_local_dir / runner_file_name).exists():
            claude_runner_cmd = str(claude_local_dir / runner_file_name)
        else:
            claude_runner_cmd = str(claude_local_dir / fallback_runner)
        claude_local_env = get_claude_local_env(llm_base_url, active_model)

    # Hermes Agent sync
    try:
        from local_ai_launcher.harnesses.hermes import HermesAdapter
        hermes_adapter = HermesAdapter(AI_WORKSPACE_DIR, [])
        hermes_adapter._ensure_user_profile_junction()
        from local_ai_launcher.bridges.gvs5h_solver import ensure_deep_solve_skills
        ensure_deep_solve_skills([config.hermes_dir / "skills" / "deep-solve"])
        from local_ai_launcher.bridges.cpu_tools_cli import ensure_cpu_tools_skills
        ensure_cpu_tools_skills([config.hermes_dir / "skills" / "cpu-tools"])
        logger.info("  -> Synced Hermes profile and auxiliary skills for Herdr integration.")
    except Exception as hermes_err:
        logger.debug(f"Hermes configuration notice for Herdr: {hermes_err}")

    # 4. Synchronize Universal Preferences (USER.md)
    user_md_src = CONFIG_DIR / "USER.md"
    if user_md_src.exists():
        try:
            shutil.copy2(user_md_src, herdr_home / "USER.md")
            logger.info(f"  -> Synced USER.md to {herdr_home / 'USER.md'}")
        except Exception as e:
            logger.warning(f"  -> Warning: Could not sync USER.md to Herdr: {e}")

    # 5. Ensure Git Repository Readiness in Start Directory
    try:
        res = _run_silent(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(start_dir),
        )
        if res.returncode != 0:
            logger.info(f"  -> Initializing Git repository in {start_dir} for Herdr session management...")
            _run_silent(["git", "init"], cwd=str(start_dir), check=True)
            _run_silent(
                ["git", "-c", "user.name=ai_launcher", "-c", "user.email=ai@launcher.local", "commit", "--allow-empty", "-m", "chore: initial commit for herdr session management"],
                cwd=str(start_dir), check=True
            )
            logger.info("  -> Git repository initialized successfully for Herdr.")
    except Exception as git_err:
        logger.warning(f"  -> Warning: Git readiness verification failed for Herdr: {git_err}")

    # 6. Generate Portable Herdr config.toml
    # Set onboarding = false, new_cwd = follow, resume_agents_on_restore = true,
    # and provide keybinding commands for child agent sessions.
    config_toml_path = herdr_home / "config.toml"
    
    # Resolve command strings for inner agents
    pi_bin = str(AI_WORKSPACE_DIR / "node" / "pi.ps1") if (AI_WORKSPACE_DIR / "node" / "pi.ps1").exists() else "pi"
    pi_cmd_str = f"pi --provider openai --model {model_name}"
    omp_cmd_str = f"omp --provider openai --model {model_name}"
    hermes_bin = str(AI_WORKSPACE_DIR / ".hermes" / "bin" / "hermes.exe") if (AI_WORKSPACE_DIR / ".hermes" / "bin" / "hermes.exe").exists() else "hermes"
    hermes_cmd_str = f"hermes"

    agent_commands = {
        "pi": pi_cmd_str,
        "omp": omp_cmd_str,
        "hermes": hermes_cmd_str,
        "claude": claude_runner_cmd if claude_offline else "claude",
    }
    default_agent_cmd = agent_commands.get(inner_cli, inner_cli)
    default_agent_cmd_clean = default_agent_cmd.replace('\\', '/')
    pi_cmd_clean = pi_cmd_str.replace('\\', '/')
    claude_cmd_clean = (claude_runner_cmd if claude_offline else 'claude').replace('\\', '/')
    hermes_cmd_clean = hermes_cmd_str.replace('\\', '/')
    omp_cmd_clean = omp_cmd_str.replace('\\', '/')
    default_shell_str = 'powershell.exe' if sys.platform == 'win32' else 'bash'

    toml_content = f"""# Herdr configuration generated by Local AI Launcher
# Meta-harness environment for multi-agent terminal operations

onboarding = false

[terminal]
new_cwd = "follow"
shell_mode = "auto"
default_shell = "{default_shell_str}"

[server]
resume_agents_on_restore = true

# Project customizations & agent shortcuts
[[keys.command]]
key = "prefix+a"
type = "pane"
command = "{default_agent_cmd_clean}"
description = "launch default agent ({inner_cli})"

[[keys.command]]
key = "prefix+p"
type = "pane"
command = "{pi_cmd_clean}"
description = "launch Pi coding agent"

[[keys.command]]
key = "prefix+c"
type = "pane"
command = "{claude_cmd_clean}"
description = "launch Claude Code"

[[keys.command]]
key = "prefix+h"
type = "pane"
command = "{hermes_cmd_clean}"
description = "launch Hermes agent"

[[keys.command]]
key = "prefix+o"
type = "pane"
command = "{omp_cmd_clean}"
description = "launch Oh My Pi agent"
"""
    try:
        atomic_write_text(config_toml_path, toml_content)
        logger.info(f"  -> Configured Herdr settings in {config_toml_path}")
    except Exception as toml_err:
        logger.warning(f"  -> Warning: Could not write Herdr config.toml: {toml_err}")

    # 7. Write herdr-runtime.json metadata
    runtime_meta = {
        "schemaVersion": 1,
        "defaultInnerCliAgent": inner_cli,
        "activeModel": model_name,
        "openaiBaseUrl": llm_base_url,
        "workspaceDir": str(start_dir),
        "claudeOffline": claude_offline,
    }
    try:
        atomic_write_json(herdr_home / "herdr-runtime.json", runtime_meta)
    except Exception as meta_err:
        logger.debug(f"Herdr runtime meta write notice: {meta_err}")

    # 8. Ensure Herdr release-matched skill is available in gateway
    herdr_bin = get_herdr_binary_path()
    if herdr_bin and herdr_bin.exists():
        try:
            res = _run_silent([str(herdr_bin), "--skill"])
            if res.returncode == 0 and res.stdout.strip():
                skill_dir = herdr_home / "skills" / "herdr"
                skill_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_text(skill_dir / "SKILL.md", res.stdout)
                logger.info("  -> Exported release-matched Herdr skill.")
        except Exception as sk_err:
            logger.debug(f"Herdr --skill export notice: {sk_err}")


class HerdrAdapter(HarnessAdapter):
    """Adapter for Herdr terminal runtime multiplexer (meta-harness)."""

    TRACK_CONSOLE_CLOSE = True

    def __init__(self, root_dir, processes):
        super().__init__("herdr", root_dir, processes)
        self.herdr_home = getattr(config, "herdr_dir", HERDR_DIR)
        self.herdr_exe = self._resolve_executable()

    def _resolve_executable(self) -> Path:
        """Find the Herdr executable across workspace tools, bin, Cargo home, AppData, or PATH."""
        bin_path = get_herdr_binary_path()
        if bin_path and bin_path.exists():
            return bin_path
        bin_name = "herdr.exe" if sys.platform == 'win32' else "herdr"
        candidates = [
            self.herdr_home / "bin" / bin_name,
            self.herdr_home / bin_name,
            AI_WORKSPACE_DIR / "tools" / "herdr" / bin_name,
            AI_WORKSPACE_DIR / "tools" / "herdr" / "target" / "release" / bin_name,
            AI_WORKSPACE_DIR / "bin" / bin_name,
        ]
        try:
            user_home = Path.home()
            candidates.extend([
                user_home / ".cargo" / "bin" / bin_name,
                user_home / ".herdr" / "bin" / bin_name,
                user_home / ".herdr" / bin_name,
            ])
        except Exception:
            pass
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        which_path = shutil.which("herdr") or shutil.which("herdr.exe")
        if which_path:
            return Path(which_path)
        return candidates[0]

    def _install_herdr(self) -> bool:
        """Attempt to install Herdr via cargo or official script installer with automatic fallback."""
        logger.info("  -> Attempting Herdr terminal runtime installation...")
        if not self._should_install():
            logger.info("  -> Herdr installation disabled by configuration.")
            return False

        # 1. Pre-wire portable junctions so installer writes into ~/.herdr land in workspace
        try:
            self.herdr_home.mkdir(parents=True, exist_ok=True)
            (self.herdr_home / "bin").mkdir(parents=True, exist_ok=True)
            user_home = Path.home()
            if user_home.exists():
                ensure_directory_junction(user_home / ".herdr", self.herdr_home)
                ensure_directory_junction(user_home / ".config" / "herdr", self.herdr_home)
            appdata_str = os.environ.get("APPDATA")
            if appdata_str:
                ensure_directory_junction(Path(appdata_str) / "herdr", self.herdr_home)
        except Exception as j_err:
            logger.debug(f"Pre-install junction setup notice: {j_err}")

        # 2. Try Cargo if cargo is available
        cargo_bin = shutil.which("cargo") or shutil.which("cargo.exe")
        if cargo_bin:
            try:
                logger.info("  -> Installing Herdr via Cargo (cargo install herdr)...")
                res = subprocess.run([cargo_bin, "install", "herdr", "--locked"], check=False, capture_output=True, text=True, timeout=180)
                if res.returncode == 0:
                    logger.info("  -> Herdr successfully installed via cargo.")
            except Exception as e:
                logger.warning(f"Cargo install notice: {e}")

        # 3. On Windows, try official PowerShell installer
        if sys.platform == 'win32' and not (self._resolve_executable().exists() or shutil.which("herdr")):
            try:
                logger.info("  -> Installing Herdr via official PowerShell installer...")
                ps_cmd = 'irm https://herdr.dev/install.ps1 | iex'
                res = subprocess.run(["powershell.exe", "-ExecutionPolicy", "Bypass", "-NoProfile", "-Command", ps_cmd], check=False, capture_output=True, text=True, timeout=120)
                if res.returncode == 0:
                    logger.info("  -> Herdr successfully installed via official installer.")
                else:
                    logger.debug(f"PowerShell installer output: {res.stdout or res.stderr}")
            except Exception as e:
                logger.warning(f"Official PowerShell installer notice: {e}")

        # 4. On Windows, try curl.exe install.cmd fallback (recommended for endpoint-protected environments)
        if sys.platform == 'win32' and not (self._resolve_executable().exists() or shutil.which("herdr")):
            curl_bin = shutil.which("curl.exe") or shutil.which("curl")
            if curl_bin:
                try:
                    logger.info("  -> Trying official curl install.cmd fallback for endpoint-protected Windows...")
                    temp_cmd = AI_WORKSPACE_DIR / "tmp" / "install_herdr.cmd"
                    temp_cmd.parent.mkdir(parents=True, exist_ok=True)
                    dl_res = subprocess.run([curl_bin, "-fsSLo", str(temp_cmd), "https://herdr.dev/install.cmd"], check=False, capture_output=True, text=True, timeout=30)
                    if dl_res.returncode == 0 and temp_cmd.exists():
                        run_res = subprocess.run(["cmd.exe", "/c", str(temp_cmd)], check=False, capture_output=True, text=True, timeout=120)
                        if run_res.returncode == 0:
                            logger.info("  -> Herdr successfully installed via install.cmd.")
                        temp_cmd.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"install.cmd fallback notice: {e}")

        # 5. Verify resolved executable and make portable if installed externally
        resolved = self._resolve_executable()
        if resolved and resolved.exists() and resolved.is_file():
            self.herdr_exe = resolved
            target_workspace_bin = self.herdr_home / "bin" / resolved.name
            if not target_workspace_bin.exists():
                try:
                    shutil.copy2(resolved, target_workspace_bin)
                    self.herdr_exe = target_workspace_bin
                    logger.info(f"  -> Mirrored Herdr binary into workspace: {self.herdr_exe}")
                except Exception:
                    pass
            logger.info(f"  -> Herdr is ready at: {self.herdr_exe}")
            return True

        which_path = shutil.which("herdr") or shutil.which("herdr.exe")
        if which_path:
            self.herdr_exe = Path(which_path)
            logger.info(f"  -> Herdr located on system PATH: {self.herdr_exe}")
            return True

        logger.warning("  -> Herdr binary not found after installation attempts.")
        logger.warning("     To install manually, run: powershell -ExecutionPolicy Bypass -c \"irm https://herdr.dev/install.ps1 | iex\"")
        logger.warning(f"     Or place 'herdr.exe' into: {self.herdr_home / 'bin'}")
        return False

    def _update_herdr(self) -> bool:
        """Check for updates to Herdr terminal runtime."""
        logger.info("Checking for Herdr updates...")
        return True

    def check(self) -> bool:
        logger.info("\nChecking for Herdr meta-harness...")
        is_installed = self.herdr_exe.exists() or bool(shutil.which("herdr"))

        if is_installed:
            logger.info(f"  -> Herdr is installed at: {self.herdr_exe}")
            if self._should_update():
                if self._update_herdr():
                    self.record_update()
            return True
        else:
            logger.info("  -> Herdr is not installed.")
            return self._install_herdr()

    def start(self, llm_base_url: str, active_model: Optional[str] = None, launch_tui: bool = True) -> None:
        model_name = resolve_model_name(active_model)

        # Pre-flight configuration
        setup_herdr_config(self.herdr_home, llm_base_url, active_model)

        # Clean environment for Herdr and child agent panes
        herdr_env = os.environ.copy()
        herdr_env["OPENAI_BASE_URL"] = llm_base_url
        herdr_env["OPENAI_API_BASE"] = llm_base_url
        herdr_env["OPENAI_API_KEY"] = "dummy"
        herdr_env["OPENAI_MODEL"] = model_name
        herdr_env["MODEL"] = model_name
        herdr_env["HOME"] = str(AI_WORKSPACE_DIR)
        herdr_env["USERPROFILE"] = str(AI_WORKSPACE_DIR)
        herdr_env["PI_HOME"] = str(AI_WORKSPACE_DIR / ".pi")
        herdr_env["PI_CODING_AGENT_DIR"] = str(AI_WORKSPACE_DIR / ".pi" / "agent")
        herdr_env["PI_SKIP_VERSION_CHECK"] = "1"
        herdr_env["OMP_HOME"] = str(AI_WORKSPACE_DIR / ".omp")
        herdr_env["HERMES_HOME"] = str(AI_WORKSPACE_DIR / ".hermes")
        herdr_env["HF_HOME"] = str(AI_WORKSPACE_DIR / "models" / "huggingface")
        herdr_env["HF_HUB_CACHE"] = str(AI_WORKSPACE_DIR / "models" / "huggingface")
        herdr_env["npm_config_allow_git"] = "all"
        herdr_env["HERDR_CONFIG_PATH"] = str(self.herdr_home / "config.toml")
        herdr_env["HERDR_HOME"] = str(self.herdr_home)
        herdr_env["HERDR_ENV"] = "1"
        herdr_env["HERDR_BIN_PATH"] = str(self.herdr_exe)

        if getattr(config, "aside_browser_enabled", True):
            try:
                from local_ai_launcher.paths import get_aside_binary_path
                aside_bin = getattr(config, "aside_browser_bin", None) or get_aside_binary_path()
                if aside_bin and Path(aside_bin).is_file():
                    herdr_env["ASIDE_CLI_DIR"] = str(Path(aside_bin).parent)
                    herdr_env["ASIDE_BINARY"] = str(aside_bin)
            except Exception:
                pass

        if not launch_tui:
            logger.info("  -> Herdr meta-harness configured (headless/auxiliary mode).")
            return

        resolved_exe = self._resolve_executable()
        if not resolved_exe.exists() and not shutil.which("herdr") and not shutil.which("herdr.exe"):
            logger.error(f"Cannot start Herdr: executable not found at '{self.herdr_exe}' or on PATH.")
            return

        self.herdr_exe = resolved_exe
        from local_ai_launcher.harness_model import resolve_meta_harness_inner_cli_harness
        inner_cli = resolve_meta_harness_inner_cli_harness(
            getattr(config, 'choose_harness', []),
            model_name=model_name,
            cfg=config
        )

        target_cwd = str(config.harness_start_dir or PROJECTS_DIR)
        herdr_bin_str = str(self.herdr_exe) if self.herdr_exe.exists() else "herdr"

        logger.info(f"Starting Herdr meta-harness (inner agent: {inner_cli}, model: {model_name}, endpoint: {llm_base_url})...")
        try:
            if sys.platform == 'win32':
                env_setup = build_ps_env(herdr_env, harness_name="herdr")
                herdr_proc = start_in_wt_tab(
                    f'powershell.exe -NoExit -Command "chcp 65001 | Out-Null; [console]::InputEncoding=[console]::OutputEncoding=[System.Text.Encoding]::UTF8; {env_setup}& \'{herdr_bin_str}\'"',
                    cwd=target_cwd,
                    title="Herdr"
                )
            else:
                herdr_proc = subprocess.Popen(
                    [herdr_bin_str],
                    cwd=target_cwd,
                    env=herdr_env
                )
            if herdr_proc:
                assign_process_to_job(herdr_proc)
                self.processes.append(('Herdr meta-harness', herdr_proc, self.TRACK_CONSOLE_CLOSE))
                logger.info("  -> Herdr meta-harness launched successfully.")
        except Exception as e:
            logger.error(f"Error starting Herdr: {e}")
