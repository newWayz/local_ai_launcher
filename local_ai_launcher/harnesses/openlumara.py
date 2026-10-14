import subprocess
import time
import sys
import os
import shutil
from pathlib import Path
from typing import Optional

from local_ai_launcher.config import *
from local_ai_launcher.environment import *
from local_ai_launcher.models import *
from local_ai_launcher.process import *
from local_ai_launcher.process import run_silent_subprocess as _run_silent
from local_ai_launcher.harness_base import HarnessAdapter
from local_ai_launcher.bridges.orca_openlumara.configurator import (
    configure_orca_profile_for_openlumara,
    install_openlumara_wrappers,
    setup_openlumara_config,
)


class OpenLumaraAdapter(HarnessAdapter):
    """Adapter for OpenLumara AI framework harness and Orca ADE bridge."""

    TRACK_CONSOLE_CLOSE = True

    def __init__(self, root_dir, processes):
        super().__init__("openlumara", root_dir, processes)
        self.repo_url = getattr(config, 'openlumara_repo_url', "https://github.com/Rose22/openlumara.git")
        self.lumara_dir = getattr(config, 'openlumara_dir', AI_WORKSPACE_DIR / "openlumara")
        self.lumara_venv = self.lumara_dir / "venv"
        self.lumara_python = (self.lumara_venv / "Scripts" / "python.exe") if sys.platform == 'win32' else (self.lumara_venv / "bin" / "python")

    def _install_openlumara(self) -> bool:
        """Clone and install OpenLumara into its own isolated virtual environment."""
        logger.info(f"  -> Installing OpenLumara from {self.repo_url}...")
        try:
            if not self.lumara_dir.exists():
                subprocess.run(["git", "clone", self.repo_url, str(self.lumara_dir)], check=True)
                logger.info(f"  -> Cloned OpenLumara into {self.lumara_dir}")

            # Create isolated venv using uv for speed and reliability
            uv_exe = ensure_uv()
            python_base = AI_WORKSPACE_DIR / ".venv" / "Scripts" / "python.exe"
            if not python_base.exists():
                python_base = Path(sys.executable)

            if not self.lumara_venv.exists():
                logger.info("  -> Creating dedicated virtual environment for OpenLumara...")
                if uv_exe:
                    subprocess.run([uv_exe, "venv", str(self.lumara_venv), "--python", str(python_base)], check=True)
                else:
                    subprocess.run([str(python_base), "-m", "venv", str(self.lumara_venv)], check=True)

            req_file = self.lumara_dir / "requirements.txt"
            if req_file.exists():
                logger.info("  -> Installing OpenLumara requirements...")
                pkgs = ["-r", str(req_file), "prompt_toolkit", "rich", "pip"]
                # Add pyreadline3 for Windows prompt_toolkit compatibility
                if sys.platform == 'win32':
                    pkgs.append("pyreadline3")

                if uv_exe:
                    _run_silent([uv_exe, "pip", "install"] + pkgs + ["--python", str(self.lumara_python)], check=True)
                else:
                    _run_silent([str(self.lumara_python), "-m", "pip", "install"] + pkgs, check=True)

            # Deploy executable wrappers and sync config
            install_openlumara_wrappers(AI_WORKSPACE_DIR)
            setup_openlumara_config(self.lumara_dir)
            logger.info("  -> OpenLumara installation completed successfully.")
            return True

        except Exception as e:
            logger.error(f"  -> Error installing OpenLumara: {e}")
            return False

    def _update_openlumara(self) -> bool:
        """Pull latest upstream changes without touching any custom code."""
        if not self.lumara_dir.exists():
            return False

        logger.info("  -> Checking for OpenLumara updates...")
        try:
            res = _run_silent(["git", "pull", "--ff-only"], cwd=str(self.lumara_dir))
            if res.returncode == 0:
                logger.info(f"  -> OpenLumara updated: {res.stdout.strip()}")
            else:
                logger.warning(f"  -> OpenLumara git pull notice: {res.stderr.strip()}")

            install_openlumara_wrappers(AI_WORKSPACE_DIR)
            setup_openlumara_config(self.lumara_dir)
            return True
        except Exception as e:
            logger.warning(f"  -> Could not update OpenLumara: {e}")
            return False

    def check(self) -> bool:
        """Check if OpenLumara is installed and ready. Register with Orca."""
        logger.info("\nChecking for OpenLumara harness...")
        is_installed = self.lumara_dir.exists() and self.lumara_python.exists()

        if not is_installed:
            if self._should_install():
                if not self._install_openlumara():
                    return False
            else:
                logger.info("  -> OpenLumara is not installed.")
                return False
        else:
            logger.info(f"  -> OpenLumara is installed at: {self.lumara_dir}")
            if self._should_update():
                if self._update_openlumara():
                    self.record_update()

        # Always ensure Orca ADE profile, wrapper scripts, and on-disk config are synced
        install_openlumara_wrappers(AI_WORKSPACE_DIR)
        configure_orca_profile_for_openlumara()
        setup_openlumara_config(self.lumara_dir)
        return True

    def start(self, llm_base_url: str, active_model: Optional[str] = None, launch_tui: bool = True) -> None:
        """Start OpenLumara or configure it for Orca ADE."""
        model_name = resolve_model_name(active_model)
        mode = getattr(config, 'openlumara_mode', 'cli').lower().strip()

        # Update on-disk config conforming to active engine (url, model, contextWindow, thinking)
        setup_openlumara_config(
            lumara_dir=self.lumara_dir,
            llm_base_url=llm_base_url,
            active_model=model_name,
            mode=mode
        )

        # Update Orca ADE settings with active LLM endpoint and model
        configure_orca_profile_for_openlumara(
            llm_base_url=llm_base_url,
            active_model=model_name
        )

        # If launched as an auxiliary agent for Orca, configuration is sufficient
        if not launch_tui:
            logger.info(f"OpenLumara is configured and available in Orca ADE (model: {model_name}, endpoint: {llm_base_url}).")
            return

        start_dir = config.harness_start_dir or AI_WORKSPACE_DIR
        config_file = self.lumara_dir / "config.yml"

        env_setup = build_ps_env({
            'OPENAI_BASE_URL': llm_base_url,
            'OPENAI_API_BASE': llm_base_url,
            'OPENAI_API_KEY': 'dummy',
            'OPENAI_MODEL': model_name,
            'MODEL': model_name,
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUNBUFFERED': '1',
        })

        encoding_preamble = (
            "chcp 65001 | Out-Null; "
            "[console]::InputEncoding=[console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        )

        if mode == 'webui':
            logger.info(f"Starting OpenLumara WebUI (model: {model_name}, endpoint: {llm_base_url})...")
            cmd = [
                str(self.lumara_python),
                str(self.lumara_dir / "main.py"),
                "--disable_auto_installer",
                "--config", str(config_file),
                "--api.url", llm_base_url,
                "--model.name", model_name
            ]
            lumara_proc = start_process(
                "OpenLumara WebUI",
                cmd,
                str(self.lumara_dir),
                show_window=False
            )
            if lumara_proc:
                self.processes.append(('OpenLumara WebUI', lumara_proc, self.TRACK_CONSOLE_CLOSE))
                time.sleep(2)
                import webbrowser
                webbrowser.open("http://localhost:5000")
        else:
            logger.info(f"Starting OpenLumara CLI (model: {model_name}, endpoint: {llm_base_url})...")
            from local_ai_launcher.harness_base import escape_ps_arg
            esc_py = escape_ps_arg(self.lumara_python)
            esc_main = escape_ps_arg(self.lumara_dir / 'main.py')
            esc_cfg = escape_ps_arg(config_file)
            esc_url = escape_ps_arg(llm_base_url)
            esc_model = escape_ps_arg(model_name)
            cmd = (
                f"& '{esc_py}' '{esc_main}' "
                f"--cli --disable_auto_installer --config '{esc_cfg}' "
                f"--api.url '{esc_url}' --model.name '{esc_model}'"
            )
            try:
                full_command = f'powershell.exe -NoExit -Command "{encoding_preamble}{env_setup}{cmd}"'
                lumara_proc = start_in_wt_tab(full_command, cwd=str(start_dir))
                self.processes.append(('OpenLumara CLI', lumara_proc, self.TRACK_CONSOLE_CLOSE))
            except Exception as e:
                logger.error(f"Error starting OpenLumara CLI: {e}")
