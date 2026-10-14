import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional

from local_ai_launcher.config import (
    config,
    logger,
    prompt_choice,
    AI_WORKSPACE_DIR,
    LOCALAI_REPO_DIR,
    LOCALAI_BIN_DIR,
    LOCALAI_EXE
)


def check_localai() -> bool:
    """
    Checks if LocalAI repository or assets exist, and handles install/update behaviors.
    Guarantees that local_ai_launcher remains 100% portable by keeping all cloned/downloaded
    artifacts strictly within AI_WORKSPACE_DIR.
    """
    repo_dir = LOCALAI_REPO_DIR
    web_dir = AI_WORKSPACE_DIR / "local_ai_launcher" / "localai" / "web"
    web_static_dir = web_dir / "static"
    is_cloned = repo_dir.exists() and (repo_dir / ".git").exists()

    # 1. Update Handling
    if is_cloned:
        logger.info(f"LocalAI repository found at {repo_dir}.")
        update_behavior = getattr(config, 'localai_update_behavior', 'never_update')
        update_frequency_hrs = getattr(config, 'engine_update_frequency_hrs', {}).get(
            'localai', getattr(config, 'update_frequency_hrs', 168)
        )
        force = getattr(config, 'force_update', False)

        from local_ai_launcher.update_frequency import should_update_component, record_successful_update
        do_update = should_update_component(
            component_name="localai",
            category="engine",
            behavior=update_behavior,
            frequency=update_frequency_hrs,
            force=force,
            prompt_if_existing=True,
        )

        if do_update:
            logger.info("Updating LocalAI repository from GitHub...")
            try:
                res = subprocess.run(
                    ["git", "pull", "--autostash"],
                    cwd=str(repo_dir),
                    timeout=300,
                    capture_output=True,
                    text=True
                )
                if res.returncode == 0:
                    logger.info("  -> LocalAI repository updated successfully.")
                    record_successful_update("engine:localai")
                    upstream_static = repo_dir / "core" / "http" / "static"
                    if upstream_static.exists():
                        _sync_web_assets(upstream_static, web_static_dir)
                    return True
                else:
                    logger.warning(f"  -> Warning updating LocalAI git repo: {res.stderr.strip()}")
            except Exception as e:
                logger.warning(f"  -> Failed to update LocalAI repository: {e}")
        return True

    # 2. Installation Handling
    install_behavior = getattr(config, 'localai_install_behavior', 'existing')
    if install_behavior == 'never_install':
        logger.info("  -> Skipping LocalAI git clone as per configuration (using built-in native engine).")
        return True

    if install_behavior == 'always_install':
        do_install = True
    elif install_behavior == 'existing':
        # If built-in web assets are already present, we have a fully functional portable native engine
        if (web_dir / "index.html").exists() and web_static_dir.exists():
            logger.info("  -> LocalAI built-in native engine & assets ready (portable mode).")
            return True
        do_install = prompt_choice("LocalAI repository not cloned. Clone official mudler/LocalAI from GitHub? (y/n): ", default='n')
    else:
        do_install = False

    if do_install:
        repo_url = getattr(config, 'localai_repo_url', "https://github.com/mudler/LocalAI.git")
        logger.info(f"Cloning LocalAI repository from {repo_url}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
                cwd=str(AI_WORKSPACE_DIR),
                check=True,
                timeout=300
            )
            logger.info(f"  -> LocalAI repository cloned into {repo_dir}.")
            upstream_static = repo_dir / "core" / "http" / "static"
            if upstream_static.exists():
                _sync_web_assets(upstream_static, web_static_dir)
            return True
        except Exception as e:
            logger.warning(f"  -> Failed to clone LocalAI repository: {e}. Falling back to built-in native engine.")
            return False

    return True


def _sync_web_assets(src_dir: Path, dst_dir: Path):
    """Synchronizes upstream Web GUI static assets into local_ai_launcher."""
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for item in src_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src_dir)
                target = dst_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        logger.info(f"  -> Synchronized official Web GUI assets from {src_dir} to {dst_dir}.")
    except Exception as e:
        logger.warning(f"  -> Warning syncing Web GUI assets: {e}")
