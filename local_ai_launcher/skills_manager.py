"""
Unified Skill Lifecycle & Gateway Manager for local_ai_launcher.

Manages portable installation, Git upstream tracking, and nested router
exposition for agent skill suites (gstack, hyperframes, remotion, superpowers, etc.)
across Antigravity, Pi coding agent, and Hermes Agent.

Architecture:
- Vendor Store (.agents/vendor/): Raw upstream Git clones of entire suites.
- Gateway (.agents/skills/): Curated router junctions exposing only entry routers
  and selected core process skills, eliminating system prompt token bloat (~11.4k -> ~1.8k tokens).
- Builtin Skills (local_ai_launcher/skills/): Checked-in, git-tracked canonical versions
  of custom workspace skills (cpu-tools, deep-solve, reliable-file-read, wipe-agy-chat, update-skills)
  so they can never vanish if the workspace skills directory is wiped.
- Harness Strategy:
  - Pi Coding Agent natively auto-discovers .agents/skills/ across parent directories.
    Skills are never duplicated into .pi/skills/.
  - Hermes Agent requires explicit gateway junctions in .hermes/skills/.
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

from local_ai_launcher.paths import (
    AI_WORKSPACE_DIR,
    LOCAL_AI_LAUNCHER_DIR,
    PI_DIR,
    DSH_DIR,
    HERMES_DIR,
    atomic_write_text,
)
from local_ai_launcher.environment import ensure_directory_junction

logger = logging.getLogger("skills_manager")

# Standard workspace directories
AGENTS_DIR = AI_WORKSPACE_DIR / ".agents"
VENDOR_DIR = AGENTS_DIR / "vendor"
GATEWAY_DIR = AGENTS_DIR / "skills"
BUILTIN_SKILLS_DIR = LOCAL_AI_LAUNCHER_DIR / "skills"
PI_SKILLS_DIR = PI_DIR / "skills"
HERMES_SKILLS_DIR = HERMES_DIR / "skills"
DSH_SKILLS_DIR = DSH_DIR / "skills"

# Default upstream skill suite specifications
DEFAULT_SUITES: Dict[str, Dict[str, Any]] = {
    "gstack": {
        "git": "https://github.com/garrytan/gstack.git",
        "branch": "main",
        "router_subpath": "",  # Root has SKILL.md router
        "post_update": "setup",
        "description": "Router for the gstack skill suite.",
    },
    "hyperframes": {
        "git": "https://github.com/heygen-com/hyperframes.git",
        "branch": "main",
        "router_subpath": "skills/hyperframes",
        "description": "Mandatory entry point for HyperFrames video & animation suite.",
    },
    "remotion": {
        "git": "https://github.com/remotion-dev/remotion.git",
        "branch": "main",
        "router_subpath": "packages/skills/skills/remotion-best-practices",
        "expose_name": "remotion-best-practices",
        "description": "Router for all Remotion skills.",
    },
    "superpowers": {
        "git": "https://github.com/obra/superpowers.git",
        "branch": "main",
        "router_subpath": "skills/using-superpowers",
        "expose_name": "using-superpowers",
        "description": "Superpowers agentic software engineering workflows.",
        "expose_subskills": [
            ("brainstorming", "skills/brainstorming"),
            ("systematic-debugging", "skills/systematic-debugging"),
            ("test-driven-development", "skills/test-driven-development"),
            ("writing-plans", "skills/writing-plans"),
            ("executing-plans", "skills/executing-plans"),
            ("subagent-driven-development", "skills/subagent-driven-development"),
            ("verification-before-completion", "skills/verification-before-completion"),
        ],
    },
    "orca": {
        "git": "https://github.com/stablyai/orca.git",
        "branch": "main",
        "router_subpath": "skills/orca-cli",
        "expose_name": "orca-cli",
        "description": "Operate Orca worktrees, terminals, and automations.",
        "expose_subskills": [
            ("orchestration", "skills/orchestration"),
        ],
    },
    "herdr": {
        "git": "https://github.com/herdrdev/herdr.git",
        "branch": "main",
        "router_subpath": "skills/herdr",
        "expose_name": "herdr",
        "description": "Operate Herdr multiplexer, session manager, and multi-agent coordination.",
    },
    "humanizer": {
        "git": "https://github.com/blader/humanizer.git",
        "branch": "main",
        "router_subpath": "",
        "description": "Rewrite AI-sounding text to read naturally.",
    },
    "impeccable": {
        "git": "https://github.com/pbakaus/impeccable.git",
        "branch": "main",
        "router_subpath": "",
        "description": "Frontend design, polish, typography, and UX review.",
    },
    "skill-inspector": {
        "git": "https://github.com/NVIDIA/SkillSpector.git",
        "branch": "main",
        "router_subpath": "",
        "description": "Review AI agent skills for safety and trustworthiness.",
    },
    "watch": {
        "git": "https://github.com/bradautomates/claude-video.git",
        "branch": "main",
        "router_subpath": "",
        "description": "Watch video files/URLs via yt-dlp and ffmpeg.",
    },
    "find-skills": {
        "git": "https://github.com/vercel-labs/skills.git",
        "branch": "main",
        "router_subpath": "skills/find-skills",
        "description": "Helps users discover and install agent skills.",
    },
    "rea": {
        "git": "https://github.com/morluto/rea.git",
        "branch": "main",
        "router_subpath": ".agents/skills/reverse-engineer-anything",
        "expose_name": "reverse-engineer-anything",
        "description": "Reverse engineer anything across binaries, applications, and runtime behavior.",
        "expose_subskills": [
            ("rea", ".agents/skills/reverse-engineer-anything"),
        ],
    },
}

# Known subskills that were flattened into .agents/skills but should be nested inside vendor
FLATTENED_CLUTTER_NAMES = {
    # rea internal maintenance skills
    "rea-changelog-update", "rea-tool-design",
    # gstack subskills (handled via gstack router)
    "autoplan", "benchmark", "benchmark-models", "browse", "canary", "careful", "codex",
    "context-restore", "context-save", "cso", "design-consultation", "design-html",
    "design-review", "design-shotgun", "deslop-shared-libs", "devex-review", "diagram",
    "document-generate", "document-release", "freeze", "gstack-upgrade", "guard", "health",
    "ios-clean", "ios-design-review", "ios-fix", "ios-qa", "ios-sync", "land-and-deploy",
    "landing-report", "make-pdf", "office-hours", "open-gstack-browser", "pair-agent",
    "plan-ceo-review", "plan-design-review", "plan-devex-review", "plan-eng-review",
    "plan-tune", "qa", "qa-only", "retro", "review", "scrape", "setup-browser-cookies",
    "setup-deploy", "setup-gbrain", "ship", "skillify", "spec", "sync-gbrain", "unfreeze",
    # hyperframes subskills (handled via hyperframes router)
    "embedded-captions", "faceless-explainer", "figma", "general-video", "hyperframes-animation",
    "hyperframes-audio", "hyperframes-cli", "hyperframes-core", "hyperframes-creative",
    "hyperframes-keyframes", "hyperframes-registry", "hyperframes-studio", "media-use",
    "motion-graphics", "music-to-video", "pr-to-video", "product-launch-video", "slideshow",
    "talking-head-recut",
    # remotion subskills (handled via remotion-best-practices router)
    "remotion-captions", "remotion-create", "remotion-docs", "remotion-interactivity",
    "remotion-maps", "remotion-markup", "remotion-multimedia", "remotion-render",
    "remotion-saas", "remotion-studio", "remotion-to-hyperframes", "remotion-upgrade",
    # superpowers subskills that don't need top-level gateway entries
    "diagnosing-superpowers", "dispatching-parallel-agents", "finishing-a-development-branch",
    "receiving-code-review", "requesting-code-review", "using-git-worktrees", "writing-skills",
}


class SkillManager:
    """Orchestrates portable workspace skill installation, updates, and harness gateways."""

    def __init__(
        self,
        workspace_dir: Path = AI_WORKSPACE_DIR,
        vendor_dir: Path = VENDOR_DIR,
        gateway_dir: Path = GATEWAY_DIR,
        builtin_dir: Path = BUILTIN_SKILLS_DIR,
        suites: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.vendor_dir = Path(vendor_dir)
        self.gateway_dir = Path(gateway_dir)
        self.builtin_dir = Path(builtin_dir)

        # Merge user suites cleanly over DEFAULT_SUITES
        merged = {k: dict(v) for k, v in DEFAULT_SUITES.items()}
        if suites:
            for k, v in suites.items():
                if isinstance(v, dict):
                    if k in merged:
                        merged[k].update(v)
                    else:
                        merged[k] = dict(v)
        self.suites = merged

    def ensure_builtin_skills(self, target_dirs: Optional[List[Path]] = None) -> List[str]:
        """
        Deploy git-tracked canonical custom skills from local_ai_launcher/skills/
        to gateway and harness directories, preventing accidental loss.
        """
        if not self.builtin_dir.exists():
            logger.warning(f"Builtin skills directory not found: {self.builtin_dir}")
            return []

        if target_dirs is None:
            # Pi natively discovers .agents/skills/ directly; Hermes requires its own harness directory
            target_dirs = [self.gateway_dir, HERMES_SKILLS_DIR]

        provisioned = []
        for skill_dir in self.builtin_dir.iterdir():
            if not skill_dir.is_dir() or skill_dir.name == "scoped":
                continue
            skill_name = skill_dir.name
            for dest_root in target_dirs:
                try:
                    dest_root.mkdir(parents=True, exist_ok=True)
                    dest_skill = dest_root / skill_name
                    
                    # If dest_skill is a junction, remove it before copying
                    if dest_skill.is_symlink() or (dest_skill.exists() and hasattr(os, "readlink")):
                        try:
                            dest_skill.unlink()
                        except OSError:
                            shutil.rmtree(dest_skill, ignore_errors=True)
                    
                    # Synchronize folder content
                    shutil.copytree(skill_dir, dest_skill, dirs_exist_ok=True)
                    provisioned.append(f"{dest_root.name}/{skill_name}")
                except Exception as e:
                    logger.warning(f"Failed to provision builtin skill '{skill_name}' to {dest_root}: {e}")

        logger.info(f"  -> Verified builtin workspace skills from {self.builtin_dir}")
        return provisioned

    def ensure_scoped_harness_skills(self, harness_name: str, target_dir: Optional[Path] = None) -> List[str]:
        """
        Deploy harness-scoped skills (e.g. donsetch for Pi and DSH) to their respective
        harness directories without leaking into the shared .agents/skills gateway.
        """
        norm_name = "deepseek" if harness_name in ("dsh", "deepseek") else harness_name
        scoped_src = self.builtin_dir / "scoped" / norm_name
        if not scoped_src.exists():
            return []

        if target_dir is None:
            if norm_name == "pi":
                target_dir = PI_DIR / "agent" / "skills"
            elif norm_name == "deepseek":
                target_dir = DSH_SKILLS_DIR
            else:
                return []

        provisioned = []
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for skill_dir in scoped_src.iterdir():
            if not skill_dir.is_dir():
                continue
            dest_skill = target_dir / skill_dir.name
            try:
                if dest_skill.is_symlink() or (dest_skill.exists() and hasattr(os, "readlink")):
                    try:
                        dest_skill.unlink()
                    except OSError:
                        shutil.rmtree(dest_skill, ignore_errors=True)
                shutil.copytree(skill_dir, dest_skill, dirs_exist_ok=True)
                provisioned.append(f"{norm_name}/{skill_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to provision scoped skill '{skill_dir.name}' to {target_dir}: {e}")

        logger.info(f"  -> Verified scoped skills for {harness_name} in {target_dir}")
        return provisioned

    def sync_suite(self, name: str, cfg: Dict[str, Any], update: bool = False) -> bool:
        """Clone or fast-forward update a single vendor suite."""
        if cfg.get("enabled", True) is False:
            logger.info(f"  -> Skipping disabled suite: {name}")
            return False

        git_url = cfg.get("git")
        if not git_url:
            return False

        suite_dest = self.vendor_dir / name
        branch = cfg.get("branch", "main")
        self.vendor_dir.mkdir(parents=True, exist_ok=True)

        if not suite_dest.exists() or not (suite_dest / ".git").exists():
            logger.info(f"  -> Cloning {name} from {git_url} (branch: {branch})...")
            try:
                cmd = ["git", "clone", "--depth", "1", "-b", branch, git_url, str(suite_dest)]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode != 0:
                    logger.warning(f"Failed to clone {name}: {res.stderr.strip()}")
                    return False
                logger.info(f"  -> Successfully cloned {name}.")
            except Exception as e:
                logger.error(f"Error cloning {name}: {e}")
                return False
        elif update:
            logger.info(f"  -> Updating {name}...")
            try:
                # Discard regenerable markdown dirt before pull
                subprocess.run(
                    ["git", "-C", str(suite_dest), "checkout", "--", "SKILL.md", "*/SKILL.md"],
                    capture_output=True,
                    check=False,
                )
                res = subprocess.run(
                    ["git", "-C", str(suite_dest), "pull", "--ff-only", "--autostash", "origin", branch],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0:
                    logger.info(f"  -> Updated {name}: {res.stdout.strip()}")
                else:
                    logger.warning(f"  -> Pull refused or failed for {name}: {res.stderr.strip()}")
            except Exception as e:
                logger.warning(f"Failed to pull {name}: {e}")

        # Post-update hook (e.g. gstack setup script)
        post_update = cfg.get("post_update")
        if post_update and update:
            hook_script = suite_dest / post_update
            if hook_script.exists():
                logger.info(f"  -> Executing post-update hook for {name} ({post_update})...")
                self._run_bash_script(hook_script, cwd=suite_dest)

        return True

    def _run_bash_script(self, script_path: Path, cwd: Path) -> None:
        """Run a shell script portably on Windows using Git Bash if available."""
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        posix_cwd = str(cwd).replace("\\", "/")
        posix_script = str(script_path).replace("\\", "/")

        try:
            if git_bash.exists():
                subprocess.run([str(git_bash), "-c", f"cd '{posix_cwd}' && ./{script_path.name}"], check=False)
            else:
                subprocess.run(["bash", "-c", f"cd '{posix_cwd}' && ./{script_path.name}"], check=False)
        except Exception as e:
            logger.debug(f"Bash hook execution notice: {e}")

    def ensure_gateway(self) -> List[str]:
        """Create junctions in .agents/skills/ for entry routers and exposed subskills."""
        self.gateway_dir.mkdir(parents=True, exist_ok=True)
        linked = []

        for name, cfg in self.suites.items():
            expose_name = cfg.get("expose_name", name)
            link_path = self.gateway_dir / expose_name

            # If suite is explicitly disabled, remove gateway junction
            if cfg.get("enabled", True) is False:
                if link_path.exists():
                    try:
                        if sys.platform == "win32":
                            os.rmdir(link_path)
                        else:
                            link_path.unlink()
                        logger.info(f"  -> Unlinked disabled gateway router: {expose_name}")
                    except Exception as e:
                        logger.debug(f"Notice unlinking {expose_name}: {e}")
                for sub_entry in cfg.get("expose_subskills", []):
                    if isinstance(sub_entry, (list, tuple)):
                        sub_name = sub_entry[0]
                    elif isinstance(sub_entry, str) and ":" in sub_entry:
                        sub_name = sub_entry.split(":", 1)[0]
                    else:
                        sub_name = Path(sub_entry).name
                    sub_link = self.gateway_dir / sub_name
                    if sub_link.exists():
                        try:
                            os.rmdir(sub_link) if sys.platform == "win32" else sub_link.unlink()
                        except Exception:
                            pass
                continue

            suite_dir = self.vendor_dir / name
            if not suite_dir.exists():
                continue

            subpath = cfg.get("router_subpath") or cfg.get("router_path") or ""
            router_target = (suite_dir / subpath).resolve() if subpath else suite_dir.resolve()

            if router_target.exists():
                if ensure_directory_junction(link_path, router_target):
                    linked.append(expose_name)

            # Expose selected subskills
            for sub_entry in cfg.get("expose_subskills", []):
                if isinstance(sub_entry, (list, tuple)):
                    sub_name, sub_rel = sub_entry[0], sub_entry[1]
                elif isinstance(sub_entry, str) and ":" in sub_entry:
                    sub_name, sub_rel = sub_entry.split(":", 1)
                else:
                    sub_name = Path(sub_entry).name
                    sub_rel = sub_entry
                sub_target = (suite_dir / sub_rel).resolve()
                sub_link = self.gateway_dir / sub_name
                if sub_target.exists():
                    if ensure_directory_junction(sub_link, sub_target):
                        linked.append(sub_name)

        logger.info(f"  -> Verified gateway router junctions in {self.gateway_dir}")
        return linked

    def cleanup_flattened_clutter(self, dry_run: bool = False) -> List[str]:
        """
        Safely remove flattened suite subskills from .agents/skills/
        that are now accessible via their nested vendor routers.
        """
        if not self.gateway_dir.exists():
            return []

        removed = []
        for item in self.gateway_dir.iterdir():
            if not item.is_dir():
                continue
            name = item.name

            # Check if this item is in the flattened clutter list
            if name in FLATTENED_CLUTTER_NAMES:
                # Never remove if it's already a junction pointing to vendor
                try:
                    import stat
                    st = item.lstat()
                    if bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                        continue
                except Exception:
                    pass

                if dry_run:
                    removed.append(name)
                else:
                    try:
                        shutil.rmtree(item)
                        removed.append(name)
                    except Exception as e:
                        logger.warning(f"Could not remove flattened clutter directory '{name}': {e}")

        if removed and not dry_run:
            logger.info(f"  -> Pruned {len(removed)} flattened suite subskills from {self.gateway_dir}")
        return removed

    def propagate_to_harnesses(self) -> None:
        """
        Synchronize gateway skills into harness directories and invalidate prompt caches.
        Note: Pi natively auto-discovers .agents/skills/ and does not require duplication.
        Hermes requires explicit gateway junctions in .hermes/skills/.
        """
        # 1. Hermes
        if HERMES_SKILLS_DIR.parent.exists():
            HERMES_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            for skill_dir in self.gateway_dir.iterdir():
                if skill_dir.is_dir():
                    link_dest = HERMES_SKILLS_DIR / skill_dir.name
                    ensure_directory_junction(link_dest, skill_dir.resolve())

            # Invalidate Hermes prompt snapshot so prompt regenerates cleanly without 71KB bloat
            snapshot_file = HERMES_DIR / ".skills_prompt_snapshot.json"
            if snapshot_file.exists():
                try:
                    snapshot_file.unlink()
                    logger.info("  -> Invalidated stale Hermes .skills_prompt_snapshot.json")
                except Exception as e:
                    logger.debug(f"Could not remove Hermes snapshot: {e}")

    def generate_update_script(self) -> Path:
        """Generate and maintain update-skills.ps1 within the workspace."""
        target_agents_dir = (self.workspace_dir / ".agents") if self.workspace_dir else AGENTS_DIR
        script_path = target_agents_dir / "update-skills.ps1"
        target_agents_dir.mkdir(parents=True, exist_ok=True)

        ws_root = str(self.workspace_dir or AI_WORKSPACE_DIR).replace("/", "\\")
        ps_content = """<#
.SYNOPSIS
    Workspace Skill Suite Updater (Managed by local_ai_launcher)
    Updates all vendor Git suites, preserves router nesting, and clears prompt cache.
#>
param(
    [switch]$CheckOnly,
    [switch]$Force
)

$ErrorActionPreference = "Continue"
$WorkspaceRoot = "__WORKSPACE_ROOT__"
$VendorDir = Join-Path $WorkspaceRoot ".agents\\vendor"
$HermesSnapshot = Join-Path $WorkspaceRoot ".hermes\\.skills_prompt_snapshot.json"
$GitBash = "C:\\Program Files\\Git\\bin\\bash.exe"

Write-Host "=== Updating Workspace Skill Suites ===" -ForegroundColor Cyan

$Suites = Get-ChildItem -Directory $VendorDir -ErrorAction SilentlyContinue | Where-Object {
    Test-Path (Join-Path $_.FullName ".git")
}

if (-not $Suites) {
    Write-Warning "No git suites found in $VendorDir. Run 'python -m local_ai_launcher --sync-skills' first."
    exit 0
}

foreach ($suite in $Suites) {
    Write-Host "`nChecking $($suite.Name)..." -ForegroundColor Yellow
    
    # 1. Discard regenerable markdown dirt before pull
    git -C $suite.FullName checkout -- "SKILL.md" "*/SKILL.md" 2>$null

    # 2. Fast-forward pull with autostash
    if ($CheckOnly) {
        git -C $suite.FullName fetch origin
        $behind = (git -C $suite.FullName rev-list HEAD..origin/main --count 2>$null)
        if ($behind) {
            Write-Host "  -> Behind origin/main by $($behind.Trim()) commit(s)."
        } else {
            Write-Host "  -> Up to date with origin."
        }
    } else {
        $pullOut = git -C $suite.FullName pull --ff-only --autostash origin main 2>&1
        Write-Host "  -> $pullOut"

        # 3. Execute setup/post-update hooks if present (using Git Bash on Windows)
        $setupScript = Join-Path $suite.FullName "setup"
        if (Test-Path $setupScript) {
            Write-Host "  -> Running post-update setup via Bash..." -ForegroundColor DarkGray
            if (Test-Path $GitBash) {
                & $GitBash -c "cd '$($suite.FullName.Replace('\\','/'))' && ./setup"
            } else {
                bash -c "cd '$($suite.FullName.Replace('\\','/'))' && ./setup"
            }
        }
    }
}

# 4. Invalidate Hermes Prompt Snapshot to keep prompt lean
if (Test-Path $HermesSnapshot) {
    Remove-Item $HermesSnapshot -Force
    Write-Host "`nInvalidated Hermes skills prompt snapshot." -ForegroundColor Green
}

Write-Host "`n=== Skill update complete ===" -ForegroundColor Cyan
"""
        atomic_write_text(script_path, ps_content.replace("__WORKSPACE_ROOT__", ws_root))
        logger.info(f"  -> Generated updater script at {script_path}")
        return script_path

    def sync(self, update: bool = False, cleanup: bool = True) -> Dict[str, Any]:
        """Execute full skill lifecycle synchronization."""
        logger.info("Synchronizing workspace skills & upstream suites...")
        
        # 1. Ensure git-tracked custom builtin skills exist in gateway and harnesses
        builtins = self.ensure_builtin_skills()

        # 2. Clone or update vendor git suites
        suites_synced = []
        for name, cfg in self.suites.items():
            if self.sync_suite(name, cfg, update=update):
                suites_synced.append(name)

        # 3. Create clean router gateway junctions
        routers = self.ensure_gateway()

        # 4. Clean up flattened clutter from .agents/skills/
        pruned = []
        if cleanup:
            pruned = self.cleanup_flattened_clutter()

        # 5. Propagate clean gateway to Pi and Hermes
        self.propagate_to_harnesses()

        # 6. Generate standalone updater script
        script_path = self.generate_update_script()

        logger.info(f"Skill synchronization complete: {len(suites_synced)} suites active, {len(pruned)} clutter folders pruned.")
        return {
            "builtins": builtins,
            "suites": suites_synced,
            "routers": routers,
            "pruned": pruned,
            "update_script": str(script_path),
        }

    def status(self) -> Dict[str, Any]:
        """Audit and return current status of builtins, vendor suites, and gateway routers."""
        builtins_status = {}
        if self.builtin_dir.exists():
            for item in self.builtin_dir.iterdir():
                if item.is_dir() and item.name != "scoped":
                    gateway_target = self.gateway_dir / item.name
                    builtins_status[item.name] = {
                        "source": str(item),
                        "in_gateway": gateway_target.exists(),
                    }

        scoped_status = {}
        scoped_dir = self.builtin_dir / "scoped"
        if scoped_dir.exists():
            for harness_folder in scoped_dir.iterdir():
                if harness_folder.is_dir():
                    harness_name = harness_folder.name
                    target = PI_DIR / "agent" / "skills" if harness_name == "pi" else (DSH_SKILLS_DIR if harness_name in ("deepseek", "dsh") else None)
                    for sk in harness_folder.iterdir():
                        if sk.is_dir():
                            in_target = (target / sk.name).exists() if target else False
                            scoped_status[f"{harness_name}/{sk.name}"] = {
                                "source": str(sk),
                                "target": str(target / sk.name) if target else None,
                                "provisioned": in_target,
                            }

        vendor_status = {}
        for name, cfg in self.suites.items():
            suite_dir = self.vendor_dir / name
            expose_name = cfg.get("expose_name", name)
            gateway_link = self.gateway_dir / expose_name
            is_junction = False
            if gateway_link.exists():
                try:
                    import stat
                    st = gateway_link.lstat()
                    is_junction = bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
                except Exception:
                    pass

            vendor_status[name] = {
                "cloned": suite_dir.exists() and (suite_dir / ".git").exists(),
                "enabled": cfg.get("enabled", True),
                "expose_name": expose_name,
                "in_gateway": gateway_link.exists(),
                "is_junction": is_junction,
                "git": cfg.get("git", ""),
            }

        total_gateway = len([d for d in self.gateway_dir.iterdir() if d.is_dir()]) if self.gateway_dir.exists() else 0
        return {
            "builtin_skills": builtins_status,
            "scoped_skills": scoped_status,
            "vendor_suites": vendor_status,
            "total_gateway_skills": total_gateway,
        }


def main():
    """CLI entrypoint for managing workspace skills and upstream git suites."""
    import argparse
    parser = argparse.ArgumentParser(description="Manage workspace skills and upstream git suites.")
    parser.add_argument("action", nargs="?", default="status", choices=["sync", "update", "clean", "status"],
                        help="Action to perform: sync, update, clean, status (default: status)")
    parser.add_argument("--update", action="store_true", help="Pull latest updates for git suites during sync")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without writing or deleting")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from local_ai_launcher.config import config
    mgr = SkillManager(suites=config.skill_suites)

    if args.action == "status":
        st = mgr.status()
        print("\n--- Workspace Skills Status ---")
        print(f"Total Gateway Skills (.agents/skills/): {st['total_gateway_skills']}")
        print("\n[Tier 1 Builtin Custom Skills Insurance]:")
        for k, v in sorted(st["builtin_skills"].items()):
            gw = "[OK] in gateway" if v["in_gateway"] else "[MISSING] in gateway"
            print(f"  * {k:20} -> {gw}")
        print("\n[Tier 2 External Git Suites Insurance]:")
        for k, v in sorted(st["vendor_suites"].items()):
            cloned = "[CLONED]" if v["cloned"] else "[NOT CLONED]"
            en = "enabled" if v["enabled"] else "DISABLED"
            gw = "linked" if v["in_gateway"] else "unlinked"
            print(f"  * {k:18} ({en:8}) -> {cloned:12} -> router: {v['expose_name']} ({gw})")
    elif args.action in ("sync", "update") or args.update:
        res = mgr.sync(update=(args.action == "update" or args.update))
        print("\nSync complete:")
        print(f"  Active Suites:          {len(res['suites'])}")
        print(f"  Gateway Routers:        {len(res['routers'])}")
        print(f"  Builtin Insured Skills: {len(res['builtins'])}")
        print(f"  Pruned Clutter Folders: {len(res['pruned'])}")
    elif args.action == "clean":
        pruned = mgr.cleanup_flattened_clutter(dry_run=args.dry_run)
        print(f"Pruned {len(pruned)} clutter folders.")


if __name__ == "__main__":
    main()
