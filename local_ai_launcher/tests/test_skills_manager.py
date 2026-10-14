import pytest
from pathlib import Path

from local_ai_launcher.skills_manager import (
    SkillManager,
    DEFAULT_SUITES,
    FLATTENED_CLUTTER_NAMES,
)
from local_ai_launcher.config import LauncherConfig, _flatten_overrides


def test_skills_config_flattening():
    """Verify that [skills] section in launcher.toml maps cleanly to LauncherConfig."""
    overrides = {
        "skills": {
            "enable_gateway": True,
            "auto_update": True,
            "suites": {
                "custom_suite": {
                    "git": "https://example.com/repo.git",
                    "branch": "main",
                }
            }
        }
    }
    flat = _flatten_overrides(overrides)
    assert flat.get("enable_skill_gateway") is True
    assert flat.get("auto_update_skills_on_launch") is True
    assert "custom_suite" in flat.get("skill_suites", {})


def test_ensure_builtin_skills_copies_custom_skills(tmp_path):
    """Verify builtin skills (wipe-agy-chat, cpu-tools, deep-solve, etc.) are copied to destination."""
    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()
    
    # Create fake custom skill
    fake_skill = builtin_dir / "fake-custom"
    fake_skill.mkdir()
    (fake_skill / "SKILL.md").write_text("---\nname: fake-custom\n---\n# Test", encoding="utf-8")
    scripts_dir = fake_skill / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.ps1").write_text("Write-Host 'hello'", encoding="utf-8")

    dest_dir = tmp_path / "gateway"
    dest_dir.mkdir()

    mgr = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=tmp_path / "vendor",
        gateway_dir=dest_dir,
        builtin_dir=builtin_dir,
        suites={},
    )

    provisioned = mgr.ensure_builtin_skills(target_dirs=[dest_dir])
    assert len(provisioned) == 1
    assert (dest_dir / "fake-custom" / "SKILL.md").exists()
    assert (dest_dir / "fake-custom" / "scripts" / "run.ps1").exists()


def test_generate_update_script(tmp_path):
    """Verify update-skills.ps1 is generated with correct workspace paths."""
    mgr = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=tmp_path / ".agents" / "vendor",
        gateway_dir=tmp_path / ".agents" / "skills",
        builtin_dir=tmp_path / "builtin",
    )
    script_path = mgr.generate_update_script()
    assert script_path.exists()
    assert script_path == tmp_path / ".agents" / "update-skills.ps1"
    content = script_path.read_text(encoding="utf-8")
    assert "=== Updating Workspace Skill Suites ===" in content
    assert "git -C $suite.FullName pull --ff-only" in content
    assert ".skills_prompt_snapshot.json" in content


def test_cleanup_flattened_clutter_dry_run(tmp_path):
    """Verify flattened clutter names are correctly identified for pruning."""
    gateway_dir = tmp_path / "skills"
    gateway_dir.mkdir()

    # Create clutter skills
    (gateway_dir / "qa").mkdir()
    (gateway_dir / "ship").mkdir()
    (gateway_dir / "hyperframes-animation").mkdir()
    (gateway_dir / "remotion-markup").mkdir()

    # Create non-clutter skills
    (gateway_dir / "custom-keep").mkdir()
    (gateway_dir / "deep-solve").mkdir()

    mgr = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=tmp_path / "vendor",
        gateway_dir=gateway_dir,
        builtin_dir=tmp_path / "builtin",
    )

    pruned = mgr.cleanup_flattened_clutter(dry_run=True)
    assert "qa" in pruned
    assert "ship" in pruned
    assert "hyperframes-animation" in pruned
    assert "remotion-markup" in pruned
    assert "custom-keep" not in pruned
    assert "deep-solve" not in pruned

    # Actual cleanup
    removed = mgr.cleanup_flattened_clutter(dry_run=False)
    assert not (gateway_dir / "qa").exists()
    assert not (gateway_dir / "ship").exists()
    assert (gateway_dir / "custom-keep").exists()
    assert (gateway_dir / "deep-solve").exists()


def test_extract_skill_sources_and_flattening():
    """Verify that _extract_skill_sources and _flatten_overrides extract suites without leaking 'sources' key."""
    from local_ai_launcher.config import _extract_skill_sources

    # 1. Test [skills.sources.gstack]
    nested_data = {
        "skills": {
            "sources": {
                "gstack": {
                    "git": "https://github.com/garrytan/gstack.git",
                    "enabled": True,
                }
            }
        }
    }
    extracted = _extract_skill_sources(nested_data)
    assert "gstack" in extracted
    assert "sources" not in extracted

    # 2. Test flattening via _flatten_overrides
    overrides = {
        "external_sources": extracted,
        "skills": {
            "enable_gateway": True,
            "sources": {
                "custom_suite": {
                    "git": "https://example.com/custom.git",
                }
            }
        }
    }
    flat = _flatten_overrides(overrides)
    assert flat["enable_skill_gateway"] is True
    suites = flat["skill_suites"]
    assert "gstack" in suites
    assert "custom_suite" in suites
    assert "sources" not in suites
    assert suites["gstack"]["git"] == "https://github.com/garrytan/gstack.git"


def test_suite_enable_disable_gateway_lifecycle(tmp_path, monkeypatch):
    """Verify that enabling/disabling a suite creates and unlinks its gateway junction."""
    vendor_dir = tmp_path / "vendor"
    gateway_dir = tmp_path / "gateway"
    vendor_dir.mkdir()
    gateway_dir.mkdir()

    # Create dummy suite in vendor
    suite_dir = vendor_dir / "my_suite"
    suite_dir.mkdir()
    (suite_dir / "SKILL.md").write_text("# My Suite", encoding="utf-8")

    # In pytest, ensure_directory_junction normally short-circuits. Mock it to create a real directory.
    def fake_junction(link, target):
        Path(link).mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr("local_ai_launcher.skills_manager.ensure_directory_junction", fake_junction)

    # 1. Enabled suite -> junction should be created
    suites_cfg = {
        "my_suite": {
            "git": "https://example.com/suite.git",
            "enabled": True,
            "expose_name": "my-suite",
        }
    }
    mgr = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=vendor_dir,
        gateway_dir=gateway_dir,
        builtin_dir=tmp_path / "builtin",
        suites=suites_cfg,
    )

    linked = mgr.ensure_gateway()
    assert "my-suite" in linked
    assert (gateway_dir / "my-suite").exists()

    # Check status
    st = mgr.status()
    assert st["vendor_suites"]["my_suite"]["enabled"] is True
    assert st["vendor_suites"]["my_suite"]["in_gateway"] is True

    # 2. Disabled suite -> junction should be unlinked
    suites_cfg["my_suite"]["enabled"] = False
    mgr_disabled = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=vendor_dir,
        gateway_dir=gateway_dir,
        builtin_dir=tmp_path / "builtin",
        suites=suites_cfg,
    )
    linked_after = mgr_disabled.ensure_gateway()
    assert "my-suite" not in linked_after
    assert not (gateway_dir / "my-suite").exists()

    # Check status after disabling
    st_disabled = mgr_disabled.status()
    assert st_disabled["vendor_suites"]["my_suite"]["enabled"] is False
    assert st_disabled["vendor_suites"]["my_suite"]["in_gateway"] is False


def test_ensure_scoped_harness_skills(tmp_path):
    """Verify scoped skills are provisioned to specific harness targets without cross-pollination."""
    builtin_dir = tmp_path / "builtin"
    pi_scoped = builtin_dir / "scoped" / "pi" / "donsetch"
    pi_scoped.mkdir(parents=True, exist_ok=True)
    (pi_scoped / "SKILL.md").write_text("# Pi Donsetch", encoding="utf-8")

    dsh_scoped = builtin_dir / "scoped" / "deepseek" / "donsetch"
    dsh_scoped.mkdir(parents=True, exist_ok=True)
    (dsh_scoped / "SKILL.md").write_text("# DSH Donsetch", encoding="utf-8")

    mgr = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=tmp_path / "vendor",
        gateway_dir=tmp_path / "gateway",
        builtin_dir=builtin_dir,
    )

    pi_target = tmp_path / "pi_agent_skills"
    prov_pi = mgr.ensure_scoped_harness_skills("pi", target_dir=pi_target)
    assert prov_pi == ["pi/donsetch"]
    assert (pi_target / "donsetch" / "SKILL.md").exists()
    assert (pi_target / "donsetch" / "SKILL.md").read_text(encoding="utf-8") == "# Pi Donsetch"

    dsh_target = tmp_path / "dsh_skills"
    prov_dsh = mgr.ensure_scoped_harness_skills("dsh", target_dir=dsh_target)
    assert prov_dsh == ["deepseek/donsetch"]
    assert (dsh_target / "donsetch" / "SKILL.md").exists()
    assert (dsh_target / "donsetch" / "SKILL.md").read_text(encoding="utf-8") == "# DSH Donsetch"

    # Unsupported or non-existent harness returns empty list
    prov_unknown = mgr.ensure_scoped_harness_skills("unknown_harness", target_dir=tmp_path / "other")
    assert prov_unknown == []


def test_ensure_builtin_skills_skips_scoped_directory(tmp_path):
    """Verify ensure_builtin_skills never copies the scoped directory into shared gateway or other harnesses."""
    builtin_dir = tmp_path / "builtin"
    (builtin_dir / "wipe-agy-chat").mkdir(parents=True, exist_ok=True)
    (builtin_dir / "wipe-agy-chat" / "SKILL.md").write_text("# Wipe", encoding="utf-8")

    # Scoped directory contains pi/donsetch
    scoped_dir = builtin_dir / "scoped" / "pi" / "donsetch"
    scoped_dir.mkdir(parents=True, exist_ok=True)
    (scoped_dir / "SKILL.md").write_text("# Scoped Donsetch", encoding="utf-8")

    dest_dir = tmp_path / "gateway"
    dest_dir.mkdir(parents=True, exist_ok=True)

    mgr = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=tmp_path / "vendor",
        gateway_dir=dest_dir,
        builtin_dir=builtin_dir,
    )

    provisioned = mgr.ensure_builtin_skills(target_dirs=[dest_dir])
    assert f"{dest_dir.name}/wipe-agy-chat" in provisioned
    assert not any("scoped" in item for item in provisioned)
    assert not any("donsetch" in item for item in provisioned)

    # Verify filesystem
    assert (dest_dir / "wipe-agy-chat" / "SKILL.md").exists()
    assert not (dest_dir / "scoped").exists()
    assert not (dest_dir / "donsetch").exists()


def test_skills_status_audit_with_scoped_skills(tmp_path):
    """Verify status() correctly isolates scoped skills from builtin_skills without false [MISSING] reports."""
    builtin_dir = tmp_path / "builtin"
    (builtin_dir / "deep-solve").mkdir(parents=True, exist_ok=True)
    (builtin_dir / "scoped" / "pi" / "donsetch").mkdir(parents=True, exist_ok=True)

    dest_dir = tmp_path / "gateway"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "deep-solve").mkdir(parents=True, exist_ok=True)

    mgr = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=tmp_path / "vendor",
        gateway_dir=dest_dir,
        builtin_dir=builtin_dir,
    )

    st = mgr.status()
    # 'scoped' should NOT be in builtin_skills
    assert "scoped" not in st["builtin_skills"]
    assert "deep-solve" in st["builtin_skills"]
    assert st["builtin_skills"]["deep-solve"]["in_gateway"] is True

    # 'scoped_skills' should report pi/donsetch
    assert "scoped_skills" in st
    assert "pi/donsetch" in st["scoped_skills"]


def test_rea_suite_lifecycle(tmp_path, monkeypatch):
    """Verify that REA is included in DEFAULT_SUITES and creates both canonical and alias routers."""
    assert "rea" in DEFAULT_SUITES
    assert DEFAULT_SUITES["rea"]["expose_name"] == "reverse-engineer-anything"

    vendor_dir = tmp_path / "vendor"
    gateway_dir = tmp_path / "gateway"
    vendor_dir.mkdir()
    gateway_dir.mkdir()

    # Create dummy REA repository layout in vendor
    rea_skill_dir = vendor_dir / "rea" / ".agents" / "skills" / "reverse-engineer-anything"
    rea_skill_dir.mkdir(parents=True)
    (rea_skill_dir / "SKILL.md").write_text("# REA Skill", encoding="utf-8")

    def fake_junction(link, target):
        Path(link).mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr("local_ai_launcher.skills_manager.ensure_directory_junction", fake_junction)

    # 1. Enabled REA suite
    mgr = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=vendor_dir,
        gateway_dir=gateway_dir,
        builtin_dir=tmp_path / "builtin",
        suites={"rea": {"enabled": True}},
    )

    linked = mgr.ensure_gateway()
    assert "reverse-engineer-anything" in linked
    assert "rea" in linked
    assert (gateway_dir / "reverse-engineer-anything").exists()
    assert (gateway_dir / "rea").exists()

    # 2. Toggle disabled
    mgr_disabled = SkillManager(
        workspace_dir=tmp_path,
        vendor_dir=vendor_dir,
        gateway_dir=gateway_dir,
        builtin_dir=tmp_path / "builtin",
        suites={"rea": {"enabled": False}},
    )
    linked_disabled = mgr_disabled.ensure_gateway()
    assert "reverse-engineer-anything" not in linked_disabled
    assert "rea" not in linked_disabled
    assert not (gateway_dir / "reverse-engineer-anything").exists()
    assert not (gateway_dir / "rea").exists()




