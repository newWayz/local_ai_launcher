"""
Unit and integration tests for update frequency management and gating.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from local_ai_launcher.config import LauncherConfig, config
from local_ai_launcher.harness_base import HarnessAdapter
from local_ai_launcher.update_frequency import (
    parse_frequency_hours,
    parse_frequency_seconds,
    format_duration,
    format_frequency_hrs,
    get_last_update,
    record_successful_update,
    is_update_due,
    should_update_component,
    load_update_state,
    save_update_state,
    ONE_DAY,
    ONE_WEEK,
    ONE_MONTH,
    ONE_HOUR,
    INFINITY,
)


def test_parse_frequency_named():
    """Verify parsing of standard named frequency strings."""
    assert parse_frequency_seconds("always") == 0.0
    assert parse_frequency_seconds("every") == 0.0
    assert parse_frequency_seconds("hourly") == ONE_HOUR
    assert parse_frequency_seconds("daily") == ONE_DAY
    assert parse_frequency_seconds("weekly") == ONE_WEEK
    assert parse_frequency_seconds("biweekly") == 14 * ONE_DAY
    assert parse_frequency_seconds("monthly") == ONE_MONTH
    assert parse_frequency_seconds("never") == INFINITY
    assert parse_frequency_seconds("disabled") == INFINITY
    assert parse_frequency_seconds("off") == INFINITY


def test_parse_frequency_numeric_and_units():
    """Verify numeric values in hours, unit strings, and special values."""
    # Bare numeric values represent hours
    assert parse_frequency_seconds(1) == ONE_HOUR
    assert parse_frequency_seconds(24) == ONE_DAY
    assert parse_frequency_seconds(168) == ONE_WEEK
    assert parse_frequency_seconds(1.5) == 1.5 * ONE_HOUR
    assert parse_frequency_seconds(0) == 0.0
    assert parse_frequency_seconds(-1) == INFINITY

    # Bare numeric strings represent hours
    assert parse_frequency_seconds("1") == ONE_HOUR
    assert parse_frequency_seconds("24") == ONE_DAY
    assert parse_frequency_seconds("0") == 0.0

    # Custom unit strings
    assert parse_frequency_seconds("3d") == 3 * ONE_DAY
    assert parse_frequency_seconds("3 days") == 3 * ONE_DAY
    assert parse_frequency_seconds("12h") == 12 * ONE_HOUR
    assert parse_frequency_seconds("12 hours") == 12 * ONE_HOUR
    assert parse_frequency_seconds("30m") == 30 * 60.0
    assert parse_frequency_seconds("60s") == 60.0
    assert parse_frequency_seconds("2w") == 14 * ONE_DAY
    assert parse_frequency_seconds("1mo") == ONE_MONTH


def test_parse_frequency_hours():
    """Verify parsing frequency directly to hours."""
    assert parse_frequency_hours(1) == 1.0
    assert parse_frequency_hours(24) == 24.0
    assert parse_frequency_hours(0) == 0.0
    assert parse_frequency_hours(-1) == INFINITY
    assert parse_frequency_hours("1") == 1.0
    assert parse_frequency_hours("24") == 24.0
    assert parse_frequency_hours("hourly") == 1.0
    assert parse_frequency_hours("daily") == 24.0
    assert parse_frequency_hours("weekly") == 168.0
    assert parse_frequency_hours("monthly") == 720.0
    assert parse_frequency_hours("always") == 0.0
    assert parse_frequency_hours("never") == INFINITY
    assert parse_frequency_hours("12h") == 12.0
    assert parse_frequency_hours("3d") == 72.0
    assert parse_frequency_hours("30m") == 0.5


def test_parse_frequency_defaults_and_fallbacks():
    """Verify fallback on invalid or empty values."""
    assert parse_frequency_seconds(None, default="weekly") == ONE_WEEK
    assert parse_frequency_seconds("", default="daily") == ONE_DAY
    assert parse_frequency_seconds("invalid_garbage", default="weekly") == ONE_WEEK


def test_format_duration():
    """Verify human-readable duration formatting."""
    assert format_duration(INFINITY) == "never"
    assert format_duration(0.0) == "always"
    assert format_duration(ONE_DAY) == "1 day"
    assert format_duration(7 * ONE_DAY) == "7 days"
    assert format_duration(ONE_HOUR) == "1 hour"
    assert format_duration(ONE_DAY + 2 * ONE_HOUR) == "1 day 2 hours"


def test_update_state_io_and_persistence(tmp_path):
    """Verify atomic state file writing, reading, and corrupted file handling."""
    state_file = tmp_path / "update_state.json"

    # 1. Non-existent file returns clean structure
    empty_state = load_update_state(state_file=state_file)
    assert empty_state.get("last_updates") == {}

    # 2. Record update
    t0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
    record_successful_update("harness:hermes", timestamp=t0, state_file=state_file)

    loaded_t0 = get_last_update("harness:hermes", state_file=state_file)
    assert loaded_t0 == t0

    # 3. Corrupted file recovery
    state_file.write_text("{ broken json content ...", encoding="utf-8")
    recovered_state = load_update_state(state_file=state_file)
    assert recovered_state.get("last_updates") == {}
    assert (tmp_path / "update_state.json.corrupted").exists()


def test_is_update_due(tmp_path):
    """Verify update due logic across elapsed times and frequencies."""
    state_file = tmp_path / "update_state.json"

    # 1. Never updated previously -> is_due = True
    due, elapsed, interval = is_update_due("harness:pi", frequency="weekly", state_file=state_file)
    assert due is True
    assert elapsed is None
    assert interval == ONE_WEEK

    # 2. Frequency 'always' -> is_due = True
    due, _, interval = is_update_due("harness:pi", frequency="always", state_file=state_file)
    assert due is True
    assert interval == 0.0

    # 3. Frequency 'never' -> is_due = False
    due, _, interval = is_update_due("harness:pi", frequency="never", state_file=state_file)
    assert due is False
    assert interval == INFINITY

    # 4. Updated 2 days ago with weekly frequency (not due)
    now = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
    two_days_ago = now - timedelta(days=2)
    record_successful_update("harness:pi", timestamp=two_days_ago, state_file=state_file)

    due, elapsed, interval = is_update_due("harness:pi", frequency="weekly", state_file=state_file, now=now)
    assert due is False
    assert abs(elapsed - 2 * ONE_DAY) < 1.0

    # 5. Updated 8 days ago with weekly frequency (due)
    eight_days_ago = now - timedelta(days=8)
    record_successful_update("harness:pi", timestamp=eight_days_ago, state_file=state_file)

    due, elapsed, interval = is_update_due("harness:pi", frequency="weekly", state_file=state_file, now=now)
    assert due is True
    assert abs(elapsed - 8 * ONE_DAY) < 1.0

    # 6. Frequency in hours: value of 1 (>= 1 hour ago)
    thirty_mins_ago = now - timedelta(minutes=30)
    record_successful_update("harness:orca", timestamp=thirty_mins_ago, state_file=state_file)
    due, elapsed, interval = is_update_due("harness:orca", frequency=1, state_file=state_file, now=now)
    assert due is False
    assert interval == ONE_HOUR

    one_hr_ago = now - timedelta(hours=1)
    record_successful_update("harness:orca", timestamp=one_hr_ago, state_file=state_file)
    due, elapsed, interval = is_update_due("harness:orca", frequency=1, state_file=state_file, now=now)
    assert due is True

    # 7. Frequency in hours: value of 24 (>= 24 hours ago)
    twenty_three_hrs_ago = now - timedelta(hours=23)
    record_successful_update("harness:t3", timestamp=twenty_three_hrs_ago, state_file=state_file)
    due, elapsed, interval = is_update_due("harness:t3", frequency=24, state_file=state_file, now=now)
    assert due is False
    assert interval == ONE_DAY

    twenty_four_hrs_ago = now - timedelta(hours=24)
    record_successful_update("harness:t3", timestamp=twenty_four_hrs_ago, state_file=state_file)
    due, elapsed, interval = is_update_due("harness:t3", frequency=24, state_file=state_file, now=now)
    assert due is True

    # 8. Frequency 0: due regardless of when last update time was
    one_min_ago = now - timedelta(minutes=1)
    record_successful_update("harness:zero", timestamp=one_min_ago, state_file=state_file)
    due, elapsed, interval = is_update_due("harness:zero", frequency=0, state_file=state_file, now=now)
    assert due is True
    assert interval == 0.0

    # 9. Frequency -1: never due
    hundred_days_ago = now - timedelta(days=100)
    record_successful_update("harness:neg", timestamp=hundred_days_ago, state_file=state_file)
    due, elapsed, interval = is_update_due("harness:neg", frequency=-1, state_file=state_file, now=now)
    assert due is False
    assert interval == INFINITY


def test_should_update_component_behaviors(tmp_path):
    """Verify should_update_component respecting behavior, frequency, and force flags."""
    state_file = tmp_path / "update_state.json"
    now = datetime.now(timezone.utc)
    one_day_ago = now - timedelta(days=1)
    record_successful_update("harness:hermes", timestamp=one_day_ago, state_file=state_file)

    # 1. Behavior 'never_update' -> False regardless of frequency
    assert should_update_component(
        "hermes", category="harness", behavior="never_update", frequency="daily",
        state_file=state_file
    ) is False

    # 2. Behavior 'always_update', but updated 1 day ago and frequency is 'weekly' -> False
    assert should_update_component(
        "hermes", category="harness", behavior="always_update", frequency="weekly",
        state_file=state_file
    ) is False

    # 3. Force flag overrides frequency gating
    assert should_update_component(
        "hermes", category="harness", behavior="always_update", frequency="weekly",
        force=True, state_file=state_file
    ) is True

    # 4. Behavior 'always_update' with frequency 'daily' (elapsed >= 1 day) -> True
    two_days_ago = now - timedelta(days=2)
    record_successful_update("harness:hermes", timestamp=two_days_ago, state_file=state_file)
    assert should_update_component(
        "hermes", category="harness", behavior="always_update", frequency="daily",
        state_file=state_file
    ) is True

    # 5. Behavior 'existing' skips prompt when not due
    with patch("local_ai_launcher.config.prompt_choice") as mock_prompt:
        record_successful_update("harness:hermes", timestamp=one_day_ago, state_file=state_file)
        res = should_update_component(
            "hermes", category="harness", behavior="existing", frequency="weekly",
            prompt_if_existing=True, state_file=state_file
        )
        assert res is False
        mock_prompt.assert_not_called()

    # 6. Behavior 'existing' prompts when interval has elapsed
    with patch("local_ai_launcher.config.prompt_choice", return_value=True) as mock_prompt:
        record_successful_update("harness:hermes", timestamp=two_days_ago, state_file=state_file)
        res = should_update_component(
            "hermes", category="harness", behavior="existing", frequency="daily",
            prompt_if_existing=True, state_file=state_file
        )
        assert res is True
        mock_prompt.assert_called_once()


def test_harness_adapter_frequency_integration(tmp_path):
    """Verify HarnessAdapter._should_update and record_update integration."""
    state_file = tmp_path / "update_state.json"

    class ConcreteHarness(HarnessAdapter):
        def check(self): return True
        def start(self, url, model=None): pass

    with patch("local_ai_launcher.update_frequency.get_update_state_file", return_value=state_file):
        adapter = ConcreteHarness("test_harness", tmp_path, [])
        adapter.update_behavior = "always_update"
        adapter.update_frequency_hrs = 168

        # First run: no prior update -> True
        assert adapter._should_update() is True

        # Record successful update
        adapter.record_update()

        # Immediate second run -> False (weekly interval has not elapsed)
        assert adapter._should_update() is False

        # Set frequency to 0 (always) -> True
        adapter.update_frequency_hrs = 0
        assert adapter._should_update() is True


def test_config_toml_frequency_overrides(tmp_path):
    """Verify loading update frequency settings from TOML configuration with backward compatibility."""
    toml_path = tmp_path / "launcher.toml"
    toml_path.write_text(
        """
        update_frequency = "monthly"
        force_update = true

        [harness_update_frequency]
        pi = "daily"
        hermes = "weekly"

        [engine_update_frequency]
        localai = "daily"
        llamacpp = "monthly"
        """,
        encoding="utf-8"
    )

    cfg = LauncherConfig(config_file=toml_path)
    assert getattr(cfg, "update_frequency", None) is None
    assert cfg.update_frequency_hrs == 720
    assert cfg.force_update is True
    assert cfg.harness_update_frequency_hrs.get("pi") == 24
    assert cfg.harness_update_frequency_hrs.get("hermes") == 168
    assert cfg.engine_update_frequency_hrs.get("localai") == 24
    assert cfg.engine_update_frequency_hrs.get("llamacpp") == 720


def test_config_toml_frequency_hrs_native(tmp_path):
    """Verify loading native update_frequency_hrs numeric settings from TOML configuration."""
    toml_path = tmp_path / "launcher_hrs.toml"
    toml_path.write_text(
        """
        update_frequency_hrs = 24
        force_update = true

        [harness_update_frequency_hrs]
        pi = 1
        hermes = 168

        [engine_update_frequency_hrs]
        localai = 0
        llamacpp = 720
        """,
        encoding="utf-8"
    )

    cfg = LauncherConfig(config_file=toml_path)
    assert cfg.update_frequency_hrs == 24
    assert getattr(cfg, "update_frequency", None) is None
    assert cfg.force_update is True
    assert cfg.harness_update_frequency_hrs.get("pi") == 1
    assert cfg.harness_update_frequency_hrs.get("hermes") == 168
    assert cfg.engine_update_frequency_hrs.get("localai") == 0
    assert cfg.engine_update_frequency_hrs.get("llamacpp") == 720


def test_config_frequency_hrs_normalization():
    """Verify normalization and sanitization of update_frequency_hrs and component dicts."""
    cfg = LauncherConfig()
    cfg.update_frequency_hrs = 24
    assert cfg.update_frequency_hrs == 24

    cfg.update_frequency_hrs = "hourly"
    assert cfg.update_frequency_hrs == 1

    cfg.update_frequency_hrs = 0
    assert cfg.update_frequency_hrs == 0

    cfg.update_frequency_hrs = -1
    assert cfg.update_frequency_hrs == INFINITY

    cfg.harness_update_frequency_hrs = {"pi": 1, "hermes": "daily"}
    assert cfg.harness_update_frequency_hrs.get("pi") == 1
    assert cfg.harness_update_frequency_hrs.get("hermes") == 24

    cfg.engine_update_frequency_hrs = {"localai": "always", "llamacpp": "monthly"}
    assert cfg.engine_update_frequency_hrs.get("localai") == 0
    assert cfg.engine_update_frequency_hrs.get("llamacpp") == 720
