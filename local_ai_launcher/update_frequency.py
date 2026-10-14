"""
Update Frequency Management for Local AI Launcher.

Manages update schedules and maximum update frequencies across harnesses, engines,
and workspace environments (e.g. daily, weekly, monthly, custom duration, always, never).
Tracks last successful update timestamps in config/update_state.json to prevent
unwanted repetitive updates and git pulls on every launch.
"""

import os
import sys
import json
import re
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from local_ai_launcher.paths import CONFIG_DIR, AI_WORKSPACE_DIR

logger = logging.getLogger("update_frequency")

# Standard duration constants in seconds
ONE_MINUTE = 60.0
ONE_HOUR = 3600.0
ONE_DAY = 86400.0
ONE_WEEK = 7.0 * ONE_DAY
ONE_BIWEEK = 14.0 * ONE_DAY
ONE_MONTH = 30.0 * ONE_DAY
INFINITY = float("inf")

NAMED_FREQUENCIES_HOURS: Dict[str, float] = {
    "always": 0.0,
    "every": 0.0,
    "all": 0.0,
    "hourly": 1.0,
    "daily": 24.0,
    "weekly": 168.0,
    "biweekly": 336.0,
    "monthly": 720.0,
    "never": INFINITY,
    "none": INFINITY,
    "off": INFINITY,
    "disable": INFINITY,
    "disabled": INFINITY,
}

NAMED_FREQUENCIES: Dict[str, float] = {
    k: (v * ONE_HOUR if v != INFINITY else INFINITY)
    for k, v in NAMED_FREQUENCIES_HOURS.items()
}


def parse_frequency_hours(freq: Union[str, int, float, None], default: Union[str, int, float] = 168.0) -> float:
    """
    Convert a frequency representation into float hours.

    Supported formats:
      - Named keywords: 'always' (0h), 'hourly' (1h), 'daily' (24h),
        'weekly' (168h), 'biweekly' (336h), 'monthly' (720h), 'never' (infinity).
      - Unit strings: '12h', '12 hours', '3d', '3 days', '2w', '30m', '60s'.
      - Numeric (int/float): treated as hours:
        * 0 -> 0.0 (always update, regardless of when last update time was)
        * 1 -> 1.0 (update if last update was >= 1 hour ago)
        * 24 -> 24.0 (update if last update was >= 24 hours ago)
        * 168 -> 168.0 (weekly)
        * < 0 -> INFINITY (never update)
      - None or empty string: evaluates `default`.
    """
    if freq is None or (isinstance(freq, str) and not freq.strip()):
        if default is not None and default != freq:
            return parse_frequency_hours(default, default=168.0)
        return 168.0

    if isinstance(freq, bool):
        return 0.0 if freq else INFINITY

    if isinstance(freq, (int, float)):
        if freq < 0:
            return INFINITY
        if freq == 0:
            return 0.0
        return float(freq)

    val_str = str(freq).strip().lower()

    if val_str in NAMED_FREQUENCIES_HOURS:
        return NAMED_FREQUENCIES_HOURS[val_str]

    # Match numeric values with optional time unit suffixes
    pattern = r"^(\d+(?:\.\d+)?)\s*([a-z]+)?$"
    match = re.match(pattern, val_str)
    if not match:
        logger.warning(f"Unrecognized update frequency format '{freq}'. Falling back to default '{default}'.")
        return parse_frequency_hours(default, default=168.0)

    num_part = float(match.group(1))
    unit_part = match.group(2) or "h"  # Default unit is hours if plain number string is provided

    if num_part < 0:
        return INFINITY
    if num_part == 0:
        return 0.0

    if unit_part in ("h", "hr", "hrs", "hour", "hours"):
        return num_part
    elif unit_part in ("d", "day", "days"):
        return num_part * 24.0
    elif unit_part in ("w", "wk", "wks", "week", "weeks"):
        return num_part * 168.0
    elif unit_part in ("mo", "mon", "mons", "month", "months"):
        return num_part * 720.0
    elif unit_part in ("m", "min", "mins", "minute", "minutes"):
        return num_part / 60.0
    elif unit_part in ("s", "sec", "secs", "second", "seconds"):
        return num_part / 3600.0
    elif unit_part in ("y", "yr", "yrs", "year", "years"):
        return num_part * 8760.0
    else:
        logger.warning(f"Unknown frequency unit '{unit_part}' in '{freq}'. Interpreting as hours.")
        return num_part


def parse_frequency_seconds(freq: Union[str, int, float, None], default: Union[str, int, float] = 168.0) -> float:
    """
    Convert a frequency representation into float seconds.
    Delegates to parse_frequency_hours.
    """
    hrs = parse_frequency_hours(freq, default=default)
    if hrs == INFINITY:
        return INFINITY
    if hrs <= 0.0:
        return 0.0
    return hrs * ONE_HOUR


def format_frequency_hrs(hours: float) -> str:
    """Format frequency in hours to clean human-readable text."""
    if hours == INFINITY:
        return "never"
    if hours <= 0.0:
        return "always (0h)"
    if hours == 1.0:
        return "1 hour"
    if hours == 24.0:
        return "24 hours (1 day)"
    if hours == 168.0:
        return "168 hours (1 week)"
    if hours % 24.0 == 0:
        days = int(hours // 24.0)
        return f"{int(hours)} hours ({days} days)"
    return f"{int(hours) if hours == int(hours) else hours} hours"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to clean human-readable text."""
    if seconds == INFINITY:
        return "never"
    if seconds <= 0.0:
        return "always"

    total_secs = int(seconds)
    days = total_secs // 86400
    hours = (total_secs % 86400) // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60

    parts = []
    if days > 0:
        parts.append(f"{days} day" if days == 1 else f"{days} days")
    if hours > 0 and days < 14:
        parts.append(f"{hours} hour" if hours == 1 else f"{hours} hours")
    if minutes > 0 and days == 0 and hours < 3:
        parts.append(f"{minutes} min" if minutes == 1 else f"{minutes} mins")
    if not parts:
        parts.append(f"{secs} sec" if secs == 1 else f"{secs} secs")

    return " ".join(parts)


def get_update_state_file() -> Path:
    """Return the absolute path to update_state.json in CONFIG_DIR or test override."""
    override = os.environ.get("LOCALAI_UPDATE_STATE_FILE")
    if override:
        return Path(override)
    return CONFIG_DIR / "update_state.json"


def load_update_state(state_file: Optional[Path] = None) -> Dict[str, Any]:
    """Safely load the update state file, returning an empty dict on missing/corrupted file."""
    path = state_file or get_update_state_file()
    if not path.is_file():
        return {"version": 1, "last_updates": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "last_updates" not in data or not isinstance(data.get("last_updates"), dict):
                    data["last_updates"] = {}
                return data
    except Exception as e:
        logger.warning(f"Failed to read update state from {path}: {e}. Initializing fresh state.")
        try:
            backup_path = path.with_suffix(".json.corrupted")
            path.replace(backup_path)
        except Exception:
            pass

    return {"version": 1, "last_updates": {}}


def save_update_state(state: Dict[str, Any], state_file: Optional[Path] = None) -> None:
    """Atomically write update state to disk with flush and fsync."""
    path = state_file or get_update_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (AttributeError, OSError):
                pass
        tmp_path.replace(path)
    except Exception as e:
        logger.error(f"Failed to atomically persist update state to {path}: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def get_last_update(component_key: str, state_file: Optional[Path] = None) -> Optional[datetime]:
    """Retrieve last successful update timestamp for a component in UTC."""
    state = load_update_state(state_file=state_file)
    ts_str = state.get("last_updates", {}).get(component_key)
    if not ts_str:
        return None

    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid timestamp '{ts_str}' for component '{component_key}': {e}")
        return None


def record_successful_update(
    component_key: str,
    timestamp: Optional[datetime] = None,
    state_file: Optional[Path] = None
) -> None:
    """Record a successful update timestamp for a given component."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    elif timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    state = load_update_state(state_file=state_file)
    state.setdefault("last_updates", {})[component_key] = timestamp.isoformat()
    save_update_state(state, state_file=state_file)
    logger.debug(f"Recorded successful update for '{component_key}' at {timestamp.isoformat()}")


def is_update_due(
    component_key: str,
    frequency: Union[str, int, float, None] = None,
    state_file: Optional[Path] = None,
    now: Optional[datetime] = None
) -> Tuple[bool, Optional[float], float]:
    """
    Check if an update is due based on elapsed time and frequency.

    Returns:
      (is_due: bool, elapsed_seconds: Optional[float], interval_seconds: float)
    """
    interval = parse_frequency_seconds(frequency)
    if interval == INFINITY:
        return False, None, INFINITY
    if interval == 0.0:
        return True, 0.0, 0.0

    last_dt = get_last_update(component_key, state_file=state_file)
    if last_dt is None:
        # Never updated previously; update is due immediately
        return True, None, interval

    curr_now = now or datetime.now(timezone.utc)
    if curr_now.tzinfo is None:
        curr_now = curr_now.replace(tzinfo=timezone.utc)

    elapsed = (curr_now - last_dt).total_seconds()
    if elapsed < 0:
        # Clock skew / future timestamp: treat as due to reset state
        return True, 0.0, interval

    is_due = elapsed >= interval
    return is_due, elapsed, interval


def should_update_component(
    component_name: str,
    category: str = "harness",
    behavior: Optional[str] = None,
    frequency: Optional[str] = None,
    force: bool = False,
    prompt_if_existing: bool = True,
    state_file: Optional[Path] = None
) -> bool:
    """
    Evaluate whether a component should be updated.

    Takes into account:
      1. behavior ('always_update', 'never_update', 'existing')
      2. frequency ('always', 'daily', 'weekly', 'monthly', 'never', '3d', etc.)
      3. elapsed time since last successful update in update_state.json
      4. force flag (--force-update CLI override)
      5. prompt for 'existing' if interval elapsed
    """
    from local_ai_launcher.config import config, prompt_choice

    # Normalize component key
    key = component_name if ":" in component_name else f"{category}:{component_name}"

    # 1. Resolve behavior
    if behavior is None:
        if category == "harness":
            behavior = config.harness_update_behaviors.get(component_name, "never_update")
        elif category == "engine":
            behavior = getattr(config, f"{component_name}_update_behavior", "never_update")
        else:
            behavior = "never_update"

    # 2. Check never_update (unless forced)
    if behavior == "never_update" and not force:
        logger.info(f"  -> Skipping {component_name} update as per configuration ({behavior}).")
        return False

    # 3. Resolve frequency if not provided
    if frequency is None:
        if category == "harness":
            frequency = getattr(config, "harness_update_frequency_hrs", {}).get(
                component_name,
                getattr(config, "update_frequency_hrs", 168.0)
            )
        elif category == "engine":
            frequency = getattr(config, "engine_update_frequency_hrs", {}).get(
                component_name,
                getattr(config, "update_frequency_hrs", 168.0)
            )
        else:
            frequency = getattr(config, "update_frequency_hrs", 168.0)

    # 4. Handle force flag
    if force:
        logger.info(f"  -> Forcing update for {component_name} (force update requested).")
        return True

    # 5. Check if update is due by frequency
    is_due, elapsed, interval = is_update_due(key, frequency=frequency, state_file=state_file)

    if interval == INFINITY:
        logger.info(f"  -> Skipping {component_name} update as update frequency is set to 'never'.")
        return False

    if not is_due:
        elapsed_str = format_duration(elapsed) if elapsed is not None else "unknown"
        interval_str = format_duration(interval)
        logger.info(f"  -> [{component_name}] Last updated {elapsed_str} ago (frequency: {interval_str}). Skipping update.")
        return False

    # Update is due! Check behavior mode
    interval_str = format_duration(interval)
    if elapsed is not None:
        elapsed_str = format_duration(elapsed)
        logger.info(f"  -> [{component_name}] Update interval of {interval_str} has elapsed (last update: {elapsed_str} ago).")
    else:
        logger.info(f"  -> [{component_name}] No prior update recorded. Update interval ({interval_str}) is active.")

    if behavior == "always_update":
        return True
    elif behavior == "existing":
        if prompt_if_existing:
            return prompt_choice(f"Do you want to update {component_name}? (y/n): ", default="n")
        return False
    else:
        return True
