"""Deterministic operator library for chain-of-options.

Every operator is a pure function over parsed values (datetimes, durations,
zones, numbers, series). Chains reference them by name; nothing here reads
the state text except `table_series`, which is the one table-replay primitive
(it extracts a numeric column from line-oriented tables so a chain can sum or
scan it). Adding an operator = adding a function to OPS.
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ----------------------------------------------------------------- helpers --

def _tz(zone: Optional[Dict[str, Any]]):
    if not zone:
        return None
    if "iana" in zone and ZoneInfo is not None:
        return ZoneInfo(zone["iana"])
    if "offset_h" in zone:
        return timezone(timedelta(hours=zone["offset_h"]))
    return None


def _aware(dt: datetime, zone: Optional[Dict[str, Any]]) -> datetime:
    tz = _tz(zone)
    if dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=tz) if tz else dt


def _dt(x: Any) -> datetime:
    """Accept a Span, a dict {value, meta}, or a datetime."""
    if isinstance(x, datetime):
        return x
    if hasattr(x, "value") and isinstance(getattr(x, "value"), datetime):
        return _aware(x.value, (x.meta or {}).get("zone"))
    raise TypeError(f"not a datetime: {x!r}")


def _dur(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict) and "unit" in x:
        return x
    if hasattr(x, "value") and isinstance(x.value, dict):
        return x.value
    raise TypeError(f"not a duration: {x!r}")


def _num(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "value") and isinstance(x.value, (int, float)):
        return float(x.value)
    raise TypeError(f"not a number: {x!r}")


def _zone(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if hasattr(x, "value") and isinstance(x.value, dict):
        return x.value
    raise TypeError(f"not a zone: {x!r}")


# --------------------------------------------------------------- operators --

def add_duration(start: Any, duration: Any) -> datetime:
    """start + duration. Months/years clamp to the last day of the target month."""
    dt, d = _dt(start), _dur(duration)
    n, unit = d["n"], d["unit"]
    if unit in ("months", "years"):
        months = int(round(n * (12 if unit == "years" else 1)))
        y, m = divmod(dt.month - 1 + months, 12)
        y, m = dt.year + y, m + 1
        last = calendar.monthrange(y, m)[1]
        return dt.replace(year=y, month=m, day=min(dt.day, last))
    if unit == "business_days":
        cur, left = dt, int(n)
        while left > 0:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                left -= 1
        return cur
    if unit in ("hours", "minutes") and dt.tzinfo is not None:
        # Elapsed time is real time: add in UTC so a DST change inside the
        # window shifts the wall clock instead of being silently skipped.
        return (dt.astimezone(timezone.utc) + timedelta(**{unit: n})).astimezone(dt.tzinfo)
    return dt + timedelta(**{unit: n})


def end_of_day(x: Any) -> datetime:
    dt = _dt(x)
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def in_zone(x: Any, zone: Any) -> datetime:
    """Reinterpret a naive wall-clock time as being in `zone`, or convert an aware one."""
    dt, z = _dt(x), _zone(zone)
    tz = _tz(z)
    if tz is None:
        return dt
    return dt.astimezone(tz) if dt.tzinfo is not None else dt.replace(tzinfo=tz)


def elapsed_hours(a: Any, b: Any) -> float:
    """b - a in real elapsed hours (aware datetimes compare in UTC)."""
    da, db = _dt(a), _dt(b)
    if (da.tzinfo is None) != (db.tzinfo is None):
        da, db = da.replace(tzinfo=None), db.replace(tzinfo=None)
    return (db - da).total_seconds() / 3600.0


def elapsed_hours_abs(a: Any, b: Any) -> float:
    """|b - a| in hours: for 'time between two moments' when order is not the question."""
    return abs(elapsed_hours(a, b))


def elapsed_days(a: Any, b: Any) -> float:
    return elapsed_hours(a, b) / 24.0


def days_inclusive(a: Any, b: Any) -> int:
    da, db = _dt(a).date(), _dt(b).date()
    return (db - da).days + 1


def is_leap_year(x: Any) -> bool:
    y = _dt(x).year if not isinstance(x, (int, float)) else int(x)
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def weekday_name(x: Any) -> str:
    return _dt(x).strftime("%A")


def days_in_year(x: Any) -> int:
    return 366 if calendar.isleap(_dt(x).year) else 365


def prorate(amount: Any, part: Any, whole: Any) -> float:
    return _num(amount) * _num(part) / _num(whole)


def percent_of(part: Any, whole: Any) -> float:
    return 100.0 * _num(part) / _num(whole)


def mul(a: Any, b: Any) -> float:
    return _num(a) * _num(b)


def div(a: Any, b: Any) -> float:
    return _num(a) / _num(b)


def add(a: Any, b: Any) -> float:
    return _num(a) + _num(b)


def sub(a: Any, b: Any) -> float:
    return _num(a) - _num(b)


def round_up(x: Any) -> float:
    return float(math.ceil(_num(x) - 1e-9))


def round_to(x: Any, places: Any = 2) -> float:
    return round(_num(x), int(_num(places)))


def lb_to_kg(x: Any) -> float:
    return _num(x) * 0.45359237


def gb_to_bytes(x: Any) -> float:
    return _num(x) * 1_000_000_000


def tib_to_bytes(x: Any) -> float:
    return _num(x) * 1_099_511_627_776


def pct_of(limit: Any, pct: Any = None) -> float:
    """limit * pct/100; a missing pct means the whole limit."""
    return _num(limit) * (_num(pct) / 100.0 if pct is not None else 1.0)


def compare(a: Any, b: Any) -> int:
    x, y = _num(a), _num(b)
    return 0 if abs(x - y) < 1e-9 else (1 if x > y else -1)


_ROW = re.compile(r"^\s*(?P<label>[^|:]{1,40}?)\s*[|:]\s*(?P<rest>.+)$")
_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def table_series(state: str, column: str = "") -> List[Tuple[str, float]]:
    """Extract (row_label, value) pairs from line-oriented tables.

    Rows look like `Sep 01 | usage 402.10 | credits 0.00 | tax 32.17` or
    `S1 Wed 21 Oct 13:00 – 21:00`. `column` is a keyword: the value is the
    first number that follows it on the row (or the first number if empty).
    """
    out: List[Tuple[str, float]] = []
    for line in state.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        rest = m.group("rest")
        if column:
            i = rest.lower().find(column.lower())
            if i < 0:
                continue
            rest = rest[i + len(column):]
        n = _NUM.search(rest)
        if not n:
            continue
        out.append((m.group("label").strip(), float(n.group(0).replace(",", ""))))
    return out


def series_sub(a: List[Tuple[str, float]], b: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    bm = dict(b)
    return [(k, v - bm.get(k, 0.0)) for k, v in a]


def cumsum(series: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    out, acc = [], 0.0
    for k, v in series:
        acc += v
        out.append((k, acc))
    return out


def series_sum(series: List[Tuple[str, float]]) -> float:
    return float(sum(v for _, v in series))


def first_at_or_above(series: List[Tuple[str, float]], threshold: Any) -> Optional[str]:
    t = _num(threshold)
    for k, v in series:
        if v >= t - 1e-9:
            return k
    return None


def count_below(values: List[float], threshold: Any) -> int:
    t = _num(threshold)
    return sum(1 for v in values if v < t - 1e-9)


OPS: Dict[str, Callable[..., Any]] = {
    "add_duration": add_duration, "end_of_day": end_of_day, "in_zone": in_zone,
    "elapsed_hours": elapsed_hours, "elapsed_hours_abs": elapsed_hours_abs, "elapsed_days": elapsed_days, "days_inclusive": days_inclusive,
    "days_in_year": days_in_year, "is_leap_year": is_leap_year, "weekday_name": weekday_name, "prorate": prorate, "percent_of": percent_of,
    "mul": mul, "div": div, "add": add, "sub": sub, "round_up": round_up, "round_to": round_to,
    "pct_of": pct_of, "lb_to_kg": lb_to_kg, "gb_to_bytes": gb_to_bytes, "tib_to_bytes": tib_to_bytes, "compare": compare,
    "table_series": table_series, "series_sub": series_sub, "cumsum": cumsum, "series_sum": series_sum,
    "first_at_or_above": first_at_or_above, "count_below": count_below,
}
