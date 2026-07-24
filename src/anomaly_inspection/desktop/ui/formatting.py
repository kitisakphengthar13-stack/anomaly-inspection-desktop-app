from __future__ import annotations

from enum import Enum
from typing import Any


EMPTY_VALUE = ""


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_optional_int(value: Any) -> int | None:
    parsed = parse_optional_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def format_score(value: Any) -> str:
    parsed = parse_optional_float(value)
    if parsed is None:
        return EMPTY_VALUE
    abs_value = abs(parsed)
    if parsed != 0 and (abs_value < 0.001 or abs_value >= 100_000):
        return f"{parsed:.3g}"
    if abs_value < 1:
        return _trim_float(f"{parsed:.3f}")
    if abs_value < 100:
        return _trim_float(f"{parsed:.2f}")
    return _trim_float(f"{parsed:.1f}")


def format_threshold(value: Any) -> str:
    return format_score(value)


def format_ratio_percent(value: Any) -> str:
    parsed = parse_optional_float(value)
    if parsed is None:
        return EMPTY_VALUE
    percent = parsed * 100.0
    if 0 < abs(percent) < 0.1:
        return "<0.1%" if percent > 0 else ">-0.1%"
    return f"{percent:.1f}%"


def format_duration_ms(value: Any) -> str:
    parsed = parse_optional_float(value)
    if parsed is None:
        return EMPTY_VALUE
    sign = "-" if parsed < 0 else ""
    ms = abs(parsed)
    if ms < 1_000:
        return f"{sign}{ms:.0f} ms"
    seconds = ms / 1_000.0
    if seconds < 10:
        return f"{sign}{seconds:.2f} s"
    if seconds < 60:
        return f"{sign}{seconds:.1f} s"
    minutes = int(seconds // 60)
    remaining_seconds = int(round(seconds % 60))
    if remaining_seconds == 60:
        minutes += 1
        remaining_seconds = 0
    return f"{sign}{minutes} min {remaining_seconds} s"


def format_pixel_count(value: Any) -> str:
    parsed = parse_optional_int(value)
    if parsed is None:
        return EMPTY_VALUE
    return f"{parsed:,} px"


def format_pixel_area(value: Any) -> str:
    parsed = parse_optional_int(value)
    if parsed is None:
        return EMPTY_VALUE
    return f"{parsed:,} px^2"


def format_mean_pixel_difference(value: Any) -> str:
    parsed = parse_optional_float(value)
    if parsed is None:
        return EMPTY_VALUE
    return f"{parsed:.1f}"


def format_presence_status(value: Any) -> str:
    raw = _raw_value(value).upper()
    labels = {
        "PART_PRESENT": "Part present",
        "NO_PART": "No part",
        "ERROR": "Error",
    }
    return labels.get(raw, _title_from_token(raw))


def format_anomaly_decision(value: Any) -> str:
    raw = _raw_value(value).strip()
    if raw == "":
        return EMPTY_VALUE
    normalized = raw.lower()
    if normalized == "true":
        return "Anomaly detected"
    if normalized == "false":
        return "No anomaly"
    return _title_from_token(raw)


def format_bool(value: Any) -> str:
    raw = _raw_value(value).strip().lower()
    if raw == "":
        return EMPTY_VALUE
    if raw in {"true", "1", "yes"}:
        return "Yes"
    if raw in {"false", "0", "no"}:
        return "No"
    return _title_from_token(raw)


def format_final_result(value: Any) -> str:
    raw = _raw_value(value).upper()
    if raw == "NO_PART":
        return "NO PART"
    return raw


def format_log_value(key: str, value: Any) -> str:
    if key == "presence_status":
        return format_presence_status(value)
    if key == "foreground_ratio":
        return format_ratio_percent(value)
    if key == "mean_diff":
        return format_mean_pixel_difference(value)
    if key == "largest_blob_area":
        return format_pixel_area(value)
    if key in {"changed_pixel_count", "zone_pixel_count"}:
        return format_pixel_count(value)
    if key == "anomaly_pred_label":
        return format_anomaly_decision(value)
    if key == "anomaly_score":
        return format_score(value)
    if key == "fallback_anomaly_threshold":
        return format_threshold(value)
    if key in {"presence_time_ms", "anomaly_time_ms", "total_time_ms"}:
        return format_duration_ms(value)
    if key == "anomaly_ran":
        return format_bool(value)
    if key == "final_result":
        return format_final_result(value)
    return _raw_value(value)


def _raw_value(value: Any) -> str:
    if value is None:
        return EMPTY_VALUE
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _title_from_token(value: str) -> str:
    token = value.strip()
    if not token:
        return EMPTY_VALUE
    return token.replace("_", " ").replace("-", " ").title()


def _trim_float(text: str) -> str:
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")
