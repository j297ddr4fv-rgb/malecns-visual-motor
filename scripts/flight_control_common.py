"""Small shared primitives for the hand-built flight-control adapters."""

from collections.abc import Mapping
import math
from typing import Any


def normal_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def clip(value: float, low: float, high: float) -> float:
    return min(float(high), max(float(low), float(value)))


def number(value: Any) -> tuple[float, bool]:
    if isinstance(value, bool):
        return float(value), False
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0, False
    return result, math.isfinite(result)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return repr(value)
