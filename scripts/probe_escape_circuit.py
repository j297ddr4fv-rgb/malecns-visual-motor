#!/usr/bin/env python3
"""Read-only offline probe for the MaleCNS looming/escape circuit.

The probe consumes an already recorded MaleCNS output.  It does not construct
an engine, import Torch, create a CUDA context, or turn a measurement into a
motor command.  The same reader accepts the compact ``telemetry.jsonl`` rows
written by :mod:`run_digital_fly`, the ``summary.json`` tick tables produced by
``lif_flyb.py``, sparse ``spikes.npz`` traces, and in-memory
``NeuralOutput``-like objects.

The primary public functions are :func:`load_trace`, :func:`analyze_trace`,
and :func:`compare_traces`.  A small command line wrapper writes the resulting
JSON report, for example::

    python3 scripts/probe_escape_circuit.py \
        --trace sham=artifacts/sham/summary.json \
        --trace loom=artifacts/loom-escape/summary.json \
        --trace palm=artifacts/swat-escape/summary.json \
        --collision-ms 100 --out artifacts/escape-circuit-report.json

``first_spike_ms`` and ``peak_rate_hz`` are descriptive measurements only.
``relative_collision_ms`` is positive when an event occurred before the
provided collision time.  No metric in this module is an action gate.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = "escape-circuit-probe-1"
DEFAULT_CSR = Path("/data/cache/malecns-gpu/csr-flyb")
DEFAULT_POPULATIONS: tuple[str, ...] = (
    "LC4",
    "LPLC2",
    "DNp01",
    "DNp02",
    "DNp04",
    "DNp11",
    "DNp06",
    "TTMn",
    "DLM",
)

# These aliases are intentionally descriptive.  They do not encode a motor
# policy; they merely make the exact names used by the existing reports easier
# to query in an offline trace.
POPULATION_ALIASES: Mapping[str, tuple[str, ...]] = {
    "LC4": ("LC4",),
    "LPLC2": ("LPLC2",),
    "DNp01": ("DNp01",),
    "DNp02": ("DNp02",),
    "DNp04": ("DNp04",),
    "DNp11": ("DNp11",),
    "DNp06": ("DNp06",),
    "TTMn": ("TTMn",),
    # The CSR and the fly-brain-minecraft report use both DLM and DLMn.
    "DLM": ("DLM", "DLMn"),
}

_LABEL_ALIASES = {
    "baseline": "sham",
    "control": "sham",
    "sham": "sham",
    "loom": "loom",
    "looming": "loom",
    "palm": "palm",
    "swat": "palm",
}
_SIDE_RE = re.compile(r"(?:[/_:\-](?P<side>[LR]))$|(?:[ _](?P<long>left|right))$", re.I)
_RESERVED_RATE_KEYS = {
    "population_rates_hz",
    "population_rates",
    "rates_hz",
    "rates",
    "firing_rates_hz",
    "firing_rates",
    "rate_hz",
}
_RESERVED_COUNT_KEYS = {
    "population_spike_counts",
    "population_spikes",
    "spike_counts",
    "spikes_by_population",
}


class ProbeError(ValueError):
    """Raised for malformed or ambiguous offline trace input."""


def _finite(value: Any, name: str, *, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ProbeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProbeError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ProbeError(f"{name} must be finite")
    return number


def _optional_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProbeError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProbeError(f"{name} must be an integer") from exc
    if number < 0:
        raise ProbeError(f"{name} must be >= 0")
    return number


def _json_safe(value: Any) -> Any:
    """Convert NumPy/dataclass-adjacent values to bounded JSON values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return repr(value)


def canonical_label(label: Any) -> str:
    """Return a stable protocol label while retaining custom labels."""

    text = str(label).strip()
    if not text:
        return "trace"
    low = text.lower()
    if low in _LABEL_ALIASES:
        return _LABEL_ALIASES[low]
    # ``loom_fast`` and ``palm_gain_1.2`` are still useful as custom labels,
    # but their protocol class is available in metadata via this helper.
    for needle, normalized in (("sham", "sham"), ("control", "sham"),
                               ("loom", "loom"), ("palm", "palm"),
                               ("swat", "palm")):
        if needle in low:
            return normalized
    return text


def _infer_label(source: Any, metadata: Optional[Mapping[str, Any]] = None) -> str:
    if metadata:
        for key in ("label", "trace_label", "condition", "scenario", "stimulus",
                    "stimulus_type", "trace_type", "kind"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return canonical_label(value)
    if isinstance(source, (str, Path)):
        return canonical_label(Path(source).stem)
    return "trace"


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        try:
            value = value.item()
        except Exception:
            return {}
        return value if isinstance(value, Mapping) else {}
    if hasattr(value, "items") and callable(value.items):
        try:
            candidate = dict(value.items())
        except Exception:
            candidate = None
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _to_1d_float(value: Any, name: str, *, allow_nan: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProbeError(f"{name} must be numeric") from exc
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ProbeError(f"{name} must be one-dimensional")
    if not allow_nan and not np.all(np.isfinite(array)):
        raise ProbeError(f"{name} contains non-finite values")
    if np.any(array < 0):
        raise ProbeError(f"{name} must be non-negative")
    return array.astype(np.float64, copy=True)


def _coerce_series(value: Any, n: int, name: str, *, allow_nan: bool = True) -> np.ndarray:
    array = _to_1d_float(value, name, allow_nan=allow_nan)
    if array.size == 1 and n != 1:
        array = np.full(n, float(array[0]), dtype=np.float64)
    if array.size != n:
        raise ProbeError(f"{name} length {array.size} != time length {n}")
    return array


def _coerce_nonnegative_series(value: Any, n: int, name: str) -> np.ndarray:
    return _coerce_series(value, n, name, allow_nan=False)


def _optional_nonnegative_series(
    value: Any, n: int, name: str, *, maximum: Optional[float] = None,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    result = _coerce_nonnegative_series(value, n, name)
    if maximum is not None and np.any(result > maximum):
        raise ProbeError(f"{name} must be <= {maximum:g}")
    return result


def _optional_series(
    values: Sequence[Any], *, dtype: Any = np.float64, missing: Any = 0,
) -> Optional[np.ndarray]:
    return (
        np.asarray([missing if value is None else value for value in values], dtype=dtype)
        if any(value is not None for value in values) else None
    )


def _nested_values(value: Any, *, max_depth: int = 5) -> Iterator[Mapping[str, Any]]:
    """Yield mappings in a row without assuming one telemetry nesting shape."""

    if max_depth < 0:
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # Telemetry rows can carry large pixel/vector lists.  Only descend
        # into sequence containers that could contain mappings; a flat numeric
        # vector cannot contribute a named telemetry field.
        if value and not isinstance(value[0], (Mapping, Sequence)):
            return
        for child in value:
            if isinstance(child, Mapping) or isinstance(child, Sequence):
                yield from _nested_values(child, max_depth=max_depth - 1)
        return
    if not isinstance(value, Mapping):
        return
    yield value
    for child in value.values():
        if isinstance(child, Mapping):
            yield from _nested_values(child, max_depth=max_depth - 1)


def _first_nested(value: Any, keys: Sequence[str]) -> Any:
    wanted = {str(key).lower() for key in keys}
    for item in _nested_values(value):
        for key, candidate in item.items():
            if str(key).lower() in wanted and candidate is not None:
                return candidate
    return None


def _time_ms_from_mapping(row: Mapping[str, Any]) -> Optional[float]:
    # Direct fields are preferred so a nested summary does not accidentally
    # override a row's frame time.
    for key in ("time_ms", "t_ms", "frame_time_ms", "timestamp_ms", "time"):
        if key in row:
            value = row[key]
            if isinstance(value, Mapping):
                continue
            try:
                return _finite(value, f"{key}")
            except ProbeError:
                pass
    for key in ("time_s", "t_s", "frame_time_s", "timestamp_s"):
        if key in row and not isinstance(row[key], Mapping):
            try:
                value = _finite(row[key], key)
            except ProbeError:
                continue
            assert value is not None
            return value * 1000.0
    timestamp = row.get("timestamp")
    if isinstance(timestamp, Mapping):
        for key in ("t_s", "time_s"):
            if key in timestamp:
                value = _finite(timestamp[key], f"timestamp.{key}")
                assert value is not None
                return value * 1000.0
        for key in ("t_ms", "time_ms"):
            if key in timestamp:
                return _finite(timestamp[key], f"timestamp.{key}")
    # Timestamps can be nested under the neural contract.
    neural = row.get("neural")
    if isinstance(neural, Mapping):
        nested = _time_ms_from_mapping(neural)
        if nested is not None:
            return nested
    return None


def _extract_numeric_map(
    row: Mapping[str, Any], keys: set[str], field: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in _nested_values(row):
        for key, value in item.items():
            if str(key).lower() not in keys:
                continue
            for pop, number in _mapping(value).items():
                if isinstance(number, Mapping):
                    continue
                try:
                    number = _finite(number, f"{field}[{pop}]")
                except ProbeError:
                    continue
                assert number is not None
                if number >= 0:
                    result[str(pop)] = number
    return result


def _extract_rate_map(row: Mapping[str, Any]) -> dict[str, float]:
    result = _extract_numeric_map(row, _RESERVED_RATE_KEYS, "rate")
    # Structured maps have precedence over scalar aliases.
    # Compact engine telemetry also keeps scalar aliases next to the map.
    for item in _nested_values(row):
        for pop in DEFAULT_POPULATIONS:
            if pop in item and not isinstance(item[pop], Mapping):
                try:
                    number = _finite(item[pop], f"rate[{pop}]")
                except ProbeError:
                    continue
                assert number is not None
                if number >= 0:
                    result.setdefault(pop, number)
        for key, value in item.items():
            if str(key) in {"step_index", "spike_count", "active_neurons", "deliveries"}:
                continue
            # Preserve explicit side-labelled scalar aliases, e.g. DNp01/L.
            if _SIDE_RE.search(str(key)) and not isinstance(value, Mapping):
                try:
                    number = _finite(value, f"rate[{key}]")
                except ProbeError:
                    continue
                assert number is not None
                if number >= 0:
                    result.setdefault(str(key), number)
    return result


def _extract_count_map(row: Mapping[str, Any]) -> dict[str, float]:
    return _extract_numeric_map(row, _RESERVED_COUNT_KEYS, "spike_count")


def _extract_scalar(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    wanted = {str(key).lower() for key in keys}
    for item in _nested_values(row):
        for key, value in item.items():
            if str(key).lower() not in wanted or isinstance(value, Mapping):
                continue
            try:
                number = _finite(value, str(key))
            except ProbeError:
                continue
            if number is not None and number >= 0:
                return number
    return None


def _extract_bool(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[bool]:
    wanted = {str(key).lower() for key in keys}
    for item in _nested_values(row):
        for key, value in item.items():
            if str(key).lower() not in wanted:
                continue
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "yes", "on", "1", "saturated"}:
                    return True
                if normalized in {"false", "no", "off", "0", "none", "null"}:
                    return False
    return None


def _extract_neural_bool(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[bool]:
    """Read saturation flags only from neural/network containers.

    A full flight telemetry row also contains ``nm_out.saturated``.  That is
    a motor-readout clipping flag, not a global MaleCNS network saturation
    observation, so a broad recursive lookup would conflate the two.
    """

    wanted = {str(key).lower() for key in keys}
    containers = {"neural", "cns", "network", "neural_network", "named_outputs"}
    for key, value in row.items():
        if str(key).lower() in containers and isinstance(value, Mapping):
            candidate = _extract_bool(value, tuple(wanted))
            if candidate is not None:
                return candidate
    # Direct global fields are safe even when the row has other nested
    # ``saturated`` fields.
    for key in ("global_saturation", "network_saturated"):
        if key in row:
            candidate = _extract_bool(row, (key,))
            if candidate is not None:
                return candidate
    return None


def _extract_neuron_count(value: Any) -> Optional[int]:
    candidate = _first_nested(value, (
        "neuron_count", "neurons", "n_neurons", "n", "total_neurons",
    ))
    return _optional_int(candidate, "neuron_count") if candidate is not None else None


def _event_time_ms(
    value: Any, millisecond_aliases: Sequence[Sequence[str]],
    second_aliases: Sequence[str], name: str, *, ignore_bad_ms: bool = False,
) -> Optional[float]:
    candidate = None
    for aliases in millisecond_aliases:
        if candidate is None:
            candidate = _first_nested(value, aliases)
    if candidate is not None and not isinstance(candidate, Mapping):
        try:
            return _finite(candidate, name)
        except ProbeError:
            if not ignore_bad_ms:
                raise
    candidate = _first_nested(value, second_aliases)
    if candidate is None or isinstance(candidate, Mapping):
        return None
    seconds = _finite(candidate, f"{name.removesuffix('_ms')}_s")
    assert seconds is not None
    return seconds * 1000.0


def _extract_collision_ms(value: Any) -> Optional[float]:
    # Absolute-time names outrank the historical event/offset aliases.
    return _event_time_ms(
        value,
        (("collision_time_ms", "impact_time_ms", "collision_t_ms", "impact_t_ms"),
         ("impact_ms",), ("collision_ms",)),
        ("collision_time_s", "impact_time_s", "collision_t_s", "impact_t_s"),
        "collision_time_ms", ignore_bad_ms=True,
    )


def _extract_onset_ms(value: Any) -> Optional[float]:
    return _event_time_ms(
        value, (("stimulus_onset_ms", "stim_onset_ms", "loom_start_ms", "onset_ms"),),
        ("stimulus_onset_s", "stim_onset_s", "loom_start_s"), "stimulus_onset_ms",
    )


def _parse_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            try:
                value = value.item()
            except Exception:
                return {}
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"meta_text": value}
        return dict(parsed) if isinstance(parsed, Mapping) else {"meta": parsed}
    return dict(value) if isinstance(value, Mapping) else {}


def _infer_dt_ms(times: np.ndarray, explicit: Optional[float]) -> Optional[float]:
    if explicit is not None:
        value = _finite(explicit, "dt_ms")
        assert value is not None
        if value <= 0:
            raise ProbeError("dt_ms must be > 0")
        return value
    if times.size < 2:
        return None
    diffs = np.diff(times)
    positive = diffs[diffs > 0]
    return float(np.median(positive)) if positive.size else None


@dataclass
class NeuralTrace:
    """Normalised, in-memory trace used by the offline probe."""

    label: str
    times_ms: Any
    population_rates_hz: Mapping[str, Any] = field(default_factory=dict)
    population_spike_counts: Mapping[str, Any] = field(default_factory=dict)
    active_neurons: Any = None
    spike_counts: Any = None
    active_fraction: Any = None
    saturation_fraction: Any = None
    global_saturation: Any = None
    neuron_count: Optional[int] = None
    collision_time_ms: Optional[float] = None
    stimulus_onset_ms: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    dt_ms: Optional[float] = None

    def __post_init__(self) -> None:
        label = str(self.label).strip() or "trace"
        times = _to_1d_float(self.times_ms, "times_ms")
        if times.size == 0:
            raise ProbeError("times_ms must contain at least one sample")
        if np.any(np.diff(times) < 0):
            raise ProbeError("times_ms must be monotonic")
        n = int(times.size)
        rates: dict[str, np.ndarray] = {}
        for key, values in _mapping(self.population_rates_hz).items():
            rates[str(key)] = _coerce_series(values, n, f"population_rates_hz[{key!r}]")
        counts: dict[str, np.ndarray] = {}
        for key, values in _mapping(self.population_spike_counts).items():
            counts[str(key)] = _coerce_nonnegative_series(
                values, n, f"population_spike_counts[{key!r}]"
            )
        active = _optional_nonnegative_series(self.active_neurons, n, "active_neurons")
        global_spikes = _optional_nonnegative_series(self.spike_counts, n, "spike_counts")
        active_fraction = _optional_nonnegative_series(
            self.active_fraction, n, "active_fraction", maximum=1,
        )
        saturation_fraction = _optional_nonnegative_series(
            self.saturation_fraction, n, "saturation_fraction", maximum=1,
        )
        global_saturation = None
        if self.global_saturation is not None:
            try:
                global_saturation = np.asarray(self.global_saturation, dtype=bool)
            except (TypeError, ValueError) as exc:
                raise ProbeError("global_saturation must be boolean") from exc
            if global_saturation.ndim == 0:
                global_saturation = np.full(n, bool(global_saturation), dtype=bool)
            if global_saturation.ndim != 1 or global_saturation.size != n:
                raise ProbeError("global_saturation length must equal time length")
            global_saturation = global_saturation.copy()
        neuron_count = _optional_int(self.neuron_count, "neuron_count")
        if neuron_count == 0:
            raise ProbeError("neuron_count must be > 0 when provided")
        collision = _finite(self.collision_time_ms, "collision_time_ms", allow_none=True)
        onset = _finite(self.stimulus_onset_ms, "stimulus_onset_ms", allow_none=True)
        dt_ms = _infer_dt_ms(times, self.dt_ms)
        metadata = dict(_mapping(self.metadata))
        metadata.setdefault("trace_label", label)
        metadata.setdefault("sample_count", n)
        if dt_ms is not None:
            metadata.setdefault("dt_ms", dt_ms)
        if neuron_count is not None:
            metadata.setdefault("neuron_count", neuron_count)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "times_ms", times)
        object.__setattr__(self, "population_rates_hz", rates)
        object.__setattr__(self, "population_spike_counts", counts)
        object.__setattr__(self, "active_neurons", active)
        object.__setattr__(self, "spike_counts", global_spikes)
        object.__setattr__(self, "active_fraction", active_fraction)
        object.__setattr__(self, "saturation_fraction", saturation_fraction)
        object.__setattr__(self, "global_saturation", global_saturation)
        object.__setattr__(self, "neuron_count", neuron_count)
        object.__setattr__(self, "collision_time_ms", collision)
        object.__setattr__(self, "stimulus_onset_ms", onset)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "dt_ms", dt_ms)

    @property
    def sample_count(self) -> int:
        return int(self.times_ms.size)

    @property
    def protocol(self) -> str:
        return canonical_label(self.label)

    def to_dict(self) -> dict[str, Any]:
        """Return compact trace metadata without embedding raw arrays."""

        return {
            "label": self.label,
            "protocol": self.protocol,
            "source": self.source,
            "sample_count": self.sample_count,
            "time_start_ms": float(self.times_ms[0]),
            "time_end_ms": float(self.times_ms[-1]),
            "dt_ms": self.dt_ms,
            "neuron_count": self.neuron_count,
            "collision_time_ms": self.collision_time_ms,
            "stimulus_onset_ms": self.stimulus_onset_ms,
            "population_keys": sorted(set(self.population_rates_hz) | set(self.population_spike_counts)),
            "metadata": _json_safe(self.metadata),
        }


Trace = NeuralTrace


def trace_from_arrays(
    times_ms: Any,
    population_rates_hz: Optional[Mapping[str, Any]] = None,
    *,
    population_spike_counts: Optional[Mapping[str, Any]] = None,
    active_neurons: Any = None,
    spike_counts: Any = None,
    active_fraction: Any = None,
    saturation_fraction: Any = None,
    global_saturation: Any = None,
    neuron_count: Optional[int] = None,
    collision_time_ms: Optional[float] = None,
    stimulus_onset_ms: Optional[float] = None,
    label: str = "trace",
    metadata: Optional[Mapping[str, Any]] = None,
    source: str = "arrays",
    dt_ms: Optional[float] = None,
) -> NeuralTrace:
    """Build a trace from test fixtures or other in-memory recordings."""

    return NeuralTrace(
        label=label,
        times_ms=times_ms,
        population_rates_hz={} if population_rates_hz is None else population_rates_hz,
        population_spike_counts=(
            {} if population_spike_counts is None else population_spike_counts
        ),
        active_neurons=active_neurons,
        spike_counts=spike_counts,
        active_fraction=active_fraction,
        saturation_fraction=saturation_fraction,
        global_saturation=global_saturation,
        neuron_count=neuron_count,
        collision_time_ms=collision_time_ms,
        stimulus_onset_ms=stimulus_onset_ms,
        metadata={} if metadata is None else metadata,
        source=source,
        dt_ms=dt_ms,
    )


def _rows_to_trace(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: Optional[str] = None,
    source: str = "rows",
    metadata: Optional[Mapping[str, Any]] = None,
    dt_ms: Optional[float] = None,
    collision_time_ms: Optional[float] = None,
    neuron_count: Optional[int] = None,
    population_indices: Optional[Mapping[str, Any]] = None,
) -> NeuralTrace:
    if not rows:
        raise ProbeError("trace contains no rows")
    clean_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not clean_rows:
        raise ProbeError("trace contains no mapping rows")
    meta = dict(_mapping(metadata))
    for row in clean_rows:
        # Preserve only small, descriptive metadata from the first row.  Raw
        # neural vectors are deliberately not copied into the report metadata.
        for key in ("scenario", "config_scenario", "stimulus", "stimulus_type",
                    "condition", "backend", "device", "dt_ms", "neurons",
                    "neuron_count", "collision_time_ms", "collision_time_s",
                    "impact_ms", "impact_time_ms", "stimulus_onset_ms", "loom_start_ms"):
            if key in row and key not in meta and not isinstance(row[key], (Mapping, list, tuple)):
                meta[key] = row[key]
    explicit_dt = dt_ms
    if explicit_dt is None:
        raw_dt = _first_nested(meta, ("dt_ms", "tick_dt_ms", "frame_ms"))
        if raw_dt is not None:
            explicit_dt = _finite(raw_dt, "dt_ms")
    times: list[Optional[float]] = [_time_ms_from_mapping(row) for row in clean_rows]
    if any(value is None for value in times):
        step = _infer_dt_ms(np.array([0.0, 1.0]), explicit_dt)
        if step is None:
            # A missing timestamp is allowed for tiny fixtures, but make the
            # inferred timebase explicit in metadata rather than hiding it.
            step = 1.0
            meta["time_inferred"] = True
        times = [float(index) * step for index in range(len(clean_rows))]
        meta["time_inferred"] = True
    time_array = np.asarray(times, dtype=np.float64)

    all_rate_keys: set[str] = set()
    all_count_keys: set[str] = set()
    row_rate_maps: list[dict[str, float]] = []
    row_count_maps: list[dict[str, float]] = []
    active_values: list[Optional[float]] = []
    active_fraction_values: list[Optional[float]] = []
    saturation_fraction_values: list[Optional[float]] = []
    saturated_count_values: list[Optional[float]] = []
    global_saturation_values: list[Optional[bool]] = []
    global_spikes: list[Optional[float]] = []
    dense_population_indices = {
        str(key): np.asarray(value, dtype=np.int64).reshape(-1)
        for key, value in _mapping(population_indices).items()
    }
    for row in clean_rows:
        rates = _extract_rate_map(row)
        counts = _extract_count_map(row)
        dense_spikes = row.get("spikes")
        dense_rates = row.get("rates_hz")
        if dense_population_indices and dense_rates is not None and not isinstance(dense_rates, Mapping):
            dense_rate_array = np.asarray(dense_rates, dtype=np.float64).reshape(-1)
            for pop, ids in dense_population_indices.items():
                if np.any(ids < 0) or np.any(ids >= dense_rate_array.size):
                    raise ProbeError(f"population_indices[{pop!r}] contains an invalid dense rate index")
                if ids.size:
                    rates.setdefault(pop, float(np.mean(dense_rate_array[ids])))
        if dense_population_indices and dense_spikes is not None and not isinstance(dense_spikes, Mapping):
            dense = np.asarray(dense_spikes, dtype=np.float64).reshape(-1)
            for pop, ids in dense_population_indices.items():
                if ids.size:
                    if np.any(ids < 0) or np.any(ids >= dense.size):
                        raise ProbeError(f"population_indices[{pop!r}] contains an invalid dense spike index")
                    counts.setdefault(pop, float(np.sum(dense[ids])))
        row_rate_maps.append(rates)
        row_count_maps.append(counts)
        all_rate_keys.update(rates)
        all_count_keys.update(counts)
        active_value = _extract_scalar(row, ("active_neurons", "active_count", "n_active", "active"))
        total_spike_value = _extract_scalar(row, ("spike_count", "spikes"))
        if dense_spikes is not None and not isinstance(dense_spikes, Mapping):
            try:
                dense_array = np.asarray(dense_spikes, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                dense_array = np.empty(0, dtype=np.float64)
            if dense_array.size:
                if active_value is None:
                    active_value = float(np.count_nonzero(dense_array))
                if total_spike_value is None:
                    total_spike_value = float(np.sum(dense_array))
        if active_value is None and dense_rates is not None and not isinstance(dense_rates, Mapping):
            try:
                dense_rate_array = np.asarray(dense_rates, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                dense_rate_array = np.empty(0, dtype=np.float64)
            if dense_rate_array.size:
                active_value = float(np.count_nonzero(dense_rate_array > 0.0))
        active_values.append(active_value)
        active_fraction_values.append(_extract_scalar(
            row, ("active_fraction", "global_active_fraction", "fraction_active")
        ))
        saturation_fraction_values.append(_extract_scalar(
            row, ("saturation_fraction", "global_saturation_fraction", "fraction_saturated")
        ))
        saturated_count_values.append(_extract_scalar(
            row, ("saturated_count", "saturation_count", "n_saturated")
        ))
        global_saturation_values.append(_extract_neural_bool(
            row, ("global_saturation", "saturated", "network_saturated")
        ))
        global_spikes.append(total_spike_value)

    rates_by_pop = {
        key: np.asarray([float(item.get(key, 0.0)) for item in row_rate_maps], dtype=np.float64)
        for key in sorted(all_rate_keys)
    }
    counts_by_pop = {
        key: np.asarray([float(item.get(key, 0.0)) for item in row_count_maps], dtype=np.float64)
        for key in sorted(all_count_keys)
    }
    active = _optional_series(active_values)
    total_spikes = _optional_series(global_spikes)
    active_fraction = _optional_series(active_fraction_values)
    if neuron_count is None:
        neuron_count = _extract_neuron_count(meta) or _extract_neuron_count(clean_rows[0])
    saturation_fraction = _optional_series(saturation_fraction_values)
    if saturation_fraction is None and neuron_count:
        saturation_fraction = _optional_series(
            [None if value is None else value / float(neuron_count)
             for value in saturated_count_values]
        )
    global_saturation = _optional_series(global_saturation_values, dtype=bool, missing=False)
    if collision_time_ms is None:
        collision_time_ms = _extract_collision_ms(meta)
    if collision_time_ms is None:
        collision_time_ms = _extract_collision_ms(clean_rows)
    onset_ms = _extract_onset_ms(meta)
    if onset_ms is None:
        onset_ms = _extract_onset_ms(clean_rows)
    if label is None:
        label = _infer_label(source, meta)
    meta.setdefault("input_kind", "rows")
    return trace_from_arrays(
        time_array,
        rates_by_pop,
        population_spike_counts=counts_by_pop,
        active_neurons=active,
        spike_counts=total_spikes,
        active_fraction=active_fraction,
        saturation_fraction=saturation_fraction,
        global_saturation=global_saturation,
        neuron_count=neuron_count,
        collision_time_ms=collision_time_ms,
        stimulus_onset_ms=onset_ms,
        label=label,
        metadata=meta,
        source=source,
        dt_ms=explicit_dt,
    )


def trace_from_outputs(
    outputs: Iterable[Any],
    *,
    label: str = "trace",
    collision_time_ms: Optional[float] = None,
    neuron_count: Optional[int] = None,
    dt_ms: Optional[float] = None,
    population_indices: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> NeuralTrace:
    """Extract rows from existing ``NeuralOutput`` objects without stepping.

    The function intentionally uses the public ``timestamp``, ``rates_hz``,
    ``spikes`` and ``named_outputs`` fields.  It accepts plain mappings too,
    which keeps CPU fixtures independent of the live engine.
    """

    rows: list[dict[str, Any]] = []
    for output in outputs:
        if isinstance(output, Mapping):
            row = dict(output)
        else:
            named = getattr(output, "named_outputs", {})
            row = {
                "timestamp": getattr(output, "timestamp", None),
                "rates_hz": getattr(output, "rates_hz", None),
                "spikes": getattr(output, "spikes", None),
                "named_outputs": dict(_mapping(named)),
            }
            timestamp = row.get("timestamp")
            if timestamp is not None and not isinstance(timestamp, Mapping):
                row["timestamp"] = {
                    "t_s": getattr(timestamp, "t_s", None),
                    "tick": getattr(timestamp, "tick", None),
                }
        # Dense NeuralOutput vectors are useful when a caller supplies
        # population indices.  The row parser handles counts; rates are left
        # untouched unless the named population map is present.
        rows.append(row)
    if not rows:
        raise ProbeError("outputs contains no items")
    inferred_n = neuron_count
    if inferred_n is None:
        for row in rows:
            dense = row.get("spikes")
            if dense is not None and not isinstance(dense, Mapping):
                try:
                    arr = np.asarray(dense).reshape(-1)
                except Exception:
                    continue
                if arr.size:
                    inferred_n = int(arr.size)
                    break
            if inferred_n is None:
                dense_rates = row.get("rates_hz")
                if dense_rates is not None and not isinstance(dense_rates, Mapping):
                    try:
                        arr = np.asarray(dense_rates).reshape(-1)
                    except Exception:
                        continue
                    if arr.size:
                        inferred_n = int(arr.size)
                        break
    return _rows_to_trace(
        rows,
        label=label,
        source="NeuralOutput",
        metadata=metadata,
        dt_ms=dt_ms,
        collision_time_ms=collision_time_ms,
        neuron_count=inferred_n,
        population_indices=population_indices,
    )


def _resolve_default_population_indices(
    *,
    csr_path: Optional[Path] = None,
    populations: Sequence[str] = DEFAULT_POPULATIONS,
) -> dict[str, np.ndarray]:
    """Resolve named populations against the default MaleCNS CSR.

    ``lif_flyb.py`` stores dense neuron indices in ``spikes.npz``.  The trace
    itself has no annotation table, so use the adjacent run metadata only for
    timing and use this authoritative CSR for population membership.  This is
    deliberately a small read-only resolver and does not import the engine or
    any GPU runtime.
    """

    # Resolve the module-level default at call time so an embedding caller or
    # a CPU fixture can override the annotation location without reloading
    # this module.
    root = Path(DEFAULT_CSR if csr_path is None else csr_path)
    required = (root / "annotations.npz", root / "tables.json")
    if not root.is_dir() or not all(path.is_file() for path in required):
        return {}
    try:
        tables_payload = json.loads((root / "tables.json").read_text(encoding="utf-8"))
        types = np.asarray(tables_payload["types"]).astype(str)
        with np.load(root / "annotations.npz", allow_pickle=False) as annotations:
            type_indices = np.asarray(annotations["typeIdx"], dtype=np.int64)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ProbeError(f"cannot read MaleCNS CSR annotations from {root}: {exc}") from exc
    if type_indices.ndim != 1:
        raise ProbeError(f"MaleCNS CSR typeIdx must be one-dimensional: {root}")
    type_names = types[type_indices]
    resolved: dict[str, np.ndarray] = {}
    type_lowers = np.char.lower(np.char.strip(type_names))
    for requested in populations:
        name = str(requested)
        aliases = tuple(POPULATION_ALIASES.get(name, (name,)))
        alias_lowers = {alias.lower() for alias in aliases}
        matches = np.isin(type_lowers, np.asarray(sorted(alias_lowers), dtype=str))
        # DLMn is represented in the CSR as ``DLMn a, b`` and ``DLMn c-f``;
        # retain the DLM alias without broadening other population names.
        if name.lower() == "dlm":
            matches |= np.char.startswith(type_lowers, "dlmn")
        indices = np.flatnonzero(matches).astype(np.int64)
        if indices.size:
            resolved[name] = indices
    return resolved


def _sibling_summary(path: Path) -> tuple[dict[str, Any], Optional[Path]]:
    """Read a sparse trace's adjacent summary without loading raw trace rows."""

    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        return {}, None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read sibling summary {summary_path}: {exc}") from exc
    return (dict(payload) if isinstance(payload, Mapping) else {}), summary_path


def _npz_indexed_trace(
    path: Path, *, times: Any, rates: Mapping[str, Any], counts: Mapping[str, Any],
    active: Any, spikes: Any, neuron_count: Optional[int], collision_time_ms: Optional[float],
    label: Optional[str], metadata: Mapping[str, Any], kind: str, dt_ms: Optional[float],
    auto_indices: bool, sibling_path: Optional[Path],
) -> NeuralTrace:
    return trace_from_arrays(
        times, rates, population_spike_counts=counts, active_neurons=active,
        spike_counts=spikes, neuron_count=neuron_count,
        collision_time_ms=collision_time_ms, label=label or _infer_label(path, metadata),
        metadata={
            **metadata,
            "input_kind": kind,
            "population_index_source": str(DEFAULT_CSR) if auto_indices else "caller",
            "sibling_summary": str(sibling_path) if sibling_path else None,
        },
        source=str(path), dt_ms=dt_ms,
    )


def _population_index_map(
    population_indices: Optional[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], bool]:
    indices = _mapping(population_indices)
    if indices:
        return indices, False
    indices = _resolve_default_population_indices()
    if not indices:
        raise ProbeError(
            "spikes.npz contains dense neuron indices but no population mapping; "
            "provide population_indices/--population-indices or install the "
            f"default MaleCNS CSR at {DEFAULT_CSR}"
        )
    return indices, True


def _indexed_population_series(
    indices: Mapping[str, Any], count_at: Callable[[np.ndarray], np.ndarray],
    dt_ms: Optional[float], *, neuron_limit: Optional[int] = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    counts: dict[str, np.ndarray] = {}
    rates: dict[str, np.ndarray] = {}
    for pop, value in indices.items():
        ids = np.asarray(value, dtype=np.int64).reshape(-1)
        if neuron_limit is not None and (np.any(ids < 0) or np.any(ids >= neuron_limit)):
            raise ProbeError(f"population_indices[{pop!r}] contains an invalid dense index")
        pop_counts = count_at(ids)
        counts[str(pop)] = pop_counts
        if ids.size and dt_ms:
            rates[str(pop)] = pop_counts / (ids.size * dt_ms / 1000.0)
    return counts, rates


def _load_npz(
    path: Path,
    *,
    label: Optional[str] = None,
    collision_time_ms: Optional[float] = None,
    neuron_count: Optional[int] = None,
    population_indices: Optional[Mapping[str, Any]] = None,
) -> NeuralTrace:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ProbeError(f"cannot read NPZ trace {path}: {exc}") from exc
    with data:
        keys = set(data.files)
        metadata = _parse_meta(data["meta"]) if "meta" in keys else {}
        if "metadata" in keys:
            metadata.update(_parse_meta(data["metadata"]))
        sibling_summary, sibling_summary_path = _sibling_summary(path)
        sibling_neurons = _extract_neuron_count(sibling_summary)
        sibling_collision = _extract_collision_ms(sibling_summary)
        sibling_dt = _first_nested(
            sibling_summary,
            ("trace_frame_ms", "frame_ms", "tick_dt_ms", "dt_ms"),
        )
        if neuron_count is None:
            neuron_count = _extract_neuron_count(metadata) or sibling_neurons
        if collision_time_ms is None:
            collision_time_ms = _extract_collision_ms(metadata)
        if collision_time_ms is None:
            collision_time_ms = sibling_collision
        dt_ms: Optional[float] = None
        for key in ("dt_ms", "frame_ms", "tick_dt_ms"):
            if key in keys:
                try:
                    dt_ms = float(np.asarray(data[key]).reshape(-1)[0])
                except (TypeError, ValueError, IndexError):
                    pass
                break
            if key in metadata:
                try:
                    dt_ms = float(metadata[key])
                except (TypeError, ValueError):
                    pass
                break
        if dt_ms is None and sibling_dt is not None:
            try:
                dt_ms = float(sibling_dt)
            except (TypeError, ValueError):
                dt_ms = None

        raw_rates = None
        for key in ("population_rates_hz", "population_rates", "rates_by_population"):
            if key in keys:
                raw_rates = _parse_meta(data[key])
                break
        raw_counts = None
        for key in ("population_spike_counts", "spike_counts_by_population"):
            if key in keys:
                raw_counts = _parse_meta(data[key])
                break
        if isinstance(raw_rates, Mapping):
            times_key = "times_ms" if "times_ms" in keys else "times"
            if times_key not in keys:
                raise ProbeError(f"population-rate NPZ trace {path} lacks times_ms/times")
            return trace_from_arrays(
                data[times_key], raw_rates,
                population_spike_counts=raw_counts if isinstance(raw_counts, Mapping) else {},
                active_neurons=data["active_neurons"] if "active_neurons" in keys else None,
                spike_counts=data["spike_count"] if "spike_count" in keys else None,
                active_fraction=data["active_fraction"] if "active_fraction" in keys else None,
                saturation_fraction=(
                    data["saturation_fraction"] if "saturation_fraction" in keys else None
                ),
                global_saturation=(
                    data["global_saturation"] if "global_saturation" in keys else None
                ),
                neuron_count=neuron_count,
                collision_time_ms=collision_time_ms,
                label=label or _infer_label(path, metadata),
                metadata={**metadata, "input_kind": "npz_population_rates"},
                source=str(path),
                dt_ms=dt_ms,
            )

        if {"times", "offsets", "neurons"}.issubset(keys):
            times = np.asarray(data["times"], dtype=np.float64).reshape(-1)
            offsets = np.asarray(data["offsets"], dtype=np.int64).reshape(-1)
            neurons = np.asarray(data["neurons"], dtype=np.int64).reshape(-1)
            if offsets.size != times.size + 1:
                raise ProbeError("sparse spikes.npz offsets must have len(times)+1")
            if offsets.size and (offsets[0] != 0 or offsets[-1] != neurons.size):
                raise ProbeError("sparse spikes.npz offsets do not span neurons")
            if np.any(np.diff(offsets) < 0):
                raise ProbeError("sparse spikes.npz offsets must be monotonic")
            inferred_dt = _infer_dt_ms(times, None)
            if inferred_dt is not None:
                # Sparse trace timestamps describe the recorded window, while
                # a sibling summary's ``dt_ms`` may be the 0.5 ms integration
                # step.  Population rates must use the trace window width.
                dt_ms = inferred_dt
            active = np.empty(times.size, dtype=np.float64)
            spikes = np.empty(times.size, dtype=np.float64)
            for index in range(times.size):
                chunk = neurons[int(offsets[index]):int(offsets[index + 1])]
                active[index] = np.unique(chunk).size
                spikes[index] = chunk.size
            # Sparse files do not carry annotation/population membership.  A
            # caller can provide dense index lists for an exact offline count.
            indices, auto_indices = _population_index_map(population_indices)
            counts, rates = _indexed_population_series(
                indices,
                lambda ids: np.fromiter((
                    np.count_nonzero(np.isin(
                        neurons[int(offsets[i]):int(offsets[i + 1])], ids,
                    )) for i in range(times.size)
                ), dtype=np.float64, count=times.size),
                dt_ms,
            )
            return _npz_indexed_trace(
                path, times=times, rates=rates, counts=counts, active=active, spikes=spikes,
                neuron_count=neuron_count, collision_time_ms=collision_time_ms, label=label,
                metadata=metadata, kind="npz_sparse_spikes", dt_ms=dt_ms,
                auto_indices=auto_indices, sibling_path=sibling_summary_path,
            )

        dense_key = next((key for key in ("spikes", "spike_matrix") if key in keys), None)
        if dense_key is not None:
            dense = np.asarray(data[dense_key])
            if dense.ndim != 2:
                raise ProbeError(f"{dense_key} must be a 2-D [time, neuron] array")
            times_key = "times_ms" if "times_ms" in keys else "times"
            if times_key in keys:
                times = np.asarray(data[times_key], dtype=np.float64).reshape(-1)
            elif dt_ms:
                times = np.arange(dense.shape[0], dtype=np.float64) * dt_ms
            else:
                raise ProbeError(f"dense spike NPZ trace {path} lacks times/times_ms or dt_ms")
            if times.size != dense.shape[0]:
                raise ProbeError("dense spike NPZ times length differs from spike rows")
            active = np.count_nonzero(dense, axis=1).astype(np.float64)
            total = np.asarray(dense, dtype=np.float64).sum(axis=1)
            indices, auto_indices = _population_index_map(population_indices)
            counts, rates = _indexed_population_series(
                indices,
                lambda ids: np.asarray(dense[:, ids], dtype=np.float64).sum(axis=1),
                dt_ms, neuron_limit=dense.shape[1],
            )
            return _npz_indexed_trace(
                path, times=times, rates=rates, counts=counts, active=active, spikes=total,
                neuron_count=neuron_count or int(dense.shape[1]),
                collision_time_ms=collision_time_ms, label=label, metadata=metadata,
                kind="npz_dense_spikes", dt_ms=dt_ms, auto_indices=auto_indices,
                sibling_path=sibling_summary_path,
            )
    raise ProbeError(f"NPZ trace {path} has no supported times/spikes/population arrays")


def _load_json(path: Path, *, label: Optional[str] = None,
               collision_time_ms: Optional[float] = None,
               neuron_count: Optional[int] = None) -> NeuralTrace:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read JSON trace {path}: {exc}") from exc
    return _load_payload(payload, source=str(path), label=label,
                         collision_time_ms=collision_time_ms,
                         neuron_count=neuron_count)


def _load_payload(payload: Any, *, source: str, label: Optional[str] = None,
                  collision_time_ms: Optional[float] = None,
                  neuron_count: Optional[int] = None) -> NeuralTrace:
    if isinstance(payload, list):
        return _rows_to_trace(payload, label=label, source=source,
                              collision_time_ms=collision_time_ms,
                              neuron_count=neuron_count)
    if not isinstance(payload, Mapping):
        raise ProbeError(f"JSON trace {source} must be an object or row list")
    # Explicit array representation is convenient for small hand-authored
    # fixtures and keeps the loader symmetric with trace_from_arrays.
    if any(key in payload for key in ("times_ms", "times", "times_s", "time_ms")) and any(
        key in payload for key in (
            "population_rates_hz", "population_rates", "rates_by_population",
            "rates_hz", "rates", "populations",
        )
    ):
        times = payload.get(
            "times_ms", payload.get("times", payload.get("time_ms", payload.get("times_s")))
        )
        if "times_ms" not in payload and "time_ms" not in payload and "times_s" in payload:
            times = np.asarray(times, dtype=np.float64) * 1000.0
        rates = payload.get(
            "population_rates_hz",
            payload.get(
                "population_rates",
                payload.get("rates_by_population", payload.get("rates_hz", payload.get("rates", payload.get("populations", {})))),
            ),
        )
        return trace_from_arrays(
            times, rates,
            population_spike_counts=payload.get(
                "population_spike_counts", payload.get("spikes_by_population", {})
            ),
            active_neurons=payload.get("active_neurons"),
            spike_counts=payload.get("spike_counts", payload.get("spike_count")),
            active_fraction=payload.get("active_fraction", payload.get("global_active_fraction")),
            saturation_fraction=payload.get(
                "saturation_fraction", payload.get("global_saturation_fraction")
            ),
            global_saturation=payload.get("global_saturation", payload.get("saturated")),
            neuron_count=neuron_count or _extract_neuron_count(payload),
            collision_time_ms=(collision_time_ms if collision_time_ms is not None
                               else _extract_collision_ms(payload)),
            stimulus_onset_ms=_extract_onset_ms(payload),
            label=label or _infer_label(source, payload),
            metadata={**payload.get("metadata", {})} if isinstance(payload.get("metadata"), Mapping) else {},
            source=source,
            dt_ms=payload.get("dt_ms", payload.get("frame_ms")),
        )
    for key in ("rows", "telemetry", "ticks", "samples", "records"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            metadata = {str(k): v for k, v in payload.items()
                        if k not in {key, "rows", "telemetry", "ticks", "samples", "records"}
                        and isinstance(v, (str, int, float, bool))}
            nested_metadata = payload.get("metadata")
            if isinstance(nested_metadata, Mapping):
                metadata["metadata"] = dict(nested_metadata)
            resolved_collision = (
                collision_time_ms if collision_time_ms is not None
                else _extract_collision_ms(payload)
            )
            resolved_neurons = (
                neuron_count if neuron_count is not None
                else _extract_neuron_count(payload)
            )
            return _rows_to_trace(candidate, label=label, source=source,
                                  metadata=metadata,
                                  collision_time_ms=resolved_collision,
                                  neuron_count=resolved_neurons)
    # A single telemetry row is still a valid one-sample trace.
    return _rows_to_trace([payload], label=label, source=source,
                          collision_time_ms=collision_time_ms,
                          neuron_count=neuron_count)


def _load_jsonl(path: Path, *, label: Optional[str] = None,
                collision_time_ms: Optional[float] = None,
                neuron_count: Optional[int] = None) -> NeuralTrace:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ProbeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, Mapping):
                    raise ProbeError(f"JSONL row at {path}:{line_number} is not an object")
                rows.append(value)
    except OSError as exc:
        raise ProbeError(f"cannot read JSONL trace {path}: {exc}") from exc
    return _rows_to_trace(rows, label=label, source=str(path),
                          collision_time_ms=collision_time_ms,
                          neuron_count=neuron_count)


def _load_csv(path: Path, *, label: Optional[str] = None,
              collision_time_ms: Optional[float] = None,
              neuron_count: Optional[int] = None) -> NeuralTrace:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ProbeError(f"cannot read CSV trace {path}: {exc}") from exc
    if not rows:
        raise ProbeError(f"CSV trace {path} contains no rows")
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value == "":
                continue
            text = value.strip()
            try:
                item[key] = json.loads(text)
            except (ValueError, json.JSONDecodeError):
                try:
                    item[key] = float(text)
                except ValueError:
                    item[key] = text
        cleaned.append(item)
    # Long-form CSV: ``time_ms,population,rate_hz``.
    if all(("population" in row or "population_name" in row) and
           ("rate_hz" in row or "rate" in row) for row in cleaned):
        grouped: dict[float, dict[str, Any]] = {}
        for row in cleaned:
            time = float(row.get("time_ms", row.get("t_ms", row.get("time_s", 0.0))))
            if "time_s" in row and "time_ms" not in row and "t_ms" not in row:
                time *= 1000.0
            grouped.setdefault(time, {}).setdefault("rates", {})[
                str(row.get("population", row.get("population_name")))
            ] = float(row.get("rate_hz", row.get("rate")))
        cleaned = [{"time_ms": time, **value} for time, value in sorted(grouped.items())]
    return _rows_to_trace(cleaned, label=label, source=str(path),
                          collision_time_ms=collision_time_ms,
                          neuron_count=neuron_count)


def _overlay_trace_metadata(
    trace: NeuralTrace,
    metadata: Mapping[str, Any],
    *,
    neuron_count: Optional[int] = None,
    collision_time_ms: Optional[float] = None,
) -> NeuralTrace:
    """Merge a sibling manifest into a trace without changing its samples."""

    resolved_neurons = trace.neuron_count
    if resolved_neurons is None:
        resolved_neurons = neuron_count if neuron_count is not None else _extract_neuron_count(metadata)
    resolved_collision = trace.collision_time_ms
    if resolved_collision is None:
        resolved_collision = (
            collision_time_ms if collision_time_ms is not None else _extract_collision_ms(metadata)
        )
    if resolved_neurons == trace.neuron_count and resolved_collision == trace.collision_time_ms:
        return trace
    return trace_from_arrays(
        trace.times_ms,
        trace.population_rates_hz,
        population_spike_counts=trace.population_spike_counts,
        active_neurons=trace.active_neurons,
        spike_counts=trace.spike_counts,
        active_fraction=trace.active_fraction,
        saturation_fraction=trace.saturation_fraction,
        global_saturation=trace.global_saturation,
        neuron_count=resolved_neurons,
        collision_time_ms=resolved_collision,
        stimulus_onset_ms=trace.stimulus_onset_ms,
        label=trace.label,
        metadata={**dict(_mapping(metadata)), **dict(trace.metadata)},
        source=trace.source,
        dt_ms=trace.dt_ms,
    )


def load_trace(
    source: Any,
    *,
    label: Optional[str] = None,
    collision_time_ms: Optional[float] = None,
    neuron_count: Optional[int] = None,
    population_indices: Optional[Mapping[str, Any]] = None,
) -> NeuralTrace:
    """Load one existing trace or normalise a trace-like object.

    Direct ``NeuralOutput`` sequences should use :func:`trace_from_outputs`;
    this function also recognises a sequence of mappings for fixture use.
    """

    if isinstance(source, NeuralTrace):
        if label is None and collision_time_ms is None and neuron_count is None:
            return source
        return trace_from_arrays(
            source.times_ms, source.population_rates_hz,
            population_spike_counts=source.population_spike_counts,
            active_neurons=source.active_neurons,
            spike_counts=source.spike_counts,
            active_fraction=source.active_fraction,
            saturation_fraction=source.saturation_fraction,
            global_saturation=source.global_saturation,
            neuron_count=source.neuron_count if neuron_count is None else neuron_count,
            collision_time_ms=(source.collision_time_ms if collision_time_ms is None
                               else collision_time_ms),
            stimulus_onset_ms=source.stimulus_onset_ms,
            label=source.label if label is None else label,
            metadata=source.metadata,
            source=source.source,
            dt_ms=source.dt_ms,
        )
    if isinstance(source, Mapping):
        return _load_payload(source, source="mapping", label=label,
                             collision_time_ms=collision_time_ms,
                             neuron_count=neuron_count)
    if hasattr(source, "timestamp") and hasattr(source, "named_outputs"):
        return trace_from_outputs(
            [source], label=label or "trace",
            collision_time_ms=collision_time_ms,
            neuron_count=neuron_count,
            population_indices=population_indices,
        )
    if not isinstance(source, (str, Path)):
        if isinstance(source, Sequence):
            if source and not isinstance(source[0], Mapping):
                return trace_from_outputs(source, label=label or "trace",
                                           collision_time_ms=collision_time_ms,
                                           neuron_count=neuron_count,
                                           population_indices=population_indices)
            return _rows_to_trace(source, label=label, source="rows",
                                  collision_time_ms=collision_time_ms,
                                  neuron_count=neuron_count,
                                  population_indices=population_indices)
        raise ProbeError(f"unsupported trace source type: {type(source).__name__}")
    path = Path(source)
    if path.is_dir():
        manifest_metadata: dict[str, Any] = {}
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                manifest_payload = {}
            if isinstance(manifest_payload, Mapping):
                manifest_metadata = dict(manifest_payload)
        for candidate in (path / "telemetry.jsonl", path / "summary.json", path / "spikes.npz"):
            if candidate.is_file():
                trace = load_trace(candidate, label=label or _infer_label(path),
                                   collision_time_ms=collision_time_ms,
                                   neuron_count=neuron_count,
                                   population_indices=population_indices)
                manifest_summary = {
                    "manifest_neuron_count": _extract_neuron_count(manifest_metadata),
                    "manifest_collision_time_ms": _extract_collision_ms(manifest_metadata),
                }
                manifest_neurons = _extract_neuron_count(manifest_metadata)
                manifest_collision = _extract_collision_ms(manifest_metadata)
                return _overlay_trace_metadata(
                    trace,
                    manifest_summary,
                    neuron_count=neuron_count if neuron_count is not None else manifest_neurons,
                    collision_time_ms=(
                        collision_time_ms if collision_time_ms is not None else manifest_collision
                    ),
                )
        raise ProbeError(f"directory {path} has no telemetry.jsonl, summary.json, or spikes.npz")
    if not path.is_file():
        raise ProbeError(f"trace path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path, label=label, collision_time_ms=collision_time_ms,
                           neuron_count=neuron_count)
    if suffix == ".json":
        return _load_json(path, label=label, collision_time_ms=collision_time_ms,
                          neuron_count=neuron_count)
    if suffix == ".npz":
        return _load_npz(path, label=label, collision_time_ms=collision_time_ms,
                         neuron_count=neuron_count,
                         population_indices=population_indices)
    if suffix == ".csv":
        return _load_csv(path, label=label, collision_time_ms=collision_time_ms,
                         neuron_count=neuron_count)
    raise ProbeError(f"unsupported trace extension {path.suffix!r}: {path}")


def _normalise_population_name(name: Any) -> str:
    text = str(name).strip()
    if text.lower().startswith("prefix:"):
        text = text.split(":", 1)[1]
    return text


def _base_population_name(name: Any) -> str:
    text = _normalise_population_name(name)
    match = _SIDE_RE.search(text)
    if match:
        text = text[:match.start()]
    low = text.lower()
    if low == "dlmn":
        return "DLM"
    return text


def _split_side(name: Any) -> Optional[tuple[str, str]]:
    text = _normalise_population_name(name)
    match = _SIDE_RE.search(text)
    if not match:
        return None
    side = (match.group("side") or match.group("long") or "").upper()
    if side == "LEFT":
        side = "L"
    elif side == "RIGHT":
        side = "R"
    return text[:match.start()], side


def _key_matches_population(key: str, requested: str) -> bool:
    key_norm = _normalise_population_name(key).lower()
    requested_norm = _normalise_population_name(requested).lower()
    aliases = {item.lower() for item in POPULATION_ALIASES.get(requested, (requested,))}
    if key_norm in aliases:
        return True
    split = _split_side(key)
    if split is not None:
        base = split[0].lower()
        return base in aliases
    return False


def _population_series(trace: NeuralTrace, requested: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    rates: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    rate_candidates = [(key, value) for key, value in trace.population_rates_hz.items()
                       if _key_matches_population(key, requested)]
    count_candidates = [(key, value) for key, value in trace.population_spike_counts.items()
                        if _key_matches_population(key, requested)]
    # Exact canonical key wins over averaging side-labelled channels.
    for key, value in rate_candidates:
        if _normalise_population_name(key).lower() in {
            _normalise_population_name(requested).lower(),
            *(item.lower() for item in POPULATION_ALIASES.get(requested, ())),
        }:
            rates = value.copy()
            break
    if rates is None and rate_candidates:
        rates = np.nanmean(np.vstack([value for _, value in rate_candidates]), axis=0)
    for key, value in count_candidates:
        if _normalise_population_name(key).lower() in {
            _normalise_population_name(requested).lower(),
            *(item.lower() for item in POPULATION_ALIASES.get(requested, ())),
        }:
            counts = value.copy()
            break
    if counts is None and count_candidates:
        counts = np.nansum(np.vstack([value for _, value in count_candidates]), axis=0)
    aggregation = "exact"
    if len(rate_candidates) > 1 or len(count_candidates) > 1:
        aggregation = "mean_rate_sum_count_across_sides"
    # A rate-only telemetry trace may still carry exact scalar aliases under a
    # different spelling.  The selector grammar is intentionally conservative.
    return rates, counts, aggregation


def _first_index(values: Optional[np.ndarray]) -> Optional[int]:
    if values is None or values.size == 0:
        return None
    finite = np.isfinite(values)
    indices = np.flatnonzero(finite & (values > 0.0))
    return int(indices[0]) if indices.size else None


def _peak(values: Optional[np.ndarray]) -> tuple[Optional[float], Optional[int]]:
    if values is None or values.size == 0:
        return None, None
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, None
    masked = np.where(finite, values, -np.inf)
    index = int(np.argmax(masked))
    return float(masked[index]), index


def _event_relative_collision(time_ms: Optional[float], collision_ms: Optional[float]) -> Optional[float]:
    if time_ms is None or collision_ms is None:
        return None
    return float(collision_ms - time_ms)


def _population_metric(trace: NeuralTrace, requested: str) -> dict[str, Any]:
    rates, counts, aggregation = _population_series(trace, requested)
    event_values = counts if counts is not None else rates
    first_index = _first_index(event_values)
    first_ms = None if first_index is None else float(trace.times_ms[first_index])
    peak_rate, peak_rate_index = _peak(rates)
    peak_count, peak_count_index = _peak(counts)
    peak_index = peak_rate_index if peak_rate_index is not None else peak_count_index
    peak_ms = None if peak_index is None else float(trace.times_ms[peak_index])
    last_index = None
    if event_values is not None and event_values.size:
        indices = np.flatnonzero(np.isfinite(event_values) & (event_values > 0.0))
        if indices.size:
            last_index = int(indices[-1])
    last_ms = None if last_index is None else float(trace.times_ms[last_index])
    result: dict[str, Any] = {
        "population": requested,
        "aggregation": aggregation,
        "observed": rates is not None or counts is not None,
        "n_samples": trace.sample_count,
        "first_spike_ms": first_ms,
        "first_spike_time_ms": first_ms,
        "first_spike_index": first_index,
        "relative_collision_ms": _event_relative_collision(first_ms, trace.collision_time_ms),
        "first_spike_to_collision_ms": _event_relative_collision(first_ms, trace.collision_time_ms),
        "peak_rate_hz": peak_rate,
        "peak_hz": peak_rate,
        "peak_rate_time_ms": peak_ms if peak_rate_index is not None else None,
        "peak_spike_count": peak_count,
        "peak_spike_count_time_ms": (
            None if peak_count_index is None else float(trace.times_ms[peak_count_index])
        ),
        "peak_relative_collision_ms": _event_relative_collision(peak_ms, trace.collision_time_ms),
        "peak_to_collision_ms": _event_relative_collision(peak_ms, trace.collision_time_ms),
        "last_spike_ms": last_ms,
        "response_duration_ms": (
            None if first_ms is None or last_ms is None else float(last_ms - first_ms)
        ),
        "total_spike_count": None if counts is None else float(np.nansum(counts)),
    }
    if trace.stimulus_onset_ms is not None and first_ms is not None:
        result["latency_from_stimulus_ms"] = float(first_ms - trace.stimulus_onset_ms)
    else:
        result["latency_from_stimulus_ms"] = None
    return result


def _global_activity(trace: NeuralTrace) -> dict[str, Any]:
    active = trace.active_neurons
    spikes = trace.spike_counts
    result: dict[str, Any] = {
        "observed": active is not None or spikes is not None,
        "neuron_count": trace.neuron_count,
        "active_neurons_first": None,
        "active_neurons_peak": None,
        "active_neurons_peak_time_ms": None,
        "active_neurons_final": None,
        "active_neurons_mean": None,
        "peak_active_fraction": None,
        "global_saturation_fraction": None,
        "peak_declared_saturation_fraction": None,
        "global_saturation_observed": None,
        "saturation_source": None,
        "spike_count_first": None,
        "spike_count_peak": None,
        "spike_count_peak_time_ms": None,
        "spike_count_total": None,
        "saturation_definition": (
            "peak_active_neurons / neuron_count; continuous descriptive index, "
            "not an action threshold"
        ),
    }
    if active is not None and active.size:
        peak_index = int(np.argmax(active))
        result.update({
            "active_neurons_first": float(active[0]),
            "active_neurons_peak": float(active[peak_index]),
            "active_neurons_peak_time_ms": float(trace.times_ms[peak_index]),
            "active_neurons_final": float(active[-1]),
            "active_neurons_mean": float(np.mean(active)),
        })
        if trace.neuron_count:
            fraction = active / float(trace.neuron_count)
            peak_fraction = float(np.max(fraction))
            result["peak_active_fraction"] = peak_fraction
            result["global_saturation_fraction"] = peak_fraction
            result["saturation_source"] = "derived_active_fraction"
    if trace.active_fraction is not None and trace.active_fraction.size:
        result["peak_active_fraction"] = float(np.max(trace.active_fraction))
    if trace.saturation_fraction is not None and trace.saturation_fraction.size:
        result["peak_declared_saturation_fraction"] = float(np.max(trace.saturation_fraction))
        result["global_saturation_fraction"] = result["peak_declared_saturation_fraction"]
        result["saturation_source"] = "telemetry_saturation_fraction"
    if trace.global_saturation is not None and trace.global_saturation.size:
        result["global_saturation_observed"] = bool(np.any(trace.global_saturation))
    if spikes is not None and spikes.size:
        peak_index = int(np.argmax(spikes))
        result.update({
            "spike_count_first": float(spikes[0]),
            "spike_count_peak": float(spikes[peak_index]),
            "spike_count_peak_time_ms": float(trace.times_ms[peak_index]),
            "spike_count_total": float(np.sum(spikes)),
        })
    return result


def _directional_differences(trace: NeuralTrace) -> dict[str, Any]:
    keys = set(trace.population_rates_hz) | set(trace.population_spike_counts)
    grouped: MutableMapping[str, MutableMapping[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for key in keys:
        split = _split_side(key)
        if split is None:
            continue
        grouped[_base_population_name(split[0])][split[1]].append(key)
    result: dict[str, Any] = {}
    for base, sides in sorted(grouped.items()):
        if "L" not in sides or "R" not in sides:
            continue
        def side_series(side: str) -> tuple[Optional[np.ndarray], str]:
            side_keys = sides[side]
            rate_arrays = [trace.population_rates_hz[key] for key in side_keys
                           if key in trace.population_rates_hz]
            if rate_arrays:
                return np.nanmean(np.vstack(rate_arrays), axis=0), "Hz"
            count_arrays = [trace.population_spike_counts[key] for key in side_keys
                            if key in trace.population_spike_counts]
            if count_arrays:
                return np.nansum(np.vstack(count_arrays), axis=0), "spikes_per_sample"
            return None, "unknown"
        left, units = side_series("L")
        right, right_units = side_series("R")
        if left is None or right is None:
            continue
        if left.size != right.size:
            continue
        diff = left - right
        denom = left + right
        normalized = np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)
        peak_idx = int(np.nanargmax(np.abs(diff))) if diff.size and np.any(np.isfinite(diff)) else None
        first_left = _first_index(left)
        first_right = _first_index(right)
        result[base] = {
            "left_keys": sorted(sides["L"]),
            "right_keys": sorted(sides["R"]),
            "units": units if units == right_units else f"{units}/{right_units}",
            "mean_left_minus_right": float(np.nanmean(diff)),
            "peak_abs_left_minus_right": None if peak_idx is None else float(abs(diff[peak_idx])),
            "peak_signed_left_minus_right": None if peak_idx is None else float(diff[peak_idx]),
            "peak_time_ms": None if peak_idx is None else float(trace.times_ms[peak_idx]),
            "mean_normalized_left_minus_right": float(np.nanmean(normalized)),
            "peak_abs_normalized_left_minus_right": float(np.nanmax(np.abs(normalized))) if normalized.size else None,
            "first_left_ms": None if first_left is None else float(trace.times_ms[first_left]),
            "first_right_ms": None if first_right is None else float(trace.times_ms[first_right]),
            "first_left_minus_right_ms": (
                None if first_left is None or first_right is None
                else float(trace.times_ms[first_left] - trace.times_ms[first_right])
            ),
            "relative_collision_left_ms": (
                None if first_left is None else _event_relative_collision(
                    float(trace.times_ms[first_left]), trace.collision_time_ms
                )
            ),
            "relative_collision_right_ms": (
                None if first_right is None else _event_relative_collision(
                    float(trace.times_ms[first_right]), trace.collision_time_ms
                )
            ),
        }
    return result


def analyze_trace(
    trace: NeuralTrace | Any,
    *,
    populations: Sequence[str] = DEFAULT_POPULATIONS,
) -> dict[str, Any]:
    """Extract descriptive escape-circuit metrics from one offline trace."""

    normalised = load_trace(trace) if not isinstance(trace, NeuralTrace) else trace
    population_metrics = {
        str(name): _population_metric(normalised, str(name)) for name in populations
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "trace": normalised.to_dict(),
        "populations": population_metrics,
        "global_activity": _global_activity(normalised),
        "direction_differences": _directional_differences(normalised),
        "direction_difference": _directional_differences(normalised),
        "notes": {
            "interpretation": "read-only descriptive telemetry; no behavior or motor command is generated",
            "relative_collision_ms": "collision_time_ms - event_time_ms; positive means before collision",
            "missing_population": "reported with observed=false and null event times",
        },
    }


# Short aliases make the module convenient to import from notebooks and tests.
probe_trace = analyze_trace
extract_metrics = analyze_trace
extract_probe_metrics = analyze_trace
report_from_trace = analyze_trace


def _delta(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    try:
        value = float(left) - float(right)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _compare_report(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    pop_delta: dict[str, Any] = {}
    for name, metric in _mapping(report.get("populations")).items():
        base_metric = _mapping(_mapping(baseline.get("populations")).get(name))
        pop_delta[name] = {
            "first_spike_ms": _delta(metric.get("first_spike_ms"), base_metric.get("first_spike_ms")),
            "peak_rate_hz": _delta(metric.get("peak_rate_hz"), base_metric.get("peak_rate_hz")),
            "peak_rate_time_ms": _delta(metric.get("peak_rate_time_ms"), base_metric.get("peak_rate_time_ms")),
            "relative_collision_ms": _delta(metric.get("relative_collision_ms"), base_metric.get("relative_collision_ms")),
            "peak_relative_collision_ms": _delta(metric.get("peak_relative_collision_ms"), base_metric.get("peak_relative_collision_ms")),
            "total_spike_count": _delta(metric.get("total_spike_count"), base_metric.get("total_spike_count")),
        }
    global_now = _mapping(report.get("global_activity"))
    global_base = _mapping(baseline.get("global_activity"))
    global_delta = {
        key: _delta(global_now.get(key), global_base.get(key))
        for key in ("active_neurons_peak", "active_neurons_mean", "peak_active_fraction",
                    "global_saturation_fraction", "peak_declared_saturation_fraction",
                    "spike_count_peak", "spike_count_total")
    }
    return {
        "populations": pop_delta,
        "global_activity": global_delta,
        "direction_differences": report.get("direction_differences", {}),
    }


def compare_traces(
    traces: Mapping[str, Any] | Iterable[Any],
    *,
    baseline: Optional[str] = "sham",
    populations: Sequence[str] = DEFAULT_POPULATIONS,
    collision_time_ms: Optional[float] = None,
    neuron_count: Optional[int] = None,
) -> dict[str, Any]:
    """Analyse sham/loom/palm traces and compute descriptive deltas.

    ``traces`` may be ``{"sham": path, "loom": path, "palm": path}`` or an
    iterable of :class:`NeuralTrace` objects.  Deltas are always reported as
    ``condition - baseline``; they never feed an action or command path.
    """

    loaded: dict[str, NeuralTrace] = {}
    if isinstance(traces, Mapping):
        for name, source in traces.items():
            loaded[str(name)] = load_trace(
                source, label=str(name), collision_time_ms=collision_time_ms,
                neuron_count=neuron_count,
            )
    else:
        for index, source in enumerate(traces):
            trace = load_trace(source, collision_time_ms=collision_time_ms,
                               neuron_count=neuron_count)
            key = trace.label if trace.label != "trace" else f"trace_{index}"
            loaded[str(key)] = trace
    if not loaded:
        raise ProbeError("at least one trace is required")
    reports = {name: analyze_trace(trace, populations=populations)
               for name, trace in loaded.items()}
    chosen_baseline: Optional[str] = baseline if baseline in reports else None
    if chosen_baseline is None:
        for name, trace in loaded.items():
            if trace.protocol == "sham":
                chosen_baseline = name
                break
    comparisons: dict[str, Any] = {}
    if chosen_baseline is not None:
        base_report = reports[chosen_baseline]
        for name, report in reports.items():
            if name != chosen_baseline:
                comparisons[name] = _compare_report(report, base_report)
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": chosen_baseline,
        "traces": reports,
        "comparisons_vs_baseline": comparisons,
        "comparisons": comparisons,
        "notes": {
            "delta_convention": "condition - baseline",
            "protocols": {name: loaded[name].protocol for name in loaded},
            "interpretation": "read-only offline comparison; no threshold or action generation",
        },
    }


compare_trace_set = compare_traces
compare_conditions = compare_traces
compare_trace_files = compare_traces


def write_json_report(report: Mapping[str, Any], output: str | Path) -> Path:
    """Write one JSON report and return its path."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


write_report = write_json_report
load_telemetry = load_trace


def _parse_trace_spec(spec: str, *, index: int, label: Optional[str]) -> tuple[str, str]:
    text = str(spec).strip()
    if not text:
        raise ProbeError("--trace value must not be empty")
    if "=" in text:
        left, right = text.split("=", 1)
        if left.strip() and right.strip():
            return left.strip(), right.strip()
    if label and index == 0:
        return label, text
    return _infer_label(text), text


def _load_population_indices(path: Optional[str]) -> Optional[dict[str, np.ndarray]]:
    if path is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read population indices {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ProbeError("population indices JSON must be an object")
    return {str(key): np.asarray(value, dtype=np.int64).reshape(-1)
            for key, value in payload.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", default=[],
                        help="existing trace path, optionally LABEL=PATH; repeat for sham/loom/palm")
    parser.add_argument("paths", nargs="*", help="trace paths (labels inferred from names)")
    parser.add_argument("--label", default=None, help="label for the first positional/--trace path")
    parser.add_argument("--baseline", default="sham", help="comparison baseline label (default: sham)")
    parser.add_argument("--collision-ms", type=float, default=None,
                        help="collision/impact time in milliseconds, if absent from trace metadata")
    parser.add_argument("--neuron-count", type=int, default=None,
                        help="total MaleCNS neurons for active/saturation fraction")
    parser.add_argument("--population-indices", type=str, default=None,
                        help="JSON mapping population names to dense indices for spikes.npz")
    parser.add_argument("--population", action="append", dest="populations", default=None,
                        help="population to report; may be repeated (default: escape circuit set)")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = parser.parse_args(argv)
    specs = list(args.trace) + list(args.paths)
    if not specs:
        parser.error("at least one --trace or positional trace path is required")
    try:
        trace_specs = [_parse_trace_spec(spec, index=index, label=args.label)
                       for index, spec in enumerate(specs)]
        indices = _load_population_indices(args.population_indices)
        traces = {
            label: load_trace(path, label=label,
                              collision_time_ms=args.collision_ms,
                              neuron_count=args.neuron_count,
                              population_indices=indices)
            for label, path in trace_specs
        }
        report = compare_traces(
            traces,
            baseline=args.baseline,
            populations=tuple(args.populations) if args.populations else DEFAULT_POPULATIONS,
        )
        if args.out is not None:
            write_json_report(report, args.out)
        else:
            print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
        return 0
    except ProbeError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
