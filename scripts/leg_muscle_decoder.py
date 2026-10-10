#!/usr/bin/env python3
"""Decode exact leg motor-neuron types into independent muscle-group drives.

The decoder reads only the CSR annotations and a dense per-neuron signal (or
an explicitly named sparse mapping).  Each leg/type pair is averaged and
normalized independently.  The 30 Hz full-scale anchor is an engineering
assumption recorded in ``config/leg_muscle_map.json``, not a fitted muscle
calibration.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    from .neuromuscular_adapter import AnnotationIndex, DEFAULT_CSR
except ImportError:  # direct script/test import with scripts/ on sys.path
    from neuromuscular_adapter import AnnotationIndex, DEFAULT_CSR


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = ROOT / "config" / "leg_muscle_map.json"


def _finite_nonnegative(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"neural signal value {value!r} is not numeric") from None
    if not math.isfinite(result):
        raise ValueError("neural signal values must be finite")
    if result < 0.0:
        raise ValueError("neural rates and spike counts must be non-negative")
    return result


def _mean(values: Any) -> float:
    """Mean a finite, non-negative population vector."""

    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("neural signal values must be numeric") from exc
    if array.size == 0:
        return 0.0
    if not np.isfinite(array).all():
        raise ValueError("neural signal values must be finite")
    if np.any(array < 0.0):
        raise ValueError("neural rates and spike counts must be non-negative")
    return float(array.mean())


class LegMuscleDecoder:
    """Read each exact type on each of the six legs as an independent drive.

    ``channels`` is a tuple/list of dictionaries with ``name``, ``leg``,
    ``group``, and exact CSR ``type_name`` fields.  By default, the 19 exact
    types in the checked map are expanded over LF/RF/LM/RM/LH/RH.  An
    ``AnnotationIndex`` can be injected for synthetic fixtures without
    reading the machine's CSR bundle.

    ``decode`` accepts either a dense vector in AnnotationIndex order or a
    mapping keyed by channel name, exact selector, exact type name, or body
    ID.  Mapping values can be a population mean/scalar or a vector aligned
    with the population (or full dense index).  Spike counts are converted
    to Hz with ``dt_s`` before normalization.
    """

    def __init__(self, csr_path: Path | str = DEFAULT_CSR,
                 map_path: Path | str = DEFAULT_MAP, *,
                 population_index: Any = None,
                 channels: Optional[Sequence[Mapping[str, Any]]] = None):
        self.map_path = Path(map_path)
        self.map_data = json.loads(self.map_path.read_text(encoding="utf-8"))
        self._rate_anchor_hz = float(self.map_data["rate_anchor"]["maximum_rate_hz"])
        if not math.isfinite(self._rate_anchor_hz) or self._rate_anchor_hz <= 0.0:
            raise ValueError("rate_anchor.maximum_rate_hz must be finite and positive")
        self.population_index = (
            population_index if population_index is not None else AnnotationIndex(csr_path)
        )
        self._n = int(getattr(self.population_index, "n", 0))
        self._ids = getattr(self.population_index, "ids", None)
        if self._ids is not None:
            self._ids = np.asarray(self._ids, dtype=np.int64)
            if self._n and self._ids.size != self._n:
                raise ValueError("AnnotationIndex ids and dense population length disagree")
            if not self._n:
                self._n = int(self._ids.size)
        self._legs = {str(item["name"]): dict(item) for item in self.map_data["legs"]}
        self._types = {str(item["type_name"]): dict(item)
                       for item in self.map_data["muscle_types"]}
        self.channels = tuple(
            dict(channel) for channel in (channels if channels is not None
                                          else self._default_channels())
        )
        self._channel_indices: dict[str, np.ndarray] = {}
        self._channel_specs: dict[str, str] = {}
        self._channel_defs: dict[str, dict[str, Any]] = {}
        self._leg_indices: dict[str, np.ndarray] = {}
        self._validate_and_resolve_channels()
        self._unmapped_by_leg = self._resolve_unmapped()
        self._last_input: dict[str, Any] = {"kind": None, "dt_s": None}

    def _default_channels(self) -> list[dict[str, str]]:
        return [
            {
                "name": f"{leg['name']}_{muscle['group']}",
                "leg": str(leg["name"]),
                "group": str(muscle["group"]),
                "type_name": str(muscle["type_name"]),
            }
            for leg in self.map_data["legs"]
            for muscle in self.map_data["muscle_types"]
        ]

    def _spec(self, leg_name: str, type_name: Optional[str] = None) -> str:
        leg = self._legs[leg_name]
        terms = [
            "superclass:vnc_motor",
        ]
        if type_name is not None:
            terms.append(f"type:{type_name}")
        # The population selector grammar applies /L or /R to the final
        # intersection term, so subclass must be last.
        terms.append(f"subclass:{leg['subclass']}/{leg['side']}")
        return "&".join(terms)

    def _resolve(self, spec: str) -> np.ndarray:
        result = np.asarray(self.population_index.resolve(spec), dtype=np.int64).reshape(-1)
        if result.size and (int(result.min()) < 0 or
                            (self._n and int(result.max()) >= self._n)):
            raise ValueError(f"population index returned out-of-range indices for {spec!r}")
        return np.unique(result)

    def _validate_and_resolve_channels(self) -> None:
        names: set[str] = set()
        for raw in self.channels:
            missing = {"name", "leg", "group", "type_name"} - set(raw)
            if missing:
                raise ValueError(f"channel is missing required fields: {sorted(missing)}")
            channel = {key: str(raw[key]) for key in ("name", "leg", "group", "type_name")}
            if not all(channel.values()):
                raise ValueError("channel name, leg, group, and type_name must be non-empty")
            if channel["name"] in names:
                raise ValueError(f"duplicate leg muscle channel name: {channel['name']}")
            names.add(channel["name"])
            if channel["leg"] not in self._legs:
                raise ValueError(f"unknown leg name {channel['leg']!r}")
            # The expression language uses '&', commas, and a terminal /L|/R
            # suffix as syntax.  Refuse these in raw names instead of letting
            # a display label silently broaden or alter an exact selector.
            if any(token in channel["type_name"] for token in ("&", ",")):
                raise ValueError("type_name must be a single exact CSR type name")
            spec = self._spec(channel["leg"], channel["type_name"])
            self._channel_defs[channel["name"]] = channel
            self._channel_specs[channel["name"]] = spec
            self._channel_indices[channel["name"]] = self._resolve(spec)

        for leg_name in self._legs:
            self._leg_indices[leg_name] = self._resolve(self._spec(leg_name))

    def _body_id(self, dense_index: int) -> Optional[int]:
        if self._ids is None or dense_index >= self._ids.size:
            return None
        return int(self._ids[dense_index])

    def _type_at(self, dense_index: int) -> Optional[str]:
        annotations = getattr(self.population_index, "annotations", {})
        tables = getattr(self.population_index, "tables", {})
        if not isinstance(annotations, Mapping) or not isinstance(tables, Mapping):
            return None
        type_codes = annotations.get("typeIdx")
        types = tables.get("types")
        if type_codes is None or types is None or dense_index >= len(type_codes):
            return None
        code = int(type_codes[dense_index])
        if code < 0 or code >= len(types):
            return None
        return str(types[code])

    def _resolve_unmapped(self) -> dict[str, list[dict[str, Any]]]:
        mapped: dict[str, set[int]] = {name: set() for name in self._legs}
        for name, indices in self._channel_indices.items():
            leg = self._channel_defs[name]["leg"]
            mapped[leg].update(map(int, indices.tolist()))
        result: dict[str, list[dict[str, Any]]] = {}
        for leg_name, population in self._leg_indices.items():
            missing = [int(index) for index in population if int(index) not in mapped[leg_name]]
            entries: list[dict[str, Any]] = []
            for index in missing:
                entry: dict[str, Any] = {
                    "type_name": self._type_at(index),
                    "reason": "no_exact_muscle_type_mapping",
                }
                body_id = self._body_id(index)
                if body_id is not None:
                    entry["body_id"] = body_id
                entries.append(entry)
            result[leg_name] = entries
        return result

    def _population_mean(self, signal: Any, channel_name: str, kind: str) -> float:
        indices = self._channel_indices[channel_name]
        if indices.size == 0:
            return 0.0
        if not isinstance(signal, Mapping):
            try:
                dense = np.asarray(signal, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{kind} must be a dense vector or mapping") from exc
            if dense.ndim != 1:
                raise ValueError(f"dense {kind} must be one-dimensional")
            if self._n and dense.size != self._n:
                raise ValueError(f"dense {kind} length {dense.size} does not match population {self._n}")
            if not self._n:
                raise ValueError("population index does not expose a dense population length")
            return _mean(dense[indices])

        # A named population mean takes precedence over per-body values.  The
        # most specific key wins, preserving independent side/leg readouts.
        definition = self._channel_defs[channel_name]
        spec = self._channel_specs[channel_name]
        keys = (channel_name, spec, f"type:{definition['type_name']}", definition["type_name"])
        for key in keys:
            if key in signal:
                value = signal[key]
                try:
                    array = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"invalid {kind} value for {key!r}") from exc
                if array.ndim == 0:
                    return _finite_nonnegative(array.item())
                array = array.reshape(-1)
                if array.size == self._n:
                    return _mean(array[indices])
                if array.size == indices.size:
                    return _mean(array)
                raise ValueError(
                    f"{kind} mapping value for {key!r} has {array.size} entries; "
                    f"expected {indices.size} selected or {self._n} dense entries"
                )

        # Sparse per-neuron mappings use real body IDs.  Omitted neurons are
        # silent for spike maps; they are also treated as zero in rate maps.
        if self._ids is None:
            return 0.0
        values = []
        for index in indices:
            body_id = int(self._ids[int(index)])
            if body_id in signal:
                values.append(signal[body_id])
            elif str(body_id) in signal:
                values.append(signal[str(body_id)])
            else:
                values.append(0.0)
        return _mean(values)

    def decode(self, rates_hz: Any = None, spikes: Any = None,
               dt_s: Optional[float] = None) -> dict[str, float]:
        """Return ``channel name -> excitation in [0, 1]`` for this sample."""

        if rates_hz is not None and spikes is not None:
            raise TypeError("pass rates_hz or spikes, not both")
        if rates_hz is None and spikes is None:
            self._last_input = {"kind": None, "dt_s": None}
            return {name: 0.0 for name in self._channel_defs}
        signal = spikes if spikes is not None else rates_hz
        if isinstance(signal, Mapping):
            # Validate even unused keys: a malformed value should not become
            # an apparently valid silent channel merely because no selector
            # happened to consume it on this tick.
            for key, value in signal.items():
                try:
                    _mean(value)
                except ValueError as exc:
                    raise ValueError(f"invalid neural signal at key {key!r}: {exc}") from exc
        else:
            try:
                dense = np.asarray(signal, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("neural signal must be numeric") from exc
            if dense.ndim != 1:
                raise ValueError("dense neural signal must be one-dimensional")
            if self._n and dense.size != self._n:
                raise ValueError(
                    f"dense neural signal length {dense.size} does not match population {self._n}"
                )
            if not self._n:
                raise ValueError("population index does not expose a dense population length")
            if not np.isfinite(dense).all():
                raise ValueError("neural signal values must be finite")
            if np.any(dense < 0.0):
                raise ValueError("neural rates and spike counts must be non-negative")
        if spikes is not None:
            try:
                duration = float(dt_s) if dt_s is not None else math.nan
            except (TypeError, ValueError, OverflowError):
                duration = math.nan
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("dt_s must be finite and positive when decoding spike counts")
            signal, kind = spikes, "spikes"
            scale = 1.0 / duration
        else:
            signal, kind = rates_hz, "rates_hz"
            duration = None
            scale = 1.0

        decoded: dict[str, float] = {}
        for name in self._channel_defs:
            mean_rate = self._population_mean(signal, name, kind) * scale
            decoded[name] = min(1.0, max(0.0, mean_rate / self._rate_anchor_hz))
        self._last_input = {"kind": kind, "dt_s": duration}
        return decoded

    def provenance(self) -> dict[str, Any]:
        """Return JSON-friendly source, selector, counts, IDs, and unmapped MNs."""

        channel_info: dict[str, Any] = {}
        body_ids_available = self._ids is not None
        for name, channel in self._channel_defs.items():
            indices = self._channel_indices[name]
            type_info = self._types.get(channel["type_name"], {})
            ids = []
            if self._ids is not None:
                ids = [int(self._ids[int(index)]) for index in indices]
            channel_info[name] = {
                **channel,
                "selector": self._channel_specs[name],
                "resolved_count": int(indices.size),
                "body_ids": ids,
                "body_ids_available": body_ids_available,
                "rate_anchor_hz": self._rate_anchor_hz,
                "mapping_hypothesis": type_info.get(
                    "mapping_hypothesis",
                    "Custom exact type mapping supplied by caller; anatomical target is unvalidated.",
                ),
            }
        unmapped = []
        for leg_name, entries in self._unmapped_by_leg.items():
            for item in entries:
                unmapped.append({"leg": leg_name, **item})
        counts = {name: int(len(items)) for name, items in self._unmapped_by_leg.items()}
        return {
            "schema_version": 1,
            "source": deepcopy(self.map_data["source"]),
            "rate_anchor": deepcopy(self.map_data["rate_anchor"]),
            "input": dict(self._last_input),
            "channels": channel_info,
            "resolved_counts": {
                name: info["resolved_count"] for name, info in channel_info.items()
            },
            "unmapped_mn": unmapped,
            "unmapped_counts_by_leg": counts,
            "unmapped_count": len(unmapped),
            "body_ids_available": body_ids_available,
            "mapping_policy": deepcopy(self.map_data["mapping_policy"]),
        }


__all__ = ["LegMuscleDecoder"]
