#!/usr/bin/env python3
"""Stream and compare strict MaleCNS escape-run JSONL artifacts.

The strict escape runner writes one large JSON object per telemetry frame.  A
full run can be hundreds of megabytes, so this module intentionally keeps one
decoded row plus a small set of scalar accumulators in memory.  It extracts
trace-level timing and integrity evidence; it does not turn temporal ordering
into a biological or causal claim.

Examples
--------
Summarise one run (a run directory or ``telemetry.jsonl`` is accepted)::

    python scripts/summarize_escape_run.py RUN_DIR \
        --output /data/artifacts/by-project/malecns-gpu/engineering/run.json \
        --markdown /data/artifacts/by-project/malecns-gpu/engineering/run.md

Compare the usual moving/static/sham block::

    python scripts/summarize_escape_run.py --compare \
        --moving moving/ --static static/ --sham sham/ --output block.json

The JSON report is compact and suitable for downstream ledgers.  The markdown
report is deliberately explicit that a digital-simulation trace is not
biological validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = "escape-run-summary-1"
COMPARISON_SCHEMA_VERSION = "escape-run-comparison-1"
NOT_VALIDATED = "NOT_VALIDATED"
EVENT_NAMES = ("LC4", "LPLC2", "DNp01", "TTMn", "wing_motor", "liftoff")
POPULATION_EVENTS = ("LC4", "LPLC2", "DNp01", "TTMn", "wing_motor")
DEFAULT_EVENT_THRESHOLD = 1.0e-12
MAX_ERROR_SAMPLES = 20
MAX_VALUE_SAMPLES = 8


class EscapeSummaryError(ValueError):
    """Raised when a strict-run artifact cannot be read safely."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    """Return a finite real number, excluding booleans."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _scalar(value: Any) -> Any:
    """Make a JSON scalar safe for compact report metadata."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    number = _number(value)
    if number is not None:
        return int(number) if number.is_integer() else number
    return str(value)


def _scalar_projection(mapping: Mapping[str, Any], fields: str) -> dict[str, Any]:
    return {key: _scalar(mapping.get(key)) for key in fields.split() if key in mapping}


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_get(value: Any, path: Sequence[str]) -> Any:
    current = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_path(value: Any, paths: Iterable[Sequence[str]]) -> Any:
    for path in paths:
        candidate = _path_get(value, path)
        if candidate is not None:
            return candidate
    return None


def _normalise_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _safe_rel_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _resolve_input(source: str | Path) -> tuple[Path, Path | None]:
    """Resolve a run directory or JSONL path without reading its contents."""

    path = Path(source).expanduser()
    if path.is_dir():
        telemetry = path / "telemetry.jsonl"
        if not telemetry.is_file():
            candidates = sorted(path.glob("*.jsonl"))
            if not candidates:
                raise EscapeSummaryError(f"run directory has no JSONL telemetry: {path}")
            telemetry = candidates[0]
        manifest = path / "manifest.json"
    else:
        telemetry = path
        manifest = path.parent / "manifest.json"
    if not telemetry.is_file():
        raise EscapeSummaryError(f"telemetry JSONL does not exist: {telemetry}")
    return telemetry, manifest if manifest.is_file() else None


def _load_manifest(path: Path | None) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if path is None:
        return {}, {"present": False, "path": None, "parse_error": None}
    meta: dict[str, Any] = {"present": True, "path": _safe_rel_path(path), "parse_error": None}
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("manifest root is not an object")
        meta["sha256"] = hashlib.sha256(raw).hexdigest()
        return payload, meta
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        meta["parse_error"] = str(exc)
        return {}, meta


def _condition_from(manifest: Mapping[str, Any], row: Mapping[str, Any] | None, path: Path) -> str | None:
    candidates: list[Any] = []
    components = _mapping(manifest.get("components"))
    stimulus = _mapping(components.get("physical_stimulus"))
    effective = _mapping(manifest.get("effective_execution"))
    requested = _mapping(manifest.get("requested_execution"))
    candidates.extend(
        (
            stimulus.get("condition"),
            stimulus.get("name"),
            _mapping(effective.get("physical_stimulus")).get("resolved_name"),
            requested.get("stimulus_condition"),
            requested.get("stimulus_name"),
        )
    )
    if row:
        candidates.extend((row.get("stimulus_condition"), row.get("stimulus_name")))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    lowered = path.parent.name.lower()
    for marker in ("looming", "moving", "dynamic", "static", "sham"):
        if marker in lowered:
            return marker
    return None


def _manifest_compact(manifest: Mapping[str, Any], condition: str | None) -> dict[str, Any]:
    """Retain manifest evidence without copying large nested component maps."""

    execution = _mapping(manifest.get("effective_execution"))
    requested = _mapping(manifest.get("requested_execution"))
    provenance = _mapping(manifest.get("provenance"))
    policy = _mapping(manifest.get("autonomy_policy"))
    limits = _mapping(manifest.get("resource_limits"))
    usage = _mapping(manifest.get("resource_usage"))
    cuda_usage = _mapping(usage.get("cuda"))
    stimulus = _mapping(_mapping(manifest.get("components")).get("physical_stimulus"))
    return {
        "path": None,  # filled by caller
        "schema_version": _scalar(manifest.get("schema_version")),
        "execution_policy": _scalar(manifest.get("execution_policy") or _mapping(manifest.get("autonomy_policy")).get("name")),
        "seed": _scalar(manifest.get("seed")),
        "condition": condition,
        "scenario_names": [_scalar(v) for v in manifest.get("scenario_names", [])] if isinstance(manifest.get("scenario_names"), list) else [],
        "requested": _scalar_projection(requested, "mode strict neural_backend neural_device vision_device neural_gain stimulus_condition stimulus_name"),
        "effective": _scalar_projection(execution, "neural_backend neural_device vision_device neural_gain neural_window_api visual_control_readout flight_state_drive flight_body_stabilizer"),
        "stimulus": _scalar_projection(stimulus, "name condition interaction visual_only non_interacting state_for_control planned_contact_deadline_s"),
        "provenance": _scalar_projection(provenance, "evidence_class execution_policy world vision neural neuromuscular readout body sensory_feedback autopilot flight_state_drive flight_body_stabilizer visual_control_readout"),
        "autonomy_policy": _scalar_projection(policy, "name strict control_authority stimulus_labels stimulus_direction future_state stimulus_interaction gpu_allowed fail_closed"),
        "resource_limits": _scalar_projection(limits, "cgroup_memory_max_bytes torch_allocator_max_vram_gib cns_dt_s flyvis_frame_dt_s neural_substeps_per_frame"),
        "resource_usage": {
            **_scalar_projection(usage, "peak_host_rss_bytes current_host_rss_bytes"),
            **({"cuda": _scalar_projection(cuda_usage, "device total_bytes peak_allocated_bytes peak_reserved_bytes current_allocated_bytes current_reserved_bytes")} if cuda_usage else {}),
        },
    }


def _time_from_row(row: Mapping[str, Any]) -> float | None:
    value = _first_path(row, (("time_s",), ("timestamp", "t_s"), ("frame_time_s",), ("t_s",), ("t",)))
    return _number(value)


def _interval_from_row(row: Mapping[str, Any], time_s: float | None) -> tuple[float | None, float | None]:
    start = _first_path(
        row,
        (
            ("neural_window", "start_s"),
            ("neural", "named_outputs", "window_start_timestamp", "t_s"),
            ("interval_start_s",),
            ("start_s",),
        ),
    )
    end = _first_path(
        row,
        (
            ("neural_window", "end_s"),
            ("neural", "named_outputs", "window_end_timestamp", "t_s"),
            ("interval_end_s",),
        ),
    )
    start_n = _number(start)
    end_n = _number(end)
    dt = _number(row.get("dt_s"))
    if start_n is None and time_s is not None and dt is not None:
        start_n = time_s - dt
    if end_n is None:
        end_n = time_s
    if start_n is None:
        start_n = end_n
    if end_n is None:
        end_n = start_n
    if start_n is not None and end_n is not None and end_n < start_n:
        start_n, end_n = end_n, start_n
    return start_n, end_n


def _row_dt(row: Mapping[str, Any], start: float | None, end: float | None) -> float:
    for value in (row.get("dt_s"), _path_get(row, ("neural_window", "dt_s")), _path_get(row, ("neural_window", "window_dt_s"))):
        number = _number(value)
        if number is not None and number >= 0:
            return number
    if start is not None and end is not None and end >= start:
        return end - start
    return 0.0


def _row_maps(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    neural = _mapping(row.get("neural"))
    named = _first_mapping(neural.get("named_outputs"), row.get("named_outputs"))
    rates = _first_mapping(
        named.get("population_rates_hz"), neural.get("population_rates_hz"),
        neural.get("rates_hz"), row.get("population_rates_hz"), row.get("rates_hz"),
    )
    counts = _first_mapping(
        named.get("population_spike_counts"), neural.get("population_spike_counts"),
        neural.get("spike_counts"), row.get("population_spike_counts"), row.get("spike_counts"),
    )
    spikes = _mapping(neural.get("spikes"))
    return named, rates, counts, spikes


def _first_mapping(*values: Any) -> Mapping[str, Any]:
    return next((value for value in values if isinstance(value, Mapping) and value), {})


_RowMaps = tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]


_ALIASES: dict[str, tuple[str, ...]] = {
    "LC4": ("LC4", "lc4"),
    "LPLC2": ("LPLC2", "lplc2"),
    "DNp01": ("DNp01", "dnp01", "DNp01_activity", "dnp01_activity"),
    "TTMn": ("TTMn", "ttmn", "TTMn_activity", "ttmn_activity"),
    "wing_motor": (
        "wing_motor",
        "wingMotor",
        "wing_motor_activity",
        "superclass:vnc_motor&subclass:wm",
        "wm",
    ),
}
_POPULATION_ALIAS_KEYS = frozenset(alias for event in POPULATION_EVENTS for alias in _ALIASES[event])


def _lookup(mapping: Mapping[str, Any], aliases: Sequence[str]) -> tuple[float | None, str | None]:
    for key in aliases:
        if key in mapping:
            number = _number(mapping[key])
            if number is not None:
                return number, key
    # A few fixture producers vary only case.  Keep this fallback after exact
    # lookup so strict names such as TTMn remain preferred.
    lower = {_normalise_name(key): key for key in mapping}
    for alias in aliases:
        key = lower.get(_normalise_name(alias))
        if key is not None:
            number = _number(mapping[key])
            if number is not None:
                return number, str(key)
    return None, None


def _event_value(
    row: Mapping[str, Any], event: str, sources: _RowMaps | None = None,
) -> tuple[float | None, str | None, str | None]:
    """Find one scalar event value and retain its source/semantics."""

    named, rates, counts, spikes = sources if sources is not None else _row_maps(row)
    neural = _mapping(row.get("neural"))
    nm_out = _mapping(row.get("nm_out"))
    channels = _mapping(nm_out.get("channels"))
    decision = _mapping(row.get("jump_decision_telemetry"))

    if event in POPULATION_EVENTS:
        aliases = _ALIASES[event]
        # ``wing_motor`` is a physical neuromuscular channel when the strict
        # runner emits it.  Prefer that normalized command over the upstream
        # wm population rate for the chain timing/value summary; population
        # spike totals are still collected independently below.
        if event == "wing_motor":
            value, key = _lookup(channels, ("wing_motor", "wingMotor", "wing_motor_activity"))
            if value is not None:
                return value, f"nm_out.channels.{key}", "channel"
        for mapping, source, kind in (
            (named, "neural.named_outputs", "rate_or_activity"),
            (rates, "neural.named_outputs.population_rates_hz", "rate"),
            (_mapping(neural.get("rates_hz")), "neural.rates_hz", "rate"),
            (counts, "neural.named_outputs.population_spike_counts", "spike_count"),
            (spikes, "neural.spikes", "spike_count"),
        ):
            value, key = _lookup(mapping, aliases)
            if value is not None:
                return value, f"{source}.{key}", kind
        # Explicit motor/decision readouts are useful when named population
        # outputs were disabled in a compact fixture.
        if event == "DNp01":
            value, key = _lookup(decision, ("dnp01_activity", "DNp01", "dnp01"))
            if value is not None:
                return value, f"jump_decision_telemetry.{key}", "activity"
            value, key = _lookup(channels, ("jump", "jump_event", "jumpEvent"))
            if value is not None:
                return value, f"nm_out.channels.{key}", "event"
            value, key = _lookup(row, ("DNp01", "dnp01", "dnp01_activity"))
            if value is not None:
                return value, f"row.{key}", "activity"
        if event == "TTMn":
            value, key = _lookup(decision, ("ttmn_activity", "TTMn", "ttmn"))
            if value is not None:
                return value, f"jump_decision_telemetry.{key}", "activity"
            value, key = _lookup(channels, ("ttmn_activity", "TTMn", "TTMn_activity", "ttmn"))
            if value is not None:
                return value, f"nm_out.channels.{key}", "activity"
            value, key = _lookup(row, ("TTMn", "ttmn", "ttmn_activity"))
            if value is not None:
                return value, f"row.{key}", "activity"
        if event == "wing_motor":
            value, key = _lookup(nm_out, ("wing_motor", "wingMotor", "wing_motor_activity"))
            if value is not None:
                return value, f"nm_out.{key}", "channel"
        value, key = _lookup(row, aliases)
        if value is not None:
            return value, f"row.{key}", "activity"
    return None, None, None


def _population_count(
    row: Mapping[str, Any], event: str, sources: _RowMaps | None = None,
) -> float | None:
    """Resolve a population spike count without double-counting aliases."""

    _, _, counts, spikes = sources if sources is not None else _row_maps(row)
    aliases = _ALIASES[event]
    value, _ = _lookup(counts, aliases)
    if value is None:
        value, _ = _lookup(spikes, aliases)
    return value


def _new_event() -> dict[str, Any]:
    return {
        "observed": False,
        "first_nonzero_s": None,
        "first_nonzero_time_s": None,
        "first_nonzero_interval_s": None,
        "first_nonzero_frame_index": None,
        "first_nonzero_value": None,
        "first_nonzero_source": None,
        "first_nonzero_after_stimulus_s": None,
        "first_nonzero_after_stimulus_interval_s": None,
        "peak_value": None,
        "peak_time_s": None,
        "peak_interval_s": None,
        "last_value": None,
        "sample_count": 0,
        "nonzero_frames": 0,
        "sum_value": 0.0,
        "integral_value_s": 0.0,
        "total_spikes": 0.0,
        "spike_count_observed": False,
        "sources": set(),
    }


def _clean_event(event: MutableMapping[str, Any]) -> dict[str, Any]:
    output = dict(event)
    output["sources"] = sorted(str(source) for source in output.get("sources", set()))
    for key in ("sum_value", "integral_value_s", "total_spikes"):
        value = _number(output.get(key))
        output[key] = 0.0 if value is None else (int(value) if value.is_integer() else value)
    return output


def _append_observed_value(
    stats: MutableMapping[str, Any],
    value: float,
    source: str,
    kind: str | None,
    *,
    time_s: float | None,
    start_s: float | None,
    end_s: float | None,
    frame_index: Any,
    stimulus_onset_s: float,
    threshold: float,
    dt_s: float,
) -> None:
    stats["sample_count"] += 1
    stats["last_value"] = _scalar(value)
    stats["sum_value"] += value
    stats["integral_value_s"] += value * dt_s
    stats["sources"].add(source)
    if value > (stats.get("peak_value") if stats.get("peak_value") is not None else -math.inf):
        stats["peak_value"] = _scalar(value)
        stats["peak_time_s"] = _scalar(time_s)
        stats["peak_interval_s"] = [_scalar(start_s), _scalar(end_s)]
    if value <= threshold:
        return
    stats["observed"] = True
    stats["nonzero_frames"] += 1
    if stats["first_nonzero_s"] is None:
        stats["first_nonzero_s"] = _scalar(start_s if start_s is not None else time_s)
        stats["first_nonzero_time_s"] = _scalar(time_s)
        stats["first_nonzero_interval_s"] = [_scalar(start_s), _scalar(end_s)]
        stats["first_nonzero_frame_index"] = _scalar(frame_index)
        stats["first_nonzero_value"] = _scalar(value)
        stats["first_nonzero_source"] = source
    event_time = start_s if start_s is not None else time_s
    # If an aggregate window straddles stimulus onset, retain the declared
    # interval and anchor the post-onset scalar at its onset boundary.  This
    # avoids claiming sub-window precision while still reporting the event.
    if event_time is not None and end_s is not None and event_time < stimulus_onset_s <= end_s:
        event_time = stimulus_onset_s
    if event_time is not None and event_time >= stimulus_onset_s and stats["first_nonzero_after_stimulus_s"] is None:
        stats["first_nonzero_after_stimulus_s"] = _scalar(event_time)
        stats["first_nonzero_after_stimulus_interval_s"] = [_scalar(start_s), _scalar(end_s)]


def _body_state(row: Mapping[str, Any], time_s: float | None, frame_index: Any) -> dict[str, Any]:
    body = _mapping(row.get("body"))
    def pick(*paths: Sequence[str]) -> Any:
        return _first_path(row, paths) if paths else None

    position = pick(("position_m",), ("body", "position_m"), ("body_state", "position_m"))
    velocity = pick(("velocity_mps",), ("body", "velocity_mps"), ("body_state", "velocity_mps"))
    orientation = pick(("orientation_rad",), ("body", "orientation_rad"), ("body_state", "orientation_rad"))
    angular_velocity = pick(("angular_velocity_rad_s",), ("body", "angular_velocity_rad_s"), ("body_state", "angular_velocity_rad_s"))
    def vector(value: Any) -> list[Any] | None:
        if not isinstance(value, (list, tuple)):
            return None
        return [_scalar(v) for v in value]
    return {
        "time_s": _scalar(time_s),
        "frame_index": _scalar(frame_index),
        "position_m": vector(position),
        "velocity_mps": vector(velocity),
        "orientation_rad": vector(orientation),
        "angular_velocity_rad_s": vector(angular_velocity),
        "grounded": row.get("grounded", body.get("grounded")),
        "contact": row.get("contact", body.get("contact")),
        "target_contact": row.get("target_contact", body.get("target_contact")),
        "flight_mode": _scalar(row.get("flight_mode", body.get("flight_mode"))),
        "controller_mode": _scalar(row.get("controller_mode")),
        "collision_count": _scalar(row.get("collision_count", _path_get(row, ("diagnostics", "counters", "collision_count")))),
        "source": _scalar(body.get("source") or row.get("source")),
    }


def _provenance_paths(row: Mapping[str, Any]) -> dict[str, Any]:
    frame = _mapping(row.get("frame_provenance"))
    provenance = _mapping(row.get("provenance"))
    nm_provenance = _mapping(_mapping(row.get("nm_out")).get("provenance"))
    return {
        "execution_policy": row.get("execution_policy"),
        "evidence_class": provenance.get("evidence_class"),
        "neural": provenance.get("neural"),
        "neuromuscular": provenance.get("neuromuscular"),
        "body": provenance.get("body"),
        "vision": provenance.get("vision"),
        "frame_policy": frame.get("policy"),
        "control_authority": frame.get("control_authority"),
        "autopilot_overlay": frame.get("autopilot_overlay"),
        "flight_state_drive": frame.get("flight_state_drive", provenance.get("flight_state_drive")),
        "flight_body_stabilizer": frame.get("flight_body_stabilizer", provenance.get("flight_body_stabilizer")),
        "mission_goal_phase": frame.get("mission_goal_phase"),
        "stimulus_label_control": frame.get("stimulus_label_control"),
        "stimulus_direction_control": frame.get("stimulus_direction_control"),
        "future_state_control": frame.get("future_state_control"),
        "direct_neuromuscular_to_dynamics": frame.get("direct_neuromuscular_to_dynamics"),
        "stimulus_interaction": frame.get("stimulus_interaction", provenance.get("stimulus_interaction")),
        "map_sha256": nm_provenance.get("map_sha256"),
        "dataset": nm_provenance.get("dataset"),
        "calibration_id": nm_provenance.get("calibration_id"),
    }


def _prohibited_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "false", "0", "off", "disabled", "prohibited", "record_only", "record-only", "not_used"}:
            return False
        if "prohibit" in lowered or "record_only" in lowered or "record-only" in lowered:
            return False
    return True


def _resource_integrity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    limits = _mapping(manifest.get("resource_limits"))
    usage = _mapping(manifest.get("resource_usage"))
    cuda = _mapping(usage.get("cuda"))
    host_peak = _number(usage.get("peak_host_rss_bytes"))
    host_limit = _number(limits.get("cgroup_memory_max_bytes"))
    vram_peak = _number(cuda.get("peak_reserved_bytes"))
    vram_limit_gib = _number(limits.get("torch_allocator_max_vram_gib"))
    vram_limit = vram_limit_gib * (1024 ** 3) if vram_limit_gib is not None else None
    host_ok = host_peak is not None and host_limit is not None and host_peak <= host_limit
    vram_ok = vram_peak is not None and vram_limit is not None and vram_peak <= vram_limit
    if not limits and not usage:
        integrity_status = "INCOMPLETE"
    elif host_ok is False or vram_ok is False:
        integrity_status = "FAIL"
    elif host_ok is None or vram_ok is None:
        integrity_status = "INCOMPLETE"
    else:
        integrity_status = "PASS"
    return {
        "limits": _scalar_projection(limits, "cgroup_memory_max_bytes torch_allocator_max_vram_gib cns_dt_s flyvis_frame_dt_s neural_substeps_per_frame"),
        "usage": {
            "peak_host_rss_bytes": _scalar(host_peak),
            "current_host_rss_bytes": _scalar(_number(usage.get("current_host_rss_bytes"))),
            "cuda_device": _scalar(cuda.get("device")),
            "cuda_peak_allocated_bytes": _scalar(_number(cuda.get("peak_allocated_bytes"))),
            "cuda_peak_reserved_bytes": _scalar(vram_peak),
        },
        "host_peak_within_declared_limit": host_ok if host_peak is not None and host_limit is not None else None,
        "cuda_peak_within_declared_limit": vram_ok if vram_peak is not None and vram_limit is not None else None,
        "declared_under_5gb": host_limit <= 5 * 1024 ** 3 if host_limit is not None else None,
        "integrity_status": integrity_status,
    }


def _provenance_integrity(
    manifest: Mapping[str, Any],
    observed: Mapping[str, set[str]],
    forbidden: Sequence[str],
    rows_seen: int,
) -> dict[str, Any]:
    manifest_provenance = _mapping(manifest.get("provenance"))
    policy = _mapping(manifest.get("autonomy_policy"))
    strict_declared = bool(policy.get("strict") or manifest.get("execution_policy") == "strict-free-behavior-escape-experiment")
    compact_observed = {key: sorted(values)[:MAX_VALUE_SAMPLES] for key, values in observed.items() if values}
    consistency = all(len(values) <= 1 for values in observed.values())
    expected = {
        "evidence_class": "digital_simulation",
        "autopilot_overlay": "prohibited",
        "flight_state_drive": "prohibited",
        "flight_body_stabilizer": "prohibited",
    }
    expected_matches: dict[str, bool | None] = {}
    for key, wanted in expected.items():
        values = observed.get(key, set())
        # ``observed`` stores canonical JSON strings to keep heterogeneous
        # provenance values comparable.  Compare using the same encoding.
        expected_matches[key] = (json.dumps(wanted, sort_keys=True, separators=(",", ":")) in values) if values else None
    if not manifest:
        status = "INCOMPLETE"
    elif forbidden or not consistency or any(value is False for value in expected_matches.values()):
        status = "FAIL"
    elif rows_seen == 0 or any(value is None for value in expected_matches.values()):
        status = "INCOMPLETE"
    else:
        status = "PASS"
    return {
        "manifest_present": bool(manifest),
        "manifest_evidence_class": _scalar(manifest_provenance.get("evidence_class")),
        "manifest_provenance_sha256": _json_hash(manifest_provenance) if manifest_provenance else None,
        "strict_declared": strict_declared,
        "rows_with_observed_provenance": rows_seen,
        "observed_values": compact_observed,
        "stable_selected_fields": consistency,
        "expected_values": expected_matches,
        "forbidden_or_active_fields": list(forbidden),
        "status": status,
    }


def _event_delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float | None:
    a = _number(left.get(key))
    b = _number(right.get(key))
    if a is None or b is None:
        return None
    return a - b


def _vector_delta(left: Any, right: Any) -> list[float] | None:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
        return None
    values: list[float] = []
    for a, b in zip(left, right):
        a_n, b_n = _number(a), _number(b)
        if a_n is None or b_n is None:
            return None
        values.append(a_n - b_n)
    return values


def summarize_jsonl(
    source: str | Path,
    manifest_path: str | Path | None = None,
    *,
    compute_hash: bool = True,
    event_threshold: float = DEFAULT_EVENT_THRESHOLD,
) -> dict[str, Any]:
    """Stream one telemetry JSONL file and return a compact JSON-compatible map.

    ``compute_hash`` only controls the input SHA-256.  Parsing remains a
    single binary line iteration in either mode.  Invalid lines are counted
    and retained as integrity evidence rather than aborting a long run.
    """

    telemetry, discovered_manifest = _resolve_input(source)
    manifest_file = Path(manifest_path).expanduser() if manifest_path is not None else discovered_manifest
    manifest, manifest_meta = _load_manifest(manifest_file)
    stats = {event: _new_event() for event in EVENT_NAMES}
    population_spike_totals: dict[str, float] = {}
    population_rate_sums: dict[str, float] = {}
    population_nonzero_frames: Counter[str] = Counter()
    schema_versions: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    stimulus_names: Counter[str] = Counter()
    stimulus_conditions: Counter[str] = Counter()
    warnings_count = 0
    malformed: list[dict[str, Any]] = []
    provenance_values: dict[str, set[str]] = {}
    forbidden_fields: set[str] = set()
    line_count = 0
    valid_row_count = 0
    empty_lines = 0
    byte_count = 0
    input_hash = hashlib.sha256() if compute_hash else None
    first_time: float | None = None
    last_time: float | None = None
    previous_time: float | None = None
    monotonic_time = True
    frame_indices_monotonic = True
    previous_frame: float | None = None
    stimulus_onset_s = _number(_first_path(manifest, (("stimulus_onset_s",), ("experiment", "stimulus_onset_s"), ("experiment", "visual_baseline", "duration_s"), ("components", "experiment", "stimulus_onset_s"), ("components", "experiment", "visual_baseline", "duration_s"), ("config", "experiment", "visual_baseline", "duration_s"), ("timing", "stimulus_onset_s"))))
    if stimulus_onset_s is None:
        stimulus_onset_s = 0.0
    last_row: Mapping[str, Any] | None = None
    first_row: Mapping[str, Any] | None = None
    previous_grounded: bool | None = None
    previous_contact: bool | None = None
    liftoff_recorded = False
    first_liftoff: tuple[float | None, float | None, float | None, Any, str] | None = None
    condition: str | None = _condition_from(manifest, None, telemetry)
    neural_spikes_total = 0.0
    neural_spike_count_observed = False
    active_neurons_max: float | None = None
    active_neurons_last: float | None = None
    collision_max: float | None = None

    try:
        handle = telemetry.open("rb")
    except OSError as exc:
        raise EscapeSummaryError(f"cannot open telemetry JSONL: {telemetry}: {exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            byte_count += len(raw_line)
            if input_hash is not None:
                input_hash.update(raw_line)
            if not raw_line.strip():
                empty_lines += 1
                continue
            line_count += 1
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
                if len(malformed) < MAX_ERROR_SAMPLES:
                    malformed.append({"line": line_number, "error": str(exc)})
                continue
            if not isinstance(row, Mapping):
                if len(malformed) < MAX_ERROR_SAMPLES:
                    malformed.append({"line": line_number, "error": "JSON value is not an object"})
                continue
            valid_row_count += 1
            if first_row is None:
                first_row = row
            last_row = row
            schema = row.get("schema_version")
            if schema is not None:
                schema_versions[str(schema)] += 1
            scenario = row.get("scenario")
            if scenario is not None:
                scenarios[str(scenario)] += 1
            stimulus_name = row.get("stimulus_name")
            if stimulus_name is not None:
                stimulus_names[str(stimulus_name)] += 1
            stimulus_condition = row.get("stimulus_condition")
            if stimulus_condition is not None:
                stimulus_conditions[str(stimulus_condition)] += 1
            condition = condition or _condition_from(manifest, row, telemetry)
            time_s = _time_from_row(row)
            start_s, end_s = _interval_from_row(row, time_s)
            dt_s = _row_dt(row, start_s, end_s)
            if time_s is not None:
                if first_time is None:
                    first_time = time_s
                if previous_time is not None and time_s < previous_time:
                    monotonic_time = False
                previous_time = time_s
                last_time = time_s
            frame_index = row.get("frame_index", row.get("step_index"))
            frame_number = _number(frame_index)
            if frame_number is not None:
                if previous_frame is not None and frame_number < previous_frame:
                    frame_indices_monotonic = False
                previous_frame = frame_number
            sources = _row_maps(row)
            named, rates, counts, _ = sources
            neural = _mapping(row.get("neural"))
            total_spikes = _number(named.get("spike_count"))
            if total_spikes is None:
                total_spikes = _number(neural.get("spike_count"))
            if total_spikes is not None:
                neural_spikes_total += total_spikes
                neural_spike_count_observed = True
            active = _number(named.get("active_neurons"))
            if active is None:
                active = _number(neural.get("active_neurons"))
            if active is not None:
                active_neurons_last = active
                active_neurons_max = active if active_neurons_max is None else max(active_neurons_max, active)
            for key, value in counts.items():
                number = _number(value)
                if number is not None:
                    population_spike_totals[str(key)] = population_spike_totals.get(str(key), 0.0) + number
                    if number > event_threshold:
                        population_nonzero_frames[str(key)] += 1
            for key, value in rates.items():
                number = _number(value)
                if number is not None:
                    population_rate_sums[str(key)] = population_rate_sums.get(str(key), 0.0) + number
            warning_rows = _path_get(row, ("diagnostics", "warnings"))
            if isinstance(warning_rows, list):
                warnings_count += len(warning_rows)
            collision = _number(row.get("collision_count", _path_get(row, ("diagnostics", "counters", "collision_count"))))
            if collision is not None:
                collision_max = collision if collision_max is None else max(collision_max, collision)
            selected_provenance = _provenance_paths(row)
            for key, value in selected_provenance.items():
                if value is not None:
                    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                    provenance_values.setdefault(key, set())
                    if len(provenance_values[key]) < MAX_VALUE_SAMPLES:
                        provenance_values[key].add(rendered)
            for key in ("autopilot_overlay", "flight_state_drive", "flight_body_stabilizer", "mission_goal_phase"):
                if _prohibited_value(selected_provenance.get(key)):
                    forbidden_fields.add(key)
            for event in POPULATION_EVENTS:
                value, source_name, kind = _event_value(row, event, sources)
                if value is not None and source_name is not None:
                    _append_observed_value(
                        stats[event], value, source_name, kind,
                        time_s=time_s, start_s=start_s, end_s=end_s,
                        frame_index=frame_index, stimulus_onset_s=stimulus_onset_s,
                        threshold=event_threshold, dt_s=dt_s,
                    )
                    count = _population_count(row, event, sources)
                    if count is not None:
                        stats[event]["total_spikes"] += count
                        stats[event]["spike_count_observed"] = True
            grounded_value = row.get("grounded")
            grounded = grounded_value if isinstance(grounded_value, bool) else None
            contact_value = row.get("contact")
            contact = contact_value if isinstance(contact_value, bool) else None
            if grounded is not None and previous_grounded is True and grounded is False and not liftoff_recorded:
                liftoff_recorded = True
                first_liftoff = (start_s, end_s, time_s, frame_index, "grounded_transition")
            elif grounded is None and contact is not None and previous_contact is True and contact is False and not liftoff_recorded:
                liftoff_recorded = True
                first_liftoff = (start_s, end_s, time_s, frame_index, "contact_transition")
            previous_grounded = grounded if grounded is not None else previous_grounded
            previous_contact = contact if contact is not None else previous_contact
    if first_liftoff is not None:
        start_s, end_s, time_s, frame_index, source_name = first_liftoff
        liftoff = stats["liftoff"]
        _append_observed_value(
            liftoff, 1.0, source_name, "transition",
            time_s=time_s, start_s=start_s, end_s=end_s,
            frame_index=frame_index, stimulus_onset_s=stimulus_onset_s,
            threshold=event_threshold, dt_s=_row_dt({}, start_s, end_s),
        )
    elif first_row is not None and first_row.get("grounded") is False:
        row_time = _time_from_row(first_row)
        start_s, end_s = _interval_from_row(first_row, row_time)
        _append_observed_value(
            stats["liftoff"], 1.0, "initial_airborne", "state",
            time_s=row_time, start_s=start_s, end_s=end_s,
            frame_index=first_row.get("frame_index"), stimulus_onset_s=stimulus_onset_s,
            threshold=event_threshold, dt_s=_row_dt(first_row, start_s, end_s),
        )
    cleaned_events = {event: _clean_event(stats[event]) for event in EVENT_NAMES}
    for event in EVENT_NAMES:
        # The event's first nonzero and peak values are scalar evidence.  Do
        # not expose internal Counter/set objects in the JSON report.
        cleaned_events[event]["threshold"] = event_threshold
    first_after = {event: cleaned_events[event].get("first_nonzero_after_stimulus_s") for event in EVENT_NAMES}
    observed_order = [event for event in EVENT_NAMES if first_after[event] is not None]
    observed_order.sort(key=lambda event: float(first_after[event]))
    chain_times = {event: first_after[event] for event in EVENT_NAMES}
    chain_complete = all(chain_times[event] is not None for event in EVENT_NAMES)
    observed_positions = [EVENT_NAMES.index(event) for event in observed_order]
    chain_order_matches = observed_positions == sorted(observed_positions)
    manifest_compact = _manifest_compact(manifest, condition)
    manifest_compact["path"] = manifest_meta.get("path")
    resources = _resource_integrity(manifest) if manifest else {
        "limits": {}, "usage": {}, "host_peak_within_declared_limit": None,
        "cuda_peak_within_declared_limit": None, "declared_under_5gb": None,
        "integrity_status": "INCOMPLETE",
    }
    final_state = _body_state(last_row, last_time, last_row.get("frame_index") if last_row else None) if last_row else None
    input_stat = telemetry.stat()
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": NOT_VALIDATED,
        "biological_validation": False,
        "interpretation": "Temporal ordering and engineering endpoints only; this report makes no biological or causal inference.",
        "input": {
            "path": _safe_rel_path(telemetry),
            "size_bytes": input_stat.st_size,
            "bytes_streamed": byte_count,
            "sha256": input_hash.hexdigest() if input_hash is not None else None,
            "manifest": manifest_meta,
        },
        "run": {
            "condition": condition,
            "scenario": scenarios.most_common(1)[0][0] if scenarios else None,
            "stimulus_name": stimulus_names.most_common(1)[0][0] if stimulus_names else None,
            "stimulus_condition": stimulus_conditions.most_common(1)[0][0] if stimulus_conditions else None,
            "schema_versions": dict(schema_versions),
            "line_count": line_count,
            "valid_row_count": valid_row_count,
            "empty_lines": empty_lines,
            "malformed_lines": len(malformed),
            "malformed_samples": malformed,
            "first_time_s": _scalar(first_time),
            "last_time_s": _scalar(last_time),
            "duration_s": _scalar(last_time - first_time if first_time is not None and last_time is not None else None),
            "stimulus_onset_s": _scalar(stimulus_onset_s),
            "monotonic_time": monotonic_time,
            "monotonic_frame_index": frame_indices_monotonic,
        },
        "causal_chain": {
            "events": cleaned_events,
            "declared_order": list(EVENT_NAMES),
            "observed_order_after_stimulus": observed_order,
            "first_nonzero_after_stimulus_s": chain_times,
            "all_events_observed": chain_complete,
            "temporal_order_matches_declared": chain_order_matches,
            "temporal_order_is_not_causality": True,
        },
        "totals": {
            "neural_spikes_total": _scalar(neural_spikes_total) if neural_spike_count_observed else None,
            "neural_spike_count_observed": neural_spike_count_observed,
            "active_neurons_max": _scalar(active_neurons_max),
            "active_neurons_final": _scalar(active_neurons_last),
            "event_value_sums": {event: _mapping(cleaned_events[event]).get("sum_value") for event in EVENT_NAMES},
            "event_integrals_value_s": {event: _mapping(cleaned_events[event]).get("integral_value_s") for event in EVENT_NAMES},
            "population_spike_totals": {key: _scalar(value) for key, value in sorted(population_spike_totals.items()) if key in _POPULATION_ALIAS_KEYS},
            "population_spike_nonzero_frames": {key: int(population_nonzero_frames[key]) for key in sorted(population_nonzero_frames) if key in _POPULATION_ALIAS_KEYS},
            "collision_count_max": _scalar(collision_max),
            "diagnostic_warning_count": warnings_count,
        },
        "final_body_state": final_state,
        "resources": resources,
        "provenance": _provenance_integrity(manifest, provenance_values, sorted(forbidden_fields), valid_row_count),
        "manifest": manifest_compact,
    }
    return summary


# Friendly aliases used by callers that prefer a run-oriented name.
summarize_file = summarize_jsonl
summarize_run = summarize_jsonl
summarize_escape_run = summarize_jsonl


def _summary_label(summary: Mapping[str, Any], fallback: str) -> str:
    run = _mapping(summary.get("run"))
    condition = run.get("condition") or run.get("stimulus_condition") or run.get("stimulus_name")
    return str(condition) if condition else fallback


def _pair_delta(moving: Mapping[str, Any], baseline: Mapping[str, Any], moving_label: str, baseline_label: str) -> dict[str, Any]:
    moving_chain = _mapping(moving.get("causal_chain")).get("events", {})
    baseline_chain = _mapping(baseline.get("causal_chain")).get("events", {})
    timing: dict[str, Any] = {}
    for event in EVENT_NAMES:
        left = _mapping(_mapping(moving_chain).get(event))
        right = _mapping(_mapping(baseline_chain).get(event))
        timing[event] = {
            "moving_first_nonzero_after_stimulus_s": left.get("first_nonzero_after_stimulus_s"),
            "baseline_first_nonzero_after_stimulus_s": right.get("first_nonzero_after_stimulus_s"),
            "delta_s_moving_minus_baseline": _event_delta(left, right, "first_nonzero_after_stimulus_s"),
            "moving_observed": bool(left.get("observed")),
            "baseline_observed": bool(right.get("observed")),
        }
    moving_totals = _mapping(moving.get("totals"))
    baseline_totals = _mapping(baseline.get("totals"))
    moving_pop = _mapping(moving_totals.get("population_spike_totals"))
    baseline_pop = _mapping(baseline_totals.get("population_spike_totals"))
    pop_keys = sorted(set(moving_pop) | set(baseline_pop))
    population_deltas = {
        key: {
            "moving": moving_pop.get(key),
            "baseline": baseline_pop.get(key),
            "delta": _event_delta(moving_pop, baseline_pop, key),
        }
        for key in pop_keys
    }
    moving_events = _mapping(moving_chain)
    baseline_events = _mapping(baseline_chain)
    channel_deltas = {
        event: {
            "moving_integral_value_s": _mapping(moving_events.get(event)).get("integral_value_s"),
            "baseline_integral_value_s": _mapping(baseline_events.get(event)).get("integral_value_s"),
            "delta": _event_delta(_mapping(moving_events.get(event)), _mapping(baseline_events.get(event)), "integral_value_s"),
        }
        for event in EVENT_NAMES
    }
    moving_body = _mapping(moving.get("final_body_state"))
    baseline_body = _mapping(baseline.get("final_body_state"))
    return {
        "moving_label": moving_label,
        "baseline_label": baseline_label,
        "timing": timing,
        "population_spike_totals": population_deltas,
        "channel_integral_deltas": channel_deltas,
        "neural_spikes_total_delta": _event_delta(moving_totals, baseline_totals, "neural_spikes_total"),
        "final_body": {
            "position_delta_m": _vector_delta(moving_body.get("position_m"), baseline_body.get("position_m")),
            "velocity_delta_mps": _vector_delta(moving_body.get("velocity_mps"), baseline_body.get("velocity_mps")),
            "grounded_moving": moving_body.get("grounded"),
            "grounded_baseline": baseline_body.get("grounded"),
            "collision_count_delta": _event_delta(moving_body, baseline_body, "collision_count"),
        },
        "interpretation": "Differences are descriptive deltas between digital traces; they do not establish a causal or biological effect.",
    }


def compare_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    moving: str | None = None,
    static: str | None = None,
    sham: str | None = None,
) -> dict[str, Any]:
    """Compare compact summaries, reporting moving-vs-static and vs-sham."""

    if not summaries:
        raise EscapeSummaryError("at least one summary is required")
    labels = list(summaries)
    def choose(explicit: str | None, markers: Sequence[str]) -> str | None:
        if explicit is not None:
            if explicit not in summaries:
                raise EscapeSummaryError(f"comparison label is not present: {explicit}")
            return explicit
        for label in labels:
            haystack = f"{label} {_summary_label(summaries[label], label)}".lower()
            if any(marker in haystack for marker in markers):
                return label
        return None
    moving_label = choose(moving, ("moving", "looming", "dynamic", "core"))
    static_label = choose(static, ("static", "matched"))
    sham_label = choose(sham, ("sham", "blank"))
    report: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "decision": NOT_VALIDATED,
        "biological_validation": False,
        "interpretation": "Descriptive digital-trace comparison only; no biological or causal inference.",
        "runs": {
            label: {
                "condition": _mapping(summary.get("run")).get("condition"),
                "path": _mapping(summary.get("input")).get("path"),
                "line_count": _mapping(summary.get("run")).get("line_count"),
                "summary_schema": summary.get("schema_version"),
            }
            for label, summary in summaries.items()
        },
        "selected": {"moving": moving_label, "static": static_label, "sham": sham_label},
        "deltas": {},
        "notes": [],
    }
    if moving_label is None:
        report["notes"].append("moving run was not identified")
    if static_label is not None and moving_label is not None:
        report["deltas"]["moving_vs_static"] = _pair_delta(summaries[moving_label], summaries[static_label], moving_label, static_label)
    elif static_label is not None:
        report["notes"].append("moving_vs_static unavailable without moving and static runs")
    if sham_label is not None and moving_label is not None:
        report["deltas"]["moving_vs_sham"] = _pair_delta(summaries[moving_label], summaries[sham_label], moving_label, sham_label)
    elif sham_label is not None:
        report["notes"].append("moving_vs_sham unavailable without moving and sham runs")
    return report


def compare_runs(
    runs: Mapping[str, str | Path | Mapping[str, Any]],
    *,
    moving: str | None = None,
    static: str | None = None,
    sham: str | None = None,
    compute_hash: bool = True,
) -> dict[str, Any]:
    """Summarise and compare labelled runs.

    Values may be input paths or already-computed summary dictionaries.
    """

    summaries: dict[str, Mapping[str, Any]] = {}
    for label, source in runs.items():
        if isinstance(source, Mapping) and source.get("schema_version") == SCHEMA_VERSION:
            summaries[str(label)] = source
        else:
            summaries[str(label)] = summarize_jsonl(source, compute_hash=compute_hash)  # type: ignore[arg-type]
    return compare_summaries(summaries, moving=moving, static=static, sham=sham)


compare_artifacts = compare_runs


def _format_number(value: Any, digits: int = 6) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number:.{digits}g}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render either a single summary or a comparison report as compact MD."""

    if summary.get("schema_version") == COMPARISON_SCHEMA_VERSION:
        lines = [
            "# Strict escape-run comparison",
            "",
            "> Engineering comparison of digital-simulation traces. Temporal deltas are descriptive; they are not causal or biological validation.",
            "",
            f"Decision: **{summary.get('decision', NOT_VALIDATED)}**",
            "",
            "## Runs",
            "",
            "| Label | Condition | Frames | Path |",
            "|---|---|---:|---|",
        ]
        for label, run in _mapping(summary.get("runs")).items():
            lines.append(f"| {label} | {_mapping(run).get('condition') or '—'} | {_mapping(run).get('line_count') or '—'} | {_mapping(run).get('path') or '—'} |")
        selected = _mapping(summary.get("selected"))
        lines.extend(["", f"Selected moving/static/sham: `{selected.get('moving') or '—'}` / `{selected.get('static') or '—'}` / `{selected.get('sham') or '—'}`", ""])
        for name, delta in _mapping(summary.get("deltas")).items():
            lines.extend([f"## {name.replace('_', ' ')}", "", "| Event | Moving first (s) | Baseline first (s) | Δ (s) |", "|---|---:|---:|---:|"])
            for event, timing in _mapping(_mapping(delta).get("timing")).items():
                lines.append(f"| {event} | {_format_number(_mapping(timing).get('moving_first_nonzero_after_stimulus_s'))} | {_format_number(_mapping(timing).get('baseline_first_nonzero_after_stimulus_s'))} | {_format_number(_mapping(timing).get('delta_s_moving_minus_baseline'))} |")
            lines.append("")
        if summary.get("notes"):
            lines.extend(["## Notes", ""])
            lines.extend(f"- {note}" for note in summary["notes"])
        return "\n".join(lines).rstrip() + "\n"

    run = _mapping(summary.get("run"))
    chain = _mapping(summary.get("causal_chain"))
    events = _mapping(chain.get("events"))
    lines = [
        "# Strict escape-run summary",
        "",
        "> Engineering summary of a digital-simulation trace. Nonzero neural/motor values and temporal ordering do not establish biological validation or causality.",
        "",
        f"Decision: **{summary.get('decision', NOT_VALIDATED)}**",
        f"Condition: `{run.get('condition') or 'unknown'}`  ",
        f"Frames: `{run.get('line_count', 0)}`; time: `{_format_number(run.get('first_time_s'))}`–`{_format_number(run.get('last_time_s'))}` s; malformed lines: `{run.get('malformed_lines', 0)}`",
        "",
        "## Trace timing",
        "",
        "| Event | First after stimulus | Interval | Peak | Nonzero frames | Total spikes |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for event in EVENT_NAMES:
        item = _mapping(events.get(event))
        interval = item.get("first_nonzero_after_stimulus_interval_s") or item.get("first_nonzero_interval_s")
        interval_text = "—" if not interval else f"{_format_number(interval[0])}–{_format_number(interval[1])}"
        lines.append(f"| {event} | {_format_number(item.get('first_nonzero_after_stimulus_s'))} | {interval_text} | {_format_number(item.get('peak_value'))} | {item.get('nonzero_frames', 0)} | {_format_number(item.get('total_spikes')) if item.get('spike_count_observed') else '—'} |")
    lines.extend([
        "",
        f"Observed order after stimulus: `{', '.join(chain.get('observed_order_after_stimulus', [])) or 'none'}`",
        f"Declared order matched in the trace: `{chain.get('temporal_order_matches_declared')}` (ordering is not causality)",
        "",
        "## Totals and final body state",
        "",
        f"Neural spikes (if reported): `{_format_number(_mapping(summary.get('totals')).get('neural_spikes_total'))}`; peak active neurons: `{_format_number(_mapping(summary.get('totals')).get('active_neurons_max'))}`; diagnostic warnings: `{_mapping(summary.get('totals')).get('diagnostic_warning_count', 0)}`.",
        "",
        "```json",
        json.dumps(summary.get("final_body_state"), ensure_ascii=False, sort_keys=True, indent=2),
        "```",
        "",
        "## Integrity",
        "",
        f"Resource integrity: `{_mapping(summary.get('resources')).get('integrity_status', 'INCOMPLETE')}`; provenance integrity: `{_mapping(summary.get('provenance')).get('status', 'INCOMPLETE')}`.",
        f"Input SHA-256: `{_mapping(summary.get('input')).get('sha256') or 'not computed'}`",
    ])
    return "\n".join(lines).rstrip() + "\n"


def write_report(summary: Mapping[str, Any], output_json: str | Path | None = None, output_markdown: str | Path | None = None) -> None:
    if output_json is not None:
        path = Path(output_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if output_markdown is not None:
        path = Path(output_markdown).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(summary), encoding="utf-8")


def _parse_label_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label.strip(), path.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="telemetry.jsonl paths or run directories")
    parser.add_argument("--input", dest="input_alias", action="append", default=[], help="additional telemetry path")
    parser.add_argument("--manifest", help="manifest JSON for a single input")
    parser.add_argument("--compare", action="store_true", help="compare labelled moving/static/sham runs")
    parser.add_argument("--moving", help="moving run path (compare mode)")
    parser.add_argument("--static", help="static matched run path (compare mode)")
    parser.add_argument("--sham", help="blank sham run path (compare mode)")
    parser.add_argument("--run", dest="runs", action="append", type=_parse_label_path, default=[], help="labelled run LABEL=PATH (repeatable)")
    parser.add_argument("--output", "--out", dest="output_json", help="JSON report path")
    parser.add_argument("--markdown", "--out-md", dest="output_markdown", help="Markdown report path")
    parser.add_argument("--no-hash", action="store_true", help="skip input SHA-256 to reduce CPU work")
    parser.add_argument("--threshold", type=float, default=DEFAULT_EVENT_THRESHOLD, help="strictly-positive event threshold (default: %(default)s)")
    args = parser.parse_args(argv)
    try:
        if args.compare or args.runs or args.moving or args.static or args.sham or len(args.inputs) + len(args.input_alias) > 1:
            labelled: dict[str, str | Path] = {label: path for label, path in args.runs}
            for label, path in (("moving", args.moving), ("static", args.static), ("sham", args.sham)):
                if path:
                    labelled[label] = path
            positional = list(args.inputs) + list(args.input_alias)
            for index, path in enumerate(positional):
                labelled.setdefault(Path(path).parent.name or f"run_{index + 1}", path)
            if not labelled:
                parser.error("compare mode needs --moving/--static/--sham, --run LABEL=PATH, or multiple inputs")
            report = compare_runs(labelled, moving="moving" if "moving" in labelled else None, static="static" if "static" in labelled else None, sham="sham" if "sham" in labelled else None, compute_hash=not args.no_hash)
        else:
            positional = list(args.inputs) + list(args.input_alias)
            if len(positional) != 1:
                parser.error("one telemetry path or run directory is required")
            report = summarize_jsonl(positional[0], manifest_path=args.manifest, compute_hash=not args.no_hash, event_threshold=args.threshold)
        if args.output_json or args.output_markdown:
            write_report(report, args.output_json, args.output_markdown)
        else:
            sys.stdout.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        return 0
    except (EscapeSummaryError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main())
