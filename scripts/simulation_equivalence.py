#!/usr/bin/env python3
"""Dependency-light equivalence checks for digital-fly simulation effects.

This module compares two telemetry JSONL streams (or run directories) without
loading the complete trace into memory.  It deliberately separates exact
discrete evidence (frame cadence, event identity, visual-input hashes) from
numeric continuous evidence (body state and wing outputs).  Missing optional
evidence is reported as ``unavailable``; values are never inferred from a
different channel.

The public entry point is :func:`compare_runs`.  The command line interface
prints a short result by default and emits the complete JSON report with
``--json`` or writes it with ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "simulation-effect-equivalence-1"
DEFAULT_TOLERANCES: dict[str, float] = {
    "timestamp_abs": 1.0e-9,
    "event_time_abs": 1.0e-8,
    "position_max_abs": 1.0e-6,
    "position_rmse": 2.0e-7,
    "velocity_max_abs": 1.0e-6,
    "velocity_rmse": 2.0e-7,
    "orientation_max_abs": 1.0e-6,
    "orientation_rmse": 2.0e-7,
    "angular_velocity_max_abs": 1.0e-5,
    "angular_velocity_rmse": 2.0e-6,
    "wing_max_abs": 2.0e-6,
    "wing_rmse": 5.0e-7,
    "population_rate_max_abs": 1.0e-4,
    "population_rate_rmse": 5.0e-5,
    "repeat_spike_relative": 0.02,
}

EVENTS = ("LC4", "LPLC2", "DNp01", "TTMn", "wing_motor", "liftoff")
POPULATION_EVENTS = ("LC4", "LPLC2", "DNp01", "TTMn")
VISUAL_HASH_FIELDS = (
    ("left_columns", ("visual_input", "left_columns", "sha256")),
    ("right_columns", ("visual_input", "right_columns", "sha256")),
    ("t4t5_drive", ("visual_input", "t4t5_drive", "sha256")),
)
METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "position": ("position_m",),
    "velocity": ("velocity_mps",),
    "orientation": ("orientation_rad",),
    "orientation_quat": ("orientation_quat",),
    "angular_velocity": ("angular_velocity_rad_s",),
    "wing": ("left_wing", "right_wing", "wing_drive_left", "wing_drive_right"),
}
WING_CHANNEL_ALIASES = {
    "left_wing": ("left_wing_activation", "wing_motor_left", "wing_drive_left"),
    "right_wing": ("right_wing_activation", "wing_motor_right", "wing_drive_right"),
    "wing_drive_left": ("wing_drive_left", "left_wing_activation", "wing_motor_left"),
    "wing_drive_right": ("wing_drive_right", "right_wing_activation", "wing_motor_right"),
}


class EquivalenceError(ValueError):
    """Raised for malformed comparison inputs."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _path(value: Any, parts: Sequence[str]) -> Any:
    current = value
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _as_vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    result: list[float] = []
    for item in value:
        if not _finite(item):
            return None
        result.append(float(item))
    return result


def _nonfinite(value: Any, prefix: str = "") -> list[str]:
    """Find nonfinite numeric leaves, bounded to avoid huge error reports."""

    found: list[str] = []
    if len(found) >= 16:
        return found
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix or "<root>"]
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_nonfinite(item, f"{prefix}.{key}" if prefix else str(key)))
            if len(found) >= 16:
                break
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_nonfinite(item, f"{prefix}[{index}]"))
            if len(found) >= 16:
                break
    return found[:16]


def _source_paths(source: str | Path) -> tuple[Path, Path | None]:
    path = Path(source).expanduser()
    if path.is_dir():
        manifest = path / "manifest.json"
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise EquivalenceError(f"cannot read manifest {manifest}: {exc}") from exc
            outputs = payload.get("outputs", {}) if isinstance(payload, Mapping) else {}
            raw = outputs.get("telemetry") if isinstance(outputs, Mapping) else None
            telemetry = Path(str(raw)).expanduser() if raw else path / "telemetry.jsonl"
            if not telemetry.is_absolute():
                telemetry = path / telemetry
            return telemetry, manifest
        return path / "telemetry.jsonl", None
    return path, path.parent / "manifest.json" if (path.parent / "manifest.json").is_file() else None


def iter_rows(source: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL objects from a telemetry file or run directory."""

    telemetry, _ = _source_paths(source)
    try:
        handle = telemetry.open("r", encoding="utf-8")
    except OSError as exc:
        raise EquivalenceError(f"cannot open telemetry {telemetry}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise EquivalenceError(f"blank telemetry line {line_number} in {telemetry}")
            try:
                row = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise EquivalenceError(f"invalid JSON at {telemetry}:{line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise EquivalenceError(f"telemetry row {line_number} is not an object")
            yield dict(row)


def _manifest_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Keep stable workload identity while excluding run-specific paths/timing."""

    config = manifest.get("config")
    config_hash = config.get("sha256") if isinstance(config, Mapping) else None
    requested = manifest.get("requested_execution")
    effective = manifest.get("effective_execution")
    return {
        "schema_version": manifest.get("schema_version"),
        "seed": manifest.get("seed"),
        "config_sha256": config_hash,
        "requested_backends": manifest.get("requested_backends"),
        "requested_renderer_backend": manifest.get("requested_renderer_backend"),
        "effective_backends": manifest.get("effective_backends"),
        "effective_renderer_backend": manifest.get("effective_renderer_backend"),
        "requested_execution": _stable_execution(requested),
        "effective_execution": _stable_execution(effective),
        "scenario_names": manifest.get("scenario_names"),
        "resource_limits": manifest.get("resource_limits"),
    }


def _stable_execution(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    omit = {
        "frame_count", "wall_time_s", "timestamp_utc", "out", "output",
        "full_debug", "telemetry_mode", "full_neural_telemetry",
        "renderer_transfer", "neural_warmup", "timing",
    }
    return {str(k): v for k, v in value.items() if str(k) not in omit}


def _identity_report(reference: str | Path, candidate: str | Path) -> dict[str, Any]:
    _, ref_manifest_path = _source_paths(reference)
    _, cand_manifest_path = _source_paths(candidate)
    if ref_manifest_path is None or cand_manifest_path is None:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "reason": "both inputs need sibling manifest.json for workload identity",
        }
    try:
        ref_manifest = json.loads(ref_manifest_path.read_text(encoding="utf-8"))
        cand_manifest = json.loads(cand_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "status": "UNAVAILABLE", "reason": str(exc)}
    if not isinstance(ref_manifest, Mapping) or not isinstance(cand_manifest, Mapping):
        return {"available": False, "status": "UNAVAILABLE", "reason": "manifest is not an object"}
    ref = _manifest_projection(ref_manifest)
    cand = _manifest_projection(cand_manifest)
    mismatches = {
        key: {"reference": ref.get(key), "candidate": cand.get(key)}
        for key in sorted(set(ref) | set(cand)) if ref.get(key) != cand.get(key)
    }
    return {"available": True, "status": "PASS" if not mismatches else "FAIL",
            "reference": ref, "candidate": cand, "mismatches": mismatches}


class _Accumulator:
    def __init__(self) -> None:
        self.count = 0
        self.max_abs = 0.0
        self.sum_sq = 0.0
        self.max_index: int | None = None

    def add(self, values: Iterable[float], index: int) -> None:
        for value in values:
            magnitude = abs(float(value))
            self.count += 1
            self.sum_sq += magnitude * magnitude
            if magnitude > self.max_abs:
                self.max_abs = magnitude
                self.max_index = index

    def report(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "max_abs": self.max_abs if self.count else None,
            "rmse": math.sqrt(self.sum_sq / self.count) if self.count else None,
            "max_index": self.max_index,
        }


def _metric_values(row: Mapping[str, Any], metric: str) -> list[float] | None:
    if metric == "wing":
        values: list[float] = []
        nm = row.get("nm_out")
        channels = nm.get("channels") if isinstance(nm, Mapping) else None
        for key in METRIC_FIELDS[metric]:
            value = row.get(key)
            if value is None and isinstance(channels, Mapping):
                value = next(
                    (channels[alias] for alias in WING_CHANNEL_ALIASES.get(key, ()) if alias in channels),
                    None,
                )
            if not _finite(value):
                # wing_drive fields are optional in older traces; do not make
                # the entire metric unavailable when only the legacy pair exists.
                if key.startswith("wing_drive"):
                    continue
                return None
            values.append(float(value))
        return values or None
    value = row.get(METRIC_FIELDS[metric][0])
    return _as_vector(value)


def _quat_delta(reference: Sequence[float], candidate: Sequence[float]) -> list[float]:
    dot = sum(float(a) * float(b) for a, b in zip(reference, candidate))
    sign = -1.0 if dot < 0.0 else 1.0
    return [float(a) - sign * float(b) for a, b in zip(reference, candidate)]


def _population_counts(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    neural = row.get("neural")
    named = neural.get("named_outputs") if isinstance(neural, Mapping) else None
    counts = named.get("population_spike_counts") if isinstance(named, Mapping) else None
    if isinstance(counts, Mapping):
        return counts
    return None


def _event_value(row: Mapping[str, Any], event: str) -> tuple[bool, bool]:
    """Return (evidence_available, observed_in_this_row)."""

    if event in POPULATION_EVENTS:
        counts = _population_counts(row)
        if counts is None:
            return False, False
        value = counts.get(event)
        return _finite(value), _finite(value) and float(value) > 0.0
    if event == "wing_motor":
        nm = row.get("nm_out")
        channels = nm.get("channels") if isinstance(nm, Mapping) else None
        value = channels.get("wing_motor") if isinstance(channels, Mapping) else None
        if not _finite(value):
            value = row.get("wing_drive_left", row.get("left_wing"))
        return _finite(value), _finite(value) and float(value) > 1.0e-12
    if event == "liftoff":
        # A grounded transition is a discrete event.  The first row cannot
        # prove a transition, even when it is already airborne.
        return "grounded" in row, False
    return False, False


def _visual_hashes(row: Mapping[str, Any]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name, parts in VISUAL_HASH_FIELDS:
        value = _path(row, parts)
        result[name] = value if isinstance(value, str) and value else None
    if result["t4t5_drive"] is None:
        value = row.get("visual_drive_sha256")
        result["t4t5_drive"] = value if isinstance(value, str) and value else None
    return result


def _timestamp(row: Mapping[str, Any]) -> float | None:
    for key in ("interval_start_s", "frame_time_s", "time_s"):
        value = row.get(key)
        if _finite(value):
            return float(value)
    timestamp = row.get("timestamp")
    value = timestamp.get("t_s") if isinstance(timestamp, Mapping) else None
    return float(value) if _finite(value) else None


def _interval(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    start = row.get("interval_start_s", row.get("frame_time_s"))
    end = row.get("interval_end_s", row.get("time_s"))
    return (float(start) if _finite(start) else None, float(end) if _finite(end) else None)


def _hash_sequence(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical(record))
    return digest.hexdigest()


def _load_tolerances(value: Mapping[str, Any] | str | Path | None) -> dict[str, float]:
    result = dict(DEFAULT_TOLERANCES)
    if value is None:
        return result
    payload: Any = value
    if isinstance(value, (str, Path)):
        text = str(value)
        path = Path(text).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(text)
        except (OSError, ValueError) as exc:
            raise EquivalenceError(f"invalid tolerances: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EquivalenceError("tolerances must be a JSON object")
    for key, item in payload.items():
        if not _finite(item) or float(item) < 0.0:
            raise EquivalenceError(f"tolerance {key!r} must be a non-negative finite number")
        result[str(key)] = float(item)
    return result


def compare_runs(
    reference: str | Path,
    candidate: str | Path,
    *,
    tolerances: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Compare two telemetry streams and return a structured report."""

    tol = _load_tolerances(tolerances)
    failures: list[str] = []
    unavailable: list[dict[str, Any]] = []
    identity = _identity_report(reference, candidate)
    if identity.get("status") == "FAIL":
        failures.append("workload identity mismatch")
    elif not identity.get("available"):
        unavailable.append({"observable": "workload_identity", "reason": identity.get("reason")})

    metric_acc = {name: _Accumulator() for name in ("position", "velocity", "orientation", "orientation_quat", "angular_velocity", "wing")}
    metric_available = {name: [False, False] for name in metric_acc}
    metric_missing_rows = {name: 0 for name in metric_acc}
    event_info = {
        side: {event: {"available": False, "observed": False, "frame_index": None,
                       "interval_s": None, "time_s": None} for event in EVENTS}
        for side in ("reference", "candidate")
    }
    hash_records: dict[str, list[dict[str, Any]]] = {"reference": [], "candidate": []}
    frame_count = {"reference": 0, "candidate": 0}
    malformed: list[str] = []
    cadence = {"reference": {"first": None, "last": None, "dt_min": None, "dt_max": None},
               "candidate": {"first": None, "last": None, "dt_min": None, "dt_max": None}}
    previous_time: dict[str, float | None] = {"reference": None, "candidate": None}
    previous_grounded: dict[str, bool | None] = {"reference": None, "candidate": None}
    # Consume through zip_longest so an extra tail cannot be mistaken for a
    # successful aligned comparison.
    try:
        iterator_pairs = itertools.zip_longest(iter_rows(reference), iter_rows(candidate), fillvalue=None)  # type: ignore[assignment]
        for index, pair in enumerate(iterator_pairs):
            ref_row, cand_row = pair
            if ref_row is None or cand_row is None:
                side = "reference" if ref_row is None else "candidate"
                failures.append(f"{side} has missing/extra row at aligned index {index}")
                continue
            metric_values = {
                metric: (_metric_values(ref_row, metric), _metric_values(cand_row, metric))
                for metric in metric_acc
            }
            for side, row in (("reference", ref_row), ("candidate", cand_row)):
                frame_count[side] += 1
                bad = _nonfinite(row)
                if bad:
                    malformed.append(f"{side} row {index} nonfinite: {', '.join(bad[:3])}")
                frame = row.get("frame_index")
                if not isinstance(frame, int) or isinstance(frame, bool):
                    failures.append(f"{side} row {index} missing integer frame_index")
                elif frame != index:
                    failures.append(f"{side} frame reorder/missing at row {index}: got {frame}")
                timestamp = _timestamp(row)
                if timestamp is None:
                    failures.append(f"{side} row {index} has no finite timestamp")
                else:
                    if cadence[side]["first"] is None:
                        cadence[side]["first"] = timestamp
                    cadence[side]["last"] = timestamp
                    if previous_time[side] is not None:
                        dt = timestamp - float(previous_time[side])
                        cadence[side]["dt_min"] = dt if cadence[side]["dt_min"] is None else min(cadence[side]["dt_min"], dt)
                        cadence[side]["dt_max"] = dt if cadence[side]["dt_max"] is None else max(cadence[side]["dt_max"], dt)
                    previous_time[side] = timestamp
                hashes = _visual_hashes(row)
                hash_records[side].append({"frame": index, **hashes})
                for event in EVENTS:
                    available, observed = _event_value(row, event)
                    item = event_info[side][event]
                    if available:
                        item["available"] = True
                    if event == "liftoff" and "grounded" in row:
                        grounded = row.get("grounded")
                        if previous_grounded[side] is not None and previous_grounded[side] and grounded is False and item["frame_index"] is None:
                            observed = True
                        previous_grounded[side] = grounded if isinstance(grounded, bool) else None
                    if observed and item["frame_index"] is None:
                        start, end = _interval(row)
                        item.update({"observed": True, "frame_index": index,
                                     "interval_s": [start, end], "time_s": timestamp})
                side_index = 0 if side == "reference" else 1
                for metric in metric_acc:
                    metric_available[metric][side_index] = metric_values[metric][side_index] is not None
            for metric in metric_acc:
                ref_values, cand_values = metric_values[metric]
                if ref_values is None or cand_values is None:
                    metric_missing_rows[metric] += 1
                    if (ref_values is None) != (cand_values is None):
                        failures.append(f"{metric} missing on one side at frame {index}")
                    continue
                if metric == "orientation_quat":
                    delta = _quat_delta(ref_values, cand_values) if len(ref_values) == len(cand_values) == 4 else None
                else:
                    delta = [a - b for a, b in zip(ref_values, cand_values)] if len(ref_values) == len(cand_values) else None
                if delta is None:
                    failures.append(f"{metric} shape mismatch at frame {index}")
                else:
                    metric_acc[metric].add(delta, index)
    except EquivalenceError as exc:
        failures.append(str(exc))
    if malformed:
        failures.extend(malformed)

    if frame_count["reference"] != frame_count["candidate"]:
        failures.append(f"frame count mismatch: reference={frame_count['reference']} candidate={frame_count['candidate']}")
    if frame_count["reference"] == 0 or frame_count["candidate"] == 0:
        failures.append("one or both telemetry streams are empty")

    cadence_report: dict[str, Any] = {"reference": cadence["reference"], "candidate": cadence["candidate"]}
    if cadence["reference"]["first"] is not None and cadence["candidate"]["first"] is not None:
        if abs(float(cadence["reference"]["first"]) - float(cadence["candidate"]["first"])) > tol["timestamp_abs"]:
            failures.append("first timestamp mismatch")
        if abs(float(cadence["reference"]["last"]) - float(cadence["candidate"]["last"])) > tol["timestamp_abs"]:
            failures.append("last timestamp mismatch")
    for side in ("reference", "candidate"):
        if frame_count[side] > 1 and cadence[side]["dt_min"] is None:
            failures.append(f"{side} cadence is unavailable")
        elif cadence[side]["dt_min"] is not None and cadence[side]["dt_min"] <= 0:
            failures.append(f"{side} cadence is not strictly increasing")

    hash_report: dict[str, Any] = {}
    for name, _ in VISUAL_HASH_FIELDS:
        refs = [item[name] for item in hash_records["reference"]]
        cands = [item[name] for item in hash_records["candidate"]]
        available_ref = all(isinstance(item, str) for item in refs) and bool(refs)
        available_cand = all(isinstance(item, str) for item in cands) and bool(cands)
        sequence = {
            "reference": _hash_sequence({"frame": i, name: value} for i, value in enumerate(refs)) if available_ref else None,
            "candidate": _hash_sequence({"frame": i, name: value} for i, value in enumerate(cands)) if available_cand else None,
            "available": available_ref and available_cand,
        }
        hash_report[name] = sequence
        if available_ref and available_cand and refs != cands:
            failures.append(f"visual input hash mismatch: {name}")
        elif available_ref != available_cand:
            unavailable.append({"observable": f"visual_input.{name}", "reason": "hash present on only one side"})
        elif not available_ref:
            unavailable.append({"observable": f"visual_input.{name}", "reason": "hash-only visual evidence absent"})

    metric_report: dict[str, Any] = {}
    for metric, accumulator in metric_acc.items():
        item = accumulator.report()
        metric_report[metric] = item
        if not all(metric_available[metric]):
            unavailable.append({"observable": metric, "reason": "continuous field absent on one or both sides"})
            continue
        if metric_missing_rows[metric]:
            unavailable.append({"observable": metric, "reason": f"continuous field missing at {metric_missing_rows[metric]} aligned row(s)"})
        max_key = "orientation_max_abs" if metric in {"orientation", "orientation_quat"} else f"{metric}_max_abs"
        rmse_key = "orientation_rmse" if metric in {"orientation", "orientation_quat"} else f"{metric}_rmse"
        if item["max_abs"] is not None and item["max_abs"] > tol[max_key]:
            failures.append(f"{metric} max_abs={item['max_abs']:.9g} exceeds {tol[max_key]:.9g}")
        if item["rmse"] is not None and item["rmse"] > tol[rmse_key]:
            failures.append(f"{metric} rmse={item['rmse']:.9g} exceeds {tol[rmse_key]:.9g}")

    event_report: dict[str, Any] = {}
    for event in EVENTS:
        ref = event_info["reference"][event]
        cand = event_info["candidate"][event]
        available = bool(ref["available"] and cand["available"])
        item = {"available": available, "reference": ref, "candidate": cand}
        event_report[event] = item
        if not available:
            unavailable.append({"observable": f"event.{event}", "reason": "event channel absent on one or both sides"})
            continue
        if ref["frame_index"] != cand["frame_index"]:
            failures.append(f"event frame mismatch: {event} reference={ref['frame_index']} candidate={cand['frame_index']}")
        for side in ("reference", "candidate"):
            interval = event_info[side][event]["interval_s"]
            if interval and any(item is None for item in interval):
                unavailable.append({"observable": f"event.{event}.interval_s", "reason": f"{side} interval unavailable"})
        ref_time = ref.get("time_s")
        cand_time = cand.get("time_s")
        if ref_time is not None and cand_time is not None and abs(float(ref_time) - float(cand_time)) > tol["event_time_abs"]:
            failures.append(f"event time mismatch: {event}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "reference": str(reference),
        "candidate": str(candidate),
        "tolerances": tol,
        "workload_identity": identity,
        "cadence": {"frame_count": frame_count, "timestamps": cadence_report},
        "visual_input_hashes": hash_report,
        "events": event_report,
        "continuous_metrics": metric_report,
        "unavailable": unavailable,
        "failures": failures,
    }
    return report


def compare_telemetry(reference: str | Path, candidate: str | Path, *, tolerances: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Alias emphasizing that inputs may be JSONL files rather than runs."""

    return compare_runs(reference, candidate, tolerances=tolerances)


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="reference run directory or telemetry JSONL")
    parser.add_argument("candidate", help="candidate run directory or telemetry JSONL")
    parser.add_argument("--tolerances", help="JSON file or inline JSON object of tolerance overrides")
    parser.add_argument("--output", type=Path, help="write the complete JSON report to this path")
    parser.add_argument("--json", action="store_true", help="print the complete JSON report")
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.reference, args.candidate, tolerances=args.tolerances)
    except (EquivalenceError, OSError, ValueError) as exc:
        report = {"schema_version": SCHEMA_VERSION, "status": "FAIL", "failures": [str(exc)], "unavailable": []}
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        print(f"{report.get('status', 'FAIL')} simulation-effect-equivalence "
              f"failures={len(report.get('failures', []))} "
              f"unavailable={len(report.get('unavailable', []))}")
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


main = _cli


__all__ = [
    "SCHEMA_VERSION", "DEFAULT_TOLERANCES", "EquivalenceError", "iter_rows",
    "compare_runs", "compare_telemetry", "main",
]
