#!/usr/bin/env python3
"""Verify and summarize a recorded long Torch-CUDA articulated-leg run.

The analyzer streams telemetry, per-tick spike counts, and optional sensory
feedback.  It reports host timing and sampled body behavior as engineering
evidence; it does not establish autonomous walking or biological equivalence.

Example::

    python scripts/analyze_leg_long_run.py RUN_DIR --out /data/artifacts/by-project/malecns-gpu/leg-long-run-analysis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterator, Mapping


SCHEMA_VERSION = "leg-long-run-analysis-1"
PERFORMANCE_SCHEMA_VERSION = "leg-neuromuscular-performance-1"
STAGES = ("neural_s", "muscle_physics_s", "sensory_encode_s", "other_s")
WINDOW_TICKS = 2000
HASH_CHUNK_BYTES = 1024 * 1024


class LegLongRunAnalysisError(ValueError):
    """Raised when a run is incomplete, inconsistent, or not the requested backend."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LegLongRunAnalysisError(f"{where} must be a JSON object")
    return value


def _number(value: Any, where: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        raise LegLongRunAnalysisError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LegLongRunAnalysisError(f"{where} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise LegLongRunAnalysisError(f"{where} must be {qualifier}")
    return result


def _integer(value: Any, where: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool):
        raise LegLongRunAnalysisError(f"{where} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegLongRunAnalysisError(f"{where} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise LegLongRunAnalysisError(f"{where} must be an integer")
    if isinstance(value, str) and str(result) != value.strip():
        raise LegLongRunAnalysisError(f"{where} must be an integer")
    if nonnegative and result < 0:
        raise LegLongRunAnalysisError(f"{where} must be nonnegative")
    return result


def _json(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegLongRunAnalysisError(f"cannot read {where}: {exc}") from exc
    return dict(_mapping(value, where))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(block)
    except OSError as exc:
        raise LegLongRunAnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _check_hash(path: Path, expected: Any, where: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise LegLongRunAnalysisError(f"{where} SHA-256 is missing or malformed")
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise LegLongRunAnalysisError(f"{where} SHA-256 does not match the manifest")
    return actual


def _jsonl(path: Path, digest: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield one parsed JSONL object at a time while hashing the exact bytes."""
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                if not raw.strip():
                    raise LegLongRunAnalysisError(f"{path.name} has a blank line at {line_number}")
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise LegLongRunAnalysisError(
                        f"{path.name} has invalid JSON at line {line_number}: {exc}") from exc
                yield line_number, dict(_mapping(value, f"{path.name}:{line_number}"))
    except OSError as exc:
        raise LegLongRunAnalysisError(f"cannot read {path}: {exc}") from exc


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _body_from_row(row: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    body = row.get("body")
    if not isinstance(body, Mapping):
        raise LegLongRunAnalysisError(f"{where}.body is missing")
    return body


def _root_from_body(body: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    root = body.get("root")
    if not isinstance(root, Mapping):
        raise LegLongRunAnalysisError(f"{where}.root is missing")
    return root


def _vector(value: Any, length: int, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise LegLongRunAnalysisError(f"{where} must contain {length} numbers")
    return [_number(part, f"{where}[{index}]") for index, part in enumerate(value)]


def _all_numeric_finite(value: Any, where: str) -> None:
    """Check numeric leaves in a physical snapshot without copying its arrays."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _all_numeric_finite(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _all_numeric_finite(item, f"{where}[{index}]")
        return
    if isinstance(value, (int, float)):
        _number(value, where)


def _snapshot_compact(value: Any, where: str) -> dict[str, Any]:
    snapshot = _mapping(value, where)
    _all_numeric_finite(snapshot, where)
    telemetry = _mapping(snapshot.get("telemetry"), f"{where}.telemetry")
    root = _root_from_body(telemetry, where)
    position = _vector(root.get("position_m"), 3, f"{where}.telemetry.root.position_m")
    velocity = _vector(root.get("linear_velocity_mps"), 3,
                       f"{where}.telemetry.root.linear_velocity_mps")
    feet = _mapping(telemetry.get("feet"), f"{where}.telemetry.feet")
    joints = _mapping(telemetry.get("joints"), f"{where}.telemetry.joints")
    return {
        "time_s": _number(telemetry.get("time_s"), f"{where}.telemetry.time_s"),
        "root_position_m": position,
        "root_height_m": position[2],
        "root_linear_velocity_mps": velocity,
        "joint_count": len(joints),
        "foot_count": sum(bool(_mapping(foot, f"{where}.telemetry.feet.{leg}").get("contact"))
                           for leg, foot in feet.items()),
        "total_foot_normal_force_N": _number(
            telemetry.get("total_foot_normal_force_N"),
            f"{where}.telemetry.total_foot_normal_force_N", nonnegative=True),
        "physical_state_finite": True,
    }


def _stage_map(value: Any, where: str) -> dict[str, float]:
    stages = _mapping(value, where)
    missing = [name for name in STAGES if name not in stages]
    if missing:
        raise LegLongRunAnalysisError(f"{where} is missing stages: {', '.join(missing)}")
    return {name: _number(stages[name], f"{where}.{name}", nonnegative=True)
            for name in STAGES}


def _close(actual: float, expected: float, where: str, *, rel: float = 2e-5,
           abs_tol: float = 2e-7) -> None:
    if not math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol):
        raise LegLongRunAnalysisError(
            f"{where} is inconsistent (recorded {actual:.9g}, expected {expected:.9g})")


def _performance_summary(run_dir: Path, manifest: Mapping[str, Any],
                         summary: Mapping[str, Any], dt_s: float,
                         expected_ticks: int) -> dict[str, Any]:
    metadata = _mapping(manifest.get("performance"), "manifest.performance")
    relpath = metadata.get("path")
    if not isinstance(relpath, str) or not relpath:
        raise LegLongRunAnalysisError("manifest.performance.path is missing")
    performance_path = (run_dir / relpath).resolve()
    if not _inside(performance_path, run_dir.resolve()):
        raise LegLongRunAnalysisError("performance metadata path escapes the run directory")
    performance = _json(performance_path, "performance.json")
    digest = _check_hash(performance_path, metadata.get("sha256"), "performance.json")
    if performance.get("schema_version") != PERFORMANCE_SCHEMA_VERSION:
        raise LegLongRunAnalysisError("unsupported performance.json schema_version")
    initialization_s = _number(performance.get("initialization_s"),
                               "performance.initialization_s", nonnegative=True)
    loop_s = _number(performance.get("simulation_loop_s"),
                     "performance.simulation_loop_s", nonnegative=True)
    if loop_s <= 0:
        raise LegLongRunAnalysisError("performance.simulation_loop_s must be positive")
    stages = _stage_map(performance.get("stages_s"), "performance.stages_s")
    _close(sum(stages.values()), loop_s, "performance stage sum", rel=1e-2, abs_tol=1e-3)
    ticks = _integer(performance.get("neural_ticks"), "performance.neural_ticks",
                     nonnegative=True)
    if ticks != expected_ticks:
        raise LegLongRunAnalysisError("performance.neural_ticks does not match the complete run")
    sim_duration = _number(performance.get("simulated_duration_s"),
                           "performance.simulated_duration_s", nonnegative=True)
    _close(sim_duration, expected_ticks * dt_s, "performance simulated duration")
    _close(sim_duration, _number(summary.get("duration_s"), "summary.duration_s", nonnegative=True),
           "summary/performance duration")
    if "simulation_loop_wall_s" in summary:
        _close(loop_s, _number(summary["simulation_loop_wall_s"],
                               "summary.simulation_loop_wall_s", nonnegative=True),
               "summary/performance simulation loop wall time", rel=1e-7, abs_tol=1e-8)
    if "initialization_wall_s" in summary:
        _close(initialization_s, _number(summary["initialization_wall_s"],
                                         "summary.initialization_wall_s", nonnegative=True),
               "summary/performance initialization wall time", rel=1e-7, abs_tol=1e-8)

    raw_windows = performance.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise LegLongRunAnalysisError("performance.windows must be a nonempty array")
    windows: list[dict[str, Any]] = []
    cursor = 0
    window_stage_totals = dict.fromkeys(STAGES, 0.0)
    window_elapsed_total = 0.0
    for index, raw in enumerate(raw_windows):
        entry = _mapping(raw, f"performance.windows[{index}]")
        start = _integer(entry.get("start_tick"), f"performance.windows[{index}].start_tick",
                         nonnegative=True)
        end = _integer(entry.get("end_tick"), f"performance.windows[{index}].end_tick",
                       nonnegative=True)
        expected_span = min(WINDOW_TICKS, expected_ticks - cursor)
        if start != cursor or end < start or end - start + 1 != expected_span:
            raise LegLongRunAnalysisError(
                f"performance window {index} does not cover the expected contiguous tick schedule")
        simulated_end = _number(entry.get("simulated_end_s"),
                                 f"performance.windows[{index}].simulated_end_s",
                                 nonnegative=True)
        _close(simulated_end, (end + 1) * dt_s,
               f"performance window {index} simulated end")
        elapsed = _number(entry.get("elapsed_s"),
                          f"performance.windows[{index}].elapsed_s", nonnegative=True)
        window_stages = _stage_map(entry.get("stages_s"),
                                   f"performance.windows[{index}].stages_s")
        _close(sum(window_stages.values()), elapsed,
               f"performance window {index} stage sum", rel=1e-2, abs_tol=1e-4)
        for stage in STAGES:
            window_stage_totals[stage] += window_stages[stage]
        window_elapsed_total += elapsed
        windows.append({"start_tick": start, "end_tick": end,
                        "simulated_duration_s": (end - start + 1) * dt_s,
                        "elapsed_s": elapsed, "stages_s": window_stages})
        cursor = end + 1
    if cursor != expected_ticks:
        raise LegLongRunAnalysisError("performance windows do not cover every neural tick")
    _close(window_elapsed_total, loop_s, "sum of performance window elapsed times",
           rel=1e-2, abs_tol=1e-3)
    for stage in STAGES:
        _close(window_stage_totals[stage], stages[stage], f"window sum for {stage}")

    later = windows[1:]
    later_slowdowns = [window["elapsed_s"] / window["simulated_duration_s"]
                       for window in later]
    return {
        "performance_sha256": digest,
        "initialization_s": initialization_s,
        "simulation_loop_s": loop_s,
        "simulated_duration_s": sim_duration,
        "slowdown_factor_host_seconds_per_simulated_second": loop_s / sim_duration,
        "realtime_factor_simulated_seconds_per_host_second": sim_duration / loop_s,
        "stages": {
            name: {"elapsed_s": value, "simulation_loop_percent": 100.0 * value / loop_s}
            for name, value in stages.items()
        },
        "windows": windows,
        "post_initial_window_timing": {
            "label": "first window excluded as initialization or settling window",
            "window_count": len(later_slowdowns),
            "slowdown_factor_host_seconds_per_simulated_second": {
                "median": statistics.median(later_slowdowns) if later_slowdowns else None,
                "minimum": min(later_slowdowns) if later_slowdowns else None,
                "maximum": max(later_slowdowns) if later_slowdowns else None,
            },
            "interpretation": "descriptive windows from one run; not repeated-run confidence intervals",
        },
    }


def _read_feedback(path: Path, expected_sha: Any, *, required: bool) -> tuple[int, str | None]:
    if not path.is_file():
        if required:
            raise LegLongRunAnalysisError("sensory feedback is enabled but sensory_feedback.jsonl is missing")
        if expected_sha is not None:
            raise LegLongRunAnalysisError("manifest records a sensory feedback hash but its file is missing")
        return 0, None
    digest = _check_hash(path, expected_sha, "sensory_feedback.jsonl") if expected_sha is not None else _sha256(path)
    rows = 0
    stream_digest = hashlib.sha256()
    for _line_number, _record in _jsonl(path, stream_digest):
        rows += 1
    if stream_digest.hexdigest() != digest:
        raise LegLongRunAnalysisError("sensory_feedback.jsonl changed while it was being analyzed")
    return rows, digest


def _analyze_body(run_dir: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any],
                  telemetry_path: Path, expected_duration_s: float,
                  dt_s: float, expected_ticks: int) -> tuple[dict[str, Any], str]:
    body_provenance = _mapping(_mapping(manifest.get("pipeline"), "manifest.pipeline").get("body"),
                               "manifest.pipeline.body")
    mass = _number(body_provenance.get("total_model_mass_kg"),
                   "manifest.pipeline.body.total_model_mass_kg", nonnegative=True)
    effective_raw = body_provenance.get("effective_physics_parameters")
    effective = (_mapping(effective_raw, "manifest.pipeline.body.effective_physics_parameters")
                 if effective_raw is not None else {})
    gravity_source: dict[str, Any]
    if "gravity_mps2" in effective:
        gravity = _number(effective["gravity_mps2"],
                          "manifest.pipeline.body.effective_physics_parameters.gravity_mps2",
                          nonnegative=True)
        gravity_source = {"source": "manifest.pipeline.body.effective_physics_parameters",
                          "source_config_sha256": None}
    else:
        snapshot_directory = manifest.get("source_snapshot_directory")
        source_hashes = _mapping(manifest.get("source_sha256"), "manifest.source_sha256")
        expected_config_sha = source_hashes.get("articulated_leg_body.json")
        body_config_sha = body_provenance.get("config_sha256")
        if not isinstance(snapshot_directory, str) or not isinstance(expected_config_sha, str) \
                or body_config_sha != expected_config_sha:
            raise LegLongRunAnalysisError(
                "gravity metadata is absent and the articulated body config provenance is not hash-verifiable")
        config_path = (run_dir / snapshot_directory / "articulated_leg_body.json").resolve()
        if not _inside(config_path, run_dir.resolve()):
            raise LegLongRunAnalysisError("articulated body config snapshot path escapes the run directory")
        config_sha = _check_hash(config_path, expected_config_sha,
                                 "sources/articulated_leg_body.json")
        config = _json(config_path, "sources/articulated_leg_body.json")
        physics = _mapping(config.get("physics"), "articulated_leg_body.json.physics")
        gravity = _number(physics.get("gravity_mps2"),
                          "articulated_leg_body.json.physics.gravity_mps2", nonnegative=True)
        gravity_source = {"source": "hash-verified sources/articulated_leg_body.json",
                          "source_config_sha256": config_sha}
    weight_N = mass * gravity
    if weight_N <= 0:
        raise LegLongRunAnalysisError("model body weight must be positive")

    initial = _snapshot_compact(summary.get("initial"), "summary.initial")
    final = _snapshot_compact(summary.get("final"), "summary.final")
    if not math.isclose(final["time_s"], expected_duration_s, rel_tol=0.0,
                        abs_tol=max(1e-8, dt_s * 1.01)):
        raise LegLongRunAnalysisError("summary.final physical state time does not reach the requested duration")

    stream_digest = hashlib.sha256()
    first_position: list[float] | None = None
    last_position: list[float] | None = None
    first_sample_time: float | None = None
    last_sample_time: float | None = None
    samples = 0
    min_height = math.inf
    final_height: float | None = None
    max_root_speed = 0.0
    min_foot_count = math.inf
    max_foot_count = -math.inf
    foot_count_sum = 0
    min_weight_ratio = math.inf
    max_weight_ratio = -math.inf
    weight_ratio_sum = 0.0
    nonfoot_contacts = 0
    late_samples = 0
    late_height_min = math.inf
    late_height_max = -math.inf
    late_height_sum = 0.0
    late_speed_max = 0.0
    late_foot_count_sum = 0
    late_weight_ratio_sum = 0.0
    late_nonfoot_contacts = 0
    last_sample_foot_count: int | None = None
    last_sample_normal_force: float | None = None
    late_start_s = max(0.0, expected_duration_s - 1.0 - dt_s / 2.0)
    record_stride = max(1, round(0.005 / dt_s))
    regular_sample_count = (expected_ticks - 1) // record_stride + 1
    expected_sample_count = regular_sample_count + int((expected_ticks - 1) % record_stride != 0)

    for line_number, row in _jsonl(telemetry_path, stream_digest):
        where = f"telemetry.jsonl:{line_number}"
        body = _body_from_row(row, where)
        root = _root_from_body(body, where)
        neural = _mapping(row.get("neural"), f"{where}.neural")
        tick = _integer(neural.get("tick"), f"{where}.neural.tick", nonnegative=True)
        sample_index = samples
        if sample_index < regular_sample_count:
            expected_tick = sample_index * record_stride
        elif (expected_ticks - 1) % record_stride:
            expected_tick = expected_ticks - 1
        else:
            raise LegLongRunAnalysisError("telemetry has extra rows beyond its tick schedule")
        if tick != expected_tick:
            raise LegLongRunAnalysisError(
                f"telemetry sample tick {tick} at line {line_number} violates the recorded cadence")
        timestamp = _number(row.get("time_s", body.get("time_s")), f"{where}.time_s",
                            nonnegative=True)
        body_time = _number(body.get("time_s"), f"{where}.body.time_s", nonnegative=True)
        _close(timestamp, body_time, f"{where} row/body clock", rel=0.0, abs_tol=1e-10)
        _close(timestamp, (tick + 1) * dt_s, f"{where} end-of-tick clock", rel=0.0,
               abs_tol=1e-8)
        if last_sample_time is not None and timestamp <= last_sample_time:
            raise LegLongRunAnalysisError("telemetry sample times must increase strictly")
        if timestamp > expected_duration_s + dt_s * 1.1:
            raise LegLongRunAnalysisError("telemetry contains a sample after the declared duration")
        position = _vector(root.get("position_m"), 3, f"{where}.body.root.position_m")
        velocity = _vector(root.get("linear_velocity_mps"), 3,
                           f"{where}.body.root.linear_velocity_mps")
        height = position[2]
        speed = math.sqrt(sum(component * component for component in velocity))
        feet = _mapping(body.get("feet"), f"{where}.body.feet")
        if not feet:
            raise LegLongRunAnalysisError(f"{where}.body.feet must be nonempty")
        foot_contacts = 0
        summed_normal = 0.0
        for leg, raw_foot in feet.items():
            foot = _mapping(raw_foot, f"{where}.body.feet.{leg}")
            contact = foot.get("contact")
            if not isinstance(contact, bool):
                raise LegLongRunAnalysisError(f"{where}.body.feet.{leg}.contact must be boolean")
            foot_contacts += int(contact)
            force = _number(foot.get("normal_force_N"),
                            f"{where}.body.feet.{leg}.normal_force_N", nonnegative=True)
            summed_normal += force
        if "total_foot_normal_force_N" in body:
            total_normal = _number(body["total_foot_normal_force_N"],
                                   f"{where}.body.total_foot_normal_force_N", nonnegative=True)
            _close(total_normal, summed_normal, f"{where} sum of per-foot normal forces",
                   rel=1e-4, abs_tol=1e-10)
        else:
            total_normal = summed_normal
        weight_ratio = total_normal / weight_N
        nonfoot = body.get("body_ground_contact")
        if not isinstance(nonfoot, bool):
            raise LegLongRunAnalysisError(f"{where}.body.body_ground_contact must be boolean")

        first_position = position if first_position is None else first_position
        first_sample_time = timestamp if first_sample_time is None else first_sample_time
        last_position = position
        last_sample_time = timestamp
        last_sample_foot_count = foot_contacts
        last_sample_normal_force = total_normal
        samples += 1
        min_height = min(min_height, height)
        final_height = height
        max_root_speed = max(max_root_speed, speed)
        min_foot_count = min(min_foot_count, foot_contacts)
        max_foot_count = max(max_foot_count, foot_contacts)
        foot_count_sum += foot_contacts
        min_weight_ratio = min(min_weight_ratio, weight_ratio)
        max_weight_ratio = max(max_weight_ratio, weight_ratio)
        weight_ratio_sum += weight_ratio
        nonfoot_contacts += int(nonfoot)
        if timestamp >= late_start_s:
            late_samples += 1
            late_height_min = min(late_height_min, height)
            late_height_max = max(late_height_max, height)
            late_height_sum += height
            late_speed_max = max(late_speed_max, speed)
            late_foot_count_sum += foot_contacts
            late_weight_ratio_sum += weight_ratio
            late_nonfoot_contacts += int(nonfoot)

    telemetry_hash = stream_digest.hexdigest()
    if samples == 0 or first_position is None or last_position is None or final_height is None:
        raise LegLongRunAnalysisError("telemetry.jsonl is empty")
    if samples != expected_sample_count:
        raise LegLongRunAnalysisError(
            f"telemetry has {samples} samples; recorded cadence requires {expected_sample_count}")
    actual_hash = _check_hash(telemetry_path, manifest.get("telemetry_sha256"),
                              "telemetry.jsonl")
    if telemetry_hash != actual_hash:
        raise LegLongRunAnalysisError("telemetry.jsonl changed while it was being analyzed")
    if late_samples == 0:
        raise LegLongRunAnalysisError("telemetry has no recorded samples in the final simulated second")
    _close(last_sample_time, final["time_s"], "final telemetry/snapshot time",
           rel=0.0, abs_tol=max(1e-8, dt_s * 1.01))
    for axis, label in enumerate(("x", "y", "z")):
        _close(last_position[axis], final["root_position_m"][axis],
               f"final telemetry/snapshot root position {label}", rel=0.0, abs_tol=1e-10)
    if last_sample_foot_count != final["foot_count"]:
        raise LegLongRunAnalysisError("final telemetry/snapshot foot contact counts disagree")
    _close(last_sample_normal_force, final["total_foot_normal_force_N"],
           "final telemetry/snapshot foot normal force", rel=1e-4, abs_tol=1e-10)

    displacement = [last_position[i] - first_position[i] for i in range(3)]
    body_result = {
        "initial_state": initial,
        "final_state": final,
        "telemetry_samples": samples,
        "sampled_time_span_s": [first_sample_time, last_sample_time],
        "sampling_claim": "metrics below use recorded telemetry samples only; no claim is made about unsampled ticks",
        "sampled_root_height_m": {
            "minimum": min_height,
            "final_sample": final_height,
            "late_1s_minimum": late_height_min,
            "late_1s_mean": late_height_sum / late_samples,
            "late_1s_maximum": late_height_max,
            "late_1s_sample_count": late_samples,
        },
        "sampled_root_stability": {
            "maximum_root_speed_mps": max_root_speed,
            "late_1s_maximum_root_speed_mps": late_speed_max,
            "telemetry_endpoint_drift_m": displacement,
            "telemetry_endpoint_horizontal_drift_m": math.hypot(displacement[0], displacement[1]),
        },
        "sampled_foot_support": {
            "contacting_feet_per_sample": {
                "minimum": int(min_foot_count),
                "mean": foot_count_sum / samples,
                "maximum": int(max_foot_count),
                "final_sample": final["foot_count"],
            },
            "foot_normal_force_to_model_weight_ratio": {
                "minimum": min_weight_ratio,
                "mean": weight_ratio_sum / samples,
                "maximum": max_weight_ratio,
                "final_sample": final["total_foot_normal_force_N"] / weight_N,
                "model_weight_N": weight_N,
                "model_weight_basis": {"mass_kg": mass, "gravity_mps2": gravity,
                                       "gravity_source": gravity_source},
            },
            "late_1s_mean_normal_force_to_model_weight_ratio": late_weight_ratio_sum / late_samples,
            "nonfoot_ground_contact_sample_fraction": nonfoot_contacts / samples,
            "late_1s_nonfoot_ground_contact_sample_fraction": late_nonfoot_contacts / late_samples,
            "contact_sampling_basis": "recorded telemetry samples only",
        },
    }
    return body_result, telemetry_hash


def analyze_run(run_dir: str | Path, out_path: str | Path, *, allow_cpu: bool = False) -> dict[str, Any]:
    """Validate one completed run and build its compact engineering summary.

    ``allow_cpu`` exists for explicit CPU fixture analysis. Production use
    requires the recorded summary to prove that CUDA was actually used.
    """
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    if not run_dir.is_dir():
        raise LegLongRunAnalysisError(f"run directory does not exist: {run_dir}")
    out_path = Path(out_path).expanduser().resolve(strict=False)
    if _inside(out_path, run_dir):
        raise LegLongRunAnalysisError("analysis output must be outside the input run directory")
    if out_path.exists():
        raise LegLongRunAnalysisError(f"analysis output already exists; refusing to overwrite: {out_path}")

    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    telemetry_path = run_dir / "telemetry.jsonl"
    counts_path = run_dir / "neural_spike_counts.jsonl"
    feedback_path = run_dir / "sensory_feedback.jsonl"
    for path in (manifest_path, summary_path, telemetry_path, counts_path):
        if not path.is_file():
            raise LegLongRunAnalysisError(f"required run input is missing: {path.name}")

    manifest = _json(manifest_path, "manifest.json")
    summary = _json(summary_path, "summary.json")
    manifest_summary = _mapping(manifest.get("summary"), "manifest.summary")
    for field in ("duration_s", "neural_dt_s", "neural_spike_shape", "full_neural_trace_ticks",
                  "total_spikes", "afferent_spikes", "mapped_motor_spikes", "gpu_used",
                  "neural_backend_actual", "neural_device_actual"):
        if field in manifest_summary and manifest_summary.get(field) != summary.get(field):
            raise LegLongRunAnalysisError(f"manifest.summary.{field} disagrees with summary.json")

    gpu_used = summary.get("gpu_used") is True
    backend = summary.get("neural_backend_actual")
    device = summary.get("neural_device_actual")
    cuda_truth = gpu_used and backend in ("torch_cuda", "triton") and device == "cuda"
    if not cuda_truth and not allow_cpu:
        raise LegLongRunAnalysisError(
            "run summary does not prove actual CUDA neural execution; use --allow-cpu only for an explicit fixture or CPU analysis")
    if gpu_used and not cuda_truth:
        raise LegLongRunAnalysisError("GPU execution metadata is internally inconsistent")

    duration_s = _number(summary.get("duration_s"), "summary.duration_s", nonnegative=True)
    dt_s = _number(summary.get("neural_dt_s"), "summary.neural_dt_s", nonnegative=True)
    if duration_s <= 0 or dt_s <= 0:
        raise LegLongRunAnalysisError("summary duration and neural timestep must be positive")
    expected_ticks = round(duration_s / dt_s)
    _close(expected_ticks * dt_s, duration_s, "summary duration/neural timestep", rel=0.0,
           abs_tol=1e-9)
    shape = summary.get("neural_spike_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise LegLongRunAnalysisError("summary.neural_spike_shape must be [ticks, neurons]")
    shape_ticks = _integer(shape[0], "summary.neural_spike_shape[0]", nonnegative=True)
    neurons = _integer(shape[1], "summary.neural_spike_shape[1]", nonnegative=True)
    if shape_ticks != expected_ticks or neurons <= 0:
        raise LegLongRunAnalysisError("summary neural spike shape does not match the actual tick schedule")
    if _integer(summary.get("full_neural_trace_ticks"),
                "summary.full_neural_trace_ticks", nonnegative=True) != expected_ticks:
        raise LegLongRunAnalysisError("summary full neural trace tick count is incomplete")

    performance_result = _performance_summary(run_dir, manifest, summary, dt_s, expected_ticks)
    feedback_mode = str(summary.get("sensory_feedback_mode", "unrecorded"))
    feedback_rows, feedback_sha = _read_feedback(
        feedback_path, manifest.get("sensory_feedback_sha256"),
        required=feedback_mode in ("on", "muted"))
    if feedback_mode in ("on", "muted") and feedback_rows != expected_ticks:
        raise LegLongRunAnalysisError("sensory feedback trace does not contain one row per neural tick")

    counts_digest = hashlib.sha256()
    totals = {name: 0 for name in
              ("total_spikes", "target_spikes", "afferent_spikes", "mapped_motor_spikes")}
    optional_count_fields = ("afferent_spikes", "mapped_motor_spikes")
    optional_field_seen: dict[str, bool | None] = {name: None for name in optional_count_fields}
    sensory_first: dict[str, Any] | None = None
    sensory_last: dict[str, Any] | None = None
    last_second_totals = dict.fromkeys(totals, 0)
    count_rows = 0
    feedback_stream = iter(_jsonl(feedback_path, hashlib.sha256())) if feedback_rows else None
    feedback_digest = hashlib.sha256()
    for line_number, row in _jsonl(counts_path, counts_digest):
        tick = _integer(row.get("tick"), f"neural_spike_counts.jsonl:{line_number}.tick",
                        nonnegative=True)
        if tick != count_rows:
            raise LegLongRunAnalysisError("neural spike count ticks must start at zero and be contiguous")
        timestamp = _number(row.get("t_s"),
                            f"neural_spike_counts.jsonl:{line_number}.t_s", nonnegative=True)
        _close(timestamp, tick * dt_s, f"neural spike count tick {tick} time", rel=0.0,
               abs_tol=max(1e-10, dt_s * 1e-6))
        for name in ("total_spikes", "target_spikes"):
            totals[name] += _integer(row.get(name),
                                     f"neural_spike_counts.jsonl:{line_number}.{name}",
                                     nonnegative=True)
        for name in optional_count_fields:
            present = name in row
            if optional_field_seen[name] is None:
                optional_field_seen[name] = present
            elif optional_field_seen[name] != present:
                raise LegLongRunAnalysisError(
                    f"neural spike count field {name} is present on only part of the trace")
            if present:
                totals[name] += _integer(row[name],
                                         f"neural_spike_counts.jsonl:{line_number}.{name}",
                                         nonnegative=True)
        if timestamp >= duration_s - 1.0 - dt_s / 2.0:
            for name in ("total_spikes", "target_spikes"):
                last_second_totals[name] += _integer(row[name],
                    f"neural_spike_counts.jsonl:{line_number}.{name}", nonnegative=True)
            for name in optional_count_fields:
                if name in row:
                    last_second_totals[name] += _integer(row[name],
                        f"neural_spike_counts.jsonl:{line_number}.{name}", nonnegative=True)
        afferent_count = _integer(row["afferent_spikes"],
            f"neural_spike_counts.jsonl:{line_number}.afferent_spikes", nonnegative=True) \
            if "afferent_spikes" in row else None
        if afferent_count is not None and afferent_count > 0:
            event = {"tick": tick, "time_s": timestamp,
                     "afferent_spikes": afferent_count}
            if sensory_first is None:
                sensory_first = event
            sensory_last = event
        if feedback_stream is not None:
            try:
                feedback_line_number, feedback = next(feedback_stream)
            except StopIteration as exc:
                raise LegLongRunAnalysisError("sensory feedback rows end before the neural count trace") from exc
            feedback_digest.update(json.dumps(feedback, sort_keys=True).encode("utf-8"))
            feedback_tick = _integer(feedback.get("tick"),
                                     f"sensory_feedback.jsonl:{feedback_line_number}.tick",
                                     nonnegative=True)
            if feedback_tick != tick:
                raise LegLongRunAnalysisError("sensory feedback ticks do not align with neural count ticks")
            if afferent_count is not None and "afferent_spikes" in feedback and _integer(
                    feedback["afferent_spikes"], "sensory feedback afferent_spikes",
                    nonnegative=True) != afferent_count:
                raise LegLongRunAnalysisError("sensory feedback and neural count afferent spikes disagree")
        count_rows += 1
    if feedback_stream is not None:
        try:
            next(feedback_stream)
        except StopIteration:
            pass
        else:
            raise LegLongRunAnalysisError("sensory feedback contains rows beyond the neural count trace")
    counts_sha = counts_digest.hexdigest()
    if count_rows != expected_ticks:
        raise LegLongRunAnalysisError(
            f"neural spike count trace has {count_rows} rows; expected {expected_ticks}")
    actual_counts_sha = _check_hash(counts_path, manifest.get("neural_counts_sha256"),
                                    "neural_spike_counts.jsonl")
    if counts_sha != actual_counts_sha:
        raise LegLongRunAnalysisError("neural_spike_counts.jsonl changed while it was being analyzed")
    # Recheck optional sensory input against its exact bytes after streaming.
    if feedback_path.is_file() and feedback_sha != _sha256(feedback_path):
        raise LegLongRunAnalysisError("sensory_feedback.jsonl changed while it was being analyzed")

    inferred_zero_subcounts = False
    for name in optional_count_fields:
        if optional_field_seen[name] is False:
            if totals["total_spikes"] == 0:
                totals[name] = 0
                last_second_totals[name] = 0
                inferred_zero_subcounts = True
            else:
                totals[name] = None
                last_second_totals[name] = None
        elif optional_field_seen[name] is None:
            raise LegLongRunAnalysisError("neural spike count trace is empty")

    for field in ("total_spikes", "target_spikes"):
        recorded = _integer(summary.get(field), f"summary.{field}", nonnegative=True)
        if totals[field] != recorded:
            raise LegLongRunAnalysisError(
                f"summary.{field} does not equal the streamed per-tick total")
    for field in optional_count_fields:
        if totals[field] is not None and field in summary:
            recorded = _integer(summary[field], f"summary.{field}", nonnegative=True)
            if totals[field] != recorded:
                raise LegLongRunAnalysisError(
                    f"summary.{field} does not equal the streamed per-tick total")
        elif field in summary and totals[field] is None:
            raise LegLongRunAnalysisError(
                f"summary.{field} exists but the count trace does not record it")

    body_result, telemetry_sha = _analyze_body(
        run_dir, summary, manifest, telemetry_path, duration_s, dt_s, expected_ticks)
    body_provenance = _mapping(_mapping(manifest.get("pipeline"), "manifest.pipeline").get("body"),
                               "manifest.pipeline.body")
    passive_support_metadata = body_provenance.get("passive_support")
    if isinstance(passive_support_metadata, Mapping) and isinstance(
            passive_support_metadata.get("enabled"), bool):
        passive_support_interpretation = (
            "The configured joint spring and damping are engineering terms; this run cannot establish biological support or neural tonus."
            if passive_support_metadata["enabled"] else
            "Passive joint support was disabled for this run; observed metrics describe this model configuration only.")
    else:
        passive_support_interpretation = (
            "Passive-support status was not recorded in this run's provenance; this report makes no claim about it.")
    report = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "path": str(run_dir),
            "case": summary.get("case"),
            "leg": summary.get("leg"),
            "evidence_class": summary.get("evidence_class", "digital_simulation"),
            "duration_s": duration_s,
            "neural_dt_s": dt_s,
            "neural_ticks": expected_ticks,
            "neuron_count": neurons,
            "integrity": "PASS",
            "sha256": {
                "telemetry.jsonl": telemetry_sha,
                "neural_spike_counts.jsonl": counts_sha,
                "sensory_feedback.jsonl": feedback_sha,
                "performance.json": performance_result["performance_sha256"],
            },
        },
        "execution": {
            "neural_backend_actual": backend,
            "neural_device_actual": device,
            "gpu_used_recorded": gpu_used,
            "gpu_execution_verified": cuda_truth,
            "cpu_analysis_exception": bool(not cuda_truth and allow_cpu),
            "physics_backend": summary.get("physics_backend"),
            "sensory_feedback_mode": feedback_mode,
        },
        "performance": performance_result,
        "neural_activity": {
            "total_spikes": totals["total_spikes"],
            "target_spikes": totals["target_spikes"],
            "total_afferent_spikes": totals["afferent_spikes"],
            "total_mapped_motor_spikes": totals["mapped_motor_spikes"],
            "subcount_provenance": (
                "zero subcounts inferred from a complete all-zero spike trace"
                if inferred_zero_subcounts else
                "recorded per tick" if optional_field_seen["afferent_spikes"] else
                "subcounts were absent and cannot be inferred because total spikes were nonzero"),
            "first_sensory_spike": sensory_first,
            "last_sensory_spike": sensory_last,
            "last_1s_response": {
                name: {"spikes": last_second_totals[name],
                       "status": ("not_recorded" if last_second_totals[name] is None else
                                  "observed" if last_second_totals[name] else "none_observed")}
                for name in totals
            },
            "sensory_feedback_rows": feedback_rows,
            "sensory_feedback_events_observed": sensory_first is not None,
        },
        "body": body_result,
        "interpretation": {
            "classification": "engineering_diagnostic_only",
            "passive_support": passive_support_interpretation,
            "autonomous_walking": "not established by this simulation",
            "biological_equivalence": "not established by this simulation",
            "performance_equivalence": "not established; host wall-clock timing is not a biological timing comparison",
            "performance_scope": "host wall-clock stage timing for one run; not a repeated-run benchmark or confidence interval",
            "body_metric_scope": "recorded telemetry samples only; unsampled physical states are not inferred",
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="new JSON report path outside the run directory")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="allow a recorded CPU backend (intended for explicit fixtures or CPU analysis)")
    args = parser.parse_args(argv)
    try:
        report = analyze_run(args.run_dir, args.out, allow_cpu=args.allow_cpu)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (LegLongRunAnalysisError, OSError, FileExistsError) as exc:
        parser.error(str(exc))
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
