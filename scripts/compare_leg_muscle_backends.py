#!/usr/bin/env python3
"""Compare matched CPU-reference and neural-device leg-muscle artifacts.

This is a read-only engineering agreement check. It does not construct a
neural engine, import a GPU stack, run physics, or treat agreement as biological
validation or evidence of performance equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "articulated-leg-backend-comparison-1"
NEURAL_DT_FIELD = "neural_dt_s"
SPIKE_ENCODING = "tick-major dense uint8 C-order"
CSR_FILES = frozenset({
    "ids.npy", "offsets.npy", "target.npy", "weight.npy", "sign.npy",
    "annotations.npz", "tables.json", "meta.json",
})
DEFAULT_TOLERANCES = {
    "neural": 0.0,
    "joints": 1e-8,
    "muscles": 1e-9,
    "root": 1e-9,
    "contact": 1e-8,
    "other": 1e-9,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ComparisonError(ValueError):
    """An artifact is corrupt or the pair is not a matched comparison."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number {value!r}")
    return number


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_json(raw: str, label: str) -> Any:
    try:
        return json.loads(raw, parse_constant=_reject_constant,
                          parse_float=_finite_json_float,
                          object_pairs_hook=_no_duplicate_keys)
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ComparisonError(f"{label}: invalid JSON: {exc}") from exc


def _read_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComparisonError(f"cannot read {path}: {exc}") from exc
    return _loads_json(raw, str(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ComparisonError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read and hash JSONL in one pass, retaining no input handles."""

    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise ComparisonError(f"{path}:{line_number}: blank JSONL row")
                try:
                    record = _loads_json(raw_line.decode("utf-8"), f"{path}:{line_number}")
                except UnicodeDecodeError as exc:
                    raise ComparisonError(f"{path}:{line_number}: invalid UTF-8") from exc
                if not isinstance(record, dict):
                    raise ComparisonError(f"{path}:{line_number}: JSONL row must be an object")
                rows.append(record)
    except OSError as exc:
        raise ComparisonError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise ComparisonError(f"{path}: JSONL artifact is empty")
    return rows, digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{label} must be a JSON object")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ComparisonError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _json_equal(left: Any, right: Any) -> bool:
    """Structural equality that does not conflate booleans and numbers."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (set(left) == set(right)
                and all(_json_equal(left[key], right[key]) for key in left))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return type(left) is type(right) and left == right


def _hash_map(value: Any, label: str, *, exact_names: frozenset[str] | None = None) -> dict[str, str]:
    mapping = _mapping(value, label)
    result = dict(mapping)
    if not result:
        raise ComparisonError(f"{label} must not be empty")
    if exact_names is not None and set(result) != set(exact_names):
        missing = sorted(set(exact_names) - set(result))
        extra = sorted(set(result) - set(exact_names))
        raise ComparisonError(f"{label} file set differs (missing={missing}, extra={extra})")
    for name, digest in result.items():
        if not isinstance(name, str) or not name or not _is_sha256(digest):
            raise ComparisonError(f"{label}[{name!r}] must be a lowercase SHA-256 digest")
    return result


def _load_run(directory: Path, role: str) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ComparisonError(f"{role} artifact directory does not exist: {root}")
    paths = {name: root / name for name in (
        "manifest.json", "summary.json", "telemetry.jsonl", "neural_spike_counts.jsonl",
    )}
    for name, path in paths.items():
        if not path.is_file():
            raise ComparisonError(f"{role} artifact is missing {name}: {path}")

    manifest = _mapping(_read_json(paths["manifest.json"]), f"{role} manifest.json")
    if "performance" in manifest:
        performance = _mapping(manifest["performance"], f"{role} manifest.performance")
        _require(performance.get("path") == "performance.json", f"{role} invalid performance path")
        _require(_is_sha256(performance.get("sha256"))
                 and (root / "performance.json").is_file()
                 and _sha256_file(root / "performance.json") == performance["sha256"],
                 f"{role} performance.json SHA-256 does not match manifest")
    summary = _mapping(_read_json(paths["summary.json"]), f"{role} summary.json")
    telemetry, telemetry_sha256 = _read_jsonl(paths["telemetry.jsonl"])
    counts, counts_sha256 = _read_jsonl(paths["neural_spike_counts.jsonl"])
    feedback = []
    if "sensory_feedback_sha256" in manifest:
        feedback_path = root / "sensory_feedback.jsonl"
        _require(feedback_path.is_file(), f"{role} missing sensory_feedback.jsonl")
        feedback_hash = _sha256_file(feedback_path)
        _require(_is_sha256(manifest["sensory_feedback_sha256"])
                 and feedback_hash == manifest["sensory_feedback_sha256"],
                 f"{role} sensory_feedback.jsonl SHA-256 does not match manifest")
        mode = summary.get("sensory_feedback_mode")
        _require(mode in {"off", "on", "muted"}, f"{role} invalid sensory feedback mode")
        if mode == "off":
            _require(feedback_path.stat().st_size == 0, f"{role} disabled feedback stream must be empty")
        else:
            feedback, _ = _read_jsonl(feedback_path)
            _require(len(feedback) == len(counts), f"{role} incomplete per-tick sensory feedback stream")
            for tick, (record, count) in enumerate(zip(feedback, counts)):
                _require(record.get("tick") == tick, f"{role} sensory feedback tick mismatch")
                source_time = _finite_number(record.get("source_time_s"), "feedback.source_time_s")
                applied_time = _finite_number(record.get("applied_at_s"), "feedback.applied_at_s")
                _require(abs(source_time - count["t_s"]) <= 1e-10
                         and abs(applied_time - count["t_s"]) <= 1e-10,
                         f"{role} feedback causality/time alignment mismatch at tick {tick}")
                _require(record.get("afferent_spikes") == count.get("afferent_spikes"),
                         f"{role} sensory response disagrees with count stream at tick {tick}")
                afferent_count = count.get("afferent_spikes")
                _require(isinstance(afferent_count, int) and not isinstance(afferent_count, bool)
                         and 0 <= afferent_count <= count["total_spikes"],
                         f"{role} invalid afferent spike count at tick {tick}")
            _require(summary.get("afferent_spikes") == sum(c["afferent_spikes"] for c in counts),
                     f"{role} total afferent spikes disagree with per-tick trace")

    _require(_is_sha256(manifest.get("telemetry_sha256")),
             f"{role} manifest.telemetry_sha256 is missing or malformed")
    _require(manifest["telemetry_sha256"] == telemetry_sha256,
             f"{role} telemetry.jsonl SHA-256 does not match manifest")
    _require(_is_sha256(manifest.get("neural_counts_sha256")),
             f"{role} manifest.neural_counts_sha256 is missing or malformed")
    _require(manifest["neural_counts_sha256"] == counts_sha256,
             f"{role} neural_spike_counts.jsonl SHA-256 does not match manifest")

    source_hashes = _hash_map(manifest.get("source_sha256"), f"{role} manifest.source_sha256")
    snapshot_name = manifest.get("source_snapshot_directory")
    if snapshot_name is not None:
        if not isinstance(snapshot_name, str) or not snapshot_name:
            raise ComparisonError(f"{role} manifest.source_snapshot_directory is malformed")
        snapshot_dir = (root / snapshot_name).resolve()
        if root not in snapshot_dir.parents or not snapshot_dir.is_dir():
            raise ComparisonError(f"{role} source snapshot directory is missing or escapes the artifact: {snapshot_name!r}")
        for name, expected_digest in source_hashes.items():
            if Path(name).name != name or name in (".", ".."):
                raise ComparisonError(f"{role} source hash key is not a simple snapshot filename: {name!r}")
            snapshot_path = snapshot_dir / name
            if not snapshot_path.is_file():
                raise ComparisonError(f"{role} source snapshot is missing {name}")
            if _sha256_file(snapshot_path) != expected_digest:
                raise ComparisonError(f"{role} source snapshot {name} does not match manifest.source_sha256")
    csr_hashes = _hash_map(manifest.get("csr_sha256"), f"{role} manifest.csr_sha256",
                           exact_names=CSR_FILES)
    pipeline = _mapping(manifest.get("pipeline"), f"{role} manifest.pipeline")
    neural = _mapping(manifest.get("neural"), f"{role} manifest.neural")
    _require(pipeline.get("evidence_class") == "digital_simulation",
             f"{role} pipeline evidence_class must be digital_simulation")
    _require(pipeline.get("biological_validation") == "NOT_VALIDATED",
             f"{role} pipeline must preserve biological_validation=NOT_VALIDATED")
    csr = _mapping(neural.get("csr"), f"{role} neural.csr")
    _require(csr.get("valid") is True and not csr.get("blockers"),
             f"{role} neural CSR preflight is invalid")

    recorded_summary = _mapping(manifest.get("summary"), f"{role} manifest.summary")
    for key, value in recorded_summary.items():
        _require(key in summary and _json_equal(value, summary[key]),
                 f"{role} manifest.summary.{key} disagrees with summary.json")

    return {
        "role": role,
        "directory": root,
        "manifest": dict(manifest),
        "summary": dict(summary),
        "pipeline": dict(pipeline),
        "neural": dict(neural),
        "source_sha256": source_hashes,
        "csr_sha256": csr_hashes,
        "telemetry": telemetry,
        "counts": counts,
        "feedback": feedback,
        "telemetry_sha256": telemetry_sha256,
        "counts_sha256": counts_sha256,
    }


def _engine_signature(neural: Mapping[str, Any]) -> dict[str, Any]:
    """Retain the engine contract while omitting the requested backend's runtime details."""

    ignored = {
        "generated_at", "backend_requested", "backend", "device_requested", "device",
        "max_vram_gib", "triton_timing", "capabilities",
    }
    result = {key: value for key, value in neural.items() if key not in ignored}
    csr = result.get("csr")
    if isinstance(csr, Mapping):
        result["csr"] = {key: value for key, value in csr.items() if key != "path"}
    return result


def _without_keys(value: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    excluded = set(keys)
    return {key: item for key, item in value.items() if key not in excluded}


def _validate_role(run: Mapping[str, Any], *, candidate: bool) -> None:
    role = str(run["role"])
    summary = run["summary"]
    neural = run["neural"]
    manifest = run["manifest"]
    required = (
        "case", "leg", "source", "control_authority", "fixed_root", "duration_s",
        NEURAL_DT_FIELD, "seed", "neural_backend_requested", "neural_backend_actual",
        "neural_device_actual", "neural_spike_sha256", "neural_observation_dtype",
        "full_neural_trace_ticks", "neural_spike_shape", "neural_spike_encoding",
        "current_mV_per_ms", "prescribed_rate_hz", "initial", "final",
    )
    for key in required:
        _require(key in summary, f"{role} summary is missing required field {key!r}")
    _require(summary.get("source") == "malecns",
             f"{role} source must be the MaleCNS engine; prescribed-rate controls are not CNS runs")
    _require(summary.get("evidence_class") == "digital_simulation"
             and summary.get("biological_validation") == "NOT_VALIDATED",
             f"{role} summary must preserve the digital-simulation evidence boundary")
    _require(summary.get("physics_backend") == "mujoco_cpu",
             f"{role} physics backend must be CPU MuJoCo")
    _require(summary.get("control_authority") != "prescribed_per_neuron_rate_control",
             f"{role} uses prescribed rates, not the MaleCNS neural engine")
    _require(summary.get("prescribed_rate_hz") is None,
             f"{role} contains prescribed-rate input and cannot be treated as a CNS run")
    _require(_is_sha256(summary.get("neural_spike_sha256")),
             f"{role} neural_spike_sha256 is missing or malformed")
    _require(summary.get("neural_observation_dtype") == "uint8",
             f"{role} neural_observation_dtype must be uint8")
    _require(summary.get("neural_spike_encoding") == SPIKE_ENCODING,
             f"{role} neural_spike_encoding must be {SPIKE_ENCODING!r}")
    _require(isinstance(summary.get("neural_spike_shape"), list)
             and len(summary["neural_spike_shape"]) == 2,
             f"{role} neural_spike_shape must be [ticks, neurons]")

    requested = summary.get("neural_backend_requested")
    actual = summary.get("neural_backend_actual")
    device = summary.get("neural_device_actual")
    engine_backend = neural.get("backend")
    engine_device = neural.get("device")
    expected_engine_backend = {"numpy_cpu": "numpy_cpu", "torch_cuda": "torch_cuda",
                               "triton": "triton"}.get(actual)
    _require(actual in {"numpy_cpu", "torch_cuda", "triton"},
             f"{role} neural backend actual is unsupported: {actual!r}")
    _require(engine_backend == expected_engine_backend,
             f"{role} summary neural backend disagrees with manifest.neural.backend")
    _require(device == engine_device,
             f"{role} summary neural device disagrees with manifest.neural.device")
    _require(summary.get("gpu_used") is (device == "cuda"),
             f"{role} gpu_used flag disagrees with actual neural device")
    if candidate:
        _require(requested in ("torch_cuda", "triton"),
                 f"candidate requested backend must be torch_cuda or triton, got {requested!r}")
        _require(actual in ("torch_cuda", "triton") and device == "cuda",
                 f"candidate did not run the requested neural backend on CUDA (actual={actual!r}, device={device!r})")
        _require(actual == requested,
                 f"candidate requested {requested!r} but actual backend was {actual!r}")
    else:
        _require(requested == "cpu" and actual == "numpy_cpu" and device == "cpu",
                 "reference must be the actual NumPy CPU neural backend")
    _require(neural.get("seed") == summary.get("seed"),
             f"{role} summary seed disagrees with neural engine manifest")
    engine_request = {"cpu": "cpu", "torch_cuda": "torch", "triton": "triton"}.get(requested)
    _require(neural.get("backend_requested") in (None, engine_request),
             f"{role} requested backend disagrees with neural engine manifest")
    preflight = _mapping(manifest.get("neural_backend_preflight"),
                         f"{role} manifest.neural_backend_preflight")
    requires_cuda = requested != "cpu"
    _require(preflight.get("requested_backend") == requested
             and preflight.get("requires_cuda") is requires_cuda
             and preflight.get("gate_passed") is True
             and preflight.get("gate_checked") is requires_cuda
             and preflight.get("blockers") == [],
             f"{role} neural backend preflight does not match request or is not clear")

    duration_s = _finite_number(summary.get("duration_s"), f"{role} duration_s", positive=True)
    dt_s = _finite_number(summary.get(NEURAL_DT_FIELD), f"{role} {NEURAL_DT_FIELD}", positive=True)
    dt_ms = _finite_number(neural.get("dt_ms"), f"{role} neural.dt_ms", positive=True)
    _require(math.isclose(dt_ms / 1000.0, dt_s, rel_tol=0.0, abs_tol=1e-15),
             f"{role} neural dt_s disagrees with manifest.neural.dt_ms")
    ticks_float = duration_s / dt_s
    ticks = int(round(ticks_float))
    _require(ticks > 0 and math.isclose(ticks_float, ticks, rel_tol=0.0, abs_tol=1e-8),
             f"{role} duration is not an integer number of neural ticks")
    _require(summary.get("full_neural_trace_ticks") == ticks,
             f"{role} full_neural_trace_ticks does not match duration/dt")
    _require(isinstance(summary.get("full_neural_trace_ticks"), int)
             and not isinstance(summary.get("full_neural_trace_ticks"), bool),
             f"{role} full_neural_trace_ticks must be an integer")

    n = neural.get("n")
    _require(isinstance(n, int) and not isinstance(n, bool) and n > 0,
             f"{role} manifest.neural.n must be a positive integer")
    _require(summary["neural_spike_shape"] == [ticks, n],
             f"{role} neural_spike_shape does not match full trace ticks and neuron count")
    _require(all(isinstance(value, int) and not isinstance(value, bool)
                 for value in summary["neural_spike_shape"]),
             f"{role} neural_spike_shape dimensions must be integers")
    for key in ("gain", "edges"):
        _require(key in neural, f"{role} neural engine manifest is missing {key!r}")
    _finite_number(neural.get("gain"), f"{role} neural.gain", positive=True)
    _require(isinstance(neural.get("edges"), int) and not isinstance(neural.get("edges"), bool)
             and neural["edges"] > 0,
             f"{role} manifest.neural.edges must be a positive integer")
    _require(isinstance(summary.get("seed"), int) and not isinstance(summary.get("seed"), bool),
             f"{role} seed must be an integer")
    _require(isinstance(summary.get("initial"), Mapping)
             and isinstance(summary.get("final"), Mapping),
             f"{role} initial/final physics snapshots must be objects")

    counts = run["counts"]
    _require(len(counts) == ticks,
             f"{role} neural spike-count trace has {len(counts)} rows; expected {ticks}")
    for expected_tick, row in enumerate(counts):
        _require(isinstance(row.get("tick"), int) and not isinstance(row.get("tick"), bool)
                 and row.get("tick") == expected_tick,
                 f"{role} spike-count ticks must be contiguous from zero (row {expected_tick})")
        t_s = _finite_number(row.get("t_s"), f"{role} spike count tick {expected_tick} t_s")
        _require(math.isclose(t_s, expected_tick * dt_s, rel_tol=0.0, abs_tol=1e-12),
                 f"{role} spike-count timestamp is misaligned at tick {expected_tick}")
        for key in ("total_spikes", "target_spikes"):
            count = row.get(key)
            _require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                     f"{role} spike-count tick {expected_tick} {key} must be a nonnegative integer")
    for key in ("total_spikes", "target_spikes"):
        aggregate = summary.get(key)
        _require(isinstance(aggregate, int) and not isinstance(aggregate, bool)
                 and aggregate == sum(row[key] for row in counts),
                 f"{role} summary.{key} disagrees with per-tick spike counts")

    stride = max(1, round(0.005 / dt_s))
    expected_samples = list(range(0, ticks, stride))
    if not expected_samples or expected_samples[-1] != ticks - 1:
        expected_samples.append(ticks - 1)
    telemetry = run["telemetry"]
    _require(len(telemetry) == len(expected_samples),
             f"{role} telemetry has {len(telemetry)} rows; expected {len(expected_samples)} 5 ms samples plus final")
    for row_index, (record, expected_tick) in enumerate(zip(telemetry, expected_samples)):
        row_time = _finite_number(record.get("time_s"), f"{role} telemetry row {row_index} time_s")
        expected_time = (expected_tick + 1) * dt_s
        _require(math.isclose(row_time, expected_time, rel_tol=0.0, abs_tol=1e-12),
                 f"{role} telemetry time/tick is misaligned at row {row_index}")
        row_neural = _mapping(record.get("neural"), f"{role} telemetry row {row_index}.neural")
        _require(isinstance(row_neural.get("tick"), int)
                 and not isinstance(row_neural.get("tick"), bool)
                 and row_neural.get("tick") == expected_tick,
                 f"{role} telemetry row {row_index} neural.tick is misaligned")
        _require(row_neural.get("target_spikes") == counts[expected_tick]["target_spikes"],
                 f"{role} telemetry target_spikes disagrees with per-tick trace at tick {expected_tick}")
        _require(row_neural.get("total_spikes") == counts[expected_tick]["total_spikes"],
                 f"{role} telemetry total_spikes disagrees with per-tick trace at tick {expected_tick}")


def _category(path: tuple[str, ...]) -> str:
    lowered = tuple(part.lower() for part in path)
    if "neural" in lowered or "spike_counts" in lowered or "currents_mv_per_ms" in lowered:
        return "neural"
    if any(part in {"muscles", "muscle_parameters"} for part in lowered):
        return "muscles"
    if any(part in {"joints", "joint_rest_q_rad"} for part in lowered):
        return "joints"
    if any(part in {"root", "qpos", "qvel", "qacc_warmstart"} for part in lowered):
        return "root"
    if any("contact" in part or part == "feet" or "ground" in part for part in lowered):
        return "contact"
    leaf = lowered[-1] if lowered else ""
    if leaf.endswith("spikes"):
        return "neural"
    if any(token in leaf for token in ("contact", "ground", "foot", "impulse")):
        return "contact"
    if any(token in leaf for token in ("force", "activation", "excitation", "length", "velocity")):
        return "muscles"
    if any(token in leaf for token in ("joint", "moment_arm", "torque_nm", "q_rad", "qd_rad")):
        return "joints"
    if any(token in leaf for token in ("position", "quaternion", "velocity", "displacement", "com_", "center_of_mass")):
        return "root"
    return "other"


def _category_for_index(path: tuple[str, ...], index: int,
                        fixed_root: bool) -> str | None:
    if not path:
        return None
    field = path[-1].lower()
    if field == "qpos":
        return "joints" if fixed_root or index >= 7 else "root"
    if field in {"qvel", "qacc_warmstart"}:
        return "joints" if fixed_root or index >= 6 else "root"
    return None


def _new_stats(tolerances: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    return {name: {"max_abs_error": 0.0, "tolerance": value,
                   "within_tolerance": True, "numeric_leaves": 0,
                   "max_error_path": None}
            for name, value in tolerances.items()}


def _compare_values(left: Any, right: Any, path: tuple[str, ...], *,
                    tolerances: Mapping[str, float], stats: dict[str, dict[str, Any]],
                    failures: list[dict[str, Any]], fixed_root: bool,
                    exact: bool = False, state_values: bool = False) -> None:
    label = ".".join(path) or "<root>"
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            if not state_values or type(left) is not type(right):
                raise ComparisonError(f"matched artifact schema/state differs at {label}")
            failures.append({"path": label, "category": _category(path),
                             "reference": left, "candidate": right,
                             "absolute_error": 1.0, "tolerance": 0.0})
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ComparisonError(f"matched artifact schema differs at {label}")
        if set(left) != set(right):
            missing = sorted(set(left) - set(right))
            extra = sorted(set(right) - set(left))
            raise ComparisonError(f"matched artifact fields differ at {label} (candidate missing={missing}, extra={extra})")
        for key in sorted(left):
            _compare_values(left[key], right[key], path + (str(key),), tolerances=tolerances,
                            stats=stats, failures=failures, fixed_root=fixed_root,
                            exact=exact or str(key) in {"time_s", "tick", "schema_version"},
                            state_values=state_values)
        return
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            if (state_values and path and path[-1] == "body_contacts"
                    and isinstance(left, list) and isinstance(right, list)):
                failures.append({"path": label + ".length", "category": "contact",
                                 "reference": len(left), "candidate": len(right),
                                 "absolute_error": abs(len(left) - len(right)), "tolerance": 0.0})
                for index, (a, b) in enumerate(zip(left, right)):
                    _compare_values(a, b, path + (str(index),), tolerances=tolerances,
                                    stats=stats, failures=failures, fixed_root=fixed_root,
                                    exact=exact, state_values=state_values)
                return
            raise ComparisonError(f"matched artifact array shape differs at {label}")
        for index, (a, b) in enumerate(zip(left, right)):
            array_category = _category_for_index(path, index, fixed_root)
            child_path = path + (str(index),)
            if array_category is None:
                _compare_values(a, b, child_path, tolerances=tolerances, stats=stats,
                                failures=failures, fixed_root=fixed_root, exact=exact,
                                state_values=state_values)
            else:
                _compare_numeric(a, b, child_path, array_category, exact=exact,
                                 tolerances=tolerances, stats=stats, failures=failures)
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and not isinstance(left, bool) and isinstance(right, int) and not isinstance(right, bool):
            if left != right:
                leaf = path[-1].lower() if path else ""
                if state_values and (_category(path) in {"neural", "contact"}
                                     or leaf in {"total_spikes", "target_spikes", "ncon"}
                                     or leaf.endswith("_count")):
                    _compare_numeric(left, right, path, _category(path), exact=True,
                                     tolerances=tolerances, stats=stats, failures=failures)
                    return
                raise ComparisonError(f"matched integer/control value differs at {label}: {left} vs {right}")
            if state_values:
                _compare_numeric(left, right, path, _category(path), exact=True,
                                 tolerances=tolerances, stats=stats, failures=failures)
            return
        _compare_numeric(left, right, path, _category(path), exact=exact,
                         tolerances=tolerances, stats=stats, failures=failures)
        return
    if type(left) is not type(right) or left != right:
        if state_values and "body_contacts" in path:
            failures.append({"path": label, "category": "contact", "reference": left,
                             "candidate": right, "absolute_error": None,
                             "tolerance": "exact runtime contact metadata"})
            return
        raise ComparisonError(f"matched artifact value differs at {label}: {left!r} vs {right!r}")


def _compare_numeric(left: Any, right: Any, path: tuple[str, ...], category: str, *,
                     exact: bool, tolerances: Mapping[str, float],
                     stats: dict[str, dict[str, Any]],
                     failures: list[dict[str, Any]]) -> None:
    label = ".".join(path)
    if isinstance(left, bool) or isinstance(right, bool):
        raise ComparisonError(f"numeric field type differs at {label}")
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        raise ComparisonError(f"numeric field type differs at {label}")
    a, b = float(left), float(right)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ComparisonError(f"non-finite numeric value at {label}")
    error = abs(a - b)
    if category not in stats:
        category = "other"
    item = stats[category]
    item["numeric_leaves"] += 1
    if error > item["max_abs_error"]:
        item["max_abs_error"] = error
        item["max_error_path"] = label
    limit = 0.0 if exact else float(tolerances[category])
    if error > limit:
        item["within_tolerance"] = False
        failures.append({"path": label, "category": category,
                         "reference": a, "candidate": b,
                         "absolute_error": error, "tolerance": limit})


def _validate_pair(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    _validate_role(reference, candidate=False)
    _validate_role(candidate, candidate=True)
    ref_summary, cand_summary = reference["summary"], candidate["summary"]
    ref_manifest, cand_manifest = reference["manifest"], candidate["manifest"]

    _require(reference["source_sha256"] == candidate["source_sha256"],
             "source_sha256 maps differ; modified-source comparisons are rejected")
    _require(reference["csr_sha256"] == candidate["csr_sha256"],
             "CSR dataset file hashes differ")
    _require(_json_equal(reference["pipeline"], candidate["pipeline"]),
             "pipeline/configuration/topology provenance differs between runs")
    _require(_json_equal(_engine_signature(reference["neural"]),
                         _engine_signature(candidate["neural"])),
             "neural model, dataset, topology, or integration controls differ")

    controls = (
        "case", "leg", "source", "control_authority", "fixed_root", "duration_s",
        NEURAL_DT_FIELD, "seed", "selector", "selected_neuron_count", "muscle_group",
        "selected_channel", "selected_body_ids", "intervention_window_s",
        "effective_intervention_window_s", "current_mV_per_ms", "prescribed_rate_hz",
        "full_neural_trace_ticks", "neural_observation_dtype", "neural_spike_shape",
        "neural_spike_encoding", "evidence_class", "biological_validation",
        "autonomous_behavior", "physics_backend", "max_vram_gib",
    )
    for key in controls:
        _require(key in ref_summary and key in cand_summary,
                 f"matched summary control {key!r} is missing")
        _require(_json_equal(ref_summary[key], cand_summary[key]),
                 f"matched control {key!r} differs between runs")
    for key in ("sensory_feedback_mode", "muscle_force_scale", "passive_support_enabled"):
        if key in ref_summary or key in cand_summary:
            _require(key in ref_summary and key in cand_summary
                     and _json_equal(ref_summary[key], cand_summary[key]),
                     f"matched control {key!r} differs between runs")
    _require(ref_manifest.get("csr_sha256") == cand_manifest.get("csr_sha256"),
             "CSR dataset hash manifests differ")

    manifest_ignored = {
        "summary", "neural", "pipeline", "source_sha256", "telemetry_sha256",
        "neural_counts_sha256", "csr_sha256", "generated_at", "created_at", "run_id",
        "neural_backend_preflight", "source_snapshot_directory",
        "sensory_feedback_sha256",
        "performance",
        "output_name", "run_name", "artifact_name", "output_dir", "out",
    }
    ref_extras = _without_keys(ref_manifest, manifest_ignored)
    cand_extras = _without_keys(cand_manifest, manifest_ignored)
    _require(_json_equal(ref_extras, cand_extras),
             "manifest contract/configuration fields differ between runs")

    return {
        "matched_controls": list(controls),
        "same_source_sha256": True,
        "same_csr_dataset_sha256": True,
        "same_pipeline_and_body_topology": True,
        "same_neural_topology_and_integration_controls": True,
        "full_dense_spike_trace_exactness_is_reported_as_a_numerical_result": True,
    }


def _compare_run_states(reference: Mapping[str, Any], candidate: Mapping[str, Any],
                        tolerances: Mapping[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats = _new_stats(tolerances)
    failures: list[dict[str, Any]] = []
    ref_summary, cand_summary = reference["summary"], candidate["summary"]
    fixed_root = bool(ref_summary["fixed_root"])

    # The full neural digest is the primary numerical gate. Count rows provide
    # an independently inspectable per-tick consistency check.
    _compare_values(reference["counts"], candidate["counts"], ("neural_spike_counts",),
                    tolerances=tolerances, stats=stats, failures=failures,
                    fixed_root=fixed_root, state_values=True)
    _compare_values(reference["feedback"], candidate["feedback"], ("sensory_feedback",),
                    tolerances=tolerances, stats=stats, failures=failures,
                    fixed_root=fixed_root, state_values=True)

    # Telemetry rows are sampled on the same 5 ms schedule. All fields are
    # checked recursively: keys, array shapes, booleans, strings, and integer
    # control values must match; continuous numeric leaves use declared limits.
    _compare_values(reference["telemetry"], candidate["telemetry"], ("telemetry",),
                    tolerances=tolerances, stats=stats, failures=failures,
                    fixed_root=fixed_root, state_values=True)

    for phase in ("initial", "final"):
        _compare_values(ref_summary[phase], cand_summary[phase], (phase,),
                        tolerances=tolerances, stats=stats, failures=failures,
                        fixed_root=fixed_root, state_values=True)

    # Derived outcome metrics are retained in summaries. Backend and wall
    # timing fields are handled separately; paths/output labels are ignored.
    ignored = {
        "backend", "gpu_used", "wall_s", "neural_backend_requested",
        "simulation_loop_wall_s", "initialization_wall_s",
        "neural_backend_actual", "neural_device_actual", "neural_spike_sha256",
        "initial", "final", "output_name", "run_name", "artifact_name", "output_dir", "out",
    }
    ref_outcomes = _without_keys(ref_summary, ignored)
    cand_outcomes = _without_keys(cand_summary, ignored)
    _compare_values(ref_outcomes, cand_outcomes, ("summary",), tolerances=tolerances,
                    stats=stats, failures=failures, fixed_root=fixed_root,
                    state_values=True)

    result = {
        category: {**values, "tolerance": tolerances[category]}
        for category, values in stats.items()
    }
    return result, failures


def compare_runs(reference_dir: str | Path, candidate_dir: str | Path, *,
                 tolerances: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Compare matched run artifacts without modifying either input directory.

    Raises :class:`ComparisonError` for corrupt or incompatible inputs.
    Numerical disagreement is represented in a returned ``status=FAILED``
    report so callers can persist and inspect the full comparison.
    """

    resolved_tolerances = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        unknown = set(tolerances) - set(resolved_tolerances)
        if unknown:
            raise ComparisonError(f"unknown tolerance categories: {sorted(unknown)}")
        for category, value in tolerances.items():
            resolved_tolerances[category] = _finite_number(value, f"tolerances.{category}")
            if resolved_tolerances[category] < 0:
                raise ComparisonError(f"tolerances.{category} must be nonnegative")

    reference = _load_run(Path(reference_dir), "reference")
    candidate = _load_run(Path(candidate_dir), "candidate")
    _require(reference["directory"] != candidate["directory"],
             "reference and candidate resolve to the same artifact directory")
    compatibility = _validate_pair(reference, candidate)
    numerical, failures = _compare_run_states(reference, candidate, resolved_tolerances)
    spike_equal = (reference["summary"]["neural_spike_sha256"]
                   == candidate["summary"]["neural_spike_sha256"])
    if not spike_equal:
        failures.insert(0, {
            "path": "summary.neural_spike_sha256", "category": "neural",
            "reference": reference["summary"]["neural_spike_sha256"],
            "candidate": candidate["summary"]["neural_spike_sha256"],
            "absolute_error": None, "tolerance": "exact equality",
        })
    counts_equal = all(
        ref["total_spikes"] == cand["total_spikes"]
        and ref["target_spikes"] == cand["target_spikes"]
        for ref, cand in zip(reference["counts"], candidate["counts"])
    )
    same_spikes = spike_equal and counts_equal
    wall_ref = reference["summary"].get("wall_s")
    wall_cand = candidate["summary"].get("wall_s")
    wall_ratio = None
    try:
        wall_ref = _finite_number(wall_ref, "reference wall_s", positive=True)
        wall_cand = _finite_number(wall_cand, "candidate wall_s", positive=True)
        wall_ratio = wall_ref / wall_cand
    except ComparisonError:
        wall_ref = wall_cand = None

    numerical_pass = not failures and same_spikes
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if numerical_pass else "FAILED",
        "report_scope": "engineering numerical agreement for matched digital simulations",
        "biological_validation": "NOT_VALIDATED",
        "performance_equivalence": False,
        "interpretation": (
            "A pass establishes numerical agreement for this matched artifact pair only. "
            "It does not validate biological behavior or establish performance equivalence."
        ),
        "compatibility": {"status": "MATCHED", "checks": compatibility},
        "reference": {
            "artifact_directory": str(reference["directory"]),
            "backend_requested": reference["summary"]["neural_backend_requested"],
            "backend_actual": reference["summary"]["neural_backend_actual"],
            "device_actual": reference["summary"]["neural_device_actual"],
            "source_sha256": reference["source_sha256"],
            "telemetry_sha256": reference["telemetry_sha256"],
            "neural_counts_sha256": reference["counts_sha256"],
        },
        "candidate": {
            "artifact_directory": str(candidate["directory"]),
            "backend_requested": candidate["summary"]["neural_backend_requested"],
            "backend_actual": candidate["summary"]["neural_backend_actual"],
            "device_actual": candidate["summary"]["neural_device_actual"],
            "source_sha256": candidate["source_sha256"],
            "telemetry_sha256": candidate["telemetry_sha256"],
            "neural_counts_sha256": candidate["counts_sha256"],
        },
        "neural_spike_trace": {
            "encoding": SPIKE_ENCODING,
            "dtype": "uint8",
            "shape": reference["summary"]["neural_spike_shape"],
            "ticks": reference["summary"]["full_neural_trace_ticks"],
            "reference_sha256": reference["summary"]["neural_spike_sha256"],
            "candidate_sha256": candidate["summary"]["neural_spike_sha256"],
            "exact_hash_match": spike_equal,
            "per_tick_spike_counts_match": counts_equal,
            "sha256_recomputed_from_counts": False,
        },
        "trajectory_comparison": {
            "telemetry_samples": len(reference["telemetry"]),
            "initial_final_physics_states_compared": ["initial", "final"],
            "max_absolute_error_by_category": numerical,
            "numerical_failures": failures,
        },
        "tolerances": resolved_tolerances,
        "wall_time_context_only": {
            "reference_wall_s": wall_ref,
            "candidate_wall_s": wall_cand,
            "reference_over_candidate_ratio": wall_ratio,
            "interpretation": "Context only; not a GPU speed or performance-equivalence claim.",
        },
    }
    return report


def _validate_output_path(out: Path, reference: Path, candidate: Path) -> Path:
    target = out.expanduser().resolve()
    for raw_directory in (reference, candidate):
        directory = raw_directory.expanduser().resolve()
        if target == directory or directory in target.parents:
            raise ComparisonError("--out must be outside both input artifact directories")
    if not target.parent.is_dir():
        raise ComparisonError(f"--out parent directory does not exist: {target.parent}")
    if target.exists():
        raise ComparisonError(f"--out already exists; refusing to overwrite: {target}")
    return target


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise ComparisonError(f"--out already exists; refusing to overwrite: {path}") from exc
    except OSError as exc:
        raise ComparisonError(f"cannot write comparison report {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="matched CPU reference artifact directory")
    parser.add_argument("candidate", type=Path, help="matched neural-device candidate artifact directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="new report JSON path (exclusive create; never overwritten)")
    args = parser.parse_args(argv)
    output: Path | None = None
    try:
        output = _validate_output_path(args.out, args.reference, args.candidate)
        report = compare_runs(args.reference, args.candidate)
        _write_exclusive(output, report)
    except ComparisonError as exc:
        if output is None:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        error_report = {
            "schema_version": SCHEMA_VERSION,
            "status": "INCOMPATIBLE",
            "report_scope": "engineering artifact compatibility check",
            "biological_validation": "NOT_VALIDATED",
            "error": str(exc),
        }
        try:
            _write_exclusive(output, error_report)
        except ComparisonError as write_exc:
            print(f"ERROR: {exc}; {write_exc}", file=sys.stderr)
            return 2
        print(f"INCOMPATIBLE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "report": str(output)}, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
