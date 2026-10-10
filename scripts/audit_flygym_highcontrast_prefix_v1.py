#!/usr/bin/env python3
"""Read-only exact scientific-prefix comparison of completed online captures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

FRAMES = 800
TICKS = 4000
NATIVE = 40001
WORKER_FIELDS = (
    "seq", "body_time_s", "time_s", "pattern_time_s", "render_time_s",
    "neural_frame_id", "vision_frame_id", "input_pose", "input_quaternion_wxyz",
    "neural_window_end_s", "neural_window_start_step_index",
    "neural_window_end_step_index", "population_rates_hz", "population_spike_counts",
    "population_report_count", "leg_power", "leg_steering",
    "left_descending_drive", "right_descending_drive", "eye_column_hashes",
    "visual_drive_hash", "visual_drive_mean", "visual_drive_std", "visual_drive_abs_mean",
    "flyvis_target_indices_sha256", "flyvis_target_units", "effective_backends",
    "diagnostic_neuron_vm_mv_by_csr_index", "diagnostic_population_vm_mv_mean",
    "diagnostic_population_vm_sample_count", "diagnostic_vm_sample_time_s",
    "diagnostic_vm_sample_offset_within_window_s", "diagnostic_vm_sample_semantics",
    "neural_intervention", "motor_output_source",
)
JSONL_SPECS = {
    "online-frames.jsonl": (FRAMES, None, {"worker_reply_sha256", "worker_request_sha256"}),
    "worker-requests.jsonl": (FRAMES, None, set()),
    "control-ticks.jsonl": (TICKS, None, set()),
    "worker/frames.jsonl": (FRAMES, WORKER_FIELDS, set()),
}
WORKER_NPZ = ("eye_columns.npz", "visual_drive.npz", "muscle_channels.npz", "neural_state_diagnostic.npz")
EXCLUDED_NPZ_METADATA = {"metadata_json", "provenance_json"}
NON_ACTION_TIMERS = {
    f"controller_telemetry.hind_clearance.per_leg.{leg}.hind_pose_priority.cpu_s"
    for leg in ("lh", "rh")
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return True


def strip_non_action_timers(value: Any, path: str) -> tuple[Any, set[str]]:
    removed: set[str] = set()
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if child_path in NON_ACTION_TIMERS:
                if not isinstance(item, (int, float)) or not np.isfinite(item) or item < 0:
                    raise ValueError(f"invalid non-action timing value: {child_path}")
                removed.add(child_path)
            else:
                clean[key], child_removed = strip_non_action_timers(item, child_path)
                removed.update(child_removed)
        return clean, removed
    return value, removed


def jsonl_prefix(candidate: Path, reference: Path, count: int,
                 fields: tuple[str, ...] | None, excluded: set[str]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    ignored: set[str] = set()
    checked = 0
    with candidate.open() as cf, reference.open() as rf:
        for index in range(count):
            cs, rs = cf.readline(), rf.readline()
            if not cs or not rs:
                mismatches.append({"row": index, "reason": "prefix missing"})
                break
            c, r = json.loads(cs), json.loads(rs)
            keys = list(fields) if fields is not None else sorted(set(r) - excluded)
            ignored.update(set(r) - set(keys))
            bad = []
            for key in keys:
                if key not in r or key not in c:
                    bad.append(key)
                    continue
                cv, cr = strip_non_action_timers(c[key], key)
                rv, rr = strip_non_action_timers(r[key], key)
                ignored.update(rr)
                if cr != rr or not finite_tree(rv) or not finite_tree(cv) or cv != rv:
                    bad.append(key)
            if bad:
                mismatches.append({"row": index, "fields": bad})
            checked += 1
    return {"passed": checked == count and not mismatches, "checked_rows": checked,
            "required_rows": count, "mismatch_count": len(mismatches),
            "first_mismatches": mismatches[:10], "excluded_reference_fields": sorted(ignored)}


def array_result(c: np.ndarray, r: np.ndarray) -> dict[str, Any]:
    finite = True
    if np.issubdtype(r.dtype, np.number):
        finite = bool(np.isfinite(c).all() and np.isfinite(r).all())
    exact = c.shape == r.shape and c.dtype == r.dtype and bool(np.array_equal(c, r))
    return {"passed": finite and exact, "finite": finite, "exact": exact,
            "reference_shape": list(r.shape), "candidate_prefix_shape": list(c.shape),
            "reference_dtype": str(r.dtype), "candidate_dtype": str(c.dtype)}


def npz_prefix(candidate: Path, reference: Path, mode: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    excluded: list[str] = []
    with np.load(candidate, allow_pickle=False) as cz, np.load(reference, allow_pickle=False) as rz:
        missing = sorted(set(rz.files) - set(cz.files))
        if mode == "contacts":
            ref_idx, cand_idx = rz["step_index"], cz["step_index"]
            if np.any(ref_idx > NATIVE - 1):
                raise ValueError("reference contacts extend beyond four seconds")
            mask = cand_idx <= NATIVE - 1
            if np.any(cand_idx < 0) or np.any(np.diff(cand_idx) < 0):
                raise ValueError("candidate contact indices are negative or unordered")
        for key in rz.files:
            if key not in cz.files:
                continue
            if mode == "worker" and key in EXCLUDED_NPZ_METADATA:
                excluded.append(key)
                continue
            r, c = rz[key], cz[key]
            if mode == "trajectory" and key != "body_mass_native":
                if r.ndim == 0 or r.shape[0] != NATIVE:
                    raise ValueError(f"unexpected reference trajectory dimension: {key}")
                c = c[:NATIVE]
            elif mode == "contacts":
                if r.ndim == 0 or r.shape[0] != ref_idx.size or c.shape[0] != cand_idx.size:
                    raise ValueError(f"unexpected contact dimension: {key}")
                c = c[mask]
            elif mode == "worker" and r.ndim and c.ndim and c.shape[0] != r.shape[0]:
                # Time-major captures grow; immutable cell/target axes do not.
                allowed_rows = {FRAMES, 40}
                if r.shape[0] not in allowed_rows or c.shape[0] < r.shape[0]:
                    raise ValueError(f"unexpected changed worker axis: {key}: {r.shape} -> {c.shape}")
                c = c[:r.shape[0]]
            results[key] = array_result(c, r)
    return {"passed": not missing and all(x["passed"] for x in results.values()),
            "missing_reference_keys": missing, "excluded_metadata_keys": excluded,
            "arrays": results}


def audit(candidate: Path, reference: Path) -> dict[str, Any]:
    c_run, r_run = read_json(candidate / "run-record.json"), read_json(reference / "run-record.json")
    checks: dict[str, Any] = {}
    checks["completed_runs"] = {"passed": c_run.get("status") == r_run.get("status") == "complete"}
    checks["reference_counts"] = {"passed": r_run.get("control_tick_count") == TICKS
                                   and r_run.get("physics_step_count") == NATIVE - 1}
    if not all(x["passed"] for x in checks.values()):
        return {"schema_version": "flygym-highcontrast-prefix-audit-v1", "passed": False, "checks": checks}
    for name, (count, fields, excluded) in JSONL_SPECS.items():
        checks[name] = jsonl_prefix(candidate / name, reference / name, count, fields, excluded)
    for name, mode in (("trajectory.npz", "trajectory"), ("contacts.npz", "contacts")):
        checks[name] = npz_prefix(candidate / name, reference / name, mode)
    for name in WORKER_NPZ:
        checks[f"worker/{name}"] = npz_prefix(candidate / "worker" / name, reference / "worker" / name, "worker")
    snapshots = sorted((reference / "worker/debug-neural-state").glob("*.npz"))
    snap_results = {}
    for path in snapshots:
        target = candidate / "worker/debug-neural-state" / path.name
        snap_results[path.name] = {"passed": target.is_file() and sha256(target) == sha256(path)}
    checks["full_neural_debug_prefix"] = {"passed": len(snapshots) == 41 and all(x["passed"] for x in snap_results.values()),
                                          "reference_snapshots": len(snapshots), "snapshots": snap_results}
    return {"schema_version": "flygym-highcontrast-prefix-audit-v1", "passed": all(x["passed"] for x in checks.values()),
            "candidate_run": str(candidate), "reference_run": str(reference), "checks": checks,
            "run_record_sha256": {"candidate": sha256(candidate / "run-record.json"), "reference": sha256(reference / "run-record.json")},
            "evidence_boundary": "Exact first 800 neural/retinal/decoder frames, 4000 action ticks, 40001 native samples and their contacts; no conclusion about later dynamics or biological matching.",
            "excluded_metadata_reason": "Artifact paths, case labels, reply hashes, capture metadata and two hind IK performance timers differ across runs; scientific fields and arrays are compared directly."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.candidate_run.resolve(strict=True), args.reference_run.resolve(strict=True))
        code = 0 if result["passed"] else 2
    except Exception as exc:
        result = {"schema_version": "flygym-highcontrast-prefix-audit-v1", "passed": False,
                  "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "exit_code": code}))
    return code


if __name__ == "__main__":
    sys.exit(main())
