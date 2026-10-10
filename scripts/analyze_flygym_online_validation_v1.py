#!/usr/bin/env python3
"""Read-only mechanics summary for the 12-case online validation cohort.

This script reads only completed, integrity-checked run artifacts. It does not
create a FlyGym session, advance physics, start a neural worker, or use a GPU.
Incomplete cases are emitted as pending with null metrics; they are never
converted to numerical zeros.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True

import numpy as np

DEFAULT_BUNDLE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/online-validation-v1"
)
PROJECT = Path(__file__).resolve().parents[1]
POSITIVE_CONTROL_ANALYSIS = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/free-body-stripes/"
    "contact-criterion-reference-analysis.json"
)
POSITIVE_CONTROL_REVIEW = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/free-body-stripes/"
    "contact-criterion-reference-review.md"
)
STRICT_EVENT_ANALYZER = PROJECT / "scripts/analyze_flygym_free_body_trace_v1.py"

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
WINDOWS = {
    "early_1_1p5_s": (1.0, 1.5),
    "late_2_4_s": (2.0, 4.0),
    "stimulus_1_4_s": (1.0, 4.0),
}
CORE_FILES = (
    "trajectory.npz",
    "contacts.npz",
    "array_order.json",
    "summary.json",
    "effective_config.json",
    "initial_state.json",
)
INITIAL_STATE_HASH_FIELDS = (
    "qpos_sha256",
    "qvel_sha256",
    "cpg_phase_rad_sha256",
    "cpg_magnitude_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verify_hash_record(path: Path, record: dict[str, Any], index_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{index_name} lists missing file: {path}")
    expected_sha = record.get("sha256")
    expected_size = record.get("bytes", record.get("size_bytes"))
    if expected_sha is None or expected_size is None:
        raise ValueError(f"{index_name} has an incomplete file record: {path}")
    actual_sha = sha256(path)
    actual_size = path.stat().st_size
    if actual_sha != expected_sha or actual_size != int(expected_size):
        raise ValueError(f"{index_name} hash/size mismatch for {path}")
    return {"index": index_name, "sha256": actual_sha, "size_bytes": actual_size}


def _lookup_absolute_record(records: dict[str, Any], path: Path) -> dict[str, Any] | None:
    resolved = str(path.resolve())
    candidate = records.get(resolved)
    if isinstance(candidate, dict):
        return candidate
    for raw_path, value in records.items():
        if not isinstance(value, dict):
            continue
        if value.get("path") and str(Path(value["path"]).resolve()) == resolved:
            return value
        if str(Path(raw_path).resolve()) == resolved:
            return value
    return None


def _verify_core_outputs(
    run_dir: Path,
    manifest: dict[str, Any],
    integrity: dict[str, Any],
    online: dict[str, Any],
) -> dict[str, Any]:
    manifest_records = manifest.get("output_files")
    integrity_records = integrity.get("files")
    online_records = online.get("files")
    if not isinstance(manifest_records, dict) or not isinstance(integrity_records, dict) or not isinstance(online_records, dict):
        raise ValueError("Run integrity indexes have an unexpected schema")

    verified: dict[str, Any] = {}
    for filename in CORE_FILES:
        path = run_dir / filename
        checks: list[dict[str, Any]] = []
        manifest_record = manifest_records.get(filename)
        if not isinstance(manifest_record, dict):
            raise ValueError(f"Body manifest omits required output {filename}")
        checks.append(_verify_hash_record(path, manifest_record, "manifest.output_files"))

        integrity_record = _lookup_absolute_record(integrity_records, path)
        if integrity_record is None:
            raise ValueError(f"integrity.json omits required output {filename}")
        checks.append(_verify_hash_record(path, integrity_record, "integrity.json"))

        online_record = _lookup_absolute_record(online_records, path)
        if online_record is None:
            raise ValueError(f"online-integrity.json omits required output {filename}")
        checks.append(_verify_hash_record(path, online_record, "online-integrity.json"))

        hashes = {record["sha256"] for record in checks}
        sizes = {record["size_bytes"] for record in checks}
        if len(hashes) != 1 or len(sizes) != 1:
            raise ValueError(f"Run integrity indexes disagree for {filename}")
        verified[filename] = {
            "sha256": next(iter(hashes)),
            "size_bytes": next(iter(sizes)),
            "indexes_checked": [record["index"] for record in checks],
        }

    for filename in ("run-record.json", "window-summary.json"):
        path = run_dir / filename
        record = _lookup_absolute_record(online_records, path)
        if record is None:
            raise ValueError(f"online-integrity.json omits required output {filename}")
        verified[filename] = _verify_hash_record(path, record, "online-integrity.json")
    return verified


def _expected_case_map(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    conditions = protocol.get("stimuli", {}).get("conditions")
    queue_order = protocol.get("queue_order")
    if not isinstance(conditions, list) or not isinstance(queue_order, list):
        raise ValueError("Protocol lacks the registered case list or queue order")
    by_id = {str(row.get("case_id")): row for row in conditions if isinstance(row, dict)}
    if len(by_id) != len(conditions) or set(by_id) != set(map(str, queue_order)):
        raise ValueError("Protocol cases and queue_order do not form the same unique case set")
    if len(by_id) != 12:
        raise ValueError(f"Expected the registered 12 cases, found {len(by_id)}")
    return [by_id[str(case_id)] for case_id in queue_order], list(map(str, protocol.get("cohort_ids", [])))


def _case_record_matches_protocol(record: dict[str, Any], case: dict[str, Any]) -> list[str]:
    failures = []
    direct = {
        "case_id": case["case_id"],
        "cohort_id": case["cohort_id"],
        "body_seed": case["body_seed"],
        "neural_seed": case["neural_seed"],
        "stimulus_direction": case["direction"],
    }
    for key, expected in direct.items():
        if record.get(key) != expected:
            failures.append(f"run-record.{key} expected {expected!r}, got {record.get(key)!r}")
    try:
        if not math.isclose(
            float(record.get("speed_magnitude_deg_s")),
            float(case["speed_magnitude_deg_s"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            failures.append("run-record.speed_magnitude_deg_s differs from protocol")
    except (TypeError, ValueError):
        failures.append("run-record.speed_magnitude_deg_s is missing or nonnumeric")
    return failures


def _empty_pending(case: dict[str, Any], run_dir: Path, reason: str, **extra: Any) -> dict[str, Any]:
    result = {
        "case_id": case["case_id"],
        "cohort_id": case["cohort_id"],
        "direction": case["direction"],
        "body_seed": case["body_seed"],
        "neural_seed": case["neural_seed"],
        "speed_magnitude_deg_s": case["speed_magnitude_deg_s"],
        "head_direction_sign": case["head_direction_sign"],
        "run_path": str(run_dir),
        "status": "pending",
        "pending_reason": reason,
        "metrics": None,
    }
    result.update(extra)
    return result


def _posture_and_authority(arrays: Any, thorax_index: int) -> dict[str, Any]:
    q = np.asarray(arrays["body_xquat_wxyz"][:, thorax_index], dtype=np.float64)
    qnorm = np.linalg.norm(q, axis=1)
    if not np.all(np.isfinite(qnorm)) or np.any(qnorm <= 0.0):
        raise ValueError("c_thorax quaternion has a nonfinite or zero norm")
    q = q / qnorm[:, None]
    w, x, y, z = q.T
    yaw_q = np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

    yaw_stored = np.unwrap(np.asarray(arrays["thorax_heading_yaw_rad"], dtype=np.float64))
    heading_xy = np.asarray(arrays["thorax_heading_xy"], dtype=np.float64)
    yaw_xy = np.unwrap(np.arctan2(heading_xy[:, 1], heading_xy[:, 0]))
    xy_error = np.abs(np.angle(np.exp(1j * (yaw_xy - yaw_stored))))
    quat_error = np.abs(np.angle(np.exp(1j * (yaw_q - yaw_stored))))

    thorax_z = np.asarray(arrays["body_xpos_mm"][:, thorax_index, 2], dtype=np.float64)
    clearance = np.asarray(arrays["thorax_collision_clearance_min_mm"], dtype=np.float64)
    thorax_contact = np.asarray(arrays["thorax_ground_contact"], dtype=bool)
    nonleg_contact = np.asarray(arrays["nonleg_ground_contact"], dtype=bool)
    max_abs_roll = float(np.degrees(np.max(np.abs(roll))))
    max_abs_pitch = float(np.degrees(np.max(np.abs(pitch))))
    min_clearance = float(np.min(clearance))
    any_body_contact = bool(np.any(thorax_contact) or np.any(nonleg_contact))
    inverted = bool(max_abs_roll >= 90.0 or max_abs_pitch >= 90.0)
    clearance_nonpositive = bool(min_clearance <= 0.0)

    authority_names = (
        "root_qpos_write_count",
        "root_qvel_write_count",
        "root_external_wrench_native",
        "body_xfrc_applied_native",
        "qfrc_applied_native",
    )
    authority = {name: float(np.max(np.abs(arrays[name]))) for name in authority_names}
    return {
        "max_abs_roll_deg": max_abs_roll,
        "max_abs_pitch_deg": max_abs_pitch,
        "thorax_center_z_min_mm": float(np.min(thorax_z)),
        "thorax_clearance_min_mm": min_clearance,
        "thorax_ground_contact_samples": int(np.count_nonzero(thorax_contact)),
        "nonleg_ground_contact_samples": int(np.count_nonzero(nonleg_contact)),
        "heading_xy_norm_min": float(np.min(np.linalg.norm(heading_xy, axis=1))),
        "quaternion_norm_min": float(np.min(qnorm)),
        "stored_yaw_vs_xy_projection_max_error_deg": float(np.degrees(np.max(xy_error))),
        "stored_yaw_vs_quaternion_max_error_deg": float(np.degrees(np.max(quat_error))),
        "root_or_body_authority_max_abs": authority,
        "root_state_write_counts_zero": authority["root_qpos_write_count"] == 0.0 and authority["root_qvel_write_count"] == 0.0,
        "external_and_generalized_applied_forces_zero": all(
            authority[name] == 0.0
            for name in ("root_external_wrench_native", "body_xfrc_applied_native", "qfrc_applied_native")
        ),
        "fall_screen": {
            "status": "contact_or_inversion_observed" if (any_body_contact or inverted or clearance_nonpositive) else "no_contact_or_inversion_flags_observed",
            "any_body_contact": any_body_contact,
            "orientation_inversion_ge_90deg": inverted,
            "nonpositive_thorax_clearance": clearance_nonpositive,
            "screen_scope": "geometric contact, nonpositive thorax clearance, or |Euler roll/pitch| >= 90 deg; descriptive screen only",
        },
    }


def _trajectory_shape_schema(
    order: dict[str, Any], manifest: dict[str, Any], expected_n: int,
) -> tuple[dict[str, tuple[int, ...]], int, int, int]:
    """Build the complete v1 trajectory schema from the saved model ordering.

    Every trajectory field is checked, including fields not used in the
    summary. ``body_mass_native`` is the sole static field; every other field
    has exactly ``expected_n`` native samples on axis zero.
    """
    body_names = order.get("body_names")
    body_ids = order.get("body_ids")
    leg_order = order.get("leg_order")
    joint_order = order.get("joint_order")
    native_joint_names = order.get("native_joint_names")
    qpos_addresses = order.get("native_joint_qposadr")
    dof_addresses = order.get("native_joint_dofadr")
    position_actuator_order = order.get("position_actuator_joint_order")
    native_actuator_names = order.get("native_actuator_names")
    thorax_collision_names = order.get("thorax_collision_geom_names")
    geom_names = order.get("geom_names_by_id")

    list_fields = {
        "body_names": body_names,
        "body_ids": body_ids,
        "leg_order": leg_order,
        "joint_order": joint_order,
        "native_joint_names": native_joint_names,
        "native_joint_qposadr": qpos_addresses,
        "native_joint_dofadr": dof_addresses,
        "position_actuator_joint_order": position_actuator_order,
        "native_actuator_names": native_actuator_names,
        "thorax_collision_geom_names": thorax_collision_names,
        "geom_names_by_id": geom_names,
    }
    invalid = [name for name, value in list_fields.items() if not isinstance(value, list) or not value]
    if invalid:
        raise ValueError(f"array_order.json has missing or empty schema lists: {invalid}")
    if not all(isinstance(name, str) and name for name in body_names):
        raise ValueError("array_order.json body_names must contain nonempty strings")
    if len(set(body_names)) != len(body_names) or len(set(body_ids)) != len(body_ids):
        raise ValueError("array_order.json body names and IDs must each be unique")
    body_count = len(body_names)
    leg_count = len(leg_order)
    if leg_order not in (list(LEGS), [leg.lower() for leg in LEGS]):
        raise ValueError(f"unexpected leg order: {leg_order!r}")
    if len(set(leg_order)) != len(leg_order):
        raise ValueError("array_order.json leg_order contains duplicates")
    if not all(isinstance(name, str) for name in joint_order + native_joint_names + position_actuator_order):
        raise ValueError("array_order.json joint and actuator orders must contain strings")
    if not all(isinstance(name, str) and name for name in thorax_collision_names + geom_names):
        raise ValueError("array_order.json geometry orders must contain nonempty strings")
    if len(set(geom_names)) != len(geom_names) or len(set(thorax_collision_names)) != len(thorax_collision_names):
        raise ValueError("array_order.json geometry orders contain duplicate names")
    if len(native_joint_names) != len(qpos_addresses) or len(native_joint_names) != len(dof_addresses):
        raise ValueError("array_order.json native joint address arrays do not match native_joint_names")
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in body_ids + qpos_addresses + dof_addresses):
        raise ValueError("array_order.json body IDs and native joint addresses must be integers")

    if manifest.get("body_names") != body_names or manifest.get("body_ids") != body_ids:
        raise ValueError("body manifest ordering differs from array_order.json")
    if manifest.get("thorax_clearance_geom_names") != thorax_collision_names:
        raise ValueError("body manifest thorax clearance geometry differs from array_order.json")
    mass_by_body = order.get("body_mass_native_by_body")
    manifest_mass_by_body = manifest.get("body_mass_native_by_body")
    if not isinstance(mass_by_body, dict) or not isinstance(manifest_mass_by_body, dict):
        raise ValueError("array_order.json or body manifest lacks body_mass_native_by_body")
    if set(mass_by_body) != set(body_names) or mass_by_body != manifest_mass_by_body:
        raise ValueError("body mass map keys/order metadata disagree with body_names or manifest")
    if not all(math.isfinite(float(mass_by_body[name])) and float(mass_by_body[name]) > 0 for name in body_names):
        raise ValueError("body_mass_native_by_body contains invalid masses")

    effective_options = manifest.get("effective_model_options")
    if not isinstance(effective_options, dict):
        raise ValueError("body manifest lacks effective_model_options")
    try:
        nq = int(effective_options["nq"])
        nv = int(effective_options["nv"])
        nbody = int(effective_options["nbody"])
        ngeom = int(effective_options["ngeom"])
        njnt = int(effective_options["njnt"])
        nu = int(effective_options["nu"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("body manifest effective_model_options lacks model dimension counts") from exc
    if min(nq, nv, nbody, ngeom, njnt, nu) <= 0:
        raise ValueError("body manifest has invalid model dimension counts")
    if body_count != nbody - 1 or set(body_ids) != set(range(1, nbody)):
        raise ValueError("array_order.json body ordering does not cover all non-world manifest bodies")
    if len(geom_names) != ngeom or len(native_joint_names) != njnt:
        raise ValueError("array_order.json geometry or joint order length differs from manifest model dimensions")
    if len(native_actuator_names) != nu or len(position_actuator_order) + len(leg_order) != nu:
        raise ValueError("array_order actuator order lengths differ from manifest nu")
    if not set(thorax_collision_names).issubset(set(geom_names)):
        raise ValueError("thorax clearance geometries are absent from geom_names_by_id")
    if any(address < 0 or address >= nq for address in qpos_addresses):
        raise ValueError("native_joint_qposadr contains an address outside manifest nq")
    if any(address < 0 or address >= nv for address in dof_addresses):
        raise ValueError("native_joint_dofadr contains an address outside manifest nv")

    joint_count = len(joint_order)
    position_actuator_count = len(position_actuator_order)
    clearance_geom_count = len(thorax_collision_names)
    shapes: dict[str, tuple[int, ...]] = {
        "time_s": (expected_n,),
        "qpos": (expected_n, nq),
        "qvel": (expected_n, nv),
        "joint_angle_rad": (expected_n, joint_count),
        "joint_velocity_rad_s": (expected_n, joint_count),
        "body_xpos_mm": (expected_n, body_count, 3),
        "body_xquat_wxyz": (expected_n, body_count, 4),
        "body_xvel_mm_s": (expected_n, body_count, 3),
        "body_xangvel_rad_s": (expected_n, body_count, 3),
        "body_xipos_mm": (expected_n, body_count, 3),
        "body_xfrc_applied_native": (expected_n, body_count, 6),
        "qfrc_applied_native": (expected_n, nv),
        "com_position_mm": (expected_n, 3),
        "com_velocity_mm_s": (expected_n, 3),
        "thorax_heading_xy": (expected_n, 2),
        "thorax_heading_yaw_rad": (expected_n,),
        "tarsus5_xpos_mm": (expected_n, leg_count, 3),
        "tarsus5_z_mm": (expected_n, leg_count),
        "foot_contact_found": (expected_n, leg_count),
        "foot_contact_force_local_native": (expected_n, leg_count, 3),
        "foot_contact_torque_local_native": (expected_n, leg_count, 3),
        "foot_contact_pos_mm": (expected_n, leg_count, 3),
        "foot_contact_normal_world": (expected_n, leg_count, 3),
        "foot_contact_tangent_world": (expected_n, leg_count, 3),
        "foot_contact_fz_up_native": (expected_n, leg_count),
        "thorax_collision_geom_clearance_mm": (expected_n, clearance_geom_count),
        "thorax_collision_clearance_min_mm": (expected_n,),
        "position_actuator_ctrl_rad": (expected_n, position_actuator_count),
        "position_actuator_force_native": (expected_n, position_actuator_count),
        "adhesion_actuator_ctrl": (expected_n, leg_count),
        "adhesion_actuator_force_native": (expected_n, leg_count),
        "cpg_phase_rad": (expected_n, leg_count),
        "cpg_magnitude": (expected_n, leg_count),
        "descending_drive_left_right": (expected_n, 2),
        "leg_stance_command": (expected_n, leg_count),
        "root_qpos_write_count": (expected_n,),
        "root_qvel_write_count": (expected_n,),
        "root_external_wrench_native": (expected_n, 6),
        "nonleg_ground_contact": (expected_n,),
        "thorax_ground_contact": (expected_n,),
        # Static model metadata: intentionally has no time/sample axis.
        "body_mass_native": (body_count,),
    }
    return shapes, body_count, leg_count, clearance_geom_count, len(geom_names)


def _validate_trajectory_shapes(arrays: Any, shapes: dict[str, tuple[int, ...]]) -> None:
    actual_keys = set(arrays.files)
    expected_keys = set(shapes)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(f"trajectory.npz schema key mismatch: missing={missing}, unexpected={extra}")
    for name, expected_shape in shapes.items():
        values = np.asarray(arrays[name])
        if values.shape != expected_shape:
            axis_note = "static field" if name == "body_mass_native" else "native per-sample array"
            raise ValueError(
                f"trajectory.npz {name} shape {values.shape} does not match {axis_note} schema {expected_shape}"
            )
        if values.dtype.kind not in "fiu":
            raise ValueError(f"trajectory.npz {name} has nonnumeric dtype {values.dtype}")
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise ValueError(f"trajectory.npz {name} contains nonfinite values")


def _validate_contacts_archive(path: Path, time_grid: np.ndarray, geom_count: int) -> int:
    """Validate the variable-length contact event table without assuming M=N."""
    expected_n = len(time_grid)
    with np.load(path, allow_pickle=False) as contacts:
        expected_keys = {
            "step_index", "time_s", "geom1_id", "geom2_id", "position_mm", "normal_world",
            "force_world_on_geom2_native", "force_world_on_fly_native", "is_ground_contact",
            "fly_body_segment", "leg_index",
        }
        actual_keys = set(contacts.files)
        if actual_keys != expected_keys:
            raise ValueError(
                f"contacts.npz schema key mismatch: missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        step_index = np.asarray(contacts["step_index"])
        if step_index.ndim != 1:
            raise ValueError(f"contacts.npz step_index must have shape (M,), got {step_index.shape}")
        row_count = int(step_index.shape[0])
        expected_shapes = {
            "step_index": (row_count,), "time_s": (row_count,), "geom1_id": (row_count,),
            "geom2_id": (row_count,), "position_mm": (row_count, 3), "normal_world": (row_count, 3),
            "force_world_on_geom2_native": (row_count, 3), "force_world_on_fly_native": (row_count, 3),
            "is_ground_contact": (row_count,), "fly_body_segment": (row_count,), "leg_index": (row_count,),
        }
        for name, shape in expected_shapes.items():
            values = np.asarray(contacts[name])
            if values.shape != shape:
                raise ValueError(f"contacts.npz {name} shape {values.shape} does not match event-table schema {shape}")
        if step_index.dtype.kind not in "iu":
            raise ValueError("contacts.npz step_index must use an integer dtype")
        if row_count and (np.any(step_index < 0) or np.any(step_index >= expected_n)):
            raise ValueError("contacts.npz step_index is outside the native trajectory")
        for name in ("time_s", "position_mm", "normal_world", "force_world_on_geom2_native", "force_world_on_fly_native"):
            values = np.asarray(contacts[name])
            if values.dtype.kind != "f" or not np.all(np.isfinite(values)):
                raise ValueError(f"contacts.npz {name} must contain finite floating-point values")
        if row_count and not np.allclose(
            np.asarray(contacts["time_s"], dtype=np.float64),
            np.asarray(time_grid, dtype=np.float64)[step_index],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("contacts.npz event times do not match their native step_index values")
        for name in ("geom1_id", "geom2_id"):
            values = np.asarray(contacts[name])
            if values.dtype.kind not in "iu" or (row_count and (np.any(values < 0) or np.any(values >= geom_count))):
                raise ValueError(f"contacts.npz {name} has invalid geometry IDs")
        if contacts["fly_body_segment"].dtype.kind not in "US":
            raise ValueError("contacts.npz fly_body_segment must use a fixed-width string dtype")
        return row_count


def _window_metrics(
    arrays: Any,
    t: np.ndarray,
    yaw: np.ndarray,
    thorax_index: int,
    body_weight: float,
    window_name: str,
    bounds: tuple[float, float],
) -> dict[str, Any]:
    start, end = bounds
    if not (t[0] <= start < end <= t[-1] + 1e-9):
        raise ValueError(f"Requested window {window_name} lies outside captured time")
    # Occupancy uses half-open windows to avoid double-counting shared endpoints.
    mask = (t >= start) & (t < end)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Window {window_name} has fewer than two native samples")
    fz = np.asarray(arrays["foot_contact_fz_up_native"][mask], dtype=np.float64)
    threshold = 0.01 * body_weight
    loaded = fz >= threshold
    support_count = loaded.sum(axis=1)
    support_values, support_counts = np.unique(support_count, return_counts=True)

    heading0 = float(np.interp(start, t, yaw))
    heading1 = float(np.interp(end, t, yaw))
    thorax_xy = np.asarray(arrays["body_xpos_mm"][:, thorax_index, :2], dtype=np.float64)
    xy0 = np.asarray([np.interp(start, t, thorax_xy[:, axis]) for axis in range(2)])
    xy1 = np.asarray([np.interp(end, t, thorax_xy[:, axis]) for axis in range(2)])
    delta = xy1 - xy0
    forward = np.asarray([math.cos(heading0), math.sin(heading0)])
    left = np.asarray([-math.sin(heading0), math.cos(heading0)])
    return {
        "window_s": [start, end],
        "sample_interval": "[start,end)",
        "native_sample_count": int(np.count_nonzero(mask)),
        "native_dt_s": float(np.median(np.diff(t))),
        "yaw_start_native_deg": float(np.degrees(heading0)),
        "yaw_end_native_deg": float(np.degrees(heading1)),
        "yaw_net_native_deg": float(np.degrees(heading1 - heading0)),
        "thorax_world_dx_mm": float(delta[0]),
        "thorax_world_dy_mm": float(delta[1]),
        "thorax_forward_delta_start_heading_mm": float(np.dot(delta, forward)),
        "thorax_lateral_left_delta_start_heading_mm": float(np.dot(delta, left)),
        "body_weight_native": body_weight,
        "per_leg_loaded_threshold_native": threshold,
        "minimum_loaded_legs": int(support_count.min()),
        "mean_loaded_legs": float(support_count.mean()),
        "fraction_samples_with_at_least_two_loaded_legs": float(np.mean(support_count >= 2)),
        "support_count_distribution": {str(int(k)): int(v) for k, v in zip(support_values, support_counts)},
    }


def analyze_case(case: dict[str, Any], run_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    record_path = run_dir / "run-record.json"
    if not run_dir.is_dir():
        return _empty_pending(case, run_dir, "run_directory_missing")
    if not record_path.is_file():
        return _empty_pending(case, run_dir, "run_record_missing")
    try:
        record = read_json(record_path)
    except Exception as exc:
        return _empty_pending(case, run_dir, "run_record_unreadable", diagnostic_error=str(exc))
    execution_status = record.get("status", "unknown")
    if execution_status != "complete":
        progress = {
            key: record.get(key)
            for key in ("completed_neural_frames", "completed_body_ticks", "control_tick_count", "native_sample_count")
            if key in record
        }
        return _empty_pending(
            case, run_dir, "execution_not_complete", execution_status=execution_status,
            progress=progress, execution_error=record.get("error"),
        )

    mismatches = _case_record_matches_protocol(record, case)
    if mismatches:
        return _empty_pending(case, run_dir, "run_record_protocol_mismatch", execution_status=execution_status, mismatches=mismatches)

    manifest_path = run_dir / "manifest.json"
    integrity_path = run_dir / "integrity.json"
    online_integrity_path = run_dir / "online-integrity.json"
    summary_path = run_dir / "summary.json"
    window_summary_path = run_dir / "window-summary.json"
    required = (manifest_path, integrity_path, online_integrity_path, summary_path, window_summary_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return _empty_pending(case, run_dir, "completion_artifacts_missing", execution_status=execution_status, missing_files=missing)

    try:
        manifest = read_json(manifest_path)
        integrity = read_json(integrity_path)
        online = read_json(online_integrity_path)
        summary = read_json(summary_path)
        window_summary = read_json(window_summary_path)
        if (
            manifest.get("status") != "complete"
            or summary.get("status") != "complete"
            or window_summary.get("status") != "complete"
            or online.get("status") != "complete"
        ):
            return _empty_pending(case, run_dir, "completion_manifest_status_incomplete", execution_status=execution_status, statuses={
                "body_manifest": manifest.get("status"),
                "summary": summary.get("status"),
                "window_summary": window_summary.get("status"),
                "online_integrity": online.get("status"),
            })
        if not integrity.get("files") or not online.get("files"):
            return _empty_pending(case, run_dir, "completion_integrity_index_empty", execution_status=execution_status)
        core_hashes = _verify_core_outputs(run_dir, manifest, integrity, online)
        if int(record.get("completed_neural_frames", -1)) != int(protocol["body"]["frame_count"]):
            raise ValueError("completed_neural_frames differs from protocol")
        if int(record.get("completed_body_ticks", -1)) != int(protocol["body"]["expected_control_ticks_per_case"]):
            raise ValueError("completed_body_ticks differs from protocol")
        if int(record.get("native_sample_count", -1)) != int(protocol["body"]["expected_native_samples_per_case"]):
            raise ValueError("native_sample_count differs from protocol")
        if int(record.get("physics_step_count", -1)) != int(protocol["body"]["expected_physics_steps_per_case"]):
            raise ValueError("physics_step_count differs from protocol")
        if not math.isclose(float(manifest.get("native_timestep_s", math.nan)), float(protocol["body"]["native_dt_s"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("body manifest native timestep differs from protocol")
        if int(manifest.get("seed", -1)) != int(case["body_seed"]):
            raise ValueError("body manifest seed differs from protocol")
        if int(summary.get("native_sample_count", -1)) != int(protocol["body"]["expected_native_samples_per_case"]):
            raise ValueError("summary native sample count differs from protocol")

        order = read_json(run_dir / "array_order.json")
        body_names = order.get("body_names", [])
        if body_names.count("c_thorax") != 1:
            raise ValueError("array_order.json does not identify exactly one c_thorax")
        thorax_index = body_names.index("c_thorax")
        expected_n = int(protocol["body"]["expected_native_samples_per_case"])
        trajectory_shapes, body_count, leg_count, clearance_geom_count, geom_count = _trajectory_shape_schema(
            order, manifest, expected_n,
        )

        with np.load(run_dir / "trajectory.npz", allow_pickle=False) as arrays:
            _validate_trajectory_shapes(arrays, trajectory_shapes)
            t = np.asarray(arrays["time_s"], dtype=np.float64)
            if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
                raise ValueError(f"invalid time grid: samples={len(t)}")
            native_dt = float(np.median(np.diff(t)))
            if not np.allclose(np.diff(t), float(protocol["body"]["native_dt_s"]), rtol=0.0, atol=1e-10):
                raise ValueError("trajectory native time grid differs from protocol")
            if not math.isclose(float(t[0]), 0.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(float(t[-1]), float(protocol["body"]["duration_s"]), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("trajectory does not span the protocol body duration")
            body_pos = np.asarray(arrays["body_xpos_mm"], dtype=np.float64)
            body_quat = np.asarray(arrays["body_xquat_wxyz"], dtype=np.float64)
            heading_xy = np.asarray(arrays["thorax_heading_xy"], dtype=np.float64)
            yaw_raw = np.asarray(arrays["thorax_heading_yaw_rad"], dtype=np.float64)
            fz = np.asarray(arrays["foot_contact_fz_up_native"], dtype=np.float64)
            mass_by_body = order["body_mass_native_by_body"]
            expected_mass = np.asarray([mass_by_body[name] for name in body_names], dtype=np.float64)
            if not np.array_equal(np.asarray(arrays["body_mass_native"], dtype=np.float64), expected_mass):
                raise ValueError("static body_mass_native differs from array_order.json body order/mass map")
            contact_row_count = _validate_contacts_archive(run_dir / "contacts.npz", t, geom_count)

            yaw = np.unwrap(yaw_raw.copy())
            posture = _posture_and_authority(arrays, thorax_index)
            body_weight = float(np.sum(np.asarray(arrays["body_mass_native"], dtype=np.float64)) * 9810.0)
            summary_weight = float(summary.get("body_weight_native", math.nan))
            if not math.isclose(body_weight, summary_weight, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("body_weight_native differs between trajectory and summary")
            window_metrics = {
                name: _window_metrics(arrays, t, yaw, thorax_index, body_weight, name, bounds)
                for name, bounds in WINDOWS.items()
            }
            metrics = {
                "windows": window_metrics,
                "whole_trace_posture_and_authority": posture,
                "source_integrity": {
                    "core_files_verified_against_manifest_integrity_and_online_integrity": core_hashes,
                    "manifest_sha256": sha256(manifest_path),
                    "integrity_json_sha256": sha256(integrity_path),
                    "online_integrity_json_sha256": sha256(online_integrity_path),
                    "run_record_sha256": sha256(record_path),
                },
                "initial_state_hashes": {
                    field: record.get("initial_state", {}).get(field)
                    for field in INITIAL_STATE_HASH_FIELDS
                },
                "validated_counts": {
                    "neural_frames": int(record["completed_neural_frames"]),
                    "control_ticks": int(record["completed_body_ticks"]),
                    "physics_steps": int(record["physics_step_count"]),
                    "native_samples": int(len(t)),
                    "native_dt_s": native_dt,
                },
                "trace_schema_validation": {
                    "trajectory_array_count": len(trajectory_shapes),
                    "trajectory_per_sample_array_count": len(trajectory_shapes) - 1,
                    "trajectory_expected_native_samples_on_axis_0": expected_n,
                    "static_body_mass_native_shape": list(trajectory_shapes["body_mass_native"]),
                    "dimensions_from_array_order_and_manifest": {
                        "body_count": body_count,
                        "leg_count": leg_count,
                        "joint_count": len(order["joint_order"]),
                        "position_actuator_count": len(order["position_actuator_joint_order"]),
                        "thorax_clearance_geom_count": clearance_geom_count,
                        "compiled_geom_count": geom_count,
                        "qpos_count": int(manifest["effective_model_options"]["nq"]),
                        "qvel_count": int(manifest["effective_model_options"]["nv"]),
                    },
                    "contacts_event_table_rows": contact_row_count,
                    "contacts_event_table_variable_length": True,
                    "all_trajectory_keys_and_shapes_checked": True,
                    "all_trajectory_float_arrays_finite": True,
                },
            }
        return {
            "case_id": case["case_id"],
            "cohort_id": case["cohort_id"],
            "direction": case["direction"],
            "body_seed": case["body_seed"],
            "neural_seed": case["neural_seed"],
            "speed_magnitude_deg_s": case["speed_magnitude_deg_s"],
            "head_direction_sign": case["head_direction_sign"],
            "head_angular_speed_deg_s": case["head_angular_speed_deg_s"],
            "arena_angular_speed_deg_s": case["arena_angular_speed_deg_s"],
            "run_path": str(run_dir),
            "execution_status": execution_status,
            "status": "complete",
            "pending_reason": None,
            "metrics": metrics,
        }
    except Exception as exc:
        # A complete run-record without complete, verifiable raw inputs is
        # still pending for analysis. No missing or invalid metric becomes 0.
        return _empty_pending(
            case, run_dir, "complete_run_not_yet_analysis_ready",
            execution_status=execution_status,
            diagnostic_error=f"{type(exc).__name__}: {exc}",
        )


def _numeric_metric(members: dict[str, dict[str, Any]], window: str, metric: str, direction: str) -> float:
    return float(members[direction]["metrics"]["windows"][window][metric])


def build_cohort_contrasts(
    cohorts: list[str],
    cases_by_id: dict[str, dict[str, Any]],
    expected_cases: list[dict[str, Any]],
    protocol_valid: bool,
) -> list[dict[str, Any]]:
    spec_by_cohort: dict[str, dict[str, dict[str, Any]]] = {cohort: {} for cohort in cohorts}
    for spec in expected_cases:
        spec_by_cohort.setdefault(spec["cohort_id"], {})[spec["direction"]] = spec
    output = []
    for cohort in cohorts:
        specs = spec_by_cohort.get(cohort, {})
        member_cases = {
            direction: cases_by_id.get(spec.get("case_id"))
            for direction, spec in specs.items()
        }
        missing = [
            direction
            for direction in ("static", "positive", "negative")
            if not member_cases.get(direction) or member_cases[direction].get("status") != "complete"
        ]
        initial_match: bool | None = None
        initial_hashes: dict[str, Any] = {}
        if not missing:
            field_sets = {
                field: {
                    member_cases[direction]["metrics"]["initial_state_hashes"].get(field)
                    for direction in ("static", "positive", "negative")
                }
                for field in INITIAL_STATE_HASH_FIELDS
            }
            initial_hashes = {field: sorted(str(value) for value in values) for field, values in field_sets.items()}
            initial_match = all(len(values) == 1 and None not in values for values in field_sets.values())
        if missing or not protocol_valid or initial_match is not True:
            reason = (
                "missing_or_incomplete_members" if missing
                else ("prepared_protocol_hash_mismatch" if not protocol_valid else "within_cohort_initial_state_mismatch")
            )
            output.append({
                "cohort_id": cohort,
                "status": "pending",
                "pending_reason": reason,
                "missing_directions": missing,
                "within_cohort_initial_state_match": initial_match,
                "initial_state_hashes_by_field": initial_hashes,
                "contrasts": None,
            })
            continue

        members = {direction: member_cases[direction] for direction in ("static", "positive", "negative")}
        contrasts_by_window = {}
        metrics_to_compare = (
            "yaw_net_native_deg",
            "thorax_forward_delta_start_heading_mm",
            "thorax_lateral_left_delta_start_heading_mm",
            "fraction_samples_with_at_least_two_loaded_legs",
        )
        for window in WINDOWS:
            static_values = {metric: _numeric_metric(members, window, metric, "static") for metric in metrics_to_compare}
            positive_values = {metric: _numeric_metric(members, window, metric, "positive") for metric in metrics_to_compare}
            negative_values = {metric: _numeric_metric(members, window, metric, "negative") for metric in metrics_to_compare}
            plus_spec = specs["positive"]
            minus_spec = specs["negative"]
            contrasts_by_window[window] = {
                "positive_minus_static": {metric: positive_values[metric] - static_values[metric] for metric in metrics_to_compare},
                "negative_minus_static": {metric: negative_values[metric] - static_values[metric] for metric in metrics_to_compare},
                "positive_minus_negative": {metric: positive_values[metric] - negative_values[metric] for metric in metrics_to_compare},
                "direction_normalized_by_registered_head_direction_sign": {
                    "definition": "head_direction_sign * (moving_condition_metric - matched_static_metric); descriptive engineering coordinate normalization only",
                    "positive_yaw": int(plus_spec["head_direction_sign"]) * (positive_values["yaw_net_native_deg"] - static_values["yaw_net_native_deg"]),
                    "negative_yaw": int(minus_spec["head_direction_sign"]) * (negative_values["yaw_net_native_deg"] - static_values["yaw_net_native_deg"]),
                    "positive_lateral_left": int(plus_spec["head_direction_sign"]) * (positive_values["thorax_lateral_left_delta_start_heading_mm"] - static_values["thorax_lateral_left_delta_start_heading_mm"]),
                    "negative_lateral_left": int(minus_spec["head_direction_sign"]) * (negative_values["thorax_lateral_left_delta_start_heading_mm"] - static_values["thorax_lateral_left_delta_start_heading_mm"]),
                },
                "raw_member_metrics": {
                    "static": static_values,
                    "positive": positive_values,
                    "negative": negative_values,
                },
            }
        output.append({
            "cohort_id": cohort,
            "body_seed": members["static"]["body_seed"],
            "neural_seed": members["static"]["neural_seed"],
            "speed_magnitude_deg_s": members["static"]["speed_magnitude_deg_s"],
            "status": "complete_descriptive_contrasts",
            "pending_reason": None,
            "missing_directions": [],
            "within_cohort_initial_state_match": True,
            "initial_state_hashes_by_field": initial_hashes,
            "contrasts": contrasts_by_window,
        })
    return output


def _sign(value: float) -> int:
    if abs(float(value)) <= 1e-12:
        return 0
    return 1 if value > 0.0 else -1


def build_cross_seed_signs(cohort_contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["cohort_id"]: row for row in cohort_contrasts}
    cohorts = ("seed1731_speed60", "seed1747_speed60")
    if any(cohort not in lookup or lookup[cohort].get("status") != "complete_descriptive_contrasts" for cohort in cohorts):
        return {
            "status": "pending",
            "cohorts_required": list(cohorts),
            "comparisons": None,
            "interpretation": "Wait for both matched 60 deg/s seed cohorts to have complete verified static/positive/negative triplets.",
        }
    comparisons = []
    for window in WINDOWS:
        for direction in ("positive", "negative"):
            field = f"{direction}_yaw"
            values = [
                float(lookup[cohort]["contrasts"][window]["direction_normalized_by_registered_head_direction_sign"][field])
                for cohort in cohorts
            ]
            signs = [_sign(value) for value in values]
            comparisons.append({
                "window": window,
                "direction": direction,
                "cohort_ids": list(cohorts),
                "direction_normalized_yaw_contrasts_deg": values,
                "signs": signs,
                "descriptive_sign_agreement": signs[0] == signs[1],
                "both_zero": signs == [0, 0],
                "not_a_generalization_gate": True,
            })
    return {
        "status": "complete_descriptive_sign_comparison",
        "cohorts_required": list(cohorts),
        "comparisons": comparisons,
        "interpretation": "Only the two matched seed cohorts at 60 deg/s can be compared as cross-seed signs here. The seed-1729 speed-30 and speed-120 cohorts are single-seed speed conditions and do not add independent-seed replication. Agreement is descriptive and does not establish generalization or biological following.",
    }


def _positive_control_context() -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable",
        "analysis_path": str(POSITIVE_CONTROL_ANALYSIS),
        "review_path": str(POSITIVE_CONTROL_REVIEW),
    }
    if not POSITIVE_CONTROL_ANALYSIS.is_file() or not POSITIVE_CONTROL_REVIEW.is_file():
        result["interpretation"] = "Accepted positive-control rule result could not be verified from the archived audit files."
        return result
    try:
        source = read_json(POSITIVE_CONTROL_ANALYSIS)
        accepted = next(row for row in source["cases"] if row.get("case_id") == "accepted_reference_drive_1_1")
        events = accepted["strict_root_analyzer_output"]["per_leg_events"]
        counts = {row["leg"]: int(row["complete_events"]) for row in events}
        all_zero = set(counts) == set(LEGS) and all(value == 0 for value in counts.values())
        result.update({
            "status": "verified_zero_strict_cycles_on_accepted_upstream_simulation_positive_control" if all_zero else "reference_result_differs",
            "analysis_sha256": sha256(POSITIVE_CONTROL_ANALYSIS),
            "review_sha256": sha256(POSITIVE_CONTROL_REVIEW),
            "strict_analyzer_sha256": source.get("analyzer", {}).get("sha256"),
            "current_strict_analyzer_path": str(STRICT_EVENT_ANALYZER),
            "current_strict_analyzer_sha256": sha256(STRICT_EVENT_ANALYZER) if STRICT_EVENT_ANALYZER.is_file() else None,
            "reference_rule_source_matches_current": (
                STRICT_EVENT_ANALYZER.is_file()
                and source.get("analyzer", {}).get("sha256") == sha256(STRICT_EVENT_ANALYZER)
            ),
            "positive_control_case_id": accepted["case_id"],
            "window_s": accepted.get("window_s"),
            "strict_complete_events_by_leg": counts,
            "prior_acceptance_decision": source.get("archive_context", {}).get("reference_prior_acceptance_report", {}).get("decision"),
            "prior_positive_control_cycle_summary": source.get("archive_context", {}).get("reference_prior_acceptance_report", {}).get("cycle_result_reported_by_that_review"),
            "interpretation": "The unfiltered immediate 2 ms loaded-boundary rule reports zero on this accepted upstream FlyGym simulation positive control. A zero is the outcome of that narrow detector and must not be used as a no-gait verdict. This is a digital simulation control, not a biological fly measurement.",
        })
    except Exception as exc:
        result.update({"status": "reference_result_unverified", "diagnostic_error": f"{type(exc).__name__}: {exc}"})
    return result


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Online validation v1: read-only body mechanics review",
        "",
        "This report reads saved 4 s CPU free-body traces. It does not launch FlyGym, advance physics, start a neural worker, or use a GPU. Metrics remain pending when a case is not complete and integrity-verified; pending values are never reported as zero.",
        "",
        f"Analysis status: **{result['analysis_status']}** ({result['completion']['complete_verified_case_count']}/12 cases complete and verified; {result['completion']['pending_case_count']} pending). Queue status at read: {result['queue_status'].get('status', 'missing')}.",
        "",
        "## Case status",
        "",
        "| Case | Direction | Seed | Speed (deg/s) | Status | Progress | Reason |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for case in result["cases"]:
        progress = case.get("progress", {})
        if progress:
            progress_text = f"frames {progress.get('completed_neural_frames', '?')}/800; ticks {progress.get('completed_body_ticks', '?')}/4000"
        elif case.get("status") == "complete":
            progress_text = "800 frames; 4000 ticks; 40001 samples"
        else:
            progress_text = ""
        reason = case.get("pending_reason") or "—"
        lines.append(
            f"| {case['case_id']} | {case['direction']} | {case['body_seed']} | "
            f"{case['speed_magnitude_deg_s']:g} | {case['status']} | {progress_text} | {reason} |"
        )

    complete_cases = [case for case in result["cases"] if case.get("status") == "complete"]
    if complete_cases:
        lines.extend([
            "",
            "## Native heading, body-frame motion, support, and posture",
            "",
            "Yaw is measured from thorax_heading_yaw_rad in the saved native FlyGym sign, unwrapped and interpolated at exact window endpoints. Thorax displacement is projected into the starting thorax heading frame: forward and lateral-left. Support uses raw per-leg upward Fz at 1% BW; the fraction is the share of native samples with at least two loaded legs.",
            "",
            "| Case | Window | Native yaw Δ (deg) | Thorax forward Δ (mm) | Lateral-left Δ (mm) | Samples ≥2 loaded legs | Min loaded legs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for case in complete_cases:
            for window, metrics in case["metrics"]["windows"].items():
                lines.append(
                    f"| {case['case_id']} | {window} | {metrics['yaw_net_native_deg']:+.4f} | "
                    f"{metrics['thorax_forward_delta_start_heading_mm']:+.4f} | "
                    f"{metrics['thorax_lateral_left_delta_start_heading_mm']:+.4f} | "
                    f"{metrics['fraction_samples_with_at_least_two_loaded_legs']:.4%} | {metrics['minimum_loaded_legs']} |"
                )
        lines.extend([
            "",
            "| Case | Max |roll| / |pitch| | Min thorax clearance (mm) | Thorax / nonleg contact samples | External and generalized applied forces | Root state writes | Fall screen |",
            "|---|---:|---:|---:|---|---|---|",
        ])
        for case in complete_cases:
            posture = case["metrics"]["whole_trace_posture_and_authority"]
            lines.append(
                f"| {case['case_id']} | {posture['max_abs_roll_deg']:.2f}° / {posture['max_abs_pitch_deg']:.2f}° | "
                f"{posture['thorax_clearance_min_mm']:+.4f} | {posture['thorax_ground_contact_samples']} / "
                f"{posture['nonleg_ground_contact_samples']} | "
                f"{'zero' if posture['external_and_generalized_applied_forces_zero'] else 'nonzero observed'} | "
                f"{'zero' if posture['root_state_write_counts_zero'] else 'nonzero observed'} | "
                f"{posture['fall_screen']['status']} |"
            )

    lines.extend([
        "",
        "## Matched cohort contrasts against static",
        "",
        "Forward/lateral motion is body-heading normalized. For direction normalization, the moving-minus-static yaw and lateral-left contrasts are multiplied by the registered head_direction_sign (+1 or −1). This is a descriptive engineering coordinate transform, not an expected-sign assertion or biological success test.",
        "",
        "| Cohort | Window | Positive yaw Δ vs static (deg) | Negative yaw Δ vs static (deg) | Direction-normalized yaw (+ / −) | Forward Δ (+ / −, mm) | Lateral Δ (+ / −, mm) | Support Δ (+ / −) | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for cohort in result["cohort_contrasts"]:
        if cohort.get("status") != "complete_descriptive_contrasts":
            lines.append(
                f"| {cohort['cohort_id']} | all | pending | pending | pending | pending | pending | pending | {cohort['pending_reason']} |"
            )
            continue
        for window, contrast in cohort["contrasts"].items():
            pos = contrast["positive_minus_static"]
            neg = contrast["negative_minus_static"]
            normalized = contrast["direction_normalized_by_registered_head_direction_sign"]
            lines.append(
                f"| {cohort['cohort_id']} | {window} | {pos['yaw_net_native_deg']:+.4f} | "
                f"{neg['yaw_net_native_deg']:+.4f} | {normalized['positive_yaw']:+.4f} / "
                f"{normalized['negative_yaw']:+.4f} | "
                f"{pos['thorax_forward_delta_start_heading_mm']:+.4f} / {neg['thorax_forward_delta_start_heading_mm']:+.4f} | "
                f"{pos['thorax_lateral_left_delta_start_heading_mm']:+.4f} / {neg['thorax_lateral_left_delta_start_heading_mm']:+.4f} | "
                f"{pos['fraction_samples_with_at_least_two_loaded_legs']:+.4f} / "
                f"{neg['fraction_samples_with_at_least_two_loaded_legs']:+.4f} | {cohort['status']} |"
            )

    signs = result["cross_seed_direction_signs"]
    lines.extend(["", "## Cross-seed direction signs", ""])
    if signs.get("status") == "pending":
        lines.append(
            "Pending: both matched 60 deg/s seed triplets must be complete and verified. "
            "The speed-30 and speed-120 cohorts each use seed 1729 and do not provide additional independent-seed replication."
        )
    else:
        lines.extend([
            "The sign values compare direction-normalized yaw contrasts for the two matched 60 deg/s cohorts, seeds 1731 and 1747. Agreement is descriptive only.",
            "",
            "| Window | Direction | Seed 1731 normalized yaw Δ (deg) | Seed 1747 normalized yaw Δ (deg) | Signs | Descriptive agreement |",
            "|---|---|---:|---:|---|---|",
        ])
        for row in signs["comparisons"]:
            lines.append(
                f"| {row['window']} | {row['direction']} | "
                f"{row['direction_normalized_yaw_contrasts_deg'][0]:+.4f} | "
                f"{row['direction_normalized_yaw_contrasts_deg'][1]:+.4f} | "
                f"{row['signs'][0]:+d} / {row['signs'][1]:+d} | {row['descriptive_sign_agreement']} |"
            )

    control = result["strict_contact_cycle_positive_control_context"]
    lines.extend([
        "",
        "## Strict contact-cycle detector boundary",
        "",
        f"The prior accepted upstream FlyGym simulation positive-control trace has strict complete-event counts {control.get('strict_complete_events_by_leg')} under the raw 1% BW, 0.05 mm, immediate 2 ms loaded-before/after rule (audit: {control.get('analysis_path')}). Its earlier accepted cycle assessment used a different touchdown rule. This detector returns zero on an accepted positive control, so its zero count is not used here as evidence of no gait. The positive control is a digital simulation, not a biological fly measurement.",
        "",
        "## Interpretation boundary",
        "",
        "The registered decoder baseline is marked outside its calibrated fixed-head scope for free-body online-pose runs. These summaries remain single-run digital mechanics measurements. The 12 cases form four matched static/positive/negative cohorts; only two independent seed cohorts exist at 60 deg/s, while seed 1729 is reused for speed 30 and 120. No repeatability or generalization claim is issued while any case is pending; a fully complete table still supports descriptive comparisons only.",
        "",
        f"Protocol SHA-256: {result['provenance']['protocol_sha256']}. Analyzer SHA-256: {result['provenance']['analyzer_sha256']}.",
        "",
    ])
    return "\n".join(lines)


def build_result(bundle: Path) -> dict[str, Any]:
    protocol_path = bundle / "protocol.json"
    prepare_manifest_path = bundle / "prepare-manifest.json"
    protocol = read_json(protocol_path)
    prepare_manifest = read_json(prepare_manifest_path)
    cases_spec, cohorts = _expected_case_map(protocol)
    expected_protocol_record = prepare_manifest.get("protocol", {})
    protocol_matches_prepare = (
        str(Path(expected_protocol_record.get("path", "")).resolve()) == str(protocol_path.resolve())
        and expected_protocol_record.get("sha256") == sha256(protocol_path)
        and int(expected_protocol_record.get("size_bytes", -1)) == protocol_path.stat().st_size
    )
    case_results = [
        analyze_case(spec, bundle / "runs" / spec["case_id"], protocol)
        for spec in cases_spec
    ]
    cases_by_id = {case["case_id"]: case for case in case_results}
    cohort_contrasts = build_cohort_contrasts(cohorts, cases_by_id, cases_spec, protocol_matches_prepare)
    cross_seed = build_cross_seed_signs(cohort_contrasts)

    queue_path = bundle / "queue-status.json"
    try:
        queue_status = read_json(queue_path) if queue_path.is_file() else {"status": "missing"}
    except Exception as exc:
        queue_status = {"status": "unreadable", "diagnostic_error": f"{type(exc).__name__}: {exc}"}
    complete_count = sum(case["status"] == "complete" for case in case_results)
    pending_count = len(case_results) - complete_count
    all_cases_complete = complete_count == len(case_results)
    all_cohorts_complete = all(row.get("status") == "complete_descriptive_contrasts" for row in cohort_contrasts)
    return {
        "schema": "flygym-online-validation-mechanics-review-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(bundle),
        "analysis_status": "complete_descriptive_only" if all_cases_complete and all_cohorts_complete else "pending",
        "completion": {
            "expected_case_count": len(cases_spec),
            "complete_verified_case_count": complete_count,
            "pending_case_count": pending_count,
            "all_cases_complete_and_verified": all_cases_complete,
            "all_matched_cohorts_complete": all_cohorts_complete,
            "repeatability_or_generalization_claim": "withheld_until_all_cases_complete; descriptive_only_even_after_completion",
        },
        "queue_status": queue_status,
        "provenance": {
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "prepare_manifest_path": str(prepare_manifest_path),
            "prepare_manifest_sha256": sha256(prepare_manifest_path),
            "protocol_matches_prepare_manifest": protocol_matches_prepare,
            "analyzer_path": str(Path(__file__).resolve()),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "reference_rule_audit_sha256": sha256(POSITIVE_CONTROL_ANALYSIS) if POSITIVE_CONTROL_ANALYSIS.is_file() else None,
            "reference_rule_review_sha256": sha256(POSITIVE_CONTROL_REVIEW) if POSITIVE_CONTROL_REVIEW.is_file() else None,
        },
        "windows_s": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "methods": {
            "yaw": "unwrapped thorax_heading_yaw_rad; exact interpolation at window endpoints; native FlyGym sign preserved",
            "body_translation": "c_thorax body_xpos_mm endpoint displacement projected into forward and lateral-left axes defined by the starting thorax yaw for each window",
            "support": "per-leg foot_contact_fz_up_native >= 1% of total body weight; report fraction of half-open window samples with at least two loaded legs",
            "fall_screen": "whole-trace thorax/nonleg ground contacts, thorax clearance, body quaternion roll/pitch; descriptive screen only",
            "force_and_authority": "whole-trace max absolute root external wrench, body xfrc, qfrc, and root qpos/qvel write counters",
            "cohort_contrasts": "positive/negative moving condition minus same-cohort static; normalized yaw/lateral contrast multiplies by registered head_direction_sign, as a descriptive coordinate normalization only",
            "pending_policy": "incomplete run, missing files, incomplete integrity, hash mismatch, or unexpected trace schema yields status=pending and metrics=null",
        },
        "strict_contact_cycle_positive_control_context": _positive_control_context(),
        "cases": case_results,
        "cohort_contrasts": cohort_contrasts,
        "cross_seed_direction_signs": cross_seed,
        "interpretation": "This is a saved-trace mechanics readout. It is not a biological success criterion. Strict contact-cycle zero is not used as a no-gait judgment because the same rule returned zero on the accepted upstream simulation positive control.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace only this analyzer's JSON/Markdown outputs")
    args = parser.parse_args(argv)
    bundle = args.bundle.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else bundle / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "online-validation-mechanics-review-v1.json"
    markdown_path = output_dir / "online-validation-mechanics-review-v1.md"
    if not args.overwrite and (json_path.exists() or markdown_path.exists()):
        raise FileExistsError(f"Refusing to overwrite prior analysis output in {output_dir}; pass --overwrite after confirming it is this analyzer's output")

    result = build_result(bundle)
    markdown = _markdown_report(result)
    json_text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    markdown_tmp = markdown_path.with_name(markdown_path.name + ".tmp")
    json_tmp.write_text(json_text, encoding="utf-8")
    markdown_tmp.write_text(markdown, encoding="utf-8")
    json_tmp.replace(json_path)
    markdown_tmp.replace(markdown_path)
    print(json.dumps({
        "analysis_status": result["analysis_status"],
        "complete_verified_cases": result["completion"]["complete_verified_case_count"],
        "pending_cases": result["completion"]["pending_case_count"],
        "json": str(json_path),
        "markdown": str(markdown_path),
        "analyzer_sha256": result["provenance"]["analyzer_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
