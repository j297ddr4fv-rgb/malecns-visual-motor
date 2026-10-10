#!/usr/bin/env python3
"""Run a three-condition, CPU-only free-ground AP/body-yaw screen.

The registered cohort holds symmetric P=0.30 constant and changes only the
current phase-locked AP endpoint command (d=0, +0.20, -0.20). It uses a shared
seed and the reviewed V4.1/AP position-servo controller. This is one-seed
mechanics screening, not accepted walking or biological turning evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.update({
    "MUJOCO_GL": "osmesa",
    "PYOPENGL_PLATFORM": "osmesa",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
})
os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", "/data/models/flygym/assets")
os.environ.setdefault("MPLCONFIGDIR", "/data/cache/flygym/matplotlib")

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
ARTIFACT_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "standardized-repair-20261007-v1/mechanics"
)
CASE_PARENT = ARTIFACT_ROOT / "mechanics_cases"
ANALYSIS_WINDOW_S = (0.5, 3.0)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
P_DEADBAND = 1e-6
CONDITIONS = (
    ("static_d0", 0.0),
    ("ap_dplus_0p20", 0.20),
    ("ap_dminus_0p20", -0.20),
)
V41_BASELINE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-phase-speed-repair-20261006-v1/development/v4-1/weak_0p30"
)
WALK_EFFECTIVE_CONFIG = V41_BASELINE / "effective_config.json"
CONTROLLER_CONFIG = V41_BASELINE / "controller_config.json"

sys.path.insert(0, str(SCRIPTS))


class CalibrationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes(order="C")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationError(f"expected JSON object at {path}")
    return value


def _source_records() -> dict[str, dict[str, Any]]:
    paths = (
        __file__,
        SCRIPTS / "flygym_phase_speed_gait_repaired_v1.py",
        SCRIPTS / "flygym_rest_contract.py",
        SCRIPTS / "flygym_ap_endpoint_gesture_prototype.py",
        SCRIPTS / "flygym_phase_speed_gait_v4_1.py",
        SCRIPTS / "flygym_phase_speed_gait_v3.py",
        SCRIPTS / "flygym_phase_speed_gait_v2.py",
        SCRIPTS / "flygym_phase_speed_gait.py",
        SCRIPTS / "flygym_pose_priority_ik_v2.py",
        SCRIPTS / "run_flygym_reference_walk.py",
        CONTROLLER_CONFIG,
        WALK_EFFECTIVE_CONFIG,
        PROJECT / "config/flygym_cartesian_repair_protocol.json",
    )
    records: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=True)
        records[str(path)] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return records


def _write_protocol_and_freeze(
    *, out_root: Path, duration_s: float, seed: int, power: float,
) -> dict[str, Any]:
    protocol = {
        "schema_version": "flygym-ap-body-yaw-mechanics-protocol-v1",
        "classification": "CPU free-ground engineering mechanics calibration; one seed; no neural forward or visual input",
        "condition_ids": [item[0] for item in CONDITIONS],
        "common": {
            "duration_s": duration_s,
            "seed": seed,
            "symmetric_side_power_P": [power, power],
            "analysis_window_s": list(ANALYSIS_WINDOW_S),
            "world": "flat ground; free-body six-leg FlyGym mechanics",
            "controller": "PhaseSpeedGaitControllerRepairedV1, wrapper over hash-pinned current V4.1/AP controller",
            "mechanics_change_from_current_parent": "none; same V4.1 gait, same hind clearance, same AP command rule",
            "shared_rest_contract": {
                "p_deadband": P_DEADBAND,
                "active_rule": "P > p_deadband",
                "input_P": power,
                "input_is_active": bool(power > P_DEADBAND),
            },
            "neural_or_visual_input": False,
            "stimulus_label_passed_to_controller": False,
            "external_body_wrench": False,
            "root_state_write_after_initialization": False,
            "gpu": False,
        },
        "conditions": [
            {
                "case_id": case_id,
                "differential_ap_d": d,
                "dna02_command": abs(d),
                "dng13_command": abs(d),
                "soma_side": "left" if d >= 0.0 else "right",
                "target_side_mapping": {
                    "dna02_like_aep_posterior": "same as soma side",
                    "dng13_like_aep_anterior_pep_posterior": "opposite soma side",
                },
                "endpoint_shift_mm": 0.05,
                "extra_gain": 1.0,
            }
            for case_id, d in CONDITIONS
        ],
        "mechanical_acceptance_reference": {
            "minimum_complete_cycles_per_leg": 2,
            "minimum_tarsus5_clearance_mm": 0.05,
            "minimum_unloaded_s": 0.002,
            "minimum_loaded_touchdown_s": 0.002,
            "loaded_force_threshold_bodyweight_fraction": 0.01,
            "symmetric_static_yaw_limit_deg": 10.0,
            "straightness_seed_count": 3,
            "scope": "thresholds are engineering criteria, not biological standards; this cohort alone cannot pass three-seed acceptance",
        },
        "interpretation": {
            "body_yaw_source": "thorax_heading_yaw_rad from free-ground body, +X heading projected into world XY",
            "ball_omega_z_used": False,
            "turn_sign": "reported as measured; never flipped to fit an expectation",
            "biological_claim": False,
        },
        "output_root": str(out_root.resolve()),
    }
    source_freeze = {
        "schema_version": "flygym-ap-body-yaw-mechanics-source-freeze-v1",
        "status": "frozen_before_capture",
        "files": _source_records(),
        "upstream_controller_sha256": {
            str((SCRIPTS / "flygym_phase_speed_gait_v4_1.py").resolve()): "126b641e2998992e4cad351f4a48292882063851350e4006ba40552a38d7e2eb",
            str((SCRIPTS / "flygym_ap_endpoint_gesture_prototype.py").resolve()): "689364bc661c4fc37a15ddc617a398450a3f52d2fe4da7d202b74687cda91b8f",
        },
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "input_config_sha256": {
            str(WALK_EFFECTIVE_CONFIG.resolve()): sha256(WALK_EFFECTIVE_CONFIG),
            str(CONTROLLER_CONFIG.resolve()): sha256(CONTROLLER_CONFIG),
        },
        "output_root": str(out_root.resolve()),
        "raw_data_copied": False,
    }
    write_json(out_root / "protocol.json", protocol)
    write_json(out_root / "source_freeze.json", source_freeze)
    return source_freeze


def _extract_control_telemetry(controller, tick: int, time_s: float, power: float, d: float) -> dict[str, Any]:
    snapshot = controller.telemetry_snapshot()
    phases = snapshot.get("cpg_phase_rad_by_leg")
    clearance = snapshot.get("hind_clearance", {})
    per_leg = clearance.get("per_leg", {}) if isinstance(clearance, dict) else {}
    reduced: dict[str, Any] = {}
    for leg in LEGS:
        item = per_leg.get(leg)
        if not isinstance(item, dict):
            continue
        selected = (
            "phase_rad", "native_endpoint_ap_delta_mm", "prescribed_ap_task_delta_mm",
            "aep_pep_command_terms_mm", "nominal_foot_body_ap_mm", "target_foot_body_ap_mm",
            "solved_foot_body_ap_mm", "position_residual_after_pose_priority_mm",
            "hind_clearance_envelope", "hind_clearance_goal_world_z_mm",
        )
        reduced[leg] = {key: item[key] for key in selected if key in item}
    return {
        "tick": int(tick),
        "control_time_s": float(time_s),
        "P_left_right": [float(power), float(power)],
        "differential_ap_d": float(d),
        "cpg_phase_rad_by_leg": phases,
        "ap_or_hind_target_by_leg": reduced,
        "controller_metadata_mode": clearance.get("mode") if isinstance(clearance, dict) else None,
    }


def _window_summary(session, power: float, d: float, initial_qpos_sha: str, initial_qvel_sha: str) -> dict[str, Any]:
    a = session.all_arrays
    count = session._native_samples
    t = np.asarray(a["time_s"][:count], dtype=np.float64)
    mask = (t >= ANALYSIS_WINDOW_S[0]) & (t <= ANALYSIS_WINDOW_S[1])
    if mask.sum() < 2:
        raise CalibrationError("saved trace does not cover the registered analysis window")
    yaw = np.unwrap(np.asarray(a["thorax_heading_yaw_rad"][:count], dtype=np.float64))
    yaw_start = float(np.interp(ANALYSIS_WINDOW_S[0], t, yaw))
    yaw_end = float(np.interp(ANALYSIS_WINDOW_S[1], t, yaw))
    body_weight = float(np.sum(session.body_mass) * 9810.0)
    loaded_threshold = body_weight * 0.01
    fz = np.asarray(a["foot_contact_fz_up_native"][:count], dtype=np.float64)
    loaded = fz >= loaded_threshold
    support_two = np.sum(loaded[mask], axis=1) >= 2
    qfrc = np.asarray(a["qfrc_applied_native"][:count], dtype=np.float64)
    xfrc = np.asarray(session.root_external_wrench_by_sample[:count], dtype=np.float64)
    body_xfrc = np.asarray(a["body_xfrc_applied_native"][:count], dtype=np.float64)
    return {
        "analysis_window_s": list(ANALYSIS_WINDOW_S),
        "P": power,
        "d": d,
        "thorax_heading_yaw_start_rad": yaw_start,
        "thorax_heading_yaw_end_rad": yaw_end,
        "thorax_heading_yaw_change_deg_0p5_3s": math.degrees(yaw_end - yaw_start),
        "loaded_force_threshold_bodyweight_fraction": 0.01,
        "fraction_samples_with_at_least_two_loaded_legs": float(np.mean(support_two)),
        "mean_loaded_legs": float(np.mean(np.sum(loaded[mask], axis=1))),
        "loaded_fraction_by_leg": {
            leg: float(np.mean(loaded[mask, i])) for i, leg in enumerate(LEGS)
        },
        "thorax_ground_contact_samples": int(np.count_nonzero(session.thorax_ground_contact_by_sample[:count][mask])),
        "nonleg_ground_contact_samples": int(np.count_nonzero(session.nonleg_ground_contact_by_sample[:count][mask])),
        "root_qpos_write_count_max": int(np.max(session.root_qpos_write_count_by_sample[:count])),
        "root_qvel_write_count_max": int(np.max(session.root_qvel_write_count_by_sample[:count])),
        "root_external_wrench_peak_native": float(np.max(np.linalg.norm(xfrc[mask], axis=1))),
        "body_external_wrench_peak_native": float(np.max(np.linalg.norm(body_xfrc[mask], axis=2))),
        "generalized_applied_force_peak_native": float(np.max(np.abs(qfrc[mask]))),
        "initial_qpos_sha256": initial_qpos_sha,
        "initial_qvel_sha256": initial_qvel_sha,
        "cpg_phase_trace_sha256": array_sha256(session.cpg_phase_by_sample[:count]),
        "descending_drive_trace_sha256": array_sha256(session.descending_drive_by_sample[:count]),
    }


def _run_condition(
    *,
    case_id: str,
    differential_ap_d: float,
    duration_s: float,
    seed: int,
    power: float,
    out_root: Path,
    walk_values: dict[str, Any],
    controller_values: dict[str, Any],
) -> dict[str, Any]:
    from run_flygym_reference_walk import FlyGymReferenceSession, WalkConfig
    from flygym_phase_speed_gait_repaired_v1 import (
        PhaseSpeedGaitController,
        controller_config_for_differential_ap,
        verify_parent_sources,
    )
    from flygym_ap_endpoint_gesture_prototype import APEndpointGestureConfig

    case_dir = out_root / "mechanics_cases" / case_id
    if case_dir.exists():
        raise FileExistsError(f"refusing to overwrite mechanics case: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=False)
    run_id = f"standardized-mechanics-{case_id}-T{duration_s:g}s-seed{seed}"
    controller_base = APEndpointGestureConfig(**controller_values)
    selected_controller_config = controller_config_for_differential_ap(
        differential_ap_d, base_config=controller_base
    )
    config_values = dict(walk_values)
    config_values["duration_s"] = float(duration_s)
    config_values["seed"] = int(seed)
    for key in ("spawn_position_mm", "spawn_quaternion_wxyz", "position_actuator_forcerange_native", "camera_resolution_hw"):
        if key in config_values:
            config_values[key] = tuple(config_values[key])
    walk_config = WalkConfig(**config_values)
    if not math.isclose(float(walk_config.cpg_frequency_hz), 7.4074074074074066, rel_tol=0.0, abs_tol=1e-12):
        raise CalibrationError("walking config no longer matches the reviewed V4.1 7.4074 Hz template")
    if float(selected_controller_config.active_threshold) != P_DEADBAND:
        raise CalibrationError("selected V4.1 active threshold differs from shared rest contract")

    run_record = {
        "schema_version": "flygym-standardized-mechanics-run-v1",
        "status": "running",
        "case_id": case_id,
        "run_id": run_id,
        "condition": {
            "P_left_right": [float(power), float(power)],
            "differential_ap_d": float(differential_ap_d),
            "dna02_command": abs(float(differential_ap_d)),
            "dng13_command": abs(float(differential_ap_d)),
            "soma_side": "left" if differential_ap_d >= 0.0 else "right",
            "seed": int(seed),
            "duration_s": float(duration_s),
        },
        "analysis_window_s": list(ANALYSIS_WINDOW_S),
        "input_source": "constant externally specified engineering P/d values; no neural worker or stimulus label",
        "stimulus_label_sent_to_controller": False,
        "neural_forward_called": False,
        "gpu_used": False,
        "controller_source_hashes": verify_parent_sources(),
        "controller_config": asdict(selected_controller_config),
        "body_config": asdict(walk_config),
        "shared_rest_contract": {
            "p_deadband": P_DEADBAND,
            "active_rule": "P > p_deadband",
            "static_boundary_inclusive": True,
        },
        "root_state_writes_after_initialization": 0,
        "root_external_wrench_applied": False,
        "body_external_wrench_applied": False,
        "started_monotonic_s": time.monotonic(),
    }
    write_json(case_dir / "run-record.json", run_record)

    session = None
    try:
        session = FlyGymReferenceSession(
            duration_s=duration_s,
            seed=seed,
            render_video=False,
            config=walk_config,
            run_id=run_id,
        )
        initial_qpos = np.asarray(session.data.qpos, dtype=np.float64).copy()
        initial_qvel = np.asarray(session.data.qvel, dtype=np.float64).copy()
        initial_qpos_sha = array_sha256(initial_qpos)
        initial_qvel_sha = array_sha256(initial_qvel)
        if not math.isclose(session.time_s, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise CalibrationError("new free-ground session did not start at time zero")
        controller = PhaseSpeedGaitController(session, selected_controller_config)
        session.controller = controller
        metadata = controller.controller_metadata()
        metadata.update({
            "input_source": "constant P/d calibration protocol",
            "constant_side_drive_left_right": [power, power],
            "differential_ap_d": differential_ap_d,
            "stimulus_label_branching": False,
            "neural_worker": False,
            "external_body_wrench": False,
            "root_state_write_after_initialization": False,
        })
        run_record.update({
            "initial_state": {
                "qpos_sha256": initial_qpos_sha,
                "qvel_sha256": initial_qvel_sha,
                "time_s": float(session.time_s),
            },
            "controller_metadata": metadata,
            "body_session_started": True,
            "control_tick_count_expected": round(duration_s / walk_config.control_timestep_s),
            "native_sample_count_expected": round(duration_s / walk_config.native_timestep_s) + 1,
        })
        write_json(case_dir / "run-record.json", run_record)

        telemetry_path = case_dir / "controller_telemetry.jsonl"
        with telemetry_path.open("x", encoding="utf-8", buffering=1) as telemetry_stream:
            for tick in range(round(duration_s / walk_config.control_timestep_s)):
                control_time = tick * walk_config.control_timestep_s
                result = session.step_control_tick((power, power))
                record = _extract_control_telemetry(
                    controller, tick, control_time, power, differential_ap_d
                )
                record["body_time_after_tick_s"] = float(result["time_s"])
                telemetry_stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                if (tick + 1) % 500 == 0:
                    print(f"{case_id}: {tick + 1}/{round(duration_s / walk_config.control_timestep_s)} control ticks", flush=True)

        expected_ticks = round(duration_s / walk_config.control_timestep_s)
        expected_samples = round(duration_s / walk_config.native_timestep_s) + 1
        if session._control_samples != expected_ticks or session._native_samples != expected_samples:
            raise CalibrationError(
                f"capture ended with unexpected counts: ticks={session._control_samples}, samples={session._native_samples}"
            )
        if not np.array_equal(session.data.qpos, initial_qpos) and session.time_s == 0.0:
            raise CalibrationError("session state unexpectedly changed without a physics step")
        output_manifest = session.save(
            case_dir,
            constant_drive_values_left_right=(power, power),
            controller_metadata=metadata,
        )
        run_record.update({
            "raw_capture_status": "saved_before_window_summary",
            "trajectory_path": str(case_dir / "trajectory.npz"),
            "contacts_path": str(case_dir / "contacts.npz"),
            "session_manifest_path": str(case_dir / "manifest.json"),
        })
        write_json(case_dir / "run-record.json", run_record)
        summary = _window_summary(session, power, differential_ap_d, initial_qpos_sha, initial_qvel_sha)
        run_record.update({
            "status": "complete",
            "raw_capture_status": "saved",
            "finished_monotonic_s": time.monotonic(),
            "control_tick_count": int(session._control_samples),
            "native_sample_count": int(session._native_samples),
            "physics_step_count": int(expected_ticks * round(walk_config.control_timestep_s / walk_config.native_timestep_s)),
            "window_summary": summary,
            "authority": {
                "root_qpos_write_count_max": summary["root_qpos_write_count_max"],
                "root_qvel_write_count_max": summary["root_qvel_write_count_max"],
                "root_external_wrench_peak_native": summary["root_external_wrench_peak_native"],
                "body_external_wrench_peak_native": summary["body_external_wrench_peak_native"],
                "generalized_applied_force_peak_native": summary["generalized_applied_force_peak_native"],
            },
            "capture_files": {
                "trajectory.npz": {"path": str(case_dir / "trajectory.npz"), "sha256": sha256(case_dir / "trajectory.npz"), "size_bytes": (case_dir / "trajectory.npz").stat().st_size},
                "contacts.npz": {"path": str(case_dir / "contacts.npz"), "sha256": sha256(case_dir / "contacts.npz"), "size_bytes": (case_dir / "contacts.npz").stat().st_size},
                "manifest.json": {"path": str(case_dir / "manifest.json"), "sha256": sha256(case_dir / "manifest.json"), "size_bytes": (case_dir / "manifest.json").stat().st_size},
                "controller_telemetry.jsonl": {"path": str(telemetry_path), "sha256": sha256(telemetry_path), "size_bytes": telemetry_path.stat().st_size},
            },
            "session_manifest_status": output_manifest.get("status"),
        })
        write_json(case_dir / "run-record.json", run_record)
        return run_record
    except BaseException as exc:
        run_record.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_monotonic_s": time.monotonic(),
        })
        write_json(case_dir / "run-record.json", run_record)
        raise
    finally:
        if session is not None:
            session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True,
                        help="explicitly execute the registered CPU mechanics cohort")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--power", type=float, default=0.30)
    parser.add_argument("--out", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    if not math.isclose(args.duration, 3.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("this registered calibration pins --duration 3.0")
    if args.seed != 1729:
        raise SystemExit("this registered calibration pins --seed 1729")
    if not math.isclose(args.power, 0.30, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("this registered calibration pins symmetric --power 0.30")
    if args.power <= P_DEADBAND:
        raise SystemExit("registered P must be active under the shared FlyGym rest contract")
    out_root = args.out.expanduser().resolve()
    if out_root != ARTIFACT_ROOT.resolve():
        raise SystemExit(f"this registered cohort is pinned to {ARTIFACT_ROOT}")
    if any((out_root / "mechanics_cases" / case_id).exists() for case_id, _ in CONDITIONS):
        raise SystemExit("one or more registered mechanics case directories already exist; refusing overwrite")
    out_root.mkdir(parents=True, exist_ok=True)

    freeze = _write_protocol_and_freeze(
        out_root=out_root,
        duration_s=args.duration,
        seed=args.seed,
        power=args.power,
    )
    if Path(__file__).resolve().as_posix() not in freeze["files"]:
        raise CalibrationError("source freeze omitted this runner")
    effective_record = read_json(WALK_EFFECTIVE_CONFIG)
    body_values = dict(effective_record["walk_config"])
    controller_values = read_json(CONTROLLER_CONFIG)["config"]
    results = []
    for case_id, d in CONDITIONS:
        results.append(_run_condition(
            case_id=case_id,
            differential_ap_d=d,
            duration_s=args.duration,
            seed=args.seed,
            power=args.power,
            out_root=out_root,
            walk_values=body_values,
            controller_values=controller_values,
        ))
    cohort = {
        "schema_version": "flygym-standardized-mechanics-cohort-v1",
        "status": "complete",
        "protocol": {"path": str(out_root / "protocol.json"), "sha256": sha256(out_root / "protocol.json")},
        "source_freeze": {"path": str(out_root / "source_freeze.json"), "sha256": sha256(out_root / "source_freeze.json")},
        "condition_ids": [record["case_id"] for record in results],
        "cases": {
            record["case_id"]: {
                "run_record": str(out_root / "mechanics_cases" / record["case_id"] / "run-record.json"),
                "trajectory": record["capture_files"]["trajectory.npz"],
                "manifest": record["capture_files"]["manifest.json"],
                "body_yaw_change_deg_0p5_3s": record["window_summary"]["thorax_heading_yaw_change_deg_0p5_3s"],
            }
            for record in results
        },
        "all_conditions_share_seed": True,
        "all_conditions_share_body_config": True,
        "all_conditions_share_power": True,
        "cpu_only": True,
        "single_seed_screen_only": True,
    }
    write_json(out_root / "cohort-manifest.json", cohort)
    print(json.dumps({
        "status": "complete",
        "case_ids": cohort["condition_ids"],
        "output_root": str(out_root),
        "cohort_manifest_sha256": sha256(out_root / "cohort-manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
