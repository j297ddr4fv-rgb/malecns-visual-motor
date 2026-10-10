#!/usr/bin/env python3
"""Prepare/run a frozen five-case, 13 s C=1 free-body online dynamics screen.

Preparation is CPU-only. Execution is review-gated and delegates the existing
frozen lowcontrast-v2 online body/worker loop while replacing only its retinal
renderer with the separately pinned continuous-phase schedule adapter.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, TextIO


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
ENGINEERING = Path("/data/artifacts/by-project/malecns-gpu/engineering")
PARENT = ENGINEERING / "feedback-diagnosis-20261007-v1"
BASE = PARENT / "online-highcontrast-dynamics-v1-revision-4"
PREPARATION = BASE / "preparation"
SPECS = PREPARATION / "angular-stimulus-specs"
SCHEDULES = PREPARATION / "retina-schedules"
RUNS = BASE / "runs"
ANALYSIS = BASE / "analysis"

LOW_BASE = PARENT / "online-lowcontrast-v2"
LOW_PROTOCOL_PATH = LOW_BASE / "protocol.json"
LOW_FREEZE_PATH = LOW_BASE / "preparation/source-freeze.json"
LOW_PREPARE_MANIFEST_PATH = LOW_BASE / "prepare-manifest.json"
LOW_COHORT_SUMMARY_PATH = LOW_BASE / "cohort-summary.json"
LOW_RUNNER = SCRIPTS / "run_flygym_free_body_online_lowcontrast_v2.py"
WORKER = SCRIPTS / "flygym_online_neural_worker_eye_parity_state_v2.py"
BOOTSTRAP = SCRIPTS / "flygym_highcontrast_continuous_retina_bootstrap_v1.py"
RETINA_ADAPTER = SCRIPTS / "flygym_highcontrast_continuous_retina_v1.py"
STANDARD_RENDERER = SCRIPTS / "standard_optomotor_stimulus.py"
REFERENCE_V2 = PARENT / "online-free-body-stripes-v2"
REFERENCE_V2_PROTOCOL = REFERENCE_V2 / "protocol.json"
REFERENCE_V2_FREEZE = REFERENCE_V2 / "preparation/source-freeze.json"
BIOLOGICAL_PLAN = PARENT / "biological-comparison-plan.md"

FRAME_DT_S = 0.005
FRAME_RATE_HZ = 200
FRAME_COUNT = 2600
DURATION_S = 13.0
GRAY_END_S = 1.0
SWITCH_S = 7.0
SEED = 1729
CONTRAST = 1.0
WAVELENGTH_DEG = 60.0
P_DEADBAND = 1e-6
BODY_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
WORKER_PYTHON = Path("/usr/bin/python3.12")
GLOBAL_NN_LOCK = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-recorded-retina-neural-ball-harness-20261007-v2/locks/"
    "neural-worker-and-ball-case.lock"
)
CASE_RUNTIME_S = 1800
QUEUE_RUNTIME_S = 6 * 60 * 60
RESOURCE_LIMITS = {
    "host_memory_max": "5G",
    "cpu_quota": "1200%",
    "cpu_quota_per_sec_usec_max": 12_000_000,
    "worker_max_vram_gib": 8.0,
}

WINDOWS = {
    "gray_0_1": (0.0, 1.0),
    "stim_early_1_1p5": (1.0, 1.5),
    "pre_switch_2_7": (2.0, 7.0),
    "post_switch_7_7p5": (7.0, 7.5),
    "late_8_13": (8.0, 13.0),
    "paper_relative_0_0p5_absolute_1_1p5": (1.0, 1.5),
    "paper_relative_1_3_absolute_2_4": (2.0, 4.0),
    "paper_relative_1_5_absolute_2_6": (2.0, 6.0),
    "full_stimulus_1_13": (1.0, 13.0),
    "stimulus_segment_1_7": (1.0, 7.0),
    "stimulus_segment_7_13": (7.0, 13.0),
}
ANALYSIS_WINDOWS_REQUIRED = {
    "gray_0_1": [0.0, 1.0],
    "stim_early_1_1p5": [1.0, 1.5],
    "pre_switch_2_7": [2.0, 7.0],
    "post_switch_7_7p5": [7.0, 7.5],
    "late_8_13": [8.0, 13.0],
    "paper_relative_0_0p5_absolute_1_1p5": [1.0, 1.5],
    "paper_relative_1_3_absolute_2_4": [2.0, 4.0],
    "paper_relative_1_5_absolute_2_6": [2.0, 6.0],
}
PREFIX_AUDIT = SCRIPTS / "audit_flygym_highcontrast_prefix_v1.py"
PREFIX_AUDIT_PYTHON = Path("/usr/bin/python3.12")


class RunnerError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"expected a JSON object: {path}")
    return value


def build_cases() -> tuple[dict[str, Any], ...]:
    definitions = (
        ("c1_static_highcontrast", "static_c1", 1, 0.0, "stripes_static"),
        ("c1_positive60_highcontrast", "positive60", 1, 60.0, "stripes_positive60"),
        ("c1_negative60_highcontrast", "negative60", -1, -60.0, "stripes_negative60"),
        ("c1_reverse_positive_to_negative_highcontrast", "positive_to_negative", 1, 60.0, "stripes_positive60"),
        ("c1_reverse_negative_to_positive_highcontrast", "negative_to_positive", -1, -60.0, "stripes_negative60"),
    )
    return tuple({
        "case_id": case_id,
        "cohort_id": "highcontrast-dynamics-seed1729",
        "direction": direction,
        "body_seed": SEED,
        "neural_seed": SEED,
        "seed": SEED,
        "contrast": CONTRAST,
        "wavelength_deg": WAVELENGTH_DEG,
        "head_direction_sign": sign,
        "head_angular_speed_deg_s": speed,
        "arena_angular_speed_deg_s": -speed,
        "direction_deg_s": speed,
        "temporal_frequency_hz": abs(speed) / WAVELENGTH_DEG,
        "speed_magnitude_deg_s": abs(speed),
        "switch_time_s": SWITCH_S if "to_" in direction else None,
        "duration_s": DURATION_S,
        "frame_count": FRAME_COUNT,
        "prefix_reference_case_id": reference,
        "interpretation": "retina-only stimulus schedule; condition label is never supplied to decoder or controller",
    } for case_id, direction, sign, speed, reference in definitions)


CASES = build_cases()
CASE_IDS = tuple(row["case_id"] for row in CASES)


def _visual_segment(case: dict[str, Any], start: float, end: float, sign: int, frequency: float, phase_rad: float, label: str) -> dict[str, Any]:
    direction = sign * WAVELENGTH_DEG * frequency
    return {
        "segment_id": label,
        "start_s": start,
        "end_s": end,
        "phase_label": label,
        "visual": {
            "kind": "sinusoidal_grating",
            "contrast": CONTRAST,
            "mean_luminance": 0.5,
            "wavelength_deg": WAVELENGTH_DEG,
            "head_direction_sign": sign,
            "temporal_frequency_hz": frequency,
            "direction_deg_s": direction,
            "arena_angular_speed_deg_s": -direction,
            "phase_at_onset_rad": phase_rad,
        },
    }


def make_schedule(case: dict[str, Any]) -> dict[str, Any]:
    gray = {
        "segment_id": "gray_0_1", "phase_label": "common_gray_prelude",
        "start_s": 0.0, "end_s": GRAY_END_S,
        "visual": {
            "kind": "uniform_gray", "contrast": 0.0, "mean_luminance": 0.5,
            "wavelength_deg": WAVELENGTH_DEG, "head_direction_sign": 1,
            "temporal_frequency_hz": 0.0, "direction_deg_s": 0.0,
            "arena_angular_speed_deg_s": 0.0, "phase_at_onset_rad": 0.0,
        },
    }
    direction = case["direction"]
    if direction == "static_c1":
        segments = [gray, _visual_segment(case, 1.0, 13.0, 1, 0.0, 0.0, "static_c1_1_13")]
    elif direction in {"positive60", "negative60"}:
        sign = int(case["head_direction_sign"])
        segments = [gray, _visual_segment(case, 1.0, 13.0, sign, 1.0, 0.0, f"constant_{direction}_1_13")]
    else:
        first_sign, second_sign = ((1, -1) if direction == "positive_to_negative" else (-1, 1))
        first = _visual_segment(case, 1.0, 7.0, first_sign, 1.0, 0.0, f"phase1_{direction}_1_7")
        first_visual = first["visual"]
        phase_at_switch = float(first_visual["phase_at_onset_rad"]) - (
            2.0 * math.pi * float(first_visual["arena_angular_speed_deg_s"]) * 6.0 / WAVELENGTH_DEG
        )
        second = _visual_segment(case, 7.0, 13.0, second_sign, 1.0, phase_at_switch, f"phase2_{direction}_7_13")
        segments = [gray, first, second]
    return {
        "schema_version": "flygym-highcontrast-continuous-retina-schedule-v1",
        "case_id": case["case_id"],
        "cohort_id": case["cohort_id"],
        "condition": {
            "direction": direction,
            "contrast": CONTRAST,
            "wavelength_deg": WAVELENGTH_DEG,
            "direction_frame": "initial FlyGym head azimuth; +head-Y is fly-left",
            "direction_deg_s_positive": case["direction_deg_s"],
            "arena_speed_is_renderer_coordinate_only": True,
            "switch_time_s": SWITCH_S if "to_" in direction else None,
        },
        "clock": {
            "frame_dt_s": FRAME_DT_S, "frame_rate_hz": FRAME_RATE_HZ,
            "frame_count": FRAME_COUNT, "gray_end_s": 1.0,
            "switch_s": 7.0, "duration_s": 13.0,
            "interval_convention": "half-open [start_s,end_s)",
        },
        "segments": segments,
        "phase_continuity": {
            "continuous_integrated_arena_displacement": True,
            "reversal_boundary_s": SWITCH_S if "to_" in direction else None,
            "no_phase_reset": True,
            "no_luminance_jump_at_reversal": True,
            "phase_boundary_rule": "phi_next=phi_prev-2*pi*arena_speed_prev*segment_duration/lambda; compare modulo 2*pi",
            "retina_only": True,
        },
        "pose_sign_conventions": {
            "behavior_yaw": "native thorax +X heading atan2(world_y,world_x); positive is fly-left",
            "renderer_pose": "VirtualWorld pose applies F=diag(1,-1,1), so virtual display yaw is -native yaw; this applies only to ray rendering",
            "stimulus_direction": "direction_deg_s is initial-head/fly-left-positive; arena_angular_speed_deg_s is its negative for fixed-arena rendering",
        },
        "motor_authority": "none; schedule selects retinal luminance only",
        "condition_changes_actions": False,
        "neural_decoder_cpg_and_body_reset_at_t7": False,
        "retinal_luminance_formula": "L=0.5*(1+c*cos(2*pi*(arena_azimuth-arena_motion_integral)/60+phase0)); gray L=0.5",
    }


def make_spec(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "angular-optomotor-stimulus-spec-1",
        "display_geometry": "fixed_arena_infinite_distance_angular_field",
        "stimulus": {
            "name": f"{case['case_id']}-continuous-phase-envelope",
            "contrast": CONTRAST,
            "head_direction_sign": int(case["head_direction_sign"]),
            "wavelength_deg": WAVELENGTH_DEG,
            "temporal_frequency_hz": float(case["temporal_frequency_hz"]),
            "mean_luminance": 0.5,
            "azimuth_extent_deg": [-135.0, 135.0],
            "elevation_extent_deg": [-45.0, 45.0],
            "onset_s": 1.0,
            "duration_s": 12.0,
            "phase_at_onset_rad": 0.0,
        },
    }


def _prefix_references() -> dict[str, Any]:
    names = (
        "online-frames.jsonl", "worker-requests.jsonl", "control-ticks.jsonl",
        "worker/frames.jsonl", "trajectory.npz", "contacts.npz",
        "worker/eye_columns.npz", "worker/visual_drive.npz",
        "worker/muscle_channels.npz", "worker/neural_state_diagnostic.npz",
    )
    refs: dict[str, Any] = {}
    for case in CASES:
        run_dir = REFERENCE_V2 / "runs" / case["prefix_reference_case_id"]
        records = {"run-record.json": sha_record(run_dir / "run-record.json")}
        records.update({name: sha_record(run_dir / name) for name in names})
        debug_snapshots = sorted((run_dir / "worker/debug-neural-state").glob("*.npz"))
        if len(debug_snapshots) != 41:
            raise RunnerError(f"prefix reference must contain 41 full-state snapshots: {run_dir}")
        records.update({
            f"worker/debug-neural-state/{path.name}": sha_record(path)
            for path in debug_snapshots
        })
        refs[case["case_id"]] = {
            "reference_case_id": case["prefix_reference_case_id"],
            "reference_run_dir": str(run_dir.resolve()),
            "reference_run_record": sha_record(run_dir / "run-record.json"),
            "files": records,
            "audit_scope": "post-completion exact-prefix audit only; old outputs are never inputs to new actions",
        }
    return refs


def _read_systemd_memory_peak() -> dict[str, Any]:
    unit = os.environ.get("ONLINE_FREEBODY_CASE_UNIT")
    cgroup_path = None
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                cgroup_path = parts[2]
                break
        if not cgroup_path:
            raise OSError("cgroup v2 path was not present in /proc/self/cgroup")
        cgroup_dir = Path("/sys/fs/cgroup") / cgroup_path.lstrip("/")
        raw_max = (cgroup_dir / "memory.max").read_text().strip()
        raw_peak = (cgroup_dir / "memory.peak").read_text().strip()
        return {
            "unit": unit,
            "control_group": cgroup_path,
            "memory_max_bytes": None if raw_max == "max" else int(raw_max),
            "memory_peak_bytes": int(raw_peak),
            "source": str(cgroup_dir / "memory.peak"),
            "recorded_while_case_unit_active": True,
            "measurement_scope": "systemd case scope cumulative peak, read after physical capture and before small final metadata sidecars",
        }
    except (OSError, ValueError) as exc:
        return {
            "unit": unit,
            "control_group": cgroup_path,
            "memory_max_bytes": None,
            "memory_peak_bytes": None,
            "source": "unavailable",
            "recorded_while_case_unit_active": False,
            "error": str(exc),
            "risk_boundary": "5 GiB MemoryMax is a hard cap, not an observed peak; report peak as pending",
            "measurement_scope": "unavailable",
        }


def _resource_record_matches(resource: dict[str, Any]) -> bool:
    peak = resource.get("memory_peak_bytes")
    cap = resource.get("memory_max_bytes")
    return (
        resource.get("recorded_while_case_unit_active") is True
        and isinstance(peak, int) and peak > 0
        and isinstance(cap, int) and cap > 0
        and peak <= cap <= 5 * 1024**3
        and isinstance(resource.get("unit"), str) and bool(resource["unit"])
        and isinstance(resource.get("control_group"), str)
        and resource["unit"] in resource["control_group"]
    )


def _all_case_files(run_dir: Path, omit: set[str] | None = None) -> dict[str, Any]:
    omit = omit or set()
    return {
        str(path.resolve()): sha_record(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in omit
    }


def _finalize_case(case: dict[str, Any], protocol: dict[str, Any], record: dict[str, Any]) -> None:
    run_dir = RUNS / case["case_id"]
    record_path = run_dir / "run-record.json"
    summary_path = run_dir / "window-summary.json"
    summary = read_json(summary_path)
    summary.update({
        "schema_version": "flygym-highcontrast-dynamics-window-summary-v1",
        "classification": protocol["classification"],
        "evidence_boundary": protocol["evidence_boundary"],
        "analysis_windows_s_half_open": protocol["analysis_windows_s_half_open"],
        "supplemental_windows_s_half_open": protocol["supplemental_analysis_windows_s_half_open"],
        "condition": {key: case[key] for key in ("case_id", "direction", "contrast", "wavelength_deg", "direction_deg_s", "temporal_frequency_hz")},
        "per_case_gray_baseline_subtraction_required": True,
        "per_direction_results_remain_separate": True,
        "matched_static_is_an_additional_engineering_control": True,
        "raw_retinal_slip_inputs": {
            "worker_eye_columns": str(run_dir / "worker/eye_columns.npz"),
            "online_pose_and_frame_times": str(run_dir / "online-frames.jsonl"),
            "raw_retinal_column_hashes_in_online_frames": True,
            "condition_label_enters_decoder_or_controller": False,
        },
        "response_units": "native thorax-heading yaw rad; positive is fly-left; display pose yaw is renderer-only -native",
        "interpretation_boundary": protocol["biological_comparison"],
    })
    write_json(summary_path, summary)
    record["schema_version"] = "flygym-free-body-online-highcontrast-dynamics-run-v1"
    record["classification"] = protocol["classification"]
    record["evidence_classification"] = protocol["classification"]
    record["evidence_boundary"] = protocol["evidence_boundary"]
    record["scope_status"] = protocol["evidence_boundary"]["scope_checker"]["status"]
    record["scope_interpretation"] = "candidate outside fixed-head calibration because free-body/full-pose input differs; C=1 contrast and wavelength match"
    record["protocol_path"] = str((BASE / "protocol.json").resolve())
    record["protocol_sha256"] = sha256(BASE / "protocol.json")
    record["post_capture_metadata_reconciled"] = {
        "inherited_lowcontrast_labels_replaced": True,
        "raw_physical_and_neural_capture_arrays_modified": False,
        "reused_runner_framework": str(LOW_RUNNER.resolve()),
    }
    record["resource_peak"] = _read_systemd_memory_peak()
    record["resource_peak_within_hard_limit"] = _resource_record_matches(record["resource_peak"])
    record["runtime_contract"] = {
        "duration_s": DURATION_S,
        "visual_frames": FRAME_COUNT,
        "neural_ticks": FRAME_COUNT * 10,
        "body_control_ticks": FRAME_COUNT * 5,
        "physics_steps": FRAME_COUNT * 5 * 10,
        "native_samples_including_initial": FRAME_COUNT * 5 * 10 + 1,
        "state_reset_at_t7": False,
    }
    write_json(record_path, record)
    integrity = {
        "schema_version": "flygym-highcontrast-dynamics-case-integrity-v1",
        "status": record.get("status"),
        "prefix_audit_status": "pending post-run audit",
        "resource_peak_observed": record["resource_peak_within_hard_limit"],
        "files": _all_case_files(run_dir, omit={"online-integrity.json"}),
    }
    write_json(run_dir / "online-integrity.json", integrity)


def _audit_completed_case(case_id: str) -> dict[str, Any]:
    protocol = read_json(BASE / "protocol.json")
    if sha256(PREFIX_AUDIT) != protocol["prefix_audit"]["auditor_sha256"]:
        raise RunnerError("read-only prefix auditor changed after preparation")
    case = next(row for row in protocol["stimuli"]["conditions"] if row["case_id"] == case_id)
    run_dir = RUNS / case_id
    reference = protocol["prefix_audit_references"][case_id]
    for role, record in reference["files"].items():
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RunnerError(f"frozen post-run prefix reference changed: {role} at {path}")
    candidate_record_path = run_dir / "run-record.json"
    candidate_record_bytes = candidate_record_path.read_bytes()
    pre_audit_path = run_dir / "analysis/pre-prefix-run-record.json"
    pre_audit_path.parent.mkdir(parents=True, exist_ok=True)
    pre_audit_path.write_bytes(candidate_record_bytes)
    audit_path = run_dir / "analysis/prefix-audit.json"
    command = [
        str(PREFIX_AUDIT_PYTHON), str(PREFIX_AUDIT),
        "--candidate-run", str(run_dir),
        "--reference-run", reference["reference_run_dir"],
        "--output", str(audit_path),
    ]
    completed = subprocess.run(command, cwd=PROJECT, check=False, capture_output=True, text=True)
    if not audit_path.is_file():
        raise RunnerError(f"prefix auditor did not write its report (exit={completed.returncode}): {completed.stderr[-1000:]}")
    audit_report = read_json(audit_path)
    audit_record = {
        "status": "passed" if completed.returncode == 0 and audit_report.get("passed") is True else "failed",
        "exit_code": completed.returncode,
        "schema_version": audit_report.get("schema_version"),
        "report_path": str(audit_path.resolve()),
        "report_sha256": sha256(audit_path),
        "pre_audit_run_record_path": str(pre_audit_path.resolve()),
        "pre_audit_run_record_sha256": sha256(pre_audit_path),
        "scientific_prefix_pass": audit_report.get("passed") is True,
        "post_run_only": True,
        "old_capture_is_not_an_action_input": True,
    }
    record = read_json(candidate_record_path)
    record["prefix_audit"] = audit_record
    record["prefix_audit_pass"] = audit_record["scientific_prefix_pass"] and completed.returncode == 0
    record["audit_run_record_hash_stage"] = "pre-prefix-audit run-record bytes; final record adds the audit index"
    record["prefix_audit_pass"] = bool(record["prefix_audit_pass"])
    write_json(candidate_record_path, record)
    index = {
        "schema_version": "flygym-highcontrast-prefix-audit-index-v1",
        "status": audit_record["status"],
        "auditor": sha_record(PREFIX_AUDIT),
        "report": sha_record(audit_path),
        "pre_audit_run_record": sha_record(pre_audit_path),
        "final_run_record_sha256": sha256(candidate_record_path),
        "audit_report_candidate_run_record_sha256": audit_report.get("run_record_sha256", {}).get("candidate"),
        "candidate_run_record_sha_stage": "bytes at audit invocation, before final audit index was attached",
    }
    write_json(run_dir / "analysis/prefix-audit-index.json", index)
    integrity = {
        "schema_version": "flygym-highcontrast-dynamics-case-integrity-v1",
        "status": record.get("status"),
        "prefix_audit_status": audit_record["status"],
        "prefix_audit_report": sha_record(audit_path),
        "resource_peak_observed": record.get("resource_peak_within_hard_limit"),
        "files": _all_case_files(run_dir, omit={"online-integrity.json"}),
    }
    write_json(run_dir / "online-integrity.json", integrity)
    if completed.returncode not in (0, 2):
        raise RunnerError(f"prefix audit errored with exit {completed.returncode}; report retained at {audit_path}")
    return record


def _prefix_match_gate(protocol: dict[str, Any]) -> dict[str, Any]:
    if len(CASES) != 5:
        raise RunnerError("the initial high-contrast batch must remain exactly five cases")
    if protocol.get("analysis_windows_s_half_open") != ANALYSIS_WINDOWS_REQUIRED:
        raise RunnerError("the frozen required analysis windows changed")
    return {"case_ids": list(CASE_IDS), "first_prefix_frames": 800, "post_run_audit_only": True}


def prepare() -> dict[str, Any]:
    if BASE.exists() and any(BASE.iterdir()):
        raise FileExistsError(f"refusing to overwrite prepared/executed bundle: {BASE}")
    low_protocol = read_json(LOW_PROTOCOL_PATH)
    low_freeze = read_json(LOW_FREEZE_PATH)
    low_manifest = read_json(LOW_PREPARE_MANIFEST_PATH)
    low_summary = read_json(LOW_COHORT_SUMMARY_PATH)
    if (low_manifest.get("status") != "prepared_not_executed"
            or low_summary.get("status") != "complete"
            or len(low_summary.get("case_records", [])) != 3):
        raise RunnerError("the frozen lowcontrast-v2 base cohort is incomplete")
    if low_protocol.get("source_freeze", {}).get("sha256") != sha256(LOW_FREEZE_PATH):
        raise RunnerError("lowcontrast-v2 source freeze does not match its protocol")
    for reference_case in ("stripes_static", "stripes_positive60", "stripes_negative60"):
        run_dir = REFERENCE_V2 / "runs" / reference_case
        for required in ("run-record.json", "online-frames.jsonl", "worker-requests.jsonl", "control-ticks.jsonl", "worker/frames.jsonl", "worker/eye_columns.npz"):
            if not (run_dir / required).is_file():
                raise RunnerError(f"same-direction 4 s prefix reference is missing: {run_dir / required}")
    PREPARATION.mkdir(parents=True, exist_ok=True)
    SPECS.mkdir(parents=True, exist_ok=False)
    SCHEDULES.mkdir(parents=True, exist_ok=False)
    RUNS.mkdir(parents=True, exist_ok=False)
    ANALYSIS.mkdir(parents=True, exist_ok=False)

    prepared_files = {
        path: record
        for path, record in low_protocol.get("prepared_files", {}).items()
        if "/online-lowcontrast-v2/preparation/angular-stimulus-specs/" not in path
        and path != str(Path(low_protocol["runtime"]["worker_config"]).resolve())
    }
    worker_config_source = Path(low_protocol["runtime"]["worker_config"]).resolve(strict=True)
    worker_config_path = PREPARATION / "worker-config-highcontrast.json"
    worker_config = read_json(worker_config_source)
    if worker_config.get("max_frames") != 800:
        raise RunnerError("lowcontrast source worker config no longer has its pinned 800-frame value")
    worker_config["max_frames"] = FRAME_COUNT
    write_json(worker_config_path, worker_config)
    worker_config_record = sha_record(worker_config_path)
    prepared_files[worker_config_record["path"]] = worker_config_record
    case_records: list[dict[str, Any]] = []
    schedules: dict[str, dict[str, Any]] = {}
    for case in CASES:
        spec_path = SPECS / f"{case['case_id']}.json"
        schedule_path = SCHEDULES / f"{case['case_id']}.json"
        write_json(spec_path, make_spec(case))
        write_json(schedule_path, make_schedule(case))
        spec_record, schedule_record = sha_record(spec_path), sha_record(schedule_path)
        prepared_files[spec_record["path"]] = spec_record
        prepared_files[schedule_record["path"]] = schedule_record
        row = {
            **case,
            "spec_path": spec_record["path"], "spec_sha256": spec_record["sha256"],
            "angular_spec_path": spec_record["path"], "angular_spec_sha256": spec_record["sha256"],
            "schedule_path": schedule_record["path"], "schedule_sha256": schedule_record["sha256"],
            "gray_window_s": [0.0, 1.0], "stimulus_window_s": [1.0, 13.0],
            "prefix_reference_path": str((REFERENCE_V2 / "runs" / case["prefix_reference_case_id"]).resolve()),
        }
        case_records.append(row)
        schedules[case["case_id"]] = read_json(schedule_path)

    prefix_refs = _prefix_references()
    source_files = dict(low_freeze.get("files", {}))
    for name, path in {
        "lowcontrast_protocol": LOW_PROTOCOL_PATH,
        "lowcontrast_source_freeze": LOW_FREEZE_PATH,
        "lowcontrast_prepare_manifest": LOW_PREPARE_MANIFEST_PATH,
        "lowcontrast_completed_cohort": LOW_COHORT_SUMMARY_PATH,
        "highcontrast_reference_protocol": REFERENCE_V2_PROTOCOL,
        "highcontrast_reference_source_freeze": REFERENCE_V2_FREEZE,
        "biological_comparison_plan": BIOLOGICAL_PLAN,
        "base_runner": LOW_RUNNER,
        "neural_worker": WORKER,
        "new_runner": Path(__file__),
        "new_retina_adapter": RETINA_ADAPTER,
        "new_worker_bootstrap": BOOTSTRAP,
        "prefix_audit_tool": PREFIX_AUDIT,
        "prefix_audit_python": PREFIX_AUDIT_PYTHON,
        "base_renderer": STANDARD_RENDERER,
    }.items():
        if not Path(path).is_file():
            raise RunnerError(f"required source/provenance file is missing: {path}")
        source_files[str(Path(path).resolve())] = sha_record(Path(path))
    source_freeze = {
        "schema_version": "flygym-highcontrast-dynamics-source-freeze-v1",
        "status": "prepared_not_executed",
        "files": dict(sorted(source_files.items())),
        "direct_runtime_sources": {
            "runner": sha_record(Path(__file__)),
            "retina_adapter": sha_record(RETINA_ADAPTER),
            "worker_bootstrap": sha_record(BOOTSTRAP),
            "reused_body_worker_runner": sha_record(LOW_RUNNER),
            "reused_neural_worker": sha_record(WORKER),
            "base_angular_renderer": sha_record(STANDARD_RENDERER),
            "post_run_prefix_auditor": sha_record(PREFIX_AUDIT),
            "prefix_audit_python": sha_record(PREFIX_AUDIT_PYTHON),
            "highcontrast_worker_config": worker_config_record,
        },
        "prefix_audit_reference_hashes_are_not_runtime_action_inputs": True,
    }
    source_freeze_path = PREPARATION / "source-freeze.json"
    write_json(source_freeze_path, source_freeze)

    protocol = dict(low_protocol)
    protocol.update({
        "schema_version": "flygym-free-body-online-highcontrast-dynamics-protocol-v1",
        "suite_id": "online-highcontrast-dynamics-v1",
        "prepare_revision": 2,
        "supersedes_prepared_bundle": str((PARENT / "online-highcontrast-dynamics-v1").resolve()),
        "title": "C=1 long-duration free-body online stripe response and direction reversal screen",
        "classification": "C=1 true-online free-body visual response and direction-reversal engineering screen; outside the fixed-head calibration pose scope and not a biological validation",
        "status": "prepared_not_executed",
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "source_freeze": {
            "path": str(source_freeze_path.resolve()),
            "sha256": sha256(source_freeze_path),
            "schema_version": source_freeze["schema_version"],
        },
        "prepared_files": dict(sorted(prepared_files.items())),
        "case_ids": list(CASE_IDS),
        "cohort_ids": ["highcontrast-dynamics-seed1729"],
        "queue_order": list(CASE_IDS),
        "prepare_manifest_path": str((BASE / "prepare-manifest.json").resolve()),
        "prepared_inputs": {
            **low_protocol.get("prepared_inputs", {}),
            "worker_config": str(worker_config_path.resolve()),
            "retina_adapter": str(RETINA_ADAPTER.resolve()),
            "worker_bootstrap": str(BOOTSTRAP.resolve()),
            "case_angular_specs_dir": str(SPECS.resolve()),
            "case_retina_schedules_dir": str(SCHEDULES.resolve()),
        },
        "stimuli": {
            "schema_version": "flygym-highcontrast-continuous-retina-schedule-v1",
            "duration_s": DURATION_S,
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_dt_s": FRAME_DT_S,
            "frame_count": FRAME_COUNT,
            "body_control_ticks_per_case": FRAME_COUNT * 5,
            "physics_steps_per_case": FRAME_COUNT * 5 * 10,
            "native_samples_per_case": FRAME_COUNT * 5 * 10 + 1,
            "neural_ticks_per_case": FRAME_COUNT * 10,
            "seed": SEED,
            "contrast_c": CONTRAST,
            "wavelength_deg": WAVELENGTH_DEG,
            "gray_interval_s_half_open": [0.0, 1.0],
            "stimulus_interval_s_half_open": [1.0, 13.0],
            "switch_time_s_exact": 7.0,
            "condition_changes_retina_schedule_only": True,
            "condition_label_or_direction_supplied_to_action_path": False,
            "conditions": case_records,
            "schedules": schedules,
            "phase_continuity": "phase is integrated in fixed-arena coordinates; reversal changes velocity at exactly t=7 without resetting phase, worker, neural model, decoder, CPG, or body state",
        },
        "runtime": {
            **low_protocol.get("runtime", {}),
            "worker_config": str(worker_config_path.resolve()),
            "worker_max_frames": FRAME_COUNT,
            "stimulus_source": "live full-pose retinal rendering with per-frame schedule from the independent C=1 continuous-phase adapter",
            "retina_adapter_path": str(RETINA_ADAPTER.resolve()),
            "worker_bootstrap_path": str(BOOTSTRAP.resolve()),
            "one_worker_one_body_session_per_case": True,
            "all_13s_of_each_case_are_continuous": True,
            "condition_label_used_by_decoder_or_controller": False,
            "worker_eye_columns_capture_is_preserved": True,
        },
        "body": {
            "duration_s": DURATION_S, "frame_count": FRAME_COUNT,
            "frame_dt_s": FRAME_DT_S, "control_dt_s": 0.001,
            "control_ticks_per_frame": 5,
            "expected_control_ticks_per_case": FRAME_COUNT * 5,
            "native_dt_s": 0.0001, "native_steps_per_control_tick": 10,
            "expected_physics_steps_per_case": FRAME_COUNT * 5 * 10,
            "expected_native_samples_per_case": FRAME_COUNT * 5 * 10 + 1,
            "expected_neural_ticks_per_case": FRAME_COUNT * 10,
            "body_seed": SEED, "initial_neutral_action": {"P_control": 0.0, "signed_d": 0.0},
            "body_yaw_source": "native thorax-heading yaw: atan2(world_y, world_x) of local thorax +X; positive is fly-left",
            "renderer_pose_only_transform": "VirtualWorld F=diag(1,-1,1); virtual display yaw=-native yaw; never applied to body-yaw response sign",
        },
        "analysis_windows_s_half_open": ANALYSIS_WINDOWS_REQUIRED,
        "analysis_windows": {
            "units": "absolute seconds in the 0..13 s trace; every window is half-open [start,end)",
            "windows_s": {key: list(value) for key, value in WINDOWS.items()},
            "yaw_source": "unwrapped native thorax_heading_yaw_rad from local thorax +X projected into world XY; positive yaw is fly-left",
            "display_pose_yaw_mapping": "VirtualWorld display yaw=-native yaw only for renderer geometry",
            "ball_omega_z_substitution": False,
            "direction_interpretation": "report measured response; no automatic expected-sign assertion or biological success label",
            "gray_baseline_subtraction_per_case": True,
            "matched_static_is_separate_engineering_control": True,
            "direction_specific_results_stay_separate": True,
        },
        "supplemental_analysis_windows_s_half_open": {
            "full_stimulus_1_13": [1.0, 13.0],
            "stimulus_segment_1_7": [1.0, 7.0],
            "stimulus_segment_7_13": [7.0, 13.0],
        },
        "analysis_contract": {
            "yaw_rate_bin_width_s": 0.05,
            "report_all_time_bins_from_0_to_13_s": True,
            "report_raw_native_yaw_and_fly_left_positive_yaw": True,
            "gray_baseline_subtraction_per_case": True,
            "direction_specific_results_stay_separate": True,
            "matched_static_is_separate_engineering_control": True,
            "report_each_segment_cumulative_yaw": True,
            "reversal_latency_rule": "first time after t=7 with at least ten consecutive 50 ms bins whose gray- and matched-static-subtracted signed yaw rate is in the second stimulus direction; descriptive only, not a pass threshold",
            "time_bins_are_not_independent_biological_samples": True,
        },
        "biological_comparison": {
            "reference_plan": str(BIOLOGICAL_PLAN.resolve()),
            "reference_plan_sha256": sha256(BIOLOGICAL_PLAN),
            "mano_c1_specific_reference": "early stimulus-following response then delayed counter-stimulus response in the named c=1 tethered-ball reference; not a universal full-trial behavior rule",
            "classification": ["reference-compatible", "reference-incompatible", "not-directly-comparable"],
            "do_not_label_late_same_direction_as_universal_biological_pass": True,
            "do_not_label_late_reversal_as_automatic_bug": True,
            "report_measured_system_response_separately_from_mano_comparison": True,
            "single_seed_biological_pass_fail": False,
        },
        "decoder": {
            **low_protocol.get("decoder", {}),
            "baseline_payload": low_protocol["prepared_inputs"]["baseline_payload"],
            "baseline_rates_hz": {"DNp15/L": 0.0, "DNp15/R": 142.5},
            "normalization_hz": 142.5,
            "static_baseline_scope_extrapolation": "C=1 calibration contrast and wavelength match; full-pose online inputs remain outside the fixed-head calibration scope",
            "stimulus_label_or_case_branching": False,
        },
        "evidence_boundary": {
            "biological_follow_success_claim": False,
            "calibrated_static_reference": "historical fixed-head stationary C=1, lambda=60 deg baseline bL=0, bR=142.5 Hz; this cohort matches C=1 but uses free-body/full-pose input outside the fixed-head calibration scope",
            "calibration_pass": False,
            "high_contrast_primary_reference": "frozen v2 C=1 seed1729 static/+60/-60 captures supply only a post-run exact-prefix audit; their four-second windows are not the long-duration matched control",
            "low_contrast_reference_comparison": "not used by this C=1 cohort; Mano comparison is limited to the specific C=1 reference described under biological_comparison",
            "online_pose_scope": "C=1 contrast and lambda=60 match the fixed-head reference; free-body/full-pose rendering remains outside its fixed-head pose calibration family",
            "scope_checker": {
                "status": "candidate_outside_calibrated_scope",
                "exception_type": "UnsupportedBaselineError",
                "claim": "candidate only; no calibration pass and no biological follow-success claim",
                "calibrated_static_baseline_still_used": True,
                "calibrated_scope": {
                    "contrast": 1.0,
                    "wavelength_deg": WAVELENGTH_DEG,
                    "display_geometry": "fixed_arena_infinite_distance_angular_field",
                    "visual_family": "sinusoidal_angular_stripe",
                    "pose_mode": "fixed_head",
                },
                "requested_scope": {
                    "calibration_source_cohort_id": low_protocol["evidence_boundary"]["scope_checker"]["requested_scope"]["calibration_source_cohort_id"],
                    "contrast": CONTRAST,
                    "wavelength_deg": WAVELENGTH_DEG,
                    "display_geometry": "fixed_arena_infinite_distance_angular_field",
                    "visual_family": "sinusoidal_angular_stripe",
                    "pose_mode": "free_body_online_pose",
                    "runtime_source_identity": {
                        "runner_sha256": sha256(Path(__file__)),
                        "retina_adapter_sha256": sha256(RETINA_ADAPTER),
                        "worker_config_sha256": worker_config_record["sha256"],
                        "worker_source_sha256": sha256(WORKER),
                        "source_freeze_sha256": sha256(source_freeze_path),
                        "seed": SEED,
                    },
                },
                "message": "the C=1 fixed-head reference matches stimulus contrast and wavelength but is out of scope because pose_mode='free_body_online_pose' (requires 'fixed_head')",
                "yaw_coordinate": "native thorax local +X projected into FlyGym world XY; positive is fly-left",
            },
        },
        "causal_clock": {
            **low_protocol["causal_clock"],
            "terminal_seq": FRAME_COUNT - 1,
            "terminal": f"seq{FRAME_COUNT - 1} response is decoded and saved but has no following body frame; record as pending and unapplied",
            "trace_duration_s": DURATION_S,
            "visual_frame_count": FRAME_COUNT,
            "expected_body_control_ticks": FRAME_COUNT * 5,
            "expected_physics_steps": FRAME_COUNT * 5 * 10,
            "expected_native_samples_including_initial": FRAME_COUNT * 5 * 10 + 1,
        },
        "resource_enforcement": {
            **low_protocol.get("resource_enforcement", {}),
            "case_child_environment_marker_is_not_enforcement_proof": True,
            "case_launcher": "sequential systemd transient case unit with MemoryMax=5G, CPUQuota=1200%, RuntimeMaxSec=1800; actual CPUQuotaPerSecUSec must be <=12000000",
            "case_runtime_verification": "record unit ControlGroup, MemoryMax and cumulative memory.peak after physical capture; require the unit path and peak<=max<=5 GiB; run post-capture prefix audit outside this scope",
            "cpu_quota_scope": "12-core-equivalent capacity cap applies to execution only; neural/body/stimulus parameters and sample cadence are unchanged by this resource limit",
            "queue_hard_limit": "runner takes an exclusive queue lock, runs cases sequentially, and stops at a 6 hour monotonic deadline",
        },
        "artifacts_per_case": [
            "run-record.json", "window-summary.json", "online-frames.jsonl", "worker-requests.jsonl",
            "control-ticks.jsonl", "worker-protocol-stdout.jsonl", "worker-stderr.log",
            "trajectory.npz", "contacts.npz", "online-integrity.json", "worker/frames.jsonl",
            "worker/provenance.json", "worker/eye_columns.npz", "worker/visual_drive.npz",
            "worker/muscle_channels.npz", "worker/neural_state_diagnostic.npz",
            "worker/debug-neural-state/", "analysis/pre-prefix-run-record.json",
            "analysis/prefix-audit.json", "analysis/prefix-audit-index.json",
        ],
        "prefix_audit_references": prefix_refs,
        "prefix_audit": {
            "reference_suite": str(REFERENCE_V2.resolve()),
            "auditor_path": str(PREFIX_AUDIT.resolve()),
            "auditor_sha256": sha256(PREFIX_AUDIT),
            "auditor_python": sha_record(PREFIX_AUDIT_PYTHON),
            "output_pattern": "runs/<case_id>/analysis/prefix-audit.json",
            "exact_prefix_duration_s": 4.0,
            "frames": 800,
            "compares": [
                "800 frames of online, worker request and worker frame records",
                "all scientific arrays in four worker NPZ files",
                "41 full-state debug snapshots by file hash",
                "4000 body control ticks except two validated perf_counter fields",
                "40001 native trajectory samples and all contacts within that prefix",
            ],
            "reference_is_read_only_after_new_case_completion": True,
            "reference_output_never_drives_new_case_actions": True,
            "audit_runs_outside_per_case_memory_peak_scope": True,
        },
        "resource_limits": {
            **RESOURCE_LIMITS,
            "runtime_max_seconds_per_case": CASE_RUNTIME_S,
            "runtime_max_seconds_per_queue": QUEUE_RUNTIME_S,
            "shared_nn_lock": str(GLOBAL_NN_LOCK),
            "queue_order": list(CASE_IDS),
            "case_execution": "sequential systemd user scopes under the shared exclusive neural-worker lock",
            "runtime_gate": "per-case RuntimeMaxSec=1800; 12-core CPUQuota; MemoryMax=5G",
        },
        "resource_memory_risk": {
            "historical_7s_static_trajectory_compressed_bytes": 414000000,
            "historical_7s_static_trajectory_uncompressed_gib": 0.777,
            "historical_7s_static_contacts_compressed_bytes": 12000000,
            "historical_7s_static_contacts_uncompressed_gib": 0.097,
            "linear_13s_estimate_trajectory_uncompressed_gib": 1.443,
            "linear_13s_estimate_contacts_uncompressed_gib": 0.180,
            "estimate_excludes": ["Python contact arrays and object overhead", "worker diagnostics", "serialization temporaries"],
            "historical_cgroup_memory_peak": "not retained in prior case records; only MemoryMax was recorded",
            "new_case_must_record_cgroup_memory_peak": True,
            "new_case_memory_peak_scope": "whole systemd case scope, read after physical capture/window summary and before small final sidecars; post-run prefix audit runs outside the case cgroup",
            "risk_statement": "5 GiB is a hard cap, not evidence of sufficient headroom; no raw fields or sample cadence are removed",
        },
        "locks": {
            "queue_lock": str(BASE / "queue.lock"),
            "global_nn_lock": str(GLOBAL_NN_LOCK),
            "policy": "nonblocking flock; one case at a time; stop after first failure; no automatic retry",
        },
        "interpretation_limits": {
            "classification": "single-seed engineering dynamics screen",
            "natural_behavior_pass_claim": False,
            "statistical_validation": False,
            "closed_loop": True,
            "body_to_retina_feedback": True,
            "brain_proprioceptive_feedback": False,
            "gait_claim": False,
            "net_yaw_alone_is_success": False,
        },
        "status": "prepared_not_executed",
        "commands": {
            "working_directory": str(PROJECT.resolve()),
            "prepare": f"/usr/bin/python3.12 {Path(__file__).resolve()} --prepare",
            "execute_after_root_review": f"/usr/bin/python3.12 {Path(__file__).resolve()} --execute",
            "prefix_audit_after_each_completed_case": f"/usr/bin/python3.12 {PREFIX_AUDIT.resolve()} --candidate-run <run-dir> --reference-run <matched-v2-run-dir> --output <run-dir>/analysis/prefix-audit.json",
        },
    })
    protocol_path = BASE / "protocol.json"
    write_json(protocol_path, protocol)

    readme = (
        "# Online high-contrast dynamics v1\n\n"
        "Preparation revision 4. Earlier preparations and zero-frame startup failures are preserved; this active bundle is a sibling `online-highcontrast-dynamics-v1-revision-4/`. Asset validation reads the hash-bound external freeze; schedule validation reads the worker's nested stimulus description.\n\n"
        "Prepared only; no NN/GPU/body execution was started. This five-case, seed-1729 engineering screen uses 13 s at 200 Hz: gray `[0,1)`, then C=1, 60-degree sinusoidal stripes through `[1,13)`. Three cases are static/+60/-60 deg/s; two reverse exactly at t=7 s in opposite orders.\n\n"
        "The independent retinal adapter integrates fixed-arena phase across the reversal seam. It selects retinal pixels only; the frozen worker, decoder, CPG, and free-body session continue without reset. Native thorax yaw positive is fly-left; the reflected VirtualWorld renderer pose is a separate rendering coordinate.\n\n"
        "The first 800 frames of each completed run are checked against the matching 4 s `online-free-body-stripes-v2` case only after that new run finishes. The old outputs never drive new actions. The analysis reports each direction separately, subtracts each case gray baseline, and includes matched static as an additional engineering control. The Mano c=1 trace is a specific early-follow/late-counter reference, not a universal pass rule.\n\n"
        "Each 13 s run contains 2,600 visual frames, 26,000 neural ticks, 13,000 body-control ticks, 130,000 physics steps, and 130,001 native samples. The 5 GiB cap is a hard ceiling; the linear trajectory+contacts estimate is about 1.623 GiB uncompressed before worker arrays, contact temporaries, and Python overhead. Runtime must record cgroup `memory.peak`.\n\n"
        "Execution requires a separate root review bound to the protocol, runner, and source-freeze hashes.\n"
        "\nCommands (from the repository root):\n\n"
        "```bash\n"
        "/usr/bin/python3.12 scripts/run_flygym_free_body_online_highcontrast_dynamics_v1.py --prepare\n"
        "/usr/bin/python3.12 scripts/run_flygym_free_body_online_highcontrast_dynamics_v1.py --execute\n"
        "```\n\n"
        "`--execute` is fail-closed until `root-execution-review.json` matches the frozen pins. It was not run while preparing this bundle.\n"
    )
    (BASE / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "schema_version": "flygym-highcontrast-dynamics-prepare-manifest-v1",
        "prepare_revision": 4,
        "status": "prepared_not_executed",
        "runner": sha_record(Path(__file__)),
        "protocol": sha_record(protocol_path),
        "source_freeze": sha_record(source_freeze_path),
        "case_ids": list(CASE_IDS),
        "case_count": len(CASES),
        "frames_per_case": FRAME_COUNT,
        "duration_s_per_case": DURATION_S,
        "resource_limits": {**RESOURCE_LIMITS, "runtime_max_seconds_per_case": CASE_RUNTIME_S},
        "prefix_auditor": sha_record(PREFIX_AUDIT),
        "worker_config": worker_config_record,
        "nn_forward_started": False,
        "gpu_used": False,
        "body_physics_steps": 0,
        "root_execution_review": "required and not present; preparation grants no execution authority",
        "revision_1_archive": str((PARENT / "online-highcontrast-dynamics-v1-revisions/revision-1").resolve()),
    }
    write_json(BASE / "prepare-manifest.json", manifest)
    integrity = {
        "schema_version": "flygym-highcontrast-dynamics-preparation-integrity-v1",
        "status": "prepared_not_executed",
        "files": {
            str(path.resolve()): sha_record(path)
            for path in sorted(BASE.rglob("*")) if path.is_file() and path.name != "preparation-integrity.json"
        },
    }
    write_json(BASE / "preparation-integrity.json", integrity)
    return {
        "status": "prepared_not_executed",
        "bundle": str(BASE),
        "protocol_sha256": sha256(protocol_path),
        "source_freeze_sha256": sha256(source_freeze_path),
        "case_ids": list(CASE_IDS),
        "nn_forward_started": False,
        "gpu_used": False,
        "body_physics_steps": 0,
    }


def verify_frozen_inputs(protocol: dict[str, Any]) -> None:
    freeze_path = Path(protocol["source_freeze"]["path"])
    if sha256(freeze_path) != protocol["source_freeze"]["sha256"]:
        raise RunnerError("highcontrast source-freeze file hash mismatch")
    freeze = read_json(freeze_path)
    for raw_path, record in freeze.get("files", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RunnerError(f"frozen runtime source changed or vanished: {path}")
    for raw_path, record in protocol.get("prepared_files", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RunnerError(f"prepared schedule/config changed or vanished: {path}")


def check_root_review(protocol: dict[str, Any]) -> dict[str, Any]:
    review_path = BASE / "root-execution-review.json"
    review = read_json(review_path)
    if (review.get("status") != "approved"
            or review.get("protocol_sha256") != sha256(BASE / "protocol.json")
            or review.get("runner_sha256") != sha256(Path(__file__))
            or review.get("source_freeze_sha256") != protocol["source_freeze"]["sha256"]
            or review.get("execution_authority") != "authorized_after_review"):
        raise RunnerError("root review does not approve this exact protocol, runner, and source freeze")
    return review


def _phase_record(schedule: dict[str, Any], time_s: float) -> dict[str, Any]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from flygym_highcontrast_continuous_retina_v1 import phase_record

    return phase_record(schedule, time_s)


_ACTIVE_CASE: dict[str, Any] | None = None
_ACTIVE_SCHEDULE: dict[str, Any] | None = None


def _configure_base(protocol: dict[str, Any], cases: tuple[dict[str, Any], ...]) -> Any:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_flygym_free_body_online_lowcontrast_v2 as base

    base.BASE, base.PREPARATION, base.SPECS, base.RUNS = BASE, PREPARATION, SPECS, RUNS
    base.FRAME_COUNT, base.DURATION_S = FRAME_COUNT, DURATION_S
    base.ONSET_S, base.STIMULUS_DURATION_S, base.STATIC_WARMUP_S = 1.0, 12.0, 1.0
    base.STUDY_CONTRAST = CONTRAST
    base.CASES = cases
    base.CASE_IDS = tuple(row["case_id"] for row in cases)
    base.COHORT_IDS = tuple(dict.fromkeys(row["cohort_id"] for row in cases))
    base.WINDOWS = dict(WINDOWS)
    base.P_DEADBAND = P_DEADBAND
    base.GLOBAL_NN_LOCK = GLOBAL_NN_LOCK
    base.WORKER_PYTHON = WORKER_PYTHON
    base.WORKER = WORKER
    base.CASE_RUNTIME_S = CASE_RUNTIME_S
    base.QUEUE_RUNTIME_S = QUEUE_RUNTIME_S
    base.RESOURCE_LIMITS = {
        **base.RESOURCE_LIMITS,
        **RESOURCE_LIMITS,
        "runtime_max_seconds_per_case": CASE_RUNTIME_S,
        "runtime_max_seconds_per_queue": QUEUE_RUNTIME_S,
    }
    base.verify_frozen_inputs = verify_frozen_inputs
    base._worker_command = _worker_command

    def external_frozen_source_record(frozen: dict[str, Any], path: Path) -> dict[str, Any]:
        # The reused validator expects an inline map; this suite freezes it in
        # a separate file. Keep its path, size and live asset hash checks intact.
        freeze_spec = frozen["source_freeze"]
        freeze_path = Path(freeze_spec["path"]).resolve(strict=True)
        if sha256(freeze_path) != freeze_spec["sha256"]:
            raise RunnerError("external source-freeze hash changed during asset validation")
        resolved = Path(path).resolve(strict=True)
        record = read_json(freeze_path).get("files", {}).get(str(resolved))
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise RunnerError(f"runtime asset absent from external source freeze: {resolved}")
        return record

    base._frozen_source_record = external_frozen_source_record
    original_writer = base._write_jsonl

    def write_scheduled_frame(stream: TextIO, value: dict[str, Any]) -> None:
        if Path(stream.name).name == "online-frames.jsonl":
            t_s = float(value["retinal_pattern_time_s"])
            phase = _phase_record(_ACTIVE_SCHEDULE, t_s)  # type: ignore[arg-type]
            hashes = value.get("retinal_column_hashes", {})
            value["retinal_schedule_record"] = {
                **phase,
                "left_pixel_sha256": hashes.get("left"),
                "right_pixel_sha256": hashes.get("right"),
                "pixel_arrays_saved_in_worker_eye_columns": True,
                "action_authority": "none; retina record is not passed to decoder/controller",
            }
        original_writer(stream, value)

    base._write_jsonl = write_scheduled_frame
    original_ready = base._validate_worker_ready

    def validate_ready(ready: dict[str, Any], case: dict[str, Any], frozen: dict[str, Any]) -> None:
        original_ready(ready, case, frozen)
        assets = ready.get("source_assets", {})
        active_spec = assets.get("angular_stimulus_spec", {}) if isinstance(assets, dict) else {}
        source = active_spec.get("source", {}) if isinstance(active_spec, dict) else {}
        captured = active_spec.get("captured_copy", {}) if isinstance(active_spec, dict) else {}
        expected_spec_path = Path(case["spec_path"]).resolve(strict=True)
        expected_spec_sha = case["spec_sha256"]
        expected_spec_size = expected_spec_path.stat().st_size
        for label, item in (("source", source), ("captured_copy", captured)):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise RunnerError(f"current angular spec {label} path/hash/size role does not match the case pin")
            role_path = Path(item["path"]).resolve(strict=True)
            if (item.get("sha256") != expected_spec_sha
                    or item.get("size_bytes") != expected_spec_size
                    or role_path.stat().st_size != expected_spec_size
                    or sha256(role_path) != expected_spec_sha):
                raise RunnerError(f"current angular spec {label} bytes differ from the case pin")
        if Path(str(source.get("path"))).resolve() != expected_spec_path:
            raise RunnerError("worker active angular spec source path is not the current case spec")
        expected_capture = Path(RUNS / case["case_id"] / "worker/angular_stimulus_spec.json").resolve()
        if Path(str(captured.get("path"))).resolve() != expected_capture:
            raise RunnerError("worker angular spec captured-copy path is not in the current case output")
        angular = ready.get("angular_stimulus", {}).get("stimulus_description", {})
        if (angular.get("runtime_schedule_adapter_id") != "flygym-highcontrast-continuous-retina-v1"
                or angular.get("runtime_schedule_adapter_path") != str(RETINA_ADAPTER.resolve())
                or angular.get("runtime_schedule_adapter_sha256") != sha256(RETINA_ADAPTER)
                or angular.get("runtime_schedule_path") != case["schedule_path"]
                or angular.get("runtime_schedule_sha256") != case["schedule_sha256"]):
            raise RunnerError("worker ready did not bind the current retina adapter and schedule roles")
        if angular.get("runtime_schedule", {}).get("case_id") != case["case_id"]:
            raise RunnerError("worker ready schedule case ID differs from the current case")

    base._validate_worker_ready = validate_ready
    original_controller_metadata = base._controller_metadata

    def controller_metadata(controller: Any, selected_case: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
        result = original_controller_metadata(controller, selected_case, frozen)
        result.update({
            "highcontrast_schedule_path": selected_case["schedule_path"],
            "highcontrast_schedule_sha256": selected_case["schedule_sha256"],
            "condition_label_or_direction_to_controller": False,
            "neural_decoder_cpg_body_continuous_across_t7": True,
        })
        return result

    base._controller_metadata = controller_metadata
    return base


def _worker_command(case: dict[str, Any], output_dir: Path, protocol: dict[str, Any]) -> list[str]:
    return [
        str(WORKER_PYTHON), str(BOOTSTRAP), str(WORKER),
        str(case["schedule_path"]), str(case["schedule_sha256"]), str(case["case_id"]),
        "--config", str(protocol["runtime"]["worker_config"]),
        "--stimulus-condition", "angular",
        "--angular-stimulus-spec", str(case["spec_path"]),
        "--static-warmup-s", str(GRAY_END_S),
        "--seed", str(int(case["neural_seed"])),
        "--max-frames", str(FRAME_COUNT),
        "--neural-device", "cuda", "--neural-backend", "triton",
        "--vision-device", "cuda:0", "--max-vram-gib", "8.0",
        "--deterministic-vision", "--deterministic-neural-delivery",
        "--capture-motor-channels", "--capture-visual-drive",
        "--capture-neural-state-diagnostic", "--debug-neural-state-frames", "40",
        "--out", str(output_dir),
    ]


def _case_child(case_id: str) -> None:
    if os.environ.get("ONLINE_FREEBODY_CASE_SCOPE") != "1":
        raise RunnerError("case process is not inside the required systemd scope")
    protocol = read_json(BASE / "protocol.json")
    verify_frozen_inputs(protocol)
    check_root_review(protocol)
    case = next((row for row in protocol["stimuli"]["conditions"] if row["case_id"] == case_id), None)
    if case is None:
        raise RunnerError(f"unregistered case ID: {case_id}")
    schedule = read_json(Path(case["schedule_path"]))
    if sha256(Path(case["schedule_path"])) != case["schedule_sha256"]:
        raise RunnerError("runtime retina schedule changed after preparation")
    global _ACTIVE_CASE, _ACTIVE_SCHEDULE
    _ACTIVE_CASE, _ACTIVE_SCHEDULE = case, schedule
    base = _configure_base(protocol, tuple(protocol["stimuli"]["conditions"]))
    GLOBAL_NN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_NN_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = base._run_case(case, protocol)
    if record.get("status") != "complete":
        raise RunnerError("reused online body runner did not complete the case")
    _finalize_case(case, protocol, record)


def _launch_case(case_id: str) -> None:
    unit_name = f"online-highcontrast-{case_id}-{os.getpid()}-{time.monotonic_ns()}"
    command = [
        "systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet",
        f"--unit={unit_name}",
        f"--property=MemoryMax={RESOURCE_LIMITS['host_memory_max']}",
        f"--property=CPUQuota={RESOURCE_LIMITS['cpu_quota']}",
        f"--property=RuntimeMaxSec={CASE_RUNTIME_S}",
        "--setenv=ONLINE_FREEBODY_CASE_SCOPE=1",
        f"--setenv=ONLINE_FREEBODY_CASE_UNIT={unit_name}",
        str(BODY_PYTHON), str(Path(__file__).resolve()), "--case-child", "--case", case_id,
    ]
    subprocess.run(command, cwd=PROJECT, check=True, timeout=CASE_RUNTIME_S + 60)


def execute(case_id: str | None) -> None:
    protocol = read_json(BASE / "protocol.json")
    verify_frozen_inputs(protocol)
    check_root_review(protocol)
    if case_id is not None:
        if case_id not in CASE_IDS:
            raise RunnerError(f"unsupported case ID: {case_id}")
        _launch_case(case_id)
        record = _audit_completed_case(case_id)
        if not record.get("prefix_audit_pass"):
            raise RunnerError("exact first-800-frame prefix audit failed; completed case and audit report are retained")
        if not record.get("resource_peak_within_hard_limit"):
            raise RunnerError("case cgroup MemoryPeak was unavailable or exceeded its recorded 5 GiB cap")
        return
    queue_lock_path = BASE / "queue.lock"
    with queue_lock_path.open("a+") as queue_lock:
        fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {
            "schema_version": "flygym-highcontrast-dynamics-queue-status-v1",
            "status": "running", "case_order": list(CASE_IDS),
            "completed": [], "current": None, "started_utc_epoch_s": time.time(),
        }
        write_json(BASE / "queue-status.json", status)
        started = time.monotonic()
        try:
            for cid in CASE_IDS:
                if time.monotonic() - started > QUEUE_RUNTIME_S:
                    raise TimeoutError("highcontrast dynamics queue exceeded its registered wall limit")
                status["current"] = cid
                write_json(BASE / "queue-status.json", status)
                _launch_case(cid)
                record = _audit_completed_case(cid)
                if not record.get("prefix_audit_pass"):
                    raise RunnerError(f"exact prefix audit failed for {cid}; stopping the batch")
                if not record.get("resource_peak_within_hard_limit"):
                    raise RunnerError(f"cgroup MemoryPeak is unavailable or outside limits for {cid}; stopping the batch")
                status["completed"].append(cid)
                write_json(BASE / "queue-status.json", status)
            records = [read_json(RUNS / cid / "run-record.json") for cid in CASE_IDS]
            complete = all(
                row.get("status") == "complete"
                and row.get("prefix_audit_pass")
                and row.get("resource_peak_within_hard_limit")
                for row in records
            )
            write_json(BASE / "cohort-summary.json", {
                "schema_version": "flygym-highcontrast-dynamics-cohort-summary-v1",
                "status": "complete" if complete else "incomplete",
                "classification": "single-seed engineering dynamics screen; no biological pass/fail",
                "case_count": len(records), "case_records": records,
                "all_prefix_audits_pass": all(row.get("prefix_audit_pass") for row in records),
                "all_memory_peaks_observed": all(row.get("resource_peak_within_hard_limit") for row in records),
                "natural_behavior_success_claim": False,
                "statistical_validation": False,
            })
            status.update(status="complete" if complete else "incomplete", current=None, finished_utc_epoch_s=time.time())
        except BaseException:
            status.update(status="failed_stopped", current=status.get("current"), error=traceback.format_exc())
            raise
        finally:
            write_json(BASE / "queue-status.json", status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--case-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASE_IDS)
    args = parser.parse_args(argv)
    if args.prepare:
        if args.case:
            parser.error("--case cannot be combined with --prepare")
        print(json.dumps(prepare(), indent=2, sort_keys=True))
        return 0
    if args.case_child:
        if not args.case:
            parser.error("--case-child requires --case")
        _case_child(args.case)
        return 0
    if args.execute:
        execute(args.case)
        return 0
    parser.error("select --prepare or --execute")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_flygym_free_body_online_highcontrast_dynamics_v1: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
