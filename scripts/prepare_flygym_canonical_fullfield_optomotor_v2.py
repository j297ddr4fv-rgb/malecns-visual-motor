#!/usr/bin/env python3
"""Prepare revision 2 of the pinned nine-case canonical full-field cohort.

Preparation uses JSON and file hashes only.  It does not import FlyGym,
MuJoCo, MaleCNS, FlyVis, CUDA, or any simulation module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

sys.dont_write_bytecode = True

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
PARENT = Path("/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1")
BASE = PARENT / "canonical-fullfield-optomotor-v1/revision-2"
PREP = BASE / "preparation"
FREEZE_PATH = PREP / "source-freeze.json"
RUNS = BASE / "runs"
ANALYSIS = BASE / "analysis"

HIGH = PARENT / "online-highcontrast-dynamics-v1-revision-4"
HIGH_PROTOCOL = HIGH / "protocol.json"
HIGH_MANIFEST = HIGH / "prepare-manifest.json"
HIGH_FREEZE = HIGH / "preparation/source-freeze.json"
HIGH_RUNNER = SCRIPTS / "run_flygym_free_body_online_highcontrast_dynamics_v1.py"
BODY_RUNNER = SCRIPTS / "run_flygym_free_body_online_lowcontrast_v2.py"
RUNNER = SCRIPTS / "run_flygym_canonical_fullfield_optomotor_v2.py"
PREPARER = Path(__file__).resolve()

NATIVE_BASE = PARENT / "mechanical-native-model-reference-v1"
NATIVE_CAPTURE = NATIVE_BASE / "capture.json"
NATIVE_PREFLIGHT = NATIVE_BASE / "preflight.json"
NATIVE_MJB = NATIVE_BASE / "actual_session_model.mjb"
NATIVE_OPTIONS = NATIVE_BASE / "effective_model_options.json"
NATIVE_ORDER = NATIVE_BASE / "runtime_array_order.json"
EXPECTED_MJB_SHA256 = "6d7b1f8b9a7558cd5367be9e12e1a866adca3d98e6312d431b62d27ad747b2f3"
EXPECTED_MJB_SIZE = 23230821
EXPECTED_OPTIONS_SHA256 = "e6ed4b2ee31ccd3b3027afaa888391d36c6ce2f8b8dc2ed51e85c271cab2408e"
EXPECTED_ORDER_SHA256 = "0c8cdbb7006f774f42a19f068312605879a4ea7ca73e3db77e21e4903db14b0b"
EXPECTED_QPOS_SHA256 = "d292b37df0f9a83265423da2ce53b0b140d8ef6a49d6909e464b805453e75b99"
EXPECTED_QVEL_SHA256 = "1a0295f4bf5986c5f74eca9153a6a4cb10b073a01a76ba4a457fd862c78966a4"
BODY_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
GLOBAL_NN_LOCK = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-recorded-retina-neural-ball-harness-20261007-v2/locks/"
    "neural-worker-and-ball-case.lock"
)

OLD_AUDIT_FILES = (
    "analysis/positive-prefix-neural-divergence.json",
    "analysis/positive-prefix-stimulus-clock-audit.json",
    "analysis/negative-diagnostic-only-prefix-discrepancy-v1.json",
    "analysis/neural-execution-reproducibility-code-audit.json",
    "queue-status.json",
)


class PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise PreparationError(f"expected JSON object: {path}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def case_defs() -> list[dict[str, Any]]:
    rows = []
    for seed in (1729, 1730, 1731):
        cohort = f"canonical_fullfield_seed{seed}"
        for direction, sign, speed, old_ref in (
            ("static_c1", 1, 0.0, "stripes_static"),
            ("positive60", 1, 60.0, "stripes_positive60"),
            ("negative60", -1, -60.0, "stripes_negative60"),
        ):
            cid = f"{cohort}_{direction}"
            rows.append({
                "case_id": cid,
                "cohort_id": cohort,
                "direction": direction,
                "body_seed": seed,
                "neural_seed": seed,
                "seed_role": "same seed initializes body and neural worker; condition only selects retinal pixels",
                "contrast": 1.0,
                "wavelength_deg": 60.0,
                "head_direction_sign": sign,
                "head_angular_speed_deg_s": speed,
                "arena_angular_speed_deg_s": -speed,
                "direction_deg_s": speed,
                "speed_magnitude_deg_s": abs(speed),
                "temporal_frequency_hz": abs(speed) / 60.0,
                "switch_time_s": None,
                "duration_s": 13.0,
                "onset_s": 1.0,
                "frame_count": 2600,
                "prefix_reference_case_id": old_ref if seed == 1729 else None,
                "condition_label_passed_to_worker_decoder_or_controller": False,
            })
    return rows


def verify_highcontrast_parent() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, manifest, freeze = read_json(HIGH_PROTOCOL), read_json(HIGH_MANIFEST), read_json(HIGH_FREEZE)
    if manifest.get("status") != "prepared_not_executed":
        raise PreparationError("revision-4 prepare manifest is not the expected frozen prepared source")
    if manifest.get("protocol", {}).get("sha256") != sha256(HIGH_PROTOCOL):
        raise PreparationError("revision-4 protocol hash differs from its manifest pin")
    if manifest.get("source_freeze", {}).get("sha256") != sha256(HIGH_FREEZE):
        raise PreparationError("revision-4 source-freeze hash differs from its manifest pin")
    if protocol.get("source_freeze", {}).get("sha256") != sha256(HIGH_FREEZE):
        raise PreparationError("revision-4 protocol does not bind its source-freeze bytes")
    for raw, item in freeze.get("files", {}).items():
        path = Path(raw)
        if not path.is_file() or path.stat().st_size != item.get("size_bytes") or sha256(path) != item.get("sha256"):
            raise PreparationError(f"frozen revision-4 source or historical artifact changed: {path}")
    return protocol, manifest, freeze


def verify_native_reference(high_protocol: dict[str, Any]) -> dict[str, Any]:
    capture, preflight = read_json(NATIVE_CAPTURE), read_json(NATIVE_PREFLIGHT)
    if capture.get("status") != "complete":
        raise PreparationError("constructor-only native MJB capture is not complete")
    model = capture.get("model", {})
    if (model.get("actual_session_mjb_sha256") != EXPECTED_MJB_SHA256
            or model.get("actual_session_mjb_size_bytes") != EXPECTED_MJB_SIZE
            or sha256(NATIVE_MJB) != EXPECTED_MJB_SHA256):
        raise PreparationError("accepted constructor-only native MJB pin is absent or changed")
    if (sha256(NATIVE_OPTIONS) != EXPECTED_OPTIONS_SHA256
            or sha256(NATIVE_ORDER) != EXPECTED_ORDER_SHA256):
        raise PreparationError("accepted native model option/order reference changed")
    state = capture.get("constructor_state", {})
    if (state.get("qpos_sha256") != EXPECTED_QPOS_SHA256
            or state.get("qvel_sha256") != EXPECTED_QVEL_SHA256
            or capture.get("constructor_counters", {}).get("data_time_s") != 0.0
            or capture.get("constructor_counters", {}).get("constructor_native_samples") != 0
            or capture.get("constructor_counters", {}).get("control_samples") != 0
            or capture.get("integration_calls") not in (0, None)):
        raise PreparationError("constructor-only MJB capture no longer proves zero-step initial state")
    mech_config = preflight.get("walk_config", {})
    old_walk_path = Path(high_protocol["prepared_inputs"]["walk_effective_config"])
    old_walk = read_json(old_walk_path)["walk_config"]
    normalized_mech = {k: v for k, v in mech_config.items() if k not in {"duration_s", "seed"}}
    normalized_old = {k: v for k, v in old_walk.items() if k not in {"duration_s", "seed"}}
    if normalized_mech != normalized_old:
        raise PreparationError("canonical body WalkConfig differs from the accepted MJB factory outside runtime duration/seed")
    old_run = PARENT / "online-highcontrast-dynamics-v1-revision-4/runs/c1_static_highcontrast"
    if read_json(old_run / "model_options.json") != read_json(NATIVE_OPTIONS):
        raise PreparationError("revision-4 actual model options no longer match the native reference")
    if read_json(old_run / "array_order.json") != read_json(NATIVE_ORDER):
        raise PreparationError("revision-4 runtime array order no longer matches the native reference")
    initial = read_json(old_run / "run-record.json").get("initial_state", {})
    if initial.get("qpos_sha256") != EXPECTED_QPOS_SHA256 or initial.get("qvel_sha256") != EXPECTED_QVEL_SHA256:
        raise PreparationError("revision-4 seed-1729 no-step state differs from the accepted native capture")
    return {
        "status": "static_factory_equivalence_supported_runtime_mjb_guard_required",
        "accepted_native_mjb": record(NATIVE_MJB),
        "capture_json": record(NATIVE_CAPTURE),
        "preflight_json": record(NATIVE_PREFLIGHT),
        "effective_model_options": record(NATIVE_OPTIONS),
        "runtime_array_order": record(NATIVE_ORDER),
        "expected_seed1729_qpos_sha256": EXPECTED_QPOS_SHA256,
        "expected_seed1729_qvel_sha256": EXPECTED_QVEL_SHA256,
        "walk_config_comparison": "all fields exact after excluding only runtime duration_s and seed; these are separately pinned per case",
        "current_revision4_model_options_exact_match": True,
        "current_revision4_array_order_exact_match": True,
        "runtime_gate": "Each actual new session must be saved through mj_saveModel(session.model) before PhaseSpeedGaitController, worker launch, or stepping and byte-match the accepted MJB. Any mismatch fails closed.",
    }


def _old_reference_map(high_protocol: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    old_refs = high_protocol["prefix_audit_references"]
    high_id_by_ref = {
        row["prefix_reference_case_id"]: row["case_id"]
        for row in high_protocol["stimuli"]["conditions"]
    }
    mapping: dict[str, Any] = {}
    for case in cases:
        old_ref = case["prefix_reference_case_id"]
        if old_ref is None:
            continue
        old_high_id = high_id_by_ref[old_ref]
        mapping[case["case_id"]] = old_refs[old_high_id]
    return mapping


def _source_freeze(high_freeze: dict[str, Any], high_protocol: dict[str, Any], native: dict[str, Any],
                   prepared_files: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    files = dict(high_freeze["files"])
    explicit = [
        HIGH_PROTOCOL, HIGH_MANIFEST, HIGH_FREEZE, HIGH_RUNNER, BODY_RUNNER,
        NATIVE_CAPTURE, NATIVE_PREFLIGHT, NATIVE_MJB, NATIVE_OPTIONS, NATIVE_ORDER,
        SCRIPTS / "run_flygym_reference_walk.py",
        SCRIPTS / "flygym_highcontrast_continuous_retina_v1.py",
        SCRIPTS / "flygym_highcontrast_continuous_retina_bootstrap_v1.py",
        SCRIPTS / "flygym_online_neural_worker_eye_parity_state_v2.py",
        SCRIPTS / "flygym_phase_speed_gait_repaired_v1.py",
        SCRIPTS / "dnp15_opponent_steering_decoder_repaired_v1.py",
        SCRIPTS / "run_flygym_online_optomotor_eye_parity_v4_1_attempt5.py",
        SCRIPTS / "replay_flygym_free_body_stripes_v1.py",
        SCRIPTS / "standard_optomotor_stimulus.py",
        BODY_PYTHON,
        Path("/usr/bin/python3.12"),
        RUNNER, PREPARER,
    ]
    for leaf in OLD_AUDIT_FILES:
        explicit.append(HIGH / leaf)
    for case in cases:
        if case["body_seed"] != 1729:
            continue
        high_case = next(row for row in high_protocol["stimuli"]["conditions"]
                         if row["prefix_reference_case_id"] == case["prefix_reference_case_id"])
        high_run = HIGH / "runs" / high_case["case_id"]
        explicit.extend((high_run / "run-record.json", high_run / "analysis/prefix-audit.json"))
    for path in explicit:
        if not path.is_file():
            raise PreparationError(f"required frozen runtime/provenance input is missing: {path}")
        rec = record(path)
        files[rec["path"]] = rec
    for raw, rec in prepared_files.items():
        files[raw] = rec
    runtime_paths = sorted(
        raw for raw in files
        if ("/scripts/" in raw or raw.startswith("/data/environments/projects/flygym-reference/")
            or raw.startswith("/data/models/flyvis/") or raw.startswith("/data/cache/malecns-gpu/")
            or ("/runs/" not in raw and "/analysis/" not in raw and raw not in {str(HIGH_PROTOCOL), str(HIGH_MANIFEST)}))
    )
    return {
        "schema_version": "canonical-fullfield-optomotor-source-freeze-v2",
        "status": "prepared_not_executed",
        "source_bindings": {
            "highcontrast_revision4_protocol": record(HIGH_PROTOCOL),
            "highcontrast_revision4_source_freeze": record(HIGH_FREEZE),
            "highcontrast_runner_reused_as_binding_only": record(HIGH_RUNNER),
            "body_single_case_loop_reused_unchanged": record(BODY_RUNNER),
            "canonical_runner": record(RUNNER),
            "canonical_preparer": record(PREPARER),
        },
        "native_model_reference": {
            "capture_json_path": str(NATIVE_CAPTURE.resolve()),
            "model_mjb_path": str(NATIVE_MJB.resolve()),
            "model_mjb_sha256": EXPECTED_MJB_SHA256,
            "model_mjb_size_bytes": EXPECTED_MJB_SIZE,
            "effective_model_options_path": str((PREP / "native-model-effective-options.json").resolve()),
            "array_order_path": str((PREP / "native-model-array-order.json").resolve()),
        },
        "runtime_file_paths": runtime_paths,
        "historical_failures_are_immutable_inputs": True,
        "files": files,
    }


def _make_protocol(high_protocol: dict[str, Any], cases: list[dict[str, Any]], prepared_files: dict[str, Any],
                   native_proof: dict[str, Any], source_freeze: dict[str, Any], source_freeze_sha: str,
                   ref_map: dict[str, Any], prepare_manifest_path: Path) -> dict[str, Any]:
    p = json.loads(json.dumps(high_protocol))
    p.update({
        "schema_version": "canonical-fullfield-optomotor-protocol-v2",
        "suite_id": "canonical-fullfield-optomotor-v1-revision-2",
        "title": "Matched-seed C=1 full-field online optomotor engineering replication",
        "classification": "three-seed engineering visual-motor replication; descriptive only; no biological pass",
        "prepare_revision": 2,
        "status": "prepared_not_executed",
        "source_freeze": {"path": str(FREEZE_PATH.resolve()), "sha256": source_freeze_sha},
        "prepare_manifest_path": str(prepare_manifest_path.resolve()),
        "prepared_files": prepared_files,
        "case_ids": [row["case_id"] for row in cases],
        "cohort_ids": [f"canonical_fullfield_seed{s}" for s in (1729, 1730, 1731)],
        "queue_order": [row["case_id"] for row in cases],
        "prefix_audit_references": ref_map,
        "historical_v2_reproducibility": {
            "role": "secondary descriptive post-capture diagnostic for seed1729 only",
            "blocking": False,
            "existing_statuses": {
                "static": "exact 800-frame prefix pass",
                "positive60": "broad numeric neural/drive/body/retina prefix failure; root cause unresolved",
                "negative60": "one numeric whole-network Vm p50 difference at seq639; all other checked prefix fields pass",
            },
            "source_report_paths": [str((HIGH / leaf).resolve()) for leaf in OLD_AUDIT_FILES],
            "failure_policy": "Retain old failures in place and in this new source freeze. Save each new historical comparison report after capture. Its failure/exit-2 never aborts the new nine-case queue or becomes a behavior pass.",
        },
        "primary_preonset_integrity": {
            "status_before_execution": "not_run",
            "within_seed_reference": "static",
            "case_triplets": {str(seed): [row["case_id"] for row in cases if row["body_seed"] == seed]
                               for seed in (1729, 1730, 1731)},
            "exact_axes": {
                "visual_seq": [0, 199], "frame_time_s_half_open": [0.0, 1.0],
                "body_control_tick": [0, 999], "physics_step": [0, 9999],
                "native_sample_including_initial_and_t1_boundary": [0, 10000],
                "contact_event_cutoff_s_inclusive": 1.0,
            },
            "fields": ["constructor MJB/options/array order/qpos/qvel", "initial CPG phase and magnitude",
                       "requests and pose", "both retinal columns", "full saved visual current drive",
                       "neural spike/rate/P and available Vm/g/ref diagnostics", "queued/applied one-frame commands",
                       "control ticks/CPG phases/controller telemetry", "trajectory and contacts"],
            "failure_policy": "Any numeric pre-onset mismatch is retained with first differing field/axis and leaves that seed triplet unresolved for stimulus attribution. Continue capture for complete descriptive data; never relax equality or report a biological pass.",
            "coverage_limit": "Full mutable neural-state snapshots cover seq0..39 only; later prefix uses available selected-cell state, spikes/drives and periodic whole-network Vm summaries.",
        },
        "analysis_contract": {
            "analysis_scope": "descriptive constant-direction stimulus cohort plus exact same-seed pre-onset integrity",
            "report_raw_native_yaw_and_fly_left_positive_yaw": True,
            "report_each_registered_condition_separately": True,
            "time_bins_are_not_independent_biological_samples": True,
            "response_or_biological_pass_threshold": None,
            "reversal_condition_present": False,
            "reversal_latency_or_post-reversal_metric": "not measured in this constant-direction cohort",
        },
        "analysis_windows_s_half_open": {
            "gray_0_1": [0.0, 1.0],
            "stimulus_1_13": [1.0, 13.0],
            "stimulus_early_1_1p5": [1.0, 1.5],
            "stimulus_mid_2_4": [2.0, 4.0],
            "stimulus_mid_2_6": [2.0, 6.0],
            "stimulus_late_8_13": [8.0, 13.0],
        },
        "supplemental_analysis_windows_s_half_open": {},
        "analysis_windows": {
            "units": "absolute seconds in the 0..13 s trace; every window is half-open [start,end)",
            "windows_s": {
                "gray_0_1": [0.0, 1.0],
                "stimulus_1_13": [1.0, 13.0],
                "stimulus_early_1_1p5": [1.0, 1.5],
                "stimulus_mid_2_4": [2.0, 4.0],
                "stimulus_mid_2_6": [2.0, 6.0],
                "stimulus_late_8_13": [8.0, 13.0],
            },
            "yaw_source": "unwrapped native thorax_heading_yaw_rad; positive yaw is fly-left",
            "direction_interpretation": "descriptive measured response only; no automatic expected-sign assertion or pass label",
            "condition_scope": "static, constant +60 deg/s, and constant -60 deg/s; no direction reversal",
        },
        "biological_comparison": {
            "classification": "not_directly_comparable",
            "comparison_scope": "no biological comparison or follow-success claim is made by this engineering cohort",
            "reversal_scope": "no reversal condition is present; reversal latency and reversal-response metrics are not measured",
            "single_seed_biological_pass_fail": False,
        },
        "classification": "three-seed engineering visual-motor replication; constant-direction conditions; descriptive only; no biological pass",
        "evidence_boundary": {
            **high_protocol["evidence_boundary"],
            "biological_follow_success_claim": False,
            "low_contrast_reference_comparison": "no biological comparison is made by this constant-direction C=1 engineering cohort",
            "condition_scope": "static, constant +60 deg/s, and constant -60 deg/s; no stimulus direction reversal",
            "reversal_latency_or_response_claim": "not measured",
        },
        "native_model_identity": native_proof,
        "runtime": {
            **high_protocol["runtime"],
            "worker_config": str((PREP / "worker-config-highcontrast.json").resolve()),
            "condition_label_used_by_decoder_or_controller": False,
            "same_worker_body_controller_cpg_decoder_and_assets": True,
            "one_frame_nn_delay": "first frame applies neutral P=0,d=0; frame seq's neural output is available at (seq+1)*0.005s and applied to the next body frame; unchanged from pinned source",
            "reviewed_binding": "call frozen highcontrast runner._configure_base, then the existing lowcontrast runner._run_case exactly once per case, then frozen highcontrast runner._finalize_case; do not call the five-case highcontrast executor",
        },
        "body": {
            **high_protocol["body"],
            "duration_s": 13.0, "frame_count": 2600, "body_seed_role": "same body and neural seed within each triplet",
            "seed_triplets": [1729, 1730, 1731],
        },
        "stimuli": {
            **high_protocol["stimuli"],
            "contrast_c": 1.0, "wavelength_deg": 60.0, "duration_s": 13.0,
            "onset_s": 1.0, "gray_interval_s_half_open": [0.0, 1.0],
            "stimulus_interval_s_half_open": [1.0, 13.0],
            "frame_count": 2600, "frame_dt_s": 0.005,
            "switch_time_s_exact": None,
            "condition_label_or_direction_supplied_to_action_path": False,
            "condition_changes_retina_schedule_only": True,
            "phase_continuity": "the angular phase continues within each constant-speed condition; this protocol has no direction reversal or t=7 switch",
            "schedules": {row["case_id"]: read_json(Path(row["schedule_path"])) for row in cases},
            "conditions": cases,
        },
        "resource_limits": {
            **high_protocol["resource_limits"],
            "queue_order": [row["case_id"] for row in cases],
            "cpu_quota": "1000% per case",
            "cpu_quota_per_sec_usec_max": 10000000,
            "host_memory_max": "5G per case",
            "case_cpu_quota": "1000% (10 CPU cores)",
            "coordinator_cpu_quota": "200% (2 CPU cores)",
            "combined_cpu_quota_cores": 12,
            "runtime_max_seconds_per_case": 1800,
            "worker_max_vram_gib": 8.0,
            "runtime_max_seconds_per_queue": 18000,
            "case_execution": "sequential under the global exclusive NN lock; case child 10 CPU cores plus coordinator 2 CPU cores",
            "peak_measurement": "read and record actual memory.peak from each active case cgroup after capture; do not substitute configured cap or missing values",
        },
        "resource_enforcement": {
            "case_launcher": "sequential transient systemd case service, CPUQuota=1000%, MemoryMax=5G, RuntimeMaxSec=1800",
            "coordinator_launcher": "reviewed outer systemd service, CPUQuota=200%, RuntimeMaxSec=18000",
            "effective_total_cpu_cap_cores": 12,
            "case_memory_peak": "actual cgroup-v2 memory.peak must be read in the case service before teardown; missing or over-cap values fail the case",
            "global_nn_lock": str(GLOBAL_NN_LOCK.resolve()),
        },
        "execution_bindings": {
            "retina_and_body_loop": "frozen revision-4 runner._configure_base binding to frozen run_flygym_free_body_online_lowcontrast_v2._run_case; invoked once per case",
            "case_finalization": "frozen revision-4 runner._finalize_case",
            "old_five_case_executor_used": False,
            "old_hard_prefix_gate_used_for_queue_control": False,
            "precontroller_hook": "temporary in-process wrapper around FlyGymReferenceSession factory that calls mj_saveModel on session.model and checks MJB/options/order/initial state before returning; no body/NN/source parameter changes",
            "old_v2_prefix_auditor": "frozen read-only auditor invoked after each seed-1729 capture as a nonblocking historical diagnostic",
        },
        "prefix_audit": {
            **high_protocol["prefix_audit"],
            "role": "secondary historical reproducibility diagnostic against the four-second V2 reference",
            "blocking": False,
            "old_800_frame_exact_prefix_is_primary_gate": False,
            "report_location": "runs/<case-id>/analysis/prefix-audit.json; seed1729 cases only",
            "failure_handling": "retained in the case report and does not abort new case capture; no failure is converted to a pass",
        },
        "queue_fail_policy": {
            "stop_on": ["case runtime/nonzero exit", "native MJB or initial-state identity mismatch", "missing/over-cap actual case memory peak", "outer runtime cap"],
            "do_not_stop_on": ["secondary historical V2 prefix audit exit 2 or numeric mismatch", "pre-onset triplet comparison mismatch"],
            "nonblocking_mismatch_action": "retain raw files, record unresolved status, complete remaining registered cases when execution itself remains valid",
        },
        "interpretation_limits": {
            "classification": "multi-seed engineering visual-motor replication",
            "natural_behavior_pass_claim": False,
            "biological_pass_claim": False,
            "statistical_validation": False,
            "postonset_historical_reproducibility": "unresolved until the old V2 prefix discrepancy is explained; new matched runs do not erase it",
            "directional_claim": "describe observed condition and seed responses only; no manufactured pass threshold",
        },
        "locks": {
            **high_protocol.get("locks", {}),
            "queue_lock": str((BASE / "queue.lock").resolve()),
            "global_nn_lock": str(GLOBAL_NN_LOCK.resolve()),
            "policy": "one sequential queue; one exclusive global NN lock; no automatic retry",
        },
        "approval": {
            "execution_review_required": True,
            "review_path": str((BASE / "root-execution-review.json").resolve()),
            "prepare_creates_approval": False,
            "execution_authority": "not granted by preparation",
        },
        "commands": {
            "working_directory": str(PROJECT.resolve()),
            "prepare": f"/usr/bin/python3.12 {PREPARER.resolve()} --prepare",
            "verify": f"{BODY_PYTHON} {RUNNER} --verify",
            "execute_after_root_review": (
                f"systemd-run --user --wait --pipe --collect --unit=canonical-fullfield-optomotor-v1-r2 "
                f"--property=CPUQuota=200% --property=RuntimeMaxSec=18000 "
                f"--setenv=CANONICAL_COORDINATOR_UNIT=canonical-fullfield-optomotor-v1-r2.service "
                f"{BODY_PYTHON} {RUNNER} --execute"
            ),
            "case_child": "runner-owned: 10 CPU cores, MemoryMax=5G, RuntimeMaxSec=1800, max worker VRAM 8 GiB",
        },
    })
    p["source_freeze"]["files_sha256"] = source_freeze_sha
    return p


def prepare() -> dict[str, Any]:
    if BASE.exists():
        preserved = BASE / "provenance"
        unexpected = [path.name for path in BASE.iterdir() if path.name != "provenance"]
        if unexpected or not preserved.is_dir():
            raise FileExistsError(f"refusing to overwrite canonical cohort artifacts: {BASE}")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    high_module = importlib.import_module("run_flygym_free_body_online_highcontrast_dynamics_v1")
    high_protocol, high_manifest, high_freeze = verify_highcontrast_parent()
    native_proof = verify_native_reference(high_protocol)
    cases = case_defs()
    created_base = not BASE.exists()
    try:
        if created_base:
            BASE.mkdir(parents=True, exist_ok=False)
        (PREP / "stimulus-specs").mkdir(parents=True)
        (PREP / "retina-schedules").mkdir(parents=True)
        RUNS.mkdir()
        ANALYSIS.mkdir()
        worker_src = Path(high_protocol["runtime"]["worker_config"]).resolve(strict=True)
        worker_dest = PREP / "worker-config-highcontrast.json"
        shutil.copy2(worker_src, worker_dest)
        prepared_files: dict[str, Any] = {str(worker_dest.resolve()): record(worker_dest)}
        spec_schedule_meta: dict[str, dict[str, Any]] = {}
        for case in cases:
            spec_path = PREP / "stimulus-specs" / f"{case['case_id']}.json"
            schedule_path = PREP / "retina-schedules" / f"{case['case_id']}.json"
            spec = high_module.make_spec(case)
            schedule = high_module.make_schedule(case)
            # The frozen adapter validates a legacy `switch_s=7.0` clock field,
            # but these registered constant-direction schedules have no switch.
            # Keep that compatibility field exact and remove inapplicable
            # reversal claims from the saved, case-specific schedule metadata.
            schedule["clock"]["switch_s_role"] = "adapter compatibility value; no direction-change boundary"
            schedule["clock"]["direction_change_boundary_s"] = None
            schedule["phase_continuity"].pop("reversal_boundary_s", None)
            schedule["phase_continuity"].pop("no_luminance_jump_at_reversal", None)
            schedule["phase_continuity"]["direction_change_present"] = False
            schedule["phase_continuity"]["direction_change_boundary_s"] = None
            write_json(spec_path, spec)
            write_json(schedule_path, schedule)
            case.update({
                "spec_path": str(spec_path.resolve()), "spec_sha256": sha256(spec_path),
                "schedule_path": str(schedule_path.resolve()), "schedule_sha256": sha256(schedule_path),
            })
            prepared_files[str(spec_path.resolve())] = record(spec_path)
            prepared_files[str(schedule_path.resolve())] = record(schedule_path)
            old_walk = read_json(Path(high_protocol["prepared_inputs"]["walk_effective_config"]))["walk_config"]
            walk = dict(old_walk)
            walk.update({"duration_s": 13.0, "seed": case["body_seed"]})
            case["effective_walk_config"] = walk
            spec_schedule_meta[case["case_id"]] = {
                "spec": record(spec_path), "schedule": record(schedule_path),
                "gray_segment": [0.0, 1.0], "stimulus_interval": [1.0, 13.0],
            }
        options_dest = PREP / "native-model-effective-options.json"
        order_dest = PREP / "native-model-array-order.json"
        shutil.copy2(NATIVE_OPTIONS, options_dest)
        shutil.copy2(NATIVE_ORDER, order_dest)
        prepared_files[str(options_dest.resolve())] = record(options_dest)
        prepared_files[str(order_dest.resolve())] = record(order_dest)
        historical_refs = _old_reference_map(high_protocol, cases)
        freeze = _source_freeze(high_freeze, high_protocol, native_proof, prepared_files, cases)
        freeze_path = PREP / "source-freeze.json"
        write_json(freeze_path, freeze)
        freeze_sha = sha256(freeze_path)
        source_freeze_record = record(freeze_path)
        protocol_path = BASE / "protocol.json"
        manifest_path = BASE / "prepare-manifest.json"
        protocol = _make_protocol(high_protocol, cases, prepared_files, native_proof,
                                  freeze, freeze_sha, historical_refs, manifest_path)
        write_json(protocol_path, protocol)
        # Prepared bytes are also pinned in the source freeze; add that exact
        # ledger to the protocol only after the freeze's content is stable.
        manifest = {
            "schema_version": "canonical-fullfield-optomotor-prepare-manifest-v2",
            "status": "prepared_not_executed",
            "classification": "metadata-only preparation; no execution authority",
            "runner": record(RUNNER), "preparer": record(PREPARER),
            "protocol": record(protocol_path), "source_freeze": source_freeze_record,
            "cases": len(cases), "seeds": list((1729, 1730, 1731)),
            "frames_per_case": 2600, "duration_s_per_case": 13.0,
            "native_model_reference_sha256": EXPECTED_MJB_SHA256,
            "native_mjb_guard": "required before controller, worker launch or stepping",
            "nn_forward_started": False, "gpu_used": False, "body_physics_steps": 0,
            "root_execution_review": "required; no approval was created",
        }
        write_json(manifest_path, manifest)
        # Keep the prepare-manifest pin in the immutable plan after its final
        # protocol hash is known; the source freeze intentionally excludes the
        # derived protocol/manifest pair to avoid circular hashes.
        write_json(BASE / "root-execution-review.template.json", {
            "schema_version": "canonical-fullfield-root-review-template-v2",
            "status": "not_approved",
            "execution_authority": "not_granted",
            "protocol_sha256": sha256(protocol_path),
            "runner_sha256": sha256(RUNNER),
            "prepare_manifest_sha256": sha256(manifest_path),
            "source_freeze_sha256": freeze_sha,
            "review_scope": "revision 2: nine sequential cases; case CPU10/Memory5GiB/1800s, coordinator CPU2/outer18000s; exact precontroller MJB; historical V2 gate is secondary; same-seed pre-onset integrity is primary; constant-direction stimuli only and no reversal metric",
        })
        write_json(ANALYSIS / "native-model-static-compatibility.json", native_proof)
        write_json(ANALYSIS / "historical-v2-status.json", {
            "status": "unresolved_historical_reproducibility_diagnostic",
            "seed1729_case_prefix_statuses": {
                "static": "pass", "positive60": "fail_numeric_prefix", "negative60": "fail_one_numeric_vm_p50",
            },
            "original_failures_modified": False,
            "new_historical_audits_are_nonblocking": True,
            "source_paths": [str((HIGH / leaf).resolve()) for leaf in OLD_AUDIT_FILES],
        })
        write_json(BASE / "cohort-plan.json", {
            "schema_version": "canonical-fullfield-cohort-plan-v2",
            "status": "prepared_not_executed",
            "case_count": len(cases), "case_order": [row["case_id"] for row in cases],
            "seed_triplets": {str(seed): [row["case_id"] for row in cases if row["body_seed"] == seed]
                               for seed in (1729, 1730, 1731)},
            "case_schedule_hashes": spec_schedule_meta,
            "protocol_sha256": sha256(protocol_path),
            "source_freeze_sha256": freeze_sha,
            "historical_status": "unresolved; old failures retained",
            "no_model_or_simulation_constructed": True,
        })
        (BASE / "README.md").write_text(
            "# Canonical full-field optomotor revision 2\n\n"
            "Status: prepared, not executed. This metadata-only package defines nine sequential cases: "
            "three seeds, each with static, +60 deg/s, and -60 deg/s C=1 stripes. The period is 60 degrees, "
            "the gray prelude is [0,1) s, and each trace lasts 13 s.\n\n"
            "The revision-4 worker, retina adapter, body/CPG controller, decoder, and source assets are bound by "
            "the source freeze. The existing 4 s V2 prefix comparison remains a nonblocking historical diagnostic. "
            "The primary common-control check compares each same-seed triplet before onset, including frame, tick, "
            "physics-step, native-sample, and contact axes. A mismatch remains unresolved and is never turned into a pass.\n\n"
            "This revision compares static and constant +60/-60 deg/s conditions; it contains no direction reversal and reports no reversal latency.\n\n"
            "Each actual session must match the accepted native MJB before controller construction, worker launch, "
            "or stepping. Preparation does not authorize execution.\n\n"
            f"Prepare: `{protocol['commands']['prepare']}`\n\n"
            f"Verify pins only: `{protocol['commands']['verify']}`\n\n"
            f"Execute after a separate exact root review: `{protocol['commands']['execute_after_root_review']}`\n",
            encoding="utf-8",
        )
        integrity_files = {}
        for path in sorted(BASE.rglob("*")):
            if path.is_file() and path.name != "preparation-integrity.json":
                rec = record(path)
                integrity_files[rec["path"]] = rec
        write_json(PREP / "preparation-integrity.json", {
            "schema_version": "canonical-fullfield-preparation-integrity-v1",
            "status": "prepared_not_executed", "files": integrity_files,
        })
        return {
            "status": "prepared_not_executed", "bundle": str(BASE),
            "protocol_sha256": sha256(protocol_path), "runner_sha256": sha256(RUNNER),
            "prepare_manifest_sha256": sha256(manifest_path), "source_freeze_sha256": freeze_sha,
            "case_count": len(cases), "nn_forward_started": False, "gpu_used": False,
            "body_physics_steps": 0, "root_execution_review_created": False,
        }
    except BaseException:
        if created_base:
            shutil.rmtree(BASE, ignore_errors=True)
        elif BASE.exists():
            for path in BASE.iterdir():
                if path.name == "provenance":
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    args = parser.parse_args()
    if not args.prepare:
        parser.error("only --prepare is supported")
    print(json.dumps(prepare(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
