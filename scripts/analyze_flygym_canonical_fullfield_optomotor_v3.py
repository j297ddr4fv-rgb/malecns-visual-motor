#!/usr/bin/env python3
"""Offline descriptive analysis for the completed canonical full-field R3 cohort.

This program refuses to inspect case traces until it has a matching successful
root terminal receipt, nine complete case records, passing source/MJB/resource
gates, and the frozen same-seed pre-onset audit records. It never starts a
worker, body, neural runtime, or GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT = Path("/home/xvhang/Projects/malecns-gpu")
BASE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1/"
    "canonical-fullfield-optomotor-v1/revision-3"
)
RUNNER = PROJECT / "scripts/run_flygym_canonical_fullfield_optomotor_v3.py"
PREPARER = PROJECT / "scripts/prepare_flygym_canonical_fullfield_optomotor_v3.py"
ANALYZER = Path(__file__).resolve()
PROTOCOL_PATH = BASE / "protocol.json"
MANIFEST_PATH = BASE / "prepare-manifest.json"
FREEZE_PATH = BASE / "preparation/source-freeze.json"
ROOT_REVIEW_PATH = BASE / "root-execution-review.json"
ANALYSIS_CONTRACT_PATH = BASE / "analysis/offline-analysis-protocol-v3.json"
QUEUE_PATH = BASE / "queue-status.json"
COHORT_SUMMARY_PATH = BASE / "cohort-summary.json"
START_OBSERVATION_PATH = BASE / "analysis/root-live-start-observation-v1.json"
TERMINAL_RECEIPT_PATH = BASE / "analysis/root-execution-terminal-v1.json"
OUTPUT_JSON = BASE / "analysis/final-offline-analysis-v3.json"
OUTPUT_MD = BASE / "analysis/final-offline-analysis-v3.md"
BODY_TRACE_READER = PROJECT / "scripts/analyze_flygym_free_body_trace_v1.py"
FROZEN_PREFIX_AUDITOR = PROJECT / "scripts/run_flygym_canonical_fullfield_optomotor_v2.py"
FROZEN_PREFIX_AUDITOR_SHA256 = "69cfea56b30214f9777e795a42870988aaff957b726c4ef310b5ff27ae9ef2b4"
R7_REFERENCE_SUMMARY = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1/"
    "mechanical-steering-dose-calibration-v1/revision-7/analysis/"
    "v8-analyzer-prefix-fix-v1/snapshot-6f1286a59e715b84/mechanical-dose-summary.json"
)
R7_ROOT_ACCEPTANCE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1/"
    "mechanical-steering-dose-calibration-v1/revision-7/analysis/root-final-acceptance-v1.json"
)
R7_PROTOCOL = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1/"
    "mechanical-steering-dose-calibration-v1/revision-7/protocol.json"
)
ANG4_PROTOCOL = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1/"
    "online-highcontrast-dynamics-v1-revision-4/protocol.json"
)
ANG4_COMPLETED_MECHANICS = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1/"
    "online-highcontrast-dynamics-v1-revision-4/analysis/completed-two-case-mechanics.json"
)
ANG4_DIRECTIONAL_REVIEW = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1/"
    "online-highcontrast-dynamics-v1-revision-4/analysis/review-positive-negative-reversal-v1.json"
)

CASE_FRAMES = 2600
CASE_NEURAL_TICKS = 26000
CASE_BODY_TICKS = 13000
CASE_PHYSICS_STEPS = 130000
CASE_NATIVE_SAMPLES = 130001
CASE_RUNTIME_USEC = 1800 * 1_000_000
CASE_CPU_QUOTA_USEC = 10 * 1_000_000
CASE_MEMORY_MAX_BYTES = 5 * 1024**3
MECHANICS_WINDOW_S = (1.0, 13.0)
MECHANICS_EVENT_CRITERION = {
    "rise_mm": 0.05,
    "unloaded_min_s": 0.002,
    "loaded_before_min_s": 0.002,
    "loaded_after_min_s": 0.002,
    "complete_events_only": True,
    "zero_events_rule": "Zero detected events under this strict criterion is descriptive and is not a failure or proof that the leg did not cycle.",
}
MECHANICS_LOAD_DEFINITION = (
    "Per-leg loaded sample iff upward Fz >= 0.01 x total body weight, using the frozen reader formula body_mass_native.sum() x 9810 mm/s^2."
)
MECHANICS_EXPECTED_DTYPES = {
    "time_s": "float64",
    "thorax_heading_yaw_rad": "float64",
    "body_xpos_mm": "float32",
    "body_xquat_wxyz": "float32",
    "foot_contact_fz_up_native": "float32",
    "qpos": "float64",
    "qvel": "float64",
    "contacts.time_s": "float64",
    "contacts.step_index": "integer",
    "contacts.leg_index": "integer",
    "contacts.fly_body_segment": "unicode or bytes",
}
WINDOWS = {
    "stimulus_1_13": (1.0, 13.0),
    "segment_1_4": (1.0, 4.0),
    "segment_4_7": (4.0, 7.0),
    "segment_7_13": (7.0, 13.0),
}


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {label} at {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} is not a JSON object: {path}")
    return value


def check_hash_record(record: Any, path: Path, label: str) -> None:
    require(isinstance(record, dict), f"{label} has no file record")
    resolved = Path(path).resolve(strict=True)
    require(record.get("path") == str(resolved), f"{label} path differs from canonical path")
    require(record.get("size_bytes") == resolved.stat().st_size, f"{label} size differs")
    require(record.get("sha256") == sha256_file(resolved), f"{label} SHA-256 differs")


def _verify_prepared_pins(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pins = contract["prepared_pins"]
    manifest = read_json(MANIFEST_PATH, "R3 prepare manifest")
    protocol = read_json(PROTOCOL_PATH, "R3 protocol")
    freeze = read_json(FREEZE_PATH, "R3 source freeze")
    require(manifest.get("status") == "prepared_not_executed", "prepared manifest status differs")
    require(sha256_file(MANIFEST_PATH) == pins["prepare_manifest_sha256"],
            "current prepare manifest differs from the analysis contract pin")
    expected_pairs = (
        ("runner", RUNNER, manifest.get("runner")),
        ("preparer", PREPARER, manifest.get("preparer")),
        ("protocol", PROTOCOL_PATH, manifest.get("protocol")),
        ("source_freeze", FREEZE_PATH, manifest.get("source_freeze")),
    )
    for name, path, manifest_record in expected_pairs:
        expected_sha = pins[f"{name}_sha256"]
        require(sha256_file(path) == expected_sha, f"current {name} differs from the analysis contract")
        check_hash_record(manifest_record, path, f"prepared {name}")
    require(protocol.get("prepare_revision") == 3
            and protocol.get("suite_id") == "canonical-fullfield-optomotor-v1-revision-3"
            and protocol.get("status") == "prepared_not_executed",
            "R3 protocol identity or status differs")
    require(protocol.get("source_freeze", {}).get("sha256") == pins["source_freeze_sha256"],
            "protocol does not bind the prepared R3 source freeze")
    require(contract.get("schema_version") == "canonical-fullfield-r3-offline-analysis-protocol-v5",
            "offline-analysis contract schema differs")
    corrected_prefix = contract.get("corrected_scientific_prefix_audit", {})
    require(corrected_prefix == {
                "schema_version": "canonical-fullfield-corrected-scientific-prefix-v1",
                "frozen_auditor_path": str(FROZEN_PREFIX_AUDITOR),
                "frozen_auditor_sha256": FROZEN_PREFIX_AUDITOR_SHA256,
                "frame_jsonl": "worker/frames.jsonl",
                "frame_prefix_count": 200,
                "debug_frame_indices": list(range(40)),
                "debug_record_keys": sorted([
                    "schema_version", "capture_kind", "frame_index", "seq", "time_s",
                    "neural_window_start_step_index", "neural_window_end_step_index",
                    "engine_next_step_index", "host_held_input_sha256",
                    "host_gpu_input_array_equal", "host_gpu_input_shapes_and_dtypes_match",
                    "pinned_input_staging", "pinned_input_reason", "extra_device_to_host_capture",
                    "timing_equivalent_to_normal_run", "arrays", "path", "size_bytes", "sha256",
                ]),
                "debug_array_dtypes": {
                    "current_gpu": "float32", "probability_gpu": "float32", "v": "float32",
                    "g": "float32", "ref_until": "int32", "ring": "float32",
                    "spk": "uint8", "spike_tick": "int32",
                    "fixed_delivery_accumulator": "int64",
                },
                "debug_array_shape_rule": "all non-ring arrays are nonempty vectors of one shared length N; ring is [5,N]",
                "debug_reference_checks": [
                    "exact_case_local_canonical_path_and_no_symlink_escape",
                    "file_sha256_and_size_bytes",
                    "metadata_json_equals_frame_record_except_path_size_sha256",
                    "actual_npz_member_shape_dtype_and_array_sha256",
                    "cross_condition_debug_metadata_exact_after_path_removal",
                ],
                "only_normalized_field": "debug_neural_state.path",
                "raw_recomputation": "rerun the hash-pinned frozen audit_triplet and require its raw status/comparison rows to match the persisted audit exactly",
                "raw_registered_audit_is_preserved": True,
                "raw_gate_is_not_overridden": True,
            }, "corrected scientific prefix contract differs from the offline analyzer")
    expected_analysis_windows = {key: list(value) for key, value in WINDOWS.items()}
    require(contract.get("analysis_windows_s_half_open") == expected_analysis_windows,
            "post-capture analysis windows differ from the pinned analysis contract")
    require(contract.get("case_order") == protocol.get("queue_order"),
            "analysis contract case order differs from the prepared protocol")
    require(contract.get("completion_gates", {}).get("case_caps") == {
                "CPUQuotaPerSecUSec": CASE_CPU_QUOTA_USEC,
                "MemoryMax_bytes": CASE_MEMORY_MAX_BYTES,
                "RuntimeMaxUSec": CASE_RUNTIME_USEC,
            }, "analysis contract case caps differ from analyzer gates")
    mechanics_contract = contract.get("mechanics_analysis", {})
    require(mechanics_contract.get("window_s_half_open") == list(MECHANICS_WINDOW_S)
            and mechanics_contract.get("reference_event_criterion") == MECHANICS_EVENT_CRITERION
            and mechanics_contract.get("load_definition") == MECHANICS_LOAD_DEFINITION
            and mechanics_contract.get("window_contract", {}).get("native_sample_rate_hz") == 10000
            and mechanics_contract.get("window_contract", {}).get("expected_native_samples") == CASE_NATIVE_SAMPLES
            and mechanics_contract.get("window_contract", {}).get("expected_inclusive_window_samples") == 120001
            and mechanics_contract.get("window_contract", {}).get("required_dtypes") == MECHANICS_EXPECTED_DTYPES,
            "mechanical metrics/window/event/dtype contract differs from the analyzer source")
    require(mechanics_contract.get("output_scope") == {
                "all_nine_cases": True,
                "biological_gait_claim": False,
                "mechanical_pass_fail_gate": False,
                "zero_strict_foot_events_alone_fails": False,
                "initial_qpos_qvel_match": True,
                "contact_record_and_trace_flag_comparison": True,
            }, "mechanical evidence scope/claim boundary differs from the analyzer source")
    _verify_mechanics_reference_sources(contract)
    require(contract.get("analyzer", {}).get("sha256") == sha256_file(ANALYZER),
            "analysis source differs from the analysis contract pin")
    require(contract.get("analyzer", {}).get("path") == str(ANALYZER),
            "analysis source path differs from the analysis contract pin")
    return manifest, protocol, freeze


def _reference_ranges_from_r7(summary: dict[str, Any]) -> dict[str, Any]:
    case_results = summary.get("case_results", {})
    require(isinstance(case_results, dict) and len(case_results) == 15,
            "pinned R7 reference does not contain the complete 15-case cohort")
    rows = []
    for case_id, result in case_results.items():
        require(result.get("status") == "complete_validated",
                f"pinned R7 reference case is incomplete: {case_id}")
        metrics = result.get("metrics", {})
        require(metrics.get("authority_integrity_pass_recomputed") is True
                and result.get("model_gate_status") == "passed",
                f"pinned R7 reference case lacks validated model/authority metrics: {case_id}")
        rows.append((case_id, metrics))

    def range_of(values: list[float]) -> list[float]:
        return [float(min(values)), float(max(values))]

    ranges: dict[str, Any] = {}
    for name in (
        "forward_displacement_mm", "lateral_displacement_mm", "mean_contact_legs",
        "samples_with_at_least_two_contacting_legs_fraction", "net_yaw_change_deg",
        "peak_abs_yaw_rate_deg_s", "mean_roll_deg", "mean_pitch_deg", "sd_roll_deg",
        "sd_pitch_deg", "nonleg_ground_contact_sample_count",
        "thorax_ground_contact_sample_count",
    ):
        ranges[name] = range_of([float(metrics[name]) for _, metrics in rows])
    for name, source in (("max_abs_roll_deg", "roll_range_deg"),
                         ("max_abs_pitch_deg", "pitch_range_deg")):
        values = [max(abs(float(value)) for value in metrics[source]) for _, metrics in rows]
        ranges[name] = range_of(values)
    ranges["quaternion_norm_min_global"] = min(
        float(metrics["quaternion_quality"]["quaternion_norm_min"]) for _, metrics in rows
    )
    ranges["quaternion_norm_max_global"] = max(
        float(metrics["quaternion_quality"]["quaternion_norm_max"]) for _, metrics in rows
    )
    ranges["quaternion_norm_max_abs_error_global"] = max(
        float(metrics["quaternion_quality"]["quaternion_norm_max_abs_error_from_one"])
        for _, metrics in rows
    )
    ranges["per_leg_positive_fz_fraction"] = {
        leg: range_of([float(metrics["per_leg_positive_fz_fraction"][leg])
                       for _, metrics in rows])
        for leg in rows[0][1]["per_leg_positive_fz_fraction"]
    }
    return ranges


def _reference_ranges_from_ang4(report: dict[str, Any]) -> dict[str, Any]:
    cases = report.get("cases", {})
    require(set(cases) == {"c1_static_highcontrast", "c1_positive60_highcontrast"},
            "ANG4 partial mechanics source case coverage differs")
    rows = list(cases.values())
    windows = [row["windows"]["1_13"] for row in rows]

    def range_of(values: list[float]) -> list[float]:
        return [float(min(values)), float(max(values))]

    return {
        "native_yaw_delta_deg": range_of([float(row["native_yaw_delta_deg"]) for row in windows]),
        "world_xy_displacement_mm": {
            axis: range_of([float(row["thorax_world_displacement_mm"][axis]) for row in windows])
            for axis in ("x", "y")
        },
        "mean_contact_legs_found": range_of([
            float(row["support_contact_found"]["mean_legs"]) for row in windows
        ]),
        "fraction_samples_at_least_two_contact_found": range_of([
            sum(float(value) for key, value in row["support_contact_found"][
                "support_count_fractions_0_to_6"].items() if int(key) >= 2)
            for row in windows
        ]),
        "thorax_ground_contact_samples": range_of([
            float(row["thorax_ground_contact_samples"]) for row in windows
        ]),
        "nonleg_ground_contact_samples": range_of([
            float(row["nonleg_ground_contact_samples"]) for row in windows
        ]),
        "per_leg_contact_found_fraction": {
            leg: range_of([float(row["per_leg"][leg]["contact_found_fraction"])
                           for row in windows])
            for leg in rows[0]["windows"]["1_13"]["per_leg"]
        },
        "whole_capture_thorax_z_min_mm": range_of([
            float(row["whole_capture_mechanics"]["thorax_world_z_min_mm"]) for row in rows
        ]),
        "whole_capture_thorax_collision_clearance_min_mm": range_of([
            float(row["whole_capture_mechanics"]["minimum_thorax_collision_clearance_mm"])
            for row in rows
        ]),
    }


def _verify_mechanics_reference_sources(contract: dict[str, Any]) -> None:
    reference = contract["mechanics_analysis"]
    pins = reference["reference_sources"]
    for name, path in (
        ("body_trace_reader", BODY_TRACE_READER),
        ("r7_reference_summary", R7_REFERENCE_SUMMARY),
        ("r7_root_acceptance", R7_ROOT_ACCEPTANCE),
        ("r7_protocol", R7_PROTOCOL),
        ("ang4_protocol", ANG4_PROTOCOL),
        ("ang4_completed_mechanics", ANG4_COMPLETED_MECHANICS),
        ("ang4_directional_review", ANG4_DIRECTIONAL_REVIEW),
    ):
        expected = pins[name]
        require(expected.get("path") == str(path)
                and expected.get("sha256") == sha256_file(path),
                f"mechanics reference input changed: {name}")
    r7 = read_json(R7_REFERENCE_SUMMARY, "accepted R7 mechanics reference")
    acceptance = read_json(R7_ROOT_ACCEPTANCE, "R7 root acceptance record")
    require(r7.get("analysis_status") == "COMPLETE"
            and r7.get("biological_acceptance_status") == "NOT_ASSESSED"
            and len(r7.get("case_results", {})) == 15
            and acceptance.get("actual_cohort_exit_code") == 0
            and acceptance.get("actual_analysis_exit_code") == 0
            and acceptance.get("validated_cases") == 15
            and acceptance.get("analysis_json_sha256") == sha256_file(R7_REFERENCE_SUMMARY),
            "R7 prior engineering reference is not bound to the accepted complete cohort")
    expected_envelope = reference["prior_r7_engineering_reference_envelope"]
    require(expected_envelope.get("window_s") == r7.get("design", {}).get("analysis_window_s")
            and expected_envelope.get("case_count") == len(r7["case_results"])
            and expected_envelope.get("metric_ranges") == _reference_ranges_from_r7(r7),
            "predeclared R7 mechanics reference ranges differ from the accepted summary")
    ang4 = read_json(ANG4_COMPLETED_MECHANICS, "ANG4 partial mechanics reference")
    review = read_json(ANG4_DIRECTIONAL_REVIEW, "ANG4 directional review")
    require(ang4.get("status") == "partial_two_complete_three_not_started"
            and ang4.get("coverage", {}).get("complete_count") == 2
            and ang4.get("protocol", {}).get("gait_claim") is False
            and review.get("status") == "complete_read_only_independent_analysis",
            "ANG4 historical mechanics source no longer matches its declared partial scope")
    ang4_reference = reference["ang4_scope_context"]["partial_mechanics_1_13_reference"]
    require(ang4_reference.get("window_s") == [1.0, 13.0]
            and ang4_reference.get("cases") == ang4["coverage"]["complete_cases"]
            and ang4_reference.get("metric_ranges") == _reference_ranges_from_ang4(ang4),
            "predeclared ANG4 partial mechanics ranges differ from the pinned historical report")


def _verify_root_authority(contract: dict[str, Any], manifest: dict[str, Any],
                           protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    start = read_json(START_OBSERVATION_PATH, "root live-start observation")
    review = read_json(ROOT_REVIEW_PATH, "root execution review")
    terminal = read_json(TERMINAL_RECEIPT_PATH, "root execution terminal receipt")
    expected = contract["execution_binding"]
    require(review.get("status") == "approved"
            and review.get("execution_authority") == "authorized_after_review"
            and review.get("runtime_authorized") is True,
            "root execution review is absent or does not authorize runtime")
    require(review.get("protocol_sha256") == sha256_file(PROTOCOL_PATH)
            and review.get("runner_sha256") == sha256_file(RUNNER)
            and review.get("prepare_manifest_sha256") == sha256_file(MANIFEST_PATH)
            and review.get("source_freeze_sha256") == sha256_file(FREEZE_PATH),
            "root review does not bind the current R3 pins")
    review_sha = sha256_file(ROOT_REVIEW_PATH)
    require(review_sha == expected["root_approval_sha256"],
            "current root approval differs from the analysis contract pin")
    require(start.get("unit") == expected["unit"]
            and start.get("status") == "actual_live_first_case"
            and start.get("invocation_id") == expected["invocation_id"]
            and start.get("exec_session_id") == expected["exec_session_id"]
            and start.get("root_approval_sha256") == review_sha,
            "live-start observation differs from the expected R3 execution binding")
    require(start.get("scientific_outcome_accepted") is False,
            "live-start record must not claim a scientific outcome")
    require(terminal.get("schema_version") == "canonical-fullfield-r3-root-terminal-v1"
            and terminal.get("status") == "complete"
            and terminal.get("unit") == expected["unit"]
            and terminal.get("invocation_id") == expected["invocation_id"]
            and terminal.get("exec_session_id") == expected["exec_session_id"]
            and terminal.get("actual_exit_code") == 0
            and terminal.get("result") == "success",
            "root terminal receipt does not attest successful completion of the bound invocation")
    require(isinstance(terminal.get("terminal_source"), str)
            and "exec" in terminal["terminal_source"].lower()
            and "completion" in terminal["terminal_source"].lower(),
            "terminal receipt is not based on the actual execution completion")
    for key, path in (
        ("runner_sha256", RUNNER), ("preparer_sha256", PREPARER),
        ("protocol_sha256", PROTOCOL_PATH), ("prepare_manifest_sha256", MANIFEST_PATH),
        ("source_freeze_sha256", FREEZE_PATH), ("root_approval_sha256", ROOT_REVIEW_PATH),
    ):
        require(terminal.get(key) == sha256_file(path),
                f"terminal receipt does not bind current {key}")
    runtime_s = terminal.get("actual_runtime_s")
    require(isinstance(runtime_s, (int, float)) and not isinstance(runtime_s, bool)
            and 0 < float(runtime_s) <= 18000.0,
            "terminal receipt has an invalid actual runtime")
    require(terminal.get("case_count") == 9
            and terminal.get("case_ids") == protocol["queue_order"],
            "terminal receipt does not bind the full nine-case queue")
    require(manifest.get("status") == "prepared_not_executed"
            and protocol.get("execution_authorized") in (None, False),
            "prepared bundle status/authority fields changed unexpectedly")
    return start, review, terminal


def _verify_queue_completion(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = read_json(QUEUE_PATH, "R3 queue status")
    summary = read_json(COHORT_SUMMARY_PATH, "R3 cohort summary")
    case_ids = list(protocol["queue_order"])
    require(len(case_ids) == 9 and len(set(case_ids)) == 9,
            "prepared R3 queue does not contain nine unique cases")
    require(queue.get("status") == "complete_descriptive_capture"
            and queue.get("current") is None
            and queue.get("completed") == case_ids,
            "R3 queue is incomplete or case order differs; no case payloads were opened")
    require(summary.get("status") == "complete_descriptive_capture"
            and summary.get("case_count") == 9
            and [row.get("case_id") for row in summary.get("case_records", [])] == case_ids
            and summary.get("biological_pass_claim") is False,
            "R3 cohort summary is incomplete or makes an invalid biological claim")
    return queue, summary


def _verify_case_record(case: dict[str, Any], protocol: dict[str, Any],
                        freeze: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    run_dir = BASE / "runs" / case_id
    record = read_json(run_dir / "run-record.json", f"run record {case_id}")
    require(record.get("status") == "complete" and record.get("case_id") == case_id,
            f"case {case_id} is not complete")
    initial = record.get("initial_state", {})
    body_identity = record.get("body_session_identity", {})
    require(initial.get("cohort_id") == case.get("cohort_id")
            and initial.get("body_seed") == case.get("body_seed")
            and initial.get("neural_seed") == case.get("neural_seed")
            and body_identity.get("session_seed_argument") == case.get("body_seed")
            and body_identity.get("neural_worker_seed_expected") == case.get("neural_seed"),
            f"case {case_id} body/neural seed identity differs from the prepared row")
    require(record.get("protocol_path") == str(PROTOCOL_PATH.resolve())
            and record.get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
            f"case {case_id} did not use the pinned R3 protocol")
    require(record.get("trajectory_path") == str((run_dir / "trajectory.npz").resolve())
            and record.get("window_summary_path") == str((run_dir / "window-summary.json").resolve()),
            f"case {case_id} summary/trajectory paths differ from the run directory")
    require(record.get("worker_ready_pass") is True
            and record.get("worker_exit_code") == 0
            and record.get("worker_shutdown") is True,
            f"case {case_id} worker completion/ready gate failed")
    run_design = record.get("canonical_fullfield_protocol", {})
    require(run_design.get("condition_label_passed_to_decoder_or_controller") is False
            and run_design.get("one_frame_nn_command_delay_preserved") is True
            and run_design.get("direction_reversal_present") is False,
            f"case {case_id} full-field control-path invariants differ")
    expected_counts = {
        "completed_neural_frames": CASE_FRAMES,
        "completed_body_ticks": CASE_BODY_TICKS,
        "control_tick_count": CASE_BODY_TICKS,
        "physics_step_count": CASE_PHYSICS_STEPS,
        "native_sample_count": CASE_NATIVE_SAMPLES,
    }
    for key, value in expected_counts.items():
        require(record.get(key) == value, f"case {case_id} {key} differs: {record.get(key)!r}")
    runtime = record.get("runtime_contract", {})
    runtime_expected = {
        "visual_frames": CASE_FRAMES,
        "neural_ticks": CASE_NEURAL_TICKS,
        "body_control_ticks": CASE_BODY_TICKS,
        "physics_steps": CASE_PHYSICS_STEPS,
        "native_samples_including_initial": CASE_NATIVE_SAMPLES,
        "direction_reversal_present": False,
    }
    for key, value in runtime_expected.items():
        require(runtime.get(key) == value, f"case {case_id} runtime_contract.{key} differs")
    stop_ack = record.get("worker_stop_ack", {})
    require(stop_ack.get("frame_count") == CASE_FRAMES
            and stop_ack.get("last_seq") == CASE_FRAMES - 1
            and math.isclose(float(stop_ack.get("last_time_s", math.nan)), 12.995,
                             rel_tol=0.0, abs_tol=1e-9),
            f"case {case_id} worker stop acknowledgement is incomplete")
    validation = record.get("worker_artifact_validation", {})
    for key in ("retinal_frames", "visual_drive_frames", "motor_channel_frames", "neural_state_rows"):
        require(validation.get(key) == CASE_FRAMES,
                f"case {case_id} worker artifact validation {key} differs")
    ready = record.get("worker_ready", {})
    require(ready.get("max_frames") == CASE_FRAMES
            and ready.get("seed") == case["neural_seed"]
            and ready.get("n_columns_per_eye") == 721,
            f"case {case_id} worker ready record differs from R3 contract")

    scope = record.get("systemd_case_scope", {})
    require(scope.get("unit") == record.get("resource_peak", {}).get("unit")
            and scope.get("cgroup_membership_matches_unit") is True
            and scope.get("CPUQuotaPerSecUSec") == CASE_CPU_QUOTA_USEC
            and scope.get("RuntimeMaxUSec") == CASE_RUNTIME_USEC
            and scope.get("MemoryMax_bytes") == CASE_MEMORY_MAX_BYTES,
            f"case {case_id} actual systemd case caps/cgroup binding differ")
    peak = record.get("resource_peak", {})
    require(record.get("resource_peak_within_hard_limit") is True
            and peak.get("recorded_while_case_unit_active") is True
            and peak.get("unit") == scope.get("unit")
            and peak.get("control_group") == scope.get("ControlGroup")
            and peak.get("memory_max_bytes") == CASE_MEMORY_MAX_BYTES
            and isinstance(peak.get("memory_peak_bytes"), int)
            and 0 < peak["memory_peak_bytes"] <= CASE_MEMORY_MAX_BYTES,
            f"case {case_id} lacks a valid live cgroup memory.peak")

    identity = read_json(run_dir / "precontroller-identity.json", f"precontroller identity {case_id}")
    native = protocol["native_model_identity"]
    accepted_mjb = native["accepted_native_mjb"]
    require(identity.get("guard_status") == "passed" and identity.get("gate_failures") == []
            and identity.get("case_id") == case_id
            and identity.get("seed") == case.get("body_seed")
            and identity.get("capture_position") == (
                "after FlyGymReferenceSession constructor; before PhaseSpeedGaitController, worker launch, and any body control/physics step"
            )
            and identity.get("compiled_model_mjb_sha256") == accepted_mjb["sha256"]
            and identity.get("compiled_model_mjb_size_bytes") == accepted_mjb["size_bytes"]
            and identity.get("effective_model_options_sha256") == native["effective_model_options"]["sha256"]
            and identity.get("array_order_sha256") == native["runtime_array_order"]["sha256"],
            f"case {case_id} constructor-only native MJB/options/array-order identity failed")
    require(identity.get("session_time_before_capture") == identity.get("session_time_after_capture")
            and identity.get("session_time_before_capture", {}).get("time_s") == 0.0
            and identity.get("session_time_after_capture", {}).get("time_s") == 0.0,
            f"case {case_id} constructor-only capture advanced the model")
    options_record = read_json(run_dir / "precontroller-model-options.json",
                               f"precontroller model options {case_id}")
    order_record = read_json(run_dir / "precontroller-array-order.json",
                             f"precontroller array order {case_id}")
    require(identity.get("effective_model_options") == options_record
            and identity.get("array_order") == order_record
            and sha256_file(run_dir / "precontroller-model-options.json")
                == identity.get("effective_model_options_sha256")
            and sha256_file(run_dir / "precontroller-array-order.json")
                == identity.get("array_order_sha256"),
            f"case {case_id} constructor model option/array-order artifacts differ from identity")
    mjb_path = Path(identity.get("compiled_model_mjb_path", ""))
    require(mjb_path.resolve(strict=True).is_relative_to(run_dir.resolve())
            and mjb_path.stat().st_size == accepted_mjb["size_bytes"]
            and sha256_file(mjb_path) == accepted_mjb["sha256"],
            f"case {case_id} saved native MJB file differs from the accepted model")
    runtime_sources = identity.get("runtime_upstream_source_hashes", {})
    require(isinstance(runtime_sources, dict) and bool(runtime_sources),
            f"case {case_id} omitted actual runtime source hashes")
    freeze_files = freeze.get("files", {})
    require(identity.get("runtime_upstream_source_hashes_sha256") == canonical_sha256(runtime_sources)
            and runtime_sources.get("source_commit_match") is True,
            f"case {case_id} runtime source provenance hash/commit gate differs")
    for map_name in ("source_sha256", "installed_source_sha256"):
        source_map = runtime_sources.get(map_name)
        require(isinstance(source_map, dict) and bool(source_map),
                f"case {case_id} runtime source provenance omitted {map_name}")
        for raw_path, digest in source_map.items():
            frozen = freeze_files.get(raw_path)
            require(isinstance(frozen, dict) and frozen.get("sha256") == digest,
                    f"case {case_id} runtime source is not bound by source freeze: {raw_path}")

    integrity_path = run_dir / "online-integrity.json"
    integrity = read_json(integrity_path, f"online integrity {case_id}")
    require(integrity.get("status") == "complete" and integrity.get("files"),
            f"case {case_id} online integrity record is incomplete")
    listed: set[Path] = set()
    for raw_path, file_record in integrity["files"].items():
        path = Path(raw_path).resolve(strict=True)
        require(path.is_relative_to(run_dir.resolve()) and file_record.get("path") == str(path),
                f"case {case_id} integrity path escapes its run directory")
        require(path.stat().st_size == file_record.get("size_bytes")
                and sha256_file(path) == file_record.get("sha256"),
                f"case {case_id} artifact hash/size mismatch: {path.name}")
        listed.add(path)
    required_artifacts = (
        "run-record.json", "window-summary.json", "trajectory.npz", "contacts.npz",
        "manifest.json", "array_order.json", "control-ticks.jsonl", "worker/eye_columns.npz",
        "worker/visual_drive.npz", "worker/neural_state_diagnostic.npz",
        "precontroller-model-options.json", "precontroller-array-order.json",
    )
    for relative in required_artifacts:
        require((run_dir / relative).resolve(strict=True) in listed,
                f"case {case_id} integrity omits required artifact {relative}")
    return {"record": record, "run_dir": run_dir, "precontroller_identity": identity,
            "online_integrity": integrity, "verified_artifact_count": len(listed)}


def _triplet_gate(protocol: dict[str, Any], queue: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    seeds = sorted({int(row["body_seed"]) for row in protocol["stimuli"]["conditions"]})
    require(seeds == [1729, 1730, 1731], "prepared R3 seed triplets differ from the analysis contract")
    for seed in seeds:
        path = BASE / "analysis" / f"preonset-triplet-audit-seed{seed}.json"
        audit = read_json(path, f"pre-onset triplet audit seed {seed}")
        allowed_statuses = (
            "exact_preonset_common_prefix",
            "preonset_mismatch_unresolved",
            "preonset_audit_error_unresolved",
        )
        require(audit.get("schema_version") == "canonical-fullfield-preonset-triplet-audit-v1"
                and audit.get("seed") == seed
                and audit.get("status") in allowed_statuses,
                f"seed {seed} audit has an unknown status/schema")
        comparisons = audit.get("comparisons", [])
        exact = (audit.get("status") == "exact_preonset_common_prefix"
                 and bool(comparisons)
                 and all(row.get("equal") is True for row in comparisons))
        if audit.get("status") == "exact_preonset_common_prefix":
            require(exact, f"seed {seed} claims exact prefix without all comparisons equal")
        expected_triplet = protocol["primary_preonset_integrity"]["case_triplets"][str(seed)]
        require(expected_triplet == [
                    f"canonical_fullfield_seed{seed}_static_c1",
                    f"canonical_fullfield_seed{seed}_positive60",
                    f"canonical_fullfield_seed{seed}_negative60",
                ], f"seed {seed} protocol does not define the registered same-seed triplet")
        if audit.get("status") != "preonset_audit_error_unresolved":
            require(audit.get("reference_condition") == "static_c1"
                    and audit.get("comparison_conditions") == ["positive60", "negative60"],
                    f"seed {seed} audit does not use the registered same-seed comparisons")
        queue_statuses = queue.get("primary_preonset_by_seed", {})
        if isinstance(queue_statuses, dict) and str(seed) in queue_statuses:
            require(queue_statuses[str(seed)] == audit["status"],
                    f"seed {seed} queue/audit terminal statuses differ")
        results[str(seed)] = {
            "status": audit["status"],
            "comparison_rows": len(comparisons),
            "all_comparisons_equal": exact,
            "audit_path": str(path),
            "audit_sha256": sha256_file(path),
            "error": audit.get("error"),
        }
    queue_statuses = queue.get("primary_preonset_by_seed", {})
    expected_statuses = {seed: row["status"] for seed, row in results.items()}
    require(queue_statuses == expected_statuses,
            "queue status does not preserve the terminal status of every pre-onset seed audit")
    passed = all(row["all_comparisons_equal"] for row in results.values())
    return {
        "status": "exact_preonset_common_prefix" if passed else "preonset_mismatch_unresolved",
        "passed": passed,
        "by_seed": results,
        "gate_rule": "strict per-comparison equality; no median or directional outcome can override a mismatch",
    }


def _load_frozen_prefix_auditor():
    require(sha256_file(FROZEN_PREFIX_AUDITOR) == FROZEN_PREFIX_AUDITOR_SHA256,
            "frozen R2 prefix auditor source changed")
    spec = importlib.util.spec_from_file_location(
        "canonical_fullfield_frozen_prefix_auditor_r2", FROZEN_PREFIX_AUDITOR,
    )
    require(spec is not None and spec.loader is not None,
            "could not load the pinned R2 prefix-audit helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The frozen implementation is reused with only its artifact-root globals
    # rebound to this completed R3 capture. No source or helper is modified.
    module.BASE = BASE
    module.RUNS = BASE / "runs"
    module.ANALYSIS = BASE / "analysis"
    module.PROTOCOL_PATH = PROTOCOL_PATH
    module.FREEZE_PATH = FREEZE_PATH
    module.ONSET_S = 1.0
    return module


def _debug_array_digest(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _validate_debug_frame_record(run_dir: Path, row: dict[str, Any], seq: int,
                                 np: Any) -> dict[str, Any]:
    debug = row.get("debug_neural_state")
    expected_keys = {
        "schema_version", "capture_kind", "frame_index", "seq", "time_s",
        "neural_window_start_step_index", "neural_window_end_step_index",
        "engine_next_step_index", "host_held_input_sha256",
        "host_gpu_input_array_equal", "host_gpu_input_shapes_and_dtypes_match",
        "pinned_input_staging", "pinned_input_reason", "extra_device_to_host_capture",
        "timing_equivalent_to_normal_run", "arrays", "path", "size_bytes", "sha256",
    }
    require(isinstance(debug, dict) and set(debug) == expected_keys,
            f"frame seq {seq} debug-state JSON schema differs")
    require(debug.get("schema_version") == "flygym-online-neural-state-capture-1"
            and debug.get("capture_kind") == "post_step_window_before_neuromuscular_decode"
            and debug.get("seq") == seq and debug.get("frame_index") == seq
            and debug.get("time_s") == row.get("time_s")
            and debug.get("neural_window_start_step_index")
                == row.get("neural_window_start_step_index")
            and debug.get("neural_window_end_step_index")
                == row.get("neural_window_end_step_index")
            and debug.get("engine_next_step_index")
                == debug.get("neural_window_end_step_index", -2) + 1
            and debug.get("extra_device_to_host_capture") is True
            and debug.get("timing_equivalent_to_normal_run") is False
            and isinstance(debug.get("pinned_input_staging"), bool),
            f"frame seq {seq} debug-state timing or capture fields differ")
    for key in ("host_held_input_sha256", "host_gpu_input_array_equal",
                "host_gpu_input_shapes_and_dtypes_match"):
        require(isinstance(debug.get(key), dict), f"frame seq {seq} debug-state {key} is invalid")
    require(set(debug["host_held_input_sha256"]) == {"current", "probability"}
            and all(isinstance(value, str) and len(value) == 64
                    for value in debug["host_held_input_sha256"].values()),
            f"frame seq {seq} held-input hashes are malformed")
    for key in ("host_gpu_input_array_equal", "host_gpu_input_shapes_and_dtypes_match"):
        require(set(debug[key]) == {"current", "probability"}
                and all(isinstance(value, bool) for value in debug[key].values()),
                f"frame seq {seq} debug-state {key} schema is invalid")

    expected_path = run_dir / "worker" / "debug-neural-state" / f"frame-{seq:04d}.npz"
    resolved_run_dir = run_dir.resolve(strict=True)
    resolved_expected = expected_path.resolve(strict=True)
    require(resolved_expected.is_relative_to(resolved_run_dir)
            and resolved_expected == expected_path
            and debug.get("path") == str(resolved_expected),
            f"frame seq {seq} debug-state path is not its own canonical case-local artifact")
    require(isinstance(debug.get("size_bytes"), int)
            and not isinstance(debug.get("size_bytes"), bool)
            and debug["size_bytes"] == resolved_expected.stat().st_size
            and isinstance(debug.get("sha256"), str)
            and debug["sha256"] == sha256_file(resolved_expected),
            f"frame seq {seq} debug-state NPZ size/SHA differs from its frame reference")

    expected_dtypes = {
        "current_gpu": "float32", "probability_gpu": "float32", "v": "float32",
        "g": "float32", "ref_until": "int32", "ring": "float32", "spk": "uint8",
        "spike_tick": "int32", "fixed_delivery_accumulator": "int64",
    }
    array_records = debug.get("arrays")
    require(isinstance(array_records, dict) and set(array_records) == set(expected_dtypes),
            f"frame seq {seq} debug-state array-name schema differs")
    with np.load(resolved_expected, allow_pickle=False) as archive:
        require(set(archive.files) == set(expected_dtypes) | {"metadata_json"},
                f"frame seq {seq} debug-state NPZ members differ from the capture schema")
        stored_json = np.asarray(archive["metadata_json"])
        require(stored_json.ndim == 0 and stored_json.dtype.kind in ("U", "S"),
                f"frame seq {seq} debug-state metadata_json is not a string scalar")
        try:
            stored_metadata = json.loads(str(stored_json.item()))
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"frame seq {seq} debug-state metadata_json is malformed") from exc
        expected_metadata = {key: value for key, value in debug.items()
                             if key not in {"path", "size_bytes", "sha256"}}
        require(isinstance(stored_metadata, dict)
                and canonical_sha256(stored_metadata) == canonical_sha256(expected_metadata),
                f"frame seq {seq} NPZ metadata differs from its frame record")
        inferred_n = None
        for name, expected_dtype in expected_dtypes.items():
            array = np.asarray(archive[name])
            record = array_records[name]
            require(isinstance(record, dict)
                    and set(record) == {"shape", "dtype", "sha256"}
                    and record.get("shape") == list(array.shape)
                    and record.get("dtype") == str(array.dtype)
                    and record.get("dtype") == expected_dtype
                    and record.get("sha256") == _debug_array_digest(array),
                    f"frame seq {seq} debug-state array metadata differs for {name}")
            if name == "ring":
                require(array.ndim == 2 and array.shape[0] == 5 and array.shape[1] > 0,
                        f"frame seq {seq} debug-state ring shape differs")
                inferred_n = int(array.shape[1])
            else:
                require(array.ndim == 1 and array.shape[0] > 0,
                        f"frame seq {seq} debug-state vector shape differs for {name}")
                if inferred_n is None:
                    inferred_n = int(array.shape[0])
                require(array.shape[0] == inferred_n,
                        f"frame seq {seq} debug-state vector length differs for {name}")
    return {key: value for key, value in debug.items() if key != "path"}


def _corrected_scientific_prefix_gate(protocol: dict[str, Any],
                                      raw_gate: dict[str, Any]) -> dict[str, Any]:
    """Recompute the registered prefix and normalize only validated path pointers."""
    auditor = _load_frozen_prefix_auditor()
    import numpy as np

    results: dict[str, Any] = {}
    seeds = sorted({int(row["body_seed"]) for row in protocol["stimuli"]["conditions"]})
    for seed in seeds:
        audit_path = BASE / "analysis" / f"preonset-triplet-audit-seed{seed}.json"
        saved_raw = read_json(audit_path, f"raw pre-onset audit seed {seed}")
        seed_cases = [row for row in protocol["stimuli"]["conditions"]
                      if int(row["body_seed"]) == seed]
        require([row["direction"] for row in seed_cases]
                == ["static_c1", "positive60", "negative60"],
                f"seed {seed} case ordering differs from the registered triplet")
        case_by_direction = {row["direction"]: row for row in seed_cases}
        raw_recomputed: dict[str, Any] | None = None
        try:
            raw_recomputed = auditor.audit_triplet(seed, seed_cases)
            raw_recomputation_matches = (
                auditor.canonical(raw_recomputed) == auditor.canonical(saved_raw)
            )
            require(raw_recomputation_matches,
                    f"seed {seed} frozen raw prefix recomputation differs from the saved audit")

            fresh_comparisons = list(raw_recomputed.get("comparisons", []))
            registered_non_frame = [row for row in fresh_comparisons
                                    if row.get("dataset") != "worker/frames.jsonl"]
            normalized_selected: dict[str, list[dict[str, Any]]] = {}
            validated_debug_count = 0
            for direction in ("static_c1", "positive60", "negative60"):
                case = case_by_direction[direction]
                run_dir = (BASE / "runs" / case["case_id"]).resolve(strict=True)
                rows = auditor._read_jsonl(run_dir / "worker/frames.jsonl", 200)
                normalized_rows: list[dict[str, Any]] = []
                for seq, row in enumerate(rows):
                    require(row.get("seq") == seq,
                            f"seed {seed}/{direction} worker frame sequence differs at {seq}")
                    selected = auditor._selected_row("worker/frames.jsonl", row)
                    if seq < 40:
                        normalized_debug = _validate_debug_frame_record(run_dir, row, seq, np)
                        selected["debug_neural_state"] = normalized_debug
                        validated_debug_count += 1
                    else:
                        require(row.get("debug_neural_state") is None,
                                f"seed {seed}/{direction} has an unregistered debug capture at seq {seq}")
                    normalized_rows.append(selected)
                normalized_selected[direction] = normalized_rows

            frame_checks: list[dict[str, Any]] = []
            reference_rows = normalized_selected["static_c1"]
            for direction in ("positive60", "negative60"):
                candidate_rows = normalized_selected[direction]
                mismatch = None
                for index, (left, right) in enumerate(zip(reference_rows, candidate_rows)):
                    if auditor.canonical(left) != auditor.canonical(right):
                        keys = set(left) | set(right)
                        fields = sorted(key for key in keys
                                        if key not in left or key not in right
                                        or left.get(key) != right.get(key))
                        mismatch = {"first_mismatch_index": index, "fields": fields}
                        break
                frame_checks.append({
                    "condition": direction,
                    "dataset": "worker/frames.jsonl",
                    "equal": mismatch is None and len(candidate_rows) == 200,
                    "axis": "seq-0..199; debug path removed only after case-local validation",
                    "rows_compared": min(len(reference_rows), len(candidate_rows)),
                    **(mismatch or {}),
                })

            non_frame_equal = all(row.get("equal") is True for row in registered_non_frame)
            frames_equal = all(row.get("equal") is True for row in frame_checks)
            raw_seed = raw_gate["by_seed"].get(str(seed), {})
            raw_seed_matches_saved = (
                raw_seed.get("status") == saved_raw.get("status")
                and raw_seed.get("audit_path") == str(audit_path)
                and raw_seed.get("audit_sha256") == sha256_file(audit_path)
            )
            corrected_pass = (raw_recomputation_matches and non_frame_equal and frames_equal
                              and validated_debug_count == 120 and raw_seed_matches_saved)
            results[str(seed)] = {
                "status": ("scientific_prefix_exact_after_validated_debug_path_exclusion"
                           if corrected_pass else "scientific_prefix_unresolved"),
                "raw_saved_status": saved_raw.get("status"),
                "raw_saved_audit_sha256": sha256_file(audit_path),
                "raw_recomputation_status": raw_recomputed.get("status"),
                "raw_recomputation_matches_saved_audit": raw_recomputation_matches,
                "raw_false_comparison_rows": [row for row in fresh_comparisons
                                               if row.get("equal") is not True],
                "corrected_worker_frames_comparisons": frame_checks,
                "non_frame_registered_comparisons_all_exact": non_frame_equal,
                "validated_debug_frame_references": validated_debug_count,
                "per_condition_frame_prefix_count": 200,
                "path_field_removed_after_validation": "debug_neural_state.path",
                "raw_gate_summary_matches_saved_audit": raw_seed_matches_saved,
            }
        except Exception as exc:
            results[str(seed)] = {
                "status": "scientific_prefix_unresolved",
                "raw_saved_status": saved_raw.get("status"),
                "raw_saved_audit_sha256": sha256_file(audit_path),
                "raw_recomputation_status": (raw_recomputed or {}).get("status"),
                "raw_recomputation_matches_saved_audit": False,
                "error": f"{type(exc).__name__}: {exc}",
                "path_field_removed_after_validation": "debug_neural_state.path",
                "raw_gate_summary_matches_saved_audit": False,
            }
    passed = all(row.get("status") == "scientific_prefix_exact_after_validated_debug_path_exclusion"
                 for row in results.values())
    return {
        "schema_version": "canonical-fullfield-corrected-scientific-prefix-v1",
        "status": ("scientific_preonset_prefix_exact_after_validated_debug_path_exclusion"
                   if passed else "scientific_preonset_prefix_unresolved"),
        "passed": passed,
        "raw_registered_gate_status": raw_gate["status"],
        "raw_registered_gate_is_preserved_and_not_overridden": True,
        "normalization_scope": "Only the debug_neural_state.path field is removed after validating the exact own-case canonical path, file hash/size, embedded metadata_json and every NPZ member's shape/dtype/content hash.",
        "by_seed": results,
        "frozen_auditor": {
            "path": str(FROZEN_PREFIX_AUDITOR),
            "sha256": sha256_file(FROZEN_PREFIX_AUDITOR),
            "raw_audit_triplet_recomputed_after_terminal": True,
        },
        "scientific_boundary": "This is a corrected common-input integrity scope only. It does not rewrite the raw audit status or establish a behavioral, causal, or biological response.",
    }


def _yaw_windows(run_dir: Path, run_record: dict[str, Any],
                 case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    trajectory = (run_dir / "trajectory.npz").resolve(strict=True)
    with np.load(trajectory, allow_pickle=False) as archive:
        require("time_s" in archive.files and "thorax_heading_yaw_rad" in archive.files,
                "trajectory NPZ lacks the native time/yaw arrays")
        times = np.asarray(archive["time_s"])
        raw_yaw = np.asarray(archive["thorax_heading_yaw_rad"])
    require(times.dtype == np.dtype("float64") and raw_yaw.dtype == np.dtype("float64")
            and times.shape == (CASE_NATIVE_SAMPLES,) and raw_yaw.shape == (CASE_NATIVE_SAMPLES,),
            "trajectory time/yaw dtype or sample counts differ from the native trace contract")
    require(bool(np.all(np.isfinite(times))) and bool(np.all(np.isfinite(raw_yaw)))
            and bool(np.all(np.diff(times) > 0.0))
            and math.isclose(float(times[0]), 0.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(times[-1]), 13.0, rel_tol=0.0, abs_tol=1e-9),
            "trajectory time/yaw values are nonfinite or do not cover 0..13 s")
    yaw = np.unwrap(raw_yaw)
    raw_hash = hashlib.sha256(np.ascontiguousarray(raw_yaw).tobytes(order="C")).hexdigest()
    unwrapped_hash = hashlib.sha256(np.ascontiguousarray(yaw).tobytes(order="C")).hexdigest()
    summary = read_json(run_dir / "window-summary.json", "window summary")
    require(summary.get("status") == "complete"
            and summary.get("case_id") == case["case_id"]
            and summary.get("cohort_id") == case["cohort_id"]
            and summary.get("body_seed") == case["body_seed"]
            and summary.get("neural_seed") == case["neural_seed"]
            and summary.get("stimulus_direction") == case["direction"]
            and math.isclose(float(summary.get("speed_magnitude_deg_s", math.nan)),
                             float(case["speed_magnitude_deg_s"]), rel_tol=0.0, abs_tol=1e-12),
            f"case {case['case_id']} window summary identity differs from prepared condition")
    trace = summary.get("raw_trace", {})
    require(trace.get("samples") == CASE_NATIVE_SAMPLES
            and trace.get("thorax_heading_yaw_rad_sha256") == raw_hash
            and trace.get("thorax_heading_yaw_rad_unwrapped_sha256") == unwrapped_hash
            and trace.get("ball_omega_z_used") is False,
            "window-summary yaw dtype/hash provenance differs from the raw trajectory arrays")
    measured: dict[str, Any] = {}
    for label, (start, end) in WINDOWS.items():
        protocol_window = protocol["analysis_windows_s_half_open"].get(label)
        if label == "stimulus_1_13":
            require(protocol_window == [start, end], "registered 1..13 s yaw window changed")
        yaw_start = float(np.interp(start, times, yaw))
        yaw_end = float(np.interp(end, times, yaw))
        change_deg = math.degrees(yaw_end - yaw_start)
        duration = end - start
        measured[label] = {
            "absolute_window_s_half_open": [start, end],
            "yaw_start_rad": yaw_start,
            "yaw_end_rad": yaw_end,
            "yaw_change_deg": change_deg,
            "mean_native_yaw_rate_deg_s": change_deg / duration,
        }
    recorded_full = summary.get("stimulus_1_13", {})
    require(math.isclose(measured["stimulus_1_13"]["yaw_change_deg"],
                         float(recorded_full.get("thorax_heading_yaw_change_deg", math.nan)),
                         rel_tol=0.0, abs_tol=1e-8),
            "independent 1..13 s yaw calculation differs from the runtime summary")
    measured["raw_trace"] = {
        "samples": CASE_NATIVE_SAMPLES,
        "time_dtype": str(times.dtype),
        "yaw_dtype": str(raw_yaw.dtype),
        "yaw_sha256": raw_hash,
        "unwrapped_yaw_sha256": unwrapped_hash,
        "coordinate": "native thorax local +X heading in world XY; positive yaw is fly-left",
        "unwrap_method": "numpy.unwrap, matching the frozen body summary reader",
        "ball_omega_z_used": False,
    }
    return measured


def _load_body_trace_reader(contract: dict[str, Any]):
    expected_sha = contract["mechanics_analysis"]["reference_sources"]["body_trace_reader"]["sha256"]
    require(sha256_file(BODY_TRACE_READER) == expected_sha,
            "frozen body-trace reader changed after the analysis contract was prepared")
    spec = importlib.util.spec_from_file_location(
        "canonical_fullfield_frozen_body_trace_reader", BODY_TRACE_READER,
    )
    require(spec is not None and spec.loader is not None,
            "could not load the pinned body-trace mechanical reader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mechanics_for_case(reader: Any, run_dir: Path, case: dict[str, Any],
                        run_record: dict[str, Any], precontroller_identity: dict[str, Any],
                        yaw_windows: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    mechanics_contract = contract["mechanics_analysis"]
    start_s, end_s = (float(value) for value in mechanics_contract["window_s_half_open"])
    # Reuse the established frozen trace reader for support, posture, per-leg
    # swing-event, contact, authority, and yaw/quaternion measurements.
    frozen_metrics = reader.analyze(run_dir.resolve(), start_s, end_s)
    trace_path = run_dir / "trajectory.npz"
    order_path = run_dir / "array_order.json"
    order = read_json(order_path, f"body array order {case['case_id']}")
    body_names = order.get("body_names")
    require(isinstance(body_names, list) and body_names.count("c_thorax") == 1,
            f"case {case['case_id']} body order lacks a unique c_thorax")
    thorax_index = body_names.index("c_thorax")

    with np.load(trace_path, allow_pickle=False) as archive:
        required = (
            "time_s", "qpos", "qvel", "thorax_heading_yaw_rad", "body_xpos_mm", "body_xquat_wxyz",
            "body_mass_native", "foot_contact_fz_up_native",
            "thorax_collision_clearance_min_mm", "thorax_ground_contact",
            "nonleg_ground_contact",
        )
        require(all(key in archive.files for key in required),
                f"case {case['case_id']} trajectory is missing mechanical fields")
        times = np.asarray(archive["time_s"])
        raw_yaw = np.asarray(archive["thorax_heading_yaw_rad"])
        qpos_array = np.asarray(archive["qpos"])
        require(qpos_array.dtype == np.dtype("float64") and qpos_array.ndim == 2
                and qpos_array.shape[0] == CASE_NATIVE_SAMPLES
                and bool(np.all(np.isfinite(qpos_array))),
                f"case {case['case_id']} qpos trace dtype/shape differs from the native schema")
        qpos_first_sha = hashlib.sha256(
            np.ascontiguousarray(qpos_array[0]).tobytes(order="C")
        ).hexdigest()
        qpos_shape = list(qpos_array.shape)
        del qpos_array
        qvel_array = np.asarray(archive["qvel"])
        require(qvel_array.dtype == np.dtype("float64") and qvel_array.ndim == 2
                and qvel_array.shape[0] == CASE_NATIVE_SAMPLES
                and bool(np.all(np.isfinite(qvel_array))),
                f"case {case['case_id']} qvel trace dtype/shape differs from the native schema")
        qvel_first_sha = hashlib.sha256(
            np.ascontiguousarray(qvel_array[0]).tobytes(order="C")
        ).hexdigest()
        qvel_shape = list(qvel_array.shape)
        del qvel_array
        positions_all = np.asarray(archive["body_xpos_mm"])
        quaternions_all = np.asarray(archive["body_xquat_wxyz"])
        masses = np.asarray(archive["body_mass_native"])
        fz_up = np.asarray(archive["foot_contact_fz_up_native"])
        clearance_all = np.asarray(archive["thorax_collision_clearance_min_mm"])
        thorax_contact_all = np.asarray(archive["thorax_ground_contact"])
        nonleg_contact_all = np.asarray(archive["nonleg_ground_contact"])

    require(times.dtype == np.dtype("float64") and raw_yaw.dtype == np.dtype("float64")
            and positions_all.dtype == np.dtype("float32")
            and quaternions_all.dtype == np.dtype("float32")
            and fz_up.dtype == np.dtype("float32"),
            f"case {case['case_id']} mechanical trace dtype differs from source allocation")
    initial_state = run_record.get("initial_state", {})
    require(qpos_first_sha == initial_state.get("qpos_sha256")
            == precontroller_identity.get("qpos_sha256")
            and qvel_first_sha == initial_state.get("qvel_sha256")
            == precontroller_identity.get("qvel_sha256"),
            f"case {case['case_id']} trajectory initial qpos/qvel do not match saved constructor identity")
    require(times.shape == (CASE_NATIVE_SAMPLES,)
            and raw_yaw.shape == (CASE_NATIVE_SAMPLES,)
            and positions_all.shape == (CASE_NATIVE_SAMPLES, len(body_names), 3)
            and quaternions_all.shape == (CASE_NATIVE_SAMPLES, len(body_names), 4)
            and masses.shape == (len(body_names),)
            and fz_up.shape == (CASE_NATIVE_SAMPLES, 6)
            and clearance_all.shape == (CASE_NATIVE_SAMPLES,)
            and thorax_contact_all.shape == (CASE_NATIVE_SAMPLES,)
            and nonleg_contact_all.shape == (CASE_NATIVE_SAMPLES,),
            f"case {case['case_id']} mechanical trace shape differs from the native 13 s schema")
    require(bool(np.all(np.isfinite(times))) and bool(np.all(np.diff(times) > 0.0))
            and bool(np.all(np.isfinite(raw_yaw)))
            and bool(np.all(np.isfinite(positions_all[:, thorax_index, :])))
            and bool(np.all(np.isfinite(quaternions_all[:, thorax_index, :])))
            and bool(np.all(np.isfinite(masses))) and bool(np.all(np.isfinite(fz_up)))
            and bool(np.all(np.isfinite(clearance_all))),
            f"case {case['case_id']} mechanical arrays contain nonfinite values")
    require(math.isclose(float(times[0]), 0.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(times[-1]), 13.0, rel_tol=0.0, abs_tol=1e-9),
            f"case {case['case_id']} native mechanical trace does not cover 0..13 s")
    mask = (times >= start_s - 1e-8) & (times <= end_s + 1e-8)
    require(int(mask.sum()) == 120001,
            f"case {case['case_id']} does not contain the exact native 1..13 s sample window")

    positions = positions_all[:, thorax_index, :].astype(np.float64)
    yaw = np.unwrap(raw_yaw)
    start_position = np.asarray([np.interp(start_s, times, positions[:, axis]) for axis in range(3)])
    end_position = np.asarray([np.interp(end_s, times, positions[:, axis]) for axis in range(3)])
    initial_heading = float(np.interp(start_s, times, yaw))
    forward_axis = np.asarray([math.cos(initial_heading), math.sin(initial_heading)])
    left_axis = np.asarray([-math.sin(initial_heading), math.cos(initial_heading)])
    displacement_xy = end_position[:2] - start_position[:2]
    in_window_xy = positions[mask, :2]
    path_points = np.concatenate((start_position[None, :2], in_window_xy, end_position[None, :2]), axis=0)
    path_distance = float(np.linalg.norm(np.diff(path_points, axis=0), axis=1).sum())

    q = quaternions_all[mask, thorax_index, :].astype(np.float64)
    q_norm = np.linalg.norm(q, axis=1)
    require(bool(np.all(q_norm > 0.0)), f"case {case['case_id']} has a zero-norm thorax quaternion")
    q /= q_norm[:, None]
    qw, qx, qy, qz = q.T
    up_vector_world = np.column_stack((
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ))
    up_z = np.clip(up_vector_world[:, 2], -1.0, 1.0)
    max_tilt_deg = float(np.degrees(np.arccos(up_z)).max())

    body_weight_native = float(masses.sum()) * 9810.0
    loaded_threshold_native = 0.01 * body_weight_native
    loaded = fz_up[mask] >= loaded_threshold_native
    support_count = loaded.sum(axis=1)
    per_leg_fraction = {
        leg: float(loaded[:, index].mean())
        for index, leg in enumerate(reader.LEGS)
    }
    min_loaded = int(support_count.min())
    fraction_ge_two = float((support_count >= 2).mean())
    require(min_loaded == int(frozen_metrics["support"]["minimum_loaded_legs"])
            and math.isclose(fraction_ge_two,
                             float(frozen_metrics["support"]["fraction_at_least_two"]),
                             rel_tol=0.0, abs_tol=1e-15),
            f"case {case['case_id']} frozen reader support results differ from independent extraction")
    thorax_z_min = float(positions[mask, 2].min())
    collision_clearance_min = float(clearance_all[mask].min())
    thorax_contacts = int(np.count_nonzero(thorax_contact_all[mask]))
    nonleg_contacts = int(np.count_nonzero(nonleg_contact_all[mask]))
    require(thorax_contacts == int(frozen_metrics["posture"]["thorax_contact_samples"])
            and nonleg_contacts == int(frozen_metrics["posture"]["nonleg_contact_samples"])
            and math.isclose(thorax_z_min, float(frozen_metrics["posture"]["thorax_z_min_mm"]),
                             rel_tol=0.0, abs_tol=1e-7)
            and math.isclose(collision_clearance_min,
                             float(frozen_metrics["posture"]["collision_clearance_min_mm"]),
                             rel_tol=0.0, abs_tol=1e-7),
            f"case {case['case_id']} frozen reader contact results differ from captured fields")

    measured_yaw = math.degrees(float(np.interp(end_s, times, yaw) - np.interp(start_s, times, yaw)))
    require(math.isclose(measured_yaw,
                         float(yaw_windows["stimulus_1_13"]["yaw_change_deg"]),
                         rel_tol=0.0, abs_tol=1e-8),
            f"case {case['case_id']} mechanics-reader and independent yaw window differ")

    contacts_path = run_dir / "contacts.npz"
    with np.load(contacts_path, allow_pickle=False) as archive:
        contact_required = ("step_index", "time_s", "is_ground_contact", "fly_body_segment", "leg_index")
        require(all(key in archive.files for key in contact_required),
                f"case {case['case_id']} contact archive is missing native record fields")
        contact_steps = np.asarray(archive["step_index"])
        contact_times = np.asarray(archive["time_s"])
        is_ground = np.asarray(archive["is_ground_contact"]).astype(bool, copy=False)
        contact_segments = np.asarray(archive["fly_body_segment"])
        leg_indices = np.asarray(archive["leg_index"])
    record_count = len(contact_steps)
    require(contact_steps.shape == contact_times.shape == is_ground.shape
            == contact_segments.shape == leg_indices.shape
            and contact_steps.ndim == 1
            and np.issubdtype(contact_steps.dtype, np.integer)
            and np.issubdtype(contact_times.dtype, np.floating)
            and contact_segments.dtype.kind in ("U", "S")
            and contact_times.dtype == np.dtype("float64")
            and np.issubdtype(leg_indices.dtype, np.integer)
            and (record_count == 0 or (
                int(contact_steps.min()) >= 0
                and int(contact_steps.max()) < CASE_NATIVE_SAMPLES
                and bool(np.all(np.isfinite(contact_times)))
            )), f"case {case['case_id']} contact archive dtype/shape/index is invalid")
    if record_count:
        require(bool(np.allclose(contact_times, times[contact_steps.astype(np.int64)],
                                  rtol=0.0, atol=1e-12)),
                f"case {case['case_id']} contact event timestamps differ from indexed native samples")
    contact_window = (contact_times >= start_s - 1e-8) & (contact_times <= end_s + 1e-8)
    ground_window = contact_window & is_ground
    ground_leg_records: dict[str, int] = {}
    for leg_index, leg_name in enumerate(reader.LEGS):
        ground_leg_records[leg_name] = int(np.count_nonzero(ground_window & (leg_indices == leg_index)))
    ground_nonleg = ground_window & (leg_indices < 0)
    ground_thorax = ground_window & (contact_segments.astype(str) == "c_thorax")
    record_nonleg_steps = set(int(index) for index in contact_steps[ground_nonleg])
    record_thorax_steps = set(int(index) for index in contact_steps[ground_thorax])
    trace_nonleg_steps = set(int(index) for index in np.flatnonzero(nonleg_contact_all)
                              if start_s - 1e-8 <= times[index] <= end_s + 1e-8)
    trace_thorax_steps = set(int(index) for index in np.flatnonzero(thorax_contact_all)
                             if start_s - 1e-8 <= times[index] <= end_s + 1e-8)

    r7_envelope = mechanics_contract["prior_r7_engineering_reference_envelope"]["metric_ranges"]
    reference_max_roll = float(r7_envelope["max_abs_roll_deg"][1])
    reference_max_pitch = float(r7_envelope["max_abs_pitch_deg"][1])
    posture = frozen_metrics["posture"]
    complete_events_by_leg = {
        row["leg"]: int(row["complete_events"]) for row in frozen_metrics["per_leg_events"]
    }
    return {
        "case_id": case["case_id"],
        "body_seed": case["body_seed"],
        "window_s_half_open": [start_s, end_s],
        "frozen_reader_metrics": frozen_metrics,
        "geometry_metrics": {
            "qpos_shape": qpos_shape,
            "qvel_shape": qvel_shape,
            "qpos_dtype": "float64",
            "qvel_dtype": "float64",
            "initial_qpos_sha256": qpos_first_sha,
            "initial_qvel_sha256": qvel_first_sha,
            "forward_displacement_initial_heading_frame_mm": float(np.dot(displacement_xy, forward_axis)),
            "lateral_displacement_initial_heading_frame_mm": float(np.dot(displacement_xy, left_axis)),
            "world_xy_displacement_mm": [float(displacement_xy[0]), float(displacement_xy[1])],
            "horizontal_path_distance_mm": path_distance,
            "mean_forward_speed_from_endpoint_mm_s": float(np.dot(displacement_xy, forward_axis) / (end_s - start_s)),
            "initial_heading_at_window_start_rad": initial_heading,
            "thorax_z_min_mm": thorax_z_min,
            "thorax_collision_clearance_min_mm": collision_clearance_min,
            "thorax_up_vector_world_z_min": float(up_z.min()),
            "thorax_up_vector_world_z_max": float(up_z.max()),
            "max_tilt_from_world_vertical_deg": max_tilt_deg,
            "local_body_up_remained_in_world_up_hemisphere": bool(np.all(up_z > 0.0)),
            "quaternion_norm_min_before_normalization": float(q_norm.min()),
            "quaternion_norm_max_before_normalization": float(q_norm.max()),
            "body_xpos_mm_dtype": str(positions_all.dtype),
            "body_xquat_wxyz_dtype": str(quaternions_all.dtype),
            "foot_contact_fz_up_native_dtype": str(fz_up.dtype),
            "thorax_heading_yaw_rad_dtype": str(raw_yaw.dtype),
            "native_window_sample_count": int(mask.sum()),
        },
        "support_metrics": {
            "threshold_definition": "per-leg Fz-up >= 0.01 x body weight, from the frozen body-trace reader",
            "body_weight_native": body_weight_native,
            "per_leg_loaded_fraction": per_leg_fraction,
            "minimum_loaded_legs": min_loaded,
            "fraction_samples_with_at_least_two_loaded_legs": fraction_ge_two,
            "complete_tarsus_swing_events_by_leg": complete_events_by_leg,
            "complete_tarsus_swing_event_total": sum(complete_events_by_leg.values()),
            "event_criterion": frozen_metrics["event_definition"],
            "zero_event_interpretation": "No events passing the strict descriptive criterion were detected; zero events alone is not a failure or evidence of no leg cycling.",
        },
        "contact_metrics": {
            "thorax_ground_contact_samples": thorax_contacts,
            "nonleg_ground_contact_samples": nonleg_contacts,
            "contact_archive_ground_records": int(np.count_nonzero(ground_window)),
            "contact_archive_ground_records_by_leg": ground_leg_records,
            "contact_archive_nonleg_ground_records": int(np.count_nonzero(ground_nonleg)),
            "contact_archive_thorax_ground_records": int(np.count_nonzero(ground_thorax)),
            "contact_archive_nonleg_unique_steps_match_trace_flags": record_nonleg_steps == trace_nonleg_steps,
            "contact_archive_thorax_unique_steps_match_trace_flags": record_thorax_steps == trace_thorax_steps,
            "contact_record_source_sha256": sha256_file(contacts_path),
        },
        "heading_integrity": {
            "yaw_delta_deg_from_trajectory": measured_yaw,
            "quaternion_derived_yaw_max_abs_error_deg": float(posture["quaternion_heading_max_error_deg"]),
            "quaternion_heading_comparison": "diagnostic of stored thorax_heading_yaw_rad against c_thorax body_xquat_wxyz; not a fitted direction response",
        },
        "prior_r7_posture_envelope_comparison": {
            "max_abs_roll_deg": float(posture["max_abs_roll_deg"]),
            "max_abs_pitch_deg": float(posture["max_abs_pitch_deg"]),
            "prior_r7_max_abs_roll_deg": reference_max_roll,
            "prior_r7_max_abs_pitch_deg": reference_max_pitch,
            "within_prior_r7_roll_pitch_envelope": bool(
                float(posture["max_abs_roll_deg"]) <= reference_max_roll
                and float(posture["max_abs_pitch_deg"]) <= reference_max_pitch
            ),
            "interpretation": "Contextual comparison with the full accepted R7 15-case engineered envelope on 1.25..7 s; the R3 1..13 s interval and visual-input conditions differ, so this is not an acceptance gate.",
        },
        "reader_provenance": {
            "path": str(BODY_TRACE_READER),
            "sha256": sha256_file(BODY_TRACE_READER),
            "reader_source_files": frozen_metrics["source_files"],
        },
        "classification": "descriptive physical-body trace integrity; not a gait, visual-following, or biological pass",
    }


def _descriptive_directional_summary(case_outputs: dict[str, dict[str, Any]],
                                     cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(int(case["body_seed"]), str(case["direction"])): case for case in cases}
    outcomes: dict[str, Any] = {}
    for window in WINDOWS:
        outcomes[window] = {}
        for direction in ("positive60", "negative60"):
            sign = 1.0 if direction == "positive60" else -1.0
            per_seed = []
            for seed in (1729, 1730, 1731):
                static = by_key[(seed, "static_c1")]
                moving = by_key[(seed, direction)]
                static_delta = case_outputs[static["case_id"]][window]["yaw_change_deg"]
                moving_window = case_outputs[moving["case_id"]][window]
                delta = moving_window["yaw_change_deg"] - static_delta
                duration = WINDOWS[window][1] - WINDOWS[window][0]
                per_seed.append({
                    "body_seed": seed,
                    "direction": direction,
                    "stimulus_direction_deg_s": float(moving["direction_deg_s"]),
                    "static_yaw_change_deg": static_delta,
                    "moving_yaw_change_deg": moving_window["yaw_change_deg"],
                    "static_subtracted_native_yaw_change_deg": delta,
                    "static_subtracted_native_yaw_rate_deg_s": delta / duration,
                    "direction_label_aligned_arithmetic_deg": sign * delta,
                })
            values = [row["static_subtracted_native_yaw_change_deg"] for row in per_seed]
            outcomes[window][direction] = {
                "per_seed": per_seed,
                "matched_seed_distribution": {
                    "n": len(values),
                    "mean_deg": statistics.fmean(values),
                    "median_deg_descriptive_only": statistics.median(values),
                    "min_deg": min(values),
                    "max_deg": max(values),
                },
            }
    return {
        "windows_s_half_open": {key: list(value) for key, value in WINDOWS.items()},
        "native_yaw_sign": "positive is fly-left",
        "static_subtraction": "moving yaw change minus matched same-seed static_c1 yaw change",
        "direction_label_aligned_metric": "multiplies the measured static-subtracted yaw by the sign of direction_deg_s; arithmetic label only, not a claim that stimulus sign equals anatomical fly-left",
        "by_window_and_direction": outcomes,
        "interpretation_boundary": "three matched seeds are a descriptive distribution only; no statistical/biological success test is performed",
    }


def analyze() -> dict[str, Any]:
    require(not OUTPUT_JSON.exists() and not OUTPUT_MD.exists(),
            "final analysis output already exists; refusing to overwrite")
    contract = read_json(ANALYSIS_CONTRACT_PATH, "offline analysis contract")
    manifest, protocol, freeze = _verify_prepared_pins(contract)
    start, review, terminal = _verify_root_authority(contract, manifest, protocol)
    queue, cohort_summary = _verify_queue_completion(protocol)
    cases = list(protocol["stimuli"]["conditions"])
    require(len(cases) == 9 and [row["case_id"] for row in cases] == protocol["queue_order"],
            "protocol case rows do not match the registered nine-case queue")

    # First verify all nine records and every artifact hash. No NPZ arrays are
    # decoded until the entire cohort has cleared these terminal/integrity gates.
    verified_cases: dict[str, dict[str, Any]] = {}
    case_gate_records: dict[str, Any] = {}
    for case in cases:
        verified = _verify_case_record(case, protocol, freeze)
        case_id = case["case_id"]
        verified_cases[case_id] = verified
        case_gate_records[case_id] = {
            "status": verified["record"]["status"],
            "counts": {
                "neural_frames": verified["record"]["completed_neural_frames"],
                "neural_ticks": verified["record"]["runtime_contract"]["neural_ticks"],
                "body_ticks": verified["record"]["completed_body_ticks"],
                "physics_steps": verified["record"]["physics_step_count"],
                "native_samples": verified["record"]["native_sample_count"],
            },
            "mjb_sha256": verified["precontroller_identity"]["compiled_model_mjb_sha256"],
            "mjb_bytes": verified["precontroller_identity"]["compiled_model_mjb_size_bytes"],
            "cgroup_memory_peak_bytes": verified["record"]["resource_peak"]["memory_peak_bytes"],
            "cgroup_memory_max_bytes": verified["record"]["resource_peak"]["memory_max_bytes"],
            "integrity_artifact_count_verified": verified["verified_artifact_count"],
        }

    triplet = _triplet_gate(protocol, queue)
    corrected_triplet = _corrected_scientific_prefix_gate(protocol, triplet)
    case_outputs: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case["case_id"]
        verified = verified_cases[case_id]
        case_outputs[case_id] = _yaw_windows(
            verified["run_dir"], verified["record"], case, protocol,
        )

    # The mechanical reader is imported only after the full-cohort, terminal,
    # source/model/resource, integrity, and prefix-audit gates pass.
    body_reader = _load_body_trace_reader(contract)
    mechanics_outputs = {
            case["case_id"]: _mechanics_for_case(
                body_reader,
                verified_cases[case["case_id"]]["run_dir"],
                case,
                verified_cases[case["case_id"]]["record"],
                verified_cases[case["case_id"]]["precontroller_identity"],
                case_outputs[case["case_id"]],
                contract,
        )
        for case in cases
    }
    unresolved = not triplet["passed"]
    analysis_status = (
        "complete_descriptive_preonset_triplets_exact"
        if not unresolved else "complete_descriptive_preonset_gate_unresolved"
    )
    result = {
        "schema_version": "canonical-fullfield-optomotor-r3-final-offline-analysis-v2",
        "status": analysis_status,
        "classification": "multi-seed engineering visual-motor replication; no biological pass",
        "execution_binding": {
            "unit": terminal["unit"],
            "invocation_id": terminal["invocation_id"],
            "exec_session_id": terminal["exec_session_id"],
            "actual_exit_code": terminal["actual_exit_code"],
            "actual_runtime_s": terminal["actual_runtime_s"],
            "terminal_receipt_sha256": sha256_file(TERMINAL_RECEIPT_PATH),
            "root_review_sha256": sha256_file(ROOT_REVIEW_PATH),
            "runner_sha256": sha256_file(RUNNER),
            "preparer_sha256": sha256_file(PREPARER),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "prepare_manifest_sha256": sha256_file(MANIFEST_PATH),
            "source_freeze_sha256": sha256_file(FREEZE_PATH),
        },
        "completion_gates": {
            "queue_status": queue["status"],
            "completed_cases": queue["completed"],
            "case_count": cohort_summary["case_count"],
            "all_case_counts_and_cgroup_peaks_valid": True,
            "all_case_output_hashes_validated": True,
            "all_native_mjb_and_runtime_identity_gates_passed": True,
            "all_nine_body_trace_mechanics_summaries_present": len(mechanics_outputs) == 9,
            "body_mechanics_acceptance_gate": False,
            "strict_same_seed_preonset_triplet_gate": triplet,
            "corrected_scientific_prefix_gate": corrected_triplet,
        },
        "case_records": case_gate_records,
        "yaw_windows_by_case": case_outputs,
        "mechanical_integrity": {
            "status": "descriptive_all_nine_cases",
            "window_s_half_open": contract["mechanics_analysis"]["window_s_half_open"],
            "by_case": mechanics_outputs,
            "prior_r7_engineering_reference_envelope": contract["mechanics_analysis"]["prior_r7_engineering_reference_envelope"],
            "prior_ang4_scope": contract["mechanics_analysis"]["ang4_scope_context"],
            "acceptance_gate": False,
            "interpretation": "These measurements check actual free-body orientation, support, movement, contact and heading representation. They do not establish biological gait or visual following. Zero strict tarsus swing events alone is not a failure.",
        },
        "directional_outcomes": _descriptive_directional_summary(case_outputs, cases),
        "claims": {
            "biological_success_claim": False,
            "statistical_or_biological_validation": False,
            "directional_sign_is_not_interpreted_as_biological_success": True,
            "median_is_descriptive_and_never_overrides_the_strict_triplet_gate": True,
            "historical_v2_prefix_is_secondary_descriptive_only": True,
            "raw_preonset_audit_status_is_preserved": True,
            "corrected_scientific_prefix_does_not_override_raw_gate": True,
            "condition_label_selects_retinal_pixels_only": True,
            "mechanics_are_descriptive_integrity_evidence_only": True,
            "biological_gait_success_claim": False,
            "mechanical_metrics_are_not_pass_fail_criteria": True,
            "zero_strict_tarsus_swing_events_is_not_a_failure_by_itself": True,
        },
        "analysis_contract_sha256": sha256_file(ANALYSIS_CONTRACT_PATH),
        "analysis_source_sha256": sha256_file(ANALYZER),
    }
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    markdown = _render_markdown(result)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
    with OUTPUT_MD.open("x", encoding="utf-8") as stream:
        stream.write(markdown)
        stream.flush()
    return {
        "status": result["status"],
        "output_json": str(OUTPUT_JSON),
        "output_markdown": str(OUTPUT_MD),
        "preonset_triplet_gate": triplet["status"],
        "corrected_scientific_prefix_gate": corrected_triplet["status"],
        "biological_success_claim": False,
    }


def _render_markdown(result: dict[str, Any]) -> str:
    rows = [
        "# Canonical full-field optomotor R3 offline analysis",
        "",
        f"Status: `{result['status']}`.",
        "",
        "This is a descriptive engineering analysis. It makes no biological success or statistical validation claim.",
        "",
        "## Completion and integrity gates",
        "",
        f"- Queue: `{result['completion_gates']['queue_status']}`; {result['completion_gates']['case_count']} cases.",
        f"- Strict same-seed pre-onset gate: `{result['completion_gates']['strict_same_seed_preonset_triplet_gate']['status']}`.",
        f"- Corrected scientific-prefix scope: `{result['completion_gates']['corrected_scientific_prefix_gate']['status']}`; it removes only validated case-local debug artifact paths and leaves the raw audit status unchanged.",
        "- Artifact hashes, constructor-only MJB identity, case counts, and live cgroup memory peaks were checked before outcome analysis.",
        "- A prefix mismatch remains unresolved and is never overridden by a median response.",
        "",
        "## Raw and corrected triplet evidence",
        "",
        "The persisted raw audit is reported unchanged. The corrected scope independently reruns its hash-pinned frozen comparator, validates each case-local debug snapshot reference and NPZ contents, then removes only the verified absolute `debug_neural_state.path` field for exact scientific JSONL comparison. A corrected exact scope does not relabel or override a raw mismatch.",
        "",
        "| Seed | Raw audit | Corrected scientific prefix | Frame rows per condition | Validated debug references |",
        "|---:|---|---|---:|---:|",
    ]
    raw_triplet = result["completion_gates"]["strict_same_seed_preonset_triplet_gate"]
    corrected_triplet = result["completion_gates"]["corrected_scientific_prefix_gate"]
    for seed in sorted(corrected_triplet["by_seed"], key=int):
        raw_seed = raw_triplet["by_seed"][seed]
        corrected_seed = corrected_triplet["by_seed"][seed]
        rows.append(
            f"| {seed} | `{raw_seed['status']}` | `{corrected_seed['status']}` | "
            f"{corrected_seed.get('per_condition_frame_prefix_count', 0)} | "
            f"{corrected_seed.get('validated_debug_frame_references', 0)} |"
        )
    rows.extend([
        "",
        "## Body mechanics and heading integrity (1–13 s)",
        "",
        "The frozen body-trace reader reports native support, posture, contact, per-leg swing-event, authority, and quaternion-derived heading diagnostics. The extra geometry fields use thorax position and quaternion arrays from the same trace. Contact sample flags are also compared with indexed records in contacts.npz. These are descriptive checks, not biological gait or visual-following criteria.",
        "",
        "The pinned complete R7 15-case engineering cohort supplies a non-gating reference envelope for a different 1.25–7 s, nonvisual fixed-P/d setup. The older ANG4 mechanics capture is partial. Neither source defines a biological acceptance threshold for this R3 visual cohort.",
        "",
        "| Case | Forward (mm) | XY path (mm) | Max abs roll (deg) | Max abs pitch (deg) | Max tilt (deg) | Thorax Z min (mm) | Clearance min (mm) | Min up-vector Z | Quaternion yaw error max (deg) | Min loaded legs | Fraction ≥2 loaded | Thorax/nonleg contact samples | Strict tarsus events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case_id, data in result["mechanical_integrity"]["by_case"].items():
        geometry = data["geometry_metrics"]
        support = data["support_metrics"]
        contacts = data["contact_metrics"]
        posture = data["frozen_reader_metrics"]["posture"]
        rows.append(
            f"| `{case_id}` | {geometry['forward_displacement_initial_heading_frame_mm']:.6g} | "
            f"{geometry['horizontal_path_distance_mm']:.6g} | {posture['max_abs_roll_deg']:.6g} | "
            f"{posture['max_abs_pitch_deg']:.6g} | {geometry['max_tilt_from_world_vertical_deg']:.6g} | "
            f"{geometry['thorax_z_min_mm']:.6g} | {geometry['thorax_collision_clearance_min_mm']:.6g} | "
            f"{geometry['thorax_up_vector_world_z_min']:.6g} | "
            f"{data['heading_integrity']['quaternion_derived_yaw_max_abs_error_deg']:.6g} | "
            f"{support['minimum_loaded_legs']} | "
            f"{support['fraction_samples_with_at_least_two_loaded_legs']:.6g} | "
            f"{contacts['thorax_ground_contact_samples']}/{contacts['nonleg_ground_contact_samples']} | "
            f"{support['complete_tarsus_swing_event_total']} |"
        )
    rows.extend([
        "",
        "Support uses the frozen reader’s per-leg Fz-up ≥1% body-weight rule. A reported count of zero events means no swing passed its ≥0.05 mm tarsus-rise and ≥2 ms swing/loaded-flank criteria; it does not establish that the leg did not cycle and is not a failure by itself.",
        "The R7 envelope is shown only as prior engineering context. R3 uses a longer window, visual input, and different conditions; falling outside the R7 range is not an automatic failure.",
        "",
    ])
    reference = result["mechanical_integrity"]["prior_r7_engineering_reference_envelope"]
    r7_ranges = reference["metric_ranges"]
    ang4 = result["mechanical_integrity"]["prior_ang4_scope"]["partial_mechanics_1_13_reference"]
    ang4_ranges = ang4["metric_ranges"]
    rows.extend([
        "Prior non-gating reference ranges:",
        "",
        f"- R7 complete 15-case cohort, window `{reference['window_s']}` s: forward displacement "
        f"{r7_ranges['forward_displacement_mm'][0]:.4g}–{r7_ranges['forward_displacement_mm'][1]:.4g} mm; "
        f"mean contact-found legs {r7_ranges['mean_contact_legs'][0]:.4g}–{r7_ranges['mean_contact_legs'][1]:.4g}; "
        f"max abs roll {r7_ranges['max_abs_roll_deg'][0]:.4g}–{r7_ranges['max_abs_roll_deg'][1]:.4g}°; "
        f"max abs pitch {r7_ranges['max_abs_pitch_deg'][0]:.4g}–{r7_ranges['max_abs_pitch_deg'][1]:.4g}°.",
        f"- ANG4 partial {len(ang4['cases'])}-case mechanics capture, window `{ang4['window_s']}` s: "
        f"yaw change {ang4_ranges['native_yaw_delta_deg'][0]:.4g}–{ang4_ranges['native_yaw_delta_deg'][1]:.4g}°; "
        f"mean contact-found legs {ang4_ranges['mean_contact_legs_found'][0]:.4g}–{ang4_ranges['mean_contact_legs_found'][1]:.4g}; "
        f"thorax/nonleg contacts {ang4_ranges['thorax_ground_contact_samples'][0]:g}/"
        f"{ang4_ranges['nonleg_ground_contact_samples'][0]:g} samples; whole-capture thorax Z min "
        f"{ang4_ranges['whole_capture_thorax_z_min_mm'][0]:.4g}–"
        f"{ang4_ranges['whole_capture_thorax_z_min_mm'][1]:.4g} mm and clearance min "
        f"{ang4_ranges['whole_capture_thorax_collision_clearance_min_mm'][0]:.4g}–"
        f"{ang4_ranges['whole_capture_thorax_collision_clearance_min_mm'][1]:.4g} mm.",
        "",
        "The R7 and ANG4 support bases use contact-found/positive-Fz occupancy, while the R3 frozen reader uses Fz-up ≥1% body weight. These reference ranges are not thresholds.",
        "",
    ])
    rows.extend([
        "## Matched-seed yaw outcomes",
        "",
        "Native yaw is positive fly-left. Each moving response is shown after subtracting the same-seed static case. The direction-label-aligned column is arithmetic only; it does not assert that a stimulus label is an anatomical heading direction.",
        "",
        "| Window (s) | Stimulus label | Matched-seed static-subtracted yaw change (deg) | Mean | Median (descriptive) | Range |",
        "|---|---|---|---:|---:|---:|",
    ])
    for window, window_data in result["directional_outcomes"]["by_window_and_direction"].items():
        for direction, record in window_data.items():
            distribution = record["matched_seed_distribution"]
            values = ", ".join(
                f"seed {row['body_seed']}: {row['static_subtracted_native_yaw_change_deg']:.6g}"
                for row in record["per_seed"]
            )
            low, high = distribution["min_deg"], distribution["max_deg"]
            rows.append(
                f"| `{window}` | `{direction}` | {values} | {distribution['mean_deg']:.6g} | "
                f"{distribution['median_deg_descriptive_only']:.6g} | [{low:.6g}, {high:.6g}] |"
            )
    rows.extend(["", "The median summarizes three matched seeds only. It is not an acceptance threshold or evidence of biological validation.", ""])
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true", required=True,
                        help="analyze only after the R3 queue and terminal receipt pass all gates")
    args = parser.parse_args()
    if not args.analyze:
        parser.error("only --analyze is supported")
    try:
        result = analyze()
    except AnalysisError as exc:
        print(json.dumps({"status": "analysis_blocked", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
