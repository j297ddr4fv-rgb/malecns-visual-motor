#!/usr/bin/env python3
"""Prepare the frozen revision-6 per-case-process FlyGym bundle.

This standard-library-only preparer inherits the exact revision-5 native-model
gate, protocol, trace format, and source/assets. It freezes the actual revision-5
OOM terminal records and configures one sequential child process per case so
simulation/contact memory is released at each process exit. It does not import
FlyGym or MuJoCo, compile a model, run a simulation, or create an approval.

Prepare with:

    python scripts/prepare_flygym_steering_dose_calibration_v6.py --prepare
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
BASE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1"
)
OUTPUT_ROOT = BASE / "mechanical-steering-dose-calibration-v1/revision-6"
REVISION_5_ROOT = BASE / "mechanical-steering-dose-calibration-v1/revision-5"
RUNNER_PATH = PROJECT / "scripts/run_flygym_steering_dose_calibration_v6.py"
SESSION_SOURCE_PATH = PROJECT / "scripts/run_flygym_reference_walk.py"
CONTROLLER_SOURCE_PATH = PROJECT / "scripts/flygym_phase_speed_gait.py"

EXPECTED_NATIVE_MJB_SHA256 = "6d7b1f8b9a7558cd5367be9e12e1a866adca3d98e6312d431b62d27ad747b2f3"
EXPECTED_R5_TERMINAL_SHA256 = "5a9707eb986f941478fac953cd3621498da47102521c6272050657165de072b7"
EXPECTED_R5_COHORT_SHA256 = "320e3e1ee4267bd9aab6adda1ace8e4cfdca00eff6e7ce89a89cc516c0752214"
EXPECTED_R5_FIRST_RECORD_SHA256 = "630db3a303f7f8a5a1aa45a317437b47fe1d9dcadbe9036fef19dff12e580e7d"
EXPECTED_R5_PARTIAL_RECORD_SHA256 = "afac9c7d9fcd78265a0a9eda05a41db7f66414eeb6539153be33dd2d3151fe5c"
EXPECTED_R5_LIVE_SHA256 = "135404509ab277a0ee435a826ae5914a4324e02174eee452abd790288b406228"
EXPECTED_R5_REVIEW_SHA256 = "ec880a170241dfce1eef73da89c19c3de5dcb73130671435e48b25335987f6c8"


class PreparationError(RuntimeError):
    """Raised when a parent freeze, OOM record, or source pin changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreparationError(f"expected a JSON object: {path}")
    return value


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def pin_file(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreparationError(f"required pinned file is missing: {path}")
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size, "role": role}


def verify_file(path: Path, pin: dict[str, Any], description: str) -> None:
    if not path.is_file():
        raise PreparationError(f"missing {description}: {path}")
    if path.stat().st_size != pin.get("size_bytes") or sha256_file(path) != pin.get("sha256"):
        raise PreparationError(f"changed {description}: {path}")


def copy_verified(source: Path, destination: Path, expected_sha: str) -> dict[str, Any]:
    if not source.is_file():
        raise PreparationError(f"required input is missing: {source}")
    observed = sha256_file(source)
    if observed != expected_sha:
        raise PreparationError(f"input hash differs from the parent freeze: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = sha256_file(destination)
    if copied != observed:
        raise PreparationError(f"copy verification failed: {destination}")
    return {"sha256": copied, "size_bytes": destination.stat().st_size}


def validate_r5_outcome() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence_paths = {
        "root_terminal": REVISION_5_ROOT / "root-terminal-observation.json",
        "root_live": REVISION_5_ROOT / "root-live-observation.json",
        "root_execution_review": REVISION_5_ROOT / "root-execution-review.json",
        "cohort_manifest": REVISION_5_ROOT / "runs/cohort-manifest.json",
        "completed_case_run_record": REVISION_5_ROOT / "runs/seed1729_dzero_0p00/run-record.json",
        "partial_case_run_record": REVISION_5_ROOT / "runs/seed1729_dplus_0p20/run-record.json",
    }
    expected_hashes = {
        "root_terminal": EXPECTED_R5_TERMINAL_SHA256,
        "root_live": EXPECTED_R5_LIVE_SHA256,
        "root_execution_review": EXPECTED_R5_REVIEW_SHA256,
        "cohort_manifest": EXPECTED_R5_COHORT_SHA256,
        "completed_case_run_record": EXPECTED_R5_FIRST_RECORD_SHA256,
        "partial_case_run_record": EXPECTED_R5_PARTIAL_RECORD_SHA256,
    }
    pins = {
        name: pin_file(path, role=f"immutable revision-5 OOM evidence: {name}")
        for name, path in evidence_paths.items()
    }
    for name, expected in expected_hashes.items():
        if pins[name]["sha256"] != expected:
            raise PreparationError(f"revision-5 evidence hash changed: {name}")
    terminal = read_json(evidence_paths["root_terminal"])
    live = read_json(evidence_paths["root_live"])
    execution_review = read_json(evidence_paths["root_execution_review"])
    cohort = read_json(evidence_paths["cohort_manifest"])
    completed_case = read_json(evidence_paths["completed_case_run_record"])
    partial_case = read_json(evidence_paths["partial_case_run_record"])
    if (terminal.get("actual_exec_exit_code") != 1
            or terminal.get("systemd_result") != "oom-kill"
            or terminal.get("process_termination") != "code=killed/status=KILL"
            or terminal.get("actual_terminal_runtime_s") != 340.719
            or terminal.get("partial_case_last_logged_control_tick") != 6500
            or terminal.get("cohort_manifest_status_stale_after_unhandled_sigkill") != "running"
            or terminal.get("cohort_manifest_sha256") != EXPECTED_R5_COHORT_SHA256):
        raise PreparationError("revision-5 terminal record is not the pinned OOM-kill outcome")
    if (cohort.get("status") != "running"
            or cohort.get("completed_case_count") != 1
            or completed_case.get("status") != "complete"
            or partial_case.get("status") != "running"):
        raise PreparationError("revision-5 partial outputs no longer match the preserved terminal record")
    if live.get("terminal_claim") is not False or live.get("actual_process_live") is not True:
        raise PreparationError("revision-5 live observation is not the pre-terminal observation")
    if (execution_review.get("execution_authority") != "authorized_after_review"
            or execution_review.get("schema_version") != "mechanical-steering-dose-root-review-v5"):
        raise PreparationError("revision-5 execution review is not the archived reviewed attempt")
    return terminal, live, execution_review, cohort, completed_case, partial_case


def protocol_for_revision6(r5_protocol: dict[str, Any], runner_sha: str,
                           r5_freeze_sha: str, r5_protocol_sha: str,
                           evidence_pins: dict[str, Any]) -> dict[str, Any]:
    protocol = json.loads(json.dumps(r5_protocol))
    protocol["schema_version"] = "flygym-steering-dose-calibration-protocol-v6"
    protocol["revision"] = 6
    protocol["entrypoint_source_sha256"] = runner_sha
    protocol["preparation_timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    protocol["source_parent"] = {
        "revision_5_bundle": str(REVISION_5_ROOT),
        "revision_5_source_freeze_sha256": r5_freeze_sha,
        "revision_5_protocol_sha256": r5_protocol_sha,
        "revision_5_terminal_record_path": str(REVISION_5_ROOT / "root-terminal-observation.json"),
        "revision_5_terminal_record_sha256": evidence_pins["root_terminal"]["sha256"],
        "revision_5_cohort_manifest_sha256": evidence_pins["cohort_manifest"]["sha256"],
        "revision_5_partial_case_record_sha256": evidence_pins["partial_case_run_record"]["sha256"],
    }
    protocol["execution"]["entrypoint"] = str(RUNNER_PATH.resolve())
    protocol["execution"]["invocation"] = (
        "/data/environments/projects/flygym-reference/bin/python "
        "scripts/run_flygym_steering_dose_calibration_v6.py --execute"
    )
    protocol["execution"]["case_worker_policy"] = {
        "mode": "one_sequential_subprocess_per_case",
        "max_concurrent_case_workers": 1,
        "wait_for_worker_exit_before_next_case": True,
        "workers_inherit_service_cgroup_limits": True,
        "cpu_limit_cores": protocol["execution"]["cpu_limit_cores"],
        "memory_limit_bytes": protocol["execution"]["memory_limit_bytes"],
        "case_timeout_s": protocol["execution"]["case_timeout_s"],
        "batch_timeout_s": protocol["execution"]["batch_timeout_s"],
        "session_controller_cycle_broken_before_worker_exit": True,
        "trace_and_contact_capture_format_unchanged": True,
        "controller_and_physics_design_unchanged": True,
        "coordinator_retained_data": "small JSON run records, same-seed pairing fields, and analysis summaries only",
        "trace_saving": "unchanged per-case compressed trajectory/contact outputs; controller telemetry remains line-streamed",
    }
    protocol["revision_6_memory_policy"] = {
        "r5_root_cause": "session.controller retained the controller while controller.session retained the session; session.close only closes the renderer, leaving preallocated trace arrays and contact lists eligible only through a strong-reference cycle in the long-lived cohort process",
        "supporting_capture_sizes": {
            "r5_completed_case_trajectory_npz_bytes": evidence_pins["r5_trajectory_bytes"],
            "r5_completed_case_contacts_npz_bytes": evidence_pins["r5_contacts_bytes"],
        },
        "repair": "one child process per case plus explicit session-controller cycle break, renderer close, local reference release, and gc.collect before worker exit",
        "per_case_and_batch_timeouts_unchanged": True,
        "CPU2_Mem2G_unchanged": True,
        "full_trace_and_contact_format_unchanged": True,
    }
    protocol["revision_6_changes"] = [
        "Run each frozen case in one sequential short-lived Python worker; do not start the next case until the worker exits and its run record is validated.",
        "Break the session/controller reference cycle and collect before a successful worker exits; process exit remains the isolation boundary for all native and Python allocations.",
        "Keep the exact native MJB, options/order/qpos/qvel gates, 15 cases, same-seed prefix checks, P/d schedule, controller, trace/contact files, analysis, CPU2/Mem2G limits, and timeouts unchanged.",
        "Freeze the actual revision-5 OOM terminal, stale running cohort manifest, completed first-case record, and second-case partial run record without rewriting them as complete or failed.",
    ]
    return protocol


def prepare() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise PreparationError(f"refusing to overwrite existing revision-6 bundle: {OUTPUT_ROOT}")
    r5_freeze_path = REVISION_5_ROOT / "source_freeze.json"
    r5_protocol_path = REVISION_5_ROOT / "protocol.json"
    r5_review_path = REVISION_5_ROOT / "preparation-review.json"
    r5_freeze = read_json(r5_freeze_path)
    r5_protocol = read_json(r5_protocol_path)
    r5_review = read_json(r5_review_path)
    if r5_freeze.get("revision") != 5 or r5_protocol.get("revision") != 5:
        raise PreparationError("revision-5 parent bundle has an unexpected revision number")
    if r5_freeze.get("protocol_sha256") != sha256_file(r5_protocol_path):
        raise PreparationError("revision-5 protocol no longer matches its source freeze")
    if r5_review.get("status") != "prepared_not_executed":
        raise PreparationError("revision-5 source review is not the prepared parent record")
    if r5_review.get("runner_sha256") != r5_freeze.get("runner_sha256"):
        raise PreparationError("revision-5 review does not identify its frozen runner")

    # Verify all inherited sources, assets and snapshots without loading physics.
    for raw_path, pin in r5_freeze.get("source_files", {}).items():
        verify_file(Path(raw_path), pin, "revision-5 inherited source")
    assets = r5_freeze.get("asset_manifest", {})
    if assets.get("asset_count") != 79 or assets.get("verified_asset_count") != 79:
        raise PreparationError("revision-5 inherited upstream asset count is not 79")
    for logical_name, pin in assets.get("files", {}).items():
        verify_file(Path(pin["physical_path"]), pin, f"upstream asset {logical_name}")
    for filename, pin in r5_freeze.get("copied_inputs", {}).items():
        verify_file(Path(pin["snapshot_path"]), pin, f"revision-5 copied input {filename}")
    for filename, pin in r5_freeze.get("bundle_files", {}).items():
        verify_file(REVISION_5_ROOT / filename, pin, f"revision-5 bundle file {filename}")

    terminal, live, r5_execution_review, cohort, completed_case, partial_case = validate_r5_outcome()
    session_text = SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    controller_text = CONTROLLER_SOURCE_PATH.read_text(encoding="utf-8")
    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    if ("self.controller = HybridTurningController(" not in session_text
            or "self.session = session" not in controller_text
            or "def close(self)" not in session_text
            or "self.sim.renderer.close()" not in session_text
            or "self.all_arrays = self._allocate_state_arrays(n_native + 1)" not in session_text
            or "self.contact_position.append(" not in session_text
            or "position_mm=np.asarray(self.contact_position" not in session_text):
        raise PreparationError("static source review no longer shows the session/controller and trace/contact memory ownership")
    close_start = session_text.index("    def close(self) -> None:")
    close_end = session_text.index("    def __enter__", close_start)
    close_body = session_text[close_start:close_end]
    if "self.all_arrays" in close_body or "contact_position" in close_body:
        raise PreparationError("reference close implementation unexpectedly frees the trace/contact buffers")
    if ("session.controller = None" not in runner_text
            or "gc.collect()" not in runner_text
            or "one_sequential_subprocess_per_case" not in runner_text
            or "--case-worker-input" not in runner_text
            or "subprocess.Popen(command" not in runner_text):
        raise PreparationError("revision-6 runner does not include isolated per-case workers and cycle cleanup")

    r5_source_files: dict[str, Any] = {}
    r5_evidence_paths = {
        "root_terminal": REVISION_5_ROOT / "root-terminal-observation.json",
        "root_live": REVISION_5_ROOT / "root-live-observation.json",
        "root_execution_review": REVISION_5_ROOT / "root-execution-review.json",
        "cohort_manifest": REVISION_5_ROOT / "runs/cohort-manifest.json",
        "completed_case_run_record": REVISION_5_ROOT / "runs/seed1729_dzero_0p00/run-record.json",
        "partial_case_run_record": REVISION_5_ROOT / "runs/seed1729_dplus_0p20/run-record.json",
    }
    evidence_pins = {
        name: pin_file(path, role=f"immutable revision-5 OOM evidence: {name}")
        for name, path in r5_evidence_paths.items()
    }
    # Preserve parent outputs as references in review metadata, without copying
    # their 446 MB trajectory or changing the in-flight manifest/partial record.
    first_run_dir = REVISION_5_ROOT / "runs/seed1729_dzero_0p00"
    evidence_pins["r5_trajectory_bytes"] = (first_run_dir / "trajectory.npz").stat().st_size
    evidence_pins["r5_contacts_bytes"] = (first_run_dir / "contacts.npz").stat().st_size
    for name, path in r5_evidence_paths.items():
        r5_source_files[str(path.resolve())] = evidence_pins[name]

    output_parent = OUTPUT_ROOT.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mechanical-dose-rev6-", dir=output_parent))
    try:
        copied_inputs: dict[str, Any] = {}
        for filename, pin in r5_freeze.get("copied_inputs", {}).items():
            source = Path(pin["snapshot_path"])
            destination = stage / "frozen_inputs" / filename
            record = copy_verified(source, destination, pin["sha256"])
            copied_inputs[filename] = {
                **record,
                "source_path": str(source),
                "snapshot_path": str(OUTPUT_ROOT / "frozen_inputs" / filename),
                "role": pin.get("role", "inherited frozen input"),
                "verified_unchanged_from_revision_5": True,
            }

        evidence_destinations = {
            "root_terminal": "frozen_inputs/historical_revision_5/root-terminal-observation.json",
            "root_live": "frozen_inputs/historical_revision_5/root-live-observation.json",
            "root_execution_review": "frozen_inputs/historical_revision_5/root-execution-review.json",
            "cohort_manifest": "frozen_inputs/historical_revision_5/cohort-manifest.json",
            "completed_case_run_record": "frozen_inputs/historical_revision_5/seed1729_dzero_0p00-run-record.json",
            "partial_case_run_record": "frozen_inputs/historical_revision_5/seed1729_dplus_0p20-partial-run-record.json",
        }
        for name, relative in evidence_destinations.items():
            source = r5_evidence_paths[name]
            record = copy_verified(source, stage / relative, evidence_pins[name]["sha256"])
            copied_inputs[f"historical_revision_5/{name}"] = {
                **record,
                "source_path": str(source),
                "snapshot_path": str(OUTPUT_ROOT / relative),
                "role": f"immutable revision-5 OOM evidence: {name}",
            }

        bundle_files: dict[str, Any] = {}

        def copy_bundle_file(source: Path, relative: str, expected_sha: str,
                             role: str) -> None:
            record = copy_verified(source, stage / relative, expected_sha)
            bundle_files[relative] = {**record, "role": role}

        for filename, pin in r5_freeze.get("bundle_files", {}).items():
            record = copy_verified(REVISION_5_ROOT / filename, stage / filename, pin["sha256"])
            bundle_files[filename] = {
                **record,
                "role": "inherited revision-5 immutable bundle file",
            }
        prep_path = Path(__file__).resolve(strict=True)
        copy_bundle_file(prep_path, f"source_snapshots/revision-6/{prep_path.name}",
                         sha256_file(prep_path), "revision-6 standard-library preparation script")
        copy_bundle_file(RUNNER_PATH, f"source_snapshots/revision-6/{RUNNER_PATH.name}",
                         sha256_file(RUNNER_PATH), "revision-6 frozen per-case-worker runner")
        for filename, value in r5_source_files.items():
            r5_source_files[filename] = {
                "sha256": value["sha256"],
                "size_bytes": value["size_bytes"],
                "role": value["role"],
            }

        source_files: dict[str, Any] = {
            path: {key: value for key, value in pin.items() if key in ("sha256", "size_bytes", "role")}
            for path, pin in r5_freeze.get("source_files", {}).items()
        }
        source_files.update(r5_source_files)
        source_files[str(prep_path.resolve())] = pin_file(prep_path, role="revision-6 standard-library preparation script")
        source_files[str(RUNNER_PATH.resolve())] = pin_file(RUNNER_PATH, role="revision-6 per-case-worker execution entrypoint")

        protocol = protocol_for_revision6(
            r5_protocol,
            source_files[str(RUNNER_PATH.resolve())]["sha256"],
            sha256_file(r5_freeze_path),
            sha256_file(r5_protocol_path),
            evidence_pins,
        )
        if protocol["design"] != r5_protocol["design"] or protocol["execution"]["ordered_cases"] != r5_protocol["execution"]["ordered_cases"]:
            raise PreparationError("revision-6 case design differs from revision 5")
        for key in ("duration_s", "power_P_left_right", "common_prefix_s", "ramp_duration_s",
                    "case_timeout_s", "batch_timeout_s", "cpu_limit_cores", "memory_limit_bytes",
                    "d_schedule", "analysis_window_s"):
            if protocol["execution"].get(key) != r5_protocol["execution"].get(key):
                raise PreparationError(f"revision-6 execution setting changed: {key}")
        dump_json(stage / "protocol.json", protocol)

        asset_manifest = json.loads(json.dumps(assets))
        asset_manifest["inherited_from_revision_5_source_freeze_sha256"] = sha256_file(r5_freeze_path)
        native_reference = json.loads(json.dumps(r5_freeze.get("native_model_reference")))
        native_reference["actual_session_mjb_snapshot_path"] = str(
            OUTPUT_ROOT / native_reference["evidence"]["mjb_snapshot"]
        )
        evidence_snapshot_paths = {
            key: value for key, value in evidence_destinations.items()
        }
        source_freeze = {
            "schema_version": "flygym-steering-dose-calibration-source-freeze-v6",
            "revision": 6,
            "status": "preparation_frozen_execution_entrypoint_ready_not_run",
            "prior_revision_5_source_freeze": {
                "path": str(r5_freeze_path),
                "sha256": sha256_file(r5_freeze_path),
                "all_source_files_reverified": True,
                "all_assets_reverified": True,
                "source_file_count_inherited": len(r5_freeze["source_files"]),
                "asset_count_inherited": assets["asset_count"],
                "verified_asset_count_inherited": assets["verified_asset_count"],
            },
            "source_files": source_files,
            "asset_manifest": asset_manifest,
            "copied_inputs": copied_inputs,
            "bundle_files": bundle_files,
            "native_model_reference": native_reference,
            "historical_revision_5": {
                "root_cause": "strong session/controller reference cycle retained preallocated traces and per-contact lists after renderer-only close in the long-lived r5 process",
                "root_terminal_path": str(r5_evidence_paths["root_terminal"]),
                "root_terminal_sha256": evidence_pins["root_terminal"]["sha256"],
                "cohort_manifest_sha256": evidence_pins["cohort_manifest"]["sha256"],
                "cohort_manifest_status_at_failure": cohort.get("status"),
                "completed_case_count_at_failure": cohort.get("completed_case_count"),
                "completed_case_id": "seed1729_dzero_0p00",
                "completed_case_status": completed_case.get("status"),
                "partial_case_id": "seed1729_dplus_0p20",
                "partial_case_status_at_failure": partial_case.get("status"),
                "partial_case_last_logged_control_tick": terminal.get("partial_case_last_logged_control_tick"),
                "actual_exec_exit_code": terminal.get("actual_exec_exit_code"),
                "systemd_result": terminal.get("systemd_result"),
                "actual_terminal_runtime_s": terminal.get("actual_terminal_runtime_s"),
                "memory_peak_bytes": terminal.get("systemd_reported_memory_peak_bytes"),
                "partial_manifest_and_run_record_not_rewritten": True,
                "execution_review_is_historical_only": True,
                "evidence_snapshots": evidence_snapshot_paths,
                "evidence_pins": evidence_pins,
                "completed_case_output_bytes": {
                    "trajectory_npz": evidence_pins["r5_trajectory_bytes"],
                    "contacts_npz": evidence_pins["r5_contacts_bytes"],
                },
            },
            "memory_isolation_policy": protocol["execution"]["case_worker_policy"],
            "runner_sha256": source_files[str(RUNNER_PATH.resolve())]["sha256"],
            "preparation_script_sha256": source_files[str(prep_path.resolve())]["sha256"],
            "protocol_sha256": sha256_file(stage / "protocol.json"),
            "raw_asset_binaries_copied": False,
            "future_execution_must_reverify_all_source_asset_input_and_reference_hashes": True,
            "no_execution_approval_created": True,
            "revision_5_files_modified": False,
        }
        dump_json(stage / "source_freeze.json", source_freeze)
        freeze_sha = sha256_file(stage / "source_freeze.json")

        review = {
            "schema_version": "flygym-steering-dose-calibration-preparation-review-v6",
            "status": "prepared_not_executed",
            "conclusion": "Revision 6 is prepared for independent review. Each of the unchanged 15 cases runs in one sequential child process under the existing service CPU2/Mem2G limits; the child exits before the next case. The native model gate and full trace/contact/analysis protocol remain unchanged. No simulation, model compilation, test, NN, GPU, or execution approval was performed.",
            "entrypoint_path": str(RUNNER_PATH.resolve()),
            "runner_sha256": source_freeze["runner_sha256"],
            "preparation_script_sha256": source_freeze["preparation_script_sha256"],
            "source_freeze_sha256": freeze_sha,
            "protocol_sha256": source_freeze["protocol_sha256"],
            "case_count": len(protocol["execution"]["ordered_cases"]),
            "design_unchanged_from_revision_5": protocol["design"] == r5_protocol["design"],
            "native_model_gate_unchanged": protocol["execution"]["model_identity_gate"] == r5_protocol["execution"]["model_identity_gate"],
            "resource_limits_unchanged": {
                "cpu_limit_cores": protocol["execution"]["cpu_limit_cores"],
                "memory_limit_bytes": protocol["execution"]["memory_limit_bytes"],
                "case_timeout_s": protocol["execution"]["case_timeout_s"],
                "batch_timeout_s": protocol["execution"]["batch_timeout_s"],
            },
            "source_and_asset_inheritance": {
                "revision_5_source_file_count": len(r5_freeze["source_files"]),
                "revision_5_asset_count": assets["asset_count"],
                "verified_asset_count": assets["verified_asset_count"],
                "all_revision_5_sources_reverified": True,
                "all_79_assets_reverified": True,
            },
            "revision_5_oom_evidence": source_freeze["historical_revision_5"],
            "memory_root_cause": {
                "session_controller_cycle": "runner assigned session.controller = controller; controller base retained self.session = session",
                "close_behavior": "FlyGymReferenceSession.close closes only the renderer and does not clear all_arrays or contact lists",
                "large_per_case_state": "70,001 native-sample arrays are preallocated for the 7-second case; contact events are retained in Python lists and converted for contacts.npz",
                "observed_failure": "r5 completed one case, then systemd OOM-killed the long-lived cohort during the second case at 6,500/7,000 ticks with the 2-GiB memory cap",
                "fix": "case-level process exit releases native/Python allocations; explicit cycle break and gc.collect add cleanup within each child",
            },
            "case_worker_policy": protocol["execution"]["case_worker_policy"],
            "trace_contact_and_analysis_unchanged": True,
            "scope_boundary": "Preparation only; no revision-6 run output or approval artifact was created.",
        }
        dump_json(stage / "preparation-review.json", review)
        review_md = [
            "# Mechanical steering dose calibration — revision 6",
            "",
            "**Status: prepared, not executed.** The r5 bundle, OOM terminal, one completed case, and partial second-case files remain unchanged. This bundle has no revision-6 execution approval or run output.",
            "",
            "## Frozen r5 failure evidence",
            "",
            f"The pinned r5 root terminal reports systemd `oom-kill`, exit code `{terminal['actual_exec_exit_code']}`, service runtime `{terminal['actual_terminal_runtime_s']}` seconds, and a 2-GiB memory peak. It retains one completed case (`seed1729_dzero_0p00`) and the second case (`seed1729_dplus_0p20`) at `{terminal['partial_case_last_logged_control_tick']}/7000` control ticks. The old cohort manifest remains in its observed `running` state after SIGKILL and is copied verbatim as historical evidence; it is not rewritten as complete or failed.",
            "",
            "## Memory root cause and repair",
            "",
            "The frozen runner assigned `session.controller = controller`; the controller base retained `self.session = session`. This made a strong reference cycle. `FlyGymReferenceSession.close()` closes the renderer but leaves its 70,001-sample trace arrays and per-contact Python lists attached. Each case's trace is large (the completed r5 trajectory archive is 445,882,536 bytes), and `session.save()` materializes contact-list arrays to write the compressed contact archive. In the long-lived r5 process, a later case could allocate against retained memory from the previous session.",
            "",
            "Revision 6 keeps the same model/data/trace and runs exactly one case per child process. The coordinator waits for the child to exit before launching the next case; the worker inherits the service's CPU2/Mem2G cgroup. Its `finally` block closes the renderer, removes the session/controller edge, releases the local references, and calls `gc.collect()`. The process boundary releases native and Python allocations even if an internal cycle remains.",
            "",
            "The parent process stores only case JSON records and same-seed pairing metadata. The existing line-streamed controller telemetry and per-case `np.savez_compressed` trajectory/contact formats remain unchanged. The strict native MJB SHA, options/array-order/qpos/qvel gates, 15 cases, schedules, controller, analysis, CPU2/Mem2G limits, and 600/9000-second timeouts are unchanged.",
            "",
            "## Frozen model identity and source pins",
            "",
            f"The strict full native `session.model` MJB pin remains `{EXPECTED_NATIVE_MJB_SHA256}`. All revision-5 sources and all 79 assets are reverified and inherited. The new worker records include their PIDs and terminal codes so each case's process boundary can be reviewed.",
            "",
            "No body simulation, model compilation, test, NN, GPU, or 15-case run was performed in preparing this bundle.",
            "",
        ]
        (stage / "preparation-review.md").write_text("\n".join(review_md), encoding="utf-8")
        stage.rename(OUTPUT_ROOT)
        return review
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True,
                        help="verify the r5 outcome and prepare the revision-6 bundle")
    parser.parse_args()
    try:
        review = prepare()
    except PreparationError as exc:
        print(f"preparation failed: {exc}")
        return 2
    print(json.dumps({
        "status": review["status"],
        "bundle_root": str(OUTPUT_ROOT),
        "runner_sha256": review["runner_sha256"],
        "source_freeze_sha256": review["source_freeze_sha256"],
        "protocol_sha256": review["protocol_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
