#!/usr/bin/env python3
"""Prepare a reviewed-only revision-7 mechanical dose bundle.

Revision 7 keeps the exact revision-6 scientific protocol and per-case worker
memory boundary. It binds each hidden worker to the live coordinator process,
shared queue flock, active hash-pinned cohort prefix, and actual systemd service
with OOMPolicy=continue. This preparer is standard-library-only: it does not
import FlyGym/MuJoCo/NumPy, compile a model, run a body, start NN/GPU work, or
create execution approval.

Prepare with:
    python scripts/prepare_flygym_steering_dose_calibration_v7.py --prepare
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
BASE = Path("/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1")
OUTPUT_ROOT = BASE / "mechanical-steering-dose-calibration-v1/revision-7"
REVISION_6_ROOT = BASE / "mechanical-steering-dose-calibration-v1/revision-6"
REVISION_6_REVIEW_JSON = BASE / "analysis/mechanical-dose-revision-6-independent-review-v1.json"
REVISION_6_REVIEW_MD = BASE / "analysis/mechanical-dose-revision-6-independent-review-v1.md"
RUNNER_PATH = PROJECT / "scripts/run_flygym_steering_dose_calibration_v7.py"
SERVICE_UNIT_NAME = "flygm-mechanical-steering-dose-20261008-v7.service"
PYTHON_PATH = "/data/environments/projects/flygym-reference/bin/python"
CASE_TIMEOUT_S = 600
BATCH_TIMEOUT_S = 9000
SERVICE_RUNTIME_LIMIT_S = 9600
CPU_LIMIT_CORES = 2.0
MEMORY_LIMIT_BYTES = 2147483648
R5_OOM_TERMINAL_SHA256 = "5a9707eb986f941478fac953cd3621498da47102521c6272050657165de072b7"
R6_RUNNER_SHA256 = "cf03fb97d106dcb30c876444b35b2883ed468fd35f49f930bc9cce4194cd20d3"
R6_PREPARER_SHA256 = "99ea7671630a1d941c87e26bcfb1bbb146c4cbb2e871990101c4aeedc2a5d653"
R6_PROTOCOL_SHA256 = "0ef8bed18ddba53126bc5443622974d8ba6ef3f465b7afc94c8d7c7665fdcd2b"
R6_SOURCE_FREEZE_SHA256 = "a0fcbdaeafbbfecd52099984224ea2dc619f8d842589008f0d900a391989a10e"
R6_PREPARATION_REVIEW_SHA256 = "51c65fcfd126faee1e886605162767e7cf24f7816e7b5e019797ba89a06023fb"
R6_REVIEW_JSON_SHA256 = "2dd36ac5bf91ffed0d519298b2e699c7b855dc85000e07f02a356bf80e1f6653"
R6_REVIEW_MD_SHA256 = "676511ae7482fcd8c9dc67a341b442963c565c8e9ec5d97d386520ace5038402"


class PreparationError(RuntimeError):
    """Raised when an inherited r6 pin or source invariant changed."""


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


def dump_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def pin_file(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreparationError(f"required file is missing: {path}")
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size, "role": role}


def verify_pin(path: Path, pin: dict[str, Any], label: str) -> None:
    if not path.is_file() or path.stat().st_size != pin.get("size_bytes") or sha256_file(path) != pin.get("sha256"):
        raise PreparationError(f"frozen {label} changed or is missing: {path}")


def copy_verified(source: Path, destination: Path, expected_sha: str) -> dict[str, Any]:
    if not source.is_file() or sha256_file(source) != expected_sha:
        raise PreparationError(f"source differs from its inherited pin: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    observed = sha256_file(destination)
    if observed != expected_sha:
        raise PreparationError(f"copy verification failed: {destination}")
    return {"sha256": observed, "size_bytes": destination.stat().st_size}


def _r6_bundle_pins() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_path = REVISION_6_ROOT / "protocol.json"
    freeze_path = REVISION_6_ROOT / "source_freeze.json"
    preparation_path = REVISION_6_ROOT / "preparation-review.json"
    protocol = read_json(protocol_path)
    freeze = read_json(freeze_path)
    prep_review = read_json(preparation_path)
    independent_review_json = read_json(REVISION_6_REVIEW_JSON)
    independent_review_md = REVISION_6_REVIEW_MD.read_text(encoding="utf-8")

    expected_files = {
        "runner": (REVISION_6_ROOT / "source_snapshots/revision-6/run_flygym_steering_dose_calibration_v6.py", R6_RUNNER_SHA256),
        "preparer": (REVISION_6_ROOT / "source_snapshots/revision-6/prepare_flygym_steering_dose_calibration_v6.py", R6_PREPARER_SHA256),
        "protocol": (protocol_path, R6_PROTOCOL_SHA256),
        "source_freeze": (freeze_path, R6_SOURCE_FREEZE_SHA256),
        "preparation_review": (preparation_path, R6_PREPARATION_REVIEW_SHA256),
        "independent_review_json": (REVISION_6_REVIEW_JSON, R6_REVIEW_JSON_SHA256),
        "independent_review_md": (REVISION_6_REVIEW_MD, R6_REVIEW_MD_SHA256),
    }
    for name, (path, expected) in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise PreparationError(f"revision-6 frozen {name} pin changed: {path}")
    if (protocol.get("revision") != 6 or freeze.get("revision") != 6
            or freeze.get("status") != "preparation_frozen_execution_entrypoint_ready_not_run"
            or prep_review.get("status") != "prepared_not_executed"
            or independent_review_json.get("review_outcome") != "block_launch_until_oom_policy_and_worker_entrypoint_authorization_are_resolved"):
        raise PreparationError("revision-6 candidate or independent review has an unexpected status")
    if freeze.get("protocol_sha256") != R6_PROTOCOL_SHA256:
        raise PreparationError("revision-6 source freeze does not bind the frozen protocol")
    if independent_review_json.get("pins", {}).get("runner_sha256") != R6_RUNNER_SHA256:
        raise PreparationError("revision-6 independent report does not identify the reviewed runner")
    if independent_review_json.get("pins", {}).get("source_freeze_sha256") != R6_SOURCE_FREEZE_SHA256:
        raise PreparationError("revision-6 independent report does not identify the reviewed freeze")
    if (REVISION_6_ROOT / "runs").exists() or (REVISION_6_ROOT / "root-execution-review.json").exists():
        raise PreparationError("revision-6 bundle unexpectedly contains run or execution-approval artifacts")

    execution = protocol.get("execution", {})
    if (len(execution.get("ordered_cases", [])) != 15
            or execution.get("duration_s") != 7.0
            or execution.get("power_P_left_right") != [0.258, 0.258]
            or execution.get("cpu_limit_cores") != CPU_LIMIT_CORES
            or execution.get("memory_limit_bytes") != MEMORY_LIMIT_BYTES
            or execution.get("case_timeout_s") != CASE_TIMEOUT_S
            or execution.get("batch_timeout_s") != BATCH_TIMEOUT_S
            or execution.get("model_identity_gate", {}).get("expected_sha256") != "6d7b1f8b9a7558cd5367be9e12e1a866adca3d98e6312d431b62d27ad747b2f3"):
        raise PreparationError("revision-6 scientific or resource protocol does not match its pinned design")
    old_history = freeze.get("historical_revision_5", {})
    if (old_history.get("root_terminal_sha256") != R5_OOM_TERMINAL_SHA256
            or old_history.get("systemd_result") != "oom-kill"
            or old_history.get("partial_case_last_logged_control_tick") != 6500):
        raise PreparationError("revision-6 does not preserve the original r5 OOM terminal")

    # Verify all inherited references before making any r7 copy.
    for path_text, pin in freeze.get("source_files", {}).items():
        verify_pin(Path(path_text), pin, "r6 inherited source")
    for logical_name, pin in freeze.get("asset_manifest", {}).get("files", {}).items():
        verify_pin(Path(pin["physical_path"]), pin, f"upstream asset {logical_name}")
    for logical_name, pin in freeze.get("copied_inputs", {}).items():
        verify_pin(Path(pin["snapshot_path"]), pin, f"r6 copied input {logical_name}")
    for relative, pin in freeze.get("bundle_files", {}).items():
        verify_pin(REVISION_6_ROOT / relative, pin, f"r6 bundle file {relative}")
    return protocol, freeze, prep_review, independent_review_json


def _systemd_launch() -> tuple[list[str], str]:
    runner = str(RUNNER_PATH.resolve())
    argv = [
        "systemd-run", "--user", "--wait", "--pipe",
        f"--unit={SERVICE_UNIT_NAME}",
        "--property=CPUQuota=200%",
        f"--property=MemoryMax={MEMORY_LIMIT_BYTES}",
        f"--property=RuntimeMaxSec={SERVICE_RUNTIME_LIMIT_S}",
        "--property=OOMPolicy=continue",
        "--", PYTHON_PATH, runner, "--execute",
    ]
    command = " ".join(argv)
    return argv, command


def _relative_copy_inputs(r6_freeze: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for logical_name, old_pin in r6_freeze.get("copied_inputs", {}).items():
        pin = dict(old_pin)
        original = Path(old_pin["snapshot_path"])
        relative = original.relative_to(REVISION_6_ROOT)
        pin["snapshot_path"] = str(OUTPUT_ROOT / relative)
        pin["inherited_from_revision_6_path"] = old_pin["snapshot_path"]
        output[logical_name] = pin
    return output


def prepare() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise PreparationError(f"refusing to overwrite the existing revision-7 bundle: {OUTPUT_ROOT}")
    if not RUNNER_PATH.is_file():
        raise PreparationError(f"revision-7 runner is missing: {RUNNER_PATH}")
    protocol_r6, freeze_r6, prep_review_r6, independent_review = _r6_bundle_pins()
    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    required_markers = (
        "def _systemd_service_attestation(",
        'properties["OOMPolicy"] != "continue"',
        "def _verify_worker_parent_and_queue_lock(",
        "def _load_cohort_worker_contract(",
        "active_cohort_manifest_sha256",
        "expected_pair_run_record_sha256",
        "WORKER_BATCH_RESERVE_S = 120",
        "pass_fds=(queue_lock_handle.fileno(),)",
        "memory_current_bytes",
    )
    if any(marker not in runner_text for marker in required_markers):
        raise PreparationError("revision-7 runner is missing a required OOM, parent, cohort, queue, or timeout guard")

    protocol = json.loads(json.dumps(protocol_r6))
    runner_sha = sha256_file(RUNNER_PATH)
    prep_path = Path(__file__).resolve(strict=True)
    prep_sha = sha256_file(prep_path)
    protocol["schema_version"] = "flygym-steering-dose-calibration-protocol-v7"
    protocol["revision"] = 7
    protocol["entrypoint_source_sha256"] = runner_sha
    protocol["preparation_timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    protocol["source_parent"]["revision_6_bundle"] = str(REVISION_6_ROOT)
    protocol["source_parent"]["revision_6_source_freeze_sha256"] = R6_SOURCE_FREEZE_SHA256
    protocol["source_parent"]["revision_6_independent_review_json_sha256"] = R6_REVIEW_JSON_SHA256
    protocol["source_parent"]["revision_6_independent_review_md_sha256"] = R6_REVIEW_MD_SHA256
    protocol["execution"]["entrypoint"] = str(RUNNER_PATH.resolve())
    launch_argv, invocation = _systemd_launch()
    protocol["execution"]["invocation"] = invocation
    protocol["execution"]["systemd_service"] = {
        "unit_name": SERVICE_UNIT_NAME,
        "manager": "systemd --user transient service",
        "launch_argv": launch_argv,
        "required_properties_at_runtime": {
            "OOMPolicy": "continue",
            "MemoryMax": str(MEMORY_LIMIT_BYTES),
            "CPUQuota": "200%",
            "RuntimeMaxSec": SERVICE_RUNTIME_LIMIT_S,
        },
        "runtime_readback_properties": [
            "OOMPolicy", "MemoryMax", "CPUQuotaPerSecUSec", "RuntimeMaxUSec", "ControlGroup", "InvocationID",
        ],
        "required_runtime_checks": [
            "actual OOMPolicy equals continue",
            "actual MemoryMax equals 2147483648",
            "INVOCATION_ID matches the active unit InvocationID",
            "unit ControlGroup equals the process unified cgroup path",
            "effective cgroup cap is at most 2 CPU cores and exactly 2147483648 bytes",
        ],
        "cohort_and_worker_store_actual_properties": True,
    }
    protocol["execution"]["revision_7_worker_entry_contract"] = {
        "hidden_worker_entry_is_accepted_only_from_active_execute_coordinator": True,
        "verify_parent_pid_starttime_argv_and_same_cgroup": True,
        "inherit_and_verify_coordinator_shared_queue_flock_fd": True,
        "require_active_hash_pinned_cohort_manifest_and_exact_case_index": True,
        "derive_pair_record_and_constant_P_from_completed_prefix_hashes": True,
        "case_timeout_max_s": CASE_TIMEOUT_S,
        "batch_timeout_max_s": BATCH_TIMEOUT_S,
        "worker_startup_and_termination_reserve_s": 120,
        "required_nonzero_case_pair": "same-seed completed d=0 run-record ID and SHA-256 verified from active cohort prefix",
        "required_P_trace_after_first_case": "exact cohort constant_P_trace_sha256 and every completed prefix record",
    }
    protocol["execution"]["systemd_terminal_fallback"] = {
        "coordinator_or_unit_death_may_prevent_python_manifest_finalization": True,
        "root_must_read_actual_unit_result_exec_exit_status_and_journal_before_interpreting_stale_manifest": True,
        "required_terminal_fields": [
            "systemd unit InvocationID, Result, ExecMainCode, ExecMainStatus, MemoryPeak, OOMPolicy",
            "root observation time and terminal service/cgroup evidence source hashes",
            "cohort manifest status/hash and every extant worker terminal/run-record hash",
        ],
        "stale_running_manifest_is_not_complete": True,
        "fallback_artifact_created_by_preparer": False,
    }
    protocol["revision_7_changes"] = [
        "Retain the exact r6 native MJB, 15-case seed/d order, P/d/gait/controller, initial model/state gates, trace/contact formats, and CPU2/Mem2G/600/9000 limits.",
        "Pin a fixed systemd-run --user invocation with OOMPolicy=continue and read back actual unit properties in coordinator, workers, cohort and case terminal records.",
        "Keep one sequential child per case; require live parent PID/starttime/argv, same cgroup, the inherited shared fixed-pose queue flock, and a hash-pinned active cohort manifest.",
        "Derive same-seed d=0 pairing and cohort-wide P checks from completed run-record hashes in the cohort prefix; reject absent or stale pair/P fields and timeouts above their hard caps.",
        "If the coordinator or service dies before writing a final manifest, preserve actual systemd terminal evidence externally; never call a stale running manifest complete.",
    ]

    output_parent = OUTPUT_ROOT.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mechanical-dose-rev7-", dir=output_parent))
    try:
        # Preserve a byte-for-byte r6 bundle snapshot under the new revision,
        # along with the direct frozen inputs the unchanged science runner reads.
        shutil.copytree(REVISION_6_ROOT / "frozen_inputs", stage / "frozen_inputs")
        r6_snapshot = stage / "frozen_inputs/revision_6_bundle"
        shutil.copytree(REVISION_6_ROOT, r6_snapshot)
        r6_snapshot_files = {}
        for path in sorted(p for p in r6_snapshot.rglob("*") if p.is_file()):
            relative = str(path.relative_to(stage))
            original = REVISION_6_ROOT / path.relative_to(r6_snapshot)
            if sha256_file(path) != sha256_file(original):
                raise PreparationError(f"revision-6 snapshot copy failed: {relative}")
            r6_snapshot_files[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "role": "byte-preserved revision-6 candidate bundle snapshot",
            }
        copied_screen = copy_verified(
            REVISION_6_ROOT / "paired-screen-reference.json",
            stage / "paired-screen-reference.json",
            sha256_file(REVISION_6_ROOT / "paired-screen-reference.json"),
        )
        review_copy_paths = {
            "frozen_inputs/revision_6_independent_review/mechanical-dose-revision-6-independent-review-v1.json": REVISION_6_REVIEW_JSON,
            "frozen_inputs/revision_6_independent_review/mechanical-dose-revision-6-independent-review-v1.md": REVISION_6_REVIEW_MD,
        }
        copied_review = {}
        for relative, source in review_copy_paths.items():
            expected = R6_REVIEW_JSON_SHA256 if relative.endswith(".json") else R6_REVIEW_MD_SHA256
            copied_review[relative] = copy_verified(source, stage / relative, expected)

        snapshot_dir = stage / "source_snapshots/revision-7"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prep_path, snapshot_dir / prep_path.name)
        shutil.copyfile(RUNNER_PATH, snapshot_dir / RUNNER_PATH.name)
        for name in (prep_path.name, RUNNER_PATH.name):
            if sha256_file(snapshot_dir / name) != sha256_file(PROJECT / "scripts" / name):
                raise PreparationError(f"revision-7 source snapshot changed while copied: {name}")

        # The top-level copied_inputs point at r7's byte-verified copy; the
        # full r6 bundle remains separately pinned as historical review input.
        copied_inputs = _relative_copy_inputs(freeze_r6)
        for logical_name, pin in copied_inputs.items():
            relative = Path(pin["snapshot_path"]).relative_to(OUTPUT_ROOT)
            copied_path = stage / relative
            if not copied_path.is_file() or sha256_file(copied_path) != pin["sha256"]:
                raise PreparationError(f"revision-7 copied input differs from r6: {logical_name}")
        copied_inputs.update({
            "revision_6_independent_review_json": {
                **copied_review["frozen_inputs/revision_6_independent_review/mechanical-dose-revision-6-independent-review-v1.json"],
                "snapshot_path": str(OUTPUT_ROOT / "frozen_inputs/revision_6_independent_review/mechanical-dose-revision-6-independent-review-v1.json"),
                "role": "immutable independent revision-6 blocker report",
            },
            "revision_6_independent_review_md": {
                **copied_review["frozen_inputs/revision_6_independent_review/mechanical-dose-revision-6-independent-review-v1.md"],
                "snapshot_path": str(OUTPUT_ROOT / "frozen_inputs/revision_6_independent_review/mechanical-dose-revision-6-independent-review-v1.md"),
                "role": "immutable independent revision-6 blocker report text",
            },
        })

        bundle_files: dict[str, Any] = dict(r6_snapshot_files)
        bundle_files["paired-screen-reference.json"] = {
            **copied_screen,
            "role": "inherited revision-6 paired-screen reference used by the unchanged runner",
        }
        for relative, info in copied_review.items():
            bundle_files[relative] = {**info, "role": "copied independent revision-6 review report"}
        bundle_files[f"source_snapshots/revision-7/{prep_path.name}"] = {
            **pin_file(snapshot_dir / prep_path.name, "revision-7 standard-library preparer source snapshot")
        }
        bundle_files[f"source_snapshots/revision-7/{RUNNER_PATH.name}"] = {
            **pin_file(snapshot_dir / RUNNER_PATH.name, "revision-7 frozen runner source snapshot")
        }

        source_files = dict(freeze_r6.get("source_files", {}))
        source_files[str(prep_path.resolve())] = pin_file(prep_path, "revision-7 standard-library preparation script")
        source_files[str(RUNNER_PATH.resolve())] = pin_file(RUNNER_PATH, "revision-7 execution entrypoint")
        native_reference = json.loads(json.dumps(freeze_r6["native_model_reference"]))
        native_reference["actual_session_mjb_snapshot_path"] = str(OUTPUT_ROOT / "frozen_inputs/native_model_reference/actual_session_model.mjb")
        freeze = {
            "schema_version": "flygym-steering-dose-calibration-source-freeze-v7",
            "revision": 7,
            "status": "preparation_frozen_execution_entrypoint_ready_not_run",
            "prior_revision_6_source_freeze": {
                "path": str(REVISION_6_ROOT / "source_freeze.json"),
                "sha256": R6_SOURCE_FREEZE_SHA256,
                "independent_review_json_sha256": R6_REVIEW_JSON_SHA256,
                "independent_review_md_sha256": R6_REVIEW_MD_SHA256,
                "source_file_count_inherited": len(freeze_r6["source_files"]),
                "asset_count_inherited": freeze_r6["asset_manifest"]["asset_count"],
                "verified_asset_count_inherited": freeze_r6["asset_manifest"]["verified_asset_count"],
                "r6_bundle_snapshot_inventory": r6_snapshot_files,
                "r6_review_pins": {
                    "runner_sha256": R6_RUNNER_SHA256,
                    "preparer_sha256": R6_PREPARER_SHA256,
                    "protocol_sha256": R6_PROTOCOL_SHA256,
                    "source_freeze_sha256": R6_SOURCE_FREEZE_SHA256,
                    "preparation_review_sha256": R6_PREPARATION_REVIEW_SHA256,
                    "independent_review_json_sha256": R6_REVIEW_JSON_SHA256,
                    "independent_review_md_sha256": R6_REVIEW_MD_SHA256,
                },
                "r6_runs_directory_absent": True,
                "r6_execution_approval_absent": True,
            },
            "source_files": source_files,
            "asset_manifest": json.loads(json.dumps(freeze_r6["asset_manifest"])),
            "copied_inputs": copied_inputs,
            "bundle_files": bundle_files,
            "native_model_reference": native_reference,
            "historical_revision_5": json.loads(json.dumps(freeze_r6["historical_revision_5"])),
            "historical_revision_6_review": {
                "review_outcome": independent_review["review_outcome"],
                "review_json_snapshot": "frozen_inputs/revision_6_independent_review/mechanical-dose-revision-6-independent-review-v1.json",
                "review_md_snapshot": "frozen_inputs/revision_6_independent_review/mechanical-dose-revision-6-independent-review-v1.md",
                "review_json_sha256": R6_REVIEW_JSON_SHA256,
                "review_md_sha256": R6_REVIEW_MD_SHA256,
                "blockers_addressed_by_revision_7": [
                    "explicit OOMPolicy=continue service with live systemd property readback",
                    "hidden worker requires live parent identity, same cgroup, inherited fixed-pose queue flock, active cohort, exact completed prefix hashes, required pair/P gates, and hard time limits",
                ],
            },
            "case_process_isolation_policy": protocol["execution"]["case_worker_policy"],
            "systemd_service_policy": protocol["execution"]["systemd_service"],
            "systemd_terminal_fallback": protocol["execution"]["systemd_terminal_fallback"],
            "runner_sha256": sha256_file(RUNNER_PATH),
            "preparation_script_sha256": sha256_file(prep_path),
            "protocol_sha256": None,
            "raw_asset_binaries_copied": False,
            "future_execution_must_reverify_all_source_asset_input_and_reference_hashes": True,
            "no_execution_approval_created": True,
            "revision_6_files_modified": False,
            "revision_5_failure_records_modified": False,
        }
        dump_json(stage / "protocol.json", protocol)
        freeze["protocol_sha256"] = sha256_file(stage / "protocol.json")
        dump_json(stage / "source_freeze.json", freeze)
        freeze_sha = sha256_file(stage / "source_freeze.json")

        review = {
            "schema_version": "flygym-steering-dose-calibration-preparation-review-v7",
            "status": "prepared_not_executed",
            "conclusion": "Revision 7 is prepared for independent review. It retains the strict native MJB gate, exact 15-case scientific design, traces and analysis, and CPU2/Mem2G/600/9000 limits. It explicitly runs under a transient systemd unit with OOMPolicy=continue, records live unit properties, and binds each case worker to its active coordinator, queue flock, and completed cohort prefix. No execution or runtime memory improvement is claimed.",
            "entrypoint_path": str(RUNNER_PATH.resolve()),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "preparation_script_sha256": prep_sha,
            "source_freeze_sha256": freeze_sha,
            "protocol_sha256": freeze["protocol_sha256"],
            "case_count": len(protocol["execution"]["ordered_cases"]),
            "design_unchanged_from_revision_6": protocol["design"] == protocol_r6["design"] and protocol["execution"]["ordered_cases"] == protocol_r6["execution"]["ordered_cases"],
            "model_identity_gate_unchanged": protocol["execution"]["model_identity_gate"] == protocol_r6["execution"]["model_identity_gate"],
            "resource_limits_unchanged": {
                "cpu_limit_cores": protocol["execution"]["cpu_limit_cores"],
                "memory_limit_bytes": protocol["execution"]["memory_limit_bytes"],
                "case_timeout_s": protocol["execution"]["case_timeout_s"],
                "batch_timeout_s": protocol["execution"]["batch_timeout_s"],
            },
            "r6_independent_review": {
                "review_outcome": independent_review["review_outcome"],
                "json_sha256": R6_REVIEW_JSON_SHA256,
                "md_sha256": R6_REVIEW_MD_SHA256,
                "preserved_without_relabeling": True,
            },
            "oom_policy_resolution": {
                "fixed_launch_unit": SERVICE_UNIT_NAME,
                "required_oom_policy": "continue",
                "runner_reads_actual_unit_properties": True,
                "root_terminal_fallback_if_coordinator_dies": True,
                "runtime_attestation_not_observed_in_preparation": True,
            },
            "worker_entrypoint_resolution": {
                "parent_pid_starttime_argv_attested": True,
                "same_service_cgroup_attested": True,
                "inherited_queue_flock_fd_attested_in_child_and_parent": True,
                "active_cohort_manifest_hash_and_exact_case_index_required": True,
                "completed_prefix_runrecord_hashes_rechecked": True,
                "same_seed_dzero_and_constant_p_required_after_first_case": True,
                "worker_case_timeout_max_s": CASE_TIMEOUT_S,
                "cohort_batch_timeout_max_s": BATCH_TIMEOUT_S,
            },
            "r5_oom_history": {
                "root_terminal_sha256": R5_OOM_TERMINAL_SHA256,
                "cohort_remains_stale_running_after_parent_kill": True,
                "one_case_complete_and_second_case_partial_at_6500_of_7000_ticks": True,
                "old_r5_and_r6_artifacts_modified": False,
            },
            "memory_evidence_boundary": "Separate child processes are a code-level memory lifetime boundary only. A single case can still hit the unchanged 2-GiB limit; no memory runtime check or 15-case run was made.",
            "scope_boundary": "Preparation only: no body, worker, model compilation, physics, NN, GPU, tests, execution approval, or approval artifact was created.",
        }
        dump_json(stage / "preparation-review.json", review)
        review_md = [
            "# Mechanical steering dose calibration — revision 7",
            "",
            "**Status: prepared for independent review; not executed.** Revision 5 and revision 6 artifacts remain unchanged. This bundle contains no run output or root execution approval.",
            "",
            "## Preserved science and model identity",
            "",
            "The 15-case seed-major design, 7-second duration, P=0.258, AP d schedule and ramp, exact native `session.model` MJB SHA, initial qpos/qvel/options/array-order gates, controller, body configuration, traces, contacts, and analysis are copied unchanged from r6. CPU2, Mem2G, 600-second case and 9000-second cohort limits are unchanged.",
            "",
            "## Resolution of the r6 review blockers",
            "",
            f"The pinned launch is `{invocation}`. It sets `OOMPolicy=continue`, CPUQuota=200%, MemoryMax=2147483648, and RuntimeMaxSec=9600, leaving room to record the 9000-second cohort terminal. Before any physics import, the coordinator reads back the live unit properties, InvocationID and ControlGroup and verifies the effective cgroup cap. Workers repeat the readback, and cohort/worker terminal JSON records retain the observed properties.",
            "",
            "Each hidden worker must be the direct child of the active r7 `--execute` coordinator. It verifies parent PID and `/proc` start time, interpreter/runner/argv, same cgroup, and the inherited shared flock descriptor for the fixed-pose queue. The coordinator holds that queue lock for the full cohort and publishes one active case in the cohort manifest before spawning the worker.",
            "",
            "The worker checks the active manifest hash, cohort order/index, every completed-prefix run-record hash, and the expected same-seed d=0 run record. After the first case it requires the cohort P trace hash and confirms every completed prefix record agrees. The JSON input cannot omit those gates. Case timeout is an integer no larger than 600 seconds; the batch is no larger than 9000 seconds, with time reserved for worker setup and termination.",
            "",
            "## Terminal evidence and limits",
            "",
            "`OOMPolicy=continue` lets a child OOM be observed by the coordinator when the coordinator survives. It does not guarantee the coordinator itself survives. If the process or service ends before finalizing `cohort-manifest.json`, root must collect actual systemd unit result, exit status, journal and cgroup evidence. A stale `running` manifest is never a complete result. This bundle creates no root terminal record or approval.",
            "",
            "R6's independent review and both blockers are copied and pinned under `frozen_inputs/revision_6_independent_review/`; the entire r6 bundle is copied byte-for-byte under `frozen_inputs/revision_6_bundle/`. R5's original OOM terminal and partial second-case record remain intact. The per-case process boundary has not been runtime-validated, and one case may still exceed 2 GiB.",
            "",
            "No body simulation, model compilation, worker, test, NN, GPU, or execution was performed while preparing this bundle.",
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
                        help="verify r6 and prepare the revision-7 bundle without execution")
    parser.parse_args()
    try:
        review = prepare()
    except (PreparationError, OSError, ValueError, KeyError) as exc:
        print(f"preparation failed: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps({
        "status": review["status"],
        "bundle_root": str(OUTPUT_ROOT),
        "runner_sha256": review["runner_sha256"],
        "preparation_script_sha256": review["preparation_script_sha256"],
        "source_freeze_sha256": review["source_freeze_sha256"],
        "protocol_sha256": review["protocol_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
