#!/usr/bin/env python3
"""CAS-backed dispatch for the frozen 15-case mechanical steering cohort.

Prepare and run from the pinned FlyGym interpreter:

    /data/environments/projects/flygym-reference/bin/python scripts/run_flygym_storage_dispatch_v1.py --prepare
    /data/environments/projects/flygym-reference/bin/python scripts/run_flygym_storage_dispatch_v1.py --execute
    /data/environments/projects/flygym-reference/bin/python scripts/run_flygym_storage_dispatch_v1.py --policy-check

The coordinator starts each mechanical capture in a separate systemd unit.  It
imports revision 7's `_run_case` without editing it, and uses a temporary
symlink view for the small frozen JSON inputs.  Policy-check captures stay in
temporary staging; only H1 objects from H2-mismatching cases are retained.
Requirement definitions remain in REQUIREMENTS.md; this protocol cites R1–R4.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

if __package__:
    from . import artifact_common
    from .artifact_common import read_json as read_json_value, sha256_file, utc_now_iso
else:
    import artifact_common
    from artifact_common import read_json as read_json_value, sha256_file, utc_now_iso


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
PYTHON_PATH = "/data/environments/projects/flygym-reference/bin/python"
LEGACY_RUNNER = SCRIPTS / "run_flygym_steering_dose_calibration_v7.py"
LEGACY_BUNDLE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/mechanical-steering-dose-calibration-v1/revision-7"
)
OBJECT_STORE_ROOT = Path("/data/artifacts/by-project/malecns-gpu/object-store")
ARTIFACT_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "storage-remediation-implementation-20261008-v1/mechanical-dispatch"
)
PROTOCOL_PATH = ARTIFACT_ROOT / "protocol.json"
STAGING_PARENT = Path("/data/incoming/staging/malecns-gpu/mechanical-dispatch-v1")
GLOBAL_NN_LOCK = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-recorded-retina-neural-ball-harness-20261007-v2/locks/"
    "neural-worker-and-ball-case.lock"
)
QUEUE_STATUS_PATH = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/spatial-fixed-pose-calibration-v1-revision-2/queue-status.json"
)
QUEUE_LOCK_PATH = QUEUE_STATUS_PATH.with_name("queue.lock")

PARENT_CPU_CORES = 2.0
PARENT_MEMORY_BYTES = 2 * 1024**3
PARENT_RUNTIME_S = 10_800
CASE_CPU_CORES = 4.0
CASE_MEMORY_BYTES = 5 * 1024**3
CASE_RUNTIME_S = 900
CASE_INNER_TIMEOUT_S = 780
CGROUP_TOPOLOGY_PROBE = {
    "probe_date": "2026-10-08",
    "parent_and_case_units_are": "sibling systemd --user services under app.slice",
    "case_unit_inherits_parent_limits": False,
    "parent_effective_limits": {"cpu.max": "200000 100000", "memory.max": str(PARENT_MEMORY_BYTES)},
    "case_effective_limits": {"cpu.max": "400000 100000", "memory.max": str(CASE_MEMORY_BYTES)},
    "single_serial_case_combined_caps": {"cpu_cores": 6.0, "memory_bytes": PARENT_MEMORY_BYTES + CASE_MEMORY_BYTES},
    "max_concurrent_case_units": 1,
    "probe_units": [
        "malecns-storage-dispatch-parent-child-probe-20261008.service",
        "malecns-storage-dispatch-child-resource-probe-20261008.service",
    ],
}
BASELINE_POLICY = {
    "CPUQuota": "200%",
    "MemoryMax": 2 * 1024**3,
    "RuntimeMaxSec": 600,
    "OOMPolicy": "continue",
    "threads": {name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    )},
}
CANDIDATE_POLICY = {
    "CPUQuota": "400%",
    "MemoryMax": CASE_MEMORY_BYTES,
    "RuntimeMaxSec": CASE_RUNTIME_S,
    "inner_timeout_s": CASE_INNER_TIMEOUT_S,
    "OOMPolicy": "continue",
    "threads": {name: "2" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    )},
}

# This is the independent capture contract.  It must not be derived from the
# files observed in a run directory.
REQUIRED_MEMBER_MANIFEST: dict[str, str] = {
    "trajectory.npz": "E2",
    "contacts.npz": "E2",
    "controller_telemetry.jsonl": "E2",
    "analysis-summary.json": "E3",
    "array_order.json": "E3",
    "compiled_model.mjb": "E3",
    "compiled_model.xml": "E3",
    "diagnostics.png": "E3",
    "effective_config.json": "E3",
    "initial_state.json": "E3",
    "integrity.json": "E3",
    "manifest.json": "E3",
    "model_options.json": "E3",
    "run-record.json": "E3",
    "seed.json": "E3",
    "summary.json": "E3",
    "upstream_asset_hashes.json": "E3",
}

# Each entry is a direct H1-resolvable CAS input used by the unchanged v7
# capture function.  The path is a temporary view path, not a copy of history.
FROZEN_INPUT_PATHS = {
    "prior_model_options": "frozen_inputs/prior_model_options.json",
    "native_model_options": "frozen_inputs/native_model_reference/effective_model_options.json",
    "prior_array_order": "frozen_inputs/prior_array_order.json",
    "native_array_order": "frozen_inputs/native_model_reference/runtime_array_order.json",
    "paired_screen_reference": "paired-screen-reference.json",
}

REQUIREMENT_REFERENCES = {
    "R1": ["R1.1", "R1.2", "R1.3.1–R1.3.8", "R1.4", "R1.5"],
    "R2": ["R2.1–R2.5"],
    "R3": ["R3.1–R3.5"],
    "R4": ["R4.1–R4.4"],
    "single_sources_of_truth": ["§1.4.1", "§1.4.2"],
}


class DispatchError(RuntimeError):
    """Invalid protocol, incomplete case, or failed systemd capture."""


def load_dispatch_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Keep the dispatcher's JSON-object requirement above shared parsing."""
    value = read_json_value(path)
    if not isinstance(value, dict):
        raise DispatchError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def import_legacy_runner():
    spec = importlib.util.spec_from_file_location("run_flygym_steering_dose_calibration_v7_storage_adapter", LEGACY_RUNNER)
    if spec is None or spec.loader is None:
        raise DispatchError(f"cannot import the immutable v7 runner: {LEGACY_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _storage_api():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from research_artifact_store import ArtifactInput, ArtifactReference, ResearchArtifactStore, compute_physics_config_hash
    return ArtifactInput, ArtifactReference, ResearchArtifactStore, compute_physics_config_hash


def _content_api():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from measurement_content_hash import MECHANICAL_CASE_SCHEMA, compare_cases
    return MECHANICAL_CASE_SCHEMA, compare_cases


def load_legacy_context() -> dict[str, Any]:
    legacy = import_legacy_runner()
    old_protocol, source_freeze, old_protocol_sha = legacy.verify_frozen_bundle()
    cases = legacy._expected_case_order(old_protocol)
    input_root = LEGACY_BUNDLE / "frozen_inputs"
    walk_effective = load_dispatch_record(input_root / "v4_1_effective_config.json")
    controller_config = load_dispatch_record(input_root / "v4_1_controller_config.json")["config"]
    paired_screen = load_dispatch_record(LEGACY_BUNDLE / "paired-screen-reference.json")["summary"]
    baseline_model_signature = paired_screen["matched_inputs"]["model_signatures"]["static_d0"]
    baseline_initial_state_hashes = paired_screen["matched_inputs"]["initial_state_hashes"]["static_d0"]
    if baseline_model_signature.get("compiled_model_sha256") != legacy.EXPECTED_PAIRED_SCREEN_RAW_XML_SHA256:
        raise DispatchError("paired-screen raw XML identity differs from the v7 pin")
    if baseline_initial_state_hashes != {
        "qpos_sha256": legacy.EXPECTED_HISTORICAL_QPOS_SHA256,
        "qvel_sha256": legacy.EXPECTED_HISTORICAL_QVEL_SHA256,
    }:
        raise DispatchError("paired-screen initial state differs from the v7 pin")
    return {
        "legacy": legacy,
        "protocol": old_protocol,
        "protocol_sha256": old_protocol_sha,
        "freeze": source_freeze,
        "source_freeze_sha256": sha256_file(LEGACY_BUNDLE / "source_freeze.json"),
        "cases": cases,
        "walk_values": dict(walk_effective["walk_config"]),
        "controller_values": controller_config,
        "baseline_model_signature": baseline_model_signature,
        "baseline_initial_state_hashes": baseline_initial_state_hashes,
    }


def _historical_cases(context: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    legacy = context["legacy"]
    protocol = context["protocol"]
    ordered = context["cases"]
    cohort_path = legacy.RUN_ROOT / "cohort-manifest.json"
    cohort = load_dispatch_record(cohort_path)
    if cohort.get("status") != "complete" or cohort.get("completed_case_count") != len(ordered) or len(ordered) != 15:
        raise DispatchError("revision-7 historical cohort is not the complete 15-case reference")
    if cohort.get("case_order") != [item["case_id"] for item in ordered]:
        raise DispatchError("revision-7 case ordering differs from its cohort manifest")
    expected_files = set(REQUIRED_MEMBER_MANIFEST)
    records: dict[str, dict[str, Any]] = {}
    for case in ordered:
        case_id = str(case["case_id"])
        entry = cohort.get("cases", {}).get(case_id)
        case_dir = legacy.RUN_ROOT / case_id
        run_path = case_dir / "run-record.json"
        if not isinstance(entry, dict) or entry.get("status") != "complete" or not run_path.is_file():
            raise DispatchError(f"historical case is not complete: {case_id}")
        if sha256_file(run_path) != entry.get("run_record_sha256"):
            raise DispatchError(f"historical run-record hash differs from cohort: {case_id}")
        record = load_dispatch_record(run_path)
        if record.get("status") != "complete" or record.get("case_id") != case_id:
            raise DispatchError(f"historical run-record is not complete for {case_id}")
        actual = {path.name for path in case_dir.iterdir() if path.is_file()}
        if actual != expected_files:
            raise DispatchError(f"historical case member set differs for {case_id}: {sorted(actual ^ expected_files)}")
        capture_files = record.get("capture_files")
        if not isinstance(capture_files, dict) or set(capture_files) != expected_files - {"run-record.json"}:
            raise DispatchError(f"historical capture integrity table is incomplete for {case_id}")
        policy_env = record.get("initialization_provenance_preconstruction", {}).get("environment", {})
        if any(policy_env.get(name) != "1" for name in BASELINE_POLICY["threads"]):
            raise DispatchError(f"historical thread policy is not the frozen 1-thread baseline for {case_id}")
        if (record.get("cgroup_effective_cpu_cap_cores") != 2.0
                or record.get("cgroup_effective_memory_cap_bytes") != 2 * 1024**3):
            raise DispatchError(f"historical cgroup policy differs from the recorded 2-CPU/2-GiB baseline for {case_id}")
        records[case_id] = record
    first_p_trace = records[ordered[0]["case_id"]].get("analysis_summary", {}).get("constant_P_trace_sha256")
    if not first_p_trace:
        raise DispatchError("historical first case lacks the cohort P-trace hash")
    if any(record.get("analysis_summary", {}).get("constant_P_trace_sha256") != first_p_trace for record in records.values()):
        raise DispatchError("historical cases do not share the frozen constant-P trace")
    return records, cohort


def _body_config(context: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(context["walk_values"])
    value["duration_s"] = float(context["protocol"]["execution"]["duration_s"])
    value["seed"] = int(case["seed"])
    for key in ("spawn_position_mm", "spawn_quaternion_wxyz", "position_actuator_forcerange_native", "camera_resolution_hw"):
        if key in value:
            value[key] = tuple(value[key])
    return value


def physics_config(context: Mapping[str, Any], case: Mapping[str, Any], policy: Mapping[str, Any], dispatcher_sha256: str) -> dict[str, Any]:
    """Keep unverified runtime fields in the config hash, per §1.5 and R1.4."""
    legacy = context["legacy"]
    execution = context["protocol"]["execution"]
    freeze = context["freeze"]
    frozen_input_hashes = {
        logical_name: sha256_file(LEGACY_BUNDLE / relative_path)
        for logical_name, relative_path in FROZEN_INPUT_PATHS.items()
    }
    h2_schema, _compare = _content_api()
    h2_module_sha256 = sha256_file(SCRIPTS / "measurement_content_hash.py")
    return {
        "case": {"case_id": str(case["case_id"]), "seed": int(case["seed"]), "d": float(case["d"])},
        "stimulus_schedule": {
            "power_P_left_right": execution["power_P_left_right"],
            "duration_s": execution["duration_s"],
            "common_prefix_s": execution["common_prefix_s"],
            "ramp_duration_s": execution["ramp_duration_s"],
            "d_schedule": execution["d_schedule"],
            "analysis_window_s": execution["analysis_window_s"],
        },
        "model_identity": {
            "compiled_model_mjb_sha256": legacy.EXPECTED_COMPILED_MJB_SHA256,
            "compiled_model_mjb_size_bytes": legacy.EXPECTED_COMPILED_MJB_SIZE_BYTES,
            "baseline_model_signature": context["baseline_model_signature"],
            "baseline_initial_state_hashes": context["baseline_initial_state_hashes"],
        },
        "body_config": _body_config(context, case),
        "controller_config": context["controller_values"],
        "frozen_model_reference_h1": frozen_input_hashes,
        "capture_format": {
            "schema_id": h2_schema.schema_id,
            "case_member_manifest": REQUIRED_MEMBER_MANIFEST,
        },
        "source_hashes": {
            "dispatcher_sha256": dispatcher_sha256,
            "artifact_common_sha256": sha256_file(artifact_common.__file__),
            "legacy_runner_sha256": sha256_file(LEGACY_RUNNER),
            "measurement_content_hash_module_sha256": h2_module_sha256,
            "source_freeze_sha256": context["source_freeze_sha256"],
            "legacy_source_files": {
                path: row["sha256"] for path, row in freeze.get("source_files", {}).items()
            },
            "assets": {
                name: row["sha256"] for name, row in freeze.get("asset_manifest", {}).get("files", {}).items()
            },
        },
        # CPU quota, BLAS threads, memory and timeout remain part of the
        # physics hash until full-cohort H2 evidence supports this exact policy
        # bundle's separation under R1.3.  No field is presumed inert here.
        "runtime_settings_pending_R1_3": dict(policy),
        "python_realpath": str(Path(PYTHON_PATH).resolve()),
        "package_versions": {"flygym": "2.1.0", "mujoco": "3.9.0", "numpy": "2.5.3"},
    }


def build_protocol(context: Mapping[str, Any], store: Any, *, dispatcher_sha256: str | None = None) -> dict[str, Any]:
    ArtifactInput, _ArtifactReference, _Store, config_hash = _storage_api()
    h2_schema, _compare_cases = _content_api()
    dispatcher_sha256 = dispatcher_sha256 or sha256_file(__file__)
    records, cohort = _historical_cases(context)
    frozen_references = []
    for logical_name, relative_path in FROZEN_INPUT_PATHS.items():
        source = LEGACY_BUNDLE / relative_path
        reference = store.put_file(
            source,
            original_name=source.name,
            artifact_class="E3",
            locator=str(source),
        )
        frozen_references.append({"logical_name": logical_name, "relative_path": relative_path, **reference.to_dict()})
    cases = []
    for case in context["cases"]:
        case_id = str(case["case_id"])
        baseline_config = physics_config(context, case, BASELINE_POLICY, dispatcher_sha256)
        candidate_config = physics_config(context, case, CANDIDATE_POLICY, dispatcher_sha256)
        base_record = records[case_id]
        cases.append({
            "case_id": case_id,
            "seed": int(case["seed"]),
            "d": float(case["d"]),
            "historical_case_locator": str(context["legacy"].RUN_ROOT / case_id),
            "historical_run_record_sha256": sha256_file(context["legacy"].RUN_ROOT / case_id / "run-record.json"),
            "baseline_physics_config_hash": config_hash(baseline_config),
            "candidate_physics_config_hash": config_hash(candidate_config),
            "baseline_cgroup": {
                "effective_cpu_cap_cores": base_record["cgroup_effective_cpu_cap_cores"],
                "effective_memory_cap_bytes": base_record["cgroup_effective_memory_cap_bytes"],
                "memory_peak_bytes": base_record["cgroup_memory_peak_after_bytes"],
                "memory_peak_source": "revision-7 run-record cgroup memory.peak",
            },
        })
    source_freeze = context["freeze"]
    return {
        "schema_version": "malecns-mechanical-storage-dispatch-v1",
        "prepared_utc": utc_now_iso(),
        "requirements_path": "/data/artifacts/by-project/malecns-gpu/engineering/storage-remediation-directive-20261008-v1/REQUIREMENTS.md",
        "requirements_citations": REQUIREMENT_REFERENCES,
        "object_store_root": str(OBJECT_STORE_ROOT),
        "historical_bundle_locator": str(LEGACY_BUNDLE),
        "historical_protocol_sha256": context["protocol_sha256"],
        "historical_source_freeze_sha256": context["source_freeze_sha256"],
        "dispatcher_sha256": dispatcher_sha256,
        "artifact_common_sha256": sha256_file(artifact_common.__file__),
        "legacy_runner_sha256": sha256_file(LEGACY_RUNNER),
        "measurement_content_hash_module_sha256": sha256_file(SCRIPTS / "measurement_content_hash.py"),
        "research_artifact_store_module_sha256": sha256_file(SCRIPTS / "research_artifact_store.py"),
        "legacy_source_files": {path: row["sha256"] for path, row in source_freeze.get("source_files", {}).items()},
        "frozen_inputs": frozen_references,
        "case_member_manifest": dict(REQUIRED_MEMBER_MANIFEST),
        "h2_schema_id": h2_schema.schema_id,
        "policy_trial": {
            "scope": "one combined execution-policy bundle; no individual-field attribution",
            "classification_before_measurement": "unverified hypothesis; CPU quota, BLAS threads, memory and timeout remain in physics_config hashes",
            "baseline": BASELINE_POLICY,
            "candidate": CANDIDATE_POLICY,
            "unchanged_physical_design": ["model", "controller", "seed", "dose schedule", "P schedule", "dt", "capture fields", "source code"],
        },
        "coordinator_limits": {
            "CPUQuota": "200%", "MemoryMax": PARENT_MEMORY_BYTES, "RuntimeMaxSec": PARENT_RUNTIME_S,
            "serial_cases": True,
            "global_nn_body_lock": str(GLOBAL_NN_LOCK),
        },
        "candidate_case_unit_limits": {
            "CPUQuota": "400%", "MemoryMax": CASE_MEMORY_BYTES, "RuntimeMaxSec": CASE_RUNTIME_S,
            "inner_case_timeout_s": CASE_INNER_TIMEOUT_S, "OOMPolicy": "continue",
            "memory_peak_source": "case process reads current unified cgroup v2 memory.peak",
        },
        "observed_cgroup_topology": CGROUP_TOPOLOGY_PROBE,
        "historical_cohort": {
            "status": cohort["status"],
            "completed_case_count": cohort["completed_case_count"],
            "case_order": [case["case_id"] for case in cases],
            "cohort_manifest_sha256": sha256_file(context["legacy"].RUN_ROOT / "cohort-manifest.json"),
        },
        "cases": cases,
    }


def prepare() -> dict[str, Any]:
    context = load_legacy_context()
    _ArtifactInput, _ArtifactReference, Store, _config_hash = _storage_api()
    store = Store(OBJECT_STORE_ROOT)
    protocol = build_protocol(context, store)
    if PROTOCOL_PATH.exists():
        existing = load_dispatch_record(PROTOCOL_PATH)
        if {key: value for key, value in existing.items() if key != "prepared_utc"} != {
            key: value for key, value in protocol.items() if key != "prepared_utc"
        }:
            raise DispatchError(f"refusing to replace an existing protocol: {PROTOCOL_PATH}")
        return existing
    atomic_json(PROTOCOL_PATH, protocol)
    return protocol


def load_prepared_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PROTOCOL_PATH.is_file():
        raise DispatchError("run --prepare first; the dispatcher protocol is missing")
    protocol = load_dispatch_record(PROTOCOL_PATH)
    if protocol.get("schema_version") != "malecns-mechanical-storage-dispatch-v1":
        raise DispatchError("unexpected dispatcher protocol schema")
    if protocol.get("dispatcher_sha256") != sha256_file(__file__):
        raise DispatchError("dispatcher source changed after prepare; preserve protocol and prepare a new version")
    if protocol.get("artifact_common_sha256") != sha256_file(artifact_common.__file__):
        raise DispatchError("shared artifact source changed after prepare; preserve protocol and prepare a new version")
    context = load_legacy_context()
    if protocol.get("historical_protocol_sha256") != context["protocol_sha256"]:
        raise DispatchError("historical v7 protocol changed after prepare")
    if protocol.get("historical_source_freeze_sha256") != context["source_freeze_sha256"]:
        raise DispatchError("historical v7 source freeze changed after prepare")
    if protocol.get("measurement_content_hash_module_sha256") != sha256_file(SCRIPTS / "measurement_content_hash.py"):
        raise DispatchError("H2 implementation changed after prepare")
    if protocol.get("research_artifact_store_module_sha256") != sha256_file(SCRIPTS / "research_artifact_store.py"):
        raise DispatchError("CAS implementation changed after prepare")
    return protocol, context


def create_reference_view(protocol: Mapping[str, Any], store: Any, root: Path) -> Path:
    """Build only symlinks to H1-resolved CAS E3 inputs, outside case trees."""
    _ArtifactInput, ArtifactReference, _Store, _config_hash = _storage_api()
    view = root / "reference-view"
    view.mkdir(parents=True, exist_ok=False)
    for item in protocol["frozen_inputs"]:
        reference = ArtifactReference.from_dict(item)
        source = store.resolve_reference(reference, verify_integrity=True)
        target = view / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return view


def expected_pair_inputs(context: Mapping[str, Any], case_id: str) -> tuple[dict[str, Any] | None, str]:
    records, _cohort = _historical_cases(context)
    case = next((row for row in context["cases"] if row["case_id"] == case_id), None)
    if case is None:
        raise DispatchError(f"case is not in the frozen cohort: {case_id}")
    expected_pair = None
    if float(case["d"]) != 0.0:
        zero_id = f"seed{int(case['seed'])}_dzero_0p00"
        expected_pair = records.get(zero_id)
        if expected_pair is None:
            raise DispatchError(f"same-seed d=0 pair is missing for {case_id}")
    first_id = context["cases"][0]["case_id"]
    expected_p = records[first_id]["analysis_summary"]["constant_P_trace_sha256"]
    return expected_pair, expected_p


def run_legacy_case(context: Mapping[str, Any], protocol: Mapping[str, Any], store: Any,
                    case_id: str, staging_root: Path) -> dict[str, Any]:
    """One-case child body: unchanged v7 _run_case, CAS symlink view only."""
    if str(Path(sys.executable).resolve()) != str(Path(PYTHON_PATH).resolve()):
        raise DispatchError(f"case child requires pinned interpreter {PYTHON_PATH}")
    legacy = context["legacy"]
    view = create_reference_view(protocol, store, staging_root)
    queue_lock, queue_attestation = legacy.acquire_terminal_fixed_pose_queue_lock()
    try:
        legacy.BUNDLE_ROOT = view
        legacy.CPU_LIMIT_CORES = CASE_CPU_CORES
        legacy.MEMORY_LIMIT_BYTES = CASE_MEMORY_BYTES
        legacy.CASE_TIMEOUT_S = CASE_INNER_TIMEOUT_S
        legacy._set_simulation_environment()
        os.environ.update(CANDIDATE_POLICY["threads"])
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        import numpy as np
        import run_flygym_ap_body_yaw_calibration_v1 as helper
        case = next(row for row in context["cases"] if row["case_id"] == case_id)
        expected_pair, expected_p_trace = expected_pair_inputs(context, case_id)
        return legacy._run_case(
            case=case,
            protocol=context["protocol"],
            freeze=context["freeze"],
            expected_pair=expected_pair,
            expected_p_trace_sha=expected_p_trace,
            serial_queue_attestation=queue_attestation,
            out_root=staging_root,
            source_freeze_sha=context["source_freeze_sha256"],
            walk_values=context["walk_values"],
            controller_values=context["controller_values"],
            helper=helper,
            np=np,
            baseline_model_signature=context["baseline_model_signature"],
            baseline_initial_state_hashes=context["baseline_initial_state_hashes"],
            case_timeout_s=CASE_INNER_TIMEOUT_S,
        )
    finally:
        fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)
        queue_lock.close()


def validate_case_directory(case_dir: Path, *, require_complete: bool = True) -> tuple[dict[str, str], dict[str, Any]]:
    actual = {path.name for path in case_dir.iterdir() if path.is_file()}
    if actual != set(REQUIRED_MEMBER_MANIFEST):
        raise DispatchError(f"case member set differs from frozen 17-file contract: {sorted(actual ^ set(REQUIRED_MEMBER_MANIFEST))}")
    record = load_dispatch_record(case_dir / "run-record.json")
    if record.get("status") != "complete":
        if require_complete:
            raise DispatchError(f"case capture did not complete: {record.get('status')}")
        return {}, record
    capture_files = record.get("capture_files")
    if not isinstance(capture_files, dict) or set(capture_files) != set(REQUIRED_MEMBER_MANIFEST) - {"run-record.json"}:
        raise DispatchError("run-record capture_files does not list the required 16 non-run-record files")
    integrity: dict[str, str] = {}
    for relative_path in sorted(REQUIRED_MEMBER_MANIFEST):
        path = case_dir / relative_path
        found = sha256_file(path)
        if relative_path != "run-record.json":
            recorded = capture_files[relative_path]
            if recorded.get("sha256") != found or recorded.get("size_bytes") != path.stat().st_size:
                raise DispatchError(f"captured file disagrees with run-record H1/size: {relative_path}")
        integrity[relative_path] = found
    return integrity, record


def store_complete_case(store: Any, case_dir: Path, case: Mapping[str, Any], config: Mapping[str, Any], config_hash: str):
    ArtifactInput, _ArtifactReference, _Store, _compute_hash = _storage_api()
    integrity, record = validate_case_directory(case_dir)
    files = {
        relative_path: ArtifactInput(case_dir / relative_path, REQUIRED_MEMBER_MANIFEST[relative_path], original_name=relative_path)
        for relative_path in REQUIRED_MEMBER_MANIFEST
    }
    stored = store.register_case_payload(
        case_identity=str(case["case_id"]),
        files=files,
        physics_config=config,
        declared_physics_config_hash=config_hash,
        completion_status=record["status"],
        recorded_integrity=integrity,
        required_member_manifest=REQUIRED_MEMBER_MANIFEST,
    )
    return stored


def persist_novel_mismatch_objects(store: Any, case_dir: Path, comparison: Any) -> list[dict[str, Any]]:
    """Persist only case-file H1s absent from E6; never index unresolved as reusable."""
    rows = [*comparison.candidate.payload_files, *comparison.candidate.metadata_files]
    references: list[dict[str, Any]] = []
    for row in rows:
        target = store.object_path(row.sha256)
        if target.exists():
            store.resolve_reference({
                "H1": row.sha256,
                "original_name": Path(row.path).name,
                "artifact_class": row.role,
                "size_bytes": row.size_bytes,
            }, verify_integrity=True)
            state = "already_present"
        else:
            ref = store.put_file(
                case_dir / row.path,
                original_name=Path(row.path).name,
                artifact_class=row.role,
                locator=f"policy-check:{case_dir.name}/{row.path}",
            )
            if ref.h1 != row.sha256:
                raise DispatchError(f"candidate changed while persisting {row.path}")
            state = "new_object"
        references.append({"path": row.path, "H1": row.sha256, "role": row.role, "size_bytes": row.size_bytes, "store_action": state})
    return references


def policy_case_decision(store: Any, reference_dir: Path, candidate_dir: Path, *, case_id: str) -> dict[str, Any]:
    schema, compare_cases = _content_api()
    comparison = compare_cases(reference_dir, candidate_dir, schema)
    row: dict[str, Any] = {
        "case_id": case_id,
        **comparison.to_dict(),
        "h2_match": comparison.reference.case_h2 == comparison.candidate.case_h2,
        "persisted_novel_objects": [],
    }
    if not row["h2_match"]:
        row["persisted_novel_objects"] = persist_novel_mismatch_objects(store, candidate_dir, comparison)
    return row


def find_reusable_record(store: Any, *, case_id: str, config_hash: str):
    candidates = store.query_cases(config_hash)
    last_rejected = None
    last_assessment = None
    for record in reversed(candidates):
        assessment = store.check_engineering_reuse(
            record,
            expected_physics_config_hash=config_hash,
            expected_member_manifest=REQUIRED_MEMBER_MANIFEST,
        )
        if record.case_identity != case_id:
            last_rejected = record
            last_assessment = assessment
            continue
        if assessment.eligible:
            certificate = store.reuse_certificate(
                record,
                expected_physics_config_hash=config_hash,
                expected_member_manifest=REQUIRED_MEMBER_MANIFEST,
            )
            return record, certificate
        last_rejected = record
        last_assessment = assessment
    if last_rejected is not None and last_assessment is not None:
        certificate = store.reuse_certificate(
            last_rejected,
            expected_physics_config_hash=config_hash,
            expected_member_manifest=REQUIRED_MEMBER_MANIFEST,
        )
        certificate["lookup_hit"] = True
        if last_rejected.case_identity != case_id:
            certificate["engineering_reuse_check"]["eligible_for_engineering_reuse"] = False
            certificate["engineering_reuse_check"].setdefault("reasons", []).append("case_identity_mismatch")
        return None, certificate
    return None, {
        "schema_version": "research-artifact-reuse-check-1",
        "case_identity": case_id,
        "physics_config_hash": config_hash,
        "lookup_hit": bool(candidates),
        "case_payload_hash": candidates[-1].case_payload_hash if candidates else None,
        "engineering_reuse_check": {
            "eligible_for_engineering_reuse": False,
            "completion_status_complete": False,
            "byte_integrity_match": False,
            "physics_config_match": False,
            "required_member_manifest_match": False,
            "reasons": ["no_eligible_case_payload"],
        },
    }


def acquire_global_lock():
    if not GLOBAL_NN_LOCK.is_file() or GLOBAL_NN_LOCK.is_symlink():
        raise DispatchError(f"shared NN/body lock must already be a regular file: {GLOBAL_NN_LOCK}")
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(GLOBAL_NN_LOCK, flags)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
        os.close(fd)
        raise DispatchError("shared NN/body lock is not a user-owned regular file")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise DispatchError("another NN/body experiment holds the shared lock") from exc
    return fd


def _current_unit_name() -> str:
    for row in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
        parts = row.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            units = [part for part in parts[2].split("/") if part.endswith(".service")]
            if units:
                return units[-1]
    raise DispatchError("cannot determine current systemd service from unified cgroup")


def _read_service_properties(unit_name: str) -> dict[str, str]:
    names = ("OOMPolicy", "MemoryMax", "CPUQuotaPerSecUSec", "RuntimeMaxUSec", "ControlGroup", "InvocationID", "Result", "ExecMainCode", "ExecMainStatus")
    result = subprocess.run(
        ["systemctl", "--user", "show", "--no-pager", *[f"--property={name}" for name in names], unit_name],
        check=False, capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise DispatchError(f"systemd show failed for {unit_name}: {result.stderr.strip()}")
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if any(name not in properties for name in names):
        raise DispatchError(f"systemd omitted required properties for {unit_name}")
    return properties


def _expected_systemd_usec(value: str) -> int:
    legacy = import_legacy_runner()
    return legacy._systemd_usec(value)


def _case_unit_ok(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("exit_code") != 0 or receipt.get("systemd_readback_error") or receipt.get("systemd_stop_exit_code") != 0:
        return False
    runtime = receipt.get("runtime_receipt")
    terminal = receipt.get("terminal_properties", {})
    client = receipt.get("systemd_client_start", {})
    runtime_props = runtime.get("properties_at_runtime", {}) if isinstance(runtime, Mapping) else {}
    runtime_cgroup = runtime.get("cgroup_at_runtime", {}) if isinstance(runtime, Mapping) else {}
    try:
        return (
            terminal.get("Result") == "success"
            and terminal.get("ExecMainCode") == "1"
            and terminal.get("ExecMainStatus") == "0"
            and terminal.get("ActiveState") == "active"
            and terminal.get("SubState") == "exited"
            and terminal.get("InvocationID") == runtime.get("invocation_id")
            and terminal.get("InvocationID") == client.get("invocation_id")
            and terminal.get("Id") == client.get("unit_name")
            and runtime_props.get("OOMPolicy") == "continue"
            and runtime_props.get("MemoryMax") == str(CASE_MEMORY_BYTES)
            and _expected_systemd_usec(runtime_props["CPUQuotaPerSecUSec"]) == int(CASE_CPU_CORES * 1_000_000)
            and _expected_systemd_usec(runtime_props["RuntimeMaxUSec"]) == CASE_RUNTIME_S * 1_000_000
            and runtime_props.get("ControlGroup")
            and runtime_props.get("ControlGroup") == runtime_cgroup.get("membership_path")
            and runtime_cgroup.get("effective_cpu_cap_cores") == CASE_CPU_CORES
            and runtime_cgroup.get("effective_memory_cap_bytes") == CASE_MEMORY_BYTES
        )
    except (AttributeError, KeyError, DispatchError, ValueError):
        return False


def _failure_summary(case_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    run_path = case_dir / "run-record.json"
    return {
        "run_record_sha256": sha256_file(run_path) if run_path.is_file() else None,
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "raw_capture_status": record.get("raw_capture_status"),
        "run_record": dict(record),
        "partial_artifacts": record.get("partial_artifacts", record.get("capture_files", {})),
        "cgroup_after": record.get("cgroup_after", record.get("cgroup_after_worker_exit")),
    }


def attest_current_unit(*, cpu_cores: float, memory_bytes: int, runtime_s: int) -> dict[str, Any]:
    unit = _current_unit_name()
    prop = _read_service_properties(unit)
    invocation_id = os.environ.get("INVOCATION_ID")
    cgroup = next((row.split(":", 2)[2] for row in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
                   if row.startswith("0::")), None)
    if prop["OOMPolicy"] != "continue" or prop["MemoryMax"] != str(memory_bytes):
        raise DispatchError(f"systemd unit properties do not match resource contract: {unit}")
    if _expected_systemd_usec(prop["CPUQuotaPerSecUSec"]) != int(cpu_cores * 1_000_000):
        raise DispatchError(f"systemd CPU quota differs from {cpu_cores:g} cores: {unit}")
    if _expected_systemd_usec(prop["RuntimeMaxUSec"]) != runtime_s * 1_000_000:
        raise DispatchError(f"systemd runtime differs from {runtime_s}s: {unit}")
    if prop["InvocationID"] != invocation_id or prop["ControlGroup"] != cgroup:
        raise DispatchError(f"service invocation/cgroup does not match current process: {unit}")
    legacy = import_legacy_runner()
    legacy.CPU_LIMIT_CORES = cpu_cores
    legacy.MEMORY_LIMIT_BYTES = memory_bytes
    actual_cgroup = legacy.cgroup_attestation()
    if (actual_cgroup["effective_cpu_cap_cores"] != cpu_cores
            or actual_cgroup["effective_memory_cap_bytes"] != memory_bytes):
        raise DispatchError(f"current cgroup effective limits differ from requested unit: {unit}")
    return {"unit_name": unit, "properties": prop, "cgroup": actual_cgroup}


def coordinator_systemd_command(mode: str) -> list[str]:
    unit = f"malecns-storage-dispatch-{mode}-{int(time.time())}.service"
    return [
        "systemd-run", "--user", "--wait", "--pipe", "--collect", f"--unit={unit}",
        "--property=CPUQuota=200%", f"--property=MemoryMax={PARENT_MEMORY_BYTES}",
        f"--property=RuntimeMaxSec={PARENT_RUNTIME_S}", "--property=OOMPolicy=continue",
        "--property=WorkingDirectory=" + str(PROJECT), "--", PYTHON_PATH, str(Path(__file__).resolve()), f"--{mode}",
    ]


def child_systemd_command(case_id: str, staging_root: Path, mode: str) -> tuple[list[str], str]:
    unit_base = f"malecns-storage-{mode}-{case_id.replace('_', '-')}-{int(time.time_ns())}"
    unit = unit_base + ".service"
    cmd = [
        # Keep the exited unit loaded long enough for a terminal property
        # readback.  --no-block is required because --wait would wait for the
        # remain-after-exit unit to be explicitly stopped by the coordinator.
        "systemd-run", "--user", "--no-block", "--remain-after-exit", f"--unit={unit}",
        "--property=CPUQuota=400%", f"--property=MemoryMax={CASE_MEMORY_BYTES}",
        f"--property=RuntimeMaxSec={CASE_RUNTIME_S}", "--property=OOMPolicy=continue",
        "--property=StandardOutput=append:" + str(staging_root / f"{case_id}.stdout.log"),
        "--property=StandardError=append:" + str(staging_root / f"{case_id}.stderr.log"),
        "--property=WorkingDirectory=" + str(PROJECT),
        "--setenv=OMP_NUM_THREADS=2", "--setenv=OPENBLAS_NUM_THREADS=2",
        "--setenv=MKL_NUM_THREADS=2", "--setenv=NUMEXPR_NUM_THREADS=2",
        "--setenv=PYTHONUNBUFFERED=1", "--setenv=PYTHONDONTWRITEBYTECODE=1",
        "--setenv=MALCENS_DISPATCH_CHILD=1", "--",
        PYTHON_PATH, str(Path(__file__).resolve()), "--case-child", case_id,
        "--staging-root", str(staging_root),
    ]
    return cmd, unit


def _parse_systemd_run_start(output: str) -> dict[str, str]:
    match = re.search(
        r"Running as unit:\s*(?P<unit>[^;\s]+)(?:;\s*invocation ID:\s*(?P<invocation>[0-9a-fA-F]+))?",
        output,
    )
    if not match:
        return {}
    invocation_id = match.group("invocation")
    return {
        "unit_name": match.group("unit"),
        "invocation_id": invocation_id.lower() if invocation_id else None,
        "invocation_id_source": "systemd_run_start" if invocation_id else None,
    }


def _read_terminal_unit_properties(unit: str, timeout_s: float) -> tuple[dict[str, str], str | None]:
    names = (
        "Id", "OOMPolicy", "MemoryMax", "CPUQuotaPerSecUSec", "RuntimeMaxUSec",
        "ControlGroup", "InvocationID", "Result", "ExecMainCode", "ExecMainStatus",
        "ActiveState", "SubState",
    )
    deadline = time.monotonic() + timeout_s
    last_error = "unit did not reach a retained terminal state before timeout"
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["systemctl", "--user", "show", "--no-pager", *[f"--property={name}" for name in names], unit],
                check=False, capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"systemctl show timed out: {exc}"
            time.sleep(0.2)
            continue
        if result.returncode != 0:
            last_error = f"systemctl show failed: {result.stderr.strip()}"
            time.sleep(0.2)
            continue
        properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        required = {"Id", "InvocationID", "Result", "ExecMainCode", "ExecMainStatus", "ActiveState", "SubState"}
        if not required.issubset(properties) or not properties.get("InvocationID"):
            last_error = "systemctl show omitted retained invocation/terminal properties"
            time.sleep(0.2)
            continue
        if properties.get("ActiveState") == "active" and properties.get("SubState") == "exited":
            return properties, None
        if properties.get("ActiveState") == "failed":
            return properties, None
        time.sleep(0.2)
    return {}, last_error


def _read_invocation_journal(invocation_id: str) -> dict[str, Any]:
    command = [
        "journalctl", "--user", "--no-pager", "--output=json", "--lines=100",
        f"_SYSTEMD_INVOCATION_ID={invocation_id}",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
        entries = []
        for line in result.stdout.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                entries.append({"raw": line[:2000]})
                continue
            entries.append({
                key: row[key] for key in ("__REALTIME_TIMESTAMP", "_SYSTEMD_UNIT", "PRIORITY", "MESSAGE")
                if key in row
            })
        return {
            "query_exit_code": result.returncode,
            "entry_count": len(entries),
            "entries": entries[-20:],
            "stderr": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {"query_error": f"TimeoutExpired: {exc}", "entry_count": 0, "entries": []}


def _read_tail(path: Path, limit_bytes: int = 8000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit_bytes), os.SEEK_SET)
        return stream.read(limit_bytes).decode("utf-8", "replace")


def launch_case_unit(case_id: str, staging_root: Path, mode: str) -> dict[str, Any]:
    cmd, unit = child_systemd_command(case_id, staging_root, mode)
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
        client_timeout = False
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["systemctl", "--user", "stop", unit], check=False, capture_output=True, text=True, timeout=30)
        exit_code = 124
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        client_timeout = True
    client_output = _parse_systemd_run_start(stdout + "\n" + stderr)
    if client_output and client_output.get("unit_name") != unit:
        readback_error = f"systemd-run started unexpected unit {client_output['unit_name']!r}; expected {unit!r}"
        terminal_properties = {}
    elif not client_output:
        readback_error = "systemd-run output omitted the started unit name"
        terminal_properties = {}
    elif exit_code != 0 or client_timeout:
        readback_error = "systemd-run did not successfully start the case unit"
        terminal_properties = {}
    else:
        terminal_properties, readback_error = _read_terminal_unit_properties(unit, CASE_RUNTIME_S + 120)
    receipt_path = staging_root / f"{case_id}.systemd-receipt.json"
    runtime_receipt = load_dispatch_record(receipt_path) if receipt_path.is_file() else None
    if client_output and not client_output.get("invocation_id"):
        terminal_invocation = terminal_properties.get("InvocationID")
        if terminal_invocation:
            client_output["invocation_id"] = terminal_invocation
            client_output["invocation_id_source"] = "retained_terminal_properties"
    if client_output and not client_output.get("invocation_id"):
        readback_error = readback_error or "systemd-run and retained properties omitted the unit invocation ID"
    journal = _read_invocation_journal(client_output["invocation_id"]) if client_output else None
    child_stdout = _read_tail(staging_root / f"{case_id}.stdout.log")
    child_stderr = _read_tail(staging_root / f"{case_id}.stderr.log")
    try:
        stop_result = subprocess.run(
            ["systemctl", "--user", "stop", unit],
            check=False, capture_output=True, text=True, timeout=10,
        )
        stop_exit_code, stop_stderr = stop_result.returncode, stop_result.stderr.strip()
    except subprocess.TimeoutExpired as exc:
        stop_exit_code, stop_stderr = 124, str(exc)
    return {
        "case_id": case_id,
        "unit_name": unit,
        "command": cmd,
        "exit_code": exit_code,
        "client_timeout": client_timeout,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
        "child_stdout": child_stdout,
        "child_stderr": child_stderr,
        "systemd_client_start": client_output,
        "terminal_properties": terminal_properties,
        "runtime_receipt": runtime_receipt,
        "invocation_journal": journal,
        "systemd_readback_error": readback_error,
        "systemd_stop_exit_code": stop_exit_code,
        "systemd_stop_stderr": stop_stderr,
    }


def _run_case_child(case_id: str, staging_root: Path) -> int:
    if os.environ.get("MALCENS_DISPATCH_CHILD") != "1" or not os.environ.get("INVOCATION_ID"):
        raise DispatchError("--case-child is an internal systemd case entry")
    context = load_legacy_context()
    protocol = load_dispatch_record(PROTOCOL_PATH)
    if protocol.get("dispatcher_sha256") != sha256_file(__file__):
        raise DispatchError("dispatcher changed after --prepare")
    if protocol.get("artifact_common_sha256") != sha256_file(artifact_common.__file__):
        raise DispatchError("shared artifact source changed after --prepare")
    try:
        staging_root.resolve(strict=True).relative_to(STAGING_PARENT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DispatchError("case staging root must be under /data/incoming/staging/malecns-gpu/mechanical-dispatch-v1") from exc
    if not staging_root.is_dir():
        raise DispatchError("case staging root does not exist")
    runtime_attestation = attest_current_unit(cpu_cores=CASE_CPU_CORES, memory_bytes=CASE_MEMORY_BYTES, runtime_s=CASE_RUNTIME_S)
    atomic_json(staging_root / f"{case_id}.systemd-receipt.json", {
        "case_id": case_id,
        "captured_utc": utc_now_iso(),
        "invocation_id": runtime_attestation["properties"]["InvocationID"],
        "unit_name": runtime_attestation["unit_name"],
        "properties_at_runtime": runtime_attestation["properties"],
        "cgroup_at_runtime": runtime_attestation["cgroup"],
    })
    _ArtifactInput, _ArtifactReference, Store, _config_hash = _storage_api()
    store = Store(OBJECT_STORE_ROOT)
    run_legacy_case(context, protocol, store, case_id, staging_root)
    return 0


def _register_partial_objects(store: Any, case_dir: Path) -> list[dict[str, Any]]:
    """Keep only actual partial-file H1s from a failed normal capture."""
    refs = []
    if not case_dir.is_dir():
        return refs
    for path in sorted(case_dir.iterdir()):
        if not path.is_file():
            continue
        role = REQUIRED_MEMBER_MANIFEST.get(path.name)
        if role is None:
            continue
        h1 = sha256_file(path)
        object_path = store.object_path(h1)
        if object_path.exists():
            store.resolve_reference({"H1": h1, "original_name": path.name, "artifact_class": role, "size_bytes": path.stat().st_size})
            action = "already_present"
        else:
            ref = store.put_file(path, original_name=path.name, artifact_class=role, locator=f"failed-capture:{case_dir.name}/{path.name}")
            action = "new_object" if ref.h1 == h1 else "changed_during_store"
            if ref.h1 != h1:
                raise DispatchError(f"partial output changed while storing {path.name}")
        refs.append({"path": path.name, "H1": h1, "role": role, "store_action": action})
    return refs


def _case_config_hash(context: Mapping[str, Any], case: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _ArtifactInput, _ArtifactReference, _Store, compute_hash = _storage_api()
    config = physics_config(context, case, policy, sha256_file(__file__))
    return config, compute_hash(config)


def execute() -> dict[str, Any]:
    protocol, context = load_prepared_protocol()
    _ArtifactInput, _ArtifactReference, Store, _compute_hash = _storage_api()
    store = Store(OBJECT_STORE_ROOT)
    coordinator = attest_current_unit(cpu_cores=PARENT_CPU_CORES, memory_bytes=PARENT_MEMORY_BYTES, runtime_s=PARENT_RUNTIME_S)
    lock_fd = acquire_global_lock()
    queue_lock = None
    try:
        queue_lock, queue_attestation = context["legacy"].acquire_terminal_fixed_pose_queue_lock()
        cases = []
        run_records: dict[str, dict[str, Any]] = {}
        expected_p_trace = None
        parent_created = not STAGING_PARENT.exists()
        STAGING_PARENT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dispatch-execute-", dir=STAGING_PARENT) as temp_name:
            staging_root = Path(temp_name)
            for case in context["cases"]:
                case_id = str(case["case_id"])
                config, config_hash = _case_config_hash(context, case, CANDIDATE_POLICY)
                record, reuse_record = find_reusable_record(store, case_id=case_id, config_hash=config_hash)
                if record is not None:
                    run_path = store.resolve_reference(record.files["run-record.json"], verify_integrity=True)
                    run_record = load_dispatch_record(run_path)
                    if run_record.get("case_id") != case_id or run_record.get("status") != "complete":
                        raise DispatchError(f"CAS run record is not complete for {case_id}")
                    action = "referenced_from_cas"
                    unit_receipt = None
                    payload_hash = record.case_payload_hash
                else:
                    unit_receipt = launch_case_unit(case_id, staging_root, "execute")
                    case_dir = staging_root / case_id
                    unit_ok = _case_unit_ok(unit_receipt)
                    if not unit_ok:
                        failed = load_dispatch_record(case_dir / "run-record.json") if (case_dir / "run-record.json").is_file() else {}
                        partial_refs = _register_partial_objects(store, case_dir)
                        cases.append({
                            "case_id": case_id, "action": "capture_failed", "cache_decision": reuse_record,
                            "run_record_status": failed.get("status", "missing"),
                            "failure": _failure_summary(case_dir, failed),
                            "partial_objects": partial_refs,
                            "systemd": unit_receipt,
                        })
                        break
                    record = store_complete_case(store, case_dir, case, config, config_hash)
                    run_record = load_dispatch_record(case_dir / "run-record.json")
                    action = "captured_registered_in_cas"
                    payload_hash = record.case_payload_hash
                    shutil.rmtree(case_dir)
                if expected_p_trace is None:
                    expected_p_trace = run_record.get("analysis_summary", {}).get("constant_P_trace_sha256")
                elif run_record.get("analysis_summary", {}).get("constant_P_trace_sha256") != expected_p_trace:
                    raise DispatchError(f"constant P-trace changed across cache/capture cases: {case_id}")
                run_records[case_id] = run_record
                cases.append({
                    "case_id": case_id, "action": action, "cache_decision": reuse_record,
                    "case_payload_hash": payload_hash,
                    "physics_config_hash": config_hash,
                    "case_unit": unit_receipt,
                })
        report = {
            "schema_version": "malecns-mechanical-storage-execution-report-v1",
            "created_utc": utc_now_iso(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "status": "complete" if (
                len(cases) == len(context["cases"])
                and all(row["action"] in {"referenced_from_cas", "captured_registered_in_cas"} for row in cases)
            ) else "failed_stopped",
            "coordinator_systemd": coordinator,
            "observed_cgroup_topology": CGROUP_TOPOLOGY_PROBE,
            "queue_serialization_attestation": queue_attestation,
            "case_count": len(cases),
            "cases_completed": sum(row.get("action") in {"referenced_from_cas", "captured_registered_in_cas"} for row in cases),
            "cases": cases,
        }
        report_path = ARTIFACT_ROOT / f"execution-report-{int(time.time())}.json"
        atomic_json(report_path, report)
        report["report_path"] = str(report_path)
        return report
    finally:
        if queue_lock is not None:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)
            queue_lock.close()
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        if 'parent_created' in locals() and parent_created:
            try:
                STAGING_PARENT.rmdir()
            except OSError:
                pass


def policy_check() -> dict[str, Any]:
    protocol, context = load_prepared_protocol()
    _ArtifactInput, _ArtifactReference, Store, _compute_hash = _storage_api()
    store = Store(OBJECT_STORE_ROOT)
    coordinator = attest_current_unit(cpu_cores=PARENT_CPU_CORES, memory_bytes=PARENT_MEMORY_BYTES, runtime_s=PARENT_RUNTIME_S)
    lock_fd = acquire_global_lock()
    queue_lock = None
    cases: list[dict[str, Any]] = []
    case_by_id = {str(row["case_id"]): row for row in context["cases"]}
    try:
        queue_lock, queue_attestation = context["legacy"].acquire_terminal_fixed_pose_queue_lock()
        baseline_records, _cohort = _historical_cases(context)
        parent_created = not STAGING_PARENT.exists()
        STAGING_PARENT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dispatch-policy-check-", dir=STAGING_PARENT) as temp_name:
            staging_root = Path(temp_name)
            for case in context["cases"]:
                case_id = str(case["case_id"])
                unit_receipt = launch_case_unit(case_id, staging_root, "policy-check")
                case_dir = staging_root / case_id
                if not _case_unit_ok(unit_receipt):
                    failed = load_dispatch_record(case_dir / "run-record.json") if (case_dir / "run-record.json").is_file() else {}
                    cases.append({
                        "case_id": case_id, "status": "capture_failed", "h2_match": False,
                        "run_record_status": failed.get("status", "missing"),
                        "failure": _failure_summary(case_dir, failed), "systemd": unit_receipt,
                    })
                    break
                validate_case_directory(case_dir)
                old_case_dir = Path(protocol["historical_bundle_locator"]) / "runs" / case_id
                result = policy_case_decision(store, old_case_dir, case_dir, case_id=case_id)
                result["systemd"] = unit_receipt
                result["candidate_cgroup_peak_bytes"] = load_dispatch_record(case_dir / "run-record.json").get("cgroup_memory_peak_after_bytes")
                result["historical_cgroup_peak_bytes"] = baseline_records[case_id].get("cgroup_memory_peak_after_bytes")
                cases.append(result)
                shutil.rmtree(case_dir)
        checked = sum(1 for row in cases if row.get("h2_match") is True or row.get("status") == "content_mismatch_unresolved")
        mismatched = sum(1 for row in cases if row.get("h2_match") is False and row.get("status") != "capture_failed")
        encoding = sum(1 for row in cases if row.get("status") == "encoding_difference")
        uncovered = [str(row["case_id"]) for row in context["cases"] if str(row["case_id"]) not in {r["case_id"] for r in cases if r.get("status") != "capture_failed"}]
        if mismatched:
            scope = "unresolved"
        elif checked == len(context["cases"]) and checked > 0:
            scope = "full_cohort"
        else:
            scope = "partial"
        report = {
            "schema_version": "malecns-mechanical-policy-check-report-v1",
            "created_utc": utc_now_iso(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "policy_inertness_scope": scope,
            "status": "complete" if checked == len(context["cases"]) else "partial",
            "cases_checked": checked,
            "cases_matched": sum(1 for row in cases if row.get("h2_match") is True),
            "cases_mismatched": mismatched,
            "encoding_differences": encoding,
            "uncovered_cases": uncovered,
            "comparison_object": "H2/object_content_hash as provided by measurement_content_hash.MECHANICAL_CASE_SCHEMA",
            "policy_trial_scope": protocol["policy_trial"]["scope"],
            "candidate_policy": CANDIDATE_POLICY,
            "coordinator_systemd": coordinator,
            "observed_cgroup_topology": CGROUP_TOPOLOGY_PROBE,
            "queue_serialization_attestation": queue_attestation,
            "cases": cases,
        }
        report_path = ARTIFACT_ROOT / f"policy-check-report-{int(time.time())}.json"
        atomic_json(report_path, report)
        report["report_path"] = str(report_path)
        return report
    finally:
        if queue_lock is not None:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)
            queue_lock.close()
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        if 'parent_created' in locals() and parent_created:
            try:
                STAGING_PARENT.rmdir()
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--policy-check", action="store_true")
    modes.add_argument("--case-child", metavar="CASE_ID")
    parser.add_argument("--staging-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            result = prepare()
        elif args.case_child:
            if args.staging_root is None:
                raise DispatchError("internal --case-child requires --staging-root")
            return _run_case_child(args.case_child, args.staging_root)
        elif args.execute or args.policy_check:
            mode = "execute" if args.execute else "policy-check"
            if not os.environ.get("INVOCATION_ID"):
                command = coordinator_systemd_command(mode)
                result = subprocess.run(command, check=False)
                return int(result.returncode)
            if args.execute:
                result = execute()
            else:
                result = policy_check()
        else:
            raise DispatchError("no mode selected")
    except (DispatchError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    if result.get("status") == "failed_stopped" or result.get("policy_inertness_scope") == "partial":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
