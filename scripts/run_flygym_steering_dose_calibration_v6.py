#!/usr/bin/env python3
"""Execute the frozen 15-case CPU FlyGym AP dose calibration (revision 6).

The runner is deliberately pinned to its revision-6 bundle and accepts no dose,
seed, duration, output, or resource-limit overrides. It records runtime/module
provenance and the constructor's raw MJCF, then gates on the complete saved
MuJoCo model bytes from the actual session model before controller creation.
It reuses the existing V4.1/AP controller and free-body session, changing only
the externally supplied AP command on each control tick. It is an engineered
position-servo/CPG experiment, not an NN-driven or biological steering experiment.

Preparation and execution are separate: this file only runs body simulations
when invoked with ``--execute`` and a separately reviewed revision-6 bundle.
The coordinator stays lightweight and launches one sequential child process
per case so FlyGym trace/contact memory is released before the next case.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import subprocess
import tempfile
import time
from typing import Any, Iterator


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
BUNDLE_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/mechanical-steering-dose-calibration-v1/revision-6"
)
RUN_ROOT = BUNDLE_ROOT / "runs"
PROTOCOL_PATH = BUNDLE_ROOT / "protocol.json"
SOURCE_FREEZE_PATH = BUNDLE_ROOT / "source_freeze.json"
FIXED_POSE_QUEUE_BUNDLE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/spatial-fixed-pose-calibration-v1-revision-2"
)
FIXED_POSE_QUEUE_STATUS_PATH = FIXED_POSE_QUEUE_BUNDLE / "queue-status.json"
FIXED_POSE_QUEUE_LOCK_PATH = FIXED_POSE_QUEUE_BUNDLE / "queue.lock"
CASE_TIMEOUT_S = 600
BATCH_TIMEOUT_S = 9000
CPU_LIMIT_CORES = 2.0
MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
EXPECTED_PAIRED_SCREEN_RAW_XML_SHA256 = "bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57"
EXPECTED_COMPILED_MJB_SHA256 = "6d7b1f8b9a7558cd5367be9e12e1a866adca3d98e6312d431b62d27ad747b2f3"
EXPECTED_COMPILED_MJB_SIZE_BYTES = 23230821
KNOWN_REVISION_3_RAW_XML_SHA256 = "976e72a119b25b591e8b88aac63c2a5f06913013ec787330ae727f0f8be432fd"
EXPECTED_HISTORICAL_QPOS_SHA256 = "d292b37df0f9a83265423da2ce53b0b140d8ef6a49d6909e464b805453e75b99"
EXPECTED_HISTORICAL_QVEL_SHA256 = "1a0295f4bf5986c5f74eca9153a6a4cb10b073a01a76ba4a457fd862c78966a4"
EXPECTED_HISTORICAL_ARRAY_ORDER_SHA256 = "0c8cdbb7006f774f42a19f068312605879a4ea7ca73e3db77e21e4903db14b0b"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


class RunnerError(RuntimeError):
    """Raised for a frozen-protocol, resource, or capture-integrity violation."""


class CaseTimeout(TimeoutError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(value)
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"expected JSON object at {path}")
    return value


_PROVENANCE_MODULES = (
    "numpy", "scipy", "mujoco", "flygym", "flygym.compose", "flygym.simulation",
    "flygym_demo", "flygym_demo.complex_terrain", "flygym_demo.complex_terrain.common",
    "flygym_demo.complex_terrain.preprogrammed", "flygym_demo.complex_terrain.cpg_controller",
    "run_flygym_reference_walk", "flygym_legacy_rest_compat",
    "flygym_ap_endpoint_gesture_prototype", "flygym_phase_speed_gait_repaired_v1",
    "flygym_phase_speed_gait_v4_1", "flygym_phase_speed_gait_v3",
    "flygym_phase_speed_gait_v2", "flygym_phase_speed_gait",
)
_PROVENANCE_ENV = (
    "PYTHONPATH", "PYTHONHASHSEED", "MUJOCO_GL", "PYOPENGL_PLATFORM",
    "LIBGL_ALWAYS_SOFTWARE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "FLYGYM_ASSET_CACHE_DIR", "MPLCONFIGDIR",
)


def runtime_provenance(walk_config: Any, run_id: str) -> dict[str, Any]:
    loaded = {}
    for name in _PROVENANCE_MODULES:
        module = sys.modules.get(name)
        if module is None:
            loaded[name] = None
            continue
        spec = getattr(module, "__spec__", None)
        loaded[name] = {
            "file": getattr(module, "__file__", None),
            "spec_origin": getattr(spec, "origin", None),
        }
    versions = {}
    for name in ("flygym", "mujoco", "numpy", "scipy"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    flygym = sys.modules.get("flygym")
    assets_root = getattr(flygym, "assets_dir", None) if flygym is not None else None
    return {
        "sys_executable": sys.executable,
        "python": sys.version,
        "sys_path": list(sys.path),
        "environment": {key: os.environ.get(key) for key in _PROVENANCE_ENV},
        "package_versions": versions,
        "loaded_module_origins": loaded,
        "assets_root": None if assets_root is None else str(assets_root),
        "assets_root_resolved": None if assets_root is None else str(Path(assets_root).resolve()),
        "walk_config": asdict(walk_config),
        "walk_config_sha256": canonical_sha256(asdict(walk_config)),
        "run_id": run_id,
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _decode_mountinfo(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), value)


def _parse_cpu_list(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    cpus: set[int] = set()
    try:
        for piece in value.split(","):
            bounds = piece.split("-", 1)
            start = int(bounds[0])
            end = int(bounds[1]) if len(bounds) == 2 else start
            if end < start:
                return None
            cpus.update(range(start, end + 1))
    except ValueError:
        return None
    return len(cpus) or None


def _cgroup_v2_dir() -> tuple[Path, str]:
    membership = None
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            membership = PurePosixPath(fields[2])
            break
    if membership is None:
        raise RunnerError("could not find unified cgroup-v2 membership in /proc/self/cgroup")

    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        left_right = line.split(" - ", 1)
        if len(left_right) != 2:
            continue
        before = left_right[0].split()
        after = left_right[1].split()
        if len(before) < 5 or not after or after[0] != "cgroup2":
            continue
        mount_root = PurePosixPath(_decode_mountinfo(before[3]))
        mount_point = Path(_decode_mountinfo(before[4]))
        try:
            relative = membership.relative_to(mount_root)
        except ValueError:
            continue
        candidate = mount_point.joinpath(*relative.parts)
        if candidate.is_dir():
            return candidate, str(membership)
    raise RunnerError("could not map the process cgroup-v2 path to a mounted cgroup directory")


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return None


def cgroup_attestation() -> dict[str, Any]:
    current, membership = _cgroup_v2_dir()
    ancestors: list[Path] = []
    cursor = current
    while True:
        ancestors.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent

    cpu_records = []
    finite_cpu_caps: list[float] = []
    cpuset_caps: list[int] = []
    memory_records = []
    finite_memory_caps: list[int] = []
    for directory in ancestors:
        cpu_entry: dict[str, Any] = {"directory": str(directory)}
        cpu_raw = _read_optional(directory / "cpu.max")
        if cpu_raw:
            parts = cpu_raw.split()
            if len(parts) == 2:
                quota_text, period_text = parts
                quota = None if quota_text == "max" else int(quota_text)
                period = int(period_text)
                if quota is not None and period > 0:
                    finite_cpu_caps.append(quota / period)
                cpu_entry["cpu.max"] = cpu_raw
                cpu_entry["quota_cores"] = None if quota is None else quota / period
        cpuset_raw = _read_optional(directory / "cpuset.cpus.effective")
        if cpuset_raw:
            cpuset_count = _parse_cpu_list(cpuset_raw)
            if cpuset_count is not None:
                cpuset_caps.append(cpuset_count)
            cpu_entry["cpuset.cpus.effective"] = cpuset_raw
        if len(cpu_entry) > 1:
            cpu_records.append(cpu_entry)

        memory_raw = _read_optional(directory / "memory.max")
        if memory_raw:
            memory_value = None if memory_raw == "max" else int(memory_raw)
            if memory_value is not None:
                finite_memory_caps.append(memory_value)
            memory_records.append({
                "directory": str(directory),
                "memory.max": memory_raw,
            })

    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = None
    cpu_candidates = finite_cpu_caps + [float(value) for value in cpuset_caps]
    if affinity_count is not None:
        cpu_candidates.append(float(affinity_count))
    if not cpu_candidates:
        raise RunnerError("no finite cgroup CPU quota, cpuset, or process affinity was observable")
    effective_cpu_cap = min(cpu_candidates)
    if not finite_memory_caps:
        raise RunnerError("no finite memory.max was observable in the cgroup hierarchy")
    effective_memory_cap = min(finite_memory_caps)
    memory_peak_path = current / "memory.peak"
    memory_peak_raw = _read_optional(memory_peak_path)
    if memory_peak_raw is None:
        raise RunnerError(f"cgroup memory.peak is unavailable: {memory_peak_path}")
    memory_peak = int(memory_peak_raw)

    result = {
        "cgroup_version": 2,
        "membership_path": membership,
        "current_cgroup_directory": str(current),
        "cpu_limit_sources": cpu_records,
        "memory_limit_sources": memory_records,
        "process_affinity_cpu_count": affinity_count,
        "effective_cpu_cap_cores": effective_cpu_cap,
        "effective_memory_cap_bytes": effective_memory_cap,
        "memory_peak_path": str(memory_peak_path),
        "memory_peak_bytes": memory_peak,
        "memory_peak_semantics": "cgroup high-water value; cumulative until the cgroup is reset or removed",
        "attested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if effective_cpu_cap > CPU_LIMIT_CORES + 1e-9:
        raise RunnerError(
            f"effective CPU cap {effective_cpu_cap:g} exceeds {CPU_LIMIT_CORES:g} cores"
        )
    if effective_memory_cap > MEMORY_LIMIT_BYTES:
        raise RunnerError(
            f"effective memory cap {effective_memory_cap} exceeds {MEMORY_LIMIT_BYTES} bytes"
        )
    return result


def verify_frozen_bundle() -> tuple[dict[str, Any], dict[str, Any], str]:
    review_path = BUNDLE_ROOT / "preparation-review.json"
    if not PROTOCOL_PATH.is_file() or not SOURCE_FREEZE_PATH.is_file() or not review_path.is_file():
        raise RunnerError(f"revision-6 preparation bundle is missing under {BUNDLE_ROOT}")
    protocol = read_json(PROTOCOL_PATH)
    freeze = read_json(SOURCE_FREEZE_PATH)
    review = read_json(review_path)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    source_freeze_sha = sha256_file(SOURCE_FREEZE_PATH)
    if freeze.get("protocol_sha256") != protocol_sha:
        raise RunnerError("protocol hash differs from revision-6 source_freeze.json")
    if freeze.get("status") != "preparation_frozen_execution_entrypoint_ready_not_run":
        raise RunnerError("revision-6 source freeze has an unexpected status")
    if freeze.get("revision") != 6:
        raise RunnerError("revision-6 source freeze revision number is unexpected")
    if review.get("status") != "prepared_not_executed":
        raise RunnerError("revision-6 preparation review is not in prepared_not_executed status")
    if review.get("source_freeze_sha256") != source_freeze_sha:
        raise RunnerError("preparation review does not identify the current source freeze")
    runner_hash = freeze.get("source_files", {}).get(str(Path(__file__).resolve()), {}).get("sha256")
    if not runner_hash or review.get("runner_sha256") != runner_hash:
        raise RunnerError("preparation review does not identify the frozen revision-6 runner")
    if protocol.get("entrypoint_source_sha256") != runner_hash:
        raise RunnerError("protocol does not identify the frozen revision-6 execution runner")
    if protocol.get("revision") != 6:
        raise RunnerError("revision-6 protocol is required by this execution runner")
    model_gate = protocol.get("execution", {}).get("model_identity_gate", {})
    if model_gate.get("expected_sha256") != EXPECTED_COMPILED_MJB_SHA256:
        raise RunnerError("protocol complete-MJB identity hash differs from the runner's strict gate")
    if model_gate.get("expected_size_bytes") != EXPECTED_COMPILED_MJB_SIZE_BYTES:
        raise RunnerError("protocol complete-MJB identity size differs from the runner's strict gate")
    if "raw XML equality is provenance only" not in model_gate.get("raw_xml", ""):
        raise RunnerError("protocol must state that raw XML equality is provenance only")

    problems: list[str] = []
    for raw_path, pinned in freeze.get("source_files", {}).items():
        path = Path(raw_path)
        if not path.is_file():
            problems.append(f"missing source file {path}")
        elif sha256_file(path) != pinned.get("sha256"):
            problems.append(f"changed source file {path}")
    for logical_name, pinned in freeze.get("asset_manifest", {}).get("files", {}).items():
        path = Path(pinned["physical_path"])
        if not path.is_file():
            problems.append(f"missing asset {logical_name}: {path}")
        elif sha256_file(path) != pinned.get("sha256"):
            problems.append(f"changed asset {logical_name}: {path}")
    for filename, pinned in freeze.get("copied_inputs", {}).items():
        path = Path(pinned["snapshot_path"])
        if not path.is_file():
            problems.append(f"missing frozen input {filename}: {path}")
        elif sha256_file(path) != pinned.get("sha256"):
            problems.append(f"changed frozen input {filename}: {path}")
    for filename, pinned in freeze.get("bundle_files", {}).items():
        path = BUNDLE_ROOT / filename
        if not path.is_file() or sha256_file(path) != pinned.get("sha256"):
            problems.append(f"missing or changed bundle file {path}")
    native = freeze.get("native_model_reference", {})
    if native.get("expected_mjb_sha256") != EXPECTED_COMPILED_MJB_SHA256:
        problems.append("frozen native reference SHA differs from the hard-coded identity gate")
    if native.get("expected_mjb_size_bytes") != EXPECTED_COMPILED_MJB_SIZE_BYTES:
        problems.append("frozen native reference size differs from the hard-coded identity gate")
    evidence = native.get("evidence", {})
    native_capture_path = BUNDLE_ROOT / evidence.get("capture_snapshot", "")
    native_comparison_path = BUNDLE_ROOT / evidence.get("comparison_snapshot", "")
    native_terminal_path = BUNDLE_ROOT / evidence.get("terminal_snapshot", "")
    native_manifest_path = BUNDLE_ROOT / evidence.get("manifest_snapshot", "")
    try:
        native_capture = read_json(native_capture_path)
        native_comparison = read_json(native_comparison_path)
        native_terminal = read_json(native_terminal_path)
        native_manifest = read_json(native_manifest_path)
    except (OSError, ValueError, RunnerError):
        native_capture = native_comparison = native_terminal = native_manifest = {}
        problems.append("frozen native-reference evidence JSON is missing or invalid")
    if native_capture.get("status") != "complete" or native_terminal.get("native_capture_status") != "complete":
        problems.append("native-reference capture is not complete")
    if native_comparison != native_capture.get("comparison"):
        problems.append("standalone native comparison differs from the embedded capture comparison")
    if native_terminal.get("actual_exec_exit_code") != 0 or native_terminal.get("systemd_result") != "success":
        problems.append("native-reference capture did not exit successfully")
    expected_versions = native_capture.get("package_versions", {})
    if expected_versions != {"flygym": "2.1.0", "mujoco": "3.9.0", "numpy": "2.5.3"}:
        problems.append("native-reference package-version pins are unexpected")
    else:
        for package, expected_version in expected_versions.items():
            try:
                installed_version = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                installed_version = None
            if installed_version != expected_version:
                problems.append(f"installed {package} version {installed_version!r} differs from native-reference {expected_version!r}")
    expected_python_realpath = native_capture.get("expected_python_realpath")
    if expected_python_realpath and str(Path(sys.executable).resolve()) != expected_python_realpath:
        problems.append("Python interpreter realpath differs from the native-reference capture")
    upstream_preconstruction = native_capture.get("upstream_source_preconstruction", {})
    if (upstream_preconstruction.get("all_match") is not True
            or upstream_preconstruction.get("commit_match") is not True
            or upstream_preconstruction.get("actual_commit") != "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"):
        problems.append("native-reference FlyGym source/commit proof is not complete")
    capture_comparison = native_capture.get("comparison", {})
    counters = capture_comparison.get("constructor_counters", {})
    if (counters.get("session_time_s") != 0.0
            or counters.get("data_time_s") != 0.0
            or counters.get("control_samples") != 0
            or counters.get("constructor_native_samples") != 0
            or counters.get("simulation_curr_step") != 0):
        problems.append("native-reference capture was not a zero-time, zero-sample constructor capture")
    captured_model = capture_comparison.get("captured_model", {})
    if (captured_model.get("actual_session_mjb_sha256") != EXPECTED_COMPILED_MJB_SHA256
            or captured_model.get("actual_session_mjb_size_bytes") != EXPECTED_COMPILED_MJB_SIZE_BYTES
            or captured_model.get("native_source") != "mujoco.mj_saveModel(session.model, ...), where session.model is session.sim.mj_model"):
        problems.append("native-reference evidence does not identify the pinned actual session.model MJB")
    if captured_model.get("matches_direct_xml_mjb_sha256") is not False:
        problems.append("native-reference evidence unexpectedly identifies the XML-roundtrip MJB")
    historical_bridge = capture_comparison.get("historical_bridge", {})
    if (historical_bridge.get("qpos_bytes_equal_to_paired_screen_initial") is not True
            or historical_bridge.get("qvel_bytes_equal_to_paired_screen_initial") is not True
            or historical_bridge.get("array_order_equal_to_frozen_paired_screen") is not True
            or capture_comparison.get("effective_model_options_match_paired_screen") is not True
            or historical_bridge.get("paired_screen_qpos_sha256") != EXPECTED_HISTORICAL_QPOS_SHA256
            or historical_bridge.get("paired_screen_qvel_sha256") != EXPECTED_HISTORICAL_QVEL_SHA256
            or historical_bridge.get("array_order_sha256") != EXPECTED_HISTORICAL_ARRAY_ORDER_SHA256):
        problems.append("native-reference evidence does not preserve the exact historical state/options/order bridge")
    manifest_mjb = native_manifest.get("files", {}).get("actual_session_model.mjb", {})
    if (manifest_mjb.get("sha256") != EXPECTED_COMPILED_MJB_SHA256
            or manifest_mjb.get("size_bytes") != EXPECTED_COMPILED_MJB_SIZE_BYTES):
        problems.append("native-reference manifest MJB pin differs from the hard-coded gate")
    if (native_capture.get("baseline_promotion") is not False
            or capture_comparison.get("baseline_promotion") is not False
            or native_manifest.get("gate_modified") is not False
            or native_manifest.get("historical_failed_records_modified") is not False):
        problems.append("native capture must remain evidence without changing gates or old records")
    mjb_snapshot = BUNDLE_ROOT / evidence.get("mjb_snapshot", "")
    if not mjb_snapshot.is_file() or mjb_snapshot.stat().st_size != EXPECTED_COMPILED_MJB_SIZE_BYTES or sha256_file(mjb_snapshot) != EXPECTED_COMPILED_MJB_SHA256:
        problems.append("frozen native MJB bytes differ from the strict runtime identity gate")
    if problems:
        raise RunnerError("frozen revision-6 provenance check failed: " + "; ".join(problems[:20]))

    worker_policy = protocol.get("execution", {}).get("case_worker_policy", {})
    if worker_policy != {
        "mode": "one_sequential_subprocess_per_case",
        "max_concurrent_case_workers": 1,
        "wait_for_worker_exit_before_next_case": True,
        "workers_inherit_service_cgroup_limits": True,
        "cpu_limit_cores": CPU_LIMIT_CORES,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "case_timeout_s": CASE_TIMEOUT_S,
        "batch_timeout_s": BATCH_TIMEOUT_S,
        "session_controller_cycle_broken_before_worker_exit": True,
        "trace_and_contact_capture_format_unchanged": True,
        "controller_and_physics_design_unchanged": True,
        "coordinator_retained_data": "small JSON run records, same-seed pairing fields, and analysis summaries only",
        "trace_saving": "unchanged per-case compressed trajectory/contact outputs; controller telemetry remains line-streamed",
    }:
        raise RunnerError("case-worker isolation policy differs from the reviewed revision-6 policy")

    historical = freeze.get("historical_revision_5", {})
    historical_files = historical.get("evidence_snapshots", {})
    try:
        old_terminal = read_json(BUNDLE_ROOT / historical_files["root_terminal"])
        old_cohort = read_json(BUNDLE_ROOT / historical_files["cohort_manifest"])
        old_complete_case = read_json(BUNDLE_ROOT / historical_files["completed_case_run_record"])
        old_partial_case = read_json(BUNDLE_ROOT / historical_files["partial_case_run_record"])
    except (KeyError, OSError, ValueError, RunnerError):
        old_terminal = old_cohort = old_complete_case = old_partial_case = {}
        problems.append("historical revision-5 OOM evidence snapshots are missing or invalid")
    if (old_terminal.get("actual_exec_exit_code") != 1
            or old_terminal.get("systemd_result") != "oom-kill"
            or old_terminal.get("partial_case_last_logged_control_tick") != 6500
            or old_terminal.get("cohort_manifest_status_stale_after_unhandled_sigkill") != "running"
            or old_cohort.get("status") != "running"
            or old_cohort.get("completed_case_count") != 1
            or old_complete_case.get("status") != "complete"
            or old_partial_case.get("status") != "running"):
        problems.append("revision-5 OOM and partial-output history was altered or is not the reviewed event")
    if problems:
        raise RunnerError("frozen revision-6 historical evidence check failed: " + "; ".join(problems[:20]))

    cases = protocol.get("execution", {}).get("ordered_cases")
    if not isinstance(cases, list) or len(cases) != 15:
        raise RunnerError("protocol must pin exactly 15 ordered cases")
    if protocol.get("execution", {}).get("hard_case_count_cap") != 15:
        raise RunnerError("protocol hard case-count cap is not 15")
    if protocol.get("execution", {}).get("duration_s") != 7.0:
        raise RunnerError("protocol duration is not pinned to 7.0 s")
    if protocol.get("execution", {}).get("power_P_left_right") != [0.258, 0.258]:
        raise RunnerError("protocol P schedule differs from the frozen design")
    execution = protocol["execution"]
    if execution.get("common_prefix_s") != 1.0 or execution.get("ramp_duration_s") != 0.25:
        raise RunnerError("shared prefix or ramp differs from the frozen design")
    if execution.get("case_timeout_s") != CASE_TIMEOUT_S or execution.get("batch_timeout_s") != BATCH_TIMEOUT_S:
        raise RunnerError("timeout policy differs from the runner hard limits")
    if execution.get("cpu_limit_cores") != CPU_LIMIT_CORES or execution.get("memory_limit_bytes") != MEMORY_LIMIT_BYTES:
        raise RunnerError("resource limits differ from the runner hard limits")
    serialization = execution.get("serialization_gate", {})
    if serialization.get("queue_status_path") != str(FIXED_POSE_QUEUE_STATUS_PATH):
        raise RunnerError("fixed-pose queue status path differs from the runner gate")
    if serialization.get("queue_lock_path") != str(FIXED_POSE_QUEUE_LOCK_PATH):
        raise RunnerError("fixed-pose queue lock path differs from the runner gate")
    if serialization.get("accepted_terminal_states") != ["complete", "failed_stopped"]:
        raise RunnerError("fixed-pose queue terminal-state policy differs from the runner gate")
    return protocol, freeze, protocol_sha


def acquire_terminal_fixed_pose_queue_lock() -> tuple[Any, dict[str, Any]]:
    """Require the fixed-pose queue to be terminal and prevent it restarting."""
    if not FIXED_POSE_QUEUE_LOCK_PATH.is_file():
        raise RunnerError(f"fixed-pose queue lock is missing: {FIXED_POSE_QUEUE_LOCK_PATH}")
    lock_handle = FIXED_POSE_QUEUE_LOCK_PATH.open("r", encoding="ascii")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("fixed-pose NN queue still holds its exclusive lock") from exc
        if not FIXED_POSE_QUEUE_STATUS_PATH.is_file():
            raise RunnerError(f"fixed-pose queue status is missing: {FIXED_POSE_QUEUE_STATUS_PATH}")
        status = read_json(FIXED_POSE_QUEUE_STATUS_PATH)
        queue_state = status.get("status")
        current_marker = status.get("current")
        if queue_state not in ("complete", "failed_stopped"):
            raise RunnerError(
                "fixed-pose NN queue is not terminal; mechanical dose run must wait"
            )
        if queue_state == "complete" and current_marker is not None:
            raise RunnerError("fixed-pose queue claims complete but still has a current case")
        attestation = {
            "queue_bundle": str(FIXED_POSE_QUEUE_BUNDLE),
            "queue_status_path": str(FIXED_POSE_QUEUE_STATUS_PATH),
            "queue_status_sha256": sha256_file(FIXED_POSE_QUEUE_STATUS_PATH),
            "queue_status": status.get("status"),
            "queue_current_case_marker": current_marker,
            "queue_current_marker_is_stale_allowed": queue_state == "failed_stopped" and current_marker is not None,
            "queue_completed_conditions": status.get("completed"),
            "queue_lock_path": str(FIXED_POSE_QUEUE_LOCK_PATH),
            "queue_lock_mode": "shared nonblocking flock held for the full 15-case cohort",
            "serialized_after_fixed_pose_queue": True,
        }
        return lock_handle, attestation
    except BaseException:
        lock_handle.close()
        raise


def _dose_at_time(target_d: float, time_s: float, prefix_s: float, ramp_s: float) -> tuple[float, float]:
    if target_d == 0.0 or time_s <= prefix_s:
        return 0.0, 0.0
    u = min(1.0, max(0.0, (time_s - prefix_s) / ramp_s))
    smooth = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    return target_d * smooth, u


def _array_interpolate(np: Any, times: Any, values: Any, target: float) -> Any:
    if values.ndim == 1:
        return float(np.interp(target, times, values))
    return np.asarray([np.interp(target, times, values[:, index]) for index in range(values.shape[1])])


def _case_summary(np: Any, session: Any, power: float, target_d: float,
                  analysis_window_s: tuple[float, float], prefix_s: float,
                  initial_qpos_sha: str, initial_qvel_sha: str,
                  initial_phase_sha: str, control_d_values: list[float],
                  array_sha256: Any) -> dict[str, Any]:
    count = int(session._native_samples)
    arrays = session.all_arrays
    times = np.asarray(arrays["time_s"][:count], dtype=np.float64)
    yaw = np.unwrap(np.asarray(arrays["thorax_heading_yaw_rad"][:count], dtype=np.float64))
    yaw_rate = np.gradient(yaw, times) if count > 1 else np.zeros_like(yaw)
    body_x = np.asarray(arrays["body_xpos_mm"][:count, session.thorax_index, :], dtype=np.float64)
    h0 = np.asarray(arrays["thorax_heading_xy"][0], dtype=np.float64)
    hnorm = float(np.linalg.norm(h0))
    if hnorm <= 0.0:
        raise RunnerError("initial thorax heading vector is zero")
    h0 /= hnorm
    lateral_axis = np.asarray([-h0[1], h0[0]], dtype=np.float64)
    t0, t1 = analysis_window_s
    pos0 = _array_interpolate(np, times, body_x[:, :2], t0)
    pos1 = _array_interpolate(np, times, body_x[:, :2], t1)
    displacement_xy = pos1 - pos0
    yaw0 = _array_interpolate(np, times, yaw, t0)
    yaw1 = _array_interpolate(np, times, yaw, t1)
    fz = np.asarray(arrays["foot_contact_fz_up_native"][:count], dtype=np.float64)
    contact_found = np.asarray(arrays["foot_contact_found"][:count], dtype=bool)
    mask = (times >= t0) & (times <= t1)
    if int(mask.sum()) < 2:
        raise RunnerError("captured native trace does not cover the analysis window")
    prefix_mask = times <= prefix_s + 1e-12
    if not bool(prefix_mask.any()):
        raise RunnerError("captured trace lacks the shared prefix")
    phase = np.asarray(session.cpg_phase_by_sample[:count], dtype=np.float64)
    phase_prefix = phase[prefix_mask]
    qfrc = np.asarray(arrays["qfrc_applied_native"][:count], dtype=np.float64)
    body_xfrc = np.asarray(arrays["body_xfrc_applied_native"][:count], dtype=np.float64)
    root_xfrc = np.asarray(session.root_external_wrench_by_sample[:count], dtype=np.float64)
    root_qpos_writes = np.asarray(session.root_qpos_write_count_by_sample[:count], dtype=np.int64)
    root_qvel_writes = np.asarray(session.root_qvel_write_count_by_sample[:count], dtype=np.int64)
    authority = {
        "root_qpos_write_count_max": int(root_qpos_writes.max(initial=0)),
        "root_qvel_write_count_max": int(root_qvel_writes.max(initial=0)),
        "root_external_wrench_peak_native": float(np.linalg.norm(root_xfrc, axis=1).max(initial=0.0)),
        "body_external_wrench_peak_native": float(np.linalg.norm(body_xfrc, axis=2).max(initial=0.0)),
        "generalized_applied_force_peak_native": float(np.abs(qfrc).max(initial=0.0)),
    }
    summary = {
        "classification": "engineered CPU mechanics dose calibration; not NN-driven or biological evidence",
        "analysis_window_s": [t0, t1],
        "power_P_left_right": [power, power],
        "target_d": float(target_d),
        "nominal_endpoint_request_mm": 0.05 * abs(float(target_d)),
        "native_sample_count": count,
        "control_tick_count": int(session._control_samples),
        "physics_step_count": count - 1,
        "thorax_heading_yaw_change_deg": math.degrees(float(yaw1 - yaw0)),
        "yaw_rate_mean_deg_s": math.degrees(float(np.mean(yaw_rate[mask]))),
        "yaw_rate_sd_deg_s": math.degrees(float(np.std(yaw_rate[mask]))),
        "yaw_rate_peak_abs_deg_s": math.degrees(float(np.max(np.abs(yaw_rate[mask])))),
        "initial_heading_xy": h0.tolist(),
        "thorax_displacement_xy_mm_analysis": displacement_xy.tolist(),
        "thorax_forward_displacement_mm_analysis": float(np.dot(displacement_xy, h0)),
        "thorax_lateral_displacement_mm_analysis": float(np.dot(displacement_xy, lateral_axis)),
        "mean_forward_velocity_mm_s_analysis": float(np.dot(displacement_xy, h0) / (t1 - t0)),
        "per_leg_contact_found_fraction_analysis": {
            leg: float(np.mean(contact_found[mask, i])) for i, leg in enumerate(LEGS)
        },
        "per_leg_positive_fz_fraction_analysis": {
            leg: float(np.mean(fz[mask, i] > 0.0)) for i, leg in enumerate(LEGS)
        },
        "mean_contact_legs_analysis": float(np.mean(np.sum(contact_found[mask], axis=1))),
        "samples_with_at_least_two_contacting_legs_fraction_analysis": float(
            np.mean(np.sum(contact_found[mask], axis=1) >= 2)
        ),
        "ground_contact_record_count": int(sum(int(value) for value in session.contact_is_ground)),
        "thorax_ground_contact_samples": int(np.count_nonzero(session.thorax_ground_contact_by_sample[:count][mask])),
        "nonleg_ground_contact_samples": int(np.count_nonzero(session.nonleg_ground_contact_by_sample[:count][mask])),
        "initial_qpos_sha256": initial_qpos_sha,
        "initial_qvel_sha256": initial_qvel_sha,
        "initial_cpg_phase_sha256": initial_phase_sha,
        "cpg_phase_prefix_sha256_through_1s": array_sha256(phase_prefix),
        "cpg_phase_prefix_sample_count": int(phase_prefix.shape[0]),
        "control_d_trace_sha256": array_sha256(np.asarray(control_d_values, dtype=np.float64)),
        "constant_P_trace_sha256": array_sha256(
            np.asarray(session.descending_drive_by_sample[:count], dtype=np.float64)
        ),
        "authority": authority,
        "authority_integrity_pass": all(value == 0 for value in authority.values()),
    }
    return summary


def _validate_and_prepare_pairing(case: dict[str, Any], expected: dict[str, Any] | None,
                                  config_hash: str, initial_state: dict[str, Any]) -> None:
    if expected is None:
        if float(case["d"]) != 0.0:
            raise RunnerError("first case for each seed must be the d=0 pairing reference")
        return
    for field in ("qpos_sha256", "qvel_sha256", "initial_cpg_phase_sha256"):
        if initial_state[field] != expected["initial_state"][field]:
            raise RunnerError(f"same-seed pairing mismatch before stepping: {field}")
    if config_hash != expected["base_controller_config_sha256"]:
        raise RunnerError("same-seed base controller configuration changed")


@contextmanager
def _case_deadline(seconds: int) -> Iterator[None]:
    if not hasattr(signal, "SIGALRM"):
        raise RunnerError("per-case timeout requires SIGALRM support")
    old_handler = signal.getsignal(signal.SIGALRM)

    def on_alarm(_signum: int, _frame: Any) -> None:
        raise CaseTimeout(f"case exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _case_artifacts(case_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(case_dir.iterdir())
        if path.is_file() and path.name != "run-record.json"
    }


def _run_case(*, case: dict[str, Any], protocol: dict[str, Any], freeze: dict[str, Any],
              expected_pair: dict[str, Any] | None, expected_p_trace_sha: str | None,
              serial_queue_attestation: dict[str, Any], out_root: Path,
              source_freeze_sha: str, walk_values: dict[str, Any], controller_values: dict[str, Any],
              helper: Any, np: Any, baseline_model_signature: dict[str, Any],
              baseline_initial_state_hashes: dict[str, str],
              case_timeout_s: int) -> dict[str, Any]:
    from run_flygym_reference_walk import FlyGymReferenceSession, WalkConfig
    from flygym_ap_endpoint_gesture_prototype import APEndpointGestureConfig
    from flygym_phase_speed_gait_repaired_v1 import (
        PhaseSpeedGaitController,
        controller_config_for_differential_ap,
        verify_parent_sources,
    )

    execution = protocol["execution"]
    seed = int(case["seed"])
    target_d = float(case["d"])
    case_id = str(case["case_id"])
    case_dir = out_root / case_id
    case_dir.mkdir(parents=False, exist_ok=False)
    duration_s = float(execution["duration_s"])
    power = float(execution["power_P_left_right"][0])
    prefix_s = float(execution["common_prefix_s"])
    ramp_s = float(execution["ramp_duration_s"])
    window = tuple(float(v) for v in execution["analysis_window_s"])
    control_dt = float(walk_values["control_timestep_s"])
    native_dt = float(walk_values["native_timestep_s"])
    expected_ticks = round(duration_s / control_dt)
    expected_samples = round(duration_s / native_dt) + 1
    if not math.isclose(expected_ticks * control_dt, duration_s, rel_tol=0, abs_tol=1e-9):
        raise RunnerError("duration is not an integer multiple of control_dt")
    if not math.isclose(expected_samples - 1, duration_s / native_dt, rel_tol=0, abs_tol=1e-9):
        raise RunnerError("duration is not an integer multiple of native_dt")

    body_values = dict(walk_values)
    body_values["duration_s"] = duration_s
    body_values["seed"] = seed
    for key in ("spawn_position_mm", "spawn_quaternion_wxyz", "position_actuator_forcerange_native", "camera_resolution_hw"):
        if key in body_values:
            body_values[key] = tuple(body_values[key])
    walk_config = WalkConfig(**body_values)
    if float(walk_config.cpg_frequency_hz) != 7.4074074074074066:
        raise RunnerError("V4.1 configured CPG frequency differs from the frozen template")
    base_config = APEndpointGestureConfig(**controller_values)
    base_config_hash = canonical_sha256(asdict(base_config))
    start_mono = time.monotonic()
    cgroup_before = cgroup_attestation()
    source_hashes = verify_parent_sources()
    run_id = f"steering-dose-{case_id}-T{duration_s:g}s"
    runtime_before_session = runtime_provenance(walk_config, run_id)
    initial_record: dict[str, Any] = {}
    run_record: dict[str, Any] = {
        "schema_version": "flygym-steering-dose-calibration-run-v6",
        "status": "running",
        "case_process_isolation": {
            "mode": "one_sequential_subprocess_per_case",
            "worker_pid": os.getpid(),
            "coordinator_pid": os.getppid(),
            "max_concurrent_case_workers": 1,
            "worker_process_exits_after_case_capture": True,
            "session_controller_cycle_break_on_exit": True,
        },
        "case_id": case_id,
        "run_id": run_id,
        "condition": {
            "P_left_right": [power, power],
            "target_d": target_d,
            "dna02_command_target": abs(target_d),
            "dng13_command_target": abs(target_d),
            "soma_side": "left" if target_d >= 0 else "right",
            "seed": seed,
            "duration_s": duration_s,
        },
        "classification": "CPU engineered position-servo/CPG mechanics; direct P/d; no NN forward or visual input",
        "input_source": "protocol-defined external P and smooth-ramped AP d schedule",
        "stimulus_label_sent_to_controller": False,
        "neural_forward_called": False,
        "gpu_used": False,
        "source_freeze_sha256": source_freeze_sha,
        "fixed_pose_queue_terminal_attestation": serial_queue_attestation,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "controller_parent_source_hashes": source_hashes,
        "adaptation_source": {
            "source_path": str(SCRIPTS / "run_flygym_ap_body_yaw_calibration_v1.py"),
            "source_sha256": freeze["source_files"][str(SCRIPTS / "run_flygym_ap_body_yaw_calibration_v1.py")]["sha256"],
            "adapted_pattern": "existing _run_condition session construction/save/telemetry flow; existing _extract_control_telemetry reused",
            "new_control_operation": "replace controller.gesture_config with factory-built immutable config at each 1 ms tick; no source controller edit",
        },
        "body_config_sha256": canonical_sha256(asdict(walk_config)),
        "body_config": asdict(walk_config),
        "base_controller_config_sha256": base_config_hash,
        "initialization_provenance_preconstruction": runtime_before_session,
        "compiled_xml_capture_status": "pending_session_constructor",
        "root_state_writes_after_initialization": 0,
        "root_external_wrench_applied": False,
        "body_external_wrench_applied": False,
        "generalized_applied_force_applied": False,
        "cgroup_memory_peak_before_bytes": cgroup_before["memory_peak_bytes"],
        "cgroup_path": cgroup_before["current_cgroup_directory"],
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_monotonic_s": start_mono,
        "expected_control_tick_count": expected_ticks,
        "expected_native_sample_count": expected_samples,
        "timeout_s": int(case_timeout_s),
    }
    run_record_path = case_dir / "run-record.json"
    atomic_json(run_record_path, run_record)
    telemetry_path = case_dir / "controller_telemetry.jsonl"
    session = None
    raw_saved = False
    try:
        with _case_deadline(case_timeout_s):
            session = FlyGymReferenceSession(
                duration_s=duration_s,
                seed=seed,
                render_video=False,
                config=walk_config,
                run_id=run_record["run_id"],
            )
            compiled_xml = session.compiled_mjcf_xml.encode("utf-8")
            compiled_xml_sha = hashlib.sha256(compiled_xml).hexdigest()
            compiled_xml_path = case_dir / "compiled_model.xml"
            atomic_bytes(compiled_xml_path, compiled_xml)
            compiled_mjb_path = case_dir / "compiled_model.mjb"
            if session.model is not session.sim.mj_model or session.data is not session.sim.mj_data:
                raise RunnerError("session model/data do not directly reference this FlyGym Simulation")
            session.mj.mj_saveModel(session.model, str(compiled_mjb_path), None)
            if not compiled_mjb_path.is_file():
                raise RunnerError("MuJoCo did not write the complete compiled-model MJB")
            compiled_mjb_sha = sha256_file(compiled_mjb_path)
            compiled_mjb_size = compiled_mjb_path.stat().st_size
            session_time_after_constructor = float(session.time_s)
            constructor_control_samples = int(session._control_samples)
            constructor_native_samples = int(session._native_samples)
            constructor_root_qpos_writes = int(session._root_qpos_write_count)
            constructor_root_qvel_writes = int(session._root_qvel_write_count)
            session_zero_time_and_samples = (
                math.isclose(session_time_after_constructor, 0.0, rel_tol=0.0, abs_tol=1e-12)
                and constructor_control_samples == 0
                and constructor_native_samples == 0
                and constructor_root_qpos_writes == 0
                and constructor_root_qvel_writes == 0
            )
            prior_model_options = read_json(BUNDLE_ROOT / "frozen_inputs/prior_model_options.json")
            native_reference_model_options = read_json(
                BUNDLE_ROOT / "frozen_inputs/native_model_reference/effective_model_options.json"
            )
            runtime_model_options = dict(session.effective_model_options)
            model_options_match_paired_screen = runtime_model_options == prior_model_options
            model_options_match_native_reference = runtime_model_options == native_reference_model_options
            model_options_sha256 = canonical_sha256(runtime_model_options)
            runtime_array_order = session._array_order_names()
            prior_array_order = read_json(BUNDLE_ROOT / "frozen_inputs/prior_array_order.json")
            native_reference_array_order = read_json(
                BUNDLE_ROOT / "frozen_inputs/native_model_reference/runtime_array_order.json"
            )
            array_order_match_paired_screen = runtime_array_order == prior_array_order
            array_order_match_native_reference = runtime_array_order == native_reference_array_order
            array_order_sha256 = canonical_sha256(runtime_array_order)
            initial_qpos = np.ascontiguousarray(session.data.qpos, dtype=np.float64).copy()
            initial_qvel = np.ascontiguousarray(session.data.qvel, dtype=np.float64).copy()
            initial_qpos_sha = helper.array_sha256(initial_qpos)
            initial_qvel_sha = helper.array_sha256(initial_qvel)
            historical_initial_state_match = (
                initial_qpos_sha == baseline_initial_state_hashes.get("qpos_sha256")
                and initial_qvel_sha == baseline_initial_state_hashes.get("qvel_sha256")
            )
            same_seed_initial_state_match = (
                expected_pair is None
                or (
                    initial_qpos_sha == expected_pair["initial_state"]["qpos_sha256"]
                    and initial_qvel_sha == expected_pair["initial_state"]["qvel_sha256"]
                )
            )
            constructor_time_zero = math.isclose(
                float(session.data.time), 0.0, rel_tol=0.0, abs_tol=1e-12
            )
            reference_raw_xml_sha = str(baseline_model_signature["compiled_model_sha256"])
            raw_xml_matches_reference = compiled_xml_sha == reference_raw_xml_sha
            raw_xml_matches_known_r3 = compiled_xml_sha == KNOWN_REVISION_3_RAW_XML_SHA256
            initialization_capture = {
                "captured_immediately_after_constructor": True,
                "actual_session_model_saved_directly": True,
                "session_model_is_simulation_model": True,
                "session_data_is_simulation_data": True,
                "compiled_model_sha256": session.compiled_mjcf_sha256,
                "raw_xml_sha256": compiled_xml_sha,
                "compiled_xml_sha256": compiled_xml_sha,
                "compiled_xml_path": compiled_xml_path.name,
                "compiled_xml_bytes": len(compiled_xml),
                "compiled_xml_sha_matches_session": compiled_xml_sha == session.compiled_mjcf_sha256,
                "paired_screen_reference_raw_xml_sha256": reference_raw_xml_sha,
                "raw_xml_matches_paired_screen_reference": raw_xml_matches_reference,
                "known_revision_3_raw_xml_sha256": KNOWN_REVISION_3_RAW_XML_SHA256,
                "raw_xml_matches_known_revision_3_constructor_output": raw_xml_matches_known_r3,
                "raw_xml_is_provenance_only": True,
                "raw_xml_difference_attribution": "Raw MJSpec serialization is retained as provenance only; the independent native reference capture shows it differs from the XML-roundtrip MJB in binary identity, so no raw-text hash substitutes for the native MJB gate.",
                "compiled_model_mjb_path": compiled_mjb_path.name,
                "compiled_model_mjb_sha256": compiled_mjb_sha,
                "compiled_model_mjb_size_bytes": compiled_mjb_size,
                "expected_complete_saved_mjb_sha256": EXPECTED_COMPILED_MJB_SHA256,
                "expected_complete_saved_mjb_size_bytes": EXPECTED_COMPILED_MJB_SIZE_BYTES,
                "compiled_model_mjb_exact_hash_match": compiled_mjb_sha == EXPECTED_COMPILED_MJB_SHA256,
                "compiled_model_mjb_exact_size_match": compiled_mjb_size == EXPECTED_COMPILED_MJB_SIZE_BYTES,
                "compiled_model_identity_gate": "SHA-256 and size of the complete MJB saved from the actual session.model; exact equality to the independent native constructor capture required; no tolerance or hash allowlist",
                "session_zero_time_and_zero_samples_after_constructor": session_zero_time_and_samples,
                "constructor_data_time_is_zero": constructor_time_zero,
                "constructor_root_qpos_write_count": constructor_root_qpos_writes,
                "constructor_root_qvel_write_count": constructor_root_qvel_writes,
                "effective_model_options": runtime_model_options,
                "effective_model_options_sha256": model_options_sha256,
                "expected_paired_screen_model_options_sha256": canonical_sha256(prior_model_options),
                "effective_model_options_match_paired_screen": model_options_match_paired_screen,
                "effective_model_options_match_native_reference": model_options_match_native_reference,
                "runtime_array_order_sha256": array_order_sha256,
                "expected_paired_screen_array_order_sha256": canonical_sha256(prior_array_order),
                "array_order_matches_paired_screen": array_order_match_paired_screen,
                "array_order_matches_native_reference": array_order_match_native_reference,
                "initial_qpos_sha256": initial_qpos_sha,
                "initial_qvel_sha256": initial_qvel_sha,
                "historical_qpos_sha256": baseline_initial_state_hashes.get("qpos_sha256"),
                "historical_qvel_sha256": baseline_initial_state_hashes.get("qvel_sha256"),
                "initial_state_matches_historical_paired_screen": historical_initial_state_match,
                "initial_state_matches_same_seed_reference": same_seed_initial_state_match,
                "upstream_source_hashes": session.upstream_source_hashes,
                "runtime_after_constructor": runtime_provenance(walk_config, run_id),
                "session_time_s": session_time_after_constructor,
                "control_samples": constructor_control_samples,
                "constructor_native_samples": constructor_native_samples,
            }
            run_record.update({
                "body_session_started": True,
                "compiled_model_sha256": session.compiled_mjcf_sha256,
                "raw_xml_sha256": compiled_xml_sha,
                "compiled_model_options": session.effective_model_options,
                "compiled_xml_capture_status": "saved",
                "compiled_model_mjb_capture_status": "saved",
                "compiled_model_identity_gate_status": "passed" if compiled_mjb_sha == EXPECTED_COMPILED_MJB_SHA256 else "failed",
                "model_initialization_provenance": initialization_capture,
            })
            atomic_json(run_record_path, run_record)
            if not initialization_capture["compiled_xml_sha_matches_session"]:
                raise RunnerError("saved raw XML SHA differs from session.compiled_mjcf_sha256")
            if not session_zero_time_and_samples:
                raise RunnerError("new session did not remain at zero time with zero samples after construction")
            if not constructor_time_zero:
                raise RunnerError("MuJoCo data time is not zero immediately after constructor")
            if constructor_root_qpos_writes or constructor_root_qvel_writes:
                raise RunnerError("session constructor changed root qpos/qvel through a tracked runner write")
            if compiled_mjb_sha != EXPECTED_COMPILED_MJB_SHA256:
                raise RunnerError(
                    "complete compiled-model MJB SHA differs from the frozen reference: "
                    f"{compiled_mjb_sha} != {EXPECTED_COMPILED_MJB_SHA256}"
                )
            if compiled_mjb_size != EXPECTED_COMPILED_MJB_SIZE_BYTES:
                raise RunnerError(
                    "complete compiled-model MJB size differs from the frozen reference: "
                    f"{compiled_mjb_size} != {EXPECTED_COMPILED_MJB_SIZE_BYTES}"
                )
            if not model_options_match_paired_screen or not model_options_match_native_reference:
                raise RunnerError("effective model options differ from the frozen paired-screen options")
            if not array_order_match_paired_screen or not array_order_match_native_reference:
                raise RunnerError("runtime model array order differs from the frozen paired-screen order")
            if not historical_initial_state_match:
                raise RunnerError("constructor qpos/qvel differ from the exact historical paired-screen state")
            if not same_seed_initial_state_match:
                raise RunnerError("constructor qpos/qvel differ from the same-seed reference case")
            if not math.isclose(session.time_s, 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise RunnerError("new free-ground session did not start at time zero")
            zero_config = controller_config_for_differential_ap(0.0, base_config=base_config)
            if float(zero_config.active_threshold) != helper.P_DEADBAND:
                raise RunnerError("AP active threshold differs from shared rest contract")
            controller = PhaseSpeedGaitController(session, zero_config)
            session.controller = controller
            initial_phase = np.asarray(controller.base.cpg_network.curr_phases, dtype=np.float64).copy()
            initial_phase_sha = helper.array_sha256(initial_phase)
            initial_record = {
                "qpos_sha256": initial_qpos_sha,
                "qvel_sha256": initial_qvel_sha,
                "initial_cpg_phase_sha256": initial_phase_sha,
                "time_s": float(session.time_s),
                "qpos_write_count": int(session._root_qpos_write_count),
                "qvel_write_count": int(session._root_qvel_write_count),
            }
            _validate_and_prepare_pairing(case, expected_pair, base_config_hash, initial_record)
            metadata = controller.controller_metadata()
            metadata.update({
                "classification": "engineered V4.1/AP position-servo; not neural-force actuation",
                "input_source": "fixed external P plus per-tick protocol AP command",
                "P_left_right": [power, power],
                "AP_command_schedule": execution["d_schedule"],
                "dynamic_command_update": "assign a newly factory-built frozen APEndpointGestureConfig to controller.gesture_config before each control tick",
                "neural_worker": False,
                "visual_input": False,
                "root_state_write_after_initialization": False,
                "external_body_wrench": False,
            })
            run_record.update({
                "initial_state": initial_record,
                "controller_metadata": metadata,
                "body_session_started": True,
                "compiled_model_sha256": session.compiled_mjcf_sha256,
                "effective_model_options": session.effective_model_options,
                "body_config": asdict(walk_config),
                "controller_config_at_initialization": asdict(zero_config),
                "shared_rest_contract": {
                    "p_deadband": helper.P_DEADBAND,
                    "active_rule": "P > p_deadband",
                    "P_input_is_active": bool(power > helper.P_DEADBAND),
                },
            })
            atomic_json(run_record_path, run_record)

            d_control_values: list[float] = []
            with telemetry_path.open("x", encoding="utf-8", buffering=1) as stream:
                for tick in range(expected_ticks):
                    control_time = tick * control_dt
                    current_d, ramp_u = _dose_at_time(target_d, control_time, prefix_s, ramp_s)
                    selected_config = controller_config_for_differential_ap(
                        current_d, base_config=base_config
                    )
                    controller.gesture_config = selected_config
                    result = session.step_control_tick((power, power))
                    record = helper._extract_control_telemetry(
                        controller, tick, control_time, power, current_d
                    )
                    record.update({
                        "target_d": target_d,
                        "ramp_u": ramp_u,
                        "ramp_phase": "shared_prefix" if control_time <= prefix_s else (
                            "hold" if ramp_u >= 1.0 else "quintic_ramp"
                        ),
                        "body_time_after_tick_s": float(result["time_s"]),
                        "current_config_sha256": canonical_sha256(asdict(selected_config)),
                    })
                    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                    d_control_values.append(float(current_d))
                    if (tick + 1) % 500 == 0:
                        print(f"{case_id}: {tick + 1}/{expected_ticks} control ticks", flush=True)

            if session._control_samples != expected_ticks or session._native_samples != expected_samples:
                raise RunnerError(
                    f"unexpected sample counts: control={session._control_samples}, native={session._native_samples}"
                )
            output_manifest = session.save(
                case_dir,
                constant_drive_values_left_right=(power, power),
                controller_metadata=metadata,
            )
            raw_saved = True
            summary = _case_summary(
                np, session, power, target_d, window, prefix_s,
                initial_qpos_sha, initial_qvel_sha, initial_phase_sha,
                d_control_values, helper.array_sha256,
            )
            if not summary["authority_integrity_pass"]:
                raise RunnerError("authority audit found a root state write or nonzero applied force/wrench")
            if expected_pair is not None:
                pairing = {
                    "initial_qpos_match": initial_qpos_sha == expected_pair["initial_state"]["qpos_sha256"],
                    "initial_qvel_match": initial_qvel_sha == expected_pair["initial_state"]["qvel_sha256"],
                    "initial_cpg_phase_match": initial_phase_sha == expected_pair["initial_state"]["initial_cpg_phase_sha256"],
                    "cpg_phase_prefix_match": summary["cpg_phase_prefix_sha256_through_1s"] == expected_pair["analysis_summary"]["cpg_phase_prefix_sha256_through_1s"],
                    "base_controller_config_match": base_config_hash == expected_pair["base_controller_config_sha256"],
                    "body_config_match": run_record["body_config_sha256"] == expected_pair["body_config_sha256"],
                }
                if not all(pairing.values()):
                    raise RunnerError(f"post-capture same-seed prefix pairing check failed: {pairing}")
            else:
                pairing = {
                    "reference_case": True,
                    "initial_qpos_match": True,
                    "initial_qvel_match": True,
                    "initial_cpg_phase_match": True,
                    "cpg_phase_prefix_match": True,
                    "base_controller_config_match": True,
                    "body_config_match": True,
                }
            p_trace_match = (
                expected_p_trace_sha is None
                or summary["constant_P_trace_sha256"] == expected_p_trace_sha
            )
            pairing["constant_P_trace_match_across_cohort"] = p_trace_match
            if not p_trace_match:
                raise RunnerError("constant P drive trace changed between cohort cases")
            atomic_json(case_dir / "analysis-summary.json", summary)
            cgroup_after = cgroup_attestation()
            artifacts = _case_artifacts(case_dir)
            legacy_window_summary = run_record.pop("window_summary", None)
            initial_controller_config = run_record.pop("controller_config", None)
            run_record.update({
                "status": "complete",
                "raw_capture_status": "saved",
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "finished_monotonic_s": time.monotonic(),
                "elapsed_wall_s": time.monotonic() - start_mono,
                "control_tick_count": int(session._control_samples),
                "native_sample_count": int(session._native_samples),
                "physics_step_count": int(session._native_samples - 1),
                "analysis_summary": summary,
                "window_summary": summary,
                "legacy_helper_window_summary_0p5_3s": legacy_window_summary,
                "controller_config_at_initialization": initial_controller_config,
                "dose_schedule": execution["d_schedule"],
                "scheduled_d_target": target_d,
                "same_seed_pairing": pairing,
                "authority": summary["authority"],
                "capture_files": artifacts,
                "session_manifest_status": output_manifest.get("status"),
                "cgroup_effective_cpu_cap_cores": cgroup_after["effective_cpu_cap_cores"],
                "cgroup_effective_memory_cap_bytes": cgroup_after["effective_memory_cap_bytes"],
                "cgroup_memory_peak_after_bytes": cgroup_after["memory_peak_bytes"],
                "cgroup_memory_peak_before_bytes": cgroup_before["memory_peak_bytes"],
                "cgroup_memory_peak_is_batch_high_water": True,
            })
            atomic_json(run_record_path, run_record)
            run_record["run_record_sha256"] = sha256_file(run_record_path)
            return run_record
    except BaseException as exc:
        if session is not None and not raw_saved and session._native_samples > 0:
            try:
                session.save(
                    case_dir,
                    constant_drive_values_left_right=(power, power),
                    controller_metadata=run_record.get("controller_metadata", {}),
                )
                raw_saved = True
            except BaseException as save_exc:
                run_record["partial_capture_save_error"] = f"{type(save_exc).__name__}: {save_exc}"
        try:
            cgroup_after = cgroup_attestation()
        except BaseException as cgroup_exc:
            cgroup_after = {"error": f"{type(cgroup_exc).__name__}: {cgroup_exc}"}
        run_record.update({
            "status": "failed",
            "raw_capture_status": "saved" if raw_saved else (
                "partial_saved" if session is not None and session._native_samples > 0 else "not_saved"
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_monotonic_s": time.monotonic(),
            "elapsed_wall_s": time.monotonic() - start_mono,
            "cgroup_after": cgroup_after,
            "partial_artifacts": _case_artifacts(case_dir),
        })
        atomic_json(run_record_path, run_record)
        raise
    finally:
        if session is not None:
            try:
                session.close()
            finally:
                # The session owns its controller and the controller owns its
                # session. Break that cycle explicitly before this per-case
                # worker exits, then release local references to the large
                # preallocated traces and contact lists.
                try:
                    session.controller = None
                except BaseException:
                    pass
                if "controller" in locals():
                    del controller
                session = None
                gc.collect()


def _set_simulation_environment() -> None:
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


def run_case_worker(input_path: Path) -> int:
    """Run exactly one frozen case in its own short-lived Python process."""
    worker_input = read_json(input_path)
    protocol, freeze, protocol_sha = verify_frozen_bundle()
    source_freeze_sha = sha256_file(SOURCE_FREEZE_PATH)
    if worker_input.get("schema_version") != "flygym-steering-dose-case-worker-input-v6":
        raise RunnerError("case worker input schema is unexpected")
    if worker_input.get("protocol_sha256") != protocol_sha:
        raise RunnerError("case worker protocol pin differs from the frozen bundle")
    if worker_input.get("source_freeze_sha256") != source_freeze_sha:
        raise RunnerError("case worker source-freeze pin differs from the frozen bundle")

    case = worker_input.get("case")
    expected_cases = _expected_case_order(protocol)
    if case not in expected_cases:
        raise RunnerError("case worker was given an unpinned case")
    expected_pair = worker_input.get("expected_pair")
    expected_pair_hash = worker_input.get("expected_p_trace_sha")
    if expected_pair is not None and int(expected_pair.get("condition", {}).get("seed", -1)) != int(case["seed"]):
        raise RunnerError("case worker pair reference uses a different seed")
    if expected_pair is not None and float(case["d"]) == 0.0:
        raise RunnerError("the seed's d=0 case must start without a prior pair reference")

    _set_simulation_environment()
    sys.path.insert(0, str(SCRIPTS))
    import numpy as np
    import run_flygym_ap_body_yaw_calibration_v1 as helper

    input_root = BUNDLE_ROOT / "frozen_inputs"
    walk_effective = read_json(input_root / "v4_1_effective_config.json")
    controller_values = read_json(input_root / "v4_1_controller_config.json")["config"]
    walk_values = dict(walk_effective["walk_config"])
    prior_screen = read_json(BUNDLE_ROOT / "paired-screen-reference.json")["summary"]
    baseline_model_signature = prior_screen["matched_inputs"]["model_signatures"]["static_d0"]
    baseline_initial_state_hashes = prior_screen["matched_inputs"]["initial_state_hashes"]["static_d0"]
    if baseline_model_signature.get("compiled_model_sha256") != EXPECTED_PAIRED_SCREEN_RAW_XML_SHA256:
        raise RunnerError("pinned paired-screen raw-XML baseline hash changed")
    if baseline_initial_state_hashes != {
        "qpos_sha256": EXPECTED_HISTORICAL_QPOS_SHA256,
        "qvel_sha256": EXPECTED_HISTORICAL_QVEL_SHA256,
    }:
        raise RunnerError("pinned paired-screen initial qpos/qvel hashes changed")

    record = _run_case(
        case=case,
        protocol=protocol,
        freeze=freeze,
        expected_pair=expected_pair,
        expected_p_trace_sha=expected_pair_hash,
        serial_queue_attestation=worker_input["serial_queue_attestation"],
        out_root=RUN_ROOT,
        source_freeze_sha=source_freeze_sha,
        walk_values=walk_values,
        controller_values=controller_values,
        helper=helper,
        np=np,
        baseline_model_signature=baseline_model_signature,
        baseline_initial_state_hashes=baseline_initial_state_hashes,
        case_timeout_s=int(worker_input["case_timeout_s"]),
    )
    if record.get("status") != "complete":
        raise RunnerError("case worker returned without a completed run record")
    return 0


def _run_case_in_subprocess(*, case: dict[str, Any], expected_pair: dict[str, Any] | None,
                            expected_p_trace_sha: str | None,
                            serial_queue_attestation: dict[str, Any],
                            source_freeze_sha: str, timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = RUN_ROOT / case_id
    run_record_path = case_dir / "run-record.json"
    terminal_root = RUN_ROOT / "worker-terminals"
    terminal_root.mkdir(parents=True, exist_ok=True)
    terminal_path = terminal_root / f"{case_id}.json"
    if terminal_path.exists():
        raise RunnerError(f"refusing to overwrite existing case-worker terminal record: {terminal_path}")
    worker_payload = {
        "schema_version": "flygym-steering-dose-case-worker-input-v6",
        "case": case,
        "expected_pair": expected_pair,
        "expected_p_trace_sha": expected_p_trace_sha,
        "serial_queue_attestation": serial_queue_attestation,
        "source_freeze_sha256": source_freeze_sha,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "case_timeout_s": int(timeout_s),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="flygym-mechanical-v6-",
        suffix=".json", dir="/tmp", delete=False,
    ) as stream:
        worker_input_path = Path(stream.name)
        json.dump(worker_payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    environment = os.environ.copy()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started_mono = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    worker_pid: int | None = None
    exit_code = -1
    command: list[str] = []
    timed_out = False
    try:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case-worker-input",
            str(worker_input_path),
        ]
        process = subprocess.Popen(command, env=environment)
        worker_pid = process.pid
        # The inner signal alarm owns the 600-second simulation deadline. This
        # outer guard allows bounded cleanup and partial-output persistence.
        try:
            exit_code = process.wait(timeout=max(1, int(timeout_s)) + 45)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait()
    finally:
        try:
            worker_input_path.unlink()
        except FileNotFoundError:
            pass

    finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    terminal = {
        "schema_version": "flygym-steering-dose-case-worker-terminal-v6",
        "case_id": case_id,
        "status": "complete" if exit_code == 0 and not timed_out else "failed_stopped",
        "worker_pid": worker_pid,
        "coordinator_pid": os.getpid(),
        "worker_exit_code": exit_code,
        "outer_timeout_triggered": timed_out,
        "inner_case_timeout_s": int(timeout_s),
        "worker_started_utc": started_utc,
        "worker_finished_utc": finished_utc,
        "worker_elapsed_wall_s": time.monotonic() - started_mono,
        "worker_command": command,
        "service_cgroup_attestation_after_exit": cgroup_attestation(),
        "case_run_record_path": str(run_record_path),
        "case_run_record_existed": run_record_path.is_file(),
    }
    if run_record_path.is_file():
        worker_record = read_json(run_record_path)
        terminal["case_run_record_status_at_worker_exit"] = worker_record.get("status")
        if exit_code != 0 and worker_record.get("status") == "running":
            worker_record.update({
                "status": "failed_stopped",
                "error_type": "IsolatedWorkerProcessExit",
                "error": f"isolated case worker exited with code {exit_code}; partial outputs are retained",
                "worker_process_terminal": terminal,
                "finished_utc": finished_utc,
                "finished_monotonic_s": time.monotonic(),
                "partial_artifacts": _case_artifacts(case_dir),
                "cgroup_after_worker_exit": terminal["service_cgroup_attestation_after_exit"],
            })
            atomic_json(run_record_path, worker_record)
        terminal["case_run_record_sha256"] = sha256_file(run_record_path)
    terminal["terminal_record_path"] = str(terminal_path)
    atomic_json(terminal_path, terminal)
    terminal["terminal_record_sha256"] = sha256_file(terminal_path)

    if exit_code != 0 or timed_out:
        raise RunnerError(
            f"isolated worker failed for {case_id}: exit={exit_code}, outer_timeout={timed_out}; "
            f"terminal={terminal_path}"
        )
    if not run_record_path.is_file():
        raise RunnerError(f"isolated worker exited successfully without a run record: {case_id}")
    record = read_json(run_record_path)
    if record.get("status") != "complete":
        raise RunnerError(f"isolated worker did not complete case {case_id}: {record.get('status')}")
    record["run_record_sha256"] = sha256_file(run_record_path)
    record["worker_terminal"] = terminal
    return record


def _expected_case_order(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    cases = protocol["execution"]["ordered_cases"]
    d_values = [0.0, 0.2, -0.2, 0.4, -0.4]
    seeds = [1729, 1730, 1731]
    expected = []
    for seed in seeds:
        for d in d_values:
            sign = "zero" if d == 0 else ("plus" if d > 0 else "minus")
            magnitude = "0p00" if d == 0 else f"{abs(d):.2f}".replace(".", "p")
            expected.append({
                "case_id": f"seed{seed}_d{sign}_{magnitude}",
                "seed": seed,
                "d": d,
            })
    if cases != expected:
        raise RunnerError("ordered cases differ from the fixed seed/dose design")
    return expected


def run_cohort(*, execute_confirmed: bool) -> int:
    if not execute_confirmed:
        raise RunnerError("physics dependencies load only for explicit --execute")
    protocol, freeze, protocol_sha = verify_frozen_bundle()
    case_order = _expected_case_order(protocol)
    if RUN_ROOT.exists():
        raise RunnerError(f"refusing to overwrite an existing run directory: {RUN_ROOT}")
    # Resource and provenance gates run before loading the physics stack.
    queue_lock_handle, serial_queue_attestation = acquire_terminal_fixed_pose_queue_lock()
    cgroup = cgroup_attestation()
    cgroup_start_peak = cgroup["memory_peak_bytes"]

    _set_simulation_environment()
    prior_screen = read_json(BUNDLE_ROOT / "paired-screen-reference.json")["summary"]
    baseline_model_signature = prior_screen["matched_inputs"]["model_signatures"]["static_d0"]
    baseline_initial_state_hashes = prior_screen["matched_inputs"]["initial_state_hashes"]["static_d0"]
    if baseline_model_signature.get("compiled_model_sha256") != EXPECTED_PAIRED_SCREEN_RAW_XML_SHA256:
        raise RunnerError("pinned paired-screen raw-XML baseline hash changed")
    if baseline_initial_state_hashes != {
        "qpos_sha256": EXPECTED_HISTORICAL_QPOS_SHA256,
        "qvel_sha256": EXPECTED_HISTORICAL_QVEL_SHA256,
    }:
        raise RunnerError("pinned paired-screen initial qpos/qvel hashes changed")

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    cohort_path = RUN_ROOT / "cohort-manifest.json"
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cohort: dict[str, Any] = {
        "schema_version": "flygym-steering-dose-calibration-cohort-v6",
        "status": "running",
        "classification": "CPU engineered position-servo/CPG; direct P/d; no NN forward, vision, or biological pass threshold",
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "source_freeze_path": str(SOURCE_FREEZE_PATH),
        "source_freeze_sha256": sha256_file(SOURCE_FREEZE_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "hard_case_count_cap": 15,
        "case_count_planned": len(case_order),
        "case_order": [case["case_id"] for case in case_order],
        "started_utc": started_utc,
        "started_monotonic_s": started,
        "batch_timeout_s": BATCH_TIMEOUT_S,
        "cgroup_start_attestation": cgroup,
        "fixed_pose_queue_terminal_attestation": serial_queue_attestation,
        "cgroup_memory_peak_start_bytes": cgroup_start_peak,
        "cgroup_memory_peak_semantics": "actual service-cgroup cumulative high-water; each case simulation runs in one sequential child process that exits before the next case",
        "case_process_isolation": protocol["execution"]["case_worker_policy"],
        "cases": {},
    }
    atomic_json(cohort_path, cohort)
    expected_pair_by_seed: dict[int, dict[str, Any]] = {}
    completed_records: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(case_order, start=1):
            elapsed = time.monotonic() - started
            remaining = BATCH_TIMEOUT_S - elapsed
            if remaining < 1:
                raise CaseTimeout(f"cohort exceeded {BATCH_TIMEOUT_S} seconds before case {index}")
            per_case = expected_pair_by_seed.get(int(case["seed"]))
            try:
                record = _run_case_in_subprocess(
                    case=case,
                    expected_pair=per_case,
                    expected_p_trace_sha=cohort.get("constant_P_trace_sha256"),
                    serial_queue_attestation=serial_queue_attestation,
                    source_freeze_sha=cohort["source_freeze_sha256"],
                    timeout_s=min(CASE_TIMEOUT_S, int(remaining)),
                )
                if time.monotonic() - started > BATCH_TIMEOUT_S:
                    raise CaseTimeout(f"case {index} finished after the {BATCH_TIMEOUT_S} second cohort limit")
            except BaseException:
                case_path = RUN_ROOT / str(case["case_id"]) / "run-record.json"
                failure_ref = {
                    "status": "failed",
                    "case_id": case["case_id"],
                    "run_record_path": str(case_path),
                    "run_record_sha256": sha256_file(case_path) if case_path.is_file() else None,
                    "worker_terminal_path": str(RUN_ROOT / "worker-terminals" / f"{case['case_id']}.json"),
                    "worker_terminal_sha256": sha256_file(RUN_ROOT / "worker-terminals" / f"{case['case_id']}.json") if (RUN_ROOT / "worker-terminals" / f"{case['case_id']}.json").is_file() else None,
                }
                cohort["cases"][case["case_id"]] = failure_ref
                cohort["failed_case_index"] = index
                cohort["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                cohort["elapsed_wall_s"] = time.monotonic() - started
                cohort["cgroup_memory_peak_end_bytes"] = cgroup_attestation().get("memory_peak_bytes")
                cohort["status"] = "failed_stopped"
                atomic_json(cohort_path, cohort)
                raise
            completed_records.append(record)
            cohort["cases"][case["case_id"]] = {
                "status": record["status"],
                "run_record_path": str(RUN_ROOT / case["case_id"] / "run-record.json"),
                "run_record_sha256": record["run_record_sha256"],
                "worker_terminal_path": record["worker_terminal"]["terminal_record_path"],
                "worker_terminal_sha256": record["worker_terminal"]["terminal_record_sha256"],
                "worker_pid": record["worker_terminal"]["worker_pid"],
                "trajectory": record["capture_files"].get("trajectory.npz"),
                "contacts": record["capture_files"].get("contacts.npz"),
                "controller_telemetry": record["capture_files"].get("controller_telemetry.jsonl"),
                "yaw_change_deg": record["analysis_summary"]["thorax_heading_yaw_change_deg"],
                "forward_displacement_mm": record["analysis_summary"]["thorax_forward_displacement_mm_analysis"],
                "cgroup_memory_peak_bytes": record["cgroup_memory_peak_after_bytes"],
            }
            if per_case is None:
                expected_pair_by_seed[int(case["seed"])] = record
            current_p_trace_sha = record["analysis_summary"]["constant_P_trace_sha256"]
            cohort["constant_P_trace_sha256"] = current_p_trace_sha
            cohort["completed_case_count"] = len(completed_records)
            cohort["last_completed_case_index"] = index
            cohort["cgroup_memory_peak_last_bytes"] = record["cgroup_memory_peak_after_bytes"]
            atomic_json(cohort_path, cohort)
            print(f"completed {index}/15: {case['case_id']}", flush=True)

        cgroup_end = cgroup_attestation()
        cohort.update({
            "status": "complete",
            "completed_case_count": len(completed_records),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_wall_s": time.monotonic() - started,
            "cgroup_end_attestation": cgroup_end,
            "cgroup_memory_peak_end_bytes": cgroup_end["memory_peak_bytes"],
            "same_seed_pairing_status": "all initial qpos/qvel/phase and 1s phase-prefix checks passed",
            "biological_acceptance_status": "not assessed; no biological threshold registered",
        })
        atomic_json(cohort_path, cohort)
        (RUN_ROOT / "cohort-manifest.sha256").write_text(
            sha256_file(cohort_path) + "  cohort-manifest.json\n", encoding="ascii"
        )
        return 0
    except BaseException as exc:
        if cohort.get("status") == "running":
            try:
                cgroup_end = cgroup_attestation()
            except BaseException as cgroup_exc:
                cgroup_end = {"error": f"{type(cgroup_exc).__name__}: {cgroup_exc}"}
            cohort.update({
                "status": "failed_stopped",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "elapsed_wall_s": time.monotonic() - started,
                "cgroup_end_attestation": cgroup_end,
            })
            atomic_json(cohort_path, cohort)
        if cohort_path.is_file():
            (RUN_ROOT / "cohort-manifest.sha256").write_text(
                sha256_file(cohort_path) + "  cohort-manifest.json\n", encoding="ascii"
            )
        print(f"cohort stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        fcntl.flock(queue_lock_handle.fileno(), fcntl.LOCK_UN)
        queue_lock_handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true",
                       help="execute the pinned 15-case CPU cohort; no per-case overrides are accepted")
    mode.add_argument("--case-worker-input", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.case_worker_input is not None:
        try:
            return run_case_worker(args.case_worker_input)
        except BaseException as exc:
            print(f"case worker stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 2
    return run_cohort(execute_confirmed=bool(args.execute))


if __name__ == "__main__":
    raise SystemExit(main())
