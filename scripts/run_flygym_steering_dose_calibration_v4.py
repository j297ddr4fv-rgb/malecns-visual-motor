#!/usr/bin/env python3
"""Execute the frozen 15-case CPU FlyGym AP dose calibration (revision 4).

The runner is deliberately pinned to its revision-4 bundle and accepts no dose,
seed, duration, output, or resource-limit overrides. It records runtime/module
provenance and the constructor's raw MJCF, then gates on the complete saved
MuJoCo model bytes from the actual session model before controller creation.
It reuses the existing V4.1/AP controller and free-body session, changing only
the externally supplied AP command on each control tick. It is an engineered
position-servo/CPG experiment, not an NN-driven or biological steering experiment.

Preparation and execution are separate: this file only runs a body simulation
when invoked with ``--execute`` and a separately reviewed revision-4 bundle.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import time
from typing import Any, Iterator


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
BUNDLE_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/mechanical-steering-dose-calibration-v1/revision-4"
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
EXPECTED_COMPILED_MJB_SHA256 = "c21516b1229e4da7ac44a513b8ec1315d8a7cce6546668983473fc37b7c7e69f"
EXPECTED_COMPILED_MJB_SIZE_BYTES = 23230821
KNOWN_REVISION_3_RAW_XML_SHA256 = "976e72a119b25b591e8b88aac63c2a5f06913013ec787330ae727f0f8be432fd"
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
        raise RunnerError(f"revision-4 preparation bundle is missing under {BUNDLE_ROOT}")
    protocol = read_json(PROTOCOL_PATH)
    freeze = read_json(SOURCE_FREEZE_PATH)
    review = read_json(review_path)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    source_freeze_sha = sha256_file(SOURCE_FREEZE_PATH)
    if freeze.get("protocol_sha256") != protocol_sha:
        raise RunnerError("protocol hash differs from revision-4 source_freeze.json")
    if freeze.get("status") != "preparation_frozen_execution_entrypoint_ready_not_run":
        raise RunnerError("revision-4 source freeze has an unexpected status")
    if freeze.get("revision") != 4:
        raise RunnerError("revision-4 source freeze revision number is unexpected")
    if review.get("status") != "prepared_not_executed":
        raise RunnerError("revision-4 preparation review is not in prepared_not_executed status")
    if review.get("source_freeze_sha256") != source_freeze_sha:
        raise RunnerError("preparation review does not identify the current source freeze")
    runner_hash = freeze.get("source_files", {}).get(str(Path(__file__).resolve()), {}).get("sha256")
    if not runner_hash or review.get("runner_sha256") != runner_hash:
        raise RunnerError("preparation review does not identify the frozen revision-4 runner")
    if protocol.get("entrypoint_source_sha256") != runner_hash:
        raise RunnerError("protocol does not identify the frozen revision-4 execution runner")
    if protocol.get("revision") != 4:
        raise RunnerError("revision-4 protocol is required by this execution runner")
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
    forensic = freeze.get("compiled_identity_forensics", {})
    if forensic.get("expected_complete_saved_mjb_sha256") != EXPECTED_COMPILED_MJB_SHA256:
        problems.append("frozen compiled-model reference SHA differs from the hard-coded identity gate")
    if forensic.get("expected_complete_saved_mjb_size_bytes") != EXPECTED_COMPILED_MJB_SIZE_BYTES:
        problems.append("frozen compiled-model reference size differs from the hard-coded identity gate")
    if forensic.get("complete_saved_mjb_bytes_identical") is not True:
        problems.append("frozen forensics do not prove complete saved-MJB byte identity")
    if forensic.get("model_array_bytes_identical") is not True:
        problems.append("frozen forensics do not prove complete model-array byte identity")
    if forensic.get("model_array_count_compared") != 468:
        problems.append("frozen forensics do not cover the expected 468 model arrays")
    if problems:
        raise RunnerError("frozen revision-4 provenance check failed: " + "; ".join(problems[:20]))

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
        "schema_version": "flygym-steering-dose-calibration-run-v4",
        "status": "running",
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
            if session.model is not session.sim.mj_model:
                raise RunnerError("session.model is not the MuJoCo model compiled by this FlyGym Simulation")
            session.mj.mj_saveModel(session.model, str(compiled_mjb_path), None)
            if not compiled_mjb_path.is_file():
                raise RunnerError("MuJoCo did not write the complete compiled-model MJB")
            compiled_mjb_sha = sha256_file(compiled_mjb_path)
            compiled_mjb_size = compiled_mjb_path.stat().st_size
            session_time_after_constructor = float(session.time_s)
            constructor_control_samples = int(session._control_samples)
            constructor_native_samples = int(session._native_samples)
            session_zero_time_and_samples = (
                math.isclose(session_time_after_constructor, 0.0, rel_tol=0.0, abs_tol=1e-12)
                and constructor_control_samples == 0
                and constructor_native_samples == 0
            )
            reference_raw_xml_sha = str(baseline_model_signature["compiled_model_sha256"])
            raw_xml_matches_reference = compiled_xml_sha == reference_raw_xml_sha
            raw_xml_matches_known_r3 = compiled_xml_sha == KNOWN_REVISION_3_RAW_XML_SHA256
            initialization_capture = {
                "captured_immediately_after_constructor": True,
                "actual_session_model_saved_directly": True,
                "session_model_is_simulation_model": True,
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
                "raw_xml_difference_attribution": (
                    "The pinned revision-3 XML differs from the paired-screen XML only by which member of each same-file left/right mesh pair explicitly carries content_type='model/stl'; the pinned forensic run saved complete byte-identical MJBs and found all 468 model arrays byte-identical."
                    if raw_xml_matches_known_r3 and not raw_xml_matches_reference
                    else "No specific XML serialization difference is inferred from this runtime hash alone; the preserved XML and pinned forensic inputs are available for review."
                ),
                "compiled_model_mjb_path": compiled_mjb_path.name,
                "compiled_model_mjb_sha256": compiled_mjb_sha,
                "compiled_model_mjb_size_bytes": compiled_mjb_size,
                "expected_complete_saved_mjb_sha256": EXPECTED_COMPILED_MJB_SHA256,
                "expected_complete_saved_mjb_size_bytes": EXPECTED_COMPILED_MJB_SIZE_BYTES,
                "compiled_model_mjb_exact_hash_match": compiled_mjb_sha == EXPECTED_COMPILED_MJB_SHA256,
                "compiled_model_mjb_exact_size_match": compiled_mjb_size == EXPECTED_COMPILED_MJB_SIZE_BYTES,
                "compiled_model_identity_gate": "SHA-256 of the complete MJB saved from the actual session.model; exact equality required; no tolerance or hash allowlist",
                "session_zero_time_and_zero_samples_after_constructor": session_zero_time_and_samples,
                "effective_model_options": session.effective_model_options,
                "effective_model_options_sha256": canonical_sha256(session.effective_model_options),
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
            initial_qpos = np.asarray(session.data.qpos, dtype=np.float64).copy()
            initial_qvel = np.asarray(session.data.qvel, dtype=np.float64).copy()
            if not math.isclose(session.time_s, 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise RunnerError("new free-ground session did not start at time zero")
            initial_qpos_sha = helper.array_sha256(initial_qpos)
            initial_qvel_sha = helper.array_sha256(initial_qvel)
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
            session.close()


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
    sys.path.insert(0, str(SCRIPTS))
    import numpy as np
    import run_flygym_ap_body_yaw_calibration_v1 as helper

    input_root = BUNDLE_ROOT / "frozen_inputs"
    walk_effective = read_json(input_root / "v4_1_effective_config.json")
    controller_values = read_json(input_root / "v4_1_controller_config.json")["config"]
    walk_values = dict(walk_effective["walk_config"])
    prior_screen = read_json(BUNDLE_ROOT / "paired-screen-reference.json")["summary"]
    baseline_model_signature = prior_screen["matched_inputs"]["model_signatures"]["static_d0"]
    if baseline_model_signature.get("compiled_model_sha256") != EXPECTED_PAIRED_SCREEN_RAW_XML_SHA256:
        raise RunnerError("pinned paired-screen raw-XML baseline hash changed")

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    cohort_path = RUN_ROOT / "cohort-manifest.json"
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cohort: dict[str, Any] = {
        "schema_version": "flygym-steering-dose-calibration-cohort-v4",
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
        "cgroup_memory_peak_semantics": "actual current cgroup cumulative high-water; not reset between cases",
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
                record = _run_case(
                    case=case,
                    protocol=protocol,
                    freeze=freeze,
                    expected_pair=per_case,
                    expected_p_trace_sha=cohort.get("constant_P_trace_sha256"),
                    serial_queue_attestation=serial_queue_attestation,
                    out_root=RUN_ROOT,
                    source_freeze_sha=cohort["source_freeze_sha256"],
                    walk_values=walk_values,
                    controller_values=controller_values,
                    helper=helper,
                    np=np,
                    baseline_model_signature=baseline_model_signature,
                    case_timeout_s=min(CASE_TIMEOUT_S, int(remaining)),
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
    parser.add_argument("--execute", action="store_true", required=True,
                        help="execute the pinned 15-case CPU cohort; no per-case overrides are accepted")
    args = parser.parse_args(argv)
    return run_cohort(execute_confirmed=bool(args.execute))


if __name__ == "__main__":
    raise SystemExit(main())
