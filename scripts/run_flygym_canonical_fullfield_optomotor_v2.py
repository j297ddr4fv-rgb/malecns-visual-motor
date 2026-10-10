#!/usr/bin/env python3
"""Run a reviewed nine-case, matched-seed full-field optomotor cohort.

This deliberately binds the frozen revision-4 angular retina to its existing
single-case body/neural loop.  It never calls revision-4's five-case executor:
that executor makes the old 800-frame V2 comparison blocking.  The old audit
is retained as a post-capture descriptive check; the primary integrity check
compares the three same-seed conditions before stimulus onset.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from typing import Any

sys.dont_write_bytecode = True

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
PARENT = Path("/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1")
BASE = PARENT / "canonical-fullfield-optomotor-v1/revision-2"
PREP = BASE / "preparation"
RUNS = BASE / "runs"
ANALYSIS = BASE / "analysis"
PROTOCOL_PATH = BASE / "protocol.json"
FREEZE_PATH = PREP / "source-freeze.json"
RUNNER = Path(__file__).resolve()
PREPARER = SCRIPTS / "prepare_flygym_canonical_fullfield_optomotor_v2.py"

HIGH_RUNNER = SCRIPTS / "run_flygym_free_body_online_highcontrast_dynamics_v1.py"
BODY_RUNNER = SCRIPTS / "run_flygym_free_body_online_lowcontrast_v2.py"
SESSION_MODULE = SCRIPTS / "run_flygym_reference_walk.py"
PREFIX_AUDITOR = SCRIPTS / "audit_flygym_highcontrast_prefix_v1.py"
PREFIX_AUDITOR_PYTHON = Path("/usr/bin/python3.12")
BODY_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
GLOBAL_NN_LOCK = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-recorded-retina-neural-ball-harness-20261007-v2/locks/"
    "neural-worker-and-ball-case.lock"
)

SEEDS = (1729, 1730, 1731)
FRAME_DT_S = 0.005
FRAME_COUNT = 2600
DURATION_S = 13.0
ONSET_S = 1.0
CONTRAST = 1.0
WAVELENGTH_DEG = 60.0
CASE_CPU_CORES = 10
COORDINATOR_CPU_CORES = 2
CASE_MEMORY_BYTES = 5 * 1024**3
CASE_RUNTIME_S = 1800
QUEUE_RUNTIME_S = 18000
MAX_VRAM_GIB = 8.0
EXPECTED_NATIVE_MJB_SHA256 = "6d7b1f8b9a7558cd5367be9e12e1a866adca3d98e6312d431b62d27ad747b2f3"
EXPECTED_NATIVE_MJB_BYTES = 23230821
EXPECTED_NATIVE_OPTIONS_SHA256 = "e6ed4b2ee31ccd3b3027afaa888391d36c6ce2f8b8dc2ed51e85c271cab2408e"
EXPECTED_ARRAY_ORDER_SHA256 = "0c8cdbb7006f774f42a19f068312605879a4ea7ca73e3db77e21e4903db14b0b"
EXPECTED_SEED1729_QPOS_SHA256 = "d292b37df0f9a83265423da2ce53b0b140d8ef6a49d6909e464b805453e75b99"
EXPECTED_SEED1729_QVEL_SHA256 = "1a0295f4bf5986c5f74eca9153a6a4cb10b073a01a76ba4a457fd862c78966a4"
EXPECTED_MOTOR_MAPPING_PATH = PROJECT / "config/leg_muscle_map.json"
EXPECTED_MOTOR_MAPPING_SHA256 = "b65c34b619bb0a93ecc0ecbb204d27586c06740496f42c509accd50b17ff00a4"
EXPECTED_MOTOR_MEMBERSHIP_COUNTS = {"total": 381, "mapped": 328, "unmapped": 53}


class RunnerError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RunnerError(f"expected JSON object at {path}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_frozen_inputs(protocol: dict[str, Any], *, full: bool = True) -> None:
    manifest = read_json(BASE / "prepare-manifest.json")
    if manifest.get("protocol", {}).get("sha256") != sha256(PROTOCOL_PATH):
        raise RunnerError("protocol differs from the immutable preparation manifest")
    if manifest.get("runner", {}).get("sha256") != sha256(RUNNER):
        raise RunnerError("runner source differs from its immutable preparation manifest")
    if manifest.get("preparer", {}).get("sha256") != sha256(PREPARER):
        raise RunnerError("preparer source differs from its immutable preparation manifest")
    if manifest.get("source_freeze", {}).get("sha256") != protocol.get("source_freeze", {}).get("sha256"):
        raise RunnerError("protocol and preparation manifest bind different source freezes")
    freeze_path = Path(protocol["source_freeze"]["path"])
    if sha256(freeze_path) != protocol["source_freeze"]["sha256"]:
        raise RunnerError("canonical source-freeze file hash mismatch")
    freeze = read_json(freeze_path)
    files = freeze.get("files", {})
    selected = files.items() if full else (
        (raw, record) for raw, record in files.items()
        if raw in set(freeze.get("runtime_file_paths", []))
    )
    for raw, record in selected:
        path = Path(raw)
        if (not path.is_file() or path.stat().st_size != record.get("size_bytes")
                or sha256(path) != record.get("sha256")):
            raise RunnerError(f"frozen source or asset changed: {path}")
    for raw, record in protocol.get("prepared_files", {}).items():
        path = Path(raw)
        if (not path.is_file() or path.stat().st_size != record.get("size_bytes")
                or sha256(path) != record.get("sha256")):
            raise RunnerError(f"prepared schedule or config changed: {path}")


def check_root_review(protocol: dict[str, Any]) -> dict[str, Any]:
    review_path = BASE / "root-execution-review.json"
    review = read_json(review_path)
    expected = {
        "status": "approved",
        "execution_authority": "authorized_after_review",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(RUNNER),
        "prepare_manifest_sha256": sha256(BASE / "prepare-manifest.json"),
        "source_freeze_sha256": protocol["source_freeze"]["sha256"],
    }
    for key, value in expected.items():
        if review.get(key) != value:
            raise RunnerError(f"root execution review does not bind {key}")
    return review


def _cgroup_path() -> str:
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            return fields[2]
    raise RunnerError("process has no cgroup-v2 membership")


def _systemd_properties(unit: str) -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit,
         "-p", "CPUQuotaPerSecUSec", "-p", "MemoryMax", "-p", "RuntimeMaxUSec", "-p", "ControlGroup"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


_TIMESPAN_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*(usec|us|µs|msec|ms|sec|s|min|m|hr|h|day|d|week|w)", re.IGNORECASE)
_TIME_FACTORS_USEC = {
    "usec": 1, "us": 1, "µs": 1,
    "msec": 1000, "ms": 1000,
    "sec": 1_000_000, "s": 1_000_000,
    "min": 60_000_000, "m": 60_000_000,
    "hr": 3_600_000_000, "h": 3_600_000_000,
    "day": 86_400_000_000, "d": 86_400_000_000,
    "week": 604_800_000_000, "w": 604_800_000_000,
}


def _systemd_timespan_usec(raw: str) -> int:
    """Parse systemctl's human-readable *USec output without relaxing equality gates."""
    value = str(raw).strip()
    if value.isdecimal():
        return int(value)
    matches = list(_TIMESPAN_TOKEN.finditer(value))
    if not matches:
        raise ValueError(f"unrecognized systemd time span: {raw!r}")
    cursor = 0
    total = Decimal(0)
    for match in matches:
        if value[cursor:match.start()].strip():
            raise ValueError(f"unrecognized systemd time span: {raw!r}")
        number = Decimal(match.group(1))
        unit = match.group(2).lower()
        total += number * _TIME_FACTORS_USEC[unit]
        cursor = match.end()
    if value[cursor:].strip() or total != total.to_integral_value():
        raise ValueError(f"systemd time span is not an integer microsecond value: {raw!r}")
    return int(total)


def _memory_property_bytes(raw: str) -> int:
    value = str(raw).strip()
    if value.isdecimal():
        return int(value)
    if value.lower() == "infinity":
        raise ValueError("MemoryMax is unlimited")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgtpe])(?:i?b)?", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"unrecognized systemd MemoryMax value: {raw!r}")
    power = "kmgtpe".index(match.group(2).lower()) + 1
    amount = Decimal(match.group(1)) * (Decimal(1024) ** power)
    if amount != amount.to_integral_value():
        raise ValueError(f"MemoryMax is not an integer byte value: {raw!r}")
    return int(amount)


def _effective_cgroup_caps() -> dict[str, Any]:
    cgroup = _cgroup_path()
    root = Path("/sys/fs/cgroup").resolve()
    current = (root / cgroup.lstrip("/")).resolve()
    if current != root and root not in current.parents:
        raise RunnerError("current cgroup path resolves outside the cgroup-v2 mount")
    cpu_fields = (current / "cpu.max").read_text(encoding="ascii").split()
    if len(cpu_fields) != 2 or int(cpu_fields[1]) <= 0:
        raise RunnerError("current cgroup cpu.max has an invalid format")
    cpu_cores = None if cpu_fields[0] == "max" else Decimal(cpu_fields[0]) / Decimal(cpu_fields[1])
    memory_raw = (current / "memory.max").read_text(encoding="ascii").strip()
    memory_bytes = None if memory_raw == "max" else int(memory_raw)
    return {
        "control_group": cgroup,
        "cpu_max_raw": " ".join(cpu_fields),
        "effective_cpu_cap_cores": None if cpu_cores is None else str(cpu_cores.normalize()),
        "memory_max_raw": memory_raw,
        "effective_memory_cap_bytes": memory_bytes,
    }


def verify_coordinator_scope() -> dict[str, Any]:
    unit = os.environ.get("CANONICAL_COORDINATOR_UNIT", "")
    if not unit.endswith(".service"):
        raise RunnerError("execute must run inside the reviewed 2-CPU coordinator systemd service")
    props = _systemd_properties(unit)
    try:
        cpu = _systemd_timespan_usec(props["CPUQuotaPerSecUSec"])
        runtime = _systemd_timespan_usec(props["RuntimeMaxUSec"])
    except (KeyError, ValueError) as exc:
        raise RunnerError("coordinator systemd CPU/runtime properties are missing") from exc
    cgroup = props.get("ControlGroup", "")
    if cpu != COORDINATOR_CPU_CORES * 1_000_000:
        raise RunnerError(f"coordinator CPU readback must be exactly 2 cores, got {cpu}")
    if runtime != QUEUE_RUNTIME_S * 1_000_000:
        raise RunnerError(f"coordinator RuntimeMax readback must be {QUEUE_RUNTIME_S}s")
    if cgroup != _cgroup_path():
        raise RunnerError("coordinator cgroup membership differs from systemd readback")
    effective = _effective_cgroup_caps()
    if Decimal(effective["effective_cpu_cap_cores"] or "0") != Decimal(COORDINATOR_CPU_CORES):
        raise RunnerError("coordinator cgroup cpu.max differs from the exact 2-core cap")
    return {"unit": unit, "properties": props, "systemd_cpu_quota_usec": cpu,
            "systemd_runtime_max_usec": runtime, "effective_cgroup": effective}


def verify_case_scope(unit: str) -> dict[str, Any]:
    props = _systemd_properties(unit)
    try:
        cpu = _systemd_timespan_usec(props["CPUQuotaPerSecUSec"])
        runtime = _systemd_timespan_usec(props["RuntimeMaxUSec"])
        memory = _memory_property_bytes(props["MemoryMax"])
    except (KeyError, ValueError, InvalidOperation) as exc:
        raise RunnerError("case systemd CPU, memory, or runtime properties are missing or malformed") from exc
    cgroup = props.get("ControlGroup", "")
    if (cpu != CASE_CPU_CORES * 1_000_000
            or runtime != CASE_RUNTIME_S * 1_000_000
            or memory != CASE_MEMORY_BYTES
            or cgroup != _cgroup_path()):
        raise RunnerError("case systemd resource readback differs from the reviewed hard caps")
    effective = _effective_cgroup_caps()
    try:
        effective_cpu = Decimal(effective["effective_cpu_cap_cores"])
    except (InvalidOperation, TypeError) as exc:
        raise RunnerError("case cgroup cpu.max does not provide a finite CPU cap") from exc
    if (effective_cpu != Decimal(CASE_CPU_CORES)
            or effective["effective_memory_cap_bytes"] != CASE_MEMORY_BYTES):
        raise RunnerError("case cgroup cpu.max or memory.max differs from the exact hard caps")
    return {"unit": unit, "properties": props, "systemd_cpu_quota_usec": cpu,
            "systemd_runtime_max_usec": runtime, "systemd_memory_max_bytes": memory,
            "effective_cgroup": effective}


def case_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return list(protocol["stimuli"]["conditions"])


def _normalized_controller_telemetry(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    telemetry = result.get("controller_telemetry")
    if isinstance(telemetry, dict):
        per_leg = telemetry.get("hind_clearance", {}).get("per_leg", {})
        if isinstance(per_leg, dict):
            for leg in ("lh", "rh"):
                priority = per_leg.get(leg, {}).get("hind_pose_priority")
                if isinstance(priority, dict):
                    priority.pop("cpu_s", None)
    return result


def _read_jsonl(path: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if len(rows) >= count:
                break
            rows.append(json.loads(line))
    if len(rows) != count:
        raise RunnerError(f"{path} has {len(rows)} rows; expected {count}")
    return rows


def _selected_row(name: str, row: dict[str, Any]) -> dict[str, Any]:
    if name == "worker-requests.jsonl":
        return {key: row.get(key) for key in ("seq", "time_s", "dt_s", "pose")}
    if name == "control-ticks.jsonl":
        return _normalized_controller_telemetry(row)
    if name == "worker/frames.jsonl":
        fields = (
            "seq", "time_s", "body_time_s", "pattern_time_s", "render_time_s",
            "neural_window_start_step_index", "neural_window_end_step_index",
            "input_pose", "input_quaternion_wxyz", "eye_column_hashes",
            "population_rates_hz", "population_spike_counts", "left_descending_drive",
            "right_descending_drive", "leg_power", "leg_steering",
            "diagnostic_neuron_vm_mv_by_csr_index", "diagnostic_population_vm_mv_mean",
            "diagnostic_vm_sample_count", "diagnostic_vm_sample_time_s",
            "neural_state_diagnostic", "debug_neural_state", "visual_drive_hash",
            "visual_drive_mean", "visual_drive_std", "visual_drive_abs_mean",
        )
        return {key: row.get(key) for key in fields if key in row}
    fields = (
        "seq", "body_pose_time_s", "body_pose_operation_time_s", "body_pose_state_sample_time_s",
        "input_pose", "source_body_pose", "virtual_eye_origins_m", "retinal_pattern_time_s",
        "worker_neural_window_start_s", "worker_neural_window_end_s", "decoder_output_time_s",
        "body_action_start_s", "body_action_end_s", "retinal_column_hashes",
        "raw_worker_leg_power_P", "raw_worker_DNp15_L_rate_hz", "raw_worker_DNp15_R_rate_hz",
        "worker_native_bilateral_drive_not_applied", "decoder_inputs_only", "decoder_output",
        "body_command_applied_this_frame", "decoder_output_queued_for_next_frame",
        "stimulus_label_passed_to_decoder",
    )
    return {key: row.get(key) for key in fields if key in row}


def compare_jsonl_prefix(path_a: Path, path_b: Path, name: str, count: int) -> dict[str, Any]:
    a = [_selected_row(name, row) for row in _read_jsonl(path_a, count)]
    b = [_selected_row(name, row) for row in _read_jsonl(path_b, count)]
    for index, (left, right) in enumerate(zip(a, b)):
        if canonical(left) != canonical(right):
            left_keys, right_keys = set(left), set(right)
            fields = sorted(key for key in left_keys | right_keys
                            if key not in left or key not in right or left.get(key) != right.get(key))
            return {"dataset": name, "equal": False, "axis": "seq" if "seq" in left else "tick",
                    "first_mismatch_index": index, "fields": fields}
    return {"dataset": name, "equal": True, "rows_compared": count}


def _first_array_row_difference(a: np.ndarray, b: np.ndarray) -> int | None:
    if a.ndim == 0 or b.ndim == 0:
        return None if a.tobytes() == b.tobytes() else 0
    if a.shape != b.shape:
        return 0
    row_bytes_a = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape[0], -1)
    row_bytes_b = np.ascontiguousarray(b).view(np.uint8).reshape(b.shape[0], -1)
    differing = np.flatnonzero(np.any(row_bytes_a != row_bytes_b, axis=1))
    return int(differing[0]) if differing.size else None


def _scalar_json_array(arrays: Any, key: str) -> Any:
    value = np.asarray(arrays[key])
    if value.ndim != 0 or value.dtype.kind not in "US":
        raise RunnerError(f"{key} must be a scalar Unicode/byte string array")
    try:
        parsed = json.loads(str(value.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{key} is not valid JSON") from exc
    return parsed


def _pinned_motor_mapping() -> tuple[dict[str, Any], str]:
    path = EXPECTED_MOTOR_MAPPING_PATH.resolve(strict=True)
    freeze = read_json(FREEZE_PATH)
    record = freeze.get("files", {}).get(str(path))
    if (not isinstance(record, dict) or record.get("sha256") != EXPECTED_MOTOR_MAPPING_SHA256
            or record.get("size_bytes") != path.stat().st_size or sha256(path) != EXPECTED_MOTOR_MAPPING_SHA256):
        raise RunnerError("leg muscle mapping config does not match its frozen source pin")
    mapping = read_json(path)
    return mapping, EXPECTED_MOTOR_MAPPING_SHA256


def _validate_motor_channel_provenance(path: Path, arrays: Any, *, expected_seed: int | None) -> dict[str, Any]:
    """Validate case-specific motor provenance against pinned mapping and NPZ arrays.

    Full-run spike totals and file paths are intentionally treated as metadata:
    they can differ after stimulus onset. Numeric count rows and membership
    arrays remain independently compared by ``compare_npz_prefix``.
    """
    provenance = _scalar_json_array(arrays, "provenance_json")
    membership_json = _scalar_json_array(arrays, "membership_json")
    if not isinstance(provenance, dict) or not isinstance(membership_json, list):
        raise RunnerError("motor provenance/membership JSON has the wrong top-level type")
    mapping, mapping_sha = _pinned_motor_mapping()
    counts = np.asarray(arrays["counts"])
    seq = np.asarray(arrays["seq"])
    times = np.asarray(arrays["time_s"])
    if counts.ndim != 2 or counts.dtype != np.dtype("uint16") or counts.shape[1] != EXPECTED_MOTOR_MEMBERSHIP_COUNTS["total"]:
        raise RunnerError("motor counts array shape/dtype differs from the pinned 381-cell channel schema")
    if np.any(counts > 10):
        raise RunnerError("motor counts array exceeds the recorder's 10-spike window cap")
    frame_count, cell_count = counts.shape
    if (frame_count != FRAME_COUNT or seq.shape != (frame_count,) or seq.dtype.kind not in "iu"
            or not np.array_equal(seq, np.arange(frame_count, dtype=seq.dtype))
            or times.dtype != np.dtype("float64") or times.shape != (frame_count,) or not np.all(np.isfinite(times))
            or not np.allclose(times, seq.astype(np.float64) * FRAME_DT_S, rtol=0.0, atol=1e-9)):
        raise RunnerError("motor seq/time_s arrays do not match the registered frame clock")
    if (provenance.get("schema_version") != 1
            or provenance.get("max_frames") != FRAME_COUNT
            or provenance.get("frames_captured") != frame_count
            or provenance.get("counts_shape") != [frame_count, cell_count]
            or provenance.get("counts_dtype") != str(counts.dtype)):
        raise RunnerError("motor provenance frame/count schema does not match actual arrays")
    actual_counts = provenance.get("actual_counts")
    if (not isinstance(actual_counts, dict) or actual_counts.get("shape") != list(counts.shape)
            or actual_counts.get("dtype") != str(counts.dtype)
            or isinstance(actual_counts.get("total_spikes"), bool)
            or not isinstance(actual_counts.get("total_spikes"), int)
            or actual_counts["total_spikes"] < 0
            or isinstance(actual_counts.get("maximum_cell_window_count"), bool)
            or not isinstance(actual_counts.get("maximum_cell_window_count"), int)
            or not 0 <= actual_counts["maximum_cell_window_count"] <= 10
            or actual_counts["total_spikes"] != int(counts.sum(dtype=np.uint64))
            or actual_counts["maximum_cell_window_count"] != int(counts.max(initial=0))):
        raise RunnerError("motor provenance full-run count summary has invalid shape/type/range")
    membership_counts = provenance.get("expected_membership_counts")
    if (provenance.get("total_leg_motor_cells") != cell_count
            or provenance.get("mapping_config_sha256") != mapping_sha
            or provenance.get("mapping_config") != mapping
            or provenance.get("mapping_path") != str(EXPECTED_MOTOR_MAPPING_PATH.resolve())
            or provenance.get("mapping_rate_anchor_hz") != mapping.get("rate_anchor", {}).get("maximum_rate_hz")
            or provenance.get("window_steps") != 10
            or not np.isclose(float(provenance.get("window_dt_s", float("nan"))), FRAME_DT_S, rtol=0.0, atol=1e-12)
            or provenance.get("vm_recorded") is not False):
        raise RunnerError("motor provenance mapping/window bindings differ from frozen config and worker contract")
    if (len(membership_json) != cell_count
            or provenance.get("cell_membership") != membership_json):
        raise RunnerError("motor provenance membership differs from strict membership_json")
    derived_membership_counts = {
        "total": len(membership_json),
        "mapped": sum(entry.get("mapped") is True for entry in membership_json if isinstance(entry, dict)),
        "unmapped": sum(entry.get("mapped") is False for entry in membership_json if isinstance(entry, dict)),
    }
    if (derived_membership_counts != EXPECTED_MOTOR_MEMBERSHIP_COUNTS
            or provenance.get("mapped_cells") != derived_membership_counts["mapped"]
            or membership_counts != derived_membership_counts):
        raise RunnerError("motor mapped/unmapped totals differ from actual membership and pinned 381/328/53 contract")
    type_mapping = {
        str(item["type_name"]): (str(item["group"]), str(item.get("slug", "")))
        for item in mapping.get("muscle_types", [])
    }
    configured_legs = {str(item["name"]): str(item["side"]) for item in mapping.get("legs", [])}
    dense_indices = []
    for entry in membership_json:
        if not isinstance(entry, dict):
            raise RunnerError("motor membership contains a non-object cell record")
        leg = str(entry.get("leg", ""))
        type_name = str(entry.get("type_name", ""))
        mapped_record = type_mapping.get(type_name)
        expected_group, expected_slug = mapped_record if mapped_record else ("unmapped", "")
        if (leg not in configured_legs
                or entry.get("side") != configured_legs[leg]
                or entry.get("name") != f"{leg}:body:{entry.get('body_id')}"
                or entry.get("mapped") is not (mapped_record is not None)
                or entry.get("mapped_group") != expected_group
                or entry.get("muscle_slug") != expected_slug
                or entry.get("mapping_reason") != ("exact_type_name" if mapped_record else "no_exact_muscle_type_mapping")):
            raise RunnerError("motor cell membership does not follow the pinned leg/type mapping")
        dense_indices.append(entry.get("dense_index"))
    if (len(set(dense_indices)) != cell_count
            or any(isinstance(value, bool) or not isinstance(value, int) for value in dense_indices)):
        raise RunnerError("motor membership dense indices are not unique integer values")
    if [entry for entry in membership_json if not entry["mapped"]] != provenance.get("unmapped_cells"):
        raise RunnerError("motor provenance unmapped-cell list differs from pinned membership")
    member_fields = {
        "dense_indices": "dense_index", "body_ids": "body_id", "names": "name",
        "sides": "side", "legs": "leg", "types": "type_name",
        "mapped_groups": "mapped_group", "mapped": "mapped",
    }
    expected_dtypes = {
        "dense_indices": np.dtype("int64"), "body_ids": np.dtype("int64"),
        "names": None, "sides": None, "legs": None, "types": None,
        "mapped_groups": None, "mapped": np.dtype("bool"),
    }
    for array_name, field_name in member_fields.items():
        actual = np.asarray(arrays[array_name])
        required_dtype = expected_dtypes[array_name]
        if ((required_dtype is not None and actual.dtype != required_dtype)
                or (required_dtype is None and actual.dtype.kind != "U")):
            raise RunnerError(f"motor provenance membership dtype differs from writer contract for {array_name}")
        expected_values = [entry[field_name] for entry in membership_json]
        expected = np.asarray(expected_values, dtype=actual.dtype)
        if actual.shape != (cell_count,) or not np.array_equal(actual, expected):
            raise RunnerError(f"motor provenance membership does not match NPZ {array_name}")
    leg_names = [str(item["name"]) for item in mapping.get("legs", [])]
    type_names = [str(item["type_name"]) for item in mapping.get("muscle_types", [])]
    muscle_groups = [str(item["group"]) for item in mapping.get("muscle_types", [])]
    for array_name, expected_values in (("leg_names", leg_names), ("muscle_types", type_names),
                                       ("muscle_groups", muscle_groups)):
        actual = np.asarray(arrays[array_name])
        if actual.dtype.kind != "U":
            raise RunnerError(f"motor provenance mapping array is not Unicode: {array_name}")
        expected = np.asarray(expected_values, dtype=actual.dtype)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise RunnerError(f"motor provenance mapping does not match NPZ {array_name}")
    per_leg = provenance.get("per_leg")
    if not isinstance(per_leg, dict) or set(per_leg) != set(leg_names):
        raise RunnerError("motor provenance per-leg membership keys differ from pinned mapping")
    for leg in leg_names:
        rows = [entry for entry in membership_json if entry.get("leg") == leg]
        expected_leg = {
            "total": len(rows),
            "mapped": sum(bool(entry.get("mapped")) for entry in rows),
            "unmapped": sum(not bool(entry.get("mapped")) for entry in rows),
        }
        if per_leg.get(leg) != expected_leg:
            raise RunnerError(f"motor provenance per-leg counts do not match membership for {leg}")
    expected_output = path.resolve(strict=True)
    expected_sidecar = (expected_output.parent / "muscle_channels_membership.json").resolve(strict=True)
    if (Path(str(provenance.get("output", ""))).resolve() != expected_output
            or Path(str(provenance.get("membership_path", ""))).resolve() != expected_sidecar):
        raise RunnerError("motor provenance output/membership paths do not bind to this worker artifact")
    sidecar = read_json(expected_sidecar)
    if (sidecar.get("schema_version") != 1
            or sidecar.get("mapping_config_sha256") != mapping_sha
            or sidecar.get("dense_indices") != np.asarray(arrays["dense_indices"]).tolist()
            or sidecar.get("cells") != membership_json):
        raise RunnerError("motor membership sidecar does not match NPZ arrays and pinned config")
    engine = provenance.get("engine")
    if (not isinstance(engine, dict) or not isinstance(engine.get("backend"), str)
            or not isinstance(engine.get("device"), str)
            or not np.isclose(float(engine.get("dt_ms", float("nan"))), 0.5, rtol=0.0, atol=1e-9)
            or (expected_seed is not None and engine.get("seed") != expected_seed)
            or not isinstance(engine.get("graph_n"), int) or engine["graph_n"] <= 0
            or not isinstance(engine.get("graph_edges"), int) or engine["graph_edges"] <= 0):
        raise RunnerError("motor provenance engine metadata differs from the registered same-seed worker")
    return {
        "status": "validated_case_specific_provenance_excluded_from_prefix_equality",
        "mapping_config_sha256": mapping_sha,
        "frames_captured": frame_count,
        "motor_cells": cell_count,
        "mapped_cells": EXPECTED_MOTOR_MEMBERSHIP_COUNTS["mapped"],
        "unmapped_cells": EXPECTED_MOTOR_MEMBERSHIP_COUNTS["unmapped"],
        "membership_json_and_sidecar_match_arrays": True,
        "case_specific_full_run_totals_and_paths_compared_between_conditions": False,
    }


def compare_npz_prefix(path_a: Path, path_b: Path, dataset: str, *, expected_seed: int | None = None) -> list[dict[str, Any]]:
    summary_fields = {
        "whole_network_vm_summary_seq",
        "whole_network_vm_summary_sample_time_s",
        "whole_network_vm_summary_step_index",
        "whole_network_vm_min_mv",
        "whole_network_vm_max_mv",
        "whole_network_vm_fraction_below_minus_100_mv",
        "whole_network_vm_percentiles_mv",
    }
    with np.load(path_a, allow_pickle=False) as za, np.load(path_b, allow_pickle=False) as zb:
        if set(za.files) != set(zb.files):
            return [{"dataset": dataset, "field": "<key-set>", "equal": False,
                     "axis": "array-name", "detail": "NPZ member sets differ"}]
        contact_mask_a = contact_mask_b = None
        frame_mask_a = frame_mask_b = None
        summary_mask_a = summary_mask_b = None
        if dataset == "contacts.npz":
            ta = np.asarray(za["time_s"])
            tb = np.asarray(zb["time_s"])
            contact_mask_a, contact_mask_b = ta <= ONSET_S, tb <= ONSET_S
        if dataset == "worker/neural_state_diagnostic.npz":
            sa = np.asarray(za["whole_network_vm_summary_seq"])
            sb = np.asarray(zb["whole_network_vm_summary_seq"])
            summary_mask_a, summary_mask_b = sa < 200, sb < 200
            frame_seq_a = np.asarray(za["seq"])
            frame_seq_b = np.asarray(zb["seq"])
            frame_mask_a, frame_mask_b = frame_seq_a < 200, frame_seq_b < 200
            if (sa.ndim != 1 or sb.ndim != 1 or frame_seq_a.ndim != 1 or frame_seq_b.ndim != 1
                    or not np.array_equal(sa[summary_mask_a], sb[summary_mask_b])
                    or not np.array_equal(frame_seq_a[frame_mask_a], frame_seq_b[frame_mask_b])):
                return [{"dataset": dataset, "field": "<prefix-coordinates>", "equal": False,
                         "axis": "seq", "detail": "neural diagnostic frame/summary prefix coordinates differ"}]
        checks: list[dict[str, Any]] = []
        if dataset == "worker/muscle_channels.npz":
            for label, path, archive in (("reference", path_a, za), ("candidate", path_b, zb)):
                try:
                    detail = _validate_motor_channel_provenance(path, archive, expected_seed=expected_seed)
                except (RunnerError, OSError, TypeError, ValueError, KeyError) as exc:
                    checks.append({"dataset": dataset, "field": "provenance_json", "equal": False,
                                   "axis": "validated-case-metadata", "case_role": label,
                                   "detail": f"{type(exc).__name__}: {exc}"})
                else:
                    checks.append({"dataset": dataset, "field": "provenance_json", "equal": True,
                                   "axis": "validated-case-metadata", "case_role": label, **detail})
        for key in za.files:
            a, b = np.asarray(za[key]), np.asarray(zb[key])
            axis = "static-array"
            if dataset == "worker/muscle_channels.npz" and key == "provenance_json":
                continue
            if key == "metadata_json":
                try:
                    if a.ndim != 0 or b.ndim != 0 or a.dtype.kind not in "US" or b.dtype.kind not in "US":
                        raise ValueError("metadata_json is not a scalar string")
                    json.loads(str(a.item()))
                    json.loads(str(b.item()))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    checks.append({"dataset": dataset, "field": key, "equal": False,
                                   "axis": "non-scientific-metadata", "detail": str(exc)})
                else:
                    checks.append({"dataset": dataset, "field": key, "equal": True,
                                   "axis": "non-scientific-metadata", "detail": "parsed; case-specific metadata excluded from numeric prefix equality"})
                continue
            if dataset == "contacts.npz" and a.ndim and a.shape[0] == len(contact_mask_a):
                a, b = a[contact_mask_a], b[contact_mask_b]
                axis = "contact-row-through-1s"
            elif dataset == "trajectory.npz" and a.ndim and a.shape[0] == 130001:
                a, b = a[:10001], b[:10001]
                axis = "native-sample-0..10000"
            elif dataset == "worker/neural_state_diagnostic.npz":
                if key in summary_fields:
                    if (a.ndim == 0 or b.ndim == 0 or a.shape[0] != len(summary_mask_a)
                            or b.shape[0] != len(summary_mask_b)):
                        checks.append({"dataset": dataset, "field": key, "equal": False,
                                       "axis": "whole-network-summary-seq<200",
                                       "detail": "registered summary array does not share the summary_seq axis"})
                        continue
                    a, b = a[summary_mask_a], b[summary_mask_b]
                    axis = "whole-network-summary-seq<200"
                elif a.ndim and a.shape[0] == len(frame_mask_a):
                    a, b = a[frame_mask_a], b[frame_mask_b]
                    axis = "seq-0..199"
            elif a.ndim and a.shape[0] == FRAME_COUNT:
                a, b = a[:200], b[:200]
                axis = "seq-0..199"
            if a.dtype != b.dtype or a.shape != b.shape:
                checks.append({"dataset": dataset, "field": key, "equal": False,
                               "axis": axis, "detail": "dtype or selected shape differs"})
                continue
            if np.ascontiguousarray(a).tobytes(order="C") != np.ascontiguousarray(b).tobytes(order="C"):
                checks.append({"dataset": dataset, "field": key, "equal": False, "axis": axis,
                               "first_mismatch_index": _first_array_row_difference(a, b)})
        if not checks:
            checks.append({"dataset": dataset, "equal": True, "axis": "registered-prefix-fields",
                           "members_compared": len(za.files)})
        return checks


def compare_initial_identity(run_a: Path, run_b: Path) -> list[dict[str, Any]]:
    ra, rb = read_json(run_a / "run-record.json"), read_json(run_b / "run-record.json")
    ia, ib = ra.get("initial_state", {}), rb.get("initial_state", {})
    fields = ("time_s", "qpos_sha256", "qvel_sha256", "cpg_phase_rad_sha256", "cpg_magnitude_sha256")
    bad = [key for key in fields if ia.get(key) != ib.get(key)]
    pa, pb = read_json(run_a / "precontroller-identity.json"), read_json(run_b / "precontroller-identity.json")
    pfields = ("compiled_model_mjb_sha256", "effective_model_options_sha256", "array_order_sha256",
               "qpos_sha256", "qvel_sha256", "walk_config_nonseed_duration_sha256",
               "runtime_upstream_source_hashes_sha256", "package_versions")
    pbad = [key for key in pfields if pa.get(key) != pb.get(key)]
    return [
        {"dataset": "run-record.initial_state", "equal": not bad, "axis": "constructor/initial-CPG", "fields": bad},
        {"dataset": "precontroller-identity.json", "equal": not pbad,
         "axis": "compiled-model/options/order/qpos/qvel/config", "fields": pbad},
    ]


def audit_triplet(seed: int, cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_direction = {row["direction"]: row for row in cases}
    reference = RUNS / by_direction["static_c1"]["case_id"]
    comparisons: list[dict[str, Any]] = []
    for direction in ("positive60", "negative60"):
        candidate = RUNS / by_direction[direction]["case_id"]
        comparisons.extend({"condition": direction, **row} for row in compare_initial_identity(reference, candidate))
        for name, rows in (("worker-requests.jsonl", 200), ("online-frames.jsonl", 200),
                           ("worker/frames.jsonl", 200), ("control-ticks.jsonl", 1000)):
            comparisons.append({"condition": direction, **compare_jsonl_prefix(reference / name, candidate / name, name, rows)})
        for name in ("worker/eye_columns.npz", "worker/visual_drive.npz",
                     "worker/muscle_channels.npz", "worker/neural_state_diagnostic.npz",
                     "trajectory.npz", "contacts.npz"):
            comparisons.extend({"condition": direction, **row}
                               for row in compare_npz_prefix(reference / name, candidate / name, name,
                                                             expected_seed=seed))
        debug_a, debug_b = reference / "worker/debug-neural-state", candidate / "worker/debug-neural-state"
        for leaf in sorted(path.name for path in debug_a.glob("*.npz")):
            if leaf.startswith("frame-") and int(leaf[6:10]) >= 40:
                continue
            comparisons.extend({"condition": direction, **row}
                               for row in compare_npz_prefix(debug_a / leaf, debug_b / leaf,
                                                             f"worker/debug-neural-state/{leaf}"))
    all_equal = all(row.get("equal") is True for row in comparisons)
    return {
        "schema_version": "canonical-fullfield-preonset-triplet-audit-v1",
        "seed": seed,
        "status": "exact_preonset_common_prefix" if all_equal else "preonset_mismatch_unresolved",
        "reference_condition": "static_c1",
        "comparison_conditions": ["positive60", "negative60"],
        "frame_axis": {"seq_first": 0, "seq_last_inclusive": 199, "frame_time_half_open_s": [0.0, 1.0]},
        "body_control_tick_axis": {"first": 0, "last_inclusive": 999, "count": 1000},
        "physics_step_axis": {"first": 0, "last_inclusive": 9999, "count": 10000},
        "native_sample_axis": {"first": 0, "last_inclusive": 10000, "count": 10001,
                                "includes_initial_sample": True, "includes_t1_boundary_sample": True},
        "contact_event_cutoff_s_inclusive": ONSET_S,
        "neural_full_state_coverage_limit": "Full mutable snapshots are only captured through seq39; later comparisons cover available per-frame diagnostics, spikes, drives and periodic full-network Vm summaries.",
        "comparisons": comparisons,
        "interpretation": "This is an engineering common-input integrity check. A mismatch makes the seed triplet unresolved for stimulus attribution; it does not stop capture of the remaining conditions and is never converted to a response pass.",
    }


def _case_source(protocol: dict[str, Any], case_id: str) -> dict[str, Any]:
    for row in case_rows(protocol):
        if row["case_id"] == case_id:
            return row
    raise RunnerError(f"case is not registered: {case_id}")


def _install_bound_runtime(protocol: dict[str, Any], case: dict[str, Any]) -> tuple[Any, Any]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    high = importlib.import_module("run_flygym_free_body_online_highcontrast_dynamics_v1")
    high.BASE, high.PREPARATION = BASE, PREP
    high.SPECS, high.SCHEDULES, high.RUNS, high.ANALYSIS = PREP / "stimulus-specs", PREP / "retina-schedules", RUNS, ANALYSIS
    high.FRAME_DT_S, high.FRAME_RATE_HZ, high.FRAME_COUNT, high.DURATION_S = FRAME_DT_S, 200, FRAME_COUNT, DURATION_S
    high.GRAY_END_S, high.SWITCH_S, high.SEED, high.CONTRAST, high.WAVELENGTH_DEG = ONSET_S, 7.0, case["body_seed"], CONTRAST, WAVELENGTH_DEG
    high.WINDOWS = {key: tuple(value) for key, value in protocol["analysis_windows_s_half_open"].items()}
    high.BODY_PYTHON, high.WORKER_PYTHON = BODY_PYTHON, PREFIX_AUDITOR_PYTHON
    high.CASE_RUNTIME_S, high.QUEUE_RUNTIME_S = CASE_RUNTIME_S, QUEUE_RUNTIME_S
    high.GLOBAL_NN_LOCK = GLOBAL_NN_LOCK
    high.RESOURCE_LIMITS = {"host_memory_max": "5G", "cpu_quota": "1000%",
                            "cpu_quota_per_sec_usec_max": CASE_CPU_CORES * 1_000_000,
                            "worker_max_vram_gib": MAX_VRAM_GIB}
    high.PREFIX_AUDIT, high.PREFIX_AUDIT_PYTHON = PREFIX_AUDITOR, PREFIX_AUDITOR_PYTHON
    high.verify_frozen_inputs = lambda frozen_protocol: verify_frozen_inputs(frozen_protocol, full=False)
    high.CASES = tuple(case_rows(protocol))
    high.CASE_IDS = tuple(row["case_id"] for row in case_rows(protocol))
    high._ACTIVE_CASE = case
    high._ACTIVE_SCHEDULE = read_json(Path(case["schedule_path"]))
    if sha256(Path(case["schedule_path"])) != case["schedule_sha256"]:
        raise RunnerError("active case retina schedule hash changed")
    base = high._configure_base(protocol, tuple(case_rows(protocol)))
    return high, base


def _model_guard(expected_case: dict[str, Any]) -> None:
    """Instrument only the existing session factory; never changes model data."""
    module = importlib.import_module("run_flygym_reference_walk")
    original_factory = module.FlyGymReferenceSession
    run_dir = RUNS / expected_case["case_id"]
    output = run_dir / "compiled_model.mjb"
    ref = read_json(Path(read_json(FREEZE_PATH)["native_model_reference"]["capture_json_path"]))
    expected_config = expected_case["effective_walk_config"]

    def guarded_factory(*args: Any, **kwargs: Any) -> Any:
        session = original_factory(*args, **kwargs)
        try:
            before = {
                "time_s": float(session.time_s),
                "data_time_s": float(session.data.time),
                "control_samples": int(session._control_samples),
                "native_samples": int(session._native_samples),
            }
            if before["time_s"] != 0.0 or before["data_time_s"] != 0.0 or before["control_samples"] != 0:
                raise RunnerError("constructor returned a nonzero-time or stepped session")
            if before["native_samples"] != 0:
                raise RunnerError("constructor native sample count differs from the pinned zero-step capture")
            output.parent.mkdir(parents=True, exist_ok=True)
            session.mj.mj_saveModel(session.model, str(output), None)
            model_sha = sha256(output)
            model_options = session.effective_model_options
            options_path = run_dir / "precontroller-model-options.json"
            write_json(options_path, model_options)
            options_sha = sha256(options_path)
            array_order = session._array_order_names()
            order_path = run_dir / "precontroller-array-order.json"
            write_json(order_path, array_order)
            order_sha = sha256(order_path)
            qpos = np.ascontiguousarray(session.data.qpos)
            qvel = np.ascontiguousarray(session.data.qvel)
            qpos_sha, qvel_sha = sha_bytes(qpos.tobytes()), sha_bytes(qvel.tobytes())
            config = json.loads(json.dumps(asdict(session.config), allow_nan=False))
            cfg_no_seed_duration = {k: v for k, v in config.items() if k not in {"seed", "duration_s"}}
            reference_options = json.loads((Path(FREEZE_PATH).parent / "native-model-effective-options.json").read_text())
            reference_order = json.loads((Path(FREEZE_PATH).parent / "native-model-array-order.json").read_text())
            source_provenance = session.upstream_source_hashes
            expected_source_provenance = ref.get("runtime_upstream_source_hashes")
            package_versions = {
                "flygym": source_provenance.get("installed_flygym_version"),
                "mujoco": str(session.mj.__version__),
                "numpy": np.__version__,
            }
            expected_package_versions = ref.get("package_versions")
            identity = {
                "schema_version": "canonical-fullfield-precontroller-native-identity-v1",
                "case_id": expected_case["case_id"], "seed": int(expected_case["body_seed"]),
                "session_time_before_capture": before,
                "compiled_model_mjb_path": str(output), "compiled_model_mjb_sha256": model_sha,
                "compiled_model_mjb_size_bytes": output.stat().st_size,
                "effective_model_options": model_options, "effective_model_options_sha256": options_sha,
                "array_order": array_order, "array_order_sha256": order_sha,
                "qpos_sha256": qpos_sha, "qvel_sha256": qvel_sha,
                "walk_config": config,
                "walk_config_nonseed_duration_sha256": sha_bytes(canonical(cfg_no_seed_duration)),
                "runtime_upstream_source_hashes": source_provenance,
                "runtime_upstream_source_hashes_sha256": sha_bytes(canonical(source_provenance)),
                "package_versions": package_versions,
                "capture_method": "mujoco.mj_saveModel(session.model); session.model is the constructed native runtime model",
                "capture_position": "after FlyGymReferenceSession constructor; before PhaseSpeedGaitController, worker launch, and any body control/physics step",
                "historical_seed1729_qpos_qvel_reference": {
                    "qpos_sha256": EXPECTED_SEED1729_QPOS_SHA256,
                    "qvel_sha256": EXPECTED_SEED1729_QVEL_SHA256,
                },
                "native_reference_status": ref.get("status"),
            }
            failures = []
            if model_sha != EXPECTED_NATIVE_MJB_SHA256 or output.stat().st_size != EXPECTED_NATIVE_MJB_BYTES:
                failures.append("actual compiled native MJB differs from the accepted constructor-only reference")
            if options_sha != EXPECTED_NATIVE_OPTIONS_SHA256 or model_options != reference_options:
                failures.append("effective native model options differ from the accepted reference")
            if order_sha != EXPECTED_ARRAY_ORDER_SHA256 or array_order != reference_order:
                failures.append("runtime model/actuator/sensor array order differs from the accepted reference")
            if source_provenance != expected_source_provenance:
                failures.append("runtime installed package/source hashes differ from the accepted constructor-only reference")
            if package_versions != expected_package_versions:
                failures.append("runtime package versions differ from the accepted constructor-only reference")
            if int(expected_case["body_seed"]) == 1729 and (qpos_sha != EXPECTED_SEED1729_QPOS_SHA256 or qvel_sha != EXPECTED_SEED1729_QVEL_SHA256):
                failures.append("seed-1729 t=0 qpos/qvel differ from the accepted no-step capture")
            if config != expected_config:
                failures.append("effective WalkConfig differs from the frozen case config")
            after = {"time_s": float(session.time_s), "data_time_s": float(session.data.time),
                     "control_samples": int(session._control_samples), "native_samples": int(session._native_samples)}
            identity["session_time_after_capture"] = after
            identity["guard_status"] = "fail_closed" if failures else "passed"
            identity["gate_failures"] = failures
            write_json(run_dir / "precontroller-identity.json", identity)
            if after != before:
                raise RunnerError("native model capture advanced or changed constructor state")
            if failures:
                raise RunnerError("pre-controller native-model identity gate failed: " + "; ".join(failures))
            return session
        except BaseException:
            try:
                session.close()
            except Exception:
                pass
            raise

    module.FlyGymReferenceSession = guarded_factory


def case_child(case_id: str) -> None:
    if os.environ.get("ONLINE_FREEBODY_CASE_SCOPE") != "1":
        raise RunnerError("case process is not inside its reviewed systemd case unit")
    protocol = read_json(PROTOCOL_PATH)
    verify_frozen_inputs(protocol, full=False)
    check_root_review(protocol)
    case = _case_source(protocol, case_id)
    unit = os.environ.get("ONLINE_FREEBODY_CASE_UNIT", "")
    if not unit:
        raise RunnerError("case unit name is missing")
    resource_scope = verify_case_scope(unit)
    high, base = _install_bound_runtime(protocol, case)
    _model_guard(case)
    GLOBAL_NN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_NN_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = base._run_case(case, protocol)
    if record.get("status") != "complete":
        raise RunnerError("reused source-pinned single-case body/neural loop did not complete")
    high._finalize_case(case, protocol, record)
    record_path = RUNS / case_id / "run-record.json"
    final = read_json(record_path)
    runtime_contract = final.get("runtime_contract")
    if isinstance(runtime_contract, dict):
        runtime_contract.pop("state_reset_at_t7", None)
        runtime_contract["direction_reversal_present"] = False
        runtime_contract["reversal_latency_or_response_metric"] = "not measured"
    final["canonical_resource_scope"] = resource_scope
    final["canonical_fullfield_protocol"] = {
        "case_seed_role": "same seed initializes body and neural worker; condition selects retinal pixels only",
        "condition_label_passed_to_decoder_or_controller": False,
        "one_frame_nn_command_delay_preserved": True,
        "primary_integrity_axis": "same-seed triplet gray prelude seq0..199, ticks0..999, native samples0..10000",
        "historical_v2_prefix_is_secondary": True,
        "direction_reversal_present": False,
        "reversal_latency_or_response_metric": "not measured",
        "precontroller_identity_path": str(RUNS / case_id / "precontroller-identity.json"),
    }
    write_json(record_path, final)
    integrity_path = RUNS / case_id / "online-integrity.json"
    if integrity_path.is_file():
        integrity = read_json(integrity_path)
        integrity["files"] = high._all_case_files(RUNS / case_id, omit={"online-integrity.json"})
        write_json(integrity_path, integrity)


def resource_preflight(which: str) -> dict[str, Any]:
    if which == "coordinator":
        return {"status": "metadata_only_resource_preflight_passed", "scope": verify_coordinator_scope(),
                "simulation_constructed": False, "body_steps": 0, "nn_started": False, "gpu_used": False}
    if which == "case":
        if os.environ.get("ONLINE_FREEBODY_CASE_SCOPE") != "1":
            raise RunnerError("case preflight must run inside the exact-cap transient case unit")
        unit = os.environ.get("ONLINE_FREEBODY_CASE_UNIT", "")
        if not unit:
            raise RunnerError("case preflight unit name is missing")
        return {"status": "metadata_only_resource_preflight_passed", "scope": verify_case_scope(unit),
                "simulation_constructed": False, "body_steps": 0, "nn_started": False, "gpu_used": False}
    raise RunnerError(f"unknown resource preflight scope: {which}")


def _legacy_v2_audit(case: dict[str, Any]) -> dict[str, Any]:
    """Run the old 4 s exact check for seed 1729 as a nonblocking diagnostic."""
    case_dir = RUNS / case["case_id"]
    if int(case["body_seed"]) != 1729:
        result = {"status": "not_available_for_seed", "historical_seed": 1729,
                  "blocking": False, "reason": "the frozen V2 triplet exists only for seed 1729"}
        write_json(case_dir / "analysis/legacy-v2-prefix-audit-status.json", result)
        return result
    import run_flygym_free_body_online_highcontrast_dynamics_v1 as high
    high.BASE, high.RUNS, high.PREFIX_AUDIT, high.PREFIX_AUDIT_PYTHON = BASE, RUNS, PREFIX_AUDITOR, PREFIX_AUDITOR_PYTHON
    try:
        record = high._audit_completed_case(case["case_id"])
        result = {
            "status": "passed" if record.get("prefix_audit_pass") else "failed_numeric_or_recorded_prefix",
            "blocking": False,
            "reference_case_id": case["prefix_reference_case_id"],
            "report": record.get("prefix_audit"),
            "preserve_failure_data": True,
        }
    except BaseException as exc:
        result = {"status": "auditor_error", "blocking": False,
                  "reference_case_id": case["prefix_reference_case_id"],
                  "error": f"{type(exc).__name__}: {exc}", "preserve_failure_data": True}
    write_json(case_dir / "analysis/legacy-v2-prefix-audit-status.json", result)
    return result


def _launch_case(case_id: str) -> None:
    unit = f"canonical-fullfield-{case_id}-{os.getpid()}-{time.monotonic_ns()}.service"
    command = [
        "systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet",
        f"--unit={unit}", f"--property=MemoryMax={CASE_MEMORY_BYTES}",
        f"--property=CPUQuota={CASE_CPU_CORES * 100}%",
        f"--property=RuntimeMaxSec={CASE_RUNTIME_S}",
        "--setenv=ONLINE_FREEBODY_CASE_SCOPE=1", f"--setenv=ONLINE_FREEBODY_CASE_UNIT={unit}",
        f"--setenv=PYTHONPATH={SCRIPTS}", "--setenv=PYTHONUNBUFFERED=1",
        "--setenv=FLYVIS_ROOT_DIR=/data/models/flyvis",
        str(BODY_PYTHON), str(RUNNER), "--case-child", "--case", case_id,
    ]
    subprocess.run(command, cwd=PROJECT, check=True, timeout=CASE_RUNTIME_S + 30)


def execute() -> None:
    protocol = read_json(PROTOCOL_PATH)
    verify_frozen_inputs(protocol, full=True)
    check_root_review(protocol)
    coordinator = verify_coordinator_scope()
    if any((RUNS / row["case_id"]).exists() and any((RUNS / row["case_id"]).iterdir()) for row in case_rows(protocol)):
        raise RunnerError("refusing to overwrite any existing case output")
    lock_path = BASE / "queue.lock"
    started = time.monotonic()
    status: dict[str, Any] = {
        "schema_version": "canonical-fullfield-cohort-status-v1", "status": "running",
        "case_order": protocol["queue_order"], "completed": [], "current": None,
        "coordinator_readback": coordinator, "started_utc_epoch_s": time.time(),
        "primary_preonset_prefix": "pending", "historical_v2_prefix": "secondary_descriptive_only",
    }
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(BASE / "queue-status.json", status)
        try:
            for case in case_rows(protocol):
                if time.monotonic() - started >= QUEUE_RUNTIME_S:
                    raise TimeoutError("canonical full-field queue exceeded its 18000 s outer cap")
                cid = case["case_id"]
                status["current"] = cid
                write_json(BASE / "queue-status.json", status)
                _launch_case(cid)
                record_path = RUNS / cid / "run-record.json"
                record = read_json(record_path)
                if record.get("status") != "complete":
                    raise RunnerError(f"case did not complete: {cid}")
                if record.get("resource_peak_within_hard_limit") is not True:
                    raise RunnerError(f"case cgroup memory peak missing or over cap: {cid}")
                legacy = _legacy_v2_audit(case)
                status["completed"].append(cid)
                status.setdefault("historical_v2_case_audits", {})[cid] = legacy["status"]
                write_json(BASE / "queue-status.json", status)
                seed_cases = [row for row in case_rows(protocol) if row["cohort_id"] == case["cohort_id"]]
                if all((RUNS / row["case_id"] / "run-record.json").is_file() for row in seed_cases):
                    try:
                        audit = audit_triplet(int(case["body_seed"]), seed_cases)
                    except BaseException as exc:
                        audit = {
                            "schema_version": "canonical-fullfield-preonset-triplet-audit-v1",
                            "seed": int(case["body_seed"]),
                            "status": "preonset_audit_error_unresolved",
                            "error": f"{type(exc).__name__}: {exc}",
                            "interpretation": "Capture data retained. Audit errors do not become a pass or erase the triplet.",
                        }
                    write_json(ANALYSIS / f"preonset-triplet-audit-seed{case['body_seed']}.json", audit)
                    status.setdefault("primary_preonset_by_seed", {})[str(case["body_seed"])] = audit["status"]
                    write_json(BASE / "queue-status.json", status)
            primary = status.get("primary_preonset_by_seed", {})
            primary_pass = len(primary) == len(SEEDS) and all(v == "exact_preonset_common_prefix" for v in primary.values())
            status.update({
                "status": "complete_descriptive_capture",
                "current": None,
                "finished_utc_epoch_s": time.time(),
                "primary_preonset_prefix": "all_seed_triplets_exact" if primary_pass else "one_or_more_triplets_unresolved",
                "historical_v2_prefix": "unresolved_historical_reproducibility_diagnostic",
                "biological_pass_claim": False,
                "statistical_or_biological_validation": False,
            })
            write_json(BASE / "cohort-summary.json", {
                "schema_version": "canonical-fullfield-cohort-summary-v1",
                "status": status["status"],
                "primary_preonset_by_seed": primary,
                "historical_v2_prefix": status["historical_v2_prefix"],
                "case_count": len(status["completed"]),
                "classification": "multi-seed engineering visual-motor replication; no biological pass",
                "biological_pass_claim": False,
                "case_records": [read_json(RUNS / cid / "run-record.json") for cid in status["completed"]],
            })
        except BaseException:
            status.update(status="failed_stopped", current=status.get("current"), error=traceback.format_exc())
            raise
        finally:
            write_json(BASE / "queue-status.json", status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify", action="store_true", help="verify source/config pins only")
    modes.add_argument("--execute", action="store_true", help="run after the separate exact root review")
    modes.add_argument("--case-child", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--resource-preflight-coordinator", action="store_true",
                       help="read actual coordinator systemd/cgroup caps only; no model or simulation")
    modes.add_argument("--resource-preflight-case", action="store_true",
                       help="read actual case systemd/cgroup caps only; no model or simulation")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify:
        protocol = read_json(PROTOCOL_PATH)
        verify_frozen_inputs(protocol, full=True)
        print(json.dumps({"status": "pins_verified_only", "body_steps": 0, "nn_started": False}, indent=2))
        return 0
    if args.case_child:
        if not args.case:
            parser.error("--case-child needs --case")
        case_child(args.case)
        return 0
    if args.resource_preflight_coordinator:
        print(json.dumps(resource_preflight("coordinator"), indent=2, sort_keys=True))
        return 0
    if args.resource_preflight_case:
        print(json.dumps(resource_preflight("case"), indent=2, sort_keys=True))
        return 0
    execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
