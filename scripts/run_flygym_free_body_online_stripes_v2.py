#!/usr/bin/env python3
"""Prepare or run a causal free-body FlyGym/FlyVis/MaleCNS stripe loop.

The runner uses live thorax pose to render each angular retina frame. The
worker's current-frame P and DNp15 rates enter one fixed calibrated-static
decoder; its output is queued and applied over the *next* five 1 ms body ticks.
The first body frame is neutral and the final decoded frame remains queued at
the 4 s endpoint. Stimulus condition selects pixels only; it never selects an
action. This is an engineered candidate, not a validated biological response.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import select
import subprocess
import sys
import time
import traceback
from typing import Any, TextIO

sys.dont_write_bytecode = True

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
ENGINEERING = Path("/data/artifacts/by-project/malecns-gpu/engineering")
BASE = ENGINEERING / "feedback-diagnosis-20261007-v1/online-free-body-stripes-v2"
PREPARATION = BASE / "preparation"
SPECS = PREPARATION / "angular-stimulus-specs"
RUNS = BASE / "runs"

FREE_BODY_REPLAY = SCRIPTS / "replay_flygym_free_body_stripes_v1.py"
AP_MECHANICS_RUNNER = SCRIPTS / "run_flygym_ap_body_yaw_calibration_v1.py"
REFERENCE_WALK = SCRIPTS / "run_flygym_reference_walk.py"
AP_CONTROLLER = SCRIPTS / "flygym_phase_speed_gait_repaired_v1.py"
AP_PARENT = SCRIPTS / "flygym_ap_endpoint_gesture_prototype.py"
V41_CONTROLLER = SCRIPTS / "flygym_phase_speed_gait_v4_1.py"
ONLINE_POSE_BRIDGE = SCRIPTS / "run_flygym_online_optomotor_eye_parity_v4_1_attempt5.py"
WORKER = SCRIPTS / "flygym_online_neural_worker_eye_parity_state_v2.py"
WORKER_PARENT = SCRIPTS / "flygym_online_neural_worker_eye_parity_v2.py"
WORKER_LEGACY_PARENT = SCRIPTS / "flygym_online_neural_worker.py"
FLYVIS_ADAPTER = SCRIPTS / "flyvis_runtime_adapter_eye_parity_v2.py"
FLYVIS_ADAPTER_PARENT = SCRIPTS / "flyvis_runtime_adapter.py"
MALECNS_ENGINE = SCRIPTS / "malecns_engine.py"
NEUROMUSCULAR_ADAPTER = SCRIPTS / "neuromuscular_adapter.py"
TWIN_CONTRACTS = SCRIPTS / "twin_contracts.py"
POPULATION_SELECTORS = SCRIPTS / "population_selectors.py"
ANGULAR_RENDERER = SCRIPTS / "standard_optomotor_stimulus.py"
REST_CONTRACT = SCRIPTS / "flygym_rest_contract.py"
POSE_IK = SCRIPTS / "flygym_pose_priority_ik_v2.py"
GAIT_V3 = SCRIPTS / "flygym_phase_speed_gait_v3.py"
GAIT_V2 = SCRIPTS / "flygym_phase_speed_gait_v2.py"
GAIT_V1 = SCRIPTS / "flygym_phase_speed_gait.py"
DECODER = SCRIPTS / "dnp15_opponent_steering_decoder_repaired_v1.py"
DECODER_LEGACY = SCRIPTS / "dnp15_opponent_steering_decoder.py"
DIRECTION_CALIBRATION = SCRIPTS / "flygym_direction_calibration.py"

WALK_EFFECTIVE_CONFIG = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-phase-speed-repair-20261006-v1/development/v4-1/weak_0p30/effective_config.json"
)
CONTROLLER_CONFIG = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-phase-speed-repair-20261006-v1/development/v4-1/weak_0p30/controller_config.json"
)
WORKER_CONFIG_SOURCE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-fixed-head-contrast-ch-diagnostic-20261007-v2-r2/worker-config/"
    "flygym_online_neural_worker_contrast_ch.json"
)
BASELINE_SOURCE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-dnp15-template-steering-decoder-20261007-v1/static-baseline-payload.json"
)
DIRECTION_CONFIG = PROJECT / "config/flygym_direction_calibration_v1.json"
FLYVIS_TARGET_DIRECTIONS = PROJECT / "artifacts/flyvis-eye/t4t5_dirs.npz"
FLYVIS_MODEL_ROOT = Path("/data/models/flyvis/results/flow/0000/000")
FLYVIS_CHECKPOINT = FLYVIS_MODEL_ROOT / "best_chkpt"
FLYVIS_MODEL_META = FLYVIS_MODEL_ROOT / "_meta.yaml"
CSR_ROOT = Path("/data/cache/malecns-gpu/csr-flyb")
NEUROMUSCULAR_MAP = PROJECT / "config/neuromuscular_map.json"
SENSORY_MAP = PROJECT / "config/sensory_map.json"
MOTOR_CHANNEL_MAP = PROJECT / "config/leg_muscle_map.json"
UPSTREAM_BODY_RUN = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-phase-speed-repair-20261006-v1/development/v4-1/weak_0p30"
)
UPSTREAM_BODY_MANIFEST = UPSTREAM_BODY_RUN / "manifest.json"
UPSTREAM_BODY_ASSET_HASHES = UPSTREAM_BODY_RUN / "upstream_asset_hashes.json"

# The calibrated_static reference belongs to the original fixed-head C=1
# family. These hashes bind its source identity without extending its scope.
LOCKED_SHA256 = {
    str(WORKER.resolve()): "543e82ed40c339d000cf38cb179f60e5818ef9de65747d36fb19fdb346adeb3d",
    str(WORKER_PARENT.resolve()): "23579a136547698b7eec9cd4a0ca2c3a8f8d5d0144601b08f67dcac383cc7ccb",
    str(FLYVIS_ADAPTER.resolve()): "993eb7e193c5d8a49f8e88412b6c5378bc9cd1ab026dc9896e2c5ea31bf320e5",
    str(MALECNS_ENGINE.resolve()): "e3a3cd922713aef715247a6bfc23fa7f250803042688a04de0cf3c709ed4aa1b",
    str(WORKER_CONFIG_SOURCE.resolve()): "4838331e3a0bdf142cfd98a2db3657d0eee621ab35f829e3be828300ce08fcf6",
    str(DECODER.resolve()): "3d4a2c608c83cbfaffdd4ea901b72b1a1174e8fdcd9c1de82bd95400184e346b",
    str(BASELINE_SOURCE.resolve()): "25543795ad983ba78ee244d9e551ff4a460024a2cf65ebfb004efd63f69f4a39",
}

BODY_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
WORKER_PYTHON = Path("/usr/bin/python3.12")
FRAME_COUNT = 800
FRAME_DT_S = 0.005
ONSET_S = 1.0
STIMULUS_DURATION_S = 3.0
STATIC_WARMUP_S = 1.0
CONTROL_DT_S = 0.001
CONTROL_TICKS_PER_FRAME = 5
NATIVE_DT_S = 0.0001
NATIVE_STEPS_PER_TICK = 10
DURATION_S = FRAME_COUNT * FRAME_DT_S
BODY_SEED = 1729
NEURAL_SEED = 1729
P_DEADBAND = 1e-6
BASELINE_RATES_HZ = {"DNp15/L": 0.0, "DNp15/R": 142.5}
NORMALIZATION_HZ = 142.5
MAX_VRAM_GIB = 8.0
CASE_RUNTIME_S = 900
QUEUE_RUNTIME_S = 4 * 60 * 60
RESOURCE_LIMITS = {
    "host_memory_max": "5G",
    "cpu_quota": "200%",
    "runtime_max_seconds_per_case": CASE_RUNTIME_S,
    "runtime_max_seconds_per_queue": QUEUE_RUNTIME_S,
    "worker_max_vram_gib": MAX_VRAM_GIB,
}
GLOBAL_NN_LOCK = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-recorded-retina-neural-ball-harness-20261007-v2/locks/"
    "neural-worker-and-ball-case.lock"
)

PROTOCOL_NAME = "flygym-online-v1"
EXPECTED_BACKENDS = {
    "renderer": "fixed_arena_angular_field_cpu",
    "vision": "flyvis",
    "vision_device": "cuda:0",
    "neural": "triton",
    "neural_device": "cuda",
    "neural_source": "connectome",
    "motor_decoder": "NeuromuscularAdapter",
}
CASES = (
    {
        "case_id": "stripes_static",
        "head_direction_sign": 1,
        "head_angular_speed_deg_s": 0.0,
        "arena_angular_speed_deg_s": 0.0,
        "temporal_frequency_hz": 0.0,
        "interpretation": "stationary C=1, wavelength 60 deg, phase 0",
    },
    {
        "case_id": "stripes_positive60",
        "head_direction_sign": 1,
        "head_angular_speed_deg_s": 60.0,
        "arena_angular_speed_deg_s": -60.0,
        "temporal_frequency_hz": 1.0,
        "interpretation": "head +60 deg/s maps to fixed-arena -60 deg/s",
    },
    {
        "case_id": "stripes_negative60",
        "head_direction_sign": -1,
        "head_angular_speed_deg_s": -60.0,
        "arena_angular_speed_deg_s": 60.0,
        "temporal_frequency_hz": 1.0,
        "interpretation": "head -60 deg/s maps to fixed-arena +60 deg/s",
    },
)
CASE_IDS = tuple(row["case_id"] for row in CASES)

WINDOWS = {
    "absolute_gray_0_1": (0.0, 1.0),
    "absolute_early_stimulus_1_1p5": (1.0, 1.5),
    "absolute_late_stimulus_2_4": (2.0, 4.0),
    "stimulus_relative_0_0p5_absolute_1_1p5": (1.0, 1.5),
    "stimulus_relative_1_3_absolute_2_4": (2.0, 4.0),
    "stimulus_relative_0_3_absolute_1_4": (1.0, 4.0),
}


class RunnerError(RuntimeError):
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
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"expected a JSON object: {path}")
    return value


def _sha_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def _write_jsonl(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def _all_source_paths() -> set[Path]:
    paths = {
        Path(__file__).resolve(), FREE_BODY_REPLAY, AP_MECHANICS_RUNNER,
        REFERENCE_WALK, AP_CONTROLLER, AP_PARENT, V41_CONTROLLER,
        SCRIPTS / "flygym_phase_speed_gait_v3.py",
        SCRIPTS / "flygym_phase_speed_gait_v2.py",
        SCRIPTS / "flygym_phase_speed_gait.py", POSE_IK, REST_CONTRACT,
        ONLINE_POSE_BRIDGE, WORKER, WORKER_PARENT, WORKER_LEGACY_PARENT,
        FLYVIS_ADAPTER, FLYVIS_ADAPTER_PARENT, MALECNS_ENGINE,
        NEUROMUSCULAR_ADAPTER, TWIN_CONTRACTS, POPULATION_SELECTORS,
        ANGULAR_RENDERER, DECODER, DECODER_LEGACY, DIRECTION_CALIBRATION,
        WALK_EFFECTIVE_CONFIG, CONTROLLER_CONFIG, WORKER_CONFIG_SOURCE,
        BASELINE_SOURCE, DIRECTION_CONFIG, FLYVIS_TARGET_DIRECTIONS,
        FLYVIS_CHECKPOINT, FLYVIS_MODEL_META, NEUROMUSCULAR_MAP, SENSORY_MAP,
        MOTOR_CHANNEL_MAP, SCRIPTS / "motor_channel_capture.py",
        UPSTREAM_BODY_MANIFEST, UPSTREAM_BODY_ASSET_HASHES,
    }
    worker_config = read_json(WORKER_CONFIG_SOURCE)
    for key in ("world_config", "scene_manifest", "stimulus_config"):
        value = worker_config.get(key)
        if not isinstance(value, str):
            raise RunnerError(f"worker config lacks path field {key}")
        paths.add(Path(value))
    scene_path = Path(worker_config["scene_manifest"])
    scene = read_json(scene_path)
    asset_root = Path(str(scene.get("asset_root", ""))).expanduser()
    if not asset_root.is_absolute():
        asset_root = (scene_path.parent / asset_root).resolve()
    for asset in scene.get("assets", []):
        if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
            raise RunnerError("worker scene manifest contains a malformed asset entry")
        paths.add((asset_root / asset["path"]).resolve(strict=True))
    for asset_path in (
        Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym/assets/model/neuromechfly/vision.yaml"),
        Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym/assets/model/neuromechfly/rigging.yaml"),
        Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym_demo/complex_terrain/preprogrammed.py"),
        Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl"),
    ):
        paths.add(asset_path)

    # Reuse the hash inventory from the exact V4.1 body cohort whose effective
    # config is the parent of this runner.  Each path is rehashed below; the
    # old manifest is only a completeness index, not runtime evidence.
    body_manifest = read_json(UPSTREAM_BODY_MANIFEST)
    installed_source_hashes = body_manifest.get("installed_source_sha256")
    if not isinstance(installed_source_hashes, dict) or not installed_source_hashes:
        raise RunnerError("V4.1 body manifest lacks installed FlyGym source hashes")
    for raw_path, expected_hash in installed_source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_hash:
            raise RunnerError(f"installed FlyGym source differs from V4.1 parent manifest: {path}")
        paths.add(path)

    body_asset_index = read_json(UPSTREAM_BODY_ASSET_HASHES)
    if not body_asset_index:
        raise RunnerError("V4.1 body manifest has no upstream asset inventory")
    for record in body_asset_index.values():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RunnerError("V4.1 body asset inventory contains a malformed record")
        path = Path(record["path"])
        if (not path.is_file() or int(record.get("bytes", -1)) != path.stat().st_size
                or sha256(path) != record.get("sha256")):
            raise RunnerError(f"FlyGym package asset differs from V4.1 parent manifest: {path}")
        paths.add(path)

    if not CSR_ROOT.is_dir():
        raise RunnerError(f"MaleCNS/FlyVis CSR directory is missing: {CSR_ROOT}")
    csr_files = [path for path in CSR_ROOT.iterdir() if path.is_file()]
    if not csr_files:
        raise RunnerError(f"MaleCNS/FlyVis CSR directory is empty: {CSR_ROOT}")
    paths.update(csr_files)
    return {Path(path).resolve(strict=True) for path in paths}


def _spec_document(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "angular-optomotor-stimulus-spec-1",
        "display_geometry": "fixed_arena_infinite_distance_angular_field",
        "stimulus": {
            "name": f"{case['case_id']}-c1-lambda60-phase0",
            "contrast": 1.0,
            "head_direction_sign": int(case["head_direction_sign"]),
            "wavelength_deg": 60.0,
            "temporal_frequency_hz": float(case["temporal_frequency_hz"]),
            "mean_luminance": 0.5,
            "azimuth_extent_deg": [-135.0, 135.0],
            "elevation_extent_deg": [-45.0, 45.0],
            "onset_s": ONSET_S,
            "duration_s": STIMULUS_DURATION_S,
            "phase_at_onset_rad": 0.0,
        },
    }


def _frozen_source_record(protocol: dict[str, Any], path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    record = protocol.get("source_freeze", {}).get("files", {}).get(str(resolved))
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        raise RunnerError(f"runtime asset is absent from the source freeze: {resolved}")
    return record


def _validate_worker_asset_record(
    asset: Any, expected_path: Path, protocol: dict[str, Any], label: str,
) -> dict[str, Any]:
    if not isinstance(asset, dict) or asset.get("exists") is not True:
        raise RunnerError(f"worker provenance omitted required asset {label}")
    actual_path = Path(str(asset.get("path", ""))).resolve(strict=True)
    frozen_path = Path(expected_path).resolve(strict=True)
    frozen = _frozen_source_record(protocol, frozen_path)
    if actual_path != frozen_path:
        raise RunnerError(f"worker loaded unexpected {label} path: {actual_path} != {frozen_path}")
    if int(asset.get("size_bytes", -1)) != int(frozen["size_bytes"]):
        raise RunnerError(f"worker-loaded {label} size differs from the source freeze")
    declared_hash = asset.get("sha256")
    # The worker intentionally skips hashing files larger than 1 MiB. Rehash
    # their resolved path here; verify_frozen_inputs has already pinned it.
    observed_hash = declared_hash if isinstance(declared_hash, str) else sha256(actual_path)
    if observed_hash != frozen["sha256"]:
        raise RunnerError(f"worker-loaded {label} hash differs from the source freeze")
    return {"path": str(actual_path), "size_bytes": int(frozen["size_bytes"]),
            "sha256": observed_hash}


_SYSTEMD_TIMESPAN_TOKEN = re.compile(
    r"\s*(\d+(?:\.\d+)?)\s*(usec|us|µs|msec|ms|sec|s|min|m|hr|h|day|d|week|w)(?![A-Za-z])",
    re.IGNORECASE,
)
_SYSTEMD_TIMESPAN_USEC = {
    "usec": 1, "us": 1, "µs": 1,
    "msec": 1_000, "ms": 1_000,
    "sec": 1_000_000, "s": 1_000_000,
    "min": 60_000_000, "m": 60_000_000,
    "hr": 3_600_000_000, "h": 3_600_000_000,
    "day": 86_400_000_000, "d": 86_400_000_000,
    "week": 604_800_000_000, "w": 604_800_000_000,
}


def _parse_systemd_timespan_usec(value: str, property_name: str) -> int:
    """Parse finite systemctl-show timespans such as ``2s`` or ``15min``."""
    raw = str(value).strip()
    if not raw or raw.lower() == "infinity":
        raise RunnerError(f"{property_name} must be a finite systemd timespan, got {raw!r}")
    offset = 0
    total_usec = 0.0
    while offset < len(raw):
        match = _SYSTEMD_TIMESPAN_TOKEN.match(raw, offset)
        if match is None:
            raise RunnerError(f"could not parse {property_name} systemd timespan: {raw!r}")
        amount = float(match.group(1))
        unit = match.group(2).lower()
        total_usec += amount * _SYSTEMD_TIMESPAN_USEC[unit]
        offset = match.end()
    rounded = round(total_usec)
    if not math.isfinite(total_usec) or not math.isclose(total_usec, rounded, rel_tol=0.0, abs_tol=1e-6):
        raise RunnerError(f"{property_name} did not resolve to an integer microsecond count: {raw!r}")
    return int(rounded)


def _self_cgroup_v2_path() -> str:
    try:
        rows = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RunnerError(f"could not read /proc/self/cgroup: {exc}") from exc
    for row in rows:
        parts = row.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            path = parts[2]
            if not path.startswith("/") or ".." in path.split("/"):
                raise RunnerError(f"invalid unified cgroup path in /proc/self/cgroup: {path!r}")
            return path
    raise RunnerError("/proc/self/cgroup has no unified cgroup v2 membership")


def _systemd_case_scope_record() -> dict[str, Any]:
    unit = os.environ.get("ONLINE_FREEBODY_CASE_UNIT")
    if not unit:
        raise RunnerError("case execution did not receive its transient systemd unit identity")
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=MemoryMax",
         "--property=CPUQuotaPerSecUSec", "--property=RuntimeMaxUSec",
         "--property=ControlGroup"],
        check=True, capture_output=True, text=True, timeout=10.0,
    )
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    try:
        memory_max = int(properties["MemoryMax"])
        cpu_quota_usec = _parse_systemd_timespan_usec(
            properties["CPUQuotaPerSecUSec"], "CPUQuotaPerSecUSec",
        )
        runtime_max_usec = _parse_systemd_timespan_usec(
            properties["RuntimeMaxUSec"], "RuntimeMaxUSec",
        )
        control_group = properties["ControlGroup"]
    except (KeyError, ValueError) as exc:
        raise RunnerError(f"could not read actual case-unit resource caps: {properties}") from exc
    if (not control_group.startswith("/") or control_group == "/"
            or ".." in control_group.split("/")):
        raise RunnerError(f"systemd reported an invalid ControlGroup path: {control_group!r}")
    current_cgroup = _self_cgroup_v2_path()
    cgroup_match = (
        current_cgroup == control_group
        or current_cgroup.startswith(control_group.rstrip("/") + "/")
    )
    if not cgroup_match:
        raise RunnerError(
            f"current process is outside the requested systemd unit cgroup: "
            f"{current_cgroup!r} != {control_group!r}"
        )
    if (not 0 < memory_max <= 5 * 1024**3
            or not 0 < cpu_quota_usec <= 2_000_000
            or not 0 < runtime_max_usec <= CASE_RUNTIME_S * 1_000_000):
        raise RunnerError(f"actual case-unit resource caps differ from protocol: {properties}")
    return {
        "unit": unit,
        "MemoryMax_bytes": memory_max,
        "CPUQuotaPerSecUSec": cpu_quota_usec,
        "RuntimeMaxUSec": runtime_max_usec,
        "ControlGroup": control_group,
        "proc_self_cgroup_v2": current_cgroup,
        "cgroup_membership_matches_unit": cgroup_match,
        "queried_by": "systemctl --user show while case unit was active",
        "environment_marker_is_not_cap_evidence": True,
    }


def _expected_calibration_scope(config: dict[str, Any]) -> dict[str, Any]:
    mode = config["baseline_policy"]["modes"]["calibrated_static"]
    family = mode["application_scope"]
    source = mode["calibration_source"]
    return {
        "visual_family": family["visual_family"],
        "pose_mode": "free_body_online_pose",
        "display_geometry": family["display_geometry"],
        "contrast": family["contrast"],
        "wavelength_deg": family["wavelength_deg"],
        "calibration_source_cohort_id": source["source_cohort_id"],
        "runtime_source_identity": source["source_cohort_identity"],
    }


def prepare() -> dict[str, Any]:
    if BASE.exists():
        raise FileExistsError(f"refusing to overwrite online free-body stripe bundle: {BASE}")
    if not BODY_PYTHON.is_file() or not WORKER_PYTHON.is_file():
        raise RunnerError("registered FlyGym or worker Python executable is missing")
    source_records = {str(path): _sha_record(path) for path in sorted(_all_source_paths())}
    body_asset_index = read_json(UPSTREAM_BODY_ASSET_HASHES)
    for name, expected in LOCKED_SHA256.items():
        actual = source_records.get(name, {}).get("sha256")
        if actual != expected:
            raise RunnerError(f"locked source changed: {name}: {actual} != {expected}")

    PREPARATION.mkdir(parents=True, exist_ok=False)
    SPECS.mkdir(parents=True, exist_ok=False)

    worker_cfg = read_json(WORKER_CONFIG_SOURCE)
    if worker_cfg.get("max_frames") != 400:
        raise RunnerError("the source worker config template no longer has the reviewed 400-frame setting")
    selectors = worker_cfg.get("diagnostic_populations", {})
    if not all(key in selectors for key in ("DNp15/L", "DNp15/R")):
        raise RunnerError("worker configuration does not report bilateral DNp15 rates")
    if not math.isclose(float(read_json(Path(worker_cfg["world_config"]))["experiment"]["neural_gain"]),
                        1.2, rel_tol=0.0, abs_tol=1e-12):
        raise RunnerError("worker experiment config must pin neural gain 1.2")

    effective_worker_cfg = dict(worker_cfg)
    effective_worker_cfg["max_frames"] = FRAME_COUNT
    effective_worker_cfg["static_warmup_s"] = STATIC_WARMUP_S
    effective_worker_cfg_path = PREPARATION / "worker-config-online-angular.json"
    write_json(effective_worker_cfg_path, effective_worker_cfg)
    walk_copy_path = PREPARATION / "walk-effective-config.json"
    controller_copy_path = PREPARATION / "controller-config.json"
    baseline_copy_path = PREPARATION / "static-baseline-payload.json"
    direction_copy_path = PREPARATION / "flygym-direction-calibration-v1.json"
    for source, target in (
        (WALK_EFFECTIVE_CONFIG, walk_copy_path),
        (CONTROLLER_CONFIG, controller_copy_path),
        (BASELINE_SOURCE, baseline_copy_path),
        (DIRECTION_CONFIG, direction_copy_path),
    ):
        target.write_bytes(Path(source).read_bytes())

    baseline_payload = read_json(baseline_copy_path)
    rates = baseline_payload.get("cells", {})
    if (rates.get("DNp15/L", {}).get("rate_hz") != 0.0
            or rates.get("DNp15/R", {}).get("rate_hz") != 142.5
            or baseline_payload.get("moving_case_data_used") is not False
            or baseline_payload.get("observed_yaw_used") is not False):
        raise RunnerError("calibrated_static payload does not match its frozen 0/142.5 Hz provenance")

    prepared_files = {
        str(path.resolve()): _sha_record(path)
        for path in (effective_worker_cfg_path, walk_copy_path, controller_copy_path,
                     baseline_copy_path, direction_copy_path)
    }
    case_records: list[dict[str, Any]] = []
    for case in CASES:
        spec_path = SPECS / f"{case['case_id']}.json"
        write_json(spec_path, _spec_document(case))
        spec_hash = sha256(spec_path)
        case_records.append({
            **case,
            "spec_path": str(spec_path.resolve()),
            "spec_sha256": spec_hash,
            "frame_count": FRAME_COUNT,
            "duration_s": DURATION_S,
            "gray_window_s": [0.0, ONSET_S],
            "stimulus_window_s": [ONSET_S, DURATION_S],
        })
        prepared_files[str(spec_path.resolve())] = _sha_record(spec_path)

    direction_config = read_json(direction_copy_path)
    scope_request = _expected_calibration_scope(direction_config)
    sys.path.insert(0, str(SCRIPTS))
    from flygym_direction_calibration import (
        UnsupportedBaselineError, validate_static_stripe_scope,
    )
    try:
        # This deliberately invokes the existing scope checker. The result is
        # expected to reject the walking/full-pose extrapolation; it is kept as
        # an explicit candidate limitation instead of bypassed or relabelled.
        scope_validation: dict[str, Any] = {
            "status": "validated",
            "result": validate_static_stripe_scope(scope_request, direction_config),
        }
    except UnsupportedBaselineError as exc:
        mode = direction_config["baseline_policy"]["modes"]["calibrated_static"]
        if mode.get("status") != "calibrated_scope_limited":
            raise RunnerError("calibrated_static source is no longer scope-limited calibration") from exc
        family = mode["application_scope"]
        source = mode["calibration_source"]
        expected_scope = {
            "visual_family": family["visual_family"],
            "pose_mode": family["pose_mode"],
            "display_geometry": family["display_geometry"],
            "contrast": family["contrast"],
            "wavelength_deg": family["wavelength_deg"],
            "calibration_source_cohort_id": source["source_cohort_id"],
            "runtime_source_identity": source["source_cohort_identity"],
        }
        mismatches = {
            key: (scope_request.get(key), value)
            for key, value in expected_scope.items()
            if scope_request.get(key) != value
        }
        if set(mismatches) != {"pose_mode"}:
            raise RunnerError(
                "static-baseline rejection was not limited to the documented pose_mode extrapolation: "
                f"{mismatches}"
            ) from exc
        scope_validation = {
            "status": "candidate_outside_calibrated_scope",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "requested_scope": scope_request,
            "calibrated_scope": direction_config["baseline_policy"]["modes"]["calibrated_static"]["application_scope"],
            "calibrated_static_baseline_still_used": True,
            "claim": "candidate only; no calibration pass and no biological follow-success claim",
        }

    # Parse each frozen schema-1 stimulus through the worker's pure loader.
    # This validates its exact schema and 5 ms clock without constructing any
    # vision/neural backend or starting a worker.
    sys.path.insert(0, str(SCRIPTS))
    from flygym_online_neural_worker_eye_parity_state_v2 import _load_angular_stimulus_spec
    for case in case_records:
        spec_path = Path(case["spec_path"])
        stimulus, _, raw = _load_angular_stimulus_spec(spec_path)
        if (raw != spec_path.read_bytes() or stimulus.onset_s != ONSET_S
                or stimulus.duration_s != STIMULUS_DURATION_S
                or stimulus.end_s != DURATION_S
                or stimulus.contrast != 1.0 or stimulus.wavelength_deg != 60.0
                or stimulus.phase_at_onset_rad != 0.0
                or stimulus.temporal_frequency_hz != case["temporal_frequency_hz"]
                or stimulus.head_direction_sign != case["head_direction_sign"]):
            raise RunnerError(f"prepared angular stimulus failed worker schema/clock validation: {spec_path}")

    protocol = {
        "schema_version": "flygym-free-body-online-stripes-protocol-v1",
        "status": "prepared_not_executed",
        "classification": "candidate true-online pose-conditioned angular vision plus free-ground V4.1/AP control",
        "evidence_boundary": {
            "calibrated_static_reference": "historical fixed-head stationary C=1, lambda=60 deg baseline bL=0, bR=142.5 Hz",
            "online_pose_scope": "free-body walking/full-pose is outside that fixed-head calibration family",
            "scope_checker": scope_validation,
            "calibration_pass": False,
            "biological_follow_success_claim": False,
            "yaw_coordinate": "thorax local +X projected into native FlyGym world XY; report sign as measured",
        },
        "source_freeze": {
            "files": source_records,
            "locked_known_hashes": LOCKED_SHA256,
            "source_freeze_sha256_semantics": "file SHA-256 of source-freeze.json bytes",
            "upstream_body_asset_reference": str(UPSTREAM_BODY_ASSET_HASHES),
            "upstream_body_package_asset_count": len(body_asset_index),
            "male_cns_csr_root": str(CSR_ROOT.resolve()),
        },
        "prepared_files": prepared_files,
        "prepared_inputs": {
            "worker_config": str(effective_worker_cfg_path.resolve()),
            "walk_effective_config": str(walk_copy_path.resolve()),
            "controller_config": str(controller_copy_path.resolve()),
            "baseline_payload": str(baseline_copy_path.resolve()),
            "direction_calibration": str(direction_copy_path.resolve()),
        },
        "runtime": {
            "body_python": str(BODY_PYTHON),
            "worker_python": str(WORKER_PYTHON),
            "worker_script": str(WORKER),
            "worker_config": str(effective_worker_cfg_path.resolve()),
            "worker_renderer_mode": "fixed_arena_angular_field_cpu",
            "native_recorded_columns": False,
            "stimulus_source": "live angular spec selected at worker startup; current body pose supplied every 5 ms",
            "worker_neural_gain": 1.2,
            "worker_neural_backend_required": "triton/cuda",
            "worker_vision_backend_required": "flyvis/cuda:0",
            "worker_max_vram_gib": MAX_VRAM_GIB,
            "deterministic_vision": True,
            "deterministic_neural_delivery": True,
            "neural_intervention": False,
            "mechanosensory_feedback": "unsupported and not injected",
            "selected_state_capture": "40 first-frame debug dumps of full-network v/g/current/probability and related engine arrays, plus periodic full-network Vm summaries; timing-equivalence is not claimed",
        },
        "body": {
            "session": str(REFERENCE_WALK),
            "free_ground_reference_replay": str(FREE_BODY_REPLAY),
            "controller": str(AP_CONTROLLER),
            "body_seed": BODY_SEED,
            "initial_neural_seed": NEURAL_SEED,
            "duration_s": DURATION_S,
            "frame_count": FRAME_COUNT,
            "frame_dt_s": FRAME_DT_S,
            "control_ticks_per_frame": CONTROL_TICKS_PER_FRAME,
            "control_dt_s": CONTROL_DT_S,
            "native_dt_s": NATIVE_DT_S,
            "native_steps_per_control_tick": NATIVE_STEPS_PER_TICK,
            "expected_control_ticks_per_case": FRAME_COUNT * CONTROL_TICKS_PER_FRAME,
            "expected_physics_steps_per_case": FRAME_COUNT * CONTROL_TICKS_PER_FRAME * NATIVE_STEPS_PER_TICK,
            "expected_native_samples_per_case": FRAME_COUNT * CONTROL_TICKS_PER_FRAME * NATIVE_STEPS_PER_TICK + 1,
            "world": "flat-ground free body; existing V4.1/AP position-servo gait; no root writes or external wrench",
            "initial_body_command": {"P_control": 0.0, "signed_d": 0.0, "source_neural_seq": None},
        },
        "causal_clock": {
            "worker_frame_time_s": "seq * 0.005; pose sampled at current body time",
            "neural_window": "[frame_time_s, frame_time_s + 0.005)",
            "body_window": "same 5 ms interval, five 1 ms ticks",
            "command_latency_frames": 1,
            "rule": "apply previous neural-frame decoder result to current body frame; current response becomes next frame's command",
            "seq0": "first body frame receives the explicit neutral P=0,d=0 startup command",
            "terminal": "seq799 response is decoded and saved but has no following body frame; record as pending and unapplied",
            "timestamps_saved": ["body_pose_time_s", "worker_window_start_s", "worker_window_end_s", "decoder_output_time_s", "body_action_start_s", "body_action_end_s"],
        },
        "decoder": {
            "implementation": str(DECODER),
            "mode": "calibrated_static",
            "baseline_payload": str(baseline_copy_path.resolve()),
            "baseline_rates_hz": BASELINE_RATES_HZ,
            "normalization_hz": NORMALIZATION_HZ,
            "p_deadband": P_DEADBAND,
            "ema_tau_s": 0.150,
            "frame_dt_s": FRAME_DT_S,
            "inputs_only": ["current worker leg_power", "current worker population_rates_hz DNp15/L", "current worker population_rates_hz DNp15/R"],
            "P_control": "DecoderFrame.control_power applied symmetrically to both CPG sides",
            "signed_d": "DecoderFrame.clipped_opponent_signal applied through existing AP config helper",
            "stimulus_label_or_case_branching": False,
            "saved_offline_Pd_schedule_read": False,
            "worker_native_bilateral_drive_applied": False,
            "static_baseline_scope_extrapolation": "candidate only; checker result above is retained",
        },
        "stimuli": {
            "spec_schema_version": "angular-optomotor-stimulus-spec-1",
            "display_geometry": "fixed_arena_infinite_distance_angular_field",
            "common": {
                "contrast": 1.0,
                "wavelength_deg": 60.0,
                "phase_at_onset_rad": 0.0,
                "mean_luminance": 0.5,
                "onset_s": ONSET_S,
                "duration_s": STIMULUS_DURATION_S,
                "gray_interval_s": [0.0, ONSET_S],
                "stimulus_interval_s": [ONSET_S, DURATION_S],
            },
            "conditions": case_records,
            "timeline_delta_from_legacy_static_reference": {
                "legacy_reference_onset_duration_s": [5.0, 5.0],
                "new_online_schedule_onset_duration_s": [ONSET_S, STIMULUS_DURATION_S],
                "reason": "the 4 s body study uses 1 s gray then 3 s stimulus; no matching 1+3 s angular specs existed, so new schema-1 specs are saved and hash-pinned",
            },
            "head_to_arena_sign": "head +60 deg/s maps to fixed-arena -60 deg/s; head -60 deg/s maps to fixed-arena +60 deg/s; no output-driven sign flip",
        },
        "analysis_windows": {
            "units": "absolute seconds in the 0..4 s trace",
            "windows_s": {key: list(value) for key, value in WINDOWS.items()},
            "yaw_source": "unwrapped thorax_heading_yaw_rad from local thorax +X heading projected into native world XY",
            "ball_omega_z_substitution": False,
            "direction_interpretation": "engineering measurements only; no automatic expected-sign assertion or biological success label",
        },
        "artifacts_per_case": [
            "run-record.json", "online-frames.jsonl", "control-ticks.jsonl",
            "worker-requests.jsonl", "worker-protocol-stdout.jsonl", "worker-stderr.log",
            "trajectory.npz", "contacts.npz", "manifest.json", "window-summary.json",
            "online-integrity.json", "worker/frames.jsonl", "worker/provenance.json",
            "worker/eye_columns.npz", "worker/visual_drive.npz", "worker/muscle_channels.npz",
            "worker/neural_state_diagnostic.npz", "worker/debug-neural-state/",
        ],
        "locks": {
            "queue_lock": str((BASE / "queue.lock").resolve()),
            "shared_global_neural_lock": str(GLOBAL_NN_LOCK),
            "policy": "nonblocking exclusive flock; queue lock serializes this cohort, shared lock excludes other neural/body batches",
        },
        "resource_limits": RESOURCE_LIMITS,
        "resource_enforcement": {
            "case_launcher": "systemd transient unit with MemoryMax=5G, CPUQuota=200%, RuntimeMaxSec=900",
            "case_child_environment_marker_is_not_enforcement_proof": True,
            "case_runtime_verification": "read actual transient unit properties and worker reported cgroup memory limit",
            "queue_hard_limit": "external root-owned systemd service; runner also checks a 4 hour monotonic deadline",
        },
        "queue_order": list(CASE_IDS),
        "queue_fail_policy": "stop on first failed case; keep raw partial body trace and worker logs; no retry or silent case skip",
        "gpu_or_simulation_started_during_prepare": False,
    }
    source_freeze_path = PREPARATION / "source-freeze.json"
    protocol_path = BASE / "protocol.json"
    write_json(source_freeze_path, {
        "schema_version": "flygym-free-body-online-stripes-source-freeze-v1",
        "status": "frozen_at_prepare_before_any_online_run",
        "files": source_records,
        "locked_known_hashes": LOCKED_SHA256,
        "upstream_body_parent": {
            "manifest_path": str(UPSTREAM_BODY_MANIFEST),
            "asset_inventory_path": str(UPSTREAM_BODY_ASSET_HASHES),
            "package_asset_count": len(body_asset_index),
            "csr_path": str(CSR_ROOT.resolve()),
            "csr_file_count": len([path for path in source_records if str(CSR_ROOT) in path]),
        },
    })
    protocol["source_freeze"]["path"] = str(source_freeze_path.resolve())
    protocol["source_freeze"]["sha256"] = sha256(source_freeze_path)
    prepare_manifest_path = BASE / "prepare-manifest.json"
    protocol["prepare_manifest_path"] = str(prepare_manifest_path.resolve())
    write_json(protocol_path, protocol)
    prepare_manifest = {
        "schema_version": "flygym-free-body-online-stripes-prepare-manifest-v1",
        "status": "prepared_not_executed",
        "runner": _sha_record(Path(__file__)),
        "source_freeze": _sha_record(source_freeze_path),
        "protocol": _sha_record(protocol_path),
        "prepared_files": prepared_files,
        "case_count": len(case_records),
        "expected_worker_frames": FRAME_COUNT,
        "expected_body_control_ticks_per_case": FRAME_COUNT * CONTROL_TICKS_PER_FRAME,
        "scope_checker_status": scope_validation["status"],
        "gpu_worker_started": False,
        "body_physics_steps": 0,
    }
    write_json(prepare_manifest_path, prepare_manifest)
    return {
        "status": "prepared_not_executed",
        "protocol_path": str(protocol_path),
        "source_freeze_path": str(source_freeze_path),
        "prepare_manifest_path": str(prepare_manifest_path),
        "scope_checker_status": scope_validation["status"],
        "case_count": len(case_records),
    }


def verify_frozen_inputs(protocol: dict[str, Any]) -> None:
    freeze_path = Path(protocol["source_freeze"]["path"])
    if sha256(freeze_path) != protocol["source_freeze"]["sha256"]:
        raise RunnerError("source-freeze file changed after prepare")
    freeze = read_json(freeze_path)
    for raw_path, record in freeze["files"].items():
        if sha256(Path(raw_path)) != record["sha256"]:
            raise RunnerError(f"frozen source changed: {raw_path}")
    for raw_path, record in protocol["prepared_files"].items():
        if sha256(Path(raw_path)) != record["sha256"]:
            raise RunnerError(f"prepared input changed: {raw_path}")


def _validate_worker_ready(ready: dict[str, Any], case: dict[str, Any], protocol: dict[str, Any]) -> None:
    expected = {
        "protocol": PROTOCOL_NAME,
        "type": "ready",
        "source": "connectome",
        "stimulus_condition": "angular",
        "stimulus_mode": "fixed_arena_angular_field",
        "seed": NEURAL_SEED,
        "max_frames": FRAME_COUNT,
        "n_columns_per_eye": 721,
        "neural_steps_per_frame": 10,
        "male_cns_initial_step_index": 0,
        "male_cns_steps_during_visual_warmup": 0,
    }
    for key, value in expected.items():
        if ready.get(key) != value:
            raise RunnerError(f"worker ready mismatch {key}: {ready.get(key)!r} != {value!r}")
    for key, value in (("frame_dt_s", FRAME_DT_S), ("neural_tick_dt_s", 0.0005),
                       ("stimulus_onset_s", ONSET_S), ("static_warmup_s", STATIC_WARMUP_S)):
        if not math.isclose(float(ready.get(key, math.nan)), value, rel_tol=0.0, abs_tol=1e-12):
            raise RunnerError(f"worker ready clock mismatch {key}")
    backends = ready.get("effective_backends")
    if not isinstance(backends, dict) or backends != EXPECTED_BACKENDS:
        raise RunnerError(f"worker backend identity mismatch: {backends!r}")
    parameters = ready.get("malecns_model_parameters", {})
    effective_gain = float(np.float32(1.2))
    if (not isinstance(parameters, dict)
            or float(parameters.get("gain", math.nan)) != effective_gain
            or float(parameters.get("gain_configured_by_world_experiment", math.nan)) != 1.2):
        raise RunnerError("effective MaleCNS neural gain is not exactly 1.2")
    if ready.get("neural_intervention", {}).get("enabled") is not False:
        raise RunnerError("neural intervention must be disabled")
    if ready.get("standard_visual_input", {}).get("native_recorded_columns_used") is not False:
        raise RunnerError("angular live rendering must not bypass pose with recorded columns")
    if ready.get("deterministic_vision", {}).get("requested") is not True:
        raise RunnerError("deterministic FlyVis vision was not requested")
    if ready.get("deterministic_vision", {}).get("effective_flags", {}).get("settings_applied") is not True:
        raise RunnerError("deterministic FlyVis settings were not applied")
    if ready.get("deterministic_neural_delivery", {}).get("requested") is not True or ready.get(
            "deterministic_neural_delivery", {}).get("enabled") is not True:
        raise RunnerError("deterministic neural delivery did not activate")
    angular = ready.get("angular_stimulus")
    if not isinstance(angular, dict):
        raise RunnerError("worker omitted angular stimulus metadata")
    stimulus = next(row for row in protocol["stimuli"]["conditions"] if row["case_id"] == case["case_id"])
    for key, value in (
        ("spec_sha256", case["spec_sha256"]),
        ("head_direction_sign", case["head_direction_sign"]),
        ("head_angular_speed_deg_s", case["head_angular_speed_deg_s"]),
        ("arena_angular_speed_deg_s", case["arena_angular_speed_deg_s"]),
        ("wavelength_deg", 60.0),
        ("temporal_frequency_hz", case["temporal_frequency_hz"]),
        ("contrast_study_parameter_c", 1.0),
        ("onset_s", ONSET_S),
        ("duration_s", STIMULUS_DURATION_S),
    ):
        if isinstance(value, float):
            if not math.isclose(float(angular.get(key, math.nan)), value, rel_tol=0.0, abs_tol=1e-12):
                raise RunnerError(f"angular worker metadata mismatch {key}")
        elif angular.get(key) != value:
            raise RunnerError(f"angular worker metadata mismatch {key}")
    if angular.get("phase_at_onset_rad") not in (None, 0.0):
        raise RunnerError("angular stimulus phase origin is not zero")
    if not 0 < int(ready.get("host_memory_limit_bytes", 0)) <= 5 * 1024**3:
        raise RunnerError("worker effective cgroup memory cap exceeds 5 GiB or is unavailable")

    assets = ready.get("source_assets", {})
    if not isinstance(assets, dict):
        raise RunnerError("worker omitted loaded source asset metadata")
    worker_assets = assets.get("worker_source", {})
    _validate_worker_asset_record(worker_assets.get("candidate"), WORKER, protocol, "worker source")
    _validate_worker_asset_record(
        worker_assets.get("state_observer_parent"), WORKER_PARENT, protocol, "worker observer parent",
    )
    _validate_worker_asset_record(
        worker_assets.get("legacy"), WORKER_LEGACY_PARENT, protocol, "worker legacy parent",
    )
    flyvis = assets.get("flyvis", {})
    _validate_worker_asset_record(
        flyvis.get("adapter_source"), FLYVIS_ADAPTER, protocol, "FlyVis adapter",
    )
    _validate_worker_asset_record(
        flyvis.get("legacy_adapter_source"), FLYVIS_ADAPTER_PARENT, protocol,
        "FlyVis legacy adapter",
    )
    malecns_assets = assets.get("malecns", {})
    _validate_worker_asset_record(
        malecns_assets.get("neuromuscular_map"), NEUROMUSCULAR_MAP, protocol,
        "neuromuscular map",
    )
    _validate_worker_asset_record(
        malecns_assets.get("sensory_map"), SENSORY_MAP, protocol, "MaleCNS sensory map",
    )
    _validate_worker_asset_record(
        assets.get("motor_channel_mapping"), MOTOR_CHANNEL_MAP, protocol,
        "motor-channel mapping",
    )
    _validate_worker_asset_record(
        assets.get("motor_channel_observer"), SCRIPTS / "motor_channel_capture.py",
        protocol, "motor-channel capture observer",
    )
    csr_files = malecns_assets.get("csr_files")
    if not isinstance(csr_files, dict) or not csr_files:
        raise RunnerError("worker did not report the loaded MaleCNS CSR assets")
    for name, asset_record in csr_files.items():
        _validate_worker_asset_record(asset_record, CSR_ROOT / name, protocol, f"MaleCNS CSR {name}")

    # The worker config itself is a newly prepared copy; the remaining config
    # files and models are pinned to their actual source paths in source-freeze.
    worker_config_path = Path(protocol["runtime"]["worker_config"]).resolve(strict=True)
    worker_config_record = assets.get("worker_config", {})
    expected_worker_config = protocol["prepared_files"].get(str(worker_config_path), {})
    if (worker_config_record.get("path") != str(worker_config_path)
            or worker_config_record.get("sha256") != expected_worker_config.get("sha256")):
        raise RunnerError("worker loaded a different prepared worker config")
    worker_cfg = read_json(worker_config_path)
    for key in ("world_config", "scene_manifest", "stimulus_config"):
        _validate_worker_asset_record(
            assets.get(key), Path(worker_cfg[key]), protocol, f"worker {key}",
        )
    visual_scene_assets = assets.get("visual_scene_assets", {})
    if not isinstance(visual_scene_assets, dict):
        raise RunnerError("worker provenance omitted scene asset list")
    for label, asset_record in visual_scene_assets.items():
        if not isinstance(asset_record, dict) or not isinstance(asset_record.get("path"), str):
            raise RunnerError(f"worker scene asset record is malformed: {label}")
        _validate_worker_asset_record(
            asset_record, Path(asset_record["path"]), protocol, f"visual scene asset {label}",
        )

    flyvis = assets.get("flyvis", {})
    model_files = flyvis.get("model_files", {})
    if not isinstance(model_files, dict):
        raise RunnerError("worker omitted FlyVis model files")
    for name, path in (("best_chkpt", FLYVIS_CHECKPOINT), ("_meta.yaml", FLYVIS_MODEL_META)):
        _validate_worker_asset_record(model_files.get(name), path, protocol, f"FlyVis model {name}")
    _validate_worker_asset_record(
        flyvis.get("target_mapping"), FLYVIS_TARGET_DIRECTIONS, protocol, "FlyVis target mapping",
    )
    # The actual motor-capture manifest has no "enabled" field. When
    # disabled it has enabled=False; an active startup capture binds
    # its frame allocation, channel count, mapping and empty initial counter.
    motor_capture = ready.get("motor_channel_capture")
    if (not isinstance(motor_capture, dict)
            or motor_capture.get("enabled") is False
            or motor_capture.get("max_frames") != FRAME_COUNT
            or motor_capture.get("frames_captured") != 0
            or motor_capture.get("total_leg_motor_cells") != 381):
        raise RunnerError("motor capture startup allocation is missing or inconsistent")
    _validate_worker_asset_record(
        {"path": motor_capture.get("mapping_path"), "exists": True,
         "size_bytes": MOTOR_CHANNEL_MAP.stat().st_size,
         "sha256": motor_capture.get("mapping_config_sha256")},
        MOTOR_CHANNEL_MAP, protocol, "motor-capture startup map",
    )
    if ready.get("neural_state_diagnostic_capture", {}).get("enabled") is not True:
        raise RunnerError("raw neural-state diagnostics were not captured")
    diagnostic_arrays = set(ready["neural_state_diagnostic_capture"].get("selected_arrays", []))
    if not {"post_final_tick_synaptic_g_state", "held_external_current_mV_per_ms",
            "held_external_probability"}.issubset(diagnostic_arrays):
        raise RunnerError("selected neural-state g/current/probability capture is incomplete")
    if ready.get("debug_neural_state", {}).get("requested_frame_captures") != 40:
        raise RunnerError("worker did not prepare the pinned 40-frame whole-network state capture")


def _validate_worker_provenance(
    worker_dir: Path, ready: dict[str, Any], stop_ack: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    provenance_path = worker_dir / "provenance.json"
    integrity_path = worker_dir / "integrity.json"
    provenance = read_json(provenance_path)
    integrity = read_json(integrity_path)
    caps = provenance.get("resource_caps", {})
    if float(caps.get("max_vram_gib_per_backend", math.nan)) != MAX_VRAM_GIB:
        raise RunnerError("worker provenance max_vram_gib_per_backend differs from the 8 GiB command cap")
    host_cap = int(caps.get("host_memory_limit_bytes", 0))
    if not 0 < host_cap <= 5 * 1024**3 or host_cap != int(ready["host_memory_limit_bytes"]):
        raise RunnerError("worker provenance host-memory cap differs from worker ready")
    if provenance.get("effective_backends") != EXPECTED_BACKENDS:
        raise RunnerError("worker provenance backend identity differs from ready")
    params = provenance.get("malecns", {}).get("model_parameters", {})
    if (float(params.get("gain", math.nan)) != float(np.float32(1.2))
            or float(params.get("gain_configured_by_world_experiment", math.nan)) != 1.2):
        raise RunnerError("worker provenance MaleCNS gain differs from the approved float32/configured values")
    if provenance.get("source_assets") != ready.get("source_assets"):
        raise RunnerError("worker provenance loaded assets changed after ready")
    if provenance.get("stimulus_mode") != "fixed_arena_angular_field":
        raise RunnerError("worker provenance did not preserve angular renderer mode")
    visual_capture = provenance.get("visual_capture", {})
    if visual_capture.get("frames_saved") != FRAME_COUNT:
        raise RunnerError("worker did not persist all retinal frames")
    if provenance.get("visual_drive_capture", {}).get("saved_frames") != FRAME_COUNT:
        raise RunnerError("worker did not persist all visual-drive frames")
    if provenance.get("motor_channel_capture", {}).get("frames_captured") != FRAME_COUNT:
        raise RunnerError("worker did not persist all motor-channel frames")
    neural_state = provenance.get("neural_state_diagnostic", {})
    if not neural_state.get("enabled") or neural_state.get("frames_saved") != FRAME_COUNT:
        raise RunnerError("worker did not persist every selected neural-state diagnostic row")
    debug_state = provenance.get("debug_neural_state", {})
    if debug_state.get("requested_frame_captures") != 40 or len(debug_state.get("captured_frames", [])) != 40:
        raise RunnerError("worker did not persist the requested 40 neural debug frame captures")
    debug_records = [debug_state.get("initial_capture"), *debug_state.get("captured_frames", [])]
    for debug_record in debug_records:
        if not isinstance(debug_record, dict):
            raise RunnerError("worker omitted an initial or per-frame full neural-state capture record")
        debug_path = Path(str(debug_record.get("path", "")))
        if (not debug_path.is_file() or int(debug_record.get("size_bytes", -1)) != debug_path.stat().st_size
                or debug_record.get("sha256") != sha256(debug_path)):
            raise RunnerError("worker full neural-state debug capture failed its path/size/hash check")
    motor_capture = provenance.get("motor_channel_capture", {})
    mapping_path = Path(str(motor_capture.get("mapping_path", "")))
    _validate_worker_asset_record(
        {"path": str(mapping_path), "exists": mapping_path.is_file(),
         "size_bytes": mapping_path.stat().st_size if mapping_path.is_file() else -1,
         "sha256": motor_capture.get("mapping_config_sha256")},
        MOTOR_CHANNEL_MAP, protocol, "motor channel map",
    )

    worker_integrity = integrity.get("files", {})
    for name in ("frames.jsonl", "provenance.json", "eye_columns.npz", "visual_drive.npz",
                 "muscle_channels.npz", "neural_state_diagnostic.npz"):
        file_record = worker_integrity.get(name)
        path = worker_dir / name
        if not isinstance(file_record, dict) or not path.is_file():
            raise RunnerError(f"worker integrity omitted required artifact {name}")
        if file_record.get("sha256") != sha256(path):
            raise RunnerError(f"worker integrity hash mismatch for {name}")
    if (stop_ack.get("artifact_checksums", {}).get("provenance_json_sha256")
            != sha256(provenance_path)):
        raise RunnerError("worker stop acknowledgement does not bind the final provenance file")
    if (stop_ack.get("artifact_checksums", {}).get("integrity_json_sha256")
            != sha256(integrity_path)):
        raise RunnerError("worker stop acknowledgement does not bind the final integrity file")
    if integrity.get("frame_count") != FRAME_COUNT or integrity.get("last_seq") != FRAME_COUNT - 1:
        raise RunnerError("worker integrity frame count is incomplete")
    return {
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256(provenance_path),
        "integrity_path": str(integrity_path),
        "integrity_sha256": sha256(integrity_path),
        "resource_caps": caps,
        "retinal_frames": visual_capture.get("frames_saved"),
        "visual_drive_frames": provenance["visual_drive_capture"]["saved_frames"],
        "motor_channel_frames": provenance["motor_channel_capture"]["frames_captured"],
        "neural_state_rows": neural_state.get("frames_saved"),
        "debug_state_frames": len(debug_state.get("captured_frames", [])),
        "debug_initial_capture": debug_state["initial_capture"]["path"],
    }


def _worker_command(case: dict[str, Any], output_dir: Path, protocol: dict[str, Any]) -> list[str]:
    return [
        str(WORKER_PYTHON), str(WORKER),
        "--config", str(Path(protocol["runtime"]["worker_config"])),
        "--stimulus-condition", "angular",
        "--angular-stimulus-spec", case["spec_path"],
        "--static-warmup-s", str(STATIC_WARMUP_S),
        "--seed", str(NEURAL_SEED),
        "--max-frames", str(FRAME_COUNT),
        "--neural-device", "cuda", "--neural-backend", "triton",
        "--vision-device", "cuda:0", "--max-vram-gib", str(MAX_VRAM_GIB),
        "--deterministic-vision", "--deterministic-neural-delivery",
        "--capture-motor-channels", "--capture-visual-drive",
        "--capture-neural-state-diagnostic", "--debug-neural-state-frames", "40",
        "--out", str(output_dir),
    ]


def _protocol_read(process: subprocess.Popen[str], timeout_s: float) -> tuple[str, dict[str, Any]]:
    assert process.stdout is not None
    readable, _, _ = select.select([process.stdout], [], [], timeout_s)
    if not readable:
        return_code = process.poll()
        if return_code is not None:
            raise RunnerError(
                f"neural worker exited with status {return_code} before another protocol record"
            )
        raise TimeoutError(f"worker did not produce a protocol record in {timeout_s:g} seconds")
    raw = process.stdout.readline()
    if not raw:
        return_code = process.poll()
        raise RunnerError(f"worker closed protocol stdout with status {return_code}")
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"worker emitted invalid protocol JSON: {raw[:400]!r}") from exc
    if not isinstance(record, dict):
        raise RunnerError("worker protocol record must be a JSON object")
    return raw, record


def _validate_drive(reply: dict[str, Any], seq: int, frame_time_s: float, ready: dict[str, Any]) -> tuple[float, float, float]:
    if (reply.get("protocol") != PROTOCOL_NAME or reply.get("type") != "drive"
            or reply.get("seq") != seq
            or not math.isclose(float(reply.get("time_s", math.nan)), frame_time_s, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(reply.get("render_time_s", math.nan)), frame_time_s,
                                rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(reply.get("pattern_time_s", math.nan)), frame_time_s,
                                rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(reply.get("neural_window_end_s", math.nan)),
                                frame_time_s + FRAME_DT_S, rel_tol=0.0, abs_tol=1e-12)):
        raise RunnerError(f"worker frame/window clock mismatch at seq={seq}")
    if reply.get("effective_backends", ready.get("effective_backends")) != ready.get("effective_backends"):
        raise RunnerError(f"worker backend changed at seq={seq}")
    retinal_hashes = reply.get("eye_column_hashes")
    if (not isinstance(retinal_hashes, dict)
            or any(not isinstance(retinal_hashes.get(eye), str)
                   or len(retinal_hashes[eye]) != 64 for eye in ("left", "right"))):
        raise RunnerError(f"worker omitted live retinal column hashes at seq={seq}")
    rates = reply.get("population_rates_hz")
    if not isinstance(rates, dict) or not {"DNp15/L", "DNp15/R"}.issubset(rates):
        raise RunnerError(f"worker omitted bilateral DNp15 rate at seq={seq}")
    values = (float(reply.get("leg_power", math.nan)), float(rates["DNp15/L"]), float(rates["DNp15/R"]))
    if (not all(math.isfinite(value) for value in values)
            or not 0.0 <= values[0] <= 1.0 or values[1] < 0.0 or values[2] < 0.0):
        raise RunnerError(f"invalid worker P/DNp15 values at seq={seq}: {values}")
    return values


def _partial_body_trace(session: Any, run_dir: Path) -> Path | None:
    count = int(session._native_samples)
    if count <= 0:
        return None
    arrays = session.all_arrays
    names = (
        "time_s", "qpos", "qvel", "body_xpos_mm", "body_xquat_wxyz",
        "body_xvel_mm_s", "body_xangvel_rad_s", "thorax_heading_yaw_rad",
        "thorax_heading_xy", "tarsus5_xpos_mm", "foot_contact_found",
        "foot_contact_force_local_native", "foot_contact_torque_local_native",
        "foot_contact_pos_mm", "foot_contact_normal_world", "foot_contact_tangent_world",
        "foot_contact_fz_up_native", "root_external_wrench_native", "qfrc_applied_native",
    )
    path = run_dir / "partial-body-trace.npz"
    np.savez_compressed(
        path,
        **{name: arrays[name][:count] for name in names if name in arrays},
        cpg_phase_rad=session.cpg_phase_by_sample[:count],
        cpg_magnitude=session.cpg_magnitude_by_sample[:count],
        descending_drive_left_right=session.descending_drive_by_sample[:count],
        root_qpos_write_count=session.root_qpos_write_count_by_sample[:count],
        root_qvel_write_count=session.root_qvel_write_count_by_sample[:count],
    )
    return path


def _body_window_summary(session: Any) -> dict[str, Any]:
    count = int(session._native_samples)
    trace = session.all_arrays
    times = np.asarray(trace["time_s"][:count], dtype=np.float64)
    raw_yaw = np.asarray(trace["thorax_heading_yaw_rad"][:count], dtype=np.float64)
    if count < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(raw_yaw)):
        raise RunnerError("body trace lacks finite time/yaw samples")
    yaw = np.unwrap(raw_yaw)
    result: dict[str, Any] = {}
    for label, (start_s, end_s) in WINDOWS.items():
        if not (times[0] <= start_s < end_s <= times[-1] + 1e-9):
            raise RunnerError(f"body trace does not cover window {label}")
        yaw_start = float(np.interp(start_s, times, yaw))
        yaw_end = float(np.interp(end_s, times, yaw))
        result[label] = {
            "absolute_window_s": [start_s, end_s],
            "thorax_heading_yaw_start_rad": yaw_start,
            "thorax_heading_yaw_end_rad": yaw_end,
            "thorax_heading_yaw_change_deg": math.degrees(yaw_end - yaw_start),
            "mean_thorax_heading_yaw_rate_rad_s": (yaw_end - yaw_start) / (end_s - start_s),
        }
    result["raw_trace"] = {
        "thorax_heading_yaw_rad_sha256": array_sha256(raw_yaw),
        "thorax_heading_yaw_rad_unwrapped_sha256": array_sha256(yaw),
        "thorax_heading_xy_sha256": array_sha256(trace["thorax_heading_xy"][:count]),
        "samples": count,
        "yaw_coordinate": "thorax local +X projected into native FlyGym world XY; unwrapped before differences",
        "ball_omega_z_used": False,
    }
    return result


def _controller_metadata(controller: Any, case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    metadata = controller.controller_metadata()
    metadata.update({
        "input_source": "fresh current-pose angular retina and current worker P/DNp15 rates through fixed calibrated_static decoder",
        "closed_loop_visual_pose": True,
        "stimulus_label_branching": False,
        "case_label_selects_pixels_only": True,
        "worker_native_bilateral_drive_applied": False,
        "decoder_baseline_mode": "calibrated_static",
        "decoder_baseline_rates_hz": BASELINE_RATES_HZ,
        "decoder_scope_status": protocol["evidence_boundary"]["scope_checker"]["status"],
        "scope_classification": "candidate outside fixed-head baseline scope; not a calibration pass",
        "case_id_for_provenance_only": case["case_id"],
        "expected_or_biological_turn_sign_assertion": False,
        "root_state_write_after_initialization": False,
        "root_external_wrench": False,
        "brain_or_body_sensory_feedback_injected": False,
    })
    return metadata


def _run_case(case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    # Preserve the actual CUDA environment for the independent neural worker
    # before importing the free-body replay, which intentionally hides CUDA.
    worker_env = os.environ.copy()
    worker_env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "FLYVIS_ROOT_DIR": "/data/models/flyvis",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(SCRIPTS),
    })
    verify_frozen_inputs(protocol)
    if case["case_id"] not in CASE_IDS:
        raise RunnerError(f"unknown case id {case['case_id']}")
    run_dir = RUNS / case["case_id"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite case data: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    worker_dir = run_dir / "worker"
    worker_dir.mkdir()
    run_record: dict[str, Any] = {
        "schema_version": "flygym-free-body-online-stripe-run-v1",
        "status": "running",
        "case_id": case["case_id"],
        "condition_source": "prepared angular stimulus specification only",
        "action_source": "fresh online worker population outputs through fixed calibrated_static decoder",
        "offline_Pd_schedule_read": False,
        "stimulus_label_passed_to_decoder": False,
        "scope_status": protocol["evidence_boundary"]["scope_checker"]["status"],
        "evidence_classification": "candidate full-pose extrapolation; not validated stripe turning",
        "body_seed": BODY_SEED,
        "neural_seed": NEURAL_SEED,
        "started_utc_epoch_s": time.time(),
        "worker_command": _worker_command(case, worker_dir, protocol),
    }
    run_record_path = run_dir / "run-record.json"
    write_json(run_record_path, run_record)

    worker: subprocess.Popen[str] | None = None
    session: Any | None = None
    body_raw_saved = False
    completed_frames = 0
    completed_body_ticks = 0
    previous_command = {
        "P_control": 0.0,
        "signed_d": 0.0,
        "source_neural_seq": None,
        "source_neural_window_end_s": None,
        "held_soma_side": "left",
    }
    try:
        run_record["systemd_case_scope"] = _systemd_case_scope_record()
        write_json(run_record_path, run_record)
        sys.path.insert(0, str(SCRIPTS))
        # This project-owned replay provides the exact current free-ground
        # body/controller config. Its import masks CUDA only in this body proc;
        # worker_env was copied above and explicitly restored for the child.
        import replay_flygym_free_body_stripes_v1 as free_body_replay
        mechanics = free_body_replay.m
        from run_flygym_reference_walk import FlyGymReferenceSession, WalkConfig
        from flygym_phase_speed_gait_repaired_v1 import (
            PhaseSpeedGaitController, controller_config_for_differential_ap,
        )
        from flygym_ap_endpoint_gesture_prototype import APEndpointGestureConfig
        from dnp15_opponent_steering_decoder_repaired_v1 import (
            DecoderConfig, Dnp15OpponentSteeringDecoder, StaticRateBaseline,
        )
        import run_flygym_online_optomotor_eye_parity_v4_1_attempt5 as pose_bridge

        verify_frozen_inputs(protocol)
        walk_values = dict(mechanics.read_json(Path(protocol["prepared_inputs"]["walk_effective_config"]))["walk_config"])
        walk_values.update(duration_s=DURATION_S, seed=BODY_SEED)
        for key in ("spawn_position_mm", "spawn_quaternion_wxyz", "position_actuator_forcerange_native", "camera_resolution_hw"):
            if key in walk_values:
                walk_values[key] = tuple(walk_values[key])
        walk_config = WalkConfig(**walk_values)
        controller_base = APEndpointGestureConfig(**mechanics.read_json(
            Path(protocol["prepared_inputs"]["controller_config"])
        )["config"])
        if float(controller_base.active_threshold) != P_DEADBAND:
            raise RunnerError("body AP controller active threshold differs from decoder deadband")
        baseline = StaticRateBaseline.from_payload(mechanics.read_json(
            Path(protocol["prepared_inputs"]["baseline_payload"])
        ))
        if baseline.left_hz != 0.0 or baseline.right_hz != 142.5:
            raise RunnerError("loaded calibrated_static baseline rates changed")
        decoder_config = DecoderConfig(
            baseline_mode="calibrated_static",
            normalization_hz=NORMALIZATION_HZ,
            p_deadband=P_DEADBAND,
        )
        decoder = Dnp15OpponentSteeringDecoder(baseline, decoder_config)
        run_record["decoder_provenance"] = decoder.provenance()
        run_record["body_config"] = asdict(walk_config)
        run_record["controller_base_config"] = asdict(controller_base)
        run_record["controller_config_scope"] = "AP command changes per queued neural frame; V4.1 gait config remains fixed"
        write_json(run_record_path, run_record)

        session = FlyGymReferenceSession(
            duration_s=DURATION_S, seed=BODY_SEED, render_video=False,
            config=walk_config, run_id=f"online-free-body-{case['case_id']}-seed{BODY_SEED}",
        )
        if not math.isclose(session.time_s, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise RunnerError("body session did not start at time zero")
        initial_qpos = np.asarray(session.data.qpos, dtype=np.float64).copy()
        initial_qvel = np.asarray(session.data.qvel, dtype=np.float64).copy()
        run_record["initial_state"] = {
            "time_s": session.time_s,
            "qpos_sha256": array_sha256(initial_qpos),
            "qvel_sha256": array_sha256(initial_qvel),
            "initial_command": dict(previous_command),
        }
        controller = PhaseSpeedGaitController(
            session, controller_config_for_differential_ap(0.0, base_config=controller_base)
        )
        session.controller = controller

        vision_path, sensors = pose_bridge._load_vision_camera_config(session)
        body_ids = pose_bridge._body_ids_by_name(session)
        initial_camera = pose_bridge._camera_geometry(session, sensors, body_ids)
        initial_eye_midpoint = initial_camera["midpoint_mm"].copy()
        eye_separation_m = float(initial_camera["baseline_m"])
        if not math.isclose(eye_separation_m, 0.00076, rel_tol=0.0, abs_tol=2e-6):
            raise RunnerError(f"official eye-camera baseline changed: {eye_separation_m}")
        rigging_path = Path(vision_path).with_name("rigging.yaml")
        if not rigging_path.is_file():
            raise FileNotFoundError(f"FlyGym rigging file missing: {rigging_path}")
        run_record["camera_geometry"] = {
            "vision_config_path": str(vision_path),
            "vision_config_sha256": sha256(Path(vision_path)),
            "rigging_config_path": str(rigging_path),
            "rigging_config_sha256": sha256(rigging_path),
            "official_eye_camera_midpoint_mm": initial_eye_midpoint.tolist(),
            "eye_separation_m": eye_separation_m,
            "virtual_scene_origin_xy_m": [0.0, 0.0],
            "head_pose_source": "thorax pose; c_head fused to c_thorax in the pinned body model",
        }
        run_record["body_session_started"] = True
        write_json(run_record_path, run_record)

        protocol_path = run_dir / "worker-protocol-stdout.jsonl"
        requests_path = run_dir / "worker-requests.jsonl"
        frames_path = run_dir / "online-frames.jsonl"
        ticks_path = run_dir / "control-ticks.jsonl"
        stderr_path = run_dir / "worker-stderr.log"
        command = _worker_command(case, worker_dir, protocol)
        # This explicit child environment repairs the CPU body process's
        # CUDA_VISIBLE_DEVICES masking; no fallback backend is permitted.
        worker_env["CUDA_VISIBLE_DEVICES"] = "0"
        worker = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=worker_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_path.open("x", encoding="utf-8", buffering=1),
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if worker.stdin is None or worker.stdout is None:
            raise RunnerError("worker did not expose the JSONL protocol pipes")
        run_record["worker_pid"] = worker.pid
        run_record["worker_started"] = True
        write_json(run_record_path, run_record)

        with protocol_path.open("x", encoding="utf-8", buffering=1) as protocol_log, \
                requests_path.open("x", encoding="utf-8", buffering=1) as request_log, \
                frames_path.open("x", encoding="utf-8", buffering=1) as frame_log, \
                ticks_path.open("x", encoding="utf-8", buffering=1) as tick_log:
            raw_ready, ready = _protocol_read(worker, 180.0)
            protocol_log.write(raw_ready)
            protocol_log.flush()
            _validate_worker_ready(ready, case, protocol)
            run_record["worker_ready_pass"] = True
            run_record["worker_ready"] = ready
            run_record["worker_effective_backends"] = ready["effective_backends"]
            run_record["worker_source_assets"] = ready.get("source_assets")
            write_json(run_record_path, run_record)

            for seq in range(FRAME_COUNT):
                frame_time_s = seq * FRAME_DT_S
                if not math.isclose(session.time_s, frame_time_s, rel_tol=0.0, abs_tol=1e-9):
                    raise RunnerError(f"body clock mismatch before pose request seq={seq}")
                pose_state = pose_bridge._build_virtual_pose(
                    session, sensors, body_ids, (0.0, 0.0), initial_eye_midpoint,
                )
                request = {
                    "protocol": PROTOCOL_NAME,
                    "type": "frame",
                    "seq": seq,
                    "time_s": frame_time_s,
                    "dt_s": FRAME_DT_S,
                    "pose": pose_state["pose"],
                }
                request_raw = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                request_log.write(request_raw)
                request_log.flush()
                worker.stdin.write(request_raw)
                worker.stdin.flush()
                raw_reply, reply = _protocol_read(worker, 90.0)
                protocol_log.write(raw_reply)
                protocol_log.flush()
                raw_power, left_rate, right_rate = _validate_drive(reply, seq, frame_time_s, ready)
                decoded = decoder.step(raw_power, left_rate, right_rate)
                current_decoded_command = {
                    "P_control": float(decoded.control_power),
                    "signed_d": float(decoded.clipped_opponent_signal),
                    "held_soma_side": str(decoded.soma_side or previous_command["held_soma_side"]),
                    "source_neural_seq": seq,
                    "source_neural_window_start_s": frame_time_s,
                    "source_neural_window_end_s": frame_time_s + FRAME_DT_S,
                    "available_at_body_time_s": frame_time_s + FRAME_DT_S,
                }
                applied_command = dict(previous_command)
                applied_command.update({
                    "body_frame_seq": seq,
                    "body_action_start_s": frame_time_s,
                    "body_action_end_s": frame_time_s + FRAME_DT_S,
                    "command_delay_frames": 1,
                    "initial_neutral_command": seq == 0,
                })
                frame_row = {
                    "seq": seq,
                    "body_pose_time_s": frame_time_s,
                    "worker_neural_window_start_s": frame_time_s,
                    "worker_neural_window_end_s": frame_time_s + FRAME_DT_S,
                    "decoder_output_time_s": frame_time_s + FRAME_DT_S,
                    "body_action_start_s": frame_time_s,
                    "body_action_end_s": frame_time_s + FRAME_DT_S,
                    "input_pose": pose_state["pose"],
                    "source_body_pose": pose_state["source_pose"],
                    "virtual_eye_origins_m": pose_state["virtual_eye_origins_m"],
                    "worker_request_sha256": hashlib.sha256(request_raw.encode("utf-8")).hexdigest(),
                    "worker_reply_sha256": hashlib.sha256(raw_reply.encode("utf-8")).hexdigest(),
                    "retinal_column_hashes": reply.get("eye_column_hashes"),
                    "retinal_columns_saved_by_worker": True,
                    "raw_worker_leg_power_P": raw_power,
                    "raw_worker_DNp15_L_rate_hz": left_rate,
                    "raw_worker_DNp15_R_rate_hz": right_rate,
                    "worker_native_bilateral_drive_not_applied": [
                        reply.get("left_descending_drive"), reply.get("right_descending_drive")
                    ],
                    "decoder_inputs_only": [raw_power, left_rate, right_rate],
                    "decoder_output": decoded.to_dict(),
                    "body_command_applied_this_frame": applied_command,
                    "decoder_output_queued_for_next_frame": current_decoded_command,
                    "stimulus_label_passed_to_decoder": False,
                }
                _write_jsonl(frame_log, frame_row)

                controller.gesture_config = controller_config_for_differential_ap(
                    float(applied_command["signed_d"]), base_config=controller_base,
                )
                for tick_in_frame in range(CONTROL_TICKS_PER_FRAME):
                    tick = seq * CONTROL_TICKS_PER_FRAME + tick_in_frame
                    tick_start_s = frame_time_s + tick_in_frame * CONTROL_DT_S
                    if not math.isclose(session.time_s, tick_start_s, rel_tol=0.0, abs_tol=1e-9):
                        raise RunnerError(f"body control clock mismatch seq/tick={seq}/{tick_in_frame}")
                    native_first = int(session._native_samples)
                    step = session.step_control_tick((float(applied_command["P_control"]),) * 2)
                    if (int(step["native_samples_added"]) != NATIVE_STEPS_PER_TICK
                            or not math.isclose(session.time_s, tick_start_s + CONTROL_DT_S,
                                                rel_tol=0.0, abs_tol=1e-9)):
                        raise RunnerError(f"body tick duration mismatch at {seq}/{tick_in_frame}")
                    native_last = int(session._native_samples)
                    telemetry = controller.telemetry_snapshot()
                    tick_row = {
                        "tick": tick,
                        "source_visual_neural_seq_current": seq,
                        "applied_command_source_neural_seq": applied_command["source_neural_seq"],
                        "tick_in_frame": tick_in_frame,
                        "control_time_s": tick_start_s,
                        "control_end_time_s": float(session.time_s),
                        "body_pose_frame_time_s": frame_time_s,
                        "P_control": float(applied_command["P_control"]),
                        "signed_d": float(applied_command["signed_d"]),
                        "held_soma_side": applied_command["held_soma_side"],
                        "actual_descending_signal_lr": [float(applied_command["P_control"])] * 2,
                        "actual_AP_amplitude": abs(float(applied_command["signed_d"])),
                        "ap_soma_side_from_signed_d": "left" if float(applied_command["signed_d"]) >= 0.0 else "right",
                        "native_sample_indices_halfopen": [native_first, native_last],
                        "native_samples_added": int(step["native_samples_added"]),
                        "thorax_heading_yaw_rad_after_tick": float(step["thorax_heading_yaw_rad"]),
                        "cpg_phase_rad_after_tick": np.asarray(step["cpg_phase_rad"], dtype=float).tolist(),
                        "controller_telemetry": telemetry,
                        "stimulus_label_passed_to_controller": False,
                    }
                    _write_jsonl(tick_log, tick_row)
                    completed_body_ticks += 1
                completed_frames += 1
                previous_command = current_decoded_command
                if (seq + 1) % 100 == 0:
                    print(json.dumps({"case": case["case_id"], "frames": seq + 1,
                                      "sim_time_s": session.time_s}, sort_keys=True), flush=True)

            if (session._control_samples != FRAME_COUNT * CONTROL_TICKS_PER_FRAME
                    or session._native_samples != FRAME_COUNT * CONTROL_TICKS_PER_FRAME * NATIVE_STEPS_PER_TICK + 1
                    or completed_body_ticks != FRAME_COUNT * CONTROL_TICKS_PER_FRAME):
                raise RunnerError("final body control/native sample count mismatch")

            # Persist the complete physical trace before stop-ack or analysis.
            controller_metadata = _controller_metadata(controller, case, protocol)
            session.save(
                run_dir,
                constant_drive_values_left_right=None,
                controller_metadata=controller_metadata,
            )
            body_raw_saved = True
            run_record.update({
                "body_raw_capture_saved_before_summary": True,
                "trajectory_path": str(run_dir / "trajectory.npz"),
                "contacts_path": str(run_dir / "contacts.npz"),
                "body_manifest_path": str(run_dir / "manifest.json"),
                "control_tick_count": int(session._control_samples),
                "native_sample_count": int(session._native_samples),
                "physics_step_count": FRAME_COUNT * CONTROL_TICKS_PER_FRAME * NATIVE_STEPS_PER_TICK,
                "last_decoded_command_pending": dict(previous_command),
                "last_decoded_command_was_applied": False,
            })
            write_json(run_record_path, run_record)

            worker.stdin.write(json.dumps({"protocol": PROTOCOL_NAME, "type": "stop"}) + "\n")
            worker.stdin.flush()
            raw_stop, stop_ack = _protocol_read(worker, 120.0)
            protocol_log.write(raw_stop)
            protocol_log.flush()
            if (stop_ack.get("type") != "stopped" or stop_ack.get("frame_count") != FRAME_COUNT
                    or stop_ack.get("last_seq") != FRAME_COUNT - 1
                    or not math.isclose(float(stop_ack.get("last_time_s", math.nan)),
                                        DURATION_S - FRAME_DT_S, rel_tol=0.0, abs_tol=1e-12)):
                raise RunnerError(f"worker stop acknowledgement is incomplete: {stop_ack}")
            worker.stdin.close()
            exit_code = worker.wait(timeout=60.0)
            if exit_code != 0:
                raise RunnerError(f"worker exited with status {exit_code}")
            run_record["worker_stop_ack"] = stop_ack
            run_record["worker_exit_code"] = int(exit_code)
            write_json(run_record_path, run_record)
            run_record["worker_artifact_validation"] = _validate_worker_provenance(
                worker_dir, ready, stop_ack, protocol,
            )
            write_json(run_record_path, run_record)

        summary = _body_window_summary(session)
        summary.update({
            "status": "complete",
            "classification": "candidate engineering measurements; not a biological stripe-follow result",
            "case_id": case["case_id"],
            "stimulus_metadata": {key: case[key] for key in (
                "head_direction_sign", "head_angular_speed_deg_s", "arena_angular_speed_deg_s",
                "temporal_frequency_hz", "interpretation",
            )},
            "decoder_baseline_scope_status": protocol["evidence_boundary"]["scope_checker"]["status"],
            "body_yaw_not_ball_omega_z": True,
            "source_neural_frames": FRAME_COUNT,
            "applied_body_control_frames": FRAME_COUNT,
            "startup_neutral_body_frames": 1,
            "terminal_decoded_frame_unapplied": True,
        })
        write_json(run_dir / "window-summary.json", summary)
        run_record.update({
            "status": "complete",
            "finished_utc_epoch_s": time.time(),
            "completed_neural_frames": completed_frames,
            "completed_body_ticks": completed_body_ticks,
            "worker_shutdown": True,
            "body_raw_capture_saved_before_summary": True,
            "window_summary_path": str(run_dir / "window-summary.json"),
            "scope_interpretation": "candidate; fixed-head calibrated_static baseline was explicitly rejected for full-pose walking scope by the checker",
        })
        write_json(run_record_path, run_record)
        output_records = {
            str(path.resolve()): _sha_record(path)
            for path in sorted(run_dir.rglob("*")) if path.is_file()
        }
        write_json(run_dir / "online-integrity.json", {
            "schema_version": "flygym-free-body-online-stripes-integrity-v1",
            "status": "complete",
            "files": output_records,
            "record_created_after_all_capture_and_summary_files": True,
        })
        return run_record
    except BaseException:
        run_record.update({
            "status": "failed_stopped",
            "failed_utc_epoch_s": time.time(),
            "completed_neural_frames": completed_frames,
            "completed_body_ticks": completed_body_ticks,
            "error": traceback.format_exc(),
        })
        if session is not None and not body_raw_saved:
            try:
                partial = _partial_body_trace(session, run_dir)
                run_record["partial_body_trace_path"] = str(partial) if partial else None
                run_record["partial_body_trace_saved_before_error_record"] = partial is not None
            except BaseException as save_error:
                run_record["partial_body_trace_error"] = repr(save_error)
        try:
            write_json(run_record_path, run_record)
        except BaseException:
            pass
        raise
    finally:
        if worker is not None and worker.poll() is None:
            try:
                if worker.stdin is not None:
                    worker.stdin.write(json.dumps({"protocol": PROTOCOL_NAME, "type": "stop"}) + "\n")
                    worker.stdin.flush()
                    _protocol_read(worker, 15.0)
            except BaseException:
                pass
            try:
                if worker.stdin is not None:
                    worker.stdin.close()
            except OSError:
                pass
            try:
                worker.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=3.0)
        if session is not None:
            session.close()


def _check_root_review(protocol: dict[str, Any]) -> dict[str, Any]:
    path = BASE / "root-execution-review.json"
    review = read_json(path)
    if (review.get("status") != "approved"
            or review.get("protocol_sha256") != sha256(BASE / "protocol.json")
            or review.get("runner_sha256") != sha256(Path(__file__))
            or review.get("source_freeze_sha256") != protocol["source_freeze"]["sha256"]):
        raise RunnerError("root execution review does not bind this runner, protocol and source freeze")
    return review


def _case_child(case_id: str) -> None:
    if os.environ.get("ONLINE_FREEBODY_CASE_SCOPE") != "1":
        raise RunnerError("case launcher mode marker is missing")
    protocol = read_json(BASE / "protocol.json")
    verify_frozen_inputs(protocol)
    review = _check_root_review(protocol)
    if review.get("execution_authority") != "authorized_after_review":
        raise RunnerError("root review did not explicitly authorize the reviewed CPU/GPU case run")
    if case_id not in CASE_IDS:
        raise RunnerError(f"unsupported case: {case_id}")
    GLOBAL_NN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_NN_LOCK.open("a+") as neural_lock:
        fcntl.flock(neural_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        case = next(row for row in protocol["stimuli"]["conditions"] if row["case_id"] == case_id)
        _run_case(case, protocol)


def _launch_case(case_id: str) -> None:
    unit_name = f"online-freebody-{case_id}-{os.getpid()}-{time.monotonic_ns()}"
    command = [
        "systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet",
        f"--unit={unit_name}",
        f"--property=MemoryMax={RESOURCE_LIMITS['host_memory_max']}",
        f"--property=CPUQuota={RESOURCE_LIMITS['cpu_quota']}",
        f"--property=RuntimeMaxSec={CASE_RUNTIME_S}",
        "--setenv=ONLINE_FREEBODY_CASE_SCOPE=1",
        f"--setenv=ONLINE_FREEBODY_CASE_UNIT={unit_name}",
        str(BODY_PYTHON), str(Path(__file__).resolve()),
        "--case-child", "--case", case_id,
    ]
    subprocess.run(command, cwd=PROJECT, check=True, timeout=CASE_RUNTIME_S + 60)


def _execute(case_id: str | None) -> None:
    protocol = read_json(BASE / "protocol.json")
    verify_frozen_inputs(protocol)
    _check_root_review(protocol)
    if case_id is not None:
        if case_id not in CASE_IDS:
            raise RunnerError(f"unsupported case: {case_id}")
        _launch_case(case_id)
        return

    queue_lock_path = BASE / "queue.lock"
    with queue_lock_path.open("a+") as queue_lock:
        fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = BASE / "queue-status.json"
        status: dict[str, Any] = {
            "schema_version": "flygym-free-body-online-stripes-queue-status-v1",
            "status": "running",
            "completed": [],
            "current": None,
            "started_utc_epoch_s": time.time(),
            "max_runtime_seconds": QUEUE_RUNTIME_S,
        }
        write_json(status_path, status)
        started = time.monotonic()
        try:
            for case_id in CASE_IDS:
                if time.monotonic() - started > QUEUE_RUNTIME_S:
                    raise TimeoutError("online stripe queue exceeded the 4 hour outer limit")
                status["current"] = case_id
                write_json(status_path, status)
                _launch_case(case_id)
                status["completed"].append(case_id)
                write_json(status_path, status)
            records = [read_json(RUNS / case_id / "run-record.json") for case_id in CASE_IDS]
            if any(row.get("status") != "complete" for row in records):
                raise RunnerError("case run record is incomplete after the three-case queue")
            qpos_hashes = {row["initial_state"]["qpos_sha256"] for row in records}
            qvel_hashes = {row["initial_state"]["qvel_sha256"] for row in records}
            if len(qpos_hashes) != 1 or len(qvel_hashes) != 1:
                raise RunnerError("paired cases did not share the same initial body state")
            write_json(BASE / "cohort-summary.json", {
                "schema_version": "flygym-free-body-online-stripes-cohort-summary-v1",
                "status": "complete",
                "classification": "candidate online body/vision loop; no biological follow-success claim",
                "case_records": records,
                "initial_qpos_hash_shared": len(qpos_hashes) == 1,
                "initial_qvel_hash_shared": len(qvel_hashes) == 1,
                "body_yaw_coordinate": "thorax local +X projected into native FlyGym world XY",
                "ball_omega_z_used": False,
                "baseline_scope_checker": protocol["evidence_boundary"]["scope_checker"],
                "scope_extrapolation_candidate_only": True,
            })
            status.update(status="complete", current=None, finished_utc_epoch_s=time.time())
        except BaseException:
            status.update(status="failed_stopped", current=status.get("current"),
                          error=traceback.format_exc(), failed_utc_epoch_s=time.time())
            raise
        finally:
            write_json(status_path, status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--case-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASE_IDS)
    args = parser.parse_args(argv)
    if args.prepare:
        if args.case:
            parser.error("--case is only valid with a root-authorized execution")
        result = prepare()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    if args.case_child:
        if not args.case:
            parser.error("--case-child requires --case")
        _case_child(args.case)
        return 0
    if not args.execute:
        parser.error("select --prepare or --execute")
    _execute(args.case)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_flygym_free_body_online_stripes_v1: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
