#!/usr/bin/env python3
"""Replay frozen 800-frame binocular columns through the frozen neural worker and ball plant.

CPU-only input validation is the default. Execution is one explicitly
authorized case at a time: recorded eye columns -> frozen FlyVis/MaleCNS worker
-> frozen DNp15 decoder -> recorded-command tethered-ball adapter. Stimulus
labels never enter the decoder or body controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import traceback
from typing import Any, TextIO

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path("/data/artifacts/by-project/malecns-gpu/engineering")
CORE_FREEZE = ARTIFACTS / "final-frozen-assays-20261007-v1/core-freeze.json"
CORE_FREEZE_SHA256 = "fdfb4f20ef145d07937c8fb01ffb1e4c1d610292f7d50001cbccd2766ce53b94"
WORKER = PROJECT / "scripts/flygym_online_neural_worker_eye_parity_state_v2.py"
WORKER_SHA256 = "543e82ed40c339d000cf38cb179f60e5818ef9de65747d36fb19fdb346adeb3d"
ADAPTER = PROJECT / "scripts/flyvis_runtime_adapter_eye_parity_v2.py"
ADAPTER_SHA256 = "993eb7e193c5d8a49f8e88412b6c5378bc9cd1ab026dc9896e2c5ea31bf320e5"
WORKER_CONFIG = ARTIFACTS / "flygym-fixed-head-contrast-ch-diagnostic-20261007-v2-r2/worker-config/flygym_online_neural_worker_contrast_ch.json"
WORKER_CONFIG_SHA256 = "4838331e3a0bdf142cfd98a2db3657d0eee621ab35f829e3be828300ce08fcf6"
DECODER = PROJECT / "scripts/dnp15_opponent_steering_decoder_repaired_v1.py"
DECODER_SHA256 = "9913dd39b1a733d044dc6baf5fbfa9984a10fc327b6a735e0d309f09469caec2"
BASELINE = ARTIFACTS / "flygym-dnp15-template-steering-decoder-20261007-v1/static-baseline-payload.json"
BASELINE_SHA256 = "25543795ad983ba78ee244d9e551ff4a460024a2cf65ebfb004efd63f69f4a39"
BALL_RUNNER = PROJECT / "scripts/run_flygym_tethered_ball_recorded_input_replay.py"
BALL_RUNNER_SHA256 = "e5680c0c73bb35f9747bac3b9370142c79396abb8d7c425bb6c7736b5de98008"
CANONICALIZER = PROJECT / "scripts/run_flygym_tethered_treadmill_pilot.py"
CANONICALIZER_SHA256 = "00f8be3eb69a6dc215cdeb9fed0e4130b91204cdc5f45a2e9c08a250863b3b80"
BALL_SESSION = PROJECT / "scripts/flygym_tethered_treadmill_session_v1.py"
BALL_SESSION_SHA256 = "26b164e4f246f6a0371bd7ea27e19f6108c8400a3ac2186511c47ed3b1741c41"
FIXED_POSE_PROTOCOL = ARTIFACTS / "flygym-fixed-pose-angular-calibration-20261006-v1/protocol-attempt-4.json"
FIXED_POSE_PROTOCOL_SHA256 = "5f4b4f971dae60d7738d7af5bb79fbe63fbcced6dbb3ae223eb83539db0e255f"
LEGACY_BALL_PROTOCOL = ARTIFACTS / "flygym-tethered-ball-recorded-input-replay-20261007-v1/protocol.json"
LEGACY_BALL_PROTOCOL_SHA256 = "21e697a263dfdedbc8413d86886cdf6784480d7ce65e3648960b0210d9e9b83e"
BODY_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
WORKER_PYTHON = Path("/usr/bin/python3.12")
OUTPUT_ROOT = ARTIFACTS / "standardized-repair-20261007-v1"
LOCK_PATH = ARTIFACTS / "flygym-recorded-retina-neural-ball-harness-20261007-v2/locks/neural-worker-and-ball-case.lock"
UV_PATH = ARTIFACTS / "visual-motor-feasibility-20261006-v1/standard-stimulus-preparation/angular-retinal-samples.npz"
UV_SHA256 = "e244a411df2f8fa5bcbc8e1fb14fe68b8d1836116cd35ac98082d294f4a1e5ef"
LEFT_RAYS_SHA256 = "2f847637ae64bea5a5454aa90e86e440a23d76d5c3b85a866936e502691654f1"
RIGHT_RAYS_SHA256 = "5037edbec7e4709ddc5f31643de643cdadef4492145756e72634d52727b7b4f6"
FRAME_COUNT = 800
FRAME_DT_S = 0.005
ONSET_S = 1.0
STATIC_WARMUP_S = 1.0
CONTROL_DT_S = 0.001
CONTROL_TICKS_PER_FRAME = 5
NATIVE_DT_S = 0.0001
NATIVE_STEPS_PER_TICK = 10
SEED = 1729
BASELINE_RATES_HZ = {"DNp15/L": 0.0, "DNp15/R": 142.5}
RESOURCE_LIMITS = {
    "memory_max": "5G", "cpu_quota": "200%", "runtime_max_seconds": 900,
    "worker_max_vram_gib": 8.0,
}
PROTOCOL_NAME = "flygym-online-v1"


class AssayError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes(order="C")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssayError(f"expected JSON object: {path}")
    return value


def _frozen_sources() -> dict[str, dict[str, Any]]:
    extension_freeze = load_json(OUTPUT_ROOT / "repair-source-freeze.json")
    for path, record in extension_freeze["files"].items():
        if sha256(Path(path)) != record["sha256"]:
            raise AssayError(f"repair source changed: {path}")
    return {
        "core_freeze": {"path": str(CORE_FREEZE), "sha256": CORE_FREEZE_SHA256},
        "worker": {"path": str(WORKER), "sha256": WORKER_SHA256},
        "adapter": {"path": str(ADAPTER), "sha256": ADAPTER_SHA256},
        "worker_config": {"path": str(WORKER_CONFIG), "sha256": WORKER_CONFIG_SHA256},
        "decoder": extension_freeze["files"][str(DECODER)],
        "decoder_baseline": {"path": str(BASELINE), "sha256": BASELINE_SHA256},
        "repair_freeze": {"path": str(OUTPUT_ROOT / "repair-source-freeze.json"), "sha256": sha256(OUTPUT_ROOT / "repair-source-freeze.json")},
        "ball_runner": {"path": str(BALL_RUNNER), "sha256": BALL_RUNNER_SHA256},
        "canonicalizer": {"path": str(CANONICALIZER), "sha256": CANONICALIZER_SHA256},
        "ball_session": {"path": str(BALL_SESSION), "sha256": BALL_SESSION_SHA256},
        "fixed_pose_protocol": {"path": str(FIXED_POSE_PROTOCOL), "sha256": FIXED_POSE_PROTOCOL_SHA256},
        "legacy_ball_protocol": {"path": str(LEGACY_BALL_PROTOCOL), "sha256": LEGACY_BALL_PROTOCOL_SHA256},
    }


def _validate_zero_gray_reference() -> None:
    references = ARTIFACTS / "flygym-recorded-retina-neural-ball-harness-20261007-v2/runs"
    summaries = sorted(references.glob("*/*/case_summary.json"))
    if len(summaries) != 17:
        raise AssayError("zero-gray evidence cohort must contain the frozen 17 cases")
    for path in summaries:
        summary = load_json(path)
        record = load_json(path.parent / "run-record.json")
        gray = summary["neural_windows"]["gray_0_1"]
        if (summary["status"] != "complete" or gray["frames"] != 200
                or gray["mean_DNp15_L_Hz"] != 0.0 or gray["mean_DNp15_R_Hz"] != 0.0
                or record["source_hashes_before"]["worker"]["sha256"] != WORKER_SHA256
                or record["source_hashes_before"]["worker_config"]["sha256"] != WORKER_CONFIG_SHA256):
            raise AssayError(f"zero-gray evidence/setup mismatch: {path}")


def verify_sources() -> dict[str, dict[str, Any]]:
    _validate_zero_gray_reference()
    result = {}
    for label, expected in _frozen_sources().items():
        path = Path(expected["path"])
        if not path.is_file():
            raise FileNotFoundError(f"frozen {label} source is missing: {path}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise AssayError(f"frozen {label} source hash changed: {actual} != {expected['sha256']}")
        result[label] = {**expected, "size_bytes": path.stat().st_size}
    return result


def _validate_pose(pose: Any) -> dict[str, Any]:
    if not isinstance(pose, dict):
        raise AssayError("case fixed_pose must be a JSON object")
    required = {"position_m", "quaternion_wxyz", "yaw_rad", "pitch_rad", "roll_rad",
                "velocity_mps", "angular_velocity_radps"}
    if set(pose) != required:
        raise AssayError(f"fixed_pose fields mismatch: missing={sorted(required-set(pose))}, extra={sorted(set(pose)-required)}")
    expected = [3, 4, 1, 1, 1, 3, 3]
    names = list(pose)
    # Keep the schema validation explicit while preserving the original JSON values.
    for name, size in zip(("position_m", "quaternion_wxyz", "yaw_rad", "pitch_rad", "roll_rad",
                           "velocity_mps", "angular_velocity_radps"), expected):
        value = pose[name]
        arr = np.asarray([value] if size == 1 else value, dtype=np.float64)
        if arr.shape != (size,) or not np.all(np.isfinite(arr)):
            raise AssayError(f"fixed_pose.{name} must contain {size} finite value(s)")
    if not math.isclose(float(np.linalg.norm(pose["quaternion_wxyz"])), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise AssayError("fixed_pose quaternion must be normalized")
    return pose


def _case_ray_hashes(metadata: dict[str, Any]) -> dict[str, str] | None:
    direct = metadata.get("direction_arrays")
    if isinstance(direct, dict):
        return {"left_directions": direct.get("left_directions"),
                "right_directions": direct.get("right_directions"),
                "column_uv": direct.get("column_uv")}
    if all(key in metadata for key in ("left_parity_ray_sha256", "right_parity_ray_sha256",
                                       "column_uv_sha256")):
        return {"left_directions": metadata["left_parity_ray_sha256"],
                "right_directions": metadata["right_parity_ray_sha256"],
                "column_uv": metadata["column_uv_sha256"]}
    stimulus = metadata.get("stimulus")
    if isinstance(stimulus, dict):
        eye = stimulus.get("eye_parity_and_axes", {})
        grid = eye.get("ray_grid_metadata", {}) if isinstance(eye, dict) else {}
        hashes = grid.get("direction_array_hashes", {}) if isinstance(grid, dict) else {}
        if isinstance(hashes, dict) and "left" in hashes and "right" in hashes:
            sampling = stimulus.get("retinal_sampling", {})
            uv_hash = sampling.get("column_uv_array_sha256") if isinstance(sampling, dict) else None
            if uv_hash is None:
                uv_hash = grid.get("column_uv_sha256")
            if uv_hash is None:
                uv_hash = grid.get("direction_array_hashes", {}).get("column_uv")
            return {"left_directions": hashes.get("left"), "right_directions": hashes.get("right"),
                    "column_uv": uv_hash}
    return None


def _normalize_manifest(raw: dict[str, Any], path: Path, canonical_pose: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    schema = raw.get("schema_version", raw.get("schema"))
    assay_id = raw.get("assay_id") or schema
    if not isinstance(assay_id, str) or not assay_id:
        raise AssayError("input manifest has no stable assay identifier")
    ray_hashes = {"left_directions": LEFT_RAYS_SHA256, "right_directions": RIGHT_RAYS_SHA256,
                  "column_uv": UV_SHA256}
    cases = raw.get("cases")
    normalized: list[dict[str, Any]] = []
    supported = {
        "flygym-recorded-retina-ball-assay-input-manifest-1",
        "flygym-recorded-retina-neural-ball-case-manifest-1",
        "flygym-object-fixation-input-manifest-1",
        "looming-input-preparation-manifest-v1",
    }
    if schema not in supported:
        raise AssayError(f"unsupported assay input manifest schema: {schema!r}")

    if schema == "looming-input-preparation-manifest-v1":
        if not isinstance(cases, dict) or not cases:
            raise AssayError("looming manifest cases must be a non-empty case->NPZ mapping")
        source_rays = raw.get("source_inputs", {}).get("active_worker_direction_arrays", {})
        if (source_rays.get("left_sha256") != LEFT_RAYS_SHA256
                or source_rays.get("right_sha256") != RIGHT_RAYS_SHA256
                or source_rays.get("column_uv_sha256") != UV_SHA256
                or source_rays.get("parity_by_registered_eye") != {"L": -1, "R": 1}):
            raise AssayError("looming source_inputs do not pin the active parity-v2 ray grid")
        summaries = raw.get("case_summaries", {})
        for case_id, record in cases.items():
            summary = summaries.get(case_id, {})
            normalized.append({"case_id": case_id, "input_npz": record, "fixed_pose": canonical_pose,
                               "worker_condition": "static", "stimulus_metadata": {},
                               "expected_left_array_sha256": summary.get("input_left_sha256"),
                               "expected_right_array_sha256": summary.get("input_right_sha256"),
                               "source_ray_hashes": ray_hashes})
    else:
        if not isinstance(cases, list) or not cases:
            raise AssayError("assay cases must be a non-empty list")
        fixed_pose = canonical_pose
        if schema == "flygym-object-fixation-input-manifest-1":
            geometry = raw.get("native_geometry", {})
            if (geometry.get("left_ray_sha256") != LEFT_RAYS_SHA256
                    or geometry.get("right_ray_sha256") != RIGHT_RAYS_SHA256
                    or geometry.get("column_uv_sha256") != UV_SHA256):
                raise AssayError("object-fixation manifest does not pin parity-v2 ray geometry")
            pose_record = raw.get("source_binding", {}).get("fixed_pose_source", {})
            pose_path = Path(pose_record.get("path", "")).expanduser().resolve(strict=True)
            if sha256(pose_path) != pose_record.get("sha256"):
                raise AssayError("object-fixation fixed-pose source hash differs from manifest")
            pose_source = load_json(pose_path)
            fixed_pose = _validate_pose(pose_source.get("source_metadata", {}).get("fixed_pose"))
            if fixed_pose != canonical_pose:
                raise AssayError("object-fixation pose source differs from shared frozen pose")
        for raw_case in cases:
            row = dict(raw_case)
            if schema == "flygym-object-fixation-input-manifest-1":
                row["fixed_pose"] = fixed_pose
                row["worker_condition"] = "static"
                row["stimulus_metadata"] = {key: value for key, value in raw_case.items()
                                             if key not in {"input_npz", "fixed_pose", "worker_condition"}}
            sidecar = path.parent / "inputs" / f"{row.get('case_id')}.manifest.json"
            if sidecar.is_file() and schema == "flygym-recorded-retina-ball-assay-input-manifest-1":
                side = load_json(sidecar)
                row["expected_left_array_sha256"] = side.get("left_columns_array_sha256")
                row["expected_right_array_sha256"] = side.get("right_columns_array_sha256")
            if schema == "flygym-object-fixation-input-manifest-1":
                row["expected_left_array_sha256"] = raw_case.get("left_columns_sha256")
                row["expected_right_array_sha256"] = raw_case.get("right_columns_sha256")
                row["source_ray_hashes"] = ray_hashes
            if schema == "flygym-recorded-retina-neural-ball-case-manifest-1":
                row["source_ray_hashes"] = ray_hashes
            normalized.append(row)

    ids = [row.get("case_id") for row in normalized]
    if len(set(ids)) != len(ids) or any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise AssayError("case_id values must be unique, non-empty strings")
    return assay_id, normalized, ray_hashes


def validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    manifest = load_json(path)
    if sha256(CORE_FREEZE) != CORE_FREEZE_SHA256:
        raise AssayError("global core-freeze hash changed")
    sources = verify_sources()
    with np.load(UV_PATH, allow_pickle=False) as uv_file:
        canonical_uv = np.asarray(uv_file["column_uv"])
    if (canonical_uv.shape != (721, 2) or canonical_uv.dtype != np.int32
            or array_sha256(canonical_uv) != UV_SHA256):
        raise AssayError("canonical int32 UV grid differs from the frozen parity-v2 column order")
    fixed_pose_protocol = load_json(FIXED_POSE_PROTOCOL)
    canonical_pose = fixed_pose_protocol["fixed_pose_source"]["fixed_pose"]
    assay_id, cases, global_ray_hashes = _normalize_manifest(manifest, path, canonical_pose)
    source_freeze_path = path.parent / "source_freeze.json"
    if manifest.get("schema_version") == "flygym-recorded-retina-neural-ball-case-manifest-1" and source_freeze_path.is_file():
        source_freeze = load_json(source_freeze_path)
        ray_record = source_freeze.get("ray_array_hashes", {})
        if (ray_record.get("left_directions") != LEFT_RAYS_SHA256
                or ray_record.get("right_directions") != RIGHT_RAYS_SHA256
                or ray_record.get("column_uv") != UV_SHA256):
            raise AssayError("landmark source freeze does not pin parity-v2 ray arrays")
    validated_cases = {}
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or "/" in case_id or ".." in case_id:
            raise AssayError(f"invalid case_id: {case_id!r}")
        if case.get("worker_condition") != "static":
            raise AssayError(f"worker_condition must be static for recorded native columns: {case_id}")
        pose = _validate_pose(case.get("fixed_pose"))
        if pose != canonical_pose:
            raise AssayError(f"case pose differs from frozen fixed-pose source: {case_id}")
        input_record = case.get("input_npz")
        if not isinstance(input_record, dict):
            raise AssayError(f"missing input_npz record: {case_id}")
        input_path = Path(input_record["path"]).expanduser().resolve(strict=True)
        if (input_path.stat().st_size != int(input_record["size_bytes"])
                or sha256(input_path) != input_record["sha256"]):
            raise AssayError(f"input NPZ hash/size differs from manifest: {case_id}")
        with np.load(input_path, allow_pickle=False) as archive:
            if set(archive.files) != {"left_columns", "right_columns", "time_s", "column_uv", "metadata_json"}:
                raise AssayError(f"input NPZ keys differ from worker-v2 contract: {case_id}: {archive.files}")
            left = np.asarray(archive["left_columns"])
            right = np.asarray(archive["right_columns"])
            time_s = np.asarray(archive["time_s"])
            column_uv = np.asarray(archive["column_uv"])
            metadata = json.loads(str(archive["metadata_json"].item()))
        if (left.shape != (FRAME_COUNT, 721) or right.shape != (FRAME_COUNT, 721)
                or left.dtype != np.float32 or right.dtype != np.float32
                or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
                or np.any(left < 0) or np.any(left > 1) or np.any(right < 0) or np.any(right > 1)):
            raise AssayError(f"eye columns must be finite float32 [800,721] in [0,1]: {case_id}")
        if (time_s.dtype != np.float64 or time_s.shape != (FRAME_COUNT,)
                or not np.array_equal(time_s, np.arange(FRAME_COUNT, dtype=np.float64) * FRAME_DT_S)):
            raise AssayError(f"time_s must be exact float64 arange(800)*0.005: {case_id}")
        if column_uv.dtype != np.int32 or column_uv.shape != (721, 2) or not np.array_equal(column_uv, canonical_uv):
            raise AssayError(f"column_uv must be the exact frozen int32 parity-v2 grid: {case_id}")
        expected_metadata = {
            "case_id": case_id, "condition": "static", "onset_s": ONSET_S,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise AssayError(f"native input metadata identity/clock mismatch: {case_id}")
        expected_directions = {"left_directions": LEFT_RAYS_SHA256, "right_directions": RIGHT_RAYS_SHA256,
                               "column_uv": UV_SHA256}
        metadata_rays = _case_ray_hashes(metadata)
        case_rays = case.get("source_ray_hashes", expected_directions)
        if case_rays != expected_directions or (metadata_rays is not None and metadata_rays != expected_directions):
            raise AssayError(f"ray-grid provenance differs from parity-v2 for {case_id}: {metadata_rays}")
        meta_pose = metadata.get("fixed_pose")
        if isinstance(meta_pose, dict) and "pose" in meta_pose:
            meta_pose = meta_pose["pose"]
        if meta_pose is not None and meta_pose != pose:
            raise AssayError(f"NPZ metadata fixed_pose differs from manifest/frozen source: {case_id}")
        if not (np.array_equal(left[:200], np.full((200, 721), 0.5, dtype=np.float32))
                and np.array_equal(right[:200], np.full((200, 721), 0.5, dtype=np.float32))):
            raise AssayError(f"seq 0..199 must be uniform 0.5 gray in both eyes: {case_id}")
        left_hash, right_hash = array_sha256(left), array_sha256(right)
        array_meta = case.get("input_arrays", {})
        expected_left = case.get("expected_left_array_sha256", array_meta.get("left_sha256"))
        expected_right = case.get("expected_right_array_sha256", array_meta.get("right_sha256"))
        if ((expected_left is not None and expected_left != left_hash)
                or (expected_right is not None and expected_right != right_hash)
                or (array_meta.get("column_uv_sha256") not in (None, UV_SHA256))
                or case.get("column_uv_sha256") not in (None, UV_SHA256)):
            raise AssayError(f"manifest eye-array identities differ from NPZ: {case_id}")
        stimulus_meta = case.get("stimulus_metadata", {})
        onset_meta = stimulus_meta.get("onset_s", metadata.get("onset_s", ONSET_S))
        if float(onset_meta) != ONSET_S:
            raise AssayError(f"stimulus metadata onset differs from the recorded frame clock: {case_id}")
        if not stimulus_meta:
            stimulus_meta = metadata.get("stimulus", {})
        validated_cases[case_id] = {
            "case_id": case_id,
            "input_npz": {"path": str(input_path), "size_bytes": input_path.stat().st_size,
                          "sha256": sha256(input_path)},
            "left_array_sha256": left_hash, "right_array_sha256": right_hash,
            "column_uv_sha256": UV_SHA256, "direction_array_sha256": expected_directions,
            "frame_count": FRAME_COUNT, "gray_frames": 200,
            "stimulus_interval_s": [ONSET_S, 4.0],
            "stimulus_metadata": stimulus_meta,
            "metadata_json": metadata,
            "fixed_pose": pose,
        }
    manifest_record = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    canonical_manifest = {"assay_id": assay_id, "source_schema": manifest.get("schema_version", manifest.get("schema")),
                         "cases": cases}
    return {"assay_id": assay_id, "manifest": canonical_manifest, "raw_manifest": manifest,
            "cases": validated_cases,
            "manifest_record": manifest_record}, sources


def _protocol_record(stream: TextIO, timeout_s: float) -> tuple[str, dict[str, Any]]:
    selector = selectors.DefaultSelector()
    selector.register(stream, selectors.EVENT_READ)
    try:
        if not selector.select(timeout_s):
            raise TimeoutError(f"worker produced no protocol record within {timeout_s}s")
    finally:
        selector.close()
    raw = stream.readline()
    if not raw:
        raise EOFError("worker closed stdout without a protocol record")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_NAME:
        raise AssayError(f"worker stdout is not flygym-online-v1: {raw[:240]!r}")
    return raw, value


def _fixed_frame_request(seq: int, pose: dict[str, Any]) -> dict[str, Any]:
    return {"protocol": PROTOCOL_NAME, "type": "frame", "seq": seq,
            "time_s": seq * FRAME_DT_S, "dt_s": FRAME_DT_S, "pose": pose}


def _check_worker_ready(ready: dict[str, Any], case: dict[str, Any], input_record: dict[str, Any]) -> None:
    if ready.get("type") != "ready" or ready.get("source") != "connectome" or ready.get("seed") != SEED:
        raise AssayError("worker ready record is not the expected connectome seed-1729 worker")
    if ready.get("stimulus_condition") != "static" or ready.get("max_frames") != FRAME_COUNT:
        raise AssayError("worker condition/frame cap differs from recorded input")
    for field, expected in (("frame_dt_s", FRAME_DT_S), ("neural_tick_dt_s", 0.0005),
                            ("stimulus_onset_s", ONSET_S), ("static_warmup_s", STATIC_WARMUP_S)):
        if not math.isclose(float(ready.get(field, math.nan)), expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssayError(f"worker {field} differs: {ready.get(field)!r} != {expected}")
    if ready.get("n_columns_per_eye") != 721 or ready.get("neural_steps_per_frame") != 10:
        raise AssayError("worker eye/neural column contract changed")
    expected_backends = {
        "renderer": "flyvis_native_recorded_columns", "vision": "flyvis",
        "vision_device": "cuda:0", "neural": "triton", "neural_device": "cuda",
        "neural_source": "connectome", "motor_decoder": "NeuromuscularAdapter",
    }
    if any(ready.get("effective_backends", {}).get(k) != v for k, v in expected_backends.items()):
        raise AssayError(f"worker did not select strict FlyVis CUDA / MaleCNS Triton backends: {ready.get('effective_backends')}")
    if ready.get("eye_column_hashes") != {"left": LEFT_RAYS_SHA256, "right": RIGHT_RAYS_SHA256}:
        raise AssayError("worker ray-grid hashes differ from frozen parity-v2 L/R rays")
    model = ready.get("malecns_model_parameters", {})
    if float(model.get("gain", math.nan)) != float(np.float32(1.2)):
        raise AssayError(f"worker MaleCNS gain differs from the frozen 1.2 baseline: {model}")
    if ready.get("neural_intervention", {}).get("enabled") is not False:
        raise AssayError("unexpected direct neural intervention is enabled")
    if ready.get("standard_visual_input", {}).get("enabled") is not True:
        raise AssayError("worker did not confirm recorded native eye columns")
    native = ready["standard_visual_input"]
    file_rec = native.get("file", {})
    file_path = Path(file_rec.get("path", "")).resolve()
    file_size = file_rec.get("size_bytes")
    reported_sha = file_rec.get("sha256")
    hash_is_bounded_omission = (reported_sha is None and file_size is not None
                                and file_size > 1024 * 1024
                                and bool(file_rec.get("hash_skipped_reason")))
    if (native.get("mode") != "flyvis_native_standard_columns"
            or file_path != Path(input_record["path"]).resolve()
            or file_size != input_record["size_bytes"]
            or (reported_sha != input_record["sha256"] and not hash_is_bounded_omission)
            or native.get("metadata") != case["metadata_json"]):
        raise AssayError("worker ready native-input identity differs from the frozen case")
    if ready.get("male_cns_steps_during_visual_warmup") != 0:
        raise AssayError("MaleCNS advanced during FlyVis-only pre-worker gray adaptation")
    if ready.get("host_memory_limit_bytes", 0) > 5 * 1024**3:
        raise AssayError("worker did not inherit the required <=5 GiB cgroup cap")


def _check_drive(reply: dict[str, Any], seq: int, ready: dict[str, Any]) -> tuple[float, float, float]:
    expected_time = seq * FRAME_DT_S
    if (reply.get("type") != "drive" or reply.get("seq") != seq
            or not math.isclose(float(reply.get("time_s", math.nan)), expected_time, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(float(reply.get("neural_window_end_s", math.nan)), expected_time + FRAME_DT_S,
                                rel_tol=0.0, abs_tol=1e-9)):
        raise AssayError(f"worker frame/clock mismatch at seq={seq}")
    if reply.get("effective_backends", ready.get("effective_backends")) != ready.get("effective_backends"):
        raise AssayError(f"worker backend changed at seq={seq}")
    rates = reply.get("population_rates_hz")
    if not isinstance(rates, dict) or "DNp15/L" not in rates or "DNp15/R" not in rates:
        raise AssayError(f"worker omitted bilateral DNp15 rates at seq={seq}")
    values = (float(reply.get("leg_power", math.nan)), float(rates["DNp15/L"]), float(rates["DNp15/R"]))
    if not all(math.isfinite(x) for x in values) or not 0.0 <= values[0] <= 1.0 or min(values[1:]) < 0:
        raise AssayError(f"worker P/DNp15 sample invalid at seq={seq}: {values}")
    return values


def _verify_worker_artifacts(worker_dir: Path, case: dict[str, Any], input_record: dict[str, Any],
                             left: np.ndarray, right: np.ndarray, time_s: np.ndarray) -> dict[str, Any]:
    integrity = load_json(worker_dir / "integrity.json")
    if integrity.get("frame_count") != FRAME_COUNT or integrity.get("last_seq") != FRAME_COUNT - 1:
        raise AssayError("worker stop/integrity record does not cover all 800 frames")
    for rec in integrity.get("files", {}).values():
        p = Path(rec["path"])
        if not p.is_file() or p.stat().st_size != rec["size_bytes"] or sha256(p) != rec["sha256"]:
            raise AssayError(f"worker artifact integrity mismatch: {p}")
    with np.load(worker_dir / "eye_columns.npz", allow_pickle=False) as eye:
        if (not np.array_equal(eye["seq"], np.arange(FRAME_COUNT, dtype=np.int64))
                or not np.array_equal(eye["frame_time_s"], time_s)
                or not np.array_equal(eye["left_columns"], left)
                or not np.array_equal(eye["right_columns"], right)):
            raise AssayError("worker did not replay exact frozen binocular input arrays")
    provenance = load_json(worker_dir / "provenance.json")
    native = provenance.get("standard_visual_input", {})
    if native.get("metadata") != case["metadata_json"] or native.get("mode") != "flyvis_native_standard_columns":
        raise AssayError("worker provenance does not retain the exact native input metadata")
    n = 0
    with (worker_dir / "frames.jsonl").open("r", encoding="utf-8") as stream:
        for n, line in enumerate(stream, start=1):
            row = json.loads(line)
            seq = n - 1
            if row.get("seq") != seq or not math.isclose(float(row.get("body_time_s", math.nan)),
                                                            seq * FRAME_DT_S, rel_tol=0.0, abs_tol=1e-9):
                raise AssayError(f"persisted worker frames order/time mismatch at {seq}")
            if row.get("eye_column_hashes") != {"left": array_sha256(left[seq]), "right": array_sha256(right[seq])}:
                raise AssayError(f"persisted worker eye hashes differ from exact input at {seq}")
    if n != FRAME_COUNT:
        raise AssayError(f"worker frames.jsonl has {n} rows, expected {FRAME_COUNT}")
    return {"frame_count": n, "integrity_sha256": sha256(worker_dir / "integrity.json"),
            "provenance_sha256": sha256(worker_dir / "provenance.json"),
            "frames_sha256": sha256(worker_dir / "frames.jsonl"),
            "eye_columns_sha256": sha256(worker_dir / "eye_columns.npz"),
            "input_npz_sha256": input_record["sha256"]}


def _window_summary(trace: dict[str, np.ndarray], start: float, end: float) -> dict[str, Any]:
    times = np.asarray(trace["time_s"])
    mask = (times >= start) & (times < end)
    if not np.any(mask):
        raise AssayError(f"no native samples in body summary window [{start},{end})")
    omega = np.asarray(trace["ball_body_angvel_world_rad_s"])[mask]
    contact = np.asarray(trace["foot_contact_fz_up_native"])[mask]
    return {
        "window_s_half_open": [start, end], "native_samples": int(mask.sum()),
        "ball_body_angvel_world_rad_s_mean": omega.mean(axis=0).astype(float).tolist(),
        "ball_body_angvel_world_rad_s_sd": omega.std(axis=0).astype(float).tolist(),
        "max_abs_ball_angular_velocity_rad_s": float(np.max(np.abs(omega))),
        "positive_fz_fraction_by_leg": np.mean(contact > 0, axis=0).astype(float).tolist(),
        "mean_positive_fz_by_leg_native": np.mean(np.maximum(contact, 0), axis=0).astype(float).tolist(),
    }


def _execute_case(args: argparse.Namespace, manifest_bundle: dict[str, Any], sources: dict[str, Any]) -> int:
    case_id = args.case_id
    if case_id not in manifest_bundle["cases"]:
        raise AssayError(f"case not found in input manifest: {case_id}")
    manifest = manifest_bundle["manifest"]
    case_raw = next(item for item in manifest["cases"] if item["case_id"] == case_id)
    case = manifest_bundle["cases"][case_id]
    auth_path = args.root_review.expanduser().resolve(strict=True)
    auth = load_json(auth_path)
    expected_auth = {
        "schema_version": "flygym-recorded-retina-neural-ball-root-review-1",
        "status": "authorized", "assay_id": manifest_bundle["assay_id"],
        "input_manifest_sha256": manifest_bundle["manifest_record"]["sha256"],
        "core_freeze_sha256": CORE_FREEZE_SHA256,
        "repair_source_freeze_sha256": sha256(OUTPUT_ROOT / "repair-source-freeze.json"),
        "harness_source_sha256": sha256(Path(__file__)),
    }
    if any(auth.get(k) != v for k, v in expected_auth.items()):
        raise PermissionError(f"root review does not bind the current assay manifest/core: {expected_auth}")
    if case_id not in auth.get("authorized_case_ids", []):
        raise PermissionError(f"case is not explicitly authorized: {case_id}")
    if auth.get("resource_limits") != RESOURCE_LIMITS:
        raise PermissionError("root review resource limits differ from the frozen 5G/200%/900s plan")
    if not args.execute:
        raise PermissionError("runtime requires --execute with explicit root review")
    sources_before = verify_sources()
    if not BODY_PYTHON.is_file() or not WORKER_PYTHON.is_file():
        raise FileNotFoundError("pinned FlyGym/worker Python runtime is unavailable")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("a+")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AssayError(f"another assay case holds the neural/body case lock: {LOCK_PATH}") from exc
        run_dir = OUTPUT_ROOT / "runs" / manifest_bundle["assay_id"] / case_id
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite an existing case: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        worker_dir = run_dir / "neural-worker"
        worker_dir.mkdir()
        input_path = Path(case["input_npz"]["path"])
        if sha256(input_path) != case["input_npz"]["sha256"]:
            raise AssayError("input changed between preflight and execution")
        with np.load(input_path, allow_pickle=False) as archive:
            left = np.asarray(archive["left_columns"]).copy()
            right = np.asarray(archive["right_columns"]).copy()
            time_s = np.asarray(archive["time_s"]).copy()
        if (sha256(input_path) != case["input_npz"]["sha256"]
                or array_sha256(left) != case["left_array_sha256"]
                or array_sha256(right) != case["right_array_sha256"]):
            raise AssayError("input reread differs from preflight")
        run_record = {
            "schema_version": "flygym-recorded-retina-neural-ball-run-1", "status": "running",
            "assay_id": manifest_bundle["assay_id"], "case_id": case_id,
            "closed_loop": False, "readout_candidate": True, "suite_assay": case_raw.get("assay"),
            "input_manifest": manifest_bundle["manifest_record"],
            "input_npz": case["input_npz"], "source_hashes_before": sources_before,
            "root_review_path": str(auth_path), "root_review_sha256": sha256(auth_path),
            "resource_limits": RESOURCE_LIMITS, "worker_started": False,
            "body_session_started": False, "stimulus_label_sent_to_decoder_or_controller": False,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "run-record.json", run_record)
        worker_env = os.environ.copy()
        worker_env.update({
            "CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FLYVIS_ROOT_DIR": "/data/models/flyvis", "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        old_pythonpath = worker_env.get("PYTHONPATH", "")
        worker_env["PYTHONPATH"] = str(PROJECT / "scripts") + (os.pathsep + old_pythonpath if old_pythonpath else "")
        worker_command = [
            str(WORKER_PYTHON), str(WORKER), "--config", str(WORKER_CONFIG),
            "--stimulus-condition", "static", "--stimulus-onset-s", "1",
            "--static-warmup-s", "1", "--seed", str(SEED), "--max-frames", str(FRAME_COUNT),
            "--neural-device", "cuda", "--neural-backend", "triton", "--vision-device", "cuda:0",
            "--max-vram-gib", "8.0", "--deterministic-vision", "--deterministic-neural-delivery",
            "--capture-motor-channels", "--capture-neural-state-diagnostic",
            "--standard-visual-input", str(input_path), "--allow-standard-visual-input",
            "--out", str(worker_dir),
        ]
        run_record.update({"worker_command": worker_command, "worker_python": str(WORKER_PYTHON),
                           "worker_started": True})
        write_json(run_dir / "run-record.json", run_record)
        requests_path = run_dir / "worker_requests.jsonl"
        protocol_stdout_path = run_dir / "worker_protocol_stdout.jsonl"
        stderr_path = run_dir / "worker_stderr.log"
        decoder_frames_path = run_dir / "decoded_frames.jsonl"
        worker: subprocess.Popen[str] | None = None
        raw_responses: list[dict[str, Any]] = []
        try:
            with protocol_stdout_path.open("x", encoding="utf-8") as protocol_log, \
                    stderr_path.open("x", encoding="utf-8") as stderr, \
                    requests_path.open("x", encoding="utf-8", buffering=1) as request_log:
                worker = subprocess.Popen(worker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=stderr, text=True, bufsize=1, env=worker_env)
                run_record["worker_pid"] = worker.pid
                write_json(run_dir / "run-record.json", run_record)
                ready_raw, ready = _protocol_record(worker.stdout, 180)
                protocol_log.write(ready_raw); protocol_log.flush()
                _check_worker_ready(ready, case, case["input_npz"])
                run_record["worker_ready"] = ready
                run_record["worker_ready_pass"] = True
                write_json(run_dir / "run-record.json", run_record)
                for seq in range(FRAME_COUNT):
                    request = _fixed_frame_request(seq, case["fixed_pose"])
                    raw_request = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                    request_log.write(raw_request)
                    worker.stdin.write(raw_request); worker.stdin.flush()
                    response_raw, response = _protocol_record(worker.stdout, 90)
                    protocol_log.write(response_raw); protocol_log.flush()
                    power, left_rate, right_rate = _check_drive(response, seq, ready)
                    if seq < 200 and (left_rate != 0.0 or right_rate != 0.0):
                        raise AssayError("fresh gray prelude contradicts zero-rate candidate reference")
                    raw_responses.append({"time_s": seq * FRAME_DT_S, "leg_power": power,
                                          "dn_p15_left_rate_hz": left_rate,
                                          "dn_p15_right_rate_hz": right_rate,
                                          "worker_frame_sha256": hashlib.sha256(response_raw.encode("utf-8")).hexdigest()})
                    if (seq + 1) % 100 == 0:
                        print(f"{case_id}: worker {seq + 1}/{FRAME_COUNT} frames", file=sys.stderr, flush=True)
                worker.stdin.write(json.dumps({"protocol": PROTOCOL_NAME, "type": "stop"}) + "\n")
                worker.stdin.flush()
                stopped_raw, stopped = _protocol_record(worker.stdout, 120)
                protocol_log.write(stopped_raw); protocol_log.flush()
                if (stopped.get("type") != "stopped" or stopped.get("frame_count") != FRAME_COUNT
                        or stopped.get("last_seq") != FRAME_COUNT - 1
                        or not math.isclose(float(stopped.get("last_time_s", math.nan)), 3.995,
                                            rel_tol=0.0, abs_tol=1e-12)):
                    raise AssayError(f"worker terminal stop acknowledgement incomplete: {stopped}")
                worker.stdin.close()
                worker.wait(timeout=120)
                if worker.returncode != 0:
                    raise AssayError(f"worker exited {worker.returncode} after stopped acknowledgement")
                run_record["worker_shutdown"] = stopped
                run_record["worker_exit_code"] = worker.returncode
            run_record["worker_artifact_audit"] = _verify_worker_artifacts(
                worker_dir, case, case["input_npz"], left, right, time_s)
            write_json(run_dir / "run-record.json", run_record)
        except BaseException:
            if worker is not None and worker.poll() is None:
                try:
                    if worker.stdin and not worker.stdin.closed:
                        worker.stdin.write(json.dumps({"protocol": PROTOCOL_NAME, "type": "stop"}) + "\n")
                        worker.stdin.flush()
                        worker.wait(timeout=5)
                except Exception:
                    worker.terminate()
                    try:
                        worker.wait(timeout=5)
                    except Exception:
                        worker.kill()
            raise
        # Keep the Neural CUDA process fully stopped before importing/running the CPU body session.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", "/data/models/flygym/assets")
        sys.path.insert(0, str(PROJECT / "scripts"))
        import run_flygym_tethered_ball_recorded_input_replay as ball_runner
        import run_flygym_tethered_treadmill_pilot as canonicalizer
        from flygym_tethered_treadmill_session_v1 import TetheredPilotConfig
        from dnp15_opponent_steering_decoder_repaired_v1 import DecoderConfig, Dnp15OpponentSteeringDecoder, StaticRateBaseline
        baseline = StaticRateBaseline.from_payload(load_json(BASELINE))
        if (baseline.left_hz != BASELINE_RATES_HZ["DNp15/L"]
                or baseline.right_hz != BASELINE_RATES_HZ["DNp15/R"]):
            raise AssayError("frozen decoder baseline rates differ from 0/142.5 Hz")
        decoder = Dnp15OpponentSteeringDecoder(None, DecoderConfig(), baseline_mode="zero_gray", normalization_hz=142.5)
        from dnp15_opponent_steering_decoder import Dnp15OpponentSteeringDecoder as LegacyDecoder
        shadow_decoder = LegacyDecoder(baseline)
        decoder_provenance = decoder.provenance()
        write_json(run_dir / "decoder-policy.json", {"candidate": decoder_provenance, "legacy_shadow": shadow_decoder.provenance(), "shadow_controls_body": False, "legacy_shadow_scope": "static-c1-stripe-specific; outside matched family diagnostic_only"})
        config = TetheredPilotConfig(duration_s=4.0, seed=SEED, leg_power_p=0.0, differential_ap_d=0.0)
        session = ball_runner.RecordedInputTetheredBallSession(config)
        session.run_id = f"retina-ball-{manifest_bundle['assay_id']}-{case_id}-T4s-seed{SEED}"
        run_record.update({"body_session_started": True, "worker_process_inactive_before_body": True,
                           "decoder_provenance": decoder_provenance,
                           "body_config": {key: value for key, value in vars(config).items()},
                           "neural_forward_completed_before_body": True})
        write_json(run_dir / "run-record.json", run_record)
        world_xml = session.world.mjcf_root.to_xml()
        (run_dir / "actual_world.raw.xml").write_text(world_xml, encoding="utf-8")
        canonical_xml, canonical_audit = canonicalizer.canonicalize_world_xml(world_xml)
        canonical_path = run_dir / "actual_world.canonical.xml"
        canonical_path.write_text(canonical_xml, encoding="utf-8")
        if sha256(canonical_path) != ball_runner.CANONICAL_XML_SHA256:
            raise AssayError("runtime tethered-ball XML differs from the reviewed canonical plant")
        write_json(run_dir / "ball_preflight.json", {
            "status": "pass", "canonical_world_xml_sha256": sha256(canonical_path),
            "session_time_s_before_first_step": session.time_s,
            "control_samples_before_first_step": session._control_samples,
            "native_samples_before_first_step": session._native_samples,
            "physics_steps_before_first_step": session._physics_step_count,
            "canonicalization_audit": canonical_audit,
            "no_physics_step_in_preflight": True,
        })
        if session.time_s != 0.0 or session._physics_step_count != 0:
            raise AssayError("body session advanced before first recorded P/d command")
        held_side = "left"
        tick_count = 0
        decoded_records = []
        with decoder_frames_path.open("x", encoding="utf-8", buffering=1) as decoded_log, \
                (run_dir / "control_ticks.jsonl").open("x", encoding="utf-8", buffering=1) as tick_log:
            for seq, record in enumerate(raw_responses):
                shadow = shadow_decoder.step(record["leg_power"], record["dn_p15_left_rate_hz"], record["dn_p15_right_rate_hz"])
                decoded = decoder.step(record["leg_power"], record["dn_p15_left_rate_hz"],
                                       record["dn_p15_right_rate_hz"])
                if decoded.soma_side is not None:
                    held_side = decoded.soma_side
                decoded_row = {
                    "seq": seq, "frame_time_s": record["time_s"],
                    "leg_power_P": decoded.leg_power_left,
                    "control_power": decoded.control_power,
                    "static_gate_active": decoded.static_gate_active,
                    "power_gate_transition": decoded.power_gate_transition,
                    "raw_P": decoded.raw_leg_power,
                    "legacy_shadow_d": shadow.clipped_opponent_signal,
                    "raw_ema_left_rate_hz": decoded.raw_ema_left_hz,
                    "raw_ema_right_rate_hz": decoded.raw_ema_right_hz,
                    "ungated_opponent_signal": decoded.ungated_opponent_signal,
                    "dn_p15_left_rate_hz": decoded.dn_p15_left_rate_hz,
                    "dn_p15_right_rate_hz": decoded.dn_p15_right_rate_hz,
                    "ema_left_rate_hz": decoded.ema_left_rate_hz,
                    "ema_right_rate_hz": decoded.ema_right_rate_hz,
                    "raw_opponent_signal": decoded.raw_opponent_signal,
                    "signed_d": decoded.clipped_opponent_signal,
                    "command_amplitude": decoded.command_amplitude,
                    "held_soma_side": held_side,
                    "zero_power_baseline_hold": decoded.zero_power_baseline_hold,
                    "stimulus_label_passed_to_decoder": False,
                }
                decoded_log.write(json.dumps(decoded_row, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False) + "\n")
                decoded_records.append(decoded_row)
                frame_time = seq * FRAME_DT_S
                if not math.isclose(session.time_s, frame_time, rel_tol=0.0, abs_tol=1e-9):
                    raise AssayError(f"ball session clock differs before frame {seq}: {session.time_s}")
                for tick in range(CONTROL_TICKS_PER_FRAME):
                    start_time = frame_time + tick * CONTROL_DT_S
                    if not math.isclose(session.time_s, start_time, rel_tol=0.0, abs_tol=1e-9):
                        raise AssayError(f"ball control clock mismatch at frame {seq}, tick {tick}")
                    native_first = int(session._native_samples)
                    outcome = session.step_recorded_control_tick(
                        decoded.control_power, decoded.clipped_opponent_signal, held_side)
                    if (outcome.get("native_samples_added") != NATIVE_STEPS_PER_TICK
                            or outcome.get("descending_signal_lr") != [decoded.control_power] * 2
                            or not math.isclose(session.time_s, start_time + CONTROL_DT_S,
                                                rel_tol=0.0, abs_tol=1e-9)):
                        raise AssayError(f"recorded P/d did not advance the ball plant at 1ms/tick: {seq}/{tick}")
                    tick_record = {
                        "seq": seq, "tick_in_frame": tick,
                        "control_time_s": start_time, "P": decoded.control_power, "raw_P": decoded.raw_leg_power,
                        "static_gate_active": decoded.static_gate_active,
                        "signed_d": decoded.clipped_opponent_signal,
                        "held_soma_side": held_side,
                        "native_sample_indices_halfopen": [native_first, int(session._native_samples)],
                        "native_samples_added": int(outcome["native_samples_added"]),
                        "position_targets_rad": outcome["position_targets_rad"],
                        "controller_telemetry": outcome["controller_telemetry"],
                        "stimulus_label_passed_to_controller": False,
                    }
                    tick_log.write(json.dumps(tick_record, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False) + "\n")
                    tick_count += 1
                if (seq + 1) % 100 == 0:
                    print(f"{case_id}: ball {seq + 1}/{FRAME_COUNT} frames; t={session.time_s:.3f}s",
                          file=sys.stderr, flush=True)
        if (len(decoded_records) != FRAME_COUNT or tick_count != FRAME_COUNT * CONTROL_TICKS_PER_FRAME
                or session._native_samples != 40001 or session._physics_step_count != 40000
                or not math.isclose(session.time_s, 4.0, rel_tol=0.0, abs_tol=1e-9)):
            raise AssayError("body terminal counts differ from 4s / 800 frames / 4000 ticks / 40000 physics steps")
        trajectory_path = run_dir / "trajectory.npz"
        session.save_trajectory(trajectory_path)
        trace = {key: value[:session._native_samples] for key, value in session.trace.items()}
        windows = {"gray_0_1": _window_summary(trace, 0.0, 1.0),
                   "early_1_1p5": _window_summary(trace, 1.0, 1.5),
                   "late_2_4": _window_summary(trace, 2.0, 4.0)}
        frame_times = np.asarray([x["frame_time_s"] for x in decoded_records], dtype=np.float64)
        p_values = np.asarray([x["leg_power_P"] for x in decoded_records], dtype=np.float64)
        d_values = np.asarray([x["signed_d"] for x in decoded_records], dtype=np.float64)
        rates_l = np.asarray([x["dn_p15_left_rate_hz"] for x in decoded_records], dtype=np.float64)
        rates_r = np.asarray([x["dn_p15_right_rate_hz"] for x in decoded_records], dtype=np.float64)
        decoded_windows = {}
        for label, start, end in (("gray_0_1", 0.0, 1.0), ("early_1_1p5", 1.0, 1.5), ("late_2_4", 2.0, 4.0)):
            mask = (frame_times >= start) & (frame_times < end)
            decoded_windows[label] = {
                "frames": int(mask.sum()), "mean_P": float(np.mean(p_values[mask])),
                "mean_control_P": float(np.mean([x["control_power"] for x,m in zip(decoded_records,mask) if m])),
                "static_gate_fraction": float(np.mean([x["static_gate_active"] for x,m in zip(decoded_records,mask) if m])),
                "mean_legacy_shadow_d": float(np.mean([x["legacy_shadow_d"] for x,m in zip(decoded_records,mask) if m])),
                "mean_ungated_signed_d": float(np.mean([x["ungated_opponent_signal"] for x,m in zip(decoded_records,mask) if m])),
                "mean_raw_opponent_signal": float(np.mean([x["raw_opponent_signal"] for x,m in zip(decoded_records,mask) if m])),
                "mean_signed_d": float(np.mean(d_values[mask])), "mean_abs_d": float(np.mean(np.abs(d_values[mask]))),
                "mean_DNp15_L_Hz": float(np.mean(rates_l[mask])),
                "mean_DNp15_R_Hz": float(np.mean(rates_r[mask])),
            }
        summary = {
            "schema_version": "flygym-recorded-retina-neural-ball-summary-1", "status": "complete",
            "assay_id": manifest_bundle["assay_id"], "case_id": case_id,
            "closed_loop": False, "readout_candidate": True, "suite_assay": case_raw.get("assay"),
            "input_npz_sha256": case["input_npz"]["sha256"],
            "input_manifest_sha256": manifest_bundle["manifest_record"]["sha256"],
            "core_freeze_sha256": CORE_FREEZE_SHA256,
            "worker_frame_count": FRAME_COUNT, "worker_neural_ticks": FRAME_COUNT * 10,
            "body_control_ticks": tick_count, "body_native_samples": session._native_samples,
            "body_physics_steps": session._physics_step_count, "body_final_time_s": session.time_s,
            "stimulus_label_or_target_to_controller": False,
            "worker_before_body_sequential": True,
            "decoder": decoder_provenance,
            "neural_windows": decoded_windows, "ball_windows": windows,
            "trajectory_sha256": sha256(trajectory_path),
            "contacts_sha256": sha256(run_dir / "contacts.npz"),
            "decoded_frames_sha256": sha256(decoder_frames_path),
            "control_ticks_sha256": sha256(run_dir / "control_ticks.jsonl"),
            "run_interpretation": "recorded retinal input drives frozen neural outputs and an engineered DNp15-to-template decoder; ball omega/contact are mechanics outputs, not biological preference or natural DNp15-muscle wiring",
        }
        write_json(run_dir / "case_summary.json", summary)
        sources_after = verify_sources()
        if sources_after != sources_before:
            raise AssayError("a frozen source/core file changed during the case")
        run_record.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                           "source_hashes_after": sources_after,
                           "summary_sha256": sha256(run_dir / "case_summary.json"),
                           "trajectory_sha256": summary["trajectory_sha256"],
                           "contacts_sha256": summary["contacts_sha256"],
                           "body_control_ticks": tick_count, "body_native_samples": session._native_samples,
                           "body_physics_steps": session._physics_step_count})
        write_json(run_dir / "run-record.json", run_record)
        integrity_files = ["case_summary.json", "run-record.json", "decoded_frames.jsonl",
                           "control_ticks.jsonl", "trajectory.npz", "contacts.npz",
                           "neural-worker/frames.jsonl", "neural-worker/provenance.json",
                           "neural-worker/integrity.json", "neural-worker/eye_columns.npz",
                           "neural-worker/muscle_channels.npz", "neural-worker/neural_state_diagnostic.npz",
                           "worker_requests.jsonl", "worker_protocol_stdout.jsonl", "worker_stderr.log"]
        integrity = {"schema_version": "flygym-recorded-retina-neural-ball-integrity-1",
                     "case_id": case_id, "status": "complete",
                     "files": {name: {"path": str(run_dir / name), "size_bytes": (run_dir / name).stat().st_size,
                                      "sha256": sha256(run_dir / name)} for name in integrity_files}}
        write_json(run_dir / "run-integrity.json", integrity)
        print(json.dumps({"status": "complete", "case_id": case_id, "run_dir": str(run_dir),
                          "summary": summary}, sort_keys=True), flush=True)
        session.close()
        return 0
    except BaseException as exc:
        if "run_dir" in locals() and run_dir.exists():
            failed_record = locals().get("run_record", {})
            failed_record.update({"status": "failed_partial_preserved", "error": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc(),
                                  "completed_utc": datetime.now(timezone.utc).isoformat()})
            try:
                write_json(run_dir / "failure.json", {"case_id": case_id, "error": failed_record["error"],
                                                       "traceback": failed_record["traceback"]})
                write_json(run_dir / "run-record.json", failed_record)
                partial = locals().get("session")
                if partial is not None and getattr(partial, "_native_samples", 0) > 0:
                    arrays = {key: value[:partial._native_samples] for key, value in partial.trace.items()}
                    np.savez_compressed(run_dir / "trajectory_partial.npz", **arrays)
            except Exception:
                pass
            try:
                if locals().get("session") is not None:
                    session.close()
            except Exception:
                pass
        raise
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-input-manifest", action="store_true",
                        help="validate NPZ/schema/source pins on CPU only; no worker or physics")
    parser.add_argument("--case-id")
    parser.add_argument("--root-review", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    bundle, sources = validate_manifest(args.manifest)
    if args.check_input_manifest:
        if args.case_id is not None or args.root_review is not None or args.execute:
            parser.error("--check-input-manifest cannot be combined with execution arguments")
        print(json.dumps({"status": "cpu_input_manifest_pass", "assay_id": bundle["assay_id"],
                          "manifest": bundle["manifest_record"], "core_freeze_sha256": CORE_FREEZE_SHA256,
                          "source_hashes": sources, "cases": bundle["cases"],
                          "worker_forward": False, "body_physics": False, "gpu_used": False},
                         indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.case_id is None or args.root_review is None or not args.execute:
        parser.error("execution requires --case-id, --root-review, and --execute")
    return _execute_case(args, bundle, sources)


if __name__ == "__main__":
    raise SystemExit(main())
