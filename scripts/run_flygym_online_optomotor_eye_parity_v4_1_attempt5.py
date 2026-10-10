#!/usr/bin/env python3
"""Run a synchronous FlyGym → FlyVis → MaleCNS closed-loop optomotor trial.

The FlyGym body advances under its existing 1 ms CPG/position-servo interface.
At each 5 ms boundary this driver sends the measured eye-camera midpoint pose
to a separate GPU neural worker, waits for the matching decoded bilateral
drive, then holds that drive over the next five 1 ms controller ticks. It does
not select a motor response from the stimulus condition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import select
import subprocess
import sys
import time
from typing import Any, TextIO
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BODY_RUNNER = PROJECT_ROOT / "scripts" / "run_flygym_reference_walk.py"
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "flygym_online_neural_worker_eye_parity_v3.py"
WORKER_V2_SCRIPT = PROJECT_ROOT / "scripts" / "flygym_online_neural_worker_eye_parity_v2.py"
WORKER_CONFIG = PROJECT_ROOT / "config" / "flygym_online_neural_worker.json"
BODY_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
DEFAULT_WORKER_PYTHON = Path(
    "/data/environments/projects/flyvis-opticlobe/bin/python"
)
ARTIFACT_ROOT = Path("/data/artifacts/by-project/malecns-gpu/engineering")
ATTEMPT3_BUNDLE_ROOT = ARTIFACT_ROOT / "flygym-online-eye-parity-v4-1-attempt3-20261006-v1"
ONLINE_BUNDLE_ROOT = ARTIFACT_ROOT / "flygym-online-eye-parity-v4-1-attempt5-20261006-v1"
PREPARE_ROOT = ONLINE_BUNDLE_ROOT / "prepare"
DEFAULT_OUTPUT_ROOT = ONLINE_BUNDLE_ROOT / "runs"
ARTIFACT_CONTAINMENT_ROOT = Path("/data/artifacts/by-project/malecns-gpu")
PHASE_SPEED_ROOT = ARTIFACT_ROOT / "flygym-phase-speed-repair-20261006-v1" / "development" / "v4-1"
WEAK_REFERENCE_ROOT = PHASE_SPEED_ROOT / "weak_0p30"
V4_SOURCE_FREEZE = PHASE_SPEED_ROOT / "source_freeze.json"
V4_FREEZE_MANIFEST = PHASE_SPEED_ROOT / "freeze_manifest.json"
V4_PROTOCOL = PHASE_SPEED_ROOT / "protocol.json"
V4_CONTROLLER_CONFIG = WEAK_REFERENCE_ROOT / "controller_config.json"
V4_COMPILED_MODEL = WEAK_REFERENCE_ROOT / "compiled_model.xml"
V4_INITIAL_STATE = WEAK_REFERENCE_ROOT / "initial_state.json"
V4_REFERENCE_MANIFEST = WEAK_REFERENCE_ROOT / "manifest.json"
V4_REFERENCE_INTEGRITY = WEAK_REFERENCE_ROOT / "integrity.json"
LEGACY_REST_FACTORY = PROJECT_ROOT / "scripts" / "flygym_legacy_rest_compat.py"
PREPROGRAMMED_SOURCE = Path(
    "/data/environments/projects/flygym-reference/lib/python3.12/site-packages/"
    "flygym_demo/complex_terrain/preprogrammed.py"
)
PREPROGRAMMED_STEPS_ASSET = Path(
    "/data/environments/projects/flygym-reference/lib/python3.12/site-packages/"
    "flygym_demo/complex_terrain/assets/single_steps_untethered.pkl"
)
FLYGym_VISION_CONFIG = Path(
    "/data/environments/projects/flygym-reference/lib/python3.12/site-packages/"
    "flygym/assets/model/neuromechfly/vision.yaml"
)
FLYGym_RIGGING_CONFIG = FLYGym_VISION_CONFIG.with_name("rigging.yaml")
PARITY_CANDIDATE_ROOT = ARTIFACT_ROOT / "visual-motor-feasibility-20261006-v1" / "eye-side-parity-candidate-v2"
PARITY_STATIC_AUDIT = PARITY_CANDIDATE_ROOT / "candidate-static-audit.json"
PARITY_INTEGRITY = PARITY_CANDIDATE_ROOT / "integrity.json"
RETINAL_REFERENCE = ARTIFACT_ROOT / "visual-motor-feasibility-20261006-v1" / "retinal-direction" / "retinal-samples.npz"
TARGET_DIRECTIONS_ASSET = PROJECT_ROOT / "artifacts/flyvis-eye/t4t5_dirs.npz"
FROZEN_CPG_FREQUENCY_HZ = 7.4074074074074066
FROZEN_CONTROLLER_SHA256 = "126b641e2998992e4cad351f4a48292882063851350e4006ba40552a38d7e2eb"
FROZEN_V3_SHA256 = "a2eadd1d85f76f74cd0fa7510b5614b9357f610027689c47e236b809070dd3dd"
FROZEN_V2_SHA256 = "b133d95152c1c37e9204908d8d123b317d1375ad6d59dfacd15a4496b0cd52f8"
FROZEN_BASE_SHA256 = "030cc38d89010b25efdb751a0458bbbd15a507f3d112413bfe319d0ae31b5b88"
FROZEN_HELPER_SHA256 = "88d9586c758e41444980e9a5b19eb92ab2f7a5a90f4ac695253fab1d130f2a5b"
FROZEN_FACTORY_SHA256 = "4538575af7a946641be99aac66fb53b8ecbbe5cb7333723191621122c00195ee"
FROZEN_TEMPLATE_SHA256 = "1e5b28bb6b3f50ac95a04773ea37af28d05e26d03bd90fdba57fa1b3eacfaf8c"
FROZEN_COMPILED_MODEL_SHA256 = "c4db94617aae9f138af2526fb9fa751f99ed230602e4e7e5c9670c1bf8421631"
FROZEN_COMPILED_MODEL_KNOWN_CONTENT_TYPE_VARIANT_SHA256 = "bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57"
FROZEN_CONTROLLER_CONFIG_SHA256 = "c8d4266aecaa942a0e954b2363b047ba703e7507073adb0967af508115006415"
FROZEN_SOURCE_FREEZE_SHA256 = "f7dc3d758857ddeda4923194027311bd90f53aeff9d57515f6e4987f786f8bd0"
FROZEN_WEAK_INTEGRITY_SHA256 = "d8bf6b7ea9cfa34cd6af07d2054fb422d4e485fcb31878eb3d7a111ad6d226d9"
EXPECTED_BODY_ENV = Path("/data/environments/projects/flygym-reference/bin/python")
EXPECTED_WORKER_ENV = Path("/data/environments/projects/flyvis-opticlobe/bin/python")
EXPECTED_NEURAL_GAIN = 1.2
EXPECTED_T4T5_TARGET_COUNT = 13582
EXPECTED_MOTOR_CHANNEL_COUNT = 381
FROZEN_WORKER_V2_SHA256 = "23579a136547698b7eec9cd4a0ca2c3a8f8d5d0144601b08f67dcac383cc7ccb"
FROZEN_TARGET_DIRECTIONS_SHA256 = "66bedaf757e967fdd89bcd00bd3be7e49f477df54fe025212c180d12e6f749ed"
CASE_SPEC_PATHS = {
    "static-c1-seed1729": ARTIFACT_ROOT / "flygym-fixed-pose-angular-calibration-20261006-v1" / "pilot-runs/attempt-4/c1/stationary-c1-seed1729/neural-worker/angular_stimulus_spec.json",
    "positive-head-azimuth-60degps-c1-seed1729": ARTIFACT_ROOT / "flygym-fixed-pose-angular-calibration-20261006-v1" / "pilot-runs/attempt-4/c1/positive-head-azimuth-60degps-c1-seed1729/neural-worker/angular_stimulus_spec.json",
    "negative-head-azimuth-60degps-c1-seed1729": ARTIFACT_ROOT / "flygym-fixed-pose-angular-calibration-20261006-v1" / "pilot-runs/attempt-4/c1/negative-head-azimuth-60degps-c1-seed1729/neural-worker/angular_stimulus_spec.json",
}
CASE_EXPECTED = {
    "static-c1-seed1729": {"head_direction_sign": 1, "speed_deg_s": 0.0, "temporal_frequency_hz": 0.0},
    "positive-head-azimuth-60degps-c1-seed1729": {"head_direction_sign": 1, "speed_deg_s": 60.0, "temporal_frequency_hz": 1.0},
    "negative-head-azimuth-60degps-c1-seed1729": {"head_direction_sign": -1, "speed_deg_s": -60.0, "temporal_frequency_hz": 1.0},
}
ATTEMPT3_STATIC_READY_PROVENANCE = (
    ATTEMPT3_BUNDLE_ROOT / "runs/static-c1-seed1729/neural-worker/provenance.json"
)
ATTEMPT3_STATIC_FAILURE_RECORD = (
    ATTEMPT3_BUNDLE_ROOT / "runs/static-c1-seed1729/failure.json"
)
ATTEMPT3_STATIC_CAPTURED_STIMULUS = (
    ATTEMPT3_BUNDLE_ROOT / "runs/static-c1-seed1729/neural-worker/angular_stimulus_spec.json"
)
SUCCESSFUL_TERMINAL_CAPTURE_FIXTURE = (
    ARTIFACT_ROOT / "flygym-fixed-pose-angular-calibration-20261006-v1"
    / "pilot-runs/attempt-4/c1/stationary-c1-seed1729/neural-worker"
)
SUCCESSFUL_TERMINAL_READY_PROTOCOL = (
    SUCCESSFUL_TERMINAL_CAPTURE_FIXTURE.parent / "worker_protocol_stdout.jsonl"
)
PROTOCOL = "flygym-online-v1"
FRAME_DT_S = 0.005
NEURAL_STEPS_PER_FRAME = 10
CONTROL_DT_S = 0.001
NATIVE_DT_S = 0.0001
CONTROL_TICKS_PER_FRAME = 5
NATIVE_STEPS_PER_CONTROL_TICK = 10
MAX_VRAM_GIB = 8.0
PARENT_MEMORY_MAX = "7G"
MAX_WORKER_CAPTURE_FRAMES = 4000
DEFAULT_MAX_DURATION_S = 2.0
MAX_BOUNDED_DURATION_S = 10.0
MAX_SHUTDOWN_TIMEOUT_S = 60.0
DEFAULT_PANEL_VELOCITY_VIRTUALWORLD_MPS = {
    "left": np.asarray([0.0, 0.004, 0.0], dtype=np.float64),
    "right": np.asarray([0.0, -0.004, 0.0], dtype=np.float64),
    "static": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
}
WORLD_REFLECTION = np.diag([1.0, -1.0, 1.0])
AXIAL_REFLECTION = np.linalg.det(WORLD_REFLECTION) * WORLD_REFLECTION

COMPILED_MODEL_ARRAY_FIELDS = (
    "body_parentid", "body_mass", "body_inertia", "body_ipos", "body_iquat",
    "body_pos", "body_quat", "geom_bodyid", "geom_type", "geom_size",
    "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity", "geom_condim",
    "geom_friction", "geom_solref", "geom_solimp", "geom_rgba", "jnt_bodyid",
    "jnt_pos", "jnt_axis", "jnt_stiffness", "jnt_type", "jnt_limited",
    "jnt_range", "jnt_qposadr", "jnt_dofadr", "qpos0", "qpos_spring",
    "dof_damping", "dof_armature", "dof_frictionloss", "actuator_trnid",
    "actuator_ctrllimited", "actuator_ctrlrange", "actuator_forcelimited",
    "actuator_forcerange", "actuator_gaintype", "actuator_biastype",
    "actuator_dynprm", "actuator_gainprm", "actuator_biasprm", "pair_geom1",
    "pair_geom2", "mesh_vert", "mesh_face", "mesh_vertadr", "mesh_vertnum",
    "mesh_faceadr", "mesh_facenum", "mesh_graphadr", "mesh_graph", "mesh_scale",
    "mesh_pos", "mesh_quat", "key_time", "key_qpos", "key_qvel", "key_ctrl", "key_act",
)
ALLOWED_MESH_CONTENT_TYPES = {None, "model/stl"}


def _expected_panel_velocity_virtualworld_mps(
    condition: str, speed_override_mm_s: float | None
) -> np.ndarray:
    configured = DEFAULT_PANEL_VELOCITY_VIRTUALWORLD_MPS[condition]
    if speed_override_mm_s is None:
        return configured.copy()
    requested_mps = float(speed_override_mm_s) / 1000.0
    configured_speed = float(np.linalg.norm(configured))
    if requested_mps == 0.0:
        return np.zeros(3, dtype=np.float64)
    if configured_speed <= 0.0:
        raise ValueError("positive panel speed requires a configured moving-condition direction")
    return configured * (requested_mps / configured_speed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return array


def _quat_wxyz_to_matrix(value: Any) -> np.ndarray:
    q = _finite_vector(value, (4,), "quaternion_wxyz")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion_wxyz must have nonzero norm")
    w, x, y, z = q / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _matrix_to_quat_wxyz(matrix: Any) -> np.ndarray:
    r = _finite_vector(matrix, (3, 3), "rotation_matrix")
    if not np.allclose(r.T @ r, np.eye(3), rtol=0.0, atol=2e-6) or not math.isclose(
        float(np.linalg.det(r)), 1.0, rel_tol=0.0, abs_tol=2e-6
    ):
        raise ValueError("rotation matrix must be a proper orthonormal matrix")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
        q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                      (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
        q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                      0.25 * s, (r[1, 2] + r[2, 1]) / s])
    else:
        s = math.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
        q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                      (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def _virtualworld_rpy_from_matrix(matrix: Any) -> tuple[float, float, float]:
    """Extract angles for VirtualWorld's Rz(yaw) Ry(-pitch) Rx(roll) basis."""
    r = _finite_vector(matrix, (3, 3), "virtual_rotation_matrix")
    pitch = math.asin(max(-1.0, min(1.0, float(r[2, 0]))))
    cp = math.cos(pitch)
    if abs(cp) <= 1e-7:
        raise ValueError("VirtualWorld Euler pitch is too close to its singularity")
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    return yaw, pitch, roll


def _check_finite_tree(value: Any, where: str = "reply") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite_tree(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite_tree(child, f"{where}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} contains a non-finite number")


class NeuralWorker:
    def __init__(self, command: list[str], env: dict[str, str], timeout_s: float,
                 expected_renderer: str = "virtual_world_raycast"):
        if expected_renderer not in {
            "virtual_world_raycast", "flyvis_native_recorded_columns",
            "fixed_arena_angular_field_cpu",
        }:
            raise ValueError("unsupported explicitly expected renderer")
        self.command = command
        self.timeout_s = timeout_s
        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.stdin: TextIO = self.process.stdin
        self.stdout: TextIO = self.process.stdout
        try:
            self.ready = self._read_record(timeout_s)
            self._validate_ready(self.ready, expected_renderer)
        except BaseException:
            try:
                self.stdin.close()
            except OSError:
                pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            raise

    def _read_record(self, timeout_s: float) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"neural worker exited early with {self.process.returncode}")
        ready, _, _ = select.select([self.stdout], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"neural worker produced no JSONL record within {timeout_s:g} s")
        line = self.stdout.readline()
        if not line:
            raise RuntimeError(f"neural worker closed stdout (exit={self.process.poll()})")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"neural worker emitted malformed JSONL: {line[:400]!r}") from exc
        if not isinstance(record, dict):
            raise ValueError("neural worker records must be JSON objects")
        _check_finite_tree(record)
        return record

    @staticmethod
    def _validate_ready(record: dict[str, Any], expected_renderer: str = "virtual_world_raycast") -> None:
        if record.get("protocol") != PROTOCOL or record.get("type") != "ready":
            raise ValueError("worker startup must begin with a flygym-online-v1 ready record")
        expected = {
            "renderer": expected_renderer,
            "vision": "flyvis",
            "vision_device": "cuda:0",
            "neural": "triton",
            "neural_device": "cuda",
            "neural_source": "connectome",
            "motor_decoder": "NeuromuscularAdapter",
        }
        backends = record.get("effective_backends")
        if not isinstance(backends, dict):
            raise ValueError("worker ready record lacks effective_backends")
        for key, required in expected.items():
            if backends.get(key) != required:
                raise RuntimeError(
                    f"worker backend mismatch for {key}: expected {required!r}, "
                    f"got {backends.get(key)!r}"
                )
        if record.get("source") != "connectome":
            raise RuntimeError("worker ready source must be connectome")
        if expected_renderer == "flyvis_native_recorded_columns":
            native = record.get("standard_visual_input", {})
            if (native.get("enabled") is not True
                    or native.get("classification") != "controlled_visual_calibration"):
                raise RuntimeError("native columns require the declared standard visual calibration mode")
        if expected_renderer == "fixed_arena_angular_field_cpu":
            angular = record.get("angular_stimulus")
            if (record.get("stimulus_mode") != "fixed_arena_angular_field"
                    or not isinstance(angular, dict)
                    or angular.get("spec_sha256") != record.get("angular_stimulus_spec_sha256")
                    or record.get("standard_visual_input", {}).get("native_recorded_columns_used") is not False):
                raise RuntimeError("angular renderer requires an acknowledged live spec and no recorded-column bypass")
        if record.get("n_columns_per_eye") != 721:
            raise RuntimeError("worker must report the fitted 721-column-per-eye input")
        if record.get("neural_steps_per_frame") != 10 or not math.isclose(
            float(record.get("neural_tick_dt_s", float("nan"))),
            0.0005, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("worker must integrate ten 0.5 ms neural ticks per 5 ms frame")

    def request(self, record: dict[str, Any]) -> tuple[dict[str, Any], float]:
        if self.process.poll() is not None:
            raise RuntimeError(f"neural worker exited with {self.process.returncode}")
        started = time.monotonic()
        self.stdin.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.stdin.flush()
        reply = self._read_record(self.timeout_s)
        return reply, time.monotonic() - started

    def stop(self, timeout_s: float = 10.0) -> dict[str, Any]:
        ack: dict[str, Any] | None = None
        stop_error: BaseException | None = None
        if self.process.poll() is None:
            try:
                self.stdin.write(json.dumps({"protocol": PROTOCOL, "type": "stop"}) + "\n")
                self.stdin.flush()
                ack = self._read_record(timeout_s)
                if ack.get("protocol") != PROTOCOL or ack.get("type") != "stopped":
                    raise RuntimeError("worker stop response must be flygym-online-v1/stopped")
            except BaseException as exc:
                stop_error = exc
            try:
                self.stdin.close()
            except OSError:
                pass
        try:
            return_code = self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            self.process.terminate()
            try:
                return_code = self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                return_code = self.process.wait()
            raise RuntimeError(
                f"worker did not exit after stop ack; forced termination, exit={return_code}"
            ) from exc
        if stop_error is not None:
            raise RuntimeError(
                f"worker stop acknowledgement failed; child exit={return_code}: {stop_error}"
            ) from stop_error
        if return_code != 0:
            raise RuntimeError(f"neural worker exited with status {return_code}")
        return {"ack": ack, "return_code": return_code}


def _load_vision_camera_config(session: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the pinned camera sensor offsets through FlyGym's own asset path."""
    vision_path = Path(session.fly.vision_config_path)
    if not vision_path.is_file():
        raise FileNotFoundError(f"FlyGym vision config unavailable: {vision_path}")
    import yaml

    value = yaml.safe_load(vision_path.read_text(encoding="utf-8"))
    sensors = value.get("sensors") if isinstance(value, dict) else None
    if not isinstance(sensors, dict):
        raise ValueError("FlyGym vision config has no sensors mapping")
    selected = {}
    for eye in ("left", "right"):
        sensor_name = "l_eye_cam" if eye == "left" else "r_eye_cam"
        sensor = sensors.get(sensor_name)
        if not isinstance(sensor, dict):
            raise ValueError(f"FlyGym vision config lacks {sensor_name}")
        parent = str(sensor.get("parent", ""))
        if parent != ("l_eye" if eye == "left" else "r_eye"):
            raise ValueError(f"unexpected parent for {sensor_name}: {parent!r}")
        selected[eye] = {
            "sensor_name": sensor_name,
            "parent_body": parent,
            "rel_pos_mm": _finite_vector(sensor.get("rel_pos"), (3,),
                                         f"{sensor_name}.rel_pos").tolist(),
            "orientation_euler_rad": _finite_vector(
                sensor.get("orientation"), (3,), f"{sensor_name}.orientation"
            ).tolist(),
        }
    return vision_path, selected


def _body_ids_by_name(session: Any) -> dict[str, int]:
    return {str(name): int(body_id)
            for name, body_id in zip(session.body_names, session.body_ids)}


def _body_velocity_world(session: Any, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros(6, dtype=np.float64)
    session.mj.mj_objectVelocity(session.model, session.data,
                                 session.mj.mjtObj.mjOBJ_BODY, int(body_id),
                                 velocity, 0)
    omega = _finite_vector(velocity[:3], (3,), "body angular velocity")
    linear = _finite_vector(velocity[3:], (3,), "body linear velocity")
    return omega, linear


def _camera_geometry(session: Any, sensors: dict[str, Any], body_ids: dict[str, int]
                     ) -> dict[str, Any]:
    data = session.data
    origins: dict[str, np.ndarray] = {}
    for eye in ("left", "right"):
        body_id = body_ids.get(sensors[eye]["parent_body"])
        if body_id is None or body_id <= 0:
            raise RuntimeError(f"compiled camera parent body is unavailable for {eye}")
        body_rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        origin = np.asarray(data.xpos[body_id], dtype=np.float64) + body_rotation @ np.asarray(
            sensors[eye]["rel_pos_mm"], dtype=np.float64
        )
        if not np.all(np.isfinite(origin)):
            raise ValueError(f"non-finite FlyGym {eye} camera origin")
        origins[eye] = origin
    delta = origins["left"] - origins["right"]
    separation_mm = float(np.linalg.norm(delta))
    if not math.isfinite(separation_mm) or separation_mm <= 0:
        raise ValueError("FlyGym eye-camera centers do not define a positive baseline")
    return {
        "left_origin_mm": origins["left"],
        "right_origin_mm": origins["right"],
        "midpoint_mm": 0.5 * (origins["left"] + origins["right"]),
        "baseline_m": separation_mm * 1e-3,
        "baseline_vector_mm": delta,
    }


def _build_virtual_pose(
    session: Any,
    sensors: dict[str, Any],
    body_ids: dict[str, int],
    eye_origin_xy_m: tuple[float, float],
    initial_eye_midpoint_mm: np.ndarray,
) -> dict[str, Any]:
    thorax_id = int(session.thorax_body_id)
    data = session.data
    thorax_position_mm = _finite_vector(data.xpos[thorax_id], (3,), "thorax position")
    thorax_quaternion = _finite_vector(data.xquat[thorax_id], (4,), "thorax quaternion")
    thorax_rotation = _quat_wxyz_to_matrix(thorax_quaternion)
    # The compiled c_head alias is fused to c_thorax and has neutral identity
    # rigging; the official eye-camera centers are fixed under the thorax.
    camera = _camera_geometry(session, sensors, body_ids)
    eye_midpoint_mm = camera["midpoint_mm"]
    omega_fg, thorax_linear_mm_s = _body_velocity_world(session, thorax_id)
    eye_offset_fg_m = (eye_midpoint_mm - thorax_position_mm) * 1e-3
    eye_velocity_fg_m_s = thorax_linear_mm_s * 1e-3 + np.cross(omega_fg, eye_offset_fg_m)

    # FlyGym's +Y points fly-left. VirtualWorld's raster camera-right is +Y at
    # yaw=0, and it maps FlyVis head +Y through -camera_right. Conjugating the
    # body rotation and reflecting world polar vectors preserves the same rays.
    virtual_rotation = WORLD_REFLECTION @ thorax_rotation @ WORLD_REFLECTION
    virtual_quaternion = _matrix_to_quat_wxyz(virtual_rotation)
    yaw, pitch, roll = _virtualworld_rpy_from_matrix(virtual_rotation)
    virtual_eye_delta_m = (eye_midpoint_mm - initial_eye_midpoint_mm) * 1e-3
    virtual_position_m = np.asarray([
        eye_origin_xy_m[0] + virtual_eye_delta_m[0],
        eye_origin_xy_m[1] - virtual_eye_delta_m[1],
        eye_midpoint_mm[2] * 1e-3,
    ], dtype=np.float64)
    virtual_velocity_m_s = WORLD_REFLECTION @ eye_velocity_fg_m_s
    virtual_omega_rad_s = AXIAL_REFLECTION @ omega_fg

    left_origin_vw = np.asarray([
        eye_origin_xy_m[0] + (camera["left_origin_mm"][0] - initial_eye_midpoint_mm[0]) * 1e-3,
        eye_origin_xy_m[1] - (camera["left_origin_mm"][1] - initial_eye_midpoint_mm[1]) * 1e-3,
        camera["left_origin_mm"][2] * 1e-3,
    ])
    right_origin_vw = np.asarray([
        eye_origin_xy_m[0] + (camera["right_origin_mm"][0] - initial_eye_midpoint_mm[0]) * 1e-3,
        eye_origin_xy_m[1] - (camera["right_origin_mm"][1] - initial_eye_midpoint_mm[1]) * 1e-3,
        camera["right_origin_mm"][2] * 1e-3,
    ])

    virtual = {
        "position_m": virtual_position_m,
        "quaternion_wxyz": virtual_quaternion,
        "yaw_rad": yaw,
        "pitch_rad": pitch,
        "roll_rad": roll,
        "velocity_mps": virtual_velocity_m_s,
        "angular_velocity_radps": virtual_omega_rad_s,
    }
    for key, value in virtual.items():
        if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
            raise ValueError(f"mapped pose field {key} is non-finite")
    return {
        "pose": {key: (value.tolist() if isinstance(value, np.ndarray) else float(value))
                 for key, value in virtual.items()},
        "source_pose": {
            "thorax_position_mm": thorax_position_mm.tolist(),
            "thorax_quaternion_wxyz": (thorax_quaternion / np.linalg.norm(thorax_quaternion)).tolist(),
            "thorax_rotation_body_to_world": thorax_rotation.tolist(),
            "thorax_linear_velocity_mm_s": thorax_linear_mm_s.tolist(),
            "thorax_angular_velocity_radps": omega_fg.tolist(),
            "head_frame": "neutral c_head rigging fused to c_thorax; no independent head joint",
            "eye_camera_midpoint_mm": eye_midpoint_mm.tolist(),
            "eye_camera_left_origin_mm": camera["left_origin_mm"].tolist(),
            "eye_camera_right_origin_mm": camera["right_origin_mm"].tolist(),
            "eye_camera_baseline_m": camera["baseline_m"],
            "eye_velocity_flygym_world_mps": eye_velocity_fg_m_s.tolist(),
        },
        "virtual_eye_origins_m": {
            "left": left_origin_vw.tolist(),
            "right": right_origin_vw.tolist(),
        },
    }


def _validate_drive_reply(reply: dict[str, Any], seq: int, frame_time_s: float,
                          ready: dict[str, Any]) -> tuple[float, float]:
    if reply.get("protocol") != PROTOCOL or reply.get("type") != "drive":
        raise ValueError(f"worker reply {seq} must be a flygym-online-v1 drive record")
    if reply.get("seq") != seq:
        raise ValueError(f"worker drive sequence mismatch: sent {seq}, got {reply.get('seq')!r}")
    reply_time = reply.get("time_s")
    window_end = reply.get("neural_window_end_s")
    if not isinstance(reply_time, (int, float)) or not math.isclose(
        float(reply_time), frame_time_s, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(f"worker drive time mismatch at seq={seq}: {reply_time!r}")
    if not isinstance(window_end, (int, float)) or not math.isclose(
        float(window_end), frame_time_s + FRAME_DT_S, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(f"worker neural window must end at frame_time+5ms (seq={seq})")
    backends = reply.get("effective_backends", ready.get("effective_backends"))
    if backends != ready.get("effective_backends"):
        raise RuntimeError(f"effective neural backends changed at seq={seq}")
    left = reply.get("left_descending_drive")
    right = reply.get("right_descending_drive")
    for name, value in (("left_descending_drive", left), ("right_descending_drive", right)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"worker {name} is missing or non-finite at seq={seq}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"worker {name} must preserve normalized [0,1] output at seq={seq}")
    return float(left), float(right)


def _validate_motor_channel_capture_metadata(
    metadata: Any,
    *,
    output_dir: Path,
    expected_frames: int,
    stage: str,
) -> dict[str, Any]:
    """Validate MotorChannelCapture's real schema; it has no enabled=True field."""
    if not isinstance(metadata, dict):
        raise RuntimeError("worker motor-channel capture metadata is absent")
    if metadata.get("enabled") is False:
        raise RuntimeError("worker explicitly disabled motor-channel capture")
    if metadata.get("schema_version") != 1:
        raise RuntimeError("worker motor-channel capture metadata schema is not version 1")
    if metadata.get("total_leg_motor_cells") != EXPECTED_MOTOR_CHANNEL_COUNT:
        raise RuntimeError("worker motor-channel capture does not describe the frozen 381-cell membership")
    if metadata.get("max_frames") != expected_frames:
        raise RuntimeError("worker motor-channel capture frame capacity differs from the frozen run")
    expected_captured = 0 if stage == "ready" else expected_frames
    expected_shape = [expected_captured, EXPECTED_MOTOR_CHANNEL_COUNT]
    if (metadata.get("frames_captured") != expected_captured
            or metadata.get("counts_shape") != expected_shape
            or metadata.get("counts_dtype") != "uint16"):
        raise RuntimeError(f"worker motor-channel {stage} metadata has an invalid count shape/dtype")
    if (metadata.get("window_steps") != NEURAL_STEPS_PER_FRAME
            or not math.isclose(float(metadata.get("window_dt_s", float("nan"))),
                                FRAME_DT_S, rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError("worker motor-channel window does not match the 10-step / 5 ms neural frame")
    counts = metadata.get("actual_counts", {})
    if (not isinstance(counts, dict) or counts.get("shape") != expected_shape
            or counts.get("dtype") != "uint16"):
        raise RuntimeError(f"worker motor-channel {stage} actual_counts summary is inconsistent")
    expected_membership = metadata.get("expected_membership_counts", {})
    mapped = metadata.get("mapped_cells")
    unmapped = metadata.get("unmapped_cells")
    if (not isinstance(expected_membership, dict)
            or expected_membership.get("total") != EXPECTED_MOTOR_CHANNEL_COUNT
            or expected_membership.get("mapped") != mapped
            or expected_membership.get("unmapped") != unmapped
            or not isinstance(mapped, int) or not isinstance(unmapped, int)
            or mapped + unmapped != EXPECTED_MOTOR_CHANNEL_COUNT):
        raise RuntimeError("worker motor-channel membership totals do not sum to the expected 381 cells")
    expected_output = (output_dir / "muscle_channels.npz").resolve()
    expected_membership_path = (output_dir / "muscle_channels_membership.json").resolve()
    if (Path(str(metadata.get("output", ""))).expanduser().resolve() != expected_output
            or Path(str(metadata.get("membership_path", ""))).expanduser().resolve()
            != expected_membership_path):
        raise RuntimeError(f"worker motor-channel {stage} output paths do not match its run directory")
    expected_mapping = (PROJECT_ROOT / "config/leg_muscle_map.json").resolve()
    if Path(str(metadata.get("mapping_path", ""))).expanduser().resolve() != expected_mapping:
        raise RuntimeError("worker motor-channel mapping path differs from the frozen leg map")
    return {
        "schema_version": 1,
        "stage": stage,
        "metadata_shape": expected_shape,
        "max_frames": expected_frames,
        "window_steps": NEURAL_STEPS_PER_FRAME,
        "window_dt_s": FRAME_DT_S,
        "total_leg_motor_cells": EXPECTED_MOTOR_CHANNEL_COUNT,
        "mapped_cells": mapped,
        "unmapped_cells": unmapped,
        "output_path_matches_run_directory": True,
        "membership_path_matches_run_directory": True,
        "mapping_path_matches_frozen_config": True,
        "enabled_evidence": "complete MotorChannelCapture metadata; no enabled=true field is emitted by schema v1",
    }


def _validate_eye_parity_ready(
    ready: dict[str, Any], freeze: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Fail closed unless this process is using the frozen parity V3 path."""
    if ready.get("source") != "connectome" or ready.get("seed") != 1729:
        raise RuntimeError("worker ready record is not the registered MaleCNS connectome seed 1729")
    expected_backends = {
        "renderer": "fixed_arena_angular_field_cpu",
        "vision": "flyvis",
        "vision_device": "cuda:0",
        "neural": "triton",
        "neural_device": "cuda",
        "neural_source": "connectome",
        "motor_decoder": "NeuromuscularAdapter",
    }
    backends = ready.get("effective_backends")
    if not isinstance(backends, dict) or backends != expected_backends:
        raise RuntimeError(f"strict V3 backend identity mismatch: {backends!r}")
    if ready.get("max_frames") != 2000 or ready.get("n_columns_per_eye") != 721:
        raise RuntimeError("worker ready record does not match 2,000 frames / 721 columns per eye")
    if ready.get("n_t4t5_targets") != EXPECTED_T4T5_TARGET_COUNT:
        raise RuntimeError("worker mapped T4/T5 target count differs from the frozen 13,582 targets")
    if not math.isclose(float(ready.get("frame_dt_s", float("nan"))), FRAME_DT_S,
                        rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("worker visual frame cadence is not 5 ms")
    if not math.isclose(float(ready.get("neural_tick_dt_s", float("nan"))), 0.0005,
                        rel_tol=0.0, abs_tol=1e-12) or ready.get("neural_steps_per_frame") != 10:
        raise RuntimeError("worker neural integration is not ten 0.5 ms steps per visual frame")
    raw_host_limit = ready.get("host_memory_limit_bytes")
    if (isinstance(raw_host_limit, bool) or not isinstance(raw_host_limit, (int, float))
            or not math.isfinite(float(raw_host_limit))):
        raise RuntimeError("worker host-memory cap is not finite")
    host_limit = int(raw_host_limit)
    if not 0 < host_limit <= 7 * 1024**3:
        raise RuntimeError(f"worker reports an invalid host-memory cgroup limit: {host_limit}")

    angular = ready.get("angular_stimulus")
    if (ready.get("stimulus_mode") != "fixed_arena_angular_field"
            or ready.get("stimulus_condition") != "angular"
            or ready.get("angular_stimulus_spec_sha256") != args.angular_stimulus_spec_sha256
            or not isinstance(angular, dict)
            or ready.get("standard_visual_input", {}).get("native_recorded_columns_used") is not False):
        raise RuntimeError("worker did not acknowledge live arena angular rendering without recorded-image bypass")
    if not math.isclose(float(ready.get("stimulus_onset_s", float("nan"))), 5.0,
                        rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("worker angular renderer onset does not match the five-second gray interval")
    if not math.isclose(float(ready.get("static_warmup_s", float("nan"))), 5.0,
                        rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("worker angular renderer warmup does not match the preregistered timeline")
    ready_stimulus = angular.get("metadata", angular)
    if not math.isclose(float(ready_stimulus.get("contrast_study_parameter_c", float("nan"))), 1.0,
                        rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("worker angular contrast differs from c=1")
    if not math.isclose(float(ready_stimulus.get("wavelength_deg", float("nan"))), 60.0,
                        rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("worker angular period differs from 60 degrees")
    if not math.isclose(float(ready_stimulus.get("duration_s", float("nan"))), 5.0,
                        rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("worker angular stimulus interval is not five seconds")

    params = ready.get("malecns_model_parameters", {})
    if (not math.isclose(float(params.get("gain", float("nan"))), EXPECTED_NEURAL_GAIN,
                         rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(float(params.get("gain_configured_by_world_experiment", float("nan"))),
                                EXPECTED_NEURAL_GAIN, rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError("worker effective MaleCNS neural gain is not the frozen 1.2")
    if ready.get("neural_intervention", {}).get("enabled") is not False:
        raise RuntimeError("worker must run without an optogenetic/current intervention")
    deterministic_vision = ready.get("deterministic_vision", {})
    if (deterministic_vision.get("requested") is not True
            or deterministic_vision.get("effective_flags", {}).get("settings_applied") is not True):
        raise RuntimeError("worker did not apply the requested deterministic FlyVis settings")
    deterministic_delivery = ready.get("deterministic_neural_delivery", {})
    if (deterministic_delivery.get("requested") is not True
            or deterministic_delivery.get("enabled") is not True):
        raise RuntimeError("worker did not enable requested deterministic spike delivery")

    visual_capture = ready.get("visual_drive_capture", {})
    motor_capture = ready.get("motor_channel_capture", {})
    if visual_capture.get("enabled") is not True:
        raise RuntimeError("worker did not enable visual-drive capture")
    artifact_dir_raw = ready.get("artifact_dir")
    if not isinstance(artifact_dir_raw, str) or not artifact_dir_raw:
        raise RuntimeError("worker ready record omitted its artifact output directory")
    motor_capture_validation = _validate_motor_channel_capture_metadata(
        motor_capture, output_dir=Path(artifact_dir_raw).expanduser().resolve(),
        expected_frames=2000, stage="ready",
    )
    if ready.get("motor_source") != "decoded_leg_power_leg_steering_mix":
        raise RuntimeError("worker motor source differs from the frozen NeuromuscularAdapter path")

    assets = ready.get("source_assets", {})
    source_asset_freeze_audit = _verify_worker_ready_assets_locked(ready, freeze)
    flyvis_assets = assets.get("flyvis", {})
    worker_assets = assets.get("worker_source", {})
    expected_freeze = freeze.get("flyvis_model_files", {})
    expected_policy = {"L": -1, "R": 1}
    if worker_assets.get("candidate", {}).get("sha256") != _sha256_file(WORKER_SCRIPT):
        raise RuntimeError("worker candidate source hash does not match the frozen V3 worker")
    parent_record = worker_assets.get("parent_v2", {})
    if (parent_record.get("sha256") != FROZEN_WORKER_V2_SHA256
            or parent_record.get("frozen_sha256") != FROZEN_WORKER_V2_SHA256
            or parent_record.get("resource_only_changes_in_v3") is not True):
        raise RuntimeError("worker V3 does not acknowledge its exact V2 parent source")
    adapter_record = flyvis_assets.get("adapter_source", {})
    if adapter_record.get("sha256") != _sha256_file(PROJECT_ROOT / "scripts/flyvis_runtime_adapter_eye_parity_v2.py"):
        raise RuntimeError("worker loaded a different FlyVis parity adapter source")
    policy = flyvis_assets.get("eye_coordinate_policy", {})
    if (policy.get("azimuth_parity_by_registered_side") != expected_policy
            or policy.get("parity_applied_to") != [
                "ray_sampling", "nearest_spatial_target_column_mapping"
            ]):
        raise RuntimeError("worker runtime did not apply L=-1/R=+1 to both ray and nearest-target mapping")
    mapping_hashes = policy.get("mapping_assignment_hashes", {})
    if set(mapping_hashes) != {
        "gathered_node_indices_sha256", "gather_group_sizes_sha256", "eye_batch_rows_sha256"
    } or any(not isinstance(value, str) or len(value) != 64 for value in mapping_hashes.values()):
        raise RuntimeError("worker runtime omitted actual mapping assignment hashes")
    if flyvis_assets.get("n_targets") != EXPECTED_T4T5_TARGET_COUNT:
        raise RuntimeError("loaded FlyVis adapter target count differs from the frozen asset")
    if flyvis_assets.get("target_mapping", {}).get("sha256") != FROZEN_TARGET_DIRECTIONS_SHA256:
        raise RuntimeError("worker loaded a different fitted T4/T5 target-direction asset")
    expected_directions = freeze.get("eye_parity_static_guard", {}).get("expected_ray_hashes", {})
    if expected_directions != {
        "L": "2f847637ae64bea5a5454aa90e86e440a23d76d5c3b85a866936e502691654f1",
        "R": "5037edbec7e4709ddc5f31643de643cdadef4492145756e72634d52727b7b4f6",
    }:
        raise RuntimeError("prepare bundle does not bind the reviewed V2 left-parity ray hashes")
    source_direction_hashes = flyvis_assets.get("direction_hashes", {})
    ready_direction_hashes = ready.get("eye_column_hashes", {})
    for side, eye in (("L", "left"), ("R", "right")):
        if source_direction_hashes.get(side) != expected_directions[side]:
            raise RuntimeError(f"runtime {side} ray-grid hash differs from the static parity audit")
        if ready_direction_hashes.get(eye) != expected_directions[side]:
            raise RuntimeError(f"ready {eye} ray-grid hash differs from the static parity audit")
    actual_models = flyvis_assets.get("model_files", {})
    for name, expected in expected_freeze.items():
        actual = actual_models.get(name, {})
        if actual.get("sha256") != expected.get("sha256"):
            raise RuntimeError(f"worker loaded a different FlyVis model file: {name}")
    expected_environment = EXPECTED_WORKER_ENV.parent.parent.resolve()
    runtime_environment = Path(str(flyvis_assets.get("environment_path", ""))).resolve()
    if runtime_environment != expected_environment:
        raise RuntimeError("worker FlyVis environment path differs from the frozen package fingerprint")
    return {
        "backend_gate": "pass",
        "registered_side_parity": policy.get("azimuth_parity_by_registered_side"),
        "mapped_target_count": flyvis_assets.get("n_targets"),
        "target_mapping_assignment_hashes": mapping_hashes,
        "direction_hashes": expected_directions,
        "worker_source_sha256": worker_assets.get("candidate", {}).get("sha256"),
        "worker_parent_v2_sha256": parent_record.get("sha256"),
        "adapter_source_sha256": adapter_record.get("sha256"),
        "worker_host_memory_limit_bytes": host_limit,
        "source_asset_freeze_audit": source_asset_freeze_audit,
        "neural_gain": params.get("gain"),
        "visual_drive_capture": visual_capture,
        "motor_channel_capture": motor_capture,
        "motor_channel_capture_validation": motor_capture_validation,
    }


def _run_id(condition: str, body_seed: int, neural_seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    return f"flygym-online-optomotor-{condition}-body{body_seed}-neural{neural_seed}-{stamp}"


def _project_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_angular_spec(path: Path) -> tuple[Any, dict[str, Any], bytes]:
    try:
        raw_bytes = path.read_bytes()
        document = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"could not read angular stimulus spec {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "angular-optomotor-stimulus-spec-1":
        raise ValueError("angular stimulus spec must use schema_version angular-optomotor-stimulus-spec-1")
    if set(document) != {"schema_version", "display_geometry", "stimulus"}:
        raise ValueError("angular stimulus spec has unexpected or missing top-level fields")
    if document.get("display_geometry") != "fixed_arena_infinite_distance_angular_field":
        raise ValueError("angular stimulus spec has an unsupported display_geometry")
    values = document.get("stimulus")
    expected = {
        "name", "contrast", "head_direction_sign", "wavelength_deg",
        "temporal_frequency_hz", "mean_luminance", "azimuth_extent_deg",
        "elevation_extent_deg", "onset_s", "duration_s", "phase_at_onset_rad",
    }
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("angular stimulus spec must contain exactly the documented stimulus fields")
    if isinstance(values.get("head_direction_sign"), bool) or not isinstance(
        values.get("head_direction_sign"), int
    ):
        raise ValueError("angular stimulus head_direction_sign must be integer +1 or -1")
    try:
        from standard_optomotor_stimulus import AngularOptomotorStimulus

        stimulus = AngularOptomotorStimulus(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid angular stimulus parameters: {exc}") from exc
    return stimulus, document, raw_bytes


def _load_registered_case(case_id: str) -> tuple[Any, dict[str, Any], bytes]:
    if case_id not in CASE_SPEC_PATHS:
        raise ValueError(f"case is not preregistered: {case_id}")
    stimulus, document, raw = _load_angular_spec(CASE_SPEC_PATHS[case_id])
    expected = CASE_EXPECTED[case_id]
    values = document["stimulus"]
    if (values.get("contrast") != 1.0
            or values.get("wavelength_deg") != 60.0
            or values.get("onset_s") != 5.0
            or values.get("duration_s") != 5.0
            or values.get("head_direction_sign") != expected["head_direction_sign"]
            or values.get("temporal_frequency_hz") != expected["temporal_frequency_hz"]
            or not math.isclose(stimulus.angular_speed_deg_s, expected["speed_deg_s"],
                                rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError(f"registered case {case_id} differs from its static/±60 deg/s protocol")
    if document.get("display_geometry") != "fixed_arena_infinite_distance_angular_field":
        raise ValueError(f"registered case {case_id} changes its fixed-arena display geometry")
    return stimulus, document, raw


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _python_package_fingerprint(executable: Path, packages: tuple[str, ...]) -> dict[str, Any]:
    """Read interpreter/package metadata only; this does not import model runtimes."""
    script = (
        "import importlib.metadata as m,json,platform,sys; "
        f"names={list(packages)!r}; "
        "versions={}; "
        "\nfor n in names:\n"
        " try: versions[n]=m.version(n)\n"
        " except m.PackageNotFoundError: versions[n]=None\n"
        "print(json.dumps({'executable':sys.executable,'version':sys.version,'platform':platform.platform(),'packages':versions},sort_keys=True))"
    )
    result = subprocess.run(
        [str(executable), "-c", script], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    record = json.loads(result.stdout)
    record["resolved_executable"] = str(executable.resolve())
    return record


def _assert_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen {label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"frozen {label} changed: {path} sha256={actual} expected={expected}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual}


def _verify_v4_1_source_contract() -> dict[str, Any]:
    if _sha256_file(V4_SOURCE_FREEZE) != FROZEN_SOURCE_FREEZE_SHA256:
        raise RuntimeError("the V4.1 development source-freeze record changed")
    freeze = json.loads(V4_SOURCE_FREEZE.read_text(encoding="utf-8"))
    phase_freeze = json.loads(V4_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    controller_config = json.loads(V4_CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    effective = json.loads((WEAK_REFERENCE_ROOT / "effective_config.json").read_text(encoding="utf-8"))
    reference_manifest = json.loads(V4_REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    reference_integrity = json.loads(V4_REFERENCE_INTEGRITY.read_text(encoding="utf-8"))

    if phase_freeze.get("source_freeze_sha256") != FROZEN_SOURCE_FREEZE_SHA256:
        raise RuntimeError("V4.1 freeze manifest does not identify the frozen source-freeze record")
    if phase_freeze.get("development_gate_pass") is not False:
        raise RuntimeError("V4.1 development gate classification changed; review protocol before use")
    if freeze.get("candidate_module_expected_sha256") != FROZEN_CONTROLLER_SHA256:
        raise RuntimeError("source-freeze controller hash differs from the registered V4.1 controller")
    if freeze.get("controller_config") != controller_config.get("config"):
        raise RuntimeError("weak_0p30 controller config differs from the V4.1 source freeze")
    if protocol.get("candidate_config_effective") != controller_config.get("config"):
        raise RuntimeError("V4.1 global controller config differs from weak_0p30 controller config")
    if controller_config.get("controller_class") != "flygym_phase_speed_gait_v4_1.PhaseSpeedGaitController":
        raise RuntimeError("V4.1 controller class identity changed")
    if controller_config.get("model_rest_condition") != "legacy_rest_compatibility=True":
        raise RuntimeError("V4.1 source case was not built with legacy-rest compatibility")
    if controller_config.get("no_cartesian_controller") is not True:
        raise RuntimeError("V4.1 reference did not use the registered phase-speed controller")
    if controller_config.get("no_external_wrench_or_root_state_write") is not True:
        raise RuntimeError("V4.1 reference source violates the no-wrench/no-root-write boundary")
    walk = effective.get("walk_config", {})
    if walk.get("legacy_rest_compatibility") is not True:
        raise RuntimeError("V4.1 body reference was not legacy-rest compatible")
    if int(walk.get("seed", -1)) != 1729 or int(effective.get("seed", -1)) != 1729:
        raise RuntimeError("V4.1 weak_0p30 reference was not seeded with body seed 1729")
    if not math.isclose(float(walk.get("cpg_frequency_hz", float("nan"))),
                        FROZEN_CPG_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("V4.1 reference CPG frequency differs from the frozen 7.407407 Hz rhythm")
    if not math.isclose(float(walk.get("control_timestep_s", float("nan"))), CONTROL_DT_S,
                        rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("V4.1 reference control tick is not 1 ms")
    if not math.isclose(float(walk.get("native_timestep_s", float("nan"))), NATIVE_DT_S,
                        rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("V4.1 reference native physics timestep is not 100 us")

    source_specs = {
        "controller": (PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v4_1.py", FROZEN_CONTROLLER_SHA256),
        "ancestor_v3": (PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v3.py", FROZEN_V3_SHA256),
        "ancestor_v2": (PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v2.py", FROZEN_V2_SHA256),
        "ancestor_base": (PROJECT_ROOT / "scripts/flygym_phase_speed_gait.py", FROZEN_BASE_SHA256),
        "pose_priority_helper": (PROJECT_ROOT / "scripts/flygym_pose_priority_ik_v2.py", FROZEN_HELPER_SHA256),
        "legacy_rest_factory": (LEGACY_REST_FACTORY, FROZEN_FACTORY_SHA256),
        "preprogrammed_template_source": (PREPROGRAMMED_SOURCE, "ad945cee5af607e24f1a17ada0b8becd622c1fabebbb9f4f35e49dc7fd69e97f"),
        "preprogrammed_template_asset": (PREPROGRAMMED_STEPS_ASSET, FROZEN_TEMPLATE_SHA256),
        "controller_config": (V4_CONTROLLER_CONFIG, FROZEN_CONTROLLER_CONFIG_SHA256),
        "compiled_model_reference": (V4_COMPILED_MODEL, FROZEN_COMPILED_MODEL_SHA256),
        "weak_integrity_record": (V4_REFERENCE_INTEGRITY, FROZEN_WEAK_INTEGRITY_SHA256),
    }
    records = {name: _assert_hash(path, expected, name)
               for name, (path, expected) in source_specs.items()}
    expected_ancestors = {"flygym_phase_speed_gait_v3": FROZEN_V3_SHA256,
                          "flygym_phase_speed_gait_v2": FROZEN_V2_SHA256,
                          "flygym_phase_speed_gait": FROZEN_BASE_SHA256}
    if {item["module"]: item["sha256"] for item in freeze.get("ancestor_module_chain", [])} != expected_ancestors:
        raise RuntimeError("V4.1 frozen ancestor module chain changed")
    if freeze.get("dependency_modules") != [{"module": "flygym_pose_priority_ik_v2", "sha256": FROZEN_HELPER_SHA256}]:
        raise RuntimeError("V4.1 helper dependency identity changed")
    freeze_sources = freeze.get("sources", {})
    for key, expected in (
        ("controller", FROZEN_CONTROLLER_SHA256),
        ("legacy_rest_factory", FROZEN_FACTORY_SHA256),
        ("native_preprogrammed_py", records["preprogrammed_template_source"]["sha256"]),
        ("native_assets_single_steps_untethered_pkl", FROZEN_TEMPLATE_SHA256),
    ):
        record = freeze_sources.get(key, {})
        if record.get("sha256") != expected:
            raise RuntimeError(f"V4.1 source freeze entry changed: {key}")
    freeze_outputs = phase_freeze.get("outputs", {})
    for rel, path, expected in (
        ("weak_0p30/controller_config.json", V4_CONTROLLER_CONFIG, FROZEN_CONTROLLER_CONFIG_SHA256),
        ("weak_0p30/compiled_model.xml", V4_COMPILED_MODEL, FROZEN_COMPILED_MODEL_SHA256),
        ("weak_0p30/initial_state.json", V4_INITIAL_STATE, "ad389756d8935b4d31920fd9cc45154faf61d6f8c863372fb14ce2811a4a3586"),
    ):
        record = freeze_outputs.get(rel, {})
        if record.get("sha256") != expected or _sha256_file(path) != expected:
            raise RuntimeError(f"V4.1 freeze manifest output mismatch: {rel}")
    integrity_files = reference_integrity.get("files", {})
    for path in (V4_COMPILED_MODEL, V4_INITIAL_STATE, V4_REFERENCE_MANIFEST,
                 WEAK_REFERENCE_ROOT / "effective_config.json"):
        item = integrity_files.get(str(path))
        if not isinstance(item, dict) or item.get("sha256") != _sha256_file(path):
            raise RuntimeError(f"V4.1 weak_0p30 integrity record does not bind {path}")
    return {
        "source_freeze_sha256": FROZEN_SOURCE_FREEZE_SHA256,
        "freeze_manifest_sha256": _sha256_file(V4_FREEZE_MANIFEST),
        "controller_config_sha256": FROZEN_CONTROLLER_CONFIG_SHA256,
        "controller_config": controller_config["config"],
        "body_reference": {
            "case": "weak_0p30; legacy_rest_compatibility=True; side drives [0.3,0.3] only in this reference audit",
            "live_drive_source": "current MaleCNS descending L/R output; never hard-coded [0.3,0.3] or [1,1]",
            "cpg_frequency_hz": FROZEN_CPG_FREQUENCY_HZ,
            "control_dt_s": CONTROL_DT_S,
            "native_dt_s": NATIVE_DT_S,
            "compiled_model_sha256": FROZEN_COMPILED_MODEL_SHA256,
            "weak_reference_manifest_sha256": _sha256_file(V4_REFERENCE_MANIFEST),
            "development_gate_pass": False,
        },
        "frozen_sources": records,
        "source_freeze_record": freeze,
    }


def _verify_eye_parity_geometry() -> dict[str, Any]:
    if not PARITY_STATIC_AUDIT.is_file() or not PARITY_INTEGRITY.is_file():
        raise FileNotFoundError("the frozen V2 eye-parity candidate static audit is unavailable")
    static = json.loads(PARITY_STATIC_AUDIT.read_text(encoding="utf-8"))
    integrity = json.loads(PARITY_INTEGRITY.read_text(encoding="utf-8"))
    for rel, record in integrity.get("files", {}).items():
        path = PARITY_CANDIDATE_ROOT / rel
        if (not path.is_file() or path.stat().st_size != int(record["size_bytes"])
                or _sha256_file(path) != record["sha256"]):
            raise RuntimeError(f"V2 eye-parity static-bundle integrity mismatch: {path}")
    if static.get("candidate_status") != "not executed; opt-in source candidate only":
        raise RuntimeError("V2 eye-parity artifact is not a static-only candidate record")
    expected_sources = {
        "scripts/flyvis_runtime_adapter_eye_parity_v2.py": "993eb7e193c5d8a49f8e88412b6c5378bc9cd1ab026dc9896e2c5ea31bf320e5",
        "scripts/flygym_online_neural_worker_eye_parity_v2.py": "23579a136547698b7eec9cd4a0ca2c3a8f8d5d0144601b08f67dcac383cc7ccb",
    }
    _assert_hash(TARGET_DIRECTIONS_ASSET, FROZEN_TARGET_DIRECTIONS_SHA256,
                 "FlyVis T4/T5 fitted direction asset")
    target_input_record = static.get("input_files", {}).get(str(TARGET_DIRECTIONS_ASSET), {})
    if target_input_record.get("sha256") != FROZEN_TARGET_DIRECTIONS_SHA256:
        raise RuntimeError("V2 static geometry audit does not bind the frozen T4/T5 target directions")

    for name, expected in expected_sources.items():
        if static.get("source_files", {}).get(name, {}).get("sha256") != expected:
            raise RuntimeError(f"V2 parity static report does not bind {name}")
        live = PROJECT_ROOT / name
        _assert_hash(live, expected, name)
    policy = static.get("policy", {})
    if policy.get("azimuth_parity_by_registered_side") != {"L": -1, "R": 1}:
        raise RuntimeError("V2 eye-parity candidate policy is not the registered L-only reflection")
    if policy.get("reflection") != "M(u,v)=(u+v,-v)" or policy.get("applied_to") != [
        "ray_sampling", "nearest_spatial_target_column_mapping"
    ]:
        raise RuntimeError("V2 eye-parity policy does not bind the ray and mapping transforms")
    mapping = static.get("nearest_column_mapping_equivariance", [])
    if len(mapping) != 2 or any(row.get("nearest_uv_mismatch_count") != 0
                                or row.get("nearest_uv_exact_equivariance_count") != row.get("target_count")
                                for row in mapping):
        raise RuntimeError("V2 nearest-column equivalence guard failed")

    with np.load(RETINAL_REFERENCE, allow_pickle=False) as data:
        uv = np.asarray(data["column_uv"], dtype=np.int32)
        left = np.asarray(data["eye_directions_left"], dtype=np.float32)
        right = np.asarray(data["eye_directions_right"], dtype=np.float32)
    if uv.shape != (721, 2) or np.unique(uv, axis=0).shape != (721, 2):
        raise RuntimeError("saved retinal geometry must contain 721 unique FlyVis columns")
    lookup = {tuple(row): i for i, row in enumerate(uv.tolist())}
    try:
        mirror = np.asarray([lookup[(int(u + v), int(-v))] for u, v in uv], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("M(u,v)=(u+v,-v) does not cover the saved retinal lattice") from exc
    if not np.array_equal(mirror[mirror], np.arange(721, dtype=np.int64)):
        raise RuntimeError("saved retinal mirror mapping is not a 721-column involution")
    derived_candidate_left = left[mirror]
    derived_candidate_right = right.copy()
    ray_checks = static.get("ray_equivariance", {})
    if _array_sha256(derived_candidate_left) != ray_checks.get("candidate_left_ray_array_sha256"):
        raise RuntimeError("independent V2 L-ray hash does not equal the frozen candidate")
    if _array_sha256(derived_candidate_right) != ray_checks.get("candidate_right_ray_array_sha256"):
        raise RuntimeError("independent V2 R-ray hash does not equal the frozen candidate")
    if ray_checks.get("candidate_left_matches_legacy_reflected_uv_max_abs_error") != 0.0:
        raise RuntimeError("V2 L-ray candidate is not exactly the legacy mirror permutation")
    if ray_checks.get("candidate_right_matches_legacy_max_abs_error") != 0.0:
        raise RuntimeError("V2 candidate changed the R ray array")
    return {
        "candidate_static_audit_sha256": _sha256_file(PARITY_STATIC_AUDIT),
        "candidate_integrity_sha256": _sha256_file(PARITY_INTEGRITY),
        "source_policy": policy,
        "uv_shape": list(uv.shape),
        "uv_sha256": _array_sha256(uv),
        "mirror_index_sha256": _array_sha256(mirror),
        "mirror_bijection_and_involution": True,
        "expected_ray_hashes": {
            "L": ray_checks["candidate_left_ray_array_sha256"],
            "R": ray_checks["candidate_right_ray_array_sha256"],
        },
        "nearest_target_counts": {row["registered_side"]: row["target_count"] for row in mapping},
        "nearest_target_mismatches": {row["registered_side"]: row["nearest_uv_mismatch_count"] for row in mapping},
    }


def _verify_worker_v3_resource_delta() -> dict[str, Any]:
    """Verify V3 is a resource-guard-only fork of the frozen V2 worker."""
    _assert_hash(WORKER_V2_SCRIPT, FROZEN_WORKER_V2_SHA256, "V2 parity worker parent")
    v2 = WORKER_V2_SCRIPT.read_text(encoding="utf-8")
    v3 = WORKER_SCRIPT.read_text(encoding="utf-8")
    replacements = (
        (
            "MAX_HOST_MEMORY_BYTES = 5 * 1024**3\n",
            "MAX_HOST_MEMORY_BYTES = 7 * 1024**3\n"
            "FROZEN_V2_PARENT_SHA256 = \"23579a136547698b7eec9cd4a0ca2c3a8f8d5d0144601b08f67dcac383cc7ccb\"\n",
        ),
        (
            "worker must inherit the existing <=5 GiB host-memory cgroup cap ",
            "worker must inherit the registered <=7 GiB host-memory cgroup cap ",
        ),
        (
            '                "legacy": _small_file_record(SCRIPTS / "flygym_online_neural_worker.py"),\n',
            '                "legacy": _small_file_record(SCRIPTS / "flygym_online_neural_worker.py"),\n'
            '                "parent_v2": {\n'
            '                    **_small_file_record(SCRIPTS / "flygym_online_neural_worker_eye_parity_v2.py"),\n'
            '                    "frozen_sha256": FROZEN_V2_PARENT_SHA256,\n'
            '                    "resource_only_changes_in_v3": True,\n'
            '                },\n',
        ),
        (
            '"host_memory_policy": "verified inherited cgroup limit, at most 5 GiB",',
            '"host_memory_policy": "verified inherited cgroup limit, at most 7 GiB",',
        ),
    )
    expected = v2
    for old, new in replacements:
        count = expected.count(old)
        if count != 1:
            raise RuntimeError(f"V2 worker resource-delta anchor must occur exactly once: {old!r}")
        expected = expected.replace(old, new, 1)
    if expected != v3:
        raise RuntimeError("V3 worker has source changes beyond the authorized memory-cap/provenance delta")
    return {
        "parent_v2_path": str(WORKER_V2_SCRIPT),
        "parent_v2_sha256": _sha256_file(WORKER_V2_SCRIPT),
        "candidate_v3_path": str(WORKER_SCRIPT),
        "candidate_v3_sha256": _sha256_file(WORKER_SCRIPT),
        "diff_whitelist": [
            "MAX_HOST_MEMORY_BYTES: 5 GiB to 7 GiB",
            "guard message: <=5 GiB to <=7 GiB",
            "host-memory provenance: at most 5 GiB to at most 7 GiB",
            "source_assets.worker_source.parent_v2 exact SHA reference",
        ],
        "all_other_worker_source_bytes_match_v2": True,
    }


def _source_input_paths() -> dict[str, Path]:
    flyvis_model_view = Path("/data/models/flyvis/results/flow/0000/000")
    csr_root = Path("/data/cache/malecns-gpu/csr-flyb")
    paths = {
        "new_online_wrapper": Path(__file__).resolve(),
        "legacy_online_wrapper": PROJECT_ROOT / "scripts/run_flygym_online_optomotor.py",
        "body_runner": BODY_RUNNER,
        "worker_v2": WORKER_V2_SCRIPT,
        "worker_v3": WORKER_SCRIPT,
        "worker_legacy": PROJECT_ROOT / "scripts/flygym_online_neural_worker.py",
        "worker_config": WORKER_CONFIG,
        "flyvis_model_best_checkpoint": flyvis_model_view / "best_chkpt",
        "flyvis_model_metadata": flyvis_model_view / "_meta.yaml",
        "world_experiment_config": PROJECT_ROOT / "config/optomotor_pilot_experiment.json",
        "world_stimulus_config": PROJECT_ROOT / "config/optomotor_pilot_stimuli.json",
        "world_scene_manifest": PROJECT_ROOT / "config/optomotor_panel_scene.json",
        "controller_v4_1": PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v4_1.py",
        "controller_v3": PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v3.py",
        "controller_v2": PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v2.py",
        "controller_base": PROJECT_ROOT / "scripts/flygym_phase_speed_gait.py",
        "controller_helper": PROJECT_ROOT / "scripts/flygym_pose_priority_ik_v2.py",
        "legacy_rest_factory": LEGACY_REST_FACTORY,
        "flygym_template_source": PREPROGRAMMED_SOURCE,
        "flygym_template_asset": PREPROGRAMMED_STEPS_ASSET,
        "flygym_vision_config": FLYGym_VISION_CONFIG,
        "flygym_rigging_config": FLYGym_RIGGING_CONFIG,
        "source_freeze": V4_SOURCE_FREEZE,
        "freeze_manifest": V4_FREEZE_MANIFEST,
        "controller_protocol": V4_PROTOCOL,
        "controller_config": V4_CONTROLLER_CONFIG,
        "controller_compiled_model": V4_COMPILED_MODEL,
        "controller_initial_state": V4_INITIAL_STATE,
        "controller_reference_manifest": V4_REFERENCE_MANIFEST,
        "controller_reference_integrity": V4_REFERENCE_INTEGRITY,
        "controller_effective_config": WEAK_REFERENCE_ROOT / "effective_config.json",
        "acceptance_protocol": PROJECT_ROOT / "config/flygym_cartesian_repair_protocol.json",
        "angular_stimulus_helper": PROJECT_ROOT / "scripts/standard_optomotor_stimulus.py",
        "retinal_geometry": RETINAL_REFERENCE,
        "t4t5_direction_asset": TARGET_DIRECTIONS_ASSET,
        "parity_candidate_adapter": PROJECT_ROOT / "scripts/flyvis_runtime_adapter_eye_parity_v2.py",
        "parity_candidate_worker": PROJECT_ROOT / "scripts/flygym_online_neural_worker_eye_parity_v2.py",
        "parity_static_audit": PARITY_STATIC_AUDIT,
        "parity_static_integrity": PARITY_INTEGRITY,
        "prior_model_mismatch_diagnosis": (
            ONLINE_BUNDLE_ROOT.parent / "flygym-online-eye-parity-v4-1-20261006-v1"
            / "diagnosis/static-model-mismatch/diagnosis.json"
        ),
        "prior_xml_recompile_comparison": (
            ONLINE_BUNDLE_ROOT.parent / "flygym-online-eye-parity-v4-1-20261006-v1"
            / "diagnosis/static-model-mismatch/xml-recompile-comparison.json"
        ),
        "attempt3_runtime_model_preflight": (
            ATTEMPT3_BUNDLE_ROOT / "diagnosis/static-runtime-preflight/runtime-model-preflight.json"
        ),
        "attempt3_runtime_model_arrays": (
            ATTEMPT3_BUNDLE_ROOT / "diagnosis/static-runtime-preflight/runtime-model-preflight.npz"
        ),
        "attempt3_runtime_compiled_xml": (
            ATTEMPT3_BUNDLE_ROOT / "diagnosis/static-runtime-preflight/runtime-current-compiled-model.xml"
        ),
        "attempt3_v3_ready_provenance": ATTEMPT3_STATIC_READY_PROVENANCE,
        "attempt3_v3_worker_ready_failure_record": ATTEMPT3_STATIC_FAILURE_RECORD,
        "attempt3_v3_ready_stimulus_copy": ATTEMPT3_STATIC_CAPTURED_STIMULUS,
        "successful_fixed_head_ready_protocol": SUCCESSFUL_TERMINAL_READY_PROTOCOL,
        # The ready record reports these worker imports and effective maps.
        # Bind them at prepare time instead of relying on the worker-script hash.
        "flyvis_legacy_adapter": PROJECT_ROOT / "scripts/flyvis_runtime_adapter.py",
        "malecns_engine": PROJECT_ROOT / "scripts/malecns_engine.py",
        "malecns_gate": PROJECT_ROOT / "scripts/malecns_gate.py",
        "neuromuscular_adapter": PROJECT_ROOT / "scripts/neuromuscular_adapter.py",
        "neuromuscular_map": PROJECT_ROOT / "config/neuromuscular_map.json",
        "sensory_map": PROJECT_ROOT / "config/sensory_map.json",
        "twin_contracts": PROJECT_ROOT / "scripts/twin_contracts.py",
        "population_selectors": PROJECT_ROOT / "scripts/population_selectors.py",
        "virtual_world": PROJECT_ROOT / "scripts/virtual_world.py",
        "motor_channel_map": PROJECT_ROOT / "config/leg_muscle_map.json",
        "motor_channel_capture": PROJECT_ROOT / "scripts/motor_channel_capture.py",
    }
    for index, path in enumerate(sorted(csr_root.iterdir())):
        if path.is_file():
            paths[f"malecns_csr_{index:02d}_{path.name}"] = path
    scene_document = json.loads(
        (PROJECT_ROOT / "config/optomotor_panel_scene.json").read_text(encoding="utf-8")
    )
    scene_root = Path(str(scene_document.get("asset_root", ""))).expanduser()
    if not scene_root.is_absolute():
        scene_root = (PROJECT_ROOT / "config" / scene_root).resolve()
    for index, asset in enumerate(scene_document.get("assets", [])):
        if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
            continue
        paths[f"scene_asset_{index:02d}_{Path(asset['path']).name}"] = (
            scene_root / str(asset["path"])
        ).resolve()
    paths.update({f"stimulus_spec_{name}": path for name, path in CASE_SPEC_PATHS.items()})
    return paths


def _verify_live_source_inputs(freeze: dict[str, Any]) -> None:
    records = freeze.get("source_inputs", {})
    for role, record in records.items():
        path = Path(str(record.get("path", "")))
        if (not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1))
                or _sha256_file(path) != record.get("sha256")):
            raise RuntimeError(f"source changed after prepare freeze ({role}): {path}")
        snapshot = record.get("snapshot")
        if snapshot:
            snapshot_path = Path(str(snapshot))
            if (not snapshot_path.is_file()
                    or _sha256_file(snapshot_path) != record.get("snapshot_sha256")
                    or record.get("snapshot_sha256") != record.get("sha256")):
                raise RuntimeError(f"source snapshot changed or no longer matches ({role}): {snapshot_path}")


def _array_sha_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "elements": int(array.size),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _mesh_asset_manifest(xml_bytes: bytes) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    assets: dict[str, dict[str, Any]] = {}
    for mesh in root.iter("mesh"):
        name, file_name = mesh.attrib.get("name"), mesh.attrib.get("file")
        if not name or not file_name:
            continue
        path = Path(file_name)
        if not path.is_file():
            assets[name] = {"path": str(path), "exists": False}
            continue
        assets[name] = {
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return assets


def _compare_xml_except_mesh_stl_content_type(
    reference_bytes: bytes, current_bytes: bytes
) -> list[dict[str, Any]]:
    """Allow only missing/model-stl metadata on STL mesh tags; keep every other XML value exact."""
    reference = ET.fromstring(reference_bytes)
    current = ET.fromstring(current_bytes)
    changes: list[dict[str, Any]] = []

    def compare(left: ET.Element, right: ET.Element, location: str) -> None:
        if left.tag != right.tag or left.text != right.text or left.tail != right.tail:
            raise RuntimeError(f"compiled XML structure/text differs at {location}")
        left_attrs = dict(left.attrib)
        right_attrs = dict(right.attrib)
        if left.tag == "mesh":
            left_kind = left_attrs.pop("content_type", None)
            right_kind = right_attrs.pop("content_type", None)
            if left_kind not in ALLOWED_MESH_CONTENT_TYPES or right_kind not in ALLOWED_MESH_CONTENT_TYPES:
                raise RuntimeError(f"unsupported mesh content_type at {location}: {left_kind!r}->{right_kind!r}")
            if left_kind != right_kind:
                file_name = left_attrs.get("file")
                if not isinstance(file_name, str) or Path(file_name).suffix.lower() != ".stl":
                    raise RuntimeError(f"content_type exception applies only to STL-backed meshes: {location}")
                changes.append({
                    "mesh_name": left_attrs.get("name"),
                    "file": file_name,
                    "frozen_content_type": left_kind,
                    "runtime_content_type": right_kind,
                })
        if left_attrs != right_attrs:
            raise RuntimeError(f"compiled XML attributes other than mesh content_type differ at {location}")
        left_children, right_children = list(left), list(right)
        if len(left_children) != len(right_children):
            raise RuntimeError(f"compiled XML child count differs at {location}")
        for index, (left_child, right_child) in enumerate(zip(left_children, right_children)):
            compare(left_child, right_child, f"{location}/{left_child.tag}[{index}]")

    compare(reference, current, "/mujoco[0]")
    return changes


def _model_options_for_preflight(model: Any) -> dict[str, Any]:
    opt = model.opt
    return {
        "native_timestep_s": float(opt.timestep),
        "gravity_mm_s2": np.asarray(opt.gravity, dtype=float).tolist(),
        "integrator": int(opt.integrator),
        "solver": int(opt.solver),
        "iterations": int(opt.iterations),
        "noslip_iterations": int(opt.noslip_iterations),
        "cone": int(opt.cone),
        "jacobian": int(opt.jacobian),
        **{key: int(getattr(model, key)) for key in (
            "nq", "nv", "nu", "nbody", "ngeom", "njnt", "nsensor", "npair"
        )},
    }


def _named_model_objects(model: Any, mj: Any) -> dict[str, list[str]]:
    groups = (
        ("body", mj.mjtObj.mjOBJ_BODY, "nbody"),
        ("geom", mj.mjtObj.mjOBJ_GEOM, "ngeom"),
        ("joint", mj.mjtObj.mjOBJ_JOINT, "njnt"),
        ("actuator", mj.mjtObj.mjOBJ_ACTUATOR, "nu"),
    )
    return {
        name: [mj.mj_id2name(model, obj_type, i) or f"#{i}"
              for i in range(int(getattr(model, count_name)))]
        for name, obj_type, count_name in groups
    }


def _compile_xml_model_audit(session: Any, output_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    current_xml = session.compiled_mjcf_xml.encode("utf-8")
    frozen_xml = V4_COMPILED_MODEL.read_bytes()
    current_xml_path = output_dir / "runtime-current-compiled-model.xml"
    current_xml_path.write_bytes(current_xml)
    current_sha = hashlib.sha256(current_xml).hexdigest()
    known_sha_variant = current_sha in {
        FROZEN_COMPILED_MODEL_SHA256,
        FROZEN_COMPILED_MODEL_KNOWN_CONTENT_TYPE_VARIANT_SHA256,
    }
    content_type_changes = _compare_xml_except_mesh_stl_content_type(frozen_xml, current_xml)
    current_assets = _mesh_asset_manifest(current_xml)
    frozen_assets = _mesh_asset_manifest(frozen_xml)
    if not current_assets or current_assets != frozen_assets or any(
        not record.get("exists") for record in current_assets.values()
    ):
        raise RuntimeError("compiled XML mesh assets differ from the frozen STL asset manifest")

    # Compare same-stage XML recompile models. These checks deliberately do not
    # compare the in-memory MjSpec runtime model with XML-roundtrip arrays.
    mj = session.mj
    current_model = mj.MjModel.from_xml_path(str(current_xml_path))
    frozen_model = mj.MjModel.from_xml_path(str(V4_COMPILED_MODEL))
    names_current = _named_model_objects(current_model, mj)
    names_frozen = _named_model_objects(frozen_model, mj)
    fields: dict[str, Any] = {}
    recompiled_arrays: dict[str, np.ndarray] = {}
    all_fields_exact = True
    for field_name in COMPILED_MODEL_ARRAY_FIELDS:
        if not hasattr(current_model, field_name) or not hasattr(frozen_model, field_name):
            fields[field_name] = {"available": False}
            all_fields_exact = False
            continue
        a = np.asarray(getattr(current_model, field_name)).copy()
        b = np.asarray(getattr(frozen_model, field_name))
        same_shape = a.shape == b.shape
        exact = bool(same_shape and np.array_equal(a, b))
        max_abs = (
            float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            if same_shape and a.size else None
        )
        fields[field_name] = {
            "available": True,
            "shape_current": list(a.shape),
            "shape_frozen": list(b.shape),
            "shape_equal": same_shape,
            "exact_equal": exact,
            "max_abs_difference": max_abs,
            "current_sha256": _array_sha_record(a)["sha256"],
            "frozen_sha256": _array_sha_record(b)["sha256"],
        }
        all_fields_exact &= exact
        recompiled_arrays[f"xml_recompiled_{field_name}"] = a

    current_options = _model_options_for_preflight(current_model)
    frozen_options = _model_options_for_preflight(frozen_model)
    audit = {
        "stage": "current_and_frozen_xml_both_compiled_by_MjModel.from_xml_path",
        "raw_xml_current_sha256": current_sha,
        "raw_xml_frozen_sha256": hashlib.sha256(frozen_xml).hexdigest(),
        "raw_xml_sha256_matches_frozen": current_sha == FROZEN_COMPILED_MODEL_SHA256,
        "raw_xml_sha256_is_known_serialization_variant": known_sha_variant,
        "raw_xml_sha256_is_provenance_only_not_a_gate": True,
        "content_type_changes_only": content_type_changes,
        "xml_parameter_tree_exact_except_explicit_mesh_stl_content_type": True,
        "mesh_asset_manifest_exact": True,
        "mesh_asset_count": len(current_assets),
        "current_mesh_assets": current_assets,
        "frozen_mesh_assets": frozen_assets,
        "compiled_options_exact": current_options == frozen_options,
        "current_compiled_options": current_options,
        "frozen_compiled_options": frozen_options,
        "compiled_object_names_and_order_exact": names_current == names_frozen,
        "current_object_names": names_current,
        "frozen_object_names": names_frozen,
        "array_field_count": len(fields),
        "all_array_fields_available": all(field.get("available") for field in fields.values()),
        "all_available_compiled_array_fields_exact": all_fields_exact,
        "compiled_array_fields": fields,
        "recompiled_arrays_path": str(output_dir / "runtime-model-preflight.npz"),
    }
    audit["same_stage_xml_compile_gate_pass"] = bool(
        audit["xml_parameter_tree_exact_except_explicit_mesh_stl_content_type"]
        and audit["mesh_asset_manifest_exact"]
        and audit["compiled_options_exact"]
        and audit["compiled_object_names_and_order_exact"]
        and audit["all_array_fields_available"]
        and audit["all_available_compiled_array_fields_exact"]
    )
    return audit, recompiled_arrays


def _runtime_initial_state_audit(session: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    baseline_order = json.loads((V4_COMPILED_MODEL.parent / "array_order.json").read_text(encoding="utf-8"))
    baseline_manifest = json.loads(V4_REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    baseline_state = json.loads(V4_INITIAL_STATE.read_text(encoding="utf-8"))
    current_order = session._array_order_names()
    current_options = session._model_options()
    qpos = np.asarray(session.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(session.data.qvel, dtype=np.float64).copy()
    expected_qpos = np.asarray(baseline_state["qpos"], dtype=np.float64)
    expected_qvel = np.asarray(baseline_state["qvel"], dtype=np.float64)
    options_expected = baseline_manifest["effective_model_options"]
    order_exact = current_order == baseline_order
    options_exact = current_options == options_expected
    qpos_exact = bool(np.array_equal(qpos, expected_qpos))
    qvel_exact = bool(np.array_equal(qvel, expected_qvel))

    model_arrays: dict[str, np.ndarray] = {}
    for name in COMPILED_MODEL_ARRAY_FIELDS:
        if hasattr(session.model, name):
            model_arrays[f"runtime_model_{name}"] = np.asarray(getattr(session.model, name)).copy()

    data = session.data
    body_ids = np.asarray(session.body_ids, dtype=np.int32)
    body_velocity = np.zeros((len(body_ids), 6), dtype=np.float64)
    for index, body_id in enumerate(body_ids):
        session.mj.mj_objectVelocity(
            session.model, data, session.mj.mjtObj.mjOBJ_BODY,
            int(body_id), body_velocity[index], 0,
        )
    body_pos = np.asarray(data.xpos[body_ids], dtype=np.float32)
    body_quat = np.asarray(data.xquat[body_ids], dtype=np.float32)
    body_ipos = np.asarray(data.xipos[body_ids], dtype=np.float32)
    body_linvel = np.asarray(body_velocity[:, 3:], dtype=np.float32)
    body_angvel = np.asarray(body_velocity[:, :3], dtype=np.float32)
    masses = np.asarray(session.body_mass, dtype=np.float64)
    com_pos = np.average(data.xipos[body_ids], axis=0, weights=masses)
    com_vel = np.average(body_linvel, axis=0, weights=masses)
    thorax_heading = np.asarray(data.xmat[session.thorax_body_id]).reshape(3, 3)[:, 0]
    foot_pos = np.asarray(data.xpos[np.asarray(session.foot_body_ids, dtype=np.int32)], dtype=np.float32)
    contact_values = session.sim.get_ground_contact_info(session.fly.name)
    found, force_local, torque_local, contact_pos, contact_normal, contact_tangent = contact_values
    foot_found = np.asarray(found > 0, dtype=np.uint8)
    per_leg_fz = np.zeros(6, dtype=np.float64)
    nonleg_ground = False
    thorax_ground = False
    contact_wrenches = np.zeros((int(data.ncon), 6), dtype=np.float64)
    contact_geom1 = np.zeros(int(data.ncon), dtype=np.int32)
    contact_geom2 = np.zeros(int(data.ncon), dtype=np.int32)
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        contact_geom1[contact_id], contact_geom2[contact_id] = g1, g2
        session.mj.mj_contactForce(session.model, data, contact_id, contact_wrenches[contact_id])
        frame = contact.frame.reshape(3, 3)
        force_world_on_geom2 = frame.T @ contact_wrenches[contact_id, :3]
        fly_geom = g1 if g1 in session.geom_id_to_segment else (g2 if g2 in session.geom_id_to_segment else -1)
        is_ground = (g1 == session.ground_geom_id) or (g2 == session.ground_geom_id)
        if is_ground and fly_geom >= 0:
            leg_index = session.geom_id_to_leg.get(fly_geom, -1)
            force_on_fly = -force_world_on_geom2 if fly_geom == g1 else force_world_on_geom2
            if leg_index >= 0:
                per_leg_fz[leg_index] += force_on_fly[2]
            else:
                nonleg_ground = True
                if int(session.model.geom_bodyid[fly_geom]) == session.thorax_body_id:
                    thorax_ground = True

    runtime_values: dict[str, np.ndarray] = {
        "initial_qpos": qpos,
        "initial_qvel": qvel,
        "body_xpos_mm": body_pos,
        "body_xquat_wxyz": body_quat,
        "body_xipos_mm": body_ipos,
        "body_xvel_mm_s": body_linvel,
        "body_xangvel_rad_s": body_angvel,
        "body_xfrc_applied_native": np.asarray(data.xfrc_applied[body_ids], dtype=np.float64),
        "qfrc_applied_native": np.asarray(data.qfrc_applied, dtype=np.float64).copy(),
        "qfrc_bias_native": np.asarray(data.qfrc_bias, dtype=np.float64).copy(),
        "qfrc_passive_native": np.asarray(data.qfrc_passive, dtype=np.float64).copy(),
        "qfrc_constraint_native": np.asarray(data.qfrc_constraint, dtype=np.float64).copy(),
        "qfrc_actuator_native": np.asarray(data.qfrc_actuator, dtype=np.float64).copy(),
        "qacc_native": np.asarray(data.qacc, dtype=np.float64).copy(),
        "actuator_force_native": np.asarray(data.actuator_force, dtype=np.float64).copy(),
        "actuator_ctrl_native": np.asarray(data.ctrl, dtype=np.float64).copy(),
        "com_position_mm": np.asarray(com_pos, dtype=np.float64),
        "com_velocity_mm_s": np.asarray(com_vel, dtype=np.float64),
        "thorax_heading_xy": np.asarray(thorax_heading[:2], dtype=np.float32),
        "thorax_heading_yaw_rad": np.asarray(
            math.atan2(float(thorax_heading[1]), float(thorax_heading[0])), dtype=np.float64
        ),
        "tarsus5_xpos_mm": foot_pos,
        "foot_contact_found": foot_found,
        "foot_contact_force_local_native": np.asarray(force_local, dtype=np.float32),
        "foot_contact_torque_local_native": np.asarray(torque_local, dtype=np.float32),
        "foot_contact_pos_mm": np.asarray(contact_pos, dtype=np.float32),
        "foot_contact_normal_world": np.asarray(contact_normal, dtype=np.float32),
        "foot_contact_tangent_world": np.asarray(contact_tangent, dtype=np.float32),
        "foot_contact_fz_up_native": np.asarray(per_leg_fz, dtype=np.float32),
        "root_external_wrench_native": np.asarray(data.xfrc_applied[session.thorax_body_id], dtype=np.float64),
        "nonleg_ground_contact": np.asarray(int(nonleg_ground), dtype=np.uint8),
        "thorax_ground_contact": np.asarray(int(thorax_ground), dtype=np.uint8),
        "body_mass_native": masses.copy(),
        "position_actuator_force_native": np.asarray(
            data.actuator_force[session.position_actuator_ids], dtype=np.float64
        ).copy(),
        "adhesion_actuator_force_native": np.asarray(
            data.actuator_force[session.adhesion_actuator_ids], dtype=np.float64
        ).copy(),
        "contact_geom1": contact_geom1,
        "contact_geom2": contact_geom2,
        "contact_wrenches_local_native": contact_wrenches,
        "time_s": np.asarray(float(data.time), dtype=np.float64),
    }
    compare_to_trace = {
        "initial_qpos": "qpos",
        "initial_qvel": "qvel",
        "time_s": "time_s",
        "body_xpos_mm": "body_xpos_mm",
        "body_xquat_wxyz": "body_xquat_wxyz",
        "body_xipos_mm": "body_xipos_mm",
        "body_xvel_mm_s": "body_xvel_mm_s",
        "body_xangvel_rad_s": "body_xangvel_rad_s",
        "body_xfrc_applied_native": "body_xfrc_applied_native",
        "body_mass_native": "body_mass_native",
        "qfrc_applied_native": "qfrc_applied_native",
        "position_actuator_force_native": "position_actuator_force_native",
        "adhesion_actuator_force_native": "adhesion_actuator_force_native",
        "com_position_mm": "com_position_mm",
        "com_velocity_mm_s": "com_velocity_mm_s",
        "thorax_heading_xy": "thorax_heading_xy",
        "thorax_heading_yaw_rad": "thorax_heading_yaw_rad",
        "tarsus5_xpos_mm": "tarsus5_xpos_mm",
        "foot_contact_found": "foot_contact_found",
        "foot_contact_force_local_native": "foot_contact_force_local_native",
        "foot_contact_torque_local_native": "foot_contact_torque_local_native",
        "foot_contact_pos_mm": "foot_contact_pos_mm",
        "foot_contact_normal_world": "foot_contact_normal_world",
        "foot_contact_tangent_world": "foot_contact_tangent_world",
        "foot_contact_fz_up_native": "foot_contact_fz_up_native",
        "root_external_wrench_native": "root_external_wrench_native",
        "nonleg_ground_contact": "nonleg_ground_contact",
        "thorax_ground_contact": "thorax_ground_contact",
    }
    trace_path = V4_COMPILED_MODEL.parent / "trajectory.npz"
    trace_comparisons: dict[str, Any] = {}
    with np.load(trace_path, allow_pickle=False) as frozen_trace:
        if float(frozen_trace["time_s"][0]) != 0.0:
            raise RuntimeError("frozen V4.1 trajectory does not begin at t=0")
        for current_name, frozen_name in compare_to_trace.items():
            current_value = np.asarray(runtime_values[current_name])
            frozen_array = np.asarray(frozen_trace[frozen_name])
            static_body_mass = frozen_name == "body_mass_native"
            frozen_value = frozen_array if static_body_mass else np.asarray(frozen_array[0])
            same_shape = current_value.shape == frozen_value.shape
            exact = bool(same_shape and np.array_equal(current_value, frozen_value))
            max_abs = (
                float(np.max(np.abs(current_value.astype(np.float64) - frozen_value.astype(np.float64))))
                if same_shape and current_value.size else None
            )
            trace_comparisons[current_name] = {
                "frozen_trace_field": frozen_name,
                "frozen_trace_sample_index": None if static_body_mass else 0,
                "shape_current": list(current_value.shape),
                "shape_frozen": list(frozen_value.shape),
                "shape_equal": same_shape,
                "exact_equal": exact,
                "max_abs_difference": max_abs,
            }
    trace_exact = all(item["exact_equal"] for item in trace_comparisons.values())
    runtime_values.update(model_arrays)
    runtime_report = {
        "runtime_stage": "in_memory FlyGym world.compile MjSpec model after constructor mj_forward, before worker/control ticks",
        "runtime_vs_frozen_runtime_array_exact_claim": False,
        "frozen_runtime_model_arrays_available": False,
        "runtime_vs_xml_recompile_comparison_is_cross_stage_only": True,
        "session_time_s": float(data.time),
        "controller_ticks": int(session._control_samples),
        "native_samples": int(session._native_samples),
        "runtime_named_order_exact_to_frozen_array_order": order_exact,
        "runtime_model_options_exact_to_frozen_manifest": options_exact,
        "runtime_initial_qpos_exact_to_frozen_initial_state": qpos_exact,
        "runtime_initial_qvel_exact_to_frozen_initial_state": qvel_exact,
        "runtime_initial_body_mass_exact_to_frozen_manifest": (
            current_order.get("body_mass_native_by_body")
            == baseline_order.get("body_mass_native_by_body")
        ),
        "runtime_initial_data_exact_to_frozen_trace_sample0": trace_exact,
        "runtime_initial_trace_comparisons": trace_comparisons,
        "runtime_force_and_kinematic_capture_sha256": {
            name: _array_sha_record(value) for name, value in runtime_values.items()
        },
        "captured_runtime_array_path": "runtime-model-preflight.npz",
        "full_current_runtime_array_fields": len(model_arrays),
        "full_runtime_data_fields": len(runtime_values) - len(model_arrays),
        "session_array_order": current_order,
        "frozen_array_order_file_sha256": _sha256_file(V4_COMPILED_MODEL.parent / "array_order.json"),
        "runtime_model_options": current_options,
        "frozen_model_options": options_expected,
        "initial_state": {
            "time_s": float(data.time),
            "qpos_sha256": _array_sha_record(qpos)["sha256"],
            "qvel_sha256": _array_sha_record(qvel)["sha256"],
        },
    }
    runtime_gate = (
        order_exact and options_exact and qpos_exact and qvel_exact
        and runtime_report["runtime_initial_body_mass_exact_to_frozen_manifest"]
        and trace_exact
    )
    runtime_report["runtime_available_reference_gate_pass"] = bool(runtime_gate)
    return runtime_report, runtime_values


def _verify_runtime_model_contract(session: Any, output_dir: Path) -> dict[str, Any]:
    xml_audit, xml_arrays = _compile_xml_model_audit(session, output_dir)
    runtime_audit, runtime_arrays = _runtime_initial_state_audit(session)
    cross_stage: dict[str, Any] = {}
    for field_name in COMPILED_MODEL_ARRAY_FIELDS:
        runtime_key = f"runtime_model_{field_name}"
        xml_key = f"xml_recompiled_{field_name}"
        if runtime_key not in runtime_arrays or xml_key not in xml_arrays:
            continue
        runtime_value = np.asarray(runtime_arrays[runtime_key])
        xml_value = np.asarray(xml_arrays[xml_key])
        same_shape = runtime_value.shape == xml_value.shape
        cross_stage[field_name] = {
            "comparison_stage": "in-memory FlyGym MjSpec model vs XML round-trip model; diagnostic only",
            "shape_equal": same_shape,
            "exact_equal": bool(same_shape and np.array_equal(runtime_value, xml_value)),
            "max_abs_difference": (
                float(np.max(np.abs(runtime_value.astype(np.float64) - xml_value.astype(np.float64))))
                if same_shape and runtime_value.size else None
            ),
        }
    runtime_audit["runtime_vs_xml_roundtrip_cross_stage_diagnostic"] = cross_stage
    all_arrays = dict(xml_arrays)
    all_arrays.update(runtime_arrays)
    arrays_path = output_dir / "runtime-model-preflight.npz"
    np.savez_compressed(arrays_path, **all_arrays)
    record = {
        "schema": "flygym-online-v4-1-runtime-model-preflight-1",
        "xml_recompile_comparison": xml_audit,
        "in_memory_runtime_reference_comparison": runtime_audit,
        "array_capture_path": str(arrays_path),
        "array_capture_sha256": _sha256_file(arrays_path),
        "same_stage_xml_recompile_exact": xml_audit["same_stage_xml_compile_gate_pass"],
        "runtime_vs_frozen_saved_state_gate_pass": runtime_audit["runtime_available_reference_gate_pass"],
    }
    _write_json(output_dir / "runtime-model-preflight.json", record)
    if not record["same_stage_xml_recompile_exact"]:
        raise RuntimeError("current and frozen XML recompile preflight did not pass")
    if not record["runtime_vs_frozen_saved_state_gate_pass"]:
        raise RuntimeError("current runtime options/order/initial state/trace sample differ from frozen V4.1 evidence")
    return record


def _verify_worker_ready_assets_locked(ready: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    """Bind every reported ready source asset to a prepared source-input hash."""
    source_records = freeze.get("source_inputs", {})
    by_resolved_path: dict[Path, tuple[str, dict[str, Any]]] = {}
    for role, record in source_records.items():
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        by_resolved_path[path] = (str(role), record)

    checked: dict[str, dict[str, Any]] = {}
    visited: set[Path] = set()
    artifact_dir_raw = ready.get("artifact_dir")
    artifact_dir = (
        Path(str(artifact_dir_raw)).expanduser().resolve()
        if isinstance(artifact_dir_raw, str) and artifact_dir_raw else None
    )
    angular_source = (
        ready.get("source_assets", {}).get("angular_stimulus_spec", {})
        if isinstance(ready.get("source_assets"), dict) else {}
    )
    angular_spec_source = angular_source.get("source", {}) if isinstance(angular_source, dict) else {}

    def visit(value: Any, label: str) -> None:
        if isinstance(value, dict):
            if label == "source_assets.malecns.csr_preflight":
                preflight_path = Path(str(value.get("path", ""))).expanduser().resolve()
                csr_assets = ready.get("source_assets", {}).get("malecns", {}).get("csr_files", {})
                frozen_csr_names = {
                    Path(str(item.get("path", ""))).name
                    for item in csr_assets.values()
                    if isinstance(item, dict) and item.get("path")
                }
                preflight_files = value.get("files", {})
                preflight_names = {
                    str(name) for name, exists in preflight_files.items() if exists is True
                } if isinstance(preflight_files, dict) else set()
                if (value.get("exists") is not True or not preflight_path.is_dir()
                        or ready.get("source_assets", {}).get("malecns", {}).get("csr_path")
                        != str(preflight_path)
                        or value.get("valid") is not True or value.get("blockers") != []
                        or preflight_names != frozen_csr_names
                        or any(not (preflight_path / name).is_file() for name in preflight_names)):
                    raise RuntimeError("worker CSR directory preflight does not match its frozen per-file assets")
                checked[label] = {
                    "path": str(preflight_path),
                    "path_class": "directory_preflight_bound_to_individually_frozen_csr_files",
                    "valid": True,
                    "blocker_count": 0,
                    "file_names": sorted(preflight_names),
                }
                return
            if "path" in value and "exists" in value:
                path = Path(str(value["path"])).expanduser().resolve()
                if label == "source_assets.angular_stimulus_spec.captured_copy":
                    if (artifact_dir is None or path != artifact_dir / "angular_stimulus_spec.json"
                            or value.get("exists") is not True or not path.is_file()):
                        raise RuntimeError("worker-captured angular spec is not the expected per-run output copy")
                    source_path = Path(str(angular_spec_source.get("path", ""))).expanduser().resolve()
                    source_frozen = by_resolved_path.get(source_path)
                    if (source_frozen is None or not source_path.is_file()
                            or angular_spec_source.get("exists") is not True):
                        raise RuntimeError("worker angular spec source is not a frozen registered stimulus input")
                    source_role, source_record = source_frozen
                    source_sha = _sha256_file(source_path)
                    source_size = source_path.stat().st_size
                    if (source_sha != source_record.get("sha256")
                            or source_size != int(source_record.get("size_bytes", -1))
                            or value.get("sha256") != source_sha
                            or int(value.get("size_bytes", -1)) != source_size
                            or _sha256_file(path) != source_sha
                            or angular_source.get("sha256") != source_sha):
                        raise RuntimeError("worker-captured angular spec differs from the frozen registered stimulus")
                    visited.add(path)
                    checked[label] = {
                        "source_input_role": source_role,
                        "path": str(path),
                        "size_bytes": source_size,
                        "sha256": source_sha,
                        "ready_record_sha256_present": True,
                        "path_class": "per_run_output_copy_bound_to_registered_stimulus_hash",
                    }
                    return
                if path in visited:
                    return
                visited.add(path)
                if value.get("exists") is not True or not path.is_file():
                    raise RuntimeError(f"worker ready reports a missing required source asset: {label}: {path}")
                frozen = by_resolved_path.get(path)
                if frozen is None:
                    raise RuntimeError(f"worker ready source asset was not frozen during prepare: {label}: {path}")
                role, record = frozen
                size = path.stat().st_size
                if size != int(record.get("size_bytes", -1)):
                    raise RuntimeError(f"worker ready asset size changed after prepare ({role}): {path}")
                actual_sha = _sha256_file(path)
                if actual_sha != record.get("sha256"):
                    raise RuntimeError(f"worker ready asset hash changed after prepare ({role}): {path}")
                ready_sha = value.get("sha256")
                if ready_sha is not None and ready_sha != actual_sha:
                    raise RuntimeError(f"worker ready asset hash differs from live file ({role}): {path}")
                ready_size = value.get("size_bytes")
                if ready_size is not None and int(ready_size) != size:
                    raise RuntimeError(f"worker ready asset size differs from live file ({role}): {path}")
                checked[label] = {
                    "source_input_role": role,
                    "path": str(path),
                    "size_bytes": size,
                    "sha256": actual_sha,
                    "ready_record_sha256_present": ready_sha is not None,
                }
                return
            for key, nested in value.items():
                visit(nested, f"{label}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{label}[{index}]")

    visit(ready.get("source_assets"), "source_assets")
    # Import-only dependencies such as the decoder and engine are not emitted
    # by V3's source_assets object, so revalidate every prepared input at ready.
    _verify_live_source_inputs(freeze)
    return {
        "all_reported_file_assets_match_prepare_freeze": True,
        "ready_file_asset_count": len(checked),
        "checked_assets": checked,
        "source_input_count_revalidated_at_ready": len(source_records),
    }


def _attempt3_ready_fixture_from_saved_v3_provenance() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load attempt3's parsed worker-ready object and pair it with its provenance."""
    provenance = json.loads(ATTEMPT3_STATIC_READY_PROVENANCE.read_text(encoding="utf-8"))
    failure = json.loads(ATTEMPT3_STATIC_FAILURE_RECORD.read_text(encoding="utf-8"))
    ready = failure.get("worker_ready")
    if not isinstance(ready, dict) or ready.get("type") != "ready" or ready.get("protocol") != PROTOCOL:
        raise RuntimeError("saved attempt3 failure record has no parsed worker v3 ready object")
    if (provenance.get("source_code", {}).get("worker_script", {}).get("sha256") != _sha256_file(WORKER_SCRIPT)
            or ready.get("source_assets", {}).get("worker_source", {}).get("candidate", {}).get("sha256")
            != _sha256_file(WORKER_SCRIPT)):
        raise RuntimeError("saved attempt3 startup provenance does not identify the current frozen V3 worker")
    case_id = "static-c1-seed1729"
    spec_path = CASE_SPEC_PATHS[case_id]
    spec_sha256 = _sha256_file(spec_path)
    angular = ready.get("angular_stimulus", {})
    angular_metadata = angular.get("metadata", angular) if isinstance(angular, dict) else {}
    if (ready.get("stimulus_condition") != "angular"
            or ready.get("angular_stimulus_spec_sha256") != spec_sha256
            or angular_metadata.get("head_direction_sign") != 1
            or angular_metadata.get("angular_speed_units") != "degrees per second"
            or ready.get("motor_channel_capture") != provenance.get("motor_channel_capture")):
        raise RuntimeError("saved attempt3 V3 ready object disagrees with its static provenance/spec")
    return ready, provenance, failure


def _write_prepare_validation_fixtures(freeze: dict[str, Any]) -> dict[str, Any]:
    ready_fixture, provenance, failure = _attempt3_ready_fixture_from_saved_v3_provenance()
    spec_path = CASE_SPEC_PATHS["static-c1-seed1729"]
    args = argparse.Namespace(angular_stimulus_spec_sha256=_sha256_file(spec_path))
    ready_audit = _validate_eye_parity_ready(ready_fixture, freeze, args)
    motor_output_dir = Path(ready_fixture["artifact_dir"]).resolve()
    disabled_was_rejected = False
    disabled_reason = None
    try:
        _validate_motor_channel_capture_metadata(
            {"enabled": False}, output_dir=motor_output_dir,
            expected_frames=2000, stage="ready",
        )
    except RuntimeError as exc:
        disabled_was_rejected = True
        disabled_reason = str(exc)
    if not disabled_was_rejected:
        raise RuntimeError("offline validation accepted an explicitly disabled motor capture")

    wire_ready = None
    with SUCCESSFUL_TERMINAL_READY_PROTOCOL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "ready":
                wire_ready = record
                break
    if not isinstance(wire_ready, dict):
        raise RuntimeError("successful fixed-head protocol log has no actual worker ready record")
    wire_motor_ready_audit = _validate_motor_channel_capture_metadata(
        wire_ready.get("motor_channel_capture"),
        output_dir=Path(str(wire_ready["artifact_dir"])).expanduser().resolve(),
        expected_frames=2000, stage="ready",
    )
    terminal_audit = _verify_worker_capture_artifacts(
        SUCCESSFUL_TERMINAL_CAPTURE_FIXTURE, expected_frames=2000
    )
    captured_ready_source = {
        "path": str(ATTEMPT3_STATIC_READY_PROVENANCE),
        "size_bytes": ATTEMPT3_STATIC_READY_PROVENANCE.stat().st_size,
        "sha256": _sha256_file(ATTEMPT3_STATIC_READY_PROVENANCE),
    }
    captured_failure_source = {
        "path": str(ATTEMPT3_STATIC_FAILURE_RECORD),
        "size_bytes": ATTEMPT3_STATIC_FAILURE_RECORD.stat().st_size,
        "sha256": _sha256_file(ATTEMPT3_STATIC_FAILURE_RECORD),
    }
    captured_spec_source = {
        "path": str(ATTEMPT3_STATIC_CAPTURED_STIMULUS),
        "size_bytes": ATTEMPT3_STATIC_CAPTURED_STIMULUS.stat().st_size,
        "sha256": _sha256_file(ATTEMPT3_STATIC_CAPTURED_STIMULUS),
    }
    wire_protocol_source = {
        "path": str(SUCCESSFUL_TERMINAL_READY_PROTOCOL),
        "size_bytes": SUCCESSFUL_TERMINAL_READY_PROTOCOL.stat().st_size,
        "sha256": _sha256_file(SUCCESSFUL_TERMINAL_READY_PROTOCOL),
    }
    ready_record = {
        "schema_version": "attempt5-ready-validator-fixture-1",
        "evidence_class": "offline validator replay; no neural forward or physics",
        "fixture_origin": {
            "saved_attempt3_v3_startup_provenance": captured_ready_source,
            "saved_attempt3_failure_with_parsed_worker_ready": captured_failure_source,
            "saved_attempt3_worker_stimulus_copy": captured_spec_source,
            "successful_fixed_head_worker_wire_ready": wire_protocol_source,
            "raw_worker_protocol_stdout_was_separately_persisted": False,
            "parsed_worker_ready_is_retained": True,
            "projection_basis": "the exact parsed worker_ready JSON object retained in attempt3 failure.json; no field synthesis",
            "attempt3_worker_sha256": provenance["source_code"]["worker_script"]["sha256"],
            "path_binding_scope": "fixture output paths resolve within the preserved attempt3 worker output directory; live attempt5 paths are checked separately at runtime",
        },
        "ready_payload": ready_fixture,
        "validator_result": ready_audit,
        "disabled_motor_capture_rejected": {
            "passed": disabled_was_rejected,
            "reason": disabled_reason,
        },
        "successful_fixed_head_wire_motor_capture_ready": {
            "protocol": wire_ready.get("protocol"),
            "source": wire_ready.get("source"),
            "worker_candidate_sha256": wire_ready.get("source_assets", {}).get("worker_source", {}).get("candidate", {}).get("sha256"),
            "motor_channel_capture": wire_ready["motor_channel_capture"],
            "validator_result": wire_motor_ready_audit,
        },
    }
    validation_root = PREPARE_ROOT / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    _write_json(validation_root / "attempt3-v3-ready-contract.json", ready_record)
    _write_json(validation_root / "successful-terminal-capture-contract.json", {
        "schema_version": "attempt5-terminal-capture-validator-fixture-1",
        "evidence_class": "offline verification against a completed fixed-head worker capture; no new forward or physics",
        "fixture_root": str(SUCCESSFUL_TERMINAL_CAPTURE_FIXTURE),
        "fixture_provenance_sha256": _sha256_file(SUCCESSFUL_TERMINAL_CAPTURE_FIXTURE / "provenance.json"),
        "fixture_integrity_sha256": _sha256_file(SUCCESSFUL_TERMINAL_CAPTURE_FIXTURE / "integrity.json"),
        "validator_result": terminal_audit,
        "raw_motor_channel_arrays": {
            "file": str(SUCCESSFUL_TERMINAL_CAPTURE_FIXTURE / "muscle_channels.npz"),
            "sha256": terminal_audit["required_files"]["muscle_channels.npz"]["sha256"],
            "membership_file_sha256": terminal_audit["required_files"]["muscle_channels_membership.json"]["sha256"],
        },
    })
    return {
        "attempt3_v3_ready_contract": "pass",
        "explicit_disabled_motor_capture": "rejected",
        "successful_terminal_capture_contract": "pass",
        "terminal_fixture_frame_count": terminal_audit["frame_record_count"],
    }


def _write_prepare_bundle() -> dict[str, Any]:
    if PREPARE_ROOT.exists() and any(PREPARE_ROOT.iterdir()):
        existing = PREPARE_ROOT / "preparation-audit.json"
        if existing.is_file():
            current = json.loads(existing.read_text(encoding="utf-8"))
            freeze_path = PREPARE_ROOT / "source-freeze.json"
            if (freeze_path.is_file()
                    and current.get("source_freeze_sha256") == _sha256_file(freeze_path)
                    and json.loads(freeze_path.read_text(encoding="utf-8")).get(
                        "wrapper_sha256") == _sha256_file(Path(__file__).resolve())):
                freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
                _verify_live_source_inputs(freeze)
                _verify_v4_1_source_contract()
                _verify_eye_parity_geometry()
                _verify_worker_v3_resource_delta()
                _verify_integrity_manifest(PREPARE_ROOT)
                return current
        raise FileExistsError(
            f"prepare bundle already exists but does not bind the current source; "
            f"preserve it and create a new attempt: {PREPARE_ROOT}"
        )
    PREPARE_ROOT.mkdir(parents=True, exist_ok=True)
    source_contract = _verify_v4_1_source_contract()
    parity_guard = _verify_eye_parity_geometry()
    worker_delta = _verify_worker_v3_resource_delta()
    body_fingerprint = _python_package_fingerprint(
        EXPECTED_BODY_ENV, ("flygym", "mujoco", "numpy", "scipy")
    )
    worker_fingerprint = _python_package_fingerprint(
        EXPECTED_WORKER_ENV, ("flyvis", "torch", "triton", "numpy", "scipy")
    )
    model_root = Path("/data/models/flyvis")
    model_view = model_root / "results/flow/0000/000"
    model_files: dict[str, Any] = {}
    for name in ("best_chkpt", "_meta.yaml"):
        path = model_view / name
        if not path.is_file():
            raise FileNotFoundError(f"required FlyVis model source is missing: {path}")
        model_files[name] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    package_manifest_python = json.loads(V4_REFERENCE_MANIFEST.read_text(encoding="utf-8")).get("python")
    if package_manifest_python and not body_fingerprint["version"].startswith(package_manifest_python.split(" ")[0]):
        raise RuntimeError("current FlyGym Python fingerprint differs from the V4.1 reference")
    pilot_config_path = PROJECT_ROOT / "config/optomotor_pilot_experiment.json"
    pilot_config = json.loads(pilot_config_path.read_text(encoding="utf-8"))
    if (int(pilot_config.get("seed", -1)) != 1729
            or not math.isclose(float(pilot_config.get("experiment", {}).get("neural_gain", float("nan"))),
                                EXPECTED_NEURAL_GAIN, rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError("frozen optomotor experiment config no longer specifies seed 1729 / neural gain 1.2")
    worker_config = json.loads(WORKER_CONFIG.read_text(encoding="utf-8"))
    if worker_config.get("world_config") != pilot_config_path.name:
        raise RuntimeError("online worker config no longer resolves the frozen gain/seed experiment config")

    source_paths = _source_input_paths()
    source_records = {}
    snapshot_root = PREPARE_ROOT / "source-snapshots"
    for role, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required frozen source/input is missing ({role}): {path}")
        record = {"path": str(path), "size_bytes": path.stat().st_size,
                  "sha256": _sha256_file(path)}
        # Large model checkpoints are hashed in place rather than copied into the report bundle.
        if path.stat().st_size <= 100 * 1024 * 1024:
            destination = snapshot_root / role / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            record["snapshot"] = str(destination)
            record["snapshot_sha256"] = _sha256_file(destination)
            record["snapshot_matches"] = record["snapshot_sha256"] == record["sha256"]
            if not record["snapshot_matches"]:
                raise RuntimeError(f"source snapshot copy mismatch: {path}")
        source_records[role] = record
    case_rows = []
    for case_id, spec_path in CASE_SPEC_PATHS.items():
        stimulus_object, spec, raw_spec = _load_registered_case(case_id)
        stimulus = spec["stimulus"]
        case_rows.append({
            "case_id": case_id,
            "stimulus_spec_path": str(spec_path),
            "stimulus_spec_sha256": hashlib.sha256(raw_spec).hexdigest(),
            "stimulus": stimulus,
            "derived_signed_speed_deg_s": stimulus_object.angular_speed_deg_s,
            "derived_arena_speed_deg_s": stimulus_object.arena_angular_speed_deg_s,
            "run_path": str(DEFAULT_OUTPUT_ROOT / case_id),
            "total_duration_s": 10.0,
            "frame_count_200_hz": 2000,
            "body_seed": 1729,
            "neural_seed": 1729,
            "cpg_frequency_hz": FROZEN_CPG_FREQUENCY_HZ,
            "legacy_rest_compatibility": True,
        })
    if len(case_rows) != 3:
        raise RuntimeError("prepare protocol must preregister exactly three angular cases")
    _write_json(PREPARE_ROOT / "protocol.json", {
        "schema_version": "flygym-online-eye-parity-v4-1-attempt5-protocol-1",
        "evidence_class": "exploratory integrated online closed loop; not development or biological acceptance",
        "execution_default": "prepare_only; this mode does not create a FlyGym session, import MuJoCo, start FlyVis, or run physics",
        "execute_flag_required": "--execute",
        "cases": case_rows,
        "frozen_body_contract": {
            "controller": "PhaseSpeedGaitControllerV4_1",
            "controller_config_path": str(V4_CONTROLLER_CONFIG),
            "controller_config_sha256": FROZEN_CONTROLLER_CONFIG_SHA256,
            "body_seed": 1729,
            "cpg_frequency_hz": FROZEN_CPG_FREQUENCY_HZ,
            "legacy_rest_compatibility": True,
            "controller_input": "raw current decoded MaleCNS left/right descending values, held unchanged for five 1 ms controller ticks",
            "controller_role": "neural outputs modulate an engineered CPG/native-template gait; not direct motor-neuron-to-muscle actuation",
            "stimulus_selection_of_motor_command": False,
            "fixed_drive_fallback": False,
            "pre_stimulus_gray_drive": "must be exactly [0,0] on every frame before onset or the case aborts before body tick; raw worker value is never replaced",
            "legacy_swing_clearance_stack": False,
            "external_wrench_or_root_qpos_qvel_injection": False,
        },
        "timeline": {
            "total_duration_s": 10.0,
            "grey_interval_s": [0.0, 5.0],
            "stimulus_interval_s": [5.0, 10.0],
            "visual_body_frame_hz": 200,
            "body_control_hz": 1000,
            "native_mujoco_hz": 10000,
            "native_samples_per_run": 100000,
            "visual_neural_frames_per_run": 2000,
            "control_ticks_per_run": 10000,
            "neural_tick_hz": 2000,
            "neural_ticks_per_visual_frame": 10,
        },
        "backend_contract": {
            "body": "FlyGym CPU; 1 ms controller ticks and 100 us MuJoCo steps",
            "vision": "FlyVis CUDA; 721 input columns per eye",
            "neural": "MaleCNS Triton; no CPU/stub fallback",
            "worker_script": str(WORKER_SCRIPT),
            "max_vram_gib": MAX_VRAM_GIB,
            "worker_capture_flags": ["--capture-visual-drive", "--capture-motor-channels"],
            "motor_channel_capture_ready_evidence": {
                "schema_version": 1,
                "total_leg_motor_cells": EXPECTED_MOTOR_CHANNEL_COUNT,
                "ready_frames_captured": 0,
                "ready_counts_shape": [0, EXPECTED_MOTOR_CHANNEL_COUNT],
                "window_dt_s": FRAME_DT_S,
                "window_steps": NEURAL_STEPS_PER_FRAME,
                "explicit_enabled_true_field": False,
                "explicit_enabled_false_rejects": True,
            },
            "serial_execution": "one parent and one neural worker for a single case at a time; no concurrent cases",
            "host_memory_plan": {
                "parent_scope_MemoryMax": PARENT_MEMORY_MAX,
                "rationale": "10 s native body trace ~1.11 GiB raw; parent/list/stack duplication plus neural worker may exceed original 5 GiB cgroup",
                "worker_source_constraint": "V3 is byte-equivalent to frozen V2 outside the authorized <=7 GiB host-memory guard/provenance delta",
            },
        },
        "capture_and_sampling": {
            "body_full_state": "session.save trajectory.npz at native 100 us includes qpos/qvel/body state and actuator/CPG records",
            "contacts": "contacts.npz plus per-native-sample contact arrays",
            "saved_state_contact_phase_relation": "preserve upstream sample indices: prior audit found contact/body/com/force[k] corresponds to qpos[k-1] while qpos[k] is post-integration; do not call mj_forward each step or change sample cadence",
            "drive_trace": "worker_reply decoded L/R must equal controller telemetry and native descending_drive_left_right for every control tick",
            "controller_tick_telemetry": "tick index, raw signal, phase, magnitude, commanded frequency, adhesion, and V4.1 hind-lift diagnostics saved per 1 ms tick",
        },
        "model_preflight_contract": {
            "frozen_v4_1_xml_sha256": FROZEN_COMPILED_MODEL_SHA256,
            "known_runtime_mesh_metadata_variant_sha256": FROZEN_COMPILED_MODEL_KNOWN_CONTENT_TYPE_VARIANT_SHA256,
            "only_xml_attribute_exception": "mesh.content_type may be missing or model/stl on an STL-backed mesh; all tags/text/other attributes/order remain exact",
            "mesh_asset_gate": "every mesh path, size, and SHA256 must exactly match the frozen V4.1 assets",
            "uniform_recompile_gate": "current and frozen XML are both compiled with MjModel.from_xml_path; effective options, object names/order, and all 61 named model arrays must be exact",
            "runtime_stage_gate": "save in-memory FlyGym MjSpec runtime arrays and t=0 force/kinematics; compare actual options/order/qpos/qvel/body mass and same-stage initial body trace fields to frozen evidence",
            "runtime_array_limit": "V4.1 freeze has no saved in-memory runtime model arrays; do not claim direct old runtime-array equality or compare them as if XML-recompiled arrays were the same stage",
        },
        "interpretation": {
            "known_limit": "the retained V4.1 development freeze has development_gate_pass=false; this online batch is exploratory even if its video shows gait",
            "retain_foot_clearance_and_support_gates": json.loads((PROJECT_ROOT / "config/flygym_cartesian_repair_protocol.json").read_text(encoding="utf-8"))["mechanical_acceptance"],
            "symmetric_yaw_context_limit_deg": 5.0,
            "no_gain_or_sign_tuning": True,
            "V2_static_parity_scope": "L azimuth reflection only; horizontal parity intervention, not full 2D subtype calibration",
            "mechanical_development_gates_remain_unmet": True,
            "readiness_requires_root_review_before_execute": True,
        },
    })
    dependency_roles = {
        "worker_config", "world_experiment_config", "world_stimulus_config",
        "world_scene_manifest", "worker_v2", "worker_v3", "worker_legacy",
        "flyvis_legacy_adapter", "parity_candidate_adapter", "malecns_engine",
        "malecns_gate", "neuromuscular_adapter", "neuromuscular_map", "sensory_map",
        "twin_contracts", "population_selectors", "virtual_world", "motor_channel_map",
        "motor_channel_capture", "t4t5_direction_asset", "flyvis_model_best_checkpoint",
        "flyvis_model_metadata", "prior_model_mismatch_diagnosis",
        "prior_xml_recompile_comparison",
    }
    dependency_roles.update(
        role for role in source_records
        if role.startswith("malecns_csr_") or role.startswith("scene_asset_")
    )
    source_freeze = {
        "schema_version": "flygym-online-eye-parity-v4-1-attempt5-source-freeze-1",
        "created_utc": _utc_now(),
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": _sha256_file(Path(__file__).resolve()),
        "v4_1_source_contract": source_contract,
        "eye_parity_static_guard": parity_guard,
        "worker_v3_resource_delta": worker_delta,
        "worker_config_effective_gain": EXPECTED_NEURAL_GAIN,
        "worker_config_experiment_sha256": _sha256_file(pilot_config_path),
        "python_package_fingerprints": {
            "body": body_fingerprint,
            "worker": worker_fingerprint,
        },
        "flyvis_model_files": model_files,
        "source_inputs": source_records,
        "worker_dependency_source_inputs": {
            role: source_records[role] for role in sorted(dependency_roles)
        },
        "ready_validation_fixture_inputs": {
            role: source_records[role]
            for role in (
                "attempt3_v3_ready_provenance", "attempt3_v3_ready_stimulus_copy",
                "attempt3_v3_worker_ready_failure_record", "successful_fixed_head_ready_protocol",
            )
        },
        "registered_condition_specs": case_rows,
    }
    _write_json(PREPARE_ROOT / "source-freeze.json", source_freeze)
    command_cases = {}
    for case in case_rows:
        command_cases[case["case_id"]] = [
            "systemd-run", "--user", "--wait", "--pipe", "--collect",
            f"--property=MemoryMax={PARENT_MEMORY_MAX}",
            "--property=LimitMEMLOCK=64M", "--property=RuntimeMaxSec=1800",
            "--property=CPUQuota=200%", "--",
            str(BODY_PYTHON), str(Path(__file__).resolve()),
            "--execute", "--case", case["case_id"],
        ]
    _write_json(PREPARE_ROOT / "command-plan.json", {
        "schema_version": "flygym-online-eye-parity-v4-1-attempt5-command-plan-1",
        "execution_authorized": False,
        "requires_root_review": True,
        "serial_order": [case["case_id"] for case in case_rows],
        "parent_commands": command_cases,
        "resource_note": "V3 worker accepts <=7 GiB and records its V2 parent hash; each serial case uses MemoryMax=7G, CPUQuota=200%, RuntimeMaxSec=1800, and existing <=8 GiB VRAM guard.",
    })
    validator_fixtures = _write_prepare_validation_fixtures(source_freeze)
    audit = {
        "schema_version": "flygym-online-eye-parity-v4-1-attempt5-prepare-audit-1",
        "status": "prepared_static_no_forward_no_physics_no_gpu",
        "created_utc": _utc_now(),
        "source_freeze_sha256": _sha256_file(PREPARE_ROOT / "source-freeze.json"),
        "protocol_sha256": _sha256_file(PREPARE_ROOT / "protocol.json"),
        "v4_1_source_contract_status": "pass",
        "eye_parity_independent_static_geometry_guard": "pass",
        "body_python_fingerprint": body_fingerprint,
        "worker_python_fingerprint": worker_fingerprint,
        "gpu_forward_started": False,
        "mujoco_physics_steps": 0,
        "neural_forward_frames": 0,
        "case_count_preregistered": len(case_rows),
        "development_gate_pass": False,
        "worker_ready_and_terminal_capture_schema_validation": validator_fixtures,
    }
    _write_json(PREPARE_ROOT / "preparation-audit.json", audit)
    outputs = {}
    for path in sorted(PREPARE_ROOT.rglob("*")):
        if path.is_file() and path.name != "integrity.json":
            outputs[str(path.relative_to(PREPARE_ROOT))] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _write_json(PREPARE_ROOT / "integrity.json", {
        "schema_version": "flygym-online-eye-parity-v4-1-attempt5-prepare-integrity-1",
        "file_count_excluding_integrity": len(outputs),
        "files": outputs,
    })
    audit["integrity_sha256"] = _sha256_file(PREPARE_ROOT / "integrity.json")
    return audit


def _verify_integrity_manifest(root: Path) -> None:
    path = root / "integrity.json"
    if not path.is_file():
        raise RuntimeError(f"missing integrity record: {path}")
    integrity = json.loads(path.read_text(encoding="utf-8"))
    for rel, record in integrity.get("files", {}).items():
        target = root / rel
        if (not target.is_file() or target.stat().st_size != int(record["size_bytes"])
                or _sha256_file(target) != record["sha256"]):
            raise RuntimeError(f"integrity mismatch for prepared source: {target}")


def _verify_worker_capture_artifacts(root: Path, expected_frames: int) -> dict[str, Any]:
    required = (
        "angular_stimulus_spec.json", "eye_columns.npz", "frames.jsonl",
        "visual_drive.npz", "muscle_channels.npz",
        "muscle_channels_membership.json", "provenance.json",
    )
    integrity_path = root / "integrity.json"
    if not integrity_path.is_file():
        raise RuntimeError("worker stopped without writing its integrity manifest")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    file_records = integrity.get("files")
    if not isinstance(file_records, dict):
        raise RuntimeError("worker integrity manifest has no file map")
    records: dict[str, Any] = {}
    for name in required:
        item = file_records.get(name)
        path = root / name
        if (not isinstance(item, dict) or not path.is_file()
                or int(item.get("size_bytes", -1)) != path.stat().st_size
                or item.get("sha256") != _sha256_file(path)):
            raise RuntimeError(f"required worker capture is absent or fails its integrity record: {name}")
        records[name] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": item["sha256"],
        }
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    visual = provenance.get("visual_drive_capture", {})
    motor = provenance.get("motor_channel_capture", {})
    if (visual.get("enabled") is not True
            or visual.get("input_drive_rows_captured") != expected_frames
            or visual.get("saved_frames") != expected_frames
            or visual.get("drive_values_shape", [None])[0] != expected_frames):
        raise RuntimeError("worker visual-drive capture is incomplete")
    motor_capture_validation = _validate_motor_channel_capture_metadata(
        motor, output_dir=root.resolve(), expected_frames=expected_frames,
        stage="terminal",
    )
    resource_caps = provenance.get("resource_caps", {})
    if (int(resource_caps.get("host_memory_limit_bytes", 0)) <= 0
            or int(resource_caps.get("host_memory_limit_bytes", 0)) > 7 * 1024**3
            or float(resource_caps.get("max_vram_gib_per_backend", float("inf"))) > MAX_VRAM_GIB
            or float(provenance.get("malecns", {}).get("max_vram_gib", float("inf"))) > MAX_VRAM_GIB):
        raise RuntimeError("worker completed outside its frozen <=7 GiB host / <=8 GiB VRAM limits")
    with np.load(root / "muscle_channels.npz", allow_pickle=False) as motor_arrays:
        required_motor_arrays = {"counts", "seq", "time_s"}
        if not required_motor_arrays.issubset(motor_arrays.files):
            raise RuntimeError("worker muscle-channel NPZ is missing counts/seq/time_s arrays")
        raw_counts = np.asarray(motor_arrays["counts"])
        raw_seq = np.asarray(motor_arrays["seq"])
        raw_time = np.asarray(motor_arrays["time_s"], dtype=np.float64)
    if (list(raw_counts.shape) != [expected_frames, EXPECTED_MOTOR_CHANNEL_COUNT]
            or raw_counts.dtype != np.dtype("uint16")
            or raw_seq.shape != (expected_frames,)
            or not np.array_equal(raw_seq, np.arange(expected_frames, dtype=raw_seq.dtype))
            or raw_time.shape != (expected_frames,) or not np.isfinite(raw_time).all()
            or (expected_frames > 1 and not np.allclose(
                np.diff(raw_time), FRAME_DT_S, rtol=0.0, atol=1e-9
            ))):
        raise RuntimeError("worker muscle-channel arrays do not match the recorded frame/count clock")
    with (root / "frames.jsonl").open("rb") as handle:
        line_count = sum(1 for line in handle if line.strip())
    if line_count != expected_frames:
        raise RuntimeError(f"worker saved {line_count} frame records, expected {expected_frames}")
    return {
        "integrity_manifest_sha256": _sha256_file(integrity_path),
        "frame_record_count": line_count,
        "visual_drive_capture": visual,
        "motor_channel_capture": motor,
        "motor_channel_capture_validation": motor_capture_validation,
        "motor_channel_arrays": {
            "counts_shape": list(raw_counts.shape),
            "counts_dtype": str(raw_counts.dtype),
            "sequential_frame_indices": True,
            "five_ms_frame_clock": True,
        },
        "required_files": records,
    }


def _require_frozen_prepare() -> dict[str, Any]:
    for name in ("protocol.json", "source-freeze.json", "preparation-audit.json", "integrity.json"):
        path = PREPARE_ROOT / name
        if not path.is_file():
            raise RuntimeError(f"run default prepare mode before --execute; missing {path}")
    _verify_integrity_manifest(PREPARE_ROOT)
    audit = json.loads((PREPARE_ROOT / "preparation-audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((PREPARE_ROOT / "source-freeze.json").read_text(encoding="utf-8"))
    if audit.get("status") != "prepared_static_no_forward_no_physics_no_gpu":
        raise RuntimeError("prepare bundle does not certify its static-only boundary")
    if _sha256_file(Path(__file__).resolve()) != freeze.get("wrapper_sha256"):
        raise RuntimeError("V4.1 online wrapper changed after the prepare source freeze")
    _verify_live_source_inputs(freeze)
    _verify_v4_1_source_contract()
    _verify_eye_parity_geometry()
    _verify_worker_v3_resource_delta()
    body_fp = _python_package_fingerprint(EXPECTED_BODY_ENV, ("flygym", "mujoco", "numpy", "scipy"))
    worker_fp = _python_package_fingerprint(EXPECTED_WORKER_ENV, ("flyvis", "torch", "triton", "numpy", "scipy"))
    if body_fp != freeze.get("python_package_fingerprints", {}).get("body"):
        raise RuntimeError("FlyGym Python/package fingerprint differs from the frozen prepare record")
    if worker_fp != freeze.get("python_package_fingerprints", {}).get("worker"):
        raise RuntimeError("FlyVis/MaleCNS Python/package fingerprint differs from the frozen prepare record")
    return {"audit": audit, "freeze": freeze, "protocol": json.loads((PREPARE_ROOT / "protocol.json").read_text(encoding="utf-8"))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Without --execute this writes/revalidates a CPU-only prepare bundle. "
            "--execute is one frozen case and requires review of the prepare bundle."
        ),
    )
    parser.add_argument("--execute", action="store_true",
                        help="run exactly one preregistered 10 s case; default is prepare-only")
    parser.add_argument("--case", choices=tuple(CASE_SPEC_PATHS),
                        help="one frozen static, +60 deg/s, or -60 deg/s case; required with --execute")
    parser.add_argument("--out", type=Path,
                        help="new case directory under the frozen run root; defaults to its case id")
    parser.add_argument("--worker-timeout-s", type=float, default=120.0,
                        help="maximum wall time for worker startup or one 5 ms neural window")
    parser.add_argument("--shutdown-timeout-s", type=float, default=60.0,
                        help="bounded worker stop acknowledgement/exit wait (maximum 60 s)")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.execute:
        return
    if args.case is None:
        parser.error("--execute requires exactly one preregistered --case")
    if not math.isfinite(args.worker_timeout_s) or args.worker_timeout_s <= 0:
        parser.error("--worker-timeout-s must be finite and positive")
    if (not math.isfinite(args.shutdown_timeout_s)
            or not 0 < args.shutdown_timeout_s <= MAX_SHUTDOWN_TIMEOUT_S):
        parser.error(f"--shutdown-timeout-s must be in (0, {MAX_SHUTDOWN_TIMEOUT_S:g}]")
    args.stimulus_condition = "angular"
    args.angular_stimulus_spec = CASE_SPEC_PATHS[args.case].resolve()
    try:
        (args.angular_stimulus, args.angular_stimulus_spec_document,
         args.angular_stimulus_spec_raw_bytes) = _load_registered_case(args.case)
    except ValueError as exc:
        parser.error(str(exc))
    args.angular_stimulus_spec_sha256 = hashlib.sha256(
        args.angular_stimulus_spec_raw_bytes
    ).hexdigest()
    stimulus = args.angular_stimulus
    if (not math.isclose(float(stimulus.contrast), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(stimulus.wavelength_deg), 60.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(stimulus.onset_s), 5.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(stimulus.duration_s), 5.0, rel_tol=0.0, abs_tol=1e-12)):
        parser.error("registered case must remain c1 / 60-degree period / 5 s gray + 5 s stimulus")
    expected_case = CASE_EXPECTED[args.case]
    if (stimulus.head_direction_sign != expected_case["head_direction_sign"]
            or not math.isclose(float(stimulus.angular_speed_deg_s), expected_case["speed_deg_s"],
                                rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(stimulus.temporal_frequency_hz),
                                expected_case["temporal_frequency_hz"],
                                rel_tol=0.0, abs_tol=1e-12)):
        parser.error("case speed/sign does not match the preregistered static, +60, or -60 deg/s condition")
    args.duration_s = 10.0
    args.stimulus_onset_s = 5.0
    args.stimulus_speed_mm_s = None
    args.seed = 1729
    args.neural_seed = 1729
    args.legacy_rest_compatibility = True
    args.cpg_frequency_hz = FROZEN_CPG_FREQUENCY_HZ
    args.worker_python = EXPECTED_WORKER_ENV.resolve()
    args.allow_extended_duration = True
    args.deterministic_vision = True
    args.deterministic_neural_delivery = True
    args.scene_origin_x_m = 0.0
    args.scene_origin_y_m = 0.0
    args.run_id = args.case
    args.out = args.out or DEFAULT_OUTPUT_ROOT / args.case
    args.frame_count = int(round(args.duration_s / FRAME_DT_S))
    if args.frame_count != 2000 or MAX_WORKER_CAPTURE_FRAMES < args.frame_count:
        parser.error("the frozen online case requires exactly 2,000 worker frames")
    if not BODY_PYTHON.is_file() or not args.worker_python.is_file():
        parser.error("registered FlyGym or FlyVis Python executable is unavailable")
    if not BODY_RUNNER.is_file() or not WORKER_SCRIPT.is_file() or not WORKER_CONFIG.is_file():
        parser.error("a registered runtime source/config is unavailable")


def _partial_state(session: Any) -> dict[str, Any]:
    data = session.data
    thorax = int(session.thorax_body_id)
    return {
        "time_s": float(data.time),
        "thorax_position_mm": np.asarray(data.xpos[thorax], dtype=float).tolist(),
        "thorax_quaternion_wxyz": np.asarray(data.xquat[thorax], dtype=float).tolist(),
        "qpos": np.asarray(data.qpos, dtype=float).tolist(),
        "qvel": np.asarray(data.qvel, dtype=float).tolist(),
    }


def _write_partial_native_trace(session: Any, output_dir: Path) -> Path | None:
    count = int(session._native_samples)
    if count <= 0:
        return None
    arrays = session.all_arrays
    keep = (
        "time_s", "qpos", "qvel", "body_xpos_mm", "body_xquat_wxyz",
        "body_xvel_mm_s", "body_xangvel_rad_s", "foot_contact_found",
        "foot_contact_force_local_native", "foot_contact_torque_local_native",
        "foot_contact_pos_mm", "foot_contact_normal_world",
        "foot_contact_fz_up_native", "tarsus5_xpos_mm",
        "root_external_wrench_native", "qfrc_applied_native",
        "thorax_heading_yaw_rad", "thorax_collision_clearance_min_mm",
    )
    target = output_dir / "partial_native_trace.npz"
    np.savez_compressed(target, **{name: arrays[name][:count] for name in keep},
                        cpg_phase_rad=session.cpg_phase_by_sample[:count],
                        cpg_magnitude=session.cpg_magnitude_by_sample[:count],
                        descending_drive_left_right=session.descending_drive_by_sample[:count],
                        root_qpos_write_count=session.root_qpos_write_count_by_sample[:count],
                        root_qvel_write_count=session.root_qvel_write_count_by_sample[:count])
    return target


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        if args.case is not None or args.out is not None:
            parser.error("--case and --out apply only to an explicit --execute case")
        audit = _write_prepare_bundle()
        print(json.dumps({
            "status": audit["status"],
            "prepare_root": str(PREPARE_ROOT),
            "source_freeze_sha256": audit["source_freeze_sha256"],
            "protocol_sha256": audit["protocol_sha256"],
            "integrity_sha256": audit.get("integrity_sha256"),
            "cases_preregistered": audit["case_count_preregistered"],
            "gpu_forward_started": False,
            "mujoco_physics_steps": 0,
        }, indent=2, sort_keys=True))
        return 0
    if args.case is None:
        parser.error("--execute requires one --case after prepare has been reviewed")
    frozen_prepare = _require_frozen_prepare()
    _validate_args(parser, args)
    if Path(sys.executable).resolve() != BODY_PYTHON.resolve():
        parser.error(
            f"--execute must run under the registered FlyGym Python: {BODY_PYTHON} "
            f"(current interpreter: {sys.executable})"
        )
    n_frames = int(args.frame_count)

    # Preserve this environment before FlyGymReferenceSession intentionally
    # hides CUDA from the CPU physics process.
    worker_environment = os.environ.copy()
    worker_environment["FLYVIS_ROOT_DIR"] = worker_environment.get(
        "FLYVIS_ROOT_DIR", "/data/models/flyvis"
    )
    worker_environment["PYTHONUNBUFFERED"] = "1"
    scripts_path = str(PROJECT_ROOT / "scripts")
    existing_pythonpath = worker_environment.get("PYTHONPATH", "")
    worker_environment["PYTHONPATH"] = scripts_path + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    run_id = args.case
    output_dir = (_project_path(args.out).resolve() if args.out is not None
                  else (DEFAULT_OUTPUT_ROOT / run_id).resolve())
    containment_root = ARTIFACT_CONTAINMENT_ROOT.resolve()
    if (output_dir == containment_root or not output_dir.is_relative_to(containment_root)
            or not output_dir.is_relative_to(DEFAULT_OUTPUT_ROOT.resolve())):
        parser.error(f"--out must be a new case directory below {DEFAULT_OUTPUT_ROOT}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_output = output_dir / "neural-worker"

    from run_flygym_reference_walk import FlyGymReferenceSession, WalkConfig

    worker: NeuralWorker | None = None
    session: Any = None
    frame_file: TextIO | None = None
    tick_file: TextIO | None = None
    ready_record: dict[str, Any] | None = None
    worker_command: list[str] = []
    worker_shutdown: dict[str, Any] | None = None
    error: BaseException | None = None
    started_utc = _utc_now()
    started_wall = time.monotonic()
    last_seq = -1
    successful_frames = 0
    scene_origin_xy = (float(args.scene_origin_x_m), float(args.scene_origin_y_m))
    rigging_sources: dict[str, Any] = {}
    controller_binding: dict[str, Any] | None = None
    runtime_model_preflight: dict[str, Any] | None = None
    body_tick_count = 0
    pre_stimulus_zero_frames = 0
    direct_drive_match_ticks = 0
    body_wall_s = 0.0
    worker_wait_wall_s = 0.0
    body_initialization_wall_s = 0.0
    worker_startup_wall_s = 0.0

    try:
        body_init_start = time.monotonic()
        reference_effective = json.loads(
            (WEAK_REFERENCE_ROOT / "effective_config.json").read_text(encoding="utf-8")
        )
        reference_walk_config = reference_effective["walk_config"]
        allowed_walk_fields = {item.name for item in fields(WalkConfig)}
        body_values = {key: value for key, value in reference_walk_config.items()
                       if key in allowed_walk_fields}
        body_values["duration_s"] = float(args.duration_s)
        for tuple_field in (
            "position_actuator_forcerange_native", "spawn_position_mm",
            "spawn_quaternion_wxyz", "camera_resolution_hw",
        ):
            if isinstance(body_values.get(tuple_field), list):
                body_values[tuple_field] = tuple(body_values[tuple_field])
        body_config = WalkConfig(**body_values)
        expected_body_config = dict(reference_walk_config)
        expected_body_config["duration_s"] = float(args.duration_s)
        actual_body_config = asdict(body_config)
        if {key: (list(value) if isinstance(value, tuple) else value)
            for key, value in actual_body_config.items()} != expected_body_config:
            raise RuntimeError("online body configuration differs from the frozen V4.1 development plant")
        session = FlyGymReferenceSession(
            duration_s=float(args.duration_s),
            seed=int(args.seed),
            render_video=False,
            config=body_config,
            run_id=run_id,
        )
        frozen_initial_state = json.loads(V4_INITIAL_STATE.read_text(encoding="utf-8"))
        if (not math.isclose(float(session.time_s), 0.0, rel_tol=0.0, abs_tol=1e-12)
                or not np.allclose(np.asarray(session.data.qpos, dtype=np.float64),
                                   np.asarray(frozen_initial_state["qpos"], dtype=np.float64),
                                   rtol=0.0, atol=1e-12)
                or not np.allclose(np.asarray(session.data.qvel, dtype=np.float64),
                                   np.asarray(frozen_initial_state["qvel"], dtype=np.float64),
                                   rtol=0.0, atol=1e-12)):
            raise RuntimeError("initial FlyGym keyframe qpos/qvel differs from the frozen V4.1 state")
        runtime_model_preflight = _verify_runtime_model_contract(session, output_dir)
        controller_config_doc = json.loads(V4_CONTROLLER_CONFIG.read_text(encoding="utf-8"))
        controller_config_values = controller_config_doc.get("config")
        if controller_config_doc.get("controller_class") != (
                "flygym_phase_speed_gait_v4_1.PhaseSpeedGaitController"):
            raise RuntimeError("frozen controller config identifies a different controller class")
        from flygym_phase_speed_gait_v4_1 import (
            PhaseSpeedGaitConfig, PhaseSpeedGaitController,
        )
        session.controller = PhaseSpeedGaitController(
            session, PhaseSpeedGaitConfig(**controller_config_values)
        )
        controller_binding = session.controller.controller_metadata()
        if (controller_binding.get("controller") != "PhaseSpeedGaitControllerV4_1"
                or not math.isclose(
                    float(controller_binding.get("effective_cycle_frequency_hz_at_unit_drive", float("nan"))),
                    FROZEN_CPG_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-9)
                or controller_binding.get("stimulus_or_desired_yaw_feedback") is not False
                or controller_binding.get("external_force_or_root_state_write") is not False):
            raise RuntimeError("constructed V4.1 controller does not match its frozen phase-speed contract")
        _write_json(output_dir / "controller_binding.json", {
            "schema_version": "flygym-v4-1-controller-binding-1",
            "status": "frozen controller attached before first control tick",
            "development_gate_pass": False,
            "module": str(PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v4_1.py"),
            "module_sha256": FROZEN_CONTROLLER_SHA256,
            "config_path": str(V4_CONTROLLER_CONFIG),
            "config_sha256": FROZEN_CONTROLLER_CONFIG_SHA256,
            "config": controller_config_values,
            "controller_metadata": controller_binding,
            "neural_drive_source": "raw decoded MaleCNS left/right output from V3 worker; no transform/fallback",
            "actuation_semantics": "neural outputs modulate engineered CPG phase-speed/native-template gait; not direct MN-to-muscle actuation",
        })
        body_initialization_wall_s = time.monotonic() - body_init_start
        if not math.isclose(session.model.opt.timestep, NATIVE_DT_S, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("body runner did not compile with the required 100 µs timestep")
        if not math.isclose(session.config.control_timestep_s, CONTROL_DT_S,
                            rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("body runner did not use the required 1 ms control tick")

        vision_path, sensors = _load_vision_camera_config(session)
        body_ids = _body_ids_by_name(session)
        camera_initial = _camera_geometry(session, sensors, body_ids)
        initial_eye_midpoint = camera_initial["midpoint_mm"].copy()
        eye_separation_m = float(camera_initial["baseline_m"])
        if not math.isclose(eye_separation_m, 0.00076, rel_tol=0.0, abs_tol=2e-6):
            raise RuntimeError(
                f"pinned FlyGym camera baseline changed unexpectedly: {eye_separation_m:.9g} m"
            )
        # The pinned NeuroMechFly loader exposes the vision path; its sibling
        # rigging file is the exact default used when that model was built.
        rigging_path = vision_path.with_name("rigging.yaml")
        if not rigging_path.is_file():
            raise FileNotFoundError(f"FlyGym rigging config unavailable: {rigging_path}")
        rigging_sources = {
            "vision_config": {"path": str(vision_path), "sha256": _sha256_file(vision_path)},
            "rigging_config": {"path": str(rigging_path), "sha256": _sha256_file(rigging_path)},
            "compiled_model_sha256": session.compiled_mjcf_sha256,
            "compiled_camera_sensor_offsets_mm": sensors,
        }

        worker_command = [
            str(args.worker_python), str(WORKER_SCRIPT),
            "--config", str(WORKER_CONFIG),
            "--stimulus-condition", args.stimulus_condition,
            "--stimulus-onset-s", format(args.stimulus_onset_s, ".17g"),
            "--seed", str(args.neural_seed),
            "--max-frames", str(n_frames),
            "--neural-device", "cuda",
            "--neural-backend", "triton",
            "--vision-device", "cuda:0",
            "--max-vram-gib", str(MAX_VRAM_GIB),
            "--eye-separation-m", format(eye_separation_m, ".17g"),
            "--capture-visual-drive",
            "--capture-motor-channels",
            "--out", str(worker_output),
        ]
        if args.angular_stimulus is not None:
            worker_command.extend([
                "--angular-stimulus-spec", str(args.angular_stimulus_spec),
                "--static-warmup-s", format(args.angular_stimulus.onset_s, ".17g"),
            ])
        if args.deterministic_vision:
            worker_command.append("--deterministic-vision")
        if args.deterministic_neural_delivery:
            worker_command.append("--deterministic-neural-delivery")
        if args.stimulus_speed_mm_s is not None:
            worker_command.extend([
                "--stimulus-speed-mm-s",
                format(args.stimulus_speed_mm_s, ".17g"),
            ])
        worker_start = time.monotonic()
        expected_renderer = (
            "fixed_arena_angular_field_cpu" if args.angular_stimulus is not None
            else "virtual_world_raycast"
        )
        worker = NeuralWorker(
            worker_command, worker_environment, args.worker_timeout_s,
            expected_renderer=expected_renderer,
        )
        worker_startup_wall_s = time.monotonic() - worker_start
        ready_record = worker.ready
        parity_ready_record = _validate_eye_parity_ready(
            ready_record, frozen_prepare["freeze"], args
        )
        if ready_record.get("seed") != args.neural_seed:
            raise RuntimeError("worker neural seed differs from the requested seed")
        if not math.isclose(float(ready_record.get("eye_separation_m", float("nan"))),
                            eye_separation_m, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("worker eye separation differs from compiled FlyGym optical geometry")
        if ready_record.get("stimulus_condition") != args.stimulus_condition:
            raise RuntimeError("worker scene condition differs from requested visual condition")
        if args.angular_stimulus is not None:
            expected_angular = {
                "spec_sha256": args.angular_stimulus_spec_sha256,
                "name": args.angular_stimulus.name,
                "contrast_study_parameter_c": args.angular_stimulus.contrast,
                "head_direction_sign": args.angular_stimulus.head_direction_sign,
                "head_angular_speed_deg_s": args.angular_stimulus.angular_speed_deg_s,
                "arena_angular_speed_deg_s": args.angular_stimulus.arena_angular_speed_deg_s,
                "wavelength_deg": args.angular_stimulus.wavelength_deg,
                "temporal_frequency_hz": args.angular_stimulus.temporal_frequency_hz,
                "onset_s": args.angular_stimulus.onset_s,
                "duration_s": args.angular_stimulus.duration_s,
                "end_exclusive_s": args.angular_stimulus.end_s,
            }
            actual_angular = ready_record.get("angular_stimulus")
            if (ready_record.get("stimulus_mode") != "fixed_arena_angular_field"
                    or ready_record.get("angular_stimulus_spec_sha256") != args.angular_stimulus_spec_sha256
                    or not isinstance(actual_angular, dict)):
                raise RuntimeError("worker did not acknowledge the frozen live angular stimulus spec")
            for field, expected_value in expected_angular.items():
                actual_value = (ready_record.get("angular_stimulus_spec_sha256")
                                if field == "spec_sha256" else actual_angular.get(field))
                if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                    if (isinstance(actual_value, bool) or not isinstance(actual_value, (int, float))
                            or not math.isclose(float(actual_value), float(expected_value),
                                                rel_tol=0.0, abs_tol=1e-9)):
                        raise RuntimeError(f"worker angular spec mismatch for {field}")
                elif actual_value != expected_value:
                    raise RuntimeError(f"worker angular spec mismatch for {field}")
            if (actual_angular.get("display_geometry")
                    != "fixed arena angular field; infinite-distance angular-screen approximation"
                    or ready_record.get("stimulus_speed_effective_mm_s") is not None
                    or ready_record.get("stimulus_velocity_virtualworld_mps") is not None):
                raise RuntimeError("worker angular renderer/backend metadata is inconsistent")
        else:
            requested_ready_speed = ready_record.get("stimulus_speed_override_mm_s")
            if args.stimulus_speed_mm_s is None:
                if requested_ready_speed is not None:
                    raise RuntimeError("worker unexpectedly overrode the configured panel speed")
            elif (requested_ready_speed is None
                  or not math.isclose(float(requested_ready_speed), args.stimulus_speed_mm_s,
                                      rel_tol=0.0, abs_tol=1e-12)):
                raise RuntimeError("worker speed-override metadata differs from the requested magnitude")
            expected_panel_velocity = _expected_panel_velocity_virtualworld_mps(
                args.stimulus_condition, args.stimulus_speed_mm_s,
            )
            actual_panel_velocity = _finite_vector(
                ready_record.get("stimulus_velocity_virtualworld_mps"), (3,),
                "worker ready stimulus_velocity_virtualworld_mps",
            )
            if (ready_record.get("stimulus_velocity_frame")
                    != "VirtualWorld scene-coordinate frame"
                    or not np.allclose(actual_panel_velocity, expected_panel_velocity,
                                       rtol=0.0, atol=1e-12)):
                raise RuntimeError(
                    "worker's actual VirtualWorld panel velocity differs from the condition/speed request"
                )
            expected_effective_speed_mm_s = float(np.linalg.norm(expected_panel_velocity) * 1000.0)
            if not math.isclose(
                float(ready_record.get("stimulus_speed_effective_mm_s", float("nan"))),
                expected_effective_speed_mm_s, rel_tol=0.0, abs_tol=1e-9,
            ):
                raise RuntimeError("worker's effective panel speed metadata disagrees with its velocity vector")
        if not math.isclose(float(ready_record.get("stimulus_onset_s", float("nan"))),
                            args.stimulus_onset_s, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("worker stimulus onset differs from requested onset")
        if not math.isclose(float(ready_record.get("static_warmup_s", float("nan"))),
                            (args.angular_stimulus.onset_s if args.angular_stimulus is not None else 1.0),
                            rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("worker static visual warmup differs from the effective stimulus timeline")

        frame_file = (output_dir / "online_bridge.jsonl").open("x", encoding="utf-8", buffering=1)
        tick_file = (output_dir / "body_control_ticks.jsonl").open("x", encoding="utf-8", buffering=1)
        for seq in range(n_frames):
            frame_time_s = seq * FRAME_DT_S
            if not math.isclose(session.time_s, frame_time_s, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(
                    f"body time is not on the next 5 ms frame boundary: "
                    f"body={session.time_s:.12g}, frame={frame_time_s:.12g}"
                )
            state = _build_virtual_pose(session, sensors, body_ids, scene_origin_xy,
                                        initial_eye_midpoint)
            input_record = {
                "protocol": PROTOCOL,
                "type": "frame",
                "seq": seq,
                "time_s": frame_time_s,
                "dt_s": FRAME_DT_S,
                "pose": state["pose"],
            }
            reply, worker_elapsed = worker.request(input_record)
            worker_wait_wall_s += worker_elapsed
            left, right = _validate_drive_reply(reply, seq, frame_time_s, ready_record)
            raw_drive = np.asarray([left, right], dtype=np.float64)
            if frame_time_s < args.stimulus_onset_s:
                if not np.array_equal(raw_drive, np.zeros(2, dtype=np.float64)):
                    raise RuntimeError(
                        f"MaleCNS emitted nonzero drive during the gray interval at seq={seq}; "
                        "no body control tick for this reply was run"
                    )
                pre_stimulus_zero_frames += 1
            reply_hash = _sha256_json(reply)
            frame_row = {
                "protocol": PROTOCOL,
                "seq": seq,
                "frame_time_s": frame_time_s,
                "neural_window_start_s": frame_time_s,
                "neural_window_end_s": frame_time_s + FRAME_DT_S,
                "input_pose": state["pose"],
                "source_body_pose": state["source_pose"],
                "mapped_eye_origins_m": state["virtual_eye_origins_m"],
                "worker_request_sha256": _sha256_json(input_record),
                "worker_reply_sha256": reply_hash,
                "worker_reply": reply,
                "decoded_drive_left_right": [left, right],
                "body_controller_input_left_right": [left, right],
                "raw_worker_drive_passed_without_transform": True,
                "pre_stimulus_gray_zero_gate_pass": (
                    bool(np.array_equal(raw_drive, np.zeros(2, dtype=np.float64)))
                    if frame_time_s < args.stimulus_onset_s else None
                ),
                "worker_roundtrip_wall_s": worker_elapsed,
                "body_advance_follows_reply": True,
            }
            frame_file.write(json.dumps(frame_row, sort_keys=True, allow_nan=False) + "\n")
            frame_file.flush()

            # The frozen controller receives zero during gray and the actual
            # decoded values from the first stimulus-active neural window.
            for tick_in_frame in range(CONTROL_TICKS_PER_FRAME):
                tick_start_s = session.time_s
                expected_start = frame_time_s + tick_in_frame * CONTROL_DT_S
                if not math.isclose(tick_start_s, expected_start,
                                    rel_tol=0.0, abs_tol=1e-9):
                    raise RuntimeError("body control tick split out of sync with neural window")
                step_wall_start = time.monotonic()
                native_first = int(session._native_samples)
                step = session.step_control_tick(raw_drive)
                step_elapsed = time.monotonic() - step_wall_start
                body_wall_s += step_elapsed
                body_tick_count += 1
                if int(step["native_samples_added"]) != NATIVE_STEPS_PER_CONTROL_TICK:
                    raise RuntimeError("FlyGym control tick did not advance ten native MuJoCo steps")
                expected_end = expected_start + CONTROL_DT_S
                if not math.isclose(session.time_s, expected_end,
                                    rel_tol=0.0, abs_tol=1e-9):
                    raise RuntimeError("FlyGym control tick ended outside its 1 ms window")
                controller_telemetry = session.controller.telemetry_snapshot()
                controller_drive = np.asarray(
                    controller_telemetry.get("descending_signal_lr", []), dtype=np.float64
                )
                step_drive = np.asarray(step.get("descending_signal", []), dtype=np.float64)
                native_last = int(session._native_samples)
                native_drives = np.asarray(
                    session.descending_drive_by_sample[native_first:native_last], dtype=np.float64
                )
                if (controller_drive.shape != (2,) or step_drive.shape != (2,)
                        or not np.array_equal(controller_drive, raw_drive)
                        or not np.array_equal(step_drive, raw_drive)
                        or native_drives.shape[0] == 0
                        or not np.all(native_drives == raw_drive[None, :])):
                    raise RuntimeError(
                        "worker L/R output did not reach the V4.1 controller and native trace unchanged"
                    )
                if frame_time_s < args.stimulus_onset_s and (
                    controller_telemetry.get("neutral_hold") is not True
                    or controller_telemetry.get("side_active_lr") != [False, False]
                ):
                    raise RuntimeError("V4.1 did not hold its neutral zero-drive state during gray")
                direct_drive_match_ticks += 1
                sample = int(session._native_samples - 1)
                tick = {
                    "seq": seq,
                    "tick_in_frame": tick_in_frame,
                    "control_window_start_s": expected_start,
                    "control_window_end_s": expected_end,
                    "neural_reply_sha256": reply_hash,
                    "descending_drive_left_right": [left, right],
                    "controller_received_drive_left_right": controller_drive.tolist(),
                    "native_trace_drive_left_right": raw_drive.tolist(),
                    "direct_worker_to_controller_drive_match": True,
                    "controller_tick_telemetry": controller_telemetry,
                    "native_sample_indices_inclusive": [native_first, native_last - 1],
                    "cpg_phase_rad": np.asarray(session.cpg_phase_by_sample[sample], dtype=float).tolist(),
                    "cpg_magnitude": np.asarray(session.cpg_magnitude_by_sample[sample], dtype=float).tolist(),
                    "foot_contact_fz_up_native": np.asarray(
                        session.all_arrays["foot_contact_fz_up_native"][sample], dtype=float
                    ).tolist(),
                    "foot_contact_found": np.asarray(
                        session.all_arrays["foot_contact_found"][sample], dtype=int
                    ).tolist(),
                    "root_external_wrench_native": np.asarray(
                        session.root_external_wrench_by_sample[sample], dtype=float
                    ).tolist(),
                    "qfrc_applied_abs_max_native": float(np.max(np.abs(
                        session.all_arrays["qfrc_applied_native"][sample]
                    ))),
                    "root_qpos_write_count": int(session.root_qpos_write_count_by_sample[sample]),
                    "root_qvel_write_count": int(session.root_qvel_write_count_by_sample[sample]),
                    "native_samples_added": int(step["native_samples_added"]),
                    "step_wall_s": step_elapsed,
                }
                tick_file.write(json.dumps(tick, sort_keys=True, allow_nan=False) + "\n")
            tick_file.flush()
            successful_frames += 1
            last_seq = seq
            if not args.quiet_progress and (seq % 100 == 99 or seq == n_frames - 1):
                print(f"online optomotor progress: t={session.time_s:.3f}s / {args.duration_s:.3f}s "
                      f"frames={seq + 1}/{n_frames}", file=sys.stderr, flush=True)

        if not math.isclose(session.time_s, args.duration_s, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("body duration differs from the requested online duration")
        if pre_stimulus_zero_frames != round(args.stimulus_onset_s / FRAME_DT_S):
            raise RuntimeError("gray interval did not contain exactly 1,000 zero-drive neural frames")
        if direct_drive_match_ticks != n_frames * CONTROL_TICKS_PER_FRAME:
            raise RuntimeError("not every one of the 10,000 body ticks has a direct-drive match record")
        worker_shutdown = worker.stop(timeout_s=args.shutdown_timeout_s)
        stop_ack = worker_shutdown.get("ack") or {}
        if stop_ack.get("frame_count") != n_frames or stop_ack.get("last_seq") != n_frames - 1:
            raise RuntimeError("worker stop acknowledgement does not cover every requested frame")
        if not math.isclose(float(stop_ack.get("last_time_s", float("nan"))),
                            args.duration_s - FRAME_DT_S, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("worker stop acknowledgement has the wrong final frame timestamp")
        worker = None
        frame_file.close()
        frame_file = None
        tick_file.close()
        tick_file = None

        worker_artifacts = _verify_worker_capture_artifacts(worker_output, n_frames)

        runner_result = session.save(output_dir)
        runner_manifest_path = output_dir / "manifest.json"
        runner_manifest = json.loads(runner_manifest_path.read_text(encoding="utf-8"))
        if (runner_manifest.get("actuation_boundary", {}).get("qpos_qvel_assignments_after_keyframe_reset") != 0
                or runner_manifest.get("actuation_boundary", {}).get("root_external_wrench_applied") is not False
                or runner_manifest.get("render_backend", {}).get("gpu_used") is not False):
            raise RuntimeError("FlyGym runner manifest violates the CPU/no-root-input contract")
        outputs = [
            "manifest.json", "integrity.json", "trajectory.npz", "contacts.npz",
            "compiled_model.xml", "model_options.json", "array_order.json",
            "upstream_asset_hashes.json", "summary.json", "diagnostics.png",
            "effective_config.json", "initial_state.json", "seed.json",
            "online_bridge.jsonl", "body_control_ticks.jsonl",
            "controller_binding.json", "runtime-model-preflight.json",
            "runtime-model-preflight.npz", "runtime-current-compiled-model.xml",
        ]
        bridge_manifest = {
            "schema_version": "flygym-online-optomotor-artifact-1",
            "protocol": PROTOCOL,
            "status": "complete",
            "created_utc": _utc_now(),
            "started_utc": started_utc,
            "run_id": run_id,
            "project_root": str(PROJECT_ROOT),
            "driver_file": str(Path(__file__).resolve()),
            "driver_sha256": _sha256_file(Path(__file__).resolve()),
            "body_runner_file": str(BODY_RUNNER),
            "body_runner_sha256": _sha256_file(BODY_RUNNER),
            "worker_script_sha256": _sha256_file(WORKER_SCRIPT),
            "angular_stimulus_helper_sha256": (
                _sha256_file(PROJECT_ROOT / "scripts" / "standard_optomotor_stimulus.py")
                if args.angular_stimulus is not None else None
            ),
            "worker_command": worker_command,
            "worker_python_symlink_path": str(args.worker_python),
            "worker_python_resolved_target": str(args.worker_python.resolve()),
            "worker_ready": ready_record,
            "worker_ready_gate": parity_ready_record,
            "worker_shutdown": worker_shutdown,
            "worker_artifact_directory": str(worker_output),
            "worker_capture_artifacts": worker_artifacts,
            "prepare_source_freeze_path": str(PREPARE_ROOT / "source-freeze.json"),
            "prepare_source_freeze_sha256": _sha256_file(PREPARE_ROOT / "source-freeze.json"),
            "worker_environment_cuda_visible_devices_before_body_hide": (
                worker_environment.get("CUDA_VISIBLE_DEVICES")
            ),
            "body_python": sys.version,
            "platform": platform.platform(),
            "parameters": {
                "duration_s": args.duration_s,
                "stimulus_condition": args.stimulus_condition,
                "stimulus_mode": (
                    "fixed_arena_angular_field" if args.angular_stimulus is not None
                    else "physical_world_panel"
                ),
                "stimulus_onset_s": args.stimulus_onset_s,
                "stimulus_speed_mm_s": args.stimulus_speed_mm_s,
                "stimulus_speed_effective_mm_s": ready_record.get("stimulus_speed_effective_mm_s"),
                "stimulus_velocity_virtualworld_mps": ready_record.get("stimulus_velocity_virtualworld_mps"),
                "stimulus_velocity_virtualworld_frame": ready_record.get("stimulus_velocity_frame"),
                "stimulus_velocity_flygym_source_mps": (
                    None if ready_record.get("stimulus_velocity_virtualworld_mps") is None
                    else (WORLD_REFLECTION @ np.asarray(
                        ready_record["stimulus_velocity_virtualworld_mps"], dtype=np.float64
                    )).tolist()
                ),
                "stimulus_velocity_flygym_source_frame": (
                    None if args.angular_stimulus is not None
                    else "FlyGym source frame (+X forward, +Y fly-left, +Z up)"
                ),
                "flygym_to_virtualworld_velocity_matrix": (
                    None if args.angular_stimulus is not None else WORLD_REFLECTION.tolist()
                ),
                "angular_stimulus_spec_path": (
                    str(args.angular_stimulus_spec) if args.angular_stimulus is not None else None
                ),
                "angular_stimulus_spec_sha256": args.angular_stimulus_spec_sha256,
                "angular_stimulus": ready_record.get("angular_stimulus"),
                "body_cpg_seed": args.seed,
                "neural_seed": args.neural_seed,
                "legacy_rest_compatibility": args.legacy_rest_compatibility,
                "cpg_frequency_hz": args.cpg_frequency_hz,
                "scene_eye_midpoint_origin_xy_m": list(scene_origin_xy),
                "max_vram_gib": MAX_VRAM_GIB,
                "worker_timeout_s": args.worker_timeout_s,
                "worker_shutdown_timeout_s": args.shutdown_timeout_s,
                "worker_max_frames": n_frames,
                "extended_duration_guard_enabled": args.allow_extended_duration,
                "deterministic_vision_requested": args.deterministic_vision,
                "deterministic_neural_delivery_requested": args.deterministic_neural_delivery,
                "pre_stimulus_zero_frames": pre_stimulus_zero_frames,
                "expected_pre_stimulus_zero_frames": round(args.stimulus_onset_s / FRAME_DT_S),
                "direct_worker_drive_matches_controller_and_native_trace": direct_drive_match_ticks,
                "expected_direct_drive_match_ticks": n_frames * CONTROL_TICKS_PER_FRAME,
            },
            "timing_model": {
                "worker_visual_neural_window_s": FRAME_DT_S,
                "body_control_tick_s": CONTROL_DT_S,
                "native_mujoco_step_s": NATIVE_DT_S,
                "controller_ticks_per_neural_window": CONTROL_TICKS_PER_FRAME,
                "native_steps_per_control_tick": NATIVE_STEPS_PER_CONTROL_TICK,
                "execution": "synchronous: send pose, wait for matching 5 ms neural drive, then advance five 1 ms body ticks",
                "worker_roundtrip_wall_s_total": worker_wait_wall_s,
                "body_step_wall_s_total": body_wall_s,
                "body_session_initialization_wall_s": body_initialization_wall_s,
                "worker_startup_wall_s": worker_startup_wall_s,
                "elapsed_wall_s_through_body_bundle_write": time.monotonic() - started_wall,
                "wall_time_includes_worker_startup_and_final_artifact_write": True,
                "no_interpolation_or_latest_drive_substitution": True,
            },
            "pose_mapping": {
                "source_flygym_world": "+X forward, +Y fly-left, +Z up; position and velocity in mm-based native coordinates",
                "virtual_world_scene": "+X forward, +Y raster-camera-right, +Z up",
                "world_reflection_matrix": WORLD_REFLECTION.tolist(),
                "orientation_mapping": "R_virtual=F*R_flygym*F; proper rotation preserving FlyVis head rays under renderer's head +Y to -camera_right mapping",
                "virtualworld_euler_basis": "Rz(yaw) Ry(-pitch) Rx(roll)",
                "angular_velocity_mapping": "axial vector det(F)*F*omega_flygym",
                "pose_position_anchor": "midpoint between the two official FlyGym eye camera origins, not thorax origin",
                "initial_xy_origin": "subtract initial eye-camera midpoint X/Y and add --scene-origin-x-m/--scene-origin-y-m; keep absolute converted Z",
                "head_orientation": "neutral c_head frame is fused to c_thorax; no independent head joint in this compiled model",
                "rigging_and_sensor_sources": rigging_sources,
                "eye_camera_baseline_m": eye_separation_m,
                "flyvis_eye_direction_mapping": "worker's fitted per-eye EYE_FIT is used; FlyGym vision.yaml camera Euler offsets are recorded but not applied a second time",
                "eye_origin_offsets_applied_by_worker": "VirtualWorld applies the measured baseline once as +/- eye_separation/2",
            },
            "behavior_policy": {
                "stimulus_condition_used_for_motor_branching": False,
                "autopilot": "prohibited",
                "state_drive": "prohibited",
                "stabilizer": "prohibited",
                "motor_commands": "actual MaleCNS decoded bilateral values passed unchanged to the frozen V4.1 engineered controller; gray-interval outputs are required to be exactly zero",
                "drive_hold": "exact worker output held for five successive 1 ms FlyGym control ticks; per-tick controller and native-trace equality recorded",
                "gait_semantics": "neural outputs modulate engineered CPG phase-speed/native-template locomotion; this is not direct MN-to-muscle actuation",
                "fixed_drive_or_stimulus_selected_command": False,
            },
            "body_run": runner_result,
            "v4_1_controller": controller_binding,
            "runtime_model_preflight": runtime_model_preflight,
            "worker_capture_artifacts": worker_artifacts,
            "frame_count": successful_frames,
            "control_tick_count": body_tick_count,
            "pre_stimulus_zero_frames": pre_stimulus_zero_frames,
            "direct_drive_match_ticks": direct_drive_match_ticks,
            "files": {},
        }
        for name in outputs:
            path = output_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"expected online-run artifact is missing: {path}")
            bridge_manifest["files"][name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        _write_json(output_dir / "online_bridge_manifest.json", bridge_manifest)
        print(json.dumps({
            "status": "complete",
            "run_id": run_id,
            "output_dir": str(output_dir),
            "frames": successful_frames,
            "control_ticks": body_tick_count,
            "summary": runner_result["summary"],
        }, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as exc:
        error = exc
        try:
            if frame_file is not None:
                frame_file.flush()
                frame_file.close()
                frame_file = None
            if tick_file is not None:
                tick_file.flush()
                tick_file.close()
                tick_file = None
        except OSError:
            pass
        if worker is not None:
            try:
                worker_shutdown = worker.stop(timeout_s=min(args.shutdown_timeout_s, args.worker_timeout_s))
            except BaseException as stop_exc:
                worker_shutdown = {
                    "error_type": type(stop_exc).__name__,
                    "error": str(stop_exc),
                    "return_code": worker.process.poll(),
                }
            worker = None
        failure = {
            "schema_version": "flygym-online-optomotor-failure-1",
            "protocol": PROTOCOL,
            "status": "failed",
            "failed_utc": _utc_now(),
            "run_id": run_id,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "last_completed_seq": last_seq,
            "successful_frames": successful_frames,
            "body_control_ticks": body_tick_count,
            "worker_max_frames": n_frames,
            "elapsed_wall_s": time.monotonic() - started_wall,
            "worker_command": worker_command,
            "worker_ready": ready_record,
            "worker_shutdown": worker_shutdown,
            "partial_body_state": _partial_state(session) if session is not None else None,
            "runtime_model_preflight": runtime_model_preflight,
            "pose_mapping_sources": rigging_sources,
            "notes": "Failure artifacts are partial evidence; no complete runner manifest is emitted.",
        }
        if session is not None:
            trace = _write_partial_native_trace(session, output_dir)
            failure["partial_native_trace"] = (
                {"path": trace.name, "bytes": trace.stat().st_size,
                 "sha256": _sha256_file(trace)} if trace is not None else None
            )
        _write_json(output_dir / "failure.json", failure)
        print(f"online optomotor failed: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1
    finally:
        if frame_file is not None:
            frame_file.close()
        if tick_file is not None:
            tick_file.close()
        if session is not None:
            session.close()
        # A successfully stopped worker was set to None above. This last guard
        # covers unexpected exceptions raised while writing the success bundle.
        if worker is not None:
            try:
                worker.stop(timeout_s=min(args.shutdown_timeout_s, args.worker_timeout_s))
            except BaseException:
                pass
        elapsed = time.monotonic() - started_wall
        if error is not None:
            print(f"online optomotor wall time: {elapsed:.3f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
