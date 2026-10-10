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
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import subprocess
import sys
import time
from typing import Any, TextIO

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BODY_RUNNER = PROJECT_ROOT / "scripts" / "run_flygym_reference_walk.py"
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "flygym_online_neural_worker.py"
WORKER_CONFIG = PROJECT_ROOT / "config" / "flygym_online_neural_worker.json"
BODY_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
DEFAULT_WORKER_PYTHON = Path(
    "/data/environments/projects/flyvis-opticlobe/bin/python"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/flygym-online"
)
ARTIFACT_CONTAINMENT_ROOT = Path("/data/artifacts/by-project/malecns-gpu")
PROTOCOL = "flygym-online-v1"
FRAME_DT_S = 0.005
CONTROL_DT_S = 0.001
NATIVE_DT_S = 0.0001
CONTROL_TICKS_PER_FRAME = 5
NATIVE_STEPS_PER_CONTROL_TICK = 10
MAX_VRAM_GIB = 8.0
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "FlyGym advances on CPU. The GPU neural worker must report FlyVis/CUDA "
            "and connectome/Triton at startup; backend fallback is rejected."
        ),
    )
    parser.add_argument("--duration-s", type=float, default=None,
                        help="simulated duration; defaults to 2 s in physical mode or spec end in angular mode")
    parser.add_argument("--allow-extended-duration", action="store_true",
                        help="allow a duration above 2 s, still capped at 10 s / 2,000 frames")
    parser.add_argument("--stimulus-condition", choices=("left", "right", "static", "angular"),
                        default="left", help="visual scene condition only")
    parser.add_argument("--angular-stimulus-spec", type=Path, default=None,
                        help="frozen angular-optomotor-stimulus-spec-1 JSON; requires --stimulus-condition angular")
    parser.add_argument("--stimulus-speed-mm-s", type=float, default=None,
                        help="optional panel-speed magnitude in mm/s; keeps the selected condition's configured direction")
    parser.add_argument("--stimulus-onset-s", type=float, default=None,
                        help="time when a moving stripe begins; static remains static")
    parser.add_argument("--seed", type=int, default=0,
                        help="FlyGym tripod CPG/controller seed")
    parser.add_argument("--neural-seed", type=int, default=1729,
                        help="FlyVis/MaleCNS worker seed")
    parser.add_argument("--legacy-rest-compatibility", action="store_true", default=False,
                        help="use the explicitly experimental legacy zero-spring/rest-state body factory")
    parser.add_argument("--cpg-frequency-hz", type=float, default=12.0,
                        help="FlyGym tripod CPG intrinsic frequency in Hz")
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON,
                        help="registered flyvis-opticlobe Python executable")
    parser.add_argument("--worker-timeout-s", type=float, default=120.0,
                        help="maximum wall time for worker startup or one 5 ms neural window")
    parser.add_argument("--shutdown-timeout-s", type=float, default=10.0,
                        help="bounded worker stop acknowledgement/exit wait (maximum 60 s)")
    parser.add_argument("--deterministic-vision", action="store_true",
                        help="request strict PyTorch determinism in the FlyVis worker")
    parser.add_argument("--deterministic-neural-delivery", action="store_true",
                        help="request atomic deterministic MaleCNS/Triton spike delivery")
    parser.add_argument("--experimental-swing-clearance", type=Path,
                        help="opt-in JSON for engineered flat-floor foot clearance; neural drives remain unchanged")
    parser.add_argument("--scene-origin-x-m", type=float, default=0.0,
                        help="VirtualWorld X location of the initial eye-camera midpoint")
    parser.add_argument("--scene-origin-y-m", type=float, default=0.0,
                        help="VirtualWorld Y location of the initial eye-camera midpoint")
    parser.add_argument("--out", type=Path,
                        help="new run directory; defaults below the online engineering artifacts")
    parser.add_argument("--run-id", help="single-component run identifier")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.angular_stimulus_spec is not None:
        args.angular_stimulus_spec = _project_path(args.angular_stimulus_spec).resolve()
        if not args.angular_stimulus_spec.is_file():
            parser.error(f"angular stimulus spec is missing: {args.angular_stimulus_spec}")
        if args.stimulus_condition != "angular":
            parser.error("--angular-stimulus-spec requires --stimulus-condition angular")
        try:
            (args.angular_stimulus, args.angular_stimulus_spec_document,
             args.angular_stimulus_spec_raw_bytes) = _load_angular_spec(args.angular_stimulus_spec)
        except ValueError as exc:
            parser.error(str(exc))
        args.angular_stimulus_spec_sha256 = hashlib.sha256(
            args.angular_stimulus_spec_raw_bytes
        ).hexdigest()
        if args.stimulus_onset_s is None:
            args.stimulus_onset_s = args.angular_stimulus.onset_s
        elif not math.isclose(args.stimulus_onset_s, args.angular_stimulus.onset_s,
                              rel_tol=0.0, abs_tol=1e-9):
            parser.error("--stimulus-onset-s must match the angular stimulus spec")
        if args.duration_s is None:
            args.duration_s = args.angular_stimulus.end_s
        elif args.duration_s + 1e-9 < args.angular_stimulus.end_s:
            parser.error(
                "--duration-s ends before the angular spec's full moving interval; partial calibration is refused"
            )
        if args.stimulus_speed_mm_s is not None:
            parser.error("--stimulus-speed-mm-s applies only to physical stripe conditions")
    else:
        if args.stimulus_condition == "angular":
            parser.error("--stimulus-condition angular requires --angular-stimulus-spec")
        args.angular_stimulus = None
        args.angular_stimulus_spec_document = None
        args.angular_stimulus_spec_raw_bytes = None
        args.angular_stimulus_spec_sha256 = None
        if args.duration_s is None:
            args.duration_s = DEFAULT_MAX_DURATION_S
        if args.stimulus_onset_s is None:
            args.stimulus_onset_s = 1.0
    for name, value in (("--duration-s", args.duration_s),
                        ("--stimulus-onset-s", args.stimulus_onset_s),
                        ("--worker-timeout-s", args.worker_timeout_s),
                        ("--shutdown-timeout-s", args.shutdown_timeout_s),
                        ("--scene-origin-x-m", args.scene_origin_x_m),
                        ("--scene-origin-y-m", args.scene_origin_y_m),
                        ("--cpg-frequency-hz", args.cpg_frequency_hz)):
        if not math.isfinite(value):
            parser.error(f"{name} must be finite")
    if args.duration_s <= 0 or args.duration_s > MAX_BOUNDED_DURATION_S:
        parser.error(
            f"--duration-s must be >0 and <={MAX_BOUNDED_DURATION_S:g} for this bounded online trial"
        )
    if args.duration_s > DEFAULT_MAX_DURATION_S and not args.allow_extended_duration:
        parser.error(
            "--duration-s above 2.0 requires --allow-extended-duration "
            f"(hard ceiling {MAX_BOUNDED_DURATION_S:g} s)"
        )
    if args.stimulus_onset_s < 0 or args.stimulus_onset_s >= args.duration_s:
        parser.error("--stimulus-onset-s must fall inside the simulated duration")
    if args.stimulus_onset_s < 1.0:
        parser.error("--stimulus-onset-s must be >=1.0 s to preserve the static visual warmup")
    if args.stimulus_speed_mm_s is not None:
        if not math.isfinite(args.stimulus_speed_mm_s) or args.stimulus_speed_mm_s < 0.0:
            parser.error("--stimulus-speed-mm-s must be finite and non-negative")
        if args.stimulus_condition == "static" and args.stimulus_speed_mm_s != 0.0:
            parser.error("static condition requires --stimulus-speed-mm-s 0 when overriding speed")
    if args.worker_timeout_s <= 0:
        parser.error("--worker-timeout-s must be positive")
    if not 0 < args.shutdown_timeout_s <= MAX_SHUTDOWN_TIMEOUT_S:
        parser.error(
            f"--shutdown-timeout-s must be in (0, {MAX_SHUTDOWN_TIMEOUT_S:g}]"
        )
    if args.seed < 0 or args.neural_seed < 0:
        parser.error("--seed and --neural-seed must be non-negative")
    if args.cpg_frequency_hz <= 0:
        parser.error("--cpg-frequency-hz must be positive")
    for name, value in (("duration", args.duration_s), ("stimulus onset", args.stimulus_onset_s)):
        units = value / FRAME_DT_S
        if not math.isclose(units, round(units), rel_tol=0.0, abs_tol=1e-8):
            parser.error(f"{name} must align to the 5 ms frame cadence")
    frame_count = int(round(args.duration_s / FRAME_DT_S))
    if not 1 <= frame_count <= MAX_WORKER_CAPTURE_FRAMES:
        parser.error(
            f"the requested duration requires {frame_count} frames; worker capture limit is "
            f"{MAX_WORKER_CAPTURE_FRAMES}"
        )
    args.frame_count = frame_count
    worker_python = args.worker_python.expanduser().absolute()
    if not worker_python.is_file():
        parser.error(f"worker Python is unavailable: {worker_python}")
    if not BODY_RUNNER.is_file():
        parser.error(f"FlyGym reference runner is missing: {BODY_RUNNER}")
    if not WORKER_SCRIPT.is_file() or not WORKER_CONFIG.is_file():
        parser.error("online neural worker script/config are not installed in this checkout")
    args.worker_python = worker_python
    if args.experimental_swing_clearance is not None:
        path = _project_path(args.experimental_swing_clearance).resolve()
        if not path.is_file():
            parser.error(f"swing clearance configuration is missing: {path}")
        args.experimental_swing_clearance = path


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
    _validate_args(parser, args)
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

    run_id = args.run_id or _run_id(args.stimulus_condition, args.seed, args.neural_seed)
    if not run_id.strip() or Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("--run-id must be one nonempty path component")
    output_dir = (_project_path(args.out).resolve() if args.out is not None
                  else (DEFAULT_OUTPUT_ROOT / run_id).resolve())
    containment_root = ARTIFACT_CONTAINMENT_ROOT.resolve()
    if output_dir == containment_root or not output_dir.is_relative_to(containment_root):
        parser.error(f"--out must be a new run directory below {containment_root}")
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
    gait_adapter_record: dict[str, Any] | None = None
    body_tick_count = 0
    body_wall_s = 0.0
    worker_wait_wall_s = 0.0
    body_initialization_wall_s = 0.0
    worker_startup_wall_s = 0.0

    try:
        body_init_start = time.monotonic()
        body_config = WalkConfig(
            duration_s=float(args.duration_s),
            seed=int(args.seed),
            legacy_rest_compatibility=bool(args.legacy_rest_compatibility),
            cpg_frequency_hz=float(args.cpg_frequency_hz),
        )
        session = FlyGymReferenceSession(
            duration_s=float(args.duration_s),
            seed=int(args.seed),
            render_video=False,
            config=body_config,
            run_id=run_id,
        )
        if args.experimental_swing_clearance is not None:
            from flygym_swing_clearance import SwingClearanceController
            gait_path = args.experimental_swing_clearance
            gait_config = json.loads(gait_path.read_text(encoding="utf-8"))
            if gait_config.get("schema_version") != "flygym-swing-clearance-1":
                raise ValueError("unsupported swing clearance configuration schema")
            session.controller = SwingClearanceController(
                session,
                gait_config["stance_marker_z_mm"],
                clearance_mm=float(gait_config["clearance_mm"]),
                legs_to_lift=gait_config.get("legs_to_lift"),
                damping=float(gait_config.get("damping", 0.03)),
                iterations=int(gait_config.get("iterations", 6)),
            )
            gait_adapter_record = {
                "classification": "engineered_flat_floor_position_target_correction",
                "config_path": str(gait_path),
                "config_sha256": _sha256_file(gait_path),
                "config": gait_config,
                "adapter_source_sha256": _sha256_file(
                    PROJECT_ROOT / "scripts" / "flygym_swing_clearance.py"),
                "neural_drive_transformation": "none",
                "stimulus_condition_used": False,
                "live_root_state_writes": False,
                "scope": "Mechanical gait aid; no claim of neural generation of gait",
            }
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
                "worker_roundtrip_wall_s": worker_elapsed,
                "body_advance_follows_reply": True,
            }
            frame_file.write(json.dumps(frame_row, sort_keys=True, allow_nan=False) + "\n")
            frame_file.flush()

            # No warm-up gate is applied to motor output: transient commands
            # from the actual neural decoder are used from the first window.
            for tick_in_frame in range(CONTROL_TICKS_PER_FRAME):
                tick_start_s = session.time_s
                expected_start = frame_time_s + tick_in_frame * CONTROL_DT_S
                if not math.isclose(tick_start_s, expected_start,
                                    rel_tol=0.0, abs_tol=1e-9):
                    raise RuntimeError("body control tick split out of sync with neural window")
                step_wall_start = time.monotonic()
                step = session.step_control_tick([left, right])
                step_elapsed = time.monotonic() - step_wall_start
                body_wall_s += step_elapsed
                body_tick_count += 1
                if int(step["native_samples_added"]) != NATIVE_STEPS_PER_CONTROL_TICK:
                    raise RuntimeError("FlyGym control tick did not advance ten native MuJoCo steps")
                expected_end = expected_start + CONTROL_DT_S
                if not math.isclose(session.time_s, expected_end,
                                    rel_tol=0.0, abs_tol=1e-9):
                    raise RuntimeError("FlyGym control tick ended outside its 1 ms window")
                sample = int(session._native_samples - 1)
                tick = {
                    "seq": seq,
                    "tick_in_frame": tick_in_frame,
                    "control_window_start_s": expected_start,
                    "control_window_end_s": expected_end,
                    "neural_reply_sha256": reply_hash,
                    "descending_drive_left_right": [left, right],
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
                if gait_adapter_record is not None:
                    tick["engineered_swing_clearance"] = session.controller.last_clearance_info
                tick_file.write(json.dumps(tick, sort_keys=True, allow_nan=False) + "\n")
            tick_file.flush()
            successful_frames += 1
            last_seq = seq
            if not args.quiet_progress and (seq % 100 == 99 or seq == n_frames - 1):
                print(f"online optomotor progress: t={session.time_s:.3f}s / {args.duration_s:.3f}s "
                      f"frames={seq + 1}/{n_frames}", file=sys.stderr, flush=True)

        if not math.isclose(session.time_s, args.duration_s, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("body duration differs from the requested online duration")
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
        ]
        if gait_adapter_record is not None:
            _write_json(output_dir / "gait_adapter_manifest.json", gait_adapter_record)
            outputs.append("gait_adapter_manifest.json")
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
            "worker_shutdown": worker_shutdown,
            "worker_artifact_directory": str(worker_output),
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
                "motor_commands": "actual MaleCNS decoded bilateral values used from the first window, including initial visual transient",
                "drive_hold": "exact worker output held for five successive 1 ms FlyGym control ticks",
            },
            "body_run": runner_result,
            "engineered_gait_adapter": gait_adapter_record,
            "frame_count": successful_frames,
            "control_tick_count": body_tick_count,
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
