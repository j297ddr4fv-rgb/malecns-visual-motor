#!/usr/bin/env python3
"""Run an auditable native FlyGym v2.1 flat-ground walking reference.

The runner uses the upstream NeuroMechFly position actuators, tripod CPG,
preprogrammed steps, adhesion actuator and HybridTurningController.  It advances
MuJoCo at the model's native timestep and never assigns to qpos/qvel after the
initial FlyGym keyframe reset.  The two descending-drive channels are explicit
inputs; no visual stimulus labels are interpreted here.

Run from the project root with the dedicated runtime, for example:

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
      CUDA_VISIBLE_DEVICES='' FLYGYM_ASSET_CACHE_DIR=/data/models/flygym/assets \
      /data/environments/projects/flygym-reference/bin/python \
      scripts/run_flygym_reference_walk.py --duration-s 2 --seed 0 \
      --out /data/artifacts/by-project/malecns-gpu/engineering/flygym-reference-20261005-v1

The module API can also be used by another controller without constructing a
second simulation: instantiate :class:`FlyGymReferenceSession`, then call
``step_control_tick([left_drive, right_drive])`` for each 1 ms control update.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Callable, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path("/home/xvhang/Projects/external-references/flygym-v2.1.0")
REFERENCE_COMMIT = "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
DEFAULT_OUTPUT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-reference-20261005-v1/walk"
)
NATIVE_DT_S = 1e-4
CONTROL_DT_S = 1e-3
FLY_NAME = "nmf"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
GRAVITY_MM_S2 = 9810.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _native_name(mj, model, obj_type, obj_id: int) -> str:
    return mj.mj_id2name(model, obj_type, int(obj_id)) or f"unnamed-{obj_id}"


def read_drive_schedule(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV with ``time_s,left_descending_drive,right_descending_drive``."""
    rows: list[tuple[float, float, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        needed = {
            "time_s",
            "left_descending_drive",
            "right_descending_drive",
        }
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError(
                "drive CSV must have columns: time_s,left_descending_drive,"
                "right_descending_drive"
            )
        for row in reader:
            rows.append(
                (
                    float(row["time_s"]),
                    float(row["left_descending_drive"]),
                    float(row["right_descending_drive"]),
                )
            )
    if not rows:
        raise ValueError("drive CSV contains no samples")
    arr = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("drive CSV values must be finite")
    if np.any(np.diff(arr[:, 0]) <= 0):
        raise ValueError("drive CSV time_s values must increase strictly")
    if np.any(np.abs(arr[:, 1:]) > 1):
        raise ValueError("descending drive values must be in [-1, 1]")
    return arr[:, 0], arr[:, 1:]


def scheduled_drive_provider(
    times_s: np.ndarray, values: np.ndarray
) -> Callable[[float, object], np.ndarray]:
    """Make a side-drive callback using linear interpolation in its time series."""
    times_s = np.asarray(times_s, dtype=np.float64).copy()
    values = np.asarray(values, dtype=np.float64).copy()
    if times_s.ndim != 1 or values.shape != (len(times_s), 2) or len(times_s) == 0:
        raise ValueError("expected times (n,) and values (n,2)")

    def provider(time_s: float, _observation: object) -> np.ndarray:
        return np.array(
            [np.interp(time_s, times_s, values[:, col]) for col in range(2)],
            dtype=np.float64,
        )

    return provider


@dataclass(frozen=True)
class WalkConfig:
    duration_s: float = 2.0
    seed: int = 0
    native_timestep_s: float = NATIVE_DT_S
    control_timestep_s: float = CONTROL_DT_S
    position_actuator_gain_native: float = 45.0
    position_actuator_forcerange_native: tuple[float, float] = (-65.0, 65.0)
    joint_stiffness_native: float = 0.05
    joint_damping_native: float = 0.06
    passive_tarsus_stiffness_native: float = 7.5
    passive_tarsus_damping_native: float = 1e-2
    legacy_rest_compatibility: bool = False
    adhesion_enabled: bool = True
    adhesion_gain_native: float = 40.0
    cpg_frequency_hz: float = 12.0
    cpg_amplitude: float = 1.0
    cpg_coupling_strength: float = 10.0
    cpg_convergence_coefficient: float = 20.0
    spawn_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.7)
    spawn_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    camera_resolution_hw: tuple[int, int] = (368, 640)
    video_fps: int = 30
    video_playback_speed: float = 1.0


class FlyGymReferenceSession:
    """Stateful upstream walking simulation for a live external drive loop.

    A caller can inspect ``observation()`` and send independent left/right
    descending values on every 1 ms controller update.  The physics and trace
    capture continue at 100 microseconds.  No constructor or import side effects
    occur until a session object is explicitly created.
    """

    def __init__(
        self,
        duration_s: float = 2.0,
        *,
        seed: int = 0,
        render_video: bool = True,
        config: WalkConfig | None = None,
        run_id: str = "flygym-reference-flatground-cpg-tripod-seed0-20261005-v1",
        capture_mode: str = "full",
        capture_stride: int = 50,
    ) -> None:
        if capture_mode not in {"full", "light"}:
            raise ValueError("capture_mode must be 'full' or 'light'")
        if isinstance(capture_stride, bool) or int(capture_stride) != capture_stride or capture_stride < 1:
            raise ValueError("capture_stride must be a positive integer")
        self.capture_mode = capture_mode
        self.capture_stride = int(capture_stride)
        self.config = config or WalkConfig(duration_s=duration_s, seed=seed)
        if not math.isclose(self.config.native_timestep_s, NATIVE_DT_S, rel_tol=0, abs_tol=1e-12):
            raise ValueError("this reference requires the pinned native 100 µs timestep")
        if not math.isclose(self.config.control_timestep_s / self.config.native_timestep_s,
                            round(self.config.control_timestep_s / self.config.native_timestep_s),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("control timestep must be an integer multiple of native timestep")
        if self.config.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.duration_s = self.config.duration_s
        n_native = round(self.duration_s / self.config.native_timestep_s)
        if not math.isclose(n_native * self.config.native_timestep_s, self.duration_s,
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError("duration_s must be an integer multiple of the native timestep")
        n_control = round(self.duration_s / self.config.control_timestep_s)
        if not math.isclose(n_control * self.config.control_timestep_s, self.duration_s,
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError("duration_s must be an integer multiple of the control timestep")
        self.run_id = run_id
        self.render_video = render_video and capture_mode == "full"
        self._closed = False
        self._native_samples = 0
        self._control_samples = 0
        self._root_qpos_write_count = 0
        self._root_qvel_write_count = 0
        self._physics_step_s = 0.0
        self._capture_s = 0.0

        # Keep package and rendering caches away from home/repository paths.
        os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", "/data/models/flygym/assets")
        os.environ.setdefault("MPLCONFIGDIR", "/data/cache/flygym/matplotlib")
        # Fail closed to software rendering and keep CUDA devices invisible.
        # Set these before importing MuJoCo, since its GL context backend is
        # selected at import time.  The video writer below explicitly uses
        # libx264, a software encoder.
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        import mujoco as mj
        from flygym.compose import FlatGroundWorld
        from flygym.simulation import Simulation
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import (
            HybridControllerObservation,
            HybridTurningController,
            PreprogrammedSteps,
            apply_locomotion_action,
            make_locomotion_fly,
            make_tripod_cpg_network,
        )

        self.mj = mj
        self.HybridControllerObservation = HybridControllerObservation
        self.apply_locomotion_action = apply_locomotion_action
        fly_factory = make_locomotion_fly
        if self.config.legacy_rest_compatibility:
            from flygym_legacy_rest_compat import make_legacy_rest_locomotion_fly
            fly_factory = make_legacy_rest_locomotion_fly
        self.fly = fly_factory(
            name=FLY_NAME,
            joint_stiffness=self.config.joint_stiffness_native,
            joint_damping=self.config.joint_damping_native,
            passive_tarsus_stiffness=self.config.passive_tarsus_stiffness_native,
            passive_tarsus_damping=self.config.passive_tarsus_damping_native,
            actuator_gain=self.config.position_actuator_gain_native,
            actuator_forcerange=self.config.position_actuator_forcerange_native,
            add_adhesion=self.config.adhesion_enabled,
            adhesion_gain=self.config.adhesion_gain_native,
            colorize=True,
        )
        self.tracking_camera = self.fly.add_tracking_camera()
        self.world = FlatGroundWorld()
        self.world.add_fly(
            self.fly,
            self.config.spawn_position_mm,
            Rotation3D(format="quat", values=self.config.spawn_quaternion_wxyz),
        )
        self.sim = Simulation(self.world, timestep=self.config.native_timestep_s)
        self.model = self.sim.mj_model
        self.data = self.sim.mj_data
        # Simulation.__init__ resets the native keyframe.  Populate derived
        # kinematics and contact fields before the first observation without
        # changing qpos/qvel or advancing time.
        mj.mj_forward(self.model, self.data)
        if not math.isclose(float(self.model.opt.timestep), self.config.native_timestep_s,
                            rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"compiled MuJoCo timestep mismatch: {self.model.opt.timestep}")

        cpg = make_tripod_cpg_network(
            timestep=self.config.control_timestep_s,
            intrinsic_frequency=self.config.cpg_frequency_hz,
            intrinsic_amplitude=self.config.cpg_amplitude,
            coupling_strength=self.config.cpg_coupling_strength,
            convergence_coef=self.config.cpg_convergence_coefficient,
            seed=self.config.seed,
        )
        self.preprogrammed_steps = PreprogrammedSteps()
        self.controller = HybridTurningController(
            timestep=self.config.control_timestep_s,
            cpg_network=cpg,
            preprogrammed_steps=self.preprogrammed_steps,
            enable_adhesion=self.config.adhesion_enabled,
        )

        all_body_segments = list(self.fly.get_bodysegs_order())
        all_body_names = [segment.name for segment in all_body_segments]
        all_body_ids = np.asarray(
            self.sim._internal_bodyids_by_fly[FLY_NAME], dtype=np.int32
        )
        if len(all_body_segments) != len(all_body_ids):
            raise RuntimeError("FlyGym body segment and compiled body ID orders differ")
        self.uncompiled_body_segment_aliases = [
            {"segment_name": name, "compiled_body_id": int(body_id)}
            for name, body_id in zip(all_body_names, all_body_ids)
            if body_id < 0 or body_id >= self.model.nbody
        ]
        valid = all_body_ids >= 0
        self.body_segments = [
            segment for segment, keep in zip(all_body_segments, valid) if keep
        ]
        self.body_names = [name for name, keep in zip(all_body_names, valid) if keep]
        self.body_ids = all_body_ids[valid]
        expected_compiled_fly_body_ids = np.arange(1, self.model.nbody, dtype=np.int32)
        if (
            len(np.unique(self.body_ids)) != len(self.body_ids)
            or not np.array_equal(np.sort(self.body_ids), expected_compiled_fly_body_ids)
        ):
            raise RuntimeError(
                "FlyGym body telemetry must cover each compiled non-world body exactly once"
            )
        self.body_mass = self.model.body_mass[self.body_ids].copy()
        self.body_masses_are_positive = self.body_mass > 0
        thorax_segment = type(self.fly).BODY_SEGMENT_CLASS("c_thorax")
        self.thorax_index = self.body_segments.index(thorax_segment)
        self.thorax_body_id = int(self.body_ids[self.thorax_index])
        self.foot_body_ids = np.asarray(
            [self.body_ids[self.body_names.index(f"{leg}_tarsus5")] for leg in LEGS],
            dtype=np.int32,
        )
        self.position_actuator_ids = np.asarray(
            self.sim._intern_actuatorids_by_type_by_fly[
                __import__("flygym.compose", fromlist=["ActuatorType"]).ActuatorType.POSITION
            ][FLY_NAME],
            dtype=np.int32,
        )
        if self.config.adhesion_enabled:
            self.adhesion_actuator_ids = np.asarray(
                self.sim._intern_adhesionactuatorids_by_fly[FLY_NAME], dtype=np.int32
            )
        else:
            self.adhesion_actuator_ids = np.empty(0, dtype=np.int32)
        if len(self.adhesion_actuator_ids) not in (0, 6):
            raise RuntimeError("expected six adhesion actuator channels")
        self.joint_angles_ids = np.asarray(self.sim._intern_qposadrs_by_fly[FLY_NAME], dtype=np.int32)
        self.joint_vel_ids = np.asarray(self.sim._intern_qveladrs_by_fly[FLY_NAME], dtype=np.int32)
        self.root_joint_id = self._find_root_free_joint()
        self.root_qpos_adr = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_qvel_adr = int(self.model.jnt_dofadr[self.root_joint_id])
        self.ground_geom_ids = np.asarray(self.sim._internal_ground_geom_ids, dtype=np.int32)
        if len(self.ground_geom_ids) != 1:
            raise RuntimeError(f"expected one flat-ground geom, got {len(self.ground_geom_ids)}")
        self.ground_geom_id = int(self.ground_geom_ids[0])

        self.geom_names = [
            _native_name(mj, self.model, mj.mjtObj.mjOBJ_GEOM, i)
            for i in range(self.model.ngeom)
        ]
        body_id_to_segment = {int(bid): name for bid, name in zip(self.body_ids, self.body_names)}
        self.fly_geom_ids = np.asarray(
            [i for i, bid in enumerate(self.model.geom_bodyid) if int(bid) in body_id_to_segment],
            dtype=np.int32,
        )
        self.geom_id_to_leg: dict[int, int] = {}
        self.geom_id_to_segment: dict[int, str] = {}
        for geom_id in self.fly_geom_ids:
            body_name = body_id_to_segment[int(self.model.geom_bodyid[geom_id])]
            self.geom_id_to_segment[int(geom_id)] = body_name
            if body_name[:2] in LEGS:
                self.geom_id_to_leg[int(geom_id)] = LEGS.index(body_name[:2])
        # FlyGym deliberately disables automatic collision masks and declares
        # allowed pairs explicitly.  Determine the ground-contact geometry from
        # those compiled pairs; retain all thorax geometry as a fallback only
        # if upstream has no thorax-ground pair, so geometric floor clearance is
        # still visible without silently changing the model.
        pair_ground_geoms: set[int] = set()
        for pair_id in range(self.model.npair):
            p1 = int(self.model.pair_geom1[pair_id])
            p2 = int(self.model.pair_geom2[pair_id])
            if p1 == self.ground_geom_id:
                pair_ground_geoms.add(p2)
            elif p2 == self.ground_geom_id:
                pair_ground_geoms.add(p1)
        thorax_all_geom_ids = [
            int(i)
            for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) == self.thorax_body_id
        ]
        self.thorax_collision_geom_ids = np.asarray(
            [i for i in thorax_all_geom_ids if i in pair_ground_geoms], dtype=np.int32
        )
        self.thorax_clearance_uses_fallback = len(self.thorax_collision_geom_ids) == 0
        if self.thorax_clearance_uses_fallback:
            self.thorax_collision_geom_ids = np.asarray(thorax_all_geom_ids, dtype=np.int32)
        self.thorax_collision_geom_names = [
            self.geom_names[i] for i in self.thorax_collision_geom_ids
        ]
        self.ground_paired_geom_ids = sorted(pair_ground_geoms)
        if len(self.thorax_collision_geom_ids) == 0:
            raise RuntimeError("no compiled collision geom found on the thorax")

        if self.capture_mode == "light":
            self._initialize_light_capture(n_native)
        else:
            self.position_control_by_sample = np.zeros((n_native + 1, len(self.position_actuator_ids)))
            self.position_force_by_sample = np.zeros_like(self.position_control_by_sample)
            self.adhesion_control_by_sample = np.zeros((n_native + 1, 6))
            self.adhesion_force_by_sample = np.zeros_like(self.adhesion_control_by_sample)
            self.cpg_phase_by_sample = np.zeros((n_native + 1, 6))
            self.cpg_magnitude_by_sample = np.zeros((n_native + 1, 6))
            self.descending_drive_by_sample = np.zeros((n_native + 1, 2))
            self.leg_stance_command_by_sample = np.zeros((n_native + 1, 6), dtype=np.uint8)
            self.root_qpos_write_count_by_sample = np.zeros(n_native + 1, dtype=np.int32)
            self.root_qvel_write_count_by_sample = np.zeros(n_native + 1, dtype=np.int32)
            self.root_external_wrench_by_sample = np.zeros((n_native + 1, 6))
            self.nonleg_ground_contact_by_sample = np.zeros(n_native + 1, dtype=np.uint8)
            self.thorax_ground_contact_by_sample = np.zeros(n_native + 1, dtype=np.uint8)
            self.all_arrays = self._allocate_state_arrays(n_native + 1)
            self.contact_step: list[int] = []
            self.contact_geom1: list[int] = []
            self.contact_geom2: list[int] = []
            self.contact_position: list[np.ndarray] = []
            self.contact_normal: list[np.ndarray] = []
            self.contact_force_on_geom2: list[np.ndarray] = []
            self.contact_force_on_fly: list[np.ndarray] = []
            self.contact_is_ground: list[int] = []
            self.contact_fly_segment: list[str] = []
            self.contact_leg_index: list[int] = []

        if render_video:
            self.sim.set_renderer(
                self.tracking_camera,
                camera_res=self.config.camera_resolution_hw,
                playback_speed=self.config.video_playback_speed,
                output_fps=self.config.video_fps,
                buffer_frames=True,
            )

        self.upstream_source_hashes = self._collect_source_hashes()
        if self.capture_mode == "full":
            self.compiled_mjcf_xml = self.world.mjcf_root.to_xml()
            self.compiled_mjcf_sha256 = _sha256_bytes(self.compiled_mjcf_xml.encode("utf-8"))
        else:
            self.compiled_mjcf_xml = None
            self.compiled_mjcf_sha256 = None
        self.effective_model_options = self._model_options()
        self._record_initial = False
        self.last_action = None

    @property
    def time_s(self) -> float:
        return float(self.data.time)

    def _find_root_free_joint(self) -> int:
        mj = self.mj
        suffix = FLY_NAME
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_FREE:
                continue
            name = _native_name(mj, self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            if name == suffix or name.endswith("/" + suffix):
                return joint_id
        raise RuntimeError("could not find free root joint in compiled MuJoCo model")

    def _allocate_state_arrays(self, n: int) -> dict[str, np.ndarray]:
        return {
            "time_s": np.zeros(n, dtype=np.float64),
            "qpos": np.zeros((n, self.model.nq), dtype=np.float64),
            "qvel": np.zeros((n, self.model.nv), dtype=np.float64),
            "joint_angle_rad": np.zeros((n, len(self.joint_angles_ids)), dtype=np.float64),
            "joint_velocity_rad_s": np.zeros((n, len(self.joint_vel_ids)), dtype=np.float64),
            "body_xpos_mm": np.zeros((n, len(self.body_ids), 3), dtype=np.float32),
            "body_xquat_wxyz": np.zeros((n, len(self.body_ids), 4), dtype=np.float32),
            "body_xvel_mm_s": np.zeros((n, len(self.body_ids), 3), dtype=np.float32),
            "body_xangvel_rad_s": np.zeros((n, len(self.body_ids), 3), dtype=np.float32),
            "body_xipos_mm": np.zeros((n, len(self.body_ids), 3), dtype=np.float32),
            "body_mass_native": self.body_mass.copy(),
            "body_xfrc_applied_native": np.zeros((n, len(self.body_ids), 6), dtype=np.float64),
            "qfrc_applied_native": np.zeros((n, self.model.nv), dtype=np.float64),
            "com_position_mm": np.zeros((n, 3), dtype=np.float64),
            "com_velocity_mm_s": np.zeros((n, 3), dtype=np.float64),
            "thorax_heading_xy": np.zeros((n, 2), dtype=np.float32),
            "thorax_heading_yaw_rad": np.zeros(n, dtype=np.float64),
            "tarsus5_xpos_mm": np.zeros((n, 6, 3), dtype=np.float32),
            "tarsus5_z_mm": np.zeros((n, 6), dtype=np.float32),
            "foot_contact_found": np.zeros((n, 6), dtype=np.uint8),
            "foot_contact_force_local_native": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_torque_local_native": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_pos_mm": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_normal_world": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_tangent_world": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_fz_up_native": np.zeros((n, 6), dtype=np.float32),
            "thorax_collision_geom_clearance_mm": np.zeros((n, len(self.thorax_collision_geom_ids)), dtype=np.float32),
            "thorax_collision_clearance_min_mm": np.zeros(n, dtype=np.float32),
        }

    def _initialize_light_capture(self, n_native_steps: int) -> None:
        """Allocate only stride samples and the 1 kHz contact/control trace."""
        n_stride = (n_native_steps + self.capture_stride - 1) // self.capture_stride + 1
        n_control = round(self.duration_s / self.config.control_timestep_s)
        self._light_last_native_step = n_native_steps
        self._light_sample_count = 0
        self._light_control_count = 0
        self._light_native_step, self._light_time_s = np.zeros(n_stride, dtype=np.int64), np.zeros(n_stride)
        self._light_thorax_position_mm = np.zeros((n_stride, 3), dtype=np.float32)
        self._light_thorax_quat_wxyz = np.zeros((n_stride, 4), dtype=np.float32)
        self._light_thorax_yaw_rad, self._light_com_position_mm = np.zeros(n_stride, dtype=np.float32), np.zeros((n_stride, 3), dtype=np.float32)
        self._light_thorax_heading_xy = np.zeros((n_stride, 2), dtype=np.float32)
        self._light_joint_angle_rad = np.zeros((n_stride, len(self.joint_angles_ids)), dtype=np.float32)
        self._light_foot_position_mm = np.zeros((n_stride, 6, 3), dtype=np.float32)
        self._light_cpg_phase_rad, self._light_cpg_magnitude = np.zeros((n_stride, 6), dtype=np.float32), np.zeros((n_stride, 6), dtype=np.float32)
        self._light_position_ctrl_rad = np.zeros((n_stride, len(self.position_actuator_ids)), dtype=np.float32)
        self._light_adhesion_ctrl = np.zeros((n_stride, 6), dtype=np.float32)
        self._light_descending_drive = np.zeros((n_stride, 2), dtype=np.float32)
        self._light_control_native_step = np.zeros(n_control, dtype=np.int64)
        self._light_control_time_s = np.zeros(n_control, dtype=np.float64)
        self._light_control_foot_fz = np.zeros((n_control, 6), dtype=np.float32)
        self._light_control_stance = np.zeros((n_control, 6), dtype=np.uint8)
        # Reuse one current and one previous contact vector at every native
        # step.  The two buffers stay distinct so transition detection always
        # compares this sample with the preceding sample.
        self._light_current_contact_state = np.zeros(6, dtype=bool)
        self._light_previous_contact_state = np.zeros(6, dtype=bool)
        self._light_contact_force_scratch = np.zeros(6, dtype=np.float64)
        self._light_per_leg_fz_scratch = np.zeros(6, dtype=np.float64)
        self._light_open_contact_start = [None] * 6
        self._light_contact_events: list[tuple[int, int, int]] = []
        self._light_last_foot_fz = np.zeros(6, dtype=np.float64)
        self._light_per_leg_contact_record_count = np.zeros(6, dtype=np.int64)
        self._light_foot_fz_min = np.full(6, np.inf, dtype=np.float64)
        self._light_foot_fz_max = np.full(6, -np.inf, dtype=np.float64)
        self._light_contact_record_count = 0
        self._light_thorax_ground_contact_samples = 0
        self._light_nonleg_ground_contact_samples = 0
        self._light_thorax_z_min_mm = math.inf
        self._light_com_z_min_mm = math.inf
        self._light_thorax_clearance_min_mm = math.inf
        self._light_root_wrench_peak = 0.0
        self._light_body_wrench_peak = 0.0
        self._light_qfrc_peak = 0.0
        self._light_root_qpos_write_min = self._light_root_qpos_write_max = self._root_qpos_write_count
        self._light_root_qvel_write_min = self._light_root_qvel_write_max = self._root_qvel_write_count

    def _record_light_native_sample(self, signal: np.ndarray) -> None:
        """Capture native contact transitions and scalar extrema without full-body telemetry."""
        k = self._native_samples
        if k > self._light_last_native_step:
            raise RuntimeError("duration buffer exhausted")
        data, model, mj = self.data, self.model, self.mj
        thorax_pos = data.xpos[self.thorax_body_id]
        self._light_thorax_z_min_mm = min(self._light_thorax_z_min_mm, float(thorax_pos[2]))
        root_wrench = np.asarray(data.xfrc_applied[self.thorax_body_id])
        root_norm = float(np.linalg.norm(root_wrench))
        self._light_root_wrench_peak = max(self._light_root_wrench_peak, root_norm)
        body_wrenches = np.asarray(data.xfrc_applied[1:])
        if body_wrenches.size:
            body_norms = np.linalg.norm(body_wrenches, axis=1)
            self._light_body_wrench_peak = max(self._light_body_wrench_peak, float(body_norms.max()))
        applied = np.asarray(data.qfrc_applied)
        if applied.size:
            self._light_qfrc_peak = max(self._light_qfrc_peak, float(np.max(np.abs(applied))))
        self._light_root_qpos_write_min = min(self._light_root_qpos_write_min, self._root_qpos_write_count); self._light_root_qpos_write_max = max(self._light_root_qpos_write_max, self._root_qpos_write_count)
        self._light_root_qvel_write_min = min(self._light_root_qvel_write_min, self._root_qvel_write_count); self._light_root_qvel_write_max = max(self._light_root_qvel_write_max, self._root_qvel_write_count)

        per_leg_fz = self._light_per_leg_fz_scratch
        per_leg_fz.fill(0.0)
        contact_state = self._light_current_contact_state
        contact_state.fill(False)
        nonleg_ground = False
        thorax_ground = False
        self._light_contact_record_count += int(data.ncon)
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            is_ground = g1 == self.ground_geom_id or g2 == self.ground_geom_id
            if not is_ground:
                continue
            fly_geom = g1 if g1 in self.geom_id_to_segment else (g2 if g2 in self.geom_id_to_segment else -1)
            if fly_geom < 0:
                continue
            leg_idx = self.geom_id_to_leg.get(fly_geom, -1)
            if leg_idx >= 0:
                contact_state[leg_idx] = True
                self._light_per_leg_contact_record_count[leg_idx] += 1
            else:
                nonleg_ground = True
                thorax_ground |= int(model.geom_bodyid[fly_geom]) == self.thorax_body_id
            wrench = self._light_contact_force_scratch
            wrench.fill(0.0)
            mj.mj_contactForce(model, data, contact_id, wrench)
            force_world_on_geom2 = contact.frame.reshape(3, 3).T @ wrench[:3]
            force_world_on_fly = -force_world_on_geom2 if fly_geom == g1 else force_world_on_geom2
            if leg_idx >= 0:
                per_leg_fz[leg_idx] += force_world_on_fly[2]
        for leg_idx, grounded in enumerate(contact_state):
            was_grounded = bool(self._light_previous_contact_state[leg_idx])
            if grounded and not was_grounded:
                self._light_open_contact_start[leg_idx] = k
            elif was_grounded and not grounded:
                start = self._light_open_contact_start[leg_idx]
                if start is not None:
                    self._light_contact_events.append((leg_idx, start, k - 1))
                self._light_open_contact_start[leg_idx] = None
        self._light_previous_contact_state[:] = contact_state
        self._light_last_foot_fz[:] = per_leg_fz
        self._light_foot_fz_min = np.minimum(self._light_foot_fz_min, per_leg_fz)
        self._light_foot_fz_max = np.maximum(self._light_foot_fz_max, per_leg_fz)
        self._light_nonleg_ground_contact_samples += int(nonleg_ground)
        self._light_thorax_ground_contact_samples += int(thorax_ground)

        if k % self.capture_stride == 0 or k == self._light_last_native_step:
            i = self._light_sample_count
            xmat = data.xmat[self.thorax_body_id].reshape(3, 3)
            heading = xmat[:, 0]
            self._light_native_step[i] = k
            self._light_time_s[i] = float(data.time)
            self._light_thorax_position_mm[i] = thorax_pos
            self._light_thorax_quat_wxyz[i] = data.xquat[self.thorax_body_id]
            self._light_thorax_yaw_rad[i] = math.atan2(float(heading[1]), float(heading[0]))
            self._light_thorax_heading_xy[i] = heading[:2]
            self._light_com_position_mm[i] = np.average(
                data.xipos[self.body_ids], axis=0, weights=self.body_mass
            )
            self._light_com_z_min_mm = min(self._light_com_z_min_mm, float(self._light_com_position_mm[i, 2]))
            for geom_id in self.thorax_collision_geom_ids:
                fromto = np.zeros(6, dtype=np.float64)
                clearance = mj.mj_geomDistance(model, data, int(geom_id), self.ground_geom_id, 1e6, fromto)
                self._light_thorax_clearance_min_mm = min(self._light_thorax_clearance_min_mm, float(clearance))
            self._light_joint_angle_rad[i] = data.qpos[self.joint_angles_ids]
            self._light_foot_position_mm[i] = data.xpos[self.foot_body_ids]
            self._light_cpg_phase_rad[i] = self.controller.cpg_network.curr_phases
            self._light_cpg_magnitude[i] = self.controller.cpg_network.curr_magnitudes
            self._light_position_ctrl_rad[i] = data.ctrl[self.position_actuator_ids]
            if len(self.adhesion_actuator_ids):
                ctrl = data.ctrl[self.adhesion_actuator_ids]
                self._light_adhesion_ctrl[i, :len(ctrl)] = ctrl
            self._light_descending_drive[i] = signal
            if k == 0:
                self._light_initial_state = {
                    "time_s": float(data.time), "qpos": np.asarray(data.qpos).copy(),
                    "qvel": np.asarray(data.qvel).copy(),
                }
            self._light_sample_count += 1
        self._native_samples += 1

    def _record_light_control_sample(self, signal: np.ndarray) -> None:
        i = self._light_control_count
        if i >= len(self._light_control_time_s):
            raise RuntimeError("control trace buffer exhausted")
        self._light_control_native_step[i] = self._native_samples - 1
        self._light_control_time_s[i] = self.time_s
        self._light_control_foot_fz[i] = self._light_last_foot_fz
        self._light_control_stance[i] = np.asarray(self.last_action.adhesion_onoff, dtype=np.uint8)
        self._light_control_count += 1

    def _light_step_result(self, signal: np.ndarray, control_time: float,
                           first_sample: int) -> dict[str, np.ndarray | float | int]:
        xmat = self.data.xmat[self.thorax_body_id].reshape(3, 3)
        heading = xmat[:, 0]
        return {
            "time_s": self.time_s,
            "thorax_x_mm": float(self.data.xpos[self.thorax_body_id, 0]),
            "thorax_heading_yaw_rad": math.atan2(float(heading[1]), float(heading[0])),
            "foot_contact_fz_up_native": self._light_last_foot_fz.copy(),
            "cpg_phase_rad": np.asarray(self.controller.cpg_network.curr_phases).copy(),
            "descending_signal": signal.copy(), "native_samples_added": self._native_samples - first_sample,
            "control_time_s": control_time,
        }

    def light_trace_arrays(self) -> dict[str, np.ndarray]:
        """Return the sampled yaw window inputs, including the terminal native sample."""
        if self.capture_mode != "light":
            raise RuntimeError("light_trace_arrays is only available in light capture mode")
        n = self._light_sample_count
        return {
            "time_s": self._light_time_s[:n],
            "thorax_heading_yaw_rad": self._light_thorax_yaw_rad[:n],
            "thorax_heading_xy": self._light_thorax_heading_xy[:n],
        }

    def observation(self):
        """Read upstream hybrid-controller observation without stepping physics."""
        return self.HybridControllerObservation.from_sim(self.sim, FLY_NAME)

    def step_control_tick(
        self,
        descending_signal: Iterable[float],
        *,
        compact_result: bool = False,
    ) -> dict[str, np.ndarray | float | int]:
        """Apply one upstream controller update and advance one 1 ms interval.

        ``descending_signal`` is ``[left, right]``; each signed channel must be
        in ``[-1, 1]``. Positive values drive the side's nominal forward CPG,
        negative values reverse its phase direction, and absolute magnitude
        sets CPG step amplitude (the base intrinsic rhythm remains 12 Hz).
        ``compact_result=True`` returns only ``native_samples_added`` and asks
        controllers that advertise support to omit diagnostic snapshots.
        """
        if self._closed:
            raise RuntimeError("session is closed")
        signal = np.asarray(tuple(descending_signal), dtype=np.float64)
        if signal.shape != (2,) or not np.all(np.isfinite(signal)) or np.any(np.abs(signal) > 1):
            raise ValueError("descending_signal must be two finite values in [-1,1]")
        control_time = self.time_s
        obs = self.observation()
        if compact_result and bool(
            getattr(self.controller, "supports_compact_result", False)
        ):
            action = self.controller.step(signal, obs, compact_result=True)
        else:
            action = self.controller.step(signal, obs)
        self.apply_locomotion_action(self.sim, FLY_NAME, action)
        self.last_action = action
        self._control_samples += 1

        if not self._record_initial:
            capture_started = time.perf_counter()
            if self.capture_mode == "light":
                self._record_light_native_sample(signal)
            else:
                self._record_native_sample(signal)
            self._capture_s += time.perf_counter() - capture_started
            self._record_initial = True

        n_substeps = round(self.config.control_timestep_s / self.config.native_timestep_s)
        first_sample = self._native_samples
        for _ in range(n_substeps):
            physics_started = time.perf_counter()
            self.sim.step()
            self._physics_step_s += time.perf_counter() - physics_started
            capture_started = time.perf_counter()
            if self.capture_mode == "light":
                self._record_light_native_sample(signal)
            else:
                self._record_native_sample(signal)
            self._capture_s += time.perf_counter() - capture_started
            if self.render_video:
                self.sim.render_as_needed()
        if self.capture_mode == "light":
            capture_started = time.perf_counter()
            self._record_light_control_sample(signal)
            self._capture_s += time.perf_counter() - capture_started
            if compact_result:
                return {"native_samples_added": self._native_samples - first_sample}
            return self._light_step_result(signal, control_time, first_sample)
        if compact_result:
            return {"native_samples_added": self._native_samples - first_sample}
        final_idx = self._native_samples - 1
        return {
            "time_s": float(self.all_arrays["time_s"][final_idx]),
            "thorax_x_mm": float(self.all_arrays["body_xpos_mm"][final_idx, self.thorax_index, 0]),
            "thorax_heading_yaw_rad": float(self.all_arrays["thorax_heading_yaw_rad"][final_idx]),
            "foot_contact_fz_up_native": self.all_arrays["foot_contact_fz_up_native"][final_idx].copy(),
            "cpg_phase_rad": self.cpg_phase_by_sample[final_idx].copy(),
            "descending_signal": signal.copy(),
            "native_samples_added": self._native_samples - first_sample,
            "control_time_s": control_time,
        }

    def _record_native_sample(self, signal: np.ndarray) -> None:
        k = self._native_samples
        if k >= len(self.all_arrays["time_s"]):
            raise RuntimeError("duration buffer exhausted")
        mj = self.mj
        model, data = self.model, self.data
        arrays = self.all_arrays
        arrays["time_s"][k] = float(data.time)
        arrays["qpos"][k] = data.qpos
        arrays["qvel"][k] = data.qvel
        arrays["joint_angle_rad"][k] = data.qpos[self.joint_angles_ids]
        arrays["joint_velocity_rad_s"][k] = data.qvel[self.joint_vel_ids]
        body_pos = data.xpos[self.body_ids]
        arrays["body_xpos_mm"][k] = body_pos
        arrays["body_xquat_wxyz"][k] = data.xquat[self.body_ids]
        arrays["body_xipos_mm"][k] = data.xipos[self.body_ids]
        arrays["body_xfrc_applied_native"][k] = data.xfrc_applied[self.body_ids]
        arrays["qfrc_applied_native"][k] = data.qfrc_applied
        for j, body_id in enumerate(self.body_ids):
            vel = np.zeros(6, dtype=np.float64)
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, int(body_id), vel, 0)
            arrays["body_xangvel_rad_s"][k, j] = vel[:3]
            arrays["body_xvel_mm_s"][k, j] = vel[3:]
        masses = self.body_mass
        mass_sum = float(masses.sum())
        if mass_sum <= 0:
            raise RuntimeError("fly model has no positive body mass")
        arrays["com_position_mm"][k] = np.average(data.xipos[self.body_ids], axis=0, weights=masses)
        arrays["com_velocity_mm_s"][k] = np.average(
            arrays["body_xvel_mm_s"][k], axis=0, weights=masses
        )
        thorax_xmat = data.xmat[self.thorax_body_id].reshape(3, 3)
        heading = thorax_xmat[:, 0]
        arrays["thorax_heading_xy"][k] = heading[:2]
        arrays["thorax_heading_yaw_rad"][k] = math.atan2(float(heading[1]), float(heading[0]))
        foot_pos = data.xpos[self.foot_body_ids]
        arrays["tarsus5_xpos_mm"][k] = foot_pos
        arrays["tarsus5_z_mm"][k] = foot_pos[:, 2]

        found, force, torque, pos, normal, tangent = self.sim.get_ground_contact_info(FLY_NAME)
        arrays["foot_contact_found"][k] = found > 0
        arrays["foot_contact_force_local_native"][k] = force
        arrays["foot_contact_torque_local_native"][k] = torque
        arrays["foot_contact_pos_mm"][k] = pos
        arrays["foot_contact_normal_world"][k] = normal
        arrays["foot_contact_tangent_world"][k] = tangent

        per_leg_fz = np.zeros(6, dtype=np.float64)
        nonleg_ground = False
        thorax_ground = False
        ncon = int(data.ncon)
        for contact_id in range(ncon):
            contact = data.contact[contact_id]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            is_ground = (g1 == self.ground_geom_id) or (g2 == self.ground_geom_id)
            wrench = np.zeros(6, dtype=np.float64)
            mj.mj_contactForce(model, data, contact_id, wrench)
            frame = contact.frame.reshape(3, 3)
            force_world_on_geom2 = frame.T @ wrench[:3]
            force_world_on_fly = np.zeros(3, dtype=np.float64)
            fly_geom = g1 if g1 in self.geom_id_to_segment else (g2 if g2 in self.geom_id_to_segment else -1)
            if fly_geom >= 0:
                force_world_on_fly = -force_world_on_geom2 if fly_geom == g1 else force_world_on_geom2
            segment_name = self.geom_id_to_segment.get(fly_geom, "")
            leg_idx = self.geom_id_to_leg.get(fly_geom, -1)
            if is_ground and fly_geom >= 0:
                if leg_idx >= 0:
                    per_leg_fz[leg_idx] += force_world_on_fly[2]
                else:
                    nonleg_ground = True
                    if int(model.geom_bodyid[fly_geom]) == self.thorax_body_id:
                        thorax_ground = True
            self.contact_step.append(k)
            self.contact_geom1.append(g1)
            self.contact_geom2.append(g2)
            self.contact_position.append(np.asarray(contact.pos, dtype=np.float32).copy())
            self.contact_normal.append(np.asarray(frame[0], dtype=np.float32).copy())
            self.contact_force_on_geom2.append(np.asarray(force_world_on_geom2, dtype=np.float32))
            self.contact_force_on_fly.append(np.asarray(force_world_on_fly, dtype=np.float32))
            self.contact_is_ground.append(int(is_ground))
            self.contact_fly_segment.append(segment_name)
            self.contact_leg_index.append(leg_idx)
        arrays["foot_contact_fz_up_native"][k] = per_leg_fz
        self.nonleg_ground_contact_by_sample[k] = int(nonleg_ground)
        self.thorax_ground_contact_by_sample[k] = int(thorax_ground)

        clearance = np.zeros(len(self.thorax_collision_geom_ids), dtype=np.float64)
        for i, geom_id in enumerate(self.thorax_collision_geom_ids):
            fromto = np.zeros(6, dtype=np.float64)
            clearance[i] = mj.mj_geomDistance(
                model, data, int(geom_id), self.ground_geom_id, 1e6, fromto
            )
        arrays["thorax_collision_geom_clearance_mm"][k] = clearance
        arrays["thorax_collision_clearance_min_mm"][k] = float(clearance.min())

        self.position_control_by_sample[k] = data.ctrl[self.position_actuator_ids]
        self.position_force_by_sample[k] = data.actuator_force[self.position_actuator_ids]
        if len(self.adhesion_actuator_ids):
            adhesion_ctrl = data.ctrl[self.adhesion_actuator_ids]
            self.adhesion_control_by_sample[k, : len(adhesion_ctrl)] = adhesion_ctrl
            self.adhesion_force_by_sample[k, : len(adhesion_ctrl)] = data.actuator_force[self.adhesion_actuator_ids]
        self.cpg_phase_by_sample[k] = self.controller.cpg_network.curr_phases
        self.cpg_magnitude_by_sample[k] = self.controller.cpg_network.curr_magnitudes
        self.descending_drive_by_sample[k] = signal
        self.leg_stance_command_by_sample[k] = np.asarray(action_stance := self.last_action.adhesion_onoff, dtype=np.uint8)
        self.root_external_wrench_by_sample[k] = data.xfrc_applied[self.thorax_body_id]
        # The reference code path has no assignment to the free-root qpos/qvel
        # arrays after Simulation.__init__ performs its documented keyframe reset.
        self.root_qpos_write_count_by_sample[k] = self._root_qpos_write_count
        self.root_qvel_write_count_by_sample[k] = self._root_qvel_write_count
        self._native_samples += 1

    def _model_options(self) -> dict[str, object]:
        opt = self.model.opt
        return {
            "native_timestep_s": float(opt.timestep),
            "gravity_mm_s2": np.asarray(opt.gravity, dtype=float).tolist(),
            "integrator": int(opt.integrator),
            "solver": int(opt.solver),
            "iterations": int(opt.iterations),
            "noslip_iterations": int(opt.noslip_iterations),
            "cone": int(opt.cone),
            "jacobian": int(opt.jacobian),
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "nbody": int(self.model.nbody),
            "ngeom": int(self.model.ngeom),
            "njnt": int(self.model.njnt),
            "nsensor": int(self.model.nsensor),
            "npair": int(self.model.npair),
            "nexclude": int(self.model.nexclude),
        }

    def _collect_source_hashes(self) -> dict[str, object]:
        relpaths = [
            "src/flygym/__init__.py",
            "src/flygym/anatomy.py",
            "src/flygym/compose/physics.py",
            "src/flygym/compose/fly/base_fly.py",
            "src/flygym/compose/fly/neuromechfly.py",
            "src/flygym/compose/world/base_world.py",
            "src/flygym/compose/world/flat_ground.py",
            "src/flygym/simulation.py",
            "src/flygym/rendering.py",
            "src/flygym/utils/mjcf.py",
            "src/flygym_demo/complex_terrain/common.py",
            "src/flygym_demo/complex_terrain/preprogrammed.py",
            "src/flygym_demo/complex_terrain/cpg_controller.py",
            "src/flygym_demo/complex_terrain/hybrid_controller.py",
            "src/flygym_demo/complex_terrain/turning_controller.py",
        ]
        try:
            installed_version = importlib.metadata.version("flygym")
        except importlib.metadata.PackageNotFoundError:
            installed_version = "unknown"
        installed_root = Path(importlib.metadata.distribution("flygym").locate_file(""))
        installed_src = installed_root
        installed_hashes: dict[str, str | None] = {}
        checkout_hashes: dict[str, str | None] = {}
        comparisons: dict[str, bool | None] = {}
        for rel in relpaths:
            package_rel = rel.removeprefix("src/")
            installed = installed_src / package_rel
            checkout = REFERENCE_ROOT / rel
            ih = _sha256_file(installed) if installed.is_file() else None
            ch = _sha256_file(checkout) if checkout.is_file() else None
            installed_hashes[rel] = ih
            checkout_hashes[rel] = ch
            comparisons[rel] = (ih == ch) if ih and ch else None
        commit = subprocess.run(
            ["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "reference_repository": str(REFERENCE_ROOT),
            "reference_commit_expected": REFERENCE_COMMIT,
            "reference_commit_actual": commit,
            "source_commit_match": commit == REFERENCE_COMMIT,
            "installed_flygym_version": installed_version,
            "installed_source_hashes": installed_hashes,
            "pinned_checkout_source_hashes": checkout_hashes,
            "source_sha256": {
                str(REFERENCE_ROOT / rel): checkout_hashes[rel]
                for rel in relpaths
                if checkout_hashes[rel]
            },
            "installed_source_sha256": {
                str(installed_src / rel.removeprefix("src/")): installed_hashes[rel]
                for rel in relpaths
                if installed_hashes[rel]
            },
            "installed_vs_checkout_match": comparisons,
            "license": "Apache-2.0",
            "license_path": str(REFERENCE_ROOT / "LICENSE"),
            "license_sha256": _sha256_file(REFERENCE_ROOT / "LICENSE"),
        }

    def run(self, drive_provider: Callable[[float, object], Iterable[float]] | None = None,
            progress_callback: Callable[[dict[str, float]], None] | None = None) -> None:
        """Advance to configured duration; default drive is constant [1, 1]."""
        n_ticks = round(self.duration_s / self.config.control_timestep_s)
        for tick in range(n_ticks):
            obs = self.observation()
            signal = (
                np.array([1.0, 1.0])
                if drive_provider is None
                else np.asarray(tuple(drive_provider(self.time_s, obs)), dtype=np.float64)
            )
            result = self.step_control_tick(signal)
            if progress_callback and (tick % max(1, int(0.25 / self.config.control_timestep_s)) == 0):
                progress_callback({"tick": float(tick), "time_s": float(result["time_s"])})

    def _find_module_asset_hashes(self) -> dict[str, object]:
        import flygym

        site_root = Path(importlib.metadata.distribution("flygym").locate_file(""))
        package_asset_roots = [
            ("flygym/assets", Path(flygym.assets_dir)),
            *[
                (
                    f"flygym_demo/{package_name}/assets",
                    site_root / "flygym_demo" / package_name / "assets",
                )
                for package_name in (
                    "complex_terrain",
                    "muscle_imitation",
                    "ball_flybody_data",
                    "spotlight_data",
                )
            ],
        ]
        hashes: dict[str, object] = {}
        for logical_root, asset_root in package_asset_roots:
            if not asset_root.is_dir():
                continue
            physical_root = asset_root.resolve(strict=True)
            for current, directories, filenames in os.walk(physical_root, followlinks=True):
                directories.sort()
                for filename in sorted(filenames):
                    physical_path = Path(current) / filename
                    relative = physical_path.relative_to(physical_root)
                    logical_path = asset_root / relative
                    key = f"{logical_root}/{relative.as_posix()}"
                    hashes[key] = {
                        "path": str(logical_path),
                        "physical_path": str(physical_path.resolve()),
                        "sha256": _sha256_file(physical_path),
                        "bytes": physical_path.stat().st_size,
                    }
        return hashes

    def _array_order_names(self) -> dict[str, list[str]]:
        joint_order = self.fly.get_jointdofs_order()
        native_joint_names = [
            _native_name(self.mj, self.model, self.mj.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        native_actuator_names = [
            _native_name(self.mj, self.model, self.mj.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        return {
            "body_names": self.body_names,
            "body_ids": self.body_ids.astype(int).tolist(),
            "flygym_segment_aliases_without_compiled_body_id": (
                self.uncompiled_body_segment_aliases
            ),
            "leg_order": list(LEGS),
            "joint_order": [
                f"{dof.parent.link}.{dof.child.link}.{dof.axis.value}"
                for dof in joint_order
            ],
            "position_actuator_joint_order": [
                f"{dof.parent.link}.{dof.child.link}.{dof.axis.value}"
                for dof in self.fly.get_actuated_jointdofs_order(
                    __import__("flygym.compose", fromlist=["ActuatorType"]).ActuatorType.POSITION
                )
            ],
            "thorax_collision_geom_names": self.thorax_collision_geom_names,
            "geom_names_by_id": self.geom_names,
            "body_mass_native_by_body": {
                name: float(mass) for name, mass in zip(self.body_names, self.body_mass)
            },
            "thorax_clearance_geometry_source": (
                "all compiled thorax geoms; no thorax-ground pair in model"
                if self.thorax_clearance_uses_fallback
                else "compiled explicit thorax-ground pair"
            ),
            "ground_paired_geom_names": [
                self.geom_names[i] for i in self.ground_paired_geom_ids
            ],
            "native_joint_names": native_joint_names,
            "native_joint_qposadr": self.model.jnt_qposadr.astype(int).tolist(),
            "native_joint_dofadr": self.model.jnt_dofadr.astype(int).tolist(),
            "native_joint_type": self.model.jnt_type.astype(int).tolist(),
            "native_actuator_names": native_actuator_names,
            "native_actuator_trnid": self.model.actuator_trnid.astype(int).tolist(),
        }

    def _light_contact_event_arrays(self) -> dict[str, np.ndarray]:
        events = list(self._light_contact_events)
        last_step = max(0, self._native_samples - 1)
        events.extend((leg, start, last_step) for leg, start in enumerate(self._light_open_contact_start)
                      if start is not None)
        event_array = np.asarray(events, dtype=np.int64).reshape(-1, 3)
        return {
            "leg_index": event_array[:, 0].astype(np.int8),
            "start_native_step": event_array[:, 1],
            "end_native_step_inclusive": event_array[:, 2],
            "start_time_s": event_array[:, 1].astype(np.float64) * self.config.native_timestep_s,
            "end_time_s": event_array[:, 2].astype(np.float64) * self.config.native_timestep_s,
        }

    def save_light(
        self,
        output_dir: Path,
        *,
        drive_schedule_sha256: str | None = None,
        constant_drive_values_left_right: tuple[float, float] | None = None,
        controller_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Write compact, explicitly versioned light-capture artifacts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite an existing run: {output_dir}")
        if self._native_samples == 0:
            raise RuntimeError("no physics samples recorded")
        count, n_control = self._light_sample_count, self._light_control_count
        requested_native_count = self._light_last_native_step + 1
        requested_control_count = round(self.duration_s / self.config.control_timestep_s)
        capture_status = (
            "complete" if self._native_samples == requested_native_count
            and self._control_samples == requested_control_count else "partial"
        )
        actual_duration_s = max(0.0, self.time_s - float(self._light_initial_state["time_s"]))
        initial_path = output_dir / "initial_state.json"
        save_started = time.perf_counter()
        initial = self._light_initial_state
        _json_dump(initial_path, {
            "time_s": initial["time_s"], "qpos": initial["qpos"].tolist(),
            "qvel": initial["qvel"].tolist(), "root_qpos_write_count_after_initialization": 0,
            "root_qvel_write_count_after_initialization": 0,
            "source": "FlyGym keyframe reset followed by mj_forward; no runner qpos/qvel assignment",
        })
        trajectory_path = output_dir / "trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            schema=np.asarray("flygym-reference-light-trajectory/v1"),
            capture_stride_native_steps=np.asarray(self.capture_stride, dtype=np.int32),
            native_step=self._light_native_step[:count], time_s=self._light_time_s[:count],
            thorax_position_mm=self._light_thorax_position_mm[:count],
            thorax_quat_wxyz=self._light_thorax_quat_wxyz[:count],
            thorax_heading_xy=self._light_thorax_heading_xy[:count],
            thorax_yaw_rad=self._light_thorax_yaw_rad[:count],
            com_position_mm=self._light_com_position_mm[:count],
            joint_angle_rad=self._light_joint_angle_rad[:count],
            tarsus5_position_mm=self._light_foot_position_mm[:count],
            cpg_phase_rad=self._light_cpg_phase_rad[:count], cpg_magnitude=self._light_cpg_magnitude[:count],
            position_actuator_ctrl_rad=self._light_position_ctrl_rad[:count],
            adhesion_actuator_ctrl=self._light_adhesion_ctrl[:count],
            descending_drive_left_right=self._light_descending_drive[:count],
            control_native_step=self._light_control_native_step[:n_control],
            control_time_s=self._light_control_time_s[:n_control],
            control_foot_contact_fz_up_native=self._light_control_foot_fz[:n_control],
            control_leg_stance_command=self._light_control_stance[:n_control],
        )
        contacts_path = output_dir / "contact_events.npz"
        events = self._light_contact_event_arrays()
        np.savez_compressed(contacts_path, **events)
        save_compress_s = time.perf_counter() - save_started
        time_s = self._light_time_s[:count]
        thorax = self._light_thorax_position_mm[:count]
        yaw = np.unwrap(self._light_thorax_yaw_rad[:count].astype(np.float64))
        analysis_begin = min(0.2, float(time_s[-1]))
        mask = time_s >= analysis_begin
        control_mask = self._light_control_time_s[:n_control] >= analysis_begin
        mass_sum = float(self.body_mass.sum())
        event_counts = {leg: int(np.count_nonzero(events["leg_index"] == i)) for i, leg in enumerate(LEGS)}
        summary = {
            "schema": "flygym-reference-light-summary/v1", "capture_mode": "light", "run_id": self.run_id,
            "status": capture_status, "duration_s": actual_duration_s,
            "requested_duration_s": self.duration_s, "actual_duration_s": actual_duration_s,
            "native_sample_count": int(self._native_samples), "native_sample_hz": 1.0 / self.config.native_timestep_s,
            "control_sample_count": int(self._control_samples), "control_sample_hz": 1.0 / self.config.control_timestep_s,
            "requested_native_sample_count": int(requested_native_count),
            "requested_control_sample_count": int(requested_control_count),
            "trajectory_end_s": float(time_s[-1]),
            "analysis_start_s": analysis_begin, "analysis_end_s": float(time_s[-1]),
            "thorax_x_advance_mm_analysis": float(thorax[mask][-1, 0] - thorax[mask][0, 0]),
            "thorax_x_ls_slope_mm_s_analysis": float(np.polyfit(time_s[mask], thorax[mask, 0], 1)[0]) if mask.sum() > 1 else None,
            "thorax_heading_yaw_change_rad_analysis": float(yaw[mask][-1] - yaw[mask][0]),
            "thorax_z_min_mm": float(self._light_thorax_z_min_mm), "com_z_min_mm": float(self._light_com_z_min_mm),
            "thorax_collision_clearance_min_mm": float(self._light_thorax_clearance_min_mm),
            "com_z_min_sampling": "stride sampled; may miss native-step minima",
            "thorax_collision_clearance_min_sampling": "stride sampled; may miss native-step minima",
            "thorax_ground_contact_count": int(self._light_thorax_ground_contact_samples),
            "nonleg_ground_contact_sample_count": int(self._light_nonleg_ground_contact_samples),
            "native_contact_event_count": int(self._light_contact_record_count), "native_contact_episode_count": int(len(events["leg_index"])),
            "root_qpos_write_count": int(self._root_qpos_write_count), "root_qvel_write_count": int(self._root_qvel_write_count),
            "root_qpos_write_count_min": int(self._light_root_qpos_write_min), "root_qpos_write_count_max": int(self._light_root_qpos_write_max),
            "root_qvel_write_count_min": int(self._light_root_qvel_write_min), "root_qvel_write_count_max": int(self._light_root_qvel_write_max),
            "root_external_wrench_peak_native": self._light_root_wrench_peak,
            "fly_body_external_wrench_peak_native": self._light_body_wrench_peak,
            "generalized_applied_force_peak_native": self._light_qfrc_peak,
            "body_mass_native_sum": mass_sum, "body_weight_native": mass_sum * GRAVITY_MM_S2,
            "per_leg_positive_fz_contact_fraction_analysis": {
                leg: float((self._light_control_foot_fz[:n_control, i][control_mask] > 0).mean())
                if control_mask.any() else None for i, leg in enumerate(LEGS)
            },
            "per_leg_native_ground_contact_event_count": {
                leg: int(self._light_per_leg_contact_record_count[i]) for i, leg in enumerate(LEGS)
            },
            "per_leg_native_ground_contact_episode_count": {leg: event_counts[leg] for leg in LEGS},
            "per_leg_fz_min_native": self._light_foot_fz_min.tolist(), "per_leg_fz_max_native": self._light_foot_fz_max.tolist(),
            "native_units": {"length": "mm", "time": "s", "joint_angle": "rad", "mass": "g", "force": "µN", "torque": "µN·mm"},
            "timing_s": {"physics_step_s": self._physics_step_s, "capture_s": self._capture_s},
        }
        summary_path = output_dir / "summary.json"
        summary["timing_s"]["save_compress_s"] = save_compress_s
        _json_dump(summary_path, summary)
        source_hashes = self.upstream_source_hashes
        output_files = {}
        for path in (trajectory_path, contacts_path, initial_path, summary_path):
            output_files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        manifest = {
            "schema": "flygym-reference-light-manifest/v1", "run_id": self.run_id,
            "status": capture_status, "created_utc": _utc_now(),
            "requested_duration_s": self.duration_s, "actual_duration_s": actual_duration_s,
            "capture": {"mode": "light", "stride_native_steps": self.capture_stride, "trajectory_sample_count": int(count),
                        "control_sample_count": int(n_control), "requested_native_sample_count": int(requested_native_count),
                        "requested_control_sample_count": int(requested_control_count),
                        "contact_transition_timing": "exact native step; interval end inclusive", "body_velocity_capture": False,
                        "sampled_minima": {"com_z_min_mm": "stride samples; may miss native-step minima",
                                           "thorax_collision_clearance_min_mm": "stride samples; may miss native-step minima"}},
            "walk_config": asdict(self.config), "duration_s": self.duration_s, "seed": self.config.seed,
            "drive_schedule_sha256": drive_schedule_sha256, "constant_drive_values_left_right": constant_drive_values_left_right,
            "controller_metadata": controller_metadata or {},
            "native_timestep_s": float(self.model.opt.timestep), "control_timestep_s": self.config.control_timestep_s,
            "effective_model_options": self.effective_model_options,
            "body_names": self.body_names, "body_ids": self.body_ids.tolist(),
            "body_mass_native_by_body": {name: float(mass) for name, mass in zip(self.body_names, self.body_mass)},
            "joint_angle_qpos_addresses": self.joint_angles_ids.tolist(),
            "position_actuator_ids": self.position_actuator_ids.tolist(),
            "adhesion_actuator_ids": self.adhesion_actuator_ids.tolist(),
            "leg_order": list(LEGS), "upstream_source": source_hashes,
            "package_versions": {"flygym": source_hashes["installed_flygym_version"], "mujoco": str(self.mj.__version__),
                                 "numpy": np.__version__, "python": platform.python_version()},
            "output_files": output_files,
            "timing_scope": {"physics_step_s": "sim.step calls only", "capture_s": "recording only; excludes rendering",
                             "save_compress_s": "initial state and trajectory/contact serialization; excludes hashes"},
        }
        _json_dump(manifest_path, manifest)
        return {"manifest": str(manifest_path), "summary": summary, "output_dir": str(output_dir)}

    def save(
        self,
        output_dir: Path,
        *,
        drive_schedule_sha256: str | None = None,
        constant_drive_values_left_right: tuple[float, float] | None = None,
        controller_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Write the recorded native trace, video, plots, and hashed manifest."""
        if self.capture_mode == "light":
            return self.save_light(
                output_dir, drive_schedule_sha256=drive_schedule_sha256,
                constant_drive_values_left_right=constant_drive_values_left_right,
                controller_metadata=controller_metadata,
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite an existing run: {output_dir}")
        if self._native_samples == 0:
            raise RuntimeError("no physics samples recorded")
        count = self._native_samples
        config_path = output_dir / "effective_config.json"
        _json_dump(
            config_path,
            {
                "run_id": self.run_id,
                "walk_config": asdict(self.config),
                "duration_s": self.duration_s,
                "seed": self.config.seed,
                "constant_drive_values_left_right": constant_drive_values_left_right,
                "drive_schedule_sha256": drive_schedule_sha256,
                "command_line_argv": sys.argv,
                "render_backend": {
                    "physics": "MuJoCo CPU",
                    "gl_context": "OSMesa software rendering",
                    "mujoco_gl": os.environ.get("MUJOCO_GL"),
                    "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
                    "libgl_always_software": os.environ.get("LIBGL_ALWAYS_SOFTWARE"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "video_codec": "libx264 software encoder",
                    "gpu_used": False,
                },
                "upstream_source_commit": REFERENCE_COMMIT,
                "native_options": self.effective_model_options,
                "thorax_clearance_geom_names": self.thorax_collision_geom_names,
                "thorax_clearance_uses_fallback": self.thorax_clearance_uses_fallback,
            },
        )
        initial_state_path = output_dir / "initial_state.json"
        _json_dump(
            initial_state_path,
            {
                "time_s": float(self.all_arrays["time_s"][0]),
                "qpos": self.all_arrays["qpos"][0].tolist(),
                "qvel": self.all_arrays["qvel"][0].tolist(),
                "root_qpos_write_count_after_initialization": 0,
                "root_qvel_write_count_after_initialization": 0,
                "source": "FlyGym keyframe reset followed by mj_forward; no qpos/qvel assignment by runner",
            },
        )
        seed_path = output_dir / "seed.json"
        _json_dump(
            seed_path,
            {"seed": self.config.seed, "rng": "numpy.random.RandomState via upstream CPGNetwork"},
        )
        trace_path = output_dir / "trajectory.npz"
        trace_arrays = dict(self.all_arrays)
        trace_arrays.update(
            {
                "position_actuator_ctrl_rad": self.position_control_by_sample,
                "position_actuator_force_native": self.position_force_by_sample,
                "adhesion_actuator_ctrl": self.adhesion_control_by_sample,
                "adhesion_actuator_force_native": self.adhesion_force_by_sample,
                "cpg_phase_rad": self.cpg_phase_by_sample,
                "cpg_magnitude": self.cpg_magnitude_by_sample,
                "descending_drive_left_right": self.descending_drive_by_sample,
                "leg_stance_command": self.leg_stance_command_by_sample,
                "root_qpos_write_count": self.root_qpos_write_count_by_sample,
                "root_qvel_write_count": self.root_qvel_write_count_by_sample,
                "root_external_wrench_native": self.root_external_wrench_by_sample,
                "nonleg_ground_contact": self.nonleg_ground_contact_by_sample,
                "thorax_ground_contact": self.thorax_ground_contact_by_sample,
            }
        )
        np.savez_compressed(
            trace_path,
            **{
                key: (value if key == "body_mass_native" else value[:count])
                for key, value in trace_arrays.items()
            },
        )

        contacts_path = output_dir / "contacts.npz"
        np.savez_compressed(
            contacts_path,
            step_index=np.asarray(self.contact_step, dtype=np.int32),
            time_s=self.all_arrays["time_s"][np.asarray(self.contact_step, dtype=np.int32)],
            geom1_id=np.asarray(self.contact_geom1, dtype=np.int32),
            geom2_id=np.asarray(self.contact_geom2, dtype=np.int32),
            position_mm=np.asarray(self.contact_position, dtype=np.float32).reshape(-1, 3),
            normal_world=np.asarray(self.contact_normal, dtype=np.float32).reshape(-1, 3),
            force_world_on_geom2_native=np.asarray(self.contact_force_on_geom2, dtype=np.float32).reshape(-1, 3),
            force_world_on_fly_native=np.asarray(self.contact_force_on_fly, dtype=np.float32).reshape(-1, 3),
            is_ground_contact=np.asarray(self.contact_is_ground, dtype=np.uint8),
            fly_body_segment=np.asarray(self.contact_fly_segment, dtype="U40"),
            leg_index=np.asarray(self.contact_leg_index, dtype=np.int8),
        )

        compiled_xml_path = output_dir / "compiled_model.xml"
        compiled_xml_path.write_text(self.compiled_mjcf_xml)
        _json_dump(output_dir / "model_options.json", self.effective_model_options)
        order_names = self._array_order_names()
        _json_dump(output_dir / "array_order.json", order_names)
        source_assets = self._find_module_asset_hashes()
        _json_dump(output_dir / "upstream_asset_hashes.json", source_assets)
        asset_storage_manifest_path = Path("/data/models/flygym/asset-storage-manifest.json")
        asset_storage_manifest = (
            {
                "path": str(asset_storage_manifest_path),
                "sha256": _sha256_file(asset_storage_manifest_path),
                "bytes": asset_storage_manifest_path.stat().st_size,
            }
            if asset_storage_manifest_path.is_file()
            else None
        )

        arrays = trace_arrays
        time_s = arrays["time_s"][:count]
        thorax_x = arrays["body_xpos_mm"][:count, self.thorax_index, 0].astype(np.float64)
        heading_yaw = np.unwrap(arrays["thorax_heading_yaw_rad"][:count])
        total_mass = float(self.body_mass.sum())
        body_weight_native = total_mass * GRAVITY_MM_S2
        root_wrench_norm = np.linalg.norm(self.root_external_wrench_by_sample[:count], axis=1)
        body_wrench_norm = np.linalg.norm(
            arrays["body_xfrc_applied_native"][:count], axis=2
        )
        qfrc_applied_peak = float(np.max(np.abs(arrays["qfrc_applied_native"][:count])))
        analysis_begin = min(0.2, float(time_s[-1]))
        mask = time_s >= analysis_begin
        summary = {
            "run_id": self.run_id,
            "status": "complete",
            "duration_s": float(time_s[-1] - time_s[0]),
            "native_sample_count": int(count),
            "native_sample_hz": 1.0 / self.config.native_timestep_s,
            "control_sample_count": int(self._control_samples),
            "control_sample_hz": 1.0 / self.config.control_timestep_s,
            "analysis_start_s": analysis_begin,
            "analysis_end_s": float(time_s[-1]),
            "thorax_x_advance_mm_analysis": float(thorax_x[mask][-1] - thorax_x[mask][0]),
            "thorax_x_ls_slope_mm_s_analysis": float(np.polyfit(time_s[mask], thorax_x[mask], 1)[0]) if mask.sum() > 1 else None,
            "thorax_heading_yaw_change_rad_analysis": float(heading_yaw[mask][-1] - heading_yaw[mask][0]),
            "thorax_z_min_mm": float(arrays["body_xpos_mm"][:count, self.thorax_index, 2].min()),
            "com_z_min_mm": float(arrays["com_position_mm"][:count, 2].min()),
            "thorax_collision_clearance_min_mm": float(arrays["thorax_collision_clearance_min_mm"][:count].min()),
            "thorax_ground_contact_count": int(self.thorax_ground_contact_by_sample[:count].sum()),
            "nonleg_ground_contact_sample_count": int(self.nonleg_ground_contact_by_sample[:count].sum()),
            "native_contact_event_count": int(len(self.contact_step)),
            "root_qpos_write_count": int(self._root_qpos_write_count),
            "root_qvel_write_count": int(self._root_qvel_write_count),
            "root_external_wrench_peak_native": float(root_wrench_norm.max()),
            "fly_body_external_wrench_peak_native": float(body_wrench_norm.max()),
            "generalized_applied_force_peak_native": qfrc_applied_peak,
            "body_mass_native_sum": total_mass,
            "body_weight_native": body_weight_native,
            "per_leg_positive_fz_contact_fraction_analysis": {
                leg: float((arrays["foot_contact_fz_up_native"][mask, i] > 0).mean())
                for i, leg in enumerate(LEGS)
            },
            "per_leg_native_ground_contact_event_count": {
                leg: int(sum(1 for idx, grounded in zip(self.contact_leg_index, self.contact_is_ground)
                             if idx == i and grounded))
                for i, leg in enumerate(LEGS)
            },
            "native_units": {
                "length": "mm (FlyGym/MuJoCo model coordinates)",
                "time": "s",
                "joint_angle": "rad",
                "mass": "g (FlyGym model convention)",
                "force": "µN (FlyGym model convention: g·mm/s²)",
                "torque": "µN·mm",
                "position_actuator_ctrl": "rad",
                "position_actuator_force": "µN·mm",
                "adhesion_ctrl": "unitless [0,1]",
            },
        }
        _json_dump(output_dir / "summary.json", summary)

        if self.render_video:
            video_path = output_dir / "walking.mp4"
            self.sim.renderer.save_video(video_path)

        self._save_diagnostics(output_dir / "diagnostics.png", count)
        manifest = {
            "run_id": self.run_id,
            "status": "complete",
            "created_utc": _utc_now(),
            "project_root": str(PROJECT_ROOT),
            "runner_path": str(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "config": asdict(self.config),
            "effective_config_file": config_path.name,
            "effective_config_sha256": _sha256_file(config_path),
            "initial_state_file": initial_state_path.name,
            "initial_state_sha256": _sha256_file(initial_state_path),
            "seed_file": seed_path.name,
            "seed_sha256": _sha256_file(seed_path),
            "driver_file": str(Path(__file__).resolve()),
            "driver_sha256": _sha256_file(Path(__file__).resolve()),
            "command_line_argv": sys.argv,
            "constant_drive_values_left_right": constant_drive_values_left_right,
            "controller": {
                "class": "flygym_demo.complex_terrain.HybridTurningController",
                "pattern": "upstream tripod CPG + preprogrammed six-leg steps + hybrid retraction/stumbling corrections",
                "left_right_channel_order": ["left", "right"],
                "channel_range": [-1.0, 1.0],
                "positive_channel": "forward CPG phase; absolute channel magnitude sets CPG step amplitude",
                "fixed_base_intrinsic_frequency_hz": self.config.cpg_frequency_hz,
                "input_source": "constant descending command" if drive_schedule_sha256 is None else "time-series descending command",
                "drive_schedule_sha256": drive_schedule_sha256,
                "stimulus_label_branching": False,
                **(controller_metadata or {}),
            },
            "actuation_boundary": {
                "body_model": "NeuroMechFly v2.1.0 six-leg articulated micro-CT model",
                "actuator_type": "upstream position actuators",
                "adhesion_enabled": self.config.adhesion_enabled,
                "legacy_rest_compatibility": self.config.legacy_rest_compatibility,
                "physics": "native MuJoCo CPU step; no trajectory replay or qpos overwrite",
                "qpos_qvel_assignments_after_keyframe_reset": 0,
                "root_external_wrench_applied": False,
            },
            "native_timestep_s": float(self.model.opt.timestep),
            "package_versions": {
                "flygym": self.upstream_source_hashes["installed_flygym_version"],
                "mujoco": str(self.mj.__version__),
                "numpy": np.__version__,
                "python": platform.python_version(),
            },
            "render_backend": {
                "physics": "MuJoCo CPU",
                "gl_context": "OSMesa software rendering",
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
                "libgl_always_software": os.environ.get("LIBGL_ALWAYS_SOFTWARE"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "video_codec": "libx264 software encoder",
                "gpu_used": False,
            },
            "effective_model_options": self.effective_model_options,
            "effective_model_options_sha256": _sha256_bytes(
                json.dumps(self.effective_model_options, sort_keys=True).encode()
            ),
            "compiled_mjcf_xml": {
                "file": "compiled_model.xml",
                "sha256": self.compiled_mjcf_sha256,
            },
            "compiled_model_xml_file": "compiled_model.xml",
            "compiled_model_sha256": self.compiled_mjcf_sha256,
            "upstream_source": self.upstream_source_hashes,
            "source_sha256": self.upstream_source_hashes["source_sha256"],
            "installed_source_sha256": self.upstream_source_hashes["installed_source_sha256"],
            "asset_sha256": {
                str(asset["physical_path"]): str(asset["sha256"])
                for asset in source_assets.values()
            },
            "asset_storage_manifest": asset_storage_manifest,
            "upstream_assets": {
                "file": "upstream_asset_hashes.json",
                "sha256_manifest": _sha256_file(output_dir / "upstream_asset_hashes.json"),
                "asset_count": len(source_assets),
            },
            "license": "Apache-2.0",
            "seed": self.config.seed,
            "spawn_position_mm": list(self.config.spawn_position_mm),
            "spawn_quaternion_wxyz": list(self.config.spawn_quaternion_wxyz),
            "body_names": self.body_names,
            "body_ids": self.body_ids.tolist(),
            "flygym_segment_aliases_without_compiled_body_id": (
                self.uncompiled_body_segment_aliases
            ),
            "body_mass_native_by_body": {
                name: float(mass) for name, mass in zip(self.body_names, self.body_mass)
            },
            "thorax_clearance_geometry_source": (
                "all compiled thorax geoms; no thorax-ground pair in model"
                if self.thorax_clearance_uses_fallback
                else "compiled explicit thorax-ground pair"
            ),
            "thorax_clearance_geom_names": self.thorax_collision_geom_names,
            "ground_paired_geom_names": [self.geom_names[i] for i in self.ground_paired_geom_ids],
            "array_order_file": "array_order.json",
            "array_order_sha256": _sha256_file(output_dir / "array_order.json"),
            "output_files": {},
        }
        output_names = [
            "trajectory.npz",
            "contacts.npz",
            "compiled_model.xml",
            "model_options.json",
            "array_order.json",
            "upstream_asset_hashes.json",
            "summary.json",
            "diagnostics.png",
            "effective_config.json",
            "initial_state.json",
            "seed.json",
        ]
        if self.render_video:
            output_names.append("walking.mp4")
        for name in output_names:
            path = output_dir / name
            manifest["output_files"][name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        manifest["artifact_sha256"] = {
            str(output_dir / name): _sha256_file(output_dir / name)
            for name in output_names
        }
        _json_dump(manifest_path, manifest)
        integrity = {
            "hash_algorithm": "SHA-256",
            "files": {
                str(output_dir / name): {
                    "bytes": (output_dir / name).stat().st_size,
                    "sha256": _sha256_file(output_dir / name),
                }
                for name in output_names + ["manifest.json"]
            },
        }
        _json_dump(output_dir / "integrity.json", integrity)
        return {"manifest": str(manifest_path), "summary": summary, "output_dir": str(output_dir)}

    def _save_diagnostics(self, path: Path, count: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a = self.all_arrays
        t = a["time_s"][:count]
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
        thorax = a["body_xpos_mm"][:count, self.thorax_index]
        axes[0].plot(t, thorax[:, 0], label="thorax X")
        axes[0].plot(t, thorax[:, 1], label="thorax Y", alpha=0.75)
        axes[0].set_ylabel("position (mm)")
        axes[0].legend(ncol=2)
        axes[1].plot(t, np.unwrap(a["thorax_heading_yaw_rad"][:count]))
        axes[1].set_ylabel("heading yaw (rad)")
        for idx, leg in enumerate(LEGS):
            axes[2].plot(t, a["tarsus5_z_mm"][:count, idx], label=leg)
        axes[2].set_ylabel("tarsus5 Z (mm)")
        axes[2].legend(ncol=6)
        for idx, leg in enumerate(LEGS):
            axes[3].plot(t, a["foot_contact_fz_up_native"][:count, idx], label=leg)
        axes[3].set_ylabel("native upward contact")
        axes[3].set_xlabel("simulation time (s)")
        axes[3].legend(ncol=6)
        fig.suptitle(f"{self.run_id}: native MuJoCo body, heading and contact trace")
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def close(self) -> None:
        if self._closed:
            return
        if self.sim.renderer is not None:
            self.sim.renderer.close()
        self._closed = True

    def __enter__(self) -> "FlyGymReferenceSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def run_locomotion_trial(
    output_dir: Path,
    *,
    duration_s: float = 2.0,
    seed: int = 0,
    drive_provider: Callable[[float, object], Iterable[float]] | None = None,
    render_video: bool = True,
    run_id: str = "flygym-reference-flatground-cpg-tripod-seed0-20261005-v1",
    drive_schedule_sha256: str | None = None,
    constant_drive_values_left_right: tuple[float, float] | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, object]:
    """Run the standard flat-ground walk or an explicit external side-drive law."""
    session = FlyGymReferenceSession(
        duration_s=duration_s,
        seed=seed,
        render_video=render_video,
        run_id=run_id,
    )
    try:
        session.run(drive_provider=drive_provider, progress_callback=progress_callback)
        return session.save(
            Path(output_dir),
            drive_schedule_sha256=drive_schedule_sha256,
            constant_drive_values_left_right=constant_drive_values_left_right,
        )
    finally:
        session.close()


def _progress(info: dict[str, float]) -> None:
    print(f"physics progress: t={info['time_s']:.3f}s / tick={int(info['tick'])}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--left-descending-drive", type=float, default=1.0)
    parser.add_argument("--right-descending-drive", type=float, default=1.0)
    parser.add_argument("--drive-csv", type=Path, help="time_s,left_descending_drive,right_descending_drive")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="flygym-reference-flatground-cpg-tripod-seed0-20261005-v1")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args(argv)

    if args.drive_csv:
        times, values = read_drive_schedule(args.drive_csv)
        provider = scheduled_drive_provider(times, values)
        schedule_hash = _sha256_file(args.drive_csv)
        constant_drive = None
    else:
        values = np.asarray([args.left_descending_drive, args.right_descending_drive], dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1):
            parser.error("descending drive constants must be finite and in [-1,1]")
        provider = lambda _time_s, _obs: values
        schedule_hash = None
        constant_drive = (float(values[0]), float(values[1]))
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output directory must be new or empty: {args.out}")
    try:
        result = run_locomotion_trial(
            args.out,
            duration_s=args.duration_s,
            seed=args.seed,
            drive_provider=provider,
            render_video=not args.no_video,
            run_id=args.run_id,
            drive_schedule_sha256=schedule_hash,
            constant_drive_values_left_right=constant_drive,
            progress_callback=None if args.quiet_progress else _progress,
        )
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        _json_dump(
            args.out / "failure.json",
            {
                "run_id": args.run_id,
                "failed_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )
        raise
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Artifacts: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
