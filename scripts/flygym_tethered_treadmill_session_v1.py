"""Tethered spherical-treadmill session built around the accepted V3 AP gait.

The session is CPU-only and never creates a renderer or neural worker.  Its
``step_control_tick`` method advances MuJoCo only when a caller explicitly
invokes it; the compile-preflight runner does not invoke that method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from prepare_flygym_tethered_treadmill_pilot import (
    ENGINEERING_BALL_PRESET,
    TetheredBallWorld,
)


FLY_NAME = "nmf"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
EXPECTED_AP_SHA256 = "689364bc661c4fc37a15ddc617a398450a3f52d2fe4da7d202b74687cda91b8f"
EXPECTED_V41_SHA256 = "126b641e2998992e4cad351f4a48292882063851350e4006ba40552a38d7e2eb"
BODY_BINDING = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-v4-1-command-transfer-calibration-20261007-v1/"
    "cases/static_ps_reference_replay/controller_binding.json"
)
BODY_EFFECTIVE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-v4-1-command-transfer-calibration-20261007-v1/"
    "cases/static_ps_reference_replay/effective_config.json"
)


@dataclass(frozen=True)
class TetheredPilotConfig:
    """Frozen body configuration and separate AP gesture command for the pilot."""

    duration_s: float = 6.0
    seed: int = 1729
    native_timestep_s: float = 0.0001
    control_timestep_s: float = 0.001
    position_actuator_gain_native: float = 45.0
    position_actuator_forcerange_native: tuple[float, float] = (-65.0, 65.0)
    joint_stiffness_native: float = 0.05
    joint_damping_native: float = 0.06
    passive_tarsus_stiffness_native: float = 7.5
    passive_tarsus_damping_native: float = 0.01
    legacy_rest_compatibility: bool = True
    adhesion_enabled: bool = True
    adhesion_gain_native: float = 40.0
    cpg_frequency_hz: float = 7.4074074074074066
    cpg_amplitude: float = 1.0
    cpg_coupling_strength: float = 10.0
    cpg_convergence_coefficient: float = 20.0
    spawn_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.7)
    spawn_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    leg_power_p: float = 0.30
    differential_ap_d: float = 0.0


class TetheredBallSessionV1:
    """A six-leg FlyGym session whose only support surface is a rotating ball."""

    def __init__(
        self,
        config: TetheredPilotConfig = TetheredPilotConfig(),
        *,
        ball_preset=ENGINEERING_BALL_PRESET,
    ) -> None:
        if config.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if not math.isclose(config.native_timestep_s, 1e-4, rel_tol=0, abs_tol=1e-12):
            raise ValueError("the pilot pins FlyGym's 100 us native timestep")
        ratio = config.control_timestep_s / config.native_timestep_s
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-12):
            raise ValueError("control timestep must be an integer multiple of native dt")
        if not math.isclose(
            round(config.duration_s / config.native_timestep_s) * config.native_timestep_s,
            config.duration_s,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("duration must be an integer multiple of native timestep")
        if not math.isclose(
            round(config.duration_s / config.control_timestep_s) * config.control_timestep_s,
            config.duration_s,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("duration must be an integer multiple of control timestep")
        if not 0.0 <= config.leg_power_p <= 1.0:
            raise ValueError("leg power P must be in [0,1]")
        if not -1.0 <= config.differential_ap_d <= 1.0:
            raise ValueError("AP differential d must be in [-1,1]")

        self.config = config
        self.run_id = (
            f"tethered-ball-P{config.leg_power_p:.3f}-d{config.differential_ap_d:+.3f}"
            f"-T{config.duration_s:g}s-seed{int(config.seed)}"
        )
        self.ball_preset = ball_preset
        self.duration_s = float(config.duration_s)
        self.n_native_expected = round(self.duration_s / config.native_timestep_s)
        self.n_control_expected = round(self.duration_s / config.control_timestep_s)
        self.n_native_steps_per_control_tick = round(
            config.control_timestep_s / config.native_timestep_s
        )
        self._native_samples = 0
        self._control_samples = 0
        self._physics_step_count = 0
        self._root_qpos_write_count = 0
        self._root_qvel_write_count = 0
        self._initialization_root_mocap_pose_write_count = 0
        self._runtime_root_mocap_pose_write_count = 0
        self._root_mocap_pos_before_initialization_override: np.ndarray | None = None
        self._closed = False
        self._trace_started = False
        self._last_descending_signal = np.asarray(
            [config.leg_power_p, config.leg_power_p], dtype=np.float64
        )
        self._contact_rows: list[tuple[Any, ...]] = []

        # Rendering/GPU are disabled before importing the runtime.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", "/data/models/flygym/assets")

        import mujoco as mj
        from flygym.compose import ActuatorType
        from flygym.utils.math import Rotation3D
        from flygym.simulation import Simulation
        from flygym_demo.complex_terrain import (
            HybridControllerObservation,
            HybridTurningController,
            PreprogrammedSteps,
            apply_locomotion_action,
            make_tripod_cpg_network,
        )
        from flygym_legacy_rest_compat import make_legacy_rest_locomotion_fly
        from flygym_ap_endpoint_gesture_prototype import (
            APEndpointGestureConfig,
            APEndpointGestureController,
        )

        self.mj = mj
        self.ActuatorType = ActuatorType
        self.HybridControllerObservation = HybridControllerObservation
        self.apply_locomotion_action = apply_locomotion_action
        self.fly = make_legacy_rest_locomotion_fly(
            name=FLY_NAME,
            joint_stiffness=config.joint_stiffness_native,
            joint_damping=config.joint_damping_native,
            passive_tarsus_stiffness=config.passive_tarsus_stiffness_native,
            passive_tarsus_damping=config.passive_tarsus_damping_native,
            actuator_gain=config.position_actuator_gain_native,
            actuator_forcerange=config.position_actuator_forcerange_native,
            add_adhesion=config.adhesion_enabled,
            adhesion_gain=config.adhesion_gain_native,
            colorize=True,
        )
        self.tracking_camera = self.fly.add_tracking_camera()
        self.world = TetheredBallWorld(ball_preset)
        self.world.add_fly(
            self.fly,
            config.spawn_position_mm,
            Rotation3D(format="quat", values=config.spawn_quaternion_wxyz),
            add_ground_contact_sensors=True,
        )
        self.sim = Simulation(self.world, timestep=config.native_timestep_s)
        self.model = self.sim.mj_model
        self.data = self.sim.mj_data
        # FlyGym's neutral keyframe reset restores the mocap body to its MJCF
        # body reference pose and does not retain the TetheredWorld spawn-site
        # translation. Apply the requested experimental root pose explicitly
        # once at initialization, before controller/private-FK construction.
        self._map_fly_environment_ids()
        self._root_mocap_pos_before_initialization_override = np.asarray(
            self.data.mocap_pos[self.root_mocap_id], dtype=np.float64
        ).copy()
        self.data.mocap_pos[self.root_mocap_id] = np.asarray(
            config.spawn_position_mm, dtype=np.float64
        )
        self.data.mocap_quat[self.root_mocap_id] = np.asarray(
            config.spawn_quaternion_wxyz, dtype=np.float64
        )
        self._initialization_root_mocap_pose_write_count = 1
        self.mj.mj_forward(self.model, self.data)
        if not math.isclose(float(self.data.time), 0.0, rel_tol=0, abs_tol=1e-15):
            raise RuntimeError("initial forward kinematics unexpectedly advanced time")
        if not np.allclose(
            self.data.xpos[self.root_body_id],
            np.asarray(config.spawn_position_mm, dtype=np.float64),
            rtol=0,
            atol=1e-9,
        ):
            raise RuntimeError("thorax/root pose does not match the requested initialization spawn")
        if not math.isclose(float(self.model.opt.timestep), config.native_timestep_s, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("compiled timestep differs from the pinned 100 us native dt")

        self.base_cpg = make_tripod_cpg_network(
            timestep=config.control_timestep_s,
            intrinsic_frequency=config.cpg_frequency_hz,
            intrinsic_amplitude=config.cpg_amplitude,
            coupling_strength=config.cpg_coupling_strength,
            convergence_coef=config.cpg_convergence_coefficient,
            seed=config.seed,
        )
        self.preprogrammed_steps = PreprogrammedSteps()
        self.controller = HybridTurningController(
            timestep=config.control_timestep_s,
            cpg_network=self.base_cpg,
            preprogrammed_steps=self.preprogrammed_steps,
            enable_adhesion=config.adhesion_enabled,
        )

        self.position_actuator_ids = np.asarray(
            self.sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][FLY_NAME],
            dtype=np.int32,
        )
        self.adhesion_actuator_ids = np.asarray(
            self.sim._intern_adhesionactuatorids_by_fly[FLY_NAME], dtype=np.int32
        )
        self.joint_angles_ids = np.asarray(
            self.sim._intern_qposadrs_by_fly[FLY_NAME], dtype=np.int32
        )
        self.joint_vel_ids = np.asarray(
            self.sim._intern_qveladrs_by_fly[FLY_NAME], dtype=np.int32
        )
        if len(self.position_actuator_ids) != 42 or len(self.adhesion_actuator_ids) != 6:
            raise RuntimeError("expected exactly 42 position and 6 adhesion actuator channels")

        if not BODY_BINDING.is_file() or not BODY_EFFECTIVE.is_file():
            raise FileNotFoundError("frozen V4.1 body-binding inputs are unavailable")
        self.body_binding = json.loads(BODY_BINDING.read_text())
        self.body_effective = json.loads(BODY_EFFECTIVE.read_text())
        frozen_walk_config = self.body_effective.get("walk_config", self.body_effective)
        config_as_dict = asdict(config)
        for key, expected_value in config_as_dict.items():
            if key in {"duration_s", "leg_power_p", "differential_ap_d"}:
                continue  # pilot duration/commands are not fields in the frozen body config
            observed_value = frozen_walk_config.get(key)
            if isinstance(expected_value, tuple):
                expected_value = list(expected_value)
            if observed_value != expected_value:
                raise RuntimeError(
                    f"body configuration differs from frozen V4.1 input at {key}: "
                    f"expected {expected_value!r}, observed {observed_value!r}"
                )
        bound_controller_sha = self.body_binding.get("controller_module_sha256")
        if bound_controller_sha != EXPECTED_V41_SHA256:
            raise RuntimeError("saved V4.1 controller binding does not match expected source")

        ap_path = Path(__file__).with_name("flygym_ap_endpoint_gesture_prototype.py")
        v41_path = Path(__file__).with_name("flygym_phase_speed_gait_v4_1.py")
        if source_sha256(ap_path) != EXPECTED_AP_SHA256 or source_sha256(v41_path) != EXPECTED_V41_SHA256:
            raise RuntimeError("accepted AP/V4.1 controller source hash differs from frozen binding")
        c = config.differential_ap_d
        soma_side = "left" if c >= 0.0 else "right"
        ap_values = dict(self.body_binding["controller_config"])
        ap_values.update(
            {
                "dna02_command": abs(c),
                "dng13_command": abs(c),
                "dna02_soma_side": soma_side,
                "dng13_soma_side": soma_side,
                "dna02_aep_posterior_shift_mm": 0.05,
                "dng13_endpoint_shift_mm": 0.05,
            }
        )
        self.gesture_config = APEndpointGestureConfig(**ap_values)
        self.controller = APEndpointGestureController(self, self.gesture_config)
        self.sim_controller = self.controller
        self._last_action = None
        self._last_control_metadata: dict[str, Any] = {}

        # Initialize the private FK state from the actual post-reset, explicitly
        # placed live state. The accepted controller allocates its own MjData;
        # its constructor does not retain the TetheredWorld spawn-site pose.
        self.mj.mj_copyData(self.controller._ik_data, self.model, self.data)
        self.mj.mj_forward(self.model, self.controller._ik_data)
        if not np.array_equal(self.data.mocap_pos, self.controller._ik_data.mocap_pos):
            raise RuntimeError("controller private FK mocap positions differ from live session")
        if not np.array_equal(self.data.mocap_quat, self.controller._ik_data.mocap_quat):
            raise RuntimeError("controller private FK mocap quaternions differ from live session")

        self._allocate_trace()

    @property
    def time_s(self) -> float:
        return float(self.data.time)

    def _map_fly_environment_ids(self) -> None:
        self.fly_segments = list(self.fly.get_bodysegs_order())
        self.fly_segment_names = [segment.name for segment in self.fly_segments]
        raw_fly_body_ids = np.asarray(
            self.sim._internal_bodyids_by_fly[FLY_NAME], dtype=np.int32
        )
        if len(raw_fly_body_ids) != len(self.fly_segment_names):
            raise RuntimeError("FlyGym body segment and body ID counts differ")
        self.fly_body_id_by_name: dict[str, int] = {}
        self.fly_segment_body_mapping: list[dict[str, Any]] = []
        actual_body_ids: list[int] = []
        actual_body_names: list[str] = []
        for name, raw_body_id in zip(self.fly_segment_names, raw_fly_body_ids):
            body_id = int(raw_body_id)
            if body_id >= 0:
                self.fly_body_id_by_name[name] = body_id
                if body_id not in actual_body_ids:
                    actual_body_ids.append(body_id)
                    actual_body_names.append(name)
                representation = (
                    "mujoco_body"
                    if actual_body_names[-1] == name and body_id == actual_body_ids[-1]
                    else "merged_alias_to_existing_mujoco_body"
                )
                self.fly_segment_body_mapping.append(
                    {
                        "segment_name": name,
                        "mujoco_body_id": body_id,
                        "representation": representation,
                    }
                )
            elif name == "c_head":
                self.fly_segment_body_mapping.append(
                    {
                        "segment_name": name,
                        "mujoco_body_id": None,
                        "representation": "fused_geom_alias_on_c_thorax",
                    }
                )
            else:
                self.fly_segment_body_mapping.append(
                    {
                        "segment_name": name,
                        "mujoco_body_id": None,
                        "representation": "unresolved_missing_body_id",
                    }
                )
                raise RuntimeError(
                    f"FlyGym segment {name!r} has no compiled body ID and is not the known fused c_head alias"
                )
        self.fly_body_ids = np.asarray(actual_body_ids, dtype=np.int32)
        self.fly_body_names = actual_body_names
        if np.any(self.fly_body_ids < 0):
            raise RuntimeError("invalid body IDs must not enter the FlyGym body trace")
        self.root_body_id = int(self.fly_body_id_by_name[self.fly.root_segment.name])
        self.thorax_body_id = int(self.fly_body_id_by_name["c_thorax"])
        self.head_geom_id = int(
            self.mj.mj_name2id(
                self.model, self.mj.mjtObj.mjOBJ_GEOM, f"{FLY_NAME}/c_head"
            )
        )
        if self.head_geom_id < 0:
            raise RuntimeError("compiled model has neither an independent c_head body nor its fused head geom")
        self.head_geom_parent_body_id = int(self.model.geom_bodyid[self.head_geom_id])
        if self.head_geom_parent_body_id != self.thorax_body_id:
            raise RuntimeError("compiled c_head geom is not attached to the fixed thorax body")
        self.foot_body_ids = np.asarray(
            [self.fly_body_id_by_name[f"{leg}_tarsus5"] for leg in LEGS],
            dtype=np.int32,
        )
        self.ball_body_id = self.mj.mj_name2id(
            self.model, self.mj.mjtObj.mjOBJ_BODY, "treadmill_ball_body"
        )
        self.ball_geom_id = self.mj.mj_name2id(
            self.model, self.mj.mjtObj.mjOBJ_GEOM, "treadmill_ball_geom"
        )
        self.ball_joint_id = self.mj.mj_name2id(
            self.model, self.mj.mjtObj.mjOBJ_JOINT, "treadmill_ball_spin"
        )
        if min(self.ball_body_id, self.ball_geom_id, self.ball_joint_id) < 0:
            raise RuntimeError("compiled treadmill ball body/geom/joint is missing")
        self.ball_qpos_adr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_dof_adr = int(self.model.jnt_dofadr[self.ball_joint_id])

        all_body_ids = set(range(1, self.model.nbody))
        fly_body_id_set = set(int(i) for i in self.fly_body_ids)
        if self.ball_body_id in fly_body_id_set:
            raise RuntimeError("ball body was incorrectly included in FlyGym fly body IDs")
        self.environment_body_ids = sorted(all_body_ids - fly_body_id_set)
        self.root_mocap_id = int(self.model.body_mocapid[self.root_body_id])
        if self.root_mocap_id < 0:
            raise RuntimeError("tethered fly root is not a mocap body")
        if np.any(self.model.jnt_type == self.mj.mjtJoint.mjJNT_FREE):
            raise RuntimeError("tethered ball model unexpectedly contains a free joint")
        if any(
            int(self.model.jnt_bodyid[joint_id]) == self.root_body_id
            for joint_id in range(self.model.njnt)
        ):
            raise RuntimeError("tethered fly root unexpectedly has a joint")

        self.ground_geom_ids = np.asarray(
            self.sim._internal_ground_geom_ids, dtype=np.int32
        )
        if self.ground_geom_ids.tolist() != [int(self.ball_geom_id)]:
            raise RuntimeError("only the treadmill ball geom must be routed as support")
        self.geom_names = [
            self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, i) or f"geom-{i}"
            for i in range(self.model.ngeom)
        ]
        self.geom_id_to_segment: dict[int, str] = {}
        self.geom_id_to_leg: dict[int, int] = {}
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            segment_name = next(
                (name for name, bid in self.fly_body_id_by_name.items() if bid == int(body_id)),
                None,
            )
            if segment_name is not None:
                self.geom_id_to_segment[geom_id] = segment_name
                if segment_name[:2] in LEGS:
                    self.geom_id_to_leg[geom_id] = LEGS.index(segment_name[:2])
        # The source FlyGym anatomy exposes c_head as a segment alias, but the
        # compiled model fuses it into the thorax as a named geom.
        self.geom_id_to_segment[self.head_geom_id] = "c_head"
        self.fly_geom_ids = set(self.geom_id_to_segment)

        self.tarsus5_geom_ids_by_leg: dict[str, list[int]] = {}
        for leg in LEGS:
            name = f"{leg}_tarsus5"
            body_id = self.fly_body_id_by_name[name]
            self.tarsus5_geom_ids_by_leg[leg] = [
                geom_id
                for geom_id, mapped_body in enumerate(self.model.geom_bodyid)
                if int(mapped_body) == body_id
            ]

    def _allocate_trace(self) -> None:
        n = self.n_native_expected + 1
        nb = len(self.fly_body_ids)
        self.trace: dict[str, np.ndarray] = {
            "time_s": np.zeros(n, dtype=np.float64),
            "qpos": np.zeros((n, self.model.nq), dtype=np.float64),
            "qvel": np.zeros((n, self.model.nv), dtype=np.float64),
            "body_xpos_mm": np.zeros((n, nb, 3), dtype=np.float32),
            "body_xquat_wxyz": np.zeros((n, nb, 4), dtype=np.float32),
            "thorax_position_world_mm": np.zeros((n, 3), dtype=np.float32),
            "thorax_quaternion_world_wxyz": np.zeros((n, 4), dtype=np.float32),
            "head_geom_position_world_mm": np.zeros((n, 3), dtype=np.float32),
            "head_geom_xmat_world": np.zeros((n, 3, 3), dtype=np.float32),
            "root_mocap_pos_world_mm": np.zeros((n, 3), dtype=np.float64),
            "root_mocap_quat_world_wxyz": np.zeros((n, 4), dtype=np.float64),
            "ball_qpos_quaternion_wxyz": np.zeros((n, 4), dtype=np.float64),
            "ball_qvel_rad_s": np.zeros((n, 3), dtype=np.float64),
            "ball_body_angvel_world_rad_s": np.zeros((n, 3), dtype=np.float64),
            "ball_body_linvel_world_mm_s": np.zeros((n, 3), dtype=np.float64),
            "ball_contact_force_world_native": np.zeros((n, 3), dtype=np.float64),
            "ball_contact_torque_center_world_native": np.zeros((n, 3), dtype=np.float64),
            "ball_contact_count": np.zeros(n, dtype=np.int16),
            "foot_contact_found": np.zeros((n, 6), dtype=np.uint8),
            "foot_contact_force_local_native": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_torque_local_native": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_pos_world_mm": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_normal_world": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_tangent_world": np.zeros((n, 6, 3), dtype=np.float32),
            "foot_contact_fz_up_native": np.zeros((n, 6), dtype=np.float32),
            "position_ctrl_native": np.zeros((n, 42), dtype=np.float64),
            "position_force_native": np.zeros((n, 42), dtype=np.float64),
            "adhesion_ctrl_native": np.zeros((n, 6), dtype=np.float64),
            "root_external_wrench_native": np.zeros((n, 6), dtype=np.float64),
            "ball_external_wrench_native": np.zeros((n, 6), dtype=np.float64),
            "qfrc_applied_native": np.zeros((n, self.model.nv), dtype=np.float64),
            "root_qpos_write_count": np.zeros(n, dtype=np.int32),
            "root_qvel_write_count": np.zeros(n, dtype=np.int32),
            "thorax_ground_contact": np.zeros(n, dtype=np.uint8),
            "nonleg_ball_contact": np.zeros(n, dtype=np.uint8),
            "descending_signal_lr": np.zeros((n, 2), dtype=np.float32),
            "ap_differential_command": np.zeros(n, dtype=np.float32),
            "cpg_phase_rad_by_leg": np.zeros((n, 6), dtype=np.float32),
        }
        self._native_samples = 0

    def observation(self):
        return self.HybridControllerObservation.from_sim(self.sim, FLY_NAME)

    def _record_native_sample(self, signal: np.ndarray, *, detailed_contacts: bool) -> None:
        k = self._native_samples
        if k >= len(self.trace["time_s"]):
            raise RuntimeError("native trace buffer exhausted")
        model, data, mj = self.model, self.data, self.mj
        t = self.trace
        t["time_s"][k] = float(data.time)
        t["qpos"][k] = data.qpos
        t["qvel"][k] = data.qvel
        t["body_xpos_mm"][k] = data.xpos[self.fly_body_ids]
        t["body_xquat_wxyz"][k] = data.xquat[self.fly_body_ids]
        t["thorax_position_world_mm"][k] = data.xpos[self.thorax_body_id]
        t["thorax_quaternion_world_wxyz"][k] = data.xquat[self.thorax_body_id]
        t["head_geom_position_world_mm"][k] = data.geom_xpos[self.head_geom_id]
        t["head_geom_xmat_world"][k] = data.geom_xmat[self.head_geom_id].reshape(3, 3)
        t["root_mocap_pos_world_mm"][k] = data.mocap_pos[self.root_mocap_id]
        t["root_mocap_quat_world_wxyz"][k] = data.mocap_quat[self.root_mocap_id]
        t["ball_qpos_quaternion_wxyz"][k] = data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 4]
        t["ball_qvel_rad_s"][k] = data.qvel[self.ball_dof_adr : self.ball_dof_adr + 3]
        ball_velocity = np.zeros(6, dtype=np.float64)
        mj.mj_objectVelocity(
            model, data, mj.mjtObj.mjOBJ_BODY, self.ball_body_id, ball_velocity, 0
        )
        t["ball_body_angvel_world_rad_s"][k] = ball_velocity[:3]
        t["ball_body_linvel_world_mm_s"][k] = ball_velocity[3:]
        t["root_external_wrench_native"][k] = data.xfrc_applied[self.root_body_id]
        t["ball_external_wrench_native"][k] = data.xfrc_applied[self.ball_body_id]
        t["qfrc_applied_native"][k] = data.qfrc_applied
        t["root_qpos_write_count"][k] = self._root_qpos_write_count
        t["root_qvel_write_count"][k] = self._root_qvel_write_count
        t["descending_signal_lr"][k] = signal
        t["ap_differential_command"][k] = self.config.differential_ap_d
        t["cpg_phase_rad_by_leg"][k] = np.asarray(
            self.controller.base.cpg_network.curr_phases, dtype=np.float32
        )
        t["position_ctrl_native"][k] = data.ctrl[self.position_actuator_ids]
        t["position_force_native"][k] = data.actuator_force[self.position_actuator_ids]
        t["adhesion_ctrl_native"][k] = data.ctrl[self.adhesion_actuator_ids]

        found, force, torque, pos, normal, tangent = self.sim.get_ground_contact_info(FLY_NAME)
        t["foot_contact_found"][k] = found > 0
        t["foot_contact_force_local_native"][k] = force
        t["foot_contact_torque_local_native"][k] = torque
        t["foot_contact_pos_world_mm"][k] = pos
        t["foot_contact_normal_world"][k] = normal
        t["foot_contact_tangent_world"][k] = tangent

        ball_center = np.asarray(data.xpos[self.ball_body_id], dtype=np.float64)
        total_force = np.zeros(3, dtype=np.float64)
        total_torque = np.zeros(3, dtype=np.float64)
        per_leg_fz = np.zeros(6, dtype=np.float64)
        nonleg_contact = False
        thorax_contact = False
        count = 0
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if self.ball_geom_id not in (geom1, geom2):
                continue
            count += 1
            wrench = np.zeros(6, dtype=np.float64)
            mj.mj_contactForce(model, data, contact_id, wrench)
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            force_on_geom2 = frame.T @ wrench[:3]
            torque_on_geom2 = frame.T @ wrench[3:]
            if geom2 == self.ball_geom_id:
                force_on_ball = force_on_geom2
                couple_on_ball = torque_on_geom2
            else:
                force_on_ball = -force_on_geom2
                couple_on_ball = -torque_on_geom2
            r_ball_to_contact = np.asarray(contact.pos, dtype=np.float64) - ball_center
            moment_at_ball_center = np.cross(r_ball_to_contact, force_on_ball) + couple_on_ball
            total_force += force_on_ball
            total_torque += moment_at_ball_center

            fly_geom = geom1 if geom1 in self.fly_geom_ids else (geom2 if geom2 in self.fly_geom_ids else -1)
            segment = self.geom_id_to_segment.get(fly_geom, "")
            leg_index = self.geom_id_to_leg.get(fly_geom, -1)
            if leg_index >= 0:
                force_on_fly = -force_on_geom2 if fly_geom == geom1 else force_on_geom2
                per_leg_fz[leg_index] += force_on_fly[2]
            elif fly_geom >= 0:
                nonleg_contact = True
                if int(model.geom_bodyid[fly_geom]) == self.thorax_body_id:
                    thorax_contact = True

            if detailed_contacts:
                self._contact_rows.append(
                    (
                        k,
                        geom1,
                        geom2,
                        leg_index,
                        float(contact.pos[0]), float(contact.pos[1]), float(contact.pos[2]),
                        *frame[0].tolist(),
                        *force_on_ball.tolist(),
                        *moment_at_ball_center.tolist(),
                        *wrench.tolist(),
                    )
                )

        t["ball_contact_force_world_native"][k] = total_force
        t["ball_contact_torque_center_world_native"][k] = total_torque
        t["ball_contact_count"][k] = count
        t["foot_contact_fz_up_native"][k] = per_leg_fz
        t["thorax_ground_contact"][k] = int(thorax_contact)
        t["nonleg_ball_contact"][k] = int(nonleg_contact)
        self._native_samples += 1

    def step_control_tick(self, descending_signal: Iterable[float]) -> dict[str, Any]:
        """Advance one 1 ms CPG/action tick and ten 100 us MuJoCo steps.

        The AP differential lives only in ``gesture_config``. The phase-speed
        controller receives the unchanged bilateral descending drive (P,P).
        This method is for a separately authorized run; compile preflight never
        calls it.
        """
        if self._closed:
            raise RuntimeError("session is closed")
        signal = np.asarray(tuple(descending_signal), dtype=np.float64)
        if signal.shape != (2,) or not np.all(np.isfinite(signal)) or np.any(np.abs(signal) > 1):
            raise ValueError("descending_signal must contain two finite values in [-1,1]")
        if not np.array_equal(signal, np.asarray([self.config.leg_power_p] * 2, dtype=np.float64)):
            raise ValueError("pilot must preserve the symmetric (P,P) CPG command")
        control_time = self.time_s
        action = self.controller.step(signal, self.observation())
        self.apply_locomotion_action(self.sim, FLY_NAME, action)
        self._last_action = action
        if not self._trace_started:
            self._record_native_sample(signal, detailed_contacts=True)
            self._trace_started = True
        native_first = self._native_samples
        for substep in range(self.n_native_steps_per_control_tick):
            self.sim.step()
            self._physics_step_count += 1
            self._record_native_sample(
                signal,
                detailed_contacts=(substep == self.n_native_steps_per_control_tick - 1),
            )
        self._control_samples += 1
        return {
            "control_time_s": control_time,
            "time_s": self.time_s,
            "native_samples_added": self._native_samples - native_first,
            "descending_signal_lr": signal.tolist(),
            "ap_differential_command": float(self.config.differential_ap_d),
            "ap_dna02_command": float(self.controller.gesture_config.dna02_command),
            "ap_dng13_command": float(self.controller.gesture_config.dng13_command),
            "ap_soma_side": str(self.controller.gesture_config.dna02_soma_side),
            "position_targets_rad": np.asarray(action.joint_angles, dtype=np.float64).tolist(),
            "controller_telemetry": self.controller.telemetry_snapshot(),
        }

    def run_fixed_condition(self) -> None:
        signal = np.asarray([self.config.leg_power_p, self.config.leg_power_p], dtype=np.float64)
        for _ in range(self.n_control_expected):
            self.step_control_tick(signal)

    def preflight_snapshot(self) -> dict[str, Any]:
        mj, model, data = self.mj, self.model, self.data
        ball_body = int(self.ball_body_id)
        qpos_addr = self.ball_qpos_adr
        qvel_addr = self.ball_dof_adr
        found, force, torque, pos, normal, tangent = self.sim.get_ground_contact_info(FLY_NAME)
        pair_rows: list[dict[str, Any]] = []
        for pair_id in range(model.npair):
            g1, g2 = int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id])
            if self.ball_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == self.ball_geom_id else g1
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "geom1": self.geom_names[g1],
                    "geom2": self.geom_names[g2],
                    "other_fly_segment": self.geom_id_to_segment.get(other),
                    "other_leg": self.geom_id_to_segment.get(other, "")[:2],
                    "condim": int(model.pair_dim[pair_id]),
                    "friction_coefficients_mu1_mu2_torsion_roll1_roll2": np.asarray(
                        model.pair_friction[pair_id], dtype=np.float64
                    ).tolist(),
                    "active_friction_components": (
                        ["tangential_sliding_mu1", "tangential_sliding_mu2"]
                        if int(model.pair_dim[pair_id]) == 3
                        else (
                            ["tangential_sliding_mu1", "tangential_sliding_mu2", "torsional"]
                            if int(model.pair_dim[pair_id]) == 4
                            else [
                                "tangential_sliding_mu1",
                                "tangential_sliding_mu2",
                                "torsional",
                                "rolling_mu1",
                                "rolling_mu2",
                            ]
                            if int(model.pair_dim[pair_id]) >= 6
                            else ["normal_only"]
                        )
                    ),
                }
            )
        if not pair_rows:
            raise RuntimeError("compiled model contains no explicit FlyGym ball-contact pairs")
        invalid_pairs = [row for row in pair_rows if row["other_leg"] not in LEGS]
        if invalid_pairs:
            raise RuntimeError(f"ball contact routing contains non-leg pairs: {invalid_pairs[:3]}")
        if any(row["condim"] != 3 for row in pair_rows):
            raise RuntimeError("tethered pilot expects explicit pairs with MuJoCo condim=3")

        initial_ball_force = np.zeros(3, dtype=np.float64)
        initial_ball_torque_center = np.zeros(3, dtype=np.float64)
        initial_ball_contact_rows: list[dict[str, Any]] = []
        ball_contact_count = 0
        ball_center = np.asarray(data.xpos[ball_body], dtype=np.float64)
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if self.ball_geom_id not in (geom1, geom2):
                continue
            ball_contact_count += 1
            wrench = np.zeros(6, dtype=np.float64)
            mj.mj_contactForce(model, data, contact_id, wrench)
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            force_on_geom2 = frame.T @ wrench[:3]
            couple_on_geom2 = frame.T @ wrench[3:]
            if geom2 == self.ball_geom_id:
                force_ball = force_on_geom2
                couple_ball = couple_on_geom2
            else:
                force_ball = -force_on_geom2
                couple_ball = -couple_on_geom2
            moment = np.cross(np.asarray(contact.pos, dtype=np.float64) - ball_center, force_ball) + couple_ball
            initial_ball_force += force_ball
            initial_ball_torque_center += moment
            initial_ball_contact_rows.append(
                {
                    "geom1": self.geom_names[geom1],
                    "geom2": self.geom_names[geom2],
                    "leg_segment": self.geom_id_to_segment.get(
                        geom1 if geom1 in self.fly_geom_ids else geom2
                    ),
                    "position_world_mm": np.asarray(contact.pos).tolist(),
                    "normal_world": frame[0].tolist(),
                    "raw_contact_wrench_native": wrench.tolist(),
                    "force_on_ball_world_native": force_ball.tolist(),
                    "torque_about_ball_center_world_native": moment.tolist(),
                }
            )

        foot_projection: dict[str, Any] = {}
        radius = float(self.ball_preset.radius_mm)
        for leg, body_id in zip(LEGS, self.foot_body_ids):
            geom_distances: list[float] = []
            nearest_points: list[list[float]] = []
            for geom_id in self.tarsus5_geom_ids_by_leg[leg]:
                fromto = np.zeros(6, dtype=np.float64)
                distance = float(mj.mj_geomDistance(model, data, geom_id, self.ball_geom_id, 1e6, fromto))
                geom_distances.append(distance)
                nearest_points.append(fromto.tolist())
            foot_position = np.asarray(data.xpos[int(body_id)], dtype=np.float64)
            radial_clearance = float(np.linalg.norm(foot_position - ball_center) - radius)
            foot_projection[leg] = {
                "tarsus5_body_origin_world_mm": foot_position.tolist(),
                "tarsus5_body_origin_radial_clearance_to_sphere_mm": radial_clearance,
                "tarsus5_geom_distance_to_sphere_mm": geom_distances,
                "tarsus5_geom_nearest_points_fromto_mm": nearest_points,
                "ground_contact_sensor_found": float(found[LEGS.index(leg)]),
                "sensor_force_contact_frame_native": np.asarray(force[LEGS.index(leg)]).tolist(),
                "sensor_contact_frame_torque_native": np.asarray(torque[LEGS.index(leg)]).tolist(),
                "sensor_position_world_mm": np.asarray(pos[LEGS.index(leg)]).tolist(),
                "sensor_normal_world": np.asarray(normal[LEGS.index(leg)]).tolist(),
                "sensor_tangent_world": np.asarray(tangent[LEGS.index(leg)]).tolist(),
            }

        angular_velocity = np.zeros(6, dtype=np.float64)
        mj.mj_objectVelocity(
            model, data, mj.mjtObj.mjOBJ_BODY, ball_body, angular_velocity, 0
        )
        root_id = self.root_body_id
        root_mocap_id = self.root_mocap_id
        free_joints = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
            if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE
        ]
        actuator_order = []
        for channel, actuator_id in enumerate(self.position_actuator_ids):
            actuator_id = int(actuator_id)
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            actuator_order.append(
                {
                    "channel": channel,
                    "actuator_name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                    "joint_name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id),
                    "dof_name": str(self.controller.dof_order[channel]),
                }
            )
        if len(actuator_order) != 42:
            raise RuntimeError("compiled actuator inventory is not exactly 42 channels")
        if len(self.sim._intern_groundcontactsensorids_by_fly[FLY_NAME]) != 96:
            raise RuntimeError("six per-leg 16-value contact sensors were not routed")

        return {
            "status": "compiled_preflight_no_physics_step",
            "run_id": self.run_id,
            "physics_step_count": int(self._physics_step_count),
            "mj_step_called_by_runner": False,
            "data_time_s": float(data.time),
            "effective_native_timestep_s": float(model.opt.timestep),
            "effective_native_rate_hz": 1.0 / float(model.opt.timestep),
            "effective_control_timestep_s": self.config.control_timestep_s,
            "control_ticks_planned": self.n_control_expected,
            "native_steps_planned": self.n_native_expected,
            "config": asdict(self.config),
            "ball_preset": {
                **asdict(self.ball_preset),
                "solid_sphere_inertia_preset_g_mm2": self.ball_preset.solid_sphere_inertia_g_mm2,
                "compiled_body_mass_g": float(model.body_mass[ball_body]),
                "compiled_body_inertia_g_mm2": np.asarray(model.body_inertia[ball_body]).tolist(),
                "body_id": ball_body,
                "geom_id": int(self.ball_geom_id),
                "joint_id": int(self.ball_joint_id),
                "qpos_address": qpos_addr,
                "dof_address": qvel_addr,
                "qpos_quaternion_wxyz": data.qpos[qpos_addr : qpos_addr + 4].tolist(),
                "qvel_rad_s": data.qvel[qvel_addr : qvel_addr + 3].tolist(),
                "world_angular_velocity_rad_s": angular_velocity[:3].tolist(),
                "world_linear_velocity_mm_s": angular_velocity[3:].tolist(),
                "external_wrench_native": data.xfrc_applied[ball_body].tolist(),
                "applied_generalized_force_native": data.qfrc_applied[qvel_addr : qvel_addr + 3].tolist(),
                "compiled_mass_positive": bool(model.body_mass[ball_body] > 0),
                "compiled_inertia_all_positive": bool(np.all(model.body_inertia[ball_body] > 0)),
                "center_fixed_by_ball_joint": True,
                "sphere_apex_world_z_mm": float(self.ball_preset.center_world_mm[2] + self.ball_preset.radius_mm),
                "tarsus5_projection_by_leg": foot_projection,
                "explicit_ball_contact_pairs": pair_rows,
                "explicit_ball_pair_count": len(pair_rows),
                "pair_condim": sorted({int(row["condim"]) for row in pair_rows}),
                "active_contact_friction_components": [
                    "two tangential sliding directions (mu1, mu2)"
                ],
                "inactive_preset_components_under_condim_3": [
                    "torsional friction coefficient",
                    "both rolling friction coefficients",
                ],
                "mechanics_limitation": (
                    "MuJoCo explicit pairs default to condim=3 here; torsional and rolling "
                    "coefficients are recorded but inactive. The sphere has no added "
                    "rolling/torsional resistance model, so ball spin is an engineering "
                    "mechanics response, not a calibrated treadmill response."
                ),
                "all_contact_count_after_mj_forward": int(data.ncon),
                "ball_contact_count_after_mj_forward": ball_contact_count,
                "ball_contact_force_world_native_after_mj_forward": initial_ball_force.tolist(),
                "ball_contact_torque_center_world_native_after_mj_forward": initial_ball_torque_center.tolist(),
                "ball_contact_details_after_mj_forward": initial_ball_contact_rows,
            },
            "model_inventory": {
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
                "nbody": int(model.nbody),
                "ngeom": int(model.ngeom),
                "njnt": int(model.njnt),
                "npair": int(model.npair),
                "nsensor": int(model.nsensor),
                "free_joints": free_joints,
                "no_free_root_joint": not free_joints,
                "fly_body_ids": self.fly_body_ids.tolist(),
                "fly_body_names": list(self.fly_body_names),
                "fly_segment_body_mapping": self.fly_segment_body_mapping,
                "environment_body_ids": self.environment_body_ids,
                "ball_body_excluded_from_fly_ids": ball_body not in set(self.fly_body_ids.tolist()),
                "root_body_id": root_id,
                "root_body_name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, root_id),
                "root_has_mocap": root_mocap_id >= 0,
                "root_mocap_id": root_mocap_id,
                "root_mocap_pos_world_mm": data.mocap_pos[root_mocap_id].tolist(),
                "root_mocap_quat_world_wxyz": data.mocap_quat[root_mocap_id].tolist(),
                "requested_initial_root_mocap_position_mm": list(self.config.spawn_position_mm),
                "requested_initial_root_mocap_quaternion_wxyz": list(self.config.spawn_quaternion_wxyz),
                "root_mocap_pos_before_initialization_override_mm": (
                    self._root_mocap_pos_before_initialization_override.tolist()
                    if self._root_mocap_pos_before_initialization_override is not None
                    else None
                ),
                "root_initialization_mocap_pose_write_count": int(
                    self._initialization_root_mocap_pose_write_count
                ),
                "root_runtime_mocap_pose_write_count": int(
                    self._runtime_root_mocap_pose_write_count
                ),
                "thorax_position_world_mm": data.xpos[self.thorax_body_id].tolist(),
                "thorax_quaternion_world_wxyz": data.xquat[self.thorax_body_id].tolist(),
                "head_reference": {
                    "representation": "named_geom_fused_to_c_thorax; no independent head body or joint",
                    "geom_id": int(self.head_geom_id),
                    "geom_name": self.geom_names[self.head_geom_id],
                    "parent_thorax_body_id": int(self.head_geom_parent_body_id),
                    "position_world_mm": data.geom_xpos[self.head_geom_id].tolist(),
                    "rotation_matrix_world": data.geom_xmat[self.head_geom_id].reshape(3, 3).tolist(),
                    "geom_local_position_mm": self.model.geom_pos[self.head_geom_id].tolist(),
                    "geom_local_quaternion_wxyz": self.model.geom_quat[self.head_geom_id].tolist(),
                },
                "position_actuator_count": len(actuator_order),
                "position_actuator_order": actuator_order,
                "adhesion_actuator_count": len(self.adhesion_actuator_ids),
                "ground_geom_ids": self.ground_geom_ids.tolist(),
                "ground_geom_names": [self.geom_names[i] for i in self.ground_geom_ids],
                "ground_contact_sensor_value_count": int(len(self.sim._intern_groundcontactsensorids_by_fly[FLY_NAME])),
                "private_fk_mocap_pos_exact_match": bool(
                    np.array_equal(data.mocap_pos, self.controller._ik_data.mocap_pos)
                ),
                "private_fk_mocap_quat_exact_match": bool(
                    np.array_equal(data.mocap_quat, self.controller._ik_data.mocap_quat)
                ),
                "initial_qpos": data.qpos.tolist(),
                "initial_qvel": data.qvel.tolist(),
                "initial_time_s": float(data.time),
                "root_state_write_counts": {
                    "qpos": self._root_qpos_write_count,
                    "qvel": self._root_qvel_write_count,
                },
                "root_external_wrench_native": data.xfrc_applied[root_id].tolist(),
                "max_abs_generalized_force_applied_native": float(np.max(np.abs(data.qfrc_applied))),
            },
            "drive_and_controller": {
                "descending_signal_lr": [self.config.leg_power_p, self.config.leg_power_p],
                "d_changes_descending_signal": False,
                "ap_gesture_config": asdict(self.controller.gesture_config),
                "d_soma_side_rule": "left if d >= 0, otherwise right",
                "d_ap_command_rule": "dna02_command = dng13_command = abs(d); shifts = 0.05 mm",
                "base_controller": self.controller.provenance.get("base_controller"),
                "base_cycle_template_frequency_hz": float(self.preprogrammed_steps.step_cycle_frequency_hz),
                "commanded_intrinsic_frequency_hz_at_P": float(
                    self.preprogrammed_steps.step_cycle_frequency_hz * self.config.leg_power_p
                ),
                "neural_worker": False,
            },
        }

    def save_trajectory(self, path: Path) -> None:
        if self._native_samples != self.n_native_expected + 1:
            raise RuntimeError(
                f"expected {self.n_native_expected + 1} native samples, got {self._native_samples}"
            )
        used = {key: value[: self._native_samples] for key, value in self.trace.items()}
        np.savez_compressed(path, **used)
        contact_columns = [
            "sample_index", "geom1_id", "geom2_id", "leg_index", "x_mm", "y_mm", "z_mm",
            "normal_x", "normal_y", "normal_z", "force_ball_x", "force_ball_y", "force_ball_z",
            "moment_ball_center_x", "moment_ball_center_y", "moment_ball_center_z",
            "wrench_0", "wrench_1", "wrench_2", "wrench_3", "wrench_4", "wrench_5",
        ]
        contact_dtype = np.float64
        contacts = np.asarray(self._contact_rows, dtype=contact_dtype)
        if contacts.size == 0:
            contacts = np.empty((0, len(contact_columns)), dtype=contact_dtype)
        np.savez_compressed(
            path.with_name("contacts.npz"),
            columns=np.asarray(contact_columns, dtype="U32"),
            rows=contacts,
            geom_names=np.asarray(self.geom_names, dtype="U128"),
            leg_names=np.asarray(LEGS, dtype="U8"),
        )

    def step_metadata(self) -> dict[str, Any]:
        return {
            "control_samples": self._control_samples,
            "native_samples": self._native_samples,
            "time_s": self.time_s,
            "drive_lr": [self.config.leg_power_p, self.config.leg_power_p],
            "d": self.config.differential_ap_d,
            "ap_commands": {
                "dna02": float(self.controller.gesture_config.dna02_command),
                "dng13": float(self.controller.gesture_config.dng13_command),
                "soma_side": self.controller.gesture_config.dna02_soma_side,
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        if getattr(self.sim, "renderer", None) is not None:
            self.sim.renderer.close()
        self._closed = True


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
