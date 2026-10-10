"""Preparation-only source for a tethered, mechanically active FlyGym ball.

This file deliberately has no CLI or main entrypoint.  It defines a small
world-composition seam and a guarded binding seam for the accepted V3 AP
endpoint controller.  Do not instantiate/compile the world or start physics
until the pilot specification has been reviewed.

All dimensions/masses follow the installed FlyGym model convention (mm, g,
seconds; see the pinned model's -9810 mm/s^2 gravity).  The values below are
engineering presets for a first mechanics screen, not a reconstruction or
calibration of Mano et al.'s treadmill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from flygym.compose.physics import ContactParams
from flygym.compose.world.base_world import _GroundContactMixin
from flygym.compose.world.tethered_world import TetheredWorld


V3_AP_CONTROLLER_SHA256 = (
    "689364bc661c4fc37a15ddc617a398450a3f52d2fe4da7d202b74687cda91b8f"
)
V4_1_BASE_CONTROLLER_SHA256 = (
    "126b641e2998992e4cad351f4a48292882063851350e4006ba40552a38d7e2eb"
)


@dataclass(frozen=True)
class EngineeringBallPreset:
    """Uncalibrated first-pass ball and contact values in FlyGym native units."""

    name: str = "engineering_5mm_10mg_mu0p8_v1"
    radius_mm: float = 5.0
    mass_g: float = 0.010
    center_world_mm: tuple[float, float, float] = (0.0, 0.0, -5.0)
    sliding_friction: float = 0.8
    torsional_friction: float = 0.002
    rolling_friction: float = 0.0001
    contact_margin_mm: float = 0.001

    @property
    def solid_sphere_inertia_g_mm2(self) -> float:
        return 0.4 * self.mass_g * self.radius_mm**2


ENGINEERING_BALL_PRESET = EngineeringBallPreset()


class TetheredBallWorld(_GroundContactMixin, TetheredWorld):
    """Tethered FlyGym world with a finite-inertia, fixed-center sphere.

    The ball is a child body of the static worldbody with a MuJoCo ball joint:
    its center cannot translate, while its quaternion has three rotational
    DoFs.  FlyGym's explicit contact-pair machinery is used to connect all
    selected leg geoms to this one sphere geom.  The ball joint is returned as
    a world DoF so BaseWorld's neutral keyframe initializes its quaternion.

    Construction of this class is intentionally not invoked by this module.
    ``BaseWorld.add_fly`` compiles a model while rebuilding the neutral
    keyframe; that remains behind the explicit review gate.
    """

    def __init__(
        self,
        preset: EngineeringBallPreset = ENGINEERING_BALL_PRESET,
        *,
        name: str = "tethered_ball_pilot_world",
    ) -> None:
        if preset.radius_mm <= 0 or preset.mass_g <= 0:
            raise ValueError("ball radius and mass must be positive")
        TetheredWorld.__init__(self, name=name)
        self.preset = preset
        ball_body = self.mjcf_root.worldbody.add_body(
            name="treadmill_ball_body",
            pos=preset.center_world_mm,
        )
        self.ball_joint = ball_body.add_joint(
            name="treadmill_ball_spin",
            type=mj.mjtJoint.mjJNT_BALL,
            damping=0.0,
        )
        self.ball_geom = ball_body.add_geom(
            name="treadmill_ball_geom",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=(preset.radius_mm, 0.0, 0.0),
            mass=preset.mass_g,
            contype=0,
            conaffinity=0,
            group=3,
        )
        # FlyGym Simulation indexes this list for its contact/ground observers.
        # There is exactly one support surface here: the treadmill sphere.
        self.ground_geoms = [self.ball_geom]
        self.leg_contact_params = ContactParams(
            sliding_friction=preset.sliding_friction,
            torsional_friction=preset.torsional_friction,
            rolling_friction=preset.rolling_friction,
            margin=preset.contact_margin_mm,
        )

    def _attach_fly_mjcf(
        self,
        fly: Any,
        spawn_position: Any,
        spawn_rotation: Any,
        *,
        bodysegs_with_ground_contact: list[Any] | None = None,
        ground_contact_params: ContactParams | None = None,
        add_ground_contact_sensors: bool = True,
        **kwargs: Any,
    ) -> set[str]:
        """Attach the fly by its mocap root and declare foot–sphere pairs."""
        if kwargs:
            raise TypeError(f"unsupported tethered-ball options: {sorted(kwargs)}")
        # Call this concrete base explicitly: the mixin's implementation would
        # create a free root, which is the opposite of the tethered experiment.
        TetheredWorld._attach_fly_mjcf(
            self, fly, spawn_position, spawn_rotation
        )
        segments = (
            list(bodysegs_with_ground_contact)
            if bodysegs_with_ground_contact is not None
            else [segment for segment in fly.get_bodysegs_order() if segment.is_leg()]
        )
        if not segments or any(not segment.is_leg() for segment in segments):
            raise ValueError("this pilot permits leg segments as ball contacts only")
        params = ground_contact_params or self.leg_contact_params
        _GroundContactMixin._set_ground_contact(self, fly, segments, params)
        if add_ground_contact_sensors:
            _GroundContactMixin._add_ground_contact_sensors(self, fly, segments)
        return {self.ball_joint.name}


def bind_accepted_v3_ap_controller(session: Any) -> Any:
    """Bind the already-reviewed V3 AP wrapper to a compatible pilot session.

    A complete tethered session adapter is intentionally not provided here:
    the existing reference session assumes a free root and flat plane.  This
    helper only prevents accidental use of a changed controller source once a
    reviewed session supplies the controller's existing interface.
    """
    here = Path(__file__).resolve().parent
    ap_path = here / "flygym_ap_endpoint_gesture_prototype.py"
    v41_path = here / "flygym_phase_speed_gait_v4_1.py"
    actual_ap = hashlib.sha256(ap_path.read_bytes()).hexdigest()
    actual_v41 = hashlib.sha256(v41_path.read_bytes()).hexdigest()
    if actual_ap != V3_AP_CONTROLLER_SHA256:
        raise RuntimeError(f"V3 AP controller source hash changed: {actual_ap}")
    if actual_v41 != V4_1_BASE_CONTROLLER_SHA256:
        raise RuntimeError(f"V4.1 base controller source hash changed: {actual_v41}")

    from flygym_ap_endpoint_gesture_prototype import (
        APEndpointGestureConfig,
        APEndpointGestureController,
    )

    controller = APEndpointGestureController(session, APEndpointGestureConfig())
    session.controller = controller
    return controller


def collect_native_measurement(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    ball_joint_id: int,
    ball_body_id: int,
    ball_geom_id: int,
    fly_root_body_id: int,
    thorax_body_id: int,
    head_body_id: int,
    position_actuator_ids: np.ndarray,
    position_actuator_names: list[str],
    private_fk_data: mj.MjData,
) -> dict[str, Any]:
    """Collect one already-existing native state without advancing dynamics.

    Contacts retain ordered geom identities and their raw contact-frame wrench;
    this intentionally does not guess which body's contact frame wrench sign
    corresponds to ball torque. A reviewed postprocessor can derive world
    force/moment and torque about the ball center after checking that convention.
    """
    if len(position_actuator_ids) != 42 or len(position_actuator_names) != 42:
        raise ValueError("the candidate FlyGym model must expose 42 ordered position actuators")
    qpos_adr = int(model.jnt_qposadr[ball_joint_id])
    dof_adr = int(model.jnt_dofadr[ball_joint_id])
    ball_qpos = np.asarray(data.qpos[qpos_adr : qpos_adr + 4], dtype=np.float64).copy()
    ball_qvel = np.asarray(data.qvel[dof_adr : dof_adr + 3], dtype=np.float64).copy()
    ball_velocity_world = np.zeros(6, dtype=np.float64)
    mj.mj_objectVelocity(
        model,
        data,
        mj.mjtObj.mjOBJ_BODY,
        int(ball_body_id),
        ball_velocity_world,
        0,
    )

    contacts: list[dict[str, Any]] = []
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if ball_geom_id not in (geom1, geom2):
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mj.mj_contactForce(model, data, contact_id, wrench)
        contacts.append(
            {
                "contact_id": contact_id,
                "geom1_id": geom1,
                "geom1_name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom1),
                "geom2_id": geom2,
                "geom2_name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom2),
                "position_world_mm": np.asarray(contact.pos, dtype=np.float64).tolist(),
                "frame_world_axes_flat": np.asarray(contact.frame, dtype=np.float64).tolist(),
                "normal_world_from_contact_frame_axis0": np.asarray(
                    contact.frame[:3], dtype=np.float64
                ).tolist(),
                "distance_mm": float(contact.dist),
                "raw_contact_frame_force_native": wrench[:3].tolist(),
                "raw_contact_frame_torque_native": wrench[3:].tolist(),
                "force_torque_sign_interpretation": "not assigned; preserve geom ordering",
            }
        )

    mocap_id = int(model.body_mocapid[fly_root_body_id])
    if mocap_id < 0:
        raise ValueError("tethered fly root is not a mocap body")
    fk_mocap_matches = bool(
        np.array_equal(data.mocap_pos[mocap_id], private_fk_data.mocap_pos[mocap_id])
        and np.array_equal(data.mocap_quat[mocap_id], private_fk_data.mocap_quat[mocap_id])
    )
    position_actuator_ids = np.asarray(position_actuator_ids, dtype=np.int32)
    return {
        "ball_joint_qpos_quaternion_wxyz": ball_qpos.tolist(),
        "ball_joint_qvel_rad_s": ball_qvel.tolist(),
        "ball_joint_applied_torque_native": np.asarray(
            data.qfrc_applied[dof_adr : dof_adr + 3], dtype=np.float64
        ).tolist(),
        "ball_body_world_angular_velocity_rad_s": ball_velocity_world[:3].tolist(),
        "ball_body_world_linear_velocity_mm_s": ball_velocity_world[3:].tolist(),
        "ball_center_world_mm": np.asarray(data.xpos[ball_body_id], dtype=np.float64).tolist(),
        "ball_external_wrench_native": np.asarray(
            data.xfrc_applied[ball_body_id], dtype=np.float64
        ).tolist(),
        "ball_contacts": contacts,
        "thorax_position_world_mm": np.asarray(data.xpos[thorax_body_id], dtype=np.float64).tolist(),
        "thorax_quaternion_world_wxyz": np.asarray(data.xquat[thorax_body_id], dtype=np.float64).tolist(),
        "head_position_world_mm": np.asarray(data.xpos[head_body_id], dtype=np.float64).tolist(),
        "head_quaternion_world_wxyz": np.asarray(data.xquat[head_body_id], dtype=np.float64).tolist(),
        "root_external_wrench_native": np.asarray(
            data.xfrc_applied[fly_root_body_id], dtype=np.float64
        ).tolist(),
        "full_model_generalized_force_applied_native": np.asarray(
            data.qfrc_applied, dtype=np.float64
        ).tolist(),
        "root_mocap_position_world_mm": np.asarray(data.mocap_pos[mocap_id], dtype=np.float64).tolist(),
        "root_mocap_quaternion_world_wxyz": np.asarray(data.mocap_quat[mocap_id], dtype=np.float64).tolist(),
        "private_fk_mocap_position_world_mm": np.asarray(
            private_fk_data.mocap_pos[mocap_id], dtype=np.float64
        ).tolist(),
        "private_fk_mocap_quaternion_world_wxyz": np.asarray(
            private_fk_data.mocap_quat[mocap_id], dtype=np.float64
        ).tolist(),
        "private_fk_mocap_exact_match": fk_mocap_matches,
        "position_actuator_names_ordered": list(position_actuator_names),
        "position_actuator_ctrl_ordered": np.asarray(
            data.ctrl[position_actuator_ids], dtype=np.float64
        ).tolist(),
        "root_state_write_counts": "must be recorded by the session at each write site",
    }


def pilot_recording_contract() -> dict[str, Any]:
    """Return the measurement contract; this function performs no model work."""
    return {
        "status": "preparation_only_review_required_before_world_construction",
        "ball": {
            **asdict(ENGINEERING_BALL_PRESET),
            "solid_sphere_inertia_g_mm2": ENGINEERING_BALL_PRESET.solid_sphere_inertia_g_mm2,
            "center_translation_locked": True,
            "initial_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "externally_prescribed_rotation": False,
        },
        "fly": {
            "root": "mocap-fixed thorax; no free root joint",
            "head": "fixed relative to thorax in the reused LEGS_ONLY model",
            "external_root_force_or_state_write": False,
            "spawn_position_mm": [0.0, 0.0, 0.7],
            "spawn_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "controller_candidate": {
            "name": "APEndpointGestureController over PhaseSpeedGaitControllerV4_1",
            "actuator_channels": 42,
            "preserve_native_adhesion": True,
            "ap_endpoint_path": "selected-leg IK on native six-leg phase/template targets",
            "neural_output_used": False,
        },
        "later_mechanical_sign_screen": {
            "leg_power_P": 0.30,
            "differential_drive_values": [0.0, 0.5, -0.5],
            "d_to_left_right_mapping": "must be explicitly bound and logged after session review",
            "yaw_or_ball_turn_sign_fit": False,
        },
        "required_saved_measurements": [
            "ball_joint_qpos_quaternion_wxyz",
            "ball_joint_qvel_rad_s",
            "ball_body_world_angular_velocity_rad_s",
            "per-contact ordered geom IDs, world position/normal, and raw contact-frame force/torque",
            "per-contact moment about ball center after contact-wrench sign convention is audited",
            "fixed thorax and head world pose (position and quaternion)",
            "mocap position/quaternion copied into the controller private FK data",
            "42 actuator target names/order and per-tick target values",
            "native timestep, control tick, initial state, and zero external root/ball input audits",
        ],
        "run_gate": "root review required before composing/compiling a model or advancing physics",
    }
