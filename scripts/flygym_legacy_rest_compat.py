"""Legacy-rest reference-state factory for the pinned FlyGym v2 walker.

This paired-ablation factory keeps the modern NEUTRAL pose as the initial qpos
for active joints and as the POSITION-actuator neutral input. It restores the
official v1.2.1 leg-joint spring references to zero, and sets passive distal
tarsus2–5 neutral-keyframe qpos to zero. Geometry, joint axes, gains, damping,
adhesion, and trajectory/CPG configuration remain those of the v2 builder.

Provenance: official FlyGym v1.2.1 source commit
fec6ac70775e57984f8bc2237024e2a535ab1ed1 has zero leg spring references;
pinned v2.1.0 commit ca65a510c2afe6ac61c51df4f274c8d190c2f95f derives both
springref and neutral qpos from KinematicPosePreset.NEUTRAL in
flygym/compose/fly/base_fly.py:add_joints/_get_neutral_qpos. The v2 preset's
passive tarsus2–5 values are −5 degrees. This factory pairs the v1 spring
reference with a neutral startup keyframe whose active qpos is still the v2
baseline; it does not reproduce the v1 "stretch" startup pose or prove a
dynamic cause for prior failures.
"""

from __future__ import annotations

from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    JointPreset,
    PASSIVE_TARSAL_LINKS,
    Skeleton,
)
from flygym.compose import ActuatorType, KinematicPosePreset, NeuroMechFly


LEGACY_REST_COMPATIBILITY_PROVENANCE = {
    "v1_source_commit": "fec6ac70775e57984f8bc2237024e2a535ab1ed1",
    "pinned_v2_source_commit": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
    "leg_joint_springref_rad": 0.0,
    "passive_tarsus2345_neutral_qpos_rad": 0.0,
    "active_qpos_and_position_neutral_input": "pinned-v2 NEUTRAL preset",
}


def make_legacy_rest_locomotion_fly(
    name: str = "nmf",
    *,
    joint_stiffness: float = 0.05,
    joint_damping: float = 0.06,
    passive_tarsus_stiffness: float = 7.5,
    passive_tarsus_damping: float = 1e-2,
    actuator_gain: float = 45.0,
    actuator_forcerange: tuple[float, float] = (-65.0, 65.0),
    add_adhesion: bool = True,
    adhesion_gain: float = 40.0,
    colorize: bool = False,
) -> NeuroMechFly:
    """Create the standard v2 locomotion fly with legacy zero spring refs."""
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        AxisOrder.YAW_PITCH_ROLL
    )
    neutral_lookup = neutral_pose.joint_angles_lookup_rad
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    fly = NeuroMechFly(name=name)
    joints = fly.add_joints(
        skeleton,
        neutral_pose=None,
        stiffness=joint_stiffness,
        damping=joint_damping,
    )

    for jointdof, joint in joints.items():
        # v1 uses zero spring references for all leg hinges. Keep the v2 active
        # startup pose separately in jointdof_to_neutralangle (below).
        joint.springref = 0.0
        fly.jointdof_to_neutralangle[jointdof] = (
            0.0
            if jointdof.child.link in PASSIVE_TARSAL_LINKS
            else neutral_lookup.get(jointdof.name, 0.0)
        )
        if jointdof.child.link in PASSIVE_TARSAL_LINKS:
            # MuJoCo 3.7+ stores these polynomial coefficients in arrays.
            joint.stiffness[0] = passive_tarsus_stiffness
            joint.damping[0] = passive_tarsus_damping

    # Rebuild the source keyframe before it is attached/compiled into the world.
    fly._rebuild_neutral_keyframe()
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=actuator_gain,
        forcerange=actuator_forcerange,
    )
    if add_adhesion:
        fly.add_leg_adhesion(gain=adhesion_gain)
    if colorize:
        fly.colorize()
    return fly
