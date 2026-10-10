"""Reduced six-leg rigid-body mechanics using CPU MuJoCo.

This is an engineering prototype: its segment dimensions, inertias, fixed
moment arms, and joint sign conventions are explicit hypotheses, not measured
or biologically validated fly anatomy. It has no gait generator, pose writes,
or externally applied body traction. Muscle channels create joint torques;
ground forces come from MuJoCo contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "articulated_leg_body.json"
_JOINTS = (
    "coxa_yaw",
    "coxa_roll",
    "coxa_pitch",
    "femur_pitch",
    "femur_roll",
    "tibia_pitch",
    "tarsus_pitch",
)
_LEG_ANGLES_RAD = {
    "LF": math.radians(35.0),
    "LM": math.radians(90.0),
    "LH": math.radians(145.0),
    "RF": math.radians(-35.0),
    "RM": math.radians(-90.0),
    "RH": math.radians(-145.0),
}


@dataclass(frozen=True)
class MuscleDescriptor:
    """One functional muscle channel and its reduced fixed torque arms."""

    name: str
    leg: str
    group: str
    joint_moment_arms_m: Mapping[str, float]
    optimal_length_m: float
    notes: str
    anatomical_alias: str | None = None

    @property
    def moment_arms_m(self) -> Mapping[str, float]:
        """Short compatibility alias for callers that use moment_arms_m."""

        return self.joint_moment_arms_m


class ArticulatedLegBody:
    """CPU MuJoCo free-root fly body with six articulated seven-DOF legs.

    Args:
        config_path: JSON model and reduced-muscle configuration. Defaults to
            ``config/articulated_leg_body.json`` in this repository.
        fixed_root: Attach the thorax to the world, useful for isolated tests.
        friction: Optional isotropic Coulomb sliding coefficient. Zero is
            supported and is useful for checking whether motion comes only
            from internal joint forces.
        passive_support: Enable the configured engineering spring on all 42
            leg hinges. ``None`` uses the config default. Joint damping remains
            enabled when this spring term is disabled.
    """

    schema_version = 1

    def __init__(
        self,
        config_path: str | Path | None = None,
        fixed_root: bool = False,
        friction: float | None = None,
        passive_support: bool | None = None,
    ) -> None:
        try:
            import mujoco  # type: ignore[import-not-found]
            import numpy as np
        except ImportError as exc:  # keep importing this module dependency-light
            raise ImportError(
                "ArticulatedLegBody requires MuJoCo Python bindings; use the "
                "project CPU environment or install the mujoco package."
            ) from exc

        self._mujoco = mujoco
        self._np = np
        self._config_path = Path(config_path).expanduser() if config_path else _DEFAULT_CONFIG_PATH
        try:
            self._config = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read articulated body config: {self._config_path}") from exc
        if self._config.get("schema_version") != 1 or self._config.get("units") != "SI":
            raise ValueError("articulated body config must use schema_version 1 and SI units")

        self.fixed_root = bool(fixed_root)
        self._physics = self._config["physics"]
        self._geometry = self._config["geometry"]
        self._muscle_config = self._config["muscles"]
        self._base_timestep_s = float(self._physics["timestep_s"])
        self._gravity_mps2 = float(self._physics["gravity_mps2"])
        self._joint_armature = float(self._physics["joint_armature_kg_m2"])
        support_config = self._physics.get("passive_support", {})
        configured_support_enabled = support_config.get("enabled")
        if not isinstance(configured_support_enabled, bool):
            raise ValueError("physics.passive_support.enabled must be a boolean")
        if passive_support is not None and not isinstance(passive_support, bool):
            raise ValueError("passive_support override must be True, False, or None")
        self.passive_support = (
            configured_support_enabled if passive_support is None else passive_support
        )
        self.passive_support_enabled = self.passive_support
        self._joint_damping = self._finite_nonnegative(
            support_config.get("damping_Nms_rad", self._physics.get("joint_damping_Nms_rad")),
            "physics.passive_support.damping_Nms_rad",
        )
        self._passive_stiffness_Nm_rad = self._finite_nonnegative(
            support_config.get("stiffness_Nm_rad"), "physics.passive_support.stiffness_Nm_rad"
        )
        self._spring_reference_rad = self._finite_number(
            support_config.get("spring_reference_rad"), "physics.passive_support.spring_reference_rad"
        )
        solver_config = self._physics.get("contact_solver", {})
        self._contact_time_constant_s = self._finite_number(
            solver_config.get("time_constant_s"), "physics.contact_solver.time_constant_s"
        )
        self._contact_damping_ratio = self._finite_number(
            solver_config.get("damping_ratio"), "physics.contact_solver.damping_ratio"
        )
        self.friction = float(self._physics["friction_coefficient"] if friction is None else friction)
        if not math.isfinite(self.friction) or self.friction < 0:
            raise ValueError("friction must be a finite, non-negative coefficient")
        if not math.isfinite(self._base_timestep_s) or self._base_timestep_s <= 0:
            raise ValueError("configured timestep_s must be finite and positive")
        if self._contact_time_constant_s <= 0 or self._contact_damping_ratio <= 0:
            raise ValueError("contact solver time constant and damping ratio must be positive")
        if self._contact_time_constant_s < 2.0 * self._base_timestep_s:
            raise ValueError("contact solver time_constant_s must be at least twice timestep_s")

        self._legs = tuple(self._geometry["legs"])
        if set(self._legs) != set(_LEG_ANGLES_RAD) or len(self._legs) != 6:
            raise ValueError("the reduced model requires the six LF/LM/LH/RF/RM/RH legs")
        self._group_config = self._muscle_config["groups"]
        self.muscles: dict[str, MuscleDescriptor] = self._build_muscle_descriptors()

        self.mjcf = self._build_mjcf()
        self.mjcf_sha256 = hashlib.sha256(self.mjcf.encode("utf-8")).hexdigest()
        try:
            self.model = mujoco.MjModel.from_xml_string(self.mjcf)
        except Exception as exc:
            raise ValueError(f"MuJoCo rejected generated articulated body MJCF: {exc}") from exc
        self.data = mujoco.MjData(self.model)
        self._root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "thorax")
        self._root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
        self._joint_ids: dict[str, int] = {}
        self._joint_qposadr: dict[str, int] = {}
        self._joint_dofadr: dict[str, int] = {}
        self._foot_geom_ids: dict[str, int] = {}
        for leg in self._legs:
            for joint_suffix in _JOINTS:
                name = f"{leg}_{joint_suffix}"
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0:
                    raise ValueError(f"generated MJCF is missing joint {name}")
                self._joint_ids[name] = joint_id
                self._joint_qposadr[name] = int(self.model.jnt_qposadr[joint_id])
                self._joint_dofadr[name] = int(self.model.jnt_dofadr[joint_id])
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
            if foot_id < 0:
                raise ValueError(f"generated MJCF is missing foot geom {leg}_foot")
            self._foot_geom_ids[leg] = foot_id

        self._last_joint_torques_Nm = {name: 0.0 for name in self._joint_ids}
        self._last_muscle_forces_N = {name: 0.0 for name in self.muscles}
        self._joint_rest_q_rad: dict[str, float] = {}
        self._failed_state: str | None = None
        self.reset()

    def _build_muscle_descriptors(self) -> dict[str, MuscleDescriptor]:
        optimal_length = float(self._muscle_config["optimal_length_m"])
        if not math.isfinite(optimal_length) or optimal_length <= 0:
            raise ValueError("muscles.optimal_length_m must be finite and positive")
        descriptors: dict[str, MuscleDescriptor] = {}
        for leg in self._legs:
            for group, record in self._group_config.items():
                arms = record.get("joint_moment_arms_m", {})
                full_arms: dict[str, float] = {}
                for joint_suffix, arm in arms.items():
                    if joint_suffix not in _JOINTS:
                        raise ValueError(f"unknown joint {joint_suffix!r} in muscle group {group!r}")
                    value = float(arm)
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite moment arm in muscle group {group!r}")
                    full_arms[f"{leg}_{joint_suffix}"] = value
                name = f"{leg}_{group}"
                descriptors[name] = MuscleDescriptor(
                    name=name,
                    leg=leg,
                    group=group,
                    joint_moment_arms_m=full_arms,
                    optimal_length_m=optimal_length,
                    notes=str(record.get("notes", "")),
                    anatomical_alias=record.get("anatomical_alias"),
                )
        return descriptors

    @staticmethod
    def _f(value: float) -> str:
        return format(float(value), ".12g")

    @staticmethod
    def _finite_number(value: Any, label: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be finite") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        return number

    @classmethod
    def _finite_nonnegative(cls, value: Any, label: str) -> float:
        number = cls._finite_number(value, label)
        if number < 0:
            raise ValueError(f"{label} must be non-negative")
        return number

    def _build_mjcf(self) -> str:
        geom = self._geometry
        physics = self._physics
        vector = geom["segment_vectors_local_m"]
        masses = geom["segment_masses_kg"]
        radii = geom["segment_radii_m"]
        friction = self._f(self.friction)
        hinge_stiffness = self._passive_stiffness_Nm_rad if self.passive_support else 0.0
        stiffness_attr = (
            f' stiffness="{self._f(hinge_stiffness)}"'
            f' springref="{self._f(self._spring_reference_rad)}"'
            f' damping="{self._f(self._joint_damping)}"'
            f' armature="{self._f(self._joint_armature)}" frictionloss="0"'
        )
        body_height = self._f(physics["body_height_m"])
        body_inertia = " ".join(self._f(v) for v in physics["body_inertia_kg_m2"])
        thorax_radii = self._vec(geom["thorax_ellipsoid_semiaxes_m"])
        lines = [
            '<mujoco model="reduced_six_leg_articulated_body">',
            '  <compiler angle="radian" coordinate="local" inertiafromgeom="false" autolimits="true"/>',
            f'  <option timestep="{self._f(self._base_timestep_s)}" integrator="implicitfast" '
            f'gravity="0 0 {self._f(-self._gravity_mps2)}" cone="elliptic" iterations="60" tolerance="1e-10"/>',
            '  <size njmax="3000" nconmax="500"/>',
            '  <default>',
            '    <joint limited="true" damping="0" armature="0" frictionloss="0"/>',
            f'    <geom condim="3" friction="{friction} 0.005 0.0001" solref="{self._f(self._contact_time_constant_s)} {self._f(self._contact_damping_ratio)}" solimp="0.95 0.99 0.001"/>',
            '  </default>',
            '  <contact>',
        ]
        # Appendages originate inside the thorax shell. Exclude these
        # intentional attachment overlaps while preserving shell-ground and
        # foot-ground contact solving.
        for leg in self._legs:
            for body_suffix in ("carrier", "coxa", "femur", "tibia", "tarsus", "foot_body"):
                body_name = f"{leg}_{body_suffix}"
                lines.append(f'    <exclude body1="thorax" body2="{body_name}"/>')
        lines.extend(
            [
            '  </contact>',
            '  <worldbody>',
            '    <geom name="ground" type="plane" pos="0 0 0" size="0 0 0.1" '
            'condim="3" friction="' + friction + ' 0.005 0.0001" rgba="0.5 0.5 0.5 1"/>',
            f'    <body name="thorax" pos="0 0 {body_height}">',
            ]
        )
        if not self.fixed_root:
            lines.append('      <freejoint name="root_free"/>')
        lines.extend(
            [
                f'      <inertial pos="0 0 0" mass="{self._f(physics["body_mass_kg"])}" diaginertia="{body_inertia}"/>',
                f'      <geom name="thorax_shell" type="ellipsoid" size="{thorax_radii}" '
                'contype="1" conaffinity="1" condim="3" friction="' + friction +
                ' 0.005 0.0001" rgba="0.25 0.3 0.18 1"/>',
            ]
        )
        anchor_cfg = geom["anchors_m"]
        for leg in self._legs:
            stage = "front" if leg[1] == "F" else "middle" if leg[1] == "M" else "hind"
            side = "left" if leg[0] == "L" else "right"
            anchor_x = float(anchor_cfg[f"{stage}_x"])
            anchor_y = float(anchor_cfg[f"{side}_y"])
            yaw = _LEG_ANGLES_RAD[leg]
            yaw_text = self._f(yaw)
            # Joint coordinates describe homologous anatomical actions. Under
            # sagittal reflection, angular axes are pseudovectors: local yaw
            # and roll reverse, while pitch keeps its sign. Mirror the axes
            # so identical muscle moment arms remain valid on both sides.
            axial_sign = "1" if side == "left" else "-1"
            coxa = vector["coxa"]
            femur = vector["femur"]
            tibia = vector["tibia"]
            tarsus = vector["tarsus"]
            lines.extend(
                [
                    f'      <body name="{leg}_carrier" pos="{self._f(anchor_x)} {self._f(anchor_y)} 0" euler="0 0 {yaw_text}">',
                    f'        <body name="{leg}_coxa" pos="0 0 0">',
                    f'          <joint name="{leg}_coxa_yaw" type="hinge" axis="0 0 {axial_sign}" range="-1.2 1.2"{stiffness_attr}/>',
                    f'          <joint name="{leg}_coxa_roll" type="hinge" axis="{axial_sign} 0 0" range="-0.9 0.9"{stiffness_attr}/>',
                    f'          <joint name="{leg}_coxa_pitch" type="hinge" axis="0 1 0" range="-1.2 1.2"{stiffness_attr}/>',
                    self._segment_inertial("coxa", coxa, masses, radii),
                    f'          <geom name="{leg}_coxa_geom" type="capsule" fromto="0 0 0 {self._f(coxa[0])} {self._f(coxa[1])} {self._f(coxa[2])}" size="{self._f(radii["coxa"])}" rgba="0.65 0.49 0.26 1"/>',
                    f'          <body name="{leg}_femur" pos="{self._vec(coxa)}">',
                    f'            <joint name="{leg}_femur_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5"{stiffness_attr}/>',
                    f'            <joint name="{leg}_femur_roll" type="hinge" axis="{axial_sign} 0 0" range="-1.0 1.0"{stiffness_attr}/>',
                    self._segment_inertial("femur", femur, masses, radii),
                    f'            <geom name="{leg}_femur_geom" type="capsule" fromto="0 0 0 {self._f(femur[0])} {self._f(femur[1])} {self._f(femur[2])}" size="{self._f(radii["femur"])}" rgba="0.67 0.50 0.27 1"/>',
                    f'            <body name="{leg}_tibia" pos="{self._vec(femur)}">',
                    f'              <joint name="{leg}_tibia_pitch" type="hinge" axis="0 1 0" range="-1.6 1.6"{stiffness_attr}/>',
                    self._segment_inertial("tibia", tibia, masses, radii),
                    f'              <geom name="{leg}_tibia_geom" type="capsule" fromto="0 0 0 {self._f(tibia[0])} {self._f(tibia[1])} {self._f(tibia[2])}" size="{self._f(radii["tibia"])}" rgba="0.72 0.56 0.31 1"/>',
                    f'              <body name="{leg}_tarsus" pos="{self._vec(tibia)}">',
                    f'                <joint name="{leg}_tarsus_pitch" type="hinge" axis="0 1 0" range="-1.6 1.6"{stiffness_attr}/>',
                    self._segment_inertial("tarsus", tarsus, masses, radii),
                    f'                <geom name="{leg}_tarsus_geom" type="capsule" fromto="0 0 0 {self._f(tarsus[0])} {self._f(tarsus[1])} {self._f(tarsus[2])}" size="{self._f(radii["tarsus"])}" rgba="0.78 0.62 0.37 1"/>',
                    f'                <body name="{leg}_foot_body" pos="{self._vec(tarsus)}">',
                    f'                  <geom name="{leg}_foot" type="sphere" size="{self._f(geom["foot_radius_m"])}" '
                    'condim="3" friction="' + friction + ' 0.005 0.0001" rgba="0.18 0.2 0.16 1"/>',
                    '                </body>',
                    '              </body>',
                    '            </body>',
                    '          </body>',
                    '        </body>',
                    '      </body>',
                ]
            )
        lines.extend(['    </body>', '  </worldbody>', '</mujoco>'])
        return "\n".join(lines)

    def _segment_inertial(
        self,
        name: str,
        vector: list[float],
        masses: Mapping[str, float],
        radii: Mapping[str, float],
    ) -> str:
        mass = float(masses[name])
        length = math.sqrt(sum(float(v) ** 2 for v in vector))
        radius = float(radii[name])
        # A finite-cylinder approximation with positive principal inertias.
        # MuJoCo rejects moving bodies with principal inertia <= mjMINVAL
        # (1e-15 kg m^2); the floor is recorded as a numerical-scale hypothesis.
        axial = max(0.5 * mass * radius * radius, 1.1e-15)
        transverse = max(mass * (length * length / 12.0 + radius * radius / 4.0), 1.1e-15)
        inertia = f"{self._f(transverse)} {self._f(max(transverse, axial))} {self._f(transverse)}"
        pos = self._vec([float(v) * 0.5 for v in vector])
        return f'          <inertial pos="{pos}" mass="{self._f(mass)}" diaginertia="{inertia}"/>'

    def _vec(self, values: list[float] | tuple[float, ...]) -> str:
        return " ".join(self._f(v) for v in values)

    def muscle_kinematics(self) -> dict[str, dict[str, float]]:
        """Return constant-moment-arm fiber length and shortening velocity.

        Length is ``optimal_length_m - sum(r * (q - q_rest))`` and velocity is
        ``-sum(r * qd)``. Positive generalized torque and shortening therefore
        have matching moment-arm signs. It is a kinematic proxy, not a tendon
        path reconstruction or Hill-type muscle model.
        """

        result: dict[str, dict[str, float]] = {}
        for name, descriptor in self.muscles.items():
            length = descriptor.optimal_length_m
            velocity = 0.0
            for joint_name, arm in descriptor.joint_moment_arms_m.items():
                qadr = self._joint_qposadr[joint_name]
                dadr = self._joint_dofadr[joint_name]
                q = float(self.data.qpos[qadr])
                qd = float(self.data.qvel[dadr])
                q_rest = self._joint_rest_q_rad[joint_name]
                length -= float(arm) * (q - q_rest)
                velocity -= float(arm) * qd
            result[name] = {"length_m": length, "velocity_mps": velocity}
        return result

    def step(self, muscle_forces_N: Mapping[str, float], dt_s: float) -> dict[str, Any]:
        """Advance rigid-body/contact physics for ``dt_s`` with held muscle force.

        Muscle tension is non-negative. Each channel applies only
        ``moment_arm * force`` to its named hinge's ``qfrc_applied`` entry; the
        six free-root DOFs receive no muscle wrench. A fractional last physics
        substep is used when ``dt_s`` is not a multiple of the configured
        100-microsecond timestep.
        """

        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt_s must be finite and positive")
        if self._failed_state is not None:
            raise RuntimeError(f"articulated body is fail-closed after a prior physics fault: {self._failed_state}")
        unknown = set(muscle_forces_N) - set(self.muscles)
        if unknown:
            raise KeyError(f"unknown muscle channels: {sorted(unknown)[:5]}")
        forces = {name: 0.0 for name in self.muscles}
        for name, raw_force in muscle_forces_N.items():
            force = float(raw_force)
            if not math.isfinite(force) or force < 0:
                raise ValueError(f"muscle force for {name} must be finite and non-negative")
            forces[name] = force

        torques = {joint: 0.0 for joint in self._joint_ids}
        for name, force in forces.items():
            if force == 0:
                continue
            for joint_name, arm in self.muscles[name].joint_moment_arms_m.items():
                torques[joint_name] += float(arm) * force
        self.data.qfrc_applied[:] = 0.0
        for joint_name, torque in torques.items():
            self.data.qfrc_applied[self._joint_dofadr[joint_name]] = torque
        self._last_joint_torques_Nm = torques
        self._last_muscle_forces_N = forces

        time_before = float(self.data.time)
        if time_before + dt <= time_before:
            raise ValueError("dt_s is too small to advance the MuJoCo simulation clock")
        warning_baseline = self.data.warning.number.copy()
        remaining = dt
        try:
            while remaining > 0.0:
                substep = min(self._base_timestep_s, remaining)
                if substep == remaining:
                    substep = remaining
                self.model.opt.timestep = substep
                substep_time = float(self.data.time)
                self._mujoco.mj_step(self.model, self.data)
                self._check_state_health(warning_baseline, substep_time)
                if substep == remaining:
                    remaining = 0.0
                else:
                    remaining -= substep
        finally:
            self.model.opt.timestep = self._base_timestep_s
        # Refresh contacts and their solved forces at the returned state.
        self._mujoco.mj_forward(self.model, self.data)
        self._check_state_health(warning_baseline, time_before + dt)
        expected_time = time_before + dt
        time_tolerance = max(1e-18, 8.0 * math.ulp(expected_time))
        if float(self.data.time) <= time_before or abs(float(self.data.time) - expected_time) > time_tolerance:
            self._failed_state = f"time mismatch: expected {expected_time:.17g}, got {self.data.time:.17g}"
            raise RuntimeError(f"MuJoCo physics step failed closed: {self._failed_state}")
        return self._telemetry()

    def _check_state_health(self, warning_baseline: Any, expected_time_s: float) -> None:
        warning_indices = {
            int(self._mujoco.mjtWarning.mjWARN_BADQPOS): "BADQPOS",
            int(self._mujoco.mjtWarning.mjWARN_BADQVEL): "BADQVEL",
            int(self._mujoco.mjtWarning.mjWARN_BADQACC): "BADQACC",
        }
        warning_counts = self.data.warning.number
        for warning_index, warning_name in warning_indices.items():
            if int(warning_counts[warning_index]) > int(warning_baseline[warning_index]):
                self._failed_state = f"MuJoCo warning {warning_name}"
                raise RuntimeError(f"MuJoCo physics step failed closed: {self._failed_state}")
        if float(self.data.time) < expected_time_s - max(1e-15, abs(expected_time_s) * 1e-13):
            self._failed_state = f"physics time moved backward at {self.data.time:.17g}s"
            raise RuntimeError(f"MuJoCo physics step failed closed: {self._failed_state}")
        for label, state in (("qpos", self.data.qpos), ("qvel", self.data.qvel), ("qacc", self.data.qacc)):
            if not bool(self._np.isfinite(state).all()):
                self._failed_state = f"non-finite {label}"
                raise RuntimeError(f"MuJoCo physics step failed closed: {self._failed_state}")

    def reset(self, snapshot_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Reset to the rest configuration or restore a compatible snapshot."""

        if snapshot_state is None:
            self._mujoco.mj_resetData(self.model, self.data)
            self.model.opt.timestep = self._base_timestep_s
            self._failed_state = None
            self._joint_rest_q_rad = {
                name: float(self.data.qpos[qadr]) for name, qadr in self._joint_qposadr.items()
            }
        else:
            self._failed_state = None
            if snapshot_state.get("schema_version") != self.schema_version:
                raise ValueError("snapshot schema_version is not supported")
            if snapshot_state.get("mjcf_sha256") != self.mjcf_sha256:
                raise ValueError("snapshot MJCF hash does not match this model")
            qpos = snapshot_state.get("qpos")
            qvel = snapshot_state.get("qvel")
            if not isinstance(qpos, list) or len(qpos) != self.model.nq:
                raise ValueError("snapshot qpos has the wrong shape")
            if not isinstance(qvel, list) or len(qvel) != self.model.nv:
                raise ValueError("snapshot qvel has the wrong shape")
            self._mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            self.data.time = float(snapshot_state.get("time_s", 0.0))
            warmstart = snapshot_state.get("qacc_warmstart")
            if isinstance(warmstart, list) and len(warmstart) == self.model.nv:
                self.data.qacc_warmstart[:] = warmstart
            rest = snapshot_state.get("joint_rest_q_rad", {})
            self._joint_rest_q_rad = {
                name: float(rest.get(name, self.data.qpos[qadr]))
                for name, qadr in self._joint_qposadr.items()
            }
        self.data.qfrc_applied[:] = 0.0
        self._last_joint_torques_Nm = {name: 0.0 for name in self._joint_ids}
        self._last_muscle_forces_N = {name: 0.0 for name in self.muscles}
        self._mujoco.mj_forward(self.model, self.data)
        return self._telemetry()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable physics state and the corresponding telemetry."""

        return {
            "schema_version": self.schema_version,
            "mjcf_sha256": self.mjcf_sha256,
            "time_s": float(self.data.time),
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "qacc_warmstart": self.data.qacc_warmstart.tolist(),
            "joint_rest_q_rad": dict(self._joint_rest_q_rad),
            "telemetry": self._telemetry(),
        }

    def provenance(self) -> dict[str, Any]:
        """Describe the model source, units, evidence boundary, and channel map."""

        support_config = self._physics["passive_support"]
        effective_stiffness = self._passive_stiffness_Nm_rad if self.passive_support else 0.0
        return {
            "model": "reduced_six_leg_articulated_body",
            "implementation": "CPU MuJoCo rigid-body/contact model",
            "mujoco_version": getattr(self._mujoco, "__version__", "unknown"),
            "fixed_root": self.fixed_root,
            "units": "SI",
            "joint_coordinate_convention": {
                "bilateral_actions": "equal homologous joint coordinates produce sagittally mirrored leg poses",
                "right_yaw_roll_axes": "reversed local angular axes; pitch axes unchanged",
                "muscle_moment_arms": "shared signs in anatomical joint coordinates",
                "evidence": "engineering mirror consistency; anatomical action signs remain hypotheses",
            },
            "timestep_s": self._base_timestep_s,
            "friction_coefficient": self.friction,
            "physics_parameters": dict(self._physics),
            "effective_physics_parameters": {
                "timestep_s": self._base_timestep_s,
                "gravity_mps2": self._gravity_mps2,
                "friction_coefficient": self.friction,
                "joint_damping_Nms_rad": self._joint_damping,
                "joint_armature_kg_m2": self._joint_armature,
                "contact_solver_time_constant_s": self._contact_time_constant_s,
                "contact_solver_damping_ratio": self._contact_damping_ratio,
            },
            "contact_solver": {
                "time_constant_s": self._contact_time_constant_s,
                "damping_ratio": self._contact_damping_ratio,
                "constraint": "time_constant_s >= 2 * timestep_s",
            },
            "passive_support": {
                "enabled": self.passive_support,
                "configured_enabled": support_config["enabled"],
                "classification": support_config["classification"],
                "stiffness_Nm_rad": effective_stiffness,
                "configured_stiffness_Nm_rad": self._passive_stiffness_Nm_rad,
                "spring_reference_rad": self._spring_reference_rad,
                "joint_damping_Nms_rad": self._joint_damping,
                "damping_Nms_rad": self._joint_damping,
                "spring_scope": "42 articulated leg hinge joints only; no root spring",
                "damping_scope": "42 articulated leg hinge joints only",
                "evidence_label": support_config["evidence_label"],
                "notes": support_config["notes"],
                "context_source": support_config["context_source"],
            },
            "total_model_mass_kg": float(self.model.body_mass.sum()),
            "segment_inertia_floor_kg_m2": 1.1e-15,
            "config_path": str(self._config_path),
            "config_sha256": hashlib.sha256(self._config_path.read_bytes()).hexdigest(),
            "mjcf_sha256": self.mjcf_sha256,
            "muscle_channel_count": len(self.muscles),
            "muscle_groups": list(self._group_config),
            "joint_moment_arm_hypotheses_m": {
                group: dict(record.get("joint_moment_arms_m", {}))
                for group, record in self._group_config.items()
            },
            "muscle_mapping_evidence": {
                group: {k: record[k] for k in (
                    "anatomical_action", "action_evidence", "moment_arm_evidence",
                    "numerical_proxy", "omitted_route", "notes") if k in record}
                for group, record in self._group_config.items()
            },
            "muscle_kinematics": "constant moment arms relative to reset pose; linear length proxy",
            "anatomy_evidence": "engineering hypotheses; not measured or biologically validated",
            "passive_support_evidence_boundary": (
                "The engineering spring is not measured biology or neural tonus; "
                "passive elasticity alone is reported insufficient for natural standing."
            ),
            "physics_authority": "native hinge springs/damping, muscle joint torques, and MuJoCo contact forces only; no root spring, gravity compensation, gait, or body-position writes",
        }

    def _telemetry(self) -> dict[str, Any]:
        mujoco = self._mujoco
        root_position = self.data.xpos[self._root_body_id].tolist()
        root_quaternion = self.data.xquat[self._root_body_id].tolist()
        if self._root_joint_id >= 0:
            qadr = int(self.model.jnt_qposadr[self._root_joint_id])
            dadr = int(self.model.jnt_dofadr[self._root_joint_id])
            root_linear_velocity = self.data.qvel[dadr : dadr + 3].tolist()
            root_angular_velocity = self.data.qvel[dadr + 3 : dadr + 6].tolist()
            root_position = self.data.qpos[qadr : qadr + 3].tolist()
            root_quaternion = self.data.qpos[qadr + 3 : qadr + 7].tolist()
        else:
            root_linear_velocity = [0.0, 0.0, 0.0]
            root_angular_velocity = [0.0, 0.0, 0.0]

        joints: dict[str, dict[str, float]] = {}
        effective_stiffness = self._passive_stiffness_Nm_rad if self.passive_support else 0.0
        passive_elastic_energy = 0.0
        passive_damping_power = 0.0
        for name, qadr in self._joint_qposadr.items():
            dadr = self._joint_dofadr[name]
            q = float(self.data.qpos[qadr])
            qd = float(self.data.qvel[dadr])
            spring_torque = -effective_stiffness * (q - self._spring_reference_rad)
            damping_torque = -self._joint_damping * qd
            passive_elastic_energy += 0.5 * effective_stiffness * (q - self._spring_reference_rad) ** 2
            passive_damping_power += self._joint_damping * qd * qd
            joints[name] = {
                "q_rad": q,
                "qd_rad_s": qd,
                "torque_Nm": float(self._last_joint_torques_Nm.get(name, 0.0)),
                "muscle_torque_Nm": float(self._last_joint_torques_Nm.get(name, 0.0)),
                "passive_spring_torque_Nm": spring_torque,
                "passive_damping_torque_Nm": damping_torque,
            }

        if self._root_joint_id >= 0:
            root_dadr = int(self.model.jnt_dofadr[self._root_joint_id])
            root_applied = self.data.qfrc_applied[root_dadr : root_dadr + 6]
            root_applied_force = [float(v) for v in root_applied[:3]]
            root_applied_torque = [float(v) for v in root_applied[3:6]]
        else:
            root_applied_force = [0.0, 0.0, 0.0]
            root_applied_torque = [0.0, 0.0, 0.0]

        ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        feet = {
            leg: {
                "position_m": self.data.geom_xpos[self._foot_geom_ids[leg]].tolist(),
                "contact": False,
                "normal_force_N": 0.0,
                "tangential_force_N": [0.0, 0.0, 0.0],
                "contact_force_N": [0.0, 0.0, 0.0],
            }
            for leg in self._legs
        }
        body_contacts: list[dict[str, Any]] = []
        total_ground_normal = 0.0
        any_nonfoot_ground_contact = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            local_force = self._np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, index, local_force)
            frame = contact.frame.reshape(3, 3)
            world_force = frame.T @ local_force[:3]
            force_on_geom2 = [float(v) for v in world_force]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or str(geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or str(geom2)
            body1_id = int(self.model.geom_bodyid[geom1])
            body2_id = int(self.model.geom_bodyid[geom2])
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or str(body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or str(body2_id)
            foot_leg = next(
                (leg for leg, foot_geom_id in self._foot_geom_ids.items() if foot_geom_id in (geom1, geom2)),
                None,
            )
            is_ground_contact = ground_geom_id in (geom1, geom2)
            is_foot_ground = bool(is_ground_contact and foot_leg is not None)
            body_contacts.append(
                {
                    "geom1": geom1_name,
                    "geom2": geom2_name,
                    "body1": body1_name,
                    "body2": body2_name,
                    "ground_contact": is_ground_contact,
                    "foot_ground_contact": is_foot_ground,
                    "force_on_geom2_N": force_on_geom2,
                }
            )
            if is_ground_contact:
                total_ground_normal += max(0.0, force_on_geom2[2] if geom1 == ground_geom_id else -force_on_geom2[2])
                if not is_foot_ground:
                    any_nonfoot_ground_contact = True
            if not is_foot_ground or foot_leg is None:
                continue
            feet[foot_leg]["contact"] = True
            foot_force = world_force if geom2 == self._foot_geom_ids[foot_leg] else -world_force
            force = [float(v) for v in foot_force]
            normal_force = max(0.0, force[2])
            tangential = [force[0], force[1], force[2] - normal_force]
            for axis in range(3):
                feet[foot_leg]["contact_force_N"][axis] += force[axis]
                feet[foot_leg]["tangential_force_N"][axis] += tangential[axis]
            feet[foot_leg]["normal_force_N"] += normal_force

        total_normal = sum(float(foot["normal_force_N"]) for foot in feet.values())
        total_tangential = [
            sum(float(foot["tangential_force_N"][axis]) for foot in feet.values())
            for axis in range(3)
        ]
        total_tangential_magnitude = math.sqrt(sum(v * v for v in total_tangential))
        total_tangential_sum = sum(
            math.sqrt(sum(float(v) ** 2 for v in foot["tangential_force_N"])) for foot in feet.values()
        )

        total_mass = 0.0
        com = [0.0, 0.0, 0.0]
        momentum = [0.0, 0.0, 0.0]
        for body_id in range(1, self.model.nbody):
            mass = float(self.model.body_mass[body_id])
            if mass <= 0:
                continue
            total_mass += mass
            position = self.data.xipos[body_id]
            jacp = self._np.zeros((3, self.model.nv), dtype=float)
            jacr = self._np.zeros((3, self.model.nv), dtype=float)
            mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, body_id)
            velocity = jacp @ self.data.qvel
            for axis in range(3):
                com[axis] += mass * float(position[axis])
                momentum[axis] += mass * float(velocity[axis])
        if total_mass > 0:
            com = [value / total_mass for value in com]

        full_mass = self._np.zeros((self.model.nv, self.model.nv), dtype=float)
        mujoco.mj_fullM(self.model, full_mass, self.data.qM)
        qvel = self.data.qvel
        kinetic = 0.5 * float(qvel @ full_mass @ qvel)
        potential = 0.0
        for body_id in range(1, self.model.nbody):
            mass = float(self.model.body_mass[body_id])
            potential += mass * self._gravity_mps2 * float(self.data.xipos[body_id, 2])

        return {
            "time_s": float(self.data.time),
            "root": {
                "position_m": [float(v) for v in root_position],
                "quaternion_wxyz": [float(v) for v in root_quaternion],
                "linear_velocity_mps": [float(v) for v in root_linear_velocity],
                "angular_velocity_radps": [float(v) for v in root_angular_velocity],
            },
            "root_applied_wrench": {
                "available": self._root_joint_id >= 0,
                "force_N": root_applied_force,
                "torque_Nm": root_applied_torque,
                "max_abs_generalized_force": max(
                    [abs(v) for v in root_applied_force + root_applied_torque], default=0.0
                ),
                "source": "MuJoCo qfrc_applied at root_free DOFs",
            },
            "joints": joints,
            "feet": feet,
            "body_contacts": body_contacts,
            "body_ground_contact": any_nonfoot_ground_contact,
            "center_of_mass_position_m": [float(v) for v in com],
            "linear_momentum_kg_mps": [float(v) for v in momentum],
            "total_foot_normal_force_N": total_normal,
            "total_foot_tangential_force_N": total_tangential,
            "total_foot_tangential_force_magnitude_N": total_tangential_magnitude,
            "total_foot_tangential_force_sum_N": total_tangential_sum,
            "total_ground_normal_force_N": total_ground_normal,
            "contact_summary": {
                "normal_force_N": total_normal,
                "tangential_force_N": total_tangential,
                "tangential_force_magnitude_N": total_tangential_magnitude,
            },
            "applied_muscle_forces_N": dict(self._last_muscle_forces_N),
            "passive_support": {
                "enabled": self.passive_support,
                "stiffness_Nm_rad": effective_stiffness,
                "spring_reference_rad": self._spring_reference_rad,
                "joint_damping_Nms_rad": self._joint_damping,
                "elastic_energy_J": passive_elastic_energy,
                "damping_power_W": passive_damping_power,
            },
            "energy": {
                "kinetic_J": float(kinetic),
                "potential_J": float(potential),
                "passive_elastic_J": passive_elastic_energy,
                "total_J": float(kinetic + potential + passive_elastic_energy),
            },
        }
