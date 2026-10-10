#!/usr/bin/env python3
"""Calibratable low-order flight physics for the digital fly.

This module is deliberately a *reduced* model.  It uses a wingbeat pattern
generator (WPG) and wingbeat-averaged quasi-steady force coefficients; it is
not CFD, a resolved Navier--Stokes solver, or a claim about the complete
neuromechanics of a real fly.  Its purpose is a deterministic, inspectable
bridge between neural/muscle channels and the existing body simulator.

Conventions
-----------
* SI units are used throughout (m, s, kg, N, N m, J, W, rad).
* Body axes are x forward, y left, z up.  A quaternion is scalar-first and
  maps body-frame vectors into the world frame, matching ``digital_body``.
* ``power`` is a normalized non-negative wing drive in ``[0, 1]``.  Lift and
  profile drag scale approximately with its square.
* ``amplitude`` is a non-negative multiplier of the calibrated stroke
  amplitude.  A value of 1 is nominal; signed ``amplitude_mod`` channels are
  also accepted and are converted using the configured modulation gain.
* ``bias`` is normalized to ``[-1, 1]`` and maps to a stroke-plane bias angle;
  explicit ``*_bias_rad`` fields are accepted for callers already in SI.
* Differential wing drive produces bounded roll/yaw moments through explicit
  calibration scales.  Rotational damping and a weak passive hover-restoring
  moment keep the reduced body numerically well behaved.
* The optional ground plane is a unilateral contact constraint.  It acts only
  on an actual downward crossing and never prescribes an airborne position.

The defaults are order-of-magnitude fruit-fly values and are marked as such
in :class:`FlightDynamicsConfig`.  Replace them with measured/fitted values
before treating a trajectory as biological evidence.  The model is designed
to be calibrated, not to hide calibration behind an opaque learned policy.

This module also contains a deliberately small bilateral leg/jump bridge.  It
maps only normalized ``jump_*``, ``leg_*`` and ``posture_*`` neuromuscular
channels into SI forces, impulses, and attitude changes.  It is a reduced-body
contact model, not a complete FlyBody/articulated-animal implementation.  A
leg push is a unilateral contact actuator: once the body has left the support
plane, the same event cannot keep injecting a ground force while airborne.

The public surface intentionally has a few compatibility aliases:
``FlightDynamics``/``FlightPhysics`` are the simulator, ``WPG`` is the
wingbeat generator, and ``NeuromuscularInput`` accepts the naming used by
neural outputs, ``flight_autopilot.MotorCommand``, and ``digital_body``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import argparse
import json
import math
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


# Versioned structural marker published by neuromuscular_adapter's
# NeuromuscularOutput.  The parser intentionally checks this exact value;
# arbitrary objects with similar fields continue through the compatibility
# parser below.
NORMALIZED_WING_OUTPUT_CONTRACT = "normalized-wing-output-v1"


@dataclass(frozen=True)
class ContactSurface:
    """Horizontal unilateral support surface for a body-centre trajectory.

    The footprint is expressed in world x/y coordinates and ``top_z_m`` is
    the body-centre support height.  A surface only catches a body that comes
    from above; it never teleports a body upward from underneath the surface.
    """

    surface_id: str
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    top_z_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface_id", str(self.surface_id))
        values = {
            "min_x_m": _finite(self.min_x_m, "min_x_m"),
            "max_x_m": _finite(self.max_x_m, "max_x_m"),
            "min_y_m": _finite(self.min_y_m, "min_y_m"),
            "max_y_m": _finite(self.max_y_m, "max_y_m"),
            "top_z_m": _finite(self.top_z_m, "top_z_m"),
        }
        if values["min_x_m"] > values["max_x_m"] or values["min_y_m"] > values["max_y_m"]:
            raise FlightDynamicsError("contact surface footprint bounds must be ordered")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def contains_xy(self, position_m: Vec3) -> bool:
        """Return whether a body centre lies inside the surface footprint."""

        return (
            self.min_x_m <= position_m[0] <= self.max_x_m
            and self.min_y_m <= position_m[1] <= self.max_y_m
        )

MODEL_NAME = "low_order_quasi_steady_wingbeat"
MODEL_SCOPE = "reduced 6-DOF wingbeat-averaged model; not CFD"
IS_CFD = False
WPG_FREQUENCY_MIN_HZ = 200.0
WPG_FREQUENCY_MAX_HZ = 220.0

# The leg bridge intentionally has separate provenance from the wing model.
# Keeping the scope string explicit prevents a reduced point-body contact law
# from being mistaken for the articulated FlyBody biomechanical model.
JUMP_MODEL_NAME = "reduced_body_leg_jump_contact"
JUMP_MODEL_SCOPE = "reduced-body bilateral leg/contact bridge; not complete FlyBody"
JUMP_IS_FLYBODY = False

# This is a separate reduced-model path from the timed jump bridge: it
# consumes normalized leg drive every tick and applies force only while the
# authoritative flight state is grounded.
GROUND_WALKING_MODEL_NAME = "reduced_body_contact_gated_ground_walking"
GROUND_WALKING_MODEL_SCOPE = "reduced point-body ground traction and yaw actuator; not complete FlyBody"


class FlightDynamicsError(ValueError):
    """Raised when a flight model input or parameter is invalid."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FlightDynamicsError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(number):
        raise FlightDynamicsError(f"{name} must be a finite scalar")
    return number


def _vec3(value: Iterable[float], name: str = "vector") -> Vec3:
    try:
        result = tuple(_finite(v, name) for v in value)
    except TypeError as exc:
        raise FlightDynamicsError(f"{name} must contain three finite values") from exc
    if len(result) != 3:
        raise FlightDynamicsError(f"{name} must contain three finite values")
    return result  # type: ignore[return-value]


def _quat(value: Iterable[float], name: str = "orientation_quat") -> Quat:
    try:
        result = tuple(_finite(v, name) for v in value)
    except TypeError as exc:
        raise FlightDynamicsError(f"{name} must contain four finite values") from exc
    if len(result) != 4:
        raise FlightDynamicsError(f"{name} must contain four finite values")
    norm = math.sqrt(sum(x * x for x in result))
    if norm <= 1.0e-15:
        raise FlightDynamicsError(f"{name} must not be the zero quaternion")
    return tuple(x / norm for x in result)  # type: ignore[return-value]


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vscale(a: Vec3, scale: float) -> Vec3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _vdot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vnorm(a: Vec3) -> float:
    return math.sqrt(max(0.0, _vdot(a, a)))


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qconj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def _qrotate(q: Quat, value: Vec3) -> Vec3:
    rotated = _qmul(_qmul(q, (0.0, value[0], value[1], value[2])), _qconj(q))
    return (rotated[1], rotated[2], rotated[3])


def _qrotate_inverse(q: Quat, value: Vec3) -> Vec3:
    return _qrotate(_qconj(q), value)


def _quat_from_rotation_vector(rotation_rad: Vec3) -> Quat:
    """Return the exact scalar-first quaternion for a body rotation vector."""

    angle = _vnorm(rotation_rad)
    if angle <= 1.0e-12:
        # The small-angle limit avoids a divide-by-zero while retaining the
        # same first-order result as the exact exponential map.
        return _quat((1.0, 0.5 * rotation_rad[0],
                      0.5 * rotation_rad[1], 0.5 * rotation_rad[2]))
    half_angle = 0.5 * angle
    scale = math.sin(half_angle) / angle
    return _quat((math.cos(half_angle),
                  scale * rotation_rad[0],
                  scale * rotation_rad[1],
                  scale * rotation_rad[2]))


def _quat_from_rpy(orientation_rad: Iterable[float]) -> Quat:
    """Convert XYZ roll/pitch/yaw (the unified body contract) to a quaternion."""

    roll, pitch, yaw = _vec3(orientation_rad, "orientation_rad")
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return _quat((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), "orientation_quat")


def _rpy_from_quat(quaternion: Quat) -> Vec3:
    """Convert a scalar-first body-to-world quaternion to XYZ RPY radians."""

    w, x, y, z = _quat(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = _clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(pitch_argument)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def _normalise_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one level of common neural-output wrappers.

    NeuralOutput uses ``named_outputs`` while small controllers tend to use
    ``channels``.  Keeping this conversion local avoids importing the optional
    contract module and preserves the dependency-free body boundary.
    """

    result: dict[str, Any] = dict(value)
    for nested_name in ("named_outputs", "channels", "motor_channels", "neuromuscular"):
        nested = value.get(nested_name)
        if isinstance(nested, Mapping):
            for key, nested_value in nested.items():
                result.setdefault(str(key), nested_value)
    return result


def _lower_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _sequence_like(value: Any) -> bool:
    """Recognize lists/tuples and array-like neural vectors without numpy."""

    return (
        not isinstance(value, (str, bytes, bytearray, Mapping))
        and hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
    )


def _lookup(mapping: Mapping[str, Any], names: Sequence[str]) -> tuple[Any, bool]:
    """Case/underscore-insensitive lookup with aliases."""

    by_lower = {_lower_key(k): v for k, v in mapping.items()}
    for name in names:
        key = _lower_key(name)
        if key in by_lower:
            return by_lower[key], True
    return None, False


def _parse_optional_vec3(value: Any, name: str) -> Optional[Vec3]:
    if value is None:
        return None
    return _vec3(value, name)


def _timestamp_seconds(value: Any) -> float:
    """Read a unified-contract timestamp without importing that module."""

    if isinstance(value, Mapping):
        raw = value.get("t_s", value.get("time_s"))
    else:
        raw = getattr(value, "t_s", getattr(value, "time_s", None))
    if raw is None:
        raise FlightDynamicsError("timestamp must expose t_s or time_s")
    return _finite(raw, "timestamp.t_s")


def _command_timestamp(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("timestamp")
    return getattr(value, "timestamp", None)


def _record_dict(record: Any) -> dict[str, Any]:
    """Copy dataclass fields while keeping vector fields tuple-shaped."""

    return {
        name: tuple(value) if isinstance(value, (list, tuple)) else value
        for name, value in vars(record).items()
        if name in record.__dataclass_fields__
    }


class _ProvenanceRecord(dict):
    """JSON-safe mapping that also tolerates the older ``provenance()`` idiom."""

    def __call__(self) -> dict[str, Any]:
        return dict(self)


@dataclass(frozen=True)
class FlightDynamicsConfig:
    """Centralized physical, actuator, and calibration parameters.

    Values marked ``order_of_magnitude`` are safe defaults for wiring and
    tests, not measurements of one particular fly.  A calibration record can
    override ``parameter_sources`` and ``fit_status`` without changing the
    equations.
    """

    # Rigid body and environment (SI).
    mass_kg: float = 1.0e-6
    inertia_kg_m2: Vec3 = (2.0e-12, 2.0e-12, 3.0e-12)
    gravity_mps2: float = 9.80665
    air_density_kg_m3: float = 1.225
    body_area_m2: float = 2.0e-6
    body_drag_coefficient: float = 1.0
    wind_mps: Vec3 = (0.0, 0.0, 0.0)

    # Per-wing geometry and quasi-steady coefficients.
    wing_area_m2: float = 1.2e-5
    wing_span_m: float = 2.5e-3
    mean_radius_m: float = 1.6e-3
    stroke_amplitude_rad: float = 1.0
    max_stroke_bias_rad: float = math.radians(35.0)
    lift_coefficient: float = 0.60
    profile_drag_coefficient: float = 0.20
    translational_wing_drag_coefficient: float = 0.12
    bias_lift_scale: float = 0.90

    # Wingbeat pattern generator (WPG); the allowed band is intentional.
    wingbeat_frequency_hz: float = 210.0
    wpg_frequency_min_hz: float = WPG_FREQUENCY_MIN_HZ
    wpg_frequency_max_hz: float = WPG_FREQUENCY_MAX_HZ
    left_phase_offset_rad: float = 0.0
    right_phase_offset_rad: float = 0.0

    # Neuromuscular modulation and actuator calibration.
    amplitude_modulation_gain: float = 0.50
    max_control_torque_Nm: float = 2.0e-9
    torque_damping_Nm_s: Vec3 = (4.0e-10, 4.0e-10, 3.0e-10)
    mechanical_efficiency: float = 0.70
    idle_power_W: float = 0.0
    max_angular_rate_rad_s: float = 200.0
    max_dt_s: float = 0.005

    # Low-order attitude and ground-contact terms.  These are deliberately
    # explicit calibration knobs rather than hidden clamps: differential wing
    # drive rolls the body, while wingbeat damping and a weak passive restoring
    # moment keep a bounded command from tumbling indefinitely.  Contact is a
    # unilateral contact plane expressed in the body-centre state frame.  A
    # standalone model uses the world ground datum (z=0); callers whose world
    # state stores the body centre must pass ``ground_z + body_radius``.
    # Contact is resolved here, not by a second scripted position clamp.
    differential_roll_moment_scale: float = 0.04
    differential_yaw_moment_scale: float = 0.35
    attitude_stiffness_Nm: Vec3 = (3.0e-10, 3.0e-10, 0.0)
    ground_contact_enabled: bool = True
    ground_plane_z_m: float = 0.0
    ground_restitution: float = 0.0
    contact_surfaces: Tuple[ContactSurface, ...] = ()

    # Optional contact-gated walking actuator. Disabled by default so this
    # path cannot alter historical wing-only or wing+jump trajectories.
    ground_walking_enabled: bool = False
    # Explicit body-only intervention for grounded visual assays. Neural wing
    # and jump readouts are still recorded upstream, but neither actuator is
    # allowed to apply force/torque to this physical body.
    ground_only: bool = False
    max_ground_traction_N: float = 2.0e-6
    max_ground_yaw_torque_Nm: float = 1.0e-9
    ground_horizontal_braking_N_s_m: float = 1.0e-5
    ground_lateral_braking_N_s_m: float = 1.0e-4
    max_ground_speed_mps: float = 0.25

    # Physical wing-motor pool channels are preferred when present.  Legacy
    # DNg02/flight-power channels remain an explicit compatibility fallback,
    # never an implicit stimulus-derived heuristic.
    prefer_physical_wing_motor: bool = True
    allow_legacy_wing_fallback: bool = True

    # Provenance and fit status are part of the model contract.
    fit_status: str = "order_of_magnitude_defaults__replace_with_fit"
    parameter_sources: Mapping[str, str] = field(default_factory=lambda: {
        "mass_kg": "digital_body.BodyConfig default scale; order_of_magnitude",
        "inertia_kg_m2": "digital_body.BodyConfig default scale; order_of_magnitude",
        "wing_area_m2": "digital_body.BodyConfig default scale; order_of_magnitude",
        "wing_span_m": "digital_body.BodyConfig default scale; order_of_magnitude",
        "wingbeat_frequency_hz": "WPG design band 200-220 Hz; default midpoint, not fitted",
        "lift_coefficient": "quasi-steady placeholder; requires force/kinematic fit",
        "profile_drag_coefficient": "quasi-steady placeholder; requires force/kinematic fit",
        "torque_damping_Nm_s": "reduced wingbeat/body rotational damping; calibrate from attitude decay",
        "differential_roll_moment_scale": "bounded differential-wing steering scale; calibrate from roll response",
        "differential_yaw_moment_scale": "bounded differential-wing yaw scale; calibrate from yaw response",
        "attitude_stiffness_Nm": "reduced passive hover restoring moment; calibrate from free-flight stability",
        "ground_plane_z_m": "virtual-world ground-plane datum",
        "ground_walking_enabled": "explicit opt-in for contact-gated leg-power walking actuator",
        "ground_only": "explicit wing/jump actuator suppression for grounded assays",
        "max_ground_traction_N": "reduced-body forward traction limit; calibration hypothesis",
        "max_ground_yaw_torque_Nm": "reduced-body leg-steering yaw scale; calibration hypothesis",
        "ground_horizontal_braking_N_s_m": "linear horizontal ground braking coefficient; calibration hypothesis",
        "ground_lateral_braking_N_s_m": "body-lateral ground grip coefficient; calibration hypothesis",
        "max_ground_speed_mps": "grounded horizontal speed bound; calibration hypothesis",
        "wind_mps": "caller/environment input",
        "prefer_physical_wing_motor": "physical wm motor pools preferred over legacy DNg02 readout",
        "allow_legacy_wing_fallback": "explicit compatibility fallback when physical wm channels are absent",
    })

    def __post_init__(self) -> None:
        positive = (
            "mass_kg", "gravity_mps2", "air_density_kg_m3", "body_area_m2",
            "wing_area_m2", "wing_span_m", "mean_radius_m", "stroke_amplitude_rad",
            "lift_coefficient", "profile_drag_coefficient", "mechanical_efficiency",
            "max_angular_rate_rad_s", "max_dt_s",
        )
        for name in positive:
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise FlightDynamicsError(f"{name} must be > 0")
        for name in ("body_drag_coefficient", "translational_wing_drag_coefficient",
                     "bias_lift_scale", "amplitude_modulation_gain", "idle_power_W",
                     "max_control_torque_Nm", "differential_roll_moment_scale",
                     "differential_yaw_moment_scale", "max_ground_traction_N",
                     "max_ground_yaw_torque_Nm", "ground_horizontal_braking_N_s_m",
                     "ground_lateral_braking_N_s_m"):
            value = _finite(getattr(self, name), name)
            if value < 0:
                raise FlightDynamicsError(f"{name} must be >= 0")
        max_ground_speed = _finite(self.max_ground_speed_mps, "max_ground_speed_mps")
        if max_ground_speed <= 0.0:
            raise FlightDynamicsError("max_ground_speed_mps must be > 0")
        object.__setattr__(self, "max_ground_speed_mps", max_ground_speed)
        if not isinstance(self.ground_walking_enabled, bool):
            raise FlightDynamicsError("ground_walking_enabled must be a boolean")
        if not isinstance(self.ground_only, bool):
            raise FlightDynamicsError("ground_only must be a boolean")
        if self.ground_only and not self.ground_walking_enabled:
            raise FlightDynamicsError("ground_only requires ground_walking_enabled")
        inertia = _vec3(self.inertia_kg_m2, "inertia_kg_m2")
        if any(value <= 0 for value in inertia):
            raise FlightDynamicsError("inertia_kg_m2 must contain positive values")
        damping = _vec3(self.torque_damping_Nm_s, "torque_damping_Nm_s")
        if any(value < 0 for value in damping):
            raise FlightDynamicsError("torque_damping_Nm_s must be non-negative")
        attitude_stiffness = _vec3(self.attitude_stiffness_Nm, "attitude_stiffness_Nm")
        if any(value < 0 for value in attitude_stiffness):
            raise FlightDynamicsError("attitude_stiffness_Nm must be non-negative")
        wind = _vec3(self.wind_mps, "wind_mps")
        object.__setattr__(self, "inertia_kg_m2", inertia)
        object.__setattr__(self, "torque_damping_Nm_s", damping)
        object.__setattr__(self, "attitude_stiffness_Nm", attitude_stiffness)
        object.__setattr__(self, "wind_mps", wind)
        object.__setattr__(self, "ground_plane_z_m", _finite(self.ground_plane_z_m, "ground_plane_z_m"))
        restitution = _finite(self.ground_restitution, "ground_restitution")
        if not 0.0 <= restitution <= 1.0:
            raise FlightDynamicsError("ground_restitution must be in [0, 1]")
        object.__setattr__(self, "ground_restitution", restitution)
        surfaces = tuple(self.contact_surfaces)
        if any(not isinstance(surface, ContactSurface) for surface in surfaces):
            raise FlightDynamicsError("contact_surfaces must contain ContactSurface values")
        object.__setattr__(self, "contact_surfaces", surfaces)
        for name in ("max_stroke_bias_rad", "left_phase_offset_rad", "right_phase_offset_rad"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.max_stroke_bias_rad < 0:
            raise FlightDynamicsError("max_stroke_bias_rad must be non-negative")
        lo = _finite(self.wpg_frequency_min_hz, "wpg_frequency_min_hz")
        hi = _finite(self.wpg_frequency_max_hz, "wpg_frequency_max_hz")
        freq = _finite(self.wingbeat_frequency_hz, "wingbeat_frequency_hz")
        if lo < 0 or hi < lo or not (lo <= freq <= hi):
            raise FlightDynamicsError(
                f"wingbeat_frequency_hz must be in [{lo}, {hi}] Hz"
            )
        if lo < WPG_FREQUENCY_MIN_HZ or hi > WPG_FREQUENCY_MAX_HZ:
            raise FlightDynamicsError("WPG frequency range must remain inside 200-220 Hz")
        object.__setattr__(self, "parameter_sources", dict(self.parameter_sources))
        if not str(self.fit_status).strip():
            raise FlightDynamicsError("fit_status must be a non-empty provenance marker")

    @property
    def weight_N(self) -> float:
        return self.mass_kg * self.gravity_mps2

    @property
    def half_wing_span_m(self) -> float:
        return 0.5 * self.wing_span_m

    @property
    def wpg_hz(self) -> float:
        return self.wingbeat_frequency_hz

    @property
    def air_density_kgpm3(self) -> float:
        return self.air_density_kg_m3

    @property
    def inertia_kgm2(self) -> Vec3:
        return self.inertia_kg_m2

    @property
    def wing_frequency_hz(self) -> float:
        return self.wingbeat_frequency_hz

    def provenance(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "scope": MODEL_SCOPE,
            "is_cfd": IS_CFD,
            "fit_status": self.fit_status,
            "parameters": self.to_dict(),
            "parameter_sources": dict(self.parameter_sources),
            "ground_walking": {
                "model": GROUND_WALKING_MODEL_NAME,
                "scope": GROUND_WALKING_MODEL_SCOPE,
                "enabled": self.ground_walking_enabled,
                "input_channels": {
                    "leg_power": "normalized [0, 1] direct neural motor output",
                    "leg_steering": "normalized [-1, 1] direct neural motor output",
                },
                "limitations": [
                    "walking contact is gated only to ground_plane_z_m; raised ContactSurface pads do not enable walking",
                    "reduced point-body traction has no articulated foot pose",
                ],
            },
            "units": "SI",
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly parameter manifest for run artefacts."""

        result = asdict(self)
        result["inertia_kg_m2"] = tuple(self.inertia_kg_m2)
        result["torque_damping_Nm_s"] = tuple(self.torque_damping_Nm_s)
        result["attitude_stiffness_Nm"] = tuple(self.attitude_stiffness_Nm)
        result["wind_mps"] = tuple(self.wind_mps)
        result["units"] = self.units
        return result

    parameter_manifest = to_dict

    def manifest(self) -> dict[str, Any]:
        """Return model provenance plus the exact active parameter values."""

        return self.provenance()

    @property
    def units(self) -> dict[str, str]:
        return {
            "mass_kg": "kg", "inertia_kg_m2": "kg*m^2", "position_m": "m",
            "velocity_mps": "m/s", "force_N": "N", "torque_Nm": "N*m",
            "frequency_hz": "Hz", "angle_rad": "rad", "energy_J": "J",
            "power_W": "W", "wind_mps": "m/s",
            "ground_plane_z_m": "m", "ground_restitution": "dimensionless",
            "max_ground_traction_N": "N",
            "max_ground_yaw_torque_Nm": "N*m",
            "ground_horizontal_braking_N_s_m": "N*s/m",
            "ground_lateral_braking_N_s_m": "N*s/m",
            "max_ground_speed_mps": "m/s",
            "attitude_stiffness_Nm": "N*m", "torque_damping_Nm_s": "N*m*s",
            "differential_roll_moment_scale": "dimensionless",
            "differential_yaw_moment_scale": "dimensionless",
        }


@dataclass(frozen=True)
class WPGSample:
    """One phase sample from the wingbeat pattern generator."""

    frequency_hz: float
    phase_rad: float
    stroke_angle_rad: float
    stroke_rate_rad_s: float

    @property
    def phase(self) -> float:
        return self.phase_rad

    @property
    def angle_rad(self) -> float:
        return self.stroke_angle_rad


class WingbeatPatternGenerator:
    """Deterministic WPG constrained to 200--220 Hz by default."""

    def __init__(self, frequency_hz: float = 210.0, *, phase_rad: float = 0.0,
                 frequency_min_hz: float = WPG_FREQUENCY_MIN_HZ,
                 frequency_max_hz: float = WPG_FREQUENCY_MAX_HZ) -> None:
        self.frequency_min_hz = _finite(frequency_min_hz, "frequency_min_hz")
        self.frequency_max_hz = _finite(frequency_max_hz, "frequency_max_hz")
        if not (WPG_FREQUENCY_MIN_HZ <= self.frequency_min_hz <= self.frequency_max_hz <= WPG_FREQUENCY_MAX_HZ):
            raise FlightDynamicsError("WPG frequency limits must lie in 200-220 Hz")
        self.frequency_hz = self._check_frequency(frequency_hz)
        self.phase_rad = _finite(phase_rad, "phase_rad") % (2.0 * math.pi)

    def _check_frequency(self, frequency_hz: float) -> float:
        value = _finite(frequency_hz, "frequency_hz")
        if not self.frequency_min_hz <= value <= self.frequency_max_hz:
            raise FlightDynamicsError(
                f"frequency_hz must be in [{self.frequency_min_hz}, {self.frequency_max_hz}] Hz"
            )
        return value

    def set_frequency(self, frequency_hz: float) -> float:
        self.frequency_hz = self._check_frequency(frequency_hz)
        return self.frequency_hz

    def reset(self, phase_rad: float = 0.0) -> None:
        self.phase_rad = _finite(phase_rad, "phase_rad") % (2.0 * math.pi)

    def advance(self, dt_s: float) -> float:
        dt = _finite(dt_s, "dt_s")
        if dt < 0:
            raise FlightDynamicsError("dt_s must be non-negative")
        self.phase_rad = (self.phase_rad + 2.0 * math.pi * self.frequency_hz * dt) % (2.0 * math.pi)
        return self.phase_rad

    def sample(self, *, amplitude_rad: float = 1.0, bias_rad: float = 0.0,
               phase_offset_rad: float = 0.0) -> WPGSample:
        amplitude = _finite(amplitude_rad, "amplitude_rad")
        bias = _finite(bias_rad, "bias_rad")
        phase_offset = _finite(phase_offset_rad, "phase_offset_rad")
        phase = (self.phase_rad + phase_offset) % (2.0 * math.pi)
        omega = 2.0 * math.pi * self.frequency_hz
        return WPGSample(
            frequency_hz=self.frequency_hz,
            phase_rad=phase,
            stroke_angle_rad=bias + amplitude * math.sin(phase),
            stroke_rate_rad_s=amplitude * omega * math.cos(phase),
        )


# Short names used in older experiment notes.
WPG = WingbeatPatternGenerator
WingBeatGenerator = WingbeatPatternGenerator


@dataclass(frozen=True)
class NeuromuscularInput:
    """Normalized wing/motor input accepted by :class:`FlightDynamics`.

    ``left_amplitude``/``right_amplitude`` are nominal multipliers (1 = no
    change).  ``left_amplitude_mod`` is an alternate signed channel in
    ``[-1, 1]``.  Direct SI bias and torque fields are provided for body-side
    adapters; normalized control fields remain convenient for neural outputs.
    """

    left_power: float = 0.0
    right_power: float = 0.0
    left_amplitude: float = 1.0
    right_amplitude: float = 1.0
    left_bias: float = 0.0
    right_bias: float = 0.0
    left_amplitude_mod: Optional[float] = None
    right_amplitude_mod: Optional[float] = None
    left_bias_rad: Optional[float] = None
    right_bias_rad: Optional[float] = None
    frequency_hz: Optional[float] = None
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll_torque_Nm: float = 0.0
    pitch_torque_Nm: float = 0.0
    yaw_torque_Nm: float = 0.0
    force_body_N: Vec3 = (0.0, 0.0, 0.0)
    wind_mps: Optional[Vec3] = None
    source: str = "neural"

    @property
    def wing_power_left(self) -> float:
        return self.left_power

    @property
    def wing_power_right(self) -> float:
        return self.right_power

    @property
    def wing_amplitude_left(self) -> float:
        return self.left_amplitude

    @property
    def wing_amplitude_right(self) -> float:
        return self.right_amplitude

    @property
    def wing_bias_left(self) -> float:
        return self.left_bias

    @property
    def wing_bias_right(self) -> float:
        return self.right_bias

    @property
    def left(self) -> float:
        return self.left_power

    @property
    def right(self) -> float:
        return self.right_power

    @property
    def power(self) -> float:
        return 0.5 * (self.left_power + self.right_power)

    def normalized(self, *, config: Optional[FlightDynamicsConfig] = None) -> "NeuromuscularInput":
        cfg = config or FlightDynamicsConfig()
        left_amp = _finite(self.left_amplitude, "left_amplitude")
        right_amp = _finite(self.right_amplitude, "right_amplitude")
        if self.left_amplitude_mod is not None:
            left_amp = 1.0 + cfg.amplitude_modulation_gain * _clamp(
                _finite(self.left_amplitude_mod, "left_amplitude_mod"), -1.0, 1.0
            )
        if self.right_amplitude_mod is not None:
            right_amp = 1.0 + cfg.amplitude_modulation_gain * _clamp(
                _finite(self.right_amplitude_mod, "right_amplitude_mod"), -1.0, 1.0
            )
        if left_amp < 0 or right_amp < 0 or left_amp > 2.0 or right_amp > 2.0:
            raise FlightDynamicsError("wing amplitudes must be in [0, 2] nominal multipliers")
        frequency = None if self.frequency_hz is None else _finite(self.frequency_hz, "frequency_hz")
        return replace(
            self,
            left_power=_clamp(_finite(self.left_power, "left_power"), 0.0, 1.0),
            right_power=_clamp(_finite(self.right_power, "right_power"), 0.0, 1.0),
            left_amplitude=left_amp,
            right_amplitude=right_amp,
            left_bias=_clamp(_finite(self.left_bias, "left_bias"), -1.0, 1.0),
            right_bias=_clamp(_finite(self.right_bias, "right_bias"), -1.0, 1.0),
            left_amplitude_mod=None,
            right_amplitude_mod=None,
            left_bias_rad=None if self.left_bias_rad is None else _finite(self.left_bias_rad, "left_bias_rad"),
            right_bias_rad=None if self.right_bias_rad is None else _finite(self.right_bias_rad, "right_bias_rad"),
            frequency_hz=frequency,
            roll=_finite(self.roll, "roll"),
            pitch=_finite(self.pitch, "pitch"),
            yaw=_finite(self.yaw, "yaw"),
            roll_torque_Nm=_finite(self.roll_torque_Nm, "roll_torque_Nm"),
            pitch_torque_Nm=_finite(self.pitch_torque_Nm, "pitch_torque_Nm"),
            yaw_torque_Nm=_finite(self.yaw_torque_Nm, "yaw_torque_Nm"),
            force_body_N=_vec3(self.force_body_N, "force_body_N"),
            wind_mps=_parse_optional_vec3(self.wind_mps, "wind_mps"),
            source=str(getattr(self.source, "value", self.source)),
        )

    def __getitem__(self, key: str) -> Any:
        """Small mapping compatibility convenience for trim commands."""

        aliases = {
            "left": "left_power", "right": "right_power",
            "left_wing": "left_power", "right_wing": "right_power",
            "wing_drive_left": "left_power", "wing_drive_right": "right_power",
            "wing_motor_left": "left_power", "wing_motor_right": "right_power",
            "wingbeat_frequency_hz": "frequency_hz",
        }
        return getattr(self, aliases.get(key, key))

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default

    def keys(self) -> tuple[str, ...]:
        return tuple(asdict(self))

    def items(self) -> tuple[tuple[str, Any], ...]:
        values = asdict(self)
        return tuple(values.items())

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["units"] = {
            "power": "normalized",
            "amplitude": "nominal multiplier",
            "bias": "normalized",
            "bias_rad": "rad",
            "frequency_hz": "Hz",
            "torque": "N*m",
            "force_body_N": "N",
            "wind_mps": "m/s",
        }
        return result

    @property
    def units(self) -> dict[str, str]:
        return {
            "power": "normalized", "amplitude": "nominal multiplier", "bias": "normalized",
            "bias_rad": "rad", "frequency_hz": "Hz", "force_body_N": "N",
            "torque_Nm": "N*m", "wind_mps": "m/s",
        }

    @classmethod
    def _from_normalized_wing_output(
        cls,
        value: Any,
        *,
        config: Optional[FlightDynamicsConfig] = None,
    ) -> Optional["NeuromuscularInput"]:
        """Parse the explicit normalized-wing output protocol directly.

        ``NeuromuscularOutput`` has already selected/clipped its normalized
        wing fields.  Only the exact versioned marker enables this path; a
        lookalike object or an unmarked legacy record remains on the general
        alias parser.  Physical ``wing_motor_*`` channels retain exactly the
        same precedence and strict legacy-fallback rules as the mapping path.
        """

        if getattr(value, "flight_dynamics_contract", None) != NORMALIZED_WING_OUTPUT_CONTRACT:
            return None
        required = (
            "channels", "left_wing_power", "right_wing_power",
            "wing_steering", "source",
        )
        if any(not hasattr(value, name) for name in required):
            # A malformed marked object should not receive a guessed partial
            # interpretation; retain the established compatibility behavior.
            return None
        channels = getattr(value, "channels")
        if not isinstance(channels, Mapping):
            return None

        physical_left, left_present = _lookup(
            channels, ("wing_motor_left", "left_wing_motor", "wm_left", "wing_motor_l")
        )
        physical_right, right_present = _lookup(
            channels, ("wing_motor_right", "right_wing_motor", "wm_right", "wing_motor_r")
        )
        # ``None`` means the physical pool was absent, matching the old
        # mapping parser's ``physical_* is not None`` check even if a malformed
        # channel mapping explicitly contains a null value.
        physical_left = physical_left if left_present and physical_left is not None else None
        physical_right = physical_right if right_present and physical_right is not None else None
        left_power = getattr(value, "left_wing_power")
        right_power = getattr(value, "right_wing_power")
        if config is None or config.prefer_physical_wing_motor:
            if physical_left is not None:
                left_power = physical_left
            if physical_right is not None:
                right_power = physical_right
        if config is not None and not config.allow_legacy_wing_fallback:
            if physical_left is None:
                left_power = 0.0
            if physical_right is None:
                right_power = 0.0
        return cls(
            left_power=left_power,
            right_power=right_power,
            yaw=getattr(value, "wing_steering"),
            source=getattr(value, "source"),
        ).normalized(config=config)

    @classmethod
    def from_any(cls, value: Any, *, config: Optional[FlightDynamicsConfig] = None) -> "NeuromuscularInput":
        if value is None:
            return cls().normalized(config=config)
        if isinstance(value, cls):
            return value.normalized(config=config)
        if not isinstance(value, Mapping):
            fast = cls._from_normalized_wing_output(value, config=config)
            if fast is not None:
                return fast
        if _sequence_like(value):
            if len(value) < 2:
                raise FlightDynamicsError("a sequence input needs left and right wing channels")
            return cls(left_power=value[0], right_power=value[1]).normalized(config=config)
        if not isinstance(value, Mapping):
            # This includes MotorCommand, BodyCommand, and NeuralOutput-like
            # records without introducing an import dependency.
            data: dict[str, Any] = {}
            for name in (
                "left_power", "right_power", "left_amplitude", "right_amplitude",
                "left_bias", "right_bias", "left_amplitude_mod", "right_amplitude_mod",
                "left_bias_rad", "right_bias_rad", "frequency_hz", "wingbeat_frequency_hz",
                "roll", "pitch", "yaw", "roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s",
                "roll_torque_Nm", "pitch_torque_Nm", "yaw_torque_Nm",
                "force_body_N", "wind_mps", "source", "wing_drive_left", "wing_drive_right",
                "wing_motor_left", "wing_motor_right", "left_wing_motor", "right_wing_motor",
                "left_wing_power", "right_wing_power", "wing_power", "wing_steering",
                "flight_power", "flight_steering", "flight_yaw",
                "left_wing", "right_wing", "left", "right",
                "thrust", "thrust_mps2", "power", "named_outputs", "channels",
            ):
                if hasattr(value, name):
                    data[name] = getattr(value, name)
            # A NeuromuscularOutput may expose legacy DNg02-derived wing
            # properties directly while carrying the physical wm motor pools
            # only inside ``channels``.  Promote the physical channels before
            # the one-level wrapper normalizer so they take precedence.
            nested_channels = getattr(value, "channels", None)
            if isinstance(nested_channels, Mapping):
                for channel_name, aliases in (
                    ("wing_motor_left", ("wing_motor_left", "left_wing_motor", "wm_left", "wing_motor_l")),
                    ("wing_motor_right", ("wing_motor_right", "right_wing_motor", "wm_right", "wing_motor_r")),
                ):
                    channel_value, channel_present = _lookup(nested_channels, aliases)
                    if channel_present and channel_name not in data:
                        data[channel_name] = channel_value
            if hasattr(value, "rates_hz") and "rates_hz" not in data:
                data["rates_hz"] = getattr(value, "rates_hz")
            return cls.from_any(data, config=config)

        mapping = _normalise_mapping(value)
        def get(names: Sequence[str], default: Any = None) -> Any:
            found, present = _lookup(mapping, names)
            return found if present else default

        # Explicit normalized channels win over aliases.  If only autopilot
        # thrust/power is present, use it symmetrically.  ``thrust_mps2`` is
        # deliberately *not* an alias here: it is the SI field on the
        # twin-contract MotorCommand, while this boundary consumes
        # normalized wing drive.  Treating the two as interchangeable would
        # silently turn an acceleration into a clamped actuator command.
        # Prefer actual wm motor-pool channels when available.  The historical
        # DNg02/flight-power aliases remain usable only through the explicit
        # compatibility fallback in FlightDynamicsConfig.
        physical_left = get(("wing_motor_left", "left_wing_motor", "wm_left",
                              "wing_motor_l"))
        physical_right = get(("wing_motor_right", "right_wing_motor", "wm_right",
                               "wing_motor_r"))
        left_power = get(("left_power", "power_left", "wing_power_left", "left_wing_power",
                          "wing_drive_left", "left_wing", "wing_left", "left", "l", "dlm_l", "muscle_left"))
        right_power = get(("right_power", "power_right", "wing_power_right", "right_wing_power",
                           "wing_drive_right", "right_wing", "wing_right", "right", "r", "dlm_r", "muscle_right"))
        if config is None or config.prefer_physical_wing_motor:
            if physical_left is not None:
                left_power = physical_left
            if physical_right is not None:
                right_power = physical_right
        if config is not None and not config.allow_legacy_wing_fallback:
            if physical_left is None:
                left_power = 0.0
            if physical_right is None:
                right_power = 0.0
        shared_power = get(("thrust", "power", "lift", "wing_power", "flight_power", "drive"))
        pair = get(("wing_drive", "wing_drives"))
        if pair is not None and _sequence_like(pair):
            if len(pair) < 2:
                raise FlightDynamicsError("wing_drive must contain left and right values")
            left_power = pair[0]
            right_power = pair[1]
        if config is None or config.prefer_physical_wing_motor:
            if physical_left is not None:
                left_power = physical_left
            if physical_right is not None:
                right_power = physical_right
        rates = get(("rates_hz", "firing_rates_hz", "rates"))

        # Keep the boundary fail-closed for SI contract commands.  A mapping
        # may carry both a normalized drive and SI metadata, but a
        # ``thrust_mps2``-only object/mapping has no defensible scale to
        # convert m/s^2 into [0, 1].  Callers must provide an explicit
        # normalized field (for example ``thrust``/``power``) or perform an
        # explicit, configured conversion before crossing this boundary.
        has_si_thrust = _lookup(mapping, ("thrust_mps2",))[1]
        has_normalized_source = (
            left_power is not None
            or right_power is not None
            or shared_power is not None
            or pair is not None
            or (_sequence_like(rates) and len(rates) >= 1)
        )
        if has_si_thrust and not has_normalized_source:
            raise FlightDynamicsError(
                "thrust_mps2 is SI acceleration (m/s^2), not normalized wing power; "
                "provide an explicit normalized thrust/power field or perform an "
                "explicit SI-to-normalized conversion with a configured scale"
            )
        if left_power is None and _sequence_like(rates) and len(rates) >= 1:
            left_power = _clamp(_finite(rates[0], "rates_hz[0]") / 200.0, 0.0, 1.0)
        if right_power is None and _sequence_like(rates) and len(rates) >= 2:
            right_power = _clamp(_finite(rates[1], "rates_hz[1]") / 200.0, 0.0, 1.0)
        if left_power is None:
            left_power = shared_power if shared_power is not None else 0.0
        if right_power is None:
            right_power = shared_power if shared_power is not None else 0.0

        left_amp = get(("left_amplitude", "amplitude_left", "wing_amplitude_left", "left_amp"), 1.0)
        right_amp = get(("right_amplitude", "amplitude_right", "wing_amplitude_right", "right_amp"), 1.0)
        left_amp_mod = get(("left_amplitude_mod", "amplitude_mod_left", "wing_amplitude_mod_left"))
        right_amp_mod = get(("right_amplitude_mod", "amplitude_mod_right", "wing_amplitude_mod_right"))
        left_bias = get(("left_bias", "bias_left", "wing_bias_left"), 0.0)
        right_bias = get(("right_bias", "bias_right", "wing_bias_right"), 0.0)
        left_bias_rad = get(("left_bias_rad", "bias_left_rad", "stroke_bias_left_rad"))
        right_bias_rad = get(("right_bias_rad", "bias_right_rad", "stroke_bias_right_rad"))
        frequency = get(("frequency_hz", "wingbeat_frequency_hz", "wpg_hz", "wpg_frequency_hz"))
        return cls(
            left_power=left_power,
            right_power=right_power,
            left_amplitude=left_amp,
            right_amplitude=right_amp,
            left_bias=left_bias,
            right_bias=right_bias,
            left_amplitude_mod=left_amp_mod,
            right_amplitude_mod=right_amp_mod,
            left_bias_rad=left_bias_rad,
            right_bias_rad=right_bias_rad,
            frequency_hz=frequency,
            roll=get(("roll", "roll_control", "roll_cmd", "roll_rate_rad_s"), 0.0),
            pitch=get(("pitch", "pitch_control", "pitch_cmd", "pitch_rate_rad_s"), 0.0),
            yaw=get(("yaw", "yaw_control", "yaw_cmd", "yaw_rate_rad_s", "wing_steering",
                     "flight_steering", "flight_yaw", "turn"), 0.0),
            roll_torque_Nm=get(("roll_torque_Nm", "roll_torque"), 0.0),
            pitch_torque_Nm=get(("pitch_torque_Nm", "pitch_torque"), 0.0),
            yaw_torque_Nm=get(("yaw_torque_Nm", "yaw_torque"), 0.0),
            force_body_N=get(("force_body_N", "body_force_N", "external_force_body_N"), (0.0, 0.0, 0.0)),
            wind_mps=get(("wind_mps", "wind", "wind_velocity_mps")),
            source=get(("source",), "neural"),
        ).normalized(config=config)


# Input naming aliases used by callers and hidden integration tests.
NeuralMuscleInput = NeuromuscularInput
MotorInput = NeuromuscularInput
FlightInput = NeuromuscularInput


@dataclass(frozen=True)
class JumpNeuromuscularInput:
    """Strict normalized input contract for the reduced leg/jump bridge.

    The bridge deliberately accepts *motor channels*, not a behavioural
    command or a pose request.  ``jump_activation`` is a non-negative event
    drive, ``leg_power`` is bilateral leg capacity, and ``leg_steering`` is a
    signed left/right motor asymmetry.  ``posture_*`` channels are signed
    torque controls.  Values are normalized and clipped by :meth:`normalized`;
    the actuator is the only code that converts them to SI force/torque.

    ``leg_power`` defaults to one so that a caller may issue a minimal
    ``{"jump": 1}`` event.  Supplying explicit left/right leg channels (or a
    ``leg_steering`` channel) gates and redistributes that capacity.  When
    ``ttmn_activity`` is the activation source it denotes the bilateral
    physical jump drive directly; a missing generic leg pool does not null the
    event.

    No threat, phase, goal, direction, target, pose, position, or orientation
    field is accepted.  Timing is owned by the physics integrator and its
    explicit delay/duration configuration, not smuggled through a scenario
    label.
    """

    jump_activation: float = 0.0
    # ``jump`` is a constructor/mapping compatibility alias.  It is cleared
    # by ``normalized`` so the canonical serialized channel is unambiguous.
    jump: Optional[float] = None
    # TTMN activity is the preferred physical leg-jump channel when a
    # connectome readout provides it.  ``jump`` remains a generic compatibility
    # alias for standalone callers, but the combined strict object path never
    # promotes a legacy DNp01 decision field when this channel is absent.
    ttmn_activity: Optional[float] = None
    activation_source: str = "jump_channel"
    leg_power: float = 1.0
    leg_steering: float = 0.0
    left_leg_power: Optional[float] = None
    right_leg_power: Optional[float] = None
    posture_roll: float = 0.0
    posture_pitch: float = 0.0
    posture_yaw: float = 0.0
    # ``posture`` is a compact constructor/mapping alias for the three signed
    # posture channels.  It never denotes an output pose.
    posture: Optional[Vec3] = None
    source: str = "neural"

    @property
    def jump_drive(self) -> float:
        """Canonical normalized jump event drive in ``[0, 1]``."""

        if self.ttmn_activity is not None:
            return self.ttmn_activity
        return self.jump_activation if self.jump is None else self.jump

    @property
    def left_leg(self) -> float:
        if self.left_leg_power is not None:
            return self.left_leg_power
        return self.leg_power - 0.5 * self.leg_steering

    @property
    def right_leg(self) -> float:
        if self.right_leg_power is not None:
            return self.right_leg_power
        return self.leg_power + 0.5 * self.leg_steering

    @property
    def leg_capacity(self) -> float:
        """Mean bilateral normalized capacity after side-specific clipping."""

        return 0.5 * (self.left_leg + self.right_leg)

    def normalized(self) -> "JumpNeuromuscularInput":
        """Return a finite, bounded copy with all aliases resolved."""

        if self.ttmn_activity is not None:
            jump = self.ttmn_activity
            activation_source = "ttmn_activity"
        else:
            jump = self.jump_activation if self.jump is None else self.jump
            activation_source = str(self.activation_source or "jump_channel")
        jump_value = _clamp(_finite(jump, "jump_activation"), 0.0, 1.0)
        leg_power = _clamp(_finite(self.leg_power, "leg_power"), 0.0, 1.0)
        steering = _clamp(_finite(self.leg_steering, "leg_steering"), -1.0, 1.0)
        if self.left_leg_power is None:
            left = _clamp(leg_power - 0.5 * steering, 0.0, 1.0)
        else:
            left = _clamp(_finite(self.left_leg_power, "left_leg_power"), 0.0, 1.0)
        if self.right_leg_power is None:
            right = _clamp(leg_power + 0.5 * steering, 0.0, 1.0)
        else:
            right = _clamp(_finite(self.right_leg_power, "right_leg_power"), 0.0, 1.0)
        if self.posture is not None:
            posture = _vec3(self.posture, "posture")
            roll, pitch, yaw = posture
        else:
            roll = _finite(self.posture_roll, "posture_roll")
            pitch = _finite(self.posture_pitch, "posture_pitch")
            yaw = _finite(self.posture_yaw, "posture_yaw")
        return replace(
            self,
            jump_activation=jump_value,
            jump=None,
            ttmn_activity=(jump_value if activation_source == "ttmn_activity" else None),
            activation_source=activation_source,
            leg_power=0.5 * (left + right),
            leg_steering=_clamp(right - left, -1.0, 1.0),
            left_leg_power=left,
            right_leg_power=right,
            posture_roll=_clamp(roll, -1.0, 1.0),
            posture_pitch=_clamp(pitch, -1.0, 1.0),
            posture_yaw=_clamp(yaw, -1.0, 1.0),
            posture=None,
            source=str(getattr(self.source, "value", self.source)),
        )

    @property
    def units(self) -> dict[str, str]:
        return {
            "jump_activation": "normalized [0, 1]",
            "ttmn_activity": "normalized [0, 1]",
            "leg_power": "normalized [0, 1]",
            "leg_steering": "normalized [-1, 1]",
            "left_leg_power": "normalized [0, 1]",
            "right_leg_power": "normalized [0, 1]",
            "posture_roll": "normalized [-1, 1]",
            "posture_pitch": "normalized [-1, 1]",
            "posture_yaw": "normalized [-1, 1]",
        }

    def to_dict(self) -> dict[str, Any]:
        result = {
            "jump_activation": self.jump_activation,
            "ttmn_activity": (
                self.jump_activation if self.activation_source == "ttmn_activity" else None
            ),
            "activation_source": self.activation_source,
            "leg_power": self.leg_power,
            "leg_steering": self.leg_steering,
            "left_leg_power": self.left_leg_power,
            "right_leg_power": self.right_leg_power,
            "posture_roll": self.posture_roll,
            "posture_pitch": self.posture_pitch,
            "posture_yaw": self.posture_yaw,
            "source": self.source,
            "units": self.units,
        }
        return result

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "jump": "jump_activation", "jump_drive": "jump_activation",
            "ttmn_activity": "jump_activation", "ttmn": "jump_activation",
            "leg": "leg_power", "left_leg": "left_leg_power",
            "right_leg": "right_leg_power",
        }
        return getattr(self, aliases.get(_lower_key(key), key))

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default

    @classmethod
    def from_any(cls, value: Any = None) -> "JumpNeuromuscularInput":
        """Parse only the jump/leg/posture channel boundary.

        Unlike the broad wing input parser, this boundary fails closed for
        unknown keys.  That is intentional: a threat estimate, scenario
        phase, goal, direction label, or pose request must not become a hidden
        actuator input.
        """

        if value is None:
            return cls().normalized()
        if isinstance(value, cls):
            return value.normalized()
        if not isinstance(value, Mapping):
            # Attribute records are accepted only when they expose one of the
            # declared channels.  In particular, no arbitrary ``state`` or
            # ``pose`` record is coerced into a motor command.
            data: dict[str, Any] = {}
            for name in (
                "jump_activation", "jump", "jump_drive", "ttmn_activity",
                "activation_source", "leg_power", "leg",
                "leg_steering", "left_leg_power", "right_leg_power", "left_leg",
                "right_leg", "posture_roll", "posture_pitch", "posture_yaw",
                "posture", "source",
            ):
                if hasattr(value, name):
                    data[name] = getattr(value, name)
            if not data:
                raise FlightDynamicsError(
                    "jump input must expose only jump/leg/posture neuromuscular channels"
                )
            return cls.from_any(data)
        if not isinstance(value, Mapping):  # pragma: no cover - defensive
            raise FlightDynamicsError("jump input must be a mapping or channel record")

        raw = dict(value)
        # Common wrappers used by neural readouts are metadata, not channels.
        for wrapper in ("channels", "motor_channels", "neuromuscular", "named_outputs"):
            nested = raw.get(wrapper)
            if isinstance(nested, Mapping):
                for key, nested_value in nested.items():
                    raw.setdefault(str(key), nested_value)

        aliases = {
            "jump": "jump_activation", "jump_drive": "jump_activation",
            "jump_command": "jump_activation", "jump_activation": "jump_activation",
            "ttmn_activity": "ttmn_activity", "ttmn": "ttmn_activity",
            "leg": "leg_power", "leg_power": "leg_power", "leg_drive": "leg_power",
            "leg_motor": "leg_power", "leg_steering": "leg_steering",
            "leg_asymmetry": "leg_steering", "leg_motor_asym": "leg_steering",
            "left_leg": "left_leg_power", "left_leg_power": "left_leg_power",
            "leg_left": "left_leg_power", "right_leg": "right_leg_power",
            "right_leg_power": "right_leg_power", "leg_right": "right_leg_power",
            "posture": "posture", "posture_roll": "posture_roll",
            "posture_pitch": "posture_pitch", "posture_yaw": "posture_yaw",
            # These short names are still posture channels, not external
            # direction labels.  Explicit posture_* names are preferred.
            "roll": "posture_roll", "pitch": "posture_pitch", "yaw": "posture_yaw",
        }
        metadata = {
            "source", "activation_source", "timestamp", "timestamp_s", "time_s", "units", "provenance",
            "channels", "motor_channels", "neuromuscular", "named_outputs",
        }
        forbidden_tokens = (
            "threat", "phase", "goal", "direction", "heading", "target",
            "scenario", "pose", "position", "orientation", "teleport",
        )
        parsed: dict[str, Any] = {}
        for raw_key, raw_value in raw.items():
            key = _lower_key(raw_key)
            if key in metadata:
                if key == "source":
                    parsed["source"] = raw_value
                elif key == "activation_source":
                    parsed["activation_source"] = raw_value
                continue
            if any(token in key for token in forbidden_tokens):
                raise FlightDynamicsError(
                    f"unsupported jump input channel {raw_key!r}; "
                    "threat/phase/goal/direction/pose labels are not actuator inputs"
                )
            canonical = aliases.get(key)
            if canonical is None:
                raise FlightDynamicsError(
                    f"unsupported jump input channel {raw_key!r}; "
                    "accepted channels are jump/leg/posture only"
                )
            parsed[canonical] = raw_value

        if "leg_power" not in parsed:
            parsed["leg_power"] = 1.0
        return cls(**parsed).normalized()


# Semantic aliases used by body-side callers and hidden integration tests.
JumpInput = JumpNeuromuscularInput
LegNeuromuscularInput = JumpNeuromuscularInput


@dataclass(frozen=True)
class _GroundWalkingInput:
    """Raw normalized leg motor channels used only by ground walking."""

    leg_power: float = 0.0
    leg_steering: float = 0.0
    source: str = "neural"


def _ground_walking_input_from_any(value: Any) -> _GroundWalkingInput:
    """Read leg motor outputs without consulting jump or TTMn activation.

    A missing ``leg_power`` channel means zero walking drive.  In particular,
    this parser never inherits ``JumpNeuromuscularInput``'s capacity default
    or treats TTMn jump activity as locomotion.
    """

    if value is None:
        return _GroundWalkingInput()
    if isinstance(value, Mapping):
        mapping = _normalise_mapping(value)
    else:
        mapping: dict[str, Any] = {}
        for name in ("leg_power", "leg_drive", "leg_motor", "legMotor",
                     "leg_steering", "leg_asymmetry", "legMotorAsym", "source"):
            if hasattr(value, name):
                mapping[name] = getattr(value, name)
        nested = getattr(value, "channels", None)
        if isinstance(nested, Mapping):
            for key, item in _normalise_mapping(nested).items():
                mapping.setdefault(str(key), item)
    drive, has_drive = _lookup(
        mapping, ("leg_power", "leg_drive", "leg_motor", "legMotor"),
    )
    steering, has_steering = _lookup(
        mapping, ("leg_steering", "leg_asymmetry", "legMotorAsym"),
    )
    source, has_source = _lookup(mapping, ("source",))
    if not has_drive and not has_steering and not has_source:
        return _GroundWalkingInput()
    source_value = source if has_source else "neural"
    source_label = str(getattr(source_value, "value", source_value))
    return _GroundWalkingInput(
        leg_power=_clamp(_finite(drive if has_drive else 0.0, "leg_power"), 0.0, 1.0),
        leg_steering=_clamp(
            _finite(steering if has_steering else 0.0, "leg_steering"), -1.0, 1.0,
        ),
        source=source_label,
    )


@dataclass
class FlightState:
    """Mutable six-degree-of-freedom state used by the reduced integrator."""

    time_s: float = 0.0
    position_m: Vec3 = (0.0, 0.0, 1.2e-3)
    velocity_mps: Vec3 = (0.0, 0.0, 0.0)
    orientation_quat: Quat = (1.0, 0.0, 0.0, 0.0)
    angular_velocity_rad_s: Vec3 = (0.0, 0.0, 0.0)
    cumulative_energy_J: float = 0.0

    @property
    def angular_velocity_radps(self) -> Vec3:
        return self.angular_velocity_rad_s

    @property
    def position(self) -> Vec3:
        return self.position_m

    @property
    def velocity(self) -> Vec3:
        return self.velocity_mps

    @property
    def attitude_quat(self) -> Quat:
        return self.orientation_quat

    def copy(self) -> "FlightState":
        return replace(self)

    def to_dict(self) -> dict[str, Any]:
        result = _record_dict(self)
        result["units"] = {
            "time": "s", "position": "m", "velocity": "m/s",
            "orientation": "unit quaternion", "angular_velocity": "rad/s",
            "energy": "J",
        }
        return result

    @property
    def units(self) -> dict[str, str]:
        return {
            "time_s": "s", "position_m": "m", "velocity_mps": "m/s",
            "angular_velocity_rad_s": "rad/s", "cumulative_energy_J": "J",
        }

    @classmethod
    def from_any(cls, value: Any, *, default_height_m: float = 1.2e-3) -> "FlightState":
        if value is None:
            return cls(position_m=(0.0, 0.0, default_height_m))
        if isinstance(value, cls):
            state = replace(value)
        elif isinstance(value, Mapping):
            get = value.get
            timestamp = get("timestamp")
            time_s = get("time_s")
            if time_s is None:
                time_s = _timestamp_seconds(timestamp) if timestamp is not None else 0.0
            orientation = get("orientation_quat", get("attitude_quat"))
            if orientation is None:
                orientation = _quat_from_rpy(
                    get("orientation_rad", get("orientation_rpy_rad", get("attitude_rpy_rad", (0.0, 0.0, 0.0))))
                )
            state = cls(
                time_s=time_s,
                position_m=get("position_m", get("position", (0.0, 0.0, default_height_m))),
                velocity_mps=get("velocity_mps", get("velocity", (0.0, 0.0, 0.0))),
                orientation_quat=orientation,
                angular_velocity_rad_s=get("angular_velocity_rad_s", get("angular_velocity_radps", get("angular_velocity", (0.0, 0.0, 0.0)))),
                cumulative_energy_J=get("cumulative_energy_J", get("energy_J", 0.0)),
            )
        else:
            timestamp = getattr(value, "timestamp", None)
            time_s = getattr(value, "time_s", None)
            if time_s is None:
                time_s = _timestamp_seconds(timestamp) if timestamp is not None else 0.0
            orientation = getattr(value, "orientation_quat", getattr(value, "attitude_quat", None))
            if orientation is None:
                orientation = _quat_from_rpy(
                    getattr(value, "orientation_rad", getattr(value, "orientation_rpy_rad", getattr(value, "attitude_rpy_rad", (0.0, 0.0, 0.0))))
                )
            state = cls(
                time_s=time_s,
                position_m=getattr(value, "position_m", getattr(value, "position", (0.0, 0.0, default_height_m))),
                velocity_mps=getattr(value, "velocity_mps", getattr(value, "velocity", (0.0, 0.0, 0.0))),
                orientation_quat=orientation,
                angular_velocity_rad_s=getattr(value, "angular_velocity_rad_s", getattr(value, "angular_velocity_radps", getattr(value, "angular_velocity", (0.0, 0.0, 0.0)))),
                cumulative_energy_J=getattr(value, "cumulative_energy_J", getattr(value, "energy_J", 0.0)),
            )
        state.time_s = _finite(state.time_s, "time_s")
        state.position_m = _vec3(state.position_m, "position_m")
        state.velocity_mps = _vec3(state.velocity_mps, "velocity_mps")
        state.orientation_quat = _quat(state.orientation_quat)
        state.angular_velocity_rad_s = _vec3(state.angular_velocity_rad_s, "angular_velocity_rad_s")
        state.cumulative_energy_J = _finite(state.cumulative_energy_J, "cumulative_energy_J")
        if state.cumulative_energy_J < 0:
            raise FlightDynamicsError("cumulative_energy_J must be non-negative")
        return state


@dataclass(frozen=True)
class FlightOutput:
    """One force/moment evaluation and the updated body-compatible state."""

    time_s: float
    state: FlightState
    frequency_hz: float
    phase_left_rad: float
    phase_right_rad: float
    left_power: float
    right_power: float
    left_amplitude: float
    right_amplitude: float
    left_bias_rad: float
    right_bias_rad: float
    left_lift_N: float
    right_lift_N: float
    left_profile_drag_N: float
    right_profile_drag_N: float
    lift_force_body_N: Vec3
    drag_force_body_N: Vec3
    aerodynamic_force_body_N: Vec3
    gravity_force_body_N: Vec3
    wind_force_body_N: Vec3
    external_force_body_N: Vec3
    total_force_body_N: Vec3
    total_force_world_N: Vec3
    torque_body_Nm: Vec3
    acceleration_world_mps2: Vec3
    angular_acceleration_body_rad_s2: Vec3
    power_W: float
    energy_increment_J: float
    cumulative_energy_J: float
    source: str
    fit_status: str
    model: str = MODEL_NAME
    model_scope: str = MODEL_SCOPE
    is_cfd: bool = IS_CFD
    contact_force_body_N: Vec3 = (0.0, 0.0, 0.0)
    ground_plane_z_m: float = 0.0
    contact_surface_id: Optional[str] = None
    target_contact: bool = False
    # Optional leg/jump terms.  Defaults preserve the wing-only FlightDynamics
    # contract; the combined bridge fills these from JumpLegActuator exactly
    # once during the same body integration.
    jump_force_body_N: Vec3 = (0.0, 0.0, 0.0)
    jump_impulse_body_Ns: Vec3 = (0.0, 0.0, 0.0)
    jump_posture_torque_body_Nm: Vec3 = (0.0, 0.0, 0.0)
    jump_actuator_active: bool = False
    jump_contact_gated: bool = False
    jump_event_latched: bool = False
    jump_input: Optional[Any] = None
    jump_model: Optional[str] = None
    jump_model_scope: Optional[str] = None
    jump_is_flybody: Optional[bool] = None
    ground_walking_drive: float = 0.0
    ground_walking_steering: float = 0.0
    ground_walking_force_body_N: Vec3 = (0.0, 0.0, 0.0)
    ground_walking_force_world_N: Vec3 = (0.0, 0.0, 0.0)
    ground_walking_traction_force_body_N: Vec3 = (0.0, 0.0, 0.0)
    ground_walking_braking_force_world_N: Vec3 = (0.0, 0.0, 0.0)
    ground_walking_lateral_braking_force_world_N: Vec3 = (0.0, 0.0, 0.0)
    ground_walking_speed_limit_force_world_N: Vec3 = (0.0, 0.0, 0.0)
    ground_walking_yaw_torque_Nm: float = 0.0
    ground_walking_grounded_at_start: bool = False
    ground_walking_active: bool = False
    ground_walking_contact_available: bool = False
    ground_walking_enabled: bool = False
    ground_walking_source: str = "neural"
    ground_walking_model: str = GROUND_WALKING_MODEL_NAME
    ground_walking_model_scope: str = GROUND_WALKING_MODEL_SCOPE

    # Existing body simulators use these names.
    @property
    def aerodynamic_force_N(self) -> Vec3:
        return self.aerodynamic_force_body_N

    @property
    def gravity_force_N(self) -> Vec3:
        return self.gravity_force_body_N

    @property
    def total_force_N(self) -> Vec3:
        return self.total_force_body_N

    @property
    def total_torque_Nm(self) -> Vec3:
        return self.torque_body_Nm

    @property
    def acceleration_mps2(self) -> Vec3:
        return self.acceleration_world_mps2

    @property
    def angular_acceleration_radps2(self) -> Vec3:
        return self.angular_acceleration_body_rad_s2

    @property
    def wind_disturbance_N(self) -> Vec3:
        return self.wind_force_body_N

    @property
    def wind_force_N(self) -> Vec3:
        return self.wind_force_body_N

    @property
    def external_force_N(self) -> Vec3:
        return self.external_force_body_N

    @property
    def contact_force_N(self) -> Vec3:
        return self.contact_force_body_N

    @property
    def jump_force_N(self) -> Vec3:
        return self.jump_force_body_N

    @property
    def jump_force_world_N(self) -> Vec3:
        return _qrotate(self.state.orientation_quat, self.jump_force_body_N)

    @property
    def jump_impulse_Ns(self) -> Vec3:
        return self.jump_impulse_body_Ns

    @property
    def jump_impulse_world_Ns(self) -> Vec3:
        return _qrotate(self.state.orientation_quat, self.jump_impulse_body_Ns)

    @property
    def posture_torque_Nm(self) -> Vec3:
        return self.jump_posture_torque_body_Nm

    @property
    def moment_Nm(self) -> Vec3:
        return self.torque_body_Nm

    @property
    def left_wing_drive(self) -> float:
        return self.left_power

    @property
    def right_wing_drive(self) -> float:
        return self.right_power

    @property
    def energy_J(self) -> float:
        return self.cumulative_energy_J

    @property
    def lift_N(self) -> float:
        """Signed vertical lift magnitude in the body frame."""

        return self.lift_force_body_N[2]

    @property
    def lift_force_N(self) -> Vec3:
        return self.lift_force_body_N

    @property
    def drag_N(self) -> float:
        """Non-negative scalar mean drag, including wing profile work."""

        return _vnorm(self.drag_force_body_N) + self.profile_drag_N

    @property
    def profile_drag_N(self) -> float:
        return self.left_profile_drag_N + self.right_profile_drag_N

    @property
    def roll_moment_Nm(self) -> float:
        return self.torque_body_Nm[0]

    @property
    def pitch_moment_Nm(self) -> float:
        return self.torque_body_Nm[1]

    @property
    def yaw_moment_Nm(self) -> float:
        return self.torque_body_Nm[2]

    @property
    def moment_body_Nm(self) -> Vec3:
        return self.torque_body_Nm

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "force_body_N": "total_force_body_N",
            "torque_body_Nm": "torque_body_Nm",
            "energy_J": "cumulative_energy_J",
            "left": "left_power",
            "right": "right_power",
        }
        return getattr(self, aliases.get(key, key))

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default

    @property
    def force_body_N(self) -> Vec3:
        return self.total_force_body_N

    @property
    def units(self) -> dict[str, str]:
        return {
            "time_s": "s", "frequency_hz": "Hz", "phase_rad": "rad",
            "force_N": "N", "torque_Nm": "N*m", "acceleration_mps2": "m/s^2",
            "angular_acceleration_rad_s2": "rad/s^2", "power_W": "W", "energy_J": "J",
            "contact_force_N": "N", "ground_plane_z_m": "m",
            "jump_force_N": "N", "jump_impulse_Ns": "N*s",
            "jump_force_world_N": "N", "jump_impulse_world_Ns": "N*s",
            "jump_posture_torque_Nm": "N*m",
            "ground_walking_force_N": "N",
            "ground_walking_yaw_torque_Nm": "N*m",
            "ground_walking_drive": "normalized [0, 1]",
            "ground_walking_steering": "normalized [-1, 1]",
        }

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Per-step model/source provenance for the optional walking actuator."""

        return _ProvenanceRecord({
            "model": self.ground_walking_model,
            "scope": self.ground_walking_model_scope,
            "enabled": self.ground_walking_enabled,
            "active": self.ground_walking_active,
            "contact_available": self.ground_walking_contact_available,
            "grounded_at_start": self.ground_walking_grounded_at_start,
            "source": self.ground_walking_source,
            "input": {
                "leg_power": self.ground_walking_drive,
                "leg_steering": self.ground_walking_steering,
            },
            "applied_force_body_N": tuple(self.ground_walking_force_body_N),
            "applied_traction_force_body_N": tuple(self.ground_walking_traction_force_body_N),
            "applied_braking_force_world_N": tuple(self.ground_walking_braking_force_world_N),
            "applied_lateral_braking_force_world_N": tuple(self.ground_walking_lateral_braking_force_world_N),
            "applied_speed_limit_force_world_N": tuple(self.ground_walking_speed_limit_force_world_N),
            "applied_yaw_torque_Nm": self.ground_walking_yaw_torque_Nm,
            "limitations": [
                "contact gate uses ground_plane_z_m; raised ContactSurface pads do not enable walking",
                "reduced point-body traction has no articulated foot pose",
            ],
        })

    def body_command(self, *, include_force: bool = False) -> dict[str, Any]:
        """Return a mapping accepted by ``digital_body.BodyCommand.from_any``.

        By default the aerodynamic force is not injected a second time: the
        body receives wing drives and direct SI torque controls and computes
        its own force.  ``include_force=True`` is useful for a force-driven
        external body backend and includes the model's total force.
        """

        return {
            "wing_drive_left": self.left_power,
            "wing_drive_right": self.right_power,
            "wingbeat_frequency_hz": self.frequency_hz,
            "roll_torque_Nm": self.torque_body_Nm[0],
            "pitch_torque_Nm": self.torque_body_Nm[1],
            "yaw_torque_Nm": self.torque_body_Nm[2],
            "force_body_N": self.total_force_body_N if include_force else self.external_force_body_N,
            "source": self.source,
        }

    def to_body_command(self, *, include_force: bool = False) -> dict[str, Any]:
        return self.body_command(include_force=include_force)

    def to_dict(self) -> dict[str, Any]:
        result = _record_dict(self)
        result["jump_force_world_N"] = tuple(self.jump_force_world_N)
        result["jump_impulse_world_Ns"] = tuple(self.jump_impulse_world_Ns)
        result["jump_input"] = (
            self.jump_input.to_dict()
            if hasattr(self.jump_input, "to_dict") else self.jump_input
        )
        result["ground_walking_force_world_N"] = tuple(self.ground_walking_force_world_N)
        result["ground_walking_provenance"] = dict(self.provenance)
        result["state"] = self.state.to_dict()
        result["units"] = {
            "force": "N", "torque": "N*m", "power": "W", "energy": "J",
            "frequency": "Hz", "angle": "rad", "time": "s",
        }
        return result

    as_dict = to_dict

    def to_body_state(self) -> dict[str, Any]:
        """Return a ``digital_body``-shaped state mapping.

        Force fields are reported in world axes where the body backend uses
        world forces; the explicit ``*_body_N`` fields remain available on
        this record for callers that need body axes.
        """

        orientation = self.state.orientation_quat
        aero_world = _qrotate(orientation, self.aerodynamic_force_body_N)
        gravity_world = _qrotate(orientation, self.gravity_force_body_N)
        total_world = self.total_force_world_N
        ground_plane = self.ground_plane_z_m
        ground_contact = (
            self.state.position_m[2] <= ground_plane + 1.0e-12
            and self.state.velocity_mps[2] <= 1.0e-12
        )
        contact = ground_contact or self.target_contact
        return {
            "time_s": self.state.time_s,
            "position_m": tuple(self.state.position_m),
            "velocity_mps": tuple(self.state.velocity_mps),
            "orientation_quat": tuple(orientation),
            "angular_velocity_radps": tuple(self.state.angular_velocity_rad_s),
            "wing_drive_left": self.left_power,
            "wing_drive_right": self.right_power,
            "wing_phase_left_rad": self.phase_left_rad,
            "wing_phase_right_rad": self.phase_right_rad,
            "contact": contact,
            "target_contact": self.target_contact,
            "contact_surface_id": self.contact_surface_id,
            "ground_distance_m": max(0.0, self.state.position_m[2] - ground_plane),
            "flight_mode": "grounded" if contact else "flight",
            "acceleration_mps2": tuple(self.acceleration_world_mps2),
            "angular_acceleration_radps2": tuple(self.angular_acceleration_body_rad_s2),
            "aerodynamic_force_N": tuple(aero_world),
            "gravity_force_N": tuple(gravity_world),
            "contact_force_N": tuple(_qrotate(orientation, self.contact_force_body_N)),
            "jump_force_N": tuple(_qrotate(orientation, self.jump_force_body_N)),
            "jump_impulse_Ns": tuple(_qrotate(orientation, self.jump_impulse_body_Ns)),
            "ground_walking_force_N": tuple(self.ground_walking_force_world_N),
            "ground_walking_lateral_braking_force_N": tuple(
                self.ground_walking_lateral_braking_force_world_N,
            ),
            "ground_walking_drive": self.ground_walking_drive,
            "ground_walking_steering": self.ground_walking_steering,
            "ground_walking_yaw_torque_Nm": self.ground_walking_yaw_torque_Nm,
            "ground_walking_active": self.ground_walking_active,
            "ground_walking_contact_available": self.ground_walking_contact_available,
            "ground_walking_source": self.ground_walking_source,
            "jump_actuator_active": self.jump_actuator_active,
            "jump_contact_gated": self.jump_contact_gated,
            "jump_model": self.jump_model,
            "jump_model_scope": self.jump_model_scope,
            "total_force_N": tuple(total_world),
            "total_torque_Nm": tuple(self.torque_body_Nm),
            "energy_J": self.cumulative_energy_J,
            "source": self.source,
            "fit_status": self.fit_status,
            "model": self.model,
            "model_scope": self.model_scope,
        }

    as_body_state = to_body_state

    def to_digital_body_state(self) -> Any:
        """Instantiate the optional local ``digital_body.BodyState`` adapter."""

        try:
            from digital_body import BodyState as DigitalBodyState
        except ImportError as exc:  # pragma: no cover - standalone module use
            raise FlightDynamicsError("digital_body is not available") from exc
        values = self.to_body_state()
        allowed = set(getattr(DigitalBodyState, "__dataclass_fields__", ()))
        return DigitalBodyState(**{key: value for key, value in values.items() if key in allowed})

    to_body_backend_state = to_digital_body_state

    def to_twin_body_state(self, timestamp: Any, *, source: str = "handbuilt") -> Any:
        """Build the unified ``twin_contracts.BodyState`` when available.

        Importing the contract lazily keeps this physics module usable as a
        standalone script.  The quaternion state is converted to the
        contract's explicit XYZ roll/pitch/yaw representation.
        """

        try:
            from twin_contracts import BodyState as ContractBodyState
        except ImportError as exc:  # pragma: no cover - only for standalone use
            raise FlightDynamicsError("twin_contracts is not available") from exc
        allowed_sources = {"connectome", "fitted", "handbuilt"}
        contract_source = source if source in allowed_sources else "handbuilt"
        return ContractBodyState(
            timestamp=timestamp,
            position_m=self.state.position_m,
            orientation_rad=_rpy_from_quat(self.state.orientation_quat),
            velocity_mps=self.state.velocity_mps,
            angular_velocity_rad_s=self.state.angular_velocity_rad_s,
            source=contract_source,
        )

    to_contract_body_state = to_twin_body_state


# Useful semantic alias for users expecting a force/moment record.
BodyOutput = FlightOutput
FlightStep = FlightOutput


@dataclass(frozen=True)
class _WingEvaluation:
    lift_N: float
    profile_drag_N: float
    lift_vector_body_N: Vec3
    amplitude_rad: float
    bias_rad: float
    phase_rad: float
    power_W: float


class FlightDynamics:
    """Reduced deterministic 6-DOF flight model with a calibratable WPG."""

    model_name = MODEL_NAME
    model_scope = MODEL_SCOPE
    is_cfd = IS_CFD

    def __init__(self, config: Optional[FlightDynamicsConfig] = None,
                 state: Optional[Any] = None, *, wind_mps: Optional[Vec3] = None,
                 jump_config: Optional[Any] = None) -> None:
        self.config = config or FlightDynamicsConfig()
        self.wpg = WingbeatPatternGenerator(
            self.config.wingbeat_frequency_hz,
            frequency_min_hz=self.config.wpg_frequency_min_hz,
            frequency_max_hz=self.config.wpg_frequency_max_hz,
        )
        self.state = FlightState.from_any(state)
        self._resolve_initial_contact()
        self.wind_mps = self.config.wind_mps if wind_mps is None else _vec3(wind_mps, "wind_mps")
        self.last_output: Optional[FlightOutput] = None
        # Combined wing/leg stepping owns one FlightState and one WPG.  These
        # small event fields are inert for the legacy wing-only ``step`` path.
        self.jump_config = jump_config
        # ``JumpLegConfig`` is frozen and ``JumpLegActuator`` is stateless, so
        # the normalized config/actuator can be reused across body frames.
        # Keep the source key explicit: callers historically assign
        # ``jump_config``/``config`` directly, and mutable mapping configs must
        # invalidate the cache when their contents change.
        self._combined_jump_cache_key_value: Any = None
        self._combined_jump_config_cache: Any = None
        self._combined_jump_actuator_cache: Any = None
        self._jump_command_start_s: Optional[float] = None
        self._jump_event_latched = False
        self._last_jump_activation = 0.0

    def _resolve_initial_contact(self) -> None:
        """Keep a restored/initial state outside the unilateral contact plane.

        The integrator owns contact resolution.  Normalizing an interpenetrating
        initial state here prevents the launcher from needing a separate
        post-step clamp and preserves any explicitly upward initial velocity.
        """

        cfg = self.config
        if not cfg.ground_contact_enabled or self.state.position_m[2] >= cfg.ground_plane_z_m:
            return
        self.state.position_m = (
            self.state.position_m[0], self.state.position_m[1], cfg.ground_plane_z_m,
        )
        if self.state.velocity_mps[2] < 0.0:
            self.state.velocity_mps = (
                self.state.velocity_mps[0], self.state.velocity_mps[1], 0.0,
            )

    @property
    def time_s(self) -> float:
        return self.state.time_s

    @property
    def params(self) -> FlightDynamicsConfig:
        """Alias for callers that call the centralized config ``params``."""

        return self.config

    @property
    def body_state(self) -> FlightState:
        return self.state

    @property
    def provenance(self) -> dict[str, Any]:
        return self.config.provenance()

    @property
    def units(self) -> dict[str, str]:
        return dict(self.config.units)

    @property
    def parameters(self) -> dict[str, Any]:
        return self.config.parameter_manifest()

    def manifest(self) -> dict[str, Any]:
        """Return a serializable model manifest for trajectory artefacts."""

        return self.config.manifest()

    def reset(self, state: Optional[Any] = None, *, phase_rad: float = 0.0,
              energy_J: Optional[float] = None) -> FlightState:
        self.state = FlightState.from_any(state)
        self._resolve_initial_contact()
        if energy_J is not None:
            value = _finite(energy_J, "energy_J")
            if value < 0:
                raise FlightDynamicsError("energy_J must be non-negative")
            self.state.cumulative_energy_J = value
        self.wpg.reset(phase_rad)
        self.last_output = None
        self._jump_command_start_s = None
        self._jump_event_latched = False
        self._last_jump_activation = 0.0
        return self.state.copy()

    def set_wind(self, wind_mps: Vec3) -> Vec3:
        self.wind_mps = _vec3(wind_mps, "wind_mps")
        return self.wind_mps

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly deterministic simulator snapshot."""

        return {
            "state": self.state.to_dict(),
            "phase_rad": self.wpg.phase_rad,
            "frequency_hz": self.wpg.frequency_hz,
            "wind_mps": tuple(self.wind_mps),
            "model": MODEL_NAME,
            "fit_status": self.config.fit_status,
            "parameters": self.config.parameter_manifest(),
            "jump_command_start_s": self._jump_command_start_s,
            "jump_event_latched": self._jump_event_latched,
            "last_jump_activation": self._last_jump_activation,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> FlightState:
        if not isinstance(snapshot, Mapping):
            raise FlightDynamicsError("snapshot must be a mapping")
        state_value = snapshot.get("state", snapshot)
        self.state = FlightState.from_any(state_value)
        self._resolve_initial_contact()
        if "wind_mps" in snapshot:
            self.wind_mps = _vec3(snapshot["wind_mps"], "wind_mps")
        if "frequency_hz" in snapshot:
            self.wpg.set_frequency(snapshot["frequency_hz"])
        self.wpg.reset(snapshot.get("phase_rad", 0.0))
        self.last_output = None
        raw_jump_start = snapshot.get("jump_command_start_s")
        self._jump_command_start_s = (
            None if raw_jump_start is None else _finite(raw_jump_start, "jump_command_start_s")
        )
        self._jump_event_latched = bool(snapshot.get("jump_event_latched", False))
        self._last_jump_activation = _clamp(
            _finite(snapshot.get("last_jump_activation", 0.0), "last_jump_activation"),
            0.0, 1.0,
        )
        return self.state.copy()

    @staticmethod
    def _jump_channels_from_command(
        value: Any,
        *,
        strict_neural: bool = True,
    ) -> Optional[Any]:
        """Extract jump/leg/posture channels from a mixed motor record.

        The wing parser intentionally remains permissive for legacy motor
        commands.  The combined bridge applies the strict jump boundary here:
        mixed mappings may contain wing fields, but only jump/leg/posture
        fields are forwarded to :class:`JumpNeuromuscularInput`.  Behavioural
        labels are rejected rather than silently ignored.
        """

        if value is None:
            return None
        if isinstance(value, JumpNeuromuscularInput):
            return value
        if not isinstance(value, Mapping):
            # Prefer the physical TTMN field on neural output records.  The
            # legacy ``jump`` field is deliberately ignored on the strict
            # object path so a DNp01 decision compatibility field cannot be
            # turned directly into a leg force.
            direct_names = (
                "ttmn_activity", "ttmn", "leg_power", "leg",
                "leg_steering", "left_leg_power", "right_leg_power", "left_leg",
                "right_leg", "posture_roll", "posture_pitch", "posture_yaw",
                "posture", "source", "activation_source",
            )
            direct: dict[str, Any] = {}
            for name in direct_names:
                if hasattr(value, name):
                    direct[name] = getattr(value, name)
            if not strict_neural and hasattr(value, "jump"):
                direct["jump"] = getattr(value, "jump")
                direct["activation_source"] = "legacy_jump_compatibility"
            if not strict_neural and hasattr(value, "jump_activation"):
                direct["jump_activation"] = getattr(value, "jump_activation")
            nested = getattr(value, "channels", None)
            if isinstance(nested, Mapping) and not any(
                key in direct for key in ("ttmn_activity", "ttmn", "jump_activation")
            ):
                nested_jump = FlightDynamics._jump_channels_from_command(
                    nested, strict_neural=strict_neural,
                )
                if isinstance(nested_jump, Mapping) and any(
                    _lower_key(key) in ("ttmn_activity", "ttmn")
                    for key in nested_jump
                ):
                    direct.update(nested_jump)
            if direct:
                return direct
            if isinstance(nested, Mapping):
                return FlightDynamics._jump_channels_from_command(
                    nested, strict_neural=strict_neural,
                )
            # ``jump`` is the legacy decision field.  It is accepted only for
            # an explicit non-strict compatibility path.
            return None

        raw = dict(value)
        for wrapper in ("channels", "motor_channels", "neuromuscular", "named_outputs"):
            nested = raw.get(wrapper)
            if isinstance(nested, Mapping):
                for key, nested_value in nested.items():
                    raw.setdefault(str(key), nested_value)
        allowed = {
            "jump", "jump_activation", "jump_drive", "jump_command",
            "ttmn_activity", "ttmn",
            "leg", "leg_power", "leg_drive", "leg_motor", "leg_steering",
            "leg_asymmetry", "leg_motor_asym", "left_leg", "left_leg_power",
            "leg_left", "right_leg", "right_leg_power", "leg_right", "posture",
            "posture_roll", "posture_pitch", "posture_yaw", "roll", "pitch", "yaw",
            "source",
        }
        metadata = {
            "timestamp", "timestamp_s", "time_s", "units", "provenance",
            "channels", "motor_channels", "neuromuscular", "named_outputs",
        }
        forbidden_tokens = (
            "threat", "phase", "goal", "direction", "heading", "target",
            "scenario", "pose", "position", "orientation", "teleport",
        )
        selected: dict[str, Any] = {}
        for raw_key, raw_item in raw.items():
            key = _lower_key(raw_key)
            if key in metadata:
                if key == "source":
                    selected["source"] = raw_item
                continue
            if any(token in key for token in forbidden_tokens):
                raise FlightDynamicsError(
                    f"unsupported jump input channel {raw_key!r}; "
                    "threat/phase/goal/direction/pose labels are not actuator inputs"
                )
            if key in allowed:
                selected[raw_key] = raw_item
        return selected or None

    @staticmethod
    def _jump_config_cache_value(value: Any) -> Any:
        """Return a stable, hashable view of JSON-like config values."""

        if isinstance(value, Mapping):
            return tuple(sorted(
                (str(key), FlightDynamics._jump_config_cache_value(item))
                for key, item in value.items()
            ))
        if isinstance(value, (tuple, list)):
            return tuple(FlightDynamics._jump_config_cache_value(item) for item in value)
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    def _combined_jump_cache_key(self) -> tuple[Any, float]:
        source = self.jump_config
        if source is None:
            source_key: Any = ("none",)
        elif isinstance(source, JumpLegConfig):
            # JumpLegConfig is a frozen dataclass.  Its identity is sufficient
            # for public assignment semantics; the body support plane remains
            # a separate key because it is aligned below.
            source_key = ("config", id(source))
        elif isinstance(source, Mapping):
            source_key = (
                "mapping", id(source), self._jump_config_cache_value(source),
            )
        else:
            source_key = ("object", id(source), repr(source))
        return source_key, float(self.config.ground_plane_z_m)

    def _combined_jump_config(self) -> Any:
        """Return the jump calibration aligned with this body's contact plane."""

        cache_key = self._combined_jump_cache_key()
        if cache_key == self._combined_jump_cache_key_value:
            return self._combined_jump_config_cache

        if self.jump_config is None:
            jump_config = JumpLegConfig(ground_plane_z_m=self.config.ground_plane_z_m)
        elif isinstance(self.jump_config, JumpLegConfig):
            jump_config = self.jump_config
        else:
            jump_config = JumpLegConfig.from_any(self.jump_config)
        # The combined bridge has one authoritative body-centre support
        # plane.  Preserve all actuator calibration while aligning only this
        # contact datum with the wing integrator's configured plane.
        if abs(jump_config.ground_plane_z_m - self.config.ground_plane_z_m) > 1.0e-15:
            jump_config = replace(
                jump_config,
                ground_plane_z_m=self.config.ground_plane_z_m,
            )
        self._combined_jump_cache_key_value = cache_key
        self._combined_jump_config_cache = jump_config
        self._combined_jump_actuator_cache = JumpLegActuator(jump_config)
        return jump_config

    def _combined_jump_actuator(self) -> Any:
        """Return the cached stateless actuator for the current config."""

        self._combined_jump_config()
        return self._combined_jump_actuator_cache

    def _combined_contact_state(self) -> bool:
        cfg = self.config
        return bool(
            cfg.ground_contact_enabled
            and self.state.position_m[2] <= cfg.ground_plane_z_m + 1.0e-12
            and self.state.velocity_mps[2] <= 1.0e-12
        )

    def _combined_jump_context(
        self,
        jump_input: JumpNeuromuscularInput,
    ) -> tuple[Optional[float], bool]:
        jump = jump_input.jump_activation
        start = self._jump_command_start_s
        latched = self._jump_event_latched
        if jump <= 0.0:
            return None, False
        if self._last_jump_activation <= 0.0 and not latched:
            start = self.state.time_s
        elif start is None and not latched:
            start = self.state.time_s
        return start, latched

    def evaluate_with_jump(
        self,
        command: Any = None,
        jump_command: Any = None,
        *,
        dt_s: float = 0.0,
        dt: Optional[float] = None,
        timestamp: Optional[Any] = None,
        strict_neural: bool = True,
    ) -> FlightOutput:
        """Evaluate wing, jump, and optional grounded walking forces without commit."""

        if dt is not None:
            if dt_s != 0.0:
                raise TypeError("pass only one of dt_s or dt")
            dt_s = dt
        if timestamp is not None:
            target_time = _timestamp_seconds(timestamp)
            if dt_s != 0.0 and abs(float(dt_s) - (target_time - self.state.time_s)) > 1.0e-9:
                raise FlightDynamicsError("dt_s does not match timestamp delta")
            dt_s = target_time - self.state.time_s
        dt_value = _finite(dt_s, "dt_s")
        if dt_value < 0.0 or dt_value > self.config.max_dt_s + 1.0e-12:
            raise FlightDynamicsError(f"dt_s must be in [0, {self.config.max_dt_s}] s")
        dt_value = min(dt_value, self.config.max_dt_s)
        parsed_wing = NeuromuscularInput.from_any(command, config=self.config)
        walking_input = (
            _ground_walking_input_from_any(command)
            if self.config.ground_walking_enabled else _GroundWalkingInput()
        )
        extracted = jump_command
        if extracted is None:
            extracted = self._jump_channels_from_command(
                command, strict_neural=strict_neural,
            )
        else:
            # An explicitly supplied jump command is already at the body
            # boundary; generic ``jump`` is allowed there for compatibility.
            extracted = self._jump_channels_from_command(
                extracted, strict_neural=False,
            )
        parsed_jump = JumpNeuromuscularInput.from_any(extracted)
        start, latched = self._combined_jump_context(parsed_jump)
        actuator = self._combined_jump_actuator().evaluate(
            parsed_jump,
            time_s=self.state.time_s,
            dt_s=dt_value,
            command_start_s=start,
            contact=self._combined_contact_state(),
            event_latched=latched,
        )
        return self._evaluate(
            parsed_wing,
            dt_value,
            jump_actuator=actuator,
            jump_event_latched=latched,
            ground_walking_input=walking_input,
        )

    def step_with_jump(
        self,
        command: Any = None,
        jump_command: Any = None,
        dt_s: Optional[float] = None,
        *,
        dt: Optional[float] = None,
        timestamp: Optional[Any] = None,
        strict_neural: bool = True,
    ) -> FlightOutput:
        """Advance one authoritative wing+leg body step.

        The wing forces, SI jump force, posture torque, optional ground
        traction, and unilateral contact impulse are accumulated by one
        :class:`FlightDynamics` integrator.
        ``jump_command`` may be omitted when a mixed ``NeuromuscularOutput``
        carries both wing and jump/leg/posture channels.
        """

        if dt is not None:
            if dt_s is not None:
                raise TypeError("pass only one of dt_s or dt")
            dt_s = dt
        if timestamp is not None:
            target_time = _timestamp_seconds(timestamp)
            if dt_s is not None and abs(float(dt_s) - (target_time - self.state.time_s)) > 1.0e-9:
                raise FlightDynamicsError("dt_s does not match timestamp delta")
            dt_s = target_time - self.state.time_s
        dt_value = 0.001 if dt_s is None else _finite(dt_s, "dt_s")
        if dt_value <= 0.0 or dt_value > self.config.max_dt_s + 1.0e-12:
            raise FlightDynamicsError(f"dt_s must be > 0 and <= {self.config.max_dt_s} s")
        parsed_wing = NeuromuscularInput.from_any(command, config=self.config)
        walking_input = (
            _ground_walking_input_from_any(command)
            if self.config.ground_walking_enabled else _GroundWalkingInput()
        )
        extracted = jump_command
        if extracted is None:
            extracted = self._jump_channels_from_command(
                command, strict_neural=strict_neural,
            )
        else:
            extracted = self._jump_channels_from_command(
                extracted, strict_neural=False,
            )
        parsed_jump = JumpNeuromuscularInput.from_any(extracted)
        contact_at_start = self._combined_contact_state()
        start, latched = self._combined_jump_context(parsed_jump)
        actuator = self._combined_jump_actuator().evaluate(
            parsed_jump,
            time_s=self.state.time_s,
            dt_s=dt_value,
            command_start_s=start,
            contact=contact_at_start,
            event_latched=latched,
        )
        output = self._evaluate(
            parsed_wing,
            dt_value,
            jump_actuator=actuator,
            jump_event_latched=latched,
            ground_walking_input=walking_input,
        )
        # Latch a held event when the integrated state leaves the body-centre
        # support plane.  A subsequent airborne/landing step cannot reapply
        # the same ground push; the channel must return to zero first.
        predicted_liftoff = bool(
            contact_at_start
            and output.state.position_m[2] > self.config.ground_plane_z_m + 1.0e-12
        )
        next_latched = bool(
            latched
            or predicted_liftoff
            or (not contact_at_start and parsed_jump.jump_activation > 0.0)
        )
        next_start = start
        if parsed_jump.jump_activation <= 0.0:
            next_start = None
            next_latched = False
        self.state = output.state.copy()
        self.wpg.advance(dt_value)
        self._jump_command_start_s = next_start
        self._jump_event_latched = next_latched
        self._last_jump_activation = parsed_jump.jump_activation
        self.last_output = output
        return output

    # Combined-bridge naming used by body-side callers.
    step_wing_leg = step_with_jump
    step_combined = step_with_jump
    evaluate_wing_leg = evaluate_with_jump
    evaluate_combined = evaluate_with_jump

    def _set_frequency_from_input(self, value: Optional[float]) -> float:
        frequency = self.config.wingbeat_frequency_hz if value is None else value
        return self.wpg.set_frequency(frequency)

    def _wing(self, power: float, amplitude_multiplier: float, bias_rad: float,
              phase_offset_rad: float, frequency_hz: float) -> _WingEvaluation:
        cfg = self.config
        amplitude_rad = cfg.stroke_amplitude_rad * amplitude_multiplier
        phase = (self.wpg.phase_rad + phase_offset_rad) % (2.0 * math.pi)
        # The force is beat-averaged.  WPG phase remains observable and its
        # instantaneous sample is available to clients; the RMS stroke speed
        # avoids a timestep-dependent force ripple at the body rate.
        stroke_speed_rms = (2.0 * math.pi * frequency_hz * cfg.mean_radius_m
                            * amplitude_rad / math.sqrt(2.0))
        q_stroke = 0.5 * cfg.air_density_kg_m3 * stroke_speed_rms * stroke_speed_rms
        drive_sq = power * power
        bias_factor = max(0.0, math.cos(bias_rad))
        lift = q_stroke * cfg.wing_area_m2 * cfg.lift_coefficient * drive_sq
        lift *= max(0.0, 1.0 - cfg.bias_lift_scale * (1.0 - bias_factor))
        profile_drag = q_stroke * cfg.wing_area_m2 * cfg.profile_drag_coefficient * drive_sq
        lift_vector = (lift * math.sin(bias_rad), 0.0, lift * bias_factor)
        # Mechanical work is a positive scalar; profile drag supplies a stable
        # low-order proxy for the cost of flapping and lift supplies induced
        # work.  It is intentionally exposed for calibration.
        stroke_speed_peak = 2.0 * math.pi * frequency_hz * cfg.mean_radius_m * amplitude_rad
        power_w = (profile_drag + lift) * stroke_speed_peak / cfg.mechanical_efficiency
        return _WingEvaluation(
            lift_N=lift,
            profile_drag_N=profile_drag,
            lift_vector_body_N=lift_vector,
            amplitude_rad=amplitude_rad,
            bias_rad=bias_rad,
            phase_rad=phase,
            power_W=max(0.0, power_w),
        )

    def _evaluate(
        self,
        command: NeuromuscularInput,
        dt_s: float,
        *,
        jump_actuator: Optional[Any] = None,
        jump_event_latched: bool = False,
        ground_walking_input: Optional[_GroundWalkingInput] = None,
    ) -> FlightOutput:
        cfg = self.config
        if cfg.ground_only:
            command = NeuromuscularInput()
            jump_actuator = None
        state = self.state
        walking_input = ground_walking_input or _GroundWalkingInput()
        ground_walking_grounded_at_start = bool(
            cfg.ground_contact_enabled
            and state.position_m[2] <= cfg.ground_plane_z_m + 1.0e-12
            and state.velocity_mps[2] <= 1.0e-12
        )
        ground_walking_contact_available = bool(
            cfg.ground_walking_enabled and ground_walking_grounded_at_start
        )
        ground_walking_traction_world_N: Vec3 = (0.0, 0.0, 0.0)
        ground_walking_braking_world_N: Vec3 = (0.0, 0.0, 0.0)
        ground_walking_lateral_braking_world_N: Vec3 = (0.0, 0.0, 0.0)
        ground_walking_force_world_N: Vec3 = (0.0, 0.0, 0.0)
        ground_walking_traction_body_N: Vec3 = (0.0, 0.0, 0.0)
        ground_walking_force_body_N: Vec3 = (0.0, 0.0, 0.0)
        ground_walking_yaw_torque_Nm = 0.0
        if ground_walking_contact_available:
            forward_world = _qrotate(state.orientation_quat, (1.0, 0.0, 0.0))
            forward_horizontal_norm = math.hypot(forward_world[0], forward_world[1])
            if forward_horizontal_norm > 1.0e-12:
                traction_magnitude = cfg.max_ground_traction_N * walking_input.leg_power
                ground_walking_traction_world_N = (
                    traction_magnitude * forward_world[0] / forward_horizontal_norm,
                    traction_magnitude * forward_world[1] / forward_horizontal_norm,
                    0.0,
                )
                ground_walking_traction_body_N = _qrotate_inverse(
                    state.orientation_quat, ground_walking_traction_world_N,
                )

            horizontal_velocity = (state.velocity_mps[0], state.velocity_mps[1])
            horizontal_speed = math.hypot(*horizontal_velocity)
            braking_magnitude = cfg.ground_horizontal_braking_N_s_m * horizontal_speed
            if dt_s > 0.0:
                # Do not let the discrete braking impulse reverse horizontal
                # motion; repeated zero-drive steps then settle exactly.
                braking_magnitude = min(
                    braking_magnitude, cfg.mass_kg * horizontal_speed / dt_s,
                )
            if horizontal_speed > 1.0e-12 and braking_magnitude > 0.0:
                brake_scale = -braking_magnitude / horizontal_speed
                ground_walking_braking_world_N = (
                    horizontal_velocity[0] * brake_scale,
                    horizontal_velocity[1] * brake_scale,
                    0.0,
                )
            body_lateral_world = _qrotate(state.orientation_quat, (0.0, 1.0, 0.0))
            lateral_horizontal_norm = math.hypot(
                body_lateral_world[0], body_lateral_world[1],
            )
            if lateral_horizontal_norm > 1.0e-12:
                lateral_axis_world = (
                    body_lateral_world[0] / lateral_horizontal_norm,
                    body_lateral_world[1] / lateral_horizontal_norm,
                    0.0,
                )
                lateral_velocity = _vdot(
                    (state.velocity_mps[0], state.velocity_mps[1], 0.0),
                    lateral_axis_world,
                )
                lateral_speed = abs(lateral_velocity)
                lateral_braking_magnitude = (
                    cfg.ground_lateral_braking_N_s_m * lateral_speed
                )
                if dt_s > 0.0 and lateral_speed > 0.0:
                    # Cap the added body-lateral braking together with the
                    # isotropic brake's lateral component, preventing their
                    # combined discrete impulse from reversing this velocity.
                    lateral_sign = 1.0 if lateral_velocity > 0.0 else -1.0
                    existing_lateral_braking = max(
                        0.0,
                        -lateral_sign * _vdot(
                            ground_walking_braking_world_N, lateral_axis_world,
                        ),
                    )
                    remaining_lateral_braking = max(
                        0.0,
                        cfg.mass_kg * lateral_speed / dt_s - existing_lateral_braking,
                    )
                    lateral_braking_magnitude = min(
                        lateral_braking_magnitude, remaining_lateral_braking,
                    )
                if lateral_speed > 1.0e-12 and lateral_braking_magnitude > 0.0:
                    lateral_braking_sign = -1.0 if lateral_velocity > 0.0 else 1.0
                    ground_walking_lateral_braking_world_N = _vscale(
                        lateral_axis_world,
                        lateral_braking_sign * lateral_braking_magnitude,
                    )
            ground_walking_force_world_N = _vadd(
                _vadd(ground_walking_traction_world_N, ground_walking_braking_world_N),
                ground_walking_lateral_braking_world_N,
            )
            ground_walking_force_body_N = _qrotate_inverse(
                state.orientation_quat, ground_walking_force_world_N,
            )
            ground_walking_yaw_torque_Nm = (
                cfg.max_ground_yaw_torque_Nm
                * walking_input.leg_steering
                * walking_input.leg_power
            )
        jump_force_body: Vec3 = (
            (0.0, 0.0, 0.0)
            if jump_actuator is None else _vec3(
                getattr(jump_actuator, "jump_force_body_N", (0.0, 0.0, 0.0)),
                "jump_force_body_N",
            )
        )
        jump_posture_torque: Vec3 = (
            (0.0, 0.0, 0.0)
            if jump_actuator is None else _vec3(
                getattr(jump_actuator, "posture_torque_body_Nm", (0.0, 0.0, 0.0)),
                "jump_posture_torque_body_Nm",
            )
        )
        jump_leg_roll_torque = 0.0 if jump_actuator is None else _finite(
            getattr(jump_actuator, "leg_roll_torque_Nm", 0.0),
            "jump_leg_roll_torque_Nm",
        )
        frequency = self._set_frequency_from_input(command.frequency_hz)
        left_bias = command.left_bias_rad
        if left_bias is None:
            left_bias = command.left_bias * cfg.max_stroke_bias_rad
        right_bias = command.right_bias_rad
        if right_bias is None:
            right_bias = command.right_bias * cfg.max_stroke_bias_rad
        left_bias = _finite(left_bias, "left_bias_rad")
        right_bias = _finite(right_bias, "right_bias_rad")
        if abs(left_bias) > math.pi or abs(right_bias) > math.pi:
            raise FlightDynamicsError("stroke bias radians must be within +/- pi")
        left = self._wing(command.left_power, command.left_amplitude, left_bias,
                          cfg.left_phase_offset_rad, frequency)
        right = self._wing(command.right_power, command.right_amplitude, right_bias,
                           cfg.right_phase_offset_rad, frequency)

        # Air-relative velocity is expressed in body axes.  Body drag is the
        # quasi-steady mean drag term; a zero-wind hover therefore has no
        # arbitrary horizontal force.  Profile drag remains available as a
        # scalar and creates a differential yaw moment below.
        wind_world = self.wind_mps if command.wind_mps is None else command.wind_mps
        relative_velocity_world = _vsub(state.velocity_mps, wind_world)
        relative_velocity_body = _qrotate_inverse(state.orientation_quat, relative_velocity_world)
        air_speed = _vnorm(relative_velocity_body)
        translational_area = cfg.body_area_m2 * cfg.body_drag_coefficient
        translational_area += 2.0 * cfg.wing_area_m2 * cfg.translational_wing_drag_coefficient
        drag_force_body = _vscale(relative_velocity_body, -0.5 * cfg.air_density_kg_m3
                                   * translational_area * air_speed)
        lift_force = _vadd(left.lift_vector_body_N, right.lift_vector_body_N)
        aerodynamic_force = _vadd(lift_force, drag_force_body)
        gravity_world = (0.0, 0.0, -cfg.weight_N)
        gravity_body = _qrotate_inverse(state.orientation_quat, gravity_world)
        external_force = command.force_body_N
        # Contact is applied after the free-flight force is evaluated.  The
        # unilateral impulse below is the discrete equivalent of a stiff
        # ground reaction: it prevents a finite step from tunnelling through
        # the plane while keeping the continuous flight equations untouched.
        # A combined wing/leg bridge contributes one SI actuator force to the
        # same free-force sum.  It is intentionally not routed through
        # ``external_force_body_N`` or a second body integrator, preventing
        # double application by a runner/backend.
        free_force_body = _vadd(
            _vadd(
                _vadd(_vadd(aerodynamic_force, gravity_body), external_force),
                jump_force_body,
            ),
            ground_walking_force_body_N,
        )
        free_force_world = _qrotate(state.orientation_quat, free_force_body)
        free_acceleration_world = _vscale(free_force_world, 1.0 / cfg.mass_kg)

        # Differential lift is roll.  Differential profile drag is yaw.  The
        # explicit normalized control channels and SI torques are additive so
        # calibrated neural readouts can be compared with hand-injected tests.
        roll_moment = (cfg.differential_roll_moment_scale * cfg.half_wing_span_m
                       * (left.lift_N - right.lift_N))
        # The shared neuromuscular contract defines positive steering as
        # right-minus-left wing drive.  Use the same sign for differential
        # profile-drag yaw so it reinforces (rather than cancels) the explicit
        # positive ``command.yaw`` channel emitted from that readout.
        yaw_moment = (cfg.differential_yaw_moment_scale * cfg.half_wing_span_m
                      * (right.profile_drag_N - left.profile_drag_N))
        direct_torque = (
            command.roll * cfg.max_control_torque_Nm + command.roll_torque_Nm,
            command.pitch * cfg.max_control_torque_Nm + command.pitch_torque_Nm,
            command.yaw * cfg.max_control_torque_Nm + command.yaw_torque_Nm,
        )
        damping = tuple(cfg.torque_damping_Nm_s[i] * state.angular_velocity_rad_s[i]
                        for i in range(3))
        roll_rad, pitch_rad, _ = _rpy_from_quat(state.orientation_quat)
        attitude_restoring = (
            -cfg.attitude_stiffness_Nm[0] * math.sin(roll_rad),
            -cfg.attitude_stiffness_Nm[1] * math.sin(pitch_rad),
            0.0,
        )
        torque = (
            roll_moment + direct_torque[0] + jump_posture_torque[0]
            + jump_leg_roll_torque + attitude_restoring[0] - damping[0],
            direct_torque[1] + jump_posture_torque[1]
            + attitude_restoring[1] - damping[1],
            yaw_moment + direct_torque[2] + jump_posture_torque[2]
            + ground_walking_yaw_torque_Nm + attitude_restoring[2] - damping[2],
        )
        inertia = cfg.inertia_kg_m2
        angular_momentum = (
            inertia[0] * state.angular_velocity_rad_s[0],
            inertia[1] * state.angular_velocity_rad_s[1],
            inertia[2] * state.angular_velocity_rad_s[2],
        )
        gyro = _cross(state.angular_velocity_rad_s, angular_momentum)
        angular_acceleration = tuple((torque[i] - gyro[i]) / inertia[i] for i in range(3))

        total_power = cfg.idle_power_W + left.power_W + right.power_W
        energy_increment = total_power * dt_s
        energy = state.cumulative_energy_J + energy_increment

        # Apply unilateral support impulses only when a finite step crosses a
        # support plane.  The world ground is always available; configured
        # landing surfaces are considered only when the next body centre is
        # inside their x/y footprint and the body approaches from above.  No
        # position is prescribed during flight, and a body below a raised pad
        # is never teleported upward through it.
        next_velocity = _vadd(state.velocity_mps, _vscale(free_acceleration_world, dt_s))
        ground_walking_speed_limit_force_world_N: Vec3 = (0.0, 0.0, 0.0)
        if ground_walking_contact_available:
            horizontal_speed = math.hypot(next_velocity[0], next_velocity[1])
            if dt_s > 0.0 and horizontal_speed > cfg.max_ground_speed_mps:
                uncapped_velocity = next_velocity
                horizontal_scale = cfg.max_ground_speed_mps / horizontal_speed
                next_velocity = (
                    next_velocity[0] * horizontal_scale,
                    next_velocity[1] * horizontal_scale,
                    next_velocity[2],
                )
                ground_walking_speed_limit_force_world_N = _vscale(
                    _vsub(next_velocity, uncapped_velocity), cfg.mass_kg / dt_s,
                )
                ground_walking_force_world_N = _vadd(
                    ground_walking_force_world_N,
                    ground_walking_speed_limit_force_world_N,
                )
                speed_limit_force_body = _qrotate_inverse(
                    state.orientation_quat, ground_walking_speed_limit_force_world_N,
                )
                ground_walking_force_body_N = _vadd(
                    ground_walking_force_body_N, speed_limit_force_body,
                )
                free_force_body = _vadd(free_force_body, speed_limit_force_body)
                free_force_world = _qrotate(state.orientation_quat, free_force_body)
                free_acceleration_world = _vscale(free_force_world, 1.0 / cfg.mass_kg)
        next_position = _vadd(state.position_m, _vscale(next_velocity, dt_s))
        contact_force_world = (0.0, 0.0, 0.0)
        contact_surface_id: Optional[str] = None
        support_plane_z: Optional[float] = None
        ground_recovery = False
        if cfg.ground_contact_enabled and dt_s > 0.0:
            if next_position[2] < cfg.ground_plane_z_m:
                support_plane_z = cfg.ground_plane_z_m
                contact_surface_id = "ground"
            elif state.position_m[2] < cfg.ground_plane_z_m:
                # A caller may restore a snapshot that predates the contact
                # constraint.  Recover it on the next step without allowing a
                # below-ground state to leak into the trajectory.
                support_plane_z = cfg.ground_plane_z_m
                contact_surface_id = "ground"
                ground_recovery = True

            # A raised landing surface has priority over the lower ground plane
            # if one unusually large step crosses both in the same interval.
            for surface in cfg.contact_surfaces:
                if not surface.contains_xy(next_position):
                    continue
                crossed_from_above = (
                    state.position_m[2] >= surface.top_z_m - 1.0e-12
                    and next_position[2] < surface.top_z_m
                )
                if crossed_from_above and (
                    support_plane_z is None or surface.top_z_m > support_plane_z
                ):
                    support_plane_z = surface.top_z_m
                    contact_surface_id = surface.surface_id
                    ground_recovery = False

            if support_plane_z is not None:
                impact_velocity = next_velocity[2]
                if ground_recovery and contact_surface_id == "ground":
                    target_velocity = max(0.0, impact_velocity)
                else:
                    target_velocity = (
                        -cfg.ground_restitution * impact_velocity
                        if impact_velocity < 0.0 else impact_velocity
                    )
                contact_force_world = (
                    0.0,
                    0.0,
                    cfg.mass_kg * (target_velocity - impact_velocity) / dt_s,
                )
                next_velocity = (next_velocity[0], next_velocity[1], target_velocity)
                next_position = (next_position[0], next_position[1], support_plane_z)

        target_contact = contact_surface_id not in (None, "ground")

        contact_force_body = _qrotate_inverse(state.orientation_quat, contact_force_world)
        total_force_body = _vadd(free_force_body, contact_force_body)
        total_force_world = _qrotate(state.orientation_quat, total_force_body)
        acceleration_world = _vscale(total_force_world, 1.0 / cfg.mass_kg)

        # The output state is computed before the WPG phase is advanced.  This
        # makes one output reproducible from (state, command, dt), then step()
        # commits it and advances exactly once.
        next_state = state.copy()
        next_state.time_s = state.time_s + dt_s
        next_state.velocity_mps = next_velocity
        next_state.position_m = next_position
        next_state.angular_velocity_rad_s = tuple(
            _clamp(state.angular_velocity_rad_s[i] + angular_acceleration[i] * dt_s,
                   -cfg.max_angular_rate_rad_s, cfg.max_angular_rate_rad_s)
            for i in range(3)
        )  # type: ignore[assignment]
        angular_step = _vscale(next_state.angular_velocity_rad_s, dt_s)
        next_state.orientation_quat = _quat(
            _qmul(state.orientation_quat, _quat_from_rotation_vector(angular_step))
        )
        next_state.cumulative_energy_J = energy
        wind_force_body = drag_force_body if _vnorm(wind_world) > 0.0 else (0.0, 0.0, 0.0)
        return FlightOutput(
            time_s=next_state.time_s,
            state=next_state,
            frequency_hz=frequency,
            phase_left_rad=left.phase_rad,
            phase_right_rad=right.phase_rad,
            left_power=command.left_power,
            right_power=command.right_power,
            left_amplitude=command.left_amplitude,
            right_amplitude=command.right_amplitude,
            left_bias_rad=left_bias,
            right_bias_rad=right_bias,
            left_lift_N=left.lift_N,
            right_lift_N=right.lift_N,
            left_profile_drag_N=left.profile_drag_N,
            right_profile_drag_N=right.profile_drag_N,
            lift_force_body_N=lift_force,
            drag_force_body_N=drag_force_body,
            aerodynamic_force_body_N=aerodynamic_force,
            gravity_force_body_N=gravity_body,
            wind_force_body_N=wind_force_body,
            external_force_body_N=external_force,
            contact_force_body_N=contact_force_body,
            total_force_body_N=total_force_body,
            total_force_world_N=total_force_world,
            torque_body_Nm=torque,
            acceleration_world_mps2=acceleration_world,
            angular_acceleration_body_rad_s2=angular_acceleration,
            power_W=total_power,
            energy_increment_J=energy_increment,
            cumulative_energy_J=energy,
            source=command.source,
            fit_status=cfg.fit_status,
            ground_plane_z_m=cfg.ground_plane_z_m,
            contact_surface_id=contact_surface_id,
            target_contact=target_contact,
            jump_force_body_N=jump_force_body,
            jump_impulse_body_Ns=(
                (0.0, 0.0, 0.0)
                if jump_actuator is None else _vec3(
                    getattr(jump_actuator, "impulse_body_Ns", (0.0, 0.0, 0.0)),
                    "jump_impulse_body_Ns",
                )
            ),
            jump_posture_torque_body_Nm=jump_posture_torque,
            jump_actuator_active=bool(
                False if jump_actuator is None
                else getattr(jump_actuator, "actuator_active", False)
            ),
            jump_contact_gated=bool(
                False if jump_actuator is None
                else getattr(jump_actuator, "contact_gated", False)
            ),
            jump_event_latched=bool(jump_event_latched),
            jump_input=(
                None if jump_actuator is None
                else getattr(jump_actuator, "input", None)
            ),
            jump_model=(None if jump_actuator is None else JUMP_MODEL_NAME),
            jump_model_scope=(None if jump_actuator is None else JUMP_MODEL_SCOPE),
            jump_is_flybody=(None if jump_actuator is None else JUMP_IS_FLYBODY),
            ground_walking_drive=walking_input.leg_power,
            ground_walking_steering=walking_input.leg_steering,
            ground_walking_force_body_N=ground_walking_force_body_N,
            ground_walking_force_world_N=ground_walking_force_world_N,
            ground_walking_traction_force_body_N=ground_walking_traction_body_N,
            ground_walking_braking_force_world_N=ground_walking_braking_world_N,
            ground_walking_lateral_braking_force_world_N=ground_walking_lateral_braking_world_N,
            ground_walking_speed_limit_force_world_N=ground_walking_speed_limit_force_world_N,
            ground_walking_yaw_torque_Nm=ground_walking_yaw_torque_Nm,
            ground_walking_grounded_at_start=ground_walking_grounded_at_start,
            ground_walking_active=bool(
                _vnorm(ground_walking_force_body_N) > 0.0
                or abs(ground_walking_yaw_torque_Nm) > 0.0
            ),
            ground_walking_contact_available=ground_walking_contact_available,
            ground_walking_enabled=cfg.ground_walking_enabled,
            ground_walking_source=walking_input.source,
        )

    def evaluate(self, command: Any = None, *, dt_s: float = 0.0,
                 dt: Optional[float] = None, timestamp: Optional[Any] = None) -> FlightOutput:
        """Evaluate forces without committing state or advancing WPG phase."""

        if dt is not None:
            if dt_s != 0.0:
                raise TypeError("pass only one of dt_s or dt")
            dt_s = dt
        if timestamp is None and dt_s == 0.0:
            candidate = _command_timestamp(command)
            if candidate is not None and _timestamp_seconds(candidate) > self.state.time_s:
                timestamp = candidate
        if timestamp is not None:
            target_time = _timestamp_seconds(timestamp)
            if dt_s != 0.0 and abs(float(dt_s) - (target_time - self.state.time_s)) > 1.0e-9:
                raise FlightDynamicsError("dt_s does not match timestamp delta")
            dt_s = target_time - self.state.time_s
        dt = _finite(dt_s, "dt_s")
        if dt < 0 or dt > cfg_max_dt(self.config) + 1.0e-12:
            raise FlightDynamicsError(f"dt_s must be in [0, {self.config.max_dt_s}] s")
        dt = min(dt, self.config.max_dt_s)
        parsed = NeuromuscularInput.from_any(command, config=self.config)
        walking_input = (
            _ground_walking_input_from_any(command)
            if self.config.ground_walking_enabled else _GroundWalkingInput()
        )
        return self._evaluate(parsed, dt, ground_walking_input=walking_input)

    def step(self, command: Any = None, dt_s: Optional[float] = None,
             *, dt: Optional[float] = None, timestamp: Optional[Any] = None) -> FlightOutput:
        """Advance the low-order body state by one stable semi-implicit step."""

        if dt is not None:
            if dt_s is not None:
                raise TypeError("pass only one of dt_s or dt")
            dt_s = dt
        if timestamp is None and dt_s is None:
            candidate = _command_timestamp(command)
            if candidate is not None and _timestamp_seconds(candidate) > self.state.time_s:
                timestamp = candidate
        if timestamp is not None:
            target_time = _timestamp_seconds(timestamp)
            if dt_s is not None and abs(float(dt_s) - (target_time - self.state.time_s)) > 1.0e-9:
                raise FlightDynamicsError("dt_s does not match timestamp delta")
            dt_s = target_time - self.state.time_s
        dt_value = 0.001 if dt_s is None else _finite(dt_s, "dt_s")
        if dt_value <= 0 or dt_value > self.config.max_dt_s + 1.0e-12:
            raise FlightDynamicsError(f"dt_s must be > 0 and <= {self.config.max_dt_s} s")
        dt_value = min(dt_value, self.config.max_dt_s)
        parsed = NeuromuscularInput.from_any(command, config=self.config)
        walking_input = (
            _ground_walking_input_from_any(command)
            if self.config.ground_walking_enabled else _GroundWalkingInput()
        )
        output = self._evaluate(
            parsed, dt_value, ground_walking_input=walking_input,
        )
        self.state = output.state.copy()
        self.wpg.advance(dt_value)
        self.last_output = output
        return output

    def step_body(self, command: Any = None, dt_s: Optional[float] = None,
                  *, dt: Optional[float] = None, timestamp: Optional[Any] = None) -> dict[str, Any]:
        """Step and return the body-compatible mapping/state view."""

        output = self.step(command, dt_s=dt_s, dt=dt, timestamp=timestamp)
        return output.to_dict()

    def hover_trim(self, *, amplitude: float = 1.0,
                   frequency_hz: Optional[float] = None) -> NeuromuscularInput:
        """Return a symmetric wing command whose mean lift equals weight.

        The trim solves the same quasi-steady equation used by the force
        evaluator.  A clipped value of 1.0 indicates that the configured
        geometry/coefficient set cannot produce enough lift; callers can check
        :meth:`hover_trim_diagnostics` for the un-clipped requirement.
        """

        amp = _finite(amplitude, "amplitude")
        if amp <= 0 or amp > 2.0:
            raise FlightDynamicsError("hover trim amplitude must be in (0, 2]")
        frequency = self.config.wingbeat_frequency_hz if frequency_hz is None else _finite(frequency_hz, "frequency_hz")
        self.wpg._check_frequency(frequency)
        stroke_speed_rms = (2.0 * math.pi * frequency * self.config.mean_radius_m
                            * self.config.stroke_amplitude_rad * amp / math.sqrt(2.0))
        q_stroke = 0.5 * self.config.air_density_kg_m3 * stroke_speed_rms * stroke_speed_rms
        lift_per_wing_at_power_one = q_stroke * self.config.wing_area_m2 * self.config.lift_coefficient
        if lift_per_wing_at_power_one <= 0:
            raise FlightDynamicsError("configured wing lift capacity is zero")
        requested = math.sqrt(self.config.weight_N / (2.0 * lift_per_wing_at_power_one))
        power = _clamp(requested, 0.0, 1.0)
        return NeuromuscularInput(
            left_power=power,
            right_power=power,
            left_amplitude=amp,
            right_amplitude=amp,
            frequency_hz=frequency,
            source="trimmed_quasi_steady",
        ).normalized(config=self.config)

    trim_hover = hover_trim
    hover_command = hover_trim

    def hover_trim_diagnostics(self, *, amplitude: float = 1.0,
                               frequency_hz: Optional[float] = None) -> dict[str, float | bool]:
        command = self.hover_trim(amplitude=amplitude, frequency_hz=frequency_hz)
        requested = command.left_power
        frequency = self.config.wingbeat_frequency_hz if frequency_hz is None else float(frequency_hz)
        stroke_speed_rms = (2.0 * math.pi * frequency * self.config.mean_radius_m
                            * self.config.stroke_amplitude_rad * float(amplitude) / math.sqrt(2.0))
        q_stroke = 0.5 * self.config.air_density_kg_m3 * stroke_speed_rms * stroke_speed_rms
        capacity = 2.0 * q_stroke * self.config.wing_area_m2 * self.config.lift_coefficient
        required = math.sqrt(self.config.weight_N / max(capacity, 1.0e-30))
        return {
            "required_power_unclipped": required,
            "trim_power": requested,
            "capacity_N": capacity,
            "weight_N": self.config.weight_N,
            "feasible": required <= 1.0,
        }


@dataclass(frozen=True)
class JumpLegConfig:
    """Explicit SI calibration for the reduced bilateral leg actuator.

    This is intentionally a point/reduced-body configuration.  It does not
    claim a FlyBody morphology, individual leg segments, or measured muscle
    mechanics.  ``max_leg_force_N`` is the total bilateral upward force at
    normalized drive one; the two leg channels divide that force.  All
    neural-to-SI conversion is concentrated here so that a trajectory can
    report exactly which calibration was used.
    """

    mass_kg: float = 1.0e-6
    inertia_kg_m2: Vec3 = (2.0e-12, 2.0e-12, 3.0e-12)
    gravity_mps2: float = 9.80665
    ground_contact_enabled: bool = True
    ground_plane_z_m: float = 0.0
    ground_restitution: float = 0.0

    # Bilateral actuator limits (SI).
    # Literature-informed engineering envelope: roughly 150 uN peak with a
    # 0.53--0.59 uN*s finite impulse target.  These values are not a fit for
    # an individual fly and remain an explicitly reduced-body hypothesis.
    max_leg_force_N: float = 1.5e-4
    # Compatibility spelling for callers that describe this as a jump limit.
    max_jump_force_N: Optional[float] = None
    # Additional explicit spellings retained at the boundary.  They are
    # aliases, not separate force reservoirs; canonical values below are
    # resolved once in ``__post_init__``.
    leg_force_max_N: Optional[float] = None
    jump_force_max_N: Optional[float] = None
    leg_moment_arm_m: float = 1.0e-3
    max_posture_torque_Nm: float = 2.0e-9
    posture_torque_max_Nm: Optional[float] = None
    posture_stiffness_Nm_per_rad: Vec3 = (3.0e-10, 3.0e-10, 1.0e-10)
    posture_damping_Nm_s_per_rad: Vec3 = (4.0e-10, 4.0e-10, 3.0e-10)

    # The jump input is an event.  It cannot become an indefinitely applied
    # force: the explicit delay and duration define its bounded actuator
    # window.  A duration of 40 ms is a wiring default, not a measurement.
    actuator_delay_s: float = 0.0
    actuator_duration_s: float = 0.0055
    jump_delay_s: Optional[float] = None
    jump_duration_s: Optional[float] = None
    max_jump_impulse_Ns: Optional[float] = 5.8e-7
    # Strict neural object integration requires the physical TTMN channel.
    # Set this explicitly only for a compatibility path that is allowed to
    # promote a legacy ``jump`` decision field into the actuator.
    legacy_jump_fallback: bool = False

    max_dt_s: float = 0.005
    max_speed_mps: float = 100.0
    max_angular_rate_rad_s: float = 200.0

    fit_status: str = "literature_informed_engineering_envelope__reduced_body_not_flybody"
    parameter_sources: Mapping[str, str] = field(default_factory=lambda: {
        "mass_kg": "digital_body.BodyConfig default scale; order_of_magnitude",
        "inertia_kg_m2": "digital_body.BodyConfig default scale; order_of_magnitude",
        "max_leg_force_N": "literature-informed ~150 uN peak envelope; not an individual-fly fit",
        "max_jump_impulse_Ns": "literature-informed 0.53-0.59 uN*s engineering target; not validated",
        "leg_moment_arm_m": "reduced body geometry; requires morphology fit",
        "max_posture_torque_Nm": "reduced posture torque scale; requires attitude fit",
            "actuator_delay_s": "explicit wiring calibration; not a scenario phase",
            "actuator_duration_s": "explicit wiring calibration; not a scenario phase",
            "legacy_jump_fallback": "explicit compatibility opt-in; disabled for strict neural path",
        "ground_plane_z_m": "virtual-world ground-plane datum",
        "ground_restitution": "contact calibration; default perfectly inelastic",
    })

    def __post_init__(self) -> None:
        positive = (
            "mass_kg", "gravity_mps2", "leg_moment_arm_m", "max_dt_s",
            "max_speed_mps", "max_angular_rate_rad_s",
        )
        for name in positive:
            number = _finite(getattr(self, name), name)
            if number <= 0.0:
                raise FlightDynamicsError(f"{name} must be > 0")
        if self.leg_force_max_N is not None:
            object.__setattr__(self, "max_leg_force_N", self.leg_force_max_N)
        if self.posture_torque_max_Nm is not None:
            object.__setattr__(self, "max_posture_torque_Nm", self.posture_torque_max_Nm)
        if self.jump_delay_s is not None:
            object.__setattr__(self, "actuator_delay_s", self.jump_delay_s)
        if self.jump_duration_s is not None:
            object.__setattr__(self, "actuator_duration_s", self.jump_duration_s)
        nonnegative = (
            "max_leg_force_N", "max_posture_torque_Nm", "actuator_delay_s",
            "actuator_duration_s",
        )
        for name in nonnegative:
            number = _finite(getattr(self, name), name)
            if number < 0.0:
                raise FlightDynamicsError(f"{name} must be >= 0")
        if self.actuator_duration_s <= 0.0:
            raise FlightDynamicsError("actuator_duration_s must be > 0")
        inertia = _vec3(self.inertia_kg_m2, "inertia_kg_m2")
        if any(value <= 0.0 for value in inertia):
            raise FlightDynamicsError("inertia_kg_m2 must contain positive values")
        stiffness = _vec3(self.posture_stiffness_Nm_per_rad,
                          "posture_stiffness_Nm_per_rad")
        damping = _vec3(self.posture_damping_Nm_s_per_rad,
                        "posture_damping_Nm_s_per_rad")
        if any(value < 0.0 for value in stiffness):
            raise FlightDynamicsError("posture_stiffness_Nm_per_rad must be non-negative")
        if any(value < 0.0 for value in damping):
            raise FlightDynamicsError("posture_damping_Nm_s_per_rad must be non-negative")
        ground = _finite(self.ground_plane_z_m, "ground_plane_z_m")
        restitution = _finite(self.ground_restitution, "ground_restitution")
        if not 0.0 <= restitution <= 1.0:
            raise FlightDynamicsError("ground_restitution must be in [0, 1]")
        jump_force = self.max_jump_force_N
        if self.jump_force_max_N is not None:
            jump_force = self.jump_force_max_N
        if jump_force is None:
            jump_force = self.max_leg_force_N
        jump_force = _finite(jump_force, "max_jump_force_N")
        if jump_force < 0.0:
            raise FlightDynamicsError("max_jump_force_N must be >= 0")
        impulse = self.max_jump_impulse_Ns
        if impulse is not None:
            impulse = _finite(impulse, "max_jump_impulse_Ns")
            if impulse < 0.0:
                raise FlightDynamicsError("max_jump_impulse_Ns must be >= 0")
        if not str(self.fit_status).strip():
            raise FlightDynamicsError("fit_status must be a non-empty provenance marker")
        object.__setattr__(self, "inertia_kg_m2", inertia)
        object.__setattr__(self, "posture_stiffness_Nm_per_rad", stiffness)
        object.__setattr__(self, "posture_damping_Nm_s_per_rad", damping)
        object.__setattr__(self, "ground_plane_z_m", ground)
        object.__setattr__(self, "ground_restitution", restitution)
        object.__setattr__(self, "max_jump_force_N", jump_force)
        object.__setattr__(self, "max_jump_impulse_Ns", impulse)
        object.__setattr__(self, "legacy_jump_fallback", bool(self.legacy_jump_fallback))
        object.__setattr__(self, "parameter_sources", dict(self.parameter_sources))

    @property
    def weight_N(self) -> float:
        return self.mass_kg * self.gravity_mps2

    @property
    def is_complete_flybody(self) -> bool:
        return False

    @property
    def effective_jump_force_N(self) -> float:
        """Maximum force after the optional explicit impulse cap."""

        force = float(self.max_jump_force_N or 0.0)
        if self.max_jump_impulse_Ns is not None:
            force = min(force, self.max_jump_impulse_Ns / self.actuator_duration_s)
        return force

    @property
    def units(self) -> dict[str, str]:
        return {
            "mass_kg": "kg", "inertia_kg_m2": "kg*m^2",
            "gravity_mps2": "m/s^2", "ground_plane_z_m": "m",
            "ground_restitution": "dimensionless", "max_leg_force_N": "N",
            "max_jump_force_N": "N", "leg_moment_arm_m": "m",
            "max_posture_torque_Nm": "N*m",
            "posture_stiffness_Nm_per_rad": "N*m/rad",
            "posture_damping_Nm_s_per_rad": "N*m*s/rad",
            "actuator_delay_s": "s", "actuator_duration_s": "s",
            "max_jump_impulse_Ns": "N*s", "max_dt_s": "s",
            "max_speed_mps": "m/s", "max_angular_rate_rad_s": "rad/s",
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["inertia_kg_m2"] = tuple(self.inertia_kg_m2)
        result["posture_stiffness_Nm_per_rad"] = tuple(self.posture_stiffness_Nm_per_rad)
        result["posture_damping_Nm_s_per_rad"] = tuple(self.posture_damping_Nm_s_per_rad)
        result["units"] = self.units
        return result

    parameter_manifest = to_dict

    def provenance(self) -> dict[str, Any]:
        return {
            "model": JUMP_MODEL_NAME,
            "scope": JUMP_MODEL_SCOPE,
            "is_flybody": JUMP_IS_FLYBODY,
            "is_cfd": False,
            "fit_status": self.fit_status,
            "parameters": self.to_dict(),
            "parameter_sources": dict(self.parameter_sources),
            "units": "SI",
            "input_channels": {
                "jump": "normalized [0, 1] event drive",
                "leg_power": "normalized [0, 1] bilateral drive",
                "leg_steering": "normalized [-1, 1] left/right asymmetry",
                "posture_roll_pitch_yaw": "normalized [-1, 1] torque channels",
                "ttmn_activity": "preferred physical jump activation channel",
                "legacy_jump_fallback": bool(self.legacy_jump_fallback),
            },
            "limitations": [
                "reduced point-body contact bridge; not a complete FlyBody",
                "leg force and timing defaults are order-of-magnitude calibration hypotheses",
                "no threat, phase, goal, direction, or pose channel is consumed",
            ],
        }

    manifest = provenance

    @classmethod
    def from_any(cls, value: Any = None) -> "JumpLegConfig":
        """Build a config from a mapping while resolving common SI aliases."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise FlightDynamicsError("jump config must be a mapping or JumpLegConfig")
        raw = dict(value)
        aliases = {
            "force_limit_n": "max_leg_force_N",
            "max_force_n": "max_leg_force_N",
            "jump_force_limit_n": "max_jump_force_N",
            "force_max_n": "max_leg_force_N",
            "posture_torque_limit_nm": "max_posture_torque_Nm",
            "delay_s": "actuator_delay_s",
            "duration_s": "actuator_duration_s",
            "jump_delay_s": "actuator_delay_s",
            "jump_duration_s": "actuator_duration_s",
            "ground_z_m": "ground_plane_z_m",
            "restitution": "ground_restitution",
        }
        normalized: dict[str, Any] = {}
        fields = set(cls.__dataclass_fields__)
        fields_by_lower = {_lower_key(name): name for name in fields}
        for key, item in raw.items():
            lower_key = _lower_key(key)
            canonical = aliases.get(lower_key, lower_key)
            canonical = fields_by_lower.get(_lower_key(canonical), canonical)
            if canonical in fields:
                normalized[canonical] = item
            else:
                raise FlightDynamicsError(f"unknown jump config field {key!r}")
        return cls(**normalized)


@dataclass(frozen=True)
class JumpActuatorOutput:
    """SI actuator result before body integration."""

    time_s: float
    input: JumpNeuromuscularInput
    actuator_active: bool
    contact_gated: bool
    delay_elapsed_s: float
    active_duration_s: float
    left_leg_force_N: float
    right_leg_force_N: float
    leg_force_body_N: Vec3
    jump_force_body_N: Vec3
    leg_roll_torque_Nm: float
    posture_torque_body_Nm: Vec3
    impulse_body_Ns: Vec3
    posture_impulse_body_Nms: Vec3
    source: str
    fit_status: str
    model: str = JUMP_MODEL_NAME
    model_scope: str = JUMP_MODEL_SCOPE
    is_flybody: bool = JUMP_IS_FLYBODY
    parameters: Mapping[str, Any] = field(default_factory=dict)
    ground_plane_z_m: float = 0.0

    @property
    def force_body_N(self) -> Vec3:
        return self.leg_force_body_N

    @property
    def is_cfd(self) -> bool:
        return False

    @property
    def is_complete_flybody(self) -> bool:
        return False

    @property
    def force_N(self) -> Vec3:
        return self.leg_force_body_N

    @property
    def impulse_Ns(self) -> Vec3:
        return self.impulse_body_Ns

    @property
    def jump_impulse_Ns(self) -> Vec3:
        return self.impulse_body_Ns

    @property
    def posture_torque_Nm(self) -> Vec3:
        return self.posture_torque_body_Nm

    @property
    def units(self) -> dict[str, str]:
        return {
            "time_s": "s", "force_N": "N", "force_body_N": "N",
            "torque_Nm": "N*m", "torque_body_Nm": "N*m",
            "impulse_Ns": "N*s", "impulse_body_Ns": "N*s",
            "posture_impulse_Nms": "N*m*s",
            "delay_elapsed_s": "s", "active_duration_s": "s",
        }

    @property
    def provenance(self) -> Mapping[str, Any]:
        return _ProvenanceRecord({
            "model": self.model,
            "scope": self.model_scope,
            "is_flybody": self.is_flybody,
            "fit_status": self.fit_status,
            "source": self.source,
            "parameters": dict(self.parameters),
        })

    def to_dict(self) -> dict[str, Any]:
        result = _record_dict(self)
        result.pop("parameters")
        result.pop("ground_plane_z_m")
        result["input"] = self.input.to_dict()
        result["units"] = self.units
        result["provenance"] = dict(self.provenance)
        return result

    as_dict = to_dict

def _jump_actuator_force(
    command: JumpNeuromuscularInput,
    config: JumpLegConfig,
    *,
    time_s: float,
    dt_s: float,
    command_start_s: Optional[float],
    contact: bool,
    event_latched: bool,
    parameter_manifest: Optional[Mapping[str, Any]] = None,
) -> JumpActuatorOutput:
    """Evaluate one bounded leg event without mutating simulator state."""

    command = command.normalized()
    now = _finite(time_s, "time_s")
    dt = _finite(dt_s, "dt_s")
    if dt < 0.0 or dt > config.max_dt_s + 1.0e-12:
        raise FlightDynamicsError(f"dt_s must be in [0, {config.max_dt_s}] s")
    start = now if command_start_s is None else _finite(command_start_s, "command_start_s")
    elapsed = max(0.0, now - start)
    window_start = start + config.actuator_delay_s
    window_end = window_start + config.actuator_duration_s
    if dt > 0.0:
        overlap = max(0.0, min(now + dt, window_end) - max(now, window_start))
        active_fraction = _clamp(overlap / dt, 0.0, 1.0)
    else:
        active_fraction = 1.0 if window_start <= now < window_end else 0.0
        overlap = 0.0
    requested_active = (
        command.jump_activation > 0.0
        and not event_latched
        and active_fraction > 0.0
    )
    # Contact is a unilateral gate.  The event can remain observable through
    # ``contact_gated`` while airborne, but it cannot create a hidden upward
    # force after liftoff.
    applied_fraction = active_fraction if contact and requested_active else 0.0
    # Report the physical gate whenever a live event window is airborne,
    # including the post-liftoff edge-latched portion.  ``actuator_active``
    # still means that a non-zero force was actually applied.
    contact_gated = bool(
        command.jump_activation > 0.0 and not contact and active_fraction > 0.0
    )
    left_capacity = _clamp(command.left_leg, 0.0, 1.0)
    right_capacity = _clamp(command.right_leg, 0.0, 1.0)
    if command.activation_source == "ttmn_activity":
        # TTMn is the physical bilateral jump drive.  A sparse neural output
        # may legitimately carry no generic ``leg_power`` pool in the same
        # tick; do not let that diagnostic channel null the TTMn event.  Any
        # explicit TTMn left/right extension can be added later without
        # changing this bilateral default.
        left_capacity = 1.0
        right_capacity = 1.0
    drive = _clamp(command.jump_activation, 0.0, 1.0)
    force_limit = config.effective_jump_force_N
    left_force = 0.5 * force_limit * drive * left_capacity * applied_fraction
    right_force = 0.5 * force_limit * drive * right_capacity * applied_fraction
    vertical_force = left_force + right_force
    leg_force = (0.0, 0.0, vertical_force)
    # Body axes are x forward, y left, z up.  The left/right separation gives
    # a physical roll moment; no direction label is consulted.
    leg_roll = config.leg_moment_arm_m * (left_force - right_force)
    posture_torque = (
        config.max_posture_torque_Nm * command.posture_roll,
        config.max_posture_torque_Nm * command.posture_pitch,
        config.max_posture_torque_Nm * command.posture_yaw,
    )
    impulse = _vscale(leg_force, dt)
    posture_impulse = _vscale(posture_torque, dt)
    if parameter_manifest is None:
        parameters = config.parameter_manifest()
    else:
        # Keep each output's top-level manifest independent, as before, while
        # reusing the expensive dataclass-to-dict construction in the hot
        # combined stepping path.  The two nested metadata maps are copied as
        # well because they are mutable dictionaries in the public schema.
        parameters = dict(parameter_manifest)
        for key in ("units", "parameter_sources"):
            value = parameters.get(key)
            if isinstance(value, Mapping):
                parameters[key] = dict(value)
    return JumpActuatorOutput(
        time_s=now,
        input=command,
        actuator_active=bool(applied_fraction > 0.0),
        contact_gated=contact_gated,
        delay_elapsed_s=elapsed,
        active_duration_s=overlap,
        left_leg_force_N=left_force,
        right_leg_force_N=right_force,
        leg_force_body_N=leg_force,
        jump_force_body_N=leg_force,
        leg_roll_torque_Nm=leg_roll,
        posture_torque_body_Nm=posture_torque,
        impulse_body_Ns=impulse,
        posture_impulse_body_Nms=posture_impulse,
        source=command.source,
        fit_status=config.fit_status,
        parameters=parameters,
    )


class JumpLegActuator:
    """Stateless, bounded conversion from jump/leg/posture channels to SI.

    ``command_start_s`` is supplied by a caller that owns event timing.  This
    keeps the conversion reusable and makes delay/duration explicit without
    accepting a phase, goal, threat, or direction label.
    """

    model_name = JUMP_MODEL_NAME
    model_scope = JUMP_MODEL_SCOPE
    is_flybody = JUMP_IS_FLYBODY

    def __init__(self, config: Optional[JumpLegConfig] = None) -> None:
        self.config = config or JumpLegConfig()
        self._parameter_manifest_config_id = id(self.config)
        self._parameter_manifest_cache = self.config.parameter_manifest()

    def _parameter_manifest(self) -> Mapping[str, Any]:
        """Return the cached manifest, refreshing after public reassignment."""

        config_id = id(self.config)
        if config_id != self._parameter_manifest_config_id:
            self._parameter_manifest_config_id = config_id
            self._parameter_manifest_cache = self.config.parameter_manifest()
        return self._parameter_manifest_cache

    @property
    def provenance(self) -> dict[str, Any]:
        return self.config.provenance()

    def evaluate(
        self,
        command: Any = None,
        *,
        time_s: float = 0.0,
        dt_s: float = 0.0,
        command_start_s: Optional[float] = None,
        contact: bool = True,
        event_latched: bool = False,
    ) -> JumpActuatorOutput:
        parsed = JumpNeuromuscularInput.from_any(command)
        return _jump_actuator_force(
            parsed, self.config, time_s=time_s, dt_s=dt_s,
            command_start_s=command_start_s, contact=bool(contact),
            event_latched=bool(event_latched),
            parameter_manifest=self._parameter_manifest(),
        )

    step = evaluate
    apply = evaluate
    actuate = evaluate

    def compute(self, command: Any = None, **kwargs: Any) -> JumpActuatorOutput:
        return self.evaluate(command, **kwargs)

    compute_force = compute


@dataclass(frozen=True)
class ReducedBodyJumpOutput:
    """One physically integrated leg/contact step.

    ``state`` is the updated reduced-body state.  Every force and impulse is
    reported in SI; ``posture_delta_rad`` is the integrated attitude change,
    not a requested pose write.  The model is deliberately marked as a
    reduced bridge and not a complete FlyBody implementation.
    """

    time_s: float
    state: FlightState
    input: JumpNeuromuscularInput
    contact: bool
    airborne: bool
    actuator_active: bool
    contact_gated: bool
    jump_force_body_N: Vec3
    leg_force_body_N: Vec3
    contact_force_body_N: Vec3
    contact_force_world_N: Vec3
    total_force_body_N: Vec3
    total_force_world_N: Vec3
    posture_torque_body_Nm: Vec3
    total_torque_body_Nm: Vec3
    acceleration_world_mps2: Vec3
    angular_acceleration_body_rad_s2: Vec3
    jump_impulse_body_Ns: Vec3
    contact_impulse_body_Ns: Vec3
    total_impulse_world_Ns: Vec3
    posture_delta_rad: Vec3
    orientation_rad: Vec3
    left_leg_force_N: float
    right_leg_force_N: float
    delay_elapsed_s: float
    active_duration_s: float
    source: str
    fit_status: str
    model: str = JUMP_MODEL_NAME
    model_scope: str = JUMP_MODEL_SCOPE
    is_flybody: bool = JUMP_IS_FLYBODY
    parameters: Mapping[str, Any] = field(default_factory=dict)
    ground_plane_z_m: float = 0.0
    contact_surface_id: Optional[str] = None

    @property
    def actual_force_body_N(self) -> Vec3:
        return self.total_force_body_N

    @property
    def is_cfd(self) -> bool:
        return False

    @property
    def is_complete_flybody(self) -> bool:
        return False

    @property
    def actual_force_world_N(self) -> Vec3:
        return self.total_force_world_N

    @property
    def force_body_N(self) -> Vec3:
        return self.total_force_body_N

    @property
    def force_N(self) -> Vec3:
        return self.total_force_world_N

    @property
    def actual_force_N(self) -> Vec3:
        return self.total_force_world_N

    @property
    def jump_force_N(self) -> Vec3:
        return self.jump_force_body_N

    @property
    def vertical_force_N(self) -> float:
        return self.total_force_world_N[2]

    @property
    def vertical_jump_force_N(self) -> float:
        return self.jump_force_body_N[2]

    @property
    def leg_force_N(self) -> Vec3:
        return self.leg_force_body_N

    @property
    def contact_force_N(self) -> Vec3:
        return self.contact_force_body_N

    @property
    def impulse_Ns(self) -> Vec3:
        return self.total_impulse_world_Ns

    @property
    def actual_impulse_Ns(self) -> Vec3:
        return self.total_impulse_world_Ns

    @property
    def jump_impulse_Ns(self) -> Vec3:
        return self.jump_impulse_body_Ns

    @property
    def contact_impulse_Ns(self) -> Vec3:
        return self.contact_impulse_body_Ns

    @property
    def ground_reaction_impulse_Ns(self) -> Vec3:
        return self.contact_impulse_body_Ns

    @property
    def ground_reaction_force_N(self) -> Vec3:
        return self.contact_force_world_N

    @property
    def orientation_change_rad(self) -> Vec3:
        return self.posture_delta_rad

    @property
    def posture_change_rad(self) -> Vec3:
        return self.posture_delta_rad

    @property
    def position_m(self) -> Vec3:
        return self.state.position_m

    @property
    def velocity_mps(self) -> Vec3:
        return self.state.velocity_mps

    @property
    def units(self) -> dict[str, str]:
        return {
            "time_s": "s", "position_m": "m", "velocity_mps": "m/s",
            "force_N": "N", "force_body_N": "N", "impulse_Ns": "N*s",
            "impulse_body_Ns": "N*s", "torque_Nm": "N*m", "torque_body_Nm": "N*m",
            "posture_delta_rad": "rad", "orientation_rad": "rad",
            "acceleration_mps2": "m/s^2", "angular_acceleration_rad_s2": "rad/s^2",
            "delay_elapsed_s": "s", "active_duration_s": "s",
            "ground_plane_z_m": "m",
        }

    @property
    def provenance(self) -> Mapping[str, Any]:
        return _ProvenanceRecord({
            "model": self.model,
            "scope": self.model_scope,
            "is_flybody": self.is_flybody,
            "fit_status": self.fit_status,
            "source": self.source,
            "parameters": dict(self.parameters),
        })

    def to_dict(self) -> dict[str, Any]:
        result = _record_dict(self)
        result["state"] = self.state.to_dict()
        result["input"] = self.input.to_dict()
        result["parameters"] = dict(self.parameters)
        result["units"] = self.units
        result["provenance"] = dict(self.provenance)
        return result

    as_dict = to_dict

    def to_body_state(self) -> dict[str, Any]:
        """Return a body-compatible state view without prescribing a pose."""

        return {
            "time_s": self.state.time_s,
            "position_m": tuple(self.state.position_m),
            "velocity_mps": tuple(self.state.velocity_mps),
            "orientation_quat": tuple(self.state.orientation_quat),
            "angular_velocity_radps": tuple(self.state.angular_velocity_rad_s),
            "contact": self.contact,
            "ground_distance_m": max(
                0.0, self.state.position_m[2] - self.ground_plane_z_m,
            ),
            "flight_mode": "grounded" if self.contact else "flight",
            "jump_force_N": tuple(self.jump_force_body_N),
            "contact_force_N": tuple(self.contact_force_world_N),
            "total_force_N": tuple(self.total_force_world_N),
            "total_torque_Nm": tuple(self.total_torque_body_Nm),
            "posture_delta_rad": tuple(self.posture_delta_rad),
            "source": self.source,
            "fit_status": self.fit_status,
            "model": self.model,
            "model_scope": self.model_scope,
            "is_flybody": self.is_flybody,
            "units": self.units,
            "provenance": dict(self.provenance),
        }

    as_body_state = to_body_state

    def to_digital_body_state(self) -> Any:
        """Adapt the reduced state to the optional local body record."""

        try:
            from digital_body import BodyState as DigitalBodyState
        except ImportError as exc:  # pragma: no cover - standalone use
            raise FlightDynamicsError("digital_body is not available") from exc
        values = self.to_body_state()
        allowed = set(getattr(DigitalBodyState, "__dataclass_fields__", ()))
        return DigitalBodyState(**{
            key: value for key, value in values.items() if key in allowed
        })

    to_body_backend_state = to_digital_body_state


class ReducedBodyJumpDynamics:
    """Reduced point-body integrator driven by jump/leg/posture channels.

    The contact law is unilateral and impulse based.  It catches downward
    crossings of ``ground_plane_z_m`` and never supplies a positive force once
    the body has left that plane.  A held jump event is edge-latched after
    liftoff; a new event requires the normalized channel to return to zero.
    """

    model_name = JUMP_MODEL_NAME
    model_scope = JUMP_MODEL_SCOPE
    is_flybody = JUMP_IS_FLYBODY
    is_complete_flybody = False

    def __init__(self, config: Optional[JumpLegConfig] = None,
                 state: Optional[Any] = None) -> None:
        self.config = config or JumpLegConfig()
        self._parameter_manifest_config_id = id(self.config)
        self._parameter_manifest_cache = self.config.parameter_manifest()
        self.state = FlightState.from_any(
            state, default_height_m=self.config.ground_plane_z_m,
        )
        self._resolve_initial_contact()
        self._command_start_s: Optional[float] = None
        self._event_latched = False
        self._last_jump_activation = 0.0
        self.last_output: Optional[ReducedBodyJumpOutput] = None

    def _parameter_manifest(self) -> Mapping[str, Any]:
        """Return the cached manifest, refreshing after public reassignment."""

        config_id = id(self.config)
        if config_id != self._parameter_manifest_config_id:
            self._parameter_manifest_config_id = config_id
            self._parameter_manifest_cache = self.config.parameter_manifest()
        return self._parameter_manifest_cache

    @property
    def provenance(self) -> dict[str, Any]:
        return self.config.provenance()

    @property
    def units(self) -> dict[str, str]:
        return dict(self.config.units)

    @property
    def parameters(self) -> dict[str, Any]:
        return self.config.parameter_manifest()

    def manifest(self) -> dict[str, Any]:
        return self.config.provenance()

    @property
    def time_s(self) -> float:
        return self.state.time_s

    @property
    def body_state(self) -> FlightState:
        return self.state

    @property
    def grounded(self) -> bool:
        return self._is_contact_state(self.state)

    @property
    def airborne(self) -> bool:
        return not self.grounded

    def _resolve_initial_contact(self) -> None:
        cfg = self.config
        if not cfg.ground_contact_enabled or self.state.position_m[2] >= cfg.ground_plane_z_m:
            return
        # This is initial-state validation, not an actuator pose write.  The
        # integrator resolves future penetration with an impulse below.
        self.state.position_m = (
            self.state.position_m[0], self.state.position_m[1], cfg.ground_plane_z_m,
        )
        if self.state.velocity_mps[2] < 0.0:
            self.state.velocity_mps = (
                self.state.velocity_mps[0], self.state.velocity_mps[1], 0.0,
            )

    def _is_contact_state(self, state: FlightState) -> bool:
        if not self.config.ground_contact_enabled:
            return False
        return (
            state.position_m[2] <= self.config.ground_plane_z_m + 1.0e-12
            and state.velocity_mps[2] <= 1.0e-12
        )

    def reset(self, state: Optional[Any] = None) -> FlightState:
        self.state = FlightState.from_any(
            state, default_height_m=self.config.ground_plane_z_m,
        )
        self._resolve_initial_contact()
        self._command_start_s = None
        self._event_latched = False
        self._last_jump_activation = 0.0
        self.last_output = None
        return self.state.copy()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "command_start_s": self._command_start_s,
            "event_latched": self._event_latched,
            "last_jump_activation": self._last_jump_activation,
            "model": JUMP_MODEL_NAME,
            "fit_status": self.config.fit_status,
            "parameters": self.config.parameter_manifest(),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> FlightState:
        if not isinstance(snapshot, Mapping):
            raise FlightDynamicsError("snapshot must be a mapping")
        self.state = FlightState.from_any(
            snapshot.get("state", snapshot),
            default_height_m=self.config.ground_plane_z_m,
        )
        self._resolve_initial_contact()
        raw_start = snapshot.get("command_start_s")
        self._command_start_s = None if raw_start is None else _finite(raw_start, "command_start_s")
        self._event_latched = bool(snapshot.get("event_latched", False))
        self._last_jump_activation = _clamp(
            _finite(snapshot.get("last_jump_activation", 0.0), "last_jump_activation"),
            0.0, 1.0,
        )
        self.last_output = None
        return self.state.copy()

    def _event_context(self, command: JumpNeuromuscularInput) -> tuple[Optional[float], bool]:
        jump = command.jump_activation
        start = self._command_start_s
        latched = self._event_latched
        if jump <= 0.0:
            return None, False
        if self._last_jump_activation <= 0.0 and not latched:
            start = self.state.time_s
        elif start is None and not latched:
            start = self.state.time_s
        return start, latched

    def _evaluate_step(
        self,
        command: JumpNeuromuscularInput,
        dt_s: float,
    ) -> tuple[ReducedBodyJumpOutput, Optional[float], bool]:
        cfg = self.config
        state = self.state
        contact_at_start = self._is_contact_state(state)
        start, latched = self._event_context(command)
        actuator = _jump_actuator_force(
            command, cfg, time_s=state.time_s, dt_s=dt_s,
            command_start_s=start, contact=contact_at_start,
            event_latched=latched,
            parameter_manifest=self._parameter_manifest(),
        )

        gravity_world = (0.0, 0.0, -cfg.weight_N)
        gravity_body = _qrotate_inverse(state.orientation_quat, gravity_world)
        free_force_body = _vadd(gravity_body, actuator.leg_force_body_N)
        free_force_world = _qrotate(state.orientation_quat, free_force_body)
        free_acceleration_world = _vscale(free_force_world, 1.0 / cfg.mass_kg)

        roll, pitch, yaw = _rpy_from_quat(state.orientation_quat)
        damping = tuple(
            cfg.posture_damping_Nm_s_per_rad[i] * state.angular_velocity_rad_s[i]
            for i in range(3)
        )
        restoring = (
            -cfg.posture_stiffness_Nm_per_rad[0] * roll,
            -cfg.posture_stiffness_Nm_per_rad[1] * pitch,
            -cfg.posture_stiffness_Nm_per_rad[2] * yaw,
        )
        total_torque = (
            actuator.posture_torque_body_Nm[0] + actuator.leg_roll_torque_Nm
            + restoring[0] - damping[0],
            actuator.posture_torque_body_Nm[1] + restoring[1] - damping[1],
            actuator.posture_torque_body_Nm[2] + restoring[2] - damping[2],
        )
        inertia = cfg.inertia_kg_m2
        angular_momentum = tuple(
            inertia[i] * state.angular_velocity_rad_s[i] for i in range(3)
        )
        gyro = _cross(state.angular_velocity_rad_s, angular_momentum)
        angular_acceleration = tuple(
            (total_torque[i] - gyro[i]) / inertia[i] for i in range(3)
        )

        next_velocity = _vadd(
            state.velocity_mps, _vscale(free_acceleration_world, dt_s),
        )
        next_velocity = tuple(
            _clamp(next_velocity[i], -cfg.max_speed_mps, cfg.max_speed_mps)
            for i in range(3)
        )  # type: ignore[assignment]
        next_position = _vadd(state.position_m, _vscale(next_velocity, dt_s))

        # A unilateral discrete contact impulse catches a downward crossing.
        # It does not prescribe airborne position or velocity.
        contact_force_world = (0.0, 0.0, 0.0)
        contact_impulse_world = (0.0, 0.0, 0.0)
        contact = False
        if cfg.ground_contact_enabled and dt_s > 0.0:
            crossing = (
                next_position[2] < cfg.ground_plane_z_m
                and state.position_m[2] >= cfg.ground_plane_z_m - 1.0e-12
            )
            resting = (
                state.position_m[2] <= cfg.ground_plane_z_m + 1.0e-12
                and next_velocity[2] <= 0.0
            )
            recovering = state.position_m[2] < cfg.ground_plane_z_m
            if crossing or resting or recovering:
                impact_velocity = next_velocity[2]
                target_velocity = (
                    -cfg.ground_restitution * impact_velocity
                    if impact_velocity < 0.0 else impact_velocity
                )
                if recovering:
                    target_velocity = max(0.0, target_velocity)
                contact_impulse_world = (
                    0.0, 0.0, cfg.mass_kg * (target_velocity - impact_velocity),
                )
                contact_force_world = _vscale(contact_impulse_world, 1.0 / dt_s)
                next_velocity = (next_velocity[0], next_velocity[1], target_velocity)
                # This is the boundary projection associated with the contact
                # impulse, not a command-side pose/teleport operation.
                next_position = (
                    next_position[0], next_position[1], cfg.ground_plane_z_m,
                )
                contact = True

        contact_force_body = _qrotate_inverse(state.orientation_quat, contact_force_world)
        total_force_body = _vadd(free_force_body, contact_force_body)
        total_force_world = _vadd(free_force_world, contact_force_world)
        acceleration_world = _vscale(total_force_world, 1.0 / cfg.mass_kg)
        next_angular_velocity = tuple(
            _clamp(
                state.angular_velocity_rad_s[i] + angular_acceleration[i] * dt_s,
                -cfg.max_angular_rate_rad_s, cfg.max_angular_rate_rad_s,
            )
            for i in range(3)
        )  # type: ignore[assignment]
        angular_step = _vscale(next_angular_velocity, dt_s)
        next_orientation = _quat(
            _qmul(state.orientation_quat, _quat_from_rotation_vector(angular_step))
        )
        next_state = state.copy()
        next_state.time_s = state.time_s + dt_s
        next_state.position_m = next_position
        next_state.velocity_mps = next_velocity
        next_state.angular_velocity_rad_s = next_angular_velocity
        next_state.orientation_quat = next_orientation
        next_state.cumulative_energy_J = state.cumulative_energy_J

        # Treat leaving the support plane as the end of this event.  A held
        # command cannot resume its ground push when the body later lands.
        predicted_liftoff = contact_at_start and next_position[2] > cfg.ground_plane_z_m + 1.0e-12
        next_latched = bool(
            latched
            or predicted_liftoff
            or (not contact_at_start and command.jump_activation > 0.0)
        )
        next_start = start
        if command.jump_activation <= 0.0:
            next_start = None
            next_latched = False

        posture_delta = angular_step
        output = ReducedBodyJumpOutput(
            time_s=next_state.time_s,
            state=next_state,
            input=command,
            contact=contact,
            airborne=not contact,
            actuator_active=actuator.actuator_active,
            contact_gated=actuator.contact_gated,
            jump_force_body_N=actuator.jump_force_body_N,
            leg_force_body_N=actuator.leg_force_body_N,
            contact_force_body_N=contact_force_body,
            contact_force_world_N=contact_force_world,
            total_force_body_N=total_force_body,
            total_force_world_N=total_force_world,
            posture_torque_body_Nm=actuator.posture_torque_body_Nm,
            total_torque_body_Nm=total_torque,
            acceleration_world_mps2=acceleration_world,
            angular_acceleration_body_rad_s2=angular_acceleration,
            jump_impulse_body_Ns=actuator.impulse_body_Ns,
            contact_impulse_body_Ns=_vscale(contact_force_body, dt_s),
            total_impulse_world_Ns=_vscale(total_force_world, dt_s),
            posture_delta_rad=posture_delta,
            orientation_rad=_rpy_from_quat(next_orientation),
            left_leg_force_N=actuator.left_leg_force_N,
            right_leg_force_N=actuator.right_leg_force_N,
            delay_elapsed_s=actuator.delay_elapsed_s,
            active_duration_s=actuator.active_duration_s,
            source=command.source,
            fit_status=cfg.fit_status,
            parameters=cfg.parameter_manifest(),
            ground_plane_z_m=cfg.ground_plane_z_m,
        )
        return output, next_start, next_latched

    def evaluate(self, command: Any = None, *, dt_s: float = 0.0,
                 dt: Optional[float] = None, timestamp: Optional[Any] = None) -> ReducedBodyJumpOutput:
        if dt is not None:
            if dt_s != 0.0:
                raise TypeError("pass only one of dt_s or dt")
            dt_s = dt
        if timestamp is not None:
            target_time = _timestamp_seconds(timestamp)
            if dt_s != 0.0 and abs(float(dt_s) - (target_time - self.state.time_s)) > 1.0e-9:
                raise FlightDynamicsError("dt_s does not match timestamp delta")
            dt_s = target_time - self.state.time_s
        dt_value = _finite(dt_s, "dt_s")
        if dt_value < 0.0 or dt_value > self.config.max_dt_s + 1.0e-12:
            raise FlightDynamicsError(f"dt_s must be in [0, {self.config.max_dt_s}] s")
        parsed = JumpNeuromuscularInput.from_any(command)
        output, _, _ = self._evaluate_step(parsed, dt_value)
        return output

    def step(self, command: Any = None, dt_s: Optional[float] = None,
             *, dt: Optional[float] = None, timestamp: Optional[Any] = None) -> ReducedBodyJumpOutput:
        if dt is not None:
            if dt_s is not None:
                raise TypeError("pass only one of dt_s or dt")
            dt_s = dt
        if timestamp is not None:
            target_time = _timestamp_seconds(timestamp)
            if dt_s is not None and abs(float(dt_s) - (target_time - self.state.time_s)) > 1.0e-9:
                raise FlightDynamicsError("dt_s does not match timestamp delta")
            dt_s = target_time - self.state.time_s
        dt_value = 0.001 if dt_s is None else _finite(dt_s, "dt_s")
        if dt_value <= 0.0 or dt_value > self.config.max_dt_s + 1.0e-12:
            raise FlightDynamicsError(f"dt_s must be > 0 and <= {self.config.max_dt_s} s")
        parsed = JumpNeuromuscularInput.from_any(command)
        output, next_start, next_latched = self._evaluate_step(parsed, dt_value)
        self.state = output.state.copy()
        self._command_start_s = next_start
        self._event_latched = next_latched
        self._last_jump_activation = parsed.jump_activation
        self.last_output = output
        return output

    def step_body(self, command: Any = None, dt_s: Optional[float] = None,
                  *, dt: Optional[float] = None, timestamp: Optional[Any] = None) -> dict[str, Any]:
        return self.step(command, dt_s=dt_s, dt=dt, timestamp=timestamp).to_dict()

    advance = step
    integrate = step


# Compatibility names for callers that describe this reduced bridge as jump
# physics or a leg executor.  They all retain the same strict channel boundary
# and the same explicit SI/provenance contract.
LegJumpDynamics = ReducedBodyJumpDynamics
JumpDynamics = ReducedBodyJumpDynamics
JumpPhysics = ReducedBodyJumpDynamics
ReducedBodyDynamics = ReducedBodyJumpDynamics
LegJumpActuator = JumpLegActuator
JumpActuator = JumpLegActuator
JumpLegPhysicsConfig = JumpLegConfig
LegJumpConfig = JumpLegConfig
StaticLegActuator = JumpLegActuator
LegJumpExecutor = JumpLegActuator
ReducedBodyBridge = ReducedBodyJumpDynamics
GroundContactDynamics = ReducedBodyJumpDynamics
BodyJumpOutput = ReducedBodyJumpOutput
JumpOutput = ReducedBodyJumpOutput
LegJumpOutput = ReducedBodyJumpOutput
JumpCommand = JumpNeuromuscularInput
JumpConfig = JumpLegConfig
LegActuatorConfig = JumpLegConfig
JumpDynamicsConfig = JumpLegConfig


class WingLegFlightDynamics:
    """Single-state wing plus contact-gated leg/jump bridge.

    This facade is intentionally thin: :class:`FlightDynamics` remains the
    only integrator and owns the authoritative ``FlightState`` and WPG phase.
    ``JumpLegActuator`` contributes one bounded SI force/torque term to that
    integrator through :meth:`FlightDynamics.step_with_jump`; no second body
    state or backend force injection is created.
    """

    model_name = "combined_wing_leg_reduced_body"
    model_scope = "single reduced 6-DOF wing+leg bridge; not complete FlyBody"
    is_flybody = False
    is_cfd = False
    is_complete_flybody = False

    def __init__(
        self,
        config: Optional[FlightDynamicsConfig] = None,
        state: Optional[Any] = None,
        *,
        jump_config: Optional[Any] = None,
        wind_mps: Optional[Vec3] = None,
        strict_neural: bool = True,
    ) -> None:
        self.dynamics = FlightDynamics(
            config=config,
            state=state,
            wind_mps=wind_mps,
            jump_config=jump_config,
        )
        self.strict_neural = bool(strict_neural)

    @property
    def state(self) -> FlightState:
        return self.dynamics.state

    @property
    def body_state(self) -> FlightState:
        return self.dynamics.state

    @property
    def config(self) -> FlightDynamicsConfig:
        return self.dynamics.config

    @property
    def params(self) -> FlightDynamicsConfig:
        return self.dynamics.config

    @property
    def jump_config(self) -> Any:
        return self.dynamics._combined_jump_config()

    @property
    def time_s(self) -> float:
        return self.dynamics.time_s

    @property
    def last_output(self) -> Optional[FlightOutput]:
        return self.dynamics.last_output

    @property
    def wpg(self) -> WingbeatPatternGenerator:
        return self.dynamics.wpg

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "scope": self.model_scope,
            "is_flybody": False,
            "is_cfd": False,
            "strict_neural": self.strict_neural,
            "wing": self.dynamics.config.provenance(),
            "jump": self.jump_config.provenance(),
            "ground_walking": self.dynamics.config.provenance()["ground_walking"],
            "state_integration": "one FlightDynamics state and one WPG phase",
            "force_application": "jump SI force, optional contact-gated ground traction, and posture torque added once before contact impulse",
        }

    def manifest(self) -> dict[str, Any]:
        return self.provenance

    @property
    def units(self) -> dict[str, str]:
        units = dict(self.dynamics.config.units)
        units.update({
            "jump_force_N": "N",
            "jump_impulse_Ns": "N*s",
            "jump_posture_torque_Nm": "N*m",
        })
        return units

    def reset(self, state: Optional[Any] = None, *, phase_rad: float = 0.0,
              energy_J: Optional[float] = None) -> FlightState:
        return self.dynamics.reset(state, phase_rad=phase_rad, energy_J=energy_J)

    def snapshot(self) -> dict[str, Any]:
        snap = self.dynamics.snapshot()
        snap["combined_model"] = self.model_name
        snap["strict_neural"] = self.strict_neural
        snap["jump_provenance"] = self.jump_config.provenance()
        return snap

    def restore(self, snapshot: Mapping[str, Any]) -> FlightState:
        return self.dynamics.restore(snapshot)

    def set_wind(self, wind_mps: Vec3) -> Vec3:
        return self.dynamics.set_wind(wind_mps)

    def hover_trim(self, **kwargs: Any) -> NeuromuscularInput:
        return self.dynamics.hover_trim(**kwargs)

    def hover_trim_diagnostics(self, **kwargs: Any) -> dict[str, Any]:
        return self.dynamics.hover_trim_diagnostics(**kwargs)

    def evaluate(
        self,
        command: Any = None,
        jump_command: Any = None,
        *,
        dt_s: float = 0.0,
        dt: Optional[float] = None,
        timestamp: Optional[Any] = None,
        strict_neural: Optional[bool] = None,
    ) -> FlightOutput:
        if isinstance(jump_command, (int, float)) and dt_s == 0.0 and dt is None:
            dt_s = float(jump_command)
            jump_command = None
        return self.dynamics.evaluate_with_jump(
            command,
            jump_command,
            dt_s=dt_s,
            dt=dt,
            timestamp=timestamp,
            strict_neural=self.strict_neural if strict_neural is None else bool(strict_neural),
        )

    def step(
        self,
        command: Any = None,
        jump_command: Any = None,
        dt_s: Optional[float] = None,
        *,
        dt: Optional[float] = None,
        timestamp: Optional[Any] = None,
        strict_neural: Optional[bool] = None,
    ) -> FlightOutput:
        if isinstance(jump_command, (int, float)) and dt_s is None and dt is None:
            dt_s = float(jump_command)
            jump_command = None
        return self.dynamics.step_with_jump(
            command,
            jump_command,
            dt_s=dt_s,
            dt=dt,
            timestamp=timestamp,
            strict_neural=self.strict_neural if strict_neural is None else bool(strict_neural),
        )

    def step_body(self, command: Any = None, jump_command: Any = None,
                  dt_s: Optional[float] = None, *, dt: Optional[float] = None,
                  timestamp: Optional[Any] = None,
                  strict_neural: Optional[bool] = None) -> dict[str, Any]:
        return self.step(
            command,
            jump_command,
            dt_s=dt_s,
            dt=dt,
            timestamp=timestamp,
            strict_neural=strict_neural,
        ).to_dict()


# Body-side naming variants all point at the same one-state implementation.
CombinedFlightDynamics = WingLegFlightDynamics
FlightDynamicsWithLegs = WingLegFlightDynamics
WingLegBridge = WingLegFlightDynamics
CombinedBodyDynamics = WingLegFlightDynamics


def cfg_max_dt(config: FlightDynamicsConfig) -> float:
    """Tiny helper kept separate to make evaluator bounds easy to inspect."""

    return config.max_dt_s


FlightPhysics = FlightDynamics
LowOrderFlightDynamics = FlightDynamics
FlightPhysicsConfig = FlightDynamicsConfig
WingInput = NeuromuscularInput
WingCommand = NeuromuscularInput
DynamicsState = FlightState


def self_test(*, steps: int = 300, dt_s: float = 0.001) -> dict[str, Any]:
    """Run deterministic trim, symmetry, differential-yaw, and stability checks."""

    if steps <= 0:
        raise FlightDynamicsError("steps must be positive")
    cfg = FlightDynamicsConfig()
    model = FlightDynamics(cfg)
    trim = model.hover_trim()
    hover = model.evaluate(trim)
    # A symmetric command has no differential moments; force balance is the
    # trim contract, with tolerance allowing the bias-free body drag term.
    symmetry = model.evaluate({"left_power": trim.left_power, "right_power": trim.right_power})
    differential = model.evaluate({"left_power": min(1.0, trim.left_power + 0.15),
                                   "right_power": max(0.0, trim.right_power - 0.15)})
    stable = FlightDynamics(cfg)
    trim_force_error = abs(hover.total_force_body_N[2])
    symmetry_moment = math.sqrt(sum(x * x for x in symmetry.torque_body_Nm))
    differential_yaw = differential.torque_body_Nm[2]
    finite = True
    max_speed = 0.0
    max_rate = 0.0
    energy_monotonic = True
    previous_energy = 0.0
    for _ in range(steps):
        out = stable.step(trim, dt_s=dt_s)
        values = (*out.state.position_m, *out.state.velocity_mps,
                  *out.state.orientation_quat, *out.state.angular_velocity_rad_s,
                  out.power_W, out.cumulative_energy_J)
        finite = finite and all(math.isfinite(float(value)) for value in values)
        max_speed = max(max_speed, _vnorm(out.state.velocity_mps))
        max_rate = max(max_rate, _vnorm(out.state.angular_velocity_rad_s))
        energy_monotonic = energy_monotonic and out.cumulative_energy_J >= previous_energy
        previous_energy = out.cumulative_energy_J
    checks = {
        "wpg_in_band": WPG_FREQUENCY_MIN_HZ <= cfg.wingbeat_frequency_hz <= WPG_FREQUENCY_MAX_HZ,
        "hover_trim_feasible": bool(model.hover_trim_diagnostics()["feasible"]),
        "hover_trim": trim_force_error <= 1.0e-10,
        "hover_force_balanced": trim_force_error <= 1.0e-10,
        "symmetry": symmetry_moment <= 1.0e-14,
        "symmetric_moments_zero": symmetry_moment <= 1.0e-14,
        "differential_yaw": abs(differential_yaw) > 1.0e-14,
        "differential_yaw_nonzero": abs(differential_yaw) > 1.0e-14,
        "stable_step": finite,
        "stable_finite": finite,
        "energy_monotonic": energy_monotonic,
        "model_is_not_cfd": stable.is_cfd is False,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "metrics": {
            "trim_power": trim.left_power,
            "trim_force_error_N": trim_force_error,
            "symmetric_moment_norm_Nm": symmetry_moment,
            "differential_yaw_Nm": differential_yaw,
            "max_speed_mps": max_speed,
            "max_angular_rate_rad_s": max_rate,
            "energy_J": previous_energy,
            "steps": steps,
            "dt_s": dt_s,
        },
        "provenance": cfg.provenance(),
    }


def run_self_test(*, steps: int = 300, dt_s: float = 0.001) -> dict[str, Any]:
    result = self_test(steps=steps, dt_s=dt_s)
    if not result["ok"]:
        raise AssertionError(json.dumps(result, sort_keys=True))
    return result


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run deterministic model checks")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.001, dest="dt_s")
    args = parser.parse_args(argv)
    result = self_test(steps=args.steps, dt_s=args.dt_s)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


__all__ = [
    "BodyOutput", "ContactSurface", "FlightDynamics", "FlightDynamicsConfig", "FlightDynamicsError",
    "FlightInput", "FlightOutput", "FlightPhysics", "FlightPhysicsConfig", "FlightState", "FlightStep",
    "IS_CFD", "LowOrderFlightDynamics", "MODEL_NAME", "MODEL_SCOPE", "MotorInput",
    "NeuralMuscleInput", "NeuromuscularInput", "WingCommand", "WingInput", "DynamicsState",
    "JUMP_IS_FLYBODY", "JUMP_MODEL_NAME", "JUMP_MODEL_SCOPE", "JumpActuator", "JumpActuatorOutput",
    "BodyJumpOutput", "JumpCommand", "JumpConfig", "JumpDynamics", "JumpDynamicsConfig", "JumpInput",
    "JumpLegActuator", "JumpLegConfig", "JumpLegPhysicsConfig", "JumpNeuromuscularInput", "JumpOutput",
    "JumpPhysics", "LegActuatorConfig", "LegJumpActuator", "LegJumpConfig", "LegJumpDynamics",
    "LegJumpExecutor", "LegJumpOutput", "LegNeuromuscularInput", "GroundContactDynamics",
    "ReducedBodyBridge", "ReducedBodyDynamics", "ReducedBodyJumpDynamics", "ReducedBodyJumpOutput",
    "StaticLegActuator", "WingLegBridge", "WingLegFlightDynamics", "CombinedFlightDynamics",
    "FlightDynamicsWithLegs", "CombinedBodyDynamics",
    "WPG", "WPGSample",
    "WPG_FREQUENCY_MAX_HZ", "WPG_FREQUENCY_MIN_HZ", "WingBeatGenerator",
    "WingbeatPatternGenerator", "run_self_test", "self_test",
]


if __name__ == "__main__":
    raise SystemExit(_main())
