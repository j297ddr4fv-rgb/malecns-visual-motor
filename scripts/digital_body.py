"""Small, deterministic body engine for the MaleCNS digital fly.

The module deliberately has no mandatory third-party dependencies.  The default
backend is a reduced six degree-of-freedom rigid body with two actuated wings.
It is useful for wiring and closed-loop experiments; it is not a substitute for
a calibrated CFD or multibody model.

Conventions
-----------
* SI units are used throughout: metres, seconds, kilograms, newtons, and
  radians.  Wing drive commands are dimensionless in ``[0, 1]`` and wingbeat
  frequency is in hertz.
* ``position_m`` is the centre-of-mass position in a world frame whose z=0 is
  the ground plane.  ``orientation_quat`` is scalar-first ``(w, x, y, z)`` and
  maps body-frame vectors to world-frame vectors.
* Angular velocity is body-frame radians/second.  The simple backend uses a
  fixed semi-implicit Euler step, so identical inputs produce identical state.

The public ``DigitalBody`` facade intentionally accepts both ``BodyCommand``
and mapping/attribute-like objects.  This keeps it loosely coupled to a future
``twin_contracts`` package without requiring that package today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import importlib.util
import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


class BodyBackendError(RuntimeError):
    """Base class for backend-selection errors."""


class OptionalBackendUnavailable(BodyBackendError):
    """Raised when a requested optional simulator is not installed."""


class OptionalBackendNotValidated(BodyBackendError):
    """Raised instead of silently pretending an optional adapter works."""


def _vec3(value: Iterable[float], *, name: str = "vector") -> Vec3:
    try:
        vals = tuple(float(x) for x in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain three finite SI values") from exc
    if len(vals) != 3 or not all(math.isfinite(x) for x in vals):
        raise ValueError(f"{name} must contain three finite SI values")
    return vals  # type: ignore[return-value]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("command values must be finite scalars") from exc
    if not math.isfinite(value):
        raise ValueError("command values must be finite scalars")
    return min(high, max(low, value))


def _scalar(value: Any, *, name: str) -> float:
    """Coerce one scalar while keeping NaN/inf out of the physics state."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vscale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _vdot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vnorm(a: Vec3) -> float:
    return math.sqrt(_vdot(a, a))


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


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


def _qnormalize(q: Quat) -> Quat:
    n = math.sqrt(sum(x * x for x in q))
    if n <= 1.0e-15 or not math.isfinite(n):
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(x / n for x in q)  # type: ignore[return-value]


def _qrotate(q: Quat, v: Vec3) -> Vec3:
    p = _qmul(_qmul(q, (0.0, v[0], v[1], v[2])), _qconj(q))
    return (p[1], p[2], p[3])


def _quat_from_rpy(rpy: Iterable[float]) -> Quat:
    """Convert shared-contract roll/pitch/yaw radians to scalar-first q."""

    roll, pitch, yaw = _vec3(rpy, name="orientation_rad")
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return _qnormalize((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def _rpy_from_quat(q: Quat) -> Vec3:
    """Convert scalar-first q to the contract's XYZ roll/pitch/yaw order."""

    w, x, y, z = _qnormalize(q)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return (roll, pitch, yaw)


def _timestamp_seconds(value: Any) -> float:
    """Read a shared ``Timestamp`` without importing its contract module."""

    if isinstance(value, Mapping):
        raw = value.get("t_s", value.get("time_s"))
    else:
        raw = getattr(value, "t_s", getattr(value, "time_s", None))
    if raw is None:
        raise ValueError("timestamp must expose t_s/time_s")
    return _scalar(raw, name="timestamp.t_s")


def _timestamp_period(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        raw = value.get("dt_s", value.get("period_s"))
    else:
        raw = getattr(value, "dt_s", getattr(value, "period_s", None))
    if raw is None:
        return None
    period = _scalar(raw, name="timestamp.dt_s")
    return period if period > 0 else None


def _record_dict(record: Any) -> dict[str, Any]:
    return {
        name: tuple(value) if isinstance(value, (list, tuple)) else value
        for name, value in vars(record).items()
        if name in record.__dataclass_fields__
    }


@dataclass(frozen=True)
class BodyConfig:
    """Physical and numerical parameters for the reduced fly body.

    Defaults are intentionally order-of-magnitude fruit-fly values.  They are
    a stable control scaffold, not fitted aerodynamic coefficients.
    """

    mass_kg: float = 1.0e-6
    gravity_mps2: float = 9.80665
    air_density_kgpm3: float = 1.225
    body_height_m: float = 1.2e-3
    body_area_m2: float = 2.0e-6
    drag_coefficient: float = 1.0
    wing_area_m2: float = 1.2e-5
    wing_span_m: float = 2.5e-3
    max_lift_per_wing_N: Optional[float] = None
    wingbeat_frequency_hz: float = 190.0
    wing_drive_tau_s: float = 0.004
    linear_damping_Nspm: float = 1.0e-7
    angular_damping_Nmsp: float = 2.0e-12
    inertia_kgm2: Vec3 = (2.0e-12, 2.0e-12, 3.0e-12)
    contact_stiffness_Npm: float = 2.0e-3
    contact_damping_Nspm: float = 2.0e-6
    restitution: float = 0.0
    max_dt_s: float = 0.02
    angular_rate_tau_s: float = 0.01

    def __post_init__(self) -> None:
        scalars = {
            "mass_kg": self.mass_kg,
            "gravity_mps2": self.gravity_mps2,
            "air_density_kgpm3": self.air_density_kgpm3,
            "body_height_m": self.body_height_m,
            "body_area_m2": self.body_area_m2,
            "drag_coefficient": self.drag_coefficient,
            "wing_area_m2": self.wing_area_m2,
            "wing_span_m": self.wing_span_m,
            "wingbeat_frequency_hz": self.wingbeat_frequency_hz,
            "wing_drive_tau_s": self.wing_drive_tau_s,
            "linear_damping_Nspm": self.linear_damping_Nspm,
            "angular_damping_Nmsp": self.angular_damping_Nmsp,
            "angular_rate_tau_s": self.angular_rate_tau_s,
            "contact_stiffness_Npm": self.contact_stiffness_Npm,
            "contact_damping_Nspm": self.contact_damping_Nspm,
            "restitution": self.restitution,
            "max_dt_s": self.max_dt_s,
        }
        for name, value in scalars.items():
            scalars[name] = _scalar(value, name=name)
            object.__setattr__(self, name, scalars[name])
        if self.max_lift_per_wing_N is not None:
            object.__setattr__(
                self, "max_lift_per_wing_N", _scalar(self.max_lift_per_wing_N, name="max_lift_per_wing_N")
            )
        if self.mass_kg <= 0 or self.gravity_mps2 <= 0:
            raise ValueError("mass_kg and gravity_mps2 must be positive SI values")
        if self.air_density_kgpm3 < 0 or self.body_height_m < 0:
            raise ValueError("air density and body height must be non-negative SI values")
        if self.body_area_m2 < 0 or self.wing_area_m2 <= 0 or self.wing_span_m <= 0:
            raise ValueError("body and wing geometry must be non-negative/positive SI values")
        if self.drag_coefficient < 0 or self.linear_damping_Nspm < 0:
            raise ValueError("drag and linear damping coefficients must be non-negative")
        if self.wingbeat_frequency_hz <= 0 or self.wing_drive_tau_s <= 0:
            raise ValueError("wingbeat_frequency_hz and wing_drive_tau_s must be positive")
        try:
            inertia = tuple(_scalar(i, name="inertia_kgm2") for i in self.inertia_kgm2)
        except (TypeError, ValueError) as exc:
            raise ValueError("inertia_kgm2 must be positive") from exc
        if len(inertia) != 3 or any(i <= 0 for i in inertia):
            raise ValueError("inertia_kgm2 must be positive")
        object.__setattr__(self, "inertia_kgm2", inertia)
        if self.max_lift_per_wing_N is not None and self.max_lift_per_wing_N < 0:
            raise ValueError("max_lift_per_wing_N must be non-negative")
        if self.angular_damping_Nmsp < 0 or self.angular_rate_tau_s <= 0:
            raise ValueError("angular damping must be non-negative and angular_rate_tau_s must be positive")
        if self.max_dt_s <= 0 or self.contact_stiffness_Npm < 0 or self.contact_damping_Nspm < 0:
            raise ValueError("invalid integration/contact parameter")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be between 0 and 1")

    @property
    def weight_N(self) -> float:
        return self.mass_kg * self.gravity_mps2

    @property
    def lift_capacity_N(self) -> float:
        # Explicit capacity is useful for calibrations.  The default is chosen
        # so symmetric full drive can climb, while hover remains controllable.
        if self.max_lift_per_wing_N is not None:
            return float(self.max_lift_per_wing_N)
        return 0.92 * self.weight_N


@dataclass(frozen=True)
class BodyCommand:
    """Wing and body controls consumed by :meth:`DigitalBody.step`.

    ``wing_drive_*`` are normalized motor commands.  Optional force/torque
    fields are in SI and are intended for calibrated controllers or tests.
    """

    wing_drive_left: float = 0.0
    wing_drive_right: float = 0.0
    wingbeat_frequency_hz: Optional[float] = None
    pitch_torque_Nm: float = 0.0
    roll_torque_Nm: float = 0.0
    yaw_torque_Nm: float = 0.0
    force_body_N: Vec3 = (0.0, 0.0, 0.0)
    thrust_accel_mps2: float = 0.0
    target_angular_velocity_radps: Optional[Vec3] = None

    def normalized(self) -> "BodyCommand":
        freq = self.wingbeat_frequency_hz
        if freq is not None:
            freq = _scalar(freq, name="wingbeat_frequency_hz")
            if freq <= 0:
                raise ValueError("wingbeat_frequency_hz must be positive hertz")
        return BodyCommand(
            wing_drive_left=_clamp(self.wing_drive_left),
            wing_drive_right=_clamp(self.wing_drive_right),
            wingbeat_frequency_hz=freq,
            pitch_torque_Nm=_scalar(self.pitch_torque_Nm, name="pitch_torque_Nm"),
            roll_torque_Nm=_scalar(self.roll_torque_Nm, name="roll_torque_Nm"),
            yaw_torque_Nm=_scalar(self.yaw_torque_Nm, name="yaw_torque_Nm"),
            force_body_N=_vec3(self.force_body_N, name="force_body_N"),
            thrust_accel_mps2=_scalar(self.thrust_accel_mps2, name="thrust_accel_mps2"),
            target_angular_velocity_radps=(
                None if self.target_angular_velocity_radps is None
                else _vec3(self.target_angular_velocity_radps, name="target_angular_velocity_radps")
            ),
        )

    @classmethod
    def from_any(cls, value: Any) -> "BodyCommand":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value.normalized()
        if isinstance(value, Mapping):
            def get(*names: str, default: Any = 0.0) -> Any:
                for name in names:
                    if name in value:
                        return value[name]
                return default

            pair = value.get("wing_drive", value.get("wing_drives"))
            left = get("wing_drive_left", "left_wing", "left", default=0.0)
            right = get("wing_drive_right", "right_wing", "right", default=0.0)
            if pair is not None:
                try:
                    left, right = pair
                except (TypeError, ValueError) as exc:
                    raise ValueError("wing_drive must contain left and right values") from exc
            force = get("force_body_N", "body_force_N", default=(0.0, 0.0, 0.0))
            target_rate = get(
                "target_angular_velocity_radps", "angular_velocity_rad_s", "angular_rate_rad_s",
                default=None,
            )
            if target_rate is None and any(
                name in value for name in ("roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s")
            ):
                target_rate = (
                    get("roll_rate_rad_s", default=0.0),
                    get("pitch_rate_rad_s", default=0.0),
                    get("yaw_rate_rad_s", default=0.0),
                )
            return cls(
                left,
                right,
                get("wingbeat_frequency_hz", "frequency_hz", default=None),
                get("pitch_torque_Nm", "pitch_torque", default=0.0),
                get("roll_torque_Nm", "roll_torque", default=0.0),
                get("yaw_torque_Nm", "yaw_torque", default=0.0),
                force,
                get("thrust_accel_mps2", "thrust_mps2", default=0.0),
                target_rate,
            ).normalized()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) < 2:
                raise ValueError("a sequence command needs left and right wing drive")
            return cls(value[0], value[1]).normalized()
        # Attribute-like future contract object.  ``getattr`` is deliberately
        # limited to known names to keep the coupling one-way and predictable.
        target_rate = getattr(value, "target_angular_velocity_radps", getattr(value, "angular_velocity_rad_s", None))
        if target_rate is None and any(
            hasattr(value, name) for name in ("roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s")
        ):
            target_rate = (
                getattr(value, "roll_rate_rad_s", 0.0),
                getattr(value, "pitch_rate_rad_s", 0.0),
                getattr(value, "yaw_rate_rad_s", 0.0),
            )
        return cls(
            getattr(value, "wing_drive_left", getattr(value, "left_wing", getattr(value, "left", 0.0))),
            getattr(value, "wing_drive_right", getattr(value, "right_wing", getattr(value, "right", 0.0))),
            getattr(value, "wingbeat_frequency_hz", None),
            getattr(value, "pitch_torque_Nm", 0.0),
            getattr(value, "roll_torque_Nm", 0.0),
            getattr(value, "yaw_torque_Nm", 0.0),
            getattr(value, "force_body_N", (0.0, 0.0, 0.0)),
            getattr(value, "thrust_accel_mps2", getattr(value, "thrust_mps2", 0.0)),
            target_rate,
        ).normalized()


@dataclass
class BodyState:
    """Mutable simulation state; values remain plain Python scalars/tuples."""

    time_s: float = 0.0
    position_m: Vec3 = (0.0, 0.0, 1.2e-3)
    velocity_mps: Vec3 = (0.0, 0.0, 0.0)
    orientation_quat: Quat = (1.0, 0.0, 0.0, 0.0)
    angular_velocity_radps: Vec3 = (0.0, 0.0, 0.0)
    wing_drive_left: float = 0.0
    wing_drive_right: float = 0.0
    wing_phase_left_rad: float = 0.0
    wing_phase_right_rad: float = 0.0
    contact: bool = True
    ground_distance_m: float = 0.0
    flight_mode: str = "grounded"
    acceleration_mps2: Vec3 = (0.0, 0.0, 0.0)
    angular_acceleration_radps2: Vec3 = (0.0, 0.0, 0.0)
    aerodynamic_force_N: Vec3 = (0.0, 0.0, 0.0)
    gravity_force_N: Vec3 = (0.0, 0.0, 0.0)
    contact_force_N: Vec3 = (0.0, 0.0, 0.0)
    total_force_N: Vec3 = (0.0, 0.0, 0.0)
    total_torque_Nm: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.time_s = _scalar(self.time_s, name="time_s")
        self.position_m = _vec3(self.position_m, name="position_m")
        self.velocity_mps = _vec3(self.velocity_mps, name="velocity_mps")
        try:
            quat = tuple(float(x) for x in self.orientation_quat)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("orientation_quat must contain four finite values") from exc
        if len(quat) != 4 or not all(math.isfinite(x) for x in quat):
            raise ValueError("orientation_quat must contain four finite values")
        qnorm = math.sqrt(sum(x * x for x in quat))
        if qnorm <= 1.0e-15:
            raise ValueError("orientation_quat must have non-zero norm")
        self.orientation_quat = _qnormalize(quat)  # type: ignore[assignment]
        self.angular_velocity_radps = _vec3(self.angular_velocity_radps, name="angular_velocity_radps")
        self.wing_drive_left = _clamp(self.wing_drive_left)
        self.wing_drive_right = _clamp(self.wing_drive_right)
        self.wing_phase_left_rad = _scalar(self.wing_phase_left_rad, name="wing_phase_left_rad") % (2.0 * math.pi)
        self.wing_phase_right_rad = _scalar(self.wing_phase_right_rad, name="wing_phase_right_rad") % (2.0 * math.pi)
        self.contact = bool(self.contact)
        self.ground_distance_m = max(0.0, _scalar(self.ground_distance_m, name="ground_distance_m"))
        self.flight_mode = str(self.flight_mode)
        self.acceleration_mps2 = _vec3(self.acceleration_mps2, name="acceleration_mps2")
        self.angular_acceleration_radps2 = _vec3(
            self.angular_acceleration_radps2, name="angular_acceleration_radps2"
        )
        self.aerodynamic_force_N = _vec3(self.aerodynamic_force_N, name="aerodynamic_force_N")
        self.gravity_force_N = _vec3(self.gravity_force_N, name="gravity_force_N")
        self.contact_force_N = _vec3(self.contact_force_N, name="contact_force_N")
        self.total_force_N = _vec3(self.total_force_N, name="total_force_N")
        self.total_torque_Nm = _vec3(self.total_torque_Nm, name="total_torque_Nm")

    @property
    def position(self) -> Vec3:
        """Compatibility alias; canonical field is :attr:`position_m`."""

        return self.position_m

    @property
    def velocity(self) -> Vec3:
        return self.velocity_mps

    @property
    def attitude_quat(self) -> Quat:
        return self.orientation_quat

    @property
    def angular_velocity(self) -> Vec3:
        return self.angular_velocity_radps

    @property
    def angular_velocity_rad_s(self) -> Vec3:
        """Shared-contract spelling for angular velocity."""

        return self.angular_velocity_radps

    @property
    def orientation_rad(self) -> Vec3:
        """Shared-contract roll/pitch/yaw view of the quaternion attitude."""

        return _rpy_from_quat(self.orientation_quat)

    @property
    def orientation_rpy_rad(self) -> Vec3:
        return self.orientation_rad

    @property
    def attitude_rpy_rad(self) -> Vec3:
        return self.orientation_rad

    @property
    def left_wing_drive(self) -> float:
        return self.wing_drive_left

    @property
    def right_wing_drive(self) -> float:
        return self.wing_drive_right

    @property
    def left_wing(self) -> float:
        """Short alias for the normalized left-wing drive."""

        return self.wing_drive_left

    @property
    def right_wing(self) -> float:
        """Short alias for the normalized right-wing drive."""

        return self.wing_drive_right

    @property
    def wing_drive(self) -> tuple[float, float]:
        return (self.wing_drive_left, self.wing_drive_right)

    @property
    def grounded(self) -> bool:
        return self.contact

    def copy(self) -> "BodyState":
        return BodyState(**self.to_dict(include_units=False))

    def to_dict(self, *, include_units: bool = True) -> dict[str, Any]:
        data = _record_dict(self)
        if include_units:
            data["units"] = {
                "position_m": "m", "velocity_mps": "m/s", "orientation_quat": "unitless",
                "angular_velocity_radps": "rad/s", "wing_drive": "normalized",
                "wing_phase_rad": "rad", "ground_distance_m": "m", "acceleration_mps2": "m/s^2",
                "angular_acceleration_radps2": "rad/s^2", "force": "N", "torque": "N*m",
                "time_s": "s",
            }
        return data

    # ``as_dict`` is a common spelling in the rest of this project.  Keep it
    # as a small alias so consumers do not need to know which twin record
    # implementation produced the state.
    def as_dict(self, *, include_units: bool = True) -> dict[str, Any]:
        return self.to_dict(include_units=include_units)

    @classmethod
    def from_contract(cls, value: Any, *, default_time_s: float = 0.0) -> "BodyState":
        """Adapt the shared ``twin_contracts.BodyState`` by duck typing.

        Importing :mod:`twin_contracts` is deliberately unnecessary: the
        adapter only reads its documented timestamp/pose/rate attributes.  It
        therefore also accepts a JSON-like mapping or another small body
        record with the same fields.
        """

        if isinstance(value, cls):
            return value.copy()

        def get(*names: str, default: Any = None) -> Any:
            if isinstance(value, Mapping):
                for name in names:
                    if name in value:
                        return value[name]
                return default
            for name in names:
                if hasattr(value, name):
                    return getattr(value, name)
            return default

        timestamp = get("timestamp", default=None)
        if timestamp is None:
            time_s = get("time_s", "t_s", "timestamp_s", default=default_time_s)
        elif isinstance(timestamp, Mapping):
            time_s = timestamp.get("t_s", timestamp.get("time_s", default_time_s))
        else:
            time_s = getattr(timestamp, "t_s", getattr(timestamp, "time_s", default_time_s))
        position = get("position_m", "position", default=(0.0, 0.0, 1.2e-3))
        velocity = get("velocity_mps", "velocity", default=(0.0, 0.0, 0.0))
        angular = get(
            "angular_velocity_radps", "angular_velocity_rad_s", "angular_velocity", default=(0.0, 0.0, 0.0)
        )
        quat = get("orientation_quat", "attitude_quat", default=None)
        orientation = get(
            "orientation_rad", "orientation_rpy_rad", "attitude_rpy_rad", "rpy_rad", default=None
        )
        if quat is None:
            quat = _quat_from_rpy((0.0, 0.0, 0.0) if orientation is None else orientation)
        return cls(
            time_s=_scalar(time_s, name="timestamp.t_s"),
            position_m=_vec3(position, name="position_m"),
            velocity_mps=_vec3(velocity, name="velocity_mps"),
            orientation_quat=tuple(float(x) for x in quat),  # type: ignore[arg-type]
            angular_velocity_radps=_vec3(angular, name="angular_velocity_rad_s"),
            wing_drive_left=get("wing_drive_left", "left_wing", "left_wing_drive", default=0.0),
            wing_drive_right=get("wing_drive_right", "right_wing", "right_wing_drive", default=0.0),
            contact=get("contact", "grounded", default=True),
        )

    from_any = from_contract

    def to_contract(self, timestamp: Any = None, *, source: Any = "handbuilt") -> Any:
        """Return a ``twin_contracts.BodyState`` without making it mandatory."""

        try:
            from twin_contracts import BodyState as ContractBodyState, Timestamp
        except ImportError as exc:  # pragma: no cover - exercised without repo path
            raise RuntimeError("twin_contracts is unavailable; use to_dict() instead") from exc
        if timestamp is None:
            timestamp = Timestamp(self.time_s, max(0, int(round(self.time_s))), "simulation", 1.0)
        elif isinstance(timestamp, Mapping):
            timestamp = Timestamp.from_dict(timestamp)
        if not isinstance(timestamp, Timestamp):
            raise TypeError("timestamp must be a twin_contracts.Timestamp or mapping")
        return ContractBodyState(
            timestamp=timestamp,
            position_m=self.position_m,
            orientation_rad=self.orientation_rad,
            velocity_mps=self.velocity_mps,
            angular_velocity_rad_s=self.angular_velocity_rad_s,
            source=source,
        )

    to_contract_body_state = to_contract

    # Mapping-like access is convenient for CNS code while preserving typed
    # attributes for callers that prefer them.
    def __getitem__(self, key: str) -> Any:
        aliases = {
            "position": "position_m", "velocity": "velocity_mps", "attitude_quat": "orientation_quat",
            "angular_velocity": "angular_velocity_radps", "left_wing_drive": "wing_drive_left",
            "right_wing_drive": "wing_drive_right", "left_wing": "wing_drive_left",
            "right_wing": "wing_drive_right", "grounded": "contact",
        }
        if key in ("wing_drive", "wing_drives"):
            return (self.wing_drive_left, self.wing_drive_right)
        return self.to_dict(include_units=False)[aliases.get(key, key)]


@dataclass(frozen=True)
class BackendCapability:
    name: str
    installed: bool
    validated: bool
    module: Optional[str] = None
    version: Optional[str] = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "installed": self.installed,
            "validated": self.validated,
            "module": self.module,
            "version": self.version,
            "note": self.note,
        }


def detect_optional_backends() -> dict[str, BackendCapability]:
    """Report optional backend presence without importing simulator modules."""

    result: dict[str, BackendCapability] = {}
    for name, module in (("mujoco", "mujoco"), ("flybody", "flybody")):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ModuleNotFoundError, ValueError):
            # A partially installed namespace package can make ``find_spec``
            # raise instead of returning ``None``.  Treat that as unavailable;
            # importantly, do not import a simulator just to inspect it.
            spec = None
        installed = spec is not None
        version: Optional[str] = None
        if installed:
            try:
                version = importlib.metadata.version(module)
            except importlib.metadata.PackageNotFoundError:
                # Some source checkouts are importable without distribution
                # metadata.  Presence is still useful, version is unknown.
                version = None
        result[name] = BackendCapability(
            name=name,
            installed=installed,
            validated=False,
            module=module if installed else None,
            version=version,
            note=("installed; adapter entry exists but is intentionally not validated in this module"
                   if installed else "not installed; use the dependency-free simple backend"),
        )
    return result


def twin_contracts_status() -> dict[str, Any]:
    """Non-invasive capability hint for an optional future ``twin_contracts``."""

    spec = importlib.util.find_spec("twin_contracts")
    return {
        "available": spec is not None,
        "module": "twin_contracts" if spec is not None else None,
        "used": False,
        "note": "DigitalBody accepts mappings/attribute objects and does not require this package",
    }


class SimpleWingedBodyBackend:
    """Deterministic reduced 6-DOF body with two sinusoidal wing drives."""

    backend_name = "simple"
    name = "simple"
    validated = True

    def __init__(self, config: Optional[BodyConfig] = None) -> None:
        self.config = config or BodyConfig()
        self._state = BodyState(position_m=(0.0, 0.0, self.config.body_height_m))
        self._previous_contact = True
        self.reset()

    @property
    def state(self) -> BodyState:
        return self._state

    def _state_from_mapping(self, state: Mapping[str, Any]) -> BodyState:
        """Build a typed state from a snapshot or a small caller mapping."""

        base = BodyState(position_m=(0.0, 0.0, self.config.body_height_m)).to_dict(include_units=False)
        aliases = {
            "position": "position_m",
            "velocity": "velocity_mps",
            "orientation": "orientation_quat",
            "attitude_quat": "orientation_quat",
            "angular_velocity": "angular_velocity_radps",
            "left_wing": "wing_drive_left",
            "right_wing": "wing_drive_right",
            "left_wing_drive": "wing_drive_left",
            "right_wing_drive": "wing_drive_right",
            "grounded": "contact",
        }
        for key, value in state.items():
            canonical = aliases.get(key, key)
            if canonical in base:
                base[canonical] = value
        # A pair-valued wing command is convenient in hand-written reset
        # states, and is deliberately accepted only when canonical individual
        # drives were not supplied.
        if "wing_drive" in state or "wing_drives" in state:
            pair = state.get("wing_drive", state.get("wing_drives"))
            try:
                left, right = pair
            except (TypeError, ValueError) as exc:
                raise ValueError("wing_drive must contain left and right values") from exc
            if "wing_drive_left" not in state and "left_wing" not in state:
                base["wing_drive_left"] = left
            if "wing_drive_right" not in state and "right_wing" not in state:
                base["wing_drive_right"] = right
        return BodyState(**base)

    def reset(self, state: Optional[Any] = None, *, seed: Optional[int] = None) -> BodyState:
        del seed  # deterministic backend; accepted for Gym-like callers
        if state is None:
            self._state = BodyState(position_m=(0.0, 0.0, self.config.body_height_m))
        elif isinstance(state, BodyState):
            self._state = state.copy()
        elif isinstance(state, Mapping):
            self._state = self._state_from_mapping(state["body"] if "body" in state else state)
        elif hasattr(state, "body") and hasattr(state, "timestamp"):
            # Shared ``twin_contracts.WorldState``: reset from its nested body
            # record while retaining this backend's richer wing/contact state.
            self._state = BodyState.from_contract(getattr(state, "body"), default_time_s=0.0)
        else:
            # Shared ``twin_contracts.BodyState`` and compatible records are
            # accepted without importing that module at module load time.
            try:
                self._state = BodyState.from_contract(state, default_time_s=0.0)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TypeError("reset state must be None, BodyState, a mapping, or a contract body record") from exc
        floor = self.config.body_height_m
        z = self._state.position_m[2]
        if z < floor:
            # A reset is a physical initial condition, so do not leave the
            # body interpenetrating the ground.  Preserve upward velocity but
            # discard a downward component that would immediately re-penetrate.
            self._state.position_m = (self._state.position_m[0], self._state.position_m[1], floor)
            if self._state.velocity_mps[2] < 0:
                self._state.velocity_mps = (
                    self._state.velocity_mps[0], self._state.velocity_mps[1], 0.0
                )
            z = floor
        self._state.ground_distance_m = max(0.0, z - floor)
        self._state.contact = self._state.ground_distance_m <= 1.0e-12 and self._state.velocity_mps[2] <= 0.0
        self._state.flight_mode = "grounded" if self._state.contact else "flight"
        self._previous_contact = self._state.contact
        return self._state.copy()

    def step(self, command: Any = None, dt_s: Optional[float] = None, *, dt: Optional[float] = None) -> BodyState:
        cmd = BodyCommand.from_any(command)
        if dt is not None:
            if dt_s is not None:
                raise TypeError("pass only one of dt_s or dt")
            dt_s = dt
        dt = 0.001 if dt_s is None else float(dt_s)
        if not math.isfinite(dt) or dt <= 0 or dt > self.config.max_dt_s:
            raise ValueError(f"dt_s must be finite, > 0, and <= {self.config.max_dt_s} s")
        old = self._state
        cfg = self.config
        freq = cfg.wingbeat_frequency_hz if cmd.wingbeat_frequency_hz is None else cmd.wingbeat_frequency_hz
        phase_l = (old.wing_phase_left_rad + 2.0 * math.pi * freq * dt) % (2.0 * math.pi)
        phase_r = (old.wing_phase_right_rad + 2.0 * math.pi * freq * dt) % (2.0 * math.pi)
        alpha = min(1.0, dt / cfg.wing_drive_tau_s)
        dl = old.wing_drive_left + alpha * (cmd.wing_drive_left - old.wing_drive_left)
        dr = old.wing_drive_right + alpha * (cmd.wing_drive_right - old.wing_drive_right)

        # Sinusoidal wing stroke power; the small nonzero floor captures the
        # mean lift over a wingbeat while retaining deterministic beat timing.
        stroke_l = 0.72 + 0.28 * math.sin(phase_l) ** 2
        stroke_r = 0.72 + 0.28 * math.sin(phase_r) ** 2
        lift_l = cfg.lift_capacity_N * dl * dl * stroke_l
        lift_r = cfg.lift_capacity_N * dr * dr * stroke_r
        wing_force_body = (
            0.0,
            0.15 * (lift_l - lift_r),
            lift_l + lift_r,
        )
        v = old.velocity_mps
        speed = _vnorm(v)
        # Quadratic body drag plus the optional viscous term.  Writing this as
        # a scalar multiplying ``v`` avoids a division by zero at rest while
        # keeping the force exactly opposite the world-frame velocity.
        drag_factor = (
            0.5 * cfg.air_density_kgpm3 * cfg.drag_coefficient * cfg.body_area_m2 * speed
            + cfg.linear_damping_Nspm
        )
        drag_world = _vscale(v, -drag_factor)
        gravity_world = (0.0, 0.0, -cfg.weight_N)
        thrust_force_body = (0.0, 0.0, cfg.mass_kg * cmd.thrust_accel_mps2)
        aero_world = _vadd(
            _qrotate(old.orientation_quat, _vadd(_vadd(wing_force_body, cmd.force_body_N), thrust_force_body)),
            drag_world,
        )
        total_pre_contact = _vadd(aero_world, gravity_world)
        acceleration = _vscale(total_pre_contact, 1.0 / cfg.mass_kg)
        velocity = _vadd(v, _vscale(acceleration, dt))
        position = _vadd(old.position_m, _vscale(velocity, dt))

        contact_force = (0.0, 0.0, 0.0)
        if position[2] <= cfg.body_height_m:
            penetration = cfg.body_height_m - position[2]
            # A penalty estimate is retained for diagnostics; the hard clamp
            # makes the reduced model stable for takeoff/landing controllers.
            normal = max(0.0, cfg.contact_stiffness_Npm * penetration - cfg.contact_damping_Nspm * velocity[2])
            contact_force = (0.0, 0.0, normal)
            position = (position[0], position[1], cfg.body_height_m)
            if velocity[2] < 0:
                velocity = (velocity[0], velocity[1], -cfg.restitution * velocity[2])
            contact = True
        else:
            contact = False
        total_force = _vadd(total_pre_contact, contact_force)
        acceleration = _vscale(total_force, 1.0 / cfg.mass_kg)

        # Differential lift rolls the body; force application on the two wing
        # arms creates a yaw component.  Optional commanded torques are SI.
        rate_torque = (0.0, 0.0, 0.0)
        if cmd.target_angular_velocity_radps is not None:
            rate_torque = tuple(
                cfg.inertia_kgm2[i]
                * (cmd.target_angular_velocity_radps[i] - old.angular_velocity_radps[i])
                / cfg.angular_rate_tau_s
                for i in range(3)
            )
        torque = (
            cmd.roll_torque_Nm + rate_torque[0] + cfg.wing_span_m * 0.5 * (lift_l - lift_r),
            cmd.pitch_torque_Nm + rate_torque[1],
            cmd.yaw_torque_Nm + rate_torque[2] + 0.10 * cfg.wing_span_m * (lift_l - lift_r),
        )
        angular_accel = tuple(
            (torque[i] - cfg.angular_damping_Nmsp * old.angular_velocity_radps[i]) / cfg.inertia_kgm2[i]
            for i in range(3)
        )
        angular_velocity = _vadd(old.angular_velocity_radps, _vscale(angular_accel, dt))
        q_delta = (1.0, 0.5 * angular_velocity[0] * dt, 0.5 * angular_velocity[1] * dt, 0.5 * angular_velocity[2] * dt)
        orientation = _qnormalize(_qmul(old.orientation_quat, q_delta))
        distance = max(0.0, position[2] - cfg.body_height_m)
        if contact and not old.contact and old.velocity_mps[2] < 0:
            # Expose a touchdown frame so a controller can distinguish landing
            # from having already settled on the ground.
            mode = "landing"
        elif contact:
            mode = "grounded"
        elif old.contact and position[2] > cfg.body_height_m:
            mode = "takeoff"
        elif not old.contact and position[2] <= cfg.body_height_m + 2.0e-5 and velocity[2] < 0:
            mode = "landing"
        else:
            mode = "flight"
        self._state = BodyState(
            time_s=old.time_s + dt,
            position_m=position,
            velocity_mps=velocity,
            orientation_quat=orientation,
            angular_velocity_radps=angular_velocity,
            wing_drive_left=dl,
            wing_drive_right=dr,
            wing_phase_left_rad=phase_l,
            wing_phase_right_rad=phase_r,
            contact=contact,
            ground_distance_m=distance,
            flight_mode=mode,
            acceleration_mps2=acceleration,
            angular_acceleration_radps2=angular_accel,
            aerodynamic_force_N=aero_world,
            gravity_force_N=gravity_world,
            contact_force_N=contact_force,
            total_force_N=total_force,
            total_torque_Nm=torque,
        )
        self._previous_contact = contact
        return self._state.copy()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly, immutable-by-convention state snapshot."""

        return self._state.to_dict(include_units=True)

    def restore(self, snapshot: Mapping[str, Any]) -> BodyState:
        """Restore a snapshot produced by :meth:`snapshot`."""

        if not isinstance(snapshot, Mapping):
            raise TypeError("snapshot must be a mapping produced by snapshot()")
        # Unlike reset(), restore preserves telemetry and transient mode
        # labels exactly.  This makes a snapshot a deterministic checkpoint,
        # including the one frame in which touchdown is reported as landing.
        restored = self._state_from_mapping(snapshot)
        if restored.position_m[2] < self.config.body_height_m - 1.0e-12:
            raise ValueError("snapshot position is below the configured ground plane")
        self._state = restored
        self._previous_contact = restored.contact
        return self._state.copy()


class MuJoCoBackend:
    """Explicit entry point for a future MuJoCo/Flybody adapter.

    This class intentionally does not claim to simulate anything yet.  It
    gives callers a stable, explicit failure mode when the optional runtime is
    present but a model/actuator/contact mapping has not been validated.  The
    factory never selects it from ``backend='auto'``.
    """

    backend_name = "mujoco"
    name = "mujoco"
    validated = False

    def __init__(self, *args: Any, simulator: str = "mujoco", **kwargs: Any) -> None:
        del args, kwargs  # reserved for the eventual model adapter contract
        choice = str(simulator).lower()
        if choice not in ("mujoco", "flybody"):
            raise ValueError("simulator must be 'mujoco' or 'flybody'")
        capability = detect_optional_backends()[choice]
        self.backend_name = choice
        self.name = choice
        self.capability = capability
        if not capability.installed:
            package = f"'{choice}'"
            raise OptionalBackendUnavailable(
                f"{choice} backend requested, but the optional {package} package is not installed; "
                "use backend='simple' or configure the optional simulator outside this module."
            )
        raise OptionalBackendNotValidated(
            f"{choice} is installed, but the DigitalBody {choice} adapter is not validated: "
            "provide an explicit model and actuator/contact mapping before using it."
        )


class FlybodyBackend(MuJoCoBackend):
    """Named alias for callers that use the Flybody backend spelling."""

    backend_name = "flybody"
    name = "flybody"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, simulator="flybody", **kwargs)


def create_body_backend(backend: str = "auto", config: Optional[BodyConfig] = None) -> Any:
    """Create a body backend, defaulting safely to the dependency-free model."""

    choice = str(backend).lower()
    if choice in ("auto", "simple", "reduced", "fallback"):
        return SimpleWingedBodyBackend(config=config)
    if choice == "mujoco":
        return MuJoCoBackend(config=config, simulator="mujoco")
    if choice == "flybody":
        return FlybodyBackend(config=config)
    raise ValueError(f"unknown body backend {backend!r}; choose 'auto', 'simple', 'mujoco', or 'flybody'")


# Short names used by a few callers that do not need to distinguish whether
# the deterministic backend models wings explicitly or as an aggregate drive.
SimpleBodyBackend = SimpleWingedBodyBackend
DeterministicBodyBackend = SimpleWingedBodyBackend


class DigitalBody:
    """Public facade used by CNS/brain controllers."""

    def __init__(self, backend: str = "auto", config: Optional[BodyConfig] = None) -> None:
        self.backend = create_body_backend(backend, config=config)

    @property
    def state(self) -> BodyState:
        return self.backend.state

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend, "backend_name", type(self.backend).__name__))

    @property
    def backend_validated(self) -> bool:
        return bool(getattr(self.backend, "validated", False))

    def reset(self, state: Optional[Any] = None, *, seed: Optional[int] = None) -> BodyState:
        # ``seed`` is accepted for Gym-like callers.  The reduced engine has no
        # stochastic terms, so consuming it would only make reset less clear.
        del seed
        return self.backend.reset(state)

    def step(
        self,
        command: Any = None,
        dt_s: Optional[float] = None,
        *,
        dt: Optional[float] = None,
        timestamp: Any = None,
    ) -> BodyState:
        # The shared WorldModel protocol passes ``Timestamp`` as the second
        # positional argument.  Recognise that shape without changing the
        # existing ``step(command, dt_s=...)`` API.
        if timestamp is None and dt_s is not None and not isinstance(dt_s, (int, float)):
            if hasattr(dt_s, "t_s") or hasattr(dt_s, "time_s") or isinstance(dt_s, Mapping):
                timestamp, dt_s = dt_s, None
        if timestamp is not None:
            if dt is not None or dt_s is not None:
                raise TypeError("pass timestamp or dt_s/dt, not both")
            return self.step_contract(command, timestamp)
        if dt is not None:
            if dt_s is not None:
                raise TypeError("pass only one of dt_s or dt")
            dt_s = dt
        return self.backend.step(command, dt_s=dt_s)

    def step_contract(self, command: Any, timestamp: Any) -> BodyState:
        """Step from a shared-contract timestamp using its elapsed SI time.

        ``MotorCommand`` is accepted by :meth:`BodyCommand.from_any`; its
        thrust acceleration becomes a body-frame force and its three angular
        rate fields become a rate target controlled by ``angular_rate_tau_s``.
        """

        target_time = _timestamp_seconds(timestamp)
        delta = target_time - self.state.time_s
        if delta < -1.0e-12:
            raise ValueError(
                f"contract timestamp {target_time} s precedes body time {self.state.time_s} s"
            )
        if delta <= 1.0e-12:
            period = _timestamp_period(timestamp)
            if period is None:
                raise ValueError("contract timestamp must advance beyond the current body time")
            delta = period
        return self.step(command, dt_s=delta)

    def contract_state(self, timestamp: Any = None, *, source: Any = "handbuilt") -> Any:
        """Export the current state as the optional shared BodyState record."""

        return self.state.to_contract(timestamp, source=source)

    def snapshot(self) -> dict[str, Any]:
        return self.backend.snapshot()

    def restore(self, snapshot: Mapping[str, Any]) -> BodyState:
        return self.backend.restore(snapshot)

    def step_gym(self, command: Any = None, dt_s: Optional[float] = None) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Convenience bridge to Gymnasium-style consumers.

        This body has no task reward and no terminal condition; reward is zero,
        ``terminated`` is always false, and ``truncated`` is false.
        """

        state = self.step(command, dt_s=dt_s)
        return state.to_dict(), 0.0, False, False, {"flight_mode": state.flight_mode}


__all__ = [
    "BackendCapability", "BodyBackendError", "BodyCommand", "BodyConfig", "BodyState",
    "DigitalBody", "FlybodyBackend", "MuJoCoBackend", "OptionalBackendNotValidated",
    "OptionalBackendUnavailable", "SimpleBodyBackend", "SimpleWingedBodyBackend",
    "DeterministicBodyBackend", "create_body_backend", "detect_optional_backends",
    "twin_contracts_status",
]


if __name__ == "__main__":
    body = DigitalBody()
    print("backend:", body.backend.backend_name)
    print("optional:", {k: v.to_dict() for k, v in detect_optional_backends().items()})
    body.reset()
    for _ in range(20):
        body.step({"left": 1.0, "right": 1.0})
    print(body.snapshot())
