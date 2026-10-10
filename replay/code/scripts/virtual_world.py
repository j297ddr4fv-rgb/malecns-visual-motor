#!/usr/bin/env python3
"""A small, deterministic, headless 3-D flight world.

The project uses this module as a stimulus source for the visual and flight
controllers.  It intentionally does not depend on a game/physics engine: the
world is a handful of axis-aligned boxes, a ground plane, analytic
band-limited textures, and a small pin-hole ray caster.  This keeps test runs
portable and makes the output reproducible on CPU-only machines.

Coordinate convention
---------------------
All distances are metres, time is seconds, linear velocity is metres/second,
and angles are radians.  ``+z`` is up.  At yaw=0 a fly looks along ``+x``;
positive yaw turns the view towards ``+y``.  A pose's position is its body
reference point (not the bottom of the body), and ``altitude_m`` is the
reference-point height above the ground.

Public entry points
-------------------
``VirtualWorld.from_json`` loads ``config/scenarios.json``.  ``run_scenario``
returns a :class:`ScenarioResult`, including telemetry and optional left/right
frames or optic-flow ground truth.  ``VirtualWorld.manifest`` is a stable
description of the scene and is suitable for recording beside a dataset.
``VirtualWorld.goal_at`` and ``phase_at`` expose goal-only mission commands
and phase timing without exposing or scripting the integrated body state.
``PhysicalStimulus`` describes a deterministic moving rectangular board.  A
stimulus supplied to ``render_eye``/``render_stereo``, ``trace_ray``,
``collision_events`` or ``run_scenario`` is part of the same geometry query at
that timestamp; ``stimulus_state_at`` is explicitly logging/acceptance-only.

The command line is intentionally modest::

    python scripts/virtual_world.py --self-test
    python scripts/virtual_world.py --scenario hover --render --flow

No display, package installation, or writable cache is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence


_EPS = 1.0e-9
_TWO_PI = 2.0 * math.pi
_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "scenarios.json"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _vec3(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain three numbers")
    try:
        vals = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain three numbers") from exc
    if len(vals) != 3:
        raise ValueError(f"{name} must contain three numbers")
    return tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(vals))  # type: ignore[return-value]


def _vec2(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain two numbers")
    try:
        vals = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two numbers") from exc
    if len(vals) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(vals))  # type: ignore[return-value]


def _bounds4(value: Any, name: str) -> tuple[float, float, float, float]:
    try:
        vals = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain four numbers") from exc
    if len(vals) != 4:
        raise ValueError(f"{name} must contain four numbers")
    out = tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(vals))
    if out[0] >= out[1] or out[2] >= out[3]:
        raise ValueError(f"{name} must be [xmin, xmax, ymin, ymax]")
    return out  # type: ignore[return-value]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def _smoothstep(u: float) -> float:
    """Cubic ease-in/ease-out used for reference commands."""
    x = _clamp(u)
    return x * x * (3.0 - 2.0 * x)


def _smoothstep_derivative(u: float) -> float:
    """Derivative of :func:`_smoothstep` with respect to its unit interval."""
    x = _clamp(u)
    return 6.0 * x * (1.0 - x)


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Sequence[float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, _dot(a, a)))


def _unit(a: Sequence[float], fallback: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> tuple[float, float, float]:
    n = _norm(a)
    return fallback if n < _EPS else _scale(a, 1.0 / n)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi), with deterministic endpoint handling."""
    out = (angle + math.pi) % _TWO_PI - math.pi
    return -math.pi if out == math.pi else out


def _hash_phase(seed: int, text: str) -> float:
    """A stable phase in [0, 2*pi), independent of Python's hash randomisation."""
    raw = hashlib.sha256(f"{int(seed)}:{text}".encode("utf-8")).digest()
    return (int.from_bytes(raw[:8], "big") / float(1 << 64)) * _TWO_PI


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return float(value)
    return value


def _box_basis(direction_m: Sequence[float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return right/up/normal axes for a rectangular board.

    ``direction_m`` is the board's outward normal, not a motor command.  The
    deterministic reference-up choice keeps the orientation continuous for
    ordinary palm-like poses while still handling a vertical normal.
    """
    normal = _unit(direction_m, (1.0, 0.0, 0.0))
    reference_up = (0.0, 0.0, 1.0)
    if abs(_dot(normal, reference_up)) > 0.98:
        reference_up = (0.0, 1.0, 0.0)
    right = _unit(_cross(reference_up, normal), (0.0, 1.0, 0.0))
    up = _unit(_cross(normal, right), reference_up)
    return right, up, normal


def _local_point(point: Sequence[float], center: Sequence[float], basis: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    delta = _sub(point, center)
    return tuple(_dot(delta, axis) for axis in basis)  # type: ignore[return-value]


def _world_point(point: Sequence[float], center: Sequence[float], basis: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    return _add(center, tuple(sum(point[j] * basis[j][i] for j in range(3)) for i in range(3)))


def _stimulus_in_plane_half_size(size_m: Sequence[float]) -> float:
    """Visible angular half-size of a board face.

    ``size_m`` is width/height/thickness.  The thickness is not part of the
    visual looming radius; use the larger in-plane half extent so a square
    9 cm face has radius 0.045 m instead of the 3-D diagonal proxy.
    """
    return 0.5 * max(_finite(size_m[0], "size_m[0]"), _finite(size_m[1], "size_m[1]"))


@dataclass(frozen=True)
class Pose:
    """Body pose and velocity in world coordinates."""

    position_m: tuple[float, float, float]
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_radps: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vec3(self.position_m, "position_m"))
        object.__setattr__(self, "yaw_rad", _finite(self.yaw_rad, "yaw_rad"))
        object.__setattr__(self, "pitch_rad", _finite(self.pitch_rad, "pitch_rad"))
        object.__setattr__(self, "roll_rad", _finite(self.roll_rad, "roll_rad"))
        object.__setattr__(self, "velocity_mps", _vec3(self.velocity_mps, "velocity_mps"))
        object.__setattr__(self, "angular_velocity_radps", _vec3(self.angular_velocity_radps, "angular_velocity_radps"))

    @property
    def altitude_m(self) -> float:
        return self.position_m[2]

    @property
    def speed_mps(self) -> float:
        return _norm(self.velocity_mps)

    @property
    def heading_rad(self) -> float:
        return self.yaw_rad

    @property
    def orientation_rpy_rad(self) -> tuple[float, float, float]:
        """Shared-contract ordering: roll, pitch, yaw."""
        return (self.roll_rad, self.pitch_rad, self.yaw_rad)

    @property
    def orientation_rad(self) -> tuple[float, float, float]:
        return self.orientation_rpy_rad

    @property
    def angular_velocity_rad_s(self) -> tuple[float, float, float]:
        return self.angular_velocity_radps

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Pose":
        if not isinstance(value, Mapping):
            raise ValueError("pose must be a mapping")
        nested = value.get("body")
        if isinstance(nested, Mapping):
            value = nested
        orientation = value.get("orientation_rad", value.get("orientation_rpy_rad"))
        if orientation is not None:
            rpy = _vec3(orientation, "orientation_rad")
            yaw = rpy[2]
            pitch = rpy[1]
            roll = rpy[0]
        else:
            yaw = value.get("yaw_rad", value.get("yaw", 0.0))
            pitch = value.get("pitch_rad", value.get("pitch", 0.0))
            roll = value.get("roll_rad", value.get("roll", 0.0))
        return cls(
            position_m=_vec3(value.get("position_m", value.get("position", (0.0, 0.0, 0.0))), "position_m"),
            yaw_rad=yaw,
            pitch_rad=pitch,
            roll_rad=roll,
            velocity_mps=_vec3(value.get("velocity_mps", value.get("velocity", (0.0, 0.0, 0.0))), "velocity_mps"),
            angular_velocity_radps=_vec3(value.get("angular_velocity_radps", value.get("angular_velocity_rad_s", (0.0, 0.0, 0.0))), "angular_velocity_radps"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": list(self.position_m),
            "yaw_rad": self.yaw_rad,
            "pitch_rad": self.pitch_rad,
            "roll_rad": self.roll_rad,
            "orientation_rad": list(self.orientation_rpy_rad),
            "velocity_mps": list(self.velocity_mps),
            "angular_velocity_radps": list(self.angular_velocity_radps),
            "angular_velocity_rad_s": list(self.angular_velocity_rad_s),
        }


def _coerce_pose(value: Any) -> Pose:
    """Accept Pose, mapping, BodyState, or WorldState at public boundaries."""
    if isinstance(value, Pose):
        return value
    # A WorldState owns a BodyState; accepting it keeps the renderer aligned
    # with the shared twin contract without importing that module here.
    body = getattr(value, "body", value)
    position = getattr(body, "position_m", None)
    if position is not None:
        orientation = getattr(body, "orientation_rad", (0.0, 0.0, 0.0))
        velocity = getattr(body, "velocity_mps", (0.0, 0.0, 0.0))
        angular = getattr(body, "angular_velocity_rad_s", (0.0, 0.0, 0.0))
        # The contract uses roll, pitch, yaw; this module's public Pose uses
        # yaw, pitch, roll.
        return Pose(position, orientation[2], orientation[1], orientation[0], velocity, angular)
    if isinstance(value, Mapping):
        return Pose.from_mapping(value)
    raise TypeError("pose must be Pose, a pose mapping, BodyState, or WorldState")


@dataclass(frozen=True)
class AABB:
    """An axis-aligned scene box."""

    id: str
    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]
    texture: str = "checker"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        lo, hi = _vec3(self.min_m, f"{self.id}.min_m"), _vec3(self.max_m, f"{self.id}.max_m")
        if any(lo[i] >= hi[i] for i in range(3)):
            raise ValueError(f"{self.id} min_m must be less than max_m")
        object.__setattr__(self, "min_m", lo)
        object.__setattr__(self, "max_m", hi)
        object.__setattr__(self, "texture", str(self.texture))

    @property
    def size_m(self) -> tuple[float, float, float]:
        return _sub(self.max_m, self.min_m)

    @property
    def center_m(self) -> tuple[float, float, float]:
        return _scale(_add(self.min_m, self.max_m), 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "min_m": list(self.min_m), "max_m": list(self.max_m), "texture": self.texture}


Obstacle = AABB


@dataclass(frozen=True)
class Target:
    id: str
    position_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    texture: str = "concentric_checker"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "position_m", _vec3(self.position_m, f"{self.id}.position_m"))
        size = _vec3(self.size_m, f"{self.id}.size_m")
        if any(v <= 0.0 for v in size):
            raise ValueError(f"{self.id}.size_m must be positive")
        object.__setattr__(self, "size_m", size)
        object.__setattr__(self, "texture", str(self.texture))

    @property
    def min_m(self) -> tuple[float, float, float]:
        return _sub(self.position_m, _scale(self.size_m, 0.5))

    @property
    def max_m(self) -> tuple[float, float, float]:
        return _add(self.position_m, _scale(self.size_m, 0.5))

    @property
    def box(self) -> AABB:
        return AABB(self.id, self.min_m, self.max_m, self.texture)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "position_m": list(self.position_m), "size_m": list(self.size_m), "texture": self.texture}


@dataclass(frozen=True)
class AirDisturbance:
    """Optional local air field attached to a physical visual stimulus.

    The field is deliberately an environmental input only.  It carries a
    velocity and/or acceleration perturbation and never contains a motor,
    task, or behavioural command.  ``radius_m == 0`` means a spatially
    uniform field; otherwise the configured falloff is applied around the
    moving stimulus centre.
    """

    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_m: float = 0.0
    start_time_s: float = 0.0
    duration_s: Optional[float] = None
    falloff: str = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "velocity_mps", _vec3(self.velocity_mps, "air_disturbance.velocity_mps"))
        object.__setattr__(self, "acceleration_mps2", _vec3(self.acceleration_mps2, "air_disturbance.acceleration_mps2"))
        object.__setattr__(self, "radius_m", max(0.0, _finite(self.radius_m, "air_disturbance.radius_m")))
        object.__setattr__(self, "start_time_s", _finite(self.start_time_s, "air_disturbance.start_time_s"))
        if self.duration_s is not None:
            duration = max(0.0, _finite(self.duration_s, "air_disturbance.duration_s"))
            object.__setattr__(self, "duration_s", duration)
        mode = str(self.falloff).lower().replace("-", "_")
        if mode not in {"linear", "hard", "inverse_square", "inverse"}:
            raise ValueError("air_disturbance.falloff must be linear, hard, or inverse_square")
        object.__setattr__(self, "falloff", mode)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AirDisturbance | None":
        if value is None:
            return None
        if isinstance(value, AirDisturbance):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("air_disturbance must be a mapping")
        return cls(
            value.get("velocity_mps", value.get("air_velocity_mps", value.get("vector_mps", (0.0, 0.0, 0.0)))),
            value.get("acceleration_mps2", value.get("air_acceleration_mps2", (0.0, 0.0, 0.0))),
            value.get("radius_m", value.get("influence_radius_m", 0.0)),
            value.get("start_time_s", value.get("start_s", 0.0)),
            value.get("duration_s"),
            value.get("falloff", "linear"),
        )

    def active(self, t_s: float) -> bool:
        t = _finite(t_s, "t_s")
        if t < self.start_time_s:
            return False
        return self.duration_s is None or t <= self.start_time_s + self.duration_s + _EPS

    def weight(self, position_m: Sequence[float], center_m: Sequence[float], t_s: float) -> float:
        if not self.active(t_s):
            return 0.0
        if self.radius_m <= _EPS:
            return 1.0
        distance = _norm(_sub(position_m, center_m))
        ratio = distance / self.radius_m
        if self.falloff == "hard":
            return 1.0 if ratio <= 1.0 else 0.0
        if self.falloff in {"inverse", "inverse_square"}:
            return 1.0 / (1.0 + ratio * ratio)
        return _clamp(1.0 - ratio)

    def at(self, position_m: Sequence[float], center_m: Sequence[float], t_s: float) -> dict[str, Any]:
        weight = self.weight(position_m, center_m, t_s)
        return {
            "weight": weight,
            "velocity_mps": list(_scale(self.velocity_mps, weight)),
            "acceleration_mps2": list(_scale(self.acceleration_mps2, weight)),
            "active": bool(weight > 0.0),
        }

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "velocity_mps": list(self.velocity_mps),
            "acceleration_mps2": list(self.acceleration_mps2),
            "radius_m": self.radius_m,
            "start_time_s": self.start_time_s,
            "falloff": self.falloff,
        }
        if self.duration_s is not None:
            out["duration_s"] = self.duration_s
        return out


@dataclass(frozen=True)
class OrientedBox:
    """A rectangular moving board in world coordinates.

    ``size_m`` is ordered as local width, height, and thickness.  The local
    normal is ``direction_m``; the other two axes are derived deterministically
    by :func:`_box_basis`.
    """

    id: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    direction_m: tuple[float, float, float]
    texture: str = "palm"
    kind: str = "stimulus"
    albedo: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "center_m", _vec3(self.center_m, f"{self.id}.center_m"))
        size = _vec3(self.size_m, f"{self.id}.size_m")
        if any(v <= 0.0 for v in size):
            raise ValueError(f"{self.id}.size_m must be positive")
        object.__setattr__(self, "size_m", size)
        direction = _vec3(self.direction_m, f"{self.id}.direction_m")
        if _norm(direction) < _EPS:
            raise ValueError(f"{self.id}.direction_m must be non-zero")
        object.__setattr__(self, "direction_m", _unit(direction))
        object.__setattr__(self, "texture", str(self.texture))
        object.__setattr__(self, "kind", str(self.kind))
        if self.albedo is not None:
            object.__setattr__(self, "albedo", _clamp(_finite(self.albedo, f"{self.id}.albedo")))

    @property
    def basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        return _box_basis(self.direction_m)

    @property
    def min_m(self) -> tuple[float, float, float]:
        """A conservative world AABB, useful for broad-phase checks."""
        half = _scale(self.size_m, 0.5)
        extents = tuple(sum(abs(axis[i]) * half[j] for j, axis in enumerate(self.basis)) for i in range(3))
        return _sub(self.center_m, extents)

    @property
    def max_m(self) -> tuple[float, float, float]:
        half = _scale(self.size_m, 0.5)
        extents = tuple(sum(abs(axis[i]) * half[j] for j, axis in enumerate(self.basis)) for i in range(3))
        return _add(self.center_m, extents)

    @property
    def box(self) -> AABB:
        return AABB(self.id, self.min_m, self.max_m, self.texture)

    def to_dict(self) -> dict[str, Any]:
        out = {"id": self.id, "center_m": list(self.center_m), "size_m": list(self.size_m),
               "direction_m": list(self.direction_m), "texture": self.texture, "kind": self.kind}
        if self.albedo is not None:
            out["albedo"] = self.albedo
        return out


@dataclass(frozen=True)
class StimulusState:
    """Time-sampled state of one physical stimulus geometry."""

    id: str
    time_s: float
    active: bool
    renderable: bool
    collidable: bool
    mode: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    direction_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    texture: str
    albedo: Optional[float] = None
    forbidden_for_control: bool = True
    visual_only: bool = False
    non_interacting: bool = False
    counterfactual_contact_time_s: Optional[float] = None

    @property
    def center_m(self) -> tuple[float, float, float]:
        return self.position_m

    @property
    def geometry(self) -> OrientedBox:
        return OrientedBox(self.id, self.position_m, self.size_m, self.direction_m, self.texture, "stimulus", self.albedo)

    @property
    def box(self) -> OrientedBox:
        return self.geometry

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "time_s": self.time_s, "active": self.active,
                "renderable": self.renderable, "collidable": self.collidable, "mode": self.mode,
                "position_m": list(self.position_m), "velocity_mps": list(self.velocity_mps),
                "direction_m": list(self.direction_m), "size_m": list(self.size_m), "texture": self.texture,
                "albedo": self.albedo,
                "forbidden_for_control": True,
                "visual_only": self.visual_only,
                "non_interacting": self.non_interacting,
                "counterfactual_contact_time_s": self.counterfactual_contact_time_s}


@dataclass(frozen=True)
class PhysicalStimulus:
    """Deterministic parameterised moving rectangular stimulus.

    The stimulus is a piece of scene geometry.  ``start_position_m`` is its
    centre at ``start_time_s``; ``velocity_mps`` is translational velocity;
    ``direction_m`` is the board normal; and ``size_m`` is local
    width/height/thickness.  ``mode`` names only the physical visual
    trajectory (for example ``looming`` or ``contracting``).
    """

    id: str = "palm"
    start_position_m: tuple[float, float, float] = (1.5, 0.0, 0.42)
    velocity_mps: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    direction_m: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    size_m: tuple[float, float, float] = (0.70, 0.48, 0.04)
    start_time_s: float = 0.0
    duration_s: Optional[float] = None
    mode: str = "looming"
    texture: str = "palm"
    end_size_m: Optional[tuple[float, float, float]] = None
    air_disturbance: Optional[AirDisturbance] = None
    renderable: Optional[bool] = None
    collidable: Optional[bool] = None
    seed: Optional[int] = None
    anchor: str = "world"
    offset_m: Optional[tuple[float, float, float]] = None
    contact_radius_m: Optional[float] = None
    albedo: Optional[float] = None
    appearance_provenance: str = "configured"
    visual_only: bool = False
    non_interacting: bool = False
    counterfactual_contact_time_s: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "start_position_m", _vec3(self.start_position_m, f"{self.id}.start_position_m"))
        object.__setattr__(self, "velocity_mps", _vec3(self.velocity_mps, f"{self.id}.velocity_mps"))
        direction = _vec3(self.direction_m, f"{self.id}.direction_m")
        if _norm(direction) < _EPS:
            direction = _unit(self.velocity_mps, (1.0, 0.0, 0.0))
        object.__setattr__(self, "direction_m", _unit(direction))
        size = _vec3(self.size_m, f"{self.id}.size_m")
        if any(v <= 0.0 for v in size):
            raise ValueError(f"{self.id}.size_m must be positive")
        object.__setattr__(self, "size_m", size)
        if self.end_size_m is not None:
            end_size = _vec3(self.end_size_m, f"{self.id}.end_size_m")
            if any(v <= 0.0 for v in end_size):
                raise ValueError(f"{self.id}.end_size_m must be positive")
            object.__setattr__(self, "end_size_m", end_size)
        object.__setattr__(self, "start_time_s", _finite(self.start_time_s, f"{self.id}.start_time_s"))
        if self.duration_s is not None:
            duration = max(0.0, _finite(self.duration_s, f"{self.id}.duration_s"))
            object.__setattr__(self, "duration_s", duration)
        mode = str(self.mode).lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "loom": "looming", "approach": "looming", "expand": "looming",
            "contract": "contracting", "shrink": "contracting",
            "matched_control": "translation", "translation_control": "translation",
            "stationary": "static", "none": "sham",
        }
        mode = aliases.get(mode, mode)
        allowed = {"looming", "contracting", "translation", "static", "sham"}
        if mode not in allowed:
            raise ValueError(f"{self.id}.mode must be one of {sorted(allowed)}")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "texture", str(self.texture))
        disturbance = AirDisturbance.from_mapping(self.air_disturbance) if isinstance(self.air_disturbance, Mapping) else self.air_disturbance
        object.__setattr__(self, "air_disturbance", disturbance)
        if self.renderable is None:
            object.__setattr__(self, "renderable", mode != "sham")
        else:
            object.__setattr__(self, "renderable", bool(self.renderable))
        if self.collidable is None:
            object.__setattr__(self, "collidable", mode != "sham")
        else:
            object.__setattr__(self, "collidable", bool(self.collidable))
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))
        anchor = str(self.anchor).lower().replace("-", "_")
        if anchor in {"body", "fly", "pose", "relative", "relative_pose"}:
            anchor = "body"
        elif anchor not in {"world", "absolute"}:
            raise ValueError(f"{self.id}.anchor must be world or body")
        object.__setattr__(self, "anchor", "world" if anchor == "absolute" else anchor)
        if self.offset_m is not None:
            object.__setattr__(self, "offset_m", _vec3(self.offset_m, f"{self.id}.offset_m"))
        if self.contact_radius_m is not None:
            object.__setattr__(self, "contact_radius_m", max(0.0, _finite(self.contact_radius_m, f"{self.id}.contact_radius_m")))
        if self.albedo is not None:
            object.__setattr__(self, "albedo", _clamp(_finite(self.albedo, f"{self.id}.albedo")))
        object.__setattr__(self, "appearance_provenance", str(self.appearance_provenance))
        visual_only = bool(self.visual_only)
        non_interacting = bool(self.non_interacting or visual_only)
        object.__setattr__(self, "visual_only", visual_only)
        object.__setattr__(self, "non_interacting", non_interacting)
        if non_interacting:
            object.__setattr__(self, "collidable", False)
        if self.counterfactual_contact_time_s is not None:
            object.__setattr__(self, "counterfactual_contact_time_s", max(0.0, _finite(self.counterfactual_contact_time_s, f"{self.id}.counterfactual_contact_time_s")))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, seed: Optional[int] = None) -> "PhysicalStimulus":
        if isinstance(value, PhysicalStimulus):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("stimulus must be a mapping")
        d = dict(value)
        random_spec = d.get("randomization", d.get("randomize"))
        if random_spec:
            random_map = random_spec if isinstance(random_spec, Mapping) else {}
            random_seed = seed if seed is not None else random_map.get("seed", d.get("seed", 1729))
            generated = cls.randomized(
                int(random_seed), id=d.get("id", "stimulus"),
                mode=d.get("mode", d.get("condition", "looming")),
                start_position_bounds=random_map.get("start_position_bounds", ((0.9, 1.8), (-0.45, 0.45), (0.25, 0.75))),
                speed_range_mps=random_map.get("speed_range_mps", (0.5, 1.5)),
                size_range_m=random_map.get("size_range_m", (0.38, 0.78)),
                start_time_s=random_map.get("start_time_s", d.get("start_time_s", 0.5)),
                duration_s=random_map.get("duration_s", d.get("duration_s", 1.8)),
            )
            generated_map = generated.to_dict()
            generated_map.update(d)
            d = generated_map
            d.pop("randomization", None)
            d.pop("randomize", None)
        direction_value = d.get("direction_m", d.get("direction", d.get("normal_m", (-1.0, 0.0, 0.0))))
        direction = _vec3(direction_value, "stimulus.direction_m")
        velocity_value = d.get("velocity_mps", d.get("velocity"))
        if velocity_value is None and ("speed_mps" in d or "speed" in d):
            velocity_value = _scale(_unit(direction), _finite(d.get("speed_mps", d.get("speed")), "stimulus.speed_mps"))
        if velocity_value is None:
            velocity_value = (0.0, 0.0, 0.0)
        air_value = d.get("air_disturbance", d.get("airflow"))
        if air_value is None and any(key in d for key in ("air_velocity_mps", "air_acceleration_mps2", "air_radius_m")):
            air_value = {
                "velocity_mps": d.get("air_velocity_mps", (0.0, 0.0, 0.0)),
                "acceleration_mps2": d.get("air_acceleration_mps2", (0.0, 0.0, 0.0)),
                "radius_m": d.get("air_radius_m", 0.0),
                "start_time_s": d.get("air_start_time_s", 0.0),
                "duration_s": d.get("air_duration_s"),
                "falloff": d.get("air_falloff", "linear"),
            }
        return cls(
            id=d.get("id", d.get("name", "palm")),
            start_position_m=d.get("start_position_m", d.get("start_point_m", d.get("start_m", d.get("start", d.get("position_m", (1.5, 0.0, 0.42)))))),
            velocity_mps=velocity_value,
            direction_m=direction,
            size_m=d.get("size_m", d.get("size", d.get("dimensions_m", (0.70, 0.48, 0.04)))),
            start_time_s=d.get("start_time_s", d.get("start_time", d.get("start_s", d.get("onset_s", d.get("time_start_s", 0.0))))),
            duration_s=d.get("duration_s"),
            mode=d.get("mode", d.get("condition", d.get("variant", d.get("stimulus_type", "looming")))),
            texture=d.get("texture", "palm"),
            end_size_m=d.get("end_size_m", d.get("final_size_m", d.get("size_end_m"))),
            air_disturbance=air_value,
            renderable=d.get("renderable"),
            collidable=d.get("collidable"),
            seed=d.get("seed", seed),
            anchor=d.get("anchor", d.get("relative_to", "world")),
            offset_m=d.get("offset_m", d.get("start_offset_m")),
            contact_radius_m=d.get("contact_radius_m", d.get("collision_radius_m")),
            albedo=d.get("albedo", d.get("base_luminance", d.get("luminance"))),
            appearance_provenance=d.get("appearance_provenance", d.get("albedo_provenance", "configured")),
            visual_only=d.get("visual_only", False),
            non_interacting=d.get("non_interacting", False),
            counterfactual_contact_time_s=d.get("counterfactual_contact_time_s", d.get("theoretical_contact_time_s", d.get("first_contact_time_s"))),
        )

    @classmethod
    def from_parameters(cls, *, start_position_m: Sequence[float], speed_mps: float,
                        direction_m: Sequence[float], size_m: Sequence[float], start_time_s: float = 0.0,
                        duration_s: Optional[float] = None, mode: str = "looming", **kwargs: Any) -> "PhysicalStimulus":
        """Construct from the compact start/speed/direction protocol."""
        direction = _unit(_vec3(direction_m, "direction_m"))
        return cls(start_position_m=_vec3(start_position_m, "start_position_m"),
                   velocity_mps=_scale(direction, _finite(speed_mps, "speed_mps")),
                   direction_m=direction, size_m=_vec3(size_m, "size_m"),
                   start_time_s=start_time_s, duration_s=duration_s, mode=mode, **kwargs)

    from_spec = from_parameters

    @classmethod
    def randomized(cls, seed: int = 1729, *, id: str = "palm", mode: str = "looming",
                   start_position_bounds: Sequence[Sequence[float]] = ((0.9, 1.8), (-0.45, 0.45), (0.25, 0.75)),
                   speed_range_mps: Sequence[float] = (0.5, 1.5),
                   size_range_m: Sequence[float] = (0.38, 0.78),
                   start_time_s: float = 0.5, duration_s: Optional[float] = 1.8) -> "PhysicalStimulus":
        """Create a deterministic randomised stimulus from an integer seed.

        All random draws use a private ``random.Random`` instance.  Recording
        :meth:`to_dict` is therefore sufficient for exact replay without
        relying on process-global random state.
        """
        rng = random.Random(int(seed))
        bounds = tuple(tuple(_finite(v, "start_position_bounds") for v in pair) for pair in start_position_bounds)
        if len(bounds) != 3 or any(len(pair) != 2 or pair[0] > pair[1] for pair in bounds):
            raise ValueError("start_position_bounds must contain three [min, max] pairs")
        start = tuple(rng.uniform(pair[0], pair[1]) for pair in bounds)
        speed_lo, speed_hi = (_finite(speed_range_mps[0], "speed_range_mps[0]"), _finite(speed_range_mps[1], "speed_range_mps[1]"))
        size_lo, size_hi = (_finite(size_range_m[0], "size_range_m[0]"), _finite(size_range_m[1], "size_range_m[1]"))
        if speed_lo > speed_hi or size_lo <= 0.0 or size_lo > size_hi:
            raise ValueError("invalid speed_range_mps or size_range_m")
        speed = rng.uniform(speed_lo, speed_hi)
        # The randomised board approaches the origin along its centreline;
        # lateral and vertical placement still vary independently.
        direction = _unit((-1.0, rng.uniform(-0.20, 0.20), rng.uniform(-0.12, 0.12)))
        velocity = _scale(direction, speed)
        width = rng.uniform(size_lo, size_hi)
        height = width * rng.uniform(0.55, 0.78)
        thickness = max(0.01, width * 0.06)
        return cls(id=id, start_position_m=start, velocity_mps=velocity, direction_m=direction,
                   size_m=(width, height, thickness), start_time_s=start_time_s,
                   duration_s=duration_s, mode=mode, seed=int(seed))

    # Common spelling used by experiment drivers.
    randomised = randomized
    randomize = randomized

    @classmethod
    def from_json(cls, path: str | Path, name: str | None = None, *, seed: Optional[int] = None) -> Any:
        loaded = load_stimuli(path, seed=seed)
        if name is None:
            return next(iter(loaded.values())) if len(loaded) == 1 else loaded
        key = str(name)
        if key not in loaded:
            raise KeyError(f"unknown stimulus {name!r}; available: {sorted(loaded)}")
        return loaded[key]

    def active_at(self, t_s: float) -> bool:
        t = _finite(t_s, "t_s")
        if t < self.start_time_s:
            return False
        return self.duration_s is None or t <= self.start_time_s + self.duration_s + _EPS

    @property
    def speed_mps(self) -> float:
        return _norm(self.velocity_mps)

    @property
    def start_m(self) -> tuple[float, float, float]:
        return self.start_position_m

    @property
    def direction(self) -> tuple[float, float, float]:
        return self.direction_m

    def _fraction_at(self, t_s: float) -> float:
        if t_s <= self.start_time_s:
            return 0.0
        if self.duration_s is None or self.duration_s <= _EPS:
            return 1.0
        return _clamp((t_s - self.start_time_s) / self.duration_s)

    def position_at(self, t_s: float, reference_pose: Any = None) -> tuple[float, float, float]:
        t = _finite(t_s, "t_s")
        elapsed = max(0.0, t - self.start_time_s)
        if self.duration_s is not None:
            elapsed = min(elapsed, self.duration_s)
        start = self.start_position_m
        if self.anchor == "body" and reference_pose is not None:
            reference = _coerce_pose(reference_pose)
            offset = self.start_position_m if self.offset_m is None else self.offset_m
            start = _add(reference.position_m, offset)
        return _add(start, _scale(self.velocity_mps, elapsed))

    def size_at(self, t_s: float) -> tuple[float, float, float]:
        if self.mode not in {"contracting"}:
            return self.size_m
        end = self.end_size_m if self.end_size_m is not None else _scale(self.size_m, 0.35)
        u = self._fraction_at(_finite(t_s, "t_s"))
        return tuple(self.size_m[i] + (end[i] - self.size_m[i]) * u for i in range(3))

    def state_at(self, t_s: float, reference_pose: Any = None) -> StimulusState:
        t = _finite(t_s, "t_s")
        active = self.active_at(t)
        return StimulusState(self.id, t, active, bool(active and self.renderable), bool(active and self.collidable),
                             self.mode, self.position_at(t, reference_pose), self.velocity_mps, self.direction_m,
                             self.size_at(t), self.texture, self.albedo, True, self.visual_only,
                             self.non_interacting, self.counterfactual_contact_time_s)

    at = state_at

    def geometry_at(self, t_s: float) -> OrientedBox:
        return self.state_at(t_s).geometry

    moving_box_at = geometry_at

    def contact_time_s(self, reference_position_m: Sequence[float] = (0.0, 0.0, 0.0),
                       radius_m: Optional[float] = None, reference_pose: Any = None) -> Optional[float]:
        """Solve the first physical surface-contact time for a fixed point.

        This is a geometry-only timestamp useful for experiment telemetry;
        it does not issue or infer a behavioural command.
        """
        reference = _vec3(reference_position_m, "reference_position_m")
        radius = self.contact_radius_m if radius_m is None and self.contact_radius_m is not None else (0.0 if radius_m is None else _finite(radius_m, "radius_m"))
        radius = max(0.0, radius)

        def gap(t_s: float) -> float:
            state = self.state_at(t_s, reference_pose)
            if not state.active or not state.renderable:
                return float("inf")
            box = state.geometry
            local = _local_point(reference, box.center_m, box.basis)
            half = _scale(box.size_m, 0.5)
            outside = tuple(max(abs(local[i]) - half[i], 0.0) for i in range(3))
            outside_distance = _norm(outside)
            surface = outside_distance if outside_distance > _EPS else -min(half[i] - abs(local[i]) for i in range(3))
            return surface - radius

        lo = self.start_time_s
        if gap(lo) <= 0.0:
            return lo
        if self.duration_s is not None:
            hi = self.start_time_s + self.duration_s
        else:
            speed = max(self.speed_mps, _EPS)
            hi = self.start_time_s + max(10.0, 2.0 * (_norm(_sub(self.start_position_m, reference)) + 1.0) / speed)
        if gap(hi) > 0.0:
            return None
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if gap(mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        return hi

    time_to_surface_contact = contact_time_s

    def stimulus_state_at(self, t_s: float, reference_pose: Any = None) -> dict[str, Any]:
        """Return a log/acceptance record, explicitly excluded from control."""
        out = self.state_at(t_s, reference_pose).to_dict()
        out["anchor"] = self.anchor
        if self.offset_m is not None:
            out["offset_m"] = list(self.offset_m)
        if self.contact_radius_m is not None:
            out["contact_radius_m"] = self.contact_radius_m
        out["albedo"] = self.albedo
        out["appearance_provenance"] = self.appearance_provenance
        out["visual_only"] = self.visual_only
        out["non_interacting"] = self.non_interacting
        out["counterfactual_contact_time_s"] = self.counterfactual_contact_time_s
        return out

    def air_at(self, position_m: Sequence[float], t_s: float, reference_pose: Any = None) -> dict[str, Any]:
        state = self.state_at(t_s, reference_pose)
        if not state.active or self.air_disturbance is None:
            return {"weight": 0.0, "velocity_mps": [0.0, 0.0, 0.0], "acceleration_mps2": [0.0, 0.0, 0.0], "active": False}
        return self.air_disturbance.at(position_m, state.position_m, t_s)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id, "start_position_m": list(self.start_position_m),
            "velocity_mps": list(self.velocity_mps), "speed_mps": self.speed_mps,
            "direction_m": list(self.direction_m),
            "size_m": list(self.size_m), "start_time_s": self.start_time_s,
            "mode": self.mode, "texture": self.texture, "renderable": bool(self.renderable),
            "collidable": bool(self.collidable),
            "anchor": self.anchor,
        }
        if self.duration_s is not None:
            out["duration_s"] = self.duration_s
        if self.end_size_m is not None:
            out["end_size_m"] = list(self.end_size_m)
        if self.air_disturbance is not None:
            out["air_disturbance"] = self.air_disturbance.to_dict()
        if self.seed is not None:
            out["seed"] = self.seed
        if self.offset_m is not None:
            out["offset_m"] = list(self.offset_m)
        if self.contact_radius_m is not None:
            out["contact_radius_m"] = self.contact_radius_m
        if self.albedo is not None:
            out["albedo"] = self.albedo
        out["appearance_provenance"] = self.appearance_provenance
        out["visual_only"] = self.visual_only
        out["non_interacting"] = self.non_interacting
        if self.counterfactual_contact_time_s is not None:
            out["counterfactual_contact_time_s"] = self.counterfactual_contact_time_s
        return out


# Public aliases keep the physical layer discoverable under the terminology
# used by different experiment drivers.
Stimulus = PhysicalStimulus
MovingStimulus = PhysicalStimulus
DynamicStimulus = PhysicalStimulus
StimulusConfig = PhysicalStimulus
PalmStimulus = PhysicalStimulus
SlapBoardStimulus = PhysicalStimulus
MovingBoard = PhysicalStimulus


def load_stimuli(path: str | Path, *, seed: Optional[int] = None) -> dict[str, PhysicalStimulus]:
    """Load an independent stimulus protocol without touching world scenes."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError("stimulus config root must be an object")
    base_seed = int(raw.get("seed", 1729) if seed is None else seed)
    entries = raw.get("stimuli", raw.get("conditions", raw.get("stimulus")))
    if isinstance(entries, Mapping) and any(key in entries for key in ("start_position_m", "start_point_m", "start_m", "start", "position_m", "velocity_mps", "speed_mps", "speed", "size_m")):
        entries = {str(entries.get("id", "stimulus")): entries}
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        items = {}
        for index, item in enumerate(entries):
            if not isinstance(item, Mapping):
                raise ValueError(f"stimuli[{index}] must be a mapping")
            items[str(item.get("id", item.get("name", f"stimulus_{index}")))] = item
        entries = items
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("stimulus config must contain a non-empty stimuli object")
    out: dict[str, PhysicalStimulus] = {}
    for name, item in entries.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"stimulus {name!r} must be a mapping")
        local = dict(item)
        local.setdefault("id", str(name))
        out[str(name)] = PhysicalStimulus.from_mapping(local, seed=base_seed)
    return out


load_stimulus_config = load_stimuli


@dataclass(frozen=True)
class LightConfig:
    ambient: float = 0.28
    direction_m: tuple[float, float, float] = (-0.45, 0.35, 0.82)
    intensity: float = 0.72
    sky_brightness: float = 0.10

    def __post_init__(self) -> None:
        object.__setattr__(self, "ambient", _clamp(_finite(self.ambient, "ambient")))
        direction = _vec3(self.direction_m, "direction_m")
        if _norm(direction) < _EPS:
            raise ValueError("direction_m must be non-zero")
        # Preserve the configured vector in manifests; the shader normalises
        # it at use time so input files remain round-trippable.
        object.__setattr__(self, "direction_m", direction)
        object.__setattr__(self, "intensity", max(0.0, _finite(self.intensity, "intensity")))
        object.__setattr__(self, "sky_brightness", _clamp(_finite(self.sky_brightness, "sky_brightness")))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "LightConfig":
        d = value or {}
        return cls(d.get("ambient", 0.28), d.get("direction_m", (-0.45, 0.35, 0.82)),
                   d.get("intensity", 0.72), d.get("sky_brightness", 0.10))

    def to_dict(self) -> dict[str, Any]:
        return {"ambient": self.ambient, "direction_m": list(self.direction_m), "intensity": self.intensity, "sky_brightness": self.sky_brightness}


@dataclass(frozen=True)
class TextureConfig:
    ground_cell_m: float = 0.16
    wall_cell_m: float = 0.20
    contrast: float = 0.42

    def __post_init__(self) -> None:
        object.__setattr__(self, "ground_cell_m", max(1e-4, _finite(self.ground_cell_m, "ground_cell_m")))
        object.__setattr__(self, "wall_cell_m", max(1e-4, _finite(self.wall_cell_m, "wall_cell_m")))
        object.__setattr__(self, "contrast", _clamp(_finite(self.contrast, "contrast")))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TextureConfig":
        d = value or {}
        return cls(d.get("ground_cell_m", 0.16), d.get("wall_cell_m", 0.20), d.get("contrast", 0.42))

    def to_dict(self) -> dict[str, Any]:
        return {"ground_cell_m": self.ground_cell_m, "wall_cell_m": self.wall_cell_m, "contrast": self.contrast}


@dataclass(frozen=True)
class Gust:
    start_s: float
    duration_s: float
    vector_mps: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_s", max(0.0, _finite(self.start_s, "gust.start_s")))
        object.__setattr__(self, "duration_s", max(0.0, _finite(self.duration_s, "gust.duration_s")))
        object.__setattr__(self, "vector_mps", _vec3(self.vector_mps, "gust.vector_mps"))

    def weight(self, t_s: float) -> float:
        if self.duration_s <= 0.0 or t_s < self.start_s or t_s > self.start_s + self.duration_s:
            return 0.0
        # The configured vector is a reproducible, directly controllable
        # gust.  Keeping its plateau exact makes boundary probes at
        # ``start_s`` and ``start_s + duration_s`` unambiguous.
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"start_s": self.start_s, "duration_s": self.duration_s, "vector_mps": list(self.vector_mps)}


@dataclass(frozen=True)
class WindConfig:
    base_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spatial_amplitude_mps: float = 0.03
    spatial_wavelength_m: float = 1.7
    gusts: tuple[Gust, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_mps", _vec3(self.base_mps, "base_mps"))
        object.__setattr__(self, "spatial_amplitude_mps", max(0.0, _finite(self.spatial_amplitude_mps, "spatial_amplitude_mps")))
        object.__setattr__(self, "spatial_wavelength_m", max(1e-4, _finite(self.spatial_wavelength_m, "spatial_wavelength_m")))
        object.__setattr__(self, "gusts", tuple(self.gusts))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "WindConfig":
        d = value or {}
        gusts = tuple(Gust(g.get("start_s", 0.0), g.get("duration_s", 0.0), g.get("vector_mps", (0.0, 0.0, 0.0))) for g in d.get("gusts", ()))
        return cls(d.get("base_mps", (0.0, 0.0, 0.0)), d.get("spatial_amplitude_mps", 0.03), d.get("spatial_wavelength_m", 1.7), gusts)

    def with_override(self, override: Mapping[str, Any] | None) -> "WindConfig":
        if not override:
            return self
        d = dict(override)
        return WindConfig(
            d.get("base_mps", self.base_mps),
            d.get("spatial_amplitude_mps", self.spatial_amplitude_mps),
            d.get("spatial_wavelength_m", self.spatial_wavelength_m),
            tuple(Gust(g.get("start_s", 0.0), g.get("duration_s", 0.0), g.get("vector_mps", (0.0, 0.0, 0.0))) for g in d.get("gusts", self.gusts)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"base_mps": list(self.base_mps), "spatial_amplitude_mps": self.spatial_amplitude_mps,
                "spatial_wavelength_m": self.spatial_wavelength_m, "gusts": [g.to_dict() for g in self.gusts]}


@dataclass(frozen=True)
class WorldConfig:
    ground_z_m: float
    bounds_xy_m: tuple[float, float, float, float]
    wall_height_m: float
    target: Target
    obstacles: tuple[AABB, ...]
    light: LightConfig
    texture: TextureConfig
    wind: WindConfig
    seed: int = 1729
    body_radius_m: float = 0.015

    def __post_init__(self) -> None:
        object.__setattr__(self, "ground_z_m", _finite(self.ground_z_m, "ground_z_m"))
        object.__setattr__(self, "bounds_xy_m", _bounds4(self.bounds_xy_m, "bounds_xy_m"))
        object.__setattr__(self, "wall_height_m", max(0.0, _finite(self.wall_height_m, "wall_height_m")))
        if not isinstance(self.target, Target):
            raise TypeError("target must be a Target")
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "body_radius_m", max(1e-5, _finite(self.body_radius_m, "body_radius_m")))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, seed: int = 1729) -> "WorldConfig":
        d = value
        target_d = d.get("target") or {}
        target = Target(target_d.get("id", "landing_target"), target_d.get("position_m", (0.85, 0.0, 0.02)), target_d.get("size_m", (0.32, 0.32, 0.04)), target_d.get("texture", "concentric_checker"))
        obstacles = tuple(AABB(o.get("id", f"obstacle_{i}"), o["min_m"], o["max_m"], o.get("texture", "checker")) for i, o in enumerate(d.get("obstacles", ())))
        return cls(d.get("ground_z_m", 0.0), d.get("bounds_xy_m", (-2.5, 2.5, -2.5, 2.5)), d.get("wall_height_m", 2.0), target, obstacles,
                   LightConfig.from_mapping(d.get("light")), TextureConfig.from_mapping(d.get("texture")), WindConfig.from_mapping(d.get("wind")), seed=seed, body_radius_m=d.get("body_radius_m", 0.015))

    def to_dict(self) -> dict[str, Any]:
        return {"ground_z_m": self.ground_z_m, "bounds_xy_m": list(self.bounds_xy_m), "wall_height_m": self.wall_height_m,
                "target": self.target.to_dict(), "obstacles": [o.to_dict() for o in self.obstacles], "light": self.light.to_dict(),
                "texture": self.texture.to_dict(), "wind": self.wind.to_dict(), "seed": self.seed, "body_radius_m": self.body_radius_m}


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    duration_s: float
    dt_s: float
    initial_pose: Pose
    profile: Mapping[str, Any]
    acceptance: Mapping[str, Any] = field(default_factory=dict)
    wind_override: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "duration_s", max(0.0, _finite(self.duration_s, f"{self.name}.duration_s")))
        object.__setattr__(self, "dt_s", _finite(self.dt_s, f"{self.name}.dt_s"))
        if self.dt_s <= 0.0:
            raise ValueError(f"{self.name}.dt_s must be > 0")
        object.__setattr__(self, "profile", dict(self.profile))
        object.__setattr__(self, "acceptance", dict(self.acceptance))
        object.__setattr__(self, "wind_override", None if self.wind_override is None else dict(self.wind_override))

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "ScenarioConfig":
        return cls(name, value.get("duration_s", 1.0), value.get("dt_s", 0.02), Pose.from_mapping(value.get("initial_pose", {})), value.get("profile", {"kind": "hold"}), value.get("acceptance", {}), value.get("wind_override"))

    def to_dict(self) -> dict[str, Any]:
        out = {"duration_s": self.duration_s, "dt_s": self.dt_s, "initial_pose": self.initial_pose.to_dict(), "profile": dict(self.profile), "acceptance": dict(self.acceptance)}
        if self.wind_override is not None:
            out["wind_override"] = dict(self.wind_override)
        return out


@dataclass(frozen=True)
class Hit:
    object_id: str
    distance_m: float
    position_m: tuple[float, float, float]
    normal_m: tuple[float, float, float]
    texture: str
    kind: str = "object"


@dataclass(frozen=True)
class CollisionEvent:
    t_s: float
    object_id: str
    kind: str
    position_m: tuple[float, float, float]
    normal_m: tuple[float, float, float]
    penetration_m: float = 0.0

    @property
    def is_collision(self) -> bool:
        return self.kind == "collision"

    def to_dict(self) -> dict[str, Any]:
        return {"t_s": self.t_s, "object_id": self.object_id, "kind": self.kind,
                "position_m": list(self.position_m), "normal_m": list(self.normal_m), "penetration_m": self.penetration_m}


@dataclass(frozen=True)
class TargetObservation:
    target_id: str
    distance_m: float
    bearing_rad: float
    elevation_rad: float
    visible: bool
    on_target: bool

    @property
    def target_bearing_rad(self) -> float:
        return self.bearing_rad

    def to_dict(self) -> dict[str, Any]:
        return {"target_id": self.target_id, "distance_m": self.distance_m, "bearing_rad": self.bearing_rad,
                "target_bearing_rad": self.bearing_rad, "elevation_rad": self.elevation_rad,
                "visible": self.visible, "on_target": self.on_target}


@dataclass(frozen=True)
class StimulusObservation:
    """Geometry-only observation of a moving stimulus at one timestamp."""

    stimulus_id: str
    time_s: float
    active: bool
    visible: bool
    colliding: bool
    distance_m: float
    surface_distance_m: float
    angular_radius_rad: float
    angular_rate_rad_s: float
    position_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    mode: str
    forbidden_for_control: bool = True
    surface_contact: bool = False
    contact_time_s: Optional[float] = None
    albedo: Optional[float] = None
    appearance_provenance: str = "configured"
    visual_only: bool = False
    non_interacting: bool = False
    counterfactual_contact_time_s: Optional[float] = None

    @property
    def id(self) -> str:
        return self.stimulus_id

    @property
    def looming(self) -> bool:
        return bool(self.angular_rate_rad_s > 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus_id": self.stimulus_id,
            "time_s": self.time_s,
            "active": self.active,
            "visible": self.visible,
            "colliding": self.colliding,
            "distance_m": self.distance_m,
            "surface_distance_m": self.surface_distance_m,
            "angular_radius_rad": self.angular_radius_rad,
            "angular_rate_rad_s": self.angular_rate_rad_s,
            "looming": self.looming,
            "position_m": list(self.position_m),
            "size_m": list(self.size_m),
            "mode": self.mode,
            "forbidden_for_control": True,
            "surface_contact": self.surface_contact,
            "contact_time_s": self.contact_time_s,
            "albedo": self.albedo,
            "appearance_provenance": self.appearance_provenance,
            "visual_only": self.visual_only,
            "non_interacting": self.non_interacting,
            "counterfactual_contact_time_s": self.counterfactual_contact_time_s,
        }


@dataclass
class OpticFlow:
    """Per-pixel static-scene flow from one pose to the next."""

    flow_px: list[list[tuple[float, float]]]
    valid: list[list[bool]]
    dt_s: float
    eye: str
    hfov_rad: float = math.radians(100.0)
    vfov_rad: float = math.radians(75.0)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.flow_px), len(self.flow_px[0]) if self.flow_px else 0, 2)

    @property
    def max_magnitude_px(self) -> float:
        return max((math.hypot(v[0], v[1]) for row, good in zip(self.flow_px, self.valid) for v, ok in zip(row, good) if ok), default=0.0)

    @property
    def flow_rad_s(self) -> list[list[tuple[float, float]]]:
        """Flow in angular SI units (radians/second) as well as pixel units."""
        h = len(self.flow_px)
        w = len(self.flow_px[0]) if h else 0
        if self.dt_s <= 0.0 or w <= 0 or h <= 0:
            return [[(0.0, 0.0) for _ in row] for row in self.flow_px]
        sx, sy = self.hfov_rad / w, self.vfov_rad / h
        return [[(v[0] * sx / self.dt_s, -v[1] * sy / self.dt_s) for v in row] for row in self.flow_px]

    def to_dict(self, *, include_flow: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"shape": list(self.shape), "dt_s": self.dt_s, "eye": self.eye,
                               "valid_count": sum(sum(1 for x in row if x) for row in self.valid),
                               "max_magnitude_px": self.max_magnitude_px}
        if include_flow:
            out["flow_px"] = [[[float(v[0]), float(v[1])] for v in row] for row in self.flow_px]
            out["flow_rad_s"] = [[[float(v[0]), float(v[1])] for v in row] for row in self.flow_rad_s]
            out["valid"] = self.valid
        return out


class GrayFrame(list):
    """A list-compatible grayscale frame with a NumPy-like ``shape`` view."""

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), len(self[0]) if self else 0)

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self]


@dataclass
class StereoFrames:
    left: GrayFrame
    right: GrayFrame
    timestamp_s: float = 0.0
    eye_separation_m: float = 0.006

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.left), len(self.left[0]) if self.left else 0)

    def __getitem__(self, key: str) -> Any:
        if key == "left":
            return self.left
        if key == "right":
            return self.right
        if key == "shape":
            return self.shape
        raise KeyError(key)

    def to_dict(self, *, include_pixels: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"shape": list(self.shape), "timestamp_s": self.timestamp_s, "eye_separation_m": self.eye_separation_m}
        if include_pixels:
            out["left"], out["right"] = self.left, self.right
        return out


@dataclass
class ScenarioResult:
    name: str
    seed: int
    times_s: list[float]
    poses: list[Pose]
    observations: list[TargetObservation]
    collisions: list[CollisionEvent]
    contacts: list[CollisionEvent]
    metrics: dict[str, Any]
    acceptance: dict[str, Any]
    manifest: dict[str, Any]
    frames: Optional[list[StereoFrames]] = None
    optic_flow: Optional[list[dict[str, OpticFlow]]] = None
    # Per-sample environmental input is kept beside the pose trajectory.  It
    # is optional at the type boundary so callers that constructed a result
    # with the pre-telemetry positional contract remain valid.
    wind_mps: list[tuple[float, float, float]] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)

    @property
    def collision_count(self) -> int:
        return len(self.collisions)

    @property
    def passed(self) -> bool:
        return bool(self.acceptance.get("passed", False))

    @property
    def final_pose(self) -> Pose:
        return self.poses[-1]

    @property
    def left_frames(self) -> list[list[list[float]]]:
        return [] if self.frames is None else [f.left for f in self.frames]

    @property
    def right_frames(self) -> list[list[list[float]]]:
        return [] if self.frames is None else [f.right for f in self.frames]

    @property
    def flow_ground_truth(self) -> Optional[list[dict[str, OpticFlow]]]:
        return self.optic_flow

    @property
    def wind_samples_mps(self) -> list[tuple[float, float, float]]:
        """Wind sampled at each pose in ``times_s`` order."""
        return self.wind_mps

    @property
    def wind(self) -> list[tuple[float, float, float]]:
        """Short compatibility alias for per-sample wind telemetry."""
        return self.wind_mps

    @property
    def target_observations(self) -> list[TargetObservation]:
        """Explicit alias for the time-aligned target sensor stream."""
        return self.observations

    @property
    def phase_labels(self) -> tuple[str, ...]:
        """Distinct reference-phase labels in trajectory order."""
        labels: list[str] = []
        for row in self.trajectory:
            label = str(row.get("phase", ""))
            if label and (not labels or labels[-1] != label):
                labels.append(label)
        return tuple(labels)

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        """JSON-friendly pose/target/wind records for video and validation.

        Older callers use ``poses`` and ``observations`` directly.  The
        records here deliberately repeat those values in one row so a video
        writer or validator does not have to reconstruct time alignment.
        """
        if self.telemetry:
            return self.telemetry
        return [
            {
                "time_s": float(t),
                "position_m": list(p.position_m),
                "orientation_rad": list(p.orientation_rpy_rad),
                "velocity_mps": list(p.velocity_mps),
                "angular_velocity_rad_s": list(p.angular_velocity_rad_s),
                "target": o.to_dict(),
                "wind_mps": list(w),
            }
            for t, p, o, w in zip(self.times_s, self.poses, self.observations, self.wind_mps)
        ]

    def to_dict(self, *, include_frames: bool = False, include_flow: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": "virtual-world-result-1",
            "name": self.name,
            "seed": self.seed,
            "metadata": {
                "evidence_class": "digital_simulation",
                "source": "virtual_world_raycast",
                "scenario": self.name,
            },
            "times_s": list(self.times_s),
            "poses": [p.to_dict() for p in self.poses],
            "observations": [o.to_dict() for o in self.observations],
            "wind_mps": [list(w) for w in self.wind_mps],
            "trajectory": _jsonable(self.trajectory),
            "phases": _jsonable(self.metrics.get("phase_timings", {})),
            "collisions": [c.to_dict() for c in self.collisions],
            "contacts": [c.to_dict() for c in self.contacts],
            "metrics": _jsonable(self.metrics),
            "acceptance": _jsonable(self.acceptance),
            "manifest": _jsonable(self.manifest),
        }
        if include_frames and self.frames is not None:
            out["frames"] = [f.to_dict(include_pixels=True) for f in self.frames]
        if include_flow and self.optic_flow is not None:
            out["optic_flow"] = [{eye: flow.to_dict(include_flow=True) for eye, flow in pair.items()} for pair in self.optic_flow]
        return out


@dataclass(frozen=True)
class CameraConfig:
    """Pinhole camera settings shared by both eyes."""

    # Small defaults keep a headless smoke run quick; callers can request any
    # deterministic resolution without changing the scene or trajectory.
    width: int = 32
    height: int = 24
    hfov_rad: float = math.radians(100.0)
    vfov_rad: Optional[float] = None
    eye_separation_m: float = 0.006
    near_m: float = 0.01
    far_m: float = 8.0

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("camera width and height must be positive")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        hfov = _finite(self.hfov_rad, "hfov_rad")
        if not 0.0 < hfov < math.pi:
            raise ValueError("hfov_rad must be in (0, pi)")
        object.__setattr__(self, "hfov_rad", hfov)
        vfov = self.vfov_rad
        if vfov is None:
            # Preserve the requested horizontal field of view while respecting
            # the pixel aspect ratio.
            vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * self.height / self.width)
        vfov = _finite(vfov, "vfov_rad")
        if not 0.0 < vfov < math.pi:
            raise ValueError("vfov_rad must be in (0, pi)")
        object.__setattr__(self, "vfov_rad", vfov)
        object.__setattr__(self, "eye_separation_m", max(0.0, _finite(self.eye_separation_m, "eye_separation_m")))
        object.__setattr__(self, "near_m", max(1e-5, _finite(self.near_m, "near_m")))
        object.__setattr__(self, "far_m", max(self.near_m, _finite(self.far_m, "far_m")))

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "hfov_rad": self.hfov_rad,
                "vfov_rad": self.vfov_rad, "eye_separation_m": self.eye_separation_m,
                "near_m": self.near_m, "far_m": self.far_m}


@dataclass(frozen=True)
class _RayHit:
    object_id: str
    t: float
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    texture: str
    kind: str
    albedo: Optional[float] = None


@dataclass(frozen=True)
class _PreparedRender:
    """Per-frame geometry and camera terms shared by every ray.

    The reference renderer resolved this data inside ``_trace`` for each
    pixel.  Keeping the same immutable boxes and scalar ray operations, but
    resolving them once per eye/frame, avoids repeated dataclass validation
    and stimulus sampling without changing hit ordering or shader arithmetic.
    """

    camera: CameraConfig
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    origin: tuple[float, float, float]
    stimulus_boxes: tuple["_PreparedStimulus", ...]


@dataclass(frozen=True)
class _PreparedStimulus:
    """A timestamped board plus basis terms reused by every ray."""

    geometry: OrientedBox
    basis: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    half_size: tuple[float, float, float]
    mesh_accel: "_StimulusMeshAccel | None" = None


@dataclass(frozen=True)
class _SceneTriangle:
    """One immutable visual triangle imported from an optional scene asset."""

    a: tuple[float, float, float]
    b: tuple[float, float, float]
    c: tuple[float, float, float]
    object_id: str
    material_id: str | None
    albedo: float
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    centroid: tuple[float, float, float]


@dataclass(frozen=True)
class _TriangleBVHNode:
    """Small CPU reference BVH over scene triangles."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    left: "_TriangleBVHNode | None" = None
    right: "_TriangleBVHNode | None" = None
    triangle_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _StimulusMeshAccel:
    """Immutable local-space triangle data and BVH for one bound stimulus."""

    scene_mesh: Any
    triangles: tuple[_SceneTriangle, ...]
    bvh: _TriangleBVHNode | None


def _build_triangle_bvh(triangles: Sequence[_SceneTriangle], indices: Sequence[int] | None = None,
                        leaf_size: int = 8) -> _TriangleBVHNode | None:
    if not triangles:
        return None
    items = list(range(len(triangles))) if indices is None else list(indices)
    lo = tuple(min(triangles[i].lo[axis] for i in items) for axis in range(3))
    hi = tuple(max(triangles[i].hi[axis] for i in items) for axis in range(3))
    if len(items) <= leaf_size:
        return _TriangleBVHNode(lo, hi, triangle_indices=tuple(items))
    center_lo = tuple(min(triangles[i].centroid[axis] for i in items) for axis in range(3))
    center_hi = tuple(max(triangles[i].centroid[axis] for i in items) for axis in range(3))
    axis = max(range(3), key=lambda k: (center_hi[k] - center_lo[k], -k))
    if center_hi[axis] - center_lo[axis] < _EPS:
        return _TriangleBVHNode(lo, hi, triangle_indices=tuple(items))
    items.sort(key=lambda i: (triangles[i].centroid[axis], i))
    middle = len(items) // 2
    return _TriangleBVHNode(lo, hi,
                            left=_build_triangle_bvh(triangles, items[:middle], leaf_size),
                            right=_build_triangle_bvh(triangles, items[middle:], leaf_size))


def _ray_bounds_near(origin: Sequence[float], direction: Sequence[float], lo: Sequence[float],
                     hi: Sequence[float], near: float, far: float) -> float | None:
    """Return the first ray/AABB parameter, or ``None`` when it misses."""
    enter, leave = near, far
    for axis in range(3):
        d = direction[axis]
        if abs(d) < _EPS:
            if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                return None
            continue
        t0 = (lo[axis] - origin[axis]) / d
        t1 = (hi[axis] - origin[axis]) / d
        if t0 > t1:
            t0, t1 = t1, t0
        enter = max(enter, t0)
        leave = min(leave, t1)
        if leave < enter:
            return None
    return enter


def _ray_triangle(origin: Sequence[float], direction: Sequence[float], triangle: _SceneTriangle,
                  near: float, far: float) -> tuple[float, tuple[float, float, float]] | None:
    """Two-sided Möller–Trumbore intersection used by the CPU reference."""
    edge1 = _sub(triangle.b, triangle.a)
    edge2 = _sub(triangle.c, triangle.a)
    pvec = _cross(direction, edge2)
    det = _dot(edge1, pvec)
    if abs(det) < 1.0e-12:
        return None
    inv_det = 1.0 / det
    tvec = _sub(origin, triangle.a)
    u = _dot(tvec, pvec) * inv_det
    if u < -1.0e-10 or u > 1.0 + 1.0e-10:
        return None
    qvec = _cross(tvec, edge1)
    v = _dot(direction, qvec) * inv_det
    if v < -1.0e-10 or u + v > 1.0 + 1.0e-10:
        return None
    t = _dot(edge2, qvec) * inv_det
    if t < near or t > far:
        return None
    # Scene triangles are two-sided: report a ray-facing normal so the public
    # Hit record and Lambert shader do not depend on glTF winding when viewed
    # from the back side.
    normal = _unit(_cross(edge1, edge2), (0.0, 0.0, 1.0))
    if _dot(normal, direction) > 0.0:
        normal = _scale(normal, -1.0)
    return t, normal


def _ray_box(origin: Sequence[float], direction: Sequence[float], lo: Sequence[float], hi: Sequence[float], near: float, far: float) -> tuple[float, tuple[float, float, float]] | None:
    """Return the first ray/AABB hit and an outward face normal."""
    t_min, t_max = near, far
    enter_normal = (0.0, 0.0, 0.0)
    exit_normal = (0.0, 0.0, 0.0)
    for axis in range(3):
        o, d, a, b = origin[axis], direction[axis], lo[axis], hi[axis]
        if abs(d) < _EPS:
            if o < a or o > b:
                return None
            continue
        t0, t1 = (a - o) / d, (b - o) / d
        n0 = [0.0, 0.0, 0.0]
        n1 = [0.0, 0.0, 0.0]
        n0[axis], n1[axis] = (-1.0 if d > 0.0 else 1.0), (1.0 if d > 0.0 else -1.0)
        if t0 > t1:
            t0, t1, n0, n1 = t1, t0, n1, n0
        if t0 > t_min:
            t_min, enter_normal = t0, tuple(n0)
        if t1 < t_max:
            t_max, exit_normal = t1, tuple(n1)
        if t_min > t_max:
            return None
    if t_min >= near + _EPS:
        return t_min, enter_normal
    if t_max >= near + _EPS:
        # The origin can be inside a box (for example, a probe started at a
        # wall).  Returning the exit face is more useful than a zero-distance
        # self-hit.
        return t_max, exit_normal
    return None


def _ray_oriented_box(origin: Sequence[float], direction: Sequence[float], box: OrientedBox,
                      near: float, far: float, *,
                      basis: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None,
                      half_size: tuple[float, float, float] | None = None) -> tuple[float, tuple[float, float, float], tuple[float, float, float]] | None:
    """Intersect a world ray with an oriented rectangular board."""
    if basis is None:
        basis = box.basis
    local_origin = _local_point(origin, box.center_m, basis)
    local_direction = tuple(_dot(direction, axis) for axis in basis)
    half = _scale(box.size_m, 0.5) if half_size is None else half_size
    result = _ray_box(local_origin, local_direction, _scale(half, -1.0), half, near, far)
    if result is None:
        return None
    distance, local_normal = result
    local_point = _add(local_origin, _scale(local_direction, distance))
    world_point = _world_point(local_point, box.center_m, basis)
    world_normal = _unit(_world_point(local_normal, (0.0, 0.0, 0.0), basis), (1.0, 0.0, 0.0))
    return distance, world_point, world_normal


class VirtualWorld:
    """Deterministic analytic-reference scene, ray caster, and telemetry.

    This class is the scalar/golden backend for the digital-twin contract;
    its geometry and pixel values are intentionally stable and auditable.
    A future accelerated backend (for example ``warp_cuda``) should implement
    the same public render/trace seam and be selected by the caller rather than
    turning this reference world into a general-purpose renderer.
    """

    backend_name = "virtual_world_raycast"
    backend_role = "analytic_reference"
    backend_contract = "world-renderer-v1"
    is_stub = True

    # TODO(warp_cuda): add an external backend adapter implementing this
    # contract.  Keep the analytic reference path as the equivalence oracle;
    # do not replace it with a device-specific implementation in this module.

    def __init__(self, config: WorldConfig, scenarios: Mapping[str, ScenarioConfig] | None = None,
                 *, seed: Optional[int] = None, camera: CameraConfig | None = None,
                 source_path: str | Path | None = None,
                 scene_path: str | Path | None = None,
                 stimuli: Mapping[str, PhysicalStimulus] | Sequence[PhysicalStimulus] | PhysicalStimulus | None = None):
        if isinstance(config, Mapping):
            config = WorldConfig.from_mapping(config, seed=1729 if seed is None else int(seed))
        if not isinstance(config, WorldConfig):
            raise TypeError("config must be a WorldConfig")
        self.config = config
        self.seed = config.seed if seed is None else int(seed)
        clean_scenarios: dict[str, ScenarioConfig] = {}
        for name, item in (scenarios or {}).items():
            clean_scenarios[str(name)] = item if isinstance(item, ScenarioConfig) else ScenarioConfig.from_mapping(str(name), item)
        self.scenarios = clean_scenarios
        self.camera = camera or CameraConfig()
        self.source_path = None if source_path is None else str(source_path)
        self.scene_path = None if scene_path is None else str(Path(scene_path).expanduser().resolve())
        self._scene_manifest: Any = None
        self._scene_triangles: tuple[_SceneTriangle, ...] = ()
        self._scene_bvh: _TriangleBVHNode | None = None
        self._stimulus_visual_meshes: dict[str, Any] = {}
        self._stimulus_mesh_accels: dict[str, _StimulusMeshAccel] = {}
        if self.scene_path is not None:
            self._load_external_scene(self.scene_path)
        self.stimuli = self._normalise_stimuli(stimuli)
        self._wall_boxes = self._make_wall_boxes()
        # These scene terms are immutable for the lifetime of a world.  The
        # old per-ray path rebuilt the target AABB and normalised the light
        # vector for every hit, so retain the exact scalar values once.
        self._target_box = self.config.target.box
        self._ray_boxes = (
            tuple((box, "wall") for box in self._wall_boxes)
            + ((self._target_box, "target"),)
        )
        self._light_direction_unit = _unit(self.config.light.direction_m)
        self._texture_phase_cache: dict[tuple[int, str], float] = {}
        self._state: Any = None

    @property
    def scenario_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.scenarios))

    @property
    def target(self) -> Target:
        return self.config.target

    @property
    def obstacles(self) -> tuple[AABB, ...]:
        return self.config.obstacles

    @property
    def scene_triangles(self) -> tuple[_SceneTriangle, ...]:
        """Normalized visual triangles shared with optional renderer backends."""
        return self._scene_triangles

    @property
    def stimulus_visual_meshes(self) -> Mapping[str, Any]:
        """Bound local meshes keyed by physical stimulus ID for render backends.

        The values are the loader's :class:`SceneMesh` instances. Their
        ``vertices_m`` and ``triangles`` stay in the asset's local stimulus
        frame; a backend poses them from the sampled ``StimulusState``. The
        mapping never includes an asset that remains part of static geometry.
        """
        return MappingProxyType(self._stimulus_visual_meshes)

    @property
    def external_scene_manifest(self) -> dict[str, Any] | None:
        """Stable provenance for an attached visual-only scene, if configured."""
        if self.scene_path is None:
            return None
        binding_records = {}
        for stimulus_id, mesh in sorted(self._stimulus_visual_meshes.items()):
            accel = self._stimulus_mesh_accels[stimulus_id]
            source_path = getattr(mesh, "source_path", None)
            binding_records[stimulus_id] = {
                "object_id": str(getattr(mesh, "object_id", "")),
                "role": str(getattr(mesh, "role", "")),
                "source_path": None if source_path is None else str(source_path),
                "source_sha256": getattr(mesh, "source_sha256", None),
                "source_files_sha256": _jsonable(getattr(mesh, "source_files_sha256", {})),
                "provenance": _jsonable(getattr(mesh, "provenance", {})),
                "texture_metadata": _jsonable(getattr(mesh, "texture_metadata", {})),
                "vertex_count": len(getattr(mesh, "vertices_m", ())),
                "triangle_count": len(accel.triangles),
                "pose_frame": "local +X normal, +Y width, +Z up; rigidly posed from PhysicalStimulus state",
            }
        return {
            "manifest_path": self.scene_path,
            "visual_object_count": len({triangle.object_id for triangle in self._scene_triangles}),
            "triangle_count": len(self._scene_triangles),
            "stimulus_visual_bindings": binding_records,
            "triangle_sidedness": "two_sided",
            "normal_convention": "hit normals point against the incoming ray",
            "collision_integration": False,
            "loader_manifest": _jsonable(self._scene_manifest),
        }

    def _load_external_scene(self, scene_path: str) -> None:
        """Load normalized triangle geometry without changing flight collisions."""
        try:
            from gltf_mesh_loader import load_scene_assets  # type: ignore
        except ImportError:
            try:
                from .gltf_mesh_loader import load_scene_assets  # type: ignore
            except ImportError as exc:
                raise RuntimeError("external scene loading requires scripts/gltf_mesh_loader.py") from exc
        loaded = load_scene_assets(scene_path)
        scene_manifest = getattr(loaded, "manifest", {})
        if not isinstance(scene_manifest, Mapping):
            raise ValueError("loaded scene manifest must be an object")
        simulation = scene_manifest.get("simulation", {})
        if not isinstance(simulation, Mapping):
            raise ValueError("simulation must be an object")
        bindings = simulation.get("stimulus_object_ids", {})
        if not isinstance(bindings, Mapping):
            raise ValueError("simulation.stimulus_object_ids must be an object")
        materials_out: list[_SceneTriangle] = []
        objects = tuple(getattr(loaded, "objects", ()))
        objects_by_id: dict[str, Any] = {}
        for object_index, obj in enumerate(objects):
            object_id = getattr(obj, "object_id", None)
            if not isinstance(object_id, str) or not object_id.strip():
                raise ValueError(f"scene object {object_index} must have a non-empty object_id")
            if object_id in objects_by_id:
                raise ValueError(f"duplicate loaded scene object_id: {object_id}")
            objects_by_id[object_id] = obj

        bound_object_ids: set[str] = set()
        for stimulus_id, object_id in bindings.items():
            if not isinstance(stimulus_id, str) or not stimulus_id.strip():
                raise ValueError("simulation.stimulus_object_ids keys must be non-empty stimulus IDs")
            if not isinstance(object_id, str) or not object_id.strip():
                raise ValueError(f"simulation.stimulus_object_ids[{stimulus_id!r}] must be a non-empty object ID")
            obj = objects_by_id.get(object_id)
            if obj is None:
                raise ValueError(
                    f"stimulus {stimulus_id!r} is bound to missing scene object_id {object_id!r}"
                )
            role = str(getattr(obj, "role", "")).lower()
            if role not in {"visual", "visual_collision"}:
                raise ValueError(
                    f"stimulus {stimulus_id!r} must bind to a visual scene asset; "
                    f"{object_id!r} has role {role!r}"
                )
            accel = self._prepare_stimulus_mesh_accel(obj, stimulus_id)
            self._stimulus_visual_meshes[stimulus_id] = obj
            self._stimulus_mesh_accels[stimulus_id] = accel
            bound_object_ids.add(object_id)

        for object_index, obj in enumerate(objects):
            role = str(getattr(obj, "role", "visual")).lower()
            if role not in {"visual", "visual_collision"}:
                continue
            object_id = str(getattr(obj, "object_id", f"scene_object_{object_index}"))
            if object_id in bound_object_ids:
                # Bound meshes move with their physical stimulus and are
                # queried in local coordinates. They must never also shade as
                # a static scene object at the asset origin.
                continue
            try:
                vertices_raw = tuple(getattr(obj, "vertices_m"))
                triangles_raw = tuple(getattr(obj, "triangles"))
            except (TypeError, AttributeError) as exc:
                raise ValueError(f"scene object {object_id!r} must provide vertices_m and triangles") from exc
            vertices = tuple(_vec3(vertex, f"{object_id}.vertices_m[{index}]")
                             for index, vertex in enumerate(vertices_raw))
            triangle_material_ids = tuple(getattr(obj, "triangle_material_ids", ()))
            if triangle_material_ids and len(triangle_material_ids) != len(triangles_raw):
                raise ValueError(f"scene object {object_id!r} has mismatched triangle material IDs")
            material_table = getattr(obj, "materials", {}) or {}
            for triangle_index, raw_indices in enumerate(triangles_raw):
                try:
                    indices = tuple(int(value) for value in raw_indices)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{object_id}.triangles[{triangle_index}] must contain three indices") from exc
                if len(indices) != 3 or any(index < 0 or index >= len(vertices) for index in indices):
                    raise ValueError(f"{object_id}.triangles[{triangle_index}] has an invalid vertex index")
                a, b, c = (vertices[index] for index in indices)
                raw_material_id = triangle_material_ids[triangle_index] if triangle_material_ids else None
                material_id = None if raw_material_id is None else str(raw_material_id)
                material = material_table.get(raw_material_id, material_table.get(material_id, {})) if isinstance(material_table, Mapping) else {}
                rgba = material.get("base_color_rgba", material.get("base_color_factor", (0.68, 0.68, 0.68, 1.0))) if isinstance(material, Mapping) else (0.68, 0.68, 0.68, 1.0)
                try:
                    color = tuple(_clamp(_finite(component, f"{object_id}.material.base_color_rgba")) for component in rgba)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"scene object {object_id!r} has invalid material color") from exc
                if len(color) < 3:
                    raise ValueError(f"scene object {object_id!r} material color must have at least three channels")
                albedo = _clamp(0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])
                lo = tuple(min(a[axis], b[axis], c[axis]) for axis in range(3))
                hi = tuple(max(a[axis], b[axis], c[axis]) for axis in range(3))
                centroid = tuple((a[axis] + b[axis] + c[axis]) / 3.0 for axis in range(3))
                materials_out.append(_SceneTriangle(a, b, c, object_id, material_id, albedo,
                                                    lo, hi, centroid))
        self._scene_triangles = tuple(materials_out)
        self._scene_bvh = _build_triangle_bvh(self._scene_triangles)
        self._scene_manifest = scene_manifest

    @staticmethod
    def _prepare_stimulus_mesh_accel(obj: Any, stimulus_id: str) -> _StimulusMeshAccel:
        """Validate one bound local mesh and build its local BVH once."""
        object_id = str(getattr(obj, "object_id", ""))
        try:
            vertices_raw = tuple(getattr(obj, "vertices_m"))
            triangles_raw = tuple(getattr(obj, "triangles"))
        except (TypeError, AttributeError) as exc:
            raise ValueError(
                f"bound scene object {object_id!r} must provide vertices_m and triangles"
            ) from exc
        if len(vertices_raw) < 3 or not triangles_raw:
            raise ValueError(f"bound scene object {object_id!r} must contain vertices and triangles")
        vertices = tuple(
            _vec3(vertex, f"{object_id}.vertices_m[{index}]")
            for index, vertex in enumerate(vertices_raw)
        )
        triangle_material_ids = tuple(getattr(obj, "triangle_material_ids", ()))
        if triangle_material_ids and len(triangle_material_ids) != len(triangles_raw):
            raise ValueError(f"bound scene object {object_id!r} has mismatched triangle material IDs")
        triangle_albedos_raw = getattr(obj, "triangle_albedos", None)
        if triangle_albedos_raw is None:
            triangle_albedos: tuple[tuple[float, float, float], ...] | None = None
        else:
            try:
                triangle_albedos = tuple(
                    _vec3(row, f"{object_id}.triangle_albedos[{index}]")
                    for index, row in enumerate(triangle_albedos_raw)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bound scene object {object_id!r} has malformed triangle_albedos") from exc
            if len(triangle_albedos) != len(triangles_raw):
                raise ValueError(f"bound scene object {object_id!r} has mismatched triangle albedos")
            if any(channel < 0.0 or channel > 1.0 for rgb in triangle_albedos for channel in rgb):
                raise ValueError(f"bound scene object {object_id!r} triangle albedos must be in [0, 1]")

        material_table = getattr(obj, "materials", {}) or {}
        prepared_triangles: list[_SceneTriangle] = []
        for triangle_index, raw_indices in enumerate(triangles_raw):
            try:
                values = tuple(raw_indices)
            except TypeError as exc:
                raise ValueError(f"{object_id}.triangles[{triangle_index}] must contain three indices") from exc
            if len(values) != 3:
                raise ValueError(f"{object_id}.triangles[{triangle_index}] must contain three indices")
            indices: list[int] = []
            for local_index, raw_value in enumerate(values):
                if isinstance(raw_value, bool):
                    raise ValueError(f"{object_id}.triangles[{triangle_index}] index {local_index} is invalid")
                number = _finite(raw_value, f"{object_id}.triangles[{triangle_index}][{local_index}]")
                if not number.is_integer():
                    raise ValueError(f"{object_id}.triangles[{triangle_index}] indices must be integers")
                index = int(number)
                if index < 0 or index >= len(vertices):
                    raise ValueError(f"{object_id}.triangles[{triangle_index}] has an invalid vertex index")
                indices.append(index)
            a, b, c = (vertices[index] for index in indices)
            raw_material_id = triangle_material_ids[triangle_index] if triangle_material_ids else None
            material_id = None if raw_material_id is None else str(raw_material_id)
            if triangle_albedos is not None:
                rgb = triangle_albedos[triangle_index]
            else:
                material = (material_table.get(raw_material_id, material_table.get(material_id, {}))
                            if isinstance(material_table, Mapping) else {})
                rgba = (material.get("base_color_rgba", material.get("base_color_factor", (0.68, 0.68, 0.68, 1.0)))
                        if isinstance(material, Mapping) else (0.68, 0.68, 0.68, 1.0))
                try:
                    color = tuple(_clamp(_finite(channel, f"{object_id}.material.base_color_rgba"))
                                  for channel in rgba)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"bound scene object {object_id!r} has invalid material color") from exc
                if len(color) < 3:
                    raise ValueError(f"bound scene object {object_id!r} material color needs RGB channels")
                rgb = (color[0], color[1], color[2])
            albedo = _clamp(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
            lo = tuple(min(a[axis], b[axis], c[axis]) for axis in range(3))
            hi = tuple(max(a[axis], b[axis], c[axis]) for axis in range(3))
            centroid = tuple((a[axis] + b[axis] + c[axis]) / 3.0 for axis in range(3))
            prepared_triangles.append(_SceneTriangle(a, b, c, stimulus_id, material_id,
                                                     albedo, lo, hi, centroid))
        frozen_triangles = tuple(prepared_triangles)
        bvh = _build_triangle_bvh(frozen_triangles)
        if bvh is None:
            raise ValueError(f"bound scene object {object_id!r} has no triangulated mesh")
        return _StimulusMeshAccel(obj, frozen_triangles, bvh)

    @property
    def stimulus_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.stimuli))

    def create_renderer(self, backend: str = "analytic_reference", *,
                        device: str = "cuda:0", **kwargs: Any) -> Any:
        """Select the analytic reference or an explicit optional renderer.

        The default returns this world unchanged.  ``warp_cuda`` is imported
        lazily and fails closed when Warp/CUDA is unavailable; callers that
        want an analytic fallback must request ``analytic_reference``.
        """
        requested = str(backend).strip().lower()
        if requested in {"analytic_reference", "virtual_world_raycast"}:
            return self
        if requested != "warp_cuda":
            raise ValueError(f"unsupported renderer backend {backend!r}")
        try:
            from world_renderer_opt import create_renderer  # type: ignore
        except ImportError:
            from .world_renderer_opt import create_renderer  # type: ignore
        return create_renderer(self, backend=requested, device=device, **kwargs)

    renderer = create_renderer

    @staticmethod
    def _normalise_stimuli(value: Mapping[str, PhysicalStimulus] | Sequence[PhysicalStimulus] | PhysicalStimulus | None) -> dict[str, PhysicalStimulus]:
        if value is None:
            return {}
        if isinstance(value, PhysicalStimulus):
            return {value.id: value}
        if isinstance(value, Mapping):
            # A single mapping with physical fields is accepted as one item;
            # otherwise mapping keys are protocol names.
            if any(key in value for key in ("start_position_m", "start_point_m", "start_m", "start", "position_m", "velocity_mps", "speed_mps", "speed", "size_m")):
                item = PhysicalStimulus.from_mapping(value)
                return {item.id: item}
            out: dict[str, PhysicalStimulus] = {}
            for name, item in value.items():
                stim = item if isinstance(item, PhysicalStimulus) else PhysicalStimulus.from_mapping(item)
                out[str(name)] = stim
            return out
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            out = {}
            for item in value:
                stim = item if isinstance(item, PhysicalStimulus) else PhysicalStimulus.from_mapping(item)
                out[stim.id] = stim
            return out
        raise TypeError("stimuli must be a stimulus, mapping, or sequence")

    def _resolve_stimuli(self, stimulus: Any = None, stimuli: Any = None) -> tuple[PhysicalStimulus, ...]:
        value = stimulus if stimulus is not None else stimuli
        if value is None:
            return tuple(self.stimuli[name] for name in sorted(self.stimuli))
        if isinstance(value, str):
            if value not in self.stimuli:
                raise KeyError(f"unknown stimulus {value!r}; available: {sorted(self.stimuli)}")
            return (self.stimuli[value],)
        return tuple(self._normalise_stimuli(value).values())

    def stimulus(self, name: str) -> PhysicalStimulus:
        """Return a named stimulus loaded on this world."""
        key = str(name)
        if key not in self.stimuli:
            raise KeyError(f"unknown stimulus {name!r}; available: {sorted(self.stimuli)}")
        return self.stimuli[key]

    def stimulus_state_at(self, t_s: float, stimulus: Any = None, *, stimuli: Any = None,
                          reference_pose: Any = None) -> dict[str, dict[str, Any]]:
        """Return physical stimulus states for logging/acceptance only.

        Every record carries ``forbidden_for_control=true``.  The flag is a
        contract boundary: callers may archive or validate these geometry
        states, but should not feed them as a hidden behavioural command.
        """
        return {item.id: item.stimulus_state_at(t_s, reference_pose) for item in self._resolve_stimuli(stimulus, stimuli)}

    @property
    def walls(self) -> tuple[AABB, ...]:
        return self._wall_boxes

    @classmethod
    def from_json(cls, path: str | Path | Mapping[str, Any] = _DEFAULT_CONFIG, *, camera: CameraConfig | None = None,
                  stimuli: Mapping[str, PhysicalStimulus] | Sequence[PhysicalStimulus] | PhysicalStimulus | str | Path | None = None,
                  stimulus_path: str | Path | None = None,
                  scene_path: str | Path | None = None) -> "VirtualWorld":
        if isinstance(path, Mapping):
            return cls.from_mapping(path, camera=camera, stimuli=stimuli, scene_path=scene_path)
        source = Path(path)
        with source.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, Mapping):
            raise ValueError("scenario config root must be an object")
        seed = int(raw.get("seed", 1729))
        config = WorldConfig.from_mapping(raw.get("world", {}), seed=seed)
        scenario_map = raw.get("scenarios", {})
        if not isinstance(scenario_map, Mapping):
            raise ValueError("scenarios must be an object")
        scenarios = {str(name): ScenarioConfig.from_mapping(str(name), item) for name, item in scenario_map.items()}
        loaded_stimuli: Any = stimuli
        if stimulus_path is not None:
            loaded_stimuli = load_stimuli(stimulus_path, seed=seed)
        elif isinstance(stimuli, (str, Path)):
            loaded_stimuli = load_stimuli(stimuli, seed=seed)
        return cls(config, scenarios, seed=seed, camera=camera, source_path=source,
                   scene_path=scene_path, stimuli=loaded_stimuli)

    @classmethod
    def from_config(cls, path: str | Path = _DEFAULT_CONFIG, **kwargs: Any) -> "VirtualWorld":
        return cls.from_json(path, **kwargs)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, camera: CameraConfig | None = None,
                     stimuli: Mapping[str, PhysicalStimulus] | Sequence[PhysicalStimulus] | PhysicalStimulus | str | Path | None = None,
                     scene_path: str | Path | None = None) -> "VirtualWorld":
        """Load the same schema as JSON from an in-memory mapping."""
        if not isinstance(value, Mapping):
            raise ValueError("world mapping must be an object")
        seed = int(value.get("seed", 1729))
        config = WorldConfig.from_mapping(value.get("world", value), seed=seed)
        scenario_map = value.get("scenarios", {})
        scenarios = {str(name): ScenarioConfig.from_mapping(str(name), item) for name, item in scenario_map.items()}
        loaded_stimuli: Any = stimuli
        if isinstance(stimuli, (str, Path)):
            loaded_stimuli = load_stimuli(stimuli, seed=seed)
        return cls(config, scenarios, seed=seed, camera=camera, scene_path=scene_path, stimuli=loaded_stimuli)

    from_dict = from_mapping

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_CONFIG, **kwargs: Any) -> "VirtualWorld":
        return cls.from_json(path, **kwargs)

    def _make_wall_boxes(self) -> tuple[AABB, ...]:
        xmin, xmax, ymin, ymax = self.config.bounds_xy_m
        # A small thickness gives walls proper ray-caster faces while the
        # collision plane remains at the nominal world boundary.
        thick = 0.04
        z0, z1 = self.config.ground_z_m, self.config.ground_z_m + self.config.wall_height_m
        return (
            AABB("wall_xmin", (xmin - thick, ymin - thick, z0), (xmin, ymax + thick, z1), "wall"),
            AABB("wall_xmax", (xmax, ymin - thick, z0), (xmax + thick, ymax + thick, z1), "wall"),
            AABB("wall_ymin", (xmin, ymin - thick, z0), (xmax, ymin, z1), "wall"),
            AABB("wall_ymax", (xmin, ymax, z0), (xmax, ymax + thick, z1), "wall"),
        )

    def _resolve_camera(self, resolution: Any = None, *, width: int | None = None, height: int | None = None) -> CameraConfig:
        if resolution is None:
            w, h = self.camera.width, self.camera.height
        else:
            try:
                w, h = tuple(resolution)
            except (TypeError, ValueError) as exc:
                raise ValueError("resolution must be (width, height)") from exc
        if width is not None:
            w = width
        if height is not None:
            h = height
        return CameraConfig(int(w), int(h), self.camera.hfov_rad, self.camera.vfov_rad,
                            self.camera.eye_separation_m, self.camera.near_m, self.camera.far_m)

    def manifest(self, scenario: str | ScenarioConfig | None = None, *, resolution: Any = None,
                 stimulus: Any = None, stimuli: Any = None) -> dict[str, Any]:
        """Return a stable, JSON-serialisable description of the experiment."""
        camera = self._resolve_camera(resolution)
        selected_stimuli = self._resolve_stimuli(stimulus, stimuli)
        stimulus_payload = {item.id: item.to_dict() for item in selected_stimuli}
        config_payload = {
            "schema_version": "virtual-world-1",
            "units": "SI",
            "seed": self.seed,
            "world": self.config.to_dict(),
            "scenarios": {name: sc.to_dict() for name, sc in sorted(self.scenarios.items())},
            "stimuli": stimulus_payload,
        }
        external_scene = self.external_scene_manifest
        if external_scene is not None:
            config_payload["external_scene"] = external_scene
        canonical = json.dumps(config_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        selected: str | None
        if isinstance(scenario, ScenarioConfig):
            selected = scenario.name
        else:
            selected = None if scenario is None else str(scenario)
        return {
            "schema_version": "virtual-world-1",
            "manifest_version": "1",
            "units": "SI",
            "deterministic": True,
            "engine": "python-raycast",
            "seed": self.seed,
            "config_sha256": hashlib.sha256(canonical).hexdigest(),
            "source_config": self.source_path,
            "selected_scenario": selected,
            "config": config_payload,
            "world": self.config.to_dict(),
            "camera": camera.to_dict(),
            "stimuli": stimulus_payload,
            "geometry": {
                "ground": {"z_m": self.config.ground_z_m, "finite": True},
                "walls": {"bounds_xy_m": list(self.config.bounds_xy_m), "height_m": self.config.wall_height_m},
                "landing_pad": self.config.target.to_dict(),
                "ground_z_m": self.config.ground_z_m,
                "bounds_xy_m": list(self.config.bounds_xy_m),
                "wall_height_m": self.config.wall_height_m,
                "target": self.config.target.to_dict(),
                "obstacles": [o.to_dict() for o in self.config.obstacles],
            },
            "looming": {"enabled": True, "cue": "angular_radius_and_tau", "obstacle_ids": [o.id for o in self.config.obstacles]},
            "textures": {
                "analytic": True,
                "band_limited": True,
                "ground_cell_m": self.config.texture.ground_cell_m,
                "wall_cell_m": self.config.texture.wall_cell_m,
                "contrast": self.config.texture.contrast,
                "max_spatial_frequency_cycles_per_m": 1.0 / min(self.config.texture.ground_cell_m, self.config.texture.wall_cell_m),
                "objects": {"ground": "checker", "walls": "wall", "landing_pad": self.config.target.texture,
                            **{o.id: o.texture for o in self.config.obstacles}},
            },
            "lighting": self.config.light.to_dict(),
            "wind": self.config.wind.to_dict(),
            "scenarios": {
                name: {
                    "duration_s": sc.duration_s,
                    "dt_s": sc.dt_s,
                    "profile": dict(sc.profile),
                    "acceptance": dict(sc.acceptance),
                    "wind": self.config.wind.with_override(sc.wind_override).to_dict(),
                    "phases": self._phase_manifest(sc),
                }
                for name, sc in sorted(self.scenarios.items())
            },
            **({"external_scene": external_scene} if external_scene is not None else {}),
        }

    def manifest_json(self, scenario: str | ScenarioConfig | None = None, *, resolution: Any = None,
                      stimulus: Any = None, stimuli: Any = None, **kwargs: Any) -> str:
        options = {"sort_keys": True, "indent": 2, "allow_nan": False}
        options.update(kwargs)
        return json.dumps(self.manifest(scenario, resolution=resolution, stimulus=stimulus, stimuli=stimuli), **options)

    def wind_at(self, position_m: Sequence[float] | float, t_s: float | Sequence[float], wind: WindConfig | None = None,
                *, stimulus: Any = None, stimuli: Any = None, reference_pose: Any = None) -> tuple[float, float, float]:
        """Evaluate the deterministic spatial field and active gusts."""
        # Accept both ``wind_at(position, t)`` (the canonical form) and the
        # natural-language ``wind_at(t, position)`` spelling used by a few
        # controller probes.
        if isinstance(position_m, (int, float)) and not isinstance(t_s, (int, float)):
            position_m, t_s = t_s, position_m
        p = _vec3(position_m, "position_m")
        t = _finite(t_s, "t_s")
        if stimulus is None and isinstance(wind, PhysicalStimulus):
            stimulus, wind = wind, None
        if wind is None:
            w = self.config.wind
        elif isinstance(wind, Mapping):
            w = WindConfig.from_mapping(wind)
        else:
            w = wind
        phase = _hash_phase(self.seed, "wind") + _TWO_PI * (0.73 * p[0] + 0.41 * p[1] + 0.19 * p[2]) / w.spatial_wavelength_m + 0.37 * t
        amp = w.spatial_amplitude_mps
        out = (w.base_mps[0] + amp * math.sin(phase),
               w.base_mps[1] + amp * math.cos(phase + 1.1),
               w.base_mps[2] + 0.25 * amp * math.sin(0.5 * phase - 0.7))
        for gust in w.gusts:
            q = gust.weight(t)
            out = _add(out, _scale(gust.vector_mps, q))
        # A stimulus-associated air field is an optional physical input.  It
        # is added after the base/gust field and remains deterministic for a
        # fixed stimulus parameter set and timestamp.
        for item in self._resolve_stimuli(stimulus, stimuli):
            if item.air_disturbance is not None:
                local = item.air_at(p, t, reference_pose)
                out = _add(out, local["velocity_mps"])
        return out

    # A short alias is convenient for controllers that call the scene an
    # environment rather than a world.
    wind = wind_at

    def air_disturbance_at(self, position_m: Sequence[float], t_s: float, *, stimulus: Any = None,
                           stimuli: Any = None, reference_pose: Any = None) -> dict[str, Any]:
        """Return the optional stimulus air velocity/acceleration field."""
        p = _vec3(position_m, "position_m")
        total_v = (0.0, 0.0, 0.0)
        total_a = (0.0, 0.0, 0.0)
        active = False
        contributions: list[dict[str, Any]] = []
        for item in self._resolve_stimuli(stimulus, stimuli):
            local = item.air_at(p, t_s, reference_pose)
            if local["active"]:
                active = True
            total_v = _add(total_v, local["velocity_mps"])
            total_a = _add(total_a, local["acceleration_mps2"])
            contributions.append({"id": item.id, **local})
        return {"active": active, "velocity_mps": list(total_v), "acceleration_mps2": list(total_a),
                "contributions": contributions}

    def _camera_basis(self, pose: Pose) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        cy, sy = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
        cp, sp = math.cos(pose.pitch_rad), math.sin(pose.pitch_rad)
        forward = (cp * cy, cp * sy, sp)
        right0 = (-sy, cy, 0.0)
        up0 = (-sp * cy, -sp * sy, cp)
        cr, sr = math.cos(pose.roll_rad), math.sin(pose.roll_rad)
        right = _unit(_add(_scale(right0, cr), _scale(up0, sr)), (0.0, 1.0, 0.0))
        up = _unit(_add(_scale(right0, -sr), _scale(up0, cr)), (0.0, 0.0, 1.0))
        return _unit(forward), right, up

    def _prepare_render(self, pose: Pose, eye: str, camera: CameraConfig, timestamp_s: float,
                        stimulus: Any = None, stimuli: Any = None) -> _PreparedRender:
        """Resolve all frame-invariant render state once for one eye.

        ``state_at`` is intentionally evaluated with the same reference pose
        and timestamp that the legacy ``_trace`` path receives.  The prepared
        geometry therefore represents the exact same stimulus state while
        making it independent of pixel count.
        """
        forward, right, up = self._camera_basis(pose)
        origin = _add(pose.position_m, _scale(right, camera.eye_separation_m * self._eye_sign(eye)))
        boxes: list[_PreparedStimulus] = []
        for item in self._resolve_stimuli(stimulus, stimuli):
            state = item.state_at(timestamp_s, pose)
            if state.active and state.renderable:
                geometry = state.geometry
                mesh_accel = self._stimulus_mesh_accels.get(state.id)
                boxes.append(_PreparedStimulus(geometry, geometry.basis,
                                               _scale(geometry.size_m, 0.5), mesh_accel))
        return _PreparedRender(camera, forward, right, up, origin, tuple(boxes))

    def eye_pose(self, pose: Pose, eye: str = "left") -> tuple[float, float, float]:
        sign = self._eye_sign(eye)
        _, right, _ = self._camera_basis(pose)
        return _add(pose.position_m, _scale(right, self.camera.eye_separation_m * sign))

    @staticmethod
    def _eye_sign(eye: str) -> float:
        e = str(eye).lower()
        if e in ("l", "left"):
            return -0.5
        if e in ("r", "right"):
            return 0.5
        if e in ("mono", "center", "centre"):
            return 0.0
        raise ValueError("eye must be left, right, or mono")

    def _ray(self, pose: Pose, eye: str, px: float, py: float, camera: CameraConfig) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        forward, right, up = self._camera_basis(pose)
        origin = _add(pose.position_m, _scale(right, camera.eye_separation_m * self._eye_sign(eye)))
        nx = (2.0 * ((px + 0.5) / camera.width) - 1.0) * math.tan(camera.hfov_rad / 2.0)
        ny = (1.0 - 2.0 * ((py + 0.5) / camera.height)) * math.tan(camera.vfov_rad / 2.0)
        return origin, _unit(_add(forward, _add(_scale(right, nx), _scale(up, ny))))

    @staticmethod
    def _prepared_ray(prepared: _PreparedRender, px: float, py: float) -> tuple[float, float, float]:
        """Build one raster direction from already prepared camera axes."""
        camera = prepared.camera
        nx = (2.0 * ((px + 0.5) / camera.width) - 1.0) * math.tan(camera.hfov_rad / 2.0)
        ny = (1.0 - 2.0 * ((py + 0.5) / camera.height)) * math.tan(camera.vfov_rad / 2.0)
        return _unit(_add(prepared.forward,
                          _add(_scale(prepared.right, nx), _scale(prepared.up, ny))))

    def _trace(self, origin: Sequence[float], direction: Sequence[float], camera: CameraConfig | None = None,
               *, timestamp_s: float = 0.0, stimulus: Any = None, stimuli: Any = None,
               reference_pose: Any = None, _prepared: _PreparedRender | None = None) -> _RayHit | None:
        """Trace ground, static boxes, then stimuli; first exact tie wins."""
        camera = _prepared.camera if _prepared is not None else (self.camera if camera is None else camera)
        c = camera
        hits: list[_RayHit] = []
        if direction[2] < -_EPS:
            t = (self.config.ground_z_m - origin[2]) / direction[2]
            if c.near_m <= t <= c.far_m:
                p = _add(origin, _scale(direction, t))
                xmin, xmax, ymin, ymax = self.config.bounds_xy_m
                if xmin - _EPS <= p[0] <= xmax + _EPS and ymin - _EPS <= p[1] <= ymax + _EPS:
                    hits.append(_RayHit("ground", t, p, (0.0, 0.0, 1.0), "ground", "ground"))
        # The prebuilt sequence preserves walls before the target. Obstacles
        # and stimuli follow below in their configured order.
        for box, kind in self._ray_boxes:
            result = _ray_box(origin, direction, box.min_m, box.max_m, c.near_m, c.far_m)
            if result is not None:
                t, n = result
                hits.append(_RayHit(box.id, t, _add(origin, _scale(direction, t)), n, box.texture, kind))
        for obstacle in self.config.obstacles:
            result = _ray_box(origin, direction, obstacle.min_m, obstacle.max_m, c.near_m, c.far_m)
            if result is not None:
                t, n = result
                hits.append(_RayHit(obstacle.id, t, _add(origin, _scale(direction, t)), n, obstacle.texture, "obstacle"))
        if _prepared is None:
            for item in self._resolve_stimuli(stimulus, stimuli):
                state = item.state_at(timestamp_s, reference_pose)
                if not state.active or not state.renderable:
                    continue
                mesh_accel = self._stimulus_mesh_accels.get(state.id)
                if mesh_accel is not None:
                    mesh_hit = self._trace_stimulus_mesh(
                        origin, direction, state.geometry, mesh_accel, c.near_m, c.far_m
                    )
                    if mesh_hit is not None:
                        hits.append(mesh_hit)
                    continue
                result = _ray_oriented_box(origin, direction, state.geometry, c.near_m, c.far_m)
                if result is not None:
                    t, point, normal = result
                    hits.append(_RayHit(state.id, t, point, normal, state.geometry.texture, "stimulus", state.geometry.albedo))
        else:
            for prepared_box in _prepared.stimulus_boxes:
                box = prepared_box.geometry
                if prepared_box.mesh_accel is not None:
                    mesh_hit = self._trace_stimulus_mesh(
                        origin, direction, box, prepared_box.mesh_accel, c.near_m, c.far_m
                    )
                    if mesh_hit is not None:
                        hits.append(mesh_hit)
                    continue
                result = _ray_oriented_box(origin, direction, box, c.near_m, c.far_m,
                                           basis=prepared_box.basis, half_size=prepared_box.half_size)
                if result is not None:
                    t, point, normal = result
                    hits.append(_RayHit(box.id, t, point, normal, box.texture, "stimulus", box.albedo))
        if self._scene_bvh is not None:
            scene_hit = self._trace_external_scene(origin, direction, c.near_m, c.far_m)
            if scene_hit is not None:
                hits.append(scene_hit)
        return min(hits, key=lambda h: h.t) if hits else None

    def _trace_stimulus_mesh(self, origin: Sequence[float], direction: Sequence[float],
                             geometry: OrientedBox, accel: _StimulusMeshAccel,
                             near: float, far: float) -> _RayHit | None:
        """Trace a rigidly posed local mesh without rebuilding its BVH."""
        # Mesh local axes are +X normal, +Y width, +Z up. `_box_basis`
        # returns (right, up, normal), so reorder those world axes to match
        # the mesh frame before transforming the ray into local coordinates.
        right, up, normal = geometry.basis
        mesh_basis = (normal, right, up)
        local_origin = _local_point(origin, geometry.center_m, mesh_basis)
        local_direction = _unit(tuple(_dot(direction, axis) for axis in mesh_basis))
        root = accel.bvh
        if root is None:
            return None
        first = _ray_bounds_near(local_origin, local_direction, root.lo, root.hi, near, far)
        if first is None:
            return None
        best_t = far
        best: tuple[int, tuple[float, float, float]] | None = None
        stack: list[tuple[float, _TriangleBVHNode]] = [(first, root)]
        while stack:
            entry, node = stack.pop()
            if entry > best_t:
                continue
            if node.triangle_indices:
                for triangle_index in node.triangle_indices:
                    triangle = accel.triangles[triangle_index]
                    result = _ray_triangle(local_origin, local_direction, triangle, near, best_t)
                    if result is not None and result[0] < best_t:
                        best_t, local_normal = result
                        best = (triangle_index, local_normal)
                continue
            children: list[tuple[float, _TriangleBVHNode]] = []
            for child in (node.left, node.right):
                if child is None:
                    continue
                child_entry = _ray_bounds_near(local_origin, local_direction,
                                               child.lo, child.hi, near, best_t)
                if child_entry is not None:
                    children.append((child_entry, child))
            children.sort(key=lambda item: item[0], reverse=True)
            stack.extend(children)
        if best is None:
            return None
        triangle_index, local_normal = best
        world_normal = _unit(tuple(sum(local_normal[j] * mesh_basis[j][axis]
                                       for j in range(3)) for axis in range(3)))
        point = _add(origin, _scale(direction, best_t))
        return _RayHit(geometry.id, best_t, point, world_normal,
                       "stimulus_mesh", "stimulus_mesh", accel.triangles[triangle_index].albedo)

    def _trace_external_scene(self, origin: Sequence[float], direction: Sequence[float],
                              near: float, far: float) -> _RayHit | None:
        """Find the nearest normalized external triangle without affecting physics."""
        root = self._scene_bvh
        if root is None:
            return None
        first = _ray_bounds_near(origin, direction, root.lo, root.hi, near, far)
        if first is None:
            return None
        best_t = far
        best: tuple[int, tuple[float, float, float]] | None = None
        stack: list[tuple[float, _TriangleBVHNode]] = [(first, root)]
        triangles = self._scene_triangles
        while stack:
            entry, node = stack.pop()
            if entry > best_t:
                continue
            if node.triangle_indices:
                for triangle_index in node.triangle_indices:
                    triangle = triangles[triangle_index]
                    result = _ray_triangle(origin, direction, triangle, near, best_t)
                    if result is not None and result[0] < best_t:
                        best_t, normal = result
                        best = (triangle_index, normal)
                continue
            children: list[tuple[float, _TriangleBVHNode]] = []
            for child in (node.left, node.right):
                if child is None:
                    continue
                child_entry = _ray_bounds_near(origin, direction, child.lo, child.hi, near, best_t)
                if child_entry is not None:
                    children.append((child_entry, child))
            # A stack is LIFO: visit the nearer child first to tighten pruning.
            children.sort(key=lambda item: item[0], reverse=True)
            stack.extend(children)
        if best is None:
            return None
        triangle_index, normal = best
        triangle = triangles[triangle_index]
        point = _add(origin, _scale(direction, best_t))
        return _RayHit(triangle.object_id, best_t, point, normal, "scene_mesh", "scene_mesh", triangle.albedo)

    def trace_ray(self, origin_m: Sequence[float], direction: Sequence[float], *, near_m: float | None = None,
                  far_m: float | None = None, timestamp_s: float = 0.0, stimulus: Any = None,
                  stimuli: Any = None, reference_pose: Any = None) -> Hit | None:
        """Trace a ray and expose the hit as a public geometry record."""
        c = CameraConfig(1, 1, self.camera.hfov_rad, self.camera.vfov_rad, self.camera.eye_separation_m,
                         self.camera.near_m if near_m is None else near_m, self.camera.far_m if far_m is None else far_m)
        d = _unit(_vec3(direction, "direction"))
        h = self._trace(_vec3(origin_m, "origin_m"), d, c, timestamp_s=timestamp_s, stimulus=stimulus,
                        stimuli=stimuli, reference_pose=reference_pose)
        return None if h is None else Hit(h.object_id, h.t, h.point, h.normal, h.texture, h.kind)

    def _texture(self, texture: str, point: Sequence[float], *, kind: str, object_id: str) -> float:
        """Evaluate a smooth, finite-bandwidth luminance pattern."""
        cell = self.config.texture.wall_cell_m if kind == "wall" else self.config.texture.ground_cell_m
        if kind in ("target", "obstacle", "stimulus"):
            cell = max(0.04, 0.75 * self.config.texture.ground_cell_m)
        phase_key = (int(self.seed), str(object_id))
        phase = self._texture_phase_cache.get(phase_key)
        if phase is None:
            phase = _hash_phase(self.seed, object_id)
            self._texture_phase_cache[phase_key] = phase
        x, y = point[0], point[1]
        if texture in ("vertical_bars", "bars"):
            pattern = 0.5 + 0.5 * math.sin(_TWO_PI * x / cell + phase)
        elif texture in ("horizontal_bars",):
            pattern = 0.5 + 0.5 * math.sin(_TWO_PI * y / cell + phase)
        elif texture in ("concentric_checker", "concentric", "target"):
            center = self.config.target.position_m
            radius = math.hypot(x - center[0], y - center[1])
            rings = 0.5 + 0.5 * math.sin(_TWO_PI * radius / cell + phase)
            sectors = 0.5 + 0.5 * math.sin(4.0 * math.atan2(y - center[1], x - center[0]) + phase * 0.37)
            pattern = 0.5 + 0.5 * (2.0 * rings - 1.0) * (2.0 * sectors - 1.0)
        elif texture in ("checker", "ground", "wall", "grid"):
            sx = math.sin(_TWO_PI * x / cell + phase)
            sy = math.sin(_TWO_PI * y / cell - phase * 0.61)
            pattern = 0.5 + 0.5 * sx * sy
        elif texture in ("palm", "dark_palm", "matte", "board", "hand"):
            # A broad deterministic two-tone pattern keeps the board visible
            # at low resolutions without adding stochastic image noise.
            sx = math.sin(_TWO_PI * x / max(cell * 4.0, 0.16) + phase)
            sy = math.sin(_TWO_PI * y / max(cell * 5.0, 0.20) - phase * 0.37)
            pattern = 0.5 + 0.25 * sx + 0.25 * sy
        else:
            pattern = 0.5 + 0.5 * math.sin(_TWO_PI * (0.7 * x + 0.3 * y) / cell + phase)
        return _clamp(pattern)

    def _shade(self, hit: _RayHit) -> float:
        if hit.kind == "ground":
            base = 0.43
        elif hit.kind == "target":
            base = 0.74
        elif hit.kind == "obstacle":
            base = 0.52
        elif hit.kind == "stimulus":
            base = 0.66 if hit.albedo is None else hit.albedo
        elif hit.kind == "stimulus_mesh":
            base = 0.66 if hit.albedo is None else hit.albedo
        elif hit.kind == "scene_mesh":
            base = 0.68 if hit.albedo is None else hit.albedo
        else:
            base = 0.55
        if hit.kind in {"scene_mesh", "stimulus_mesh"}:
            albedo = _clamp(base)
        else:
            pattern = self._texture(hit.texture, hit.point, kind=hit.kind, object_id=hit.object_id)
            contrast = self.config.texture.contrast * (1.25 if hit.kind == "target" else 1.0)
            albedo = _clamp(base * (1.0 - contrast) + base * contrast * (0.35 + 1.3 * pattern))
        lambert = max(0.0, _dot(hit.normal, self._light_direction_unit))
        shade = self.config.light.ambient + self.config.light.intensity * lambert
        return _clamp(albedo * shade)

    def render_eye(self, pose: Pose, eye: str = "left", resolution: Any = None, *, width: int | None = None,
                   height: int | None = None, timestamp_s: float = 0.0, stimulus: Any = None,
                   stimuli: Any = None, t_s: float | None = None) -> list[list[float]]:
        """Render one independent eye as a deterministic [row][column] image."""
        pose = _coerce_pose(pose)
        if stimulus is None and (isinstance(resolution, (PhysicalStimulus, str)) or
                                 (isinstance(resolution, Mapping) and any(key in resolution for key in ("start_position_m", "start", "velocity_mps", "speed_mps", "size_m")))):
            stimulus, resolution = resolution, None
        if t_s is not None:
            timestamp_s = t_s
        camera = self._resolve_camera(resolution, width=width, height=height)
        prepared = self._prepare_render(pose, eye, camera, timestamp_s, stimulus, stimuli)
        out = GrayFrame()
        for py in range(camera.height):
            row: list[float] = []
            for px in range(camera.width):
                direction = self._prepared_ray(prepared, px, py)
                hit = self._trace(prepared.origin, direction, _prepared=prepared)
                if hit is None:
                    # A very slight vertical sky gradient makes an empty ray
                    # distinguishable without introducing stochastic noise.
                    row.append(_clamp(self.config.light.sky_brightness + 0.045 * max(-1.0, min(1.0, direction[2]))))
                else:
                    row.append(self._shade(hit))
            out.append(row)
        return out

    def render_eye_directions(self, pose: Pose, directions: Any, eye: str = "left", *,
                              timestamp_s: float = 0.0, stimulus: Any = None,
                              stimuli: Any = None, t_s: float | None = None) -> list[float]:
        """Render exact luminance samples along head-frame directions.

        ``directions`` must have shape ``(N, 3)`` and contains finite vectors
        in the fly head frame: ``+x`` is forward, ``+y`` is fly-left, and
        ``+z`` is up.  The camera basis in this module uses a fly-right axis,
        so the head-left component is mapped to ``-right`` before the ray is
        traced.  Each ray starts at the selected eye's current world-space
        origin (including the configured inter-eye separation), and is then
        passed through the same timestamped stimulus geometry and shading path
        as :meth:`render_eye`.

        The returned list has exactly one finite ``[0, 1]`` luminance value
        per input direction.  A zero direction is rejected rather than being
        silently replaced by a forward ray; this prevents malformed compound
        eye maps from changing the visual stimulus.
        """
        pose = _coerce_pose(pose)
        if t_s is not None:
            timestamp_s = t_s
        timestamp_s = _finite(timestamp_s, "timestamp_s")
        # ``render_eye`` places the eye before its raster resolution.  Accept
        # that positional spelling as well (``pose, eye, directions``) while
        # keeping the documented ``pose, directions, eye`` form unambiguous.
        if isinstance(directions, str) and not isinstance(eye, str):
            directions, eye = eye, directions
        # Validate the eye before doing any work so a malformed call cannot
        # produce a partial output.  ``_eye_sign`` also normalises aliases.
        eye_sign = self._eye_sign(eye)

        if isinstance(directions, (str, bytes)):
            raise ValueError("directions must have shape (N, 3)")
        try:
            rows = tuple(directions)
        except TypeError as exc:
            raise ValueError("directions must have shape (N, 3)") from exc
        if not rows:
            raise ValueError("directions must contain at least one direction")
        head_dirs: list[tuple[float, float, float]] = []
        for index, direction in enumerate(rows):
            try:
                head = _vec3(direction, f"directions[{index}]")
            except ValueError as exc:
                raise ValueError("directions must have shape (N, 3)") from exc
            if _norm(head) < _EPS:
                raise ValueError(f"directions[{index}] must be non-zero")
            head_dirs.append(_unit(head))

        camera = self.camera
        prepared = self._prepare_render(pose, eye, camera, timestamp_s, stimulus, stimuli)
        forward, right, up = prepared.forward, prepared.right, prepared.up
        origin = prepared.origin
        out: list[float] = []
        for head in head_dirs:
            # Head +y is fly-left, whereas ``right`` is explicitly fly-right.
            direction = _unit(_add(_scale(forward, head[0]),
                                   _add(_scale(right, -head[1]), _scale(up, head[2]))))
            hit = self._trace(origin, direction, _prepared=prepared)
            if hit is None:
                luminance = _clamp(self.config.light.sky_brightness +
                                   0.045 * max(-1.0, min(1.0, direction[2])))
            else:
                luminance = self._shade(hit)
            if not math.isfinite(luminance) or not 0.0 <= luminance <= 1.0:
                raise RuntimeError("ray shading produced a non-finite/out-of-range luminance")
            out.append(float(luminance))
        return out

    # Descriptive aliases for callers that call the samples columns or use
    # the head-frame terminology directly.
    render_eye_columns = render_eye_directions
    render_head_directions = render_eye_directions

    render_frame = render_eye

    def render_stereo(self, pose: Pose, resolution: Any = None, *, width: int | None = None,
                      height: int | None = None, timestamp_s: float = 0.0, stimulus: Any = None,
                      stimuli: Any = None, t_s: float | None = None) -> StereoFrames:
        if stimulus is None and (isinstance(resolution, (PhysicalStimulus, str)) or
                                 (isinstance(resolution, Mapping) and any(key in resolution for key in ("start_position_m", "start", "velocity_mps", "speed_mps", "size_m")))):
            stimulus, resolution = resolution, None
        if t_s is not None:
            timestamp_s = t_s
        camera = self._resolve_camera(resolution, width=width, height=height)
        left = self.render_eye(pose, "left", (camera.width, camera.height), timestamp_s=timestamp_s,
                               stimulus=stimulus, stimuli=stimuli)
        right = self.render_eye(pose, "right", (camera.width, camera.height), timestamp_s=timestamp_s,
                                stimulus=stimulus, stimuli=stimuli)
        return StereoFrames(left, right, float(timestamp_s), camera.eye_separation_m)

    render_stereo_frames = render_stereo

    def _project(self, point: Sequence[float], pose: Pose, eye: str, camera: CameraConfig) -> tuple[float, float, float] | None:
        forward, right, up = self._camera_basis(pose)
        origin = _add(pose.position_m, _scale(right, camera.eye_separation_m * self._eye_sign(eye)))
        delta = _sub(point, origin)
        depth = _dot(delta, forward)
        if depth <= camera.near_m:
            return None
        nx = _dot(delta, right) / depth / math.tan(camera.hfov_rad / 2.0)
        ny = _dot(delta, up) / depth / math.tan(camera.vfov_rad / 2.0)
        return ((nx + 1.0) * 0.5 * camera.width - 0.5,
                (1.0 - ny) * 0.5 * camera.height - 0.5, depth)

    def optic_flow_ground_truth(self, pose0: Pose, pose1: Pose, *, dt_s: float | None = None, eye: str = "left",
                                resolution: Any = None, timestamp_s: float = 0.0, timestamp_s0: float | None = None,
                                timestamp_s1: float | None = None, stimulus: Any = None, stimuli: Any = None) -> OpticFlow:
        """Project static scene points between poses to obtain exact pixel flow."""
        pose0 = _coerce_pose(pose0)
        pose1 = _coerce_pose(pose1)
        camera = self._resolve_camera(resolution)
        if dt_s is None:
            dt_s = 0.0
        dt_s = _finite(dt_s, "dt_s")
        t0 = timestamp_s if timestamp_s0 is None else _finite(timestamp_s0, "timestamp_s0")
        t1 = t0 + dt_s if timestamp_s1 is None else _finite(timestamp_s1, "timestamp_s1")
        flow: list[list[tuple[float, float]]] = []
        valid: list[list[bool]] = []
        for py in range(camera.height):
            flow_row: list[tuple[float, float]] = []
            valid_row: list[bool] = []
            for px in range(camera.width):
                origin, direction = self._ray(pose0, eye, px, py, camera)
                hit = self._trace(origin, direction, camera, timestamp_s=t0, stimulus=stimulus, stimuli=stimuli,
                                  reference_pose=pose0)
                if hit is None:
                    flow_row.append((0.0, 0.0))
                    valid_row.append(False)
                    continue
                projected_point = hit.point
                # For a moving board, preserve the hit's local material point
                # between timestamps so flow does not silently treat it as a
                # static obstacle.  Static-scene hits retain the old path.
                if hit.kind == "stimulus":
                    for item in self._resolve_stimuli(stimulus, stimuli):
                        state0 = item.state_at(t0, pose0)
                        state1 = item.state_at(t1, pose1)
                        if state0.active and state1.active and state0.id == hit.object_id:
                            local = _local_point(hit.point, state0.position_m, state0.geometry.basis)
                            projected_point = _world_point(local, state1.position_m, state1.geometry.basis)
                            break
                projected = self._project(projected_point, pose1, eye, camera)
                if projected is None or not (-0.5 <= projected[0] < camera.width - 0.5 and -0.5 <= projected[1] < camera.height - 0.5):
                    flow_row.append((0.0, 0.0))
                    valid_row.append(False)
                    continue
                du, dv = projected[0] - px, projected[1] - py
                flow_row.append((0.0 if abs(du) < 1e-12 else du, 0.0 if abs(dv) < 1e-12 else dv))
                valid_row.append(True)
            flow.append(flow_row)
            valid.append(valid_row)
        return OpticFlow(flow, valid, dt_s, str(eye).lower(), camera.hfov_rad, camera.vfov_rad)

    optic_flow = optic_flow_ground_truth
    flow_ground_truth = optic_flow_ground_truth
    compute_optic_flow = optic_flow_ground_truth

    def target_observation(self, pose: Pose, t_s: float = 0.0, *, stimulus: Any = None, stimuli: Any = None) -> TargetObservation:
        pose = _coerce_pose(pose)
        target = self.config.target
        delta = _sub(target.position_m, pose.position_m)
        horizontal = math.hypot(delta[0], delta[1])
        distance = _norm(delta)
        # At the exact target centre the azimuth is undefined; returning zero
        # is the stable, controller-friendly convention for a landed body.
        bearing = 0.0 if horizontal < _EPS else _wrap_angle(math.atan2(delta[1], delta[0]) - pose.yaw_rad)
        elevation = math.atan2(delta[2], max(horizontal, _EPS))
        direction = _unit(delta, (1.0, 0.0, 0.0))
        first = self._trace(pose.position_m, direction, self.camera, timestamp_s=t_s, stimulus=stimulus,
                            stimuli=stimuli, reference_pose=pose)
        visible = first is not None and first.object_id == target.id
        xmin, xmax, ymin, ymax = self.config.bounds_xy_m  # noqa: F841 - documents room frame
        tmin, tmax = target.min_m, target.max_m
        on_target = tmin[0] <= pose.position_m[0] <= tmax[0] and tmin[1] <= pose.position_m[1] <= tmax[1]
        return TargetObservation(target.id, distance, bearing, elevation, visible, on_target)

    target_bearing = target_observation

    def altitude_m(self, pose: Pose) -> float:
        pose = _coerce_pose(pose)
        return pose.position_m[2] - self.config.ground_z_m

    height_m = altitude_m

    def target_bearing_rad(self, pose: Pose) -> float:
        return self.target_observation(pose).bearing_rad

    def looming_at(self, pose: Pose, previous_pose: Pose | None = None, *, dt_s: float | None = None,
                   t_s: float = 0.0, previous_t_s: float | None = None,
                   stimulus: Any = None, stimuli: Any = None) -> dict[str, dict[str, Any]]:
        """Return deterministic looming cues for every configured obstacle.

        ``angular_radius_rad`` is the obstacle's projected half-size and
        ``tau_s`` is the usual time-to-contact proxy ``theta / dtheta`` when
        the angular size is expanding.  The cue is geometry-only; it does not
        add a second physics engine or random event stream.
        """
        pose = _coerce_pose(pose)
        if previous_pose is not None:
            previous_pose = _coerce_pose(previous_pose)
        if dt_s is not None and _finite(dt_s, "dt_s") <= 0.0:
            raise ValueError("dt_s must be > 0 when supplied")
        result: dict[str, dict[str, Any]] = {}
        for obstacle in self.config.obstacles:
            center = obstacle.center_m
            delta = _sub(center, pose.position_m)
            distance = max(_norm(delta), _EPS)
            radius = 0.5 * _norm(obstacle.size_m)
            theta = math.atan2(radius, distance)
            if previous_pose is not None and dt_s is not None:
                prev_d = max(_norm(_sub(center, previous_pose.position_m)), _EPS)
                prev_theta = math.atan2(radius, prev_d)
                rate = (theta - prev_theta) / dt_s
            else:
                closing = max(0.0, _dot(pose.velocity_mps, _unit(delta)))
                rate = theta * closing / distance
            # ``theta / dtheta`` is positive for an expanding (approaching)
            # obstacle and is the conventional looming time constant.
            tau = theta / rate if rate > _EPS else float("inf")
            line = self._trace(pose.position_m, _unit(delta), self.camera)
            visible = line is not None and line.object_id == obstacle.id
            result[obstacle.id] = {"distance_m": distance, "angular_radius_rad": theta,
                                   "angular_rate_rad_s": rate, "tau_s": None if not math.isfinite(tau) else tau,
                                   "looming": bool(rate > 0.0), "visible": visible}
        selected_stimuli = self._resolve_stimuli(stimulus, stimuli)
        if selected_stimuli:
            previous_t = t_s - dt_s if previous_t_s is None and dt_s is not None else previous_t_s
            for item in selected_stimuli:
                state = item.state_at(t_s, pose)
                if not state.active:
                    result[item.id] = {"distance_m": float("inf"), "angular_radius_rad": 0.0,
                                       "angular_rate_rad_s": 0.0, "tau_s": None,
                                       "looming": False, "visible": False, "active": False,
                                       "mode": item.mode}
                if not state.renderable:
                    result[item.id] = {"distance_m": _norm(_sub(state.position_m, pose.position_m)),
                                       "angular_radius_rad": 0.0, "angular_rate_rad_s": 0.0,
                                       "tau_s": None, "looming": False, "visible": False,
                                       "active": True, "mode": item.mode, "colliding": False}
                    continue
                    continue
                delta = _sub(state.position_m, pose.position_m)
                distance = max(_norm(delta), _EPS)
                radius = _stimulus_in_plane_half_size(state.size_m)
                theta = math.atan2(radius, distance)
                rate = 0.0
                if previous_t is not None and dt_s is not None and dt_s > 0.0:
                    previous = item.state_at(previous_t, previous_pose if previous_pose is not None else pose)
                    prev_distance = max(_norm(_sub(previous.position_m, pose.position_m)), _EPS)
                    prev_radius = _stimulus_in_plane_half_size(previous.size_m)
                    prev_theta = math.atan2(prev_radius, prev_distance)
                    rate = (theta - prev_theta) / dt_s
                elif previous_pose is not None and dt_s is not None and dt_s > 0.0:
                    previous = item.state_at(t_s - dt_s, previous_pose)
                    prev_distance = max(_norm(_sub(previous.position_m, previous_pose.position_m)), _EPS)
                    prev_radius = _stimulus_in_plane_half_size(previous.size_m)
                    prev_theta = math.atan2(prev_radius, prev_distance)
                    rate = (theta - prev_theta) / dt_s
                else:
                    closing = max(0.0, _dot(_sub(pose.velocity_mps, state.velocity_mps), _unit(delta)))
                    rate = theta * closing / distance
                tau = theta / rate if rate > _EPS else float("inf")
                line = self._trace(pose.position_m, _unit(delta), self.camera, timestamp_s=t_s,
                                   stimulus=selected_stimuli, reference_pose=pose)
                visible = line is not None and line.object_id == item.id
                hard, _ = self.collision_events(pose, t_s, stimulus=item)
                surface_distance = self._stimulus_surface_distance(pose.position_m, state.geometry)
                contact_radius = item.contact_radius_m if item.contact_radius_m is not None else self.config.body_radius_m
                contact_time = item.counterfactual_contact_time_s if item.counterfactual_contact_time_s is not None else item.contact_time_s(reference_position_m=pose.position_m, radius_m=contact_radius, reference_pose=pose)
                result[item.id] = {
                    "distance_m": distance, "angular_radius_rad": theta,
                    "angular_rate_rad_s": rate, "tau_s": None if not math.isfinite(tau) else tau,
                    "looming": bool(rate > 0.0), "visible": visible, "active": True,
                    "mode": item.mode, "surface_distance_m": surface_distance,
                    "surface_contact": surface_distance <= contact_radius + 1.0e-9,
                    "contact_time_s": contact_time,
                    "counterfactual_contact_time_s": contact_time,
                    "visual_only": item.visual_only,
                    "non_interacting": item.non_interacting,
                    "colliding": any(event.object_id == item.id for event in hard),
                }
        return result

    looming = looming_at
    looming_signal = looming_at

    @staticmethod
    def _stimulus_surface_distance(point_m: Sequence[float], box: OrientedBox) -> float:
        """Signed point-to-box distance (negative while inside the box)."""
        local = _local_point(point_m, box.center_m, box.basis)
        half = _scale(box.size_m, 0.5)
        outside = tuple(max(abs(local[i]) - half[i], 0.0) for i in range(3))
        outside_distance = _norm(outside)
        if outside_distance > _EPS:
            return outside_distance
        # Inside: report the negative distance to the shallowest face.
        return -min(half[i] - abs(local[i]) for i in range(3))

    def stimulus_observation(self, pose: Pose, stimulus: Any, t_s: float = 0.0, *,
                             previous_t_s: float | None = None, dt_s: float | None = None) -> StimulusObservation:
        """Measure one moving board's geometry, visibility and contact."""
        pose = _coerce_pose(pose)
        items = self._resolve_stimuli(stimulus)
        if len(items) != 1:
            raise ValueError("stimulus_observation expects exactly one stimulus")
        item = items[0]
        state = item.state_at(t_s, pose)
        counterfactual_contact_time = item.counterfactual_contact_time_s if item.counterfactual_contact_time_s is not None else item.contact_time_s(reference_pose=pose)
        if not state.active:
            return StimulusObservation(item.id, float(t_s), False, False, False, float("inf"), float("inf"),
                                       0.0, 0.0, state.position_m, state.size_m, item.mode,
                                       contact_time_s=counterfactual_contact_time,
                                       albedo=item.albedo, appearance_provenance=item.appearance_provenance,
                                       visual_only=item.visual_only, non_interacting=item.non_interacting,
                                       counterfactual_contact_time_s=counterfactual_contact_time)
        if not state.renderable:
            return StimulusObservation(item.id, float(t_s), True, False, False,
                                       _norm(_sub(state.position_m, pose.position_m)),
                                       float("inf"), 0.0, 0.0, state.position_m, state.size_m, item.mode,
                                       contact_time_s=None, albedo=item.albedo,
                                       appearance_provenance=item.appearance_provenance,
                                       visual_only=item.visual_only, non_interacting=item.non_interacting,
                                       counterfactual_contact_time_s=counterfactual_contact_time)
        delta = _sub(state.position_m, pose.position_m)
        distance = max(_norm(delta), _EPS)
        angular_radius = math.atan2(_stimulus_in_plane_half_size(state.size_m), distance)
        rate = 0.0
        if dt_s is not None and _finite(dt_s, "dt_s") > 0.0:
            previous_time = t_s - dt_s if previous_t_s is None else previous_t_s
            prior = item.state_at(previous_time, pose)
            prior_distance = max(_norm(_sub(prior.position_m, pose.position_m)), _EPS)
            prior_radius = _stimulus_in_plane_half_size(prior.size_m)
            rate = (angular_radius - math.atan2(prior_radius, prior_distance)) / float(dt_s)
        line = self._trace(pose.position_m, _unit(delta), self.camera, timestamp_s=t_s, stimulus=item,
                           reference_pose=pose)
        visible = bool(line is not None and line.object_id == item.id and state.renderable)
        hard, _ = self.collision_events(pose, t_s, stimulus=item)
        contact_radius = item.contact_radius_m if item.contact_radius_m is not None else self.config.body_radius_m
        surface_distance = self._stimulus_surface_distance(pose.position_m, state.geometry)
        contact_time = item.counterfactual_contact_time_s if item.counterfactual_contact_time_s is not None else item.contact_time_s(reference_position_m=pose.position_m, radius_m=contact_radius, reference_pose=pose)
        return StimulusObservation(item.id, float(t_s), True, visible,
                                   any(event.object_id == item.id for event in hard), distance,
                                   surface_distance, angular_radius, rate, state.position_m, state.size_m, item.mode,
                                   surface_contact=surface_distance <= contact_radius + 1.0e-9,
                                   contact_time_s=contact_time, albedo=item.albedo,
                                   appearance_provenance=item.appearance_provenance,
                                   visual_only=item.visual_only, non_interacting=item.non_interacting,
                                   counterfactual_contact_time_s=contact_time)

    observe_stimulus = stimulus_observation
    stimulus_distance = lambda self, pose, stimulus, t_s=0.0: self.stimulus_observation(pose, stimulus, t_s).surface_distance_m
    distance_to_stimulus = stimulus_distance

    def collision_events(self, pose: Pose, t_s: float = 0.0, stimulus: Any = None,
                         stimuli: Any = None) -> tuple[list[CollisionEvent], list[CollisionEvent]]:
        """Return (hard collisions, benign ground/pad contacts)."""
        pose = _coerce_pose(pose)
        r = self.config.body_radius_m
        p = pose.position_m
        collisions: list[CollisionEvent] = []
        contacts: list[CollisionEvent] = []
        ground = self.config.ground_z_m
        if p[2] - r <= ground + 1e-8:
            contacts.append(CollisionEvent(t_s, "ground", "contact", (p[0], p[1], ground), (0.0, 0.0, 1.0), max(0.0, ground + r - p[2])))
        xmin, xmax, ymin, ymax = self.config.bounds_xy_m
        if p[2] - r < ground + self.config.wall_height_m and p[2] + r > ground:
            for oid, depth, normal in (("wall_xmin", xmin - (p[0] - r), (1.0, 0.0, 0.0)),
                                        ("wall_xmax", (p[0] + r) - xmax, (-1.0, 0.0, 0.0)),
                                        ("wall_ymin", ymin - (p[1] - r), (0.0, 1.0, 0.0)),
                                        ("wall_ymax", (p[1] + r) - ymax, (0.0, -1.0, 0.0))):
                if depth > 0.0:
                    collisions.append(CollisionEvent(t_s, oid, "collision", p, normal, depth))
        for obstacle in self.config.obstacles:
            closest = tuple(max(obstacle.min_m[i], min(p[i], obstacle.max_m[i])) for i in range(3))
            delta = _sub(p, closest)
            d = _norm(delta)
            if d < r:
                if d > _EPS:
                    normal = _scale(delta, 1.0 / d)
                else:
                    # Body centre inside the box: choose its shallowest escape
                    # face, yielding a stable normal rather than NaNs.
                    distances = ((p[0] - obstacle.min_m[0], (1.0, 0.0, 0.0)), (obstacle.max_m[0] - p[0], (-1.0, 0.0, 0.0)),
                                 (p[1] - obstacle.min_m[1], (0.0, 1.0, 0.0)), (obstacle.max_m[1] - p[1], (0.0, -1.0, 0.0)),
                                 (p[2] - obstacle.min_m[2], (0.0, 0.0, 1.0)), (obstacle.max_m[2] - p[2], (0.0, 0.0, -1.0)))
                    _, normal = min(distances, key=lambda q: q[0])
                collisions.append(CollisionEvent(t_s, obstacle.id, "collision", p, normal, r - d))
        target = self.config.target
        tbox = target.box
        within_xy = tbox.min_m[0] - r <= p[0] <= tbox.max_m[0] + r and tbox.min_m[1] - r <= p[1] <= tbox.max_m[1] + r
        if within_xy and p[2] - r <= tbox.max_m[2] + 1e-8 and p[2] + r >= tbox.min_m[2]:
            contacts.append(CollisionEvent(t_s, target.id, "contact", (p[0], p[1], tbox.max_m[2]), (0.0, 0.0, 1.0), max(0.0, tbox.max_m[2] + r - p[2])))
        for item in self._resolve_stimuli(stimulus, stimuli):
            state = item.state_at(t_s, pose)
            if not state.active or not state.collidable:
                continue
            stimulus_radius = item.contact_radius_m if item.contact_radius_m is not None else r
            box = state.geometry
            local = _local_point(p, box.center_m, box.basis)
            half = _scale(box.size_m, 0.5)
            closest_local = tuple(max(-half[i], min(local[i], half[i])) for i in range(3))
            delta_local = _sub(local, closest_local)
            distance = _norm(delta_local)
            if distance < stimulus_radius:
                if distance > _EPS:
                    normal_local = _scale(delta_local, 1.0 / distance)
                else:
                    distances = tuple((half[i] - abs(local[i]), i) for i in range(3))
                    _, axis = min(distances, key=lambda q: q[0])
                    sign = 1.0 if local[axis] >= 0.0 else -1.0
                    normal_local = tuple(sign if i == axis else 0.0 for i in range(3))
                    distance = 0.0
                normal = _unit(_world_point(normal_local, (0.0, 0.0, 0.0), box.basis), (1.0, 0.0, 0.0))
                collisions.append(CollisionEvent(t_s, item.id, "collision", p, normal, stimulus_radius - distance))
        return collisions, contacts

    def check_collisions(self, pose: Pose, t_s: float = 0.0, stimulus: Any = None,
                         stimuli: Any = None) -> dict[str, Any]:
        hard, contacts = self.collision_events(pose, t_s, stimulus=stimulus, stimuli=stimuli)
        return {"colliding": bool(hard), "collision_count": len(hard), "collisions": [c.to_dict() for c in hard], "contacts": [c.to_dict() for c in contacts]}

    check_collision = check_collisions
    collisions = collision_events

    @staticmethod
    def _profile_kind(scenario: ScenarioConfig) -> str:
        return str(scenario.profile.get("kind", "hold")).lower().replace("-", "_")

    @staticmethod
    def _normalise_phase_kind(value: Any) -> str:
        return str(value or "hold").lower().replace("-", "_")

    def _mission_phases(self, scenario: ScenarioConfig) -> list[dict[str, Any]]:
        """Validate and normalize a piecewise goal-only mission profile.

        A mission phase contains reference commands (``kind``, goal fields,
        and optional easing), never a body pose or velocity.  Explicit start
        and end times make the phase clock auditable and keep video/validator
        annotations deterministic.
        """
        raw = scenario.profile.get("phases", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise ValueError("mission profile.phases must be a non-empty sequence")
        phases: list[dict[str, Any]] = []
        previous_end = 0.0
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValueError(f"mission phase {index} must be a mapping")
            phase = dict(item)
            label = str(phase.get("label", phase.get("name", f"phase-{index}"))).strip()
            if not label:
                raise ValueError(f"mission phase {index} label must not be empty")
            start = _finite(phase.get("start_s", previous_end), f"mission phase {label}.start_s")
            if "end_s" in phase:
                end = _finite(phase["end_s"], f"mission phase {label}.end_s")
            else:
                duration = _finite(phase.get("duration_s", 0.0), f"mission phase {label}.duration_s")
                end = start + duration
            if start < -1.0e-9 or abs(start - previous_end) > 1.0e-8:
                raise ValueError(f"mission phases must be contiguous at {label!r}")
            if end <= start + _EPS:
                raise ValueError(f"mission phase {label!r} must have positive duration")
            phase["label"] = label
            phase["start_s"] = max(0.0, start)
            phase["end_s"] = end
            phase["duration_s"] = end - start
            phase["kind"] = self._normalise_phase_kind(phase.get("kind", "hold"))
            if phase["kind"] == "mission":
                raise ValueError("mission phases cannot contain nested mission profiles")
            phases.append(phase)
            previous_end = end
        if previous_end > scenario.duration_s + 1.0e-8:
            raise ValueError("mission phases cannot extend past scenario.duration_s")
        return phases

    def _mission_phase(self, scenario: ScenarioConfig, t_s: float) -> tuple[dict[str, Any], float, int]:
        phases = self._mission_phases(scenario)
        t = max(0.0, _finite(t_s, "t_s"))
        for index, phase in enumerate(phases):
            if t < phase["end_s"] - _EPS or index == len(phases) - 1:
                local_t = _clamp(t - phase["start_s"], 0.0, phase["duration_s"])
                return phase, local_t, index
        # The validation above guarantees at least one phase, and the final
        # branch handles a tiny floating-point overshoot at the mission end.
        phase = phases[-1]
        return phase, phase["duration_s"], len(phases) - 1

    @staticmethod
    def _phase_scenario(scenario: ScenarioConfig, phase: Mapping[str, Any]) -> ScenarioConfig:
        profile = dict(phase)
        for key in ("label", "name", "start_s", "end_s"):
            profile.pop(key, None)
        return ScenarioConfig(
            f"{scenario.name}:{phase.get('label', 'phase')}",
            float(phase["duration_s"]),
            scenario.dt_s,
            scenario.initial_pose,
            profile,
            {},
            None,
        )

    def _phase_manifest(self, scenario: ScenarioConfig) -> list[dict[str, Any]]:
        if self._profile_kind(scenario) != "mission":
            return []
        return [
            {
                "index": index,
                "label": phase["label"],
                "kind": phase["kind"],
                "start_s": phase["start_s"],
                "end_s": phase["end_s"],
                "duration_s": phase["duration_s"],
            }
            for index, phase in enumerate(self._mission_phases(scenario))
        ]

    def phase_at(self, scenario: str | ScenarioConfig | Mapping[str, Any], t_s: float) -> dict[str, Any]:
        """Return the active mission phase and its absolute timing.

        Non-mission scenarios still return one synthetic phase, allowing
        callers to use one annotation path for every video.
        """
        if isinstance(scenario, str):
            lookup = scenario if scenario in self.scenarios else scenario.replace("_", "-")
            if lookup not in self.scenarios:
                raise KeyError(f"unknown scenario {scenario!r}; available: {sorted(self.scenarios)}")
            sc = self.scenarios[lookup]
        elif isinstance(scenario, Mapping):
            sc = ScenarioConfig.from_mapping(str(scenario.get("name", "custom")), scenario)
        else:
            sc = scenario
        if self._profile_kind(sc) == "mission":
            phase, local_t, index = self._mission_phase(sc, t_s)
            return {
                "index": index,
                "label": phase["label"],
                "kind": phase["kind"],
                "start_s": float(phase["start_s"]),
                "end_s": float(phase["end_s"]),
                "duration_s": float(phase["duration_s"]),
                "local_time_s": float(local_t),
            }
        return {
            "index": 0,
            "label": self._profile_kind(sc),
            "kind": self._profile_kind(sc),
            "start_s": 0.0,
            "end_s": float(sc.duration_s),
            "duration_s": float(sc.duration_s),
            "local_time_s": _clamp(_finite(t_s, "t_s"), 0.0, sc.duration_s),
        }

    def goal_at(self, scenario: str | ScenarioConfig | Mapping[str, Any], t_s: float) -> dict[str, Any]:
        """Return the goal-only reference command at time ``t_s``.

        The returned position and velocity are commands for the reduced
        dynamics, not the simulated body state.  This public helper gives the
        runner/video/validator one authoritative phase and goal clock.
        """
        if isinstance(scenario, str):
            lookup = scenario if scenario in self.scenarios else scenario.replace("_", "-")
            if lookup not in self.scenarios:
                raise KeyError(f"unknown scenario {scenario!r}; available: {sorted(self.scenarios)}")
            sc = self.scenarios[lookup]
        elif isinstance(scenario, Mapping):
            sc = ScenarioConfig.from_mapping(str(scenario.get("name", "custom")), scenario)
        else:
            sc = scenario
        t = _clamp(_finite(t_s, "t_s"), 0.0, sc.duration_s)
        position, yaw, _, angular = self._profile_values(sc, t)
        velocity = self._profile_velocity(sc, t)
        phase = self.phase_at(sc, t)
        return {
            "time_s": float(t),
            "phase": phase["label"],
            "phase_index": phase["index"],
            "phase_kind": phase["kind"],
            "phase_start_s": phase["start_s"],
            "phase_end_s": phase["end_s"],
            "position_m": list(position),
            "velocity_mps": list(velocity),
            "yaw_rad": float(yaw),
            "yaw_rate_radps": float(angular[2]),
            "target_id": self.config.target.id,
            "target_position_m": list(self.config.target.position_m),
        }

    @staticmethod
    def _profile_easing(profile: Mapping[str, Any]) -> tuple[bool, str]:
        raw = str(profile.get("easing", profile.get("interpolation", "linear"))).lower().replace("-", "_")
        if raw in ("linear", "lerp"):
            return False, raw
        if raw in ("smoothstep", "smooth_step", "cubic"):
            return True, raw
        raise ValueError(f"unsupported profile easing: {raw!r}")

    @staticmethod
    def _profile_fraction(profile: Mapping[str, Any], t_s: float, duration_s: float) -> tuple[float, float]:
        """Return normalized progress and its time derivative."""
        u = _clamp(t_s / max(_EPS, duration_s))
        smooth, _ = VirtualWorld._profile_easing(profile)
        if smooth:
            return _smoothstep(u), _smoothstep_derivative(u) / max(_EPS, duration_s)
        return u, (1.0 / max(_EPS, duration_s) if 0.0 < t_s < duration_s else 0.0)

    def _profile_points(self, scenario: ScenarioConfig) -> tuple[list[tuple[float, float, float]], float]:
        """Build a finite polyline for a translating profile.

        Waypoints are reference commands for the point-mass controller; the
        body is still integrated from acceleration and velocity in
        ``_trajectory``.  This is what lets the forward-target scene go
        around the looming block without teleporting through it.
        """
        p0 = scenario.initial_pose.position_m
        profile = scenario.profile
        start = _vec3(profile.get("start_position_m", profile.get("start_m", p0)), "profile.start_position_m")
        end_value = profile.get("end_position_m", profile.get("end_m"))
        if end_value is None and str(profile.get("target", "")).lower() in ("landing_target", "target", "scene_target"):
            offset = _vec3(profile.get("target_offset_m", (0.0, 0.0, 0.0)), "profile.target_offset_m")
            end_value = _add(self.config.target.position_m, offset)
        end = _vec3(end_value if end_value is not None else p0, "profile.end_position_m")
        points = [start]
        raw_waypoints = profile.get("waypoints_m", profile.get("waypoints", ()))
        if raw_waypoints is None:
            raw_waypoints = ()
        try:
            points.extend(_vec3(point, f"profile.waypoints_m[{i}]") for i, point in enumerate(raw_waypoints))
        except TypeError as exc:
            raise ValueError("profile.waypoints_m must contain three-number points") from exc
        points.append(end)
        return points, max(_EPS, _finite(profile.get("duration_s", scenario.duration_s), "profile.duration_s"))

    @staticmethod
    def _polyline_reference(points: Sequence[Sequence[float]], duration_s: float, fraction: float,
                            fraction_rate: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lengths = [_norm(_sub(points[i + 1], points[i])) for i in range(len(points) - 1)]
        total = sum(lengths)
        if total < _EPS:
            point = _vec3(points[-1], "profile.end_position_m")
            return point, (0.0, 0.0, 0.0)
        distance = _clamp(fraction) * total
        traversed = 0.0
        for i, length in enumerate(lengths):
            if length < _EPS:
                continue
            if distance <= traversed + length + _EPS or i == len(lengths) - 1:
                u = _clamp((distance - traversed) / length)
                point = _add(points[i], _scale(_sub(points[i + 1], points[i]), u))
                tangent = _scale(_sub(points[i + 1], points[i]), 1.0 / length)
                speed = total * fraction_rate
                return point, _scale(tangent, speed)
            traversed += length
        return _vec3(points[-1], "profile.end_position_m"), (0.0, 0.0, 0.0)

    def _profile_values(self, scenario: ScenarioConfig, t_s: float) -> tuple[tuple[float, float, float], float, float, tuple[float, float, float]]:
        """Evaluate deterministic reference commands, not body coordinates.

        The values are inputs to the trajectory controller.  ``_trajectory``
        integrates acceleration and angular rate, so a reference ramp cannot
        teleport the body or silently bypass collision handling.
        """
        p0 = scenario.initial_pose.position_m
        profile = scenario.profile
        kind = self._profile_kind(scenario)
        if kind == "mission":
            phase, local_t, _ = self._mission_phase(scenario, t_s)
            child = self._phase_scenario(scenario, phase)
            return self._profile_values(child, local_t)
        if kind in ("hold", "hover"):
            desired = _vec3(profile.get("position_m", profile.get("position", p0)), "profile.position_m")
            yaw = _finite(profile.get("yaw_rad", profile.get("yaw", scenario.initial_pose.yaw_rad)), "profile.yaw_rad")
            return desired, yaw, 0.0, (0.0, 0.0, 0.0)
        if kind == "yaw_step":
            at = max(0.0, _finite(profile.get("step_at_s", profile.get("start_s", 0.0)), "profile.step_at_s"))
            rate = _finite(profile.get("yaw_rate_radps", profile.get("yaw_rate_rad_s", 0.0)), "profile.yaw_rate_radps")
            elapsed = max(0.0, t_s - at)
            yaw = scenario.initial_pose.yaw_rad + rate * elapsed
            return p0, yaw, 0.0, (0.0, 0.0, rate if t_s >= at else 0.0)
        if kind in ("vertical_ramp", "ramp"):
            start = _finite(profile.get("start_z_m", profile.get("start_z", p0[2])), "profile.start_z_m")
            end = _finite(profile.get("end_z_m", profile.get("end_z", p0[2])), "profile.end_z_m")
            duration = max(_EPS, _finite(profile.get("duration_s", scenario.duration_s), "profile.duration_s"))
            fraction, fraction_rate = self._profile_fraction(profile, t_s, duration)
            z = start + (end - start) * fraction
            vz = (end - start) * fraction_rate
            horizontal = _vec3(profile.get("position_m", profile.get("start_position_m", p0)), "profile.position_m")
            yaw = _finite(profile.get("yaw_rad", profile.get("yaw", scenario.initial_pose.yaw_rad)), "profile.yaw_rad")
            return (horizontal[0], horizontal[1], z), yaw, vz, (0.0, 0.0, 0.0)
        if kind in ("yaw_ramp", "heading_ramp"):
            start = _finite(profile.get("start_yaw_rad", profile.get("start_yaw", scenario.initial_pose.yaw_rad)), "profile.start_yaw_rad")
            end = _finite(profile.get("end_yaw_rad", profile.get("end_yaw", start)), "profile.end_yaw_rad")
            duration = max(_EPS, _finite(profile.get("duration_s", scenario.duration_s), "profile.duration_s"))
            fraction, fraction_rate = self._profile_fraction(profile, t_s, duration)
            yaw = start + (end - start) * fraction
            yaw_rate = (end - start) * fraction_rate
            position = _vec3(profile.get("position_m", p0), "profile.position_m")
            return position, yaw, 0.0, (0.0, 0.0, yaw_rate)
        if kind in ("forward_ramp", "forward_target", "position_ramp", "translation_ramp", "waypoint_ramp", "target_approach"):
            points, duration = self._profile_points(scenario)
            fraction, fraction_rate = self._profile_fraction(profile, t_s, duration)
            pos, velocity = self._polyline_reference(points, duration, fraction, fraction_rate)
            yaw = _finite(profile.get("yaw_rad", profile.get("yaw", scenario.initial_pose.yaw_rad)), "profile.yaw_rad")
            return pos, yaw, velocity[2], (0.0, 0.0, 0.0)
        raise ValueError(f"unsupported scenario profile kind: {kind!r}")

    def _profile_velocity(self, scenario: ScenarioConfig, t_s: float) -> tuple[float, float, float]:
        kind = self._profile_kind(scenario)
        if kind == "mission":
            phase, local_t, _ = self._mission_phase(scenario, t_s)
            return self._profile_velocity(self._phase_scenario(scenario, phase), local_t)
        if kind in ("vertical_ramp", "ramp"):
            return (0.0, 0.0, self._profile_values(scenario, t_s)[2])
        if kind in ("forward_ramp", "forward_target", "position_ramp", "translation_ramp", "waypoint_ramp", "target_approach"):
            points, duration = self._profile_points(scenario)
            fraction, fraction_rate = self._profile_fraction(scenario.profile, t_s, duration)
            return self._polyline_reference(points, duration, fraction, fraction_rate)[1]
        return (0.0, 0.0, 0.0)

    def _resolve_integrated_motion(self, position: Sequence[float], velocity: Sequence[float], *, t_s: float = 0.0,
                                    stimulus: Any = None, stimuli: Any = None, reference_pose: Any = None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve contacts after one integration step.

        This is a small impulse/contact boundary, not a second trajectory
        generator.  It keeps the reduced world inside its finite room and
        prevents numerical overshoot from turning a safe reference path into
        a reported hard collision.
        """
        p = list(_vec3(position, "position_m"))
        v = list(_vec3(velocity, "velocity_mps"))
        r = self.config.body_radius_m
        floor = self.config.ground_z_m + r
        if p[2] < floor:
            p[2] = floor
            if v[2] < 0.0:
                v[2] = 0.0
        xmin, xmax, ymin, ymax = self.config.bounds_xy_m
        if self.config.ground_z_m <= p[2] <= self.config.ground_z_m + self.config.wall_height_m:
            if p[0] < xmin + r:
                p[0] = xmin + r
                if v[0] < 0.0:
                    v[0] = 0.0
            elif p[0] > xmax - r:
                p[0] = xmax - r
                if v[0] > 0.0:
                    v[0] = 0.0
            if p[1] < ymin + r:
                p[1] = ymin + r
                if v[1] < 0.0:
                    v[1] = 0.0
            elif p[1] > ymax - r:
                p[1] = ymax - r
                if v[1] > 0.0:
                    v[1] = 0.0
        for obstacle in self.config.obstacles:
            expanded_min = (obstacle.min_m[0] - r, obstacle.min_m[1] - r, obstacle.min_m[2] - r)
            expanded_max = (obstacle.max_m[0] + r, obstacle.max_m[1] + r, obstacle.max_m[2] + r)
            if not all(expanded_min[i] < p[i] < expanded_max[i] for i in range(3)):
                continue
            escapes = (
                (p[0] - expanded_min[0], (1.0, 0.0, 0.0)),
                (expanded_max[0] - p[0], (-1.0, 0.0, 0.0)),
                (p[1] - expanded_min[1], (0.0, 1.0, 0.0)),
                (expanded_max[1] - p[1], (0.0, -1.0, 0.0)),
                (p[2] - expanded_min[2], (0.0, 0.0, 1.0)),
                (expanded_max[2] - p[2], (0.0, 0.0, -1.0)),
            )
            penetration, normal = min(escapes, key=lambda item: item[0])
            p = list(_add(p, _scale(normal, penetration + 1.0e-7)))
            inward = _dot(v, normal)
            if inward < 0.0:
                v = list(_sub(v, _scale(normal, inward)))

        # The landing target is a physical pad, not only a visual marker.  Its
        # top surface supports the body at ``pad_top + body_radius``.  Without
        # this boundary the telemetry can report target contact while the
        # integrated body continues to sink through the pad until the global
        # ground clamp catches it.  Keep the lower threshold so a body that
        # starts below a raised pad is not teleported upward through its
        # underside; normal approach from above is stopped and downward speed
        # is removed at the support plane.
        target = self.config.target
        target_min, target_max = target.min_m, target.max_m
        support_z = target_max[2] + r
        inside_target_xy = (
            target_min[0] - r <= p[0] <= target_max[0] + r
            and target_min[1] - r <= p[1] <= target_max[1] + r
        )
        if inside_target_xy and target_max[2] - r <= p[2] < support_z:
            p[2] = support_z
            if v[2] < 0.0:
                v[2] = 0.0
        for item in self._resolve_stimuli(stimulus, stimuli):
            state = item.state_at(t_s, reference_pose if reference_pose is not None else Pose(p))
            if not state.active or not state.collidable:
                continue
            stimulus_radius = item.contact_radius_m if item.contact_radius_m is not None else r
            box = state.geometry
            local = _local_point(p, box.center_m, box.basis)
            half = _scale(box.size_m, 0.5)
            expanded = _add(half, (stimulus_radius, stimulus_radius, stimulus_radius))
            if not all(-expanded[i] < local[i] < expanded[i] for i in range(3)):
                continue
            escapes = (
                (local[0] + expanded[0], (1.0, 0.0, 0.0)),
                (expanded[0] - local[0], (-1.0, 0.0, 0.0)),
                (local[1] + expanded[1], (0.0, 1.0, 0.0)),
                (expanded[1] - local[1], (0.0, -1.0, 0.0)),
                (local[2] + expanded[2], (0.0, 0.0, 1.0)),
                (expanded[2] - local[2], (0.0, 0.0, -1.0)),
            )
            penetration, normal_local = min(escapes, key=lambda item: item[0])
            normal = _unit(_world_point(normal_local, (0.0, 0.0, 0.0), box.basis), (1.0, 0.0, 0.0))
            p = list(_add(p, _scale(normal, penetration + 1.0e-7)))
            relative = _sub(v, state.velocity_mps)
            inward = _dot(relative, normal)
            if inward < 0.0:
                v = list(_sub(v, _scale(normal, inward)))
        return tuple(p), tuple(v)

    def _trajectory(self, scenario: ScenarioConfig, *, stimulus: Any = None, stimuli: Any = None) -> tuple[list[float], list[Pose]]:
        selected_stimuli = self._resolve_stimuli(stimulus, stimuli)
        steps = int(round(scenario.duration_s / scenario.dt_s))
        # Keep a final sample at exactly duration_s even when a user supplied a
        # duration that is not an integer number of dt intervals.
        times = [min(scenario.duration_s, i * scenario.dt_s) for i in range(steps + 1)]
        if not times or times[-1] < scenario.duration_s - 1e-10:
            times.append(scenario.duration_s)
        wind = self.config.wind.with_override(scenario.wind_override)
        p = scenario.initial_pose.position_m
        velocity = scenario.initial_pose.velocity_mps
        yaw = scenario.initial_pose.yaw_rad
        yaw_rate = scenario.initial_pose.angular_velocity_radps[2]
        poses: list[Pose] = []
        last_t = 0.0
        kind = self._profile_kind(scenario)
        for index, t in enumerate(times):
            goal, yaw_goal, _, angular = self._profile_values(scenario, t)
            control_kind = kind
            if kind == "mission":
                phase, _, _ = self._mission_phase(scenario, t)
                control_kind = self._normalise_phase_kind(phase.get("kind", "hold"))
            if index == 0:
                p = scenario.initial_pose.position_m
                velocity = scenario.initial_pose.velocity_mps
                yaw = scenario.initial_pose.yaw_rad
                yaw_rate = scenario.initial_pose.angular_velocity_radps[2]
            else:
                dt = t - last_t
                field = self.wind_at(p, t, wind, stimuli=selected_stimuli, reference_pose=Pose(p))
                reference_velocity = self._profile_velocity(scenario, t)
                error = _sub(goal, p)
                # A critically damped-ish point-mass response is enough to
                # make the scene deterministic and inspectable while keeping
                # the body state an integration result, not a scripted pose.
                if control_kind in ("vertical_ramp", "ramp"):
                    kp, kd, wind_gain, max_accel = 4.0, 10.0, 0.18, 8.0
                elif control_kind in ("forward_ramp", "forward_target", "position_ramp", "translation_ramp", "waypoint_ramp", "target_approach"):
                    kp, kd, wind_gain, max_accel = 5.0, 7.0, 0.55, 6.0
                else:
                    kp, kd, wind_gain, max_accel = 3.4, 3.0, 0.82, 4.0
                acceleration = _add(_scale(error, kp), _add(_scale(_sub(reference_velocity, velocity), kd), _scale(field, wind_gain)))
                air = self.air_disturbance_at(p, t, stimuli=selected_stimuli, reference_pose=Pose(p))
                acceleration = _add(acceleration, air["acceleration_mps2"])
                acceleration_norm = _norm(acceleration)
                if acceleration_norm > max_accel:
                    acceleration = _scale(acceleration, max_accel / acceleration_norm)
                next_velocity = _add(velocity, _scale(acceleration, dt))
                next_position = _add(p, _add(_scale(velocity, dt), _scale(acceleration, 0.5 * dt * dt)))
                p, velocity = self._resolve_integrated_motion(next_position, next_velocity, t_s=t,
                                                               stimuli=selected_stimuli, reference_pose=Pose(p))

                # A yaw command has actuator dynamics too.  The profile's
                # rate is a requested rate; the pose stores the integrated
                # rate, preserving meaningful optic flow without teleporting
                # orientation at the step boundary.
                requested_rate = angular[2]
                yaw_rate += (requested_rate - yaw_rate) * _clamp(12.0 * dt)
                if control_kind != "yaw_step":
                    yaw_error = _wrap_angle(yaw_goal - yaw)
                    yaw_rate += (5.0 * yaw_error - 3.0 * yaw_rate) * dt
                # Integrate the updated angular rate exactly once per sample.
                yaw += yaw_rate * dt
            pose = Pose(p, yaw, scenario.initial_pose.pitch_rad, scenario.initial_pose.roll_rad,
                        velocity, (scenario.initial_pose.angular_velocity_radps[0],
                                   scenario.initial_pose.angular_velocity_radps[1], yaw_rate))
            poses.append(pose)
            last_t = t
        return times, poses

    @staticmethod
    def _horizontal_distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _evaluate_acceptance(self, acceptance: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
        checks: dict[str, dict[str, Any]] = {}
        violations: list[str] = []
        for key, limit_value in acceptance.items():
            if key in ("gust_start_s",):
                observed = bool(metrics.get("gust_observed", False))
                check = {"observed": observed, "requested_s": float(limit_value), "passed": observed}
            elif key.startswith("max_"):
                metric_key = key if key in metrics else key[4:]
                observed = float(metrics.get(metric_key, float("inf")))
                limit = float(limit_value)
                check = {"observed": observed, "limit": limit, "passed": observed <= limit + 1e-9}
            elif key.startswith("min_"):
                metric_key = key if key in metrics else key[4:]
                observed = float(metrics.get(metric_key, float("-inf")))
                limit = float(limit_value)
                check = {"observed": observed, "limit": limit, "passed": observed >= limit - 1e-9}
            elif key == "final_speed_mps":
                observed = float(metrics.get("final_speed_mps", float("inf")))
                limit = float(limit_value)
                check = {"observed": observed, "limit": limit, "passed": observed <= limit + 1e-9}
            else:
                # Unknown acceptance fields remain visible in the manifest but
                # are not silently treated as a pass/fail assertion.
                check = {"observed": metrics.get(key), "requested": limit_value, "passed": True, "ignored": True}
            checks[key] = check
            if not check["passed"]:
                violations.append(key)
        return {"passed": not violations, "checks": checks, "violations": violations}

    def run_scenario(self, scenario: str | ScenarioConfig | Mapping[str, Any], *, render: bool = False,
                     include_frames: bool | None = None, flow: bool = False, include_flow: bool | None = None,
                     resolution: Any = None, stimulus: Any = None, stimuli: Any = None) -> ScenarioResult:
        """Run one configured scenario and return reproducible telemetry."""
        if isinstance(scenario, str):
            lookup = scenario if scenario in self.scenarios else scenario.replace("_", "-")
            if lookup not in self.scenarios:
                raise KeyError(f"unknown scenario {scenario!r}; available: {sorted(self.scenarios)}")
            sc = self.scenarios[lookup]
        elif isinstance(scenario, Mapping):
            sc = ScenarioConfig.from_mapping(str(scenario.get("name", "custom")), scenario)
        else:
            sc = scenario
        if include_frames is not None:
            render = bool(include_frames)
        if include_flow is not None:
            flow = bool(include_flow)
        if flow and resolution is None:
            # Flow is usually used as a compact ground truth; the default
            # camera is still intentional and explicit in the result manifest.
            resolution = (self.camera.width, self.camera.height)
        selected_stimuli = self._resolve_stimuli(stimulus, stimuli)
        times, poses = self._trajectory(sc, stimuli=selected_stimuli)
        observations = [self.target_observation(p, t, stimuli=selected_stimuli) for p, t in zip(poses, times)]
        looming_samples = [self.looming_at(p, t_s=t, stimuli=selected_stimuli) for p, t in zip(poses, times)]
        wind_model = self.config.wind.with_override(sc.wind_override)
        wind_samples = [self.wind_at(p.position_m, t, wind_model, stimuli=selected_stimuli) for p, t in zip(poses, times)]
        goal_samples = [self.goal_at(sc, t) for t in times]
        reference_positions = [tuple(goal["position_m"]) for goal in goal_samples]
        stimulus_observations: list[dict[str, StimulusObservation]] = []
        for index, (pose, t) in enumerate(zip(poses, times)):
            previous_t = times[index - 1] if index else None
            dt = (t - previous_t) if previous_t is not None else None
            row: dict[str, StimulusObservation] = {}
            for item in selected_stimuli:
                row[item.id] = self.stimulus_observation(pose, item, t, previous_t_s=previous_t, dt_s=dt)
            stimulus_observations.append(row)
        stimulus_contact_times = {
            item.id: (item.counterfactual_contact_time_s if item.counterfactual_contact_time_s is not None else
                       item.contact_time_s(reference_position_m=poses[0].position_m if poses else sc.initial_pose.position_m,
                                           radius_m=item.contact_radius_m if item.contact_radius_m is not None else self.config.body_radius_m,
                                           reference_pose=poses[0] if poses else sc.initial_pose))
            for item in selected_stimuli
        }
        collisions: list[CollisionEvent] = []
        contacts: list[CollisionEvent] = []
        collision_counts: list[int] = []
        contact_counts: list[int] = []
        for t, p in zip(times, poses):
            hard, soft = self.collision_events(p, t, stimuli=selected_stimuli)
            collisions.extend(hard)
            contacts.extend(soft)
            collision_counts.append(len(hard))
            contact_counts.append(len(soft))
        profile_goal = reference_positions[-1] if reference_positions else sc.initial_pose.position_m
        initial = poses[0]
        phase_labels: list[str] = []
        phase_timings: dict[str, dict[str, float]] = {}
        for goal in goal_samples:
            label = str(goal["phase"])
            if not phase_labels or phase_labels[-1] != label:
                phase_labels.append(label)
            phase_timings[label] = {
                "start_s": float(goal["phase_start_s"]),
                "end_s": float(goal["phase_end_s"]),
                "duration_s": float(goal["phase_end_s"] - goal["phase_start_s"]),
            }
        target_visible_fraction = sum(1 for item in observations if item.visible) / max(1, len(observations))
        target_on_target_fraction = sum(1 for item in observations if item.on_target) / max(1, len(observations))
        target_acquisition_time = next((float(t) for t, item in zip(times, observations) if item.visible), None)
        max_wind_speed = max((_norm(w) for w in wind_samples), default=0.0)
        max_wind_component = max((max(abs(value) for value in w) for w in wind_samples), default=0.0)
        mean_wind_speed = sum(_norm(w) for w in wind_samples) / max(1, len(wind_samples))
        max_altitude = max((p.position_m[2] - self.config.ground_z_m for p in poses), default=0.0)
        initial_altitude = initial.position_m[2] - self.config.ground_z_m if poses else 0.0
        horizontal_displacement = self._horizontal_distance(poses[-1].position_m, initial.position_m) if poses else 0.0
        metrics: dict[str, Any] = {
            "sample_count": len(times),
            "duration_s": sc.duration_s,
            "dt_s": sc.dt_s,
            "collision_count": len(collisions),
            "contact_count": len(contacts),
            "max_altitude_m": max_altitude,
            "min_altitude_m": min((p.position_m[2] - self.config.ground_z_m for p in poses), default=0.0),
            "max_horizontal_drift_m": max((self._horizontal_distance(p.position_m, reference) for p, reference in zip(poses, reference_positions)), default=0.0),
            "max_altitude_error_m": max((abs(p.position_m[2] - reference[2]) for p, reference in zip(poses, reference_positions)), default=0.0),
            "final_altitude_m": poses[-1].position_m[2] - self.config.ground_z_m if poses else 0.0,
            "final_speed_mps": poses[-1].speed_mps if poses else 0.0,
            "max_speed_mps": max((p.speed_mps for p in poses), default=0.0),
            **dict.fromkeys(("altitude_gain_m", "min_altitude_gain_m"), max_altitude - initial_altitude),
            **dict.fromkeys(("forward_displacement_m", "min_forward_displacement_m", "horizontal_displacement_m"), horizontal_displacement),
            "final_target_distance_m": observations[-1].distance_m if observations else float("inf"),
            "min_target_distance_m": min((o.distance_m for o in observations), default=float("inf")),
            "max_abs_target_bearing_rad": max((abs(o.bearing_rad) for o in observations), default=0.0),
            **dict.fromkeys(("target_visible_fraction", "min_target_visibility_fraction", "target_visibility_fraction"), target_visible_fraction),
            "target_on_target_fraction": target_on_target_fraction,
            "target_acquisition_time_s": target_acquisition_time,
            "target_id": self.config.target.id,
            "target_position_m": list(self.config.target.position_m),
            **dict.fromkeys(("max_wind_speed_mps", "wind_peak_speed_mps"), max_wind_speed),
            "mean_wind_speed_mps": mean_wind_speed,
            "max_wind_component_mps": max_wind_component,
            "yaw_change_rad": abs(_wrap_angle(poses[-1].yaw_rad - initial.yaw_rad)) if poses else 0.0,
            "gust_observed": bool(sc.wind_override and sc.wind_override.get("gusts")),
            "gust_active_sample_count": sum(1 for t in times if any(g.weight(t) > 0.0 for g in wind_model.gusts)),
            "reference_final_position_m": list(profile_goal),
            "phase_labels": phase_labels,
            "phase_timings": phase_timings,
            "phase_transition_times_s": [phase_timings[label]["start_s"] for label in phase_labels],
            **dict.fromkeys(("max_looming_rate_rad_s", "max_looming_rate_radps"), max((float(cue["angular_rate_rad_s"]) for sample in looming_samples for cue in sample.values()), default=0.0)),
            "stimulus_ids": [item.id for item in selected_stimuli],
            "active_stimulus_count": sum(sum(1 for observation in row.values() if observation.active) for row in stimulus_observations),
            "stimulus_collision_count": sum(sum(1 for observation in row.values() if observation.colliding) for row in stimulus_observations),
            "max_stimulus_angular_rate_rad_s": max((observation.angular_rate_rad_s for row in stimulus_observations for observation in row.values()), default=0.0),
            "stimulus_contact_time_s": stimulus_contact_times,
            "first_stimulus_contact_time_s": min((value for value in stimulus_contact_times.values() if value is not None), default=None),
            "stimulus_visual_only": {item.id: item.visual_only for item in selected_stimuli},
            "stimulus_non_interacting": {item.id: item.non_interacting for item in selected_stimuli},
        }
        telemetry = []
        for index, (t, pose, observation, wind_sample, reference, goal) in enumerate(
            zip(times, poses, observations, wind_samples, reference_positions, goal_samples)
        ):
            gust_active = any(g.weight(t) > 0.0 for g in wind_model.gusts)
            telemetry.append({
                "time_s": float(t),
                "t_s": float(t),
                "scenario": sc.name,
                "source": "handbuilt",
                "position_m": list(pose.position_m),
                "orientation_rad": list(pose.orientation_rpy_rad),
                "velocity_mps": list(pose.velocity_mps),
                "angular_velocity_rad_s": list(pose.angular_velocity_rad_s),
                "speed_mps": float(pose.speed_mps),
                "reference_position_m": list(reference),
                "goal_position_m": list(goal["position_m"]),
                "goal_velocity_mps": list(goal["velocity_mps"]),
                "goal_yaw_rad": float(goal["yaw_rad"]),
                "goal_yaw_rate_radps": float(goal["yaw_rate_radps"]),
                "phase": goal["phase"],
                "phase_index": int(goal["phase_index"]),
                "phase_kind": goal["phase_kind"],
                "phase_start_s": float(goal["phase_start_s"]),
                "phase_end_s": float(goal["phase_end_s"]),
                "target_position_m": list(self.config.target.position_m),
                "target": observation.to_dict(),
                "target_distance_m": float(observation.distance_m),
                "target_bearing_rad": float(observation.bearing_rad),
                "target_elevation_rad": float(observation.elevation_rad),
                "target_visible": bool(observation.visible),
                "target_on_target": bool(observation.on_target),
                "target_contact": bool(
                    observation.on_target
                    and pose.position_m[2] - self.config.body_radius_m <= self.config.target.max_m[2] + 1.0e-8
                ),
                "wind_mps": list(wind_sample),
                "wind_speed_mps": float(_norm(wind_sample)),
                "gust_active": gust_active,
                "collision_count": collision_counts[index],
                "contact_count": contact_counts[index],
                "contact": bool(contact_counts[index]),
                "grounded": bool(pose.position_m[2] - self.config.body_radius_m <= self.config.ground_z_m + 1.0e-8),
                "stimuli": {name: observation.to_dict() for name, observation in stimulus_observations[index].items()},
                "stimulus_states": self.stimulus_state_at(t, stimuli=selected_stimuli, reference_pose=pose),
            })
        acceptance = self._evaluate_acceptance(sc.acceptance, metrics)
        frames: Optional[list[StereoFrames]] = None
        if render:
            frames = [self.render_stereo(p, resolution, timestamp_s=t, stimuli=selected_stimuli) for p, t in zip(poses, times)]
        optic: Optional[list[dict[str, OpticFlow]]] = None
        if flow:
            optic = []
            for i in range(len(poses) - 1):
                dt = times[i + 1] - times[i]
                optic.append({"left": self.optic_flow_ground_truth(poses[i], poses[i + 1], dt_s=dt, eye="left", resolution=resolution,
                                                                      timestamp_s=times[i], stimuli=selected_stimuli),
                              "right": self.optic_flow_ground_truth(poses[i], poses[i + 1], dt_s=dt, eye="right", resolution=resolution,
                                                                       timestamp_s=times[i], stimuli=selected_stimuli)})
            metrics["max_flow_magnitude_px"] = max((f.max_magnitude_px for pair in optic for f in pair.values()), default=0.0)
        else:
            metrics["max_flow_magnitude_px"] = 0.0
        manifest = self.manifest(sc, resolution=resolution, stimuli=selected_stimuli)
        manifest["result"] = {"acceptance": acceptance, "rendered": bool(render), "optic_flow": bool(flow),
                               "sample_count": len(times), "collision_count": len(collisions),
                               "target": self.config.target.to_dict(),
                               "wind": wind_model.to_dict(),
                               "stimulus_ids": [item.id for item in selected_stimuli],
                               "stimulus_contact_time_s": stimulus_contact_times,
                               "stimulus_visual_only": {item.id: item.visual_only for item in selected_stimuli},
                               "stimulus_non_interacting": {item.id: item.non_interacting for item in selected_stimuli},
                               "phases": phase_timings,
                               "trajectory_fields": sorted(telemetry[0]) if telemetry else []}
        result_payload = {"name": sc.name, "times_s": times, "poses": [p.to_dict() for p in poses],
                          "observations": [o.to_dict() for o in observations],
                          "wind_mps": [list(w) for w in wind_samples],
                          "collisions": [c.to_dict() for c in collisions],
                          "stimuli": [self.stimulus_state_at(t, stimuli=selected_stimuli, reference_pose=pose)
                                      for t, pose in zip(times, poses)]}
        metrics["trajectory_sha256"] = hashlib.sha256(json.dumps(result_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        return ScenarioResult(sc.name, self.seed, times, poses, observations, collisions, contacts, metrics, acceptance, manifest,
                              frames, optic, wind_samples, telemetry)

    simulate = run_scenario
    run = run_scenario

    def run_all(self, *, render: bool = False, flow: bool = False, resolution: Any = None) -> dict[str, ScenarioResult]:
        return {name: self.run_scenario(name, render=render, flow=flow, resolution=resolution) for name in sorted(self.scenarios)}

    # ------------------------------------------------------------------
    # Adapters for the shared ``twin_contracts`` records.  The import is
    # lazy so this module remains usable as a standalone script.
    def _contract_timestamp(self, t_s: float, dt_s: float, *, clock: str = "render") -> Any:
        try:
            from twin_contracts import Timestamp  # type: ignore
        except ImportError:
            return {"t_s": float(t_s), "tick": int(round(t_s / dt_s)), "clock": clock, "rate_hz": 1.0 / dt_s}
        return Timestamp(float(t_s), int(round(t_s / dt_s)), clock, 1.0 / dt_s)

    def body_state(self, pose: Pose, t_s: float = 0.0, *, dt_s: float = 0.02) -> Any:
        """Convert a pose to the canonical BodyState (or a JSON-like fallback)."""
        pose = _coerce_pose(pose)
        ts = self._contract_timestamp(t_s, dt_s)
        # twin_contracts uses roll, pitch, yaw ordering; Pose exposes yaw,
        # pitch, roll to match the camera convention.
        try:
            from twin_contracts import BodyState, SourceLabel  # type: ignore
            return BodyState(timestamp=ts, position_m=pose.position_m,
                             orientation_rad=(pose.roll_rad, pose.pitch_rad, pose.yaw_rad),
                             velocity_mps=pose.velocity_mps,
                             angular_velocity_rad_s=pose.angular_velocity_radps,
                             source=SourceLabel.HANDBUILT)
        except ImportError:
            return {"timestamp": ts, "position_m": list(pose.position_m),
                    "orientation_rad": [pose.roll_rad, pose.pitch_rad, pose.yaw_rad],
                    "velocity_mps": list(pose.velocity_mps), "angular_velocity_rad_s": list(pose.angular_velocity_radps),
                    "source": "handbuilt"}

    def world_state(self, pose: Pose, t_s: float = 0.0, *, dt_s: float = 0.02) -> Any:
        body = self.body_state(pose, t_s, dt_s=dt_s)
        ts = self._contract_timestamp(t_s, dt_s)
        try:
            from twin_contracts import WorldState, SourceLabel  # type: ignore
            return WorldState(timestamp=ts, body=body, world_id="virtual-world", geometry=self.manifest(), source=SourceLabel.HANDBUILT)
        except ImportError:
            return {"timestamp": ts, "body": body, "world_id": "virtual-world", "geometry": self.manifest(), "source": "handbuilt"}

    def _world_state_at_timestamp(self, pose: Pose, timestamp: Any) -> Any:
        """Build a state using an already-owned Timestamp without retiming it."""
        try:
            from twin_contracts import BodyState, WorldState, SourceLabel  # type: ignore
            body = BodyState(timestamp=timestamp, position_m=pose.position_m,
                             orientation_rad=pose.orientation_rpy_rad,
                             velocity_mps=pose.velocity_mps,
                             angular_velocity_rad_s=pose.angular_velocity_rad_s,
                             source=SourceLabel.HANDBUILT)
            return WorldState(timestamp=timestamp, body=body, world_id="virtual-world",
                              geometry=self.manifest(), source=SourceLabel.HANDBUILT)
        except ImportError:
            return {"timestamp": timestamp, "body": {"timestamp": timestamp, "position_m": list(pose.position_m),
                    "orientation_rad": list(pose.orientation_rpy_rad), "velocity_mps": list(pose.velocity_mps),
                    "angular_velocity_rad_s": list(pose.angular_velocity_rad_s), "source": "handbuilt"},
                    "world_id": "virtual-world", "geometry": self.manifest(), "source": "handbuilt"}

    def reset(self, state: Any = None, *, seed: Optional[int] = None) -> Any:
        """WorldModel-compatible reset for the shared runtime.

        Scenario generation remains the preferred batch API.  This small
        stateful adapter lets the same scene be injected into the runtime's
        ``WorldModel`` protocol without importing a physics engine.
        """
        if seed is not None:
            self.seed = int(seed)
        if state is None:
            if "hover" in self.scenarios:
                pose = self.scenarios["hover"].initial_pose
            else:
                pose = Pose((0.0, 0.0, self.config.ground_z_m + self.config.body_radius_m))
            t_s, dt_s = 0.0, 0.02
        else:
            pose = _coerce_pose(state)
            stamp = getattr(state, "timestamp", None)
            t_s = float(getattr(stamp, "t_s", 0.0))
            rate = float(getattr(stamp, "rate_hz", 1.0 / 0.02))
            dt_s = 1.0 / rate if rate > 0.0 else 0.02
        stamp = getattr(state, "timestamp", None) if state is not None else None
        self._state = self._world_state_at_timestamp(pose, stamp) if stamp is not None else self.world_state(pose, t_s, dt_s=dt_s)
        self._observe_prior_pose = None
        return self._state

    def step(self, command: Any, timestamp: Any) -> Any:
        """Advance the lightweight point-mass adapter by one contract tick."""
        if self._state is None:
            self.reset()
        pose = _coerce_pose(self._state)
        old_t = float(getattr(getattr(self._state, "timestamp", None), "t_s", 0.0))
        t_s = float(getattr(timestamp, "t_s", timestamp))
        dt = max(0.0, t_s - old_t)
        yaw_rate = float(getattr(command, "yaw_rate_rad_s", command.get("yaw_rate_rad_s", command.get("yaw", 0.0)) if isinstance(command, Mapping) else 0.0))
        pitch_rate = float(getattr(command, "pitch_rate_rad_s", command.get("pitch_rate_rad_s", command.get("pitch", 0.0)) if isinstance(command, Mapping) else 0.0))
        roll_rate = float(getattr(command, "roll_rate_rad_s", command.get("roll_rate_rad_s", command.get("roll", 0.0)) if isinstance(command, Mapping) else 0.0))
        thrust = float(getattr(command, "thrust_mps2", command.get("thrust_mps2", command.get("thrust", 0.0)) if isinstance(command, Mapping) else 0.0))
        vx, vy, vz = pose.velocity_mps
        vz = max(-0.5, min(0.5, vz + thrust * dt))
        velocity = (vx * max(0.0, 1.0 - 2.0 * dt), vy * max(0.0, 1.0 - 2.0 * dt), vz)
        p = _add(pose.position_m, _scale(velocity, dt))
        p = (p[0], p[1], max(self.config.ground_z_m + self.config.body_radius_m, p[2]))
        orientation = (pose.roll_rad + roll_rate * dt, pose.pitch_rad + pitch_rate * dt, pose.yaw_rad + yaw_rate * dt)
        next_pose = Pose(p, orientation[2], orientation[1], orientation[0], velocity, (roll_rate, pitch_rate, yaw_rate))
        if hasattr(timestamp, "t_s"):
            self._state = self._world_state_at_timestamp(next_pose, timestamp)
        else:
            rate = float(getattr(timestamp, "rate_hz", 1.0 / max(dt, 0.02)))
            self._state = self.world_state(next_pose, t_s, dt_s=1.0 / rate if rate > 0 else 0.02)
        return self._state

    def visual_observation(self, stereo: StereoFrames, t_s: float = 0.0, *, dt_s: float = 0.02,
                            optic_flow_rad_s: Sequence[float] = (0.0, 0.0, 0.0), metadata: Mapping[str, Any] | None = None) -> Any:
        ts = self._contract_timestamp(t_s, dt_s)
        meta = {"left_pixels": stereo.left, "right_pixels": stereo.right, "eye_separation_m": stereo.eye_separation_m}
        if metadata:
            meta.update(metadata)
        try:
            from twin_contracts import VisualObservation, SourceLabel  # type: ignore
            # The canonical contract carries one pixel array.  Left is the
            # primary channel and the right eye is preserved in metadata.
            return VisualObservation(timestamp=ts, pixels=stereo.left, optic_flow_rad_s=optic_flow_rad_s,
                                     sensor_id="virtual-world-stereo-left", source=SourceLabel.HANDBUILT, metadata=meta)
        except ImportError:
            return {"timestamp": ts, "pixels": stereo.left, "optic_flow_rad_s": list(optic_flow_rad_s),
                    "sensor_id": "virtual-world-stereo-left", "source": "handbuilt", "metadata": meta}

    def observe(self, world: Any, timestamp: Any) -> Any:
        """Sensor-compatible observation of a WorldState or Pose."""
        pose = _coerce_pose(world)
        t_s = float(getattr(timestamp, "t_s", timestamp))
        rate = float(getattr(timestamp, "rate_hz", 25.0))
        dt_s = 1.0 / rate if rate > 0.0 else 0.04
        stereo = self.render_stereo(pose, timestamp_s=t_s)
        flow_vector = (0.0, 0.0, 0.0)
        prior = getattr(self, "_observe_prior_pose", None)
        if prior is not None and dt_s > 0.0:
            flow = self.optic_flow_ground_truth(prior, pose, dt_s=dt_s, eye="left")
            values = [v for row, good in zip(flow.flow_rad_s, flow.valid) for v, ok in zip(row, good) if ok]
            if values:
                flow_vector = (sum(v[0] for v in values) / len(values), sum(v[1] for v in values) / len(values), 0.0)
        self._observe_prior_pose = pose
        if hasattr(timestamp, "t_s"):
            try:
                from twin_contracts import VisualObservation, SourceLabel  # type: ignore
                return VisualObservation(timestamp=timestamp, pixels=stereo.left, optic_flow_rad_s=flow_vector,
                                         sensor_id="virtual-world-stereo-left", source=SourceLabel.HANDBUILT,
                                         metadata={"right_pixels": stereo.right, "eye_separation_m": stereo.eye_separation_m})
            except ImportError:
                pass
        return self.visual_observation(stereo, t_s, dt_s=dt_s, optic_flow_rad_s=flow_vector)


def load_world(path: str | Path = _DEFAULT_CONFIG, **kwargs: Any) -> VirtualWorld:
    """Functional loader used by small scripts and tests."""
    return VirtualWorld.from_json(path, **kwargs)


# Small vocabulary aliases keep adapters readable without multiplying the
# implementation: a landing pad is represented by the same textured box as a
# target, and a scenario is the parsed ScenarioConfig record.
SceneConfig = WorldConfig
LandingPad = Target
Scenario = ScenarioConfig
World = VirtualWorld


def self_test(path: str | Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    """Run compact deterministic checks without producing project artefacts."""
    world = VirtualWorld.from_json(path, camera=CameraConfig(16, 12))
    names = ("takeoff", "hover", "yaw-step", "forward-target", "gust-recovery", "landing", "flight-demo", "full-mission")
    results = {name: world.run_scenario(name) for name in names}
    repeat = world.run_scenario("hover")
    deterministic = repeat.metrics["trajectory_sha256"] == results["hover"].metrics["trajectory_sha256"]
    stereo = world.render_stereo(results["hover"].poses[0], (16, 12))
    stereo_nonidentical = stereo.left != stereo.right
    flow = world.optic_flow_ground_truth(results["yaw-step"].poses[20], results["yaw-step"].poses[80], dt_s=1.2, resolution=(16, 12))
    all_passed = all(r.acceptance["passed"] for r in results.values()) and deterministic and stereo_nonidentical and flow.max_magnitude_px > 0.0
    return {"passed": all_passed, "scenarios": {name: {"passed": r.acceptance["passed"], "metrics": r.metrics, "violations": r.acceptance["violations"]} for name, r in results.items()},
            "deterministic": deterministic, "stereo_nonidentical": stereo_nonidentical, "flow_max_magnitude_px": flow.max_magnitude_px,
            "manifest_sha256": world.manifest()["config_sha256"]}


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--scenario", default=None, help="scenario name; omit to print all summaries")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--render", "--frames", dest="render", action="store_true", help="include left/right frames in JSON output")
    parser.add_argument("--flow", "--flow-ground-truth", dest="flow", action="store_true", help="include optic-flow ground truth")
    parser.add_argument("--manifest-only", "--manifest", dest="manifest_only", action="store_true", help="print the stable world manifest")
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="write result JSON to this path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        payload = self_test(args.config)
    else:
        world = VirtualWorld.from_json(args.config, camera=CameraConfig(args.width, args.height))
        manifest = world.manifest(args.scenario, resolution=(args.width, args.height))
        if args.manifest_out is not None:
            args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
            args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if args.manifest_only:
            payload = manifest
        elif args.scenario is None:
            payload = {"manifest": manifest, "scenarios": {name: world.run_scenario(name).to_dict() for name in sorted(world.scenarios)}}
        else:
            result = world.run_scenario(args.scenario, render=args.render, flow=args.flow, resolution=(args.width, args.height))
            payload = result.to_dict(include_frames=args.render, include_flow=args.flow)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
