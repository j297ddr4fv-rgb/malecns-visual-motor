#!/usr/bin/env python3
"""World-anchored moving female-fly visual target for courtship assays.

The target uses the existing :class:`virtual_world.PhysicalStimulus` seam so
the CPU and Warp renderers can bind a metre-scale local glTF mesh to its
stimulus ID.  Geometry metadata and the trajectory are separate inputs: the
mesh describes what the target looks like, while this module only samples a
time-indexed pose.  The trajectory never reads or follows the male pose.

Local mesh convention follows the renderer's stimulus-mesh contract:
``+X`` is forward, ``+Y`` is lateral, ``+Z`` is up, the origin is at the
thorax centre, and units are metres.  The supplied geometry config records an
adult female Drosophila scale and semantic body-part metadata for matching an
external GLB asset.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from virtual_world import PhysicalStimulus, StimulusState  # type: ignore
except ImportError:  # pragma: no cover - package-style import convenience.
    from .virtual_world import PhysicalStimulus, StimulusState  # type: ignore


_ROOT = Path(__file__).resolve().parents[1]
_GEOMETRY_DEFAULT = _ROOT / "config" / "courtship_target_geometry.json"
_TRAJECTORY_DEFAULT = _ROOT / "config" / "courtship_target_trajectory.json"
_TWO_PI = 2.0 * math.pi
_EPS = 1.0e-12


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vec3(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly three finite numbers") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(values))  # type: ignore[return-value]


def _wrap_angle(angle: float) -> float:
    wrapped = (angle + math.pi) % _TWO_PI - math.pi
    return -math.pi if wrapped == math.pi else wrapped


def _read_json(source: str | Path) -> Mapping[str, Any]:
    path = Path(source).expanduser()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read courtship target config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid courtship target JSON in {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"courtship target config {path} must contain a JSON object")
    return loaded


@dataclass(frozen=True)
class TrajectoryKeyframe:
    """One absolute-time, world-coordinate target pose."""

    t_s: float
    position_m: tuple[float, float, float]
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_s", _finite(self.t_s, "keyframe.t_s"))
        object.__setattr__(self, "position_m", _vec3(self.position_m, "keyframe.position_m"))
        for name in ("yaw_rad", "pitch_rad", "roll_rad"):
            object.__setattr__(self, name, _finite(getattr(self, name), f"keyframe.{name}"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("keyframe.metadata must be an object")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrajectoryKeyframe":
        if not isinstance(value, Mapping):
            raise ValueError("each trajectory keyframe must be a JSON object")
        orientation = value.get("orientation_rpy_rad")
        if orientation is not None:
            roll, pitch, yaw = _vec3(orientation, "keyframe.orientation_rpy_rad")
        else:
            yaw = value.get("yaw_rad", 0.0)
            pitch = value.get("pitch_rad", 0.0)
            roll = value.get("roll_rad", 0.0)
        known = {"t_s", "time_s", "position_m", "yaw_rad", "pitch_rad", "roll_rad",
                 "orientation_rpy_rad"}
        return cls(
            t_s=value.get("t_s", value.get("time_s")),
            position_m=value.get("position_m"),
            yaw_rad=yaw,
            pitch_rad=pitch,
            roll_rad=roll,
            metadata={key: item for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.metadata)
        out.update({
            "t_s": self.t_s,
            "position_m": list(self.position_m),
            "yaw_rad": self.yaw_rad,
            "pitch_rad": self.pitch_rad,
            "roll_rad": self.roll_rad,
        })
        return out


@dataclass(frozen=True)
class FemaleTargetPose:
    """Interpolated target pose and translational/angular velocity."""

    t_s: float
    position_m: tuple[float, float, float]
    yaw_rad: float
    pitch_rad: float
    roll_rad: float
    velocity_mps: tuple[float, float, float]
    angular_velocity_radps: tuple[float, float, float]

    @property
    def direction_m(self) -> tuple[float, float, float]:
        """Forward axis used by VirtualWorld's local stimulus-mesh transform."""
        cp = math.cos(self.pitch_rad)
        return (cp * math.cos(self.yaw_rad), cp * math.sin(self.yaw_rad), math.sin(self.pitch_rad))

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_s": self.t_s,
            "position_m": list(self.position_m),
            "yaw_rad": self.yaw_rad,
            "pitch_rad": self.pitch_rad,
            "roll_rad": self.roll_rad,
            "velocity_mps": list(self.velocity_mps),
            "angular_velocity_radps": list(self.angular_velocity_radps),
            "direction_m": list(self.direction_m),
        }


@dataclass(frozen=True)
class FemaleTargetTrajectory:
    """Immutable, piecewise-linear target trajectory in absolute world time."""

    keyframes: tuple[TrajectoryKeyframe, ...]
    interpolation: str = "linear"
    trajectory_id: str = "female_courtship_default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = tuple(frame if isinstance(frame, TrajectoryKeyframe)
                       else TrajectoryKeyframe.from_mapping(frame) for frame in self.keyframes)
        if not frames:
            raise ValueError("trajectory.keyframes must contain at least one keyframe")
        if any(frames[i].t_s >= frames[i + 1].t_s for i in range(len(frames) - 1)):
            raise ValueError("trajectory keyframe times must be strictly increasing")
        method = str(self.interpolation).strip().lower()
        if method not in {"linear", "linear_shortest_angle"}:
            raise ValueError("trajectory interpolation must be 'linear' or 'linear_shortest_angle'")
        object.__setattr__(self, "keyframes", frames)
        object.__setattr__(self, "interpolation", "linear_shortest_angle")
        object.__setattr__(self, "trajectory_id", str(self.trajectory_id))
        if not self.trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("trajectory.metadata must be an object")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FemaleTargetTrajectory":
        if not isinstance(value, Mapping):
            raise ValueError("trajectory config must be a JSON object")
        frames = value.get("keyframes")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise ValueError("trajectory.keyframes must be a list")
        known = {"schema_version", "trajectory_id", "id", "interpolation", "keyframes"}
        return cls(
            tuple(TrajectoryKeyframe.from_mapping(item) for item in frames),
            str(value.get("interpolation", "linear_shortest_angle")),
            str(value.get("trajectory_id", value.get("id", "female_courtship_default"))),
            {key: item for key, item in value.items() if key not in known},
        )

    @classmethod
    def from_json(cls, source: str | Path | Mapping[str, Any] | None = None) -> "FemaleTargetTrajectory":
        if source is None:
            source = _TRAJECTORY_DEFAULT
        if isinstance(source, Mapping):
            return cls.from_mapping(source)
        return cls.from_mapping(_read_json(source))

    @property
    def start_time_s(self) -> float:
        return self.keyframes[0].t_s

    @property
    def end_time_s(self) -> float:
        return self.keyframes[-1].t_s

    def sample(self, t_s: float) -> FemaleTargetPose:
        """Sample with linear position and shortest-arc Euler interpolation.

        Values outside the keyframe range clamp to the endpoint pose with
        zero velocity. At interior keyframes the following segment supplies
        the piecewise-constant velocity.
        """
        t = _finite(t_s, "t_s")
        if len(self.keyframes) == 1 or t <= self.keyframes[0].t_s:
            frame = self.keyframes[0]
            return FemaleTargetPose(t, frame.position_m, frame.yaw_rad, frame.pitch_rad,
                                    frame.roll_rad, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if t >= self.keyframes[-1].t_s:
            frame = self.keyframes[-1]
            return FemaleTargetPose(t, frame.position_m, frame.yaw_rad, frame.pitch_rad,
                                    frame.roll_rad, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        times = [frame.t_s for frame in self.keyframes]
        right_index = bisect.bisect_right(times, t)
        left, right = self.keyframes[right_index - 1], self.keyframes[right_index]
        dt = right.t_s - left.t_s
        u = (t - left.t_s) / dt
        velocity = tuple((right.position_m[i] - left.position_m[i]) / dt for i in range(3))

        def angle_pair(a: float, b: float) -> tuple[float, float]:
            delta = _wrap_angle(b - a)
            return _wrap_angle(a + u * delta), delta / dt

        yaw, yaw_rate = angle_pair(left.yaw_rad, right.yaw_rad)
        pitch, pitch_rate = angle_pair(left.pitch_rad, right.pitch_rad)
        roll, roll_rate = angle_pair(left.roll_rad, right.roll_rad)
        position = tuple(left.position_m[i] + u * (right.position_m[i] - left.position_m[i])
                         for i in range(3))
        return FemaleTargetPose(t, position, yaw, pitch, roll, velocity,
                                (roll_rate, pitch_rate, yaw_rate))

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.metadata)
        out.update({
            "schema_version": 1,
            "trajectory_id": self.trajectory_id,
            "interpolation": self.interpolation,
            "time_basis": "absolute simulation seconds",
            "coordinate_frame": "world",
            "keyframes": [frame.to_dict() for frame in self.keyframes],
        })
        return out


@dataclass(frozen=True)
class FemaleTargetGeometry:
    """Renderer-independent dimensions and semantic asset metadata."""

    target_extent_m: tuple[float, float, float]
    body_length_m: float
    body_width_m: float
    body_height_m: float
    wingspan_m: float
    asset_object_id: str = "female_fly_mesh"
    semantic_role: str = "adult_female_drosophila_visual_target"
    local_origin: str = "thorax_center"
    forward_axis: str = "+X"
    lateral_axis: str = "+Y"
    up_axis: str = "+Z"
    geometry_id: str = "adult_female_drosophila_mm_scale"
    body_parts: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        extent = _vec3(self.target_extent_m, "target_extent_m")
        if any(v <= 0.0 for v in extent):
            raise ValueError("target_extent_m dimensions must be positive")
        object.__setattr__(self, "target_extent_m", extent)
        for name in ("body_length_m", "body_width_m", "body_height_m", "wingspan_m"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("asset_object_id", "semantic_role", "local_origin", "forward_axis",
                     "lateral_axis", "up_axis", "geometry_id"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        parts: list[dict[str, Any]] = []
        names: set[str] = set()
        for index, raw in enumerate(self.body_parts):
            if not isinstance(raw, Mapping):
                raise ValueError(f"body_parts[{index}] must be an object")
            part = dict(raw)
            name = str(part.get("name", ""))
            if not name or name in names:
                raise ValueError("body part names must be non-empty and unique")
            names.add(name)
            primitive = str(part.get("primitive", "ellipsoid"))
            if primitive != "ellipsoid":
                raise ValueError(f"body_parts[{index}].primitive must be 'ellipsoid'")
            center = _vec3(part.get("center_m"), f"body_parts[{index}].center_m")
            radii = _vec3(part.get("radii_m"), f"body_parts[{index}].radii_m")
            if any(value <= 0.0 for value in radii):
                raise ValueError(f"body_parts[{index}].radii_m dimensions must be positive")
            part.update({"name": name, "primitive": primitive,
                         "center_m": list(center), "radii_m": list(radii)})
            parts.append(part)
        object.__setattr__(self, "body_parts", tuple(parts))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FemaleTargetGeometry":
        if not isinstance(value, Mapping):
            raise ValueError("geometry config must be a JSON object")
        dimensions = value.get("dimensions_m", {})
        if not isinstance(dimensions, Mapping):
            raise ValueError("geometry.dimensions_m must be an object")
        extent = value.get("target_extent_m", dimensions.get("target_extent_m"))
        if extent is None:
            raise ValueError("geometry.target_extent_m is required")
        asset = value.get("asset", {})
        if not isinstance(asset, Mapping):
            raise ValueError("geometry.asset must be an object")
        semantic = value.get("semantic", {})
        if not isinstance(semantic, Mapping):
            raise ValueError("geometry.semantic must be an object")
        body_parts = value.get("body_parts", ())
        if not isinstance(body_parts, Sequence) or isinstance(body_parts, (str, bytes)):
            raise ValueError("geometry.body_parts must be a list")
        return cls(
            target_extent_m=extent,
            body_length_m=dimensions.get("body_length_m"),
            body_width_m=dimensions.get("body_width_m"),
            body_height_m=dimensions.get("body_height_m"),
            wingspan_m=dimensions.get("wingspan_m"),
            asset_object_id=asset.get("object_id", "female_fly_mesh"),
            semantic_role=semantic.get("role", "adult_female_drosophila_visual_target"),
            local_origin=asset.get("local_origin", "thorax_center"),
            forward_axis=asset.get("forward_axis", "+X"),
            lateral_axis=asset.get("lateral_axis", "+Y"),
            up_axis=asset.get("up_axis", "+Z"),
            geometry_id=str(value.get("geometry_id", "adult_female_drosophila_mm_scale")),
            body_parts=tuple(body_parts),
        )

    @classmethod
    def from_json(cls, source: str | Path | Mapping[str, Any] | None = None) -> "FemaleTargetGeometry":
        if source is None:
            source = _GEOMETRY_DEFAULT
        if isinstance(source, Mapping):
            return cls.from_mapping(source)
        return cls.from_mapping(_read_json(source))

    @property
    def stimulus_size_m(self) -> tuple[float, float, float]:
        """Bounding dimensions in PhysicalStimulus's width/height/thickness order."""
        length, lateral, height = self.target_extent_m
        return (lateral, height, length)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "geometry_id": self.geometry_id,
            "semantic": {"role": self.semantic_role, "sex": "female", "species": "Drosophila melanogaster"},
            "dimensions_m": {
                "body_length_m": self.body_length_m,
                "body_width_m": self.body_width_m,
                "body_height_m": self.body_height_m,
                "wingspan_m": self.wingspan_m,
            },
            "target_extent_m": list(self.target_extent_m),
            "body_parts": [dict(part) for part in self.body_parts],
            "asset": {
                "object_id": self.asset_object_id,
                "format": "glTF 2.0 / GLB",
                "units": "metres",
                "local_origin": self.local_origin,
                "forward_axis": self.forward_axis,
                "lateral_axis": self.lateral_axis,
                "up_axis": self.up_axis,
                "external_mesh_required_for_anatomical_appearance": True,
            },
        }


class FemaleTrajectoryStimulus(PhysicalStimulus):
    """Female-fly stimulus whose world pose is sampled from a JSON path.

    ``condition`` is one of ``moving``, ``static`` or ``hidden``. Static holds
    the first trajectory pose while retaining the same stimulus/asset identity;
    hidden follows the trajectory in metadata but disables rendering. Every
    condition remains visual-only and non-interacting. ``reference_pose`` is
    accepted for PhysicalStimulus compatibility but deliberately ignored.
    """

    CONDITIONS = ("moving", "static", "hidden")

    def __init__(
        self,
        trajectory: FemaleTargetTrajectory | Mapping[str, Any] | str | Path | None = None,
        *,
        geometry: FemaleTargetGeometry | Mapping[str, Any] | str | Path | None = None,
        stimulus_id: str = "female_fly",
        texture: str = "female_fly",
        albedo: float = 0.72,
        condition: str = "moving",
    ) -> None:
        self.trajectory = (trajectory if isinstance(trajectory, FemaleTargetTrajectory)
                           else FemaleTargetTrajectory.from_json(trajectory))
        self.target_geometry = (geometry if isinstance(geometry, FemaleTargetGeometry)
                                else FemaleTargetGeometry.from_json(geometry))
        condition_value = str(condition).strip().lower()
        if condition_value not in self.CONDITIONS:
            raise ValueError(f"condition must be one of {self.CONDITIONS}")
        self.condition = condition_value
        first = self.trajectory.sample(self.trajectory.start_time_s)
        super().__init__(
            id=str(stimulus_id),
            start_position_m=first.position_m,
            velocity_mps=first.velocity_mps,
            direction_m=first.direction_m,
            size_m=self.target_geometry.stimulus_size_m,
            start_time_s=self.trajectory.start_time_s,
            duration_s=None,
            mode="static",
            texture=str(texture),
            renderable=condition_value != "hidden",
            collidable=False,
            anchor="world",
            albedo=albedo,
            appearance_provenance=f"courtship_target_geometry:{self.target_geometry.geometry_id}",
            visual_only=True,
            non_interacting=True,
        )

    @classmethod
    def from_configs(
        cls,
        *,
        trajectory_path: str | Path | None = None,
        geometry_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "FemaleTrajectoryStimulus":
        """Convenience constructor spelling out the two independent config paths."""
        return cls(trajectory=trajectory_path, geometry=geometry_path, **kwargs)

    def pose_at(self, t_s: float) -> FemaleTargetPose:
        """Return an interpolated pose without consulting a male/reference pose."""
        if self.condition == "static":
            initial = self.trajectory.sample(self.trajectory.start_time_s)
            return FemaleTargetPose(_finite(t_s, "t_s"), initial.position_m,
                                    initial.yaw_rad, initial.pitch_rad, initial.roll_rad,
                                    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        return self.trajectory.sample(t_s)

    def active_at(self, t_s: float) -> bool:
        # Endpoint clamping makes the target a persistent world object across
        # the whole simulation, even if a caller asks for a time before frame 0.
        _finite(t_s, "t_s")
        return True

    def position_at(self, t_s: float, reference_pose: Any = None) -> tuple[float, float, float]:
        del reference_pose
        return self.pose_at(t_s).position_m

    def state_at(self, t_s: float, reference_pose: Any = None) -> StimulusState:
        del reference_pose
        t = _finite(t_s, "t_s")
        pose = self.pose_at(t)
        renderable = self.condition != "hidden"
        return StimulusState(
            id=self.id,
            time_s=t,
            active=True,
            renderable=renderable,
            collidable=False,
            mode="static" if self.condition == "static" else "translation",
            position_m=pose.position_m,
            velocity_mps=pose.velocity_mps,
            direction_m=pose.direction_m,
            size_m=self.target_geometry.stimulus_size_m,
            texture=self.texture,
            albedo=self.albedo,
            forbidden_for_control=True,
            visual_only=True,
            non_interacting=True,
            counterfactual_contact_time_s=None,
        )

    at = state_at

    def stimulus_state_at(self, t_s: float, reference_pose: Any = None) -> dict[str, Any]:
        """Return a logging record with the full interpolated Euler pose."""
        state = self.state_at(t_s, reference_pose)
        pose = self.pose_at(t_s)
        out = state.to_dict()
        out.update({
            "condition": self.condition,
            "trajectory_id": self.trajectory.trajectory_id,
            "geometry_id": self.target_geometry.geometry_id,
            "semantic_role": self.target_geometry.semantic_role,
            "yaw_rad": pose.yaw_rad,
            "pitch_rad": pose.pitch_rad,
            "roll_rad": pose.roll_rad,
            "angular_velocity_radps": list(pose.angular_velocity_radps),
            "anchor": "world",
            "visual_only": True,
            "non_interacting": True,
            "male_pose_influences_target": False,
        })
        return out

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out.update({
            "condition": self.condition,
            "semantic_role": self.target_geometry.semantic_role,
            "geometry_id": self.target_geometry.geometry_id,
            "trajectory": self.trajectory.to_dict(),
            "target_geometry": self.target_geometry.to_dict(),
            "pose_time_basis": "absolute simulation seconds; independent of reference_pose",
        })
        return out


__all__ = [
    "FemaleTargetGeometry",
    "FemaleTargetPose",
    "FemaleTargetTrajectory",
    "FemaleTrajectoryStimulus",
    "TrajectoryKeyframe",
]
