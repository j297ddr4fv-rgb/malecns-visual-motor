#!/usr/bin/env python3
"""Render a deterministic, camera-tracked digital-fly flight replay.

The flight launcher writes a compact ``trajectory.json`` containing telemetry
rows.  This module turns those rows into RGB frames with Pillow and pipes the
frames to ``ffmpeg`` for a portable H.264/MP4 file.  It intentionally does not
import the simulator, CUDA, or any optional visualization stack: a replay
should remain renderable on a CPU-only machine after a run has finished.

Typical use::

    python3 scripts/render_flight_video.py \
        --trajectory artifacts/gpu-e2e-smoke-final-20260921/trajectory.json \
        --out artifacts/gpu-e2e-smoke-final-20260921/flight_replay.mp4 \
        --fps 30

The renderer is a visualization of a digital simulation, not a biological
claim.  That provenance is burned into every frame and also stored as MP4
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

try:  # Keep the error actionable when the script is copied to another host.
    import numpy as np
except Exception as exc:  # pragma: no cover - exercised only on minimal hosts.
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover - exercised only on minimal hosts.
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None


class RenderError(RuntimeError):
    """Raised for malformed trajectory input or unavailable render tools."""


@dataclass(frozen=True)
class TrajectorySample:
    """One normalized trajectory sample in SI units.

    The launcher currently emits all of these fields, but optional values are
    normalized here so small hand-written traces and older runs can be replayed
    as well.  Vectors are tuples to keep the public API easy to inspect and
    deterministic in tests.
    """

    time_s: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    orientation_rad: tuple[float, float, float]  # roll, pitch, yaw
    wing_left: float = 1.0
    wing_right: float = 1.0
    target_distance_m: Optional[float] = None
    target_bearing_rad: Optional[float] = None
    target_position_m: Optional[tuple[float, float, float]] = None
    ground_distance_m: Optional[float] = None
    flight_mode: str = "flight"
    contact: bool = False
    visual_confidence: Optional[float] = None
    yaw_rate_rad_s: Optional[float] = None
    scenario: str = "flight"
    velocity_present: bool = True
    # Mission/state-drive phase fields are optional in older trajectory files.
    # They are kept separate from ``flight_mode`` because a run can be in the
    # ``takeoff`` phase while the body is still reported as ``grounded``.
    phase_progress: Optional[float] = None
    phase_label: str = ""
    phase_index: Optional[int] = None
    # Recorded wing phase is in cycles [0, 1) when available.  The renderer
    # deliberately does not invent a wingbeat when this field is absent.
    wing_phase_cycles: Optional[float] = None

    @property
    def phase(self) -> Optional[float]:
        """Compatibility alias for the normalized recorded phase clock."""

        return self.phase_progress

    @property
    def mission_phase(self) -> str:
        """Compatibility alias for the recorded mission phase label."""

        return self.phase_label


@dataclass(frozen=True)
class TrajectoryData:
    """Normalized input trace and provenance metadata."""

    samples: tuple[TrajectorySample, ...]
    metadata: Mapping[str, Any]
    source_path: Path
    target_position_m: tuple[float, float, float]
    ground_z_m: float = 0.0


@dataclass(frozen=True)
class FrameState:
    """Interpolated state used by the frame renderer."""

    time_s: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    orientation_rad: tuple[float, float, float]
    wing_left: float
    wing_right: float
    target_position_m: tuple[float, float, float]
    target_distance_m: float
    target_bearing_rad: float
    ground_distance_m: float
    flight_mode: str
    contact: bool
    visual_confidence: float
    yaw_rate_rad_s: float
    scenario: str
    phase_progress: Optional[float] = None
    phase_label: str = ""
    phase_index: Optional[int] = None
    wing_phase_cycles: Optional[float] = None

    @property
    def phase(self) -> Optional[float]:
        return self.phase_progress

    @property
    def mission_phase(self) -> str:
        return self.phase_label


def _require_runtime() -> None:
    missing: list[str] = []
    if np is None:
        missing.append(f"numpy ({_NUMPY_IMPORT_ERROR})")
    if Image is None:
        missing.append(f"Pillow ({_PIL_IMPORT_ERROR})")
    if missing:
        raise RenderError("render dependencies unavailable: " + ", ".join(missing))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_scalar(value: Any, name: str, *, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RenderError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise RenderError(f"{name} must be a finite number, got {value!r}")
    return number


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RenderError(f"{name} must contain three finite numbers")
    try:
        values = list(value)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"{name} must contain three finite numbers") from exc
    if len(values) != 3:
        raise RenderError(f"{name} must contain three finite numbers; got {len(values)}")
    result = tuple(_finite_scalar(item, name) for item in values)
    assert all(item is not None for item in result)
    return (float(result[0]), float(result[1]), float(result[2]))


def _lookup(maps: Sequence[Mapping[str, Any]], names: Sequence[str]) -> tuple[Any, bool]:
    """Look up aliases in a row and its common nested body/world mappings."""

    normalized = {str(name).strip().lower().replace("-", "_"): name for name in names}
    for candidate in maps:
        for key, value in candidate.items():
            key_norm = str(key).strip().lower().replace("-", "_")
            if key_norm in normalized:
                return value, True
    return None, False


def _row_maps(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    maps: list[Mapping[str, Any]] = [row]
    body = row.get("body")
    if isinstance(body, Mapping):
        maps.append(body)
        timestamp = body.get("timestamp")
        if isinstance(timestamp, Mapping):
            maps.append(timestamp)
    world = row.get("world")
    if isinstance(world, Mapping):
        maps.append(world)
        world_body = world.get("body")
        if isinstance(world_body, Mapping):
            maps.append(world_body)
            timestamp = world_body.get("timestamp")
            if isinstance(timestamp, Mapping):
                maps.append(timestamp)
    timestamp = row.get("timestamp")
    if isinstance(timestamp, Mapping):
        maps.append(timestamp)
    return maps


def _find_vector(
    maps: Sequence[Mapping[str, Any]], names: Sequence[str], name: str,
) -> tuple[Optional[tuple[float, float, float]], bool]:
    value, found = _lookup(maps, names)
    if not found or value is None:
        return None, False
    return _vector3(value, name), True


def _find_scalar(
    maps: Sequence[Mapping[str, Any]], names: Sequence[str], name: str,
    *, default: Optional[float] = None,
) -> tuple[Optional[float], bool]:
    value, found = _lookup(maps, names)
    if not found or value is None:
        return default, False
    return _finite_scalar(value, name), True


def _phase_contexts(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the recorded phase containers in precedence order.

    Current launcher rows expose ``mission_phase`` and ``phase_contract`` at
    the top level, while older state-drive traces put the same values under
    ``flight_state_drive.context``.  Keeping this lookup explicit prevents a
    generic nested lookup from accidentally treating an unrelated ``phase``
    field (for example a unit description) as the mission clock.
    """

    contexts: list[Mapping[str, Any]] = [row]
    contract = row.get("phase_contract")
    if isinstance(contract, Mapping):
        contexts.append(contract)
    state_drive = row.get("flight_state_drive")
    if isinstance(state_drive, Mapping):
        contexts.append(state_drive)
        context = state_drive.get("context")
        if isinstance(context, Mapping):
            contexts.append(context)
            nested_contract = context.get("phase_contract")
            if isinstance(nested_contract, Mapping):
                contexts.append(nested_contract)
    for key in ("state_drive", "phase"):
        value = row.get(key)
        if isinstance(value, Mapping):
            contexts.append(value)
    return contexts


def _phase_fields(row: Mapping[str, Any]) -> tuple[Optional[float], str, Optional[int]]:
    """Extract only phase data that was recorded by the flight runner."""

    contexts = _phase_contexts(row)
    progress: Optional[float] = None
    # Top-level aliases win over nested compatibility fields.  A string
    # ``phase`` is a label in a few older hand-written traces, not a number.
    for context in contexts:
        for key in ("phase_progress", "mission_phase_progress", "phase_fraction", "progress", "phase"):
            if key not in context or context[key] is None:
                continue
            value = context[key]
            if key == "phase" and isinstance(value, str):
                continue
            progress = _finite_scalar(value, "phase_progress")
            break
        if progress is not None:
            break

    label = ""
    for context in contexts:
        for key in ("phase_label", "mission_phase", "phase_name"):
            value = context.get(key)
            if value is not None and not isinstance(value, (Mapping, list, tuple)):
                label = str(value)
                break
        if label:
            break
        # Some compact traces use a string ``phase`` for the phase label.
        value = context.get("phase")
        if isinstance(value, str):
            label = value
            break

    phase_index: Optional[int] = None
    for context in contexts:
        for key in ("phase_index", "mission_phase_index", "index"):
            if key not in context or context[key] is None:
                continue
            value = _finite_scalar(context[key], "phase_index")
            assert value is not None
            phase_index = int(round(value))
            break
        if phase_index is not None:
            break
    return progress, label, phase_index


def _wing_phase_cycles(row: Mapping[str, Any]) -> Optional[float]:
    """Read a recorded wing phase without synthesizing one for replay."""

    contexts: list[Mapping[str, Any]] = [row]
    for key in ("mechanosensory", "sensory", "motor", "nm_out"):
        value = row.get(key)
        if isinstance(value, Mapping):
            contexts.append(value)
            for nested_key in ("raw_channels", "channels", "normalized", "metadata"):
                nested = value.get(nested_key)
                if isinstance(nested, Mapping):
                    contexts.append(nested)
    for context in contexts:
        for key in ("wing_phase_cycles", "wing_phase", "wing_phase_left"):
            if key not in context or context[key] is None:
                continue
            value = context[key]
            # ``channel_units`` and diagnostics sometimes use the same name
            # for a string description; those are not recorded measurements.
            if isinstance(value, str):
                continue
            return _finite_scalar(value, "wing_phase_cycles")
        for key in ("wing_phase_rad", "wing_phase_left_rad"):
            if key not in context or context[key] is None:
                continue
            value = _finite_scalar(context[key], key)
            assert value is not None
            return value / (2.0 * math.pi)
    return None


def _quat_to_rpy(quaternion: Iterable[float]) -> tuple[float, float, float]:
    w, x, y, z = _vector4(quaternion, "orientation_quat")
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def _vector4(value: Any, name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RenderError(f"{name} must contain four finite numbers")
    try:
        values = list(value)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"{name} must contain four finite numbers") from exc
    if len(values) != 4:
        raise RenderError(f"{name} must contain four finite numbers; got {len(values)}")
    numbers = tuple(_finite_scalar(item, name) for item in values)
    assert all(item is not None for item in numbers)
    norm = math.sqrt(sum(float(item) * float(item) for item in numbers))
    if norm <= 1.0e-15:
        raise RenderError(f"{name} must not be the zero quaternion")
    return tuple(float(item) / norm for item in numbers)  # type: ignore[return-value]


def _extract_rows(payload: Any) -> list[Mapping[str, Any]]:
    """Extract runner rows while accepting common wrapper names."""

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for key in ("trajectory", "rows", "samples", "records", "telemetry", "data"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
            if isinstance(candidate, Mapping):
                nested = _extract_rows(candidate)
                if nested:
                    rows = nested
                    break
        if rows is None:
            rows = []
    else:
        rows = []
    result: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RenderError(f"trajectory row {index} must be an object")
        result.append(row)
    return result


def _parse_sample(row: Mapping[str, Any], index: int, default_dt: float) -> TrajectorySample:
    maps = _row_maps(row)
    time, time_found = _find_scalar(
        maps, ("time_s", "t_s", "timestamp_s", "time", "t"), "time_s",
    )
    if not time_found:
        # A hand-written trace without timestamps is still unambiguous when
        # rows are ordered.  The fallback is explicit and deterministic.
        time = index * default_dt
    assert time is not None
    if time < 0.0:
        raise RenderError(f"trajectory row {index} has a negative time_s={time}")

    position, position_found = _find_vector(
        maps, ("position_m", "position", "pos_m", "pos", "xyz"), "position_m",
    )
    if not position_found or position is None:
        raise RenderError(
            f"trajectory row {index} is missing position_m (a finite 3-vector)"
        )
    velocity, velocity_found = _find_vector(
        maps, ("velocity_mps", "velocity", "vel_mps", "vel"), "velocity_mps",
    )
    if velocity is None:
        velocity = (0.0, 0.0, 0.0)

    orientation, orientation_found = _find_vector(
        maps,
        ("orientation_rad", "orientation_rpy_rad", "attitude_rpy_rad", "rpy_rad"),
        "orientation_rad",
    )
    if not orientation_found or orientation is None:
        quaternion, quaternion_found = _lookup(maps, ("orientation_quat", "quaternion", "quat"))
        if quaternion_found and quaternion is not None:
            orientation = _quat_to_rpy(quaternion)
        else:
            roll, _ = _find_scalar(maps, ("roll", "roll_rad"), "roll", default=0.0)
            pitch, _ = _find_scalar(maps, ("pitch", "pitch_rad"), "pitch", default=0.0)
            yaw, _ = _find_scalar(maps, ("yaw", "yaw_rad", "heading_rad"), "yaw", default=0.0)
            orientation = (float(roll or 0.0), float(pitch or 0.0), float(yaw or 0.0))

    left_wing, _ = _find_scalar(
        maps, ("left_wing", "wing_drive_left", "left_wing_power", "wing_left"),
        "left_wing", default=1.0,
    )
    right_wing, _ = _find_scalar(
        maps, ("right_wing", "wing_drive_right", "right_wing_power", "wing_right"),
        "right_wing", default=1.0,
    )
    target_distance, _ = _find_scalar(
        maps, ("target_distance_m", "target_distance", "distance_to_target_m"),
        "target_distance_m", default=None,
    )
    target_bearing, _ = _find_scalar(
        maps, ("target_bearing_rad", "target_bearing", "bearing_rad"),
        "target_bearing_rad", default=None,
    )
    target_position, _ = _find_vector(
        maps, ("target_position_m", "target_position", "goal_position_m"),
        "target_position_m",
    )
    ground_distance, _ = _find_scalar(
        maps, ("ground_distance_m", "altitude_m", "height_m"),
        "ground_distance_m", default=None,
    )
    visual_confidence, _ = _find_scalar(
        maps, ("visual_confidence", "confidence"), "visual_confidence", default=None,
    )
    yaw_rate, _ = _find_scalar(
        maps, ("yaw_rate", "yaw_rate_rad_s", "angular_velocity_yaw_rad_s"),
        "yaw_rate_rad_s", default=None,
    )
    flight_mode_value, flight_mode_found = _lookup(maps, ("flight_mode", "mode"))
    flight_mode = str(flight_mode_value) if flight_mode_found and flight_mode_value is not None else "flight"
    contact_value, contact_found = _lookup(maps, ("contact", "grounded", "on_ground"))
    contact = bool(contact_value) if contact_found else False
    scenario_value, scenario_found = _lookup(maps, ("scenario", "scenario_name"))
    scenario = str(scenario_value) if scenario_found and scenario_value is not None else "flight"
    phase_progress, phase_label, phase_index = _phase_fields(row)
    wing_phase_cycles = _wing_phase_cycles(row)

    return TrajectorySample(
        time_s=float(time),
        position_m=position,
        velocity_mps=velocity,
        orientation_rad=orientation,
        wing_left=float(left_wing if left_wing is not None else 1.0),
        wing_right=float(right_wing if right_wing is not None else 1.0),
        target_distance_m=None if target_distance is None else float(target_distance),
        target_bearing_rad=None if target_bearing is None else float(target_bearing),
        target_position_m=target_position,
        ground_distance_m=None if ground_distance is None else float(ground_distance),
        flight_mode=flight_mode,
        contact=contact,
        visual_confidence=None if visual_confidence is None else float(visual_confidence),
        yaw_rate_rad_s=None if yaw_rate is None else float(yaw_rate),
        scenario=scenario,
        velocity_present=velocity_found,
        phase_progress=phase_progress,
        phase_label=phase_label,
        phase_index=phase_index,
        wing_phase_cycles=wing_phase_cycles,
    )


def _read_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise RenderError(f"{description} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(f"{description} is not valid JSON ({path}): {exc.msg}") from exc
    except OSError as exc:
        raise RenderError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RenderError(f"{description} must contain a JSON object: {path}")
    return value


def _metadata_target(metadata: Mapping[str, Any]) -> Optional[tuple[float, float, float]]:
    for key in ("target_position_m", "goal_position_m"):
        value = metadata.get(key)
        if value is not None:
            return _vector3(value, key)
    target = metadata.get("target")
    if isinstance(target, Mapping):
        value = target.get("position_m", target.get("position"))
        if value is not None:
            return _vector3(value, "target.position_m")
    world = metadata.get("world")
    if isinstance(world, Mapping):
        nested = world.get("target")
        if isinstance(nested, Mapping):
            value = nested.get("position_m", nested.get("position"))
            if value is not None:
                return _vector3(value, "world.target.position_m")
    return None


def _target_from_manifest(source_path: Path, metadata: Mapping[str, Any]) -> tuple[Optional[tuple[float, float, float]], Optional[float]]:
    manifest_value = metadata.get("manifest_path")
    if not manifest_value:
        return None, None
    manifest_path = Path(str(manifest_value))
    if not manifest_path.is_absolute():
        manifest_path = source_path.parent / manifest_path
    try:
        manifest = _read_json(manifest_path, "trajectory manifest")
    except RenderError:
        # A moved/copied trajectory should remain renderable; provenance text
        # still identifies the missing manifest rather than failing the replay.
        return None, None
    config_value = manifest.get("config")
    if not config_value:
        return None, None
    config_path = Path(str(config_value))
    if not config_path.is_absolute():
        config_path = manifest_path.parent / config_path
    try:
        config = _read_json(config_path, "world config")
    except RenderError:
        return None, None
    world = config.get("world")
    if not isinstance(world, Mapping):
        return None, None
    target = world.get("target")
    if not isinstance(target, Mapping):
        return None, None
    position = target.get("position_m", target.get("position"))
    if position is None:
        return None, None
    ground_z = _finite_scalar(world.get("ground_z_m"), "world.ground_z_m", default=0.0)
    return _vector3(position, "world.target.position_m"), ground_z


def _infer_target(samples: Sequence[TrajectorySample]) -> tuple[float, float, float]:
    # Prefer an explicit target position in any row.
    for sample in samples:
        if sample.target_position_m is not None:
            return sample.target_position_m
    # Distance/bearing are body-relative observations.  Use the first row to
    # reconstruct a world point.  This is exact for the runner's static target
    # when the manifest is unavailable, and remains a useful fallback for a
    # short hand-written trace.
    first = samples[0]
    if first.target_distance_m is not None and first.target_bearing_rad is not None:
        yaw = first.orientation_rad[2]
        horizontal = max(0.0, float(first.target_distance_m))
        angle = yaw + float(first.target_bearing_rad)
        return (
            first.position_m[0] + horizontal * math.cos(angle),
            first.position_m[1] + horizontal * math.sin(angle),
            0.0,
        )
    # A deterministic target ahead of the initial heading makes bare position
    # traces visually legible without pretending the input contained a target.
    yaw = first.orientation_rad[2]
    distance = max(0.8, min(2.0, 0.8 + 2.0 * _path_extent(samples)))
    return (
        first.position_m[0] + distance * math.cos(yaw),
        first.position_m[1] + distance * math.sin(yaw),
        0.0,
    )


def _path_extent(samples: Sequence[TrajectorySample]) -> float:
    origin = samples[0].position_m
    return max(
        math.hypot(sample.position_m[0] - origin[0], sample.position_m[1] - origin[1])
        for sample in samples
    )


def _stable_unique_samples(samples: Sequence[TrajectorySample]) -> list[TrajectorySample]:
    ordered = sorted(samples, key=lambda sample: sample.time_s)
    unique: list[TrajectorySample] = []
    for sample in ordered:
        if unique and abs(sample.time_s - unique[-1].time_s) <= 1.0e-12:
            unique[-1] = sample
        else:
            unique.append(sample)
    return unique


def load_trajectory(path: os.PathLike[str] | str) -> TrajectoryData:
    """Load and normalize a runner ``trajectory.json``.

    Timestamps are sorted stably and duplicate timestamps are collapsed by
    keeping their last row.  This handles interrupted/restarted traces without
    producing NaNs during interpolation.
    """

    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise RenderError(f"trajectory input not found: {source_path}")
    if not source_path.is_file():
        raise RenderError(f"trajectory input is not a file: {source_path}")
    payload = _read_json(source_path, "trajectory input")
    rows = _extract_rows(payload)
    if not rows:
        raise RenderError(
            f"trajectory input contains no rows; expected a list under 'trajectory' or 'rows': {source_path}"
        )
    # This is only used for timestamp-less rows.  A 30-Hz fallback is a
    # conservative display cadence and is replaced by the requested output FPS
    # during interpolation.
    samples = [_parse_sample(row, index, 1.0 / 30.0) for index, row in enumerate(rows)]
    unique = _stable_unique_samples(samples)

    metadata_value = payload.get("metadata", {})
    metadata: MutableMapping[str, Any] = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    metadata.setdefault("schema_version", payload.get("schema_version", "unknown"))
    metadata.setdefault("source_path", str(source_path))
    manifest_target, manifest_ground = _target_from_manifest(source_path, metadata)
    target = _metadata_target(metadata) or manifest_target or _infer_target(unique)
    ground = manifest_ground
    if ground is None:
        ground_value = metadata.get("ground_z_m", metadata.get("ground_z"))
        ground = _finite_scalar(ground_value, "ground_z_m", default=0.0)
    assert ground is not None
    metadata.setdefault("target_position_m", list(target))
    metadata.setdefault("ground_z_m", float(ground))
    return TrajectoryData(tuple(unique), dict(metadata), source_path, target, float(ground))


def _as_samples(data: TrajectoryData | Sequence[TrajectorySample] | Sequence[Mapping[str, Any]]) -> tuple[TrajectorySample, ...]:
    if isinstance(data, TrajectoryData):
        samples = data.samples
    else:
        raw_samples = tuple(data)
        if raw_samples and all(isinstance(sample, Mapping) for sample in raw_samples):
            samples = tuple(
                _parse_sample(sample, index, 1.0 / 30.0)
                for index, sample in enumerate(raw_samples)
            )
        else:
            samples = raw_samples  # type: ignore[assignment]
    if not samples:
        raise RenderError("cannot interpolate an empty trajectory")
    if not all(isinstance(sample, TrajectorySample) for sample in samples):
        raise RenderError("interpolate_trajectory expects trajectory rows, TrajectorySample objects, or TrajectoryData")
    return samples


def _linear_interp(times: Any, values: Any, query: Any) -> Any:
    return np.interp(query, times, values, left=values[0], right=values[-1])


def _angle_interp(times: Any, values: Any, query: Any) -> Any:
    unwrapped = np.unwrap(np.asarray(values, dtype=float))
    return _linear_interp(times, unwrapped, query)


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_trajectory(
    data: TrajectoryData | Sequence[TrajectorySample],
    fps: float = 30.0,
    duration_s: Optional[float] = None,
) -> list[FrameState]:
    """Safely resample a trajectory to constant-FPS frame states.

    ``fps`` can be 30 or 60 (or another positive value for a small test).  The
    output uses constant frame spacing; endpoints are held when a short trace
    does not fill one complete frame interval.  Angles are unwrapped before
    interpolation so a yaw crossing ±pi does not spin the fly the long way.
    """

    _require_runtime()
    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"fps must be a positive number, got {fps!r}") from exc
    if not math.isfinite(fps_value) or fps_value <= 0.0:
        raise RenderError(f"fps must be a positive finite number, got {fps!r}")
    samples = _as_samples(data)
    # Apply the same stable ordering and duplicate-time rule to direct calls.
    samples = tuple(_stable_unique_samples(samples))
    times = np.asarray([sample.time_s for sample in samples], dtype=float)
    start = float(times[0])
    end = float(times[-1])
    if duration_s is not None:
        try:
            requested_duration = float(duration_s)
        except (TypeError, ValueError) as exc:
            raise RenderError(f"duration_s must be a positive number, got {duration_s!r}") from exc
        if not math.isfinite(requested_duration) or requested_duration <= 0.0:
            raise RenderError(f"duration_s must be a positive finite number, got {duration_s!r}")
        # ``duration_s`` is a cap, never a request to replay a one-sample
        # trace.  Extending a trace with held state would make an absent
        # flight look like a recorded hover and would violate replay-only
        # provenance.
        end = min(end, start + requested_duration) if end > start else start
    span = max(0.0, end - start)
    step = 1.0 / fps_value
    if span <= 0.0:
        # Keep a one-sample replay playable while preserving its recorded
        # timestamp; do not turn it into a synthetic hold over ``duration``.
        query = np.asarray([start, start], dtype=float)
    else:
        frame_count = max(2, int(math.ceil(span * fps_value - 1.0e-10)) + 1)
        # The final cadence interval can be shorter than one output frame.
        # Clamping to the last recorded timestamp avoids even a held
        # extrapolation beyond the supplied trajectory.
        query = np.minimum(start + np.arange(frame_count, dtype=float) * step, end)

    positions = np.asarray([sample.position_m for sample in samples], dtype=float)
    velocities = np.asarray([sample.velocity_mps for sample in samples], dtype=float)
    orientations = np.asarray([sample.orientation_rad for sample in samples], dtype=float)
    wing_left = np.asarray([sample.wing_left for sample in samples], dtype=float)
    wing_right = np.asarray([sample.wing_right for sample in samples], dtype=float)
    confidences = np.asarray([
        np.nan if sample.visual_confidence is None else sample.visual_confidence
        for sample in samples
    ], dtype=float)
    yaw_rates = np.asarray([
        np.nan if sample.yaw_rate_rad_s is None else sample.yaw_rate_rad_s
        for sample in samples
    ], dtype=float)

    # Fill optional scalar channels before interpolation.  NaNs are never fed
    # to np.interp because that would silently poison all later frames.
    def fill(values: Any, fallback: float) -> Any:
        finite = np.isfinite(values)
        if not finite.any():
            return np.full(values.shape, fallback, dtype=float)
        indices = np.arange(values.size)
        return np.interp(indices, indices[finite], values[finite])

    confidences = fill(confidences, 0.0)
    yaw_rates = fill(yaw_rates, 0.0)

    # Older traces omitted velocity.  Detect that case rather than reporting a
    # misleading zero-speed overlay for a clearly moving position trace.
    if not any(sample.velocity_present for sample in samples) and len(samples) > 1:
        velocities = np.column_stack([
            np.gradient(positions[:, axis], times, edge_order=1) for axis in range(3)
        ])
    elif len(samples) == 1 and not samples[0].velocity_present:
        velocities = np.zeros((1, 3), dtype=float)

    if isinstance(data, TrajectoryData):
        default_target = np.asarray(data.target_position_m, dtype=float)
        ground_z = float(data.ground_z_m)
    else:
        explicit = next((sample.target_position_m for sample in samples if sample.target_position_m is not None), None)
        default_target = np.asarray(explicit or _infer_target(samples), dtype=float)
        ground_z = 0.0
    target_arrays = [
        sample.target_position_m if sample.target_position_m is not None else tuple(default_target)
        for sample in samples
    ]
    target_positions = np.asarray(target_arrays, dtype=float)

    frames: list[FrameState] = []
    for frame_time in query:
        position = tuple(float(_linear_interp(times, positions[:, axis], frame_time)) for axis in range(3))
        velocity = tuple(float(_linear_interp(times, velocities[:, axis], frame_time)) for axis in range(3))
        orientation = tuple(
            float(_angle_interp(times, orientations[:, axis], frame_time)) for axis in range(3)
        )
        target = tuple(float(_linear_interp(times, target_positions[:, axis], frame_time)) for axis in range(3))
        yaw = orientation[2]
        delta = (target[0] - position[0], target[1] - position[1], target[2] - position[2])
        distance = float(math.sqrt(sum(component * component for component in delta)))
        bearing = _wrap_angle(math.atan2(delta[1], delta[0]) - yaw) if math.hypot(delta[0], delta[1]) > 1.0e-12 else 0.0
        sample_index = int(np.searchsorted(times, frame_time, side="right") - 1)
        sample = samples[max(0, min(len(samples) - 1, sample_index))]
        frames.append(FrameState(
            time_s=float(frame_time),
            position_m=position,
            velocity_mps=velocity,
            orientation_rad=orientation,
            wing_left=float(np.clip(_linear_interp(times, wing_left, frame_time), 0.0, 1.5)),
            wing_right=float(np.clip(_linear_interp(times, wing_right, frame_time), 0.0, 1.5)),
            target_position_m=target,
            target_distance_m=distance,
            target_bearing_rad=bearing,
            ground_distance_m=max(0.0, position[2] - ground_z),
            flight_mode=sample.flight_mode,
            contact=sample.contact,
            visual_confidence=float(np.clip(_linear_interp(times, confidences, frame_time), 0.0, 1.0)),
            yaw_rate_rad_s=float(_linear_interp(times, yaw_rates, frame_time)),
            scenario=sample.scenario,
            phase_progress=sample.phase_progress,
            phase_label=sample.phase_label,
            phase_index=sample.phase_index,
            wing_phase_cycles=sample.wing_phase_cycles,
        ))
    return frames


# Compatibility alias for callers that use the shorter term.
resample_trajectory = interpolate_trajectory


def _font(size: int, bold: bool = False) -> Any:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _project(
    point: Sequence[float], state: FrameState, cam_yaw: float, *,
    center: tuple[float, float], scale: float, horizon: float, ground_z: float,
) -> tuple[float, float]:
    dx = float(point[0]) - state.position_m[0]
    dy = float(point[1]) - state.position_m[1]
    dz = float(point[2]) - ground_z
    forward = dx * math.cos(cam_yaw) + dy * math.sin(cam_yaw)
    right = -dx * math.sin(cam_yaw) + dy * math.cos(cam_yaw)
    # A shallow perspective-like projection: far ground approaches the horizon
    # while altitude remains visually obvious.  The camera origin follows the
    # fly every frame, so the fly itself stays centered and the world moves.
    vertical_scale = scale * 0.76
    depth_scale = scale * 0.42
    return (
        center[0] + right * scale,
        horizon + (state.position_m[2] - dz) * vertical_scale - forward * depth_scale,
    )


def _rotated_polygon(center: tuple[float, float], direction: tuple[float, float], length: float, width: float) -> list[tuple[float, float]]:
    ux, uy = direction
    px, py = -uy, ux
    cx, cy = center
    return [
        (cx + ux * length, cy + uy * length),
        (cx + ux * length * 0.56 + px * width, cy + uy * length * 0.56 + py * width),
        (cx + ux * length * 0.18 + px * width * 0.72, cy + uy * length * 0.18 + py * width * 0.72),
        (cx - ux * length * 0.10, cy - uy * length * 0.10),
        (cx + ux * length * 0.18 - px * width * 0.72, cy + uy * length * 0.18 - py * width * 0.72),
        (cx + ux * length * 0.56 - px * width, cy + uy * length * 0.56 - py * width),
    ]


def _ellipse_polygon(center: tuple[float, float], major: float, minor: float, angle: float, steps: int = 28) -> list[tuple[float, float]]:
    ca, sa = math.cos(angle), math.sin(angle)
    cx, cy = center
    return [
        (cx + major * math.cos(theta) * ca - minor * math.sin(theta) * sa,
         cy + major * math.cos(theta) * sa + minor * math.sin(theta) * ca)
        for theta in (2.0 * math.pi * index / steps for index in range(steps))
    ]


def _draw_ground(image: Any, state: FrameState, *, center: tuple[float, float], scale: float, horizon: float, ground_z: float) -> None:
    width, height = image.size
    draw = ImageDraw.Draw(image)
    # Subtle deterministic depth gradient.
    for y in range(max(0, int(horizon)), height):
        depth = (y - horizon) / max(1.0, height - horizon)
        color = (
            int(18 + 13 * depth),
            int(43 + 18 * depth),
            int(38 + 13 * depth),
        )
        draw.line((0, y, width, y), fill=color)
    draw.line((0, horizon, width, horizon), fill=(101, 151, 142), width=2)

    cam_yaw = state.orientation_rad[2] - 0.40
    forward_vec = (math.cos(cam_yaw), math.sin(cam_yaw), 0.0)
    right_vec = (-math.sin(cam_yaw), math.cos(cam_yaw), 0.0)
    grid_span = max(5.0, 180.0 / max(scale, 1.0))
    # Lines recede toward the horizon; alternating strips add depth without a
    # texture asset and keep every frame reproducible.
    for offset in np.arange(-grid_span, grid_span + 0.001, 0.32):
        a = (
            state.position_m[0] + forward_vec[0] * -3.0 + right_vec[0] * offset,
            state.position_m[1] + forward_vec[1] * -3.0 + right_vec[1] * offset,
            ground_z,
        )
        b = (
            state.position_m[0] + forward_vec[0] * 9.0 + right_vec[0] * offset,
            state.position_m[1] + forward_vec[1] * 9.0 + right_vec[1] * offset,
            ground_z,
        )
        pa = _project(a, state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z)
        pb = _project(b, state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z)
        draw.line((pa, pb), fill=(53, 94, 80), width=1)
    for forward in np.arange(-2.5, 9.01, 0.38):
        a = (
            state.position_m[0] + forward_vec[0] * forward - right_vec[0] * grid_span,
            state.position_m[1] + forward_vec[1] * forward - right_vec[1] * grid_span,
            ground_z,
        )
        b = (
            state.position_m[0] + forward_vec[0] * forward + right_vec[0] * grid_span,
            state.position_m[1] + forward_vec[1] * forward + right_vec[1] * grid_span,
            ground_z,
        )
        pa = _project(a, state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z)
        pb = _project(b, state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z)
        draw.line((pa, pb), fill=(47, 84, 73), width=1)


def _draw_trail(image: Any, frames: Sequence[FrameState], frame_index: int, *, state: FrameState, center: tuple[float, float], scale: float, horizon: float, ground_z: float, trail_seconds: float) -> None:
    if frame_index <= 0:
        return
    draw = ImageDraw.Draw(image)
    cam_yaw = state.orientation_rad[2] - 0.40
    start_index = max(0, frame_index - int(max(1.0, trail_seconds / max(1.0e-9, frames[1].time_s - frames[0].time_s))))
    points: list[tuple[float, float]] = []
    for prior in frames[start_index:frame_index + 1]:
        points.append(_project(prior.position_m, state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z))
    for index in range(1, len(points)):
        strength = index / max(1, len(points) - 1)
        color = (int(44 + 45 * strength), int(202 * strength), int(166 + 55 * strength))
        draw.line((points[index - 1], points[index]), fill=color, width=3 if strength > 0.65 else 2)
    if points:
        px, py = points[-1]
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(125, 255, 199), outline=(230, 255, 241))


def _draw_target(image: Any, state: FrameState, *, center: tuple[float, float], scale: float, horizon: float, ground_z: float) -> None:
    draw = ImageDraw.Draw(image)
    cam_yaw = state.orientation_rad[2] - 0.40
    target = _project(state.target_position_m, state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z)
    tx, ty = target
    # Perspective squash for the landing pad/ring.
    radius = _clamp(0.09 * scale, 12.0, 32.0)
    ring = [(tx + radius * math.cos(theta), ty + radius * 0.35 * math.sin(theta)) for theta in np.linspace(0, 2 * math.pi, 40)]
    draw.polygon(ring, fill=(237, 170, 63), outline=(255, 228, 136))
    inner = [(tx + radius * 0.58 * math.cos(theta), ty + radius * 0.20 * math.sin(theta)) for theta in np.linspace(0, 2 * math.pi, 40)]
    draw.polygon(inner, fill=(33, 65, 57), outline=(255, 222, 106))
    draw.line((tx - radius * 0.78, ty, tx + radius * 0.78, ty), fill=(255, 217, 105), width=2)
    draw.line((tx, ty - radius * 0.28, tx, ty + radius * 0.28), fill=(255, 217, 105), width=2)
    label_font = _font(13, bold=True)
    label_x = tx + radius + 8
    label_y = ty - 11
    if label_x > image.size[0] - 80:
        label_x = tx - radius - 80
    if label_y < 8:
        label_y = ty + 8
    draw.text((label_x, label_y), "TARGET", font=label_font, fill=(255, 224, 133), stroke_width=2, stroke_fill=(22, 45, 40))


def _side_view_axis(frames: Sequence[FrameState]) -> tuple[float, float]:
    """Choose a stable along-track axis for the x/z side view."""

    origin = frames[0].position_m
    for frame in frames[1:]:
        dx = frame.position_m[0] - origin[0]
        dy = frame.position_m[1] - origin[1]
        length = math.hypot(dx, dy)
        if length > 1.0e-9:
            return dx / length, dy / length
    for frame in frames:
        dx, dy = frame.velocity_mps[0], frame.velocity_mps[1]
        length = math.hypot(dx, dy)
        if length > 1.0e-9:
            return dx / length, dy / length
    yaw = frames[0].orientation_rad[2]
    return math.cos(yaw), math.sin(yaw)


def _draw_side_view(
    image: Any,
    frames: Sequence[FrameState],
    frame_index: int,
    *,
    state: FrameState,
    ground_z: float,
) -> None:
    """Draw a compact recorded x/z side projection beside the 3-D replay.

    The main scene is a shallow perspective projection of all three world
    coordinates.  This inset makes the altitude component unambiguous without
    claiming that the Pillow drawing is a metric 3-D reconstruction.
    """

    width, height = image.size
    # At tiny smoke-test resolutions the telemetry panel is intentionally more
    # useful than a sub-100px inset.  Normal output (>=640x360) always shows it.
    if width < 420 or height < 240 or not frames:
        return
    panel_width = min(286, max(220, int(width * 0.30)))
    panel_height = min(148, max(96, int(height * 0.26)))
    panel_width = min(panel_width, width - 24)
    panel_height = min(panel_height, height - 48)
    left = width - panel_width - 14
    top = height - panel_height - 38
    right = left + panel_width
    bottom = top + panel_height

    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (left, top, right, bottom), radius=9,
        fill=(7, 20, 27), outline=(108, 177, 159), width=1,
    )
    title_font = _font(max(10, min(14, int(height / 34))), bold=True)
    tiny_font = _font(max(9, min(11, int(height / 48))))
    draw.text((left + 10, top + 7), "SIDE VIEW / RECORDED X-Z", font=title_font, fill=(224, 250, 238))

    ux, uy = _side_view_axis(frames)
    origin = frames[0].position_m

    def along(position: Sequence[float]) -> float:
        return (float(position[0]) - origin[0]) * ux + (float(position[1]) - origin[1]) * uy

    horizontal = [along(frame.position_m) for frame in frames]
    heights = [frame.position_m[2] for frame in frames]
    # Include the recorded target only as a visual reference; it is not added
    # to the trajectory path.
    target_along = along(state.target_position_m)
    horizontal.append(target_along)
    heights.append(state.target_position_m[2])
    x_min, x_max = min(horizontal), max(horizontal)
    z_min, z_max = min(min(heights), ground_z), max(max(heights), ground_z)
    x_span = max(0.25, x_max - x_min)
    z_span = max(0.20, z_max - z_min)
    x_margin = x_span * 0.12
    z_margin = z_span * 0.16
    x_min -= x_margin
    x_max += x_margin
    z_min = min(ground_z, z_min - z_margin)
    z_max += z_margin
    plot_left, plot_top = left + 12, top + 28
    plot_right, plot_bottom = right - 12, bottom - 13

    def project(position: Sequence[float]) -> tuple[float, float]:
        x_value = along(position)
        z_value = float(position[2])
        px = plot_left + (x_value - x_min) / max(1.0e-9, x_max - x_min) * (plot_right - plot_left)
        py = plot_bottom - (z_value - z_min) / max(1.0e-9, z_max - z_min) * (plot_bottom - plot_top)
        return px, py

    ground_y = project((origin[0], origin[1], ground_z))[1]
    draw.line((plot_left, ground_y, plot_right, ground_y), fill=(83, 128, 115), width=1)

    past = frames[: max(1, min(len(frames), frame_index + 1))]
    points = [project(frame.position_m) for frame in past]
    for index in range(1, len(points)):
        strength = index / max(1, len(points) - 1)
        draw.line(
            (points[index - 1], points[index]),
            fill=(72, int(155 + 80 * strength), int(148 + 60 * strength)),
            width=2,
        )
    current_x, current_y = project(state.position_m)
    draw.ellipse(
        (current_x - 4, current_y - 4, current_x + 4, current_y + 4),
        fill=(255, 206, 100), outline=(255, 248, 200),
    )
    # A tiny side silhouette keeps posture/altitude readable even when the
    # main camera is tracking the fly at the centre of the frame.
    body_half = max(6.0, min(16.0, (plot_right - plot_left) * 0.045))
    draw.line((current_x - body_half, current_y, current_x + body_half, current_y), fill=(239, 186, 100), width=2)
    draw.line((current_x - body_half * 0.25, current_y, current_x - body_half * 0.75, current_y - body_half * 0.65), fill=(170, 226, 242), width=1)
    draw.line((current_x + body_half * 0.25, current_y, current_x + body_half * 0.75, current_y - body_half * 0.65), fill=(170, 226, 242), width=1)
    draw.text((plot_left, plot_bottom - 1), f"ground {ground_z:+.2f} m", font=tiny_font, fill=(153, 208, 190))
    draw.text((plot_right - 67, plot_top - 1), f"z {state.position_m[2]:+.2f} m", font=tiny_font, fill=(255, 223, 150))


def _draw_fly(image: Any, state: FrameState, frame_index: int, *, center: tuple[float, float], scale: float, horizon: float, ground_z: float) -> Any:
    width, height = image.size
    cam_yaw = state.orientation_rad[2] - 0.40
    fly_px = _project(state.position_m, state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z)
    head_world = (
        state.position_m[0] + 0.10 * math.cos(state.orientation_rad[2]),
        state.position_m[1] + 0.10 * math.sin(state.orientation_rad[2]),
        state.position_m[2],
    )
    heading_px = _project(head_world, state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z)
    hx, hy = heading_px[0] - fly_px[0], heading_px[1] - fly_px[1]
    norm = math.hypot(hx, hy)
    axis = (hx / norm, hy / norm) if norm > 1.0e-9 else (1.0, 0.0)
    perp = (-axis[1], axis[0])
    body_angle = math.atan2(axis[1], axis[0])
    size = _clamp(scale * 0.13, 23.0, 46.0)
    thorax = (fly_px[0] + axis[0] * 2.0, fly_px[1] + axis[1] * 2.0)
    abdomen = (fly_px[0] - axis[0] * size * 0.27, fly_px[1] - axis[1] * size * 0.27)
    # Use a recorded wing phase when the trajectory contains one.  A replay
    # must not imply a neural/flight cycle that was never present in the
    # telemetry; older traces therefore render a stable posture instead of a
    # frame-index animation.
    wing_phase = (
        2.0 * math.pi * float(state.wing_phase_cycles)
        if state.wing_phase_cycles is not None else 0.0
    )
    draw = ImageDraw.Draw(image)

    # Motion shadow on the ground anchors the fly in the camera view.
    ground_point = _project((state.position_m[0], state.position_m[1], ground_z), state, cam_yaw, center=center, scale=scale, horizon=horizon, ground_z=ground_z)
    gx, gy = ground_point
    draw.ellipse((gx - size * 0.55, gy - size * 0.13, gx + size * 0.55, gy + size * 0.13), fill=(10, 24, 24))

    # Wings live on a translucent layer so their blue membrane reads against
    # both sky and ground.  Wing drives modulate span; phase comes from the
    # recorded telemetry when available.
    wing_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    wing_draw = ImageDraw.Draw(wing_layer)
    for side, drive in ((1.0, state.wing_left), (-1.0, state.wing_right)):
        flap = math.sin(wing_phase + (0.18 if side > 0 else math.pi + 0.18))
        sweep = side * (math.pi * 0.49 + flap * 0.22)
        wing_angle = body_angle + sweep
        wing_dir = (math.cos(wing_angle), math.sin(wing_angle))
        root = (thorax[0] + perp[0] * side * 2.0, thorax[1] + perp[1] * side * 2.0)
        length = (size * (1.26 + 0.16 * abs(flap))) * _clamp(0.55 + 0.45 * drive, 0.25, 1.25)
        poly = _rotated_polygon(root, wing_dir, length, size * 0.23)
        wing_draw.polygon(poly, fill=(119, 211, 239, 112), outline=(188, 239, 255, 215))
        wing_draw.line((root, (root[0] + wing_dir[0] * length * 0.92, root[1] + wing_dir[1] * length * 0.92)), fill=(218, 250, 255, 190), width=1)
        wing_draw.line((root, (root[0] + wing_dir[0] * length * 0.56 + perp[0] * side * size * 0.18, root[1] + wing_dir[1] * length * 0.56 + perp[1] * side * size * 0.18)), fill=(181, 239, 255, 150), width=1)
    image = Image.alpha_composite(image.convert("RGBA"), wing_layer).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Abdomen segments, thorax, head, eyes, antennae, and six legs keep the
    # silhouette unmistakably insect-like at modest video resolutions.
    draw.polygon(_ellipse_polygon(abdomen, size * 0.54, size * 0.22, body_angle), fill=(42, 26, 31), outline=(236, 169, 83))
    for fraction in (-0.25, 0.02, 0.29):
        px = abdomen[0] + axis[0] * size * fraction
        py = abdomen[1] + axis[1] * size * fraction
        draw.line((px - perp[0] * size * 0.19, py - perp[1] * size * 0.19,
                   px + perp[0] * size * 0.19, py + perp[1] * size * 0.19), fill=(214, 130, 62), width=2)
    draw.polygon(_ellipse_polygon(thorax, size * 0.34, size * 0.29, body_angle), fill=(81, 48, 38), outline=(253, 193, 96))
    head = (thorax[0] + axis[0] * size * 0.39, thorax[1] + axis[1] * size * 0.39)
    draw.polygon(_ellipse_polygon(head, size * 0.23, size * 0.22, body_angle), fill=(41, 31, 39), outline=(235, 166, 90))
    eye_offset = size * 0.12
    for side in (-1.0, 1.0):
        eye = (head[0] + perp[0] * eye_offset * side + axis[0] * size * 0.05,
               head[1] + perp[1] * eye_offset * side + axis[1] * size * 0.05)
        draw.ellipse((eye[0] - 3, eye[1] - 3, eye[0] + 3, eye[1] + 3), fill=(246, 80, 70), outline=(255, 224, 167))
        antenna = (eye[0] + axis[0] * size * 0.42 + perp[0] * side * size * 0.14,
                   eye[1] + axis[1] * size * 0.42 + perp[1] * side * size * 0.14)
        draw.line((eye, antenna), fill=(239, 194, 115), width=1)
    for leg_side in (-1.0, 1.0):
        for leg_index, longitudinal in enumerate((-0.20, 0.06, 0.28)):
            root = (thorax[0] - axis[0] * size * longitudinal + perp[0] * leg_side * size * 0.17,
                    thorax[1] - axis[1] * size * longitudinal + perp[1] * leg_side * size * 0.17)
            tip = (root[0] + perp[0] * leg_side * size * (0.46 + 0.06 * leg_index) - axis[0] * size * (0.12 if leg_index == 0 else -0.06),
                   root[1] + perp[1] * leg_side * size * (0.46 + 0.06 * leg_index) - axis[1] * size * (0.12 if leg_index == 0 else -0.06))
            draw.line((root, tip), fill=(224, 174, 106), width=1)
    return image


def _provenance(metadata: Mapping[str, Any]) -> str:
    evidence = str(metadata.get("evidence_class", "digital_simulation"))
    source = str(metadata.get("source", "trajectory"))
    scenario_value = metadata.get("scenarios")
    if isinstance(scenario_value, list) and scenario_value:
        scenario = ",".join(str(item) for item in scenario_value[:2])
        return f"{evidence} / {source} / {scenario}"
    return f"{evidence} / {source}"


def render_frame(
    state: FrameState,
    frame_index: int,
    frames: Sequence[FrameState],
    *,
    metadata: Mapping[str, Any],
    width: int,
    height: int,
    fps: float,
    ground_z: float = 0.0,
    trail_seconds: float = 1.6,
    scale: Optional[float] = None,
) -> Any:
    """Render one RGB Pillow frame; exposed for quick visual/unit tests."""

    _require_runtime()
    if width <= 0 or height <= 0:
        raise RenderError("frame width and height must be positive")
    if scale is None:
        scale = min(300.0, max(80.0, width * 0.38 / max(1.0, state.target_distance_m + 0.8)))
    image = Image.new("RGB", (width, height), (12, 25, 34))
    # ``np.asarray(Image)`` may expose a read-only view; use a fresh buffer for
    # the row-wise gradient so this also works across Pillow versions.
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = float(height) * 0.57
    # Sky gradient is generated by rows for a stable, lightweight background.
    for y in range(max(0, int(horizon))):
        fraction = y / max(1.0, horizon)
        pixels[y, :, :] = (
            int(18 + 25 * fraction),
            int(38 + 48 * fraction),
            int(65 + 62 * fraction),
        )
    image = Image.fromarray(pixels, mode="RGB")
    center = (width * 0.47, horizon)
    _draw_ground(image, state, center=center, scale=float(scale), horizon=horizon, ground_z=ground_z)
    _draw_trail(image, frames, frame_index, state=state, center=center, scale=float(scale), horizon=horizon, ground_z=ground_z, trail_seconds=trail_seconds)
    _draw_target(image, state, center=center, scale=float(scale), horizon=horizon, ground_z=ground_z)
    image = _draw_fly(image, state, frame_index, center=center, scale=float(scale), horizon=horizon, ground_z=ground_z)
    _draw_side_view(image, frames, frame_index, state=state, ground_z=ground_z)

    # Telemetry overlay is intentionally plain text: it survives video codecs,
    # remains readable in a screenshot, and carries the evidence class.
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    small = _font(max(9, min(16, int(height / 46))))
    tiny = _font(max(9, min(13, int(height / 58))))
    title = _font(max(11, min(21, int(height / 28))), bold=True)
    # Keep the telemetry card left of the camera-tracked fly (which is near
    # 47% of the frame width).  This matters for both small smoke frames and
    # normal 16:9 output: the card should annotate the fly, not hide it.
    panel_width = min(int(width * 0.42), 390)
    panel_width = max(72, min(width - 20, panel_width))
    line_gap = max(2, int(height / 120))
    phase_label = (state.phase_label or "unreported").strip() or "unreported"
    phase_text = phase_label[:24]
    if state.phase_progress is not None and math.isfinite(float(state.phase_progress)):
        phase_text += f" {100.0 * _clamp(float(state.phase_progress), 0.0, 1.0):.0f}%"
    phase_index_text = ""
    if state.phase_index is not None:
        phase_index_text = f" #{state.phase_index}"
    phase_text += phase_index_text
    mode_text = (state.flight_mode or "unreported").strip() or "unreported"
    mode_text = mode_text[:18]
    scenario_text = (state.scenario or "flight").strip() or "flight"
    scenario_text = scenario_text[:18]
    lines = [
        f"time {state.time_s:6.2f} s    speed {math.sqrt(sum(component * component for component in state.velocity_mps)):5.2f} m/s",
        f"pos [{state.position_m[0]:+.2f}, {state.position_m[1]:+.2f}, {state.position_m[2]:+.2f}] m",
        f"heading {math.degrees(state.orientation_rad[2]) % 360:6.1f} deg   yaw {math.degrees(state.yaw_rate_rad_s):+.1f} deg/s",
        f"phase {phase_text}",
        f"mode {mode_text:<18} scenario {scenario_text}",
        f"target {state.target_distance_m:5.2f} m   bearing {math.degrees(state.target_bearing_rad):+.1f} deg",
    ]
    # At tiny QA resolutions keep the required time/phase/mode evidence
    # visible instead of letting long telemetry strings run into the scene.
    max_chars = max(14, int((panel_width - 22) / max(1.0, float(getattr(small, "size", 10)) * 0.56)))
    lines = [line[:max_chars] for line in lines]
    line_height = max(11, int(getattr(small, "size", 11))) + line_gap
    required_height = 37 + len(lines) * line_height
    panel_height = min(max(required_height, int(height * 0.34)), max(40, height - 24))
    odraw.rounded_rectangle((16, 16, panel_width, panel_height), radius=10, fill=(7, 17, 25, 205), outline=(131, 215, 190, 170), width=1)
    title_text = "DIGITAL FLY  /  REPLAY" if width < 500 else "DIGITAL FLY  /  FLIGHT REPLAY"
    odraw.text((28, 22), title_text, font=title, fill=(226, 255, 242, 255))
    y = 22 + max(16, int(getattr(title, "size", 16))) + 7
    for line in lines:
        odraw.text((28, y), line, font=small, fill=(188, 225, 215, 255))
        y += line_height
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    provenance = _provenance(metadata)
    footer = f"PROVENANCE: {provenance}   |   3D TRACK + SIDE VIEW   |   {float(fps):g} FPS"
    footer_top = max(0, height - 31)
    footer_bottom = max(footer_top + 2, height - 11)
    draw.rounded_rectangle((16, footer_top, width - 16, footer_bottom), radius=6, fill=(7, 17, 25), outline=(92, 154, 143), width=1)
    draw.text((26, max(0, footer_top + 3)), footer[: max(1, int(width / 6.2))], font=tiny, fill=(204, 238, 225))
    return image


def _scene_scale(frames: Sequence[FrameState], width: int) -> float:
    if not frames:
        return 160.0
    origin = frames[0].position_m
    extent = 0.8
    for frame in frames:
        extent = max(extent, math.hypot(frame.position_m[0] - origin[0], frame.position_m[1] - origin[1]))
        extent = max(extent, math.hypot(frame.target_position_m[0] - frame.position_m[0], frame.target_position_m[1] - frame.position_m[1]))
    return _clamp(width * 0.42 / extent, 78.0, 285.0)


def _validate_dimensions(width: int, height: int) -> tuple[int, int]:
    if width < 64 or height < 64:
        raise RenderError("video width and height must each be at least 64 pixels")
    if width % 2 or height % 2:
        raise RenderError("video width and height must be even for yuv420p MP4 output")
    return width, height


def render_flight_video(
    trajectory: os.PathLike[str] | str | TrajectoryData,
    output: os.PathLike[str] | str,
    *,
    fps: float = 30.0,
    width: int = 960,
    height: int = 540,
    duration_s: Optional[float] = None,
    trail_seconds: float = 1.6,
    ffmpeg: Optional[os.PathLike[str] | str] = None,
    crf: int = 20,
    preset: str = "veryfast",
) -> Path:
    """Render ``trajectory`` into a playable H.264 MP4 and return its path."""

    _require_runtime()
    width, height = _validate_dimensions(int(width), int(height))
    if isinstance(trajectory, TrajectoryData):
        data = trajectory
    else:
        data = load_trajectory(trajectory)
    frames = interpolate_trajectory(data, fps=fps, duration_s=duration_s)
    if not frames:
        raise RenderError("trajectory produced no renderable frames")
    output_path = Path(output).expanduser()
    if output_path.suffix.lower() != ".mp4":
        raise RenderError(f"output must use the .mp4 extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_program = str(ffmpeg) if ffmpeg is not None else shutil.which("ffmpeg")
    if not ffmpeg_program:
        raise RenderError("ffmpeg executable not found; install ffmpeg and ensure it is on PATH")
    if ffmpeg is not None and not Path(ffmpeg_program).exists():
        raise RenderError(f"ffmpeg executable not found: {ffmpeg_program}")

    scale = _scene_scale(frames, width)
    provenance = _provenance(data.metadata)
    command = [
        ffmpeg_program,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{float(fps):g}",
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-metadata", f"title=Digital Fly Flight Replay",
        "-metadata", f"comment=PROVENANCE: {provenance}; deterministic Pillow renderer",
        "-metadata", "evidence_class=digital_simulation",
        str(output_path),
    ]
    process: Optional[subprocess.Popen[bytes]] = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert process.stdin is not None
        for frame_index, state in enumerate(frames):
            frame = render_frame(
                state, frame_index, frames,
                metadata=data.metadata,
                width=width, height=height, fps=fps,
                ground_z=data.ground_z_m, trail_seconds=trail_seconds, scale=scale,
            )
            array = np.asarray(frame, dtype=np.uint8)
            if array.shape != (height, width, 3):
                raise RenderError(f"renderer produced unexpected frame shape {array.shape}")
            process.stdin.write(np.ascontiguousarray(array).tobytes())
        process.stdin.close()
        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.stderr is not None:
            process.stderr.close()
    except BrokenPipeError as exc:
        if process is not None:
            process.wait()
        raise RenderError("ffmpeg stopped while receiving frames; check the ffmpeg installation") from exc
    except OSError as exc:
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass
            process.wait()
        raise RenderError(f"could not start ffmpeg ({ffmpeg_program}): {exc}") from exc
    finally:
        if process is not None and process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
    if return_code != 0:
        detail = stderr.strip() or "no ffmpeg diagnostic"
        raise RenderError(f"ffmpeg failed with exit code {return_code}: {detail}")
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RenderError(f"ffmpeg reported success but did not create an MP4: {output_path}")
    return output_path


# Common alternate name used by small callers.
render_video = render_flight_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a camera-tracked digital-fly flight replay from trajectory.json")
    parser.add_argument("--trajectory", "--input", dest="trajectory", type=Path, required=True, help="runner trajectory.json")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True, help="output MP4 path")
    parser.add_argument("--fps", type=float, default=30.0, help="constant output frame rate (30 or 60 recommended)")
    parser.add_argument("--width", type=int, default=960, help="video width; must be even")
    parser.add_argument("--height", type=int, default=540, help="video height; must be even")
    parser.add_argument("--duration", type=float, default=None, dest="duration_s", help="optional duration cap in seconds")
    parser.add_argument("--trail-seconds", type=float, default=1.6, help="camera-space trajectory trail length")
    parser.add_argument("--ffmpeg", type=Path, default=None, help="explicit ffmpeg executable")
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality (lower is higher quality)")
    parser.add_argument("--preset", default="veryfast", help="ffmpeg x264 preset")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = render_flight_video(
            args.trajectory, args.out,
            fps=args.fps, width=args.width, height=args.height,
            duration_s=args.duration_s, trail_seconds=args.trail_seconds,
            ffmpeg=args.ffmpeg, crf=args.crf, preset=args.preset,
        )
    except RenderError as exc:
        print(f"render_flight_video: error: {exc}", file=sys.stderr)
        return 2
    print(f"rendered {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI exercised separately.
    raise SystemExit(main())
