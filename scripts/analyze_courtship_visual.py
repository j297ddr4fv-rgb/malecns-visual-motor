#!/usr/bin/env python3
"""Read-only analysis of target orientation and tracking in strict fly runs.

This assay is intentionally narrower than courtship: it measures whether a
digital fly orients toward and follows a visible moving target.  It does not
measure courtship actions, mating, or a P1 neural state.  A stimulus being
configured as renderable, or its centre being in the camera frustum, is not
treated as proof that the target reached the retina.  Visibility is confirmed
only from exact retinal samples that differ from a matched hidden-target run.

The module uses only the Python standard library.  It reads ``manifest.json``
and ``telemetry.jsonl`` and writes no files unless the command line caller asks
for a JSON report with ``--output``.

Example::

    python scripts/analyze_courtship_visual.py \
        --trial moving=/data/.../moving-seed1729 \
        --trial hidden=/data/.../hidden-seed1729 \
        --trial static=/data/.../static-seed1729 \
        --output /data/artifacts/by-project/malecns-gpu/courtship/report.json

The trial condition in a ``--trial`` item overrides the condition declared in
the manifest.  The API accepts the same items as ``TrialSpec`` objects,
``{"condition": ..., "path": ...}`` mappings, ``(condition, path)`` pairs,
or a mapping from condition names to run paths.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from itertools import zip_longest
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence
import zlib


SCHEMA_VERSION = "courtship-visual-analysis-1"
COURTSHIP_MANIFEST_SCHEMA = "courtship-visual-assay-1"
REPORTED_POPULATIONS = {
    "LC10a": {
        "selector": "LC10a",
        "aliases": ("LC10a", "lc10a"),
        "interpretation": "recorded population spike-count and window-rate summary",
    },
    "pC1_labeled_population_proxy": {
        "selector": "prefix:pC1_",
        "aliases": ("prefix:pC1_", "pC1_"),
        "interpretation": (
            "selector-labelled spike proxy only; not a validated P1 state or arousal measure"
        ),
    },
}

# Criteria are deliberately simple and exposed in every report.  Angles are
# radians, retinal deltas are normalized luminance units, and positions are m.
DEFAULT_CRITERIA: dict[str, float | int] = {
    "minimum_paired_seeds": 2,
    # One exactly matched frame is enough to show a pixel-level difference at
    # the retinal input seam.  Later frames are often excluded because the
    # body can already have responded and no longer provides a fair control.
    "minimum_target_in_fov_frames_per_trial": 1,
    "minimum_confirmed_visible_frames_per_trial": 1,
    "minimum_pixel_delta_rms_per_eye": 1.0e-4,
    "maximum_pose_position_mismatch_m": 0.001,
    "maximum_pose_yaw_mismatch_rad": 0.005,
    "maximum_frame_time_mismatch_s": 1.0e-6,
    "alignment_gain_rad": 0.10,
    "moving_vs_hidden_alignment_gain_rad": 0.10,
    "moving_vs_static_alignment_gain_rad": 0.10,
    "minimum_target_directed_yaw_fraction": 0.60,
    "behavior_window_s": 0.25,
}


class CourtshipVisualAnalysisError(ValueError):
    """Raised when a trial bundle is invalid or cannot be compared safely."""


@dataclass(frozen=True)
class TrialSpec:
    """One run directory with an optional explicit condition and display label."""

    path: Path | str
    condition: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class _RunPaths:
    path: Path
    manifest_path: Path
    telemetry_path: Path


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    numbers = tuple(_number(item) for item in value[:3])
    if any(item is None for item in numbers):
        return None
    return numbers  # type: ignore[return-value]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _normalise_condition(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_trial_paths(source: Path | str) -> _RunPaths:
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        manifest_path = path / "manifest.json"
        telemetry_path = path / "telemetry.jsonl"
    else:
        telemetry_path = path
        manifest_path = path.parent / "manifest.json"
    if not telemetry_path.is_file():
        raise CourtshipVisualAnalysisError(f"missing telemetry JSONL: {telemetry_path}")
    if not manifest_path.is_file():
        raise CourtshipVisualAnalysisError(f"missing manifest JSON: {manifest_path}")
    return _RunPaths(path if path.is_dir() else path.parent, manifest_path, telemetry_path)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CourtshipVisualAnalysisError(f"could not read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CourtshipVisualAnalysisError(f"manifest root must be an object: {path}")
    return value


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise CourtshipVisualAnalysisError(f"could not open telemetry {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CourtshipVisualAnalysisError(
                    f"invalid telemetry JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise CourtshipVisualAnalysisError(
                    f"telemetry row must be an object at {path}:{line_number}"
                )
            yield value


def _first_number(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _named_population_values(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    neural = _mapping(row.get("neural"))
    named = _mapping(neural.get("named_outputs"))
    candidates = (
        named.get(field),
        neural.get(field),
        row.get(field),
    )
    return next((value for value in candidates if isinstance(value, Mapping)), {})


def _population_accumulator() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "spike_total": 0.0,
            "spike_windows": 0,
            "rate_weighted_sum": 0.0,
            "rate_weight_s": 0.0,
            "rate_windows": 0,
        }
        for name in REPORTED_POPULATIONS
    }


def _record_population_window(
    accumulators: dict[str, dict[str, Any]], row: Mapping[str, Any],
) -> None:
    counts = _named_population_values(row, "population_spike_counts")
    rates = _named_population_values(row, "population_rates_hz")
    neural_window = _mapping(row.get("neural_window"))
    weight_s = _first_number(neural_window, ("window_dt_s", "dt_s"))
    if weight_s is None or weight_s <= 0.0:
        weight_s = _first_number(row, ("dt_s",))
    if weight_s is None or weight_s <= 0.0:
        weight_s = 1.0
    for name, definition in REPORTED_POPULATIONS.items():
        aliases = definition["aliases"]
        count = next((_number(counts.get(alias)) for alias in aliases if _number(counts.get(alias)) is not None), None)
        rate = next((_number(rates.get(alias)) for alias in aliases if _number(rates.get(alias)) is not None), None)
        accumulator = accumulators[name]
        if count is not None and count >= 0.0:
            accumulator["spike_total"] += count
            accumulator["spike_windows"] += 1
        if rate is not None and rate >= 0.0:
            accumulator["rate_weighted_sum"] += rate * weight_s
            accumulator["rate_weight_s"] += weight_s
            accumulator["rate_windows"] += 1


def _clean_population_accumulators(
    accumulators: Mapping[str, Mapping[str, Any]], total_frames: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, definition in REPORTED_POPULATIONS.items():
        values = accumulators[name]
        spike_windows = int(values["spike_windows"])
        rate_windows = int(values["rate_windows"])
        spike_total = float(values["spike_total"])
        result[name] = {
            "selector": definition["selector"],
            "spike_count_observed": spike_windows > 0,
            "total_spikes": (
                int(spike_total) if spike_total.is_integer() else spike_total
            ) if spike_windows else None,
            "spike_window_count": spike_windows,
            "spike_window_coverage_fraction": spike_windows / total_frames if total_frames else None,
            "mean_rate_hz": (
                float(values["rate_weighted_sum"]) / float(values["rate_weight_s"])
                if rate_windows and float(values["rate_weight_s"]) > 0.0 else None
            ),
            "rate_window_count": rate_windows,
            "rate_window_coverage_fraction": rate_windows / total_frames if total_frames else None,
            "rate_weighting": "neural window duration; falls back to row dt_s, then equal-window weight",
            "interpretation": definition["interpretation"],
        }
    return result


def _target_record(row: Mapping[str, Any], target_id: str) -> Mapping[str, Any]:
    stimuli = _mapping(row.get("stimulus_state"))
    exact = _mapping(stimuli.get(target_id))
    if exact:
        return exact
    normalized_target = target_id.lower().replace("-", "_")
    for key, value in stimuli.items():
        if "female" in str(key).lower() or str(key).lower().replace("-", "_") == normalized_target:
            return _mapping(value)
    for key in ("target_state", "female_fly", "target"):
        candidate = _mapping(row.get(key))
        if candidate:
            return candidate
    return {}


def _camera_config(manifest: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    # The courtship manifest is the run-level configuration record, including
    # compact runs where per-frame world geometry is intentionally omitted.
    declared = _mapping(_mapping(manifest.get("courtship_visual")).get("camera"))
    if declared:
        return declared
    world = _mapping(row.get("world")) if row is not None else {}
    geometry = _mapping(world.get("geometry"))
    camera = _mapping(geometry.get("camera"))
    if camera:
        return camera
    trajectory_meta = _mapping(_mapping(manifest.get("trajectory")).get("metadata"))
    for candidate in (
        _mapping(_mapping(manifest.get("components")).get("world")).get("camera"),
        _mapping(trajectory_meta.get("world_geometry")).get("camera"),
        _mapping(manifest.get("world_geometry")).get("camera"),
        _mapping(_mapping(manifest.get("config")).get("camera")),
    ):
        if isinstance(candidate, Mapping) and candidate:
            return candidate
    return {}


def _world_geometry(manifest: Mapping[str, Any], row: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if row is not None:
        geometry = _mapping(_mapping(row.get("world")).get("geometry"))
        if geometry:
            return geometry
    return _mapping(manifest.get("world_geometry"))


def _body_pose(row: Mapping[str, Any]) -> tuple[tuple[float, float, float] | None, float | None, float | None]:
    body = _mapping(row.get("body"))
    position = _vec3(row.get("position_m", body.get("position_m")))
    orientation = row.get("orientation_rad", body.get("orientation_rad"))
    yaw = _number(row.get("yaw"))
    pitch = _number(row.get("pitch"))
    if isinstance(orientation, (list, tuple)) and len(orientation) >= 3:
        if yaw is None:
            yaw = _number(orientation[2])
        if pitch is None:
            pitch = _number(orientation[1])
    return position, yaw, pitch


def _target_metrics(
    row: Mapping[str, Any],
    target: Mapping[str, Any],
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    body_position, yaw, pitch = _body_pose(row)
    target_position = _vec3(target.get("position_m", target.get("center_m")))
    target_size = target.get("size_m")
    result: dict[str, Any] = {
        "target_bearing_rad": None,
        "target_distance_m": None,
        "target_elevation_rad": None,
        "target_in_camera_fov": None,
        "target_geometry_available": bool(body_position is not None and yaw is not None and target_position is not None),
        "target_state_active": target.get("active"),
        "target_state_renderable": target.get("renderable"),
        "target_visibility": "unknown",
    }
    if body_position is None or yaw is None or target_position is None:
        return result

    dx = target_position[0] - body_position[0]
    dy = target_position[1] - body_position[1]
    dz = target_position[2] - body_position[2]
    horizontal = math.hypot(dx, dy)
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    bearing = _wrap_angle(math.atan2(dy, dx) - yaw)
    elevation = math.atan2(dz, max(horizontal, 1.0e-15)) - (pitch or 0.0)
    result.update({
        "target_bearing_rad": bearing,
        "target_distance_m": distance,
        "target_elevation_rad": elevation,
    })

    hfov = _number(camera.get("hfov_rad"))
    vfov = _number(camera.get("vfov_rad"))
    near_m = _number(camera.get("near_m"))
    far_m = _number(camera.get("far_m"))
    if hfov is None or vfov is None:
        return result
    near_m = near_m if near_m is not None else 0.0
    far_m = far_m if far_m is not None else math.inf
    half_diagonal = 0.0
    if isinstance(target_size, (list, tuple)) and len(target_size) >= 3:
        sizes = [_number(value) for value in target_size[:3]]
        if all(value is not None and value >= 0 for value in sizes):
            half_diagonal = 0.5 * math.sqrt(sum(float(value) ** 2 for value in sizes))
    angular_radius = math.asin(min(1.0, half_diagonal / max(distance, 1.0e-15)))
    # A bounding-sphere/frustum test is conservative about extent but does not
    # model mesh occlusion.  It is explicitly an in-camera-FOV test, not proof
    # of retinal visibility.
    forward_depth = horizontal * math.cos(bearing)
    inside = (
        abs(bearing) <= hfov / 2.0 + angular_radius
        and abs(elevation) <= vfov / 2.0 + angular_radius
        and forward_depth + half_diagonal >= near_m
        and forward_depth - half_diagonal <= far_m
    )
    result["target_in_camera_fov"] = bool(inside)
    result["target_visibility"] = "in_camera_fov_only" if inside else "outside_camera_fov"
    return result


def _array_from_record(value: Any) -> tuple[float, ...] | None:
    """Decode a lossless compact array or a JSON numeric vector, if present."""
    if isinstance(value, (list, tuple)):
        values = tuple(_number(item) for item in value)
        if values and all(item is not None for item in values):
            return values  # type: ignore[return-value]
        return None
    record = _mapping(value)
    if record.get("encoding") != "zlib+base64" or not isinstance(record.get("data"), str):
        return None
    dtype = str(record.get("dtype", ""))
    formats = {"float32": "f", "float64": "d", "int32": "i", "int64": "q", "uint8": "B"}
    fmt = formats.get(dtype)
    if fmt is None:
        return None
    raw_size = _number(record.get("bytes"))
    if raw_size is None or raw_size < 0 or raw_size > 16 * 1024 * 1024:
        return None
    try:
        compressed = base64.b64decode(record["data"], validate=True)
        compressed_digest = record.get("compressed_sha256")
        if isinstance(compressed_digest, str) and hashlib.sha256(compressed).hexdigest() != compressed_digest.lower():
            return None
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, int(raw_size) + 1)
        if decompressor.unconsumed_tail or not decompressor.eof:
            return None
    except (ValueError, zlib.error, TypeError):
        return None
    if len(raw) != int(raw_size):
        return None
    digest = record.get("sha256")
    if isinstance(digest, str) and hashlib.sha256(raw).hexdigest() != digest.lower():
        return None
    width = struct.calcsize("<" + fmt)
    if len(raw) % width:
        return None
    try:
        values = struct.unpack("<" + fmt * (len(raw) // width), raw)
    except struct.error:
        return None
    if not values or any(not math.isfinite(float(item)) for item in values):
        return None
    return tuple(float(item) for item in values)


def _retinal_pair(row: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    visual_input = _mapping(row.get("visual_input"))
    left = _array_from_record(visual_input.get("left_columns"))
    right = _array_from_record(visual_input.get("right_columns"))
    if left is not None and right is not None:
        return left, right
    visual = _mapping(row.get("visual"))
    pixels = visual.get("pixels")
    if isinstance(pixels, (list, tuple)) and len(pixels) >= 2:
        left = _array_from_record(pixels[0])
        right = _array_from_record(pixels[1])
        if left is not None and right is not None:
            return left, right
    # Some full-debug products put exact eye vectors directly in the input.
    left = _array_from_record(visual_input.get("left"))
    right = _array_from_record(visual_input.get("right"))
    if left is not None and right is not None:
        return left, right
    return None


def _manifest_condition(manifest: Mapping[str, Any]) -> str | None:
    assay = _mapping(manifest.get("courtship_visual"))
    condition = _normalise_condition(assay.get("target_condition"))
    if condition:
        return condition
    requested = _mapping(manifest.get("requested_execution"))
    condition = _normalise_condition(requested.get("courtship_target_condition"))
    if condition:
        return condition
    stimulus = _mapping(_mapping(manifest.get("components")).get("physical_stimulus"))
    name = str(stimulus.get("condition") or stimulus.get("name") or "")
    for candidate in ("moving", "static", "hidden"):
        if candidate in name.lower():
            return candidate
    return None


def _manifest_seed(manifest: Mapping[str, Any]) -> str | int | None:
    seed = manifest.get("seed")
    if seed is None:
        seed = _mapping(manifest.get("config")).get("seed")
    if seed is None or isinstance(seed, (str, int)) and not isinstance(seed, bool):
        return seed
    number = _number(seed)
    return int(number) if number is not None and number.is_integer() else None


def _trial_specs(trials: Any) -> list[TrialSpec]:
    if isinstance(trials, Mapping):
        values: list[TrialSpec] = []
        for condition, entry in trials.items():
            if isinstance(entry, (list, tuple)) and not (
                len(entry) == 2 and isinstance(entry[0], str) and isinstance(entry[1], (str, Path))
            ):
                values.extend(TrialSpec(path=item, condition=str(condition)) for item in entry)
            elif isinstance(entry, Mapping):
                values.append(TrialSpec(
                    path=entry.get("path", entry.get("run_dir")),
                    condition=str(entry.get("condition", condition)),
                    label=entry.get("label"),
                ))
            else:
                values.append(TrialSpec(path=entry, condition=str(condition)))
        return values
    if isinstance(trials, (str, Path)):
        return [TrialSpec(path=trials)]
    result: list[TrialSpec] = []
    try:
        values = list(trials)
    except TypeError as exc:
        raise CourtshipVisualAnalysisError("trials must be a path, mapping, or sequence") from exc
    for value in values:
        if isinstance(value, TrialSpec):
            result.append(value)
        elif isinstance(value, Mapping):
            result.append(TrialSpec(
                path=value.get("path", value.get("run_dir")),
                condition=value.get("condition"),
                label=value.get("label"),
            ))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            result.append(TrialSpec(path=value[1], condition=str(value[0])))
        else:
            result.append(TrialSpec(path=value))
    if any(item.path is None for item in result):
        raise CourtshipVisualAnalysisError("every trial needs a path or run_dir")
    return result


def _trial_summary(spec: TrialSpec, index: int) -> dict[str, Any]:
    paths = _resolve_trial_paths(spec.path)
    manifest = _read_manifest(paths.manifest_path)
    assay = _mapping(manifest.get("courtship_visual"))
    condition = _normalise_condition(spec.condition) or _manifest_condition(manifest) or "unlabelled"
    label = spec.label or paths.path.name or f"trial-{index + 1}"
    seed = _manifest_seed(manifest)
    camera: Mapping[str, Any] = {}
    frames: list[dict[str, Any]] = []
    row_count = 0
    population_accumulators = _population_accumulator()
    first_pose: dict[str, Any] | None = None
    provenance = _mapping(manifest.get("provenance"))
    policy = _normalise_condition(manifest.get("execution_policy"))
    evidence_class = provenance.get("evidence_class")
    target_id = str(assay.get("target_id") or assay.get("stimulus_id") or "female_fly")

    for row in _iter_rows(paths.telemetry_path):
        row_count += 1
        _record_population_window(population_accumulators, row)
        if not camera:
            camera = _camera_config(manifest, row)
        target = _target_record(row, target_id)
        geometry = _target_metrics(row, target, camera)
        position, yaw, pitch = _body_pose(row)
        velocity = _vec3(row.get("velocity_mps", _mapping(row.get("body")).get("velocity_mps")))
        target_position = _vec3(target.get("position_m", target.get("center_m")))
        target_velocity = _vec3(target.get("velocity_mps"))
        t_s = _first_number(row, ("time_s", "frame_time_s"))
        dt_s = _first_number(row, ("dt_s",)) or 0.0
        if first_pose is None:
            first_pose = {
                "position_m": list(position) if position is not None else None,
                "yaw_rad": yaw,
                "pitch_rad": pitch,
                "target_position_m": list(target_position) if target_position is not None else None,
            }
        relative_target_velocity_mps = None
        male_velocity_toward_target_mps = None
        closing_speed_mps = None
        target_world_bearing = None
        if position is not None and target_position is not None:
            dx = target_position[0] - position[0]
            dy = target_position[1] - position[1]
            horizontal = math.hypot(dx, dy)
            if horizontal > 1.0e-12:
                target_world_bearing = math.atan2(dy, dx)
                if velocity is not None:
                    toward = (dx / horizontal, dy / horizontal)
                    male_velocity_toward_target_mps = velocity[0] * toward[0] + velocity[1] * toward[1]
                    if target_velocity is not None:
                        target_toward_male = target_velocity[0] * toward[0] + target_velocity[1] * toward[1]
                        closing_speed_mps = male_velocity_toward_target_mps - target_toward_male
                        relative_target_velocity_mps = list(target_velocity)
        grounded = row.get("grounded")
        if grounded is None:
            grounded = row.get("contact")
        if not isinstance(grounded, bool):
            grounded = None
        frames.append({
            "frame_index": row.get("frame_index"),
            "time_s": t_s,
            "dt_s": dt_s,
            "position_m": list(position) if position is not None else None,
            "velocity_mps": list(velocity) if velocity is not None else None,
            "yaw_rad": yaw,
            "pitch_rad": pitch,
            "yaw_rate_rad_s": _first_number(row, ("yaw_rate", "yaw_rate_rad_s")),
            "grounded": grounded,
            "target_position_m": list(target_position) if target_position is not None else None,
            "target_velocity_mps": relative_target_velocity_mps,
            "target_world_bearing_rad": target_world_bearing,
            "target_bearing_rad": geometry["target_bearing_rad"],
            "target_distance_m": geometry["target_distance_m"],
            "target_elevation_rad": geometry["target_elevation_rad"],
            "target_in_camera_fov": geometry["target_in_camera_fov"],
            "target_state_active": geometry["target_state_active"],
            "target_state_renderable": geometry["target_state_renderable"],
            "target_visibility": geometry["target_visibility"],
            "male_velocity_toward_target_mps": male_velocity_toward_target_mps,
            "relative_closing_speed_mps": closing_speed_mps,
        })

    if row_count == 0:
        raise CourtshipVisualAnalysisError(f"telemetry has no rows: {paths.telemetry_path}")
    behavior = _behavior_summary(frames, float(DEFAULT_CRITERIA["behavior_window_s"]))
    stride = max(1, math.ceil(len(frames) / 200))
    sampled_frames = frames[::stride]
    if frames and (not sampled_frames or sampled_frames[-1] is not frames[-1]):
        sampled_frames.append(frames[-1])
    return {
        "label": str(label),
        "condition": condition,
        "seed": seed,
        "run_dir": str(paths.path),
        "manifest_path": str(paths.manifest_path),
        "telemetry_path": str(paths.telemetry_path),
        "manifest_schema_version": manifest.get("schema_version"),
        "courtship_visual_manifest": dict(assay) if assay else None,
        "courtship_visual_manifest_valid": assay.get("schema_version") == COURTSHIP_MANIFEST_SCHEMA,
        "execution_policy": manifest.get("execution_policy"),
        "evidence_class": evidence_class,
        "strict_free_behavior": policy in {
            "strict_free_behavior_escape_experiment", "strict", "free_behavior",
        } or bool(_mapping(manifest.get("autonomy_policy")).get("strict")),
        "frame_count": row_count,
        "neural_population_summary": _clean_population_accumulators(
            population_accumulators, row_count,
        ),
        "camera": dict(camera) if camera else None,
        "initial_pose": first_pose,
        "behavior": behavior,
        "trajectory_samples": [
            {
                key: frame.get(key)
                for key in (
                    "frame_index", "time_s", "target_bearing_rad", "target_distance_m",
                    "target_in_camera_fov", "target_state_active", "target_state_renderable",
                    "yaw_rad", "yaw_rate_rad_s", "grounded",
                    "male_velocity_toward_target_mps", "relative_closing_speed_mps",
                )
            }
            for frame in sampled_frames
        ],
        "target_state_active_fraction": _fraction(
            [frame["target_state_active"] for frame in frames], True
        ),
        "target_state_renderable_fraction": _fraction(
            [frame["target_state_renderable"] for frame in frames], True
        ),
        "target_in_camera_fov_frame_count": sum(frame["target_in_camera_fov"] is True for frame in frames),
        "target_in_camera_fov_fraction": _fraction(
            [frame["target_in_camera_fov"] for frame in frames], True
        ),
        "target_visible_confirmed_frame_count": 0,
        "target_visible_confirmed_fraction": None,
        "target_visibility_status": "unknown_without_matched_retinal_control",
        "_frames": frames,
        "_paths": paths,
    }


def _fraction(values: Sequence[Any], expected: Any) -> float | None:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(value == expected for value in known) / len(known)


def _median(values: Iterable[float]) -> float | None:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(cleaned) if cleaned else None


def _behavior_summary(frames: Sequence[Mapping[str, Any]], window_s: float) -> dict[str, Any]:
    available = [frame for frame in frames if _number(frame.get("target_bearing_rad")) is not None]
    if not available:
        return {
            "available": False,
            "reason": "target or body position/orientation is not present in telemetry",
            "target_in_camera_fov": None,
            "yaw_alignment_gain_rad": None,
            "initial_abs_target_bearing_rad": None,
            "initial_signed_target_bearing_rad": None,
            "final_abs_target_bearing_rad": None,
            "final_signed_target_bearing_rad": None,
            "target_distance_start_m": None,
            "target_distance_end_m": None,
            "target_distance_change_m": None,
            "target_distance_min_m": None,
            "target_distance_max_m": None,
            "target_distance_median_m": None,
            "target_directed_yaw_fraction": None,
            "target_angular_travel_rad": None,
            "target_motion_turn_concordance_fraction": None,
            "target_motion_turn_observation_count": 0,
            "turn_toward_target_rad": None,
            "turn_away_from_target_rad": None,
            "mean_male_velocity_toward_target_mps": None,
            "mean_relative_closing_speed_mps": None,
            "grounded_fraction": None,
            "airborne_fraction": None,
            "liftoff_time_s": None,
        }
    visible_geometry = [frame for frame in available if frame.get("target_in_camera_fov") is True]
    analysis = visible_geometry if visible_geometry else available
    start_t = _number(analysis[0].get("time_s"))
    end_t = _number(analysis[-1].get("time_s"))
    if start_t is None or end_t is None:
        start_window = analysis[: max(1, len(analysis) // 5)]
        end_window = analysis[-max(1, len(analysis) // 5):]
    else:
        actual_window = min(window_s, max(0.0, end_t - start_t) / 2.0)
        start_window = [frame for frame in analysis if (_number(frame.get("time_s")) or 0.0) <= start_t + actual_window]
        end_window = [frame for frame in analysis if (_number(frame.get("time_s")) or 0.0) >= end_t - actual_window]
        if not start_window:
            start_window = [analysis[0]]
        if not end_window:
            end_window = [analysis[-1]]
    start_bearing = _median(abs(float(frame["target_bearing_rad"])) for frame in start_window)
    end_bearing = _median(abs(float(frame["target_bearing_rad"])) for frame in end_window)
    start_signed_bearing = _median(float(frame["target_bearing_rad"]) for frame in start_window)
    end_signed_bearing = _median(float(frame["target_bearing_rad"]) for frame in end_window)
    start_distance = _median(
        float(frame["target_distance_m"]) for frame in start_window
        if _number(frame.get("target_distance_m")) is not None
    )
    end_distance = _median(
        float(frame["target_distance_m"]) for frame in end_window
        if _number(frame.get("target_distance_m")) is not None
    )
    analysis_distances = [
        float(frame["target_distance_m"]) for frame in analysis
        if _number(frame.get("target_distance_m")) is not None
    ]
    alignment_gain = (
        start_bearing - end_bearing
        if start_bearing is not None and end_bearing is not None else None
    )

    toward = 0.0
    away = 0.0
    signed_turn_observations = 0
    target_angular_travel = 0.0
    target_motion_turn_matches = 0
    target_motion_turn_observations = 0
    previous: Mapping[str, Any] | None = None
    for frame in analysis:
        yaw = _number(frame.get("yaw_rad"))
        bearing = _number(frame.get("target_bearing_rad"))
        if yaw is not None and previous is not None:
            old_yaw = _number(previous.get("yaw_rad"))
            old_bearing = _number(previous.get("target_bearing_rad"))
            old_world_bearing = _number(previous.get("target_world_bearing_rad"))
            world_bearing = _number(frame.get("target_world_bearing_rad"))
            if old_yaw is not None and old_bearing is not None:
                turn = _wrap_angle(yaw - old_yaw)
                direction = old_bearing + bearing if bearing is not None else old_bearing
                direction *= 0.5
                if abs(turn) > 1.0e-9 and abs(direction) > 1.0e-9:
                    signed_turn_observations += 1
                    if turn * direction > 0:
                        toward += abs(turn)
                    else:
                        away += abs(turn)
                if old_world_bearing is not None and world_bearing is not None:
                    target_turn = _wrap_angle(world_bearing - old_world_bearing)
                    target_angular_travel += abs(target_turn)
                    if abs(target_turn) >= 1.0e-3:
                        target_motion_turn_observations += 1
                        if turn * target_turn > 0.0:
                            target_motion_turn_matches += 1
        previous = frame
    total_turn = toward + away
    directed_fraction = toward / total_turn if total_turn > 0.0 else None

    grounded_values = [frame["grounded"] for frame in frames if isinstance(frame.get("grounded"), bool)]
    liftoff_time = None
    previous_grounded: bool | None = None
    for frame in frames:
        grounded = frame.get("grounded")
        if previous_grounded is True and grounded is False:
            liftoff_time = _number(frame.get("time_s"))
            break
        if isinstance(grounded, bool):
            previous_grounded = grounded

    velocity_values = [
        float(frame["male_velocity_toward_target_mps"])
        for frame in analysis
        if _number(frame.get("male_velocity_toward_target_mps")) is not None
    ]
    closing_values = [
        float(frame["relative_closing_speed_mps"])
        for frame in analysis
        if _number(frame.get("relative_closing_speed_mps")) is not None
    ]
    return {
        "available": True,
        "analysis_frame_count": len(analysis),
        "analysis_basis": "camera_fov_frames" if visible_geometry else "all_frames_geometry_only",
        "initial_abs_target_bearing_rad": start_bearing,
        "initial_signed_target_bearing_rad": start_signed_bearing,
        "final_abs_target_bearing_rad": end_bearing,
        "final_signed_target_bearing_rad": end_signed_bearing,
        "yaw_alignment_gain_rad": alignment_gain,
        "target_distance_start_m": start_distance,
        "target_distance_end_m": end_distance,
        "target_distance_change_m": (
            end_distance - start_distance
            if start_distance is not None and end_distance is not None else None
        ),
        "target_distance_min_m": min(analysis_distances) if analysis_distances else None,
        "target_distance_max_m": max(analysis_distances) if analysis_distances else None,
        "target_distance_median_m": _median(analysis_distances),
        "target_directed_yaw_fraction": directed_fraction,
        "target_angular_travel_rad": target_angular_travel,
        "target_motion_turn_concordance_fraction": (
            target_motion_turn_matches / target_motion_turn_observations
            if target_motion_turn_observations else None
        ),
        "target_motion_turn_observation_count": target_motion_turn_observations,
        "turn_toward_target_rad": toward,
        "turn_away_from_target_rad": away,
        "signed_turn_observation_count": signed_turn_observations,
        "mean_male_velocity_toward_target_mps": statistics.fmean(velocity_values) if velocity_values else None,
        "mean_relative_closing_speed_mps": statistics.fmean(closing_values) if closing_values else None,
        "grounded_fraction": _fraction(grounded_values, True),
        "airborne_fraction": _fraction(grounded_values, False),
        "liftoff_time_s": liftoff_time,
    }


def _frame_identity(row: Mapping[str, Any]) -> tuple[int | None, float | None]:
    index_value = row.get("frame_index")
    index = None
    if isinstance(index_value, int) and not isinstance(index_value, bool):
        index = index_value
    return index, _first_number(row, ("time_s", "frame_time_s"))


def _pose_match(
    visible: Mapping[str, Any],
    hidden: Mapping[str, Any],
    criteria: Mapping[str, float | int],
) -> bool:
    visible_position = _vec3(visible.get("position_m"))
    hidden_position = _vec3(hidden.get("position_m"))
    visible_yaw = _number(visible.get("orientation_rad", [None, None, None])[2]) if isinstance(visible.get("orientation_rad"), (list, tuple)) and len(visible.get("orientation_rad")) >= 3 else _number(visible.get("yaw"))
    hidden_yaw = _number(hidden.get("orientation_rad", [None, None, None])[2]) if isinstance(hidden.get("orientation_rad"), (list, tuple)) and len(hidden.get("orientation_rad")) >= 3 else _number(hidden.get("yaw"))
    if visible_position is None or hidden_position is None or visible_yaw is None or hidden_yaw is None:
        return False
    delta_position = math.sqrt(sum((visible_position[i] - hidden_position[i]) ** 2 for i in range(3)))
    return (
        delta_position <= float(criteria["maximum_pose_position_mismatch_m"])
        and abs(_wrap_angle(visible_yaw - hidden_yaw)) <= float(criteria["maximum_pose_yaw_mismatch_rad"])
    )


def _raw_row_summary(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    camera = _camera_config(manifest, row)
    assay = _mapping(manifest.get("courtship_visual"))
    target_id = str(assay.get("target_id") or assay.get("stimulus_id") or "female_fly")
    target = _target_record(row, target_id)
    # Visual columns were rendered before dynamics advanced this row.  Compare
    # retinal samples only at frame/time-matched pre-step poses so later
    # treatment-induced pose divergence cannot masquerade as target contrast.
    body_before = _mapping(row.get("body_before"))
    render_row: Mapping[str, Any] = row
    if body_before:
        render_row = {
            **row,
            "position_m": body_before.get("position_m", row.get("position_m")),
            "orientation_rad": body_before.get("orientation_rad", row.get("orientation_rad")),
            "yaw": body_before.get("yaw", row.get("yaw")),
            "pitch": body_before.get("pitch", row.get("pitch")),
        }
    geometry = _target_metrics(render_row, target, camera)
    position, yaw, pitch = _body_pose(render_row)
    return {
        "frame_index": row.get("frame_index"),
        "time_s": _first_number(row, ("time_s", "frame_time_s")),
        "position_m": list(position) if position is not None else None,
        "orientation_rad": [0.0, pitch or 0.0, yaw] if yaw is not None else None,
        "yaw": yaw,
        "target_in_camera_fov": geometry["target_in_camera_fov"],
        "target_bearing_rad": geometry["target_bearing_rad"],
        "retina": _retinal_pair(row),
    }


def _matched_retinal_delta(
    visible: Mapping[str, Any], hidden: Mapping[str, Any], criteria: Mapping[str, float | int],
) -> dict[str, Any]:
    visible_paths: _RunPaths = visible["_paths"]
    hidden_paths: _RunPaths = hidden["_paths"]
    visible_manifest = _read_manifest(visible_paths.manifest_path)
    hidden_manifest = _read_manifest(hidden_paths.manifest_path)
    visible_rows = _iter_rows(visible_paths.telemetry_path)
    hidden_rows = _iter_rows(hidden_paths.telemetry_path)
    per_eye: list[list[float]] = [[], []]
    matched_frames = 0
    pose_matched_frames = 0
    retinal_frames = 0
    fov_frames = 0
    frame_mismatches = 0
    for visible_row, hidden_row in zip_longest(visible_rows, hidden_rows):
        if visible_row is None or hidden_row is None:
            frame_mismatches += 1
            continue
        vi, vt = _frame_identity(visible_row)
        hi, ht = _frame_identity(hidden_row)
        if vi != hi or vt is None or ht is None or abs(vt - ht) > float(criteria["maximum_frame_time_mismatch_s"]):
            frame_mismatches += 1
            continue
        matched_frames += 1
        visible_reduced = _raw_row_summary(visible_row, visible_manifest)
        hidden_reduced = _raw_row_summary(hidden_row, hidden_manifest)
        if not _pose_match(visible_reduced, hidden_reduced, criteria):
            continue
        pose_matched_frames += 1
        if visible_reduced.get("target_in_camera_fov") is not True:
            continue
        fov_frames += 1
        visible_retina = visible_reduced.get("retina")
        hidden_retina = hidden_reduced.get("retina")
        if visible_retina is None or hidden_retina is None:
            continue
        if any(len(visible_retina[index]) != len(hidden_retina[index]) for index in (0, 1)):
            continue
        retinal_frames += 1
        for eye in (0, 1):
            left = visible_retina[eye]
            right = hidden_retina[eye]
            squared = math.fsum((a - b) ** 2 for a, b in zip(left, right)) / max(1, len(left))
            per_eye[eye].append(math.sqrt(squared))
    medians = [_median(values) for values in per_eye]
    minimum = float(criteria["minimum_pixel_delta_rms_per_eye"])
    observed_eyes = [eye for eye, value in enumerate(medians) if value is not None and value >= minimum]
    frame_count_threshold = int(criteria["minimum_confirmed_visible_frames_per_trial"])
    # The report confirms visibility if at least one eye has a persistent exact
    # paired-pixel difference.  It separately reports whether both eyes pass.
    eye_visible_frames = [
        sum(value >= minimum for value in values)
        for values in per_eye
    ]
    confirmed_frames = max(eye_visible_frames, default=0)
    confirmed = confirmed_frames >= frame_count_threshold and bool(observed_eyes) and fov_frames >= frame_count_threshold
    return {
        "method": "same_seed_hidden_control_exact_retinal_vector_rms",
        "control_label": hidden.get("label"),
        "control_condition": hidden.get("condition"),
        "control_seed": hidden.get("seed"),
        "matched_frame_count": matched_frames,
        "pose_matched_frame_count": pose_matched_frames,
        "target_in_camera_fov_matched_frame_count": fov_frames,
        "paired_retinal_frame_count": retinal_frames,
        "confirmed_visible_frame_count": confirmed_frames,
        "frame_mismatch_count": frame_mismatches,
        "eye_rms_pixel_delta_median": {"left": medians[0], "right": medians[1]},
        "eye_frames_over_threshold": {"left": eye_visible_frames[0], "right": eye_visible_frames[1]},
        "pixel_delta_rms_threshold": minimum,
        "minimum_confirmed_visible_frames": frame_count_threshold,
        "visible_confirmed": confirmed,
        "both_eyes_confirmed": all(count >= frame_count_threshold for count in eye_visible_frames),
        "status": "confirmed" if confirmed else (
            "insufficient_matched_retinal_evidence"
        ),
    }


def _pairs_by_seed(
    target_trials: Sequence[Mapping[str, Any]],
    hidden_trials: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    targets: dict[Any, list[Mapping[str, Any]]] = {}
    hidden: dict[Any, list[Mapping[str, Any]]] = {}
    for trial in target_trials:
        if trial.get("seed") is not None:
            targets.setdefault(trial.get("seed"), []).append(trial)
    for trial in hidden_trials:
        if trial.get("seed") is not None:
            hidden.setdefault(trial.get("seed"), []).append(trial)
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for seed in sorted(set(targets) & set(hidden), key=str):
        if len(targets[seed]) != 1 or len(hidden[seed]) != 1:
            continue
        pairs.append((targets[seed][0], hidden[seed][0]))
    return pairs


def _pair_behavior_delta(
    conditioned: Mapping[str, Any], control: Mapping[str, Any],
    retinal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    condition_gain = _number(_mapping(conditioned.get("behavior")).get("yaw_alignment_gain_rad"))
    control_gain = _number(_mapping(control.get("behavior")).get("yaw_alignment_gain_rad"))
    condition_turn = _number(_mapping(conditioned.get("behavior")).get("target_directed_yaw_fraction"))
    control_turn = _number(_mapping(control.get("behavior")).get("target_directed_yaw_fraction"))
    delta = condition_gain - control_gain if condition_gain is not None and control_gain is not None else None
    result = {
        "seed": conditioned.get("seed"),
        "condition_label": conditioned.get("label"),
        "condition": conditioned.get("condition"),
        "control_label": control.get("label"),
        "control_condition": control.get("condition"),
        "condition_yaw_alignment_gain_rad": condition_gain,
        "control_yaw_alignment_gain_rad": control_gain,
        "condition_minus_control_alignment_gain_rad": delta,
        "condition_target_directed_yaw_fraction": condition_turn,
        "control_target_directed_yaw_fraction": control_turn,
    }
    # Stable moving/hidden names make downstream callers uncomplicated while
    # retaining neutral keys for moving/static and other pairings.
    if conditioned.get("condition") == "moving" and control.get("condition") == "hidden":
        result.update({
            "moving_label": conditioned.get("label"),
            "hidden_label": control.get("label"),
            "moving_yaw_alignment_gain_rad": condition_gain,
            "hidden_yaw_alignment_gain_rad": control_gain,
            "moving_minus_hidden_alignment_gain_rad": delta,
            "moving_target_directed_yaw_fraction": condition_turn,
            "hidden_target_directed_yaw_fraction": control_turn,
        })
    if retinal is not None:
        result["visible_target_stimulation"] = dict(retinal)
    return result


def _comparison_summary(
    values: Sequence[float], threshold: float, *,
    condition_name: str = "moving", control_name: str = "hidden",
) -> dict[str, Any]:
    median = _median(values)
    return {
        "pair_count": len(values),
        f"median_{condition_name}_minus_{control_name}_rad": median,
        f"mean_{condition_name}_minus_{control_name}_rad": statistics.fmean(values) if values else None,
        "pairs_at_or_above_threshold": sum(value >= threshold for value in values),
        "required_difference_rad": threshold,
    }


def analyze_courtship_visual(
    trials: Any,
    *,
    moving_condition: str = "moving",
    hidden_condition: str = "hidden",
    static_condition: str = "static",
    criteria: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Analyze target orientation and movement against matched visual controls.

    ``trials`` may be a sequence of ``TrialSpec``/mapping/path inputs or a
    mapping from condition labels to run directories.  Explicit conditions
    override manifest labels.  Group comparisons are matched by manifest seed.

    The top-level ``verdict`` can only support or fail the narrow
    visual-orientation/tracking endpoint.  Missing telemetry/control evidence
    returns ``INCONCLUSIVE``.  This function never infers a P1 state or complete
    courtship from body motion or telemetry labels.
    """
    selected_criteria = dict(DEFAULT_CRITERIA)
    if criteria is not None:
        unknown = set(criteria) - set(DEFAULT_CRITERIA)
        if unknown:
            raise CourtshipVisualAnalysisError(f"unknown criterion(s): {', '.join(sorted(unknown))}")
        selected_criteria.update(criteria)
    if int(selected_criteria["minimum_paired_seeds"]) < 1:
        raise CourtshipVisualAnalysisError("minimum_paired_seeds must be at least one")
    specs = _trial_specs(trials)
    if not specs:
        raise CourtshipVisualAnalysisError("at least one trial is required")
    summaries = [_trial_summary(spec, index) for index, spec in enumerate(specs)]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for trial in summaries:
        by_condition.setdefault(str(trial["condition"]), []).append(trial)

    moving_key = _normalise_condition(moving_condition) or "moving"
    hidden_key = _normalise_condition(hidden_condition) or "hidden"
    static_key = _normalise_condition(static_condition) or "static"
    moving_trials = by_condition.get(moving_key, [])
    hidden_trials = by_condition.get(hidden_key, [])
    static_trials = by_condition.get(static_key, [])

    target_hidden_pairs = _pairs_by_seed(moving_trials, hidden_trials)
    moving_static_pairs = _pairs_by_seed(moving_trials, static_trials)
    static_hidden_pairs = _pairs_by_seed(static_trials, hidden_trials)
    behavior_pairs: list[dict[str, Any]] = []
    for moving, hidden in target_hidden_pairs:
        retinal = _matched_retinal_delta(moving, hidden, selected_criteria)
        moving["retinal_visibility_vs_hidden"] = retinal
        moving["target_visible_confirmed_frame_count"] = int(retinal["confirmed_visible_frame_count"])
        moving["target_visible_confirmed_fraction"] = (
            int(retinal["confirmed_visible_frame_count"])
            / max(1, int(retinal["target_in_camera_fov_matched_frame_count"]))
        )
        moving["target_visibility_status"] = retinal["status"]
        pair = _pair_behavior_delta(moving, hidden, retinal)
        behavior_pairs.append(pair)
    for static, hidden in static_hidden_pairs:
        retinal = _matched_retinal_delta(static, hidden, selected_criteria)
        static["retinal_visibility_vs_hidden"] = retinal
        static["target_visible_confirmed_frame_count"] = int(retinal["confirmed_visible_frame_count"])
        static["target_visible_confirmed_fraction"] = (
            int(retinal["confirmed_visible_frame_count"])
            / max(1, int(retinal["target_in_camera_fov_matched_frame_count"]))
        )
        static["target_visibility_status"] = retinal["status"]

    paired_gains = [
        float(pair["moving_minus_hidden_alignment_gain_rad"])
        for pair in behavior_pairs
        if _number(pair.get("moving_minus_hidden_alignment_gain_rad")) is not None
    ]
    hidden_gain_threshold = float(selected_criteria["moving_vs_hidden_alignment_gain_rad"])
    paired_delta_summary = _comparison_summary(paired_gains, hidden_gain_threshold)
    moving_static_behavior_pairs = [
        _pair_behavior_delta(moving, static)
        for moving, static in moving_static_pairs
    ]
    moving_static_gains = [
        float(pair["condition_minus_control_alignment_gain_rad"])
        for pair in moving_static_behavior_pairs
        if _number(pair.get("condition_minus_control_alignment_gain_rad")) is not None
    ]
    static_gain_threshold = float(selected_criteria["moving_vs_static_alignment_gain_rad"])
    moving_static_delta_summary = _comparison_summary(
        moving_static_gains, static_gain_threshold, control_name="static",
    )
    moving_gains = [
        float(_mapping(trial.get("behavior")).get("yaw_alignment_gain_rad"))
        for trial in moving_trials
        if _number(_mapping(trial.get("behavior")).get("yaw_alignment_gain_rad")) is not None
    ]
    moving_turn_fractions = [
        float(_mapping(trial.get("behavior")).get("target_directed_yaw_fraction"))
        for trial in moving_trials
        if _number(_mapping(trial.get("behavior")).get("target_directed_yaw_fraction")) is not None
    ]
    visible_trial_count = sum(
        bool(_mapping(trial.get("retinal_visibility_vs_hidden")).get("visible_confirmed"))
        for trial in moving_trials
    )
    minimum_pairs = int(selected_criteria["minimum_paired_seeds"])
    minimum_visible_frames = int(selected_criteria["minimum_confirmed_visible_frames_per_trial"])
    minimum_fov_frames = int(selected_criteria["minimum_target_in_fov_frames_per_trial"])
    matched_moving_trials = [moving for moving, _hidden in target_hidden_pairs]
    target_visible_pass = (
        len(target_hidden_pairs) >= minimum_pairs
        and visible_trial_count >= minimum_pairs
        and all(
            int(_mapping(trial.get("retinal_visibility_vs_hidden")).get("target_in_camera_fov_matched_frame_count", 0))
            >= minimum_fov_frames
            and int(_mapping(trial.get("retinal_visibility_vs_hidden")).get("confirmed_visible_frame_count", 0))
            >= minimum_visible_frames
            for trial in matched_moving_trials
        )
    )
    target_orientation_pass = (
        len(paired_gains) >= minimum_pairs
        and _median(paired_gains) is not None
        and float(_median(paired_gains) or 0.0) >= hidden_gain_threshold
        and _median(moving_gains) is not None
        and float(_median(moving_gains) or 0.0) >= float(selected_criteria["alignment_gain_rad"])
        and _median(moving_turn_fractions) is not None
        and float(_median(moving_turn_fractions) or 0.0) >= float(selected_criteria["minimum_target_directed_yaw_fraction"])
    )
    target_tracking_pass = (
        len(moving_static_pairs) >= minimum_pairs
        and len(moving_static_gains) >= minimum_pairs
        and _median(moving_static_gains) is not None
        and float(_median(moving_static_gains) or 0.0) >= static_gain_threshold
    )
    target_behavior_pass = target_orientation_pass and target_tracking_pass
    protocol_valid = all(
        trial.get("courtship_visual_manifest_valid")
        and trial.get("evidence_class") == "digital_simulation"
        and trial.get("strict_free_behavior")
        for trial in summaries
    )
    if (
        not protocol_valid
        or len(target_hidden_pairs) < minimum_pairs
        or len(paired_gains) < minimum_pairs
        or visible_trial_count < minimum_pairs
        or len(moving_static_pairs) < minimum_pairs
        or len(moving_static_gains) < minimum_pairs
    ):
        verdict = "INCONCLUSIVE"
    elif target_visible_pass and target_behavior_pass:
        verdict = "VISUAL_ORIENTATION_TRACKING_SUPPORTED"
    else:
        verdict = "VISUAL_ORIENTATION_TRACKING_NOT_SUPPORTED_BY_PREDEFINED_CRITERIA"

    effect_report = {
        "moving_vs_hidden": {
            **paired_delta_summary,
            "matched_pairs": behavior_pairs,
            "interpretation": "condition effect includes target visibility and any resulting initial scene adaptation; it is not a motion-only contrast",
        },
        "moving_vs_static": {
            **moving_static_delta_summary,
            "matched_pairs": moving_static_behavior_pairs,
            "interpretation": "primary matched motion contrast: both conditions render the same target at the same first pose, then only the moving condition follows the independent trajectory",
        },
        "static_vs_hidden": {
            "pair_count": len(static_hidden_pairs),
            "matched_seeds": sorted({moving["seed"] for moving, _hidden in static_hidden_pairs}, key=str),
            "matched_pairs": [
                _pair_behavior_delta(static, hidden, _mapping(static.get("retinal_visibility_vs_hidden")))
                for static, hidden in static_hidden_pairs
            ],
            "interpretation": "static-control retinal and behavior metrics are reported descriptively; the visible-versus-hidden comparison includes the target-visibility/context change and is not a motion-only contrast",
        },
    }

    for trial in summaries:
        trial.pop("_frames", None)
        trial.pop("_paths", None)

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "digital_simulation",
        "assay_scope": "visual orientation and tracking only",
        "full_courtship_endpoint": "not_assessed",
        "P1_state": {"status": "not_measured_or_inferred_by_this_analyzer"},
        "pC1_labeled_spike_proxy_note": (
            "prefix:pC1_ totals and mean rates are reported as selector-labelled MaleCNS telemetry only; they are not a validated P1 state or arousal measure. P1_state remains unmeasured."
        ),
        "criteria": {
            **selected_criteria,
            "visibility_rule": (
                "target geometry must place the target within the camera frustum and exact left/right retinal vectors must differ from a same-seed, frame/time/pose-matched hidden-target control; renderable state or centre position alone never confirms visibility"
            ),
            "orientation_tracking_rule": (
                "at least minimum_paired_seeds matched seeds in both contrasts; the moving target must be visibly confirmed against hidden controls; orientation requires the moving-versus-hidden alignment and target-directed-turn criteria; motion-specific tracking additionally requires median moving-minus-static within-run yaw-alignment gain to meet moving_vs_static_alignment_gain_rad. Moving and static controls share the first target pose and initial visual adaptation."
            ),
            "statistical_scope": "descriptive paired thresholds; no p-value or biological inference",
            "FOV_scope": "bounding-sphere frustum estimate; does not resolve mesh occlusion",
            "pursuit_scope": "velocity projected toward the target and relative closing speed are descriptive and are not required for the orientation/tracking verdict",
        },
        "verdict": verdict,
        "criterion_results": {
            "strict_digital_courtship_protocol": "PASS" if protocol_valid else "INCONCLUSIVE",
            "moving_target_visible_vs_hidden": "PASS" if target_visible_pass else (
                "FAIL" if len(target_hidden_pairs) >= minimum_pairs else "INCONCLUSIVE"
            ),
            "moving_target_orientation_vs_hidden": "PASS" if target_orientation_pass else (
                "FAIL" if len(paired_gains) >= minimum_pairs else "INCONCLUSIVE"
            ),
            "moving_target_tracking_vs_static": "PASS" if target_tracking_pass else (
                "FAIL" if len(moving_static_gains) >= minimum_pairs else "INCONCLUSIVE"
            ),
        },
        "counts": {
            "trial_count": len(summaries),
            "moving_trial_count": len(moving_trials),
            "hidden_trial_count": len(hidden_trials),
            "static_trial_count": len(static_trials),
            "matched_moving_hidden_pair_count": len(target_hidden_pairs),
            "matched_moving_static_pair_count": len(moving_static_pairs),
            "retinally_confirmed_moving_trial_count": visible_trial_count,
        },
        "condition_effects": effect_report,
        "trials": summaries,
        "limitations": [
            "A positive result supports only target-oriented body motion in this digital assay, not full courtship or mating.",
            "No P1 state is inferred unless a direct, validated P1 channel is separately measured; this report does not test that channel.",
            "The prefix:pC1_ spike-count/rate proxy is not a validated P1 state or arousal measure.",
            "A target's presence in the camera frustum does not resolve occlusion; paired retinal differences are the visibility evidence.",
            "Male-to-target projected velocity and distance closing can reflect target and body dynamics and are descriptive measures.",
            "The paired threshold summary is not a statistical significance test and is not biological validation.",
        ],
    }


def _parse_trial_argument(value: str) -> TrialSpec:
    if "=" not in value:
        return TrialSpec(path=value)
    pieces = value.split("=", 2)
    if len(pieces) == 2:
        condition, path = pieces
        label = None
    else:
        label, condition, path = pieces
    if not condition.strip() or not path.strip() or (len(pieces) == 3 and not label.strip()):
        raise argparse.ArgumentTypeError(
            "--trial format is CONDITION=RUN_DIR or LABEL=CONDITION=RUN_DIR"
        )
    return TrialSpec(
        path=path.strip(), condition=condition.strip(),
        label=label.strip() if label else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial", action="append", type=_parse_trial_argument, required=True,
        metavar="LABEL=CONDITION=RUN_DIR", help="repeat for moving, hidden, and optional static runs; LABEL= may be omitted",
    )
    parser.add_argument("--moving-condition", default="moving")
    parser.add_argument("--hidden-condition", default="hidden")
    parser.add_argument("--static-condition", default="static")
    parser.add_argument("--minimum-paired-seeds", type=int, default=int(DEFAULT_CRITERIA["minimum_paired_seeds"]))
    parser.add_argument("--output", type=Path, default=None,
                        help="optional JSON report destination; the caller chooses its /data path")
    args = parser.parse_args(argv)
    criteria = dict(DEFAULT_CRITERIA)
    criteria["minimum_paired_seeds"] = args.minimum_paired_seeds
    try:
        report = analyze_courtship_visual(
            args.trial,
            moving_condition=args.moving_condition,
            hidden_condition=args.hidden_condition,
            static_condition=args.static_condition,
            criteria=criteria,
        )
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output is None:
            sys.stdout.write(payload)
        else:
            args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            args.output.expanduser().resolve().write_text(payload, encoding="utf-8")
    except (CourtshipVisualAnalysisError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
