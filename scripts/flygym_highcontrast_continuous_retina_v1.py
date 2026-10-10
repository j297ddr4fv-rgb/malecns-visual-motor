#!/usr/bin/env python3
"""Visual-only continuous-phase schedule adapter for long C=1 online runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "flygym-highcontrast-continuous-retina-schedule-v1"
ADAPTER_ID = "flygym-highcontrast-continuous-retina-v1"
FRAME_DT_S = 0.005
WAVELENGTH_DEG = 60.0
MEAN_LUMINANCE = 0.5
SCHEDULE_ENV_PATH = "FLYGYM_HIGHCONTRAST_SCHEDULE_PATH"
SCHEDULE_ENV_SHA256 = "FLYGYM_HIGHCONTRAST_SCHEDULE_SHA256"
SCHEDULE_ENV_CASE = "FLYGYM_HIGHCONTRAST_CASE_ID"


class ScheduleError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScheduleError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ScheduleError(f"{label} must be finite")
    return result


def validate_schedule(document: Any, expected_case_id: str) -> None:
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ScheduleError(f"schedule must use {SCHEMA_VERSION}")
    if document.get("case_id") != expected_case_id:
        raise ScheduleError("schedule case ID differs from worker launch pin")
    clock = document.get("clock", {})
    if (clock.get("frame_dt_s") != FRAME_DT_S
            or clock.get("duration_s") != 13.0
            or clock.get("gray_end_s") != 1.0
            or clock.get("switch_s") != 7.0):
        raise ScheduleError("schedule clock must be 13 s, 200 Hz, gray to t=1 and switch at t=7")
    if document.get("motor_authority") != "none; schedule selects retinal luminance only":
        raise ScheduleError("visual schedule must declare no motor authority")
    segments = document.get("segments")
    if not isinstance(segments, list) or len(segments) not in (2, 3):
        raise ScheduleError("schedule requires gray plus one or two contiguous grating segments")
    expected_start = 0.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ScheduleError(f"segment {index} must be an object")
        start = _finite(segment.get("start_s"), f"segments[{index}].start_s")
        end = _finite(segment.get("end_s"), f"segments[{index}].end_s")
        for bound in (start, end):
            if abs(bound / FRAME_DT_S - round(bound / FRAME_DT_S)) > 1e-9:
                raise ScheduleError(f"segment {index} boundary is not aligned to a 5 ms frame")
        if start != expected_start or end <= start:
            raise ScheduleError("segments must be contiguous, non-empty, and ordered")
        visual = segment.get("visual")
        if not isinstance(visual, dict) or visual.get("kind") not in {"uniform_gray", "sinusoidal_grating"}:
            raise ScheduleError(f"segment {index} has unsupported visual kind")
        if visual.get("mean_luminance") != MEAN_LUMINANCE:
            raise ScheduleError("every segment must preserve mean luminance 0.5")
        contrast = _finite(visual.get("contrast"), f"segments[{index}].contrast")
        sign = visual.get("head_direction_sign")
        frequency = _finite(visual.get("temporal_frequency_hz"), f"segments[{index}].temporal_frequency_hz")
        phase = _finite(visual.get("phase_at_onset_rad"), f"segments[{index}].phase_at_onset_rad")
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ScheduleError("head direction sign must be +1 or -1")
        if visual["kind"] == "uniform_gray":
            if index != 0 or (start, end, contrast, frequency) != (0.0, 1.0, 0.0, 0.0):
                raise ScheduleError("the first segment must be exactly one second of uniform gray")
        else:
            if contrast != 1.0 or frequency not in (0.0, 1.0):
                raise ScheduleError("C=1 grating frequency must be 0 or 1 Hz")
            if visual.get("wavelength_deg") != WAVELENGTH_DEG:
                raise ScheduleError("spatial wavelength must be 60 degrees")
            expected_speed = sign * WAVELENGTH_DEG * frequency
            if _finite(visual.get("direction_deg_s"), "direction_deg_s") != expected_speed:
                raise ScheduleError("direction_deg_s must be fly-left-positive initial-head grating speed")
            if _finite(visual.get("arena_angular_speed_deg_s"), "arena_angular_speed_deg_s") != -expected_speed:
                raise ScheduleError("fixed-arena grating speed must invert initial-head direction sign")
            if not math.isfinite(phase):
                raise ScheduleError("spatial phase must be finite")
        expected_start = end
    if expected_start != 13.0:
        raise ScheduleError("segments must cover the full [0,13) s interval")
    conditions = document.get("condition", {})
    direction = conditions.get("direction") if isinstance(conditions, dict) else None
    grating_segments = [s for s in segments if s["visual"]["kind"] == "sinusoidal_grating"]
    allowed = {"static_c1", "positive60", "negative60", "positive_to_negative", "negative_to_positive"}
    if direction not in allowed:
        raise ScheduleError("condition label is not one of the five registered C=1 cases")
    if direction == "static_c1" and (len(grating_segments) != 1 or grating_segments[0]["visual"]["temporal_frequency_hz"] != 0.0):
        raise ScheduleError("static C=1 case must contain one 12 s static grating")
    if direction in {"positive60", "negative60"} and (
            len(grating_segments) != 1 or grating_segments[0]["visual"]["temporal_frequency_hz"] != 1.0):
        raise ScheduleError("constant-motion C=1 case must contain one 12 s moving grating")
    if direction in {"positive_to_negative", "negative_to_positive"}:
        if len(grating_segments) != 2 or [s["start_s"] for s in grating_segments] != [1.0, 7.0]:
            raise ScheduleError("reversal cases must switch exactly at t=7.0 s")
        first, second = grating_segments
        first_visual, second_visual = first["visual"], second["visual"]
        phase_end = float(first_visual["phase_at_onset_rad"]) - (
            2.0 * math.pi * float(first_visual["arena_angular_speed_deg_s"])
            * (float(first["end_s"]) - float(first["start_s"])) / WAVELENGTH_DEG
        )
        phase_start = float(second_visual["phase_at_onset_rad"])
        phase_error = math.remainder(phase_end - phase_start, 2.0 * math.pi)
        if abs(phase_error) > 1e-9:
            raise ScheduleError("spatial phase is discontinuous at the t=7 reversal")
        if second_visual["head_direction_sign"] != -first_visual["head_direction_sign"]:
            raise ScheduleError("reversal segment directions must be opposite")
        required_start_sign = 1 if direction == "positive_to_negative" else -1
        if first_visual["head_direction_sign"] != required_start_sign:
            raise ScheduleError("reversal label does not match its registered initial direction")
        if second_visual["head_direction_sign"] != -required_start_sign:
            raise ScheduleError("reversal label does not match its registered final direction")
    else:
        if len(grating_segments) != 1:
            raise ScheduleError("non-reversal cases must have one grating segment")
        static = direction == "static_c1"
        required_sign = -1 if direction == "negative60" else 1
        if grating_segments[0]["visual"]["head_direction_sign"] != required_sign:
            raise ScheduleError("constant case label does not match its registered direction sign")
        if static and float(grating_segments[0]["visual"]["temporal_frequency_hz"]) != 0.0:
            raise ScheduleError("static case must have zero temporal frequency")
        if not static and float(grating_segments[0]["visual"]["temporal_frequency_hz"]) != 1.0:
            raise ScheduleError("moving constant case must have 1 Hz temporal frequency")


def load_schedule_from_environment() -> tuple[dict[str, Any], Path, str]:
    raw_path = os.environ.get(SCHEDULE_ENV_PATH)
    expected_sha = os.environ.get(SCHEDULE_ENV_SHA256)
    case_id = os.environ.get(SCHEDULE_ENV_CASE)
    if not raw_path or not expected_sha or not case_id:
        raise ScheduleError("schedule path, hash, and case pins are required")
    path = Path(raw_path).expanduser().resolve(strict=True)
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ScheduleError(f"schedule hash mismatch: {actual_sha} != {expected_sha}")
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_schedule(document, case_id)
    return document, path, actual_sha


def phase_record(document: Mapping[str, Any], time_s: float) -> dict[str, Any]:
    t = _finite(time_s, "time_s")
    for segment in document["segments"]:
        start, end = float(segment["start_s"]), float(segment["end_s"])
        if start <= t < end:
            visual = segment["visual"]
            grating = visual["kind"] == "sinusoidal_grating"
            arena_speed = float(visual.get("arena_angular_speed_deg_s", 0.0)) if grating else 0.0
            elapsed = max(0.0, t - start)
            phase_offset = float(visual["phase_at_onset_rad"]) - (
                2.0 * math.pi * arena_speed * elapsed / WAVELENGTH_DEG
            ) if grating else None
            return {
                "segment_id": segment["segment_id"],
                "interval_s_half_open": [start, end],
                "visual_kind": visual["kind"],
                "contrast": float(visual["contrast"]),
                "mean_luminance": MEAN_LUMINANCE,
                "wavelength_deg": WAVELENGTH_DEG if grating else None,
                "direction_deg_s_fly_left_positive": float(visual["direction_deg_s"]) if grating else None,
                "head_direction_sign": int(visual["head_direction_sign"]) if grating else None,
                "arena_angular_speed_deg_s": arena_speed if grating else None,
                "phase_at_segment_onset_rad": float(visual["phase_at_onset_rad"]) if grating else None,
                "spatial_phase_offset_unwrapped_rad": phase_offset,
                "elapsed_in_segment_s": elapsed,
                "retina_only_condition_metadata": True,
                "action_authority": "none; stimulus schedule does not enter decoder or controller",
            }
    raise ScheduleError(f"time {t:g} s is outside [0,13)")


def make_scheduled_class(original_class: type[Any]) -> type[Any]:
    class HighContrastContinuousStimulus:
        def __init__(self, **values: Any) -> None:
            self._base = original_class(**values)
            self.schedule, self.schedule_path, self.schedule_sha256 = load_schedule_from_environment()
            self._last_render_time: float | None = None
            self._segment_renderers: dict[str, Any] = {}
            for segment in self.schedule["segments"]:
                visual = segment["visual"]
                if visual["kind"] != "sinusoidal_grating":
                    continue
                self._segment_renderers[segment["segment_id"]] = original_class(
                    name=f"{self.schedule['case_id']}:{segment['segment_id']}",
                    contrast=float(visual["contrast"]),
                    head_direction_sign=int(visual["head_direction_sign"]),
                    wavelength_deg=float(visual["wavelength_deg"]),
                    temporal_frequency_hz=float(visual["temporal_frequency_hz"]),
                    mean_luminance=float(visual["mean_luminance"]),
                    azimuth_extent_deg=tuple(self._base.azimuth_extent_deg),
                    elevation_extent_deg=tuple(self._base.elevation_extent_deg),
                    onset_s=float(segment["start_s"]),
                    duration_s=float(segment["end_s"] - segment["start_s"]),
                    phase_at_onset_rad=float(visual["phase_at_onset_rad"]),
                )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._base, name)

        @property
        def angular_speed_deg_s(self) -> float:
            if self._last_render_time is None:
                return float(self._base.angular_speed_deg_s)
            return float(phase_record(self.schedule, self._last_render_time).get("direction_deg_s_fly_left_positive") or 0.0)

        @property
        def arena_angular_speed_deg_s(self) -> float:
            if self._last_render_time is None:
                return float(self._base.arena_angular_speed_deg_s)
            return float(phase_record(self.schedule, self._last_render_time).get("arena_angular_speed_deg_s") or 0.0)

        @property
        def onset_s(self) -> float:
            if self._last_render_time is None:
                return float(self._base.onset_s)
            record = phase_record(self.schedule, self._last_render_time)
            return float(record["interval_s_half_open"][0]) if record["visual_kind"] == "sinusoidal_grating" else math.inf

        @property
        def end_s(self) -> float:
            if self._last_render_time is None:
                return float(self._base.end_s)
            record = phase_record(self.schedule, self._last_render_time)
            return float(record["interval_s_half_open"][1]) if record["visual_kind"] == "sinusoidal_grating" else -math.inf

        def render_world_directions(self, directions_world_xyz: Any, time_s: float) -> np.ndarray:
            self._last_render_time = _finite(time_s, "render time")
            record = phase_record(self.schedule, self._last_render_time)
            if record["visual_kind"] == "uniform_gray":
                rays = np.asarray(directions_world_xyz)
                if rays.ndim != 2 or rays.shape[1] != 3:
                    raise ScheduleError("world directions must have shape (N,3)")
                return np.full((rays.shape[0],), MEAN_LUMINANCE, dtype=np.float32)
            renderer = self._segment_renderers[record["segment_id"]]
            return renderer.render_world_directions(directions_world_xyz, self._last_render_time)

        def to_dict(self) -> dict[str, Any]:
            result = self._base.to_dict()
            result.update({
                "runtime_schedule_adapter_id": ADAPTER_ID,
                "runtime_schedule_adapter_path": str(Path(__file__).resolve()),
                "runtime_schedule_adapter_sha256": sha256_file(Path(__file__)),
                "runtime_schedule_path": str(self.schedule_path),
                "runtime_schedule_sha256": self.schedule_sha256,
                "runtime_schedule": self.schedule,
                "phase_continuity_rule": "integrate arena pattern displacement across segments; no phase reset at t=7",
                "motor_authority": "none; schedule selects retinal luminance only",
            })
            return result

    HighContrastContinuousStimulus.__name__ = "HighContrastContinuousStimulusV1"
    HighContrastContinuousStimulus.__qualname__ = "HighContrastContinuousStimulusV1"
    return HighContrastContinuousStimulus


def install_worker_renderer() -> dict[str, str]:
    import standard_optomotor_stimulus

    standard_optomotor_stimulus.AngularOptomotorStimulus = make_scheduled_class(
        standard_optomotor_stimulus.AngularOptomotorStimulus
    )
    schedule, path, digest = load_schedule_from_environment()
    return {
        "adapter_id": ADAPTER_ID,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(Path(__file__)),
        "schedule_path": str(path),
        "schedule_sha256": digest,
        "case_id": schedule["case_id"],
    }
