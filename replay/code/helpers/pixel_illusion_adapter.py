"""Pixel-only retinal schedules for visual-illusion engineering captures.

The live worker still owns the eye rays and angular display geometry.  This
adapter only maps those world rays to normalized luminance values; it has no
decoder or controller interface.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "flygym-visual-illusion-pixels-v1"
ADAPTER_ID = "malecns-visual-illusions-pixel-adapter-v1"
FRAME_DT_S = 0.005
MEAN = 0.5
WAVELENGTH_DEG = 60.0
MOTOR_AUTHORITY = "none; schedule selects retinal luminance only"


class IllusionScheduleError(ValueError):
    """Invalid or incompatible pixel schedule."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise IllusionScheduleError(f"{label} must be finite")
    return result


def validate_schedule(doc: Any, case_id: str) -> None:
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA:
        raise IllusionScheduleError(f"schedule must use {SCHEMA}")
    if doc.get("adapter_id") != ADAPTER_ID:
        raise IllusionScheduleError("schedule adapter identity is invalid")
    if doc.get("case_id") != case_id or doc.get("motor_authority") != MOTOR_AUTHORITY:
        raise IllusionScheduleError("case identity or motor-authority declaration is invalid")
    clock = doc.get("clock", {})
    dt = _finite(clock.get("frame_dt_s", 0), "frame_dt_s")
    duration = _finite(clock.get("duration_s", 0), "duration_s")
    count = int(clock.get("frame_count", -1))
    if (dt != FRAME_DT_S or duration <= 0 or count != round(duration / dt)
            or clock.get("interval_convention") != "half-open [start_s,end_s)"):
        raise IllusionScheduleError("clock must use the 5 ms frame grid and a matching frame count")

    segments = doc.get("segments")
    if not isinstance(segments, list) or not segments:
        raise IllusionScheduleError("segments must be a non-empty list")
    at = 0.0
    allowed = {
        "uniform_gray", "stepped_grating", "reversephi_grating", "flicker_grating",
        "counterphase_grating", "interleaved_stepped_grating", "stationary_sawtooth",
        "continuous_grating",
    }
    for index, segment in enumerate(segments):
        start, end = _finite(segment.get("start_s"), "segment start"), _finite(segment.get("end_s"), "segment end")
        if (abs(start / dt - round(start / dt)) > 1e-9 or abs(end / dt - round(end / dt)) > 1e-9
                or start != at or end <= start or end > duration):
            raise IllusionScheduleError("segments must be contiguous, ordered, and frame-aligned")
        visual = segment.get("visual", {})
        kind = visual.get("kind")
        if kind not in allowed:
            raise IllusionScheduleError(f"unsupported visual kind in segment {index}: {kind!r}")
        if kind == "uniform_gray":
            if visual.get("mean_luminance") != MEAN:
                raise IllusionScheduleError("uniform gray must remain at mean luminance 0.5")
        else:
            if visual.get("mean_luminance", MEAN) != MEAN:
                raise IllusionScheduleError("all stimuli must retain mean luminance 0.5")
            wavelength = _finite(visual.get("wavelength_deg", WAVELENGTH_DEG), "wavelength_deg")
            contrast = _finite(visual.get("contrast", 1.0), "contrast")
            if wavelength <= 0 or not 0.0 <= contrast <= 1.0:
                raise IllusionScheduleError("wavelength or contrast is outside the normalized display range")
            if kind in {"stepped_grating", "reversephi_grating", "flicker_grating",
                        "interleaved_stepped_grating"}:
                interval = _finite(visual.get("step_interval_s"), "step_interval_s")
                step = _finite(visual.get("step_deg"), "step_deg")
                if (interval <= 0 or abs(interval / dt - round(interval / dt)) > 1e-9
                        or (step == 0 and kind != "flicker_grating")):
                    raise IllusionScheduleError("step size and interval must be nonzero and frame-aligned")
            if kind == "counterphase_grating":
                frequency = _finite(visual.get("temporal_frequency_hz"), "temporal_frequency_hz")
                amplitude = _finite(visual.get("component_amplitude", 0.25), "component_amplitude")
                if frequency <= 0 or amplitude < 0 or 2 * amplitude > 0.5 + 1e-12:
                    raise IllusionScheduleError("counterphase component amplitudes must sum to at most 0.5")
            if kind == "interleaved_stepped_grating":
                period = _finite(visual.get("step_interval_s"), "step_interval_s")
                gap = _finite(visual.get("gray_gap_s"), "gray_gap_s")
                if gap <= 0 or gap >= period or abs(gap / dt - round(gap / dt)) > 1e-9:
                    raise IllusionScheduleError("interleave gray gap must be within and aligned to each step period")
            if kind == "continuous_grating":
                speed = _finite(visual.get("direction_deg_s"), "direction_deg_s")
                frequency = _finite(visual.get("temporal_frequency_hz"), "temporal_frequency_hz")
                if frequency < 0:
                    raise IllusionScheduleError("continuous grating frequency must be non-negative")
                if abs(speed) > 0 and not math.isclose(abs(speed), wavelength * frequency, abs_tol=1e-9):
                    raise IllusionScheduleError("continuous grating speed and temporal frequency disagree")
            if kind == "stationary_sawtooth" and int(visual.get("mirror_sign", 0)) not in (-1, 1):
                raise IllusionScheduleError("stationary sawtooth mirror_sign must be -1 or +1")
        at = end
    if not math.isclose(at, duration, abs_tol=1e-12):
        raise IllusionScheduleError("segments must cover the full schedule duration")


def load_schedule() -> tuple[dict, Path, str]:
    path = Path(os.environ["FLYGYM_EXTENSION_SCHEDULE_PATH"]).resolve(strict=True)
    expected = os.environ["FLYGYM_EXTENSION_SCHEDULE_SHA256"]
    case_id = os.environ["FLYGYM_EXTENSION_CASE_ID"]
    actual = sha256_file(path)
    if actual != expected:
        raise IllusionScheduleError("pixel schedule SHA-256 mismatch")
    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_schedule(doc, case_id)
    return doc, path, actual


def _active_segment(doc: Mapping[str, Any], time_s: float) -> tuple[dict, dict]:
    duration = float(doc["clock"]["duration_s"])
    if not 0 <= time_s < duration:
        raise IllusionScheduleError(f"time {time_s:g} s is outside [0,{duration:g})")
    for segment in doc["segments"]:
        if float(segment["start_s"]) <= time_s < float(segment["end_s"]):
            return segment, segment["visual"]
    raise IllusionScheduleError(f"time {time_s:g} s is not covered by a segment")


def phase_record(doc: Mapping[str, Any], time_s: float) -> dict[str, Any]:
    t = _finite(time_s, "time_s")
    segment, visual = _active_segment(doc, t)
    kind = str(visual["kind"])
    rec: dict[str, Any] = {
        "segment_id": segment["segment_id"],
        "interval_s_half_open": [float(segment["start_s"]), float(segment["end_s"])],
        "visual_kind": kind,
        "mean_luminance": MEAN,
        "contrast": 0.0 if kind == "uniform_gray" else float(visual.get("contrast", 1.0)),
        "retina_only_condition_metadata": True,
        "action_authority": MOTOR_AUTHORITY,
    }
    if kind == "uniform_gray":
        rec.update({"gray": True, "direction_deg_s_fly_left_positive": 0.0,
                    "arena_angular_speed_deg_s": 0.0, "temporal_frequency_hz": 0.0})
        return rec

    wavelength = float(visual.get("wavelength_deg", WAVELENGTH_DEG))
    elapsed = t - float(segment["start_s"])
    if kind in {"stepped_grating", "reversephi_grating", "flicker_grating",
                "interleaved_stepped_grating"}:
        interval = float(visual["step_interval_s"])
        index = int(math.floor((elapsed + 1e-12) / interval))
        step = float(visual["step_deg"])
        shift = index * step if kind != "flicker_grating" else 0.0
        rec.update({
            "step_index": index,
            "step_interval_s": interval,
            "step_deg_head_left_positive": step if kind != "flicker_grating" else 0.0,
            "pattern_shift_head_left_positive_deg": shift,
            "pattern_shift_arena_deg": -shift,
            "direction_deg_s_fly_left_positive": step / interval if kind != "flicker_grating" else 0.0,
            "arena_angular_speed_deg_s": -step / interval if kind != "flicker_grating" else 0.0,
            "temporal_frequency_hz": (abs(step / interval) / wavelength
                                       if kind != "flicker_grating" else 1.0 / (2.0 * interval)),
            "polarity_transition_rate_hz": (1.0 / interval
                                            if kind in {"reversephi_grating", "flicker_grating"} else 0.0),
            "gray": False,
        })
        if kind == "reversephi_grating":
            rec["polarity_sign"] = 1 if index % 2 == 0 else -1
            rec["luminance_reversal_about_mean"] = index % 2 == 1
        if kind == "flicker_grating":
            rec["polarity_sign"] = 1 if index % 2 == 0 else -1
            rec["luminance_reversal_about_mean"] = index % 2 == 1
        if kind == "interleaved_stepped_grating":
            gap = float(visual["gray_gap_s"])
            period_frames = int(round(interval / FRAME_DT_S))
            gap_frames = int(round(gap / FRAME_DT_S))
            elapsed_frame = int(round(elapsed / FRAME_DT_S))
            frame_in_period = elapsed_frame % period_frames
            phase_s = frame_in_period * FRAME_DT_S
            rec.update({"gray_gap_s": gap, "step_period_s": interval,
                        "exposure_duration_s": interval - gap,
                        "gray": frame_in_period >= period_frames - gap_frames,
                        "within_step_period_s": phase_s})
    elif kind == "counterphase_grating":
        rec.update({"direction_deg_s_fly_left_positive": 0.0, "arena_angular_speed_deg_s": 0.0,
                    "temporal_frequency_hz": float(visual["temporal_frequency_hz"]),
                    "component_amplitude": float(visual.get("component_amplitude", 0.25)),
                    "total_luminance_amplitude_max": 2.0 * float(visual.get("component_amplitude", 0.25)),
                    "gray": False})
    elif kind == "continuous_grating":
        speed = float(visual["direction_deg_s"])
        rec.update({"direction_deg_s_fly_left_positive": speed,
                    "arena_angular_speed_deg_s": -speed,
                    "temporal_frequency_hz": float(visual["temporal_frequency_hz"]),
                    "phase_at_onset_rad": float(visual.get("phase_at_onset_rad", 0.0)), "gray": False})
    elif kind == "stationary_sawtooth":
        rec.update({"direction_deg_s_fly_left_positive": 0.0, "arena_angular_speed_deg_s": 0.0,
                    "temporal_frequency_hz": 0.0, "mirror_sign": int(visual["mirror_sign"]),
                    "gray": False})
    return rec


def make_scheduled_class(original_class: type):
    class PixelOnlyIllusionStimulus:
        def __init__(self, **values: Any) -> None:
            self._base = original_class(**values)
            self.schedule, self.schedule_path, self.schedule_sha256 = load_schedule()
            clock = self.schedule["clock"]
            self.end_s = float(clock["duration_s"])
            self.duration_s = self.end_s - float(self.schedule.get("stimulus_onset_s", 0.0))
            self.frame_count = int(clock["frame_count"])
            self.onset_s = float(self.schedule.get("stimulus_onset_s", 0.0))
            self._last_render_time: float | None = None

        def _nominal_record(self):
            return phase_record(self.schedule, min(self.end_s - FRAME_DT_S / 2,
                                                   self.onset_s + FRAME_DT_S / 2))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._base, name)

        def _record(self):
            if self._last_render_time is None:
                return None
            return phase_record(self.schedule, self._last_render_time)

        @property
        def angular_speed_deg_s(self) -> float:
            rec = self._nominal_record()
            return float(rec.get("direction_deg_s_fly_left_positive", 0.0)) if rec else 0.0

        @property
        def arena_angular_speed_deg_s(self) -> float:
            rec = self._nominal_record()
            return float(rec.get("arena_angular_speed_deg_s", 0.0)) if rec else 0.0

        @property
        def temporal_frequency_hz(self) -> float:
            return float(self._nominal_record().get("temporal_frequency_hz", 0.0))

        @property
        def contrast(self) -> float:
            return float(self._nominal_record().get("contrast", 1.0))

        @property
        def wavelength_deg(self) -> float:
            _, visual = _active_segment(self.schedule, min(self.end_s - FRAME_DT_S / 2,
                                                           self.onset_s + FRAME_DT_S / 2))
            return float(visual.get("wavelength_deg", getattr(self._base, "wavelength_deg", WAVELENGTH_DEG)))

        @property
        def head_direction_sign(self) -> int:
            nominal_speed = self.angular_speed_deg_s
            if nominal_speed > 0:
                return 1
            if nominal_speed < 0:
                return -1
            return int(self._base.head_direction_sign)

        def render_world_directions(self, directions_world_xyz: Any, time_s: float) -> np.ndarray:
            t = _finite(time_s, "time_s")
            self._last_render_time = t
            rays = np.asarray(directions_world_xyz, dtype=np.float64)
            if rays.ndim != 2 or rays.shape[1] != 3 or not np.all(np.isfinite(rays)):
                raise IllusionScheduleError("world directions must be finite with shape (N,3)")
            norms = np.linalg.norm(rays, axis=1)
            if np.any(norms <= 1e-12):
                raise IllusionScheduleError("world directions must be nonzero")
            unit = rays / norms[:, None]
            az = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
            el = np.degrees(np.arctan2(unit[:, 2], np.hypot(unit[:, 0], unit[:, 1])))
            mask = self._base.field_mask(az, el)
            out = np.full((len(rays),), MEAN, dtype=np.float32)
            segment, visual = _active_segment(self.schedule, t)
            kind = visual["kind"]
            if kind == "uniform_gray":
                return out
            record = phase_record(self.schedule, t)
            wavelength = float(visual.get("wavelength_deg", WAVELENGTH_DEG))
            contrast = float(visual.get("contrast", 1.0))
            elapsed = t - float(segment["start_s"])
            az_in = az[mask]
            if kind == "stepped_grating":
                n = int(record["step_index"])
                phase = 2 * math.pi * (az_in + n * float(visual["step_deg"])) / wavelength
                values = MEAN * (1 + contrast * np.cos(phase))
            elif kind == "reversephi_grating":
                n = int(record["step_index"])
                polarity = float(record["polarity_sign"])
                phase = 2 * math.pi * (az_in + n * float(visual["step_deg"])) / wavelength
                values = MEAN + MEAN * contrast * polarity * np.cos(phase)
            elif kind == "flicker_grating":
                polarity = float(record["polarity_sign"])
                phase = 2 * math.pi * az_in / wavelength
                values = MEAN + MEAN * contrast * polarity * np.cos(phase)
            elif kind == "counterphase_grating":
                frequency = float(visual["temporal_frequency_hz"])
                amplitude = float(visual.get("component_amplitude", 0.25))
                values = MEAN + (2 * amplitude) * np.cos(2 * math.pi * az_in / wavelength) * np.cos(2 * math.pi * frequency * elapsed)
            elif kind == "interleaved_stepped_grating":
                if record["gray"]:
                    return out
                n = int(record["step_index"])
                phase = 2 * math.pi * (az_in + n * float(visual["step_deg"])) / wavelength
                values = MEAN * (1 + contrast * np.cos(phase))
            elif kind == "stationary_sawtooth":
                mirror = int(visual["mirror_sign"])
                ramp = 2.0 * np.mod(mirror * az_in / wavelength, 1.0) - 1.0
                values = MEAN + MEAN * contrast * ramp
            elif kind == "continuous_grating":
                speed_arena = -float(visual["direction_deg_s"])
                phase = (2 * math.pi * (az_in - speed_arena * elapsed) / wavelength
                         + float(visual.get("phase_at_onset_rad", 0.0)))
                values = MEAN * (1 + contrast * np.cos(phase))
            else:
                raise IllusionScheduleError(f"unsupported visual kind: {kind}")
            out[mask] = np.asarray(np.clip(values, 0.0, 1.0), dtype=np.float32)
            return out

        def to_dict(self) -> dict[str, Any]:
            result = self._base.to_dict()
            onset = float(self.schedule.get("stimulus_onset_s", 0.0))
            nominal = self._nominal_record()
            contrast = float(nominal.get("contrast", 1.0))
            result.update({
                "head_angular_speed_deg_s": float(nominal.get("direction_deg_s_fly_left_positive", 0.0)),
                "arena_angular_speed_deg_s": float(nominal.get("arena_angular_speed_deg_s", 0.0)),
                "temporal_frequency_hz": float(nominal.get("temporal_frequency_hz", 0.0)),
                "contrast_study_parameter_c": contrast,
                "normalized_luminance_min_max": [MEAN * (1.0 - contrast), MEAN * (1.0 + contrast)],
            })
            result.update({
                "pattern": "scheduled pixel-only visual stimulus",
                "normalized_luminance_mean": MEAN,
                "azimuth_extent_deg": list(self._base.azimuth_extent_deg),
                "elevation_extent_deg": list(self._base.elevation_extent_deg),
                "outside_field_and_prepost_luminance": MEAN,
                "onset_s": onset,
                "duration_s": self.duration_s,
                "end_exclusive_s": self.end_s,
                "display_geometry": "canonical fixed-arena angular field; per-eye world rays and field mask inherited from AngularOptomotorStimulus",
                "runtime_schedule_adapter_id": ADAPTER_ID,
                "runtime_schedule_adapter_path": str(Path(__file__).resolve()),
                "runtime_schedule_adapter_sha256": sha256_file(Path(__file__).resolve()),
                "runtime_schedule_schema": SCHEMA,
                "runtime_schedule_path": str(self.schedule_path),
                "runtime_schedule_sha256": self.schedule_sha256,
                "runtime_schedule": self.schedule,
                "retina_only_condition_metadata": True,
                "motor_authority": MOTOR_AUTHORITY,
            })
            return result

    return PixelOnlyIllusionStimulus


def install_worker_renderer() -> None:
    import standard_optomotor_stimulus
    standard_optomotor_stimulus.AngularOptomotorStimulus = make_scheduled_class(
        standard_optomotor_stimulus.AngularOptomotorStimulus)
