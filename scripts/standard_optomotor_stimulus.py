#!/usr/bin/env python3
"""Optional CPU angular grating for Mano-style optomotor calibration.

This module is deliberately independent of ``PhysicalStimulus`` and the live
worker.  It describes a fixed-arena angular field as an infinite-distance
screen approximation and returns sampled luminance columns.  It does not
reconstruct finite-radius cylinder geometry or translation parallax.  A future online worker may
explicitly call it after transforming each head-frame eye ray into the fixed
arena frame.  It never emits a motor command.

Coordinate conventions
----------------------
The public direction label is expressed in the initial FlyGym head frame:
``head azimuth = atan2(head_y, head_x)`` and positive azimuth points toward
head ``+Y`` (fly-left).  VirtualWorld mirrors FlyGym's Y axis.  At zero yaw,
head ``+Y`` rays therefore have negative VirtualWorld/arena azimuth.  The
class stores this sign mapping explicitly when producing a fixed-arena field.

``render_world_directions`` accepts unit rays already transformed into the
static arena frame.  This is the entry point for a future live loop: the caller
must transform current head rays with the current body pose before sampling.
``render_head_directions`` is a convenience for fixed-pose calibration only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


MANO_PAPER_DOI = "10.7554/eLife.86076"
MANO_DATA_DOI = "10.5281/zenodo.19424913"
MANO_WAVELENGTH_DEG = 60.0
MANO_TEMPORAL_FREQUENCY_HZ = 1.0
MANO_AZIMUTH_EXTENT_DEG = (-135.0, 135.0)
MANO_ELEVATION_EXTENT_DEG = (-45.0, 45.0)
MANO_PRE_GRAY_S = 5.0
MANO_MOVING_DURATION_S = 5.0
MANO_POST_GRAY_S = 5.0
MANO_MEAN_LUMINANCE = 0.5
MANO_CONTRASTS = (1.0, 0.25)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class AngularOptomotorStimulus:
    """One direction/contrast arm of a panoramic sinusoidal grating.

    Parameters are normalized display values rather than photometric units.
    The ``contrast`` parameter is the author's study label ``c``; the local
    normalized luminance formula is ``mean * (1 + c*cos(phase))``.  Thus for
    mean 0.5, c=1 yields 0..1 and c=0.25 yields 0.375..0.625.  This reproduces
    the contrast numbers and mean-luminance setting, not the paper's physical
    100 cd/m² screen calibration.
    """

    contrast: float = 1.0
    head_direction_sign: int = 1
    wavelength_deg: float = MANO_WAVELENGTH_DEG
    temporal_frequency_hz: float = MANO_TEMPORAL_FREQUENCY_HZ
    mean_luminance: float = MANO_MEAN_LUMINANCE
    azimuth_extent_deg: tuple[float, float] = MANO_AZIMUTH_EXTENT_DEG
    elevation_extent_deg: tuple[float, float] = MANO_ELEVATION_EXTENT_DEG
    onset_s: float = MANO_PRE_GRAY_S
    duration_s: float = MANO_MOVING_DURATION_S
    phase_at_onset_rad: float = 0.0
    name: str = "angular-optomotor"

    def __post_init__(self) -> None:
        contrast = _finite(self.contrast, "contrast")
        if not 0.0 <= contrast <= 1.0:
            raise ValueError("contrast must be in [0, 1]")
        if isinstance(self.head_direction_sign, bool) or self.head_direction_sign not in (-1, 1):
            raise ValueError("head_direction_sign must be +1 or -1")
        wavelength = _finite(self.wavelength_deg, "wavelength_deg")
        if wavelength <= 0.0:
            raise ValueError("wavelength_deg must be positive")
        frequency = _finite(self.temporal_frequency_hz, "temporal_frequency_hz")
        if frequency < 0.0:
            raise ValueError("temporal_frequency_hz must be non-negative")
        mean = _finite(self.mean_luminance, "mean_luminance")
        if not 0.0 <= mean <= 1.0 or mean * (1.0 + contrast) > 1.0:
            raise ValueError("mean_luminance and contrast must keep luminance in [0, 1]")
        az = tuple(_finite(v, "azimuth_extent_deg") for v in self.azimuth_extent_deg)
        el = tuple(_finite(v, "elevation_extent_deg") for v in self.elevation_extent_deg)
        if len(az) != 2 or az[0] >= az[1] or az[0] < -180.0 or az[1] > 180.0:
            raise ValueError("azimuth_extent_deg must be an increasing interval within [-180, 180]")
        if len(el) != 2 or el[0] >= el[1] or el[0] < -90.0 or el[1] > 90.0:
            raise ValueError("elevation_extent_deg must be an increasing interval within [-90, 90]")
        if (el[0] + el[1]) != 0.0:
            raise ValueError("this symmetric vertical field must have bounds around zero elevation")
        onset = _finite(self.onset_s, "onset_s")
        duration = _finite(self.duration_s, "duration_s")
        phase = _finite(self.phase_at_onset_rad, "phase_at_onset_rad")
        if onset < 0.0 or duration <= 0.0:
            raise ValueError("onset_s must be non-negative and duration_s must be positive")
        object.__setattr__(self, "contrast", contrast)
        object.__setattr__(self, "wavelength_deg", wavelength)
        object.__setattr__(self, "temporal_frequency_hz", frequency)
        object.__setattr__(self, "mean_luminance", mean)
        object.__setattr__(self, "azimuth_extent_deg", az)
        object.__setattr__(self, "elevation_extent_deg", el)
        object.__setattr__(self, "onset_s", onset)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "phase_at_onset_rad", phase)
        object.__setattr__(self, "name", str(self.name))

    @property
    def angular_speed_deg_s(self) -> float:
        """Pattern speed in the initial head-azimuth convention (+ is fly-left)."""
        return float(self.head_direction_sign * self.wavelength_deg * self.temporal_frequency_hz)

    @property
    def arena_angular_speed_deg_s(self) -> float:
        """Signed speed of the same pattern in VirtualWorld's mirrored Y frame."""
        return -self.angular_speed_deg_s

    @property
    def offset_range(self) -> tuple[float, float]:
        return (
            self.mean_luminance * (1.0 - self.contrast),
            self.mean_luminance * (1.0 + self.contrast),
        )

    @property
    def end_s(self) -> float:
        return self.onset_s + self.duration_s

    def field_mask(self, arena_azimuth_deg: Any, elevation_deg: Any) -> np.ndarray:
        az, el = np.broadcast_arrays(
            np.asarray(arena_azimuth_deg, dtype=np.float64),
            np.asarray(elevation_deg, dtype=np.float64),
        )
        return (
            (az >= self.azimuth_extent_deg[0])
            & (az <= self.azimuth_extent_deg[1])
            & (el >= self.elevation_extent_deg[0])
            & (el <= self.elevation_extent_deg[1])
        )

    def render_arena_angles(
        self, arena_azimuth_deg: Any, elevation_deg: Any, time_s: float,
    ) -> np.ndarray:
        """Sample the fixed-arena angular display at its world azimuth/elevation."""
        time_s = _finite(time_s, "time_s")
        az, el = np.broadcast_arrays(
            np.asarray(arena_azimuth_deg, dtype=np.float64),
            np.asarray(elevation_deg, dtype=np.float64),
        )
        out = np.full(az.shape, self.mean_luminance, dtype=np.float32)
        if not self.onset_s <= time_s < self.end_s:
            return out
        mask = self.field_mask(az, el)
        elapsed = time_s - self.onset_s
        phase = (
            2.0 * math.pi * (
                az[mask] - self.arena_angular_speed_deg_s * elapsed
            ) / self.wavelength_deg
            + self.phase_at_onset_rad
        )
        values = self.mean_luminance * (1.0 + self.contrast * np.cos(phase))
        out[mask] = np.asarray(np.clip(values, 0.0, 1.0), dtype=np.float32)
        return out

    def render_world_directions(self, directions_world_xyz: Any, time_s: float) -> np.ndarray:
        """Render rays already transformed into the static arena coordinate frame.

        The arena convention follows VirtualWorld: +X is initial forward, +Z is
        up, and +Y is fly-right at zero yaw.  Pattern direction itself remains
        labeled in initial FlyGym head coordinates; the sign inversion is
        explicit in ``arena_angular_speed_deg_s``.
        """
        directions = np.asarray(directions_world_xyz, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions_world_xyz must have shape (N, 3)")
        norms = np.linalg.norm(directions, axis=1)
        if not np.all(np.isfinite(directions)) or np.any(norms <= 1e-12):
            raise ValueError("world directions must be finite non-zero vectors")
        unit = directions / norms[:, None]
        azimuth = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
        elevation = np.degrees(np.arctan2(unit[:, 2], np.hypot(unit[:, 0], unit[:, 1])))
        return self.render_arena_angles(azimuth, elevation, time_s)

    def render_head_directions(self, directions_head_xyz: Any, time_s: float) -> np.ndarray:
        """Fixed-pose convenience; assumes the initial zero-yaw orientation.

        Head ``+Y`` is fly-left, while VirtualWorld arena ``+Y`` is fly-right;
        convert the sampled head azimuth into arena azimuth before rendering.
        For moving bodies call ``render_world_directions`` after transforming
        the current eye rays through the live body pose.
        """
        directions = np.asarray(directions_head_xyz, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions_head_xyz must have shape (N, 3)")
        norms = np.linalg.norm(directions, axis=1)
        if not np.all(np.isfinite(directions)) or np.any(norms <= 1e-12):
            raise ValueError("head directions must be finite non-zero vectors")
        unit = directions / norms[:, None]
        # FlyGym head +Y -> VirtualWorld arena -Y at the initial pose.
        arena_directions = unit * np.asarray((1.0, -1.0, 1.0), dtype=np.float64)
        return self.render_world_directions(arena_directions, time_s)

    def to_dict(self) -> dict[str, Any]:
        low, high = self.offset_range
        return {
            "name": self.name,
            "pattern": "vertical sinusoidal grating varying across azimuth",
            "direction_sign": self.head_direction_sign,
            "direction_frame": "initial FlyGym head azimuth atan2(head_y, head_x); +head_Y is fly-left",
            "arena_frame": "fixed VirtualWorld/world arena; +X forward, +Y fly-right at zero yaw, +Z up",
            "head_angular_speed_deg_s": self.angular_speed_deg_s,
            "arena_angular_speed_deg_s": self.arena_angular_speed_deg_s,
            "wavelength_deg": self.wavelength_deg,
            "temporal_frequency_hz": self.temporal_frequency_hz,
            "contrast_study_parameter_c": self.contrast,
            "local_contrast_formula": "L = mean_luminance * (1 + c*cos(2*pi*(arena_azimuth - arena_speed*(t-onset))/wavelength + phase))",
            "normalized_luminance_min_max": [low, high],
            "mean_luminance_normalized": self.mean_luminance,
            "azimuth_extent_deg": list(self.azimuth_extent_deg),
            "elevation_extent_deg": list(self.elevation_extent_deg),
            "outside_field_and_prepost_luminance": self.mean_luminance,
            "onset_s": self.onset_s,
            "duration_s": self.duration_s,
            "end_exclusive_s": self.end_s,
            "phase_at_onset_rad": self.phase_at_onset_rad,
            "display_geometry": "fixed arena angular field; infinite-distance angular-screen approximation, not a finite-radius cylinder",
            "translation_parallax": "none; body translation does not change screen angles or induce panorama parallax",
            "scene_occlusion": "none; the angular field is the complete stimulus surface inside its field mask",
            "lighting": "direct normalized monochrome luminance samples; no Lambert shading or world texture",
            "primary_reference": {
                "paper_doi": MANO_PAPER_DOI,
                "data_doi": MANO_DATA_DOI,
                "reference_geometry": "front 270 degrees around fly; approximately +/-45 degrees elevation",
            },
            "photometric_limit": "normalized pixel levels match the author luminance parameter and c labels; physical cd/m^2 is not calibrated",
        }


def mano_reference_conditions(
    *, onset_s: float = MANO_PRE_GRAY_S,
    duration_s: float = MANO_MOVING_DURATION_S,
    mean_luminance: float = MANO_MEAN_LUMINANCE,
) -> tuple[AngularOptomotorStimulus, ...]:
    """Return high/low contrast by signed angular direction, with fixed order."""
    result = []
    for contrast in MANO_CONTRASTS:
        for sign, direction in ((1, "positive-head-azimuth"), (-1, "negative-head-azimuth")):
            result.append(AngularOptomotorStimulus(
                name=f"{direction}-c{contrast:g}",
                contrast=contrast,
                head_direction_sign=sign,
                onset_s=onset_s,
                duration_s=duration_s,
                mean_luminance=mean_luminance,
            ))
    return tuple(result)


def standard_calibration_conditions(
    *, onset_s: float = MANO_PRE_GRAY_S,
    duration_s: float = MANO_MOVING_DURATION_S,
    mean_luminance: float = MANO_MEAN_LUMINANCE,
) -> tuple[AngularOptomotorStimulus, ...]:
    """Return c={1,.25} and angular speeds 0, ±30, ±60 deg/s.

    The moving arms correspond to temporal frequencies 0.5 and 1 Hz at the
    60-degree spatial wavelength.  Zero speed is one stationary grating per
    contrast; it has no left/right sign.
    """
    result: list[AngularOptomotorStimulus] = []
    for contrast in MANO_CONTRASTS:
        result.append(AngularOptomotorStimulus(
            name=f"stationary-c{contrast:g}", contrast=contrast,
            head_direction_sign=1, temporal_frequency_hz=0.0,
            onset_s=onset_s, duration_s=duration_s,
            mean_luminance=mean_luminance,
        ))
        for temporal_frequency in (0.5, 1.0):
            for sign, direction in ((1, "positive-head-azimuth"), (-1, "negative-head-azimuth")):
                speed = sign * MANO_WAVELENGTH_DEG * temporal_frequency
                result.append(AngularOptomotorStimulus(
                    name=f"{direction}-{abs(speed):g}degps-c{contrast:g}",
                    contrast=contrast,
                    head_direction_sign=sign,
                    temporal_frequency_hz=temporal_frequency,
                    onset_s=onset_s,
                    duration_s=duration_s,
                    mean_luminance=mean_luminance,
                ))
    return tuple(result)


__all__ = [
    "AngularOptomotorStimulus",
    "mano_reference_conditions",
    "standard_calibration_conditions",
]
