#!/usr/bin/env python3
"""Deterministic tripod gait and planar ground-body kinematics.

This small, externally driven model is for an engineering crawling demo.  It
does not select behavior, model muscles or contact forces, or claim neural
autonomy.  The body stays at a fixed height above flat ground; three legs are
in stance during an active tripod gait, and only swinging feet are lifted.
Coordinates use SI units, with body +X forward, +Y left, and +Z up.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


Vector3 = tuple[float, float, float]
LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")

# Alternating tripod: left front + right middle + left hind, then its mirror.
_TRIPOD_OFFSETS_RAD = {
    "lf": 0.0,
    "lm": math.pi,
    "lh": 0.0,
    "rf": math.pi,
    "rm": 0.0,
    "rh": math.pi,
}
_LEG_ROOTS_M: Mapping[str, tuple[float, float]] = {
    "lf": (0.00052, 0.00043),
    "lm": (0.0, 0.00048),
    "lh": (-0.00052, 0.00043),
    "rf": (0.00052, -0.00043),
    "rm": (0.0, -0.00048),
    "rh": (-0.00052, -0.00043),
}
# Toes splay laterally beyond their hip roots so the stance footprint is wide
# enough to read as six legs in the ground-walking renderer.
_FOOT_SPLAY_M = 0.00052


@dataclass(frozen=True)
class GroundWalkingConfig:
    """Tunable bounds and gait dimensions for :class:`GroundWalkingBody`."""

    body_height_m: float = 0.0012
    max_speed_mps: float = 0.006
    max_turn_rate_rad_s: float = 2.0
    max_acceleration_mps2: float = 0.025
    max_turn_acceleration_rad_s2: float = 8.0
    step_frequency_hz: float = 3.0
    stance_fraction: float = 0.62
    stride_length_m: float = 0.0012
    foot_lift_m: float = 0.00035

    def __post_init__(self) -> None:
        for name in (
            "body_height_m", "max_speed_mps", "max_turn_rate_rad_s",
            "max_acceleration_mps2", "max_turn_acceleration_rad_s2",
            "step_frequency_hz", "stance_fraction", "stride_length_m",
            "foot_lift_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.body_height_m <= 0.0:
            raise ValueError("body_height_m must be positive")
        if self.max_speed_mps <= 0.0 or self.max_turn_rate_rad_s <= 0.0:
            raise ValueError("speed and turn-rate limits must be positive")
        if self.max_acceleration_mps2 <= 0.0 or self.max_turn_acceleration_rad_s2 <= 0.0:
            raise ValueError("acceleration limits must be positive")
        if self.step_frequency_hz <= 0.0:
            raise ValueError("step_frequency_hz must be positive")
        if not 0.0 < self.stance_fraction < 1.0:
            raise ValueError("stance_fraction must be between zero and one")
        if self.stride_length_m <= 0.0 or self.foot_lift_m < 0.0:
            raise ValueError("stride_length_m must be positive and foot_lift_m nonnegative")


@dataclass(frozen=True)
class GroundWalkingFrame:
    """One body and foot snapshot; vectors are SI and velocity is world-frame."""

    time_s: float
    position_m: Vector3
    yaw_rad: float
    velocity_mps: Vector3
    yaw_rate_rad_s: float
    leg_foot_positions_body_m: Mapping[str, Vector3]
    leg_contact: Mapping[str, bool]
    gait_phase_rad: float


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _approach(current: float, target: float, max_delta: float) -> float:
    """Move a scalar toward its target without exceeding its slew bound."""

    return current + min(max_delta, max(-max_delta, target - current))


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


class GroundWalkingBody:
    """A bounded-speed body driven by normalized forward and turn commands.

    ``forward_drive`` and ``turn_drive`` are finite scalars in ``[-1, 1]``.
    The body accelerates toward the corresponding bounded speed and yaw rate;
    zero drive or ``enabled=False`` brakes at the configured acceleration
    limits. The tripod phase and contacts follow residual braking motion and
    all feet plant when the body reaches rest.
    """

    def __init__(self, config: GroundWalkingConfig | None = None) -> None:
        if config is not None and not isinstance(config, GroundWalkingConfig):
            raise TypeError("config must be GroundWalkingConfig or None")
        self.config = config or GroundWalkingConfig()
        self._time_s = 0.0
        self._position_x_m = 0.0
        self._position_y_m = 0.0
        self._yaw_rad = 0.0
        self._forward_speed_mps = 0.0
        self._yaw_rate_rad_s = 0.0
        self._phase_rad = 0.0

    def step(
        self,
        dt_s: float,
        forward_drive: float,
        turn_drive: float,
        enabled: bool = True,
    ) -> GroundWalkingFrame:
        """Advance body and six feet by one positive time step."""

        dt = _finite_scalar(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        forward = _finite_scalar(forward_drive, "forward_drive")
        turn = _finite_scalar(turn_drive, "turn_drive")
        if not -1.0 <= forward <= 1.0 or not -1.0 <= turn <= 1.0:
            raise ValueError("forward_drive and turn_drive must be in [-1, 1]")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        if not enabled:
            forward = turn = 0.0

        cfg = self.config
        target_speed = forward * cfg.max_speed_mps
        target_yaw_rate = turn * cfg.max_turn_rate_rad_s
        self._forward_speed_mps = _approach(
            self._forward_speed_mps, target_speed, cfg.max_acceleration_mps2 * dt
        )
        self._yaw_rate_rad_s = _approach(
            self._yaw_rate_rad_s,
            target_yaw_rate,
            cfg.max_turn_acceleration_rad_s2 * dt,
        )

        # Integrate planar motion in the body's updated heading.  Height is
        # fixed and the feet below remain constrained to the ground in stance.
        self._yaw_rad = math.remainder(self._yaw_rad + self._yaw_rate_rad_s * dt, 2.0 * math.pi)
        vx = self._forward_speed_mps * math.cos(self._yaw_rad)
        vy = self._forward_speed_mps * math.sin(self._yaw_rad)
        self._position_x_m += vx * dt
        self._position_y_m += vy * dt
        self._time_s += dt

        motion_fraction = max(
            abs(self._forward_speed_mps) / cfg.max_speed_mps,
            abs(self._yaw_rate_rad_s) / cfg.max_turn_rate_rad_s,
        )
        moving = motion_fraction > 1.0e-12
        if moving:
            self._phase_rad = (self._phase_rad + 2.0 * math.pi * cfg.step_frequency_hz * motion_fraction * dt) % (2.0 * math.pi)

        swing_fraction = 1.0 - cfg.stance_fraction
        feet: dict[str, Vector3] = {}
        contacts: dict[str, bool] = {}
        for leg in LEG_NAMES:
            root_x, root_y = _LEG_ROOTS_M[leg]
            foot_y = root_y + math.copysign(_FOOT_SPLAY_M, root_y)
            # Forward motion and yaw both shape the stance sweep. During an
            # in-place turn the left and right sides therefore sweep oppositely.
            leg_drive = self._forward_speed_mps - self._yaw_rate_rad_s * foot_y
            travel_sign = -1.0 if leg_drive < -1.0e-12 else 1.0
            phase = (self._phase_rad + _TRIPOD_OFFSETS_RAD[leg]) % (2.0 * math.pi)
            normalized = phase / (2.0 * math.pi)
            if not moving:
                # A stopped body plants all six toes on the flat ground.
                feet[leg] = (root_x, foot_y, -cfg.body_height_m)
                contacts[leg] = True
            elif normalized < swing_fraction:
                progress = normalized / swing_fraction
                smooth = _smoothstep(progress)
                x = root_x + travel_sign * cfg.stride_length_m * (smooth - 0.5)
                z = -cfg.body_height_m + cfg.foot_lift_m * math.sin(math.pi * progress)
                feet[leg] = (x, foot_y, z)
                contacts[leg] = False
            else:
                progress = (normalized - swing_fraction) / cfg.stance_fraction
                x = root_x + travel_sign * cfg.stride_length_m * (0.5 - progress)
                feet[leg] = (x, foot_y, -cfg.body_height_m)
                contacts[leg] = True

        return GroundWalkingFrame(
            time_s=self._time_s,
            position_m=(self._position_x_m, self._position_y_m, cfg.body_height_m),
            yaw_rad=self._yaw_rad,
            velocity_mps=(vx, vy, 0.0),
            yaw_rate_rad_s=self._yaw_rate_rad_s,
            leg_foot_positions_body_m=feet,
            leg_contact=contacts,
            gait_phase_rad=self._phase_rad,
        )
