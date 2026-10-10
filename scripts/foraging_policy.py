#!/usr/bin/env python3
"""Explicitly engineered odor-guided motor baseline for the food pilot.

This policy is deliberately separate from MaleCNS. Its inputs are the mean
left and right food-odor receptor rates, one physical-support contact boolean,
a body-frame airflow proxy, and a hunger scalar. It does not accept world pose,
coordinates, target data, visual labels, or scene objects. Its stages and
thresholds are hand-built engineering choices, not a fitted or biologically
validated fly controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from typing import Any, Dict

try:
    from .flight_dynamics import NeuromuscularInput
except ImportError:  # direct script import
    from flight_dynamics import NeuromuscularInput


POLICY_NAME = "engineered_bilateral_odor_foraging_baseline"
POLICY_SCOPE = (
    "hand-built odor/wind heuristic; not MaleCNS, not a learned controller, "
    "and not biologically validated"
)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _finite_wind_vector(value: Any) -> tuple[float, float]:
    """Validate a body-frame (forward, left) ambient-wind pair."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("wind_body_mps must be a finite two-component vector")
    components: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError("wind_body_mps must be a finite two-component vector")
        try:
            number = float(component)
        except (ValueError, OverflowError) as exc:
            raise ValueError("wind_body_mps must be a finite two-component vector") from exc
        if not math.isfinite(number):
            raise ValueError("wind_body_mps must be a finite two-component vector")
        components.append(number)
    return components[0], components[1]


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class ForagingPolicyConfig:
    """Bounded command gains and odor thresholds for the reduced-body pilot."""

    dt_s: float = 0.005
    takeoff_s: float = 0.100
    odor_present_hz: float = 8.0
    brake_odor_hz: float = 54.0
    land_odor_hz: float = 58.0
    hover_power: float = 0.69
    wind_hover_power: float = 0.706
    takeoff_power: float = 0.86
    low_odor_power: float = 0.69
    land_power: float = 0.58
    approach_bias_rad: float = 0.040
    brake_bias_rad: float = -0.035
    max_yaw_torque_Nm: float = 4.0e-10
    cast_yaw_torque_Nm: float = 2.0e-10
    binaural_scale_hz: float = 0.60
    yaw_deadband_hz: float = 0.015
    temporal_window_s: float = 0.100
    temporal_drop_hz: float = 0.08
    wind_landing_delay_s: float = 1.20
    wind_landing_alignment_rad: float = 0.40
    wind_landing_brake_lead_s: float = 0.35
    cast_pulse_s: float = 0.20
    cast_refractory_s: float = 0.85

    def __post_init__(self) -> None:
        for name in ("dt_s", "takeoff_s", "odor_present_hz", "brake_odor_hz",
                     "land_odor_hz", "hover_power", "wind_hover_power", "takeoff_power",
                     "low_odor_power", "land_power", "approach_bias_rad",
                     "brake_bias_rad", "max_yaw_torque_Nm", "cast_yaw_torque_Nm",
                     "binaural_scale_hz", "yaw_deadband_hz", "temporal_window_s",
                     "temporal_drop_hz", "wind_landing_delay_s",
                     "wind_landing_alignment_rad", "wind_landing_brake_lead_s",
                     "cast_pulse_s", "cast_refractory_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.dt_s != 0.005:
            raise ValueError("the foraging baseline is fixed to the existing 5 ms dynamics step")
        if self.takeoff_s <= 0.0:
            raise ValueError("takeoff_s must be > 0")
        if not (0.0 <= self.odor_present_hz < self.brake_odor_hz < self.land_odor_hz):
            raise ValueError("odor thresholds must satisfy present < brake < land")
        for name in ("hover_power", "wind_hover_power", "takeoff_power", "low_odor_power", "land_power"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if abs(self.approach_bias_rad) > 0.25 or abs(self.brake_bias_rad) > 0.25:
            raise ValueError("stroke-bias commands must remain within +/- 0.25 rad")
        if (self.max_yaw_torque_Nm <= 0.0 or self.cast_yaw_torque_Nm <= 0.0
                or self.binaural_scale_hz <= 0.0):
            raise ValueError("yaw gain and binaural scale must be positive")
        if max(self.max_yaw_torque_Nm, self.cast_yaw_torque_Nm) > 2.0e-9:
            raise ValueError("engineered yaw torque must remain within the configured 2e-9 N*m scale")
        if min(self.yaw_deadband_hz, self.temporal_drop_hz) < 0.0:
            raise ValueError("yaw deadband and temporal drop must be nonnegative")
        if min(self.temporal_window_s, self.cast_pulse_s, self.cast_refractory_s) <= 0.0:
            raise ValueError("temporal and cast durations must be positive")
        if (min(self.wind_landing_delay_s, self.wind_landing_brake_lead_s) < 0.0
                or not 0.0 <= self.wind_landing_alignment_rad <= math.pi):
            raise ValueError("wind landing delay/alignment bounds are invalid")


@dataclass(frozen=True)
class ForagingDecision:
    """Motor command plus policy-only telemetry for one 5 ms tick."""

    command: NeuromuscularInput
    phase: str
    left_odor_hz: float
    right_odor_hz: float
    mean_odor_hz: float
    binaural_difference_hz: float
    yaw_drive: float
    wind_body_mps: tuple[float, float]
    upwind_bearing_rad: float | None
    odor_gated_upwind_drive: float
    hunger: float
    odor_surge: float
    wind_guided_descent: bool
    crosswind_cast_drive: float
    physical_support_contact: bool

    def telemetry(self) -> Dict[str, Any]:
        return {
            "policy": POLICY_NAME,
            "policy_scope": POLICY_SCOPE,
            "phase": self.phase,
            "observations": {
                "left_odor_rate_hz": self.left_odor_hz,
                "right_odor_rate_hz": self.right_odor_hz,
                "ambient_wind_body_mps": list(self.wind_body_mps),
                "hunger": self.hunger,
                "physical_support_contact": self.physical_support_contact,
            },
            "derived": {
                "mean_odor_rate_hz": self.mean_odor_hz,
                "left_minus_right_hz": self.binaural_difference_hz,
                "yaw_drive": self.yaw_drive,
                "upwind_bearing_rad": self.upwind_bearing_rad,
                "odor_gated_upwind_drive": self.odor_gated_upwind_drive,
                "odor_surge": self.odor_surge,
                "wind_guided_descent": self.wind_guided_descent,
                "crosswind_cast_drive": self.crosswind_cast_drive,
            },
            "motor_command": self.command.to_dict(),
        }


class ForagingPolicy:
    """Rate-only, deterministic motor policy for an engineered baseline.

    A short initial takeoff pulse is shared across odor-on and odor-off runs.
    During flight, odor rate gates forward stroke bias and an ambient-wind
    proxy for upwind yaw alignment, rising odor invokes a braking bias, and
    very strong odor permits a low-power descent in the zero-wind condition;
    with wind, a timed aligned-odor corridor controls descent and braking. A
    bilateral rate difference turns toward the stronger antenna. Contact after
    descent has begun yields a low-power settling command. Wind alignment,
    odor surge, and crosswind cast are bounded engineering heuristics, not
    MaleCNS and not biologically validated.

    The policy receives no world pose, coordinates, target data, scene/mesh
    data, or CNS output. The motion mapping is a hand-authored
    reduced-model convenience, not an account of fly foraging biology.
    """

    def __init__(self, config: ForagingPolicyConfig | None = None) -> None:
        self.config = config or ForagingPolicyConfig()
        self.elapsed_s = 0.0
        self._started = False
        self._previous_contact = False
        self._landing_armed = False
        self._launch_remaining_s = self.config.takeoff_s
        self._odor_history: deque[tuple[float, float]] = deque()
        self._cast_remaining_s = 0.0
        self._cast_refractory_remaining_s = 0.0
        self._cast_drive = 1.0
        self._crosswind_side = 1.0
        self._previous_odor_present = False
        self._aligned_odor_exposure_s = 0.0

    def step(
        self,
        left_odor_hz: Any,
        right_odor_hz: Any,
        physical_support_contact: bool,
        *,
        wind_body_mps: tuple[float, float] = (0.0, 0.0),
        hunger: float = 1.0,
    ) -> ForagingDecision:
        """Convert rates, contact, wind proxy, and hunger to motor commands.

        ``wind_body_mps`` is configured ambient wind in body forward/left
        components; this simplified direction proxy ignores self-motion and
        is not a measured antennal signal.
        No world pose, position, or target information is accepted.
        """

        left = _finite_nonnegative(left_odor_hz, "left_odor_hz")
        right = _finite_nonnegative(right_odor_hz, "right_odor_hz")
        wind = _finite_wind_vector(wind_body_mps)
        hunger_value = _unit_interval(hunger, "hunger")
        if type(physical_support_contact) is not bool:
            raise ValueError("physical_support_contact must be a bool")
        cfg = self.config
        mean = 0.5 * (left + right)
        difference = left - right
        yaw_drive = 0.0
        if abs(difference) > cfg.yaw_deadband_hz:
            yaw_drive = max(-1.0, min(1.0, difference / cfg.binaural_scale_hz))

        odor_present = mean >= cfg.odor_present_hz
        wind_forward, wind_left = wind
        upwind_bearing: float | None = None
        upwind_drive = 0.0
        if math.hypot(wind_forward, wind_left) > 1.0e-9:
            # Ambient wind points downwind; its negation points upwind.
            upwind_bearing = math.atan2(-wind_left, -wind_forward)
            if odor_present:
                # Hunger has a deliberately modest effect on odor response.
                odor_responsiveness = 0.75 + 0.25 * hunger_value
                upwind_drive = (upwind_bearing / math.pi) * odor_responsiveness

        wind_heading_aligned = bool(
            upwind_bearing is not None
            and abs(upwind_bearing) <= cfg.wind_landing_alignment_rad
        )
        if odor_present and wind_heading_aligned:
            self._aligned_odor_exposure_s += cfg.dt_s

        # Small odor-on surge in the approach band, scaled modestly by hunger.
        odor_surge = 0.025 * (0.75 + 0.25 * hunger_value) if odor_present else 0.0
        wind_guided_descent = bool(
            odor_present
            and wind_heading_aligned
            and self._aligned_odor_exposure_s >= cfg.wind_landing_delay_s
        )

        # A short turn pulse breaks exact bilateral symmetry when odor has
        # fallen over a short history window. It is followed by bilateral
        # correction; no heading or food direction is read.
        self._odor_history.append((self.elapsed_s, mean))
        while (len(self._odor_history) > 1
               and self.elapsed_s - self._odor_history[0][0] > cfg.temporal_window_s):
            self._odor_history.popleft()
        self._cast_refractory_remaining_s = max(
            0.0, self._cast_refractory_remaining_s - cfg.dt_s
        )
        # Losing a previously present plume starts a brief crosswind cast.
        # Alternate sides across losses for a deterministic bounded pattern.
        if (self._previous_odor_present and not odor_present
                and self._cast_refractory_remaining_s <= 0.0):
            self._crosswind_side *= -1.0
            crosswind_bearing = upwind_bearing if upwind_bearing is not None else 0.0
            crosswind_bearing = math.atan2(
                math.sin(crosswind_bearing + self._crosswind_side * math.pi / 2.0),
                math.cos(crosswind_bearing + self._crosswind_side * math.pi / 2.0),
            )
            self._cast_drive = max(-1.0, min(1.0, crosswind_bearing / math.pi))
            if abs(self._cast_drive) < 0.15:
                self._cast_drive = self._crosswind_side * 0.15
            self._cast_remaining_s = cfg.cast_pulse_s
            self._cast_refractory_remaining_s = cfg.cast_refractory_s
        if (odor_present and len(self._odor_history) >= 2
                and self.elapsed_s - self._odor_history[0][0] >= cfg.temporal_window_s * 0.8
                and mean <= self._odor_history[0][1] - cfg.temporal_drop_hz
                and self._cast_refractory_remaining_s <= 0.0):
            self._cast_remaining_s = cfg.cast_pulse_s
            self._cast_refractory_remaining_s = cfg.cast_refractory_s
            self._cast_drive = 1.0

        landing_triggered = bool(
            wind_guided_descent
            or (upwind_bearing is None and mean >= cfg.land_odor_hz)
        )
        if landing_triggered:
            self._landing_armed = True

        # The command starts with the same brief launch pulse in both matched
        # conditions.  Once the body has left support, the state is switched
        # exclusively by this policy's elapsed time, rates, and contact bit.
        if not self._started:
            self._started = True
            phase = "takeoff"
            power = cfg.takeoff_power
            bias = cfg.approach_bias_rad if odor_present and (upwind_bearing is None or wind_heading_aligned) else 0.0
            self._launch_remaining_s = max(0.0, round(self._launch_remaining_s - cfg.dt_s, 12))
        elif self._launch_remaining_s > 1.0e-12:
            phase = "takeoff"
            power = cfg.takeoff_power
            bias = cfg.approach_bias_rad if odor_present and (upwind_bearing is None or wind_heading_aligned) else 0.0
            self._launch_remaining_s = max(0.0, round(self._launch_remaining_s - cfg.dt_s, 12))
        elif physical_support_contact and self._landing_armed:
            if odor_present:
                phase = "landing_settle"
                power = cfg.land_power
                bias = 0.0
            else:
                phase = "grounded_no_odor"
                power = 0.0
                bias = 0.0
        elif wind_guided_descent or (upwind_bearing is None and mean >= cfg.land_odor_hz):
            phase = "odor_triggered_descent"
            power = cfg.land_power
            bias = 0.0
        elif (odor_present and wind_heading_aligned
              and self._aligned_odor_exposure_s >= max(
                  0.0, cfg.wind_landing_delay_s - cfg.wind_landing_brake_lead_s
              )):
            phase = "odor_guided_brake"
            power = cfg.hover_power
            bias = cfg.brake_bias_rad
        elif odor_present and upwind_bearing is not None and not wind_heading_aligned:
            # Turn toward the upwind airflow before adding forward drive; this
            # prevents a wrong initial heading from carrying the body away.
            phase = "odor_upwind_reorientation"
            power = cfg.wind_hover_power
            bias = 0.0
        elif odor_present and upwind_bearing is not None:
            # In a transported plume, absolute rate also varies with puff age.
            # Use the aligned odor corridor for approach instead of treating a
            # single concentration peak as a nearby landing cue.
            phase = "odor_guided_approach"
            power = cfg.hover_power
            bias = cfg.approach_bias_rad
        elif mean >= cfg.brake_odor_hz:
            phase = "odor_guided_brake"
            power = cfg.hover_power
            bias = cfg.brake_bias_rad
        elif mean >= cfg.odor_present_hz:
            phase = "odor_guided_approach"
            power = cfg.hover_power
            bias = cfg.approach_bias_rad
        else:
            phase = "search_no_odor"
            # Hunger zero reduces search effort while retaining a bounded
            # positive search command during free flight.
            power = cfg.low_odor_power * (0.65 + 0.35 * hunger_value)
            bias = 0.0

        if odor_present:
            bias *= 0.75 + 0.25 * hunger_value
            if phase == "odor_guided_approach":
                # A small forward stroke-bias increase is the surge cue; keep
                # wing power fixed so the odor response does not add lift.
                bias = min(0.25, bias + odor_surge)

        # Stronger odor on the left gives positive yaw torque, matching the
        # body-frame convention where positive yaw turns toward +Y/left.
        if self._cast_remaining_s > 0.0 and abs(difference) <= cfg.yaw_deadband_hz * 3.0:
            yaw_torque = cfg.cast_yaw_torque_Nm * self._cast_drive
            self._cast_remaining_s = max(0.0, self._cast_remaining_s - cfg.dt_s)
        else:
            combined_yaw_drive = max(-1.0, min(1.0, yaw_drive + upwind_drive))
            yaw_torque = cfg.max_yaw_torque_Nm * combined_yaw_drive
        command = NeuromuscularInput(
            left_power=power,
            right_power=power,
            left_bias_rad=bias,
            right_bias_rad=bias,
            yaw_torque_Nm=yaw_torque,
            source="safety-handbuilt",
        ).normalized()
        self.elapsed_s += cfg.dt_s
        self._previous_contact = physical_support_contact
        self._previous_odor_present = odor_present
        return ForagingDecision(
            command=command,
            phase=phase,
            left_odor_hz=left,
            right_odor_hz=right,
            mean_odor_hz=mean,
            binaural_difference_hz=difference,
            yaw_drive=yaw_drive,
            wind_body_mps=wind,
            upwind_bearing_rad=upwind_bearing,
            odor_gated_upwind_drive=upwind_drive,
            hunger=hunger_value,
            odor_surge=odor_surge if phase == "odor_guided_approach" else 0.0,
            wind_guided_descent=wind_guided_descent,
            crosswind_cast_drive=self._cast_drive if self._cast_remaining_s > 0.0 else 0.0,
            physical_support_contact=physical_support_contact,
        )


__all__ = [
    "ForagingDecision", "ForagingPolicy", "ForagingPolicyConfig",
    "POLICY_NAME", "POLICY_SCOPE",
]
