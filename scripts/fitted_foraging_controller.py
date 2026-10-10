"""Small sensor-driven flight controller with fit-ready parameters.

This is an engineered upwind-corridor controller, not a neural model. Height,
body velocity and yaw rate are idealized proprioceptive observations supplied
by the simulator. A fitted travel duration encodes the training corridor's
scale; it is not a visual range estimate or a general food-localization model.
No food coordinate, target displacement or body XY position enters the policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from flight_dynamics import NeuromuscularInput


FEEDBACK_PARAMETER_BOUNDS = {
    "cruise_height_m": (0.09, 0.18),
    "hover_power": (0.69, 0.72),
    "altitude_kp": (8.0, 40.0),
    "vertical_kd": (3.0, 12.0),
    "cruise_speed_mps": (0.08, 0.24),
    "forward_velocity_gain": (2.0, 8.0),
    "transit_time_s": (0.9, 2.7),
    "yaw_gain_Nm": (4.0e-10, 2.0e-9),
    "descent_speed_mps": (0.02, 0.09),
    "brake_time_s": (0.1, 0.6),
}
FEEDBACK_SCOPE = (
    "engineered flight feedback with fitted corridor parameters; idealized "
    "proprioception and ambient wind; no MaleCNS; not biologically validated"
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class FeedbackForagingConfig:
    cruise_height_m: float = 0.12
    hover_power: float = 0.706353168
    altitude_kp: float = 25.0
    vertical_kd: float = 8.0
    cruise_speed_mps: float = 0.16
    forward_velocity_gain: float = 5.0
    transit_time_s: float = 1.5
    yaw_gain_Nm: float = 1.4e-9
    descent_speed_mps: float = 0.04
    brake_time_s: float = 0.3

    def __post_init__(self) -> None:
        for name, (low, high) in FEEDBACK_PARAMETER_BOUNDS.items():
            value = _finite(getattr(self, name), name)
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")


@dataclass(frozen=True)
class FeedbackDecision:
    command: NeuromuscularInput
    phase: str
    observations: dict[str, Any]
    derived: dict[str, Any]

    def telemetry(self) -> dict[str, Any]:
        return {
            "policy": "feedback_foraging_corridor",
            "policy_scope": FEEDBACK_SCOPE,
            "phase": self.phase,
            "observations": self.observations,
            "derived": self.derived,
            "motor_command": self.command.to_dict(),
        }


class FeedbackForagingController:
    """Altitude/velocity feedback plus odor-gated travel, brake and descent.

    Constants below are explicit engineering choices: 5 ms step, 8 Hz odor
    gate, 0.04 rad alignment gate, a gravity scale of 9.80665 m/s², 0.25 rad
    stroke bound, and 2e-9 N*m yaw bound. The forward integral compensates
    steady wind drift; it has no access to desired food position.
    """

    def __init__(self, config: FeedbackForagingConfig | None = None) -> None:
        self.config = config or FeedbackForagingConfig()
        self.travel_s = 0.0
        self.brake_s = 0.0
        self.descending = False
        self.landed = False
        self.has_flown = False
        self.forward_integral = 0.0
        self.elapsed_s = 0.0

    def step(
        self, left_odor_hz: float, right_odor_hz: float,
        physical_support_contact: bool, *,
        wind_body_mps: tuple[float, float], hunger: float,
        height_m: float, velocity_body_mps: tuple[float, float, float],
        yaw_rate_rad_s: float,
    ) -> FeedbackDecision:
        cfg = self.config
        dt = 0.005
        left = _finite(left_odor_hz, "left_odor_hz")
        right = _finite(right_odor_hz, "right_odor_hz")
        hunger = _finite(hunger, "hunger")
        height = _finite(height_m, "height_m")
        yaw_rate = _finite(yaw_rate_rad_s, "yaw_rate_rad_s")
        if min(left, right) < 0.0 or not 0.0 <= hunger <= 1.0:
            raise ValueError("odor must be nonnegative and hunger in [0, 1]")
        if type(physical_support_contact) is not bool:
            raise ValueError("physical_support_contact must be bool")
        if len(wind_body_mps) != 2 or len(velocity_body_mps) != 3:
            raise ValueError("wind must have two and body velocity three components")
        wind = tuple(_finite(v, "wind_body_mps") for v in wind_body_mps)
        velocity = tuple(_finite(v, "velocity_body_mps") for v in velocity_body_mps)
        vx, vy, vz = velocity
        odor = (left + right) * 0.5 >= 8.0
        wind_present = math.hypot(*wind) > 1.0e-6
        bearing = math.atan2(-wind[1], -wind[0]) if wind_present else None
        if bearing is None:
            # An odor-gradient/search fallback is exposed but the current fit
            # is a windy corridor task, not validated zero-wind navigation.
            steering = math.tanh((left - right) / 0.1) if odor else 0.0
        else:
            steering = bearing if odor else 0.0
        yaw_torque = _clip(cfg.yaw_gain_Nm * steering - 3e-11 * yaw_rate,
                           -2e-9, 2e-9)
        aligned = odor and (bearing is None or abs(bearing) < 0.04)
        ready = aligned and height > cfg.cruise_height_m * 0.85
        self.has_flown |= not physical_support_contact and height > 0.02
        if self.descending and physical_support_contact and self.has_flown:
            self.landed = True

        forward_target = 0.0
        if self.landed:
            phase = "settle_on_support"
        elif self.descending:
            phase = "controlled_descent"
        elif self.travel_s >= cfg.transit_time_s:
            phase = "velocity_brake"
            self.brake_s += dt
            if self.brake_s >= cfg.brake_time_s:
                self.descending = True
        elif ready and hunger > 0.05:
            phase = "odor_guided_transit"
            forward_target = cfg.cruise_speed_mps
            self.travel_s += dt
        else:
            phase = "climb_and_align" if odor else "hover_without_odor"

        # Gate initial takeoff by the engineering hunger variable. It does
        # not represent a biological motivational circuit.
        resting = hunger <= 0.05 and physical_support_contact and not self.has_flown
        self.forward_integral = _clip(
            self.forward_integral + (forward_target - vx) * dt, -0.1, 0.1,
        )
        ax = cfg.forward_velocity_gain * (forward_target - vx) + 2.0 * self.forward_integral
        if self.landed:
            power = 0.60
            az = 0.0
        else:
            az = (cfg.vertical_kd * (-cfg.descent_speed_mps - vz)
                  if self.descending else
                  cfg.altitude_kp * (cfg.cruise_height_m - height) - cfg.vertical_kd * vz)
            power = _clip(cfg.hover_power * math.sqrt(max(0.05, 1.0 + az / 9.80665)),
                          0.3, 0.92)
        # In the reduced body a common stroke bias redirects average lift
        # into forward force. This controls an actuator, never body position.
        effective_lift_accel = 9.80665 * (power / cfg.hover_power) ** 2
        bias = _clip(math.atan2(ax, max(1.0, effective_lift_accel)), -0.25, 0.25)
        if resting:
            phase, power, bias, yaw_torque = "rest_sated", 0.0, 0.0, 0.0
        command = NeuromuscularInput(
            left_power=power, right_power=power,
            left_bias_rad=bias, right_bias_rad=bias,
            yaw_torque_Nm=yaw_torque, source="safety-handbuilt",
        ).normalized()
        self.elapsed_s += dt
        return FeedbackDecision(command, phase, {
            "left_odor_rate_hz": left, "right_odor_rate_hz": right,
            "ambient_wind_body_mps": list(wind), "hunger": hunger,
            "physical_support_contact": physical_support_contact,
            "height_m": height, "velocity_body_mps": list(velocity),
            "yaw_rate_rad_s": yaw_rate,
        }, {
            "travel_s": self.travel_s, "brake_s": self.brake_s,
            "upwind_bearing_rad": bearing, "forward_target_mps": forward_target,
            "vertical_acceleration_command_mps2": az,
            "forward_integral": self.forward_integral,
            "parameters": asdict(cfg),
        })
