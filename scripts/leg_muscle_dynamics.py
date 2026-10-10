#!/usr/bin/env python3
"""Small CPU Hill-type muscle model for leg actuators.

All physical quantities use SI units: metres, seconds, newtons, and watts.
Muscle shortening velocity is negative; lengthening velocity is positive.
The model has a first-order activation state, a Gaussian active length-tension
curve, a bounded Hill-style force-velocity curve, and a passive elastic
tension curve.  It produces tensile force only: it cannot push when a muscle
is slack.  There is no imposed gait or default excitation.

The defaults are engineering hypotheses intended to make the interface usable
in a digital mechanism.  They are not calibrated fly physiology.  This
reduced model omits tendon series elasticity, pennation, and history-dependent
effects such as force-length hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def _finite_scalar(value: float, name: str) -> float:
    """Return a finite float, rejecting bools and non-scalar inputs."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class MuscleParameters:
    """Hill muscle parameters, expressed in SI units where dimensional.

    ``passive_slack_length_m=None`` means that passive tension starts at the
    optimal fibre length.  Passive tension rises quadratically with strain and
    is explicitly capped by ``passive_force_cap_multiplier * F_iso``.  The cap
    is a numerical/engineering bound, not a claim about muscle physiology.
    """

    max_isometric_force_N: float = 1.0e-5
    optimal_length_m: float = 0.0006
    max_shortening_velocity_mps: float = 0.0018
    activation_time_constant_s: float = 0.010
    deactivation_time_constant_s: float = 0.040
    length_tension_width: float = 0.45
    force_velocity_shape: float = 0.25
    eccentric_force_multiplier: float = 1.5
    eccentric_velocity_scale: float = 3.0
    passive_slack_length_m: Optional[float] = None
    passive_strain_scale: float = 0.5
    passive_force_multiplier: float = 1.0
    passive_force_cap_multiplier: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            "max_isometric_force_N", "optimal_length_m",
            "max_shortening_velocity_mps", "activation_time_constant_s",
            "deactivation_time_constant_s", "length_tension_width",
            "force_velocity_shape", "eccentric_velocity_scale",
            "passive_strain_scale", "passive_force_cap_multiplier",
        )
        for name in positive:
            value = _finite_scalar(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

        eccentric_cap = _finite_scalar(
            self.eccentric_force_multiplier, "eccentric_force_multiplier"
        )
        if eccentric_cap < 1.0:
            raise ValueError("eccentric_force_multiplier must be at least 1")
        object.__setattr__(self, "eccentric_force_multiplier", eccentric_cap)

        passive_multiplier = _finite_scalar(
            self.passive_force_multiplier, "passive_force_multiplier"
        )
        if passive_multiplier < 0.0:
            raise ValueError("passive_force_multiplier must be nonnegative")
        object.__setattr__(self, "passive_force_multiplier", passive_multiplier)

        slack_length = self.passive_slack_length_m
        if slack_length is None:
            slack_length = self.optimal_length_m
        else:
            slack_length = _finite_scalar(slack_length, "passive_slack_length_m")
            if slack_length <= 0.0:
                raise ValueError("passive_slack_length_m must be positive")
        object.__setattr__(self, "passive_slack_length_m", slack_length)


@dataclass(frozen=True)
class MuscleState:
    """One muscle snapshot.  ``power_W`` is positive during shortening."""

    time_s: float
    excitation: float
    activation: float
    length_m: float
    velocity_mps: float
    active_force_N: float
    passive_force_N: float
    total_force_N: float
    power_W: float


class HillMuscle:
    """Stateful, deterministic Hill-type muscle actuator.

    Activation follows the continuous-time first-order equation

    ``da/dt = (excitation - activation) / tau``

    with ``tau=activation_time_constant_s`` while activating and
    ``tau=deactivation_time_constant_s`` while deactivating.  ``step`` applies
    its exact exponential update over ``dt_s`` (seconds), rather than an
    Euler approximation.  Length and velocity are sampled at the end of that
    tick.  Excitation must already be normalized to [0, 1]; invalid values
    raise instead of being silently clipped.

    The returned total force is nonnegative and tensile.  ``power_W`` uses
    ``-force * velocity`` so muscle shortening (negative velocity) is positive
    mechanical output, while lengthening is negative.
    """

    def __init__(self, params: Optional[MuscleParameters] = None) -> None:
        self.params = params if params is not None else MuscleParameters()
        if not isinstance(self.params, MuscleParameters):
            raise ValueError("params must be a MuscleParameters instance")
        self._time_s = 0.0
        self._state = self._make_state(
            time_s=0.0,
            excitation=0.0,
            activation=0.0,
            length_m=self.params.optimal_length_m,
            velocity_mps=0.0,
        )

    @property
    def state(self) -> MuscleState:
        """Most recently computed immutable muscle snapshot."""

        return self._state

    def reset(
        self,
        activation: float = 0.0,
        length_m: Optional[float] = None,
        velocity_mps: float = 0.0,
    ) -> MuscleState:
        """Reset activation and elapsed time; default pose is optimal length."""

        activation_value = _finite_scalar(activation, "activation")
        if not 0.0 <= activation_value <= 1.0:
            raise ValueError("activation must be normalized to [0, 1]")
        length_value = self.params.optimal_length_m if length_m is None else _finite_scalar(
            length_m, "length_m"
        )
        velocity_value = _finite_scalar(velocity_mps, "velocity_mps")
        if length_value <= 0.0:
            raise ValueError("length_m must be positive")

        state = self._make_state(
            time_s=0.0,
            excitation=activation_value,
            activation=activation_value,
            length_m=length_value,
            velocity_mps=velocity_value,
        )
        self._time_s = 0.0
        self._state = state
        return state

    def step(
        self,
        excitation: float,
        length_m: float,
        velocity_mps: float,
        dt_s: float,
    ) -> MuscleState:
        """Advance activation and evaluate force for one SI-second tick.

        ``velocity_mps < 0`` denotes shortening; ``velocity_mps > 0`` denotes
        lengthening.  Zero ``dt_s`` is allowed and evaluates force at the
        supplied pose without advancing activation or simulation time.
        """

        excitation_value = _finite_scalar(excitation, "excitation")
        length_value = _finite_scalar(length_m, "length_m")
        velocity_value = _finite_scalar(velocity_mps, "velocity_mps")
        dt_value = _finite_scalar(dt_s, "dt_s")
        if not 0.0 <= excitation_value <= 1.0:
            raise ValueError("excitation must be normalized to [0, 1]")
        if length_value <= 0.0:
            raise ValueError("length_m must be positive")
        if dt_value < 0.0:
            raise ValueError("dt_s must be nonnegative")

        new_time = self._time_s + dt_value
        if not math.isfinite(new_time):
            raise ValueError("simulation time overflow")

        old_activation = self._state.activation
        if excitation_value > old_activation:
            tau_s = self.params.activation_time_constant_s
        else:
            tau_s = self.params.deactivation_time_constant_s
        # The exact solution of da/dt=(u-a)/tau is stable for long ticks and
        # makes a fixed excitation independent of how a caller subdivides dt.
        activation = excitation_value + (old_activation - excitation_value) * math.exp(
            -dt_value / tau_s
        )
        # Roundoff at extreme time constants must not escape the normalized
        # activation invariant.
        activation = min(1.0, max(0.0, activation))

        state = self._make_state(
            time_s=new_time,
            excitation=excitation_value,
            activation=activation,
            length_m=length_value,
            velocity_mps=velocity_value,
        )
        self._time_s = new_time
        self._state = state
        return state

    def _make_state(
        self,
        *,
        time_s: float,
        excitation: float,
        activation: float,
        length_m: float,
        velocity_mps: float,
    ) -> MuscleState:
        params = self.params
        length_ratio = length_m / params.optimal_length_m
        if not math.isfinite(length_ratio):
            raise ValueError("length_m is outside the model's numeric range")

        # A normalized Gaussian gives a smooth, nonnegative force-length
        # relation with its unique maximum at optimal fibre length.
        length_offset = (length_ratio - 1.0) / params.length_tension_width
        active_length_factor = math.exp(-0.5 * length_offset * length_offset)
        velocity_factor = self._force_velocity_factor(velocity_mps)
        active_force = (
            params.max_isometric_force_N
            * activation
            * active_length_factor
            * velocity_factor
        )

        slack_length = params.passive_slack_length_m
        assert slack_length is not None
        strain_ratio = length_m / slack_length
        if not math.isfinite(strain_ratio):
            raise ValueError("length_m is outside the passive model's numeric range")
        strain = max(0.0, strain_ratio - 1.0)
        passive_force_multiplier = params.passive_force_multiplier
        if strain <= 0.0 or passive_force_multiplier == 0.0:
            passive_force = 0.0
        else:
            cap = params.passive_force_cap_multiplier
            # Compare before squaring so even extreme finite lengths fail safe
            # without overflowing an intermediate value.
            cap_strain = params.passive_strain_scale * math.sqrt(
                cap / passive_force_multiplier
            )
            if strain >= cap_strain:
                passive_factor = cap
            else:
                scaled_strain = strain / params.passive_strain_scale
                passive_factor = passive_force_multiplier * scaled_strain * scaled_strain
            passive_force = params.max_isometric_force_N * passive_factor

        total_force = active_force + passive_force
        power = -total_force * velocity_mps
        values = (active_force, passive_force, total_force, power)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("computed muscle state is outside the model's numeric range")
        if active_force < 0.0 or passive_force < 0.0 or total_force < 0.0:
            raise ValueError("computed muscle tension must be nonnegative")

        return MuscleState(
            time_s=time_s,
            excitation=excitation,
            activation=activation,
            length_m=length_m,
            velocity_mps=velocity_mps,
            active_force_N=active_force,
            passive_force_N=passive_force,
            total_force_N=total_force,
            power_W=power,
        )

    def _force_velocity_factor(self, velocity_mps: float) -> float:
        params = self.params
        max_shortening = params.max_shortening_velocity_mps
        if velocity_mps < 0.0:
            shortening_speed = -velocity_mps
            # Force reaches zero at and above maximum shortening speed.  The
            # comparison also avoids overflow for extreme finite velocities.
            if shortening_speed >= max_shortening:
                return 0.0
            shortening_fraction = shortening_speed / max_shortening
            factor = (1.0 - shortening_fraction) / (
                1.0 + shortening_fraction / params.force_velocity_shape
            )
            return max(0.0, factor)
        if velocity_mps == 0.0 or params.eccentric_force_multiplier == 1.0:
            return 1.0

        # The eccentric branch approaches a finite cap.  The two forms avoid
        # overflow in either velocity ratio for extreme but finite inputs.
        if velocity_mps >= max_shortening:
            cap_ratio = params.eccentric_velocity_scale * (
                max_shortening / velocity_mps
            )
            fraction = 1.0 / (1.0 + cap_ratio)
        else:
            velocity_ratio = velocity_mps / max_shortening
            scaled_ratio = velocity_ratio / params.eccentric_velocity_scale
            if scaled_ratio >= 1.0:
                fraction = 1.0 / (1.0 + 1.0 / scaled_ratio)
            else:
                fraction = scaled_ratio / (1.0 + scaled_ratio)
        factor = 1.0 + (params.eccentric_force_multiplier - 1.0) * fraction
        return min(params.eccentric_force_multiplier, max(1.0, factor))


__all__ = ["HillMuscle", "MuscleParameters", "MuscleState"]
