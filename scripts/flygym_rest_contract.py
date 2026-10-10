"""Shared, deterministic power-deadband contract for FlyGym control adapters.

The contract is intentionally small: the caller supplies neural-derived
leg-power ``P`` and receives the unchanged raw value, a control value, the
active/static decision, and an event describing a threshold crossing. The
default threshold matches the existing phase-speed controller's engineering
default. It is a software interface value, not a biological threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


CONTRACT_SCHEMA_VERSION = "flygym-rest-contract-1"
DEFAULT_P_DEADBAND = 1e-6


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def validate_p_deadband(p_deadband: Any) -> float:
    """Validate the engineering power threshold shared by decoder and gait."""
    threshold = _finite_nonnegative(p_deadband, "p_deadband")
    if threshold >= 1.0:
        raise ValueError("p_deadband must be less than the maximum normalized P (1)")
    return threshold


def validate_leg_power(leg_power: Any) -> float:
    """Return a finite normalized P in [0, 1]."""
    power = _finite_nonnegative(leg_power, "leg_power")
    if power > 1.0:
        raise ValueError("leg_power must be at most 1")
    return power


def power_is_active(leg_power: Any, p_deadband: Any = DEFAULT_P_DEADBAND) -> bool:
    """Use the shared strict boundary: only ``P > deadband`` is active."""
    power = validate_leg_power(leg_power)
    threshold = validate_p_deadband(p_deadband)
    return power > threshold


@dataclass(frozen=True)
class RestContractTick:
    """One power-gate decision, with raw and controller-facing P separated."""

    schema_version: str
    tick_index: int
    raw_power: float
    control_power: float
    p_deadband: float
    active: bool
    static: bool
    state: str
    transition: str
    previous_active: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FlyGymRestContract:
    """Stateful threshold-crossing recorder shared by neural control adapters.

    Static is defined as ``P <= p_deadband``. That boundary matches a
    controller which activates only for ``abs(signal) > active_threshold``.
    On static ticks, ``control_power`` is exactly zero while ``raw_power`` is
    retained for diagnosis. The class does not hold neural or gait state.
    """

    def __init__(self, p_deadband: float = DEFAULT_P_DEADBAND):
        self.p_deadband = validate_p_deadband(p_deadband)
        self.reset()

    def reset(self) -> None:
        """Start a new sequence; the first subsequent tick has no prior state."""
        self._tick_index = 0
        self._previous_active: bool | None = None

    @property
    def tick_index(self) -> int:
        return self._tick_index

    @property
    def previous_active(self) -> bool | None:
        return self._previous_active

    def step(self, leg_power: float) -> RestContractTick:
        power = validate_leg_power(leg_power)
        active = power > self.p_deadband
        previous = self._previous_active
        if previous is None:
            transition = "active" if active else "static"
        elif not previous and active:
            transition = "inactive_to_active"
        elif previous and not active:
            transition = "active_to_static"
        elif active:
            transition = "remain_active"
        else:
            transition = "remain_inactive"

        tick = RestContractTick(
            schema_version=CONTRACT_SCHEMA_VERSION,
            tick_index=self._tick_index,
            raw_power=power,
            control_power=power if active else 0.0,
            p_deadband=self.p_deadband,
            active=active,
            static=not active,
            state="active" if active else "static",
            transition=transition,
            previous_active=previous,
        )
        self._tick_index += 1
        self._previous_active = active
        return tick


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "DEFAULT_P_DEADBAND",
    "FlyGymRestContract",
    "RestContractTick",
    "power_is_active",
    "validate_leg_power",
    "validate_p_deadband",
]
