#!/usr/bin/env python3
"""Small deterministic hunger/satiety state for the foraging pilot.

This scalar model is an engineering approximation. It is not a physiological
model and is not biologically validated.
"""
from __future__ import annotations

import math
from typing import Any


FORAGING_STATE_STATUS = "engineering_approximation"
FORAGING_STATE_SCOPE = (
    "deterministic scalar hunger/satiety bookkeeping; not a physiological "
    "model and not biologically validated"
)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


class ForagingState:
    """Track normalized hunger in ``[0, 1]`` with contact-gated satiety.

    Hunger increases at ``basal_hunger_per_s``. During edible contact only,
    ``satiety_per_s`` is subtracted from that rate. The state is clamped at
    both ends. All rates and durations are engineering parameters.
    """

    def __init__(
        self,
        initial_hunger: float = 1.0,
        basal_hunger_per_s: float = 0.0005,
        satiety_per_s: float = 0.25,
    ) -> None:
        hunger = _finite_number(initial_hunger, "initial_hunger")
        basal = _finite_number(basal_hunger_per_s, "basal_hunger_per_s")
        satiety = _finite_number(satiety_per_s, "satiety_per_s")
        if not 0.0 <= hunger <= 1.0:
            raise ValueError("initial_hunger must be in [0, 1]")
        if basal < 0.0:
            raise ValueError("basal_hunger_per_s must be nonnegative")
        if satiety < 0.0:
            raise ValueError("satiety_per_s must be nonnegative")

        self._hunger = hunger
        self.basal_hunger_per_s = basal
        self.satiety_per_s = satiety
        self.cumulative_edible_contact_s = 0.0

    @property
    def hunger(self) -> float:
        """Current normalized hunger, bounded to ``[0, 1]``."""

        return self._hunger

    def advance(self, dt_s: float, edible_contact: bool) -> float:
        """Advance by a finite positive duration and return updated hunger."""

        dt = _finite_number(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        if type(edible_contact) is not bool:
            raise ValueError("edible_contact must be a bool")

        contact_total = self.cumulative_edible_contact_s
        if edible_contact:
            contact_total += dt
            if not math.isfinite(contact_total):
                raise ValueError("cumulative edible contact seconds must remain finite")

        # Use threshold checks before multiplication so even very large finite
        # durations saturate cleanly instead of overflowing intermediate math.
        net_rate = self.basal_hunger_per_s
        if edible_contact:
            net_rate -= self.satiety_per_s
        if net_rate > 0.0:
            time_to_bound = (1.0 - self._hunger) / net_rate
            hunger = 1.0 if dt >= time_to_bound else self._hunger + net_rate * dt
        elif net_rate < 0.0:
            time_to_bound = self._hunger / -net_rate
            hunger = 0.0 if dt >= time_to_bound else self._hunger + net_rate * dt
        else:
            hunger = self._hunger

        self._hunger = min(1.0, max(0.0, hunger))
        self.cumulative_edible_contact_s = contact_total
        return self._hunger

    def telemetry(self) -> dict[str, Any]:
        """Return state plus explicit engineering status and model scope."""

        return {
            "hunger": self._hunger,
            "cumulative_edible_contact_s": self.cumulative_edible_contact_s,
            "basal_hunger_per_s": self.basal_hunger_per_s,
            "satiety_per_s": self.satiety_per_s,
            "status": FORAGING_STATE_STATUS,
            "scope": FORAGING_STATE_SCOPE,
        }
