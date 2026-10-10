#!/usr/bin/env python3
"""Load and query an explicit schedule of direct neural-input interventions.

This module is a read-only schedule source.  It does not resolve selectors,
draw Poisson samples, advance neural state, issue motor/body commands, set
goals, or choose behavior.  A caller must opt in and pass the returned
selector-specific values through its separate neural-input channel.

Schedule windows are half-open ``[start_s, end_s)`` intervals.  Overlapping
entries with the same selector and units add; rate and current channels are
kept separate.  The query methods are pure and return fresh dictionaries.

Import API::

    schedule = load_schedule("config/neural_intervention_ttmn_pulse.json")
    rates = schedule.rates_at(time_s)       # {selector: poisson_hz}
    currents = schedule.currents_at(time_s) # {selector: current_mV_per_ms}

The numeric limits below mirror the MaleCNS external-drive admission limits.
They are software bounds only and make no calibrated or biological claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = "neural-intervention-schedule-1"
POISSON_HZ = "poisson_hz"
CURRENT_UNITS = "current_mV_per_ms"
SUPPORTED_UNITS = frozenset({POISSON_HZ, CURRENT_UNITS})

# Engineering input caps enforced by the MaleCNS drive boundary.  The limits
# prevent this helper from emitting values the runtime will reject; they are
# not firing-rate or current calibrations.
RUNTIME_BOUNDS: Mapping[str, tuple[float, float]] = MappingProxyType({
    POISSON_HZ: (0.0, 2000.0),
    CURRENT_UNITS: (-20.0, 20.0),
})

_SELECTOR_KINDS = frozenset({
    "prefix", "contains", "type", "class", "superclass", "subclass",
    "nt", "nerve", "neuromere", "dimorphism", "frudsx", "body",
})
_ROOT_KEYS = frozenset({
    "schema_version", "name", "enabled", "opt_in_required",
    "intervention_type", "counts_as_autonomous_foraging",
    "classification", "prohibited_outputs", "bounds", "entries",
})
_ENTRY_KEYS = frozenset({
    "selector", "units", "start_s", "end_s", "value", "expected_match_count",
})


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_selector(selector: Any, entry_number: int) -> str:
    """Validate the repository population-selector grammar without resolving it."""

    if not isinstance(selector, str) or not selector.strip():
        raise ValueError(f"entries[{entry_number}].selector must be a non-empty string")
    text = selector.strip()
    terms = text.split(",")
    if any(not term.strip() for term in terms):
        raise ValueError(f"entries[{entry_number}].selector has an empty comma term")

    for raw_term in terms:
        term = raw_term.strip()
        if term.endswith(("/L", "/R")):
            term = term[:-2].strip()
        if "/" in term:
            raise ValueError(
                f"entries[{entry_number}].selector has an invalid side suffix in {raw_term!r}"
            )
        atoms = term.split("&")
        if any(not atom.strip() for atom in atoms):
            raise ValueError(f"entries[{entry_number}].selector has an empty intersection atom")
        for raw_atom in atoms:
            atom = raw_atom.strip()
            if any(char.isspace() for char in atom):
                raise ValueError(
                    f"entries[{entry_number}].selector atoms cannot contain internal whitespace"
                )
            if atom.count(":") > 1:
                raise ValueError(f"entries[{entry_number}].selector has invalid atom {atom!r}")
            if ":" in atom:
                kind, value = atom.split(":", 1)
                if kind.lower() not in _SELECTOR_KINDS or not value:
                    raise ValueError(f"entries[{entry_number}].selector has invalid atom {atom!r}")
                if any(char in value for char in ",&/()"):
                    raise ValueError(f"entries[{entry_number}].selector has invalid value {value!r}")
                if kind.lower() == "body":
                    try:
                        body_id = int(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"entries[{entry_number}].selector body ID must be an integer"
                        ) from exc
                    if body_id < 0 or str(body_id) != value:
                        raise ValueError(
                            f"entries[{entry_number}].selector body ID must be a non-negative integer"
                        )
            elif any(char in atom for char in ",&/()"):
                raise ValueError(f"entries[{entry_number}].selector has invalid exact name {atom!r}")
    return text


@dataclass(frozen=True)
class ScheduledInput:
    """One immutable, selector-specific input window."""

    selector: str
    units: str
    start_s: float
    end_s: float
    value: float
    expected_match_count: int | None = None


@dataclass(frozen=True)
class NeuralInterventionSchedule:
    """Immutable, explicit neural-input schedule with pure time queries."""

    name: str
    enabled: bool
    opt_in_required: bool
    intervention_type: str
    counts_as_autonomous_foraging: bool
    classification: str
    prohibited_outputs: tuple[str, ...]
    bounds: Mapping[str, tuple[float, float]]
    entries: tuple[ScheduledInput, ...]

    def _values_at(self, time_s: float, units: str) -> dict[str, float]:
        at = _number(time_s, "time_s")
        if at < 0.0:
            raise ValueError("time_s must be non-negative")
        if not self.enabled:
            return {}
        active: dict[str, list[float]] = {}
        for entry in self.entries:
            if entry.units == units and entry.start_s <= at < entry.end_s:
                active.setdefault(entry.selector, []).append(entry.value)
        low, high = self.bounds[units]
        result: dict[str, float] = {}
        for selector in sorted(active):
            value = math.fsum(active[selector])
            if value < low or value > high:
                raise ValueError(
                    f"overlapping {units} entries for {selector!r} sum to {value:g}, "
                    f"outside configured bounds [{low:g}, {high:g}]"
                )
            result[selector] = value
        return result

    def rates_at(self, time_s: float) -> dict[str, float]:
        """Return active Poisson rates keyed by selector, in Hz per neuron."""

        return self._values_at(time_s, POISSON_HZ)

    def currents_at(self, time_s: float) -> dict[str, float]:
        """Return active current inputs keyed by selector, in mV/ms."""

        return self._values_at(time_s, CURRENT_UNITS)


def _parse_bounds(raw: Any) -> Mapping[str, tuple[float, float]]:
    if not isinstance(raw, dict) or set(raw) != SUPPORTED_UNITS:
        raise ValueError("bounds must define exactly poisson_hz and current_mV_per_ms")
    parsed: dict[str, tuple[float, float]] = {}
    for units in (POISSON_HZ, CURRENT_UNITS):
        bounds = raw[units]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"bounds.{units} must be [minimum, maximum]")
        low = _number(bounds[0], f"bounds.{units}[0]")
        high = _number(bounds[1], f"bounds.{units}[1]")
        if low > high:
            raise ValueError(f"bounds.{units} minimum must not exceed maximum")
        runtime_low, runtime_high = RUNTIME_BOUNDS[units]
        if low < runtime_low or high > runtime_high:
            raise ValueError(
                f"bounds.{units} must stay within the MaleCNS software limits "
                f"[{runtime_low:g}, {runtime_high:g}]"
            )
        if units == POISSON_HZ and low < 0.0:
            raise ValueError("poisson_hz bounds cannot include negative rates")
        parsed[units] = (low, high)
    return MappingProxyType(parsed)


def _validate_aggregate_bounds(
    entries: list[ScheduledInput], bounds: Mapping[str, tuple[float, float]],
) -> None:
    """Reject same-selector overlays that exceed the declared signal bounds."""

    grouped: dict[tuple[str, str], list[ScheduledInput]] = {}
    for entry in entries:
        grouped.setdefault((entry.selector, entry.units), []).append(entry)
    for (selector, units), group in grouped.items():
        low, high = bounds[units]
        # Any change in an active sum occurs at a start/end boundary. Checking
        # every unique start captures the value for each half-open segment.
        for at in sorted({entry.start_s for entry in group}):
            value = math.fsum(
                entry.value for entry in group if entry.start_s <= at < entry.end_s
            )
            if value < low or value > high:
                raise ValueError(
                    f"overlapping {units} entries for {selector!r} sum to {value:g} "
                    f"at t={at:g}s, outside configured bounds [{low:g}, {high:g}]"
                )


def load_schedule(path: str | Path) -> NeuralInterventionSchedule:
    """Read, validate, and return a static schedule from a JSON file.

    No file is created or modified. Selector syntax is validated here, while
    selector resolution and expected population-size checks belong to the
    caller's population index at the neural boundary.
    """

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read intervention schedule {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("schedule root must be a JSON object")
    unknown = set(raw) - _ROOT_KEYS
    missing = _ROOT_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            f"schedule keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError("name must be a non-empty string")
    for key in ("enabled", "opt_in_required", "counts_as_autonomous_foraging"):
        if not isinstance(raw[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    if raw["opt_in_required"] is not True:
        raise ValueError("opt_in_required must be true for neural interventions")
    if raw["counts_as_autonomous_foraging"] is not False:
        raise ValueError("neural interventions cannot count as autonomous foraging")
    if raw["intervention_type"] != "direct_neural_input_stimulation":
        raise ValueError("intervention_type must identify direct neural-input stimulation")
    if raw["classification"] != "neuron_stimulation_intervention":
        raise ValueError("classification must be neuron_stimulation_intervention")
    prohibited = raw["prohibited_outputs"]
    required_prohibited = {
        "body_command", "motor_command", "target_goal", "autopilot_decision",
    }
    if (
        not isinstance(prohibited, list)
        or any(not isinstance(item, str) for item in prohibited)
        or not required_prohibited.issubset(prohibited)
    ):
        raise ValueError("prohibited_outputs must list body, motor, goal, and autopilot outputs")
    bounds = _parse_bounds(raw["bounds"])
    rows = raw["entries"]
    if not isinstance(rows, list):
        raise ValueError("entries must be a JSON array")
    entries: list[ScheduledInput] = []
    for number, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"entries[{number}] must be a JSON object")
        unknown_entry = set(row) - _ENTRY_KEYS
        required_entry = _ENTRY_KEYS - {"expected_match_count"}
        missing_entry = required_entry - set(row)
        if unknown_entry or missing_entry:
            raise ValueError(
                f"entries[{number}] keys mismatch; missing={sorted(missing_entry)}, "
                f"unknown={sorted(unknown_entry)}"
            )
        selector = _validate_selector(row["selector"], number)
        units = row["units"]
        if units not in SUPPORTED_UNITS:
            raise ValueError(f"entries[{number}].units must be one of {sorted(SUPPORTED_UNITS)}")
        start_s = _number(row["start_s"], f"entries[{number}].start_s")
        end_s = _number(row["end_s"], f"entries[{number}].end_s")
        value = _number(row["value"], f"entries[{number}].value")
        if start_s < 0.0 or end_s <= start_s:
            raise ValueError(f"entries[{number}] must use a non-empty non-negative time window")
        low, high = bounds[units]
        if value < low or value > high:
            raise ValueError(
                f"entries[{number}].value {value:g} is outside configured {units} bounds "
                f"[{low:g}, {high:g}]"
            )
        expected_count = row.get("expected_match_count")
        if expected_count is not None and (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 1
        ):
            raise ValueError(f"entries[{number}].expected_match_count must be a positive integer")
        entries.append(ScheduledInput(
            selector=selector,
            units=units,
            start_s=start_s,
            end_s=end_s,
            value=value,
            expected_match_count=expected_count,
        ))

    _validate_aggregate_bounds(entries, bounds)

    return NeuralInterventionSchedule(
        name=raw["name"].strip(),
        enabled=raw["enabled"],
        opt_in_required=raw["opt_in_required"],
        intervention_type=raw["intervention_type"],
        counts_as_autonomous_foraging=raw["counts_as_autonomous_foraging"],
        classification=raw["classification"],
        prohibited_outputs=tuple(prohibited),
        bounds=bounds,
        entries=tuple(entries),
    )


__all__ = [
    "CURRENT_UNITS", "NeuralInterventionSchedule", "POISSON_HZ",
    "RUNTIME_BOUNDS", "SCHEMA_VERSION", "ScheduledInput", "load_schedule",
]
