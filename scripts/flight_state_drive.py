#!/usr/bin/env python3
"""Deterministic high-level state drives for the MaleCNS external boundary.

This module is deliberately narrower than a flight controller.  It converts a
scenario label, normalized phase, and *already computed* body/goal errors into
explicit external population inputs for :class:`malecns_engine.MaleCNSEngine`.
It never writes a body pose, integrates a trajectory, creates spikes, or
emits a motor command.

The default configuration uses disjoint ``prefix:DNg02/L`` and
``prefix:DNg02/R`` Poisson-rate selectors.  A 35 Hz per-neuron rate is the
hand-built hover anchor.  Takeoff and landing descending-neuron populations
are optional state channels in the same packet; their roles and selectors are
documented in ``config/flight_state_drive.json`` and cross-referenced to the
checked ``neuromuscular_map.json``.  Their absolute rates remain hypotheses,
not in-vivo measurements.

Runner example::

    adapter = FlightStateDrive.from_json()
    packet = adapter.step({
        "scenario": "takeoff",
        "phase": 0.25,
        "body_error": {"altitude_error_m": 0.18},
    })
    sensory_channels = packet.as_engine_channels()
    # SensoryFrame(..., channels=sensory_channels)

``packet.as_engine_channels()`` contains only the engine's explicit
``poisson_hz`` and ``current_mV_per_ms`` population mappings.  Use
``packet.to_dict()`` for audit metadata, units, bounds, and provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

try:
    from .flight_control_common import clip as _clip, json_safe as _json_safe, normal_key as _normal_key
except ImportError:
    from flight_control_common import clip as _clip, json_safe as _json_safe, normal_key as _normal_key


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "config" / "flight_state_drive.json"

SCHEMA_VERSION = "flight-state-drive-1"
POISSON_UNITS = "poisson_hz"
CURRENT_UNITS = "current_mV_per_ms"
SUPPORTED_UNITS = frozenset({POISSON_UNITS, CURRENT_UNITS})
SOURCE = "handbuilt"
BIOLOGICAL_STATUS = "hypothesis"
CALIBRATION_KIND = "calibration"


class StateDriveError(ValueError):
    """Base error for malformed state-drive configuration or input."""


class StateDriveConfigError(StateDriveError):
    """Raised for an incomplete, unsafe, or ambiguous configuration."""


class StateDriveInputError(StateDriveError):
    """Raised when the runner supplies a non-finite or invalid state."""


# Compatibility spellings are useful to small runner adapters and keep the
# public errors discoverable without importing another project module.
FlightStateDriveError = StateDriveError
FlightStateDriveConfigError = StateDriveConfigError
FlightStateDriveInputError = StateDriveInputError


def _finite(value: Any, name: str, *, error: type[StateDriveError] = StateDriveInputError) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise error(f"{name} must be a finite numeric scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise error(f"{name} must be a finite numeric scalar") from exc
    if not math.isfinite(result):
        raise error(f"{name} must be a finite numeric scalar")
    return result


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        return {field_name: getattr(value, field_name) for field_name in fields}
    result: dict[str, Any] = {}
    for field_name in (
        "scenario", "scenario_name", "phase", "scenario_phase", "progress",
        "goal", "body_error", "errors", "goal_error",
        "altitude_error_m", "altitude_error", "vertical_error_m", "z_error_m",
        "heading_error_rad", "heading_error", "yaw_error_rad", "yaw_error",
        "target_heading_error_rad", "lateral_error_m", "lateral_error",
        "cross_track_error_m", "forward_error_m", "forward_error",
        "vertical_speed_mps", "vertical_speed", "vz_mps", "speed_error_mps",
        "speed_error", "airspeed_error_mps", "roll_error_rad", "pitch_error_rad",
        "yaw_rate_rad_s", "yaw_rate", "grounded", "contact",
    ):
        if hasattr(value, field_name):
            result[field_name] = getattr(value, field_name)
    if not result:
        raise StateDriveInputError(
            f"{name} must be a mapping or an object exposing state-drive fields"
        )
    return result


_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "altitude_error_m": (
        "altitude_error_m", "altitude_error", "vertical_error_m", "z_error_m",
    ),
    "heading_error_rad": (
        "heading_error_rad", "heading_error", "yaw_error_rad", "yaw_error",
        "target_heading_error_rad",
    ),
    "lateral_error_m": (
        "lateral_error_m", "lateral_error", "cross_track_error_m",
    ),
    "forward_error_m": ("forward_error_m", "forward_error", "longitudinal_error_m"),
    "vertical_speed_mps": ("vertical_speed_mps", "vertical_speed", "vz_mps"),
    "speed_error_mps": ("speed_error_mps", "speed_error", "airspeed_error_mps"),
    "roll_error_rad": ("roll_error_rad", "roll_error"),
    "pitch_error_rad": ("pitch_error_rad", "pitch_error"),
    "yaw_rate_rad_s": ("yaw_rate_rad_s", "yaw_rate", "yaw_rate_radps"),
}


def _lookup(raw: Mapping[str, Any], names: Sequence[str]) -> tuple[Any, bool]:
    lowered = {_normal_key(key): value for key, value in raw.items()}
    for name in names:
        key = _normal_key(name)
        if key in lowered:
            return lowered[key], True
    return None, False


def _merged_state_sources(raw: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return error sources in precedence order without reading body poses."""

    sources: list[Mapping[str, Any]] = []
    for key in ("body_error", "errors", "goal_error", "goal"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            sources.append(_as_mapping(value, key))
        except StateDriveInputError:
            raise StateDriveInputError(f"{key} must expose precomputed error fields")
    sources.append(raw)
    return tuple(sources)


@dataclass(frozen=True)
class FlightState:
    """The small state contract consumed by :class:`FlightStateDrive`.

    All error fields are supplied by the runner; this class does not derive
    them from positions.  ``heading_error_rad`` is positive when the desired
    heading is to the right of the current heading.  In the default
    calibration, DNg02 is contralateral: positive error therefore increases
    DNg02/L and decreases DNg02/R, which is the positive right-wing steering
    convention in ``neuromuscular_map.json``.  The sign is explicit in the
    config manifest.
    """

    scenario: str = "hover"
    phase: float = 0.0
    altitude_error_m: float = 0.0
    heading_error_rad: float = 0.0
    lateral_error_m: float = 0.0
    forward_error_m: float = 0.0
    vertical_speed_mps: float = 0.0
    speed_error_mps: float = 0.0
    roll_error_rad: float = 0.0
    pitch_error_rad: float = 0.0
    yaw_rate_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, str) or not self.scenario.strip():
            raise StateDriveInputError("state.scenario must be non-empty text")
        scenario = self.scenario.strip()
        phase = _finite(self.phase, "state.phase")
        if not 0.0 <= phase <= 1.0:
            raise StateDriveInputError("state.phase must be in [0, 1]")
        object.__setattr__(self, "scenario", scenario)
        object.__setattr__(self, "phase", phase)
        for field_name in _FEATURE_ALIASES:
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), f"state.{field_name}"),
            )

    scenario_name = property(lambda self: self.scenario)
    scenario_phase = property(lambda self: self.phase)
    altitude_error = property(lambda self: self.altitude_error_m)
    heading_error = property(lambda self: self.heading_error_rad)

    def features(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in _FEATURE_ALIASES
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlightState":
        if not isinstance(value, Mapping):
            raise StateDriveInputError("state must be a mapping")
        raw = dict(value)
        scenario, present = _lookup(raw, ("scenario", "scenario_name", "phase_name"))
        scenario = "hover" if not present else scenario
        if not isinstance(scenario, str) or not scenario.strip():
            raise StateDriveInputError("state.scenario must be non-empty text")
        phase, present = _lookup(raw, ("phase", "scenario_phase", "progress"))
        phase = 0.0 if not present else phase
        sources = _merged_state_sources(raw)
        features: dict[str, Any] = {}
        for feature, aliases in _FEATURE_ALIASES.items():
            found = False
            for source in sources:
                candidate, candidate_present = _lookup(source, aliases)
                if candidate_present:
                    features[feature] = candidate
                    found = True
                    break
            if not found:
                features[feature] = 0.0
        return cls(scenario=scenario.strip(), phase=phase, **features)

    @classmethod
    def coerce(cls, value: Any = None) -> "FlightState":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(scenario=value)
        return cls.from_mapping(_as_mapping(value, "state"))


# Names used by adjacent runner code and by earlier prototypes.
FlightStateInput = FlightState
StateDriveRequest = FlightState
StateDriveInput = FlightState
FlightStateContext = FlightState


@dataclass(frozen=True)
class PopulationRule:
    """One explicit selector and its bounded state-to-input calibration."""

    name: str
    population: str
    units: str
    bounds: tuple[float, float]
    profiles: Mapping[str, Any]
    coefficients: Mapping[str, float] = field(default_factory=dict)
    source: str = SOURCE
    biological_status: str = BIOLOGICAL_STATUS
    calibration: str = CALIBRATION_KIND
    rationale: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise StateDriveConfigError("population input name must be non-empty")
        if not self.population.strip():
            raise StateDriveConfigError(f"population {self.name!r} selector must be non-empty")
        if self.units not in SUPPORTED_UNITS:
            raise StateDriveConfigError(f"{self.name} has unsupported units {self.units!r}")
        low = _finite(self.bounds[0], f"{self.name}.bounds[0]", error=StateDriveConfigError)
        high = _finite(self.bounds[1], f"{self.name}.bounds[1]", error=StateDriveConfigError)
        if low >= high:
            raise StateDriveConfigError(f"{self.name}.bounds must be increasing")
        if self.units == POISSON_UNITS and low < 0.0:
            raise StateDriveConfigError(f"{self.name}.poisson_hz lower bound cannot be negative")
        object.__setattr__(self, "bounds", (low, high))
        object.__setattr__(self, "profiles", dict(self.profiles))
        object.__setattr__(self, "coefficients", dict(self.coefficients))
        if self.source != SOURCE:
            raise StateDriveConfigError(
                f"{self.name}.source must be {SOURCE!r}; state drives are not connectome output"
            )
        if not isinstance(self.biological_status, str) or not self.biological_status.strip():
            raise StateDriveConfigError(f"{self.name}.biological_status must be non-empty")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> "PopulationRule":
        if not isinstance(raw, Mapping):
            raise StateDriveConfigError(f"population_inputs[{index}] must be an object")
        name = raw.get("name", f"population_{index}")
        if not isinstance(name, str) or not name.strip():
            raise StateDriveConfigError(f"population_inputs[{index}].name must be non-empty")
        target = raw.get("target", {})
        if isinstance(target, Mapping):
            population = raw.get("population", raw.get("spec", target.get("population", target.get("spec"))))
        else:
            population = raw.get("population", raw.get("spec"))
        if not isinstance(population, str) or not population.strip():
            raise StateDriveConfigError(f"{name}.population must be a non-empty selector")
        units = str(raw.get("units", raw.get("input_units", POISSON_UNITS))).strip()
        if units in {"poisson", "rate", "rates_hz", "hz"}:
            units = POISSON_UNITS
        if units in {"current", "current_mV_ms", "mV_per_ms"}:
            units = CURRENT_UNITS
        if units not in SUPPORTED_UNITS:
            raise StateDriveConfigError(f"{name}.units must be poisson_hz or current_mV_per_ms")
        bounds_raw = raw.get("bounds", raw.get("output_bounds"))
        if bounds_raw is None:
            bounds_raw = (0.0, 80.0) if units == POISSON_UNITS else (-1.0, 1.0)
        if isinstance(bounds_raw, (str, bytes)) or len(bounds_raw) != 2:
            raise StateDriveConfigError(f"{name}.bounds must contain two values")
        bounds = (
            _finite(bounds_raw[0], f"{name}.bounds[0]", error=StateDriveConfigError),
            _finite(bounds_raw[1], f"{name}.bounds[1]", error=StateDriveConfigError),
        )
        profiles = raw.get("profiles", raw.get("phase_profiles", raw.get("rates_hz")))
        if profiles is None:
            profiles = {"hover": raw.get("value", raw.get("base", 0.0))}
        if not isinstance(profiles, Mapping):
            raise StateDriveConfigError(f"{name}.profiles must be an object")
        coefficients_raw = raw.get("coefficients", raw.get("error_coefficients", {}))
        if coefficients_raw is None:
            coefficients_raw = {}
        if not isinstance(coefficients_raw, Mapping):
            raise StateDriveConfigError(f"{name}.coefficients must be an object")
        aliases = {_normal_key(alias): feature for feature, names in _FEATURE_ALIASES.items() for alias in names}
        coefficients: dict[str, float] = {}
        for key, value in coefficients_raw.items():
            feature = aliases.get(_normal_key(key))
            if feature is None:
                raise StateDriveConfigError(f"{name}.coefficients has unsupported feature {key!r}")
            coefficients[feature] = _finite(
                value, f"{name}.coefficients[{key!r}]", error=StateDriveConfigError
            )
        source = str(raw.get("source", SOURCE)).strip()
        status = str(raw.get("biological_status", raw.get("status", BIOLOGICAL_STATUS))).strip()
        calibration = str(raw.get("calibration", CALIBRATION_KIND)).strip()
        rationale = raw.get("rationale", "")
        description = raw.get("description", "")
        if not isinstance(rationale, str) or not isinstance(description, str):
            raise StateDriveConfigError(f"{name}.rationale and description must be text")
        return cls(
            name=name.strip(), population=population.strip(), units=units,
            bounds=bounds, profiles=profiles, coefficients=coefficients,
            source=source, biological_status=status, calibration=calibration,
            rationale=rationale, description=description,
        )

    @staticmethod
    def _profile_scalar(profile: Any, phase: float, name: str) -> float:
        if isinstance(profile, Mapping):
            if "value" in profile:
                return _finite(profile["value"], f"{name}.profile.value", error=StateDriveConfigError)
            for key in ("rate_hz", "current_mV_per_ms", "base"):
                if key in profile:
                    return _finite(profile[key], f"{name}.profile.{key}", error=StateDriveConfigError)
            start = profile.get("start", profile.get("from", profile.get("start_value")))
            end = profile.get("end", profile.get("to", profile.get("end_value")))
            if start is None and end is None:
                raise StateDriveConfigError(f"{name}.profile needs value or start/end")
            if start is None:
                start = end
            if end is None:
                end = start
            start_f = _finite(start, f"{name}.profile.start", error=StateDriveConfigError)
            end_f = _finite(end, f"{name}.profile.end", error=StateDriveConfigError)
            easing = _normal_key(profile.get("easing", "smoothstep"))
        elif isinstance(profile, Sequence) and not isinstance(profile, (str, bytes)):
            if len(profile) != 2:
                raise StateDriveConfigError(f"{name}.profile sequence must contain start/end")
            start_f = _finite(profile[0], f"{name}.profile[0]", error=StateDriveConfigError)
            end_f = _finite(profile[1], f"{name}.profile[1]", error=StateDriveConfigError)
            easing = "smoothstep"
        else:
            return _finite(profile, f"{name}.profile", error=StateDriveConfigError)
        if easing in {"linear", "lerp"}:
            fraction = phase
        elif easing in {"smoothstep", "smooth_step", "cubic"}:
            fraction = phase * phase * (3.0 - 2.0 * phase)
        else:
            raise StateDriveConfigError(f"{name}.profile has unsupported easing {easing!r}")
        return start_f + (end_f - start_f) * fraction

    def base_value(self, scenario: str, phase: float, fallback: str = "hover") -> float:
        profile = self.profiles.get(scenario)
        if profile is None:
            profile = self.profiles.get(fallback)
        if profile is None:
            profile = self.profiles.get("flight")
        if profile is None:
            profile = next(iter(self.profiles.values()), 0.0)
        return self._profile_scalar(profile, phase, self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "population": self.population,
            "units": self.units,
            "bounds": list(self.bounds),
            "profiles": _json_safe(self.profiles),
            "coefficients": dict(self.coefficients),
            "source": self.source,
            "biological_status": self.biological_status,
            "calibration": self.calibration,
            "rationale": self.rationale,
            "description": self.description,
        }


@dataclass(frozen=True)
class StateDriveConfig:
    """Validated immutable state-drive configuration."""

    name: str
    rules: tuple[PopulationRule, ...]
    provenance: Mapping[str, Any]
    default_profile: str = "hover"
    input_contract: Mapping[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    config_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise StateDriveConfigError("config.name must be non-empty")
        if not self.rules:
            raise StateDriveConfigError("config must declare at least one population input")
        seen_names: set[str] = set()
        seen_populations: set[str] = set()
        for rule in self.rules:
            if rule.name in seen_names:
                raise StateDriveConfigError(f"duplicate population input name {rule.name!r}")
            if rule.population in seen_populations:
                raise StateDriveConfigError(
                    f"population selectors must be disjoint; duplicate {rule.population!r}"
                )
            seen_names.add(rule.name)
            seen_populations.add(rule.population)
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "input_contract", dict(self.input_contract))

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        config_path: Optional[str] = None,
        config_sha256: Optional[str] = None,
    ) -> "StateDriveConfig":
        if not isinstance(raw, Mapping):
            raise StateDriveConfigError("flight state-drive config must be an object")
        schema = raw.get("schema_version", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            raise StateDriveConfigError(
                f"unsupported schema_version {schema!r}; expected {SCHEMA_VERSION!r}"
            )
        name = raw.get("name", "flight-state-drive")
        if not isinstance(name, str) or not name.strip():
            raise StateDriveConfigError("config.name must be non-empty text")
        raw_rules = raw.get("population_inputs", raw.get("channels"))
        if isinstance(raw_rules, Mapping):
            entries: list[Mapping[str, Any]] = []
            for key, value in raw_rules.items():
                if not isinstance(value, Mapping):
                    raise StateDriveConfigError(f"channels[{key!r}] must be an object")
                entry = dict(value)
                entry.setdefault("name", str(key))
                entries.append(entry)
        elif isinstance(raw_rules, Sequence) and not isinstance(raw_rules, (str, bytes)):
            entries = list(raw_rules)
        else:
            raise StateDriveConfigError("population_inputs must be an array or object")
        rules = tuple(PopulationRule.from_mapping(entry, index=i) for i, entry in enumerate(entries))
        provenance = raw.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise StateDriveConfigError("config.provenance must be an object")
        source = provenance.get("source", SOURCE)
        if source != SOURCE:
            raise StateDriveConfigError("config.provenance.source must be 'handbuilt'")
        status = provenance.get("biological_status", provenance.get("status", BIOLOGICAL_STATUS))
        if not isinstance(status, str) or not status.strip():
            raise StateDriveConfigError("config.provenance.biological_status must be non-empty")
        input_contract = raw.get("input_contract", {})
        if not isinstance(input_contract, Mapping):
            raise StateDriveConfigError("config.input_contract must be an object")
        return cls(
            name=name.strip(), rules=rules, provenance=dict(provenance),
            default_profile=str(raw.get("default_profile", "hover")).strip(),
            input_contract=input_contract,
            config_path=config_path, config_sha256=config_sha256,
        )

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONFIG) -> "StateDriveConfig":
        config_path = Path(path).expanduser()
        try:
            payload = config_path.read_bytes()
            raw = json.loads(payload.decode("utf-8"))
        except OSError as exc:
            raise StateDriveConfigError(f"cannot read state-drive config {config_path}: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateDriveConfigError(f"invalid state-drive config {config_path}: {exc}") from exc
        return cls.from_mapping(
            raw, config_path=str(config_path),
            config_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "default_profile": self.default_profile,
            "input_contract": _json_safe(self.input_contract),
            "population_inputs": [rule.to_dict() for rule in self.rules],
            "provenance": _json_safe(self.provenance),
        }

    def manifest(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.update({
            "deterministic": True,
            "stateful_values": False,
            "output_contract": {
                "kind": "external_population_inputs",
                "units": [POISSON_UNITS, CURRENT_UNITS],
                "population_selectors_are_disjoint": True,
                "spikes_emitted": False,
                "body_positions_emitted": False,
                "motor_commands_emitted": False,
            },
            "reset": {
                "supported": True,
                "policy": "clear sequence and last-packet telemetry; no stochastic state",
            },
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "bounds": {
                rule.population: {
                    "units": rule.units,
                    "range": list(rule.bounds),
                }
                for rule in self.rules
            },
        })
        return _json_safe(payload)


@dataclass(frozen=True)
class PopulationInput:
    """One bounded explicit external population value."""

    name: str
    population: str
    units: str
    value: float
    bounds: tuple[float, float]
    source: str = SOURCE
    biological_status: str = BIOLOGICAL_STATUS
    calibration: str = CALIBRATION_KIND
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.units not in SUPPORTED_UNITS:
            raise StateDriveError(f"unsupported population input units {self.units!r}")
        value = _finite(self.value, f"{self.name}.value")
        low, high = self.bounds
        low = _finite(low, f"{self.name}.bounds[0]")
        high = _finite(high, f"{self.name}.bounds[1]")
        if low >= high or value < low - 1e-12 or value > high + 1e-12:
            raise StateDriveError(f"{self.name}.value is outside configured bounds")
        if self.source != SOURCE:
            raise StateDriveError(f"{self.name}.source must be {SOURCE!r}")
        object.__setattr__(self, "value", _clip(value, low, high))
        object.__setattr__(self, "bounds", (low, high))

    @property
    def rate_hz(self) -> float:
        if self.units != POISSON_UNITS:
            raise AttributeError("this population input is not a poisson_hz rate")
        return self.value

    @property
    def current_mV_per_ms(self) -> float:
        if self.units != CURRENT_UNITS:
            raise AttributeError("this population input is not a current_mV_per_ms value")
        return self.value

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "source": self.source,
            "biological_status": self.biological_status,
            "status": self.biological_status,
            "calibration": self.calibration,
            "kind": self.calibration,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "population": self.population,
            "units": self.units,
            "value": self.value,
            "bounds": list(self.bounds),
            "source": self.source,
            "biological_status": self.biological_status,
            "calibration": self.calibration,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class StateDriveOutput(Mapping[str, Any]):
    """A runner packet containing only explicit external population inputs."""

    channels: Mapping[str, Mapping[str, float]]
    population_inputs: tuple[PopulationInput, ...]
    state: FlightState
    mode: str
    sequence: int
    units: Mapping[str, str]
    bounds: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any]
    saturated: bool = False

    def __post_init__(self) -> None:
        allowed = SUPPORTED_UNITS
        clean_channels: dict[str, dict[str, float]] = {}
        for unit, values in self.channels.items():
            if unit not in allowed:
                raise StateDriveError(f"output contains unsupported channel unit {unit!r}")
            if not isinstance(values, Mapping):
                raise StateDriveError(f"output channel {unit!r} must be a population mapping")
            clean_channels[unit] = {
                str(population): _finite(value, f"channels[{unit}][{population}]")
                for population, value in values.items()
            }
        object.__setattr__(self, "channels", clean_channels)
        object.__setattr__(self, "population_inputs", tuple(self.population_inputs))
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "bounds", _json_safe(self.bounds))
        object.__setattr__(self, "provenance", _json_safe(self.provenance))

    @property
    def population_rates_hz(self) -> Mapping[str, float]:
        return dict(self.channels.get(POISSON_UNITS, {}))

    @property
    def rates_hz(self) -> Mapping[str, float]:
        return self.population_rates_hz

    @property
    def population_currents_mV_per_ms(self) -> Mapping[str, float]:
        return dict(self.channels.get(CURRENT_UNITS, {}))

    engine_channels = property(lambda self: self.as_engine_channels())

    def as_engine_channels(self) -> dict[str, dict[str, float]]:
        """Return a fresh ``SensoryFrame.channels``-compatible mapping."""

        # Keep both explicit unit keys present.  Empty maps are accepted by
        # MaleCNSEngine and make the packet shape stable for mixed configs.
        return {
            POISSON_UNITS: dict(self.channels.get(POISSON_UNITS, {})),
            CURRENT_UNITS: dict(self.channels.get(CURRENT_UNITS, {})),
        }

    as_channels = as_engine_channels
    to_engine_channels = as_engine_channels

    @property
    def source(self) -> str:
        return str(self.provenance.get("source", SOURCE))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": int(self.sequence),
            "mode": self.mode,
            "state": self.state.to_dict(),
            "channels": self.as_engine_channels(),
            "population_inputs": [item.to_dict() for item in self.population_inputs],
            "units": dict(self.units),
            "bounds": _json_safe(self.bounds),
            "provenance": _json_safe(self.provenance),
            "saturated": bool(self.saturated),
        }

    def __getitem__(self, key: str) -> Any:
        if key in self.channels:
            return self.channels[key]
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        # Mapping iteration is intentionally the engine-facing view, so a
        # runner may pass ``packet`` directly where a ``SensoryFrame`` expects
        # ``channels``.  Audit metadata remains available via ``to_dict`` and
        # named accessors below.
        return iter(self.as_engine_channels())

    def __len__(self) -> int:
        return len(SUPPORTED_UNITS)

    def keys(self):
        return self.as_engine_channels().keys()

    def __contains__(self, key: object) -> bool:
        return key in self.channels or key in {
            "schema_version", "sequence", "mode", "state", "channels",
            "population_inputs", "units", "bounds", "provenance", "saturated",
        }


# Compatibility nouns for integrations that call these records population
# drives rather than state-drive packets.
ExternalPopulationInput = PopulationInput
PopulationRateInput = PopulationInput
PopulationDriveOutput = StateDriveOutput


class FlightStateDrive:
    """Bounded, deterministic state-to-population external-drive adapter."""

    def __init__(self, config: StateDriveConfig | Mapping[str, Any] | str | Path | None = None) -> None:
        if config is None:
            self.config = StateDriveConfig.from_json(DEFAULT_CONFIG)
        elif isinstance(config, StateDriveConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = StateDriveConfig.from_json(config)
        else:
            self.config = StateDriveConfig.from_mapping(config)
        self._sequence = 0
        self._last: Optional[StateDriveOutput] = None

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONFIG) -> "FlightStateDrive":
        return cls(StateDriveConfig.from_json(path))

    @classmethod
    def from_config(cls, config: StateDriveConfig | Mapping[str, Any] | str | Path) -> "FlightStateDrive":
        """Alias for runners that call their JSON/config loader ``from_config``."""

        return cls(config)

    @staticmethod
    def canonical_scenario(value: Any) -> str:
        key = _normal_key(value)
        aliases = {
            "ground": "grounded", "idle": "grounded", "on_ground": "grounded",
            "land": "landing", "landing": "landing", "approach": "approach",
            "take_off": "takeoff", "takeoff": "takeoff", "hover": "hover",
            "sustained": "flight", "cruise": "flight", "forward": "flight",
            "forward_target": "flight", "flight_demo": "flight", "flight": "flight",
            "yaw_step": "flight", "gust_recovery": "flight", "translation": "flight",
        }
        if key in aliases:
            return aliases[key]
        if "takeoff" in key or "take_off" in key:
            return "takeoff"
        if "landing" in key or "approach" in key:
            return "landing" if "landing" in key else "approach"
        if key in {"", "unknown"}:
            return "hover"
        return key

    def _coerce_state(
        self,
        state: Any = None,
        *,
        scenario: Optional[str] = None,
        phase: Optional[float] = None,
        goal: Any = None,
        body_error: Any = None,
        errors: Any = None,
        features: Mapping[str, Any] | None = None,
    ) -> FlightState:
        if state is None:
            raw: dict[str, Any] = {}
        elif isinstance(state, str):
            raw = {"scenario": state}
        elif isinstance(state, FlightState):
            raw = state.to_dict()
        else:
            raw = dict(_as_mapping(state, "state"))
        if scenario is not None:
            raw["scenario"] = scenario
        if phase is not None:
            raw["phase"] = phase
        if goal is not None:
            raw["goal"] = goal
        if body_error is not None:
            raw["body_error"] = body_error
        if errors is not None:
            raw["errors"] = errors
        if features:
            raw.update(dict(features))
        return FlightState.from_mapping(raw)

    def _make_output(self, state: FlightState) -> StateDriveOutput:
        mode = self.canonical_scenario(state.scenario)
        channels: dict[str, dict[str, float]] = {
            POISSON_UNITS: {},
            CURRENT_UNITS: {},
        }
        records: list[PopulationInput] = []
        bounds: dict[str, dict[str, Any]] = {}
        saturated = False
        values = state.features()
        for rule in self.config.rules:
            raw_value = rule.base_value(mode, state.phase, self.config.default_profile)
            raw_value += sum(
                coefficient * values.get(feature, 0.0)
                for feature, coefficient in rule.coefficients.items()
            )
            low, high = rule.bounds
            value = _clip(raw_value, low, high)
            saturated |= abs(value - raw_value) > 1.0e-12
            channels[rule.units][rule.population] = value
            records.append(PopulationInput(
                name=rule.name, population=rule.population, units=rule.units,
                value=value, bounds=rule.bounds, source=SOURCE,
                biological_status=rule.biological_status,
                calibration=rule.calibration, rationale=rule.rationale,
            ))
            bounds[rule.population] = {
                "units": rule.units,
                "range": list(rule.bounds),
            }
        self._sequence += 1
        provenance = {
            **dict(self.config.provenance),
            "source": SOURCE,
            "biological_status": BIOLOGICAL_STATUS,
            "status": BIOLOGICAL_STATUS,
            "calibration": CALIBRATION_KIND,
            "kind": CALIBRATION_KIND,
            "adapter": "flight_state_drive",
            "scenario": state.scenario,
            "mode": mode,
            "phase": state.phase,
            "body_positions_used": False,
            "motor_commands_emitted": False,
            "spikes_emitted": False,
        }
        output = StateDriveOutput(
            channels=channels, population_inputs=tuple(records), state=state,
            mode=mode, sequence=self._sequence,
            units={POISSON_UNITS: "Hz per neuron", CURRENT_UNITS: "mV/ms"},
            bounds=bounds, provenance=provenance, saturated=saturated,
        )
        self._last = output
        return output

    def step(
        self,
        state: Any = None,
        *,
        scenario: Optional[str] = None,
        phase: Optional[float] = None,
        goal: Any = None,
        body_error: Any = None,
        errors: Any = None,
        **features: Any,
    ) -> StateDriveOutput:
        """Convert one runner state into bounded explicit population inputs."""

        return self._make_output(self._coerce_state(
            state, scenario=scenario, phase=phase, goal=goal,
            body_error=body_error, errors=errors, features=features,
        ))

    drive = step
    compute = step
    update = step

    def channels_for(self, state: Any = None, **kwargs: Any) -> dict[str, dict[str, float]]:
        return self.step(state, **kwargs).as_engine_channels()

    to_channels = channels_for
    as_channels = channels_for

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset packet sequencing and telemetry; no stochastic state exists."""

        del seed  # accepted for runner symmetry; this adapter has no RNG
        self._sequence = 0
        self._last = None

    @property
    def last_output(self) -> Optional[StateDriveOutput]:
        return self._last

    def manifest(self) -> dict[str, Any]:
        payload = self.config.manifest()
        payload["runtime"] = {
            "sequence": self._sequence,
            "last_packet_available": self._last is not None,
            "reset_supported": True,
        }
        return _json_safe(payload)

    @property
    def output_manifest(self) -> dict[str, Any]:
        """Alias for callers that attach a packet manifest to telemetry."""

        return self.manifest()


# Friendly aliases for callers that prefer the shorter noun.
FlightStateDriveAdapter = FlightStateDrive
StateDriveAdapter = FlightStateDrive
StateDrive = FlightStateDrive
FlightStateDriveConfig = StateDriveConfig


def load_state_drive(path: str | Path = DEFAULT_CONFIG) -> FlightStateDrive:
    return FlightStateDrive.from_json(path)


def build_flight_state_drive(config: Any = None) -> FlightStateDrive:
    return FlightStateDrive(config)


create_state_drive = build_flight_state_drive


def state_drive(state: Any = None, *, config: Any = None, **kwargs: Any) -> StateDriveOutput:
    """One-shot convenience function for small runner integrations."""

    return FlightStateDrive(config).step(state, **kwargs)


def state_drive_manifest(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return FlightStateDrive.from_json(path).manifest()


__all__ = [
    "BIOLOGICAL_STATUS", "CALIBRATION_KIND", "CURRENT_UNITS", "DEFAULT_CONFIG",
    "FlightState", "FlightStateContext", "FlightStateDrive", "FlightStateDriveAdapter", "FlightStateInput",
    "FlightStateDriveConfig", "FlightStateDriveError", "FlightStateDriveConfigError", "FlightStateDriveInputError",
    "POISSON_UNITS", "ExternalPopulationInput", "PopulationDriveOutput", "PopulationInput",
    "PopulationRateInput", "PopulationRule", "StateDrive", "StateDriveAdapter",
    "StateDriveConfig", "StateDriveConfigError", "StateDriveError", "StateDriveInput",
    "StateDriveInputError", "StateDriveOutput", "StateDriveRequest", "load_state_drive",
    "build_flight_state_drive", "create_state_drive", "state_drive", "state_drive_manifest",
]


if __name__ == "__main__":  # pragma: no cover - convenience smoke/manifest CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    adapter = FlightStateDrive.from_json(args.config)
    if args.manifest:
        print(json.dumps(adapter.manifest(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(adapter.step({"scenario": "hover", "phase": 0.5}).to_dict(), indent=2))
