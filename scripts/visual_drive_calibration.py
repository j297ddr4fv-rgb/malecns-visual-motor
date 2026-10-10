#!/usr/bin/env python3
"""Explicit calibration between fitted FlyVis and MaleCNS drives.

The fitted FlyVis adapter returns a signed *current-like* deviation for the
13,582 T4/T5 targets.  A one-frame GPU smoke happened to be only about
``1e-4 mV/ms``, but the continuous 50-frame raw trace spans approximately
``-0.726`` to ``+0.860 mV/ms`` (maximum per-frame RMS about ``0.218``).
The production calibration therefore uses an identity gain, a finite input
sanity bound, and an explicit output clip.  The old 4096x one-frame fit would
severely saturate this sustained response.

This module is the deliberately small calibration boundary between those two
components.  It does not import FlyVis, Torch, CUDA, or the MaleCNS engine.
It performs a deterministic signed gain, validates the sparse target mapping,
and applies a finite output bound.  It never creates a spike vector.  Optional
tonic state inputs are represented as explicit external current/rate channels
and are disabled by default; a caller must opt in and resolve any population
selector before they can be emitted.

The main runner-facing path is::

    calibration = VisualDriveCalibration.from_json()
    channels = calibration.apply_to_channels(
        {},
        {"values": flyvis_values,
         "target_indices": flyvis_target_indices,
         "units": "current_mV_per_ms"},
    )
    # SensoryFrame(..., channels=channels)
    # MaleCNSEngine consumes channels["flyvis_t4t5"] directly.

``channels["flyvis_t4t5"]`` is an engine-compatible mapping with extra
provenance fields.  Extra fields are intentionally JSON-safe and make it
possible for telemetry/manifests to distinguish fitted visual input from a
hand-built calibration transform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "config" / "visual_drive_calibration.json"

SCHEMA_VERSION = "visual-drive-calibration-1"
CURRENT_UNITS = "current_mV_per_ms"
POISSON_UNITS = "poisson_hz"
SUPPORTED_UNITS = frozenset({CURRENT_UNITS, POISSON_UNITS})
SOURCE_LABELS = frozenset({"connectome", "fitted", "handbuilt"})


class CalibrationError(ValueError):
    """Base class for malformed or unsafe calibration requests."""


class CalibrationConfigError(CalibrationError):
    """Raised when a calibration configuration is incomplete or ambiguous."""


class CalibrationDriveError(CalibrationError):
    """Raised when a FlyVis or tonic drive violates the configured contract."""


class CalibrationDisabledError(CalibrationError):
    """Raised when a caller attempts to use a disabled calibration explicitly."""


def _finite_float(value: Any, name: str) -> float:
    """Parse a real finite number while rejecting bool/text coercions."""

    if isinstance(value, (bool, str, bytes)):
        raise CalibrationConfigError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationConfigError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise CalibrationConfigError(f"{name} must be a finite number")
    return result


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = _finite_float(value, name)
    if result < 0 or (result == 0 and not allow_zero):
        qualifier = ">= 0" if allow_zero else "> 0"
        raise CalibrationConfigError(f"{name} must be {qualifier}")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise CalibrationConfigError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationConfigError(f"{name} must be a non-negative integer") from exc
    try:
        exact = float(value) == float(result)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or result < 0:
        raise CalibrationConfigError(f"{name} must be a non-negative integer")
    return result


def _unit(value: Any, name: str) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_UNITS:
        raise CalibrationConfigError(
            f"{name} must be one of {sorted(SUPPORTED_UNITS)}, got {value!r}"
        )
    return value


def _source(value: Any, name: str) -> str:
    if not isinstance(value, str) or value not in SOURCE_LABELS:
        raise CalibrationConfigError(
            f"{name} must be one of {sorted(SOURCE_LABELS)}, got {value!r}"
        )
    return value


def _vector(value: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray:
    """Return a finite rank-one numeric vector without coercing text."""

    if isinstance(value, (str, bytes)):
        raise CalibrationDriveError(f"{name} must be a rank-one numeric vector")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationDriveError(f"{name} must be a rank-one numeric vector") from exc
    if raw.ndim != 1:
        raise CalibrationDriveError(f"{name} must be rank one, got shape {raw.shape}")
    if raw.size == 0:
        raise CalibrationDriveError(f"{name} must not be empty")
    if raw.dtype.kind in {"b", "U", "S", "O"}:
        raise CalibrationDriveError(f"{name} must contain numeric values")
    try:
        result = np.asarray(raw, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationDriveError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise CalibrationDriveError(f"{name} contains NaN or infinite values")
    return result.copy()


def _indices(value: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise CalibrationDriveError(f"{name} must be an integer vector")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationDriveError(f"{name} must be an integer vector") from exc
    if raw.ndim != 1 or (not allow_empty and raw.size == 0):
        raise CalibrationDriveError(f"{name} must be a non-empty rank-one integer vector")
    if raw.dtype.kind not in {"i", "u"} or raw.dtype.kind == "b":
        raise CalibrationDriveError(f"{name} must have an integer dtype")
    result = np.asarray(raw, dtype=np.int64).copy()
    if np.any(result < 0):
        raise CalibrationDriveError(f"{name} must contain non-negative indices")
    if np.unique(result).size != result.size:
        raise CalibrationDriveError(f"{name} must contain unique indices")
    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class FlightStateContext:
    """Small, model-agnostic state context for optional tonic modulation.

    This is deliberately not a body/controller implementation.  A runner may
    provide the current scenario phase and measured errors; the calibration
    layer only validates and exposes those scalar inputs to configuration.
    """

    scenario: str = "unknown"
    phase: float = 0.0
    altitude_m: float = 0.0
    altitude_error_m: float = 0.0
    heading_error_rad: float = 0.0
    speed_mps: float = 0.0
    vertical_speed_mps: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, str) or not self.scenario.strip():
            raise CalibrationDriveError("state.scenario must be non-empty text")
        for name in (
            "phase", "altitude_m", "altitude_error_m", "heading_error_rad",
            "speed_mps", "vertical_speed_mps",
        ):
            value = _finite_float(getattr(self, name), f"state.{name}")
            if name == "phase" and not 0.0 <= value <= 1.0:
                raise CalibrationDriveError("state.phase must be in [0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "scenario", self.scenario.strip())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FlightStateContext":
        if not isinstance(raw, Mapping):
            raise CalibrationDriveError("state context must be a mapping")
        return cls(
            scenario=raw.get("scenario", raw.get("scenario_name", "unknown")),
            phase=raw.get("phase", raw.get("scenario_phase", 0.0)),
            altitude_m=raw.get("altitude_m", raw.get("altitude", 0.0)),
            altitude_error_m=raw.get("altitude_error_m", raw.get("altitude_error", 0.0)),
            heading_error_rad=raw.get("heading_error_rad", raw.get("heading_error", 0.0)),
            speed_mps=raw.get("speed_mps", raw.get("speed", 0.0)),
            vertical_speed_mps=raw.get("vertical_speed_mps", raw.get("vertical_speed", 0.0)),
        )

    def features(self) -> dict[str, float]:
        return {
            "phase": self.phase,
            "altitude_m": self.altitude_m,
            "altitude_error_m": self.altitude_error_m,
            "heading_error_rad": self.heading_error_rad,
            "speed_mps": self.speed_mps,
            "vertical_speed_mps": self.vertical_speed_mps,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "phase": self.phase,
            "altitude_m": self.altitude_m,
            "altitude_error_m": self.altitude_error_m,
            "heading_error_rad": self.heading_error_rad,
            "speed_mps": self.speed_mps,
            "vertical_speed_mps": self.vertical_speed_mps,
        }


def _coerce_state_context(value: Any) -> Optional[FlightStateContext]:
    if value is None:
        return None
    if isinstance(value, FlightStateContext):
        return value
    if isinstance(value, Mapping):
        return FlightStateContext.from_mapping(value)
    # Keep the API friendly to a runner's lightweight state object while not
    # importing or depending on a particular body implementation.
    fields = (
        "scenario", "phase", "altitude_m", "altitude_error_m",
        "heading_error_rad", "speed_mps", "vertical_speed_mps",
    )
    if any(hasattr(value, field) for field in fields):
        return FlightStateContext(
            scenario=getattr(value, "scenario", getattr(value, "scenario_name", "unknown")),
            phase=getattr(value, "phase", getattr(value, "scenario_phase", 0.0)),
            altitude_m=getattr(value, "altitude_m", getattr(value, "altitude", 0.0)),
            altitude_error_m=getattr(value, "altitude_error_m", getattr(value, "altitude_error", 0.0)),
            heading_error_rad=getattr(value, "heading_error_rad", getattr(value, "heading_error", 0.0)),
            speed_mps=getattr(value, "speed_mps", getattr(value, "speed", 0.0)),
            vertical_speed_mps=getattr(value, "vertical_speed_mps", getattr(value, "vertical_speed", 0.0)),
        )
    raise CalibrationDriveError("state context must be a mapping or object with scalar state fields")


@dataclass(frozen=True)
class TonicInputSpec:
    """One explicitly declared non-spike tonic input.

    A tonic target may be an exact dense-index list or a MaleCNS population
    selector.  Selectors are intentionally unresolved until the caller passes
    the engine's resolver; this prevents a typo from becoming an empty or
    guessed population.
    """

    name: str
    units: str
    value: tuple[float, ...]
    target_indices: Optional[tuple[int, ...]] = None
    population: Optional[str] = None
    enabled: bool = False
    max_abs: float = 1.0
    source: str = "handbuilt"
    biological_status: str = "hypothesis"
    rationale: str = ""
    state_base: Optional[float] = None
    state_coefficients: tuple[tuple[str, float], ...] = ()
    scenario_bias: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> "TonicInputSpec":
        if not isinstance(raw, Mapping):
            raise CalibrationConfigError(f"tonic_state_inputs[{index}] must be an object")
        name = raw.get("name", f"tonic_{index}")
        if not isinstance(name, str) or not name.strip():
            raise CalibrationConfigError(f"tonic_state_inputs[{index}].name must be non-empty")
        units = _unit(raw.get("units", CURRENT_UNITS), f"{name}.units")
        source = _source(raw.get("source", "handbuilt"), f"{name}.source")
        status = raw.get("biological_status", "hypothesis")
        if not isinstance(status, str) or not status.strip():
            raise CalibrationConfigError(f"{name}.biological_status must be non-empty")
        rationale = raw.get("rationale", "")
        if not isinstance(rationale, str):
            raise CalibrationConfigError(f"{name}.rationale must be text")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise CalibrationConfigError(f"{name}.enabled must be bool")
        max_abs = _positive_float(raw.get("max_abs", 1.0), f"{name}.max_abs")

        target = raw.get("target", {})
        if target is None:
            target = {}
        if not isinstance(target, Mapping):
            raise CalibrationConfigError(f"{name}.target must be an object")
        raw_indices = raw.get("target_indices", target.get("indices"))
        population = raw.get("population", target.get("population"))
        target_indices: Optional[tuple[int, ...]] = None
        if raw_indices is not None:
            parsed = _indices(raw_indices, f"{name}.target_indices")
            target_indices = tuple(int(x) for x in parsed)
        if population is not None:
            if not isinstance(population, str) or not population.strip():
                raise CalibrationConfigError(f"{name}.population must be non-empty text")
            population = population.strip()
        if target_indices is not None and population is not None:
            raise CalibrationConfigError(f"{name} must specify target_indices or population, not both")
        if target_indices is None and population is None:
            raise CalibrationConfigError(
                f"{name} must specify explicit target_indices or a population selector"
            )

        state_raw = raw.get("state_modulation", raw.get("state"))
        if state_raw is not None and not isinstance(state_raw, Mapping):
            raise CalibrationConfigError(f"{name}.state_modulation must be an object")
        raw_value = raw.get("value", raw.get("values"))
        if raw_value is None and isinstance(state_raw, Mapping):
            raw_value = state_raw.get("base", state_raw.get("base_value"))
        if raw_value is None:
            raise CalibrationConfigError(f"{name}.value or state_modulation.base is required")
        if isinstance(raw_value, (bool, str, bytes)):
            raise CalibrationConfigError(f"{name}.value must be numeric")
        try:
            value_array = np.asarray(raw_value)
        except (TypeError, ValueError) as exc:
            raise CalibrationConfigError(f"{name}.value must be numeric") from exc
        if value_array.ndim == 0:
            value_tuple = (_finite_float(raw_value, f"{name}.value"),)
        elif value_array.ndim == 1 and value_array.size:
            if value_array.dtype.kind in {"b", "U", "S", "O"}:
                raise CalibrationConfigError(f"{name}.value must contain numeric values")
            value_tuple = tuple(
                _finite_float(item, f"{name}.value[{i}]")
                for i, item in enumerate(value_array.tolist())
            )
        else:
            raise CalibrationConfigError(f"{name}.value must be a scalar or non-empty vector")
        state_base: Optional[float] = None
        state_coefficients: tuple[tuple[str, float], ...] = ()
        scenario_bias: tuple[tuple[str, float], ...] = ()
        if isinstance(state_raw, Mapping):
            if len(value_tuple) != 1:
                raise CalibrationConfigError(f"{name}.state_modulation requires a scalar base value")
            state_base = value_tuple[0]
            raw_coefficients = state_raw.get("coefficients", state_raw.get("features", {}))
            if raw_coefficients is None:
                raw_coefficients = {}
            if not isinstance(raw_coefficients, Mapping):
                raise CalibrationConfigError(f"{name}.state_modulation.coefficients must be an object")
            allowed_features = {
                "phase", "altitude_m", "altitude_error_m", "heading_error_rad",
                "speed_mps", "vertical_speed_mps",
            }
            parsed_coefficients: list[tuple[str, float]] = []
            for feature, coefficient in raw_coefficients.items():
                if feature not in allowed_features:
                    raise CalibrationConfigError(
                        f"{name}.state_modulation has unsupported feature {feature!r}"
                    )
                parsed_coefficients.append((str(feature), _finite_float(
                    coefficient, f"{name}.state_modulation.coefficients[{feature!r}]"
                )))
            state_coefficients = tuple(sorted(parsed_coefficients))
            raw_scenario_bias = state_raw.get("scenario_bias", {})
            if raw_scenario_bias is None:
                raw_scenario_bias = {}
            if not isinstance(raw_scenario_bias, Mapping):
                raise CalibrationConfigError(f"{name}.state_modulation.scenario_bias must be an object")
            parsed_bias: list[tuple[str, float]] = []
            for scenario, bias in raw_scenario_bias.items():
                if not isinstance(scenario, str) or not scenario.strip():
                    raise CalibrationConfigError(f"{name}.state_modulation scenario names must be text")
                parsed_bias.append((scenario.strip(), _finite_float(
                    bias, f"{name}.state_modulation.scenario_bias[{scenario!r}]"
                )))
            scenario_bias = tuple(sorted(parsed_bias))
        if units == POISSON_UNITS and any(item < 0 for item in value_tuple):
            raise CalibrationConfigError(f"{name}.value cannot be negative in poisson_hz")
        if max(abs(item) for item in value_tuple) > max_abs + 1.0e-12:
            raise CalibrationConfigError(f"{name}.value exceeds configured {name}.max_abs")
        if target_indices is not None and len(value_tuple) not in {1, len(target_indices)}:
            raise CalibrationConfigError(
                f"{name}.value length must be 1 or match target_indices ({len(target_indices)})"
            )
        return cls(
            name=name.strip(), units=units, value=value_tuple,
            target_indices=target_indices, population=population,
            enabled=enabled, max_abs=max_abs, source=source,
            biological_status=status.strip(), rationale=rationale,
            state_base=state_base, state_coefficients=state_coefficients,
            scenario_bias=scenario_bias,
        )

    @property
    def requires_state_context(self) -> bool:
        return self.state_base is not None and bool(self.state_coefficients or self.scenario_bias)

    def values_for_state(self, state: Optional[FlightStateContext]) -> tuple[float, ...]:
        if self.state_base is None:
            return self.value
        if state is None and self.requires_state_context:
            raise CalibrationDriveError(
                f"tonic input {self.name!r} has state modulation; state_context is required"
            )
        if state is None:
            return self.value
        scalar = self.state_base
        bias = dict(self.scenario_bias)
        scalar += bias.get(state.scenario, 0.0)
        features = state.features()
        scalar += sum(coefficient * features[feature] for feature, coefficient in self.state_coefficients)
        return (float(scalar),)

    def to_dict(self) -> dict[str, Any]:
        target: dict[str, Any]
        if self.target_indices is not None:
            target = {"indices": list(self.target_indices)}
        else:
            target = {"population": self.population}
        value: Any = self.value[0] if len(self.value) == 1 else list(self.value)
        result = {
            "name": self.name,
            "enabled": self.enabled,
            "target": target,
            "units": self.units,
            "value": value,
            "max_abs": self.max_abs,
            "source": self.source,
            "biological_status": self.biological_status,
            "rationale": self.rationale,
        }
        if self.state_base is not None:
            result["state_modulation"] = {
                "base": self.state_base,
                "coefficients": {key: value for key, value in self.state_coefficients},
                "scenario_bias": {key: value for key, value in self.scenario_bias},
            }
        return result


@dataclass(frozen=True)
class CalibrationConfig:
    """Validated, immutable calibration settings."""

    name: str
    enabled: bool
    input_field: str
    input_units: str
    input_source: str
    expected_target_count: Optional[int]
    input_max_abs: float
    gain: float
    max_gain: float
    output_max_abs: float
    saturation_policy: str
    tonic_enabled: bool
    tonic_specs: tuple[TonicInputSpec, ...]
    provenance: Mapping[str, Any]
    config_path: Optional[str] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, config_path: Optional[str] = None) -> "CalibrationConfig":
        if not isinstance(raw, Mapping):
            raise CalibrationConfigError("calibration config must be an object")
        schema = raw.get("schema_version", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            raise CalibrationConfigError(
                f"unsupported calibration schema_version {schema!r}; expected {SCHEMA_VERSION!r}"
            )
        name = raw.get("name", "visual_drive_calibration")
        if not isinstance(name, str) or not name.strip():
            raise CalibrationConfigError("config.name must be non-empty text")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise CalibrationConfigError("config.enabled must be bool")

        input_raw = raw.get("input", {})
        transform_raw = raw.get("transform", {})
        tonic_raw = raw.get("tonic_state_inputs", {})
        if not isinstance(input_raw, Mapping) or not isinstance(transform_raw, Mapping):
            raise CalibrationConfigError("input and transform must be objects")
        if not isinstance(tonic_raw, Mapping):
            raise CalibrationConfigError("tonic_state_inputs must be an object")

        input_field = input_raw.get("field", "flyvis_t4t5")
        if not isinstance(input_field, str) or not input_field.strip():
            raise CalibrationConfigError("input.field must be non-empty text")
        input_units = _unit(input_raw.get("units", CURRENT_UNITS), "input.units")
        input_source = _source(input_raw.get("source", "fitted"), "input.source")
        expected_raw = input_raw.get("expected_target_count")
        expected = None if expected_raw is None else _nonnegative_int(expected_raw, "input.expected_target_count")
        if expected == 0:
            raise CalibrationConfigError("input.expected_target_count must be > 0 when provided")
        input_max_abs = _positive_float(
            input_raw.get("max_abs", input_raw.get("max_abs_current_mV_per_ms", 0.25)),
            "input.max_abs",
        )

        gain = _finite_float(transform_raw.get("gain", raw.get("gain", 1.0)), "transform.gain")
        if gain <= 0:
            raise CalibrationConfigError("transform.gain must be > 0")
        max_gain = _positive_float(transform_raw.get("max_gain", 100000.0), "transform.max_gain")
        if gain > max_gain:
            raise CalibrationConfigError("transform.gain exceeds transform.max_gain")
        output_max_abs = _positive_float(
            transform_raw.get(
                "max_abs_output",
                transform_raw.get("max_abs_current_mV_per_ms", 1.0),
            ),
            "transform.max_abs_output",
        )
        saturation_policy = transform_raw.get("saturation_policy", "clip")
        if saturation_policy not in {"clip", "error"}:
            raise CalibrationConfigError("transform.saturation_policy must be 'clip' or 'error'")

        tonic_enabled = tonic_raw.get("enabled", False)
        if not isinstance(tonic_enabled, bool):
            raise CalibrationConfigError("tonic_state_inputs.enabled must be bool")
        entries = tonic_raw.get("entries", tonic_raw.get("inputs", []))
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise CalibrationConfigError("tonic_state_inputs.entries must be an array")
        tonic_specs = tuple(
            TonicInputSpec.from_mapping(entry, index=i) for i, entry in enumerate(entries)
        )
        provenance = raw.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise CalibrationConfigError("config.provenance must be an object")
        return cls(
            name=name.strip(), enabled=enabled, input_field=input_field.strip(),
            input_units=input_units, input_source=input_source,
            expected_target_count=expected, input_max_abs=input_max_abs,
            gain=gain, max_gain=max_gain, output_max_abs=output_max_abs,
            saturation_policy=saturation_policy, tonic_enabled=tonic_enabled,
            tonic_specs=tonic_specs, provenance=dict(provenance), config_path=config_path,
        )

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONFIG) -> "CalibrationConfig":
        config_path = Path(path).expanduser()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CalibrationConfigError(f"cannot read calibration config {config_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CalibrationConfigError(f"invalid calibration JSON {config_path}: {exc}") from exc
        return cls.from_mapping(raw, config_path=str(config_path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "enabled": self.enabled,
            "input": {
                "field": self.input_field,
                "units": self.input_units,
                "source": self.input_source,
                "expected_target_count": self.expected_target_count,
                "max_abs": self.input_max_abs,
            },
            "transform": {
                "kind": "signed_gain_and_bound",
                "gain": self.gain,
                "max_gain": self.max_gain,
                "max_abs_output": self.output_max_abs,
                "saturation_policy": self.saturation_policy,
            },
            "tonic_state_inputs": {
                "enabled": self.tonic_enabled,
                "entries": [spec.to_dict() for spec in self.tonic_specs],
            },
            "provenance": _json_safe(self.provenance),
        }


@dataclass(frozen=True)
class CalibratedVisualDrive(Mapping[str, Any]):
    """One bounded, mapped visual current ready for MaleCNS injection."""

    target_indices: np.ndarray
    drive_values: np.ndarray
    units: str
    target_ids: Optional[np.ndarray]
    gain: float
    output_max_abs: float
    saturated_count: int
    frame_index: Optional[int]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        indices = _indices(self.target_indices, "target_indices")
        values = _vector(self.drive_values, "values")
        if indices.shape != values.shape:
            raise CalibrationDriveError(
                f"target_indices and values must have the same shape, got {indices.shape} and {values.shape}"
            )
        if self.units not in SUPPORTED_UNITS:
            raise CalibrationDriveError(f"unsupported drive units {self.units!r}")
        if self.units == POISSON_UNITS and np.any(values < 0):
            raise CalibrationDriveError("poisson_hz values cannot be negative")
        if self.target_ids is not None:
            ids = _vector(self.target_ids, "target_ids", dtype=np.float64)
            if ids.shape != indices.shape:
                raise CalibrationDriveError("target_ids must match target_indices shape")
        else:
            ids = None
        if int(self.saturated_count) < 0 or int(self.saturated_count) > values.size:
            raise CalibrationDriveError("saturated_count is outside the vector size")
        object.__setattr__(self, "target_indices", indices)
        object.__setattr__(self, "drive_values", values.astype(np.float32, copy=False))
        object.__setattr__(self, "target_ids", ids)

    @property
    def values(self) -> np.ndarray:
        return self.drive_values.copy()

    target_drive = values
    t4t5_drive = values

    def as_engine_mapping(self) -> dict[str, Any]:
        """Return the explicit mapping consumed by ``MaleCNSEngine``."""

        mapping: dict[str, Any] = {
            "target_indices": self.target_indices.copy(),
            "values": self.values.copy(),
            "units": self.units,
            "source": "fitted",
            "provenance": _json_safe(self.provenance),
            "calibrated": True,
        }
        if self.target_ids is not None:
            mapping["target_ids"] = self.target_ids.copy()
        return mapping

    def __getitem__(self, key: str) -> Any:
        return self.as_engine_mapping()[key]

    def __iter__(self):
        return iter(self.as_engine_mapping())

    def __len__(self) -> int:
        return len(self.as_engine_mapping())

    # Compatibility spellings for small runner adapters.  All return a fresh
    # mapping so a downstream engine cannot mutate this immutable result.
    to_engine_mapping = as_engine_mapping

    @property
    def engine_mapping(self) -> dict[str, Any]:
        return self.as_engine_mapping()

    def to_dict(self, *, include_values: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "target_indices": self.target_indices.tolist(),
            "units": self.units,
            "gain": self.gain,
            "output_max_abs": self.output_max_abs,
            "saturated_count": self.saturated_count,
            "frame_index": self.frame_index,
            "provenance": _json_safe(self.provenance),
        }
        if self.target_ids is not None:
            result["target_ids"] = self.target_ids.tolist()
        if include_values:
            result["values"] = self.values.tolist()
        return result


@dataclass(frozen=True)
class TonicDrive:
    """Resolved tonic input represented without any spike fabrication."""

    name: str
    units: str
    values: np.ndarray
    target_indices: np.ndarray
    population: Optional[str]
    source: str
    biological_status: str
    rationale: str

    def __post_init__(self) -> None:
        indices = _indices(self.target_indices, f"{self.name}.target_indices")
        values = _vector(self.values, f"{self.name}.values")
        if indices.shape != values.shape:
            raise CalibrationDriveError(f"{self.name} target/value shape mismatch")
        if self.units not in SUPPORTED_UNITS:
            raise CalibrationDriveError(f"{self.name} uses unsupported units {self.units!r}")
        if self.units == POISSON_UNITS and np.any(values < 0):
            raise CalibrationDriveError(f"{self.name} poisson_hz values cannot be negative")
        object.__setattr__(self, "target_indices", indices)
        object.__setattr__(self, "values", values.astype(np.float32, copy=False))

    def as_mapping(self) -> dict[str, Any]:
        """Return an explicit sparse tonic record for diagnostics/adapters."""

        return {
            "name": self.name,
            "target_indices": self.target_indices.copy(),
            "values": self.values.copy(),
            "units": self.units,
            "population": self.population,
            "source": self.source,
            "biological_status": self.biological_status,
            "rationale": self.rationale,
            "is_spike_drive": False,
        }


def _extract_visual_drive(
    drive: Any,
    *,
    target_indices: Any = None,
    target_ids: Any = None,
    units: Optional[str] = None,
) -> tuple[Any, Any, Any, Any]:
    """Extract value/index/unit fields from adapter outputs without guessing."""

    metadata: Any = None
    if isinstance(drive, Mapping):
        metadata = drive.get("metadata", drive.get("provenance"))
        if "values" in drive:
            values = drive["values"]
        elif "drive" in drive:
            values = drive["drive"]
        elif "t4t5_drive" in drive:
            values = drive["t4t5_drive"]
        else:
            raise CalibrationDriveError("visual mapping must contain values, drive, or t4t5_drive")
        if target_indices is None:
            target_indices = drive.get("target_indices", drive.get("flyvis_target_indices"))
        if target_ids is None:
            target_ids = drive.get("target_ids", drive.get("flyvis_target_ids"))
        if units is None:
            units = drive.get("units", drive.get("flyvis_units"))
        return values, target_indices, units, metadata
    for name in ("values", "drive", "target_drive", "t4t5_drive"):
        if hasattr(drive, name):
            values = getattr(drive, name)
            break
    else:
        values = drive
    if target_indices is None:
        for name in ("target_indices", "target_drive_indices", "flyvis_target_indices"):
            candidate = getattr(drive, name, None)
            if candidate is not None:
                target_indices = candidate
                break
    if target_ids is None:
        candidate = getattr(drive, "target_ids", None)
        if candidate is not None:
            target_ids = candidate
    if units is None:
        units = getattr(drive, "units", getattr(drive, "target_units", None))
    metadata = getattr(drive, "provenance", getattr(drive, "metadata", None))
    return values, target_indices, units, metadata


class VisualDriveCalibration:
    """Deterministic, bounded FlyVis-to-MaleCNS calibration layer."""

    def __init__(self, config: CalibrationConfig | Mapping[str, Any] | str | Path = DEFAULT_CONFIG):
        if isinstance(config, CalibrationConfig):
            self.config = config
        elif isinstance(config, Mapping):
            self.config = CalibrationConfig.from_mapping(config)
        else:
            self.config = CalibrationConfig.from_json(config)

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONFIG) -> "VisualDriveCalibration":
        return cls(CalibrationConfig.from_json(path))

    @classmethod
    def from_config(cls, config: CalibrationConfig | Mapping[str, Any]) -> "VisualDriveCalibration":
        return cls(config)

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise CalibrationDisabledError(
                f"visual drive calibration {self.config.name!r} is disabled; refusing raw injection"
            )

    def with_tonic_state_inputs(self, enabled: bool = True) -> "VisualDriveCalibration":
        """Return a copy with the explicitly declared tonic block toggled.

        This is an opt-in convenience for a runner flag.  Individual entry
        ``enabled`` fields remain authoritative, so callers cannot activate an
        undeclared input by accident.
        """

        if not isinstance(enabled, bool):
            raise CalibrationDriveError("tonic enabled flag must be bool")
        raw = self.config.to_dict()
        raw["tonic_state_inputs"]["enabled"] = enabled
        return VisualDriveCalibration(raw)

    enable_tonic_state_inputs = with_tonic_state_inputs

    def calibrate(
        self,
        drive: Any,
        target_indices: Any = None,
        *,
        target_ids: Any = None,
        units: Optional[str] = None,
        frame_index: Optional[int] = None,
        state_context: Any = None,
        state: Any = None,
        context: Any = None,
    ) -> CalibratedVisualDrive:
        """Scale and bound one signed FlyVis output.

        ``target_indices`` and ``units`` are mandatory at the boundary unless
        the supplied adapter object/mapping exposes them.  In particular, a
        vector is never assumed to be dense MaleCNS order.
        """

        self._require_enabled()
        supplied_contexts = [item is not None for item in (state_context, state, context)]
        if sum(supplied_contexts) > 1:
            raise CalibrationDriveError("provide at most one of state_context, state, or context")
        # Visual gain is frame-local and does not depend on flight state.  We
        # still accept/validate the context here so a runner can pass one
        # common context to both visual and tonic calibration calls without a
        # signature-specific branch.
        context = _coerce_state_context(
            state_context if state_context is not None
            else state if state is not None else context
        )
        values_raw, indices_raw, units_raw, metadata = _extract_visual_drive(
            drive, target_indices=target_indices, target_ids=target_ids, units=units,
        )
        if units_raw is None:
            raise CalibrationDriveError("FlyVis units are required; refusing unit inference")
        actual_units = _unit(units_raw, "FlyVis units")
        if actual_units != self.config.input_units:
            raise CalibrationDriveError(
                f"FlyVis units {actual_units!r} do not match configured {self.config.input_units!r}"
            )
        values = _vector(values_raw, "FlyVis drive")
        if indices_raw is None:
            raise CalibrationDriveError(
                "FlyVis target_indices are required; refusing dense-order or positional inference"
            )
        indices = _indices(indices_raw, "FlyVis target_indices")
        if values.shape != indices.shape:
            raise CalibrationDriveError(
                f"FlyVis values and target_indices must match, got {values.shape} and {indices.shape}"
            )
        if self.config.expected_target_count is not None and values.size != self.config.expected_target_count:
            raise CalibrationDriveError(
                f"FlyVis target count {values.size} != configured {self.config.expected_target_count}"
            )
        if np.any(np.abs(values) > self.config.input_max_abs + 1.0e-12):
            observed = float(np.max(np.abs(values)))
            raise CalibrationDriveError(
                f"FlyVis input magnitude {observed:g} exceeds configured input_max_abs "
                f"{self.config.input_max_abs:g} {actual_units}"
            )
        if target_ids is not None:
            ids = _vector(target_ids, "FlyVis target_ids")
            if ids.shape != indices.shape:
                raise CalibrationDriveError("FlyVis target_ids must match target_indices")
        else:
            ids = None
        if frame_index is not None:
            if isinstance(frame_index, bool) or int(frame_index) != frame_index or int(frame_index) < 0:
                raise CalibrationDriveError("frame_index must be a non-negative integer")
            frame_index = int(frame_index)

        calibrated = values * self.config.gain
        saturated = np.abs(calibrated) > self.config.output_max_abs
        saturated_count = int(np.count_nonzero(saturated))
        if saturated_count and self.config.saturation_policy == "error":
            observed = float(np.max(np.abs(calibrated)))
            raise CalibrationDriveError(
                f"calibrated FlyVis drive magnitude {observed:g} exceeds output bound "
                f"{self.config.output_max_abs:g} {actual_units}"
            )
        if saturated_count:
            calibrated = np.clip(calibrated, -self.config.output_max_abs, self.config.output_max_abs)
        if actual_units == POISSON_UNITS and np.any(calibrated < 0):
            # Signed visual current is the intended default.  This branch
            # protects any future rate configuration from turning a negative
            # fitted deviation into a nonsensical Poisson rate.
            raise CalibrationDriveError("poisson_hz calibration cannot preserve negative values")
        if not np.all(np.isfinite(calibrated)):
            raise CalibrationDriveError("calibrated FlyVis drive contains NaN or infinite values")

        provenance = {
            "label": "fitted_flyvis -> handbuilt_bounded_gain -> connectome_injection",
            "input_source": self.config.input_source,
            "input_field": self.config.input_field,
            "input_units": actual_units,
            "transform_source": "handbuilt",
            "transform": "signed_gain_and_bound",
            "gain": self.config.gain,
            "output_max_abs": self.config.output_max_abs,
            "saturation_policy": self.config.saturation_policy,
            "saturated_count": saturated_count,
            "target_count": int(indices.size),
            "frame_index": frame_index,
            "state_context": None if context is None else context.to_dict(),
            "adapter_metadata": _json_safe(metadata),
        }
        return CalibratedVisualDrive(
            target_indices=indices, drive_values=np.asarray(calibrated, dtype=np.float32),
            units=actual_units, target_ids=ids, gain=self.config.gain,
            output_max_abs=self.config.output_max_abs, saturated_count=saturated_count,
            frame_index=frame_index, provenance=provenance,
        )

    def _resolve_tonic_indices(
        self,
        spec: TonicInputSpec,
        *,
        population_resolver: Optional[Callable[[str], Any]],
        n_neurons: Optional[int],
    ) -> np.ndarray:
        if spec.target_indices is not None:
            indices = _indices(spec.target_indices, f"{spec.name}.target_indices")
        else:
            if population_resolver is None:
                raise CalibrationDriveError(
                    f"enabled tonic input {spec.name!r} uses population {spec.population!r}; "
                    "a population_resolver is required"
                )
            try:
                resolved = population_resolver(str(spec.population))
            except Exception as exc:
                raise CalibrationDriveError(
                    f"population resolver failed for tonic input {spec.name!r}: {exc}"
                ) from exc
            indices = _indices(resolved, f"resolved {spec.name}.target_indices")
        if n_neurons is not None:
            if isinstance(n_neurons, bool) or int(n_neurons) != n_neurons or int(n_neurons) <= 0:
                raise CalibrationDriveError("n_neurons must be a positive integer")
            if np.any(indices >= int(n_neurons)):
                raise CalibrationDriveError(
                    f"tonic input {spec.name!r} contains an index outside n_neurons={int(n_neurons)}"
                )
        return indices

    def resolve_tonic_inputs(
        self,
        *,
        population_resolver: Optional[Callable[[str], Any]] = None,
        n_neurons: Optional[int] = None,
        state_context: Any = None,
        state: Any = None,
        include_disabled: bool = False,
    ) -> tuple[TonicDrive, ...]:
        """Resolve explicitly enabled tonic inputs; never emits spikes.

        ``include_disabled`` is retained as a diagnostics/API compatibility
        flag; a globally disabled tonic block always returns no drives.  It
        can never bypass the explicit global opt-in.
        """

        if state_context is not None and state is not None:
            raise CalibrationDriveError("provide at most one of state_context and state")
        context = _coerce_state_context(state_context if state_context is not None else state)
        if not self.config.tonic_enabled:
            return ()
        result: list[TonicDrive] = []
        used: dict[str, set[int]] = {CURRENT_UNITS: set(), POISSON_UNITS: set()}
        for spec in self.config.tonic_specs:
            if not spec.enabled:
                continue
            indices = self._resolve_tonic_indices(
                spec, population_resolver=population_resolver, n_neurons=n_neurons,
            )
            if any(int(index) in used[spec.units] for index in indices):
                raise CalibrationDriveError(
                    f"tonic inputs overlap in {spec.units}; refusing order-dependent summation"
                )
            used[spec.units].update(int(index) for index in indices)
            effective_values = spec.values_for_state(context)
            if len(effective_values) == 1:
                values = np.full(indices.size, effective_values[0], dtype=np.float64)
            elif len(effective_values) == indices.size:
                values = np.asarray(effective_values, dtype=np.float64)
            else:
                raise CalibrationDriveError(
                    f"tonic input {spec.name!r} value length must be 1 or {indices.size}"
                )
            if np.any(np.abs(values) > spec.max_abs + 1.0e-12):
                raise CalibrationDriveError(f"tonic input {spec.name!r} exceeds max_abs")
            if spec.units == POISSON_UNITS and np.any(values < 0):
                raise CalibrationDriveError(f"tonic input {spec.name!r} poisson_hz cannot be negative")
            result.append(TonicDrive(
                name=spec.name, units=spec.units, values=values,
                target_indices=indices, population=spec.population,
                source=spec.source, biological_status=spec.biological_status,
                rationale=spec.rationale,
            ))
        return tuple(result)

    def _tonic_channels(
        self,
        tonic: Sequence[TonicDrive],
        *,
        n_neurons: Optional[int],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Build MaleCNS-compatible tonic channels and provenance records."""

        channels: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        dense_by_units: dict[str, np.ndarray] = {}
        population_by_units: dict[str, dict[str, Any]] = {
            CURRENT_UNITS: {}, POISSON_UNITS: {},
        }
        for item in tonic:
            if item.population is not None:
                # MaleCNSEngine's population merge API is the only safe sparse
                # population path.  A population selector is retained as the
                # key, never converted to an inferred dense index.
                if item.population in population_by_units[item.units]:
                    raise CalibrationDriveError(
                        f"duplicate tonic population {item.population!r} in {item.units}"
                    )
                value: Any
                if np.all(item.values == item.values[0]):
                    value = float(item.values[0])
                else:
                    value = item.values.copy()
                population_by_units[item.units][item.population] = value
            else:
                if n_neurons is None:
                    raise CalibrationDriveError(
                        f"tonic input {item.name!r} uses exact indices; n_neurons is required "
                        "to build a dense MaleCNS channel"
                    )
                if item.units not in dense_by_units:
                    dense_by_units[item.units] = np.zeros(int(n_neurons), dtype=np.float32)
                dense_by_units[item.units][item.target_indices] = item.values
            record = item.as_mapping()
            records.append(record)

        current = population_by_units[CURRENT_UNITS]
        poisson = population_by_units[POISSON_UNITS]
        if current:
            channels["current_mV_per_ms"] = current
        if poisson:
            channels["poisson_hz"] = poisson
        if CURRENT_UNITS in dense_by_units:
            if "current_mV_per_ms" in channels:
                raise CalibrationDriveError("cannot mix dense and population tonic current channels")
            channels["current_mV_per_ms"] = dense_by_units[CURRENT_UNITS]
        if POISSON_UNITS in dense_by_units:
            if "poisson_hz" in channels:
                raise CalibrationDriveError("cannot mix dense and population tonic poisson channels")
            channels["poisson_hz"] = dense_by_units[POISSON_UNITS]
        return channels, records

    def apply_to_channels(
        self,
        channels: Mapping[str, Any],
        drive: Any,
        target_indices: Any = None,
        *,
        target_ids: Any = None,
        units: Optional[str] = None,
        frame_index: Optional[int] = None,
        population_resolver: Optional[Callable[[str], Any]] = None,
        n_neurons: Optional[int] = None,
        state_context: Any = None,
        state: Any = None,
        context: Any = None,
        replace_visual: bool = False,
    ) -> dict[str, Any]:
        """Calibrate FlyVis and merge optional tonic inputs into a frame.

        Existing visual/tonic channels are not silently overwritten unless
        ``replace_visual=True`` is explicitly selected.  The returned mapping
        contains no ``spikes`` field; the MaleCNS engine remains the sole spike
        generator.
        """

        if not isinstance(channels, Mapping):
            raise CalibrationDriveError("channels must be a mapping")
        packet = self.calibrate(
            drive, target_indices, target_ids=target_ids, units=units,
            frame_index=frame_index, state_context=state_context,
            state=state, context=context,
        )
        result = dict(channels)
        if self.config.input_field in result and not replace_visual:
            raise CalibrationDriveError(
                f"channels already contains {self.config.input_field!r}; refusing overwrite"
            )
        result[self.config.input_field] = packet.as_engine_mapping()

        tonic = self.resolve_tonic_inputs(
            population_resolver=population_resolver, n_neurons=n_neurons,
            state_context=(state_context if state_context is not None else
                           state if state is not None else context),
        )
        tonic_channels, records = self._tonic_channels(tonic, n_neurons=n_neurons)
        for key, value in tonic_channels.items():
            if key in result:
                raise CalibrationDriveError(
                    f"channels already contains tonic channel {key!r}; refusing ambiguous merge"
                )
            result[key] = value
        result["visual_drive_calibration"] = {
            "schema_version": SCHEMA_VERSION,
            "manifest_name": self.config.name,
            "visual": packet.to_dict(include_values=False),
            "tonic_state_inputs": records,
            "is_spike_drive": False,
        }
        return result

    # Short aliases make integration with small runner adapters less verbose.
    apply = apply_to_channels
    calibrate_drive = calibrate

    def manifest(self) -> dict[str, Any]:
        """Return a deterministic, JSON-safe run manifest fragment."""

        config_dict = self.config.to_dict()
        canonical = _canonical_json(config_dict)
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.config.name,
            "config_path": self.config.config_path,
            "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "enabled": self.config.enabled,
            "input": {
                "field": self.config.input_field,
                "units": self.config.input_units,
                "source": self.config.input_source,
                "expected_target_count": self.config.expected_target_count,
                "max_abs": self.config.input_max_abs,
            },
            "transform": {
                "kind": "signed_gain_and_bound",
                "gain": self.config.gain,
                "max_gain": self.config.max_gain,
                "output_max_abs": self.config.output_max_abs,
                "saturation_policy": self.config.saturation_policy,
                "preserves_sign": True,
                "deterministic": True,
            },
            "tonic_state_inputs": {
                "enabled": self.config.tonic_enabled,
                "entries": [spec.to_dict() for spec in self.config.tonic_specs],
                "spikes_fabricated": False,
                "accepted_state_context": [
                    "scenario", "phase", "altitude_m", "altitude_error_m",
                    "heading_error_rad", "speed_mps", "vertical_speed_mps",
                ],
            },
            "provenance": {
                "input": self.config.input_source,
                "calibration": "handbuilt",
                "injection": "connectome",
                "chain": "fitted_flyvis -> handbuilt_bounded_gain -> connectome_injection",
                **_json_safe(self.config.provenance),
            },
            "guarantees": [
                "finite numeric input and output",
                "explicit sparse target_indices required",
                "explicit units required and checked",
                "signed visual current is not rectified",
                "bounded output with saturation count",
                "tonic inputs are external currents/rates, never spike arrays",
            ],
        }


# Functional aliases for runner code and small scripts.
DriveCalibration = VisualDriveCalibration
load_calibration = VisualDriveCalibration.from_json


def create_calibrator(
    config: str | Path | Mapping[str, Any] | CalibrationConfig = DEFAULT_CONFIG,
    *,
    config_path: str | Path | None = None,
) -> VisualDriveCalibration:
    """Construct a calibrator for runner integration.

    The factory intentionally performs only configuration validation.  It does
    not import or initialise FlyVis, Torch, CUDA, or a MaleCNS graph.
    """

    if config_path is not None:
        if config != DEFAULT_CONFIG:
            raise CalibrationConfigError("provide at most one of config and config_path")
        config = config_path
    return VisualDriveCalibration(config)


def calibrate_flyvis_drive(
    drive: Any,
    target_indices: Any = None,
    *,
    config: str | Path | Mapping[str, Any] | CalibrationConfig = DEFAULT_CONFIG,
    target_ids: Any = None,
    units: Optional[str] = None,
    frame_index: Optional[int] = None,
    state_context: Any = None,
    state: Any = None,
    context: Any = None,
) -> dict[str, Any]:
    """Functional runner helper returning a MaleCNS engine mapping."""

    return VisualDriveCalibration(config).calibrate(
        drive, target_indices, target_ids=target_ids,
        units=units, frame_index=frame_index,
        state_context=state_context, state=state, context=context,
    ).as_engine_mapping()


def calibration_manifest(
    config: str | Path | Mapping[str, Any] | CalibrationConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    return VisualDriveCalibration(config).manifest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    args = parser.parse_args(argv)
    payload = VisualDriveCalibration.from_json(args.config).manifest()
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())


__all__ = [
    "CURRENT_UNITS", "POISSON_UNITS", "SUPPORTED_UNITS", "SCHEMA_VERSION",
    "CalibrationError", "CalibrationConfigError", "CalibrationDriveError",
    "CalibrationDisabledError", "CalibrationConfig", "FlightStateContext", "TonicInputSpec",
    "CalibratedVisualDrive", "TonicDrive", "VisualDriveCalibration",
    "DriveCalibration", "load_calibration", "create_calibrator", "calibrate_flyvis_drive",
    "calibration_manifest", "DEFAULT_CONFIG",
]
