#!/usr/bin/env python3
"""Bridge physical dust touch, MaleCNS sensory input, and grooming readout.

The bridge has separate sensory and motor stages.  ``sample_input`` encodes
only regional tactile-load fractions from :class:`GroomingBody` as externally
configured Poisson rates; it never accepts particle positions.  The caller
passes those rates into its selected sensory populations and then supplies
the decoded ``groom_antenna`` readout to ``apply_readout``.

The existing grooming body accepts a toe target, while the mapped
``groom_antenna`` channel is an activation value and contains no movement
phase or direction.  For that reason the motor stage uses an explicitly
engineered local CPG sweep, gated and amplitude-scaled by the real
``groom_antenna`` readout.  It does not claim that the connectome generated
the sweep trajectory.  A zero/missing readout produces no target and a
neutral body step.  The tactile sample is deliberately not an input to the
motor stage, so sensory-off effects must propagate through the neural
simulation and its readout.

JSON bridge configuration schema (all keys are strict):

.. code-block:: json

    {
      "schema_version": 1,
      "selectors_by_region": {"antenna": "<configured sensory selector>"},
      "rate_hz_by_region": {"antenna": 40.0},
      "readout_threshold": 0.0,
      "sweep_extent_mm": 0.12,
      "sweep_frequency_hz": 1.6
    }

Selectors and rate gains are supplied by the experiment configuration; this
module does not nominate or hard-code sensory neuron populations.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:
    from .grooming_body import GroomingBody, GroomingFrame, REGION_NAMES, REGION_PATCHES
except ImportError:  # direct script import with scripts/ on sys.path
    from grooming_body import GroomingBody, GroomingFrame, REGION_NAMES, REGION_PATCHES


_CONFIG_KEYS = frozenset({
    "schema_version", "selectors_by_region", "rate_hz_by_region",
    "readout_threshold", "sweep_extent_mm", "sweep_frequency_hz",
})
_GROOM_READOUT = "groom_antenna"
_GROOM_DN_POPULATIONS = ("DNg62", "DNge078")
_ACTION_SOURCE = (
    "normalized groom_antenna readout mapped from DNg62 and DNge078"
)
_TRAJECTORY_SOURCE = "engineered_cpg_local_antenna_sweep"


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _frozen_float_map(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({str(key): float(value) for key, value in values.items()})


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r} is not allowed")


@dataclass(frozen=True)
class GroomingBridgeConfig:
    """Caller-supplied sensory selectors/rates and explicit motor primitive."""

    selectors_by_region: Mapping[str, str]
    rate_hz_by_region: Mapping[str, float]
    readout_threshold: float = 0.0
    sweep_extent_mm: float = 0.12
    sweep_frequency_hz: float = 1.6

    def __post_init__(self) -> None:
        if not isinstance(self.selectors_by_region, Mapping):
            raise TypeError("selectors_by_region must be a mapping")
        if not isinstance(self.rate_hz_by_region, Mapping):
            raise TypeError("rate_hz_by_region must be a mapping")
        selectors: dict[str, str] = {}
        for raw_region, raw_selector in self.selectors_by_region.items():
            region = str(raw_region)
            if region not in REGION_NAMES:
                raise ValueError(f"unknown sensory region: {region!r}")
            if not isinstance(raw_selector, str) or not raw_selector.strip():
                raise ValueError(f"selector for {region!r} must be a non-empty string")
            selectors[region] = raw_selector.strip()
        if not selectors:
            raise ValueError("at least one sensory selector must be configured")
        if len(set(selectors.values())) != len(selectors):
            raise ValueError("each configured region must have a distinct selector")

        if set(self.rate_hz_by_region) != set(selectors):
            raise ValueError("rate_hz_by_region keys must match selectors_by_region")
        rates: dict[str, float] = {}
        for region, raw_rate in self.rate_hz_by_region.items():
            rate = _finite_number(raw_rate, f"rate_hz_by_region[{region!r}]")
            if rate < 0.0:
                raise ValueError(f"rate for {region!r} must be nonnegative")
            rates[str(region)] = rate

        threshold = _finite_number(self.readout_threshold, "readout_threshold")
        extent = _finite_number(self.sweep_extent_mm, "sweep_extent_mm")
        frequency = _finite_number(self.sweep_frequency_hz, "sweep_frequency_hz")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("readout_threshold must be in [0, 1]")
        if extent <= 0.0 or frequency <= 0.0:
            raise ValueError("sweep extent and frequency must be positive")

        object.__setattr__(self, "selectors_by_region", MappingProxyType(selectors))
        object.__setattr__(self, "rate_hz_by_region", _frozen_float_map(rates))
        object.__setattr__(self, "readout_threshold", threshold)
        object.__setattr__(self, "sweep_extent_mm", extent)
        object.__setattr__(self, "sweep_frequency_hz", frequency)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GroomingBridgeConfig":
        if not isinstance(raw, Mapping):
            raise TypeError("grooming bridge config must be a JSON object")
        unknown = set(raw) - _CONFIG_KEYS
        missing = _CONFIG_KEYS - set(raw)
        if unknown or missing:
            pieces = []
            if unknown:
                pieces.append(f"unknown keys: {sorted(unknown)!r}")
            if missing:
                pieces.append(f"missing keys: {sorted(missing)!r}")
            raise ValueError("invalid grooming bridge config (" + "; ".join(pieces) + ")")
        if (isinstance(raw["schema_version"], bool)
                or not isinstance(raw["schema_version"], int)
                or raw["schema_version"] != 1):
            raise ValueError("schema_version must be 1")
        return cls(
            selectors_by_region=raw["selectors_by_region"],
            rate_hz_by_region=raw["rate_hz_by_region"],
            readout_threshold=raw["readout_threshold"],
            sweep_extent_mm=raw["sweep_extent_mm"],
            sweep_frequency_hz=raw["sweep_frequency_hz"],
        )


def load_config(path: str | Path) -> GroomingBridgeConfig:
    """Load a strict, duplicate-key rejecting JSON bridge configuration."""

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read grooming bridge config {source}: {exc}") from exc
    return GroomingBridgeConfig.from_mapping(raw)


@dataclass(frozen=True)
class GroomingInputSample:
    """Auditable regional touch encoding passed to the neural input boundary."""

    dt_s: float
    raw_dust_loads: Mapping[str, float]
    observed_dust_loads: Mapping[str, float]
    rate_hz_by_region: Mapping[str, float]
    poisson_hz: Mapping[str, float]
    selectors_by_region: Mapping[str, str]
    sensor_enabled: bool
    sensory_source: str = "GroomingBody.tactile_loads"
    activity_source: str = "engineered_rate_encoding_from_physical_dust_touch"

    @property
    def sensory_channels(self) -> Mapping[str, Mapping[str, float]]:
        """Return the conventional nested shape for the MaleCNS input caller."""

        return MappingProxyType({"poisson_hz": self.poisson_hz})

    @property
    def raw_tactile_loads(self) -> Mapping[str, float]:
        return self.raw_dust_loads

    @property
    def observed_tactile_loads(self) -> Mapping[str, float]:
        return self.observed_dust_loads

    def to_dict(self) -> dict[str, Any]:
        return {
            "dt_s": self.dt_s,
            "raw_dust_loads": dict(self.raw_dust_loads),
            "observed_dust_loads": dict(self.observed_dust_loads),
            "rate_hz_by_region": dict(self.rate_hz_by_region),
            "poisson_hz": dict(self.poisson_hz),
            "selectors_by_region": dict(self.selectors_by_region),
            "sensor_enabled": self.sensor_enabled,
            "sensory_source": self.sensory_source,
            "activity_source": self.activity_source,
        }


@dataclass(frozen=True)
class GroomingActionFrame:
    """One neural-readout-driven action and the resulting physical-body frame."""

    dt_s: float
    body_frame: GroomingFrame
    groom_antenna_readout: float
    readout_present: bool
    readout_source: str
    cpg_phase_s: float
    targets_mm: Mapping[str, tuple[float, float, float]]
    locked_legs: frozenset[str]
    action_source: str = _ACTION_SOURCE
    trajectory_source: str = _TRAJECTORY_SOURCE
    action_classification: str = "neural_readout_gated_engineered_motor_primitive"
    readout_channel: str = _GROOM_READOUT
    readout_populations: tuple[str, ...] = _GROOM_DN_POPULATIONS

    @property
    def cleaned(self) -> tuple[tuple[str, int], ...]:
        return self.body_frame.cleaned

    @property
    def remaining(self) -> Mapping[str, int]:
        return self.body_frame.remaining

    @property
    def joint_angles(self) -> Mapping[str, tuple[float, float]]:
        return self.body_frame.joint_angles

    @property
    def foot_positions_mm(self) -> Mapping[str, tuple[float, float, float]]:
        return self.body_frame.foot_positions_mm

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.body_frame.time_s,
            "dt_s": self.dt_s,
            "groom_antenna_readout": self.groom_antenna_readout,
            "readout_present": self.readout_present,
            "readout_source": self.readout_source,
            "readout_channel": self.readout_channel,
            "readout_populations": list(self.readout_populations),
            "action_source": self.action_source,
            "trajectory_source": self.trajectory_source,
            "action_classification": self.action_classification,
            "cpg_phase_s": self.cpg_phase_s,
            "targets_mm": {key: list(value) for key, value in self.targets_mm.items()},
            "locked_legs": sorted(self.locked_legs),
            "joint_angles": {key: list(value) for key, value in self.joint_angles.items()},
            "foot_positions_mm": {key: list(value) for key, value in self.foot_positions_mm.items()},
            "cleaned": [list(item) for item in self.cleaned],
            "remaining": dict(self.remaining),
        }


@dataclass(frozen=True)
class GroomingBridgeAudit:
    """Combined sensory and action trace for one body/neural tick."""

    sensory: GroomingInputSample
    action: GroomingActionFrame

    def to_dict(self) -> dict[str, Any]:
        sensory_data = self.sensory.to_dict()
        action_data = self.action.to_dict()
        return {
            "time_s": action_data["time_s"],
            "raw_dust_loads": sensory_data["raw_dust_loads"],
            "observed_dust_loads": sensory_data["observed_dust_loads"],
            "sensory_rate_hz_by_region": sensory_data["rate_hz_by_region"],
            "sensory_poisson_hz": sensory_data["poisson_hz"],
            "sensory_source": sensory_data["sensory_source"],
            "activity_source": sensory_data["activity_source"],
            "sensor_enabled": sensory_data["sensor_enabled"],
            **action_data,
        }


class GroomingNeuralBridge:
    """Encode physical touch, then map real grooming readout to a local sweep."""

    def __init__(self, config: GroomingBridgeConfig) -> None:
        if not isinstance(config, GroomingBridgeConfig):
            raise TypeError("config must be a GroomingBridgeConfig")
        self.config = config
        self._cpg_phase_s = 0.0

    def sample_input(
        self,
        tactile_loads: Mapping[str, float],
        dt_s: float,
        *,
        sensor_enabled: bool = True,
    ) -> GroomingInputSample:
        """Encode regional tactile load as configured neural input rates.

        ``tactile_loads`` contains only one normalized aggregate value per
        body region.  It cannot carry particle identities or coordinates.
        Sensor masking replaces observed loads and all emitted rates with zero
        while preserving the unmasked physical loads for the audit record.
        """

        step = _finite_number(dt_s, "dt_s")
        if step <= 0.0:
            raise ValueError("dt_s must be positive")
        if not isinstance(sensor_enabled, bool):
            raise TypeError("sensor_enabled must be a bool")
        if not isinstance(tactile_loads, Mapping):
            raise TypeError("tactile_loads must be a mapping")
        if set(tactile_loads) != set(REGION_NAMES):
            raise ValueError("tactile_loads must contain exactly the known regions")
        raw: dict[str, float] = {}
        observed: dict[str, float] = {}
        for region in REGION_NAMES:
            load = _finite_number(tactile_loads[region], f"tactile_loads[{region!r}]")
            if not 0.0 <= load <= 1.0:
                raise ValueError(f"tactile load for {region!r} must be in [0, 1]")
            raw[region] = load
            observed[region] = load if sensor_enabled else 0.0
        region_rates = {
            region: (observed[region] * self.config.rate_hz_by_region[region]
                     if sensor_enabled and region in self.config.selectors_by_region else 0.0)
            for region in REGION_NAMES
        }
        rates = {
            self.config.selectors_by_region[region]: region_rates[region]
            for region in self.config.selectors_by_region
        }
        return GroomingInputSample(
            dt_s=step,
            raw_dust_loads=MappingProxyType(raw),
            observed_dust_loads=MappingProxyType(observed),
            rate_hz_by_region=MappingProxyType(region_rates),
            poisson_hz=MappingProxyType(rates),
            selectors_by_region=self.config.selectors_by_region,
            sensor_enabled=sensor_enabled,
        )

    def apply_readout(
        self,
        body: GroomingBody,
        dt_s: float,
        readout: Any,
        *,
        locked_legs: Set[str] = frozenset(),
    ) -> GroomingActionFrame:
        """Apply the decoded grooming readout to the articulated body.

        This stage consumes no tactile load or particle state.  It uses the
        normalized ``groom_antenna`` channel as an activation/amplitude input
        to the explicitly engineered CPG primitive.  Mechanical leg locks
        suppress the associated body motion independently of the readout.
        """

        if not isinstance(body, GroomingBody):
            raise TypeError("body must be a GroomingBody")
        step = _finite_number(dt_s, "dt_s")
        if step <= 0.0:
            raise ValueError("dt_s must be positive")
        disabled = _validate_locked_legs(locked_legs)
        value, present, source = _groom_readout(readout)

        active = value > self.config.readout_threshold and "fore_left" not in disabled
        targets: dict[str, tuple[float, float, float]] = {}
        if active:
            self._cpg_phase_s = (self._cpg_phase_s + step) % (1.0 / self.config.sweep_frequency_hz)
            patch = REGION_PATCHES["antenna"]
            offset_mm = (
                self.config.sweep_extent_mm * value
                * math.sin(2.0 * math.pi * self.config.sweep_frequency_hz * self._cpg_phase_s)
            )
            target = tuple(
                patch.center_mm[index] + offset_mm * patch.sweep_axis[index]
                for index in range(3)
            )
            targets["fore_left"] = target  # type: ignore[assignment]
        else:
            # Inactivity is a hold. Resetting phase makes the next activation
            # start at the center of the local patch without issuing motion
            # while the channel is zero.
            self._cpg_phase_s = 0.0

        body_frame = body.step(step, targets, disabled_legs=frozenset(disabled))
        return GroomingActionFrame(
            dt_s=step,
            body_frame=body_frame,
            groom_antenna_readout=value,
            readout_present=present,
            readout_source=source,
            cpg_phase_s=self._cpg_phase_s,
            targets_mm=MappingProxyType(targets),
            locked_legs=frozenset(disabled),
        )

    @staticmethod
    def audit(
        sample: GroomingInputSample,
        action: GroomingActionFrame,
    ) -> GroomingBridgeAudit:
        """Join sensory and motor traces without feeding touch to the action."""

        if not isinstance(sample, GroomingInputSample):
            raise TypeError("sample must be a GroomingInputSample")
        if not isinstance(action, GroomingActionFrame):
            raise TypeError("action must be a GroomingActionFrame")
        if not math.isclose(sample.dt_s, action.dt_s, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("sample and action durations must match")
        return GroomingBridgeAudit(sensory=sample, action=action)


def _validate_locked_legs(locked_legs: Set[str]) -> frozenset[str]:
    if isinstance(locked_legs, (str, bytes)) or not isinstance(locked_legs, (set, frozenset)):
        raise TypeError("locked_legs must be a set or frozenset of leg names")
    disabled = frozenset(locked_legs)
    if disabled - set(("fore_left", "fore_right")):
        raise ValueError(f"unknown locked leg(s): {sorted(disabled - {'fore_left', 'fore_right'})!r}")
    return disabled


def _groom_readout(readout: Any) -> tuple[float, bool, str]:
    """Read one normalized channel from adapter output, mapping, or channels."""

    source = "unspecified_readout_source"
    raw: Any = None
    present = False
    if isinstance(readout, Mapping):
        source = str(readout.get("source", source))
        channels = readout.get("channels")
        if isinstance(channels, Mapping):
            present = _GROOM_READOUT in channels
            raw = channels.get(_GROOM_READOUT)
        elif _GROOM_READOUT in readout:
            present = True
            raw = readout[_GROOM_READOUT]
    elif hasattr(readout, "channel") and callable(readout.channel):
        source = str(getattr(readout, "source", source))
        channels = getattr(readout, "channels", None)
        present = isinstance(channels, Mapping) and _GROOM_READOUT in channels
        if present:
            raw = channels[_GROOM_READOUT]
        else:
            try:
                raw = readout.channel(_GROOM_READOUT, 0.0)
            except (KeyError, TypeError, ValueError):
                raw = 0.0
    value = 0.0 if raw is None else _finite_number(raw, _GROOM_READOUT)
    if not 0.0 <= value <= 1.0:
        raise ValueError("groom_antenna readout must be in [0, 1]")
    return value, present, source


__all__ = [
    "GroomingActionFrame",
    "GroomingBridgeAudit",
    "GroomingBridgeConfig",
    "GroomingInputSample",
    "GroomingNeuralBridge",
    "load_config",
]
