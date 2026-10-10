#!/usr/bin/env python3
"""Optional low-level body stabilizer for the reduced digital fly.

This module is an intentionally small *hand-built* overlay at the actuator
boundary.  It consumes pre-computed goal/body errors, a mission phase, and the
already fitted neural motor channels.  It returns bounded actuator channels;
it never consumes positions as control state, integrates a body, or writes a
pose/body object.

The neural wing output remains the primary command.  The overlay has authority
only over pitch, roll, and a small equal left/right collective trim.  In
particular, it does not change yaw, the left/right neural steering difference,
or any unknown fitted channel.  The maximum hand-built contribution is an
explicit configuration value and is present in both the output and manifest.

The gains are calibration hypotheses motivated by the low-confidence flight
pitch/altitude-control note in ``docs/research/embodied-precedents.md``.  This
is not a biological controller claim: ``source``/``provenance`` always retain
the fitted neural input and the hand-built overlay as separate records.

Typical use::

    stabilizer = FlightBodyStabilizer.from_json()
    result = stabilizer.step(
        goal_error={"altitude_error_m": 0.08, "forward_error_m": 0.2},
        body_error={"pitch_error_rad": -0.01, "roll_error_rad": 0.0,
                    "vertical_speed_mps": 0.0},
        phase="forward",
        fitted_motor_channels={"left_wing": 0.706, "right_wing": 0.706,
                               "yaw": 0.1, "pitch": 0.0, "roll": 0.0},
        dt_s=0.005,
    )
    dynamics.step(result.channels, dt_s=0.005)

The class is dependency-free and accepts mappings or small dataclass/object
records at the boundary.  It is safe to disable for an ablation by passing
``enabled=False`` or calling :meth:`set_enabled`.
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
    from .flight_control_common import clip as _clip, json_safe as _json_safe, normal_key as _normal_key, number as _number
except ImportError:
    from flight_control_common import clip as _clip, json_safe as _json_safe, normal_key as _normal_key, number as _number


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "config" / "flight_body_stabilizer.json"

SCHEMA_VERSION = "flight-body-stabilizer-1"
HAND_BUILT_SOURCE = "handbuilt"
FITTED_SOURCE = "fitted"
BLENDED_SOURCE = "fitted+handbuilt"
DISABLED_SOURCE = FITTED_SOURCE

_ACTUATORS = (
    "left_wing", "right_wing", "thrust", "yaw", "pitch", "roll", "brake",
)
_SIGNED = frozenset(("yaw", "pitch", "roll"))
_ALIASES: dict[str, tuple[str, ...]] = {
    "left_wing": (
        "left_wing", "left", "wing_left", "motor_left", "l",
        "left_power", "left_wing_power", "wing_drive_left",
    ),
    "right_wing": (
        "right_wing", "right", "wing_right", "motor_right", "r",
        "right_power", "right_wing_power", "wing_drive_right",
    ),
    "thrust": ("thrust", "power", "lift", "wing_power", "flight_power"),
    "yaw": ("yaw", "turn", "heading", "wing_steering", "flight_steering"),
    "pitch": ("pitch", "pitch_control", "pitch_cmd"),
    "roll": ("roll", "roll_control", "roll_cmd"),
    "brake": ("brake", "drag"),
}

_PHASE_ALIASES = {
    "ground": "grounded",
    "grounded": "grounded",
    "idle": "grounded",
    "takeoff": "takeoff",
    "launch": "takeoff",
    "forward": "forward",
    "flight": "forward",
    "cruise": "forward",
    "translation": "forward",
    "turn": "turn",
    "yaw": "turn",
    "hover": "hover",
    "gust": "gust-recovery",
    "gust_recovery": "gust-recovery",
    "gust-recovery": "gust-recovery",
    "approach": "approach",
    "landing": "landing",
    "land": "landing",
    "settle": "settle",
}


class StabilizerError(ValueError):
    """Base exception for malformed stabilizer configuration/input."""


class StabilizerConfigError(StabilizerError):
    """Raised for an unsafe or incomplete configuration."""


class StabilizerInputError(StabilizerError):
    """Raised for a malformed high-level input shape."""


def _finite(value: Any, name: str, *, default: Optional[float] = None) -> tuple[float, bool]:
    """Return a finite scalar and whether the input was valid.

    Invalid numeric samples are fail-closed to ``default``.  The result keeps
    the command bounded while ``input_finite``/provenance make the event
    visible to a caller; no NaN is ever emitted to the body simulator.
    """

    if isinstance(value, bool):
        return float(value), False
    result, valid = _number(value)
    if not valid:
        result = 0.0 if default is None else float(default)
    return result, valid


def _required_finite(value: Any, name: str) -> float:
    result, okay = _finite(value, name)
    if not okay:
        raise StabilizerConfigError(f"{name} must be a finite scalar")
    return result


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dataclass_fields = getattr(value, "__dataclass_fields__", None)
    if dataclass_fields:
        return {str(field_name): getattr(value, field_name) for field_name in dataclass_fields}
    names = (
        "altitude_error_m", "altitude_error", "vertical_error_m", "z_error_m",
        "forward_error_m", "forward_error", "longitudinal_error_m",
        "forward_velocity_mps", "forward_velocity", "vx_mps",
        "lateral_error_m", "lateral_error", "cross_track_error_m",
        "lateral_velocity_mps", "lateral_velocity", "vy_mps",
        "heading_error_rad", "heading_error", "yaw_error_rad", "yaw_error",
        "pitch_error_rad", "pitch_error", "roll_error_rad", "roll_error",
        "vertical_speed_mps", "vertical_speed", "vz_mps",
        "pitch_rate_rad_s", "pitch_rate", "roll_rate_rad_s", "roll_rate",
        "yaw_rate_rad_s", "yaw_rate", "speed_error_mps", "speed_error",
    )
    result = {field_name: getattr(value, field_name) for field_name in names
              if hasattr(value, field_name)}
    if not result:
        raise StabilizerInputError(
            f"{name} must be a mapping or expose precomputed error fields"
        )
    return result


def _flatten_error_sources(*values: Any) -> dict[str, Any]:
    """Merge error mappings without inspecting pose/position fields.

    Nested ``goal_error``/``body_error`` wrappers are accepted for convenience;
    only scalar error aliases are copied.  A raw body state therefore cannot
    accidentally become a hidden position controller input.
    """

    result: dict[str, Any] = {}
    allowed_nested = ("goal_error", "body_error", "errors", "goal", "body")
    for value in values:
        if value is None:
            continue
        raw = _as_mapping(value, "error")
        for nested_name in allowed_nested:
            nested = raw.get(nested_name)
            if nested is None or nested is value:
                continue
            try:
                result.update(_as_mapping(nested, nested_name))
            except StabilizerInputError:
                # A body/goal record can carry a non-control metadata object;
                # ignore it rather than reading positions from it.
                continue
        # Outer fields have precedence over nested wrappers.
        for key, item in raw.items():
            if _normal_key(key) not in allowed_nested:
                result[str(key)] = item
    return result


def _lookup(raw: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    by_key = {_normal_key(key): value for key, value in raw.items()}
    for name in names:
        key = _normal_key(name)
        if key in by_key:
            return by_key[key]
    return default


def _read_error(raw: Mapping[str, Any], names: Sequence[str], default: float = 0.0) -> tuple[float, bool]:
    value = _lookup(raw, names, default)
    return _finite(value, names[0], default=default)


def _normalise_phase(phase: Any) -> tuple[str, float, bool]:
    progress = 0.0
    valid = True
    if isinstance(phase, Mapping):
        raw_name = _lookup(phase, ("label", "phase", "mission_phase", "scenario"), "hover")
        raw_progress = _lookup(phase, ("progress", "phase_progress", "local_phase"), 0.0)
        progress, progress_ok = _finite(raw_progress, "phase.progress", default=0.0)
        valid &= progress_ok
    elif phase is None:
        raw_name = "hover"
    elif isinstance(phase, (int, float)) and not isinstance(phase, bool):
        raw_name = "forward"
        progress, progress_ok = _finite(phase, "phase", default=0.0)
        valid &= progress_ok
    else:
        # Enum-like phase labels (for example flight_autopilot.FlightMode)
        # carry their wire spelling in ``value``.  Plain objects with a
        # ``label`` property are accepted as a small runner convenience.
        raw_name = getattr(phase, "value", getattr(phase, "label", phase))
    name = _normal_key(raw_name)
    canonical = _PHASE_ALIASES.get(name, name.replace("_", "-"))
    if canonical not in {
        "grounded", "takeoff", "forward", "turn", "hover", "gust-recovery",
        "approach", "landing", "settle",
    }:
        # Unknown phase is still deterministic and conservative: use the
        # configured default phase rather than inventing authority.
        canonical = "hover"
        valid = False
    return canonical, _clip(progress, 0.0, 1.0), valid


@dataclass(frozen=True)
class FlightBodyErrors:
    """Precomputed scalar errors consumed by the stabilizer.

    ``*_error`` values are goal-minus-body errors supplied by the runner.
    ``pitch_error_rad``/``roll_error_rad`` are desired attitude minus measured
    attitude.  No positions or body records are stored here.
    """

    altitude_error_m: float = 0.0
    forward_error_m: float = 0.0
    # Body-frame forward velocity.  This is an observed scalar supplied by
    # the runner, not a body pose or an integrated controller state.
    forward_velocity_mps: float = 0.0
    lateral_error_m: float = 0.0
    # Body-frame lateral velocity; positive is toward the body-left (+y)
    # axis.  It is optional so older runners remain behavior-compatible.
    lateral_velocity_mps: float = 0.0
    heading_error_rad: float = 0.0
    pitch_error_rad: float = 0.0
    roll_error_rad: float = 0.0
    vertical_speed_mps: float = 0.0
    pitch_rate_rad_s: float = 0.0
    roll_rate_rad_s: float = 0.0
    yaw_rate_rad_s: float = 0.0
    speed_error_mps: float = 0.0
    input_finite: bool = True

    @classmethod
    def from_inputs(cls, goal_error: Any = None, body_error: Any = None) -> "FlightBodyErrors":
        raw = _flatten_error_sources(goal_error, body_error)
        aliases = {
            "altitude_error_m": ("altitude_error_m", "altitude_error", "vertical_error_m", "z_error_m"),
            "forward_error_m": ("forward_error_m", "forward_error", "longitudinal_error_m"),
            "forward_velocity_mps": ("forward_velocity_mps", "forward_velocity", "vx_mps"),
            "lateral_error_m": ("lateral_error_m", "lateral_error", "cross_track_error_m"),
            "lateral_velocity_mps": ("lateral_velocity_mps", "lateral_velocity", "vy_mps"),
            "heading_error_rad": ("heading_error_rad", "heading_error", "yaw_error_rad", "yaw_error"),
            "pitch_error_rad": ("pitch_error_rad", "pitch_error"),
            "roll_error_rad": ("roll_error_rad", "roll_error"),
            "vertical_speed_mps": ("vertical_speed_mps", "vertical_speed", "vz_mps"),
            "pitch_rate_rad_s": ("pitch_rate_rad_s", "pitch_rate", "pitch_rate_radps"),
            "roll_rate_rad_s": ("roll_rate_rad_s", "roll_rate", "roll_rate_radps"),
            "yaw_rate_rad_s": ("yaw_rate_rad_s", "yaw_rate", "yaw_rate_radps"),
            "speed_error_mps": ("speed_error_mps", "speed_error", "airspeed_error_mps"),
        }
        values: dict[str, float] = {}
        finite = True
        for field_name, names in aliases.items():
            value, okay = _read_error(raw, names)
            values[field_name] = value
            finite &= okay
        values["input_finite"] = finite  # type: ignore[assignment]
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlightBodyStabilizerConfig:
    """Explicit hand-built overlay calibration and authority limits."""

    schema_version: str = SCHEMA_VERSION
    enabled: bool = True
    max_contribution_fraction: float = 0.35
    max_collective_trim: float = 0.20
    max_pitch_command: float = 0.50
    max_roll_command: float = 0.50
    gains: Mapping[str, float] = field(default_factory=lambda: {
        # Positive forward error commands positive pitch in the reduced model.
        "pitch_forward_error": 0.55,
        # Positive forward velocity commands negative pitch to brake it.
        "pitch_forward_velocity": 0.0,
        "pitch_attitude_error": 1.70,
        "pitch_rate": -0.25,
        # Positive body-left lateral error commands negative roll in the
        # scalar-first body convention used by FlightDynamics.
        "roll_lateral_error": -0.55,
        # Positive body-left velocity commands positive roll to brake it;
        # runners may omit this optional input, in which case the term is 0.
        "roll_lateral_velocity": 0.0,
        "roll_attitude_error": 1.70,
        "roll_rate": -0.25,
        "collective_altitude_error": 0.40,
        "collective_vertical_speed": -0.25,
    })
    phase_gains: Mapping[str, float] = field(default_factory=lambda: {
        "grounded": 0.0,
        "takeoff": 1.0,
        "forward": 1.0,
        "turn": 0.85,
        "hover": 1.0,
        "gust-recovery": 1.0,
        "approach": 0.85,
        "landing": 0.65,
        "settle": 0.35,
    })
    actuator_bounds: Mapping[str, tuple[float, float]] = field(default_factory=lambda: {
        "left_wing": (0.0, 1.0),
        "right_wing": (0.0, 1.0),
        "thrust": (0.0, 1.0),
        "yaw": (-1.0, 1.0),
        "pitch": (-1.0, 1.0),
        "roll": (-1.0, 1.0),
        "brake": (0.0, 1.0),
    })
    source: str = HAND_BUILT_SOURCE
    biological_status: str = "hypothesis"
    calibration: str = "calibration"
    research_reference: str = "docs/research/embodied-precedents.md"
    confidence: str = "low"
    claim: bool = False

    safety_contribution_cap = property(lambda self: self.max_contribution_fraction)
    max_stabilizer_fraction = safety_contribution_cap
    collective_trim_limit = property(lambda self: self.max_collective_trim)

    def __post_init__(self) -> None:
        if str(self.schema_version).strip() != SCHEMA_VERSION:
            raise StabilizerConfigError(
                f"schema_version must be {SCHEMA_VERSION!r}"
            )
        for name in (
            "max_contribution_fraction", "max_collective_trim",
            "max_pitch_command", "max_roll_command",
        ):
            value = _required_finite(getattr(self, name), name)
            if name == "max_contribution_fraction":
                if not 0.0 <= value <= 1.0:
                    raise StabilizerConfigError(f"{name} must be in [0, 1]")
            elif value < 0.0:
                raise StabilizerConfigError(f"{name} must be non-negative")
        gains = {str(key): _required_finite(value, f"gains.{key}")
                 for key, value in dict(self.gains).items()}
        phases: dict[str, float] = {}
        for key, value in dict(self.phase_gains).items():
            numeric = _required_finite(value, f"phase_gains.{key}")
            if not 0.0 <= numeric <= 1.0:
                raise StabilizerConfigError(f"phase_gains.{key} must be in [0, 1]")
            phases[_normal_key(key).replace("_", "-")] = numeric
        bounds: dict[str, tuple[float, float]] = {}
        for actuator in _ACTUATORS:
            raw = dict(self.actuator_bounds).get(actuator, (0.0, 1.0))
            try:
                low_raw, high_raw = raw
            except (TypeError, ValueError) as exc:
                raise StabilizerConfigError(
                    f"actuator_bounds.{actuator} must contain [low, high]"
                ) from exc
            low = _required_finite(low_raw, f"actuator_bounds.{actuator}[0]")
            high = _required_finite(high_raw, f"actuator_bounds.{actuator}[1]")
            if low >= high:
                raise StabilizerConfigError(
                    f"actuator_bounds.{actuator} low must be < high"
                )
            bounds[actuator] = (low, high)
        source = str(self.source).strip() or HAND_BUILT_SOURCE
        if source != HAND_BUILT_SOURCE:
            raise StabilizerConfigError("stabilizer source must be 'handbuilt'")
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "phase_gains", phases)
        object.__setattr__(self, "actuator_bounds", bounds)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "biological_status", str(self.biological_status).strip() or "hypothesis")
        object.__setattr__(self, "calibration", str(self.calibration).strip() or "calibration")
        object.__setattr__(self, "research_reference", str(self.research_reference).strip())
        object.__setattr__(self, "confidence", str(self.confidence).strip() or "low")
        object.__setattr__(self, "claim", bool(self.claim))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlightBodyStabilizerConfig":
        if not isinstance(value, Mapping):
            raise StabilizerConfigError("config must be a mapping")
        raw = dict(value)
        authority = raw.get("authority", {})
        if not isinstance(authority, Mapping):
            authority = {}
        limits = raw.get("limits", {})
        if not isinstance(limits, Mapping):
            limits = {}
        gains = raw.get("gains", {})
        if not isinstance(gains, Mapping):
            raise StabilizerConfigError("config.gains must be an object")
        phase_gains = raw.get("phase_gains", raw.get("phase_weights", {}))
        if not isinstance(phase_gains, Mapping):
            raise StabilizerConfigError("config.phase_gains must be an object")
        default = cls()
        merged_gains = dict(default.gains)
        merged_gains.update(gains)
        merged_phases = dict(default.phase_gains)
        merged_phases.update(phase_gains)
        bounds = raw.get("actuator_bounds", default.actuator_bounds)
        if not isinstance(bounds, Mapping):
            raise StabilizerConfigError("config.actuator_bounds must be an object")
        merged_bounds = dict(default.actuator_bounds)
        merged_bounds.update(bounds)
        enabled = raw.get("enabled", authority.get("enabled", default.enabled))
        if "disabled" in raw:
            enabled = not bool(raw["disabled"])
        if not isinstance(enabled, bool):
            raise StabilizerConfigError("config.enabled must be bool")
        max_fraction = raw.get(
            "max_contribution_fraction",
            raw.get("max_stabilizer_fraction", raw.get(
                "safety_contribution_cap", authority.get(
                    "max_contribution_fraction", default.max_contribution_fraction))),
        )
        collective = raw.get(
            "max_collective_trim",
            raw.get("collective_trim_limit", limits.get(
                "max_collective_trim", default.max_collective_trim)),
        )
        max_pitch = raw.get("max_pitch_command", limits.get("max_pitch_command", default.max_pitch_command))
        max_roll = raw.get("max_roll_command", limits.get("max_roll_command", default.max_roll_command))
        provenance = raw.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        return cls(
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
            enabled=bool(enabled),
            max_contribution_fraction=max_fraction,
            max_collective_trim=collective,
            max_pitch_command=max_pitch,
            max_roll_command=max_roll,
            gains=merged_gains,
            phase_gains=merged_phases,
            actuator_bounds=merged_bounds,
            source=str(provenance.get("source", HAND_BUILT_SOURCE)),
            biological_status=str(provenance.get("biological_status", provenance.get("status", "hypothesis"))),
            calibration=str(provenance.get("calibration", "calibration")),
            research_reference=str(provenance.get(
                "research_reference", "docs/research/embodied-precedents.md")),
            confidence=str(provenance.get("confidence", "low")),
            claim=bool(provenance.get("biological_claim", provenance.get("claim", False))),
        )

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONFIG) -> "FlightBodyStabilizerConfig":
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise StabilizerConfigError(f"cannot read stabilizer config {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise StabilizerConfigError(f"invalid JSON in stabilizer config {config_path}") from exc
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "max_contribution_fraction": self.max_contribution_fraction,
            "max_collective_trim": self.max_collective_trim,
            "max_pitch_command": self.max_pitch_command,
            "max_roll_command": self.max_roll_command,
            "gains": dict(self.gains),
            "phase_gains": dict(self.phase_gains),
            "actuator_bounds": {
                name: list(bounds) for name, bounds in self.actuator_bounds.items()
            },
            "provenance": {
                "source": self.source,
                "biological_status": self.biological_status,
                "calibration": self.calibration,
                "research_reference": self.research_reference,
                "confidence": self.confidence,
                "biological_claim": self.claim,
            },
        }

    parameter_manifest = to_dict

    def manifest(self) -> dict[str, Any]:
        payload = self.to_dict()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return {
            "schema_version": SCHEMA_VERSION,
            "name": "flight-body-stabilizer",
            "enabled": self.enabled,
            "authority": {
                "max_contribution_fraction": self.max_contribution_fraction,
                "max_collective_trim": self.max_collective_trim,
                "pitch_roll_only": True,
                "preserves_left_right_steering": True,
                "writes_body_state": False,
            },
            "config_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "config": payload,
            "provenance": {
                "source": self.source,
                "biological_status": self.biological_status,
                "calibration": self.calibration,
                "confidence": self.confidence,
                "biological_claim": self.claim,
                "research_reference": self.research_reference,
                "note": "low-confidence flight pitch/altitude control proxy; optional hand-built overlay",
            },
        }


@dataclass(frozen=True)
class StabilizerOutput:
    """One bounded blended actuator packet with explicit provenance."""

    channels: Mapping[str, float]
    fitted_channels: Mapping[str, float]
    stabilizer_channels: Mapping[str, float]
    contribution_fractions: Mapping[str, float]
    actuator_contribution_fractions: Mapping[str, Mapping[str, float]]
    phase: str
    phase_gain: float
    enabled: bool
    active: bool
    stabilizer_blend: float
    sequence: int
    time_s: float
    source: str = BLENDED_SOURCE
    neural_source: str = FITTED_SOURCE
    stabilizer_source: str = HAND_BUILT_SOURCE
    provenance: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def actuators(self) -> Mapping[str, float]:
        return self.channels

    motor_channels = actuators
    output_channels = actuators

    # Canonical actuator properties make the packet consumable by protocol
    # adapters that inspect attributes (for example NeuromuscularInput) while
    # keeping ``channels`` as the one serialized mapping.
    @property
    def left_wing(self) -> float:
        return float(self.channels.get("left_wing", 0.0))

    left_power = left_wing
    left_wing_power = left_wing

    @property
    def right_wing(self) -> float:
        return float(self.channels.get("right_wing", 0.0))

    right_power = right_wing
    right_wing_power = right_wing

    @property
    def thrust(self) -> float:
        return float(self.channels.get("thrust", 0.0))

    @property
    def wing_power(self) -> float:
        return 0.5 * (self.left_wing + self.right_wing)

    power = wing_power

    @property
    def yaw(self) -> float:
        return float(self.channels.get("yaw", 0.0))

    wing_steering = yaw
    flight_steering = yaw
    flight_yaw = yaw

    @property
    def pitch(self) -> float:
        return float(self.channels.get("pitch", 0.0))

    @property
    def roll(self) -> float:
        return float(self.channels.get("roll", 0.0))

    @property
    def brake(self) -> float:
        return float(self.channels.get("brake", 0.0))

    def to_motor_channels(self) -> dict[str, float]:
        """Return only bounded actuator channels for a downstream adapter."""

        return {str(key): float(value) for key, value in self.channels.items()}

    as_motor_channels = to_motor_channels

    @property
    def contribution_fraction(self) -> float:
        return self.stabilizer_blend

    safety_contribution = contribution_fraction

    @property
    def neural_fraction(self) -> float:
        return float(self.contribution_fractions.get("neural", 1.0))

    @property
    def handbuilt_fraction(self) -> float:
        return float(self.contribution_fractions.get("handbuilt", 0.0))

    @property
    def blend_source(self) -> str:
        """Composition label; ``source`` remains the primary fitted command."""

        return BLENDED_SOURCE if self.active else FITTED_SOURCE

    @property
    def command_source(self) -> str:
        """Closed command-source label suitable for the shared motor contract."""

        return self.source

    @property
    def safety_active(self) -> bool:
        """Compatibility telemetry alias for an active bounded overlay."""

        return self.active

    safety_blend = property(lambda self: self.stabilizer_blend)

    @property
    def safety_reason(self) -> str:
        return "low-confidence-body-stabilizer" if self.active else ""

    def __getitem__(self, key: str) -> Any:
        if key in self.channels:
            return self.channels[key]
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "channels": dict(self.channels),
            "fitted_channels": dict(self.fitted_channels),
            "stabilizer_channels": dict(self.stabilizer_channels),
            "contribution_fractions": dict(self.contribution_fractions),
            "actuator_contribution_fractions": {
                key: dict(value) for key, value in self.actuator_contribution_fractions.items()
            },
            "phase": self.phase,
            "phase_gain": self.phase_gain,
            "enabled": self.enabled,
            "active": self.active,
            "stabilizer_blend": self.stabilizer_blend,
            "safety_active": self.safety_active,
            "safety_reason": self.safety_reason,
            "sequence": self.sequence,
            "time_s": self.time_s,
            "source": self.source,
            "blend_source": self.blend_source,
            "neural_source": self.neural_source,
            "stabilizer_source": self.stabilizer_source,
            "provenance": dict(self.provenance),
            "diagnostics": dict(self.diagnostics),
        })

    as_dict = to_dict


class FlightBodyStabilizer:
    """State-light, disable-able low-level hand-built actuator overlay."""

    def __init__(
        self,
        config: Optional[FlightBodyStabilizerConfig | Mapping[str, Any]] = None,
        *,
        enabled: Optional[bool] = None,
        ablation: bool = False,
        config_path: Optional[str | Path] = None,
    ) -> None:
        if config is not None and config_path is not None:
            raise StabilizerConfigError("pass only one of config or config_path")
        if config_path is not None:
            config = FlightBodyStabilizerConfig.from_json(config_path)
        if config is None:
            cfg = FlightBodyStabilizerConfig()
        elif isinstance(config, FlightBodyStabilizerConfig):
            cfg = config
        else:
            cfg = FlightBodyStabilizerConfig.from_mapping(config)
        self.config = cfg
        self.enabled = cfg.enabled if enabled is None else bool(enabled)
        if ablation:
            self.enabled = False
        self.sequence = 0
        self.time_s = 0.0
        self.last_output: Optional[StabilizerOutput] = None

    @classmethod
    def from_json(
        cls,
        path: str | Path = DEFAULT_CONFIG,
        *,
        enabled: Optional[bool] = None,
        ablation: bool = False,
    ) -> "FlightBodyStabilizer":
        return cls(FlightBodyStabilizerConfig.from_json(path), enabled=enabled, ablation=ablation)

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self.config.manifest()["provenance"])

    manifest_data = property(lambda self: self.manifest())

    def manifest(self) -> dict[str, Any]:
        manifest = self.config.manifest()
        manifest["runtime_enabled"] = bool(self.enabled)
        manifest["runtime"] = {
            "sequence": int(self.sequence),
            "time_s": float(self.time_s),
            "deterministic_reset": True,
        }
        return manifest

    def reset(self) -> None:
        """Reset only deterministic telemetry; no body state is touched."""

        self.sequence = 0
        self.time_s = 0.0
        self.last_output = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "time_s": float(self.time_s),
            "enabled": bool(self.enabled),
            "manifest": self.config.manifest(),
        }

    def set_enabled(self, enabled: bool) -> bool:
        self.enabled = bool(enabled)
        return self.enabled

    @property
    def ablation(self) -> bool:
        return not self.enabled

    @ablation.setter
    def ablation(self, value: bool) -> None:
        self.enabled = not bool(value)

    def enable(self) -> bool:
        return self.set_enabled(True)

    def disable(self) -> bool:
        return self.set_enabled(False)

    @staticmethod
    def _coerce_channels(value: Any) -> tuple[dict[str, float], bool, str, dict[str, Any]]:
        """Adapt fitted readout records without importing the autopilot."""

        if value is None:
            raw: dict[str, Any] = {}
            input_source = FITTED_SOURCE
            source_metadata: dict[str, Any] = {}
        elif isinstance(value, Mapping):
            raw = dict(value)
            nested = raw.get("actuators", raw.get("channels"))
            if isinstance(nested, Mapping):
                outer = {key: item for key, item in raw.items()
                         if key not in {"actuators", "channels"}}
                outer.update(nested)
                raw = outer
            raw_source = raw.get("source", FITTED_SOURCE)
            input_source = str(getattr(raw_source, "value", raw_source))
            source_metadata = raw.get("provenance", {}) if isinstance(raw.get("provenance", {}), Mapping) else {}
        else:
            nested = getattr(value, "actuators", getattr(value, "channels", None))
            if isinstance(nested, Mapping):
                raw = dict(nested)
            elif isinstance(getattr(value, "named_outputs", None), Mapping):
                raw = dict(value.named_outputs)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                raw = {name: value[index] for index, name in enumerate(_ACTUATORS) if index < len(value)}
            else:
                raw = {}
                for actuator in _ACTUATORS:
                    for alias in _ALIASES[actuator]:
                        if hasattr(value, alias):
                            raw[alias] = getattr(value, alias)
                            break
            raw_source = getattr(value, "source", FITTED_SOURCE)
            input_source = str(getattr(raw_source, "value", raw_source))
            candidate_provenance = getattr(value, "provenance", {})
            source_metadata = candidate_provenance if isinstance(candidate_provenance, Mapping) else {}
        # Flatten one common nested named-output wrapper.  This is a readout
        # adaptation only; no neural state or body record is changed.
        for nested_name in ("named_outputs", "motor_channels"):
            nested = raw.get(nested_name)
            if isinstance(nested, Mapping):
                for key, item in nested.items():
                    raw.setdefault(str(key), item)
        channels: dict[str, float] = {}
        finite = True
        for actuator in _ACTUATORS:
            value_for_actuator = _lookup(raw, _ALIASES[actuator], 0.0)
            number, okay = _finite(value_for_actuator, actuator, default=0.0)
            channels[actuator] = number
            finite &= okay
        # If only one of the wings was supplied, use the available neural
        # collective for the other.  This does not introduce a stabilizer
        # steering difference.
        left_present = any(_normal_key(key) in {_normal_key(alias) for alias in _ALIASES["left_wing"]} for key in raw)
        right_present = any(_normal_key(key) in {_normal_key(alias) for alias in _ALIASES["right_wing"]} for key in raw)
        thrust_present = any(_normal_key(key) in {_normal_key(alias) for alias in _ALIASES["thrust"]} for key in raw)
        if left_present and not right_present:
            channels["right_wing"] = channels["left_wing"]
        elif right_present and not left_present:
            channels["left_wing"] = channels["right_wing"]
        elif not left_present and not right_present:
            channels["left_wing"] = channels["thrust"]
            channels["right_wing"] = channels["thrust"]
        if not thrust_present:
            channels["thrust"] = 0.5 * (channels["left_wing"] + channels["right_wing"])
        return channels, finite, input_source, dict(source_metadata)

    def _bounded(self, actuator: str, value: Any) -> float:
        numeric, okay = _finite(value, actuator, default=0.0)
        if not okay:
            numeric = 0.0
        low, high = self.config.actuator_bounds[actuator]
        return _clip(numeric, low, high)

    def _collective_target(
        self,
        fitted: Mapping[str, float],
        requested_delta: float,
    ) -> tuple[dict[str, float], float]:
        """Apply equal wing trim while preserving neural L/R steering."""

        left = self._bounded("left_wing", fitted.get("left_wing", 0.0))
        right = self._bounded("right_wing", fitted.get("right_wing", 0.0))
        # The interval below is the largest common delta that keeps both wing
        # channels inside bounds.  Therefore target_right-target_left remains
        # exactly the fitted neural steering difference.
        low_left, high_left = self.config.actuator_bounds["left_wing"]
        low_right, high_right = self.config.actuator_bounds["right_wing"]
        delta_low = max(low_left - left, low_right - right)
        delta_high = min(high_left - left, high_right - right)
        delta = _clip(requested_delta, delta_low, delta_high)
        target_left = _clip(left + delta, low_left, high_left)
        target_right = _clip(right + delta, low_right, high_right)
        result = dict(fitted)
        result["left_wing"] = target_left
        result["right_wing"] = target_right
        # Keep the scalar thrust channel coherent with the equal trim while
        # respecting its own bound.  It is not used to overwrite wing
        # asymmetry by FlightDynamics, but remains useful to other adapters.
        thrust = self._bounded("thrust", fitted.get("thrust", 0.5 * (left + right)))
        result["thrust"] = self._bounded("thrust", thrust + delta)
        return result, delta

    def step(
        self,
        goal_error: Any = None,
        body_error: Any = None,
        phase: Any = "hover",
        fitted_motor_channels: Any = None,
        *,
        dt_s: float = 0.02,
        dt: Optional[float] = None,
        errors: Any = None,
        motor_channels: Any = None,
    ) -> StabilizerOutput:
        """Blend one fitted neural packet with bounded low-level corrections.

        The positional order is ``goal_error, body_error, phase,
        fitted_motor_channels``.  Keyword calls are recommended for runner
        integration.  ``dt_s`` advances only this module's telemetry clock;
        it is never used to integrate or write body state.
        """

        if dt is not None:
            if dt_s != 0.02:
                raise StabilizerInputError("pass only one of dt_s or dt")
            dt_s = dt
        if errors is not None:
            if goal_error is not None or body_error is not None:
                raise StabilizerInputError(
                    "pass either errors or goal_error/body_error, not both"
                )
            goal_error = errors
        if motor_channels is not None:
            if fitted_motor_channels is not None:
                raise StabilizerInputError(
                    "pass only one of fitted_motor_channels or motor_channels"
                )
            fitted_motor_channels = motor_channels
        # Also accept the compact three-positional form used by a few runner
        # adapters: ``step(errors, phase, fitted_channels)``.  The explicit
        # four-argument form above remains the canonical contract.
        if fitted_motor_channels is None and isinstance(phase, Mapping) and not isinstance(body_error, Mapping):
            fitted_motor_channels = phase
            phase = body_error if body_error is not None else "hover"
            body_error = None
        dt_value, dt_ok = _finite(dt_s, "dt_s", default=0.02)
        if not dt_ok or dt_value <= 0.0 or dt_value > 10.0:
            dt_value = 0.02
            dt_ok = False
        errors = FlightBodyErrors.from_inputs(goal_error, body_error)
        phase_name, progress, phase_ok = _normalise_phase(phase)
        fitted, channels_finite, neural_source, source_metadata = self._coerce_channels(
            fitted_motor_channels
        )
        phase_gain = _clip(float(self.config.phase_gains.get(phase_name, 0.0)), 0.0, 1.0)
        # A malformed error/channel sample cannot justify hand-built
        # authority.  Keep the bounded fitted packet (with non-finite values
        # already fail-closed to zero) and expose the invalid sample in
        # diagnostics instead.
        input_valid = bool(errors.input_finite and channels_finite and dt_ok and phase_ok)
        active = bool(self.enabled and phase_gain > 0.0 and input_valid)
        blend = self.config.max_contribution_fraction * phase_gain if active else 0.0
        blend = _clip(blend, 0.0, self.config.max_contribution_fraction)

        gains = self.config.gains
        pitch_raw = (
            gains.get("pitch_forward_error", 0.0) * errors.forward_error_m
            + gains.get("pitch_forward_velocity", 0.0) * errors.forward_velocity_mps
            + gains.get("pitch_attitude_error", 0.0) * errors.pitch_error_rad
            + gains.get("pitch_rate", 0.0) * errors.pitch_rate_rad_s
        )
        roll_raw = (
            gains.get("roll_lateral_error", 0.0) * errors.lateral_error_m
            + gains.get("roll_lateral_velocity", 0.0) * errors.lateral_velocity_mps
            + gains.get("roll_attitude_error", 0.0) * errors.roll_error_rad
            + gains.get("roll_rate", 0.0) * errors.roll_rate_rad_s
        )
        collective_raw = (
            gains.get("collective_altitude_error", 0.0) * errors.altitude_error_m
            + gains.get("collective_vertical_speed", 0.0) * errors.vertical_speed_mps
        )
        pitch_target = _clip(pitch_raw, -self.config.max_pitch_command, self.config.max_pitch_command)
        roll_target = _clip(roll_raw, -self.config.max_roll_command, self.config.max_roll_command)
        collective_requested = _clip(
            collective_raw, -self.config.max_collective_trim, self.config.max_collective_trim
        )
        if not active:
            pitch_target = self._bounded("pitch", fitted.get("pitch", 0.0))
            roll_target = self._bounded("roll", fitted.get("roll", 0.0))
            collective_requested = 0.0

        # Build the common collective target first (it deliberately starts
        # from the fitted packet so equal wing trim cannot erase steering),
        # then add the independent pitch/roll targets.
        stabilizer_target, collective_delta = self._collective_target(fitted, collective_requested)
        stabilizer_target["pitch"] = self._bounded("pitch", pitch_target)
        stabilizer_target["roll"] = self._bounded("roll", roll_target)
        # Blending a target that is itself neural+equal-trim means a common
        # delta is added to both wings while their difference remains neural.
        blended: dict[str, float] = dict(fitted)
        for actuator in _ACTUATORS:
            neural_value = self._bounded(actuator, fitted.get(actuator, 0.0))
            target_value = self._bounded(actuator, stabilizer_target.get(actuator, neural_value))
            authority = blend if actuator in {"left_wing", "right_wing", "thrust", "pitch", "roll"} else 0.0
            blended[actuator] = self._bounded(
                actuator, (1.0 - authority) * neural_value + authority * target_value
            )

        # Preserve arbitrary fitted channels for downstream adapters, but do
        # not let unknown values bypass the bounded canonical contract.
        for key, value in fitted.items():
            if key not in blended:
                blended[key] = value
        effective_stabilizer = blend if active else 0.0
        actuator_fractions: dict[str, dict[str, float]] = {}
        for actuator in _ACTUATORS:
            has_authority = actuator in {"left_wing", "right_wing", "thrust", "pitch", "roll"} and active
            fraction = effective_stabilizer if has_authority else 0.0
            actuator_fractions[actuator] = {
                "neural": 1.0 - fraction,
                "handbuilt": fraction,
            }
        self.sequence += 1
        self.time_s += dt_value
        # Keep the shared motor contract's command source valid: the fitted
        # neural packet remains primary even while the explicit hand-built
        # overlay contributes.  ``blend_source`` and ``stabilizer_source``
        # carry the composition without inventing a new source vocabulary.
        source = FITTED_SOURCE
        output = StabilizerOutput(
            channels=blended,
            fitted_channels=dict(fitted),
            stabilizer_channels=stabilizer_target,
            contribution_fractions={
                "neural": 1.0 - effective_stabilizer,
                "handbuilt": effective_stabilizer,
                "stabilizer": effective_stabilizer,
            },
            actuator_contribution_fractions=actuator_fractions,
            phase=phase_name,
            phase_gain=phase_gain,
            enabled=bool(self.enabled),
            active=active,
            stabilizer_blend=effective_stabilizer,
            sequence=self.sequence,
            time_s=self.time_s,
            source=source,
            neural_source=FITTED_SOURCE,
            stabilizer_source=HAND_BUILT_SOURCE,
            provenance={
                "source": source,
                "blend_source": BLENDED_SOURCE if active else FITTED_SOURCE,
                "neural": FITTED_SOURCE,
                "fitted_readout": FITTED_SOURCE,
                "stabilizer": HAND_BUILT_SOURCE,
                "safety": HAND_BUILT_SOURCE if active else "none",
                "research_reference": self.config.research_reference,
                "biological_status": self.config.biological_status,
                "calibration": self.config.calibration,
                "confidence": self.config.confidence,
                "biological_claim": self.config.claim,
                "input_source_declared": neural_source,
                "input_metadata": _json_safe(source_metadata),
                "writes_body_state": False,
                "preserves_left_right_steering": True,
                "authority_cap": self.config.max_contribution_fraction,
                "ablation": not self.enabled,
                "input_valid": input_valid,
            },
            diagnostics={
                "input_finite": input_valid,
                "goal_body_errors": errors.to_dict(),
                "pitch_raw": pitch_raw,
                "roll_raw": roll_raw,
                "collective_requested": collective_requested,
                "collective_delta": collective_delta * effective_stabilizer,
                "collective_target_delta": collective_delta,
                "progress": progress,
                "phase_valid": phase_ok,
                "dt_valid": dt_ok,
                "left_right_neural_difference": fitted["right_wing"] - fitted["left_wing"],
                "left_right_output_difference": blended["right_wing"] - blended["left_wing"],
            },
        )
        self.last_output = output
        return output

    update = step
    blend = step
    apply = step


# Compatibility names used by small runner adapters and experiment notes.
BodyStabilizer = FlightBodyStabilizer
LowLevelFlightStabilizer = FlightBodyStabilizer
FlightStabilizer = FlightBodyStabilizer
StabilizerConfig = FlightBodyStabilizerConfig
BodyStabilizerConfig = FlightBodyStabilizerConfig
StabilizedMotorChannels = StabilizerOutput
BlendedMotorChannels = StabilizerOutput


def load_stabilizer(
    path: str | Path = DEFAULT_CONFIG,
    *,
    enabled: Optional[bool] = None,
    ablation: bool = False,
) -> FlightBodyStabilizer:
    """Load the optional stabilizer using the explicit JSON manifest."""

    return FlightBodyStabilizer.from_json(path, enabled=enabled, ablation=ablation)


def stabilize_flight_body(
    goal_error: Any,
    body_error: Any,
    phase: Any,
    fitted_motor_channels: Any,
    *,
    config: Optional[FlightBodyStabilizerConfig | Mapping[str, Any]] = None,
    enabled: Optional[bool] = None,
    dt_s: float = 0.02,
) -> StabilizerOutput:
    """Stateless convenience wrapper for one explicit stabilizer sample."""

    controller = FlightBodyStabilizer(config, enabled=enabled)
    return controller.step(goal_error, body_error, phase, fitted_motor_channels, dt_s=dt_s)


blend_motor_channels = stabilize_flight_body
apply_body_stabilizer = stabilize_flight_body


__all__ = [
    "BLENDED_SOURCE", "BodyStabilizer", "BodyStabilizerConfig", "BlendedMotorChannels",
    "DEFAULT_CONFIG", "DISABLED_SOURCE", "FITTED_SOURCE", "FlightBodyErrors",
    "FlightBodyStabilizer", "FlightBodyStabilizerConfig", "FlightStabilizer",
    "HAND_BUILT_SOURCE", "LowLevelFlightStabilizer", "SCHEMA_VERSION",
    "StabilizedMotorChannels", "StabilizerConfig", "StabilizerConfigError",
    "StabilizerError", "StabilizerInputError", "StabilizerOutput", "load_stabilizer",
    "stabilize_flight_body", "blend_motor_channels", "apply_body_stabilizer",
]


if __name__ == "__main__":  # pragma: no cover - tiny manifest CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()
    controller = FlightBodyStabilizer.from_json(args.config, enabled=not args.disabled)
    print(json.dumps(controller.manifest(), indent=2, sort_keys=True))
