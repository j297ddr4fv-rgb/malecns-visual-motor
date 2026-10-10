#!/usr/bin/env python3
"""Small, dependency-free flight behaviour controller.

This module is deliberately explicit about the boundary between the model and
the controller glue:

``ConnectomeOutput``
    The motor channels emitted by the connectome.  No state-machine signal is
    mixed into this object.
``ReadoutOutput``
    A fitted (linear) readout from those channels to actuator coordinates.
    This is a decoder, not a claim that the decoder is part of the connectome.
``SafetyLayer``
    A hand-built layer that is only allowed to respond to numeric divergence,
    stall, and the hard actuator envelope.  It can be disabled for ablations.

The state machine decides *which flight mode is active* (including the
takeoff/flight/hover/approach/landing dwell states), but in normal flight it
does not synthesize steering, altitude, or landing commands.  The command
comes from the connectome and its fitted readout.  This keeps an experiment
honest: a transition to ``TURN`` is telemetry, not a hand-built turn controller.

When a stall is confirmed, the safety layer contributes a bounded overlay and
records its blend weight; numeric divergence remains an explicit full safety
override.  Both the CNS/readout source and the safety source remain in the
command and log provenance.

No third-party package is required.  Values are normalized actuator commands:
wing/yaw/pitch/roll are in [-1, 1], while thrust and brake are in [0, 1].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Mapping, Optional, Sequence

try:
    from .flight_control_common import clip as _clip, normal_key as _normal_key, number as _number
except ImportError:
    from flight_control_common import clip as _clip, normal_key as _normal_key, number as _number


class FlightMode(str, Enum):
    TAKEOFF = "TAKEOFF"
    # ``HOVER`` is a real dwell state rather than an actuator policy.  It is
    # useful telemetry for a vehicle that is stably airborne at its target
    # altitude, and (unlike CRUISE) does not treat the expected low airspeed
    # as a stall.
    HOVER = "HOVER"
    CRUISE = "CRUISE"
    # Newer callers use FLIGHT/LANDING terminology.  Keep the older wire
    # values stable so existing telemetry and notebooks remain readable.
    FLIGHT = "CRUISE"
    TURN = "TURN"
    RECOVER = "RECOVER"
    APPROACH = "APPROACH"
    LAND = "LAND"
    LANDING = "LAND"
    GROUNDED = "GROUNDED"

    @classmethod
    def _missing_(cls, value: object) -> "FlightMode | None":
        """Accept the newer FLIGHT/LANDING spellings on input."""

        if isinstance(value, str):
            aliases = {
                "FLIGHT": cls.CRUISE,
                "LANDING": cls.LAND,
            }
            return aliases.get(value.strip().upper())
        return None


class SourceLabel(str, Enum):
    """Closed provenance vocabulary used by the flight boundary.

    ``connectome`` labels raw neural motor channels, ``fitted`` labels the
    fitted readout, and ``safety-handbuilt`` is reserved for an actual
    protective override.  A state-machine transition is deliberately not a
    source: it never writes actuator coordinates.
    """

    CONNECTOME = "connectome"
    FITTED = "fitted"
    SAFETY_HANDBUILT = "safety-handbuilt"
    # Compatibility name matching ``twin_contracts.SourceLabel``.  Commands
    # emitted here still canonicalize it to ``safety-handbuilt``.
    HANDBUILT = "handbuilt"


SourceTag = SourceLabel


_SOURCE_ALIASES = {
    "connectome": SourceLabel.CONNECTOME.value,
    "fitted": SourceLabel.FITTED.value,
    "fitted_readout": SourceLabel.FITTED.value,
    "safety-handbuilt": SourceLabel.SAFETY_HANDBUILT.value,
    "safety_handbuilt": SourceLabel.SAFETY_HANDBUILT.value,
    "safety_layer": SourceLabel.SAFETY_HANDBUILT.value,
    # Older callers used this name for the protective grounded hold.  It is
    # accepted on input only and is never emitted by the controller.
    "grounded_hold": SourceLabel.SAFETY_HANDBUILT.value,
    "handbuilt": SourceLabel.SAFETY_HANDBUILT.value,
}


def _source_value(value: Any, default: SourceLabel) -> str:
    """Return one canonical source label, rejecting untracked provenance."""

    if value is None:
        value = default
    key = value.value if hasattr(value, "value") else str(value)
    try:
        return _SOURCE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "source must be one of: connectome, fitted, safety-handbuilt"
        ) from exc


@dataclass(frozen=True)
class BodyState:
    """Measured state supplied by the body simulator or vehicle interface.

    Angles and rates are in radians and radians/second; distances and speeds
    use the caller's consistent world unit.  This is the controller's compact
    flight view, not the SI ``twin_contracts.BodyState`` record; use
    :meth:`from_contract`/ :meth:`to_contract` at that boundary.
    ``grounded=None`` means infer it from altitude and speed.
    """

    altitude: float = 0.0
    vertical_speed: float = 0.0
    airspeed: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0
    pitch: float = 0.0
    pitch_rate: float = 0.0
    roll: float = 0.0
    roll_rate: float = 0.0
    grounded: Optional[bool] = None

    @classmethod
    def from_contract(cls, value: Any) -> "BodyState":
        """Adapt the SI ``twin_contracts.BodyState`` into this flight view.

        The controller view is intentionally a projection: altitude and
        vertical/airspeed are read from the contract's position/velocity, and
        orientation/rates are read from its XYZ RPY vectors.  Nothing is
        written back to the contract object or to coordinates.
        """

        body, _ = _coerce_body(value)
        return body

    def to_contract(self, timestamp: Any = None, *, source: Any = "handbuilt") -> Any:
        """Create a SI twin-contract body snapshot without mutating this one."""

        from twin_contracts import BodyState as ContractBodyState, Timestamp

        if timestamp is None:
            timestamp = Timestamp(0.0, 0, "simulation", 1.0)
        elif isinstance(timestamp, Mapping):
            timestamp = Timestamp.from_dict(timestamp)
        if not isinstance(timestamp, Timestamp):
            raise TypeError("timestamp must be a twin_contracts.Timestamp or mapping")
        canonical_source = _source_value(source, SourceLabel.SAFETY_HANDBUILT)
        contract_source = "handbuilt" if canonical_source == SourceLabel.SAFETY_HANDBUILT.value else canonical_source

        return ContractBodyState(
            timestamp=timestamp,
            position_m=(0.0, 0.0, self.altitude),
            orientation_rad=(self.roll, self.pitch, self.yaw),
            velocity_mps=(
                self.airspeed * math.cos(self.yaw),
                self.airspeed * math.sin(self.yaw),
                self.vertical_speed,
            ),
            angular_velocity_rad_s=(self.roll_rate, self.pitch_rate, self.yaw_rate),
            source=contract_source,
        )

    to_contract_body_state = to_contract


@dataclass(frozen=True)
class VisualDirectionSignal:
    """Visual heading cue; positive direction means turn left.

    ``direction`` is normally normalized to [-1, 1].  The controller clips it
    for transition decisions, but never writes it into the neural channels.
    """

    direction: float = 0.0
    confidence: float = 1.0
    valid: bool = True


@dataclass(frozen=True)
class TargetGroundInfo:
    """Target, ground, and optional mission intent used for mode transitions.

    ``mission_phase`` is an explicit state-machine intent when supplied by a
    mission runner (for example ``turn``/``hover``/``approach``/``settle``).
    It changes telemetry mode selection only; it never contributes actuator
    coordinates.  When absent, the legacy target/visual transition rules
    remain in force for standalone callers.
    """

    target_altitude: Optional[float] = None
    target_heading: Optional[float] = None
    target_distance: Optional[float] = None
    ground_altitude: float = 0.0
    approach_distance: Optional[float] = None
    landing_distance: Optional[float] = None
    landing_altitude: Optional[float] = None
    contact: bool = False
    mission_phase: Optional[str] = None


@dataclass(frozen=True)
class MotorCommand:
    """Bounded actuator output plus provenance and safety telemetry."""

    left_wing: float = 0.0
    right_wing: float = 0.0
    thrust: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    brake: float = 0.0
    mode: FlightMode = FlightMode.GROUNDED
    source: str = SourceLabel.FITTED.value
    safety_active: bool = False
    safety_reason: str = ""
    sequence: int = 0
    # Provenance is intentionally split: the CNS/connectome packet and its
    # fitted readout remain visible even when the protective layer contributes
    # to the bounded actuator command.
    neural_source: str = SourceLabel.CONNECTOME.value
    readout_source: str = SourceLabel.FITTED.value
    safety_source: str = SourceLabel.SAFETY_HANDBUILT.value
    safety_blend: float = 0.0
    neural_actuators: Mapping[str, float] = field(default_factory=dict)
    safety_actuators: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep the public object JSON-friendly while making provenance closed
        # and canonical.  The controller only emits FITTED or
        # SAFETY_HANDBUILT; CONNECTOME remains available for explicit callers
        # that need to represent a raw channel packet.
        object.__setattr__(self, "source", _source_value(self.source, SourceLabel.FITTED))
        object.__setattr__(self, "neural_source", _source_value(self.neural_source, SourceLabel.CONNECTOME))
        object.__setattr__(self, "readout_source", _source_value(self.readout_source, SourceLabel.FITTED))
        object.__setattr__(self, "safety_source", _source_value(self.safety_source, SourceLabel.SAFETY_HANDBUILT))
        blend, blend_ok = _number(self.safety_blend)
        object.__setattr__(self, "safety_blend", _clip(blend, 0.0, 1.0) if blend_ok else 0.0)
        object.__setattr__(self, "neural_actuators", dict(self.neural_actuators or {}))
        object.__setattr__(self, "safety_actuators", dict(self.safety_actuators or {}))
        provenance = dict(self.provenance or {})
        provenance.setdefault("neural", self.neural_source)
        provenance.setdefault("cns", self.neural_source)
        provenance.setdefault("readout", self.readout_source)
        provenance.setdefault("safety", self.safety_source if self.safety_active else "none")
        provenance.setdefault("command", self.source)
        object.__setattr__(self, "provenance", provenance)
        if not isinstance(self.mode, FlightMode):
            object.__setattr__(self, "mode", FlightMode(str(self.mode)))

    # Common aliases make the object convenient for small simulators without
    # multiplying the canonical actuator fields.
    left = property(lambda self: self.left_wing)
    right = property(lambda self: self.right_wing)
    power = property(lambda self: self.thrust)

    @property
    def safety_weight(self) -> float:
        """Alias used by telemetry consumers that call the blend a weight."""

        return self.safety_blend

    @property
    def readout_blend(self) -> float:
        """Complementary fitted-readout weight in the final command."""

        return 1.0 - self.safety_blend

    @property
    def command_source(self) -> str:
        """Canonical source of the final bounded command."""

        return self.source

    @property
    def cns_source(self) -> str:
        """Explicit CNS/connectome provenance alias."""

        return self.neural_source

    def actuator_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in _ACTUATORS}

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["mode"] = self.mode.value
        result["readout_blend"] = self.readout_blend
        return result

    def to_contract(
        self,
        timestamp: Any = None,
        *,
        yaw_rate_limit_rad_s: float = 1.0,
        pitch_rate_limit_rad_s: float = 1.0,
        roll_rate_limit_rad_s: float = 1.0,
        thrust_limit_mps2: float = 1.0,
    ) -> Any:
        """Adapt normalized flight actuators to the SI twin contract.

        The limits are explicit because this module deliberately has no claim
        about the vehicle's physical scale.  Wing/brake channels remain in
        metadata; no body coordinate is changed.  ``safety-handbuilt`` maps
        to the unified contract's ``handbuilt`` vocabulary.
        """

        from twin_contracts import MotorCommand as ContractMotorCommand, Timestamp

        if timestamp is None:
            timestamp = Timestamp(0.0, 0, "simulation", 1.0)
        elif isinstance(timestamp, Mapping):
            timestamp = Timestamp.from_dict(timestamp)
        if not isinstance(timestamp, Timestamp):
            raise TypeError("timestamp must be a twin_contracts.Timestamp or mapping")
        contract_source = "handbuilt" if self.source == SourceLabel.SAFETY_HANDBUILT.value else self.source
        saturated = (
            abs(self.yaw) >= 1.0 or abs(self.pitch) >= 1.0 or abs(self.roll) >= 1.0
            or self.thrust >= 1.0 or self.thrust <= 0.0
            or self.brake >= 1.0
        )
        return ContractMotorCommand(
            timestamp=timestamp,
            yaw_rate_rad_s=self.yaw * yaw_rate_limit_rad_s,
            pitch_rate_rad_s=self.pitch * pitch_rate_limit_rad_s,
            roll_rate_rad_s=self.roll * roll_rate_limit_rad_s,
            thrust_mps2=self.thrust * thrust_limit_mps2,
            source=contract_source,
            saturated=saturated,
            metadata={
                "mode": self.mode.value,
                "sequence": self.sequence,
                "left_wing": self.left_wing,
                "right_wing": self.right_wing,
                "brake": self.brake,
                "safety_active": self.safety_active,
                "safety_reason": self.safety_reason,
                "neural_source": self.neural_source,
                "readout_source": self.readout_source,
                "safety_source": self.safety_source,
                "safety_blend": self.safety_blend,
                "readout_blend": self.readout_blend,
                "neural_actuators": dict(self.neural_actuators),
                "safety_actuators": dict(self.safety_actuators),
                "provenance": dict(self.provenance),
            },
        )

    to_twin_motor_command = to_contract
    as_contract = to_contract


@dataclass(frozen=True)
class ConnectomeOutput:
    """Raw neural motor channels, retained separately for auditability."""

    channels: Mapping[str, float]
    finite: bool = True
    source: str = SourceLabel.CONNECTOME.value

    def __post_init__(self) -> None:
        source = _source_value(self.source, SourceLabel.CONNECTOME)
        if source != SourceLabel.CONNECTOME.value:
            raise ValueError("ConnectomeOutput.source must be connectome")
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class ReadoutOutput:
    """Actuator coordinates produced by the fitted readout only."""

    actuators: Mapping[str, float]
    source: str = SourceLabel.FITTED.value
    finite: bool = True

    def __post_init__(self) -> None:
        source = _source_value(self.source, SourceLabel.FITTED)
        if source != SourceLabel.FITTED.value:
            raise ValueError("ReadoutOutput.source must be fitted")
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class SafetyDecision:
    """Result of the optional hand-built protective layer."""

    active: bool = False
    reason: str = ""
    actuators: Mapping[str, float] = field(default_factory=dict)
    source: str = SourceLabel.SAFETY_HANDBUILT.value
    blend: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _source_value(self.source, SourceLabel.SAFETY_HANDBUILT))
        blend, okay = _number(self.blend)
        object.__setattr__(self, "blend", _clip(blend, 0.0, 1.0) if okay and self.active else 0.0)
        object.__setattr__(self, "actuators", dict(self.actuators or {}))

    @property
    def safety_blend(self) -> float:
        """Alias matching :class:`MotorCommand` telemetry."""

        return self.blend


@dataclass(frozen=True)
class FlightLogEntry:
    """One JSON-friendly control-loop record."""

    sequence: int
    time_s: float
    mode: FlightMode
    previous_mode: FlightMode
    transition: str
    safety_active: bool
    safety_reason: str
    source: str
    connectome_finite: bool
    readout_finite: bool
    connectome_source: str = SourceLabel.CONNECTOME.value
    readout_source: str = SourceLabel.FITTED.value
    mode_elapsed_s: float = 0.0
    safety_blend: float = 0.0
    neural_source: str = SourceLabel.CONNECTOME.value
    safety_source: str = SourceLabel.SAFETY_HANDBUILT.value
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _source_value(self.source, SourceLabel.FITTED))
        connectome_source = _source_value(
            self.connectome_source, SourceLabel.CONNECTOME,
        )
        readout_source = _source_value(
            self.readout_source, SourceLabel.FITTED,
        )
        if connectome_source != SourceLabel.CONNECTOME.value:
            raise ValueError("FlightLogEntry.connectome_source must be connectome")
        if readout_source != SourceLabel.FITTED.value:
            raise ValueError("FlightLogEntry.readout_source must be fitted")
        object.__setattr__(self, "connectome_source", connectome_source)
        object.__setattr__(self, "readout_source", readout_source)
        object.__setattr__(self, "neural_source", _source_value(self.neural_source, SourceLabel.CONNECTOME))
        object.__setattr__(self, "safety_source", _source_value(self.safety_source, SourceLabel.SAFETY_HANDBUILT))
        blend, okay = _number(self.safety_blend)
        object.__setattr__(self, "safety_blend", _clip(blend, 0.0, 1.0) if okay else 0.0)
        provenance = dict(self.provenance or {})
        provenance.setdefault("neural", self.neural_source)
        provenance.setdefault("cns", self.neural_source)
        provenance.setdefault("readout", self.readout_source)
        provenance.setdefault("safety", self.safety_source if self.safety_active else "none")
        provenance.setdefault("command", self.source)
        object.__setattr__(self, "provenance", provenance)
        if not isinstance(self.mode, FlightMode):
            object.__setattr__(self, "mode", FlightMode(str(self.mode)))
        if not isinstance(self.previous_mode, FlightMode):
            object.__setattr__(self, "previous_mode", FlightMode(str(self.previous_mode)))

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        value["previous_mode"] = self.previous_mode.value
        value["readout_blend"] = 1.0 - self.safety_blend
        return value

    @property
    def cns_source(self) -> str:
        return self.neural_source


@dataclass
class AutopilotConfig:
    """State-machine hysteresis, timeout, and actuator safety parameters."""

    cruise_altitude: float = 1.0
    cruise_altitude_hysteresis: float = 0.06
    takeoff_thrust_threshold: float = 0.08
    ground_altitude_hysteresis: float = 0.06
    airborne_speed_threshold: float = 0.12
    climb_speed_threshold: float = -0.05
    landing_speed_threshold: float = 0.20
    stall_airspeed: float = 0.08
    stall_recover_airspeed: float = 0.20
    # The state-aware arming rule (CRUISE/TURN only) keeps the first takeoff
    # sample out of stall protection, while this short dwell also allows a
    # normal CRUISE -> HOVER slowdown to settle before declaring a stall.
    stall_grace_s: float = 0.20
    recover_clear_s: float = 0.30
    # The protective layer is deliberately a bounded overlay for ordinary
    # stalls.  Numeric divergence remains a full fail-safe override.
    safety_blend: float = 0.35
    numeric_safety_blend: float = 1.0
    hover_airspeed_threshold: float = 0.12
    hover_vertical_speed_threshold: float = 0.12
    turn_enter_threshold: float = 0.22
    turn_exit_threshold: float = 0.10
    turn_confidence: float = 0.35
    heading_turn_threshold: float = math.radians(18.0)
    approach_distance: float = 2.0
    approach_exit_distance: float = 2.20
    landing_distance: float = 0.35
    landing_altitude: float = 0.15
    state_timeout_s: float = 20.0
    takeoff_timeout_s: float = 8.0
    turn_timeout_s: float = 4.0
    approach_timeout_s: float = 15.0
    land_timeout_s: float = 12.0
    recover_timeout_s: float = 20.0
    max_altitude: float = 1000.0
    max_speed: float = 100.0
    max_rate: float = 50.0
    max_angle: float = math.radians(89.0)


_ACTUATORS = ("left_wing", "right_wing", "thrust", "yaw", "pitch", "roll", "brake")
_SIGNED = {"left_wing", "right_wing", "yaw", "pitch", "roll"}
_ALIASES: dict[str, tuple[str, ...]] = {
    "left_wing": ("left_wing", "left", "wing_left", "motor_left", "l", "L"),
    "right_wing": ("right_wing", "right", "wing_right", "motor_right", "r", "R"),
    "thrust": ("thrust", "power", "lift", "motor_thrust", "T", "t"),
    "yaw": ("yaw", "turn", "heading", "yaw_motor"),
    "pitch": ("pitch", "pitch_motor"),
    "roll": ("roll", "roll_motor"),
    "brake": ("brake", "drag"),
}


def _coerce_body(value: Any) -> tuple[BodyState, bool]:
    if isinstance(value, BodyState):
        data = asdict(value)
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        data = {
            name: getattr(value, name, None if name == "grounded" else 0.0)
            for name in BodyState.__dataclass_fields__
        }
    # ``twin_contracts.BodyState`` uses SI vectors rather than the
    # controller's named altitude/attitude fields.  Read it by protocol for
    # both objects and serialized mappings, without importing that contract.
    def vector_field(data_names: Sequence[str], attribute_names: Sequence[str]) -> Any:
        for name in data_names:
            if name in data:
                return data[name]
        for name in attribute_names:
            if hasattr(value, name):
                return getattr(value, name)
        return None

    position = vector_field(("position_m", "position"), ("position_m", "position"))
    orientation = vector_field(
        ("orientation_rad", "orientation_rpy_rad", "attitude_rpy_rad"),
        ("orientation_rad", "orientation_rpy_rad"),
    )
    velocity = vector_field(("velocity_mps", "velocity"), ("velocity_mps", "velocity"))
    angular_velocity = vector_field(
        ("angular_velocity_rad_s", "angular_velocity_radps", "angular_velocity"),
        ("angular_velocity_rad_s", "angular_velocity_radps"),
    )

    def component(vector: Any, index: int) -> tuple[float, bool]:
        try:
            item = vector[index]
        except (TypeError, IndexError, KeyError):
            return 0.0, False
        return _number(item)

    def components(vector: Any, indices: Sequence[int]) -> tuple[tuple[float, ...], bool]:
        values = tuple(component(vector, index) for index in indices)
        return tuple(number for number, _ in values), all(valid for _, valid in values)

    vectors_finite = True
    if position is not None:
        (data["altitude"],), okay = components(position, (2,))
        vectors_finite &= okay
    if velocity is not None:
        (vx, vy, vz), okay = components(velocity, (0, 1, 2))
        data["vertical_speed"] = vz
        data["airspeed"] = math.hypot(vx, vy, vz) if okay else 0.0
        vectors_finite &= okay
    if orientation is not None:
        values, okay = components(orientation, (0, 1, 2))
        data.update(zip(("roll", "pitch", "yaw"), values))
        vectors_finite &= okay
    if angular_velocity is not None:
        values, okay = components(angular_velocity, (0, 1, 2))
        data.update(zip(("roll_rate", "pitch_rate", "yaw_rate"), values))
        vectors_finite &= okay

    # A contract body has no grounded flag.  Preserve an explicit contact
    # field when a simulator adds one; otherwise inference below handles the
    # ground plane.
    contact = data.get("contact", data.get("grounded", getattr(value, "contact", None)))
    if data.get("grounded") is None and contact is not None:
        data["grounded"] = bool(contact)

    finite = vectors_finite and value is not None
    values: dict[str, Any] = {}
    for name in BodyState.__dataclass_fields__:
        raw = data.get(name, None if name == "grounded" else 0.0)
        if name == "grounded":
            if raw is None:
                values[name] = None
            elif isinstance(raw, bool):
                values[name] = raw
            else:
                number, okay = _number(raw)
                values[name] = bool(number) if okay else None
                finite &= okay
        else:
            values[name], okay = _number(raw)
            finite &= okay
    return BodyState(**values), finite


def _coerce_visual(value: Any) -> VisualDirectionSignal:
    if isinstance(value, VisualDirectionSignal):
        raw = value
    elif isinstance(value, Mapping):
        raw = VisualDirectionSignal(
            value.get("direction", value.get("yaw", value.get("horizontal", 0.0))),
            value.get("confidence", 1.0), value.get("valid", True),
        )
    elif value is None:
        raw = VisualDirectionSignal()
    elif hasattr(value, "optic_flow_rad_s"):
        flow = getattr(value, "optic_flow_rad_s")
        try:
            direction = flow[0]
        except (TypeError, IndexError, KeyError):
            direction = 0.0
        metadata = getattr(value, "metadata", {})
        confidence = metadata.get("confidence", 1.0) if isinstance(metadata, Mapping) else 1.0
        raw = VisualDirectionSignal(direction, confidence, True)
    else:
        raw = VisualDirectionSignal(value)
    direction, okay_d = _number(raw.direction)
    confidence, okay_c = _number(raw.confidence)
    direction = _clip(direction, -1.0, 1.0) if okay_d else 0.0
    return VisualDirectionSignal(direction, max(0.0, min(1.0, confidence)), bool(raw.valid and okay_d and okay_c))


def _coerce_target(value: TargetGroundInfo | Mapping[str, Any] | None) -> TargetGroundInfo:
    if isinstance(value, TargetGroundInfo):
        value = asdict(value)
    if value is None:
        return TargetGroundInfo()
    if not isinstance(value, Mapping):
        value = {name: getattr(value, name, None) for name in TargetGroundInfo.__dataclass_fields__}
    else:
        value = dict(value)
    # Mission runners have historically used all three spellings.  Preserve
    # the canonical field in the compact target record so phase intent is not
    # silently discarded at this boundary.
    if value.get("mission_phase") is None:
        for alias in ("phase", "phase_label", "scenario"):
            if value.get(alias) is not None:
                value["mission_phase"] = value[alias]
                break
    kwargs: dict[str, Any] = {}
    for name in TargetGroundInfo.__dataclass_fields__:
        raw = value.get(name, None if name not in ("ground_altitude", "contact") else (0.0 if name == "ground_altitude" else False))
        if name == "mission_phase":
            if raw is None:
                kwargs[name] = None
            else:
                phase = _normal_key(raw)
                kwargs[name] = phase or None
        elif name == "contact":
            if isinstance(raw, bool):
                kwargs[name] = raw
            elif isinstance(raw, str):
                kwargs[name] = raw.strip().lower() in {"1", "true", "yes", "on", "contact"}
            else:
                number, okay = _number(raw)
                kwargs[name] = bool(number) if okay else False
        elif raw is None:
            kwargs[name] = None
        else:
            number, okay = _number(raw)
            # An invalid optional target measurement must not silently become
            # zero (which could trigger an approach or landing transition).
            kwargs[name] = number if okay else (0.0 if name == "ground_altitude" else None)
    return TargetGroundInfo(**kwargs)


class LinearReadout:
    """A small fitted decoder: actuator = bias + sum(weight * channel).

    ``weights`` is keyed by actuator name, then by connectome channel.  With
    no matrix supplied, channels are matched by canonical aliases.  The class
    intentionally has no fitting routine: fitting belongs to the experiment
    that produced the decoder and should be supplied as data here.
    """

    def __init__(
        self,
        weights: Optional[Mapping[str, Mapping[str, float]]] = None,
        bias: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.weights = {str(k): dict(v) for k, v in (weights or {}).items()}
        self.bias = {str(k): _number(v)[0] for k, v in (bias or {}).items()}

    def apply(self, channels: Mapping[str, float]) -> ReadoutOutput:
        result: dict[str, float] = {}
        finite = True
        for actuator in _ACTUATORS:
            if actuator in self.weights:
                value = self.bias.get(actuator, 0.0)
                for channel, weight in self.weights[actuator].items():
                    x, okay_x = _number(channels.get(channel, 0.0))
                    w, okay_w = _number(weight)
                    value += x * w
                    finite &= okay_x and okay_w
            else:
                value = self.bias.get(actuator, 0.0)
                for alias in _ALIASES[actuator]:
                    if alias in channels:
                        value, okay = _number(channels[alias])
                        finite &= okay
                        break
            result[actuator] = value
            finite &= math.isfinite(value)
        return ReadoutOutput(result, source=SourceLabel.FITTED.value, finite=finite)


# ``FittedReadout`` is the terminology used in the experiment notes.
FittedReadout = LinearReadout


class SafetyLayer:
    """Hand-built stall/divergence protection; independently disable-able.

    Stall output is a bounded overlay selected by ``AutopilotConfig``;
    numeric divergence is the only full fail-safe replacement.
    """

    def __init__(self, config: AutopilotConfig, enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled

    def decide(
        self,
        body: BodyState,
        body_finite: bool,
        connectome: ConnectomeOutput,
        readout: ReadoutOutput,
        airborne: bool,
    ) -> SafetyDecision:
        values = tuple(readout.actuators.get(name, 0.0) for name in _ACTUATORS)
        readout_finite = True
        for value in values:
            _, okay = _number(value)
            readout_finite &= okay
        divergence = (
            not body_finite
            or not connectome.finite
            or not readout.finite
            or not readout_finite
            or abs(body.altitude) > self.config.max_altitude
            or abs(body.airspeed) > self.config.max_speed
            or abs(body.vertical_speed) > self.config.max_speed
            or max(abs(body.yaw_rate), abs(body.pitch_rate), abs(body.roll_rate)) > self.config.max_rate
            # Yaw is a wrapped heading and may legitimately span 2*pi; the
            # attitude envelope applies to pitch/roll only.
            or max(abs(body.pitch), abs(body.roll)) > self.config.max_angle
        )
        if self.enabled and divergence:
            return SafetyDecision(
                True, "numeric_divergence", {name: 0.0 for name in _ACTUATORS},
                SourceLabel.SAFETY_HANDBUILT.value,
                self.config.numeric_safety_blend,
            )
        stall = airborne and body.airspeed < self.config.stall_airspeed
        if self.enabled and stall:
            # The only synthesized behaviour is a stall escape: full thrust,
            # symmetric wing drive, and neutral attitude/heading channels.
            # It is blended with the fitted readout rather than replacing it;
            # the bounded blend is recorded on both the decision and command.
            return SafetyDecision(
                True,
                "stall",
                {"left_wing": 1.0, "right_wing": 1.0, "thrust": 1.0,
                 "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "brake": 0.0},
                SourceLabel.SAFETY_HANDBUILT.value,
                self.config.safety_blend,
            )
        return SafetyDecision(False, "", {}, SourceLabel.SAFETY_HANDBUILT.value)


class FlightAutopilot:
    """Stateful state machine around a connectome motor readout."""

    def __init__(
        self,
        readout: Optional[LinearReadout] = None,
        config: Optional[AutopilotConfig] = None,
        *,
        safety_enabled: bool = True,
        logger: Optional[Callable[[FlightLogEntry], None]] = None,
    ) -> None:
        self.config = config or AutopilotConfig()
        self.readout = readout or LinearReadout()
        self.safety = SafetyLayer(self.config, enabled=safety_enabled)
        self.state = FlightMode.GROUNDED
        self.previous_mode = self.state
        self.time_s = 0.0
        self.state_elapsed_s = 0.0
        self.sequence = 0
        self.recover_return_mode = FlightMode.CRUISE
        self._stall_elapsed_s = 0.0
        self._clear_elapsed_s = 0.0
        self._airborne_latched = False
        # A takeoff request can spend multiple control frames on the ground
        # while the neural readout spools up. Remember whether this TAKEOFF
        # dwell has actually observed liftoff so a later touchdown cannot be
        # mistaken for another ground run-up merely because thrust is still
        # high.
        self._takeoff_airborne_seen = False
        self._logs: list[FlightLogEntry] = []
        self._logger = logger
        self.last_connectome_output = ConnectomeOutput({})
        self.last_readout_output = ReadoutOutput({name: 0.0 for name in _ACTUATORS})
        self.last_safety_decision = SafetyDecision()
        self.last_command = MotorCommand()

    @property
    def mode(self) -> FlightMode:
        return self.state

    @property
    def safety_enabled(self) -> bool:
        """Whether stall/numeric protective overrides are enabled."""

        return self.safety.enabled

    @safety_enabled.setter
    def safety_enabled(self, value: bool) -> None:
        self.safety.enabled = bool(value)

    @property
    def logs(self) -> tuple[FlightLogEntry, ...]:
        return tuple(self._logs)

    def drain_logs(self) -> list[FlightLogEntry]:
        entries = self._logs[:]
        self._logs.clear()
        return entries

    def reset(self) -> None:
        self.__init__(self.readout, self.config, safety_enabled=self.safety.enabled, logger=self._logger)

    @staticmethod
    def _channels(value: Any) -> ConnectomeOutput:
        if isinstance(value, ConnectomeOutput):
            if not isinstance(value.channels, Mapping):
                return ConnectomeOutput({}, finite=False)
            source = value.channels
            inherited_finite = bool(value.finite)
        else:
            inherited_finite = True
            if value is None:
                return ConnectomeOutput({})
            if isinstance(value, Mapping):
                source = dict(value)
            elif isinstance(getattr(value, "named_outputs", None), Mapping):
                # Adapt the unified NeuralOutput protocol without importing
                # twin_contracts.  Its named_outputs are the only channels
                # that have actuator semantics at this boundary.
                source = dict(value.named_outputs)
                try:
                    inherited_finite = _source_value(
                        getattr(value, "source", SourceLabel.CONNECTOME),
                        SourceLabel.CONNECTOME,
                    ) == SourceLabel.CONNECTOME.value
                except ValueError:
                    inherited_finite = False
            elif isinstance(getattr(value, "channels", None), Mapping):
                source = dict(value.channels)
            else:
                try:
                    count = len(value)
                except (TypeError, AttributeError):
                    return ConnectomeOutput({}, finite=False)
                source = {name: (value[index] if index < count else 0.0)
                          for index, name in enumerate(_ACTUATORS)}
        channels: dict[str, float] = {}
        finite = inherited_finite
        for key, raw in source.items():
            number, okay = _number(raw)
            channels[str(key)] = number
            finite &= okay
        return ConnectomeOutput(channels, finite=finite)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _airborne(self, body: BodyState, target: TargetGroundInfo) -> bool:
        if body.grounded is not None:
            self._airborne_latched = not body.grounded
            return self._airborne_latched

        # Use separate thresholds for liftoff and touchdown.  This prevents
        # noisy altitude/speed measurements from chattering between GROUNDED
        # and TAKEOFF around the contact plane.
        lift_altitude = target.ground_altitude + max(0.0, self.config.ground_altitude_hysteresis)
        lift_speed = max(0.0, self.config.airborne_speed_threshold)
        if self._airborne_latched:
            touch_altitude = target.ground_altitude + 0.5 * max(0.0, self.config.ground_altitude_hysteresis)
            touch_speed = 0.5 * lift_speed
            if body.altitude <= touch_altitude and body.airspeed <= touch_speed:
                self._airborne_latched = False
        elif body.altitude > lift_altitude or body.airspeed > lift_speed:
            self._airborne_latched = True
        return self._airborne_latched

    def _turn_demand(self, visual: VisualDirectionSignal, body: BodyState, target: TargetGroundInfo) -> bool:
        visual_demand = visual.valid and visual.confidence >= self.config.turn_confidence and abs(visual.direction) >= self.config.turn_enter_threshold
        heading_demand = False
        if target.target_heading is not None and math.isfinite(target.target_heading):
            heading_demand = abs(self._wrap_angle(target.target_heading - body.yaw)) >= self.config.heading_turn_threshold
        return visual_demand or heading_demand

    def _turn_clear(self, visual: VisualDirectionSignal, body: BodyState, target: TargetGroundInfo) -> bool:
        visual_clear = (not visual.valid) or visual.confidence < self.config.turn_confidence or abs(visual.direction) <= self.config.turn_exit_threshold
        heading_clear = True
        if target.target_heading is not None and math.isfinite(target.target_heading):
            heading_clear = abs(self._wrap_angle(target.target_heading - body.yaw)) <= self.config.heading_turn_threshold * 0.65
        return visual_clear and heading_clear

    def _target_altitude(self, target: TargetGroundInfo) -> float:
        if target.target_altitude is None or not math.isfinite(target.target_altitude):
            return self.config.cruise_altitude
        return target.target_altitude

    def _hover_stable(self, body: BodyState, target: TargetGroundInfo) -> bool:
        """Return whether the measured body is in the stable hover envelope.

        Hover is a state-machine observation, not a hand-built actuator
        command.  In particular, low airspeed is expected here and must not be
        fed to the stall safety detector.  The altitude check uses the same
        target/hysteresis pair as takeoff so transitions are deterministic at
        the boundary.
        """

        altitude_error = abs(body.altitude - self._target_altitude(target))
        altitude_tolerance = max(0.0, self.config.cruise_altitude_hysteresis)
        speed_limit = max(0.0, self.config.hover_airspeed_threshold)
        vertical_limit = max(0.0, self.config.hover_vertical_speed_threshold)
        return (
            math.isfinite(altitude_error)
            and altitude_error <= altitude_tolerance
            and math.isfinite(body.airspeed)
            and body.airspeed <= speed_limit
            and math.isfinite(body.vertical_speed)
            and abs(body.vertical_speed) <= vertical_limit
        )

    def _state_timeout(self, mode: FlightMode) -> float:
        """Return the active dwell timeout, with a global safety cap.

        A non-positive timeout disables that particular limit.  The global
        limit is a cap for modes with a per-mode timeout and is also used for
        RECOVER; CRUISE is intentionally not timed out because there is no
        safe hand-built command to select when a target is absent.
        """

        specific = {
            FlightMode.TAKEOFF: self.config.takeoff_timeout_s,
            FlightMode.TURN: self.config.turn_timeout_s,
            FlightMode.APPROACH: self.config.approach_timeout_s,
            FlightMode.LAND: self.config.land_timeout_s,
            FlightMode.RECOVER: self.config.recover_timeout_s,
        }.get(mode, 0.0)
        candidates = [value for value in (specific, self.config.state_timeout_s)
                      if math.isfinite(value) and value > 0.0]
        return min(candidates) if candidates else math.inf

    def _approach_demand(self, target: TargetGroundInfo) -> bool:
        threshold = self.config.approach_distance if target.approach_distance is None else target.approach_distance
        return (
            target.target_distance is not None
            and math.isfinite(target.target_distance)
            and math.isfinite(threshold)
            and target.target_distance <= threshold
        )

    def _approach_clear(self, target: TargetGroundInfo) -> bool:
        threshold = self.config.approach_exit_distance
        if target.approach_distance is not None and math.isfinite(target.approach_distance):
            threshold = max(threshold, target.approach_distance + 0.1)
        return (
            target.target_distance is not None
            and math.isfinite(target.target_distance)
            and math.isfinite(threshold)
            and target.target_distance > threshold
        )

    def _landing_demand(self, body: BodyState, target: TargetGroundInfo) -> bool:
        distance = self.config.landing_distance if target.landing_distance is None else target.landing_distance
        altitude = self.config.landing_altitude if target.landing_altitude is None else target.landing_altitude
        near_distance = (
            target.target_distance is not None
            and math.isfinite(target.target_distance)
            and math.isfinite(distance)
            and target.target_distance <= distance
        )
        near_ground = (
            math.isfinite(body.altitude)
            and math.isfinite(target.ground_altitude)
            and math.isfinite(altitude)
            and body.altitude <= target.ground_altitude + altitude
        )
        return target.contact or near_distance or near_ground

    def _phase_mode(
        self,
        body: BodyState,
        target: TargetGroundInfo,
        airborne: bool,
        takeoff_thrust_active: bool,
    ) -> Optional[FlightMode]:
        """Map an explicit mission phase to telemetry mode intent.

        The world/mission clock is authoritative for intent when present, but
        measured contact remains authoritative for whether the vehicle can be
        reported as airborne.  Unknown phases intentionally return ``None``
        so older standalone callers retain target/visual behaviour.
        """

        phase = "" if target.mission_phase is None else _normal_key(target.mission_phase)
        if not phase:
            return None
        if phase == "takeoff":
            return FlightMode.TAKEOFF if airborne or takeoff_thrust_active else FlightMode.GROUNDED
        if phase in {"forward", "cruise", "flight", "translation"}:
            return FlightMode.CRUISE if airborne else FlightMode.GROUNDED
        if phase in {"turn", "yaw", "yaw_step", "yaw_ramp"}:
            return FlightMode.TURN if airborne else FlightMode.GROUNDED
        if phase in {"hover", "gust", "gust_recovery", "wind", "hold"}:
            # RECOVER remains reserved for a confirmed safety event; a gust
            # phase is still an airborne hold intent and must not fabricate a
            # protective command merely because its label contains “recover”.
            return FlightMode.HOVER if airborne else FlightMode.GROUNDED
        if phase in {"approach", "descent"}:
            return FlightMode.APPROACH if airborne else FlightMode.GROUNDED
        if phase in {"landing", "land", "touchdown", "settle", "grounded"}:
            return FlightMode.LAND if airborne else FlightMode.GROUNDED
        return None

    def _transition(self, new_state: FlightMode) -> str:
        if new_state is self.state:
            return ""
        old = self.state
        self.previous_mode = old
        self.state = new_state
        self.state_elapsed_s = 0.0
        if new_state is FlightMode.TAKEOFF:
            self._takeoff_airborne_seen = False
        return f"{old.value}->{new_state.value}"

    def _update_state(
        self,
        body: BodyState,
        target: TargetGroundInfo,
        visual: VisualDirectionSignal,
        neural_thrust: float,
        neural_thrust_valid: bool,
        airborne: bool,
        safety: SafetyDecision,
        dt_s: float,
    ) -> str:
        self.state_elapsed_s += dt_s

        takeoff_threshold, takeoff_threshold_valid = _number(
            self.config.takeoff_thrust_threshold,
        )
        takeoff_thrust_active = (
            neural_thrust_valid
            and takeoff_threshold_valid
            and neural_thrust >= takeoff_threshold
        )
        phase_mode = self._phase_mode(
            body, target, airborne, takeoff_thrust_active,
        )

        # Protective overrides are the only hand-built route into RECOVER.
        # Numeric divergence is retained even when the coerced body can no
        # longer establish airborne status; a non-grounded controller must
        # not silently continue with stale fitted output.
        if safety.active and safety.reason in ("stall", "numeric_divergence"):
            if self.state is not FlightMode.GROUNDED:
                if self.state is not FlightMode.RECOVER:
                    self.recover_return_mode = self.state
                return self._transition(FlightMode.RECOVER)
            if airborne:
                self.recover_return_mode = FlightMode.TAKEOFF
                return self._transition(FlightMode.RECOVER)

        if self.state is FlightMode.RECOVER:
            # Do not clear a numeric recovery while any input remains
            # non-finite.  For a stall, use a higher exit threshold than the
            # entry threshold (hysteresis), then require a dwell interval.
            if safety.active and safety.reason == "numeric_divergence":
                self._clear_elapsed_s = 0.0
                return ""
            if not airborne:
                self._clear_elapsed_s = 0.0
                return self._transition(FlightMode.GROUNDED)
            if body.airspeed < self.config.stall_recover_airspeed:
                self._clear_elapsed_s = 0.0
                return ""
            self._clear_elapsed_s += dt_s
            clear_time = max(0.0, self.config.recover_clear_s)
            if self._clear_elapsed_s >= clear_time:
                self._clear_elapsed_s = 0.0
                if phase_mode is not None:
                    return self._transition(phase_mode)
                return self._transition(self.recover_return_mode)
            # A recover timeout is a telemetry/event boundary, not permission
            # to fabricate a landing command.  Stay in RECOVER until either
            # the body is grounded or the envelope clears.
            return ""

        # A mission runner may provide a phase clock alongside the measured
        # body state.  Honor that explicit intent before target visibility,
        # visual heading cues, and target-heading heuristics.  Contact still
        # wins: a grounded body cannot be reported as an airborne mode, and a
        # landing/settle phase cannot re-arm TAKEOFF from residual thrust.
        if phase_mode is not None:
            if phase_mode is FlightMode.TAKEOFF:
                # Let the normal TAKEOFF progression reach HOVER/CRUISE
                # during a long takeoff phase; only use the explicit phase to
                # arm a grounded controller or recover from a stale ground
                # state.  Otherwise every frame would reset HOVER back to
                # TAKEOFF while the body is already airborne.
                if airborne and self.state is FlightMode.GROUNDED:
                    self._takeoff_airborne_seen = True
                    return self._transition(FlightMode.TAKEOFF)
                # Ground run-up remains allowed only during an explicit
                # takeoff phase; the GROUNDED branch below handles it.
            else:
                if not airborne:
                    return self._transition(FlightMode.GROUNDED)
                return self._transition(phase_mode)

        if self.state is FlightMode.GROUNDED:
            if airborne or takeoff_thrust_active:
                transition = self._transition(FlightMode.TAKEOFF)
                if airborne:
                    self._takeoff_airborne_seen = True
                return transition
            return ""

        # TAKEOFF has a deliberate ground-run-up dwell: while the body is
        # still grounded, a finite neural thrust command at/above the
        # configured threshold keeps the mode in TAKEOFF. This is telemetry
        # only; actuator coordinates continue to come from the readout below.
        # Once liftoff has been observed, however, a grounded sample is a real
        # touchdown even if thrust remains high, so landing can still return
        # to GROUNDED.
        if self.state is FlightMode.TAKEOFF and not airborne:
            if self._takeoff_airborne_seen or not takeoff_thrust_active:
                return self._transition(FlightMode.GROUNDED)
            if self.state_elapsed_s >= self._state_timeout(FlightMode.TAKEOFF):
                self.recover_return_mode = FlightMode.TAKEOFF
                return self._transition(FlightMode.RECOVER)
            return ""

        if self.state is FlightMode.TAKEOFF and airborne:
            self._takeoff_airborne_seen = True

        # Every airborne mode except RECOVER returns to GROUNDED only after
        # the hysteretic airborne detector says the body is down. ``contact``
        # alone is not used to rewrite the state while still above ground.
        if not airborne:
            return self._transition(FlightMode.GROUNDED)

        if self.state is FlightMode.TAKEOFF:
            cruise_hysteresis = max(0.0, self.config.cruise_altitude_hysteresis)
            if (
                body.altitude >= self._target_altitude(target) - cruise_hysteresis
                and body.vertical_speed >= self.config.climb_speed_threshold
            ):
                return self._transition(
                    FlightMode.HOVER if self._hover_stable(body, target) else FlightMode.CRUISE
                )
            if self.state_elapsed_s >= self._state_timeout(FlightMode.TAKEOFF):
                self.recover_return_mode = FlightMode.TAKEOFF
                return self._transition(FlightMode.RECOVER)
            return ""

        if self.state is FlightMode.CRUISE:
            if self._approach_demand(target):
                return self._transition(FlightMode.APPROACH)
            if self._turn_demand(visual, body, target):
                return self._transition(FlightMode.TURN)
            if self._hover_stable(body, target):
                return self._transition(FlightMode.HOVER)
            return ""
        if self.state is FlightMode.HOVER:
            # A hover target can be near a landing marker without requesting a
            # descent (the runner's hover scene is intentionally close to its
            # marker).  Only leave a stable hover for approach when the
            # landing envelope itself is reached; cruise/turn still honour
            # the wider approach distance.
            if self._landing_demand(body, target):
                return self._transition(FlightMode.APPROACH)
            if self._turn_demand(visual, body, target):
                return self._transition(FlightMode.TURN)
            if not self._hover_stable(body, target):
                return self._transition(FlightMode.CRUISE)
            return ""
        if self.state is FlightMode.TURN:
            if self._approach_demand(target):
                return self._transition(FlightMode.APPROACH)
            if self._turn_clear(visual, body, target):
                return self._transition(
                    FlightMode.HOVER if self._hover_stable(body, target) else FlightMode.CRUISE
                )
            if self.state_elapsed_s >= self._state_timeout(FlightMode.TURN):
                return self._transition(FlightMode.CRUISE)
            return ""

        if self.state is FlightMode.APPROACH:
            if self._landing_demand(body, target):
                return self._transition(FlightMode.LAND)
            if self._approach_clear(target):
                return self._transition(
                    FlightMode.HOVER if self._hover_stable(body, target) else FlightMode.CRUISE
                )
            # Keep APPROACH latched if the target briefly moves away; this is
            # the approach-side hysteresis.  The timeout still guarantees a
            # deterministic hand-off to LAND for a finite scenario.
            if self.state_elapsed_s >= self._state_timeout(FlightMode.APPROACH):
                return self._transition(FlightMode.LAND)
            return ""

        if self.state is FlightMode.LAND:
            near_ground = (
                body.altitude <= target.ground_altitude + max(0.0, self.config.ground_altitude_hysteresis)
                and body.airspeed <= self.config.landing_speed_threshold
            )
            if target.contact or near_ground:
                return self._transition(FlightMode.GROUNDED)
            if self.state_elapsed_s >= self._state_timeout(FlightMode.LAND):
                self.recover_return_mode = FlightMode.LAND
                return self._transition(FlightMode.RECOVER)
        return ""

    def _bounded(self, actuators: Mapping[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in _ACTUATORS:
            value, okay = _number(actuators.get(name, 0.0))
            low, high = (-1.0, 1.0) if name in _SIGNED else (0.0, 1.0)
            result[name] = _clip(value if okay else 0.0, low, high)
        return result

    def step(
        self,
        neural_motor_channels: Mapping[str, Any] | Sequence[Any] | ConnectomeOutput | None,
        visual_direction_signal: Any,
        body_state: Any,
        target_ground_info: TargetGroundInfo | Mapping[str, Any] | None = None,
        *,
        dt_s: float = 0.02,
    ) -> MotorCommand:
        """Consume one observation and return one bounded motor command."""

        dt, okay_dt = _number(dt_s)
        dt_valid = okay_dt and 0.0 < dt <= 10.0
        if not dt_valid:
            dt = 0.02
        body, body_finite = _coerce_body(body_state)
        target = _coerce_target(target_ground_info)
        visual = _coerce_visual(visual_direction_signal)
        connectome = self._channels(neural_motor_channels)
        readout = self.readout.apply(connectome.channels)
        airborne = self._airborne(body, target)
        # A deliberate flare/landing can be below the stall threshold.  Once
        # the state machine has an explicit landing demand, do not mistake
        # that expected low speed for an emergency stall.  Numeric divergence
        # remains protected in all modes.
        landing_proximity = (
            self.state in (FlightMode.APPROACH, FlightMode.LAND)
            and self._landing_demand(body, target)
        )
        # Stall protection is armed only after takeoff has established a
        # normal-flight state.  ``HOVER``/``APPROACH``/``LAND`` deliberately
        # allow low airspeed, while a first airborne sample must be allowed to
        # enter TAKEOFF without an immediate safety override.
        stall_monitoring = self.state in (FlightMode.CRUISE, FlightMode.TURN)
        stall_airborne = airborne and stall_monitoring and not landing_proximity
        if stall_airborne and body.airspeed < self.config.stall_airspeed:
            self._stall_elapsed_s += dt
        else:
            self._stall_elapsed_s = 0.0
        # The state-aware arming rule handles the initial takeoff sample; the
        # grace period additionally prevents a normal CRUISE->HOVER slowdown
        # from immediately entering recovery.
        stall_decision_allowed = self._stall_elapsed_s >= max(0.0, self.config.stall_grace_s)
        safety = self.safety.decide(
            body, body_finite and dt_valid, connectome, readout,
            stall_airborne and stall_decision_allowed,
        )
        previous_mode = self.state
        neural_thrust, neural_thrust_valid = _number(readout.actuators.get("thrust", 0.0))
        transition = self._update_state(
            body, target, visual, neural_thrust, neural_thrust_valid,
            airborne, safety, dt,
        )
        # Re-evaluate the command only through the optional safety layer.  The
        # state machine itself never writes actuator values in normal flight.
        source = readout.source
        readout_bounded = self._bounded(readout.actuators)
        safety_bounded = self._bounded(safety.actuators) if safety.active else {}
        safety_blend = safety.blend if safety.active else 0.0
        # SafetyDecision.__post_init__ clamps the configured weight, but keep
        # this boundary defensive for custom decision objects/subclasses.
        safety_blend = max(0.0, min(1.0, safety_blend))
        actuators: Mapping[str, Any] = readout_bounded
        if safety.active:
            actuators = {
                name: ((1.0 - safety_blend) * readout_bounded[name]
                       + safety_blend * safety_bounded[name])
                for name in _ACTUATORS
            }
            source = safety.source
        bounded = self._bounded(actuators)
        self.time_s += dt
        self.sequence += 1
        self.last_connectome_output = connectome
        self.last_readout_output = readout
        self.last_safety_decision = safety
        command = MotorCommand(
            **bounded,
            mode=self.state,
            source=source,
            safety_active=safety.active,
            safety_reason=safety.reason,
            sequence=self.sequence,
            neural_source=connectome.source,
            readout_source=readout.source,
            safety_source=safety.source,
            safety_blend=safety_blend,
            neural_actuators=readout_bounded,
            safety_actuators=safety_bounded,
            provenance={
                "neural": connectome.source,
                "cns": connectome.source,
                "readout": readout.source,
                "safety": safety.source if safety.active else "none",
                "command": source,
            },
        )
        self.last_command = command
        entry = FlightLogEntry(
            self.sequence, self.time_s, self.state, previous_mode, transition,
            safety.active, safety.reason, source, connectome.finite, readout.finite,
            connectome_source=connectome.source,
            readout_source=readout.source,
            mode_elapsed_s=self.state_elapsed_s,
            safety_blend=safety_blend,
            neural_source=connectome.source,
            safety_source=safety.source,
            provenance=command.provenance,
        )
        self._logs.append(entry)
        if self._logger is not None:
            self._logger(entry)
        return command

    # Friendly aliases for callers that use control/update terminology.
    update = step
    command = step


# Names used by a few small integrations and by older experiment notebooks.
FlightController = FlightAutopilot
Autopilot = FlightAutopilot
Body = BodyState
TargetInfo = TargetGroundInfo
VisualSignal = VisualDirectionSignal
Mode = FlightMode


def scenario_self_test() -> dict[str, Any]:
    """Run a short deterministic smoke scenario without external packages."""

    config = AutopilotConfig(recover_clear_s=0.04, stall_grace_s=0.04)
    controller = FlightAutopilot(config=config, safety_enabled=True)
    grounded = BodyState(altitude=0.0, airspeed=0.0, grounded=True)
    idle = controller.step({}, {}, grounded, TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    takeoff = controller.step({"left": 0.7, "right": 0.7, "thrust": 0.8}, {}, grounded,
                               TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    airborne = BodyState(altitude=1.0, airspeed=0.5, vertical_speed=0.1, grounded=False)
    cruise = controller.step({"left": 0.4, "right": 0.5, "thrust": 0.6}, {}, airborne,
                              TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    hover = controller.step({"left": 0.4, "right": 0.5, "thrust": 0.6}, {},
                            BodyState(altitude=1.0, airspeed=0.02, vertical_speed=0.0, grounded=False),
                            TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    turn = controller.step({"left": 0.4, "right": 0.5, "thrust": 0.6}, {"direction": 0.8}, airborne,
                            TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    cruise_after_turn = controller.step({"left": 0.4, "right": 0.5, "thrust": 0.6}, {"direction": 0.0}, airborne,
                                         TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    approach = controller.step({"left": 0.2, "right": 0.2, "thrust": 0.4}, {}, airborne,
                               TargetGroundInfo(target_altitude=1.0, target_distance=1.0), dt_s=0.02)
    land = controller.step({"left": 0.2, "right": 0.2, "thrust": 0.2}, {},
                           BodyState(altitude=0.1, airspeed=0.05, grounded=False),
                           TargetGroundInfo(target_distance=0.1), dt_s=0.02)
    final = controller.step({}, {}, BodyState(altitude=0.0, airspeed=0.0, grounded=True),
                            TargetGroundInfo(contact=True), dt_s=0.02)
    # The recovery branch is an explicit ablation of the grace period; the
    # normal controller above retains its short takeoff/hover grace window.
    recovery_controller = FlightAutopilot(
        config=AutopilotConfig(recover_clear_s=config.recover_clear_s, stall_grace_s=0.0),
        safety_enabled=True,
    )
    recovery_controller.step({"thrust": 0.8}, {}, grounded,
                              TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    recovery_controller.step({"thrust": 0.6}, {}, airborne,
                              TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    recovery = recovery_controller.step({"thrust": 0.6}, {},
                                         BodyState(altitude=1.0, airspeed=0.01, grounded=False),
                                         TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    recovery_clear_1 = recovery_controller.step({"thrust": 0.6}, {},
                                                 BodyState(altitude=1.0, airspeed=0.25, grounded=False),
                                                 TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    recovery_clear_2 = recovery_controller.step({"thrust": 0.6}, {},
                                                 BodyState(altitude=1.0, airspeed=0.25, grounded=False),
                                                 TargetGroundInfo(target_altitude=1.0), dt_s=0.02)
    modes = [idle.mode.value, takeoff.mode.value, cruise.mode.value, hover.mode.value, turn.mode.value,
             cruise_after_turn.mode.value, approach.mode.value, land.mode.value, final.mode.value]
    recovery_modes = [recovery.mode.value, recovery_clear_1.mode.value, recovery_clear_2.mode.value]
    expected = {mode.value for mode in FlightMode}
    assert expected.issubset(set(modes + recovery_modes))
    assert recovery.safety_active and recovery.source == SourceLabel.SAFETY_HANDBUILT.value
    assert all(math.isfinite(v) for v in final.actuator_dict().values())
    for command in (idle, takeoff, cruise, hover, turn, cruise_after_turn, approach, land, final,
                    recovery, recovery_clear_1, recovery_clear_2):
        assert all(
            (-1.0 <= value <= 1.0 if name in _SIGNED else 0.0 <= value <= 1.0)
            for name, value in command.actuator_dict().items()
        )
    return {
        "modes": modes,
        "recovery_modes": recovery_modes,
        "records": len(controller.logs),
        "final": final.as_dict(),
        "safety_source": recovery.source,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(scenario_self_test(), indent=2))
