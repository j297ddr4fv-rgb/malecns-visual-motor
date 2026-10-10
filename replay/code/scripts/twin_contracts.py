#!/usr/bin/env python3
"""Small, dependency-light contracts for the digital-fly closed loop.

This module deliberately contains data contracts rather than a simulator.  It
is safe to import from CPU-only tools: numpy is optional and no project
runtime (torch, CUDA, flyvis, ...) is imported here.

Canonical units
---------------
Simulation time is seconds; neural ``dt`` is commonly 0.0005 s (0.5 ms).
Positions and linear velocities are metres and metres/second.  Angles are
radians and angular velocities are radians/second.  Luminance is a unitless
float in [0, 1] when normalised, and neural rates are Hz.  A timestamp carries
both a domain tick and its rate, so a 2 kHz neural tick must not be confused
with a 200 Hz control tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Dict, Iterable, Optional, Protocol, Tuple, runtime_checkable


NEURAL_DT_S = 0.0005
DEFAULT_RATES_HZ = {
    "neural": 2000.0,
    "control": 200.0,
    "render": 25.0,
}


class ContractError(ValueError):
    """Raised when a value cannot satisfy a twin contract."""


class SourceLabel(str, Enum):
    """Provenance of a signal or model component.

    ``connectome`` means directly derived from the MaleCNS graph or its
    calibrated neuron dynamics.  ``fitted`` means estimated from data (for
    example a linear wing readout).  ``handbuilt`` means an explicit glue
    model (for example EMD vision or the point-mass body).
    """

    CONNECTOME = "connectome"
    FITTED = "fitted"
    HANDBUILT = "handbuilt"


# A short alias makes the intent obvious at call sites while keeping one enum
# and one serialised vocabulary.
SourceTag = SourceLabel


class Unit(str, Enum):
    """Canonical units used by the contracts.

    Field names carry the unit suffix as well (``position_m``, ``rates_hz``,
    ...).  Keeping this small vocabulary in one place makes adapters less
    likely to silently mix neural ticks and body/control seconds.
    """

    SECOND = "s"
    HERTZ = "Hz"
    TICK = "tick"
    METRE = "m"
    METRE_PER_SECOND = "m/s"
    METRE_PER_SECOND2 = "m/s^2"
    RADIAN = "rad"
    RADIAN_PER_SECOND = "rad/s"
    PIXEL = "pixel"
    NORMALIZED = "1"
    COUNT = "count"


# Public, immutable-by-convention unit maps.  Record classes expose the
# corresponding map through ``units``/``unit_map`` properties without adding
# a noisy field to every serialised record.
TIME_UNITS = {
    "t_s": Unit.SECOND.value,
    "tick": Unit.TICK.value,
    "rate_hz": Unit.HERTZ.value,
}
BODY_UNITS = {
    "position_m": Unit.METRE.value,
    "orientation_rad": Unit.RADIAN.value,
    "velocity_mps": Unit.METRE_PER_SECOND.value,
    "angular_velocity_rad_s": Unit.RADIAN_PER_SECOND.value,
}
VISION_UNITS = {
    "pixels": Unit.NORMALIZED.value,
    "optic_flow_rad_s": Unit.RADIAN_PER_SECOND.value,
}
NEURAL_UNITS = {
    "rates_hz": Unit.HERTZ.value,
    "spikes": Unit.COUNT.value,
}
MOTOR_UNITS = {
    "yaw_rate_rad_s": Unit.RADIAN_PER_SECOND.value,
    "pitch_rate_rad_s": Unit.RADIAN_PER_SECOND.value,
    "roll_rate_rad_s": Unit.RADIAN_PER_SECOND.value,
    "thrust_mps2": Unit.METRE_PER_SECOND2.value,
}

# Backwards/forwards-compatible names used by small adapters.
Source = SourceLabel
Provenance = SourceLabel


def _source(value: SourceLabel | str) -> SourceLabel:
    try:
        return value if isinstance(value, SourceLabel) else SourceLabel(str(value))
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "source must be one of: connectome, fitted, handbuilt"
        ) from exc


def _finite(value: Any, name: str) -> float:
    # Textual numbers are deliberately rejected.  Accepting ``"0.5"`` here
    # makes a malformed JSON/configuration value indistinguishable from a
    # properly typed measurement at the contract boundary.
    if isinstance(value, (bool, str, bytes)):
        raise ContractError(f"{name} must be a finite number, not {type(value).__name__}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractError(f"{name} must be finite")
    return out


def _nonnegative_int(value: Any, name: str) -> int:
    """Validate and canonicalise a non-negative integer field."""
    if isinstance(value, bool):
        raise ContractError(f"{name} must be a non-negative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be a non-negative integer") from exc
    try:
        exact = float(value) == float(integer)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or integer < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return integer


def _vector(value: Any, n: int, name: str) -> Tuple[float, ...]:
    """Convert a vector-like value to a finite tuple of exactly ``n`` items."""
    if isinstance(value, (str, bytes)):
        raise ContractError(f"{name} must have shape ({n},), got scalar text")
    try:
        vals = tuple(value)
    except TypeError as exc:
        raise ContractError(f"{name} must have shape ({n},)") from exc
    if len(vals) != n:
        raise ContractError(f"{name} must have shape ({n},), got ({len(vals)},)")
    return tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(vals))


def _array_shape(value: Any) -> Tuple[int, ...]:
    """Return a shape without requiring numpy.

    Numpy arrays expose ``shape`` directly.  For nested Python sequences we
    recursively require rectangular dimensions; ragged data is a contract
    error rather than silently becoming an object array.
    """
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            result = tuple(int(x) for x in shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError("array shape is not valid") from exc
        if any(x < 0 for x in result):
            raise ContractError("array shape cannot contain negative dimensions")
        return result
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    vals = list(value)
    if not vals:
        return (0,)
    first = _array_shape(vals[0])
    for item in vals[1:]:
        if _array_shape(item) != first:
            raise ContractError("array-like value is ragged")
    return (len(vals),) + first


def _check_array(value: Any, name: str, *, rank: Optional[int] = None,
                 exact_shape: Optional[Tuple[int, ...]] = None,
                 nonnegative: bool = False) -> Tuple[int, ...]:
    if value is None:
        raise ContractError(f"{name} cannot be None")
    shape = _array_shape(value)
    if rank is not None and len(shape) != rank:
        raise ContractError(f"{name} must have rank {rank}, got shape {shape}")
    if exact_shape is not None and shape != exact_shape:
        raise ContractError(f"{name} must have shape {exact_shape}, got {shape}")

    # NumPy arrays are the hot path for the real neural engine (one vector can
    # contain hundreds of thousands of neurons).  Keep NumPy optional, but
    # when an array protocol is available let NumPy validate the whole buffer
    # at once instead of converting every scalar through ``_finite`` below.
    # Lists and other ordinary iterables deliberately retain the dependency-
    # free recursive path, including its ragged-shape checks above.
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None

    if np is not None:
        has_array_protocol = any(
            hasattr(value, attr)
            for attr in ("__array__", "__array_interface__", "__array_struct__")
        )
        if isinstance(value, np.ndarray) or has_array_protocol:
            try:
                array = np.asarray(value)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ContractError(f"{name} must be a numeric array") from exc

            # An array-like object can advertise a misleading shape.  Use the
            # converted array as the source of truth and fail closed rather
            # than validating a different shape than the caller supplied.
            array_shape = tuple(int(dim) for dim in array.shape)
            if array_shape != shape:
                raise ContractError(f"{name} must have shape {shape}, got {array_shape}")

            # ``np.isfinite`` does not support object/string/structured dtypes
            # consistently, and NumPy bool scalars would otherwise be accepted
            # even though Python bool values are rejected by ``_finite``.
            # Rejecting these dtypes is both explicit and stricter than the
            # legacy scalar path for object and bool arrays.
            dtype_kind = array.dtype.kind
            if dtype_kind not in "iuf":
                raise ContractError(
                    f"{name} must contain real numeric values, got dtype {array.dtype}"
                )
            try:
                finite = np.isfinite(array)
                all_finite = bool(np.all(finite))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ContractError(f"{name} must contain finite numeric values") from exc
            if not all_finite:
                raise ContractError(f"{name} must contain finite numeric values")
            if nonnegative:
                try:
                    has_negative = bool(np.any(array < 0))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ContractError(f"{name} must contain non-negative numeric values") from exc
                if has_negative:
                    raise ContractError(f"{name} must contain non-negative numeric values")
            return shape

    def walk(item: Any, path: str) -> None:
        if isinstance(item, (str, bytes)):
            raise ContractError(f"{name} contains non-numeric value at {path}")
        # A numpy array is iterable but not a Sequence; recurse through it.
        if (isinstance(item, Sequence) or
                (hasattr(item, "shape") and not isinstance(item, (int, float, bool, complex)))):
            try:
                for i, child in enumerate(item):
                    walk(child, f"{path}[{i}]")
                return
            except TypeError:
                pass
        number = _finite(item, f"{name}{path}")
        if nonnegative and number < 0:
            raise ContractError(f"{name}{path} must be >= 0")

    walk(value, "")
    return shape


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    # numpy scalars and arrays, if supplied by a caller, are handled without
    # importing numpy.  ``tolist`` is the common protocol.
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialise value of type {type(value).__name__}")


class Serializable:
    """Mixin shared by all public records."""

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Any:
        """Restore a record from :meth:`to_json` output.

        Each public record implements ``from_dict`` so this method is kept in
        the mixin rather than duplicating JSON parsing in every class.  A
        malformed document is reported as ``ContractError`` at the same
        boundary as a malformed mapping.
        """
        if not isinstance(value, (str, bytes, bytearray)):
            raise ContractError(f"{cls.__name__}.from_json expects text or bytes")
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ContractError(f"{cls.__name__}.from_json received invalid JSON") from exc
        from_dict = getattr(cls, "from_dict", None)
        if not callable(from_dict):
            raise ContractError(f"{cls.__name__} does not implement from_dict")
        return from_dict(parsed)

    @classmethod
    def _mapping(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ContractError(f"{cls.__name__}.from_dict expects a mapping")
        return value


@dataclass(frozen=True)
class Timestamp(Serializable):
    """A timestamp in a named clock domain.

    ``tick`` is an integer count in ``clock`` at ``rate_hz``.  ``t_s`` is the
    absolute simulation time represented by that tick.  Keeping both makes
    logs easy to inspect while retaining exact integer ordering.
    """

    t_s: float
    tick: int
    clock: str = "simulation"
    rate_hz: float = 1.0

    UNITS: ClassVar[Mapping[str, str]] = TIME_UNITS

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_s", _finite(self.t_s, "t_s"))
        object.__setattr__(self, "tick", _nonnegative_int(self.tick, "tick"))
        if not isinstance(self.clock, str) or not self.clock.strip():
            raise ContractError("clock must be a non-empty string")
        object.__setattr__(self, "clock", self.clock.strip())
        object.__setattr__(self, "rate_hz", _finite(self.rate_hz, "rate_hz"))
        if self.rate_hz <= 0:
            raise ContractError("rate_hz must be > 0")

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def dt_s(self) -> float:
        """Alias for the duration represented by one tick in this domain."""
        return self.period_s

    @property
    def time_s(self) -> float:
        """Readable alias for ``t_s`` used by adapters with time-oriented APIs."""
        return self.t_s

    @property
    def domain(self) -> str:
        """Alias for the named clock domain (``clock``)."""
        return self.clock

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Timestamp":
        d = cls._mapping(value)
        try:
            # ``time_s``/``domain`` are accepted on input for adapters that
            # use the longer names; output remains canonical and stable.
            t_s = d["t_s"] if "t_s" in d else d["time_s"]
            tick = d["tick"]
        except KeyError as exc:
            raise ContractError(f"Timestamp.from_dict missing field: {exc.args[0]}") from exc
        return cls(t_s=t_s, tick=tick, clock=d.get("clock", d.get("domain", "simulation")),
                   rate_hz=d.get("rate_hz", 1.0))


@dataclass
class MultiRateClock(Serializable):
    """Independent deterministic tick domains sharing one simulation epoch."""

    rates_hz: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RATES_HZ))
    epoch_s: float = 0.0
    ticks: Dict[str, int] = field(default_factory=dict)

    UNITS: ClassVar[Mapping[str, str]] = TIME_UNITS

    def __post_init__(self) -> None:
        self.epoch_s = _finite(self.epoch_s, "epoch_s")
        if not isinstance(self.rates_hz, Mapping) or not self.rates_hz:
            raise ContractError("rates_hz must contain at least one clock domain")
        clean: Dict[str, float] = {}
        for domain, rate in self.rates_hz.items():
            if not isinstance(domain, str) or not domain.strip():
                raise ContractError("clock domain names must be non-empty strings")
            r = _finite(rate, f"rates_hz[{domain!r}]")
            if r <= 0:
                raise ContractError(f"rates_hz[{domain!r}] must be > 0")
            clean[domain] = r
        self.rates_hz = clean
        if not isinstance(self.ticks, Mapping):
            raise ContractError("ticks must be a mapping")
        clean_ticks: Dict[str, int] = {}
        for domain in clean:
            tick = self.ticks.get(domain, 0)
            clean_ticks[domain] = _nonnegative_int(tick, f"tick for {domain!r}")
        unknown = set(self.ticks) - set(clean)
        if unknown:
            raise ContractError(f"ticks contain unknown domains: {sorted(unknown)}")
        self.ticks = clean_ticks

    def timestamp(self, domain: str) -> Timestamp:
        if domain not in self.rates_hz:
            raise ContractError(f"unknown clock domain: {domain!r}")
        rate = self.rates_hz[domain]
        tick = self.ticks[domain]
        return Timestamp(self.epoch_s + tick / rate, tick, domain, rate)

    def period_s(self, domain: str) -> float:
        """Return the period of a named clock domain in simulation seconds."""
        if domain not in self.rates_hz:
            raise ContractError(f"unknown clock domain: {domain!r}")
        return 1.0 / self.rates_hz[domain]

    def advance(self, domain: str, steps: int = 1) -> Timestamp:
        if domain not in self.rates_hz:
            raise ContractError(f"unknown clock domain: {domain!r}")
        self.ticks[domain] += _nonnegative_int(steps, "steps")
        return self.timestamp(domain)

    @property
    def now_s(self) -> float:
        return max(self.timestamp(d).t_s for d in self.rates_hz)

    @property
    def domains(self) -> Tuple[str, ...]:
        """Clock domains in deterministic insertion order."""
        return tuple(self.rates_hz)

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultiRateClock":
        d = cls._mapping(value)
        if "rates_hz" not in d:
            raise ContractError("MultiRateClock.from_dict missing field: rates_hz")
        return cls(rates_hz=d["rates_hz"], epoch_s=d.get("epoch_s", 0.0),
                   ticks=d.get("ticks", {}))


@dataclass
class BodyState(Serializable):
    """Rigid-body state in a right-handed world frame.

    Orientation is XYZ roll/pitch/yaw (radians).  The current point-mass body
    in ``fly_world.py`` uses yaw and pitch only; roll remains zero but is kept
    here so future body models do not change the interface.
    """

    timestamp: Timestamp
    position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_rad_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    source: SourceLabel = SourceLabel.HANDBUILT

    UNITS: ClassVar[Mapping[str, str]] = BODY_UNITS

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, Timestamp):
            raise ContractError("BodyState.timestamp must be Timestamp")
        for name in ("position_m", "orientation_rad", "velocity_mps", "angular_velocity_rad_s"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        self.source = _source(self.source)

    @property
    def orientation_rpy_rad(self) -> Tuple[float, float, float]:
        """Explicit roll/pitch/yaw spelling for sensor adapters."""
        return self.orientation_rad

    @property
    def attitude_rpy_rad(self) -> Tuple[float, float, float]:
        """Alias used by the existing mechanosensory adapter."""
        return self.orientation_rad

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BodyState":
        d = cls._mapping(value)
        if "timestamp" not in d:
            raise ContractError("BodyState.from_dict missing field: timestamp")
        return cls(timestamp=Timestamp.from_dict(d["timestamp"]),
                   position_m=d.get("position_m", (0.0, 0.0, 0.0)),
                   orientation_rad=d.get(
                       "orientation_rad",
                       d.get("orientation_rpy_rad", d.get("attitude_rpy_rad", (0.0, 0.0, 0.0))),
                   ),
                   velocity_mps=d.get("velocity_mps", (0.0, 0.0, 0.0)),
                   angular_velocity_rad_s=d.get("angular_velocity_rad_s", (0.0, 0.0, 0.0)),
                   source=d.get("source", SourceLabel.HANDBUILT))


@dataclass
class WorldState(Serializable):
    """World snapshot consumed by a sensor and updated by a body/world model."""

    timestamp: Timestamp
    body: BodyState
    world_id: str = "default"
    geometry: Mapping[str, Any] = field(default_factory=dict)
    source: SourceLabel = SourceLabel.HANDBUILT

    UNITS: ClassVar[Mapping[str, str]] = BODY_UNITS

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, Timestamp):
            raise ContractError("WorldState.timestamp must be Timestamp")
        if not isinstance(self.body, BodyState):
            raise ContractError("WorldState.body must be BodyState")
        if self.body.timestamp != self.timestamp:
            raise ContractError("WorldState and BodyState timestamps must match")
        if not isinstance(self.world_id, str) or not self.world_id.strip():
            raise ContractError("world_id must be a non-empty string")
        self.world_id = self.world_id.strip()
        if not isinstance(self.geometry, Mapping):
            raise ContractError("geometry must be a mapping")
        self.source = _source(self.source)

    @property
    def position_m(self) -> Tuple[float, float, float]:
        """Convenience view of the contained body position."""
        return self.body.position_m

    @property
    def orientation_rad(self) -> Tuple[float, float, float]:
        return self.body.orientation_rad

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorldState":
        d = cls._mapping(value)
        for required in ("timestamp", "body"):
            if required not in d:
                raise ContractError(f"WorldState.from_dict missing field: {required}")
        return cls(timestamp=Timestamp.from_dict(d["timestamp"]),
                   body=BodyState.from_dict(d["body"]), world_id=d.get("world_id", "default"),
                   geometry=d.get("geometry", {}), source=d.get("source", SourceLabel.HANDBUILT))


@dataclass
class VisualObservation(Serializable):
    """One camera/retina sample; pixels may be a numpy array or nested lists."""

    timestamp: Timestamp
    pixels: Any = None
    optic_flow_rad_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    sensor_id: str = "compound_eye"
    source: SourceLabel = SourceLabel.HANDBUILT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    UNITS: ClassVar[Mapping[str, str]] = VISION_UNITS

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, Timestamp):
            raise ContractError("VisualObservation.timestamp must be Timestamp")
        if self.pixels is not None:
            shape = _array_shape(self.pixels)
            if len(shape) not in (2, 3):
                raise ContractError(f"pixels must be grayscale (H,W) or colour (H,W,C), got {shape}")
            if any(d <= 0 for d in shape):
                raise ContractError(f"pixels dimensions must be > 0, got {shape}")
            _check_array(self.pixels, "pixels", exact_shape=shape)
        self.optic_flow_rad_s = _vector(self.optic_flow_rad_s, 3, "optic_flow_rad_s")
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise ContractError("sensor_id must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise ContractError("metadata must be a mapping")
        self.source = _source(self.source)

    @property
    def pixel_shape(self) -> Optional[Tuple[int, ...]]:
        return None if self.pixels is None else _array_shape(self.pixels)

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualObservation":
        d = cls._mapping(value)
        if "timestamp" not in d:
            raise ContractError("VisualObservation.from_dict missing field: timestamp")
        return cls(timestamp=Timestamp.from_dict(d["timestamp"]), pixels=d.get("pixels"),
                   optic_flow_rad_s=d.get("optic_flow_rad_s", (0.0, 0.0, 0.0)),
                   sensor_id=d.get("sensor_id", "compound_eye"),
                   source=d.get("source", SourceLabel.HANDBUILT), metadata=d.get("metadata", {}))


@dataclass
class SensoryFrame(Serializable):
    """Synchronous bundle delivered to the neural model at one neural tick."""

    timestamp: Timestamp
    visual: Optional[VisualObservation] = None
    channels: Mapping[str, Any] = field(default_factory=dict)
    body: Optional[BodyState] = None
    source: SourceLabel = SourceLabel.HANDBUILT

    UNITS: ClassVar[Mapping[str, str]] = {
        "timestamp": Unit.SECOND.value,
        "channels": Unit.NORMALIZED.value,
    }

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, Timestamp):
            raise ContractError("SensoryFrame.timestamp must be Timestamp")
        if self.visual is not None:
            if not isinstance(self.visual, VisualObservation):
                raise ContractError("visual must be VisualObservation or None")
            if self.visual.timestamp != self.timestamp:
                raise ContractError("visual and sensory-frame timestamps must match")
        if self.body is not None:
            if not isinstance(self.body, BodyState):
                raise ContractError("body must be BodyState or None")
            if self.body.timestamp != self.timestamp:
                raise ContractError("body and sensory-frame timestamps must match")
        if not isinstance(self.channels, Mapping):
            raise ContractError("channels must be a mapping")
        for key in self.channels:
            if not isinstance(key, str) or not key.strip():
                raise ContractError("sensory channel names must be non-empty strings")
        self.source = _source(self.source)

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SensoryFrame":
        d = cls._mapping(value)
        if "timestamp" not in d:
            raise ContractError("SensoryFrame.from_dict missing field: timestamp")
        visual = d.get("visual")
        body = d.get("body")
        return cls(timestamp=Timestamp.from_dict(d["timestamp"]),
                   visual=None if visual is None else VisualObservation.from_dict(visual),
                   channels=d.get("channels", {}),
                   body=None if body is None else BodyState.from_dict(body),
                   source=d.get("source", SourceLabel.HANDBUILT))


@dataclass
class NeuralOutput(Serializable):
    """Output of a neural step, with optional per-neuron spikes and rates."""

    timestamp: Timestamp
    rates_hz: Any = None
    spikes: Any = None
    named_outputs: Mapping[str, Any] = field(default_factory=dict)
    source: SourceLabel = SourceLabel.CONNECTOME

    UNITS: ClassVar[Mapping[str, str]] = NEURAL_UNITS

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, Timestamp):
            raise ContractError("NeuralOutput.timestamp must be Timestamp")
        shapes = []
        if self.rates_hz is not None:
            shapes.append(("rates_hz", _check_array(self.rates_hz, "rates_hz", rank=1, nonnegative=True)))
        if self.spikes is not None:
            shapes.append(("spikes", _check_array(self.spikes, "spikes", rank=1, nonnegative=True)))
        if len(shapes) == 2 and shapes[0][1] != shapes[1][1]:
            raise ContractError("rates_hz and spikes must have the same shape")
        if not isinstance(self.named_outputs, Mapping):
            raise ContractError("named_outputs must be a mapping")
        self.source = _source(self.source)

    def with_timestamp(
        self,
        timestamp: Timestamp,
        *,
        named_outputs: Optional[Mapping[str, Any]] = None,
    ) -> "NeuralOutput":
        """Return a distinct output record at ``timestamp``.

        This is the intentionally narrow fast path for adapters that have
        already received a constructor-validated output and only need to
        attach the timestamp for the current clock tick.  The large neural
        vectors and metadata are immutable-by-contract references and are
        therefore reused without rescanning them.  Initial construction must
        still go through :class:`NeuralOutput`'s normal validation above.
        """
        if not isinstance(timestamp, Timestamp):
            raise ContractError("NeuralOutput.timestamp must be Timestamp")
        if named_outputs is not None and not isinstance(named_outputs, Mapping):
            raise ContractError("named_outputs must be a mapping")
        output = object.__new__(type(self))
        output.timestamp = timestamp
        output.rates_hz = self.rates_hz
        output.spikes = self.spikes
        output.named_outputs = self.named_outputs if named_outputs is None else named_outputs
        output.source = self.source
        return output

    @classmethod
    def _from_validated_engine_arrays(
        cls,
        timestamp: Timestamp,
        *,
        rates_hz: Any,
        spikes: Any,
        named_outputs: Mapping[str, Any],
        source: SourceLabel = SourceLabel.CONNECTOME,
    ) -> "NeuralOutput":
        """Construct from arrays whose values were produced by an engine.

        This private fast path is deliberately narrower than the public
        constructor: both payloads must be NumPy arrays with matching rank-one
        shapes and real numeric dtypes.  It omits only the full-buffer finite
        and non-negative scans because the MaleCNS window path derives
        ``spikes`` from ten bounded integer counters and ``rates_hz`` from
        those same counters divided by a positive duration.  External/plugin
        data must continue through ``NeuralOutput(...)`` and its complete
        validation.
        """

        if not isinstance(timestamp, Timestamp):
            raise ContractError("NeuralOutput.timestamp must be Timestamp")
        if not isinstance(named_outputs, Mapping):
            raise ContractError("named_outputs must be a mapping")
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - engine requires NumPy
            raise ContractError("validated engine arrays require NumPy") from exc
        if not isinstance(rates_hz, np.ndarray) or not isinstance(spikes, np.ndarray):
            raise ContractError("validated engine rates_hz/spikes must be NumPy arrays")
        if rates_hz.ndim != 1 or spikes.ndim != 1 or rates_hz.shape != spikes.shape:
            raise ContractError(
                "validated engine rates_hz/spikes must be matching rank-one arrays"
            )
        if rates_hz.dtype.kind not in "iuf" or spikes.dtype.kind not in "iu":
            raise ContractError("validated engine arrays must have real numeric dtypes")
        output = object.__new__(cls)
        output.timestamp = timestamp
        output.rates_hz = rates_hz
        output.spikes = spikes
        output.named_outputs = named_outputs
        output.source = _source(source)
        return output

    @property
    def neuron_count(self) -> Optional[int]:
        """Number of neurons when a vector output is present."""
        if self.rates_hz is not None:
            return _array_shape(self.rates_hz)[0]
        if self.spikes is not None:
            return _array_shape(self.spikes)[0]
        return None

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NeuralOutput":
        d = cls._mapping(value)
        if "timestamp" not in d:
            raise ContractError("NeuralOutput.from_dict missing field: timestamp")
        return cls(timestamp=Timestamp.from_dict(d["timestamp"]), rates_hz=d.get("rates_hz"),
                   spikes=d.get("spikes"), named_outputs=d.get("named_outputs", {}),
                   source=d.get("source", SourceLabel.CONNECTOME))


@dataclass
class MotorCommand(Serializable):
    """Body command in SI units; values are signed and saturation is explicit."""

    timestamp: Timestamp
    yaw_rate_rad_s: float = 0.0
    pitch_rate_rad_s: float = 0.0
    roll_rate_rad_s: float = 0.0
    thrust_mps2: float = 0.0
    source: SourceLabel = SourceLabel.FITTED
    saturated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    UNITS: ClassVar[Mapping[str, str]] = MOTOR_UNITS

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, Timestamp):
            raise ContractError("MotorCommand.timestamp must be Timestamp")
        for name in ("yaw_rate_rad_s", "pitch_rate_rad_s", "roll_rate_rad_s", "thrust_mps2"):
            setattr(self, name, _finite(getattr(self, name), name))
        if not isinstance(self.saturated, bool):
            raise ContractError("saturated must be bool")
        if not isinstance(self.metadata, Mapping):
            raise ContractError("metadata must be a mapping")
        self.source = _source(self.source)

    @property
    def angular_rate_rad_s(self) -> Tuple[float, float, float]:
        """Vector form of the three signed angular command components."""
        return (self.roll_rate_rad_s, self.pitch_rate_rad_s, self.yaw_rate_rad_s)

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MotorCommand":
        d = cls._mapping(value)
        if "timestamp" not in d:
            raise ContractError("MotorCommand.from_dict missing field: timestamp")
        return cls(timestamp=Timestamp.from_dict(d["timestamp"]),
                   yaw_rate_rad_s=d.get("yaw_rate_rad_s", 0.0),
                   pitch_rate_rad_s=d.get("pitch_rate_rad_s", 0.0),
                   roll_rate_rad_s=d.get("roll_rate_rad_s", 0.0),
                   thrust_mps2=d.get("thrust_mps2", 0.0),
                   source=d.get("source", SourceLabel.FITTED),
                   saturated=d.get("saturated", False), metadata=d.get("metadata", {}))


@dataclass
class StepDiagnostics(Serializable):
    """Per-step observability record; counters are intentionally extensible."""

    timestamp: Timestamp
    step_index: int
    neural_steps: int = 1
    wall_time_s: float = 0.0
    real_time_factor: Optional[float] = None
    counters: Mapping[str, int | float] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    source: SourceLabel = SourceLabel.HANDBUILT

    UNITS: ClassVar[Mapping[str, str]] = {
        "timestamp": Unit.SECOND.value,
        "step_index": Unit.COUNT.value,
        "neural_steps": Unit.COUNT.value,
        "wall_time_s": Unit.SECOND.value,
        "real_time_factor": Unit.NORMALIZED.value,
        "counters": Unit.COUNT.value,
    }

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, Timestamp):
            raise ContractError("StepDiagnostics.timestamp must be Timestamp")
        for name in ("step_index", "neural_steps"):
            setattr(self, name, _nonnegative_int(getattr(self, name), name))
        self.wall_time_s = _finite(self.wall_time_s, "wall_time_s")
        if self.wall_time_s < 0:
            raise ContractError("wall_time_s must be >= 0")
        if self.real_time_factor is not None:
            self.real_time_factor = _finite(self.real_time_factor, "real_time_factor")
            if self.real_time_factor < 0:
                raise ContractError("real_time_factor must be >= 0")
        if not isinstance(self.counters, Mapping):
            raise ContractError("counters must be a mapping")
        for key, value in self.counters.items():
            if not isinstance(key, str) or not key.strip():
                raise ContractError("diagnostic counter names must be non-empty strings")
            _finite(value, f"counters[{key!r}]")
        if isinstance(self.warnings, (str, bytes)):
            raise ContractError("warnings must be a sequence of strings")
        try:
            warnings = tuple(self.warnings)
        except TypeError as exc:
            raise ContractError("warnings must be a sequence of strings") from exc
        if any(not isinstance(w, str) for w in warnings):
            raise ContractError("warnings must contain only strings")
        self.warnings = warnings
        self.source = _source(self.source)

    @property
    def units(self) -> Mapping[str, str]:
        return dict(self.UNITS)

    @property
    def unit_map(self) -> Mapping[str, str]:
        return self.units

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StepDiagnostics":
        d = cls._mapping(value)
        if "timestamp" not in d or "step_index" not in d:
            missing = "timestamp" if "timestamp" not in d else "step_index"
            raise ContractError(f"StepDiagnostics.from_dict missing field: {missing}")
        return cls(timestamp=Timestamp.from_dict(d["timestamp"]), step_index=d["step_index"],
                   neural_steps=d.get("neural_steps", 1), wall_time_s=d.get("wall_time_s", 0.0),
                   real_time_factor=d.get("real_time_factor"), counters=d.get("counters", {}),
                   warnings=d.get("warnings", ()), source=d.get("source", SourceLabel.HANDBUILT))


@runtime_checkable
class WorldModel(Protocol):
    def reset(self, state: WorldState) -> WorldState: ...

    def step(self, command: MotorCommand, timestamp: Timestamp) -> WorldState: ...


@runtime_checkable
class Sensor(Protocol):
    def observe(self, world: WorldState, timestamp: Timestamp) -> VisualObservation: ...


@runtime_checkable
class NeuralModel(Protocol):
    def step(self, sensory: SensoryFrame, timestamp: Timestamp) -> NeuralOutput: ...


@runtime_checkable
class MotorDecoder(Protocol):
    def decode(self, neural: NeuralOutput, timestamp: Timestamp) -> MotorCommand: ...


# Descriptive aliases let adapters use the vocabulary that best matches their
# role without creating a second, subtly different protocol.
VisualSensor = Sensor
NeuralProcessor = NeuralModel
WorldSimulator = WorldModel
MotorController = MotorDecoder


__all__ = [
    "NEURAL_DT_S", "DEFAULT_RATES_HZ", "ContractError", "SourceLabel", "SourceTag",
    "Source", "Provenance", "Unit", "TIME_UNITS", "BODY_UNITS", "VISION_UNITS",
    "NEURAL_UNITS", "MOTOR_UNITS",
    "Timestamp", "MultiRateClock", "WorldState", "VisualObservation", "BodyState",
    "SensoryFrame", "NeuralOutput", "MotorCommand", "StepDiagnostics",
    "WorldModel", "WorldSimulator", "Sensor", "VisualSensor", "NeuralModel",
    "NeuralProcessor", "MotorDecoder", "MotorController",
]


if __name__ == "__main__":
    # A zero-dependency smoke test useful on machines without pytest.
    clock = MultiRateClock()
    ts = clock.timestamp("neural")
    body = BodyState(ts)
    world = WorldState(ts, body)
    obs = VisualObservation(ts, [[0.0, 1.0], [0.5, 0.25]])
    frame = SensoryFrame(ts, visual=obs, body=body)
    output = NeuralOutput(ts, rates_hz=[0.0, 12.0], spikes=[0, 1])
    command = MotorCommand(ts, yaw_rate_rad_s=0.1)
    record = StepDiagnostics(ts, step_index=0)
    assert world.to_dict()["body"]["position_m"] == [0.0, 0.0, 0.0]
    assert frame.to_dict()["visual"]["pixels"] == [[0.0, 1.0], [0.5, 0.25]]
    assert output.to_json() and command.to_json() and record.to_json()
    assert WorldState.from_json(world.to_json()).to_dict() == world.to_dict()
    assert frame.units["channels"] == Unit.NORMALIZED.value
    print("twin_contracts smoke test: OK")
