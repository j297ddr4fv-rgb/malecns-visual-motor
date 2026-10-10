#!/usr/bin/env python3
"""Decode MaleCNS spike/rate observations into named motor channels.

This module is deliberately a small boundary layer.  It does not alter the
LIF simulator or the body model: it reads a vector of per-neuron rates/spike
counts (or a mapping keyed by real population specifications), applies the
data-driven map in ``config/neuromuscular_map.json``, and returns normalized
commands.  The default map uses names that are present in the checked CSR
annotation tables; literature nicknames are retained as provenance only.

Input units
-----------
``rates_hz`` are mean spikes per neuron per second (Hz).  ``spikes`` are
per-neuron spike counts for one supplied tick, or total counts when a mapping
entry names a population.  ``dt_ms`` is the duration of that tick.
``NeuralOutput.named_outputs`` may additionally carry explicit
``population_rates_hz``/``population_spike_counts`` maps; these take
precedence for the populations they cover.  Dense sub-window outputs are
accumulated over the map's calibrated readout window before rate anchors are
applied.

Output units
------------
Continuous channel values are normalized drives.  Non-negative channels are
in ``[0, 1]`` and signed steering channels are in ``[-1, 1]``.  Wing drives in
``to_body_command`` are the normalized ``DigitalBody`` actuator coordinates;
the adapter does not invent SI forces or body IDs.

The optional ``twin_contracts`` bridge is intentionally lazy.  It can package
the last dense neural vector as ``NeuralOutput`` and project the normalized
flight power/steering into a caller-supplied SI scale in ``MotorCommand``.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    from .population_selectors import comma_terms, intersect_terms
except ImportError:
    from population_selectors import comma_terms, intersect_terms


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = ROOT / "config" / "neuromuscular_map.json"
DEFAULT_CSR = Path(os.environ.get("MALECNS_CSR", "/data/cache/malecns-gpu/csr-flyb"))
_RATE_MAP_KEYS = ("population_rates_hz", "population_rates", "rates_by_population",
                  "population_rate_hz")
_SPIKE_MAP_KEYS = ("population_spike_counts", "population_counts", "spike_counts_by_population")
_POPULATION_MAP_KEYS = frozenset(_RATE_MAP_KEYS + _SPIKE_MAP_KEYS)
_DIRECT_RATE_MAP_KEYS = _RATE_MAP_KEYS[:3]
_INPUT_WRAPPER_KEYS = _POPULATION_MAP_KEYS | {"rates_hz", "spikes", "named_outputs", "channels"}
_NAMED_METADATA_KEYS = _POPULATION_MAP_KEYS | {"units", "source", "backend", "device"}


def _finite(value: Any, default: float = 0.0) -> float:
    """Return a finite float; malformed neural values fail safe to zero."""

    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clip(value: float, low: float, high: float) -> float:
    return min(float(high), max(float(low), _finite(value)))


def _as_index_array(values: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.int64)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(arr)


class AnnotationIndex:
    """Resolve the population grammar used by the FLYB/LIF scripts.

    Resolution is performed from ``annotations.npz`` and ``tables.json``;
    no hard-coded neuron/body ID list is involved.  The grammar is compatible
    with the existing ``lif_flyb.Populations`` resolver and additionally
    applies a side suffix after an intersection expression, e.g.
    ``superclass:vnc_motor&subclass:fl/R``.
    """

    _TABLE_FIELDS = {
        "type": ("typeIdx", "types"),
        "class": ("classIdx", "classes"),
        "superclass": ("superclassIdx", "superclasses"),
        "subclass": ("subclassIdx", "subclasses"),
        "nt": ("ntIdx", "nts"),
        "nerve": ("nerveIdx", "nerves"),
        "neuromere": ("neuromereIdx", "neuromeres"),
        "dimorphism": ("dimIdx", "dimorphisms"),
        "frudsx": ("fruDsxIdx", "fruDsx"),
    }

    def __init__(self, csr_path: Path | str = DEFAULT_CSR,
                 *, tables: Optional[Mapping[str, Sequence[Any]]] = None,
                 annotations: Optional[Mapping[str, Any]] = None,
                 ids: Optional[Sequence[int]] = None):
        root = Path(csr_path)
        if tables is None:
            tables = json.loads((root / "tables.json").read_text())
        if annotations is None:
            loaded = np.load(root / "annotations.npz")
            annotations = {key: loaded[key] for key in loaded.files}
        self.tables = {str(key): np.asarray(value) for key, value in tables.items()}
        self.annotations = {str(key): np.asarray(value) for key, value in annotations.items()}
        if ids is None:
            ids = np.load(root / "ids.npy")
        self.ids = np.asarray(ids, dtype=np.int64)
        if "typeIdx" not in self.annotations:
            raise ValueError("annotations must contain typeIdx")
        self.n = int(self.annotations["typeIdx"].size)
        if self.ids.size != self.n:
            raise ValueError(f"ids ({self.ids.size}) and annotations ({self.n}) disagree")
        self._cache: dict[str, np.ndarray] = {}

    def _lookup(self, index_field: str, table_name: str, needle: str) -> np.ndarray:
        codes = self.annotations.get(index_field)
        table = self.tables.get(table_name)
        if codes is None or table is None:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(np.asarray(table, dtype=str)[codes] == str(needle)).astype(np.int64)

    def _simple(self, term: str) -> np.ndarray:
        term = term.strip()
        if not term:
            return np.empty(0, dtype=np.int64)
        if ":" not in term:
            if term.lower() == "all":
                return np.arange(self.n, dtype=np.int64)
            return self._lookup("typeIdx", "types", term)
        kind, arg = (part.strip() for part in term.split(":", 1))
        kind = kind.lower()
        if kind in {"prefix", "contains"}:
            type_codes = self.annotations.get("typeIdx")
            types = self.tables.get("types")
            if type_codes is None or types is None:
                return np.empty(0, dtype=np.int64)
            names = np.asarray(types, dtype=str)[type_codes]
            matched = np.char.startswith(names, arg) if kind == "prefix" else np.char.find(names, arg) >= 0
            return np.flatnonzero(matched).astype(np.int64)
        if kind in self._TABLE_FIELDS:
            index_field, table_name = self._TABLE_FIELDS[kind]
            return self._lookup(index_field, table_name, arg)
        if kind == "body":
            try:
                body_id = int(arg)
            except (TypeError, ValueError):
                return np.empty(0, dtype=np.int64)
            return np.flatnonzero(self.ids == body_id).astype(np.int64)
        return np.empty(0, dtype=np.int64)

    def _term(self, term: str) -> np.ndarray:
        return intersect_terms(term, self._simple, unique=True).astype(np.int64, copy=False)

    def resolve(self, spec: str) -> np.ndarray:
        spec = str(spec).strip()
        if spec in self._cache:
            return self._cache[spec]
        result: list[np.ndarray] = []
        for _raw, term, side in comma_terms(spec, ("/L", "/R")):
            ids = self._term(term)
            if side is not None:
                side_codes = self.annotations.get("sideIdx")
                side_table = self.tables.get("sides")
                if side_codes is not None and side_table is not None:
                    sides = np.asarray(side_table, dtype=str)[side_codes]
                    ids = ids[sides[ids] == side]
                else:
                    ids = np.empty(0, dtype=np.int64)
            result.append(ids)
        out = np.unique(np.concatenate(result)) if result else np.empty(0, dtype=np.int64)
        out = out.astype(np.int64, copy=False)
        self._cache[spec] = out
        return out

    def size(self, spec: str) -> int:
        return int(self.resolve(spec).size)

    def describe(self, spec: str) -> dict[str, Any]:
        idx = self.resolve(spec)
        types = np.asarray(self.tables.get("types", []), dtype=str)
        type_codes = self.annotations.get("typeIdx", np.empty(0, dtype=np.int32))
        names = sorted(set(types[type_codes[idx]].tolist())) if idx.size else []
        return {"spec": spec, "count": int(idx.size), "types": names}


# A short, public alias matches the terminology used by the Java source.
PopulationIndex = AnnotationIndex


class MappingPopulationIndex:
    """Small resolver for tests/integrations that provide a synthetic index.

    Values are dense indices.  Exact specs are preferred; ``prefix:`` and
    annotation expressions remain the responsibility of a real
    :class:`AnnotationIndex`.  This class is intentionally conservative: it
    never guesses a body ID from a name.
    """

    def __init__(self, populations: Mapping[str, Sequence[int]]):
        self._populations = {str(key): _as_index_array(value) for key, value in populations.items()}
        self.n = max((int(x.max()) for x in self._populations.values() if x.size), default=-1) + 1
        self._cache: dict[str, np.ndarray] = {}

    def resolve(self, spec: str) -> np.ndarray:
        spec = str(spec).strip()
        cached = self._cache.get(spec)
        if cached is not None:
            # Keep the historical copy-on-return behavior for exact synthetic
            # selectors.  Callers must not be able to mutate this index cache.
            return cached.copy()
        if spec in self._populations:
            result = self._populations[spec]
            self._cache[spec] = result
            return result.copy()
        result: list[np.ndarray] = []
        for raw in spec.split(","):
            raw = raw.strip()
            if raw in self._populations:
                result.append(self._populations[raw])
        resolved = np.unique(np.concatenate(result)) if result else np.empty(0, dtype=np.int64)
        resolved = resolved.astype(np.int64, copy=False)
        self._cache[spec] = resolved
        return resolved.copy()

    def size(self, spec: str) -> int:
        return int(self.resolve(spec).size)


@dataclass(frozen=True)
class ChannelTerm:
    spec: str
    weight: float = 1.0
    r_max_hz: float = 50.0
    baseline_hz: float = 0.0
    provenance: str = ""
    confidence: str = "unknown"
    # Kept at the end of the dataclass fields so positional construction from
    # older integrations (spec, weight, r_max_hz, baseline_hz, provenance,
    # confidence) remains source-compatible.
    gain: float = 1.0
    # ``units`` and ``sign`` are descriptive calibration metadata.  ``weight``
    # remains the effective contribution so adding these fields does not alter
    # positional construction used by older integrations.
    units: str = "Hz"
    sign: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChannelTerm":
        spec = value.get("spec", value.get("population", value.get("name")))
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("every neuromuscular term needs a non-empty spec")
        r_max = _finite(value.get("r_max_hz", value.get("rMaxHz", 50.0)), 50.0)
        if r_max <= 0:
            raise ValueError(f"r_max_hz must be > 0 for {spec!r}")
        baseline = _finite(value.get("baseline_hz", value.get("baselineHz", 0.0)))
        if baseline >= r_max:
            raise ValueError(f"baseline_hz must be below r_max_hz for {spec!r}")
        gain = _finite(value.get("gain", value.get("calibration_gain", 1.0)), 1.0)
        if gain < 0:
            raise ValueError(f"gain must be >= 0 for {spec!r}")
        units = str(value.get("units", value.get("input_units", "Hz")))
        sign = _finite(value.get("sign", 1.0), 1.0)
        return cls(spec.strip(), _finite(value.get("weight", 1.0), 1.0), r_max,
                   baseline, str(value.get("provenance", "")),
                   str(value.get("confidence", "unknown")), gain, units, sign)


@dataclass(frozen=True)
class ChannelDefinition:
    name: str
    terms: tuple[ChannelTerm, ...]
    kind: str = "rate"
    tau_ms: float = 100.0
    bounds: tuple[float, float] = (0.0, 1.0)
    symmetry: str = ""
    provenance: str = ""
    confidence: str = "unknown"
    description: str = ""
    # The fields below are metadata only.  They make the readout boundary
    # auditable for a physical consumer without adding a behavioral policy.
    units: str = "normalized [0,1]"
    direction: Any = ""
    source: str = ""
    compatibility_only: bool = False
    mapping_policy: str = "overlap"

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, Any]) -> "ChannelDefinition":
        raw_terms = value.get("terms", ())
        if isinstance(raw_terms, Mapping):
            raw_terms = [raw_terms]
        terms = tuple(ChannelTerm.from_dict(term) for term in raw_terms)
        if not terms:
            raise ValueError(f"channel {name!r} needs at least one term")
        bounds_raw = value.get("bounds", value.get("output_bounds", (0.0, 1.0)))
        if len(bounds_raw) != 2:
            raise ValueError(f"channel {name!r} bounds must contain two values")
        low, high = _finite(bounds_raw[0]), _finite(bounds_raw[1])
        if low >= high:
            raise ValueError(f"channel {name!r} bounds must be increasing")
        kind = str(value.get("kind", value.get("input", "rate"))).lower()
        if kind in ("spike", "spikes"):
            kind = "rate"
        if kind not in ("rate", "event"):
            raise ValueError(f"unsupported channel kind {kind!r} for {name!r}")
        tau = max(0.0, _finite(value.get("tau_ms", value.get("tauMs", 100.0)), 100.0))
        default_units = "normalized [-1,1]" if low < 0.0 else "normalized [0,1]"
        units = str(value.get("units", value.get("output_units", default_units)))
        direction = value.get("direction", value.get("positive_direction", ""))
        if isinstance(direction, Mapping):
            direction = dict(direction)
        else:
            direction = str(direction)
        source = str(value.get("source", value.get("mapping_source", "")))
        compatibility_only = bool(value.get("compatibility_only", False))
        mapping_policy = str(value.get(
            "mapping_policy", value.get("population_match", "overlap")
        )).lower()
        if mapping_policy not in {"overlap", "exact"}:
            raise ValueError(
                f"unsupported mapping_policy {mapping_policy!r} for {name!r}"
            )
        return cls(str(name), terms, kind, tau, (low, high),
                   str(value.get("symmetry", "")), str(value.get("provenance", "")),
                   str(value.get("confidence", "unknown")), str(value.get("description", "")),
                   units, direction, source, compatibility_only, mapping_policy)


@dataclass
class _DecodeStepCache:
    """Ephemeral memo tables for one channel decode.

    Population selectors are immutable for the lifetime of an adapter, so
    their indices are cached on the adapter itself.  These tables are scoped
    to one call to ``step``: signal/mapping object identity and ``dt_ms`` are
    therefore sufficient cache keys and cannot leak values across ticks.
    """

    dense_arrays: dict[tuple[int, str], Any] = field(default_factory=dict)
    term_rates: dict[tuple[int, int, str, str, float, str], float] = field(default_factory=dict)
    mapping_covers: dict[tuple[int, str, str], bool] = field(default_factory=dict)
    overlaps: dict[tuple[int, str, str], np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuromuscularOutput:
    """One decoded tick, with normalized channels and bilateral wing drives."""

    # Explicit structural protocol consumed by FlightDynamics.  Publishing a
    # versioned marker lets that hot boundary use the already-normalized
    # physical wing fields directly; arbitrary lookalike objects still take
    # the general alias/validation path.
    flight_dynamics_contract = "normalized-wing-output-v1"

    channels: Mapping[str, float]
    left_wing_power: float
    right_wing_power: float
    wing_power: float
    wing_steering: float
    left_leg_power: float
    right_leg_power: float
    leg_power: float
    leg_steering: float
    mouth: float
    groom: float
    dt_ms: float
    source: str = "connectome"
    saturated: bool = False
    units: Mapping[str, str] = None  # type: ignore[assignment]
    provenance: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        clean = {str(k): _finite(v) for k, v in self.channels.items()}
        object.__setattr__(self, "channels", clean)
        for name in (
            "left_wing_power", "right_wing_power", "wing_power", "left_leg_power",
            "right_leg_power", "leg_power", "mouth", "groom", "dt_ms",
        ):
            value = _finite(getattr(self, name))
            if name != "dt_ms":
                value = _clip(value, 0.0, 1.0)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "wing_steering", _clip(self.wing_steering, -1.0, 1.0))
        object.__setattr__(self, "leg_steering", _clip(self.leg_steering, -1.0, 1.0))
        # Preserve the historical field units and add a useful lookup for
        # newly added channels even when a caller constructs
        # ``NeuromuscularOutput`` directly (outside the adapter's richer
        # channel metadata path).
        default_units = {
            "rates_hz": "Hz", "spikes": "count/tick", "dt_ms": "ms",
            "channel": "normalized", "wing_power": "normalized [0,1]",
            "wing_steering": "normalized [-1,1]", "wing_drive": "normalized [0,1]",
            "left_wing_power": "normalized [0,1]", "right_wing_power": "normalized [0,1]",
            "left_leg_power": "normalized [0,1]", "right_leg_power": "normalized [0,1]",
            "leg_power": "normalized [0,1]", "leg_steering": "normalized [-1,1]",
            "mouth": "normalized [0,1]", "groom": "normalized [0,1]",
        }
        units = dict(default_units)
        if self.units is not None:
            units.update(self.units)
        for key in clean:
            units.setdefault(str(key), "normalized [0,1]")
        object.__setattr__(self, "units", units)
        if self.provenance is None:
            object.__setattr__(self, "provenance", {"source": self.source})

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "left_wing": "left_wing_power", "right_wing": "right_wing_power",
            "left": "left_wing_power", "right": "right_wing_power",
            "wing_power_left": "left_wing_power", "wing_power_right": "right_wing_power",
            "wing_drive_left": "left_wing_power", "wing_drive_right": "right_wing_power",
            "flight_power": "wing_power", "flightPower": "wing_power",
            "flight_steering": "wing_steering", "flight_yaw": "wing_steering",
            "flightYaw": "wing_steering", "legMotor": "leg_power",
            "legMotorAsym": "leg_steering", "feed": "mouth",
            "proboscis": "mouth", "left_leg": "left_leg_power", "right_leg": "right_leg_power",
            "left_wing_activation": "left_wing_power", "right_wing_activation": "right_wing_power",
            "wing_activation_left": "left_wing_power", "wing_activation_right": "right_wing_power",
            "left_wing_motor": "wing_motor_left", "right_wing_motor": "wing_motor_right",
            "jump_event": "jump", "jumpEvent": "jump",
            "jump_activation": "jump_motor", "jump_motor_event": "jump_motor",
            "physical_jump": "jump_motor", "physical_jump_motor": "jump_motor",
            "ttmn_motor": "jump_motor", "TTMn": "ttmn_activity",
            "TTMn_activity": "ttmn_activity", "ttmn": "ttmn_activity",
            "DNp11": "dnp11_activity", "DNp02": "dnp02_activity",
            "DNp04": "dnp04_activity", "DNp11_activity": "dnp11_activity",
            "DNp02_activity": "dnp02_activity", "DNp04_activity": "dnp04_activity",
            "DNp11Activity": "dnp11_activity", "DNp02Activity": "dnp02_activity",
            "DNp04Activity": "dnp04_activity", "dnp11": "dnp11_activity",
            "dnp02": "dnp02_activity", "dnp04": "dnp04_activity",
        }
        key = aliases.get(key, key)
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.channels:
            return self.channels[key]
        raise KeyError(key)

    def channel(self, name: str, default: float = 0.0) -> float:
        aliases = {
            "flightPower": "flight_power", "flightYaw": "flight_steering",
            "flight_yaw": "flight_steering", "wing_motor": "wing_motor",
            "wingMotor": "wing_motor", "legMotor": "leg_power",
            "legMotorAsym": "leg_steering", "feed": "mouth",
            "proboscis": "mouth",
            "left_wing_activation": "left_wing_power", "right_wing_activation": "right_wing_power",
            "wing_activation_left": "left_wing_power", "wing_activation_right": "right_wing_power",
            "left_wing_motor": "wing_motor_left", "right_wing_motor": "wing_motor_right",
            "jump_event": "jump", "jumpEvent": "jump",
            "jump_activation": "jump_motor", "jump_motor_event": "jump_motor",
            "physical_jump": "jump_motor", "physical_jump_motor": "jump_motor",
            "ttmn_motor": "jump_motor", "TTMn": "ttmn_activity",
            "TTMn_activity": "ttmn_activity", "ttmn": "ttmn_activity",
            "DNp11": "dnp11_activity", "DNp02": "dnp02_activity",
            "DNp04": "dnp04_activity", "DNp11_activity": "dnp11_activity",
            "DNp02_activity": "dnp02_activity", "DNp04_activity": "dnp04_activity",
            "DNp11Activity": "dnp11_activity", "DNp02Activity": "dnp02_activity",
            "DNp04Activity": "dnp04_activity", "dnp11": "dnp11_activity",
            "dnp02": "dnp02_activity", "dnp04": "dnp04_activity",
        }
        canonical = aliases.get(name, name)
        # These names are physical output fields rather than entries in the
        # legacy decoded-channel mapping. Resolve them here too, so callers
        # using ``channel(...)`` do not get a synthetic zero when exact VNC
        # wing motors are active.
        if canonical == "left_wing_power":
            return _finite(self.left_wing_power, default)
        if canonical == "right_wing_power":
            return _finite(self.right_wing_power, default)
        if canonical == "wing_motor_left":
            return _finite(self.channels.get("wing_motor_left", default), default)
        if canonical == "wing_motor_right":
            return _finite(self.channels.get("wing_motor_right", default), default)
        return _finite(self.channels.get(canonical, default), default)

    @property
    def flight_power(self) -> float:
        return self.wing_power

    @property
    def flight_steering(self) -> float:
        return self.wing_steering

    @property
    def flight_yaw(self) -> float:
        return self.wing_steering

    @property
    def flightPower(self) -> float:
        return self.wing_power

    @property
    def flightYaw(self) -> float:
        return self.wing_steering

    @property
    def wingMotor(self) -> float:
        return self.channel("wing_motor")

    @property
    def legMotor(self) -> float:
        return self.leg_power

    @property
    def legMotorAsym(self) -> float:
        return self.leg_steering

    @property
    def groomAntenna(self) -> float:
        return self.groom_antenna

    @property
    def groomHead(self) -> float:
        return self.groom_head

    @property
    def groomLeg(self) -> float:
        return self.groom_leg

    @property
    def groomAbdomen(self) -> float:
        return self.groom_abdomen

    @property
    def left_wing(self) -> float:
        return self.left_wing_power

    @property
    def right_wing(self) -> float:
        return self.right_wing_power

    @property
    def left_wing_activation(self) -> float:
        """Physical left-wing activation after source selection."""

        return self.left_wing_power

    @property
    def right_wing_activation(self) -> float:
        """Physical right-wing activation after source selection."""

        return self.right_wing_power

    @property
    def wing_motor_left(self) -> Optional[float]:
        # ``None`` means the exact wm source was absent on this tick; this is
        # important for strict body parsers, which must then use the explicit
        # legacy DNg02 fallback rather than treating a synthetic zero as a
        # physical motor measurement.
        if "wing_motor_left" not in self.channels:
            return None
        return self.channel("wing_motor_left")

    @property
    def wing_motor_right(self) -> Optional[float]:
        if "wing_motor_right" not in self.channels:
            return None
        return self.channel("wing_motor_right")

    @property
    def wing_power_left(self) -> float:
        return self.left_wing_power

    @property
    def wing_power_right(self) -> float:
        return self.right_wing_power

    @property
    def wing_drive_left(self) -> float:
        return self.left_wing_power

    @property
    def wing_drive_right(self) -> float:
        return self.right_wing_power

    @property
    def steering(self) -> float:
        return self.wing_steering

    @property
    def left_leg(self) -> float:
        return self.left_leg_power

    @property
    def right_leg(self) -> float:
        return self.right_leg_power

    @property
    def groom_antenna(self) -> float:
        return self.channel("groom_antenna")

    @property
    def groom_head(self) -> float:
        return self.channel("groom_head")

    @property
    def groom_leg(self) -> float:
        return self.channel("groom_leg")

    @property
    def groom_abdomen(self) -> float:
        return self.channel("groom_abdomen")

    # Escape-related population readouts are deliberately exposed as
    # independent channels.  These are aliases/properties only; no trigger,
    # state machine, or sensory condition is inferred at this boundary.
    @property
    def jump(self) -> float:
        return self.channel("jump")

    @property
    def jump_event(self) -> float:
        return self.channel("jump")

    @property
    def jump_activation(self) -> float:
        """Physical jump activation derived from the TTMn motor pool."""

        return self.channel("jump_motor")

    @property
    def ttmn_activity(self) -> float:
        return self.channel("ttmn_activity")

    @property
    def jump_motor(self) -> float:
        """Physical jump motor event/activity, sourced only from TTMn."""

        return self.channel("jump_motor")

    @property
    def physical_jump_motor(self) -> float:
        return self.channel("jump_motor")

    @property
    def dnp11_activity(self) -> float:
        return self.channel("dnp11_activity")

    @property
    def dnp02_activity(self) -> float:
        return self.channel("dnp02_activity")

    @property
    def dnp04_activity(self) -> float:
        return self.channel("dnp04_activity")

    @property
    def feed(self) -> float:
        return self.mouth

    @property
    def proboscis(self) -> float:
        return self.mouth

    @property
    def escape_channels(self) -> dict[str, float]:
        """Return independent event/activity channels for a body backend.

        The mapping intentionally excludes the legacy weighted ``takeoff``
        channel.  Consumers that need source-preserving physics inputs can
        therefore pass this mapping without recovering information from a
        behavioral aggregate.
        """

        return {
            "jump": self.channel("jump"),
            "ttmn_activity": self.channel("ttmn_activity"),
            "jump_motor": self.channel("jump_motor"),
            "dnp11_activity": self.channel("dnp11_activity"),
            "dnp02_activity": self.channel("dnp02_activity"),
            "dnp04_activity": self.channel("dnp04_activity"),
        }

    physical_channels = escape_channels

    def to_physical_channels(self) -> dict[str, float]:
        """Compatibility method form of :attr:`physical_channels`."""

        return dict(self.escape_channels)

    to_escape_channels = to_physical_channels

    @property
    def physical_wing_channels(self) -> dict[str, float]:
        """Return independent left/right physical wing activations."""

        result: dict[str, float] = {
            "left_wing_activation": self.left_wing_activation,
            "right_wing_activation": self.right_wing_activation,
        }
        if self.wing_motor_left is not None:
            result["wing_motor_left"] = self.wing_motor_left
        if self.wing_motor_right is not None:
            result["wing_motor_right"] = self.wing_motor_right
        return result

    wing_channels = physical_wing_channels

    def to_body_command(self, *, wingbeat_frequency_hz: Optional[float] = None) -> dict[str, Any]:
        """Return the normalized command accepted by ``digital_body``.

        The simple body backend consumes ``wing_drive_left`` and
        ``wing_drive_right`` in ``[0,1]``.  It computes its own SI lift and
        torque from those commands; no guessed force conversion is done here.
        """

        command: dict[str, Any] = {
            "wing_drive_left": _clip(self.left_wing_power, 0.0, 1.0),
            "wing_drive_right": _clip(self.right_wing_power, 0.0, 1.0),
        }
        # Keep event and descending-neuron activities separate at the physics
        # boundary.  DigitalBody ignores these optional diagnostic keys, while
        # a richer body backend can consume them without reconstructing neural
        # provenance from a lossy aggregate.
        for name in (
            "jump", "ttmn_activity", "jump_motor",
            "dnp11_activity", "dnp02_activity", "dnp04_activity",
        ):
            command[name] = self.channel(name)
        if self.wing_motor_left is not None or self.wing_motor_right is not None:
            command["wing_motor_left"] = self.channel("wing_motor_left")
            command["wing_motor_right"] = self.channel("wing_motor_right")
        command["left_wing_activation"] = self.left_wing_activation
        command["right_wing_activation"] = self.right_wing_activation
        command["wing_activation_left"] = self.left_wing_activation
        command["wing_activation_right"] = self.right_wing_activation
        command["jump_event"] = command["jump"]
        # The physics-facing activation is TTMn-derived.  ``jump`` and
        # ``jump_event`` remain the DNp01 decision/readout compatibility keys.
        command["jump_activation"] = command["jump_motor"]
        command["physical_jump_motor"] = command["jump_motor"]
        if wingbeat_frequency_hz is not None:
            frequency = _finite(wingbeat_frequency_hz)
            if frequency <= 0:
                raise ValueError("wingbeat_frequency_hz must be positive")
            command["wingbeat_frequency_hz"] = frequency
        return command

    body_command = to_body_command

    def to_contract(self, timestamp: Any, *,
                    max_yaw_rate_rad_s: float = 1.0,
                    max_pitch_rate_rad_s: float = 0.0,
                    max_roll_rate_rad_s: float = 0.0,
                    max_thrust_mps2: float = 1.0,
                    source: Any = None) -> Any:
        """Project this output into the shared SI ``MotorCommand`` contract."""

        try:
            from twin_contracts import MotorCommand, SourceLabel
        except ImportError as exc:  # pragma: no cover - standalone use
            raise RuntimeError("twin_contracts is unavailable") from exc
        src = SourceLabel.FITTED if source is None else source
        return MotorCommand(
            timestamp=timestamp,
            yaw_rate_rad_s=self.wing_steering * _finite(max_yaw_rate_rad_s),
            pitch_rate_rad_s=self.leg_steering * _finite(max_pitch_rate_rad_s),
            roll_rate_rad_s=self.wing_steering * _finite(max_roll_rate_rad_s),
            thrust_mps2=self.wing_power * max(0.0, _finite(max_thrust_mps2)),
            source=src,
            saturated=bool(self.saturated),
            metadata={
                "adapter": "neuromuscular_adapter",
                "normalization": "channel values are normalized; SI scales supplied by caller",
                "provenance": dict(self.provenance),
                "left_wing_power": self.left_wing_power,
                "right_wing_power": self.right_wing_power,
                "left_wing_activation": self.left_wing_activation,
                "right_wing_activation": self.right_wing_activation,
                "wing_motor_left": self.channel("wing_motor_left"),
                "wing_motor_right": self.channel("wing_motor_right"),
                "physical_wing": dict(self.provenance.get("physical_wing", {})),
                "wing_power": self.wing_power,
                "wing_steering": self.wing_steering,
                "leg_power": self.leg_power,
                "leg_steering": self.leg_steering,
                "mouth": self.mouth,
                "groom": self.groom,
                "jump": self.channel("jump"),
                "jump_event": self.channel("jump"),
                "ttmn_activity": self.channel("ttmn_activity"),
                "jump_motor": self.channel("jump_motor"),
                "physical_jump_motor": self.channel("jump_motor"),
                "jump_activation": self.channel("jump_motor"),
                "dnp11_activity": self.channel("dnp11_activity"),
                "dnp02_activity": self.channel("dnp02_activity"),
                "dnp04_activity": self.channel("dnp04_activity"),
                "source": self.source,
                "escape_channels": {
                    "jump": self.channel("jump"),
                    "ttmn_activity": self.channel("ttmn_activity"),
                    "jump_motor": self.channel("jump_motor"),
                    "dnp11_activity": self.channel("dnp11_activity"),
                    "dnp02_activity": self.channel("dnp02_activity"),
                    "dnp04_activity": self.channel("dnp04_activity"),
                },
            },
        )

    as_contract = to_contract

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": dict(self.channels),
            "left_wing_power": self.left_wing_power,
            "right_wing_power": self.right_wing_power,
            "wing_power_left": self.left_wing_power,
            "wing_power_right": self.right_wing_power,
            "wing_drive_left": self.left_wing_power,
            "wing_drive_right": self.right_wing_power,
            "left_wing_activation": self.left_wing_activation,
            "right_wing_activation": self.right_wing_activation,
            "wing_activation_left": self.left_wing_activation,
            "wing_activation_right": self.right_wing_activation,
            "wing_motor_left": self.channel("wing_motor_left"),
            "wing_motor_right": self.channel("wing_motor_right"),
            "wing_power": self.wing_power,
            "wing_steering": self.wing_steering,
            "left_leg_power": self.left_leg_power,
            "right_leg_power": self.right_leg_power,
            "leg_power": self.leg_power,
            "leg_steering": self.leg_steering,
            "mouth": self.mouth,
            "groom": self.groom,
            "jump": self.channel("jump"),
            "jump_event": self.channel("jump"),
            "ttmn_activity": self.channel("ttmn_activity"),
            "jump_motor": self.channel("jump_motor"),
            "physical_jump_motor": self.channel("jump_motor"),
            "jump_activation": self.channel("jump_motor"),
            "dnp11_activity": self.channel("dnp11_activity"),
            "dnp02_activity": self.channel("dnp02_activity"),
            "dnp04_activity": self.channel("dnp04_activity"),
            "dt_ms": self.dt_ms,
            "source": self.source,
            "saturated": bool(self.saturated),
            "units": dict(self.units),
            "provenance": dict(self.provenance),
        }


class NeuromuscularAdapter:
    """Stateful, smoothed decoder for named MaleCNS motor populations."""

    _ALIASES = {
        "flightPower": "flight_power", "flightYaw": "flight_steering",
        "flight_yaw": "flight_steering", "wingMotor": "wing_motor",
        "wing_power": "flight_power", "wing_steering": "flight_steering",
        "left_wing_activation": "wing_motor_left", "right_wing_activation": "wing_motor_right",
        "wing_activation_left": "wing_motor_left", "wing_activation_right": "wing_motor_right",
        "left_wing_motor": "wing_motor_left", "right_wing_motor": "wing_motor_right",
        "wm_left": "wing_motor_left", "wm_right": "wing_motor_right",
        "legMotor": "leg_power", "legMotorAsym": "leg_steering",
        "leg_motor": "leg_power", "leg_motor_asym": "leg_steering",
        "feed": "mouth", "proboscis": "mouth",
        "groomAntenna": "groom_antenna", "groomHead": "groom_head",
        "groomLeg": "groom_leg", "groomAbdomen": "groom_abdomen",
        "jump_event": "jump", "jumpEvent": "jump",
        "jump_activation": "jump_motor", "jump_motor_event": "jump_motor",
        "physical_jump": "jump_motor", "physical_jump_motor": "jump_motor",
        "ttmn_motor": "jump_motor", "TTMn": "ttmn_activity",
        "TTMn_activity": "ttmn_activity", "ttmn": "ttmn_activity",
        "DNp11": "dnp11_activity", "DNp02": "dnp02_activity",
        "DNp04": "dnp04_activity", "DNp11_activity": "dnp11_activity",
        "DNp02_activity": "dnp02_activity", "DNp04_activity": "dnp04_activity",
        "DNp11Activity": "dnp11_activity", "DNp02Activity": "dnp02_activity",
        "DNp04Activity": "dnp04_activity", "dnp11": "dnp11_activity",
        "dnp02": "dnp02_activity", "dnp04": "dnp04_activity",
    }

    def __init__(self, map_path: Path | str = DEFAULT_MAP,
                 csr_path: Path | str = DEFAULT_CSR, *,
                 population_index: Any = None, resolver: Any = None,
                 strict: bool = False, steering_mix: Optional[float] = None,
                 source: str = "connectome"):
        self.map_path = Path(map_path)
        self.config = json.loads(self.map_path.read_text()) if self.map_path.exists() else {
            "defaults": {}, "channels": {}, "aliases": {}
        }
        self.map_sha256 = (
            hashlib.sha256(self.map_path.read_bytes()).hexdigest()
            if self.map_path.exists() else None
        )
        self.strict = bool(strict)
        raw_calibration = self.config.get("calibration", {})
        self.calibration = dict(raw_calibration) if isinstance(raw_calibration, Mapping) else {}
        # The readout is calibrated against a MaleCNS output window rather than
        # an individual 0.5 ms LIF tick.  Keep this value explicit and bounded;
        # callers that already provide a longer aggregate simply bypass the
        # accumulator below.
        window_cfg = self.calibration.get("neural_window_ms", self.config.get("defaults", {}).get(
            "neural_window_ms", 5.0))
        self.neural_window_ms = _finite(window_cfg, 5.0)
        if self.neural_window_ms <= 0.0:
            raise ValueError("calibration neural_window_ms must be positive")
        self._dense_window_enabled = bool(
            "neural_window_ms" in self.calibration
            or "neural_window_ms" in self.config.get("defaults", {})
        )
        self.calibration_id = str(self.calibration.get(
            "id", self.calibration.get("version", "unspecified")))
        if population_index is not None:
            self.population_index = population_index
            self._population_index_error: Optional[str] = None
        elif resolver is not None:
            self.population_index = resolver
            self._population_index_error = None
        else:
            try:
                self.population_index = AnnotationIndex(csr_path)
                self._population_index_error = None
            except (OSError, ValueError) as exc:
                # Keep named-mapping/unit tests usable on a checkout without
                # the rebuildable CSR cache.  Strict mode still reports every
                # unresolved annotation selector instead of pretending it is
                # connected to a graph.
                self.population_index = MappingPopulationIndex({})
                self._population_index_error = f"{type(exc).__name__}: {exc}"
        raw_channels = self.config.get("channels", {})
        if isinstance(raw_channels, Sequence) and not isinstance(raw_channels, (str, bytes, Mapping)):
            raw_channels = {item["name"]: item for item in raw_channels}
        self.channels: dict[str, ChannelDefinition] = {}
        for name, value in raw_channels.items():
            canonical = self._canonical_name(str(name))
            self.channels[canonical] = ChannelDefinition.from_dict(canonical, value)
        self.aliases = dict(self._ALIASES)
        self.aliases.update({str(k): self._canonical_name(str(v))
                             for k, v in self.config.get("aliases", {}).items()})
        # These maps are derived exclusively from the immutable channel
        # configuration.  Reusing them avoids rebuilding the same nested
        # provenance/unit/alias structures for every neural tick.
        self._alias_pairs = tuple(self.aliases.items())
        self._channel_units = self._build_channel_units()
        self._channel_provenance = self._build_channel_provenance()
        self._resolved_selector_cache: dict[str, np.ndarray] = {}
        self.tick_ms = _finite(self.config.get("defaults", {}).get("tick_ms", 50.0), 50.0)
        if self.tick_ms <= 0:
            self.tick_ms = 50.0
        wing_calibration = self.calibration.get("wing", {})
        if not isinstance(wing_calibration, Mapping):
            wing_calibration = {}
        configured_mix = self.config.get("defaults", {}).get(
            "wing_steering_mix", wing_calibration.get("steering_mix", 0.5))
        self.steering_mix = _clip(configured_mix if steering_mix is None else steering_mix, 0.0, 1.0)
        self.source = str(getattr(source, "value", source))
        self._values = {name: 0.0 for name in self.channels}
        self._baseline: Any = None
        self._last_rates_hz: Any = None
        self._last_spikes: Any = None
        self._last_timestamp: Any = None
        self._last_input_source: Any = None
        self._last_named_outputs: dict[str, Any] = {}
        self._last_signal_provenance: dict[str, Any] = {}
        self._last_saturated = False
        self._last_output: Optional[NeuromuscularOutput] = None
        # Dense MaleCNS outputs are commonly one-hot spike vectors at 2 kHz.
        # Accumulate them into the calibrated readout window before applying
        # population rate anchors.  Mapping/named-population inputs are already
        # rates or counts for their declared tick and do not use this state.
        self._window_elapsed_ms = 0.0
        self._window_rate_integral: Optional[np.ndarray] = None
        self._window_spike_counts: Optional[np.ndarray] = None
        self._resolution = self.validate(strict=strict)

    def _canonical_name(self, name: str) -> str:
        return self._ALIASES.get(name, name)

    def _build_channel_units(self) -> dict[str, str]:
        units = {
            "rates_hz": "Hz",
            "spikes": "count/tick",
            "dt_ms": "ms",
        }
        for name, channel in self.channels.items():
            units[name] = channel.units
        for alias, canonical in self._alias_pairs:
            if canonical in units:
                units[alias] = units[canonical]
        return units

    def _build_channel_provenance(self) -> dict[str, dict[str, Any]]:
        result = {
            name: {
                "provenance": channel.provenance,
                "confidence": channel.confidence,
                "symmetry": channel.symmetry,
                "units": channel.units,
                "direction": channel.direction,
                "source": channel.source,
                "description": channel.description,
                "compatibility_only": channel.compatibility_only,
                "mapping_policy": channel.mapping_policy,
                "terms": [{
                    "spec": term.spec,
                    "provenance": term.provenance,
                    "confidence": term.confidence,
                    "units": term.units,
                    "sign": term.sign,
                    "gain": term.gain,
                } for term in channel.terms],
            }
            for name, channel in self.channels.items()
        }
        for alias, canonical in {
            "jump_event": "jump",
            "jump_activation": "jump_motor",
            "physical_jump": "jump_motor",
            "physical_jump_motor": "jump_motor",
            "ttmn_motor": "jump_motor",
            "TTMn": "ttmn_activity",
        }.items():
            if canonical in result:
                result[alias] = {**result[canonical], "alias_of": canonical}
        return result

    @property
    def last_output(self) -> Optional[NeuromuscularOutput]:
        return self._last_output

    @property
    def output(self) -> Optional[NeuromuscularOutput]:
        return self._last_output

    def _resolve(self, spec: str) -> np.ndarray:
        key = str(spec).strip()
        cached = self._resolved_selector_cache.get(key)
        if cached is not None:
            return cached
        try:
            result = self.population_index.resolve(key)
        except (AttributeError, TypeError, ValueError):
            result = np.empty(0, dtype=np.int64)
        resolved = np.asarray(result, dtype=np.int64).reshape(-1)
        # Resolver indices are static after construction.  Keep a private
        # read-only view so repeated overlap checks do not call into the CSR
        # resolver on every channel/term.
        try:
            resolved.setflags(write=False)
        except ValueError:
            pass
        self._resolved_selector_cache[key] = resolved
        return resolved

    def validate(self, *, strict: bool = False) -> dict[str, Any]:
        """Return resolution counts and optionally reject empty map terms."""

        report: dict[str, Any] = {
            "channels": {},
            "unresolved": [],
            "empty_channels": [],
            "map_valid": True,
            "calibration_id": self.calibration_id,
            "calibration_window_ms": self.neural_window_ms,
            "dense_window_enabled": self._dense_window_enabled,
            "map_sha256": self.map_sha256,
            "population_index_error": self._population_index_error,
        }
        for name, channel in self.channels.items():
            terms = []
            for term in channel.terms:
                count = int(self._resolve(term.spec).size)
                item = {
                    "spec": term.spec,
                    "count": count,
                    "provenance": term.provenance,
                    "confidence": term.confidence,
                    "gain": term.gain,
                    "units": term.units,
                    "sign": term.sign,
                }
                terms.append(item)
                if count == 0:
                    report["unresolved"].append({"channel": name, **item})
            if not any(item["count"] > 0 for item in terms):
                report["empty_channels"].append(name)
            report["channels"][name] = {
                "count": int(sum(x["count"] for x in terms)),
                "terms": terms,
                "provenance": channel.provenance,
                "confidence": channel.confidence,
                "units": channel.units,
                "direction": channel.direction,
                "source": channel.source,
                "compatibility_only": channel.compatibility_only,
                "mapping_policy": channel.mapping_policy,
            }
        report["map_valid"] = not bool(report["unresolved"])
        if strict and report["unresolved"]:
            details = ", ".join(f"{x['channel']}:{x['spec']}" for x in report["unresolved"])
            raise ValueError(
                "neuromuscular map contains unresolved populations (strict readout is "
                f"fail-closed): {details}"
            )
        return report

    def population_sizes(self) -> dict[str, int]:
        """Return resolved neuron counts per channel (no IDs are exposed)."""

        return {name: int(sum(self._resolve(term.spec).size for term in channel.terms))
                for name, channel in self.channels.items()}

    def resolve(self, spec: str) -> np.ndarray:
        """Public annotation-only population resolver (dense indices only)."""

        return self._resolve(spec).copy()

    def metadata(self) -> dict[str, Any]:
        """Return auditable map metadata without exposing body IDs."""

        source = self.config.get("source", {})
        return {
            "dataset": source.get("dataset", "MaleCNS v1.0"),
            "map": str(self.map_path),
            "map_sha256": self.map_sha256,
            "contract_provenance": dict(self.config.get("contract_provenance", {})),
            "direction_convention": dict(self.config.get("direction_convention", {})),
            "units": dict(self.config.get("units", {})),
            "calibration": dict(self.calibration),
            "calibration_id": self.calibration_id,
            "neural_window_ms": self.neural_window_ms,
            "strict": self.strict,
            "population_sizes": self.population_sizes(),
            "resolution": self.validate(),
            "body_id_mapping": "not provided",
            "population_index_error": self._population_index_error,
        }

    population_metadata = metadata

    def set_baseline(self, signal: Any = None, *, rates_hz: Any = None,
                     spikes: Any = None, dt_ms: Optional[float] = None) -> None:
        """Capture a baseline signal for later subtraction.

        Baselines are represented in the same form as the supplied signal, so
        a dense vector stays dense and a named-population mapping stays named.
        For spike baselines, counts are converted using ``dt_ms`` at decode
        time; callers should normally capture a rate mapping instead.
        """

        if signal is not None:
            if rates_hz is not None or spikes is not None:
                raise TypeError("pass signal or rates_hz/spikes, not both")
            if hasattr(signal, "rates_hz") or hasattr(signal, "spikes"):
                rates_hz, spikes = getattr(signal, "rates_hz", None), getattr(signal, "spikes", None)
            elif isinstance(signal, Mapping) and ("rates_hz" in signal or "spikes" in signal):
                rates_hz, spikes = signal.get("rates_hz"), signal.get("spikes")
            else:
                rates_hz = signal
        if rates_hz is not None and spikes is not None:
            raise TypeError("baseline accepts rates_hz or spikes, not both")
        if spikes is not None:
            self._baseline = ("spikes", spikes, _finite(dt_ms, self.tick_ms) if dt_ms is not None else self.tick_ms)
        else:
            self._baseline = ("rates", rates_hz, _finite(dt_ms, self.tick_ms) if dt_ms is not None else self.tick_ms)

    def clear_baseline(self) -> None:
        self._baseline = None

    def reset(self, *, clear_baseline: bool = False) -> None:
        self._values = {name: 0.0 for name in self.channels}
        self._last_rates_hz = None
        self._last_spikes = None
        self._last_timestamp = None
        self._last_input_source = None
        self._last_named_outputs = {}
        self._last_signal_provenance = {}
        self._last_saturated = False
        self._last_output = None
        self._clear_dense_window()
        if clear_baseline:
            self.clear_baseline()

    def _clear_dense_window(self) -> None:
        """Discard a partial dense-neural readout window."""

        self._window_elapsed_ms = 0.0
        self._window_rate_integral = None
        self._window_spike_counts = None

    @staticmethod
    def _mapping_value_is_numeric(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(key in value for key in (
                "rate_hz", "rates_hz", "rate", "hz", "spikes", "count", "counts", "value",
            ))
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            return False
        return arr.ndim == 0

    def _unresolved_mapping_keys(self, *mappings: Any) -> set[str]:
        return {
            str(key) for mapping in mappings if isinstance(mapping, Mapping)
            for key, value in mapping.items()
            if str(key) not in _INPUT_WRAPPER_KEYS
            and self._mapping_value_is_numeric(value)
            and not self._resolve(str(key)).size
        }

    @staticmethod
    def _merge_named_maps(named: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
        return {
            str(key): value for field in keys if isinstance(named.get(field), Mapping)
            for key, value in named[field].items()
        }

    def _population_maps_from_named(self, named: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract explicit population rates/counts from a neural output.

        ``MaleCNSEngine`` publishes ``population_rates_hz`` and
        ``population_spike_counts`` in ``NeuralOutput.named_outputs``.  The
        deterministic stub historically called the former ``population_rates``.
        Both spellings are accepted; unrelated diagnostics (for example the
        13,582-element FlyVis vector) are never treated as neural populations.
        """

        if not isinstance(named, Mapping):
            return {}, {}
        rates = self._merge_named_maps(named, _RATE_MAP_KEYS)
        spikes = self._merge_named_maps(named, _SPIKE_MAP_KEYS)

        # Some engines expose scalar population aliases beside the structured
        # maps.  Admit only selectors that resolve against the checked CSR (or
        # a supplied synthetic index), so arbitrary metadata cannot become a
        # silent motor input.
        for raw_key, value in named.items():
            key = str(raw_key)
            if key in _NAMED_METADATA_KEYS or key.startswith("flyvis_") or not self._mapping_value_is_numeric(value):
                continue
            if self._resolve(key).size:
                rates.setdefault(key, value)
        return rates, spikes

    def _extract_named_signal(self, signal: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Return named rates, named counts, and auditable output metadata."""

        if signal is not None and hasattr(signal, "named_outputs"):
            candidate = getattr(signal, "named_outputs")
            named = candidate if isinstance(candidate, Mapping) else {}
        elif isinstance(signal, Mapping):
            named = signal.get("named_outputs")
            if not isinstance(named, Mapping):
                channels = signal.get("channels")
                named = (channels if isinstance(channels, Mapping)
                         and any(self._population_maps_from_named(channels)) else signal)
        else:
            named = {}
        rates, spikes = self._population_maps_from_named(named)
        return rates, spikes, dict(named)

    def _unpack_signal(self, signal: Any, rates_hz: Any, spikes: Any) -> tuple[Any, Any, Any]:
        timestamp = getattr(signal, "timestamp", None)
        if signal is not None and hasattr(signal, "rates_hz") and rates_hz is None:
            rates_hz = getattr(signal, "rates_hz")
        if signal is not None and hasattr(signal, "spikes") and spikes is None:
            spikes = getattr(signal, "spikes")
        if isinstance(signal, Mapping):
            if "rates_hz" in signal and rates_hz is None:
                rates_hz = signal.get("rates_hz")
            if "spikes" in signal and spikes is None:
                spikes = signal.get("spikes")
            # Structured named outputs are a first-class neural contract.  Do
            # not pass their diagnostic wrapper as if it were a dense vector.
            named = signal.get("named_outputs")
            has_named = isinstance(named, Mapping)
            has_population_rates = any(
                isinstance(signal.get(key), Mapping)
                for key in _DIRECT_RATE_MAP_KEYS
            )
            if (rates_hz is None and spikes is None and
                    not ("channels" in signal and isinstance(signal.get("channels"), Mapping)) and
                    not has_named and not has_population_rates):
                rates_hz = signal
        return rates_hz, spikes, timestamp

    def _entry_value(self, value: Any, kind: str) -> tuple[float, str]:
        if isinstance(value, Mapping):
            if kind == "spikes":
                for key in ("spikes", "count", "counts", "value"):
                    if key in value:
                        return _finite(value[key]), "spikes"
            for key in ("rate_hz", "rates_hz", "rate", "hz", "value"):
                if key in value:
                    return _finite(value[key]), "rates"
            return 0.0, kind
        return _finite(value), kind

    def _mapping_term_rate(self, mapping: Mapping[Any, Any], spec: str,
                           indices: np.ndarray, *, kind: str,
                           dt_ms: float, match_policy: str = "overlap",
                           cache: Optional[_DecodeStepCache] = None) -> float:
        # An exact population key is authoritative.  This also makes synthetic
        # unit tests independent of the full CSR files.
        if spec in mapping:
            value, value_kind = self._entry_value(mapping[spec], kind)
            if value_kind == "spikes":
                return value / max(1, int(indices.size)) / (dt_ms / 1000.0)
            return value

        # Physical source channels can require their declared selector rather
        # than an overlapping named subpopulation.  This prevents an exact
        # TTMn packet from leaking into wm/L or wm/R merely because CSR
        # annotations place that neuron in the broader motor pool.
        if str(match_policy).lower() == "exact":
            return 0.0

        total = 0.0
        covered = 0
        for raw_key, raw_value in mapping.items():
            if not isinstance(raw_key, str):
                continue
            other = self._resolve(raw_key)
            if other.size == 0 or indices.size == 0:
                continue
            overlap_key = (id(mapping), str(spec), str(raw_key))
            if cache is not None and overlap_key in cache.overlaps:
                overlap = cache.overlaps[overlap_key]
            else:
                overlap = np.intersect1d(indices, other, assume_unique=False)
                if cache is not None:
                    cache.overlaps[overlap_key] = overlap
            if overlap.size == 0:
                continue
            value, value_kind = self._entry_value(raw_value, kind)
            if value_kind == "spikes":
                value = value / max(1, int(overlap.size)) / (dt_ms / 1000.0)
            total += value * int(overlap.size)
            covered += int(overlap.size)
        # A mapping keyed by subpopulations can intentionally leave some cells
        # unspecified; the unspecified cells contribute zero, not a renormalized
        # mean.  This preserves bilateral symmetry when one side is silent.
        return total / max(1, int(indices.size)) if covered else 0.0

    def _mapping_covers(self, mapping: Optional[Mapping[Any, Any]], spec: str,
                        *, match_policy: str = "overlap",
                        cache: Optional[_DecodeStepCache] = None) -> bool:
        """Whether a named map explicitly covers at least one target cell."""

        if not isinstance(mapping, Mapping) or not mapping:
            return False
        if spec in mapping:
            return True
        if str(match_policy).lower() == "exact":
            return False
        covers_key = (id(mapping), str(spec), str(match_policy))
        if cache is not None and covers_key in cache.mapping_covers:
            return cache.mapping_covers[covers_key]
        indices = self._resolve(spec)
        if indices.size == 0:
            if cache is not None:
                cache.mapping_covers[covers_key] = False
            return False
        for raw_key in mapping:
            if not isinstance(raw_key, str):
                continue
            other = self._resolve(raw_key)
            overlap_key = (id(mapping), str(spec), str(raw_key))
            if cache is not None and overlap_key in cache.overlaps:
                overlap = cache.overlaps[overlap_key]
            else:
                overlap = np.intersect1d(indices, other, assume_unique=False)
                if cache is not None:
                    cache.overlaps[overlap_key] = overlap
            if other.size and overlap.size:
                if cache is not None:
                    cache.mapping_covers[covers_key] = True
                return True
        if cache is not None:
            cache.mapping_covers[covers_key] = False
        return False

    def _preferred_term_rate(self, signal: Any, named: Optional[Mapping[Any, Any]], spec: str,
                             *, kind: str, dt_ms: float,
                             match_policy: str = "overlap",
                             cache: Optional[_DecodeStepCache] = None) -> float:
        """Use an explicit population readout when present, dense data otherwise."""

        cache_key = (id(signal), id(named), str(spec), str(kind), float(dt_ms), str(match_policy))
        if cache is not None and cache_key in cache.term_rates:
            return cache.term_rates[cache_key]
        if self._mapping_covers(named, spec, match_policy=match_policy, cache=cache):
            value = self._mapping_term_rate(
                named or {}, spec, self._resolve(spec), kind=kind,
                dt_ms=dt_ms, match_policy=match_policy, cache=cache,
            )
        else:
            value = self._signal_term_rate(
                signal, spec, kind=kind, dt_ms=dt_ms,
                match_policy=match_policy, cache=cache,
            )
        if cache is not None:
            cache.term_rates[cache_key] = value
        return value

    def _signal_term_rate(self, signal: Any, spec: str, *, kind: str,
                          dt_ms: float, match_policy: str = "overlap",
                          cache: Optional[_DecodeStepCache] = None) -> float:
        indices = self._resolve(spec)
        if isinstance(signal, Mapping):
            return self._mapping_term_rate(
                signal, spec, indices, kind=kind, dt_ms=dt_ms,
                match_policy=match_policy, cache=cache,
            )
        if signal is None:
            return 0.0
        arr = None
        dense_key = (id(signal), str(kind))
        if cache is not None and dense_key in cache.dense_arrays:
            arr = cache.dense_arrays[dense_key]
        else:
            try:
                raw = np.asarray(signal)
            except (TypeError, ValueError):
                raw = None
            if raw is not None and raw.ndim == 1:
                # Convert once per dense source.  For malformed object arrays,
                # retain the raw array so the historical indexed conversion
                # (and its exception behavior) remains unchanged.
                try:
                    arr = np.asarray(raw, dtype=np.float64)
                except (TypeError, ValueError, OverflowError):
                    arr = raw
                if kind != "spikes" and getattr(arr, "ndim", 0) == 1:
                    try:
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    except (TypeError, ValueError):
                        pass
            else:
                arr = raw
            if cache is not None:
                cache.dense_arrays[dense_key] = arr
        if arr is None:
            return 0.0
        if arr.ndim != 1:
            return 0.0
        max_index = int(indices.max()) if indices.size else -1
        if indices.size == 0 or int(arr.size) <= max_index:
            return 0.0
        vals = np.asarray(arr[indices], dtype=np.float64)
        if kind == "spikes":
            return float(vals.sum() / max(1, int(indices.size)) / (dt_ms / 1000.0))
        return float(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).mean())

    def _prepare_dense_window(self, rates_hz: Any, spikes: Any, dt_ms: float,
                              *, named_rates: Optional[Mapping[Any, Any]] = None,
                              named_spikes: Optional[Mapping[Any, Any]] = None
                              ) -> tuple[Any, Any, float, bool]:
        """Aggregate short dense CNS ticks into the configured rate window.

        A MaleCNS tick reports instantaneous per-neuron rates (0 or
        ``1000/dt_ms`` Hz).  Population calibration anchors are mean rates over
        a longer interval, so direct one-tick decoding makes a single spike
        look like a saturated wing command.  This helper integrates dense
        arrays only; explicit named population maps are already windowed and
        are left untouched.
        """

        if named_rates or named_spikes or isinstance(rates_hz, Mapping) or isinstance(spikes, Mapping):
            self._clear_dense_window()
            return rates_hz, spikes, dt_ms, True
        candidate = rates_hz if rates_hz is not None else spikes
        if candidate is None:
            self._clear_dense_window()
            return rates_hz, spikes, dt_ms, True
        if not self._dense_window_enabled:
            self._clear_dense_window()
            return rates_hz, spikes, dt_ms, True
        try:
            candidate_arr = np.asarray(candidate, dtype=np.float64)
        except (TypeError, ValueError):
            self._clear_dense_window()
            return rates_hz, spikes, dt_ms, True
        if candidate_arr.ndim != 1:
            self._clear_dense_window()
            return rates_hz, spikes, dt_ms, True
        if dt_ms >= self.neural_window_ms - 1.0e-9:
            self._clear_dense_window()
            return rates_hz, spikes, dt_ms, True

        # A shape change means the caller switched to a different neural graph;
        # retaining old counts would be a silent cross-population mix.
        if self._window_rate_integral is not None and self._window_rate_integral.shape != candidate_arr.shape:
            self._clear_dense_window()
        if self._window_spike_counts is not None and self._window_spike_counts.shape != candidate_arr.shape:
            self._clear_dense_window()
        if rates_hz is not None:
            arr = np.nan_to_num(np.asarray(rates_hz, dtype=np.float64), nan=0.0,
                                posinf=0.0, neginf=0.0)
            self._window_rate_integral = (
                arr * dt_ms if self._window_rate_integral is None
                else self._window_rate_integral + arr * dt_ms
            )
        if spikes is not None:
            arr = np.nan_to_num(np.asarray(spikes, dtype=np.float64), nan=0.0,
                                posinf=0.0, neginf=0.0)
            self._window_spike_counts = (
                arr if self._window_spike_counts is None
                else self._window_spike_counts + arr
            )
        self._window_elapsed_ms += dt_ms
        if self._window_elapsed_ms + 1.0e-9 < self.neural_window_ms:
            # Hold the smoothed command until a complete calibration window is
            # available.  No fabricated lift is emitted during a cold start.
            return None, None, dt_ms, False

        elapsed = self._window_elapsed_ms
        aggregate_rates = None
        if self._window_rate_integral is not None:
            aggregate_rates = self._window_rate_integral / max(elapsed, 1.0e-9)
        aggregate_spikes = None if self._window_spike_counts is None else self._window_spike_counts.copy()
        self._clear_dense_window()
        return aggregate_rates, aggregate_spikes, elapsed, True

    def _baseline_term_rate(self, spec: str, *, dt_ms: float,
                            cache: Optional[_DecodeStepCache] = None) -> float:
        if self._baseline is None:
            return 0.0
        kind, signal, captured_dt = self._baseline
        # A captured spike baseline has its own tick duration.  Convert using
        # that duration first, then compare in Hz with the current tick.
        return self._signal_term_rate(signal, spec, kind=kind,
                                      dt_ms=_finite(captured_dt, dt_ms) or dt_ms,
                                      cache=cache)

    def _decode_channels(self, rates_hz: Any, spikes: Any, dt_ms: float, *,
                         named_rates: Optional[Mapping[Any, Any]] = None,
                         named_spikes: Optional[Mapping[Any, Any]] = None) -> tuple[dict[str, float], bool]:
        out: dict[str, float] = {}
        saturated = False
        cache = _DecodeStepCache()
        for name, channel in self.channels.items():
            if channel.kind == "event":
                raw = 0.0
                event_source = spikes if spikes is not None else rates_hz
                event_named = named_spikes if named_spikes else named_rates
                for term in channel.terms:
                    if event_source is None:
                        continue
                    rate = self._preferred_term_rate(
                        event_source, event_named, term.spec,
                        kind="spikes" if spikes is not None else "rates",
                        dt_ms=dt_ms,
                        match_policy=channel.mapping_policy,
                        cache=cache,
                    )
                    # A spike-count mapping may be keyed by one side or one
                    # subtype while the event term names the bilateral
                    # population.  Resolve it exactly as a continuous term;
                    # one positive count/rate is enough to produce the event.
                    raw += term.weight * term.sign * (1.0 if rate > 0 else 0.0)
                value = 1.0 if raw > 0 else 0.0
            else:
                raw = 0.0
                signal = rates_hz if rates_hz is not None else spikes
                signal_kind = "rates" if rates_hz is not None else "spikes"
                for term in channel.terms:
                    current = self._preferred_term_rate(
                        signal, named_rates, term.spec, kind=signal_kind, dt_ms=dt_ms,
                        match_policy=channel.mapping_policy, cache=cache,
                    )
                    baseline = self._baseline_term_rate(term.spec, dt_ms=dt_ms, cache=cache)
                    # A runtime baseline supersedes the static map baseline; a
                    # static baseline remains useful when no capture was made.
                    if self._baseline is None:
                        baseline = term.baseline_hz
                    denom = max(term.r_max_hz - baseline, 1.0e-9)
                    unclipped = (current - baseline) / denom * term.gain
                    normalized = _clip(unclipped, 0.0, 1.0)
                    saturated = saturated or abs(normalized - unclipped) > 1.0e-12
                    raw += term.weight * term.sign * normalized
                low, high = channel.bounds
                clipped_raw = _clip(raw, low, high)
                saturated = saturated or abs(clipped_raw - raw) > 1.0e-12
                if channel.tau_ms <= 0.0:
                    value = clipped_raw
                else:
                    alpha = 1.0 - math.exp(-dt_ms / channel.tau_ms)
                    value = self._values[name] + alpha * (clipped_raw - self._values[name])
                clipped_value = _clip(value, low, high)
                saturated = saturated or abs(clipped_value - value) > 1.0e-12
                value = clipped_value
            out[name] = value
        return out, saturated

    def _with_aliases(self, channels: Mapping[str, float]) -> dict[str, float]:
        result = {str(k): _finite(v) for k, v in channels.items()}
        for alias, canonical in self._alias_pairs:
            if canonical in result:
                result[alias] = result[canonical]
        # A compact canonical alias for callers that use wing terminology.
        if "flight_steering" in result:
            result.setdefault("wing_steering", result["flight_steering"])
        return result

    def _make_output(self, channels: Mapping[str, float], dt_ms: float,
                     saturated: bool) -> NeuromuscularOutput:
        # DNg02 flight_power/flight_steering are retained as a compatibility
        # readout.  When the exact VNC wing-motor pools carry activity, they
        # are the physical source and remain independent left/right channels;
        # this avoids turning a silent DNg02 path into a zero wing command.
        legacy_power = _clip(channels.get("flight_power", 0.0), 0.0, 1.0)
        legacy_steering = _clip(channels.get("flight_steering", 0.0), -1.0, 1.0)
        motor_left = _clip(channels.get("wing_motor_left", 0.0), 0.0, 1.0)
        motor_right = _clip(channels.get("wing_motor_right", 0.0), 0.0, 1.0)
        use_wing_motor = max(motor_left, motor_right) > 1.0e-12
        if use_wing_motor:
            physical_wing_source = "wing_motor"
            left_raw, right_raw = motor_left, motor_right
            left, right = motor_left, motor_right
            power = _clip(0.5 * (left + right), 0.0, 1.0)
            steering = _clip(right - left, -1.0, 1.0)
        else:
            physical_wing_source = "flight_power_legacy"
            power = legacy_power
            steering = legacy_steering
            left_raw = power - self.steering_mix * steering
            right_raw = power + self.steering_mix * steering
            left = _clip(left_raw, 0.0, 1.0)
            right = _clip(right_raw, 0.0, 1.0)
        saturated = bool(saturated or abs(left - left_raw) > 1.0e-12
                         or abs(right - right_raw) > 1.0e-12)
        leg = _clip(channels.get("leg_power", 0.0), 0.0, 1.0)
        leg_steering = _clip(channels.get("leg_steering", 0.0), -1.0, 1.0)
        left_leg_raw = leg - 0.5 * leg_steering
        right_leg_raw = leg + 0.5 * leg_steering
        left_leg = _clip(left_leg_raw, 0.0, 1.0)
        right_leg = _clip(right_leg_raw, 0.0, 1.0)
        saturated = bool(saturated or abs(left_leg - left_leg_raw) > 1.0e-12
                         or abs(right_leg - right_leg_raw) > 1.0e-12)
        mouth = _clip(channels.get("mouth", 0.0), 0.0, 1.0)
        groom_values = [channels.get(name, 0.0) for name in
                        ("groom_antenna", "groom_head", "groom_leg", "groom_abdomen")]
        groom = _clip(max(groom_values, default=0.0), 0.0, 1.0)
        # The channel definitions are immutable after construction.  Copy
        # only the top-level mapping so output callers retain the historical
        # dict-shaped provenance without rebuilding every nested term record.
        channel_provenance = dict(self._channel_provenance)
        output_channels = self._with_aliases(channels)
        if not use_wing_motor:
            # Absence is meaningful at the strict wing boundary: omit wm keys
            # instead of publishing synthetic zeros that would mask the
            # explicit DNg02 compatibility fallback.
            for key in (
                "wing_motor_left", "wing_motor_right", "left_wing_motor",
                "right_wing_motor", "left_wing_activation", "right_wing_activation",
                "wing_activation_left", "wing_activation_right", "wm_left", "wm_right",
            ):
                output_channels.pop(key, None)
        channel_units = dict(self._channel_units)
        return NeuromuscularOutput(
            channels=output_channels, left_wing_power=left,
            right_wing_power=right, wing_power=power, wing_steering=steering,
            left_leg_power=left_leg, right_leg_power=right_leg, leg_power=leg,
            leg_steering=leg_steering, mouth=mouth, groom=groom, dt_ms=dt_ms,
            source=self.source, saturated=bool(saturated),
            units=channel_units,
            provenance={
                "map": str(self.map_path),
                "map_sha256": self.map_sha256,
                "dataset": self.config.get("source", {}).get("dataset", "male-cns:v1.0"),
                "calibration_id": self.calibration_id,
                "calibration": dict(self.calibration),
                "resolution": self._resolution,
                "signal": dict(self._last_signal_provenance),
                "physical_wing": {
                    "source": physical_wing_source,
                    "left_activation": left,
                    "right_activation": right,
                    "legacy_flight_power": legacy_power,
                    "legacy_flight_steering": legacy_steering,
                    "wing_motor_left": motor_left,
                    "wing_motor_right": motor_right,
                    "direction": {
                        "left_right": "independent",
                        "positive_steering": "right_wing_more_power",
                    },
                },
                # Keep channel provenance at the historical top level while
                # also exposing the grouped form for new consumers.
                **channel_provenance,
                "channels": channel_provenance,
            },
        )

    def step(self, signal: Any = None, dt_ms: Optional[float] = None, *,
             rates_hz: Any = None, spikes: Any = None,
             timestamp: Any = None) -> NeuromuscularOutput:
        """Decode one tick and update the per-channel exponential smoothers.

        ``signal`` may be a dense vector, a named mapping, or a
        ``twin_contracts.NeuralOutput``.  Explicit ``rates_hz``/``spikes``
        arguments take precedence over values carried by ``signal``.
        """

        rates_hz, spikes, signal_timestamp = self._unpack_signal(signal, rates_hz, spikes)
        explicit_mapping_keys = self._unresolved_mapping_keys(rates_hz, spikes)
        named_rates, named_spikes, named_outputs = self._extract_named_signal(signal)
        # Callers may supply the structured map through the explicit
        # ``rates_hz``/``spikes`` keyword instead of wrapping it in a
        # NeuralOutput.  Normalize that form too.
        if not named_rates and isinstance(rates_hz, Mapping):
            extracted_rates, extracted_spikes = self._population_maps_from_named(rates_hz)
            if extracted_rates or extracted_spikes:
                named_rates.update(extracted_rates)
                named_spikes.update(extracted_spikes)
                named_outputs = dict(rates_hz)
                rates_hz = extracted_rates or None
        if not named_spikes and isinstance(spikes, Mapping):
            extracted_rates, extracted_spikes = self._population_maps_from_named(spikes)
            if extracted_rates or extracted_spikes:
                # A direct ``spikes={spec: count}`` mapping is count-valued;
                # only nested structured maps may contain both rate and count
                # namespaces.
                direct_keys = {
                    str(key): value for key, value in spikes.items()
                    if str(key) not in _POPULATION_MAP_KEYS and self._resolve(str(key)).size
                }
                named_outputs.update(dict(spikes))
                if extracted_rates and any(key in spikes for key in _RATE_MAP_KEYS):
                    named_rates.update(extracted_rates)
                if extracted_spikes:
                    named_spikes.update(extracted_spikes)
                elif direct_keys:
                    named_spikes.update(direct_keys)
                spikes = named_spikes or None
        if isinstance(signal, Mapping) and not isinstance(signal.get("named_outputs"), Mapping):
            direct_unresolved = self._unresolved_mapping_keys(signal)
        else:
            direct_unresolved = set()
        unresolved_named = sorted({
            str(key) for mapping in (named_rates, named_spikes)
            for key in mapping
            if isinstance(key, str) and self._resolve(key).size == 0
        })
        unresolved_all = sorted(set(unresolved_named) | direct_unresolved | explicit_mapping_keys)
        if unresolved_all and self.strict:
            raise ValueError(
                "named MaleCNS population output contains unresolved selectors "
                f"(strict readout is fail-closed): {', '.join(unresolved_all)}"
            )
        # A named-only NeuralOutput/mapping is a valid readout signal.  Keep
        # dense arrays when present, because they still feed channels that are
        # not included in the engine's optional report map.
        if rates_hz is None and named_rates:
            rates_hz = named_rates
        if spikes is None and named_spikes and not named_rates:
            spikes = named_spikes
        if timestamp is None:
            timestamp = signal_timestamp
        continuous_rates, event_spikes = rates_hz, spikes
        dt = self.tick_ms if dt_ms is None else _finite(dt_ms, self.tick_ms)
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError("dt_ms must be a finite positive duration")
        decode_rates, decode_spikes, decode_dt, window_ready = self._prepare_dense_window(
            continuous_rates, event_spikes, dt,
            named_rates=named_rates, named_spikes=named_spikes,
        )
        if window_ready:
            channels, saturated = self._decode_channels(
                decode_rates, decode_spikes, decode_dt,
                named_rates=named_rates, named_spikes=named_spikes,
            )
        else:
            # Preserve the previous EMA values while a dense neural window is
            # accumulating.  This is a hold, not a zero/fallback command.
            channels = dict(self._values)
            saturated = False
        self._values.update({name: channels[name] for name in self.channels})
        self._last_rates_hz = rates_hz
        self._last_spikes = spikes
        self._last_named_outputs = named_outputs
        self._last_signal_provenance = {
            "input_kind": "named_population" if (named_rates or named_spikes) else "dense",
            "window_ready": bool(window_ready),
            "decode_window_ms": float(decode_dt),
            "calibration_id": self.calibration_id,
            "unresolved_named": unresolved_all,
            "neural_source": getattr(
                getattr(signal, "source", None), "value", getattr(signal, "source", self.source)
            ),
        }
        if signal is not None and hasattr(signal, "source"):
            self._last_input_source = getattr(signal, "source")
        if timestamp is not None:
            self._last_timestamp = timestamp
        self._last_saturated = saturated
        self._last_output = self._make_output(channels, dt, saturated)
        return self._last_output

    update = step

    def decode(self, neural: Any, timestamp: Any = None, *,
               dt_ms: Optional[float] = None,
               max_yaw_rate_rad_s: float = 1.0,
               max_pitch_rate_rad_s: float = 0.0,
               max_roll_rate_rad_s: float = 0.0,
               max_thrust_mps2: float = 1.0) -> Any:
        """Implement the shared ``MotorDecoder`` protocol.

        ``twin_runtime`` invokes ``decode(neural, timestamp)`` at its control
        rate.  A contract timestamp therefore supplies the smoothing interval
        when ``dt_ms`` is omitted; the neural tick rate is not accidentally
        used for the slower motor readout.
        """

        if timestamp is None:
            timestamp = getattr(neural, "timestamp", None)
        if timestamp is None:
            raise ValueError("timestamp is required for MotorCommand decoding")
        if dt_ms is None:
            rate_hz = _finite(getattr(timestamp, "rate_hz", 0.0))
            dt_ms = 1000.0 / rate_hz if rate_hz > 0 else self.tick_ms
        output = self.step(neural, dt_ms=dt_ms, timestamp=timestamp)
        return self.to_motor_command(
            timestamp,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
            max_pitch_rate_rad_s=max_pitch_rate_rad_s,
            max_roll_rate_rad_s=max_roll_rate_rad_s,
            max_thrust_mps2=max_thrust_mps2,
        )

    infer = decode

    def step_body(self, body: Any, signal: Any = None, dt_ms: Optional[float] = None,
                  *, rates_hz: Any = None, spikes: Any = None,
                  dt_s: Optional[float] = None) -> tuple[NeuromuscularOutput, Any]:
        """Decode and apply one tick to a ``DigitalBody``-compatible object."""

        output = self.step(signal, dt_ms=dt_ms, rates_hz=rates_hz, spikes=spikes)
        seconds = _finite(dt_s, output.dt_ms / 1000.0) if dt_s is not None else output.dt_ms / 1000.0
        if seconds <= 0:
            raise ValueError("dt_s must be positive")
        command = output.to_body_command()
        # DigitalBody's reduced backend intentionally caps one integration
        # step at ``config.max_dt_s`` (20 ms by default), while a brain/control
        # tick is commonly 50 ms.  Substep the same neural command so the
        # adapter remains a drop-in bridge without weakening the body guard.
        config = getattr(body, "config", None)
        if config is None:
            config = getattr(getattr(body, "backend", None), "config", None)
        max_dt = _finite(getattr(config, "max_dt_s", seconds), seconds)
        n_substeps = max(1, int(math.ceil(seconds / max_dt))) if max_dt > 0 else 1
        sub_dt = seconds / n_substeps
        state = None
        for _ in range(n_substeps):
            state = body.step(command, dt_s=sub_dt)
        return output, state

    def apply_to_body(self, body: Any, signal: Any = None, dt_ms: Optional[float] = None,
                      *, rates_hz: Any = None, spikes: Any = None,
                      dt_s: Optional[float] = None) -> Any:
        """Convenience form returning only the updated body state."""

        return self.step_body(body, signal, dt_ms, rates_hz=rates_hz, spikes=spikes,
                              dt_s=dt_s)[1]

    def to_neural_output(self, timestamp: Any = None, *, rates_hz: Any = None,
                         spikes: Any = None) -> Any:
        """Package a dense last/input vector as ``twin_contracts.NeuralOutput``.

        Named decoded channels are always included.  The contract deliberately
        permits ``rates_hz`` and ``spikes`` to be ``None`` for a named-only
        observation; dense vectors are retained when they were supplied.
        """

        try:
            from twin_contracts import NeuralOutput, SourceLabel, Timestamp
        except ImportError as exc:  # pragma: no cover - only for external use
            raise RuntimeError("twin_contracts is unavailable") from exc
        if timestamp is None:
            timestamp = self._last_timestamp
        if timestamp is None:
            # Standalone callers often decode a named mapping before they have
            # a scheduler timestamp.  Use an explicit control-domain epoch;
            # protocol callers should still pass their real Timestamp.
            timestamp = Timestamp(
                0.0, 0, "control", max(1.0, 1000.0 / self.tick_ms)
            )
        rates = self._last_rates_hz if rates_hz is None else rates_hz
        spk = self._last_spikes if spikes is None else spikes
        # The contract accepts vectors only, not named mappings.  Keep the
        # mapping in named_outputs while leaving incompatible dense fields None.
        if isinstance(rates, Mapping):
            rates = None
        elif rates is not None:
            try:
                rates = np.nan_to_num(np.asarray(rates, dtype=np.float64), nan=0.0,
                                      posinf=0.0, neginf=0.0).clip(min=0.0).tolist()
            except (TypeError, ValueError):
                rates = None
        if isinstance(spk, Mapping):
            spk = None
        elif spk is not None:
            try:
                spk = np.nan_to_num(np.asarray(spk, dtype=np.float64), nan=0.0,
                                    posinf=0.0, neginf=0.0).clip(min=0.0).tolist()
            except (TypeError, ValueError):
                spk = None
        named = dict(self._last_named_outputs)
        decoded_channels = dict(self._last_output.channels if self._last_output else {})
        # Preserve the historical flat aliases and add a namespaced copy so
        # callers can distinguish decoded channels from engine diagnostics.
        named.update(decoded_channels)
        named["decoded_channels"] = decoded_channels
        named["units"] = self._last_output.units if self._last_output else {"channel": "normalized"}
        named["neuromuscular_provenance"] = {
            "calibration_id": self.calibration_id,
            "calibration": dict(self.calibration),
            "resolution": self._resolution,
            "signal": dict(self._last_signal_provenance),
        }
        neural_source = self._last_input_source or SourceLabel.CONNECTOME
        return NeuralOutput(timestamp=timestamp, rates_hz=rates, spikes=spk,
                            named_outputs=named, source=neural_source)

    def to_motor_command(self, timestamp: Any = None, *,
                         max_yaw_rate_rad_s: float = 1.0,
                         max_pitch_rate_rad_s: float = 0.0,
                         max_roll_rate_rad_s: float = 0.0,
                         max_thrust_mps2: float = 1.0,
                         source: Any = None) -> Any:
        """Project normalized flight channels into a contract MotorCommand.

        The scale arguments are explicit because the connectome map does not
        identify a validated SI wing/aerodynamic calibration.  Normalized
        bilateral wing drives remain available in ``metadata`` and should be
        used directly with ``DigitalBody``.
        """

        try:
            from twin_contracts import MotorCommand, SourceLabel, Timestamp
        except ImportError as exc:  # pragma: no cover - only for external use
            raise RuntimeError("twin_contracts is unavailable") from exc
        if timestamp is None:
            timestamp = self._last_timestamp
        if timestamp is None:
            timestamp = Timestamp(
                0.0, 0, "control", max(1.0, 1000.0 / self.tick_ms)
            )
        out = self._last_output or self._make_output(self._values, self.tick_ms, self._last_saturated)
        # The named rates are connectome-derived, but projecting a normalized
        # readout into the contract's SI MotorCommand is a readout boundary.
        # Keep that distinction explicit: callers may override ``source``
        # when they intentionally provide a different calibration provenance.
        src = SourceLabel.FITTED if source is None else source
        return MotorCommand(
            timestamp=timestamp,
            yaw_rate_rad_s=out.wing_steering * _finite(max_yaw_rate_rad_s),
            pitch_rate_rad_s=out.leg_steering * _finite(max_pitch_rate_rad_s),
            roll_rate_rad_s=out.wing_steering * _finite(max_roll_rate_rad_s),
            thrust_mps2=out.wing_power * max(0.0, _finite(max_thrust_mps2)),
            source=src,
            saturated=bool(out.saturated),
            metadata={
                "adapter": "neuromuscular_adapter",
                "normalization": "channel values are normalized; SI scales supplied by caller",
                "calibration_id": self.calibration_id,
                "calibration": dict(self.calibration),
                "map_sha256": self.map_sha256,
                "population_resolution": self._resolution,
                "signal_provenance": dict(self._last_signal_provenance),
                "left_wing_power": out.left_wing_power,
                "right_wing_power": out.right_wing_power,
                "left_wing_activation": out.left_wing_activation,
                "right_wing_activation": out.right_wing_activation,
                "wing_motor_left": out.wing_motor_left,
                "wing_motor_right": out.wing_motor_right,
                "physical_wing": dict(out.provenance.get("physical_wing", {})),
                "wing_power": out.wing_power,
                "wing_steering": out.wing_steering,
                "leg_power": out.leg_power,
                "leg_steering": out.leg_steering,
                "mouth": out.mouth,
                "groom": out.groom,
                "jump": out.channel("jump"),
                "jump_event": out.channel("jump"),
                "ttmn_activity": out.channel("ttmn_activity"),
                "jump_motor": out.channel("jump_motor"),
                "physical_jump_motor": out.channel("jump_motor"),
                "jump_activation": out.channel("jump_motor"),
                "dnp11_activity": out.channel("dnp11_activity"),
                "dnp02_activity": out.channel("dnp02_activity"),
                "dnp04_activity": out.channel("dnp04_activity"),
                "escape_channels": {
                    "jump": out.channel("jump"),
                    "ttmn_activity": out.channel("ttmn_activity"),
                    "jump_motor": out.channel("jump_motor"),
                    "dnp11_activity": out.channel("dnp11_activity"),
                    "dnp02_activity": out.channel("dnp02_activity"),
                    "dnp04_activity": out.channel("dnp04_activity"),
                },
                "escape_units": {
                    name: out.units.get(name, "normalized [0,1]")
                    for name in (
                        "jump", "ttmn_activity", "jump_motor",
                        "dnp11_activity", "dnp02_activity", "dnp04_activity",
                    )
                },
                "escape_provenance": {
                    name: out.provenance.get("channels", {}).get(name, {})
                    for name in (
                        "jump", "ttmn_activity", "jump_motor",
                        "dnp11_activity", "dnp02_activity", "dnp04_activity",
                    )
                },
                "source": out.source,
                "neural_source": (getattr(self._last_input_source, "value", self._last_input_source)
                                   if self._last_input_source is not None else SourceLabel.CONNECTOME.value),
            },
        )

    def to_contract(self, timestamp: Any = None, **kwargs: Any) -> Any:
        """Compatibility alias for the shared ``MotorCommand`` boundary."""

        return self.to_motor_command(timestamp, **kwargs)

    as_contract = to_contract


def load_adapter(*args: Any, **kwargs: Any) -> NeuromuscularAdapter:
    """Factory kept for notebooks and small integrations."""

    return NeuromuscularAdapter(*args, **kwargs)


# Compatibility names used by small notebooks and by the Java-originating
# terminology in the project.  They all point to the same stateful decoder.
MotorDecoder = NeuromuscularAdapter
NeuromuscularDecoder = NeuromuscularAdapter
MotorOutput = NeuromuscularOutput


__all__ = [
    "AnnotationIndex", "PopulationIndex", "MappingPopulationIndex", "ChannelTerm",
    "ChannelDefinition", "NeuromuscularAdapter", "NeuromuscularDecoder", "MotorDecoder",
    "NeuromuscularOutput", "MotorOutput", "load_adapter",
]


if __name__ == "__main__":  # pragma: no cover - smoke CLI
    adapter = NeuromuscularAdapter(strict=True)
    print(json.dumps({"resolution": adapter.validate(), "population_sizes": adapter.population_sizes()}, indent=2))
    out = adapter.step({"prefix:DNg02": 50.0, "MN9": 60.0}, dt_ms=50.0)
    print(json.dumps(out.to_dict(), indent=2))
