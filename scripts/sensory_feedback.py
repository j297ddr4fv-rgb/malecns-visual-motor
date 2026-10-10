#!/usr/bin/env python3
"""Body-to-MaleCNS mechanosensory feedback.

This module is a population-drive adapter, not a claim that a body simulator
has reconstructed every fly sensory ending.  It accepts a small, duck-typed
body state and emits normalized, causal signals suitable for a MaleCNS LIF
adapter.  Population selectors come from MaleCNS annotation vocabulary and
are loaded from ``config/sensory_map.json``; no neuron/body identifiers are
stored or returned.

Canonical input units are seconds, metres, metres/second, radians and
radians/second.  ``channels`` and ``population_drive`` are dimensionless and
bounded by ``saturation``.  ``raw_channels`` retains physical values and
``channel_units`` describes those values.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "sensory_map.json"
DEFAULT_CSR = Path("/data/cache/malecns-gpu/csr-flyb")


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present field from a mapping or attribute object."""

    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    else:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return default


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _time_seconds(value: Any) -> float:
    """Coerce a numeric time or a ``twin_contracts.Timestamp``-like object."""

    if isinstance(value, Mapping):
        value = _get(value, "t_s", "timestamp_s", "time_s", "seconds", default=value)
    else:
        for name in ("t_s", "timestamp_s", "time_s", "seconds"):
            if hasattr(value, name):
                value = getattr(value, name)
                break
    return _finite(value, "timestamp_s")


def _vec3(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0), *, name: str = "vector") -> Tuple[float, float, float]:
    if value is None:
        value = default
    if isinstance(value, Mapping):
        value = (
            _get(value, "x", "roll", "forward", default=default[0]),
            _get(value, "y", "pitch", "lateral", default=default[1]),
            _get(value, "z", "yaw", "vertical", default=default[2]),
        )
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a 3-vector")
    try:
        values = tuple(_finite(item, f"{name}[{i}]") for i, item in enumerate(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be a 3-vector") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector, got {len(values)} values")
    return values  # type: ignore[return-value]


def _quaternion_to_rpy(value: Any) -> Tuple[float, float, float]:
    """Convert a scalar-first (w, x, y, z) quaternion to XYZ RPY."""

    if isinstance(value, (str, bytes)):
        raise ValueError("orientation_quat must contain four values")
    try:
        values = tuple(_finite(item, f"orientation_quat[{i}]") for i, item in enumerate(value))
    except TypeError as exc:
        raise ValueError("orientation_quat must contain four values") from exc
    if len(values) != 4:
        raise ValueError("orientation_quat must contain four values (w, x, y, z)")
    w, x, y, z = values
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("orientation_quat must not be zero")
    w, x, y, z = (item / norm for item in values)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


@dataclass(frozen=True)
class BodyState:
    """Minimal body state accepted by :class:`MechanosensoryFeedback`.

    ``from_any`` also accepts the project ``digital_body.BodyState`` and
    ``twin_contracts.BodyState`` conventions.
    """

    timestamp_s: float
    attitude_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_rad_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "attitude_rpy_rad", _vec3(self.attitude_rpy_rad, name="attitude_rpy_rad"))
        object.__setattr__(self, "angular_velocity_rad_s", _vec3(self.angular_velocity_rad_s, name="angular_velocity_rad_s"))
        object.__setattr__(self, "velocity_mps", _vec3(self.velocity_mps, name="velocity_mps"))
        object.__setattr__(self, "position_m", _vec3(self.position_m, name="position_m"))

    @classmethod
    def from_any(cls, value: Any) -> "BodyState":
        if isinstance(value, cls):
            return value
        timestamp = _get(value, "timestamp_s", "time_s", "timestamp", "t", default=None)
        if timestamp is None:
            raise ValueError("BodyState requires timestamp_s (or time_s/timestamp/t)")
        timestamp_s = _time_seconds(timestamp)

        rpy = _get(value, "attitude_rpy_rad", "orientation_rad", "rpy_rad", "attitude", "rpy", default=None)
        if rpy is None:
            quaternion = _get(value, "orientation_quat", "attitude_quat", "quaternion", default=None)
            if quaternion is not None:
                rpy = _quaternion_to_rpy(quaternion)
            else:
                degree_fields = (_get(value, "roll_deg", default=None),
                                 _get(value, "pitch_deg", default=None),
                                 _get(value, "yaw_deg", "heading_deg", default=None))
                if any(item is not None for item in degree_fields):
                    rpy = tuple(_finite(item if item is not None else 0.0, "attitude_deg") * math.pi / 180.0
                                for item in degree_fields)
                else:
                    rpy = (_get(value, "roll_rad", "roll", default=0.0),
                           _get(value, "pitch_rad", "pitch", default=0.0),
                           _get(value, "yaw_rad", "yaw", "heading", default=0.0))

        omega = _get(value, "angular_velocity_rad_s", "angular_velocity_radps",
                     "angular_velocity", "omega", default=None)
        if omega is None:
            degree_rate = (_get(value, "roll_rate_deg_s", default=None),
                           _get(value, "pitch_rate_deg_s", default=None),
                           _get(value, "yaw_rate_deg_s", default=None))
            if any(item is not None for item in degree_rate):
                omega = tuple(_finite(item if item is not None else 0.0, "angular_velocity_deg_s") * math.pi / 180.0
                              for item in degree_rate)
            else:
                omega = (_get(value, "roll_rate_rad_s", "roll_rate", "omega_roll", default=0.0),
                         _get(value, "pitch_rate_rad_s", "pitch_rate", "omega_pitch", default=0.0),
                         _get(value, "yaw_rate_rad_s", "yaw_rate", "omega_yaw", default=0.0))

        return cls(timestamp_s,
                   _vec3(rpy, name="attitude_rpy_rad"),
                   _vec3(omega, name="angular_velocity_rad_s"),
                   _vec3(_get(value, "velocity_mps", "velocity", "linear_velocity", default=(0.0, 0.0, 0.0)),
                         name="velocity_mps"),
                   _vec3(_get(value, "position_m", "position", default=(0.0, 0.0, 0.0)), name="position_m"))


@dataclass(frozen=True)
class PopulationSpec:
    """A config-backed annotation selector with an evidence trail."""

    name: str
    selectors: Tuple[str, ...]
    confidence: str
    provenance: str
    role: str
    resolved_count: Optional[int] = None
    expected_count: Optional[int] = None
    mapping_confidence: str = "low"
    mapping_provenance: str = ""
    alias_of: Optional[str] = None


def _fallback_drive_mapping() -> Dict[str, Any]:
    """Return the conservative engine-side population drive contract.

    This fallback is intentionally annotation-only: it contains selectors and
    bounded gains, never neuron IDs or body IDs.  Keeping the same contract in
    the standalone fallback makes this module safe to use outside a source
    checkout while allowing the real engine to fail closed when a malformed
    mapping is supplied at its boundary.
    """

    def entry(sources: Sequence[str], target: str, gain: float, priority: int,
              provenance: str, overlap_group: Optional[str] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "sources": list(sources),
            "target_population": target,
            "gain_mV_per_ms": gain,
            "max_abs_current_mV_per_ms": gain,
            "priority": priority,
            "provenance": provenance,
        }
        if overlap_group is not None:
            result["overlap_group"] = overlap_group
        return result

    rows = (
        ("jo", ("jo", "jo_wind_gravity", "wind_gravity"), "jo_wind_gravity", 0.4, 100,
         "Bounded JO wind/gravity proxy; no per-afferent or body-ID tuning is asserted.", None),
        ("haltere", ("haltere",), "haltere", 0.35, 100,
         "Bounded angular-rate haltere proxy; no per-afferent or body-ID tuning is asserted.", None),
        ("wing", ("wing", "wing_mechanosensory"), "wing", 0.3, 100,
         "Bounded aggregate wing proxy; aggregate input takes precedence over components.", "wing"),
        ("wing_tactile", ("wing_tactile",), "wing_tactile", 0.3, 10,
         "Bounded wing tactile proxy used without an aggregate.", "wing"),
        ("wing_proprioceptive", ("wing_proprioceptive",), "wing_proprioceptive", 0.3, 10,
         "Bounded wing proprioceptive proxy used without an aggregate.", "wing"),
        ("leg", ("leg", "leg_mechanosensory"), "leg", 0.25, 100,
         "Bounded aggregate leg proxy; aggregate input takes precedence over components.", "leg"),
        ("leg_tactile", ("leg_tactile",), "leg_tactile", 0.25, 10,
         "Bounded leg tactile proxy used without an aggregate.", "leg"),
        ("leg_proprioceptive", ("leg_proprioceptive",), "leg_proprioceptive", 0.25, 10,
         "Bounded leg proprioceptive proxy used without an aggregate.", "leg"),
    )
    return {
        "schema_version": 1,
        "input_units": "normalized",
        "output_units": "current_mV_per_ms",
        "input_range": [-1.0, 1.0],
        "max_gain_mV_per_ms": 0.5,
        "body_id_mapping": "not provided",
        "entries": {
            name: entry(sources, target, gain, priority, provenance, overlap)
            for name, sources, target, gain, priority, provenance, overlap in rows
        },
    }


def _fallback_config() -> Dict[str, Any]:
    """Safe standalone fallback if a source checkout has no config file."""

    common = "MaleCNS v1.0 annotation tables; no per-sensor body-ID mapping is used."
    def group(selectors: Sequence[str], count: int, provenance: str, role: str,
              confidence: str = "high") -> Dict[str, Any]:
        return {
            "selectors": list(selectors), "expected_count": count,
            "confidence": confidence, "provenance": common + " " + provenance,
            "role": role, "mapping_confidence": "low",
        }

    def alias(target: str, role: str, confidence: str) -> Dict[str, Any]:
        return {"alias_of": target, "role": role, "confidence": confidence}

    return {
        "schema_version": 1,
        "dataset": "MaleCNS v1.0",
        "drive_mapping": _fallback_drive_mapping(),
        "populations": {
            "jo_wind_gravity": group(
                ("class:mechanosensory&subclass:wind_gravity",), 475,
                "subclass:wind_gravity is the JO wind/gravity annotation.",
                "Johnston's organ wind/gravity",
            ),
            "jo": alias("jo_wind_gravity", "JO aggregate alias", "medium"),
            "wind_gravity": alias("jo_wind_gravity", "JO wind alias", "high"),
            "haltere": group(
                ("class:mechanosensory_proprioceptive&subclass:haltere",), 201,
                "class/subclass intersection verified in the CSR.", "haltere proprioception",
            ),
            "wing_tactile": group(
                ("class:mechanosensory_tactile&nerve:ADMN",), 352,
                "ADMN wing-touch selector used by SensoryEncoders.",
                "wing tactile mechanosensation",
            ),
            "wing_proprioceptive": group(
                ("class:mechanosensory_proprioceptive&subclass:wing",), 19,
                "class/subclass intersection verified in the CSR.", "wing proprioception",
            ),
            "wing": group(
                ("class:mechanosensory_tactile&nerve:ADMN",
                 "class:mechanosensory_proprioceptive&subclass:wing"), 371,
                "union of the verified wing tactile and proprioceptive groups.",
                "wing mechanosensory aggregate", "medium",
            ),
            "wing_mechanosensory": alias("wing", "wing compatibility alias", "medium"),
            "leg_tactile": group(
                ("class:mechanosensory_tactile&nerve:ProLN",
                 "class:mechanosensory_tactile&nerve:MesoLN",
                 "class:mechanosensory_tactile&nerve:MetaLN"), 1877,
                "ProLN/MesoLN/MetaLN are the three leg nerves.",
                "leg tactile mechanosensation",
            ),
            "leg_proprioceptive": group(
                ("class:mechanosensory_proprioceptive&subclass:leg",), 190,
                "class/subclass intersection verified in the CSR.",
                "leg proprioception",
            ),
            "leg": group(
                ("class:mechanosensory_tactile&nerve:ProLN",
                 "class:mechanosensory_tactile&nerve:MesoLN",
                 "class:mechanosensory_tactile&nerve:MetaLN",
                 "class:mechanosensory_proprioceptive&subclass:leg"), 2067,
                "union of the verified leg tactile and proprioceptive groups.",
                "leg mechanosensory aggregate", "medium",
            ),
            "leg_mechanosensory": alias("leg", "leg compatibility alias", "medium"),
        },
    }


def _read_config(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("populations"), (dict, list)):
            raise ValueError("sensory map must contain populations")
        return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return _fallback_config()


def _population_specs(config: Optional[Mapping[str, Any]] = None) -> Tuple[PopulationSpec, ...]:
    config = _fallback_config() if config is None else config
    raw = config.get("populations", {})
    entries = raw.items() if isinstance(raw, Mapping) else ((str(item.get("name")), item) for item in raw
                                                              if isinstance(item, Mapping) and item.get("name"))
    specs = []
    for name, item in entries:
        item = dict(item)
        raw_selectors = item.get("selectors", ())
        if isinstance(raw_selectors, str):
            raw_selectors = (raw_selectors,)
        specs.append(PopulationSpec(
            name=str(name),
            selectors=tuple(str(s) for s in raw_selectors),
            confidence=str(item.get("confidence", "unknown")),
            provenance=str(item.get("provenance", "No selector provenance supplied.")),
            role=str(item.get("role", "mechanosensory population")),
            expected_count=(None if item.get("expected_count") is None else int(item["expected_count"])),
            mapping_confidence=str(item.get("mapping_confidence", "low")),
            mapping_provenance=str(item.get(
                "mapping_provenance",
                "Body-to-drive tuning is hand-built; MaleCNS annotations do not provide a one-to-one body mapping.",
            )),
            alias_of=(None if item.get("alias_of") is None else str(item["alias_of"])),
        ))
    return tuple(specs)


class PopulationResolver:
    """Resolve selectors to counts without loading neuron IDs."""

    _FIELD_TABLE = {
        "type": ("typeIdx", "types"), "superclass": ("superclassIdx", "superclasses"),
        "class": ("classIdx", "classes"), "subclass": ("subclassIdx", "subclasses"),
        "nerve": ("nerveIdx", "nerves"), "side": ("sideIdx", "sides"),
    }

    def __init__(self, csr_path: Path = DEFAULT_CSR, config_path: Path = DEFAULT_CONFIG):
        self.csr_path = Path(csr_path)
        self.config_path = Path(config_path)
        self.config = _read_config(self.config_path)
        self.dataset = str(self.config.get("dataset", "MaleCNS v1.0"))
        self.specs = _population_specs(self.config)
        drive_mapping = self.config.get("drive_mapping", self.config.get(
            "population_drive_mapping", self.config.get("mechanosensory_drive_mapping", {})
        ))
        self.drive_mapping = (dict(drive_mapping)
                              if isinstance(drive_mapping, Mapping) and drive_mapping
                              else _fallback_drive_mapping())
        self.population_drive_units = str(self.drive_mapping.get("input_units", "normalized"))
        self.population_drive_range = tuple(float(value) for value in self.drive_mapping.get(
            "input_range", (-1.0, 1.0)))
        if self.drive_mapping.get("body_id_mapping", "not provided") != "not provided":
            raise ValueError("sensory map must not claim body-ID mappings")
        if (self.population_drive_units != "normalized"
                or len(self.population_drive_range) != 2
                or not all(math.isfinite(value) for value in self.population_drive_range)
                or self.population_drive_range[0] >= self.population_drive_range[1]):
            # The body adapter has one deliberately small public drive unit.
            # A malformed custom map must not make the adapter emit values that
            # a MaleCNS engine could interpret differently.
            raise ValueError("drive_mapping must declare normalized input_units and a finite input_range")
        self.counts: Dict[str, Optional[int]] = {spec.name: None for spec in self.specs}
        self.status: Dict[str, str] = {spec.name: "not_resolved" for spec in self.specs}
        self._load_counts()

    @staticmethod
    def _alternatives(selectors: Iterable[str]) -> Tuple[Tuple[str, ...], ...]:
        alternatives = []
        for selector in selectors:
            for alternative in str(selector).split(","):
                terms = tuple(part.strip() for part in alternative.split("&") if part.strip())
                if terms:
                    alternatives.append(terms)
        return tuple(alternatives)

    def _load_counts(self) -> None:
        try:
            import numpy as np
            tables = json.loads((self.csr_path / "tables.json").read_text(encoding="utf-8"))
            annotations = np.load(self.csr_path / "annotations.npz")
            fields = {key: annotations[key] for key in annotations.files}
            n = len(fields["typeIdx"])
        except (OSError, ValueError, KeyError, ImportError):
            for name in self.counts:
                self.status[name] = "csr_unavailable"
            return

        by_name = {spec.name: spec for spec in self.specs}
        resolving: set[str] = set()

        def resolve(spec: PopulationSpec) -> Optional[int]:
            if self.status[spec.name] in {"resolved", "resolved_alias", "selector_unresolved", "alias_unresolved", "alias_target_missing", "alias_cycle"}:
                return self.counts[spec.name]
            if spec.name in resolving:
                self.status[spec.name] = "alias_cycle"
                return None
            resolving.add(spec.name)
            if spec.alias_of:
                target = by_name.get(spec.alias_of)
                if target is None:
                    self.status[spec.name] = "alias_target_missing"
                    resolving.remove(spec.name)
                    return None
                count = resolve(target)
                self.counts[spec.name] = count
                self.status[spec.name] = "resolved_alias" if count is not None else "alias_unresolved"
                resolving.remove(spec.name)
                return count

            union = np.zeros(n, dtype=bool)
            any_valid = False
            for terms in self._alternatives(spec.selectors):
                mask = np.ones(n, dtype=bool)
                valid = True
                for term in terms:
                    kind, sep, arg = term.partition(":")
                    if not sep or kind not in self._FIELD_TABLE:
                        valid = False
                        break
                    field_name, table_name = self._FIELD_TABLE[kind]
                    side = None
                    if "/" in arg:
                        arg, side = arg.rsplit("/", 1)
                    values = tables.get(table_name, [])
                    if arg not in values or field_name not in fields:
                        valid = False
                        break
                    mask &= fields[field_name] == values.index(arg)
                    if side is not None:
                        side_values = tables.get("sides", [])
                        if side not in side_values or "sideIdx" not in fields:
                            valid = False
                            break
                        mask &= fields["sideIdx"] == side_values.index(side)
                if valid:
                    any_valid = True
                    union |= mask
            if not any_valid:
                self.status[spec.name] = "selector_unresolved"
                self.counts[spec.name] = 0
            else:
                self.status[spec.name] = "resolved"
                self.counts[spec.name] = int(union.sum())
            resolving.remove(spec.name)
            return self.counts[spec.name]

        for spec in self.specs:
            resolve(spec)

    def metadata(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        by_name = {spec.name: spec for spec in self.specs}
        drive_entries = self.drive_mapping.get("entries", {})
        if not isinstance(drive_entries, Mapping):
            drive_entries = {}
        drive_for_source: Dict[str, Dict[str, Any]] = {}
        for entry_name, entry in drive_entries.items():
            if not isinstance(entry, Mapping):
                continue
            for source in entry.get("sources", ()):
                drive_for_source[str(source)] = {
                    "entry": str(entry_name),
                    "target_population": entry.get("target_population"),
                    "gain_mV_per_ms": entry.get("gain_mV_per_ms"),
                    "output_units": self.drive_mapping.get("output_units"),
                }
        for spec in self.specs:
            count = self.counts.get(spec.name)
            selector_spec = by_name.get(spec.alias_of) if spec.alias_of else None
            selectors = list(spec.selectors) or (list(selector_spec.selectors) if selector_spec else [])
            expected_count = spec.expected_count if spec.expected_count is not None else (
                selector_spec.expected_count if selector_spec else None
            )
            drive = drive_for_source.get(spec.name, {})
            result[spec.name] = {
                "dataset": self.dataset,
                "selectors": selectors,
                "confidence": spec.confidence,
                "provenance": spec.provenance,
                "role": spec.role,
                "resolved_count": count,
                "expected_count": expected_count,
                "count_matches_expected": (None if count is None or expected_count is None
                                             else count == expected_count),
                "selector_status": self.status.get(spec.name, "unknown"),
                "mapping_confidence": spec.mapping_confidence,
                "mapping_provenance": spec.mapping_provenance,
                "alias_of": spec.alias_of,
                "body_id_mapping": "not provided",
                "drive_units": self.population_drive_units,
                "drive_range": list(self.population_drive_range),
                "engine_drive_mapping": drive,
            }
        return result


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


def _norm(value: float, scale: float) -> float:
    if not math.isfinite(float(scale)) or abs(float(scale)) <= 1e-12:
        raise ValueError("normalization scale must be finite and non-zero")
    return _clip(float(value) / abs(float(scale)), 1.0)


def _wrap_phase(value: float) -> float:
    """Compatibility helper: wrap a cycle phase to ``[0, 1)``."""

    return float(value) - math.floor(float(value))


def _rotation_world_to_body(rpy: Sequence[float], vector: Sequence[float]) -> Tuple[float, float, float]:
    """Apply ``Rz(yaw) Ry(pitch) Rx(roll)^T`` to a world-frame vector."""

    roll, pitch, yaw = (float(item) for item in rpy)
    x, y, z = (float(item) for item in vector)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x1, y1, z1 = cy * x + sy * y, -sy * x + cy * y, z
    x2, y2, z2 = cp * x1 - sp * z1, y1, sp * x1 + cp * z1
    return x2, cr * y2 + sr * z2, -sr * y2 + cr * z2


def _side_contact(contact: Any) -> Tuple[float, float, float]:
    """Return left, right and total contact fractions in [0, 1]."""

    if contact is None:
        return 0.0, 0.0, 0.0
    if isinstance(contact, bool):
        value = 1.0 if contact else 0.0
        return value, value, value
    if isinstance(contact, (int, float)):
        value = _clip(_finite(contact, "contact"), 1.0)
        return value, value, value
    if isinstance(contact, Mapping):
        nested = _get(contact, "contacts", "legs", "feet", default=None)
        if nested is not None:
            return _side_contact(nested)
        values: Dict[str, float] = {}
        for key, value in contact.items():
            if isinstance(value, Mapping):
                continue
            try:
                numeric = (1.0 if value else 0.0) if isinstance(value, bool) else _finite(value, "contact")
                values[str(key).lower().replace("-", "_")] = _clip(numeric, 1.0)
            except ValueError:
                continue
        left = [value for key, value in values.items() if "left" in key or key in {"l", "fl", "ml", "hl"}]
        right = [value for key, value in values.items() if "right" in key or key in {"r", "fr", "mr", "hr"}]
        aggregate = next((values[key] for key in ("contact", "any", "fraction", "grounded") if key in values), 0.0)
        l = max(left) if left else aggregate
        r = max(right) if right else aggregate
        named = left + right
        total = sum(named) / len(named) if named else aggregate
        return _clip(l, 1.0), _clip(r, 1.0), _clip(total, 1.0)
    if isinstance(contact, (str, bytes)):
        raise ValueError("contact must be bool, mapping, or numeric sequence")
    values = tuple(_clip(_finite(value, "contact"), 1.0) for value in contact)
    if not values:
        return 0.0, 0.0, 0.0
    half = max(1, len(values) // 2)
    left = sum(values[:half]) / len(values[:half])
    right = sum(values[half:]) / len(values[half:]) if values[half:] else left
    return left, right, sum(values) / len(values)


def _phase_cycles(value: Any, *, key: str) -> float:
    result = _finite(value, key)
    if "rad" in key.lower():
        result /= 2.0 * math.pi
    return result % 1.0


def _joint_features(joints: Any) -> Dict[str, float]:
    """Read common joint schemas and normalize wing phases to cycles."""

    if joints is None:
        joints = {}
    nested_left = _get(joints, "left", "left_wing", "wing_left", default={})
    nested_right = _get(joints, "right", "right_wing", "wing_right", default={})
    base_phase = _get(joints, "phase_cycle", "phase", "wing_phase", "phase_rad", default=0.0)
    left_phase = _get(joints, "left_phase", "wing_phase_left", "phase_left", "wing_phase_left_rad", default=None)
    right_phase = _get(joints, "right_phase", "wing_phase_right", "phase_right", "wing_phase_right_rad", default=None)
    left_key = "left_phase"
    right_key = "right_phase"
    if left_phase is None:
        left_phase = _get(nested_left, "phase", "phase_cycle", "phase_rad", default=base_phase)
        if _get(nested_left, "phase_rad", default=None) is not None:
            left_key = "phase_rad"
    elif _get(joints, "wing_phase_left_rad", default=None) is not None:
        left_key = "wing_phase_left_rad"
    if right_phase is None:
        right_phase = _get(nested_right, "phase", "phase_cycle", "phase_rad", default=base_phase)
        if _get(nested_right, "phase_rad", default=None) is not None:
            right_key = "phase_rad"
    elif _get(joints, "wing_phase_right_rad", default=None) is not None:
        right_key = "wing_phase_right_rad"
    if _get(joints, "phase_rad", default=None) is not None:
        left_key = right_key = "phase_rad"

    left_amp = _get(joints, "left_amplitude", "wing_amplitude_left", "amplitude_left", default=None)
    right_amp = _get(joints, "right_amplitude", "wing_amplitude_right", "amplitude_right", default=None)
    if left_amp is None:
        left_amp = _get(nested_left, "amplitude", "drive", "angle_rad", default=None)
    if right_amp is None:
        right_amp = _get(nested_right, "amplitude", "drive", "angle_rad", default=None)
    # Power is distinct from the nominal stroke amplitude.  A zero-power
    # command may still carry a nominal amplitude in the body command schema;
    # treating that as active flapping would fabricate wing feedback while the
    # animal is grounded.  Keep an explicit power field when supplied, and
    # retain the historical "joint amplitudes imply an active sample" default
    # for standalone callers that do not expose power yet.
    left_power = _get(
        joints, "left_power", "left_wing_power", "wing_power_left", "wing_drive_left",
        "left_wing", default=None,
    )
    right_power = _get(
        joints, "right_power", "right_wing_power", "wing_power_right", "wing_drive_right",
        "right_wing", default=None,
    )
    if left_power is None:
        left_power = _get(nested_left, "power", "drive", default=None)
    if right_power is None:
        right_power = _get(nested_right, "power", "drive", default=None)
    shared_power = _get(joints, "wing_power", "power", "flight_power", default=None)
    if left_power is None and shared_power is not None:
        left_power = shared_power
    if right_power is None and shared_power is not None:
        right_power = shared_power
    # If neither side has a declared power, use 1 only for the legacy input
    # form that explicitly describes a wing cycle.  BodyState/default inputs
    # have no such fields and therefore remain non-flapping.
    explicit_phase = (
        _get(joints, "phase_cycle", "phase", "wing_phase", "phase_rad",
             "left_phase", "wing_phase_left", "phase_left", "wing_phase_left_rad",
             "right_phase", "wing_phase_right", "phase_right", "wing_phase_right_rad",
             default=None) is not None
        or _get(nested_left, "phase", "phase_cycle", "phase_rad", default=None) is not None
        or _get(nested_right, "phase", "phase_cycle", "phase_rad", default=None) is not None
    )
    activity_hint = (
        left_amp is not None or right_amp is not None or explicit_phase
        or _get(joints, "wingbeat_frequency_hz", "frequency_hz", "frequency", default=None) is not None
    )
    if left_power is None:
        left_power = 1.0 if activity_hint else 0.0
    if right_power is None:
        right_power = 1.0 if activity_hint else 0.0
    if left_amp is None:
        left_amp = abs(_finite(_get(joints, "left_angle_rad", "wing_angle_left", "wing_drive_left", default=0.0),
                               "left wing amplitude"))
    if right_amp is None:
        right_amp = abs(_finite(_get(joints, "right_angle_rad", "wing_angle_right", "wing_drive_right", default=0.0),
                                "right wing amplitude"))
    frequency = _get(joints, "wingbeat_frequency_hz", "frequency_hz", "frequency", default=0.0)
    return {
        "left_phase": _phase_cycles(left_phase, key=left_key),
        "right_phase": _phase_cycles(right_phase, key=right_key),
        "left_amp": max(0.0, _finite(left_amp, "left wing amplitude")),
        "right_amp": max(0.0, _finite(right_amp, "right wing amplitude")),
        "left_power": _clip(_finite(left_power, "left wing power"), 1.0),
        "right_power": _clip(_finite(right_power, "right wing power"), 1.0),
        "frequency_hz": max(0.0, _finite(frequency, "wingbeat_frequency_hz")),
    }


@dataclass
class SensoryFrame:
    timestamp_s: float
    channels: Dict[str, float]
    population_drive: Dict[str, float]
    raw_channels: Dict[str, float]
    channel_units: Dict[str, str]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    population_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_timestamp_s: Optional[float] = None
    raw_population_drive: Dict[str, float] = field(default_factory=dict)
    population_drive_units: str = "normalized"
    population_drive_range: Tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        if self.population_drive_units != "normalized":
            raise ValueError("population_drive_units must be 'normalized'")
        if (len(self.population_drive_range) != 2
                or not all(math.isfinite(float(value)) for value in self.population_drive_range)
                or float(self.population_drive_range[0]) >= float(self.population_drive_range[1])):
            raise ValueError("population_drive_range must be a finite increasing pair")
        self.population_drive_range = (float(self.population_drive_range[0]),
                                       float(self.population_drive_range[1]))

    @property
    def population_drives(self) -> Dict[str, float]:
        return self.population_drive

    @property
    def units(self) -> Dict[str, str]:
        return self.channel_units

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "source_timestamp_s": self.source_timestamp_s,
            "channels": dict(self.channels),
            "population_drive": dict(self.population_drive),
            "raw_channels": dict(self.raw_channels),
            "raw_population_drive": dict(self.raw_population_drive),
            "channel_units": dict(self.channel_units),
            "diagnostics": dict(self.diagnostics),
            "population_metadata": dict(self.population_metadata),
            "population_drive_units": self.population_drive_units,
            "population_drive_range": list(self.population_drive_range),
            "source": self.source,
        }

    @property
    def source(self) -> str:
        """The shared-contract provenance of this adapter output.

        The population labels are connectome-backed, but the body-to-drive
        transfer law is explicit glue, so the shared ``SourceLabel`` must be
        ``handbuilt`` rather than ``connectome``.  ``to_contract`` returns the
        enum instance when ``twin_contracts`` is available.
        """

        return "handbuilt"

    def to_contract(self, *, timestamp: Any = None, body_state: Any = None,
                    clock: str = "control", rate_hz: float = 200.0) -> Any:
        """Adapt this result to ``twin_contracts.SensoryFrame``.

        The local result remains useful without importing the shared contract.
        This adapter keeps the output timestamp at the current input time
        (delay is represented by ``source_timestamp_s``), carries population
        drives as ``population:<name>`` channels, and marks the source as
        ``SourceLabel.HANDBUILT``.  It intentionally does not fabricate a
        per-population neuron vector.
        """

        try:
            from twin_contracts import BodyState as ContractBodyState
            from twin_contracts import SensoryFrame as ContractSensoryFrame
            from twin_contracts import SourceLabel, Timestamp
        except ImportError as exc:  # pragma: no cover - only when used standalone
            raise RuntimeError("twin_contracts is required for to_contract()") from exc
        if timestamp is None:
            if isinstance(body_state, ContractBodyState):
                timestamp = body_state.timestamp
            else:
                rate = _finite(rate_hz, "rate_hz")
                if rate <= 0.0:
                    raise ValueError("rate_hz must be positive")
                tick = max(0, int(round(self.timestamp_s * rate)))
                timestamp = Timestamp(self.timestamp_s, tick, clock, rate)
        elif not isinstance(timestamp, Timestamp):
            if isinstance(timestamp, Mapping):
                timestamp = Timestamp.from_dict(timestamp)
            else:
                timestamp = Timestamp(_time_seconds(timestamp),
                                      max(0, int(round(_time_seconds(timestamp) * rate_hz))),
                                      clock, rate_hz)
        contract_body = None
        if body_state is not None:
            if isinstance(body_state, ContractBodyState):
                if body_state.timestamp == timestamp:
                    contract_body = body_state
                else:
                    contract_body = ContractBodyState(
                        timestamp=timestamp,
                        position_m=body_state.position_m,
                        orientation_rad=body_state.orientation_rad,
                        velocity_mps=body_state.velocity_mps,
                        angular_velocity_rad_s=body_state.angular_velocity_rad_s,
                        source=SourceLabel.HANDBUILT,
                    )
            else:
                body = BodyState.from_any(body_state)
                contract_body = ContractBodyState(
                    timestamp=timestamp,
                    position_m=body.position_m,
                    orientation_rad=body.attitude_rpy_rad,
                    velocity_mps=body.velocity_mps,
                    angular_velocity_rad_s=body.angular_velocity_rad_s,
                    source=SourceLabel.HANDBUILT,
                )
        channels: Dict[str, Any] = dict(self.channels)
        channels.update({f"population:{name}": value for name, value in self.population_drive.items()})
        channels["source_timestamp_s"] = self.source_timestamp_s
        # This is an explicit unit declaration for callers that preserve the
        # local frame directly.  The runner also copies the scalar population
        # channels itself; the MaleCNS engine accepts the declaration when it
        # is present and still rejects any conflicting units.
        channels["population_drive_units"] = self.population_drive_units
        return ContractSensoryFrame(
            timestamp=timestamp,
            channels=channels,
            body=contract_body,
            source=SourceLabel.HANDBUILT,
        )

    as_contract = to_contract


@dataclass(frozen=True)
class PreparedStep:
    """Immutable result of the causal-state-free half of one feedback step.

    Instances are created by :meth:`MechanosensoryFeedback.prepare_step` and
    are intentionally tied to that adapter instance and its FIFO preparation
    order.  The private ownership fields are validation tokens, not caller
    inputs; all computed mappings are wrapped in read-only proxies so a
    prepared value can safely cross a thread boundary before it is committed.
    """

    state: BodyState
    channels: Mapping[str, float]
    raw_channels: Mapping[str, float]
    population_drive: Mapping[str, float]
    _owner_token: object = field(repr=False, compare=False)
    _sequence: int = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        # ``frozen=True`` protects the fields themselves, while the proxies
        # protect the dictionaries nested in those fields.  Copy first so a
        # future caller cannot retain and mutate the source mapping.
        object.__setattr__(self, "channels", MappingProxyType(dict(self.channels)))
        object.__setattr__(self, "raw_channels", MappingProxyType(dict(self.raw_channels)))
        object.__setattr__(self, "population_drive", MappingProxyType(dict(self.population_drive)))

    @property
    def timestamp_s(self) -> float:
        """Timestamp of the normalized body sample represented by this item."""

        return self.state.timestamp_s


class MechanosensoryFeedback:
    """Causal body-to-population transducer with deterministic noise/delay."""

    DEFAULT_SCALES = {
        "angle_rad": math.pi,
        "angular_velocity_rad_s": 20.0,
        "wind_mps": 2.0,
        "wing_amplitude": 1.0,
        "wing_frequency_hz": 250.0,
    }

    def __init__(self, *, delay_s: float = 0.0, saturation: float = 1.0,
                 noise_std: float | Mapping[str, float] = 0.0,
                 seed: Optional[int] = None, noise_seed: Optional[int] = None,
                 csr_path: Path = DEFAULT_CSR, config_path: Path = DEFAULT_CONFIG,
                 scales: Optional[Mapping[str, float]] = None):
        self.delay_s = _finite(delay_s, "delay_s")
        self.saturation = _finite(saturation, "saturation")
        if self.delay_s < 0.0:
            raise ValueError("delay_s must be non-negative")
        if self.saturation <= 0.0:
            raise ValueError("saturation must be positive")
        if isinstance(noise_std, Mapping):
            self.noise_std: float | Dict[str, float] = {str(k): max(0.0, _finite(v, "noise_std"))
                                                         for k, v in noise_std.items()}
        else:
            self.noise_std = max(0.0, _finite(noise_std, "noise_std"))
        self.seed = seed if seed is not None else noise_seed
        self._rng = random.Random(self.seed)
        # Retain the initial RNG state so a scenario reset is reproducible,
        # including when a caller intentionally left ``seed`` unset.  This is
        # independent of the public seed value used in diagnostics.
        self._initial_rng_state = self._rng.getstate()
        self.scales = dict(self.DEFAULT_SCALES)
        if scales:
            for key, value in scales.items():
                number = _finite(value, f"scale:{key}")
                if number <= 0.0:
                    raise ValueError(f"scale:{key} must be positive")
                self.scales[str(key)] = number
        self.resolver = PopulationResolver(csr_path, config_path)
        # These values are configuration data, not per-frame observations.
        # Keep private cached copies so frame/property consumers can receive
        # defensive copies without rebuilding selector metadata every sample.
        self._population_metadata_cache = copy.deepcopy(self.resolver.metadata())
        self._population_drive_limit = max(
            abs(self.resolver.population_drive_range[0]),
            abs(self.resolver.population_drive_range[1]),
        )
        if isinstance(self.noise_std, Mapping):
            self._noise_map = dict(self.noise_std)
        else:
            self._noise_map = {"*": float(self.noise_std)}
        self._population_drive_noise_key: Dict[str, str] = {}
        drive_entries = self.resolver.drive_mapping.get("entries", {})
        if isinstance(drive_entries, Mapping):
            for entry_name, entry in drive_entries.items():
                if not isinstance(entry, Mapping):
                    continue
                noise_key = "population:" + str(entry_name)
                for source in entry.get("sources", ()):
                    self._population_drive_noise_key[str(source)] = noise_key
        self._history: Deque[Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float]]] = deque()
        # A monotonic history can discard samples older than the delay cutoff.
        # Once timestamps go backwards, retain the legacy full-scan behavior:
        # an arbitrary later timestamp could make any earlier sample eligible.
        self._history_monotonic = True
        self._last_timestamp: Optional[float] = None
        # Prepared values are committed exactly once and in FIFO order.  The
        # queue is bookkeeping only; prepare_step never touches causal
        # history, the timestamp cursor, or the RNG stream.
        self._prepared_owner = object()
        self._pending_prepared: Deque[PreparedStep] = deque()
        self._next_prepare_sequence = 0

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Clear causal history and restore the deterministic noise stream.

        A feedback adapter is stateful because delay and noise are causal.  A
        scenario boundary must therefore reset both pieces of state; merely
        resetting the body would otherwise leak the previous scenario's
        delayed samples or consume a different noise sequence on replay.
        Calling ``reset()`` repeatedly restores the same initial stream.  An
        optional seed starts and records a new reproducible stream.
        """

        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
            self._initial_rng_state = self._rng.getstate()
        else:
            self._rng = random.Random()
            self._rng.setstate(self._initial_rng_state)
        self._history.clear()
        self._history_monotonic = True
        self._last_timestamp = None
        self._pending_prepared.clear()

    @property
    def population_metadata(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self._population_metadata_cache)

    def _noise(self, name: str) -> float:
        scale = float(self._noise_map.get(name, self._noise_map.get("*", 0.0)))
        return self._rng.gauss(0.0, scale) if scale > 0.0 else 0.0

    def _compute(self, state: BodyState, world_wind: Any, ground_contact: Any,
                 joint_state: Any) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        wind_world = _vec3(world_wind, name="world_wind")
        relative_wind = tuple(wind - velocity for wind, velocity in zip(wind_world, state.velocity_mps))
        wind_body = _rotation_world_to_body(state.attitude_rpy_rad, relative_wind)
        roll, pitch, yaw = state.attitude_rpy_rad
        roll_rate, pitch_rate, yaw_rate = state.angular_velocity_rad_s
        contact_left, contact_right, contact_fraction = _side_contact(ground_contact)
        features = _joint_features(joint_state)
        left_power = features["left_power"]
        right_power = features["right_power"]
        wing_power = 0.5 * (left_power + right_power)
        flapping = bool(
            features["frequency_hz"] > 1.0e-12
            and ((left_power > 1.0e-12 and features["left_amp"] > 1.0e-12)
                 or (right_power > 1.0e-12 and features["right_amp"] > 1.0e-12))
        )
        # A WPG phase can continue advancing in a body model while its
        # actuator power is zero.  That phase is not a physical wing cue in
        # that interval, so expose a zeroed mechanosensory cycle until there
        # is actual powered flapping on at least one side.
        phase_left = features["left_phase"] if flapping and left_power > 1.0e-12 else 0.0
        phase_right = features["right_phase"] if flapping and right_power > 1.0e-12 else 0.0
        left_amp = features["left_amp"] if left_power > 1.0e-12 else 0.0
        right_amp = features["right_amp"] if right_power > 1.0e-12 else 0.0
        frequency_hz = features["frequency_hz"] if flapping else 0.0
        phase_mean = ((phase_left + phase_right) * 0.5) % 1.0
        phase_difference = ((phase_left - phase_right + 0.5) % 1.0) - 0.5
        amplitude_difference = left_amp - right_amp
        wind_speed = math.sqrt(sum(item * item for item in wind_body))
        tilt_signal = 0.5 * (math.sin(roll) + math.sin(pitch))

        raw = {
            "attitude_roll": roll, "attitude_pitch": pitch, "attitude_yaw": yaw,
            "angular_velocity_roll": roll_rate, "angular_velocity_pitch": pitch_rate,
            "angular_velocity_yaw": yaw_rate,
            "wind_body_forward": wind_body[0], "wind_body_lateral": wind_body[1],
            "wind_body_vertical": wind_body[2], "wind_body_speed": wind_speed,
            "ground_contact_left": contact_left, "ground_contact_right": contact_right,
            "ground_contact_fraction": contact_fraction,
            "wing_phase": phase_mean, "wing_phase_left": phase_left,
            "wing_phase_right": phase_right, "wing_phase_difference": phase_difference,
            "wing_amplitude_left": left_amp, "wing_amplitude_right": right_amp,
            "wing_left_right_imbalance": amplitude_difference,
            "wing_power_left": left_power, "wing_power_right": right_power,
            "wing_power": wing_power,
            "wingbeat_frequency_hz": frequency_hz,
        }
        phase_sin = math.sin(2.0 * math.pi * phase_mean)
        phase_cos = math.cos(2.0 * math.pi * phase_mean) if flapping else 0.0
        channels = {
            "attitude_roll": _norm(roll, self.scales["angle_rad"]),
            "attitude_pitch": _norm(pitch, self.scales["angle_rad"]),
            "attitude_yaw": _norm(yaw, self.scales["angle_rad"]),
            "angular_velocity_roll": _norm(roll_rate, self.scales["angular_velocity_rad_s"]),
            "angular_velocity_pitch": _norm(pitch_rate, self.scales["angular_velocity_rad_s"]),
            "angular_velocity_yaw": _norm(yaw_rate, self.scales["angular_velocity_rad_s"]),
            "wind_body_forward": _norm(wind_body[0], self.scales["wind_mps"]),
            "wind_body_lateral": _norm(wind_body[1], self.scales["wind_mps"]),
            "wind_body_vertical": _norm(wind_body[2], self.scales["wind_mps"]),
            "wind_body_speed": _norm(wind_speed, self.scales["wind_mps"]),
            "ground_contact_left": contact_left, "ground_contact_right": contact_right,
            "ground_contact_fraction": contact_fraction,
            "wing_phase_sin": phase_sin, "wing_phase_cos": phase_cos,
            "wing_phase_difference": _norm(phase_difference, 0.5),
            "wing_left_right_imbalance": _norm(amplitude_difference, self.scales["wing_amplitude"]),
            "wing_power_left": left_power, "wing_power_right": right_power,
            "wing_power": wing_power,
            "wingbeat_frequency": _norm(frequency_hz, self.scales["wing_frequency_hz"]),
        }

        # Explicit low-dimensional glue laws. They preserve bilateral signs
        # but make no per-neuron tuning claim.
        jo_wind = _clip(0.45 * channels["wind_body_lateral"]
                        + 0.20 * channels["wind_body_vertical"]
                        + 0.20 * channels["wind_body_forward"]
                        + 0.15 * _norm(tilt_signal, 1.0), 1.0)
        haltere = _clip(0.50 * channels["angular_velocity_yaw"]
                        + 0.25 * channels["angular_velocity_roll"]
                        + 0.25 * channels["angular_velocity_pitch"], 1.0)
        wing_tactile = _clip(0.60 * channels["wing_left_right_imbalance"] + 0.40 * phase_sin, 1.0)
        wing_proprio = _clip(0.50 * phase_sin + 0.30 * channels["wing_phase_difference"]
                             + 0.20 * channels["wingbeat_frequency"], 1.0)
        wing = _clip(0.70 * wing_tactile + 0.30 * wing_proprio, 1.0)
        side_difference = contact_left - contact_right
        leg_tactile = _clip(0.65 * contact_fraction + 0.35 * side_difference, 1.0)
        leg_proprio = _clip(0.50 * contact_fraction + 0.50 * side_difference, 1.0)
        leg = _clip(0.70 * leg_tactile + 0.30 * leg_proprio, 1.0)
        drives = {
            "jo_wind_gravity": jo_wind, "jo": jo_wind, "wind_gravity": jo_wind,
            "haltere": haltere,
            "wing_tactile": wing_tactile, "wing_proprioceptive": wing_proprio,
            "wing": wing,
            "wing_mechanosensory": wing,
            "leg_tactile": leg_tactile, "leg_proprioceptive": leg_proprio,
            "leg": leg, "leg_mechanosensory": leg,
        }
        return channels, raw, drives

    def _noise_mapping(self) -> Dict[str, float]:
        # Diagnostics must not expose a mutable reference to cached settings.
        return dict(self._noise_map)

    def prepare_step(self, body_state: Any, *, world_wind: Any = (0.0, 0.0, 0.0),
                     wind_world: Any = None, wind: Any = None,
                     ground_contact: Any = None, contact: Any = None,
                     joint_state: Any = None, joints: Any = None, joint: Any = None) -> PreparedStep:
        """Normalize inputs and compute one immutable, uncommitted sample.

        This method deliberately does not advance delayed history, the last
        timestamp, or the noise RNG.  The returned object must be handed to
        :meth:`commit_prepared` on this same adapter, in preparation order.
        """

        state = BodyState.from_any(body_state)
        if wind_world is not None:
            world_wind = wind_world
        if wind is not None:
            world_wind = wind
        if ground_contact is None:
            ground_contact = contact
        if ground_contact is None:
            ground_contact = _get(body_state, "contact", "grounded", "legs_on_ground", default=None)
        if joint_state is None:
            joint_state = joints if joints is not None else (joint if joint is not None else body_state)
        channels, raw, drives = self._compute(state, world_wind, ground_contact, joint_state)
        prepared = PreparedStep(
            state=state,
            channels=channels,
            raw_channels=raw,
            population_drive=drives,
            _owner_token=self._prepared_owner,
            _sequence=self._next_prepare_sequence,
        )
        self._next_prepare_sequence += 1
        self._pending_prepared.append(prepared)
        return prepared

    def commit_prepared(self, prepared: PreparedStep) -> SensoryFrame:
        """Commit one prepared sample and return the normal sensory frame.

        Ownership and FIFO checks happen before any causal state or RNG
        mutation.  A prepared value therefore cannot be replayed, committed
        on another adapter, or committed out of order.
        """

        if not isinstance(prepared, PreparedStep):
            raise TypeError("prepared must be a PreparedStep")
        if prepared._owner_token is not self._prepared_owner:
            raise ValueError("prepared step belongs to another feedback adapter")
        if not self._pending_prepared or self._pending_prepared[0] is not prepared:
            raise ValueError("prepared steps must be committed once and in preparation order")
        frame = self._commit_prepared(prepared)
        self._pending_prepared.popleft()
        return frame

    def _commit_prepared(self, prepared: PreparedStep) -> SensoryFrame:
        """Stateful half of :meth:`step`, preserving the legacy order."""

        state = prepared.state
        channels = dict(prepared.channels)
        raw = dict(prepared.raw_channels)
        drives = dict(prepared.population_drive)
        sample = (state.timestamp_s, channels, raw, drives)
        monotonic = self._last_timestamp is None or state.timestamp_s >= self._last_timestamp
        self._last_timestamp = state.timestamp_s

        source_timestamp: Optional[float] = None
        if self.delay_s == 0.0:
            # No delayed sample can be selected in this mode.  Avoid retaining
            # every frame; clearing also handles a caller changing delay_s
            # from a delayed run to zero between samples.
            self._history.clear()
            self._history_monotonic = True
            selected = sample
            source_timestamp = selected[0]
        else:
            if not monotonic:
                self._history_monotonic = False
            # For equal timestamps the legacy ``max`` selection keeps the
            # first sample, so later ties can be discarded without changing
            # delayed output or growing a constant-time history forever.
            if not (
                self._history_monotonic
                and self._history
                and self._history[-1][0] == sample[0]
            ):
                self._history.append(sample)
            cutoff = state.timestamp_s - self.delay_s
            if self._history_monotonic:
                # Timestamps are nondecreasing, so keep the newest eligible
                # sample at the head and all not-yet-eligible samples after it.
                while len(self._history) >= 2 and self._history[0][0] <= cutoff + 1e-12:
                    # ``max(..., key=timestamp)`` historically kept the
                    # first sample when timestamps tied.  Skip an old head
                    # only when a later, strictly newer eligible timestamp
                    # exists; this preserves duplicate-timestamp behavior.
                    head_timestamp = self._history[0][0]
                    next_distinct: Optional[float] = None
                    iterator = iter(self._history)
                    next(iterator)
                    for candidate in iterator:
                        if candidate[0] > head_timestamp:
                            next_distinct = candidate[0]
                            break
                    if next_distinct is None or next_distinct > cutoff + 1e-12:
                        break
                    self._history.popleft()
                if self._history and self._history[0][0] <= cutoff + 1e-12:
                    selected = self._history[0]
                    source_timestamp = selected[0]
            else:
                # Preserve the original semantics for out-of-order input:
                # select the maximum eligible timestamp from retained history.
                eligible = [item for item in self._history if item[0] <= cutoff + 1e-12]
                if eligible:
                    selected = max(eligible, key=lambda item: item[0])
                    source_timestamp = selected[0]
        if source_timestamp is None:
            selected = (
                state.timestamp_s,
                {key: 0.0 for key in channels},
                {key: 0.0 for key in raw},
                {key: 0.0 for key in drives},
            )
        _, selected_channels, selected_raw, selected_drives = selected

        output_channels: Dict[str, float] = {}
        output_drives: Dict[str, float] = {}
        clipped: list[str] = []
        ready = source_timestamp is not None
        if ready:
            for name, value in selected_channels.items():
                noisy = float(value) + self._noise(name)
                limited = _clip(noisy, self.saturation)
                output_channels[name] = limited
                if limited != noisy:
                    clipped.append(name)
            population_noise: Dict[str, float] = {}
            for name, value in selected_drives.items():
                # Compatibility aliases for one entry (e.g. jo/jo_wind_gravity)
                # share one noise sample.  Otherwise a noisy frame would make
                # equivalent aliases disagree and the engine would correctly
                # reject the ambiguous injection.
                noise_key = self._population_drive_noise_key.get(name, "population:" + name)
                if noise_key not in population_noise:
                    population_noise[noise_key] = self._noise(noise_key)
                noisy = float(value) + population_noise[noise_key]
                # Population inputs have a separate normalized contract.  A
                # caller may use a larger channel saturation for physical
                # telemetry, but neural population drives remain in [-1, 1].
                limited = _clip(noisy, self._population_drive_limit)
                output_drives[name] = limited
                if limited != noisy:
                    clipped.append("population:" + name)
        else:
            output_channels = {name: 0.0 for name in channels}
            output_drives = {name: 0.0 for name in drives}

        channel_units = {
            "attitude_roll": "rad", "attitude_pitch": "rad", "attitude_yaw": "rad",
            "angular_velocity_roll": "rad/s", "angular_velocity_pitch": "rad/s", "angular_velocity_yaw": "rad/s",
            "wind_body_forward": "m/s", "wind_body_lateral": "m/s", "wind_body_vertical": "m/s",
            "wind_body_speed": "m/s", "ground_contact_left": "fraction",
            "ground_contact_right": "fraction", "ground_contact_fraction": "fraction",
            "wing_phase": "cycle", "wing_phase_left": "cycle", "wing_phase_right": "cycle",
            "wing_phase_difference": "cycle", "wing_amplitude_left": "input-unit",
            "wing_amplitude_right": "input-unit", "wing_left_right_imbalance": "input-unit",
            "wing_power_left": "normalized", "wing_power_right": "normalized",
            "wing_power": "normalized",
            "wingbeat_frequency_hz": "Hz",
            "wing_phase_sin": "normalized", "wing_phase_cos": "normalized",
            "wingbeat_frequency": "normalized",
        }
        scale_units = {
            "angle_rad": "rad", "angular_velocity_rad_s": "rad/s", "wind_mps": "m/s",
            "wing_amplitude": "input-unit", "wing_frequency_hz": "Hz",
        }
        drive_limit = self._population_drive_limit
        delay_error = None if source_timestamp is None else state.timestamp_s - source_timestamp
        diagnostics = {
            "dataset": self.resolver.dataset,
            "timestamp_s": state.timestamp_s,
            "source_timestamp_s": source_timestamp,
            "delay_s": self.delay_s,
            "effective_delay_s": delay_error,
            "delay_ready": ready,
            "timestamp_monotonic": monotonic,
            "time_consistent": source_timestamp is None or source_timestamp <= state.timestamp_s + 1e-12,
            "saturation": self.saturation,
            "clipped_signals": sorted(set(clipped)),
            "signal_ranges": {
                "channels": [min(output_channels.values(), default=0.0), max(output_channels.values(), default=0.0)],
                "population_drive": [min(output_drives.values(), default=0.0), max(output_drives.values(), default=0.0)],
            },
            "all_signals_in_range": (
                all(-self.saturation <= value <= self.saturation for value in output_channels.values())
                and all(-drive_limit <= value <= drive_limit for value in output_drives.values())
            ),
            "noise_seed": self.seed,
            "noise_std": self._noise_mapping(),
            "noise_applied": ready and any(value > 0.0 for value in self._noise_map.values()),
            "wing_flapping": bool(
                selected_raw.get("wingbeat_frequency_hz", 0.0) > 1.0e-12
                and selected_raw.get("wing_power", 0.0) > 1.0e-12
                and (selected_raw.get("wing_amplitude_left", 0.0) > 1.0e-12
                     or selected_raw.get("wing_amplitude_right", 0.0) > 1.0e-12)
            ),
            "normalized_signal_unit": "1",
            "population_drive_units": self.resolver.population_drive_units,
            "population_drive_range": list(self.resolver.population_drive_range),
            "unit_diagnostics": {
                "time": "s",
                "input_conventions": {"attitude": "rad", "angular_velocity": "rad/s", "wind": "m/s",
                                       "contact": "fraction [0,1]", "wing_phase": "cycle [0,1)"},
                "channel_units": channel_units,
                "normalization_scales": scale_units,
                "population_drive_units": self.resolver.population_drive_units,
                "population_drive_range": list(self.resolver.population_drive_range),
            },
        }
        return SensoryFrame(
            timestamp_s=state.timestamp_s,
            channels=output_channels,
            population_drive=output_drives,
            raw_channels=dict(selected_raw),
            channel_units=channel_units,
            diagnostics=diagnostics,
            population_metadata=copy.deepcopy(self._population_metadata_cache),
            source_timestamp_s=source_timestamp,
            raw_population_drive=dict(selected_drives),
            population_drive_units=self.resolver.population_drive_units,
            population_drive_range=self.resolver.population_drive_range,
        )

    def step(self, body_state: Any, *, world_wind: Any = (0.0, 0.0, 0.0),
             wind_world: Any = None, wind: Any = None,
             ground_contact: Any = None, contact: Any = None,
             joint_state: Any = None, joints: Any = None, joint: Any = None) -> SensoryFrame:
        return self.commit_prepared(self.prepare_step(
            body_state,
            world_wind=world_wind,
            wind_world=wind_world,
            wind=wind,
            ground_contact=ground_contact,
            contact=contact,
            joint_state=joint_state,
            joints=joints,
            joint=joint,
        ))

    def step_contract(self, body_state: Any, **kwargs: Any) -> Any:
        """Run :meth:`step` and immediately return the shared contract frame."""

        frame = self.step(body_state, **kwargs)
        return frame.to_contract(body_state=body_state)


SensoryFeedback = MechanosensoryFeedback


def compute_feedback(body_state: Any, *, world_wind: Any = (0.0, 0.0, 0.0),
                     wind_world: Any = None, wind: Any = None,
                     ground_contact: Any = None, contact: Any = None,
                     joint_state: Any = None, joints: Any = None, joint: Any = None,
                     **kwargs: Any) -> SensoryFrame:
    """Convenience wrapper for one zero-delay sample."""

    sensor = MechanosensoryFeedback(**kwargs)
    return sensor.step(body_state, world_wind=world_wind, wind_world=wind_world, wind=wind,
                       ground_contact=ground_contact, contact=contact,
                       joint_state=joint_state, joints=joints, joint=joint)


if __name__ == "__main__":
    sensor = MechanosensoryFeedback(delay_s=0.01, seed=7)
    sensor.step(BodyState(0.0), world_wind=(1.0, 0.0, 0.0),
                ground_contact={"front_left": True},
                joint_state={"left_phase": 0.1, "right_phase": 0.2,
                             "left_amplitude": 0.8, "right_amplitude": 0.6})
    frame = sensor.step(BodyState(0.02, angular_velocity_rad_s=(0.0, 0.0, 5.0)),
                        world_wind=(1.0, 0.5, 0.0),
                        joint_state={"left_phase": 0.4, "right_phase": 0.2,
                                     "left_amplitude": 0.9, "right_amplitude": 0.6})
    print(json.dumps(frame.to_dict(), indent=2, sort_keys=True))
