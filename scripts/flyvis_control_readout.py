#!/usr/bin/env python3
r"""Bounded control cues from signed FlyVis T4/T5 responses.

The fitted FlyVis front end returns one *signed* response for each of the
13,582 T4/T5 targets in ``artifacts/flyvis-eye/t4t5_dirs.npz``.  A scalar
mean of that vector loses the response direction and, in particular, makes a
left/right push--pull pair look like a weak or zero signal.  This module is a
small, dependency-light readout boundary for the visual loop:

``signed T4/T5 response -> direction-weighted yaw/pitch cues``

For target ``i`` the asset supplies a head-frame preferred direction ``d_i``.
The readout computes the signed, weighted vector

.. math:: v = \sum_i a_i r_i d_i / \sum_i a_i |r_i|

and exposes ``v_y`` as a yaw cue and ``v_z`` as a pitch cue.  The denominator
is an L1 response energy, so the cues are bounded before the configured final
clip and do not depend on the number of T4/T5 cells.  Positive and negative
responses remain positive and negative throughout.  Side and subtype metadata
are retained in per-side/per-subtype diagnostics and participate in the
static weighting, rather than being reduced to an unlabelled population mean.

This is deliberately a control *cue* readout.  It does not construct a
``MotorCommand``, call a body adapter, or inject a drive into MaleCNS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_TARGETS = ROOT / "artifacts" / "flyvis-eye" / "t4t5_dirs.npz"
# ``t4t5_dirs.npz`` was generated before the readout contract carried an
# explicit side field.  Its directions are preferred directions, not eye
# labels: 577 of the 13,582 rows cross the head midline, so ``direction_y``
# cannot be used as a production side classifier.  Keep the authoritative
# MaleCNS annotation source explicit and auditable instead.
DEFAULT_CSR = Path("/data/cache/malecns-gpu/csr-flyb")
DEFAULT_SIDE_AUTHORITY = "MaleCNS annotations.npz:sideIdx -> tables.json:sides"
DEFAULT_CONFIG = ROOT / "config" / "flyvis_control_readout.json"

SCHEMA_VERSION = "flyvis-control-readout-1"
SUPPORTED_SIDES = frozenset({"L", "R"})
DEFAULT_BOUNDS = (-1.0, 1.0)


class ControlReadoutError(ValueError):
    """Base class for malformed mapping, input, or readout configuration."""


class MappingError(ControlReadoutError):
    """Raised when the T4/T5 order or metadata cannot be trusted."""


class ReadoutShapeError(ControlReadoutError):
    """Raised when a response vector has an unexpected shape or value."""


class ControlReadoutDisabledError(ControlReadoutError):
    """Raised when a caller explicitly loads a disabled readout config."""


# Readable aliases for callers that use the longer names.
FlyVisControlReadoutError = ControlReadoutError
FlyVisControlMappingError = MappingError
FlyVisControlShapeError = ReadoutShapeError
FlyVisControlDisabledError = ControlReadoutDisabledError


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ControlReadoutError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlReadoutError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ControlReadoutError(f"{name} must be a finite number")
    return number


def _integer_vector(value: Any, name: str) -> np.ndarray:
    """Read dense target indices without silently truncating floats."""

    if isinstance(value, (str, bytes)):
        raise MappingError(f"{name} must be a one-dimensional integer array")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise MappingError(f"{name} must be a one-dimensional integer array") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise MappingError(f"{name} must be a non-empty one-dimensional array")
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise MappingError(f"{name} must contain integer values")
    try:
        numeric = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise MappingError(f"{name} must contain integer values") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise MappingError(f"{name} must contain finite integer values")
    result = numeric.astype(np.int64)
    if np.any(result < 0):
        raise MappingError(f"{name} must contain non-negative indices")
    if np.unique(result).size != result.size:
        raise MappingError(f"{name} must contain unique indices")
    return result.copy()


def _text_vector(value: Any, name: str, *, expected: Optional[int] = None) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise MappingError(f"{name} must be a one-dimensional text array")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise MappingError(f"{name} must be a one-dimensional text array") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise MappingError(f"{name} must be a non-empty one-dimensional text array")
    result = np.asarray(
        [item.decode() if isinstance(item, bytes) else str(item) for item in raw],
        dtype=str,
    )
    if expected is not None and result.shape != (expected,):
        raise MappingError(f"{name} must have shape ({expected},), got {result.shape}")
    if any(not item.strip() for item in result):
        raise MappingError(f"{name} contains an empty label")
    return result


def _directions(value: Any, name: str, *, expected: Optional[int] = None) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MappingError(f"{name} must be a numeric (N, 3) array") from exc
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] == 0:
        raise MappingError(f"{name} must have shape (N, 3), got {raw.shape}")
    if expected is not None and raw.shape != (expected, 3):
        raise MappingError(f"{name} must have shape ({expected}, 3), got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise MappingError(f"{name} contains NaN or infinite values")
    norm = np.linalg.norm(raw, axis=1)
    if np.any(norm <= 1.0e-12):
        raise MappingError(f"{name} contains a zero direction vector")
    return (raw / norm[:, None]).astype(np.float32)


def _normalise_side(value: Any, name: str, *, expected: int) -> np.ndarray:
    raw = _text_vector(value, name, expected=expected)
    aliases = {"left": "L", "l": "L", "right": "R", "r": "R"}
    result = np.asarray([aliases.get(item.strip().lower(), item.strip().upper()) for item in raw], dtype=str)
    if np.any(~np.isin(result, tuple(SUPPORTED_SIDES))):
        bad = sorted(set(result[~np.isin(result, tuple(SUPPORTED_SIDES))]))
        raise MappingError(f"{name} contains unsupported side labels: {bad}")
    return result


def _optional_float_vector(value: Any, name: str, *, expected: int) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MappingError(f"{name} must be a numeric one-dimensional array") from exc
    if raw.shape != (expected,) or not np.all(np.isfinite(raw)):
        raise MappingError(f"{name} must have finite shape ({expected},)")
    if np.any(raw < 0.0):
        raise MappingError(f"{name} must be non-negative")
    return raw.astype(np.float32)


def _mapping_value(source: Any, *names: str) -> Any:
    if source is None:
        return None
    present, value = _source_field(
        source, names, skip_none=not isinstance(source, Mapping)
    )
    return value if present else None


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _sha256_files(paths: Sequence[Path]) -> Optional[str]:
    """Hash the exact metadata files used to resolve an effective mapping."""

    digest = hashlib.sha256()
    for path in paths:
        try:
            digest.update(str(path.resolve()).encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None
    return digest.hexdigest()


def _array_digest(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


_MISSING = object()


def _source_field(
    source: Any,
    names: Sequence[str],
    *,
    skip_none: bool = False,
) -> tuple[bool, Any]:
    """Return the first published field while preserving missing-vs-None."""

    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return True, source[name]
        return False, None
    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if not skip_none or value is not None:
                return True, value
    return False, None


def _field_fingerprint(value: Any) -> tuple[str, str, tuple[int, ...], str]:
    """Fingerprint one mapping field without retaining a mutable array view.

    Runtime mappings are normally NumPy arrays.  Unsupported/object values are
    deliberately identified by object identity, so a cache hit can never skip
    validation of an opaque mutable object.
    """

    if value is None:
        return ("none", "", (), "")
    if isinstance(value, (str, bytes, int, float, bool)):
        return ("scalar", type(value).__name__, (), repr(value))
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return ("opaque", type(value).__qualname__, (), str(id(value)))
    if array.dtype.kind == "O":
        try:
            payload = repr(array.tolist()).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            return ("object", str(array.dtype), tuple(int(size) for size in array.shape), digest)
        except (TypeError, ValueError, RuntimeError):
            return ("opaque", str(array.dtype), tuple(int(size) for size in array.shape), str(id(value)))
    try:
        digest = _array_digest(array)
    except (TypeError, ValueError, RuntimeError):
        return ("opaque", str(array.dtype), tuple(int(size) for size in array.shape), str(id(value)))
    return (
        "array",
        str(array.dtype),
        tuple(int(size) for size in array.shape),
        digest,
    )


def _field_identity(value: Any) -> int:
    """Use identity for mutable containers, content for immutable scalars."""

    if value is None or isinstance(value, (str, bytes, int, float, bool, tuple)):
        return 0
    return id(value)


def _call_mapping_fingerprint(metadata: Mapping[str, Any]) -> tuple[Any, tuple[tuple[Any, ...], ...]] | None:
    """Build an identity/content key for call-time mapping validation.

    The object reference is retained by the caller's cache, preventing Python
    ``id`` reuse.  Field identities plus content fingerprints detect both a
    replaced array and in-place mutation before validation is skipped.
    """

    nested_present = "mapping" in metadata and metadata.get("mapping") is not None
    nested = metadata.get("mapping") if nested_present else _MISSING
    records: list[tuple[Any, ...]] = []
    if nested_present:
        nested_fields = {
            "target_indices": ("target_indices", "target_drive_indices", "idx"),
            "directions": ("directions", "dirs", "target_directions"),
            "subtype": ("subtype", "type", "types", "target_type"),
            "side": ("side", "sides", "target_side"),
            "target_ids": ("target_ids", "ids"),
        }
        for canonical, names in nested_fields.items():
            present, value = _source_field(nested, names)
            records.append(("nested", canonical, present, _field_identity(value) if present else 0,
                            _field_fingerprint(value) if present else None))

    explicit_fields = {
        "target_indices": ("target_indices", "flyvis_target_indices"),
        "target_ids": ("target_ids",),
        "directions": ("directions", "dirs"),
        "subtype": ("subtype", "type"),
        "side": ("side",),
        "units": ("units", "flyvis_units"),
    }
    for canonical, names in explicit_fields.items():
        present, value = _source_field(metadata, names)
        records.append(("explicit", canonical, present, _field_identity(value) if present else 0,
                        _field_fingerprint(value) if present else None))
    if not records or not any(record[2] for record in records):
        return None
    return nested, tuple(records)


@dataclass(frozen=True)
class FlyVisTargetMapping:
    """Validated order and metadata for one T4/T5 response vector.

    ``target_indices`` are dense MaleCNS indices.  They are kept in exactly
    the order expected by a FlyVis response; no sort or reindex is performed
    by the readout.  ``target_ids`` is optional because the shipped direction
    asset contains dense indices but not the separate ``ids.npy`` values.
    """

    target_indices: np.ndarray
    directions: np.ndarray
    subtype: np.ndarray
    side: np.ndarray
    target_ids: Optional[np.ndarray] = None
    asset_weights: Optional[np.ndarray] = None
    source: str = "runtime_mapping"
    mapping_sha256: Optional[str] = None
    side_source: Optional[str] = None
    side_source_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        indices = _integer_vector(self.target_indices, "target_indices")
        n_targets = int(indices.size)
        dirs = _directions(self.directions, "directions", expected=n_targets)
        subtype = _text_vector(self.subtype, "subtype", expected=n_targets)
        # The asset is specifically a T4/T5 boundary.  Rejecting another
        # population here prevents an accidental dense-vector permutation.
        invalid = ~np.char.startswith(subtype, "T4") & ~np.char.startswith(subtype, "T5")
        if np.any(invalid):
            raise MappingError(
                "target mapping contains non-T4/T5 subtype labels: "
                f"{sorted(set(subtype[invalid]))}"
            )
        side = _normalise_side(self.side, "side", expected=n_targets)
        target_ids = None
        if self.target_ids is not None:
            target_ids = _integer_vector(self.target_ids, "target_ids")
            if target_ids.shape != (n_targets,):
                raise MappingError(
                    f"target_ids must have shape ({n_targets},), got {target_ids.shape}"
                )
        weights = None
        if self.asset_weights is not None:
            weights = _optional_float_vector(self.asset_weights, "asset_weights", expected=n_targets)
        digest = self.mapping_sha256
        if digest is None:
            digest = _array_digest(indices, dirs, subtype, side)
        object.__setattr__(self, "target_indices", indices)
        object.__setattr__(self, "directions", dirs)
        object.__setattr__(self, "subtype", subtype)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "asset_weights", weights)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "mapping_sha256", str(digest))
        if self.side_source is not None:
            object.__setattr__(self, "side_source", str(self.side_source))
        if self.side_source_sha256 is not None:
            object.__setattr__(self, "side_source_sha256", str(self.side_source_sha256))

    @property
    def n_targets(self) -> int:
        return int(self.target_indices.size)

    @property
    def directions_array(self) -> np.ndarray:
        return self.directions.copy()

    @property
    def subtype_array(self) -> np.ndarray:
        return self.subtype.copy()

    @property
    def side_array(self) -> np.ndarray:
        return self.side.copy()

    def manifest(self) -> dict[str, Any]:
        counts = {side: int(np.count_nonzero(self.side == side)) for side in ("L", "R")}
        subtypes = {
            str(name): int(np.count_nonzero(self.subtype == name))
            for name in sorted(set(self.subtype.tolist()))
        }
        return {
            "source": self.source,
            "mapping_sha256": self.mapping_sha256,
            "side_authority": (
                DEFAULT_SIDE_AUTHORITY if self.side_source is not None
                else "explicit mapping side field"
            ),
            "side_source": self.side_source,
            "side_source_sha256": self.side_source_sha256,
            "target_count": self.n_targets,
            "side_counts": counts,
            "subtype_counts": subtypes,
            "target_indices_min": int(self.target_indices.min()),
            "target_indices_max": int(self.target_indices.max()),
        }


def _mapping_from_source(source: Any, *, source_name: str = "runtime_mapping") -> FlyVisTargetMapping:
    """Build a validated mapping from a runtime object or mapping dictionary."""

    if isinstance(source, FlyVisTargetMapping):
        return source
    indices = _mapping_value(source, "target_indices", "target_drive_indices", "idx")
    directions = _mapping_value(source, "directions", "dirs", "target_directions")
    subtype = _mapping_value(source, "subtype", "type", "types", "target_type")
    side = _mapping_value(source, "side", "sides", "target_side")
    target_ids = _mapping_value(source, "target_ids", "ids")
    weights = _mapping_value(source, "asset_weights", "weights", "w")
    if indices is None or directions is None or subtype is None:
        raise MappingError(
            "runtime mapping must provide target_indices, directions, and subtype/type"
        )
    dirs = _directions(directions, "directions")
    if side is None:
        raise MappingError(
            "runtime mapping must provide explicit side metadata; "
            "direction_y side inference is forbidden"
        )
    return FlyVisTargetMapping(
        target_indices=indices, directions=dirs, subtype=subtype, side=side,
        target_ids=target_ids, asset_weights=weights, source=source_name,
    )


def _load_authoritative_side_metadata(
    target_path: Path,
    side_source: Path,
) -> tuple[dict[str, np.ndarray], Optional[str]]:
    """Resolve side/type/IDs through the same CSR loader as the runtime.

    The runtime adapter already validates ``idx``/``dirs``/``type`` against
    the MaleCNS annotation tables.  Reusing that read-only loader here keeps
    the control readout and fitted runtime on one canonical side convention,
    while avoiding a second subtly different table decoder.
    """

    try:
        from flyvis_runtime_adapter import load_t4t5_targets

        canonical = load_t4t5_targets(target_path, side_source)
    except Exception as exc:
        raise MappingError(
            "could not load authoritative MaleCNS side metadata from "
            f"{side_source}: {exc}"
        ) from exc
    metadata_files = [side_source / "annotations.npz", side_source / "tables.json"]
    ids_path = side_source / "ids.npy"
    if ids_path.is_file():
        metadata_files.append(ids_path)
    return canonical, _sha256_files(metadata_files)


def load_target_mapping(
    target_path: str | Path = DEFAULT_TARGETS,
    *,
    side_source: str | Path | None = None,
) -> FlyVisTargetMapping:
    """Load and validate ``t4t5_dirs.npz`` without importing Torch.

    A production mapping must carry an explicit side field or name an
    authoritative MaleCNS CSR source.  The historical ``direction_y`` sign
    heuristic is intentionally not a fallback: preferred directions can
    cross the eye midline and therefore do not identify the annotated eye.
    """

    path = Path(target_path).expanduser()
    if not path.is_file():
        raise MappingError(f"T4/T5 direction mapping is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            fields = set(loaded.files)
            required = {"idx", "dirs", "type"}
            missing = sorted(required - fields)
            if missing:
                raise MappingError(f"{path} lacks required mapping fields: {missing}")
            indices = np.asarray(loaded["idx"])
            dirs = np.asarray(loaded["dirs"])
            subtype = np.asarray(loaded["type"])
            side = np.asarray(loaded["side"]) if "side" in fields else None
            target_ids = None
            for name in ("target_ids", "ids"):
                if name in fields:
                    target_ids = np.asarray(loaded[name])
                    break
            weights = None
            for name in ("asset_weights", "weights", "w"):
                if name in fields:
                    weights = np.asarray(loaded[name])
                    break
    except MappingError:
        raise
    except Exception as exc:
        raise MappingError(f"could not read T4/T5 direction mapping {path}: {exc}") from exc

    # The checked production asset predates its explicit side field.  Its
    # canonical side source is part of the readout contract; if that source is
    # unavailable, fail closed instead of falling back to a geometric guess.
    source_path: Optional[Path]
    if side_source is None and path.resolve() == DEFAULT_TARGETS.resolve() and side is None:
        source_path = DEFAULT_CSR
    elif side_source is None:
        source_path = None
    else:
        source_path = Path(side_source).expanduser().resolve()

    side_source_hash: Optional[str] = None
    if source_path is not None:
        canonical, side_source_hash = _load_authoritative_side_metadata(path, source_path)
        if not np.array_equal(_integer_vector(indices, "idx"), canonical["target_indices"]):
            raise MappingError(
                "direction asset and authoritative CSR disagree on target index order"
            )
        if not np.allclose(
            _directions(dirs, "dirs"), canonical["directions"], atol=1.0e-6, rtol=0.0
        ):
            raise MappingError(
                "direction asset and authoritative CSR disagree on direction rows"
            )
        if not np.array_equal(
            _text_vector(subtype, "type"), canonical["subtype"]
        ):
            raise MappingError(
                "direction asset and authoritative CSR disagree on subtype rows"
            )
        side = canonical["side"]
        if target_ids is None:
            target_ids = canonical["target_ids"]
        elif not np.array_equal(
            _integer_vector(target_ids, "target_ids"), canonical["target_ids"]
        ):
            raise MappingError(
                "direction asset and authoritative CSR disagree on target IDs"
            )
    elif side is None:
        raise MappingError(
            f"{path} lacks explicit side metadata; provide side_source from "
            "MaleCNS annotations instead of inferring side from direction_y"
        )
    return FlyVisTargetMapping(
        target_indices=indices, directions=dirs, subtype=subtype, side=side,
        target_ids=target_ids, asset_weights=weights, source=str(path),
        mapping_sha256=_sha256_file(path), side_source=None if source_path is None else str(source_path),
        side_source_sha256=side_source_hash,
    )


# Names matching the existing adapter's terminology.
load_flyvis_mapping = load_target_mapping
load_t4t5_mapping = load_target_mapping


def _default_subtype_weights() -> dict[str, float]:
    return {name: 1.0 for name in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")}


@dataclass(frozen=True)
class ControlReadoutSettings:
    """Validated numerical settings for :class:`FlyVisControlReadout`."""

    yaw_sign: float = 1.0
    pitch_sign: float = 1.0
    yaw_gain: float = 1.0
    pitch_gain: float = 1.0
    output_bound: float = 1.0
    energy_scale: float = 1.0
    side_weights: Mapping[str, float] = None  # type: ignore[assignment]
    subtype_weights: Mapping[str, float] = None  # type: ignore[assignment]
    use_asset_weights: bool = False
    allowed_units: tuple[str, ...] = ()
    name: str = "flyvis-control-readout"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("yaw_sign", "pitch_sign", "yaw_gain", "pitch_gain"):
            number = _finite_number(getattr(self, name), name)
            object.__setattr__(self, name, number)
        bound = _finite_number(self.output_bound, "output_bound")
        scale = _finite_number(self.energy_scale, "energy_scale")
        if bound <= 0.0:
            raise ControlReadoutError("output_bound must be positive")
        if scale <= 0.0:
            raise ControlReadoutError("energy_scale must be positive")
        object.__setattr__(self, "output_bound", bound)
        object.__setattr__(self, "energy_scale", scale)
        side_weights = {"L": 1.0, "R": 1.0}
        if self.side_weights is not None:
            if not isinstance(self.side_weights, Mapping):
                raise ControlReadoutError("side_weights must be a mapping")
            for key, value in self.side_weights.items():
                label = str(key).strip().upper()
                if label not in SUPPORTED_SIDES:
                    raise ControlReadoutError(f"unsupported side weight {key!r}")
                number = _finite_number(value, f"side_weights[{label}]")
                if number < 0.0:
                    raise ControlReadoutError(f"side_weights[{label}] must be non-negative")
                side_weights[label] = number
        if not any(number > 0.0 for number in side_weights.values()):
            raise ControlReadoutError("at least one side weight must be positive")
        subtype_weights = _default_subtype_weights()
        if self.subtype_weights is not None:
            if not isinstance(self.subtype_weights, Mapping):
                raise ControlReadoutError("subtype_weights must be a mapping")
            for key, value in self.subtype_weights.items():
                label = str(key)
                number = _finite_number(value, f"subtype_weights[{label}]")
                if number < 0.0:
                    raise ControlReadoutError(f"subtype_weights[{label}] must be non-negative")
                subtype_weights[label] = number
        if not any(number > 0.0 for number in subtype_weights.values()):
            raise ControlReadoutError("at least one subtype weight must be positive")
        if isinstance(self.allowed_units, (str, bytes)):
            units = (str(self.allowed_units),)
        else:
            units = tuple(str(value) for value in (self.allowed_units or ()))
        object.__setattr__(self, "side_weights", side_weights)
        object.__setattr__(self, "subtype_weights", subtype_weights)
        object.__setattr__(self, "allowed_units", units)
        object.__setattr__(self, "use_asset_weights", bool(self.use_asset_weights))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "schema_version", str(self.schema_version))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ControlReadoutSettings":
        if not isinstance(raw, Mapping):
            raise ControlReadoutError("readout config must be a mapping")
        readout = raw.get("readout", raw)
        confidence = raw.get("confidence", {})
        if not isinstance(readout, Mapping):
            raise ControlReadoutError("config.readout must be a mapping")
        if not isinstance(confidence, Mapping):
            raise ControlReadoutError("config.confidence must be a mapping")
        bounds = readout.get("bounds", readout.get("output_bounds", DEFAULT_BOUNDS))
        if isinstance(bounds, (str, bytes)):
            raise ControlReadoutError("readout bounds must contain two numbers")
        try:
            low, high = (_finite_number(bounds[0], "output_bounds[0]"),
                         _finite_number(bounds[1], "output_bounds[1]"))
        except (TypeError, IndexError) as exc:
            raise ControlReadoutError("readout bounds must contain two numbers") from exc
        if low != -high or low >= high or low >= 0.0:
            raise ControlReadoutError("readout bounds must be symmetric around zero")
        units = raw.get("input", {}).get("allowed_units", ()) if isinstance(raw.get("input", {}), Mapping) else ()
        return cls(
            yaw_sign=readout.get("yaw_sign", 1.0),
            pitch_sign=readout.get("pitch_sign", 1.0),
            yaw_gain=readout.get("yaw_gain", 1.0),
            pitch_gain=readout.get("pitch_gain", 1.0),
            output_bound=high,
            energy_scale=confidence.get("energy_scale", 1.0),
            side_weights=readout.get("side_weights", {"L": 1.0, "R": 1.0}),
            subtype_weights=readout.get("subtype_weights", _default_subtype_weights()),
            use_asset_weights=readout.get("use_asset_weights", False),
            allowed_units=units,
            name=raw.get("name", "flyvis-control-readout"),
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
        )


def load_readout_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a JSON readout config and return a detached dictionary."""

    path = Path(config_path).expanduser()
    if not path.is_file():
        raise ControlReadoutError(f"control readout config is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ControlReadoutError(f"could not read control readout config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ControlReadoutError("control readout config root must be a mapping")
    return dict(raw)


@dataclass(frozen=True)
class FlyVisControlReadoutResult:
    """One bounded visual cue sample and diagnostics.

    ``yaw`` and ``pitch`` are dimensionless normalized cues in the configured
    symmetric bound (``[-1, 1]`` by default).  ``optic_flow`` is the same pair
    in ``(yaw, pitch, roll)`` order with a zero roll cue; it is intentionally a
    cue vector, not a fabricated SI ``rad/s`` measurement.
    """

    yaw: float
    pitch: float
    confidence: float
    optic_flow: tuple[float, float, float]
    side_cues: Mapping[str, Mapping[str, Any]]
    subtype_energy: Mapping[str, float]
    response_energy: float
    directional_coherence: float
    valid: bool
    provenance: Mapping[str, Any]

    @property
    def visual_direction(self) -> float:
        """Backward-friendly scalar alias; this is yaw only, never a T4/T5 mean."""

        return self.yaw

    @property
    def yaw_cue(self) -> float:
        return self.yaw

    @property
    def pitch_cue(self) -> float:
        return self.pitch

    @property
    def confidence_score(self) -> float:
        return self.confidence

    @property
    def optic_flow_vector(self) -> tuple[float, float, float]:
        return self.optic_flow

    @property
    def optic_flow_control(self) -> dict[str, float]:
        return {"yaw": self.yaw, "pitch": self.pitch, "roll": 0.0}

    @property
    def left(self) -> Mapping[str, Any]:
        return self.side_cues["L"]

    @property
    def right(self) -> Mapping[str, Any]:
        return self.side_cues["R"]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.provenance

    @property
    def control(self) -> dict[str, float]:
        return {"yaw": self.yaw, "pitch": self.pitch}

    @property
    def bounds(self) -> tuple[float, float]:
        raw = self.provenance.get("output_bounds", DEFAULT_BOUNDS)
        try:
            return float(raw[0]), float(raw[1])
        except (TypeError, IndexError, ValueError):
            return DEFAULT_BOUNDS

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "flow": "optic_flow",
            "optic_flow_cue": "optic_flow",
            "optic_flow_vector": "optic_flow",
            "optic_flow_control": "optic_flow_control",
            "visual_direction": "yaw",
            "yaw_cue": "yaw",
            "pitch_cue": "pitch",
            "yaw_control": "yaw",
            "pitch_control": "pitch",
            "confidence_score": "confidence",
        }
        key = aliases.get(str(key), str(key))
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
            "visual_direction": float(self.visual_direction),
            "optic_flow": [float(value) for value in self.optic_flow],
            "confidence": float(self.confidence),
            "side_cues": _json_safe(self.side_cues),
            "subtype_energy": {str(key): float(value) for key, value in self.subtype_energy.items()},
            "response_energy": float(self.response_energy),
            "directional_coherence": float(self.directional_coherence),
            "valid": bool(self.valid),
            "provenance": _json_safe(self.provenance),
        }

    as_dict = to_dict


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def _extract_response(value: Any) -> tuple[Any, dict[str, Any]]:
    """Extract a response and optional mapping metadata from common adapters."""

    metadata: dict[str, Any] = {}
    if isinstance(value, Mapping):
        metadata.update({str(key): item for key, item in value.items()})
        response = None
        for name in ("values", "drive", "t4t5_drive", "target_drive", "response"):
            if name in value:
                response = value[name]
                break
        if response is None:
            raise ReadoutShapeError(
                "response mapping must contain values, drive, t4t5_drive, target_drive, or response"
            )
        return response, metadata
    response = value
    for name in ("values", "drive", "t4t5_drive", "target_drive", "response"):
        candidate = getattr(value, name, None)
        if candidate is not None:
            response = candidate
            break
    for name in ("target_indices", "target_ids", "directions", "subtype", "side", "units"):
        candidate = getattr(value, name, None)
        if candidate is not None:
            metadata[name] = candidate
    nested = getattr(value, "mapping", None)
    if nested is not None:
        metadata["mapping"] = nested
    return response, metadata


def _response_vector(value: Any, n_targets: int) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ReadoutShapeError("T4/T5 response must be a one-dimensional numeric vector")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ReadoutShapeError("T4/T5 response must be a one-dimensional numeric vector") from exc
    if raw.ndim != 1 or raw.shape != (n_targets,):
        raise ReadoutShapeError(
            f"T4/T5 response must have shape ({n_targets},), got {raw.shape}"
        )
    if raw.dtype.kind not in "biuf":
        raise ReadoutShapeError("T4/T5 response must contain numeric values")
    try:
        vector = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ReadoutShapeError("T4/T5 response must contain numeric values") from exc
    if not np.all(np.isfinite(vector)):
        raise ReadoutShapeError("T4/T5 response contains NaN or infinite values")
    return vector.astype(np.float64, copy=True)


class FlyVisControlReadout:
    """Decode a signed FlyVis T4/T5 vector into bounded visual cues."""

    name = "flyvis_control_readout"
    emits_motor_commands = False

    def __init__(
        self,
        mapping: Any = None,
        *,
        target_mapping: Any = None,
        mapping_path: str | Path | None = None,
        target_path: str | Path | None = None,
        runtime_mapping: Any = None,
        config: Mapping[str, Any] | str | Path | None = None,
        config_path: str | Path | None = None,
        expected_target_count: Optional[int] = None,
        strict_target_count: Optional[bool] = None,
        source: Optional[str] = None,
    ) -> None:
        if mapping is not None and target_mapping is not None:
            raise ControlReadoutError("provide at most one of mapping and target_mapping")
        if target_mapping is not None:
            mapping = target_mapping
        if sum(value is not None for value in (mapping, mapping_path, target_path, runtime_mapping)) > 1:
            raise ControlReadoutError(
                "provide only one of mapping, mapping_path/target_path, or runtime_mapping"
            )
        explicit_path = mapping_path if mapping_path is not None else target_path
        config_mapping_path: Optional[Path] = None
        raw_config: dict[str, Any]
        if config is None and config_path is None:
            raw_config = load_readout_config(DEFAULT_CONFIG)
        elif config_path is not None:
            raw_config = load_readout_config(config_path)
        elif isinstance(config, (str, Path)):
            raw_config = load_readout_config(config)
        elif isinstance(config, Mapping):
            raw_config = dict(config)
        else:
            raise ControlReadoutError("config must be a mapping or JSON path")
        self.config = raw_config
        if raw_config.get("enabled", True) is False:
            raise ControlReadoutDisabledError("FlyVis control readout is disabled by configuration")
        self.settings = ControlReadoutSettings.from_dict(raw_config)
        self.source = str(source if source is not None else raw_config.get("source", "fitted"))
        mapping_raw = raw_config.get("mapping", {})
        if isinstance(mapping_raw, Mapping):
            configured = mapping_raw.get("path")
            if configured:
                config_mapping_path = (ROOT / str(configured)).resolve() if not Path(str(configured)).is_absolute() else Path(str(configured)).resolve()
        mapping_loaded_from_file = runtime_mapping is None and mapping is None
        if runtime_mapping is not None:
            self.mapping = _mapping_from_source(runtime_mapping, source_name="runtime_mapping")
        elif mapping is not None:
            self.mapping = _mapping_from_source(mapping, source_name="runtime_mapping")
        else:
            selected = explicit_path
            if selected is None:
                selected = config_mapping_path if config_mapping_path is not None else DEFAULT_TARGETS
            elif not Path(str(selected)).expanduser().is_absolute():
                # Keep command-line paths convenient from the project root,
                # while still resolving them when a runner changes cwd.
                candidate = ROOT / str(selected)
                selected = candidate if candidate.exists() else selected
            side_source_raw: Any = (
                mapping_raw.get("side_source")
                if isinstance(mapping_raw, Mapping) else None
            )
            if isinstance(side_source_raw, Mapping):
                side_source = side_source_raw.get("path")
                if side_source is None:
                    raise ControlReadoutError(
                        "config.mapping.side_source must provide a path"
                    )
            else:
                side_source = side_source_raw
            if side_source is not None:
                side_source = Path(str(side_source)).expanduser()
                if not side_source.is_absolute():
                    candidate = ROOT / side_source
                    side_source = candidate if candidate.exists() else side_source
            self.mapping = load_target_mapping(selected, side_source=side_source)
        configured_count: Any = None
        if isinstance(mapping_raw, Mapping):
            configured_count = mapping_raw.get("expected_target_count")
        if expected_target_count is not None:
            configured_count = expected_target_count
        if configured_count is not None:
            try:
                count = int(configured_count)
            except (TypeError, ValueError) as exc:
                raise ControlReadoutError("expected_target_count must be a non-negative integer") from exc
            if count < 0:
                raise ControlReadoutError("expected_target_count must be non-negative")
            # A synthetic/runtime mapping is intentionally allowed to have a
            # different size unless the caller explicitly requests strict
            # checking.  Production's default artifact always gets checked.
            selected_path = Path(explicit_path).expanduser().resolve() if explicit_path is not None else None
            is_default_asset = mapping_loaded_from_file and (
                selected_path is None or selected_path == DEFAULT_TARGETS.resolve()
            )
            if strict_target_count is None:
                strict_target_count = expected_target_count is not None or is_default_asset
            if strict_target_count and self.mapping.n_targets != count:
                raise MappingError(
                    f"target mapping has {self.mapping.n_targets} rows, expected {count}"
                )
        self.target_indices = self.mapping.target_indices.copy()
        self.target_ids = None if self.mapping.target_ids is None else self.mapping.target_ids.copy()
        self.directions = self.mapping.directions.copy()
        self.subtype = self.mapping.subtype.copy()
        self.side = self.mapping.side.copy()
        self.n_targets = self.mapping.n_targets
        static = np.ones(self.n_targets, dtype=np.float64)
        for row, (subtype, side) in enumerate(zip(self.subtype, self.side)):
            subtype_key = str(subtype)
            subtype_weight = float(self.settings.subtype_weights.get(
                subtype_key,
                self.settings.subtype_weights.get(subtype_key[:2], 1.0),
            ))
            side_weight = float(self.settings.side_weights[str(side)])
            static[row] = subtype_weight * side_weight
        if self.settings.use_asset_weights:
            if self.mapping.asset_weights is None:
                raise MappingError("config requests asset weights but mapping has no weights/w field")
            # Normalize only to remove arbitrary synapse-count units.  A
            # constant asset weight therefore has no effect on the cue.
            asset = np.asarray(self.mapping.asset_weights, dtype=np.float64)
            mean = float(np.mean(asset[asset > 0.0])) if np.any(asset > 0.0) else 0.0
            if mean <= 0.0:
                raise MappingError("mapping asset weights have no positive entries")
            static *= asset / mean
        if not np.any(static > 0.0):
            raise MappingError("readout weights exclude every T4/T5 target")
        self._static_weights = static
        self._asset_path = self.mapping.source if self.mapping.source != "runtime_mapping" else None
        # Runtime callers present the same fitted mapping on every frame.  The
        # cache is intentionally only a validation fast path; the response
        # projection below always uses this readout's immutable copies.
        self._call_mapping_cache: tuple[Any, tuple[tuple[Any, ...], ...]] | None = None
        self._provenance = self._make_provenance()

    @classmethod
    def from_runtime_mapping(cls, runtime_mapping: Any, **kwargs: Any) -> "FlyVisControlReadout":
        return cls(runtime_mapping=runtime_mapping, **kwargs)

    @classmethod
    def from_artifact(cls, target_path: str | Path = DEFAULT_TARGETS, **kwargs: Any) -> "FlyVisControlReadout":
        return cls(mapping_path=target_path, **kwargs)

    from_npz = from_artifact

    @property
    def target_count(self) -> int:
        return self.n_targets

    @property
    def target_indices_array(self) -> np.ndarray:
        return self.target_indices.copy()

    @property
    def direction_vectors(self) -> np.ndarray:
        return self.directions.copy()

    @property
    def types(self) -> np.ndarray:
        return self.subtype.copy()

    @property
    def sides(self) -> np.ndarray:
        return self.side.copy()

    @property
    def units(self) -> tuple[str, ...]:
        return self.settings.allowed_units

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def _make_provenance(self) -> dict[str, Any]:
        mapping_manifest = self.mapping.manifest()
        return {
            "readout": self.name,
            "schema_version": self.settings.schema_version,
            "source": self.source,
            "algorithm": "signed_weighted_direction_projection",
            "direction_frame": {"x": "forward", "y": "left", "z": "up"},
            "direction_convention": "+x forward, +y left, +z up",
            "yaw_component": "direction_y",
            "pitch_component": "direction_z",
            "signed_responses": True,
            "preserves_signed_responses": True,
            "left_right_preserved": True,
            "preserves_left_right": True,
            "subtype_metadata_used": True,
            "emits_motor_commands": False,
            "control_only": True,
            "output_units": "normalized_cue",
            "output_bounds": [-self.settings.output_bound, self.settings.output_bound],
            "mapping": mapping_manifest,
            "mapping_sha256": mapping_manifest["mapping_sha256"],
            "mapping_source": mapping_manifest["source"],
            "target_count": mapping_manifest["target_count"],
            "mapping_path": self._asset_path,
            "allowed_input_units": list(self.settings.allowed_units),
        }

    def manifest(self) -> dict[str, Any]:
        return self.provenance

    def _validate_call_mapping(self, metadata: Mapping[str, Any]) -> None:
        fingerprint = _call_mapping_fingerprint(metadata)
        cached = self._call_mapping_cache
        if (
            fingerprint is not None
            and cached is not None
            and fingerprint[0] is cached[0]
            and fingerprint[1] == cached[1]
        ):
            return
        self._validate_call_mapping_uncached(metadata)
        # Store only after complete validation.  A malformed or changed
        # mapping therefore cannot poison the fast path.
        if fingerprint is not None:
            self._call_mapping_cache = fingerprint

    def _validate_call_mapping_uncached(self, metadata: Mapping[str, Any]) -> None:
        nested = metadata.get("mapping")
        if nested is not None:
            # A runtime object may carry a complete mapping.  Validate it
            # against the constructor mapping, including order and metadata.
            candidate = _mapping_from_source(nested, source_name="call_mapping")
            self._assert_mapping_equal(candidate)
        indices = metadata.get("target_indices", metadata.get("flyvis_target_indices"))
        if indices is not None:
            try:
                candidate = _integer_vector(indices, "target_indices")
            except MappingError as exc:
                raise MappingError(str(exc)) from exc
            if candidate.shape != self.target_indices.shape or not np.array_equal(candidate, self.target_indices):
                raise MappingError("response target_indices do not match the configured T4/T5 order")
        if self.target_ids is not None:
            if metadata.get("target_ids") is None:
                raise MappingError("response target_ids are required by the configured mapping")
            candidate_ids = _integer_vector(metadata["target_ids"], "target_ids")
            if not np.array_equal(candidate_ids, self.target_ids):
                raise MappingError("response target_ids do not match the configured T4/T5 order")
        for name, expected in (("directions", self.directions), ("dirs", self.directions),
                               ("subtype", self.subtype), ("type", self.subtype),
                               ("side", self.side)):
            if name not in metadata:
                continue
            raw = metadata[name]
            if name in {"directions", "dirs"}:
                candidate = _directions(raw, name, expected=self.n_targets)
                equal = np.allclose(candidate, expected, atol=1.0e-6, rtol=0.0)
            elif name == "side":
                candidate = _normalise_side(raw, name, expected=self.n_targets)
                equal = np.array_equal(candidate, expected)
            else:
                candidate = _text_vector(raw, name, expected=self.n_targets)
                equal = np.array_equal(candidate, expected)
            if not equal:
                raise MappingError(f"response {name} metadata does not match the configured T4/T5 mapping")
        units = metadata.get("units", metadata.get("flyvis_units"))
        if units is not None and self.settings.allowed_units and str(units) not in self.settings.allowed_units:
            raise MappingError(
                f"response units {units!r} are not allowed by this readout: {self.settings.allowed_units}"
            )

    def _assert_mapping_equal(self, candidate: FlyVisTargetMapping) -> None:
        if candidate.n_targets != self.n_targets:
            raise MappingError("runtime mapping target count does not match the configured mapping")
        if not np.array_equal(candidate.target_indices, self.target_indices):
            raise MappingError("runtime mapping target_indices do not match the configured order")
        if self.target_ids is not None:
            if candidate.target_ids is None or not np.array_equal(candidate.target_ids, self.target_ids):
                raise MappingError("runtime mapping target_ids do not match the configured mapping")
        if not np.allclose(candidate.directions, self.directions, atol=1.0e-6, rtol=0.0):
            raise MappingError("runtime mapping directions do not match the configured mapping")
        if not np.array_equal(candidate.subtype, self.subtype) or not np.array_equal(candidate.side, self.side):
            raise MappingError("runtime mapping subtype/side metadata do not match the configured mapping")

    def _cue_pair(self, vector: np.ndarray) -> tuple[float, float]:
        bound = self.settings.output_bound
        yaw = np.clip(self.settings.yaw_sign * self.settings.yaw_gain * vector[1], -bound, bound)
        pitch = np.clip(self.settings.pitch_sign * self.settings.pitch_gain * vector[2], -bound, bound)
        return float(yaw), float(pitch)

    def _side_projection(self, rows: np.ndarray, response: np.ndarray) -> dict[str, Any]:
        denominator = float(np.sum(np.abs(response[rows]) * self._static_weights[rows])) if rows.size else 0.0
        if denominator <= 1.0e-15:
            vector = np.zeros(3, dtype=np.float64)
        else:
            vector = np.sum(
                (response[rows] * self._static_weights[rows])[:, None] * self.directions[rows],
                axis=0,
            ) / denominator
        yaw, pitch = self._cue_pair(vector)
        return {
            "yaw": yaw,
            "pitch": pitch,
            "vector": [float(value) for value in vector],
            "energy": denominator,
            "count": int(rows.size),
        }

    def readout(
        self,
        response: Any,
        *,
        target_mapping: Any = None,
        mapping: Any = None,
        target_indices: Any = None,
        target_ids: Any = None,
        directions: Any = None,
        subtype: Any = None,
        side: Any = None,
        units: Optional[str] = None,
        frame_index: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> FlyVisControlReadoutResult:
        """Return bounded visual cues for one signed T4/T5 response vector.

        ``response`` may be a raw vector, a ``FlyVisDriveOutput``-like object,
        or a mapping with ``values``/``drive``/``t4t5_drive``.  Explicit
        target metadata is checked in exact order when supplied.
        """

        raw_response, metadata = _extract_response(response)
        if target_mapping is not None and mapping is not None:
            raise MappingError("provide at most one of target_mapping and mapping")
        call_mapping = target_mapping if target_mapping is not None else mapping
        if call_mapping is not None:
            metadata["mapping"] = call_mapping
        explicit = {
            "target_indices": target_indices,
            "target_ids": target_ids,
            "directions": directions,
            "subtype": subtype,
            "side": side,
            "units": units,
        }
        for key, value in explicit.items():
            if value is not None:
                metadata[key] = value
        self._validate_call_mapping(metadata)
        vector = _response_vector(raw_response, self.n_targets)
        effective = self._static_weights
        weighted_abs = np.abs(vector) * effective
        energy = float(np.sum(weighted_abs))
        if energy <= 1.0e-15:
            aggregate_vector = np.zeros(3, dtype=np.float64)
            coherence = 0.0
        else:
            resultant = np.sum((vector * effective)[:, None] * self.directions, axis=0)
            aggregate_vector = resultant / energy
            coherence = float(np.linalg.norm(resultant) / energy)
            coherence = float(np.clip(coherence, 0.0, 1.0))
        yaw, pitch = self._cue_pair(aggregate_vector)
        energy_confidence = 1.0 - math.exp(-energy / self.settings.energy_scale)
        confidence = float(np.clip(energy_confidence * (0.5 + 0.5 * coherence), 0.0, 1.0))
        side_cues = {
            label: self._side_projection(np.flatnonzero(self.side == label), vector)
            for label in ("L", "R")
        }
        subtype_energy = {
            label: float(np.sum(weighted_abs[self.subtype == label]))
            for label in sorted(set(self.subtype.tolist()))
        }
        result_provenance = dict(self._provenance)
        result_provenance.update({
            "input_shape": [self.n_targets],
            "input_units": None if metadata.get("units") is None else str(metadata["units"]),
            "response_signed": bool(np.any(vector < 0.0) and np.any(vector > 0.0)),
            "response_min": float(np.min(vector)),
            "response_max": float(np.max(vector)),
            "response_l1": energy,
            "directional_coherence": coherence,
            "energy_confidence": energy_confidence,
            "frame_index": None if frame_index is None else int(frame_index),
            "timestamp_s": None if timestamp_s is None else _finite_number(timestamp_s, "timestamp_s"),
        })
        return FlyVisControlReadoutResult(
            yaw=yaw,
            pitch=pitch,
            confidence=confidence,
            optic_flow=(yaw, pitch, 0.0),
            side_cues=side_cues,
            subtype_energy=subtype_energy,
            response_energy=energy,
            directional_coherence=coherence,
            valid=True,
            provenance=result_provenance,
        )

    __call__ = readout
    decode = readout
    process = readout
    read = readout


# Concise function aliases are convenient in small runners and tests.
ControlReadout = FlyVisControlReadout
ReadoutResult = FlyVisControlReadoutResult
ControlReadoutResult = FlyVisControlReadoutResult
ControlReadoutOutput = FlyVisControlReadoutResult
FlyVisControlOutput = FlyVisControlReadoutResult
FlyVisReadoutResult = FlyVisControlReadoutResult


def build_control_readout(**kwargs: Any) -> FlyVisControlReadout:
    return FlyVisControlReadout(**kwargs)


def create_control_readout(**kwargs: Any) -> FlyVisControlReadout:
    return FlyVisControlReadout(**kwargs)


def readout_flyvis_control(response: Any, **kwargs: Any) -> FlyVisControlReadoutResult:
    """One-shot helper using the shipped direction mapping and config."""

    return FlyVisControlReadout().readout(response, **kwargs)


def readout_t4t5_control(response: Any, **kwargs: Any) -> FlyVisControlReadoutResult:
    return readout_flyvis_control(response, **kwargs)


def readout_t4t5(response: Any, **kwargs: Any) -> FlyVisControlReadoutResult:
    return readout_flyvis_control(response, **kwargs)


bounded_control_readout = readout_flyvis_control
readout_visual_control = readout_flyvis_control


__all__ = [
    "ControlReadout",
    "ControlReadoutError",
    "ControlReadoutDisabledError",
    "ControlReadoutOutput",
    "ControlReadoutResult",
    "ControlReadoutSettings",
    "FlyVisControlMappingError",
    "FlyVisControlReadout",
    "FlyVisControlReadoutError",
    "FlyVisControlDisabledError",
    "FlyVisControlReadoutResult",
    "FlyVisControlOutput",
    "FlyVisControlShapeError",
    "FlyVisTargetMapping",
    "MappingError",
    "ReadoutResult",
    "ReadoutShapeError",
    "build_control_readout",
    "create_control_readout",
    "load_flyvis_mapping",
    "load_readout_config",
    "load_t4t5_mapping",
    "load_target_mapping",
    "readout_flyvis_control",
    "readout_t4t5",
    "readout_t4t5_control",
    "readout_visual_control",
    "bounded_control_readout",
    "FlyVisReadoutResult",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    readout = FlyVisControlReadout(mapping_path=args.targets, config_path=args.config)
    print(json.dumps(readout.manifest(), indent=2, sort_keys=True))
