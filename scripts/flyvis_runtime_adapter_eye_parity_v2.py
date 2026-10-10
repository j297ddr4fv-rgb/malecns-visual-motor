#!/usr/bin/env python3
"""Runtime adapter for the fitted FlyVis optic-lobe network.

This module is deliberately dependency-light at import time.  Importing it
does not import FlyVis, PyTorch, CUDA, or any of their transitive
dependencies.  :class:`FlyVisRuntimeAdapter` imports the runtime only after a
preflight has established that the dedicated FlyVis environment, the fitted
network, and the two MaleCNS mapping assets are present.

The boundary accepted by :class:`stereo_flyvis.OptionalFlyVisBackend` is two
independent vectors of *already sampled* luminance columns::

    ``(left_columns, right_columns)`` -> ``(n_malecns_t4t5,)``

Each vector has one value per FlyVis hexagonal column (721 for the shipped
model).  It is placed directly into FlyVis' ``(batch, frame, 1, columns)``
stimulus buffer; it is never treated as an equirectangular image and never
resampled a second time.  The two eyes are the two samples in one persistent
network state.  T4/T5 activity is returned as a signed difference from a
grey-screen baseline, so a direction-selective push/pull signal is not
rectified by ``max(response, 0)``.

The adapter's callable result is the vector expected by
``OptionalFlyVisBackend``.  The fitted node groups are flattened once into
device-side gather/reduction indices, so a CUDA frame transfers only the
13,582 mapped target values to the host rather than the complete node array.
Mapping metadata is available on the callable as ``target_indices``,
``target_ids``, ``units``, and ``last_output`` for a connectome engine that
wants to inject the values at the exact MaleCNS targets.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_ROOT = Path("/data/environments/projects/flyvis-opticlobe")
DEFAULT_MODEL_ROOT = Path(os.environ.get("FLYVIS_ROOT_DIR", "/data/models/flyvis"))
DEFAULT_TARGETS = ROOT / "artifacts" / "flyvis-eye" / "t4t5_dirs.npz"
DEFAULT_CSR = Path("/data/cache/malecns-gpu/csr-flyb")
DEFAULT_VIEW = "flow/0000/000"
DEFAULT_COLUMNS = 721
DEFAULT_MAX_VRAM_GIB = 15.5
FLYVIS_HZ = 200.0
FLYVIS_DT = 1.0 / FLYVIS_HZ
T4T5_UNITS = "current_mV_per_ms"

_BASE_SUBTYPES = (
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)


class FlyVisRuntimeError(RuntimeError):
    """Base class for closed-fail FlyVis setup/runtime errors."""


class FlyVisUnavailableError(FlyVisRuntimeError):
    """Raised when the dedicated FlyVis environment or model is unavailable."""


class FlyVisAssetError(FlyVisRuntimeError):
    """Raised when a mapping asset is absent, malformed, or inconsistent."""


class FlyVisMappingError(FlyVisRuntimeError):
    """Raised when a fitted FlyVis node cannot map to a MaleCNS target."""


def _site_package_candidates(env_root: Path) -> tuple[Path, ...]:
    """Return possible site-package directories without importing anything."""

    candidates: list[Path] = []
    # The shipped environment is Python 3.12, while this also makes a copied
    # test environment work when tests run under another minor version.
    for version in (
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "python3.12",
        "python3.11",
        "python3.10",
    ):
        candidate = env_root / "lib" / version / "site-packages"
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _flyvis_package_path(env_root: Path) -> Optional[Path]:
    for site in _site_package_candidates(env_root):
        if (site / "flyvis" / "__init__.py").is_file():
            return site
    return None


def _view_path(model_root: Path, view: str) -> Path:
    return model_root / "results" / Path(str(view))


def _results_root(model_root: Path) -> Path:
    """Return the root expected by ``flyvis.NetworkView(root_dir=...)``.

    ``FLYVIS_ROOT_DIR`` names the model/data root, while FlyVis' public
    ``NetworkView`` constructor resolves ``network_dir`` below its
    ``root_dir`` (the package's ``results_dir``).  Passing the former here
    makes it look for ``<model>/flow/...`` instead of the shipped
    ``<model>/results/flow/...``.
    """

    return model_root / "results"


def preflight_flyvis(
    *,
    env_root: str | Path = DEFAULT_ENV_ROOT,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    target_path: str | Path = DEFAULT_TARGETS,
    csr_path: str | Path = DEFAULT_CSR,
    view: str = DEFAULT_VIEW,
) -> dict[str, Any]:
    """Inspect the real FlyVis prerequisites without importing FlyVis/Torch.

    The return value is JSON-safe and intentionally useful to a launcher.  A
    false ``available`` value is a hard setup failure for a real vision
    request; callers must not turn it into a deterministic stub.
    """

    env = Path(env_root).expanduser()
    model = Path(model_root).expanduser()
    targets = Path(target_path).expanduser()
    csr = Path(csr_path).expanduser()
    python = env / "bin" / "python"
    package = _flyvis_package_path(env)
    view_dir = _view_path(model, view)
    required = {
        "environment": env,
        "environment_python": python,
        "flyvis_package": (package / "flyvis" if package else env / "lib" / "flyvis"),
        "model_root": model,
        "model_view": view_dir,
        "target_mapping": targets,
        "csr_annotations": csr / "annotations.npz",
        "csr_tables": csr / "tables.json",
    }
    missing = [
        name for name, path in required.items()
        if name != "flyvis_package" and not path.exists()
    ]
    # A package path is reported separately; do not mistake a coincidental
    # ``env/lib/flyvis`` directory for the real site-packages package.
    if package is None:
        missing.append("flyvis_package")
    reasons = tuple(f"{name}: {required[name]}" for name in missing)
    return {
        "available": not missing,
        "environment": str(env),
        "model_root": str(model),
        "view": str(view),
        "view_path": str(view_dir),
        "target_path": str(targets),
        "csr_path": str(csr),
        "flyvis_package": None if package is None else str(package / "flyvis"),
        "missing": list(missing),
        "reasons": list(reasons),
    }


# Readable aliases for callers that use the shorter terminology.
flyvis_preflight = preflight_flyvis
preflight = preflight_flyvis


def _require_preflight(**kwargs: Any) -> dict[str, Any]:
    report = preflight_flyvis(**kwargs)
    if not report["available"]:
        details = "; ".join(report["reasons"])
        raise FlyVisUnavailableError(
            "real FlyVis adapter prerequisites are unavailable; "
            f"refusing a stub fallback ({details})"
        )
    return report


def _finite_vector(
    value: Any,
    name: str,
    n: int,
    *,
    destination: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Validate an eye vector and either copy it or update a reusable buffer."""

    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector of shape ({n},)") from exc
    if result.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if destination is None:
        return result.copy()
    if destination.shape != (n,) or destination.dtype != np.float32:
        raise ValueError("internal FlyVis input buffer has an incompatible shape or dtype")
    np.copyto(destination, result, casting="unsafe")
    return destination


def _as_text_array(value: Any) -> np.ndarray:
    """Convert bytes/object/string arrays to a stable Unicode array."""

    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        return arr.astype("U")
    if arr.dtype.kind == "U":
        return arr
    return np.asarray([str(x.decode() if isinstance(x, bytes) else x) for x in arr], dtype=str)


def _normalise_directions(value: Any, name: str) -> np.ndarray:
    try:
        dirs = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FlyVisAssetError(f"{name} must be a numeric (N, 3) array") from exc
    if dirs.ndim != 2 or dirs.shape[1] != 3 or dirs.shape[0] == 0:
        raise FlyVisAssetError(f"{name} must have shape (N, 3), got {dirs.shape}")
    if not np.all(np.isfinite(dirs)):
        raise FlyVisAssetError(f"{name} contains NaN or infinite values")
    norm = np.linalg.norm(dirs, axis=1)
    if np.any(norm <= 1.0e-12):
        raise FlyVisAssetError(f"{name} contains a zero direction vector")
    return (dirs / norm[:, None]).astype(np.float32)


def load_t4t5_targets(
    target_path: str | Path = DEFAULT_TARGETS,
    csr_path: str | Path = DEFAULT_CSR,
) -> dict[str, np.ndarray]:
    """Load and validate the real MaleCNS T4/T5 target subset.

    ``idx`` in ``t4t5_dirs.npz`` is the dense MaleCNS neuron index used by the
    LIF engine.  The annotation tables provide the canonical subtype and eye
    side; the asset's ``type`` field is cross-checked when present.
    """

    target = Path(target_path)
    csr = Path(csr_path)
    if not target.is_file():
        raise FlyVisAssetError(f"T4/T5 direction asset is missing: {target}")
    ann_path = csr / "annotations.npz"
    tables_path = csr / "tables.json"
    ids_path = csr / "ids.npy"
    for path in (ann_path, tables_path):
        if not path.is_file():
            raise FlyVisAssetError(f"MaleCNS annotation asset is missing: {path}")
    try:
        with np.load(target, allow_pickle=False) as loaded:
            names = set(loaded.files)
            for required in ("idx", "dirs", "type"):
                if required not in names:
                    raise FlyVisAssetError(f"{target} lacks required field {required!r}")
            idx = np.asarray(loaded["idx"], dtype=np.int64)
            dirs = _normalise_directions(loaded["dirs"], "t4t5_dirs.dirs")
            subtype_asset = _as_text_array(loaded["type"])
            side_asset = _as_text_array(loaded["side"]) if "side" in names else None
    except FlyVisAssetError:
        raise
    except Exception as exc:
        raise FlyVisAssetError(f"could not read {target}: {exc}") from exc
    if (
        idx.ndim != 1 or idx.size == 0 or dirs.shape[0] != idx.size
        or subtype_asset.shape != idx.shape
        or (side_asset is not None and side_asset.shape != idx.shape)
    ):
        raise FlyVisAssetError(
            "target fields disagree: "
            f"idx={idx.shape}, dirs={dirs.shape}, type={subtype_asset.shape}, "
            f"side={None if side_asset is None else side_asset.shape}"
        )
    if np.unique(idx).size != idx.size or np.any(idx < 0):
        raise FlyVisAssetError("T4/T5 target indices must be unique non-negative dense indices")

    try:
        with np.load(ann_path, allow_pickle=False) as loaded:
            annotations = {str(key): np.asarray(loaded[key]) for key in loaded.files}
        tables_raw = json.loads(tables_path.read_text())
    except Exception as exc:
        raise FlyVisAssetError(f"could not read MaleCNS annotations: {exc}") from exc
    tables = {str(key): np.asarray(value) for key, value in tables_raw.items()}
    n_annotations = int(np.asarray(annotations.get("typeIdx", ())).size)
    if n_annotations == 0:
        raise FlyVisAssetError("annotations.npz lacks a non-empty typeIdx field")
    if np.any(idx >= n_annotations):
        raise FlyVisAssetError(
            f"T4/T5 target index exceeds annotation table ({idx.max()} >= {n_annotations})"
        )

    type_codes = np.asarray(annotations.get("typeIdx"))
    side_codes = np.asarray(annotations.get("sideIdx"))
    if type_codes.shape != (n_annotations,) or side_codes.shape != (n_annotations,):
        raise FlyVisAssetError("annotation typeIdx and sideIdx must be dense one-dimensional arrays")
    types_table = _as_text_array(tables.get("types", ()))
    sides_table = _as_text_array(tables.get("sides", ()))
    if types_table.size == 0 or sides_table.size == 0:
        raise FlyVisAssetError("tables.json must contain non-empty types and sides tables")
    if np.any(type_codes < 0) or np.any(type_codes >= types_table.size):
        raise FlyVisAssetError("annotation typeIdx contains an invalid table code")
    if np.any(side_codes < 0) or np.any(side_codes >= sides_table.size):
        raise FlyVisAssetError("annotation sideIdx contains an invalid table code")
    subtype = types_table[type_codes[idx]]
    side = sides_table[side_codes[idx]]
    if not np.all(np.char.startswith(subtype, "T4") | np.char.startswith(subtype, "T5")):
        bad = sorted(set(subtype[~(np.char.startswith(subtype, "T4") | np.char.startswith(subtype, "T5"))]))
        raise FlyVisAssetError(f"target asset contains non-T4/T5 annotation types: {bad}")
    if np.any(~np.isin(side, ("L", "R"))):
        bad = sorted(set(side[~np.isin(side, ("L", "R"))]))
        raise FlyVisAssetError(f"T4/T5 target annotations contain unsupported eye sides: {bad}")
    if side_asset is not None and np.any(side_asset != side):
        first = int(np.flatnonzero(side_asset != side)[0])
        raise FlyVisAssetError(
            "T4/T5 direction asset and MaleCNS annotations disagree on eye side at "
            f"target {int(idx[first])}: {side_asset[first]!r} != {side[first]!r}"
        )
    if subtype_asset.shape == subtype.shape and np.any(subtype_asset != subtype):
        first = int(np.flatnonzero(subtype_asset != subtype)[0])
        raise FlyVisAssetError(
            "T4/T5 direction asset and MaleCNS annotations disagree at "
            f"target {int(idx[first])}: {subtype_asset[first]!r} != {subtype[first]!r}"
        )

    target_ids = idx.copy()
    if ids_path.is_file():
        ids = np.asarray(np.load(ids_path, allow_pickle=False), dtype=np.int64)
        if ids.shape != (n_annotations,):
            raise FlyVisAssetError(f"ids.npy must have shape ({n_annotations},), got {ids.shape}")
        target_ids = ids[idx]
    return {
        "target_indices": idx.astype(np.int64, copy=True),
        "target_ids": target_ids.astype(np.int64, copy=True),
        "directions": dirs.astype(np.float32, copy=True),
        "subtype": subtype.astype(str, copy=True),
        "side": side.astype(str, copy=True),
    }


def _normalise_eye_azimuth_parity(
    value: Optional[Mapping[str, int]],
) -> dict[str, int]:
    """Validate explicit handedness for each native eye-coordinate grid.

    +1 is the historical mapping. -1 reflects only the azimuth coordinate and
    must be passed to both ray sampling and nearest target-column mapping.
    """
    if value is None:
        return {"L": 1, "R": 1}
    if not isinstance(value, Mapping) or set(value) != {"L", "R"}:
        raise ValueError("eye_azimuth_parity must map exactly L and R to +1 or -1")
    result: dict[str, int] = {}
    for side in ("L", "R"):
        parity = value[side]
        if isinstance(parity, (bool, np.bool_)) or parity not in (-1, 1):
            raise ValueError("eye_azimuth_parity values must be integer +1 or -1")
        result[side] = int(parity)
    return result


def _hex_to_dirs(
    uv: np.ndarray,
    side: str,
    *,
    azimuth_parity: int = 1,
) -> np.ndarray:
    """Convert FlyVis ``(u, v)`` hex columns to fitted head-frame vectors."""

    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise FlyVisMappingError(f"FlyVis column coordinates must have shape (N, 2), got {uv.shape}")
    if side not in {"L", "R"}:
        raise FlyVisMappingError(f"unsupported eye side {side!r}")
    if isinstance(azimuth_parity, (bool, np.bool_)) or azimuth_parity not in (-1, 1):
        raise FlyVisMappingError("azimuth_parity must be integer +1 or -1")
    azimuth_parity = int(azimuth_parity)
    u, v = uv[:, 0], uv[:, 1]
    x = v / 15.0
    y = (u + v / 2.0) / 15.0
    rr = np.sqrt(x * x + y * y)
    max_rr = float(np.max(rr)) if rr.size else 0.0
    if max_rr > 1.0e-12:
        rr = rr / max_rr
    phi = np.arctan2(y, x)
    # Same fitted eye alignment used to produce t4t5_dirs.npz.  +y in the
    # head frame is the fly's left, hence the independent L/R fits.
    az0 = 70.0 if side == "L" else -70.0
    az = np.radians(az0 + azimuth_parity * rr * 85.0 * np.cos(phi))
    el = np.radians(10.0 + rr * 85.0 * np.sin(phi))
    return np.stack((np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)), axis=1).astype(np.float32)


def _array_sha256(value: Any) -> str:
    """Return the SHA-256 of an array's canonical C-order byte payload.

    The manifest records dtype and shape alongside this digest.  Keeping the
    digest over the payload itself (rather than over a repr) makes it useful
    for independently checking that a launcher and the fitted network agree
    on the exact input-column order.
    """

    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _ordered_hex_columns(
    connectome: Any,
    input_index: Any,
    expected_columns: int,
) -> np.ndarray:
    """Resolve the unique hex columns in FlyVis stimulus-column order.

    FlyVis' ``Stimulus.input_index`` is not merely a shape declaration: each
    row is a cell-type-specific list of node indices and its *second axis* is
    the canonical 721-column order consumed by ``add_input``.  Deriving
    columns from ``np.unique(nodes.u/v)`` happens to match the shipped model,
    but loses that contract if a future connectome stores nodes differently.
    Use the first input row to recover ``(u, v)`` and verify every remaining
    input row has the same ordered geometry before allowing the runtime to
    proceed.
    """

    nodes = getattr(connectome, "nodes", connectome)
    u = np.asarray(_nodes_field(nodes, "u"))
    v = np.asarray(_nodes_field(nodes, "v"))
    if u.ndim != 1 or v.shape != u.shape or u.size == 0:
        raise FlyVisMappingError("FlyVis connectome u/v arrays have inconsistent lengths")
    try:
        indices = np.asarray(input_index)
    except Exception as exc:
        raise FlyVisMappingError(f"could not read FlyVis stimulus input_index: {exc}") from exc
    if indices.ndim != 2 or indices.shape[0] == 0 or indices.shape[1] != expected_columns:
        raise FlyVisMappingError(
            "FlyVis stimulus input_index must have shape "
            f"(n_input_types, {expected_columns}), got {indices.shape}"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise FlyVisMappingError("FlyVis stimulus input_index must contain integer node indices")
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= u.size):
        raise FlyVisMappingError("FlyVis stimulus input_index contains an out-of-range node index")

    ordered = np.stack((u[indices[0]], v[indices[0]]), axis=1)
    if np.unique(ordered, axis=0).shape[0] != expected_columns:
        raise FlyVisMappingError(
            "FlyVis stimulus input_index first row does not expose "
            f"{expected_columns} unique hex columns"
        )
    for row, row_indices in enumerate(indices[1:], start=1):
        row_columns = np.stack((u[row_indices], v[row_indices]), axis=1)
        if not np.array_equal(row_columns, ordered):
            raise FlyVisMappingError(
                "FlyVis stimulus input_index rows disagree on ordered hex columns "
                f"(row 0 versus row {row})"
            )
    # Normalize the integer width so hashes and downstream comparisons are
    # stable across the int32 arrays used by the shipped HDF5 connectome and
    # int64 arrays commonly produced by test doubles.
    return np.asarray(ordered, dtype=np.int32)


def _nodes_field(nodes: Any, name: str) -> np.ndarray:
    value = getattr(nodes, name, None)
    if value is None and isinstance(nodes, Mapping):
        value = nodes.get(name)
    if value is None:
        raise FlyVisMappingError(f"FlyVis connectome nodes lack {name!r}")
    try:
        return np.asarray(value)
    except Exception as exc:
        raise FlyVisMappingError(f"could not read FlyVis connectome nodes.{name}: {exc}") from exc


def _input_cell_type_names(connectome: Any, n_rows: int) -> Optional[tuple[str, ...]]:
    """Return JSON-safe input cell-type labels when FlyVis exposes them."""

    raw = getattr(connectome, "input_cell_types", None)
    if raw is None and isinstance(connectome, Mapping):
        raw = connectome.get("input_cell_types")
    if raw is None:
        return None
    try:
        labels = tuple(str(value.decode() if isinstance(value, bytes) else value) for value in np.asarray(raw).reshape(-1))
    except Exception as exc:
        raise FlyVisMappingError(f"could not read FlyVis input cell types: {exc}") from exc
    if len(labels) != n_rows:
        raise FlyVisMappingError(
            "FlyVis input_cell_types length does not match stimulus.input_index rows: "
            f"{len(labels)} != {n_rows}"
        )
    return labels


def _subtype_family(subtype: str) -> tuple[str, ...]:
    subtype = str(subtype)
    if subtype in _BASE_SUBTYPES:
        return (subtype,)
    if subtype in {"T4_unclear", "T5_unclear"}:
        family = subtype[:2]
        return tuple(item for item in _BASE_SUBTYPES if item.startswith(family))
    unclear_suffix = "_unclear"
    if subtype.endswith(unclear_suffix) and subtype[:-len(unclear_suffix)] in _BASE_SUBTYPES:
        return (subtype[:-len(unclear_suffix)],)
    raise FlyVisMappingError(f"unsupported T4/T5 subtype {subtype!r}")


@dataclass(frozen=True)
class FlyVisTargetMapping:
    """Resolved MaleCNS target rows and their FlyVis node groups."""

    target_indices: np.ndarray
    target_ids: np.ndarray
    directions: np.ndarray
    subtype: np.ndarray
    side: np.ndarray
    node_indices: tuple[np.ndarray, ...]
    units: str = T4T5_UNITS

    @property
    def n_targets(self) -> int:
        return int(self.target_indices.size)


def _mapping_index_arrays(mapping: FlyVisTargetMapping) -> dict[str, np.ndarray]:
    """Flatten target groups for a vectorized gather/reduction.

    ``node_indices`` is intentionally kept in the public mapping object: it is
    useful for diagnostics and is the clearest representation of the fitted
    correspondence.  The runtime path uses these parallel arrays instead of
    iterating over every target row.  Entries for one target remain contiguous
    so a target mean is exactly the same operation as the historical
    ``np.mean(delta[eye, nodes])`` implementation.
    """

    n_targets = mapping.n_targets
    if n_targets <= 0 or len(mapping.node_indices) != n_targets:
        raise FlyVisMappingError("FlyVis mapping has no one-to-one target group table")
    group_sizes = np.empty(n_targets, dtype=np.int64)
    clean_groups: list[np.ndarray] = []
    for row, group in enumerate(mapping.node_indices):
        values = np.asarray(group, dtype=np.int64)
        if values.ndim != 1 or values.size == 0 or np.any(values < 0):
            raise FlyVisMappingError(f"FlyVis target mapping group {row} is empty or invalid")
        group_sizes[row] = values.size
        clean_groups.append(values)
    flat_nodes = np.concatenate(clean_groups).astype(np.int64, copy=False)
    target_rows = np.repeat(np.arange(n_targets, dtype=np.int64), group_sizes)
    # The connectome itself has no eye axis; L/R selects one sample from the
    # persistent two-eye FlyVis batch.  Keep that choice per gathered node.
    eye_rows = np.repeat((np.asarray(mapping.side, dtype=str) == "R").astype(np.int64), group_sizes)
    return {
        "flat_nodes": flat_nodes,
        "target_rows": target_rows,
        "eye_rows": eye_rows,
        "group_sizes": group_sizes,
    }


def _resolve_mapping(
    connectome: Any,
    targets: Mapping[str, np.ndarray],
    *,
    columns: Optional[np.ndarray] = None,
    eye_azimuth_parity: Optional[Mapping[str, int]] = None,
) -> FlyVisTargetMapping:
    parity = _normalise_eye_azimuth_parity(eye_azimuth_parity)
    nodes = getattr(connectome, "nodes", connectome)
    uv = np.stack((_nodes_field(nodes, "u"), _nodes_field(nodes, "v")), axis=1)
    node_type = _as_text_array(_nodes_field(nodes, "type"))
    if uv.shape[0] != node_type.shape[0] or uv.shape[0] == 0:
        raise FlyVisMappingError("FlyVis connectome u/v/type arrays have inconsistent lengths")
    if columns is None:
        # Compatibility for small direct callers.  The production runtime
        # passes the stricter order recovered from ``stimulus.input_index``.
        columns, node_column = np.unique(uv, axis=0, return_inverse=True)
    else:
        columns = np.asarray(columns, dtype=np.int32)
        if columns.ndim != 2 or columns.shape[1] != 2 or columns.shape[0] == 0:
            raise FlyVisMappingError(
                f"ordered FlyVis columns must have shape (N, 2), got {columns.shape}"
            )
        if np.unique(columns, axis=0).shape[0] != columns.shape[0]:
            raise FlyVisMappingError("ordered FlyVis columns must be unique")
        lookup = {tuple(value): index for index, value in enumerate(columns.tolist())}
        try:
            node_column = np.asarray([lookup[tuple(value)] for value in uv], dtype=np.int64)
        except KeyError as exc:
            raise FlyVisMappingError(
                "ordered FlyVis input columns do not cover every connectome node column"
            ) from exc
    if columns.shape[0] == 0:
        raise FlyVisMappingError("FlyVis connectome contains no hex columns")
    col_dirs = {
        side: _hex_to_dirs(columns, side, azimuth_parity=parity[side])
        for side in ("L", "R")
    }
    subtype = np.asarray(targets["subtype"], dtype=str)
    side = np.asarray(targets["side"], dtype=str)
    directions = np.asarray(targets["directions"], dtype=np.float32)
    if not (subtype.shape == side.shape == (directions.shape[0],)):
        raise FlyVisMappingError("target mapping fields have inconsistent lengths")

    # A subtype/column can contain more than one fitted node in a future
    # model.  Keeping all matching node indices makes *_unclear mappings an
    # explicit mean instead of silently selecting an arbitrary node.
    lookup: dict[tuple[str, str, int], np.ndarray] = {}
    for family in _BASE_SUBTYPES:
        present = node_type == family
        for col in range(columns.shape[0]):
            lookup[(family, "L", col)] = np.flatnonzero(present & (node_column == col)).astype(np.int64)
            lookup[(family, "R", col)] = lookup[(family, "L", col)]

    groups: list[np.ndarray] = []
    for row, (direction, target_side, target_subtype) in enumerate(zip(directions, side, subtype)):
        if target_side not in {"L", "R"}:
            raise FlyVisMappingError(f"target row {row} has unsupported side {target_side!r}")
        families = _subtype_family(str(target_subtype))
        scores = col_dirs[target_side] @ direction
        col = int(np.argmax(scores))
        pieces = [lookup[(family, target_side, col)] for family in families]
        if any(piece.size == 0 for piece in pieces):
            missing = [family for family, piece in zip(families, pieces) if piece.size == 0]
            raise FlyVisMappingError(
                f"no FlyVis node for target row {row} subtype={target_subtype!r} "
                f"side={target_side!r} column={columns[col].tolist()}: {missing}"
            )
        groups.append(np.unique(np.concatenate(pieces)).astype(np.int64))
    return FlyVisTargetMapping(
        target_indices=np.asarray(targets["target_indices"], dtype=np.int64).copy(),
        target_ids=np.asarray(targets["target_ids"], dtype=np.int64).copy(),
        directions=directions.copy(), subtype=subtype.copy(), side=side.copy(),
        node_indices=tuple(groups), units=T4T5_UNITS,
    )


def _normalise_device(value: Any) -> str:
    device = str(value).strip().lower()
    if device == "cuda":
        device = "cuda:0"
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("FlyVis device must be 'cpu', 'cuda', or 'cuda:N'")
    if device.startswith("cuda:"):
        try:
            index = int(device.split(":", 1)[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("FlyVis CUDA device must be written as cuda:N") from exc
        if index < 0:
            raise ValueError("FlyVis CUDA device index must be non-negative")
    return device


def _require_cuda_gate() -> None:
    """Check the project safety gate before anything can import Torch/CUDA."""

    try:
        gate = importlib.import_module("malecns_gate")
        gate.require_cuda()
    except Exception as exc:
        raise FlyVisUnavailableError(
            f"FlyVis CUDA safety gate is closed: {exc}"
        ) from exc


def _configure_torch_device(torch: Any, device: str,
                            max_vram_gib: Optional[float]) -> None:
    """Select the requested device and bound PyTorch's caching allocator."""

    torch_device = torch.device(device)
    if device.startswith("cuda:"):
        if not bool(torch.cuda.is_available()):
            raise FlyVisUnavailableError(
                f"FlyVis requested {device}, but torch.cuda.is_available() is false"
            )
        count = int(torch.cuda.device_count())
        index = int(device.split(":", 1)[1])
        if index >= count:
            raise FlyVisUnavailableError(
                f"FlyVis requested {device}, but only {count} CUDA device(s) are visible"
            )
        if max_vram_gib is not None:
            total = int(torch.cuda.get_device_properties(torch_device).total_memory)
            cap_bytes = int(float(max_vram_gib) * (1024 ** 3))
            fraction = min(1.0, cap_bytes / max(1, total))
            torch.cuda.set_per_process_memory_fraction(fraction, torch_device)
    setter = getattr(torch, "set_default_device", None)
    if callable(setter):
        setter(torch_device)


def _import_runtime(env_root: Path, model_root: Path, *, device: str,
                    max_vram_gib: Optional[float]) -> tuple[Any, Any]:
    """Import FlyVis and Torch from the validated environment, on demand.

    CUDA requests are gate-checked before importing either module because
    importing FlyVis imports Torch and selects a default device as a side
    effect.
    """

    site = _flyvis_package_path(env_root)
    if site is None:
        raise FlyVisUnavailableError(f"FlyVis package is absent from {env_root}")
    # Put the dedicated environment first.  The environment currently uses
    # system site packages for Torch, but FlyVis itself must come from here.
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))
    os.environ["FLYVIS_ROOT_DIR"] = str(model_root)
    if device.startswith("cuda:"):
        _require_cuda_gate()
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise FlyVisUnavailableError(f"could not import PyTorch for FlyVis: {exc}") from exc
    _configure_torch_device(torch, device, max_vram_gib)
    existing = sys.modules.get("flyvis")
    if existing is not None:
        origin = str(getattr(getattr(existing, "__spec__", None), "origin", ""))
        if not origin.startswith(str(site / "flyvis")):
            raise FlyVisUnavailableError(
                f"a FlyVis module from outside the validated environment is already loaded: {origin}"
            )
        flyvis = existing
    else:
        try:
            flyvis = importlib.import_module("flyvis")
        except Exception as exc:
            raise FlyVisUnavailableError(f"could not import FlyVis from {site}: {exc}") from exc
    # FlyVis chooses its global device at import time.  Reassert the explicit
    # request so a CPU request on a CUDA host cannot drift back to CUDA.
    flyvis.device = torch.device(device)
    _configure_torch_device(torch, device, max_vram_gib)
    return flyvis, torch


def _call_network_view_factory(factory: Callable[..., Any], view: str, results_root: Path) -> Any:
    """Call an injected/mock factory while accepting common small signatures."""

    attempts = (
        lambda: factory(view, root_dir=results_root),
        lambda: factory(view, results_root),
        lambda: factory(view),
        lambda: factory(),
    )
    last: Optional[Exception] = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last = exc
    raise FlyVisRuntimeError(f"FlyVis NetworkView factory has no supported signature: {last}")


def _state_activity_native(state: Any) -> Any:
    """Return ``nodes.activity`` without moving a tensor off its device.

    The old adapter called ``cpu().numpy()`` here.  That was harmless for the
    tiny mock network, but on the fitted network it copied every node (tens of
    thousands of values) to the host on every 5 ms frame.  Keep this helper
    deliberately free of NumPy conversion so the caller can gather only the
    mapped T4/T5 rows on the native device.
    """

    if state is None:
        raise FlyVisRuntimeError("FlyVis returned no network state")
    nodes = getattr(state, "nodes", None)
    if nodes is None and isinstance(state, Mapping):
        nodes = state.get("nodes", state)
    activity = getattr(nodes, "activity", None)
    if activity is None and isinstance(nodes, Mapping):
        activity = nodes.get("activity")
    if activity is None:
        raise FlyVisRuntimeError("FlyVis network state lacks nodes.activity")

    value = activity
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    # FlyVis' state API is (batch, nodes).  A singleton temporal dimension is
    # tolerated for small mock networks but never averaged across batch eyes.
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            value = np.asarray(value, dtype=np.float32)
            shape = value.shape
        except (TypeError, ValueError) as exc:
            raise FlyVisRuntimeError(f"FlyVis nodes.activity is not numeric: {exc}") from exc
    try:
        shape = tuple(int(size) for size in shape)
    except (TypeError, ValueError) as exc:
        raise FlyVisRuntimeError("FlyVis nodes.activity has no usable shape") from exc
    if len(shape) == 3 and shape[1] == 1:
        try:
            value = value[:, 0, :]
            shape = tuple(int(size) for size in value.shape)
        except Exception as exc:
            raise FlyVisRuntimeError("could not remove FlyVis activity singleton dimension") from exc
    if len(shape) != 2 or shape[0] != 2:
        raise FlyVisRuntimeError(
            f"FlyVis nodes.activity must preserve a two-eye batch, got {shape}"
        )
    return value


def _state_activity(state: Any) -> np.ndarray:
    """Compatibility wrapper returning a host array for small CPU callers.

    The real per-frame path uses :func:`_state_activity_native`; retaining this
    wrapper keeps the module-level diagnostic helper and existing CPU tests
    backwards compatible.
    """

    value = _state_activity_native(state)
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise FlyVisRuntimeError(f"FlyVis nodes.activity is not numeric: {exc}") from exc
    if not np.all(np.isfinite(array)):
        raise FlyVisRuntimeError("FlyVis nodes.activity contains NaN or infinite values")
    return array.copy()


@dataclass(frozen=True)
class FlyVisDriveOutput:
    """One signed mapped FlyVis output with explicit MaleCNS provenance."""

    drive: np.ndarray
    target_indices: np.ndarray
    target_ids: np.ndarray
    units: str
    frame_index: int
    baseline: str

    @property
    def target_drive(self) -> np.ndarray:
        return self.drive

    @property
    def t4t5_drive(self) -> np.ndarray:
        """Name used by the stereo/CNS contracts for the target vector."""
        return self.drive


class FlyVisRuntimeAdapter:
    """Callable real FlyVis backend for ``OptionalFlyVisBackend``.

    Parameters are intentionally explicit so a launcher can report the exact
    fitted model and mapping.  ``network_view_factory`` and ``torch_module``
    are test seams; production callers omit both and trigger the lazy import
    only after preflight succeeds.
    """

    name = "flyvis"
    is_stub = False
    is_recorded = False
    n_columns = DEFAULT_COLUMNS
    units = T4T5_UNITS
    target_units = T4T5_UNITS

    def __init__(
        self,
        *,
        view: str = DEFAULT_VIEW,
        device: str = "cuda:0",
        max_vram_gib: Optional[float] = DEFAULT_MAX_VRAM_GIB,
        env_root: str | Path = DEFAULT_ENV_ROOT,
        model_root: str | Path = DEFAULT_MODEL_ROOT,
        target_path: str | Path = DEFAULT_TARGETS,
        csr_path: str | Path = DEFAULT_CSR,
        baseline_mode: str = "steady_state",
        baseline_value: float = 0.5,
        baseline_activity: Any = None,
        baseline_seconds: float = 1.0,
        output_gain: float = 1.0,
        n_columns: int = DEFAULT_COLUMNS,
        static_batch_frames: int = 32,
        validate_activity_finite: bool = True,
        timing_enabled: bool = True,
        eye_azimuth_parity: Optional[Mapping[str, int]] = None,
        network_view_factory: Optional[Callable[..., Any]] = None,
        network_view: Any = None,
        torch_module: Any = None,
    ) -> None:
        self.view = str(view)
        self.device = _normalise_device(device)
        if max_vram_gib is None:
            self.max_vram_gib = None
        else:
            try:
                self.max_vram_gib = float(max_vram_gib)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_vram_gib must be a positive finite number or None") from exc
            if not np.isfinite(self.max_vram_gib) or self.max_vram_gib <= 0:
                raise ValueError("max_vram_gib must be a positive finite number or None")
        self.env_root = Path(env_root).expanduser()
        self.model_root = Path(model_root).expanduser()
        self.results_root = _results_root(self.model_root)
        self.target_path = Path(target_path).expanduser()
        self.csr_path = Path(csr_path).expanduser()
        try:
            columns = int(n_columns)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_columns must be a positive integer") from exc
        if isinstance(n_columns, bool) or columns <= 0 or columns != n_columns:
            raise ValueError("n_columns must be a positive integer")
        self.n_columns = columns
        self.baseline_mode = str(baseline_mode).lower()
        if self.baseline_mode not in {"steady_state", "fixed", "first", "none"}:
            raise ValueError("baseline_mode must be steady_state, fixed, first, or none")
        try:
            self.baseline_value = float(baseline_value)
            self.baseline_seconds = float(baseline_seconds)
            self.output_gain = float(output_gain)
        except (TypeError, ValueError) as exc:
            raise ValueError("baseline and gain values must be numeric") from exc
        if not np.isfinite(self.baseline_value) or not np.isfinite(self.baseline_seconds) or self.baseline_seconds <= 0:
            raise ValueError("baseline_value must be finite and baseline_seconds must be positive")
        if not np.isfinite(self.output_gain):
            raise ValueError("output_gain must be finite")
        if self.units != T4T5_UNITS:
            raise ValueError(f"signed FlyVis target drive uses {T4T5_UNITS!r} units")
        try:
            static_batch_frames_value = int(static_batch_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError("static_batch_frames must be a positive integer") from exc
        if (isinstance(static_batch_frames, bool)
                or static_batch_frames_value <= 0
                or static_batch_frames_value != static_batch_frames):
            raise ValueError("static_batch_frames must be a positive integer")
        if not isinstance(validate_activity_finite, (bool, np.bool_)):
            raise ValueError("validate_activity_finite must be a boolean")
        if not isinstance(timing_enabled, (bool, np.bool_)):
            raise ValueError("timing_enabled must be a boolean")
        self.static_batch_frames = static_batch_frames_value
        self.validate_activity_finite = bool(validate_activity_finite)
        self.timing_enabled = bool(timing_enabled)
        self.eye_azimuth_parity = _normalise_eye_azimuth_parity(eye_azimuth_parity)

        # Always validate the mapping assets before importing the optional
        # runtime.  This prevents a half-constructed real backend when the
        # target ordering or annotation tables are stale.
        self.preflight = _require_preflight(
            env_root=self.env_root, model_root=self.model_root,
            target_path=self.target_path, csr_path=self.csr_path, view=self.view,
        )
        self.targets = load_t4t5_targets(self.target_path, self.csr_path)

        if network_view is not None and network_view_factory is not None:
            raise ValueError("provide at most one of network_view and network_view_factory")
        if network_view is None:
            if network_view_factory is not None:
                network_view = _call_network_view_factory(network_view_factory, self.view, self.results_root)
            else:
                flyvis, imported_torch = _import_runtime(
                    self.env_root, self.model_root, device=self.device,
                    max_vram_gib=self.max_vram_gib,
                )
                self._flyvis_module = flyvis
                if torch_module is None:
                    torch_module = imported_torch
                factory = getattr(flyvis, "NetworkView", None)
                if not callable(factory):
                    raise FlyVisUnavailableError("FlyVis module exposes no NetworkView")
                try:
                    network_view = factory(self.view, root_dir=self.results_root)
                except Exception as exc:
                    raise FlyVisUnavailableError(
                        f"could not construct FlyVis NetworkView {self.view!r}: {exc}"
                    ) from exc
        self.network_view = network_view
        try:
            self.net = self.network_view.init_network()
        except Exception as exc:
            raise FlyVisUnavailableError(f"could not initialize FlyVis network: {exc}") from exc
        move = getattr(self.net, "to", None)
        if callable(move):
            try:
                moved = move(self.device)
            except Exception as exc:
                raise FlyVisUnavailableError(
                    f"could not move FlyVis network to {self.device}: {exc}"
                ) from exc
            if moved is not None:
                self.net = moved
        eval_method = getattr(self.net, "eval", None)
        if callable(eval_method):
            eval_method()
        self.torch = torch_module
        if self.torch is None:
            # This branch is only reachable with an injected NetworkView.  It
            # still imports lazily and keeps tests from importing Torch at
            # module import time.
            try:
                self.torch = importlib.import_module("torch")
            except Exception as exc:
                raise FlyVisUnavailableError(f"could not import PyTorch for the injected network: {exc}") from exc
        self.connectome = getattr(self.network_view, "connectome", getattr(self.net, "connectome", None))
        if self.connectome is None:
            raise FlyVisMappingError("FlyVis NetworkView exposes no connectome")

        stimulus = getattr(self.net, "stimulus", None)
        if stimulus is None:
            raise FlyVisRuntimeError("FlyVis network exposes no stimulus buffer")
        input_index = np.asarray(getattr(stimulus, "input_index", ()))
        self._column_uv = _ordered_hex_columns(
            self.connectome, input_index, self.n_columns
        )
        # These are the exact direction samples consumed by the two eyes.
        # They deliberately reuse the fitted _hex_to_dirs transform and the
        # input_index-derived column order; callers must not reconstruct them
        # with a separately sorted lattice.
        self._column_directions = {
            side: _hex_to_dirs(
                self._column_uv, side,
                azimuth_parity=self.eye_azimuth_parity[side],
            )
            for side in ("L", "R")
        }
        self._input_index = input_index.astype(np.int64, copy=True)
        self._input_cell_types = _input_cell_type_names(self.connectome, input_index.shape[0])

        self.mapping = _resolve_mapping(
            self.connectome, self.targets, columns=self._column_uv,
            eye_azimuth_parity=self.eye_azimuth_parity,
        )
        self.target_indices = self.mapping.target_indices.copy()
        self.target_ids = self.mapping.target_ids.copy()
        self._output_target_indices = np.ascontiguousarray(
            self.target_indices
        ).copy()
        self._output_target_ids = np.ascontiguousarray(self.target_ids).copy()
        self._output_target_indices.setflags(write=False)
        self._output_target_ids.setflags(write=False)
        self.target_drive_indices = self.target_indices
        self.n_t4t5 = self.mapping.n_targets
        self.target_count = self.n_t4t5
        self.unit_map = {"target_drive": self.units, "target_indices": "dense_neuron_index"}

        self._network_nodes = int(_nodes_field(getattr(self.connectome, "nodes", self.connectome), "type").size)
        mapping_arrays = _mapping_index_arrays(self.mapping)
        if np.any(mapping_arrays["flat_nodes"] >= self._network_nodes):
            raise FlyVisMappingError(
                "FlyVis target mapping references a node outside the network activity array"
            )
        self._mapping_flat_nodes = mapping_arrays["flat_nodes"]
        self._mapping_target_rows = mapping_arrays["target_rows"]
        self._mapping_eye_rows = mapping_arrays["eye_rows"]
        self._mapping_group_sizes = mapping_arrays["group_sizes"]
        self._mapping_group_sizes_f32 = self._mapping_group_sizes.astype(np.float32)

        self._baseline_activity_config = None if baseline_activity is None else np.asarray(baseline_activity, dtype=np.float32)
        self._state: Any = None
        self._baseline_activity: Any = None
        self._frame = 0
        self.last_output: Optional[FlyVisDriveOutput] = None
        # A strict escape run may initialise the fitted network on the exact
        # static scene that it is about to observe.  Keep this provenance
        # separate from the ordinary grey/steady-state baseline: it is a
        # sensory state initialisation record, not a controller or a response
        # frame.  ``reset`` clears it and ``preadapt_static_scene`` replaces
        # it after retaining the final native network state.
        self._scene_baseline_manifest: Optional[dict[str, Any]] = None
        # Host-side stage timings are intentionally cumulative and JSON-safe.
        # A native target transfer synchronizes CUDA, so the mapping stage
        # includes the only unavoidable per-frame device-to-host wait.
        self._timing_totals = {
            "frames": 0,
            "scene_frames": 0,
            "input_prepare_s": 0.0,
            "stimulus_s": 0.0,
            "network_s": 0.0,
            "mapping_s": 0.0,
            "output_copy_s": 0.0,
            "frame_s": 0.0,
            "scene_s": 0.0,
        }
        self._last_frame_timing: Optional[dict[str, float]] = None
        if self._baseline_activity_config is not None:
            shape = self._baseline_activity_config.shape
            if shape not in {(self._network_nodes,), (2, self._network_nodes)}:
                raise ValueError(
                    "baseline_activity must have shape "
                    f"({self._network_nodes},) or (2, {self._network_nodes}), got {shape}"
                )
        self._build_mapping_buffers()
        self._build_input_buffers()
        # Do not run a grey-screen simulation in the constructor.  The first
        # callable/reset initializes it, keeping construction inspectable and
        # allowing a launcher to fail before any forward pass.
        self.reset()

    @property
    def n_targets(self) -> int:
        return self.n_t4t5

    @property
    def column_uv(self) -> np.ndarray:
        """The unique ``(u, v)`` hex columns in stimulus input order."""

        return self._column_uv.copy()

    @property
    def column_directions(self) -> dict[str, np.ndarray]:
        """Exact left/right head-frame directions in ``input_index`` order.

        Each returned array has shape ``(n_columns, 3)`` and is a copy so a
        caller cannot mutate the fitted runtime's sampling contract.
        """

        return {side: directions.copy() for side, directions in self._column_directions.items()}

    @property
    def left_column_directions(self) -> np.ndarray:
        """Alias for the exact left-eye direction array."""

        return self._column_directions["L"].copy()

    @property
    def right_column_directions(self) -> np.ndarray:
        """Alias for the exact right-eye direction array."""

        return self._column_directions["R"].copy()

    @property
    def target_indices_array(self) -> np.ndarray:
        return self.target_indices.copy()

    @property
    def t4t5_drive(self) -> Optional[np.ndarray]:
        """Last signed MaleCNS target vector, if a frame has been processed."""
        return None if self.last_output is None else self.last_output.t4t5_drive.copy()

    def _tensor_buffer(self, value: np.ndarray, *, dtype: Any = None) -> Any:
        """Create one persistent mapping buffer on the requested device."""

        ctor = getattr(self.torch, "as_tensor", None) or getattr(self.torch, "tensor", None)
        if not callable(ctor):
            raise FlyVisRuntimeError("PyTorch module exposes neither as_tensor nor tensor")
        if dtype is None:
            dtype = getattr(self.torch, "float32", None)
        try:
            tensor = ctor(value, dtype=dtype) if dtype is not None else ctor(value)
        except TypeError:
            # A small injected/mock tensor module may not accept ``dtype``.
            tensor = ctor(value)
        move = getattr(tensor, "to", None)
        if callable(move):
            tensor = move(self.device)
        return tensor

    def _build_mapping_buffers(self) -> None:
        """Precompute device-side indices used by every frame.

        Only the compact index/reduction buffers are copied at construction.
        Per-frame activity stays on FlyVis' device until this mapping gathers
        the selected T4/T5 nodes and reduces them to the target vector.
        """

        integer_dtype = getattr(self.torch, "int64", np.int64)
        self._mapping_node_buffer = self._tensor_buffer(
            self._mapping_flat_nodes, dtype=integer_dtype
        )
        self._mapping_target_buffer = self._tensor_buffer(
            self._mapping_target_rows, dtype=integer_dtype
        )
        self._mapping_eye_buffer = self._tensor_buffer(
            self._mapping_eye_rows, dtype=integer_dtype
        )
        self._mapping_count_buffer = self._tensor_buffer(
            self._mapping_group_sizes_f32,
            dtype=getattr(self.torch, "float32", np.float32),
        )
        # Reusing the reduction destination removes one CUDA allocation and
        # one allocator bookkeeping round-trip from every frame.  Tiny test
        # tensor seams may not provide ``zeros``/``zero_``; those paths retain
        # the allocation fallback in ``_native_mapping``.
        self._mapping_sums_buffer: Any = None
        zeros = getattr(self.torch, "zeros", None)
        if callable(zeros):
            try:
                self._mapping_sums_buffer = zeros(
                    (self.n_t4t5,),
                    dtype=getattr(self.torch, "float32", np.float32),
                    device=self.device,
                )
            except (TypeError, RuntimeError):
                try:
                    self._mapping_sums_buffer = zeros(
                        (self.n_t4t5,),
                        dtype=getattr(self.torch, "float32", np.float32),
                    )
                    move = getattr(self._mapping_sums_buffer, "to", None)
                    if callable(move):
                        self._mapping_sums_buffer = move(self.device)
                except (TypeError, RuntimeError):
                    self._mapping_sums_buffer = None

        # The network's stimulus uses these indices for every frame.  Keep a
        # native copy ready for the direct-write fast path; unlike the legacy
        # ``Stimulus.add_input`` route this avoids a repeated NumPy-index
        # conversion and an intermediate ``x.to(buffer.device)`` call.
        integer_dtype = getattr(self.torch, "int64", np.int64)
        self._stimulus_input_index_buffer = self._tensor_buffer(
            self._input_index, dtype=integer_dtype
        )

    def _build_input_buffers(self) -> None:
        """Allocate one reusable binocular input tensor.

        ``torch.as_tensor`` on a NumPy array is a view on CPU.  We retain that
        view and its device copy, then update only the 721-column host buffer
        per frame.  This keeps the exact ``(2, 1, 1, columns)`` contract while
        avoiding ``np.stack`` plus a fresh device allocation on every call.
        """

        self._input_values = np.empty(
            (2, 1, 1, self.n_columns), dtype=np.float32
        )
        self._input_left_view = self._input_values[0, 0, 0]
        self._input_right_view = self._input_values[1, 0, 0]
        self._input_host_tensor: Any = None
        self._input_device_tensor: Any = None
        ctor = getattr(self.torch, "as_tensor", None) or getattr(self.torch, "tensor", None)
        if not callable(ctor):
            raise FlyVisRuntimeError("PyTorch module exposes neither as_tensor nor tensor")
        dtype = getattr(self.torch, "float32", np.float32)
        try:
            # FlyVis sets PyTorch's global default device during runtime
            # import.  Explicitly pin this NumPy-view tensor to CPU; without
            # ``device='cpu'`` torch.as_tensor would silently allocate it on
            # CUDA, detach it from ``_input_values``, and leave every frame's
            # persistent input stale.
            self._input_host_tensor = ctor(
                self._input_values, dtype=dtype, device="cpu"
            )
        except TypeError:
            try:
                self._input_host_tensor = ctor(self._input_values, dtype=dtype)
            except TypeError:
                self._input_host_tensor = ctor(self._input_values)
        self._input_device_tensor = self._input_host_tensor
        if self.device != "cpu":
            move = getattr(self._input_host_tensor, "to", None)
            if callable(move):
                try:
                    self._input_device_tensor = move(self.device)
                except (TypeError, RuntimeError):
                    self._input_device_tensor = None
        self._input_tensor_persistent = self._input_device_tensor is not None
        self._input_device_views: Optional[tuple[Any, Any]] = None
        if self._input_device_tensor is not None:
            try:
                self._input_device_views = (
                    self._input_device_tensor[0, 0, 0, :],
                    self._input_device_tensor[1, 0, 0, :],
                )
            except (TypeError, IndexError):
                self._input_device_views = None

        # Direct buffer writes are only enabled for the shipped FlyVis
        # Stimulus implementation.  Injected test doubles and third-party
        # implementations keep their public add_input semantics unchanged.
        stimulus = getattr(self.net, "stimulus", None)
        module_name = str(getattr(type(stimulus), "__module__", ""))
        self._direct_stimulus_supported = module_name.startswith("flyvis.")

    @staticmethod
    def _is_numpy_activity(value: Any) -> bool:
        return isinstance(value, np.ndarray)

    @staticmethod
    def _tensor_device_string(value: Any) -> Optional[str]:
        """Return a tensor-like object's device string, if it has one."""

        device = getattr(value, "device", None)
        if device is None:
            return None
        return str(device).strip().lower()

    @classmethod
    def _is_tensor_input(cls, value: Any) -> bool:
        """Recognize CPU/CUDA tensor inputs without importing PyTorch."""

        return cls._tensor_device_string(value) is not None and callable(
            getattr(value, "detach", None)
        )

    def _validate_tensor_input(self, value: Any, name: str) -> Any:
        """Validate a device-resident column vector without NumPy conversion."""

        shape = getattr(value, "shape", None)
        try:
            shape = tuple(int(size) for size in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric vector of shape ({self.n_columns},)") from exc
        if shape != (self.n_columns,):
            raise ValueError(
                f"{name} must have shape ({self.n_columns},), got {shape}"
            )
        device = self._tensor_device_string(value)
        if device is None:
            raise ValueError(f"{name} is not a device-resident tensor")
        if self.device.startswith("cuda:") and device.startswith("cuda:") and device != self.device:
            raise ValueError(
                f"{name} is on {device}, but FlyVis is configured for {self.device}"
            )
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        isfinite = getattr(self.torch, "isfinite", None)
        if callable(isfinite):
            try:
                finite = isfinite(value).all()
                # ``_assert_async`` schedules a device-side assertion and
                # avoids copying 721 values or synchronizing the host.  Small
                # injected tensor modules generally do not expose it, so use
                # their scalar fallback for diagnostics/tests.
                assert_async = getattr(self.torch, "_assert_async", None)
                if callable(assert_async):
                    assert_async(finite, f"{name} contains NaN or infinite values")
                else:
                    item = getattr(finite, "item", None)
                    finite_value = item() if callable(item) else finite
                    if not bool(finite_value):
                        raise ValueError(f"{name} contains NaN or infinite values")
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"could not validate {name}: {exc}") from exc
        else:
            # A minimal tensor seam without ``isfinite`` may still expose a
            # safe CPU ``numpy`` view; CUDA tensors fail closed rather than
            # silently forcing a full device-to-host copy.
            if device.startswith("cuda:"):
                raise FlyVisRuntimeError(
                    f"cannot validate CUDA {name} without torch.isfinite"
                )
            try:
                array = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} is not numeric") from exc
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinite values")
        return value

    def _validate_activity(self, activity: Any) -> Any:
        """Validate a native activity tensor without forcing a host copy."""

        shape = getattr(activity, "shape", None)
        try:
            shape = tuple(int(size) for size in shape)
        except (TypeError, ValueError) as exc:
            raise FlyVisRuntimeError("FlyVis nodes.activity has no usable shape") from exc
        if shape != (2, self._network_nodes):
            raise FlyVisRuntimeError(
                "FlyVis nodes.activity must have shape "
                f"(2, {self._network_nodes}), got {shape}"
            )
        # ``torch.isfinite(activity).all().item()`` forces a full-network
        # CUDA synchronization on every 5 ms frame.  When validation is
        # enabled (the safe default), modern PyTorch uses ``_assert_async`` so
        # the check stays on the current CUDA stream; older/mock tensor
        # modules use the scalar fallback.  Callers auditing a known-safe
        # fitted model may disable the check explicitly.
        if self.validate_activity_finite:
            if self._is_numpy_activity(activity):
                try:
                    finite = bool(np.all(np.isfinite(activity)))
                except (TypeError, ValueError) as exc:
                    raise FlyVisRuntimeError("FlyVis nodes.activity is not numeric") from exc
            else:
                isfinite = getattr(self.torch, "isfinite", None)
                if not callable(isfinite):
                    raise FlyVisRuntimeError(
                        "cannot validate non-NumPy FlyVis activity without torch.isfinite"
                    )
                try:
                    finite_value = isfinite(activity).all()
                    assert_async = getattr(self.torch, "_assert_async", None)
                    if callable(assert_async):
                        # Keep validation fail-closed without an
                        # ``.item()`` host synchronization on every frame.
                        assert_async(
                            finite_value,
                            "FlyVis nodes.activity contains NaN or infinite values",
                        )
                        finite = True
                    else:
                        item = getattr(finite_value, "item", None)
                        if callable(item):
                            finite_value = item()
                        finite = bool(finite_value)
                except Exception as exc:
                    raise FlyVisRuntimeError(
                        f"could not validate FlyVis nodes.activity: {exc}"
                    ) from exc
            if not finite:
                raise FlyVisRuntimeError("FlyVis nodes.activity contains NaN or infinite values")
        return activity

    @staticmethod
    def _copy_activity(activity: Any) -> Any:
        clone = getattr(activity, "clone", None)
        if callable(clone):
            return clone()
        if not isinstance(activity, np.ndarray):
            raise FlyVisRuntimeError(
                "cannot preserve a native FlyVis baseline without tensor.clone"
            )
        return np.asarray(activity, dtype=np.float32).copy()

    def _baseline_for_activity(self, activity: Any) -> Any:
        """Put a configured CPU baseline beside a native activity tensor."""

        baseline = self._baseline_activity
        if baseline is None:
            return None
        if self._is_numpy_activity(activity):
            if self._is_numpy_activity(baseline):
                return baseline
            cpu = getattr(baseline, "detach", None)
            if callable(cpu):
                baseline = cpu()
            cpu = getattr(baseline, "cpu", None)
            if callable(cpu):
                baseline = cpu()
            numpy_method = getattr(baseline, "numpy", None)
            if callable(numpy_method):
                baseline = numpy_method()
            try:
                baseline = np.asarray(baseline, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise FlyVisRuntimeError("configured baseline activity is not numeric") from exc
            self._baseline_activity = baseline
            return baseline

        if self._is_numpy_activity(baseline):
            dtype = getattr(activity, "dtype", getattr(self.torch, "float32", None))
            baseline = self._tensor_buffer(baseline, dtype=dtype)
            self._baseline_activity = baseline
        else:
            device = getattr(activity, "device", self.device)
            # A steady-state/native baseline is already on the network's
            # device.  Calling ``baseline.to(device)`` on every frame is
            # normally a no-op in PyTorch, but it still adds Python dispatch
            # overhead and can make a custom tensor seam allocate or sync.
            # Move only when the baseline does not advertise the activity's
            # device; objects without a device attribute retain the historic
            # best-effort move behavior.
            baseline_device = getattr(baseline, "device", None)
            same_device = (
                baseline_device is not None
                and str(baseline_device).strip().lower()
                == str(device).strip().lower()
            )
            move = getattr(baseline, "to", None)
            if callable(move) and not same_device:
                try:
                    baseline = move(device)
                except TypeError:
                    baseline = move(str(device))
                self._baseline_activity = baseline
        return baseline

    def _native_mapping(self, activity: Any, baseline: Any = None) -> Optional[np.ndarray]:
        """Gather/reduce a native tensor and transfer only target rows.

        Baseline subtraction is intentionally performed *after* the indexed
        gather.  The previous implementation formed a full ``(2, n_nodes)``
        ``activity - baseline`` tensor before selecting 13,582 values; this
        version keeps that operation on the compact target subset and reuses a
        persistent scatter destination.
        """

        if self._is_numpy_activity(activity):
            return None
        # ``new_zeros`` and in-place ``scatter_add_`` are stable across the
        # supported PyTorch releases.  A NumPy mock uses the vectorized path
        # below; unsupported non-NumPy tensors fail closed at the caller.
        sums = self._mapping_sums_buffer
        scatter_add = getattr(sums, "scatter_add_", None) if sums is not None else None
        if not callable(scatter_add):
            new_zeros = getattr(activity, "new_zeros", None)
            if callable(new_zeros):
                try:
                    float32 = getattr(self.torch, "float32", None)
                    if float32 is not None and getattr(activity, "dtype", float32) != float32:
                        try:
                            sums = new_zeros((self.n_t4t5,), dtype=float32)
                        except TypeError:
                            sums = new_zeros((self.n_t4t5,))
                    else:
                        sums = new_zeros((self.n_t4t5,))
                    scatter_add = getattr(sums, "scatter_add_", None)
                except Exception:
                    sums = None
            else:
                sums = None
        if not callable(scatter_add):
            return None
        try:
            zero = getattr(sums, "zero_", None)
            if callable(zero):
                zero()
            selected = activity[self._mapping_eye_buffer, self._mapping_node_buffer]
            if baseline is not None:
                selected = selected - baseline[
                    self._mapping_eye_buffer, self._mapping_node_buffer
                ]
            float32 = getattr(self.torch, "float32", None)
            if (float32 is not None and getattr(sums, "dtype", float32) == float32
                    and getattr(selected, "dtype", float32) != float32):
                cast = getattr(selected, "to", None)
                if callable(cast):
                    try:
                        selected = cast(dtype=float32)
                    except TypeError:
                        selected = cast(float32)
            scatter_add(0, self._mapping_target_buffer, selected)
            counts = self._mapping_count_buffer
            # Injected/mock tensors may not implement device propagation.  A
            # real FlyVis tensor always shares the requested device with these
            # constructor-created buffers.
            drive = (sums / counts) * self.output_gain
        except Exception as exc:
            raise FlyVisRuntimeError(
                f"native FlyVis target gather/reduction failed: {exc}"
            ) from exc
        transfer_start = time.perf_counter() if self.timing_enabled else 0.0
        detach = getattr(drive, "detach", None)
        if callable(detach):
            drive = detach()
        cpu = getattr(drive, "cpu", None)
        if callable(cpu):
            drive = cpu()
        numpy_method = getattr(drive, "numpy", None)
        if callable(numpy_method):
            drive = numpy_method()
        try:
            output = np.asarray(drive, dtype=np.float32).copy()
        except (TypeError, ValueError) as exc:
            raise FlyVisRuntimeError("native FlyVis target drive is not numeric") from exc
        if self.timing_enabled:
            self._timing_totals["output_copy_s"] += time.perf_counter() - transfer_start
        if output.shape != (self.n_t4t5,):
            raise FlyVisRuntimeError(
                f"native FlyVis target drive has unexpected shape {output.shape}"
            )
        return output

    def _steady_state(self) -> tuple[Any, Any]:
        method = getattr(self.net, "steady_state", None)
        if not callable(method):
            raise FlyVisRuntimeError("baseline_mode='steady_state' requires net.steady_state")
        with self._inference_context():
            try:
                state = method(
                    t_pre=self.baseline_seconds, dt=FLYVIS_DT,
                    batch_size=2, value=self.baseline_value,
                )
            except TypeError:
                state = method(self.baseline_seconds, FLYVIS_DT, 2, self.baseline_value)
        activity = _state_activity_native(state)
        return state, self._validate_activity(activity)

    def _inference_context(self):
        """Return the strongest no-gradient context available in the runtime."""

        inference_mode = getattr(self.torch, "inference_mode", None)
        if callable(inference_mode):
            return inference_mode()
        no_grad = getattr(self.torch, "no_grad", None)
        if callable(no_grad):
            return no_grad()
        return nullcontext()

    def reset(self) -> None:
        """Clear both eye states and the frame counter.

        Grey-state initialization is deferred until the next call, but it is
        one state with batch size two, never two unrelated one-eye networks.
        """

        self._state = None
        self._baseline_activity = None
        self._frame = 0
        self.last_output = None
        self._scene_baseline_manifest = None

    def preadapt_static_scene(
        self,
        left_columns: Any,
        right_columns: Any,
        *,
        seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        """Initialise FlyVis on one fixed binocular scene without emitting output.

        This is deliberately a narrow sensory-initialisation operation for a
        fitted strict run.  The adapter starts from its ordinary grey
        ``steady_state`` (so the initial condition is deterministic), feeds
        the exact same left/right column vectors for the requested duration,
        retains the final native recurrent state, and uses that final native
        activity as the signed-drive baseline.  It never calls
        :meth:`_map_activity`, never increments the exposed frame counter,
        and never creates or invokes a MaleCNS/NM/body object.

        The caller owns rendering the scene.  Passing columns rather than a
        scene label makes it impossible for this method to obtain a
        privileged stimulus direction, future state, or motor cue.
        """

        left = (
            self._validate_tensor_input(left_columns, "left_columns")
            if self._is_tensor_input(left_columns)
            else _finite_vector(left_columns, "left_columns", self.n_columns)
        )
        right = (
            self._validate_tensor_input(right_columns, "right_columns")
            if self._is_tensor_input(right_columns)
            else _finite_vector(right_columns, "right_columns", self.n_columns)
        )
        duration = self.baseline_seconds if seconds is None else seconds
        try:
            duration = float(duration)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("static-scene baseline seconds must be positive and finite") from exc
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("static-scene baseline seconds must be positive and finite")
        frame_float = duration / FLYVIS_DT
        frame_count = int(round(frame_float))
        if frame_count <= 0 or abs(duration - frame_count * FLYVIS_DT) > 1.0e-9:
            raise ValueError(
                "static-scene baseline seconds must be a positive multiple of "
                f"the FlyVis frame period {FLYVIS_DT:g}s"
            )

        # Start from a fresh, deterministic grey state.  In particular, do
        # not use a prior experiment's recurrent state as the initial scene
        # condition.  A fitted strict path fails closed if the network cannot
        # provide the normal steady-state constructor.
        self.reset()
        if self.baseline_mode != "steady_state":
            raise FlyVisRuntimeError(
                "static-scene preadaptation requires baseline_mode='steady_state'"
            )
        self._state, grey_activity = self._steady_state()
        self._baseline_activity = grey_activity
        final_activity: Any = None
        scene_started = time.perf_counter() if self.timing_enabled else 0.0
        remaining = frame_count
        while remaining:
            chunk = min(remaining, self.static_batch_frames)
            # This intentionally bypasses step_detailed: adaptation must not
            # map, transfer, or publish a T4/T5 response frame.  FlyVis'
            # recurrent forward loop still integrates each frame in order;
            # bounded chunks only remove Python/stimulus setup overhead.
            tensor = self._prepare_static_tensor(left, right, chunk)
            final_activity = self._network_sequence(tensor, chunk)
            remaining -= chunk
            self._timing_totals["scene_frames"] += chunk
        if final_activity is None:  # defensive; frame_count is validated > 0
            raise FlyVisRuntimeError("static-scene preadaptation produced no final activity")
        self._baseline_activity = self._copy_activity(final_activity)
        self._validate_activity(self._baseline_activity)
        # The recurrent state above is deliberately retained.  Only the
        # externally visible frame/output state is rewound to experiment t=0.
        self._frame = 0
        self.last_output = None
        if self.timing_enabled:
            scene_elapsed = time.perf_counter() - scene_started
            self._timing_totals["scene_s"] += scene_elapsed
        else:
            scene_elapsed = 0.0
        self._scene_baseline_manifest = {
            "enabled": True,
            "mode": "static_scene",
            "purpose": "sensory_initialization_only",
            "control_path": "none",
            "male_cns_invoked": False,
            "neuromuscular_invoked": False,
            "body_invoked": False,
            "scene_time_s": 0.0,
            "seconds": float(frame_count * FLYVIS_DT),
            "frames": int(frame_count),
            "frame_dt_s": float(FLYVIS_DT),
            "input": "fixed_exact_left_right_columns",
            "left_columns_sha256": self._input_hash(left),
            "right_columns_sha256": self._input_hash(right),
            "state_retained": True,
            "baseline_activity_shape": [int(size) for size in getattr(self._baseline_activity, "shape", ())],
            "baseline_activity_source": "final_static_scene_frame_native_activity",
            "exposed_frame_index_after_adaptation": int(self._frame),
            "last_output_after_adaptation": self.last_output,
            "timing": {
                "enabled": bool(self.timing_enabled),
                "wall_s": float(scene_elapsed),
                "frames": int(frame_count),
                "batch_frames": int(self.static_batch_frames),
                "network_calls": int((frame_count + self.static_batch_frames - 1) // self.static_batch_frames),
            },
        }
        return dict(self._scene_baseline_manifest)

    # Readable aliases for launchers that use the shorter adaptation wording.
    adapt_static_scene = preadapt_static_scene
    preadapt_scene = preadapt_static_scene

    @property
    def scene_baseline_manifest(self) -> Optional[dict[str, Any]]:
        """Copy of the last static-scene sensory initialisation record."""

        return None if self._scene_baseline_manifest is None else dict(self._scene_baseline_manifest)

    def _ensure_state(self) -> None:
        if self._state is not None:
            return
        if self.baseline_mode == "steady_state":
            self._state, self._baseline_activity = self._steady_state()
            return
        if self.baseline_mode == "fixed":
            if self._baseline_activity_config is None:
                self._baseline_activity = np.full((2, self._network_nodes), self.baseline_value, dtype=np.float32)
            else:
                base = self._baseline_activity_config
                self._baseline_activity = np.repeat(base[None, :], 2, axis=0) if base.ndim == 1 else base.copy()
        elif self.baseline_mode in {"none", "first"}:
            self._baseline_activity = None
        # ``None`` state makes FlyVis allocate its normal initial two-sample
        # batch.  The call path below validates that it really did so.

    def _prepare_frame_tensor(self, left: np.ndarray, right: np.ndarray) -> Any:
        """Update and return the reusable ``(2, 1, 1, columns)`` tensor."""

        started = time.perf_counter() if self.timing_enabled else 0.0
        if left is not self._input_left_view:
            np.copyto(self._input_left_view, left, casting="unsafe")
        if right is not self._input_right_view:
            np.copyto(self._input_right_view, right, casting="unsafe")
        tensor = self._input_device_tensor
        if tensor is None:
            ctor = getattr(self.torch, "as_tensor", None) or getattr(self.torch, "tensor", None)
            if not callable(ctor):
                raise FlyVisRuntimeError("PyTorch module exposes neither as_tensor nor tensor")
            dtype = getattr(self.torch, "float32", None)
            try:
                tensor = ctor(self._input_values, dtype=dtype) if dtype is not None else ctor(self._input_values)
            except TypeError:
                tensor = ctor(self._input_values)
            move = getattr(tensor, "to", None)
            if callable(move):
                tensor = move(self.device)
        elif tensor is not self._input_host_tensor:
            # On CUDA the host tensor is a stable NumPy view and ``copy_`` is
            # the only per-frame transfer.  It is intentionally synchronous
            # here: a pageable host array may be reused on the next frame.
            copy = getattr(tensor, "copy_", None)
            if callable(copy):
                copy(self._input_host_tensor)
            else:
                ctor = getattr(self.torch, "as_tensor", None) or getattr(self.torch, "tensor", None)
                if not callable(ctor):
                    raise FlyVisRuntimeError("PyTorch module exposes neither as_tensor nor tensor")
                dtype = getattr(self.torch, "float32", None)
                try:
                    tensor = ctor(self._input_values, dtype=dtype) if dtype is not None else ctor(self._input_values)
                except TypeError:
                    tensor = ctor(self._input_values)
                move = getattr(tensor, "to", None)
                if callable(move):
                    tensor = move(self.device)
        if self.timing_enabled:
            self._timing_totals["input_prepare_s"] += time.perf_counter() - started
        return tensor

    def _prepare_device_frame_tensor(self, left: Any, right: Any) -> Any:
        """Copy CPU/CUDA vectors directly into the persistent device tensor.

        A Warp renderer can provide each eye as a CUDA tensor.  In that case
        only the 721 values are copied into the adapter's stable input tensor;
        no intermediate NumPy array or host round-trip is created.  CPU and
        NumPy vectors remain accepted for backwards compatibility, including
        mixed CPU/CUDA binocular calls.
        """

        started = time.perf_counter() if self.timing_enabled else 0.0
        tensor = self._input_device_tensor
        if tensor is None:
            # This only applies to a minimal injected tensor seam.  Real
            # PyTorch always provides the persistent constructor above.
            return self._prepare_frame_tensor(
                np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
            )

        target_shape = getattr(tensor, "shape", ())
        if tuple(int(size) for size in target_shape) != (2, 1, 1, self.n_columns):
            raise FlyVisRuntimeError(
                "persistent FlyVis input tensor has an incompatible shape "
                f"{target_shape}"
            )
        # Keep the host view current for any CPU/NumPy side.  For CUDA sides,
        # ``copy_`` below reads directly from the caller's device tensor.
        sources = (left, right)
        for side, source in enumerate(sources):
            source_device = self._tensor_device_string(source)
            if self._input_device_views is not None:
                destination = self._input_device_views[side]
            else:
                destination = tensor[side, 0, 0, :]
            if source_device is None:
                try:
                    array = np.asarray(source, dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"eye {'left' if side == 0 else 'right'} input is not numeric"
                    ) from exc
                np.copyto(
                    self._input_left_view if side == 0 else self._input_right_view,
                    array,
                    casting="unsafe",
                )
                source = self._input_host_tensor[side, 0, 0, :]
            copy = getattr(destination, "copy_", None)
            if not callable(copy):
                # CPU NumPy-backed mocks may expose the tensor as an ndarray;
                # assignment preserves their existing seam behavior.
                try:
                    destination[...] = np.asarray(source)
                except Exception as exc:
                    raise FlyVisRuntimeError(
                        "persistent FlyVis input tensor cannot accept eye data"
                    ) from exc
            else:
                try:
                    copy(source)
                except TypeError:
                    copy(source, non_blocking=False)
        if self.timing_enabled:
            self._timing_totals["input_prepare_s"] += time.perf_counter() - started
        return tensor

    def _prepare_static_tensor(self, left: Any, right: Any, n_frames: int) -> Any:
        """Build one bounded repeated-scene input tensor for preadaptation."""

        if n_frames == 1:
            if self._is_tensor_input(left) or self._is_tensor_input(right):
                return self._prepare_device_frame_tensor(left, right)
            return self._prepare_frame_tensor(left, right)
        started = time.perf_counter() if self.timing_enabled else 0.0
        if self._is_tensor_input(left) or self._is_tensor_input(right):
            # Keep a CUDA-rendered static scene on CUDA while constructing a
            # bounded temporal chunk.  ``expand`` is a view; only the final
            # concatenated sequence is materialized on the device.
            cat = getattr(self.torch, "cat", None)
            reshape = getattr(left, "reshape", None)
            if not callable(cat) or not callable(reshape):
                raise FlyVisRuntimeError(
                    "device-resident static-scene tensors require torch.cat/reshape"
                )
            dtype = getattr(self.torch, "float32", None)
            sources = []
            for value in (left, right):
                if self._is_tensor_input(value):
                    source = value
                else:
                    ctor = getattr(self.torch, "as_tensor", None) or getattr(self.torch, "tensor", None)
                    if not callable(ctor):
                        raise FlyVisRuntimeError(
                            "PyTorch module exposes neither as_tensor nor tensor"
                        )
                    try:
                        source = ctor(value, dtype=dtype) if dtype is not None else ctor(value)
                    except TypeError:
                        source = ctor(value)
                    move = getattr(source, "to", None)
                    if callable(move):
                        source = move(self.device)
                if dtype is not None and getattr(source, "dtype", dtype) != dtype:
                    cast = getattr(source, "to", None)
                    if callable(cast):
                        try:
                            source = cast(dtype=dtype)
                        except TypeError:
                            source = cast(dtype)
                shape_view = source.reshape(1, 1, 1, self.n_columns)
                expand = getattr(shape_view, "expand", None)
                if not callable(expand):
                    raise FlyVisRuntimeError(
                        "device-resident static-scene tensors require Tensor.expand"
                    )
                sources.append(expand(1, n_frames, 1, self.n_columns))
            try:
                tensor = cat(tuple(sources), dim=0)
            except TypeError:
                tensor = cat(tuple(sources), 0)
            if self.timing_enabled:
                self._timing_totals["input_prepare_s"] += time.perf_counter() - started
            return tensor
        values = np.empty(
            (2, n_frames, 1, self.n_columns), dtype=np.float32
        )
        values[0, :, 0, :] = left
        values[1, :, 0, :] = right
        ctor = getattr(self.torch, "as_tensor", None) or getattr(self.torch, "tensor", None)
        if not callable(ctor):
            raise FlyVisRuntimeError("PyTorch module exposes neither as_tensor nor tensor")
        dtype = getattr(self.torch, "float32", None)
        try:
            tensor = ctor(values, dtype=dtype) if dtype is not None else ctor(values)
        except TypeError:
            tensor = ctor(values)
        move = getattr(tensor, "to", None)
        if callable(move):
            tensor = move(self.device)
        if self.timing_enabled:
            self._timing_totals["input_prepare_s"] += time.perf_counter() - started
        return tensor

    def _input_hash(self, value: Any) -> str:
        """Hash one scene vector for provenance, copying only adaptation inputs."""

        if self._is_tensor_input(value):
            detached = value.detach() if callable(getattr(value, "detach", None)) else value
            cpu = getattr(detached, "cpu", None)
            if callable(cpu):
                detached = cpu()
            numpy_method = getattr(detached, "numpy", None)
            if callable(numpy_method):
                detached = numpy_method()
            try:
                detached = np.asarray(detached, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise FlyVisRuntimeError("cannot hash device-resident scene input") from exc
            return _array_sha256(detached)
        return _array_sha256(value)

    def _write_stimulus(self, tensor: Any, n_frames: int) -> None:
        """Write input columns into the stimulus, using direct writes when safe."""

        started = time.perf_counter() if self.timing_enabled else 0.0
        stimulus = self.net.stimulus
        zero = getattr(stimulus, "zero", None)
        add_input = getattr(stimulus, "add_input", None)
        if not callable(zero) or not callable(add_input):
            raise FlyVisRuntimeError("FlyVis stimulus lacks zero/add_input")

        direct = False
        if self._direct_stimulus_supported:
            buffer = getattr(stimulus, "buffer", None)
            shape = getattr(buffer, "shape", ())
            try:
                shape = tuple(int(size) for size in shape)
            except (TypeError, ValueError):
                shape = ()
            if len(shape) == 3 and shape[:2] == (2, n_frames):
                try:
                    buffer.zero_()
                    buffer[:, :, self._stimulus_input_index_buffer] = tensor
                    if hasattr(stimulus, "_nonzero"):
                        stimulus._nonzero = True
                    direct = True
                except (TypeError, RuntimeError, IndexError):
                    # A custom tensor/index implementation can reject the
                    # direct assignment.  The public path below remains the
                    # compatibility fallback and receives the same tensor.
                    direct = False
        if not direct:
            # The public FlyVis Stimulus API guarantees that add_input resets
            # a nonzero buffer.  Keep the explicit zero for injected/custom
            # implementations whose add_input may only write a provided
            # buffer, preserving the historical adapter contract.
            zero(2, n_frames)
            add_input(tensor)
        if self.timing_enabled:
            self._timing_totals["stimulus_s"] += time.perf_counter() - started

    def _network_sequence(self, tensor: Any, n_frames: int) -> Any:
        """Run one recurrent network call over one or more temporal frames."""

        call = self.net
        with self._inference_context():
            # FlyVis' Stimulus allocates an inference tensor during the
            # steady-state prelude when inference_mode is available.  Keep
            # the subsequent zero/direct-write inside the same context: an
            # in-place update to an inference tensor outside that context is
            # rejected by modern PyTorch.
            self._write_stimulus(tensor, n_frames)
            started = time.perf_counter() if self.timing_enabled else 0.0
            try:
                states = call(self.net.stimulus(), dt=FLYVIS_DT, state=self._state, as_states=True)
            except TypeError:
                states = call(self.net.stimulus(), FLYVIS_DT, self._state, True)
        if self.timing_enabled:
            self._timing_totals["network_s"] += time.perf_counter() - started
        if isinstance(states, (list, tuple)):
            if not states:
                raise FlyVisRuntimeError("FlyVis returned an empty state sequence")
            state = states[-1]
        else:
            state = states
        activity = self._validate_activity(_state_activity_native(state))
        # Keep the actual state object (on its native device) for the next
        # frame; only mapped target activity is moved to NumPy below.
        self._state = state
        return activity

    def _network_step(self, left: np.ndarray, right: np.ndarray) -> Any:
        """Run one 5 ms frame with a persistent direct-column input tensor."""

        tensor = self._prepare_frame_tensor(left, right)
        return self._network_sequence(tensor, 1)

    def _map_activity(self, activity: Any, *, _already_validated: bool = False) -> np.ndarray:
        """Map two-eye activity with one native gather/reduction operation.

        The returned vector is necessarily a host array because the
        MaleCNS/``OptionalFlyVisBackend`` boundary is NumPy.  The source
        activity is not: CUDA activity is indexed and reduced on CUDA first,
        so only the 13,582 target values cross the device boundary.
        """

        # ``_network_sequence`` validates the activity before returning it.
        # The public/direct helper remains fail-closed by default, while the
        # internal step path can avoid a second full-network finite scan.
        if not _already_validated:
            activity = self._validate_activity(activity)
        baseline = self._baseline_for_activity(activity)
        # Keep baseline subtraction on the mapped target subset.  This avoids
        # allocating a full network-sized delta tensor for every CUDA frame.
        drive = self._native_mapping(activity, baseline)
        if drive is None:
            if not self._is_numpy_activity(activity):
                # Never recover from an unsupported native tensor by calling
                # ``np.asarray``: that would silently reintroduce the full
                # activity transfer this adapter is designed to avoid.
                raise FlyVisRuntimeError(
                    "native FlyVis activity cannot be gathered with the configured tensor API"
                )
            try:
                activity_array = np.asarray(activity, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise FlyVisRuntimeError("FlyVis activity cannot be mapped on CPU") from exc
            selected = activity_array[self._mapping_eye_rows, self._mapping_flat_nodes]
            if baseline is not None:
                baseline_array = np.asarray(baseline, dtype=np.float32)
                selected = selected - baseline_array[
                    self._mapping_eye_rows, self._mapping_flat_nodes
                ]
            sums = np.zeros(self.n_t4t5, dtype=np.float32)
            # ``np.add.at`` performs the complete reduction in C-level NumPy
            # code and has no Python loop over the target rows.
            np.add.at(sums, self._mapping_target_rows, selected)
            drive = (sums / self._mapping_group_sizes_f32) * self.output_gain
            drive = np.asarray(drive, dtype=np.float32)

        if self.baseline_mode == "first" and self._baseline_activity is None:
            # Explicit first-frame baseline: startup is silent, subsequent
            # values retain both positive and negative deviations.
            self._baseline_activity = self._copy_activity(activity)
            drive.fill(0.0)
        if not np.all(np.isfinite(drive)):
            raise FlyVisRuntimeError("mapped FlyVis target drive contains NaN or infinite values")
        return drive

    def step_detailed(self, left_columns: Any, right_columns: Any) -> FlyVisDriveOutput:
        """Run one 5 ms two-eye frame and return drive plus target metadata."""

        frame_started = time.perf_counter() if self.timing_enabled else 0.0
        left_tensor = (
            self._validate_tensor_input(left_columns, "left_columns")
            if self._is_tensor_input(left_columns) else None
        )
        right_tensor = (
            self._validate_tensor_input(right_columns, "right_columns")
            if self._is_tensor_input(right_columns) else None
        )
        if left_tensor is None and right_tensor is None:
            left = _finite_vector(
                left_columns, "left_columns", self.n_columns,
                destination=self._input_left_view,
            )
            right = _finite_vector(
                right_columns, "right_columns", self.n_columns,
                destination=self._input_right_view,
            )
        else:
            # Validate the non-tensor eye through the existing finite vector
            # path, then let the device-aware copier combine both sides with
            # no NumPy conversion of the CUDA eye.
            left = left_tensor if left_tensor is not None else _finite_vector(
                left_columns, "left_columns", self.n_columns
            )
            right = right_tensor if right_tensor is not None else _finite_vector(
                right_columns, "right_columns", self.n_columns
            )
        self._ensure_state()
        if left_tensor is None and right_tensor is None:
            activity = self._network_step(left, right)
        else:
            tensor = self._prepare_device_frame_tensor(left, right)
            activity = self._network_sequence(tensor, 1)
        mapping_started = time.perf_counter() if self.timing_enabled else 0.0
        # ``_network_sequence`` has already validated this native activity;
        # avoid repeating the full ``(2, n_nodes)`` finite check here.
        drive = self._map_activity(activity, _already_validated=True)
        if self.timing_enabled:
            self._timing_totals["mapping_s"] += time.perf_counter() - mapping_started
        output = FlyVisDriveOutput(
            # ``_map_activity`` returns a new host array for this frame.  The
            # mapping arrays are immutable for the adapter lifetime and are
            # exposed read-only, so none of these three payloads needs a
            # fresh defensive copy on every 5 ms step.
            drive=drive, target_indices=self._output_target_indices,
            target_ids=self._output_target_ids, units=self.units,
            frame_index=self._frame, baseline=self.baseline_mode,
        )
        self.last_output = output
        self._frame += 1
        if self.timing_enabled:
            elapsed = time.perf_counter() - frame_started
            self._timing_totals["frames"] += 1
            self._timing_totals["frame_s"] += elapsed
            self._last_frame_timing = {
                "frame_s": float(elapsed),
                "mapping_s": float(self._timing_totals["mapping_s"]),
            }
        else:
            self._timing_totals["frames"] += 1
        return output

    def __call__(self, left_columns: Any, right_columns: Any) -> np.ndarray:
        """Return the signed target vector expected by OptionalFlyVisBackend."""

        return self.step_detailed(left_columns, right_columns).drive

    step = __call__

    def _timing_manifest(self) -> dict[str, Any]:
        """Return auditable host-stage timings without tensor payloads."""

        totals = dict(self._timing_totals)
        frames = int(totals["frames"])
        scene_frames = int(totals["scene_frames"])
        totals.update({
            "enabled": bool(self.timing_enabled),
            "scope": "adapter_lifetime",
            "mean_frame_s": float(totals["frame_s"] / frames) if frames else 0.0,
            "mean_network_s": float(totals["network_s"] / frames) if frames else 0.0,
            "mean_mapping_s": float(totals["mapping_s"] / frames) if frames else 0.0,
            "scene_mean_frame_s": float(totals["scene_s"] / scene_frames) if scene_frames else 0.0,
            "last_frame": None if self._last_frame_timing is None else dict(self._last_frame_timing),
        })
        return totals

    def manifest(self) -> dict[str, Any]:
        """Return the effective fitted-vision contract without array payloads."""

        group_sizes = self._mapping_group_sizes
        mapping_method = "precomputed_gather_scatter_mean"
        return {
            "backend": self.name,
            "device": self.device,
            "max_vram_gib": self.max_vram_gib,
            "model_root": str(self.model_root),
            "view": self.view,
            "target_path": str(self.target_path),
            "csr_path": str(self.csr_path),
            "n_columns_per_eye": self.n_columns,
            "n_targets": self.n_t4t5,
            "units": self.units,
            "baseline_mode": self.baseline_mode,
            "scene_baseline": self.scene_baseline_manifest,
            "output_gain": self.output_gain,
            "flyvis_hz": float(FLYVIS_HZ),
            "flyvis_dt_s": float(FLYVIS_DT),
            "frame_period_s": float(FLYVIS_DT),
            "column_order": {
                "source": "net.stimulus.input_index",
                "index_axis": 1,
                "input_index_shape": [int(size) for size in self._input_index.shape],
                "input_index_dtype": str(self._input_index.dtype),
                "input_index_sha256": _array_sha256(self._input_index),
                "input_cell_types": None if self._input_cell_types is None else list(self._input_cell_types),
                "uv_shape": [int(size) for size in self._column_uv.shape],
                "uv_dtype": str(self._column_uv.dtype),
                "uv_sha256": _array_sha256(self._column_uv),
                "unique_columns": int(np.unique(self._column_uv, axis=0).shape[0]),
                "row_alignment": "all stimulus.input_index rows resolve to the same ordered (u,v) columns",
            },
            "column_directions": {
                "generator": "scripts/flyvis_runtime_adapter_eye_parity_v2.py:_hex_to_dirs",
                "frame": {"x": "forward", "y": "left", "z": "up"},
                "order": "column_order.input_index axis 1",
                "L": {
                    "shape": [int(size) for size in self._column_directions["L"].shape],
                    "dtype": str(self._column_directions["L"].dtype),
                    "sha256": _array_sha256(self._column_directions["L"]),
                },
                "R": {
                    "shape": [int(size) for size in self._column_directions["R"].shape],
                    "dtype": str(self._column_directions["R"].dtype),
                    "sha256": _array_sha256(self._column_directions["R"]),
                },
            },
            "timing": {
                "flyvis_hz": float(FLYVIS_HZ),
                "flyvis_dt_s": float(FLYVIS_DT),
                "baseline_seconds": float(self.baseline_seconds),
                **self._timing_manifest(),
            },
            "optimization": {
                "inference_mode": callable(getattr(self.torch, "inference_mode", None)),
                "persistent_input_tensor": bool(self._input_tensor_persistent),
                "device_input_path": "native_tensor_copy_into_persistent_tensor",
                "device_input_stream": "current_torch_stream_no_device_sync",
                "cpu_input_compatibility": True,
                "direct_stimulus_buffer_write": bool(self._direct_stimulus_supported),
                "persistent_mapping_sums": self._mapping_sums_buffer is not None,
                "mapped_baseline_subtraction": "target_gather_only",
                "activity_finite_validation": bool(self.validate_activity_finite),
                "static_batch_frames": int(self.static_batch_frames),
                "bounded_static_vram": True,
            },
            "target_index_space": "dense_neuron_index",
            "target_id_space": "canonical_neuron_id",
            "mapping_method": mapping_method,
            "mapping": {
                "method": mapping_method,
                "target_index_space": "dense_neuron_index",
                "target_id_space": "canonical_neuron_id",
                "side_authority": "MaleCNS annotations.npz:sideIdx -> tables.json:sides",
                "subtype_authority": "MaleCNS annotations.npz:typeIdx -> tables.json:types",
                "n_targets": int(self.n_t4t5),
                "n_gather_nodes": int(self._mapping_flat_nodes.size),
                "group_size_min": int(np.min(group_sizes)),
                "group_size_max": int(np.max(group_sizes)),
                "preserves_sign": True,
                "units": self.units,
            },
            "eye_coordinate_policy": {
                "name": "native_uv_azimuth_parity_v1",
                "azimuth_parity_by_registered_side": dict(self.eye_azimuth_parity),
                "parity_applied_to": ["ray_sampling", "nearest_spatial_target_column_mapping"],
                "reflection_when_parity_is_negative": "M(u,v)=(u+v,-v)",
                "mapping_assignment_hashes": {
                    "gathered_node_indices_sha256": _array_sha256(self._mapping_flat_nodes),
                    "gather_group_sizes_sha256": _array_sha256(self._mapping_group_sizes),
                    "eye_batch_rows_sha256": _array_sha256(self._mapping_eye_rows),
                },
                "target_direction_vectors_changed": False,
                "male_cns_registered_side_annotations_changed": False,
            },
        }


# Naming aliases used by small launchers and integrations.
RealFlyVisAdapter = FlyVisRuntimeAdapter
FlyVisAdapter = FlyVisRuntimeAdapter


def build_flyvis_response(**kwargs: Any) -> FlyVisRuntimeAdapter:
    """Construct the real callable response; errors intentionally propagate."""

    return FlyVisRuntimeAdapter(**kwargs)


def create_flyvis_adapter(**kwargs: Any) -> FlyVisRuntimeAdapter:
    return FlyVisRuntimeAdapter(**kwargs)


def build_optional_flyvis_backend(**kwargs: Any) -> Any:
    """Return a configured ``stereo_flyvis.OptionalFlyVisBackend``.

    The import is local for the same reason as the FlyVis import above: merely
    importing this adapter module must remain safe on CPU-only hosts.  The
    response callable is retained on ``backend.response_fn`` and its mapping
    metadata is copied onto the backend for callers configuring MaleCNS.
    """

    from stereo_flyvis import OptionalFlyVisBackend

    runtime = FlyVisRuntimeAdapter(**kwargs)
    backend = OptionalFlyVisBackend(runtime, runtime.n_columns, runtime.n_t4t5)
    backend.target_indices = runtime.target_indices.copy()
    backend.target_ids = runtime.target_ids.copy()
    backend.units = runtime.units
    backend.runtime_adapter = runtime
    return backend


__all__ = [
    "DEFAULT_COLUMNS", "DEFAULT_CSR", "DEFAULT_ENV_ROOT", "DEFAULT_MAX_VRAM_GIB", "DEFAULT_MODEL_ROOT",
    "DEFAULT_TARGETS", "DEFAULT_VIEW", "FLYVIS_DT", "FLYVIS_HZ", "T4T5_UNITS",
    "FlyVisAdapter", "FlyVisAssetError", "FlyVisDriveOutput", "FlyVisMappingError",
    "FlyVisRuntimeAdapter", "FlyVisRuntimeError", "FlyVisTargetMapping",
    "FlyVisUnavailableError", "RealFlyVisAdapter", "build_flyvis_response",
    "build_optional_flyvis_backend", "create_flyvis_adapter", "flyvis_preflight",
    "load_t4t5_targets", "preflight", "preflight_flyvis",
]
