#!/usr/bin/env python3
"""Bounded validation and benchmark harness for the optional Warp renderer.

The analytic :class:`virtual_world.VirtualWorld` renderer is the oracle.  The
Warp implementation is deliberately loaded only when :func:`run_validation`
or the command line entry point is invoked.  Consequently this module is safe
to import on a CPU-only machine, in a documentation build, or before Warp is
installed.

The harness has three jobs which are intentionally kept together:

* render both eyes at exact head-frame directions and compare them with the
  scalar ``VirtualWorld`` reference at static and (when a stimulus is
  selected) dynamic timestamps;
* record output contract checks, determinism, requested/effective backend
  provenance, and bounded cold/JIT versus warm/steady timing; and
* record process RSS and any already-visible CUDA allocator measurements,
  applying caller-provided RAM/VRAM limits.

No artifact is written implicitly.  The CLI writes JSON only when ``--artifact``
is supplied, and the public :func:`write_artifact` function is the sole file
writer in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

try:  # NumPy is optional at import time, just like Warp.
    import numpy as _np
except Exception as _exc:  # pragma: no cover - exercised in minimal images.
    _np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR: Exception | None = _exc
else:
    _NUMPY_IMPORT_ERROR = None


DEFAULT_BACKEND_MODULE = "world_renderer_opt"
DEFAULT_REQUESTED_BACKEND = "warp_cuda"
DEFAULT_ABS_TOL = 1.0e-5
DEFAULT_REL_TOL = 1.0e-4
DEFAULT_DETERMINISM_TOL = 1.0e-6
DEFAULT_RANGE_TOL = 1.0e-6
DEFAULT_RAM_LIMIT_GB = 5.0
DEFAULT_VRAM_LIMIT_GB = 15.5
DEFAULT_MEAN_ABS_TOL = 1.0e-6

# These are safety bounds, not performance targets.  A validation invocation
# should never accidentally become an all-night benchmark because a malformed
# config supplied a huge count.
MAX_DIRECTIONS = 100_000
MAX_TIMESTAMPS = 256
MAX_WARMUP = 1_000
MAX_ITERATIONS = 1_000


class WarpValidationError(RuntimeError):
    """Base class for explicit validation failures."""


class BackendUnavailableError(WarpValidationError):
    """Raised when the requested optional backend cannot be constructed."""


class ReferenceUnavailableError(WarpValidationError):
    """Raised when the analytic reference cannot be loaded."""


# ``VirtualWorld._resolve_stimuli(None, None)`` intentionally means "all
# configured stimuli".  That is useful for normal experiments but is not a
# static renderer baseline: overlapping configured boards can produce a BVH
# face tie.  The validator therefore carries explicit sentinels all the way to
# the renderer seam.
_NO_STIMULI = object()
_ALL_CONFIGURED_STIMULI = object()


@dataclass(frozen=True)
class DirectionSet:
    """Bilateral exact-direction input.

    ``common`` is used for both eyes.  ``by_eye`` allows a runtime or JSON
    artifact to provide explicitly different left/right columns while still
    making the bilateral nature of the validation visible in the report.
    """

    common: Any = None
    by_eye: Mapping[str, Any] | None = None
    source: str = "caller"

    def for_eye(self, eye: str) -> Any:
        if self.by_eye is not None and eye in self.by_eye:
            return self.by_eye[eye]
        if self.common is None:
            raise WarpValidationError(f"no exact directions supplied for {eye} eye")
        return self.common

    @property
    def eyes(self) -> tuple[str, str]:
        return ("left", "right")


@dataclass(frozen=True)
class ValidationLimits:
    """Resource limits in bytes."""

    ram_bytes: int | None = int(DEFAULT_RAM_LIMIT_GB * 1024**3)
    vram_bytes: int | None = int(DEFAULT_VRAM_LIMIT_GB * 1024**3)


def _require_numpy() -> Any:
    if _np is None:
        detail = f": {_NUMPY_IMPORT_ERROR}" if _NUMPY_IMPORT_ERROR is not None else ""
        raise WarpValidationError("Warp validation requires NumPy for array comparisons" + detail)
    return _np


def _json_safe(value: Any) -> Any:
    """Convert runtime values to strict-JSON-compatible Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if _np is not None:
        if isinstance(value, _np.generic):
            return _json_safe(value.item())
        if isinstance(value, _np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256_bytes(payload)


def _canonical_lines_hash(values: Iterable[Any]) -> str:
    chunks = []
    for value in values:
        chunks.append(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False))
        chunks.append("\n")
    return _sha256_bytes("".join(chunks).encode("utf-8"))


def _direction_metadata(direction_set: DirectionSet, directions_path: str | Path | None = None) -> dict[str, Any]:
    """Return stable shape/left/right hashes without writing a direction file."""
    np = _require_numpy()
    left = np.asarray(direction_set.for_eye("left"), dtype=np.float32)
    right = np.asarray(direction_set.for_eye("right"), dtype=np.float32)
    left_rows = left.tolist()
    right_rows = right.tolist()
    steady_combined = {**steady_record, **steady}
    return {
        "source": direction_set.source,
        "file": None if directions_path is None else str(Path(directions_path).expanduser()),
        "shape": {"left": list(left.shape), "right": list(right.shape)},
        "shape_hash": _canonical_lines_hash([list(left.shape), list(right.shape)]),
        "left_direction_hash": _canonical_lines_hash(left_rows),
        "right_direction_hash": _canonical_lines_hash(right_rows),
        "direction_hash": _canonical_lines_hash(left_rows + right_rows),
        "algorithm": "sha256-canonical-json-lines-v1",
    }


def _config_declaration(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        path = str(Path(config).expanduser().resolve())
        return {"path": path, "sha256": _sha256_file(path), "kind": "virtual_world_config"}
    return {"path": None, "sha256": _canonical_hash(config), "kind": "inline_config"}


def _scene_declaration(renderer: Any, config_declaration: Mapping[str, Any]) -> dict[str, Any]:
    for owner in (renderer,):
        for name in ("scene_path", "scene_file", "scene_source"):
            value = getattr(owner, name, None)
            if value is None or callable(value):
                continue
            path = Path(str(value)).expanduser()
            digest = _sha256_file(path)
            if digest is not None:
                return {"path": str(path.resolve()), "sha256": digest, "kind": "renderer_scene"}
    # VirtualWorld has no separate mesh file; its immutable config is the
    # auditable scene source.  This declaration is explicit rather than an
    # invented scene hash.
    return {
        "path": config_declaration.get("path"),
        "sha256": config_declaration.get("sha256"),
        "kind": "virtual_world_config_scene",
    }


def _runtime_gpu_metadata(module: ModuleType, renderer: Any) -> dict[str, Any]:
    """Collect Warp version/device identity without importing Warp eagerly."""
    version = None
    for owner in (renderer, module, sys.modules.get("warp")):
        if owner is None:
            continue
        for name in ("warp_version", "version", "__version__"):
            value = getattr(owner, name, None)
            if value is not None and not callable(value):
                version = str(value)
                break
        if version:
            break
    device = getattr(renderer, "device_name", None) or getattr(renderer, "device", None)
    device_text = None if device is None else str(device)
    gpu_name = None
    arch = None
    compute = None
    torch = sys.modules.get("torch")
    try:
        if torch is not None and bool(torch.cuda.is_available()):
            index = 0
            if device_text and ":" in device_text:
                index = int(device_text.rsplit(":", 1)[1])
            properties = torch.cuda.get_device_properties(index)
            gpu_name = str(getattr(properties, "name", "")) or None
            major = getattr(properties, "major", None)
            minor = getattr(properties, "minor", None)
            if major is not None and minor is not None:
                compute = f"{int(major)}.{int(minor)}"
                arch = f"sm_{int(major)}{int(minor)}"
    except Exception:
        pass
    return {
        "version": version,
        "device": device_text,
        "name": gpu_name,
        "arch": arch,
        "compute_capability": compute,
        "status": "PASS" if version and gpu_name and arch and compute else "UNKNOWN",
    }


def _bounded_count(value: int, *, name: str, maximum: int, minimum: int = 0) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if count < minimum or count > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {count}")
    return count


def _direction_array(value: Any, *, source: str = "directions") -> Any:
    """Validate and normalize one ``(N, 3)`` exact-direction array."""
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{source} must have shape (N, 3), got {tuple(array.shape)}")
    _bounded_count(int(array.shape[0]), name=f"{source} count", maximum=MAX_DIRECTIONS, minimum=1)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{source} must be numeric, got dtype {array.dtype}")
    array = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{source} contains non-finite values")
    lengths = np.linalg.norm(array, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError(f"{source} contains a zero-length direction")
    return np.ascontiguousarray(array)


def default_directions(count: int = 721) -> Any:
    """Return the deterministic exact-direction probe used by the CLI."""
    count = _bounded_count(count, name="direction count", maximum=MAX_DIRECTIONS, minimum=1)
    # Keep this formula aligned with WarpWorldRenderer.benchmark.  Directions
    # are intentionally not normalized here; both renderers normalize rays as
    # part of their public exact-direction seam.
    rows = [
        (1.0, 0.35 * math.sin(index * 0.017), 0.22 * math.cos(index * 0.013))
        for index in range(count)
    ]
    return _direction_array(rows, source="default directions")


def _extract_direction_payload(payload: Any, *, source: str = "caller") -> DirectionSet:
    if not isinstance(payload, Mapping):
        # A compact ``(2, N, 3)`` archive is a common way to preserve
        # bilateral columns without JSON object overhead.  Axis zero is
        # defined as left/right in that form.
        try:
            candidate = _require_numpy().asarray(payload)
            if candidate.ndim == 3 and candidate.shape[0] == 2:
                return DirectionSet(
                    by_eye={
                        "left": _direction_array(candidate[0], source="left directions"),
                        "right": _direction_array(candidate[1], source="right directions"),
                    },
                    source=source,
                )
        except (TypeError, ValueError):
            pass
        return DirectionSet(common=_direction_array(payload), source=source)
    if isinstance(payload, Mapping):
        # Prefer an explicit bilateral payload.  ``left``/``right`` are
        # accepted even if only one side is present; the missing side falls
        # back to ``directions`` when provided.
        by_eye: dict[str, Any] = {}
        for eye, aliases in (
            ("left", ("left", "left_eye", "left_directions", "directions_left")),
            ("right", ("right", "right_eye", "right_directions", "directions_right")),
        ):
            for key in aliases:
                if key in payload:
                    value = payload[key]
                    if isinstance(value, Mapping):
                        value = value.get("directions", value.get("head_directions", value.get("columns")))
                    by_eye[eye] = _direction_array(value, source=f"{eye} directions")
                    break
        common_value = None
        for key in ("directions", "head_directions", "exact_directions", "columns", "direction", "head_frame_directions"):
            if key in payload:
                common_value = _direction_array(payload[key], source=key)
                break
        if by_eye:
            return DirectionSet(common=common_value, by_eye=by_eye, source=source)
        if common_value is not None:
            return DirectionSet(common=common_value, source=source)
        raise ValueError("direction JSON object must contain directions or left/right arrays")
    return DirectionSet(common=_direction_array(payload), source=source)


def load_directions(path: str | Path) -> DirectionSet:
    """Load exact directions from ``.npy``, ``.npz``, JSON, or JSONL.

    JSON accepts a plain ``[[x, y, z], ...]`` array or an object containing
    ``directions``/``head_directions`` and optional ``left``/``right`` arrays.
    NPZ accepts the same keys; a single unnamed array uses the first stored
    member.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"direction file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        np = _require_numpy()
        loaded = np.load(path, allow_pickle=False)
        if suffix == ".npy":
            return _extract_direction_payload(loaded, source="file")
        try:
            keys = list(loaded.files)
            if not keys:
                raise ValueError(f"direction archive is empty: {path}")
            payload: dict[str, Any] = {key: loaded[key] for key in keys}
        finally:
            loaded.close()
        return _extract_direction_payload(payload, source="file")
    try:
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # A JSONL file is convenient for runtime probes and remains
            # unambiguous as long as every non-empty line is an array/object.
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            payload = rows
    except OSError as exc:
        raise WarpValidationError(f"could not read direction file {path}: {exc}") from exc
    return _extract_direction_payload(payload, source="file")


def _runtime_direction_payload(renderer: Any, world: Any) -> Any:
    """Find exact directions advertised by the runtime, without requiring it."""
    for owner in (renderer, world):
        if owner is None:
            continue
        for name in (
            "exact_directions", "head_directions", "eye_directions", "directions",
            "get_exact_directions", "get_head_directions", "get_directions",
        ):
            if not hasattr(owner, name):
                continue
            value = getattr(owner, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if value is not None:
                return value
    return None


def resolve_directions(path: str | Path | None = None, *, renderer: Any = None,
                       world: Any = None, count: int = 721) -> DirectionSet:
    """Prefer runtime-advertised columns, then a caller file, then defaults.

    A caller-supplied file always wins over runtime metadata: it is the
    explicit reproducibility boundary for comparing an archived stimulus.
    """
    if path is not None:
        return load_directions(path)
    payload = _runtime_direction_payload(renderer, world)
    if payload is not None:
        return _extract_direction_payload(payload, source="runtime")
    return DirectionSet(common=default_directions(count), source="default")


def _as_numpy_output(value: Any) -> tuple[Any, str]:
    """Materialize a renderer result and preserve a useful dtype label."""
    np = _require_numpy()
    original_dtype: str | None = None
    # Torch tensors and Warp arrays deliberately remain optional.  Duck typing
    # avoids importing either package just to inspect a result.
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
        if hasattr(value, "dtype"):
            original_dtype = str(value.dtype)
        if hasattr(value, "cpu") and callable(value.cpu):
            value = value.cpu()
        if hasattr(value, "numpy") and callable(value.numpy):
            value = value.numpy()
    elif hasattr(value, "numpy") and callable(value.numpy) and not isinstance(value, np.ndarray):
        if hasattr(value, "dtype"):
            original_dtype = str(value.dtype)
        value = value.numpy()
    elif hasattr(value, "to_numpy") and callable(value.to_numpy):
        if hasattr(value, "dtype"):
            original_dtype = str(value.dtype)
        value = value.to_numpy()
    if original_dtype is None and hasattr(value, "dtype"):
        original_dtype = str(value.dtype)
    array = np.asarray(value)
    return array, original_dtype or str(array.dtype)


def output_contract(value: Any, expected_count: int, *, range_tolerance: float = DEFAULT_RANGE_TOL) -> dict[str, Any]:
    """Check shape, dtype, finiteness, and normalized luminance range."""
    np = _require_numpy()
    try:
        array, dtype_label = _as_numpy_output(value)
    except Exception as exc:
        return {
            "status": "FAIL",
            "shape": None,
            "expected_shape": [int(expected_count)],
            "dtype": None,
            "finite": False,
            "range": {"min": None, "max": None, "valid": False},
            "error": f"could not materialize output: {exc}",
        }
    expected_shape = (int(expected_count),)
    shape_ok = tuple(array.shape) == expected_shape
    dtype_ok = array.dtype.kind == "f"
    finite = bool(np.all(np.isfinite(array))) if array.size else False
    if array.size:
        minimum = float(np.min(array))
        maximum = float(np.max(array))
    else:
        minimum = maximum = None
    range_ok = bool(
        finite and minimum is not None and maximum is not None
        and minimum >= -float(range_tolerance)
        and maximum <= 1.0 + float(range_tolerance)
    )
    return {
        "status": "PASS" if shape_ok and dtype_ok and finite and range_ok else "FAIL",
        "shape": [int(item) for item in array.shape],
        "expected_shape": [int(item) for item in expected_shape],
        "shape_ok": shape_ok,
        "dtype": dtype_label,
        "dtype_kind": str(array.dtype.kind),
        "dtype_ok": dtype_ok,
        "finite": finite,
        "range": {"min": minimum, "max": maximum, "tolerance": float(range_tolerance), "valid": range_ok},
    }


def compare_arrays(reference: Any, candidate: Any, *, abs_tol: float = DEFAULT_ABS_TOL,
                   rel_tol: float = DEFAULT_REL_TOL, mismatch_tol: float | None = None) -> dict[str, Any]:
    """Return exact-column error metrics without silently broadcasting shapes."""
    np = _require_numpy()
    if mismatch_tol is None:
        mismatch_tol = abs_tol
    reference_array, reference_dtype = _as_numpy_output(reference)
    candidate_array, candidate_dtype = _as_numpy_output(candidate)
    shape_match = tuple(reference_array.shape) == tuple(candidate_array.shape)
    if not shape_match or reference_array.size == 0:
        return {
            "status": "FAIL",
            "shape_match": shape_match,
            "reference_shape": [int(item) for item in reference_array.shape],
            "candidate_shape": [int(item) for item in candidate_array.shape],
            "reference_dtype": reference_dtype,
            "candidate_dtype": candidate_dtype,
            "max_abs": None,
            "rms": None,
            "rmse": None,
            "mean_abs": None,
            "mismatch_count": None,
            "mismatch_fraction": None,
            "abs_tol": float(abs_tol),
            "rel_tol": float(rel_tol),
            "mismatch_tol": float(mismatch_tol),
        }
    try:
        diff = np.asarray(candidate_array, dtype=np.float64) - np.asarray(reference_array, dtype=np.float64)
        finite = bool(np.all(np.isfinite(diff)))
    except Exception:
        finite = False
        diff = np.asarray([], dtype=np.float64)
    if not finite or diff.size == 0:
        return {
            "status": "FAIL",
            "shape_match": True,
            "reference_shape": [int(item) for item in reference_array.shape],
            "candidate_shape": [int(item) for item in candidate_array.shape],
            "reference_dtype": reference_dtype,
            "candidate_dtype": candidate_dtype,
            "max_abs": None,
            "rms": None,
            "rmse": None,
            "mean_abs": None,
            "mismatch_count": None,
            "mismatch_fraction": None,
            "finite": finite,
            "abs_tol": float(abs_tol),
            "rel_tol": float(rel_tol),
            "mismatch_tol": float(mismatch_tol),
        }
    abs_diff = np.abs(diff)
    reference_float = np.asarray(reference_array, dtype=np.float64)
    allowed = float(abs_tol) + float(rel_tol) * np.abs(reference_float)
    mismatch = abs_diff > np.maximum(allowed, float(mismatch_tol))
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    rms = float(np.sqrt(np.mean(np.square(diff))))
    mismatch_count = int(np.count_nonzero(mismatch))
    return {
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "shape_match": True,
        "reference_shape": [int(item) for item in reference_array.shape],
        "candidate_shape": [int(item) for item in candidate_array.shape],
        "reference_dtype": reference_dtype,
        "candidate_dtype": candidate_dtype,
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms": rms,
        "rmse": rms,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": float(mismatch_count / diff.size),
        "sample_count": int(diff.size),
        "abs_tol": float(abs_tol),
        "rel_tol": float(rel_tol),
        "mismatch_tol": float(mismatch_tol),
    }


def _first_value(owner: Any, names: Iterable[str], *, call: bool = False) -> Any:
    """Read the first metadata field, optionally invoking zero-argument getters."""
    if owner is None:
        return None
    for name in names:
        try:
            value = getattr(owner, name)
        except Exception:
            continue
        if callable(value):
            if not call:
                continue
            try:
                value = value()
            except Exception:
                continue
        if value is not None:
            return value
    return None


def backend_metadata(module: ModuleType, renderer: Any, *, requested: str) -> dict[str, Any]:
    """Collect backend identity and provenance, rejecting silent fallback."""
    observed_requested = _first_value(renderer, ("requested_backend", "requested_renderer_backend"))
    if observed_requested is None:
        observed_requested = _first_value(module, ("requested_backend", "requested_renderer_backend"))
    effective = _first_value(renderer, ("effective_backend", "effective_renderer_backend", "backend_name", "backend"))
    if effective is None:
        effective = _first_value(module, ("effective_backend", "effective_renderer_backend", "backend_name", "backend"))
    provenance = _first_value(renderer, ("provenance", "renderer_provenance", "metadata", "get_provenance"), call=True)
    if provenance is None:
        provenance = _first_value(module, ("provenance", "renderer_provenance", "metadata", "get_provenance"), call=True)
    if not isinstance(provenance, Mapping):
        provenance = {} if provenance is None else {"value": _json_safe(provenance)}
    effective_text = None if effective is None else str(effective)
    requested_text = str(requested)
    # ``warp`` and ``nvidia_warp`` are accepted aliases, but an analytic or
    # CPU backend is never accepted as a successful explicit Warp run.
    effective_lower = "" if effective_text is None else effective_text.lower()
    explicit_warp = any(token in effective_lower for token in ("warp", "cuda"))
    fallback = effective_lower in {"analytic_reference", "virtual_world_raycast", "cpu", "numpy_cpu"}
    identity_ok = bool(effective_text) and explicit_warp and not fallback
    observed_match = observed_requested is None or str(observed_requested) == requested_text
    provenance_payload = dict(provenance)
    provenance_payload.setdefault("module", getattr(module, "__name__", None))
    provenance_payload.setdefault("renderer_class", None if renderer is None else type(renderer).__name__)
    provenance_payload.setdefault("requested_backend", requested_text)
    provenance_payload.setdefault("effective_backend", effective_text)
    provenance_payload.setdefault("fallback_detected", fallback)
    return {
        "status": "PASS" if identity_ok and observed_match else "FAIL",
        "requested": requested_text,
        "observed_requested": None if observed_requested is None else str(observed_requested),
        "effective": effective_text,
        "effective_is_explicit_warp": identity_ok,
        "observed_requested_matches": observed_match,
        "fallback_detected": fallback,
        "module": getattr(module, "__name__", None),
        "renderer_class": None if renderer is None else type(renderer).__name__,
        "provenance": _json_safe(provenance_payload),
    }


def load_backend_module(module_name: str = DEFAULT_BACKEND_MODULE) -> ModuleType:
    """Import an optional backend only on explicit request."""
    try:
        return importlib.import_module(str(module_name))
    except Exception as exc:
        raise BackendUnavailableError(
            f"Warp backend module {module_name!r} is unavailable: {exc}. "
            "Install/enable NVIDIA Warp and ensure the backend is on PYTHONPATH."
        ) from exc


def _call_factory(factory: Callable[..., Any], world: Any, device: str | None) -> Any:
    """Call a backend factory across the small set of supported signatures."""
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    if device is not None:
        attempts.extend([
            ((world,), {"device": device}),
            ((), {"world": world, "device": device}),
        ])
    attempts.extend([
        ((world,), {}),
        ((), {"world": world}),
    ])
    errors: list[str] = []
    for args, kwargs in attempts:
        try:
            return factory(*args, **kwargs)
        except TypeError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:
            # Backend-specific unavailable errors (Warp import/device errors)
            # must be rendered as an actionable validator error rather than a
            # raw traceback from an optional module.
            raise BackendUnavailableError(
                f"Warp renderer factory {getattr(factory, '__name__', factory)!r} failed: {exc}"
            ) from exc
    raise BackendUnavailableError(
        f"could not construct Warp renderer from {getattr(factory, '__name__', factory)!r}: "
        + (errors[-1] if errors else "unsupported factory signature")
    )


def create_backend(module: ModuleType, world: Any, *, device: str | None = "cuda:0",
                   class_name: str | None = None) -> Any:
    """Construct a renderer from common module-level or class-level seams."""
    candidates: list[Any] = []
    for name in ("create_renderer", "build_renderer", "make_renderer"):
        value = getattr(module, name, None)
        if callable(value):
            candidates.append(value)
    names = [class_name] if class_name else []
    names.extend(("WorldRendererOpt", "WorldRenderer", "WarpWorldRenderer", "Renderer"))
    seen: set[int] = set()
    for name in names:
        if not name:
            continue
        value = getattr(module, name, None)
        if callable(value) and id(value) not in seen:
            candidates.append(value)
            seen.add(id(value))
    if not candidates:
        raise BackendUnavailableError(
            f"Warp backend module {module.__name__!r} exposes no renderer factory or known renderer class"
        )
    errors: list[str] = []
    for factory in candidates:
        from_world = getattr(factory, "from_world", None)
        if callable(from_world):
            try:
                return _call_factory(from_world, world, device)
            except BackendUnavailableError as exc:
                errors.append(str(exc))
        try:
            return _call_factory(factory, world, device)
        except BackendUnavailableError as exc:
            errors.append(str(exc))
    raise BackendUnavailableError(
        f"Warp backend module {module.__name__!r} could not construct a renderer: "
        + (errors[-1] if errors else "unknown construction error")
    )


def _load_reference_world(config: str | Path | Mapping[str, Any], stimuli: str | Path | None = None) -> Any:
    try:
        try:
            module = importlib.import_module("virtual_world")
        except ImportError:
            module = importlib.import_module("scripts.virtual_world")
        world_cls = getattr(module, "VirtualWorld")
    except Exception as exc:
        raise ReferenceUnavailableError(f"analytic VirtualWorld cannot be imported: {exc}") from exc
    try:
        kwargs: dict[str, Any] = {}
        if stimuli is not None:
            kwargs["stimuli"] = stimuli
        return world_cls.from_json(config, **kwargs)
    except Exception as exc:
        raise ReferenceUnavailableError(f"analytic VirtualWorld cannot load {config!r}: {exc}") from exc


def _default_pose(world: Any) -> Any:
    try:
        module = importlib.import_module("virtual_world")
    except ImportError:
        module = importlib.import_module("scripts.virtual_world")
    pose_cls = getattr(module, "Pose", None)
    if pose_cls is None:
        return (0.0, 0.0, 0.42)
    return pose_cls((0.0, 0.0, 0.42), yaw_rad=0.2, pitch_rad=0.1)


def _render_method(owner: Any) -> Callable[..., Any]:
    # Prefer a tensor-returning seam when available so the dtype check observes
    # the real CUDA output rather than a list wrapper that has already coerced
    # values to Python ``float`` objects.
    for name in (
        "render_eye_directions_tensor", "render_exact_directions_tensor",
        "render_eye_directions", "render_exact_directions", "render_directions",
    ):
        method = getattr(owner, name, None)
        if callable(method):
            return method
    raise WarpValidationError(f"renderer {type(owner).__name__} has no exact-direction rendering method")


def _invoke_render(method: Callable[..., Any], pose: Any, directions: Any, eye: str,
                   timestamp_s: float, stimulus: Any) -> Any:
    """Invoke a render seam while accommodating minor backend signature drift."""
    base_kwargs = {"timestamp_s": float(timestamp_s), "t_s": float(timestamp_s)}
    if stimulus is _NO_STIMULI:
        # Empty tuple is the public VirtualWorld convention for selecting no
        # configured stimuli.  It is deliberately different from omitting
        # both arguments, which means all configured stimuli.
        base_kwargs["stimuli"] = ()
    elif stimulus is _ALL_CONFIGURED_STIMULI:
        # Explicitly preserve the all-configured diagnostic.  Passing the
        # keyword makes the intent auditable even though ``None`` is the
        # renderer's all-configured selector.
        base_kwargs["stimuli"] = None
    elif stimulus is not None:
        base_kwargs.update({"stimulus": stimulus, "stimuli": stimulus})
    try:
        signature = inspect.signature(method)
        parameters = signature.parameters
        accepts_var_kw = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
        if stimulus is _NO_STIMULI and not accepts_var_kw and "stimuli" not in parameters:
            raise WarpValidationError(
                f"exact-direction render seam {getattr(method, '__qualname__', method)!r} "
                "must accept stimuli=() for the static no-stimulus baseline"
            )
        filtered = {key: value for key, value in base_kwargs.items() if accepts_var_kw or key in parameters}
    except (TypeError, ValueError):
        filtered = base_kwargs
    attempts = [
        ((pose, directions, eye), filtered),
        ((pose, directions), {**filtered, "eye": eye}),
        ((directions, eye), filtered),
        ((directions,), {**filtered, "eye": eye}),
    ]
    errors: list[str] = []
    for args, kwargs in attempts:
        try:
            return method(*args, **kwargs)
        except TypeError as exc:
            errors.append(str(exc))
    raise WarpValidationError(
        f"exact-direction render call failed for {getattr(method, '__qualname__', method)!r}: "
        + (errors[-1] if errors else "unsupported signature")
    )


def _synchronize(renderer: Any) -> None:
    for owner in (renderer, getattr(renderer, "device", None)):
        if owner is None:
            continue
        for name in ("synchronize", "sync"):
            method = getattr(owner, name, None)
            if callable(method):
                try:
                    method()
                except TypeError:
                    continue
                return


def _render_bilateral(renderer: Any, pose: Any, directions: DirectionSet, timestamp_s: float,
                      stimulus: Any) -> dict[str, Any]:
    method = _render_method(renderer)
    return {
        eye: _invoke_render(method, pose, directions.for_eye(eye), eye, timestamp_s, stimulus)
        for eye in directions.eyes
    }


def _stimulus_default(world: Any) -> Any:
    names = getattr(world, "stimulus_names", ())
    if callable(names):
        try:
            names = names()
        except Exception:
            names = ()
    if names:
        try:
            return sorted(names)[0]
        except Exception:
            return list(names)[0]
    stimuli = getattr(world, "stimuli", None)
    if isinstance(stimuli, Mapping) and stimuli:
        return sorted(stimuli)[0]
    return None


def _configured_stimulus_count(world: Any) -> int:
    stimuli = getattr(world, "stimuli", None)
    if isinstance(stimuli, Mapping):
        return len(stimuli)
    names = getattr(world, "stimulus_names", ())
    if callable(names):
        try:
            names = names()
        except Exception:
            names = ()
    try:
        return len(names)
    except (TypeError, AttributeError):
        return 0


def _nvidia_smi_snapshot(renderer: Any = None) -> dict[str, Any] | None:
    """Read device-level VRAM usage when the backend allocator is opaque.

    NVIDIA Warp owns its allocations outside Torch's caching allocator.  The
    ``nvidia-smi`` value is therefore a device-level observation, not a
    process-level attribution; the scope is recorded explicitly.  A missing
    binary/driver yields ``None`` so callers can report UNKNOWN rather than
    treating zero as evidence of zero VRAM use.
    """
    if renderer is None:
        return None
    device_name = getattr(renderer, "device_name", None)
    device_index = 0
    if device_name is not None:
        text = str(device_name)
        if ":" in text:
            try:
                device_index = int(text.rsplit(":", 1)[1])
            except ValueError:
                pass
        elif text.isdigit():
            device_index = int(text)
    if device_name is not None and not str(device_name).lower().startswith(("cuda", "gpu", "nvidia")):
        return None
    command = shutil.which("nvidia-smi")
    if command is None:
        return None
    try:
        result = subprocess.run(
            [command, "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            index = int(fields[0])
            used_mib = float(fields[1])
            total_mib = float(fields[2])
        except ValueError:
            continue
        if index != device_index:
            continue
        return {
            "available": True,
            "source": "nvidia-smi",
            "scope": "device_total",
            "device": f"cuda:{index}",
            "used_bytes": int(max(0.0, used_mib) * 1024**2),
            "total_bytes": int(max(0.0, total_mib) * 1024**2),
            # Normalize the device observation to the names consumed by the
            # resource report.  It is not falsely labelled process-reserved.
            "max_allocated_bytes": int(max(0.0, used_mib) * 1024**2),
            "max_reserved_bytes": int(max(0.0, used_mib) * 1024**2),
        }
    return None


def _memory_snapshot(renderer: Any = None) -> dict[str, Any]:
    """Observe RSS and non-invasive CUDA allocator counters."""
    current_rss = None
    peak_rss = None
    try:
        status_path = Path("/proc/self/status")
        if status_path.is_file():
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    current_rss = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    peak_rss = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ru_maxrss = int(usage.ru_maxrss)
        # Linux reports KiB, Darwin reports bytes.
        ru_bytes = ru_maxrss * 1024 if sys.platform != "darwin" else ru_maxrss
        peak_rss = max(peak_rss or 0, ru_bytes)
    except (AttributeError, OSError, ValueError):
        pass
    cuda: dict[str, Any] = {"available": False, "source": None}
    # Do not import torch merely to inspect memory: import safety is part of
    # this tool's contract.  If the renderer already imported it, counters are
    # cheap and valuable.
    torch = sys.modules.get("torch")
    torch_observation = False
    try:
        if torch is not None and bool(torch.cuda.is_available()):
            device = getattr(renderer, "device_name", None) or getattr(renderer, "device", None)
            kwargs = {} if device is None else {"device": device}
            cuda.update({
                "available": True,
                "source": "torch.cuda",
                "device": None if device is None else str(device),
                "allocated_bytes": int(torch.cuda.memory_allocated(**kwargs)),
                "reserved_bytes": int(torch.cuda.memory_reserved(**kwargs)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(**kwargs)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(**kwargs)),
            })
            torch_observation = any(
                int(cuda.get(key, 0) or 0) > 0
                for key in ("allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes")
            )
    except Exception as exc:
        cuda.update({"available": False, "source": "torch.cuda", "error": str(exc)})
    # A backend can expose allocator counters without importing Torch (for
    # example, via ``wp.get_device().context`` or a custom pool).  Accept a
    # small, JSON-friendly mapping and normalize the common aliases.  This is
    # observation only; the validator never resets or mutates allocator state.
    runtime_memory: Any = None
    for owner in (renderer,):
        if owner is None:
            continue
        for name in ("memory_stats", "resource_stats", "resource_usage", "memory_usage"):
            candidate = getattr(owner, name, None)
            if callable(candidate):
                try:
                    candidate = candidate()
                except Exception:
                    continue
            if isinstance(candidate, Mapping):
                runtime_memory = candidate
                break
        if runtime_memory is not None:
            break
    if isinstance(runtime_memory, Mapping):
        runtime_cuda = runtime_memory.get("cuda", runtime_memory.get("vram", runtime_memory))
        if isinstance(runtime_cuda, Mapping):
            aliases = {
                "allocated_bytes": ("allocated_bytes", "cuda_allocated_bytes", "current_allocated_bytes"),
                "reserved_bytes": ("reserved_bytes", "cuda_reserved_bytes", "current_reserved_bytes"),
                "max_allocated_bytes": ("max_allocated_bytes", "peak_allocated_bytes", "peak_cuda_allocated_bytes"),
                "max_reserved_bytes": ("max_reserved_bytes", "peak_reserved_bytes", "peak_cuda_reserved_bytes"),
            }
            for target, names in aliases.items():
                for name in names:
                    value = runtime_cuda.get(name)
                    if isinstance(value, (int, float)):
                        cuda[target] = int(value)
                        cuda["available"] = True
                        cuda["source"] = cuda.get("source") or "renderer"
                        break
    # A zero Torch reading is not meaningful for a Warp-owned allocation pool.
    # Prefer a driver observation in that case; if neither source is available
    # leave the counters unavailable instead of fabricating a zero.
    if not any(
        isinstance(cuda.get(key), (int, float)) and int(cuda.get(key) or 0) > 0
        for key in ("allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes")
    ):
        driver_cuda = _nvidia_smi_snapshot(renderer)
        if driver_cuda is not None:
            cuda = driver_cuda
        elif torch_observation:
            # Defensive branch; currently torch_observation implies a positive
            # value, but preserving this distinction makes future allocator
            # backends explicit.
            cuda["available"] = False
            cuda["source"] = "unknown"
            cuda["reason"] = "Torch allocator reported no observable bytes and Warp allocator has no telemetry"
        else:
            cuda["available"] = False
            cuda["source"] = "unknown"
            cuda["reason"] = "Warp allocator telemetry unavailable; Torch/nvidia-smi did not provide VRAM usage"
    return {
        "rss_bytes": current_rss,
        "peak_rss_bytes": peak_rss,
        "cuda": cuda,
    }


def _resource_report(before: Mapping[str, Any], after: Mapping[str, Any], limits: ValidationLimits) -> dict[str, Any]:
    rss_values = [item for item in (before.get("rss_bytes"), after.get("rss_bytes"), before.get("peak_rss_bytes"), after.get("peak_rss_bytes")) if isinstance(item, (int, float))]
    peak_rss = max(rss_values) if rss_values else None
    cuda_before = before.get("cuda") if isinstance(before.get("cuda"), Mapping) else {}
    cuda_after = after.get("cuda") if isinstance(after.get("cuda"), Mapping) else {}
    vram_values = [
        item for item in (
            cuda_before.get("max_reserved_bytes"), cuda_after.get("max_reserved_bytes"),
            cuda_before.get("reserved_bytes"), cuda_after.get("reserved_bytes"),
            cuda_before.get("max_allocated_bytes"), cuda_after.get("max_allocated_bytes"),
            cuda_before.get("allocated_bytes"), cuda_after.get("allocated_bytes"),
            cuda_before.get("used_bytes"), cuda_after.get("used_bytes"),
        ) if isinstance(item, (int, float))
    ]
    peak_vram = max(vram_values) if vram_values else None
    allocated_values = [
        item for item in (cuda_before.get("max_allocated_bytes"), cuda_after.get("max_allocated_bytes"),
                          cuda_before.get("allocated_bytes"), cuda_after.get("allocated_bytes"),
                          cuda_before.get("used_bytes"), cuda_after.get("used_bytes"))
        if isinstance(item, (int, float))
    ]
    peak_allocated = max(allocated_values) if allocated_values else None
    exceeded: list[str] = []
    if limits.ram_bytes is not None and peak_rss is not None and peak_rss > limits.ram_bytes:
        exceeded.append("ram")
    if limits.vram_bytes is not None and peak_vram is not None and peak_vram > limits.vram_bytes:
        exceeded.append("vram")
    ram_observed = peak_rss is not None
    # Some allocator probes publish numeric zero placeholders together with
    # ``available=false``.  Do not mistake those placeholders for a real
    # VRAM observation.
    vram_available = bool(cuda_before.get("available")) or bool(cuda_after.get("available"))
    vram_observed = vram_available and peak_vram is not None
    if not vram_observed:
        peak_vram = None
        peak_allocated = None
    observed_any = ram_observed or vram_observed
    # A Warp run with an unknown VRAM allocator must not be called PASS.  RSS
    # can still be reported independently for diagnosis.
    status = "FAIL" if exceeded else ("PASS" if ram_observed and vram_observed else "UNKNOWN")
    return {
        "status": status,
        "limits": {
            "ram_bytes": limits.ram_bytes,
            "vram_bytes": limits.vram_bytes,
            "host_rss_max_bytes": limits.ram_bytes,
            "cuda_allocated_max_bytes": limits.vram_bytes,
            "cuda_reserved_max_bytes": limits.vram_bytes,
            "cgroup_memory_max_bytes": limits.ram_bytes,
        },
        "observed": {
            "peak_rss_bytes": None if peak_rss is None else int(peak_rss),
            "peak_vram_bytes": None if peak_vram is None else int(peak_vram),
            "peak_cuda_allocated_bytes": None if peak_allocated is None else int(peak_allocated),
            "peak_cuda_reserved_bytes": None if peak_vram is None else int(peak_vram),
        },
        "usage": {
            "peak_host_rss_bytes": None if peak_rss is None else int(peak_rss),
            "peak_cuda_allocated_bytes": None if peak_allocated is None else int(peak_allocated),
            "peak_cuda_reserved_bytes": None if peak_vram is None else int(peak_vram),
        },
        "observation_status": {"rss": "PASS" if ram_observed else "UNKNOWN", "vram": "PASS" if vram_observed else "UNKNOWN"},
        "exceeded": exceeded,
        "before": _json_safe(before),
        "after": _json_safe(after),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _benchmark(renderer: Any, renderer_factory: Callable[[], Any] | None, pose: Any,
               directions: DirectionSet, timestamp_s: float, stimulus: Any,
               *, warmup: int, iterations: int) -> dict[str, Any]:
    warmup = _bounded_count(warmup, name="warmup", maximum=MAX_WARMUP)
    iterations = _bounded_count(iterations, name="iterations", maximum=MAX_ITERATIONS, minimum=1)
    benchmark_renderer = renderer
    cold_setup_s: float | None = None
    if renderer_factory is not None:
        started = time.perf_counter()
        benchmark_renderer = renderer_factory()
        cold_setup_s = time.perf_counter() - started
    cold_started = time.perf_counter()
    _render_bilateral(benchmark_renderer, pose, directions, timestamp_s, stimulus)
    _synchronize(benchmark_renderer)
    cold_first_render_s = time.perf_counter() - cold_started
    for _ in range(warmup):
        _render_bilateral(benchmark_renderer, pose, directions, timestamp_s, stimulus)
        _synchronize(benchmark_renderer)
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        _render_bilateral(benchmark_renderer, pose, directions, timestamp_s, stimulus)
        _synchronize(benchmark_renderer)
        samples.append(time.perf_counter() - started)
    cold_jit_s = cold_first_render_s + (cold_setup_s or 0.0)
    try:
        directions_per_batch = sum(int(_direction_array(directions.for_eye(eye)).shape[0]) for eye in directions.eyes)
    except Exception:
        directions_per_batch = 0
    device_name = str(getattr(benchmark_renderer, "device_name", "cuda:0"))
    timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    steady_wall_s = float(sum(samples))
    steady = {
        "name": "warp-steady",
        "version": "warp-renderer-validation-1",
        "backend": "warp_cuda",
        "device": device_name,
        "dtype": "float32",
        "warmup_iterations": warmup,
        "measurement_iterations": iterations,
        "sample_count": iterations,
        "simulated_duration_s": float(iterations),
        "wall_time_s": steady_wall_s,
        "throughput": {"value": (directions_per_batch * iterations / steady_wall_s) if steady_wall_s > 0 else 0.0, "unit": "eye-directions/s"},
        "synchronization": {"before": True, "after": True},
        "timestamp_utc": timestamp_utc,
        "command": "validate_warp_renderer.py --backend world_renderer_opt",
        "measurement_unit": "bilateral_render_batch",
        "iterations": iterations,
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "p95_s": _percentile(samples, 0.95),
        "includes_jit": False,
        "bilateral": True,
    }
    cold_record = {
        "name": "warp-cold-jit",
        "version": "warp-renderer-validation-1",
        "backend": "warp_cuda",
        "device": device_name,
        "dtype": "float32",
        "warmup_iterations": 0,
        "measurement_iterations": 1,
        "sample_count": 1,
        "simulated_duration_s": 1.0,
        "wall_time_s": cold_jit_s,
        "throughput": {"value": (directions_per_batch / cold_jit_s) if cold_jit_s > 0 else 0.0, "unit": "eye-directions/s"},
        "synchronization": {"before": True, "after": True},
        "timestamp_utc": timestamp_utc,
        "command": "validate_warp_renderer.py --backend world_renderer_opt",
        "measurement_unit": "bilateral_render_batch",
        "includes_jit": True,
        "bilateral": True,
        "setup_s": cold_setup_s,
        "first_render_s": cold_first_render_s,
    }
    return {
        "status": "PASS",
        "cold_setup": {
            "wall_s": cold_setup_s,
            "includes_jit": False,
            "renderer_recreated": renderer_factory is not None,
        },
        "cold_first_render": {
            "wall_s": cold_first_render_s,
            "includes_jit": True,
            "bilateral": True,
        },
        # Keep the compact aliases used by the project's existing benchmark
        # gate while retaining the more informative setup/first-render split.
        "cold_jit": {
            **cold_record,
            "wall_s": cold_jit_s,
        },
        "warmup": {"iterations": warmup, "excluded_from_steady": True},
        "steady": steady,
        "steady_host_boundary": steady,
        "warm_steady": steady,
        "steady_device_resident": {
            "status": "UNKNOWN",
            "reason": "device-resident direction probe is not available in this bounded host-boundary benchmark",
            "includes_jit": False,
        },
    }


def _case_report(world: Any, renderer: Any, directions: DirectionSet, pose: Any, *, name: str,
                 timestamp_s: float, stimulus: Any, abs_tol: float, rel_tol: float,
                 determinism_tol: float, range_tolerance: float,
                 mean_abs_tol: float = DEFAULT_MEAN_ABS_TOL) -> dict[str, Any]:
    reference_method = _render_method(world)
    candidate_method = _render_method(renderer)
    eyes: dict[str, Any] = {}
    case_pass = True
    for eye in directions.eyes:
        eye_directions = directions.for_eye(eye)
        count = int(_direction_array(eye_directions, source=f"{eye} directions").shape[0])
        reference = _invoke_render(reference_method, pose, eye_directions, eye, timestamp_s, stimulus)
        candidate = _invoke_render(candidate_method, pose, eye_directions, eye, timestamp_s, stimulus)
        repeated = _invoke_render(candidate_method, pose, eye_directions, eye, timestamp_s, stimulus)
        contract = output_contract(candidate, count, range_tolerance=range_tolerance)
        reference_contract = output_contract(reference, count, range_tolerance=range_tolerance)
        metrics = compare_arrays(reference, candidate, abs_tol=abs_tol, rel_tol=rel_tol)
        repeat_metrics = compare_arrays(candidate, repeated, abs_tol=determinism_tol, rel_tol=0.0,
                                       mismatch_tol=determinism_tol)
        determinism = {
            "status": "PASS" if repeat_metrics.get("status") == "PASS" else "FAIL",
            "exact": bool(repeat_metrics.get("max_abs") == 0.0) if repeat_metrics.get("max_abs") is not None else False,
            "max_abs": repeat_metrics.get("max_abs"),
            "rms": repeat_metrics.get("rms"),
            "mismatch_count": repeat_metrics.get("mismatch_count"),
            "tolerance": float(determinism_tol),
        }
        eye_pass = contract["status"] == "PASS" and reference_contract["status"] == "PASS" and metrics["status"] == "PASS" and determinism["status"] == "PASS"
        case_pass = case_pass and eye_pass
        eyes[eye] = {
            "status": "PASS" if eye_pass else "FAIL",
            "directions": count,
            "reference_contract": reference_contract,
            "candidate_contract": contract,
            "comparison": metrics,
            "determinism": determinism,
        }
    if stimulus is _NO_STIMULI:
        stimulus_record = None
        stimuli_mode = "explicit_empty"
    elif stimulus is _ALL_CONFIGURED_STIMULI:
        stimulus_record = None
        stimuli_mode = "all_configured_overlap"
    else:
        stimulus_record = _json_safe(stimulus)
        stimuli_mode = "selected" if stimulus is not None else "implicit_all_configured"
    return {
        "name": name,
        "status": "PASS" if case_pass else "FAIL",
        "timestamp_s": float(timestamp_s),
        "stimulus": stimulus_record,
        "stimuli_mode": stimuli_mode,
        "eyes": eyes,
    }


def _analytic_reference_report(cases: Sequence[Mapping[str, Any]], *, abs_tol: float,
                               mean_abs_tol: float, reference_id: str) -> dict[str, Any]:
    comparisons = [
        eye.get("comparison", {})
        for case in cases
        for eye in (case.get("eyes", {}) if isinstance(case.get("eyes"), Mapping) else {}).values()
    ]
    max_abs_values = [item.get("max_abs") for item in comparisons if isinstance(item.get("max_abs"), (int, float))]
    mean_abs_values = [item.get("mean_abs") for item in comparisons if isinstance(item.get("mean_abs"), (int, float))]
    mismatch_count = sum(int(item.get("mismatch_count") or 0) for item in comparisons)
    max_abs = max(max_abs_values) if max_abs_values else None
    mean_abs = max(mean_abs_values) if mean_abs_values else None
    status = "PASS" if (
        comparisons
        and all(case.get("status") == "PASS" for case in cases)
        and max_abs is not None
        and mean_abs is not None
        and max_abs <= float(abs_tol)
        and mean_abs <= float(mean_abs_tol)
        and mismatch_count == 0
    ) else "FAIL"
    return {
        "status": status,
        "reference_id": reference_id,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mismatch_count": mismatch_count,
        "tolerance": {
            "max_abs": float(abs_tol),
            "mean_abs": float(mean_abs_tol),
            "mismatch_count": 0,
        },
        "case_count": len(cases),
    }


def _timestamp_values(static_timestamps: Sequence[float],
                      dynamic_timestamps: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    static = tuple(float(item) for item in static_timestamps)
    dynamic = tuple(float(item) for item in dynamic_timestamps)
    if max(len(static), len(dynamic)) > MAX_TIMESTAMPS:
        raise ValueError(f"timestamp count must be <= {MAX_TIMESTAMPS}")
    if not all(math.isfinite(item) for item in static + dynamic):
        raise ValueError("timestamps must be finite")
    return static, dynamic


def _dynamic_status(dynamic: bool, stimulus: Any, timestamps: Sequence[float],
                    cases: Sequence[Mapping[str, Any]], start: int | None) -> dict[str, Any]:
    if not dynamic:
        return {"status": "SKIP", "reason": "dynamic validation disabled", "timestamps": list(timestamps)}
    if stimulus is None:
        return {"status": "SKIP", "reason": "no stimulus selected or loaded", "timestamps": list(timestamps)}
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in cases[start or 0:]) else "FAIL",
        "stimulus": _json_safe(stimulus),
        "timestamps": list(timestamps),
    }


def _overlap_report(world: Any, renderer: Any, directions: DirectionSet, pose: Any,
                    static_timestamps: Sequence[float], *, enabled: bool,
                    abs_tol: float, rel_tol: float, determinism_tol: float,
                    range_tolerance: float) -> dict[str, Any]:
    configured_count = _configured_stimulus_count(world)
    if enabled and configured_count > 1:
        cases = []
        for timestamp_s in static_timestamps or (0.0,):
            try:
                cases.append(_case_report(
                    world, renderer, directions, pose, name="all_configured_overlap",
                    timestamp_s=timestamp_s, stimulus=_ALL_CONFIGURED_STIMULI,
                    abs_tol=abs_tol, rel_tol=rel_tol,
                    determinism_tol=determinism_tol, range_tolerance=range_tolerance,
                ))
            except Exception as exc:
                cases.append({"status": "ERROR", "timestamp_s": float(timestamp_s), "error": str(exc)})
        return {
            "status": "NON_CORE_KNOWN_TIE",
            "core": False,
            "known_tie": True,
            "reason": "all configured stimuli can overlap; BVH face selection is not a core static baseline",
            "configured_stimulus_count": configured_count,
            "cases": cases,
        }
    if enabled:
        reason = "fewer than two configured stimuli"
    else:
        reason = "overlap diagnostic disabled"
    return {
        "status": "SKIP",
        "core": False,
        "known_tie": False,
        "reason": reason,
        "configured_stimulus_count": configured_count,
        "cases": [],
    }


def _summary_report(cases: Sequence[Mapping[str, Any]], passed: bool) -> dict[str, Any]:
    comparisons = [eye["comparison"] for case in cases for eye in case["eyes"].values()]
    fields = {"max_abs": "max_abs", "mean_abs": "mean_abs", "rms_max": "rms", "rmse": "rmse"}
    summary: dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "case_count": len(cases),
        "failed_cases": [item["name"] + "@" + str(item["timestamp_s"]) for item in cases if item["status"] != "PASS"],
    }
    summary.update({
        output: max((item[field] for item in comparisons if item.get(field) is not None), default=None)
        for output, field in fields.items()
    })
    summary["mismatch_count"] = sum(int(item.get("mismatch_count") or 0) for item in comparisons)
    return summary


def run_validation(*, config: str | Path | Mapping[str, Any] | None = None,
                   stimuli: str | Path | None = None, backend_module: str = DEFAULT_BACKEND_MODULE,
                   backend_class: str | None = None, device: str | None = "cuda:0",
                   requested_backend: str = DEFAULT_REQUESTED_BACKEND,
                   directions: DirectionSet | Any | None = None, directions_path: str | Path | None = None,
                   direction_count: int = 721, pose: Any = None,
                   static_timestamps: Sequence[float] = (0.0,),
                   dynamic_timestamps: Sequence[float] = (0.0, 0.37, 0.73),
                   dynamic: bool = True, stimulus: Any = None,
                   include_all_configured_overlap: bool = True,
                   warmup: int = 1, iterations: int = 3,
                   abs_tol: float = DEFAULT_ABS_TOL, rel_tol: float = DEFAULT_REL_TOL,
                   mean_abs_tol: float = DEFAULT_MEAN_ABS_TOL,
                   determinism_tol: float = DEFAULT_DETERMINISM_TOL,
                   range_tolerance: float = DEFAULT_RANGE_TOL,
                   ram_limit_gb: float | None = DEFAULT_RAM_LIMIT_GB,
                   vram_limit_gb: float | None = DEFAULT_VRAM_LIMIT_GB) -> dict[str, Any]:
    """Run bounded bilateral validation and return a JSON-ready report.

    ``directions`` may be a :class:`DirectionSet`, a plain ``(N, 3)`` array,
    or a bilateral mapping.  Supplying a prebuilt ``world``/``renderer`` is
    intentionally not part of this public path; unit tests can use
    :func:`validate_objects` below to avoid importing Warp.
    """
    if config is None:
        config = Path(__file__).resolve().parents[1] / "config" / "scenarios.json"
    _bounded_count(warmup, name="warmup", maximum=MAX_WARMUP)
    _bounded_count(iterations, name="iterations", maximum=MAX_ITERATIONS, minimum=1)
    static_values, dynamic_values = _timestamp_values(static_timestamps, dynamic_timestamps)
    world = _load_reference_world(config, stimuli)
    module = load_backend_module(backend_module)
    renderer_factory = lambda: create_backend(module, world, device=device, class_name=backend_class)
    before = _memory_snapshot()
    renderer = renderer_factory()
    metadata = backend_metadata(module, renderer, requested=requested_backend)
    if directions is None:
        direction_set = resolve_directions(directions_path, renderer=renderer, world=world, count=direction_count)
    elif isinstance(directions, DirectionSet):
        direction_set = directions
    elif isinstance(directions, Mapping):
        direction_set = _extract_direction_payload(directions)
    else:
        direction_set = DirectionSet(common=_direction_array(directions))
    pose = _default_pose(world) if pose is None else pose
    cases: list[dict[str, Any]] = []
    for timestamp_s in static_values:
        try:
            cases.append(_case_report(world, renderer, direction_set, pose, name="static", timestamp_s=timestamp_s,
                                      stimulus=_NO_STIMULI, abs_tol=abs_tol, rel_tol=rel_tol,
                                      determinism_tol=determinism_tol, range_tolerance=range_tolerance))
        except Exception as exc:
            raise WarpValidationError(f"static exact-direction render failed at t={timestamp_s:g}s: {exc}") from exc
    dynamic_stimulus = stimulus if stimulus is not None else _stimulus_default(world)
    dynamic_start = len(cases) if dynamic and dynamic_stimulus is not None else None
    if dynamic_start is not None:
        for timestamp_s in dynamic_values:
            try:
                cases.append(_case_report(world, renderer, direction_set, pose, name="dynamic", timestamp_s=timestamp_s,
                                          stimulus=dynamic_stimulus, abs_tol=abs_tol, rel_tol=rel_tol,
                                          determinism_tol=determinism_tol, range_tolerance=range_tolerance))
            except Exception as exc:
                raise WarpValidationError(f"dynamic exact-direction render failed at t={timestamp_s:g}s: {exc}") from exc
    dynamic_status = _dynamic_status(dynamic, dynamic_stimulus, dynamic_values, cases, dynamic_start)
    overlap_report = _overlap_report(
        world, renderer, direction_set, pose, static_values,
        enabled=include_all_configured_overlap, abs_tol=abs_tol, rel_tol=rel_tol,
        determinism_tol=determinism_tol, range_tolerance=range_tolerance,
    )
    try:
        benchmark = _benchmark(renderer, renderer_factory, pose, direction_set,
                               (static_values or dynamic_values or (0.0,))[0], _NO_STIMULI,
                               warmup=warmup, iterations=iterations)
    except Exception as exc:
        raise WarpValidationError(f"Warp cold/warm benchmark failed: {exc}") from exc
    # ``_benchmark`` intentionally keeps runtime objects out of its JSON
    # payload.  Process-wide RSS/driver counters and the original renderer's
    # memory_stats are sufficient for the post-run observation.
    after = _memory_snapshot(renderer)
    limits = ValidationLimits(
        None if ram_limit_gb is None else int(float(ram_limit_gb) * 1024**3),
        None if vram_limit_gb is None else int(float(vram_limit_gb) * 1024**3),
    )
    resources = _resource_report(before, after, limits)
    case_pass = all(item["status"] == "PASS" for item in cases) if cases else False
    # For an explicit Warp run, unknown VRAM usage is an incomplete resource
    # observation, not a successful zero-byte measurement.
    overall = case_pass and metadata["status"] == "PASS" and benchmark["status"] == "PASS" and resources["status"] == "PASS"
    config_decl = _config_declaration(config)
    scene_decl = _scene_declaration(renderer, config_decl)
    direction_decl = _direction_metadata(direction_set, directions_path)
    gpu_decl = _runtime_gpu_metadata(module, renderer)
    reference_id = f"VirtualWorld:analytic_reference:{config_decl.get('sha256')}"
    analytic_reference = _analytic_reference_report(
        cases, abs_tol=abs_tol, mean_abs_tol=mean_abs_tol, reference_id=reference_id,
    )
    exact_columns = {
        "file": direction_decl.get("file"),
        "shape_column": "shape",
        "direction_column": "direction",
        "shape_hash": direction_decl.get("shape_hash"),
        "direction_hash": direction_decl.get("direction_hash"),
        "algorithm": "sha256-canonical-json-lines-v1",
        "left_direction_hash": direction_decl.get("left_direction_hash"),
        "right_direction_hash": direction_decl.get("right_direction_hash"),
        "shape": direction_decl.get("shape"),
    }
    warp_report = {
        "version": gpu_decl.get("version") or "unknown",
        "gpu": {
            "name": gpu_decl.get("name") or "unknown",
            "arch": gpu_decl.get("arch") or "unknown",
            "compute_capability": gpu_decl.get("compute_capability") or "unknown",
            "device": gpu_decl.get("device"),
            "status": gpu_decl.get("status"),
        },
        "config": config_decl,
        "scene": scene_decl,
        "exact_columns": exact_columns,
        "benchmarks": {"cold_jit": benchmark.get("cold_jit"), "steady": benchmark.get("steady")},
        "analytic_reference": analytic_reference,
        "resources": resources,
    }
    return {
        "schema_version": "warp-renderer-validation-1",
        "status": "PASS" if overall else "FAIL",
        "backend": metadata,
        # Flat aliases make the artifact easy to consume from existing
        # experiment manifests, while ``backend`` retains the richer record.
        "renderer_backend": metadata["effective"],
        "requested_renderer_backend": metadata["requested"],
        "effective_renderer_backend": metadata["effective"],
        "provenance": metadata["provenance"],
        "config": config_decl,
        "scene": scene_decl,
        "gpu": warp_report["gpu"],
        "warp_version": warp_report["version"],
        "exact_columns": exact_columns,
        "analytic_reference": analytic_reference,
        "warp": warp_report,
        "directions": {
            "source": direction_set.source,
            "left_count": int(_direction_array(direction_set.for_eye("left")).shape[0]),
            "right_count": int(_direction_array(direction_set.for_eye("right")).shape[0]),
            "bilateral_exact": True,
        },
        "reference": {
            "backend": str(getattr(world, "backend_role", "analytic_reference")),
            "required": "analytic_reference",
            "config": _json_safe(config),
            "stimuli": _json_safe(stimuli),
        },
        "static": {"timestamps": list(static_values), "status": "PASS" if all(item["status"] == "PASS" for item in cases if item["name"] == "static") else "FAIL"},
        "dynamic": dynamic_status,
        "all_configured_overlap": overlap_report,
        "cases": cases,
        "benchmark": benchmark,
        "resources": resources,
        "limits": resources["limits"],
        "summary": _summary_report(cases, overall),
    }


def validate_objects(*, world: Any, renderer: Any, module: ModuleType | None = None,
                     renderer_factory: Callable[[], Any] | None = None,
                     requested_backend: str = DEFAULT_REQUESTED_BACKEND,
                     directions: DirectionSet | Any, pose: Any,
                   static_timestamps: Sequence[float] = (0.0,),
                   dynamic_timestamps: Sequence[float] = (0.0, 0.37), dynamic: bool = True,
                     stimulus: Any = None, include_all_configured_overlap: bool = True,
                     warmup: int = 0, iterations: int = 1,
                     abs_tol: float = DEFAULT_ABS_TOL, rel_tol: float = DEFAULT_REL_TOL,
                     mean_abs_tol: float = DEFAULT_MEAN_ABS_TOL,
                     determinism_tol: float = DEFAULT_DETERMINISM_TOL,
                     range_tolerance: float = DEFAULT_RANGE_TOL,
                     ram_limit_gb: float | None = DEFAULT_RAM_LIMIT_GB,
                     vram_limit_gb: float | None = DEFAULT_VRAM_LIMIT_GB) -> dict[str, Any]:
    """Dependency-light entry point for tests and embedded callers.

    The supplied ``world`` is the analytic reference object.  ``module`` may
    be a tiny fake module carrying backend identity; omitted metadata defaults
    to the renderer's own attributes.  No files are read or written here.
    """
    if not isinstance(directions, DirectionSet):
        directions = _extract_direction_payload(directions) if isinstance(directions, Mapping) else DirectionSet(common=_direction_array(directions))
    static_values, dynamic_values = _timestamp_values(static_timestamps, dynamic_timestamps)
    module = module or ModuleType(type(renderer).__module__)
    before = _memory_snapshot(renderer)
    metadata = backend_metadata(module, renderer, requested=requested_backend)
    cases: list[dict[str, Any]] = []
    for timestamp_s in static_values:
        cases.append(_case_report(world, renderer, directions, pose, name="static", timestamp_s=float(timestamp_s), stimulus=_NO_STIMULI,
                                  abs_tol=abs_tol, rel_tol=rel_tol, determinism_tol=determinism_tol,
                                  range_tolerance=range_tolerance))
    dynamic_stimulus = stimulus if stimulus is not None else _stimulus_default(world)
    dynamic_start = len(cases) if dynamic and dynamic_stimulus is not None else None
    if dynamic_start is not None:
        for timestamp_s in dynamic_values:
            cases.append(_case_report(world, renderer, directions, pose, name="dynamic", timestamp_s=float(timestamp_s), stimulus=dynamic_stimulus,
                                      abs_tol=abs_tol, rel_tol=rel_tol, determinism_tol=determinism_tol,
                                      range_tolerance=range_tolerance))
    dynamic_report = _dynamic_status(dynamic, dynamic_stimulus, dynamic_values, cases, dynamic_start)
    overlap_report = _overlap_report(
        world, renderer, directions, pose, static_values,
        enabled=include_all_configured_overlap, abs_tol=abs_tol, rel_tol=rel_tol,
        determinism_tol=determinism_tol, range_tolerance=range_tolerance,
    )
    benchmark = _benchmark(renderer, renderer_factory, pose, directions,
                           float((static_values or dynamic_values or (0.0,))[0]), _NO_STIMULI,
                           warmup=warmup, iterations=iterations)
    after = _memory_snapshot(renderer)
    resources = _resource_report(before, after, ValidationLimits(
        None if ram_limit_gb is None else int(float(ram_limit_gb) * 1024**3),
        None if vram_limit_gb is None else int(float(vram_limit_gb) * 1024**3),
    ))
    overall = bool(cases) and all(item["status"] == "PASS" for item in cases) and metadata["status"] == "PASS" and resources["status"] == "PASS"
    return {
        "schema_version": "warp-renderer-validation-1",
        "status": "PASS" if overall else "FAIL",
        "backend": metadata,
        "renderer_backend": metadata["effective"],
        "requested_renderer_backend": metadata["requested"],
        "effective_renderer_backend": metadata["effective"],
        "provenance": metadata["provenance"],
        "directions": {"source": directions.source, "left_count": int(_direction_array(directions.for_eye("left")).shape[0]), "right_count": int(_direction_array(directions.for_eye("right")).shape[0]), "bilateral_exact": True},
        "reference": {"backend": str(getattr(world, "backend_role", "analytic_reference")), "required": "analytic_reference"},
        "static": {"timestamps": list(static_values), "status": "PASS" if all(item["status"] == "PASS" for item in cases if item["name"] == "static") else "FAIL"},
        "dynamic": dynamic_report,
        "all_configured_overlap": overlap_report,
        "cases": cases,
        "benchmark": benchmark,
        "resources": resources,
        "limits": resources["limits"],
        "summary": _summary_report(cases, overall),
    }


def write_artifact(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write one explicit JSON artifact; never called implicitly by validation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(dict(report)), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


# Small compatibility aliases for callers that use the validator as a library.
# The canonical names remain explicit above; aliases avoid forcing an embedded
# experiment runner to care about whether this module calls the operation a
# validation or a comparison.
validate_backend = run_validation
compare_exact_directions = compare_arrays
validate_output = output_contract
save_report = write_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "scenarios.json")
    parser.add_argument("--stimuli", type=Path)
    parser.add_argument("--backend-module", "--backend", dest="backend_module", default=DEFAULT_BACKEND_MODULE)
    parser.add_argument("--backend-class")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--requested-backend", default=DEFAULT_REQUESTED_BACKEND)
    parser.add_argument("--directions", "--direction-file", dest="directions", type=Path)
    parser.add_argument("--direction-count", type=int, default=721)
    parser.add_argument("--stimulus")
    parser.add_argument("--static-timestamp", dest="static_timestamps", action="append", type=float)
    parser.add_argument("--dynamic-timestamp", dest="dynamic_timestamps", action="append", type=float)
    parser.add_argument("--no-dynamic", action="store_true")
    parser.add_argument("--no-overlap-diagnostic", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    parser.add_argument("--mean-abs-tol", type=float, default=DEFAULT_MEAN_ABS_TOL)
    parser.add_argument("--determinism-tol", type=float, default=DEFAULT_DETERMINISM_TOL)
    parser.add_argument("--range-tolerance", type=float, default=DEFAULT_RANGE_TOL)
    parser.add_argument("--ram-limit-gb", type=float, default=DEFAULT_RAM_LIMIT_GB)
    parser.add_argument("--vram-limit-gb", type=float, default=DEFAULT_VRAM_LIMIT_GB)
    parser.add_argument("--artifact", "--output", dest="artifact", type=Path, help="optional explicit JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_validation(
            config=args.config, stimuli=args.stimuli, backend_module=args.backend_module,
            backend_class=args.backend_class, device=args.device,
            requested_backend=args.requested_backend, directions_path=args.directions,
            direction_count=args.direction_count,
            static_timestamps=args.static_timestamps or (0.0,),
            dynamic_timestamps=args.dynamic_timestamps or (0.0, 0.37, 0.73),
            dynamic=not args.no_dynamic,
            include_all_configured_overlap=not args.no_overlap_diagnostic,
            stimulus=args.stimulus, warmup=args.warmup, iterations=args.iterations,
            abs_tol=args.abs_tol, rel_tol=args.rel_tol, mean_abs_tol=args.mean_abs_tol,
            determinism_tol=args.determinism_tol,
            range_tolerance=args.range_tolerance, ram_limit_gb=args.ram_limit_gb,
            vram_limit_gb=args.vram_limit_gb,
        )
    except BackendUnavailableError as exc:
        print(f"Warp validation unavailable: {exc}", file=sys.stderr)
        return 2
    except (ReferenceUnavailableError, WarpValidationError, ValueError, FileNotFoundError) as exc:
        print(f"Warp validation failed: {exc}", file=sys.stderr)
        return 1
    if args.artifact is not None:
        try:
            write_artifact(report, args.artifact)
        except OSError as exc:
            print(f"Warp validation artifact write failed: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(_json_safe(report), sort_keys=True, indent=2, allow_nan=False))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests.
    raise SystemExit(main())


__all__ = [
    "BackendUnavailableError",
    "DEFAULT_ABS_TOL",
    "DEFAULT_REL_TOL",
    "DirectionSet",
    "ReferenceUnavailableError",
    "ValidationLimits",
    "WarpValidationError",
    "backend_metadata",
    "compare_arrays",
    "compare_exact_directions",
    "create_backend",
    "default_directions",
    "load_backend_module",
    "load_directions",
    "main",
    "output_contract",
    "resolve_directions",
    "run_validation",
    "save_report",
    "validate_backend",
    "validate_output",
    "validate_objects",
    "write_artifact",
]
