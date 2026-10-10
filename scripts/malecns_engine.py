#!/usr/bin/env python3
"""Injectable MaleCNS stepping engine.

This module extracts the validated update order from ``lif_flyb.py`` into a
reusable ``MaleCNSEngine``.  It accepts the shared ``SensoryFrame``/
``Timestamp`` contracts and returns a ``NeuralOutput`` at every step.

The default backend is dependency-light NumPy CPU execution.  Torch CPU is an
explicit opt-in (``backend='torch_cpu'`` or ``backend='torch', device='cpu'``)
and is never imported by default.  CUDA Torch/Triton execution calls the
existing fail-closed ``malecns_gate`` immediately before optional imports.
The reusable Triton path mirrors the validated fused kernels from
``lif_flyb.py`` and is only selected explicitly (or by an explicit Triton
factory request); it never silently falls back.
CUDA Graph capture/replay is an additional explicit ``cuda_graph=True`` opt-in
for fixed held-input windows.  It uses device step/seed scalars so replay does
not freeze delay slots, refractory deadlines, or Poisson seeds; unavailable or
failed capture falls back to uncaptured Triton with a manifest reason.

Model order (matching ``lif_flyb.py``)
--------------------------------------
1. External Poisson generators fire *before* integration.  They reset the
   membrane and set the refractory deadline without changing synaptic ``g``.
2. The current delay-ring slot is added to ``g`` and cleared.  Refractory
   neurons clamp ``v`` and freeze ``g``; other neurons receive the exact
   two-state exponential update.
3. Thresholded spikes reset ``v``/``g`` and extend the refractory deadline.
4. Every emitted spike delivers its signed weighted CSR row into the future
   delay slot ``(step + delay_steps) % slots``.

Short-term plasticity remains in ``lif_flyb.py`` and is unsupported here.  The
real CSR is only inspected by ``preflight_csr`` unless a caller explicitly
constructs an engine and steps it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from population_selectors import comma_terms, intersect_terms  # noqa: E402
from twin_contracts import (  # noqa: E402
    ContractError,
    NeuralOutput,
    SensoryFrame,
    SourceLabel,
    Timestamp,
)


DEFAULT_CSR = Path("/data/cache/malecns-gpu/csr-flyb")
CSR = DEFAULT_CSR
DEFAULT_SENSORY_MAP = SCRIPT_DIR.parent / "config" / "sensory_map.json"
DEFAULT_DT_MS = 0.5
DEFAULT_GAIN = 0.65
DEFAULT_KC_GAIN = 0.25
DEFAULT_MAX_VRAM_GIB = 15.5
W_SYN_MV = 0.275
TAU_M_MS = 20.0
TAU_S_MS = 5.0
V_REST_MV = -52.0
V_THR_MV = -45.0
V_RESET_MV = -52.0
DELAY_MS = 1.8
REFRACTORY_MS = 2.2
SCHEMA_VERSION = "malecns-engine-1"
MECHANOSENSORY_SCHEMA_VERSION = 1
MECHANOSENSORY_INPUT_UNITS = "normalized"
MECHANOSENSORY_OUTPUT_UNITS = "current_mV_per_ms"
MECHANOSENSORY_INPUT_MIN = -1.0
MECHANOSENSORY_INPUT_MAX = 1.0
MECHANOSENSORY_MAX_GAIN_MV_PER_MS = 0.5
MAX_TRITON_WARMUP_STEPS = 32
MAX_TRITON_WINDOW_STEPS = 10
MAX_FLYVIS_CURRENT_MV_PER_MS = 20.0
MAX_FLYVIS_POISSON_HZ = 2000.0


class EngineError(RuntimeError):
    """Base class for engine setup and stepping errors."""


class CSRValidationError(EngineError):
    """Raised when a CSR directory is missing or internally inconsistent."""


class PopulationError(EngineError):
    """Raised for an unknown or empty population selector."""


class DriveError(EngineError):
    """Raised when an external drive is malformed or ambiguously specified."""


class TimestampError(EngineError):
    """Raised when a step timestamp is not monotonic at the engine ``dt``."""


class GateClosedError(EngineError):
    """Raised before importing Torch when the MaleCNS safety gate is closed."""

    def __init__(self, blockers: Sequence[str]) -> None:
        self.blockers = tuple(str(x) for x in blockers)
        message = "CUDA gate closed; refusing to import Torch/CUDA"
        if self.blockers:
            message += ":\n  - " + "\n  - ".join(self.blockers)
        super().__init__(message)


class UnsupportedBackendError(EngineError):
    """Raised when a requested backend is detected but not implemented here."""


@dataclass(frozen=True)
class MechanosensoryDriveSpec:
    """Validated config entry for one population-drive injection.

    ``sources`` are channel suffixes (the engine accepts
    ``population:<suffix>``), while ``target_population`` is an annotation
    selector name from the same config.  No dense indices or body IDs are
    persisted in this object.
    """

    name: str
    sources: tuple[str, ...]
    target_population: str
    gain_mV_per_ms: float
    max_abs_current_mV_per_ms: float
    priority: int = 0
    overlap_group: Optional[str] = None
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sources": list(self.sources),
            "target_population": self.target_population,
            "gain_mV_per_ms": self.gain_mV_per_ms,
            "max_abs_current_mV_per_ms": self.max_abs_current_mV_per_ms,
            "priority": self.priority,
            "overlap_group": self.overlap_group,
            "provenance": self.provenance,
        }


def _load_mechanosensory_map(path: str | Path = DEFAULT_SENSORY_MAP) -> tuple[dict[str, Any], tuple[MechanosensoryDriveSpec, ...], str]:
    """Read and validate the annotation-backed drive map.

    A map is an executable input contract, so malformed custom files are not
    replaced by a permissive fallback.  This keeps telemetry and a stale map
    from silently turning into a different neural experiment.
    """

    map_path = Path(path)
    try:
        raw = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DriveError(f"cannot read mechanosensory map {map_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise DriveError("mechanosensory map must contain a JSON object")
    mapping = raw.get("drive_mapping", raw.get(
        "population_drive_mapping", raw.get("mechanosensory_drive_mapping")
    ))
    if not isinstance(mapping, Mapping):
        raise DriveError("mechanosensory map is missing drive_mapping")
    schema = _int_or_none(mapping.get("schema_version"))
    if schema != MECHANOSENSORY_SCHEMA_VERSION:
        raise DriveError(
            f"unsupported mechanosensory map schema {mapping.get('schema_version')!r}; "
            f"expected {MECHANOSENSORY_SCHEMA_VERSION}"
        )
    if mapping.get("input_units") != MECHANOSENSORY_INPUT_UNITS:
        raise DriveError("mechanosensory map input_units must be 'normalized'")
    if mapping.get("output_units") != MECHANOSENSORY_OUTPUT_UNITS:
        raise DriveError(
            "mechanosensory map output_units must be 'current_mV_per_ms'"
        )
    if mapping.get("body_id_mapping", "not provided") != "not provided":
        raise DriveError("mechanosensory maps must not claim body-ID mappings")
    raw_range = mapping.get("input_range")
    if not isinstance(raw_range, Sequence) or isinstance(raw_range, (str, bytes)) or len(raw_range) != 2:
        raise DriveError("mechanosensory map input_range must contain two numbers")
    input_min = _scalar(raw_range[0], "mechanosensory input_range[0]")
    input_max = _scalar(raw_range[1], "mechanosensory input_range[1]")
    if (input_min != MECHANOSENSORY_INPUT_MIN or input_max != MECHANOSENSORY_INPUT_MAX):
        raise DriveError("mechanosensory map input_range must be exactly [-1.0, 1.0]")
    max_gain = _scalar(mapping.get("max_gain_mV_per_ms"), "mechanosensory max_gain_mV_per_ms")
    if not 0.0 < max_gain <= MECHANOSENSORY_MAX_GAIN_MV_PER_MS:
        raise DriveError(
            "mechanosensory max_gain_mV_per_ms must be in (0, "
            f"{MECHANOSENSORY_MAX_GAIN_MV_PER_MS:g}]"
        )
    entries = mapping.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise DriveError("mechanosensory map entries must be a non-empty object")
    specs: list[MechanosensoryDriveSpec] = []
    seen_sources: dict[str, str] = {}
    for raw_name, raw_entry in entries.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_entry, Mapping):
            raise DriveError(f"mechanosensory map entry {raw_name!r} is malformed")
        sources = raw_entry.get("sources")
        if isinstance(sources, str) or not isinstance(sources, Sequence) or not sources:
            raise DriveError(f"mechanosensory map entry {name!r} needs non-empty sources")
        source_values = tuple(str(source).strip() for source in sources)
        if any(not source or source.startswith("population:") for source in source_values):
            raise DriveError(f"mechanosensory map entry {name!r} has an invalid source")
        duplicate_sources = [source for source in source_values if source in seen_sources]
        if duplicate_sources:
            raise DriveError(
                f"mechanosensory source aliases are ambiguous: {duplicate_sources} "
                f"already belong to {seen_sources[duplicate_sources[0]]!r}"
            )
        target = str(raw_entry.get("target_population", raw_entry.get("target", ""))).strip()
        if not target or "body:" in target.lower():
            raise DriveError(
                f"mechanosensory map entry {name!r} needs an annotation target; body IDs are forbidden"
            )
        if any(key in raw_entry for key in ("body_id", "body_ids", "neuron_ids", "indices")):
            raise DriveError(f"mechanosensory map entry {name!r} must not contain neuron/body IDs")
        gain = _scalar(
            raw_entry.get("gain_mV_per_ms", raw_entry.get("gain")),
            f"mechanosensory {name} gain_mV_per_ms",
        )
        max_abs = _scalar(
            raw_entry.get("max_abs_current_mV_per_ms", gain),
            f"mechanosensory {name} max_abs_current_mV_per_ms",
        )
        if not 0.0 < gain <= max_gain or not 0.0 < max_abs <= max_gain:
            raise DriveError(
                f"mechanosensory {name} gain/current bound must be in (0, {max_gain:g}]"
            )
        if gain > max_abs + 1e-12:
            raise DriveError(f"mechanosensory {name} gain exceeds its current bound")
        priority = _int_or_none(raw_entry.get("priority", 0))
        if priority is None:
            raise DriveError(f"mechanosensory {name} priority must be an integer")
        overlap = raw_entry.get("overlap_group")
        overlap_group = None if overlap is None else str(overlap).strip()
        provenance = str(raw_entry.get("provenance", ""))
        spec = MechanosensoryDriveSpec(
            name=name,
            sources=source_values,
            target_population=target,
            gain_mV_per_ms=gain,
            max_abs_current_mV_per_ms=max_abs,
            priority=priority,
            overlap_group=overlap_group,
            provenance=provenance,
        )
        specs.append(spec)
        for source in source_values:
            seen_sources[source] = name
    digest = hashlib.sha256(map_path.read_bytes()).hexdigest()
    normalized_raw = dict(raw)
    # Keep one canonical in-memory key even when a caller uses one of the
    # descriptive compatibility spellings accepted above.
    normalized_raw["drive_mapping"] = dict(mapping)
    return normalized_raw, tuple(specs), digest


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def _array_summary(value: Any) -> dict[str, Any]:
    """Return bounded metadata for a dense numeric payload.

    Minimal flight telemetry must not recursively materialize a dense FlyVis
    vector with ``tolist``.  Hashing and scalar statistics retain provenance
    without retaining the payload itself in every neural output.
    """

    array = np.asarray(value)
    summary: dict[str, Any] = {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "count": int(array.size),
        "sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest(),
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        summary["min"] = float(np.min(array))
        summary["max"] = float(np.max(array))
        summary["sum"] = float(np.sum(array, dtype=np.float64))
    return summary


def _flyvis_payload_summary(value: Any) -> Any:
    """Summarize FlyVis input while keeping units and mapping provenance."""

    if isinstance(value, Mapping):
        summary: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, np.ndarray) or isinstance(item, (list, tuple)):
                summary[key_text] = _array_summary(item)
            elif isinstance(item, Mapping):
                summary[key_text] = _flyvis_payload_summary(item)
            else:
                summary[key_text] = _json_safe(item)
        # These aliases make the bounded contract explicit to consumers that
        # need only mapping cardinality/hash/units, without inspecting nested
        # payload details.
        target = summary.get("target_indices")
        values = summary.get("values")
        if isinstance(target, Mapping):
            summary["target_count"] = target.get("count", 0)
            summary["target_indices_sha256"] = target.get("sha256")
            summary.pop("target_indices", None)
        if isinstance(values, Mapping):
            summary["value_count"] = values.get("count", 0)
            summary["values_sha256"] = values.get("sha256")
            summary["values_min"] = values.get("min")
            summary["values_max"] = values.get("max")
            summary["values_sum"] = values.get("sum")
            summary.pop("values", None)
        return summary
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        return _array_summary(value)
    return _json_safe(value)


@dataclass(frozen=True)
class BackendCapability:
    """Capability discovery result; discovery never imports optional modules."""

    name: str
    installed: bool
    usable: bool
    validated: bool = False
    requires_gate: bool = False
    module: Optional[str] = None
    version: Optional[str] = None
    note: str = ""

    @property
    def available(self) -> bool:
        return self.installed and self.usable

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "installed": self.installed,
            "usable": self.usable,
            "available": self.available,
            "validated": self.validated,
            "requires_gate": self.requires_gate,
            "module": self.module,
            "version": self.version,
            "note": self.note,
        }


def _module_capability(name: str, *, requires_gate: bool = False,
                       note: str = "") -> BackendCapability:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    installed = spec is not None
    version: Optional[str] = None
    if installed:
        try:
            version = importlib.metadata.version(name)
        except Exception:
            # Package metadata is advisory; a module can be importable without
            # a conventional distribution record.
            version = None
    return BackendCapability(name, installed, installed and not requires_gate,
                             validated=False, requires_gate=requires_gate,
                             module=name if installed else None, version=version,
                             note=note)


def _import_gate() -> Any:
    """Load the read-only gate without importing any model backend."""

    try:
        return importlib.import_module("malecns_gate")
    except ModuleNotFoundError:
        path = SCRIPT_DIR / "malecns_gate.py"
        spec = importlib.util.spec_from_file_location("malecns_gate", path)
        if spec is None or spec.loader is None:
            raise EngineError(f"cannot load safety gate from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("malecns_gate", module)
        spec.loader.exec_module(module)
        return module


def _gate_blockers() -> tuple[str, ...]:
    gate = _import_gate()
    try:
        return tuple(str(x) for x in gate.cuda_blockers())
    except Exception as exc:  # pragma: no cover - unusual host fallback
        return (f"gate check failed: {type(exc).__name__}: {exc}",)


def _require_cuda_gate() -> None:
    blockers = _gate_blockers()
    if blockers:
        raise GateClosedError(blockers)
    # Re-assert immediately before importing Torch to reduce the discovery /
    # initialisation race and to use the project's canonical error text.
    try:
        _import_gate().require_cuda()
    except SystemExit as exc:
        # The standalone gate historically reports a closed gate with
        # ``SystemExit``.  The reusable engine exposes a normal typed error to
        # callers while retaining the gate's fail-closed behaviour.
        raise GateClosedError((str(exc),)) from exc


def _nvidia_smi_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def discover_capabilities() -> dict[str, BackendCapability]:
    """Discover CPU/Torch/CUDA/Triton options without importing them."""

    return {
        "numpy_cpu": BackendCapability(
            "numpy_cpu", True, True, validated=True,
            note="reference CPU engine; default and dependency-light",
        ),
        "torch_cpu": _module_capability(
            "torch", note="explicit CPU Torch only; not imported by default",
        ),
        "torch_cuda": _module_capability(
            "torch", requires_gate=True,
            note="Torch CUDA requires malecns_gate and a visible GPU",
        ),
        "triton": _module_capability(
            "triton", requires_gate=True,
            note="fused update/compact/deliver path; explicit CUDA gate required",
        ),
        "nvidia_smi": BackendCapability(
            "nvidia_smi", _nvidia_smi_present(), False,
            note="presence only; does not imply a safe CUDA context",
        ),
        "malecns_gate": BackendCapability(
            "malecns_gate", (SCRIPT_DIR / "malecns_gate.py").is_file(), True,
            validated=True, module="malecns_gate",
            note="read-only fail-closed gate checked immediately before CUDA Torch import",
        ),
    }


@dataclass
class CSRPreflight:
    """Read-only CSR shape/file report."""

    path: Path
    exists: bool
    valid: bool
    blockers: list[str] = field(default_factory=list)
    files: dict[str, bool] = field(default_factory=dict)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dtypes: dict[str, str] = field(default_factory=dict)
    neurons: Optional[int] = None
    edges: Optional[int] = None
    synapses: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "valid": self.valid,
            "blockers": list(self.blockers),
            "files": dict(self.files),
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "dtypes": dict(self.dtypes),
            "neurons": self.neurons,
            "edges": self.edges,
            "synapses": self.synapses,
            "metadata": _json_safe(self.metadata),
        }


def preflight_csr(path: str | Path = DEFAULT_CSR) -> CSRPreflight:
    """Inspect CSR files and headers only; never execute a neural step."""

    root = Path(path)
    required = ("ids.npy", "offsets.npy", "target.npy", "weight.npy", "sign.npy",
                "annotations.npz", "tables.json")
    files = {name: (root / name).is_file() for name in required}
    report = CSRPreflight(root, root.is_dir(), False, files=files)
    if not root.is_dir():
        report.blockers.append(f"CSR directory missing: {root}")
        return report
    for name in required:
        if not files[name]:
            report.blockers.append(f"missing CSR file: {root / name}")
    if report.blockers:
        return report

    arrays: dict[str, Any] = {}
    try:
        for name in ("ids", "offsets", "target", "weight", "sign"):
            arr = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            arrays[name] = arr
            report.shapes[name] = tuple(int(x) for x in arr.shape)
            report.dtypes[name] = str(arr.dtype)
        # NPZ metadata is small (~5 MiB compressed) and required to validate
        # annotation lengths; no graph arithmetic is performed here.
        with np.load(root / "annotations.npz", allow_pickle=False) as ann:
            ann_shapes = {key: tuple(int(x) for x in ann[key].shape) for key in ann.files}
        report.shapes["annotations"] = (len(ann_shapes),)
        for key, shape in ann_shapes.items():
            report.shapes[f"annotations.{key}"] = shape
        required_ann = {"typeIdx", "superclassIdx", "classIdx", "subclassIdx", "ntIdx",
                        "sideIdx", "neuromereIdx", "nerveIdx"}
        for key in sorted(required_ann - set(ann_shapes)):
            report.blockers.append(f"annotations.npz missing field: {key}")
        for key in sorted(required_ann & set(ann_shapes)):
            if ann_shapes[key] != tuple(int(x) for x in arrays["ids"].shape):
                report.blockers.append(f"annotation {key} shape {ann_shapes[key]} != ids shape {arrays['ids'].shape}")
        with (root / "tables.json").open(encoding="utf-8") as handle:
            tables = json.load(handle)
        report.metadata["table_keys"] = sorted(tables) if isinstance(tables, Mapping) else []
        meta_path = root / "meta.json"
        if meta_path.is_file():
            with meta_path.open(encoding="utf-8") as handle:
                report.metadata["meta"] = json.load(handle)
    except Exception as exc:
        report.blockers.append(f"cannot read CSR headers: {type(exc).__name__}: {exc}")
        return report

    ids, offsets, target, weight, sign = (arrays[k] for k in ("ids", "offsets", "target", "weight", "sign"))
    if ids.ndim != 1:
        report.blockers.append("ids must be rank 1")
    if offsets.ndim != 1 or offsets.shape[0] != ids.shape[0] + 1:
        report.blockers.append("offsets must have length neurons + 1")
    if target.ndim != 1 or weight.ndim != 1 or sign.ndim != 1:
        report.blockers.append("target, weight and sign must be rank 1")
    if target.shape != weight.shape:
        report.blockers.append("target and weight shapes differ")
    if sign.shape != ids.shape:
        report.blockers.append("sign shape differs from ids")
    if offsets.size and (int(offsets[0]) != 0 or int(offsets[-1]) != target.shape[0]):
        report.blockers.append("offsets do not span the target array")
    if offsets.size > 1 and np.any(np.diff(offsets) < 0):
        report.blockers.append("offsets are not monotonic")
    if target.size and (int(target.min()) < 0 or int(target.max()) >= ids.shape[0]):
        report.blockers.append("target contains an out-of-range neuron index")
    report.neurons = int(ids.shape[0])
    report.edges = int(target.shape[0])
    meta = report.metadata.get("meta", {})
    if isinstance(meta, Mapping):
        nested = meta.get("meta")
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except json.JSONDecodeError:
                nested = None
        payload = nested if isinstance(nested, Mapping) else meta
        report.synapses = _int_or_none(payload.get("synapses_in_edges"))
        declared_n = _int_or_none(meta.get("n", payload.get("neurons")))
        declared_e = _int_or_none(meta.get("edges", payload.get("edges")))
        if declared_n is not None and declared_n != report.neurons:
            report.blockers.append(f"meta n={declared_n} != ids length {report.neurons}")
        if declared_e is not None and declared_e != report.edges:
            report.blockers.append(f"meta edges={declared_e} != target length {report.edges}")
    report.valid = report.exists and not report.blockers
    return report


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class CSRGraph:
    path: Path
    ids: np.ndarray
    offsets: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    sign: np.ndarray
    tables: dict[str, np.ndarray]
    annotations: dict[str, np.ndarray]
    meta: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.ids.size)

    @property
    def e(self) -> int:
        return int(self.target.size)


def _load_graph(path: Path) -> CSRGraph:
    report = preflight_csr(path)
    if not report.valid:
        raise CSRValidationError("invalid CSR:\n  - " + "\n  - ".join(report.blockers))
    try:
        arrays = {name: np.asarray(np.load(path / f"{name}.npy", allow_pickle=False))
                  for name in ("ids", "offsets", "target", "weight", "sign")}
        with (path / "tables.json").open(encoding="utf-8") as handle:
            raw_tables = json.load(handle)
        if not isinstance(raw_tables, Mapping):
            raise TypeError("tables.json must contain a mapping")
        tables = {str(key): np.asarray(value) for key, value in raw_tables.items()}
        with np.load(path / "annotations.npz", allow_pickle=False) as ann:
            annotations = {key: np.asarray(ann[key]) for key in ann.files}
        meta_path = path / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    except Exception as exc:
        raise CSRValidationError(f"failed to load CSR {path}: {type(exc).__name__}: {exc}") from exc
    required_ann = {"typeIdx", "superclassIdx", "classIdx", "subclassIdx", "ntIdx",
                    "sideIdx", "neuromereIdx", "nerveIdx"}
    missing = sorted(required_ann - set(annotations))
    if missing:
        raise CSRValidationError(f"annotations.npz missing fields: {missing}")
    return CSRGraph(path, arrays["ids"], arrays["offsets"], arrays["target"],
                    arrays["weight"], arrays["sign"], tables, annotations, meta)


class PopulationIndex:
    """MaleCNS population grammar copied from the validated FLYB resolver."""

    _TABLE_FOR_FIELD = {
        "typeIdx": "types", "superclassIdx": "superclasses", "classIdx": "classes",
        "subclassIdx": "subclasses", "ntIdx": "nts", "sideIdx": "sides",
        "neuromereIdx": "neuromeres", "nerveIdx": "nerves",
    }

    def __init__(self, tables: Mapping[str, np.ndarray], annotations: Mapping[str, np.ndarray],
                 ids: np.ndarray) -> None:
        self.tables = {key: np.asarray(value) for key, value in tables.items()}
        self.annotations = dict(annotations)
        self.ids = np.asarray(ids)
        self.n = int(self.ids.size)
        self._cache: dict[str, np.ndarray] = {}
        for field, table in self._TABLE_FOR_FIELD.items():
            if field not in self.annotations or table not in self.tables:
                raise CSRValidationError(f"population metadata missing {field}/{table}")

    def _eq(self, codes: np.ndarray, table: str, needle: str) -> np.ndarray:
        values = self.tables[table].astype(str)
        matches = np.flatnonzero(values == needle)
        if matches.size == 0:
            raise PopulationError(f"unknown {table[:-1]} label {needle!r}")
        return np.flatnonzero(np.isin(codes, matches))

    def resolve(self, spec: str, *, strict: bool = True) -> np.ndarray:
        if not isinstance(spec, str) or not spec.strip():
            raise PopulationError("population spec must be a non-empty string")
        spec = spec.strip()
        if spec in self._cache:
            return self._cache[spec].copy()
        parts: list[np.ndarray] = []
        for raw, term, side in comma_terms(spec):
            if side is not None:
                if side not in {"L", "R", "M", "l", "r", "m"}:
                    raise PopulationError(f"unsupported side suffix in {raw!r}")
                side = side.upper()
            ids = self._term(term)
            if side is not None:
                ids = ids[self.tables["sides"][self.annotations["sideIdx"][ids]].astype(str) == side]
            parts.append(ids)
        out = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        if strict and out.size == 0:
            raise PopulationError(f"population spec resolved to zero neurons: {spec!r}")
        self._cache[spec] = out.copy()
        return out

    def _term(self, term: str) -> np.ndarray:
        if "&" in term:
            return intersect_terms(term, self._simple)
        return self._simple(term)

    def _simple(self, term: str) -> np.ndarray:
        if not term:
            raise PopulationError("empty population term")
        if ":" not in term:
            if term.lower() == "all":
                return np.arange(self.n, dtype=np.int64)
            return self._eq(self.annotations["typeIdx"], "types", term)
        kind, arg = (part.strip() for part in term.split(":", 1))
        kind = kind.lower()
        if kind == "body":
            try:
                out = np.flatnonzero(self.ids == int(arg))
            except ValueError as exc:
                raise PopulationError(f"body selector is not an integer: {arg!r}") from exc
            if out.size == 0:
                raise PopulationError(f"unknown body ID {arg!r}")
            return out
        field = {
            "type": "typeIdx", "class": "classIdx", "superclass": "superclassIdx",
            "subclass": "subclassIdx", "nt": "ntIdx", "nerve": "nerveIdx",
            "neuromere": "neuromereIdx",
        }.get(kind)
        if field is not None:
            return self._eq(self.annotations[field], self._TABLE_FOR_FIELD[field], arg)
        if kind in {"prefix", "contains"}:
            types = self.tables["types"][self.annotations["typeIdx"]].astype(str)
            if kind == "prefix":
                out = np.flatnonzero(np.char.startswith(types, arg))
            else:
                out = np.flatnonzero(np.char.find(types, arg) >= 0)
            if out.size == 0:
                raise PopulationError(f"{kind} selector matched zero neurons: {arg!r}")
            return out
        raise PopulationError(f"unsupported population term: {term!r}")


# Compatibility vocabulary with the validated command-line port.
Populations = PopulationIndex


def _as_float_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise DriveError(f"{name} is not numeric") from exc
    if not np.all(np.isfinite(array)):
        raise DriveError(f"{name} contains NaN or infinite values")
    return array


def _scalar(value: Any, name: str) -> float:
    arr = _as_float_array(value, name)
    if arr.ndim != 0:
        raise DriveError(f"{name} must be a scalar")
    return float(arr)


class MaleCNSEngine:
    """Reusable MaleCNS LIF engine with NumPy and guarded Torch backends."""

    @staticmethod
    def discover_capabilities() -> dict[str, BackendCapability]:
        return discover_capabilities()

    capability_discovery = discover_capabilities
    preflight_csr = staticmethod(preflight_csr)

    def __init__(self, csr_path: str | Path = DEFAULT_CSR, *,
                 backend: str = "cpu", device: str = "cpu", dt_ms: float = DEFAULT_DT_MS,
                 gain: float = DEFAULT_GAIN, kc_gain: float = DEFAULT_KC_GAIN,
                 max_vram_gib: Optional[float] = DEFAULT_MAX_VRAM_GIB,
                 seed: int = 20260903, poisson_hz: Any = None,
                 current_mV_per_ms: Any = None, poisson_drive: Any = None,
                 current_drive: Any = None, flyvis_drive: Any = None,
                 flyvis_targets: Optional[Mapping[Any, Any]] = None,
                 flyvis_units: Optional[str] = None,
                 sensory_map_path: str | Path = DEFAULT_SENSORY_MAP,
                 mechanosensory_map_path: Optional[str | Path] = None,
                 report_populations: Optional[Mapping[str, str] | Sequence[str]] = None,
                 stp: bool = False,
                 cuda_graph: bool = False,
                 include_flyvis_drive_echo: bool = True,
                 deterministic_delivery: bool = False) -> None:
        self.csr_path = Path(csr_path)
        self.backend_requested = str(backend).lower()
        self.device_requested = str(device).lower()
        if self.device_requested == "gpu":
            self.device_requested = "cuda"
        if self.backend_requested == "numpy":
            self.backend_requested = "cpu"
        if self.backend_requested in {"numpy_cpu", "numpy-cpu"}:
            self.backend_requested = "cpu"
        if self.backend_requested in {"torch-cpu", "torch_cpu"}:
            self.backend_requested = "torch_cpu"
        if self.backend_requested in {"torch-cuda", "torch_cuda"}:
            self.backend_requested = "torch"
            if self.device_requested == "auto":
                self.device_requested = "cuda"
        if self.device_requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if self.backend_requested not in {"auto", "cpu", "torch", "torch_cpu", "triton"}:
            raise ValueError("backend must be auto, cpu, torch_cpu, torch, or triton")
        if stp:
            raise UnsupportedBackendError(
                "Tsodyks-Markram STP is not implemented in MaleCNSEngine; use lif_flyb.py"
            )
        if not isinstance(cuda_graph, (bool, np.bool_)):
            raise ValueError("cuda_graph must be a boolean")
        if not isinstance(include_flyvis_drive_echo, (bool, np.bool_)):
            raise ValueError("include_flyvis_drive_echo must be a boolean")
        if not isinstance(deterministic_delivery, (bool, np.bool_)):
            raise ValueError("deterministic_delivery must be a boolean")
        self.deterministic_delivery_requested = bool(deterministic_delivery)
        self.deterministic_delivery = False
        self._triton_fixedpoint_scale_power: Optional[int] = None
        self._triton_fixedpoint_abs_tick_bound: Optional[int] = None
        # CUDA Graph replay is intentionally opt-in.  The default path keeps
        # the existing uncaptured Triton launch sequence and its exact timing
        # counters; a requested graph that is unavailable falls back with an
        # explicit reason in the manifest/output telemetry.
        self.cuda_graph_requested = bool(cuda_graph)
        # Dense FlyVis values remain part of the default output contract for
        # compatibility. Production minimal telemetry can disable this echo
        # while retaining bounded mapping/hash/statistics metadata.
        self.include_flyvis_drive_echo = bool(include_flyvis_drive_echo)
        self.dt_ms = float(dt_ms)
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0:
            raise ValueError("dt_ms must be finite and > 0")
        self.gain = _scalar(gain, "gain")
        self.kc_gain = _scalar(kc_gain, "kc_gain")
        if max_vram_gib is None:
            self.max_vram_gib = None
        else:
            self.max_vram_gib = float(max_vram_gib)
            if not math.isfinite(self.max_vram_gib) or self.max_vram_gib <= 0:
                raise ValueError("max_vram_gib must be a positive finite number or None")
        self.seed = int(seed)
        self.flyvis_targets = dict(flyvis_targets or {})
        self.flyvis_units = None if flyvis_units is None else str(flyvis_units)
        if self.flyvis_units not in {None, "poisson_hz", "current_mV_per_ms"}:
            raise ValueError("flyvis_units must be poisson_hz or current_mV_per_ms")
        # A fitted FlyVis target ordering is immutable for an entire run.  We
        # still fully validate ordinary/writable caller arrays on every use,
        # but an owned read-only ndarray can be validated once and then reused
        # by identity.  This removes an O(n log n) ``np.unique`` from every
        # 5 ms frame without weakening the public mutable-input contract.
        self._validated_flyvis_indices_source: Optional[np.ndarray] = None
        self._validated_flyvis_indices: Optional[np.ndarray] = None

        if mechanosensory_map_path is not None:
            sensory_map_path = mechanosensory_map_path
        self.sensory_map_path = Path(sensory_map_path)
        (self.sensory_map, self.mechanosensory_specs,
         self.sensory_map_hash) = _load_mechanosensory_map(self.sensory_map_path)
        self.mechanosensory_max_gain_mV_per_ms = _scalar(
            self.sensory_map["drive_mapping"]["max_gain_mV_per_ms"],
            "mechanosensory max_gain_mV_per_ms",
        )

        self.capabilities = discover_capabilities()
        self.device, self.backend, self._torch, self._triton = self._select_backend()
        if self.deterministic_delivery_requested and self.backend != "triton":
            raise UnsupportedBackendError(
                "deterministic_delivery=True requires backend='triton'"
            )
        self.deterministic_delivery = self.deterministic_delivery_requested
        self.graph = _load_graph(self.csr_path)
        self.populations = PopulationIndex(self.graph.tables, self.graph.annotations, self.graph.ids)
        self.n = self.graph.n
        self.e = self.graph.e
        self._build_weights()
        if self.deterministic_delivery:
            self._configure_triton_fixedpoint_delivery()
        self._mechanosensory_targets: dict[str, np.ndarray] = {}
        self._mechanosensory_target_status: dict[str, str] = {}
        self._mechanosensory_overlap_plans: dict[
            tuple[str, ...], tuple[int, tuple[np.ndarray, ...]]
        ] = {}
        self._last_mechanosensory_mapping: dict[str, Any] = {
            "input_units": MECHANOSENSORY_INPUT_UNITS,
            "output_units": MECHANOSENSORY_OUTPUT_UNITS,
            "applied": {},
            "shadowed": {},
            "telemetry_ignored": [],
        }
        # Resolve annotation targets up front when this CSR contains them so
        # manifests expose coverage.  A micro-CSR or another reduced graph may
        # omit one of these populations; that remains usable for unrelated
        # drives, but an explicit drive for the missing target will fail closed
        # at the frame boundary.
        for spec in self.mechanosensory_specs:
            try:
                self._resolve_mechanosensory_target(spec)
            except DriveError:
                self._mechanosensory_target_status.setdefault(spec.name, "unresolved")

        self.report_populations: dict[str, np.ndarray] = {}
        if report_populations is not None:
            if isinstance(report_populations, str):
                items = ((report_populations, report_populations),)
            elif isinstance(report_populations, Mapping):
                items = report_populations.items()
            else:
                items = (
                    (str(spec), str(spec)) for spec in report_populations
                )
            for name, spec in items:
                self.report_populations[str(name)] = self.populations.resolve(str(spec))
        self._report_population_names = tuple(self.report_populations)
        self._report_population_sizes = np.asarray(
            [self.report_populations[name].size for name in self._report_population_names],
            dtype=np.int64,
        )
        self._report_population_flat_indices: Optional[np.ndarray] = None
        self._report_population_offsets: Optional[np.ndarray] = None
        if self._report_population_names and np.all(self._report_population_sizes > 0):
            self._report_population_flat_indices = np.concatenate([
                self.report_populations[name] for name in self._report_population_names
            ])
            self._report_population_offsets = np.cumsum(
                np.concatenate((np.array([0], dtype=np.int64),
                                self._report_population_sizes[:-1]))
            )

        self._poisson_hz = np.zeros(self.n, dtype=np.float32)
        self._current_mV_per_ms = np.zeros(self.n, dtype=np.float32)
        if poisson_hz is not None:
            self.set_poisson_drive(poisson_hz)
        if poisson_drive is not None:
            self.set_poisson_drive(poisson_drive)
        if current_mV_per_ms is not None:
            self.set_current_drive(current_mV_per_ms)
        if current_drive is not None:
            self.set_current_drive(current_drive)
        if flyvis_drive is not None:
            self.set_flyvis_drive(flyvis_drive)
        self._step_index = 0
        self._last_timestamp: Optional[Timestamp] = None
        self._rng = np.random.default_rng(self.seed)
        self._torch_generator: Any = None
        self._allocate_state()
        self._last_named_outputs: dict[str, Any] = {}
        self._triton_warmup_active = False
        self._triton_warmup_durations: list[float] = []
        self._triton_warmup_report: Optional[dict[str, Any]] = None
        self._triton_recorded_steps = 0
        self._triton_recorded_first_wall_s: Optional[float] = None
        self._triton_recorded_steady_total_s = 0.0
        self._triton_recorded_steady_steps = 0
        self._triton_last_wall_s: Optional[float] = None
        # Lightweight launch/synchronisation counters make the GPU timing
        # contract auditable without synchronising the device just to collect
        # telemetry.  Warmup/reset clear the recorded-run totals.
        self._triton_kernel_launches = 0
        self._triton_host_device_syncs = 0
        self._triton_host_scalar_reads = 0
        self._triton_input_h2d_copies = 0
        self._triton_last_kernel_launches = 0
        self._triton_last_host_device_syncs = 0
        self._triton_last_host_scalar_reads = 0
        self._triton_last_window_counter_resets = 0
        self._triton_last_graph_host_launches = 0
        self._triton_last_graph_kernel_nodes = 0
        self._triton_last_graph_pointer_updates = 0
        self._triton_cuda_graph_pointer_updates = 0
        self._triton_cuda_graph_enabled = False
        self._triton_cuda_graph_reason: Optional[str] = None
        self._triton_cuda_graphs: dict[int, dict[str, Any]] = {}
        self._triton_cuda_graph_capture_count = 0
        self._triton_cuda_graph_replay_count = 0
        self._triton_cuda_graph_host_launches = 0
        self._triton_cuda_graph_step_shadow: Optional[int] = None
        self._triton_cuda_graph_seed_shadow: Optional[int] = None
        self._triton_cuda_graph_last_phase = "disabled"
        self._triton_cuda_graph_last_capture_wall_s: Optional[float] = None
        self._triton_cuda_graph_last_replay_wall_s: Optional[float] = None
        if self.cuda_graph_requested:
            if self.backend != "triton" or self.device != "cuda":
                self._triton_cuda_graph_reason = (
                    "CUDA Graph capture requires backend='triton' on device='cuda'"
                )
            elif not hasattr(self._torch.cuda, "CUDAGraph") or not hasattr(self._torch.cuda, "graph"):
                self._triton_cuda_graph_reason = (
                    "this Torch build does not expose torch.cuda.CUDAGraph/graph"
                )
            else:
                self._triton_cuda_graph_enabled = True
                self._triton_cuda_graph_reason = None

    # ---------------------------- setup and backend safety
    def _select_backend(self) -> tuple[str, str, Any, Any]:
        requested = self.backend_requested
        device = self.device_requested
        if requested == "auto":
            if device == "cuda":
                # An explicit CUDA device is never downgraded to CPU.  The
                # gate is checked before importing Torch below.
                requested, device = "torch", "cuda"
            elif device == "auto":
                # Auto is conservative: only a cleared safety gate and a
                # usable Torch CUDA device may select the CUDA path.  A
                # missing GPU, missing Torch, or a closed gate is an honest
                # CPU selection and does not import Torch.
                blockers = _gate_blockers()
                if blockers:
                    requested, device = "cpu", "cpu"
                elif not self.capabilities["torch_cuda"].installed:
                    requested, device = "cpu", "cpu"
                else:
                    try:
                        _require_cuda_gate()
                    except GateClosedError:
                        requested, device = "cpu", "cpu"
                    else:
                        try:
                            torch = importlib.import_module("torch")
                        except Exception:
                            # Auto capability discovery must not make a CPU run
                            # fail just because an optional Torch install is
                            # broken.  Explicit CUDA below remains fail-closed.
                            requested, device = "cpu", "cpu"
                        else:
                            try:
                                has_cuda = bool(torch.cuda.is_available())
                            except Exception:
                                has_cuda = False
                            if has_cuda:
                                self._configure_cuda_allocator(torch)
                                return "cuda", "torch_cuda", torch, None
                            requested, device = "cpu", "cpu"
            else:
                requested, device = "cpu", "cpu"
        elif requested == "triton":
            # Triton kernels are CUDA-only.  Keep the gate immediately before
            # both optional imports; an explicit CPU request is rejected
            # without touching either CUDA runtime.
            if device == "auto":
                device = "cuda"
            if device != "cuda":
                raise UnsupportedBackendError("Triton backend requires device='cuda'")
            _require_cuda_gate()
            try:
                torch = importlib.import_module("torch")
                triton = importlib.import_module("triton")
                importlib.import_module("triton.language")
            except Exception as exc:
                raise UnsupportedBackendError(
                    "Triton backend requested but Torch/Triton could not be imported"
                ) from exc
            if not bool(torch.cuda.is_available()):
                raise UnsupportedBackendError(
                    "Triton backend requested but torch.cuda.is_available() is false"
                )
            self._configure_cuda_allocator(torch)
            return "cuda", "triton", torch, triton
        elif requested == "torch_cpu":
            if device == "cuda":
                raise UnsupportedBackendError("torch_cpu cannot use device='cuda'")
            device = "cpu"
        elif requested == "torch":
            if device == "auto":
                # Naming Torch is an explicit CPU-Torch opt-in when no device
                # was selected; default construction remains NumPy CPU.
                device = "cpu"
        elif requested == "cpu":
            if device == "cuda":
                raise UnsupportedBackendError("NumPy CPU backend cannot use device='cuda'")
            device = "cpu"

        if requested == "cpu":
            return "cpu", "numpy_cpu", None, None
        if requested in {"torch", "torch_cpu"}:
            if device == "cuda":
                _require_cuda_gate()
            try:
                torch = importlib.import_module("torch")
            except Exception as exc:
                raise UnsupportedBackendError(
                    "Torch backend requested but Torch could not be imported"
                ) from exc
            if device == "cuda" and not bool(torch.cuda.is_available()):
                raise UnsupportedBackendError("Torch is installed but torch.cuda.is_available() is false")
            if device == "cuda":
                self._configure_cuda_allocator(torch)
            return device, "torch_cuda" if device == "cuda" else "torch_cpu", torch, None
        raise AssertionError(f"unhandled backend {requested!r}")

    def _configure_cuda_allocator(self, torch: Any) -> None:
        """Apply the per-process PyTorch VRAM ceiling before graph allocation."""

        if self.max_vram_gib is None:
            return
        cuda_device = torch.device("cuda:0")
        total = int(torch.cuda.get_device_properties(cuda_device).total_memory)
        cap_bytes = int(self.max_vram_gib * (1024 ** 3))
        fraction = min(1.0, cap_bytes / max(1, total))
        torch.cuda.set_per_process_memory_fraction(fraction, cuda_device)

    def _build_weights(self) -> None:
        offsets = self.graph.offsets.astype(np.int64, copy=False)
        src = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(offsets))
        post_gain = np.ones(self.n, dtype=np.float32)
        if "class:Kenyon_Cell" in self._population_candidates():
            try:
                kc = self.populations.resolve("class:Kenyon_Cell")
            except PopulationError:
                # A micro-CSR may retain the vocabulary label without
                # containing any Kenyon neurons; the optional post-gain then
                # has no effect and is not a malformed graph.
                kc = np.empty(0, dtype=np.int64)
            post_gain[kc] = self.kc_gain
        w_syn = float(self.graph.meta.get("wSynMv", W_SYN_MV)) if isinstance(self.graph.meta, Mapping) else W_SYN_MV
        self._weight = (self.graph.sign[src].astype(np.float32)
                        * (w_syn * self.gain)
                        * self.graph.weight.astype(np.float32)
                        * post_gain[self.graph.target])

    def _configure_triton_fixedpoint_delivery(self) -> None:
        """Validate an exact fixed-point encoding of the stored float32 weights.

        The kernel scales each already-built float32 edge weight by the same
        power of two, so this path does not alter signs, gains, or edge values.
        An int64 sum is exact and order-independent; construction fails closed
        unless every stored edge weight round-trips and the conservative
        all-presynaptic-neurons-spike bound fits in signed int64.
        """

        weights = self._weight
        if weights.dtype != np.float32 or not weights.flags.c_contiguous:
            raise UnsupportedBackendError(
                "deterministic Triton delivery requires contiguous float32 edge weights"
            )
        bits = weights.view(np.uint32)
        chunk_size = 1 << 20
        minimum_binary_exponent: Optional[int] = None
        for start in range(0, weights.size, chunk_size):
            stop = min(weights.size, start + chunk_size)
            chunk = weights[start:stop]
            raw = bits[start:stop]
            exponent = (raw >> np.uint32(23)) & np.uint32(0xff)
            if np.any(exponent == np.uint32(0xff)):
                raise UnsupportedBackendError(
                    "deterministic Triton delivery requires finite float32 edge weights"
                )
            fraction = raw & np.uint32(0x7fffff)
            significand = np.where(
                exponent == 0,
                fraction,
                fraction | np.uint32(0x800000),
            ).astype(np.uint32, copy=False)
            nonzero = significand != 0
            if not np.any(nonzero):
                continue
            # Remove trailing zero bits from each significand to obtain its
            # lowest exact power-of-two exponent without per-edge Python work.
            safe_significand = np.where(nonzero, significand, np.uint32(1))
            lowest_set_bit = safe_significand & (~safe_significand + np.uint32(1))
            trailing_zeroes = (
                np.frexp(lowest_set_bit.astype(np.float64))[1] - 1
            ).astype(np.int32, copy=False)
            base_exponent = np.where(
                exponent == 0,
                -149,
                exponent.astype(np.int32) - 150,
            ) + trailing_zeroes
            chunk_minimum = int(np.min(base_exponent, where=nonzero, initial=1000))
            if minimum_binary_exponent is None or chunk_minimum < minimum_binary_exponent:
                minimum_binary_exponent = chunk_minimum

        scale_power = max(0, -minimum_binary_exponent) if minimum_binary_exponent is not None else 0
        if scale_power > 149:
            raise UnsupportedBackendError(
                "deterministic Triton delivery weight exponent exceeds float32 range"
            )
        int64_limit = 1 << 63
        # Verify each edge in bounded chunks so the opt-in path does not keep
        # an extra E-sized int64 weight array in host memory or VRAM.
        for start in range(0, weights.size, chunk_size):
            stop = min(weights.size, start + chunk_size)
            chunk = weights[start:stop]
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                scaled = np.ldexp(chunk.astype(np.float64), scale_power)
            if (not np.all(np.isfinite(scaled))
                    or np.any(scaled != np.rint(scaled))
                    or np.any(np.abs(scaled) >= float(int64_limit))):
                raise UnsupportedBackendError(
                    "deterministic Triton delivery weights do not fit exact int64 fixed point"
                )
            fixed = scaled.astype(np.int64)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                restored = np.ldexp(fixed.astype(np.float64), -scale_power).astype(np.float32)
            if not np.array_equal(restored, chunk):
                raise UnsupportedBackendError(
                    "deterministic Triton delivery fixed-point round-trip changed a weight"
                )

        max_weight = max(
            abs(float(np.min(weights))) if weights.size else 0.0,
            abs(float(np.max(weights))) if weights.size else 0.0,
        )
        try:
            max_fixed_weight = int(math.ldexp(max_weight, scale_power))
        except (OverflowError, ValueError) as exc:
            raise UnsupportedBackendError(
                "deterministic Triton delivery weight bound exceeds int64"
            ) from exc
        if max_fixed_weight >= int64_limit:
            raise UnsupportedBackendError(
                "deterministic Triton delivery edge weight exceeds int64"
            )
        indegrees = np.bincount(self.graph.target, minlength=self.n)
        max_indegree = int(indegrees.max()) if indegrees.size else 0
        abs_tick_bound = max_fixed_weight * max_indegree
        if abs_tick_bound >= int64_limit:
            raise UnsupportedBackendError(
                "deterministic Triton delivery worst-case absolute incoming sum "
                "does not fit signed int64"
            )
        self._triton_fixedpoint_scale_power = int(scale_power)
        self._triton_fixedpoint_abs_tick_bound = int(abs_tick_bound)

    def _population_candidates(self) -> set[str]:
        classes = self.graph.tables.get("classes", np.asarray([], dtype=str)).astype(str)
        return {f"class:{value}" for value in classes}

    def _allocate_state(self) -> None:
        self.decay = float(np.exp(-self.dt_ms / TAU_M_MS))
        self.syn_decay = float(np.exp(-self.dt_ms / TAU_S_MS))
        if abs(TAU_M_MS - TAU_S_MS) < 1e-9:
            self.syn_coupling = (self.dt_ms / TAU_M_MS) * np.exp(-self.dt_ms / TAU_M_MS)
        else:
            self.syn_coupling = ((TAU_S_MS / (TAU_M_MS - TAU_S_MS))
                                 * (np.exp(-self.dt_ms / TAU_M_MS)
                                    - np.exp(-self.dt_ms / TAU_S_MS)))
        self.delay_steps = max(1, int(round(DELAY_MS / self.dt_ms)))
        self.ref_steps = max(1, int(round(REFRACTORY_MS / self.dt_ms)))
        self.slots = self.delay_steps + 1
        if self.backend == "numpy_cpu":
            self.v = np.full(self.n, V_REST_MV, dtype=np.float32)
            self.g = np.zeros(self.n, dtype=np.float32)
            self.ref_until = np.zeros(self.n, dtype=np.int32)
            self.ring = np.zeros((self.slots, self.n), dtype=np.float32)
            self.spike_tick = np.zeros(self.n, dtype=np.int32)
        else:
            dev = "cuda:0" if self.device == "cuda" else "cpu"
            self._torch_generator = self._torch.Generator(device=dev)
            self._torch_generator.manual_seed(self.seed)
            self.v = self._torch.full((self.n,), V_REST_MV, device=dev)
            self.g = self._torch.zeros(self.n, device=dev)
            self.ref_until = self._torch.zeros(self.n, dtype=self._torch.int32, device=dev)
            self.ring = self._torch.zeros((self.slots, self.n), device=dev)
            self.spike_tick = self._torch.zeros(self.n, dtype=self._torch.int32, device=dev)
            self.spk = self._torch.zeros(self.n, dtype=self._torch.uint8, device=dev)
            self._offsets_t = self._torch.as_tensor(self.graph.offsets.astype(np.int64), device=dev)
            self._target_t = self._torch.as_tensor(self.graph.target.astype(np.int64), device=dev)
            self._weight_t = self._torch.as_tensor(self._weight, device=dev)
            if self.backend == "triton":
                self._init_triton_kernels()

    def _init_triton_kernels(self) -> None:
        """Create the validated fused Triton update/delivery kernels.

        The kernels mirror the eager Torch order in :meth:`_step_torch` and
        the reference implementation in ``lif_flyb.py``.  This method is only
        reached after ``_require_cuda_gate`` and a successful CUDA selection,
        so importing Triton cannot open an ungated CUDA context.
        """

        triton = self._triton
        tl = importlib.import_module("triton.language")
        cuda_libdevice = importlib.import_module(
            "triton.language.extra.cuda.libdevice"
        )
        n = self.n

        @triton.jit
        def neuron_step(v_ptr, g_ptr, ref_ptr, spk_ptr, ring_ptr, slot, n,
                        step, decay, syn_coupling, syn_decay, v_rest, v_thr,
                        v_reset, ref_steps, probability_ptr, seed, current_ptr,
                        dt_ms, compact_counter_ptr, delivery_counter_ptr,
                        step_ptr, seed_ptr, slots,
                        USE_DEVICE_STEP: tl.constexpr,
                        RESET_DELIVERY: tl.constexpr,
                        RECORD_WINDOW: tl.constexpr,
                        ATOMIC_RING_CONSUME: tl.constexpr,
                        BLOCK: tl.constexpr):
            # The next kernels consume both counters after this kernel has
            # completed.  Reset them here so a tick has three launches
            # (neuron/compact/deliver), instead of dispatching separate
            # zeroing kernels from Python on every tick.  Only program zero
            # performs the reset; all programs are globally complete before
            # Triton launches the dependent compact kernel.
            if tl.program_id(0) == 0:
                tl.store(compact_counter_ptr, 0)
                if RESET_DELIVERY:
                    tl.store(delivery_counter_ptr, 0)
            if USE_DEVICE_STEP:
                step = tl.load(step_ptr).to(tl.int32)
                slot = step % slots
                seed = tl.load(seed_ptr).to(tl.int32) + step
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            v = tl.load(v_ptr + offs, mask=mask, other=v_rest)
            g = tl.load(g_ptr + offs, mask=mask, other=0.0)
            ref = tl.load(ref_ptr + offs, mask=mask, other=0)
            probability = tl.load(probability_ptr + offs, mask=mask, other=0.0)
            drive = tl.rand(seed, offs) < probability
            v = tl.where(drive, v_reset, v)
            ref = tl.where(drive, step + ref_steps, ref)
            ring_address = ring_ptr + slot * n + offs
            if ATOMIC_RING_CONSUME:
                # Opt-in diagnostic path: atomically return and clear the
                # single-owner read slot in one operation, avoiding a
                # separate global load followed by a zero store.
                ring_value = tl.atomic_xchg(
                    ring_address, tl.full((BLOCK,), 0.0, tl.float32),
                    mask=mask, sem="acq_rel", scope="gpu",
                )
            else:
                ring_value = tl.load(ring_address, mask=mask, other=0.0)
                tl.store(ring_address, tl.zeros_like(ring_value), mask=mask)
            g = g + ring_value
            current = tl.load(current_ptr + offs, mask=mask, other=0.0) * dt_ms
            v_new = v_rest + (v - v_rest) * decay + g * syn_coupling + current
            frozen = ref > step
            v = tl.where(frozen, v_reset, v_new)
            g = tl.where(frozen, g, g * syn_decay)
            spiked = (~frozen) & (v >= v_thr)
            v = tl.where(spiked, v_reset, v)
            g = tl.where(spiked, 0.0, g)
            ref = tl.where(spiked, step + ref_steps, ref)
            all_spikes = spiked | drive
            tl.store(v_ptr + offs, v, mask=mask)
            tl.store(g_ptr + offs, g, mask=mask)
            tl.store(ref_ptr + offs, ref, mask=mask)
            tl.store(spk_ptr + offs, all_spikes.to(tl.uint8), mask=mask)

        @triton.jit
        def compact(spk_ptr, list_ptr, count_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            spikes = tl.load(spk_ptr + offs, mask=mask, other=0).to(tl.int32)
            exclusive = tl.cumsum(spikes, axis=0) - spikes
            base = tl.atomic_add(count_ptr, tl.sum(spikes, axis=0))
            tl.store(list_ptr + base + exclusive, offs, mask=mask & (spikes > 0))

        @triton.jit
        def deliver(list_ptr, count_ptr, offsets_ptr, target_ptr, weight_ptr,
                    ring_ptr, slot, delivery_count_ptr, spike_tick_ptr,
                    window_count_ptr, n, step_ptr, delay_steps, slots,
                    USE_DEVICE_STEP: tl.constexpr,
                    ADVANCE_STEP: tl.constexpr, RECORD_WINDOW: tl.constexpr,
                    BLOCK_SYN: tl.constexpr):
            if USE_DEVICE_STEP:
                step = tl.load(step_ptr).to(tl.int32)
                slot = (step + delay_steps) % slots
            program = tl.program_id(0)
            count = tl.load(count_ptr)
            while program < count:
                neuron = tl.load(list_ptr + program).to(tl.int64)
                if RECORD_WINDOW:
                    # A held window can accumulate its cumulative state once
                    # after all ticks.  Keep only the per-source window count
                    # atomic in this hot delivery loop.
                    tl.atomic_add(window_count_ptr + neuron, 1)
                else:
                    # Scalar ``step`` exposes cumulative state immediately;
                    # update only the compacted active source list.
                    tl.atomic_add(spike_tick_ptr + neuron, 1)
                start = tl.load(offsets_ptr + neuron).to(tl.int64)
                end = tl.load(offsets_ptr + neuron + 1).to(tl.int64)
                cursor = start
                while cursor < end:
                    edge = cursor + tl.arange(0, BLOCK_SYN).to(tl.int64)
                    edge_mask = edge < end
                    target = tl.load(target_ptr + edge, mask=edge_mask, other=0).to(tl.int64)
                    weight = tl.load(weight_ptr + edge, mask=edge_mask, other=0.0)
                    tl.atomic_add(ring_ptr + slot * n + target, weight, mask=edge_mask)
                    cursor += BLOCK_SYN
                tl.atomic_add(delivery_count_ptr, (end - start).to(tl.int64))
                program += tl.num_programs(0)
            if ADVANCE_STEP and tl.program_id(0) == 0:
                # The graph's next neuron kernel observes this update because
                # CUDA graph nodes execute in capture order.  The increment is
                # outside the row loop so empty-spike windows advance too.
                tl.atomic_add(step_ptr, 1)

        @triton.jit
        def deliver_fixed(list_ptr, count_ptr, offsets_ptr, target_ptr,
                          weight_ptr, fixed_accum_ptr, delivery_count_ptr,
                          spike_tick_ptr, window_count_ptr, n,
                          FIXED_SCALE_POWER: tl.constexpr,
                          RECORD_WINDOW: tl.constexpr,
                          BLOCK_SYN: tl.constexpr):
            program = tl.program_id(0)
            count = tl.load(count_ptr)
            while program < count:
                neuron = tl.load(list_ptr + program).to(tl.int64)
                if RECORD_WINDOW:
                    tl.atomic_add(window_count_ptr + neuron, 1)
                else:
                    tl.atomic_add(spike_tick_ptr + neuron, 1)
                start = tl.load(offsets_ptr + neuron).to(tl.int64)
                end = tl.load(offsets_ptr + neuron + 1).to(tl.int64)
                cursor = start
                while cursor < end:
                    edge = cursor + tl.arange(0, BLOCK_SYN).to(tl.int64)
                    edge_mask = edge < end
                    target = tl.load(target_ptr + edge, mask=edge_mask, other=0).to(tl.int64)
                    weight = tl.load(weight_ptr + edge, mask=edge_mask, other=0.0)
                    fixed_weight = cuda_libdevice.ldexp(
                        weight, FIXED_SCALE_POWER,
                    ).to(tl.int64)
                    tl.atomic_add(
                        fixed_accum_ptr + target, fixed_weight, mask=edge_mask,
                    )
                    cursor += BLOCK_SYN
                tl.atomic_add(delivery_count_ptr, (end - start).to(tl.int64))
                program += tl.num_programs(0)

        @triton.jit
        def finalize_fixed_delivery(accum_ptr, ring_ptr, slot, n, step_ptr,
                                    delay_steps, slots,
                                    FIXED_SCALE_POWER: tl.constexpr,
                                    USE_DEVICE_STEP: tl.constexpr,
                                    BLOCK: tl.constexpr):
            if USE_DEVICE_STEP:
                step = tl.load(step_ptr).to(tl.int32)
                slot = (step + delay_steps) % slots
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            fixed_sum = tl.load(accum_ptr + offs, mask=mask, other=0)
            # int64->float32 is the one rounding point for a target/tick sum;
            # ldexp applies the exact inverse binary scale without changing
            # the selected float32 rounding rule.
            delivered = cuda_libdevice.ldexp(
                fixed_sum.to(tl.float32), -FIXED_SCALE_POWER,
            )
            # With slots=delay_steps+1, the write slot is distinct from the
            # current read slot and was cleared on its prior read. Exactly one
            # delivery tick writes this slot before that future read.
            tl.store(ring_ptr + slot * n + offs, delivered, mask=mask)
            tl.store(accum_ptr + offs, 0, mask=mask)

        @triton.jit
        def advance_device_step(step_ptr):
            tl.atomic_add(step_ptr, 1)

        self._triton_neuron_kernel = neuron_step
        self._triton_compact_kernel = compact
        self._triton_deliver_kernel = deliver
        self._triton_deliver_fixed_kernel = deliver_fixed
        self._triton_finalize_fixed_delivery_kernel = finalize_fixed_delivery
        self._triton_advance_device_step_kernel = advance_device_step
        self._triton_offsets_i32 = self._torch.as_tensor(
            self.graph.offsets.astype(np.int32), device=self.v.device
        )
        self._triton_target_i32 = self._torch.as_tensor(
            self.graph.target.astype(np.int32), device=self.v.device
        )
        self._triton_list = self._torch.zeros(self.n, dtype=self._torch.int32, device=self.v.device)
        self._triton_compact_counter = self._torch.zeros(1, dtype=self._torch.int32, device=self.v.device)
        self._triton_delivery_counter = self._torch.zeros(1, dtype=self._torch.int64, device=self.v.device)
        # The disabled path needs only an eight-byte dummy pointer argument
        # for Triton's compile-time-pruned branch; opt-in mode uses one int64
        # per target rather than an edge- or delay-ring-sized accumulator.
        self._triton_fixed_delivery_accumulator = self._torch.zeros(
            self.n if self.deterministic_delivery else 1,
            dtype=self._torch.int64,
            device=self.v.device,
        )
        self._triton_probability_t = self._torch.zeros(self.n, device=self.v.device)
        self._triton_current_t = self._torch.zeros(self.n, device=self.v.device)
        self._triton_probability_staging = None
        self._triton_current_staging = None
        self._triton_probability_staging_np: Optional[np.ndarray] = None
        self._triton_current_staging_np: Optional[np.ndarray] = None
        self._triton_pinned_input_staging = False
        self._triton_pinned_input_reason: Optional[str] = None
        try:
            self._triton_probability_staging = self._torch.empty(
                self.n, dtype=self._torch.float32, device="cpu", pin_memory=True,
            )
            self._triton_current_staging = self._torch.empty(
                self.n, dtype=self._torch.float32, device="cpu", pin_memory=True,
            )
            self._triton_probability_staging_np = (
                self._triton_probability_staging.numpy()
            )
            self._triton_current_staging_np = self._triton_current_staging.numpy()
            self._triton_pinned_input_staging = True
        except (RuntimeError, TypeError, AttributeError) as exc:
            # Pinned staging is a transfer optimization only.  The ordinary
            # exact float32 copy remains the semantic fallback and is reported
            # in the manifest.
            self._triton_probability_staging = None
            self._triton_current_staging = None
            self._triton_probability_staging_np = None
            self._triton_current_staging_np = None
            self._triton_pinned_input_reason = f"{type(exc).__name__}: {exc}"
        # Device scalars are only read by the opt-in CUDA-Graph variant.  A
        # graph replay therefore sees the current absolute step and seed
        # without changing captured kernel arguments or allocating tensors.
        self._triton_step_t = self._torch.zeros(
            1, dtype=self._torch.int32, device=self.v.device
        )
        self._triton_seed_t = self._torch.zeros(
            1, dtype=self._torch.int64, device=self.v.device
        )
        self._triton_probability_host: Optional[np.ndarray] = None
        self._triton_current_host: Optional[np.ndarray] = None
        self._triton_last_input_h2d_copies = 0
        # Window counts are accumulated in the delivery kernel and copied once
        # at the end of a bounded held-input window. Keeping this buffer
        # resident avoids allocating a cumulative snapshot and a temporary
        # subtraction result for every window; graph replay also resets it
        # through a captured fixed-pointer operation.
        self._triton_window_counts = self._torch.zeros(
            self.n, dtype=self._torch.int32, device=self.v.device
        )
        self._triton_grid_n = (triton.cdiv(n, 1024),)
        self._triton_grid_c = (triton.cdiv(n, 1024),)
        self._triton_grid_d = (2048,)

    # ---------------------------- population and external drives
    def resolve_population(self, spec: str) -> np.ndarray:
        return self.populations.resolve(spec)

    def _mechanosensory_target_selector(self, target: str) -> str:
        """Translate a config population name into its annotation selector."""

        populations = self.sensory_map.get("populations", {})
        if not isinstance(populations, Mapping) or target not in populations:
            # A custom map may intentionally use a direct PopulationIndex
            # selector (for example ``type:SNpp``) instead of a named config
            # population.  The resolver remains the authority for it.
            return target
        seen: set[str] = set()
        current = target
        while True:
            if current in seen:
                raise DriveError(f"mechanosensory population alias cycle at {current!r}")
            seen.add(current)
            item = populations.get(current)
            if not isinstance(item, Mapping):
                raise DriveError(f"mechanosensory population {current!r} is malformed")
            alias = item.get("alias_of")
            if alias is not None:
                current = str(alias)
                continue
            selectors = item.get("selectors")
            if isinstance(selectors, str):
                selectors = [selectors]
            if not isinstance(selectors, Sequence) or isinstance(selectors, (str, bytes)):
                raise DriveError(f"mechanosensory population {current!r} has no selectors")
            selectors = [str(selector).strip() for selector in selectors if str(selector).strip()]
            if not selectors:
                raise DriveError(f"mechanosensory population {current!r} has no selectors")
            return ",".join(selectors)

    def _resolve_mechanosensory_target(self, spec: MechanosensoryDriveSpec) -> np.ndarray:
        cached = self._mechanosensory_targets.get(spec.name)
        if cached is not None:
            return cached
        if self._mechanosensory_target_status.get(spec.name) == "unresolved":
            raise DriveError(
                f"mechanosensory target {spec.target_population!r} for {spec.name!r} "
                "did not resolve in this CSR"
            )
        selector = self._mechanosensory_target_selector(spec.target_population)
        try:
            ids = self.populations.resolve(selector)
        except (PopulationError, ValueError) as exc:
            self._mechanosensory_target_status[spec.name] = "unresolved"
            raise DriveError(
                f"mechanosensory target {spec.target_population!r} for {spec.name!r} "
                f"is not resolvable in this CSR: {exc}"
            ) from exc
        if ids.size == 0:
            self._mechanosensory_target_status[spec.name] = "unresolved"
            raise DriveError(
                f"mechanosensory target {spec.target_population!r} for {spec.name!r} resolved to zero neurons"
            )
        self._mechanosensory_targets[spec.name] = ids.copy()
        self._mechanosensory_target_status[spec.name] = "resolved"
        return ids

    @staticmethod
    def _coerce_population_drive(value: Any, name: str) -> float:
        """Validate one normalized scalar, including optional unit metadata."""

        units = None
        if isinstance(value, Mapping):
            units = value.get("units", value.get("input_units"))
            candidates = [key for key in ("value", "drive", "normalized") if key in value]
            if len(candidates) != 1:
                raise DriveError(
                    f"{name} must contain exactly one value field (value/drive/normalized)"
                )
            if units is not None and units not in {MECHANOSENSORY_INPUT_UNITS, "1"}:
                raise DriveError(
                    f"{name} declares unsupported units {units!r}; expected normalized"
                )
            source = value.get("source")
            if source is not None and str(source) not in {"handbuilt", "fitted"}:
                raise DriveError(f"{name} has an unsupported source label {source!r}")
            value = value[candidates[0]]
        elif isinstance(value, (bool, np.bool_)):
            raise DriveError(f"{name} must be a normalized numeric scalar, not bool")
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DriveError(f"{name} is not a normalized numeric scalar") from exc
        if arr.ndim != 0 or not np.isfinite(arr).item():
            raise DriveError(f"{name} must be a finite normalized scalar")
        drive = float(arr)
        if drive < MECHANOSENSORY_INPUT_MIN - 1e-7 or drive > MECHANOSENSORY_INPUT_MAX + 1e-7:
            raise DriveError(
                f"{name}={drive:g} lies outside the normalized range [-1, 1]"
            )
        # Remove harmless binary-rounding spill at the contract boundary while
        # preserving a strict rejection for actual over-range telemetry.
        return max(MECHANOSENSORY_INPUT_MIN, min(MECHANOSENSORY_INPUT_MAX, drive))

    def _compose_mechanosensory_drives(self, frame: SensoryFrame,
                                       current: np.ndarray) -> None:
        """Inject explicitly mapped population drives into ``current``.

        ``mech:*`` values are telemetry and are recorded as ignored.  Only
        ``population:<name>`` values (or the explicit ``population_drive``
        mapping) can reach the graph.  Aggregate wing/leg entries win over
        their component entries; aliases within one entry must agree exactly.
        """

        channels = frame.channels
        population_values: dict[str, float] = {}
        telemetry: list[str] = []
        declared_units = channels.get("population_drive_units")
        if declared_units is not None and declared_units not in {MECHANOSENSORY_INPUT_UNITS, "1"}:
            raise DriveError(
                f"population_drive_units={declared_units!r} is not normalized"
            )
        declared_range = channels.get("population_drive_range")
        if declared_range is not None:
            if (not isinstance(declared_range, Sequence)
                    or isinstance(declared_range, (str, bytes))
                    or len(declared_range) != 2):
                raise DriveError("population_drive_range must be [-1, 1]")
            if (self._coerce_population_drive(declared_range[0], "population_drive_range[0]") != -1.0
                    or self._coerce_population_drive(declared_range[1], "population_drive_range[1]") != 1.0):
                raise DriveError("population_drive_range must be exactly [-1, 1]")

        def add_value(source: Any, value: Any, label: str) -> None:
            source_name = str(source).strip()
            if not source_name or source_name.startswith("body:"):
                raise DriveError(f"{label} has an invalid population source {source!r}")
            parsed = self._coerce_population_drive(value, label)
            if source_name in population_values and abs(population_values[source_name] - parsed) > 1e-7:
                raise DriveError(
                    f"population drive {source_name!r} was supplied with conflicting values"
                )
            population_values[source_name] = parsed

        for key, value in channels.items():
            if key.startswith("population:"):
                add_value(key.split(":", 1)[1], value, key)
            elif key.startswith("mech:"):
                telemetry.append(key)
        explicit = channels.get("population_drive")
        if explicit is not None:
            if not isinstance(explicit, Mapping):
                raise DriveError("population_drive must be an explicit source-to-scalar mapping")
            explicit_units = explicit.get("units", explicit.get("input_units"))
            explicit_values: Mapping[Any, Any] = explicit
            if explicit_units is not None:
                if explicit_units not in {MECHANOSENSORY_INPUT_UNITS, "1"}:
                    raise DriveError(
                        f"population_drive declares unsupported units {explicit_units!r}"
                    )
                values = explicit.get("values", explicit.get("drives"))
                if not isinstance(values, Mapping):
                    raise DriveError(
                        "population_drive with units must contain a values/drives mapping"
                    )
                explicit_values = values
            for source, value in explicit_values.items():
                if source in {"units", "input_units", "values", "drives"}:
                    continue
                add_value(source, value, f"population_drive[{source!r}]")

        applied: dict[str, Any] = {}
        shadowed: dict[str, Any] = {}
        # Resolve aliases and determine which entries are active before touching
        # the current vector, so a malformed later entry cannot leave a partial
        # injection behind.
        matched: dict[str, tuple[float, tuple[str, ...]]] = {}
        known_sources = {
            source for spec in self.mechanosensory_specs for source in spec.sources
        }
        unknown_sources = sorted(set(population_values) - known_sources)
        if unknown_sources:
            raise DriveError(
                "unknown mechanosensory population source(s): "
                + ", ".join(repr(source) for source in unknown_sources)
            )
        for spec in self.mechanosensory_specs:
            sources = tuple(source for source in spec.sources if source in population_values)
            if not sources:
                continue
            values = tuple(population_values[source] for source in sources)
            if any(abs(value - values[0]) > 1e-7 for value in values[1:]):
                raise DriveError(
                    f"aliases for mechanosensory entry {spec.name!r} disagree: {list(sources)}"
                )
            matched[spec.name] = (values[0], sources)

        by_group: dict[str, list[MechanosensoryDriveSpec]] = {}
        for spec in self.mechanosensory_specs:
            if spec.name in matched and spec.overlap_group:
                by_group.setdefault(spec.overlap_group, []).append(spec)
        selected_names = set(matched)
        for group, group_specs in by_group.items():
            max_priority = max(spec.priority for spec in group_specs)
            selected_names -= {
                spec.name for spec in group_specs if spec.priority < max_priority
            }
            for spec in group_specs:
                if spec.priority < max_priority:
                    shadowed[spec.name] = {
                        "reason": "aggregate mapping has precedence",
                        "overlap_group": group,
                    }

        # Validate all selected targets before mutating ``current``.
        resolved: list[tuple[MechanosensoryDriveSpec, np.ndarray, float, tuple[str, ...]]] = []
        for spec in self.mechanosensory_specs:
            if spec.name not in selected_names:
                continue
            drive, sources = matched[spec.name]
            ids = self._resolve_mechanosensory_target(spec)
            injection = drive * spec.gain_mV_per_ms
            if abs(injection) > spec.max_abs_current_mV_per_ms + 1e-7:
                raise DriveError(f"mechanosensory {spec.name} exceeded its configured current bound")
            resolved.append((spec, ids, injection, sources))
        # Only a few thousand of the full 176k neurons are mechanosensory
        # targets.  Validate overlap on that sparse union instead of clearing
        # and scanning another full-CNS vector every 5 ms.
        if resolved:
            plan_key = tuple(spec.name for spec, _, _, _ in resolved)
            plan = self._mechanosensory_overlap_plans.get(plan_key)
            if plan is None:
                sizes = [int(ids.size) for _, ids, _, _ in resolved]
                pending_ids = np.concatenate([ids for _, ids, _, _ in resolved])
                unique_ids, inverse = np.unique(pending_ids, return_inverse=True)
                offsets = np.cumsum([0, *sizes])
                inverse_groups = tuple(
                    inverse[offsets[index]:offsets[index + 1]].copy()
                    for index in range(len(sizes))
                )
                plan = (int(unique_ids.size), inverse_groups)
                self._mechanosensory_overlap_plans[plan_key] = plan
            combined = np.zeros(plan[0], dtype=np.float32)
            for inverse_group, (_, _, injection, _) in zip(plan[1], resolved):
                np.add.at(combined, inverse_group, np.float32(injection))
            if np.any(
                np.abs(combined)
                > self.mechanosensory_max_gain_mV_per_ms + 1e-7
            ):
                raise DriveError(
                    "combined mechanosensory population injection exceeded the global current bound"
                )
        for spec, ids, injection, sources in resolved:
            current[ids] += np.float32(injection)
            applied[spec.name] = {
                "sources": list(sources),
                "drive": float(matched[spec.name][0]),
                "gain_mV_per_ms": spec.gain_mV_per_ms,
                "current_mV_per_ms": injection,
                "target_population": spec.target_population,
                "target_count": int(ids.size),
                "input_units": MECHANOSENSORY_INPUT_UNITS,
                "output_units": MECHANOSENSORY_OUTPUT_UNITS,
            }
        self._last_mechanosensory_mapping = {
            "input_units": MECHANOSENSORY_INPUT_UNITS,
            "output_units": MECHANOSENSORY_OUTPUT_UNITS,
            "applied": applied,
            "shadowed": shadowed,
            "telemetry_ignored": sorted(set(telemetry)),
            "provided_sources": sorted(population_values),
        }

    def _assign_drive(self, target: np.ndarray, drive: Any, name: str) -> None:
        if isinstance(drive, Mapping):
            for spec, value in drive.items():
                ids = self.populations.resolve(str(spec))
                arr = _as_float_array(value, f"{name}[{spec!r}]")
                if arr.ndim == 0:
                    target[ids] = float(arr)
                elif arr.shape == (ids.size,):
                    target[ids] = arr
                else:
                    raise DriveError(f"{name}[{spec!r}] must be scalar or shape ({ids.size},)")
            return
        arr = _as_float_array(drive, name)
        if arr.shape != (self.n,):
            raise DriveError(f"{name} array must have shape ({self.n},), got {arr.shape}")
        target[:] = arr

    def set_poisson_drive(self, drive: Any) -> None:
        self._assign_drive(self._poisson_hz, drive, "poisson_hz")
        if np.any(self._poisson_hz < 0):
            raise DriveError("poisson_hz cannot be negative")

    def set_current_drive(self, drive: Any) -> None:
        self._assign_drive(self._current_mV_per_ms, drive, "current_mV_per_ms")

    # Short aliases are useful for adapters that expose a generic drive API.
    set_poisson = set_poisson_drive
    set_current = set_current_drive

    def set_flyvis_drive(self, drive: Any) -> None:
        """Apply a FlyVis drive using an explicit unit/target mapping.

        Accepted forms are ``{'poisson_hz': {spec: value}}``,
        ``{'current_mV_per_ms': {spec: value}}`` or a channel vector together
        with ``flyvis_targets`` and ``flyvis_units``.  Overlapping visual and
        tonic drives add explicitly, while uncalibrated visual magnitudes are
        rejected instead of silently clipped.
        """

        self._apply_flyvis(drive, self._poisson_hz, self._current_mV_per_ms)

    set_flyvis = set_flyvis_drive

    def _apply_flyvis(self, drive: Any, poisson: np.ndarray, current: np.ndarray) -> None:
        if isinstance(drive, Mapping):
            # Real FlyVis adapters return a sparse, explicitly mapped target
            # vector.  Never interpret that vector as a dense all-neuron array:
            # target ordering is an asset-level contract and an omitted or
            # ambiguous mapping must fail closed.
            if any(key in drive for key in ("target_indices", "values", "units")):
                missing = [key for key in ("target_indices", "values", "units")
                           if key not in drive]
                if missing:
                    raise DriveError(
                        "FlyVis target drive requires target_indices, values, and units; "
                        f"missing {missing}"
                    )
                source_indices = drive["target_indices"]
                try:
                    indices_raw = np.asarray(source_indices)
                except (TypeError, ValueError) as exc:
                    raise DriveError("FlyVis target_indices must be an integer vector") from exc
                cacheable_indices = bool(
                    isinstance(source_indices, np.ndarray)
                    and indices_raw is source_indices
                    and indices_raw.flags.c_contiguous
                    and not indices_raw.flags.writeable
                    and indices_raw.base is None
                )
                if (
                    cacheable_indices
                    and source_indices is self._validated_flyvis_indices_source
                    and self._validated_flyvis_indices is not None
                ):
                    indices = self._validated_flyvis_indices
                else:
                    if indices_raw.ndim != 1 or indices_raw.size == 0:
                        raise DriveError(
                            "FlyVis target_indices must be a non-empty rank-one integer vector"
                        )
                    if (
                        not np.issubdtype(indices_raw.dtype, np.integer)
                        or np.issubdtype(indices_raw.dtype, np.bool_)
                    ):
                        raise DriveError("FlyVis target_indices must have an integer dtype")
                    indices = indices_raw.astype(np.int64, copy=False)
                    if np.any(indices < 0) or np.any(indices >= self.n):
                        raise DriveError(
                            f"FlyVis target_indices must lie in [0, {self.n})"
                        )
                    if np.unique(indices).size != indices.size:
                        raise DriveError("FlyVis target_indices must be unique")
                    if cacheable_indices:
                        self._validated_flyvis_indices_source = source_indices
                        self._validated_flyvis_indices = indices
                units = drive["units"]
                if not isinstance(units, str) or units not in {"poisson_hz", "current_mV_per_ms"}:
                    raise DriveError(
                        "FlyVis target drive units must be poisson_hz or current_mV_per_ms"
                    )
                values = _as_float_array(drive["values"], "flyvis target values")
                if values.ndim != 1 or values.shape != indices.shape:
                    raise DriveError(
                        "FlyVis target values must be rank one with the same shape as target_indices"
                    )
                if units == "poisson_hz":
                    if np.any(values < 0):
                        raise DriveError("FlyVis poisson_hz target values cannot be negative")
                    poisson[indices] += values
                else:
                    # T4/T5 is a signed current-like deviation; preserve both
                    # directions rather than rectifying the fitted response.
                    # Addition is intentional: overlapping visual and tonic
                    # drives are explicit superposition, never overwrite.
                    current[indices] += values
                self._check_flyvis_bounds(poisson, current)
                return
            recognised = False
            for key in ("poisson_hz", "poisson", "poisson_drive", "stim_hz", "rates_hz"):
                if key in drive:
                    self._merge_drive(poisson, drive[key], f"flyvis.{key}")
                    recognised = True
            for key in ("current_mV_per_ms", "current", "current_drive", "inject_mV_per_ms", "inject"):
                if key in drive:
                    self._merge_drive(current, drive[key], f"flyvis.{key}")
                    recognised = True
            if recognised:
                if np.any(poisson < 0):
                    raise DriveError("FlyVis poisson_hz cannot be negative")
                self._check_flyvis_bounds(poisson, current)
                return
        arr = _as_float_array(drive, "flyvis_t4t5")
        if arr.shape == (self.n,):
            if self.flyvis_units == "poisson_hz":
                poisson += arr
            elif self.flyvis_units == "current_mV_per_ms":
                current += arr
            else:
                raise DriveError("full-neuron FlyVis vector requires flyvis_units")
            self._check_flyvis_bounds(poisson, current)
            return
        if not self.flyvis_targets:
            raise DriveError(
                "FlyVis T4/T5 drive is not mapped; provide flyvis_targets and flyvis_units "
                "or an explicit poisson_hz/current_mV_per_ms mapping"
            )
        if arr.ndim != 1:
            raise DriveError("FlyVis channel drive must be rank 1")
        for channel, target_spec in self.flyvis_targets.items():
            try:
                index = int(channel)
            except (TypeError, ValueError):
                index = {"left": 0, "right": 1, "l": 0, "r": 1}.get(str(channel).lower(), -1)
            if index < 0 or index >= arr.size:
                raise DriveError(f"FlyVis target channel {channel!r} is out of range")
            kind = self.flyvis_units
            gain = 1.0
            spec = target_spec
            if isinstance(target_spec, Mapping):
                spec = target_spec.get("population", target_spec.get("spec"))
                kind = target_spec.get("kind", kind)
                gain = _scalar(target_spec.get("gain", 1.0), f"flyvis target {channel} gain")
            if spec is None or kind not in {"poisson_hz", "current_mV_per_ms"}:
                raise DriveError(f"FlyVis target {channel!r} lacks an explicit population/unit")
            ids = self.populations.resolve(str(spec))
            value = float(arr[index]) * gain
            if kind == "poisson_hz":
                if value < 0:
                    raise DriveError("FlyVis poisson_hz cannot be negative")
                poisson[ids] += value
            else:
                current[ids] += value
        self._check_flyvis_bounds(poisson, current)

    @staticmethod
    def _check_flyvis_bounds(poisson: np.ndarray, current: np.ndarray) -> None:
        """Reject uncalibrated visual magnitudes at the drive boundary."""

        if np.any(poisson < 0):
            raise DriveError("FlyVis poisson_hz cannot be negative")
        if np.any(poisson > MAX_FLYVIS_POISSON_HZ):
            raise DriveError(
                f"FlyVis poisson_hz exceeds calibrated bound {MAX_FLYVIS_POISSON_HZ:g}"
            )
        if np.any(np.abs(current) > MAX_FLYVIS_CURRENT_MV_PER_MS):
            raise DriveError(
                "FlyVis current_mV_per_ms exceeds calibrated bound "
                f"{MAX_FLYVIS_CURRENT_MV_PER_MS:g}"
            )

    def _compose_drives(self, frame: SensoryFrame) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        poisson = self._poisson_hz.copy()
        current = self._current_mV_per_ms.copy()
        channels = frame.channels
        for key in ("poisson_hz", "poisson", "poisson_drive", "stim_hz", "rates_hz"):
            if key in channels:
                self._merge_drive(poisson, channels[key], key)
        for key in ("current_mV_per_ms", "current", "current_drive", "inject_mV_per_ms", "inject"):
            if key in channels:
                self._merge_drive(current, channels[key], key)
        flyvis = None
        for key in ("flyvis_t4t5", "t4t5_drive", "t4t5", "flyvis_drive", "flyvis"):
            if key in channels:
                flyvis = channels[key]
                break
        if flyvis is None and frame.visual is not None:
            metadata = frame.visual.metadata
            for key in ("flyvis_t4t5", "t4t5_drive", "flyvis_drive", "flyvis"):
                if key in metadata:
                    flyvis = metadata[key]
                    break
        if flyvis is not None:
            self._apply_flyvis(flyvis, poisson, current)
        self._compose_mechanosensory_drives(frame, current)
        if np.any(poisson < 0):
            raise DriveError("poisson_hz cannot be negative")
        probability = np.clip(poisson * self.dt_ms / 1000.0, 0.0, 1.0)
        return poisson, current, probability

    def _flyvis_output_metadata(
        self, frame: SensoryFrame,
    ) -> tuple[Any, Optional[Mapping[str, Any]]]:
        """Return the optional FlyVis echo/summary for ``named_outputs``.

        Input validation and injection happen independently in
        :meth:`_compose_drives`; this helper only records output telemetry.
        A trusted adapter may attach a precomputed bounded summary to avoid
        hashing the same frame twice.  Ordinary callers retain the historical
        full echo (default) or locally computed bounded summary.
        """

        raw_flyvis: Any = None
        for key in ("flyvis_t4t5", "t4t5_drive", "t4t5", "flyvis_drive", "flyvis"):
            if key in frame.channels:
                raw_flyvis = frame.channels[key]
                break
        if raw_flyvis is None:
            return None, None
        if isinstance(raw_flyvis, Mapping) and "values" in raw_flyvis:
            if self.include_flyvis_drive_echo:
                return _json_safe(raw_flyvis["values"]), _json_safe(raw_flyvis)
            precomputed = raw_flyvis.get("_telemetry_summary")
            if isinstance(precomputed, Mapping):
                summary = _json_safe(precomputed)
            else:
                summary = _flyvis_payload_summary(raw_flyvis)
            return summary, summary
        value = (
            _json_safe(raw_flyvis)
            if self.include_flyvis_drive_echo
            else _flyvis_payload_summary(raw_flyvis)
        )
        return value, None

    def _named_output_metadata(
        self, index: int, spikes: np.ndarray, deliveries: int,
        population_counts: Mapping[str, int], population_rates: Mapping[str, float],
        poisson_sum: float, current_max: float,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        named: dict[str, Any] = {
            "step_index": index,
            "spike_count": int(spikes.sum()),
            "active_neurons": int(np.count_nonzero(spikes)),
            "deliveries": deliveries,
            "backend": self.backend,
            "device": self.device,
            "max_vram_gib": self.max_vram_gib,
            "provenance": "MaleCNS LIF update order ported from lif_flyb.py",
            "population_spike_counts": population_counts,
            "population_rates_hz": population_rates,
            "external_poisson_hz_sum": poisson_sum,
            "external_current_mV_per_ms_max": current_max,
            "mechanosensory_mapping": _json_safe(self._last_mechanosensory_mapping),
            "mechanosensory_population_drives": {
                name: details["drive"]
                for name, details in self._last_mechanosensory_mapping.get("applied", {}).items()
            },
        }
        if extra:
            named.update(extra)
        return named

    def _add_population_and_flyvis_aliases(
        self, named: dict[str, Any], population_rates: Mapping[str, float], frame: SensoryFrame,
    ) -> None:
        for name, rate in population_rates.items():
            named.setdefault(name, rate)
        flyvis_value, flyvis_mapping = self._flyvis_output_metadata(frame)
        if flyvis_value is not None:
            named["flyvis_t4t5_drive"] = flyvis_value
        if flyvis_mapping is not None:
            named["flyvis_target_mapping"] = flyvis_mapping
            if self.include_flyvis_drive_echo:
                named["flyvis_target_indices"] = flyvis_mapping.get("target_indices")
            named["flyvis_units"] = flyvis_mapping.get("units")

    def _merge_drive(self, target: np.ndarray, drive: Any, name: str) -> None:
        if isinstance(drive, Mapping):
            for spec, value in drive.items():
                ids = self.populations.resolve(str(spec))
                arr = _as_float_array(value, f"{name}[{spec!r}]")
                if arr.ndim == 0:
                    target[ids] += float(arr)
                elif arr.shape == (ids.size,):
                    target[ids] += arr
                else:
                    raise DriveError(f"{name}[{spec!r}] shape mismatch")
        else:
            arr = _as_float_array(drive, name)
            if arr.shape != target.shape:
                raise DriveError(f"{name} must have shape {target.shape}, got {arr.shape}")
            target += arr

    # ---------------------------- step, reset and snapshots
    def _validate_step_input(self, frame: SensoryFrame, timestamp: Timestamp) -> None:
        if not isinstance(frame, SensoryFrame) or not isinstance(timestamp, Timestamp):
            raise ContractError("step expects SensoryFrame and Timestamp")
        if frame.timestamp != timestamp:
            raise ContractError("SensoryFrame and step Timestamp must match")
        if self._last_timestamp is not None:
            delta = timestamp.t_s - self._last_timestamp.t_s
            expected = self.dt_ms / 1000.0
            if delta <= 0 or abs(delta - expected) > max(1e-9, expected * 1e-5):
                raise TimestampError(f"timestamp delta {delta:g}s does not equal engine dt {expected:g}s")

    def step(self, frame: SensoryFrame, timestamp: Timestamp) -> NeuralOutput:
        """Advance exactly one LIF step and return a contract ``NeuralOutput``."""

        self._validate_step_input(frame, timestamp)
        poisson, current, probability = self._compose_drives(frame)
        if self.backend == "numpy_cpu":
            spikes, deliveries = self._step_numpy(probability, current)
        elif self.backend == "triton":
            spikes, deliveries = self._step_triton(probability, current)
        else:
            spikes, deliveries = self._step_torch(probability, current)
        index = self._step_index
        self._step_index += 1
        self._last_timestamp = timestamp
        spikes_u8 = spikes.astype(np.uint8, copy=False)
        rates = spikes_u8.astype(np.float32) * (1000.0 / self.dt_ms)
        population_counts: dict[str, int] = {}
        population_rates: dict[str, float] = {}
        for name, ids in self.report_populations.items():
            count = int(spikes_u8[ids].sum())
            population_counts[name] = count
            population_rates[name] = count / (ids.size * self.dt_ms / 1000.0)
        named = self._named_output_metadata(
            index, spikes_u8, deliveries, population_counts, population_rates,
            float(poisson.sum()), float(np.max(np.abs(current))) if current.size else 0.0,
        )
        if self.backend == "triton":
            steady_mean = (
                self._triton_recorded_steady_total_s / self._triton_recorded_steady_steps
                if self._triton_recorded_steady_steps else None
            )
            named["triton_step_timing"] = {
                "wall_s": self._triton_last_wall_s,
                "phase": "warmup" if self._triton_warmup_active else "recorded",
                "recorded_steps": self._triton_recorded_steps,
                "first_recorded_wall_s": self._triton_recorded_first_wall_s,
                "steady_mean_wall_s": steady_mean,
                "input_h2d_copies": self._triton_last_input_h2d_copies,
                "input_h2d_copies_total": self._triton_input_h2d_copies,
                "kernel_launches": self._triton_last_kernel_launches,
                "host_device_syncs": self._triton_last_host_device_syncs,
                "host_scalar_reads": self._triton_last_host_scalar_reads,
                "counter_reset_launches": self._triton_last_window_counter_resets,
                "warmup_performed": self._triton_warmup_report is not None,
            }
        self._add_population_and_flyvis_aliases(named, population_rates, frame)
        self._last_named_outputs = named
        return NeuralOutput(timestamp, rates_hz=rates, spikes=spikes_u8,
                            named_outputs=named, source=SourceLabel.CONNECTOME)

    def _step_numpy(self, probability: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, int]:
        step = self._step_index
        read = step % self.slots
        write = (step + self.delay_steps) % self.slots
        if np.any(probability):
            stim = self._rng.random(self.n) < probability
            self.v[stim] = V_RESET_MV
            self.ref_until[stim] = step + self.ref_steps
        else:
            stim = np.zeros(self.n, dtype=bool)
        inp = self.ring[read]
        self.g += inp
        inp.fill(0.0)
        refractory = self.ref_until > step
        v_new = V_REST_MV + (self.v - V_REST_MV) * self.decay \
            + self.g * self.syn_coupling + current * self.dt_ms
        self.v[:] = np.where(refractory, V_RESET_MV, v_new)
        self.g[:] = np.where(refractory, self.g, self.g * self.syn_decay)
        spiked = (~refractory) & (self.v >= V_THR_MV)
        self.v[spiked] = V_RESET_MV
        self.g[spiked] = 0.0
        self.ref_until[spiked] = step + self.ref_steps
        all_spikes = spiked | stim
        deliveries = 0
        for source in np.flatnonzero(all_spikes):
            start, end = int(self.graph.offsets[source]), int(self.graph.offsets[source + 1])
            if end > start:
                np.add.at(self.ring[write], self.graph.target[start:end], self._weight[start:end])
                deliveries += end - start
        self.spike_tick += all_spikes.astype(np.int32)
        return all_spikes, deliveries

    def _prepare_triton_inputs(self, probability: np.ndarray,
                               current: np.ndarray) -> tuple[Any, Any]:
        """Copy held inputs to CUDA only when their values actually change."""

        probability_np = np.asarray(probability, dtype=np.float32)
        current_np = np.asarray(current, dtype=np.float32)
        copies = 0
        if (self._triton_probability_host is None
                or not np.array_equal(probability_np, self._triton_probability_host)):
            if self._triton_pinned_input_staging:
                np.copyto(self._triton_probability_staging_np, probability_np)
                self._triton_probability_t.copy_(
                    self._triton_probability_staging, non_blocking=True,
                )
            else:
                self._triton_probability_t.copy_(self._torch.from_numpy(probability_np))
            # ``_compose_drives`` returns a fresh owned array for this window;
            # the engine never mutates it after this call.  Retaining that
            # object is therefore a valid comparison snapshot and avoids a
            # second full-CNS host copy.
            self._triton_probability_host = probability_np
            copies += 1
        if (self._triton_current_host is None
                or not np.array_equal(current_np, self._triton_current_host)):
            if self._triton_pinned_input_staging:
                np.copyto(self._triton_current_staging_np, current_np)
                self._triton_current_t.copy_(
                    self._triton_current_staging, non_blocking=True,
                )
            else:
                self._triton_current_t.copy_(self._torch.from_numpy(current_np))
            self._triton_current_host = current_np
            copies += 1
        self._triton_last_input_h2d_copies = copies
        self._triton_input_h2d_copies += copies
        return self._triton_probability_t, self._triton_current_t

    def _launch_triton_tick(self, probability_t: Any, current_t: Any, *,
                            step: int, delivery_counter: Any,
                            record_window: bool = False,
                            use_device_step: bool = False,
                            advance_step: bool = False) -> None:
        """Launch one fused tick without a host synchronization or D2H copy."""

        read = step % self.slots
        write = (step + self.delay_steps) % self.slots
        self._triton_kernel_launches += 3 + int(self.deterministic_delivery)
        if self.deterministic_delivery and use_device_step and advance_step:
            self._triton_kernel_launches += 1
        self._triton_neuron_kernel[self._triton_grid_n](
            self.v, self.g, self.ref_until, self.spk, self.ring, read, self.n,
            step, self.decay, self.syn_coupling, self.syn_decay, V_REST_MV,
            V_THR_MV, V_RESET_MV, self.ref_steps, probability_t, self.seed + step,
            current_t, self.dt_ms, self._triton_compact_counter,
            delivery_counter,
            self._triton_step_t, self._triton_seed_t, self.slots,
            USE_DEVICE_STEP=use_device_step,
            RESET_DELIVERY=not record_window,
            RECORD_WINDOW=record_window,
            ATOMIC_RING_CONSUME=self.deterministic_delivery,
            BLOCK=1024,
        )
        self._triton_compact_kernel[self._triton_grid_c](
            self.spk, self._triton_list, self._triton_compact_counter, self.n,
            BLOCK=1024,
        )
        if self.deterministic_delivery:
            self._triton_deliver_fixed_kernel[self._triton_grid_d](
                self._triton_list, self._triton_compact_counter,
                self._triton_offsets_i32, self._triton_target_i32, self._weight_t,
                self._triton_fixed_delivery_accumulator, delivery_counter,
                self.spike_tick, self._triton_window_counts, self.n,
                FIXED_SCALE_POWER=int(self._triton_fixedpoint_scale_power or 0),
                RECORD_WINDOW=record_window, BLOCK_SYN=64,
            )
            self._triton_finalize_fixed_delivery_kernel[self._triton_grid_n](
                self._triton_fixed_delivery_accumulator, self.ring, write,
                self.n, self._triton_step_t, self.delay_steps, self.slots,
                FIXED_SCALE_POWER=int(self._triton_fixedpoint_scale_power or 0),
                USE_DEVICE_STEP=use_device_step, BLOCK=1024,
            )
            # The finalizer uses the current step to select the write slot.
            # Advance in a following one-block kernel so every finalizer CTA
            # sees the same value when a captured graph runs.
            if use_device_step and advance_step:
                self._triton_advance_device_step_kernel[(1,)](self._triton_step_t)
        else:
            # Preserve the default float32 atomic delivery kernel, signature,
            # and launch path unchanged.
            self._triton_deliver_kernel[self._triton_grid_d](
                self._triton_list, self._triton_compact_counter,
                self._triton_offsets_i32, self._triton_target_i32, self._weight_t,
                self.ring, write, delivery_counter, self.spike_tick,
                self._triton_window_counts, self.n, self._triton_step_t,
                self.delay_steps, self.slots, USE_DEVICE_STEP=use_device_step,
                ADVANCE_STEP=advance_step, RECORD_WINDOW=record_window,
                BLOCK_SYN=64,
            )

    def _step_triton(self, probability: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, int]:
        """Run one fused Triton tick with configured delivery semantics."""

        started = time.perf_counter()
        probability_t, current_t = self._prepare_triton_inputs(probability, current)
        self._launch_triton_tick(
            probability_t, current_t, step=self._step_index,
            delivery_counter=self._triton_delivery_counter,
        )
        # NeuralOutput retains per-tick spikes, so this is the one required D2H
        # transfer for the scalar step API.  ``step_window`` below holds it on
        # device across ten ticks and performs one aggregate transfer instead.
        # ``spk`` is already uint8. Returning that host array directly avoids
        # a uint8->bool copy here followed by the bool->uint8 conversion in
        # ``step``; the public output remains the same uint8 spike vector.
        spikes = self.spk.detach().cpu().numpy()
        self._triton_host_device_syncs += 1
        deliveries = int(self._triton_delivery_counter.item())
        self._triton_host_device_syncs += 1
        self._triton_host_scalar_reads += 1
        self._triton_last_kernel_launches = 3 + int(self.deterministic_delivery)
        self._triton_last_host_device_syncs = 2
        self._triton_last_host_scalar_reads = 1
        self._triton_last_window_counter_resets = 0
        wall_s = time.perf_counter() - started
        self._triton_last_wall_s = wall_s
        if self._triton_warmup_active:
            self._triton_warmup_durations.append(wall_s)
        else:
            self._triton_recorded_steps += 1
            if self._triton_recorded_first_wall_s is None:
                self._triton_recorded_first_wall_s = wall_s
            else:
                self._triton_recorded_steady_total_s += wall_s
                self._triton_recorded_steady_steps += 1
        # A scalar call uses host step/seed arguments.  If a caller switches
        # back to a graph window, the graph helper must refresh its device
        # scalars rather than assuming the graph's previous step pointer is
        # still aligned.
        if self.cuda_graph_requested:
            self._triton_cuda_graph_step_shadow = None
        return spikes, deliveries

    def _prepare_triton_cuda_graph_scalars(self) -> int:
        """Refresh graph-visible absolute step/seed scalars when needed."""

        updates = 0
        if self._triton_cuda_graph_step_shadow != self._step_index:
            self._triton_step_t.fill_(int(self._step_index))
            self._triton_cuda_graph_step_shadow = self._step_index
            updates += 1
        if self._triton_cuda_graph_seed_shadow != self.seed:
            self._triton_seed_t.fill_(int(self.seed))
            self._triton_cuda_graph_seed_shadow = self.seed
            updates += 1
        self._triton_cuda_graph_pointer_updates += updates
        return updates

    def _disable_triton_cuda_graph(self, reason: str) -> None:
        """Disable graph replay after a capture/runtime failure, fail-open to Triton."""

        self._triton_cuda_graph_enabled = False
        self._triton_cuda_graph_reason = str(reason)
        self._triton_cuda_graph_last_phase = "fallback"
        self._triton_cuda_graphs.clear()

    def _run_triton_window_cuda_graph(
        self, probability_t: Any, current_t: Any, *, requested: int,
    ) -> Optional[tuple[np.ndarray, int, float, dict[str, Any]]]:
        """Capture/replay one fixed-size held-input window.

        Device scalars carry the absolute step and configured seed. The
        default delivery kernel advances the step pointer; fixed-point mode
        finalizes its ring slot and advances the pointer in ordered kernels.
        The captured sequence therefore remains valid after replay, while the
        fixed probability/current pointers may be updated between replays by
        ``_prepare_triton_inputs``.
        A failed capture restores the pre-capture state and returns ``None``
        so the caller can execute the ordinary window kernels exactly once.
        """

        if not self._triton_cuda_graph_enabled:
            return None
        pointer_updates = self._prepare_triton_cuda_graph_scalars()
        entry = self._triton_cuda_graphs.get(requested)
        capture = entry is None
        before: Optional[dict[str, Any]] = None
        before_spk: Optional[np.ndarray] = None
        started = time.perf_counter()
        capture_wall_s: Optional[float] = None
        replay_wall_s: Optional[float] = None
        graph: Any
        # Two captured window-counter resets and the cumulative spike-count
        # add surround the per-tick kernels. Fixed-point delivery adds the
        # target finalizer and an ordered device-step advance kernel.
        captured_nodes = 3 + (
            3 + (2 if self.deterministic_delivery else 0)
        ) * requested
        kernel_launches_before = self._triton_kernel_launches
        try:
            if capture:
                # CUDA Graph capture cannot recover from a partial graph with
                # unknown dynamic state.  Keep a cold-path snapshot solely for
                # the failure branch; successful capture never copies it back.
                before = self.snapshot()
                before_spk = self.spk.detach().cpu().numpy().copy()
                graph = self._torch.cuda.CUDAGraph()
                capture_started = time.perf_counter()
                # Flush any host→device held-input copies before beginning the
                # capture.  This synchronization is cold-only and is reported
                # separately from steady replay telemetry.
                self._torch.cuda.synchronize()
                self._triton_host_device_syncs += 1
                with self._torch.cuda.graph(graph):
                    self._triton_window_counts.zero_()
                    self._triton_delivery_counter.zero_()
                    for _ in range(requested):
                        self._launch_triton_tick(
                            probability_t, current_t, step=0,
                            delivery_counter=self._triton_delivery_counter,
                            record_window=True, use_device_step=True,
                            advance_step=True,
                        )
                    # Keep cumulative state in the graph so replay has no
                    # extra Python-side GPU launch.
                    self.spike_tick.add_(self._triton_window_counts)
                capture_wall_s = time.perf_counter() - capture_started
                self._triton_cuda_graphs[requested] = {
                    "graph": graph,
                    "requested_steps": requested,
                    "captured_kernel_nodes": captured_nodes,
                    "captured_at_step": self._step_index,
                    "capture_wall_s": capture_wall_s,
                }
                self._triton_cuda_graph_capture_count += 1
                self._triton_cuda_graph_seed_shadow = self.seed
                self._triton_cuda_graph_last_phase = "capture"
                self._triton_cuda_graph_last_capture_wall_s = capture_wall_s
                # CUDA Graph capture records nodes but does not execute the
                # captured work.  Replay once now so this cold call returns
                # the first requested window rather than an all-zero buffer.
                replay_started = time.perf_counter()
                graph.replay()
                replay_wall_s = time.perf_counter() - replay_started
                self._triton_cuda_graph_replay_count += 1
                self._triton_cuda_graph_host_launches += 1
                self._triton_cuda_graph_step_shadow = self._step_index + requested
                self._triton_cuda_graph_last_replay_wall_s = replay_wall_s
                self._triton_last_graph_host_launches = 1
                self._triton_last_graph_kernel_nodes = captured_nodes
                self._triton_last_graph_pointer_updates = pointer_updates
                # Count the first replay's child nodes as logical GPU work;
                # the host only submitted one graph launch.
                # ``_launch_triton_tick`` updates the ordinary counter while
                # building the graph; replace those Python-side bookkeeping
                # increments with the actual captured node count (the two
                # counter clears and cumulative add are graph nodes too).
                self._triton_kernel_launches = kernel_launches_before + captured_nodes
                self._triton_last_kernel_launches = captured_nodes
            else:
                graph = entry["graph"]
                replay_started = time.perf_counter()
                graph.replay()
                replay_wall_s = time.perf_counter() - replay_started
                self._triton_cuda_graph_replay_count += 1
                self._triton_cuda_graph_host_launches += 1
                self._triton_cuda_graph_step_shadow = self._step_index + requested
                self._triton_cuda_graph_last_phase = "replay"
                self._triton_cuda_graph_last_replay_wall_s = replay_wall_s
                self._triton_last_graph_host_launches = 1
                self._triton_last_graph_kernel_nodes = int(entry["captured_kernel_nodes"])
                self._triton_last_graph_pointer_updates = pointer_updates
                # Replay submits one host graph node; its child kernels are
                # represented by ``graph_kernel_nodes`` rather than counted as
                # Python launches.
                self._triton_last_kernel_launches = 0

            count_vector = self._triton_window_counts.detach().cpu().numpy()
            self._triton_host_device_syncs += 1
            deliveries = int(self._triton_delivery_counter.item())
            self._triton_host_device_syncs += 1
            self._triton_host_scalar_reads += 1
            self._triton_last_host_device_syncs = 3 if capture else 2
            self._triton_last_host_scalar_reads = 1
            self._triton_last_window_counter_resets = 2
            wall_s = time.perf_counter() - started
            self._triton_last_wall_s = wall_s
            if self._triton_warmup_active:
                self._triton_warmup_durations.append(wall_s)
            else:
                self._triton_recorded_steps += requested
                per_tick = wall_s / requested
                if self._triton_recorded_first_wall_s is None:
                    self._triton_recorded_first_wall_s = per_tick
                else:
                    self._triton_recorded_steady_total_s += wall_s
                    self._triton_recorded_steady_steps += requested
            details = {
                "execution_mode": "cuda_graph_capture" if capture else "cuda_graph_replay",
                "graph_phase": self._triton_cuda_graph_last_phase,
                "graph_kernel_nodes": captured_nodes,
                "graph_host_launches": self._triton_last_graph_host_launches,
                "graph_pointer_updates": pointer_updates,
                "capture_wall_s": capture_wall_s,
                "replay_wall_s": replay_wall_s,
                "graph_capture_count": self._triton_cuda_graph_capture_count,
                "graph_replay_count": self._triton_cuda_graph_replay_count,
                "graph_fallback_reason": self._triton_cuda_graph_reason,
                "graph_requested": self.cuda_graph_requested,
                "graph_enabled": self._triton_cuda_graph_enabled,
            }
            return np.asarray(count_vector, dtype=np.int32), deliveries, wall_s, details
        except BaseException as exc:
            reason = f"CUDA Graph capture/replay failed: {type(exc).__name__}: {exc}"
            if capture and before is not None:
                try:
                    # Ensure a failed capture has left capture mode before
                    # issuing ordinary kernels, then restore all dynamic state.
                    self._torch.cuda.synchronize()
                except BaseException:
                    pass
                try:
                    self.restore(before)
                    if before_spk is not None:
                        self.spk.copy_(self._torch.as_tensor(
                            before_spk, device=self.spk.device
                        ))
                    self._triton_window_counts.zero_()
                    self._triton_delivery_counter.zero_()
                    self._triton_step_t.fill_(int(self._step_index))
                    self._triton_seed_t.fill_(int(self.seed))
                except BaseException as restore_exc:
                    reason += f"; state restore failed: {type(restore_exc).__name__}: {restore_exc}"
            self._disable_triton_cuda_graph(reason)
            self._triton_kernel_launches = kernel_launches_before
            if not capture:
                # A replay may have submitted some graph nodes before an
                # asynchronous CUDA error becomes visible. Unlike capture,
                # replay has no pre-existing snapshot to restore here; do not
                # continue with an uncaptured window from potentially partial
                # neural state. The caller must discard this engine and report
                # the failure explicitly.
                raise EngineError(reason) from exc
            return None

    @staticmethod
    def _retimestamp_frame(frame: SensoryFrame, timestamp: Timestamp) -> SensoryFrame:
        """Retimestamp a contract frame while preserving visual/body payloads."""

        visual = None if frame.visual is None else replace(frame.visual, timestamp=timestamp)
        body = None if frame.body is None else replace(frame.body, timestamp=timestamp)
        return SensoryFrame(timestamp=timestamp, visual=visual, body=body,
                            channels=frame.channels, source=frame.source)

    def step_window(self, frame: SensoryFrame, timestamp: Timestamp,
                    steps: int = MAX_TRITON_WINDOW_STEPS) -> NeuralOutput:
        """Advance a bounded held-input neural window.

        Triton keeps the ten half-millisecond ticks on device and copies one
        aggregate spike-count vector back to the host at the end.  The returned
        ``NeuralOutput`` therefore contains window spike counts in ``spikes``
        and the corresponding mean rates in ``rates_hz``; its timestamp is the
        final tick and named outputs carry the start/end timestamps.  This is
        the windowed counterpart to :meth:`step`, whose per-tick output remains
        unchanged for callers that require every spike vector.

        A CPU/Torch engine uses the same contract through an explicit host
        fallback.  Input drives are composed once for Triton and held exactly
        across all requested ticks.  ``steps`` is capped at ten to prevent an
        accidental unbounded pre-flight or recording window.  Passing
        ``cuda_graph=True`` opts into a fixed-pointer CUDA Graph for each
        requested window length; the first call reports cold capture plus its
        first replay, while later calls report steady graph replay.  Graph
        capture is never selected implicitly and falls back to this uncaptured
        path with a recorded reason when the API or capture is unavailable.
        """

        if isinstance(steps, bool):
            raise ValueError("window steps must be an integer")
        try:
            requested = int(steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("window steps must be an integer") from exc
        if requested != steps or requested < 1 or requested > MAX_TRITON_WINDOW_STEPS:
            raise ValueError(
                f"window steps must be an integer in [1, {MAX_TRITON_WINDOW_STEPS}]"
            )
        self._validate_step_input(frame, timestamp)
        dt_s = self.dt_ms / 1000.0
        rate_hz = timestamp.rate_hz
        last_timestamp = Timestamp(
            timestamp.t_s + (requested - 1) * dt_s,
            timestamp.tick + requested - 1,
            timestamp.clock,
            rate_hz,
        )
        pipeline_started = time.perf_counter() if self.backend == "triton" else 0.0
        drive_compose_s = 0.0
        input_prepare_s = 0.0

        if self.backend != "triton":
            sub_timestamps = tuple(
                Timestamp(timestamp.t_s + index * dt_s, timestamp.tick + index,
                          timestamp.clock, rate_hz)
                for index in range(requested)
            )
            outputs: list[NeuralOutput] = []
            for sub_timestamp in sub_timestamps:
                subframe = self._retimestamp_frame(frame, sub_timestamp)
                outputs.append(self.step(subframe, sub_timestamp))
            count_vector = np.sum(
                [np.asarray(output.spikes, dtype=np.int32) for output in outputs],
                axis=0,
                dtype=np.int32,
            )
            deliveries = int(sum(int(output.named_outputs.get("deliveries", 0))
                                for output in outputs))
            window_wall_s = None
        else:
            started = time.perf_counter()
            compose_started = time.perf_counter()
            poisson, current, probability = self._compose_drives(frame)
            drive_compose_s = time.perf_counter() - compose_started
            input_started = time.perf_counter()
            probability_t, current_t = self._prepare_triton_inputs(probability, current)
            input_prepare_s = time.perf_counter() - input_started
            input_h2d_copies = self._triton_last_input_h2d_copies
            graph_details: Optional[dict[str, Any]] = None
            graph_result = self._run_triton_window_cuda_graph(
                probability_t, current_t, requested=requested,
            ) if self._triton_cuda_graph_enabled else None
            if graph_result is not None:
                count_vector, deliveries, window_wall_s, graph_details = graph_result
            else:
                # Uncaptured window kernels. This is also the explicit
                # fallback after a graph feature/capture failure.
                self._triton_window_counts.zero_()
                self._triton_delivery_counter.zero_()
                self._triton_kernel_launches += 2
                for index in range(requested):
                    self._launch_triton_tick(
                        probability_t, current_t, step=self._step_index + index,
                        delivery_counter=self._triton_delivery_counter,
                        record_window=True,
                    )
                # One synchronization for the complete held-input window. The
                # neuron/delivery kernels accumulated exact per-neuron counts
                # directly, so no snapshot/subtraction tensor is needed.
                count_vector = self._triton_window_counts.detach().cpu().numpy()
                self._triton_host_device_syncs += 1
                # ``spike_tick`` is public cumulative state, but no caller can
                # observe it between ticks of this atomic window API. Add the
                # exact window counts once instead of per-tick active atomics.
                self.spike_tick.add_(self._triton_window_counts)
                self._triton_kernel_launches += 1
                deliveries = int(self._triton_delivery_counter.item())
                self._triton_host_device_syncs += 1
                self._triton_host_scalar_reads += 1
                self._triton_last_kernel_launches = (
                    3 + (3 + int(self.deterministic_delivery)) * requested
                )
                self._triton_last_host_device_syncs = 2
                self._triton_last_host_scalar_reads = 1
                self._triton_last_window_counter_resets = 2
                window_wall_s = time.perf_counter() - started
                self._triton_last_wall_s = window_wall_s
                if self._triton_warmup_active:
                    self._triton_warmup_durations.append(window_wall_s)
                else:
                    self._triton_recorded_steps += requested
                    if self._triton_recorded_first_wall_s is None:
                        self._triton_recorded_first_wall_s = window_wall_s / requested
                    else:
                        self._triton_recorded_steady_total_s += window_wall_s
                        self._triton_recorded_steady_steps += requested
            self._step_index += requested
            self._last_timestamp = last_timestamp
            count_vector = np.asarray(count_vector, dtype=np.int32)

        host_output_started = time.perf_counter() if self.backend == "triton" else 0.0
        window_dt_s = requested * dt_s
        rates = count_vector.astype(np.float32) / np.float32(window_dt_s)
        population_counts: dict[str, int] = {}
        population_rates: dict[str, float] = {}
        if (
            self._report_population_flat_indices is not None
            and self._report_population_offsets is not None
        ):
            selected_counts = count_vector[self._report_population_flat_indices]
            grouped_counts = np.add.reduceat(
                selected_counts, self._report_population_offsets,
            )
            for index, name in enumerate(self._report_population_names):
                count = int(grouped_counts[index])
                population_counts[name] = count
                population_rates[name] = (
                    count / (int(self._report_population_sizes[index]) * window_dt_s)
                )
        else:
            for name, ids in self.report_populations.items():
                count = int(count_vector[ids].sum())
                population_counts[name] = count
                population_rates[name] = (
                    count / (ids.size * window_dt_s) if ids.size else 0.0
                )
        named = self._named_output_metadata(
            self._step_index - 1, count_vector, deliveries, population_counts, population_rates,
            float(poisson.sum()) if self.backend == "triton" else 0.0,
            float(np.max(np.abs(current))) if self.backend == "triton" and current.size else 0.0,
            {
                "neural_window_steps": requested,
                "neural_window_dt_s": window_dt_s,
                "window_start_timestamp": timestamp.to_dict(),
                "window_end_timestamp": last_timestamp.to_dict(),
                "spikes_are_window_counts": True,
                "rates_semantics": "window_mean_hz",
            },
        )
        if self.backend == "triton":
            named["triton_step_timing"] = {
                "wall_s": window_wall_s,
                "wall_s_per_tick": None if window_wall_s is None else window_wall_s / requested,
                "phase": "warmup" if self._triton_warmup_active else "recorded",
                "on_device_window": True,
                "host_spike_copies": 1,
                "held_input_h2d_copies": input_h2d_copies,
                "input_h2d_copies_total": self._triton_input_h2d_copies,
                "kernel_launches": self._triton_last_kernel_launches,
                "host_device_syncs": self._triton_last_host_device_syncs,
                "host_scalar_reads": self._triton_last_host_scalar_reads,
                "counter_reset_launches": self._triton_last_window_counter_resets,
                "execution_mode": (
                    graph_details.get("execution_mode")
                    if graph_details is not None else "uncaptured_triton"
                ),
                "graph_phase": (
                    graph_details.get("graph_phase")
                    if graph_details is not None else self._triton_cuda_graph_last_phase
                ),
                "graph_kernel_nodes": (
                    graph_details.get("graph_kernel_nodes", 0)
                    if graph_details is not None else 0
                ),
                "graph_host_launches": (
                    graph_details.get("graph_host_launches", 0)
                    if graph_details is not None else 0
                ),
                "graph_pointer_updates": (
                    graph_details.get("graph_pointer_updates", 0)
                    if graph_details is not None else 0
                ),
                "graph_capture_wall_s": (
                    graph_details.get("capture_wall_s")
                    if graph_details is not None else None
                ),
                "graph_replay_wall_s": (
                    graph_details.get("replay_wall_s")
                    if graph_details is not None else None
                ),
                "graph_capture_count": self._triton_cuda_graph_capture_count,
                "graph_replay_count": self._triton_cuda_graph_replay_count,
                "graph_fallback_reason": self._triton_cuda_graph_reason,
            }
        self._add_population_and_flyvis_aliases(named, population_rates, frame)
        if self.backend == "triton":
            timing = named["triton_step_timing"]
            timing["drive_compose_s"] = drive_compose_s
            timing["input_prepare_s"] = input_prepare_s
            timing["host_output_s"] = time.perf_counter() - host_output_started
            timing["engine_pipeline_s"] = time.perf_counter() - pipeline_started
        self._last_named_outputs = named
        fast_constructor = getattr(
            NeuralOutput, "_from_validated_engine_arrays", None,
        )
        if callable(fast_constructor):
            return fast_constructor(
                last_timestamp, rates_hz=rates, spikes=count_vector,
                named_outputs=named, source=SourceLabel.CONNECTOME,
            )
        return NeuralOutput(
            last_timestamp, rates_hz=rates, spikes=count_vector,
            named_outputs=named, source=SourceLabel.CONNECTOME,
        )

    run_window = step_window

    def _step_torch(self, probability: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, int]:
        torch = self._torch
        step = self._step_index
        read = step % self.slots
        write = (step + self.delay_steps) % self.slots
        p = torch.as_tensor(probability, device=self.v.device)
        # ``probability`` is already a host array composed at the contract
        # boundary.  Inspecting it on the host avoids ``torch.any(p)``
        # synchronising CUDA merely to decide whether a Poisson generator is
        # active.  Preserve the reference behaviour of consuming RNG only
        # when at least one probability is non-zero.
        probability_active = bool(np.any(probability))
        cur = torch.as_tensor(current, device=self.v.device)
        if self.device == "cuda":
            # The CUDA tensor owns its storage, so scale in place and avoid a
            # temporary host ``current * dt`` array plus a second H2D copy.
            cur.mul_(self.dt_ms)
        else:
            # CPU ``as_tensor(numpy)`` aliases the caller's array; do not
            # mutate that contract-owned input when applying dt.
            cur = cur * self.dt_ms
        if probability_active:
            stim = torch.rand(self.n, device=self.v.device, generator=self._torch_generator) < p
            self.v.masked_fill_(stim, V_RESET_MV)
            self.ref_until.masked_fill_(stim, step + self.ref_steps)
        else:
            stim = torch.zeros(self.n, dtype=torch.bool, device=self.v.device)
        inp = self.ring[read]
        self.g.add_(inp)
        inp.zero_()
        refractory = self.ref_until > step
        v_new = V_REST_MV + (self.v - V_REST_MV) * self.decay \
            + self.g * self.syn_coupling + cur
        # Copy the update once, then clamp refractory neurons in place.  This
        # removes the full-size ``full_like``/``where`` temporary while
        # preserving the same update order and values.
        self.v.copy_(v_new)
        self.v.masked_fill_(refractory, V_RESET_MV)
        self.g.copy_(torch.where(refractory, self.g, self.g * self.syn_decay))
        spiked = (~refractory) & (self.v >= V_THR_MV)
        self.v.masked_fill_(spiked, V_RESET_MV)
        self.g.masked_fill_(spiked, 0.0)
        self.ref_until.masked_fill_(spiked, step + self.ref_steps)
        all_spikes = spiked | stim
        idx = all_spikes.nonzero().flatten()
        deliveries = 0
        if int(idx.numel()):
            starts = self._offsets_t.index_select(0, idx)
            lens = self._offsets_t.index_select(0, idx + 1) - starts
            cumulative = torch.cumsum(lens, 0)
            total = int(cumulative[-1].item())
            if total:
                positions = torch.arange(total, device=self.v.device)
                groups = torch.searchsorted(cumulative, positions, right=True)
                syn = starts.index_select(0, groups) + positions \
                    - (cumulative.index_select(0, groups) - lens.index_select(0, groups))
                self.ring[write].index_add_(0, self._target_t.index_select(0, syn),
                                             self._weight_t.index_select(0, syn))
                deliveries = total
        # In-place bool -> int32 promotion avoids a temporary cast tensor.
        self.spike_tick.add_(all_spikes)
        return all_spikes.detach().cpu().numpy().astype(bool), deliveries

    def warmup(self, steps: int = 1) -> dict[str, Any]:
        """JIT-warm the Triton path, then restore the exact engine state.

        The first Triton call can include module loading, allocator growth and
        several kernel compilations.  Callers may perform this bounded warmup
        before starting a recorded flight window.  Warmup timestamps, dynamic
        arrays, configured seed, last output, and the transient spike buffer
        are restored afterwards; the reported wall time deliberately remains
        visible in :meth:`manifest` and the returned report.

        CPU/Torch engines return an explicit no-op report so a runner can call
        this method without branching on backend.  At most
        ``MAX_TRITON_WARMUP_STEPS`` ticks are accepted to prevent an accidental
        unbounded pre-flight run.
        """

        if isinstance(steps, bool):
            raise ValueError("warmup steps must be an integer")
        try:
            requested = int(steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("warmup steps must be an integer") from exc
        if requested != steps or requested < 1 or requested > MAX_TRITON_WARMUP_STEPS:
            raise ValueError(
                f"warmup steps must be an integer in [1, {MAX_TRITON_WARMUP_STEPS}]"
            )
        if self.backend != "triton":
            report = {
                "performed": False,
                "backend": self.backend,
                "requested_steps": requested,
                "completed_steps": 0,
                "reason": "Triton warmup is only applicable to backend='triton'",
                "startup_cost_visible": True,
            }
            self._triton_warmup_report = report
            return dict(report)

        before = self.snapshot()
        before_named = dict(self._last_named_outputs)
        before_mapping = copy.deepcopy(self._last_mechanosensory_mapping)
        before_spk = self.spk.detach().cpu().numpy().copy()
        self._triton_warmup_durations = []
        self._triton_warmup_active = True
        started = time.perf_counter()
        completed = 0
        failure: Optional[BaseException] = None
        restored = False
        try:
            rate_hz = 1000.0 / self.dt_ms
            dt_s = self.dt_ms / 1000.0
            for index in range(requested):
                timestamp = Timestamp(index * dt_s, index, "neural-warmup", rate_hz)
                self.step(SensoryFrame(timestamp), timestamp)
                completed += 1
        except BaseException as exc:  # restore state before surfacing any JIT/runtime error
            failure = exc
        finally:
            self._triton_warmup_active = False
            try:
                self.restore(before)
                self.spk.copy_(self._torch.as_tensor(before_spk, device=self.spk.device))
                self._last_named_outputs = before_named
                self._last_mechanosensory_mapping = before_mapping
                restored = True
            except BaseException as exc:
                failure = failure or exc
        elapsed = time.perf_counter() - started
        durations = list(self._triton_warmup_durations)
        report: dict[str, Any] = {
            "performed": True,
            "backend": self.backend,
            "requested_steps": requested,
            "completed_steps": completed,
            "bounded_max_steps": MAX_TRITON_WARMUP_STEPS,
            "wall_s": elapsed,
            "first_step_wall_s": durations[0] if durations else None,
            "steady_mean_wall_s": (
                float(np.mean(durations[1:])) if len(durations) > 1 else None
            ),
            "state_restored": restored,
            "startup_cost_visible": True,
        }
        if failure is not None:
            report["error"] = f"{type(failure).__name__}: {failure}"
        self._triton_warmup_report = report
        # Warmup timing is retained in ``report``; keep launch/sync counters
        # scoped to the subsequent recorded run so manifests are comparable.
        self._triton_kernel_launches = 0
        self._triton_host_device_syncs = 0
        self._triton_host_scalar_reads = 0
        self._triton_input_h2d_copies = 0
        self._triton_last_kernel_launches = 0
        self._triton_last_host_device_syncs = 0
        self._triton_last_host_scalar_reads = 0
        self._triton_last_window_counter_resets = 0
        if failure is not None:
            raise EngineError(f"Triton warmup failed: {report['error']}") from failure
        return dict(report)

    warm_kernels = warmup

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset dynamic state while retaining graph and configured drives."""

        if seed is not None:
            self.seed = int(seed)
        self._step_index = 0
        self._last_timestamp = None
        self._last_named_outputs = {}
        if self.backend == "triton":
            self._triton_recorded_steps = 0
            self._triton_recorded_first_wall_s = None
            self._triton_recorded_steady_total_s = 0.0
            self._triton_recorded_steady_steps = 0
            self._triton_last_wall_s = None
            self._triton_kernel_launches = 0
            self._triton_host_device_syncs = 0
            self._triton_host_scalar_reads = 0
            self._triton_input_h2d_copies = 0
            self._triton_last_kernel_launches = 0
            self._triton_last_host_device_syncs = 0
            self._triton_last_host_scalar_reads = 0
            self._triton_last_window_counter_resets = 0
            self._triton_last_graph_host_launches = 0
            self._triton_last_graph_kernel_nodes = 0
            self._triton_last_graph_pointer_updates = 0
            self._triton_cuda_graph_capture_count = 0
            self._triton_cuda_graph_replay_count = 0
            self._triton_cuda_graph_host_launches = 0
            self._triton_cuda_graph_pointer_updates = 0
            self._triton_cuda_graph_step_shadow = None
            self._triton_cuda_graph_seed_shadow = None
            self._triton_cuda_graph_last_phase = "disabled"
            self._triton_cuda_graph_last_capture_wall_s = None
            self._triton_cuda_graph_last_replay_wall_s = None
        if self.backend == "numpy_cpu":
            self.v.fill(V_REST_MV)
            self.g.fill(0.0)
            self.ref_until.fill(0)
            self.ring.fill(0.0)
            self.spike_tick.fill(0)
            self._rng = np.random.default_rng(self.seed)
        else:
            self.v.fill_(V_REST_MV)
            self.g.zero_()
            self.ref_until.zero_()
            self.ring.zero_()
            self.spike_tick.zero_()
            if hasattr(self, "spk"):
                self.spk.zero_()
            if self.backend == "triton":
                self._triton_compact_counter.zero_()
                self._triton_delivery_counter.zero_()
                self._triton_fixed_delivery_accumulator.zero_()
                self._triton_window_counts.zero_()
                self._triton_step_t.fill_(0)
                self._triton_seed_t.fill_(int(self.seed))
            self._torch_generator.manual_seed(self.seed)

    def snapshot(self) -> dict[str, Any]:
        """Return a restorable copy of dynamic state and RNG position."""

        if self.backend == "numpy_cpu":
            arrays = {name: getattr(self, name).copy()
                      for name in ("v", "g", "ref_until", "ring", "spike_tick")}
            rng_state = copy.deepcopy(self._rng.bit_generator.state)
        else:
            arrays = {name: getattr(self, name).detach().cpu().numpy().copy()
                      for name in ("v", "g", "ref_until", "ring", "spike_tick")}
            # Torch's per-engine Generator is part of the dynamic state.  The
            # previous snapshot path restored tensors but not this generator,
            # making a Poisson replay diverge after restore.
            rng_state = None
            if self.backend == "torch_cpu" or self.backend == "torch_cuda":
                rng_state = self._torch_generator.get_state().detach().cpu().numpy().copy()
        return {
            "schema_version": SCHEMA_VERSION,
            "step_index": self._step_index,
            "last_timestamp": None if self._last_timestamp is None else self._last_timestamp.to_dict(),
            "arrays": arrays,
            "rng_state": rng_state,
            "seed": self.seed,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if not isinstance(snapshot, Mapping) or "arrays" not in snapshot:
            raise EngineError("snapshot must be a mapping produced by snapshot()")
        arrays = snapshot["arrays"]
        names = ("v", "g", "ref_until", "ring", "spike_tick")
        for name in names:
            if name not in arrays:
                raise EngineError(f"snapshot array {name!r} is missing")
            expected_shape = (self.n,) if name != "ring" else (self.slots, self.n)
            if np.asarray(arrays[name]).shape != expected_shape:
                raise EngineError(f"snapshot array {name!r} has the wrong shape")
        if self.backend == "numpy_cpu":
            for name in names:
                getattr(self, name)[:] = np.asarray(arrays[name], dtype=getattr(self, name).dtype)
            if snapshot.get("rng_state") is not None:
                self._rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        else:
            for name in names:
                target = getattr(self, name)
                target.copy_(self._torch.as_tensor(np.asarray(arrays[name]), device=target.device,
                                                   dtype=target.dtype))
            if snapshot.get("rng_state") is not None and hasattr(self, "_torch_generator"):
                state = self._torch.as_tensor(np.asarray(snapshot["rng_state"]),
                                              dtype=self._torch.uint8, device="cpu")
                self._torch_generator.set_state(state)
            if self.backend == "triton":
                # Fixed-point delivery scratch is internal and zero at every
                # completed tick; discard partial scratch after any restore,
                # including a failed CUDA Graph capture fallback.
                self._triton_fixed_delivery_accumulator.zero_()
        self._step_index = int(snapshot.get("step_index", 0))
        ts = snapshot.get("last_timestamp")
        self._last_timestamp = None if ts is None else Timestamp.from_dict(ts)
        self.seed = int(snapshot.get("seed", self.seed))
        if self.backend == "triton":
            # Graph-visible scalars are intentionally not part of the public
            # snapshot schema; refresh them lazily before the next replay.
            self._triton_cuda_graph_step_shadow = None
            self._triton_cuda_graph_seed_shadow = None

    @property
    def state(self) -> dict[str, Any]:
        snap = self.snapshot()
        return snap["arrays"]

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def last_named_outputs(self) -> Mapping[str, Any]:
        return dict(self._last_named_outputs)

    def manifest(self) -> dict[str, Any]:
        """Describe the engine without claiming unsupported backends."""

        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "csr": preflight_csr(self.csr_path).to_dict(),
            "backend_requested": self.backend_requested,
            "backend": self.backend,
            "device_requested": self.device_requested,
            "device": self.device,
            "max_vram_gib": self.max_vram_gib,
            "dt_ms": self.dt_ms,
            "gain": self.gain,
            "kc_gain": self.kc_gain,
            "include_flyvis_drive_echo": self.include_flyvis_drive_echo,
            "flyvis_drive_bounds": {
                "current_mV_per_ms_abs": MAX_FLYVIS_CURRENT_MV_PER_MS,
                "poisson_hz_max": MAX_FLYVIS_POISSON_HZ,
            },
            "seed": self.seed,
            "n": self.n,
            "edges": self.e,
            "delay_steps": self.delay_steps,
            "ref_steps": self.ref_steps,
            "deterministic_delivery": {
                "requested": self.deterministic_delivery_requested,
                "enabled": self.deterministic_delivery,
                "mode": (
                    "fixedpoint_int64_exact_float32_round_once"
                    if self.deterministic_delivery else
                    "triton_float32_atomic" if self.backend == "triton" else
                    "backend_default"
                ),
                "fixedpoint_scale_power_of_two": (
                    self._triton_fixedpoint_scale_power
                    if self.deterministic_delivery else None
                ),
                "max_abs_tick_accumulation_bound_fixed_units": (
                    str(self._triton_fixedpoint_abs_tick_bound)
                    if self._triton_fixedpoint_abs_tick_bound is not None else None
                ),
                "bound_semantics": (
                    "max_abs_fixed_edge_weight multiplied by maximum graph indegree; "
                    "assumes every presynaptic neuron emits at most one spike per tick"
                    if self.deterministic_delivery else None
                ),
                "ring_consumption": (
                    "atomic_exchange" if self.deterministic_delivery else
                    "load_then_store" if self.backend == "triton" else
                    "backend_default"
                ),
            },
            "provenance": {
                "model": "MaleCNS LIF update order ported from scripts/lif_flyb.py",
                "source_label": SourceLabel.CONNECTOME.value,
                "short_term_plasticity": "unsupported in this engine; use lif_flyb.py",
                "triton_fused": "neuron/compact/deliver kernels; deterministic_delivery adds fixed-point finalization and optional graph step advance",
                "triton_atomic_delivery": "default ring delivery uses float32 atomic_add; accumulation order is not bit-deterministic across launches",
                "triton_deterministic_delivery": (
                    "opt-in exact int64 sum of stored float32 edge weights at a common power-of-two scale, "
                    "rounded once to float32 per target/tick; may differ from order-dependent default atomic rounding"
                ),
            },
            "capabilities": {name: value.to_dict() for name, value in self.capabilities.items()},
            "reports": {name: ids.size for name, ids in self.report_populations.items()},
            "triton_timing": {
                "warmup": _json_safe(self._triton_warmup_report),
                "recorded_steps": self._triton_recorded_steps,
                "first_recorded_step_wall_s": self._triton_recorded_first_wall_s,
                "steady_recorded_steps": self._triton_recorded_steady_steps,
                "steady_recorded_mean_wall_s": (
                    self._triton_recorded_steady_total_s / self._triton_recorded_steady_steps
                    if self._triton_recorded_steady_steps else None
                ),
                "kernel_launches": self._triton_kernel_launches,
                "host_device_syncs": self._triton_host_device_syncs,
                "host_scalar_reads": self._triton_host_scalar_reads,
                "input_h2d_copies": self._triton_input_h2d_copies,
                "pinned_input_staging": bool(
                    getattr(self, "_triton_pinned_input_staging", False)
                ),
                "pinned_input_reason": getattr(
                    self, "_triton_pinned_input_reason", None
                ),
                "cuda_graph_requested": self.cuda_graph_requested,
                "cuda_graph_enabled": self._triton_cuda_graph_enabled,
                "cuda_graph_reason": self._triton_cuda_graph_reason,
                "cuda_graph_cached_window_steps": sorted(self._triton_cuda_graphs),
                "cuda_graph_capture_count": self._triton_cuda_graph_capture_count,
                "cuda_graph_replay_count": self._triton_cuda_graph_replay_count,
                "cuda_graph_host_launches": self._triton_cuda_graph_host_launches,
                "cuda_graph_pointer_updates": self._triton_cuda_graph_pointer_updates,
                "cuda_graph_last_phase": self._triton_cuda_graph_last_phase,
                "cuda_graph_last_capture_wall_s": self._triton_cuda_graph_last_capture_wall_s,
                "cuda_graph_last_replay_wall_s": self._triton_cuda_graph_last_replay_wall_s,
                "startup_cost_visible": True,
            },
            "sensory_map": {
                "path": str(self.sensory_map_path),
                "sha256": self.sensory_map_hash,
                "input_units": MECHANOSENSORY_INPUT_UNITS,
                "output_units": MECHANOSENSORY_OUTPUT_UNITS,
                "input_range": [MECHANOSENSORY_INPUT_MIN, MECHANOSENSORY_INPUT_MAX],
                "max_gain_mV_per_ms": self.mechanosensory_max_gain_mV_per_ms,
                "entries": [spec.to_dict() for spec in self.mechanosensory_specs],
                "targets_resolved": {
                    spec.name: int(self._mechanosensory_targets[spec.name].size)
                    for spec in self.mechanosensory_specs
                    if spec.name in self._mechanosensory_targets
                },
                "target_status": dict(self._mechanosensory_target_status),
                "body_id_mapping": "not provided",
            },
        }


def create_engine(seed: int, device: str = "auto", backend: str = "auto", *,
                  csr_path: str | Path = DEFAULT_CSR, **kwargs: Any) -> MaleCNSEngine:
    """Create the reusable engine with guarded automatic backend selection.

    ``device='auto', backend='auto'`` selects ``torch_cuda`` only after the
    project's read-only CUDA gate is clear and Torch reports a usable CUDA
    device.  Otherwise it selects the dependency-light NumPy CPU engine.  A
    caller that explicitly requests ``device='cuda'`` receives a typed
    fail-closed error when the gate or CUDA runtime is unavailable; it is never
    silently downgraded to CPU.  ``backend='triton'`` is an explicit CUDA-only
    request and runs the fused neuron/compact/delivery kernels after the same
    gate check; it never downgrades to Torch or CPU.  ``csr_path`` and the
    remaining keyword arguments are intentionally passed through for tests and
    validated adapters.  The public factory's first three arguments remain the
    small ``seed/device/backend`` contract used by the digital-fly runner.
    """

    return MaleCNSEngine(csr_path=csr_path, seed=int(seed), device=device,
                         backend=backend, **kwargs)


__all__ = [
    "CSR", "DEFAULT_CSR", "DEFAULT_SENSORY_MAP", "CSRPreflight", "CSRValidationError", "DriveError", "EngineError",
    "GateClosedError", "MaleCNSEngine", "PopulationError", "PopulationIndex", "Populations",
    "MechanosensoryDriveSpec", "MAX_TRITON_WARMUP_STEPS", "MAX_TRITON_WINDOW_STEPS",
    "TimestampError", "UnsupportedBackendError", "create_engine", "discover_capabilities",
    "preflight_csr",
]
