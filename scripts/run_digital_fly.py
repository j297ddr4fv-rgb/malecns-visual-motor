#!/usr/bin/env python3
"""End-to-end digital-fly launcher.

The launcher is intentionally explicit about provenance.  Its default path is
CPU-only and deterministic::

    python3 scripts/run_digital_fly.py --out artifacts/digital-fly-smoke

The five configured scenarios are run through the same loop:

``VirtualWorld -> StereoFlyVis -> neural endpoint -> NeuromuscularAdapter
-> FlightAutopilot -> FlightDynamics -> world feedback -> MechanosensoryFeedback``

``stub`` and ``reduced`` are honest engineering substitutes.  They are never
labelled biological.  ``--neural malecns`` and ``--vision flyvis`` require an
explicitly discoverable real adapter; ``--body flybody`` requires a validated
project adapter.  Missing dependencies, an unsafe CUDA gate, or an unknown
adapter contract fail before a run starts and never silently fall back.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
import zlib

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "config" / "scenarios.json"
DEFAULT_FREE_ESCAPE_CONFIG = ROOT / "config" / "free_escape_experiment.json"
DEFAULT_OUT = ROOT / "artifacts" / "digital-fly"
SCENARIOS = ("takeoff", "hover", "yaw-step", "gust-recovery", "landing")
# ``turn`` is a readable public alias for the configured yaw-step scenario.
# Keep the historical scenario names in ``SCENARIOS`` so existing manifests
# and validator fixtures remain stable.
SCENARIO_ALIASES = {"turn": "yaw-step"}
# These source aliases are the complete body-derived drive contract for each
# ablation group.  Keeping the expected aliases here lets the launcher fail
# closed if a sensory-map edit adds a path that this intervention would miss.
MECHANOSENSORY_ABLATION_ENTRIES = {
    "jo": {
        "jo": ("jo_wind_gravity", ("jo", "jo_wind_gravity", "wind_gravity")),
    },
    "haltere": {
        "haltere": ("haltere", ("haltere",)),
    },
    "wing": {
        "wing": ("wing", ("wing", "wing_mechanosensory")),
        "wing_tactile": ("wing_tactile", ("wing_tactile",)),
        "wing_proprioceptive": ("wing_proprioceptive", ("wing_proprioceptive",)),
    },
    "leg": {
        "leg": ("leg", ("leg", "leg_mechanosensory")),
        "leg_tactile": ("leg_tactile", ("leg_tactile",)),
        "leg_proprioceptive": ("leg_proprioceptive", ("leg_proprioceptive",)),
    },
}
FLYVIS_DT_S = 0.005
FLYVIS_HZ = 200.0
CNS_HZ = 2000.0
CNS_DT_S = 1.0 / CNS_HZ
CNS_STEPS_PER_FLYVIS = int(round(FLYVIS_DT_S * CNS_HZ))
DEFAULT_MAX_VRAM_GIB = 15.5
# Telemetry is deliberately versioned independently from the run manifest.
# The compact schema is a row-level contract: immutable component/configuration
# provenance lives in ``manifest.json`` and only causal/scientific frame data
# remains in each row.  ``full-debug`` preserves the historical, self-contained
# records for short probes and forensic debugging.
COMPACT_TELEMETRY_SCHEMA_VERSION = "digital-fly-telemetry-compact-1"
FULL_DEBUG_TELEMETRY_SCHEMA_VERSION = "digital-fly-telemetry-full-debug-1"
COMPACT_TRAJECTORY_SCHEMA_VERSION = "digital-fly-trajectory-compact-1"
FULL_DEBUG_TRAJECTORY_SCHEMA_VERSION = "digital-fly-trajectory-full-debug-1"
MINIMAL_TELEMETRY_SCHEMA_VERSION = "digital-fly-telemetry-minimal-1"
MINIMAL_TRAJECTORY_SCHEMA_VERSION = "digital-fly-trajectory-minimal-1"
COMPACT_TELEMETRY_MODES = frozenset({"compact", "production", "strict"})
FULL_DEBUG_TELEMETRY_MODES = frozenset({"full", "debug", "full-debug", "full_debug"})
MINIMAL_TELEMETRY_MODES = frozenset({"minimal"})
CONTROL_SOURCE_SCALE = {
    "yaw_rate_rad_s": 1.0,
    "pitch_rate_rad_s": 0.0,
    "roll_rate_rad_s": 0.0,
    "thrust_mps2": 9.80665,
}

# The free-behavior/escape path is intentionally a separate execution policy,
# rather than a flag that merely changes a controller gain.  Keep the aliases
# here because external launchers historically used both spellings.
FREE_BEHAVIOR_MODE_NAMES = frozenset({
    "strict", "free", "free-behavior", "free_behavior",
    "escape-experiment", "escape_experiment", "escape",
})
FREE_BEHAVIOR_POLICY_NAME = "strict-free-behavior-escape-experiment"
FREE_BEHAVIOR_PROHIBITED_PATHS = (
    "mission_phases",
    "mission_goals",
    "FlightStateDrive",
    "FlightBodyStabilizer",
    "FlightAutopilot",
    "normal_behavior_autopilot_overlay",
    "stub_scripted_control",
    "stimulus_labels_as_control",
    "stimulus_direction_as_control",
    "future_state_as_control",
    "external_jump_posture_providers",
    "stimulus_geometry_as_privileged_channel",
)
FREE_BEHAVIOR_ALLOWED_CONTROL_PATH = (
    "MaleCNS",
    "NeuromuscularAdapter",
    "FlightDynamics",
)


@dataclass(frozen=True)
class _TimedStageResult:
    value: Any
    wall_s: float


@dataclass(frozen=True)
class _DeferredStageCall:
    function: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]


class _FrameStagePipeline:
    """Bounded frame-local overlap without changing the causal control path.

    Only work explicitly split into a causal-state-free ``prepare`` phase may
    overlap fitted vision.  Prepare owns bounded token bookkeeping but does
    not advance sensor history, timestamps, or RNG state.  Stateful sensory
    commit, neural integration, neuromuscular decoding, dynamics, and
    telemetry remain on the main thread.
    Keeping live row objects out of worker threads avoids mutable adapter and
    NumPy-buffer races.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._sensory_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="fly-sensory-prepare")
            if self.enabled else None
        )
        self.sensory_tasks = 0
        self.sensory_compute_s = 0.0
        self.sensory_wait_s = 0.0

    @staticmethod
    def _invoke(
        function: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> _TimedStageResult:
        started = time.perf_counter()
        value = function(*args, **dict(kwargs))
        return _TimedStageResult(value=value, wall_s=time.perf_counter() - started)

    def start_sensory_prepare(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[_TimedStageResult] | _DeferredStageCall:
        call_args = tuple(args)
        call_kwargs = dict(kwargs)
        if self._sensory_executor is None:
            return _DeferredStageCall(function, call_args, call_kwargs)
        self.sensory_tasks += 1
        return self._sensory_executor.submit(
            self._invoke, function, call_args, call_kwargs
        )

    def finish_sensory_prepare(
        self, job: Future[_TimedStageResult] | _DeferredStageCall
    ) -> tuple[Any, float, float]:
        wait_started = time.perf_counter()
        if isinstance(job, _DeferredStageCall):
            result = self._invoke(job.function, job.args, job.kwargs)
            wait_s = 0.0
        else:
            result = job.result()
            wait_s = time.perf_counter() - wait_started
            self.sensory_compute_s += result.wall_s
            self.sensory_wait_s += wait_s
        return result.value, result.wall_s, wait_s

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "digital-fly-frame-stage-pipeline-1",
            "requested": self.enabled,
            "effective": self.enabled,
            "sensory_prepare_worker_count": 1 if self.enabled else 0,
            "cross_frame_control_overlap": False,
            "stateful_sensory_commit_on_main_thread": True,
            "telemetry_async": False,
            "telemetry_async_reason": "live row objects are not an immutable worker boundary",
            "sensory_tasks": self.sensory_tasks,
            "sensory_prepare_compute_wall_s": self.sensory_compute_s,
            "sensory_prepare_wait_wall_s": self.sensory_wait_s,
        }

    def close(self) -> None:
        if self._sensory_executor is not None:
            self._sensory_executor.shutdown(wait=True, cancel_futures=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from twin_contracts import (  # noqa: E402
    BodyState as ContractBodyState,
    MotorCommand as ContractMotorCommand,
    NeuralOutput,
    SensoryFrame,
    SourceLabel,
    StepDiagnostics,
    Timestamp,
    VisualObservation,
    WorldState,
)


class LauncherError(RuntimeError):
    """Base class for setup/runtime failures."""


class BackendSelectionError(LauncherError):
    """Raised when an explicitly requested backend cannot be used."""


class CalibrationError(LauncherError):
    """Raised when an explicitly requested visual calibration is unusable."""


class RendererBackendError(LauncherError):
    """Raised when an explicitly requested world renderer cannot be used."""


class _RendererBackendWorldProxy:
    """Delegate world physics/configuration while replacing render methods.

    ``VirtualWorld`` remains the analytic/reference authority for wind,
    collisions, geometry, and state.  An optional Warp adapter only owns the
    two rendering seams used by this launcher.  Keeping the proxy here means
    no fallback or mutation of ``virtual_world.py`` is needed while the
    external ``world_renderer_opt`` implementation evolves.
    """

    backend_contract = "world-renderer-v1"
    is_stub = False

    def __init__(self, world: Any, renderer: Any, *, backend_name: str):
        self._world = world
        self._renderer = renderer
        self.backend_name = str(backend_name)
        self.backend_role = str(backend_name)
        self.renderer_backend = str(backend_name)
        self.renderer = renderer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._world, name)

    @staticmethod
    @lru_cache(maxsize=256)
    def _call_plan(method: Any) -> tuple[Any, bool, frozenset[str]]:
        """Cache signature/keyword filtering for repeated renderer calls."""

        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return None, True, frozenset()
        params = signature.parameters
        accepts_kwargs = any(
            item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values()
        )
        return signature, accepts_kwargs, frozenset(params)

    @staticmethod
    def _call(method: Any, *args: Any, **kwargs: Any) -> Any:
        if not callable(method):
            raise RendererBackendError("selected renderer backend exposes no callable render method")
        try:
            signature, accepts_kwargs, parameter_names = _RendererBackendWorldProxy._call_plan(method)
        except TypeError:
            # A rare unhashable callable cannot enter the cache; preserve the
            # historical introspection fallback for that adapter shape.
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                signature = None
            accepts_kwargs = bool(signature is None) or any(
                item.kind is inspect.Parameter.VAR_KEYWORD
                for item in signature.parameters.values()
            ) if signature is not None else True
            parameter_names = frozenset(signature.parameters) if signature is not None else frozenset()
        if signature is None:
            return method(*args, **kwargs)
        if accepts_kwargs:
            return method(*args, **kwargs)
        filtered = {
            key: value for key, value in kwargs.items()
            if key in parameter_names
        }
        return method(*args, **filtered)

    def render_stereo(self, pose: Any, resolution: Any = None, **kwargs: Any) -> Any:
        method = getattr(self._renderer, "render_stereo", None)
        if method is None:
            method = getattr(self._renderer, "render", None)
        if method is None:
            raise RendererBackendError(
                "warp_cuda renderer must expose render_stereo(pose, resolution, ...)"
            )
        return self._call(method, pose, resolution, **kwargs)

    def render_eye_directions(self, pose: Any, directions: Any, eye: str = "left",
                              **kwargs: Any) -> Any:
        # The optional Warp adapter may expose both a legacy host-returning
        # method and a tensor-preserving seam.  Prefer the latter so exact
        # columns can flow directly into FlyVisRuntimeAdapter.step_detailed;
        # only fall back to the legacy method when no tensor seam exists.
        method = getattr(self._renderer, "render_eye_directions_tensor", None)
        if method is None:
            method = getattr(self._renderer, "render_eye_directions", None)
        if method is None:
            method = getattr(self._renderer, "render_columns", None)
        if method is None:
            raise RendererBackendError(
                "warp_cuda renderer must expose render_eye_directions_tensor/render_eye_directions"
            )
        return self._call(method, pose, directions, eye=eye, **kwargs)

    def manifest(self, scenario: Any = None) -> dict[str, Any]:
        base = getattr(self._world, "manifest", None)
        result = base(scenario) if callable(base) else {}
        if not isinstance(result, Mapping):
            result = {}
        value = dict(result)
        renderer_manifest = getattr(self._renderer, "manifest", None)
        if callable(renderer_manifest):
            try:
                value["renderer"] = _json_safe(renderer_manifest())
            except Exception as exc:
                raise RendererBackendError(f"renderer manifest failed: {exc}") from exc
        value["renderer_backend"] = self.backend_name
        return value


def _construct_renderer_backend(module: Any, world: Any, *, backend: str,
                                device: str) -> Any:
    """Construct an external renderer using a narrow, explicit factory seam."""

    factories = (
        "create_renderer", "build_renderer", "create_world_renderer",
        "build_world_renderer", "create_backend", "build_backend",
    )
    for name in factories:
        factory = getattr(module, name, None)
        if not callable(factory):
            continue
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            signature = None
        requested = {
            "world": world,
            "backend": backend,
            "renderer_backend": backend,
            "device": device,
        }
        try:
            if signature is None:
                return factory(**requested)
            params = signature.parameters
            accepts_kwargs = any(
                item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values()
            )
            kwargs = requested if accepts_kwargs else {
                key: value for key, value in requested.items() if key in params
            }
            return factory(**kwargs)
        except Exception as exc:
            raise RendererBackendError(
                f"world_renderer_opt.{name} failed for {backend}: {exc}"
            ) from exc
    for name in ("WarpCudaRenderer", "WarpRenderer", "WorldRenderer", "Renderer"):
        candidate = getattr(module, name, None)
        if not callable(candidate):
            continue
        try:
            signature = inspect.signature(candidate)
            params = signature.parameters
            requested = {
                "world": world,
                "backend": backend,
                "renderer_backend": backend,
                "device": device,
            }
            accepts_kwargs = any(
                item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values()
            )
            kwargs = requested if accepts_kwargs else {
                key: value for key, value in requested.items() if key in params
            }
            return candidate(**kwargs)
        except Exception as exc:
            raise RendererBackendError(
                f"world_renderer_opt.{name} failed for {backend}: {exc}"
            ) from exc
    # A module-level renderer object is accepted only when it explicitly
    # publishes the stereo method and one exact-eye method (preferably the
    # tensor-preserving seam).  There is no analytic fallback here.
    if callable(getattr(module, "render_stereo", None)) and (
        callable(getattr(module, "render_eye_directions_tensor", None))
        or callable(getattr(module, "render_eye_directions", None))
        or callable(getattr(module, "render_columns", None))
    ):
        return module
    raise RendererBackendError(
        "world_renderer_opt exposes no supported renderer factory/object; "
        "warp_cuda cannot silently fall back to analytic_reference"
    )


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LauncherError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise LauncherError(f"{name} must be finite")
    return number


def _vec3(value: Iterable[float], name: str) -> tuple[float, float, float]:
    try:
        values = tuple(_finite(item, f"{name}[{i}]") for i, item in enumerate(value))
    except TypeError as exc:
        raise LauncherError(f"{name} must contain three finite values") from exc
    if len(values) != 3:
        raise LauncherError(f"{name} must contain three finite values")
    return values  # type: ignore[return-value]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def _json_safe(value: Any) -> Any:
    """Convert tuples, enums, arrays, and non-finite numbers for JSON."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return _json_safe(value.value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())
        except TypeError:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return repr(value)


def _is_device_tensor(value: Any) -> bool:
    """Recognize torch-like device tensors without importing an optional runtime.

    The fitted FlyVis adapter accepts tensors exposing ``device`` and
    ``detach``.  Keeping this check local lets Warp hand an exact eye vector
    directly to FlyVis while the analytic/reference renderer continues to use
    NumPy.  In particular, do not call ``np.asarray`` merely to inspect a
    candidate: that would turn a CUDA render into an implicit D2H transfer.
    """

    return getattr(value, "device", None) is not None and callable(
        getattr(value, "detach", None)
    )


def _host_float32_array(value: Any, name: str = "array") -> np.ndarray:
    """Materialize one small telemetry copy from a CPU/CUDA vector.

    This is intentionally used only after the fitted control adapter has
    consumed a device-resident vector.  The exact control path therefore
    remains device-resident; the returned host copy is for JSON/debug
    telemetry and lossless replay evidence.
    """

    candidate = value
    if _is_device_tensor(candidate):
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        numpy_method = getattr(candidate, "numpy", None)
        if callable(numpy_method):
            candidate = numpy_method()
    try:
        array = np.asarray(candidate, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise LauncherError(f"{name} is not a numeric vector") from exc
    return np.ascontiguousarray(array, dtype=np.float32)


def _array_sha256(value: Any) -> Optional[str]:
    """Hash one numeric array without converting it to a Python list.

    Compact telemetry uses hashes for omitted/invariant buffers.  Keeping the
    conversion here (rather than in a caller) makes the byte order and shape
    part of one auditable convention across FlyVis, neural, and mapping data.
    """

    try:
        array = np.ascontiguousarray(np.asarray(value))
    except (TypeError, ValueError):
        return None
    if array.ndim == 0:
        array = array.reshape(1)
    try:
        return hashlib.sha256(array.tobytes(order="C")).hexdigest()
    except (TypeError, ValueError):
        return None


def _lossless_array_record(value: Any, *, include_data: bool = True,
                           sha256: Optional[str] = None) -> dict[str, Any]:
    """Describe a numeric array, optionally carrying a lossless byte stream.

    JSON lists are surprisingly expensive for the 13k-element fitted T4/T5
    vector and are easy to duplicate accidentally.  Compact rows therefore
    carry the exact float32/float64 bytes once as zlib+base64.  This is not a
    lossy quantisation: ``sha256`` covers the uncompressed C-order bytes and
    ``compressed_sha256`` covers the transport payload.  Manifest-only array
    records can omit ``data`` while retaining shape/dtype/hash provenance.
    """

    try:
        array = np.ascontiguousarray(np.asarray(value))
    except (TypeError, ValueError) as exc:
        return {
            "encoding": "unavailable",
            "lossless": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if array.ndim == 0:
        array = array.reshape(1)
    record: dict[str, Any] = {
        "encoding": "zlib+base64" if include_data else "hash-only",
        "lossless": True,
        "dtype": str(array.dtype),
        "shape": [int(size) for size in array.shape],
        "bytes": int(array.nbytes),
        "sha256": _array_sha256(array) if sha256 is None else str(sha256),
    }
    if include_data:
        raw = array.tobytes(order="C")
        compressed = zlib.compress(raw, level=6)
        record.update({
            "compressed_bytes": int(len(compressed)),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "data": base64.b64encode(compressed).decode("ascii"),
        })
    return record


def _required_hash_only_array_record(
    value: Any, *, name: str, sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Return a strict numeric-vector fingerprint or fail the run.

    Minimal production telemetry is the only retained evidence for the exact
    eye columns and fitted T4/T5 drive.  An ``unavailable`` record therefore
    is not a useful fallback: publishing it would make a nominally successful
    run impossible to audit.  Keep this stricter contract separate from the
    permissive debug serializer used by older/full-debug paths.
    """

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise LauncherError(f"{name} cannot be converted to a numeric vector") from exc
    if array.ndim != 1 or array.size <= 0:
        raise LauncherError(
            f"{name} must be a non-empty vector, got shape {tuple(array.shape)}"
        )
    array = np.ascontiguousarray(array)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise LauncherError(f"{name} must have a real numeric dtype, got {array.dtype}")
    try:
        finite = bool(np.isfinite(array).all())
    except TypeError as exc:
        raise LauncherError(f"{name} cannot be checked for finite values") from exc
    if not finite:
        raise LauncherError(f"{name} contains non-finite values")

    if sha256 is None:
        digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    else:
        digest = str(sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise LauncherError(f"{name} has an invalid precomputed SHA-256 digest")
    return {
        "encoding": "hash-only",
        "lossless": True,
        "dtype": str(array.dtype),
        "shape": [int(size) for size in array.shape],
        "bytes": int(array.nbytes),
        "sha256": digest,
    }


def _compact_json_value(value: Any, *, include_array_data: bool = False) -> Any:
    """Bound unknown adapter metadata while retaining scalar scientific data.

    Known population maps are ordinary mappings and stay readable.  Unknown
    dense arrays are represented by a lossless hash/shape record (or by the
    exact compressed bytes when explicitly requested), preventing a plugin's
    diagnostic payload from silently defeating the compact contract.
    """

    if value is None or isinstance(value, (str, int, bool, float)):
        return _json_safe(value)
    if isinstance(value, Mapping):
        return {
            str(key): _compact_json_value(item, include_array_data=include_array_data)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        # Preserve short scalar vectors as normal JSON for readability.  Long
        # numeric vectors use the same exact transport as NumPy arrays.
        if len(value) > 64:
            try:
                array = np.asarray(value)
                if array.dtype.kind in "biuf":
                    return _lossless_array_record(array, include_data=include_array_data)
            except (TypeError, ValueError):
                pass
        return [_compact_json_value(item, include_array_data=include_array_data) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _lossless_array_record(value, include_data=include_array_data)
        except Exception:
            pass
    return _json_safe(value)


def _optional_manifest(value: Any) -> Any:
    """Read an adapter manifest when present without inventing one."""

    method = getattr(value, "manifest", None)
    if not callable(method):
        return None
    try:
        return _json_safe(method())
    except Exception as exc:
        raise LauncherError(f"component manifest failed: {exc}") from exc


def _sha256_file(path: Path) -> Optional[str]:
    """Return a content hash for provenance without making hashing mandatory."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _canonical_scenario_name(name: str) -> str:
    """Resolve public scenario aliases while preserving the requested name."""

    value = str(name)
    return SCENARIO_ALIASES.get(value, value)


def _cgroup_memory_limit_bytes() -> Optional[int]:
    """Return the effective cgroup-v2 memory ceiling when one is present."""

    path = Path("/sys/fs/cgroup/memory.max")
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            if line.startswith("0::"):
                relative = line.split("::", 1)[1].lstrip("/")
                path = Path("/sys/fs/cgroup") / relative / "memory.max"
                break
    except OSError:
        pass
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _runtime_resource_usage(cuda_device: str = "cuda:0") -> dict[str, Any]:
    """Capture in-process peaks; systemd user scopes under-report on this host."""

    usage: dict[str, Any] = {
        # Linux reports ru_maxrss in KiB.
        "peak_host_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                usage["current_host_rss_bytes"] = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    torch = sys.modules.get("torch")
    if torch is None or str(cuda_device).strip().lower() == "cpu":
        return usage
    try:
        if not bool(torch.cuda.is_available()):
            return usage
        device = torch.device(cuda_device)
        torch.cuda.synchronize(device)
        usage["cuda"] = {
            "device": str(device),
            "total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    except Exception as exc:
        usage["cuda_measurement_error"] = f"{type(exc).__name__}: {exc}"
    return usage


def _module_capability(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    installed = spec is not None
    version = None
    if installed:
        try:
            version = importlib.metadata.version(module_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        except Exception:
            version = None
    return {
        "module": module_name,
        "installed": installed,
        "version": version,
        "validated": False,
    }


def _require_module(module_name: str, purpose: str) -> Any:
    capability = _module_capability(module_name)
    if not capability["installed"]:
        raise BackendSelectionError(
            f"requested {purpose} backend requires optional module {module_name!r}; "
            "it is not installed, so the launcher refuses to substitute a stub"
        )
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise BackendSelectionError(
            f"requested {purpose} backend {module_name!r} could not be imported: {exc}"
        ) from exc


def _timestamp(t_s: float, *, clock: str = "neural", rate_hz: float = CNS_HZ) -> Timestamp:
    rate = _finite(rate_hz, "rate_hz")
    if rate <= 0:
        raise LauncherError("timestamp rate_hz must be positive")
    t = _finite(t_s, "timestamp.t_s")
    tick = int(round(t * rate))
    if abs(t - tick / rate) > 1.0e-8:
        raise LauncherError(f"time {t:g}s is not aligned to {clock} at {rate:g} Hz")
    return Timestamp(t, tick, clock, rate)


def _quat_from_rpy(rpy: Sequence[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = _vec3(rpy, "orientation_rad")
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _rpy_from_quat(quaternion: Sequence[float]) -> tuple[float, float, float]:
    q = _vec4(quaternion, "orientation_quat")
    w, x, y, z = q
    norm = math.sqrt(sum(item * item for item in q))
    if norm <= 1.0e-15:
        raise LauncherError("orientation_quat must not be zero")
    w, x, y, z = (item / norm for item in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _vec4(value: Sequence[float], name: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(_finite(item, f"{name}[{i}]") for i, item in enumerate(value))
    except TypeError as exc:
        raise LauncherError(f"{name} must contain four finite values") from exc
    if len(values) != 4:
        raise LauncherError(f"{name} must contain four finite values")
    return values  # type: ignore[return-value]


def _make_contract_body(state: Any, timestamp: Timestamp) -> ContractBodyState:
    """Convert FlightState-like data to one timestamp-consistent contract."""

    position = _vec3(getattr(state, "position_m"), "position_m")
    velocity = _vec3(getattr(state, "velocity_mps"), "velocity_mps")
    angular = _vec3(
        getattr(state, "angular_velocity_rad_s", getattr(state, "angular_velocity_radps", (0.0, 0.0, 0.0))),
        "angular_velocity_rad_s",
    )
    orientation = getattr(state, "orientation_quat", None)
    if orientation is not None:
        rpy = _rpy_from_quat(orientation)
    else:
        rpy = _vec3(getattr(state, "orientation_rad", (0.0, 0.0, 0.0)), "orientation_rad")
    return ContractBodyState(
        timestamp=timestamp,
        position_m=position,
        orientation_rad=rpy,
        velocity_mps=velocity,
        angular_velocity_rad_s=angular,
        source=SourceLabel.HANDBUILT,
    )


def _pose_from_state(state: Any) -> Any:
    from virtual_world import Pose

    rpy = _rpy_from_quat(getattr(state, "orientation_quat"))
    return Pose(
        position_m=_vec3(getattr(state, "position_m"), "position_m"),
        yaw_rad=rpy[2],
        pitch_rad=rpy[1],
        roll_rad=rpy[0],
        velocity_mps=_vec3(getattr(state, "velocity_mps"), "velocity_mps"),
        angular_velocity_radps=_vec3(
            getattr(state, "angular_velocity_rad_s", getattr(state, "angular_velocity_radps", (0.0, 0.0, 0.0))),
            "angular_velocity_radps",
        ),
    )


def _scenario_goal(scenario: Any, t_s: float) -> tuple[tuple[float, float, float], float]:
    """Small public-config profile evaluator used for control/telemetry."""

    initial = scenario.initial_pose
    profile = dict(scenario.profile)
    kind = str(profile.get("kind", "hold")).lower()
    goal = tuple(initial.position_m)
    yaw = initial.yaw_rad
    if kind in {"hold", "hover"}:
        goal = tuple(profile.get("position_m", goal))
        yaw = float(profile.get("yaw_rad", yaw))
    elif kind in {"vertical_ramp", "ramp", "forward_ramp", "position_ramp", "translation_ramp"}:
        start = tuple(profile.get("start_position_m", profile.get("start_m", goal)))
        end = tuple(profile.get("end_position_m", profile.get("end_m", profile.get("position_m", goal))))
        if kind in {"vertical_ramp", "ramp"}:
            start = (goal[0], goal[1], float(profile.get("start_z_m", start[2])))
            end = (goal[0], goal[1], float(profile.get("end_z_m", end[2])))
        duration = max(1.0e-12, float(profile.get("duration_s", scenario.duration_s)))
        alpha = _clamp(t_s / duration)
        goal = tuple(start[i] + alpha * (end[i] - start[i]) for i in range(3))
    elif kind == "yaw_step":
        at = max(0.0, float(profile.get("step_at_s", profile.get("start_s", 0.0))))
        rate = float(profile.get("yaw_rate_radps", profile.get("yaw_rate_rad_s", 0.0)))
        yaw = initial.yaw_rad + max(0.0, t_s - at) * rate
    return _vec3(goal, "scenario_goal.position_m"), _finite(yaw, "scenario_goal.yaw_rad")


def _desired_stub_control(scenario_name: str, t_s: float, altitude_m: float,
                         visual_direction: float) -> tuple[float, float]:
    """Deterministic fitted-readout substitute for CPU smoke runs."""

    name = scenario_name.replace("_", "-")
    hover = 0.7063531679829521
    if name == "takeoff":
        power = 1.0 if t_s < 0.55 else 0.78
    elif name == "landing":
        power = hover if t_s < 0.65 else max(0.0, hover * (1.0 - (t_s - 0.65) / 0.75))
    else:
        power = hover
    steering = 0.0
    if name == "yaw-step" and t_s >= 0.40:
        steering = 0.45
    if name == "gust-recovery":
        steering = -0.18 * visual_direction
    # Keep the stub bounded even if a custom scene starts far outside its
    # nominal envelope.  Altitude is observed, never used as a hidden oracle.
    if name == "takeoff" and altitude_m > 0.55:
        power = min(power, hover)
    return _clamp(power), max(-1.0, min(1.0, steering + 0.12 * visual_direction))


def _population_rates(power: float, steering: float) -> dict[str, float]:
    """Encode a stub readout using the real adapter's named populations."""

    left = _clamp(power - steering)
    right = _clamp(power + steering)
    return {
        "prefix:DNg02": power * 50.0,
        "prefix:DNg02/L": left * 50.0,
        "prefix:DNg02/R": right * 50.0,
        "DNp03/L": 0.0,
        "DNp03/R": 0.0,
    }


def _target_info(world: Any, pose: Any, scenario: Any, contact: bool,
                 t_s: float = 0.0) -> dict[str, Any]:
    observation = world.target_observation(pose)
    goal_at = getattr(world, "goal_at", None)
    if callable(goal_at):
        try:
            reference = goal_at(scenario, t_s)
        except (KeyError, TypeError, ValueError):
            reference = None
    else:
        reference = None
    if isinstance(reference, Mapping):
        goal = _vec3(reference.get("position_m", scenario.initial_pose.position_m), "scenario_goal.position_m")
        goal_yaw = _finite(reference.get("yaw_rad", scenario.initial_pose.yaw_rad), "scenario_goal.yaw_rad")
    else:
        goal, goal_yaw = _scenario_goal(scenario, t_s)
    phase_label = str(getattr(scenario, "name", "")).strip().lower()
    phase_at = getattr(world, "phase_at", None)
    if callable(phase_at):
        try:
            phase_info = phase_at(scenario, t_s)
        except (KeyError, TypeError, ValueError):
            phase_info = None
        if isinstance(phase_info, Mapping):
            phase_label = str(
                phase_info.get("label", phase_info.get("phase", phase_label))
            ).strip().lower()
    landing_enabled = phase_label in {
        "approach", "landing", "land", "settle", "descent", "touchdown",
    }
    approach_distance = float(scenario.acceptance.get("approach_distance", 2.0))
    landing_distance = float(scenario.acceptance.get("landing_distance", 0.35))
    landing_altitude = float(scenario.acceptance.get("landing_altitude", 0.15))
    if not landing_enabled:
        # A physical landing target can remain visible throughout a mission;
        # visibility alone must not put a taking-off fly into APPROACH/LAND.
        approach_distance = -1.0
        landing_distance = -1.0
        landing_altitude = -1.0
    return {
        "target_altitude": goal[2],
        "target_heading": goal_yaw,
        "target_distance": observation.distance_m,
        "ground_altitude": world.config.ground_z_m + world.config.body_radius_m,
        "approach_distance": approach_distance,
        "landing_distance": landing_distance,
        "landing_altitude": landing_altitude,
        "contact": bool(contact),
        "mission_phase": phase_label,
    }


def _acceptance(acceptance: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    violations: list[str] = []
    for key, requested in acceptance.items():
        if key == "gust_start_s":
            passed = bool(metrics.get("gust_observed", False))
            check = {"observed": passed, "requested_s": float(requested), "passed": passed}
        elif str(key).startswith("max_"):
            metric = key if key in metrics else key[4:]
            observed = float(metrics.get(metric, math.inf))
            limit = float(requested)
            check = {"observed": observed, "limit": limit, "passed": observed <= limit + 1.0e-9}
        elif str(key).startswith("min_"):
            metric = key if key in metrics else key[4:]
            observed = float(metrics.get(metric, -math.inf))
            limit = float(requested)
            check = {"observed": observed, "limit": limit, "passed": observed >= limit - 1.0e-9}
        elif key == "final_speed_mps":
            observed = float(metrics.get("final_speed_mps", math.inf))
            limit = float(requested)
            check = {"observed": observed, "limit": limit, "passed": observed <= limit + 1.0e-9}
        else:
            check = {"observed": metrics.get(key), "requested": requested, "passed": True, "ignored": True}
        checks[key] = check
        if not check["passed"]:
            violations.append(str(key))
    return {"passed": not violations, "checks": checks, "violations": violations}


class StubNeural:
    backend_name = "deterministic_neural_stub"
    is_stub = True
    source = SourceLabel.HANDBUILT

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.calls = 0
        self.power = 0.0
        self.steering = 0.0

    def reset(self) -> None:
        self.calls = 0
        self.power = 0.0
        self.steering = 0.0

    def set_control(self, power: float, steering: float) -> None:
        self.power = _clamp(power)
        self.steering = max(-1.0, min(1.0, float(steering)))

    def step(self, sensory: SensoryFrame, timestamp: Timestamp) -> NeuralOutput:
        self.calls += 1
        rates = [self.power * 50.0, max(0.0, self.steering) * 50.0,
                 max(0.0, -self.steering) * 50.0, float(self.calls % 2)]
        named = {
            "population_rates": _population_rates(self.power, self.steering),
            "backend": self.backend_name,
            "call_index": self.calls,
        }
        return NeuralOutput(timestamp, rates_hz=rates,
                            spikes=[int(rate >= 25.0) for rate in rates],
                            named_outputs=named, source=self.source)


class MalecnsNeuralAdapter:
    """Strict wrapper for a project-provided ``malecns_engine`` module."""

    backend_name = "malecns_engine"
    is_stub = False
    source = SourceLabel.CONNECTOME

    def __init__(self, module: Any, *, seed: int, device: str = "auto",
                 backend: str = "auto", max_vram_gib: float = DEFAULT_MAX_VRAM_GIB,
                 csr_path: Optional[str | Path] = None,
                 flyvis_units: Optional[str] = None,
                 report_populations: Optional[Mapping[str, str]] = None,
                 neural_gain: Optional[float] = None,
                 cuda_graph: bool = False,
                 include_flyvis_drive_echo: bool = True):
        self.module = module
        self.requested_device = str(device)
        self.requested_backend = str(backend)
        self.cuda_graph_requested = bool(cuda_graph)
        requested_drive_echo = bool(include_flyvis_drive_echo)
        self.include_flyvis_drive_echo = requested_drive_echo
        self.csr_path = None if csr_path is None else Path(csr_path)
        self.flyvis_units = flyvis_units
        self.report_populations = None if report_populations is None else dict(report_populations)
        self.neural_gain = None if neural_gain is None else _finite(neural_gain, "neural_gain")
        if self.neural_gain is not None and self.neural_gain <= 0.0:
            raise BackendSelectionError("neural_gain must be positive")
        self.engine = self._construct(
            module, seed, device=device, backend=backend,
            max_vram_gib=max_vram_gib, csr_path=csr_path,
            flyvis_units=flyvis_units,
            report_populations=report_populations,
            neural_gain=self.neural_gain,
            cuda_graph=self.cuda_graph_requested,
            include_flyvis_drive_echo=bool(include_flyvis_drive_echo),
        )
        effective_drive_echo = getattr(
            self.engine, "include_flyvis_drive_echo", requested_drive_echo,
        )
        self.include_flyvis_drive_echo = bool(effective_drive_echo)
        if not requested_drive_echo and self.include_flyvis_drive_echo:
            raise BackendSelectionError(
                "minimal telemetry requested suppression of the dense FlyVis "
                "input echo, but the selected MaleCNS engine did not honor it"
            )
        # Keep the concrete selection visible to manifests and callers.  The
        # stable adapter name remains available when a third-party engine does
        # not expose backend/device metadata.
        self.backend_name = str(getattr(self.engine, "backend", self.backend_name))
        self.device_name = str(getattr(self.engine, "device", "unknown"))
        self.effective_max_vram_gib = getattr(self.engine, "max_vram_gib", max_vram_gib)
        if self.requested_device != "auto":
            expected_device = "cuda" if self.requested_device == "cuda" else "cpu"
            if self.device_name not in {expected_device, f"{expected_device}:0"}:
                raise BackendSelectionError(
                    f"malecns engine selected device {self.device_name!r}, "
                    f"but requested {self.requested_device!r}; refusing fallback"
                )
        if self.requested_backend != "auto":
            backend_aliases = {
                "cpu": {"cpu", "numpy_cpu"},
                "torch_cpu": {"torch_cpu", "torch"},
                "torch": {"torch", "torch_cuda"},
                "triton": {"triton", "triton_cuda", "triton_cpu"},
            }
            allowed = backend_aliases.get(self.requested_backend, {self.requested_backend})
            if self.backend_name not in allowed:
                raise BackendSelectionError(
                    f"malecns engine selected backend {self.backend_name!r}, "
                    f"but requested {self.requested_backend!r}; refusing fallback"
                )
        self.calls = 0
        self.window_calls = 0
        self.window_api = "scalar_step"
        self.warmup_report: Optional[Mapping[str, Any]] = None
        self._window_method = (
            getattr(self.engine, "step_window", None)
            or getattr(self.engine, "run_window", None)
        )
        self._window_steps_keyword = True
        if callable(self._window_method):
            try:
                window_signature = inspect.signature(self._window_method)
            except (TypeError, ValueError):
                window_signature = None
            if window_signature is not None:
                step_parameter = window_signature.parameters.get("steps")
                self._window_steps_keyword = bool(
                    step_parameter is not None
                    and step_parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
                )

    @staticmethod
    def _construct(module: Any, seed: int, *, device: str = "auto",
                   backend: str = "auto",
                   max_vram_gib: float = DEFAULT_MAX_VRAM_GIB,
                   csr_path: Optional[str | Path] = None,
                   flyvis_units: Optional[str] = None,
                   report_populations: Optional[Mapping[str, str]] = None,
                   neural_gain: Optional[float] = None,
                   cuda_graph: bool = False,
                   include_flyvis_drive_echo: bool = True) -> Any:
        requested = {
            "seed": int(seed),
            "device": str(device),
            "backend": str(backend),
            "max_vram_gib": max_vram_gib,
            "csr_path": csr_path,
            "flyvis_units": flyvis_units,
            "report_populations": report_populations,
            # MaleCNSEngine calls this parameter ``gain``.  Keep the public
            # runner spelling ``neural_gain`` while passing the calibrated
            # engine keyword only when explicitly requested.
            "gain": neural_gain,
            "cuda_graph": bool(cuda_graph),
            "include_flyvis_drive_echo": bool(include_flyvis_drive_echo),
        }

        def invoke(candidate: Any) -> Any:
            try:
                signature = inspect.signature(candidate)
            except (TypeError, ValueError):
                signature = None
            if signature is None:
                # Unknown signatures are not permission to drop the explicit
                # device/backend/resource contract.  Try the complete call
                # once and surface the original error if it rejects it.
                return candidate(**requested)
            params = signature.parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in params.values()
            )
            if accepts_kwargs:
                return candidate(**{key: value for key, value in requested.items()
                                    if value is not None})
            # Keep compatibility with the tiny CPU factory used by tests, but
            # never remove an explicitly requested core selection.  Optional
            # metadata is passed whenever the factory advertises it.
            for key in ("device", "backend"):
                if key not in params and requested[key] not in {"auto", ""}:
                    raise BackendSelectionError(
                        f"malecns engine factory does not expose requested {key}={requested[key]!r}"
                    )
            if neural_gain is not None and "gain" not in params:
                raise BackendSelectionError(
                    "malecns engine factory does not expose requested neural_gain/gain"
                )
            if cuda_graph and "cuda_graph" not in params:
                raise BackendSelectionError(
                    "malecns engine factory does not expose requested cuda_graph=True"
                )
            if csr_path is not None and "csr_path" not in params:
                raise BackendSelectionError(
                    "malecns engine factory does not expose the requested csr_path"
                )
            kwargs = {key: value for key, value in requested.items()
                      if key in params and value is not None}
            try:
                return candidate(**kwargs)
            except TypeError as exc:
                # A TypeError raised by the actual factory is not a signal to
                # retry with fewer arguments: doing that can silently select a
                # different device/backend or discard the drive mapping.
                raise BackendSelectionError(
                    f"malecns engine factory rejected its explicit contract: {exc}"
                ) from exc

        factories = ("create_engine", "build_engine", "create_model", "build_model")
        for name in factories:
            factory = getattr(module, name, None)
            if callable(factory):
                return invoke(factory)
        for name in ("MalecnsEngine", "MaleCNSEngine", "Engine", "Model"):
            cls = getattr(module, name, None)
            if callable(cls):
                return invoke(cls)
        raise BackendSelectionError(
            "malecns_engine is installed but exposes no supported factory "
            "(create_engine/build_engine/MalecnsEngine); refusing a silent stub"
        )

    def reset(self) -> None:
        reset = getattr(self.engine, "reset", None)
        if callable(reset):
            reset()
        self.calls = 0
        self.window_calls = 0

    def warmup(self, steps: int = 1) -> dict[str, Any]:
        """Warm optional engine kernels once and retain an auditable report."""

        method = getattr(self.engine, "warmup", None) or getattr(self.engine, "warm_kernels", None)
        if not callable(method):
            report = {
                "performed": False,
                "backend": self.backend_name,
                "requested_steps": int(steps),
                "completed_steps": 0,
                "reason": "engine exposes no warmup/warm_kernels method",
            }
            self.warmup_report = report
            return dict(report)
        try:
            raw = method(steps=steps)
        except TypeError:
            # Keep compatibility with a validated engine exposing a positional
            # warmup count, without retrying after a runtime/kernel failure.
            raw = method(steps)
        except Exception as exc:
            raise LauncherError(f"malecns engine warmup failed: {exc}") from exc
        report = dict(raw) if isinstance(raw, Mapping) else {
            "performed": True, "backend": self.backend_name,
            "requested_steps": int(steps), "result": _json_safe(raw),
        }
        self.warmup_report = _json_safe(report)
        return dict(self.warmup_report)

    def cuda_graph_manifest(self) -> dict[str, Any]:
        """Return explicit CUDA-Graph request/effective/fallback evidence."""

        engine = self.engine
        effective_raw = getattr(engine, "_triton_cuda_graph_enabled", None)
        reason = getattr(engine, "_triton_cuda_graph_reason", None)
        if self.cuda_graph_requested and effective_raw is None and reason is None:
            reason = "engine exposes no CUDA Graph status contract"
        return _json_safe({
            "requested": bool(self.cuda_graph_requested),
            "effective": bool(effective_raw),
            "reason": reason,
            "backend": self.backend_name,
            "device": self.device_name,
            "capture_count": int(getattr(engine, "_triton_cuda_graph_capture_count", 0)),
            "replay_count": int(getattr(engine, "_triton_cuda_graph_replay_count", 0)),
            "phase": getattr(engine, "_triton_cuda_graph_last_phase", "disabled"),
            "host_launches": int(getattr(engine, "_triton_cuda_graph_host_launches", 0)),
            "pointer_updates": int(getattr(engine, "_triton_cuda_graph_pointer_updates", 0)),
        })

    def step(self, sensory: SensoryFrame, timestamp: Timestamp) -> NeuralOutput:
        method = getattr(self.engine, "step", None) or getattr(self.engine, "infer", None)
        if not callable(method):
            raise LauncherError("malecns_engine adapter has no step/infer method")
        self.calls += 1
        output = method(sensory, timestamp)
        if isinstance(output, NeuralOutput):
            return output
        if isinstance(output, Mapping):
            rates = output.get("rates_hz", output.get("rates"))
            spikes = output.get("spikes")
            named = output.get("named_outputs", output.get("channels", {}))
            if rates is None and spikes is None and not isinstance(named, Mapping):
                raise LauncherError("malecns_engine output is not a NeuralOutput-compatible mapping")
            return NeuralOutput(timestamp, rates_hz=rates, spikes=spikes,
                                named_outputs=dict(named or {}), source=SourceLabel.CONNECTOME)
        raise LauncherError("malecns_engine output must be NeuralOutput or a compatible mapping")

    def step_window(self, sensory: SensoryFrame, timestamp: Timestamp,
                    steps: int = CNS_STEPS_PER_FLYVIS) -> Optional[NeuralOutput]:
        """Prefer the engine's bounded held-input window when available.

        The engine owns the ten exact CNS ticks and any device-side
        aggregation.  A validated legacy/mock engine without this API returns
        ``None`` so the runner can use its explicit scalar-step fallback.
        """

        method = self._window_method
        if not callable(method):
            return None
        try:
            if self._window_steps_keyword:
                raw = method(sensory, timestamp, steps=steps)
            else:
                raw = method(sensory, timestamp, steps)
        except Exception as exc:
            raise LauncherError(f"malecns engine window step failed: {exc}") from exc
        if isinstance(raw, Mapping):
            rates = raw.get("rates_hz", raw.get("rates"))
            spikes = raw.get("spikes")
            named = raw.get("named_outputs", raw.get("channels", {}))
            if rates is None and spikes is None and not isinstance(named, Mapping):
                raise LauncherError("malecns engine window output is not NeuralOutput-compatible")
            raw = NeuralOutput(timestamp, rates_hz=rates, spikes=spikes,
                               named_outputs=dict(named or {}), source=SourceLabel.CONNECTOME)
        if not isinstance(raw, NeuralOutput):
            raise LauncherError("malecns engine window output must be NeuralOutput or a compatible mapping")
        named = dict(raw.named_outputs)
        named.setdefault("neural_window_steps", int(steps))
        named.setdefault("neural_tick_dt_s", float(CNS_DT_S))
        named.setdefault("neural_window_dt_s", float(steps) * CNS_DT_S)
        named.setdefault("neural_window_start_s", float(timestamp.t_s))
        named.setdefault("neural_window_end_s", float(timestamp.t_s) + float(steps) * CNS_DT_S)
        named["neural_window_api"] = "engine_step_window"
        self.calls += 1
        self.window_calls += 1
        self.window_api = "engine_step_window"
        # The engine returns the final half-millisecond timestamp.  The outer
        # contract is one 5 ms control-frame sample, so retimestamp the
        # aggregate while retaining exact start/end values in named outputs.
        # Newer contracts can retimestamp without revalidating/copying dense
        # vectors.  Keep the constructor fallback for older test adapters.
        with_timestamp = getattr(raw, "with_timestamp", None)
        if callable(with_timestamp):
            try:
                return with_timestamp(timestamp, named_outputs=named)
            except TypeError:
                retimed = with_timestamp(timestamp)
                if isinstance(getattr(retimed, "named_outputs", None), Mapping):
                    retimed.named_outputs.update(named)
                return retimed
        return NeuralOutput(timestamp, rates_hz=raw.rates_hz, spikes=raw.spikes,
                            named_outputs=named, source=raw.source)


def _substep_sensory(frame: SensoryFrame, timestamp: Timestamp) -> SensoryFrame:
    """Retimestamp a held outer-frame sensory bundle for one CNS tick.

    FlyVis and mechanosensory values are sampled once per 5 ms control frame.
    The real MaleCNS engine integrates at 0.5 ms, so each substep receives the
    same values with only the contract timestamp advanced.  ``replace`` keeps
    visual/body payloads intact while satisfying ``SensoryFrame``'s strict
    timestamp equality checks.
    """

    visual = None if frame.visual is None else replace(frame.visual, timestamp=timestamp)
    body = None if frame.body is None else replace(frame.body, timestamp=timestamp)
    return SensoryFrame(timestamp=timestamp, visual=visual, body=body,
                        channels=frame.channels, source=frame.source)


def _flyvis_drive_contract(frontend: Any, output: Any,
                           mapping_override: Any = None) -> dict[str, Any]:
    """Expose a FlyVis vector together with its explicit CNS target mapping.

    ``StereoFlyVis`` deliberately keeps its response backend generic.  The
    fitted ``flyvis_runtime_adapter`` therefore publishes mapping metadata on
    the callable/backend (and on ``last_output``), while the stereo output
    carries only the signed vector.  Resolve both shapes here and preserve an
    incomplete mapping so the MaleCNS engine raises a clear ``DriveError``
    instead of silently injecting into the wrong neurons.
    """

    # The fitted runtime has already produced an owned contiguous host vector
    # at this boundary.  MaleCNS only reads it, so a second per-frame copy has
    # no safety or causal value.
    drive = np.ascontiguousarray(np.asarray(output, dtype=np.float32))
    backend = getattr(frontend, "backend", None)
    response = getattr(backend, "response_fn", None)
    model = getattr(response, "model", None)
    candidates = (
        mapping_override,
        output,
        getattr(output, "metadata", None),
        backend,
        response,
        model,
        getattr(response, "last_output", None),
        getattr(model, "last_output", None),
    )

    target_indices: Any = None
    units: Any = None
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            if target_indices is None:
                target_indices = candidate.get("target_indices", candidate.get("flyvis_target_indices"))
            if units is None:
                units = candidate.get("units", candidate.get("flyvis_units"))
        else:
            if target_indices is None:
                value = getattr(candidate, "target_indices", None)
                if value is None:
                    value = getattr(candidate, "target_drive_indices", None)
                if value is None:
                    method = getattr(candidate, "target_indices_array", None)
                    value = method() if callable(method) else None
                target_indices = value
            if units is None:
                units = getattr(candidate, "units", getattr(candidate, "target_units", None))

    target_array: Any = None
    if target_indices is not None:
        candidate = np.asarray(target_indices)
        # Cacheable mappings must be genuinely owned and immutable.  A
        # read-only view of a writable base is not sufficient because another
        # alias could still mutate it behind the engine's validation cache.
        if (
            isinstance(target_indices, np.ndarray)
            and candidate is target_indices
            and candidate.flags.c_contiguous
            and not candidate.flags.writeable
            and candidate.base is None
        ):
            target_array = candidate
        else:
            target_array = np.ascontiguousarray(candidate).copy()
            target_array.setflags(write=False)

    contract: dict[str, Any] = {
        "t4t5_drive": drive,
        "flyvis_target_indices": target_array,
        "flyvis_units": units,
    }
    if target_array is not None:
        contract["flyvis_t4t5"] = {
            "target_indices": target_array,
            "values": drive,
            "units": units,
        }
    return contract


def _aggregate_neural_outputs(outputs: Sequence[NeuralOutput], timestamp: Timestamp,
                              *, window_dt_s: float) -> NeuralOutput:
    """Collapse fixed-rate CNS outputs into one control-frame window output.

    A per-step LIF rate is ``spike * 1/dt``.  Averaging ten 0.5 ms outputs is
    therefore the same as counting spikes over the 5 ms window and dividing by
    its duration.  The aggregate is what the neuromuscular adapter consumes at
    the end of the FlyVis frame; its timestamp is the outer frame timestamp so
    the emitted telemetry remains contract-consistent.
    """

    if not outputs:
        raise LauncherError("cannot aggregate an empty neural output window")
    if not math.isfinite(float(window_dt_s)) or float(window_dt_s) <= 0:
        raise LauncherError("neural output window must have a positive finite duration")

    rates_values = [item.rates_hz for item in outputs]
    spike_values = [item.spikes for item in outputs]
    rates: Optional[np.ndarray]
    spikes: Optional[np.ndarray]

    if all(value is not None for value in spike_values):
        spike_arrays = [np.asarray(value) for value in spike_values]
        if any(array.ndim != 1 for array in spike_arrays):
            raise LauncherError("neural spikes must be rank-one vectors")
        if any(array.shape != spike_arrays[0].shape for array in spike_arrays[1:]):
            raise LauncherError("neural spike vectors changed shape within a frame")
        spikes = np.sum(np.stack(spike_arrays, axis=0), axis=0)
    else:
        spikes = None

    if all(value is not None for value in rates_values):
        rate_arrays = [np.asarray(value, dtype=np.float64) for value in rates_values]
        if any(array.ndim != 1 for array in rate_arrays):
            raise LauncherError("neural rates must be rank-one vectors")
        if any(array.shape != rate_arrays[0].shape for array in rate_arrays[1:]):
            raise LauncherError("neural rate vectors changed shape within a frame")
        rates = np.mean(np.stack(rate_arrays, axis=0), axis=0).astype(np.float32)
    elif spikes is not None:
        rates = (np.asarray(spikes, dtype=np.float64) / float(window_dt_s)).astype(np.float32)
    else:
        rates = None

    # Keep the last output's metadata, then replace fields whose meaning is
    # explicitly windowed.  This preserves engine-specific named channels
    # without inventing a schema for unknown extensions.
    named = dict(outputs[-1].named_outputs)
    counts: list[float] = []
    deliveries: list[float] = []
    active: list[float] = []
    for item in outputs:
        value = item.named_outputs
        if "spike_count" in value:
            counts.append(float(value["spike_count"]))
        if "deliveries" in value:
            deliveries.append(float(value["deliveries"]))
        if "active_neurons" in value:
            active.append(float(value["active_neurons"]))
    if counts:
        named["spike_count"] = int(round(sum(counts)))
    if deliveries:
        named["deliveries"] = int(round(sum(deliveries)))
    if active:
        named["active_neurons"] = int(round(sum(active)))

    population_counts = [item.named_outputs.get("population_spike_counts")
                         for item in outputs]
    if all(isinstance(value, Mapping) for value in population_counts):
        keys = set().union(*(value.keys() for value in population_counts))
        named["population_spike_counts"] = {
            str(key): int(round(sum(float(value.get(key, 0.0)) for value in population_counts)))
            for key in keys
        }
    population_rates = [item.named_outputs.get("population_rates_hz")
                        for item in outputs]
    if all(isinstance(value, Mapping) for value in population_rates):
        keys = set().union(*(value.keys() for value in population_rates))
        named["population_rates_hz"] = {
            str(key): float(np.mean([float(value.get(key, 0.0)) for value in population_rates]))
            for key in keys
        }
        for key, value in named["population_rates_hz"].items():
            named[key] = value

    for key in ("external_poisson_hz_sum", "external_current_mV_per_ms_max"):
        values = [item.named_outputs[key] for item in outputs
                  if key in item.named_outputs]
        if values:
            named[key] = float(np.mean([float(value) for value in values]))
    named["neural_window_steps"] = len(outputs)
    named["neural_tick_dt_s"] = float(CNS_DT_S)
    named["neural_window_dt_s"] = float(window_dt_s)
    named["neural_window_dt_ms"] = float(window_dt_s) * 1000.0
    named["neural_window_start_s"] = float(outputs[0].timestamp.t_s)
    named["neural_window_end_s"] = float(outputs[-1].timestamp.t_s) + float(window_dt_s) / len(outputs)

    return NeuralOutput(timestamp, rates_hz=rates, spikes=spikes,
                        named_outputs=named, source=outputs[-1].source)


def _drive_summary(value: Any) -> dict[str, Any]:
    """Summarise a signed FlyVis vector without serialising every target."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return {"count": 0, "finite": False}
    if array.ndim != 1 or array.size == 0:
        return {"count": int(array.size), "finite": False}
    finite = bool(np.all(np.isfinite(array)))
    if not finite:
        return {"count": int(array.size), "finite": False}
    return {
        "count": int(array.size),
        "finite": True,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "rms": float(np.sqrt(np.mean(array * array))),
        "nonzero": int(np.count_nonzero(array)),
    }


def _flyvis_mapping_metadata(frontend: Any) -> dict[str, Any]:
    """Resolve fitted FlyVis target metadata without copying target values."""

    backend = getattr(frontend, "backend", None)
    runtime = getattr(backend, "response_fn", None)
    candidates = (
        frontend, backend, runtime, getattr(runtime, "model", None),
    )
    indices: Any = None
    target_ids: Any = None
    units: Any = None
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            if indices is None:
                indices = candidate.get("target_indices", candidate.get("flyvis_target_indices"))
            if target_ids is None:
                target_ids = candidate.get("target_ids")
            if units is None:
                units = candidate.get("units", candidate.get("target_units"))
        else:
            if indices is None:
                indices = getattr(candidate, "target_indices", None)
                if indices is None:
                    indices = getattr(candidate, "target_drive_indices", None)
                if indices is None:
                    method = getattr(candidate, "target_indices_array", None)
                    indices = method() if callable(method) else None
            if target_ids is None:
                target_ids = getattr(candidate, "target_ids", None)
            if units is None:
                units = getattr(candidate, "units", getattr(candidate, "target_units", None))
    return {"target_indices": indices, "target_ids": target_ids,
            "units": None if units is None else str(units)}


def _flyvis_mapping_summary(frontend: Any) -> dict[str, Any]:
    """Return bounded, stable mapping provenance for manifests/telemetry."""

    metadata = _flyvis_mapping_metadata(frontend)
    indices = metadata.get("target_indices")
    units = metadata.get("units")
    target_ids = metadata.get("target_ids")
    result: dict[str, Any] = {"units": units}
    if indices is None:
        result.update({"count": 0, "sha256": None})
        return result
    try:
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        result.update({"count": 0, "sha256": None, "valid": False})
        return result
    digest = hashlib.sha256(index_array.tobytes(order="C")).hexdigest()
    result.update({
        "count": int(index_array.size),
        "sha256": digest,
        "min_index": None if index_array.size == 0 else int(index_array.min()),
        "max_index": None if index_array.size == 0 else int(index_array.max()),
        "unique": int(np.unique(index_array).size),
    })
    if target_ids is not None:
        try:
            result["target_id_count"] = int(np.asarray(target_ids).size)
        except (TypeError, ValueError):
            result["target_id_count"] = 0
    return result


def _compact_neural_record(output: NeuralOutput, *, drive: Any = None,
                           target_indices: Any = None,
                           mapping_hash: Optional[str] = None,
                           minimal: bool = False) -> dict[str, Any]:
    """Return bounded per-frame neural telemetry for sustained real runs.

    A full 176,422-neuron ``rates_hz`` and ``spikes`` vector is useful for a
    one-frame probe but makes a seconds-long JSONL trace unnecessarily large.
    Compact telemetry retains all quantities needed for audit and rendering:
    window counts, population rates, backend/device, and a deterministic
    summary/hash of the visual target mapping.  The dense vectors remain
    available through ``RunOptions.full_neural_telemetry``.  Minimal mode
    keeps window scalars and event/population summaries, omitting mapping
    hashes and opaque diagnostics.
    """

    minimal_keep = {
        "backend", "device", "neural_window_steps", "neural_window_api",
        "neural_tick_dt_s", "neural_window_dt_s", "neural_window_dt_ms",
        "neural_window_start_s", "neural_window_end_s", "spike_count",
        "active_neurons", "population_rates_hz", "population_spike_counts",
    }
    population_tokens = ("lc4", "lplc2", "dnp01", "ttmn", "wing", "liftoff")
    named: dict[str, Any] = {}
    for key, value in output.named_outputs.items():
        key_text = str(key)
        if minimal:
            keep = key_text in minimal_keep or any(
                token in key_text.lower() for token in population_tokens
            )
            if not keep:
                continue
        elif key_text in {"flyvis_t4t5_drive", "flyvis_target_mapping",
                          "flyvis_target_indices", "flyvis_target_values"}:
            continue
        # Population summaries/scalars remain readable; unknown dense
        # diagnostics are hash/shape records so a plugin cannot reintroduce
        # a full per-neuron vector through ``named_outputs``.
        named[key_text] = _compact_json_value(value, include_array_data=False)
    if not minimal and drive is not None:
        named["flyvis_drive_summary"] = _drive_summary(drive)
    if not minimal and target_indices is not None:
        try:
            count = int(np.asarray(target_indices).size)
        except (TypeError, ValueError):
            count = 0
        named["flyvis_target_count"] = count
    if not minimal and mapping_hash is not None:
        named["flyvis_mapping_sha256"] = str(mapping_hash)
    return {
        "timestamp": output.timestamp.to_dict(),
        # ``None`` is intentional: it makes the omission explicit and keeps
        # readers from mistaking compact telemetry for a zero-valued vector.
        "rates_hz": None,
        "spikes": None,
        "named_outputs": named,
        "source": output.source.value if hasattr(output.source, "value") else output.source,
    }


def _compact_state_drive_record(output: Any, *, timestamp_s: Optional[float] = None,
                                frame_index: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Keep state-drive rates/phase without repeating its full config manifest."""

    if output is None:
        return None
    channels: dict[str, Any] = {}
    try:
        raw_channels = output.as_engine_channels()
    except AttributeError:
        raw_channels = getattr(output, "channels", {})
    if isinstance(raw_channels, Mapping):
        for unit, values in raw_channels.items():
            if isinstance(values, Mapping):
                channels[str(unit)] = {
                    str(population): float(value)
                    for population, value in values.items()
                }
    state = getattr(output, "state", None)
    state_dict = state.to_dict() if callable(getattr(state, "to_dict", None)) else {}
    provenance = getattr(output, "provenance", {})
    return _json_safe({
        "timestamp_s": None if timestamp_s is None else float(timestamp_s),
        "frame_index": None if frame_index is None else int(frame_index),
        "mode": getattr(output, "mode", state_dict.get("scenario", "unknown")),
        "phase": float(state_dict.get("phase", 0.0)),
        "sequence": int(getattr(output, "sequence", 0)),
        "channels": channels,
        "saturated": bool(getattr(output, "saturated", False)),
        "provenance": {
            "source": provenance.get("source", "handbuilt") if isinstance(provenance, Mapping) else "handbuilt",
            "biological_status": (
                provenance.get("biological_status", "hypothesis")
                if isinstance(provenance, Mapping) else "hypothesis"
            ),
            "calibration": (
                provenance.get("calibration", "calibration")
                if isinstance(provenance, Mapping) else "calibration"
            ),
        },
    })


def _compact_dynamic_record(value: Any, *, kind: str) -> Optional[dict[str, Any]]:
    """Keep the live NM/autopilot packet and its provenance in each row.

    These records are intentionally bounded to the dynamic command surface;
    dense neural vectors remain governed by ``full_neural_telemetry``.  The
    fallback attribute path keeps tiny test/mocked adapters auditable too.
    """

    if value is None:
        return None
    raw: Any = None
    for method_name in ("to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                raw = method()
            except (TypeError, ValueError):
                raw = None
            if isinstance(raw, Mapping):
                break
    fields = (
        "left_wing_power", "right_wing_power", "wing_power", "wing_steering",
        "left_wing", "right_wing", "thrust", "yaw", "pitch", "roll", "brake",
        "mode", "source", "safety_active", "safety_reason", "safety_blend",
        "neural_source", "readout_source", "safety_source", "sequence", "dt_ms",
        "saturated", "units", "provenance", "neural_actuators", "safety_actuators",
    )
    result: dict[str, Any] = {"kind": kind}
    if isinstance(raw, Mapping):
        for key, item in raw.items():
            if str(key) in fields or str(key) in {"channels", "readout_blend"}:
                result[str(key)] = item
    for key in fields:
        if key not in result and hasattr(value, key):
            result[key] = getattr(value, key)
    return _json_safe(result)


def _compact_neuromuscular_output(value: Any, *, minimal: bool = False) -> Optional[dict[str, Any]]:
    """Return the causal NM packet without repeating map/provenance metadata.

    ``NeuromuscularOutput.to_dict()`` intentionally includes the complete
    channel-definition/provenance tree.  That tree is immutable for a run and
    can be tens of kilobytes per frame; strict compact rows retain every
    decoded scalar channel and actuator value while referencing the manifest
    for the map hash and units.  Minimal mode further keeps the live channels
    and body-boundary scalars used by strict summaries.
    """

    if value is None:
        return None
    raw: Any = None
    live_channels = getattr(value, "channels", None)
    if (
        isinstance(live_channels, Mapping)
        and all(hasattr(value, name) for name in (
            "left_wing_power", "right_wing_power", "wing_power",
            "wing_steering", "left_leg_power", "right_leg_power",
            "leg_power", "leg_steering", "mouth", "groom", "dt_ms",
        ))
    ):
        # NeuromuscularOutput is already a normalized, constructor-validated
        # record.  Its generic ``to_dict`` eagerly copies the full immutable
        # unit/provenance tree; minimal telemetry needs only live channels and
        # actuator scalars, so build that public subset directly.
        channel = lambda name, default=0.0: live_channels.get(name, default)
        raw = {
            "channels": live_channels,
            "left_wing_power": getattr(value, "left_wing_power"),
            "right_wing_power": getattr(value, "right_wing_power"),
            "wing_power_left": getattr(value, "left_wing_power"),
            "wing_power_right": getattr(value, "right_wing_power"),
            "wing_drive_left": getattr(value, "left_wing_power"),
            "wing_drive_right": getattr(value, "right_wing_power"),
            "left_wing_activation": getattr(value, "left_wing_power"),
            "right_wing_activation": getattr(value, "right_wing_power"),
            "wing_activation_left": getattr(value, "left_wing_power"),
            "wing_activation_right": getattr(value, "right_wing_power"),
            "wing_motor_left": channel("wing_motor_left"),
            "wing_motor_right": channel("wing_motor_right"),
            "wing_power": getattr(value, "wing_power"),
            "wing_steering": getattr(value, "wing_steering"),
            "left_leg_power": getattr(value, "left_leg_power"),
            "right_leg_power": getattr(value, "right_leg_power"),
            "leg_power": getattr(value, "leg_power"),
            "leg_steering": getattr(value, "leg_steering"),
            "mouth": getattr(value, "mouth"),
            "groom": getattr(value, "groom"),
            "jump": channel("jump"),
            "jump_event": channel("jump"),
            "ttmn_activity": channel("ttmn_activity"),
            "jump_motor": channel("jump_motor"),
            "physical_jump_motor": channel("jump_motor"),
            "jump_activation": channel("jump_motor"),
            "dnp11_activity": channel("dnp11_activity"),
            "dnp02_activity": channel("dnp02_activity"),
            "dnp04_activity": channel("dnp04_activity"),
            "dt_ms": getattr(value, "dt_ms"),
            "source": getattr(value, "source", "connectome"),
            "saturated": bool(getattr(value, "saturated", False)),
            "provenance": getattr(value, "provenance", {}),
        }
    else:
        method = getattr(value, "to_dict", None)
        if callable(method):
            try:
                raw = method()
            except Exception:
                raw = None
    if not isinstance(raw, Mapping):
        # Tiny test adapters often expose only attributes.  Reuse the bounded
        # compatibility helper rather than inventing missing values.
        fallback = _compact_dynamic_record(value, kind="nm_out")
        if not minimal or not isinstance(fallback, Mapping):
            return fallback
        return {key: fallback[key] for key in (
            "kind", "provenance_ref", "channels", "left_wing_power",
            "right_wing_power", "wing_power", "wing_steering", "leg_power",
            "leg_steering", "ttmn_activity", "jump_motor", "dnp11_activity",
            "dnp02_activity", "dnp04_activity", "dt_ms", "source", "saturated",
            "physical_wing",
        ) if key in fallback}

    keep = (
        (
            "kind", "provenance_ref", "channels", "left_wing_power",
            "right_wing_power", "wing_power", "wing_steering", "leg_power",
            "leg_steering", "ttmn_activity", "jump_motor", "dnp11_activity",
            "dnp02_activity", "dnp04_activity", "dt_ms", "source", "saturated",
            "physical_wing",
        ) if minimal else (
        "channels", "left_wing_power", "right_wing_power", "wing_power_left",
        "wing_power_right", "wing_drive_left", "wing_drive_right",
        "left_wing_activation", "right_wing_activation", "wing_activation_left",
        "wing_activation_right", "wing_motor_left", "wing_motor_right",
        "wing_power", "wing_steering", "left_leg_power", "right_leg_power",
        "leg_power", "leg_steering", "mouth", "groom", "jump", "jump_event",
        "ttmn_activity", "jump_motor", "physical_jump_motor", "jump_activation",
        "dnp11_activity", "dnp02_activity", "dnp04_activity", "dt_ms", "source",
        "saturated",
        )
    )
    result: dict[str, Any] = {"kind": "nm_out", "provenance_ref": "manifest.components.neuromuscular"}
    for key in keep:
        if key not in raw:
            continue
        item = raw[key]
        # ``channels`` is the live decoder output and is therefore causal;
        # provenance/unit dictionaries are deliberately not copied here.
        result[key] = _compact_json_value(item, include_array_data=False)
    provenance = raw.get("provenance")
    if isinstance(provenance, Mapping):
        physical = provenance.get("physical_wing")
        if isinstance(physical, Mapping):
            # Source selection and bilateral activations affect the body
            # boundary on this frame; retain just that dynamic subset.
            result["physical_wing"] = _compact_json_value({
                key: physical[key] for key in (
                    "source", "left_activation", "right_activation",
                    "legacy_flight_power", "legacy_flight_steering",
                    "wing_motor_left", "wing_motor_right",
                ) if key in physical
            })
    return result


def _compact_mechanosensory_record(value: Any, *, minimal: bool = False) -> Optional[dict[str, Any]]:
    """Keep causal channels; the manifest owns static metadata.

    Minimal mode omits raw/debug projections while keeping population inputs.
    """

    if value is None:
        return None
    raw: Any = None
    if all(hasattr(value, name) for name in (
        "timestamp_s", "channels", "population_drive", "raw_channels",
        "raw_population_drive", "diagnostics", "population_drive_units",
        "population_drive_range", "source",
    )):
        raw = {
            "timestamp_s": getattr(value, "timestamp_s"),
            "source_timestamp_s": getattr(value, "source_timestamp_s", None),
            "channels": getattr(value, "channels"),
            "population_drive": getattr(value, "population_drive"),
            "raw_channels": getattr(value, "raw_channels"),
            "raw_population_drive": getattr(value, "raw_population_drive"),
            "diagnostics": getattr(value, "diagnostics"),
            "population_drive_units": getattr(value, "population_drive_units"),
            "population_drive_range": getattr(value, "population_drive_range"),
            "source": getattr(value, "source"),
        }
    else:
        method = getattr(value, "to_dict", None)
        if callable(method):
            try:
                raw = method()
            except Exception:
                raw = None
    if not isinstance(raw, Mapping):
        return _compact_json_value(value)
    keep = (
        (
            "timestamp_s", "source_timestamp_s", "channels", "population_drive",
            "population_drive_units", "population_drive_range", "source",
        ) if minimal else (
        "timestamp_s", "source_timestamp_s", "channels", "population_drive",
        "raw_channels", "raw_population_drive", "diagnostics",
        "population_drive_units", "population_drive_range", "source",
        )
    )
    result = {
        str(key): _compact_json_value(raw[key], include_array_data=False)
        for key in keep if key in raw
    }
    result["provenance_ref"] = "manifest.components.sensory_feedback"
    return result


def _compact_escape_packet(value: Any) -> Any:
    """Drop repeated experiment/provenance envelopes from channel telemetry."""

    if not isinstance(value, Mapping):
        return _compact_json_value(value)
    result: dict[str, Any] = {}
    for key in ("timestamp_s", "frame_index", "channels", "records"):
        if key in value:
            result[key] = _compact_json_value(value[key], include_array_data=False)
    result["provenance_ref"] = "manifest.components.escape_channels"
    return _json_safe(result)


def _compact_motor_record(value: Any) -> Any:
    """Retain the SI command and dynamic safety source, not nested packets."""

    if not isinstance(value, Mapping):
        return _compact_json_value(value)
    keep = (
        "timestamp", "yaw_rate_rad_s", "pitch_rate_rad_s", "roll_rate_rad_s",
        "thrust_mps2", "source", "saturated",
    )
    result = {str(key): _compact_json_value(value[key]) for key in keep if key in value}
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        result["metadata"] = _compact_json_value({
            key: metadata[key] for key in (
                "normalized", "mode", "source_raw", "safety_reason",
                "stabilizer_active",
            ) if key in metadata
        })
    result["provenance_ref"] = "manifest.provenance"
    return _json_safe(result)


def _free_behavior_row(
    row: Mapping[str, Any], *,
    mode: str,
    flyvis_output: Any,
    visual_observation: Any,
    nm_output: Any,
    mechanosensory: Any,
    escape_packet: Any,
    neural_output: Optional[NeuralOutput] = None,
    exact_sampling: Mapping[str, Any] | None = None,
    preserve_neural_vectors: bool = False,
    preserve_rectangular_render: bool = False,
    drive_sha256: Optional[str] = None,
    drive_summary: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Encode one strict row as lossless compact or hash-only minimal evidence."""

    if mode not in {"compact", "minimal"}:
        raise ValueError(f"unsupported free-behavior telemetry mode: {mode!r}")
    minimal = mode == "minimal"

    def value(name: str, default: Any = None) -> Any:
        return row.get(name, default)

    left = getattr(flyvis_output, "left_columns", getattr(flyvis_output, "left", None))
    right = getattr(flyvis_output, "right_columns", getattr(flyvis_output, "right", None))
    drive = getattr(flyvis_output, "t4t5_drive", getattr(flyvis_output, "drive", None))
    neural = value("neural", {})
    named_outputs = neural.get("named_outputs", {}) if isinstance(neural, Mapping) else {}
    target_mapping_hash = (
        named_outputs.get("flyvis_mapping_sha256")
        if isinstance(named_outputs, Mapping) else None
    )
    if minimal:
        drive_record = _required_hash_only_array_record(
            drive, name="minimal visual T4/T5 drive", sha256=drive_sha256,
        )
        summary = dict(drive_summary) if isinstance(drive_summary, Mapping) else _drive_summary(drive)
        visual_drive_sha256 = drive_record.get("sha256")
        visual_input = {
            "schema_version": "digital-fly-visual-input-minimal-1",
            "timestamp_s": float(value("frame_time_s", 0.0)),
            "frame_index": int(value("frame_index", 0)),
            "backend": getattr(flyvis_output, "backend", None),
            "encoding": "hash-only",
            "left_columns": _required_hash_only_array_record(
                left, name="minimal visual left eye columns",
            ),
            "right_columns": _required_hash_only_array_record(
                right, name="minimal visual right eye columns",
            ),
            "t4t5_drive": drive_record,
            "target_mapping_sha256": target_mapping_hash,
        }
    else:
        summary = _drive_summary(drive)
        visual_drive_sha256 = _array_sha256(drive)
        visual_input = {
            "schema_version": "digital-fly-visual-input-compact-1",
            "timestamp_s": float(value("frame_time_s", 0.0)),
            "frame_index": int(value("frame_index", 0)),
            "backend": getattr(flyvis_output, "backend", None),
            "units": "luminance[0,1] for columns; fitted target units for t4t5_drive",
            "left_columns": _lossless_array_record(left, include_data=True),
            "right_columns": _lossless_array_record(right, include_data=True),
            "t4t5_drive": _lossless_array_record(drive, include_data=True),
            "target_mapping_sha256": target_mapping_hash,
        }
    if exact_sampling is not None:
        visual_input["sampling"] = _compact_json_value(exact_sampling)

    visual_flow = getattr(visual_observation, "optic_flow_rad_s", (0.0, 0.0, 0.0))
    visual_source = getattr(getattr(visual_observation, "source", None), "value", None)
    try:
        pixel_shape = [2, int(np.asarray(left).size)]
    except (TypeError, ValueError):
        pixel_shape = None
    visual_record = {
        "timestamp": value("timestamp"),
        "pixels": None,
        "pixel_shape": pixel_shape,
        "optic_flow_rad_s": _compact_json_value(visual_flow),
        "sensor_id": getattr(visual_observation, "sensor_id", "virtual-world-stereo-flyvis"),
        "source": visual_source or getattr(visual_observation, "source", "handbuilt"),
    }
    if not minimal:
        visual_record["metadata"] = _compact_json_value(
            getattr(visual_observation, "metadata", {})
        )
    visual_record["input_ref"] = "visual_input"

    if minimal:
        sensory_channels = {
            "flyvis_drive_summary": summary,
            "flyvis_drive_sha256": visual_drive_sha256,
        }
        ablation_record = value("mechanosensory_ablation")
        masked_population_channels = (
            ablation_record.get("male_cns_population_channels")
            if isinstance(ablation_record, Mapping) else None
        )
        if isinstance(masked_population_channels, Mapping):
            # Minimal mode normally omits dense sensory rows; retain the small
            # selected alias set so the intervention at the CNS boundary is
            # directly visible alongside the raw feedback record.
            sensory_channels.update(_compact_json_value(masked_population_channels))
        food_cue = value("food_cue")
        if isinstance(food_cue, Mapping):
            cns_food_rates = food_cue.get("cns_poisson_hz")
            if isinstance(cns_food_rates, Mapping):
                sensory_channels["poisson_hz"] = _compact_json_value(cns_food_rates)
        intervention = value("neural_intervention")
        if isinstance(intervention, Mapping):
            rates = intervention.get("active_poisson_hz")
            currents = intervention.get("active_current_mV_per_ms")
            if isinstance(rates, Mapping) and rates:
                sensory_channels["neural_intervention_poisson_hz"] = _compact_json_value(rates)
            if isinstance(currents, Mapping) and currents:
                sensory_channels["neural_intervention_current_mV_per_ms"] = _compact_json_value(currents)
    else:
        sensory_raw = value("sensory", {})
        sensory_channels: dict[str, Any] = {}
        if isinstance(sensory_raw, Mapping) and isinstance(sensory_raw.get("channels"), Mapping):
            skip = {
                "flyvis_t4t5", "t4t5_drive", "flyvis_target_indices",
                "flyvis_target_values", "visual_calibration",
            }
            sensory_channels = {
                str(key): _compact_json_value(item, include_array_data=False)
                for key, item in sensory_raw["channels"].items()
                if str(key) not in skip
            }
        sensory_channels["flyvis_drive_summary"] = summary
        sensory_channels["flyvis_drive_sha256"] = visual_drive_sha256
    sensory_record = {
        "timestamp": value("timestamp"),
        "visual_ref": "visual_input",
        "channels": sensory_channels,
        "source": "handbuilt",
        "body_ref": "body_before",
    }

    compact_render: Optional[dict[str, Any]] = None
    if not minimal and preserve_rectangular_render:
        render_raw = value("render", {})
        if isinstance(render_raw, Mapping):
            compact_render = {
                "timestamp_s": render_raw.get("timestamp_s"),
                "frame_index": render_raw.get("frame_index"),
                "resolution": _compact_json_value(render_raw.get("resolution")),
                "eye_separation_m": render_raw.get("eye_separation_m"),
                "pose": _compact_json_value(render_raw.get("pose")),
                "purpose": "dynamic_provider_context",
                "left": (
                    None if render_raw.get("left") is None else
                    _lossless_array_record(render_raw.get("left"), include_data=True)
                ),
                "right": (
                    None if render_raw.get("right") is None else
                    _lossless_array_record(render_raw.get("right"), include_data=True)
                ),
            }

    if minimal:
        mechanosensory_record = value("mechanosensory")
        if not isinstance(mechanosensory_record, Mapping):
            mechanosensory_record = _compact_mechanosensory_record(
                mechanosensory, minimal=True,
            )
        neural_record = value("neural")
        if (
            not isinstance(neural_record, Mapping)
            or neural_record.get("rates_hz") is not None
            or neural_record.get("spikes") is not None
        ):
            neural_record = (
                _compact_neural_record(neural_output, minimal=True)
                if isinstance(neural_output, NeuralOutput) else {}
            )
        neuromuscular_record = value("nm_out")
        if not isinstance(neuromuscular_record, Mapping):
            neuromuscular_record = _compact_neuromuscular_output(
                nm_output, minimal=True,
            )
        motor_record = value("motor", {})
        if not (
            isinstance(motor_record, Mapping)
            and motor_record.get("provenance_ref") == "manifest.provenance"
        ):
            motor_record = _compact_motor_record(motor_record)
    else:
        mechanosensory_record = _compact_mechanosensory_record(mechanosensory)
        neural_record = _compact_json_value(
            value("neural", {}), include_array_data=preserve_neural_vectors,
        )
        neuromuscular_record = _compact_neuromuscular_output(nm_output)
        motor_record = _compact_motor_record(value("motor", {}))

    compact: dict[str, Any] = {
        "schema_version": (
            MINIMAL_TELEMETRY_SCHEMA_VERSION if minimal
            else COMPACT_TELEMETRY_SCHEMA_VERSION
        ),
        "telemetry_mode": mode,
        "scenario": value("scenario"),
    }
    if minimal:
        compact["execution_policy"] = FREE_BEHAVIOR_POLICY_NAME
    compact.update({
        "frame_index": int(value("frame_index", 0)),
        "frame_time_s": float(value("frame_time_s", 0.0)),
        "time_s": float(value("time_s", value("frame_time_s", 0.0))),
        "dt_s": float(value("dt_s", FLYVIS_DT_S)),
        "interval_start_s": float(value("interval_start_s", value("frame_time_s", 0.0))),
        "interval_end_s": float(value("interval_end_s", value("time_s", 0.0))),
        "timestamp": _compact_json_value(value("timestamp")),
        "position_m": _compact_json_value(value("position_m")),
        "velocity_mps": _compact_json_value(value("velocity_mps")),
        "orientation_quat": _compact_json_value(value("orientation_quat")),
        "orientation_rad": _compact_json_value(value("orientation_rad")),
        "angular_velocity_rad_s": _compact_json_value(value("angular_velocity_rad_s")),
        "contact": bool(value("contact", False)),
        "grounded": bool(value("grounded", False)),
        "target_contact": bool(value("target_contact", False)),
        "contact_surface_id": value("contact_surface_id"),
        "ground_distance_m": float(value("ground_distance_m", 0.0)),
        "flight_mode": value("flight_mode"),
        "controller_mode": value("controller_mode"),
        "left_wing": float(value("left_wing", 0.0)),
        "right_wing": float(value("right_wing", 0.0)),
        "wing_drive_left": float(value("wing_drive_left", value("left_wing", 0.0))),
        "wing_drive_right": float(value("wing_drive_right", value("right_wing", 0.0))),
        "yaw_rate": float(value("yaw_rate", 0.0)),
        "pitch": float(value("pitch", 0.0)),
        "roll": float(value("roll", 0.0)),
        "wind_mps": _compact_json_value(value("wind_mps", (0.0, 0.0, 0.0))),
        "visual_drive_summary": summary,
        "visual_drive_sha256": visual_drive_sha256,
        "visual_directional_readout": value("visual_directional_readout", "record_only"),
        "stimulus_state": _compact_json_value(value("stimulus_state", {})),
        "stimulus_condition": value("stimulus_condition"),
        "stimulus_name": value("stimulus_name"),
        "stimulus_contact": bool(value("stimulus_contact", False)),
        "stimulus_interaction": value("stimulus_interaction"),
        "escape_channels": _compact_escape_packet(escape_packet),
        "collision_count": int(value("collision_count", 0)),
        "body_before": _compact_json_value(value("body_before", {})),
        "body": _compact_json_value(value("body", {})),
        "visual_input": visual_input,
        "visual": visual_record,
        "render": None if minimal else compact_render,
    })
    if minimal:
        compact["world"] = None
    compact.update({
        "sensory": sensory_record,
        "mechanosensory": mechanosensory_record,
        "neural": neural_record,
        "neural_window": _compact_json_value(value("neural_window", {})),
        "nm_out": neuromuscular_record,
        "jump_decision_telemetry": _compact_json_value(value("jump_decision_telemetry", {})),
        "jump_input": _compact_json_value(value("jump_input", {})),
        "jump_output": _compact_json_value(value("jump_output", {})),
        "motor": motor_record,
        "diagnostics": _compact_json_value(value("diagnostics", {})),
    })
    ground_walking_record = value("ground_walking")
    if isinstance(ground_walking_record, Mapping):
        compact["ground_walking"] = _compact_json_value(
            ground_walking_record, include_array_data=False,
        )
    ablation_record = value("mechanosensory_ablation")
    if isinstance(ablation_record, Mapping):
        compact["mechanosensory_ablation"] = _compact_json_value(
            ablation_record, include_array_data=False,
        )
    food_cue_record = value("food_cue")
    if isinstance(food_cue_record, Mapping):
        compact["food_cue"] = _compact_json_value(food_cue_record, include_array_data=False)
    neural_intervention_record = value("neural_intervention")
    if isinstance(neural_intervention_record, Mapping):
        compact["neural_intervention"] = _compact_json_value(
            neural_intervention_record, include_array_data=False,
        )
    grooming_loop_record = value("grooming_loop")
    if isinstance(grooming_loop_record, Mapping):
        compact["grooming_loop"] = _compact_json_value(
            grooming_loop_record, include_array_data=False,
        )
    if minimal:
        compact["audit"] = {
            "immutable_provenance_ref": "manifest.provenance",
            "component_manifest_ref": "manifest.components",
            "array_encoding": "hash-only",
            "hash_algorithm": "sha256 over contiguous C-order bytes",
            "lossless_array_fields": [
                "visual_input.left_columns", "visual_input.right_columns",
                "visual_input.t4t5_drive",
            ],
            "omitted_repeated_fields": [
                "world.geometry", "world.render", "render.left", "render.right",
                "visual.pixels", "dense_neural_rates_hz", "dense_neural_spikes",
                "nm_out.provenance", "motor.metadata.nm_out", "frame_provenance",
                "provenance", "mechanosensory.population_metadata",
            ],
        }
    else:
        omitted = ["world.geometry", "world.render"]
        if not preserve_rectangular_render or preserve_neural_vectors:
            omitted.extend(["render.left", "render.right"])
        omitted.append("visual.pixels")
        if not preserve_neural_vectors:
            omitted.extend(["dense_neural_rates_hz", "dense_neural_spikes"])
        omitted.extend([
            "nm_out.provenance", "motor.metadata.nm_out",
            "motor.metadata.auto_command", "frame_provenance", "provenance",
        ])
        compact["audit"] = {
            "immutable_provenance_ref": "manifest.provenance",
            "component_manifest_ref": "manifest.components",
            "lossless_array_fields": [
                "visual_input.left_columns", "visual_input.right_columns",
                "visual_input.t4t5_drive",
            ] + (["neural.rates_hz", "neural.spikes"] if preserve_neural_vectors else []),
            "omitted_repeated_fields": omitted,
        }
    return compact if minimal else _json_safe(compact)

def _canonical_execution_mode(options: Any) -> str:
    """Resolve the public execution-policy aliases to one stable value."""

    explicit = getattr(options, "execution_mode", None)
    mode = explicit if explicit not in (None, "") else getattr(options, "mode", "normal")
    mode_text = str(mode or "normal").strip().lower().replace(" ", "-")
    if (
        mode_text in FREE_BEHAVIOR_MODE_NAMES
        or bool(getattr(options, "strict", False))
        or bool(getattr(options, "free_behavior", False))
        or bool(getattr(options, "escape_experiment", False))
    ):
        return FREE_BEHAVIOR_POLICY_NAME
    if mode_text in {"normal", "mission", "default", ""}:
        return "mission"
    return mode_text


def _canonical_telemetry_mode(options: Any) -> str:
    """Resolve compact/minimal/full-debug aliases to one explicit mode."""

    if bool(getattr(options, "full_debug", False)) or bool(
        getattr(options, "full_debug_telemetry", False)
    ):
        return "full-debug"
    value = getattr(options, "telemetry_mode", None)
    if value in (None, ""):
        value = getattr(options, "telemetry", "compact")
    mode = str(value or "compact").strip().lower().replace("_", "-")
    if mode in FULL_DEBUG_TELEMETRY_MODES:
        return "full-debug"
    if mode in MINIMAL_TELEMETRY_MODES:
        return "minimal"
    if mode in COMPACT_TELEMETRY_MODES:
        return "compact"
    raise BackendSelectionError(
        f"unknown telemetry mode {value!r}; use compact, minimal, or full-debug"
    )


def _record_only_key(key: Any) -> bool:
    """Return whether a channel key is metadata, not a causal input.

    Escape providers may expose labels, preferred directions, goals, or
    look-ahead annotations beside numeric channel values.  Those fields are
    retained in the row's record, but never cross the neural input boundary.
    """

    text = str(key).strip().lower().replace("-", "_")
    if not text:
        return True
    tokens = (
        "label", "direction", "preferred", "goal", "phase", "future",
        "lookahead", "target", "scenario", "mission", "policy",
    )
    return any(token in text for token in tokens)


def _numeric_channel_value(value: Any, *, name: str) -> Any:
    """Copy a finite numeric channel payload while rejecting opaque values."""

    if isinstance(value, bool):
        raise LauncherError(f"escape channel {name!r} must be numeric, not bool")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _record_only_key(key):
                continue
            try:
                result[str(key)] = _numeric_channel_value(item, name=f"{name}.{key}")
            except LauncherError:
                # A metadata-like nested item is record-only.  An opaque
                # non-metadata value makes the provider contract ambiguous.
                raise
        return result
    if isinstance(value, (str, bytes)):
        raise LauncherError(f"escape channel {name!r} must be numeric")
    if isinstance(value, (tuple, list, np.ndarray)):
        values = list(value)
        return [_numeric_channel_value(item, name=f"{name}[{index}]")
                for index, item in enumerate(values)]
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LauncherError(f"escape channel {name!r} must be numeric") from exc
    if not math.isfinite(number):
        raise LauncherError(f"escape channel {name!r} must be finite")
    # Preserve integral event counts as integers for audit readability while
    # keeping all other values as floats.
    if isinstance(value, (int, np.integer)):
        return int(value)
    return number


def _object_payload(value: Any) -> Any:
    """Get a JSON-safe provider payload without invoking control methods."""

    if isinstance(value, Mapping):
        return value
    for method_name in ("to_dict", "as_dict", "to_mapping"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                candidate = method()
            except (TypeError, ValueError):
                candidate = None
            if isinstance(candidate, Mapping):
                return candidate
    for attr_name in ("channels", "values", "payload", "data"):
        candidate = getattr(value, attr_name, None)
        if isinstance(candidate, Mapping):
            return candidate
    return value


def _escape_control_channels(value: Any, *, role: str) -> dict[str, Any]:
    """Extract only finite numeric channels from one provider output.

    The role-specific wrapper is deliberately permissive about spelling so a
    future dynamic-channel agent can publish either ``channels`` or a named
    ``jump``/``posture`` mapping.  It remains strict about the resulting
    values and strips label/direction/future-state metadata from control.
    """

    payload = _object_payload(value)
    if not isinstance(payload, Mapping):
        raise LauncherError(f"{role} provider must return a mapping or record with channels")
    role_aliases = {
        "dynamic_stimulus": ("dynamic_stimulus", "stimulus", "dynamic", "channels"),
        "jump_channel": ("jump_channel", "jump", "jump_channels", "channels"),
        "posture_channel": ("posture_channel", "posture", "posture_channels", "channels"),
    }
    candidate: Any = None
    for key in role_aliases.get(role, (role, "channels")):
        if key in payload:
            candidate = payload[key]
            break
    if candidate is None:
        # A plain mapping is accepted only when it is not made up solely of
        # record metadata.  This keeps tiny providers ergonomic.
        candidate = {
            key: item for key, item in payload.items()
            if str(key) not in {
                "timestamp", "timestamp_s", "frame_index", "provenance",
                "metadata", "units", "source", "role", "kind",
            } and not _record_only_key(key)
        }
    if candidate is None:
        raise LauncherError(f"{role} provider did not publish channels")
    if not isinstance(candidate, Mapping):
        # A single vector is retained under the role name so the neural engine
        # can consume it without the launcher inventing a population mapping.
        candidate = {role: candidate}
    result: dict[str, Any] = {}
    for key, item in candidate.items():
        if _record_only_key(key):
            continue
        result[str(key)] = _numeric_channel_value(item, name=f"{role}.{key}")
    if not result:
        raise LauncherError(f"{role} provider returned no numeric control channels")
    return result


class EscapeChannelAdapter:
    """Compatibility adapter for dynamic stimulus/jump/posture providers.

    This adapter is intentionally small and dependency-free.  It accepts a
    module, object, or callable from the future channel implementation while
    passing only causal frame context.  It never interprets provider labels or
    directions as control values; those remain in ``raw`` telemetry.
    """

    def __init__(self, source: Any, *, role: str, reference: Any = None,
                 module_name: Optional[str] = None,
                 module_path: Optional[Path] = None):
        self.source = source
        self.role = str(role)
        self.reference = reference
        self.module_name = module_name or getattr(source, "__module__", None)
        self.module_path = module_path
        self.calls = 0
        self._last_raw: Any = None

    def reset(self) -> None:
        reset = getattr(self.source, "reset", None)
        if callable(reset):
            reset()
        self.calls = 0
        self._last_raw = None

    @staticmethod
    def _invoke(method: Any, context: Mapping[str, Any]) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            # Unknown signatures are not permission to discard context.  A
            # one-shot context call is the narrowest useful compatibility
            # path; any provider error is surfaced to fail closed.
            return method(context)
        params = list(signature.parameters.values())
        accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in params)
        if accepts_kwargs:
            return method(**dict(context))
        kwargs = {
            str(item.name): context[item.name]
            for item in params
            if item.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.KEYWORD_ONLY)
            and item.name in context
        }
        required = [item for item in params
                    if item.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and item.default is inspect.Parameter.empty]
        if required and not kwargs and len(required) == 1:
            parameter = required[0].name
            if parameter in {"context", "frame", "packet", "input"}:
                return method(dict(context))
            if parameter in {"timestamp", "ts", "timestamp_s", "t_s", "time_s"}:
                return method(context["timestamp_s"])
        return method(**kwargs)

    def _method(self) -> Any:
        for name in ("step", "sample", "next_frame", "next", "emit"):
            candidate = getattr(self.source, name, None)
            if callable(candidate):
                return candidate
        if callable(self.source):
            return self.source
        raise LauncherError(
            f"{self.role} channel source {type(self.source).__name__!r} has no step/sample callable"
        )

    def step(self, *, timestamp_s: float, dt_s: float, frame_index: int,
             body: Optional[Mapping[str, Any]] = None,
             visual: Optional[Mapping[str, Any]] = None,
             experiment: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        # No world goals, phase labels, target directions, or post-step state
        # enter this context.  ``body`` is the causal pre-step observation.
        context = {
            "timestamp_s": float(timestamp_s),
            "t_s": float(timestamp_s),
            "dt_s": float(dt_s),
            "frame_index": int(frame_index),
            "body": None if body is None else _json_safe(body),
            "visual": None if visual is None else _json_safe(visual),
            # Experiment parameters describe the external stimulus source;
            # they are not interpreted by this runner as a control signal.
            "experiment": None if experiment is None else _json_safe(experiment),
        }
        try:
            raw = self._invoke(self._method(), context)
        except Exception as exc:
            raise LauncherError(f"{self.role} channel failed at frame {frame_index}: {exc}") from exc
        self.calls += 1
        self._last_raw = raw
        try:
            channels = _escape_control_channels(raw, role=self.role)
        except LauncherError:
            raise
        return {
            "role": self.role,
            "channels": channels,
            "raw": _json_safe(raw),
            "source": self.module_name or type(self.source).__name__,
            "call_index": self.calls,
        }

    def manifest(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "role": self.role,
            "source": self.module_name or type(self.source).__name__,
            "callable": type(self.source).__name__,
            "available": True,
            "validated": True,
        }
        if self.module_path is not None:
            value["path"] = str(self.module_path)
            value["sha256"] = _sha256_file(self.module_path)
        return value


def _load_escape_module(reference: Any, *, role: str) -> tuple[Any, Optional[str], Optional[Path]]:
    """Load a channel module/object with explicit, fail-closed resolution."""

    if reference is None:
        raise BackendSelectionError(f"{role} channel reference is missing")
    if not isinstance(reference, (str, Path)):
        return reference, getattr(reference, "__module__", None), None
    raw = Path(reference).expanduser()
    module_path: Optional[Path] = None
    if raw.is_file():
        module_path = raw.resolve()
        module_name = "digital_fly_escape_channel_" + hashlib.sha256(
            str(module_path).encode("utf-8")
        ).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise BackendSelectionError(f"could not load {role} channel module {module_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            raise BackendSelectionError(f"could not import {role} channel module {module_path}: {exc}") from exc
        module_name_value = module_name
    else:
        module_name_value = str(reference)
        try:
            module = importlib.import_module(module_name_value)
        except Exception as exc:
            raise BackendSelectionError(
                f"could not import {role} channel module {module_name_value!r}: {exc}"
            ) from exc

    role_words = {
        "dynamic_stimulus": ("dynamic_stimulus", "stimulus", "escape_stimulus"),
        "jump_channel": ("jump_channel", "jump", "escape_jump"),
        "posture_channel": ("posture_channel", "posture", "escape_posture"),
    }.get(role, (role,))
    factories: list[str] = []
    for word in role_words:
        factories.extend((f"create_{word}", f"build_{word}"))
    factories.extend(("create_channel", "build_channel", "create_adapter", "build_adapter"))
    for name in factories:
        candidate = getattr(module, name, None)
        if callable(candidate):
            try:
                return candidate(), module_name_value, module_path
            except TypeError as exc:
                raise BackendSelectionError(
                    f"{role} channel factory {name!r} requires an unsupported argument: {exc}"
                ) from exc
            except Exception as exc:
                raise BackendSelectionError(f"{role} channel factory {name!r} failed: {exc}") from exc
    classes: list[str] = []
    for word in role_words:
        classes.extend(("".join(part.title() for part in word.split("_")),))
    classes.extend(("EscapeChannel", "Channel", "Adapter", "Provider"))
    for name in classes:
        candidate = getattr(module, name, None)
        if callable(candidate):
            try:
                return candidate(), module_name_value, module_path
            except TypeError as exc:
                raise BackendSelectionError(
                    f"{role} channel class {name!r} requires an unsupported argument: {exc}"
                ) from exc
            except Exception as exc:
                raise BackendSelectionError(f"{role} channel class {name!r} failed: {exc}") from exc
    # A module with a direct step/sample function is itself the adapter.
    for name in ("step", "sample", "next_frame", "next", "emit"):
        if callable(getattr(module, name, None)):
            return module, module_name_value, module_path
    # Finally support a combined module publishing role-named instances.
    for name in role_words:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate, module_name_value, module_path
    raise BackendSelectionError(
        f"{role} channel module {module_name_value!r} exposes no supported provider; refusing fallback"
    )


class EscapeChannelSet:
    """Optional causal dynamic-stimulus telemetry boundary.

    Strict free behavior does not accept external jump/posture providers;
    those channels originate in the connectome neuromuscular adapter.
    """

    _PROBE_NAMES = {
        "dynamic_stimulus": (
            "dynamic_stimulus", "escape_stimulus", "stimulus_channel",
        ),
        "jump_channel": ("jump_channel", "escape_jump_channel", "jump_posture_channels"),
        "posture_channel": ("posture_channel", "escape_posture_channel", "jump_posture_channels"),
    }

    def __init__(self, sources: Mapping[str, EscapeChannelAdapter], *, probes: Mapping[str, Any] | None = None,
                 roles: Sequence[str] = ("dynamic_stimulus", "jump_channel", "posture_channel")):
        self.sources = dict(sources)
        self.probes = {} if probes is None else dict(probes)
        self.roles = tuple(str(role) for role in roles)
        self.last_frame: Optional[dict[str, Any]] = None

    @classmethod
    def discover(cls, references: Mapping[str, Any], *, required: bool = True,
                 roles: Sequence[str] = ("dynamic_stimulus", "jump_channel", "posture_channel")) -> "EscapeChannelSet":
        adapters: dict[str, EscapeChannelAdapter] = {}
        probes: dict[str, Any] = {}
        cache: dict[str, tuple[Any, Optional[str], Optional[Path]]] = {}
        missing: list[str] = []
        active_roles = tuple(str(role) for role in roles)
        for role in active_roles:
            reference = references.get(role)
            if reference in (None, ""):
                for candidate in cls._PROBE_NAMES[role]:
                    try:
                        if importlib.util.find_spec(candidate) is not None:
                            reference = candidate
                            probes[role] = {"probed": True, "candidate": candidate}
                            break
                    except (ImportError, ModuleNotFoundError, ValueError):
                        continue
            if reference in (None, ""):
                missing.append(role)
                probes.setdefault(role, {"probed": True, "candidate": None})
                continue
            cache_key = repr(reference)
            if cache_key not in cache:
                cache[cache_key] = _load_escape_module(reference, role=role)
            source, module_name, module_path = cache[cache_key]
            adapters[role] = EscapeChannelAdapter(
                source, role=role, reference=reference,
                module_name=module_name, module_path=module_path,
            )
            probes.setdefault(role, {"probed": False, "reference": _json_safe(reference)})
        if missing and required:
            details = ", ".join(missing)
            raise BackendSelectionError(
                "strict free-behavior mode requires dynamic stimulus, jump, and posture "
                f"channels; unavailable roles: {details}; refusing a scripted/no-input fallback"
            )
        return cls(adapters, probes=probes, roles=active_roles)

    def reset(self) -> None:
        seen: set[int] = set()
        for adapter in self.sources.values():
            identity = id(adapter.source)
            if identity in seen:
                continue
            seen.add(identity)
            adapter.reset()
        self.last_frame = None

    def step(self, *, timestamp_s: float, dt_s: float, frame_index: int,
             body: Optional[Mapping[str, Any]] = None,
             visual: Optional[Mapping[str, Any]] = None,
             experiment: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "timestamp_s": float(timestamp_s),
            "frame_index": int(frame_index),
            "channels": {},
            "records": {},
            "experiment": None if experiment is None else _json_safe(experiment),
            "provenance": {
                "label_control": False,
                "direction_control": False,
                "future_state_control": False,
                "causal_context": True,
            },
        }
        cache: dict[int, dict[str, Any]] = {}
        for role in self.roles:
            adapter = self.sources.get(role)
            if adapter is None:
                raise LauncherError(f"strict escape channel {role!r} is unavailable")
            identity = id(adapter.source)
            if identity not in cache:
                cache[identity] = adapter.step(
                    timestamp_s=timestamp_s, dt_s=dt_s, frame_index=frame_index,
                    body=body, visual=visual, experiment=experiment,
                )
            record = cache[identity]
            result["records"][role] = record
            channels = record["channels"]
            result["channels"][role] = channels
            # Expose role-prefixed aliases for engines that expect flat
            # channel names while preserving the canonical nested packet.
            for key, value in channels.items():
                result["channels"][f"escape:{role}:{key}"] = value
        self.last_frame = result
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "escape-channel-set-1",
            "required_roles": list(self.roles),
            "available_roles": sorted(self.sources),
            "probes": _json_safe(self.probes),
            "channels": {role: adapter.manifest() for role, adapter in self.sources.items()},
            "record_only_fields": [
                "label", "direction", "goal", "phase", "future", "target",
            ],
        }


class OptionalFlyVisResponse:
    """Resolve a project FlyVis callable only after explicit capability check."""

    def __init__(self, module: Any):
        self.module = module
        self.model = self._construct(module)
        self.last_output: Any = None

    @staticmethod
    def _construct(module: Any) -> Any:
        for name in ("create_response", "build_response", "create_model", "build_model"):
            factory = getattr(module, name, None)
            if callable(factory):
                try:
                    return factory()
                except TypeError:
                    continue
        for name in ("FlyVis", "Flyvis", "Model"):
            cls = getattr(module, name, None)
            if callable(cls):
                try:
                    return cls()
                except TypeError:
                    continue
        for name in ("response", "infer", "step"):
            if callable(getattr(module, name, None)):
                return module
        raise BackendSelectionError(
            "FlyVis is installed but no supported response factory/callable was found; "
            "refusing to substitute the deterministic vision stub"
        )

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()
        self.last_output = None

    def __call__(self, left_columns: Any, right_columns: Any) -> Any:
        method = self.model if callable(self.model) else getattr(self.model, "step", None)
        if not callable(method):
            method = getattr(self.model, "infer", None) or getattr(self.model, "response", None)
        if not callable(method):
            raise LauncherError("FlyVis adapter has no callable response")
        raw = method(left_columns, right_columns)
        # The shipped runtime callable returns the signed vector directly,
        # while newer adapters may return a ``FlyVisDriveOutput`` or mapping
        # carrying target_indices/units metadata.  Preserve that metadata for
        # the runner and normalise only the value consumed by StereoFlyVis.
        self.last_output = raw
        if isinstance(raw, Mapping):
            for key in ("drive", "target_drive", "values", "t4t5_drive"):
                if key in raw:
                    return raw[key]
            raise LauncherError(
                "FlyVis adapter mapping must contain drive/target_drive/values/t4t5_drive"
            )
        for key in ("drive", "target_drive", "values", "t4t5_drive"):
            value = getattr(raw, key, None)
            if value is not None:
                return value
        return raw


class VisualDriveCalibration:
    """Explicit optional adapter for a fitted FlyVis-drive calibration.

    The runner does not contain a guessed gain, bias, or controller.  When a
    caller supplies ``--visual-calibration`` this wrapper loads that module
    and requires it to expose a callable transform.  Several small integration
    spellings are accepted so a calibration module can remain independent of
    this launcher: ``create_calibrator``/``build_calibrator`` factories,
    ``VisualDriveCalibration``/``Calibration`` classes, or a module-level
    ``apply_drive``/``transform``/``calibrate_drive`` callable.  A malformed
    or missing result is an error; it is never treated as an identity
    calibration.

    The callable receives the signed T4/T5 vector and may return either the
    vector itself or a mapping containing ``drive``/``values`` plus optional
    metadata.  Context keyword arguments are passed only when the callable
    declares them (or accepts ``**kwargs``), keeping the integration point
    usable by tiny CPU mocks without hiding a runtime exception.
    """

    backend_name = "visual_drive_calibration"
    is_stub = False
    source = SourceLabel.FITTED

    def __init__(self, module: Any, *, module_name: str,
                 module_path: Optional[Path] = None,
                 config: Optional[Mapping[str, Any]] = None):
        self.module = module
        self.module_name = str(module_name)
        self.module_path = module_path
        self.config = dict(config or {})
        self.calls = 0
        self.calibrator = self._construct(module)

    @staticmethod
    def _call_factory(factory: Any, config: Mapping[str, Any]) -> Any:
        """Call a calibration factory without discarding explicit config."""

        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            try:
                return factory(config=config)
            except TypeError:
                return factory()
        params = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        )
        if (accepts_kwargs or "config" in params) and config:
            return factory(config=dict(config))
        return factory()

    @classmethod
    def _construct(cls, module: Any) -> Any:
        for name in ("create_calibrator", "build_calibrator", "create_calibration",
                     "build_calibration"):
            factory = getattr(module, name, None)
            if callable(factory):
                try:
                    # ``config`` is supplied by the instance method below; a
                    # factory is first retained as a marker when it cannot be
                    # called yet.  The caller replaces this marker in
                    # ``__init__`` with the configured object.
                    return ("factory", factory)
                except TypeError:
                    continue
        for name in ("VisualDriveCalibration", "Calibration", "Calibrator"):
            candidate = getattr(module, name, None)
            if callable(candidate):
                return ("factory", candidate)
        for name in ("apply_drive", "transform_drive", "calibrate_drive",
                     "transform", "apply", "calibrate"):
            method = getattr(module, name, None)
            if callable(method):
                return method
        if callable(module):
            return module
        raise CalibrationError(
            "visual calibration module exposes no supported transform "
            "(apply_drive/transform/calibrate_drive or a calibrator factory)"
        )

    def _ensure_instance(self) -> Any:
        value = self.calibrator
        if isinstance(value, tuple) and value and value[0] == "factory":
            try:
                value = self._call_factory(value[1], self.config)
            except Exception as exc:
                raise CalibrationError(
                    f"could not construct visual calibration {self.module_name!r}: {exc}"
                ) from exc
            self.calibrator = value
        return value

    @staticmethod
    def _invoke(method: Any, drive: np.ndarray, context: Mapping[str, Any]) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            return method(drive)
        params = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        )
        kwargs = {
            str(key): value for key, value in context.items()
            if accepts_kwargs or str(key) in params
        }
        return method(drive, **kwargs)

    def reset(self) -> None:
        target = self._ensure_instance()
        reset = getattr(target, "reset", None)
        if callable(reset):
            reset()
        self.calls = 0

    def enable_tonic_state_inputs(self, enabled: bool = True) -> None:
        """Opt into explicitly declared tonic hypotheses, never implicitly."""

        target = self._ensure_instance()
        method = (getattr(target, "with_tonic_state_inputs", None)
                  or getattr(target, "enable_tonic_state_inputs", None))
        if not callable(method):
            raise CalibrationError(
                f"visual calibration {self.module_name!r} does not expose "
                "with_tonic_state_inputs/enable_tonic_state_inputs"
            )
        try:
            updated = method(bool(enabled))
        except Exception as exc:
            raise CalibrationError(f"could not set tonic-state opt-in: {exc}") from exc
        # The project calibration returns an immutable configured copy.  A
        # mutable third-party calibrator may return None after updating itself.
        if updated is not None:
            self.calibrator = updated

    def _context(self, *, scenario: str, timestamp_s: float, frame_index: int,
                 visual_output: Any, target_indices: Any, target_ids: Any,
                 units: Optional[str], state_context: Any = None) -> dict[str, Any]:
        context: dict[str, Any] = {
            "scenario": scenario,
            "timestamp_s": float(timestamp_s),
            "frame_index": int(frame_index),
            "visual_output": visual_output,
            "config": dict(self.config),
            "target_indices": None if target_indices is None else np.asarray(target_indices).copy(),
            "target_ids": None if target_ids is None else np.asarray(target_ids).copy(),
            "units": units,
        }
        if state_context is not None:
            # The calibration API accepts aliases, but rejects receiving more
            # than one simultaneously.  Use its canonical spelling here.
            context["state_context"] = state_context
        return context

    @staticmethod
    def _invoke_channels(method: Any, channels: Mapping[str, Any], drive: Any,
                         context: Mapping[str, Any]) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            return method(channels, drive)
        params = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        )
        kwargs = {
            str(key): value for key, value in context.items()
            if accepts_kwargs or str(key) in params
        }
        return method(channels, drive, **kwargs)

    def apply_to_channels(self, channels: Mapping[str, Any], drive: Any, *,
                          scenario: str, timestamp_s: float, frame_index: int,
                          visual_output: Any = None, target_indices: Any = None,
                          target_ids: Any = None, units: Optional[str] = None,
                          state_context: Any = None,
                          population_resolver: Any = None,
                          n_neurons: Optional[int] = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Use a project calibration's full visual/tonic channel API."""

        target = self._ensure_instance()
        method = getattr(target, "apply_to_channels", None)
        if not callable(method):
            # Generic modules may provide only ``calibrate``.  Preserve the
            # same output shape and make the absence of tonic support explicit.
            calibrated, metadata = self.apply(
                drive, scenario=scenario, timestamp_s=timestamp_s,
                frame_index=frame_index, visual_output=visual_output,
                target_indices=target_indices, target_ids=target_ids, units=units,
                state_context=state_context,
            )
            result = dict(channels)
            result["flyvis_t4t5"] = {
                "target_indices": _json_safe(target_indices),
                "values": calibrated,
                "units": units,
            }
            return result, metadata
        raw_drive = {
            "values": np.asarray(drive, dtype=np.float32),
            "target_indices": None if target_indices is None else np.asarray(target_indices).copy(),
            "target_ids": None if target_ids is None else np.asarray(target_ids).copy(),
            "units": units,
        }
        context = self._context(
            scenario=scenario, timestamp_s=timestamp_s, frame_index=frame_index,
            visual_output=visual_output, target_indices=target_indices,
            target_ids=target_ids, units=units, state_context=state_context,
        )
        context.update({"population_resolver": population_resolver,
                       "n_neurons": n_neurons, "replace_visual": False})
        try:
            raw = self._invoke_channels(method, channels, raw_drive, context)
        except Exception as exc:
            raise CalibrationError(
                f"visual calibration {self.module_name!r} channel application failed "
                f"at frame {frame_index}: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise CalibrationError("visual calibration apply_to_channels must return a mapping")
        result = dict(raw)
        visual_mapping = result.get("flyvis_t4t5", result.get("t4t5_drive"))
        if visual_mapping is None:
            raise CalibrationError(
                "visual calibration apply_to_channels returned no flyvis_t4t5 mapping"
            )
        metadata: dict[str, Any] = {
            "backend": self.backend_name,
            "module": self.module_name,
            "calls": self.calls + 1,
            "channels": tuple(str(key) for key in result),
            "engine_mapping": visual_mapping,
        }
        self.calls += 1
        return result, metadata

    def apply(self, drive: Any, *, scenario: str, timestamp_s: float,
              frame_index: int, visual_output: Any = None,
              target_indices: Any = None, target_ids: Any = None,
              units: Optional[str] = None,
              state_context: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            original = np.asarray(drive, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("FlyVis drive is not a numeric vector before calibration") from exc
        if original.ndim != 1 or original.size == 0 or not np.all(np.isfinite(original)):
            raise CalibrationError("FlyVis drive must be a finite rank-one vector before calibration")
        target = self._ensure_instance()
        method = target if callable(target) else None
        if method is None:
            for name in ("apply_drive", "transform_drive", "calibrate_drive",
                         "transform", "apply", "calibrate"):
                candidate = getattr(target, name, None)
                if callable(candidate):
                    method = candidate
                    break
        if method is None:
            raise CalibrationError(
                f"visual calibration {self.module_name!r} has no callable transform"
            )
        context = self._context(
            scenario=scenario, timestamp_s=timestamp_s, frame_index=frame_index,
            visual_output=visual_output, target_indices=target_indices,
            target_ids=target_ids, units=units, state_context=state_context,
        )
        try:
            raw = self._invoke(method, original.copy(), context)
        except Exception as exc:
            raise CalibrationError(
                f"visual calibration {self.module_name!r} failed at frame {frame_index}: {exc}"
            ) from exc
        metadata: dict[str, Any] = {}
        engine_mapping: Any = None
        as_engine_mapping = getattr(raw, "as_engine_mapping", None)
        if callable(as_engine_mapping):
            try:
                engine_mapping = as_engine_mapping()
            except Exception as exc:
                raise CalibrationError(f"visual calibration mapping export failed: {exc}") from exc
        elif isinstance(raw, Mapping) and callable(raw.get("as_engine_mapping")):
            try:
                engine_mapping = raw["as_engine_mapping"]()
            except Exception as exc:
                raise CalibrationError(f"visual calibration mapping export failed: {exc}") from exc
        value = raw
        if isinstance(raw, Mapping):
            metadata = {
                str(key): item for key, item in raw.items()
                if str(key) not in {"drive", "target_drive", "values", "t4t5_drive"}
            }
            for key in ("drive", "target_drive", "values", "t4t5_drive"):
                if key in raw:
                    value = raw[key]
                    break
            else:
                raise CalibrationError(
                    "visual calibration mapping must contain drive/target_drive/values/t4t5_drive"
                )
        else:
            for key in ("drive", "target_drive", "values", "t4t5_drive"):
                candidate = getattr(raw, key, None)
                if candidate is not None:
                    value = candidate
                    metadata = {
                        "result_type": type(raw).__name__,
                        "result_attribute": key,
                    }
                    break
        # Keep the engine mapping in the result metadata.  The runner will
        # use it for the current frame while the compact row retains only its
        # bounded summary/hash.
        if engine_mapping is not None:
            metadata["engine_mapping"] = engine_mapping
        if isinstance(raw, Mapping) and all(
            key in raw for key in ("target_indices", "units")
        ):
            metadata["engine_mapping"] = {
                key: raw[key] for key in ("target_indices", "values", "drive", "units")
                if key in raw
            }
        else:
            for key in ("target_indices", "target_ids", "units"):
                candidate = getattr(raw, key, None)
                if candidate is not None:
                    metadata[key] = candidate
        try:
            result = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("visual calibration returned a non-numeric drive") from exc
        if result.shape != original.shape:
            raise CalibrationError(
                f"visual calibration changed drive shape from {original.shape} to {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise CalibrationError("visual calibration returned NaN or infinite drive values")
        self.calls += 1
        metadata.update({
            "backend": self.backend_name,
            "module": self.module_name,
            "calls": self.calls,
        })
        return result.copy(), metadata

    def manifest(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "backend": self.backend_name,
            "module": self.module_name,
            "is_stub": False,
            "source": self.source.value,
            "callable": type(self._ensure_instance()).__name__,
            "config": _json_safe(self.config),
        }
        if self.module_path is not None:
            value["path"] = str(self.module_path)
            value["sha256"] = _sha256_file(self.module_path)
        manifest_fn = getattr(self._ensure_instance(), "manifest", None)
        if callable(manifest_fn):
            try:
                runtime_manifest = _json_safe(manifest_fn())
                value["runtime"] = runtime_manifest
                if isinstance(runtime_manifest, Mapping):
                    for key in ("schema_version", "name", "config_path", "config_sha256",
                                "enabled", "input", "transform", "tonic_state_inputs"):
                        if key in runtime_manifest:
                            value.setdefault(key, runtime_manifest[key])
            except Exception as exc:
                raise CalibrationError(f"visual calibration manifest failed: {exc}") from exc
        return value


@dataclass
class RunOptions:
    config: Path = DEFAULT_CONFIG
    out: Path = DEFAULT_OUT
    scenarios: tuple[str, ...] = SCENARIOS
    # ``mode`` is the public CLI spelling.  ``execution_mode`` and the three
    # boolean aliases keep programmatic callers from having to coordinate a
    # migration while making the strict policy explicit in the manifest.
    mode: str = "mission"
    execution_mode: Optional[str] = None
    strict: bool = False
    free_behavior: bool = False
    escape_experiment: bool = False
    neural: str = "stub"
    vision: str = "stub"
    body: str = "reduced"
    seed: int = 1729
    width: int = 32
    height: int = 24
    columns: int = 32
    t4t5: int = 8
    neural_device: str = "auto"
    neural_backend: str = "auto"
    neural_gain: Optional[float] = None
    vision_device: str = "cuda:0"
    max_vram_gib: float = DEFAULT_MAX_VRAM_GIB
    duration_override_s: Optional[float] = None
    max_steps: Optional[int] = None
    frame_count: Optional[int] = None
    # Real neural runs default to compact per-frame telemetry.  The dense
    # vectors are still available for a bounded probe via this explicit flag.
    full_neural_telemetry: bool = False
    # ``compact`` is the existing production contract, including strict escape
    # runs.  ``minimal`` is an explicit strict free-behavior fast path with
    # hash-only visual inputs.  ``full-debug`` retains the historical row.
    visual_calibration: Optional[Path] = None
    visual_calibration_config: Optional[Path] = None
    visual_tonic_state_inputs: bool = False
    neural_csr_path: Optional[Path] = None
    # Explicit opt-in for the documented hand-built state-to-population
    # hypothesis.  No state drive is loaded when this is omitted.
    flight_state_drive: Optional[Path] = None
    # Explicit opt-in for the bounded low-level pitch/roll/collective overlay.
    # It never writes body state and remains disabled when this path is absent.
    flight_body_stabilizer: Optional[Path] = None
    # Strict free-behavior dynamic-stimulus compatibility source.  Jump and
    # posture are deliberately not provider inputs: they are decoded from
    # NeuromuscularAdapter output at the combined body boundary.
    dynamic_stimulus: Optional[Any] = None
    jump_channel: Optional[Any] = None
    posture_channel: Optional[Any] = None
    escape_channels_required: bool = True
    stimulus_config: Optional[Path] = None
    stimulus_name: Optional[str] = None
    stimulus_condition: Optional[str] = None
    # Physical stimulus parameters are handed to the external channel
    # provider as experiment metadata.  The runner records them but never
    # maps them into motor commands (0.5 m/s is the core palm case; 10 m/s is
    # an explicit pressure-test value).
    palm_speed_mps: Optional[float] = None
    impact_time_s: Optional[float] = None
    all_configured_scenarios: bool = False
    # New options are appended so older positional RunOptions construction
    # retains its field order.
    telemetry_mode: str = "compact"
    full_debug: bool = False
    full_debug_telemetry: bool = False
    renderer_backend: str = "analytic_reference"
    neural_cuda_graph: bool = False
    # Opt-in bounded overlap for strict minimal production runs.  It overlaps
    # only causal-state-free sensory preparation; the causal control chain and
    # telemetry stay serial.
    parallel_frame_stages: bool = False
    # Optional externally authored visual scene.  The scene manifest owns
    # mesh paths, transforms, materials and provenance independently of the
    # experiment's stimulus and body dynamics.
    scene_manifest: Optional[Path] = None
    # Explicitly zero one or more body-derived MaleCNS mechanosensory drive
    # groups while retaining the unmodified world/body feedback telemetry.
    mechanosensory_ablations: tuple[str, ...] = ()
    # Optional physical food field and receptor-only interventions.  These
    # are separate from the external visual scene and from motor control.
    food_config: Optional[Path] = None
    food_cue_ablations: tuple[str, ...] = ()
    # Explicit direct neural-input schedule used only for a separately
    # classified neuron-stimulation intervention.  It never sets body or
    # actuator state and cannot count as autonomous foraging.
    neural_intervention_config: Optional[Path] = None
    # Opt-in experimental grooming micro-body. Its dust-derived sensory input
    # and engineered foreleg motor primitive are separately classified.
    grooming_loop_config: Optional[Path] = None
    grooming_dust_count: int = 16
    grooming_sensor_off: bool = False
    grooming_foreleg_lock: bool = False
    # Opt-in visual courtship assay. The target's trajectory belongs to the
    # environment; movie playback replaces only retinal columns and is an
    # explicitly open-loop visual control.
    courtship_target_trajectory: Optional[Path] = None
    courtship_target_condition: str = "moving"
    courtship_movie: Optional[Path] = None
    courtship_movie_eye_mode: str = "monocular_replicated"
    courtship_movie_stereo_layout: Optional[str] = None


class DigitalFlyRunner:
    """Build and execute one auditable multi-component digital loop."""

    def __init__(self, options: RunOptions):
        self.options = options
        self.execution_policy = _canonical_execution_mode(options)
        self.is_free_behavior = self.execution_policy == FREE_BEHAVIOR_POLICY_NAME
        raw_mechanosensory_ablations = getattr(options, "mechanosensory_ablations", ())
        if isinstance(raw_mechanosensory_ablations, str):
            raw_mechanosensory_ablations = (raw_mechanosensory_ablations,)
        if not isinstance(raw_mechanosensory_ablations, Sequence):
            raise BackendSelectionError("mechanosensory_ablations must be a sequence of group names")
        normalized_ablations = tuple(
            dict.fromkeys(str(group).strip().lower() for group in raw_mechanosensory_ablations)
        )
        unknown_ablations = sorted(set(normalized_ablations) - set(MECHANOSENSORY_ABLATION_ENTRIES))
        if unknown_ablations:
            raise BackendSelectionError(
                "--ablate-mechanosensory accepts only jo, haltere, wing, and leg; "
                f"unknown group(s): {', '.join(unknown_ablations)}"
            )
        self.mechanosensory_ablations = normalized_ablations
        self.mechanosensory_ablation_sources: dict[str, tuple[str, ...]] = {}
        raw_food_ablations = getattr(options, "food_cue_ablations", ())
        if isinstance(raw_food_ablations, str):
            raw_food_ablations = (raw_food_ablations,)
        if not isinstance(raw_food_ablations, Sequence):
            raise BackendSelectionError("food_cue_ablations must be a sequence")
        self.food_cue_ablations = tuple(dict.fromkeys(
            str(group).strip().lower() for group in raw_food_ablations
        ))
        unknown_food_ablations = sorted(set(self.food_cue_ablations) - {"odor", "taste"})
        if unknown_food_ablations:
            raise BackendSelectionError(
                "--ablate-food-sense accepts only odor and taste; unknown group(s): "
                + ", ".join(unknown_food_ablations)
            )
        self.food_model: Any = None
        self.food_config_path: Optional[Path] = None
        self.food_config_sha256: Optional[str] = None
        self.food_selector_counts: dict[str, int] = {}
        self.food_selector_names: tuple[str, ...] = ()
        self.food_config_mapping: dict[str, Any] = {}
        self.neural_intervention_schedule: Any = None
        self.neural_intervention_config_path: Optional[Path] = None
        self.neural_intervention_config_sha256: Optional[str] = None
        self.neural_intervention_selector_counts: dict[str, int] = {}
        self.neural_intervention_selector_overlaps: dict[str, int] = {}
        self.grooming_bridge_config: Any = None
        self.grooming_loop_config_path: Optional[Path] = None
        self.grooming_loop_config_sha256: Optional[str] = None
        self.grooming_selector_counts: dict[str, int] = {}
        self.courtship_target_trajectory_path: Optional[Path] = None
        self.courtship_target_trajectory_sha256: Optional[str] = None
        self.courtship_movie_path: Optional[Path] = None
        self.courtship_movie_sha256: Optional[str] = None
        self.courtship_movie_stimulus: Any = None
        self.ground_walking_settings: Optional[dict[str, Any]] = None
        self.telemetry_mode = _canonical_telemetry_mode(options)
        self.minimal_telemetry = self.telemetry_mode == "minimal"
        # Minimal is a strict production fast path and shares compact's
        # no-rectangular-raster scheduling decisions, but has its own row
        # schema and hash-only array contract.
        self.compact_telemetry = self.telemetry_mode in {"compact", "minimal"}
        self.full_debug = self.telemetry_mode == "full-debug"
        self.minimal_neural_telemetry: dict[str, Any] = {
            "requested": bool(self.minimal_telemetry),
            "effective": False,
            "interface": None,
        }
        # Updated by the exact-column renderer seam once the first fitted
        # frame reveals whether the selected renderer can hand CUDA tensors to
        # FlyVis.  The manifest reports this observed path, never an
        # optimistic backend label.
        self._exact_column_transfer: dict[str, Any] = {
            "path": "host_numpy_boundary",
            "zero_copy": False,
            "device_resident_inputs": False,
            "host_roundtrip": True,
            "control_sink": "FlyVisRuntimeAdapter.step_detailed",
            "telemetry_copy": "host_float32_lossless_record",
        }
        self._validate_options()
        if self.options.neural_intervention_config is not None:
            try:
                from .neural_intervention import load_schedule
            except ImportError:  # direct script execution
                from neural_intervention import load_schedule
            intervention_path = Path(
                self.options.neural_intervention_config
            ).expanduser().resolve()
            self.neural_intervention_schedule = load_schedule(intervention_path)
            if not self.neural_intervention_schedule.enabled:
                raise BackendSelectionError(
                    "--neural-intervention-config must contain an enabled intervention schedule"
                )
            self.neural_intervention_config_path = intervention_path
            self.neural_intervention_config_sha256 = _sha256_file(intervention_path)
        if self.options.grooming_loop_config is not None:
            try:
                from .grooming_neural_bridge import load_config
            except ImportError:  # direct script execution
                from grooming_neural_bridge import load_config
            grooming_path = Path(self.options.grooming_loop_config).expanduser().resolve()
            self.grooming_bridge_config = load_config(grooming_path)
            self.grooming_loop_config_path = grooming_path
            self.grooming_loop_config_sha256 = _sha256_file(grooming_path)
        self.capabilities = {
            name: _module_capability(name) for name in ("malecns_engine", "flyvis", "flybody", "mujoco")
        }
        self.capabilities["visual_drive_calibration"] = {
            "requested": self.options.visual_calibration is not None,
            "available": False,
            "validated": False,
            "module": None,
            "path": None,
        }
        self.capabilities["visual_control_readout"] = {
            "requested": self.options.vision == "flyvis",
            "available": False,
            "validated": False,
            "backend": None,
        }
        self.capabilities["flight_state_drive"] = {
            "requested": self.options.flight_state_drive is not None,
            "available": False,
            "validated": False,
            "path": None,
        }
        self.capabilities["flight_body_stabilizer"] = {
            "requested": self.options.flight_body_stabilizer is not None,
            "available": False,
            "validated": False,
            "path": None,
        }
        self.capabilities["escape_channels"] = {
            "requested": bool(self.is_free_behavior),
            "available": False,
            "validated": False,
            "required_roles": ["physical_world_stimulus"],
            "probes": {},
        }
        self.escape_report_populations: tuple[str, ...] = ()
        self.visual_scene_baseline_report: Optional[dict[str, Any]] = None
        self._select_optional_backends()
        self._load_components()

    def _validate_options(self) -> None:
        if self.execution_policy not in {"mission", FREE_BEHAVIOR_POLICY_NAME}:
            raise BackendSelectionError(
                f"unknown execution mode {self.execution_policy!r}; use mission or strict free-behavior"
            )
        if not isinstance(self.options.strict, bool):
            raise BackendSelectionError("strict must be bool")
        if not isinstance(self.options.free_behavior, bool):
            raise BackendSelectionError("free_behavior must be bool")
        if not isinstance(self.options.escape_experiment, bool):
            raise BackendSelectionError("escape_experiment must be bool")
        for name in (
            "full_neural_telemetry", "full_debug", "full_debug_telemetry",
            "neural_cuda_graph", "parallel_frame_stages",
        ):
            if not isinstance(getattr(self.options, name), bool):
                raise BackendSelectionError(f"{name} must be bool")
        if self.minimal_telemetry and not self.is_free_behavior:
            raise BackendSelectionError(
                "minimal telemetry is only supported for strict free-behavior runs"
            )
        if self.minimal_telemetry and self.options.full_neural_telemetry:
            raise BackendSelectionError(
                "minimal telemetry cannot retain dense neural vectors"
            )
        if self.options.parallel_frame_stages and not (
            self.minimal_telemetry and self.is_free_behavior
        ):
            raise BackendSelectionError(
                "parallel frame stages require strict free-behavior minimal telemetry"
            )
        if self.options.neural not in {"stub", "malecns"}:
            raise BackendSelectionError("--neural must be stub or malecns")
        if self.mechanosensory_ablations and self.options.neural != "malecns":
            raise BackendSelectionError(
                "--ablate-mechanosensory requires --neural malecns; "
                "refusing a no-op ablation on the stub endpoint"
            )
        if self.options.food_config is not None:
            if not self.is_free_behavior or self.options.neural != "malecns":
                raise BackendSelectionError(
                    "--food-config requires strict free-behavior with --neural malecns"
                )
            if not Path(self.options.food_config).expanduser().is_file():
                raise BackendSelectionError(
                    f"--food-config is missing: {self.options.food_config}"
                )
        elif self.food_cue_ablations:
            raise BackendSelectionError(
                "--ablate-food-sense requires --food-config"
            )
        if self.options.neural_intervention_config is not None:
            if not self.is_free_behavior or self.options.neural != "malecns":
                raise BackendSelectionError(
                    "--neural-intervention-config requires strict free-behavior "
                    "with --neural malecns"
                )
            if not Path(self.options.neural_intervention_config).expanduser().is_file():
                raise BackendSelectionError(
                    "--neural-intervention-config is missing: "
                    f"{self.options.neural_intervention_config}"
                )
        if self.options.grooming_loop_config is not None:
            if not self.is_free_behavior or self.options.neural != "malecns":
                raise BackendSelectionError(
                    "--grooming-loop-config requires strict free-behavior with --neural malecns"
                )
            if self.options.neural_intervention_config is not None:
                raise BackendSelectionError(
                    "--grooming-loop-config cannot be combined with a timed neural intervention"
                )
            if self.options.food_config is not None:
                raise BackendSelectionError(
                    "--grooming-loop-config cannot be combined with a food cue experiment"
                )
            if not Path(self.options.grooming_loop_config).expanduser().is_file():
                raise BackendSelectionError(
                    f"--grooming-loop-config is missing: {self.options.grooming_loop_config}"
                )
            count = self.options.grooming_dust_count
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise BackendSelectionError("--grooming-dust-count must be a nonnegative integer")
        elif self.options.grooming_sensor_off or self.options.grooming_foreleg_lock \
                or self.options.grooming_dust_count != 16:
            raise BackendSelectionError(
                "grooming condition switches require --grooming-loop-config"
            )
        if self.options.courtship_target_trajectory is not None:
            if not self.is_free_behavior or self.options.neural != "malecns" or self.options.vision != "flyvis":
                raise BackendSelectionError(
                    "--courtship-target-trajectory requires strict MaleCNS/FlyVis free behavior"
                )
            if any((self.options.food_config, self.options.grooming_loop_config,
                    self.options.neural_intervention_config)):
                raise BackendSelectionError(
                    "courtship visual assay cannot combine food, grooming, or direct neural interventions"
                )
            if self.options.courtship_target_condition not in {"moving", "static", "hidden"}:
                raise BackendSelectionError(
                    "--courtship-target-condition must be moving, static, or hidden"
                )
            if not Path(self.options.courtship_target_trajectory).expanduser().is_file():
                raise BackendSelectionError(
                    f"courtship target trajectory is missing: {self.options.courtship_target_trajectory}"
                )
            if self.options.courtship_movie is not None:
                raise BackendSelectionError(
                    "moving courtship target and open-loop movie are separate visual conditions"
                )
        elif self.options.courtship_target_condition != "moving":
            raise BackendSelectionError(
                "--courtship-target-condition requires --courtship-target-trajectory"
            )
        if self.options.courtship_movie is not None:
            if not self.is_free_behavior or self.options.neural != "malecns" or self.options.vision != "flyvis":
                raise BackendSelectionError(
                    "--courtship-movie requires strict MaleCNS/FlyVis free behavior"
                )
            if any((self.options.food_config, self.options.grooming_loop_config,
                    self.options.neural_intervention_config)):
                raise BackendSelectionError(
                    "courtship movie assay cannot combine food, grooming, or direct neural interventions"
                )
            if not Path(self.options.courtship_movie).expanduser().is_file():
                raise BackendSelectionError(
                    f"courtship movie is missing: {self.options.courtship_movie}"
                )
            if self.options.courtship_movie_eye_mode not in {"monocular_replicated", "binocular_pov"}:
                raise BackendSelectionError("invalid courtship movie eye mode")
            if self.options.courtship_movie_eye_mode == "binocular_pov" and self.options.courtship_movie_stereo_layout != "side_by_side_lr":
                raise BackendSelectionError(
                    "binocular courtship movie requires --courtship-movie-stereo-layout side_by_side_lr"
                )
        elif self.options.courtship_movie_eye_mode != "monocular_replicated" or self.options.courtship_movie_stereo_layout is not None:
            raise BackendSelectionError("courtship movie options require --courtship-movie")
        if self.options.vision not in {"stub", "flyvis"}:
            raise BackendSelectionError("--vision must be stub or flyvis")
        # A connectome neural endpoint needs the fitted FlyVis target mapping
        # for its visual input contract.  Reject this contradictory pair
        # before capability probing/imports so the failure is deterministic
        # and never looks like an optional-dependency problem.
        if (
            not self.is_free_behavior
            and self.options.neural == "malecns"
            and self.options.vision == "stub"
        ):
            raise BackendSelectionError(
                "--neural malecns requires --vision flyvis; the deterministic vision stub "
                "cannot satisfy the real neural visual-input contract"
            )
        if self.is_free_behavior:
            # The experiment is deliberately open-loop at the behaviour
            # boundary.  Requiring the real MaleCNS endpoint prevents a
            # scripted stub from masquerading as free behaviour.  Device
            # selection remains explicit/fail-closed in the neural and vision
            # adapters; this policy does not silently downgrade CUDA.
            if self.options.neural != "malecns":
                raise BackendSelectionError(
                    "strict free-behavior mode requires --neural malecns; "
                    "stub scripted control is prohibited"
                )
            # CPU remains the default for tests, but an explicitly requested
            # CUDA MaleCNS/FlyVis run is valid when the adapters' read-only
            # gates accept it.  The runner never downgrades a failed request.
            if self.options.flight_state_drive is not None:
                raise BackendSelectionError(
                    "FlightStateDrive is prohibited in strict free-behavior mode"
                )
            if self.options.flight_body_stabilizer is not None:
                raise BackendSelectionError(
                    "FlightBodyStabilizer is prohibited in strict free-behavior mode"
                )
            if self.options.visual_tonic_state_inputs:
                raise BackendSelectionError(
                    "visual tonic state inputs are prohibited in strict free-behavior mode"
                )
            if not isinstance(self.options.escape_channels_required, bool):
                raise BackendSelectionError("escape_channels_required must be bool")
            for name in ("palm_speed_mps", "impact_time_s"):
                value = getattr(self.options, name)
                if value is not None:
                    try:
                        number = float(value)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise BackendSelectionError(f"{name} must be finite") from exc
                    if not math.isfinite(number) or number <= 0:
                        raise BackendSelectionError(f"{name} must be a positive finite number")
        if self.options.body not in {"reduced", "flybody"}:
            raise BackendSelectionError("--body must be reduced or flybody")
        if self.options.renderer_backend not in {"analytic_reference", "warp_cuda"}:
            raise BackendSelectionError(
                "--renderer-backend must be analytic_reference or warp_cuda"
            )
        if self.options.scene_manifest is not None and not Path(self.options.scene_manifest).expanduser().is_file():
            raise BackendSelectionError(
                f"--scene-manifest is missing: {self.options.scene_manifest}"
            )
        if self.options.neural_device not in {"auto", "cpu", "cuda"}:
            raise BackendSelectionError("--neural-device must be auto, cpu, or cuda")
        if self.options.neural_backend not in {"auto", "cpu", "torch", "torch_cpu", "triton"}:
            raise BackendSelectionError(
                "--neural-backend must be auto, cpu, torch, torch_cpu, or triton"
            )
        if self.options.neural_cuda_graph and (
            self.options.neural != "malecns"
            or
            self.options.neural_device != "cuda"
            or self.options.neural_backend != "triton"
        ):
            raise BackendSelectionError(
                "--neural-cuda-graph requires --neural malecns with explicit "
                "--neural-device cuda and --neural-backend triton"
            )
        if self.options.vision_device != "cpu" and not self.options.vision_device.startswith("cuda:"):
            raise BackendSelectionError("--vision-device must be cpu or cuda:N")
        if not math.isfinite(float(self.options.max_vram_gib)) or self.options.max_vram_gib <= 0:
            raise BackendSelectionError("--max-vram-gib must be a positive finite number")
        if self.options.neural_gain is not None:
            try:
                gain = float(self.options.neural_gain)
            except (TypeError, ValueError, OverflowError) as exc:
                raise BackendSelectionError("--neural-gain must be a positive finite number") from exc
            if not math.isfinite(gain) or gain <= 0.0:
                raise BackendSelectionError("--neural-gain must be a positive finite number")
        if self.options.stimulus_condition is not None and not str(self.options.stimulus_condition).strip():
            raise BackendSelectionError("stimulus_condition must be non-empty when supplied")
        if self.options.width <= 0 or self.options.height <= 0 or self.options.columns <= 0 or self.options.t4t5 <= 0:
            raise BackendSelectionError("camera/column sizes must be positive")
        if self.options.duration_override_s is not None and self.options.duration_override_s <= 0:
            raise BackendSelectionError("duration override must be positive")
        if self.options.max_steps is not None and self.options.max_steps <= 0:
            raise BackendSelectionError("max_steps must be positive")
        if self.options.frame_count is not None and self.options.frame_count <= 0:
            raise BackendSelectionError("frame_count must be positive")
        if not isinstance(self.options.visual_tonic_state_inputs, bool):
            raise BackendSelectionError("visual_tonic_state_inputs must be bool")
        for name in ("visual_calibration", "visual_calibration_config", "neural_csr_path",
                     "flight_state_drive", "flight_body_stabilizer", "stimulus_config"):
            value = getattr(self.options, name)
            if value is not None and not isinstance(value, (str, Path)):
                raise BackendSelectionError(f"{name} must be a path or null")
        if self.is_free_behavior:
            for name in ("dynamic_stimulus", "jump_channel", "posture_channel"):
                value = getattr(self.options, name)
                if value is not None and not isinstance(value, (str, Path)) and not callable(value):
                    # Provider objects are accepted when they expose a
                    # standard step/sample API; reject opaque values now so
                    # a typo cannot become an implicit no-op.
                    if not any(callable(getattr(value, attr, None))
                               for attr in ("step", "sample", "next_frame", "next", "emit")):
                        raise BackendSelectionError(
                            f"{name} must be a module/path or a channel provider object"
                        )

    def _select_optional_backends(self) -> None:
        if self.options.neural == "malecns":
            if not self.capabilities["malecns_engine"]["installed"]:
                raise BackendSelectionError(
                    "--neural malecns requested, but malecns_engine.py/package is absent; "
                    "the launcher will not silently use the stub"
                )
            # The engine factory owns CUDA safety and may intentionally select
            # the NumPy CPU path when its read-only gate is closed.  Checking
            # the gate here used to make ``--neural malecns`` contradictory:
            # the runner rejected a safe CPU fallback before the factory saw
            # the requested ``device='auto'`` policy.
        if self.options.vision == "flyvis":
            # The fitted runtime lives in its own environment and is
            # intentionally not imported by capability discovery.  A
            # filesystem-only preflight keeps this selection fail-closed while
            # allowing the adapter to perform the actual lazy import later.
            from flyvis_runtime_adapter import preflight_flyvis
            report = preflight_flyvis()
            self.capabilities["flyvis"]["preflight"] = report
            self.capabilities["flyvis"]["installed"] = bool(report["available"])
            if not report["available"]:
                details = "; ".join(report["reasons"])
                raise BackendSelectionError(
                    "--vision flyvis requested, but the dedicated FlyVis environment/assets "
                    f"are unavailable: {details}; the launcher will not silently use the stub"
                )
        if self.options.body == "flybody":
            if not (self.capabilities["flybody"]["installed"] or self.capabilities["mujoco"]["installed"]):
                raise BackendSelectionError(
                    "--body flybody requested, but neither flybody nor mujoco is installed"
                )
            raise BackendSelectionError(
                "--body flybody detected an optional runtime, but no validated project body adapter is registered; "
                "refusing to run as reduced"
            )

    def _load_visual_calibration(self) -> Optional[VisualDriveCalibration]:
        """Load only an explicitly requested visual-drive calibration."""

        requested = self.options.visual_calibration
        if requested is None:
            return None
        raw = Path(requested).expanduser()
        module_path: Optional[Path] = None
        module_name: str
        if raw.is_file():
            module_path = raw.resolve()
            module_name = "digital_fly_visual_calibration_" + hashlib.sha256(
                str(module_path).encode("utf-8")
            ).hexdigest()[:12]
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise CalibrationError(f"could not load visual calibration module {module_path}")
            module = importlib.util.module_from_spec(spec)
            try:
                # Dataclasses and a few plugin frameworks resolve the module
                # through ``sys.modules`` while class decorators execute.
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception as exc:
                raise CalibrationError(
                    f"could not import visual calibration module {module_path}: {exc}"
                ) from exc
        else:
            module_name = str(requested)
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                raise CalibrationError(
                    f"could not import visual calibration module {module_name!r}: {exc}"
                ) from exc
        config: dict[str, Any] = {}
        config_path = self.options.visual_calibration_config
        if config_path is not None:
            config_file = Path(config_path).expanduser()
            try:
                parsed = json.loads(config_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CalibrationError(
                    f"could not read visual calibration config {config_file}: {exc}"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise CalibrationError("visual calibration config must contain a JSON object")
            config.update(parsed)
        module_config = getattr(module, "CALIBRATION_CONFIG", None)
        if isinstance(module_config, Mapping):
            merged = dict(module_config)
            merged.update(config)
            config = merged
        adapter = VisualDriveCalibration(
            module, module_name=module_name, module_path=module_path, config=config,
        )
        # Construct now so a requested but malformed calibration fails before
        # the first body or neural step.
        adapter._ensure_instance()
        if self.options.visual_tonic_state_inputs:
            adapter.enable_tonic_state_inputs(True)
        self.capabilities["visual_drive_calibration"] = {
            "requested": True,
            "available": True,
            "validated": True,
            "module": module_name,
            "path": None if module_path is None else str(module_path),
            "sha256": None if module_path is None else _sha256_file(module_path),
            "manifest": adapter.manifest(),
        }
        return adapter

    def _load_flight_state_drive(self) -> Any:
        """Load the explicit state-drive hypothesis, if requested."""

        requested = self.options.flight_state_drive
        if requested is None:
            return None
        config_path = Path(requested).expanduser()
        if not config_path.is_file():
            raise BackendSelectionError(
                f"--flight-state-drive config is missing: {config_path}"
            )
        try:
            from flight_state_drive import FlightStateDrive
            adapter = FlightStateDrive.from_json(config_path)
            manifest = adapter.manifest()
        except Exception as exc:
            raise BackendSelectionError(
                f"could not load --flight-state-drive {config_path}: {exc}"
            ) from exc
        self.capabilities["flight_state_drive"] = {
            "requested": True,
            "available": True,
            "validated": True,
            "path": str(config_path.resolve()),
            "sha256": _sha256_file(config_path),
            "manifest": manifest,
        }
        return adapter

    def _load_flight_body_stabilizer(self) -> Any:
        """Load the optional bounded body-controller hypothesis explicitly."""

        requested = self.options.flight_body_stabilizer
        if requested is None:
            return None
        config_path = Path(requested).expanduser()
        if not config_path.is_file():
            raise BackendSelectionError(
                f"--flight-body-stabilizer config is missing: {config_path}"
            )
        try:
            from flight_body_stabilizer import FlightBodyStabilizer
            adapter = FlightBodyStabilizer.from_json(config_path)
            manifest = adapter.manifest()
        except Exception as exc:
            raise BackendSelectionError(
                f"could not load --flight-body-stabilizer {config_path}: {exc}"
            ) from exc
        self.capabilities["flight_body_stabilizer"] = {
            "requested": True,
            "available": True,
            "validated": True,
            "path": str(config_path.resolve()),
            "sha256": _sha256_file(config_path),
            "manifest": manifest,
        }
        return adapter

    def _escape_channel_references(self) -> dict[str, Any]:
        """Resolve channel references from options and the experiment JSON."""

        references: dict[str, Any] = {
            "dynamic_stimulus": self.options.dynamic_stimulus,
            "jump_channel": self.options.jump_channel,
            "posture_channel": self.options.posture_channel,
        }
        config_data: Mapping[str, Any] = {}
        try:
            parsed = json.loads(Path(self.options.config).read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                config_data = parsed
        except (OSError, ValueError):
            # VirtualWorld will surface a more precise configuration error;
            # channel discovery must not turn an unreadable file into a
            # silent no-input path.
            config_data = {}
        experiment_spec = config_data.get("experiment", {})
        self.escape_experiment_spec = (
            dict(experiment_spec) if isinstance(experiment_spec, Mapping) else {}
        )
        containers: list[Mapping[str, Any]] = []
        for key in ("escape_channels", "channels", "adapters", "escape_experiment"):
            candidate = config_data.get(key)
            if isinstance(candidate, Mapping):
                containers.append(candidate)
                nested = candidate.get("channels")
                if isinstance(nested, Mapping):
                    containers.append(nested)
        aliases = {
            "dynamic_stimulus": ("dynamic_stimulus", "stimulus", "dynamic"),
            "jump_channel": ("jump_channel", "jump", "jump_channels"),
            "posture_channel": ("posture_channel", "posture", "posture_channels"),
        }
        for role, names in aliases.items():
            if references[role] not in (None, ""):
                continue
            for container in containers:
                for name in names:
                    if name not in container or container[name] in (None, ""):
                        continue
                    value = container[name]
                    if isinstance(value, Mapping):
                        value = value.get("module", value.get("path", value.get("source", value)))
                    references[role] = value
                    break
                if references[role] not in (None, ""):
                    break
        # Relative Python paths are conventionally relative to the experiment
        # config, not to whichever directory launched the process.
        base = Path(self.options.config).expanduser().resolve().parent
        for role, value in tuple(references.items()):
            if isinstance(value, str):
                candidate = Path(value).expanduser()
                if not candidate.is_absolute() and (base / candidate).is_file():
                    references[role] = base / candidate
        return references

    def _load_escape_channels(self) -> EscapeChannelSet:
        references = self._escape_channel_references()
        if self.options.palm_speed_mps is not None:
            self.escape_experiment_spec["palm_speed_mps"] = float(self.options.palm_speed_mps)
        if self.options.impact_time_s is not None:
            self.escape_experiment_spec["impact_time_s"] = float(self.options.impact_time_s)
        if not self.is_free_behavior:
            self.capabilities["escape_channels"] = {
                "requested": False,
                "available": False,
                "validated": False,
                "required_roles": list(references),
                "probes": {},
            }
            return EscapeChannelSet({}, probes={})
        if self.options.jump_channel is not None or self.options.posture_channel is not None:
            raise BackendSelectionError(
                "strict escape mode prohibits external jump/posture providers; "
                "those channels must originate at NeuromuscularAdapter and enter the combined body bridge"
            )
        # Dynamic providers are compatibility/telemetry hooks only.  Physical
        # stimulus geometry is loaded into VirtualWorld below; no provider
        # output is injected into the neural sensory packet.
        references["jump_channel"] = None
        references["posture_channel"] = None
        dynamic_reference = references.get("dynamic_stimulus")
        channels = EscapeChannelSet.discover(
            references,
            required=dynamic_reference not in (None, ""),
            roles=("dynamic_stimulus",),
        )
        if not channels.sources:
            # The physical VirtualWorld stimulus is the default source for
            # this experiment.  An absent optional dynamic provider must not
            # become a fake per-frame input role whose ``step`` then fails;
            # retain the probe record while making the telemetry hook truly
            # inactive.  Explicit provider references remain fail-closed
            # above.
            channels = EscapeChannelSet(
                {}, probes=channels.probes, roles=(),
            )
        self.capabilities["escape_channels"] = {
            "requested": True,
            "available": True,
            "validated": True,
            "required_roles": ["physical_world_stimulus"],
            "probes": channels.probes,
            "manifest": channels.manifest(),
        }
        return channels

    def _configure_renderer_backend(self, world: Any) -> Any:
        """Select the requested world renderer without implicit fallback."""

        requested = str(self.options.renderer_backend)
        if requested == "analytic_reference":
            return world
        if requested != "warp_cuda":
            raise RendererBackendError(f"unsupported renderer backend {requested!r}")
        device = str(self.options.vision_device)
        if not device.startswith("cuda:"):
            raise RendererBackendError(
                "--renderer-backend warp_cuda requires --vision-device cuda:N; "
                f"got {device!r}"
            )
        device_index = device.split(":", 1)[1]
        try:
            if not device_index or int(device_index) < 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise RendererBackendError(
                "--renderer-backend warp_cuda requires --vision-device cuda:N; "
                f"got {device!r}"
            ) from exc

        # This gate must run before importing the optional renderer (and thus
        # before importing Warp itself).  It owns the repository's UVM/HMM,
        # host-memory, and weights checks; bypassing it would make a renderer
        # request capable of opening an unsafe CUDA context.
        try:
            gate = importlib.import_module("malecns_gate")
        except Exception as exc:
            raise RendererBackendError(
                "--renderer-backend warp_cuda requires scripts/malecns_gate.py "
                f"before CUDA initialisation: {exc}"
            ) from exc
        require_cuda = getattr(gate, "require_cuda", None)
        if not callable(require_cuda):
            raise RendererBackendError(
                "--renderer-backend warp_cuda requires malecns_gate.require_cuda()"
            )
        try:
            require_cuda()
        except SystemExit as exc:
            raise RendererBackendError(
                f"CUDA gate closed for warp_cuda on {device}: {exc}"
            ) from exc
        except Exception as exc:
            raise RendererBackendError(
                f"CUDA gate check failed for warp_cuda on {device}: {exc}"
            ) from exc
        try:
            module = importlib.import_module("world_renderer_opt")
        except Exception as exc:
            raise RendererBackendError(
                "--renderer-backend warp_cuda requires world_renderer_opt; "
                f"the optional renderer is unavailable: {exc}"
            ) from exc
        renderer = _construct_renderer_backend(
            module,
            world,
            backend=requested,
            device=device,
        )
        if not callable(getattr(renderer, "render_stereo", None)):
            raise RendererBackendError(
                "warp_cuda renderer lacks required render_stereo seam"
            )
        if not any(callable(getattr(renderer, method_name, None)) for method_name in (
            "render_eye_directions_tensor", "render_eye_directions", "render_columns",
        )):
            raise RendererBackendError(
                "warp_cuda renderer lacks required render_eye_directions seam"
            )
        return _RendererBackendWorldProxy(world, renderer, backend_name=requested)

    def _load_free_world(self, camera: Any) -> Any:
        """Load only physical scene data for strict mode.

        ``VirtualWorld.from_json`` eagerly parses mission scenario profiles.
        Strict mode instead copies only the world and one initializer's
        duration/pose/wind fields into a profile-free mapping, so mission
        phases and goals are not part of the runtime input graph at all.
        """

        from virtual_world import PhysicalStimulus, VirtualWorld, load_stimuli

        try:
            raw = json.loads(Path(self.options.config).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LauncherError(f"could not read free-behavior world config {self.options.config}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise LauncherError("free-behavior world config must contain a JSON object")
        raw_world = raw.get("world", {})
        if not isinstance(raw_world, Mapping):
            raise LauncherError("free-behavior world config 'world' must be an object")
        experiment = raw.get("experiment", {})
        if not isinstance(experiment, Mapping):
            experiment = {}
        source_scenarios = raw.get("scenarios", {})
        if not isinstance(source_scenarios, Mapping):
            source_scenarios = {}
        first_scenario: Mapping[str, Any] = {}
        if source_scenarios:
            candidate = next(iter(source_scenarios.values()))
            if isinstance(candidate, Mapping):
                first_scenario = candidate
        default_ground_z = float(raw_world.get("ground_z_m", 0.0))
        default_body_radius = float(raw_world.get("body_radius_m", 0.0012))
        palm_center_z = experiment.get("palm_center_z_m")
        if palm_center_z is not None:
            try:
                palm_center_z_value = float(palm_center_z)
            except (TypeError, ValueError, OverflowError) as exc:
                raise BackendSelectionError("experiment.palm_center_z_m must be finite") from exc
            if not math.isfinite(palm_center_z_value):
                raise BackendSelectionError("experiment.palm_center_z_m must be finite")
            if abs(palm_center_z_value - (default_ground_z + default_body_radius)) > 1.0e-9:
                raise BackendSelectionError(
                    "free-behavior palm center must match ground_z_m + body_radius_m"
                )
        initial_pose = experiment.get("initial_pose", first_scenario.get("initial_pose", {
            "position_m": [0.0, 0.0, default_ground_z + default_body_radius],
            "yaw_rad": 0.0,
            "pitch_rad": 0.0,
            "roll_rad": 0.0,
            "velocity_mps": [0.0, 0.0, 0.0],
        }))
        if bool(experiment.get("resting_start", True)) and isinstance(initial_pose, Mapping):
            position = initial_pose.get("position_m", ())
            try:
                initial_z = float(position[2])
            except (TypeError, ValueError, IndexError, KeyError) as exc:
                raise BackendSelectionError(
                    "free-behavior resting start requires initial_pose.position_m[2]"
                ) from exc
            if abs(initial_z - (default_ground_z + default_body_radius)) > 1.0e-9:
                raise BackendSelectionError(
                    "strict free-behavior experiment must start at ground_z_m + body_radius_m"
                )
        duration = experiment.get("duration_s", first_scenario.get("duration_s", 2.5))
        dt_s = experiment.get("dt_s", FLYVIS_DT_S)
        initializer: dict[str, Any] = {
            "duration_s": duration,
            "dt_s": dt_s,
            "initial_pose": initial_pose,
        }
        if "wind_override" in experiment:
            initializer["wind_override"] = experiment["wind_override"]
        # Resolve the physical stimulus explicitly.  VirtualWorld does not
        # infer top-level stimulus entries from an in-memory mapping; passing
        # the selected object here ensures the same geometry affects pixels,
        # wind, and collision queries.  Stimulus state is logged later only.
        stimulus_config_value = self.options.stimulus_config
        if stimulus_config_value is None:
            stimulus_config_value = raw.get("stimulus_config", raw.get("escape_stimuli"))
        if stimulus_config_value is None:
            default_stimulus_path = ROOT / "config" / "escape_stimuli.json"
            stimulus_config_value = default_stimulus_path if default_stimulus_path.is_file() else None
        if stimulus_config_value is None:
            raise BackendSelectionError(
                "strict free-behavior experiment requires a physical stimulus config"
            )
        stimulus_path = Path(stimulus_config_value).expanduser()
        if not stimulus_path.is_absolute():
            config_relative = Path(self.options.config).expanduser().resolve().parent / stimulus_path
            stimulus_path = config_relative if config_relative.is_file() else ROOT / "config" / stimulus_path
        if not stimulus_path.is_file():
            raise BackendSelectionError(f"physical stimulus config is missing: {stimulus_path}")
        try:
            loaded_stimuli = load_stimuli(stimulus_path, seed=int(raw.get("seed", self.options.seed)))
        except Exception as exc:
            raise BackendSelectionError(f"could not load physical stimulus config {stimulus_path}: {exc}") from exc
        stimulus_source_meta: Mapping[str, Any] = {}
        try:
            stimulus_source_raw = json.loads(stimulus_path.read_text(encoding="utf-8"))
            if isinstance(stimulus_source_raw, Mapping):
                stimulus_source_meta = stimulus_source_raw
        except (OSError, ValueError):
            stimulus_source_meta = {}
        stimulus_protocol = stimulus_source_meta.get("protocol", {})
        if not isinstance(stimulus_protocol, Mapping):
            stimulus_protocol = {}
        requested_stimulus_name = (
            self.options.stimulus_condition
            or self.options.stimulus_name
            or raw.get("stimulus_name")
            or experiment.get("stimulus_name")
            or stimulus_protocol.get("grounded_stimulus", stimulus_protocol.get("default_stimulus"))
            or next(iter(loaded_stimuli), None)
        )
        requested_stimulus_name = str(requested_stimulus_name)
        resolved_stimulus_name = requested_stimulus_name
        if resolved_stimulus_name not in loaded_stimuli:
            raise BackendSelectionError(
                f"physical stimulus {requested_stimulus_name!r} is absent from {stimulus_path}"
            )
        selected_stimulus = loaded_stimuli[resolved_stimulus_name]
        override = raw.get("stimulus_overrides", {})
        if not isinstance(override, Mapping):
            override = {}
        experiment_overrides = raw.get("experiment", {})
        if isinstance(experiment_overrides, Mapping):
            nested_override = experiment_overrides.get("stimulus_overrides", {})
            if isinstance(nested_override, Mapping):
                override = {**dict(override), **dict(nested_override)}
        override_values: dict[str, Any] = {}
        for key in ("start_position_m", "direction_m", "size_m", "start_time_s",
                    "duration_s", "mode", "texture", "renderable", "collidable",
                    "anchor", "offset_m"):
            if key in override:
                override_values[key] = override[key]
        if "palm_center_z_m" in experiment:
            start = list(selected_stimulus.start_position_m)
            start[2] = float(experiment["palm_center_z_m"])
            override_values.setdefault("start_position_m", start)
        # The core escape stimulus is visual-only.  Keep it in the world so
        # pixels/airflow/state telemetry remain causal, but never let its
        # geometry become a body collision or an interaction that changes the
        # trajectory.  Ground contact remains owned by FlightDynamics.
        if str(experiment.get("stimulus_interaction", "visual_only_non_interacting")) == "visual_only_non_interacting":
            override_values["collidable"] = False
            override_values["visual_only"] = True
            override_values["non_interacting"] = True
            planned_contact = experiment.get("surface_contact_deadline_s")
            if planned_contact is not None:
                override_values["counterfactual_contact_time_s"] = float(planned_contact)
        # The source stimulus owns the condition's calibrated kinematics.
        # An experiment-level speed is metadata (0.5 m/s core, 10 m/s
        # pressure) and must not overwrite the static/sham controls.  Only a
        # command-line setup override or an explicit stimulus override may
        # change the selected physical object.
        requested_speed = (
            self.options.palm_speed_mps
            if self.options.palm_speed_mps is not None
            else override.get("speed_mps")
        )
        if requested_speed is not None:
            speed = float(requested_speed)
            direction = override_values.get("direction_m", selected_stimulus.direction_m)
            direction_array = np.asarray(direction, dtype=np.float64)
            norm = float(np.linalg.norm(direction_array))
            if norm <= 0.0 or not math.isfinite(norm):
                raise BackendSelectionError("physical stimulus direction must be non-zero")
            override_values["velocity_mps"] = (direction_array / norm * speed).tolist()
        elif "velocity_mps" in override:
            override_values["velocity_mps"] = override["velocity_mps"]
        if override_values:
            try:
                selected_stimulus = replace(selected_stimulus, **override_values)
            except Exception as exc:
                raise BackendSelectionError(f"invalid physical stimulus override: {exc}") from exc
        if self.options.courtship_target_trajectory is not None:
            try:
                from courtship_target import FemaleTrajectoryStimulus
                trajectory_path = Path(
                    self.options.courtship_target_trajectory
                ).expanduser().resolve()
                selected_stimulus = FemaleTrajectoryStimulus(
                    trajectory=trajectory_path,
                    stimulus_id="female_fly",
                    condition=self.options.courtship_target_condition,
                )
            except Exception as exc:
                raise BackendSelectionError(
                    f"could not load courtship target trajectory: {exc}"
                ) from exc
            self.courtship_target_trajectory_path = trajectory_path
            self.courtship_target_trajectory_sha256 = _sha256_file(trajectory_path)
            resolved_stimulus_name = "female_fly"
            requested_stimulus_name = "female_fly_" + self.options.courtship_target_condition
        self.escape_stimulus = selected_stimulus
        self.escape_stimulus_path = stimulus_path
        self.escape_stimulus_name = str(resolved_stimulus_name)
        self.escape_stimulus_condition = str(requested_stimulus_name)
        self.escape_stimulus_manifest = _json_safe(selected_stimulus.to_dict())
        sanitized = {
            "schema_version": "free-escape-world-1",
            "units": raw.get("units", "SI"),
            "seed": raw.get("seed", self.options.seed),
            "world": {**dict(raw_world), "body_radius_m": raw_world.get("body_radius_m", 0.0012)},
            "scenarios": {"escape": initializer},
        }
        world = VirtualWorld.from_mapping(
            sanitized,
            camera=camera,
            scene_path=self.options.scene_manifest,
            stimuli={selected_stimulus.id: selected_stimulus},
        )
        if self.options.courtship_target_trajectory is not None:
            if "female_fly" not in world.stimulus_visual_meshes:
                raise BackendSelectionError(
                    "courtship target requires --scene-manifest binding female_fly to a glTF/GLB model"
                )
        return world

    def _load_components(self) -> None:
        from virtual_world import CameraConfig, VirtualWorld
        from stereo_flyvis import OptionalFlyVisBackend, StereoFlyVis
        from flyvis_runtime_adapter import FlyVisRuntimeAdapter
        from neuromuscular_adapter import NeuromuscularAdapter
        from sensory_feedback import MechanosensoryFeedback
        from flight_dynamics import FlightDynamics
        if not self.is_free_behavior:
            from flight_autopilot import FlightAutopilot

        self.visual_calibration = self._load_visual_calibration()
        self.flight_state_drive = self._load_flight_state_drive()
        self.flight_body_stabilizer = self._load_flight_body_stabilizer()
        self.escape_channels = self._load_escape_channels()
        self.visual_control_readout = None

        camera_options: dict[str, Any] = {}
        if (self.options.courtship_target_trajectory is not None
                or self.options.courtship_movie is not None):
            raw_camera = self.escape_experiment_spec.get("camera", {})
            if not isinstance(raw_camera, Mapping):
                raise BackendSelectionError("courtship experiment.camera must be an object")
            allowed_camera = {"hfov_rad", "vfov_rad", "eye_separation_m", "near_m", "far_m"}
            unknown_camera = sorted(set(raw_camera) - allowed_camera)
            if unknown_camera:
                raise BackendSelectionError(
                    f"courtship experiment.camera has unsupported fields: {unknown_camera}"
                )
            camera_options = dict(raw_camera)
        try:
            camera = CameraConfig(self.options.width, self.options.height, **camera_options)
        except (TypeError, ValueError) as exc:
            raise BackendSelectionError(f"invalid courtship camera configuration: {exc}") from exc
        self.world = (
            self._load_free_world(camera)
            if self.is_free_behavior
            else VirtualWorld.from_json(self.options.config, camera=camera,
                                        scene_path=self.options.scene_manifest)
        )
        self.world = self._configure_renderer_backend(self.world)
        if self.options.food_config is not None:
            from food_sensing import FoodCueModel

            food_path = Path(self.options.food_config).expanduser().resolve()
            food_mapping = json.loads(food_path.read_text(encoding="utf-8"))
            if not isinstance(food_mapping, Mapping):
                raise BackendSelectionError("food config root must be an object")
            self.food_model = FoodCueModel.from_mapping(food_mapping)
            self.food_config_path = food_path
            self.food_config_sha256 = _sha256_file(food_path)
            self.food_config_mapping = dict(food_mapping)
            target_id = str(self.world.config.target.id)
            if self.food_model.contact_surface_id != target_id:
                raise BackendSelectionError(
                    "food contact_surface_id must match the physical world target id: "
                    f"{self.food_model.contact_surface_id!r} != {target_id!r}"
                )
            odor_maps = food_mapping.get("odor_selector_weights_hz_per_unit")
            sugar_rates = food_mapping.get("sugar_selector_rates_hz")
            if not isinstance(odor_maps, Mapping) or not isinstance(sugar_rates, Mapping):
                raise BackendSelectionError("food config has no receptor selector maps")
            selector_names = set(str(spec) for spec in sugar_rates)
            for side in ("left", "right"):
                side_map = odor_maps.get(side)
                if not isinstance(side_map, Mapping):
                    raise BackendSelectionError(f"food config has no {side} odor selector map")
                selector_names.update(str(spec) for spec in side_map)
            self.food_selector_names = tuple(sorted(selector_names))
            expected_counts = food_mapping.get("selector_expected_counts")
            if not isinstance(expected_counts, Mapping) or set(expected_counts) != selector_names:
                raise BackendSelectionError(
                    "food config selector_expected_counts must cover exactly its receptor selectors"
                )
        if self.options.vision == "stub":
            self.flyvis_runtime_adapter = None
            self.flyvis = StereoFlyVis(
                self.options.columns, self.options.t4t5, backend="stub", start_time_s=0.0,
            )
        else:
            # The real fitted model has one 721-column eye and returns the
            # actual MaleCNS target subset (not the launcher's small stub
            # width).  A zero frontend baseline keeps raw luminance intact;
            # the runtime adapter owns the signed grey-screen baseline in
            # activity space.
            runtime_adapter = FlyVisRuntimeAdapter(
                device=self.options.vision_device,
                max_vram_gib=self.options.max_vram_gib,
            )
            backend = OptionalFlyVisBackend(
                runtime_adapter, runtime_adapter.n_columns, runtime_adapter.n_t4t5,
            )
            # Preserve mapping provenance on the stereo backend for callers
            # that need to configure MaleCNSEngine's explicit target/unit
            # injection contract.
            backend.target_indices = runtime_adapter.target_indices.copy()
            backend.target_ids = runtime_adapter.target_ids.copy()
            backend.units = runtime_adapter.units
            self.flyvis_runtime_adapter = runtime_adapter
            # The real T4/T5 vector must be reduced with its direction/subtype/
            # side mapping; a population mean is not a valid yaw cue.  This
            # import is intentionally lazy so CPU/stub runs never require the
            # optional control-readout assets.
            from flyvis_control_readout import FlyVisControlReadout
            # Load the checked direction asset as the canonical readout
            # mapping.  Each frame also supplies the runtime adapter mapping
            # below, so a stale/reordered fitted mapping fails closed rather
            # than merely matching its own metadata.
            if not self.is_free_behavior:
                self.visual_control_readout = FlyVisControlReadout()
            self.capabilities["visual_control_readout"] = {
                "requested": not self.is_free_behavior,
                "available": True,
                "validated": True,
                "backend": (
                    "flyvis_control_readout"
                    if not self.is_free_behavior else "prohibited_free_behavior"
                ),
                "manifest": (
                    _optional_manifest(self.visual_control_readout)
                    if self.visual_control_readout is not None else None
                ),
            }
            zeros = np.zeros(runtime_adapter.n_columns, dtype=np.float32)
            self.flyvis = StereoFlyVis(
                runtime_adapter.n_columns, runtime_adapter.n_t4t5,
                backend=backend, baseline=(zeros, zeros), start_time_s=0.0,
            )
        # Exact FlyVis geometry is immutable for a fitted runtime.  Validate
        # it once (and, for Warp, stage contiguous CUDA tensors once) before
        # any scenario/frame loop begins.
        self._prepare_exact_direction_cache()
        if self.visual_control_readout is None:
            self.capabilities["visual_control_readout"] = {
                "requested": bool(self.is_free_behavior and self.options.vision == "flyvis"),
                "available": bool(self.is_free_behavior and self.options.vision == "flyvis"),
                "validated": False,
                "backend": "prohibited_free_behavior" if self.is_free_behavior else None,
                "manifest": None,
            }
        self.flyvis_mapping = _flyvis_mapping_summary(self.flyvis)
        flyvis_mapping_metadata = _flyvis_mapping_metadata(self.flyvis)
        mapping_indices = flyvis_mapping_metadata.get("target_indices")
        self._flyvis_engine_mapping: Optional[dict[str, Any]] = None
        if mapping_indices is not None:
            immutable_indices = np.ascontiguousarray(
                np.asarray(mapping_indices, dtype=np.int64)
            ).copy()
            immutable_indices.setflags(write=False)
            self._flyvis_engine_mapping = {
                "target_indices": immutable_indices,
                "units": flyvis_mapping_metadata.get("units"),
            }
        # A real connectome path must prove every decoder population resolves
        # against the CSR.  The deterministic stub retains permissive mapping
        # for CPU smoke tests, but a missing/ambiguous real map is fail-closed.
        self.neuromuscular = NeuromuscularAdapter(
            strict=(self.options.neural == "malecns"),
        )
        self.sensory_feedback = MechanosensoryFeedback(seed=self.options.seed)
        self.mechanosensory_ablation_sources = self._resolve_mechanosensory_ablation_sources()
        # Strict free-behavior never constructs or invokes the normal flight
        # autopilot.  Keeping the attribute as ``None`` makes accidental use
        # in future changes fail loudly instead of adding an overlay.
        self.autopilot = None if self.is_free_behavior else FlightAutopilot(safety_enabled=True)
        if self.options.neural == "stub":
            self.neural_kind = "stub"
        else:
            module = _require_module("malecns_engine", "neural")
            report_populations: dict[str, str] = {}
            # Expose exactly the populations consumed by the configured
            # neuromuscular map.  The engine then emits windowed population
            # rates, avoiding a guessed projection from a dense CNS vector.
            for channel_name, channel in getattr(self.neuromuscular, "channels", {}).items():
                for term in getattr(channel, "terms", ()):
                    spec = str(getattr(term, "spec", ""))
                    if spec:
                        report_populations[spec] = spec
            # TTMn is the physical leg/jump motor selector in the strict
            # escape protocol.  Keep DNp01/jump as decision telemetry only;
            # if a future neuromuscular map publishes TTMn, the loop above
            # requests its exact selector without a guessed alias.
            if self.is_free_behavior and "ttmn_activity" not in getattr(self.neuromuscular, "channels", {}):
                report_populations.setdefault("TTMn", "TTMn")
            # These visual interneuron selectors are part of the strict escape
            # audit.  They are reported for provenance only; the runner never
            # turns their rates into a directional controller cue.
            if self.is_free_behavior:
                requested_visual_reports = getattr(self, "escape_experiment_spec", {}).get(
                    "visual_report_populations", ("LC4", "LPLC2")
                )
                if isinstance(requested_visual_reports, str):
                    requested_visual_reports = (requested_visual_reports,)
                for spec in requested_visual_reports:
                    spec_name = str(spec).strip()
                    if spec_name:
                        report_populations.setdefault(spec_name, spec_name)
            if self.food_model is not None:
                for spec in self.food_selector_names:
                    report_populations.setdefault(spec, spec)
            if self.neural_intervention_schedule is not None:
                for entry in self.neural_intervention_schedule.entries:
                    report_populations.setdefault(entry.selector, entry.selector)
            if self.grooming_bridge_config is not None:
                for spec in self.grooming_bridge_config.selectors_by_region.values():
                    report_populations.setdefault(spec, spec)
            # External flight-state inputs are delivered in the same engine
            # packet as the mechanosensory/visual channels.  Request a
            # population report for every selector declared by that packet so
            # the real engine can audit all injected populations, not only the
            # two motor readout channels.
            state_drive_config = getattr(self.flight_state_drive, "config", None)
            for rule in getattr(state_drive_config, "rules", ()):
                spec = str(getattr(rule, "population", ""))
                if spec:
                    report_populations[spec] = spec
            flyvis_units = None
            if self.options.vision == "flyvis" and self.flyvis_runtime_adapter is not None:
                flyvis_units = str(getattr(self.flyvis_runtime_adapter, "units", "")) or None
            self.neural_kind = MalecnsNeuralAdapter(
                module, seed=self.options.seed, device=self.options.neural_device,
                backend=self.options.neural_backend,
                max_vram_gib=self.options.max_vram_gib,
                csr_path=self.options.neural_csr_path,
                flyvis_units=flyvis_units,
                report_populations=report_populations or None,
                neural_gain=(
                    self.options.neural_gain
                    if self.options.neural_gain is not None
                    else getattr(self, "escape_experiment_spec", {}).get("neural_gain")
                    if self.is_free_behavior else None
                ),
                cuda_graph=self.options.neural_cuda_graph,
                include_flyvis_drive_echo=not self.minimal_telemetry,
            )
            self.minimal_neural_telemetry = {
                "requested": bool(self.minimal_telemetry),
                "effective": bool(
                    self.minimal_telemetry
                    and not bool(getattr(self.neural_kind, "include_flyvis_drive_echo", True))
                ),
                "interface": "include_flyvis_drive_echo" if self.minimal_telemetry else None,
            }
            self.escape_report_populations = (
                tuple(sorted(report_populations)) if self.is_free_behavior else ()
            )
            if self.grooming_bridge_config is not None:
                populations = getattr(getattr(self.neural_kind, "engine", None), "populations", None)
                if populations is None or not callable(getattr(populations, "resolve", None)):
                    raise BackendSelectionError(
                        "grooming sensory selector audit requires MaleCNS PopulationIndex.resolve"
                    )
                for spec in self.grooming_bridge_config.selectors_by_region.values():
                    count = len(populations.resolve(spec, strict=True))
                    if count <= 0:
                        raise BackendSelectionError(
                            f"grooming sensory selector {spec!r} resolved zero neurons"
                        )
                    self.grooming_selector_counts[spec] = count
            if self.food_model is not None:
                populations = getattr(getattr(self.neural_kind, "engine", None), "populations", None)
                if populations is None or not callable(getattr(populations, "resolve", None)):
                    raise BackendSelectionError(
                        "food receptor audit requires MaleCNS PopulationIndex.resolve"
                    )
                declared_counts = self.food_config_mapping["selector_expected_counts"]
                for spec in self.food_selector_names:
                    resolved_count = len(populations.resolve(spec, strict=True))
                    declared_count = declared_counts[spec]
                    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
                        raise BackendSelectionError(
                            f"food receptor {spec!r} expected count must be an integer"
                        )
                    if resolved_count == 0 or resolved_count != declared_count:
                        raise BackendSelectionError(
                            f"food receptor {spec!r} resolved {resolved_count} cells; "
                            f"expected {declared_count}"
                        )
                    self.food_selector_counts[spec] = resolved_count
            if self.neural_intervention_schedule is not None:
                populations = getattr(getattr(self.neural_kind, "engine", None), "populations", None)
                if populations is None or not callable(getattr(populations, "resolve", None)):
                    raise BackendSelectionError(
                        "neural-intervention audit requires MaleCNS PopulationIndex.resolve"
                    )
                food_selectors = set(self.food_selector_names)
                resolved_intervention: dict[str, np.ndarray] = {}
                for entry in self.neural_intervention_schedule.entries:
                    if entry.selector in food_selectors:
                        raise BackendSelectionError(
                            "direct neuron stimulation may not overlap configured food sensory selectors: "
                            f"{entry.selector!r}"
                        )
                    resolved_ids = populations.resolve(entry.selector, strict=True)
                    resolved_count = len(resolved_ids)
                    if resolved_count == 0:
                        raise BackendSelectionError(
                            f"neural-intervention selector {entry.selector!r} resolved no neurons"
                        )
                    if (entry.expected_match_count is not None
                            and resolved_count != entry.expected_match_count):
                        raise BackendSelectionError(
                            f"neural-intervention selector {entry.selector!r} resolved "
                            f"{resolved_count} cells; expected {entry.expected_match_count}"
                        )
                    previous_count = self.neural_intervention_selector_counts.get(entry.selector)
                    if previous_count is not None and previous_count != resolved_count:
                        raise BackendSelectionError(
                            f"neural-intervention selector {entry.selector!r} resolved inconsistently"
                        )
                    self.neural_intervention_selector_counts[entry.selector] = resolved_count
                    resolved_intervention[entry.selector] = np.asarray(
                        resolved_ids, dtype=np.int64,
                    )
                selectors = sorted(resolved_intervention)
                for left_index, left in enumerate(selectors):
                    for right in selectors[left_index + 1:]:
                        overlap_count = int(np.intersect1d(
                            resolved_intervention[left], resolved_intervention[right],
                            assume_unique=True,
                        ).size)
                        if overlap_count:
                            self.neural_intervention_selector_overlaps[
                                f"{left} <> {right}"
                            ] = overlap_count
            # Warm a reusable engine before the first recorded scenario.  The
            # current engine restores its exact state after Triton warmup;
            # CPU/Torch adapters return an explicit no-op report.
            self.neural_kind.warmup(steps=1)
        self.body_kind = "reduced"
        self.body_model_cls = FlightDynamics

    def _new_neural(self, scenario_name: str) -> Any:
        if self.options.neural == "stub":
            return StubNeural(scenario_name)
        # Real engine instances are stateful; reset once per scenario if the
        # adapter provides it.  The strict constructor already rejected an
        # unsupported module contract.
        neural = self.neural_kind
        reset = getattr(neural, "reset", None)
        if callable(reset):
            reset()
        return neural

    def _ground_walking_config_kwargs(self) -> dict[str, Any]:
        """Validate the opt-in, direct leg-readout ground actuator settings.

        This only configures the physical decoder.  No scenario label, target,
        sensory source, or desired path is used as a motor command.
        """

        raw = self.escape_experiment_spec.get("ground_walking")
        if raw is None:
            self.ground_walking_settings = None
            return {}
        if not isinstance(raw, Mapping):
            raise LauncherError("experiment.ground_walking must be an object")
        allowed = {
            "enabled", "max_ground_traction_N", "max_ground_yaw_torque_Nm",
            "ground_horizontal_braking_N_s_m", "ground_lateral_braking_N_s_m",
            "max_ground_speed_mps", "ground_only",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise LauncherError(
                "experiment.ground_walking has unknown fields: " + ", ".join(sorted(unknown))
            )
        if not isinstance(raw.get("enabled"), bool):
            raise LauncherError("experiment.ground_walking.enabled must be bool")
        if raw["enabled"] and not self.is_free_behavior:
            raise LauncherError("ground walking actuator requires strict free behavior")
        if raw["enabled"] and self.options.body != "reduced":
            raise LauncherError("ground walking actuator requires the reduced body")
        limits = {
            "max_ground_traction_N": 1.0e-5,
            "max_ground_yaw_torque_Nm": 1.0e-8,
            "ground_horizontal_braking_N_s_m": 1.0e-2,
            "ground_lateral_braking_N_s_m": 1.0e-2,
            "max_ground_speed_mps": 0.1,
        }
        settings: dict[str, Any] = {"enabled": raw["enabled"]}
        if "ground_only" in raw:
            if not isinstance(raw["ground_only"], bool):
                raise LauncherError("experiment.ground_walking.ground_only must be bool")
            if raw["ground_only"] and not raw["enabled"]:
                raise LauncherError("ground_only requires enabled ground walking")
            settings["ground_only"] = raw["ground_only"]
        if raw["enabled"]:
            missing = sorted(set(limits) - set(raw))
            if missing:
                raise LauncherError(
                    "enabled ground walking requires explicit calibration fields: "
                    + ", ".join(missing)
                )
        for name, upper in limits.items():
            if name not in raw:
                continue
            value = raw[name]
            if isinstance(value, bool):
                raise LauncherError(f"experiment.ground_walking.{name} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise LauncherError(
                    f"experiment.ground_walking.{name} must be numeric"
                ) from exc
            if not math.isfinite(number) or not 0.0 < number <= upper:
                raise LauncherError(
                    f"experiment.ground_walking.{name} must be in (0, {upper:g}]"
                )
            settings[name] = number
        self.ground_walking_settings = settings
        return {
            "ground_walking_enabled": settings["enabled"],
            "ground_only": settings.get("ground_only", False),
            **{name: settings[name] for name in limits if name in settings},
        }

    def _new_body(self, scenario: Any) -> Any:
        from flight_dynamics import (
            ContactSurface, FlightDynamics, FlightDynamicsConfig, FlightState,
            JumpLegConfig,
        )

        initial = scenario.initial_pose
        state = FlightState(
            time_s=0.0,
            position_m=tuple(initial.position_m),
            velocity_mps=tuple(initial.velocity_mps),
            orientation_quat=_quat_from_rpy(initial.orientation_rpy_rad),
            angular_velocity_rad_s=tuple(initial.angular_velocity_radps),
        )
        # The reduced dynamics stores the body centre, so its unilateral
        # ground plane is the world floor plus the configured body radius.
        # The target is a raised landing surface in the same centre-state
        # coordinates; dynamics resolves it only for a downward crossing from
        # above inside the target footprint.
        body_floor = float(self.world.config.ground_z_m) + float(self.world.config.body_radius_m)
        target = self.world.config.target
        ground_walking_kwargs = self._ground_walking_config_kwargs()
        dynamics_config = FlightDynamicsConfig(
            ground_plane_z_m=body_floor,
            prefer_physical_wing_motor=True,
            allow_legacy_wing_fallback=False if self.is_free_behavior else True,
            contact_surfaces=(ContactSurface(
                surface_id=str(target.id),
                min_x_m=float(target.min_m[0]),
                max_x_m=float(target.max_m[0]),
                min_y_m=float(target.min_m[1]),
                max_y_m=float(target.max_m[1]),
                top_z_m=float(target.max_m[2]) + float(self.world.config.body_radius_m),
            ),),
            **ground_walking_kwargs,
        )
        jump_config = JumpLegConfig(
            mass_kg=float(dynamics_config.mass_kg),
            inertia_kg_m2=tuple(dynamics_config.inertia_kg_m2),
            gravity_mps2=float(dynamics_config.gravity_mps2),
            ground_plane_z_m=body_floor,
        )
        try:
            signature = inspect.signature(FlightDynamics)
        except (TypeError, ValueError):
            signature = None
        if signature is None or "jump_config" in signature.parameters:
            return FlightDynamics(config=dynamics_config, state=state, jump_config=jump_config)
        raise LauncherError(
            "strict escape experiment requires the combined wing+leg FlightDynamics bridge"
        ) if self.is_free_behavior else FlightDynamics(config=dynamics_config, state=state)

    def _wind_model(self, scenario: Any) -> Any:
        override = scenario.wind_override
        return self.world.config.wind.with_override(override) if override else self.world.config.wind

    def _phase_context(self, scenario: Any, t_s: float) -> tuple[float, str, Mapping[str, Any], dict[str, Any]]:
        """Return one authoritative normalized phase clock for all readouts.

        Mission profiles use local time within the active phase; ordinary
        scenarios have one phase spanning the scenario.  Keeping this clock
        shared prevents visual calibration and the optional state-drive
        hypothesis from assigning different meanings to the same ``phase``
        value.
        """

        phase_info: Mapping[str, Any] = {}
        phase_at = getattr(self.world, "phase_at", None)
        if callable(phase_at):
            try:
                candidate = phase_at(scenario, t_s)
                if isinstance(candidate, Mapping):
                    phase_info = candidate
            except (KeyError, TypeError, ValueError):
                phase_info = {}
        start_s = float(phase_info.get("start_s", 0.0))
        end_s = float(phase_info.get("end_s", scenario.duration_s))
        duration_s = max(float(phase_info.get("duration_s", end_s - start_s)), FLYVIS_DT_S)
        local_time_s = float(phase_info.get("local_time_s", max(0.0, t_s - start_s)))
        phase = _clamp(local_time_s / duration_s)
        phase_name = str(phase_info.get("label", phase_info.get("phase", "hold")))
        contract = {
            "name": "normalized_local_phase",
            "range": [0.0, 1.0],
            "clock": "active_world_phase",
            "phase_label": phase_name,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "local_time_s": local_time_s,
        }
        return phase, phase_name, phase_info, contract

    def _visual_state_context(self, scenario: Any, config_name: str, pose: Any,
                              t_s: float) -> dict[str, Any]:
        """Build the scalar context accepted by optional tonic calibration."""

        goal, goal_yaw = self._goal_at(scenario, t_s)
        phase, phase_name, _, phase_contract = self._phase_context(scenario, t_s)
        heading_error = (goal_yaw - float(pose.yaw_rad) + math.pi) % (2.0 * math.pi) - math.pi
        return {
            "scenario": config_name,
            "phase": phase,
            "phase_label": phase_name,
            "phase_contract": phase_contract,
            "altitude_m": float(pose.position_m[2] - self.world.config.ground_z_m),
            # Contract: goal/reference minus measured body altitude, in m.
            "altitude_error_m": float(goal[2] - pose.position_m[2]),
            "heading_error_rad": heading_error,
            "speed_mps": float(pose.speed_mps),
            "vertical_speed_mps": float(pose.velocity_mps[2]),
        }

    def _flight_state_context(self, scenario: Any, config_name: str, pose: Any,
                              t_s: float, contact: bool) -> dict[str, Any]:
        """Build precomputed goal/body errors for the optional state drive.

        ``flight_state_drive`` is an external population-input calibration;
        it must not inspect or integrate body state itself.  The runner owns
        this one error calculation and records the phase/inputs in telemetry.
        """

        goal, goal_yaw = self._goal_at(scenario, t_s)
        phase, phase_name, phase_info, phase_contract = self._phase_context(scenario, t_s)
        drive_name = (
            "grounded" if contact and phase_name.strip().lower() in {"settle", "ground", "grounded", "idle"}
            else phase_name
        )
        dx = float(goal[0]) - float(pose.position_m[0])
        dy = float(goal[1]) - float(pose.position_m[1])
        yaw = float(pose.yaw_rad)
        forward_error = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral_error = -math.sin(yaw) * dx + math.cos(yaw) * dy
        vx = float(pose.velocity_mps[0])
        vy = float(pose.velocity_mps[1])
        forward_velocity = math.cos(yaw) * vx + math.sin(yaw) * vy
        lateral_velocity = -math.sin(yaw) * vx + math.cos(yaw) * vy
        heading_error = (float(goal_yaw) - yaw + math.pi) % (2.0 * math.pi) - math.pi
        return {
            "scenario": drive_name,
            "phase": phase,
            "body_error": {
                "altitude_error_m": float(goal[2]) - float(pose.position_m[2]),
                "heading_error_rad": heading_error,
                "lateral_error_m": lateral_error,
                "forward_error_m": forward_error,
                "forward_velocity_mps": forward_velocity,
                "lateral_velocity_mps": lateral_velocity,
                "vertical_speed_mps": float(pose.velocity_mps[2]),
                "speed_error_mps": 0.0,
                "roll_error_rad": -float(pose.roll_rad),
                "pitch_error_rad": -float(pose.pitch_rad),
                "roll_rate_rad_s": float(pose.angular_velocity_radps[0]),
                "pitch_rate_rad_s": float(pose.angular_velocity_radps[1]),
                "yaw_rate_rad_s": float(pose.angular_velocity_radps[2]),
            },
            "grounded": bool(contact),
            "phase_label": phase_name,
            "phase_contract": phase_contract,
            "phase_index": phase_info.get("index", phase_info.get("phase_index", 0)),
            "goal_position_m": list(goal),
            "goal_yaw_rad": float(goal_yaw),
        }

    def _goal_at(self, scenario: Any, t_s: float) -> tuple[tuple[float, float, float], float]:
        """Use VirtualWorld's authoritative mission goal clock when present."""

        goal_at = getattr(self.world, "goal_at", None)
        if callable(goal_at):
            try:
                reference = goal_at(scenario, t_s)
            except (KeyError, TypeError, ValueError):
                reference = None
            if isinstance(reference, Mapping):
                position = reference.get("position_m")
                yaw = reference.get("yaw_rad")
                if position is not None and yaw is not None:
                    return _vec3(position, "scenario_goal.position_m"), _finite(yaw, "scenario_goal.yaw_rad")
        return _scenario_goal(scenario, t_s)

    def _contact_plane_z(self, dynamics: Any = None) -> float:
        """Return the authoritative body-centre contact plane in metres.

        The reduced dynamics integrates the body centre, so its configured
        ``ground_plane_z_m`` is the world floor plus body radius.  The world
        value remains a fallback for contract-only callers that do not expose
        a dynamics config; the normal launcher path always uses the dynamics
        value installed by ``_new_body``.
        """

        config = getattr(dynamics, "config", None)
        configured = getattr(config, "ground_plane_z_m", None)
        if configured is not None:
            return float(configured)
        return float(self.world.config.ground_z_m) + float(self.world.config.body_radius_m)

    def _body_contact(self, state: Any, dynamics: Any = None) -> bool:
        # ``FlightState.position_m`` is the body centre.  Contact therefore
        # uses the same centre-floor as the integrator, rather than the raw
        # world ground datum.
        floor = self._contact_plane_z(dynamics)
        if state.position_m[2] <= floor + 1.0e-9 and state.velocity_mps[2] <= 1.0e-9:
            return True
        config = getattr(dynamics, "config", None)
        for surface in getattr(config, "contact_surfaces", ()):
            if (
                surface.contains_xy(state.position_m)
                and abs(state.position_m[2] - float(surface.top_z_m)) <= 1.0e-9
                and state.velocity_mps[2] <= 1.0e-9
            ):
                return True
        return False

    def _food_physical_contact(self, state: Any, dynamics: Any) -> bool:
        """Sense current contact with the declared food support, never a goal cue."""

        if self.food_model is None:
            return False
        for surface in getattr(getattr(dynamics, "config", None), "contact_surfaces", ()):
            if str(surface.surface_id) != self.food_model.contact_surface_id:
                continue
            return bool(
                surface.contains_xy(state.position_m)
                and abs(state.position_m[2] - float(surface.top_z_m)) <= 1.0e-9
                and state.velocity_mps[2] <= 1.0e-9
            )
        raise LauncherError("declared food contact surface is missing from reduced dynamics")

    def _clamp_ground(self, dynamics: Any) -> bool:
        """Return post-step contact without applying a second ground clamp.

        ``FlightDynamics.step`` already resolves the unilateral impulse at its
        configured centre-floor and records the resulting contact force.  A
        runner-side position/velocity write here would create a second ground
        model (and, when the planes differ, repeatedly inject an artificial
        upward correction).  Keep this compatibility-named helper read-only.
        """

        return self._body_contact(dynamics.state, dynamics)

    def _feedback_joint_state(self, dynamics: Any) -> dict[str, float]:
        """Expose the prior applied wing command and current WPG phase.

        Mechanosensory feedback at frame ``t`` is causal: it can observe the
        command that produced the previous dynamics output and the dynamics'
        current WPG phase, but not the command that the neural loop will emit
        later in this frame.  A fresh/grounded body has no previous wing
        output, so it is represented as zero-power/no-flapping rather than a
        nominal 210 Hz, unit-amplitude wingbeat.
        """

        wpg = getattr(dynamics, "wpg", None)
        last_output = getattr(dynamics, "last_output", None)
        phase_rad = float(getattr(wpg, "phase_rad", 0.0))
        frequency_hz = float(getattr(wpg, "frequency_hz", 0.0))
        if last_output is None:
            left_power = right_power = 0.0
            left_amplitude = right_amplitude = 0.0
            frequency_hz = 0.0
        else:
            left_power = _clamp(float(getattr(last_output, "left_power", 0.0)))
            right_power = _clamp(float(getattr(last_output, "right_power", 0.0)))
            left_amplitude = max(0.0, float(getattr(last_output, "left_amplitude", 0.0)))
            right_amplitude = max(0.0, float(getattr(last_output, "right_amplitude", 0.0)))
            # Power-off wings do not flap even if the normalized command kept
            # its nominal amplitude field at 1.0.
            if left_power <= 1.0e-12:
                left_amplitude = 0.0
            if right_power <= 1.0e-12:
                right_amplitude = 0.0
            if (left_power <= 1.0e-12 and right_power <= 1.0e-12) or (
                    left_amplitude <= 1.0e-12 and right_amplitude <= 1.0e-12):
                frequency_hz = 0.0
        left_phase_rad = phase_rad + float(getattr(getattr(dynamics, "config", None), "left_phase_offset_rad", 0.0))
        right_phase_rad = phase_rad + float(getattr(getattr(dynamics, "config", None), "right_phase_offset_rad", 0.0))
        return {
            "wing_phase_left_rad": left_phase_rad,
            "wing_phase_right_rad": right_phase_rad,
            "left_power": left_power,
            "right_power": right_power,
            "wing_power": 0.5 * (left_power + right_power),
            "left_amplitude": left_amplitude,
            "right_amplitude": right_amplitude,
            "wingbeat_frequency_hz": max(0.0, frequency_hz),
        }

    def _free_scenario(self) -> Any:
        """Select a scene initializer without reading mission semantics."""

        # A free-experiment config normally contains one ``escape`` scenario.
        # If an embedding caller supplies several scene initializers, the
        # first configured initializer is used; scenario names are labels only
        # and never enter the control packet.
        if not self.world.scenarios:
            raise BackendSelectionError(
                "strict free-behavior config must publish one scene initializer under scenarios"
            )
        preferred = (
            "escape", "escape-experiment", "free-behavior", "free_behavior",
        )
        for name in preferred:
            if name in self.world.scenarios:
                return self.world.scenarios[name]
        return next(iter(self.world.scenarios.values()))

    def _free_world_geometry(self) -> dict[str, Any]:
        """Build a geometry-only world record with goals/phases removed."""
        # Do not call ``VirtualWorld.manifest`` here: that method serializes
        # every configured scenario, including mission profiles.  Strict
        # mode records only physical scene geometry and camera parameters.
        world_config = self.world.config.to_dict()
        camera = getattr(self.world, "camera", None)
        camera_config = camera.to_dict() if callable(getattr(camera, "to_dict", None)) else {}
        return _json_safe({
            "schema_version": "virtual-world-geometry-1",
            "units": "SI",
            "engine": getattr(self.world, "backend_name", "python-raycast"),
            "renderer_backend": getattr(
                self.world, "renderer_backend", getattr(self.world, "backend_role", "analytic_reference")
            ),
            "world": world_config,
            "camera": camera_config,
            "geometry": {
                "ground_z_m": self.world.config.ground_z_m,
                "bounds_xy_m": list(self.world.config.bounds_xy_m),
                "wall_height_m": self.world.config.wall_height_m,
                "target": self.world.config.target.to_dict(),
                "obstacles": [item.to_dict() for item in self.world.config.obstacles],
            },
            **({"external_scene": self.world.external_scene_manifest}
               if getattr(self.world, "external_scene_manifest", None) is not None else {}),
        })

    @staticmethod
    def _free_visual_context(stereo: Any, visual_world: Any) -> dict[str, Any]:
        """Expose only causal pixels/optic flow to a dynamic provider."""

        return {
            "left": _json_safe(getattr(stereo, "left", None)),
            "right": _json_safe(getattr(stereo, "right", None)),
            "optic_flow_rad_s": _json_safe(getattr(visual_world, "optic_flow_rad_s", (0.0, 0.0, 0.0))),
            "timestamp_s": _json_safe(getattr(stereo, "timestamp_s", 0.0)),
        }

    @staticmethod
    def _runtime_exact_direction_pair(runtime: Any) -> tuple[str, Any, Any] | None:
        """Find the fitted runtime's immutable input-index direction pair."""

        # FlyVisRuntimeAdapter keeps this private mapping stable and its
        # public properties intentionally return defensive copies.  Prefer
        # the stable mapping so a strict run does not copy/normalise 1442
        # directions on every frame just to rediscover the same geometry.
        private = getattr(runtime, "_column_directions", None)
        if isinstance(private, Mapping):
            left = private.get("L", private.get("left"))
            right = private.get("R", private.get("right"))
            if left is not None and right is not None:
                return "_column_directions", left, right
        for left_name, right_name in (
            ("left_column_directions", "right_column_directions"),
            ("column_directions_left", "column_directions_right"),
            ("left_eye_directions", "right_eye_directions"),
            ("left_directions", "right_directions"),
        ):
            left = getattr(runtime, left_name, None)
            right = getattr(runtime, right_name, None)
            if left is not None and right is not None:
                return f"{left_name}/{right_name}", left, right
        for mapping_name in ("column_directions", "eye_directions", "input_column_directions"):
            mapping = getattr(runtime, mapping_name, None)
            if isinstance(mapping, Mapping):
                left = mapping.get("left", mapping.get("L"))
                right = mapping.get("right", mapping.get("R"))
                if left is not None and right is not None:
                    return mapping_name, left, right
        return None

    def _prepare_exact_direction_cache(self) -> None:
        """Validate fitted directions once and cache Warp device tensors."""

        self._exact_direction_cache: Optional[dict[str, Any]] = None
        runtime = self.flyvis_runtime_adapter
        if runtime is None:
            return
        pair = self._runtime_exact_direction_pair(runtime)
        if pair is None:
            # Let the strict frame boundary produce the established, explicit
            # missing-direction error.  This keeps stub/legacy construction
            # compatibility while avoiding a guessed pseudo-grid.
            return
        source, left_raw, right_raw = pair
        try:
            left = np.asarray(left_raw, dtype=np.float32)
            right = np.asarray(right_raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise LauncherError("FlyVis exact column directions are not numeric") from exc
        expected = int(getattr(self.flyvis, "n_columns", 0))
        if left.shape != (expected, 3) or right.shape != (expected, 3):
            raise LauncherError(
                "FlyVis exact column directions must have shape "
                f"({expected}, 3), got left={left.shape}, right={right.shape}"
            )
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise LauncherError("FlyVis exact column directions contain non-finite values")
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        if np.any(left_norm <= 1.0e-12) or np.any(right_norm <= 1.0e-12):
            raise LauncherError("FlyVis exact column directions contain zero vectors")
        left = np.ascontiguousarray(left / left_norm[:, None], dtype=np.float32)
        right = np.ascontiguousarray(right / right_norm[:, None], dtype=np.float32)
        cache: dict[str, Any] = {
            "runtime_id": id(runtime),
            "source": source,
            "left": left,
            "right": right,
            "raw_shape_left": tuple(int(size) for size in np.asarray(left_raw).shape),
            "raw_shape_right": tuple(int(size) for size in np.asarray(right_raw).shape),
            # Hashes remain tied to the runtime's original input_index order,
            # not to a separately sorted or re-normalised lattice.
            "directions_sha256": {
                "left": _array_sha256(left_raw),
                "right": _array_sha256(right_raw),
            },
            "validate_source_each_frame": source != "_column_directions",
        }
        if self.options.renderer_backend == "warp_cuda":
            torch = getattr(runtime, "torch", None)
            ctor = getattr(torch, "as_tensor", None) if torch is not None else None
            if not callable(ctor):
                ctor = getattr(torch, "tensor", None) if torch is not None else None
            if not callable(ctor):
                raise RendererBackendError(
                    "warp_cuda exact-column direction cache requires FlyVis torch.as_tensor/tensor"
                )
            dtype = getattr(torch, "float32", None)
            device = str(self.options.vision_device)

            def to_device(array: np.ndarray, name: str) -> Any:
                try:
                    tensor = ctor(array, dtype=dtype, device=device) if dtype is not None else ctor(
                        array, device=device,
                    )
                except TypeError:
                    try:
                        tensor = ctor(array, dtype=dtype) if dtype is not None else ctor(array)
                    except Exception as exc:
                        raise RendererBackendError(
                            f"warp_cuda could not cache {name} direction tensor: {exc}"
                        ) from exc
                    move = getattr(tensor, "to", None)
                    if not callable(move):
                        raise RendererBackendError(
                            f"warp_cuda cached {name} directions cannot move to {device}"
                        )
                    try:
                        tensor = move(device)
                    except Exception as exc:
                        raise RendererBackendError(
                            f"warp_cuda could not move {name} directions to {device}: {exc}"
                        ) from exc
                except Exception as exc:
                    raise RendererBackendError(
                        f"warp_cuda could not cache {name} direction tensor: {exc}"
                    ) from exc
                contiguous = getattr(tensor, "contiguous", None)
                if callable(contiguous):
                    tensor = contiguous()
                return tensor

            cache["left_device"] = to_device(left, "left")
            cache["right_device"] = to_device(right, "right")
            cache["device"] = device
            cache["device_dtype"] = str(getattr(cache["left_device"], "dtype", "float32"))
        self._exact_direction_cache = cache

    def _free_exact_flyvis_columns(self, pose: Any, t_s: float) -> tuple[Any, Any, dict[str, Any]]:
        """Render the fitted FlyVis columns from their exact head directions.

        The rectangular stereo frame remains available to telemetry/video, but
        it is not a valid substitute for the fitted 721-column eye geometry.
        Runtime adapters may publish the pair directly or under a mapping;
        an absent mapping is a strict setup error instead of a pseudo-grid
        fallback.
        """

        if getattr(getattr(self, "options", None), "courtship_movie", None) is not None:
            if self.courtship_movie_stimulus is None:
                runtime = self.flyvis_runtime_adapter
                if runtime is None:
                    raise LauncherError("courtship movie requires the fitted FlyVis runtime")
                pair = self._runtime_exact_direction_pair(runtime)
                if pair is None:
                    raise LauncherError("courtship movie requires exact fitted FlyVis eye directions")
                _source, left_directions, right_directions = pair
                try:
                    from courtship_movie import CourtshipMovie
                    movie_path = Path(self.options.courtship_movie).expanduser().resolve()
                    eye_directions = {
                        "left_directions": np.asarray(left_directions, dtype=np.float32),
                        "right_directions": np.asarray(right_directions, dtype=np.float32),
                    }
                    if movie_path.suffix.lower() == ".npz":
                        self.courtship_movie_stimulus = CourtshipMovie.from_artifact(
                            movie_path, **eye_directions,
                        )
                    else:
                        self.courtship_movie_stimulus = CourtshipMovie.from_video(
                            movie_path,
                            **eye_directions,
                            eye_mode=self.options.courtship_movie_eye_mode,
                            stereo_layout=self.options.courtship_movie_stereo_layout,
                        )
                except Exception as exc:
                    raise LauncherError(f"could not prepare courtship movie columns: {exc}") from exc
                self.courtship_movie_path = movie_path
                self.courtship_movie_sha256 = _sha256_file(movie_path)
            try:
                left, right = self.courtship_movie_stimulus.sample(t_s)
            except Exception as exc:
                raise LauncherError(f"courtship movie sampling failed at t={t_s:g}s: {exc}") from exc
            expected = int(getattr(self.flyvis, "n_columns", 0))
            left = np.asarray(left, dtype=np.float32)
            right = np.asarray(right, dtype=np.float32)
            if (left.shape != (expected,) or right.shape != (expected,)
                    or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
                    or np.any(left < 0.0) or np.any(left > 1.0)
                    or np.any(right < 0.0) or np.any(right > 1.0)):
                raise LauncherError(
                    "courtship movie must provide finite [0,1] exact binocular FlyVis columns"
                )
            left = np.ascontiguousarray(left)
            right = np.ascontiguousarray(right)
            self._exact_column_transfer = {
                "path": "host_numpy_movie_to_flyvis",
                "zero_copy": False,
                "device_resident_inputs": False,
                "host_roundtrip": True,
                "control_sink": "FlyVisRuntimeAdapter.step_detailed",
                "telemetry_copy": "host_float32_lossless_record",
            }
            return left, right, {
                "mode": "open_loop_courtship_movie_exact_columns",
                "n_columns": expected,
                "eye_mode": self.courtship_movie_stimulus.manifest().get("eye_mode"),
                "stereo_layout": self.courtship_movie_stimulus.manifest().get("stereo_layout"),
                "pose_dependent": False,
                "transfer_path": "host_numpy_movie_to_flyvis",
                "movie_ref": "manifest.courtship_visual.movie",
            }

        runtime = self.flyvis_runtime_adapter
        if runtime is None:
            raise LauncherError(
                "strict fitted FlyVis path has no runtime adapter for exact column directions"
            )
        expected = int(getattr(self.flyvis, "n_columns", 0))
        cache = getattr(self, "_exact_direction_cache", None)
        if isinstance(cache, Mapping) and cache.get("runtime_id") == id(runtime):
            source = str(cache.get("source", "cached_runtime_directions"))
            left = cache["left_device"] if "left_device" in cache else cache["left"]
            right = cache["right_device"] if "right_device" in cache else cache["right"]
            left_shape = tuple(int(size) for size in getattr(left, "shape", ()))
            right_shape = tuple(int(size) for size in getattr(right, "shape", ()))
            if left_shape != (expected, 3) or right_shape != (expected, 3):
                raise LauncherError(
                    "cached FlyVis exact column directions have an unexpected shape; "
                    f"got left={left_shape}, right={right_shape}, expected ({expected}, 3)"
                )
            if cache.get("validate_source_each_frame"):
                current = self._runtime_exact_direction_pair(runtime)
                if current is None:
                    raise LauncherError(
                        "strict fitted FlyVis runtime no longer exposes exact column directions"
                    )
                _, current_left, current_right = current
                current_left_shape = getattr(current_left, "shape", None)
                current_right_shape = getattr(current_right, "shape", None)
                if current_left_shape is None:
                    current_left_shape = np.asarray(current_left).shape
                if current_right_shape is None:
                    current_right_shape = np.asarray(current_right).shape
                current_left_shape = tuple(int(size) for size in current_left_shape)
                current_right_shape = tuple(int(size) for size in current_right_shape)
                if (
                    current_left_shape != cache.get("raw_shape_left")
                    or current_right_shape != cache.get("raw_shape_right")
                ):
                    raise LauncherError(
                        "FlyVis exact column direction shape changed after cache validation"
                    )
        else:
            pair = self._runtime_exact_direction_pair(runtime)
            if pair is None:
                raise LauncherError(
                    "strict fitted FlyVis path requires exact left/right column directions; "
                    "refusing pseudo-grid sampling"
                )
            source, left_raw, right_raw = pair
            try:
                left = np.asarray(left_raw, dtype=np.float32)
                right = np.asarray(right_raw, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise LauncherError("FlyVis exact column directions are not numeric") from exc
            if left.shape != (expected, 3) or right.shape != (expected, 3):
                raise LauncherError(
                    "FlyVis exact column directions must have shape "
                    f"({expected}, 3), got left={left.shape}, right={right.shape}"
                )
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise LauncherError("FlyVis exact column directions contain non-finite values")
            left_norm = np.linalg.norm(left, axis=1)
            right_norm = np.linalg.norm(right, axis=1)
            if np.any(left_norm <= 1.0e-12) or np.any(right_norm <= 1.0e-12):
                raise LauncherError("FlyVis exact column directions contain zero vectors")
            left = np.ascontiguousarray(left / left_norm[:, None], dtype=np.float32)
            right = np.ascontiguousarray(right / right_norm[:, None], dtype=np.float32)
            # A direct helper invocation in a unit test may construct a runner
            # without the normal component loader.  Cache that validated pair
            # too, while preserving the normal production cache above.
            self._exact_direction_cache = {
                "runtime_id": id(runtime), "source": source,
                "left": left, "right": right,
                "raw_shape_left": tuple(int(size) for size in np.asarray(left_raw).shape),
                "raw_shape_right": tuple(int(size) for size in np.asarray(right_raw).shape),
                "directions_sha256": {
                    "left": _array_sha256(left_raw),
                    "right": _array_sha256(right_raw),
                },
                "validate_source_each_frame": source != "_column_directions",
            }
            cache = self._exact_direction_cache
        render = getattr(self.world, "render_eye_directions", None)
        if not callable(render):
            raise LauncherError(
                "strict fitted FlyVis path requires VirtualWorld.render_eye_directions"
            )
        try:
            # Warp may return a CUDA tensor.  Preserve that object through the
            # fitted adapter rather than letting ``np.asarray`` force a
            # device-to-host copy before FlyVis sees the exact columns.
            left_rendered = render(pose, left, eye="left", timestamp_s=t_s)
            right_rendered = render(pose, right, eye="right", timestamp_s=t_s)
        except Exception as exc:
            raise LauncherError(f"exact FlyVis column rendering failed: {exc}") from exc

        def validate_rendered(value: Any, name: str) -> Any:
            if _is_device_tensor(value):
                shape = getattr(value, "shape", None)
                try:
                    shape = tuple(int(size) for size in shape)
                except (TypeError, ValueError) as exc:
                    raise LauncherError(
                        f"{name} device tensor has no usable shape"
                    ) from exc
                if shape != (expected,):
                    raise LauncherError(
                        f"VirtualWorld exact column rendering returned {name} shape {shape}; "
                        f"expected ({expected},)"
                    )
                # FlyVisRuntimeAdapter performs the finite-value check with
                # torch.isfinite on its device path.  Do not inspect a CUDA
                # tensor with NumPy here.
                return value
            try:
                array = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise LauncherError(f"{name} exact columns are not numeric") from exc
            if array.shape != (expected,):
                raise LauncherError(
                    f"VirtualWorld exact column rendering returned an unexpected {name} shape "
                    f"{array.shape}; expected ({expected},)"
                )
            if not np.all(np.isfinite(array)):
                raise LauncherError(f"{name} exact columns contain non-finite values")
            return np.ascontiguousarray(array, dtype=np.float32)

        left_columns = validate_rendered(left_rendered, "left")
        right_columns = validate_rendered(right_rendered, "right")
        if getattr(left_columns, "shape", None) != (expected,) or getattr(
            right_columns, "shape", None
        ) != (expected,):
            raise LauncherError(
                "VirtualWorld exact column rendering returned an unexpected shape"
            )
        left_device = _is_device_tensor(left_columns)
        right_device = _is_device_tensor(right_columns)
        if left_device and right_device:
            transfer = {
                "path": "device_tensor_direct_to_flyvis",
                "zero_copy": True,
                "device_resident_inputs": True,
                "host_roundtrip": False,
                "device_to_device_copy": "adapter_persistent_input_copy",
                "control_sink": "FlyVisRuntimeAdapter.step_detailed",
                "telemetry_copy": "host_float32_lossless_record",
                "direction_cache": {
                    "cached": isinstance(cache, Mapping) and "left_device" in cache,
                    "device": cache.get("device") if isinstance(cache, Mapping) else None,
                    "dtype": cache.get("device_dtype") if isinstance(cache, Mapping) else None,
                    "stream": "current_torch_stream",
                },
            }
        elif left_device or right_device:
            transfer = {
                "path": "mixed_device_host_to_flyvis",
                "zero_copy": False,
                "device_resident_inputs": True,
                "host_roundtrip": True,
                "control_sink": "FlyVisRuntimeAdapter.step_detailed",
                "telemetry_copy": "host_float32_lossless_record",
                "direction_cache": {
                    "cached": isinstance(cache, Mapping) and "left_device" in cache,
                    "device": cache.get("device") if isinstance(cache, Mapping) else None,
                    "dtype": cache.get("device_dtype") if isinstance(cache, Mapping) else None,
                    "stream": "current_torch_stream",
                },
            }
        else:
            transfer = {
                "path": "host_numpy_boundary",
                "zero_copy": False,
                "device_resident_inputs": False,
                "host_roundtrip": True,
                "control_sink": "FlyVisRuntimeAdapter.step_detailed",
                "telemetry_copy": "host_float32_lossless_record",
                "direction_cache": {
                    "cached": isinstance(cache, Mapping),
                    "device": None,
                    "dtype": "float32",
                    "stream": "host_numpy",
                },
            }
        self._exact_column_transfer = transfer
        return left_columns, right_columns, {
            "mode": "exact_world_eye_directions",
            "directions_source": source,
            "n_columns": expected,
            "rectangular_stereo": "telemetry_video_only",
            "transfer_path": transfer["path"],
            "zero_copy_control": bool(transfer["zero_copy"]),
            "directions_cached": bool(
                isinstance(cache, Mapping)
                and ("left_device" in cache or "left" in cache)
            ),
            "directions_device": (
                cache.get("device") if isinstance(cache, Mapping) else None
            ),
            "directions_ref": "manifest.components.vision.runtime.column_directions",
        }

    def _process_exact_flyvis_columns(
        self, left_columns: Any, right_columns: Any, timestamp_s: float,
    ) -> Any:
        """Send exact eye columns directly to a fitted runtime when possible.

        ``StereoFlyVis.process_frame`` is a raster frontend and intentionally
        converts vectors through its NumPy sampling/blur path.  The fitted
        FlyVis runtime already exposes ``step_detailed`` with a CUDA-aware
        binocular input seam, so strict exact-column control bypasses that
        frontend.  The tiny host copies in the returned ``StereoOutput`` are
        telemetry only and are made *after* the device control call.
        """

        runtime = self.flyvis_runtime_adapter
        step_detailed = getattr(runtime, "step_detailed", None)
        if not callable(step_detailed):
            # Compatibility with older injected adapters.  Such adapters do
            # not claim a device path and retain the historical frontend.
            return self.flyvis.process_frame(
                left_columns, right_frame=right_columns, timestamp_s=timestamp_s,
            )
        try:
            detailed = step_detailed(left_columns, right_columns)
        except Exception as exc:
            raise LauncherError(
                f"exact FlyVis runtime step failed at t={timestamp_s:g}s: {exc}"
            ) from exc
        drive = getattr(detailed, "drive", getattr(detailed, "t4t5_drive", None))
        if drive is None:
            raise LauncherError(
                "exact FlyVis runtime step returned no drive/t4t5_drive vector"
            )
        try:
            from stereo_flyvis import StereoOutput
            return StereoOutput(
                left_columns=_host_float32_array(left_columns, "left exact columns"),
                right_columns=_host_float32_array(right_columns, "right exact columns"),
                t4t5_drive=_host_float32_array(drive, "FlyVis target drive"),
                frame_index=int(getattr(detailed, "frame_index", 0)),
                timestamp_s=float(timestamp_s),
                cns_tick_start=int(round(float(timestamp_s) / CNS_DT_S)),
                cns_ticks=CNS_STEPS_PER_FLYVIS,
                backend=str(getattr(runtime, "name", "flyvis")),
                is_stub=False,
            )
        except LauncherError:
            raise
        except Exception as exc:
            raise LauncherError(
                f"could not package exact FlyVis runtime output: {exc}"
            ) from exc

    def _preadapt_free_visual_scene(self, dynamics: Any) -> Optional[dict[str, Any]]:
        """Initialise fitted vision on the fixed t=0 scene without behavior.

        Only exact rendered eye columns enter FlyVis here.  No MaleCNS step,
        neuromuscular readout, body integration, wind, collision, label,
        direction, or future stimulus state is evaluated during this phase.
        """

        spec = self.escape_experiment_spec.get("visual_baseline", {})
        if not isinstance(spec, Mapping):
            raise LauncherError("experiment.visual_baseline must be a mapping")
        mode = str(spec.get("mode", "initial_scene_adapted"))
        if mode in {"none", "disabled"}:
            return {"enabled": False, "mode": mode}
        if mode != "initial_scene_adapted":
            raise LauncherError(
                "strict fitted escape vision requires visual_baseline.mode="
                "'initial_scene_adapted'"
            )
        if bool(spec.get("feed_malecns", False)):
            raise LauncherError("visual scene adaptation must not feed MaleCNS")
        if bool(spec.get("advance_body", False)):
            raise LauncherError("visual scene adaptation must not advance the body")
        if self.options.vision != "flyvis" or self.flyvis_runtime_adapter is None:
            return {
                "enabled": False,
                "mode": mode,
                "reason": "fitted_flyvis_not_selected",
                "male_cns_invoked": False,
                "body_invoked": False,
            }
        try:
            duration_s = float(spec.get("duration_s", 1.0))
            scene_time_s = float(spec.get("timestamp_s", 0.0))
        except (TypeError, ValueError) as exc:
            raise LauncherError("visual scene adaptation duration/timestamp must be numeric") from exc
        if abs(scene_time_s) > 1.0e-12:
            raise LauncherError(
                "strict initial-scene adaptation must render the fixed experiment timestamp t=0"
            )
        state_before = dynamics.state.copy()
        pose = _pose_from_state(state_before)
        left, right, sampling = self._free_exact_flyvis_columns(pose, scene_time_s)
        method = getattr(self.flyvis_runtime_adapter, "preadapt_static_scene", None)
        if not callable(method):
            raise LauncherError(
                "strict fitted FlyVis runtime exposes no static-scene preadaptation method"
            )
        try:
            report = method(left, right, seconds=duration_s)
        except Exception as exc:
            raise LauncherError(f"initial-scene FlyVis adaptation failed: {exc}") from exc
        if dynamics.state.to_dict() != state_before.to_dict():
            raise LauncherError("visual scene adaptation changed the authoritative body state")
        return _json_safe({
            **dict(report),
            "configured_mode": mode,
            "configured_timestamp_s": scene_time_s,
            "sampling": sampling,
            "male_cns_invoked": False,
            "neuromuscular_invoked": False,
            "body_invoked": False,
            "body_state_unchanged": True,
            "counts_toward_experiment_duration": False,
        })

    @staticmethod
    def _escape_channel_sensory_values(packet: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        channels = packet.get("channels", {})
        if not isinstance(channels, Mapping):
            raise LauncherError("escape channel packet channels must be a mapping")
        for role in ("dynamic_stimulus", "jump_channel", "posture_channel"):
            role_values = channels.get(role)
            if not isinstance(role_values, Mapping):
                raise LauncherError(f"escape channel packet is missing {role} mapping")
            # Nested role mappings are the canonical interface.  Flat aliases
            # are included for future engines that register named channels.
            values[f"escape:{role}"] = _json_safe(role_values)
            for key, item in role_values.items():
                if str(key).startswith("escape:"):
                    continue
                values[f"escape:{role}:{key}"] = _json_safe(item)
        for key, item in channels.items():
            key_text = str(key)
            if key_text.startswith("escape:"):
                values[key_text] = _json_safe(item)
        return values

    def _resolve_mechanosensory_ablation_sources(self) -> dict[str, tuple[str, ...]]:
        """Validate complete source-alias coverage for selected ablations."""

        if not self.mechanosensory_ablations:
            return {}
        resolver = getattr(self.sensory_feedback, "resolver", None)
        drive_mapping = getattr(resolver, "drive_mapping", None)
        entries = drive_mapping.get("entries") if isinstance(drive_mapping, Mapping) else None
        if not isinstance(entries, Mapping):
            raise BackendSelectionError(
                "--ablate-mechanosensory requires a validated sensory drive-mapping entries table"
            )

        result: dict[str, tuple[str, ...]] = {}
        claimed_sources: set[str] = set()
        for group in self.mechanosensory_ablations:
            aliases: list[str] = []
            for entry_name, (target_population, expected_sources) in \
                    MECHANOSENSORY_ABLATION_ENTRIES[group].items():
                entry = entries.get(entry_name)
                if not isinstance(entry, Mapping):
                    raise BackendSelectionError(
                        f"--ablate-mechanosensory {group} requires sensory-map entry {entry_name!r}"
                    )
                actual_target = str(entry.get("target_population", entry.get("target", ""))).strip()
                if actual_target != target_population:
                    raise BackendSelectionError(
                        f"sensory-map entry {entry_name!r} targets {actual_target!r}; "
                        f"expected {target_population!r} for {group} ablation"
                    )
                raw_sources = entry.get("sources")
                if (not isinstance(raw_sources, Sequence)
                        or isinstance(raw_sources, (str, bytes))):
                    raise BackendSelectionError(
                        f"sensory-map entry {entry_name!r} has no source-alias sequence"
                    )
                actual_sources = tuple(str(source).strip() for source in raw_sources)
                if (any(not source for source in actual_sources)
                        or len(actual_sources) != len(set(actual_sources))
                        or set(actual_sources) != set(expected_sources)):
                    raise BackendSelectionError(
                        f"sensory-map aliases for {entry_name!r} do not match the audited "
                        f"ablation contract; expected {list(expected_sources)}, got {list(actual_sources)}"
                    )
                aliases.extend(actual_sources)
            if len(aliases) != len(set(aliases)):
                raise BackendSelectionError(
                    f"sensory-map aliases overlap within {group} ablation"
                )
            overlap = claimed_sources.intersection(aliases)
            if overlap:
                raise BackendSelectionError(
                    "sensory-map aliases overlap between selected ablations: "
                    + ", ".join(sorted(overlap))
                )
            claimed_sources.update(aliases)
            result[group] = tuple(aliases)
        return result

    def _masked_mechanosensory_population_drive(
        self, mech: Any,
    ) -> tuple[Mapping[str, Any], Optional[dict[str, Any]]]:
        """Return the CNS-facing population drive and intervention record.

        The source ``mech`` object is deliberately left untouched: it remains
        the raw body/world-derived feedback record in telemetry.
        """

        raw_drive = getattr(mech, "population_drive", None)
        if not self.mechanosensory_ablations:
            if not isinstance(raw_drive, Mapping):
                raise LauncherError("mechanosensory feedback must publish population_drive")
            return raw_drive, None
        if not isinstance(raw_drive, Mapping):
            raise LauncherError(
                "mechanosensory ablation requires a population_drive mapping; refusing an unmasked packet"
            )
        source_aliases = {
            group: list(self.mechanosensory_ablation_sources[group])
            for group in self.mechanosensory_ablations
        }
        all_aliases = tuple(
            source for group in self.mechanosensory_ablations
            for source in self.mechanosensory_ablation_sources[group]
        )
        missing = [source for source in all_aliases if source not in raw_drive]
        if missing:
            raise LauncherError(
                "mechanosensory feedback omitted selected ablation aliases: "
                + ", ".join(missing)
            )
        masked = dict(raw_drive)
        raw_values = {source: _json_safe(raw_drive[source]) for source in all_aliases}
        for source in all_aliases:
            masked[source] = 0.0
        record = {
            "schema_version": "male-cns-mechanosensory-ablation-1",
            "site": "sensory_frame.channels population:<source> before MaleCNS",
            "selected_populations": list(self.mechanosensory_ablations),
            "source_aliases": source_aliases,
            "pre_ablation_population_drive": raw_values,
            "male_cns_population_channels": {
                f"population:{source}": masked[source] for source in all_aliases
            },
            "world_body_physics_unchanged": True,
            "raw_feedback_retained_in_mechanosensory_record": True,
        }
        return masked, record

    def _validate_mechanosensory_ablation_channels(
        self, sensory_channels: Mapping[str, Any],
    ) -> None:
        """Fail closed if any selected source alias reaches MaleCNS nonzero."""

        if not self.mechanosensory_ablations:
            return
        aliases = tuple(
            source for group in self.mechanosensory_ablations
            for source in self.mechanosensory_ablation_sources[group]
        )

        def require_zero(value: Any, name: str) -> None:
            candidate = value
            if isinstance(value, Mapping):
                value_fields = [key for key in ("value", "drive", "normalized") if key in value]
                if len(value_fields) != 1:
                    raise LauncherError(
                        f"mechanosensory ablation found an invalid value at {name}"
                    )
                candidate = value[value_fields[0]]
            try:
                array = np.asarray(candidate, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise LauncherError(
                    f"mechanosensory ablation found a non-numeric value at {name}"
                ) from exc
            if array.ndim != 0 or not np.isfinite(array).item() or float(array) != 0.0:
                raise LauncherError(
                    f"mechanosensory ablation did not zero MaleCNS input {name}"
                )

        for source in aliases:
            key = f"population:{source}"
            if key not in sensory_channels:
                raise LauncherError(
                    f"mechanosensory ablation packet is missing required MaleCNS alias {key}"
                )
            require_zero(sensory_channels[key], key)

        explicit = sensory_channels.get("population_drive")
        if explicit is not None:
            if not isinstance(explicit, Mapping):
                raise LauncherError(
                    "mechanosensory ablation found a malformed explicit population_drive packet"
                )
            value_field = next(
                (field for field in ("values", "drives") if field in explicit), None,
            ) if ("units" in explicit or "input_units" in explicit) else None
            explicit_values = explicit.get(value_field) if value_field else explicit
            if not isinstance(explicit_values, Mapping):
                raise LauncherError(
                    "mechanosensory ablation found a malformed explicit population_drive mapping"
                )
            for source in aliases:
                if source in explicit_values:
                    require_zero(explicit_values[source], f"population_drive[{source}]")

    def _mechanosensory_ablation_manifest(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.mechanosensory_ablations),
            "selected_populations": list(self.mechanosensory_ablations),
            "source_aliases": {
                group: list(self.mechanosensory_ablation_sources.get(group, ()))
                for group in self.mechanosensory_ablations
            },
            "site": "sensory_frame.channels population:<source> before MaleCNS",
            "operation": "set every selected source alias to normalized 0.0",
            "raw_feedback_record": "mechanosensory.population_drive is retained unmodified",
            "world_body_physics_unchanged": True,
            "telemetry_record": "mechanosensory_ablation",
        }

    def _neural_intervention_manifest(self) -> Optional[dict[str, Any]]:
        schedule = self.neural_intervention_schedule
        if schedule is None:
            return None
        return {
            "enabled": True,
            "classification": schedule.classification,
            "name": schedule.name,
            "config_path": str(self.neural_intervention_config_path),
            "config_sha256": self.neural_intervention_config_sha256,
            "intervention_type": schedule.intervention_type,
            "opt_in_required": True,
            "counts_as_autonomous_foraging": False,
            "input_site": "sensory_frame.channels.poisson_hz/current_mV_per_ms before MaleCNS",
            "resolved_selector_counts": dict(self.neural_intervention_selector_counts),
            "selector_overlap_counts": dict(self.neural_intervention_selector_overlaps),
            "overlap_drive_policy": "per-selector rates add on shared neurons at the MaleCNS drive boundary",
            "entries": [
                {
                    "selector": entry.selector,
                    "expected_match_count": entry.expected_match_count,
                    "resolved_match_count": self.neural_intervention_selector_counts.get(
                        entry.selector
                    ),
                    "units": entry.units,
                    "start_s": entry.start_s,
                    "end_s": entry.end_s,
                    "value": entry.value,
                }
                for entry in schedule.entries
            ],
            "body_or_motor_command_injected": False,
            "target_goal_or_autopilot_injected": False,
            "prohibited_outputs": list(schedule.prohibited_outputs),
            "runtime_software_bounds": {
                units: list(bounds) for units, bounds in schedule.bounds.items()
            },
        }

    def _free_neural_frame(self, neural: Any, sensory: SensoryFrame,
                           ts: Timestamp, t_s: float) -> tuple[NeuralOutput, int, str]:
        """Run exactly one MaleCNS 5 ms window, with no controller branch."""

        window_output = None
        window_method = getattr(neural, "step_window", None)
        if callable(window_method):
            window_output = window_method(sensory, ts, steps=CNS_STEPS_PER_FLYVIS)
        if window_output is not None:
            neural_output = window_output
            if neural_output.timestamp != ts:
                with_timestamp = getattr(neural_output, "with_timestamp", None)
                if callable(with_timestamp):
                    neural_output = with_timestamp(ts)
                else:
                    neural_output = NeuralOutput(
                        ts, rates_hz=neural_output.rates_hz,
                        spikes=neural_output.spikes,
                        named_outputs=neural_output.named_outputs,
                        source=neural_output.source,
                    )
            try:
                neural_steps = int(neural_output.named_outputs.get(
                    "neural_window_steps", CNS_STEPS_PER_FLYVIS
                ))
            except (TypeError, ValueError) as exc:
                raise LauncherError("free-behavior neural window has invalid step count") from exc
            if neural_steps != CNS_STEPS_PER_FLYVIS:
                raise LauncherError(
                    "free-behavior MaleCNS output must integrate exactly "
                    f"{CNS_STEPS_PER_FLYVIS} CNS ticks (got {neural_steps})"
                )
            api = str(neural_output.named_outputs.get(
                "neural_window_api", "engine_step_window"
            ))
            neural_output.named_outputs["neural_window_api"] = api
            return neural_output, neural_steps, api
        cns_outputs: list[NeuralOutput] = []
        for substep in range(CNS_STEPS_PER_FLYVIS):
            cns_t_s = t_s + substep * CNS_DT_S
            cns_ts = _timestamp(cns_t_s, clock="neural", rate_hz=CNS_HZ)
            cns_frame = _substep_sensory(sensory, cns_ts)
            cns_outputs.append(neural.step(cns_frame, cns_ts))
        return _aggregate_neural_outputs(cns_outputs, ts, window_dt_s=FLYVIS_DT_S), len(cns_outputs), "scalar_step_substeps"

    @staticmethod
    def _free_jump_input(nm_out: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the combined-body motor packet from NM output only.

        TTMn is the physical leg/jump motor activity.  DNp01/``jump`` is
        retained as decision telemetry but is deliberately not used as the
        SI actuator drive.  The combined FlightDynamics bridge owns event
        latching, contact gating, and force/impulse conversion.
        """

        physical: Mapping[str, Any] = {}
        candidate = getattr(nm_out, "to_physical_channels", None)
        if callable(candidate):
            raw = candidate()
            if isinstance(raw, Mapping):
                physical = raw
        if not physical:
            candidate = getattr(nm_out, "physical_channels", None)
            if isinstance(candidate, Mapping):
                physical = candidate
        channels = getattr(nm_out, "channels", {})
        if not isinstance(channels, Mapping):
            channels = {}
        def value(*names: str, default: Any = 0.0) -> Any:
            for name in names:
                if name in physical:
                    return physical[name]
                if name in channels:
                    return channels[name]
                attr = getattr(nm_out, name, None)
                if attr is not None:
                    return attr
            return default
        ttmn = value("ttmn_activity", "TTMn", "ttmn")
        # Do not let an absent TTMn selector silently turn DNp01 into a jump.
        if ttmn is None or not math.isfinite(float(ttmn)):
            raise LauncherError(
                "strict escape body bridge requires NeuromuscularAdapter TTMn activity"
            )
        jump_input = {
            "jump_activation": _clamp(float(ttmn)),
            "ttmn_activity": _clamp(float(ttmn)),
            "activation_source": "ttmn_activity",
            # TTMn is itself the bilateral jump-muscle actuator.  Generic
            # pooled leg-power channels must not gate an otherwise valid TTMn
            # event; explicit asymmetry/posture channels remain optional.
            "leg_power": 1.0,
            "leg_steering": 0.0,
            "posture_roll": max(-1.0, min(1.0, float(value("posture_roll")))),
            "posture_pitch": max(-1.0, min(1.0, float(value("posture_pitch")))),
            "posture_yaw": max(-1.0, min(1.0, float(value("posture_yaw")))),
            "source": "connectome",
        }
        decision = {
            "dnp01_activity": _json_safe(value("jump", "jump_activation", default=0.0)),
            "ttmn_activity": _json_safe(ttmn),
            "dnp11_activity": _json_safe(value("dnp11_activity", "DNp11", default=0.0)),
            "dnp02_activity": _json_safe(value("dnp02_activity", "DNp02", default=0.0)),
            "dnp04_activity": _json_safe(value("dnp04_activity", "DNp04", default=0.0)),
            "actuator_source": "TTMn",
        }
        return jump_input, decision

    @staticmethod
    def _free_dynamics_step(dynamics: Any, nm_out: Any, jump_input: Mapping[str, Any],
                            *, dt_s: float) -> Any:
        """Invoke exactly one combined wing+leg FlightDynamics integrator."""

        # Prefer a named combined method when the jump-physics agent exposes
        # one; this keeps the runner independent of its chosen spelling.
        for method_name in (
            "step_with_jump", "step_combined", "step_wing_leg",
            "step_neuromuscular",
        ):
            method = getattr(dynamics, method_name, None)
            if not callable(method):
                continue
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                signature = None
            if signature is None:
                return method(nm_out, jump_input, dt_s=dt_s)
            names = signature.parameters
            kwargs: dict[str, Any] = {"dt_s": dt_s}
            jump_name = next((name for name in (
                "jump_input", "jump_command", "leg_input", "jump_channels",
            ) if name in names), None)
            if jump_name is None:
                raise LauncherError(
                    f"combined body method {method_name} exposes no jump input boundary"
                )
            kwargs[jump_name] = jump_input
            if "command" in names:
                kwargs["command"] = nm_out
                return method(**kwargs)
            return method(nm_out, **kwargs)

        # A compatible bridge may extend the regular step signature with an
        # explicit jump keyword.  Never fall back to the wing-only path.
        method = getattr(dynamics, "step", None)
        if callable(method):
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                names = signature.parameters
                jump_name = next((name for name in (
                    "jump_input", "jump_command", "leg_input", "jump_channels",
                ) if name in names), None)
                if jump_name is not None:
                    kwargs = {"dt_s": dt_s, jump_name: jump_input}
                    if "command" in names:
                        kwargs["command"] = nm_out
                        return method(**kwargs)
                    return method(nm_out, **kwargs)
        raise LauncherError(
            "strict escape experiment requires one combined wing+leg FlightDynamics integration method"
        )

    def _one_free_scenario(self) -> dict[str, Any]:
        """Run the strict free-behavior escape experiment.

        This loop intentionally does not call ``_goal_at``, ``_phase_context``,
        ``_target_info``, ``FlightAutopilot``, ``FlightStateDrive``, or
        ``FlightBodyStabilizer``.  The only command crossing into dynamics is
        the object returned by ``NeuromuscularAdapter.step``.
        """

        if self.options.neural != "malecns" or isinstance(self.neural_kind, StubNeural):
            raise BackendSelectionError(
                "strict free-behavior mode cannot execute with a scripted/stub neural endpoint"
            )
        scenario = self._free_scenario()
        duration = scenario.duration_s if self.options.duration_override_s is None else self.options.duration_override_s
        if self.options.frame_count is not None:
            duration = self.options.frame_count * FLYVIS_DT_S
        if self.options.max_steps is not None:
            duration = min(duration, self.options.max_steps * FLYVIS_DT_S)
        duration = max(FLYVIS_DT_S, float(duration))
        frame_count = int(round(duration / FLYVIS_DT_S))
        if abs(duration - frame_count * FLYVIS_DT_S) > 1.0e-8:
            raise LauncherError(
                "free-behavior duration must be aligned to the 5 ms FlyVis frame period"
            )
        duration = frame_count * FLYVIS_DT_S
        dynamics = self._new_body(scenario)
        self.flyvis.reset()
        self.visual_scene_baseline_report = self._preadapt_free_visual_scene(dynamics)
        neural = self._new_neural("free-behavior")
        if self.visual_calibration is not None:
            self.visual_calibration.reset()
        self.sensory_feedback.reset()
        self.neuromuscular.reset(clear_baseline=False)
        self.escape_channels.reset()
        grooming_body = None
        grooming_bridge = None
        grooming_initial_particles = None
        if self.grooming_bridge_config is not None:
            try:
                from .grooming_body import GroomingBody
                from .grooming_neural_bridge import GroomingNeuralBridge
            except ImportError:  # direct script execution
                from grooming_body import GroomingBody
                from grooming_neural_bridge import GroomingNeuralBridge
            grooming_body = GroomingBody(
                dust_seed=self.options.seed,
                dust_by_region={"antenna": self.options.grooming_dust_count},
            )
            grooming_initial_particles = {
                region: [
                    {"particle_id": item.particle_id,
                     "position_mm": list(item.position_mm)}
                    for item in particles
                ]
                for region, particles in grooming_body.particle_positions().items()
            }
            grooming_bridge = GroomingNeuralBridge(self.grooming_bridge_config)
        wind_model = self._wind_model(scenario)
        geometry = self._free_world_geometry()
        rows: list[dict[str, Any]] = []
        collisions: list[dict[str, Any]] = []
        contact_count = 0
        max_altitude = -math.inf
        min_altitude = math.inf
        max_displacement = 0.0
        max_speed = 0.0
        max_flow = 0.0
        initial_position = tuple(float(item) for item in dynamics.state.position_m)
        stage_wall_s = {
            "world": 0.0,
            "vision": 0.0,
            "sensory": 0.0,
            "sensory_prepare_wait": 0.0,
            "neural": 0.0,
            "neuromuscular": 0.0,
            "dynamics": 0.0,
            "telemetry": 0.0,
            "frame_total": 0.0,
        }
        step_index = 0
        t_s = 0.0
        while t_s < duration - 1.0e-12:
            frame_started = time.perf_counter()
            stage_started = frame_started
            dt = min(FLYVIS_DT_S, duration - t_s)
            if abs(dt - FLYVIS_DT_S) > 1.0e-9:
                raise LauncherError("free-behavior scheduler produced a partial frame")
            ts = _timestamp(t_s)
            state_before = dynamics.state.copy()
            pose = _pose_from_state(state_before)
            contact_before = self._body_contact(state_before, dynamics)
            wind = self.world.wind_at(pose.position_m, t_s, wind_model)
            food_sample = None
            food_cue_record = None
            grooming_sample = None
            grooming_loop_record = None
            if grooming_bridge is not None and grooming_body is not None:
                grooming_sample = grooming_bridge.sample_input(
                    grooming_body.tactile_loads(), dt,
                    sensor_enabled=not self.options.grooming_sensor_off,
                )
            if self.food_model is not None:
                food_sample = self.food_model.sample(
                    position_m=pose.position_m,
                    yaw_rad=pose.yaw_rad,
                    time_s=t_s,
                    physical_contact=self._food_physical_contact(state_before, dynamics),
                    wind_mps=wind,
                )
            body_contract = _make_contract_body(state_before, ts)
            hard_hits, soft_hits = self.world.collision_events(pose, t_s)
            collisions.extend(item.to_dict() for item in hard_hits)
            # The palm is visual-only/non-interacting in this protocol.  Do
            # not turn a counterfactual surface crossing into a body event;
            # the planned deadline is recorded from experiment metadata.
            contact_count += int(contact_before) + len(soft_hits)
            # In the fitted strict path the rectangular raster is a video/
            # telemetry aid only: MaleCNS consumes the exact eye columns below
            # and no controller reads the rectangular optic-flow sample.  Skip
            # this second raycast in compact production mode when no external
            # dynamic provider requested the raster as context.  A provider
            # (or full-debug mode) keeps the historical render path and its
            # explicit manifest disclosure.
            retain_rectangular = (
                not self.compact_telemetry
                or bool(self.escape_channels.roles)
                or self.options.vision != "flyvis"
            )
            stereo = None
            visual_world = None
            if retain_rectangular:
                stereo = self.world.render_stereo(
                    pose, (self.options.width, self.options.height), timestamp_s=t_s,
                )
                visual_world = self.world.visual_observation(stereo, t_s, dt_s=FLYVIS_DT_S)
            stage_wall_s["world"] += time.perf_counter() - stage_started
            sensory_prepare_job: Any = None
            if self._frame_stage_pipeline.enabled:
                prepare_step = getattr(self.sensory_feedback, "prepare_step", None)
                commit_prepared = getattr(self.sensory_feedback, "commit_prepared", None)
                if not callable(prepare_step) or not callable(commit_prepared):
                    raise LauncherError(
                        "parallel frame stages require sensory prepare_step/commit_prepared"
                    )
                sensory_prepare_job = self._frame_stage_pipeline.start_sensory_prepare(
                    prepare_step,
                    body_contract,
                    world_wind=wind,
                    ground_contact=contact_before,
                    joint_state=self._feedback_joint_state(dynamics),
                )
            stage_started = time.perf_counter()
            if self.options.vision == "flyvis":
                # The fitted model consumes its exact hex-column samples.  A
                # rectangular frame is retained above only for telemetry and
                # video; it never gets silently resampled into a pseudo-grid.
                left_columns, right_columns, flyvis_sampling = self._free_exact_flyvis_columns(
                    pose, t_s,
                )
                flyvis_out = self._process_exact_flyvis_columns(
                    left_columns, right_columns, t_s,
                )
            else:
                flyvis_sampling = {
                    "mode": "rectangular_stub",
                    "rectangular_stereo": "controller_input",
                }
                # The deterministic stub has no exact-column runtime.  Its
                # small rectangular frame remains the controller input even
                # when a caller selected compact telemetry.
                if stereo is None or visual_world is None:
                    stereo = self.world.render_stereo(
                        pose, (self.options.width, self.options.height), timestamp_s=t_s,
                    )
                    visual_world = self.world.visual_observation(stereo, t_s, dt_s=FLYVIS_DT_S)
                flyvis_out = self.flyvis.process_frame(
                    visual_world, right_frame=stereo.right, timestamp_s=t_s,
                )
            # Physical stimulus geometry is queried for telemetry only.  The
            # same object is already consumed by the world's render/wind/
            # collision methods above; its labels, direction, and future
            # position never enter the neural/control packet.
            stimulus_state = self.world.stimulus_state_at(
                t_s, reference_pose=pose,
            )
            calibration_metadata: dict[str, Any] = {}
            calibration_mapping: Any = None
            if self.visual_calibration is not None:
                mapping = _flyvis_mapping_metadata(self.flyvis)
                calibrated_drive, calibration_metadata = self.visual_calibration.apply(
                    flyvis_out.t4t5_drive,
                    scenario="free-behavior",
                    timestamp_s=t_s,
                    frame_index=step_index,
                    visual_output=flyvis_out,
                    target_indices=mapping.get("target_indices"),
                    target_ids=mapping.get("target_ids"),
                    units=mapping.get("units"),
                )
                calibration_mapping = calibration_metadata.get("engine_mapping")
                flyvis_out = replace(flyvis_out, t4t5_drive=calibrated_drive)
            # Directional visual readouts are telemetry only in free mode.
            # They are intentionally not placed in ``sensory_channels``.
            visual_drive_summary = _drive_summary(flyvis_out.t4t5_drive)
            frame_flow = max(
                abs(float(visual_drive_summary.get("min", 0.0))),
                abs(float(visual_drive_summary.get("max", 0.0))),
            ) if visual_drive_summary.get("finite", False) else 0.0
            max_flow = max(max_flow, frame_flow)
            # Minimal rows retain one hash-only fingerprint; the row builder
            # reuses this value for both the audit record and summary field.
            visual_drive_sha256 = (
                _array_sha256(flyvis_out.t4t5_drive)
                if self.minimal_telemetry else None
            )
            visual = VisualObservation(
                timestamp=ts,
                # Keep the live exact-column arrays in the contract.  Compact
                # serialization encodes them once below; full-debug's
                # ``to_dict`` still converts them to JSON lists.
                pixels=(
                    None if self.minimal_telemetry
                    else [flyvis_out.left_columns, flyvis_out.right_columns]
                ),
                # Rectangular optic flow is telemetry-only in strict fitted
                # compact mode.  The exact FlyVis columns remain the causal
                # visual input, so an omitted raster contributes an explicit
                # zero vector rather than an invented controller cue.
                optic_flow_rad_s=(
                    tuple(visual_world.optic_flow_rad_s)
                    if visual_world is not None else (0.0, 0.0, 0.0)
                ),
                sensor_id="virtual-world-stereo-flyvis",
                source=SourceLabel.HANDBUILT if flyvis_out.is_stub else SourceLabel.FITTED,
                metadata={
                    **flyvis_out.metadata,
                    "world_sensor_backend": getattr(self.world, "backend_name", "virtual_world"),
                    "flyvis_sampling": flyvis_sampling,
                    "visual_calibration": _json_safe({
                        key: value for key, value in calibration_metadata.items()
                        if key not in {"engine_mapping", "target_indices", "target_ids"}
                    }) if calibration_metadata else None,
                    "free_behavior_directional_readout": "record_only",
                },
            )
            stage_wall_s["vision"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            sensory_prepare_compute_s = 0.0
            sensory_prepare_wait_s = 0.0
            if sensory_prepare_job is not None:
                prepared, sensory_prepare_compute_s, sensory_prepare_wait_s = (
                    self._frame_stage_pipeline.finish_sensory_prepare(
                        sensory_prepare_job
                    )
                )
                mech = self.sensory_feedback.commit_prepared(prepared)
            else:
                mech = self.sensory_feedback.step(
                    body_contract,
                    world_wind=wind,
                    ground_contact=contact_before,
                    joint_state=self._feedback_joint_state(dynamics),
                )
            if self.escape_channels.roles:
                if stereo is None or visual_world is None:
                    raise LauncherError(
                        "dynamic strict escape provider requires rectangular visual context"
                    )
                escape_packet = self.escape_channels.step(
                    timestamp_s=t_s,
                    dt_s=dt,
                    frame_index=step_index,
                    body=body_contract.to_dict(),
                    visual=self._free_visual_context(stereo, visual_world),
                    experiment=self.escape_experiment_spec,
                )
            else:
                escape_packet = {
                    "timestamp_s": float(t_s),
                    "frame_index": int(step_index),
                    "channels": {},
                    "records": {},
                    "experiment": _json_safe(self.escape_experiment_spec),
                    "provenance": {
                        "physical_world_stimulus": True,
                        "label_control": False,
                        "direction_control": False,
                        "future_state_control": False,
                        "causal_context": True,
                    },
                }
            flyvis_drive_channels = _flyvis_drive_contract(
                self.flyvis, flyvis_out.t4t5_drive,
                mapping_override=(
                    calibration_mapping
                    if calibration_mapping is not None
                    else self._flyvis_engine_mapping
                ),
            )
            mapped_flyvis = flyvis_drive_channels.get("flyvis_t4t5")
            if self.minimal_telemetry and isinstance(mapped_flyvis, dict):
                value_count = int(visual_drive_summary.get("count", 0))
                mapped_flyvis["_telemetry_summary"] = {
                    "units": mapped_flyvis.get("units"),
                    "target_count": int(self.flyvis_mapping.get("count", 0)),
                    "target_indices_sha256": self.flyvis_mapping.get("sha256"),
                    "value_count": value_count,
                    "values_sha256": visual_drive_sha256,
                    "values_min": visual_drive_summary.get("min"),
                    "values_max": visual_drive_summary.get("max"),
                    "values_sum": (
                        float(visual_drive_summary.get("mean", 0.0)) * value_count
                    ),
                }
            male_cns_population_drive, mechanosensory_ablation_record = (
                self._masked_mechanosensory_population_drive(mech)
            )
            sensory_channels: dict[str, Any] = {
                "optic_flow_x_rad_s": visual.optic_flow_rad_s[0],
                "optic_flow_y_rad_s": visual.optic_flow_rad_s[1],
                # Raw signed FlyVis drive is a sensory input to MaleCNS.  No
                # scalar directional cue or controller interpretation crosses
                # this boundary.
                **flyvis_drive_channels,
                **{f"mech:{key}": value for key, value in mech.channels.items()},
                **{f"mech_raw:{key}": value for key, value in mech.raw_channels.items()},
                **{f"population:{key}": value for key, value in male_cns_population_drive.items()},
                "population_drive_units": str(getattr(mech, "population_drive_units", "normalized")),
                "population_drive_range": list(getattr(mech, "population_drive_range", (-1.0, 1.0))),
            }
            if food_sample is not None:
                raw_odor_rates = dict(food_sample.odor_poisson_hz)
                raw_taste_rates = dict(food_sample.sugar_poisson_hz)
                cns_odor_rates = {
                    spec: 0.0 if "odor" in self.food_cue_ablations else rate
                    for spec, rate in raw_odor_rates.items()
                }
                cns_taste_rates = {
                    spec: 0.0 if "taste" in self.food_cue_ablations else rate
                    for spec, rate in raw_taste_rates.items()
                }
                cns_food_rates = {**cns_odor_rates, **cns_taste_rates}
                sensory_channels["poisson_hz"] = cns_food_rates
                food_cue_record = {
                    **dict(food_sample.telemetry),
                    "raw_odor_poisson_hz": raw_odor_rates,
                    "raw_taste_poisson_hz": raw_taste_rates,
                    "cns_poisson_hz": cns_food_rates,
                    "ablated_senses": list(self.food_cue_ablations),
                    "receptor_count_ref": "manifest.components.food_sensing.selector_counts",
                }
            if grooming_sample is not None:
                grooming_rates = dict(grooming_sample.poisson_hz)
                if grooming_rates:
                    existing_rates = dict(sensory_channels.get("poisson_hz", {}))
                    for selector, rate in grooming_rates.items():
                        existing_rates[selector] = math.fsum((
                            float(existing_rates.get(selector, 0.0)), float(rate),
                        ))
                    sensory_channels["poisson_hz"] = existing_rates
            neural_intervention_record = None
            if self.neural_intervention_schedule is not None:
                intervention_rates = self.neural_intervention_schedule.rates_at(t_s)
                intervention_currents = self.neural_intervention_schedule.currents_at(t_s)
                if intervention_rates:
                    combined_rates = dict(sensory_channels.get("poisson_hz", {}))
                    for selector, rate in intervention_rates.items():
                        combined_rates[selector] = math.fsum((
                            float(combined_rates.get(selector, 0.0)), float(rate),
                        ))
                    sensory_channels["poisson_hz"] = combined_rates
                if intervention_currents:
                    combined_currents = dict(sensory_channels.get("current_mV_per_ms", {}))
                    for selector, current in intervention_currents.items():
                        combined_currents[selector] = math.fsum((
                            float(combined_currents.get(selector, 0.0)), float(current),
                        ))
                    sensory_channels["current_mV_per_ms"] = combined_currents
                neural_intervention_record = {
                    "classification": self.neural_intervention_schedule.classification,
                    "schedule_name": self.neural_intervention_schedule.name,
                    "time_s": float(t_s),
                    "active_poisson_hz": dict(intervention_rates),
                    "active_current_mV_per_ms": dict(intervention_currents),
                    "resolved_selector_counts": dict(self.neural_intervention_selector_counts),
                    "selector_overlap_counts": dict(self.neural_intervention_selector_overlaps),
                    "counts_as_autonomous_foraging": False,
                    "body_or_motor_command_injected": False,
                    "schedule_ref": "manifest.neural_intervention",
                }
            self._validate_mechanosensory_ablation_channels(sensory_channels)
            sensory = SensoryFrame(
                timestamp=ts,
                visual=visual,
                body=body_contract,
                channels=sensory_channels,
                source=SourceLabel.HANDBUILT,
            )
            sensory_foreground_s = time.perf_counter() - stage_started
            if sensory_prepare_job is not None:
                stage_wall_s["sensory"] += (
                    sensory_prepare_compute_s
                    + max(0.0, sensory_foreground_s - sensory_prepare_wait_s)
                )
                stage_wall_s["sensory_prepare_wait"] += sensory_prepare_wait_s
            else:
                stage_wall_s["sensory"] += sensory_foreground_s
            stage_started = time.perf_counter()
            neural_output, neural_steps, neural_api = self._free_neural_frame(
                neural, sensory, ts, t_s,
            )
            stage_wall_s["neural"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            # This is the strict causal control boundary: MaleCNS output is
            # decoded once, and the exact NeuromuscularOutput object is handed
            # to FlightDynamics.  No autopilot, safety overlay, state drive,
            # stabilizer, goal, or post-step value is consulted.
            nm_out = self.neuromuscular.step(
                neural_output,
                dt_ms=FLYVIS_DT_S * 1000.0,
                timestamp=ts,
            )
            if grooming_bridge is not None and grooming_body is not None:
                grooming_action = grooming_bridge.apply_readout(
                    grooming_body, dt, nm_out,
                    locked_legs=(frozenset({"fore_left"})
                                 if self.options.grooming_foreleg_lock else frozenset()),
                )
                grooming_loop_record = grooming_bridge.audit(
                    grooming_sample, grooming_action,
                ).to_dict()
                grooming_loop_record["classification"] = "engineered_dust_neural_grooming_loop"
                grooming_loop_record["foreleg_locked"] = bool(self.options.grooming_foreleg_lock)
                grooming_loop_record["sensor_off"] = bool(self.options.grooming_sensor_off)
            jump_input, jump_decision = self._free_jump_input(nm_out)
            stage_wall_s["neuromuscular"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            set_wind = getattr(dynamics, "set_wind", None)
            if callable(set_wind):
                set_wind(tuple(float(value) for value in wind))
            else:
                dynamics.wind_mps = tuple(float(value) for value in wind)
            dynamics_out = self._free_dynamics_step(
                dynamics, nm_out, jump_input, dt_s=dt,
            )
            jump_applied = _json_safe({
                "jump_force_body_N": getattr(dynamics_out, "jump_force_body_N", (0.0, 0.0, 0.0)),
                "jump_impulse_body_Ns": getattr(dynamics_out, "jump_impulse_body_Ns", (0.0, 0.0, 0.0)),
                "jump_posture_torque_body_Nm": getattr(
                    dynamics_out, "jump_posture_torque_body_Nm", (0.0, 0.0, 0.0)
                ),
                "jump_actuator_active": bool(getattr(dynamics_out, "jump_actuator_active", False)),
                "jump_contact_gated": bool(getattr(dynamics_out, "jump_contact_gated", False)),
                "jump_event_latched": bool(getattr(dynamics_out, "jump_event_latched", False)),
                "source": getattr(dynamics_out, "source", "connectome"),
            })
            ground_walking_applied = None
            if self.ground_walking_settings is not None and self.ground_walking_settings.get("enabled"):
                ground_walking_applied = {
                    "enabled": bool(getattr(dynamics_out, "ground_walking_enabled", False)),
                    "ground_only": bool(self.ground_walking_settings.get("ground_only", False)),
                    "wing_and_jump_actuation_suppressed": bool(
                        self.ground_walking_settings.get("ground_only", False)
                    ),
                    "leg_power": float(getattr(dynamics_out, "ground_walking_drive", 0.0)),
                    "leg_steering": float(getattr(dynamics_out, "ground_walking_steering", 0.0)),
                    "grounded_at_start": bool(getattr(
                        dynamics_out, "ground_walking_grounded_at_start", False,
                    )),
                    "contact_available": bool(getattr(
                        dynamics_out, "ground_walking_contact_available", False,
                    )),
                    "active": bool(getattr(dynamics_out, "ground_walking_active", False)),
                    "force_body_N": list(getattr(
                        dynamics_out, "ground_walking_force_body_N", (0.0, 0.0, 0.0),
                    )),
                    "traction_force_body_N": list(getattr(
                        dynamics_out, "ground_walking_traction_force_body_N", (0.0, 0.0, 0.0),
                    )),
                    "braking_force_world_N": list(getattr(
                        dynamics_out, "ground_walking_braking_force_world_N", (0.0, 0.0, 0.0),
                    )),
                    "lateral_braking_force_world_N": list(getattr(
                        dynamics_out, "ground_walking_lateral_braking_force_world_N", (0.0, 0.0, 0.0),
                    )),
                    "speed_limit_force_world_N": list(getattr(
                        dynamics_out, "ground_walking_speed_limit_force_world_N", (0.0, 0.0, 0.0),
                    )),
                    "yaw_torque_Nm": float(getattr(
                        dynamics_out, "ground_walking_yaw_torque_Nm", 0.0,
                    )),
                    "source": str(getattr(dynamics_out, "ground_walking_source", "connectome")),
                    "model": str(getattr(dynamics_out, "ground_walking_model", "reduced_ground_traction")),
                    "model_scope": str(getattr(dynamics_out, "ground_walking_model_scope", "reduced_body")),
                    "foot_joint_pose_available": False,
                }
            contact_after = self._clamp_ground(dynamics)
            contact_surface_id = getattr(dynamics_out, "contact_surface_id", None)
            target_contact_after = bool(getattr(dynamics_out, "target_contact", False))
            next_state = dynamics.state
            next_t = t_s + dt
            next_pose = _pose_from_state(next_state)
            max_speed = max(max_speed, float(next_pose.speed_mps))
            displacement = math.sqrt(sum(
                (float(next_state.position_m[index]) - initial_position[index]) ** 2
                for index in range(3)
            ))
            max_displacement = max(max_displacement, displacement)
            altitude = float(next_state.position_m[2] - self.world.config.ground_z_m)
            max_altitude = max(max_altitude, altitude)
            min_altitude = min(min_altitude, altitude)
            next_ts = _timestamp(next_t)
            body_after = _make_contract_body(next_state, next_ts)
            stage_wall_s["dynamics"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            mapping_hash = self.flyvis_mapping.get("sha256")
            neural_record = (
                _compact_neural_record(neural_output, minimal=True)
                if self.minimal_telemetry else
                _compact_neural_record(
                    neural_output,
                    drive=flyvis_out.t4t5_drive,
                    target_indices=(
                        calibration_mapping.get("target_indices")
                        if isinstance(calibration_mapping, Mapping)
                        and "target_indices" in calibration_mapping
                        else _flyvis_mapping_metadata(self.flyvis).get("target_indices")
                    ),
                    mapping_hash=mapping_hash,
                )
                if self.compact_telemetry and not self.options.full_neural_telemetry
                else neural_output.to_dict()
            )
            render_record: Optional[dict[str, Any]] = None
            if not self.minimal_telemetry:
                render_record = {
                    "timestamp_s": float(t_s),
                    "frame_index": int(step_index),
                    "resolution": [int(self.options.width), int(self.options.height)],
                    "eye_separation_m": (
                        float(stereo.eye_separation_m) if stereo is not None else None
                    ),
                    "left": None if stereo is None else stereo.left,
                    "right": None if stereo is None else stereo.right,
                    "pose": pose.to_dict(),
                    "rendered": bool(stereo is not None),
                    "purpose": "telemetry_video_only" if stereo is not None else "skipped_compact_strict",
                }
            nm_out_record = (
                _compact_neuromuscular_output(nm_out, minimal=True)
                if self.minimal_telemetry else
                _compact_dynamic_record(nm_out, kind="nm_out")
            )
            if self.minimal_telemetry:
                motor_record: Mapping[str, Any] = {
                    "timestamp": ts.to_dict(),
                    "yaw_rate_rad_s": float(getattr(nm_out, "wing_steering", 0.0)),
                    "pitch_rate_rad_s": float(getattr(nm_out, "leg_steering", 0.0)),
                    "roll_rate_rad_s": 0.0,
                    "thrust_mps2": (
                        float(getattr(nm_out, "wing_power", 0.0))
                        * CONTROL_SOURCE_SCALE["thrust_mps2"]
                    ),
                    "source": "connectome",
                    "saturated": bool(getattr(nm_out, "saturated", False)),
                    "metadata": {
                        "source_raw": _json_safe(getattr(nm_out, "source", "connectome")),
                    },
                    "provenance_ref": "manifest.provenance",
                }
            else:
                contract_motor = ContractMotorCommand(
                    timestamp=ts,
                    yaw_rate_rad_s=float(getattr(nm_out, "wing_steering", 0.0)),
                    pitch_rate_rad_s=float(getattr(nm_out, "leg_steering", 0.0)),
                    roll_rate_rad_s=0.0,
                    thrust_mps2=(
                        float(getattr(nm_out, "wing_power", 0.0))
                        * CONTROL_SOURCE_SCALE["thrust_mps2"]
                    ),
                    source=SourceLabel.CONNECTOME,
                    saturated=bool(getattr(nm_out, "saturated", False)),
                    metadata={
                        "control_path": list(FREE_BEHAVIOR_ALLOWED_CONTROL_PATH),
                        "direct_neuromuscular_to_dynamics": True,
                        "normalized": _json_safe(getattr(nm_out, "channels", {})),
                        "source_raw": _json_safe(getattr(nm_out, "source", "connectome")),
                        "nm_out": nm_out_record,
                    },
                )
                motor_record = contract_motor.to_dict()
            frame_provenance = {} if self.minimal_telemetry else {
                "policy": FREE_BEHAVIOR_POLICY_NAME,
                "control_authority": "MaleCNS",
                "control_path": list(FREE_BEHAVIOR_ALLOWED_CONTROL_PATH),
                "direct_neuromuscular_to_dynamics": True,
                "autopilot_overlay": "prohibited",
                "flight_state_drive": "prohibited",
                "flight_body_stabilizer": "prohibited",
                "mission_goal_phase": "prohibited",
                "stimulus_label_control": False,
                "stimulus_direction_control": False,
                "future_state_control": False,
                "stimulus_interaction": "visual_only_non_interacting",
                "dynamic_channels": _json_safe(escape_packet.get("provenance", {})),
                "jump_actuator_source": "NeuromuscularAdapter.TTMn",
                "jump_physics": "combined_FlightDynamics_single_integrator",
            }
            diagnostic_counters = {
                    "physics_steps": 1,
                    "flyvis_frames": 1,
                    "neural_steps": neural_steps,
                    "collision_count": len(hard_hits),
            }
            if self.minimal_telemetry:
                diagnostics_record: Mapping[str, Any] = {
                    "timestamp": ts.to_dict(),
                    "step_index": int(step_index),
                    "neural_steps": int(neural_steps),
                    "counters": diagnostic_counters,
                    "warnings": [],
                    "source": "connectome",
                }
            else:
                diagnostics_record = StepDiagnostics(
                    timestamp=ts,
                    step_index=step_index,
                    neural_steps=neural_steps,
                    counters=diagnostic_counters,
                    warnings=(),
                    source=SourceLabel.CONNECTOME,
                ).to_dict()
            if self.minimal_telemetry:
                visual_row_record = {}
                sensory_row_record = {}
            elif self.compact_telemetry:
                # Keep dense exact columns in the live contract only; the
                # compact row builder below encodes them once as lossless
                # bytes.  Calling ``VisualObservation.to_dict`` here would
                # eagerly materialise a second pair of Python lists.
                visual_row_record: Mapping[str, Any] = {
                    "timestamp": ts.to_dict(),
                    "pixel_shape": [2, int(np.asarray(flyvis_out.left_columns).size)],
                    "optic_flow_rad_s": list(visual.optic_flow_rad_s),
                    "sensor_id": visual.sensor_id,
                    "source": visual.source.value if hasattr(visual.source, "value") else visual.source,
                }
                compact_sensor_channels = {
                    str(key): value for key, value in sensory.channels.items()
                    if str(key) not in {
                        "flyvis_t4t5", "t4t5_drive", "flyvis_target_indices",
                        "flyvis_target_values",
                    }
                }
                sensory_row_record: Mapping[str, Any] = {
                    "timestamp": ts.to_dict(),
                    "channels": compact_sensor_channels,
                    "source": sensory.source.value if hasattr(sensory.source, "value") else sensory.source,
                }
            else:
                visual_row_record = visual.to_dict()
                sensory_row_record = sensory.to_dict()
            row = {
                "schema_version": (
                    MINIMAL_TELEMETRY_SCHEMA_VERSION
                    if self.minimal_telemetry else
                    FULL_DEBUG_TELEMETRY_SCHEMA_VERSION
                    if self.full_debug else COMPACT_TELEMETRY_SCHEMA_VERSION
                ),
                "execution_policy": FREE_BEHAVIOR_POLICY_NAME,
                "scenario": "free-behavior",
                "frame_index": int(step_index),
                "frame_time_s": float(t_s),
                "time_s": next_t,
                "dt_s": float(dt),
                "interval_start_s": float(t_s),
                "interval_end_s": float(next_t),
                "timestamp": ts.to_dict(),
                "position_m": list(next_state.position_m),
                "velocity_mps": list(next_state.velocity_mps),
                "orientation_quat": list(next_state.orientation_quat),
                "orientation_rad": list(_rpy_from_quat(next_state.orientation_quat)),
                "angular_velocity_rad_s": list(next_state.angular_velocity_rad_s),
                "contact": bool(contact_after),
                "grounded": bool(contact_after),
                "target_contact": target_contact_after,
                "contact_surface_id": contact_surface_id,
                "ground_distance_m": max(
                    0.0,
                    next_state.position_m[2]
                    - self.world.config.ground_z_m
                    - self.world.config.body_radius_m,
                ),
                "flight_mode": "free_behavior",
                "controller_mode": "MaleCNS_direct",
                "nm_out": nm_out_record,
                "jump_decision_telemetry": jump_decision,
                "jump_input": _json_safe(jump_input),
                "jump_output": jump_applied,
                **({"ground_walking": ground_walking_applied}
                   if ground_walking_applied is not None else {}),
                "left_wing": float(getattr(nm_out, "left_wing_power", 0.0)),
                "right_wing": float(getattr(nm_out, "right_wing_power", 0.0)),
                "wing_drive_left": float(getattr(nm_out, "left_wing_power", 0.0)),
                "wing_drive_right": float(getattr(nm_out, "right_wing_power", 0.0)),
                "yaw_rate": float(next_state.angular_velocity_rad_s[2]),
                "pitch": float(_rpy_from_quat(next_state.orientation_quat)[1]),
                "roll": float(_rpy_from_quat(next_state.orientation_quat)[0]),
                "wind_mps": list(wind),
                "visual_drive_summary": visual_drive_summary,
                "visual_drive_sha256": visual_drive_sha256,
                "visual_directional_readout": "record_only",
                "stimulus_state": _json_safe(stimulus_state),
                "stimulus_condition": self.escape_stimulus_condition,
                "stimulus_name": self.escape_stimulus_name,
                "stimulus_contact": False,
                "stimulus_interaction": "visual_only_non_interacting",
                "escape_channels": _json_safe(escape_packet),
                "collision_count": len(hard_hits),
                "body_before": body_contract.to_dict(),
                "body": body_after.to_dict(),
                "world": None if self.minimal_telemetry else {
                    "timestamp": next_ts.to_dict(),
                    "world_id": "virtual-world",
                    "geometry": geometry,
                    "render": render_record,
                    "body": body_after.to_dict(),
                },
                "render": render_record,
                "visual": visual_row_record,
                "sensory": sensory_row_record,
                **({"food_cue": food_cue_record} if food_cue_record is not None else {}),
                **({"neural_intervention": neural_intervention_record}
                   if neural_intervention_record is not None else {}),
                **({"grooming_loop": grooming_loop_record}
                   if grooming_loop_record is not None else {}),
                "mechanosensory": (
                    _compact_mechanosensory_record(mech, minimal=True)
                    if self.minimal_telemetry else _json_safe(mech.to_dict())
                ),
                **({"mechanosensory_ablation": mechanosensory_ablation_record}
                   if mechanosensory_ablation_record is not None else {}),
                "neural": neural_record,
                "neural_window": {
                    "start_s": float(t_s),
                    "end_s": float(next_t),
                    "dt_s": float(CNS_DT_S),
                    "tick_dt_s": float(CNS_DT_S),
                    "window_dt_s": float(dt),
                    "steps": int(neural_steps),
                    "api": neural_api,
                    "timing": _json_safe(neural_output.named_outputs.get("triton_step_timing")),
                },
                "motor": motor_record,
                "diagnostics": diagnostics_record,
                "frame_provenance": frame_provenance,
                "provenance": None if self.minimal_telemetry else {
                    "evidence_class": "digital_simulation",
                    "world": "handbuilt",
                    "vision": "handbuilt" if flyvis_out.is_stub else "fitted",
                    "neural": "connectome",
                    "neuromuscular": "connectome",
                    "body": "handbuilt",
                    "sensory_feedback": "handbuilt",
                    "autonomy_policy": FREE_BEHAVIOR_POLICY_NAME,
                    "prohibited_paths": list(FREE_BEHAVIOR_PROHIBITED_PATHS),
                    "control_path": list(FREE_BEHAVIOR_ALLOWED_CONTROL_PATH),
                    "stimulus_labels": "record_only",
                    "stimulus_direction": "record_only",
                    "future_state": "record_only",
                    "stimulus_interaction": "visual_only_non_interacting",
                    "autopilot": "prohibited",
                    "flight_state_drive": "prohibited",
                    "flight_body_stabilizer": "prohibited",
                    "mission_phases_goals": "prohibited",
                    "direct_malecns_to_dynamics": True,
                    "frame": frame_provenance,
                },
            }
            if self.compact_telemetry:
                rows.append(_free_behavior_row(
                    row,
                    mode=self.telemetry_mode,
                    flyvis_output=flyvis_out,
                    visual_observation=visual,
                    nm_output=nm_out,
                    mechanosensory=mech,
                    escape_packet=escape_packet,
                    neural_output=neural_output,
                    exact_sampling=flyvis_sampling,
                    drive_sha256=visual_drive_sha256,
                    drive_summary=visual_drive_summary,
                    preserve_neural_vectors=bool(self.options.full_neural_telemetry),
                    preserve_rectangular_render=bool(self.escape_channels.roles),
                ))
            else:
                rows.append(_json_safe(row))
            stage_wall_s["telemetry"] += time.perf_counter() - stage_started
            stage_wall_s["frame_total"] += time.perf_counter() - frame_started
            t_s = next_t
            step_index += 1
        final_pose = _pose_from_state(dynamics.state)
        initial_altitude = initial_position[2] - self.world.config.ground_z_m
        metrics = {
            "sample_count": len(rows),
            "frame_count": len(rows),
            "neural_steps": len(rows) * CNS_STEPS_PER_FLYVIS,
            "duration_s": t_s,
            "dt_s": FLYVIS_DT_S,
            "collision_count": len(collisions),
            "contact_count": contact_count,
            "max_altitude_m": max_altitude if rows else initial_altitude,
            "min_altitude_m": min_altitude if rows else initial_altitude,
            "max_displacement_m": max_displacement,
            "final_altitude_m": float(final_pose.position_m[2] - self.world.config.ground_z_m),
            "final_speed_mps": float(final_pose.speed_mps),
            "max_speed_mps": max_speed,
            "max_flow_magnitude": max_flow,
            "stimulus_contact_deadline_s": self.escape_experiment_spec.get(
                "surface_contact_deadline_s", None,
            ),
            "stimulus_contact_observed": False,
            "stimulus_interaction": "visual_only_non_interacting",
            "stimulus_condition": self.escape_stimulus_condition,
            "stimulus_name": self.escape_stimulus_name,
            "reported_visual_populations": list(self.escape_report_populations),
            "visual_scene_baseline": self.visual_scene_baseline_report,
            "frame_stage_pipeline": self._frame_stage_pipeline.manifest(),
            "stage_wall_s": stage_wall_s,
            "stage_mean_ms": {
                key: (value * 1000.0 / len(rows) if rows else 0.0)
                for key, value in stage_wall_s.items()
            },
            **({
                "grooming_initial_dust": self.options.grooming_dust_count,
                "grooming_initial_particles": grooming_initial_particles,
                "grooming_final_dust": grooming_body.remaining()["antenna"],
                "grooming_cleaned_dust": (
                    self.options.grooming_dust_count - grooming_body.remaining()["antenna"]
                ),
            } if grooming_body is not None else {}),
        }
        return {
            "name": "free-behavior",
            "config_name": "free-behavior",
            "duration_s": t_s,
            "metrics": _json_safe(metrics),
            "acceptance": {
                "passed": True,
                "checks": {},
                "violations": [],
                "policy": "observation_only_no_mission_goals",
            },
            "rows": rows,
            "collisions": collisions,
            "final_state": _json_safe(dynamics.state.to_dict()),
        }

    def _one_scenario(self, scenario_name: str, *, config_name: Optional[str] = None) -> dict[str, Any]:
        config_name = _canonical_scenario_name(scenario_name) if config_name is None else config_name
        scenario = self.world.scenarios[config_name]
        duration = scenario.duration_s if self.options.duration_override_s is None else self.options.duration_override_s
        if self.options.frame_count is not None:
            duration = self.options.frame_count * FLYVIS_DT_S
        if self.options.max_steps is not None:
            duration = min(duration, self.options.max_steps * FLYVIS_DT_S)
        duration = max(FLYVIS_DT_S, float(duration))
        frame_count = int(round(duration / FLYVIS_DT_S))
        if abs(duration - frame_count * FLYVIS_DT_S) > 1.0e-8:
            raise LauncherError(
                f"scenario {scenario_name!r} duration {duration:g}s is not aligned to "
                f"the FlyVis frame period {FLYVIS_DT_S:g}s"
            )
        duration = frame_count * FLYVIS_DT_S
        dynamics = self._new_body(scenario)
        neural = self._new_neural(scenario_name)
        self.flyvis.reset()
        if self.visual_calibration is not None:
            self.visual_calibration.reset()
        if self.flight_state_drive is not None:
            self.flight_state_drive.reset()
        if self.flight_body_stabilizer is not None:
            self.flight_body_stabilizer.reset()
        self.sensory_feedback.reset()
        self.neuromuscular.reset(clear_baseline=False)
        self.autopilot.reset()
        wind_model = self._wind_model(scenario)
        rows: list[dict[str, Any]] = []
        collisions: list[dict[str, Any]] = []
        contact_count = 0
        max_altitude = -math.inf
        min_altitude = math.inf
        max_drift = 0.0
        max_alt_error = 0.0
        max_flow = 0.0
        max_speed = 0.0
        target_visible_count = 0
        target_on_target_count = 0
        target_distances: list[float] = []
        step_index = 0
        t_s = 0.0
        while t_s < duration - 1.0e-12:
            dt = min(FLYVIS_DT_S, duration - t_s)
            # FlyVis enforces exactly 5 ms frames.  Duration overrides used by
            # smoke tests are rounded to this grid instead of creating a
            # hidden resampling path.
            if abs(dt - FLYVIS_DT_S) > 1.0e-9:
                raise LauncherError("internal frame scheduler produced a partial FlyVis frame")
            ts = _timestamp(t_s)
            state_before = dynamics.state.copy()
            pose = _pose_from_state(state_before)
            # FlightDynamics owns the unilateral contact constraint.  Use its
            # configured centre-floor for the causal pre-step contact flag so
            # the runner cannot silently reintroduce a second datum.
            contact_before = self._body_contact(state_before, dynamics)
            body_contract = _make_contract_body(state_before, ts)
            world_state = WorldState(
                timestamp=ts,
                body=body_contract,
                world_id="virtual-world",
                geometry=self.world.manifest(scenario),
                source=SourceLabel.HANDBUILT,
            )
            wind = self.world.wind_at(pose.position_m, t_s, wind_model)
            hard_hits, soft_hits = self.world.collision_events(pose, t_s)
            collisions.extend(item.to_dict() for item in hard_hits)
            contact_count += int(contact_before) + len(soft_hits)

            # Mission telemetry keeps the historical rectangular stereo
            # render, but a fitted FlyVis controller must consume the exact
            # runtime input_index-derived eye directions.  The rectangular
            # frame is therefore an optic-flow/video observation only; it is
            # never rendered and then silently resampled into FlyVis input.
            stereo = self.world.render_stereo(
                pose, (self.options.width, self.options.height), timestamp_s=t_s,
            )
            visual_world = self.world.visual_observation(stereo, t_s, dt_s=FLYVIS_DT_S)
            if self.options.vision == "flyvis":
                left_columns, right_columns, flyvis_sampling = self._free_exact_flyvis_columns(
                    pose, t_s,
                )
                flyvis_out = self._process_exact_flyvis_columns(
                    left_columns, right_columns, t_s,
                )
            else:
                flyvis_sampling = {
                    "mode": "rectangular_stub",
                    "rectangular_stereo": "controller_input",
                }
                flyvis_out = self.flyvis.process_frame(
                    visual_world, right_frame=stereo.right, timestamp_s=t_s,
                )
            calibration_metadata: dict[str, Any] = {}
            calibration_mapping: Any = None
            calibration_channels: dict[str, Any] = {}
            if self.visual_calibration is not None:
                mapping = _flyvis_mapping_metadata(self.flyvis)
                state_context = self._visual_state_context(scenario, config_name, pose, t_s)
                if self.options.visual_tonic_state_inputs:
                    resolver = None
                    n_neurons = None
                    engine = getattr(self.neural_kind, "engine", None)
                    if engine is not None:
                        resolver = getattr(engine, "resolve_population", None)
                        n_neurons = getattr(engine, "n", None)
                    calibration_channels, calibration_metadata = self.visual_calibration.apply_to_channels(
                        {}, flyvis_out.t4t5_drive,
                        scenario=config_name,
                        timestamp_s=t_s,
                        frame_index=step_index,
                        visual_output=flyvis_out,
                        target_indices=mapping.get("target_indices"),
                        target_ids=mapping.get("target_ids"),
                        units=mapping.get("units"),
                        state_context=state_context,
                        population_resolver=resolver,
                        n_neurons=n_neurons,
                    )
                    calibration_mapping = calibration_channels.get(
                        "flyvis_t4t5", calibration_metadata.get("engine_mapping")
                    )
                    calibrated_drive = (
                        calibration_mapping.get("values", calibration_mapping.get("drive"))
                        if isinstance(calibration_mapping, Mapping) else flyvis_out.t4t5_drive
                    )
                    calibration_metadata = {
                        **calibration_metadata,
                        "tonic_state_inputs": [
                            key for key in calibration_channels
                            if key in {"current_mV_per_ms", "poisson_hz"}
                        ],
                    }
                else:
                    calibrated_drive, calibration_metadata = self.visual_calibration.apply(
                        flyvis_out.t4t5_drive,
                        scenario=config_name,
                        timestamp_s=t_s,
                        frame_index=step_index,
                        visual_output=flyvis_out,
                        target_indices=mapping.get("target_indices"),
                        target_ids=mapping.get("target_ids"),
                        units=mapping.get("units"),
                        state_context=state_context,
                    )
                    calibration_mapping = calibration_metadata.get("engine_mapping")
                flyvis_out = replace(flyvis_out, t4t5_drive=calibrated_drive)
            max_flow = max(max_flow, max((abs(float(x)) for x in flyvis_out.t4t5_drive), default=0.0))
            # The fitted T4/T5 vector is ordered by the explicit MaleCNS
            # target mapping.  A population mean is not a directional
            # control signal: opposite preferred directions can cancel even
            # when the visual response is strong.  Decode the real vector
            # with the mapping-aware readout and fail closed if the runtime
            # mapping, target order, or units drift.  The deterministic stub
            # intentionally retains its historical mean path for CPU smoke
            # tests and does not load the optional readout assets.
            control_readout_result: Any = None
            visual_pitch_cue = 0.0
            visual_readout_valid = True
            if self.visual_control_readout is not None:
                runtime_mapping = getattr(self.flyvis_runtime_adapter, "mapping", None)
                mapping = _flyvis_mapping_metadata(self.flyvis)
                required_mapping = ("target_indices", "target_ids", "units")
                missing_mapping = [
                    key for key in required_mapping
                    if mapping.get(key) is None
                ]
                if runtime_mapping is None or missing_mapping:
                    raise LauncherError(
                        "fitted FlyVis control readout requires complete runtime mapping "
                        f"(missing {missing_mapping or ['mapping']})"
                    )
                try:
                    target_count = int(np.asarray(mapping["target_indices"]).size)
                except (TypeError, ValueError) as exc:
                    raise LauncherError(
                        "fitted FlyVis control readout received an invalid target index mapping"
                    ) from exc
                if target_count != int(getattr(self.visual_control_readout, "target_count", -1)):
                    raise LauncherError(
                        "fitted FlyVis control readout target count does not match the runtime mapping"
                    )
                try:
                    control_readout_result = self.visual_control_readout.readout(
                        flyvis_out.t4t5_drive,
                        target_mapping=runtime_mapping,
                        target_indices=mapping["target_indices"],
                        target_ids=mapping["target_ids"],
                        units=mapping["units"],
                        frame_index=step_index,
                        timestamp_s=t_s,
                    )
                except Exception as exc:
                    # Do not turn a mapping mismatch into a scalar fallback.
                    # A fitted run with ambiguous target order is unusable.
                    raise LauncherError(
                        f"fitted FlyVis control readout failed at frame {step_index}: {exc}"
                    ) from exc
                visual_direction = _clamp(control_readout_result.yaw, -1.0, 1.0)
                visual_pitch_cue = _clamp(control_readout_result.pitch, -1.0, 1.0)
                visual_confidence = _clamp(control_readout_result.confidence)
                visual_readout_valid = bool(getattr(control_readout_result, "valid", False))
            else:
                visual_direction = _clamp(
                    sum(float(x) for x in flyvis_out.t4t5_drive) / max(1, len(flyvis_out.t4t5_drive)),
                    -1.0,
                    1.0,
                )
                visual_confidence = min(1.0, abs(visual_direction) + 0.1)
            control_readout_summary = None
            if control_readout_result is not None:
                readout_provenance = getattr(control_readout_result, "provenance", {})
                control_readout_summary = {
                    "yaw": float(control_readout_result.yaw),
                    "pitch": float(control_readout_result.pitch),
                    "confidence": float(control_readout_result.confidence),
                    "directional_coherence": float(control_readout_result.directional_coherence),
                    "response_energy": float(control_readout_result.response_energy),
                    "valid": visual_readout_valid,
                    "target_count": int(readout_provenance.get("target_count", target_count)),
                    "mapping_sha256": readout_provenance.get("mapping_sha256"),
                }
            visual = VisualObservation(
                timestamp=ts,
                pixels=[flyvis_out.left_columns.tolist(), flyvis_out.right_columns.tolist()],
                optic_flow_rad_s=tuple(visual_world.optic_flow_rad_s),
                sensor_id="virtual-world-stereo-flyvis",
                source=SourceLabel.HANDBUILT if flyvis_out.is_stub else SourceLabel.FITTED,
                metadata={**flyvis_out.metadata,
                          "world_sensor_backend": getattr(self.world, "backend_name", "virtual_world"),
                          "flyvis_sampling": flyvis_sampling,
                          "visual_calibration": _json_safe({
                              key: value for key, value in calibration_metadata.items()
                              if key not in {"engine_mapping", "target_indices", "target_ids"}
                          }) if calibration_metadata else None,
                          "control_readout": control_readout_summary},
            )
            mech = self.sensory_feedback.step(
                body_contract,
                world_wind=wind,
                ground_contact=contact_before,
                joint_state=self._feedback_joint_state(dynamics),
            )
            male_cns_population_drive, mechanosensory_ablation_record = (
                self._masked_mechanosensory_population_drive(mech)
            )
            state_drive_output = None
            state_drive_channels: dict[str, Any] = {}
            state_drive_context: dict[str, Any] = {}
            if self.flight_state_drive is not None:
                state_drive_context = self._flight_state_context(
                    scenario, config_name, pose, t_s, contact_before,
                )
                try:
                    state_drive_output = self.flight_state_drive.step(state_drive_context)
                except Exception as exc:
                    raise LauncherError(
                        f"flight state drive failed at frame {step_index}: {exc}"
                    ) from exc
                try:
                    state_drive_channels = state_drive_output.as_engine_channels()
                except Exception as exc:
                    raise LauncherError(
                        f"flight state drive returned invalid engine channels at frame {step_index}: {exc}"
                    ) from exc
            goal, _ = self._goal_at(scenario, t_s)
            max_drift = max(max_drift, math.hypot(
                pose.position_m[0] - goal[0], pose.position_m[1] - goal[1],
            ))
            max_altitude = max(max_altitude, pose.position_m[2] - self.world.config.ground_z_m)
            min_altitude = min(min_altitude, pose.position_m[2] - self.world.config.ground_z_m)
            max_alt_error = max(max_alt_error, abs(pose.position_m[2] - goal[2]))

            sensory_channels = {
                "optic_flow_x_rad_s": visual.optic_flow_rad_s[0],
                "optic_flow_y_rad_s": visual.optic_flow_rad_s[1],
                "flyvis_drive_mean": visual_direction,
                "flyvis_is_stub": flyvis_out.is_stub,
                "flyvis_control_yaw": visual_direction,
                "flyvis_control_pitch": visual_pitch_cue,
                "flyvis_control_confidence": visual_confidence,
                "flyvis_control_valid": visual_readout_valid,
                # Preserve the signed vector and the fitted adapter's exact
                # target ordering.  MaleCNSEngine consumes ``flyvis_t4t5``
                # when this mapping is present and rejects an incomplete
                # vector rather than guessing a dense-neuron layout.
                **_flyvis_drive_contract(self.flyvis, flyvis_out.t4t5_drive,
                                         mapping_override=calibration_mapping),
                **{f"mech:{key}": value for key, value in mech.channels.items()},
                # Preserve physical feedback observables alongside the
                # normalized channels consumed by the neural adapter.  The
                # contract ``SensoryFrame`` remains compact, while the
                # ``mech_raw:*`` fields make the actual WPG/power/frequency
                # provenance inspectable per row.
                **{f"mech_raw:{key}": value for key, value in mech.raw_channels.items()},
                **{f"population:{key}": value for key, value in male_cns_population_drive.items()},
                "population_drive_units": str(getattr(mech, "population_drive_units", "normalized")),
                "population_drive_range": list(getattr(mech, "population_drive_range", (-1.0, 1.0))),
                **{key: value for key, value in calibration_channels.items()
                   if key not in {"flyvis_t4t5", "t4t5_drive", "visual_drive_calibration"}},
            }
            # Unit-keyed state-drive channels are merged additively.  This is
            # important when a calibration also supplies an external
            # population packet: neither source silently replaces the other.
            for unit, values in state_drive_channels.items():
                if not isinstance(values, Mapping):
                    raise LauncherError(
                        f"flight state drive channel {unit!r} must be a population mapping"
                    )
                existing = sensory_channels.get(unit)
                if existing is None:
                    existing = {}
                    sensory_channels[unit] = existing
                if not isinstance(existing, Mapping):
                    raise LauncherError(
                        f"sensory channel {unit!r} is not mergeable with flight state drive"
                    )
                merged = dict(existing)
                for population, value in values.items():
                    try:
                        merged[str(population)] = float(merged.get(str(population), 0.0)) + float(value)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise LauncherError(
                            f"flight state drive channel {unit!r} has a non-numeric value"
                        ) from exc
                sensory_channels[unit] = merged
            if calibration_channels.get("visual_drive_calibration") is not None:
                sensory_channels["visual_drive_calibration"] = calibration_channels["visual_drive_calibration"]
            self._validate_mechanosensory_ablation_channels(sensory_channels)
            sensory = SensoryFrame(
                timestamp=ts,
                visual=visual,
                body=body_contract,
                channels=sensory_channels,
                source=SourceLabel.HANDBUILT,
            )

            if self.options.neural == "stub":
                power, steering = _desired_stub_control(config_name, t_s, pose.position_m[2], visual_direction)
                neural.set_control(power, steering)
                neural_output = neural.step(sensory, ts)
                neural_steps = 1
                neural_api = "stub_step"
            else:
                # A FlyVis frame is 5 ms while MaleCNS integrates at 0.5 ms.
                # Prefer the engine's bounded held-input window.  Triton can
                # keep all ten ticks on device and return one aggregate;
                # small CPU/mocked adapters without that API use the explicit
                # timestamped scalar fallback below.
                neural_api = "scalar_step_substeps"
                window_output = None
                window_method = getattr(neural, "step_window", None)
                if callable(window_method):
                    window_output = window_method(
                        sensory, ts, steps=CNS_STEPS_PER_FLYVIS,
                    )
                if window_output is not None:
                    neural_output = window_output
                    if neural_output.timestamp != ts:
                        # Window engines commonly report their final CNS tick;
                        # the outer contract is one FlyVis-frame aggregate.
                        with_timestamp = getattr(neural_output, "with_timestamp", None)
                        if callable(with_timestamp):
                            neural_output = with_timestamp(ts)
                        else:
                            neural_output = NeuralOutput(
                                ts, rates_hz=neural_output.rates_hz,
                                spikes=neural_output.spikes,
                                named_outputs=neural_output.named_outputs,
                                source=neural_output.source,
                            )
                    try:
                        neural_steps = int(neural_output.named_outputs.get(
                            "neural_window_steps", CNS_STEPS_PER_FLYVIS
                        ))
                    except (TypeError, ValueError) as exc:
                        raise LauncherError(
                            "malecns engine window output has an invalid neural_window_steps"
                        ) from exc
                    if neural_steps != CNS_STEPS_PER_FLYVIS:
                        raise LauncherError(
                            "malecns engine window output did not integrate exactly "
                            f"{CNS_STEPS_PER_FLYVIS} CNS ticks (got {neural_steps})"
                        )
                    neural_api = str(neural_output.named_outputs.get(
                        "neural_window_api", "engine_step_window"
                    ))
                else:
                    cns_outputs: list[NeuralOutput] = []
                    for substep in range(CNS_STEPS_PER_FLYVIS):
                        cns_t_s = t_s + substep * CNS_DT_S
                        cns_ts = _timestamp(cns_t_s, clock="neural", rate_hz=CNS_HZ)
                        cns_frame = _substep_sensory(sensory, cns_ts)
                        cns_outputs.append(neural.step(cns_frame, cns_ts))
                    neural_steps = len(cns_outputs)
                    neural_output = _aggregate_neural_outputs(
                        cns_outputs, ts, window_dt_s=FLYVIS_DT_S,
                    )
                neural_output.named_outputs["neural_window_api"] = neural_api
            if self.options.neural == "stub":
                adapter_signal = neural_output.named_outputs.get("population_rates", {})
            else:
                adapter_signal = neural_output
            nm_out = self.neuromuscular.step(adapter_signal, dt_ms=FLYVIS_DT_S * 1000.0, timestamp=ts)
            motor_channels = {
                "left_wing": nm_out.left_wing_power,
                "right_wing": nm_out.right_wing_power,
                "thrust": nm_out.wing_power,
                "yaw": nm_out.wing_steering,
                "pitch": 0.0,
                # ``wing_steering`` is the fitted yaw readout.  The reduced
                # body already obtains a roll moment from the left/right wing
                # power difference, so copying the same scalar into the
                # explicit roll-torque channel double-counts steering and
                # turns a yaw manoeuvre into a large lateral excursion.  Roll
                # attitude remains available to the bounded body stabilizer.
                "roll": 0.0,
                "brake": 0.0,
            }
            autopilot_body = body_contract.to_dict()
            autopilot_body["grounded"] = bool(contact_before)
            auto_command = self.autopilot.step(
                motor_channels,
                {
                    "direction": visual_direction,
                    "yaw": visual_direction,
                    "pitch": visual_pitch_cue,
                    "pitch_cue": visual_pitch_cue,
                    "confidence": visual_confidence,
                    "valid": visual_readout_valid,
                },
                autopilot_body,
                _target_info(self.world, pose, scenario, contact_before, t_s),
                dt_s=FLYVIS_DT_S,
            )
            stabilizer_output = None
            effective_motor_channels = auto_command.actuator_dict()
            if self.flight_body_stabilizer is not None:
                stabilizer_context = state_drive_context or self._flight_state_context(
                    scenario, config_name, pose, t_s, contact_before,
                )
                errors = dict(stabilizer_context.get("body_error", {}))
                goal_errors = {
                    key: errors.get(key, 0.0)
                    for key in (
                        "altitude_error_m", "forward_error_m", "lateral_error_m",
                        "heading_error_rad", "speed_error_mps",
                    )
                }
                body_errors = {
                    key: errors.get(key, 0.0)
                    for key in (
                        "pitch_error_rad", "roll_error_rad", "vertical_speed_mps",
                        "forward_velocity_mps", "lateral_velocity_mps",
                        "pitch_rate_rad_s", "roll_rate_rad_s", "yaw_rate_rad_s",
                    )
                }
                phase_packet = {
                    "label": stabilizer_context.get("phase_label", config_name),
                    "progress": stabilizer_context.get("phase", 0.0),
                }
                try:
                    stabilizer_output = self.flight_body_stabilizer.step(
                        goal_error=goal_errors,
                        body_error=body_errors,
                        phase=phase_packet,
                        fitted_motor_channels=auto_command,
                        dt_s=FLYVIS_DT_S,
                    )
                    effective_motor_channels = stabilizer_output.to_motor_channels()
                except Exception as exc:
                    raise LauncherError(
                        f"flight body stabilizer failed at frame {step_index}: {exc}"
                    ) from exc
            dyn_command = dict(effective_motor_channels)
            dyn_command["wind_mps"] = wind
            dynamics_out = dynamics.step(dyn_command, dt_s=dt)
            contact_after = self._clamp_ground(dynamics)
            target_contact_after = bool(getattr(dynamics_out, "target_contact", False))
            contact_surface_id = getattr(dynamics_out, "contact_surface_id", None)
            next_state = dynamics.state
            next_t = t_s + dt
            next_pose = _pose_from_state(next_state)
            next_ts = _timestamp(next_t)
            next_observation = self.world.target_observation(next_pose)
            target_visible_count += int(bool(next_observation.visible))
            target_on_target_count += int(bool(next_observation.on_target))
            target_distances.append(float(next_observation.distance_m))
            max_speed = max(max_speed, float(next_pose.speed_mps))
            body_after = _make_contract_body(next_state, next_ts)
            reference_goal, reference_yaw = self._goal_at(scenario, t_s)
            phase_info: Mapping[str, Any] = {}
            phase_at = getattr(self.world, "phase_at", None)
            if callable(phase_at):
                try:
                    candidate_phase = phase_at(scenario, t_s)
                    if isinstance(candidate_phase, Mapping):
                        phase_info = candidate_phase
                except (KeyError, TypeError, ValueError):
                    phase_info = {}
            stabilizer_active = bool(
                stabilizer_output is not None and getattr(stabilizer_output, "active", False)
            )
            contract_motor_source = (
                SourceLabel.HANDBUILT
                if stabilizer_active or str(auto_command.source).startswith("safety")
                else SourceLabel.FITTED
            )
            contract_motor = ContractMotorCommand(
                timestamp=ts,
                yaw_rate_rad_s=float(effective_motor_channels["yaw"]) * CONTROL_SOURCE_SCALE["yaw_rate_rad_s"],
                pitch_rate_rad_s=float(effective_motor_channels["pitch"]) * CONTROL_SOURCE_SCALE["pitch_rate_rad_s"],
                roll_rate_rad_s=float(effective_motor_channels["roll"]) * CONTROL_SOURCE_SCALE["roll_rate_rad_s"],
                thrust_mps2=float(effective_motor_channels["thrust"]) * CONTROL_SOURCE_SCALE["thrust_mps2"],
                source=contract_motor_source,
                saturated=bool(auto_command.safety_active),
                metadata={
                    "normalized": dict(effective_motor_channels),
                    "mode": auto_command.mode.value,
                    "source_raw": (
                        getattr(stabilizer_output, "source", auto_command.source)
                        if stabilizer_output is not None else auto_command.source
                    ),
                    "safety_reason": auto_command.safety_reason,
                    "stabilizer_active": stabilizer_active,
                },
            )
            nm_out_record = _compact_dynamic_record(nm_out, kind="nm_out")
            auto_command_record = _compact_dynamic_record(auto_command, kind="auto_command")
            stabilizer_record = (
                None if stabilizer_output is None else _json_safe(stabilizer_output.to_dict())
            )
            # Keep the dynamic source chain adjacent to the contract motor
            # record; the row-level copies below make the same information
            # available to consumers that do not unpack contract metadata.
            contract_motor.metadata["nm_out"] = nm_out_record
            contract_motor.metadata["auto_command"] = auto_command_record
            contract_motor.metadata["flight_body_stabilizer"] = stabilizer_record
            mapping_hash = self.flyvis_mapping.get("sha256")
            neural_record = (
                _compact_neural_record(
                    neural_output,
                    drive=flyvis_out.t4t5_drive,
                    target_indices=(
                        calibration_mapping.get("target_indices")
                        if isinstance(calibration_mapping, Mapping)
                        and "target_indices" in calibration_mapping
                        else _flyvis_mapping_metadata(self.flyvis).get("target_indices")
                    ),
                    mapping_hash=mapping_hash,
                )
                if (
                    self.compact_telemetry
                    and self.options.neural == "malecns"
                    and not self.options.full_neural_telemetry
                )
                else neural_output.to_dict()
            )
            sensory_record = sensory.to_dict()
            if (
                self.compact_telemetry
                and self.options.neural == "malecns"
                and not self.options.full_neural_telemetry
            ):
                # The dense visual target vector is needed by the live CNS
                # object above, but is represented once in the manifest for a
                # sustained trace.  Keep row-level summary values for plots.
                compact_channels = dict(sensory_record.get("channels", {}))
                for key in ("flyvis_t4t5", "t4t5_drive", "flyvis_target_indices",
                            "flyvis_target_values", "visual_drive_calibration"):
                    compact_channels.pop(key, None)
                compact_channels["flyvis_drive_summary"] = _drive_summary(flyvis_out.t4t5_drive)
                compact_channels["flyvis_target_count"] = int(self.flyvis_mapping.get("count", 0))
                compact_channels["flyvis_mapping_sha256"] = mapping_hash
                if calibration_metadata:
                    compact_channels["visual_calibration"] = {
                        "backend": calibration_metadata.get("backend"),
                        "module": calibration_metadata.get("module"),
                        "tonic_state_inputs": calibration_metadata.get("tonic_state_inputs", ()),
                    }
                sensory_record["channels"] = compact_channels
            state_drive_record = _compact_state_drive_record(
                state_drive_output, timestamp_s=t_s, frame_index=step_index,
            )
            if state_drive_record is not None:
                state_drive_record["context"] = _json_safe({
                    "timestamp_s": t_s,
                    "frame_index": step_index,
                    "scenario": state_drive_context.get("scenario"),
                    "phase": state_drive_context.get("phase"),
                    "phase_label": state_drive_context.get("phase_label"),
                    "phase_contract": state_drive_context.get("phase_contract"),
                    "phase_index": state_drive_context.get("phase_index"),
                    "body_error": state_drive_context.get("body_error", {}),
                })
            render_record = {
                "timestamp_s": float(t_s),
                "frame_index": int(step_index),
                "resolution": [int(self.options.width), int(self.options.height)],
                "eye_separation_m": float(stereo.eye_separation_m),
                "left": stereo.left,
                "right": stereo.right,
                "pose": pose.to_dict(),
            }
            diagnostics = StepDiagnostics(
                timestamp=ts,
                step_index=step_index,
                neural_steps=neural_steps,
                counters={
                    "physics_steps": 1,
                    "flyvis_frames": 1,
                    "neural_steps": neural_steps,
                    "collision_count": len(hard_hits),
                },
                warnings=("stub_neural",) if self.options.neural == "stub" else (),
                source=SourceLabel.HANDBUILT,
            )
            row = {
                "schema_version": (
                    FULL_DEBUG_TELEMETRY_SCHEMA_VERSION
                    if self.full_debug else "digital-fly-telemetry-1"
                ),
                "scenario": scenario_name,
                "config_scenario": config_name,
                "frame_index": int(step_index),
                "frame_time_s": float(t_s),
                "time_s": next_t,
                "dt_s": float(dt),
                "interval_start_s": float(t_s),
                "interval_end_s": float(next_t),
                "timestamp": ts.to_dict(),
                "goal_position_m": list(reference_goal),
                "goal_yaw_rad": float(reference_yaw),
                "reference_velocity_mps": (
                    list(self.world.goal_at(scenario, t_s).get("velocity_mps", (0.0, 0.0, 0.0)))
                    if callable(getattr(self.world, "goal_at", None)) else [0.0, 0.0, 0.0]
                ),
                "mission_phase": phase_info.get("label", phase_info.get("phase", config_name)),
                "mission_phase_index": phase_info.get("index", phase_info.get("phase_index", 0)),
                "phase_contract": {
                    "name": "normalized_local_phase",
                    "range": [0.0, 1.0],
                    "phase": _clamp(
                        float(phase_info.get("local_time_s", t_s))
                        / max(float(phase_info.get("duration_s", scenario.duration_s)), FLYVIS_DT_S)
                    ),
                    "phase_label": str(phase_info.get("label", phase_info.get("phase", config_name))),
                    "start_s": float(phase_info.get("start_s", 0.0)),
                    "end_s": float(phase_info.get("end_s", scenario.duration_s)),
                    "duration_s": float(phase_info.get("duration_s", scenario.duration_s)),
                    "local_time_s": float(phase_info.get("local_time_s", t_s)),
                },
                "position_m": list(next_state.position_m),
                "velocity_mps": list(next_state.velocity_mps),
                "orientation_quat": list(next_state.orientation_quat),
                "orientation_rad": list(_rpy_from_quat(next_state.orientation_quat)),
                "angular_velocity_rad_s": list(next_state.angular_velocity_rad_s),
                "contact": bool(contact_after),
                "grounded": bool(contact_after),
                "target_contact": target_contact_after,
                "contact_surface_id": contact_surface_id,
                "ground_distance_m": max(0.0, next_state.position_m[2] - self.world.config.ground_z_m - self.world.config.body_radius_m),
                "flight_mode": "grounded" if contact_after else (
                    "landing" if (
                        config_name == "landing"
                        or str(phase_info.get("label", config_name)).lower()
                        in {"approach", "landing"}
                    ) and next_state.velocity_mps[2] < 0
                    else "flight"
                ),
                "controller_mode": str(auto_command.mode.value),
                "nm_out": nm_out_record,
                "auto_command": auto_command_record,
                "flight_body_stabilizer": stabilizer_record,
                "left_wing": float(effective_motor_channels["left_wing"]),
                "right_wing": float(effective_motor_channels["right_wing"]),
                "wing_drive_left": float(effective_motor_channels["left_wing"]),
                "wing_drive_right": float(effective_motor_channels["right_wing"]),
                "yaw": float(next_state.orientation_rad[2]) if hasattr(next_state, "orientation_rad") else float(_rpy_from_quat(next_state.orientation_quat)[2]),
                "yaw_rate": float(next_state.angular_velocity_rad_s[2]),
                "pitch": float(_rpy_from_quat(next_state.orientation_quat)[1]),
                "roll": float(_rpy_from_quat(next_state.orientation_quat)[0]),
                "target_distance_m": float(next_observation.distance_m),
                "target_bearing_rad": float(next_observation.bearing_rad),
                "target_visible": bool(next_observation.visible),
                "target_on_target": bool(next_observation.on_target),
                "wind_mps": list(wind),
                "visual_direction": visual_direction,
                "visual_yaw_cue": visual_direction,
                "visual_pitch_cue": visual_pitch_cue,
                "visual_confidence": visual_confidence,
                "visual_readout_valid": visual_readout_valid,
                "visual_readout": control_readout_summary,
                "flight_state_drive": state_drive_record,
                "collision_count": len(hard_hits),
                "body_before": body_contract.to_dict(),
                "body": body_after.to_dict(),
                "world": {
                    "timestamp": next_ts.to_dict(),
                    "world_id": "virtual-world",
                    "render": render_record,
                    "body": body_after.to_dict(),
                },
                "render": render_record,
                "visual": visual.to_dict(),
                "sensory": sensory_record,
                "mechanosensory": _json_safe(mech.to_dict()),
                **({"mechanosensory_ablation": mechanosensory_ablation_record}
                   if mechanosensory_ablation_record is not None else {}),
                "neural": neural_record,
                "neural_window": {
                    "start_s": float(t_s),
                    "end_s": float(next_t),
                    "dt_s": float(CNS_DT_S),
                    "tick_dt_s": float(CNS_DT_S),
                    "window_dt_s": float(dt),
                    "steps": int(neural_steps),
                    "api": neural_api,
                    "timing": _json_safe(neural_output.named_outputs.get("triton_step_timing")),
                },
                "motor": contract_motor.to_dict(),
                "diagnostics": diagnostics.to_dict(),
                "provenance": {
                    "world": "handbuilt",
                    "vision": "handbuilt" if flyvis_out.is_stub else "fitted",
                    "neural": "handbuilt" if self.options.neural == "stub" else "connectome",
                    "neuromuscular": "handbuilt",
                    "readout": "fitted",
                    "autopilot": "handbuilt",
                    "body": "handbuilt",
                    "sensory_feedback": "handbuilt",
                    "visual_calibration": "fitted" if self.visual_calibration is not None else "none",
                    "visual_control_readout": "fitted" if self.visual_control_readout is not None else "none",
                    "flight_state_drive": (
                        "handbuilt_hypothesis" if self.flight_state_drive is not None else "none"
                    ),
                    "flight_body_stabilizer": (
                        "handbuilt_hypothesis" if self.flight_body_stabilizer is not None else "none"
                    ),
                },
            }
            rows.append(_json_safe(row))
            t_s = next_t
            step_index += 1

        final_pose = _pose_from_state(dynamics.state)
        final_observation = self.world.target_observation(final_pose)
        initial_pose = scenario.initial_pose
        initial_altitude = float(initial_pose.position_m[2] - self.world.config.ground_z_m)
        max_altitude_value = max_altitude if rows else initial_altitude
        final_forward_displacement = (
            (float(final_pose.position_m[0]) - float(initial_pose.position_m[0])) * math.cos(float(initial_pose.yaw_rad))
            + (float(final_pose.position_m[1]) - float(initial_pose.position_m[1])) * math.sin(float(initial_pose.yaw_rad))
        )
        target_visibility_fraction = target_visible_count / max(1, len(rows))
        target_on_target_fraction = target_on_target_count / max(1, len(rows))
        metrics = {
            "sample_count": len(rows),
            "frame_count": len(rows),
            "neural_steps": len(rows) * (1 if self.options.neural == "stub" else CNS_STEPS_PER_FLYVIS),
            "duration_s": t_s,
            "dt_s": FLYVIS_DT_S,
            "collision_count": len(collisions),
            "contact_count": contact_count,
            "max_altitude_m": max_altitude if rows else 0.0,
            "min_altitude_m": min_altitude if rows else 0.0,
            "altitude_gain_m": max_altitude_value - initial_altitude,
            "min_altitude_gain_m": max_altitude_value - initial_altitude,
            "max_horizontal_drift_m": max_drift,
            "max_altitude_error_m": max_alt_error,
            "final_altitude_m": final_pose.position_m[2] - self.world.config.ground_z_m,
            "final_speed_mps": final_pose.speed_mps,
            "max_speed_mps": max_speed,
            "forward_displacement_m": final_forward_displacement,
            "min_forward_displacement_m": final_forward_displacement,
            "horizontal_displacement_m": math.hypot(
                float(final_pose.position_m[0]) - float(initial_pose.position_m[0]),
                float(final_pose.position_m[1]) - float(initial_pose.position_m[1]),
            ),
            "final_target_distance_m": final_observation.distance_m,
            "min_target_distance_m": min(target_distances, default=math.inf),
            "target_visible_fraction": target_visibility_fraction,
            "min_target_visibility_fraction": target_visibility_fraction,
            "visibility_fraction": target_visibility_fraction,
            "target_visibility_fraction": target_visibility_fraction,
            "target_on_target_fraction": target_on_target_fraction,
            "yaw_change_rad": abs(final_pose.yaw_rad - initial_pose.yaw_rad),
            "gust_observed": bool(scenario.wind_override and scenario.wind_override.get("gusts")),
            "max_flow_magnitude": max_flow,
        }
        acceptance = _acceptance(scenario.acceptance, metrics)
        return {
            "name": scenario_name,
            "config_name": config_name,
            "duration_s": t_s,
            "metrics": _json_safe(metrics),
            "acceptance": acceptance,
            "rows": rows,
            "collisions": collisions,
            "final_state": _json_safe(dynamics.state.to_dict()),
        }

    def run(self) -> dict[str, Any]:
        pipeline = _FrameStagePipeline(
            enabled=bool(self.options.parallel_frame_stages)
        )
        self._frame_stage_pipeline = pipeline
        try:
            return self._run_impl()
        finally:
            self._last_frame_stage_pipeline = pipeline.manifest()
            pipeline.close()
            self._frame_stage_pipeline = None

    def _run_impl(self) -> dict[str, Any]:
        # Some adapters publish target metadata only after construction (or
        # after a lazy runtime preflight).  Refresh the bounded summary once
        # before rows/manifests are assembled; no target arrays are copied into
        # per-frame telemetry.
        self.flyvis_mapping = _flyvis_mapping_summary(self.flyvis)
        results: dict[str, Any] = {}
        if self.is_free_behavior:
            # One continuous escape experiment has no mission scenarios.  The
            # scene initializer is selected by ``_free_scenario`` and its name
            # is retained only as provenance, never as a controller input.
            result = self._one_free_scenario()
            results["free-behavior"] = result
        else:
            requested_scenarios = [name for name in self.options.scenarios if name != "all"]
            if "all" in self.options.scenarios:
                requested_scenarios = list(self.world.scenarios)
            # A default/all run includes newly added mission scenarios (for
            # example ``flight-demo``) without hard-coding their names here.  An
            # explicitly selected list remains exact and can still use ``turn``.
            if self.options.all_configured_scenarios:
                for configured in self.world.scenarios:
                    if configured not in requested_scenarios and configured not in SCENARIO_ALIASES:
                        requested_scenarios.append(str(configured))
            for name in requested_scenarios:
                config_name = _canonical_scenario_name(name)
                if config_name not in self.world.scenarios:
                    raise LauncherError(f"scenario {name!r} (config name {config_name!r}) is absent from {self.options.config}")
                results[name] = self._one_scenario(name, config_name=config_name)
        all_rows = [row for result in results.values() for row in result.pop("rows")]
        trajectory_schema_version = (
            MINIMAL_TRAJECTORY_SCHEMA_VERSION
            if self.minimal_telemetry
            else COMPACT_TRAJECTORY_SCHEMA_VERSION
            if self.compact_telemetry and self.is_free_behavior
            else FULL_DEBUG_TRAJECTORY_SCHEMA_VERSION
            if self.full_debug
            else "digital-fly-trajectory-1"
        )
        telemetry_schema_version = (
            MINIMAL_TELEMETRY_SCHEMA_VERSION
            if self.minimal_telemetry
            else COMPACT_TELEMETRY_SCHEMA_VERSION
            if self.compact_telemetry and self.is_free_behavior
            else FULL_DEBUG_TELEMETRY_SCHEMA_VERSION
            if self.full_debug
            else "digital-fly-telemetry-1"
        )
        trajectory = {
            "schema_version": trajectory_schema_version,
            "metadata": {
                "evidence_class": "digital_simulation",
                "source": "run_digital_fly",
                "execution_policy": self.execution_policy,
                "telemetry_mode": self.telemetry_mode,
                "telemetry_schema_version": telemetry_schema_version,
                "scenarios": list(results),
                "frame_rate_hz": 1.0 / FLYVIS_DT_S,
                "neural_rate_hz": CNS_HZ,
                "neural_substeps_per_frame": CNS_STEPS_PER_FLYVIS,
                "neural_window_api": (
                    getattr(self.neural_kind, "window_api", "stub_step")
                    if self.options.neural == "malecns" else "stub_step"
                ),
                "ground_plane_z_m": float(self.world.config.ground_z_m),
                "body_contact_plane_z_m": float(self.world.config.ground_z_m + self.world.config.body_radius_m),
                "phase_contract": (
                    "prohibited_free_behavior"
                    if self.is_free_behavior
                    else "normalized_local_phase within active VirtualWorld phase [0,1]"
                ),
                "render_fields": (
                    [] if self.minimal_telemetry else
                    ["render.left", "render.right", "render.pose"]
                    if not (self.compact_telemetry and self.is_free_behavior)
                    else ["visual_input.left_columns", "visual_input.right_columns"]
                ),
                "flight_state_drive": bool(self.flight_state_drive is not None),
                "flight_body_stabilizer": bool(self.flight_body_stabilizer is not None),
                "compact_neural_telemetry": bool(
                    self.compact_telemetry
                    and self.options.neural == "malecns"
                    and not self.options.full_neural_telemetry
                ),
                "minimal_neural_telemetry": dict(self.minimal_neural_telemetry),
                "minimal_hash_only_visual_input": bool(self.minimal_telemetry),
                "compact_contract": (
                    {
                        "schema_version": COMPACT_TELEMETRY_SCHEMA_VERSION,
                        "immutable_provenance": "manifest.json",
                        "full_debug_opt_in": True,
                        "strict_rectangular_render": bool(
                            self.is_free_behavior
                            and bool(self.escape_channels.roles)
                        ),
                    }
                    if self.compact_telemetry and self.is_free_behavior else None
                ),
                "autonomy_policy": (
                    {
                        "name": FREE_BEHAVIOR_POLICY_NAME,
                        "control_authority": "MaleCNS",
                        "control_path": list(FREE_BEHAVIOR_ALLOWED_CONTROL_PATH),
                        "prohibited_paths": list(FREE_BEHAVIOR_PROHIBITED_PATHS),
                        "fail_closed": True,
                    }
                    if self.is_free_behavior else None
                ),
            },
            "trajectory": all_rows,
        }
        resource_usage = _runtime_resource_usage(
            self.options.vision_device
            if self.options.vision_device.startswith("cuda:")
            else ("cuda:0" if self.options.neural_device == "cuda" else "cpu")
        )
        # Immutable scene/mapping/readout records are written once in the
        # manifest.  Compact rows refer to these records instead of copying
        # them (or their large target arrays) at every frame.
        if self.is_free_behavior:
            scene_manifests: dict[str, Any] = {
                "free-behavior": self._free_world_geometry(),
            }
        else:
            scene_manifests = {
                str(name): self.world.manifest(self.world.scenarios[config_name])
                for name, result in results.items()
                for config_name in (_canonical_scenario_name(result.get("config_name", name)),)
                if config_name in self.world.scenarios
            }
        mapping_metadata = _flyvis_mapping_metadata(self.flyvis)
        mapping_manifest: dict[str, Any] = {
            "summary": self.flyvis_mapping,
            "units": mapping_metadata.get("units"),
            "target_indices": _lossless_array_record(
                mapping_metadata.get("target_indices", ()), include_data=True,
            ) if mapping_metadata.get("target_indices") is not None else None,
            "target_ids": _lossless_array_record(
                mapping_metadata.get("target_ids", ()), include_data=True,
            ) if mapping_metadata.get("target_ids") is not None else None,
        }
        neuromuscular_manifest: dict[str, Any] = {
            "adapter": type(self.neuromuscular).__name__,
            "strict": bool(self.options.neural == "malecns"),
            "map_path": str(getattr(self.neuromuscular, "map_path", "")),
            "map_sha256": getattr(self.neuromuscular, "map_sha256", None),
            "calibration_id": getattr(self.neuromuscular, "calibration_id", None),
            "channel_names": sorted(
                str(name) for name in getattr(self.neuromuscular, "channels", {})
            ),
            "units": "normalized actuator channels",
        }
        neural_cuda_graph_manifest = (
            self.neural_kind.cuda_graph_manifest()
            if self.options.neural == "malecns"
            and callable(getattr(self.neural_kind, "cuda_graph_manifest", None))
            else {
                "requested": bool(self.options.neural_cuda_graph),
                "effective": False,
                "reason": (
                    "CUDA Graphs require the real MaleCNS engine"
                    if self.options.neural_cuda_graph else None
                ),
                "backend": "stub",
                "device": "cpu",
                "capture_count": 0,
                "replay_count": 0,
                "phase": "disabled",
            }
        )
        resolver = getattr(self.sensory_feedback, "resolver", None)
        sensory_manifest: dict[str, Any] = {
            "adapter": type(self.sensory_feedback).__name__,
            "dataset": getattr(resolver, "dataset", None),
            "config_path": str(getattr(resolver, "config_path", "")),
            "config_sha256": _sha256_file(Path(getattr(resolver, "config_path")))
            if getattr(resolver, "config_path", None) else None,
            "population_drive_units": getattr(
                self.sensory_feedback, "population_drive_units", "normalized"
            ),
            "population_drive_range": list(getattr(
                self.sensory_feedback, "population_drive_range", (-1.0, 1.0)
            )),
            "population_metadata": _compact_json_value(
                getattr(resolver, "metadata", lambda: {})(), include_array_data=False,
            ) if callable(getattr(resolver, "metadata", None)) else {},
            "ablation": self._mechanosensory_ablation_manifest(),
        }
        telemetry_contract = {
            "mode": self.telemetry_mode,
            "schema_version": telemetry_schema_version,
            "full_debug_opt_in": True,
            "immutable_provenance_location": "manifest.json",
            "minimal_hash_only_input_fingerprint": (
                {
                    "encoding": "hash-only",
                    "algorithm": "sha256",
                    "byte_order": "contiguous C-order",
                    "fields": [
                        "visual_input.left_columns",
                        "visual_input.right_columns",
                        "visual_input.t4t5_drive",
                    ],
                }
                if self.minimal_telemetry else None
            ),
            "strict_rectangular_render": bool(
                self.is_free_behavior and (
                    self.full_debug or bool(self.escape_channels.roles)
                )
            ),
            "rectangular_render_policy": (
                "telemetry_video_only_exact_flyvis_control"
                if self.options.vision == "flyvis" and self.is_free_behavior
                else "telemetry_video_only_mission_exact_flyvis_control"
                if self.options.vision == "flyvis"
                else "controller_input_stub"
            ),
            "exact_flyvis_control_input": (
                "runtime_input_index_eye_directions"
                if self.options.vision == "flyvis" else "not_applicable_stub"
            ),
            "retained_signals": [
                "timestamps/frame_index", "body_state_before_and_after",
                "wind/contact/collision/target-state", "mechanosensory_channels",
                *(["food_cue.raw_field/contact/poisson_hz"] if self.food_model is not None else []),
                *(["neural_intervention.active_selector_inputs"]
                  if self.neural_intervention_schedule is not None else []),
                *(["mechanosensory_ablation.pre_ablation_population_drive/"
                    "male_cns_population_channels"]
                   if self.mechanosensory_ablations else []),
                *(["exact_flyvis_columns_and_t4t5_drive_hash_only"]
                  if self.minimal_telemetry else
                  ["exact_flyvis_columns_and_t4t5_drive_lossless"]
                  if self.is_free_behavior else ["visual_drive_summary"]),
                *(
                    ["rectangular_render_lossless_dynamic_provider_context"]
                    if self.is_free_behavior and bool(self.escape_channels.roles)
                    else []
                ),
                "neural_population_summaries_and_window_timing",
                "neuromuscular_channels", "jump_events", "motor_command",
                *(["ground_walking.direct_leg_readout/contact_gate/applied_force"]
                  if self.ground_walking_settings is not None
                  and self.ground_walking_settings.get("enabled") else []),
            ],
            "omitted_or_manifest_referenced": [
                "world_geometry",
                *(
                    []
                    if self.is_free_behavior and bool(self.escape_channels.roles)
                    else ["rendered_rectangular_raster"]
                ),
                "dense_neural_vectors_unless_full_debug",
                "component/provenance/channel-definition_metadata",
                "duplicate_contract_wrappers_and_target_mapping",
                *(["dense_visual_arrays", "mechanosensory_mapping/provenance",
                   "repeated_world/render/contract_wrappers"]
                  if self.minimal_telemetry else []),
            ],
        }
        autonomy_policy = None
        provenance_autonomy_policy = None
        if self.is_free_behavior:
            autonomy_policy = {
                "name": FREE_BEHAVIOR_POLICY_NAME,
                "strict": True,
                "control_authority": "MaleCNS",
                "allowed_control_path": list(FREE_BEHAVIOR_ALLOWED_CONTROL_PATH),
                "prohibited_paths": list(FREE_BEHAVIOR_PROHIBITED_PATHS),
                "stimulus_labels": "record_only",
                "stimulus_direction": "record_only",
                "future_state": "record_only",
                "stimulus_interaction": "visual_only_non_interacting",
                "gpu_allowed": "explicit_adapter_gate",
                "fail_closed": True,
                "external_neuron_stimulation": self.neural_intervention_schedule is not None,
                "ground_walking_actuator": bool(
                    self.ground_walking_settings is not None
                    and self.ground_walking_settings.get("enabled")
                ),
                "counts_as_autonomous_foraging": (
                    self.neural_intervention_schedule is None
                    and self.options.courtship_target_trajectory is None
                    and self.options.courtship_movie is None
                ),
                **({
                    "courtship_visual_assay": True,
                    "counts_as_autonomous_courtship": False,
                } if self.options.courtship_target_trajectory is not None
                    or self.options.courtship_movie is not None else {}),
                **({
                    "engineered_grooming_sidecar": True,
                    "counts_as_autonomous_grooming": False,
                    "counts_as_autonomous_foraging": False,
                } if self.grooming_bridge_config is not None else {}),
            }
            provenance_autonomy_policy = {
                **autonomy_policy,
                "allowed_control_path": list(FREE_BEHAVIOR_ALLOWED_CONTROL_PATH),
                "prohibited_paths": list(FREE_BEHAVIOR_PROHIBITED_PATHS),
            }
        renderer_transfer = _json_safe({
            **self._exact_column_transfer,
            "exact_columns": self._exact_column_transfer.get("path"),
            "note": (
                "control handoff is observed at the exact-column seam; "
                "telemetry retains a bounded host float32 copy"
            ),
        })
        manifest = {
            "schema_version": "digital-fly-run-1",
            "launcher": "run_digital_fly.py",
            "execution_policy": self.execution_policy,
            "telemetry_contract": telemetry_contract,
            "autonomy_policy": autonomy_policy,
            "run_classification": (
                self.neural_intervention_schedule.classification
                if self.neural_intervention_schedule is not None
                else "engineered_dust_neural_grooming_loop"
                if self.grooming_bridge_config is not None
                else "courtship_open_loop_visual_probe"
                if self.options.courtship_movie is not None
                else "courtship_closed_loop_visual_assay"
                if self.options.courtship_target_trajectory is not None
                else "autonomous_candidate"
            ),
            "neural_intervention": self._neural_intervention_manifest(),
            **({"ground_walking": {
                "evidence_class": "reduced_contact_gated_neural_motor_actuator",
                "config": dict(self.ground_walking_settings),
                "motor_source": "NeuromuscularAdapter.leg_power/leg_steering",
                "no_behavior_policy": True,
                "no_articulated_leg_physics": True,
            }} if self.ground_walking_settings is not None
                 and self.ground_walking_settings.get("enabled") else {}),
            **({"courtship_visual": {
                "schema_version": "courtship-visual-assay-1",
                "target_source": "independent_world_trajectory",
                "target_condition": self.options.courtship_target_condition,
                "trajectory_path": str(self.courtship_target_trajectory_path),
                "trajectory_sha256": self.courtship_target_trajectory_sha256,
                "target_source_sha256": _sha256_file(SCRIPT_DIR / "courtship_target.py"),
                "camera": self.world.camera.to_dict(),
                "asset_binding": _json_safe(
                    self.world.external_scene_manifest.get("stimulus_visual_bindings", {}).get("female_fly")
                ) if self.world.external_scene_manifest is not None else None,
                "control_authority": "MaleCNS",
                "target_labels_enter_control": False,
                "evidence_scope": "visual orientation and tracking only; no complete courtship claim",
            }} if self.options.courtship_target_trajectory is not None else {}),
            **({"courtship_visual": {
                "schema_version": "courtship-visual-assay-1",
                "target_source": "open_loop_movie",
                "movie": self.courtship_movie_stimulus.manifest(),
                "movie_path": str(self.courtship_movie_path),
                "movie_sha256": self.courtship_movie_sha256,
                "movie_source_sha256": _sha256_file(SCRIPT_DIR / "courtship_movie.py"),
                "camera": self.world.camera.to_dict(),
                "pose_dependent": False,
                "control_authority": "MaleCNS",
                "evidence_scope": "visual pathway control only; no binocular or courtship claim for replicated mono",
            }} if self.options.courtship_movie is not None else {}),
            **({"grooming_loop": {
                    "schema_version": "male-cns-grooming-loop-1",
                    "evidence_class": "engineered_dust_transduction_and_foreleg_motor_primitive",
                    "config_path": str(self.grooming_loop_config_path),
                    "config_sha256": self.grooming_loop_config_sha256,
                    "bridge_source_sha256": _sha256_file(SCRIPT_DIR / "grooming_neural_bridge.py"),
                    "body_source_sha256": _sha256_file(SCRIPT_DIR / "grooming_body.py"),
                    "selectors_by_region": dict(self.grooming_bridge_config.selectors_by_region),
                    "resolved_selector_counts": dict(self.grooming_selector_counts),
                    "rate_hz_by_region": dict(self.grooming_bridge_config.rate_hz_by_region),
                    "dust_seed": self.options.seed,
                    "dust_by_region": {"antenna": self.options.grooming_dust_count},
                    "sensor_off": bool(self.options.grooming_sensor_off),
                    "foreleg_lock": bool(self.options.grooming_foreleg_lock),
                    "neural_readout": "groom_antenna (DNg62 and DNge078)",
                    "motor_trajectory": "handbuilt amplitude-gated foreleg CPG",
                    "counts_as_autonomous_grooming": False,
                }} if self.grooming_bridge_config is not None else {}),
            "prohibited_paths": (
                list(FREE_BEHAVIOR_PROHIBITED_PATHS) if self.is_free_behavior else []
            ),
            "seed": self.options.seed,
            "config": str(self.options.config),
                "requested_backends": {
                    "neural": self.options.neural,
                    "vision": self.options.vision,
                    "body": self.options.body,
                },
                "requested_renderer_backend": self.options.renderer_backend,
                "requested_execution": {
                "mode": self.execution_policy,
                    "strict": bool(self.is_free_behavior),
                    "renderer_backend": self.options.renderer_backend,
                "telemetry_mode": self.telemetry_mode,
                "full_debug": bool(self.full_debug),
                "neural_device": self.options.neural_device,
                "neural_backend": self.options.neural_backend,
                "neural_cuda_graph": bool(self.options.neural_cuda_graph),
                "parallel_frame_stages": bool(self.options.parallel_frame_stages),
                "vision_device": self.options.vision_device,
                "max_vram_gib": self.options.max_vram_gib,
                "frame_count": self.options.frame_count,
                "full_neural_telemetry": bool(self.options.full_neural_telemetry),
                "minimal_neural_telemetry": _json_safe(self.minimal_neural_telemetry),
                "visual_calibration": None if self.options.visual_calibration is None else str(self.options.visual_calibration),
                "visual_tonic_state_inputs": bool(self.options.visual_tonic_state_inputs),
                "neural_csr_path": None if self.options.neural_csr_path is None else str(self.options.neural_csr_path),
                "flight_state_drive": None if self.options.flight_state_drive is None else str(self.options.flight_state_drive),
                "flight_body_stabilizer": (
                    None if self.options.flight_body_stabilizer is None
                    else str(self.options.flight_body_stabilizer)
                ),
                "dynamic_stimulus": _json_safe(self.options.dynamic_stimulus),
                "jump_channel": _json_safe(self.options.jump_channel),
                "posture_channel": _json_safe(self.options.posture_channel),
                "stimulus_config": None if self.options.stimulus_config is None else str(self.options.stimulus_config),
                "scene_manifest": None if self.options.scene_manifest is None else str(Path(self.options.scene_manifest).expanduser().resolve()),
                "courtship_target_trajectory": (
                    str(self.courtship_target_trajectory_path)
                    if self.courtship_target_trajectory_path is not None else None
                ),
                "courtship_target_condition": self.options.courtship_target_condition,
                "courtship_movie": (
                    str(self.courtship_movie_path)
                    if self.courtship_movie_path is not None else None
                ),
                "courtship_movie_eye_mode": self.options.courtship_movie_eye_mode,
                "courtship_movie_stereo_layout": self.options.courtship_movie_stereo_layout,
                "stimulus_name": self.options.stimulus_name,
                "stimulus_condition": self.options.stimulus_condition,
                "neural_gain": self.options.neural_gain,
                "mechanosensory_ablations": list(self.mechanosensory_ablations),
                "neural_intervention_config": (
                    str(self.neural_intervention_config_path)
                    if self.neural_intervention_config_path is not None else None
                ),
                "neural_intervention_config_sha256": self.neural_intervention_config_sha256,
                **({
                    "grooming_loop_config": str(self.grooming_loop_config_path),
                    "grooming_loop_config_sha256": self.grooming_loop_config_sha256,
                    "grooming_dust_count": self.options.grooming_dust_count,
                    "grooming_sensor_off": bool(self.options.grooming_sensor_off),
                    "grooming_foreleg_lock": bool(self.options.grooming_foreleg_lock),
                } if self.grooming_bridge_config is not None else {}),
                **({
                    "food_config": str(self.food_config_path),
                    "food_config_sha256": self.food_config_sha256,
                    "food_cue_ablations": list(self.food_cue_ablations),
                } if self.food_model is not None else {}),
                "palm_speed_mps": self.options.palm_speed_mps,
                "impact_time_s": self.options.impact_time_s,
            },
            "effective_execution": {
                "renderer_backend": getattr(
                    self.world, "renderer_backend",
                    getattr(self.world, "backend_role", self.options.renderer_backend),
                ),
                "renderer_transfer": _json_safe(renderer_transfer),
                "neural_device": (
                    getattr(self.neural_kind, "device_name", "stub")
                    if self.options.neural == "malecns" else "cpu"
                ),
                "neural_backend": (
                    getattr(self.neural_kind, "backend_name", "stub")
                    if self.options.neural == "malecns" else StubNeural.backend_name
                ),
                "neural_window_api": (
                    getattr(self.neural_kind, "window_api", "scalar_step")
                    if self.options.neural == "malecns" else "stub_step"
                ),
                "neural_warmup": (
                    _json_safe(getattr(self.neural_kind, "warmup_report", None))
                    if self.options.neural == "malecns" else None
                ),
                "neural_cuda_graph": _json_safe(neural_cuda_graph_manifest),
                "frame_stage_pipeline": _json_safe(
                    self._frame_stage_pipeline.manifest()
                    if self._frame_stage_pipeline is not None
                    else {"requested": False, "effective": False}
                ),
                "vision_device": (
                    getattr(self.flyvis_runtime_adapter, "device", "unknown")
                    if self.options.vision == "flyvis" else "cpu"
                ),
                "max_vram_gib": self.options.max_vram_gib,
                "visual_control_readout": (
                    "flyvis_control_readout" if self.visual_control_readout is not None else "none"
                ),
                "flight_state_drive": (
                    "flight_state_drive" if self.flight_state_drive is not None else "none"
                ),
                "flight_body_stabilizer": (
                    "flight_body_stabilizer" if self.flight_body_stabilizer is not None else "none"
                ),
                "escape_channels": (
                    self.escape_channels.manifest() if self.is_free_behavior else "disabled"
                ),
                "physical_stimulus": (
                    {
                        "requested_condition": self.escape_stimulus_condition,
                        "resolved_name": self.escape_stimulus_name,
                    }
                    if self.is_free_behavior else "disabled"
                ),
                "neural_gain": (
                    getattr(self.neural_kind, "neural_gain", None)
                    if self.options.neural == "malecns" else None
                ),
                "mechanosensory_ablation": self._mechanosensory_ablation_manifest(),
                **({"food_cue_ablations": list(self.food_cue_ablations)}
                   if self.food_model is not None else {}),
            },
            "effective_backends": {
                "world": getattr(self.world, "backend_name", "virtual_world_raycast"),
                "renderer": getattr(
                    self.world, "renderer_backend",
                    getattr(self.world, "backend_role", self.options.renderer_backend),
                ),
                "vision": getattr(self.flyvis, "backend_name", getattr(self.flyvis.backend, "name", "unknown")),
                "neural": (getattr(self.neural_kind, "backend_name", "deterministic_neural_stub")
                           if self.options.neural == "malecns" else StubNeural.backend_name),
                "neuromuscular": "neuromuscular_adapter",
                "autopilot": "prohibited" if self.is_free_behavior else "flight_autopilot",
                "body": "flight_dynamics_reduced_6dof",
                "sensory_feedback": "mechanosensory_feedback",
                **({"food_sensing": "handbuilt_spatial_odor_and_contact_sugar"}
                   if self.food_model is not None else {}),
                "visual_control_readout": (
                    "prohibited_free_behavior"
                    if self.is_free_behavior
                    else ("flyvis_control_readout" if self.visual_control_readout is not None else "stub_mean")
                ),
                "flight_state_drive": (
                    "flight_state_drive" if self.flight_state_drive is not None else "disabled"
                ),
                "flight_body_stabilizer": (
                    "flight_body_stabilizer" if self.flight_body_stabilizer is not None else "disabled"
                ),
                "escape_channels": (
                    "physical_world_stimulus;telemetry_only_dynamic_probe"
                    if self.is_free_behavior else "disabled"
                ),
            },
            "provenance": {
                "evidence_class": "digital_simulation",
                "execution_policy": self.execution_policy,
                "world": "handbuilt",
                "vision": "handbuilt" if self.options.vision == "stub" else "fitted",
                "neural": "handbuilt" if self.options.neural == "stub" else "connectome",
                "neuromuscular": "connectome_adapter" if self.is_free_behavior else "handbuilt",
                "readout": "fitted",
                "autopilot": "prohibited" if self.is_free_behavior else "handbuilt",
                "body": "handbuilt",
                "sensory_feedback": "handbuilt",
                **({"food_sensing": "handbuilt_receptor_proxy"}
                   if self.food_model is not None else {}),
                "mechanosensory_ablation": (
                    "explicit_population_input_intervention"
                    if self.mechanosensory_ablations else "none"
                ),
                "neural_intervention": (
                    self.neural_intervention_schedule.classification
                    if self.neural_intervention_schedule is not None else "none"
                ),
                "visual_calibration": (
                    "fitted" if self.visual_calibration is not None else "none"
                ),
                "visual_control_readout": (
                    "fitted" if self.visual_control_readout is not None else "none"
                ),
                "flight_state_drive": (
                    "handbuilt_hypothesis" if self.flight_state_drive is not None else "none"
                ),
                "flight_body_stabilizer": (
                    "handbuilt_hypothesis" if self.flight_body_stabilizer is not None else "none"
                ),
                "autonomy_policy": provenance_autonomy_policy,
                "prohibited_paths": (
                    list(FREE_BEHAVIOR_PROHIBITED_PATHS) if self.is_free_behavior else []
                ),
            },
            "components": {
                "world": {
                    "is_stub": False,
                    "source": "handbuilt",
                    "backend": getattr(self.world, "backend_name", "virtual_world_raycast"),
                    "renderer_backend": getattr(
                        self.world, "renderer_backend",
                        getattr(self.world, "backend_role", "analytic_reference"),
                    ),
                    "renderer_transfer": _json_safe(renderer_transfer),
                    "scene_manifests": _json_safe(scene_manifests),
                },
                "vision": {
                    "is_stub": bool(getattr(self.flyvis, "backend_is_stub", False)),
                    "source": "handbuilt" if self.options.vision == "stub" else "fitted",
                    "runtime": (
                        self.flyvis_runtime_adapter.manifest()
                        if self.options.vision == "flyvis" else None
                    ),
                    "mapping": self.flyvis_mapping,
                    "mapping_arrays": _json_safe(mapping_manifest),
                    "exact_direction_mapping": (
                        {
                            "source": self._exact_direction_cache.get("source"),
                            "sha256": self._exact_direction_cache.get("directions_sha256"),
                            "normalized_for_render": True,
                            "cache": {
                                "validated_once": True,
                                "device": self._exact_direction_cache.get("device"),
                                "dtype": self._exact_direction_cache.get("device_dtype"),
                            },
                        }
                        if isinstance(self._exact_direction_cache, Mapping) else None
                    ),
                    "calibration": (
                        {
                            **self.visual_calibration.manifest(),
                            "runner_tonic_state_inputs_opt_in": bool(
                                self.options.visual_tonic_state_inputs
                            ),
                        }
                        if self.visual_calibration is not None else None
                    ),
                    "control_readout": (
                        self.visual_control_readout.manifest()
                        if self.visual_control_readout is not None else None
                    ),
                },
                "neural": {
                    "is_stub": self.options.neural == "stub",
                    "source": "handbuilt" if self.options.neural == "stub" else "connectome",
                    "runtime": (
                        _optional_manifest(getattr(self.neural_kind, "engine", None))
                        if self.options.neural == "malecns" else None
                    ),
                    "requested_device": self.options.neural_device,
                    "requested_backend": self.options.neural_backend,
                    "effective_device": (
                        getattr(self.neural_kind, "device_name", "stub")
                        if self.options.neural == "malecns" else "cpu"
                    ),
                    "effective_backend": (
                        getattr(self.neural_kind, "backend_name", "stub")
                        if self.options.neural == "malecns" else StubNeural.backend_name
                    ),
                    "window_api": (
                        getattr(self.neural_kind, "window_api", "scalar_step")
                        if self.options.neural == "malecns" else "stub_step"
                    ),
                    "warmup": (
                        _json_safe(getattr(self.neural_kind, "warmup_report", None))
                        if self.options.neural == "malecns" else None
                    ),
                    "cuda_graph": _json_safe(neural_cuda_graph_manifest),
                    "reported_populations": list(self.escape_report_populations),
                    "minimal_dense_input_echo": _json_safe(self.minimal_neural_telemetry),
                },
                "neural_intervention": self._neural_intervention_manifest(),
                "neuromuscular": _json_safe(neuromuscular_manifest),
                "sensory_feedback": _json_safe(sensory_manifest),
                **({"food_sensing": {
                    "source": "handbuilt",
                    "input_to_malecns": "poisson_hz",
                    "asset_binding": "independent_physical_config",
                    "contact_kind": "declared_horizontal_support_and_mouth_sphere_proxy",
                    "config_path": str(self.food_config_path),
                    "config_sha256": self.food_config_sha256,
                    "module_sha256": _sha256_file(SCRIPT_DIR / "food_sensing.py"),
                    "contact_surface_id": self.food_model.contact_surface_id,
                    "selector_counts": dict(self.food_selector_counts),
                    "ablated_senses": list(self.food_cue_ablations),
                    "config": _json_safe(self.food_config_mapping),
                }} if self.food_model is not None else {}),
                "flyvis_mapping": _json_safe(mapping_manifest),
                "body": {
                    "is_stub": False,
                    "reduced": True,
                    "source": "handbuilt",
                    "physical_wing_motor_required": bool(self.is_free_behavior),
                    "allow_legacy_wing_fallback": bool(not self.is_free_behavior),
                },
                "flight_state_drive": (
                    self.flight_state_drive.manifest()
                    if self.flight_state_drive is not None else None
                ),
                "flight_body_stabilizer": (
                    self.flight_body_stabilizer.manifest()
                    if self.flight_body_stabilizer is not None else None
                ),
                "escape_channels": (
                    self.escape_channels.manifest() if self.is_free_behavior else None
                ),
                "physical_stimulus": (
                    {
                        "path": str(self.escape_stimulus_path),
                        "name": self.escape_stimulus_name,
                        "condition": self.escape_stimulus_condition,
                        "state_for_control": "prohibited;telemetry_only",
                        "interaction": "visual_only_non_interacting",
                        "visual_only": True,
                        "non_interacting": True,
                        "contact": "counterfactual_telemetry_only",
                        "planned_contact_deadline_s": self.escape_experiment_spec.get(
                            "surface_contact_deadline_s", None,
                        ),
                        "manifest": self.escape_stimulus_manifest,
                    }
                    if self.is_free_behavior else None
                ),
                "experiment": (
                    _json_safe(self.escape_experiment_spec) if self.is_free_behavior else None
                ),
            },
            "resource_limits": {
                "cgroup_memory_max_bytes": _cgroup_memory_limit_bytes(),
                "torch_allocator_max_vram_gib": (
                    self.options.max_vram_gib if (
                        self.options.vision_device.startswith("cuda:")
                        or self.options.neural_device == "cuda"
                    ) else None
                ),
                "neural_substeps_per_frame": CNS_STEPS_PER_FLYVIS,
                "flyvis_frame_dt_s": FLYVIS_DT_S,
                "cns_dt_s": CNS_DT_S,
            },
            "resource_usage": resource_usage,
            "capabilities": self.capabilities,
            "units": {
                "time_s": "s", "position_m": "m", "velocity_mps": "m/s",
                "orientation_rad": "rad", "orientation_quat": "unitless",
                "angular_velocity_rad_s": "rad/s", "wing_drive": "normalized",
                "force_N": "N", "torque_Nm": "N*m",
                "ground_plane_z_m": "world-floor m",
                "body_contact_plane_z_m": "world-floor plus body radius m",
                "altitude_error_m": (
                    "prohibited_free_behavior"
                    if self.is_free_behavior else "goal-minus-measured m"
                ),
                "phase": (
                    "prohibited_free_behavior"
                    if self.is_free_behavior else "normalized local active-phase progress [0,1]"
                ),
            },
            "scenario_names": list(results),
            "scenario_results": results,
            "outputs": {
                "manifest": "manifest.json",
                "telemetry": "telemetry.jsonl",
                "trajectory": "trajectory.json",
                "validation": "validation.json",
                "video": "flight.mp4",
            },
            "validation": {
                "validator": "scripts/validate_digital_fly.py",
                "input": "trajectory.json",
                "expected_evidence_class": "digital_simulation",
                "biological_claim": False,
            },
        }
        # Rows are made JSON-safe exactly once at the frame boundary above.
        # Recursing over the complete trajectory here used to allocate a
        # second copy of every row before ``write_outputs`` could stream it.
        return {"manifest": _json_safe(manifest), "trajectory": trajectory}

    def write_outputs(self, payload: Mapping[str, Any]) -> dict[str, str]:
        out = Path(self.options.out)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "manifest.json"
        telemetry_path = out / "telemetry.jsonl"
        trajectory_path = out / "trajectory.json"
        manifest = dict(payload["manifest"])
        rows = payload["trajectory"]["trajectory"]
        manifest["outputs"] = {
            "manifest": str(manifest_path),
            "telemetry": str(telemetry_path),
            "trajectory": str(trajectory_path),
            "validation": str(out / "validation.json"),
            "video": str(out / "flight.mp4"),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        trajectory = dict(payload["trajectory"])
        trajectory["metadata"] = {**trajectory.get("metadata", {}), "manifest_path": str(manifest_path)}
        # ``trajectory`` can contain hundreds of megabytes of per-frame
        # evidence.  ``Path.write_text(json.dumps(...))`` first materialises a
        # second, equally large Python string and pushed a completed 2400-frame
        # run over its 5 GiB cgroup limit during finalisation.  ``json.dump``
        # drives ``JSONEncoder.iterencode`` directly into the file instead, so
        # only a bounded encoder chunk is live alongside the already-recorded
        # rows.  Keep the same JSON schema/indentation for validators and the
        # renderer; this changes memory behaviour, not evidence content.
        if self.minimal_telemetry:
            # Encode every minimal row once and fan the same bounded string to
            # JSONL plus the wrapped trajectory.  The historical path encoded
            # all 2,000 rows twice, spending almost two wall seconds on a 10 s
            # run even though both files contain identical row objects.
            encoder = json.JSONEncoder(
                ensure_ascii=False, allow_nan=False, separators=(",", ":"),
            )
            with telemetry_path.open("w", encoding="utf-8") as telemetry_handle, \
                    trajectory_path.open("w", encoding="utf-8") as trajectory_handle:
                trajectory_handle.write("{")
                first_field = True
                for key, value in trajectory.items():
                    if not first_field:
                        trajectory_handle.write(",")
                    first_field = False
                    trajectory_handle.write(encoder.encode(str(key)))
                    trajectory_handle.write(":")
                    if key != "trajectory":
                        trajectory_handle.write(encoder.encode(value))
                        continue
                    trajectory_handle.write("[")
                    for index, row in enumerate(rows):
                        encoded = encoder.encode(row)
                        telemetry_handle.write(encoded)
                        telemetry_handle.write("\n")
                        if index:
                            trajectory_handle.write(",")
                        trajectory_handle.write(encoded)
                    trajectory_handle.write("]")
                trajectory_handle.write("}\n")
        else:
            with telemetry_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(
                        row, ensure_ascii=False, allow_nan=False,
                        separators=(",", ":"),
                    ) + "\n")
            with trajectory_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    trajectory,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                handle.write("\n")
        return {"manifest": str(manifest_path), "telemetry": str(telemetry_path), "trajectory": str(trajectory_path)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scene-manifest", type=Path, default=None,
                        help="optional external Blender/glTF scene manifest; visual geometry only")
    parser.add_argument("--courtship-target-trajectory", type=Path, default=None,
                        help="keyframed independent female trajectory; requires a female_fly glTF scene binding")
    parser.add_argument("--courtship-target-condition", choices=("moving", "static", "hidden"),
                        default="moving", help="matched moving, first-pose static, or hidden female visual target")
    parser.add_argument("--courtship-movie", type=Path, default=None,
                        help="open-loop video or saved exact binocular 200 Hz NPZ used as the FlyVis visual source")
    parser.add_argument("--courtship-movie-eye-mode", choices=("monocular_replicated", "binocular_pov"),
                        default="monocular_replicated")
    parser.add_argument("--courtship-movie-stereo-layout", choices=("side_by_side_lr",), default=None)
    parser.add_argument(
        "--mode", "--execution-mode",
        choices=("mission", "strict", "free-behavior", "escape-experiment"),
        default="mission",
        help="execution policy; free-behavior/escape-experiment is strict and uses an explicitly gated CPU/CUDA adapter",
    )
    parser.add_argument(
        "--strict", "--strict-free-behavior", dest="strict", action="store_true",
        help="enable the strict free-behavior escape-experiment policy",
    )
    parser.add_argument(
        "--free-behavior", dest="free_behavior", action="store_true",
        help="alias for --mode free-behavior",
    )
    parser.add_argument(
        "--escape-experiment", dest="escape_experiment", action="store_true",
        help="alias for --mode escape-experiment",
    )
    parser.add_argument("--scenario", action="append", dest="scenarios",
                        help="repeat; default runs configured scenarios (turn aliases yaw-step)")
    parser.add_argument("--neural", choices=("stub", "malecns"), default="stub")
    parser.add_argument(
        "--ablate-mechanosensory",
        dest="mechanosensory_ablations",
        action="append",
        type=str.lower,
        choices=tuple(MECHANOSENSORY_ABLATION_ENTRIES),
        default=None,
        help=(
            "zero one body-derived MaleCNS mechanosensory population group "
            "(repeat for jo, haltere, wing, leg); requires --neural malecns"
        ),
    )
    parser.add_argument(
        "--food-config", type=Path, default=None,
        help="separate physical food odor/contact cue JSON for strict free-behavior",
    )
    parser.add_argument(
        "--ablate-food-sense", dest="food_cue_ablations", action="append",
        choices=("odor", "taste"), default=None,
        help="mask one food receptor input at the CNS boundary while retaining its raw field telemetry",
    )
    parser.add_argument(
        "--neural-intervention-config", type=Path, default=None,
        help=(
            "explicit direct neuron-stimulation schedule for a separately labeled "
            "strict free-behavior intervention; never counts as autonomous foraging"
        ),
    )
    parser.add_argument(
        "--grooming-loop-config", type=Path, default=None,
        help="opt-in dust-to-MaleCNS-to-articulated-foreleg grooming micro-body experiment",
    )
    parser.add_argument("--grooming-dust-count", type=int, default=16,
                        help="initial antenna dust particles in the grooming micro-body")
    parser.add_argument("--grooming-sensor-off", action="store_true",
                        help="mask only dust-derived neural input, retaining raw dust telemetry")
    parser.add_argument("--grooming-foreleg-lock", action="store_true",
                        help="mechanically lock the grooming micro-body foreleg")
    parser.add_argument("--vision", choices=("stub", "flyvis"), default="stub")
    parser.add_argument("--body", choices=("reduced", "flybody"), default="reduced")
    parser.add_argument(
        "--renderer-backend",
        choices=("analytic_reference", "warp_cuda"),
        default="analytic_reference",
        help="world renderer backend; warp_cuda requires world_renderer_opt and never falls back",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--t4t5", type=int, default=8)
    parser.add_argument("--neural-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--neural-backend", choices=("auto", "cpu", "torch", "torch_cpu", "triton"), default="auto")
    parser.add_argument(
        "--neural-cuda-graph",
        action="store_true",
        help="explicitly request CUDA Graph capture/replay (requires --neural-device cuda --neural-backend triton)",
    )
    parser.add_argument("--neural-gain", type=float, default=None,
                        help="explicit MaleCNS synaptic gain calibration (e.g. 1.2 for the 0.5 m/s candidate)")
    parser.add_argument("--vision-device", default="cuda:0", help="cpu or cuda:N")
    parser.add_argument("--max-vram-gib", type=float, default=DEFAULT_MAX_VRAM_GIB)
    parser.add_argument("--duration", type=float, dest="duration_override_s", help="override each scenario duration for a smoke run")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--frames", type=int, dest="frame_count",
                        help="run exactly this many 5 ms FlyVis frames per scenario")
    parser.add_argument("--full-neural-telemetry", action="store_true",
                        help="retain dense real-neural rates/spikes in every row (large)")
    parser.add_argument(
        "--telemetry-mode", "--telemetry",
        choices=("compact", "minimal", "full-debug"), default="compact",
        help="per-frame telemetry contract; minimal is hash-only strict free-behavior production, compact is the existing production mode, and full-debug is an explicit short-probe mode",
    )
    parser.add_argument(
        "--full-debug", "--full-debug-telemetry", "--full-debug-mode",
        dest="full_debug", action="store_true",
        help="retain historical self-contained rows and dense debug payloads (large; opt-in)",
    )
    parser.add_argument(
        "--parallel-frame-stages",
        action="store_true",
        help=(
            "overlap causal-state-free sensory preparation with fitted vision; "
            "strict free-behavior minimal telemetry only"
        ),
    )
    parser.add_argument("--visual-calibration", "--calibration", type=Path, default=None,
                        help="explicit visual_drive_calibration module/path")
    parser.add_argument("--visual-calibration-config", "--calibration-config", type=Path, default=None,
                        help="JSON object passed to the visual calibration")
    parser.add_argument("--enable-visual-tonic-state-inputs", "--enable-tonic-state-inputs", action="store_true",
                        help="explicitly opt into declared tonic visual-state hypotheses")
    parser.add_argument("--neural-csr-path", type=Path, default=None,
                        help="explicit MaleCNS CSR directory")
    parser.add_argument(
        "--flight-state-drive", "--flight-state-drive-config", "--state-drive",
        dest="flight_state_drive", type=Path, default=None,
        help="explicit hand-built flight-state population-drive JSON (opt-in)",
    )
    parser.add_argument(
        "--flight-body-stabilizer", "--body-stabilizer",
        dest="flight_body_stabilizer", type=Path, default=None,
        help="explicit bounded hand-built pitch/roll/collective stabilizer JSON (opt-in)",
    )
    parser.add_argument(
        "--dynamic-stimulus", "--stimulus-channel", dest="dynamic_stimulus",
        default=None,
        help="strict escape dynamic-stimulus provider module/path",
    )
    parser.add_argument(
        "--jump-channel", dest="jump_channel", default=None,
        help="strict escape jump-channel provider module/path",
    )
    parser.add_argument(
        "--posture-channel", dest="posture_channel", default=None,
        help="strict escape posture-channel provider module/path",
    )
    parser.add_argument(
        "--stimulus-config", type=Path, default=None,
        help="physical VirtualWorld stimulus JSON (strict escape path)",
    )
    parser.add_argument(
        "--stimulus-name", default=None,
        help="selected physical stimulus name (recorded only; never a control cue)",
    )
    parser.add_argument(
        "--stimulus-condition", default=None,
        help="selected physical stimulus condition (setup/telemetry only; never a control cue)",
    )
    parser.add_argument(
        "--allow-missing-escape-channels", dest="escape_channels_required",
        action="store_false",
        help="compatibility probe only; strict mode still refuses to control without channels",
    )
    parser.add_argument(
        "--palm-speed-mps", type=float, default=None,
        help="physical palm stimulus speed passed to the escape provider (0.5 core; 10 pressure test)",
    )
    parser.add_argument(
        "--impact-time-s", type=float, default=None,
        help="physical palm impact time passed to the escape provider",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    requested_mode = _canonical_execution_mode(args)
    config_path = args.config
    if (
        requested_mode == FREE_BEHAVIOR_POLICY_NAME
        and config_path == DEFAULT_CONFIG
        and DEFAULT_FREE_ESCAPE_CONFIG.is_file()
    ):
        config_path = DEFAULT_FREE_ESCAPE_CONFIG
    requested = tuple(args.scenarios or SCENARIOS)
    all_configured = "all" in requested
    scenarios = tuple(name for name in requested if name != "all") or SCENARIOS
    options = RunOptions(
        config=config_path,
        out=args.out,
        scene_manifest=args.scene_manifest,
        courtship_target_trajectory=args.courtship_target_trajectory,
        courtship_target_condition=args.courtship_target_condition,
        courtship_movie=args.courtship_movie,
        courtship_movie_eye_mode=args.courtship_movie_eye_mode,
        courtship_movie_stereo_layout=args.courtship_movie_stereo_layout,
        scenarios=scenarios,
        mode=args.mode,
        strict=args.strict,
        free_behavior=args.free_behavior,
        escape_experiment=args.escape_experiment,
        neural=args.neural,
        vision=args.vision,
        body=args.body,
        renderer_backend=args.renderer_backend,
        seed=args.seed,
        width=args.width,
        height=args.height,
        columns=args.columns,
        t4t5=args.t4t5,
        neural_device=args.neural_device,
        neural_backend=args.neural_backend,
        neural_cuda_graph=args.neural_cuda_graph,
        neural_gain=args.neural_gain,
        vision_device=args.vision_device,
        max_vram_gib=args.max_vram_gib,
        mechanosensory_ablations=tuple(args.mechanosensory_ablations or ()),
        food_config=args.food_config,
        food_cue_ablations=tuple(args.food_cue_ablations or ()),
        neural_intervention_config=args.neural_intervention_config,
        grooming_loop_config=args.grooming_loop_config,
        grooming_dust_count=args.grooming_dust_count,
        grooming_sensor_off=args.grooming_sensor_off,
        grooming_foreleg_lock=args.grooming_foreleg_lock,
        duration_override_s=args.duration_override_s,
        max_steps=args.max_steps,
        frame_count=args.frame_count,
        full_neural_telemetry=args.full_neural_telemetry,
        telemetry_mode=args.telemetry_mode,
        full_debug=args.full_debug,
        parallel_frame_stages=args.parallel_frame_stages,
        visual_calibration=args.visual_calibration,
        visual_calibration_config=args.visual_calibration_config,
        visual_tonic_state_inputs=args.enable_visual_tonic_state_inputs,
        neural_csr_path=args.neural_csr_path,
        flight_state_drive=args.flight_state_drive,
        flight_body_stabilizer=args.flight_body_stabilizer,
        dynamic_stimulus=args.dynamic_stimulus,
        jump_channel=args.jump_channel,
        posture_channel=args.posture_channel,
        escape_channels_required=args.escape_channels_required,
        stimulus_config=args.stimulus_config,
        stimulus_name=args.stimulus_name,
        stimulus_condition=args.stimulus_condition,
        palm_speed_mps=args.palm_speed_mps,
        impact_time_s=args.impact_time_s,
        all_configured_scenarios=all_configured,
    )
    try:
        runner = DigitalFlyRunner(options)
        payload = runner.run()
        paths = runner.write_outputs(payload)
    except (OSError, ValueError, LauncherError, KeyError, RuntimeError) as exc:
        error = {
            "schema_version": "digital-fly-run-1",
            "status": "ERROR",
            "error": str(exc),
            "requested_backends": {"neural": args.neural, "vision": args.vision, "body": args.body},
            "requested_renderer_backend": args.renderer_backend,
            "execution_policy": requested_mode,
            "fail_closed": True,
        }
        print(json.dumps(error, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "OK", "outputs": paths,
        "scenarios": ["free-behavior"] if requested_mode == FREE_BEHAVIOR_POLICY_NAME else list(scenarios),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BackendSelectionError", "CalibrationError", "RendererBackendError",
    "DigitalFlyRunner", "LauncherError",
    "EscapeChannelAdapter", "EscapeChannelSet", "MalecnsNeuralAdapter", "RunOptions",
    "DEFAULT_FREE_ESCAPE_CONFIG", "FREE_BEHAVIOR_POLICY_NAME",
    "SCENARIO_ALIASES", "SCENARIOS", "VisualDriveCalibration", "main", "parse_args",
]
