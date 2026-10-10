#!/usr/bin/env python3
"""Deterministic, multi-rate runtime for the MaleCNS digital twin.

The runtime is deliberately a small orchestration layer.  It does not import
``torch`` (or any CUDA binding) at module import time.  A CUDA backend is
selected only after :mod:`malecns_gate` has reported an open gate.  On a host
without a usable GPU, ``auto`` selects dependency-free CPU components and can
therefore be exercised with ``--dry-run``.

Clock domains
-------------
``physics`` is 10 kHz (0.1 ms), ``cns`` is 2 kHz (0.5 ms), ``flyvis`` is 200
Hz (5 ms), and control/telemetry are intentionally slower.  The scheduler is
driven by the physics grid and emits domain events when their nominal time is
reached.  A domain never silently changes the timestamp of a signal: a
``Timestamp`` records both its domain tick and the simulation time.

The model components are injectable.  The defaults are deterministic stubs,
which makes this module useful for contract and timing tests without loading
the 176k-neuron graph.  Existing GPU components can be supplied through the
``world``, ``sensor``, ``flyvis``, ``neural`` and ``motor`` arguments; the
runtime only requires the small duck-typed methods documented by
``twin_contracts.py``.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Callable, Optional

# ``twin_contracts`` is dependency-light and does not import a model runtime.
# Keep this import at module scope so callers can use the contracts directly;
# in particular, do not add torch/numpy imports here.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from twin_contracts import (  # noqa: E402
    BodyState,
    ContractError,
    MotorCommand,
    MultiRateClock,
    NeuralOutput,
    SensoryFrame,
    SourceLabel,
    StepDiagnostics,
    Timestamp,
    VisualObservation,
    WorldState,
)


ROOT = _SCRIPT_DIR.parent
PHYSICS_DT_S = 0.0001
CNS_DT_S = 0.0005
FLYVIS_DT_S = 0.005
PHYSICS_HZ = 1.0 / PHYSICS_DT_S
CNS_HZ = 1.0 / CNS_DT_S
FLYVIS_HZ = 1.0 / FLYVIS_DT_S
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_TELEMETRY_HZ = 20.0
SCHEMA_VERSION = "malecns-twin-runtime-1"


class RuntimeErrorBase(RuntimeError):
    """Base class for runtime setup and execution errors."""


class GateClosedError(RuntimeErrorBase):
    """Raised before any CUDA/Torch import when the safety gate is closed."""

    def __init__(self, blockers: Sequence[str], *, requested: str = "cuda") -> None:
        self.blockers = tuple(str(x) for x in blockers)
        message = "CUDA gate closed; refusing to import or initialise Torch"
        if self.blockers:
            message += ":\n  - " + "\n  - ".join(self.blockers)
        super().__init__(message)


class BackendUnavailableError(RuntimeErrorBase):
    """Raised when an explicitly requested backend cannot be used safely."""


@dataclass(frozen=True)
class BackendCapability:
    """Read-only capability information gathered without importing backends."""

    name: str
    installed: bool
    usable: bool
    validated: bool = False
    module: Optional[str] = None
    version: Optional[str] = None
    note: str = ""

    @property
    def available(self) -> bool:
        """Compatibility alias used by several adapter registries."""
        return bool(self.installed and self.usable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "installed": self.installed,
            "usable": self.usable,
            "available": self.available,
            "validated": self.validated,
            "module": self.module,
            "version": self.version,
            "note": self.note,
        }


@dataclass(frozen=True)
class GateSnapshot:
    """A serialisable copy of the MaleCNS CUDA gate state."""

    checked: bool
    open: bool
    blockers: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.open

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "open": self.open,
            "blockers": list(self.blockers),
            "facts": _json_safe(self.facts),
        }


def _json_safe(value: Any) -> Any:
    """Convert common runtime values to JSON without importing numpy."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, type):
        return value.__name__
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return repr(value)


def _module_capability(name: str, *, validated: bool = False,
                       note: str = "") -> BackendCapability:
    """Inspect a module using import metadata only; never import it."""

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    installed = spec is not None
    version: Optional[str] = None
    if installed:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        except Exception:
            version = None
    return BackendCapability(
        name=name,
        installed=installed,
        usable=installed,
        validated=validated,
        module=name if installed else None,
        version=version,
        note=note,
    )


def _import_gate() -> Any:
    """Import the read-only safety gate, without touching a model backend."""

    try:
        return importlib.import_module("malecns_gate")
    except ModuleNotFoundError:
        # This fallback is useful when the script is loaded by an external
        # harness that does not put its directory on sys.path.
        path = _SCRIPT_DIR / "malecns_gate.py"
        spec = importlib.util.spec_from_file_location("malecns_gate", path)
        if spec is None or spec.loader is None:
            raise RuntimeErrorBase(f"cannot load safety gate from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("malecns_gate", module)
        spec.loader.exec_module(module)
        return module


def inspect_gate() -> GateSnapshot:
    """Read the safety gate state; no Torch/CUDA import occurs here."""

    gate = _import_gate()
    try:
        facts = gate.facts()
    except Exception as exc:  # pragma: no cover - defensive on unusual hosts
        facts = {"error": f"gate facts unavailable: {type(exc).__name__}: {exc}"}
    try:
        blockers = tuple(str(x) for x in gate.cuda_blockers())
    except Exception as exc:  # pragma: no cover - defensive on unusual hosts
        blockers = (f"gate check failed: {type(exc).__name__}: {exc}",)
    return GateSnapshot(True, not blockers, blockers, facts)


def discover_capabilities(*, include_gate: bool = True) -> dict[str, BackendCapability | GateSnapshot]:
    """Discover optional runtimes without importing them.

    ``find_spec`` and package metadata are intentionally used instead of
    importing Torch, Triton, FlyVis, MuJoCo or Flybody.  This function is safe
    to call before the gate and is used by :mod:`twin_preflight`.
    """

    caps: dict[str, BackendCapability | GateSnapshot] = {
        "cpu": BackendCapability(
            "cpu", True, True, validated=True,
            note="host Python execution path",
        ),
        "cpu_stub": BackendCapability(
            "cpu_stub", True, True, validated=True,
            note="dependency-free deterministic dry-run backend",
        ),
        "numpy": _module_capability("numpy", note="optional array adapter"),
        "torch": _module_capability("torch", note="import only after CUDA gate"),
        "triton": _module_capability("triton", note="import only after CUDA gate"),
        "flyvis": _module_capability("flyvis", note="optional FlyVis adapter"),
        "mujoco": _module_capability("mujoco", note="optional body adapter"),
        "flybody": _module_capability("flybody", note="optional body adapter"),
    }
    if include_gate:
        gate = inspect_gate()
        gpu_seen = _gpu_query_looks_usable(gate.facts)
        caps["cuda"] = BackendCapability(
            "cuda", gpu_seen, gpu_seen and gate.open and caps["torch"].installed,
            validated=False, module="nvidia-smi" if gpu_seen else None,
            note="usable only when nvidia-smi, Torch CUDA and malecns_gate all pass",
        )
    else:
        gpu_seen = shutil.which("nvidia-smi") is not None
        caps["cuda"] = BackendCapability(
            "cuda", gpu_seen, False, validated=False,
            module="nvidia-smi" if gpu_seen else None,
            note="gate not checked in this capability-only view",
        )
    if include_gate:
        caps["gate"] = gate
    return caps


def _gpu_query_looks_usable(facts: Mapping[str, Any]) -> bool:
    gpu = facts.get("gpu", {}) if isinstance(facts, Mapping) else {}
    text = str(gpu.get("nvidia-smi", "")) if isinstance(gpu, Mapping) else str(gpu)
    return bool(text) and not text.startswith(("missing", "error", "exit "))


@dataclass(frozen=True)
class RatePlan:
    """Named simulation rates and their canonical periods."""

    physics_hz: float = PHYSICS_HZ
    cns_hz: float = CNS_HZ
    flyvis_hz: float = FLYVIS_HZ
    control_hz: float = DEFAULT_CONTROL_HZ
    telemetry_hz: float = DEFAULT_TELEMETRY_HZ

    def __post_init__(self) -> None:
        values = self.as_dict()
        for name, rate in values.items():
            if isinstance(rate, bool) or not math.isfinite(float(rate)) or float(rate) <= 0:
                raise ValueError(f"{name} must be a finite positive rate")
        if self.physics_hz + 1e-9 < max(self.cns_hz, self.flyvis_hz, self.control_hz, self.telemetry_hz):
            raise ValueError("physics_hz must be the highest runtime rate")
        if self.control_hz > self.cns_hz + 1e-9:
            raise ValueError("control_hz must not exceed cns_hz")
        if self.telemetry_hz > self.control_hz + 1e-9:
            raise ValueError("telemetry_hz must not exceed control_hz")

    def as_dict(self) -> dict[str, float]:
        return {
            "physics": float(self.physics_hz),
            "cns": float(self.cns_hz),
            "flyvis": float(self.flyvis_hz),
            "control": float(self.control_hz),
            "telemetry": float(self.telemetry_hz),
        }

    @property
    def rates_hz(self) -> Mapping[str, float]:
        return self.as_dict()

    def period_s(self, domain: str) -> float:
        try:
            return 1.0 / self.as_dict()[domain]
        except KeyError as exc:
            raise ValueError(f"unknown runtime domain {domain!r}") from exc

    @property
    def physics_dt_s(self) -> float:
        return self.period_s("physics")

    @property
    def cns_dt_s(self) -> float:
        return self.period_s("cns")

    @property
    def flyvis_dt_s(self) -> float:
        return self.period_s("flyvis")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rates_hz": self.as_dict(),
            "period_s": {name: self.period_s(name) for name in self.as_dict()},
        }

    @classmethod
    def from_mapping(cls, rates: Mapping[str, float]) -> "RatePlan":
        values = dict(rates)
        def pick(name: str, default: float) -> float:
            return values.get(name, values.get(f"{name}_hz", default))
        return cls(
            pick("physics", PHYSICS_HZ), pick("cns", CNS_HZ),
            pick("flyvis", FLYVIS_HZ), pick("control", DEFAULT_CONTROL_HZ),
            pick("telemetry", DEFAULT_TELEMETRY_HZ),
        )


@dataclass
class RuntimeConfig:
    """Runtime setup, timing policy and safety choices."""

    duration_s: float = 0.01
    seed: int = 1729
    device: str = "auto"
    backend: str = "auto"
    dry_run: bool = False
    physics_hz: float = PHYSICS_HZ
    cns_hz: float = CNS_HZ
    flyvis_hz: float = FLYVIS_HZ
    control_hz: float = DEFAULT_CONTROL_HZ
    telemetry_hz: float = DEFAULT_TELEMETRY_HZ
    rate_plan: Optional[RatePlan] = None
    # ``rates_hz`` is a convenience alias for callers that already keep a
    # mapping (the canonical serialised form remains ``rate_plan``).
    rates_hz: Optional[Mapping[str, float]] = None
    deadlines_s: Mapping[str, float] = field(default_factory=dict)
    deadline_s: Optional[float] = None
    drop_late: bool = False
    max_lag_s: Optional[float] = None
    realtime: bool = False
    telemetry_limit: int = 10000

    def __post_init__(self) -> None:
        self.duration_s = float(self.duration_s)
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("duration_s must be finite and >= 0")
        self.seed = int(self.seed)
        self.device = str(self.device).lower()
        self.backend = str(self.backend).lower()
        if self.device not in {"auto", "cpu", "cuda", "gpu"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.device == "gpu":
            self.device = "cuda"
        if self.backend not in {"auto", "cpu", "stub", "cpu_stub", "torch", "triton", "cuda"}:
            raise ValueError("backend must be auto, cpu/stub, torch, or triton")
        if isinstance(self.rate_plan, Mapping):
            self.rate_plan = RatePlan.from_mapping(self.rate_plan)
        if self.rate_plan is None and self.rates_hz is not None:
            rates = {
                "physics": self.physics_hz, "cns": self.cns_hz,
                "flyvis": self.flyvis_hz, "control": self.control_hz,
                "telemetry": self.telemetry_hz,
            }
            supplied = dict(self.rates_hz)
            for domain in tuple(rates):
                if f"{domain}_hz" in supplied:
                    rates[domain] = supplied[f"{domain}_hz"]
                if domain in supplied:
                    rates[domain] = supplied[domain]
            self.rate_plan = RatePlan.from_mapping(rates)
        if self.rate_plan is None:
            self.rate_plan = RatePlan(
                self.physics_hz, self.cns_hz, self.flyvis_hz,
                self.control_hz, self.telemetry_hz,
            )
        else:
            self.physics_hz = self.rate_plan.physics_hz
            self.cns_hz = self.rate_plan.cns_hz
            self.flyvis_hz = self.rate_plan.flyvis_hz
            self.control_hz = self.rate_plan.control_hz
            self.telemetry_hz = self.rate_plan.telemetry_hz
        # Keep a resolved mapping available under the familiar MultiRateClock
        # attribute name, even when the caller supplied scalar fields.
        self.rates_hz = dict(self.rate_plan.as_dict())
        raw_deadlines = dict(self.deadlines_s)
        if self.deadline_s is not None:
            raw_deadlines.update({domain: self.deadline_s for domain in
                                  ("physics", "cns", "flyvis", "control", "telemetry")})
        clean_deadlines: dict[str, float] = {}
        for domain, value in raw_deadlines.items():
            domain = {"neural": "cns", "render": "flyvis", "body": "physics"}.get(str(domain), str(domain))
            if domain not in {"physics", "cns", "flyvis", "control", "telemetry"}:
                raise ValueError(f"unknown runtime deadline domain {domain!r}")
            v = float(value)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"deadline for {domain!r} must be finite and >= 0")
            clean_deadlines[str(domain)] = v
        self.deadlines_s = clean_deadlines
        if self.deadline_s is not None:
            self.deadline_s = float(self.deadline_s)
        if self.max_lag_s is not None:
            self.max_lag_s = float(self.max_lag_s)
            if not math.isfinite(self.max_lag_s) or self.max_lag_s < 0:
                raise ValueError("max_lag_s must be finite and >= 0")
        self.telemetry_limit = int(self.telemetry_limit)
        if self.telemetry_limit < 1:
            raise ValueError("telemetry_limit must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "seed": self.seed,
            "device": self.device,
            "backend": self.backend,
            "dry_run": self.dry_run,
            "rates": self.rate_plan.to_dict() if self.rate_plan else {},
            "rates_hz": dict(self.rates_hz) if self.rates_hz is not None else None,
            "deadlines_s": dict(self.deadlines_s),
            "deadline_s": self.deadline_s,
            "drop_late": self.drop_late,
            "max_lag_s": self.max_lag_s,
            "realtime": self.realtime,
            "telemetry_limit": self.telemetry_limit,
        }

    @property
    def rates(self) -> Mapping[str, float]:
        return self.rate_plan.as_dict() if self.rate_plan else {}

    @property
    def physics_dt_s(self) -> float:
        return self.rate_plan.period_s("physics") if self.rate_plan else PHYSICS_DT_S

    @property
    def cns_dt_s(self) -> float:
        return self.rate_plan.period_s("cns") if self.rate_plan else CNS_DT_S

    @property
    def flyvis_dt_s(self) -> float:
        return self.rate_plan.period_s("flyvis") if self.rate_plan else FLYVIS_DT_S


def _resolve_execution(config: RuntimeConfig) -> tuple[str, str, GateSnapshot, dict[str, Any]]:
    """Resolve device/backend while preserving the fail-closed gate order."""

    gate = inspect_gate()
    requested_device = config.device
    requested_backend = config.backend
    gpu_backend = requested_backend in {"torch", "triton", "cuda"}

    if requested_device == "cuda":
        if not gate.open:
            raise GateClosedError(gate.blockers)
        device = "cuda"
    elif requested_device == "cpu":
        device = "cpu"
    elif config.dry_run or (requested_device == "auto" and requested_backend in {"cpu", "stub", "cpu_stub"}):
        # A dry-run is an explicit CPU contract test.  A caller that names a
        # CPU backend also should not unexpectedly load a GPU merely because
        # the safety gate happens to be open.
        device = "cpu"
    else:
        # auto is conservative: only choose CUDA when both the gate and the
        # read-only nvidia-smi query agree that a GPU is present.
        device = "cuda" if gate.open and _gpu_query_looks_usable(gate.facts) else "cpu"

    if device == "cpu":
        if gpu_backend and requested_device == "cpu":
            raise BackendUnavailableError(
                f"backend {requested_backend!r} requires CUDA but device=cpu"
            )
        effective_backend = "cpu_stub"
    elif requested_backend in {"auto", "cuda"}:
        effective_backend = "triton" if _module_capability("triton").installed else "torch"
    elif requested_backend in {"torch", "triton"}:
        effective_backend = requested_backend
    else:
        effective_backend = "cpu_stub"

    details = {
        "requested_device": requested_device,
        "requested_backend": requested_backend,
        "effective_device": device,
        "effective_backend": effective_backend,
        "fallback": device == "cpu" and (requested_device == "auto" or gpu_backend),
        "fallback_reason": (
            list(gate.blockers) if device == "cpu" and requested_device == "auto" and gate.blockers
            else ("dry_run" if config.dry_run and device == "cpu" and
                  (requested_device == "auto" or gpu_backend) else None)
        ),
    }
    return device, effective_backend, gate, details


def _import_torch_after_gate(gate: GateSnapshot) -> Any:
    """Import Torch only after a successful gate check."""

    if not gate.open:
        raise GateClosedError(gate.blockers)
    # Re-run the project's fail-closed assertion immediately before the first
    # backend import.  This closes the small TOCTOU window between discovery
    # and initialisation and keeps the ordering auditable in tests/logs.
    _import_gate().require_cuda()
    torch = importlib.import_module("torch")
    if not bool(torch.cuda.is_available()):
        raise BackendUnavailableError("CUDA backend selected but torch.cuda.is_available() is false")
    return torch


@dataclass(frozen=True)
class FlyVisDrive:
    timestamp: Timestamp
    drive: tuple[float, ...] = ()
    backend: str = "deterministic_stub"
    is_stub: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.to_dict(),
            "drive": list(self.drive),
            "backend": self.backend,
            "is_stub": self.is_stub,
        }


@dataclass
class RuntimeComponents:
    """Injected component bundle used by :class:`TwinRuntime`."""

    world: Any
    sensor: Any
    flyvis: Any
    neural: Any
    motor: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "backend": getattr(value, "backend_name", getattr(value, "name", None)),
                "is_stub": bool(getattr(value, "is_stub", False)),
            }
            for name, value in vars(self).items()
        }


class SimpleWorldModel:
    """Small deterministic point-mass world for CPU contract runs."""

    backend_name = "simple_cpu_world"
    is_stub = True

    def __init__(self) -> None:
        self._state: Optional[WorldState] = None

    def reset(self, state: Optional[WorldState] = None, *, seed: Optional[int] = None) -> WorldState:
        del seed
        if state is None:
            ts = Timestamp(0.0, 0, "physics", PHYSICS_HZ)
            state = WorldState(ts, BodyState(ts, position_m=(0.0, 0.0, 0.01)),
                               world_id="cpu_stub", geometry={"model": "point_mass"})
        self._state = _normalise_world(state, state.timestamp)
        return self._state

    def step(self, command: MotorCommand, timestamp: Timestamp) -> WorldState:
        if self._state is None:
            self.reset()
        assert self._state is not None
        old = _body_at(self._state.body, timestamp)
        dt = max(0.0, timestamp.t_s - self._state.timestamp.t_s)
        # A stable, bounded reduced body.  This is intentionally not presented
        # as aerodynamics; it only makes the timing path observable.
        yaw_rate = float(command.yaw_rate_rad_s)
        pitch_rate = float(command.pitch_rate_rad_s)
        roll_rate = float(command.roll_rate_rad_s)
        thrust = max(-1.0, min(1.0, float(command.thrust_mps2)))
        vx, vy, vz = old.velocity_mps
        vz = max(-0.5, min(0.5, vz + thrust * 0.4 * dt))
        velocity = (vx * (1.0 - min(1.0, dt * 2.0)),
                    vy * (1.0 - min(1.0, dt * 2.0)), vz)
        x, y, z = old.position_m
        position = (x + velocity[0] * dt, y + velocity[1] * dt,
                    max(0.0, z + velocity[2] * dt))
        r, p, yaw = old.orientation_rad
        orientation = (r + roll_rate * dt, p + pitch_rate * dt, yaw + yaw_rate * dt)
        body = BodyState(timestamp, position_m=position, orientation_rad=orientation,
                         velocity_mps=velocity,
                         angular_velocity_rad_s=(roll_rate, pitch_rate, yaw_rate),
                         source=SourceLabel.HANDBUILT)
        self._state = WorldState(timestamp, body, world_id=self._state.world_id,
                                 geometry=self._state.geometry, source=SourceLabel.HANDBUILT)
        return self._state


class DeterministicSensor:
    """Dependency-free sensor whose flow is a deterministic body projection."""

    backend_name = "deterministic_sensor"
    is_stub = True

    def observe(self, world: WorldState, timestamp: Timestamp) -> VisualObservation:
        body = world.body
        vx, vy, _ = body.velocity_mps
        roll, pitch, yaw = body.orientation_rad
        return VisualObservation(
            timestamp,
            pixels=[[0.5, 0.5], [0.5, 0.5]],
            optic_flow_rad_s=(vy + yaw, vx + pitch, roll),
            sensor_id="cpu_stub_eye",
            source=SourceLabel.HANDBUILT,
            metadata={"world_id": world.world_id, "backend": self.backend_name},
        )


class DeterministicFlyVis:
    """Signed, deterministic FlyVis stand-in used only when no adapter exists."""

    backend_name = "deterministic_stub"
    name = backend_name
    is_stub = True

    def __init__(self, *, seed: int = 1729) -> None:
        self.seed = int(seed)
        self.calls = 0

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.calls = 0

    def step(self, observation: VisualObservation, timestamp: Timestamp) -> FlyVisDrive:
        self.calls += 1
        flow = observation.optic_flow_rad_s
        # Keep the sign: a direction-selective response must not be converted
        # to an absolute luminance or hidden behind a ReLU.
        drive = (float(flow[0]), float(flow[1]), float(flow[2]),
                 float(flow[0] - flow[1]))
        return FlyVisDrive(timestamp, drive, self.backend_name, True)


class DeterministicNeural:
    """Tiny deterministic neural endpoint with the shared CNS contract."""

    backend_name = "deterministic_cns_stub"
    is_stub = True

    def __init__(self, *, seed: int = 1729, n_outputs: int = 4) -> None:
        if n_outputs < 1:
            raise ValueError("n_outputs must be positive")
        self.seed = int(seed)
        self.n_outputs = int(n_outputs)
        self.calls = 0
        self._rng = random.Random(self.seed)

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.calls = 0
        self._rng = random.Random(self.seed)

    def step(self, sensory: SensoryFrame, timestamp: Timestamp) -> NeuralOutput:
        self.calls += 1
        flow = sensory.visual.optic_flow_rad_s if sensory.visual is not None else (0.0, 0.0, 0.0)
        drive = sum(abs(float(x)) for x in flow)
        # A seed-dependent but deterministic phase avoids every output being
        # identical while preserving reproducibility across processes.
        phase = (self.seed % 97) / 97.0 + self.calls * 0.03125
        rates = [max(0.0, drive * (i + 1) * 10.0 + (phase % 1.0) * 0.01)
                 for i in range(self.n_outputs)]
        spikes = [1 if rate >= 0.05 else 0 for rate in rates]
        return NeuralOutput(timestamp, rates_hz=rates, spikes=spikes,
                            named_outputs={"backend": self.backend_name,
                                           "call_index": self.calls},
                            source=SourceLabel.HANDBUILT)


class DeterministicMotor:
    """Maps stub neural rates to bounded body commands."""

    backend_name = "deterministic_motor_stub"
    is_stub = True

    def decode(self, neural: NeuralOutput, timestamp: Timestamp) -> MotorCommand:
        rates = _numeric_sequence(neural.rates_hz)
        left = rates[0] if rates else 0.0
        right = rates[1] if len(rates) > 1 else left
        yaw = max(-1.0, min(1.0, (right - left) * 0.01))
        thrust = max(-1.0, min(1.0, (left + right) * 0.005))
        return MotorCommand(timestamp, yaw_rate_rad_s=yaw,
                            thrust_mps2=thrust, source=SourceLabel.HANDBUILT,
                            metadata={"backend": self.backend_name})


class TorchNeuralStub:
    """Minimal Torch endpoint for gate-open smoke tests.

    This is not the validated MaleCNS graph.  It exists so an injected runtime
    can verify that Torch is loaded *after* the gate and that the same contract
    works on a GPU.  Full LIF execution remains supplied by a caller through
    the injectable ``neural`` component.
    """

    backend_name = "torch_contract_stub"
    is_stub = True

    def __init__(self, torch_module: Any, *, device: str = "cuda") -> None:
        self.torch = torch_module
        self.device = device
        self.calls = 0

    def step(self, sensory: SensoryFrame, timestamp: Timestamp) -> NeuralOutput:
        self.calls += 1
        values = sensory.visual.optic_flow_rad_s if sensory.visual is not None else (0.0, 0.0, 0.0)
        tensor = self.torch.tensor(values, dtype=self.torch.float32, device=self.device)
        magnitude = float(self.torch.linalg.vector_norm(tensor).detach().cpu().item())
        rates = [magnitude * (i + 1) for i in range(4)]
        return NeuralOutput(timestamp, rates_hz=rates,
                            spikes=[int(rate > 0.1) for rate in rates],
                            named_outputs={"backend": self.backend_name},
                            source=SourceLabel.HANDBUILT)


def _default_components(seed: int, effective_device: str, effective_backend: str,
                        gate: GateSnapshot) -> RuntimeComponents:
    world = SimpleWorldModel()
    sensor = DeterministicSensor()
    flyvis = DeterministicFlyVis(seed=seed)
    if effective_device == "cuda":
        # The import is below the successful gate check and nowhere earlier.
        torch = _import_torch_after_gate(gate)
        neural: Any = TorchNeuralStub(torch, device="cuda")
    else:
        neural = DeterministicNeural(seed=seed)
    motor = DeterministicMotor()
    return RuntimeComponents(world, sensor, flyvis, neural, motor)


def _get(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _numeric_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return []
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            value = value.tolist()
        except Exception:
            return []
    if isinstance(value, Sequence):
        result: list[float] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                result.extend(_numeric_sequence(item))
            else:
                try:
                    number = float(item)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    result.append(number)
        return result
    try:
        number = float(value)
    except (TypeError, ValueError):
        return []
    return [number] if math.isfinite(number) else []


def _vec3(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if value is None:
        value = default
    values = _numeric_sequence(value)
    if len(values) != 3:
        return tuple(float(x) for x in default)  # type: ignore[return-value]
    return tuple(values)  # type: ignore[return-value]


def _body_at(body: Any, timestamp: Timestamp) -> BodyState:
    if isinstance(body, BodyState):
        return BodyState(timestamp, body.position_m, body.orientation_rad,
                         body.velocity_mps, body.angular_velocity_rad_s, body.source)
    orientation = _get(body, "orientation_rad", "attitude_rpy_rad", "rpy_rad", default=None)
    if orientation is None:
        orientation = (_get(body, "roll_rad", "roll", default=0.0),
                       _get(body, "pitch_rad", "pitch", default=0.0),
                       _get(body, "yaw_rad", "yaw", "heading", default=0.0))
    source = _get(body, "source", default=SourceLabel.HANDBUILT)
    try:
        return BodyState(
            timestamp,
            position_m=_vec3(_get(body, "position_m", "position", default=None)),
            orientation_rad=_vec3(orientation),
            velocity_mps=_vec3(_get(body, "velocity_mps", "velocity", default=None)),
            angular_velocity_rad_s=_vec3(_get(body, "angular_velocity_rad_s", "angular_velocity", default=None)),
            source=source,
        )
    except ContractError:
        return BodyState(timestamp, source=SourceLabel.HANDBUILT)


def _normalise_world(value: Any, timestamp: Timestamp) -> WorldState:
    if isinstance(value, WorldState):
        body = _body_at(value.body, timestamp)
        return WorldState(timestamp, body, value.world_id, value.geometry, value.source)
    body_value = _get(value, "body", "state", default=value)
    body = _body_at(body_value, timestamp)
    world_id = str(_get(value, "world_id", "id", default="injected_world"))
    geometry = _get(value, "geometry", default={})
    if not isinstance(geometry, Mapping):
        geometry = {"value": _json_safe(geometry)}
    return WorldState(timestamp, body, world_id=world_id, geometry=geometry,
                      source=SourceLabel.HANDBUILT)


def _normalise_observation(value: Any, timestamp: Timestamp,
                           *, prior: Optional[VisualObservation] = None) -> VisualObservation:
    if isinstance(value, VisualObservation):
        metadata = dict(value.metadata)
        if value.timestamp != timestamp:
            metadata.setdefault("source_timestamp_s", value.timestamp.t_s)
        return VisualObservation(timestamp, value.pixels, value.optic_flow_rad_s,
                                 value.sensor_id, value.source, metadata)
    if value is None:
        if prior is not None:
            return VisualObservation(timestamp, prior.pixels, prior.optic_flow_rad_s,
                                     prior.sensor_id, prior.source,
                                     {**dict(prior.metadata), "held": True,
                                      "source_timestamp_s": prior.timestamp.t_s})
        return VisualObservation(timestamp, pixels=None)
    if isinstance(value, Mapping):
        pixels = value.get("pixels", value.get("frame", value.get("image")))
        flow = value.get("optic_flow_rad_s", value.get("optic_flow", value.get("flow", (0, 0, 0))))
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {"metadata": _json_safe(metadata)}
        return VisualObservation(timestamp, pixels=pixels, optic_flow_rad_s=_vec3(flow),
                                 sensor_id=str(value.get("sensor_id", "injected_sensor")),
                                 source=value.get("source", SourceLabel.HANDBUILT),
                                 metadata=metadata)
    return VisualObservation(timestamp, pixels=value, metadata={"raw_sensor": True})


def _normalise_flyvis(value: Any, timestamp: Timestamp,
                      observation: VisualObservation) -> tuple[VisualObservation, FlyVisDrive]:
    if isinstance(value, FlyVisDrive):
        drive = value
        flow = _vec3(drive.drive)
        obs = VisualObservation(timestamp, observation.pixels, flow,
                                observation.sensor_id, SourceLabel.HANDBUILT,
                                {**dict(observation.metadata), "flyvis_backend": drive.backend,
                                 "flyvis_is_stub": drive.is_stub})
        return obs, drive
    if isinstance(value, VisualObservation):
        obs = _normalise_observation(value, timestamp, prior=observation)
        return obs, FlyVisDrive(timestamp, tuple(obs.optic_flow_rad_s),
                                "injected_visual", False)
    drive_value = _get(value, "drive", "t4t5_drive", "response", default=None)
    backend = str(_get(value, "backend", "name", default="injected_flyvis"))
    is_stub = bool(_get(value, "is_stub", default=False))
    if drive_value is None and isinstance(value, Mapping):
        drive_value = value.get("optic_flow_rad_s", value.get("flow"))
    if drive_value is None and value is not None and not isinstance(value, Mapping):
        # A lightweight adapter may return its signed T4/T5 vector directly.
        drive_value = value
    if drive_value is None:
        drive_value = observation.optic_flow_rad_s
    drive = tuple(_numeric_sequence(drive_value))
    flow = _vec3(drive)
    obs = VisualObservation(timestamp, observation.pixels, flow,
                            observation.sensor_id, SourceLabel.HANDBUILT,
                            {**dict(observation.metadata), "flyvis_backend": backend,
                             "flyvis_is_stub": is_stub})
    return obs, FlyVisDrive(timestamp, drive, backend, is_stub)


def _normalise_neural(value: Any, timestamp: Timestamp) -> NeuralOutput:
    if isinstance(value, NeuralOutput):
        rates = value.rates_hz
        spikes = value.spikes
        if value.timestamp != timestamp:
            return NeuralOutput(timestamp, rates, spikes, value.named_outputs, value.source)
        return value
    if isinstance(value, Mapping):
        rates = value.get("rates_hz", value.get("rates", value.get("output", [])))
        spikes = value.get("spikes")
        if spikes is None:
            seq = _numeric_sequence(rates)
            spikes = [int(x > 0) for x in seq]
        return NeuralOutput(timestamp, rates_hz=rates, spikes=spikes,
                            named_outputs=value.get("named_outputs", {}),
                            source=value.get("source", SourceLabel.HANDBUILT))
    seq = _numeric_sequence(value)
    return NeuralOutput(timestamp, rates_hz=seq, spikes=[int(x > 0) for x in seq],
                        source=SourceLabel.HANDBUILT)


def _normalise_motor(value: Any, timestamp: Timestamp) -> MotorCommand:
    if isinstance(value, MotorCommand):
        if value.timestamp == timestamp:
            return value
        return MotorCommand(timestamp, value.yaw_rate_rad_s, value.pitch_rate_rad_s,
                            value.roll_rate_rad_s, value.thrust_mps2, value.source,
                            value.saturated, value.metadata)
    if isinstance(value, Mapping):
        return MotorCommand(
            timestamp,
            yaw_rate_rad_s=value.get("yaw_rate_rad_s", value.get("yaw", value.get("turn", 0.0))),
            pitch_rate_rad_s=value.get("pitch_rate_rad_s", value.get("pitch", 0.0)),
            roll_rate_rad_s=value.get("roll_rate_rad_s", value.get("roll", 0.0)),
            thrust_mps2=value.get("thrust_mps2", value.get("thrust", value.get("power", 0.0))),
            source=value.get("source", SourceLabel.HANDBUILT),
            saturated=bool(value.get("saturated", False)),
            metadata=value.get("metadata", {}),
        )
    return MotorCommand(timestamp, source=SourceLabel.HANDBUILT)


def _invoke_method(component: Any, names: Sequence[str], payload: Any,
                   timestamp: Timestamp, *, dt_s: Optional[float] = None) -> Any:
    """Call a component with either ``(payload, timestamp)`` or ``(payload)``.

    Signature inspection avoids retrying a method after a side effect if a
    component uses ``dt_s`` instead of a timestamp.
    """

    method: Optional[Callable[..., Any]] = None
    for name in names:
        candidate = getattr(component, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None and callable(component):
        method = component
    if method is None:
        raise TypeError(f"component {type(component).__name__} has none of {tuple(names)}")
    try:
        params = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        params = []
    keyword_names = {p.name.lower(): p for p in params
                     if p.kind == p.KEYWORD_ONLY}
    if "timestamp" in keyword_names:
        return method(payload, timestamp=timestamp)
    if "timestamp_s" in keyword_names:
        return method(payload, timestamp_s=timestamp.t_s)
    if "dt_s" in keyword_names and dt_s is not None:
        return method(payload, dt_s=dt_s)
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    if has_varargs or len(positional) >= 2:
        second = positional[1].name.lower() if len(positional) >= 2 else "timestamp"
        if "dt" in second and dt_s is not None:
            return method(payload, dt_s)
        if second in {"timestamp_s", "time_s", "t_s"}:
            return method(payload, timestamp.t_s)
        return method(payload, timestamp)
    if len(positional) == 1:
        return method(payload)
    return method()


def _reset_component(component: Any, *, seed: int, state: Any = None) -> Any:
    reset = getattr(component, "reset", None)
    if not callable(reset):
        return state
    try:
        params = inspect.signature(reset).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {}
    if "seed" in params:
        kwargs["seed"] = seed
    if state is not None and params.get("state") is not None and \
            params["state"].kind == inspect.Parameter.KEYWORD_ONLY:
        kwargs["state"] = state
        return reset(**kwargs)
    if state is None:
        return reset(**kwargs)
    positional = [p for p in params.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if positional and positional[0].name.lower() in {"seed", "random_seed"}:
        return reset(seed, **{k: v for k, v in kwargs.items() if k != "seed"})
    if positional:
        return reset(state, **kwargs)
    return reset(**kwargs)


@dataclass
class DomainMetrics:
    """Timing counters retained per clock domain."""

    attempted: int = 0
    completed: int = 0
    dropped: int = 0
    deadline_misses: int = 0
    wall_time_s: float = 0.0
    max_wall_s: float = 0.0
    total_lag_s: float = 0.0
    max_lag_s: float = 0.0
    errors: int = 0

    def observe(self, wall_s: float, lag_s: float, *, completed: bool,
                dropped: bool, deadline_miss: bool, error: bool = False) -> None:
        self.attempted += 1
        self.wall_time_s += max(0.0, wall_s)
        self.max_wall_s = max(self.max_wall_s, wall_s)
        self.total_lag_s += max(0.0, lag_s)
        self.max_lag_s = max(self.max_lag_s, lag_s)
        if completed:
            self.completed += 1
        if dropped:
            self.dropped += 1
        if deadline_miss:
            self.deadline_misses += 1
        if error:
            self.errors += 1

    def to_dict(self, simulated_s: float) -> dict[str, Any]:
        wall = self.wall_time_s
        return {
            **{name: getattr(self, name) for name in (
                "attempted", "completed", "dropped", "deadline_misses", "errors",
                "wall_time_s", "max_wall_s", "total_lag_s", "max_lag_s",
            )},
            "mean_lag_s": self.total_lag_s / self.attempted if self.attempted else 0.0,
            "simulated_s": simulated_s,
            "rtf": simulated_s / wall if wall > 0 else None,
        }


@dataclass
class RuntimeResult:
    """Returned run data and the auditable manifest."""

    manifest: dict[str, Any]
    telemetry: list[dict[str, Any]]
    final_world: Optional[WorldState] = None
    latest_neural: Optional[NeuralOutput] = None
    latest_command: Optional[MotorCommand] = None

    def write_manifest(self, path: str | Path) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".json":
            target.mkdir(parents=True, exist_ok=True)
            target = target / "manifest.json"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_json_safe(self.manifest), indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        return target

    def write_telemetry(self, path: str | Path) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".jsonl":
            target.mkdir(parents=True, exist_ok=True)
            target = target / "telemetry.jsonl"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in self.telemetry:
                handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
        return target


class TwinRuntime:
    """Multi-rate digital-twin orchestrator with injectable components."""

    DOMAINS = ("physics", "cns", "flyvis", "control", "telemetry")

    def __init__(self, config: Optional[RuntimeConfig] = None, *,
                 components: Optional[RuntimeComponents | Mapping[str, Any]] = None,
                 world: Any = None, sensor: Any = None, flyvis: Any = None,
                 neural: Any = None, motor: Any = None) -> None:
        self.config = config or RuntimeConfig()
        self.effective_device, self.effective_backend, self.gate, self.execution = _resolve_execution(self.config)

        if isinstance(components, RuntimeComponents):
            injected = vars(components).copy()
        elif components is None:
            injected = {}
        elif isinstance(components, Mapping):
            injected = dict(components)
        else:
            raise TypeError("components must be RuntimeComponents or a mapping")
        for key, value in {"world": world, "sensor": sensor, "flyvis": flyvis,
                           "neural": neural, "motor": motor}.items():
            if value is not None:
                injected[key] = value
        defaults = _default_components(self.config.seed, self.effective_device,
                                       self.effective_backend, self.gate)
        self.components = RuntimeComponents(
            injected.get("world", defaults.world),
            injected.get("sensor", defaults.sensor),
            injected.get("flyvis", defaults.flyvis),
            injected.get("neural", defaults.neural),
            injected.get("motor", defaults.motor),
        )
        self.plan = self.config.rate_plan or RatePlan()
        self.clock = MultiRateClock(self._clock_rates())
        self.metrics = {domain: DomainMetrics() for domain in self.DOMAINS}
        self.telemetry: list[dict[str, Any]] = []
        self._world: Optional[WorldState] = None
        self._observation: Optional[VisualObservation] = None
        self._flyvis_drive: Optional[FlyVisDrive] = None
        self._neural: Optional[NeuralOutput] = None
        self._command = MotorCommand(Timestamp(0.0, 0, "control", self.plan.control_hz),
                                     source=SourceLabel.HANDBUILT)
        self._trace: list[dict[str, Any]] = []
        self._run_start = 0.0
        self._last_error: Optional[str] = None
        self._flyvis_primed = False
        self.last_result: Optional[RuntimeResult] = None

    @property
    def component_manifest(self) -> dict[str, Any]:
        return self.components.to_dict()

    def _clock_rates(self) -> dict[str, float]:
        """Clock map with contract vocabulary aliases (neural/render)."""
        rates = self.plan.as_dict()
        rates["neural"] = rates["cns"]
        rates["render"] = rates["telemetry"]
        return rates

    @property
    def manifest(self) -> Optional[dict[str, Any]]:
        return None if self.last_result is None else self.last_result.manifest

    def _deadline(self, domain: str) -> float:
        return float(self.config.deadlines_s.get(domain, self.plan.period_s(domain)))

    def _lag_limit(self, domain: str) -> float:
        if self.config.max_lag_s is not None:
            return self.config.max_lag_s
        return self._deadline(domain)

    def _event(self, domain: str, scheduled_s: float, callback: Callable[[], Any], *,
               droppable: bool = True) -> Any:
        """Run one domain callback and record wall/deadline/lag diagnostics."""

        metrics = self.metrics[domain]
        # Offline runs do not sleep, but still support real-time scheduling for
        # experiments that explicitly request it.
        if self.config.realtime:
            target = self._run_start + scheduled_s
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        pre_lag = 0.0
        if self.config.realtime:
            pre_lag = max(0.0, time.perf_counter() - self._run_start - scheduled_s)
            if droppable and self.config.drop_late and pre_lag > self._lag_limit(domain):
                metrics.observe(0.0, pre_lag, completed=False, dropped=True,
                                deadline_miss=True)
                return None
        started = time.perf_counter()
        result: Any = None
        error = False
        try:
            result = callback()
        except Exception as exc:
            error = True
            self._last_error = f"{domain}: {type(exc).__name__}: {exc}"
        wall_s = time.perf_counter() - started
        deadline_miss = wall_s > self._deadline(domain)
        lag_s = max(pre_lag, max(0.0, wall_s - self._deadline(domain)))
        dropped = bool(error or (droppable and self.config.drop_late and deadline_miss))
        metrics.observe(wall_s, lag_s, completed=not dropped and not error,
                        dropped=dropped, deadline_miss=deadline_miss, error=error)
        if dropped:
            return None
        return result

    def _reset(self) -> None:
        initial_ts = Timestamp(0.0, 0, "physics", self.plan.physics_hz)
        initial_world = WorldState(
            initial_ts,
            BodyState(initial_ts, position_m=(0.0, 0.0, 0.01)),
            world_id="cpu_stub",
            geometry={"model": "point_mass"},
        )
        reset_world = _reset_component(self.components.world, seed=self.config.seed,
                                       state=initial_world)
        if reset_world is None:
            reset_world = initial_world
        self._world = _normalise_world(reset_world, initial_ts)
        _reset_component(self.components.sensor, seed=self.config.seed)
        _reset_component(self.components.flyvis, seed=self.config.seed)
        _reset_component(self.components.neural, seed=self.config.seed)
        _reset_component(self.components.motor, seed=self.config.seed)
        self._observation = None
        self._flyvis_drive = None
        self._neural = None
        self._command = MotorCommand(Timestamp(0.0, 0, "control", self.plan.control_hz),
                                     source=SourceLabel.HANDBUILT)
        self._trace.clear()
        self._last_error = None
        self._flyvis_primed = False
        self.metrics = {domain: DomainMetrics() for domain in self.DOMAINS}
        self.telemetry.clear()
        self.clock = MultiRateClock(self._clock_rates())
        self.last_result = None

    def _prime_process_frame_adapter(self) -> None:
        """Prime frame-indexed FlyVis adapters at t=0 when they need it.

        ``StereoFlyVis`` intentionally treats its first frame as frame 0 and
        uses it to establish a luminance baseline.  The runtime's periodic
        events start at the first positive period (5 ms), so a one-time prime
        keeps that adapter's timestamp contract intact without pretending that
        the prime is an additional periodic FlyVis tick.
        """

        process_frame = getattr(self.components.flyvis, "process_frame", None)
        if not callable(process_frame) or self._world is None:
            return
        ts = self._timestamp("flyvis", 0, 0.0)
        world_at = WorldState(ts, _body_at(self._world.body, ts), self._world.world_id,
                              self._world.geometry, self._world.source)
        try:
            raw_obs = _invoke_method(self.components.sensor, ("observe", "step", "sample"), world_at, ts)
            observation = _normalise_observation(raw_obs, ts)
            frame = observation.pixels if observation.pixels is not None else observation
            try:
                raw_drive = process_frame(frame, timestamp_s=ts.t_s)
            except TypeError:
                raw_drive = process_frame(frame)
            self._observation, self._flyvis_drive = _normalise_flyvis(raw_drive, ts, observation)
            self._flyvis_primed = True
        except Exception as exc:
            # A custom process_frame adapter may not support a t=0 prime.  It
            # remains usable at its regular ticks; retain an auditable warning.
            self._last_error = f"flyvis prime: {type(exc).__name__}: {exc}"

    def _timestamp(self, domain: str, tick: int, scheduled_s: float) -> Timestamp:
        rate = self.plan.as_dict()[domain]
        if domain in self.clock.ticks:
            self.clock.ticks[domain] = max(self.clock.ticks[domain], int(tick))
        if domain == "cns" and "neural" in self.clock.ticks:
            self.clock.ticks["neural"] = max(self.clock.ticks["neural"], int(tick))
        if domain == "telemetry" and "render" in self.clock.ticks:
            self.clock.ticks["render"] = max(self.clock.ticks["render"], int(tick))
        return Timestamp(float(scheduled_s), int(tick), domain, rate)

    def _physics_step(self, tick: int, t_s: float, dt_s: float) -> None:
        assert self._world is not None
        ts = self._timestamp("physics", tick, t_s)
        out = self._event(
            "physics", t_s,
            lambda: _invoke_method(self.components.world, ("step", "advance", "update"),
                                    self._command, ts, dt_s=dt_s),
            droppable=False,
        )
        if out is not None:
            self._world = _normalise_world(out, ts)

    def _flyvis_step(self, tick: int, scheduled_s: float) -> None:
        assert self._world is not None
        ts = self._timestamp("flyvis", tick, scheduled_s)
        world_at = WorldState(ts, _body_at(self._world.body, ts), self._world.world_id,
                              self._world.geometry, self._world.source)

        def run_sensor_and_flyvis() -> Any:
            raw_obs = _invoke_method(self.components.sensor, ("observe", "step", "sample"), world_at, ts)
            observation = _normalise_observation(raw_obs, ts, prior=self._observation)
            # Existing StereoFlyVis uses process_frame(frame, timestamp_s=...).
            process_frame = getattr(self.components.flyvis, "process_frame", None)
            if callable(process_frame):
                frame = observation.pixels if observation.pixels is not None else observation
                try:
                    raw_drive = process_frame(frame, timestamp_s=ts.t_s)
                except TypeError:
                    raw_drive = process_frame(frame)
            else:
                raw_drive = _invoke_method(self.components.flyvis, ("step", "process", "infer"),
                                            observation, ts)
            return observation, raw_drive

        result = self._event("flyvis", scheduled_s, run_sensor_and_flyvis)
        if result is None:
            return
        observation, raw_drive = result
        observation = _normalise_observation(observation, ts, prior=self._observation)
        self._observation, self._flyvis_drive = _normalise_flyvis(raw_drive, ts, observation)

    def _cns_step(self, tick: int, scheduled_s: float) -> None:
        assert self._world is not None
        ts = self._timestamp("cns", tick, scheduled_s)
        visual = _normalise_observation(self._observation, ts, prior=self._observation)
        body = _body_at(self._world.body, ts)
        sensory = SensoryFrame(ts, visual=visual, body=body,
                               channels={"flyvis_drive": list(self._flyvis_drive.drive)
                                         if self._flyvis_drive else []},
                               source=SourceLabel.HANDBUILT)

        def run_neural() -> Any:
            return _invoke_method(self.components.neural, ("step", "infer", "forward"),
                                  sensory, ts)

        result = self._event("cns", scheduled_s, run_neural)
        if result is None:
            return
        self._neural = _normalise_neural(result, ts)
        rates = _numeric_sequence(self._neural.rates_hz)
        spikes = _numeric_sequence(self._neural.spikes)
        self._trace.append({"t_s": ts.t_s, "rates_hz": rates[:16], "spikes": spikes[:16]})

    def _control_step(self, tick: int, scheduled_s: float) -> None:
        ts = self._timestamp("control", tick, scheduled_s)
        neural = self._neural or NeuralOutput(ts, rates_hz=[], spikes=[], source=SourceLabel.HANDBUILT)
        neural_for_control = _normalise_neural(neural, ts)

        def run_motor() -> Any:
            return _invoke_method(self.components.motor, ("decode", "step", "infer"),
                                  neural_for_control, ts)

        result = self._event("control", scheduled_s, run_motor)
        if result is not None:
            self._command = _normalise_motor(result, ts)

    def _telemetry_step(self, tick: int, scheduled_s: float) -> None:
        ts = self._timestamp("telemetry", tick, scheduled_s)
        def build_record() -> None:
            wall_elapsed = max(0.0, time.perf_counter() - self._run_start)
            sim = max(scheduled_s, 1e-12)
            warnings: list[str] = []
            for domain, metrics in self.metrics.items():
                if metrics.deadline_misses:
                    warnings.append(f"{domain}:deadline_miss")
                if metrics.dropped:
                    warnings.append(f"{domain}:dropped")
            diag = StepDiagnostics(
                ts,
                step_index=self.metrics["physics"].attempted,
                neural_steps=self.metrics["cns"].completed,
                wall_time_s=wall_elapsed,
                real_time_factor=sim / wall_elapsed if wall_elapsed > 0 else None,
                counters={
                    "physics_steps": self.metrics["physics"].completed,
                    "cns_steps": self.metrics["cns"].completed,
                    "flyvis_steps": self.metrics["flyvis"].completed,
                    "control_steps": self.metrics["control"].completed,
                    "dropped_events": sum(x.dropped for x in self.metrics.values()),
                },
                warnings=tuple(warnings),
            )
            row = diag.to_dict()
            row["lag_s"] = {domain: m.max_lag_s for domain, m in self.metrics.items()}
            row["deadline_misses"] = {domain: m.deadline_misses for domain, m in self.metrics.items()}
            row["drops"] = {domain: m.dropped for domain, m in self.metrics.items()}
            if len(self.telemetry) < self.config.telemetry_limit:
                self.telemetry.append(row)

        self._event("telemetry", scheduled_s, build_record, droppable=False)

    def run(self) -> RuntimeResult:
        """Run the configured duration and return telemetry plus manifest."""

        self._reset()
        self._run_start = time.perf_counter()
        self._prime_process_frame_adapter()
        plan = self.plan
        physics_dt = plan.period_s("physics")
        # A final partial physics step is not silently invented: the grid is
        # rounded upward so a requested positive duration is never truncated,
        # and the resulting grid duration is reported separately.
        physics_steps = int(math.ceil(self.config.duration_s * plan.physics_hz - 1e-12))
        actual_duration = physics_steps * physics_dt
        domain_counts = {domain: 0 for domain in self.DOMAINS}
        previous_t = 0.0

        for p_tick in range(1, physics_steps + 1):
            t_s = p_tick * physics_dt
            dt_s = t_s - previous_t
            previous_t = t_s
            domain_counts["physics"] = p_tick
            self._physics_step(p_tick, t_s, dt_s)
            for domain in ("flyvis", "cns", "control", "telemetry"):
                rate = plan.as_dict()[domain]
                target_count = int(math.floor(t_s * rate + 1e-9))
                while domain_counts[domain] < target_count:
                    tick = domain_counts[domain] + 1
                    domain_counts[domain] = tick
                    scheduled = tick / rate
                    if domain == "flyvis":
                        self._flyvis_step(tick, scheduled)
                    elif domain == "cns":
                        self._cns_step(tick, scheduled)
                    elif domain == "control":
                        self._control_step(tick, scheduled)
                    else:
                        self._telemetry_step(tick, scheduled)

        wall_total = max(0.0, time.perf_counter() - self._run_start)
        domain_manifest: dict[str, Any] = {}
        for domain, metrics in self.metrics.items():
            simulated = actual_duration if domain == "physics" else domain_counts[domain] / plan.as_dict()[domain]
            domain_manifest[domain] = metrics.to_dict(simulated)
            domain_manifest[domain]["deadline_s"] = self._deadline(domain)
        trace_payload = json.dumps(_json_safe(self._trace), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(trace_payload.encode("utf-8")).hexdigest()
        warnings: list[str] = []
        if self._last_error:
            warnings.append(self._last_error)
        if self.execution.get("fallback"):
            warnings.append("GPU unavailable or not requested; CPU/stub backend selected")
        for domain, metrics in self.metrics.items():
            if metrics.deadline_misses:
                warnings.append(f"{domain}: {metrics.deadline_misses} deadline miss(es)")
            if metrics.dropped:
                warnings.append(f"{domain}: {metrics.dropped} dropped event(s)")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "dry-run" if self.config.dry_run else "run",
            "source_files": {
                "contracts": str(_SCRIPT_DIR / "twin_contracts.py"),
                "safety_gate": str(_SCRIPT_DIR / "malecns_gate.py"),
                "cns_reference": str(_SCRIPT_DIR / "lif_flyb.py"),
                "world_reference": str(_SCRIPT_DIR / "fly_world.py"),
            },
            "seed": self.config.seed,
            "deterministic_seed": self.config.seed,
            "config": self.config.to_dict(),
            "execution": self.execution,
            "device": self.execution["effective_device"],
            "backend": self.execution["effective_backend"],
            "gate": self.gate.to_dict(),
            "gate_passed": self.gate.open,
            "capabilities": {
                key: value.to_dict()
                for key, value in discover_capabilities(include_gate=False).items()
            },
            "rates": plan.to_dict(),
            "requested_duration_s": self.config.duration_s,
            "actual_duration_s": actual_duration,
            "wall_time_s": wall_total,
            "real_time_factor": actual_duration / wall_total if wall_total > 0 else None,
            "rtf": actual_duration / wall_total if wall_total > 0 else None,
            "deadline_s": {domain: self._deadline(domain) for domain in self.DOMAINS},
            "drop_late": self.config.drop_late,
            "max_lag_s_limit": self.config.max_lag_s,
            "event_counts": domain_counts,
            "domains": domain_manifest,
            "components": self.component_manifest,
            "initial_flyvis_primed": self._flyvis_primed,
            "diagnostics": {
                "warnings": warnings,
                "last_error": self._last_error,
                "trace_sha256": digest,
                "trace_samples": len(self._trace),
                "drop_total": sum(metrics.dropped for metrics in self.metrics.values()),
                "deadline_miss_total": sum(metrics.deadline_misses for metrics in self.metrics.values()),
                "max_lag_s": max((metrics.max_lag_s for metrics in self.metrics.values()), default=0.0),
                "wall_time_s": wall_total,
                "rtf": actual_duration / wall_total if wall_total > 0 else None,
            },
        }
        manifest["drop_count"] = manifest["diagnostics"]["drop_total"]
        manifest["deadline_misses"] = manifest["diagnostics"]["deadline_miss_total"]
        manifest["lag_max_s"] = manifest["diagnostics"]["max_lag_s"]
        if self._world is not None:
            manifest["final_world"] = self._world.to_dict()
        if self._neural is not None:
            manifest["latest_neural"] = self._neural.to_dict()
        if self._command is not None:
            manifest["latest_command"] = self._command.to_dict()
        self.last_result = RuntimeResult(manifest, list(self.telemetry), self._world,
                                         self._neural, self._command)
        return self.last_result


def cpu_dry_run(duration_s: float = 0.01, *, seed: int = 1729,
                out: str | Path | None = None, **kwargs: Any) -> RuntimeResult:
    """Run a short safe CPU/stub simulation, optionally writing a manifest."""

    config = RuntimeConfig(duration_s=duration_s, seed=seed, device="cpu",
                           backend="cpu_stub", dry_run=True, **kwargs)
    result = TwinRuntime(config).run()
    if out is not None:
        result.write_manifest(out)
        if Path(out).suffix.lower() != ".json":
            result.write_telemetry(Path(out) / "telemetry.jsonl")
    return result


# More discoverable aliases for callers and hidden smoke tests.
run_cpu_dry_run = cpu_dry_run


def preflight_report(*, device: str = "auto", backend: str = "auto") -> dict[str, Any]:
    """Return a JSON-safe read-only preflight report."""

    gate = inspect_gate()
    caps = discover_capabilities(include_gate=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "device_requested": device,
        "backend_requested": backend,
        "gate": gate.to_dict(),
        "gate_passed": gate.open,
        "cuda_allowed": gate.open,
        "cpu_available": True,
        "capabilities": {
            key: value.to_dict() for key, value in caps.items()
        },
        "safe_default": "cpu_stub" if not gate.open else "cuda_candidate",
        "torch_imported": "torch" in sys.modules,
        "cuda_imported": any(name == "cuda" or name.startswith("torch.cuda") for name in sys.modules),
        "note": "Capability discovery uses module metadata only; Torch/CUDA are not imported.",
    }


# Small aliases keep the public surface friendly to orchestration scripts that
# use either noun order, without duplicating implementation.
discover_backend_capabilities = discover_capabilities
run_preflight = preflight_report
Runtime = TwinRuntime


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", "--duration", "--seconds", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "gpu"), default="auto")
    parser.add_argument("--backend", choices=("auto", "cpu", "stub", "cpu_stub", "torch", "triton", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="mark run as CPU/stub dry-run")
    parser.add_argument("--physics-hz", type=float, default=PHYSICS_HZ)
    parser.add_argument("--cns-hz", type=float, default=CNS_HZ)
    parser.add_argument("--flyvis-hz", type=float, default=FLYVIS_HZ)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--telemetry-hz", type=float, default=DEFAULT_TELEMETRY_HZ)
    parser.add_argument("--deadline-s", "--deadline", dest="deadline_s", type=float, default=None,
                        help="one deadline applied to every domain; omit for each period")
    parser.add_argument("--drop-late", action="store_true")
    parser.add_argument("--max-lag-s", type=float, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--out", "--manifest", dest="out", type=Path, default=None,
                        help="artifact directory or manifest JSON path")
    parser.add_argument("--telemetry", type=Path, default=None, help="optional JSONL telemetry path")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    args = parser.parse_args()
    deadlines = {} if args.deadline_s is None else {domain: args.deadline_s for domain in TwinRuntime.DOMAINS}
    config = RuntimeConfig(
        duration_s=args.duration_s, seed=args.seed, device=args.device,
        backend=args.backend, dry_run=args.dry_run,
        physics_hz=args.physics_hz, cns_hz=args.cns_hz,
        flyvis_hz=args.flyvis_hz, control_hz=args.control_hz,
        telemetry_hz=args.telemetry_hz, deadlines_s=deadlines,
        drop_late=args.drop_late, max_lag_s=args.max_lag_s,
        realtime=args.realtime,
    )
    try:
        result = TwinRuntime(config).run()
    except (RuntimeErrorBase, ValueError) as exc:
        print(f"runtime refused: {exc}", file=sys.stderr)
        return 2
    manifest_path: Optional[Path] = None
    if args.out is not None:
        manifest_path = result.write_manifest(args.out)
        if args.telemetry is None and args.out.suffix.lower() != ".json":
            args.telemetry = args.out / "telemetry.jsonl"
    if args.telemetry is not None:
        result.write_telemetry(args.telemetry)
    if args.json:
        print(json.dumps(_json_safe(result.manifest), indent=2, sort_keys=True))
    else:
        print("=== MaleCNS twin runtime ===")
        print(f"device/backend : {result.manifest['execution']['effective_device']}/"
              f"{result.manifest['execution']['effective_backend']}")
        print(f"simulated      : {result.manifest['actual_duration_s']:.6f} s")
        print(f"wall/RTF       : {result.manifest['wall_time_s']:.6f} s / "
              f"{result.manifest['real_time_factor']:.2f}x")
        print(f"events         : {result.manifest['event_counts']}")
        print(f"drops/late     : {result.manifest['diagnostics']['drop_total']}/"
              f"{result.manifest['diagnostics']['deadline_miss_total']}")
        if manifest_path is not None:
            print(f"manifest       : {manifest_path}")
    return 0


__all__ = [
    "BackendCapability", "GateClosedError", "GateSnapshot", "BackendUnavailableError",
    "RatePlan", "RuntimeConfig", "RuntimeComponents", "DomainMetrics", "RuntimeResult",
    "TwinRuntime", "FlyVisDrive", "SimpleWorldModel", "DeterministicSensor",
    "DeterministicFlyVis", "DeterministicNeural", "DeterministicMotor", "TorchNeuralStub",
    "discover_capabilities", "inspect_gate", "preflight_report", "cpu_dry_run", "run_cpu_dry_run",
    "discover_backend_capabilities", "run_preflight", "Runtime",
    "PHYSICS_DT_S", "CNS_DT_S", "FLYVIS_DT_S", "PHYSICS_HZ", "CNS_HZ", "FLYVIS_HZ",
    "DEFAULT_CONTROL_HZ", "DEFAULT_TELEMETRY_HZ", "SCHEMA_VERSION",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
