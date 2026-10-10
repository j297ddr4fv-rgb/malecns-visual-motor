"""Explicit MaleCNS backend selection for the articulated leg runner.

The public names in this adapter are intentionally narrow: ``cpu``,
``torch_cuda``, and ``triton``. CUDA requests are checked against the
project's read-only MaleCNS gate before the engine factory is called. The
engine performs its own second gate check immediately before importing CUDA
libraries.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import malecns_engine


DEFAULT_MAX_VRAM_GIB = 15.5
_BACKEND_ENGINE_OPTIONS: dict[str, dict[str, str]] = {
    "cpu": {"backend": "cpu", "device": "cpu"},
    "torch_cuda": {"backend": "torch", "device": "cuda"},
    "triton": {"backend": "triton", "device": "cuda"},
}


class NeuralBackendError(RuntimeError):
    """A requested neural backend could not be selected exactly."""


class NeuralBackendGateError(NeuralBackendError):
    """A CUDA request was refused by the MaleCNS safety gate."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.preflight_report = dict(report)
        blockers = self.preflight_report.get("blockers", [])
        detail = "; ".join(str(item) for item in blockers) or "unknown gate failure"
        super().__init__(f"MaleCNS CUDA safety gate is closed: {detail}")


def _validate_backend(backend: str) -> str:
    if not isinstance(backend, str) or backend not in _BACKEND_ENGINE_OPTIONS:
        allowed = ", ".join(_BACKEND_ENGINE_OPTIONS)
        raise ValueError(f"backend must be exactly one of: {allowed}")
    return backend


def _load_gate() -> Any:
    """Load only the dependency-light gate module, never a GPU runtime."""

    return importlib.import_module("malecns_gate")


def neural_backend_preflight(backend: str) -> dict[str, Any]:
    """Return a plain-data backend/gate report without importing GPU libraries.

    CPU selection intentionally does not load or query the CUDA gate. For a
    CUDA backend, ``malecns_gate.cuda_blockers`` checks host/driver metadata
    only; the engine repeats its fail-closed check immediately before Torch
    or Triton imports.
    """

    requested = _validate_backend(backend)
    requires_cuda = requested != "cpu"
    report: dict[str, Any] = {
        "requested_backend": requested,
        "requires_cuda": requires_cuda,
        "gate_checked": False,
        "gate_passed": not requires_cuda,
        "blockers": [],
    }
    if not requires_cuda:
        return report

    report["gate_checked"] = True
    try:
        blockers = _load_gate().cuda_blockers()
        if blockers is None:
            raise RuntimeError("gate returned no blocker report")
        report["blockers"] = [str(blocker) for blocker in blockers]
        report["gate_passed"] = not report["blockers"]
    except Exception as exc:
        report["gate_passed"] = False
        report["blockers"] = [
            f"gate check failed: {type(exc).__name__}: {exc}"
        ]
    return report


def _engine_factory(**kwargs: Any) -> malecns_engine.MaleCNSEngine:
    """Small indirection point so adapter tests can use a fake engine."""

    return malecns_engine.create_engine(**kwargs)


def _audit_effective_backend(engine: Any, requested_backend: str) -> None:
    expected = {
        "cpu": ("numpy_cpu", "cpu"),
        "torch_cuda": ("torch_cuda", "cuda"),
        "triton": ("triton", "cuda"),
    }[requested_backend]
    actual = (getattr(engine, "backend", None), getattr(engine, "device", None))
    if actual != expected:
        raise NeuralBackendError(
            f"requested neural backend {requested_backend!r} requires effective "
            f"backend/device {expected[0]!r}/{expected[1]!r}; engine reported "
            f"{actual[0]!r}/{actual[1]!r}"
        )


def build_leg_neural_engine(
    csr_path: str | Path,
    backend: str = "cpu",
    seed: int = 1729,
    max_vram_gib: float | None = DEFAULT_MAX_VRAM_GIB,
) -> malecns_engine.MaleCNSEngine:
    """Build a leg-runner engine with an explicit, fail-closed backend.

    ``torch_cuda`` maps to the engine's Torch CUDA backend, while ``triton``
    maps to its Triton CUDA backend. Neither request can fall back to CPU.
    """

    requested = _validate_backend(backend)
    preflight = neural_backend_preflight(requested)
    if preflight["requires_cuda"] and not preflight["gate_passed"]:
        raise NeuralBackendGateError(preflight)

    engine_options = _BACKEND_ENGINE_OPTIONS[requested]
    engine = _engine_factory(
        csr_path=csr_path,
        seed=int(seed),
        backend=engine_options["backend"],
        device=engine_options["device"],
        max_vram_gib=max_vram_gib,
    )
    _audit_effective_backend(engine, requested)
    return engine


__all__ = [
    "DEFAULT_MAX_VRAM_GIB",
    "NeuralBackendError",
    "NeuralBackendGateError",
    "build_leg_neural_engine",
    "neural_backend_preflight",
]
