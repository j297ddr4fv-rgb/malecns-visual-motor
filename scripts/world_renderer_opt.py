#!/usr/bin/env python3
"""Stable optional renderer factory used by the flight runner.

The runner imports this module only for an explicit ``renderer_backend=warp_cuda``
request.  There is no implicit analytic fallback here: Warp import/device
failures are surfaced to the caller as ``WarpBackendUnavailableError``.
"""

from __future__ import annotations

from typing import Any

try:
    from warp_world_backend import (  # type: ignore
        WarpBackendContractError,
        WarpBackendUnavailableError,
        WarpRenderTimings,
        WarpWorldRenderer,
        warp_cuda_available,
    )
except ImportError:  # pragma: no cover - package-style import convenience.
    from .warp_world_backend import (  # type: ignore
        WarpBackendContractError,
        WarpBackendUnavailableError,
        WarpRenderTimings,
        WarpWorldRenderer,
        warp_cuda_available,
    )


def create_renderer(world: Any, *, backend: str = "warp_cuda",
                    renderer_backend: str | None = None,
                    device: str = "cuda:0", **kwargs: Any) -> Any:
    """Construct the explicitly requested renderer backend.

    ``analytic_reference`` returns the supplied world unchanged only when a
    caller explicitly requests it.  ``warp_cuda`` requires a CUDA-capable
    Warp runtime and never silently falls back.
    """
    requested = str(renderer_backend or backend).strip().lower()
    if requested in {"analytic_reference", "virtual_world_raycast"}:
        return world
    if requested != "warp_cuda":
        raise WarpBackendContractError(f"unsupported renderer backend {requested!r}")
    return WarpWorldRenderer.from_world(world, device=device, **kwargs)


build_renderer = create_renderer
create_world_renderer = create_renderer
WarpCudaRenderer = WarpWorldRenderer
WarpRenderer = WarpWorldRenderer

__all__ = [
    "WarpBackendContractError",
    "WarpBackendUnavailableError",
    "WarpCudaRenderer",
    "WarpRenderTimings",
    "WarpRenderer",
    "WarpWorldRenderer",
    "build_renderer",
    "create_renderer",
    "create_world_renderer",
    "warp_cuda_available",
]
