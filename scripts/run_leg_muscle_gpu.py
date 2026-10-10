#!/usr/bin/env python3
"""Explicit CUDA MaleCNS + CPU muscle/MuJoCo intervention launcher.

The default is Torch CUDA, never CPU fallback. --preflight-only reads gate
metadata without importing or starting a GPU runtime. Visual and afferent
closed-loop input are separate from this deterministic motor-interface probe.
"""
from pathlib import Path

from run_leg_muscle_cpu import main


if __name__ == "__main__":
    raise SystemExit(main(
        default_neural_backend="torch_cuda", default_duration=.05,
        default_out=Path("/data/artifacts/by-project/malecns-gpu/experiments/articulated-neuromuscular-gpu-20261003"),
    ))
