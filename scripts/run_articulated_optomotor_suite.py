#!/usr/bin/env python3
"""Bounded, sequential stripe repetitions with explicit body and sensory arms."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/data/environments/projects/flyvis-opticlobe/bin/python')
DEFAULT_SEEDS = (1729,)


def command(body: str, seed: int, condition: str, out: Path,
            feedback: str, duration: float,
            muscle_config: Path | None = None,
            muscle_force_scale: float | None = None,
            record_motor_units: bool = False) -> list[str]:
    if (muscle_config is not None or muscle_force_scale is not None or record_motor_units) \
            and body != 'articulated':
        raise ValueError('muscle profile options require body=articulated')
    if muscle_force_scale is not None and (
            not math.isfinite(muscle_force_scale)
            or muscle_force_scale <= 0.0
            or muscle_force_scale > 10.0):
        raise ValueError('muscle_force_scale must be finite and in (0, 10]')
    if body == 'reduced':
        return [str(PYTHON), str(ROOT/'scripts/run_digital_fly.py'),
                '--config', str(ROOT/'config/optomotor_ground_crawl_experiment.json'),
                '--out', str(out), '--scene-manifest', str(ROOT/'config/optomotor_panel_scene.json'),
                '--mode', 'free-behavior', '--neural', 'malecns', '--vision', 'flyvis',
                '--body', 'reduced', '--renderer-backend', 'warp_cuda',
                '--neural-device', 'cuda', '--neural-backend', 'torch',
                '--seed', str(seed), '--duration', str(duration),
                '--stimulus-config', str(ROOT/'config/optomotor_pilot_stimuli.json'),
                '--stimulus-condition', 'stripe_'+condition, '--max-vram-gib', '15.5']
    runtime = [str(PYTHON), str(ROOT/'scripts/run_articulated_optomotor.py'),
               '--out', str(out), '--seed', str(seed), '--duration', str(duration),
               '--condition', condition, '--sensory-feedback', feedback]
    if muscle_config is not None:
        runtime.extend(['--muscle-config', str(muscle_config)])
    if muscle_force_scale is not None:
        runtime.extend(['--muscle-force-scale', str(muscle_force_scale)])
    if record_motor_units:
        runtime.append('--record-motor-units')
    return runtime


def run_one(body: str, seed: int, condition: str, out: Path, log: Path,
            feedback: str, duration: float,
            muscle_config: Path | None = None,
            muscle_force_scale: float | None = None,
            record_motor_units: bool = False) -> dict:
    feedback = 'legacy' if body == 'reduced' else feedback
    if out.exists() or log.exists():
        raise FileExistsError(f'Refusing to replace existing evidence: {out}, {log}')
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    runtime = command(body, seed, condition, out, feedback, duration,
                      muscle_config, muscle_force_scale, record_motor_units)
    bounded = ['systemd-run', '--user', '--wait', '--pipe', '--collect',
               f'--unit=malecns-stripes-{time.time_ns()}',
               f'--property=WorkingDirectory={ROOT}', '--property=MemoryMax=5G',
               '--property=LimitMEMLOCK=64M', '--property=RuntimeMaxSec=1800',
               f'--setenv=PYTHONPATH={ROOT}/scripts',
               '--setenv=FLYVIS_ROOT_DIR=/data/models/flyvis',
               '--setenv=WARP_CACHE_PATH=/data/cache/malecns-gpu/warp',
               '--setenv=TRITON_CACHE_DIR=/data/cache/malecns-gpu/triton',
               '--setenv=CUDA_CACHE_PATH=/data/cache/malecns-gpu/cuda',
               '--setenv=PYTHONUNBUFFERED=1', '--setenv=PYTHONDONTWRITEBYTECODE=1', *runtime]
    print(f'RUN {body}/{feedback} seed={seed} {condition}', flush=True)
    started = time.monotonic()
    with log.open('x') as handle:
        result = subprocess.run(bounded, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic()-started
    record = dict(body=body, feedback=feedback, seed=seed, condition=condition,
                  duration_s=duration, path=str(out), log=str(log), command=bounded,
                  elapsed_s=elapsed, exit_code=result.returncode)
    with log.with_suffix('.invocation.json').open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    if result.returncode:
        raise RuntimeError(f'Failed trial; retained log: {log}')
    if not (out/'manifest.json').is_file() or not (out/'telemetry.jsonl').is_file():
        raise RuntimeError(f'Missing completed run evidence: {out}')
    print(f'DONE {body}/{feedback} seed={seed} {condition} wall={elapsed:.1f}s', flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--body', choices=('reduced', 'articulated'), required=True)
    parser.add_argument('--sensory-feedback', choices=('legacy', 'off', 'on'), default='on')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--conditions', nargs='+', choices=('left', 'right', 'static'),
                        default=('left', 'right', 'static'))
    parser.add_argument('--duration', type=float, default=7.5)
    parser.add_argument('--muscle-config', type=Path,
                        help='optional strict leg muscle parameter profile for articulated trials')
    parser.add_argument('--muscle-force-scale', type=float,
                        help='multiply configured muscle F_iso once; range (0, 10]')
    parser.add_argument('--record-motor-units', action='store_true',
                        help='record sparse per-tick leg motor-neuron body-ID events')
    args = parser.parse_args()
    if args.muscle_force_scale is not None and (
            not math.isfinite(args.muscle_force_scale)
            or args.muscle_force_scale <= 0.0
            or args.muscle_force_scale > 10.0):
        parser.error('--muscle-force-scale must be finite and in (0, 10]')
    if args.body != 'articulated' and (
            args.muscle_config is not None or args.muscle_force_scale is not None
            or args.record_motor_units):
        parser.error('muscle profile and motor-unit options require --body articulated')
    for seed in args.seeds:
        for condition in args.conditions:
            name = f'seed-{seed}-{condition}'
            run_one(args.body, seed, condition, args.out/name, args.logs/(name+'.log'),
                    args.sensory_feedback, args.duration,
                    args.muscle_config, args.muscle_force_scale,
                    args.record_motor_units)


if __name__ == '__main__':
    main()
