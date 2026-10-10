#!/usr/bin/env python3
"""Time a frozen pre-update leg runner without changing its physical model."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=10.)
    args = parser.parse_args()
    scripts = args.source_root.resolve()/'scripts'
    if not scripts.is_dir() or not (args.source_root/'config').is_dir():
        parser.error('--source-root requires isolated scripts and config directories')
    if args.out.exists():
        parser.error('--out exists; refusing to overwrite')
    sys.path.insert(0, str(scripts))
    import run_leg_muscle_cpu as runner
    import malecns_engine as engine
    import leg_muscle_pipeline as pipeline
    import articulated_leg_body as articulated
    for module in (runner, engine, pipeline, articulated):
        if Path(module.__file__).resolve().parent != scripts:
            raise RuntimeError('archived benchmark imported a module outside its frozen source root')
    # Wrappers add host wall-clock observation only. State, inputs, configuration,
    # update order and returned observations are not changed.
    keys = ('neural_s', 'muscle_physics_s', 'sensory_encode_s', 'other_s')
    stages = dict.fromkeys(keys, 0.)
    windows = []
    clock = {'ticks': 0, 'loop_start': None, 'window_start': None,
             'window_start_tick': 0, 'initialization_s': None, 'finished': False}
    prior_stages = dict(stages)
    expected_ticks = round(args.duration/.0005)

    def finish_window():
        now = time.perf_counter()
        parts = {k: stages[k]-prior_stages[k] for k in keys}
        elapsed = now-clock['window_start']
        parts['other_s'] = max(0., elapsed-parts['neural_s']-parts['muscle_physics_s'])
        stages['other_s'] += parts['other_s']
        windows.append({'start_tick': clock['window_start_tick'], 'end_tick': clock['ticks']-1,
                        'simulated_end_s': clock['ticks']*.0005, 'elapsed_s': elapsed,
                        'stages_s': parts})
        prior_stages.update(stages)
        clock['window_start'] = now
        clock['window_start_tick'] = clock['ticks']
        print(json.dumps({'event': 'baseline_progress', 'simulated_s': clock['ticks']*.0005,
                          'loop_wall_s': now-clock['loop_start']}), flush=True)

    original_neural = engine.MaleCNSEngine.step
    original_pipeline = pipeline.LegMusclePipeline.step
    original_snapshot = articulated.ArticulatedLegBody.snapshot

    def neural_step(self, *values, **kw):
        if clock['loop_start'] is None:
            clock['loop_start'] = clock['window_start'] = time.perf_counter()
            caller = inspect.currentframe().f_back
            if caller is None or caller.f_code.co_name != 'run_case':
                raise RuntimeError('unexpected archived neural-call context')
            clock['initialization_s'] = time.monotonic()-caller.f_locals['started']
        elif clock['ticks'] % 2000 == 0:
            finish_window()
        started = time.perf_counter()
        result = original_neural(self, *values, **kw)
        stages['neural_s'] += time.perf_counter()-started
        return result

    def physics_step(self, *values, **kw):
        started = time.perf_counter()
        result = original_pipeline(self, *values, **kw)
        stages['muscle_physics_s'] += time.perf_counter()-started
        clock['ticks'] += 1
        return result

    def snapshot(self, *values, **kw):
        if clock['ticks'] == expected_ticks and not clock['finished']:
            loop_end = time.perf_counter()
            finish_window()
            clock['simulation_loop_s'] = loop_end-clock['loop_start']
            clock['finished'] = True
        return original_snapshot(self, *values, **kw)

    engine.MaleCNSEngine.step = neural_step
    pipeline.LegMusclePipeline.step = physics_step
    articulated.ArticulatedLegBody.snapshot = snapshot
    summary = runner.run_case('sham', args.out, duration_s=args.duration,
                              neural_backend='torch_cuda', seed=1729, fixed_root=False)
    if not clock['finished'] or clock['ticks'] != expected_ticks:
        raise RuntimeError('archived loop did not complete the requested neural ticks')
    performance = {'schema_version': 'leg-neuromuscular-performance-1',
        'initialization_s': clock['initialization_s'], 'simulation_loop_s': clock['simulation_loop_s'],
        'stages_s': stages, 'windows': windows, 'neural_ticks': expected_ticks,
        'simulated_duration_s': args.duration,
        'stage_semantics': {'neural_s': 'neural.step including host observation transfer and synchronization',
            'muscle_physics_s': 'decoder, rate filter, Hill muscle substeps and MuJoCo physics/telemetry',
            'sensory_encode_s': 'absent in this archived source epoch',
            'other_s': 'dense spike hashing/counts, input assembly and telemetry serialization/I/O'},
        'scope': 'instrumented archived runner; cross-epoch scenario, not identical physical outcomes'}
    runner.write_json(args.out/'performance.json', performance)
    manifest = json.loads((args.out/'manifest.json').read_text())
    manifest['performance'] = {'path': 'performance.json', 'sha256': runner.file_sha256(args.out/'performance.json')}
    own_snapshot = args.out/'benchmark_sources'; own_snapshot.mkdir()
    own_source = Path(__file__).resolve()
    shutil.copy2(own_source, own_snapshot/own_source.name)
    manifest['benchmark_instrumentation'] = {'source': own_source.name,
        'source_sha256': hashlib.sha256(own_source.read_bytes()).hexdigest(),
        'source_snapshot_directory': 'benchmark_sources',
        'model_changes': 'none; copied runner duration cap extended to 10 s before launch',
        'comparison_scope': 'same seed, dt, source CSR and no motor intervention; physical support differs by epoch'}
    summary.update(simulation_loop_wall_s=clock['simulation_loop_s'], initialization_wall_s=clock['initialization_s'])
    manifest['summary'].update(simulation_loop_wall_s=clock['simulation_loop_s'], initialization_wall_s=clock['initialization_s'])
    runner.write_json(args.out/'summary.json', summary)
    runner.write_json(args.out/'manifest.json', manifest)
    print(json.dumps({'status': 'COMPLETED', 'out': str(args.out),
                      'simulation_loop_s': clock['simulation_loop_s']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
