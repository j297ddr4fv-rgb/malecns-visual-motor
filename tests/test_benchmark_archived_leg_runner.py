"""Timing hooks must preserve fixture state and enforce source isolation."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from scripts import benchmark_archived_leg_runner as benchmark


class ArchivedTimingTests(unittest.TestCase):
    def fixture(self, root):
        scripts = root/'old/scripts'; scripts.mkdir(parents=True)
        (root/'old/config').mkdir()
        modules = {name: ModuleType(name) for name in
                   ('run_leg_muscle_cpu','malecns_engine','leg_muscle_pipeline','articulated_leg_body')}
        for name, module in modules.items():
            module.__file__ = str(scripts/f'{name}.py')

        class Body:
            def __init__(self):
                self.accumulator = 0

            def snapshot(self):
                return {'fixture_state': self.accumulator}

        class Engine:
            def step(self, drive):
                return {'fixture_observation': drive}

        class Pipeline:
            def __init__(self, body):
                self.body = body

            def step(self, observation):
                self.body.accumulator += observation['fixture_observation']
                return {'fixture_state': self.body.accumulator}

        def write_json(path, value):
            path.write_text(json.dumps(value))

        def run_case(case, out, *, duration_s, neural_backend, seed, fixed_root):
            import time
            started = time.monotonic()  # same caller-local contract as archived runner
            out.mkdir(parents=True)
            body, engine = Body(), Engine()
            pipe = Pipeline(body)
            initial = body.snapshot()
            results = [pipe.step(engine.step(tick)) for tick in range(round(duration_s/.0005))]
            final = body.snapshot()
            summary = {'scope': 'synthetic timing fixture, not CUDA evidence',
                       'case': case, 'seed': seed, 'initial': initial, 'final': final, 'results': results}
            write_json(out/'summary.json', summary)
            write_json(out/'manifest.json', {'summary': dict(summary)})
            return summary

        modules['run_leg_muscle_cpu'].run_case = run_case
        modules['run_leg_muscle_cpu'].write_json = write_json
        modules['run_leg_muscle_cpu'].file_sha256 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        modules['malecns_engine'].MaleCNSEngine = Engine
        modules['leg_muscle_pipeline'].LegMusclePipeline = Pipeline
        modules['articulated_leg_body'].ArticulatedLegBody = Body
        return modules

    def test_timing_hooks_preserve_return_values_and_record_complete_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modules = self.fixture(root)
            out = root/'artifact'
            with mock.patch.dict(sys.modules, modules), \
                 mock.patch.object(sys, 'path', list(sys.path)), \
                 mock.patch.object(sys, 'argv', ['benchmark', '--source-root', str(root/'old'),
                                                '--out', str(out), '--duration', '.002']), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.main(), 0)
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['initial'], {'fixture_state': 0})
            self.assertEqual(summary['final'], {'fixture_state': 6})
            self.assertEqual(summary['results'], [{'fixture_state': v} for v in (0,1,3,6)])
            performance = json.loads((out/'performance.json').read_text())
            self.assertEqual(performance['neural_ticks'], 4)
            self.assertEqual(performance['windows'][0]['end_tick'], 3)
            self.assertGreater(performance['simulation_loop_s'], 0)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['performance']['sha256'], hashlib.sha256((out/'performance.json').read_bytes()).hexdigest())

    def test_importing_current_code_instead_of_archived_epoch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modules = self.fixture(root)
            modules['leg_muscle_pipeline'].__file__ = str(root/'current/leg_muscle_pipeline.py')
            with mock.patch.dict(sys.modules, modules), \
                 mock.patch.object(sys, 'path', list(sys.path)), \
                 mock.patch.object(sys, 'argv', ['benchmark', '--source-root', str(root/'old'),
                                                '--out', str(root/'artifact')]):
                with self.assertRaisesRegex(RuntimeError, 'outside its frozen source root'):
                    benchmark.main()
            self.assertFalse((root/'artifact').exists())
