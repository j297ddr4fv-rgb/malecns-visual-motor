"""Check fail-closed launch order and the host neural/physics interchange."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import run_leg_muscle_cpu as runner
from leg_muscle_neural_backend import NeuralBackendGateError


class RunnerGateTests(unittest.TestCase):
    def test_ten_second_duration_reaches_gate_and_longer_runs_are_rejected(self):
        blocked = {'gate_passed': False, 'blockers': ['GPU offline'], 'requires_cuda': True}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(runner, 'neural_backend_preflight', return_value=blocked):
            out = Path(tmp)/'run'
            with self.assertRaises(NeuralBackendGateError):
                runner.run_case('sham', out, duration_s=10., neural_backend='torch_cuda')
            with self.assertRaisesRegex(ValueError, r'\[0.025, 10\]'):
                runner.run_case('sham', out, duration_s=10.0005, neural_backend='torch_cuda')
            self.assertFalse(out.exists())
    def test_blocked_gpu_stops_before_data_model_output_or_factory(self):
        blocked = {'gate_passed': False, 'blockers': ['GPU offline'], 'requires_cuda': True}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            with mock.patch.object(runner, 'neural_backend_preflight', return_value=blocked), \
                 mock.patch.object(runner, 'AnnotationIndex') as index, \
                 mock.patch.object(runner, 'ArticulatedLegBody') as body, \
                 mock.patch.object(runner, 'build_leg_neural_engine') as factory:
                with self.assertRaises(NeuralBackendGateError):
                    runner.run_case('extensor', out, duration_s=.025,
                                    neural_backend='torch_cuda', csr_path=Path(tmp)/'missing')
                index.assert_not_called()
                body.assert_not_called()
                factory.assert_not_called()
                self.assertFalse(out.exists())

    def test_preflight_cli_has_no_simulation_side_effects(self):
        blocked = {'gate_passed': False, 'blockers': ['GPU offline'], 'requires_cuda': True}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(runner, 'neural_backend_preflight', return_value=blocked), \
             mock.patch.object(runner, 'run_case') as run, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            out = Path(tmp)/'no-output'
            status = runner.main(['--preflight-only', '--neural-backend', 'triton', '--out', str(out)])
            self.assertEqual(status, 2)
            self.assertFalse(json.loads(output.getvalue())['gate_passed'])
            run.assert_not_called()
            self.assertFalse(out.exists())

    def test_prescribed_source_cannot_claim_gpu_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            with self.assertRaisesRegex(ValueError, 'actual malecns'):
                runner.run_case('extensor', out, source='prescribed-rates', neural_backend='triton')
            self.assertFalse(out.exists())

    def test_canonical_spikes_reject_invalid_data_and_preserve_neuron_order(self):
        for value in (np.array([0., np.nan]), np.array([0., 2.]), np.array([0., .5]),
                      np.array([0., -1.]), np.array([[0, 1]])):
            with self.assertRaises(ValueError):
                runner.canonical_spikes(value, 2)
        actual = runner.canonical_spikes(np.array([True, False, True, False])[::2], 2)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(actual.tolist(), [1, 1])


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'optional CPU MuJoCo is not installed')
class RunnerInterchangeTests(unittest.TestCase):
    def test_feedback_reads_previous_body_and_drives_afferents_without_motor_intervention(self):
        from test_malecns_engine import make_micro_csr

        class FixtureFeedback:
            def __init__(self):
                self.observed = []

            def encode(self, body, timestamp_s, *, enabled=True, silenced=False):
                self.observed.append((body['time_s'], timestamp_s))
                loaded = any(f['contact'] for f in body['feet'].values())
                current = 2.0 if loaded and enabled and not silenced else 0.0
                return {'source_time_s': body['time_s'], 'applied_at_s': timestamp_s,
                        'currents_mV_per_ms': {'class:mechanosensory_proprioceptive': current}}

            def provenance(self):
                return {'scope': 'synthetic two-afferent transfer fixture; not biological tuning'}

        with tempfile.TemporaryDirectory() as tmp:
            csr = make_micro_csr(Path(tmp)/'csr')
            tables = json.loads((csr/'tables.json').read_text())
            tables['types'] = ['', 'Ti extensor MN', 'Ti flexor MN']
            tables['superclasses'] = ['', 'vnc_motor']
            tables['subclasses'] = ['', 'fl']
            tables['classes'] = ['', 'mechanosensory_proprioceptive']
            (csr/'tables.json').write_text(json.dumps(tables))
            ann = dict(np.load(csr/'annotations.npz'))
            ann['sideIdx'] = np.ones(4, dtype=np.uint8)
            ann['superclassIdx'] = np.array([1, 1, 0, 0], dtype=np.uint8)
            ann['classIdx'] = np.array([0, 0, 1, 1], dtype=np.uint8)
            np.savez(csr/'annotations.npz', **ann)
            summaries = {}
            for mode in ('on', 'muted'):
                adapter = FixtureFeedback()
                out = Path(tmp)/mode
                with mock.patch.object(runner, 'plot_case'), \
                     mock.patch('leg_sensory_feedback.LegSensoryFeedback', return_value=adapter):
                    summaries[mode] = runner.run_case('sham', out, duration_s=.025,
                        fixed_root=True, csr_path=csr, sensory_feedback=mode)
                self.assertEqual(len(adapter.observed), 50)
                for tick, (source, applied) in enumerate(adapter.observed):
                    self.assertAlmostEqual(source, tick*.0005, places=10)
                    self.assertAlmostEqual(applied, tick*.0005, places=10)
                rows = [json.loads(line) for line in (out/'sensory_feedback.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 50)
                manifest = json.loads((out/'manifest.json').read_text())
                self.assertEqual(runner.file_sha256(out/'sensory_feedback.jsonl'), manifest['sensory_feedback_sha256'])
            self.assertGreater(summaries['on']['afferent_spikes'], 0)
            self.assertEqual(summaries['muted']['afferent_spikes'], 0)
            self.assertEqual(summaries['muted']['total_spikes'], 0)
            self.assertEqual(summaries['on']['control_authority'], 'body_derived_sensory_current_proxy')

    def test_cpu_micro_cns_counts_and_evidence_describe_actual_execution(self):
        from test_malecns_engine import make_micro_csr
        with tempfile.TemporaryDirectory() as tmp:
            csr = make_micro_csr(Path(tmp)/'csr')
            tables = json.loads((csr/'tables.json').read_text())
            tables['types'] = ['', 'Ti extensor MN', 'Ti flexor MN']
            tables['superclasses'] = ['', 'vnc_motor']
            tables['subclasses'] = ['', 'fl']
            (csr/'tables.json').write_text(json.dumps(tables))
            ann = dict(np.load(csr/'annotations.npz'))
            ann['sideIdx'] = np.ones(4, dtype=np.uint8)
            np.savez(csr/'annotations.npz', **ann)
            out = Path(tmp)/'probe'
            with mock.patch.object(runner, 'plot_case'):
                summary = runner.run_case('extensor', out, duration_s=.025,
                                          fixed_root=True, csr_path=csr, seed=23)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(summary['neural_backend_actual'], 'numpy_cpu')
            self.assertEqual(summary['neural_device_actual'], 'cpu')
            self.assertFalse(summary['gpu_used'])
            self.assertEqual(summary['seed'], 23)
            self.assertEqual(summary['full_neural_trace_ticks'], 50)
            self.assertEqual(summary['neural_spike_shape'], [50, 4])
            counts = [json.loads(line) for line in (out/'neural_spike_counts.jsonl').read_text().splitlines()]
            self.assertEqual([r['tick'] for r in counts], list(range(50)))
            self.assertEqual(sum(r['total_spikes'] for r in counts), summary['total_spikes'])
            self.assertEqual(sum(r['target_spikes'] for r in counts), summary['target_spikes'])
            self.assertEqual(runner.file_sha256(out/'neural_spike_counts.jsonl'), manifest['neural_counts_sha256'])
            self.assertEqual(set(manifest['csr_sha256']), set(runner.CSR_FILES))
            performance = json.loads((out/'performance.json').read_text())
            self.assertEqual(performance['neural_ticks'], 50)
            self.assertEqual(performance['windows'][0]['start_tick'], 0)
            self.assertEqual(performance['windows'][0]['end_tick'], 49)
            self.assertEqual(performance['simulated_duration_s'], .025)
            self.assertGreater(performance['simulation_loop_s'], 0)
            self.assertTrue(all(v >= 0 for v in performance['stages_s'].values()))
            self.assertEqual(runner.file_sha256(out/'performance.json'), manifest['performance']['sha256'])
            for name, digest in manifest['source_sha256'].items():
                self.assertEqual(hashlib.sha256((out/'sources'/name).read_bytes()).hexdigest(), digest)
            telemetry = [json.loads(line) for line in (out/'telemetry.jsonl').read_text().splitlines()]
            self.assertEqual(len(telemetry[-1]['body']['joints']), 42)
            for record in telemetry:
                expected = counts[record['neural']['tick']]
                self.assertEqual(record['neural']['target_spikes'], expected['target_spikes'])
                self.assertEqual(record['neural']['total_spikes'], expected['total_spikes'])


if __name__ == '__main__':
    unittest.main()
