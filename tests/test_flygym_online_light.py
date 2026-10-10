"""Unit fixtures for compact online capture; no neural or physics runtime required."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_flygym_free_body_online_lowcontrast_v2.py'
spec = importlib.util.spec_from_file_location('online_light_test_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OnlineLightTests(unittest.TestCase):
    def test_sampled_yaw_summary_counts_saved_samples(self):
        trace = {'time_s': np.array([0., 1., 4., 8., 13.]),
                 'thorax_heading_yaw_rad': np.array([0., .1, .2, .3, .4]),
                 'thorax_heading_xy': np.ones((5, 2))}
        session = SimpleNamespace(capture_mode='light', _native_samples=130001,
                                  light_trace_arrays=lambda: trace)
        with patch.object(runner, 'WINDOWS', {'stimulus': (1., 13.)}):
            summary = runner._body_window_summary(session)
        self.assertEqual(summary['raw_trace']['samples'], 5)
        self.assertAlmostEqual(summary['stimulus']['thorax_heading_yaw_change_deg'], np.degrees(.3))
        self.assertEqual(session._native_samples, 130001)

    def test_light_drive_needs_clock_rates_but_no_retina_dump(self):
        ready = {'capture_mode': 'light', 'effective_backends': {'neural': 'triton'}}
        reply = {'protocol': runner.PROTOCOL_NAME, 'type': 'drive', 'seq': 0,
                 'time_s': 0., 'render_time_s': 0., 'pattern_time_s': 0.,
                 'neural_window_end_s': .005, 'leg_power': .2,
                 'population_rates_hz': {'DNp15/L': 10., 'DNp15/R': 20.}}
        self.assertEqual(runner._validate_drive(reply, 0, 0., ready), (.2, 10., 20.))
        with self.assertRaises(runner.RunnerError):
            runner._validate_drive(reply, 0, 0., {**ready, 'capture_mode': 'full'})
        with self.assertRaises(runner.RunnerError):
            runner._validate_drive({**reply, 'neural_window_end_s': .01}, 0, 0., ready)

    def test_launcher_runs_once_and_keeps_failure_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / 'reference' / 'protocol.json'
            reference.parent.mkdir()
            reference.write_text(json.dumps({'stimuli': {'conditions': [{
                'case_id': 'c1-static', 'contrast': 1., 'duration_s': 13.,
                'onset_s': 1., 'frame_count': 2600, 'schedule_path': '/unused'}]}}))
            output = root / 'fresh'
            def fail(command, **kwargs):
                self.assertIn('--property=CPUQuota=1200%', command)
                self.assertIn('--capture-stride', command)
                kwargs['stdout'].write('worker exited: example failure\n')
                raise RuntimeError('example failure')
            with patch.dict(runner.os.environ, {'ONLINE_LIGHT_CHILD': '0'}), \
                    patch.object(runner.subprocess, 'run', side_effect=fail) as launch:
                with self.assertRaises(RuntimeError):
                    runner._run_light(reference, output, 'c1-static', 50)
                self.assertEqual(launch.call_count, 1)
                self.assertIn('example failure', (output / 'launcher.log').read_text())
                with self.assertRaises(FileExistsError):
                    runner._run_light(reference, output, 'c1-static', 50)
                self.assertEqual(launch.call_count, 1)


if __name__ == '__main__':
    unittest.main()
