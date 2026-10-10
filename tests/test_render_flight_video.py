#!/usr/bin/env python3
"""Short CPU-only checks for the deterministic flight replay renderer."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_flight_video import (  # noqa: E402
    RenderError,
    interpolate_trajectory,
    load_trajectory,
    render_flight_video,
)


class RenderFlightVideoTest(unittest.TestCase):
    def _trace(self, directory: Path) -> Path:
        path = directory / "trajectory.json"
        rows = [
            {
                "time_s": 0.0,
                "position_m": [0.0, 0.0, 0.32],
                "velocity_mps": [0.0, 0.25, 0.0],
                "orientation_rad": [0.0, 0.0, 0.0],
                "mission_phase": "takeoff",
                "phase_contract": {"phase": 0.25, "phase_label": "takeoff", "index": 0},
                "left_wing": 1.0,
                "right_wing": 0.9,
                "target_distance_m": 0.85,
                "target_bearing_rad": 0.0,
                "flight_mode": "flight",
            },
            {
                "time_s": 0.06,
                "position_m": [0.0, 0.015, 0.34],
                "velocity_mps": [0.0, 0.25, 0.04],
                "orientation_rad": [0.0, 0.0, 0.12],
                "mission_phase": "cruise",
                "phase_contract": {"phase": 0.5, "phase_label": "cruise", "index": 1},
                "left_wing": 0.8,
                "right_wing": 1.0,
                "target_distance_m": 0.84,
                "target_bearing_rad": -0.1,
                "flight_mode": "flight",
            },
        ]
        path.write_text(json.dumps({
            "schema_version": "digital-fly-trajectory-1",
            "metadata": {"evidence_class": "digital_simulation", "source": "unit_test"},
            "trajectory": rows,
        }), encoding="utf-8")
        return path

    def test_interpolation_is_constant_rate_and_finite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = load_trajectory(self._trace(Path(tmp)))
            frames = interpolate_trajectory(data, fps=30)
            self.assertGreaterEqual(len(frames), 2)
            self.assertAlmostEqual(frames[1].time_s - frames[0].time_s, 1.0 / 30.0)
            self.assertTrue(all(value == value for frame in frames for value in frame.position_m))
            self.assertEqual(frames[0].phase_label, "takeoff")
            self.assertAlmostEqual(float(frames[0].phase_progress), 0.25)
            self.assertEqual(frames[-1].phase_label, "cruise")

    def test_one_sample_duration_does_not_extend_recorded_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.json"
            path.write_text(json.dumps({
                "trajectory": [{
                    "time_s": 1.25,
                    "position_m": [0.0, 0.0, 0.3],
                }],
            }), encoding="utf-8")
            data = load_trajectory(path)
            frames = interpolate_trajectory(data, fps=30, duration_s=10.0)
            self.assertEqual(len(frames), 2)
            self.assertEqual([frame.time_s for frame in frames], [1.25, 1.25])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for the short MP4 smoke test")
    def test_short_cpu_render_writes_playable_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output = directory / "flight.mp4"
            render_flight_video(self._trace(directory), output, fps=30, width=160, height=96)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)

    def test_missing_position_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"trajectory": [{"time_s": 0.0}]}), encoding="utf-8")
            with self.assertRaisesRegex(RenderError, "missing position_m"):
                load_trajectory(path)


if __name__ == "__main__":
    unittest.main()
