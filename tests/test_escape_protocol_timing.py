from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lif_flyb import _aligned_steps  # noqa: E402


class EscapeProtocolTimingTests(unittest.TestCase):
    def test_time_grid_never_rounds_declared_duration(self) -> None:
        self.assertEqual(_aligned_steps(130.0, 10.0, "reporting"), 13)
        self.assertEqual(_aligned_steps(10.0, 0.5, "integration"), 20)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            _aligned_steps(130.0, 50.0, "reporting")

    def test_swat_defaults_are_130_frames_at_one_millisecond(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "swat.npz"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "loom_to_lamina.py"),
                    "--swat",
                    "--out",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data["rates"].shape[0], 130)
                self.assertEqual(float(data["frame_ms"]), 1.0)
                meta = json.loads(str(data["meta"]))
            self.assertEqual(meta["kind"], "physical_swat")
            self.assertEqual(meta["duration_ms"], 130.0)
            self.assertEqual(meta["physical_swat"]["impact_ms"], 100.0)
            self.assertEqual(meta["physical_swat"]["speed_mps"], 10.0)

    def test_sham_has_no_visual_drive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sham.npz"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "loom_to_lamina.py"),
                    "--swat",
                    "--control",
                    "sham",
                    "--out",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(float(np.max(data["rates"])), 0.0)
                meta = json.loads(str(data["meta"]))
            self.assertEqual(meta["visual_control"], "sham")


if __name__ == "__main__":
    unittest.main()
