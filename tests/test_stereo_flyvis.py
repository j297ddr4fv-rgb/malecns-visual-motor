#!/usr/bin/env python3
"""Focused tests for the dependency-light stereo FlyVis adapter."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from twin_contracts import Timestamp, VisualObservation  # noqa: E402
from stereo_flyvis import (  # noqa: E402
    CNS_DT,
    CNS_TICKS_PER_FLYVIS,
    FLYVIS_DT,
    ShapeError,
    StereoFlyVis,
    aliasing_score,
    direction_selectivity,
    left_right_common_mode,
)


class _EchoBackend:
    """Tiny signed backend used to observe the two independent eye inputs."""

    name = "echo"
    is_stub = False
    n_t4t5 = 2

    def __init__(self) -> None:
        self.seen = []

    def reset(self) -> None:
        self.seen.clear()

    def step(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        self.seen.append((left.copy(), right.copy()))
        return np.array([left.mean() - right.mean(), -left.mean()], dtype=np.float32)


class StereoFlyVisTest(unittest.TestCase):
    def test_stub_is_explicit_and_deterministic(self) -> None:
        x = np.arange(4)
        movie = np.array([0.5 + 0.2 * np.sin(2 * np.pi * (x - k) / 4)
                          for k in range(3)], dtype=np.float32)
        a = StereoFlyVis(4, blur_sigma_px=0, backend="stub")
        b = StereoFlyVis(4, blur_sigma_px=0, backend="stub")
        oa, ob = a.process_frames(movie), b.process_frames(movie)
        self.assertTrue(a.backend_metadata["is_stub"])
        self.assertEqual(a.backend_metadata["backend_note"], "explicit deterministic stub")
        np.testing.assert_array_equal(oa[-1].drive, ob[-1].drive)
        self.assertTrue(np.any(oa[-1].drive < 0) or np.any(oa[-1].drive > 0))

    def test_left_and_right_are_separate_and_signed(self) -> None:
        backend = _EchoBackend()
        engine = StereoFlyVis(4, blur_sigma_px=0, backend=backend,
                               baseline={"left": np.full(4, .2), "right": np.full(4, .4)})
        out = engine.process_frame(np.full(4, .5), np.full(4, .7))
        np.testing.assert_allclose(backend.seen[-1][0], .3)
        np.testing.assert_allclose(backend.seen[-1][1], .3)
        self.assertEqual(out.drive.shape, (2,))
        self.assertLess(float(out.drive[1]), 0.0)

    def test_bandlimit_and_alias_warning(self) -> None:
        checker = (np.indices((16, 16)).sum(axis=0) % 2).astype(np.float32)
        smooth = np.full((16, 16), .5, dtype=np.float32)
        self.assertGreater(aliasing_score(checker), aliasing_score(smooth))
        engine = StereoFlyVis(8, blur_sigma_px=1.0, aliasing_threshold=.2)
        _, _, diag = engine.frame_to_columns(checker)
        self.assertTrue(diag["aliasing_warning"])
        self.assertLess(float(engine.frame_to_columns(smooth)[2]["aliasing_score_left"]), .01)

    def test_static_baseline_and_exact_multirate_expansion(self) -> None:
        engine = StereoFlyVis(4, n_t4t5=2, blur_sigma_px=0, backend="stub")
        one = engine.process_frame(np.full(4, .5))
        two = engine.process_frame(np.full(4, .5))
        self.assertEqual((one.cns_tick_start, one.cns_ticks), (0, CNS_TICKS_PER_FLYVIS))
        self.assertEqual((two.cns_tick_start, two.cns_ticks), (10, CNS_TICKS_PER_FLYVIS))
        np.testing.assert_array_equal(one.drive, 0)
        np.testing.assert_array_equal(two.drive, 0)
        expanded = engine.expand_drive_to_cns(np.array([-.25, .5], dtype=np.float32))
        self.assertEqual(expanded.shape, (CNS_TICKS_PER_FLYVIS, 2))
        np.testing.assert_array_equal(expanded[:, 0], -.25)
        self.assertAlmostEqual(FLYVIS_DT, CNS_DT * CNS_TICKS_PER_FLYVIS)

    def test_direction_and_binocular_metrics(self) -> None:
        positive = np.array([-.3, .2, .7, -1.1], dtype=np.float32)
        negative = -positive
        d = direction_selectivity(positive, negative)
        self.assertTrue(d["direction_selective"])
        self.assertLess(d["direction_correlation"], -.99)
        c = left_right_common_mode(positive, positive)
        self.assertGreater(c["left_right_common_mode_ratio"], 1e6)

    def test_shape_and_timestamp_errors_do_not_consume_state(self) -> None:
        engine = StereoFlyVis(4, blur_sigma_px=0)
        with self.assertRaises(ShapeError):
            engine.process_frame(np.ones((4, 1, 1), dtype=np.float32))
        with self.assertRaises(ValueError):
            engine.process_frame(np.ones(4, dtype=np.float32), timestamp_s=.001)
        self.assertEqual(engine._frame_index, 0)
        self.assertIsNone(engine._baseline_left)

    def test_recorded_backend_and_contract_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trace.npz"
            np.savez(path, responses=np.array([[.2, -.4], [.3, -.5]], dtype=np.float32))
            engine = StereoFlyVis(4, backend=path, blur_sigma_px=0)
            out = engine.process_frame(np.full(4, .5))
            self.assertFalse(out.is_stub)
            self.assertTrue(out.is_recorded)
            np.testing.assert_allclose(out.drive, [.2, -.4])
            self.assertEqual(out.to_visual_observation().pixel_shape, (2, 4))
            self.assertEqual(out.to_sensory_frame().timestamp, out.cns_timestamp())

    def test_visual_observation_input_uses_its_timestamp(self) -> None:
        engine = StereoFlyVis(4, blur_sigma_px=0)
        observation = VisualObservation(
            Timestamp(0.0, 0, "neural", 2000.0),
            pixels=np.full((2, 2), .5, dtype=np.float32),
        )
        output = engine.process_frame(observation)
        self.assertEqual(output.cns_timestamp(), observation.timestamp)


if __name__ == "__main__":
    unittest.main()
