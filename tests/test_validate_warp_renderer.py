#!/usr/bin/env python3
"""CPU-only contract tests for ``validate_warp_renderer``.

The real Warp package is intentionally never imported here.  Fakes exercise
the exact bilateral seam, dynamic timestamps, output contract, provenance,
bounded benchmark, and explicit artifact behavior.  A GPU smoke run belongs
to a separate operator invocation once ``world_renderer_opt`` is available.
"""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_warp_renderer as validator  # noqa: E402


class FakeWorld:
    backend_role = "analytic_reference"
    stimulus_names = ("moving",)

    @staticmethod
    def render_eye_directions(_pose, directions, eye="left", *, timestamp_s=0.0, stimulus=None, **_kwargs):
        values = np.asarray(directions, dtype=np.float32)[:, 1]
        eye_offset = 0.03 if eye == "right" else 0.0
        dynamic_offset = 0.10 * float(timestamp_s) if stimulus is not None else 0.0
        # Keep the fixture in normalized luminance range for every timestamp.
        return np.clip(0.42 + 0.06 * values + eye_offset + dynamic_offset, 0.0, 1.0).astype(np.float32)


class TrackingWorld(FakeWorld):
    stimulus_names = ("moving_a", "moving_b")

    def __init__(self) -> None:
        self.stimuli_calls: list[object] = []

    def render_eye_directions(self, pose, directions, eye="left", *, timestamp_s=0.0, stimulus=None, stimuli=None, **kwargs):  # type: ignore[no-untyped-def]
        self.stimuli_calls.append(stimuli)
        return super().render_eye_directions(pose, directions, eye, timestamp_s=timestamp_s, stimulus=stimulus, stimuli=stimuli, **kwargs)


class FakeRenderer:
    backend_name = "warp_cuda"
    requested_backend = "warp_cuda"
    effective_backend = "warp_cuda"
    provenance = {"source": "fake-warp", "fallback": False}

    @staticmethod
    def memory_stats():
        return {"cuda": {"allocated_bytes": 1, "reserved_bytes": 2,
                          "max_allocated_bytes": 1, "max_reserved_bytes": 2}}

    def render_eye_directions(self, _pose, directions, eye="left", *, timestamp_s=0.0, stimulus=None, **_kwargs):
        return FakeWorld.render_eye_directions(_pose, directions, eye, timestamp_s=timestamp_s, stimulus=stimulus)


class NonDeterministicRenderer(FakeRenderer):
    def __init__(self) -> None:
        self.calls = 0

    def render_eye_directions(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        output = super().render_eye_directions(*args, **kwargs)
        if self.calls % 2 == 0:
            output = output.copy()
            output[0] += 0.02
        return output


class BadContractRenderer(FakeRenderer):
    def render_eye_directions(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return np.array([1.25], dtype=np.float32)


class ValidateWarpRendererTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = FakeWorld()
        self.pose = object()
        self.directions = np.asarray([
            [1.0, -0.2, 0.1],
            [1.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
        ], dtype=np.float32)
        self.module = types.ModuleType("fake_world_renderer_opt")
        self.module.backend_name = "warp_cuda"

    def test_import_is_safe_without_loading_warp(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r); import validate_warp_renderer; "
            "assert 'warp' not in sys.modules"
        ) % str(SCRIPTS)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_array_comparison_reports_max_abs_rms_and_mismatches(self) -> None:
        metrics = validator.compare_arrays(
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.6, 1.0], dtype=np.float32),
            abs_tol=1.0e-3,
            rel_tol=0.0,
        )
        self.assertEqual(metrics["mismatch_count"], 1)
        self.assertAlmostEqual(metrics["max_abs"], 0.1, places=6)
        self.assertAlmostEqual(metrics["rms"], (0.01 / 3.0) ** 0.5, places=6)
        self.assertEqual(metrics["status"], "FAIL")

    def test_direction_loaders_accept_npy_and_bilateral_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npy = root / "directions.npy"
            np.save(npy, self.directions)
            loaded = validator.load_directions(npy)
            np.testing.assert_array_equal(loaded.for_eye("left"), self.directions)
            np.testing.assert_array_equal(loaded.for_eye("right"), self.directions)

            payload = {"left": self.directions.tolist(), "right": (-self.directions).tolist()}
            json_path = root / "directions.json"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            bilateral = validator.load_directions(json_path)
            np.testing.assert_array_equal(bilateral.for_eye("left"), self.directions)
            np.testing.assert_array_equal(bilateral.for_eye("right"), -self.directions)

            packed = root / "bilateral.npy"
            np.save(packed, np.stack([self.directions, -self.directions], axis=0))
            packed_directions = validator.load_directions(packed)
            np.testing.assert_array_equal(packed_directions.for_eye("left"), self.directions)
            np.testing.assert_array_equal(packed_directions.for_eye("right"), -self.directions)
            self.assertEqual(packed_directions.source, "file")

    def test_validate_objects_covers_bilateral_static_dynamic_and_benchmark(self) -> None:
        created: list[FakeRenderer] = []

        def factory() -> FakeRenderer:
            item = FakeRenderer()
            created.append(item)
            return item

        report = validator.validate_objects(
            world=self.world,
            renderer=FakeRenderer(),
            module=self.module,
            renderer_factory=factory,
            directions=self.directions,
            pose=self.pose,
            static_timestamps=(0.0,),
            dynamic_timestamps=(0.25, 0.75),
            warmup=1,
            iterations=2,
        )
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["backend"]["requested"], "warp_cuda")
        self.assertEqual(report["backend"]["effective"], "warp_cuda")
        self.assertEqual(report["directions"]["left_count"], 3)
        self.assertEqual(report["directions"]["right_count"], 3)
        self.assertEqual(len(report["cases"]), 3)
        self.assertTrue(all(case["status"] == "PASS" for case in report["cases"]))
        self.assertEqual(report["summary"]["mismatch_count"], 0)
        self.assertEqual(report["benchmark"]["steady"]["iterations"], 2)
        self.assertEqual(len(report["benchmark"]["steady"]["samples_s"]), 2)
        self.assertIsNotNone(report["benchmark"]["cold_setup"]["wall_s"])
        self.assertEqual(len(created), 1)

    def test_static_is_explicitly_empty_and_overlap_is_non_core(self) -> None:
        world = TrackingWorld()
        report = validator.validate_objects(
            world=world,
            renderer=FakeRenderer(),
            module=self.module,
            directions=self.directions,
            pose=self.pose,
            dynamic=False,
            warmup=0,
            iterations=1,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertIn((), world.stimuli_calls)
        self.assertEqual(report["all_configured_overlap"]["status"], "NON_CORE_KNOWN_TIE")
        self.assertFalse(report["all_configured_overlap"]["core"])

    def test_shape_dtype_range_and_determinism_failures_are_visible(self) -> None:
        bad = validator.validate_objects(
            world=self.world,
            renderer=BadContractRenderer(),
            module=self.module,
            directions=self.directions,
            pose=self.pose,
            dynamic=False,
            warmup=0,
            iterations=1,
        )
        self.assertEqual(bad["status"], "FAIL")
        eye = bad["cases"][0]["eyes"]["left"]
        self.assertEqual(eye["candidate_contract"]["status"], "FAIL")
        self.assertFalse(eye["candidate_contract"]["range"]["valid"])

        nondeterministic = validator.validate_objects(
            world=self.world,
            renderer=NonDeterministicRenderer(),
            module=self.module,
            directions=self.directions,
            pose=self.pose,
            dynamic=False,
            warmup=0,
            iterations=1,
        )
        self.assertEqual(nondeterministic["status"], "FAIL")
        self.assertEqual(nondeterministic["cases"][0]["eyes"]["left"]["determinism"]["status"], "FAIL")

    def test_artifact_is_only_created_when_explicitly_requested(self) -> None:
        report = validator.validate_objects(
            world=self.world,
            renderer=FakeRenderer(),
            module=self.module,
            directions=self.directions,
            pose=self.pose,
            dynamic=False,
            warmup=0,
            iterations=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "warp-validation.json"
            self.assertFalse(target.exists())
            validator.write_artifact(report, target)
            self.assertTrue(target.is_file())
            saved = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "warp-renderer-validation-1")

    def test_missing_backend_fails_clearly_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "should-not-exist.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "validate_warp_renderer.py"),
                 "--backend-module", "module_that_does_not_exist_for_warp_validation",
                 "--artifact", str(artifact)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Warp validation unavailable", result.stderr)
            self.assertIn("module_that_does_not_exist_for_warp_validation", result.stderr)
            self.assertFalse(artifact.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
