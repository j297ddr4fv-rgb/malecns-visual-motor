#!/usr/bin/env python3
"""Dependency-free tests for the flight GPU profiler.

These tests only exercise clocks, temporary JSON, and existing artifact
metadata.  They never import a model runtime or launch a CUDA process.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import profile_flight_pipeline as profiler  # noqa: E402


class FakeClock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class ProfileFlightPipelineTest(unittest.TestCase):
    def test_import_is_dependency_light(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r); "
            "import profile_flight_pipeline; "
            "assert 'torch' not in sys.modules; "
            "assert 'flyvis' not in sys.modules"
        ) % str(ROOT / "scripts")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fitted_profile_uses_exact_column_runner_seam(self) -> None:
        runner = SimpleNamespace()
        runner._process_exact_flyvis_columns = Mock(return_value="exact-output")
        runner.flyvis = SimpleNamespace(
            process_frame=Mock(side_effect=AssertionError("raster frontend used"))
        )

        result = profiler._profile_exact_flyvis_step(runner, "left", "right", 0.005)

        self.assertEqual(result, "exact-output")
        runner._process_exact_flyvis_columns.assert_called_once_with(
            "left", "right", 0.005,
        )
        runner.flyvis.process_frame.assert_not_called()

    def test_profile_serialization_record_is_hash_only_and_json_safe(self) -> None:
        timestamp = SimpleNamespace(to_dict=lambda: {"t_s": 0.0})
        output = SimpleNamespace(
            left_columns=b"left", right_columns=b"right", t4t5_drive=b"drive",
        )
        neural = SimpleNamespace(named_outputs={"spike_count": 3, "backend": "mock"})
        state = SimpleNamespace(to_dict=lambda: {"position_m": [0.0, 0.0, 0.0]})
        record = profiler._profile_compact_record(
            0, timestamp, output, neural, {"thrust": 0.0},
            SimpleNamespace(state=state), SimpleNamespace(state=state),
        )

        self.assertEqual(
            record["visual_input"]["left_columns"]["encoding"], "hash-only",
        )
        self.assertEqual(record["neural_window"]["spike_count"], 3)
        json.dumps(record, allow_nan=False)

    def test_gpu_profile_reports_failed_frame_and_completed_count(self) -> None:
        class Timestamp:
            def __init__(self, t_s: float):
                self.t_s = t_s

            def to_dict(self):
                return {"t_s": self.t_s}

        class State:
            def copy(self):
                return self

            def to_dict(self):
                return {"position_m": [0.0, 0.0, 0.0]}

        class Dynamics:
            def __init__(self):
                self.state = State()
                self.calls = 0

            def step(self, command, *, dt_s):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("body failure")
                return SimpleNamespace(state=self.state)

        class World:
            renderer_backend = "analytic_reference"
            camera = object()

            def __init__(self):
                self.scenarios = {"hover": object()}

            def _ray(self, pose, eye, px, py, camera):
                return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)

            def trace_ray(self, origin, direction, *, timestamp_s):
                return 0.0

            def render_stereo(self, pose, resolution, *, timestamp_s):
                return SimpleNamespace(left=[0.0], right=[0.0], eye_separation_m=0.001)

            def visual_observation(self, stereo, t_s, *, dt_s):
                return SimpleNamespace(optic_flow_rad_s=(0.0, 0.0, 0.0))

        class FlyVis:
            def __init__(self):
                self.process_frame = Mock(side_effect=AssertionError("raster frontend used"))

            def reset(self):
                return None

        class Neural:
            def step_window(self, sensory, timestamp, *, steps):
                return SimpleNamespace(
                    named_outputs={"spike_count": 0, "backend": "mock"},
                )

        class Neuromuscular:
            def reset(self, *, clear_baseline):
                return None

            def step(self, neural_output, *, dt_ms, timestamp):
                return SimpleNamespace(
                    left_wing_power=0.0,
                    right_wing_power=0.0,
                    wing_power=0.0,
                    wing_steering=0.0,
                )

        exact_calls = []

        class Runner:
            def __init__(self, options):
                self.world = World()
                self.flyvis = FlyVis()
                self.flyvis_runtime_adapter = SimpleNamespace(backend_name="mock-flyvis")
                self.neuromuscular = Neuromuscular()
                self.dynamics = Dynamics()

            def _new_body(self, scenario):
                return self.dynamics

            def _new_neural(self, scenario):
                return Neural()

            def _free_exact_flyvis_columns(self, pose, t_s):
                return [0.0], [0.0], {}

            def _process_exact_flyvis_columns(self, left, right, t_s):
                exact_calls.append(t_s)
                if len(exact_calls) == 2:
                    raise RuntimeError("exact seam failure")
                return SimpleNamespace(
                    left_columns=b"left",
                    right_columns=b"right",
                    t4t5_drive=b"drive",
                )

        class RunOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        launcher = SimpleNamespace(
            RunOptions=RunOptions,
            _timestamp=lambda t_s: Timestamp(t_s),
            _pose_from_state=lambda state: object(),
            _make_contract_body=lambda state, timestamp: SimpleNamespace(),
            _flyvis_drive_contract=lambda flyvis, drive: {},
            SensoryFrame=lambda **kwargs: SimpleNamespace(**kwargs),
            SourceLabel=SimpleNamespace(HANDBUILT="handbuilt"),
        )
        config = profiler.GPUProfileConfig(
            frames=2, warmup_frames=0, width=1, height=1,
            vision_device="cpu", neural_backend="cpu",
        )
        with patch.object(profiler, "_profile_gpu_runner_imports", return_value=(launcher, Runner)), \
             patch.object(profiler, "_runtime_memory_snapshot", return_value={}):
            with self.assertRaisesRegex(
                profiler.ProfileError,
                r"failed at frame 1 \(completed 1/2\).*exact seam failure",
            ):
                profiler.run_gpu_profile(config)
        self.assertEqual(exact_calls, [0.0, 0.005])

    def test_stage_profiler_records_success_and_exception(self) -> None:
        clock = FakeClock(1.0, 1.125, 2.0, 2.250)
        hook = profiler.StageProfiler(clock=clock)
        with hook.measure("flyvis", sim_s=0.005, count=1, batch_size=2):
            pass
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with hook.measure("malecns", sim_s=0.005, count=10):
                raise RuntimeError("expected")
        stats = hook.stage_stats()
        self.assertAlmostEqual(stats["flyvis"].wall_values[0], 0.125)
        self.assertAlmostEqual(stats["malecns"].wall_values[0], 0.250)
        self.assertEqual(stats["malecns"].total_count, 10)

    def test_stage_profiler_marks_warmup_and_synchronizes_only_at_boundaries(self) -> None:
        clock = FakeClock(0.0, 0.2, 1.0, 1.1)
        sync_calls = []
        hook = profiler.StageProfiler(clock=clock, synchronize=lambda: sync_calls.append("sync"))
        with hook.measure("flyvis", sim_s=0.005, count=1, phase="warmup"):
            # A stage body must not need to know about CUDA synchronization.
            pass
        with hook.measure("flyvis", sim_s=0.005, count=1, phase="steady"):
            pass
        stats = hook.stage_stats()["flyvis"]
        self.assertEqual(sync_calls, ["sync", "sync", "sync", "sync"])
        self.assertEqual(hook.synchronize_calls, 4)
        self.assertEqual(stats.warmup.samples[0].phase, "warmup")
        self.assertEqual(stats.steady.samples[0].phase, "steady")
        self.assertAlmostEqual(stats.steady.real_time_factor, 0.05)
        self.assertEqual(hook.to_dict()["synchronize_calls"], 4)

    def test_wrap_preserves_return_value(self) -> None:
        hook = profiler.StageProfiler(clock=FakeClock(0.0, 0.5))

        @hook.wrap("body", count=1)
        def add(left: int, right: int) -> int:
            return left + right

        self.assertEqual(add(2, 3), 5)
        self.assertAlmostEqual(hook.stage_stats()["body"].wall_values[0], 0.5)

    def test_batch_probe_does_not_time_generator_consumer_pause(self) -> None:
        # Two batches: the second clock interval is deliberately large.  The
        # values are consumed by ``fn`` before the result is yielded.
        hook = profiler.StageProfiler(clock=FakeClock(0.0, 0.1, 10.0, 10.2))
        probe = profiler.BatchProbe(hook, "malecns")
        values = list(probe.run(([1, 2], [3]), lambda batch: sum(batch)))
        self.assertEqual(values, [3, 3])
        self.assertEqual(hook.stage_stats()["malecns"].total_count, 3)
        self.assertAlmostEqual(hook.stage_stats()["malecns"].wall_values[0], 0.1)
        self.assertAlmostEqual(hook.stage_stats()["malecns"].wall_values[1], 0.2)

    def test_iter_batches_is_bounded_and_handles_remainder(self) -> None:
        self.assertEqual(list(profiler.iter_batches(range(5), 2)), [[0, 1], [2, 3], [4]])
        with self.assertRaises(ValueError):
            list(profiler.iter_batches(range(2), 0))

    def test_cache_probe_reports_hits_misses_and_evictions(self) -> None:
        clock = FakeClock(0.0, 0.1, 1.0, 1.2, 2.0, 2.3, 3.0, 3.4)
        hook = profiler.StageProfiler(clock=clock)
        cache = profiler.CacheProbe(max_entries=1, profiler=hook)
        self.assertEqual(cache.get_or_compute("a", lambda: 4, size_bytes=8), 4)
        self.assertEqual(cache.get_or_compute("a", lambda: 5, size_bytes=8), 4)
        self.assertEqual(cache.get_or_compute("b", lambda: 6, size_bytes=16), 6)
        # ``a`` was evicted by ``b`` only after its hit moved it to the MRU
        # end; the third call is a miss and replaces it.
        self.assertEqual(cache.get_or_compute("a", lambda: 7, size_bytes=8), 7)
        self.assertEqual(cache.stats.requests, 4)
        self.assertEqual(cache.stats.hits, 1)
        self.assertEqual(cache.stats.misses, 3)
        self.assertEqual(cache.stats.evictions, 2)
        self.assertEqual(cache.stats.hit_rate, 0.25)
        self.assertEqual(hook.stage_stats()["cache_lookup"].cache_hits, 1)
        self.assertEqual(hook.stage_stats()["cache_lookup"].cache_misses, 3)

    def test_fit_stage_uses_robust_startup_plus_throughput_model(self) -> None:
        rows = profiler.StageStats("malecns", (
            profiler.StageSample("malecns", 0.20, sim_s=1.0),
            profiler.StageSample("malecns", 0.40, sim_s=2.0),
            # One cold/outlier observation should not dominate the estimate.
            profiler.StageSample("malecns", 2.00, sim_s=0.01),
        ))
        estimate = profiler.fit_stage(rows, target_sim_s=3.0)
        self.assertEqual(estimate.axis, "sim_s")
        self.assertEqual(estimate.confidence, "measured-fit")
        self.assertIsNotNone(estimate.projected_s)
        self.assertLess(float(estimate.projected_s), 1.0)

    def test_artifact_reader_distinguishes_neural_and_legacy_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text(json.dumps({
                "backend": "triton",
                "graph": "/data/cache/malecns-gpu/csr-flyb",
                "neurons": 176422,
                "edges": 6287749,
                "dt_ms": 0.5,
                "simulated_ms": 1000.0,
                "wall_s": 0.20,
            }), encoding="utf-8")
            (root / "flight.json").write_text(json.dumps({
                "mode": "fly", "frames": 400, "frame_ms": 5.0, "wall_s": 58.0,
            }), encoding="utf-8")
            report = profiler.profile_artifacts([root])
        self.assertEqual(set(report.stage_stats()), {"legacy_closed_loop", "malecns_open_loop"})
        projection = report.projection(1.0)
        self.assertEqual(projection["malecns_steps"], 2000)
        self.assertEqual(projection["status"], "legacy-proxy-plus-neural-lower-bound")
        self.assertAlmostEqual(projection["neural_lower_bound_projected_s"], 0.20)
        self.assertAlmostEqual(projection["legacy_proxy_projected_s"], 29.0)

    def test_explicit_stage_timings_can_project_a_sum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps({
                "stage_timings": [
                    {"stage": "world", "wall_ms": 1.0, "sim_s": 0.005},
                    {"stage": "flyvis", "wall_ms": 2.0, "sim_s": 0.005},
                    {"stage": "malecns", "wall_ms": 3.0, "sim_s": 0.005},
                    {"stage": "body", "wall_ms": 1.0, "sim_s": 0.005},
                ]
            }), encoding="utf-8")
            report = profiler.profile_artifacts([path])
        projection = report.projection(0.010)
        self.assertEqual(projection["status"], "stage-sum-fit")
        self.assertAlmostEqual(projection["explicit_stage_sum_projected_s"], 0.014)
        self.assertEqual(projection["flyvis_frames"], 2)
        self.assertEqual(projection["malecns_steps"], 20)

    def test_named_timing_mapping_and_jsonl_nested_timing_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "profile.json").write_text(json.dumps({
                "timings": {
                    "world": {"wall_ms": 1.0, "sim_s": 0.005},
                    "flyvis": {"wall_ms": 2.0, "sim_s": 0.005},
                }
            }), encoding="utf-8")
            (root / "telemetry.jsonl").write_text(json.dumps({
                "profile": {"stage": "body", "wall_ms": 3.0, "sim_s": 0.005}
            }) + "\n", encoding="utf-8")
            report = profiler.profile_artifacts([root], include_telemetry=True)
        self.assertEqual(set(report.stage_stats()), {"body", "flyvis", "world"})
        self.assertEqual(len(report.samples), 3)

    def test_real_smoke_manifest_is_read_without_stage_execution(self) -> None:
        path = ROOT / "artifacts" / "gpu-e2e-smoke-final-20260921"
        if not path.exists():
            self.skipTest("real GPU artifact is not present")
        report = profiler.profile_artifacts([path])
        self.assertEqual(report.samples, [])
        memory = report.memory_summary()
        self.assertEqual(memory["ram_limit_gib"], 5.0)
        self.assertAlmostEqual(memory["peak_host_rss_gib"], 1.6393318176269531)
        self.assertAlmostEqual(memory["peak_vram_reserved_gib"], 2.734375)
        self.assertTrue(memory["within_budget"])
        self.assertEqual(report.projection(0.005)["status"], "unmeasured")

    def test_smoke_ladder_keeps_ten_cns_steps_per_visual_frame(self) -> None:
        ladder = profiler.recommended_smoke_ladder()
        self.assertEqual(ladder[1]["name"], "one-frame")
        for row in ladder[1:]:
            self.assertEqual(row["malecns_steps"], row["flyvis_frames"] * 10)
        self.assertEqual(ladder[-1]["simulated_s"], 10.0)

    def test_gpu_profile_config_is_bounded_and_splits_render_paths(self) -> None:
        config = profiler.GPUProfileConfig()
        self.assertEqual(config.frames, 2)
        self.assertEqual(config.warmup_frames, 1)
        self.assertEqual(config.renderer_backend, "analytic_reference")
        self.assertEqual(config.to_dict()["rectangular_rays_per_frame"], 1536)
        self.assertEqual(config.to_dict()["exact_rays_per_frame"], 1442)
        self.assertEqual(config.to_dict()["combined_diagnostic_rays_per_frame"], 2978)
        with self.assertRaises(profiler.ProfileError):
            profiler.GPUProfileConfig(frames=1, warmup_frames=2)
        with self.assertRaises(profiler.ProfileError):
            profiler.GPUProfileConfig(renderer_backend="silent_fallback")

    def test_report_records_requested_and_effective_renderer_backend(self) -> None:
        report = profiler.ProfileReport(facts={
            "renderer": {
                "requested_backend": "warp_cuda",
                "effective_backend": "warp_cuda",
                "timing": {
                    "cold_jit": {"wall_s": 1.16},
                    "steady_state_renderer_stages": {},
                },
            },
        })
        payload = report.to_dict(duration_s=0.0)
        self.assertEqual(payload["renderer_backend"], {
            "requested": "warp_cuda", "effective": "warp_cuda",
        })
        self.assertEqual(payload["renderer"]["timing"]["cold_jit"]["wall_s"], 1.16)
        self.assertEqual(payload["warp_cold_jit"]["wall_s"], 1.16)

    def test_profile_artifact_preserves_renderer_cold_jit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile-gpu-warp.json"
            report = profiler.ProfileReport(
                samples=[profiler.StageSample(
                    "exact_column_render", 1.16, sim_s=0.005,
                    metadata={
                        "renderer_backend": "warp_cuda",
                        "cold_jit_candidate": True,
                    },
                )],
                facts={"renderer": {
                    "requested_backend": "warp_cuda",
                    "effective_backend": "warp_cuda",
                }},
            )
            path.write_text(json.dumps(report.to_dict(duration_s=0.005)), encoding="utf-8")
            parsed = profiler.profile_artifacts([path])
        sample = parsed.samples[0]
        self.assertEqual(sample.metadata["renderer_backend"], "warp_cuda")
        self.assertTrue(sample.metadata["cold_jit_candidate"])

    def test_report_projection_exposes_throughput_and_phase_breakdown(self) -> None:
        report = profiler.ProfileReport(samples=[
            profiler.StageSample("malecns_10tick_window", 0.2, sim_s=0.005, count=10, phase="warmup"),
            profiler.StageSample("malecns_10tick_window", 0.01, sim_s=0.005, count=10, phase="steady"),
        ])
        projection = report.projection(0.005)
        stage = projection["stage_estimates"]["malecns_10tick_window"]
        self.assertIn("steady_throughput_per_s", stage)
        self.assertIn("real_time_factor", stage)
        self.assertIn("throughput", projection)
        self.assertIn("steady_state", projection)
        self.assertIn("projected_s", projection["steady_state"])
        self.assertGreater(projection["throughput"]["real_time_factor"], 0.0)

    def test_profile_report_reader_does_not_promote_setup_metadata_to_a_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile-gpu.json"
            report = profiler.ProfileReport(
                samples=[profiler.StageSample("flyvis", 0.01, sim_s=0.005)],
                facts={"profile_setup": {"wall_s": 99.0}, "resource_observations": []},
            )
            path.write_text(json.dumps(report.to_dict(duration_s=0.005)), encoding="utf-8")
            parsed = profiler.profile_artifacts([path])
        self.assertEqual(set(parsed.stage_stats()), {"flyvis"})

    def test_diagnostic_render_passes_are_not_added_to_exact_pipeline_sum(self) -> None:
        report = profiler.ProfileReport(samples=[
            profiler.StageSample("world_raycast", 1.0, sim_s=0.005),
            profiler.StageSample("rectangular_stereo_render", 2.0, sim_s=0.005),
            profiler.StageSample("exact_column_render", 3.0, sim_s=0.005),
            profiler.StageSample("flyvis", 4.0, sim_s=0.005),
        ])
        projection = report.projection(0.005)
        self.assertEqual(projection["non_additive_stage_names"], [
            "rectangular_stereo_render", "world_raycast",
        ])
        self.assertAlmostEqual(projection["recommended_projected_s"], 7.0)


if __name__ == "__main__":
    unittest.main()
