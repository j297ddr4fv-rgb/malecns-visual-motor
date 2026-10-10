import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_resident_flight import (  # noqa: E402
    ResidentBenchmarkError,
    _scene_identity,
    acceptance_record,
    aligned_duration,
    build_production_options,
    parse_args,
    run_resident_benchmark,
    write_json,
)


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.25
        return self.value


class _Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.duration_override_s = None
        self.frame_count = 99
        self.max_steps = 99
        self.out = None


class _Runner:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1
        self.options = _Options()
        self.runs = []
        self.writes = []

    def run(self):
        duration = float(self.options.duration_override_s)
        self.runs.append(duration)
        frame_count = int(round(duration / 0.005))
        return {
            "manifest": {"scenario_results": {"probe": {"duration_s": duration}}},
            "trajectory": {
                "trajectory": [
                    {"time_s": index * 0.005} for index in range(frame_count)
                ]
            },
        }

    def write_outputs(self, payload):
        self.writes.append(float(payload["manifest"]["scenario_results"]["probe"]["duration_s"]))
        return {"manifest": str(Path(self.options.out) / "manifest.json")}


class _OptionsFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ResidentBenchmarkTest(unittest.TestCase):
    def test_duration_alignment_and_rejection(self):
        self.assertEqual(aligned_duration(0.5, name="warmup"), 0.5)
        self.assertEqual(aligned_duration(0.0, name="warmup", allow_zero=True), 0.0)
        with self.assertRaises(ResidentBenchmarkError):
            aligned_duration(0.501, name="measurement")

    def test_constructs_once_and_separates_timing(self):
        _Runner.constructions = 0
        runner = None

        def factory():
            nonlocal runner
            runner = _Runner()
            return runner

        with tempfile.TemporaryDirectory() as temp_dir:
            record = run_resident_benchmark(
                factory,
                warmup_duration_s=0.5,
                measurement_duration_s=1.0,
                out=Path(temp_dir),
                clock=_Clock(),
            )
        self.assertEqual(_Runner.constructions, 1)
        self.assertEqual(runner.runs, [0.5, 1.0])
        self.assertEqual(runner.writes, [1.0])
        self.assertTrue(record["runner_constructed_once"])
        self.assertEqual(record["measurement"]["simulated_duration_s"], 1.0)
        self.assertGreater(record["measurement"]["simulation_plus_write_wall_s"], 0.0)
        self.assertEqual(record["measurement"]["rtf_simulation_plus_write"], 0.8)
        self.assertEqual(runner.options.frame_count, None)
        self.assertEqual(runner.options.max_steps, None)

    def test_optional_acceptance_record_has_explicit_scope(self):
        record = {
            "output_root": "/tmp/resident",
            "benchmark_path": "/tmp/resident/resident_benchmark.json",
            "scene_identity": {
                "manifest_path": "/tmp/scene.json",
                "scene_id": "scene-test",
                "manifest_sha256": "a" * 64,
            },
            "runner_constructed_once": True,
            "setup_wall_s": 12.0,
            "measurement": {
                "simulated_duration_s": 10.0,
                "simulation_wall_s": 78.0,
                "write_wall_s": 2.0,
                "simulation_plus_write_wall_s": 80.0,
                "rtf_simulation_plus_write": 0.125,
                "synchronization": {"before": True, "after": True},
            },
        }
        run_manifest = {
            "requested_execution": {
                "renderer_backend": "warp_cuda",
                "neural_backend": "triton",
                "neural_device": "cuda",
                "neural_cuda_graph": True,
                "vision_device": "cuda:0",
            },
            "requested_backends": {"vision": "flyvis"},
            "effective_execution": {
                "renderer_backend": "warp_cuda",
                "neural_backend": "triton",
                "neural_device": "cuda",
                "neural_cuda_graph": {"requested": True, "effective": True},
                "vision_device": "cuda:0",
            },
            "effective_backends": {"vision": "flyvis"},
            "components": {
                name: {"is_stub": False}
                for name in ("world", "vision", "neural", "body")
            },
        }
        acceptance = acceptance_record(record, run_manifest=run_manifest)
        self.assertEqual(acceptance["type"], "full_run_performance_acceptance")
        self.assertEqual(
            acceptance["wall_time_scope"], "resident-runner simulation+output"
        )
        self.assertTrue(acceptance["setup_wall_excluded"])
        self.assertEqual(acceptance["wall_time_s"], 80.0)
        self.assertEqual(acceptance["real_time_factor"], 0.125)
        self.assertEqual(
            acceptance["source_benchmark"], "/tmp/resident/resident_benchmark.json"
        )
        self.assertEqual(acceptance["scene_identity"], record["scene_identity"])
        self.assertEqual(acceptance["requested_execution"]["vision_backend"], "flyvis")
        self.assertTrue(acceptance["effective_execution"]["neural_cuda_graph"])
        self.assertFalse(acceptance["components"]["neural"]["is_stub"])

    def test_explicit_synchronization_surrounds_measured_gpu_work(self):
        events = []

        class Runner(_Runner):
            def run(self):
                events.append("run")
                return super().run()

            def write_outputs(self, payload):
                events.append("write")
                return super().write_outputs(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            record = run_resident_benchmark(
                Runner,
                warmup_duration_s=0.0,
                measurement_duration_s=1.0,
                out=Path(temp_dir),
                clock=_Clock(),
                synchronize=lambda: events.append("sync"),
            )
        self.assertEqual(events, ["sync", "sync", "run", "sync", "write", "sync"])
        self.assertEqual(
            record["measurement"]["synchronization"],
            {
                "before": True,
                "after": True,
                "scope": "CUDA work drained around the measured simulation and output window",
            },
        )

    def test_rejects_short_or_misaligned_runner_output(self):
        class ShortRunner(_Runner):
            def run(self):
                return {
                    "manifest": {"scenario_results": {"probe": {"duration_s": 0.5}}},
                    "trajectory": {"trajectory": [{"time_s": 0.495}]},
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ResidentBenchmarkError, "simulated 0.5s"):
                run_resident_benchmark(
                    ShortRunner,
                    warmup_duration_s=0.0,
                    measurement_duration_s=1.0,
                    out=Path(temp_dir),
                    clock=_Clock(),
                )

    def test_production_options_lock_requested_backends(self):
        options = build_production_options(
            _OptionsFactory,
            out=Path("/tmp/out"),
            config=Path("/tmp/escape.json"),
            seed=7,
            max_vram_gib=15.5,
        )
        self.assertEqual(options.kwargs["neural"], "malecns")
        self.assertEqual(options.kwargs["vision"], "flyvis")
        self.assertEqual(options.kwargs["renderer_backend"], "warp_cuda")
        self.assertEqual(options.kwargs["neural_backend"], "triton")
        self.assertTrue(options.kwargs["neural_cuda_graph"])
        self.assertEqual(options.kwargs["telemetry_mode"], "minimal")
        self.assertFalse(options.kwargs["parallel_frame_stages"])

        candidate = build_production_options(
            _OptionsFactory,
            out=Path("/tmp/out"),
            config=Path("/tmp/escape.json"),
            seed=7,
            max_vram_gib=15.5,
            parallel_frame_stages=True,
        )
        self.assertTrue(candidate.kwargs["parallel_frame_stages"])

    def test_optional_scene_manifest_is_parsed_and_forwarded(self):
        self.assertIsNone(parse_args([]).scene_manifest)
        scene_path = Path("config/realistic_scene.json")
        args = parse_args(["--scene-manifest", str(scene_path)])
        self.assertEqual(args.scene_manifest, scene_path)

        default_options = build_production_options(
            _OptionsFactory,
            out=Path("/tmp/out"),
            config=Path("/tmp/escape.json"),
            seed=7,
            max_vram_gib=15.5,
        )
        self.assertIsNone(default_options.kwargs["scene_manifest"])

        selected_options = build_production_options(
            _OptionsFactory,
            out=Path("/tmp/out"),
            config=Path("/tmp/escape.json"),
            seed=7,
            max_vram_gib=15.5,
            scene_manifest=args.scene_manifest,
        )
        self.assertEqual(selected_options.kwargs["scene_manifest"], scene_path)

    def test_scene_identity_records_manifest_id_and_content_hash(self):
        raw_manifest = b'{"scene_id":"scene-test","schema_version":1}\n'
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "scene.json"
            manifest_path.write_bytes(raw_manifest)

            identity = _scene_identity(manifest_path)

        self.assertEqual(identity["manifest_path"], str(manifest_path.resolve()))
        self.assertEqual(identity["scene_id"], "scene-test")
        self.assertEqual(
            identity["manifest_sha256"], hashlib.sha256(raw_manifest).hexdigest()
        )
        self.assertIsNone(_scene_identity(None))

    def test_parallel_candidate_requires_explicit_benchmark_flag(self):
        self.assertFalse(parse_args([]).parallel_frame_stages)
        self.assertTrue(
            parse_args(["--parallel-frame-stages"]).parallel_frame_stages
        )

    def test_write_json_is_valid_and_creates_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "nested" / "record.json"
            path = write_json(destination, {"value": 1})
            self.assertEqual(Path(path), destination)
            self.assertEqual(json.loads(destination.read_text()), {"value": 1})


if __name__ == "__main__":
    unittest.main()
