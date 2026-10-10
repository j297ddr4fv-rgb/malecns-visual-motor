from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_courtship_visual_suite as suite  # noqa: E402

CAMERA = {"near_m": 0.0003, "eye_separation_m": 0.0007, "far_m": 0.5}


def _manifest(*, condition: str = "moving", movie: bool = False) -> dict:
    config = ROOT / "config" / "courtship_visual_experiment.json"
    stimulus = ROOT / "config" / "courtship_visual_stimuli.json"
    trajectory = ROOT / "config" / "courtship_simon2010_female_trajectory.json"
    scene = ROOT / "config" / "courtship_female_scene.json"
    movie_path = Path("/tmp/courtship-movie.avi")
    visual = ({
        "schema_version": "courtship-visual-assay-1",
        "target_source": "open_loop_movie",
        "camera": dict(CAMERA),
        "movie_path": str(movie_path),
        "movie_sha256": "a" * 64,
        "control_authority": "MaleCNS",
    } if movie else {
        "schema_version": "courtship-visual-assay-1",
        "target_source": "independent_world_trajectory",
        "target_condition": condition,
        "camera": dict(CAMERA),
        "trajectory_path": str(trajectory),
        "trajectory_sha256": "b" * 64,
        "asset_binding": {"object_id": "female_fly_mesh"},
        "control_authority": "MaleCNS",
        "target_labels_enter_control": False,
    })
    return {
        "schema_version": "digital-fly-run-1",
        "seed": 1729,
        "config": str(config),
        "execution_policy": "strict-free-behavior-escape-experiment",
        "run_classification": (
            "courtship_open_loop_visual_probe" if movie
            else "courtship_closed_loop_visual_assay"
        ),
        "neural_intervention": None,
        "autonomy_policy": {
            "strict": True,
            "control_authority": "MaleCNS",
            "counts_as_autonomous_courtship": False,
        },
        "requested_backends": {"neural": "malecns", "vision": "flyvis", "body": "reduced"},
        "requested_renderer_backend": "warp_cuda",
        "effective_backends": {
            "neural": "triton", "vision": "flyvis", "renderer": "warp_cuda",
            "autopilot": "prohibited", "flight_state_drive": "disabled",
            "flight_body_stabilizer": "disabled",
        },
        "effective_execution": {
            "renderer_backend": "warp_cuda", "neural_device": "cuda",
            "neural_backend": "triton", "vision_device": "cuda:0",
            "max_vram_gib": 15.5,
        },
        "requested_execution": {
            "mode": "strict-free-behavior-escape-experiment",
            "strict": True,
            "renderer_backend": "warp_cuda",
            "max_vram_gib": 15.5,
            "neural_device": "cuda",
            "neural_backend": "triton",
            "vision_device": "cuda:0",
            "telemetry_mode": "compact",
            "stimulus_config": str(stimulus),
            "scene_manifest": None if movie else str(scene),
            "courtship_target_trajectory": None if movie else str(trajectory),
            "courtship_target_condition": "moving" if movie else condition,
            "courtship_movie": str(movie_path) if movie else None,
        },
        "telemetry_contract": {"mode": "compact"},
        "courtship_visual": visual,
    }


class CourtshipSuiteTests(unittest.TestCase):
    def test_default_trial_matrix_pairs_each_seed_across_three_conditions(self) -> None:
        specs = suite.make_trial_specs((1729, 1731), suite.TARGET_CONDITIONS)
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            [(item.seed, item.condition) for item in specs],
            [
                (1729, "moving"), (1729, "static"), (1729, "hidden"),
                (1731, "moving"), (1731, "static"), (1731, "hidden"),
            ],
        )
        self.assertEqual(len({item.label for item in specs}), len(specs))

    def test_target_trial_command_requests_all_cuda_backends_and_condition(self) -> None:
        spec = suite.TrialSpec(1731, "static", "seed-1731-static")
        command = suite.build_trial_command(
            spec=spec,
            trial_dir=Path("/tmp/trial"),
            config=suite.DEFAULT_CONFIG,
            stimulus_config=suite.DEFAULT_STIMULUS_CONFIG,
            scene_manifest=suite.DEFAULT_SCENE_MANIFEST,
            trajectory=suite.DEFAULT_TRAJECTORY,
            movie=None,
            duration_s=2.5,
        )
        for expected in (
            "--mode", "free-behavior", "--scenario", "courtship_visual",
            "--neural", "malecns", "--vision", "flyvis", "--body", "reduced",
            "--renderer-backend", "warp_cuda", "--neural-device", "cuda",
            "--neural-backend", "triton", "--vision-device", "cuda:0",
            "--max-vram-gib", "15.5", "--telemetry-mode", "compact",
            "--courtship-target-condition", "static",
        ):
            self.assertIn(expected, command)
        self.assertIn("--scene-manifest", command)
        self.assertIn("--courtship-target-trajectory", command)
        self.assertIn("2.5", command)

    def test_movie_trial_command_omits_scene_and_trajectory(self) -> None:
        spec = suite.TrialSpec(1729, "movie", "seed-1729-movie")
        command = suite.build_trial_command(
            spec=spec,
            trial_dir=Path("/tmp/movie-trial"),
            config=suite.DEFAULT_CONFIG,
            stimulus_config=suite.DEFAULT_STIMULUS_CONFIG,
            scene_manifest=None,
            trajectory=None,
            movie=Path("/data/datasets/raw/malecns-gpu/courtship/simon2010/movS2.avi"),
            duration_s=2.5,
        )
        self.assertIn("--courtship-movie", command)
        self.assertNotIn("--scene-manifest", command)
        self.assertNotIn("--courtship-target-trajectory", command)

    def test_movie_command_refuses_scene_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot include a scene manifest"):
            suite.build_trial_command(
                spec=suite.TrialSpec(1729, "movie", "seed-1729-movie"),
                trial_dir=Path("/tmp/movie-trial"),
                config=suite.DEFAULT_CONFIG,
                stimulus_config=suite.DEFAULT_STIMULUS_CONFIG,
                scene_manifest=suite.DEFAULT_SCENE_MANIFEST,
                trajectory=None,
                movie=Path("/tmp/movie.avi"),
                duration_s=2.5,
            )

    def test_manifest_assertion_accepts_explicit_cuda_target_run(self) -> None:
        trajectory = suite.DEFAULT_TRAJECTORY
        suite._check_manifest(
            _manifest(), spec=suite.TrialSpec(1729, "moving", "seed-1729-moving"),
            config=suite.DEFAULT_CONFIG, stimulus_config=suite.DEFAULT_STIMULUS_CONFIG,
            scene_manifest=suite.DEFAULT_SCENE_MANIFEST, trajectory=trajectory,
            movie=None, trajectory_sha256="b" * 64, movie_sha256=None,
            expected_camera=CAMERA,
        )

    def test_manifest_assertion_rejects_stub_or_cpu_fallback(self) -> None:
        raw = _manifest()
        raw["effective_backends"]["vision"] = "stub"
        with self.assertRaisesRegex(RuntimeError, "CUDA path did not take effect"):
            suite._check_manifest(
                raw, spec=suite.TrialSpec(1729, "moving", "seed-1729-moving"),
                config=suite.DEFAULT_CONFIG, stimulus_config=suite.DEFAULT_STIMULUS_CONFIG,
                scene_manifest=suite.DEFAULT_SCENE_MANIFEST,
                trajectory=suite.DEFAULT_TRAJECTORY, movie=None,
                trajectory_sha256="b" * 64, movie_sha256=None,
                expected_camera=CAMERA,
            )

    def test_manifest_assertion_accepts_open_loop_movie_without_scene(self) -> None:
        raw = _manifest(movie=True)
        movie = Path("/tmp/courtship-movie.avi")
        suite._check_manifest(
            raw, spec=suite.TrialSpec(1729, "movie", "seed-1729-movie"),
            config=suite.DEFAULT_CONFIG, stimulus_config=suite.DEFAULT_STIMULUS_CONFIG,
            scene_manifest=None, trajectory=None, movie=movie,
            trajectory_sha256=None, movie_sha256="a" * 64,
            expected_camera=CAMERA,
        )

    def test_config_stimulus_and_trajectory_inputs_match_locked_assay(self) -> None:
        _, _, camera = suite._validate_config(
            suite.DEFAULT_CONFIG, suite.DEFAULT_STIMULUS_CONFIG,
            suite.DEFAULT_TRAJECTORY, 2.5, suite.DEFAULT_GEOMETRY,
        )
        self.assertEqual(camera["near_m"], 0.0003)
        self.assertEqual(camera["eye_separation_m"], 0.0007)

    def test_config_rejects_default_camera_that_can_clip_near_target(self) -> None:
        raw = json.loads(suite.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        raw["experiment"]["camera"]["near_m"] = 0.01
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "experiment.json"
            config.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "less than 1 mm"):
                suite._validate_config(
                    config, suite.DEFAULT_STIMULUS_CONFIG,
                    suite.DEFAULT_TRAJECTORY, 2.5, suite.DEFAULT_GEOMETRY,
                )

    def test_manifest_camera_must_match_config(self) -> None:
        raw = _manifest()
        raw["courtship_visual"]["camera"]["near_m"] = 0.01
        with self.assertRaisesRegex(RuntimeError, "manifest camera near_m"):
            suite._check_manifest(
                raw, spec=suite.TrialSpec(1729, "moving", "seed-1729-moving"),
                config=suite.DEFAULT_CONFIG, stimulus_config=suite.DEFAULT_STIMULUS_CONFIG,
                scene_manifest=suite.DEFAULT_SCENE_MANIFEST,
                trajectory=suite.DEFAULT_TRAJECTORY, movie=None,
                trajectory_sha256="b" * 64, movie_sha256=None,
                expected_camera=CAMERA,
            )

    def test_scene_manifest_must_bind_the_expected_female_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "female.glb").write_bytes(b"fixture mesh")
            manifest = {
                "schema_version": 1,
                "scene_id": "courtship-test",
                "units": "m",
                "coordinate_system": {
                    "handedness": "right", "up_axis": "+Z", "forward_axis": "+X",
                },
                "materials": {},
                "assets": [{
                    "object_id": "female_fly_mesh",
                    "path": "female.glb",
                    "role": "visual",
                    "transform": {
                        "translation_m": [0, 0, 0],
                        "rotation_xyzw": [0, 0, 0, 1],
                        "scale": [1, 1, 1],
                    },
                    "provenance": {
                        "source": "test fixture", "license": "test", "exporter": "unittest",
                    },
                }],
                "simulation": {"stimulus_object_ids": {"female_fly": "female_fly_mesh"}},
            }
            path = root / "scene.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            parsed, asset = suite._validate_scene_manifest(
                path, expected_object_id="female_fly_mesh",
            )
            self.assertEqual(parsed["scene_id"], "courtship-test")
            self.assertEqual(asset["asset_object_id"], "female_fly_mesh")
            self.assertEqual(len(asset["asset_sha256"]), 64)
            with self.assertRaisesRegex(RuntimeError, "must reference geometry object_id"):
                suite._validate_scene_manifest(path, expected_object_id="different_id")

    def test_telemetry_requires_exact_frame_count_and_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            telemetry = Path(temp) / "telemetry.jsonl"
            rows = [
                {
                    "scenario": "free-behavior",
                    "frame_index": index,
                    "frame_time_s": index * suite.FRAME_DT_S,
                    "dt_s": suite.FRAME_DT_S,
                }
                for index in range(500)
            ]
            telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            count, digest = suite._validate_telemetry(telemetry, label="ok", duration_s=2.5)
            self.assertEqual(count, 500)
            self.assertEqual(len(digest), 64)
            rows[17]["frame_time_s"] += 0.001
            telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "5 ms frame grid"):
                suite._validate_telemetry(telemetry, label="bad", duration_s=2.5)


if __name__ == "__main__":
    unittest.main()
