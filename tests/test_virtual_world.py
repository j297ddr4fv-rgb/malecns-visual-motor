#!/usr/bin/env python3
"""Zero-dependency smoke tests for the deterministic virtual world."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import math
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import virtual_world as _vw  # noqa: E402
from scene_asset_manifest import SceneAssetManifest  # noqa: E402
from virtual_world import CameraConfig, PhysicalStimulus, Pose, VirtualWorld, load_stimuli  # noqa: E402


CONFIG = Path(__file__).resolve().parents[1] / "config" / "scenarios.json"
STIMULUS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "escape_stimuli.json"


def _reference_luminance(world: VirtualWorld, hit: object, direction: tuple[float, float, float]) -> float:
    """Use the pre-optimization scalar shader for a reference pixel."""
    if hit is not None:
        return world._shade(hit)  # type: ignore[arg-type]
    return max(0.0, min(1.0, world.config.light.sky_brightness +
                         0.045 * max(-1.0, min(1.0, direction[2]))))


def _reference_render_eye(world: VirtualWorld, pose: Pose, eye: str, resolution: tuple[int, int],
                          *, timestamp_s: float = 0.0, stimulus: object = None,
                          stimuli: object = None) -> list[list[float]]:
    """Reference raster loop retained to guard prepared-scene equivalence."""
    camera = world._resolve_camera(resolution)
    out: list[list[float]] = []
    for py in range(camera.height):
        row: list[float] = []
        for px in range(camera.width):
            origin, direction = world._ray(pose, eye, px, py, camera)
            hit = world._trace(origin, direction, camera, timestamp_s=timestamp_s,
                               stimulus=stimulus, stimuli=stimuli, reference_pose=pose)
            row.append(_reference_luminance(world, hit, direction))
        out.append(row)
    return out


def _reference_render_directions(world: VirtualWorld, pose: Pose,
                                 directions: list[tuple[float, float, float]], eye: str,
                                 *, timestamp_s: float = 0.0, stimulus: object = None,
                                 stimuli: object = None) -> list[float]:
    """Reference exact-direction loop retained for regression comparisons."""
    camera = world.camera
    forward, right, up = world._camera_basis(pose)
    origin = world.eye_pose(pose, eye)
    out: list[float] = []
    for direction_head in directions:
        head = _vw._unit(direction_head)
        direction = _vw._unit(_vw._add(_vw._scale(forward, head[0]),
                                      _vw._add(_vw._scale(right, -head[1]),
                                               _vw._scale(up, head[2]))))
        hit = world._trace(origin, direction, camera, timestamp_s=timestamp_s,
                           stimulus=stimulus, stimuli=stimuli, reference_pose=pose)
        out.append(_reference_luminance(world, hit, direction))
    return out


def benchmark_render_paths(iterations: int = 3) -> dict[str, float]:
    """Return bounded CPU medians for legacy/reference and prepared renderers.

    The benchmark is intentionally small and dependency-free so it can be run
    from this test module without starting a scenario or allocating artifacts.
    It is also useful to compare a local machine's Python/runtime effects:
    ``PYTHONPATH=tests:scripts python -c 'from test_virtual_world import benchmark_render_paths; print(benchmark_render_paths())'``.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(32, 24), stimuli=STIMULUS_CONFIG)
    pose = Pose((0.0, 0.0, 0.42), yaw_rad=0.2, pitch_rad=0.1)
    directions = [
        (1.0, 0.35 * math.sin(index * 0.017), 0.22 * math.cos(index * 0.013))
        for index in range(721)
    ]
    samples: dict[str, list[float]] = {"reference_stereo_s": [], "prepared_stereo_s": [],
                                       "reference_directions_2eye_s": [], "prepared_directions_2eye_s": []}
    for _ in range(iterations):
        started = time.perf_counter()
        _reference_render_eye(world, pose, "left", (32, 24), timestamp_s=0.73, stimulus="looming")
        _reference_render_eye(world, pose, "right", (32, 24), timestamp_s=0.73, stimulus="looming")
        samples["reference_stereo_s"].append(time.perf_counter() - started)

        started = time.perf_counter()
        world.render_stereo(pose, (32, 24), timestamp_s=0.73, stimulus="looming")
        samples["prepared_stereo_s"].append(time.perf_counter() - started)

        started = time.perf_counter()
        _reference_render_directions(world, pose, directions, "left", timestamp_s=0.73, stimulus="looming")
        _reference_render_directions(world, pose, directions, "right", timestamp_s=0.73, stimulus="looming")
        samples["reference_directions_2eye_s"].append(time.perf_counter() - started)

        started = time.perf_counter()
        world.render_eye_directions(pose, directions, "left", timestamp_s=0.73, stimulus="looming")
        world.render_eye_directions(pose, directions, "right", timestamp_s=0.73, stimulus="looming")
        samples["prepared_directions_2eye_s"].append(time.perf_counter() - started)
    return {name: sorted(values)[len(values) // 2] for name, values in samples.items()}


class VirtualWorldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(12, 8))

    def test_required_scenarios_are_deterministic_and_accepted(self) -> None:
        required = {"hover", "yaw-step", "forward-target", "gust-recovery", "takeoff", "landing"}
        self.assertTrue(required.issubset(self.world.scenarios))
        first = {name: self.world.run_scenario(name) for name in sorted(required)}
        second = {name: self.world.run_scenario(name) for name in sorted(required)}
        for name in required:
            self.assertTrue(first[name].acceptance["passed"], first[name].acceptance)
            self.assertEqual(first[name].metrics["trajectory_sha256"], second[name].metrics["trajectory_sha256"])
            self.assertEqual(first[name].collision_count, 0)

    def test_forward_target_uses_dynamic_waypoints_and_exposes_telemetry(self) -> None:
        result = self.world.run_scenario("forward-target")
        self.assertTrue(result.passed, result.acceptance)
        self.assertEqual(len(result.wind_samples_mps), len(result.times_s))
        self.assertEqual(len(result.trajectory), len(result.times_s))
        self.assertGreater(result.metrics["min_forward_displacement_m"], 0.60)
        self.assertGreaterEqual(result.metrics["target_visible_fraction"], 0.50)
        self.assertAlmostEqual(result.trajectory[-1]["target_distance_m"], result.metrics["final_target_distance_m"])
        self.assertIn("target_position_m", result.trajectory[0])
        self.assertIn("wind_mps", result.trajectory[0])
        # The route passes below the looming block in y, rather than crossing
        # its expanded collision envelope while translating toward the pad.
        obstacle = next(item for item in self.world.obstacles if item.id == "looming_block")
        radius = self.world.config.body_radius_m
        for pose in result.poses:
            outside = (
                pose.position_m[0] <= obstacle.min_m[0] - radius
                or pose.position_m[0] >= obstacle.max_m[0] + radius
                or pose.position_m[1] <= obstacle.min_m[1] - radius
                or pose.position_m[1] >= obstacle.max_m[1] + radius
                or pose.position_m[2] <= obstacle.min_m[2] - radius
                or pose.position_m[2] >= obstacle.max_m[2] + radius
            )
            self.assertTrue(outside)

    def test_integrated_ramps_have_velocity_and_contacts_without_teleporting(self) -> None:
        takeoff = self.world.run_scenario("takeoff")
        landing = self.world.run_scenario("landing")
        self.assertGreater(takeoff.metrics["max_speed_mps"], 0.1)
        self.assertGreater(landing.metrics["max_speed_mps"], 0.1)
        self.assertEqual(takeoff.collision_count, 0)
        self.assertEqual(landing.collision_count, 0)
        self.assertGreater(len(landing.contacts), 0)
        # Consecutive displacement is consistent with a finite integrated
        # velocity, rather than a position assignment at every sample.
        for result in (takeoff, landing):
            for before, after, t_before, t_after in zip(result.poses, result.poses[1:], result.times_s, result.times_s[1:]):
                dt = t_after - t_before
                step = sum((after.position_m[i] - before.position_m[i]) ** 2 for i in range(3)) ** 0.5
                self.assertLess(step, max(before.speed_mps, after.speed_mps, 0.5) * dt + 0.01)

    def test_flight_demo_is_a_deterministic_sustained_scene(self) -> None:
        first = self.world.run_scenario("flight-demo")
        second = self.world.run_scenario("flight-demo")
        self.assertTrue(first.passed, first.acceptance)
        self.assertEqual(first.metrics["trajectory_sha256"], second.metrics["trajectory_sha256"])
        self.assertGreater(first.metrics["max_speed_mps"], 0.15)
        self.assertGreater(first.metrics["max_wind_speed_mps"], 0.20)
        self.assertEqual(first.collision_count, 0)
        payload = first.to_dict()
        self.assertEqual(len(payload["trajectory"]), first.metrics["sample_count"])
        self.assertEqual(payload["trajectory"][0]["time_s"], 0.0)

    def test_full_mission_starts_on_ground_and_reports_contiguous_phases(self) -> None:
        scenario = self.world.scenarios["full-mission"]
        self.assertAlmostEqual(
            scenario.initial_pose.position_m[2],
            self.world.config.ground_z_m + self.world.config.body_radius_m,
            places=9,
        )
        phases = scenario.profile["phases"]
        self.assertEqual(phases[0]["label"], "takeoff")
        self.assertEqual(phases[-1]["label"], "settle")
        for previous, current in zip(phases, phases[1:]):
            self.assertAlmostEqual(previous["end_s"], current["start_s"], places=9)

        result = self.world.run_scenario("full-mission")
        self.assertTrue(result.passed, result.acceptance)
        self.assertEqual(result.phase_labels, tuple(item["label"] for item in phases))
        self.assertTrue(result.trajectory[0]["grounded"])
        self.assertEqual(result.trajectory[0]["phase"], "takeoff")
        self.assertEqual(result.trajectory[-1]["phase"], "settle")
        target_contacts = [row for row in result.trajectory if row["target_contact"]]
        self.assertTrue(target_contacts)
        self.assertTrue(all(row["phase"] in {"landing", "settle"} for row in target_contacts))
        support_z = self.world.config.target.max_m[2] + self.world.config.body_radius_m
        self.assertTrue(all(abs(row["position_m"][2] - support_z) <= 1.0e-9 for row in target_contacts))
        turn_yaws = [pose.yaw_rad for pose, row in zip(result.poses, result.trajectory) if row["phase"] == "turn"]
        self.assertGreater(turn_yaws[-1] - turn_yaws[0], 0.5)
        self.assertLessEqual(max(turn_yaws), 0.95)
        self.assertEqual(set(result.to_dict()["phases"]), {item["label"] for item in phases})
        self.assertEqual(result.collision_count, 0)

    def test_goal_at_exposes_phase_clock_and_is_continuous_at_boundaries(self) -> None:
        scenario = self.world.scenarios["full-mission"]
        phases = scenario.profile["phases"]
        for phase in phases:
            start = float(phase["start_s"])
            goal = self.world.goal_at("full-mission", start)
            self.assertEqual(goal["phase"], phase["label"])
            self.assertEqual(goal["phase_start_s"], start)
            self.assertLessEqual(goal["phase_end_s"], scenario.duration_s)
        for previous, current in zip(phases, phases[1:]):
            boundary = float(current["start_s"])
            before = self.world.goal_at("full-mission", boundary - scenario.dt_s)
            after = self.world.goal_at("full-mission", boundary)
            distance = sum((before["position_m"][i] - after["position_m"][i]) ** 2 for i in range(3)) ** 0.5
            self.assertLess(distance, 0.01)
        self.assertEqual(self.world.phase_at("full-mission", 0.0)["label"], "takeoff")
        self.assertEqual(self.world.phase_at("full-mission", scenario.duration_s)["label"], "settle")

    def test_stereo_render_and_flow_are_independent(self) -> None:
        pose = self.world.run_scenario("hover").poses[0]
        stereo = self.world.render_stereo(pose, (12, 8))
        self.assertEqual(stereo.shape, (8, 12))
        self.assertEqual(stereo.left.shape, (8, 12))
        self.assertNotEqual(stereo.left, stereo.right)
        for row in stereo.left + stereo.right:
            self.assertTrue(all(0.0 <= px <= 1.0 for px in row))
        yaw = self.world.run_scenario("yaw-step").poses
        flow = self.world.optic_flow_ground_truth(yaw[20], yaw[80], dt_s=1.2, resolution=(12, 8))
        self.assertEqual(flow.shape, (8, 12, 2))
        self.assertGreater(flow.max_magnitude_px, 0.0)
        self.assertEqual(len(flow.flow_rad_s), 8)

    def test_direction_renderer_forward_equivalence_and_finite_shape(self) -> None:
        pose = Pose((0.0, 0.0, 0.42), yaw_rad=0.37, pitch_rad=-0.12, roll_rad=0.08)
        # A 1x1 pinhole pixel is the exact camera-forward ray.  The direction
        # path must therefore agree with the existing raster renderer.
        camera = CameraConfig(1, 1, hfov_rad=1.4, vfov_rad=1.1)
        world = VirtualWorld.from_json(CONFIG, camera=camera)
        forward = world.render_eye(pose, "left", (1, 1), timestamp_s=0.27)[0][0]
        sampled = world.render_eye_directions(pose, [(1.0, 0.0, 0.0)], "left", timestamp_s=0.27)
        self.assertEqual(len(sampled), 1)
        self.assertTrue(math.isfinite(sampled[0]))
        self.assertAlmostEqual(sampled[0], forward, places=12)
        for bad in ([], [(1.0, 0.0)], [(1.0, 0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(float("nan"), 0.0, 0.0)]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    world.render_eye_directions(pose, bad)
        with self.assertRaises(ValueError):
            world.render_eye_directions(pose, [(1.0, 0.0, 0.0)], timestamp_s=float("inf"))

    def test_prepared_raster_renderer_is_bit_identical_to_reference(self) -> None:
        pose = Pose((0.10, -0.12, 0.44), yaw_rad=0.27, pitch_rad=-0.10, roll_rad=0.06)
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(18, 12), stimuli=STIMULUS_CONFIG)
        for eye in ("left", "right"):
            with self.subTest(eye=eye):
                expected = _reference_render_eye(world, pose, eye, (18, 12),
                                                 timestamp_s=0.73, stimulus="looming")
                actual = world.render_eye(pose, eye, (18, 12), timestamp_s=0.73, stimulus="looming")
                self.assertEqual(actual, expected)

    def test_prepared_direction_renderer_is_bit_identical_to_reference(self) -> None:
        pose = Pose((0.10, -0.12, 0.44), yaw_rad=0.27, pitch_rad=-0.10, roll_rad=0.06)
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(32, 24), stimuli=STIMULUS_CONFIG)
        directions = [
            (1.0, 0.40 * math.sin(index * 0.031), 0.25 * math.cos(index * 0.019))
            for index in range(721)
        ]
        for eye in ("left", "right"):
            with self.subTest(eye=eye):
                expected = _reference_render_directions(world, pose, directions, eye,
                                                        timestamp_s=0.73, stimulus="looming")
                actual = world.render_eye_directions(pose, directions, eye,
                                                     timestamp_s=0.73, stimulus="looming")
                self.assertEqual(actual, expected)

    def test_equal_distance_trace_keeps_target_ahead_of_stimulus(self) -> None:
        config = replace(self.world.config, obstacles=())
        world = VirtualWorld(config, camera=self.world.camera)
        target = world.config.target
        board = PhysicalStimulus(
            id="target_tie_board", start_position_m=target.position_m,
            velocity_mps=(0.0, 0.0, 0.0), direction_m=(0.0, 0.0, 1.0),
            size_m=target.size_m, mode="static", texture=target.texture,
        )
        origin = (target.min_m[0] - 1.0, target.position_m[1], target.position_m[2])
        direction = (1.0, 0.0, 0.0)
        scalar_hit = world._trace(origin, direction, stimulus=board)
        prepared = world._prepare_render(Pose((0.0, 0.0, 0.42)), "mono", world.camera, 0.5, board)
        prepared_hit = world._trace(origin, direction, _prepared=prepared)

        self.assertIsNotNone(scalar_hit)
        self.assertEqual(scalar_hit.object_id, target.id)
        self.assertEqual(prepared_hit, scalar_hit)

    def test_prepared_renderer_benchmark_smoke(self) -> None:
        timings = benchmark_render_paths(iterations=1)
        self.assertEqual(set(timings), {
            "reference_stereo_s", "prepared_stereo_s",
            "reference_directions_2eye_s", "prepared_directions_2eye_s",
        })
        self.assertTrue(all(math.isfinite(value) and value > 0.0 for value in timings.values()))

    def test_direction_renderer_maps_head_left_to_camera_negative_right(self) -> None:
        # Put two dark, static boards at equal and opposite lateral positions.
        # Head +y means fly-left, so +y must hit the world -camera-right board
        # under the module's explicit camera convention.
        board_left = PhysicalStimulus(
            id="left_board", start_position_m=(0.65, -0.30, 0.42),
            velocity_mps=(0.0, 0.0, 0.0), direction_m=(-1.0, 0.0, 0.0),
            size_m=(0.16, 0.16, 0.02), mode="static", texture="matte", albedo=0.05,
        )
        board_right = PhysicalStimulus(
            id="right_board", start_position_m=(0.65, 0.30, 0.42),
            velocity_mps=(0.0, 0.0, 0.0), direction_m=(-1.0, 0.0, 0.0),
            size_m=(0.16, 0.16, 0.02), mode="static", texture="matte", albedo=0.05,
        )
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(32, 24),
                                       stimuli={"left_board": board_left, "right_board": board_right})
        pose = Pose((0.0, 0.0, 0.42))
        left = world.render_eye_directions(pose, [(1.0, 0.46, 0.0)], "mono",
                                           timestamp_s=0.5, stimulus=board_left)
        right = world.render_eye_directions(pose, [(1.0, -0.46, 0.0)], "mono",
                                            timestamp_s=0.5, stimulus=board_right)
        self.assertLess(left[0], 0.30)
        self.assertLess(right[0], 0.30)
        # A mirrored head direction must not see the board on the opposite side.
        left_mirror = world.render_eye_directions(pose, [(1.0, -0.46, 0.0)], "mono",
                                                  timestamp_s=0.5, stimulus=board_left)
        right_mirror = world.render_eye_directions(pose, [(1.0, 0.46, 0.0)], "mono",
                                                   timestamp_s=0.5, stimulus=board_right)
        self.assertGreater(left_mirror[0], left[0] + 0.10)
        self.assertGreater(right_mirror[0], right[0] + 0.10)

    def test_geometry_collision_target_and_wind(self) -> None:
        obstacle_pose = Pose((0.48, 0.0, 0.20))
        checked = self.world.check_collisions(obstacle_pose)
        self.assertTrue(checked["colliding"])
        obs = self.world.target_observation(Pose((0.0, 0.0, 0.35), yaw_rad=0.0))
        self.assertGreater(obs.distance_m, 0.0)
        self.assertAlmostEqual(obs.bearing_rad, 0.0, places=6)
        gust = {"base_mps": [0.0, 0.0, 0.0], "gusts": [{"start_s": 0.7, "duration_s": 0.28, "vector_mps": [0.0, 0.55, 0.0]}]}
        self.assertEqual(self.world.wind_at(0.70, (0.0, 0.0, 0.35)), self.world.wind_at((0.0, 0.0, 0.35), 0.70))
        override = self.world.config.wind.with_override(gust)
        self.assertGreater(self.world.wind_at((0.0, 0.0, 0.35), 0.84, override)[1], 0.4)

    def test_shared_twin_contract_adapters(self) -> None:
        result = self.world.run_scenario("hover", render=True, resolution=(12, 8))
        body = self.world.body_state(result.poses[0], 0.0, dt_s=0.02)
        world_state = self.world.world_state(result.poses[0], 0.0, dt_s=0.02)
        visual = self.world.visual_observation(result.frames[0], 0.0, dt_s=0.02)
        self.assertEqual(body.position_m, result.poses[0].position_m)
        self.assertEqual(world_state.body.timestamp, body.timestamp)
        self.assertEqual(visual.pixel_shape, (8, 12))

    def test_manifest_records_scene_contract(self) -> None:
        manifest = self.world.manifest()
        self.assertTrue(manifest["deterministic"])
        self.assertEqual(manifest["units"], "SI")
        self.assertIn("ground", manifest["geometry"])
        self.assertIn("walls", manifest["geometry"])
        self.assertIn("landing_pad", manifest["geometry"])
        self.assertTrue(manifest["textures"]["band_limited"])
        self.assertEqual(len(manifest["config_sha256"]), 64)

    def test_world_backend_is_explicit_analytic_reference(self) -> None:
        self.assertEqual(self.world.backend_role, "analytic_reference")
        self.assertEqual(self.world.backend_contract, "world-renderer-v1")

    @staticmethod
    def _fake_scene_mesh(object_id: str, *, role: str = "visual",
                         vertices: tuple[tuple[float, float, float], ...] | None = None,
                         triangles: tuple[tuple[int, int, int], ...] | None = None,
                         triangle_albedos: tuple[tuple[float, float, float], ...] | None = None) -> SimpleNamespace:
        if vertices is None:
            vertices = ((0.0, -0.05, -0.05), (0.0, 0.05, -0.05),
                        (0.0, 0.05, 0.05), (0.0, -0.05, 0.05))
        if triangles is None:
            triangles = ((0, 1, 2), (0, 2, 3))
        return SimpleNamespace(
            object_id=object_id,
            role=role,
            semantic_role="stimulus",
            vertices_m=vertices,
            triangles=triangles,
            triangle_material_ids=("skin",) * len(triangles),
            triangle_albedos=triangle_albedos,
            materials={"skin": {"base_color_rgba": (0.2, 0.4, 0.6, 1.0)}},
            source_path=Path("/assets/test-hand.glb"),
            source_sha256="a" * 64,
            source_files_sha256={"textures/hand.png": "b" * 64},
            provenance={"source": "unit-test", "license": "CC0", "exporter": "unit-test"},
        )

    def _world_with_fake_scene(self, objects: tuple[SimpleNamespace, ...], bindings: dict[str, str]) -> VirtualWorld:
        loaded = SimpleNamespace(
            objects=objects,
            manifest={"scene_id": "unit-test-scene", "simulation": {"stimulus_object_ids": bindings}},
        )
        with patch("gltf_mesh_loader.load_scene_assets", return_value=loaded):
            return VirtualWorld.from_json(CONFIG, scene_path="/tmp/unit-test-scene.json")

    def test_bound_stimulus_mesh_is_local_dynamic_visual_only_and_manifested(self) -> None:
        hand = self._fake_scene_mesh(
            "moving_hand", triangle_albedos=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        background = self._fake_scene_mesh(
            "background", vertices=((5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (5.0, 0.0, 1.0)),
            triangles=((0, 1, 2),), triangle_albedos=((0.4, 0.4, 0.4),),
        )
        world = self._world_with_fake_scene(
            (hand, background), {"palm_looming_grounded": "moving_hand"}
        )
        self.assertIs(world.stimulus_visual_meshes["palm_looming_grounded"], hand)
        self.assertEqual([triangle.object_id for triangle in world.scene_triangles], ["background"])
        with self.assertRaises(TypeError):
            world.stimulus_visual_meshes["other"] = hand  # type: ignore[index]

        stimulus = PhysicalStimulus(
            id="palm_looming_grounded", start_position_m=(1.0, 1.2, 0.42),
            velocity_mps=(-0.25, 0.0, 0.0), direction_m=(-1.0, 0.0, 0.0),
            size_m=(0.09, 0.09, 0.02), mode="looming", visual_only=True,
            collidable=True, duration_s=2.0,
        )
        origin = (-1.0, 1.2, 0.42)
        at_start = world.trace_ray(origin, (1.0, 0.0, 0.0), timestamp_s=0.0, stimulus=stimulus)
        later = world.trace_ray(origin, (1.0, 0.0, 0.0), timestamp_s=1.0, stimulus=stimulus)
        self.assertIsNotNone(at_start)
        self.assertIsNotNone(later)
        self.assertEqual(at_start.kind, "stimulus_mesh")
        self.assertEqual(at_start.object_id, stimulus.id)
        self.assertAlmostEqual(at_start.distance_m, 2.0, places=9)
        self.assertAlmostEqual(later.distance_m, 1.75, places=9)
        scalar_hit = world._trace(origin, (1.0, 0.0, 0.0), timestamp_s=0.0, stimulus=stimulus)
        self.assertAlmostEqual(scalar_hit.albedo, 0.2126, places=12)
        self.assertFalse(stimulus.state_at(0.0).collidable)
        self.assertEqual(at_start.texture, "stimulus_mesh")

        prepared = world._prepare_render(Pose(origin), "mono", world.camera, 0.0, stimulus)
        prepared_hit = world._trace(origin, (1.0, 0.0, 0.0), _prepared=prepared)
        self.assertEqual(prepared_hit, world._trace(origin, (1.0, 0.0, 0.0), stimulus=stimulus))
        binding = world.external_scene_manifest["stimulus_visual_bindings"][stimulus.id]
        self.assertEqual(binding["object_id"], "moving_hand")
        self.assertEqual(binding["source_sha256"], "a" * 64)
        self.assertEqual(binding["triangle_count"], 2)

    def test_bound_stimulus_mesh_falls_back_to_material_albedo_when_unbaked(self) -> None:
        hand = self._fake_scene_mesh("moving_hand", triangle_albedos=None)
        world = self._world_with_fake_scene((hand,), {"palm": "moving_hand"})
        stimulus = PhysicalStimulus(
            id="palm", start_position_m=(1.0, 1.2, 0.42),
            direction_m=(-1.0, 0.0, 0.0), size_m=(0.09, 0.09, 0.02),
            mode="static", renderable=True, collidable=False,
        )
        hit = world._trace((-1.0, 1.2, 0.42), (1.0, 0.0, 0.0), stimulus=stimulus)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.albedo, 0.37192, places=12)

    def test_unbound_stimulus_keeps_the_original_board_hit(self) -> None:
        stimulus = PhysicalStimulus(
            id="looming", start_position_m=(1.0, 1.2, 0.42),
            velocity_mps=(-0.2, 0.0, 0.0), direction_m=(-1.0, 0.0, 0.0),
            size_m=(0.16, 0.16, 0.02), mode="looming",
        )
        no_scene = VirtualWorld.from_json(CONFIG, stimuli=stimulus)
        attached_scene = self._world_with_fake_scene((), {})
        ray = ((-1.0, 1.2, 0.42), (1.0, 0.0, 0.0))
        expected = no_scene._trace(*ray, timestamp_s=0.4, stimulus=stimulus)
        actual = attached_scene._trace(*ray, timestamp_s=0.4, stimulus=stimulus)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.kind, "stimulus")

    def test_stimulus_binding_fails_closed_for_missing_nonvisual_or_malformed_assets(self) -> None:
        valid = self._fake_scene_mesh("moving_hand")
        with self.subTest(reason="missing asset ID"):
            with self.assertRaisesRegex(ValueError, "missing scene object_id"):
                self._world_with_fake_scene((valid,), {"palm": "absent"})
        with self.subTest(reason="nonvisual asset"):
            collision = self._fake_scene_mesh("collision_only", role="collision")
            with self.assertRaisesRegex(ValueError, "must bind to a visual"):
                self._world_with_fake_scene((collision,), {"palm": "collision_only"})
        with self.subTest(reason="malformed mesh"):
            malformed = self._fake_scene_mesh("broken", triangles=((0, 1, 99),))
            with self.assertRaisesRegex(ValueError, "invalid vertex index"):
                self._world_with_fake_scene((malformed,), {"palm": "broken"})

    def test_scene_manifest_parses_and_roundtrips_simulation_bindings(self) -> None:
        payload = {
            "schema_version": 1,
            "scene_id": "binding-test",
            "units": "m",
            "coordinate_system": {"handedness": "right", "up_axis": "+Z", "forward_axis": "+X"},
            "materials": {},
            "assets": [],
            "simulation": {"stimulus_object_ids": {"palm_looming_grounded": "moving_hand"}},
            "replay": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "scene.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            parsed = SceneAssetManifest.from_json(manifest_path)
            self.assertEqual(
                parsed.simulation["stimulus_object_ids"],
                {"palm_looming_grounded": "moving_hand"},
            )
            self.assertEqual(
                parsed.to_dict()["simulation"],
                {"stimulus_object_ids": {"palm_looming_grounded": "moving_hand"}},
            )
            payload["simulation"] = {"stimulus_object_ids": {"palm": "  "}}
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty object ID"):
                SceneAssetManifest.from_json(manifest_path)

    def test_moving_board_is_binocular_geometry_with_replayable_state(self) -> None:
        stimuli = load_stimuli(STIMULUS_CONFIG)
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(24, 16), stimuli=stimuli)
        board = world.stimulus("looming")
        pose = Pose((0.0, 0.0, 0.42))
        state_a = board.stimulus_state_at(0.75)
        state_b = board.stimulus_state_at(1.25)
        self.assertNotEqual(state_a["position_m"], state_b["position_m"])
        self.assertTrue(state_a["forbidden_for_control"])
        self.assertTrue(state_b["forbidden_for_control"])
        left_a = world.render_eye(pose, "left", (24, 16), timestamp_s=0.75, stimulus=board)
        left_b = world.render_eye(pose, "left", (24, 16), timestamp_s=1.25, stimulus=board)
        self.assertNotEqual(left_a, left_b)
        center_hit = world.trace_ray(pose.position_m, (1.0, 0.0, 0.0), timestamp_s=0.75, stimulus=board)
        self.assertIsNotNone(center_hit)
        self.assertEqual(center_hit.object_id, board.id)
        # The moving board is a true collision object, not a visual overlay.
        checked = world.check_collisions(Pose((0.50, 0.0, 0.42)), 1.0, stimulus=board)
        self.assertTrue(checked["colliding"])

    def test_grounded_core_stimulus_is_world_anchored_not_body_following(self) -> None:
        stimuli = load_stimuli(STIMULUS_CONFIG)
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(16, 12), stimuli=stimuli)
        grounded = world.stimulus("looming_grounded")
        grounded_from_origin = grounded.state_at(0.8, Pose((0.0, 0.0, 0.0012)))
        grounded_after_body_motion = grounded.state_at(0.8, Pose((0.35, 0.0, 0.20)))
        self.assertEqual(grounded_from_origin.position_m, grounded_after_body_motion.position_m)
        self.assertAlmostEqual(grounded_from_origin.position_m[2], 0.0012, places=9)
        live = world.stimulus("looming_body_relative")
        live_from_origin = live.state_at(0.8, Pose((0.0, 0.0, 0.0012)))
        live_after_body_motion = live.state_at(0.8, Pose((0.35, 0.0, 0.20)))
        self.assertNotEqual(live_from_origin.position_m, live_after_body_motion.position_m)

    def test_looming_and_matched_controls_have_distinct_angular_cues(self) -> None:
        stimuli = load_stimuli(STIMULUS_CONFIG)
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(16, 12), stimuli=stimuli)
        pose = Pose((0.0, 0.0, 0.42))
        looming = world.stimulus_observation(pose, "looming", 1.0, previous_t_s=0.9, dt_s=0.1)
        contracting = world.stimulus_observation(pose, "matched_contracting", 1.0, previous_t_s=0.9, dt_s=0.1)
        static = world.stimulus_observation(pose, "matched_static", 1.0, previous_t_s=0.9, dt_s=0.1)
        self.assertGreater(looming.angular_rate_rad_s, contracting.angular_rate_rad_s)
        self.assertGreater(contracting.angular_rate_rad_s, static.angular_rate_rad_s)
        initial_looming = world.stimulus_observation(pose, "looming", 0.0)
        self.assertAlmostEqual(initial_looming.angular_radius_rad, math.atan(0.045), places=9)
        self.assertTrue(looming.looming)
        self.assertFalse(static.looming)
        self.assertFalse(world.stimulus_observation(pose, "sham", 1.0).visible)
        translation = stimuli["matched_translation"]
        self.assertAlmostEqual(translation.speed_mps, math.sqrt(sum(value * value for value in translation.velocity_mps)), places=12)
        self.assertAlmostEqual(stimuli["looming_10mps_pressure"].duration_s or 0.0, 0.13, places=12)

    def test_grounded_contact_time_and_face_radius_are_explicit(self) -> None:
        stimuli = load_stimuli(STIMULUS_CONFIG)
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(16, 12), stimuli=stimuli)
        pose = Pose((0.0, 0.0, 0.0012))
        observation = world.stimulus_observation(pose, "looming_grounded", 0.0, dt_s=0.01)
        self.assertAlmostEqual(observation.angular_radius_rad, math.atan(0.045), places=9)
        self.assertAlmostEqual(observation.contact_time_s, 1.9776, places=6)
        self.assertAlmostEqual(observation.counterfactual_contact_time_s or 0.0, 1.9776, places=6)
        self.assertTrue(observation.visual_only)
        self.assertTrue(observation.non_interacting)
        self.assertFalse(world.stimulus("looming_grounded").collidable)
        self.assertTrue(world.stimulus("looming_grounded").renderable)
        self.assertTrue(world.stimulus("static_grounded").visual_only)
        self.assertTrue(world.stimulus("static_grounded").renderable)
        self.assertFalse(world.stimulus("static_grounded").collidable)
        self.assertFalse(world.stimulus("sham_grounded").collidable)
        for name in ("looming_grounded", "static_grounded", "sham_grounded"):
            self.assertAlmostEqual(world.stimulus(name).duration_s or 0.0, 2.5, places=9)
            self.assertTrue(world.stimulus(name).state_at(2.4).active)
        at_contact = world.stimulus_observation(pose, "looming_grounded", 1.9776, dt_s=0.01)
        self.assertTrue(at_contact.surface_contact)
        self.assertFalse(at_contact.colliding)
        self.assertAlmostEqual(at_contact.counterfactual_contact_time_s or 0.0, 1.9776, places=6)
        self.assertEqual(world.check_collisions(pose, 1.9776, "looming_grounded")["collision_count"], 0)

    def test_grounded_loom_uses_explicit_dark_palm_appearance(self) -> None:
        stimuli = load_stimuli(STIMULUS_CONFIG)
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(32, 24), stimuli=stimuli)
        pose = Pose((0.0, 0.0, 0.0012))
        looming = world.render_eye(pose, "left", (32, 24), timestamp_s=1.97, stimulus="looming_grounded")
        matched = world.render_eye(pose, "left", (32, 24), timestamp_s=1.97, stimulus="static_grounded")
        self.assertLess(looming[12][16], matched[12][16])
        looming_mean = sum(value for row in looming for value in row) / (32 * 24)
        matched_mean = sum(value for row in matched for value in row) / (32 * 24)
        self.assertLess(looming_mean, matched_mean)
        self.assertAlmostEqual(stimuli["looming_grounded"].albedo or 0.0, 0.12, places=9)

    def test_stimulus_randomization_is_seeded_and_does_not_emit_behaviour_tags(self) -> None:
        first = PhysicalStimulus.randomized(41, speed_range_mps=(0.5, 0.5))
        second = PhysicalStimulus.randomized(41, speed_range_mps=(0.5, 0.5))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertAlmostEqual(first.speed_mps, 0.5)
        forbidden = {"threat", "escape", "takeoff", "desired_heading"}
        self.assertTrue(forbidden.isdisjoint(first.to_dict()))
        self.assertTrue(forbidden.isdisjoint(first.stimulus_state_at(0.5)))


if __name__ == "__main__":
    unittest.main()
