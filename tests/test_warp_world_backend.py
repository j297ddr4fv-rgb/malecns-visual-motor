#!/usr/bin/env python3
"""Focused optional NVIDIA Warp renderer tests.

Run with the registered Warp environment for CUDA coverage::

    /data/environments/projects/flyvis-opticlobe/bin/python -m unittest \
        tests.test_warp_world_backend

The module remains import-safe under the system Python; CUDA cases skip when
Warp or a CUDA device is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from virtual_world import CameraConfig, Pose, VirtualWorld  # noqa: E402
from warp_world_backend import (  # noqa: E402
    WarpBackendUnavailableError,
    WarpRenderTimings,
    WarpWorldRenderer,
    _stimulus_local_axes,
    _stimulus_mesh_payload,
    warp_cuda_available,
)
import world_renderer_opt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "scenarios.json"
STIMULI = ROOT / "config" / "escape_stimuli.json"


class StimulusVisualMeshPayloadTest(unittest.TestCase):
    def test_rgb_face_albedo_is_reduced_with_rec709_and_kept_per_triangle(self) -> None:
        mesh = SimpleNamespace(
            vertices_m=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                   (0.0, 0.0, 1.0)), dtype=np.float64),
            triangles=np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int64),
            triangle_albedos=np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64),
        )
        payload = _stimulus_mesh_payload(mesh, "hand", seed=17)
        self.assertTrue(np.array_equal(payload.indices, ((0, 1, 2), (0, 3, 1))))
        self.assertEqual([material.object_id for material in payload.materials], ["hand", "hand"])
        self.assertEqual([material.kind for material in payload.materials], [5, 5])
        self.assertEqual([material.texture_name for material in payload.materials],
                         ["stimulus_mesh", "stimulus_mesh"])
        self.assertAlmostEqual(payload.materials[0].albedo, 0.2126, places=7)
        self.assertAlmostEqual(payload.materials[1].albedo, 0.7152, places=7)

    def test_legacy_material_factor_fallback_matches_scene_mesh_luminance(self) -> None:
        mesh = SimpleNamespace(
            vertices_m=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32),
            triangles=np.asarray(((0, 1, 2),), dtype=np.int32),
            triangle_material_ids=("mat",),
            materials={"mat": {"base_color_rgba": (0.2, 0.4, 0.6, 1.0)}},
        )
        payload = _stimulus_mesh_payload(mesh, "hand", seed=0)
        self.assertAlmostEqual(payload.materials[0].albedo,
                               0.2126 * 0.2 + 0.7152 * 0.4 + 0.0722 * 0.6,
                               places=7)

    def test_local_axes_match_hand_asset_forward_width_and_up(self) -> None:
        self.assertEqual(_stimulus_local_axes((1.0, 0.0, 0.0)),
                         ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        axes = _stimulus_local_axes((0.0, 1.0, 0.0))
        self.assertEqual(axes[0], (0.0, 1.0, 0.0))
        self.assertEqual(axes[1], (-1.0, 0.0, 0.0))
        self.assertEqual(axes[2], (0.0, 0.0, 1.0))


class WarpWorldBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not warp_cuda_available():
            raise unittest.SkipTest("Warp CUDA device is unavailable")
        cls.world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(32, 24), stimuli=STIMULI)
        cls.renderer = WarpWorldRenderer(cls.world, device="cuda:0")

    def test_manifest_records_cuda_cache_stream_and_transfer_contract(self) -> None:
        manifest = self.renderer.manifest()
        self.assertEqual(manifest["backend"], "warp_cuda")
        self.assertEqual(manifest["effective_backend"], "warp_cuda")
        self.assertTrue(manifest["zero_copy"]["tensor_output"])
        self.assertFalse(manifest["zero_copy"]["list_output"])
        self.assertFalse(manifest["stream"]["device_wide_sync_in_production"])
        self.assertIn("kernel_cache", manifest)
        self.assertIn("architecture", manifest)

    def test_static_raster_and_exact_directions_match_analytic(self) -> None:
        pose = Pose((0.11, -0.13, 0.44), yaw_rad=0.27, pitch_rad=-0.1, roll_rad=0.06)
        directions = [
            (1.0, 0.4 * math.sin(index * 0.031), 0.25 * math.cos(index * 0.019))
            for index in range(721)
        ]
        static_world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(32, 24))
        static_renderer = WarpWorldRenderer(static_world, device="cuda:0")
        for eye in ("left", "right"):
            expected = np.asarray(static_world.render_eye_directions(pose, directions, eye), dtype=np.float64)
            actual = np.asarray(static_renderer.render_eye_directions(pose, directions, eye), dtype=np.float64)
            self.assertLess(float(np.max(np.abs(expected - actual))), 5.0e-6)
        expected_frame = np.asarray(static_world.render_eye(pose, "left", (32, 24)), dtype=np.float64)
        actual_frame = np.asarray(static_renderer.render_eye(pose, "left", (32, 24)), dtype=np.float64)
        self.assertLess(float(np.max(np.abs(expected_frame - actual_frame))), 5.0e-6)

        # Exercise the scalar-eye-origin path used by rectangular renders with
        # an active moving board as well as the exact-column path above.
        expected_dynamic_frame = np.asarray(
            self.world.render_eye(pose, "left", (32, 24), timestamp_s=0.73,
                                  stimulus="looming"),
            dtype=np.float64,
        )
        actual_dynamic_frame = np.asarray(
            self.renderer.render_eye(pose, "left", (32, 24), timestamp_s=0.73,
                                     stimulus="looming"),
            dtype=np.float64,
        )
        self.assertLess(float(np.max(np.abs(expected_dynamic_frame - actual_dynamic_frame))), 5.0e-6)

    def test_dynamic_palm_phase_onset_refit_and_inactive_toggle(self) -> None:
        pose = Pose((0.0, 0.0, 0.42), yaw_rad=0.1)
        forward = [(1.0, 0.0, 0.0)]
        static_mesh_id = self.renderer._scene_mesh.id
        for timestamp_s in (0.0, 0.5, 2.0):
            expected = self.world.render_eye_directions(pose, forward, "left", timestamp_s=timestamp_s,
                                                        stimulus="looming")
            actual = self.renderer.render_eye_directions(pose, forward, "left", timestamp_s=timestamp_s,
                                                         stimulus="looming")
            self.assertLess(abs(expected[0] - actual[0]), 5.0e-6)
            self.assertTrue(self.renderer.mesh_stats["dynamic_mesh_active"])
        self.assertEqual(self.renderer._scene_mesh.id, static_mesh_id)
        expected = self.world.render_eye_directions(pose, forward, "left", timestamp_s=2.1, stimulus="looming")
        actual = self.renderer.render_eye_directions(pose, forward, "left", timestamp_s=2.1, stimulus="looming")
        self.assertLess(abs(expected[0] - actual[0]), 5.0e-6)
        self.assertFalse(self.renderer.mesh_stats["dynamic_mesh_active"])
        self.assertGreaterEqual(self.renderer.mesh_stats["dynamic_refit_count"], 1)

    def test_bound_local_hand_reuses_bvh_and_matches_cpu_hit_and_shading(self) -> None:
        world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(32, 24), stimuli=STIMULI)
        hand = SimpleNamespace(
            object_id="synthetic_hand",
            vertices_m=np.asarray(((0.0, -0.05, -0.05), (0.0, 0.05, -0.05),
                                   (0.0, 0.05, 0.05), (0.0, -0.05, 0.05)), dtype=np.float64),
            triangles=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
            triangle_albedos=np.asarray(((0.25, 0.4, 0.55), (0.25, 0.4, 0.55)), dtype=np.float64),
            source_sha256="c" * 64,
        )
        world._stimulus_visual_meshes["palm_looming"] = hand
        world._stimulus_mesh_accels["palm_looming"] = world._prepare_stimulus_mesh_accel(
            hand, "palm_looming"
        )
        renderer = WarpWorldRenderer(world, device="cuda:0")
        pose = Pose((-1.0, 0.0, 0.42))
        unbound_world = VirtualWorld.from_json(CONFIG, camera=CameraConfig(32, 24), stimuli=STIMULI)
        self.assertEqual(world.collision_events(pose, 0.25, stimulus="looming"),
                         unbound_world.collision_events(pose, 0.25, stimulus="looming"))

        initial_stats = renderer.mesh_stats
        self.assertEqual(initial_stats["stimulus_visual_mesh_count"], 1)
        self.assertEqual(initial_stats["stimulus_visual_bvh_build_count"], 1)
        self.assertEqual(initial_stats["stimulus_visual_vertex_upload_count"], 1)
        self.assertEqual(initial_stats["stimulus_visual_bvh_refit_count"], 0)

        for timestamp_s in (0.0, 0.4, 1.0):
            expected_hit = world.trace_ray(pose.position_m, (1.0, 0.0, 0.0),
                                           timestamp_s=timestamp_s, stimulus="looming")
            actual_hit = renderer.trace_ray(pose.position_m, (1.0, 0.0, 0.0),
                                            timestamp_s=timestamp_s, stimulus="looming")
            self.assertIsNotNone(expected_hit)
            self.assertIsNotNone(actual_hit)
            self.assertEqual(actual_hit.kind, "stimulus_mesh")
            self.assertEqual(actual_hit.object_id, expected_hit.object_id)
            self.assertAlmostEqual(actual_hit.distance_m, expected_hit.distance_m, places=5)
            self.assertLess(float(np.max(np.abs(np.asarray(actual_hit.normal_m) -
                                                np.asarray(expected_hit.normal_m)))), 5.0e-5)

            expected_luminance = world.render_eye_directions(
                pose, [(1.0, 0.0, 0.0)], "mono", timestamp_s=timestamp_s, stimulus="looming"
            )
            actual_luminance = renderer.render_eye_directions(
                pose, [(1.0, 0.0, 0.0)], "mono", timestamp_s=timestamp_s, stimulus="looming"
            )
            self.assertLess(abs(actual_luminance[0] - expected_luminance[0]), 5.0e-6)

        final_stats = renderer.mesh_stats
        self.assertEqual(final_stats["stimulus_visual_bvh_build_count"], 1)
        self.assertEqual(final_stats["stimulus_visual_bvh_refit_count"], 0)
        self.assertEqual(final_stats["stimulus_visual_vertex_upload_count"], 1)
        self.assertGreaterEqual(final_stats["stimulus_visual_pose_upload_count"], 3)
        self.assertFalse(final_stats["dynamic_mesh_active"])

    def test_geometry_near_far_inside_exit_and_outward_mesh_normals(self) -> None:
        origin = (0.0, 0.0, 0.42)
        expected = self.world.trace_ray(origin, (1.0, 0.0, 0.0), stimulus="looming")
        actual = self.renderer.trace_ray(origin, (1.0, 0.0, 0.0), stimulus="looming")
        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual)
        self.assertEqual(actual.object_id, expected.object_id)
        self.assertAlmostEqual(actual.distance_m, expected.distance_m, places=5)
        self.assertLessEqual(abs(actual.normal_m[0]), 1.0)
        self.assertAlmostEqual(actual.normal_m[0], -1.0, places=5)

        bounded = self.renderer.trace_ray(origin, (1.0, 0.0, 0.0), far_m=0.5, stimulus="looming")
        self.assertIsNone(bounded)
        near_hit = self.renderer.trace_ray(origin, (1.0, 0.0, 0.0), near_m=0.95, far_m=1.1, stimulus="looming")
        self.assertIsNotNone(near_hit)
        self.assertGreaterEqual(near_hit.distance_m, 0.95)

        self.renderer.trace_ray((0.0, 0.0, 0.42), (1.0, 0.0, 0.0), timestamp_s=2.0, stimulus="looming")
        inside = self.renderer.trace_ray((0.0, 0.0, 0.42), (1.0, 0.0, 0.0),
                                         timestamp_s=2.0, stimulus="looming")
        self.assertIsNotNone(inside)
        self.assertEqual(inside.object_id, "palm_looming")

    def test_ground_wins_equal_distance_tie_before_static_scene_mesh(self) -> None:
        # The ray reaches (wall_xmax, ground) at one exact distance.  The
        # analytic contract inserts ground before walls; the Warp kernel
        # queries the dedicated ground BVH first and keeps strict '<' ties.
        origin = (0.0, 0.0, 0.42)
        direction = (2.5, -2.0, -0.42)
        expected = self.world.trace_ray(origin, direction)
        actual = self.renderer.trace_ray(origin, direction)
        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual)
        self.assertEqual(expected.object_id, "ground")
        self.assertEqual(actual.object_id, "ground")

    def test_tensor_output_is_cuda_resident_and_benchmark_separates_cold_steady(self) -> None:
        pose = Pose((0.0, 0.0, 0.42))
        directions = [(1.0, 0.0, 0.0)] * 721
        tensor = self.renderer.render_eye_directions_tensor(pose, directions, "left")
        self.assertTrue(bool(tensor.is_cuda))
        self.assertEqual(tuple(tensor.shape), (721,))
        self.assertTrue(bool(np.isfinite(tensor.detach().cpu().numpy()).all()))
        report = self.renderer.benchmark(iterations=2, warmup=1, directions=directions)
        self.assertGreater(report["cold_start_s"], 0.0)
        self.assertGreater(report["steady_state_median_s"], 0.0)
        self.assertEqual(report["mesh_stats"]["static_mesh_resident"], True)


class WarpUnavailableContractTest(unittest.TestCase):
    def test_timing_dict_keeps_component_fields_and_total(self) -> None:
        timings = WarpRenderTimings(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        self.assertEqual(timings.to_dict(), {
            "mesh_prepare_s": 1.0, "dynamic_rebuild_s": 2.0,
            "dynamic_refit_s": 3.0, "ray_kernel_s": 4.0,
            "shade_kernel_s": 5.0, "host_copy_s": 6.0, "total_s": 21.0,
        })

    def test_factory_keeps_analytic_default_explicit(self) -> None:
        world = VirtualWorld.from_json(CONFIG)
        self.assertIs(world_renderer_opt.create_renderer(world, backend="analytic_reference"), world)

    def test_import_safe_factory_fails_closed_without_warp(self) -> None:
        if warp_cuda_available():
            self.skipTest("CUDA environment available; unavailable path is system-Python-only")
        world = VirtualWorld.from_json(CONFIG)
        with self.assertRaises(WarpBackendUnavailableError):
            WarpWorldRenderer(world)


if __name__ == "__main__":
    unittest.main()
