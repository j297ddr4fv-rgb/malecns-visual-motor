from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from courtship_target import (  # noqa: E402
    FemaleTargetGeometry,
    FemaleTargetTrajectory,
    FemaleTrajectoryStimulus,
)
from virtual_world import Pose, VirtualWorld  # noqa: E402


class FemaleTargetTrajectoryTests(unittest.TestCase):
    def test_linear_position_and_shortest_angle_interpolation(self) -> None:
        trajectory = FemaleTargetTrajectory.from_mapping({
            "trajectory_id": "wrap-test",
            "keyframes": [
                {"t_s": 0.0, "position_m": [1.0, 0.0, 0.0], "yaw_rad": math.radians(179.0)},
                {"t_s": 2.0, "position_m": [3.0, 2.0, 0.0], "yaw_rad": math.radians(-179.0)},
            ],
        })

        pose = trajectory.sample(1.0)

        self.assertEqual(pose.position_m, (2.0, 1.0, 0.0))
        self.assertAlmostEqual(abs(pose.yaw_rad), math.pi)
        self.assertAlmostEqual(pose.velocity_mps[0], 1.0)
        self.assertAlmostEqual(pose.velocity_mps[1], 1.0)
        self.assertAlmostEqual(pose.angular_velocity_radps[2], math.radians(1.0))

    def test_endpoint_clamping_uses_zero_velocity(self) -> None:
        trajectory = FemaleTargetTrajectory.from_mapping({
            "keyframes": [
                {"t_s": 1.0, "position_m": [1.0, 2.0, 3.0]},
                {"t_s": 2.0, "position_m": [2.0, 2.0, 3.0]},
            ],
        })

        before = trajectory.sample(0.0)
        after = trajectory.sample(3.0)

        self.assertEqual(before.position_m, (1.0, 2.0, 3.0))
        self.assertEqual(before.velocity_mps, (0.0, 0.0, 0.0))
        self.assertEqual(after.position_m, (2.0, 2.0, 3.0))
        self.assertEqual(after.velocity_mps, (0.0, 0.0, 0.0))

    def test_rejects_unsorted_keyframes_and_unsupported_interpolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            FemaleTargetTrajectory.from_mapping({
                "keyframes": [
                    {"t_s": 1.0, "position_m": [0.0, 0.0, 0.0]},
                    {"t_s": 1.0, "position_m": [1.0, 0.0, 0.0]},
                ],
            })
        with self.assertRaisesRegex(ValueError, "interpolation"):
            FemaleTargetTrajectory.from_mapping({"interpolation": "spline", "keyframes": [
                {"t_s": 0.0, "position_m": [0.0, 0.0, 0.0]},
            ]})


class FemaleTrajectoryStimulusTests(unittest.TestCase):
    def test_default_configs_describe_mm_scale_female_and_world_path(self) -> None:
        geometry = FemaleTargetGeometry.from_json()
        trajectory = FemaleTargetTrajectory.from_json()

        self.assertEqual(geometry.semantic_role, "adult_female_drosophila_visual_target")
        self.assertAlmostEqual(geometry.target_extent_m[0], 0.0028)
        self.assertLess(geometry.target_extent_m[0], 0.005)
        self.assertEqual(geometry.asset_object_id, "female_fly_model")
        self.assertIn("abdomen", {part["name"] for part in geometry.body_parts})
        self.assertIn("left_compound_eye", {part["name"] for part in geometry.body_parts})
        self.assertEqual(trajectory.start_time_s, 0.0)
        self.assertGreater(trajectory.end_time_s, trajectory.start_time_s)

    def test_target_pose_is_independent_of_reference_pose(self) -> None:
        target = FemaleTrajectoryStimulus()
        reference_a = Pose((0.0, 0.0, 0.42), yaw_rad=0.0)
        reference_b = Pose((7.0, -4.0, 2.0), yaw_rad=2.2, pitch_rad=-0.4)

        state_a = target.state_at(2.25, reference_a)
        state_b = target.state_at(2.25, reference_b)

        self.assertEqual(state_a, state_b)
        self.assertEqual(target.position_at(2.25, reference_a), target.position_at(2.25, reference_b))
        self.assertEqual(target.anchor, "world")
        self.assertTrue(state_a.visual_only)
        self.assertTrue(state_a.non_interacting)
        self.assertFalse(state_a.collidable)
        self.assertEqual(target.stimulus_state_at(2.25, reference_a)["male_pose_influences_target"], False)

    def test_virtual_world_can_trace_the_target_through_stimulus_seam(self) -> None:
        target = FemaleTrajectoryStimulus()
        world = VirtualWorld.from_json(stimuli=target)
        origin = (0.0, 0.0, 0.42)
        target_position = target.state_at(0.0).position_m
        direction = tuple(target_position[i] - origin[i] for i in range(3))
        length = math.sqrt(sum(value * value for value in direction))
        unit_direction = tuple(value / length for value in direction)

        hit = world.trace_ray(origin, unit_direction, stimuli=target)

        self.assertIsNotNone(hit)
        self.assertEqual(hit.object_id, "female_fly")
        self.assertEqual(hit.kind, "stimulus")

    def test_empirical_provenance_fields_are_preserved(self) -> None:
        trajectory = FemaleTargetTrajectory.from_mapping({
            "trajectory_id": "recorded-female-track",
            "source": {"title": "recorded courtship clip", "frame_rate_hz": 30.0},
            "keyframes": [
                {"time_s": 0.0, "position_m": [0.2, -0.02, 0.42], "yaw_rad": 0.1,
                 "frame_index": 0, "confidence": 0.99},
                {"time_s": 1.0, "position_m": [0.2, 0.02, 0.42], "yaw_rad": 0.2,
                 "frame_index": 30, "confidence": 0.95},
            ],
        })

        self.assertEqual(trajectory.trajectory_id, "recorded-female-track")
        self.assertEqual(trajectory.sample(0.5).position_m, (0.2, 0.0, 0.42))
        encoded = trajectory.to_dict()
        self.assertEqual(encoded["source"]["title"], "recorded courtship clip")
        self.assertEqual(encoded["keyframes"][0]["frame_index"], 0)
        self.assertEqual(encoded["keyframes"][1]["confidence"], 0.95)

    def test_conditions_keep_identity_and_interaction_contract(self) -> None:
        moving = FemaleTrajectoryStimulus(condition="moving")
        static = FemaleTrajectoryStimulus(condition="static")
        hidden = FemaleTrajectoryStimulus(condition="hidden")

        self.assertNotEqual(moving.state_at(3.0).position_m, moving.state_at(0.0).position_m)
        self.assertEqual(static.state_at(3.0).position_m, static.state_at(0.0).position_m)
        self.assertEqual(hidden.state_at(3.0).position_m, moving.state_at(3.0).position_m)
        self.assertTrue(moving.state_at(3.0).renderable)
        self.assertTrue(static.state_at(3.0).renderable)
        self.assertFalse(hidden.state_at(3.0).renderable)
        for target in (moving, static, hidden):
            state = target.state_at(3.0)
            self.assertEqual(target.id, "female_fly")
            self.assertTrue(state.visual_only)
            self.assertTrue(state.non_interacting)
            self.assertFalse(state.collidable)
            self.assertEqual(target.target_geometry, moving.target_geometry)

    def test_rejects_unknown_condition(self) -> None:
        with self.assertRaisesRegex(ValueError, "condition"):
            FemaleTrajectoryStimulus(condition="pursue")


if __name__ == "__main__":
    unittest.main()
