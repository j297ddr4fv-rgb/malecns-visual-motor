#!/usr/bin/env python3
"""Contract and continuous-proxy tests for the optional body stabilizer."""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from flight_body_stabilizer import (  # noqa: E402
    FlightBodyStabilizer,
    FlightBodyStabilizerConfig,
    HAND_BUILT_SOURCE,
)
from flight_dynamics import FlightDynamics, FlightState  # noqa: E402


def _rpy(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


class FlightBodyStabilizerTest(unittest.TestCase):
    def test_manifest_disable_and_provenance_are_explicit(self) -> None:
        stabilizer = FlightBodyStabilizer.from_json()
        manifest = stabilizer.manifest()
        self.assertEqual(manifest["schema_version"], "flight-body-stabilizer-1")
        self.assertEqual(manifest["provenance"]["source"], HAND_BUILT_SOURCE)
        self.assertEqual(manifest["provenance"]["confidence"], "low")
        self.assertFalse(manifest["provenance"]["biological_claim"])
        self.assertIn("docs/research/embodied-precedents.md", manifest["provenance"]["research_reference"])
        self.assertFalse(manifest["authority"]["writes_body_state"])
        self.assertTrue(manifest["authority"]["preserves_left_right_steering"])
        json.dumps(manifest, sort_keys=True)

        first = stabilizer.step(
            {"altitude_error_m": 0.15, "forward_error_m": 0.2},
            {"pitch_error_rad": 0.01, "roll_error_rad": -0.02},
            "forward",
            {"left_wing": 0.60, "right_wing": 0.42, "yaw": 0.31},
            dt_s=0.005,
        )
        self.assertLessEqual(first.stabilizer_blend, 0.35)
        self.assertEqual(first.source, "fitted")
        self.assertEqual(first.blend_source, "fitted+handbuilt")
        self.assertEqual(first.provenance["neural"], "fitted")
        self.assertEqual(first.provenance["stabilizer"], HAND_BUILT_SOURCE)
        self.assertEqual(first.diagnostics["left_right_neural_difference"], first.diagnostics["left_right_output_difference"])
        self.assertEqual(first.channels["yaw"], 0.31)

        stabilizer.disable()
        disabled = stabilizer.step(
            {"altitude_error_m": 10.0, "forward_error_m": 10.0},
            {"pitch_error_rad": 2.0, "roll_error_rad": 2.0},
            "forward",
            {"left_wing": 0.60, "right_wing": 0.42, "yaw": 0.31},
            dt_s=0.005,
        )
        self.assertFalse(disabled.active)
        self.assertEqual(disabled.stabilizer_blend, 0.0)
        self.assertEqual(disabled.channels["left_wing"], 0.60)
        self.assertEqual(disabled.channels["right_wing"], 0.42)
        self.assertEqual(disabled.channels["yaw"], 0.31)
        self.assertEqual(disabled.channels["pitch"], 0.0)
        stabilizer.reset()
        self.assertEqual(stabilizer.sequence, 0)
        self.assertEqual(stabilizer.time_s, 0.0)
        self.assertIsNone(stabilizer.last_output)

    def test_bounded_blend_keeps_neural_steering_and_ignores_pose(self) -> None:
        stabilizer = FlightBodyStabilizer.from_json()
        fitted = {
            "left_wing": 0.91,
            "right_wing": 0.23,
            "yaw": -0.7,
            "pitch": 0.1,
            "roll": -0.2,
            "brake": 0.4,
        }
        # Position-like fields are deliberately irrelevant.  The module only
        # accepts precomputed error scalars and cannot write this mapping.
        body_record = {
            "position_m": [91.0, 92.0, 93.0],
            "velocity_mps": [4.0, 0.0, 0.0],
            "pitch_error_rad": -5.0,
            "roll_error_rad": 4.0,
            "pitch_rate_rad_s": 2.0,
            "roll_rate_rad_s": -2.0,
        }
        before = dict(body_record)
        result = stabilizer.step(
            {"position_m": [100.0, 100.0, 100.0], "forward_error_m": 0.4, "lateral_error_m": -0.3},
            body_record,
            "gust-recovery",
            fitted,
            dt_s=0.005,
        )
        self.assertEqual(body_record, before)
        for name, value in result.channels.items():
            if name in {"left_wing", "right_wing", "thrust", "brake"}:
                self.assertGreaterEqual(value, 0.0, name)
                self.assertLessEqual(value, 1.0, name)
            elif name in {"yaw", "pitch", "roll"}:
                self.assertGreaterEqual(value, -1.0, name)
                self.assertLessEqual(value, 1.0, name)
        self.assertEqual(result.channels["yaw"], fitted["yaw"])
        neural_difference = fitted["right_wing"] - fitted["left_wing"]
        output_difference = result.channels["right_wing"] - result.channels["left_wing"]
        self.assertAlmostEqual(output_difference, neural_difference, places=12)
        self.assertLessEqual(result.handbuilt_fraction, stabilizer.config.max_contribution_fraction)
        self.assertTrue(result.provenance["writes_body_state"] is False)

    def test_optional_body_frame_velocity_damping_is_scalar_and_bounded(self) -> None:
        config = FlightBodyStabilizerConfig.from_mapping(
            {"gains": {
                "pitch_forward_velocity": -0.45,
                "roll_lateral_velocity": 0.50,
            }}
        )
        stabilizer = FlightBodyStabilizer(config)
        result = stabilizer.step(
            {"forward_error_m": 0.0, "lateral_error_m": 0.0},
            {
                "forward_velocity_mps": 0.40,
                "lateral_velocity_mps": -0.40,
                "pitch_error_rad": 0.0,
                "roll_error_rad": 0.0,
            },
            "gust-recovery",
            {"left_wing": 0.6, "right_wing": 0.6, "thrust": 0.6},
            dt_s=0.005,
        )
        self.assertAlmostEqual(
            result.diagnostics["goal_body_errors"]["lateral_velocity_mps"],
            -0.40,
        )
        self.assertAlmostEqual(
            result.diagnostics["goal_body_errors"]["forward_velocity_mps"],
            0.40,
        )
        self.assertAlmostEqual(result.diagnostics["pitch_raw"], -0.18)
        self.assertAlmostEqual(result.diagnostics["roll_raw"], -0.20)
        self.assertGreaterEqual(result.channels["roll"], -1.0)
        self.assertLessEqual(result.channels["roll"], 1.0)

    def test_multi_second_proxy_mission_with_flight_dynamics(self) -> None:
        """Exercise takeoff, forward translation, attitude, gust, and landing.

        The wing power/steering packet is the fitted input.  The stabilizer
        only contributes its bounded pitch/roll/collective overlay; all pose
        integration remains inside :class:`FlightDynamics`.
        """

        dt = 0.005
        trim = FlightDynamics().hover_trim().left_power
        dynamics = FlightDynamics(state=FlightState(position_m=(0.0, -0.55, 0.015)))
        stabilizer = FlightBodyStabilizer.from_json()
        max_attitude = 0.0
        max_speed = 0.0
        max_altitude = 0.0
        forward_at_boundary = 0.0
        hover_speed = 0.0
        gust_speed = 0.0
        gust_end_speed = 0.0
        max_steering_change = 0.0
        previous_neural_difference = None

        for index in range(int(9.0 / dt)):
            t_s = index * dt
            if t_s < 1.2:
                goal = (0.0, -0.55, 0.015 + _smoothstep(t_s / 1.2) * (0.38 - 0.015))
                phase = "takeoff"
                # A short high-power fitted takeoff pulse, then neural trim.
                power = 1.0 if t_s < 0.20 else trim
                neural_yaw = 0.0
            elif t_s < 4.0:
                alpha = _smoothstep((t_s - 1.2) / 2.8)
                goal = (0.72 * alpha, -0.55 + 0.48 * alpha, 0.38)
                phase = "forward"
                power = trim
                neural_yaw = 0.0
            elif t_s < 4.8:
                alpha = _smoothstep((t_s - 4.0) / 0.8)
                goal = (0.72, -0.07, 0.34)
                phase = "turn"
                power = trim
                # This is CNS/fitted steering.  The stabilizer must preserve
                # the resulting left/right difference exactly.
                neural_yaw = -0.035 * alpha
            elif t_s < 5.8:
                goal = (0.72, -0.07, 0.34)
                phase = "hover"
                power = trim
                neural_yaw = 0.0
            elif t_s < 6.6:
                goal = (0.72, -0.07, 0.34)
                phase = "gust-recovery"
                power = trim
                neural_yaw = 0.0
            elif t_s < 7.4:
                alpha = _smoothstep((t_s - 6.6) / 0.8)
                goal = (0.72, -0.07, 0.34 - 0.24 * alpha)
                phase = "approach"
                power = trim
                neural_yaw = 0.0
            else:
                alpha = _smoothstep((t_s - 7.4) / 1.6)
                goal = (0.72, -0.07, 0.10 - 0.065 * alpha)
                phase = "landing"
                # A fitted neural descent command; the overlay may only trim
                # it, never replace it with a position/landing script.
                power = trim * max(0.0, 1.0 - (t_s - 7.4) / 1.6)
                neural_yaw = 0.0

            state = dynamics.state
            roll, pitch, yaw = _rpy(state.orientation_quat)
            dx = goal[0] - state.position_m[0]
            dy = goal[1] - state.position_m[1]
            dz = goal[2] - state.position_m[2]
            forward_error = math.cos(yaw) * dx + math.sin(yaw) * dy
            lateral_error = -math.sin(yaw) * dx + math.cos(yaw) * dy
            fitted = {
                "left_wing": max(0.0, min(1.0, power - neural_yaw)),
                "right_wing": max(0.0, min(1.0, power + neural_yaw)),
                "thrust": power,
                "yaw": neural_yaw,
                "pitch": 0.0,
                "roll": 0.0,
                "brake": 0.0,
            }
            result = stabilizer.step(
                {
                    "altitude_error_m": dz,
                    "forward_error_m": forward_error,
                    "lateral_error_m": lateral_error,
                },
                {
                    "pitch_error_rad": -pitch,
                    "roll_error_rad": -roll,
                    "vertical_speed_mps": state.velocity_mps[2],
                    "pitch_rate_rad_s": state.angular_velocity_rad_s[1],
                    "roll_rate_rad_s": state.angular_velocity_rad_s[0],
                },
                phase,
                fitted,
                dt_s=dt,
            )
            if phase == "gust-recovery" and 5.8 <= t_s < 6.25:
                dynamics.set_wind((0.0, -0.34, 0.0))
            elif phase == "approach" and abs(t_s - 6.6) < 1.0e-12:
                dynamics.set_wind((0.0, 0.0, 0.0))
            output = dynamics.step(result.channels, dt_s=dt)
            values = (
                *output.state.position_m,
                *output.state.velocity_mps,
                *output.state.orientation_quat,
                *output.state.angular_velocity_rad_s,
            )
            self.assertTrue(all(math.isfinite(float(value)) for value in values))
            roll_after, pitch_after, _ = _rpy(output.state.orientation_quat)
            max_attitude = max(max_attitude, abs(roll_after), abs(pitch_after))
            max_speed = max(max_speed, math.sqrt(sum(value * value for value in output.state.velocity_mps)))
            max_altitude = max(max_altitude, output.state.position_m[2])
            max_steering_change = max(
                max_steering_change,
                abs(result.channels["right_wing"] - result.channels["left_wing"]
                    - (fitted["right_wing"] - fitted["left_wing"])),
            )
            if abs(t_s - 1.2) < 1.0e-12:
                forward_at_boundary = output.state.position_m[0]
            if 4.8 <= t_s < 5.8:
                hover_speed = math.sqrt(sum(value * value for value in output.state.velocity_mps))
            if 5.8 <= t_s < 6.6:
                gust_speed = max(gust_speed, math.sqrt(sum(value * value for value in output.state.velocity_mps)))
            if abs(t_s - 6.6) < 1.0e-12:
                gust_end_speed = math.sqrt(sum(value * value for value in output.state.velocity_mps))
            previous_neural_difference = fitted["right_wing"] - fitted["left_wing"]

        self.assertGreater(dynamics.state.position_m[2], -1.0e-12)
        self.assertGreater(forward_at_boundary, -0.01)  # takeoff was physical, not a pose write
        self.assertGreater(dynamics.state.position_m[0], 0.40)  # forward displacement
        self.assertGreater(max_altitude, 0.20)  # physical takeoff altitude gain
        self.assertLess(max_attitude, 0.65)  # bounded roll/pitch throughout
        self.assertLess(max_speed, 1.5)
        self.assertLess(hover_speed, 0.55)
        self.assertLess(gust_end_speed, 0.85)
        self.assertGreaterEqual(gust_speed, gust_end_speed)
        self.assertAlmostEqual(max_steering_change, 0.0, places=12)
        self.assertGreaterEqual(dynamics.state.position_m[2], dynamics.config.ground_plane_z_m)
        self.assertAlmostEqual(dynamics.state.position_m[2], dynamics.config.ground_plane_z_m, places=12)
        self.assertAlmostEqual(dynamics.state.velocity_mps[2], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
