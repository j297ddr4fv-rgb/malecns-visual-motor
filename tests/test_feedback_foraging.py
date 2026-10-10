from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fitted_foraging_controller import FeedbackForagingController, FeedbackForagingConfig
from flight_dynamics import FlightDynamics, FlightDynamicsConfig, FlightState


class FeedbackForagingTest(unittest.TestCase):
    def test_feedback_arrests_climb_in_closed_loop_without_food(self):
        config = FeedbackForagingConfig()
        policy = FeedbackForagingController(config)
        dynamics = FlightDynamics(
            config=FlightDynamicsConfig(ground_plane_z_m=0.0012),
            state=FlightState.from_any({"position_m": (0.0, 0.0, 0.0012)}),
        )
        peak = 0.0
        for _ in range(1000):
            state = dynamics.state
            decision = policy.step(
                0.0, 0.0, state.position_m[2] <= 0.0012,
                wind_body_mps=(0.0, 0.0), hunger=1.0,
                height_m=state.position_m[2],
                velocity_body_mps=state.velocity_mps,
                yaw_rate_rad_s=state.angular_velocity_rad_s[2],
            )
            dynamics.step(decision.command, dt_s=0.005)
            peak = max(peak, dynamics.state.position_m[2])
        self.assertLess(peak, config.cruise_height_m + 0.02)
        self.assertAlmostEqual(dynamics.state.position_m[2], config.cruise_height_m, delta=0.01)
        self.assertLess(abs(dynamics.state.velocity_mps[2]), 0.01)
        self.assertEqual(policy.travel_s, 0.0)
        self.assertFalse(policy.descending)

    def test_rising_and_forward_motion_produce_braking_commands(self):
        decision = FeedbackForagingController().step(
            0.0, 0.0, False, wind_body_mps=(0.0, 0.0), hunger=1.0,
            height_m=0.12, velocity_body_mps=(0.3, 0.0, 0.3), yaw_rate_rad_s=0.0,
        )
        self.assertLess(decision.command.left_power, FeedbackForagingConfig().hover_power)
        self.assertLess(decision.command.left_bias_rad, 0.0)

    def test_sated_initial_support_does_not_launch(self):
        decision = FeedbackForagingController().step(
            40.0, 40.0, True, wind_body_mps=(-0.08, 0.0), hunger=0.0,
            height_m=0.0012, velocity_body_mps=(0.0, 0.0, 0.0), yaw_rate_rad_s=0.0,
        )
        self.assertEqual(decision.phase, "rest_sated")
        self.assertEqual(decision.command.left_power, 0.0)


if __name__ == "__main__":
    unittest.main()
