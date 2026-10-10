from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from foraging_policy import ForagingPolicy, ForagingPolicyConfig, POLICY_SCOPE  # noqa: E402


class ForagingPolicyTest(unittest.TestCase):
    def test_observation_boundary_and_validation(self) -> None:
        policy = ForagingPolicy()
        with self.assertRaisesRegex(ValueError, "bool"):
            policy.step(10.0, 10.0, 1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            policy.step(-0.1, 10.0, False)
        with self.assertRaisesRegex(ValueError, "finite"):
            policy.step(float("nan"), 10.0, False)

    def test_rate_stages_emit_bounded_neuromuscular_commands(self) -> None:
        policy = ForagingPolicy()
        takeoff = policy.step(30.0, 30.0, True)
        self.assertEqual(takeoff.phase, "takeoff")
        self.assertEqual(takeoff.command.left_power, 0.86)
        self.assertAlmostEqual(takeoff.command.left_bias_rad, 0.040)
        self.assertEqual(takeoff.command.source, "safety-handbuilt")

        for _ in range(19):
            policy.step(30.0, 30.0, False)
        approach = policy.step(30.0, 30.0, False)
        self.assertEqual(approach.phase, "odor_guided_approach")
        self.assertAlmostEqual(approach.command.left_bias_rad, 0.065)

        braking = policy.step(55.0, 55.0, False)
        self.assertEqual(braking.phase, "odor_guided_brake")
        self.assertAlmostEqual(braking.command.left_bias_rad, -0.035)

        descent = policy.step(59.0, 59.0, False)
        self.assertEqual(descent.phase, "odor_triggered_descent")
        self.assertEqual(descent.command.left_power, 0.58)
        self.assertLessEqual(descent.command.left_power, 1.0)
        self.assertGreaterEqual(descent.command.left_power, 0.0)
        self.assertIn("not MaleCNS", POLICY_SCOPE)

    def test_bilateral_turns_toward_stronger_antenna(self) -> None:
        policy = ForagingPolicy()
        left = policy.step(20.1, 20.0, False)
        right_policy = ForagingPolicy()
        right = right_policy.step(20.0, 20.1, False)
        self.assertGreater(left.command.yaw_torque_Nm, 0.0)
        self.assertLess(right.command.yaw_torque_Nm, 0.0)
        self.assertLessEqual(abs(left.command.yaw_torque_Nm), 4.0e-10)

    def test_mirrored_wind_headings_produce_mirrored_upwind_turns(self) -> None:
        left_policy = ForagingPolicy()
        right_policy = ForagingPolicy()
        left = left_policy.step(20.0, 20.0, False, wind_body_mps=(1.0, 1.0))
        right = right_policy.step(20.0, 20.0, False, wind_body_mps=(1.0, -1.0))

        self.assertAlmostEqual(left.upwind_bearing_rad, -right.upwind_bearing_rad)
        self.assertAlmostEqual(left.odor_gated_upwind_drive, -right.odor_gated_upwind_drive)
        self.assertAlmostEqual(left.command.yaw_torque_Nm, -right.command.yaw_torque_Nm)
        telemetry = left.telemetry()
        self.assertEqual(telemetry["observations"]["ambient_wind_body_mps"], [1.0, 1.0])
        derived = telemetry["derived"]
        self.assertAlmostEqual(derived["upwind_bearing_rad"], -3.0 * 3.141592653589793 / 4.0)

    def test_wind_alignment_is_gated_by_odor(self) -> None:
        wind = (1.0, 0.5)
        odor_off = ForagingPolicy().step(0.0, 0.0, False, wind_body_mps=wind)
        odor_policy = ForagingPolicy(ForagingPolicyConfig(takeoff_s=0.005))
        odor_policy.step(20.0, 20.0, False, wind_body_mps=wind)
        odor_on = odor_policy.step(20.0, 20.0, False, wind_body_mps=wind)
        self.assertEqual(odor_off.odor_gated_upwind_drive, 0.0)
        self.assertEqual(odor_off.command.yaw_torque_Nm, 0.0)
        self.assertNotEqual(odor_on.odor_gated_upwind_drive, 0.0)
        self.assertEqual(odor_on.phase, "odor_upwind_reorientation")
        self.assertEqual(odor_on.command.left_bias_rad, 0.0)

    def test_misaligned_launch_reorients_before_forward_drive(self) -> None:
        policy = ForagingPolicy(ForagingPolicyConfig(takeoff_s=0.005))
        launch = policy.step(20.0, 20.0, False, wind_body_mps=(0.0, 1.0))
        self.assertEqual(launch.phase, "takeoff")
        self.assertEqual(launch.command.left_bias_rad, 0.0)
        reorientation = policy.step(20.0, 20.0, False, wind_body_mps=(0.0, 1.0))
        self.assertEqual(reorientation.phase, "odor_upwind_reorientation")
        self.assertAlmostEqual(reorientation.command.left_power, policy.config.wind_hover_power)

    def test_wind_guided_descent_waits_for_odor_exposure_and_alignment(self) -> None:
        config = ForagingPolicyConfig(
            takeoff_s=0.005,
            wind_landing_delay_s=0.005,
            wind_landing_alignment_rad=0.20,
        )
        policy = ForagingPolicy(config)
        first = policy.step(20.0, 20.0, False, wind_body_mps=(-1.0, 0.0))
        self.assertEqual(first.phase, "takeoff")
        misaligned = policy.step(20.0, 20.0, False, wind_body_mps=(0.0, 1.0))
        self.assertEqual(misaligned.phase, "odor_upwind_reorientation")
        aligned = policy.step(20.0, 20.0, False, wind_body_mps=(-1.0, 0.0))
        self.assertEqual(aligned.phase, "odor_triggered_descent")

    def test_wind_guided_brake_precedes_descent(self) -> None:
        config = ForagingPolicyConfig(
            takeoff_s=0.005,
            wind_landing_delay_s=0.020,
            wind_landing_alignment_rad=0.20,
            wind_landing_brake_lead_s=0.010,
        )
        policy = ForagingPolicy(config)
        policy.step(20.0, 20.0, False, wind_body_mps=(-1.0, 0.0))
        braking = policy.step(20.0, 20.0, False, wind_body_mps=(-1.0, 0.0))
        self.assertEqual(braking.phase, "odor_guided_brake")
        self.assertEqual(braking.command.left_bias_rad, config.brake_bias_rad)
        descent = policy.step(20.0, 20.0, False, wind_body_mps=(-1.0, 0.0))
        self.assertEqual(descent.phase, "odor_guided_brake")
        descent = policy.step(20.0, 20.0, False, wind_body_mps=(-1.0, 0.0))
        self.assertEqual(descent.phase, "odor_triggered_descent")

    def test_hunger_scales_odor_bias_and_search_vigor(self) -> None:
        hungry = ForagingPolicy().step(20.0, 20.0, False, hunger=1.0)
        sated = ForagingPolicy().step(20.0, 20.0, False, hunger=0.0)
        self.assertGreater(hungry.command.left_bias_rad, sated.command.left_bias_rad)
        hungry_wind = ForagingPolicy().step(20.0, 20.0, False, wind_body_mps=(1.0, 1.0), hunger=1.0)
        sated_wind = ForagingPolicy().step(20.0, 20.0, False, wind_body_mps=(1.0, 1.0), hunger=0.0)
        self.assertGreater(abs(hungry_wind.odor_gated_upwind_drive), abs(sated_wind.odor_gated_upwind_drive))

        search_config = ForagingPolicyConfig(takeoff_s=0.005)
        hungry_search_policy = ForagingPolicy(search_config)
        sated_search_policy = ForagingPolicy(search_config)
        hungry_search_policy.step(0.0, 0.0, False, hunger=1.0)
        sated_search_policy.step(0.0, 0.0, False, hunger=0.0)
        hungry_search = hungry_search_policy.step(0.0, 0.0, False, hunger=1.0)
        sated_search = sated_search_policy.step(0.0, 0.0, False, hunger=0.0)
        self.assertGreater(hungry_search.command.left_power, sated_search.command.left_power)

        approach_config = ForagingPolicyConfig(takeoff_s=0.005)
        hungry_approach_policy = ForagingPolicy(approach_config)
        sated_approach_policy = ForagingPolicy(approach_config)
        hungry_approach_policy.step(20.0, 20.0, False, hunger=1.0)
        sated_approach_policy.step(20.0, 20.0, False, hunger=0.0)
        hungry_approach = hungry_approach_policy.step(20.0, 20.0, False, hunger=1.0)
        sated_approach = sated_approach_policy.step(20.0, 20.0, False, hunger=0.0)
        self.assertGreater(hungry_approach.odor_surge, sated_approach.odor_surge)

    def test_odor_loss_starts_a_bounded_crosswind_cast(self) -> None:
        policy = ForagingPolicy(ForagingPolicyConfig(takeoff_s=0.005))
        policy.step(20.0, 20.0, False, wind_body_mps=(1.0, 0.0))
        lost = policy.step(0.0, 0.0, False, wind_body_mps=(1.0, 0.0))
        self.assertEqual(lost.phase, "search_no_odor")
        self.assertNotEqual(lost.crosswind_cast_drive, 0.0)
        self.assertNotEqual(lost.command.yaw_torque_Nm, 0.0)
        self.assertLessEqual(abs(lost.command.yaw_torque_Nm), 2.0e-10)

    def test_wind_and_hunger_inputs_are_validated(self) -> None:
        policy = ForagingPolicy()
        for wind in ((1.0,), (1.0, 2.0, 3.0), [1.0, 2.0], (float("nan"), 0.0), (0.0, float("inf")), (True, 0.0)):
            with self.subTest(wind=wind), self.assertRaisesRegex(ValueError, "wind_body_mps"):
                policy.step(0.0, 0.0, False, wind_body_mps=wind)
        for hunger in (-0.01, 1.01, float("nan"), float("inf"), True):
            with self.subTest(hunger=hunger), self.assertRaisesRegex(ValueError, "hunger"):
                policy.step(0.0, 0.0, False, hunger=hunger)

    def test_support_contact_and_strong_odor_settle(self) -> None:
        policy = ForagingPolicy()
        for _ in range(20):
            decision = policy.step(59.0, 59.0, False)
        decision = policy.step(59.0, 59.0, True)
        self.assertEqual(decision.phase, "landing_settle")
        self.assertEqual(decision.command.left_bias_rad, 0.0)
        self.assertEqual(decision.command.left_power, 0.58)

    def test_temporal_drop_emits_bounded_cast_pulse(self) -> None:
        policy = ForagingPolicy(ForagingPolicyConfig(takeoff_s=0.05))
        last = None
        for index in range(35):
            rate = 30.0 - index * 0.04
            last = policy.step(rate, rate, False)
        self.assertIsNotNone(last)
        self.assertAlmostEqual(last.command.yaw_torque_Nm, 2.0e-10)
        self.assertLessEqual(abs(last.command.yaw_torque_Nm), 2.0e-10)

    def test_config_keeps_five_millisecond_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "5 ms"):
            ForagingPolicyConfig(dt_s=0.01)

if __name__ == "__main__":
    unittest.main()
