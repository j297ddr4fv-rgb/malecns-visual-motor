from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fit_foraging_policy as fit  # noqa: E402
from foraging_policy import ForagingPolicyConfig  # noqa: E402
from fitted_foraging_controller import FeedbackForagingConfig  # noqa: E402
from run_foraging_baseline import load_policy_config as load_runner_policy_config  # noqa: E402


class FitForagingPolicyTest(unittest.TestCase):
    def test_training_and_heldout_yaws_are_disjoint(self) -> None:
        training = fit._scenario_list(fit.DEFAULT_TRAIN_YAWS_DEG, "train")
        holdout = fit._scenario_list(fit.DEFAULT_HOLDOUT_YAWS_DEG, "holdout")

        self.assertEqual(fit._yaw_key(3.141592653589793), fit._yaw_key(-3.141592653589793))
        self.assertEqual(fit._yaw_key(3.0 * 3.141592653589793), fit._yaw_key(-3.141592653589793))
        fit._validate_scenario_split(training, holdout)
        feedback_holdout = fit._scenario_list(fit.DEFAULT_FEEDBACK_HOLDOUT_YAWS_DEG, "holdout")
        fit._validate_scenario_split(training, feedback_holdout)
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit._validate_scenario_split(
                training,
                fit._scenario_list((45.0, 0.0), "holdout"),
            )

    def test_objective_receives_only_the_explicit_training_scenarios(self) -> None:
        training = fit._scenario_list((0.0, 90.0, -90.0, 180.0), "train")
        seen: list[str] = []

        def fake_rollout(
            _env, _config, scenario, *, duration_s, initial_hunger, odor_off,
            controller_type=fit.LEGACY_CONTROLLER_TYPE,
        ):
            seen.append(scenario.name)
            return {
                "scenario": scenario.name,
                "case_score": 0.5,
                "duration_s": duration_s,
                "initial_hunger": initial_hunger,
                "odor_off": odor_off,
                "controller_type": controller_type,
            }

        with mock.patch.object(fit, "_simulate_case", side_effect=fake_rollout):
            result = fit._evaluate_scenarios(
                object(),
                ForagingPolicyConfig(),
                training,
                duration_s=5.0,
                initial_hunger=1.0,
            )

        self.assertEqual(seen, [scenario.name for scenario in training])
        self.assertEqual(result["aggregate_score"], 0.5)
        self.assertEqual(
            [item["scenario"] for item in result["scenarios"]],
            [scenario.name for scenario in training],
        )

    def test_artifact_round_trips_through_fit_and_baseline_loaders(self) -> None:
        config = ForagingPolicyConfig(takeoff_s=0.125, max_yaw_torque_Nm=8.0e-10)
        artifact = {
            "artifact_type": "fitted_parameters_of_handbuilt_policy",
            "policy_config": asdict(config),
            "method": {"name": "test fixture"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fitted_policy.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")

            fit_config, loaded_artifact = fit.load_policy_config(path)
            runner_config = load_runner_policy_config(path)

        self.assertEqual(fit_config, config)
        self.assertEqual(runner_config, config)
        self.assertEqual(loaded_artifact["policy_config"], asdict(config))

    def test_feedback_artifact_round_trips_with_controller_type(self) -> None:
        config = FeedbackForagingConfig(cruise_height_m=0.14, transit_time_s=1.8)
        artifact = {
            "artifact_type": "fitted_parameters_of_handbuilt_policy",
            "controller_type": fit.FEEDBACK_CONTROLLER_TYPE,
            "policy_config": asdict(config),
            "training_provenance": {"seed": 1729},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback_policy.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            fit_config, loaded_artifact = fit.load_policy_config(path)
            runner_config = load_runner_policy_config(path)

        self.assertEqual(fit_config, config)
        self.assertEqual(runner_config, config)
        self.assertEqual(loaded_artifact, artifact)

    def test_world_velocity_is_rotated_to_body_axes(self) -> None:
        body_velocity = fit._body_velocity_mps((1.0, 0.0, 0.0), 3.141592653589793 / 2.0)
        self.assertAlmostEqual(body_velocity[0], 0.0)
        self.assertAlmostEqual(body_velocity[1], -1.0)
        self.assertEqual(body_velocity[2], 0.0)

    def test_loaders_reject_malformed_and_out_of_bounds_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_policy.json"
            path.write_text(json.dumps({"unrelated": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy_config"):
                fit.load_policy_config(path)
            with self.assertRaisesRegex(ValueError, "policy_config"):
                load_runner_policy_config(path)

            invalid = asdict(ForagingPolicyConfig())
            invalid["hover_power"] = 1.5
            path.write_text(json.dumps({
                "artifact_type": "fitted_parameters_of_handbuilt_policy",
                "policy_config": invalid,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hover_power"):
                fit.load_policy_config(path)
            with self.assertRaisesRegex(ValueError, "hover_power"):
                load_runner_policy_config(path)

    def test_cross_entropy_search_is_seeded_and_respects_parameter_bounds(self) -> None:
        def evaluator(config: ForagingPolicyConfig):
            # A fixed smooth proxy objective keeps this optimizer unit test
            # cheap without running a flight simulation.
            score = 1.0 - abs(config.low_odor_power - 0.55)
            return {"aggregate_score": score}

        first = fit._cross_entropy_search(
            ForagingPolicyConfig(), evaluator, generations=2, population=4, seed=19
        )
        second = fit._cross_entropy_search(
            ForagingPolicyConfig(), evaluator, generations=2, population=4, seed=19
        )

        self.assertEqual(first, second)
        for name, (low, high) in fit.PARAMETER_BOUNDS.items():
            self.assertGreaterEqual(getattr(first[0], name), low)
            self.assertLessEqual(getattr(first[0], name), high)


if __name__ == "__main__":
    unittest.main()
