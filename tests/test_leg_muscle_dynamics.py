"""Physical invariants for the reduced Hill-type leg muscle."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from leg_muscle_dynamics import (  # noqa: E402
    HillMuscle,
    MuscleParameters,
)


class HillMuscleTest(unittest.TestCase):
    def test_default_is_quiet_and_zero_excitation_has_no_active_force(self) -> None:
        muscle = HillMuscle()
        self.assertEqual(muscle.state.activation, 0.0)
        self.assertEqual(muscle.state.total_force_N, 0.0)

        state = muscle.step(0.0, 0.0006, 0.0, 0.001)
        self.assertEqual(state.active_force_N, 0.0)
        self.assertEqual(state.passive_force_N, 0.0)
        self.assertEqual(state.total_force_N, 0.0)

    def test_stretched_muscle_has_passive_tension_without_active_drive(self) -> None:
        muscle = HillMuscle()
        state = muscle.step(0.0, 0.00072, 0.0, 0.001)

        self.assertEqual(state.active_force_N, 0.0)
        self.assertGreater(state.passive_force_N, 0.0)
        self.assertEqual(state.total_force_N, state.passive_force_N)
        self.assertGreaterEqual(state.total_force_N, 0.0)

    def test_active_length_tension_peaks_at_optimal_length(self) -> None:
        muscle = HillMuscle()
        muscle.reset(activation=1.0)
        optimal = muscle.step(1.0, 0.0006, 0.0, 0.0)
        short = muscle.step(1.0, 0.00048, 0.0, 0.0)
        long = muscle.step(1.0, 0.00072, 0.0, 0.0)

        self.assertAlmostEqual(optimal.active_force_N, 1.0e-5)
        self.assertGreater(optimal.active_force_N, short.active_force_N)
        self.assertGreater(optimal.active_force_N, long.active_force_N)
        self.assertAlmostEqual(optimal.passive_force_N, 0.0)

    def test_shortening_reduces_and_lengthening_increases_bounded_active_force(self) -> None:
        params = MuscleParameters()
        muscle = HillMuscle(params)
        muscle.reset(activation=1.0)
        isometric = muscle.step(1.0, params.optimal_length_m, 0.0, 0.0)
        shortening = muscle.step(1.0, params.optimal_length_m, -0.0003, 0.0)
        lengthening = muscle.step(1.0, params.optimal_length_m, 0.0003, 0.0)
        fast_lengthening = muscle.step(1.0, params.optimal_length_m, 1.0, 0.0)
        beyond_max_shortening = muscle.step(1.0, params.optimal_length_m, -1.0, 0.0)

        self.assertLess(shortening.active_force_N, isometric.active_force_N)
        self.assertGreater(lengthening.active_force_N, isometric.active_force_N)
        self.assertLessEqual(
            lengthening.active_force_N,
            params.eccentric_force_multiplier * isometric.active_force_N,
        )
        self.assertLessEqual(
            fast_lengthening.active_force_N,
            params.eccentric_force_multiplier * isometric.active_force_N,
        )
        self.assertEqual(beyond_max_shortening.active_force_N, 0.0)
        self.assertGreater(shortening.power_W, 0.0)
        self.assertLess(lengthening.power_W, 0.0)

    def test_activation_and_deactivation_use_distinct_time_constants(self) -> None:
        params = MuscleParameters(
            activation_time_constant_s=0.010,
            deactivation_time_constant_s=0.040,
        )
        muscle = HillMuscle(params)
        activated = muscle.step(1.0, params.optimal_length_m, 0.0, 0.010)
        expected_activation = 1.0 - math.exp(-1.0)
        self.assertAlmostEqual(activated.activation, expected_activation)

        deactivated = muscle.step(0.0, params.optimal_length_m, 0.0, 0.010)
        expected_deactivation = expected_activation * math.exp(-0.25)
        self.assertAlmostEqual(deactivated.activation, expected_deactivation)
        self.assertLess(
            activated.activation - deactivated.activation,
            activated.activation,
        )

    def test_exact_activation_update_is_invariant_to_fixed_input_dt_subdivision(self) -> None:
        params = MuscleParameters(activation_time_constant_s=0.02)
        one_tick = HillMuscle(params)
        many_ticks = HillMuscle(params)

        large = one_tick.step(0.8, params.optimal_length_m, 0.0, 0.04)
        for _ in range(4):
            split = many_ticks.step(0.8, params.optimal_length_m, 0.0, 0.01)

        self.assertAlmostEqual(large.activation, split.activation, places=14)
        self.assertAlmostEqual(large.active_force_N, split.active_force_N, places=18)
        self.assertAlmostEqual(large.time_s, split.time_s, places=14)

    def test_force_is_tensile_and_passive_curve_has_explicit_cap(self) -> None:
        params = MuscleParameters()
        muscle = HillMuscle(params)
        state = muscle.step(0.0, params.optimal_length_m * 1.0e200, 0.0, 0.0)

        self.assertEqual(state.active_force_N, 0.0)
        self.assertEqual(
            state.passive_force_N,
            params.max_isometric_force_N * params.passive_force_cap_multiplier,
        )
        self.assertGreaterEqual(state.active_force_N, 0.0)
        self.assertGreaterEqual(state.passive_force_N, 0.0)
        self.assertGreaterEqual(state.total_force_N, 0.0)

    def test_reset_clears_elapsed_time_and_invalid_inputs_do_not_advance_state(self) -> None:
        muscle = HillMuscle()
        muscle.step(1.0, 0.0006, 0.0, 0.02)
        before = muscle.state
        with self.assertRaises(ValueError):
            muscle.step(float("nan"), 0.0006, 0.0, 0.01)
        with self.assertRaises(ValueError):
            muscle.step(0.5, 0.0006, 0.0, -0.01)
        with self.assertRaises(ValueError):
            muscle.step(1.1, 0.0006, 0.0, 0.01)
        self.assertEqual(muscle.state, before)

        reset = muscle.reset()
        self.assertEqual(reset.time_s, 0.0)
        self.assertEqual(reset.activation, 0.0)
        self.assertEqual(reset.length_m, muscle.params.optimal_length_m)
        self.assertEqual(reset.total_force_N, 0.0)

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MuscleParameters(max_isometric_force_N=0.0)
        with self.assertRaises(ValueError):
            MuscleParameters(optimal_length_m=float("nan"))
        with self.assertRaises(ValueError):
            MuscleParameters(eccentric_force_multiplier=0.9)


if __name__ == "__main__":
    unittest.main()
