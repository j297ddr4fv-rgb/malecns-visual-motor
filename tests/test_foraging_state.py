from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from foraging_state import ForagingState  # noqa: E402


class ForagingStateTest(unittest.TestCase):
    def test_hunger_is_bounded_at_both_ends(self) -> None:
        state = ForagingState(
            initial_hunger=0.5,
            basal_hunger_per_s=0.2,
            satiety_per_s=0.5,
        )
        self.assertEqual(state.advance(10.0, False), 1.0)
        self.assertEqual(state.advance(10.0, True), 0.0)
        self.assertGreaterEqual(state.hunger, 0.0)
        self.assertLessEqual(state.hunger, 1.0)

    def test_satiety_requires_edible_contact(self) -> None:
        no_contact = ForagingState(
            initial_hunger=0.8,
            basal_hunger_per_s=0.1,
            satiety_per_s=0.4,
        )
        with_contact = ForagingState(
            initial_hunger=0.8,
            basal_hunger_per_s=0.1,
            satiety_per_s=0.4,
        )

        self.assertAlmostEqual(no_contact.advance(1.0, False), 0.9)
        self.assertAlmostEqual(with_contact.advance(1.0, True), 0.5)
        self.assertEqual(no_contact.cumulative_edible_contact_s, 0.0)
        self.assertEqual(with_contact.cumulative_edible_contact_s, 1.0)

    def test_basal_hunger_increase_is_configurable(self) -> None:
        slow = ForagingState(initial_hunger=0.2, basal_hunger_per_s=0.05)
        fast = ForagingState(initial_hunger=0.2, basal_hunger_per_s=0.15)

        self.assertAlmostEqual(slow.advance(2.0, False), 0.3)
        self.assertAlmostEqual(fast.advance(2.0, False), 0.5)

    def test_telemetry_labels_approximation_scope(self) -> None:
        telemetry = ForagingState().telemetry()
        self.assertEqual(telemetry["hunger"], 1.0)
        self.assertEqual(telemetry["cumulative_edible_contact_s"], 0.0)
        self.assertEqual(telemetry["status"], "engineering_approximation")
        self.assertIn("not biologically validated", telemetry["scope"])

    def test_constructor_validates_parameters(self) -> None:
        for kwargs in (
            {"initial_hunger": -0.1},
            {"initial_hunger": 1.1},
            {"initial_hunger": float("nan")},
            {"basal_hunger_per_s": -0.1},
            {"satiety_per_s": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ForagingState(**kwargs)

    def test_advance_validates_duration_and_exact_contact_bool(self) -> None:
        invalid_durations = (0.0, -0.1, float("nan"), float("inf"), True)
        for dt_s in invalid_durations:
            with self.subTest(dt_s=dt_s), self.assertRaises(ValueError):
                ForagingState().advance(dt_s, False)

        for contact in (0, 1, None, "true"):
            with self.subTest(contact=contact), self.assertRaisesRegex(ValueError, "bool"):
                ForagingState().advance(0.1, contact)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
