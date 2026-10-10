#!/usr/bin/env python3
"""Pure geometry, validation, and contact-gating tests for food cues."""

from __future__ import annotations

import math
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from food_sensing import FoodCueModel  # noqa: E402


def config() -> dict:
    return {
        "schema_version": 1,
        "concentration_unit": "odor_unit",
        "provenance": "hand-built isotropic demonstration field",
        "contact_surface_id": "food_surface",
        "field": {
            "kind": "isotropic_gaussian",
            "center_m": [0.0, 1.0, 0.0],
            "amplitude": 10.0,
            "background": 1.0,
            "sigma_m": 1.0,
        },
        "antenna_offsets_m": {
            "left": [1.0, 0.0, 0.0],
            "right": [-1.0, 0.0, 0.0],
        },
        "mouth_offset_m": [0.0, 0.0, 0.0],
        "edible_region": {"center_m": [0.0, 0.0, 0.0], "radius_m": 0.25},
        "odor_selector_weights_hz_per_unit": {
            "left": {"ORN_DM1/L": 2.0, "ORN_VA2/L": 3.0},
            "right": {"ORN_DM1/R": 4.0, "ORN_VA2/R": 5.0},
        },
        "sugar_selector_rates_hz": {"SUGAR": 55.0},
        "selector_expected_counts": {
            "ORN_DM1/L": 35,
            "ORN_VA2/L": 33,
            "ORN_DM1/R": 39,
            "ORN_VA2/R": 41,
            "SUGAR": 12,
        },
    }


class FoodCueModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FoodCueModel.from_mapping(config())

    def test_rotated_bilateral_samples_drive_only_the_matching_selectors(self) -> None:
        sample = self.model.sample((0.0, 0.0, 0.0), math.pi / 2.0, 0.5, False)
        left = 11.0
        right = 1.0 + 10.0 * math.exp(-2.0)
        self.assertAlmostEqual(sample.telemetry["left_sample"], left)
        self.assertAlmostEqual(sample.telemetry["right_sample"], right)
        self.assertAlmostEqual(sample.poisson_hz["ORN_DM1/L"], left * 2.0)
        self.assertAlmostEqual(sample.poisson_hz["ORN_VA2/L"], left * 3.0)
        self.assertAlmostEqual(sample.poisson_hz["ORN_DM1/R"], right * 4.0)
        self.assertAlmostEqual(sample.poisson_hz["ORN_VA2/R"], right * 5.0)
        self.assertEqual(sample.poisson_hz["SUGAR"], 0.0)
        self.assertEqual(sample.odor_poisson_hz["ORN_DM1/L"], sample.poisson_hz["ORN_DM1/L"])
        self.assertEqual(sample.sugar_poisson_hz["SUGAR"], 0.0)
        self.assertNotIn("ORN_DA1", sample.poisson_hz)

    def test_sugar_requires_physical_contact_and_mouth_inside_edible_region(self) -> None:
        inside_without_contact = self.model.sample((0.0, 0.0, 0.0), 0.0, 1.0, False)
        inside_with_contact = self.model.sample((0.0, 0.0, 0.0), 0.0, 1.0, True)
        outside_with_contact = self.model.sample((1.0, 0.0, 0.0), 0.0, 1.0, True)

        self.assertFalse(inside_without_contact.telemetry["contact"])
        self.assertTrue(inside_without_contact.telemetry["edible_region_contact"])
        self.assertEqual(inside_without_contact.poisson_hz["SUGAR"], 0.0)
        self.assertTrue(inside_with_contact.telemetry["contact"])
        self.assertEqual(inside_with_contact.poisson_hz["SUGAR"], 55.0)
        self.assertEqual(inside_with_contact.sugar_poisson_hz["SUGAR"], 55.0)
        self.assertNotIn("SUGAR", inside_with_contact.odor_poisson_hz)
        self.assertFalse(outside_with_contact.telemetry["edible_region_contact"])
        self.assertTrue(outside_with_contact.telemetry["physical_contact"])
        self.assertFalse(outside_with_contact.telemetry["contact"])
        self.assertEqual(outside_with_contact.poisson_hz["SUGAR"], 0.0)
        self.assertAlmostEqual(outside_with_contact.telemetry["mouth_distance_m"], 1.0)

    def test_boundary_of_edible_sphere_counts_as_inside(self) -> None:
        sample = self.model.sample((0.25, 0.0, 0.0), 0.0, 0.0, True)
        self.assertAlmostEqual(sample.telemetry["mouth_distance_m"], 0.25)
        self.assertTrue(sample.telemetry["contact"])
        self.assertEqual(sample.poisson_hz["SUGAR"], 55.0)

    def test_output_is_deterministic_and_has_provenance_and_units(self) -> None:
        first = self.model.sample((0.1, 0.2, -0.1), 0.7, 2.0, True).as_dict()
        second = self.model.sample((0.1, 0.2, -0.1), 0.7, 2.0, True).as_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["telemetry"]["provenance"], "hand-built isotropic demonstration field")
        self.assertEqual(first["telemetry"]["concentration_unit"], "odor_unit")
        self.assertEqual(first["telemetry"]["contact_surface_id"], "food_surface")
        self.assertEqual(self.model.selector_names, (
            "ORN_DM1/L", "ORN_DM1/R", "ORN_VA2/L", "ORN_VA2/R", "SUGAR"
        ))
        self.assertEqual(dict(self.model.selector_expected_counts)["ORN_DM1/L"], 35)
        self.assertIn("concentration", first["telemetry"])
        self.assertIn("left_sample", first["telemetry"])
        self.assertIn("right_sample", first["telemetry"])

    def test_contact_must_be_a_real_boolean(self) -> None:
        for not_a_bool in (1, 0, "true", None):
            with self.subTest(value=not_a_bool), self.assertRaises(ValueError):
                self.model.sample((0.0, 0.0, 0.0), 0.0, 0.0, not_a_bool)  # type: ignore[arg-type]

    def test_time_is_validated_but_does_not_modulate_a_static_field(self) -> None:
        at_zero = self.model.sample((0.0, 0.0, 0.0), 0.0, 0.0, False)
        later = self.model.sample((0.0, 0.0, 0.0), 0.0, 20.0, False)
        self.assertEqual(at_zero.poisson_hz, later.poisson_hz)
        self.assertEqual(later.telemetry["time_s"], 20.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.model.sample((0.0, 0.0, 0.0), 0.0, -0.1, False)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model.sample((0.0, 0.0, 0.0), float("nan"), 0.0, False)

    def test_semantic_role_cannot_replace_physical_food_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "field"):
            FoodCueModel.from_mapping({"schema_version": 1, "semantic_role": "food"})

    def test_odor_and_sugar_maps_must_be_disjoint(self) -> None:
        overlap = config()
        overlap["sugar_selector_rates_hz"]["ORN_DM1/L"] = 55.0
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            FoodCueModel.from_mapping(overlap)

    def test_unknown_schema_keys_and_invalid_expected_counts_fail_closed(self) -> None:
        typo = config()
        typo["odor_selector_weight_hz_per_unit"] = {}
        with self.assertRaisesRegex(ValueError, "unknown key"):
            FoodCueModel.from_mapping(typo)

        undeclared_count = config()
        undeclared_count["selector_expected_counts"]["ORN_DA1"] = 1
        with self.assertRaisesRegex(ValueError, "undeclared selector"):
            FoodCueModel.from_mapping(undeclared_count)

        zero_count = config()
        zero_count["selector_expected_counts"]["SUGAR"] = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            FoodCueModel.from_mapping(zero_count)

    def test_rate_maps_are_read_only_but_copyable(self) -> None:
        sample = self.model.sample((0.0, 0.0, 0.0), 0.0, 0.0, False)
        with self.assertRaises(TypeError):
            sample.odor_poisson_hz["new"] = 1.0  # type: ignore[index]
        editable_copy = dict(sample.odor_poisson_hz)
        editable_copy["new"] = 1.0
        self.assertEqual(editable_copy["new"], 1.0)

    def test_validation_requires_declared_units_and_physical_parameters(self) -> None:
        invalid_configs = []
        missing_unit = config()
        missing_unit.pop("concentration_unit")
        invalid_configs.append((missing_unit, "concentration_unit"))

        zero_sigma = config()
        zero_sigma["field"]["sigma_m"] = 0.0
        invalid_configs.append((zero_sigma, "sigma_m"))

        nan_amplitude = config()
        nan_amplitude["field"]["amplitude"] = float("nan")
        invalid_configs.append((nan_amplitude, "finite"))

        bad_antenna = config()
        bad_antenna["antenna_offsets_m"]["left"] = [0.0, float("inf"), 0.0]
        invalid_configs.append((bad_antenna, "finite"))

        negative_radius = config()
        negative_radius["edible_region"]["radius_m"] = -0.1
        invalid_configs.append((negative_radius, "radius_m"))

        negative_weight = config()
        negative_weight["odor_selector_weights_hz_per_unit"]["left"]["ORN_DM1/L"] = -1.0
        invalid_configs.append((negative_weight, "nonnegative"))

        duplicate_selector = config()
        duplicate_selector["odor_selector_weights_hz_per_unit"]["right"]["ORN_DM1/L"] = 1.0
        invalid_configs.append((duplicate_selector, "one antenna side"))

        wrong_kind = config()
        wrong_kind["field"]["kind"] = "semantic_role_gaussian"
        invalid_configs.append((wrong_kind, "isotropic_gaussian"))

        for invalid, message in invalid_configs:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                FoodCueModel.from_mapping(invalid)

    def test_selector_maps_are_explicit_and_empty_maps_are_supported(self) -> None:
        empty = config()
        empty["odor_selector_weights_hz_per_unit"] = {"left": {}, "right": {}}
        empty["sugar_selector_rates_hz"] = {}
        empty["selector_expected_counts"] = {}
        sample = FoodCueModel.from_mapping(empty).sample((0.0, 0.0, 0.0), 0.0, 0.0, True)
        self.assertEqual(sample.poisson_hz, {})

    def test_standalone_json_config_is_loaded_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "food_physics.json"
            path.write_text(json.dumps(config()), encoding="utf-8")
            loaded = FoodCueModel.from_json(path)
        self.assertEqual(loaded.contact_surface_id, "food_surface")
        self.assertEqual(loaded.selector_expected_counts["ORN_DM1/R"], 39)


if __name__ == "__main__":
    unittest.main()
