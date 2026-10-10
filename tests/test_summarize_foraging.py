import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import summarize_foraging as sf


class ForagingSummaryChecks(unittest.TestCase):
    def setUp(self):
        self.config = {
            "contact_surface_id": "food_surface",
            "field": {"center_m": [0.0, 0.0, 0.0]},
            "odor_selector_weights_hz_per_unit": {
                "left": {"odor_left": 2.0},
                "right": {"odor_right": 3.0},
            },
            "sugar_selector_rates_hz": {"taste": 120.0},
            "selector_expected_counts": {"odor_left": 2, "odor_right": 3, "taste": 4},
        }
        self.component = {
            "contact_surface_id": "food_surface",
            "selector_counts": {"odor_left": 2, "odor_right": 3, "taste": 4},
        }
        self.row = {
            "frame_index": 0,
            "frame_time_s": 0.0,
            "time_s": 0.005,
            "dt_s": 0.005,
            "food_cue": {
                "time_s": 0.0,
                "concentration": 0.5,
                "left_sample": 0.5,
                "right_sample": 0.5,
                "mouth_distance_m": 0.0,
                "physical_contact": True,
                "edible_region_contact": True,
                "contact": True,
                "contact_surface_id": "food_surface",
                "ablated_senses": [],
                "raw_odor_poisson_hz": {"odor_left": 1.0, "odor_right": 1.5},
                "raw_taste_poisson_hz": {"taste": 120.0},
                "cns_poisson_hz": {"odor_left": 1.0, "odor_right": 1.5, "taste": 120.0},
            },
        }

    def test_stream_alignment_uses_input_and_state_times(self):
        frame = sf._check_streams([self.row], [self.row], sf.Path("run"))[0]
        self.assertEqual(frame["frame_time_s"], 0.0)
        self.assertEqual(frame["state_time_s"], 0.005)

        shifted = dict(self.row, frame_time_s=0.001)
        with self.assertRaises(sf.ForagingSummaryError):
            sf._check_streams([self.row], [shifted], sf.Path("run"))

    def test_food_ablation_preserves_raw_signal_and_checks_cns_seam(self):
        frames = [{"frame_index": 0, "frame_time_s": 0.0}]
        sf._audit_food_rows("intact", [self.row], frames, self.config, self.component, ())

        ablated = {**self.row, "food_cue": dict(self.row["food_cue"])}
        ablated["food_cue"]["ablated_senses"] = ["odor"]
        ablated["food_cue"]["cns_poisson_hz"] = {
            "odor_left": 0.0, "odor_right": 0.0, "taste": 120.0,
        }
        sf._audit_food_rows("odor_ablated", [ablated], frames, self.config, self.component, ("odor",))
        self.assertEqual(ablated["food_cue"]["raw_odor_poisson_hz"]["odor_left"], 1.0)

    def test_sugar_input_is_rejected_without_the_gate(self):
        bad = {**self.row, "food_cue": dict(self.row["food_cue"])}
        bad["food_cue"].update(physical_contact=False, contact=False)
        with self.assertRaises(sf.ForagingSummaryError):
            sf._audit_food_rows(
                "intact", [bad], [{"frame_index": 0, "frame_time_s": 0.0}],
                self.config, self.component, (),
            )

    def test_configured_population_spike_totals(self):
        rows = [
            {"frame_index": 0, "frame_time_s": 0.0, "time_s": 0.005, "neural": {"named_outputs": {
                "population_spike_counts": {"taste": 3}, "population_rates_hz": {"taste": 600.0},
            }}},
            {"frame_index": 1, "frame_time_s": 0.005, "time_s": 0.01, "neural": {"named_outputs": {
                "population_spike_counts": {"taste": 5}, "population_rates_hz": {"taste": 1000.0},
            }}},
        ]
        summary = sf._summarize_food_neuron_outputs(rows, ["taste"])
        self.assertEqual(summary["taste"]["total_spikes"], 8)
        self.assertEqual(summary["taste"]["first_positive_spike_count"]["frame_index"], 0)


if __name__ == "__main__":
    unittest.main()
