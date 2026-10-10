#!/usr/bin/env python3
"""Focused tests for the body-to-MaleCNS mechanosensory adapter."""

from __future__ import annotations

import math
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sensory_feedback import BodyState, MechanosensoryFeedback, PopulationResolver  # noqa: E402


class SensoryFeedbackTests(unittest.TestCase):
    def test_channels_cover_required_inputs_and_units(self) -> None:
        sensor = MechanosensoryFeedback()
        frame = sensor.step(
            BodyState(0.25, attitude_rpy_rad=(0.1, -0.2, 0.3),
                      angular_velocity_rad_s=(1.0, -2.0, 3.0), velocity_mps=(0.2, 0.0, 0.0)),
            world_wind=(1.2, 0.4, -0.1),
            ground_contact={"left": 1.0, "right": 0.25},
            joint_state={"left_phase": 0.2, "right_phase": 0.3,
                         "left_amplitude": 0.8, "right_amplitude": 0.5,
                         "frequency_hz": 200.0},
        )
        for name in ("attitude_roll", "attitude_pitch", "attitude_yaw",
                     "angular_velocity_roll", "angular_velocity_pitch", "angular_velocity_yaw",
                     "wind_body_lateral", "ground_contact_fraction", "wing_phase_difference",
                     "wing_left_right_imbalance"):
            self.assertIn(name, frame.channels)
        self.assertEqual(frame.channel_units["attitude_roll"], "rad")
        self.assertEqual(frame.channel_units["angular_velocity_yaw"], "rad/s")
        self.assertEqual(frame.channel_units["wind_body_lateral"], "m/s")
        self.assertEqual(frame.timestamp_s, 0.25)
        self.assertTrue(frame.diagnostics["time_consistent"])

    def test_delay_is_causal_and_preserves_current_timestamp(self) -> None:
        sensor = MechanosensoryFeedback(delay_s=0.1)
        first = sensor.step(BodyState(0.0, angular_velocity_rad_s=(0.0, 0.0, 10.0)))
        self.assertIsNone(first.source_timestamp_s)
        self.assertTrue(all(value == 0.0 for value in first.channels.values()))
        second = sensor.step(BodyState(0.05, angular_velocity_rad_s=(0.0, 0.0, 2.0)))
        self.assertIsNone(second.source_timestamp_s)
        third = sensor.step(BodyState(0.1, angular_velocity_rad_s=(0.0, 0.0, 2.0)))
        self.assertEqual(third.timestamp_s, 0.1)
        self.assertEqual(third.source_timestamp_s, 0.0)
        self.assertEqual(third.raw_channels["angular_velocity_yaw"], 10.0)
        self.assertTrue(third.diagnostics["time_consistent"])

    def test_zero_delay_does_not_retain_unbounded_history(self) -> None:
        sensor = MechanosensoryFeedback()
        for index in range(2000):
            timestamp = index * 0.005
            frame = sensor.step(BodyState(timestamp, angular_velocity_rad_s=(0.0, 0.0, 2.0)))
            self.assertEqual(frame.source_timestamp_s, timestamp)
            self.assertEqual(frame.diagnostics["effective_delay_s"], 0.0)
        self.assertEqual(len(sensor._history), 0)

    def test_monotonic_delayed_history_is_bounded(self) -> None:
        delay_s = 0.1
        dt_s = 0.005
        sensor = MechanosensoryFeedback(delay_s=delay_s)
        for index in range(2000):
            sensor.step(BodyState(index * dt_s))
        # Keep one eligible predecessor plus the not-yet-eligible tail.  The
        # exact count depends on the epsilon at the delay boundary, so allow a
        # small constant margin while rejecting growth with run length.
        self.assertLessEqual(len(sensor._history), int(delay_s / dt_s) + 3)
        self.assertTrue(sensor._history_monotonic)

    def test_monotonic_delay_matches_reference_selection(self) -> None:
        delay_s = 0.1
        dt_s = 0.005
        sensor = MechanosensoryFeedback(delay_s=delay_s)
        reference = []
        for index in range(120):
            timestamp = index * dt_s
            state = BodyState(timestamp, angular_velocity_rad_s=(0.0, 0.0, float(index)))
            _, raw, drives = sensor._compute(state, (0.0, 0.0, 0.0), None, state)
            reference.append((timestamp, raw, drives))
            frame = sensor.step(state)
            eligible = [item for item in reference if item[0] <= timestamp - delay_s + 1e-12]
            if not eligible:
                self.assertIsNone(frame.source_timestamp_s)
            else:
                selected = max(eligible, key=lambda item: item[0])
                self.assertEqual(frame.source_timestamp_s, selected[0])
                self.assertEqual(frame.raw_channels, selected[1])
                self.assertEqual(frame.raw_population_drive, selected[2])

    def test_delayed_duplicate_timestamps_keep_first_sample_on_tie(self) -> None:
        sensor = MechanosensoryFeedback(delay_s=0.1)
        sensor.step(BodyState(0.0, angular_velocity_rad_s=(0.0, 0.0, 1.0)))
        sensor.step(BodyState(0.0, angular_velocity_rad_s=(0.0, 0.0, 2.0)))
        frame = sensor.step(BodyState(0.1, angular_velocity_rad_s=(0.0, 0.0, 3.0)))
        self.assertEqual(frame.source_timestamp_s, 0.0)
        self.assertEqual(frame.raw_channels["angular_velocity_yaw"], 1.0)

    def test_constant_monotonic_timestamp_history_stays_bounded(self) -> None:
        sensor = MechanosensoryFeedback(delay_s=0.1)
        for index in range(2000):
            sensor.step(BodyState(0.0, angular_velocity_rad_s=(0.0, 0.0, float(index))))
        self.assertEqual(len(sensor._history), 1)

    def test_out_of_order_timestamps_keep_full_scan_semantics(self) -> None:
        sensor = MechanosensoryFeedback(delay_s=0.1)
        sensor.step(BodyState(0.0, angular_velocity_rad_s=(0.0, 0.0, 1.0)))
        sensor.step(BodyState(0.2, angular_velocity_rad_s=(0.0, 0.0, 2.0)))
        # At t=0.15 the cutoff is 0.05; the retained t=0.0 sample is the
        # maximum eligible timestamp despite the timestamp moving backwards.
        frame = sensor.step(BodyState(0.15, angular_velocity_rad_s=(0.0, 0.0, 3.0)))
        self.assertFalse(frame.diagnostics["timestamp_monotonic"])
        self.assertEqual(frame.source_timestamp_s, 0.0)
        self.assertEqual(frame.raw_channels["angular_velocity_yaw"], 1.0)
        self.assertFalse(sensor._history_monotonic)

    def test_noise_seed_and_saturation_are_reproducible(self) -> None:
        state = BodyState(0.0, angular_velocity_rad_s=(0.0, 0.0, 20.0))
        a = MechanosensoryFeedback(noise_std=0.4, seed=11, saturation=0.2).step(state)
        b = MechanosensoryFeedback(noise_std=0.4, seed=11, saturation=0.2).step(state)
        self.assertEqual(a.channels, b.channels)
        self.assertTrue(a.diagnostics["noise_applied"])
        self.assertTrue(a.diagnostics["all_signals_in_range"])
        self.assertTrue(all(-0.2 <= value <= 0.2 for value in a.channels.values()))

    def test_malecns_population_map_has_no_bristle_or_body_id_claim(self) -> None:
        metadata = MechanosensoryFeedback().population_metadata
        self.assertIn("jo", metadata)
        self.assertIn("haltere", metadata)
        self.assertIn("wing_tactile", metadata)
        self.assertIn("leg_tactile", metadata)
        self.assertNotIn("wing_bristle", metadata)
        self.assertNotIn("leg_bristle", metadata)
        for item in metadata.values():
            self.assertEqual(item["body_id_mapping"], "not provided")
            self.assertIn(item["confidence"], {"high", "medium", "low", "unknown"})
            self.assertIn(item["mapping_confidence"], {"high", "medium", "low", "unknown"})

    def test_cached_metadata_is_defensively_copied(self) -> None:
        sensor = MechanosensoryFeedback()
        metadata = sensor.population_metadata
        metadata["jo"]["role"] = "caller mutation"
        frame = sensor.step(BodyState(0.0))
        self.assertNotEqual(frame.population_metadata["jo"]["role"], "caller mutation")
        noise_map = sensor._noise_mapping()
        noise_map["*"] = 999.0
        self.assertEqual(sensor._noise_mapping()["*"], 0.0)

    def test_local_csr_counts_match_annotation_intersections(self) -> None:
        resolver = PopulationResolver()
        if resolver.status["haltere"] == "csr_unavailable":
            self.skipTest("local MaleCNS CSR is not installed")
        self.assertEqual(resolver.counts["jo_wind_gravity"], 475)
        self.assertEqual(resolver.counts["haltere"], 201)
        self.assertEqual(resolver.counts["wing_tactile"], 352)
        self.assertEqual(resolver.counts["wing"], 371)
        self.assertEqual(resolver.counts["leg_tactile"], 1877)
        self.assertEqual(resolver.counts["leg_proprioceptive"], 190)
        self.assertEqual(resolver.counts["leg"], 2067)

    def test_shared_contract_adapter_uses_handbuilt_source(self) -> None:
        frame = MechanosensoryFeedback().step(BodyState(0.005))
        contract = frame.to_contract(body_state=BodyState(0.005))
        self.assertEqual(contract.timestamp.t_s, 0.005)
        self.assertEqual(contract.source.value, "handbuilt")
        self.assertIn("population:jo", contract.channels)
        self.assertEqual(contract.channels["population_drive_units"], "normalized")

    def test_population_drives_have_explicit_normalized_units_and_bounded_gains(self) -> None:
        sensor = MechanosensoryFeedback()
        frame = sensor.step(BodyState(0.0), world_wind=(100.0, -100.0, 100.0),
                            ground_contact={"left": 1.0, "right": 0.0})
        self.assertEqual(frame.population_drive_units, "normalized")
        self.assertEqual(frame.population_drive_range, (-1.0, 1.0))
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in frame.population_drive.values()))
        self.assertEqual(frame.diagnostics["population_drive_units"], "normalized")
        mapping = sensor.resolver.drive_mapping
        self.assertEqual(mapping["output_units"], "current_mV_per_ms")
        self.assertLessEqual(float(mapping["max_gain_mV_per_ms"]), 0.5)
        for entry in mapping["entries"].values():
            self.assertLessEqual(float(entry["gain_mV_per_ms"]), 0.5)

    def test_population_aliases_share_one_noise_sample(self) -> None:
        frame = MechanosensoryFeedback(noise_std=0.1, seed=19).step(
            BodyState(0.0), world_wind=(0.5, 0.0, 0.0)
        )
        self.assertEqual(frame.population_drive["jo"], frame.population_drive["jo_wind_gravity"])
        self.assertEqual(frame.population_drive["jo"], frame.population_drive["wind_gravity"])

    def test_prepare_commit_is_step_equivalent_for_delay_noise_and_timestamp_order(self) -> None:
        samples = (
            (BodyState(0.0, angular_velocity_rad_s=(0.0, 0.0, 1.0)),
             {"world_wind": (0.2, 0.1, 0.0), "ground_contact": {"left": 1.0}},
             {"left_phase": 0.1, "right_phase": 0.3, "left_amplitude": 0.7,
              "right_amplitude": 0.5, "frequency_hz": 180.0}),
            (BodyState(0.2, angular_velocity_rad_s=(0.0, 0.0, 2.0)),
             {"world_wind": (-0.3, 0.0, 0.4), "ground_contact": {"right": 0.5}},
             {"left_phase": 0.4, "right_phase": 0.2, "left_amplitude": 0.9,
              "right_amplitude": 0.6, "frequency_hz": 200.0}),
            # Deliberately repeated and out-of-order timestamps exercise both
            # delayed full-scan behavior and the first-tie rule.
            (BodyState(0.15, angular_velocity_rad_s=(0.0, 0.0, 3.0)),
             {"world_wind": (0.0, 0.4, 0.0), "ground_contact": {"left": 0.2, "right": 0.8}},
             {"left_phase": 0.8, "right_phase": 0.1, "left_amplitude": 0.4,
              "right_amplitude": 0.4, "frequency_hz": 120.0}),
            (BodyState(0.15, angular_velocity_rad_s=(0.0, 0.0, 4.0)),
             {"world_wind": (0.0, 0.4, 0.0), "ground_contact": {"left": 0.2, "right": 0.8}},
             {"left_phase": 0.2, "right_phase": 0.7, "left_amplitude": 0.3,
              "right_amplitude": 0.8, "frequency_hz": 120.0}),
        )
        for delay_s in (0.0, 0.1):
            direct = MechanosensoryFeedback(delay_s=delay_s, noise_std=0.08, seed=73)
            staged = MechanosensoryFeedback(delay_s=delay_s, noise_std=0.08, seed=73)
            for state, kwargs, joints in samples:
                expected = direct.step(state, joint_state=joints, **kwargs)
                actual = staged.commit_prepared(staged.prepare_step(state, joint_state=joints, **kwargs))
                self.assertEqual(actual, expected)

    def test_prepare_does_not_advance_causal_state_or_rng(self) -> None:
        sensor = MechanosensoryFeedback(delay_s=0.1, noise_std=0.2, seed=9)
        state = BodyState(0.25, angular_velocity_rad_s=(0.0, 0.0, 4.0))
        history_before = list(sensor._history)
        monotonic_before = sensor._history_monotonic
        timestamp_before = sensor._last_timestamp
        rng_before = sensor._rng.getstate()
        prepared = sensor.prepare_step(state)
        self.assertEqual(list(sensor._history), history_before)
        self.assertEqual(sensor._history_monotonic, monotonic_before)
        self.assertEqual(sensor._last_timestamp, timestamp_before)
        self.assertEqual(sensor._rng.getstate(), rng_before)
        with self.assertRaises(TypeError):
            prepared.channels["attitude_roll"] = 0.0  # type: ignore[index]

    def test_prepared_inputs_are_defensive_and_commit_owner_order_is_strict(self) -> None:
        body = {
            "timestamp_s": 0.0,
            "angular_velocity_rad_s": [0.0, 0.0, 3.0],
        }
        wind = [0.5, 0.0, 0.0]
        sensor = MechanosensoryFeedback(noise_std=0.1, seed=5)
        first = sensor.prepare_step(body, world_wind=wind)
        second = sensor.prepare_step(BodyState(0.1))
        body["timestamp_s"] = 99.0
        body["angular_velocity_rad_s"][2] = 99.0
        wind[0] = 99.0
        self.assertEqual(first.timestamp_s, 0.0)
        self.assertEqual(first.raw_channels["angular_velocity_yaw"], 3.0)

        other = MechanosensoryFeedback()
        with self.assertRaises(ValueError):
            other.commit_prepared(first)
        self.assertIsNone(other._last_timestamp)
        with self.assertRaises(ValueError):
            sensor.commit_prepared(second)
        self.assertIsNone(sensor._last_timestamp)

        sensor.commit_prepared(first)
        with self.assertRaises(ValueError):
            sensor.commit_prepared(first)
        sensor.commit_prepared(second)

    def test_config_does_not_claim_body_ids_or_bristle_mappings(self) -> None:
        config = json.loads((ROOT / "config" / "sensory_map.json").read_text(encoding="utf-8"))
        drive_mapping = config["drive_mapping"]
        self.assertEqual(drive_mapping["body_id_mapping"], "not provided")
        for entry in drive_mapping["entries"].values():
            self.assertNotIn("body:", str(entry.get("target_population", "")).lower())
        metadata = MechanosensoryFeedback().population_metadata
        for name in ("jo", "haltere", "wing", "leg"):
            self.assertEqual(metadata[name]["body_id_mapping"], "not provided")


if __name__ == "__main__":
    unittest.main()
