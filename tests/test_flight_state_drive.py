#!/usr/bin/env python3
"""Focused tests for the bounded MaleCNS flight state-drive adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from flight_state_drive import (  # noqa: E402
    CURRENT_UNITS,
    POISSON_UNITS,
    FlightState,
    FlightStateDrive,
    StateDriveConfig,
    StateDriveInputError,
)


class FlightStateDriveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = FlightStateDrive.from_json(ROOT / "config" / "flight_state_drive.json")

    def setUp(self) -> None:
        self.adapter.reset()

    def test_hover_is_symmetric_at_the_35_hz_anchor(self) -> None:
        packet = self.adapter.step({"scenario": "hover", "phase": 0.5})
        rates = packet.population_rates_hz
        self.assertAlmostEqual(rates["prefix:DNg02/L"], 35.0)
        self.assertAlmostEqual(rates["prefix:DNg02/R"], 35.0)
        self.assertEqual(rates["prefix:DNg02/L"], rates["prefix:DNg02/R"])
        self.assertEqual(rates["DNp11"], 0.0)
        self.assertEqual(rates["DNp07"], 0.0)
        self.assertEqual(packet.source, "handbuilt")
        self.assertEqual(packet.provenance["biological_status"], "hypothesis")
        self.assertEqual(packet.provenance["calibration"], "calibration")

    def test_heading_error_has_explicit_right_minus_left_sign(self) -> None:
        right = self.adapter.step(
            scenario="hover", phase=0.5, heading_error_rad=0.25,
        ).population_rates_hz
        left = self.adapter.step(
            scenario="hover", phase=0.5, heading_error_rad=-0.25,
        ).population_rates_hz
        # DNg02 is contralateral in the checked neuromuscular map: the
        # left-side population drives the positive right-wing steering sign.
        self.assertGreater(right["prefix:DNg02/L"], right["prefix:DNg02/R"])
        self.assertLess(left["prefix:DNg02/L"], left["prefix:DNg02/R"])
        self.assertAlmostEqual(
            right["prefix:DNg02/R"] - right["prefix:DNg02/L"],
            -(left["prefix:DNg02/R"] - left["prefix:DNg02/L"]),
        )
        # The tuned pair is a symmetric external-rate hypothesis: the
        # differential is 2 * 36 Hz/rad * heading error, before clipping.
        self.assertAlmostEqual(
            right["prefix:DNg02/L"] - right["prefix:DNg02/R"],
            18.0,
        )

    def test_one_second_turn_endpoint_stays_inside_external_rate_bounds(self) -> None:
        """A 0.9 rad goal remains a bounded rate packet, not a motor claim."""

        positive = self.adapter.step(
            scenario="flight", phase=1.0, heading_error_rad=0.9,
        )
        negative = self.adapter.step(
            scenario="flight", phase=1.0, heading_error_rad=-0.9,
        )
        positive_rates = positive.population_rates_hz
        negative_rates = negative.population_rates_hz
        self.assertAlmostEqual(positive_rates["prefix:DNg02/L"], 67.4)
        self.assertAlmostEqual(positive_rates["prefix:DNg02/R"], 2.6)
        self.assertAlmostEqual(negative_rates["prefix:DNg02/L"], 2.6)
        self.assertAlmostEqual(negative_rates["prefix:DNg02/R"], 67.4)
        for packet in (positive, negative):
            self.assertFalse(packet.saturated)
            for item in packet.population_inputs:
                self.assertGreaterEqual(item.value, item.bounds[0])
                self.assertLessEqual(item.value, item.bounds[1])
            self.assertNotIn("spikes", packet.to_dict())
            self.assertNotIn("motor_commands", packet.to_dict())
            self.assertNotIn("position_m", packet.to_dict())
        self.assertEqual(
            positive.provenance["turn_task"]["response_status"],
            "not_proven",
        )

    def test_replay_endpoint_only_tunes_external_differential(self) -> None:
        """The 1000-frame endpoint is reproducible without asserting body response."""

        # Frame 999 of gpu-flight-smoke-1000f-stabilized-20260921 records
        # heading_error_rad=0.0818214671 at the end of its 0.2 s turn slice.
        packet = self.adapter.step(
            scenario="flight", phase=0.2,
            heading_error_rad=0.08182146713728233,
        )
        rates = packet.population_rates_hz
        self.assertAlmostEqual(rates["prefix:DNg02/L"], 37.94557281694216)
        self.assertAlmostEqual(rates["prefix:DNg02/R"], 32.05442718305784)
        self.assertAlmostEqual(
            rates["prefix:DNg02/L"] - rates["prefix:DNg02/R"],
            5.891145633884328,
        )
        self.assertEqual(packet.provenance["source"], "handbuilt")
        self.assertEqual(packet.provenance["biological_status"], "hypothesis")

    def test_takeoff_and_landing_phase_transitions_are_bounded(self) -> None:
        hover = self.adapter.step(scenario="hover", phase=0.5).population_rates_hz
        takeoff_start = self.adapter.step(scenario="takeoff", phase=0.0).population_rates_hz
        takeoff_end = self.adapter.step(scenario="takeoff", phase=1.0).population_rates_hz
        landing_start = self.adapter.step(scenario="landing", phase=0.0).population_rates_hz
        landing_end = self.adapter.step(scenario="landing", phase=1.0).population_rates_hz

        self.assertGreater(takeoff_start["prefix:DNg02/L"], hover["prefix:DNg02/L"])
        self.assertAlmostEqual(takeoff_end["prefix:DNg02/L"], hover["prefix:DNg02/L"])
        self.assertAlmostEqual(landing_start["prefix:DNg02/L"], hover["prefix:DNg02/L"])
        self.assertLess(landing_end["prefix:DNg02/L"], hover["prefix:DNg02/L"])
        self.assertGreater(takeoff_start["DNp11"], 0.0)
        self.assertGreater(takeoff_start["DNp02"], 0.0)
        self.assertGreater(takeoff_start["DNp04"], 0.0)
        self.assertEqual(takeoff_end["DNp11"], 0.0)
        self.assertGreater(landing_end["DNp07"], 0.0)
        self.assertGreater(landing_end["DNp10"], 0.0)
        self.assertEqual(landing_start["DNp07"], 0.0)

    def test_large_errors_clip_every_population_to_declared_bounds(self) -> None:
        packet = self.adapter.step({
            "scenario": "hover",
            "phase": 0.5,
            "body_error": {
                "altitude_error_m": 1.0e12,
                "heading_error_rad": 1.0e12,
                "vertical_speed_mps": -1.0e12,
            },
        })
        for item in packet.population_inputs:
            self.assertGreaterEqual(item.value, item.bounds[0])
            self.assertLessEqual(item.value, item.bounds[1])
        self.assertTrue(packet.saturated)
        self.assertEqual(packet.population_rates_hz["prefix:DNg02/L"], 80.0)
        self.assertEqual(packet.population_rates_hz["prefix:DNg02/R"], 0.0)

    def test_engine_channels_are_only_population_rate_or_current_maps(self) -> None:
        packet = self.adapter.step(scenario="hover", phase=0.2)
        channels = packet.as_engine_channels()
        self.assertEqual(set(channels), {POISSON_UNITS, CURRENT_UNITS})
        self.assertEqual(dict(packet), channels)
        self.assertNotIn("spikes", channels)
        self.assertNotIn("motor_commands", packet.to_dict())
        self.assertNotIn("position_m", packet.to_dict()["state"])
        self.assertEqual(channels[POISSON_UNITS]["prefix:DNg02/L"], 35.0)
        self.assertEqual(channels[CURRENT_UNITS], {})

    def test_runner_nested_error_mapping_and_reset_are_deterministic(self) -> None:
        state = {
            "scenario_name": "cruise",
            "scenario_phase": 0.4,
            "goal": {"heading_error": 0.2},
            "body_error": {"altitude_error": 0.1},
        }
        first = self.adapter.step(state)
        self.assertEqual(first.mode, "flight")
        self.assertEqual(first.sequence, 1)
        self.adapter.step(scenario="hover")
        self.adapter.reset()
        second = self.adapter.step(state)
        self.assertEqual(second.sequence, 1)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_invalid_phase_and_nonfinite_error_fail_closed(self) -> None:
        with self.assertRaises(StateDriveInputError):
            self.adapter.step(scenario="hover", phase=1.1)
        with self.assertRaises(StateDriveInputError):
            self.adapter.step(scenario="hover", phase=0.5, heading_error_rad=float("nan"))

    def test_manifest_is_json_safe_and_declares_external_only_boundary(self) -> None:
        manifest = self.adapter.manifest()
        json.dumps(manifest, allow_nan=False)
        self.assertEqual(manifest["schema_version"], "flight-state-drive-1")
        self.assertTrue(manifest["deterministic"])
        self.assertFalse(manifest["output_contract"]["spikes_emitted"])
        self.assertFalse(manifest["output_contract"]["body_positions_emitted"])
        self.assertFalse(manifest["output_contract"]["motor_commands_emitted"])
        self.assertTrue(manifest["output_contract"]["population_selectors_are_disjoint"])
        self.assertEqual(manifest["provenance"]["source"], "handbuilt")


class CurrentChannelConfigTest(unittest.TestCase):
    def test_current_unit_is_supported_without_becoming_a_motor_command(self) -> None:
        config = StateDriveConfig.from_mapping({
            "schema_version": "flight-state-drive-1",
            "name": "current-test",
            "population_inputs": [{
                "name": "current-population",
                "population": "prefix:test/L",
                "units": "current_mV_per_ms",
                "bounds": [-0.5, 0.5],
                "profiles": {"hover": 0.2},
            }],
            "provenance": {"source": "handbuilt", "biological_status": "hypothesis"},
        })
        packet = FlightStateDrive(config).step(scenario="hover", phase=0.5)
        self.assertEqual(packet.population_currents_mV_per_ms["prefix:test/L"], 0.2)
        self.assertEqual(packet.population_rates_hz, {})
        self.assertNotIn("motor_commands", packet.to_dict())


if __name__ == "__main__":
    unittest.main()
