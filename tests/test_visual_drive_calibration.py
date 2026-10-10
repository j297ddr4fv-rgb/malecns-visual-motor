#!/usr/bin/env python3
"""Focused CPU tests for the explicit FlyVis/MaleCNS calibration boundary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from visual_drive_calibration import (  # noqa: E402
    CURRENT_UNITS,
    CalibrationConfigError,
    CalibrationDisabledError,
    CalibrationDriveError,
    VisualDriveCalibration,
    calibrate_flyvis_drive,
    create_calibrator,
)


def _config(**overrides):
    base = {
        "schema_version": "visual-drive-calibration-1",
        "name": "test-calibration",
        "enabled": True,
        "input": {
            "field": "flyvis_t4t5",
            "units": CURRENT_UNITS,
            "source": "fitted",
            "expected_target_count": 3,
            "max_abs": 0.01,
        },
        "transform": {
            "gain": 1000.0,
            "max_gain": 10000.0,
            "max_abs_output": 0.5,
            "saturation_policy": "clip",
        },
        "tonic_state_inputs": {"enabled": False, "entries": []},
        "provenance": {"test": True},
    }
    for key, value in overrides.items():
        base[key] = value
    return base


class VisualDriveCalibrationTest(unittest.TestCase):
    def test_default_config_is_explicit_and_manifest_is_json_safe(self):
        calibration = VisualDriveCalibration.from_json(ROOT / "config" / "visual_drive_calibration.json")
        manifest = calibration.manifest()
        self.assertEqual(manifest["schema_version"], "visual-drive-calibration-1")
        self.assertEqual(manifest["input"]["units"], CURRENT_UNITS)
        self.assertEqual(manifest["transform"]["gain"], 1.0)
        self.assertEqual(manifest["transform"]["max_gain"], 2.0)
        self.assertEqual(manifest["input"]["max_abs"], 1.0)
        self.assertEqual(manifest["transform"]["output_max_abs"], 1.0)
        self.assertEqual(
            manifest["provenance"]["evidence_artifact"],
            "artifacts/gpu-flight-smoke-50f-rawvis-20260921",
        )
        self.assertTrue(manifest["provenance"]["engineering_calibration"])
        self.assertFalse(manifest["provenance"]["biological_claim"])
        self.assertTrue(manifest["transform"]["preserves_sign"])
        self.assertFalse(manifest["tonic_state_inputs"]["enabled"])
        self.assertFalse(manifest["tonic_state_inputs"]["spikes_fabricated"])
        json.dumps(manifest, allow_nan=False)

    def test_default_config_accepts_the_sustained_50_frame_raw_range(self):
        """The 50-frame evidence must not be rejected as a one-frame outlier."""

        calibration = VisualDriveCalibration.from_json(ROOT / "config" / "visual_drive_calibration.json")
        target_indices = np.arange(13582, dtype=np.int64)
        extrema = np.linspace(-0.726, 0.860, 50)
        saturation_counts = []
        for frame_index, extreme in enumerate(extrema):
            values = np.zeros(target_indices.size, dtype=np.float64)
            values[0] = float(extreme)
            values[1] = -float(extreme) * 0.25
            packet = calibration.calibrate(
                {"target_indices": target_indices, "values": values, "units": CURRENT_UNITS},
                frame_index=frame_index,
            )
            np.testing.assert_allclose(packet.values[:2], values[:2], rtol=0.0, atol=1e-6)
            self.assertEqual(packet.gain, 1.0)
            self.assertEqual(packet.saturated_count, 0)
            self.assertTrue(np.all(np.abs(packet.values) <= 1.0))
            saturation_counts.append(packet.saturated_count)
        self.assertEqual(sum(saturation_counts), 0)

    def test_default_sanity_bound_fails_closed_and_output_clip_remains_explicit(self):
        calibration = VisualDriveCalibration.from_json(ROOT / "config" / "visual_drive_calibration.json")
        target_indices = np.arange(13582, dtype=np.int64)
        with self.assertRaisesRegex(CalibrationDriveError, "input_max_abs"):
            calibration.calibrate({
                "target_indices": target_indices,
                "values": np.r_[1.001, np.zeros(target_indices.size - 1)],
                "units": CURRENT_UNITS,
            })

        # Keep the output-bound test independent from the input sanity bound:
        # a future config may accept a wider raw range, but must still clip or
        # fail according to its explicit saturation policy.
        bounded = VisualDriveCalibration(_config())
        packet = bounded.calibrate({
            "target_indices": [0, 1, 2],
            "values": [0.0001, -0.0008, 0.0002],
            "units": CURRENT_UNITS,
        })
        np.testing.assert_allclose(packet.values, [0.1, -0.5, 0.2])
        self.assertEqual(packet.saturated_count, 1)
        self.assertTrue(np.all(np.abs(packet.values) <= bounded.config.output_max_abs))

    def test_signed_gain_is_bounded_and_deterministic(self):
        calibration = VisualDriveCalibration(_config())
        drive = {
            "target_indices": np.array([10, 11, 12], dtype=np.int64),
            "values": np.array([-0.0002, 0.0001, 0.001], dtype=np.float32),
            "units": CURRENT_UNITS,
        }
        first = calibration.calibrate(drive, frame_index=7)
        second = calibration.calibrate(drive, frame_index=7)
        self.assertEqual(first["units"], CURRENT_UNITS)
        np.testing.assert_array_equal(first["target_indices"], [10, 11, 12])
        np.testing.assert_allclose(first["values"], first.values)
        np.testing.assert_allclose(first.values, [-0.2, 0.1, 0.5])
        np.testing.assert_array_equal(first.target_indices, [10, 11, 12])
        np.testing.assert_array_equal(first.values, second.values)
        target_drive = first.target_drive
        t4t5_drive = first.t4t5_drive
        self.assertFalse(np.shares_memory(target_drive, first.drive_values))
        self.assertFalse(np.shares_memory(t4t5_drive, first.drive_values))
        self.assertFalse(np.shares_memory(target_drive, t4t5_drive))
        target_drive[0] = 9.0
        self.assertAlmostEqual(first.values[0], -0.2)
        self.assertEqual(first.saturated_count, 1)
        self.assertEqual(first.provenance["input_source"], "fitted")
        self.assertEqual(first.provenance["transform_source"], "handbuilt")

        saturated = calibration.calibrate({**drive, "values": [0.01, -0.01, 0.0001]})
        np.testing.assert_allclose(saturated.values, [0.5, -0.5, 0.1])
        self.assertEqual(saturated.saturated_count, 2)

    def test_mapping_units_and_target_order_are_required(self):
        calibration = VisualDriveCalibration(_config())
        with self.assertRaisesRegex(CalibrationDriveError, "target_indices are required"):
            calibration.calibrate([0.1, 0.2, 0.3], units=CURRENT_UNITS)
        with self.assertRaisesRegex(CalibrationDriveError, "units"):
            calibration.calibrate(
                {"target_indices": [0, 1, 2], "values": [0.1, 0.2, 0.3], "units": "poisson_hz"}
            )
        with self.assertRaisesRegex(CalibrationDriveError, "target count"):
            calibration.calibrate(
                {"target_indices": [0, 1], "values": [0.1, 0.2], "units": CURRENT_UNITS}
            )
        with self.assertRaisesRegex(CalibrationDriveError, "exceeds configured input_max_abs"):
            calibration.calibrate(
                {"target_indices": [0, 1, 2], "values": [0.1, 0.2, 0.3], "units": CURRENT_UNITS}
            )

    def test_output_error_policy_fails_closed_on_saturation(self):
        config = _config()
        config["transform"] = {
            **config["transform"],
            "saturation_policy": "error",
            "max_abs_output": 0.2,
        }
        calibration = VisualDriveCalibration(config)
        with self.assertRaisesRegex(CalibrationDriveError, "output bound"):
            calibration.calibrate({
                "target_indices": [0, 1, 2],
                "values": [0.0001, -0.0001, 0.0003],
                "units": CURRENT_UNITS,
            })

    def test_channels_are_engine_compatible_and_never_contain_spikes(self):
        calibration = VisualDriveCalibration(_config())
        channels = calibration.apply_to_channels(
            {"optic_flow_x_rad_s": 1.0},
            {"target_indices": [100, 101, 102], "values": [0.001, -0.001, 0.0], "units": CURRENT_UNITS},
            frame_index=2,
        )
        mapped = channels["flyvis_t4t5"]
        self.assertEqual(mapped["units"], CURRENT_UNITS)
        np.testing.assert_allclose(mapped["values"], [0.5, -0.5, 0.0])
        np.testing.assert_array_equal(mapped["target_indices"], [100, 101, 102])
        self.assertNotIn("spikes", channels)
        self.assertFalse(channels["visual_drive_calibration"]["is_spike_drive"])
        with self.assertRaisesRegex(CalibrationDriveError, "already contains"):
            calibration.apply_to_channels(
                {"flyvis_t4t5": {"preexisting": True}},
                {"target_indices": [100, 101, 102], "values": [0.0, 0.0, 0.0], "units": CURRENT_UNITS},
            )

    def test_tonic_population_input_is_opt_in_and_external_only(self):
        config = _config()
        config["tonic_state_inputs"] = {
            "enabled": True,
            "entries": [{
                "name": "descending-state",
                "enabled": True,
                "target": {"population": "superclass:descending_neuron"},
                "units": CURRENT_UNITS,
                "value": 0.05,
                "max_abs": 0.2,
                "source": "handbuilt",
                "biological_status": "hypothesis",
                "rationale": "test only",
            }],
        }
        calibration = VisualDriveCalibration(config)
        tonic = calibration.resolve_tonic_inputs(
            population_resolver=lambda spec: np.array([2, 4], dtype=np.int64),
        )
        self.assertEqual(len(tonic), 1)
        np.testing.assert_allclose(tonic[0].values, [0.05, 0.05])
        self.assertEqual(tonic[0].source, "handbuilt")
        self.assertEqual(tonic[0].as_mapping()["is_spike_drive"], False)
        channels = calibration.apply_to_channels(
            {},
            {"target_indices": [10, 11, 12], "values": [0.0, 0.0, 0.0], "units": CURRENT_UNITS},
            population_resolver=lambda spec: np.array([2, 4], dtype=np.int64),
        )
        self.assertAlmostEqual(channels["current_mV_per_ms"]["superclass:descending_neuron"], 0.05)
        self.assertFalse(
            channels["visual_drive_calibration"]["tonic_state_inputs"][0]["is_spike_drive"]
        )
        self.assertNotIn("spikes", channels)

    def test_state_context_modulates_bilateral_tonic_rates_deterministically(self):
        config = _config()
        config["tonic_state_inputs"] = {
            "enabled": True,
            "entries": [{
                "name": "left", "enabled": True,
                "target": {"population": "prefix:DNg02/L"},
                "units": "poisson_hz", "value": 35.0, "max_abs": 80.0,
                "state_modulation": {
                    "base": 35.0,
                    "coefficients": {"phase": 10.0, "heading_error_rad": 4.0},
                    "scenario_bias": {"takeoff": 15.0},
                },
            }, {
                "name": "right", "enabled": True,
                "target": {"population": "prefix:DNg02/R"},
                "units": "poisson_hz", "value": 35.0, "max_abs": 80.0,
                "state_modulation": {
                    "base": 35.0,
                    "coefficients": {"phase": 10.0, "heading_error_rad": -4.0},
                    "scenario_bias": {"takeoff": 15.0},
                },
            }],
        }
        calibration = VisualDriveCalibration(config)
        context = {
            "scenario": "takeoff", "phase": 0.5,
            "altitude_error_m": 0.0, "heading_error_rad": 0.25,
        }
        tonic = calibration.resolve_tonic_inputs(
            population_resolver=lambda spec: np.array([0 if spec.endswith("/L") else 1]),
            state_context=context,
        )
        self.assertEqual([item.name for item in tonic], ["left", "right"])
        np.testing.assert_allclose(tonic[0].values, [56.0])
        np.testing.assert_allclose(tonic[1].values, [54.0])
        with self.assertRaisesRegex(CalibrationDriveError, "state_context is required"):
            calibration.resolve_tonic_inputs(
                population_resolver=lambda _spec: np.array([0]),
            )

    def test_context_alias_and_factory_are_runner_friendly(self):
        calibration = create_calibrator(config=_config())
        output = calibration.calibrate(
            {"values": [0.0, 0.0, 0.0], "target_indices": [0, 1, 2], "units": CURRENT_UNITS},
            context={"scenario": "hover", "phase": 0.5},
        )
        self.assertEqual(output["units"], CURRENT_UNITS)
        with self.assertRaises(CalibrationConfigError):
            create_calibrator(config=_config(), config_path=ROOT / "config" / "visual_drive_calibration.json")

    def test_tonic_toggle_is_explicit_copy_not_mutation(self):
        config = _config()
        config["tonic_state_inputs"] = {
            "enabled": False,
            "entries": [{
                "name": "state", "enabled": True,
                "target": {"population": "prefix:DNg02/L"},
                "units": "poisson_hz", "value": 35.0, "max_abs": 80.0,
            }],
        }
        original = VisualDriveCalibration(config)
        enabled = original.with_tonic_state_inputs(True)
        self.assertFalse(original.config.tonic_enabled)
        self.assertTrue(enabled.config.tonic_enabled)
        tonic = enabled.resolve_tonic_inputs(
            population_resolver=lambda _spec: np.array([3], dtype=np.int64),
        )
        self.assertEqual(len(tonic), 1)

    def test_exact_index_tonic_requires_neuron_count_and_overlap_fails(self):
        config = _config()
        config["tonic_state_inputs"] = {
            "enabled": True,
            "entries": [{
                "name": "indexed-state",
                "enabled": True,
                "target": {"indices": [2, 3]},
                "units": CURRENT_UNITS,
                "value": 0.1,
                "max_abs": 0.2,
            }],
        }
        calibration = VisualDriveCalibration(config)
        with self.assertRaisesRegex(CalibrationDriveError, "n_neurons is required"):
            calibration.apply_to_channels(
                {}, {"target_indices": [10, 11, 12], "values": [0, 0, 0], "units": CURRENT_UNITS}
            )
        channels = calibration.apply_to_channels(
            {}, {"target_indices": [10, 11, 12], "values": [0, 0, 0], "units": CURRENT_UNITS},
            n_neurons=5,
        )
        np.testing.assert_allclose(channels["current_mV_per_ms"], [0, 0, 0.1, 0.1, 0])

        overlap = _config()
        overlap["tonic_state_inputs"] = {
            "enabled": True,
            "entries": [{
                "name": "overlap-a", "enabled": True, "target": {"indices": [1, 2]},
                "units": CURRENT_UNITS, "value": 0.1, "max_abs": 0.2,
            }, {
                "name": "overlap-b", "enabled": True, "target": {"indices": [2, 3]},
                "units": CURRENT_UNITS, "value": 0.1, "max_abs": 0.2,
            }],
        }
        with self.assertRaisesRegex(CalibrationDriveError, "overlap"):
            VisualDriveCalibration(overlap).resolve_tonic_inputs(n_neurons=5)

    def test_disabled_calibration_and_bad_config_fail_closed(self):
        with self.assertRaises(CalibrationDisabledError):
            VisualDriveCalibration({**_config(), "enabled": False}).calibrate({
                "target_indices": [0, 1, 2], "values": [0, 0, 0], "units": CURRENT_UNITS,
            })
        bad = _config()
        bad["input"] = {**bad["input"], "units": "arbitrary"}
        with self.assertRaises(CalibrationConfigError):
            VisualDriveCalibration(bad)

    def test_functional_runner_helper_returns_mapping(self):
        mapped = calibrate_flyvis_drive(
            [-0.001, 0.0, 0.001], [20, 21, 22],
            config=_config(), units=CURRENT_UNITS,
        )
        self.assertEqual(mapped["units"], CURRENT_UNITS)
        np.testing.assert_allclose(mapped["values"], [-0.5, 0.0, 0.5])
        self.assertTrue(mapped["calibrated"])
        self.assertEqual(mapped["source"], "fitted")


if __name__ == "__main__":
    unittest.main()
