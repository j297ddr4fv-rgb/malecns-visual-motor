"""Tests for the physical-touch to grooming neural-readout bridge."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grooming_body import GroomingBody, REGION_NAMES  # noqa: E402
from grooming_neural_bridge import (  # noqa: E402
    GroomingBridgeConfig,
    GroomingNeuralBridge,
    load_config,
)


class _AdapterLikeReadout:
    def __init__(self, value: float, source: str = "connectome") -> None:
        self.channels = {"groom_antenna": value}
        self.source = source

    def channel(self, name: str, default: float = 0.0) -> float:
        return self.channels.get(name, default)


def _bridge() -> GroomingNeuralBridge:
    return GroomingNeuralBridge(GroomingBridgeConfig(
        selectors_by_region={"antenna": "JO_candidate_antenna"},
        rate_hz_by_region={"antenna": 40.0},
        sweep_extent_mm=0.12,
        sweep_frequency_hz=1.6,
    ))


def _loads(**provided: float) -> dict[str, float]:
    return {region: provided.get(region, 0.0) for region in REGION_NAMES}


class GroomingNeuralBridgeTest(unittest.TestCase):
    def test_touch_sample_generates_configured_rates_without_particle_access(self) -> None:
        bridge = _bridge()
        loads = _loads(antenna=0.5, head=0.25)
        sample = bridge.sample_input(loads, 0.005)

        self.assertEqual(sample.raw_dust_loads, loads)
        self.assertEqual(sample.observed_dust_loads, loads)
        self.assertEqual(sample.poisson_hz, {"JO_candidate_antenna": 20.0})
        self.assertEqual(sample.sensory_channels["poisson_hz"], sample.poisson_hz)
        self.assertEqual(sample.sensory_source, "GroomingBody.tactile_loads")
        self.assertIn("physical_dust_touch", sample.activity_source)

    def test_sensor_mask_zeros_input_activity_but_preserves_raw_load_audit(self) -> None:
        sample = _bridge().sample_input(_loads(antenna=0.75), 0.005, sensor_enabled=False)

        self.assertEqual(sample.raw_dust_loads["antenna"], 0.75)
        self.assertEqual(sample.observed_dust_loads["antenna"], 0.0)
        self.assertEqual(sample.poisson_hz, {"JO_candidate_antenna": 0.0})
        self.assertFalse(sample.sensor_enabled)

    def test_zero_readout_is_stationary_even_when_physical_touch_is_present(self) -> None:
        bridge = _bridge()
        body = GroomingBody(dust_seed=19, dust_by_region={"antenna": 2})
        before = body.step(0.005)
        sample = bridge.sample_input(body.tactile_loads(), 0.005)

        action = bridge.apply_readout(body, 0.005, {"groom_antenna": 0.0})
        audit = bridge.audit(sample, action).to_dict()

        self.assertEqual(action.targets_mm, {})
        self.assertEqual(action.joint_angles, before.joint_angles)
        self.assertEqual(action.cleaned, ())
        self.assertEqual(audit["raw_dust_loads"]["antenna"], 1.0)
        self.assertEqual(audit["groom_antenna_readout"], 0.0)
        self.assertEqual(audit["cleaned"], [])

    def test_engineered_local_sweep_is_gated_and_amplitude_scaled_by_dn_readout(self) -> None:
        bridge = _bridge()
        body = GroomingBody(dust_seed=4, dust_by_region={"antenna": 1})
        targets = []
        action = None
        for _ in range(20):
            action = bridge.apply_readout(body, 0.005, _AdapterLikeReadout(0.6))
            targets.append(action.targets_mm["fore_left"])

        self.assertIsNotNone(action)
        self.assertEqual(action.groom_antenna_readout, 0.6)
        self.assertEqual(action.readout_channel, "groom_antenna")
        self.assertEqual(action.readout_populations, ("DNg62", "DNge078"))
        self.assertEqual(action.action_classification,
                         "neural_readout_gated_engineered_motor_primitive")
        self.assertEqual(action.trajectory_source, "engineered_cpg_local_antenna_sweep")
        self.assertGreater(max(point[2] for point in targets), min(point[2] for point in targets))
        self.assertTrue(action.readout_present)
        self.assertEqual(action.readout_source, "connectome")

    def test_active_readout_sweep_cleans_only_by_physical_body_contact(self) -> None:
        bridge = _bridge()
        body = GroomingBody(dust_seed=3, dust_by_region={"antenna": 1})
        cleaned = ()
        for _ in range(200):
            action = bridge.apply_readout(body, 0.005, {"groom_antenna": 0.8})
            cleaned += action.cleaned
            if cleaned:
                break

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0][0], "antenna")
        self.assertEqual(body.remaining()["antenna"], 0)

    def test_sensor_off_sample_does_not_hard_gate_the_motor_stage(self) -> None:
        """Sensory masking must be evaluated through the neural readout path."""

        bridge = _bridge()
        body = GroomingBody(dust_seed=8, dust_by_region={"antenna": 1})
        masked_sample = bridge.sample_input(body.tactile_loads(), 0.005, sensor_enabled=False)
        action = bridge.apply_readout(body, 0.005, {"groom_antenna": 0.4})
        audit = bridge.audit(masked_sample, action).to_dict()

        self.assertEqual(masked_sample.poisson_hz["JO_candidate_antenna"], 0.0)
        self.assertTrue(action.targets_mm)
        self.assertFalse(audit["sensor_enabled"])
        self.assertEqual(audit["groom_antenna_readout"], 0.4)

    def test_locked_foreleg_and_missing_readout_produce_no_targets_or_cleaning(self) -> None:
        bridge = _bridge()
        body = GroomingBody(dust_seed=11, dust_by_region={"antenna": 1})
        before = body.step(0.005)

        locked = bridge.apply_readout(
            body, 0.005, {"groom_antenna": 1.0}, locked_legs=frozenset({"fore_left"}),
        )
        missing = bridge.apply_readout(body, 0.005, {"groom_head": 1.0})

        self.assertEqual(locked.targets_mm, {})
        self.assertEqual(locked.joint_angles["fore_left"], before.joint_angles["fore_left"])
        self.assertEqual(locked.cleaned, ())
        self.assertEqual(missing.targets_mm, {})
        self.assertEqual(missing.groom_antenna_readout, 0.0)
        self.assertFalse(missing.readout_present)
        self.assertEqual(missing.cleaned, ())

    def test_no_dust_can_never_produce_a_clean_event(self) -> None:
        bridge = _bridge()
        body = GroomingBody(dust_seed=2, dust_by_region={})
        sample = bridge.sample_input(body.tactile_loads(), 0.005)
        action = bridge.apply_readout(body, 0.005, {"groom_antenna": 1.0})
        audit = bridge.audit(sample, action).to_dict()

        self.assertEqual(sample.poisson_hz["JO_candidate_antenna"], 0.0)
        self.assertEqual(action.cleaned, ())
        self.assertEqual(audit["remaining"]["antenna"], 0)

    def test_strict_json_config_rejects_unknown_keys_and_duplicate_keys(self) -> None:
        valid = {
            "schema_version": 1,
            "selectors_by_region": {"antenna": "JO_candidate_antenna"},
            "rate_hz_by_region": {"antenna": 40.0},
            "readout_threshold": 0.0,
            "sweep_extent_mm": 0.12,
            "sweep_frequency_hz": 1.6,
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "bridge.json"
            config_path.write_text(json.dumps(valid), encoding="utf-8")
            loaded = load_config(config_path)
            self.assertEqual(loaded.selectors_by_region, {"antenna": "JO_candidate_antenna"})

            invalid = dict(valid, surprise=True)
            config_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                load_config(config_path)

            config_path.write_text(
                '{"schema_version":1,"schema_version":1}', encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
