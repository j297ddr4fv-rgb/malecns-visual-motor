#!/usr/bin/env python3
"""Focused tests for causal six-leg sensory current encoding."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from leg_sensory_feedback import (  # noqa: E402
    CAMPANIFORM_SUBCLASS,
    CHORDOTONAL_SUBCLASS,
    JOINTS,
    LEG_ANNOTATIONS,
    LEGS,
    LegSensoryFeedback,
)


def _body(time_s: float = 0.01) -> dict:
    return {
        "time_s": time_s,
        "joints": {
            f"{leg}_{joint}": {"q_rad": 0.0, "qd_rad_s": 0.0}
            for leg in LEGS for joint in JOINTS
        },
        "feet": {leg: {"normal_force_N": 0.0} for leg in LEGS},
    }


class SyntheticAnnotationIndex:
    """Small annotation-index stand-in with real selector resolution behavior."""

    def __init__(self, *, drop_selector: str | None = None, motor_overlap: bool = False):
        config = json.loads((ROOT / "config" / "leg_sensory_feedback.json").read_text(encoding="utf-8"))
        self.targets: dict[str, list[int]] = {}
        cursor = 0
        for leg in LEGS:
            nerve, side, chord_count = LEG_ANNOTATIONS[leg]
            selectors = (
                f"class:mechanosensory_proprioceptive&subclass:{CHORDOTONAL_SUBCLASS}&nerve:{nerve}/{side}",
                f"class:mechanosensory_proprioceptive&subclass:{CAMPANIFORM_SUBCLASS}&nerve:{nerve}/{side}",
            )
            counts = (chord_count, config["legs"][leg]["campaniform_expected_count"])
            for selector, count in zip(selectors, counts):
                self.targets[selector] = list(range(cursor, cursor + count))
                cursor += count
        self.targets["superclass:vnc_motor"] = list(range(cursor, cursor + 8))
        if drop_selector is not None:
            self.targets[drop_selector] = self.targets[drop_selector][:-1]
        if motor_overlap:
            self.targets["superclass:vnc_motor"].append(self.targets[next(iter(self.targets))][0])
        self.ids = [900_000 + i for i in range(cursor + 16)]

    def resolve(self, selector: str) -> list[int]:
        return list(self.targets.get(selector, []))


class LegSensoryFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = LegSensoryFeedback(SyntheticAnnotationIndex())

    def _selector(self, subclass: str, nerve: str, side: str) -> str:
        return f"class:mechanosensory_proprioceptive&subclass:{subclass}&nerve:{nerve}/{side}"

    def test_zero_input_has_no_baseline_current_and_records_causality(self) -> None:
        frame = self.encoder.encode(_body(0.01), 0.02)
        self.assertEqual(frame["source_time_s"], 0.01)
        self.assertEqual(frame["applied_at_s"], 0.02)
        self.assertEqual(len(frame["currents_mV_per_ms"]), 12)
        self.assertEqual(set(frame["currents_mV_per_ms"].values()), {0.0})
        self.assertEqual(frame["current_units"], "mV/ms")
        self.assertFalse(frame["input_muted"])
        self.assertEqual(frame["derived_readouts"]["per_leg"]["LF"]["chordotonal_proxy_norm"], 0.0)
        self.assertEqual(frame["derived_readouts"]["per_leg"]["LF"]["foot_load_proxy_norm"], 0.0)

    def test_single_leg_motion_and_load_are_isolated_to_its_two_groups(self) -> None:
        body = _body()
        body["joints"]["LF_coxa_yaw"] = {"q_rad": 0.5, "qd_rad_s": 10.0}
        body["feet"]["LF"]["normal_force_N"] = self.encoder.per_leg_weight_reference_N
        frame = self.encoder.encode(body, 0.01)
        currents = frame["currents_mV_per_ms"]
        lf_chord = self._selector(CHORDOTONAL_SUBCLASS, "ProLN", "L")
        lf_camp = self._selector(CAMPANIFORM_SUBCLASS, "ProLN", "L")
        self.assertGreater(currents[lf_chord], 0.0)
        self.assertAlmostEqual(currents[lf_camp], 0.5)
        self.assertEqual(sum(value > 0 for value in currents.values()), 2)
        self.assertTrue(all(value == 0.0 for key, value in currents.items()
                            if key not in {lf_chord, lf_camp}))
        self.assertAlmostEqual(frame["derived_readouts"]["per_leg"]["LF"]["foot_load_proxy_norm"], 1.0)

    def test_all_proxy_signals_and_currents_are_bounded(self) -> None:
        body = _body()
        for leg in LEGS:
            for joint in JOINTS:
                body["joints"][f"{leg}_{joint}"] = {"q_rad": 100.0, "qd_rad_s": -10000.0}
            body["feet"][leg]["normal_force_N"] = 100.0
        frame = self.encoder.encode(body, 0.01)
        self.assertTrue(all(0.0 <= value <= 1.0
                            for leg in frame["derived_readouts"]["per_leg"].values()
                            for key, value in leg.items() if key.endswith("_norm")))
        self.assertTrue(all(0.0 <= value <= 0.5 for value in frame["currents_mV_per_ms"].values()))

    def test_muted_and_disabled_keep_readouts_and_all_selector_keys(self) -> None:
        body = _body()
        body["joints"]["RF_tibia_pitch"] = {"q_rad": 0.2, "qd_rad_s": 4.0}
        body["feet"]["RF"]["normal_force_N"] = 2e-7
        for kwargs, muted, enabled in (({"silenced": True}, True, True),
                                       ({"enabled": False}, False, False)):
            frame = self.encoder.encode(body, 0.02, **kwargs)
            self.assertEqual(len(frame["currents_mV_per_ms"]), 12)
            self.assertEqual(set(frame["currents_mV_per_ms"].values()), {0.0})
            self.assertEqual(frame["input_muted"], muted)
            self.assertEqual(frame["enabled"], enabled)
            self.assertGreater(frame["derived_readouts"]["per_leg"]["RF"]["chordotonal_proxy_norm"], 0.0)
            self.assertGreater(frame["raw_readouts"]["foot_normal_force_N"]["RF"], 0.0)

    def test_future_nonfinite_missing_leg_and_negative_force_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "future"):
            self.encoder.encode(_body(0.02), 0.019)
        bad = _body()
        bad["joints"]["LF_coxa_yaw"]["q_rad"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.encoder.encode(bad, 0.02)
        bad = _body()
        del bad["feet"]["LH"]
        with self.assertRaisesRegex(ValueError, "missing normal force"):
            self.encoder.encode(bad, 0.02)
        bad = _body()
        bad["feet"]["RH"]["normal_force_N"] = -1e-9
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.encoder.encode(bad, 0.02)

    def test_snapshot_shape_and_provenance_are_annotation_only(self) -> None:
        body = _body()
        frame = self.encoder.encode({"time_s": 0.01, "telemetry": body}, 0.02)
        self.assertEqual(frame["source_time_s"], 0.01)
        provenance = self.encoder.provenance()
        self.assertEqual(provenance["resolved_count_total"], 404)
        self.assertFalse(provenance["neuron_ids_included"])
        self.assertFalse(provenance["vnc_motor_target_overlap"])
        self.assertTrue(all(count > 0 for count in provenance["selectors"].values()))
        self.assertEqual(len(provenance["selectors"]), 12)
        self.assertEqual(provenance["selectors"], provenance["expected_selectors"])
        self.assertEqual(set(provenance["selector_body_id_sha256"]), set(provenance["selectors"]))
        self.assertEqual(provenance["vnc_motor_resolved_count"], 8)
        self.assertIn("active AnnotationIndex", provenance["identity_source"])

    def test_actual_selector_count_mismatch_fails_closed(self) -> None:
        selector = self._selector(CHORDOTONAL_SUBCLASS, "ProLN", "L")
        with self.assertRaisesRegex(ValueError, "resolved 22 neurons; expected 23"):
            LegSensoryFeedback(SyntheticAnnotationIndex(drop_selector=selector))

    def test_actual_motor_overlap_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap superclass:vnc_motor"):
            LegSensoryFeedback(SyntheticAnnotationIndex(motor_overlap=True))

    def test_unknown_config_keys_fail_closed(self) -> None:
        config = json.loads((ROOT / "config" / "leg_sensory_feedback.json").read_text(encoding="utf-8"))
        config["invented_directional_tuning"] = True
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown top-level config key"):
                LegSensoryFeedback(SyntheticAnnotationIndex(), config_path=path)


if __name__ == "__main__":
    unittest.main()
