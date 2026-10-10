from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from leg_muscle_decoder import LegMuscleDecoder  # noqa: E402
from neuromuscular_adapter import AnnotationIndex, DEFAULT_CSR  # noqa: E402


def synthetic_index() -> AnnotationIndex:
    """Tiny CSR-shaped index with real body IDs and intentionally unmapped MNs."""

    tables = {
        "types": ["Ti flexor MN", "MNml29", ""],
        "superclasses": ["vnc_motor"],
        "subclasses": ["fl", "ml", "hl"],
        "sides": ["L", "R"],
    }
    annotations = {
        "typeIdx": np.asarray([0, 0, 0, 0, 0, 1, 2], dtype=np.int32),
        "superclassIdx": np.zeros(7, dtype=np.int32),
        "subclassIdx": np.asarray([0, 0, 0, 1, 1, 2, 2], dtype=np.int32),
        "sideIdx": np.asarray([0, 0, 1, 0, 1, 0, 1], dtype=np.int32),
    }
    return AnnotationIndex(
        tables=tables,
        annotations=annotations,
        ids=[101, 102, 103, 104, 105, 106, 107],
    )


def six_leg_tibia_channels() -> list[dict[str, str]]:
    return [
        {"name": f"{leg}_tibia_flexor", "leg": leg,
         "group": "tibia_flexor", "type_name": "Ti flexor MN"}
        for leg in ("LF", "RF", "LM", "RM", "LH", "RH")
    ]


class LegMuscleDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = synthetic_index()
        self.decoder = LegMuscleDecoder(
            population_index=self.index,
            channels=six_leg_tibia_channels(),
        )

    def test_dense_rates_are_averaged_by_exact_type_and_leg(self) -> None:
        rates = np.asarray([15.0, 30.0, 30.0, 9.0, 18.0, 30.0, 0.0])
        decoded = self.decoder.decode(rates_hz=rates)
        self.assertAlmostEqual(decoded["LF_tibia_flexor"], 0.75)
        self.assertEqual(decoded["RF_tibia_flexor"], 1.0)
        self.assertAlmostEqual(decoded["LM_tibia_flexor"], 0.3)
        self.assertAlmostEqual(decoded["RM_tibia_flexor"], 0.6)
        self.assertEqual(decoded["LH_tibia_flexor"], 0.0)
        self.assertEqual(decoded["RH_tibia_flexor"], 0.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in decoded.values()))

    def test_named_exact_selectors_and_spike_counts_use_dt(self) -> None:
        provenance = self.decoder.provenance()
        left_spec = provenance["channels"]["LF_tibia_flexor"]["selector"]
        right_spec = provenance["channels"]["RF_tibia_flexor"]["selector"]
        decoded = self.decoder.decode(rates_hz={left_spec: 12.0, right_spec: 24.0})
        self.assertAlmostEqual(decoded["LF_tibia_flexor"], 0.4)
        self.assertAlmostEqual(decoded["RF_tibia_flexor"], 0.8)

        decoded = self.decoder.decode(spikes=[1, 3, 2, 0, 0, 0, 0], dt_s=0.1)
        self.assertAlmostEqual(decoded["LF_tibia_flexor"], 2.0 / 3.0)
        self.assertAlmostEqual(decoded["RF_tibia_flexor"], 2.0 / 3.0)
        with self.assertRaisesRegex(ValueError, "dt_s"):
            self.decoder.decode(spikes=[0] * 7)

    def test_provenance_has_real_ids_and_keeps_unmapped_mn_unassigned(self) -> None:
        provenance = self.decoder.provenance()
        left = provenance["channels"]["LF_tibia_flexor"]
        self.assertEqual(left["resolved_count"], 2)
        self.assertEqual(left["body_ids"], [101, 102])
        self.assertEqual(provenance["channels"]["RF_tibia_flexor"]["body_ids"], [103])
        self.assertEqual(provenance["unmapped_counts_by_leg"]["LH"], 1)
        self.assertEqual(provenance["unmapped_counts_by_leg"]["RH"], 1)
        unmapped = {(item["leg"], item["body_id"], item["type_name"])
                    for item in provenance["unmapped_mn"]}
        self.assertIn(("LH", 106, "MNml29"), unmapped)
        self.assertIn(("RH", 107, ""), unmapped)
        self.assertTrue(all("MNml" not in channel["type_name"]
                            for channel in provenance["channels"].values()))
        json.dumps(provenance)

    def test_missing_type_group_returns_zero_count_and_zero_excitation(self) -> None:
        channels = [
            {"name": "LF_missing", "leg": "LF", "group": "test_group",
             "type_name": "nonexistent exact type"}
        ]
        decoder = LegMuscleDecoder(population_index=self.index, channels=channels)
        decoded = decoder.decode(rates_hz=np.ones(7, dtype=np.float64) * 30.0)
        self.assertEqual(decoded, {"LF_missing": 0.0})
        info = decoder.provenance()["channels"]["LF_missing"]
        self.assertEqual(info["resolved_count"], 0)
        self.assertEqual(info["body_ids"], [])

    def test_nonfinite_and_negative_signal_values_fail_closed(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf"), -0.1):
            rates = np.ones(7, dtype=np.float64)
            rates[0] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.decoder.decode(rates_hz=rates)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.decoder.decode(rates_hz={"LF_tibia_flexor": float("nan")})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.decoder.decode(spikes=[0, -1, 0, 0, 0, 0, 0], dt_s=0.1)

    @unittest.skipUnless(DEFAULT_CSR.is_dir(), "checked CSR annotation bundle is unavailable")
    def test_real_csr_leg_counts_and_exact_types(self) -> None:
        decoder = LegMuscleDecoder()
        provenance = decoder.provenance()
        expected = {"LF": 68, "RF": 67, "LM": 58, "RM": 58, "LH": 66, "RH": 64}
        resolved_by_leg: dict[str, int] = {leg: 0 for leg in expected}
        for item in provenance["channels"].values():
            resolved_by_leg[item["leg"]] += item["resolved_count"]
            self.assertTrue(item["type_name"])
            self.assertTrue(item["body_ids_available"])
        for leg, expected_count in expected.items():
            resolved_by_leg[leg] += provenance["unmapped_counts_by_leg"][leg]
            self.assertEqual(resolved_by_leg[leg], expected_count)
        self.assertEqual(len(provenance["channels"]), 6 * 19)
        self.assertTrue(any(
            entry["type_name"] and entry["type_name"].startswith(("MNml", "MNhl"))
            for entry in provenance["unmapped_mn"]
        ))
        decoded = decoder.decode(rates_hz=np.zeros(decoder.population_index.n))
        self.assertEqual(len(decoded), 6 * 19)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in decoded.values()))


if __name__ == "__main__":
    unittest.main()
