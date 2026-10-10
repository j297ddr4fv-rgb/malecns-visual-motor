#!/usr/bin/env python3
"""Synthetic sign, parity, and contract tests for the FlyVis control readout."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from flyvis_control_readout import (  # noqa: E402
    DEFAULT_CSR,
    DEFAULT_TARGETS,
    FlyVisControlReadout,
    FlyVisTargetMapping,
    MappingError,
    ReadoutShapeError,
    load_target_mapping,
)
from flyvis_runtime_adapter import load_t4t5_targets  # noqa: E402


def _synthetic_mapping() -> FlyVisTargetMapping:
    # +y is left and +z is up.  The mirrored L/R pair makes a bilateral
    # common-mode response cancel yaw, while a signed push-pull pair produces
    # a deterministic yaw cue.
    return FlyVisTargetMapping(
        target_indices=np.array([11, 17, 23, 29], dtype=np.int64),
        directions=np.array([
            [1.0, +1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, +1.0],
            [1.0, 0.0, -1.0],
        ], dtype=np.float32),
        subtype=np.array(["T4a", "T4a", "T5a", "T5a"]),
        side=np.array(["L", "R", "L", "R"]),
    )


class FlyVisControlReadoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = _synthetic_mapping()
        self.readout = FlyVisControlReadout(
            mapping=self.mapping,
            config={
                "schema_version": "flyvis-control-readout-1",
                "name": "synthetic",
                "input": {"allowed_units": ["current_mV_per_ms"]},
                "readout": {
                    "bounds": [-1.0, 1.0],
                    "yaw_sign": 1.0,
                    "pitch_sign": 1.0,
                    "yaw_gain": 1.0,
                    "pitch_gain": 1.0,
                    "side_weights": {"L": 1.0, "R": 1.0},
                    "subtype_weights": {"T4a": 1.0, "T5a": 1.0},
                },
                "confidence": {"energy_scale": 1.0},
            },
        )

    def test_signed_yaw_and_pitch_survive_projection(self) -> None:
        # L/R signed push-pull: +left preferred response and -right preferred
        # response add instead of cancelling.  The vertical pair does the same
        # for pitch.  This is the failure mode of a scalar population mean.
        positive = self.readout(
            np.array([+1.0, -1.0, +1.0, -1.0], dtype=np.float32),
            target_indices=self.mapping.target_indices,
            units="current_mV_per_ms",
        )
        negative = self.readout(-np.array([+1.0, -1.0, +1.0, -1.0], dtype=np.float32))
        self.assertGreater(positive.yaw, 0.0)
        self.assertGreater(positive.pitch, 0.0)
        self.assertLess(negative.yaw, 0.0)
        self.assertLess(negative.pitch, 0.0)
        self.assertTrue(-1.0 <= positive.yaw <= 1.0)
        self.assertTrue(-1.0 <= positive.pitch <= 1.0)
        self.assertTrue(0.0 <= positive.confidence <= 1.0)
        self.assertEqual(positive["visual_direction"], positive.yaw)
        self.assertFalse(positive.provenance["emits_motor_commands"])

    def test_left_right_parity_is_explicit_and_not_a_scalar_mean(self) -> None:
        # Same signed response in mirrored eyes is a common-mode (zero yaw),
        # while swapping only the right sign creates a non-zero differential.
        common = self.readout(np.array([1.0, 1.0, 0.0, 0.0]))
        differential = self.readout(np.array([1.0, -1.0, 0.0, 0.0]))
        self.assertAlmostEqual(common.yaw, 0.0, places=7)
        self.assertGreater(differential.yaw, 0.0)
        self.assertGreater(common.side_cues["L"]["energy"], 0.0)
        self.assertGreater(common.side_cues["R"]["energy"], 0.0)
        self.assertEqual(common.side_cues["L"]["count"], 2)
        self.assertEqual(common.side_cues["R"]["count"], 2)

    def test_zero_response_is_neutral_and_confidence_zero(self) -> None:
        result = self.readout(np.zeros(self.mapping.n_targets, dtype=np.float32))
        self.assertEqual(result.yaw, 0.0)
        self.assertEqual(result.pitch, 0.0)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.response_energy, 0.0)
        self.assertTrue(result.valid)

    def test_shape_and_target_order_mismatch_fail_closed(self) -> None:
        with self.assertRaises(ReadoutShapeError):
            self.readout(np.zeros((1, self.mapping.n_targets), dtype=np.float32))
        with self.assertRaises(MappingError):
            self.readout(np.zeros(self.mapping.n_targets), target_indices=[11, 17, 29, 23])
        with self.assertRaises(MappingError):
            mismatched_directions = self.mapping.directions.copy()
            mismatched_directions[0] *= -1.0
            self.readout(
                np.zeros(self.mapping.n_targets),
                directions=mismatched_directions,
            )

    def test_unchanged_runtime_mapping_reuses_full_validation(self) -> None:
        runtime_mapping = {
            "target_indices": self.mapping.target_indices.copy(),
            "directions": self.mapping.directions.copy(),
            "subtype": self.mapping.subtype.copy(),
            "side": self.mapping.side.copy(),
        }
        response = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        with mock.patch.object(
            self.readout,
            "_validate_call_mapping_uncached",
            wraps=self.readout._validate_call_mapping_uncached,
        ) as uncached:
            first = self.readout(response, target_mapping=runtime_mapping)
            second = self.readout(response, target_mapping=runtime_mapping)
        self.assertEqual(uncached.call_count, 1)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_in_place_runtime_mapping_mutation_invalidates_cache_and_fails_closed(self) -> None:
        runtime_mapping = {
            "target_indices": self.mapping.target_indices.copy(),
            "directions": self.mapping.directions.copy(),
            "subtype": self.mapping.subtype.copy(),
            "side": self.mapping.side.copy(),
        }
        response = np.zeros(self.mapping.n_targets, dtype=np.float32)
        self.readout(response, target_mapping=runtime_mapping)
        runtime_mapping["directions"][0] = (0.0, 1.0, 0.0)
        with self.assertRaises(MappingError):
            self.readout(response, target_mapping=runtime_mapping)

    def test_changed_explicit_units_invalidates_cache_and_fails_closed(self) -> None:
        runtime_mapping = {
            "target_indices": self.mapping.target_indices.copy(),
            "directions": self.mapping.directions.copy(),
            "subtype": self.mapping.subtype.copy(),
            "side": self.mapping.side.copy(),
        }
        response = np.zeros(self.mapping.n_targets, dtype=np.float32)
        self.readout(
            response,
            target_mapping=runtime_mapping,
            units="current_mV_per_ms",
        )
        with self.assertRaises(MappingError):
            self.readout(
                response,
                target_mapping=runtime_mapping,
                units="unsupported_units",
            )

    def test_asset_mapping_loads_and_runtime_mapping_order_is_checked(self) -> None:
        mapping = load_target_mapping()
        self.assertEqual(mapping.n_targets, 13582)
        self.assertEqual(set(mapping.side.tolist()), {"L", "R"})
        # A runtime mapping can be used directly, but a reordered response is
        # never accepted merely because it has the same length.
        readout = FlyVisControlReadout(runtime_mapping=self.mapping, config=self.readout.config)
        with self.assertRaises(MappingError):
            readout(
                np.zeros(self.mapping.n_targets),
                target_indices=self.mapping.target_indices[::-1],
            )

    def test_production_side_metadata_matches_runtime_csr_authority(self) -> None:
        configured = load_target_mapping(DEFAULT_TARGETS, side_source=DEFAULT_CSR)
        runtime = load_t4t5_targets(DEFAULT_TARGETS, DEFAULT_CSR)
        np.testing.assert_array_equal(configured.target_indices, runtime["target_indices"])
        np.testing.assert_allclose(configured.directions, runtime["directions"], atol=1e-7, rtol=0.0)
        np.testing.assert_array_equal(configured.subtype, runtime["subtype"])
        np.testing.assert_array_equal(configured.side, runtime["side"])
        np.testing.assert_array_equal(configured.target_ids, runtime["target_ids"])
        self.assertEqual(
            {label: int(np.count_nonzero(configured.side == label)) for label in ("L", "R")},
            {"L": 6787, "R": 6795},
        )
        self.assertEqual(configured.manifest()["side_authority"],
                         "MaleCNS annotations.npz:sideIdx -> tables.json:sides")

    def test_side_is_not_inferred_from_direction_y(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no-side.npz"
            np.savez(
                path,
                idx=np.array([1, 2], dtype=np.int64),
                dirs=np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                type=np.array(["T4a", "T5a"]),
            )
            with self.assertRaisesRegex(MappingError, "lacks explicit side metadata"):
                load_target_mapping(path)

    def test_runtime_mapping_without_side_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(MappingError, "explicit side metadata"):
            FlyVisControlReadout(
                runtime_mapping={
                    "target_indices": np.array([1, 2], dtype=np.int64),
                    "directions": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
                    "subtype": np.array(["T4a", "T5a"]),
                },
                config=self.readout.config,
            )

    def test_npz_mapping_shape_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(
                path,
                idx=np.array([1, 2]),
                dirs=np.array([[1.0, 0.0, 0.0]]),
                type=np.array(["T4a", "T5a"]),
            )
            with self.assertRaises(MappingError):
                load_target_mapping(path)


if __name__ == "__main__":
    unittest.main()
