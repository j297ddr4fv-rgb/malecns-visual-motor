from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_courtship_yoked_replay import create_replay, decode_vector  # noqa: E402


def encoded(values: np.ndarray) -> dict:
    raw = np.ascontiguousarray(values, dtype=np.float32).tobytes()
    compressed = zlib.compress(raw)
    return {
        "encoding": "zlib+base64", "lossless": True,
        "dtype": "float32", "shape": [721],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


class YokedReplayTest(unittest.TestCase):
    def test_decodes_and_verifies_exact_columns(self) -> None:
        values = np.linspace(0.1, 0.8, 721, dtype=np.float32)
        record = encoded(values)
        np.testing.assert_array_equal(decode_vector(record, "left"), values)
        record["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            decode_vector(record, "left")

    def test_creates_binocular_replay_with_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "trial"
            trial.mkdir()
            (trial / "manifest.json").write_text(json.dumps({
                "run_classification": "courtship_closed_loop_visual_assay",
                "courtship_visual": {"target_condition": "moving", "trajectory_sha256": "a" * 64},
                "telemetry_contract": {"mode": "compact"},
                "components": {"vision": {"exact_direction_mapping": {"sha256": {
                    "left": "b" * 64, "right": "c" * 64,
                }}}},
            }), encoding="utf-8")
            with (trial / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
                for frame in range(2):
                    handle.write(json.dumps({
                        "frame_index": frame, "frame_time_s": frame * 0.005,
                        "visual_input": {
                            "sampling": {"mode": "exact_world_eye_directions"},
                            "left_columns": encoded(np.full(721, 0.2 + frame * 0.1, dtype=np.float32)),
                            "right_columns": encoded(np.full(721, 0.4 + frame * 0.1, dtype=np.float32)),
                        },
                    }) + "\n")
            output = Path(temp) / "yoke.npz"
            report = create_replay(trial, output)
            self.assertEqual(report["sample_count"], 2)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive["columns"].shape, (2, 2, 721))
                provenance = json.loads(str(archive["manifest_json"].item()))
            self.assertEqual(provenance["left_directions_sha256"], "b" * 64)
            self.assertEqual(provenance["right_directions_sha256"], "c" * 64)
            self.assertEqual(provenance["eye_mode"], "binocular_world_render")


if __name__ == "__main__":
    unittest.main()
