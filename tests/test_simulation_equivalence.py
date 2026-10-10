from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import simulation_equivalence as equivalence  # noqa: E402


def _row(index: int, *, shift: float = 0.0, hashes: bool = True,
         event: str | None = None, grounded: bool | None = None) -> dict[str, object]:
    counts = {name: 0 for name in ("LC4", "LPLC2", "DNp01", "TTMn")}
    if event in counts:
        counts[event] = 1
    nm_channels = {"wing_motor": 0.0}
    if event == "wing_motor":
        nm_channels["wing_motor"] = 0.25
    row: dict[str, object] = {
        "schema_version": "digital-fly-telemetry-compact-1",
        "frame_index": index,
        "frame_time_s": index * 0.005,
        "time_s": (index + 1) * 0.005,
        "interval_start_s": index * 0.005,
        "interval_end_s": (index + 1) * 0.005,
        "dt_s": 0.005,
        "position_m": [shift + index * 0.01, 0.0, 0.0012],
        "velocity_mps": [1.0 + shift, 0.0, 0.0],
        "orientation_rad": [0.1 + shift, 0.0, 0.0],
        "orientation_quat": [1.0, 0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.1 + shift],
        "left_wing": 0.1 + shift,
        "right_wing": 0.2 + shift,
        "wing_drive_left": 0.1 + shift,
        "wing_drive_right": 0.2 + shift,
        "grounded": True if grounded is None else grounded,
        "contact": True,
        "target_contact": False,
        "collision_count": 0,
        "neural": {"named_outputs": {"population_spike_counts": counts}},
        "nm_out": {"channels": nm_channels},
    }
    if hashes:
        row["visual_input"] = {
            "left_columns": {"sha256": f"left-{index}"},
            "right_columns": {"sha256": f"right-{index}"},
            "t4t5_drive": {"sha256": f"drive-{index}"},
        }
    return row


def _write_trace(root: Path, name: str, rows: list[dict[str, object]], *, manifest: dict[str, object] | None = None) -> Path:
    run = root / name
    run.mkdir()
    with (run / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
    if manifest is not None:
        (run / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return run


def _manifest(seed: int = 1729) -> dict[str, object]:
    return {
        "schema_version": "digital-fly-run-1",
        "seed": seed,
        "config": {"path": "config.json", "sha256": "a" * 64},
        "requested_backends": {"neural": "malecns", "vision": "flyvis", "body": "reduced"},
        "requested_renderer_backend": "warp_cuda",
        "effective_backends": {"neural": "triton", "vision": "flyvis", "renderer": "warp_cuda"},
        "requested_execution": {"neural_backend": "triton", "neural_device": "cuda", "telemetry_mode": "compact"},
        "effective_execution": {"neural_backend": "triton", "neural_device": "cuda", "renderer_backend": "warp_cuda"},
        "scenario_names": ["free-behavior"],
        "resource_limits": {"cgroup_memory_max_bytes": 5 * 1024**3},
    }


class SimulationEquivalenceTest(unittest.TestCase):
    def test_matching_stream_passes_and_quaternion_sign_is_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_row(i, event="LPLC2" if i == 1 else None) for i in range(3)]
            candidate_rows = [dict(row) for row in rows]
            for row in candidate_rows:
                row["orientation_quat"] = [-v for v in row["orientation_quat"]]  # type: ignore[index]
            ref = _write_trace(root, "ref", rows, manifest=_manifest())
            cand = _write_trace(root, "cand", candidate_rows, manifest=_manifest())
            report = equivalence.compare_runs(ref, cand)
        self.assertEqual(report["status"], "PASS", report["failures"])
        self.assertEqual(report["events"]["LPLC2"]["reference"]["frame_index"], 1)
        self.assertTrue(report["visual_input_hashes"]["left_columns"]["available"])
        self.assertEqual(report["continuous_metrics"]["orientation_quat"]["max_abs"], 0.0)

    def test_shifted_frame_and_event_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_rows = [_row(i, event="TTMn" if i == 1 else None) for i in range(3)]
            cand_rows = [_row(i, event="TTMn" if i == 2 else None) for i in range(3)]
            cand_rows[1]["frame_index"] = 2
            ref = _write_trace(root, "ref", ref_rows)
            cand = _write_trace(root, "cand", cand_rows)
            report = equivalence.compare_runs(ref, cand)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("reorder/missing" in reason for reason in report["failures"]))
        self.assertTrue(any("event frame mismatch" in reason for reason in report["failures"]))

    def test_visual_hash_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_rows = [_row(i) for i in range(2)]
            cand_rows = [_row(i) for i in range(2)]
            cand_rows[1]["visual_input"]["left_columns"]["sha256"] = "different"  # type: ignore[index]
            report = equivalence.compare_runs(
                _write_trace(root, "ref", ref_rows),
                _write_trace(root, "cand", cand_rows),
            )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("visual input hash mismatch: left_columns" in reason for reason in report["failures"]))

    def test_nonfinite_value_fails_and_missing_optional_evidence_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_rows = [_row(0, hashes=False)]
            cand_rows = [_row(0, hashes=False)]
            cand_rows[0]["position_m"] = [float("nan"), 0.0, 0.0]
            report = equivalence.compare_runs(
                _write_trace(root, "ref", ref_rows),
                _write_trace(root, "cand", cand_rows),
            )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("nonfinite" in reason for reason in report["failures"]))
        self.assertTrue(any(item["observable"] == "visual_input.left_columns" for item in report["unavailable"]))

    def test_continuous_tolerance_and_workload_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_rows = [_row(0)]
            cand_rows = [_row(0, shift=1.0e-5)]
            ref = _write_trace(root, "ref", ref_rows, manifest=_manifest())
            cand = _write_trace(root, "cand", cand_rows, manifest=_manifest(seed=7))
            report = equivalence.compare_runs(ref, cand)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("workload identity mismatch", report["failures"])
        self.assertTrue(any("position max_abs" in reason for reason in report["failures"]))

    def test_both_missing_visual_and_orientation_evidence_are_unavailable_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_rows = [_row(0, hashes=False)]
            cand_rows = [_row(0, hashes=False)]
            del ref_rows[0]["orientation_rad"]
            del cand_rows[0]["orientation_rad"]
            del ref_rows[0]["orientation_quat"]
            del cand_rows[0]["orientation_quat"]
            report = equivalence.compare_runs(
                _write_trace(root, "ref", ref_rows),
                _write_trace(root, "cand", cand_rows),
            )
        self.assertEqual(report["status"], "PASS", report["failures"])
        names = {item["observable"] for item in report["unavailable"]}
        self.assertIn("orientation", names)
        self.assertIn("orientation_quat", names)
        self.assertIn("visual_input.left_columns", names)


if __name__ == "__main__":
    unittest.main()
