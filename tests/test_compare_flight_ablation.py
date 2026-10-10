#!/usr/bin/env python3
"""Tests for the same-seed digital-flight ablation comparator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_flight_ablation import (  # noqa: E402
    AblationComparisonError,
    compare_bundles,
    main,
)


def _rows(*, ablation: bool = False, time_shift: float = 0.0, scenario: str = "takeoff") -> list[dict[str, object]]:
    heights = [0.0, 0.2, 0.42, 0.5, 0.5]
    yaw = [0.0, 0.1, 0.3, 0.5, 0.6]
    steering = [1.0, 1.0, 1.0, 1.0, 1.0]
    if ablation:
        heights = [0.0, 0.08, 0.12, 0.1, 0.1]
        yaw = [0.0, -0.1, -0.2, -0.25, -0.3]
        steering = [-1.0, -1.0, -1.0, -1.0, -1.0]
    rows: list[dict[str, object]] = []
    for index, (height, heading, drive) in enumerate(zip(heights, yaw, steering)):
        rows.append({
            "time_s": index * 0.25 + time_shift,
            "scenario": scenario,
            "position_m": [0.0, 0.0, height],
            "contact": index == 0,
            "yaw": heading,
            "steering_response": drive,
            "visual_direction": 1.0,
            "collision_count": 0,
        })
    return rows


def _write_bundle(directory: Path, *, ablation: bool = False, **kwargs: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "digital-fly-run-1",
        "seed": 37,
        "scenario_names": [str(kwargs.get("scenario", "takeoff"))],
        "provenance": {
            "evidence_class": "digital_simulation",
            "world": "handbuilt",
            "vision": "fitted",
            "neural": "connectome",
            "body": "handbuilt",
            "ablation": "visual-drive-zero" if ablation else "calibrated-visual-drive",
        },
        "outputs": {"trajectory": "trajectory.json"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    trajectory = {
        "schema_version": "digital-fly-trajectory-1",
        "metadata": {"evidence_class": "digital_simulation"},
        "trajectory": _rows(ablation=ablation, **kwargs),
    }
    (directory / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    return directory


class CompareFlightAblationTest(unittest.TestCase):
    def test_same_seed_directory_bundles_report_control_differences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _write_bundle(root / "candidate")
            ablation = _write_bundle(root / "ablation", ablation=True)
            output = root / "ablation_report.json"

            report = compare_bundles(candidate, ablation, output)

            self.assertTrue(output.is_file())
            self.assertEqual(report["decision"], "NOT_VALIDATED")
            self.assertFalse(report["biological_validation"])
            self.assertEqual(report["evidence"]["class"], "digital_simulation")
            self.assertFalse(report["evidence"]["real_biological"])
            self.assertTrue(report["compatibility"]["passed"])
            self.assertLess(report["differences"]["takeoff"]["max_altitude_m"], 0.0)
            self.assertTrue(report["causal_control_evidence"]["steering_response"]["reversed"])
            self.assertTrue(report["causal_control_evidence"]["yaw_response"]["reversed"])

            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "flight-ablation-report-1")
            self.assertNotIn("trajectory", saved["inputs"]["candidate"])

    def test_mismatched_seed_is_rejected_before_report_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _write_bundle(root / "candidate")
            ablation = _write_bundle(root / "ablation", ablation=True)
            manifest_path = ablation / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["seed"] = 38
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "should-not-exist.json"

            with self.assertRaisesRegex(AblationComparisonError, "seed mismatch"):
                compare_bundles(candidate, ablation, output)
            self.assertFalse(output.exists())

    def test_mismatched_time_grid_scenario_and_shared_provenance_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _write_bundle(root / "candidate")
            for label, kwargs, message in (
                ("time", {"time_shift": 0.01}, "time-grid mismatch"),
                ("scenario", {"scenario": "hover"}, "scenario mismatch"),
            ):
                ablation = _write_bundle(root / f"ablation-{label}", ablation=True, **kwargs)
                with self.assertRaisesRegex(AblationComparisonError, message):
                    compare_bundles(candidate, ablation)

            ablation = _write_bundle(root / "ablation-provenance", ablation=True)
            manifest_path = ablation / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["provenance"]["world"] = "different-world"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(AblationComparisonError, "provenance mismatch"):
                compare_bundles(candidate, ablation)

    def test_declared_visual_drive_channel_may_change_but_world_must_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _write_bundle(root / "candidate")
            ablation = _write_bundle(root / "ablation", ablation=True)
            manifest_path = ablation / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["provenance"]["visual_drive"] = "zeroed"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = compare_bundles(candidate, ablation)
            self.assertTrue(report["compatibility"]["provenance"]["matched"])

    def test_cli_writes_report_and_returns_nonzero_on_incompatible_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _write_bundle(root / "candidate")
            ablation = _write_bundle(root / "ablation", ablation=True)
            output = root / "cli-report.json"
            self.assertEqual(main([str(candidate), str(ablation), "--output", str(output)]), 0)
            self.assertTrue(output.is_file())

            bad_manifest = json.loads((ablation / "manifest.json").read_text(encoding="utf-8"))
            bad_manifest["seed"] = 999
            (ablation / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
            self.assertEqual(main([str(candidate), str(ablation), "--output", str(root / "bad.json")]), 2)


if __name__ == "__main__":
    unittest.main()
