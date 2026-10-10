#!/usr/bin/env python3
"""Focused tests for the safe digital-fly experiment toolbox."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_toolbox import (  # noqa: E402
    ConfigurationError,
    MAX_VRAM_GIB,
    HOST_RAM_LIMIT_GIB,
    ToolboxError,
    allowed_artifact_path,
    build_launch_spec,
    build_seed_matrix,
    derive_seed,
    inventory_artifacts,
    shell_command,
    status_report,
    validate_configuration,
    validate_protocol,
)


class ExperimentToolboxTest(unittest.TestCase):
    def test_default_configuration_and_protocol_are_valid(self) -> None:
        report = validate_configuration()
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["resource_policy"]["host_ram_gib"], HOST_RAM_LIMIT_GIB)
        self.assertEqual(report["resource_policy"]["max_vram_gib"], MAX_VRAM_GIB)

    def test_seed_derivation_is_deterministic_uint32(self) -> None:
        first = derive_seed("escape-e14-v1", 20260921, "opaque-001")
        self.assertEqual(first, derive_seed("escape-e14-v1", 20260921, "opaque-001"))
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, 0xFFFFFFFF)
        self.assertNotEqual(first, derive_seed("escape-e14-v1", 20260921, "opaque-002"))

    def test_matrix_pairs_controls_with_same_seed(self) -> None:
        rows = build_seed_matrix(replicates=2)
        self.assertEqual(len(rows), 24)
        for pair_id in {row["pair_id"] for row in rows}:
            paired = [row for row in rows if row["pair_id"] == pair_id]
            self.assertEqual(len({row["seed"] for row in paired}), 1)
            self.assertEqual(len({row["opaque_trial_id"] for row in paired}), 1)
        self.assertEqual(len({row["trial_id"] for row in rows}), len(rows))

    def test_launch_is_argv_only_and_resource_bounded(self) -> None:
        row = build_seed_matrix(replicates=1, conditions=["true_right"], tiers=["core_behavior"])[0]
        spec = build_launch_spec(row)
        self.assertTrue(spec.launchable)
        self.assertEqual(spec.argv[0], "systemd-run")
        self.assertIn("--property=MemoryMax=5G", spec.argv)
        self.assertIn("--max-vram-gib", spec.argv)
        self.assertIn(str(MAX_VRAM_GIB), spec.argv)
        self.assertFalse(any(token == "sh" for token in spec.argv))
        self.assertNotIn(";", spec.command or "")

    def test_artifact_write_boundary_rejects_unapproved_path(self) -> None:
        with self.assertRaises(ToolboxError):
            allowed_artifact_path("/tmp/not-an-artifact")
        self.assertTrue(str(allowed_artifact_path(ROOT / "artifacts" / "toolbox-test")).startswith(str(ROOT / "artifacts")))

    def test_inventory_hashes_files_and_status_reads_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run-001"
            run.mkdir()
            (run / "summary.json").write_text(json.dumps({"status": "OK", "seed": 9}), encoding="utf-8")
            (run / "manifest.json").write_text(json.dumps({"status": "ERROR", "failure_class": "technical_abort", "trial_id": "trial-1", "seed": 9}), encoding="utf-8")
            (run / "trace.csv").write_text("t,value\n0,1\n", encoding="utf-8")
            inventory = inventory_artifacts(root, hash_files=True)
            self.assertEqual(inventory["file_count"], 3)
            self.assertTrue(all("sha256" in item for item in inventory["files"]))
            status = status_report(root)
            self.assertEqual(status["run_count"], 1)
            self.assertEqual(status["status_counts"], {"ERROR": 1})
            self.assertEqual(status["failure_class_counts"], {"technical_abort": 1})

    def test_invalid_protocol_fails_closed(self) -> None:
        protocol = {"schema_version": "x", "protocol_id": "x", "conditions": [], "speed_tiers": []}
        report = validate_protocol(protocol)
        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"])

    def test_command_display_quotes_but_execution_never_uses_shell(self) -> None:
        rendered = shell_command(["python3", "script.py", "--label", "a; touch SHOULD_NOT_EXIST"])
        self.assertIn("'a; touch SHOULD_NOT_EXIST'", rendered)
        self.assertNotIn("touch SHOULD_NOT_EXIST\n", rendered)


if __name__ == "__main__":
    unittest.main()
