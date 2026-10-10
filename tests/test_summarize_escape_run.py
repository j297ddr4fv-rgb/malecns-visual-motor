#!/usr/bin/env python3
"""CPU fixtures for the streaming strict escape-run summarizer."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_escape_run import (  # noqa: E402
    COMPARISON_SCHEMA_VERSION,
    SCHEMA_VERSION,
    compare_runs,
    render_markdown,
    summarize_jsonl,
)


def _manifest(condition: str = "looming_grounded") -> dict[str, object]:
    return {
        "schema_version": "digital-fly-run-1",
        "execution_policy": "strict-free-behavior-escape-experiment",
        "seed": 1729,
        "autonomy_policy": {
            "strict": True,
            "name": "strict-free-behavior-escape-experiment",
            "control_authority": "MaleCNS",
        },
        "provenance": {
            "evidence_class": "digital_simulation",
            "world": "handbuilt",
            "vision": "fitted",
            "neural": "connectome",
            "neuromuscular": "connectome_adapter",
            "body": "handbuilt",
        },
        "components": {
            "physical_stimulus": {
                "condition": condition,
                "name": condition,
                "interaction": "visual_only_non_interacting",
                "visual_only": True,
                "state_for_control": "prohibited;telemetry_only",
            },
            "experiment": {"visual_baseline": {"duration_s": 0.1}},
        },
        "resource_limits": {
            "cgroup_memory_max_bytes": 5 * 1024**3,
            "torch_allocator_max_vram_gib": 15.5,
        },
        "resource_usage": {
            "peak_host_rss_bytes": 2 * 1024**3,
            "current_host_rss_bytes": 1024**3,
            "cuda": {"device": "cuda:0", "peak_reserved_bytes": 2 * 1024**3},
        },
    }


def _row(index: int, *, condition: str = "looming_grounded", lc4: float = 0.0, lplc2: float = 0.0, dnp01: float = 0.0, ttmn: float = 0.0, wing: float = 0.0, grounded: bool = True) -> dict[str, object]:
    time_s = index * 0.1
    return {
        "schema_version": "digital-fly-free-behavior-telemetry-1",
        "scenario": "free-behavior",
        "frame_index": index,
        "time_s": time_s,
        "dt_s": 0.1,
        "interval_start_s": time_s - 0.1,
        "interval_end_s": time_s,
        "grounded": grounded,
        "contact": grounded,
        "position_m": [0.0, 0.0, 0.0012 if grounded else 0.04],
        "velocity_mps": [0.0, 0.0, 0.0 if grounded else 0.2],
        "orientation_rad": [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "flight_mode": "free_behavior",
        "stimulus_condition": condition,
        "stimulus_name": condition,
        "neural": {
            "named_outputs": {
                "LC4": lc4,
                "LPLC2": lplc2,
                "DNp01": dnp01,
                "TTMn": ttmn,
                "superclass:vnc_motor&subclass:wm": wing,
                "spike_count": int(lc4 + lplc2 + dnp01 + ttmn + wing),
                "active_neurons": 10 + index,
                "population_spike_counts": {
                    "LC4": int(lc4 > 0),
                    "LPLC2": int(lplc2 > 0),
                    "DNp01": int(dnp01 > 0),
                    "TTMn": int(ttmn > 0),
                    "superclass:vnc_motor&subclass:wm": int(wing > 0),
                },
                "population_rates_hz": {
                    "LC4": lc4,
                    "LPLC2": lplc2,
                    "DNp01": dnp01,
                    "TTMn": ttmn,
                    "superclass:vnc_motor&subclass:wm": wing,
                },
                "window_start_timestamp": {"t_s": time_s - 0.1},
                "window_end_timestamp": {"t_s": time_s},
            }
        },
        "nm_out": {
            "channels": {"wing_motor": wing, "ttmn_activity": ttmn},
            "provenance": {"map_sha256": "map-fixture", "dataset": "fixture", "calibration_id": "cal-fixture"},
        },
        "frame_provenance": {
            "policy": "strict-free-behavior-escape-experiment",
            "control_authority": "MaleCNS",
            "autopilot_overlay": "prohibited",
            "flight_state_drive": "prohibited",
            "flight_body_stabilizer": "prohibited",
            "mission_goal_phase": "prohibited",
            "stimulus_label_control": False,
            "stimulus_direction_control": False,
            "future_state_control": False,
            "direct_neuromuscular_to_dynamics": True,
            "stimulus_interaction": "visual_only_non_interacting",
        },
        "provenance": {
            "evidence_class": "digital_simulation",
            "neural": "connectome",
            "neuromuscular": "connectome",
            "body": "handbuilt",
            "vision": "fitted",
        },
        "diagnostics": {"warnings": [], "counters": {"collision_count": 0}},
        "collision_count": 0,
    }


def _write_run(root: Path, condition: str, values: list[dict[str, object]]) -> Path:
    run = root / condition
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(_manifest(condition)), encoding="utf-8")
    with (run / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value) + "\n")
    return run


class SummarizeEscapeRunTest(unittest.TestCase):
    def test_stream_summary_extracts_intervals_chain_totals_body_and_integrity(self) -> None:
        rows = [
            _row(0),
            _row(1, lc4=20.0),
            _row(2, lplc2=30.0),
            _row(3, dnp01=100.0, wing=0.5),
            _row(4, ttmn=40.0, wing=0.8, grounded=False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run(Path(tmp), "looming_grounded", rows)
            report = summarize_jsonl(run, compute_hash=True)

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["run"]["line_count"], 5)
        self.assertEqual(report["run"]["stimulus_onset_s"], 0.1)
        events = report["causal_chain"]["events"]
        self.assertEqual(events["LC4"]["first_nonzero_after_stimulus_s"], 0.1)
        self.assertEqual(events["LC4"]["first_nonzero_interval_s"], [0.0, 0.1])
        self.assertEqual(events["DNp01"]["total_spikes"], 1)
        self.assertEqual(events["TTMn"]["first_nonzero_s"], 0.30000000000000004)
        self.assertEqual(events["liftoff"]["first_nonzero_s"], 0.30000000000000004)
        self.assertEqual(report["final_body_state"]["grounded"], False)
        self.assertEqual(report["resources"]["integrity_status"], "PASS")
        self.assertEqual(report["provenance"]["status"], "PASS")
        self.assertTrue(report["input"]["sha256"])
        self.assertIn("biological validation", render_markdown(report).lower())

    def test_malformed_line_is_counted_without_aborting_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _write_run(root, "sham_grounded", [_row(0, condition="sham_grounded"), _row(1, condition="sham_grounded")])
            telemetry = run / "telemetry.jsonl"
            telemetry.write_text(telemetry.read_text() + "{not-json}\n", encoding="utf-8")
            report = summarize_jsonl(telemetry, compute_hash=False)
        self.assertEqual(report["run"]["line_count"], 3)
        self.assertEqual(report["run"]["valid_row_count"], 2)
        self.assertEqual(report["run"]["malformed_lines"], 1)
        self.assertIsNone(report["input"]["sha256"])

    def test_compare_reports_moving_static_and_sham_deltas(self) -> None:
        base = [_row(0), _row(1)]
        moving = [_row(0), _row(1, lc4=20.0, wing=0.5, condition="looming_grounded")]
        sham = [_row(0, condition="sham_grounded"), _row(1, condition="sham_grounded")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "looming_grounded", moving)
            _write_run(root, "static_grounded", base)
            _write_run(root, "sham_grounded", sham)
            report = compare_runs({
                "moving": root / "looming_grounded",
                "static": root / "static_grounded",
                "sham": root / "sham_grounded",
            }, compute_hash=False)
        self.assertEqual(report["schema_version"], COMPARISON_SCHEMA_VERSION)
        self.assertEqual(report["selected"], {"moving": "moving", "static": "static", "sham": "sham"})
        self.assertIn("moving_vs_static", report["deltas"])
        self.assertIn("moving_vs_sham", report["deltas"])
        self.assertIsNone(report["deltas"]["moving_vs_static"]["timing"]["LC4"]["delta_s_moving_minus_baseline"])
        self.assertIsNotNone(report["deltas"]["moving_vs_static"]["channel_integral_deltas"]["wing_motor"]["delta"])
        self.assertIn("digital-simulation", render_markdown(report))

    def test_cli_writes_json_and_markdown(self) -> None:
        rows = [_row(0), _row(1, lc4=2.0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _write_run(root, "looming_grounded", rows)
            output = root / "summary.json"
            markdown = root / "summary.md"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "summarize_escape_run.py"), str(run), "--no-hash", "--output", str(output), "--markdown", str(markdown)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["schema_version"], SCHEMA_VERSION)
            self.assertIn("# Strict escape-run summary", markdown.read_text())


if __name__ == "__main__":
    unittest.main()
