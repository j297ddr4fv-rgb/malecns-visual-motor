"""Small CPU-only fixtures for the streaming leg long-run analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_leg_long_run import (  # noqa: E402
    LegLongRunAnalysisError,
    analyze_run,
)


DT = 0.0005
TICKS = 4000
DURATION = TICKS * DT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _body(time_s: float, *, height: float, x: float, contacts: int,
          normal_force: float, speed: float = 0.0) -> dict[str, object]:
    feet = {
        "LF": {"contact": contacts >= 1, "normal_force_N": normal_force / 2},
        "RF": {"contact": contacts >= 2, "normal_force_N": normal_force / 2},
    }
    return {
        "time_s": time_s,
        "root": {
            "position_m": [x, 0.0, height],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_mps": [speed, 0.0, 0.0],
            "angular_velocity_radps": [0.0, 0.0, 0.0],
        },
        "joints": {"LF_tibia_pitch": {"q_rad": 0.1, "qd_rad_s": 0.0}},
        "feet": feet,
        "total_foot_normal_force_N": normal_force,
        "body_ground_contact": False,
    }


def _snapshot(time_s: float, *, height: float, x: float, contacts: int,
              normal_force: float) -> dict[str, object]:
    return {
        "time_s": time_s,
        "qpos": [x, 0.0, height, 1.0, 0.0, 0.0, 0.0, 0.1],
        "qvel": [0.0] * 7,
        "qacc_warmstart": [0.0] * 7,
        "telemetry": _body(time_s, height=height, x=x, contacts=contacts,
                           normal_force=normal_force),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _make_run(parent: Path, *, legacy_counts: bool = False) -> Path:
    run = parent / "run"
    run.mkdir()
    stage = {"neural_s": 0.08, "muscle_physics_s": 0.2,
             "sensory_encode_s": 0.04, "other_s": 0.08}
    window_stage = {name: value / 2 for name, value in stage.items()}
    windows = [
        {"start_tick": 0, "end_tick": 1999, "simulated_end_s": 1.0,
         "elapsed_s": 0.2, "stages_s": window_stage},
        {"start_tick": 2000, "end_tick": 3999, "simulated_end_s": 2.0,
         "elapsed_s": 0.2, "stages_s": window_stage},
    ]
    performance = {
        "schema_version": "leg-neuromuscular-performance-1",
        "initialization_s": 0.15,
        "simulation_loop_s": 0.4,
        "stages_s": stage,
        "windows": windows,
        "neural_ticks": TICKS,
        "simulated_duration_s": DURATION,
    }
    _write_json(run / "performance.json", performance)

    telemetry_ticks = list(range(0, TICKS, 10))
    if telemetry_ticks[-1] != TICKS - 1:
        telemetry_ticks.append(TICKS - 1)
    telemetry_rows = []
    for tick in telemetry_ticks:
        time_s = (tick + 1) * DT
        fraction = (tick + 1) / TICKS
        height = 0.0009 if tick == 1500 else 0.001 + 0.0001 * fraction
        normal_force = 0.9e-5 if tick == 1500 else 1e-5 + 0.05e-5 * fraction
        body = _body(time_s, height=height, x=0.0003 * fraction,
                     contacts=1 if tick < 2000 else 2,
                     normal_force=normal_force, speed=0.01 + 0.03 * fraction)
        telemetry_rows.append({"time_s": time_s, "body": body, "neural": {"tick": tick}})
    with (run / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
        for row in telemetry_rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    total_spikes = target_spikes = afferent_spikes = mapped_motor_spikes = 0
    with (run / "neural_spike_counts.jsonl").open("w", encoding="utf-8") as handle:
        for tick in range(TICKS):
            total = int(tick in (10, 3999))
            target = int(tick == 10)
            afferent = int(tick in (10, 3900))
            mapped_motor = int(tick == 3999)
            row = {"tick": tick, "t_s": tick * DT,
                   "total_spikes": total, "target_spikes": target}
            if not legacy_counts:
                row.update({"afferent_spikes": afferent,
                            "mapped_motor_spikes": mapped_motor})
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            total_spikes += total
            target_spikes += target
            afferent_spikes += afferent
            mapped_motor_spikes += mapped_motor

    (run / "sensory_feedback.jsonl").write_bytes(b"")
    initial = _snapshot(0.0, height=0.001, x=0.0, contacts=1, normal_force=1e-5)
    final = _snapshot(DURATION, height=0.0011, x=0.0003, contacts=2,
                      normal_force=1.05e-5)
    summary: dict[str, object] = {
        "case": "sham", "leg": "LF", "evidence_class": "digital_simulation",
        "duration_s": DURATION, "neural_dt_s": DT,
        "neural_spike_shape": [TICKS, 2], "full_neural_trace_ticks": TICKS,
        "gpu_used": False, "neural_backend_actual": "numpy_cpu",
        "neural_device_actual": "cpu", "physics_backend": "mujoco_cpu",
        "sensory_feedback_mode": "off",
        "total_spikes": total_spikes, "target_spikes": target_spikes,
        "initial": initial, "final": final,
        "simulation_loop_wall_s": 0.4, "initialization_wall_s": 0.15,
    }
    if not legacy_counts:
        summary.update({"afferent_spikes": afferent_spikes,
                        "mapped_motor_spikes": mapped_motor_spikes})
    _write_json(run / "summary.json", summary)

    manifest_summary = {key: summary[key] for key in (
        "duration_s", "neural_dt_s", "neural_spike_shape", "full_neural_trace_ticks",
        "total_spikes", "gpu_used", "neural_backend_actual", "neural_device_actual")}
    if not legacy_counts:
        manifest_summary.update({"afferent_spikes": afferent_spikes,
                                 "mapped_motor_spikes": mapped_motor_spikes})
    body_provenance: dict[str, object] = {
        "total_model_mass_kg": 1e-6,
    }
    if not legacy_counts:
        body_provenance["passive_support"] = {"enabled": True}
    source_fields: dict[str, object] = {}
    if legacy_counts:
        source_dir = run / "sources"
        source_dir.mkdir()
        body_config = {"physics": {"gravity_mps2": 10.0}}
        body_config_path = source_dir / "articulated_leg_body.json"
        _write_json(body_config_path, body_config)
        body_config_sha = _sha(body_config_path)
        body_provenance["config_sha256"] = body_config_sha
        source_fields = {
            "source_snapshot_directory": "sources",
            "source_sha256": {"articulated_leg_body.json": body_config_sha},
        }
    else:
        body_provenance["effective_physics_parameters"] = {"gravity_mps2": 10.0}
    manifest = {
        "summary": manifest_summary,
        "telemetry_sha256": _sha(run / "telemetry.jsonl"),
        "neural_counts_sha256": _sha(run / "neural_spike_counts.jsonl"),
        "sensory_feedback_sha256": _sha(run / "sensory_feedback.jsonl"),
        "performance": {"path": "performance.json", "sha256": _sha(run / "performance.json")},
        "pipeline": {"body": body_provenance},
        **source_fields,
    }
    _write_json(run / "manifest.json", manifest)
    return run


class AnalyzeLegLongRunTests(unittest.TestCase):
    def test_cpu_fixture_requires_opt_in_and_streams_complete_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root)
            output = root / "analysis.json"
            with self.assertRaisesRegex(LegLongRunAnalysisError, "does not prove actual CUDA"):
                analyze_run(run, output)
            self.assertFalse(output.exists())

            before = {path.name: _sha(path) for path in run.iterdir() if path.is_file()}
            report = analyze_run(run, output, allow_cpu=True)
            after = {path.name: _sha(path) for path in run.iterdir() if path.is_file()}

        self.assertEqual(before, after)
        self.assertEqual(report["run"]["integrity"], "PASS")
        self.assertFalse(report["execution"]["gpu_execution_verified"])
        self.assertTrue(report["execution"]["cpu_analysis_exception"])
        self.assertEqual(report["neural_activity"]["total_afferent_spikes"], 2)
        self.assertEqual(report["neural_activity"]["first_sensory_spike"]["tick"], 10)
        self.assertEqual(report["neural_activity"]["last_sensory_spike"]["tick"], 3900)
        self.assertEqual(report["neural_activity"]["last_1s_response"]["total_spikes"]["status"],
                         "observed")
        self.assertEqual(report["performance"]["post_initial_window_timing"]["window_count"], 1)
        body = report["body"]
        self.assertAlmostEqual(body["sampled_root_height_m"]["minimum"], 0.0009)
        self.assertAlmostEqual(body["sampled_root_height_m"]["final_sample"], 0.0011)
        self.assertAlmostEqual(body["sampled_root_stability"]["telemetry_endpoint_drift_m"][0],
                               0.0003, places=6)
        self.assertEqual(body["sampled_foot_support"]["nonfoot_ground_contact_sample_fraction"], 0)
        self.assertIn("engineering terms", report["interpretation"]["passive_support"])

    def test_old_zero_spike_counts_mark_unrecorded_subcounts_as_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root, legacy_counts=True)
            summary = json.loads((run / "summary.json").read_text())
            summary["total_spikes"] = 0
            summary["target_spikes"] = 0
            _write_json(run / "summary.json", summary)
            counts_path = run / "neural_spike_counts.jsonl"
            with counts_path.open("w", encoding="utf-8") as handle:
                for tick in range(TICKS):
                    handle.write(json.dumps({"tick": tick, "t_s": tick * DT,
                                             "total_spikes": 0, "target_spikes": 0}) + "\n")
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["summary"]["total_spikes"] = 0
            manifest["summary"]["target_spikes"] = 0
            manifest["neural_counts_sha256"] = _sha(counts_path)
            _write_json(run / "manifest.json", manifest)
            report = analyze_run(run, root / "analysis.json", allow_cpu=True)

        activity = report["neural_activity"]
        self.assertEqual(activity["total_afferent_spikes"], 0)
        self.assertEqual(activity["total_mapped_motor_spikes"], 0)
        self.assertIn("inferred", activity["subcount_provenance"])
        self.assertEqual(activity["last_1s_response"]["mapped_motor_spikes"]["status"],
                         "none_observed")

    def test_old_nonzero_spike_counts_leave_unrecorded_subcounts_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root, legacy_counts=True)
            report = analyze_run(run, root / "analysis.json", allow_cpu=True)

        activity = report["neural_activity"]
        self.assertIsNone(activity["total_afferent_spikes"])
        self.assertIsNone(activity["total_mapped_motor_spikes"])
        self.assertIn("cannot be inferred", activity["subcount_provenance"])
        self.assertEqual(activity["last_1s_response"]["mapped_motor_spikes"]["status"],
                         "not_recorded")

    def test_legacy_missing_passive_support_provenance_stays_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root, legacy_counts=True)
            report = analyze_run(run, root / "analysis.json", allow_cpu=True)

        self.assertIn("status was not recorded", report["interpretation"]["passive_support"])
        self.assertIn("makes no claim", report["interpretation"]["passive_support"])

    def test_rejects_incomplete_performance_clock_hash_fault_and_nested_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root)
            nested = run / "analysis.json"
            with self.assertRaisesRegex(LegLongRunAnalysisError, "outside the input"):
                analyze_run(run, nested, allow_cpu=True)

            counts = run / "neural_spike_counts.jsonl"
            raw_counts = counts.read_bytes()
            counts.write_bytes(raw_counts[:-1] + b" \n")
            with self.assertRaisesRegex(LegLongRunAnalysisError, "SHA-256"):
                analyze_run(run, root / "hash-fault.json", allow_cpu=True)

            # In a fresh run, remove a telemetry sample while updating its
            # hash so the recorded tick cadence is exercised independently.
            clock_root = root / "clock"
            clock_root.mkdir()
            run = _make_run(clock_root)
            manifest = json.loads((run / "manifest.json").read_text())
            telemetry = run / "telemetry.jsonl"
            lines = telemetry.read_bytes().splitlines(keepends=True)
            telemetry.write_bytes(b"".join(lines[:3] + lines[4:]))
            manifest["telemetry_sha256"] = _sha(telemetry)
            _write_json(run / "manifest.json", manifest)
            with self.assertRaisesRegex(LegLongRunAnalysisError, "cadence"):
                analyze_run(run, clock_root / "incomplete-telemetry.json", allow_cpu=True)

            # A separate run with its own hash updated exercises performance
            # window coverage rather than the file-integrity rejection.
            clock_root = root / "performance-clock"
            clock_root.mkdir()
            run = _make_run(clock_root)
            manifest = json.loads((run / "manifest.json").read_text())
            performance_path = run / "performance.json"
            performance = json.loads(performance_path.read_text())
            performance["windows"] = performance["windows"][:1]
            _write_json(performance_path, performance)
            manifest["performance"]["sha256"] = _sha(performance_path)
            _write_json(run / "manifest.json", manifest)
            with self.assertRaisesRegex(LegLongRunAnalysisError, "window|tick schedule"):
                analyze_run(run, root / "incomplete-clock.json", allow_cpu=True)

    def test_cli_writes_json_only_to_a_new_external_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root)
            output = root / "cli-analysis.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "analyze_leg_long_run.py"),
                 str(run), "--out", str(output), "--allow-cpu"],
                check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["schema_version"],
                             "leg-long-run-analysis-1")
            second = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "analyze_leg_long_run.py"),
                 str(run), "--out", str(output), "--allow-cpu"],
                check=False, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
