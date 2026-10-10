from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_articulated_optomotor import (
    AnalysisError,
    _feet_contacts,
    _mode,
    _moving_minus_static,
    _permutation_summary,
    _select_historical_primary,
    analyze_trial,
    discover_runs,
    verify_declared_outputs,
)


LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
POOLS = tuple(f"superclass:vnc_motor&subclass:{leg}"
              for leg in ("fl/L", "fl/R", "ml/L", "ml/R", "hl/L", "hl/R"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HistoricalTrialTests(unittest.TestCase):
    def test_true_initial_pose_yaw_unwrap_and_unmapped_motor_total(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            telemetry = run_dir / "telemetry.jsonl"
            initial_yaw = 3.1
            counts = {pool: 1 for pool in POOLS}
            with telemetry.open("w", encoding="utf-8") as handle:
                for index in range(1500):
                    path_s = (index + 1) * 1e-6
                    yaw = (initial_yaw + (index + 1) * 0.01 + math.pi) % (2 * math.pi) - math.pi
                    row = {
                        "frame_index": index,
                        "frame_time_s": index * 0.005,
                        "position_m": [path_s, path_s / 2, 0.0012],
                        "orientation_rad": [0.0, 0.0, yaw],
                        "neural": {"named_outputs": {
                            "population_spike_counts": counts,
                            "neural_window_steps": 10,
                            "spikes_are_window_counts": True,
                            "neural_window_dt_s": 0.005,
                            "rates_semantics": "window_mean_hz",
                        }},
                    }
                    if index == 0:
                        row["body_before"] = {
                            "position_m": [0.0, 0.0, 0.0012],
                            "orientation_rad": [0.0, 0.0, initial_yaw],
                        }
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            run = {
                "path": run_dir,
                "telemetry_path": telemetry,
                "manifest": {"schema_version": "digital-fly-run-1"},
                "model": "historical_triton",
                "mode": "historical_triton",
                "seed": 1729,
                "condition": "left",
                "manifest_sha256": "a" * 64,
                "output_hashes": {},
            }
            result = analyze_trial(run)
            expected_dx = 0.0015
            expected_dy = 0.00075
            self.assertAlmostEqual(result["forward_displacement_mm"],
                                   1000 * (expected_dx * math.cos(initial_yaw)
                                           + expected_dy * math.sin(initial_yaw)), places=9)
            self.assertAlmostEqual(result["lateral_displacement_mm"],
                                   1000 * (-expected_dx * math.sin(initial_yaw)
                                           + expected_dy * math.cos(initial_yaw)), places=9)
            self.assertAlmostEqual(result["net_yaw_change_deg"], math.degrees(15.0), places=8)
            self.assertEqual(result["annotated_leg_spikes"], 9000)
            self.assertEqual(set(result["six_leg_motor_spikes"].values()), {1500})
            # The old sampled population pools are not a mapped-muscle motor total.
            self.assertIsNone(result["mapped_motor_spikes"])

    def test_canonical_historical_runs_are_primary_and_repeats_are_context(self) -> None:
        primary = [
            {"condition": condition, "repeat_index": None, "path": Path(f"primary-{condition}")}
            for condition in ("left", "right", "static")
        ]
        repeated = [
            {"condition": condition, "repeat_index": 1, "path": Path(f"repeat1-{condition}")}
            for condition in ("left", "right")
        ]
        skipped: list[dict] = []
        selected, contextual = _select_historical_primary(primary + repeated, skipped)
        self.assertEqual({row["path"].name for row in selected},
                         {"primary-left", "primary-right", "primary-static"})
        self.assertEqual({row["path"].name for row in contextual},
                         {"repeat1-left", "repeat1-right"})
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(row["status"] == "UNSELECTED_CONTEXT_ONLY" for row in skipped))


class IntegrityAndStatisticsTests(unittest.TestCase):
    def test_declared_output_hashes_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            output = run_dir / "telemetry.jsonl"
            output.write_bytes(b"one\n")
            manifest = {"artifacts": {"telemetry.jsonl": {
                "path": "telemetry.jsonl", "sha256": sha256(output),
            }}}
            self.assertEqual(verify_declared_outputs(run_dir, manifest, required=False),
                             {"telemetry.jsonl": sha256(output)})
            output.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(AnalysisError, "digest mismatch"):
                verify_declared_outputs(run_dir, manifest, required=False)

    def test_low_seed_count_is_not_given_a_permutation_p_value(self) -> None:
        result = _permutation_summary([0.134])
        self.assertIsNone(result["exact_two_sided_sign_flip_p"])
        self.assertEqual(result["test_status"], "not_estimable_low_seed_count")
        self.assertEqual(result["n_seeds"], 1)

    def test_exact_sign_flip_uses_seed_level_values(self) -> None:
        result = _permutation_summary([1.0] * 5)
        self.assertEqual(result["permutations"], 32)
        self.assertEqual(result["exact_two_sided_sign_flip_p"], 2 / 32)
        self.assertEqual(result["test_status"], "exact_sign_flip_permutation_descriptive_seedset_only")

    def test_equal_moving_conditions_and_static_have_zero_seed_contrast(self) -> None:
        metrics = {
            "forward_displacement_mm": 1.25,
            "lateral_displacement_mm": -0.5,
            "net_yaw_change_deg": 0.2,
            "annotated_leg_spikes": 120.0,
            "mapped_motor_spikes": 80.0,
            "articulated_leg_sensory_spikes": 40.0,
        }
        rows = [{"seed": 1729, "condition": condition, **metrics}
                for condition in ("left", "right", "static")]
        contrast = _moving_minus_static(rows, "test")
        for outcome in contrast["aggregate_by_metric"].values():
            self.assertEqual(outcome["mean_moving_minus_static"], 0.0)
            self.assertIsNone(outcome["exact_two_sided_sign_flip_p"])

    def test_foot_contact_geometry_does_not_become_nonfoot_contact(self) -> None:
        body = {
            "body_ground_contact": False,
            "feet": {leg: {"contact": True, "normal_force_N": 1e-6} for leg in LEGS},
            "body_contacts": [{"ground_contact": True, "geom1": "ground",
                               "geom2": f"{leg}_foot", "foot_ground_contact": True}
                              for leg in LEGS],
        }
        feet, count, nonfoot, force = _feet_contacts(body)
        self.assertEqual(count, 6)
        self.assertEqual(nonfoot, 0)
        self.assertEqual(sum(feet.values()), 6)
        self.assertAlmostEqual(force, 6e-6)

    def test_sensory_mode_is_read_from_manifest_contract(self) -> None:
        self.assertEqual(_mode({"sensory_feedback": {"mode": "on"}}), "on")


class ArticulatedSchemaTests(unittest.TestCase):
    def _build_run(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True)
        source = run_dir / "sources" / "scripts" / "source.py"
        source.parent.mkdir(parents=True)
        source.write_text("immutable source snapshot\n", encoding="utf-8")

        annotated_by_leg = dict.fromkeys(LEGS, 1500)
        mapped_by_leg = {leg: (1500 if leg in ("LF", "RF") else 0) for leg in LEGS}
        afferent_by_leg = {leg: (1500 if leg in ("LF", "RF") else 0) for leg in LEGS}
        totals = {"neural_spikes": 15000, "annotated_leg_spikes": 9000,
                  "mapped_motor_spikes": 3000, "articulated_leg_sensory_spikes": 3000,
                  "ticks": 15000,
                  "annotated_leg_spikes_by_leg": annotated_by_leg,
                  "mapped_motor_spikes_by_leg": mapped_by_leg,
                  "articulated_leg_sensory_spikes_by_leg": afferent_by_leg}
        telemetry_path = run_dir / "telemetry.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as handle:
            for index in range(1500):
                feet = {leg: {"position_m": [0.0, 0.0, 0.0001], "contact": True,
                              "normal_force_N": 1e-6} for leg in LEGS}
                row = {
                    "frame_index": index,
                    "frame_time_s": index * 0.005,
                    "timestamp": {"clock": "control", "tick": index,
                                  "rate_hz": 200.0, "t_s": index * 0.005},
                    "stimulus_condition": "stripe_left",
                    "neural": {
                        "tick_count": 10, "total_spikes": 10,
                        "annotated_leg_spikes": 6,
                        "annotated_leg_spikes_by_leg": dict.fromkeys(LEGS, 1),
                        "mapped_motor_spikes": 2,
                        "mapped_motor_spikes_by_leg": {leg: (1 if leg in ("LF", "RF") else 0)
                                                       for leg in LEGS},
                        "articulated_leg_sensory_spikes": 2,
                        "articulated_leg_sensory_spikes_by_leg": {
                            leg: (1 if leg in ("LF", "RF") else 0) for leg in LEGS},
                    },
                    "body": {
                        "root": {"position_m": [(index + 1) * 1e-6, 0.0, 0.001275],
                                 "orientation_rpy_rad": [0.0, 0.0, 0.0],
                                 "yaw_rad": 0.0, "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                        "joints": {"joint": {"q_rad": 0.01 * (index + 1)}},
                        "feet": feet, "body_contacts": [], "body_ground_contact": False,
                    },
                    "joint_delta_rad": {"joint": 0.01 * (index + 1)},
                    "muscles": {
                        "active_force_by_leg_N": dict.fromkeys(LEGS, 0.001),
                        "peak_active_force_by_leg_N": dict.fromkeys(LEGS, 0.002),
                    },
                    "sensory_feedback": {
                        "mode": "on", "source_time_s": index * 0.005,
                        "applied_at_s": index * 0.005,
                        "articulated_leg_encoder_ticks": [
                            {"tick": index * 10 + substep,
                             "source_time_s": (index * 10 + substep) * 0.0005,
                             "applied_at_s": (index * 10 + substep) * 0.0005,
                             "input_applied": True}
                            for substep in range(10)
                        ],
                    },
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

        tick_path = run_dir / "neural_spike_counts.jsonl"
        with tick_path.open("w", encoding="utf-8") as handle:
            for tick in range(15000):
                active = tick % 10 == 0
                row = {
                    "tick": tick, "t_s": tick * 0.0005,
                    "total_spikes": 10 if active else 0,
                    "annotated_leg_spikes": 6 if active else 0,
                    "annotated_leg_spikes_by_leg": dict.fromkeys(LEGS, 1 if active else 0),
                    "mapped_motor_spikes": 2 if active else 0,
                    "mapped_motor_spikes_by_leg": {
                        leg: (1 if active and leg in ("LF", "RF") else 0) for leg in LEGS},
                    "articulated_leg_sensory_spikes": 2 if active else 0,
                    "articulated_leg_sensory_spikes_by_leg": {
                        leg: (1 if active and leg in ("LF", "RF") else 0) for leg in LEGS},
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

        write_json(run_dir / "summary.json", {})
        write_json(run_dir / "performance.json", {
            "simulation_loop_s": 10.0,
            "stage_wall_s": {"neural_s": 5.0, "muscle_physics_s": 4.0},
        })
        write_json(run_dir / "final_body_snapshot.json", {})
        artifact_names = ("summary.json", "performance.json", "telemetry.jsonl",
                          "neural_spike_counts.jsonl", "final_body_snapshot.json")
        artifacts = {name: {"path": name, "sha256": sha256(run_dir / name)}
                     for name in artifact_names}
        manifest = {
            "schema_version": "articulated-optomotor-run-1",
            "seed": 1729,
            "protocol": {"condition": "stripe_left", "stimulus_name": "stripe_panel",
                         "duration_s": 7.5, "flyvis_frames": 1500, "cns_ticks": 15000,
                         "flyvis_dt_s": 0.005, "cns_dt_s": 0.0005,
                         "cns_ticks_per_frame": 10, "neural_gain": 1.2000000476837158,
                         "stimulus_anchor": "world", "synthetic_gait": False,
                         "body_traction_command": False, "direct_motor_input": False,
                         "initial_root_vs_original_camera": {"articulated_minus_original_z_m": 0.000075}},
            "sensory_feedback": {
                "mode": "on", "legacy_groups_applied": ["jo", "haltere", "wing"],
                "legacy_leg_aliases_masked": ["leg", "leg_mechanosensory", "leg_tactile",
                                               "leg_proprioceptive"],
                "articulated_leg_current_applied": True,
                "body_state_source": "current ArticulatedLegBody telemetry at every step",
            },
            "effective_backends": {"renderer": "warp_cuda", "vision": "flyvis",
                                   "vision_device": "cuda:0", "neural": "torch_cuda",
                                   "neural_device": "cuda"},
            "cuda_execution": {
                "renderer": {"requested": "warp_cuda", "effective": "warp_cuda",
                             "device": "cuda:0", "actual_cuda_tensor_seen": True},
                "vision": {"effective": "flyvis", "device": "cuda:0", "frames_stepped": 1500},
                "neural": {"effective": "torch_cuda", "device": "cuda", "ticks_stepped": 15000},
                "verified": True, "fallback_permitted": False, "fallback_used": False,
            },
            "mapping": {"leg_target_overlap": {"dense_neuron_overlap_count": 0}},
            "initial_state": {"root_pose": {"position_m": [0.0, 0.0, 0.001275],
                                               "orientation_rpy_rad": [0.0, 0.0, 0.0],
                                               "yaw_rad": 0.0}},
            "counts": totals,
            "source_sha256": {"scripts/source.py": sha256(source)},
            "source_snapshot_directory": "sources",
            "artifacts": artifacts,
        }
        write_json(run_dir / "manifest.json", manifest)

    def test_actual_articulated_schema_hashes_and_tick_counts_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "seed-1729-left-on"
            self._build_run(run_dir)
            skipped: list[dict] = []
            runs = discover_runs(root, label="articulated", seed=1729, skipped=skipped)
            self.assertEqual(len(runs), 1)
            self.assertEqual(skipped, [])
            result = analyze_trial(runs[0])
            self.assertEqual(result["frames"], 1500)
            self.assertAlmostEqual(result["initial_body_height_mm"], 1.275, places=12)
            self.assertAlmostEqual(result["forward_displacement_mm"], 1.5)
            self.assertEqual(result["annotated_leg_spikes"], 9000)
            self.assertEqual(result["mapped_motor_spikes"], 3000)
            self.assertEqual(result["articulated_leg_sensory_spikes"], 3000)
            self.assertEqual(result["foot_contact_fraction_any"], 1.0)
            self.assertEqual(result["nonfoot_ground_contact_frames"], 0)
            self.assertAlmostEqual(result["mean_total_active_muscle_force_N"], 0.006)
            self.assertEqual(result["wall_timing"]["simulation_wall_s"], 10.0)
            self.assertEqual(len(result["integrity"]["source_snapshot_hashes_verified"]), 1)


if __name__ == "__main__":
    unittest.main()
