"""Toolchain tests for scripts/validate_digital_fly.py.

The complete fixture is intentionally synthetic.  Tests assert that it can
exercise every metric and that the report still refuses to call it biological
validation.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_digital_fly as validator  # noqa: E402


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "flight_validation"


class FlightValidationTest(unittest.TestCase):
    def test_complete_synthetic_fixture_exercises_all_metrics_without_biological_claim(self) -> None:
        report = validator.validate_file(FIXTURES / "synthetic_complete.json")

        self.assertEqual(report["evidence"]["class"], validator.EvidenceClass.SYNTHETIC_FIXTURE)
        self.assertEqual(report["decision"], "NOT_VALIDATED")
        # The historical fixture covers the legacy diagnostics.  The new
        # completion gate correctly leaves forward/yaw scenario evidence
        # SKIP instead of manufacturing a sustained-flight claim from a
        # generic trace.
        self.assertEqual(report["engineering_decision"], "NOT_VALIDATED")
        self.assertEqual(report["metrics"]["forward_displacement"]["mechanical_status"], "SKIP")
        self.assertEqual(report["metrics"]["controlled_yaw"]["mechanical_status"], "SKIP")
        self.assertTrue(all(metric["validation_status"] in {"NOT_VALIDATED", "SKIP"} for metric in report["metrics"].values()))
        self.assertTrue(all(report["metrics"][name]["mechanical_status"] == "PASS" for name in report["diagnostic_metrics"]))

        # A fixture cannot be promoted by a command-line attestation.
        forced = validator.validate_file(FIXTURES / "synthetic_complete.json", evidence_class="real_biological")
        self.assertEqual(forced["decision"], "NOT_VALIDATED")
        self.assertEqual(forced["evidence"]["class"], validator.EvidenceClass.SYNTHETIC_FIXTURE)

    def test_existing_flight_artifact_is_digital_and_not_biological(self) -> None:
        path = Path(__file__).resolve().parents[1] / "artifacts" / "fly-scale" / "closed-loop-g0.3" / "flight.json"
        report = validator.validate_file(path)

        self.assertEqual(report["evidence"]["class"], validator.EvidenceClass.DIGITAL_SIMULATION)
        self.assertEqual(report["decision"], "NOT_VALIDATED")
        self.assertFalse(report["evidence"]["real_biological"])

    def test_csv_loader_and_source_labels(self) -> None:
        trace = validator.load_trace(FIXTURES / "minimal_trace.csv")

        self.assertEqual(trace.evidence_class, validator.EvidenceClass.SYNTHETIC_FIXTURE)
        self.assertAlmostEqual(trace.rows[1]["z_m"], 0.4)
        report = validator.validate_trace(trace)
        value = report["metrics"]["closed_loop_source_ratio"]["value"]
        self.assertAlmostEqual(value["ratio"], 1.0)

    def test_npz_loader_supports_compact_flight_arrays(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is optional for NPZ support")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.npz"
            np.savez(path, traj=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]]), yaw=np.array([0.0, 0.01]), omega=np.array([0.0, 0.1]), frame_ms=np.array(100.0))

            trace = validator.load_trace(path, evidence_class="synthetic_fixture")

        self.assertTrue(trace.time_inferred)
        self.assertAlmostEqual(trace.rows[1]["z_m"], 0.4)
        self.assertAlmostEqual(trace.rows[1]["time_s"], 0.1)

    def test_contract_bundle_flattens_body_visual_motor_and_step_diagnostics(self) -> None:
        row = validator._canonicalise_row({
            "timestamp": {"t_s": 0.0, "tick": 0, "clock": "control", "rate_hz": 200.0},
            "body": {
                "position_m": [0.0, 0.0, 0.4],
                "orientation_rad": [0.01, 0.02, 0.03],
                "velocity_mps": [0.0, 0.0, 0.0],
                "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                "source": "handbuilt",
            },
            "visual": {"optic_flow_rad_s": [0.0, 0.0, 1.0]},
            "motor": {"yaw_rate_rad_s": 1.0, "source": "fitted"},
            "diagnostics": {"step_index": 0, "source": "handbuilt"},
        })
        self.assertAlmostEqual(row["time_s"], 0.0)
        self.assertAlmostEqual(row["z_m"], 0.4)
        self.assertAlmostEqual(row["visual_direction"], 1.0)
        self.assertAlmostEqual(row["steering_response"], 1.0)
        self.assertEqual(row["source"], "fitted")

    def test_no_input_is_an_honest_skip(self) -> None:
        report = validator.validate_paths([])

        self.assertEqual(report["decision"], "NOT_VALIDATED")
        self.assertEqual(report["engineering_decision"], "NOT_VALIDATED")
        self.assertTrue(all(metric["status"] == "SKIP" for metric in report["metrics"].values()))

    @staticmethod
    def _completion_rows() -> list[dict[str, object]]:
        """Small compact multi-scenario trace, independent of runner internals."""

        rows: list[dict[str, object]] = []

        def add(scenario: str, samples: list[tuple[float, float, float, float, float, dict[str, object]]]) -> None:
            for time, x, z, yaw, speed, extra in samples:
                row: dict[str, object] = {
                    "scenario": scenario,
                    "time_s": time,
                    "position_m": [x, 0.0, z],
                    "velocity_mps": [speed, 0.0, 0.0],
                    "orientation_rad": [0.0, 0.0, yaw],
                    "yaw_rate_rad_s": 0.4 if scenario == "yaw-step" else 0.0,
                "speed_mps": speed,
                "collision_count": 0,
                "source": "fitted",
                }
                row.update(extra)
                rows.append(row)

        add("takeoff", [
            (0.0, 0.0, 0.015, 0.0, 0.1, {}),
            (0.2, 0.0, 0.35, 0.0, 0.5, {}),
            (0.5, 0.0, 0.40, 0.0, 0.3, {}),
        ])
        add("hover", [
            (0.0, 0.0, 0.40, 0.0, 0.1, {}),
            (0.5, 0.01, 0.40, 0.0, 0.1, {}),
            (1.1, 0.0, 0.40, 0.0, 0.1, {}),
        ])
        add("forward", [
            (0.0, 0.0, 0.40, 0.0, 0.5, {}),
            (0.5, 0.25, 0.40, 0.0, 0.5, {}),
            (1.0, 0.50, 0.40, 0.0, 0.5, {}),
        ])
        add("yaw-step", [
            (0.0, 0.0, 0.40, 0.0, 0.1, {"yaw_command": 0.5}),
            (0.5, 0.0, 0.40, 0.25, 0.1, {"yaw_command": 0.5}),
            (1.0, 0.0, 0.40, 0.40, 0.1, {"yaw_command": 0.5}),
        ])
        add("gust-recovery", [
            (0.0, 0.0, 0.40, 0.0, 0.1, {}),
            (0.5, 0.0, 0.40, 0.0, 0.5, {"gust_active": True}),
            (0.6, 0.0, 0.40, 0.0, 0.4, {"gust_active": True}),
            (0.9, 0.0, 0.40, 0.0, 0.1, {}),
            (1.5, 0.0, 0.40, 0.0, 0.1, {}),
        ])
        add("landing", [
            (0.0, 0.0, 0.40, 0.0, 0.1, {}),
            (0.3, 0.0, 0.10, 0.0, 0.1, {}),
            (0.4, 0.0, 0.00, 0.0, 0.1, {
                "contact": True, "impact_accel_mps2": 10.0,
                "target_distance_m": 0.05, "target_on_target": True,
            }),
        ])
        return rows

    def _write_completion_trace(self, directory: str, *, evidence_class: str = "digital_simulation", include_forward: bool = True) -> Path:
        rows = self._completion_rows()
        if not include_forward:
            rows = [row for row in rows if row["scenario"] != "forward"]
        path = Path(directory) / "multi_scenario.json"
        path.write_text(json.dumps({"metadata": {
            "evidence_class": evidence_class,
            # A complete engineering artifact must carry an auditable budget
            # pair; the validator must not infer resource success from its
            # absence.
            "resource_limits": {"cgroup_memory_max_bytes": 5 * 1024 ** 3, "torch_allocator_max_vram_gib": 15.5},
            "resource_usage": {"peak_host_rss_bytes": 1024 ** 3, "cuda": {"peak_reserved_bytes": 2 * 1024 ** 3}},
        }, "trajectory": rows}), encoding="utf-8")
        return path

    def test_multi_scenario_runner_trajectory_has_engineering_go_but_not_biological_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validator.validate_file(self._write_completion_trace(directory))

        self.assertEqual(report["scenario_names"], ["takeoff", "hover", "forward", "yaw-step", "gust-recovery", "landing"])
        self.assertEqual(report["engineering_decision"], "GO")
        self.assertEqual(report["mechanical_decision"], "GO")
        self.assertEqual(report["decision"], "NOT_VALIDATED")
        self.assertEqual(report["validation_status"], "NOT_VALIDATED")
        self.assertEqual(set(report["engineering"]["skipped_metrics"]), set())
        self.assertEqual(report["metrics"]["forward_displacement"]["value"]["displacement_m"], 0.5)
        self.assertAlmostEqual(report["metrics"]["controlled_yaw"]["value"]["yaw_change_rad"], 0.4)

    def test_missing_forward_scenario_is_skip_and_cannot_claim_engineering_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validator.validate_file(self._write_completion_trace(directory, include_forward=False))

        self.assertEqual(report["engineering_decision"], "NOT_VALIDATED")
        self.assertEqual(report["metrics"]["forward_displacement"]["mechanical_status"], "SKIP")
        self.assertIn("forward_displacement", report["engineering"]["skipped_metrics"])

    def test_explicit_real_data_can_change_biological_status_but_fixture_cannot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validator.validate_file(self._write_completion_trace(directory, evidence_class="real_biological"))

        self.assertEqual(report["evidence"]["class"], validator.EvidenceClass.REAL_BIOLOGICAL)
        self.assertEqual(report["engineering_decision"], "GO")
        self.assertEqual(report["decision"], "GO")
        self.assertEqual(report["validation_status"], "PASS")

    def test_multi_scenario_bad_takeoff_does_not_get_hidden_by_other_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = self._completion_rows()
            for row in rows:
                if row["scenario"] == "takeoff":
                    row["position_m"][2] = 0.10  # type: ignore[index]
            path = Path(directory) / "bad_takeoff.json"
            path.write_text(json.dumps({"metadata": {"evidence_class": "digital_simulation"}, "trajectory": rows}), encoding="utf-8")
            report = validator.validate_file(path)

        self.assertEqual(report["engineering_decision"], "NO-GO")
        self.assertEqual(report["metrics"]["takeoff_success"]["mechanical_status"], "FAIL")
        self.assertEqual(report["metrics"]["forward_displacement"]["mechanical_status"], "PASS")

    def test_bounded_speed_and_yaw_polarity_are_real_engineering_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = self._completion_rows()
            for row in rows:
                if row["scenario"] == "forward":
                    row["speed_mps"] = 6.0
                if row["scenario"] == "yaw-step":
                    row["yaw_command"] = -0.5
            path = Path(directory) / "bad_bounds.json"
            path.write_text(json.dumps({"metadata": {"evidence_class": "digital_simulation"}, "trajectory": rows}), encoding="utf-8")
            report = validator.validate_file(path)

        self.assertEqual(report["engineering_decision"], "NO-GO")
        self.assertEqual(report["metrics"]["bounded_attitude_speed"]["mechanical_status"], "FAIL")
        self.assertEqual(report["metrics"]["controlled_yaw"]["mechanical_status"], "FAIL")

    def test_landing_without_precontact_is_not_called_a_gentle_landing(self) -> None:
        rows = [{
            "scenario": "landing",
            "time_s": 0.0,
            "position_m": [0.0, 0.0, 0.0],
            "velocity_mps": [0.0, 0.0, -0.1],
            "contact": True,
            "impact_accel_mps2": 10.0,
        }]
        trace = validator.Trace(rows=rows, metadata={}, evidence_class=validator.EvidenceClass.DIGITAL_SIMULATION)
        metric = validator.metric_landing(trace, validator.ValidationThresholds())
        self.assertEqual(metric["mechanical_status"], "SKIP")

    def test_explicit_false_contact_at_liftoff_is_not_early_landing(self) -> None:
        rows = [
            {
                "scenario": "takeoff", "time_s": 0.0,
                "position_m": [0.0, 0.0, 0.015], "velocity_mps": [0.0, 0.0, 0.0],
                "contact": True, "ground_distance_m": 0.0, "collision_count": 0,
            },
            {
                "scenario": "takeoff", "time_s": 0.15,
                "position_m": [0.0, 0.0, 0.015], "velocity_mps": [0.0, 0.0, 0.01],
                "contact": False, "ground_distance_m": 0.0, "collision_count": 0,
            },
            {
                "scenario": "approach", "time_s": 8.0,
                "position_m": [0.85, 0.0, 0.10], "velocity_mps": [0.0, 0.0, -0.10],
                "vz_mps": -0.10,
                "contact": False, "target_contact": False,
                "target_distance_m": 0.05, "collision_count": 0,
            },
            {
                "scenario": "approach", "time_s": 8.5,
                "position_m": [0.85, 0.0, 0.055], "velocity_mps": [0.0, 0.0, 0.0],
                "vz_mps": 0.0,
                "contact": True, "target_contact": True, "target_on_target": True,
                "target_distance_m": 0.02, "impact_accel_mps2": 10.0,
                "collision_count": 0,
            },
        ]
        trace = validator.Trace(
            rows=rows,
            metadata={},
            evidence_class=validator.EvidenceClass.DIGITAL_SIMULATION,
        )
        metric = validator.metric_landing(trace, validator.ValidationThresholds())
        self.assertEqual(metric["mechanical_status"], "PASS")

    def test_keyed_scenario_mapping_is_flattened_without_losing_labels(self) -> None:
        rows = self._completion_rows()
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(str(row["scenario"]), []).append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keyed.json"
            path.write_text(json.dumps({"metadata": {"evidence_class": "digital_simulation"}, "trajectory": grouped}), encoding="utf-8")
            trace = validator.load_trace(path)

        self.assertEqual(set(trace.scenarios), {"takeoff", "hover", "forward", "yaw-step", "gust-recovery", "landing"})
        self.assertEqual(len(trace.rows), len(rows))

    def test_mission_phase_labels_are_used_when_one_runner_scenario_contains_all_phases(self) -> None:
        rows = self._completion_rows()
        for row in rows:
            row["mission_phase"] = row["scenario"]
            row["scenario"] = "full-mission"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mission.json"
            path.write_text(json.dumps({"metadata": {
                "evidence_class": "digital_simulation",
                "resource_limits": {"cgroup_memory_max_bytes": 5 * 1024 ** 3, "torch_allocator_max_vram_gib": 15.5},
                "resource_usage": {"peak_host_rss_bytes": 1024 ** 3, "cuda": {"peak_reserved_bytes": 2 * 1024 ** 3}},
            }, "trajectory": rows}), encoding="utf-8")
            report = validator.validate_file(path)

        self.assertEqual(report["engineering_decision"], "GO")
        self.assertEqual(report["metrics"]["forward_displacement"]["mechanical_status"], "PASS")
        self.assertEqual(report["metrics"]["controlled_yaw"]["mechanical_status"], "PASS")

    def test_nonfinite_or_missing_collision_data_cannot_be_a_mechanical_pass(self) -> None:
        rows = self._completion_rows()
        rows[0]["speed_mps"] = float("nan")
        rows[1].pop("collision_count")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_completion_trace(directory)
            # Rewrite through JSON so the loader exercises its non-finite
            # parser path as an artifact would.
            payload = json.loads(path.read_text())
            payload["trajectory"] = rows
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = validator.validate_file(path)

        self.assertEqual(report["metrics"]["finite_values"]["mechanical_status"], "FAIL")
        self.assertEqual(report["metrics"]["collision_free"]["mechanical_status"], "SKIP")
        self.assertEqual(report["engineering_decision"], "NO-GO")
        self.assertEqual(report["decision"], "NOT_VALIDATED")

    def test_resource_peak_over_declared_budget_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_completion_trace(directory)
            payload = json.loads(path.read_text())
            payload["metadata"]["resource_usage"]["cuda"]["peak_reserved_bytes"] = 16 * 1024 ** 3
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = validator.validate_file(path)

        self.assertEqual(report["metrics"]["resource_limits"]["mechanical_status"], "FAIL")
        self.assertEqual(report["engineering_decision"], "NO-GO")

    def test_manifest_and_missing_trajectory_bundle_inputs_are_compatible_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = self._write_completion_trace(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "digital-fly-run-1",
                "outputs": {"trajectory": trajectory.name},
                "provenance": {"evidence_class": "digital_simulation"},
            }), encoding="utf-8")
            report = validator.validate_file(manifest)
            self.assertEqual(report["evidence"]["class"], validator.EvidenceClass.DIGITAL_SIMULATION)
            self.assertEqual(report["evidence"]["n_samples"], len(self._completion_rows()))

            missing_manifest = root / "missing-manifest.json"
            missing_manifest.write_text(json.dumps({
                "schema_version": "digital-fly-run-1",
                "outputs": {"trajectory": "missing-trajectory.json"},
                "provenance": {"evidence_class": "digital_simulation"},
            }), encoding="utf-8")
            telemetry = root / "telemetry.jsonl"
            telemetry.write_text("{\"time_s\": 0.0}\n", encoding="utf-8")
            # The sidecar is named manifest.json for the telemetry preflight.
            (root / "manifest.json").write_text(missing_manifest.read_text(), encoding="utf-8")
            missing = validator.validate_file(telemetry)

        self.assertEqual(missing["evidence"]["n_samples"], 0)
        self.assertEqual(missing["decision"], "NOT_VALIDATED")
        self.assertTrue(any("missing" in issue for issue in missing["evidence"]["integrity_issues"]))


if __name__ == "__main__":
    unittest.main()
