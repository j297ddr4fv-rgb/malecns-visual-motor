#!/usr/bin/env python3
"""CPU fixtures for the strict digital escape acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_escape_experiment as validator  # noqa: E402


def _manifest(*, stimulus_x: float = -1.0) -> dict[str, object]:
    return {
        "schema_version": "escape-experiment-1",
        "seed": 1729,
        "evidence_class": "digital_simulation",
        "authority": {
            "phase": "none",
            "goal": "none",
            "state_drive": "none",
            "stabilizer": "none",
            "autopilot": "none",
        },
        "stimulus": {
            "id": "loom-left",
            "onset_s": 1.0,
            "position_m": [stimulus_x, 0.0, 0.0],
            "size_deg": 20.0,
            "visual_only": True,
            "non_interacting": True,
            "collidable": False,
        },
        "body_radius_m": 0.0012,
        "protocol": {
            "decision_deadline_s": 1.970,
            "expected_surface_contact_time_s": 1.9776,
            "duration_s": 2.5,
        },
        "stimulus_routing": {
            "metadata_to_sensory": False,
            "metadata_to_control": False,
        },
        "ground_plane_z_m": 0.0,
        "body_state_authority": {
            "source": "integrator",
            "direct_pose_write": False,
            "teleport": False,
            "pose_write_events": [],
            "teleport_events": [],
        },
        "acceptance": {
            "neural_response_delta": 1.0,
            "motor_response_delta": 1.0,
            "motion_displacement_m": 0.01,
            "liftoff_height_m": 0.01,
            "escape_projection_m": 0.05,
            "control_effect_projection_m": 0.05,
            "max_global_active_fraction": 0.50,
            "max_global_saturation_fraction": 0.50,
            "max_integration_accel_mps2": 100.0,
            "integration_tolerance_m": 0.01,
        },
    }


def _rows(
    *,
    direction: float = 1.0,
    response: bool = True,
    saturated: bool = False,
    teleport: bool = False,
    metadata_leak: bool = False,
) -> list[dict[str, object]]:
    # The first neural response is at t=1, motor output at t=1.25, and body
    # motion/liftoff at t=1.5.  Liftoff is intentionally before the protocol
    # decision deadline, while the trace continues past planned contact.
    if response:
        positions = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.20 * direction, 0.12), (0.35 * direction, 0.16), (0.35 * direction, 0.16)]
        active = [0, 0, 8, 8, 9, 9, 9]
        motor = [0, 0, 0, 4, 4, 3, 3]
        contact = [True, True, True, True, False, False, False]
    else:
        positions = [(0.0, 0.0)] * 7
        active = [0, 0, 0, 0, 0, 0, 0]
        motor = [0, 0, 0, 0, 0, 0, 0]
        contact = [True] * 7

    rows: list[dict[str, object]] = []
    times = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5]
    for index, ((x, z), active_count, motor_value, on_ground) in enumerate(
        zip(positions, active, motor, contact)
    ):
        if index == 0:
            velocity = [0.0, 0.0, 0.0]
        else:
            previous_x, previous_z = positions[index - 1]
            velocity = [x - previous_x, 0.0, z - previous_z]
        neural: dict[str, object] = {
            "active_count": active_count,
            "neuron_count": 100,
            "saturation_fraction": 0.08 if saturated else 0.02,
            "global_saturation": saturated,
        }
        sensory: dict[str, object] = {
            "optic_flow_rad_s": 0.5 if index >= 1 else 0.0,
        }
        if metadata_leak:
            sensory["stimulus_id"] = "loom-left"
        row: dict[str, object] = {
            "time_s": times[index],
            "position_m": [x, 0.0, z],
            "velocity_mps": velocity,
            "contact": on_ground,
            "collision_count": 0,
            "stimulus_active": index >= 1,
            "sensory": sensory,
            "neural": neural,
            "motor": {
                "escape_command": motor_value,
                "source": "neural",
            },
        }
        if teleport and index == 4:
            row["teleport"] = True
        rows.append(row)
    return rows


def _bundle_payload(rows: list[dict[str, object]], manifest: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "escape-trajectory-1",
        "metadata": {"evidence_class": "digital_simulation"},
        "trajectory": rows,
        "manifest": manifest,
    }


class EscapeValidatorTest(unittest.TestCase):
    def test_complete_candidate_passes_three_engineering_levels_but_not_biology(self) -> None:
        manifest = _manifest()
        report = validator.validate_escape_experiment(_bundle_payload(_rows(), manifest), manifest)

        self.assertEqual(report["decision"], "NOT_VALIDATED")
        self.assertFalse(report["biological_validation"])
        self.assertEqual(report["status"], "FAIL")  # CAUSAL_ESCAPE lacks controls.
        self.assertEqual(report["evidence_levels"]["NEURAL_RESPONSE"], "PASS")
        self.assertEqual(report["evidence_levels"]["MOTOR_TRANSFER"], "PASS")
        self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "PASS")
        self.assertEqual(report["evidence_levels"]["CAUSAL_ESCAPE"], "FAIL")
        self.assertTrue(report["evidence"]["single_success_is_not_biological_validation"])

    def test_vertical_jump_can_pass_initial_physical_gate_without_horizontal_flight(self) -> None:
        manifest = _manifest()
        report = validator.validate_escape_experiment(
            _bundle_payload(_rows(direction=0.0), manifest), manifest
        )

        self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "PASS")
        self.assertTrue(report["ground_escape"]["initial_jump_avoidance"]["passed"])
        self.assertEqual(report["ground_escape"]["horizontal_away_flight"]["status"], "FAIL")
        self.assertFalse(report["ground_escape"]["horizontal_away_flight"]["passed"])

    def test_late_liftoff_fails_without_collision_event(self) -> None:
        manifest = _manifest()
        rows = _rows()
        # Keep the body grounded through t=1.5 and let it leave the ground at
        # t=2.0.  There is deliberately no collision row: the configured
        # deadline itself must reject the late response.
        rows[4]["position_m"] = [0.0, 0.0, 0.0]
        rows[4]["velocity_mps"] = [0.0, 0.0, 0.0]
        rows[4]["contact"] = True
        rows[5]["velocity_mps"] = [0.35, 0.0, 0.16]
        report = validator.validate_escape_experiment(_bundle_payload(rows, manifest), manifest)

        physical = report["ground_escape"]
        self.assertIsNone(physical["collision_index"])
        self.assertEqual(physical["liftoff_time_s"], 2.0)
        self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "FAIL")
        self.assertTrue(any("decision deadline" in reason for reason in physical["reasons"]))

    def test_visual_only_counterfactual_contact_does_not_require_physics_collision_channel(self) -> None:
        manifest = _manifest()
        rows = _rows()
        for row in rows:
            row.pop("collision_count")
        report = validator.validate_escape_experiment(_bundle_payload(rows, manifest), manifest)

        physical = report["ground_escape"]
        self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "PASS")
        self.assertTrue(physical["counterfactual_contact"]["trace_reached"])
        self.assertTrue(physical["counterfactual_contact"]["overlap_is_counterfactual"])
        self.assertFalse(physical["counterfactual_contact"]["is_physics_collision"])
        self.assertFalse(physical["physics_collision"]["observed"])
        self.assertFalse(physical["physics_collision"]["channel_present"])
        self.assertFalse(physical["physics_collision"]["channel_required"])
        self.assertIsNone(physical["collision_free"])

    def test_full_configured_duration_is_required_beyond_counterfactual_contact(self) -> None:
        manifest = _manifest()
        manifest["protocol"]["duration_s"] = 3.0
        report = validator.validate_escape_experiment(_bundle_payload(_rows(), manifest), manifest)

        physical = report["ground_escape"]
        self.assertEqual(physical["configured_duration_s"], 3.0)
        self.assertEqual(physical["trace_end_time_s"], 2.5)
        self.assertTrue(physical["counterfactual_contact"]["trace_reached"])
        self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "FAIL")
        self.assertTrue(any("full duration" in reason for reason in physical["reasons"]))

    def test_actual_physics_collision_is_reported_separately_from_counterfactual_contact(self) -> None:
        manifest = _manifest()
        rows = _rows()
        rows[4]["collision_count"] = 1
        report = validator.validate_escape_experiment(_bundle_payload(rows, manifest), manifest)

        physical = report["ground_escape"]
        self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "FAIL")
        self.assertTrue(physical["physics_collision"]["observed"])
        self.assertTrue(physical["physics_collision"]["channel_present"])
        self.assertTrue(physical["counterfactual_contact"]["overlap_is_counterfactual"])
        self.assertFalse(physical["counterfactual_contact"]["is_physics_collision"])

    def test_geometry_derives_core_first_contact_deadline(self) -> None:
        manifest = _manifest()
        manifest.pop("protocol")
        manifest["decision_deadline_s"] = 1.970
        manifest["duration_s"] = 2.5
        manifest["stimulus"].update({
            "start_position_m": [1.0, 0.0, 0.0],
            "velocity_mps": [-0.5, 0.0, 0.0],
            "size_m": [0.09, 0.09, 0.02],
        })
        report = validator.validate_escape_experiment(_bundle_payload(_rows(), manifest), manifest)

        physical = report["ground_escape"]
        self.assertAlmostEqual(physical["planned_contact_time_s"], 1.9776, places=6)
        self.assertEqual(physical["planned_contact_time_source"], "derived:stimulus_geometry")
        self.assertEqual(physical["decision_deadline_s"], 1.970)
        self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "PASS")

        thresholds = validator.EscapeThresholds.from_sources(override={
            "latest_decision_ms": 1250,
            "first_contact_time_ms": 2000,
        })
        self.assertEqual(thresholds.decision_deadline_s, 1.25)
        self.assertEqual(thresholds.planned_contact_time_s, 2.0)

    def test_same_seed_sham_mirror_and_lesion_batches_establish_digital_causal_escape(self) -> None:
        candidate_manifest = _manifest(stimulus_x=-1.0)
        mirror_manifest = _manifest(stimulus_x=1.0)
        controls = {
            "sham": {"trajectory": _rows(response=False)},
            "mirror": {"trajectory": _rows(direction=-1.0), "manifest": mirror_manifest},
            "lesion": {"trajectory": _rows(response=False)},
        }
        report = validator.validate_escape_experiment(
            _bundle_payload(_rows(), candidate_manifest),
            candidate_manifest,
            controls=controls,
        )

        self.assertEqual(report["evidence_levels"]["CAUSAL_ESCAPE"], "PASS")
        self.assertTrue(report["controls"]["comparability"]["structural_comparability"])
        self.assertTrue(report["controls"]["comparability"]["effects"]["mirror_reversal"])
        self.assertEqual(report["decision"], "NOT_VALIDATED")
        self.assertFalse(report["causal_claim"])

    def test_authority_metadata_leak_teleport_and_global_saturation_fail_closed(self) -> None:
        for mutation, expected in (
            (lambda m, r: m["authority"].update({"autopilot": "active"}), "behavior-authority audit failed"),
            (lambda m, r: r[4].update({"teleport": True}), "direct pose write/teleport event observed"),
            (lambda m, r: r[2]["sensory"].update({"stimulus_id": "loom-left"}), "stimulus metadata lineage audit failed"),
            (lambda m, r: r[2]["neural"].update({"global_saturation": True}), "global saturation"),
        ):
            manifest = _manifest()
            rows = _rows()
            mutation(manifest, rows)
            report = validator.validate_escape_experiment(_bundle_payload(rows, manifest), manifest)
            flattened = "\n".join(
                reason
                for category in report["categories"].values()
                for reason in category["reasons"]
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertIn(expected, flattened)

    def test_missing_timing_network_authority_and_out_of_order_input_do_not_pass(self) -> None:
        manifest = _manifest()
        rows = _rows()
        del manifest["authority"]
        del rows[2]["neural"]
        rows[3]["time_s"] = 1.5
        report = validator.validate_escape_experiment(_bundle_payload(rows, manifest), manifest)

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["evidence_levels"]["NEURAL_RESPONSE"], "FAIL")
        self.assertEqual(report["evidence_levels"]["CAUSAL_ESCAPE"], "FAIL")
        self.assertTrue(any("strictly increasing" in reason for reason in report["temporal"]["reasons"]))

    def test_json_and_jsonl_loader_uses_manifest_outputs_and_telemetry(self) -> None:
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "manifest.json").write_text(
                json.dumps({**manifest, "outputs": {"trajectory": "trajectory.json", "telemetry": "telemetry.jsonl"}}),
                encoding="utf-8",
            )
            payload = _bundle_payload(_rows(), manifest)
            (directory / "trajectory.json").write_text(json.dumps(payload), encoding="utf-8")
            (directory / "telemetry.jsonl").write_text(
                "\n".join(json.dumps(row) for row in _rows()) + "\n", encoding="utf-8"
            )
            bundle = validator.load_bundle(directory)
            self.assertEqual(len(bundle.rows), 7)
            self.assertEqual(len(bundle.telemetry_rows), 7)
            report = validator.validate_escape_experiment(directory)
            self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "PASS")

    def test_trajectory_telemetry_mismatch_is_not_silently_accepted(self) -> None:
        manifest = _manifest()
        payload = _bundle_payload(_rows(), manifest)
        telemetry = _rows()
        telemetry[2]["time_s"] = 2.5
        bundle = validator.load_bundle(payload, manifest, telemetry)
        report = validator.validate_escape_experiment(bundle)

        self.assertFalse(report["telemetry"]["passed"])
        self.assertTrue(any("timestamps" in reason for reason in report["telemetry"]["reasons"]))
        self.assertEqual(report["evidence_levels"]["PHYSICAL_ESCAPE"], "FAIL")


if __name__ == "__main__":
    unittest.main()
