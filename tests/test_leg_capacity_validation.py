"""Focused regression tests for per-tick single-leg capacity acceptance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_leg_capacity import validate_trace


DT = 0.0005
TONIC = "LF_trochanter_flexor"
EXTENSOR = "LF_trochanter_extensor"
RAMP = {
    "shape": "linear_rise_then_hold",
    "start_rate_hz": 0.0,
    "target_rate_hz": 30.0,
    "rise_duration_s": 0.05,
    "time_basis": "phase_time_s",
}


def _trace_case(directory: Path, *, drive_ticks: int, gap_ticks: set[int] | None = None,
                rh_dropout_ticks: set[int] | None = None,
                forged_rate_tick: int | None = None) -> tuple[Path, Path]:
    """Write a minimal, clocked candidate trace and matching schedule report."""
    gap_ticks = gap_ticks or set()
    rh_dropout_ticks = rh_dropout_ticks or set()
    rows = []
    phases = (("pre_settle", 2), ("imposed_drive", drive_ticks), ("release", 40))
    global_tick = 0
    for phase, count in phases:
        for phase_tick in range(count):
            phase_time = phase_tick * DT
            rates = {TONIC: 5.0}
            if phase == "imposed_drive":
                extensor_rate = 30.0 * min(phase_time / 0.05, 1.0)
                if extensor_rate != 0.0:
                    rates[EXTENSOR] = extensor_rate
                if forged_rate_tick == phase_tick:
                    rates[EXTENSOR] += 0.25

            feet = {}
            for leg in ("LF", "LM", "LH", "RF", "RM", "RH"):
                if leg == "LF" and phase == "imposed_drive" and phase_tick in gap_ticks:
                    feet[leg] = {
                        "contact": False,
                        "normal_force_N": 0.0,
                        "clearance_m": 100e-6,
                        "center_z_m": 175e-6,
                    }
                elif leg == "RH" and phase == "imposed_drive" and phase_tick in rh_dropout_ticks:
                    feet[leg] = {
                        "contact": False, "normal_force_N": 0.0,
                        "clearance_m": 0.0, "center_z_m": 75e-6,
                    }
                else:
                    feet[leg] = {
                        "contact": True, "normal_force_N": 1e-3,
                        "clearance_m": 0.0, "center_z_m": 75e-6,
                    }

            rows.append({
                "schema": "leg-capacity-trace-sample-2",
                "case_id": "candidate",
                "role": "candidate",
                "leg_key": "LF",
                "phase": phase,
                "tick_in_phase": phase_tick,
                "time_s": (global_tick + 1) * DT,
                "phase_time_s": phase_time,
                "state_phase_time_s": (phase_tick + 1) * DT,
                "prescribed_rates_hz_by_channel": rates,
                "feet": feet,
                "root_position_m": [0.0, 0.0, 1e-3],
                "body_height_m": 1e-3,
                "root_applied_wrench": {
                    "available": True,
                    "force_N": [0.0, 0.0, 0.0],
                    "torque_Nm": [0.0, 0.0, 0.0],
                    "max_abs_generalized_force": 0.0,
                },
                "nonfoot_ground_contact": False,
            })
            global_tick += 1

    trace_path = directory / "trajectory.jsonl"
    trace_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    schedule = {
        TONIC: {"pre_settle": 5.0, "imposed_drive": 5.0, "release": 5.0},
        EXTENSOR: {"imposed_drive": RAMP},
    }
    report = {
        "schema": "leg-active-support-tonic-probe-1",
        "candidate_case_id": "candidate",
        "trajectory": {"sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()},
        "cases": [{
            "case_id": "candidate", "role": "candidate", "active_drive": True,
            "leg_key": "LF",
            "phase_seconds": {
                "pre_settle": 2 * DT,
                "imposed_drive": drive_ticks * DT,
                "release": 40 * DT,
            },
            "rate_schedule_by_channel": schedule,
        }],
    }
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return trace_path, report_path


class LegCapacityValidationTests(unittest.TestCase):
    def _validate(self, **trace_options):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        trace_path, report_path = _trace_case(Path(temp.name), **trace_options)
        result = validate_trace(trace_path, report_path=report_path)
        return result["cases"][0]

    def test_single_peak_does_not_pass_sustained_clearance(self):
        case = self._validate(drive_ticks=40, gap_ticks={10})
        metrics = case["metrics"]
        self.assertGreater(metrics["selected_leg_drive_max_gap_m"], 25e-6)
        self.assertEqual(metrics["selected_leg_longest_contiguous_gap_s"], DT)
        self.assertFalse(case["checks"]["LF_clearance_at_least_threshold_for_contiguous_duration"])
        self.assertFalse(case["accepted"])

    def test_one_other_foot_dropout_rejects_an_otherwise_qualifying_lift(self):
        # RH drops before the clearance interval: the gap qualifies, while the
        # unchanged all-drive support gate must still reject this one bad tick.
        case = self._validate(
            drive_ticks=50, gap_ticks=set(range(5, 50)), rh_dropout_ticks={0}
        )
        self.assertTrue(case["checks"]["LF_clearance_at_least_threshold_for_contiguous_duration"])
        self.assertFalse(case["checks"]["other_five_load_bearing_through_drive"])
        self.assertEqual(case["metrics"]["other_five_drive_support_failure_samples_by_leg"]["RH"], 1)
        self.assertFalse(case["accepted"])

    def test_forged_ramp_sample_fails_per_tick_schedule(self):
        case = self._validate(
            drive_ticks=100, gap_ticks=set(range(100)), forged_rate_tick=40
        )
        self.assertEqual(case["metrics"]["rate_rows"], case["metrics"]["rows"])
        self.assertEqual(case["metrics"]["rate_mismatch_rows"], 1)
        self.assertFalse(case["checks"]["prescribed_rates_recorded_and_match"])


if __name__ == "__main__":
    unittest.main()
