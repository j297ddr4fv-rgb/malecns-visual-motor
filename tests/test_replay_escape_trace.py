#!/usr/bin/env python3
"""Tests for the offline sparse-trace escape replay bridge."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from neuromuscular_adapter import MappingPopulationIndex, NeuromuscularAdapter  # noqa: E402
from replay_escape_trace import (  # noqa: E402
    EscapeTraceReplayer,
    ReplayConfig,
    ReplayError,
    SparseSpikeTrace,
    aggregate_population_windows,
    neural_output_from_window,
    obb_collision_clearance,
    write_replay_artifact,
)


class ReplayEscapeTraceTests(unittest.TestCase):
    def _adapter(self, directory: Path) -> NeuromuscularAdapter:
        mapping = {
            "defaults": {"tick_ms": 5.0},
            "calibration": {"neural_window_ms": 5.0, "id": "test"},
            "channels": {
                "wing_motor_left": {
                    "terms": [{"spec": "WL", "r_max_hz": 100.0}],
                    "tau_ms": 0.0,
                },
                "wing_motor_right": {
                    "terms": [{"spec": "WR", "r_max_hz": 100.0}],
                    "tau_ms": 0.0,
                },
                "leg_power": {
                    "terms": [{"spec": "LEG", "r_max_hz": 100.0}],
                    "tau_ms": 0.0,
                },
                "ttmn_activity": {
                    "terms": [{"spec": "TTMn", "r_max_hz": 100.0}],
                    "tau_ms": 0.0,
                },
                "jump_motor": {
                    "kind": "event",
                    "terms": [{"spec": "TTMn", "r_max_hz": 1.0}],
                },
                "jump": {
                    "kind": "event",
                    "terms": [{"spec": "DNp01", "r_max_hz": 1.0}],
                },
            },
        }
        map_path = directory / "map.json"
        map_path.write_text(json.dumps(mapping), encoding="utf-8")
        return NeuromuscularAdapter(
            map_path=map_path,
            population_index=MappingPopulationIndex({
                "WL": [0], "WR": [1], "LEG": [4], "TTMn": [2], "DNp01": [3],
            }),
            strict=True,
        )

    @staticmethod
    def _trace() -> SparseSpikeTrace:
        # Each sparse row is one recorded 1 ms frame ending at that time.  The
        # event at t=5 belongs to the first [0,5] 5 ms window.
        times = np.arange(1.0, 11.0, dtype=np.float64)
        rows = [
            [0], [], [1], [], [2, 3],
            [0], [1], [4], [2], [],
        ]
        offsets = [0]
        neurons: list[int] = []
        for row in rows:
            neurons.extend(row)
            offsets.append(len(neurons))
        return SparseSpikeTrace(times, np.asarray(offsets), np.asarray(neurons, dtype=np.int32))

    def test_exact_five_ms_population_rates_and_neural_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            adapter = self._adapter(Path(temp))
            windows = aggregate_population_windows(self._trace(), adapter, selectors=("WL", "TTMn"))
            self.assertEqual(len(windows), 2)
            self.assertEqual(windows[0].dt_ms, 5.0)
            self.assertEqual(windows[0].spike_counts["WL"], 1)
            self.assertEqual(windows[0].spike_counts["TTMn"], 1)
            self.assertEqual(windows[1].spike_counts["WL"], 1)
            self.assertEqual(windows[1].spike_counts["TTMn"], 1)
            self.assertAlmostEqual(windows[0].rates_hz["WL"], 200.0)
            output = neural_output_from_window(windows[0])
            self.assertAlmostEqual(output.timestamp.t_s, 0.005)
            self.assertAlmostEqual(output.timestamp.dt_s, 0.005)
            self.assertEqual(output.named_outputs["population_spike_counts"]["TTMn"], 1)

    def test_obb_metric_is_not_sphere_or_aabb(self) -> None:
        separated = obb_collision_clearance(
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (1.1, 0.0, 0.0), (0.5, 0.5, 0.5),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertFalse(separated.would_contact)
        self.assertAlmostEqual(separated.clearance_m, 0.1)
        overlap = obb_collision_clearance(
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.9, 0.0, 0.0), (0.5, 0.5, 0.5),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertTrue(overlap.would_contact)
        self.assertAlmostEqual(overlap.clearance_m, -0.1)

    def test_replay_marks_stimulus_geometry_only_and_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            adapter = self._adapter(directory)
            stimulus = {
                "id": "test_grounded",
                "mode": "looming",
                "start_position_m": (0.06, 0.0, 0.0012),
                "velocity_mps": (-10.0, 0.0, 0.0),
                "direction_m": (-1.0, 0.0, 0.0),
                "size_m": (0.09, 0.09, 0.02),
                "contact_radius_m": 0.0012,
                "start_time_s": 0.0,
                "duration_s": 1.0,
            }
            from virtual_world import PhysicalStimulus, VirtualWorld
            result = EscapeTraceReplayer(
                self._trace(),
                adapter=adapter,
                stimulus=PhysicalStimulus.from_mapping(stimulus),
                world=VirtualWorld.from_json(ROOT / "config" / "scenarios.json"),
                config=ReplayConfig(render=False),
            ).run()
            self.assertEqual(len(result.telemetry), 2)
            self.assertFalse(result.metadata["autonomous_closed_loop_evidence"])
            self.assertFalse(result.metadata["autonomous_behavior_claim"])
            self.assertEqual(result.metadata["stimulus"]["control_authority"], "none")
            row = result.telemetry[0]
            self.assertIn("ttmn_activity", row["neuromuscular"])
            self.assertIn("wing_motor_left", row["neuromuscular"])
            self.assertIn("jump_force_body_N", row)
            self.assertIn("pose", row)
            self.assertIn("clearance_m", row["obb_counterfactual"])
            self.assertTrue(row["obb_counterfactual"]["counterfactual"])
            self.assertFalse(row["obb_counterfactual"]["physical_effect_applied"])
            self.assertTrue(row["stimulus"]["forbidden_for_control"])
            self.assertFalse(row["stimulus"]["physical_effect_applied"])
            self.assertFalse(row["replay_policy"]["stop_on_would_contact"])
            self.assertTrue(result.metadata["replay_policy"]["continues_after_would_contact"])
            self.assertEqual(result.metadata["replay_policy"]["replayed_duration_ms"], 10.0)
            self.assertTrue(any(item["obb_counterfactual"]["would_contact"] for item in result.telemetry))
            self.assertEqual(len(result.telemetry), 2, "replay must continue after would-contact")
            path = write_replay_artifact(result, directory / "artifact")
            self.assertTrue(path.exists())
            self.assertTrue((directory / "artifact" / "telemetry.jsonl").exists())

    def test_malformed_trace_and_non_grid_window_fail_closed(self) -> None:
        with self.assertRaises(ReplayError):
            SparseSpikeTrace(np.asarray([1.0]), np.asarray([0, 2]), np.asarray([1]))
        with tempfile.TemporaryDirectory() as temp:
            adapter = self._adapter(Path(temp))
            with self.assertRaises(ReplayError):
                aggregate_population_windows(self._trace(), adapter, window_ms=3.0, selectors=("WL",))


if __name__ == "__main__":
    unittest.main()
