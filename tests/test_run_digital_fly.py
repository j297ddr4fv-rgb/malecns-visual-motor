#!/usr/bin/env python3
"""CPU-only end-to-end checks for the digital-fly launcher."""

from __future__ import annotations

import json
import base64
from dataclasses import replace
import inspect
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_digital_fly import (  # noqa: E402
    BackendSelectionError,
    COMPACT_TELEMETRY_SCHEMA_VERSION,
    MINIMAL_TELEMETRY_SCHEMA_VERSION,
    MINIMAL_TRAJECTORY_SCHEMA_VERSION,
    DigitalFlyRunner,
    EscapeChannelAdapter,
    EscapeChannelSet,
    LauncherError,
    MalecnsNeuralAdapter,
    RendererBackendError,
    RunOptions,
    SCENARIOS,
    VisualDriveCalibration,
    _FrameStagePipeline,
    _RendererBackendWorldProxy,
    _free_behavior_row,
    _required_hash_only_array_record,
    parse_args,
)
from twin_contracts import NeuralOutput, SourceLabel, Timestamp  # noqa: E402
import validate_digital_fly  # noqa: E402


class RunDigitalFlyTest(unittest.TestCase):
    def _options(self, out: Path, **overrides: object) -> RunOptions:
        values: dict[str, object] = {
            "out": out,
            "scenarios": SCENARIOS,
            "duration_override_s": 0.01,
            "width": 8,
            "height": 6,
            "columns": 8,
            "t4t5": 4,
        }
        values.update(overrides)
        return RunOptions(**values)  # type: ignore[arg-type]

    def test_frame_stage_pipeline_defers_or_overlaps_only_pure_prepare(self) -> None:
        disabled_calls: list[str] = []
        disabled = _FrameStagePipeline(enabled=False)
        try:
            deferred = disabled.start_sensory_prepare(
                lambda: disabled_calls.append("ran") or "prepared"
            )
            self.assertEqual(disabled_calls, [])
            value, compute_s, wait_s = disabled.finish_sensory_prepare(deferred)
            self.assertEqual(value, "prepared")
            self.assertEqual(disabled_calls, ["ran"])
            self.assertGreaterEqual(compute_s, 0.0)
            self.assertEqual(wait_s, 0.0)
        finally:
            disabled.close()

        started = threading.Event()
        release = threading.Event()
        enabled = _FrameStagePipeline(enabled=True)
        try:
            def prepare() -> str:
                started.set()
                if not release.wait(timeout=2.0):
                    raise RuntimeError("test prepare was not released")
                return "prepared"

            job = enabled.start_sensory_prepare(prepare)
            self.assertTrue(started.wait(timeout=2.0))
            release.set()
            value, _, _ = enabled.finish_sensory_prepare(job)
            self.assertEqual(value, "prepared")

            manifest = enabled.manifest()
            self.assertTrue(manifest["effective"])
            self.assertEqual(manifest["sensory_tasks"], 1)
            self.assertFalse(manifest["cross_frame_control_overlap"])
            self.assertFalse(manifest["telemetry_async"])
        finally:
            enabled.close()

    def test_reduced_body_uses_world_body_contact_plane_once(self) -> None:
        """The runner aligns dynamics to the body-centre floor and only reads it."""

        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(Path(tmp), scenarios=("takeoff",)))
            scenario = runner.world.scenarios["takeoff"]
            dynamics = runner._new_body(scenario)
            floor = runner.world.config.ground_z_m + runner.world.config.body_radius_m

            self.assertAlmostEqual(dynamics.config.ground_plane_z_m, floor, places=12)
            self.assertAlmostEqual(dynamics.state.position_m[2], floor, places=12)
            self.assertTrue(runner._body_contact(dynamics.state, dynamics))

            output = dynamics.step({"left_power": 0.0, "right_power": 0.0}, dt_s=0.005)
            state_after_dynamics = dynamics.state.to_dict()
            self.assertGreater(output.contact_force_body_N[2], 0.0)
            self.assertTrue(runner._clamp_ground(dynamics))
            self.assertEqual(dynamics.state.to_dict(), state_after_dynamics)

    def test_runner_dynamics_lands_on_target_pad_and_reports_target_contact(self) -> None:
        """The runner-created FlightDynamics body stops on the raised pad."""

        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(Path(tmp), scenarios=("landing",)))
            scenario = runner.world.scenarios["landing"]
            dynamics = runner._new_body(scenario)
            target = runner.world.config.target
            support_z = target.max_m[2] + runner.world.config.body_radius_m
            dynamics.state.position_m = (target.position_m[0], target.position_m[1], support_z + 0.005)
            dynamics.state.velocity_mps = (0.0, 0.0, -2.0)

            output = dynamics.step({"left_power": 0.0, "right_power": 0.0}, dt_s=0.005)

            self.assertAlmostEqual(output.state.position_m[2], support_z, places=12)
            self.assertAlmostEqual(output.state.velocity_mps[2], 0.0, places=12)
            self.assertTrue(output.target_contact)
            self.assertEqual(output.contact_surface_id, target.id)
            self.assertEqual(
                output.state.position_m[:2],
                (target.position_m[0], target.position_m[1]),
            )

    def test_runner_telemetry_exposes_target_contact_from_dynamics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("landing",), duration_override_s=0.005,
            ))
            make_body = runner._new_body

            def prepared_body(scenario: object) -> object:
                dynamics = make_body(scenario)
                target = runner.world.config.target
                support_z = target.max_m[2] + runner.world.config.body_radius_m
                dynamics.state.position_m = (
                    target.position_m[0], target.position_m[1], support_z + 0.005,
                )
                dynamics.state.velocity_mps = (0.0, 0.0, -2.0)
                return dynamics

            runner._new_body = prepared_body  # type: ignore[method-assign]
            payload = runner.run()
            row = payload["trajectory"]["trajectory"][0]
            self.assertTrue(row["target_contact"])
            self.assertEqual(row["contact_surface_id"], runner.world.config.target.id)
            self.assertAlmostEqual(row["position_m"][2], 0.055, places=12)

    def test_default_cpu_path_runs_all_five_scenarios_and_writes_validator_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(Path(tmp)))
            payload = runner.run()
            paths = runner.write_outputs(payload)
            manifest = json.loads(Path(paths["manifest"]).read_text())
            self.assertEqual(manifest["scenario_names"], list(SCENARIOS))
            self.assertEqual(manifest["requested_backends"], {
                "neural": "stub", "vision": "stub", "body": "reduced",
            })
            self.assertEqual(manifest["provenance"]["evidence_class"], "digital_simulation")
            self.assertEqual(manifest["provenance"]["neural"], "handbuilt")
            self.assertEqual(manifest["provenance"]["body"], "handbuilt")
            self.assertEqual(Path(manifest["outputs"]["validation"]).name, "validation.json")
            self.assertEqual(Path(manifest["outputs"]["video"]).name, "flight.mp4")
            self.assertTrue(all(name in manifest["scenario_results"] for name in SCENARIOS))
            self.assertTrue(all(len(result["metrics"]) > 0 for result in manifest["scenario_results"].values()))
            report = validate_digital_fly.validate_file(paths["trajectory"])
            self.assertEqual(report["evidence"]["class"], "digital_simulation")
            self.assertEqual(report["decision"], "NOT_VALIDATED")
            self.assertGreater(report["evidence"]["n_samples"], 0)

    def test_trajectory_output_streams_without_monolithic_json_string(self) -> None:
        """The large trajectory must never be passed to ``json.dumps``."""

        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("full-mission",), frame_count=3,
            ))
            payload = runner.run()
            original_dumps = json.dumps

            def guarded_dumps(value: object, *args: object, **kwargs: object) -> str:
                if isinstance(value, dict) and isinstance(value.get("trajectory"), list):
                    raise AssertionError("trajectory was materialised as one JSON string")
                return original_dumps(value, *args, **kwargs)

            with mock.patch("run_digital_fly.json.dumps", side_effect=guarded_dumps):
                paths = runner.write_outputs(payload)

            written = json.loads(Path(paths["trajectory"]).read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], payload["trajectory"]["schema_version"])
            self.assertEqual(written["trajectory"], payload["trajectory"]["trajectory"])
            self.assertEqual(written["metadata"]["manifest_path"], paths["manifest"])

    def test_cpu_run_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = DigitalFlyRunner(self._options(Path(first_tmp))).run()
            second = DigitalFlyRunner(self._options(Path(second_tmp))).run()
            self.assertEqual(first["trajectory"], second["trajectory"])

    def test_explicit_flybody_fails_closed_without_a_validated_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BackendSelectionError):
                DigitalFlyRunner(self._options(Path(tmp), body="flybody"))

    def test_output_rows_use_shared_contract_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = DigitalFlyRunner(self._options(Path(tmp), scenarios=("yaw-step",))).run()
            row = payload["trajectory"]["trajectory"][0]
            self.assertEqual(row["body"]["timestamp"], row["world"]["timestamp"])
            self.assertEqual(row["sensory"]["timestamp"], row["neural"]["timestamp"])
            self.assertEqual(row["motor"]["timestamp"], row["neural"]["timestamp"])
            self.assertIn("position_m", row["body"])
            self.assertIn("rates_hz", row["neural"])
            self.assertIn("source", row["motor"])

    def test_multiframe_mission_rows_are_renderable_and_alias_turn_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("takeoff", "hover", "turn", "landing"),
                frame_count=3,
            )).run()
            manifest = payload["manifest"]
            self.assertEqual(manifest["scenario_names"], ["takeoff", "hover", "turn", "landing"])
            self.assertEqual(len(payload["trajectory"]["trajectory"]), 12)
            self.assertEqual(
                {name: result["metrics"]["frame_count"] for name, result in manifest["scenario_results"].items()},
                {"takeoff": 3, "hover": 3, "turn": 3, "landing": 3},
            )
            row = payload["trajectory"]["trajectory"][0]
            self.assertEqual(row["frame_index"], 0)
            self.assertEqual(row["dt_s"], 0.005)
            self.assertIn("body_before", row)
            self.assertIn("pose", row["render"])
            self.assertEqual(row["render"]["resolution"], [8, 6])
            self.assertEqual(row["render"]["left"], row["world"]["render"]["left"])
            self.assertEqual(row["neural_window"]["steps"], 1)

    def test_visual_calibration_requires_mapping_and_preserves_engine_contract(self) -> None:
        seen: list[dict[str, object]] = []

        class Calibrated:
            def __init__(self, drive: np.ndarray, target_indices: np.ndarray, units: str):
                self.drive = drive * 2.0
                self.target_indices = target_indices.copy()
                self.units = units

            def as_engine_mapping(self) -> dict[str, object]:
                return {
                    "target_indices": self.target_indices,
                    "values": self.drive,
                    "units": self.units,
                }

        def calibrate(drive, target_indices=None, *, target_ids=None, units=None, frame_index=None):
            seen.append({
                "target_indices": np.asarray(target_indices).copy(),
                "target_ids": np.asarray(target_ids).copy(),
                "units": units,
                "frame_index": frame_index,
            })
            return Calibrated(np.asarray(drive), np.asarray(target_indices), str(units))

        adapter = VisualDriveCalibration(
            types.SimpleNamespace(calibrate=calibrate), module_name="mock_calibration",
        )
        drive, metadata = adapter.apply(
            np.array([0.2, -0.1], dtype=np.float32),
            scenario="hover", timestamp_s=0.005, frame_index=1,
            target_indices=np.array([11, 17], dtype=np.int64),
            target_ids=np.array([101, 202], dtype=np.int64),
            units="current_mV_per_ms",
        )
        np.testing.assert_allclose(drive, [0.4, -0.2])
        np.testing.assert_array_equal(seen[0]["target_indices"], [11, 17])
        self.assertEqual(seen[0]["units"], "current_mV_per_ms")
        self.assertEqual(metadata["engine_mapping"]["units"], "current_mV_per_ms")
        np.testing.assert_array_equal(metadata["engine_mapping"]["target_indices"], [11, 17])

    def test_project_visual_calibration_module_is_explicitly_loaded_and_applied(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = {
            "schema_version": "visual-drive-calibration-1",
            "name": "runner-test-calibration",
            "enabled": True,
            "input": {
                "field": "flyvis_t4t5", "units": "current_mV_per_ms",
                "source": "fitted", "expected_target_count": 4, "max_abs": 1.0,
            },
            "transform": {
                "gain": 2.0, "max_gain": 10.0, "max_abs_output": 1.0,
                "saturation_policy": "clip",
            },
            "tonic_state_inputs": {"enabled": False, "entries": []},
            "provenance": {"test": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "calibration.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            runner = DigitalFlyRunner(self._options(
                tmp_path / "out", scenarios=("hover",), frame_count=1, t4t5=4,
                visual_calibration=root / "scripts" / "visual_drive_calibration.py",
                visual_calibration_config=config_path,
            ))
            # A real FlyVis runtime publishes these fields.  This CPU mock
            # makes the same explicit mapping available on the deterministic
            # backend without changing its response implementation.
            runner.flyvis.backend.target_indices = np.array([10, 11, 12, 13], dtype=np.int64)
            runner.flyvis.backend.target_ids = np.array([100, 101, 102, 103], dtype=np.int64)
            runner.flyvis.backend.units = "current_mV_per_ms"
            payload = runner.run()
            row = payload["trajectory"]["trajectory"][0]
            self.assertEqual(payload["manifest"]["components"]["vision"]["calibration"]["name"],
                             "runner-test-calibration")
            self.assertEqual(row["provenance"]["visual_calibration"], "fitted")
            self.assertEqual(payload["manifest"]["components"]["vision"]["mapping"]["count"], 4)

    def test_mock_real_neural_path_runs_exactly_ten_substeps_and_compacts_dense_vectors(self) -> None:
        class FakeEngine:
            backend = "numpy_cpu"
            device = "cpu"
            max_vram_gib = 15.5

            def manifest(self):
                return {"backend": self.backend, "device": self.device, "n": 16, "edges": 0}

        class FakeNeural:
            backend_name = "numpy_cpu"
            device_name = "cpu"
            engine = FakeEngine()
            source = SourceLabel.CONNECTOME

            def __init__(self):
                self.timestamps: list[float] = []
                self.calls = 0

            def reset(self):
                self.timestamps.clear()
                self.calls = 0

            def step(self, sensory, timestamp):
                self.calls += 1
                self.timestamps.append(timestamp.t_s)
                return NeuralOutput(
                    timestamp, rates_hz=np.zeros(176422, dtype=np.float32),
                    spikes=np.zeros(176422, dtype=np.uint8),
                    named_outputs={
                        "backend": "numpy_cpu", "device": "cpu",
                        "spike_count": 0,
                        "population_rates_hz": {"prefix:DNg02": 25.0},
                    }, source=SourceLabel.CONNECTOME,
                )

        class FakeNeuromuscular:
            def reset(self, clear_baseline=False):
                del clear_baseline

            def step(self, signal, *, dt_ms, timestamp):
                del signal, dt_ms, timestamp
                return types.SimpleNamespace(
                    left_wing_power=0.5, right_wing_power=0.9,
                    wing_power=0.7, wing_steering=0.4,
                )

        with tempfile.TemporaryDirectory() as tmp:
            options = self._options(Path(tmp), scenarios=("hover",), frame_count=3,
                                    neural="stub", full_neural_telemetry=False)
            runner = DigitalFlyRunner(options)
            fake = FakeNeural()
            runner.options = replace(runner.options, neural="malecns")
            runner.neural_kind = fake
            runner.neuromuscular = FakeNeuromuscular()
            payload = runner.run()
            rows = payload["trajectory"]["trajectory"]
            self.assertEqual(fake.calls, 30)
            self.assertEqual(rows[-1]["neural_window"]["steps"], 10)
            self.assertEqual(payload["manifest"]["scenario_results"]["hover"]["metrics"]["neural_steps"], 30)
            self.assertIsNone(rows[0]["neural"]["rates_hz"])
            self.assertIsNone(rows[0]["neural"]["spikes"])
            self.assertEqual(rows[0]["neural"]["named_outputs"]["neural_window_steps"], 10)
            self.assertAlmostEqual(rows[0]["auto_command"]["yaw"], 0.4)
            self.assertAlmostEqual(rows[0]["auto_command"]["roll"], 0.0)
            self.assertLess(len(json.dumps(rows[0])), 100_000)
            deltas = np.diff(fake.timestamps)
            np.testing.assert_allclose(deltas, 0.0005, atol=1e-12)

    def test_mock_engine_window_api_is_preferred_and_timestamped_at_frame_start(self) -> None:
        class WindowNeural:
            backend_name = "numpy_cpu"
            device_name = "cpu"
            window_api = "engine_step_window"
            engine = types.SimpleNamespace(manifest=lambda: {"backend": "numpy_cpu", "device": "cpu"})

            def __init__(self):
                self.calls = 0
                self.steps: list[int] = []

            def reset(self):
                self.calls = 0
                self.steps.clear()

            def step_window(self, sensory, timestamp, *, steps):
                del sensory
                self.calls += 1
                self.steps.append(steps)
                return NeuralOutput(
                    Timestamp(timestamp.t_s + 0.0045, timestamp.tick + 9, "neural", 2000.0),
                    rates_hz=np.zeros(16, dtype=np.float32),
                    spikes=np.zeros(16, dtype=np.int32),
                    named_outputs={
                        "neural_window_steps": steps,
                        "neural_window_api": "engine_step_window",
                    },
                    source=SourceLabel.CONNECTOME,
                )

        class FakeNeuromuscular:
            def reset(self, clear_baseline=False):
                del clear_baseline

            def step(self, signal, *, dt_ms, timestamp):
                del signal, dt_ms, timestamp
                return types.SimpleNamespace(
                    left_wing_power=0.7, right_wing_power=0.7,
                    wing_power=0.7, wing_steering=0.0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("hover",), frame_count=2,
                neural="stub",
            ))
            fake = WindowNeural()
            runner.options = replace(runner.options, neural="malecns")
            runner.neural_kind = fake
            runner.neuromuscular = FakeNeuromuscular()
            payload = runner.run()
            rows = payload["trajectory"]["trajectory"]
            self.assertEqual(fake.calls, 2)
            self.assertEqual(fake.steps, [10, 10])
            self.assertEqual(rows[0]["neural_window"]["api"], "engine_step_window")
            self.assertEqual(rows[0]["neural"]["named_outputs"]["neural_window_steps"], 10)
            self.assertEqual(rows[0]["neural"]["timestamp"]["t_s"], 0.0)
            self.assertEqual(payload["manifest"]["effective_execution"]["neural_window_api"],
                             "engine_step_window")

    def test_fitted_readout_branch_uses_explicit_runtime_mapping_and_manifest(self) -> None:
        runtime_mapping = {
            "target_indices": np.array([10, 11, 12, 13], dtype=np.int64),
            "target_ids": np.array([100, 101, 102, 103], dtype=np.int64),
            "directions": np.array([
                [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            ], dtype=np.float32),
            "subtype": np.array(["T4a", "T4b", "T5a", "T5b"]),
            "side": np.array(["L", "L", "R", "R"]),
        }
        calls: list[dict[str, object]] = []

        class FakeReadout:
            target_count = 4

            def readout(self, response, **kwargs):
                del response
                calls.append(kwargs)
                return types.SimpleNamespace(
                    yaw=0.7, pitch=-0.2, confidence=0.8,
                    directional_coherence=0.9, response_energy=1.5, valid=True,
                    provenance={"target_count": 4, "mapping_sha256": "readout-map"},
                )

            def manifest(self):
                return {"readout": "fake-fitted", "mapping_sha256": "readout-map"}

        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("hover",), frame_count=1, t4t5=4,
            ))
            runner.visual_control_readout = FakeReadout()
            runner.flyvis_runtime_adapter = types.SimpleNamespace(mapping=runtime_mapping)
            runner.flyvis.backend.target_indices = runtime_mapping["target_indices"]
            runner.flyvis.backend.target_ids = runtime_mapping["target_ids"]
            runner.flyvis.backend.units = "current_mV_per_ms"
            payload = runner.run()
            row = payload["trajectory"]["trajectory"][0]
            self.assertAlmostEqual(row["visual_direction"], 0.7)
            self.assertAlmostEqual(row["visual_pitch_cue"], -0.2)
            self.assertEqual(row["visual_readout"]["mapping_sha256"], "readout-map")
            self.assertEqual(calls[0]["units"], "current_mV_per_ms")
            np.testing.assert_array_equal(calls[0]["target_indices"], [10, 11, 12, 13])
            np.testing.assert_array_equal(calls[0]["target_ids"], [100, 101, 102, 103])
            self.assertEqual(payload["manifest"]["components"]["vision"]["control_readout"]["readout"],
                             "fake-fitted")
            self.assertEqual(payload["manifest"]["provenance"]["visual_control_readout"], "fitted")

    def test_fitted_readout_mapping_mismatch_fails_closed(self) -> None:
        runtime_mapping = {
            "target_indices": np.array([10, 11, 12, 13], dtype=np.int64),
            "target_ids": np.array([100, 101, 102, 103], dtype=np.int64),
            "directions": np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (4, 1)),
            "subtype": np.array(["T4a", "T4b", "T5a", "T5b"]),
            "side": np.array(["L", "L", "R", "R"]),
        }

        class FakeReadout:
            target_count = 4

        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("hover",), frame_count=1, t4t5=4,
            ))
            runner.visual_control_readout = FakeReadout()
            runner.flyvis_runtime_adapter = types.SimpleNamespace(mapping=runtime_mapping)
            runner.flyvis.backend.target_indices = np.array([10, 11, 12], dtype=np.int64)
            runner.flyvis.backend.target_ids = np.array([100, 101, 102], dtype=np.int64)
            runner.flyvis.backend.units = "current_mV_per_ms"
            with self.assertRaises(LauncherError):
                runner.run()

    def test_explicit_flight_state_drive_is_applied_and_manifested(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("takeoff",), frame_count=2,
                flight_state_drive=root / "config" / "flight_state_drive.json",
            ))
            payload = runner.run()
            row = payload["trajectory"]["trajectory"][0]
            state_drive = row["flight_state_drive"]
            self.assertIsNotNone(state_drive)
            self.assertEqual(state_drive["provenance"]["biological_status"], "hypothesis")
            self.assertIn("poisson_hz", state_drive["channels"])
            self.assertIn("prefix:DNg02/L", state_drive["channels"]["poisson_hz"])
            self.assertIn("phase", state_drive["context"])
            manifest = payload["manifest"]
            self.assertEqual(manifest["provenance"]["flight_state_drive"], "handbuilt_hypothesis")
            self.assertEqual(manifest["components"]["flight_state_drive"]["name"],
                             "malecns-flight-state-drive-v1")

    def test_stabilizer_context_exposes_body_frame_velocity_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("full-mission",), frame_count=1,
            ))
            pose = types.SimpleNamespace(
                position_m=(0.0, -0.55, 0.30),
                velocity_mps=(1.0, 2.0, 0.25),
                yaw_rad=float(np.pi / 2.0),
                pitch_rad=0.0,
                roll_rad=0.0,
                angular_velocity_radps=(0.0, 0.0, 0.0),
            )
            context = runner._flight_state_context(
                runner.world.scenarios["full-mission"],
                "full-mission",
                pose,
                2.0,
                False,
            )
            errors = context["body_error"]
            self.assertAlmostEqual(errors["forward_velocity_mps"], 2.0)
            self.assertAlmostEqual(errors["lateral_velocity_mps"], -1.0)

    def test_explicit_body_stabilizer_blends_without_writing_pose(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner(self._options(
                Path(tmp), scenarios=("full-mission",), frame_count=3,
                flight_state_drive=root / "config" / "flight_state_drive.json",
                flight_body_stabilizer=root / "config" / "flight_body_stabilizer.json",
            ))
            payload = runner.run()
            row = payload["trajectory"]["trajectory"][-1]
            stabilizer = row["flight_body_stabilizer"]
            self.assertTrue(stabilizer["active"])
            self.assertLessEqual(stabilizer["stabilizer_blend"], 0.35)
            self.assertFalse(stabilizer["provenance"]["writes_body_state"])
            self.assertEqual(stabilizer["actuator_contribution_fractions"]["yaw"]["handbuilt"], 0.0)
            self.assertAlmostEqual(row["left_wing"], stabilizer["channels"]["left_wing"])
            self.assertAlmostEqual(row["right_wing"], stabilizer["channels"]["right_wing"])
            self.assertEqual(row["motor"]["source"], "handbuilt")
            manifest = payload["manifest"]
            self.assertEqual(
                manifest["provenance"]["flight_body_stabilizer"],
                "handbuilt_hypothesis",
            )
            self.assertFalse(
                manifest["components"]["flight_body_stabilizer"]["authority"]["writes_body_state"]
            )

    def test_missing_body_stabilizer_config_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BackendSelectionError):
                DigitalFlyRunner(self._options(
                    Path(tmp), flight_body_stabilizer=Path(tmp) / "missing.json",
                ))

    def test_strict_free_behavior_rejects_stub_and_accepts_explicit_gpu_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = self._options(
                Path(tmp), mode="free-behavior", neural="stub",
                neural_device="cpu", neural_backend="cpu", vision_device="cpu",
            )
            with self.assertRaisesRegex(BackendSelectionError, "stub scripted control"):
                DigitalFlyRunner(base)
            # Explicit GPU selection is accepted by policy; adapter preflight
            # remains responsible for the read-only CUDA gate.  Keep this
            # unit test dependency-free by checking argument acceptance only.
            parsed = parse_args([
                "--mode", "free-behavior", "--neural", "malecns",
                "--neural-device", "cuda", "--neural-backend", "torch",
                "--vision-device", "cuda:0",
            ])
            self.assertEqual(parsed.neural_device, "cuda")
            self.assertEqual(parsed.neural_backend, "torch")

    def test_renderer_backend_is_explicit_and_warp_does_not_fallback(self) -> None:
        parsed = parse_args(["--renderer-backend", "analytic_reference"])
        self.assertEqual(parsed.renderer_backend, "analytic_reference")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RendererBackendError, "requires --vision-device cuda:N"):
                DigitalFlyRunner(self._options(
                    Path(tmp), scenarios=("hover",), renderer_backend="warp_cuda",
                    vision_device="cpu",
                ))

    def test_renderer_proxy_prefers_device_eye_tensor_seam(self) -> None:
        calls: list[str] = []

        class Renderer:
            def render_eye_directions_tensor(self, pose: object, directions: object, *, eye: str) -> str:
                del pose, directions, eye
                calls.append("tensor")
                return "device-vector"

            def render_eye_directions(self, pose: object, directions: object, *, eye: str) -> str:
                del pose, directions, eye
                calls.append("host")
                return "host-vector"

        proxy = _RendererBackendWorldProxy(
            types.SimpleNamespace(), Renderer(), backend_name="warp_cuda",
        )
        value = proxy.render_eye_directions(object(), object(), eye="left")
        self.assertEqual(value, "device-vector")
        self.assertEqual(calls, ["tensor"])

    def test_renderer_proxy_caches_signature_plan(self) -> None:
        class Renderer:
            def render_eye_directions(self, pose: object, directions: object, *, eye: str) -> str:
                del pose, directions, eye
                return "host-vector"

        proxy = _RendererBackendWorldProxy(
            types.SimpleNamespace(), Renderer(), backend_name="warp_cuda",
        )
        _RendererBackendWorldProxy._call_plan.cache_clear()
        with mock.patch("run_digital_fly.inspect.signature", wraps=inspect.signature) as signature:
            proxy.render_eye_directions(object(), object(), eye="left")
            proxy.render_eye_directions(object(), object(), eye="right")
        self.assertEqual(signature.call_count, 1)
        self.assertGreaterEqual(_RendererBackendWorldProxy._call_plan.cache_info().hits, 1)

    def test_minimal_mode_is_explicit_and_strict_only(self) -> None:
        parsed = parse_args([
            "--telemetry-mode", "minimal", "--parallel-frame-stages",
        ])
        self.assertEqual(parsed.telemetry_mode, "minimal")
        self.assertTrue(parsed.parallel_frame_stages)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(BackendSelectionError, "strict free-behavior"):
                DigitalFlyRunner(self._options(
                    Path(tmp), telemetry_mode="minimal", scenarios=("hover",),
                ))
            with self.assertRaisesRegex(BackendSelectionError, "parallel frame stages"):
                DigitalFlyRunner(self._options(
                    Path(tmp), parallel_frame_stages=True, scenarios=("hover",),
                ))

    def test_minimal_hash_only_array_record_fails_closed(self) -> None:
        good = _required_hash_only_array_record(
            np.asarray([0.25, -0.5], dtype=np.float32), name="drive",
        )
        self.assertEqual(good["encoding"], "hash-only")
        self.assertEqual(good["shape"], [2])
        self.assertEqual(len(good["sha256"]), 64)

        with self.assertRaisesRegex(LauncherError, "non-empty vector"):
            _required_hash_only_array_record(np.asarray(1.0), name="drive")
        with self.assertRaisesRegex(LauncherError, "real numeric dtype"):
            _required_hash_only_array_record(np.asarray([object()]), name="drive")
        with self.assertRaisesRegex(LauncherError, "non-finite"):
            _required_hash_only_array_record(np.asarray([np.nan]), name="drive")
        with self.assertRaisesRegex(LauncherError, "invalid precomputed"):
            _required_hash_only_array_record(
                np.asarray([1.0]), name="drive", sha256="not-a-digest",
            )

    def test_minimal_row_is_hash_only_and_keeps_escape_scalars(self) -> None:
        timestamp = Timestamp(0.0, 0, "neural", 2000.0)
        left = np.linspace(0.0, 1.0, 721, dtype=np.float32)
        right = np.linspace(1.0, 0.0, 721, dtype=np.float32)
        drive = np.linspace(-1.0, 1.0, 13_582, dtype=np.float32)
        neural = NeuralOutput(
            timestamp, rates_hz=np.zeros(176422, dtype=np.float32),
            spikes=np.zeros(176422, dtype=np.uint8),
            named_outputs={
                "backend": "triton", "device": "cuda",
                "population_rates_hz": {"LC4": 4.0, "LPLC2": 3.0, "TTMn": 2.0},
                "population_spike_counts": {"DNp01": 1, "wing": 2, "liftoff": 1},
                "spike_count": 4, "active_neurons": 3,
                "neural_window_steps": 10, "neural_window_api": "engine_step_window",
                "neural_tick_dt_s": 0.0005, "neural_window_dt_s": 0.005,
            }, source=SourceLabel.CONNECTOME,
        )
        visual = types.SimpleNamespace(
            left_columns=left, right_columns=right, t4t5_drive=drive,
            backend="flyvis",
        )
        visual_observation = types.SimpleNamespace(
            optic_flow_rad_s=(0.1, -0.2, 0.0), sensor_id="flyvis",
            source=SourceLabel.FITTED, metadata={"backend": "flyvis"},
        )
        nm = types.SimpleNamespace(to_dict=lambda: {
            "channels": {"ttmn_activity": 0.2, "jump_motor": 0.3},
            "left_wing_power": 0.4, "right_wing_power": 0.5,
            "wing_power": 0.45, "wing_steering": 0.1,
            "leg_power": 0.0, "leg_steering": 0.0,
            "mouth": 0.0, "groom": 0.0, "dt_ms": 5.0,
            "source": "connectome", "saturated": False,
            "provenance": {"resolution": {"huge": ["omitted"]}},
        })
        mech = types.SimpleNamespace(to_dict=lambda: {
            "channels": {"wind_body_forward": 0.1},
            "population_drive": {"jo": 0.2},
            "raw_channels": {"wind_body_forward": 0.2},
            "population_metadata": {"static": "omitted"},
        })
        raw = {
            "scenario": "free-behavior", "frame_index": 0,
            "frame_time_s": 0.0, "time_s": 0.005, "dt_s": 0.005,
            "interval_start_s": 0.0, "interval_end_s": 0.005,
            "timestamp": timestamp.to_dict(),
            "position_m": [0.0, 0.0, 0.01], "velocity_mps": [0.0, 0.0, 0.0],
            "orientation_quat": [1.0, 0.0, 0.0, 0.0],
            "orientation_rad": [0.0, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
            "contact": True, "grounded": True, "target_contact": False,
            "ground_distance_m": 0.0, "flight_mode": "free_behavior",
            "controller_mode": "MaleCNS_direct", "left_wing": 0.4,
            "right_wing": 0.5, "wing_drive_left": 0.4, "wing_drive_right": 0.5,
            "yaw_rate": 0.0, "pitch": 0.0, "roll": 0.0,
            "wind_mps": [0.0, 0.0, 0.0],
            "visual_drive_summary": {}, "visual_directional_readout": "record_only",
            "stimulus_state": {"active": True}, "stimulus_condition": "looming_grounded",
            "stimulus_name": "looming_grounded", "stimulus_contact": False,
            "stimulus_interaction": "visual_only_non_interacting",
            "escape_channels": {"channels": {}, "records": {}},
            "collision_count": 0, "body_before": {}, "body": {},
            "sensory": {"channels": {"optic_flow_x_rad_s": 0.1}},
            "neural": neural.to_dict(), "neural_window": {"steps": 10},
            "jump_decision_telemetry": {"ttmn_activity": 0.2},
            "jump_input": {"ttmn_activity": 0.2},
            "jump_output": {"jump_actuator_active": False},
            "motor": {"timestamp": timestamp.to_dict(), "yaw_rate_rad_s": 0.0,
                       "pitch_rate_rad_s": 0.0, "roll_rate_rad_s": 0.0,
                       "thrust_mps2": 4.4, "source": "connectome", "saturated": False},
            "diagnostics": {},
        }
        row = _free_behavior_row(
            raw, flyvis_output=visual, visual_observation=visual_observation,
            nm_output=nm, mechanosensory=mech,
            escape_packet=raw["escape_channels"], neural_output=neural,
            mode="minimal",
            exact_sampling={"mode": "exact_world_eye_directions"},
            drive_sha256=__import__("hashlib").sha256(drive.tobytes()).hexdigest(),
            drive_summary={"count": int(drive.size), "finite": True},
        )
        encoded = json.dumps(row, separators=(",", ":"))
        self.assertEqual(row["schema_version"], MINIMAL_TELEMETRY_SCHEMA_VERSION)
        self.assertLess(len(encoded), 30_000)
        self.assertIsNone(row["neural"]["rates_hz"])
        self.assertIsNone(row["neural"]["spikes"])
        self.assertIn("population_rates_hz", row["neural"]["named_outputs"])
        self.assertIn("population_spike_counts", row["neural"]["named_outputs"])
        self.assertIn("ttmn_activity", row["jump_decision_telemetry"])
        for field in ("left_columns", "right_columns", "t4t5_drive"):
            record = row["visual_input"][field]
            self.assertEqual(record["encoding"], "hash-only")
            self.assertNotIn("data", record)
            self.assertNotIn("compressed_bytes", record)
        self.assertNotIn("population_metadata", row["mechanosensory"])

    def test_minimal_write_outputs_uses_compact_trajectory_and_matching_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = DigitalFlyRunner.__new__(DigitalFlyRunner)
            runner.options = types.SimpleNamespace(out=Path(tmp) / "out")
            runner.minimal_telemetry = True
            row = {"schema_version": MINIMAL_TELEMETRY_SCHEMA_VERSION, "time_s": 0.005}
            payload = {
                "manifest": {},
                "trajectory": {
                    "schema_version": MINIMAL_TRAJECTORY_SCHEMA_VERSION,
                    "metadata": {}, "trajectory": [row],
                },
            }
            paths = runner.write_outputs(payload)
            telemetry = [json.loads(line) for line in Path(paths["telemetry"]).read_text().splitlines()]
            trajectory = json.loads(Path(paths["trajectory"]).read_text())
            self.assertEqual(telemetry, trajectory["trajectory"])
            self.assertNotIn("\n  ", Path(paths["trajectory"]).read_text())

    def test_exact_flyvis_runtime_step_receives_device_vectors_before_telemetry_copy(self) -> None:
        class DeviceVector:
            device = "cuda:0"

            def __init__(self, values: object):
                self._values = np.asarray(values, dtype=np.float32)
                self.shape = self._values.shape

            def detach(self) -> "DeviceVector":
                return self

            def cpu(self) -> "DeviceVector":
                return self

            def numpy(self) -> np.ndarray:
                return self._values

        left = DeviceVector([0.1, 0.2])
        right = DeviceVector([0.3, 0.4])
        seen: list[tuple[object, object]] = []

        class Runtime:
            name = "flyvis"

            def step_detailed(self, left_input: object, right_input: object) -> object:
                seen.append((left_input, right_input))
                return types.SimpleNamespace(
                    drive=np.array([0.5], dtype=np.float32), frame_index=7,
                )

        runner = DigitalFlyRunner.__new__(DigitalFlyRunner)
        runner.flyvis_runtime_adapter = Runtime()
        runner.flyvis = types.SimpleNamespace()
        output = runner._process_exact_flyvis_columns(left, right, 0.0)
        self.assertIs(seen[0][0], left)
        self.assertIs(seen[0][1], right)
        np.testing.assert_array_equal(output.left_columns, left.numpy())
        np.testing.assert_array_equal(output.right_columns, right.numpy())
        self.assertEqual(output.frame_index, 7)

    def test_neural_cuda_graph_requires_explicit_triton_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(BackendSelectionError, "requires --neural malecns"):
                DigitalFlyRunner(self._options(
                    Path(tmp), neural_cuda_graph=True,
                    neural_device="cpu", neural_backend="cpu",
                ))
        parsed = parse_args([
            "--neural-cuda-graph", "--neural-device", "cuda",
            "--neural-backend", "triton",
        ])
        self.assertTrue(parsed.neural_cuda_graph)

    def test_free_jump_uses_ttmn_and_combined_body_step_lifts_from_contact(self) -> None:
        """TTMn is sufficient for the one-step bilateral jump event."""

        from flight_dynamics import FlightDynamics, FlightDynamicsConfig, FlightState, JumpLegConfig

        nm_out = types.SimpleNamespace(
            channels={
                "ttmn_activity": 1.0,
                "jump": 0.0,  # DNp01-like decision telemetry must not drive force.
                "wing_motor_left": 0.0,
                "wing_motor_right": 0.0,
            },
        )
        jump_input, decision = DigitalFlyRunner._free_jump_input(nm_out)
        self.assertEqual(jump_input["activation_source"], "ttmn_activity")
        self.assertAlmostEqual(jump_input["ttmn_activity"], 1.0)
        self.assertAlmostEqual(jump_input["jump_activation"], 1.0)
        self.assertAlmostEqual(jump_input["leg_power"], 1.0)
        self.assertAlmostEqual(decision["dnp01_activity"], 0.0)

        floor = 0.0012
        config = FlightDynamicsConfig(
            ground_plane_z_m=floor,
            allow_legacy_wing_fallback=False,
        )
        state = FlightState(
            time_s=0.0,
            position_m=(0.0, 0.0, floor),
            velocity_mps=(0.0, 0.0, 0.0),
        )
        dynamics = FlightDynamics(
            config=config,
            state=state,
            jump_config=JumpLegConfig(
                mass_kg=config.mass_kg,
                inertia_kg_m2=config.inertia_kg_m2,
                gravity_mps2=config.gravity_mps2,
                ground_plane_z_m=floor,
            ),
        )
        output = DigitalFlyRunner._free_dynamics_step(
            dynamics, nm_out, jump_input, dt_s=0.005,
        )
        self.assertTrue(output.jump_actuator_active)
        self.assertGreater(output.jump_force_body_N[2], 0.0)
        self.assertGreater(output.state.position_m[2], floor)

    def test_free_config_uses_calibrated_grounded_conditions(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "free_escape_experiment.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["stimulus_name"], "looming_grounded")
        self.assertEqual(
            config["stimulus_conditions"],
            ["looming_grounded", "static_grounded", "sham_grounded"],
        )
        self.assertNotIn("stimulus_overrides", config)
        self.assertFalse(config["experiment"]["allow_legacy_wing_fallback"])
        self.assertEqual(config["experiment"]["visual_report_populations"], ["LC4", "LPLC2"])
        self.assertEqual(config["experiment"]["stimulus_interaction"], "visual_only_non_interacting")
        self.assertAlmostEqual(config["experiment"]["palm_contact_distance_m"], 0.0112)
        self.assertAlmostEqual(config["experiment"]["surface_contact_deadline_s"], 1.9776)

    def test_strict_flyvis_sampling_uses_runtime_column_directions(self) -> None:
        runner = DigitalFlyRunner.__new__(DigitalFlyRunner)
        runner.flyvis_runtime_adapter = types.SimpleNamespace(
            column_directions={
                "L": np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
                "R": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32),
            },
        )
        runner.flyvis = types.SimpleNamespace(n_columns=2)
        seen: list[tuple[str, np.ndarray]] = []

        def render(pose: object, directions: np.ndarray, *, eye: str, timestamp_s: float) -> list[float]:
            del pose, timestamp_s
            seen.append((eye, directions.copy()))
            return [float(value) for value in directions[:, 0]]

        runner.world = types.SimpleNamespace(render_eye_directions=render)
        left, right, provenance = runner._free_exact_flyvis_columns(object(), 0.0)
        np.testing.assert_allclose(left, [1.0, 0.0])
        np.testing.assert_allclose(right, [1.0, 0.0])
        cached_left_id = id(runner._exact_direction_cache["left"])
        runner._free_exact_flyvis_columns(object(), 0.005)
        self.assertEqual(id(runner._exact_direction_cache["left"]), cached_left_id)
        self.assertEqual(provenance["mode"], "exact_world_eye_directions")
        self.assertEqual(provenance["n_columns"], 2)
        self.assertEqual([item[0] for item in seen], ["left", "right", "left", "right"])
        np.testing.assert_allclose(np.linalg.norm(seen[0][1], axis=1), [1.0, 1.0])
        np.testing.assert_allclose(np.linalg.norm(seen[1][1], axis=1), [1.0, 1.0])
        runner.flyvis_runtime_adapter.column_directions["L"] = np.zeros((3, 3), dtype=np.float32)
        with self.assertRaisesRegex(LauncherError, "shape changed"):
            runner._free_exact_flyvis_columns(object(), 0.01)

    def test_neural_gain_is_forwarded_to_engine_factory(self) -> None:
        captured: dict[str, object] = {}

        class Engine:
            backend = "torch_cuda"
            device = "cuda"

        class Module:
            @staticmethod
            def create_engine(**kwargs: object) -> Engine:
                captured.update(kwargs)
                return Engine()

        adapter = MalecnsNeuralAdapter(
            Module(), seed=1729, device="cuda", backend="torch",
            neural_gain=1.2,
        )
        self.assertAlmostEqual(float(captured["gain"]), 1.2)
        self.assertAlmostEqual(float(adapter.neural_gain), 1.2)

    def test_minimal_dense_flyvis_echo_flag_is_forwarded_explicitly(self) -> None:
        captured: dict[str, object] = {}

        class Engine:
            backend = "numpy_cpu"
            device = "cpu"

            def __init__(self, include_flyvis_drive_echo: bool) -> None:
                self.include_flyvis_drive_echo = include_flyvis_drive_echo

        class Module:
            @staticmethod
            def create_engine(**kwargs: object) -> Engine:
                captured.update(kwargs)
                return Engine(bool(kwargs["include_flyvis_drive_echo"]))

        MalecnsNeuralAdapter(
            Module(), seed=1729, device="cpu", backend="cpu",
            include_flyvis_drive_echo=False,
        )
        self.assertFalse(captured["include_flyvis_drive_echo"])

    def test_neural_cuda_graph_is_forwarded_and_manifested_without_silent_status(self) -> None:
        captured: dict[str, object] = {}

        class Engine:
            backend = "triton"
            device = "cuda"

        class Module:
            @staticmethod
            def create_engine(**kwargs: object) -> Engine:
                captured.update(kwargs)
                return Engine()

        adapter = MalecnsNeuralAdapter(
            Module(), seed=1729, device="cuda", backend="triton",
            cuda_graph=True,
        )
        self.assertTrue(captured["cuda_graph"])
        status = adapter.cuda_graph_manifest()
        self.assertTrue(status["requested"])
        self.assertFalse(status["effective"])
        self.assertIn("no CUDA Graph status", str(status["reason"]))

    def test_escape_channels_strip_direction_labels_from_control_but_record_them(self) -> None:
        class Provider:
            def step(self, **context: object) -> dict[str, object]:
                return {
                    "channels": {"jump_drive": 0.25, "stimulus_direction": 12.0},
                    "label": "palm",
                    "direction": "left",
                    "future_state": {"wing": 1.0},
                    "frame_index": context["frame_index"],
                }

        adapter = EscapeChannelAdapter(Provider(), role="jump_channel")
        record = adapter.step(timestamp_s=0.0, dt_s=0.005, frame_index=0)
        self.assertEqual(record["channels"], {"jump_drive": 0.25})
        self.assertEqual(record["raw"]["label"], "palm")
        self.assertEqual(record["raw"]["direction"], "left")
        self.assertEqual(record["raw"]["future_state"], {"wing": 1.0})

    def test_missing_strict_escape_channels_fail_closed(self) -> None:
        with self.assertRaisesRegex(BackendSelectionError, "requires dynamic stimulus"):
            EscapeChannelSet.discover({}, required=True)

    def test_strict_compact_row_is_lossless_and_drops_invariant_debug_payload(self) -> None:
        """Compact production rows retain exact visual input once per frame."""

        timestamp = Timestamp(0.0, 0, "neural", 2000.0)
        left = np.linspace(0.0, 1.0, 721, dtype=np.float32)
        right = np.linspace(1.0, 0.0, 721, dtype=np.float32)
        drive = np.linspace(-1.0, 1.0, 13_582, dtype=np.float32)
        visual = types.SimpleNamespace(
            left_columns=left,
            right_columns=right,
            t4t5_drive=drive,
            backend="flyvis",
        )
        visual_observation = types.SimpleNamespace(
            optic_flow_rad_s=(0.1, -0.2, 0.0),
            sensor_id="virtual-world-stereo-flyvis",
            source=SourceLabel.FITTED,
            metadata={"backend": "flyvis"},
        )
        nm_provenance = {
            "channels": {
                f"channel_{index}": {"terms": [{"spec": "all", "description": "x" * 64}]}
                for index in range(64)
            },
            "map_sha256": "immutable-map",
        }
        nm = types.SimpleNamespace(
            to_dict=lambda: {
                "channels": {"ttmn_activity": 0.2, "jump": 0.0},
                "left_wing_power": 0.4,
                "right_wing_power": 0.5,
                "wing_power": 0.45,
                "wing_steering": 0.1,
                "leg_power": 0.0,
                "leg_steering": 0.0,
                "mouth": 0.0,
                "groom": 0.0,
                "dt_ms": 5.0,
                "source": "connectome",
                "saturated": False,
                "provenance": {
                    **nm_provenance,
                    "physical_wing": {
                        "source": "wing_motor",
                        "left_activation": 0.4,
                        "right_activation": 0.5,
                    },
                },
            }
        )
        contract = timestamp.to_dict()
        raw = {
            "scenario": "free-behavior",
            "frame_index": 0,
            "frame_time_s": 0.0,
            "time_s": 0.005,
            "dt_s": 0.005,
            "interval_start_s": 0.0,
            "interval_end_s": 0.005,
            "timestamp": contract,
            "position_m": [0.0, 0.0, 0.01],
            "velocity_mps": [0.0, 0.0, 0.0],
            "orientation_quat": [1.0, 0.0, 0.0, 0.0],
            "orientation_rad": [0.0, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
            "contact": True,
            "grounded": True,
            "target_contact": False,
            "ground_distance_m": 0.0,
            "flight_mode": "free_behavior",
            "controller_mode": "MaleCNS_direct",
            "left_wing": 0.4,
            "right_wing": 0.5,
            "wing_drive_left": 0.4,
            "wing_drive_right": 0.5,
            "yaw_rate": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "wind_mps": [0.0, 0.0, 0.0],
            "visual_drive_summary": {},
            "visual_directional_readout": "record_only",
            "stimulus_state": {"active": True},
            "stimulus_condition": "looming_grounded",
            "stimulus_name": "looming_grounded",
            "stimulus_contact": False,
            "stimulus_interaction": "visual_only_non_interacting",
            "escape_channels": {
                "timestamp_s": 0.0,
                "frame_index": 0,
                "channels": {"dynamic_stimulus": {"value": 0.2}},
                "records": {"dynamic_stimulus": {"raw": {"label": "palm"}}},
                "experiment": {"immutable": "x" * 1000},
                "provenance": {"causal_context": True},
            },
            "collision_count": 0,
            "body_before": {},
            "body": {},
            "sensory": {
                "channels": {
                    "flyvis_t4t5": {
                        "target_indices": list(range(drive.size)),
                        "values": drive.tolist(),
                    },
                    "flyvis_target_indices": list(range(drive.size)),
                }
            },
            "neural": {
                "timestamp": contract,
                "rates_hz": None,
                "spikes": None,
                "named_outputs": {"neural_window_steps": 10},
                "source": "connectome",
            },
            "neural_window": {"steps": 10},
            "jump_decision_telemetry": {"ttmn_activity": 0.2},
            "jump_input": {"ttmn_activity": 0.2},
            "jump_output": {"jump_actuator_active": False},
            "motor": {
                "timestamp": contract,
                "yaw_rate_rad_s": 0.0,
                "pitch_rate_rad_s": 0.0,
                "roll_rate_rad_s": 0.0,
                "thrust_mps2": 4.4,
                "source": "connectome",
                "saturated": False,
                "metadata": {"nm_out": nm_provenance, "auto_command": nm_provenance},
            },
            "diagnostics": {},
        }
        full_size = len(json.dumps({**raw, "nm_out": nm.to_dict()}, separators=(",", ":")))
        compact = _free_behavior_row(
            raw,
            mode="compact",
            flyvis_output=visual,
            visual_observation=visual_observation,
            nm_output=nm,
            mechanosensory=types.SimpleNamespace(to_dict=lambda: {
                "channels": {"wind_body_forward": 0.1},
                "population_drive": {"jo": 0.2},
                "raw_channels": {"wind_body_forward": 0.2},
                "population_metadata": nm_provenance,
            }),
            escape_packet=raw["escape_channels"],
            exact_sampling={"mode": "exact_world_eye_directions"},
        )
        self.assertEqual(compact["schema_version"], COMPACT_TELEMETRY_SCHEMA_VERSION)
        self.assertNotIn("provenance", compact)
        self.assertNotIn("world", compact)
        self.assertIsNone(compact["visual"]["pixels"])
        self.assertLess(len(json.dumps(compact, separators=(",", ":"))), full_size * 0.25)
        encoded = compact["visual_input"]["t4t5_drive"]
        decoded = np.frombuffer(zlib.decompress(base64.b64decode(encoded["data"])), dtype=np.float32)
        np.testing.assert_array_equal(decoded, drive)
        self.assertEqual(encoded["sha256"], __import__("hashlib").sha256(drive.tobytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
