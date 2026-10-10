#!/usr/bin/env python3
"""Zero-dependency tests for the digital-twin contracts.

Run directly with ``python3 tests/test_twin_contracts.py`` or through pytest
when it is available.  The tests intentionally do not require numpy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by the dependency-free suite
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from twin_contracts import (  # noqa: E402
    BodyState,
    ContractError,
    MotorCommand,
    MultiRateClock,
    NeuralOutput,
    SensoryFrame,
    SourceLabel,
    StepDiagnostics,
    Timestamp,
    Unit,
    VisualObservation,
    WorldState,
)


class TwinContractsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = MultiRateClock({"neural": 2000.0, "control": 200.0})
        self.ts = self.clock.timestamp("neural")

    def test_multirate_ticks_are_explicit(self) -> None:
        self.assertEqual(self.ts.tick, 0)
        next_neural = self.clock.advance("neural", 1)
        self.assertEqual(next_neural.tick, 1)
        self.assertAlmostEqual(next_neural.t_s, 0.0005)
        control = self.clock.advance("control", 1)
        self.assertEqual(control.clock, "control")
        self.assertAlmostEqual(control.t_s, 0.005)
        self.assertAlmostEqual(self.clock.now_s, 0.005)

    def test_shape_and_finite_validation(self) -> None:
        body = BodyState(self.ts, position_m=[1, 2, 3])
        self.assertEqual(body.position_m, (1.0, 2.0, 3.0))
        with self.assertRaises(ContractError):
            BodyState(self.ts, position_m=[1, 2])
        with self.assertRaises(ContractError):
            VisualObservation(self.ts, pixels=[[1, 2], [3]])
        with self.assertRaises(ContractError):
            NeuralOutput(self.ts, rates_hz=[1, float("nan")])
        with self.assertRaises(ContractError):
            NeuralOutput(self.ts, rates_hz=[1, 2], spikes=[0])

    def test_records_round_trip_through_json(self) -> None:
        body = BodyState(self.ts, position_m=(0.01, 0.02, 0.03))
        world = WorldState(self.ts, body, world_id="lattice", geometry={"spacing_m": 0.04})
        obs = VisualObservation(self.ts, pixels=[[0.0, 0.5], [1.0, 0.25]],
                                source=SourceLabel.HANDBUILT)
        frame = SensoryFrame(self.ts, visual=obs, body=body, channels={"gyro": [0, 0, 0]})
        output = NeuralOutput(self.ts, rates_hz=[2.0, 4.0], spikes=[0, 1])
        command = MotorCommand(self.ts, yaw_rate_rad_s=1.2)
        diag = StepDiagnostics(self.ts, step_index=4, counters={"synapses": 12})

        records = (world, frame, output, command, diag)
        for record in records:
            encoded = record.to_json()
            self.assertIsInstance(json.loads(encoded), dict)
        restored = WorldState.from_dict(world.to_dict())
        self.assertEqual(restored.to_dict(), world.to_dict())
        self.assertEqual(SensoryFrame.from_dict(frame.to_dict()).to_dict(), frame.to_dict())
        self.assertEqual(NeuralOutput.from_dict(output.to_dict()).to_dict(), output.to_dict())
        self.assertEqual(MotorCommand.from_dict(command.to_dict()).to_dict(), command.to_dict())
        self.assertEqual(StepDiagnostics.from_dict(diag.to_dict()).to_dict(), diag.to_dict())

    def test_source_vocabulary_is_closed(self) -> None:
        self.assertEqual(MotorCommand(self.ts, source="fitted").source, SourceLabel.FITTED)
        with self.assertRaises(ContractError):
            MotorCommand(self.ts, source="learned-but-untracked")

    def test_neural_output_with_timestamp_reuses_validated_payload(self) -> None:
        rates = np.array([2.0, 4.0], dtype=np.float32) if np is not None else [2.0, 4.0]
        spikes = np.array([0, 1], dtype=np.uint8) if np is not None else [0, 1]
        named = {"population": {"wing": 1.0}}
        output = NeuralOutput(
            self.ts, rates_hz=rates, spikes=spikes,
            named_outputs=named, source=SourceLabel.FITTED,
        )
        later = self.clock.advance("neural", 1)

        retimestamped = output.with_timestamp(later)

        self.assertIsNot(retimestamped, output)
        self.assertEqual(output.timestamp, self.ts)
        self.assertEqual(retimestamped.timestamp, later)
        self.assertIs(retimestamped.rates_hz, rates)
        self.assertIs(retimestamped.spikes, spikes)
        self.assertIs(retimestamped.named_outputs, named)
        self.assertIs(retimestamped.source, output.source)

    def test_neural_output_with_timestamp_rejects_invalid_timestamp(self) -> None:
        output = NeuralOutput(self.ts, rates_hz=[2.0, 4.0], spikes=[0, 1])
        for invalid in (None, {"tick": 1}, "neural"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ContractError):
                    output.with_timestamp(invalid)  # type: ignore[arg-type]
        self.assertIs(output.timestamp, self.ts)

    def test_neural_output_with_timestamp_can_replace_named_outputs(self) -> None:
        output = NeuralOutput(self.ts, rates_hz=[2.0], spikes=[0], named_outputs={"old": 1})
        replacement = {"neural_window_steps": 10}
        retimestamped = output.with_timestamp(
            self.clock.advance("neural", 1), named_outputs=replacement,
        )
        self.assertIs(retimestamped.named_outputs, replacement)
        self.assertEqual(output.named_outputs, {"old": 1})
        with self.assertRaises(ContractError):
            output.with_timestamp(self.ts, named_outputs=[])  # type: ignore[arg-type]

    def test_neural_output_initial_invalid_data_still_fails_constructor(self) -> None:
        with self.assertRaises(ContractError):
            NeuralOutput(self.ts, rates_hz=[1.0, float("nan")])
        with self.assertRaises(ContractError):
            NeuralOutput(self.ts, rates_hz=[1.0, 2.0], spikes=[0])

    @unittest.skipUnless(np is not None, "numpy is optional")
    def test_validated_engine_array_factory_is_narrow_and_reuses_arrays(self) -> None:
        rates = np.array([0.0, 200.0], dtype=np.float32)
        spikes = np.array([0, 1], dtype=np.int32)
        named = {"neural_window_steps": 10}
        output = NeuralOutput._from_validated_engine_arrays(
            self.ts, rates_hz=rates, spikes=spikes,
            named_outputs=named, source=SourceLabel.CONNECTOME,
        )
        self.assertIs(output.rates_hz, rates)
        self.assertIs(output.spikes, spikes)
        self.assertIs(output.named_outputs, named)
        with self.assertRaises(ContractError):
            NeuralOutput._from_validated_engine_arrays(
                self.ts, rates_hz=rates.tolist(), spikes=spikes,
                named_outputs=named,
            )
        with self.assertRaises(ContractError):
            NeuralOutput._from_validated_engine_arrays(
                self.ts, rates_hz=rates,
                spikes=np.array([0.0, 1.0], dtype=np.float32),
                named_outputs=named,
            )

    def test_timestamps_must_match_across_bundle(self) -> None:
        later = self.clock.advance("neural", 1)
        body = BodyState(later)
        obs = VisualObservation(later, pixels=[[0]])
        with self.assertRaises(ContractError):
            WorldState(self.ts, body)
        with self.assertRaises(ContractError):
            SensoryFrame(self.ts, visual=obs)

    def test_json_classmethod_units_and_aliases(self) -> None:
        body = BodyState(self.ts, orientation_rad=(0.1, 0.2, 0.3))
        self.assertEqual(body.attitude_rpy_rad, (0.1, 0.2, 0.3))
        self.assertEqual(body.units["position_m"], "m")
        restored = BodyState.from_json(body.to_json())
        self.assertEqual(restored.to_dict(), body.to_dict())
        self.assertEqual(self.ts.domain, "neural")
        self.assertAlmostEqual(self.ts.dt_s, 0.0005)
        self.assertEqual(Unit.HERTZ.value, "Hz")

    def test_rejects_malformed_boundaries(self) -> None:
        with self.assertRaises(ContractError):
            Timestamp.from_json('{"tick": 0}')
        with self.assertRaises(ContractError):
            VisualObservation(self.ts, pixels=[[]])
        with self.assertRaises(ContractError):
            StepDiagnostics(self.ts, 0, warnings="not-a-list")

    @unittest.skipUnless(np is not None, "numpy is optional")
    def test_numpy_vectors_use_array_protocol_without_scalar_iteration(self) -> None:
        class ArrayLike:
            shape = (4,)

            def __init__(self) -> None:
                self.array_calls = 0

            def __array__(self, dtype=None):
                self.array_calls += 1
                return np.asarray((0.0, 1.0, 2.0, 3.0), dtype=dtype)

            def __iter__(self):
                raise AssertionError("array-like values must not be walked element by element")

        rates = ArrayLike()
        output = NeuralOutput(self.ts, rates_hz=rates)
        self.assertEqual(output.neuron_count, 4)
        self.assertEqual(rates.array_calls, 1)

        class NoScalarIteration(np.ndarray):
            def __iter__(self):
                raise AssertionError("ndarray values must not be walked element by element")

        spikes = np.zeros(176_422, dtype=np.uint8).view(NoScalarIteration)
        output = NeuralOutput(self.ts, spikes=spikes)
        self.assertEqual(output.neuron_count, 176_422)

    @unittest.skipUnless(np is not None, "numpy is optional")
    def test_numpy_vectors_validate_finite_nonnegative_and_real_dtype(self) -> None:
        valid = NeuralOutput(
            self.ts,
            rates_hz=np.array([0.0, 2.5], dtype=np.float32),
            spikes=np.array([0, 1], dtype=np.int32),
        )
        self.assertEqual(valid.neuron_count, 2)

        for bad in (
            np.array([0.0, np.nan]),
            np.array([0.0, np.inf]),
            np.array([0.0, -1.0]),
            np.array([True, False]),
            np.array([1.0 + 0.0j, 0.0 + 0.0j]),
            np.array([1.0, 2.0], dtype=object),
        ):
            with self.subTest(dtype=bad.dtype, values=bad):
                with self.assertRaises(ContractError):
                    NeuralOutput(self.ts, rates_hz=bad)

    def test_lists_keep_scalar_validation_and_ragged_rejection(self) -> None:
        with self.assertRaises(ContractError):
            NeuralOutput(self.ts, rates_hz=[True, False])
        with self.assertRaises(ContractError):
            NeuralOutput(self.ts, rates_hz=[1.0 + 0.0j])
        with self.assertRaises(ContractError):
            NeuralOutput(self.ts, rates_hz=[[1.0], [2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
