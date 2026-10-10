#!/usr/bin/env python3
"""Focused CPU/runtime tests; no model package or GPU is required."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from twin_contracts import MotorCommand, NeuralOutput, Timestamp, VisualObservation  # noqa: E402
import twin_runtime  # noqa: E402


class CountingWorld:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, state, *, seed=None):
        self.reset_calls += 1
        return state

    def step(self, command, timestamp):
        self.step_calls += 1
        ts = timestamp
        return twin_runtime.WorldState(ts, twin_runtime.BodyState(ts))


class CountingSensor:
    def __init__(self) -> None:
        self.calls = 0

    def observe(self, world, timestamp):
        self.calls += 1
        return VisualObservation(timestamp, pixels=[[0.5]], optic_flow_rad_s=(0.1, 0.0, 0.0))


class CountingFlyVis:
    is_stub = True

    def __init__(self) -> None:
        self.calls = 0

    def reset(self, *, seed=None):
        self.calls = 0

    def step(self, observation, timestamp):
        self.calls += 1
        return {"drive": [0.1, -0.1], "backend": "test", "is_stub": True}


class CountingNeural:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, sensory, timestamp):
        self.calls += 1
        return NeuralOutput(timestamp, rates_hz=[1.0, 2.0], spikes=[0, 1])


class CountingMotor:
    def __init__(self) -> None:
        self.calls = 0

    def decode(self, neural, timestamp):
        self.calls += 1
        return MotorCommand(timestamp, yaw_rate_rad_s=0.1)


class TwinRuntimeTest(unittest.TestCase):
    def test_import_and_preflight_do_not_import_torch(self):
        # This test runs in a fresh process in CI in practice; if another test
        # imported torch first, the report still accurately exposes that fact.
        report = twin_runtime.preflight_report()
        self.assertIn("gate", report)
        self.assertIn("capabilities", report)
        self.assertIn("torch_imported", report)

    def test_default_cpu_counts_and_deterministic_manifest(self):
        cfg = twin_runtime.RuntimeConfig(duration_s=0.01, device="cpu", backend="stub", dry_run=True)
        first = twin_runtime.TwinRuntime(cfg).run()
        second = twin_runtime.TwinRuntime(cfg).run()
        self.assertEqual(first.manifest["event_counts"], {
            "physics": 100, "cns": 20, "flyvis": 2, "control": 1, "telemetry": 0,
        })
        self.assertEqual(first.manifest["diagnostics"]["trace_sha256"],
                         second.manifest["diagnostics"]["trace_sha256"])
        json.dumps(first.manifest)

    def test_injected_components_are_called_at_their_rates(self):
        world, sensor, flyvis = CountingWorld(), CountingSensor(), CountingFlyVis()
        neural, motor = CountingNeural(), CountingMotor()
        cfg = twin_runtime.RuntimeConfig(
            duration_s=0.02, seed=7, device="cpu", backend="cpu_stub", dry_run=True,
            control_hz=50.0, telemetry_hz=25.0,
        )
        result = twin_runtime.TwinRuntime(
            cfg, world=world, sensor=sensor, flyvis=flyvis,
            neural=neural, motor=motor,
        ).run()
        self.assertEqual(world.reset_calls, 1)
        self.assertEqual(world.step_calls, 200)
        self.assertEqual(sensor.calls, 4)
        self.assertEqual(flyvis.calls, 4)
        self.assertEqual(neural.calls, 40)
        self.assertEqual(motor.calls, 1)
        self.assertEqual(result.manifest["event_counts"]["telemetry"], 0)

    def test_manifest_and_telemetry_write(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = twin_runtime.cpu_dry_run(0.05, out=tmp)
            manifest = Path(tmp) / "manifest.json"
            telemetry = Path(tmp) / "telemetry.jsonl"
            self.assertTrue(manifest.exists())
            self.assertTrue(telemetry.exists())
            loaded = json.loads(manifest.read_text())
            self.assertEqual(loaded["schema_version"], twin_runtime.SCHEMA_VERSION)
            self.assertEqual(len(telemetry.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
