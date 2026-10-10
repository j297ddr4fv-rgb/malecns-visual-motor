#!/usr/bin/env python3
"""Focused, dependency-free checks for the digital body backend."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from digital_body import (  # noqa: E402
    BodyCommand,
    BodyConfig,
    DigitalBody,
    OptionalBackendNotValidated,
    OptionalBackendUnavailable,
    detect_optional_backends,
)
from twin_contracts import BodyState as ContractBodyState  # noqa: E402
from twin_contracts import MotorCommand, MultiRateClock  # noqa: E402


class DigitalBodyTest(unittest.TestCase):
    def test_default_backend_is_deterministic_and_si_shaped(self) -> None:
        left = DigitalBody()
        right = DigitalBody()
        command = BodyCommand(wing_drive_left=1.0, wing_drive_right=0.9)
        for _ in range(40):
            a = left.step(command, dt_s=0.001)
            b = right.step(command, dt_s=0.001)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(len(a.position_m), 3)
        self.assertEqual(len(a.orientation_quat), 4)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in a.orientation_quat)), 1.0)
        self.assertIn("position_m", left.snapshot())
        self.assertEqual(left.snapshot()["units"]["position_m"], "m")

    def test_takeoff_flight_and_landing_modes_are_reachable(self) -> None:
        body = DigitalBody()
        seen: set[str] = set()
        for index in range(4000):
            drive = 1.0 if index < 1800 else 0.0
            seen.add(body.step({"left": drive, "right": drive}, dt_s=0.001).flight_mode)
        self.assertTrue({"takeoff", "flight", "landing", "grounded"}.issubset(seen))
        self.assertTrue(body.state.contact)
        self.assertAlmostEqual(body.state.position_m[2], body.backend.config.body_height_m)

    def test_snapshot_restore_preserves_a_touchdown_frame(self) -> None:
        body = DigitalBody()
        body.reset({"position_m": (0.0, 0.0, 0.02), "velocity_mps": (0.0, 0.0, -1.0)})
        for _ in range(100):
            state = body.step({"left": 0.0, "right": 0.0}, dt_s=0.001)
            if state.flight_mode == "landing":
                break
        self.assertEqual(state.flight_mode, "landing")
        snapshot = body.snapshot()
        restored = body.restore(snapshot)
        self.assertEqual(restored.to_dict(), snapshot)

    def test_invalid_commands_and_below_ground_reset_are_safe(self) -> None:
        with self.assertRaises(ValueError):
            BodyCommand(wing_drive_left=float("nan")).normalized()
        body = DigitalBody()
        state = body.reset({"position_m": (0.0, 0.0, -1.0), "velocity_mps": (0.0, 0.0, -1.0)})
        self.assertEqual(state.position_m[2], body.backend.config.body_height_m)
        self.assertEqual(state.velocity_mps[2], 0.0)
        self.assertTrue(state.contact)

    def test_optional_backends_are_explicitly_unvalidated(self) -> None:
        capabilities = detect_optional_backends()
        self.assertEqual(set(capabilities), {"mujoco", "flybody"})
        self.assertFalse(capabilities["mujoco"].validated)
        self.assertFalse(capabilities["flybody"].validated)
        for name, capability in capabilities.items():
            with self.subTest(name=name):
                try:
                    DigitalBody(name)
                except OptionalBackendUnavailable:
                    self.assertFalse(capability.installed)
                except OptionalBackendNotValidated:
                    self.assertTrue(capability.installed)
                else:  # pragma: no cover - optional backends must not be auto-used
                    self.fail(f"unvalidated backend {name} unexpectedly constructed")

    def test_shared_contract_body_command_timestamp_bridge(self) -> None:
        clock = MultiRateClock({"control": 200.0})
        t0 = clock.timestamp("control")
        contract_body = ContractBodyState(
            t0,
            position_m=(0.0, 0.0, 0.01),
            orientation_rad=(0.0, 0.0, 0.2),
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
        )
        body = DigitalBody()
        body.reset(contract_body)
        command = MotorCommand(
            t0,
            yaw_rate_rad_s=2.0,
            thrust_mps2=0.5,
        )
        t1 = clock.advance("control")
        state = body.step(command, timestamp=t1)
        self.assertAlmostEqual(state.time_s, t1.t_s)
        exported = body.contract_state(t1)
        self.assertEqual(exported.timestamp, t1)
        self.assertAlmostEqual(exported.position_m[2], state.position_m[2])
        self.assertAlmostEqual(exported.orientation_rad[2], state.orientation_rad[2], places=10)


if __name__ == "__main__":
    unittest.main()
