#!/usr/bin/env python3
"""Focused, dependency-free tests for the flight-autonomy boundary."""
from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from flight_autopilot import (  # noqa: E402
    AutopilotConfig,
    BodyState,
    FlightAutopilot,
    FlightMode,
    MotorCommand,
    SourceLabel,
    TargetGroundInfo,
)


class FlightAutopilotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ground = BodyState(grounded=True)
        self.air = BodyState(altitude=1.0, airspeed=0.5, vertical_speed=0.1, grounded=False)
        self.target = TargetGroundInfo(target_altitude=1.0)

    def _to_cruise(self, controller: FlightAutopilot) -> None:
        controller.step({"left": 0.8, "right": 0.8, "thrust": 0.8}, {}, self.ground, self.target)
        command = controller.step({"left": 0.4, "right": 0.5, "thrust": 0.6}, {}, self.air, self.target)
        self.assertEqual(command.mode, FlightMode.CRUISE)

    def test_all_modes_and_normal_neural_priority(self) -> None:
        controller = FlightAutopilot(config=AutopilotConfig(recover_clear_s=0.0))
        grounded = controller.step({"thrust": 0.0}, {}, self.ground, self.target)
        takeoff = controller.step({"left": 2.0, "right": -2.0, "thrust": 2.0}, {}, self.ground, self.target)
        cruise = controller.step({"left": 0.4, "right": 0.5, "thrust": 0.6}, {}, self.air, self.target)
        turn = controller.step({"left": 0.4, "right": 0.5, "thrust": 0.6}, {"direction": 0.8}, self.air, self.target)
        cruise_again = controller.step({"left": 0.4, "right": 0.5, "thrust": 0.6}, {"direction": 0.0}, self.air, self.target)
        approach = controller.step({"thrust": 0.4}, {}, self.air,
                                   TargetGroundInfo(target_altitude=1.0, target_distance=1.0))
        land = controller.step({"thrust": 0.2}, {},
                               BodyState(altitude=0.1, airspeed=0.05, grounded=False),
                               TargetGroundInfo(target_distance=0.1))
        done = controller.step({}, {}, BodyState(grounded=True), TargetGroundInfo(contact=True))

        self.assertEqual(
            [c.mode for c in (grounded, takeoff, cruise, turn, cruise_again, approach, land, done)],
            [FlightMode.GROUNDED, FlightMode.TAKEOFF, FlightMode.CRUISE, FlightMode.TURN,
             FlightMode.CRUISE, FlightMode.APPROACH, FlightMode.LAND, FlightMode.GROUNDED],
        )
        self.assertEqual(takeoff.source, SourceLabel.FITTED.value)
        self.assertEqual(takeoff.left_wing, 1.0)
        self.assertEqual(takeoff.right_wing, -1.0)
        self.assertEqual(takeoff.thrust, 1.0)
        self.assertEqual(done.source, SourceLabel.FITTED.value)

    def test_takeoff_ground_runup_liftoff_loss_of_thrust_and_landing(self) -> None:
        """TAKEOFF must dwell on the ground without fabricating a command."""

        controller = FlightAutopilot(
            config=AutopilotConfig(takeoff_thrust_threshold=0.5),
            safety_enabled=False,
        )
        takeoff_channels = {
            "left": 0.31,
            "right": -0.22,
            "thrust": 0.70,
            "yaw": 0.13,
            "pitch": -0.07,
            "roll": 0.05,
            "brake": 0.02,
        }

        runup_first = controller.step(takeoff_channels, {}, self.ground, self.target)
        runup_second = controller.step(takeoff_channels, {}, self.ground, self.target)
        self.assertEqual(runup_first.mode, FlightMode.TAKEOFF)
        self.assertEqual(runup_second.mode, FlightMode.TAKEOFF)
        for command in (runup_first, runup_second):
            self.assertEqual(command.source, SourceLabel.FITTED.value)
            self.assertEqual(command.actuator_dict(), {
                "left_wing": 0.31,
                "right_wing": -0.22,
                "thrust": 0.70,
                "yaw": 0.13,
                "pitch": -0.07,
                "roll": 0.05,
                "brake": 0.02,
            })

        # Removing neural thrust while still on the ground cancels the
        # request; the state machine does not retain TAKEOFF by itself.
        grounded_after_abort = controller.step(
            {"thrust": 0.0}, {}, self.ground, self.target,
        )
        self.assertEqual(grounded_after_abort.mode, FlightMode.GROUNDED)
        self.assertEqual(grounded_after_abort.thrust, 0.0)

        # A later thrust request can arm TAKEOFF again, and real liftoff keeps
        # the normal TAKEOFF progression to flight.
        rearmed = controller.step(takeoff_channels, {}, self.ground, self.target)
        liftoff = controller.step(
            takeoff_channels, {},
            BodyState(altitude=0.20, airspeed=0.20, vertical_speed=0.20, grounded=False),
            self.target,
        )
        cruise = controller.step(
            takeoff_channels, {},
            BodyState(altitude=1.0, airspeed=0.35, vertical_speed=0.10, grounded=False),
            self.target,
        )
        self.assertEqual(rearmed.mode, FlightMode.TAKEOFF)
        self.assertEqual(liftoff.mode, FlightMode.TAKEOFF)
        self.assertEqual(cruise.mode, FlightMode.CRUISE)

        approach = controller.step(
            takeoff_channels, {},
            BodyState(altitude=1.0, airspeed=0.30, grounded=False),
            TargetGroundInfo(target_altitude=1.0, target_distance=1.0),
        )
        land = controller.step(
            takeoff_channels, {},
            BodyState(altitude=0.10, airspeed=0.05, grounded=False),
            TargetGroundInfo(target_altitude=1.0, target_distance=0.10),
        )
        landed = controller.step(
            {"thrust": 0.0}, {}, BodyState(grounded=True),
            TargetGroundInfo(contact=True),
        )
        self.assertEqual(approach.mode, FlightMode.APPROACH)
        self.assertEqual(land.mode, FlightMode.LAND)
        self.assertEqual(landed.mode, FlightMode.GROUNDED)

    def test_ground_runup_timeout_enters_recover(self) -> None:
        controller = FlightAutopilot(
            config=AutopilotConfig(takeoff_thrust_threshold=0.5, takeoff_timeout_s=0.04),
            safety_enabled=False,
        )
        target = TargetGroundInfo(target_altitude=1.0)
        controller.step({"thrust": 0.8}, {}, self.ground, target, dt_s=0.02)
        controller.step({"thrust": 0.8}, {}, self.ground, target, dt_s=0.02)
        timed_out = controller.step({"thrust": 0.8}, {}, self.ground, target, dt_s=0.02)
        self.assertEqual(timed_out.mode, FlightMode.RECOVER)
        self.assertEqual(timed_out.source, SourceLabel.FITTED.value)
        self.assertAlmostEqual(timed_out.thrust, 0.8)

    def test_stall_safety_is_optional_and_provenance_is_explicit(self) -> None:
        protected = FlightAutopilot(
            config=AutopilotConfig(recover_clear_s=0.0, stall_grace_s=0.0),
            safety_enabled=True,
        )
        self._to_cruise(protected)
        recovery = protected.step({"left": 0.2, "right": 0.2, "thrust": 0.2}, {},
                                   BodyState(altitude=1.0, airspeed=0.01, grounded=False), self.target)
        self.assertEqual(recovery.mode, FlightMode.RECOVER)
        self.assertEqual(recovery.source, SourceLabel.SAFETY_HANDBUILT.value)
        self.assertTrue(recovery.safety_active)
        self.assertEqual(recovery.safety_reason, "stall")
        self.assertGreater(recovery.thrust, 0.2)
        self.assertLess(recovery.thrust, 1.0)
        self.assertAlmostEqual(recovery.safety_blend, protected.config.safety_blend)
        self.assertEqual(recovery.neural_source, SourceLabel.CONNECTOME.value)
        self.assertEqual(recovery.readout_source, SourceLabel.FITTED.value)
        self.assertEqual(recovery.safety_source, SourceLabel.SAFETY_HANDBUILT.value)
        self.assertEqual(recovery.provenance["cns"], SourceLabel.CONNECTOME.value)
        self.assertEqual(recovery.provenance["readout"], SourceLabel.FITTED.value)
        self.assertEqual(recovery.provenance["safety"], SourceLabel.SAFETY_HANDBUILT.value)

        ablated = FlightAutopilot(safety_enabled=False)
        self._to_cruise(ablated)
        command = ablated.step({"left": 0.2, "right": 0.2, "thrust": 0.2}, {},
                               BodyState(altitude=1.0, airspeed=0.01, grounded=False), self.target)
        self.assertEqual(command.mode, FlightMode.HOVER)
        self.assertEqual(command.source, SourceLabel.FITTED.value)
        self.assertFalse(command.safety_active)
        self.assertAlmostEqual(command.thrust, 0.2)

    def test_numeric_protection_and_ablation(self) -> None:
        protected = FlightAutopilot(safety_enabled=True)
        command = protected.step({"left": float("nan"), "thrust": float("inf")}, {}, self.air, self.target)
        self.assertEqual(command.mode, FlightMode.RECOVER)
        self.assertEqual(command.source, SourceLabel.SAFETY_HANDBUILT.value)
        self.assertEqual(command.safety_reason, "numeric_divergence")
        self.assertEqual(command.safety_blend, 1.0)
        self.assertTrue(all(math.isfinite(v) for v in command.actuator_dict().values()))

        ablated = FlightAutopilot(safety_enabled=False)
        command = ablated.step({"left": float("nan"), "thrust": float("inf")}, {}, self.air, self.target)
        self.assertEqual(command.source, SourceLabel.FITTED.value)
        self.assertFalse(command.safety_active)
        self.assertTrue(all(math.isfinite(v) for v in command.actuator_dict().values()))

    def test_airborne_start_is_takeoff_then_hover_not_immediate_recover(self) -> None:
        """A low-speed first observation is a takeoff transient, not a stall."""

        controller = FlightAutopilot(safety_enabled=True)
        target = TargetGroundInfo(target_altitude=0.35)
        body = BodyState(altitude=0.35, airspeed=0.0, vertical_speed=0.0, grounded=False)

        first = controller.step({"left": 0.65, "right": 0.65, "thrust": 0.65}, {}, body, target)
        self.assertEqual(first.mode, FlightMode.TAKEOFF)
        self.assertEqual(first.source, SourceLabel.FITTED.value)
        self.assertFalse(first.safety_active)

        modes = [first.mode]
        for _ in range(100):  # two seconds at the nominal 50 Hz control rate
            command = controller.step(
                {"left": 0.65, "right": 0.65, "thrust": 0.65}, {}, body, target,
                dt_s=0.02,
            )
            modes.append(command.mode)
            self.assertEqual(command.source, SourceLabel.FITTED.value)
            self.assertFalse(command.safety_active)
            self.assertEqual(command.safety_blend, 0.0)

        self.assertIn(FlightMode.HOVER, modes)
        self.assertNotIn(FlightMode.RECOVER, modes)
        self.assertEqual(controller.logs[-1].mode, FlightMode.HOVER)

    def test_deterministic_takeoff_flight_hover_landing_transitions(self) -> None:
        controller = FlightAutopilot(
            config=AutopilotConfig(stall_grace_s=0.20), safety_enabled=True,
        )
        target = TargetGroundInfo(target_altitude=1.0)
        commands = []
        commands.append(controller.step({"thrust": 0.8}, {}, self.ground, target))
        commands.append(controller.step(
            {"thrust": 0.8}, {},
            BodyState(altitude=0.45, airspeed=0.35, vertical_speed=0.35, grounded=False), target,
        ))
        commands.append(controller.step(
            {"thrust": 0.65}, {},
            BodyState(altitude=1.0, airspeed=0.35, vertical_speed=0.1, grounded=False), target,
        ))
        commands.append(controller.step(
            {"thrust": 0.65}, {},
            BodyState(altitude=1.0, airspeed=0.02, vertical_speed=0.0, grounded=False), target,
        ))
        commands.append(controller.step(
            {"thrust": 0.65}, {},
            BodyState(altitude=1.0, airspeed=0.30, vertical_speed=0.0, grounded=False), target,
        ))
        commands.append(controller.step(
            {"thrust": 0.55}, {},
            BodyState(altitude=1.0, airspeed=0.30, vertical_speed=0.0, grounded=False),
            TargetGroundInfo(target_altitude=1.0, target_distance=1.0),
        ))
        commands.append(controller.step(
            {"thrust": 0.25}, {},
            BodyState(altitude=0.10, airspeed=0.05, vertical_speed=-0.1, grounded=False),
            TargetGroundInfo(target_altitude=1.0, target_distance=0.10),
        ))
        commands.append(controller.step(
            {"thrust": 0.0}, {}, BodyState(grounded=True), TargetGroundInfo(contact=True),
        ))

        self.assertEqual(
            [command.mode for command in commands],
            [
                FlightMode.TAKEOFF, FlightMode.TAKEOFF, FlightMode.CRUISE,
                FlightMode.HOVER, FlightMode.CRUISE, FlightMode.APPROACH,
                FlightMode.LAND, FlightMode.GROUNDED,
            ],
        )
        self.assertTrue(all(command.source == SourceLabel.FITTED.value for command in commands))
        self.assertTrue(all(not command.safety_active for command in commands))

    def test_explicit_mission_phase_overrides_visual_target_and_ground_runup(self) -> None:
        """A mission phase is mode intent; contact prevents landing re-takeoff."""

        controller = FlightAutopilot(safety_enabled=False)
        body = BodyState(altitude=0.30, airspeed=0.40, grounded=False)
        grounded = BodyState(altitude=0.015, airspeed=0.40, grounded=True)
        target = {
            "target_altitude": 0.30,
            # Deliberately keep a large heading error and a visible/near target
            # throughout the sequence.  Neither may override the phase clock.
            "target_heading": math.pi,
            "target_distance": 0.20,
            "mission_phase": "forward",
        }

        forward = controller.step({"thrust": 0.8}, {"direction": 0.0}, body, target)
        turn = controller.step(
            {"thrust": 0.8}, {"direction": 0.0}, body,
            {**target, "mission_phase": "turn"},
        )
        hover = controller.step(
            {"thrust": 0.8}, {"direction": 0.0}, body,
            {**target, "mission_phase": "hover"},
        )
        gust = controller.step(
            {"thrust": 0.8}, {"direction": 0.0}, body,
            {**target, "mission_phase": "gust-recovery"},
        )
        approach = controller.step(
            {"thrust": 0.8}, {"direction": 0.0}, body,
            {**target, "mission_phase": "approach"},
        )
        # Contact is explicit, but neural thrust remains above the normal
        # ground-runup threshold.  Landing/settle phases must not re-arm.
        grounded_approach = controller.step(
            {"thrust": 0.8}, {"direction": 0.0}, grounded,
            {**target, "mission_phase": "approach"},
        )
        grounded_landing = controller.step(
            {"thrust": 0.8}, {"direction": 0.0}, grounded,
            {**target, "mission_phase": "landing"},
        )
        landing_airborne = controller.step(
            {"thrust": 0.8}, {"direction": 0.0}, body,
            {**target, "mission_phase": "landing"},
        )
        settled = controller.step(
            {"thrust": 0.8}, {"direction": 0.0}, grounded,
            {**target, "mission_phase": "settle"},
        )

        self.assertEqual(
            [command.mode for command in (
                forward, turn, hover, gust, approach,
                grounded_approach, grounded_landing, landing_airborne, settled,
            )],
            [
                FlightMode.CRUISE, FlightMode.TURN, FlightMode.HOVER,
                FlightMode.HOVER, FlightMode.APPROACH, FlightMode.GROUNDED,
                FlightMode.GROUNDED, FlightMode.LAND, FlightMode.GROUNDED,
            ],
        )
        # Phase arbitration is telemetry only; fitted neural output remains
        # the command source and actuator values are untouched.
        self.assertTrue(all(command.source == SourceLabel.FITTED.value for command in (
            forward, turn, hover, gust, approach,
            grounded_approach, grounded_landing, landing_airborne, settled,
        )))
        self.assertTrue(all(command.thrust == 0.8 for command in (
            forward, turn, hover, gust, approach,
            grounded_approach, grounded_landing, landing_airborne, settled,
        )))

    def test_multi_second_hover_keeps_readout_primary_and_bounded(self) -> None:
        controller = FlightAutopilot(safety_enabled=True)
        target = TargetGroundInfo(target_altitude=0.35)
        body = BodyState(altitude=0.35, airspeed=0.0, vertical_speed=0.0, grounded=False)

        commands = [
            controller.step({"left": 0.5, "right": 0.55, "thrust": 0.6}, {}, body, target),
        ]
        for _ in range(149):
            commands.append(controller.step(
                {"left": 0.5, "right": 0.55, "thrust": 0.6}, {}, body, target,
                dt_s=0.02,
            ))

        self.assertEqual(len(commands), 150)
        self.assertGreaterEqual(controller.time_s, 2.98)
        self.assertTrue(all(command.mode is FlightMode.HOVER for command in commands[1:]))
        self.assertTrue(all(command.source == SourceLabel.FITTED.value for command in commands))
        self.assertTrue(all(command.provenance["cns"] == SourceLabel.CONNECTOME.value for command in commands))
        self.assertTrue(all(command.provenance["readout"] == SourceLabel.FITTED.value for command in commands))
        self.assertTrue(all(command.provenance["safety"] == "none" for command in commands))
        self.assertTrue(all(
            -1.0 <= value <= 1.0 if name in {"left_wing", "right_wing", "yaw", "pitch", "roll"}
            else 0.0 <= value <= 1.0
            for command in commands for name, value in command.actuator_dict().items()
        ))

    def test_timeout_and_log_previous_mode(self) -> None:
        config = AutopilotConfig(takeoff_timeout_s=0.04, state_timeout_s=10.0)
        controller = FlightAutopilot(config=config, safety_enabled=False)
        controller.step({"thrust": 0.5}, {}, self.ground, self.target)
        controller.step({"thrust": 0.5}, {}, BodyState(altitude=0.2, airspeed=0.5, grounded=False), self.target,
                        dt_s=0.02)
        command = controller.step({"thrust": 0.5}, {}, BodyState(altitude=0.2, airspeed=0.5, grounded=False), self.target,
                                  dt_s=0.02)
        self.assertEqual(command.mode, FlightMode.RECOVER)
        self.assertEqual(controller.logs[-1].previous_mode, FlightMode.TAKEOFF)
        self.assertEqual(controller.logs[-1].transition, "TAKEOFF->RECOVER")

    def test_twin_contract_adapter_does_not_mutate_coordinates(self) -> None:
        try:
            from twin_contracts import BodyState as ContractBodyState, MultiRateClock
        except ImportError:  # pragma: no cover - useful when copied standalone
            self.skipTest("twin_contracts is not on the standalone path")
        clock = MultiRateClock({"control": 200.0})
        timestamp = clock.timestamp("control")
        contract_body = ContractBodyState(
            timestamp,
            position_m=(2.0, 3.0, 1.0),
            orientation_rad=(0.1, 0.2, 0.3),
            velocity_mps=(0.3, 0.4, 0.2),
            angular_velocity_rad_s=(0.01, 0.02, 0.03),
        )
        before = contract_body.to_dict()
        controller = FlightAutopilot(safety_enabled=False)
        command = controller.step({"thrust": 0.5}, {}, contract_body, self.target, dt_s=0.005)
        self.assertEqual(contract_body.to_dict(), before)
        self.assertAlmostEqual(BodyState.from_contract(contract_body).altitude, 1.0)
        restored = command.to_contract(timestamp, thrust_limit_mps2=2.0)
        self.assertEqual(restored.source.value, "fitted")
        self.assertAlmostEqual(restored.thrust_mps2, 1.0)
        self.assertEqual(restored.metadata["neural_source"], SourceLabel.CONNECTOME.value)
        self.assertEqual(restored.metadata["readout_source"], SourceLabel.FITTED.value)
        self.assertEqual(restored.metadata["safety_blend"], 0.0)


if __name__ == "__main__":
    unittest.main()
