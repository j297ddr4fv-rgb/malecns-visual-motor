#!/usr/bin/env python3
"""Focused tests for the dependency-free reduced flight model."""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from digital_body import BodyCommand, BodyState  # noqa: E402
from flight_dynamics import (  # noqa: E402
    ContactSurface,
    FlightDynamics,
    FlightDynamicsConfig,
    FlightDynamicsError,
    FlightState,
    NeuromuscularInput,
    JumpLegActuator,
    JumpLegConfig,
    JumpNeuromuscularInput,
    ReducedBodyJumpDynamics,
    WingLegFlightDynamics,
    WingbeatPatternGenerator,
    self_test,
)
from twin_contracts import MotorCommand, MultiRateClock, NeuralOutput  # noqa: E402


class FlightDynamicsTest(unittest.TestCase):
    def test_wpg_is_in_requested_band(self) -> None:
        wpg = WingbeatPatternGenerator()
        self.assertGreaterEqual(wpg.frequency_hz, 200.0)
        self.assertLessEqual(wpg.frequency_hz, 220.0)
        phase = wpg.phase_rad
        wpg.advance(0.001)
        self.assertNotEqual(wpg.phase_rad, phase)

    def test_hover_trim_and_symmetry(self) -> None:
        model = FlightDynamics()
        trim = model.hover_trim()
        result = model.evaluate(trim)
        self.assertGreater(trim.left_power, 0.0)
        self.assertAlmostEqual(trim.left_power, trim.right_power)
        self.assertAlmostEqual(result.left_lift_N, result.right_lift_N)
        self.assertAlmostEqual(result.torque_body_Nm[0], 0.0, places=15)
        self.assertAlmostEqual(result.torque_body_Nm[2], 0.0, places=15)
        self.assertAlmostEqual(result.total_force_body_N[2], 0.0, places=10)
        self.assertGreater(result.profile_drag_N, 0.0)

    def test_differential_wing_power_produces_yaw(self) -> None:
        model = FlightDynamics()
        positive = model.evaluate({"left_power": 0.4, "right_power": 0.9})
        self.assertGreater(positive.right_lift_N, positive.left_lift_N)
        self.assertGreater(positive.torque_body_Nm[2], 0.0)
        self.assertLess(positive.torque_body_Nm[0], 0.0)

        negative = model.evaluate({"left_power": 0.9, "right_power": 0.4})
        self.assertLess(negative.torque_body_Nm[2], 0.0)
        self.assertGreater(negative.torque_body_Nm[0], 0.0)

    def test_input_contracts_and_body_output(self) -> None:
        clock = MultiRateClock({"control": 200.0})
        timestamp = clock.timestamp("control")
        command = MotorCommand(timestamp, thrust_mps2=0.5, yaw_rate_rad_s=0.1,
                               source="fitted")
        with self.assertRaisesRegex(FlightDynamicsError, "thrust_mps2.*normalized"):
            NeuromuscularInput.from_any(command)
        with self.assertRaisesRegex(FlightDynamicsError, "thrust_mps2.*normalized"):
            NeuromuscularInput.from_any({"thrust_mps2": 0.5, "yaw": 0.1})

        # A normalized command remains accepted through the mapping boundary;
        # this is the shape emitted by the runner's ``auto_command.as_dict``.
        parsed = NeuromuscularInput.from_any({"thrust": 0.5, "yaw": 0.1})
        self.assertAlmostEqual(parsed.left_power, 0.5)
        self.assertAlmostEqual(parsed.right_power, 0.5)
        self.assertAlmostEqual(parsed.yaw, 0.1)
        neural = NeuralOutput(timestamp, rates_hz=[100.0, 150.0])
        neural_input = NeuromuscularInput.from_any(neural)
        self.assertAlmostEqual(neural_input.left_power, 0.5)
        self.assertAlmostEqual(neural_input.right_power, 0.75)

        body_command = BodyCommand.from_any(
            {"wing_drive": (0.4, 0.6), "wingbeat_frequency_hz": 210.0}
        )
        self.assertAlmostEqual(body_command.wing_drive_left, 0.4)
        self.assertAlmostEqual(body_command.wing_drive_right, 0.6)

        model = FlightDynamics(state=BodyState())
        output = model.step(body_command, dt_s=0.001)
        adapted = BodyCommand.from_any(output.body_command())
        self.assertAlmostEqual(adapted.wing_drive_left, 0.4)
        self.assertAlmostEqual(adapted.wing_drive_right, 0.6)
        backend_state = output.to_digital_body_state()
        self.assertEqual(backend_state.position_m, output.state.position_m)
        self.assertFalse(output.is_cfd)
        self.assertEqual(output.model_scope, "reduced 6-DOF wingbeat-averaged model; not CFD")

        # The unified contract's timestamp can drive the body-rate step too.
        model.reset()
        t1 = clock.advance("control")
        timestamped = model.step({"thrust": 0.5, "yaw": 0.1}, timestamp=t1)
        self.assertAlmostEqual(timestamped.time_s, t1.t_s)
        model.reset()
        with self.assertRaisesRegex(FlightDynamicsError, "thrust_mps2.*normalized"):
            model.step(MotorCommand(t1, thrust_mps2=0.2, source="fitted"))

    def test_marked_normalized_wing_output_matches_legacy_mapping_path(self) -> None:
        class MarkedOutput:
            flight_dynamics_contract = "normalized-wing-output-v1"

            def __init__(self) -> None:
                self.channels = {
                    "wing_motor_left": 0.82,
                    "wing_motor_right": 0.27,
                }
                self.left_wing_power = 0.11
                self.right_wing_power = 0.19
                self.wing_steering = -0.23
                self.source = "connectome"

        marked = MarkedOutput()
        config = FlightDynamicsConfig(
            prefer_physical_wing_motor=True,
            allow_legacy_wing_fallback=True,
        )
        legacy_mapping = {
            "channels": marked.channels,
            "left_wing_power": marked.left_wing_power,
            "right_wing_power": marked.right_wing_power,
            "wing_steering": marked.wing_steering,
            "source": marked.source,
        }
        expected = NeuromuscularInput.from_any(legacy_mapping, config=config)
        actual = NeuromuscularInput.from_any(marked, config=config)
        self.assertEqual(actual, expected)

    def test_marked_normalized_wing_output_strict_fallback_keeps_missing_motors_zero(self) -> None:
        class MarkedOutput:
            flight_dynamics_contract = "normalized-wing-output-v1"

            channels = {}
            left_wing_power = 0.71
            right_wing_power = 0.63
            wing_steering = 0.12
            source = "connectome"

        config = FlightDynamicsConfig(
            prefer_physical_wing_motor=True,
            allow_legacy_wing_fallback=False,
        )
        actual = NeuromuscularInput.from_any(MarkedOutput(), config=config)
        expected = NeuromuscularInput.from_any(
            {
                "channels": {},
                "left_wing_power": 0.71,
                "right_wing_power": 0.63,
                "wing_steering": 0.12,
                "source": "connectome",
            },
            config=config,
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual.left_power, 0.0)
        self.assertEqual(actual.right_power, 0.0)

    def test_wind_drag_and_energy_are_finite(self) -> None:
        model = FlightDynamics(config=FlightDynamicsConfig(wind_mps=(1.0, 0.0, 0.0)))
        output = model.step({"left_power": 0.5, "right_power": 0.5}, dt_s=0.001)
        self.assertGreater(abs(output.wind_force_body_N[0]), 0.0)
        self.assertGreater(output.power_W, 0.0)
        self.assertGreater(output.energy_increment_J, 0.0)
        self.assertTrue(all(value == value for value in output.total_force_body_N))

    def test_stable_step_self_test(self) -> None:
        result = self_test(steps=100, dt_s=0.001)
        self.assertTrue(result["ok"], result)

    def test_state_mapping_compatibility(self) -> None:
        state = FlightState.from_any({
            "position": (0.0, 0.0, 0.01),
            "velocity": (0.1, 0.0, 0.0),
            "orientation_rad": (0.0, 0.0, 0.2),
            "angular_velocity": (0.0, 0.0, 0.0),
        })
        self.assertEqual(state.position_m, (0.0, 0.0, 0.01))
        self.assertEqual(state.velocity_mps, (0.1, 0.0, 0.0))

        model = FlightDynamics(state=state)
        clock = MultiRateClock({"control": 200.0})
        contract_state = model.step({"left_power": 0.2, "right_power": 0.2}).to_twin_body_state(
            clock.timestamp("control")
        )
        self.assertEqual(contract_state.position_m, model.state.position_m)
        self.assertAlmostEqual(contract_state.orientation_rad[2], 0.2, places=6)

    def test_contact_uses_configured_body_centre_plane(self) -> None:
        """Contact resolution is at the configured centre-floor, not z=0."""

        floor = 0.015
        model = FlightDynamics(
            config=FlightDynamicsConfig(ground_plane_z_m=floor),
            state=FlightState(position_m=(0.0, 0.0, 0.0), velocity_mps=(0.0, 0.0, -1.0)),
        )
        self.assertAlmostEqual(model.state.position_m[2], floor, places=12)
        self.assertAlmostEqual(model.state.velocity_mps[2], 0.0, places=12)
        output = model.step({"left_power": 0.0, "right_power": 0.0}, dt_s=0.005)

        self.assertAlmostEqual(output.state.position_m[2], floor, places=12)
        self.assertAlmostEqual(output.state.velocity_mps[2], 0.0, places=12)
        self.assertGreater(output.contact_force_body_N[2], 0.0)
        self.assertTrue(output.to_body_state()["contact"])

    def test_raised_contact_surface_catches_only_downward_entry_from_above(self) -> None:
        surface = ContactSurface(
            surface_id="landing_target",
            min_x_m=0.69,
            max_x_m=1.01,
            min_y_m=-0.16,
            max_y_m=0.16,
            top_z_m=0.055,
        )
        model = FlightDynamics(
            config=FlightDynamicsConfig(contact_surfaces=(surface,)),
            state=FlightState(
                position_m=(0.85, 0.0, 0.060),
                velocity_mps=(0.0, 0.0, -2.0),
            ),
        )
        output = model.step({"left_power": 0.0, "right_power": 0.0}, dt_s=0.005)

        self.assertAlmostEqual(output.state.position_m[2], 0.055, places=12)
        self.assertAlmostEqual(output.state.velocity_mps[2], 0.0, places=12)
        self.assertTrue(output.target_contact)
        self.assertEqual(output.contact_surface_id, "landing_target")
        self.assertGreater(output.contact_force_body_N[2], 0.0)
        self.assertEqual(output.state.position_m[:2], (0.85, 0.0))

        model.reset({
            "position_m": (0.85, 0.0, 0.045),
            "velocity_mps": (0.0, 0.0, 1.0),
        })
        rising = model.step({"left_power": 0.0, "right_power": 0.0}, dt_s=0.005)
        self.assertLess(rising.state.position_m[2], 0.055)
        self.assertFalse(rising.target_contact)

    def test_multi_second_bounded_trajectory_takeoff_turn_gust_and_land(self) -> None:
        """Exercise the continuous body model rather than one-step force checks."""

        model = FlightDynamics(state=FlightState(position_m=(0.0, 0.0, 0.0)))
        trim = model.hover_trim()
        outputs = []

        def run(steps: int, command: object) -> None:
            for _ in range(steps):
                output = model.step(command, dt_s=0.001)
                values = (
                    *output.state.position_m,
                    *output.state.velocity_mps,
                    *output.state.orientation_quat,
                    *output.state.angular_velocity_rad_s,
                    output.power_W,
                    output.cumulative_energy_J,
                )
                self.assertTrue(all(math.isfinite(float(value)) for value in values))
                outputs.append(output)

        # Full symmetric drive leaves the ground and builds useful altitude.
        run(300, {"left_power": 1.0, "right_power": 1.0})
        takeoff_height = outputs[-1].state.position_m[2]
        self.assertGreater(takeoff_height, 0.20)

        # The trim is a true force balance, so the body remains airborne while
        # the controller has no scripted position target.
        run(700, trim)
        hover_height = outputs[-1].state.position_m[2]
        self.assertGreater(hover_height, takeoff_height)

        # A bounded differential command produces a finite, observable turn.
        turn_command = {
            "left_power": trim.left_power + 0.10,
            "right_power": trim.right_power - 0.10,
        }
        run(250, turn_command)
        turned = outputs[-1]
        self.assertGreater(abs(turned.state.orientation_quat[3]), 1.0e-2)
        self.assertLess(max(abs(rate) for rate in turned.state.angular_velocity_rad_s), 20.0)

        # A finite wind pulse is a disturbance, not a hidden trajectory write.
        model.set_wind((1.5, -0.7, 0.0))
        run(150, trim)
        model.set_wind((0.0, 0.0, 0.0))
        run(800, trim)
        recovered = outputs[-1]
        self.assertLess(max(abs(rate) for rate in recovered.state.angular_velocity_rad_s), 2.0)
        self.assertLess(math.sqrt(sum(v * v for v in recovered.state.velocity_mps)), 0.45)

        # Power-off descent terminates at the physical ground constraint with
        # zero downward impact velocity and finite contact reaction.
        run(2000, {"left_power": 0.0, "right_power": 0.0})
        landed = outputs[-1]
        self.assertGreaterEqual(landed.state.position_m[2], model.config.ground_plane_z_m)
        self.assertAlmostEqual(landed.state.velocity_mps[2], 0.0, places=12)
        self.assertGreater(landed.contact_force_body_N[2], 0.0)
        self.assertTrue(landed.to_body_state()["contact"])

    def test_parameter_manifest_is_json_safe_and_reports_trim_contract(self) -> None:
        model = FlightDynamics()
        manifest = model.config.parameter_manifest()
        self.assertEqual(manifest["units"]["mass_kg"], "kg")
        self.assertEqual(manifest["units"]["ground_plane_z_m"], "m")
        self.assertAlmostEqual(manifest["differential_roll_moment_scale"], 0.04)
        self.assertIn("parameters", model.provenance)
        json.dumps(model.provenance, sort_keys=True)
        self.assertEqual(model.manifest()["parameters"], manifest)
        diagnostics = model.hover_trim_diagnostics()
        self.assertTrue(diagnostics["feasible"])
        self.assertAlmostEqual(
            diagnostics["capacity_N"] * diagnostics["trim_power"] ** 2,
            diagnostics["weight_N"],
            places=12,
        )

    def test_reduced_jump_input_is_strictly_neuromuscular_and_bounded(self) -> None:
        parsed = JumpNeuromuscularInput.from_any({
            "jump": 2.0,
            "leg_power": -1.0,
            "leg_steering": 2.0,
            "posture_roll": 2.0,
        })
        self.assertEqual(parsed.jump_activation, 1.0)
        self.assertEqual(parsed.leg_power, 0.25)
        self.assertEqual(parsed.leg_steering, 0.5)
        self.assertEqual(parsed.left_leg_power, 0.0)
        self.assertEqual(parsed.right_leg_power, 0.5)
        self.assertEqual(parsed.posture_roll, 1.0)
        self.assertEqual(parsed.units["jump_activation"], "normalized [0, 1]")
        for forbidden in ("threat", "phase", "goal", "direction", "pose"):
            with self.assertRaisesRegex(FlightDynamicsError, "unsupported jump input channel"):
                JumpNeuromuscularInput.from_any({"jump": 1.0, forbidden: 0.5})

    def test_jump_actuator_exposes_delay_duration_units_and_provenance(self) -> None:
        config = JumpLegConfig(actuator_delay_s=0.002, actuator_duration_s=0.003)
        actuator = JumpLegActuator(config)
        before = actuator.evaluate({"jump": 1.0}, time_s=0.0, dt_s=0.001,
                                    command_start_s=0.0, contact=True)
        active = actuator.evaluate({"jump": 1.0}, time_s=0.002, dt_s=0.001,
                                    command_start_s=0.0, contact=True)
        after = actuator.evaluate({"jump": 1.0}, time_s=0.005, dt_s=0.001,
                                   command_start_s=0.0, contact=True)
        self.assertFalse(before.actuator_active)
        self.assertTrue(active.actuator_active)
        self.assertGreater(active.force_body_N[2], 0.0)
        self.assertFalse(after.actuator_active)
        self.assertEqual(active.units["force_N"], "N")
        self.assertEqual(active.units["impulse_Ns"], "N*s")
        self.assertNotIn("parameters", active.to_dict())
        self.assertNotIn("ground_plane_z_m", active.to_dict())
        self.assertFalse(config.provenance()["is_flybody"])
        self.assertIn("not complete FlyBody", config.provenance()["scope"])
        json.dumps(active.to_dict(), sort_keys=True)

    def test_jump_actuator_reuses_static_parameter_manifest(self) -> None:
        config = JumpLegConfig(actuator_delay_s=0.002, actuator_duration_s=0.003)
        original_manifest = JumpLegConfig.parameter_manifest
        calls = 0

        def counted_manifest(current: JumpLegConfig) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return original_manifest(current)

        with mock.patch.object(JumpLegConfig, "parameter_manifest", counted_manifest):
            actuator = JumpLegActuator(config)
            first = actuator.evaluate({"jump": 1.0}, time_s=0.002, dt_s=0.001)
            second = actuator.evaluate({"jump": 1.0}, time_s=0.003, dt_s=0.001)

        self.assertEqual(calls, 1)
        self.assertEqual(first.parameters, second.parameters)
        self.assertIsNot(first.parameters, second.parameters)

    def test_combined_jump_cache_is_output_equivalent_to_uncached_actuator(self) -> None:
        """Caching config/actuator must preserve forces, latching, and state."""
        floor = 0.0012
        config = FlightDynamicsConfig(ground_plane_z_m=floor)
        jump = JumpLegConfig(ground_plane_z_m=floor, actuator_duration_s=0.010)

        class UncachedFlightDynamics(FlightDynamics):
            def _combined_jump_actuator(self) -> JumpLegActuator:
                return JumpLegActuator(self.jump_config)

        cached = FlightDynamics(
            config=config,
            state=FlightState(position_m=(0.0, 0.0, floor)),
            jump_config=jump,
        )
        reference = UncachedFlightDynamics(
            config=config,
            state=FlightState(position_m=(0.0, 0.0, floor)),
            jump_config=jump,
        )
        commands = (
            {"ttmn_activity": 1.0, "leg_power": 1.0, "wing_motor_left": 0.8,
             "wing_motor_right": 0.7},
            {"ttmn_activity": 1.0, "leg_power": 1.0, "wing_motor_left": 0.8,
             "wing_motor_right": 0.7},
            {},
            {"ttmn_activity": 1.0, "leg_power": 1.0, "wing_motor_left": 0.6,
             "wing_motor_right": 0.9},
        )
        for command in commands:
            cached_output = cached.step_with_jump(command, dt_s=0.005)
            reference_output = reference.step_with_jump(command, dt_s=0.005)
            self.assertEqual(cached_output.to_dict(), reference_output.to_dict())

    def test_combined_jump_cache_invalidates_on_mapping_and_body_config_changes(self) -> None:
        floor = 0.0012
        jump_mapping: dict[str, object] = {
            "ground_plane_z_m": floor,
            "max_leg_force_N": 1.0e-4,
        }
        model = FlightDynamics(
            config=FlightDynamicsConfig(ground_plane_z_m=floor),
            state=FlightState(position_m=(0.0, 0.0, floor)),
            jump_config=jump_mapping,
        )
        first_config = model._combined_jump_config()
        first_actuator = model._combined_jump_actuator()
        self.assertAlmostEqual(first_config.max_leg_force_N, 1.0e-4)

        # In-place mutation is supported by the historical mapping boundary;
        # its normalized config and actuator must not remain stale.
        jump_mapping["max_leg_force_N"] = 2.0e-4
        second_config = model._combined_jump_config()
        second_actuator = model._combined_jump_actuator()
        self.assertIsNot(first_config, second_config)
        self.assertIsNot(first_actuator, second_actuator)
        self.assertAlmostEqual(second_config.max_leg_force_N, 2.0e-4)

        # Direct assignment of the public body config changes the aligned
        # support plane and therefore invalidates the normalized jump config.
        model.config = FlightDynamicsConfig(ground_plane_z_m=0.015)
        third_config = model._combined_jump_config()
        self.assertIsNot(second_config, third_config)
        self.assertAlmostEqual(third_config.ground_plane_z_m, 0.015)

    def test_reduced_jump_liftoff_latches_event_and_stops_ground_force(self) -> None:
        model = ReducedBodyJumpDynamics(
            state=FlightState(position_m=(0.0, 0.0, 0.0)),
        )
        first = model.step({"jump": 1.0}, dt_s=0.001)
        self.assertGreater(first.jump_force_body_N[2], 0.0)
        self.assertGreater(first.state.position_m[2], 0.0)
        self.assertTrue(first.airborne)

        # A held event is latched at liftoff.  It cannot keep injecting a
        # ground push after the body has left the support plane.
        airborne = model.step({"jump": 1.0}, dt_s=0.001)
        self.assertEqual(airborne.jump_force_body_N, (0.0, 0.0, 0.0))
        self.assertEqual(airborne.contact_force_body_N, (0.0, 0.0, 0.0))
        self.assertTrue(airborne.contact_gated)

    def test_reduced_jump_contact_reports_physical_impulse_not_pose_command(self) -> None:
        model = ReducedBodyJumpDynamics(
            state=FlightState(
                position_m=(0.0, 0.0, 0.001),
                velocity_mps=(0.0, 0.0, -1.0),
            ),
        )
        output = model.step({}, dt_s=0.005)
        self.assertTrue(output.contact)
        self.assertEqual(output.state.position_m[2], 0.0)
        self.assertEqual(output.state.velocity_mps[2], 0.0)
        self.assertGreater(output.contact_force_world_N[2], 0.0)
        self.assertGreater(output.contact_impulse_Ns[2], 0.0)
        self.assertAlmostEqual(
            output.contact_impulse_Ns[2],
            output.contact_force_world_N[2] * 0.005,
            places=15,
        )

    def test_reduced_jump_uses_body_center_contact_plane_for_resting_start(self) -> None:
        world_ground_m = 0.015
        body_radius_m = 0.0012
        body_center_floor_m = world_ground_m + body_radius_m
        model = ReducedBodyJumpDynamics(
            config=JumpLegConfig(ground_plane_z_m=body_center_floor_m),
            state=FlightState(position_m=(0.0, 0.0, body_center_floor_m)),
        )
        resting = model.step({}, dt_s=0.001)
        self.assertTrue(resting.contact)
        self.assertAlmostEqual(resting.state.position_m[2], body_center_floor_m, places=12)
        self.assertAlmostEqual(resting.state.velocity_mps[2], 0.0, places=12)
        self.assertGreater(resting.contact_force_world_N[2], 0.0)
        self.assertAlmostEqual(resting.ground_plane_z_m, body_center_floor_m, places=12)

    def test_reduced_jump_posture_channel_integrates_attitude(self) -> None:
        model = ReducedBodyJumpDynamics(
            state=FlightState(position_m=(0.0, 0.0, 0.0)),
        )
        output = model.step({"posture_roll": 1.0}, dt_s=0.001)
        self.assertNotEqual(output.posture_delta_rad, (0.0, 0.0, 0.0))
        self.assertNotEqual(output.state.orientation_quat, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(output.jump_force_body_N, (0.0, 0.0, 0.0))

    def test_combined_wing_leg_uses_one_state_and_physical_motor_channels(self) -> None:
        floor = 0.015 + 0.0012
        state = FlightState(position_m=(0.0, 0.0, floor))
        combined = WingLegFlightDynamics(
            config=FlightDynamicsConfig(ground_plane_z_m=floor),
            state=state,
        )
        command = {
            # Physical wm channels must win over any legacy flight-power alias.
            "wing_motor_left": 0.80,
            "wing_motor_right": 0.70,
            "flight_power": 0.0,
            "ttmn_activity": 1.0,
            "leg_power": 1.0,
            "posture_pitch": 0.2,
            "source": "connectome",
        }
        first = combined.step(command, dt_s=0.005)
        self.assertAlmostEqual(first.left_power, 0.80)
        self.assertAlmostEqual(first.right_power, 0.70)
        self.assertGreater(first.jump_force_body_N[2], 1.0e-4)
        self.assertGreater(first.state.velocity_mps[2], 0.40)
        self.assertGreater(first.state.position_m[2], floor)
        self.assertEqual(first.jump_input.activation_source, "ttmn_activity")

        # The same held neural object cannot keep applying a ground push after
        # liftoff; wing flight still continues through the same state integrator.
        second = combined.step(command, dt_s=0.005)
        self.assertEqual(second.jump_force_body_N, (0.0, 0.0, 0.0))
        self.assertTrue(second.jump_contact_gated)
        self.assertGreater(second.state.position_m[2], first.state.position_m[2])
        self.assertIs(combined.body_state, combined.dynamics.state)

    def test_ttmn_activity_is_bilateral_jump_drive_without_generic_leg_pool(self) -> None:
        floor = 0.0012
        model = WingLegFlightDynamics(
            config=FlightDynamicsConfig(ground_plane_z_m=floor),
            state=FlightState(position_m=(0.0, 0.0, floor)),
        )
        output = model.step(
            {"ttmn_activity": 1.0, "leg_power": 0.0},
            dt_s=0.005,
        )
        self.assertGreater(output.jump_force_body_N[2], 1.0e-4)
        self.assertGreater(output.state.velocity_mps[2], 0.40)

    def test_combined_bridge_does_not_convert_legacy_jump_object_in_strict_mode(self) -> None:
        class LegacyOutput:
            left_wing_power = 0.0
            right_wing_power = 0.0
            jump = 1.0  # legacy DNp01 decision compatibility field
            leg_power = 1.0
            leg_steering = 0.0

        floor = 0.0012
        model = WingLegFlightDynamics(
            config=FlightDynamicsConfig(ground_plane_z_m=floor),
            state=FlightState(position_m=(0.0, 0.0, floor)),
        )
        strict = model.step(LegacyOutput(), dt_s=0.005)
        self.assertEqual(strict.jump_force_body_N, (0.0, 0.0, 0.0))
        model.reset(FlightState(position_m=(0.0, 0.0, floor)))
        compatibility = model.step(LegacyOutput(), dt_s=0.005, strict_neural=False)
        self.assertGreater(compatibility.jump_force_body_N[2], 0.0)
        self.assertEqual(compatibility.jump_input.activation_source, "legacy_jump_compatibility")

    def test_combined_bridge_collision_clearance_and_finite_push_speed(self) -> None:
        floor = 0.0012
        model = WingLegFlightDynamics(
            config=FlightDynamicsConfig(ground_plane_z_m=floor),
            state=FlightState(position_m=(0.0, 0.0, floor)),
        )
        launch = model.step({"ttmn_activity": 1.0, "leg_power": 1.0}, dt_s=0.005)
        self.assertGreater(launch.jump_impulse_body_Ns[2], 0.50e-6)
        outputs = [launch]
        for _ in range(80):
            outputs.append(model.step({}, dt_s=0.005))
            self.assertGreaterEqual(outputs[-1].state.position_m[2], floor - 1.0e-12)
        landed = outputs[-1]
        self.assertTrue(landed.to_body_state()["contact"])
        self.assertAlmostEqual(landed.state.position_m[2], floor, places=12)
        self.assertAlmostEqual(landed.state.velocity_mps[2], 0.0, places=12)
        self.assertGreater(landed.contact_force_body_N[2], 0.0)


if __name__ == "__main__":
    unittest.main()
