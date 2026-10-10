#!/usr/bin/env python3
"""Self-tests for the named MaleCNS neuromuscular readout.

These tests use the checked CSR annotation tables when present and do not run
the GPU LIF loop.  They therefore exercise map provenance/resolution,
rate/spike semantics, smoothing/saturation/symmetry, and the dependency-free
DigitalBody command boundary quickly on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from neuromuscular_adapter import (  # noqa: E402
    MappingPopulationIndex,
    NeuromuscularAdapter,
)


class NeuromuscularAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = NeuromuscularAdapter(strict=True)

    def setUp(self) -> None:
        self.adapter.reset(clear_baseline=True)

    def test_map_uses_only_resolved_named_populations(self) -> None:
        report = self.adapter.validate(strict=True)
        self.assertEqual(report["unresolved"], [])
        sizes = self.adapter.population_sizes()
        self.assertEqual(sizes["flight_power"], 29)
        self.assertEqual(sizes["wing_motor"], 67)
        self.assertEqual(sizes["leg_power"], 381)
        self.assertEqual(sizes["mouth"], 10)
        self.assertEqual(sizes["groom_antenna"], 4)
        self.assertEqual(sizes["wing_motor_left"], 33)
        self.assertEqual(sizes["wing_motor_right"], 34)

    def test_strict_input_mapping_rejects_empty_population_selector(self) -> None:
        # ``DNg02`` is a historical nickname trap; the real CSR types are
        # DNg02_a..g and are addressed by the checked ``prefix:`` selector.
        with self.assertRaisesRegex(ValueError, "unresolved selectors"):
            self.adapter.step(rates_hz={"DNg02": 50.0}, dt_ms=50.0)

    def test_rate_smoothing_baseline_and_saturation(self) -> None:
        self.adapter.set_baseline({"prefix:DNg02": 10.0})
        first = self.adapter.step(rates_hz={"prefix:DNg02": 500.0}, dt_ms=50.0)
        # Static/running baseline leaves a clipped unit drive, then the 100 ms
        # EMA advances by 1-exp(-50/100).
        self.assertAlmostEqual(first.flight_power, 1.0 - np.exp(-0.5), places=7)
        value = first.flight_power
        for _ in range(30):
            value = self.adapter.step(rates_hz={"prefix:DNg02": 500.0}, dt_ms=50.0).flight_power
        self.assertLessEqual(value, 1.0)
        self.assertGreater(value, 0.99)

    def test_bilateral_symmetry_and_signed_steering(self) -> None:
        bilateral = self.adapter.step(
            rates_hz={"prefix:DNg02/L": 50.0, "prefix:DNg02/R": 50.0}, dt_ms=50.0
        )
        self.assertAlmostEqual(bilateral.left_wing_power, bilateral.right_wing_power, places=12)
        self.assertAlmostEqual(bilateral.flight_steering, 0.0, places=12)

        self.adapter.reset()
        left_only = self.adapter.step(rates_hz={"prefix:DNg02/L": 50.0}, dt_ms=50.0)
        self.assertGreater(left_only.right_wing_power, left_only.left_wing_power)
        self.assertGreaterEqual(left_only.left_wing_power, 0.0)
        self.assertLessEqual(left_only.right_wing_power, 1.0)

    def test_spike_count_event_and_rate_vector(self) -> None:
        jump = self.adapter.step(spikes={"DNp01/L": 1}, dt_ms=50.0)
        self.assertEqual(jump.channel("jump"), 1.0)
        self.adapter.reset()
        dn02 = self.adapter.population_index.resolve("prefix:DNg02")
        rates = np.zeros(self.adapter.population_index.n, dtype=np.float32)
        rates[dn02] = 50.0
        out = self.adapter.step(rates_hz=rates, dt_ms=50.0)
        self.assertGreater(out.flight_power, 0.0)
        self.assertEqual(out.channels["flightPower"], out.channels["flight_power"])

    def test_escape_readouts_remain_separate_at_physics_boundary(self) -> None:
        """The three descending-neuron activities are not collapsed for physics."""

        out = self.adapter.step(
            rates_hz={"DNp11": 50.0, "DNp02": 25.0, "DNp04": 10.0},
            dt_ms=50.0,
        )
        for name in ("jump", "dnp11_activity", "dnp02_activity", "dnp04_activity"):
            self.assertIn(name, out.channels)
            self.assertIn(name, out.units)
            self.assertIn(name, out.provenance["channels"])
        self.assertGreater(out.dnp11_activity, out.dnp02_activity)
        self.assertGreater(out.dnp02_activity, out.dnp04_activity)
        self.assertEqual(out.provenance["dnp11_activity"]["confidence"], "high")
        self.assertEqual(out.provenance["dnp11_activity"]["direction"]["laterality"], "bilateral")
        self.assertEqual(out.provenance["dnp11_activity"]["direction"]["anterior_posterior"], "not_encoded")

        command = out.to_body_command()
        for name in (
            "jump", "jump_event", "jump_activation", "ttmn_activity", "jump_motor",
            "dnp11_activity", "dnp02_activity", "dnp04_activity",
        ):
            self.assertIn(name, command)
        self.assertAlmostEqual(command["dnp11_activity"], out.dnp11_activity)
        self.assertAlmostEqual(command["dnp02_activity"], out.dnp02_activity)
        self.assertAlmostEqual(command["dnp04_activity"], out.dnp04_activity)
        self.assertEqual(set(out.physical_channels), {
            "jump", "ttmn_activity", "jump_motor",
            "dnp11_activity", "dnp02_activity", "dnp04_activity",
        })
        # ``takeoff`` remains available as a legacy aggregate, but is not the
        # physical command surface and cannot replace the three source channels.
        self.assertIn("takeoff", out.channels)
        self.assertNotIn("takeoff", command)

    def test_jump_event_is_independent_from_descending_neuron_activity(self) -> None:
        out = self.adapter.step(
            spikes={"DNp01": 1, "DNp11": 1, "DNp02": 0, "DNp04": 0},
            dt_ms=50.0,
        )
        self.assertEqual(out.jump_event, 1.0)
        self.assertGreater(out.dnp11_activity, 0.0)
        self.assertEqual(out.dnp02_activity, 0.0)
        self.assertEqual(out.dnp04_activity, 0.0)
        quiet = self.adapter.step(spikes={}, dt_ms=50.0)
        self.assertEqual(quiet.jump, 0.0)

    def test_ttmn_is_the_physical_jump_motor_not_dnp01(self) -> None:
        """DNp01 remains a decision event; TTMn alone drives jump activation."""

        decision = self.adapter.step(spikes={"DNp01": 1}, dt_ms=50.0)
        self.assertEqual(decision.jump, 1.0)
        self.assertEqual(decision.ttmn_activity, 0.0)
        self.assertEqual(decision.jump_motor, 0.0)
        self.assertEqual(decision.jump_activation, 0.0)
        self.assertEqual(decision.to_body_command()["jump_activation"], 0.0)

        self.adapter.reset()
        motor = self.adapter.step(spikes={"TTMn": 1}, dt_ms=50.0)
        self.assertEqual(motor.jump, 0.0)
        self.assertGreater(motor.ttmn_activity, 0.0)
        self.assertEqual(motor.jump_motor, 1.0)
        self.assertEqual(motor.jump_activation, 1.0)
        self.assertEqual(motor.provenance["ttmn_activity"]["terms"][0]["spec"], "TTMn")
        command = motor.to_body_command()
        self.assertEqual(command["jump_activation"], command["jump_motor"])
        self.assertEqual(command["jump"], 0.0)
        self.assertEqual(motor.provenance["jump_motor"]["terms"][0]["spec"], "TTMn")
        self.assertEqual(motor.provenance["jump_activation"]["alias_of"], "jump_motor")
        self.assertEqual(motor.provenance["jump_event"]["alias_of"], "jump")

        # TTMn is also annotated inside the broader wm pool in this CSR; the
        # physical wing selectors use exact matching so TTMn does not leak into
        # a wing activation when no wm/L or wm/R packet was supplied.
        self.assertEqual(motor.provenance["physical_wing"]["source"], "flight_power_legacy")
        self.assertNotIn("wing_motor_left", motor.channels)
        self.assertNotIn("wing_motor_right", motor.channels)

    def test_exact_wing_motor_pools_drive_physical_left_right_activation(self) -> None:
        """VNC wm/L and wm/R are physical wing sources, independently signed."""

        left_spec = "superclass:vnc_motor&subclass:wm/L"
        right_spec = "superclass:vnc_motor&subclass:wm/R"
        self.adapter.reset()
        out = self.adapter.step(
            rates_hz={left_spec: 80.0, right_spec: 40.0}, dt_ms=50.0,
        )
        self.assertGreater(out.left_wing_activation, out.right_wing_activation)
        self.assertAlmostEqual(
            out.channel("left_wing_activation"), out.left_wing_activation
        )
        self.assertAlmostEqual(
            out.channel("right_wing_activation"), out.right_wing_activation
        )
        self.assertEqual(out.provenance["physical_wing"]["source"], "wing_motor")
        self.assertEqual(out.provenance["wing_motor_left"]["terms"][0]["spec"], left_spec)
        self.assertEqual(out.provenance["wing_motor_right"]["terms"][0]["spec"], right_spec)
        command = out.to_body_command()
        self.assertAlmostEqual(command["wing_drive_left"], out.left_wing_activation)
        self.assertAlmostEqual(command["wing_drive_right"], out.right_wing_activation)
        self.assertAlmostEqual(command["wing_motor_left"], out.wing_motor_left)
        self.assertAlmostEqual(command["wing_motor_right"], out.wing_motor_right)
        self.assertEqual(set(out.physical_wing_channels), {
            "left_wing_activation", "right_wing_activation",
            "wing_motor_left", "wing_motor_right",
        })

        # A legacy DNg02-only packet still has the historical fallback, but it
        # is explicitly marked as such and does not masquerade as wm activity.
        self.adapter.reset()
        legacy = self.adapter.step(rates_hz={"prefix:DNg02": 50.0}, dt_ms=50.0)
        self.assertEqual(legacy.provenance["physical_wing"]["source"], "flight_power_legacy")

    def test_named_malecns_population_output_is_decoded_with_provenance(self) -> None:
        """The real engine's structured population readout must not be ignored."""

        from twin_contracts import NeuralOutput, Timestamp  # noqa: E402

        timestamp = Timestamp(0.005, 10, "neural", 2000.0)
        neural = NeuralOutput(
            timestamp,
            rates_hz=[0.0] * self.adapter.population_index.n,
            spikes=[0] * self.adapter.population_index.n,
            named_outputs={
                "population_rates_hz": {
                    "prefix:DNg02": 38.0,
                    "prefix:DNg02/L": 38.0,
                    "prefix:DNg02/R": 38.0,
                },
                "backend": "torch_cuda",
                "device": "cuda",
            },
        )
        out = self.adapter.step(neural, dt_ms=5.0, timestamp=timestamp)
        self.assertGreater(out.flight_power, 0.0)
        self.assertAlmostEqual(out.left_wing_power, out.right_wing_power, places=12)
        self.assertEqual(out.provenance["calibration_id"], "malecns-windowed-rate-v1")
        self.assertEqual(out.provenance["signal"]["input_kind"], "named_population")

    def test_takeoff_hover_turn_commands_and_bounds(self) -> None:
        """Calibrated bilateral lift and signed turn commands stay bounded."""

        def settle(rates: dict[str, float], n: int = 24):
            value = None
            for _ in range(n):
                value = self.adapter.step(rates_hz=rates, dt_ms=50.0)
            assert value is not None
            return value

        # Takeoff/full-lift command: equal descending-neuron drive must remain
        # bilateral, not collapse to one side through the steering mixer.
        takeoff = settle({
            "prefix:DNg02": 50.0,
            "prefix:DNg02/L": 50.0,
            "prefix:DNg02/R": 50.0,
        })
        self.assertGreater(takeoff.left_wing_power, 0.95)
        self.assertGreater(takeoff.right_wing_power, 0.95)
        self.assertAlmostEqual(takeoff.left_wing_power, takeoff.right_wing_power, places=12)

        # Hover trim is an intermediate bilateral readout, not a hard-coded
        # motor command.
        self.adapter.reset()
        hover = settle({
            "prefix:DNg02": 35.0,
            "prefix:DNg02/L": 35.0,
            "prefix:DNg02/R": 35.0,
        })
        self.assertGreater(hover.left_wing_power, 0.55)
        self.assertLess(hover.left_wing_power, 0.85)
        self.assertAlmostEqual(hover.left_wing_power, hover.right_wing_power, places=12)

        # DNg02 is contralateral: a stronger left DN population increases the
        # right wing, and the signed steering channel follows that convention.
        self.adapter.reset()
        turn_right = settle({
            "prefix:DNg02": 40.0,
            "prefix:DNg02/L": 48.0,
            "prefix:DNg02/R": 32.0,
        })
        self.assertGreater(turn_right.flight_steering, 0.0)
        self.assertGreater(turn_right.right_wing_power, turn_right.left_wing_power)

        # Malformed/oversized neural values fail safe into finite normalized
        # commands rather than leaking NaN or an unbounded wing drive.
        self.adapter.reset()
        bounded = self.adapter.step(rates_hz={
            "prefix:DNg02": float("inf"),
            "prefix:DNg02/L": float("nan"),
            "prefix:DNg02/R": 1.0e12,
        }, dt_ms=50.0)
        for value in (
            bounded.left_wing_power, bounded.right_wing_power, bounded.wing_power,
            bounded.wing_steering, bounded.flight_steering,
        ):
            self.assertTrue(np.isfinite(value))
        self.assertTrue(0.0 <= bounded.left_wing_power <= 1.0)
        self.assertTrue(0.0 <= bounded.right_wing_power <= 1.0)
        self.assertTrue(-1.0 <= bounded.wing_steering <= 1.0)

    def test_dense_malecns_ticks_are_window_aggregated(self) -> None:
        """One-tick 0/2000 Hz spikes are averaged before rate calibration."""

        indices = self.adapter.population_index.resolve("prefix:DNg02")
        rates = np.zeros(self.adapter.population_index.n, dtype=np.float32)
        rates[indices] = 2000.0
        outputs = [self.adapter.step(rates_hz=rates, dt_ms=0.5) for _ in range(10)]
        self.assertTrue(all(output.flight_power == 0.0 for output in outputs[:-1]))
        self.assertGreater(outputs[-1].flight_power, 0.0)
        self.assertLessEqual(outputs[-1].left_wing_power, 1.0)
        self.assertLessEqual(outputs[-1].right_wing_power, 1.0)
        self.assertEqual(outputs[-1].provenance["signal"]["decode_window_ms"], 5.0)

    def test_cached_decode_is_output_equivalent_for_mixed_inputs(self) -> None:
        """Static decode caches must not change values or serialized fields."""

        reference = NeuromuscularAdapter(
            map_path=self.adapter.map_path,
            population_index=self.adapter.population_index,
            strict=True,
        )
        self.adapter.reset(clear_baseline=True)
        reference.reset(clear_baseline=True)
        indices = self.adapter.population_index.resolve("prefix:DNg02")
        dense = np.zeros(self.adapter.population_index.n, dtype=np.float64)
        dense[indices] = 50.0
        signals = (
            (dict(rates_hz={"prefix:DNg02": 50.0, "prefix:DNg02/L": 40.0}), 50.0),
            (dict(spikes={"DNp01/L": 1, "TTMn": 1}), 50.0),
            (dict(rates_hz=dense), 50.0),
            (dict(rates_hz={"prefix:DNg02": float("inf"), "prefix:DNg02/R": float("nan")}), 50.0),
        )
        for kwargs, dt_ms in signals:
            left = self.adapter.step(dt_ms=dt_ms, **kwargs)
            right = reference.step(dt_ms=dt_ms, **kwargs)
            self.assertEqual(left.to_dict(), right.to_dict())

    def test_selector_resolution_is_cached_after_first_lookup(self) -> None:
        class CountingIndex:
            n = 8

            def __init__(self) -> None:
                self.calls: list[str] = []
                self.values = {
                    "prefix:DNg02": np.array([0, 1], dtype=np.int64),
                    "prefix:DNg02/L": np.array([0], dtype=np.int64),
                    "prefix:DNg02/R": np.array([1], dtype=np.int64),
                }

            def resolve(self, spec: str) -> np.ndarray:
                key = str(spec).strip()
                self.calls.append(key)
                return self.values.get(key, np.empty(0, dtype=np.int64)).copy()

        resolver = CountingIndex()
        adapter = NeuromuscularAdapter(
            map_path=self.adapter.map_path,
            population_index=resolver,
            strict=False,
        )
        # Validation warms configured selectors. A new selector is resolved
        # once, then served from the adapter cache on subsequent calls.
        before = len(resolver.calls)
        np.testing.assert_array_equal(adapter.resolve("prefix:DNg02"), np.array([0, 1]))
        after_first = len(resolver.calls)
        np.testing.assert_array_equal(adapter.resolve("prefix:DNg02"), np.array([0, 1]))
        self.assertEqual(len(resolver.calls), after_first)
        self.assertGreaterEqual(after_first, before)

        # The first decode may discover mapping keys, but a repeated tick must
        # not invoke the resolver again for the same immutable selectors.
        adapter._resolved_selector_cache.clear()
        resolver.calls.clear()
        adapter.step(
            rates_hz={"prefix:DNg02": 50.0, "prefix:DNg02/L": 40.0},
            dt_ms=50.0,
        )
        first_decode_calls = len(resolver.calls)
        self.assertGreater(first_decode_calls, 0)
        adapter.step(
            rates_hz={"prefix:DNg02": 50.0, "prefix:DNg02/L": 40.0},
            dt_ms=50.0,
        )
        self.assertEqual(len(resolver.calls), first_decode_calls)

    def test_mapping_population_index_caches_combined_selectors(self) -> None:
        index = MappingPopulationIndex({"left": [0, 2], "right": [1, 2]})
        first = index.resolve("left,right")
        first[0] = 99
        self.assertEqual(index.resolve("left,right").tolist(), [0, 1, 2])
        self.assertEqual(index.resolve("left,right").tolist(), [0, 1, 2])

    def test_digital_body_and_twin_contract_bridges(self) -> None:
        from digital_body import DigitalBody  # noqa: E402
        from twin_contracts import MultiRateClock, NeuralOutput, SourceLabel  # noqa: E402

        clock = MultiRateClock({"control": 20.0})
        timestamp = clock.timestamp("control")
        out = self.adapter.step(
            rates_hz={"prefix:DNg02": 50.0, "DNa02/R": 50.0},
            dt_ms=50.0,
            timestamp=timestamp,
        )
        body = DigitalBody()
        state = self.adapter.apply_to_body(body, rates_hz={"prefix:DNg02": 50.0}, dt_ms=50.0)
        self.assertTrue(hasattr(state, "wing_drive_left"))
        command = out.to_body_command()
        self.assertTrue(0.0 <= command["wing_drive_left"] <= 1.0)
        self.assertTrue(0.0 <= command["wing_drive_right"] <= 1.0)

        neural = self.adapter.to_neural_output()
        motor = self.adapter.to_motor_command(max_yaw_rate_rad_s=2.0, max_thrust_mps2=9.8)
        self.assertEqual(neural.source, SourceLabel.CONNECTOME)
        self.assertEqual(motor.source, SourceLabel.FITTED)
        self.assertIn("left_wing_power", motor.metadata)

        self.adapter.reset(clear_baseline=True)
        handbuilt = NeuralOutput(timestamp, rates_hz=[0.0, 0.0], spikes=[0, 0],
                                  source=SourceLabel.HANDBUILT)
        decoded = self.adapter.decode(handbuilt, timestamp)
        self.assertEqual(decoded.timestamp, timestamp)
        self.assertEqual(decoded.source, SourceLabel.FITTED)


if __name__ == "__main__":
    unittest.main()
