#!/usr/bin/env python3
"""Micro-CSR tests for the reusable MaleCNS engine.

These tests intentionally never step the real 176k-neuron graph.  The real
CSR is checked only through the read-only header/file preflight.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from twin_contracts import SensoryFrame, SourceLabel, Timestamp  # noqa: E402
import malecns_engine as engine  # noqa: E402
from lif_flyb import Populations as LifPopulations  # noqa: E402
from run_digital_fly import (  # noqa: E402
    MalecnsNeuralAdapter,
    _aggregate_neural_outputs,
    _substep_sensory,
)


def make_micro_csr(root: Path) -> Path:
    """Build a four-neuron ring with complete population annotations."""

    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "types": ["", "A", "B"],
        "superclasses": ["", "test_super"],
        "classes": ["", "test_class", "Kenyon_Cell"],
        "subclasses": ["", "test_sub"],
        "nts": ["", "acetylcholine"],
        "sides": ["", "L", "R"],
        "dimorphisms": [""],
        "fruDsx": [""],
        "neuromeres": ["", "A1"],
        "nerves": ["", "ADMN"],
    }
    (root / "tables.json").write_text(json.dumps(tables), encoding="utf-8")
    np.save(root / "ids.npy", np.array([100, 101, 102, 103], dtype=np.int64))
    np.save(root / "offsets.npy", np.array([0, 1, 2, 3, 4], dtype=np.int64))
    np.save(root / "target.npy", np.array([1, 2, 3, 0], dtype=np.int32))
    np.save(root / "weight.npy", np.ones(4, dtype=np.uint16))
    np.save(root / "sign.npy", np.array([1, 1, -1, -1], dtype=np.int8))
    fields = {
        "typeIdx": np.array([1, 2, 1, 2], dtype=np.int32),
        "superclassIdx": np.ones(4, dtype=np.uint8),
        "classIdx": np.ones(4, dtype=np.uint8),
        "subclassIdx": np.ones(4, dtype=np.int16),
        "ntIdx": np.ones(4, dtype=np.uint8),
        "sideIdx": np.array([1, 2, 1, 2], dtype=np.uint8),
        "neuromereIdx": np.ones(4, dtype=np.uint8),
        "nerveIdx": np.ones(4, dtype=np.uint8),
    }
    np.savez(root / "annotations.npz", **fields)
    (root / "meta.json").write_text(json.dumps({
        "n": 4, "edges": 4, "wSynMv": 0.275, "synapses_in_edges": 4,
    }), encoding="utf-8")
    return root


def make_mechanosensory_csr(root: Path) -> Path:
    """Build a tiny CSR with the four config-backed sensory populations."""

    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "types": ["", "JO", "HAL", "WT", "WP", "LT", "LP"],
        "superclasses": ["", "sensory"],
        "classes": ["", "mechanosensory", "mechanosensory_proprioceptive",
                     "mechanosensory_tactile"],
        "subclasses": ["", "wind_gravity", "haltere", "wing", "leg"],
        "nts": ["", "acetylcholine"],
        "sides": ["", "L", "R"],
        "dimorphisms": [""],
        "fruDsx": [""],
        "neuromeres": ["", "T1"],
        "nerves": ["", "ADMN", "ProLN", "MesoLN", "MetaLN"],
    }
    (root / "tables.json").write_text(json.dumps(tables), encoding="utf-8")
    n = 7
    np.save(root / "ids.npy", np.arange(200, 200 + n, dtype=np.int64))
    np.save(root / "offsets.npy", np.arange(n + 1, dtype=np.int64))
    np.save(root / "target.npy", np.roll(np.arange(n, dtype=np.int32), -1))
    np.save(root / "weight.npy", np.ones(n, dtype=np.uint16))
    np.save(root / "sign.npy", np.ones(n, dtype=np.int8))
    np.savez(root / "annotations.npz",
             typeIdx=np.array([1, 2, 3, 4, 5, 5, 6], dtype=np.int32),
             superclassIdx=np.ones(n, dtype=np.uint8),
             classIdx=np.array([1, 2, 3, 2, 3, 3, 2], dtype=np.uint8),
             subclassIdx=np.array([1, 2, 3, 3, 4, 4, 4], dtype=np.int16),
             ntIdx=np.ones(n, dtype=np.uint8),
             sideIdx=np.array([1, 2, 1, 2, 1, 2, 1], dtype=np.uint8),
             neuromereIdx=np.ones(n, dtype=np.uint8),
             nerveIdx=np.array([0, 0, 1, 0, 2, 3, 0], dtype=np.uint8))
    (root / "meta.json").write_text(json.dumps({
        "n": n, "edges": n, "wSynMv": 0.275, "synapses_in_edges": n,
    }), encoding="utf-8")
    return root


class MaleCNSEngineTest(unittest.TestCase):
    def test_real_csr_preflight_is_read_only_and_shape_only(self):
        report = engine.preflight_csr()
        self.assertTrue(report.valid, report.blockers)
        self.assertEqual(report.neurons, 176422)
        self.assertEqual(report.edges, 6287749)
        self.assertEqual(report.synapses, 90296905)
        self.assertEqual(report.shapes["ids"], (176422,))
        self.assertEqual(report.shapes["offsets"], (176423,))
        self.assertEqual(report.shapes["target"], (6287749,))

    def test_default_import_does_not_import_torch(self):
        code = (
            "import sys; sys.path.insert(0, %r); "
            "import malecns_engine; "
            "assert 'torch' not in sys.modules; "
            "print('ok')"
        ) % str(SCRIPTS)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "ok")

    def test_cpu_step_order_external_current_and_population_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(
                path, backend="cpu", device="cpu", gain=1.0, kc_gain=1.0,
                seed=11, report_populations={"a": "A", "left": "A/L"},
            )
            t0 = Timestamp(0.0, 0, "neural", 2000.0)
            frame = SensoryFrame(t0, channels={"current_mV_per_ms": {"A": 20.0}})
            out = e.step(frame, t0)
            # current is applied in mV/ms, then multiplied by dt=0.5 ms:
            # -52 + 10 > -45, so both A neurons spike before delivery.
            self.assertEqual(out.source, SourceLabel.CONNECTOME)
            self.assertEqual(out.named_outputs["spike_count"], 2)
            self.assertEqual(out.named_outputs["population_spike_counts"]["a"], 2)
            self.assertEqual(out.named_outputs["population_rates_hz"]["left"], 2000.0)
            self.assertEqual(out.neuron_count, 4)
            self.assertEqual(e.step_index, 1)
            self.assertEqual(e.delay_steps, 4)
            # The A spikes have delivered their rows, but delay=1.8 ms means
            # the B targets do not see that ring slot until step 4.
            for tick in range(1, 5):
                ts = Timestamp(tick * 0.0005, tick, "neural", 2000.0)
                quiet = SensoryFrame(ts)
                e.step(quiet, ts)
            self.assertGreater(float(e.g[1]), 0.0)
            self.assertLess(float(e.g[3]), 0.0)  # source neuron 2 is inhibitory

    def test_poisson_seed_reset_and_snapshot_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            kwargs = dict(csr_path=path, backend="cpu", device="cpu", seed=123,
                          poisson_hz={"B": 2000.0})
            a = engine.MaleCNSEngine(**kwargs)
            b = engine.MaleCNSEngine(**kwargs)
            outputs_a, outputs_b = [], []
            for tick in range(3):
                ts = Timestamp(tick * 0.0005, tick, "neural", 2000.0)
                outputs_a.append(a.step(SensoryFrame(ts), ts).spikes.copy())
                outputs_b.append(b.step(SensoryFrame(ts), ts).spikes.copy())
            for x, y in zip(outputs_a, outputs_b):
                np.testing.assert_array_equal(x, y)

            snap = a.snapshot()
            ts3 = Timestamp(0.0015, 3, "neural", 2000.0)
            after = a.step(SensoryFrame(ts3), ts3).spikes.copy()
            a.restore(snap)
            replay = a.step(SensoryFrame(ts3), ts3).spikes.copy()
            np.testing.assert_array_equal(after, replay)

            a.reset(seed=123)
            ts0 = Timestamp(0.0, 0, "neural", 2000.0)
            np.testing.assert_array_equal(
                a.step(SensoryFrame(ts0), ts0).spikes,
                outputs_a[0],
            )

    def test_torch_snapshot_restores_generator_state(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Torch is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(path, backend="torch_cpu", device="cpu",
                                      seed=123, poisson_hz={"B": 2000.0})
            timestamps = [Timestamp(tick * 0.0005, tick, "neural", 2000.0)
                          for tick in range(3)]
            for ts in timestamps[:2]:
                e.step(SensoryFrame(ts), ts)
            snap = e.snapshot()
            after = e.step(SensoryFrame(timestamps[2]), timestamps[2]).spikes.copy()
            e.restore(snap)
            replay = e.step(SensoryFrame(timestamps[2]), timestamps[2]).spikes.copy()
            np.testing.assert_array_equal(after, replay)
            self.assertIsNotNone(snap["rng_state"])

    def test_population_grammar_and_explicit_flyvis_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(
                path, backend="cpu", device="cpu", gain=1.0, kc_gain=1.0,
                flyvis_targets={0: {"population": "A", "kind": "current_mV_per_ms", "gain": 20.0}},
            )
            np.testing.assert_array_equal(e.resolve_population("A/L"), np.array([0, 2]))
            np.testing.assert_array_equal(e.resolve_population("A,B"), np.array([0, 1, 2, 3]))
            with self.assertRaises(engine.PopulationError):
                e.resolve_population("A/Q")
            with self.assertRaises(engine.PopulationError):
                e.resolve_population("A/M")
            cached_copy = e.resolve_population("A")
            cached_copy[0] = 99
            np.testing.assert_array_equal(e.resolve_population("A"), [0, 2])
            t0 = Timestamp(0.0, 0, "neural", 2000.0)
            frame = SensoryFrame(t0, channels={"flyvis_t4t5": [1.0]})
            out = e.step(frame, t0)
            self.assertEqual(out.named_outputs["spike_count"], 2)
            self.assertIn("flyvis_t4t5_drive", out.named_outputs)

            bad = engine.MaleCNSEngine(path, backend="cpu", device="cpu")
            with self.assertRaises(engine.DriveError):
                bad.step(SensoryFrame(t0, channels={"flyvis_drive": [1.0]}), t0)

    def test_lif_population_selector_keeps_permissive_matching_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            tables = json.loads((path / "tables.json").read_text(encoding="utf-8"))
            with np.load(path / "annotations.npz") as archive:
                annotations = {key: archive[key] for key in archive.files}
            populations = LifPopulations(tables, annotations)
            populations.ids = np.load(path / "ids.npy")

            self.assertEqual(populations.resolve("unknown-type").size, 0)
            self.assertEqual(populations.resolve("A/Q").size, 0)
            np.testing.assert_array_equal(populations.resolve("A&class:test_class"), [0, 2])
            np.testing.assert_array_equal(populations.resolve("body:102"), [2])
            with self.assertRaises(SystemExit):
                populations.resolve("unknown:value")

            cached = populations.resolve("A")
            cached[0] = 99
            self.assertIs(populations.resolve("A"), cached)

    def test_explicit_flyvis_target_indices_inject_signed_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(path, backend="cpu", device="cpu")
            t0 = Timestamp(0.0, 0, "neural", 2000.0)
            drive = {
                "target_indices": np.array([0, 2], dtype=np.int64),
                "values": np.array([3.0, -4.0], dtype=np.float32),
                "units": "current_mV_per_ms",
            }
            frame = SensoryFrame(t0, channels={"flyvis_t4t5": drive})
            _, current, _ = e._compose_drives(frame)
            np.testing.assert_array_equal(current, np.array([3.0, 0.0, -4.0, 0.0], dtype=np.float32))
            out = e.step(frame, t0)
            self.assertEqual(out.named_outputs["external_current_mV_per_ms_max"], 4.0)
            np.testing.assert_array_equal(out.named_outputs["flyvis_t4t5_drive"], [3.0, -4.0])
            self.assertEqual(out.named_outputs["flyvis_units"], "current_mV_per_ms")

            bad = {
                "target_indices": np.array([0, 0], dtype=np.int64),
                "values": np.array([1.0, 2.0], dtype=np.float32),
                "units": "current_mV_per_ms",
            }
            with self.assertRaisesRegex(engine.DriveError, "unique"):
                e.step(SensoryFrame(Timestamp(0.0005, 1, "neural", 2000.0),
                                    channels={"flyvis_t4t5": bad}),
                       Timestamp(0.0005, 1, "neural", 2000.0))

    def test_owned_read_only_flyvis_mapping_is_validated_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(path, backend="cpu", device="cpu")
            indices = np.array([0, 2], dtype=np.int64)
            indices.setflags(write=False)
            drive = {
                "target_indices": indices,
                "values": np.array([3.0, -4.0], dtype=np.float32),
                "units": "current_mV_per_ms",
            }
            ts = Timestamp(0.0, 0, "neural", 2000.0)
            frame = SensoryFrame(ts, channels={"flyvis_t4t5": drive})
            with mock.patch.object(engine.np, "unique", wraps=engine.np.unique) as unique:
                _, first, _ = e._compose_drives(frame)
                _, second, _ = e._compose_drives(frame)
            self.assertEqual(unique.call_count, 1)
            np.testing.assert_array_equal(first, second)

            # Writable inputs remain fully revalidated because an external
            # caller may mutate them between frames.
            mutable = {
                **drive,
                "target_indices": np.array([0, 2], dtype=np.int64),
            }
            mutable_frame = SensoryFrame(ts, channels={"flyvis_t4t5": mutable})
            with mock.patch.object(engine.np, "unique", wraps=engine.np.unique) as unique:
                e._compose_drives(mutable_frame)
                e._compose_drives(mutable_frame)
            self.assertEqual(unique.call_count, 2)

    def test_flyvis_drive_echo_can_be_bounded_without_dense_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(
                path, backend="cpu", device="cpu",
                include_flyvis_drive_echo=False,
            )
            ts = Timestamp(0.0, 0, "neural", 2000.0)
            drive = {
                "target_indices": np.array([0, 2], dtype=np.int64),
                "values": np.array([3.0, -4.0], dtype=np.float32),
                "units": "current_mV_per_ms",
            }
            out = e.step(SensoryFrame(ts, channels={"flyvis_t4t5": drive}), ts)
            summary = out.named_outputs["flyvis_t4t5_drive"]
            self.assertEqual(summary["units"], "current_mV_per_ms")
            self.assertEqual(summary["target_count"], 2)
            self.assertEqual(summary["value_count"], 2)
            self.assertEqual(len(summary["target_indices_sha256"]), 64)
            self.assertEqual(len(summary["values_sha256"]), 64)
            self.assertNotIn("target_indices", out.named_outputs)
            self.assertNotIn("values", summary)
            self.assertEqual(
                out.named_outputs["flyvis_target_mapping"], summary,
            )
            precomputed = {
                "units": "current_mV_per_ms",
                "target_count": 2,
                "value_count": 2,
                "target_indices_sha256": "a" * 64,
                "values_sha256": "b" * 64,
                "values_min": -4.0,
                "values_max": 3.0,
                "values_sum": -1.0,
            }
            next_ts = Timestamp(0.0005, 1, "neural", 2000.0)
            fast_drive = {**drive, "_telemetry_summary": precomputed}
            fast = e.step(
                SensoryFrame(next_ts, channels={"flyvis_t4t5": fast_drive}),
                next_ts,
            )
            self.assertEqual(fast.named_outputs["flyvis_t4t5_drive"], precomputed)

    def test_visual_and_tonic_current_drives_add_and_visual_bound_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(
                path, backend="cpu", device="cpu",
                current_mV_per_ms={"A": 1.0},
                flyvis_targets={0: {"population": "A", "kind": "current_mV_per_ms", "gain": 2.0}},
            )
            ts = Timestamp(0.0, 0, "neural", 2000.0)
            _, current, _ = e._compose_drives(
                SensoryFrame(ts, channels={"flyvis_t4t5": [3.0]})
            )
            np.testing.assert_array_equal(current, np.array([7.0, 0.0, 7.0, 0.0], dtype=np.float32))

            too_large = engine.MaleCNSEngine(
                path, backend="cpu", device="cpu",
                flyvis_targets={0: {"population": "A", "kind": "current_mV_per_ms", "gain": 10.0}},
            )
            with self.assertRaisesRegex(engine.DriveError, "calibrated bound"):
                too_large._compose_drives(
                    SensoryFrame(ts, channels={"flyvis_t4t5": [3.0]})
                )

    def test_cuda_gate_and_cpu_triton_request_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            # This host has the gate closed.  The exception must happen before
            # Torch is imported by the engine.
            if engine._gate_blockers():
                self.assertNotIn("torch", sys.modules)
                with self.assertRaises(engine.GateClosedError):
                    engine.MaleCNSEngine(path, backend="torch", device="cuda")
                self.assertNotIn("torch", sys.modules)
            with self.assertRaisesRegex(engine.UnsupportedBackendError, "requires device"):
                engine.MaleCNSEngine(path, backend="triton", device="cpu")

    def test_mechanosensory_population_drives_are_explicit_bounded_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mechanosensory_csr(Path(tmp))
            e = engine.MaleCNSEngine(path, backend="cpu", device="cpu")
            ts = Timestamp(0.0, 0, "neural", 2000.0)
            frame = SensoryFrame(ts, channels={
                # The sensor publishes aliases for compatibility.  They must
                # resolve once, not triple-inject the JO population.
                "population:jo": {"value": 1.0, "units": "normalized", "source": "handbuilt"},
                "population:jo_wind_gravity": 1.0,
                "population:wind_gravity": 1.0,
                "population:haltere": -1.0,
                "population:wing": 0.5,
                "population:wing_tactile": 0.9,
                "population:wing_proprioceptive": -0.9,
                "population:leg": 0.25,
                "population:leg_tactile": 0.8,
                "population:leg_proprioceptive": -0.8,
                "mech:wind_body_lateral": 1000.0,
                "population_drive_units": "normalized",
                "population_drive_range": [-1.0, 1.0],
            })
            _, current, _ = e._compose_drives(frame)
            self.assertAlmostEqual(float(current[0]), 0.4, places=6)
            self.assertAlmostEqual(float(current[1]), -0.35, places=6)
            # Aggregate wing/leg entries take precedence over component aliases.
            self.assertAlmostEqual(float(current[2]), 0.15, places=6)
            self.assertAlmostEqual(float(current[3]), 0.15, places=6)
            self.assertAlmostEqual(float(current[4]), 0.0625, places=6)
            self.assertAlmostEqual(float(current[5]), 0.0625, places=6)
            self.assertAlmostEqual(float(current[6]), 0.0625, places=6)
            details = e.last_named_outputs  # remains empty until a neural step
            self.assertEqual(details, {})
            out = e.step(frame, ts)
            mapping = out.named_outputs["mechanosensory_mapping"]
            self.assertEqual(set(mapping["applied"]), {"jo", "haltere", "wing", "leg"})
            self.assertIn("wing_tactile", mapping["shadowed"])
            self.assertIn("leg_proprioceptive", mapping["shadowed"])
            self.assertEqual(mapping["telemetry_ignored"], ["mech:wind_body_lateral"])

    def test_mechanosensory_mapping_is_unit_checked_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mechanosensory_csr(Path(tmp))
            e = engine.MaleCNSEngine(path, backend="cpu", device="cpu")
            for value, message in ((1.1, "outside"),
                                   ({"value": 0.2, "units": "mV"}, "units")):
                ts = Timestamp(0.0, 0, "neural", 2000.0)
                with self.assertRaisesRegex(engine.DriveError, message):
                    e.step(SensoryFrame(ts, channels={"population:jo": value}), ts)
            ts = Timestamp(0.0, 0, "neural", 2000.0)
            with self.assertRaisesRegex(engine.DriveError, "unknown"):
                e.step(SensoryFrame(ts, channels={"population:made_up": 0.2}), ts)

    def test_malformed_sensory_map_fails_closed_before_engine_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp) / "csr")
            config = json.loads((ROOT / "config" / "sensory_map.json").read_text(encoding="utf-8"))
            config["drive_mapping"]["entries"]["jo"]["gain_mV_per_ms"] = 0.6
            map_path = Path(tmp) / "bad_sensory_map.json"
            map_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(engine.DriveError, "gain/current bound"):
                engine.MaleCNSEngine(path, backend="cpu", device="cpu",
                                      sensory_map_path=map_path)

    def test_mechanosensory_drive_is_held_for_ten_half_millisecond_ticks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mechanosensory_csr(Path(tmp))
            e = engine.MaleCNSEngine(path, backend="cpu", device="cpu")
            outputs = []
            for tick in range(10):
                ts = Timestamp(tick * 0.0005, tick, "neural", 2000.0)
                frame = SensoryFrame(ts, channels={"population:jo": 0.75})
                outputs.append(e.step(frame, ts))
            drives = [item.named_outputs["mechanosensory_population_drives"]["jo"]
                      for item in outputs]
            self.assertEqual(drives, [0.75] * 10)
            self.assertEqual(e.step_index, 10)

    def test_triton_micro_csr_matches_torch_cuda_without_stimulus(self):
        try:
            import torch
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("Torch/Triton is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if engine._gate_blockers():
            self.skipTest("CUDA safety gate is closed")
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            eager = engine.MaleCNSEngine(path, backend="torch", device="cuda", seed=7)
            fused = engine.MaleCNSEngine(path, backend="triton", device="cuda", seed=7)
            for tick in range(12):
                ts = Timestamp(tick * 0.0005, tick, "neural", 2000.0)
                channels = {"current_mV_per_ms": {"A": 20.0}} if tick == 0 else {}
                frame = SensoryFrame(ts, channels=channels)
                out_eager = eager.step(frame, ts)
                out_fused = fused.step(frame, ts)
                np.testing.assert_array_equal(out_eager.spikes, out_fused.spikes)
                self.assertEqual(out_eager.named_outputs["deliveries"],
                                 out_fused.named_outputs["deliveries"])
                scalar_timing = out_fused.named_outputs["triton_step_timing"]
                self.assertEqual(scalar_timing["kernel_launches"], 3)
                self.assertEqual(scalar_timing["host_scalar_reads"], 1)
                np.testing.assert_allclose(eager.v.detach().cpu().numpy(),
                                           fused.v.detach().cpu().numpy(), atol=1e-5)
                np.testing.assert_allclose(eager.g.detach().cpu().numpy(),
                                           fused.g.detach().cpu().numpy(), atol=1e-5)
            window = engine.MaleCNSEngine(path, backend="triton", device="cuda", seed=7)
            window_ts = Timestamp(0.0, 0, "neural", 2000.0)
            window_out = window.step_window(
                SensoryFrame(window_ts, channels={"current_mV_per_ms": {"A": 20.0}}),
                window_ts, 10,
            )
            self.assertEqual(window_out.named_outputs["neural_window_steps"], 10)
            self.assertEqual(window_out.named_outputs["triton_step_timing"]["host_spike_copies"], 1)
            self.assertTrue(window_out.named_outputs["triton_step_timing"]["on_device_window"])
            timing = window_out.named_outputs["triton_step_timing"]
            # Two held-window counter clears, three dependent kernels per
            # tick, and one cumulative spike-tick add after the window; no
            # per-tick host scalar read or spike-vector D2H copy.
            self.assertEqual(timing["kernel_launches"], 33)
            self.assertEqual(timing["counter_reset_launches"], 2)
            self.assertEqual(timing["host_device_syncs"], 2)
            self.assertEqual(timing["host_scalar_reads"], 1)
            np.testing.assert_array_equal(
                window.spike_tick.detach().cpu().numpy(),
                window_out.spikes,
            )
            next_ts = Timestamp(0.005, 10, "neural", 2000.0)
            next_out = window.step_window(SensoryFrame(next_ts), next_ts, 10)
            np.testing.assert_array_equal(
                window.spike_tick.detach().cpu().numpy(),
                window_out.spikes + next_out.spikes,
            )

    def test_triton_scalar_host_spikes_stay_uint8(self):
        """The scalar Triton seam must not round-trip uint8 through bool."""

        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Counter:
            def item(self):
                return 3

        e = object.__new__(engine.MaleCNSEngine)
        e.deterministic_delivery = False
        e.spk = FakeTensor(np.array([0, 1, 0, 1], dtype=np.uint8))
        e._triton_delivery_counter = Counter()
        e._prepare_triton_inputs = lambda probability, current: (None, None)
        e._launch_triton_tick = lambda *args, **kwargs: None
        e._step_index = 0
        e._triton_host_device_syncs = 0
        e._triton_host_scalar_reads = 0
        e._triton_kernel_launches = 0
        e._triton_input_h2d_copies = 0
        e._triton_warmup_active = False
        e._triton_warmup_durations = []
        e._triton_recorded_steps = 0
        e._triton_recorded_first_wall_s = None
        e._triton_recorded_steady_total_s = 0.0
        e._triton_recorded_steady_steps = 0
        e.cuda_graph_requested = False

        spikes, deliveries = e._step_triton(
            np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32),
        )
        self.assertEqual(spikes.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(spikes, np.array([0, 1, 0, 1], dtype=np.uint8))
        self.assertEqual(deliveries, 3)

    def test_triton_cuda_graph_capture_replay_matches_triton_and_torch(self):
        try:
            import torch
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("Torch/Triton is not installed")
        if not torch.cuda.is_available() or engine._gate_blockers():
            self.skipTest("CUDA/Triton gate is unavailable")
        if not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph"):
            self.skipTest("Torch CUDA Graph API is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            kwargs = dict(csr_path=path, seed=41, poisson_hz={"B": 2000.0})
            graph = engine.MaleCNSEngine(
                backend="triton", device="cuda", cuda_graph=True, **kwargs,
            )
            plain = engine.MaleCNSEngine(
                backend="triton", device="cuda", cuda_graph=False, **kwargs,
            )
            eager = engine.MaleCNSEngine(
                backend="torch", device="cuda", cuda_graph=False, **kwargs,
            )
            graph.warmup(1)
            plain.warmup(1)
            eager.warmup(1)
            graph.reset(seed=41)
            plain.reset(seed=41)
            eager.reset(seed=41)
            timing_rows = []
            for index in range(2):
                ts = Timestamp(index * 0.005, index * 10, "neural", 2000.0)
                channels = (
                    {"current_mV_per_ms": {"A": 20.0}}
                    if index == 0 else {"current_mV_per_ms": {"A": -20.0}}
                )
                frame = SensoryFrame(ts, channels=channels)
                graph_out = graph.step_window(frame, ts, 10)
                plain_out = plain.step_window(frame, ts, 10)
                eager_out = eager.step_window(frame, ts, 10)
                np.testing.assert_array_equal(graph_out.spikes, plain_out.spikes)
                np.testing.assert_array_equal(graph_out.spikes, eager_out.spikes)
                self.assertEqual(graph_out.named_outputs["deliveries"],
                                 plain_out.named_outputs["deliveries"])
                self.assertEqual(graph_out.named_outputs["deliveries"],
                                 eager_out.named_outputs["deliveries"])
                np.testing.assert_allclose(
                    graph.v.detach().cpu().numpy(),
                    plain.v.detach().cpu().numpy(), atol=1e-5,
                )
                np.testing.assert_allclose(
                    graph.g.detach().cpu().numpy(),
                    plain.g.detach().cpu().numpy(), atol=1e-5,
                )
                timing_rows.append(graph_out.named_outputs["triton_step_timing"])
            self.assertEqual(timing_rows[0]["execution_mode"], "cuda_graph_capture")
            self.assertEqual(timing_rows[0]["graph_kernel_nodes"], 33)
            self.assertEqual(timing_rows[0]["graph_host_launches"], 1)
            self.assertEqual(timing_rows[0]["host_device_syncs"], 3)
            self.assertEqual(timing_rows[1]["execution_mode"], "cuda_graph_replay")
            self.assertEqual(timing_rows[1]["graph_host_launches"], 1)
            self.assertEqual(timing_rows[1]["host_device_syncs"], 2)
            self.assertEqual(graph.step_index, 20)
            self.assertEqual(int(graph._triton_step_t.item()), 20)
            self.assertEqual(graph.manifest()["triton_timing"]["cuda_graph_replay_count"], 2)

    def test_factory_auto_uses_cpu_when_gate_is_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            with mock.patch.object(engine, "_gate_blockers",
                                   return_value=("test gate blocker",)):
                e = engine.create_engine(seed=17, device="auto", backend="auto",
                                         csr_path=path)
            self.assertEqual((e.device, e.backend), ("cpu", "numpy_cpu"))
            self.assertEqual(e.seed, 17)

    def test_factory_explicit_cpu_skips_gate_and_triton_requires_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            with mock.patch.object(engine, "_gate_blockers",
                                   side_effect=AssertionError("CPU must not query gate")):
                e = engine.create_engine(seed=18, device="cpu", backend="auto",
                                         csr_path=path, max_vram_gib=15.5)
            self.assertEqual(e.backend, "numpy_cpu")
            self.assertEqual(e.manifest()["max_vram_gib"], 15.5)
            with self.assertRaisesRegex(engine.UnsupportedBackendError, "requires device"):
                engine.create_engine(seed=18, device="cpu", backend="triton",
                                     csr_path=path)

    def test_cuda_graph_opt_in_falls_back_explicitly_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(
                path, backend="cpu", device="cpu", cuda_graph=True,
            )
            timing = e.manifest()["triton_timing"]
            self.assertTrue(timing["cuda_graph_requested"])
            self.assertFalse(timing["cuda_graph_enabled"])
            self.assertIn("requires backend='triton'", timing["cuda_graph_reason"])

    def test_cuda_graph_replay_failure_fails_closed_without_fallback(self):
        """A partial replay must not continue through uncaptured kernels."""

        class FailingGraph:
            def replay(self):
                raise RuntimeError("injected replay failure")

        # Exercise the replay exception branch without importing CUDA or
        # constructing a full engine.  Capture failures have their own tested
        # restore path; replay failures intentionally fail closed because no
        # per-replay full-state snapshot is maintained.
        e = object.__new__(engine.MaleCNSEngine)
        e.deterministic_delivery = False
        e._triton_cuda_graph_enabled = True
        e._triton_cuda_graph_reason = None
        e._triton_cuda_graphs = {
            10: {"graph": FailingGraph(), "captured_kernel_nodes": 33},
        }
        e._triton_cuda_graph_last_phase = "replay"
        e._triton_cuda_graph_capture_count = 1
        e._triton_cuda_graph_replay_count = 1
        e._triton_cuda_graph_host_launches = 1
        e._triton_cuda_graph_pointer_updates = 0
        e._triton_cuda_graph_step_shadow = 10
        e._triton_cuda_graph_seed_shadow = 7
        e._triton_kernel_launches = 7
        e._step_index = 10
        e._prepare_triton_cuda_graph_scalars = lambda: 0

        with self.assertRaisesRegex(engine.EngineError, "capture/replay failed"):
            e._run_triton_window_cuda_graph(object(), object(), requested=10)
        self.assertFalse(e._triton_cuda_graph_enabled)
        self.assertEqual(e._triton_cuda_graphs, {})
        self.assertEqual(e._triton_kernel_launches, 7)

    def test_factory_explicit_cuda_never_downgrades_when_gate_is_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            with mock.patch.object(engine, "_gate_blockers",
                                   return_value=("test gate blocker",)):
                with self.assertRaises(engine.GateClosedError):
                    engine.create_engine(seed=19, device="cuda", backend="auto",
                                         csr_path=path)

    def test_cns_substeps_hold_drive_and_advance_without_timestamp_error(self):
        """Two 5 ms frames must be ten valid 0.5 ms engine calls each."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))

            def factory(*, seed: int, device: str, backend: str):
                self.assertEqual((device, backend), ("cpu", "auto"))
                return engine.MaleCNSEngine(path, seed=seed, device="cpu", backend="cpu")

            adapter = MalecnsNeuralAdapter(
                SimpleNamespace(create_engine=factory), seed=23,
                device="cpu", backend="auto",
            )
            for frame_index in range(2):
                frame_t = frame_index * 0.005
                outer_ts = Timestamp(frame_t, frame_index * 10, "neural", 2000.0)
                outer = SensoryFrame(
                    outer_ts, channels={"current_mV_per_ms": {"A": 0.1}},
                )
                outputs = []
                for substep in range(10):
                    t_s = frame_t + substep * 0.0005
                    sub_ts = Timestamp(t_s, frame_index * 10 + substep,
                                       "neural", 2000.0)
                    outputs.append(adapter.step(_substep_sensory(outer, sub_ts), sub_ts))
                aggregate = _aggregate_neural_outputs(outputs, outer_ts,
                                                       window_dt_s=0.005)
                self.assertEqual(aggregate.timestamp, outer_ts)
                self.assertEqual(aggregate.named_outputs["neural_window_steps"], 10)
            self.assertEqual(adapter.calls, 20)

    def test_window_api_returns_mean_rates_and_exact_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(path, backend="cpu", device="cpu")
            ts = Timestamp(0.0, 0, "neural", 2000.0)
            out = e.step_window(
                SensoryFrame(ts, channels={"current_mV_per_ms": {"A": 20.0}}),
                ts, 10,
            )
            self.assertEqual(out.named_outputs["neural_window_steps"], 10)
            self.assertTrue(out.named_outputs["spikes_are_window_counts"])
            self.assertEqual(out.timestamp.tick, 9)
            self.assertEqual(e.step_index, 10)
            np.testing.assert_allclose(
                out.rates_hz,
                out.spikes.astype(np.float32) * 200.0,
            )

    def test_triton_warmup_restores_state_and_reports_startup_cost(self):
        try:
            import torch
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("Torch/Triton is not installed")
        if not torch.cuda.is_available() or engine._gate_blockers():
            self.skipTest("CUDA/Triton gate is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            path = make_micro_csr(Path(tmp))
            e = engine.MaleCNSEngine(path, backend="triton", device="cuda", seed=9,
                                      poisson_hz={"B": 2000.0})
            before = e.snapshot()
            report = e.warmup(1)
            after = e.snapshot()
            self.assertTrue(report["state_restored"])
            self.assertTrue(report["startup_cost_visible"])
            self.assertEqual(e.step_index, 0)
            for name in before["arrays"]:
                np.testing.assert_array_equal(before["arrays"][name], after["arrays"][name])
            self.assertTrue(e.manifest()["triton_timing"]["warmup"]["performed"])


if __name__ == "__main__":
    unittest.main()
