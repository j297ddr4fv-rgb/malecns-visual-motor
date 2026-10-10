#!/usr/bin/env python3
"""CPU-only fixtures for the read-only MaleCNS escape-circuit probe."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import probe_escape_circuit as probe  # noqa: E402


class ProbeEscapeCircuitTest(unittest.TestCase):
    def test_import_does_not_load_gpu_runtime(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r); "
            "import probe_escape_circuit; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules"
        ) % str(ROOT / "scripts")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nested_telemetry_extracts_first_peak_and_collision_latency(self) -> None:
        rows = [
            {
                "time_ms": 0.0,
                "neurons": 12,
                "neural": {"named_outputs": {
                    "population_rates_hz": {
                        "LC4/L": 0.0, "LC4/R": 0.0, "DNp01": 0.0,
                    },
                    "active_neurons": 2,
                    "spike_count": 3,
                }},
            },
            {
                "time_ms": 10.0,
                "neural": {"named_outputs": {
                    "population_rates_hz": {
                        "LC4/L": 80.0, "LC4/R": 20.0, "DNp01": 100.0,
                    },
                    "active_neurons": 10,
                    "spike_count": 18,
                }},
            },
            {
                "time_ms": 20.0,
                "collision_time_ms": 50.0,
                "neural": {"named_outputs": {
                    "population_rates_hz": {
                        "LC4/L": 40.0, "LC4/R": 60.0, "DNp01": 250.0,
                    },
                    "active_neurons": 12,
                    "spike_count": 21,
                }},
            },
        ]
        trace = probe.load_trace(rows, label="palm")
        report = probe.analyze_trace(trace)
        dn = report["populations"]["DNp01"]
        self.assertEqual(dn["first_spike_ms"], 10.0)
        self.assertEqual(dn["peak_rate_hz"], 250.0)
        self.assertEqual(dn["relative_collision_ms"], 40.0)
        self.assertEqual(dn["peak_relative_collision_ms"], 30.0)
        self.assertEqual(report["global_activity"]["active_neurons_peak"], 12.0)
        self.assertAlmostEqual(report["global_activity"]["peak_active_fraction"], 12 / 12)
        direction = report["direction_differences"]["LC4"]
        self.assertEqual(direction["first_left_ms"], 10.0)
        self.assertEqual(direction["first_right_ms"], 10.0)
        self.assertAlmostEqual(direction["mean_left_minus_right"], 40.0 / 3.0)

    def test_rows_with_explicit_count_maps_prefer_counts_for_first_spike(self) -> None:
        rows = [
            {"t_ms": 0, "rates": {"DNp01": 0}, "population_spike_counts": {"DNp01": 0}},
            {"t_ms": 5, "rates": {"DNp01": 40}, "population_spike_counts": {"DNp01": 1}},
            # A rate can be non-zero in a compact average while no event was
            # recorded in the current window; counts remain the event source.
            {"t_ms": 10, "rates": {"DNp01": 20}, "population_spike_counts": {"DNp01": 0}},
        ]
        trace = probe.load_trace(rows, label="loom", collision_time_ms=20)
        metric = probe.analyze_trace(trace)["populations"]["DNp01"]
        self.assertEqual(metric["first_spike_ms"], 5.0)
        self.assertEqual(metric["peak_rate_hz"], 40.0)
        self.assertEqual(metric["total_spike_count"], 1.0)

    def test_declared_network_saturation_is_preserved_separately_from_active_fraction(self) -> None:
        rows = [
            {"time_ms": 0, "network": {
                "active_neurons": 10,
                "active_fraction": 0.1,
                "saturation_fraction": 0.02,
                "global_saturation": False,
            }},
            {"time_ms": 5, "network": {
                "active_neurons": 20,
                "active_fraction": 0.2,
                "saturation_fraction": 0.07,
                "global_saturation": True,
            }},
        ]
        trace = probe.load_trace(rows, label="loom", neuron_count=100)
        network = probe.analyze_trace(trace)["global_activity"]
        self.assertEqual(network["peak_active_fraction"], 0.2)
        self.assertEqual(network["global_saturation_fraction"], 0.07)
        self.assertEqual(network["peak_declared_saturation_fraction"], 0.07)
        self.assertTrue(network["global_saturation_observed"])
        self.assertEqual(network["saturation_source"], "telemetry_saturation_fraction")

    def test_compare_sham_loom_palm_reports_protocols_and_deltas(self) -> None:
        def make(label: str, dn_rate: list[float], active: list[float]) -> probe.NeuralTrace:
            return probe.trace_from_arrays(
                [0, 10, 20],
                {"LC4": [0, 10, 30], "DNp01": dn_rate},
                active_neurons=active,
                spike_counts=active,
                neuron_count=100,
                collision_time_ms=30,
                label=label,
            )

        result = probe.compare_traces({
            "sham": make("sham", [0, 0, 0], [1, 2, 3]),
            "loom": make("loom", [0, 50, 100], [2, 5, 10]),
            "palm": make("palm", [0, 80, 120], [3, 9, 20]),
        })
        self.assertEqual(result["baseline"], "sham")
        self.assertEqual(result["notes"]["protocols"], {
            "sham": "sham", "loom": "loom", "palm": "palm",
        })
        self.assertEqual(result["comparisons_vs_baseline"]["loom"]["populations"]["DNp01"]["peak_rate_hz"], 100.0)
        self.assertEqual(result["comparisons"]["palm"]["global_activity"]["active_neurons_peak"], 17.0)
        # There is no action field or threshold decision in a comparison.
        self.assertNotIn("action", result)
        self.assertNotIn("motor", result)

    def test_sparse_npz_trace_is_read_offline_with_optional_population_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "palm-spikes.npz"
            np.savez(
                path,
                times=np.array([0.0, 5.0, 10.0]),
                offsets=np.array([0, 1, 3, 4]),
                neurons=np.array([0, 1, 3, 1], dtype=np.int32),
                dt_ms=np.array(5.0),
                meta=json.dumps({"neurons": 8}),
            )
            trace = probe.load_trace(
                path,
                label="palm",
                population_indices={"DNp01": np.array([1, 3])},
            )
            report = probe.analyze_trace(trace)
        self.assertEqual(report["global_activity"]["active_neurons_peak"], 2.0)
        self.assertEqual(report["global_activity"]["spike_count_total"], 4.0)
        dn = report["populations"]["DNp01"]
        self.assertEqual(dn["first_spike_ms"], 5.0)
        self.assertEqual(dn["peak_rate_hz"], 200.0)
        self.assertAlmostEqual(report["global_activity"]["peak_active_fraction"], 0.25)

    def test_neural_output_like_objects_are_accepted_without_stepping_engine(self) -> None:
        class Timestamp:
            def __init__(self, t_s: float) -> None:
                self.t_s = t_s

        class Output:
            def __init__(self, t_s: float, rate: float) -> None:
                self.timestamp = Timestamp(t_s)
                self.rates_hz = np.zeros(4, dtype=np.float32)
                self.spikes = np.array([0, int(rate > 0), 0, 0], dtype=np.uint8)
                self.named_outputs = {
                    "population_rates_hz": {"DNp01": rate},
                    "active_neurons": int(rate > 0),
                    "spike_count": int(rate > 0),
                }

        trace = probe.trace_from_outputs(
            [Output(0.0, 0.0), Output(0.005, 200.0)],
            label="loom",
            collision_time_ms=20,
        )
        metric = probe.analyze_trace(trace)["populations"]["DNp01"]
        self.assertEqual(metric["first_spike_ms"], 5.0)
        self.assertEqual(metric["peak_rate_hz"], 200.0)
        self.assertEqual(metric["relative_collision_ms"], 15.0)

    def test_dense_output_can_be_reduced_with_population_indices(self) -> None:
        class Timestamp:
            def __init__(self, t_s: float) -> None:
                self.t_s = t_s

        class Output:
            def __init__(self, t_s: float, value: float) -> None:
                self.timestamp = Timestamp(t_s)
                self.rates_hz = np.array([0.0, value, 0.0, value], dtype=np.float32)
                self.spikes = np.array([0, int(value > 0), 0, int(value > 0)], dtype=np.uint8)
                self.named_outputs = {"active_neurons": int(value > 0), "spike_count": int(value > 0)}

        trace = probe.trace_from_outputs(
            [Output(0.0, 0.0), Output(0.005, 200.0)],
            label="loom",
            population_indices={"DNp01": np.array([1, 3])},
        )
        metric = probe.analyze_trace(trace)["populations"]["DNp01"]
        self.assertEqual(metric["first_spike_ms"], 5.0)
        self.assertEqual(metric["peak_rate_hz"], 200.0)
        self.assertEqual(metric["peak_spike_count"], 2.0)

    def test_json_report_cli_writes_expected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sham = root / "sham.json"
            loom = root / "loom.json"
            payload = {
                "times_ms": [0, 10],
                "population_rates_hz": {"DNp01": [0, 50]},
                "active_neurons": [0, 2],
                "neuron_count": 10,
                "collision_time_ms": 20,
            }
            sham.write_text(json.dumps({**payload, "population_rates_hz": {"DNp01": [0, 0]}}), encoding="utf-8")
            loom.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "report.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "probe_escape_circuit.py"),
                "--trace", f"sham={sham}", "--trace", f"loom={loom}", "--out", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], probe.SCHEMA_VERSION)
        self.assertEqual(report["baseline"], "sham")
        self.assertEqual(report["traces"]["loom"]["populations"]["DNp01"]["first_spike_ms"], 10.0)

    def test_compact_array_mapping_accepts_rates_alias(self) -> None:
        trace = probe.load_trace({
            "times_ms": [0, 5],
            "rates": {"DNp06": [0, 30]},
            "active_neurons": [0, 1],
            "neuron_count": 4,
        }, label="loom")
        metric = probe.analyze_trace(trace)["populations"]["DNp06"]
        self.assertEqual(metric["first_spike_ms"], 5.0)
        self.assertEqual(metric["peak_rate_hz"], 30.0)

    def test_summary_metadata_uses_absolute_collision_and_positive_lead_time(self) -> None:
        # Some recorded summaries contain a zero-valued ``collision_ms`` event
        # field alongside the absolute impact time in nested metadata.  The
        # latter is the timing annotation for relative-collision metrics.
        payload = {
            "collision_ms": 0,
            "metadata": {"impact_ms": 2000.0},
            "ticks": [
                {"t_ms": 1800.0, "rates": {"DNp01": 0.0}},
                {"t_ms": 1830.0, "rates": {"DNp01": 100.0}},
            ],
        }
        trace = probe.load_trace(payload, label="palm")
        metric = probe.analyze_trace(trace, populations=("DNp01",))["populations"]["DNp01"]
        self.assertEqual(trace.collision_time_ms, 2000.0)
        self.assertEqual(metric["first_spike_ms"], 1830.0)
        self.assertEqual(metric["peak_relative_collision_ms"], 170.0)

    @unittest.skipUnless(
        probe.DEFAULT_CSR.is_dir(),
        "the default MaleCNS CSR is not installed on this host",
    )
    def test_sparse_npz_auto_resolves_default_csr_and_sibling_summary(self) -> None:
        # Dense neuron IDs in spikes.npz should be annotated without requiring
        # a caller-side map when the documented default CSR is available.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "spikes.npz"
            # ID 0 is DNp01 and 150002 is TTMn in the default CSR.  Keep the
            # fixture tiny while exercising the named-population resolver.
            np.savez(
                path,
                times=np.array([0.0, 5.0]),
                offsets=np.array([0, 1, 2]),
                neurons=np.array([0, 150002], dtype=np.int32),
            )
            (root / "summary.json").write_text(
                json.dumps({
                    "neuron_count": 176422,
                    "metadata": {"impact_ms": 20.0},
                    "timing_contract": {"trace_frame_ms": 5.0},
                }),
                encoding="utf-8",
            )
            trace = probe.load_trace(path, label="palm")
            report = probe.analyze_trace(trace)

        self.assertEqual(trace.neuron_count, 176422)
        self.assertEqual(trace.collision_time_ms, 20.0)
        self.assertEqual(
            report["populations"]["DNp01"]["first_spike_ms"],
            0.0,
        )
        self.assertEqual(
            report["populations"]["TTMn"]["first_spike_ms"],
            5.0,
        )
        self.assertEqual(report["populations"]["DNp01"]["observed"], True)
        self.assertEqual(report["populations"]["TTMn"]["observed"], True)
        self.assertEqual(report["populations"]["TTMn"]["peak_rate_hz"], 100.0)

    def test_sparse_npz_without_mapping_fails_with_actionable_annotation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "spikes.npz"
            np.savez(
                path,
                times=np.array([0.0]),
                offsets=np.array([0, 1]),
                neurons=np.array([0], dtype=np.int32),
            )
            original_csr = probe.DEFAULT_CSR
            try:
                probe.DEFAULT_CSR = root / "missing-csr"
                with self.assertRaisesRegex(probe.ProbeError, "population mapping"):
                    probe.load_trace(path, label="loom")
            finally:
                probe.DEFAULT_CSR = original_csr


if __name__ == "__main__":
    unittest.main()
