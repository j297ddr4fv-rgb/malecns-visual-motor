#!/usr/bin/env python3
"""Focused tests for the read-only reproducibility/performance gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import engineering_gate as gate  # noqa: E402


GiB = 1024**3


class EngineeringGateTest(unittest.TestCase):
    def _write_artifact(self, directory: Path) -> dict[str, object]:
        config = {"schema_version": "test-config-1", "seed": 1729, "gain": 0.65}
        (directory / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

        trajectory_rows = [
            {"time_s": 0.0, "position_m": [0.0, 0.0, 0.0], "velocity_mps": [0.0, 0.0, 0.0]},
            {"time_s": 0.005, "position_m": [0.0, 0.0, 0.1], "velocity_mps": [0.0, 0.0, 1.0]},
        ]
        (directory / "trajectory.json").write_text(
            json.dumps({"trajectory": trajectory_rows, "metadata": {"units": "SI"}}, sort_keys=True),
            encoding="utf-8",
        )
        telemetry_rows = [
            {"time_s": 0.0, "stage": "warmup"},
            {"time_s": 0.005, "stage": "measure"},
        ]
        (directory / "telemetry.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in telemetry_rows),
            encoding="utf-8",
        )

        manifest: dict[str, object] = {
            "schema_version": "engineering-manifest-1",
            "run_id": "unit-test-1729",
            "seed": 1729,
            "deterministic": True,
            "config": {
                "path": "config.json",
                "sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
            },
            "unit_tests": {
                "command": "python3 -m unittest discover -s tests -p 'test_*.py' -v",
                "status": "PASS",
                "passed": 120,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
            },
            "preflight": {
                "gpu_requested": False,
                "cuda_gate": "not_requested",
                "uvm_disable_hmm": "not_applicable",
                "nvidia_smi": {"present": False, "query": "not requested"},
                "cgroup": {"memory_max_bytes": 5 * GiB, "cpu_max": "max"},
            },
            "authority": {
                "controller_authority": "none",
                "writes_body_state": False,
                "writes_motor_command": False,
            },
            "resources": {
                "limits": {
                    "cgroup_memory_max_bytes": 5 * GiB,
                    "host_rss_max_bytes": 5 * GiB,
                },
                "usage": {"peak_host_rss_bytes": 1 * GiB},
            },
            "artifact_budget": {
                "max_total_bytes": 32 * 1024 * 1024,
                "max_file_bytes": {
                    "manifest.json": 2 * 1024 * 1024,
                    "trajectory.json": 2 * 1024 * 1024,
                    "telemetry.jsonl": 2 * 1024 * 1024,
                    "config.json": 2 * 1024 * 1024,
                },
            },
            "column_hashes": {
                "algorithm": "sha256-canonical-json-lines-v1",
                "files": {
                    "trajectory.json": {
                        "columns": {
                            "time_s": gate.column_hash(directory / "trajectory.json", "time_s"),
                            "position_m": gate.column_hash(directory / "trajectory.json", "position_m"),
                            "velocity_mps": gate.column_hash(directory / "trajectory.json", "velocity_mps"),
                        }
                    },
                    "telemetry.jsonl": {
                        "columns": {
                            "time_s": gate.column_hash(directory / "telemetry.jsonl", "time_s"),
                        }
                    },
                },
            },
            "equivalence": {
                "status": "PASS",
                "reference_id": "cpu-reference-1729",
                "metrics": {"max_abs": 1.0e-7, "max_rel": 1.0e-6, "rmse": 1.0e-7},
                "tolerances": {"max_abs": 1.0e-5, "max_rel": 1.0e-4, "rmse": 1.0e-5},
            },
            "benchmark": {
                "name": "malecns-step",
                "version": "1",
                "backend": "numpy_cpu",
                "device": "cpu",
                "dtype": "float32",
                "warmup_iterations": 5,
                "measurement_iterations": 10,
                "sample_count": 10,
                "simulated_duration_s": 0.05,
                "wall_time_s": 0.02,
                "throughput": {"value": 500.0, "unit": "simulated_s/s"},
                "synchronization": {"before": False, "after": False},
                "timestamp_utc": "2026-09-21T00:00:00Z",
                "command": "python3 scripts/benchmark.py --backend numpy_cpu",
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    @staticmethod
    def _warp_benchmark(name: str, *, includes_jit: bool, wall_time_s: float) -> dict[str, object]:
        return {
            "name": name,
            "version": "warp-bench-1",
            "backend": "warp_cuda",
            "device": "cuda:0",
            "dtype": "float32",
            "warmup_iterations": 5,
            "measurement_iterations": 10,
            "sample_count": 10,
            "simulated_duration_s": 0.05,
            "wall_time_s": wall_time_s,
            "throughput": {"value": 500.0, "unit": "simulated_s/s"},
            "synchronization": {"before": True, "after": True},
            "timestamp_utc": "2026-09-21T00:00:00Z",
            "command": "python3 scripts/warp_benchmark.py --backend warp_cuda",
            "includes_jit": includes_jit,
        }

    def _make_warp_artifact(self, directory: Path) -> dict[str, object]:
        manifest = self._write_artifact(directory)
        scene = {"schema_version": "warp-scene-1", "seed": 1729, "objects": 2}
        (directory / "scene.json").write_text(json.dumps(scene, sort_keys=True), encoding="utf-8")
        exact_rows = [
            {"shape": [1.0, 2.0, 3.0], "direction": [0.0, 0.0, 1.0]},
            {"shape": [4.0, 5.0, 6.0], "direction": [1.0, 0.0, 0.0]},
        ]
        (directory / "exact_columns.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in exact_rows),
            encoding="utf-8",
        )
        manifest.update(
            {
                "renderer_backend": "warp_cuda",
                "requested_renderer_backend": "warp_cuda",
                "effective_renderer_backend": "warp_cuda",
            }
        )
        manifest["preflight"].update(
            {
                "gpu_requested": True,
                "cuda_gate": "open",
                "uvm_disable_hmm": "Y",
                "nvidia_smi": {"present": True, "query": "NVIDIA RTX 5060 Ti, 16 GiB"},
            }
        )
        manifest["resources"]["limits"].update(
            {
                "cuda_allocated_max_bytes": int(15.5 * GiB),
                "cuda_reserved_max_bytes": int(15.5 * GiB),
            }
        )
        manifest["resources"]["usage"].update(
            {
                "peak_cuda_allocated_bytes": 2 * GiB,
                "peak_cuda_reserved_bytes": 3 * GiB,
            }
        )
        manifest["benchmark"].update(
            {
                "backend": "warp_cuda",
                "device": "cuda:0",
                "synchronization": {"before": True, "after": True},
            }
        )
        manifest["warp"] = {
            "version": "1.8.0",
            "gpu": {"name": "RTX 5060 Ti", "arch": "sm_120", "compute_capability": "12.0"},
            "config": {
                "path": "config.json",
                "sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
            },
            "scene": {
                "path": "scene.json",
                "sha256": hashlib.sha256((directory / "scene.json").read_bytes()).hexdigest(),
            },
            "exact_columns": {
                "file": "exact_columns.jsonl",
                "shape_column": "shape",
                "direction_column": "direction",
                "shape_hash": gate.column_hash(directory / "exact_columns.jsonl", "shape"),
                "direction_hash": gate.column_hash(directory / "exact_columns.jsonl", "direction"),
                "algorithm": "sha256-canonical-json-lines-v1",
            },
            "benchmarks": {
                "cold_jit": self._warp_benchmark("warp-cold-jit", includes_jit=True, wall_time_s=2.0),
                "steady": self._warp_benchmark("warp-steady", includes_jit=False, wall_time_s=0.2),
            },
            "analytic_reference": {
                "status": "PASS",
                "reference_id": "analytic-reference-1729",
                "max_abs": 1.0e-7,
                "mean_abs": 1.0e-8,
                "mismatch_count": 0,
                "tolerance": {"max_abs": 1.0e-5, "mean_abs": 1.0e-6, "mismatch_count": 0},
            },
            "resources": {
                "peak_host_rss_bytes": 1 * GiB,
                "peak_cuda_allocated_bytes": 2 * GiB,
                "peak_cuda_reserved_bytes": 3 * GiB,
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    @staticmethod
    def _add_performance_acceptance(
        manifest: dict[str, object], *, wall_time_s: float = 20.0
    ) -> None:
        """Add a production full-run record to an otherwise valid fixture."""

        simulated_duration_s = 10.0
        manifest["preflight"].update(
            {
                "gpu_requested": True,
                "cuda_gate": "open",
                "uvm_disable_hmm": "Y",
                "nvidia_smi": {"present": True, "query": "RTX 5060 Ti"},
            }
        )
        manifest["resources"]["limits"].update(
            {
                "cuda_allocated_max_bytes": int(15.5 * GiB),
                "cuda_reserved_max_bytes": int(15.5 * GiB),
            }
        )
        manifest["resources"]["usage"].update(
            {
                "peak_cuda_allocated_bytes": 2 * GiB,
                "peak_cuda_reserved_bytes": 3 * GiB,
            }
        )
        manifest["benchmark"].update(
            {
                "backend": "triton",
                "device": "cuda:0",
                "synchronization": {"before": True, "after": True},
            }
        )
        manifest["requested_execution"] = {
            "renderer_backend": "warp_cuda",
            "neural_backend": "triton",
            "neural_device": "cuda",
            "neural_cuda_graph": True,
            "vision_device": "cuda:0",
        }
        manifest["effective_execution"] = {
            "renderer_backend": "warp_cuda",
            "neural_backend": "triton",
            "neural_device": "cuda",
            "neural_cuda_graph": True,
            "vision_backend": "flyvis",
            "vision_device": "cuda:0",
        }
        # ``malecns`` is the requested model family; ``triton`` is its
        # selected implementation.  This alias must not be treated as a
        # fallback or mismatch by the production contract.
        manifest["requested_backends"] = {
            "neural": "malecns",
            "vision": "flyvis",
            "body": "reduced",
        }
        manifest["effective_backends"] = {
            "renderer": "warp_cuda",
            "neural": "triton",
            "vision": "flyvis",
            "body": "flight_dynamics_reduced_6dof",
        }
        manifest["components"] = {
            name: {"is_stub": False}
            for name in ("world", "vision", "neural", "body")
        }
        manifest["performance_acceptance"] = {
            "type": "full_run_performance_acceptance",
            "simulated_duration_s": simulated_duration_s,
            "wall_time_s": wall_time_s,
            "real_time_factor": simulated_duration_s / wall_time_s,
            "wall_time_scope": "startup+simulation+output",
        }

    def _write_acceptance_artifact(self, directory: Path, *, wall_time_s: float = 20.0) -> dict[str, object]:
        manifest = self._write_artifact(directory)
        self._add_performance_acceptance(manifest, wall_time_s=wall_time_s)
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest

    def test_complete_artifact_passes_all_engineering_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_artifact(directory)
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["read_only"])
        self.assertEqual(
            {name for name, check in report["checks"].items() if check["status"] == "PASS"},
            {
                "manifest_schema",
                "unit_tests",
                "preflight",
                "authority",
                "determinism",
                "column_hashes",
                "resources",
                "artifact_budget",
                "equivalence",
                "benchmark",
                "backend_consistency",
                "warp_cuda",
            },
        )

    def test_full_run_performance_acceptance_passes_at_twenty_second_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_acceptance_artifact(directory, wall_time_s=20.0)
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"]["performance_acceptance"]["status"], "PASS")

    def test_resident_run_performance_scope_is_explicitly_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_acceptance_artifact(directory, wall_time_s=17.5)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["performance_acceptance"]["wall_time_scope"] = (
                "resident-runner simulation+output"
            )
            (directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"]["performance_acceptance"]["status"], "PASS")

    def test_full_run_performance_acceptance_rejects_twenty_point_zero_zero_zero_one_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_acceptance_artifact(directory, wall_time_s=20.0001)
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any("measured real-time factor" in reason for reason in report["failures"])
        )

    def test_full_run_performance_acceptance_requires_measured_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_acceptance_artifact(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            del manifest["performance_acceptance"]["real_time_factor"]
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("missing 'real_time_factor'" in reason for reason in report["failures"]))

    def test_full_run_acceptance_recomputes_rtf_instead_of_trusting_supplied_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_acceptance_artifact(directory, wall_time_s=100.0)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["performance_acceptance"]["real_time_factor"] = 0.5
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("measured real-time factor" in reason for reason in report["failures"]))

    def test_production_acceptance_allows_model_family_to_implementation_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_acceptance_artifact(directory)
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"]["performance_acceptance"]["status"], "PASS")

    def test_production_acceptance_rejects_backend_fallback_or_stub_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_acceptance_artifact(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["effective_execution"]["renderer_backend"] = "analytic_reference"
            manifest["components"]["vision"]["is_stub"] = True
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("effective renderer_backend" in reason for reason in report["failures"]))
        self.assertTrue(any("component 'vision'.is_stub must be false" in reason for reason in report["failures"]))

    def test_warp_cuda_contract_passes_with_scene_columns_benchmarks_and_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._make_warp_artifact(directory)
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"]["warp_cuda"]["status"], "PASS")
        self.assertEqual(report["checks"]["backend_consistency"]["status"], "PASS")

    def test_warp_cuda_rejects_silent_fallback_and_exact_column_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._make_warp_artifact(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["effective_renderer_backend"] = "analytic_reference"
            manifest["warp"]["exact_columns"]["shape_hash"] = "0" * 64
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("differs from effective" in reason for reason in report["failures"]))
        self.assertTrue(any("Warp effective renderer" in reason for reason in report["failures"]))
        self.assertTrue(any("Warp exact-column shape hash mismatch" in reason for reason in report["failures"]))

    def test_warp_cuda_requires_separate_jit_steady_and_analytic_tolerances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._make_warp_artifact(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["warp"]["benchmarks"]["cold_jit"]["includes_jit"] = False
            manifest["warp"]["benchmarks"]["steady"]["includes_jit"] = True
            manifest["warp"]["analytic_reference"]["max_abs"] = 1.0
            manifest["warp"]["analytic_reference"]["tolerance"]["max_abs"] = 1.0e-3
            manifest["warp"]["resources"]["peak_cuda_reserved_bytes"] = 20 * GiB
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("cold_jit.includes_jit must be True" in reason for reason in report["failures"]))
        self.assertTrue(any("steady.includes_jit must be False" in reason for reason in report["failures"]))
        self.assertTrue(any("analytic_reference tolerance max_abs" in reason for reason in report["failures"]))
        self.assertTrue(
            any(
                "Warp resource peak_cuda_reserved_bytes exceeds" in reason
                for reason in report["failures"]
            )
        )

    def test_analytic_reference_renderer_without_warp_fields_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = self._write_artifact(directory)
            manifest.update(
                {
                    "renderer_backend": "analytic_reference",
                    "requested_renderer_backend": "analytic_reference",
                    "effective_renderer_backend": "analytic_reference",
                }
            )
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"]["warp_cuda"]["status"], "PASS")

    def test_any_explicit_requested_effective_backend_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = self._write_artifact(directory)
            manifest["requested_backend"] = "warp_cuda"
            manifest["effective_backend"] = "analytic_reference"
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("backend consistency" in reason for reason in report["failures"]))

    def test_column_hash_and_config_hash_fail_closed_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_artifact(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["column_hashes"]["files"]["trajectory.json"]["columns"]["time_s"] = "0" * 64
            (directory / "config.json").write_text('{"schema_version":"test-config-1","seed":7}', encoding="utf-8")
            before = {path.name: path.read_bytes() for path in directory.iterdir()}
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)
            after = {path.name: path.read_bytes() for path in directory.iterdir()}

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("column hash mismatch" in reason for reason in report["failures"]))
        self.assertTrue(any("config SHA-256 mismatch" in reason for reason in report["failures"]))
        self.assertEqual(before["trajectory.json"], after["trajectory.json"])
        self.assertFalse((directory / "engineering_gate_report.json").exists())

    def test_gpu_preflight_requires_gate_identity_and_synchronization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_artifact(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["preflight"].update(
                {
                    "gpu_requested": True,
                    "cuda_gate": "closed",
                    "uvm_disable_hmm": "N",
                    "nvidia_smi": {"present": False, "query": ""},
                }
            )
            manifest["resources"]["limits"].update(
                {"cuda_allocated_max_bytes": int(15.5 * GiB), "cuda_reserved_max_bytes": int(15.5 * GiB)}
            )
            manifest["resources"]["usage"].update(
                {"peak_cuda_allocated_bytes": 1 * GiB, "peak_cuda_reserved_bytes": 1 * GiB}
            )
            manifest["benchmark"].update(
                {
                    "device": "cuda:0",
                    "synchronization": {"before": False, "after": False},
                }
            )
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("GPU run requires cuda_gate" in reason for reason in report["failures"]))
        self.assertTrue(any("uvm_disable_hmm=Y" in reason for reason in report["failures"]))
        self.assertTrue(any("GPU benchmark must record synchronization" in reason for reason in report["failures"]))

    def test_controller_resource_equivalence_and_benchmark_failures_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_artifact(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["authority"]["controller_authority"] = "autopilot"
            manifest["authority"]["writes_motor_command"] = True
            manifest["resources"]["usage"]["peak_host_rss_bytes"] = 6 * GiB
            manifest["equivalence"]["metrics"]["max_abs"] = 1.0
            manifest["benchmark"]["measurement_iterations"] = 1
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = gate.audit_artifact(directory)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("controller authority must be 'none'" in reason for reason in report["failures"]))
        self.assertTrue(any("peak_host_rss_bytes" in reason and "exceeds" in reason for reason in report["failures"]))
        self.assertTrue(
            any(
                "equivalence max_abs" in reason and "exceeds tolerance" in reason
                for reason in report["failures"]
            )
        )
        self.assertTrue(any("measurement_iterations=1" in reason for reason in report["failures"]))

    def test_cli_returns_nonzero_and_does_not_emit_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_artifact(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["unit_tests"]["command"] = "python3 -m pytest"
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "engineering_gate.py"), str(directory), "--json"],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(proc.stdout)
            sidecars = list(directory.glob("*gate*.json"))

        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("unit-test command", "\n".join(report["failures"]))
        self.assertEqual(sidecars, [])


if __name__ == "__main__":
    unittest.main()
