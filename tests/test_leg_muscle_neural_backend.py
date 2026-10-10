"""Fail-closed backend-selection tests for the articulated leg runner."""
from __future__ import annotations

import builtins
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import leg_muscle_neural_backend as adapter  # noqa: E402


class LegMuscleNeuralBackendTests(unittest.TestCase):
    def test_cuda_gate_runs_before_factory_without_gpu_imports(self):
        events: list[str] = []
        imported_gpu_modules: list[str] = []
        fake_gate = SimpleNamespace(
            cuda_blockers=lambda: events.append("gate") or []
        )
        fake_engine = SimpleNamespace(backend="torch_cuda", device="cuda")
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"torch", "triton"}:
                imported_gpu_modules.append(name)
                raise AssertionError(f"unexpected GPU library import: {name}")
            return real_import(name, *args, **kwargs)

        def factory(**kwargs):
            events.append("factory")
            return fake_engine

        with (
            mock.patch.object(adapter, "_load_gate", return_value=fake_gate),
            mock.patch.object(adapter, "_engine_factory", side_effect=factory),
            mock.patch("builtins.__import__", side_effect=guarded_import),
        ):
            result = adapter.build_leg_neural_engine("csr", backend="torch_cuda")

        self.assertIs(result, fake_engine)
        self.assertEqual(events, ["gate", "factory"])
        self.assertEqual(imported_gpu_modules, [])

    def test_closed_cuda_gate_stops_before_engine_factory(self):
        fake_gate = SimpleNamespace(
            cuda_blockers=lambda: ["test gate blocker"]
        )
        with (
            mock.patch.object(adapter, "_load_gate", return_value=fake_gate),
            mock.patch.object(adapter, "_engine_factory") as factory,
        ):
            with self.assertRaises(adapter.NeuralBackendGateError) as caught:
                adapter.build_leg_neural_engine("csr", backend="triton")

        self.assertIn("test gate blocker", str(caught.exception))
        self.assertEqual(
            caught.exception.preflight_report["requested_backend"], "triton"
        )
        factory.assert_not_called()

    def test_cpu_preflight_does_not_load_or_query_cuda_gate(self):
        with mock.patch.object(
            adapter, "_load_gate", side_effect=AssertionError("gate must stay idle")
        ) as load_gate:
            report = adapter.neural_backend_preflight("cpu")
            fake_engine = SimpleNamespace(backend="numpy_cpu", device="cpu")
            with mock.patch.object(
                adapter, "_engine_factory", return_value=fake_engine
            ) as factory:
                result = adapter.build_leg_neural_engine("csr", backend="cpu")

        self.assertEqual(report, {
            "requested_backend": "cpu",
            "requires_cuda": False,
            "gate_checked": False,
            "gate_passed": True,
            "blockers": [],
        })
        load_gate.assert_not_called()
        factory.assert_called_once_with(
            csr_path="csr", seed=1729, backend="cpu", device="cpu",
            max_vram_gib=adapter.DEFAULT_MAX_VRAM_GIB,
        )
        self.assertIs(result, fake_engine)

    def test_exact_backend_mappings_and_engine_options(self):
        cases = (
            ("cpu", "numpy_cpu", "cpu", "cpu"),
            ("torch_cuda", "torch_cuda", "torch", "cuda"),
            ("triton", "triton", "triton", "cuda"),
        )
        for requested, effective, engine_backend, device in cases:
            with self.subTest(backend=requested):
                engine = SimpleNamespace(backend=effective, device=device)
                gate_patch = mock.patch.object(
                    adapter,
                    "_load_gate",
                    return_value=SimpleNamespace(cuda_blockers=lambda: []),
                )
                with gate_patch:
                    report = adapter.neural_backend_preflight(requested)
                    self.assertTrue(report["gate_passed"])
                    with mock.patch.object(
                        adapter, "_engine_factory", return_value=engine
                    ) as factory:
                        result = adapter.build_leg_neural_engine(
                            Path("/tmp/test-csr"), backend=requested, seed=41,
                            max_vram_gib=7.25,
                        )

                self.assertIs(result, engine)
                factory.assert_called_once_with(
                    csr_path=Path("/tmp/test-csr"),
                    seed=41,
                    backend=engine_backend,
                    device=device,
                    max_vram_gib=7.25,
                )

    def test_effective_backend_mismatch_is_rejected(self):
        wrong_engine = SimpleNamespace(backend="numpy_cpu", device="cpu")
        with (
            mock.patch.object(
                adapter,
                "_load_gate",
                return_value=SimpleNamespace(cuda_blockers=lambda: []),
            ),
            mock.patch.object(adapter, "_engine_factory", return_value=wrong_engine),
        ):
            with self.assertRaisesRegex(adapter.NeuralBackendError, "reported"):
                adapter.build_leg_neural_engine("csr", backend="torch_cuda")

    def test_backend_names_are_exact_and_invalid_name_never_queries_gate(self):
        with mock.patch.object(adapter, "_load_gate") as load_gate:
            for invalid in ("Torch_CUDA", "torch", "auto", "gpu", ""):
                with self.subTest(backend=invalid):
                    with self.assertRaises(ValueError):
                        adapter.neural_backend_preflight(invalid)
        load_gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
