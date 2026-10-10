#!/usr/bin/env python3
"""CPU-only preflight and mock-network tests for the real FlyVis adapter."""

from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import flyvis_runtime_adapter as adapter  # noqa: E402


class _FakeTorch:
    float32 = np.float32

    @staticmethod
    def as_tensor(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    @contextmanager
    def no_grad():
        yield

    @staticmethod
    @contextmanager
    def inference_mode():
        yield


class _NativeTensor:
    """Tiny tensor seam exercising the adapter's native gather path.

    It deliberately has no conversion to a host array until ``numpy`` is
    called on the reduced target vector.  This keeps the test independent of
    an installed Torch package (and preserves the repository's import-lazy
    tests) while still covering indexed gather and scatter reduction.
    """

    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = self.value.dtype
        self.device = "cpu"

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = tuple(np.asarray(item) for item in index)
        return _NativeTensor(self.value[index])

    def new_zeros(self, shape, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return _NativeTensor(np.zeros(shape, dtype=dtype))

    def scatter_add_(self, dim, index, source):
        if dim != 0:
            raise ValueError("mock native tensor only supports one-dimensional scatter")
        np.add.at(self.value, np.asarray(index), np.asarray(source))
        return self

    def __truediv__(self, other):
        return _NativeTensor(self.value / np.asarray(other))

    def __sub__(self, other):
        return _NativeTensor(self.value - np.asarray(other))

    def __mul__(self, other):
        return _NativeTensor(self.value * other)

    def zero_(self):
        self.value.fill(0)
        return self

    def copy_(self, other):
        self.value[...] = np.asarray(other)
        return self

    def to(self, *_args, dtype=None, **_kwargs):
        if dtype is not None:
            return _NativeTensor(self.value.astype(dtype))
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)

    def all(self):
        return np.asarray(np.all(self.value))


class _TrackingNativeTensor(_NativeTensor):
    """Native tensor seam that records device moves for baseline tests."""

    def __init__(self, value: object) -> None:
        super().__init__(value)
        self.to_calls = 0

    def to(self, *args, dtype=None, **kwargs):
        self.to_calls += 1
        if dtype is None:
            return self
        return super().to(*args, dtype=dtype, **kwargs)


class _NativeTorch(_FakeTorch):
    int64 = np.int64

    @staticmethod
    def as_tensor(value, dtype=None):
        return _NativeTensor(np.asarray(value, dtype=dtype))

    @staticmethod
    def isfinite(value):
        return _NativeTensor(np.isfinite(np.asarray(value)))


class _CountingNativeTorch(_NativeTorch):
    finite_calls = 0

    @staticmethod
    def isfinite(value):
        _CountingNativeTorch.finite_calls += 1
        return _NativeTensor(np.isfinite(np.asarray(value)))


class _DeviceVector:
    """Tensor-like CPU seam used to exercise the native-input branch."""

    device = "cpu"

    def __init__(self, value: object) -> None:
        self.value = np.asarray(value, dtype=np.float32)
        self.shape = self.value.shape

    def detach(self):
        return self

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


class _FakeStimulus:
    def __init__(self, n_columns: int, input_index: np.ndarray | None = None) -> None:
        if input_index is None:
            input_index = np.tile(np.arange(n_columns, dtype=np.int64), (2, 1))
        self.input_index = np.asarray(input_index, dtype=np.int64)
        self.buffer = np.zeros((2, 1, 1, n_columns), dtype=np.float32)
        self.seen: list[np.ndarray] = []

    def zero(self, n_samples: int = 2, n_frames: int = 1) -> None:
        self.buffer = np.zeros((n_samples, n_frames, 1, self.input_index.shape[1]), dtype=np.float32)

    def add_input(self, value: np.ndarray) -> None:
        self.seen.append(np.asarray(value).copy())
        self.buffer = np.asarray(value).copy()

    def __call__(self) -> np.ndarray:
        return self.buffer


class _FakeState:
    def __init__(self, activity: np.ndarray) -> None:
        self.nodes = types.SimpleNamespace(activity=np.asarray(activity, dtype=np.float32))


class _FakeNet:
    def __init__(self, node_types: list[str], n_columns: int) -> None:
        self.connectome = types.SimpleNamespace(
            nodes=types.SimpleNamespace(
                u=np.array([0, 0, 1, 1], dtype=np.int64),
                v=np.array([0, 0, 0, 0], dtype=np.int64),
                type=np.asarray(node_types),
            )
        )
        self.stimulus = _FakeStimulus(
            n_columns, np.array([[0, 2], [1, 3]], dtype=np.int64)
        )
        self.states_seen: list[object] = []
        self._base = np.full((2, len(node_types)), 0.5, dtype=np.float32)

    def eval(self) -> "_FakeNet":
        return self

    def steady_state(self, *, t_pre: float, dt: float, batch_size: int, value: float):
        self.steady_args = (t_pre, dt, batch_size, value)
        return _FakeState(np.full((2, len(self._base[0])), value, dtype=np.float32))

    def __call__(self, _stimulus, *, dt: float, state: object, as_states: bool):
        self.states_seen.append(state)
        current = self._base.copy()
        # The mock exposes direct input provenance on two subtype/side nodes.
        current[0, 0] = float(self.stimulus.buffer[0, 0, 0, 0])
        current[1, 1] = float(self.stimulus.buffer[1, 0, 0, 0])
        return [_FakeState(current)]


class _FakeView:
    def __init__(self, net: _FakeNet) -> None:
        self.connectome = net.connectome
        self.net = net

    def init_network(self):
        return self.net


class _GroupedFakeNet(_FakeNet):
    """Mock connectome with two fitted nodes in each mapped target group."""

    def __init__(self, n_columns: int) -> None:
        self.connectome = types.SimpleNamespace(
            nodes=types.SimpleNamespace(
                u=np.array([0, 0, 0, 0, 1, 1], dtype=np.int64),
                v=np.array([0, 0, 0, 0, 0, 0], dtype=np.int64),
                type=np.array(["T4a", "T4a", "T5a", "T5a", "T4a", "T5a"]),
            )
        )
        self.stimulus = _FakeStimulus(
            n_columns, np.array([[0, 4], [1, 5]], dtype=np.int64)
        )
        self.states_seen: list[object] = []
        self._base = np.full((2, 6), 0.5, dtype=np.float32)

    def eval(self) -> "_GroupedFakeNet":
        return self


def _make_assets(root: Path, *, n_columns: int = 2) -> tuple[Path, Path, Path, Path]:
    env = root / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").write_text("#!/bin/sh\n")
    site = env / "lib" / "python3.12" / "site-packages" / "flyvis"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("# mock package; never imported by tests\n")
    model = root / "model"
    (model / "results" / "flow" / "0000" / "000").mkdir(parents=True)
    targets = root / "t4t5_dirs.npz"
    # Two real directions are enough to exercise L/R routing and signed output.
    dirs = adapter._hex_to_dirs(np.array([[0, 0], [1, 0]], dtype=np.float32), "L")
    np.savez(
        targets,
        idx=np.array([0, 1], dtype=np.int64),
        dirs=np.stack((dirs[0], adapter._hex_to_dirs(np.array([[0, 0]], dtype=np.float32), "R")[0])),
        type=np.array(["T4a", "T5a"], dtype="U4"),
    )
    csr = root / "csr"
    csr.mkdir()
    np.savez(
        csr / "annotations.npz",
        typeIdx=np.array([0, 1], dtype=np.int32),
        sideIdx=np.array([1, 2], dtype=np.uint8),
    )
    (csr / "tables.json").write_text(json.dumps({"types": ["T4a", "T5a"], "sides": ["", "L", "R"]}))
    np.save(csr / "ids.npy", np.array([100, 101], dtype=np.int64))
    return env, model, targets, csr


class FlyVisRuntimeAdapterTest(unittest.TestCase):
    def test_shipped_721_hex_order_has_stable_left_right_direction_hashes(self) -> None:
        # The fitted flow/0000/000 connectome uses the compact hexagon
        # u=-15..15 with v clipped to the 721-column hex boundary.  This
        # dependency-free fixture mirrors that order and exercises the same
        # input_index -> (u,v) resolution used by the real adapter.
        uv = np.asarray(
            [
                (u, v)
                for u in range(-15, 16)
                for v in range(max(-15, -u - 15), min(15, -u + 15) + 1)
            ],
            dtype=np.int32,
        )
        self.assertEqual(uv.shape, (721, 2))
        connectome = types.SimpleNamespace(
            nodes=types.SimpleNamespace(u=uv[:, 0], v=uv[:, 1])
        )
        ordered = adapter._ordered_hex_columns(
            connectome, np.tile(np.arange(721, dtype=np.int64), (8, 1)), 721
        )
        np.testing.assert_array_equal(ordered, uv)
        self.assertEqual(
            hashlib.sha256(ordered.tobytes()).hexdigest(),
            "e244a411df2f8fa5bcbc8e1fb14fe68b8d1836116cd35ac98082d294f4a1e5ef",
        )
        left = adapter._hex_to_dirs(ordered, "L")
        right = adapter._hex_to_dirs(ordered, "R")
        self.assertEqual(
            hashlib.sha256(left.tobytes()).hexdigest(),
            "63b137c6e29a1ee035a2cef024bbfefe5ddae03e5e8347cbf397c3acc7cd6843",
        )
        self.assertEqual(
            hashlib.sha256(right.tobytes()).hexdigest(),
            "5037edbec7e4709ddc5f31643de643cdadef4492145756e72634d52727b7b4f6",
        )
        self.assertEqual(left.shape, (721, 3))
        self.assertEqual(right.shape, (721, 3))

    def test_input_index_rows_must_preserve_column_order(self) -> None:
        uv = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
        connectome = types.SimpleNamespace(
            nodes=types.SimpleNamespace(u=uv[:, 0], v=uv[:, 1])
        )
        with self.assertRaisesRegex(adapter.FlyVisMappingError, "rows disagree"):
            adapter._ordered_hex_columns(
                connectome,
                np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64),
                3,
            )

    def test_real_target_asset_uses_csr_side_metadata_in_asset_order(self) -> None:
        targets = adapter.load_t4t5_targets()
        self.assertEqual(targets["target_indices"].shape, (13582,))
        self.assertEqual(targets["directions"].shape, (13582, 3))
        self.assertEqual(targets["subtype"].shape, (13582,))
        self.assertEqual(targets["side"].shape, (13582,))
        self.assertEqual(
            {label: int(np.count_nonzero(targets["side"] == label)) for label in ("L", "R")},
            {"L": 6787, "R": 6795},
        )

    def test_import_is_lazy_and_preflight_is_read_only(self) -> None:
        report = adapter.preflight_flyvis(
            env_root=Path("/definitely/missing/flyvis"),
            model_root=Path("/definitely/missing/model"),
            target_path=Path("/definitely/missing/targets.npz"),
            csr_path=Path("/definitely/missing/csr"),
        )
        self.assertFalse(report["available"])
        self.assertTrue(report["missing"])
        self.assertNotIn("flyvis", sys.modules)

    def test_missing_real_assets_fail_closed_without_stub(self) -> None:
        with self.assertRaises(adapter.FlyVisUnavailableError):
            adapter.FlyVisRuntimeAdapter(
                env_root=Path("/definitely/missing/flyvis"),
                model_root=Path("/definitely/missing/model"),
                target_path=Path("/definitely/missing/targets.npz"),
                csr_path=Path("/definitely/missing/csr"),
                network_view_factory=lambda *_args, **_kwargs: None,
            )

    def test_direct_two_eye_batch_mapping_and_signed_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _FakeNet(["T4a", "T5a", "T4a", "T5a"], n_columns=2)
            view = _FakeView(net)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env,
                model_root=model,
                target_path=targets,
                csr_path=csr,
                n_columns=2,
                device="cpu",
                network_view=view,
                torch_module=_FakeTorch,
                baseline_mode="steady_state",
            )
            left = np.array([0.2, 0.3], dtype=np.float32)
            right = np.array([0.8, 0.7], dtype=np.float32)
            detailed = runtime.step_detailed(left, right)
            out = detailed.t4t5_drive
            self.assertEqual(out.shape, (2,))
            # The fake network sees exact direct columns, preserving signs
            # relative to the 0.5 grey baseline.
            np.testing.assert_allclose(net.stimulus.seen[-1], np.stack((left, right))[:, None, None, :])
            np.testing.assert_allclose(out, [-0.3, 0.3], atol=1e-6)
            np.testing.assert_array_equal(detailed.target_indices, [0, 1])
            np.testing.assert_allclose(detailed.target_drive, out)
            self.assertEqual(detailed.units, "current_mV_per_ms")
            self.assertEqual(runtime.units, "current_mV_per_ms")
            np.testing.assert_array_equal(runtime.target_indices, [0, 1])
            np.testing.assert_array_equal(runtime.target_ids, [100, 101])
            self.assertIs(detailed.target_indices, runtime._output_target_indices)
            self.assertIs(detailed.target_ids, runtime._output_target_ids)
            self.assertFalse(detailed.target_indices.flags.writeable)
            self.assertFalse(detailed.target_ids.flags.writeable)
            manifest = runtime.manifest()
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["n_targets"], 2)
            self.assertEqual(manifest["units"], "current_mV_per_ms")
            self.assertEqual(manifest["flyvis_hz"], 200.0)
            self.assertAlmostEqual(manifest["flyvis_dt_s"], 0.005)
            self.assertEqual(manifest["target_index_space"], "dense_neuron_index")
            self.assertEqual(manifest["mapping"]["method"], "precomputed_gather_scatter_mean")
            self.assertTrue(manifest["mapping"]["preserves_sign"])
            np.testing.assert_array_equal(runtime.column_uv, [[0, 0], [1, 0]])
            np.testing.assert_allclose(
                runtime.left_column_directions,
                adapter._hex_to_dirs(runtime.column_uv, "L"),
                atol=0.0,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                runtime.right_column_directions,
                adapter._hex_to_dirs(runtime.column_uv, "R"),
                atol=0.0,
                rtol=0.0,
            )
            self.assertEqual(
                manifest["column_order"]["source"], "net.stimulus.input_index"
            )
            self.assertEqual(manifest["column_order"]["input_index_shape"], [2, 2])
            self.assertEqual(manifest["column_order"]["uv_sha256"], hashlib.sha256(
                np.asarray([[0, 0], [1, 0]], dtype=np.int32).tobytes()
            ).hexdigest())
            for side in ("L", "R"):
                self.assertEqual(
                    manifest["column_directions"][side]["shape"], [2, 3]
                )
                self.assertEqual(
                    manifest["column_directions"][side]["sha256"],
                    hashlib.sha256(runtime.column_directions[side].tobytes()).hexdigest(),
                )
            json.dumps(manifest, allow_nan=False)
            self.assertEqual(len(net.states_seen), 1)
            self.assertIsNotNone(net.states_seen[0])
            first_state = runtime._state

            # State is one persistent two-eye batch; the next frame is not a
            # newly initialized independent network.
            second = runtime(np.full(2, 0.6, dtype=np.float32), np.full(2, 0.4, dtype=np.float32))
            self.assertEqual(len(net.states_seen), 2)
            self.assertIs(net.states_seen[1], first_state)
            self.assertIsNot(net.states_seen[1], runtime._state)
            self.assertTrue(np.any(second > 0))
            self.assertTrue(np.any(second < 0))

    def test_device_and_vram_limits_are_validated_before_runtime_import(self) -> None:
        common = {
            "env_root": Path("/definitely/missing/flyvis"),
            "model_root": Path("/definitely/missing/model"),
            "target_path": Path("/definitely/missing/targets.npz"),
            "csr_path": Path("/definitely/missing/csr"),
        }
        with self.assertRaisesRegex(ValueError, "device"):
            adapter.FlyVisRuntimeAdapter(device="auto", **common)
        with self.assertRaisesRegex(ValueError, "max_vram_gib"):
            adapter.FlyVisRuntimeAdapter(device="cpu", max_vram_gib=0, **common)

    def test_network_view_factory_receives_results_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _FakeNet(["T4a", "T5a", "T4a", "T5a"], n_columns=2)
            view = _FakeView(net)
            calls: list[tuple[str, Path]] = []

            def factory(view_name: str, *, root_dir: Path):
                calls.append((view_name, Path(root_dir)))
                return view

            adapter.FlyVisRuntimeAdapter(
                env_root=env,
                model_root=model,
                target_path=targets,
                csr_path=csr,
                n_columns=2,
                device="cpu",
                network_view_factory=factory,
                torch_module=_FakeTorch,
            )
            self.assertEqual(calls, [(adapter.DEFAULT_VIEW, model / "results")])

    def test_shape_is_fail_closed_and_reset_clears_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _FakeNet(["T4a", "T5a", "T4a", "T5a"], n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, network_view=_FakeView(net), torch_module=_FakeTorch,
            )
            with self.assertRaises(ValueError):
                runtime(np.zeros(3), np.zeros(2))
            self.assertEqual(len(net.states_seen), 0)
            runtime(np.zeros(2), np.zeros(2))
            self.assertIsNotNone(runtime._state)
            runtime.reset()
            self.assertIsNone(runtime._state)
            self.assertIsNone(runtime.last_output)

    def test_static_scene_preadaptation_retains_state_and_emits_no_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _FakeNet(["T4a", "T5a", "T4a", "T5a"], n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_FakeTorch, baseline_mode="steady_state",
            )
            left = np.array([0.2, 0.3], dtype=np.float32)
            right = np.array([0.8, 0.7], dtype=np.float32)
            report = runtime.preadapt_static_scene(left, right, seconds=0.01)

            self.assertEqual(report["frames"], 2)
            self.assertEqual(report["purpose"], "sensory_initialization_only")
            self.assertFalse(report["male_cns_invoked"])
            self.assertFalse(report["body_invoked"])
            self.assertEqual(runtime._frame, 0)
            self.assertIsNone(runtime.last_output)
            adapted_state = runtime._state

            same = runtime.step_detailed(left, right)
            np.testing.assert_allclose(same.t4t5_drive, [0.0, 0.0], atol=1e-7)
            self.assertIs(net.states_seen[-1], adapted_state)
            self.assertEqual(same.frame_index, 0)

            changed = runtime.step_detailed(
                np.array([0.4, 0.3], dtype=np.float32),
                np.array([0.6, 0.7], dtype=np.float32),
            )
            self.assertTrue(np.any(np.abs(changed.t4t5_drive) > 0.0))
            manifest = runtime.manifest()["scene_baseline"]
            self.assertTrue(manifest["state_retained"])
            self.assertEqual(manifest["exposed_frame_index_after_adaptation"], 0)

            with self.assertRaisesRegex(ValueError, "multiple"):
                runtime.preadapt_static_scene(left, right, seconds=0.011)

    def test_vectorized_mapping_matches_grouped_reference_and_preserves_signs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _GroupedFakeNet(n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_FakeTorch, baseline_mode="none", output_gain=1.75,
            )
            activity = np.array(
                [[0.1, 0.5, -0.2, 0.2, 0.0, 0.0],
                 [0.0, 0.0, -0.8, -0.2, 0.0, 0.0]],
                dtype=np.float32,
            )
            actual = runtime._map_activity(activity)
            expected = np.array([
                np.mean(activity[0, [0, 1]]),
                np.mean(activity[1, [2, 3]]),
            ], dtype=np.float32) * np.float32(1.75)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
            self.assertGreater(actual[0], 0.0)
            self.assertLess(actual[1], 0.0)
            np.testing.assert_array_equal(runtime._mapping_group_sizes, [2, 2])

    def test_native_gather_matches_reference_without_full_activity_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _GroupedFakeNet(n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_NativeTorch, baseline_mode="none", output_gain=1.0,
            )
            activity = _NativeTensor(
                [[0.1, 0.5, -0.2, 0.2, 0.0, 0.0],
                 [0.0, 0.0, -0.8, -0.2, 0.0, 0.0]],
            )
            actual = runtime._map_activity(activity)
            expected = np.array([0.3, -0.5], dtype=np.float32)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)

    def test_native_mapping_subtracts_baseline_after_compact_gather(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _GroupedFakeNet(n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_CountingNativeTorch, baseline_mode="fixed", output_gain=1.0,
                validate_activity_finite=False,
            )
            runtime._ensure_state()
            _CountingNativeTorch.finite_calls = 0
            activity = _NativeTensor(
                [[0.1, 0.5, -0.2, 0.2, 0.0, 0.0],
                 [0.0, 0.0, -0.8, -0.2, 0.0, 0.0]],
            )
            actual = runtime._map_activity(activity)
            expected = np.array([-0.2, -1.0], dtype=np.float32)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
            # The default production path does not force a full-device
            # finite reduction; mapped output finiteness remains checked on
            # the small host target vector.
            self.assertEqual(_CountingNativeTorch.finite_calls, 0)

    def test_internal_step_validates_native_activity_once_and_preserves_output(self) -> None:
        """The network step validates activity; mapping must not scan it again."""
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _GroupedFakeNet(n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_CountingNativeTorch, baseline_mode="none",
                validate_activity_finite=True,
            )
            activity = _NativeTensor(
                [[0.1, 0.5, -0.2, 0.2, 0.0, 0.0],
                 [0.0, 0.0, -0.8, -0.2, 0.0, 0.0]],
            )
            # Keep this focused on the frame path: _network_sequence is
            # represented by a validated native activity result, while the
            # real mapping/output code remains fully exercised.
            runtime._ensure_state = lambda: None  # type: ignore[method-assign]
            runtime._network_step = (  # type: ignore[method-assign]
                lambda _left, _right: runtime._validate_activity(activity)
            )
            _CountingNativeTorch.finite_calls = 0
            output = runtime.step_detailed(
                np.array([0.2, 0.3], dtype=np.float32),
                np.array([0.8, 0.7], dtype=np.float32),
            )
            self.assertEqual(_CountingNativeTorch.finite_calls, 1)
            np.testing.assert_allclose(output.drive, [0.3, -0.5], rtol=0.0, atol=1e-7)

    def test_internal_step_nan_activity_still_fails_closed(self) -> None:
        """Skipping duplicate validation must not weaken the network boundary."""
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _GroupedFakeNet(n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_CountingNativeTorch, baseline_mode="none",
                validate_activity_finite=True,
            )
            activity = _NativeTensor(
                [[np.nan, 0.5, -0.2, 0.2, 0.0, 0.0],
                 [0.0, 0.0, -0.8, -0.2, 0.0, 0.0]],
            )
            runtime._ensure_state = lambda: None  # type: ignore[method-assign]
            runtime._network_step = (  # type: ignore[method-assign]
                lambda _left, _right: runtime._validate_activity(activity)
            )
            _CountingNativeTorch.finite_calls = 0
            with self.assertRaisesRegex(
                adapter.FlyVisRuntimeError, "NaN or infinite"
            ):
                runtime.step_detailed(
                    np.array([0.2, 0.3], dtype=np.float32),
                    np.array([0.8, 0.7], dtype=np.float32),
                )
            self.assertEqual(_CountingNativeTorch.finite_calls, 1)

    def test_native_baseline_on_activity_device_is_not_moved_each_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _GroupedFakeNet(n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_NativeTorch, baseline_mode="none",
            )
            activity = _NativeTensor(np.zeros((2, 6), dtype=np.float32))
            baseline = _TrackingNativeTensor(np.full((2, 6), 0.5, dtype=np.float32))
            runtime._baseline_activity = baseline
            self.assertIs(runtime._baseline_for_activity(activity), baseline)
            self.assertIs(runtime._baseline_for_activity(activity), baseline)
            self.assertEqual(baseline.to_calls, 0)

    def test_persistent_input_and_timing_manifest_are_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _FakeNet(["T4a", "T5a", "T4a", "T5a"], n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_FakeTorch, timing_enabled=True,
            )
            input_tensor = runtime._input_device_tensor
            runtime(np.array([0.1, 0.2]), np.array([0.8, 0.7]))
            runtime(np.array([0.3, 0.4]), np.array([0.6, 0.5]))
            self.assertIs(runtime._input_device_tensor, input_tensor)
            manifest = runtime.manifest()
            self.assertEqual(manifest["timing"]["frames"], 2)
            self.assertGreaterEqual(manifest["timing"]["frame_s"], 0.0)
            self.assertGreaterEqual(manifest["timing"]["mapping_s"], 0.0)
            self.assertTrue(manifest["optimization"]["persistent_input_tensor"])
            self.assertTrue(manifest["optimization"]["inference_mode"])
            self.assertEqual(
                manifest["optimization"]["device_input_stream"],
                "current_torch_stream_no_device_sync",
            )
            json.dumps(manifest, allow_nan=False)

    def test_device_resident_column_vectors_keep_the_public_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, model, targets, csr = _make_assets(Path(tmp))
            net = _FakeNet(["T4a", "T5a", "T4a", "T5a"], n_columns=2)
            runtime = adapter.FlyVisRuntimeAdapter(
                env_root=env, model_root=model, target_path=targets, csr_path=csr,
                n_columns=2, device="cpu", network_view=_FakeView(net),
                torch_module=_FakeTorch,
            )
            left = _DeviceVector([0.2, 0.3])
            right = _DeviceVector([0.8, 0.7])
            output = runtime.step_detailed(left, right)
            np.testing.assert_allclose(output.drive, [-0.3, 0.3], atol=1e-6)
            np.testing.assert_allclose(
                net.stimulus.seen[-1],
                np.stack((left.value, right.value))[:, None, None, :],
            )


if __name__ == "__main__":
    unittest.main()
