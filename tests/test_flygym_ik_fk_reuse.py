from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
import ast
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flygym_pose_priority_ik_v2 import (
    _body_state,
    _qpos_quaternions_are_normalization_stable,
    refine_orientation_at_position,
)


class FakeMujoco:
    mjtJoint = types.SimpleNamespace(mjJNT_FREE=0, mjJNT_BALL=1)

    def __init__(self):
        self.forward_calls = 0

    def mj_fwdPosition(self, _model, data):
        self.forward_calls += 1
        data.xpos[0] = (data.qpos.sum(), 0, 0)
        data.xmat[0] = np.eye(3).reshape(9)

    @staticmethod
    def mju_normalize4(quaternion):
        norm = float(np.linalg.norm(quaternion))
        if norm > 0.0:
            quaternion[:] /= norm
        return norm


class FKReuseTests(unittest.TestCase):
    def data(self):
        return types.SimpleNamespace(
            qpos=np.zeros(7), xpos=np.zeros((1, 3)),
            xmat=np.eye(3).reshape(1, 9).copy(),
        )

    @staticmethod
    def no_quaternion_model():
        return types.SimpleNamespace(
            jnt_type=np.zeros(0, dtype=np.int32),
            jnt_qposadr=np.zeros(0, dtype=np.int32),
        )

    def test_body_state_default_refreshes_and_prepared_state_returns_copies(self):
        mj, data = FakeMujoco(), self.data()
        xyz, rotation = _body_state(mj, None, data, 0)
        self.assertEqual(mj.forward_calls, 1)
        cached_xyz, cached_rotation = _body_state(mj, None, data, 0, fk_prepared=True)
        self.assertEqual(mj.forward_calls, 1)
        np.testing.assert_array_equal(xyz, cached_xyz)
        np.testing.assert_array_equal(rotation, cached_rotation)
        cached_xyz[:] = 10
        cached_rotation[:] = 10
        self.assertEqual(data.xpos[0, 0], 0)
        self.assertEqual(data.xmat[0, 0], 1)

    def test_prepared_baseline_skips_one_fk_and_keeps_fallback_state_and_metrics(self):
        results = []
        for prepared in (False, True):
            mj, data = FakeMujoco(), self.data()
            # Both variants enter with the same fully prepared baseline pose.
            mj.mj_fwdPosition(None, data)
            solved, metrics = refine_orientation_at_position(
                mj, None, data, 0, np.arange(7), np.arange(7),
                np.zeros(7), np.zeros(7), np.zeros(3), np.eye(3),
                baseline_fk_prepared=prepared,
            )
            metrics = {k: v for k, v in metrics.items() if k != "cpu_s"}
            results.append((solved.copy(), metrics, data.qpos.copy(),
                            data.xpos.copy(), data.xmat.copy(), mj.forward_calls))
        for index in (0, 2, 3, 4):
            np.testing.assert_array_equal(results[0][index], results[1][index])
        self.assertEqual(results[0][1], results[1][1])
        self.assertTrue(results[0][1]["fallback_to_exact_v3_solution"])
        self.assertEqual(results[0][5] - results[1][5], 1)

    def test_prepared_fk_is_reused_through_early_exit_and_final_metrics(self):
        results = []
        for prepared, collect_diagnostics in ((False, True), (True, True), (False, False)):
            mj, data = FakeMujoco(), self.data()
            model = self.no_quaternion_model()
            if prepared:
                mj.mj_fwdPosition(model, data)
            solved, metrics = refine_orientation_at_position(
                mj, model, data, 0, np.arange(7), np.arange(7),
                np.zeros(7), np.zeros(7), np.zeros(3), np.eye(3),
                baseline_fk_prepared=prepared,
                collect_diagnostics=collect_diagnostics,
                reuse_prepared_fk=True,
            )
            results.append((solved.copy(), data.qpos.copy(), data.xpos.copy(),
                            data.xmat.copy(), mj.forward_calls, metrics))
        for compact in (1, 2):
            for index in range(4):
                np.testing.assert_array_equal(results[0][index], results[compact][index])
        self.assertEqual(results[0][4], 1)
        self.assertEqual(results[1][4], 1)
        self.assertEqual(results[2][4], 1)
        self.assertTrue(results[0][5])
        self.assertEqual(results[2][5], {})

    def test_prepared_fk_refreshes_when_baseline_changes_full_qpos(self):
        mj = FakeMujoco()
        data = self.data()
        data.qpos = np.zeros(8)
        model = self.no_quaternion_model()
        mj.mj_fwdPosition(model, data)
        baseline = np.zeros(7)
        baseline[0] = 0.3
        solved, metrics = refine_orientation_at_position(
            mj, model, data, 0, np.arange(7), np.arange(7),
            baseline.copy(), baseline, np.zeros(3), np.eye(3),
            baseline_fk_prepared=True,
            reuse_prepared_fk=True,
        )
        np.testing.assert_array_equal(solved, baseline)
        self.assertEqual(mj.forward_calls, 2)
        self.assertAlmostEqual(float(data.xpos[0, 0]), 0.3)
        self.assertTrue(metrics["fallback_to_exact_v3_solution"])

    def test_v41_compact_dispatch_does_not_call_ap_override(self):
        # Load this method alone: the system-Python suite has no FlyGym demo.
        source = Path(__file__).resolve().parents[1] / "scripts/flygym_phase_speed_gait_v4_1.py"
        tree = ast.parse(source.read_text())
        controller_class = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in controller_class.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_hind_lift_compact")
        controller = types.SimpleNamespace(
            _apply_hind_lift=lambda *args, **kwargs: self.fail("AP override must not be called")
        )
        targets = np.zeros(42)
        active = np.ones(2, dtype=bool)
        holder = types.SimpleNamespace(_apply_hind_lift=lambda *args, **kwargs: (targets, {}))
        namespace = {"np": np, "PhaseSpeedGaitController": holder}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        with patch.object(holder, "_apply_hind_lift", return_value=(targets, {})) as solve:
            result = namespace["_apply_hind_lift_compact"](controller, targets, active)
        self.assertIs(result, targets)
        solve.assert_called_once_with(controller, targets, active, collect_diagnostics=False)

    def test_fk_reuse_is_opt_in_and_requires_stable_joint_quaternions(self):
        mj = FakeMujoco()
        model = self.no_quaternion_model()
        data = self.data()
        _, default_metrics = refine_orientation_at_position(
            mj, model, data, 0, np.arange(7), np.arange(7),
            np.zeros(7), np.zeros(7), np.zeros(3), np.eye(3),
        )
        # Baseline, first iteration, final evaluation, and fallback evaluation.
        self.assertEqual(mj.forward_calls, 4)
        self.assertTrue(default_metrics)

        free_joint_model = types.SimpleNamespace(
            jnt_type=np.asarray([mj.mjtJoint.mjJNT_FREE], dtype=np.int32),
            jnt_qposadr=np.asarray([0], dtype=np.int32),
        )
        self.assertFalse(
            _qpos_quaternions_are_normalization_stable(
                mj, free_joint_model, np.asarray([0, 0, 0, 2, 0, 0, 0], dtype=float)
            )
        )
        self.assertTrue(
            _qpos_quaternions_are_normalization_stable(
                mj, free_joint_model, np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=float)
            )
        )


if __name__ == "__main__":
    unittest.main()
