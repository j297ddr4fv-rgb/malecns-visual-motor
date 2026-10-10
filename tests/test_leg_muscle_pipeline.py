"""Causal end-to-end checks using two independently identified motor pools."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from leg_muscle_decoder import LegMuscleDecoder


class TwoMotorIndex:
    n = 2
    ids = np.array([1001, 1002], dtype=np.int64)
    annotations = {"typeIdx": np.array([0, 1])}
    tables = {"types": np.array(["Ti flexor MN", "Ti extensor MN"])}

    def resolve(self, selector):
        if "subclass:fl/L" not in selector:
            return np.array([], dtype=np.int64)
        if "type:Ti flexor MN" in selector:
            return np.array([0], dtype=np.int64)
        if "type:Ti extensor MN" in selector:
            return np.array([1], dtype=np.int64)
        if "type:" in selector:
            return np.array([], dtype=np.int64)
        return np.array([0, 1], dtype=np.int64)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "optional CPU MuJoCo is not installed")
class PipelineTests(unittest.TestCase):
    def make_pipeline(self):
        from articulated_leg_body import ArticulatedLegBody
        from leg_muscle_pipeline import LegMusclePipeline
        return LegMusclePipeline(
            decoder=LegMuscleDecoder(population_index=TwoMotorIndex()),
            body=ArticulatedLegBody(fixed_root=True),
        )

    def test_muscle_specific_neural_input_reaches_opposed_joint_actions(self):
        outcomes = []
        peak_forces = []
        for rates in ([0., 0.], [30., 0.], [0., 30.]):
            pipe = self.make_pipeline()
            peak = {"flexor": 0., "extensor": 0.}
            for _ in range(20):
                result = pipe.step(dt_s=.0005, rates_hz=np.asarray(rates))
                for group in peak:
                    peak[group] = max(peak[group], result["muscles"][f"LF_tibia_{group}"]["active_force_N"])
            outcomes.append(result)
            peak_forces.append(peak)
        sham, flexor, extensor = outcomes
        self.assertEqual(sham["muscles"]["LF_tibia_flexor"]["active_force_N"], 0.)
        self.assertGreater(peak_forces[1]["flexor"], 0.)
        self.assertEqual(flexor["muscles"]["LF_tibia_extensor"]["active_force_N"], 0.)
        self.assertGreater(peak_forces[2]["extensor"], 0.)
        angles = [r["body"]["joints"]["LF_tibia_pitch"]["q_rad"] for r in outcomes]
        self.assertLess((angles[1]-angles[0]) * (angles[2]-angles[0]), 0.)
        for r in outcomes:
            self.assertEqual(len(r["body"]["feet"]), 6)
            self.assertEqual(len(r["body"]["joints"]), 42)

    def test_neural_spike_filter_is_explicit_and_finite(self):
        pipe = self.make_pipeline()
        result = pipe.step(dt_s=.0005, spikes=np.array([0, 1]))
        self.assertGreater(result["muscles"]["LF_tibia_extensor"]["excitation"], 0.)
        self.assertEqual(result["muscles"]["LF_tibia_flexor"]["excitation"], 0.)
        self.assertFalse(pipe.provenance()["synthetic_gait"])
        with self.assertRaises(ValueError):
            pipe.step(dt_s=.0005, spikes=np.array([0., np.nan]))

    def test_muscle_length_derivative_matches_virtual_work_sign(self):
        pipe = self.make_pipeline()
        body = pipe.body
        joint_id = body.model.joint("LF_tibia_pitch").id
        qadr = int(body.model.jnt_qposadr[joint_id])
        dof = int(body.model.jnt_dofadr[joint_id])
        import mujoco
        body.data.qvel[dof] = .2
        mujoco.mj_forward(body.model, body.data)
        kin = body.muscle_kinematics()["LF_tibia_extensor"]
        l0 = kin["length_m"]
        body.data.qpos[qadr] += 1e-7
        mujoco.mj_forward(body.model, body.data)
        l1 = body.muscle_kinematics()["LF_tibia_extensor"]["length_m"]
        derivative = (l1-l0) / 1e-7
        self.assertAlmostEqual(kin["velocity_mps"], derivative*.2, delta=1e-12)
        expected = -body.muscles["LF_tibia_extensor"].joint_moment_arms_m["LF_tibia_pitch"]
        self.assertAlmostEqual(derivative, expected, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
