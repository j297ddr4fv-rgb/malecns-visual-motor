from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_flygym_reference_walk import FlyGymReferenceSession, WalkConfig


class FakeMujoco:
    __version__ = "mock"

    def mj_contactForce(self, _model, _data, _contact_id, wrench):
        wrench[:] = (0, 0, 5, 0, 0, 0)

    def mj_geomDistance(self, *_args):
        return 2.0


class LightCaptureTests(unittest.TestCase):
    def make_session(self):
        session = object.__new__(FlyGymReferenceSession)
        session.capture_mode = "light"
        session.capture_stride = 50
        session.config = WalkConfig(duration_s=0.0002)
        session.duration_s = 0.0002
        session._root_qpos_write_count = session._root_qvel_write_count = 0
        session._native_samples = session._control_samples = 0
        session._physics_step_s = session._capture_s = 0.0
        session.run_id = "mock-light"
        session.mj = FakeMujoco()
        session.model = types.SimpleNamespace(
            geom_bodyid=np.zeros(11, dtype=np.int32),
            opt=types.SimpleNamespace(timestep=1e-4),
        )
        session.model.geom_bodyid[10] = 2
        session.data = types.SimpleNamespace(
            time=0.0, ncon=0, contact=[],
            xpos=np.zeros((8, 3)), xipos=np.zeros((8, 3)),
            xquat=np.tile([1.0, 0.0, 0.0, 0.0], (8, 1)),
            xmat=np.tile(np.eye(3).reshape(1, 9), (8, 1)),
            qpos=np.array([0.25]), qvel=np.array([0.5]),
            xfrc_applied=np.zeros((8, 6)), qfrc_applied=np.zeros(1),
            ctrl=np.zeros(6),
        )
        session.data.xpos[1] = (3.0, 4.0, 5.0)
        session.data.xpos[2:8, 2] = 1.0
        session.data.xipos[1] = session.data.xpos[1]
        session.joint_angles_ids = np.array([0])
        session.position_actuator_ids = np.array([0])
        session.adhesion_actuator_ids = np.arange(6)
        session.foot_body_ids = np.arange(2, 8)
        session.thorax_body_id = 1
        session.body_ids = np.array([1])
        session.body_mass = np.array([1.0])
        session.ground_geom_id = 0
        session.thorax_collision_geom_ids = np.array([1])
        session.geom_id_to_segment = {10: "lf_tarsus5"}
        session.geom_id_to_leg = {10: 0}
        session.controller = types.SimpleNamespace(cpg_network=types.SimpleNamespace(
            curr_phases=np.arange(6.0), curr_magnitudes=np.ones(6)))
        session.last_action = types.SimpleNamespace(adhesion_onoff=np.ones(6, dtype=np.uint8))
        session.body_names = ["c_thorax"]
        session.effective_model_options = {"nbody": 2}
        session.upstream_source_hashes = {"installed_flygym_version": "mock"}
        session._initialize_light_capture(2)
        return session

    def test_light_records_exact_native_contact_episode_without_full_arrays_or_body_velocity(self):
        session = self.make_session()
        self.assertFalse(hasattr(session, "all_arrays"))
        self.assertEqual(session._light_native_step.shape, (2,))
        current_contact = session._light_current_contact_state
        previous_contact = session._light_previous_contact_state
        force_scratch = session._light_contact_force_scratch
        fz_scratch = session._light_per_leg_fz_scratch
        self.assertIsNot(current_contact, previous_contact)
        for step, touching in enumerate((True, True, False)):
            contact = types.SimpleNamespace(
                geom1=0, geom2=10, frame=np.eye(3).reshape(9),
            )
            session.data.time = step * 1e-4
            session.data.contact = [contact] if touching else []
            session.data.ncon = int(touching)
            session._record_light_native_sample(np.array([0.5, 0.5]))
            self.assertIs(session._light_current_contact_state, current_contact)
            self.assertIs(session._light_previous_contact_state, previous_contact)
            self.assertIs(session._light_contact_force_scratch, force_scratch)
            self.assertIs(session._light_per_leg_fz_scratch, fz_scratch)
        self.assertEqual(session._light_contact_events, [(0, 0, 1)])
        self.assertAlmostEqual(session._light_last_foot_fz[0], 0.0)
        self.assertEqual(session.light_trace_arrays()["time_s"].tolist(), [0.0, 0.0002])
        result = session._light_step_result(np.array([0.5, 0.5]), 0.0, 1)
        self.assertEqual(result["thorax_x_mm"], 3.0)
        self.assertAlmostEqual(result["thorax_heading_yaw_rad"], 0.0)
        self.assertNotIn("all_arrays", vars(session))

    def test_light_save_returns_compatible_summary_and_compact_schema(self):
        session = self.make_session()
        session._record_light_native_sample(np.zeros(2))
        session._native_samples = 1
        session._control_samples = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            result = session.save_light(Path(temp_dir))
            self.assertEqual(result["summary"]["capture_mode"], "light")
            self.assertIn("thorax_x_advance_mm_analysis", result["summary"])
            self.assertTrue((Path(temp_dir) / "contact_events.npz").is_file())
            summary = json.loads((Path(temp_dir) / "summary.json").read_text())
            self.assertEqual(summary["schema"], "flygym-reference-light-summary/v1")
            with np.load(Path(temp_dir) / "trajectory.npz") as trace:
                self.assertEqual(str(trace["schema"]), "flygym-reference-light-trajectory/v1")


if __name__ == "__main__":
    unittest.main()
