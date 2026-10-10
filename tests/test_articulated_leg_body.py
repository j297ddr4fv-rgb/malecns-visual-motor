import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.articulated_leg_body import ArticulatedLegBody


HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo is optional; run in the project CPU environment")
class ArticulatedLegBodyTests(unittest.TestCase):
    def test_homologous_joint_coordinates_mirror_feet_at_rest_and_bent_pose(self):
        body = ArticulatedLegBody(fixed_root=True)
        reflection = np.array([1., -1., 1.])
        joint_pose = {"coxa_yaw": .12, "coxa_roll": -.10, "coxa_pitch": .08,
                      "femur_pitch": .05, "femur_roll": .07,
                      "tibia_pitch": -.20, "tarsus_pitch": .11}
        for bent in (False, True):
            body.reset()
            if bent:
                for leg in body._legs:
                    for suffix, angle in joint_pose.items():
                        body.data.qpos[body._joint_qposadr[f"{leg}_{suffix}"]] = angle
                body._mujoco.mj_forward(body.model, body.data)
            baseline = body.data.qpos.copy()
            for left, right in (("LF", "RF"), ("LM", "RM"), ("LH", "RH")):
                with self.subTest(pair=(left, right), bent=bent):
                    np.testing.assert_allclose(
                        body.data.geom_xpos[body._foot_geom_ids[right]],
                        reflection * body.data.geom_xpos[body._foot_geom_ids[left]],
                        atol=1e-12, rtol=0)
                for suffix in joint_pose:
                    derivatives = []
                    for leg in (left, right):
                        positions = []
                        for delta in (-1e-6, 1e-6):
                            body.data.qpos[:] = baseline
                            body.data.qpos[body._joint_qposadr[f"{leg}_{suffix}"]] += delta
                            body._mujoco.mj_forward(body.model, body.data)
                            positions.append(body.data.geom_xpos[body._foot_geom_ids[leg]].copy())
                        derivatives.append((positions[1] - positions[0]) / 2e-6)
                    with self.subTest(pair=(left, right), joint=suffix, bent=bent):
                        np.testing.assert_allclose(derivatives[1], reflection * derivatives[0],
                                                   atol=1e-10, rtol=1e-7)
                    body.data.qpos[:] = baseline
                    body._mujoco.mj_forward(body.model, body.data)

    def test_homologous_muscle_pulses_produce_mirrored_foot_motion(self):
        reflection = np.array([1., -1., 1.])
        for pair in (("LF", "RF"), ("LM", "RM"), ("LH", "RH")):
            for group in ("sternal_anterior_rotator", "sternal_adductor",
                          "femur_reductor", "pleural_remotor_abductor", "tibia_flexor"):
                with self.subTest(pair=pair, muscle=group):
                    body = ArticulatedLegBody(fixed_root=True)
                    row = body.step({f"{leg}_{group}": 1e-5 for leg in pair}, .003)
                    left, right = pair
                    np.testing.assert_allclose(row["feet"][right]["position_m"],
                                               reflection * row["feet"][left]["position_m"],
                                               atol=1e-8, rtol=0)
                    self.assertGreater(max(abs(v["muscle_torque_Nm"])
                                           for v in row["joints"].values()), 0)

    def test_passive_support_is_native_hinge_only_and_defaults_from_config(self):
        body = ArticulatedLegBody()
        self.assertTrue(body.passive_support)
        self.assertEqual(body.model.njnt - 1, 42)
        self.assertTrue(all(abs(float(v) - 3e-8) < 1e-20 for v in body.model.jnt_stiffness[1:]))
        ground_id = body._mujoco.mj_name2id(body.model, body._mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.assertAlmostEqual(float(body.model.geom_solref[ground_id][0]), 0.0005)
        self.assertAlmostEqual(float(body.model.geom_solref[ground_id][1]), 1.0)
        root_dadr = int(body.model.jnt_dofadr[body._root_joint_id])
        self.assertTrue(all(float(v) == 0.0 for v in body.model.dof_damping[root_dadr : root_dadr + 6]))
        hinge_damping = [
            float(body.model.dof_damping[dadr]) for dadr in body._joint_dofadr.values()
        ]
        self.assertTrue(all(abs(value - 2e-11) < 1e-25 for value in hinge_damping))

        disabled = ArticulatedLegBody(passive_support=False)
        self.assertFalse(disabled.passive_support)
        self.assertTrue(all(float(v) == 0.0 for v in disabled.model.jnt_stiffness[1:]))
        self.assertTrue(all(abs(float(disabled.model.dof_damping[dadr]) - 2e-11) < 1e-25 for dadr in disabled._joint_dofadr.values()))
        self.assertEqual(disabled.provenance()["passive_support"]["evidence_label"], "engineering_assumption_not_measured_biology")

    def test_model_channels_kinematics_and_fractional_substep(self):
        body = ArticulatedLegBody(fixed_root=False)
        self.assertEqual((body.model.nq, body.model.nv), (49, 48))
        self.assertEqual(len(body.muscles), 114)
        self.assertIn("LF_sternal_adductor", body.muscles)
        self.assertEqual(body.muscles["LF_long_tendon_tibia"].anatomical_alias, "LTM1")
        self.assertEqual(body.muscles["LF_long_tendon_femur"].anatomical_alias, "LTM2")

        name = "LF_tibia_flexor"
        descriptor = body.muscles[name]
        joint = "LF_tibia_pitch"
        qadr = body._joint_qposadr[joint]
        dadr = body._joint_dofadr[joint]
        arm = descriptor.joint_moment_arms_m[joint]
        body.data.qpos[qadr] += 0.1
        body.data.qvel[dadr] = 0.5
        body._mujoco.mj_forward(body.model, body.data)
        kinematics = body.muscle_kinematics()[name]
        self.assertAlmostEqual(kinematics["length_m"], descriptor.optimal_length_m - arm * 0.1)
        self.assertAlmostEqual(kinematics["velocity_mps"], -arm * 0.5)

        telemetry = body.step({}, 0.00025)
        self.assertAlmostEqual(telemetry["time_s"], 0.00025, places=15)

    def test_positive_moment_arm_increases_joint_response_without_root_wrench(self):
        unloaded = ArticulatedLegBody(fixed_root=True, friction=0.0)
        loaded = ArticulatedLegBody(fixed_root=True, friction=0.0)
        channel = "LF_tibia_flexor"
        joint = "LF_tibia_pitch"
        no_drive = unloaded.step({}, 0.0001)
        drive = loaded.step({channel: 1e-5}, 0.0001)
        self.assertGreater(drive["joints"][joint]["qd_rad_s"], no_drive["joints"][joint]["qd_rad_s"])
        expected_torque = loaded.muscles[channel].joint_moment_arms_m[joint] * 1e-5
        self.assertAlmostEqual(drive["joints"][joint]["torque_Nm"], expected_torque)
        free_body = ArticulatedLegBody(fixed_root=False, friction=0.0)
        free_body.step({channel: 1e-5}, 0.0001)
        # The free-root generalized force entries remain exactly zero.
        self.assertTrue(all(abs(float(v)) == 0.0 for v in free_body.data.qfrc_applied[:6]))

    def test_six_foot_ground_support_and_snapshot_restore(self):
        body = ArticulatedLegBody()
        telemetry = body.step({}, 0.0005)
        self.assertTrue(all(foot["contact"] for foot in telemetry["feet"].values()))
        self.assertTrue(all(foot["normal_force_N"] > 0.0 for foot in telemetry["feet"].values()))
        self.assertGreater(telemetry["total_foot_normal_force_N"], 0.0)
        self.assertFalse(telemetry["body_ground_contact"])
        self.assertEqual(len(telemetry["linear_momentum_kg_mps"]), 3)
        self.assertTrue(all(abs(v) < 1e-6 for v in telemetry["linear_momentum_kg_mps"][:2]))

        snapshot = body.snapshot()
        restored = ArticulatedLegBody()
        restored.reset(snapshot)
        self.assertEqual(restored.data.time, body.data.time)
        self.assertEqual(restored.data.qpos.tolist(), body.data.qpos.tolist())
        self.assertEqual(restored.data.qvel.tolist(), body.data.qvel.tolist())
        self.assertEqual(restored.provenance()["mjcf_sha256"], snapshot["mjcf_sha256"])

    def test_passive_support_settles_on_feet_without_root_wrench(self):
        body = ArticulatedLegBody()
        recent = []
        # Keep outer calls short so MuJoCo's strict simulation clock advances
        # reliably while the native solver uses its configured 100 us step.
        for index in range(1000):
            telemetry = body.step({}, 0.001)
            if index >= 800:
                recent.append(telemetry)

        expected_weight = float(body.model.body_mass.sum()) * body._gravity_mps2
        self.assertTrue(all(not item["body_ground_contact"] for item in recent))
        self.assertTrue(all(all(foot["contact"] for foot in item["feet"].values()) for item in recent))
        final = recent[-1]
        self.assertAlmostEqual(final["total_foot_normal_force_N"], expected_weight, delta=expected_weight * 0.08)
        self.assertLess(max(item["root"]["position_m"][2] for item in recent) - min(item["root"]["position_m"][2] for item in recent), 8e-5)
        foot_radius = float(body._geometry["foot_radius_m"])
        self.assertGreaterEqual(min(foot["position_m"][2] - foot_radius for foot in final["feet"].values()), -6e-6)
        self.assertEqual(final["root_applied_wrench"]["max_abs_generalized_force"], 0.0)
        self.assertGreaterEqual(final["passive_support"]["elastic_energy_J"], 0.0)
        self.assertGreaterEqual(final["passive_support"]["damping_power_W"], 0.0)
        self.assertTrue(all(joint["torque_Nm"] == joint["muscle_torque_Nm"] for joint in final["joints"].values()))

    def test_springs_off_ablation_collapses_and_small_velocity_perturbation_recovers(self):
        unsupported = ArticulatedLegBody(passive_support=False)
        for _ in range(1000):
            telemetry = unsupported.step({}, 0.001)
        self.assertLess(telemetry["root"]["position_m"][2], 0.00065)
        self.assertFalse(telemetry["passive_support"]["enabled"])
        self.assertEqual(telemetry["passive_support"]["elastic_energy_J"], 0.0)
        self.assertEqual(telemetry["root_applied_wrench"]["max_abs_generalized_force"], 0.0)

        perturbed = ArticulatedLegBody(fixed_root=True)
        joint_name = "LF_tibia_pitch"
        dadr = perturbed._joint_dofadr[joint_name]
        perturbed.data.qvel[dadr] = 0.05
        perturbed._mujoco.mj_forward(perturbed.model, perturbed.data)
        initial_speed = abs(float(perturbed.data.qvel[dadr]))
        for _ in range(20):
            recovered = perturbed.step({}, 0.001)
        self.assertLess(abs(recovered["joints"][joint_name]["qd_rad_s"]), initial_speed * 0.25)

    def test_invalid_contact_solver_configuration_fails_closed(self):
        template = ArticulatedLegBody(fixed_root=True)
        for invalid_tau in (float("nan"), 1.99e-4):
            config = copy.deepcopy(template._config)
            config["physics"]["contact_solver"]["time_constant_s"] = invalid_tau
            with self.subTest(time_constant_s=invalid_tau), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "invalid.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ArticulatedLegBody(config_path=path)

    def test_trochanter_extensors_lower_proximal_joint_without_invented_roll_action(self):
        for leg in ("LF", "RF", "LM", "RM", "LH", "RH"):
            for group in ("sternotrochanter", "tergotrochanter"):
                with self.subTest(leg=leg, group=group):
                    loaded = ArticulatedLegBody(fixed_root=True)
                    baseline = ArticulatedLegBody(fixed_root=True)
                    joint = f"{leg}_femur_pitch"
                    driven = loaded.step({f"{leg}_{group}": 1e-5}, .0001)
                    sham = baseline.step({}, .0001)
                    self.assertLess(driven["joints"][joint]["torque_Nm"], 0.)
                    self.assertLess(driven["joints"][joint]["qd_rad_s"], sham["joints"][joint]["qd_rad_s"])
                    self.assertEqual(driven["joints"][f"{leg}_femur_roll"]["torque_Nm"], 0.)
                    evidence = loaded.provenance()["muscle_mapping_evidence"][group]
                    self.assertEqual(evidence["moment_arm_evidence"], "engineering_proxy_not_measured")

    def test_friction_coefficient_controls_solved_foot_tangent_load(self):
        muscle_drive = {"LF_tibia_flexor": 1e-5}
        frictionless = ArticulatedLegBody(fixed_root=True, friction=0.0)
        frictional = ArticulatedLegBody(fixed_root=True, friction=0.8)
        no_friction_record = frictionless.step(muscle_drive, 0.001)
        friction_record = frictional.step(muscle_drive, 0.001)
        self.assertLess(no_friction_record["total_foot_tangential_force_sum_N"], 1e-9)
        self.assertGreater(friction_record["total_foot_tangential_force_sum_N"], 1e-8)
        self.assertTrue(any(foot["contact"] for foot in friction_record["feet"].values()))

    def test_invalid_input_and_model_mismatch_fail_closed(self):
        body = ArticulatedLegBody(fixed_root=True)
        with self.assertRaises(ValueError):
            body.step({}, 0.0)
        with self.assertRaises(ValueError):
            body.step({"LF_tibia_flexor": -1e-6}, 0.0001)
        with self.assertRaises(KeyError):
            body.step({"missing_channel": 1e-6}, 0.0001)
        state = body.snapshot()
        state["mjcf_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            body.reset(state)


if __name__ == "__main__":
    unittest.main()
