"""Validation and scaling checks for the pipeline's shared Hill parameters."""
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from leg_muscle_dynamics import HillMuscle, MuscleParameters
from leg_muscle_parameters import DEFAULT_CONFIG_PATH, LegMuscleParameters
from leg_muscle_pipeline import LegMusclePipeline


@dataclass(frozen=True)
class Descriptor:
    group: str
    optimal_length_m: float

    @property
    def joint_moment_arms_m(self):
        return {}


class OneChannelDecoder:
    channels = ({"name": "demo"},)

    def decode(self, *, rates_hz):
        return {"demo": float(rates_hz[0])}

    def provenance(self):
        return {"fixture": True}


class OneMuscleBody:
    def __init__(self):
        self.muscles = {"demo": Descriptor("tibia_flexor", 180e-6)}

    def muscle_kinematics(self):
        # A displaced current pose must not reset the descriptor's l0.
        return {"demo": {"length_m": 360e-6, "velocity_mps": 0.0}}

    def step(self, forces, *, dt_s):
        return {"forces_N": forces, "dt_s": dt_s}

    def provenance(self):
        return {"fixture": True}


class LegMuscleParametersTests(unittest.TestCase):
    groups = {"tibia_flexor", "tibia_extensor"}

    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = Path(self.tmp.name) / "muscle-parameters.json"

    def loader(self, *, force_scale=1.0, data=None):
        self.config_path.write_text(
            json.dumps(self.config if data is None else data), encoding="utf-8"
        )
        return LegMuscleParameters(
            self.config_path, force_scale=force_scale, known_groups=self.groups
        )

    def test_normalized_velocity_uses_immutable_descriptor_l0(self):
        loader = self.loader()
        short = loader.parameters_for(Descriptor("tibia_flexor", 180e-6))
        long = loader.parameters_for(Descriptor("tibia_flexor", 300e-6))
        self.assertAlmostEqual(short.optimal_length_m, 180e-6)
        self.assertAlmostEqual(short.max_shortening_velocity_mps, 1.8e-3)
        self.assertAlmostEqual(long.max_shortening_velocity_mps, 3.0e-3)

    def test_group_override_is_local_and_inherits_shared_parameters(self):
        self.config["groups"] = {
            "tibia_flexor": {
                "max_isometric_force_N": 2e-5,
                "max_shortening_velocity_optimal_lengths_per_s": 5.0,
            }
        }
        loader = self.loader()
        descriptor_f = Descriptor("tibia_flexor", 180e-6)
        descriptor_e = Descriptor("tibia_extensor", 180e-6)
        flexor = loader.parameters_for(descriptor_f)
        extensor = loader.parameters_for(descriptor_e)
        self.assertEqual(flexor.max_isometric_force_N, 2e-5)
        self.assertAlmostEqual(flexor.max_shortening_velocity_mps, 0.9e-3)
        self.assertEqual(extensor.max_isometric_force_N, 1e-5)
        self.assertAlmostEqual(extensor.max_shortening_velocity_mps, 1.8e-3)
        self.assertEqual(flexor.activation_time_constant_s,
                         extensor.activation_time_constant_s)

    def test_force_scale_multiplies_every_group_fiso(self):
        self.config["groups"] = {"tibia_flexor": {"max_isometric_force_N": 2e-5}}
        loader = self.loader(force_scale=2.5)
        flexor = loader.parameters_for(Descriptor("tibia_flexor", 180e-6))
        extensor = loader.parameters_for(Descriptor("tibia_extensor", 180e-6))
        self.assertEqual(flexor.max_isometric_force_N, 5e-5)
        self.assertEqual(extensor.max_isometric_force_N, 2.5e-5)
        self.assertEqual(loader.provenance()["force_scale"], 2.5)

    def test_strict_config_validation_rejects_unknown_and_invalid_values(self):
        bad_documents = []
        unknown_top = dict(self.config, surprise=True)
        bad_documents.append((unknown_top, "unknown"))
        unknown_shared = json.loads(json.dumps(self.config))
        unknown_shared["shared"]["optimal_length_m"] = 1e-3
        bad_documents.append((unknown_shared, "unknown Hill parameter"))
        unknown_group = json.loads(json.dumps(self.config))
        unknown_group["groups"]["mystery"] = {"max_isometric_force_N": 1e-5}
        bad_documents.append((unknown_group, "unknown muscle groups"))
        unknown_override = json.loads(json.dumps(self.config))
        unknown_override["groups"]["tibia_flexor"] = {"max_shortening_velocity_mps": 1.0}
        bad_documents.append((unknown_override, "unknown Hill parameter"))
        invalid = json.loads(json.dumps(self.config))
        invalid["shared"]["max_isometric_force_N"] = -1e-5
        bad_documents.append((invalid, "positive"))

        for data, expected in bad_documents:
            with self.subTest(expected=expected, data=data):
                self.config_path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    LegMuscleParameters(self.config_path, known_groups=self.groups)

    def test_force_scale_must_be_finite_positive_and_at_most_ten(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        for value in (0, -1, 10.01, float("inf"), float("nan"), True):
            with self.subTest(force_scale=value), self.assertRaises(ValueError):
                LegMuscleParameters(self.config_path, value, known_groups=self.groups)
        self.assertEqual(
            LegMuscleParameters(self.config_path, 10, known_groups=self.groups).force_scale,
            10.0,
        )

    def test_default_hill_constructor_values_are_unchanged(self):
        defaults = MuscleParameters()
        self.assertEqual(defaults.optimal_length_m, 0.0006)
        self.assertEqual(defaults.max_shortening_velocity_mps, 0.0018)

    def test_pipeline_accepts_shared_config_and_force_scale(self):
        self.config["groups"] = {"tibia_flexor": {"max_isometric_force_N": 2e-5}}
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        pipeline = LegMusclePipeline(
            decoder=OneChannelDecoder(), body=OneMuscleBody(),
            muscle_config_path=self.config_path, muscle_force_scale=3.0,
        )
        params = pipeline.muscles["demo"].params
        self.assertEqual(params.optimal_length_m, 180e-6)
        self.assertAlmostEqual(params.max_isometric_force_N, 6e-5)
        self.assertAlmostEqual(params.max_shortening_velocity_mps, 1.8e-3)
        record = pipeline.step(dt_s=0.0001, rates_hz=[0.0])
        self.assertEqual(record["muscles"]["demo"]["excitation"], 0.0)
        self.assertEqual(
            pipeline.provenance()["muscle_parameter_config"]["force_scale"], 3.0
        )

    def test_activation_and_inhibition_decay_are_sane(self):
        params = self.loader().parameters_for(Descriptor("tibia_flexor", 180e-6))
        muscle = HillMuscle(params)
        idle = muscle.step(0.0, params.optimal_length_m, 0.0, 0.01)
        active = muscle.step(1.0, params.optimal_length_m, 0.0, 0.01)
        inhibited = muscle.step(0.0, params.optimal_length_m, 0.0, 0.04)
        self.assertEqual(idle.total_force_N, 0.0)
        self.assertGreater(active.activation, idle.activation)
        self.assertGreater(active.active_force_N, 0.0)
        self.assertLess(inhibited.activation, active.activation)
        self.assertGreaterEqual(inhibited.total_force_N, 0.0)


if __name__ == "__main__":
    unittest.main()
