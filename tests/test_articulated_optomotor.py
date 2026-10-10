"""CPU-only contract checks for the articulated optomotor runner seams."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_articulated_optomotor as runner  # noqa: E402
import run_articulated_optomotor_suite as suite  # noqa: E402
from sensory_feedback import _side_contact  # noqa: E402
from twin_contracts import SensoryFrame, SourceLabel, VisualObservation  # noqa: E402


class ArticulatedOptomotorTest(unittest.TestCase):
    def test_selected_muscle_profile_changes_group_force_and_scales_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = json.loads(runner.DEFAULT_MUSCLE_CONFIG_PATH.read_text(encoding="utf-8"))
            profile["shared"]["max_isometric_force_N"] = 2.0e-5
            profile["groups"] = {"tibia_extensor": {"max_isometric_force_N": 3.0e-5}}
            profile_path = Path(tmp) / "calibrated-profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            class FakeBody:
                def __init__(self) -> None:
                    self.muscles = {
                        "LF_tibia_extensor": SimpleNamespace(
                            group="tibia_extensor", optimal_length_m=1.0e-3,
                        ),
                        "LF_tibia_flexor": SimpleNamespace(
                            group="tibia_flexor", optimal_length_m=1.0e-3,
                        ),
                    }

            class FakeDecoder:
                def __init__(self, *, population_index) -> None:
                    self.population_index = population_index
                    self.channels = ({"name": "LF_tibia_extensor"},
                                     {"name": "LF_tibia_flexor"})

            class FakePopulationIndex:
                ids = np.array([101, 202], dtype=np.int64)
                n = 2
                annotations = {}
                tables = {}

                def resolve(self, _spec: str, *, strict: bool = True) -> np.ndarray:
                    return np.empty(0, dtype=np.int64)

            populations = FakePopulationIndex()
            engine = SimpleNamespace(populations=populations)
            with patch.object(runner, "ArticulatedLegBody", FakeBody), \
                    patch.object(runner, "LegMuscleDecoder", FakeDecoder):
                baseline = runner._make_pipeline(engine)
                selected = runner._make_pipeline(
                    engine, muscle_config_path=profile_path, muscle_force_scale=2.0,
                )

            self.assertIs(selected.decoder.population_index._source, populations)
            self.assertAlmostEqual(baseline.muscles["LF_tibia_extensor"].params.max_isometric_force_N,
                                   1.0e-5)
            self.assertAlmostEqual(selected.muscles["LF_tibia_extensor"].params.max_isometric_force_N,
                                   6.0e-5)
            self.assertAlmostEqual(selected.muscles["LF_tibia_flexor"].params.max_isometric_force_N,
                                   4.0e-5)

    def test_external_profile_snapshot_uses_safe_key_and_detects_later_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            selected_path = temp_root / "external" / "leg_muscle_parameters.json"
            selected_path.parent.mkdir()
            selected_config = json.loads(
                runner.DEFAULT_MUSCLE_CONFIG_PATH.read_text(encoding="utf-8")
            )
            selected_config["shared"]["max_isometric_force_N"] = 2.0e-5
            original = json.dumps(selected_config, indent=2).encode("utf-8")
            selected_path.write_bytes(original)
            resolved, scale, digest = runner.validate_muscle_options(selected_path, 1.0)
            hashes, sources = runner._source_inventory(
                None, resolved, expected_muscle_config_sha256=digest,
            )
            selected_key = "selected-config/leg_muscle_parameters.json"
            default_key = "config/leg_muscle_parameters.json"
            self.assertEqual(sources[selected_key], resolved)
            self.assertNotEqual(selected_key, default_key)
            self.assertEqual(scale, 1.0)

            out = temp_root / "run"
            out.mkdir()
            runner._snapshot_sources(out, hashes, sources)
            selected_snapshot = out / "sources" / selected_key
            default_snapshot = out / "sources" / default_key
            self.assertEqual(selected_snapshot.read_bytes(), original)
            self.assertEqual(default_snapshot.read_bytes(), runner.DEFAULT_MUSCLE_CONFIG_PATH.read_bytes())
            self.assertNotEqual(selected_snapshot.read_bytes(), default_snapshot.read_bytes())
            self.assertEqual(runner._sha256_file(selected_snapshot), digest)

            selected_path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(runner.ArticulatedOptomotorError, "selected-config"):
                runner._verify_source_hashes(hashes, sources, phase="during test")

    def test_muscle_profile_options_fail_before_runtime_setup_when_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            with self.assertRaisesRegex(ValueError, "invalid muscle profile options"):
                runner.run_trial(
                    Path(tmp) / "out", muscle_config_path=missing,
                )
        for scale in (0.0, -1.0, 10.01, float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                runner.validate_muscle_options(None, scale)

    def test_suite_forwards_only_explicit_muscle_profile_options(self) -> None:
        profile = Path("/tmp/profile.json")
        command = suite.command(
            "articulated", 1729, "left", Path("/tmp/out"), "on", 7.5,
            muscle_config=profile, muscle_force_scale=2.5,
        )
        self.assertEqual(command[command.index("--muscle-config") + 1], str(profile))
        self.assertEqual(command[command.index("--muscle-force-scale") + 1], "2.5")
        self.assertNotIn("--muscle-config", suite.command(
            "articulated", 1729, "left", Path("/tmp/default"), "on", 7.5,
        ))
        with self.assertRaisesRegex(ValueError, "require body=articulated"):
            suite.command(
                "reduced", 1729, "left", Path("/tmp/out"), "legacy", 7.5,
                muscle_config=profile,
            )

    def test_motor_unit_sparse_events_reconcile_mapped_and_unmapped_body_ids(self) -> None:
        class PopulationIndex:
            ids = np.array([101, 102, 201, 301, 999, 1000], dtype=np.int64)
            n = 6
            annotations = {}
            tables = {}

            def __init__(self) -> None:
                self.body_id_resolutions = []

            def resolve(self, spec: str, *, strict: bool = True) -> np.ndarray:
                if not spec.startswith("body:"):
                    raise ValueError("expected body-ID selector")
                self.body_id_resolutions.append(spec)
                result = np.flatnonzero(self.ids == int(spec.partition(":")[2]))
                if strict and result.size == 0:
                    raise ValueError("unknown body ID")
                return result

        source = PopulationIndex()
        decoder = SimpleNamespace(
            population_index=runner.DecoderPopulationIndex(source),
            channels=({
                "name": "LF_tibia_flexor", "leg": "LF", "group": "tibia_flexor",
                "type_name": "tib_flexor",
            },),
            _channel_indices={"LF_tibia_flexor": np.array([0], dtype=np.int64)},
            _channel_specs={"LF_tibia_flexor": "exact-selector"},
        )
        all_pools = {
            "LF": np.array([0, 1]), "RF": np.array([2]), "LM": np.array([3]),
            "RM": np.empty(0, dtype=np.int64), "LH": np.empty(0, dtype=np.int64),
            "RH": np.empty(0, dtype=np.int64),
        }
        mapped_pools = {
            "LF": np.array([0]), "RF": np.empty(0, dtype=np.int64),
            "LM": np.empty(0, dtype=np.int64), "RM": np.empty(0, dtype=np.int64),
            "LH": np.empty(0, dtype=np.int64), "RH": np.empty(0, dtype=np.int64),
        }
        mapping, context = runner.build_motor_unit_mapping(
            source, decoder, all_pools, mapped_pools,
        )
        self.assertEqual(mapping["counts"]["annotated_motor_units_total"], 4)
        self.assertEqual(mapping["counts"]["mapped_motor_units_total"], 1)
        self.assertEqual(mapping["counts"]["unmapped_motor_units_total"], 3)
        self.assertEqual(mapping["units"][1]["motor_unit_speed_classification"], "unknown")
        self.assertEqual(mapping["units"][1]["muscle_channels"], [])
        self.assertEqual(len(source.body_id_resolutions), len(mapping["units"]))
        self.assertEqual(source.body_id_resolutions,
                         ["body:101", "body:102", "body:201", "body:301"])

        spikes = np.array([1, 1, 0, 1], dtype=np.uint8)
        event = runner.sparse_motor_unit_event(
            spikes, context, tick=2, neural_t_s=0.001, physical_t_s=0.001,
            annotated_counts={"LF": 2, "RF": 0, "LM": 1, "RM": 0, "LH": 0, "RH": 0},
            mapped_counts={"LF": 1, "RF": 0, "LM": 0, "RM": 0, "LH": 0, "RH": 0},
        )
        self.assertEqual(event["active_body_ids"], [101, 102, 301])
        self.assertEqual(event["active_motor_unit_count"], 3)
        self.assertEqual(event["mapped_active_motor_unit_count"], 1)
        self.assertEqual(event["unmapped_active_motor_unit_count"], 2)
        self.assertEqual(event["unmapped_active_by_leg"]["LF"], 1)
        self.assertEqual(event["unmapped_active_by_leg"]["LM"], 1)
        self.assertEqual(event["neural_t_s"], event["physical_t_s"])
        self.assertAlmostEqual(event["physical_interval_end_t_s"], 0.0015)

    def test_motor_unit_mapping_rejects_a_different_population_index(self) -> None:
        class PopulationIndex:
            def __init__(self) -> None:
                self.ids = np.array([101], dtype=np.int64)
                self.n = 1
                self.annotations = {}
                self.tables = {}

        engine_index = PopulationIndex()
        decoder_index = PopulationIndex()
        decoder = SimpleNamespace(
            population_index=runner.DecoderPopulationIndex(decoder_index),
            channels=(), _channel_indices={}, _channel_specs={},
        )
        pools = {leg: np.empty(0, dtype=np.int64) for leg in runner.LEG_NAMES}
        with self.assertRaisesRegex(runner.ArticulatedOptomotorError, "not using this MaleCNS"):
            runner.build_motor_unit_mapping(engine_index, decoder, pools, pools)

    def test_motor_unit_event_checks_tick_alignment_and_option_defaults_off(self) -> None:
        args = runner.build_parser().parse_args(["--out", "trial", "--condition", "left"])
        self.assertFalse(args.record_motor_units)
        opted_in = runner.build_parser().parse_args([
            "--out", "trial", "--condition", "left", "--record-motor-units",
        ])
        self.assertTrue(opted_in.record_motor_units)

        context = {
            "dense_indices": np.array([0]), "body_ids": np.array([101]),
            "mapped": np.array([False]), "legs": np.array(["LF"], dtype=object),
        }
        with self.assertRaisesRegex(ValueError, "not aligned"):
            runner.sparse_motor_unit_event(
                np.array([1], dtype=np.uint8), context, tick=2,
                neural_t_s=0.001, physical_t_s=0.0015,
                annotated_counts={"LF": 1}, mapped_counts={"LF": 0},
            )

        command = suite.command(
            "articulated", 1729, "left", Path("/tmp/out"), "on", 7.5,
            record_motor_units=True,
        )
        self.assertIn("--record-motor-units", command)
        default_command = suite.command(
            "articulated", 1729, "left", Path("/tmp/default"), "on", 7.5,
        )
        self.assertNotIn("--record-motor-units", default_command)
        with self.assertRaisesRegex(ValueError, "require body=articulated"):
            suite.command(
                "reduced", 1729, "left", Path("/tmp/out"), "legacy", 7.5,
                record_motor_units=True,
            )

    def test_duration_and_cns_clock_are_exactly_nested(self) -> None:
        self.assertEqual(runner.validate_duration(7.5), 1500)
        self.assertEqual(runner.validate_duration(0.025), 5)
        with self.assertRaises(ValueError):
            runner.validate_duration(0.026)

        first = runner.neural_tick_timestamp(0)
        last = runner.neural_tick_timestamp(14999)
        self.assertEqual((first.tick, first.clock, first.rate_hz), (0, "neural", 2000.0))
        self.assertAlmostEqual(last.t_s, 7.4995, places=12)
        self.assertEqual(last.tick, 14999)

    def test_dense_spike_boundary_rejects_bad_neural_observations(self) -> None:
        actual = runner.canonical_spikes(np.array([0, 1, 0], dtype=np.uint8), 3)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual, [0, 1, 0])
        for invalid in (
            np.array([[0, 1, 0]]),
            np.array([0, 1]),
            np.array([0, np.nan, 1]),
            np.array([0, 0.5, 1]),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    runner.canonical_spikes(invalid, 3)

    def test_legacy_mode_keeps_aggregate_groups_and_off_on_mask_only_leg(self) -> None:
        original = {
            "jo": 0.25,
            "haltere": -0.125,
            "wing": 0.5,
            "leg": 0.75,
            "leg_mechanosensory": 0.75,
            "leg_tactile": 0.6,
            "leg_proprioceptive": 0.2,
        }
        legacy, legacy_masked = runner.mask_legacy_leg_drives(original, "legacy")
        self.assertEqual(legacy, original)
        self.assertEqual(legacy_masked, {})

        for mode in ("off", "on"):
            with self.subTest(mode=mode):
                masked, before = runner.mask_legacy_leg_drives(original, mode)
                self.assertEqual(before["leg"], 0.75)
                self.assertEqual(before["leg_tactile"], 0.6)
                self.assertEqual(masked["leg"], 0.0)
                self.assertEqual(masked["leg_mechanosensory"], 0.0)
                self.assertEqual(masked["leg_tactile"], 0.0)
                self.assertEqual(masked["leg_proprioceptive"], 0.0)
                self.assertEqual(masked["jo"], original["jo"])
                self.assertEqual(masked["haltere"], original["haltere"])
                self.assertEqual(masked["wing"], original["wing"])
        self.assertEqual(original["leg"], 0.75)

    def test_legacy_contact_aliases_preserve_six_foot_contact_fractions(self) -> None:
        feet = {leg: {"contact": True} for leg in runner.LEG_NAMES}
        mapped = runner.legacy_contact_mapping(feet)
        self.assertEqual(set(mapped), {
            "front_left", "front_right", "middle_left", "middle_right",
            "hind_left", "hind_right",
        })
        self.assertEqual(_side_contact(mapped), (1.0, 1.0, 1.0))
        for leg in ("RF", "RM", "RH"):
            feet[leg]["contact"] = False
        partial = runner.legacy_contact_mapping(feet)
        self.assertEqual(_side_contact(partial), (1.0, 0.0, 0.5))

    def test_per_tick_articulated_current_replaces_held_current_with_matching_clock(self) -> None:
        frame_time = runner._timestamp(0.0, clock="control", rate_hz=200.0)
        visual = VisualObservation(timestamp=frame_time, pixels=None, source=SourceLabel.FITTED)
        source = SensoryFrame(
            timestamp=frame_time,
            visual=visual,
            channels={"current_mV_per_ms": {"old": 0.0}, "population:jo": 0.2},
        )
        tick = runner.neural_tick_timestamp(1)
        current = {
            "source_time_s": tick.t_s,
            "applied_at_s": tick.t_s,
            "currents_mV_per_ms": {"class:mechanosensory_proprioceptive&subclass:chordotonal organ&nerve:ProLN/L": 0.5},
        }
        result = runner._sensory_frame_at_tick(source, tick, current)
        self.assertEqual(result.timestamp, tick)
        self.assertEqual(result.visual.timestamp, tick)
        self.assertEqual(result.channels["population:jo"], 0.2)
        self.assertEqual(result.channels["current_mV_per_ms"], current["currents_mV_per_ms"])

    def test_decoder_index_adapter_allows_empty_audited_channels_in_engine_order(self) -> None:
        class StrictIndex:
            ids = np.array([101, 202], dtype=np.int64)
            n = 2
            annotations = {"typeIdx": np.array([0, 1])}
            tables = {"types": np.array(["known", "other"])}

            def resolve(self, spec: str, *, strict: bool = True) -> np.ndarray:
                result = np.array([0], dtype=np.int64) if spec == "known" else np.empty(0, dtype=np.int64)
                if strict and result.size == 0:
                    raise ValueError("empty selector")
                return result

        index = runner.DecoderPopulationIndex(StrictIndex())
        self.assertIs(index.ids, StrictIndex.ids)
        self.assertEqual(index.n, 2)
        np.testing.assert_array_equal(index.resolve("known"), [0])
        np.testing.assert_array_equal(index.resolve("empty"), np.empty(0, dtype=np.int64))

    def test_per_leg_motor_counts_keep_annotated_and_mapped_pools_distinct(self) -> None:
        decoder = SimpleNamespace(
            _leg_indices={
                "LF": np.array([0, 1]), "RF": np.array([2]),
                "LM": np.array([3]), "RM": np.array([4]),
                "LH": np.array([5]), "RH": np.array([6]),
            },
            channels=({"name": "LF_mapped", "leg": "LF"},
                       {"name": "RH_mapped", "leg": "RH"}),
            _channel_indices={
                "LF_mapped": np.array([1]), "RH_mapped": np.array([6]),
            },
        )
        annotated, mapped = runner.per_leg_indices(decoder)
        self.assertEqual(sum(indices.size for indices in annotated.values()), 7)
        self.assertEqual(sum(indices.size for indices in mapped.values()), 2)
        np.testing.assert_array_equal(mapped["LF"], [1])
        np.testing.assert_array_equal(mapped["RH"], [6])
        self.assertEqual(sum(annotated[leg].size - mapped[leg].size for leg in runner.LEG_NAMES), 5)

    def test_source_snapshot_scope_uses_tightest_ancestor_memory_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cgroup"
            (root / "scope" / "runner").mkdir(parents=True)
            (root / "memory.max").write_text("8589934592", encoding="ascii")
            (root / "scope" / "memory.max").write_text("5368709120", encoding="ascii")
            (root / "scope" / "runner" / "memory.max").write_text("max", encoding="ascii")
            proc = Path(tmp) / "proc-cgroup"
            proc.write_text("0::/scope/runner\n", encoding="ascii")
            self.assertEqual(
                runner._cgroup_memory_limit_bytes(proc, root), 5 * 1024**3,
            )


if __name__ == "__main__":
    unittest.main()
