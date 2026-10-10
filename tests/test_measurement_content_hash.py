from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from measurement_content_hash import (  # noqa: E402
    CaseSchema,
    FLYGym_ONLINE_SCHEMA,
    FileSchema,
    MeasurementHashError,
    compare_cases,
    hash_jsonl,
    hash_npz,
    fingerprint_case,
)


def _save_npz(path: Path, *, compressed: bool, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        writer = np.savez_compressed if compressed else np.savez
        writer(stream, **arrays)


class MeasurementContentHashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_npz_hash_ignores_container_encoding_and_key_insertion_order(self) -> None:
        schema = FileSchema("npz", "fixture-v1", ("alpha", "beta"), {"metadata_json": "fixture metadata"})
        first = self.root / "first.npz"
        second = self.root / "second.npz"
        alpha = np.asarray([1.25, -2.5], dtype="<f8")
        beta = np.asarray([[1, 2], [3, 4]], dtype="<i4")
        _save_npz(first, compressed=True, arrays={"beta": beta, "metadata_json": np.asarray("a"), "alpha": alpha})
        _save_npz(second, compressed=False, arrays={"alpha": alpha.copy(), "metadata_json": np.asarray("different metadata"), "beta": beta.copy()})

        self.assertNotEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(hash_npz(first, schema), hash_npz(second, schema))

    def test_npz_canonicalizes_endian_without_changing_dtype_precision(self) -> None:
        schema = FileSchema("npz", "endian-fixture-v1", ("values",))
        little = self.root / "little.npz"
        big = self.root / "big.npz"
        _save_npz(little, compressed=True, arrays={"values": np.asarray([1.5, -0.25], dtype="<f4")})
        _save_npz(big, compressed=True, arrays={"values": np.asarray([1.5, -0.25], dtype=">f4")})
        self.assertNotEqual(little.read_bytes(), big.read_bytes())
        self.assertEqual(hash_npz(little, schema), hash_npz(big, schema))

        lower_precision = self.root / "lower.npz"
        _save_npz(lower_precision, compressed=True, arrays={"values": np.asarray([1.5, -0.25], dtype="<f2")})
        self.assertNotEqual(hash_npz(little, schema), hash_npz(lower_precision, schema))

    def test_npz_float_hash_is_bit_exact_for_nan_payloads_and_signed_zero(self) -> None:
        schema = FileSchema("npz", "float-bits-v1", ("values",))
        same_nan_a = self.root / "same-nan-a.npz"
        same_nan_b = self.root / "same-nan-b.npz"
        other_nan = self.root / "other-nan.npz"
        positive_zero = self.root / "positive-zero.npz"
        negative_zero = self.root / "negative-zero.npz"
        nan_bits = np.asarray([0x7FF8000000000001], dtype="<u8").view("<f8")
        other_nan_bits = np.asarray([0x7FF8000000000002], dtype="<u8").view("<f8")
        _save_npz(same_nan_a, compressed=True, arrays={"values": nan_bits})
        _save_npz(same_nan_b, compressed=False, arrays={"values": nan_bits.copy()})
        _save_npz(other_nan, compressed=True, arrays={"values": other_nan_bits})
        _save_npz(positive_zero, compressed=True, arrays={"values": np.asarray([0.0], dtype="<f8")})
        _save_npz(negative_zero, compressed=True, arrays={"values": np.asarray([-0.0], dtype="<f8")})

        self.assertEqual(hash_npz(same_nan_a, schema), hash_npz(same_nan_b, schema))
        self.assertNotEqual(hash_npz(same_nan_a, schema), hash_npz(other_nan, schema))
        self.assertNotEqual(hash_npz(positive_zero, schema), hash_npz(negative_zero, schema))

    def test_npz_shape_and_unknown_array_are_fail_closed(self) -> None:
        schema = FileSchema("npz", "shape-fixture-v1", ("values",))
        one = self.root / "one.npz"
        two = self.root / "two.npz"
        unknown = self.root / "unknown.npz"
        _save_npz(one, compressed=True, arrays={"values": np.asarray([1, 2], dtype="<i4")})
        _save_npz(two, compressed=True, arrays={"values": np.asarray([[1, 2]], dtype="<i4")})
        _save_npz(unknown, compressed=True, arrays={"values": np.asarray([1, 2], dtype="<i4"), "extra": np.asarray([9])})
        self.assertNotEqual(hash_npz(one, schema), hash_npz(two, schema))
        with self.assertRaisesRegex(MeasurementHashError, "unknown NPZ arrays"):
            hash_npz(unknown, schema)

    def test_jsonl_uses_explicit_fields_and_preserves_float_bits(self) -> None:
        schema = FileSchema(
            "jsonl", "json-fixture-v1", ("seq", "value"), {"origin": "metadata"},
            nested_allowed_children={"value": ("a", "b")},
        )
        first = self.root / "first.jsonl"
        second = self.root / "second.jsonl"
        first.write_text('{"seq":0,"value":{"a":-0.0,"b":1.0},"origin":"one"}\n', encoding="utf-8")
        second.write_text('{ "origin" : "two", "value" : { "b" : 1.0, "a" : -0.0 }, "seq" : 0 }\n', encoding="utf-8")
        self.assertEqual(hash_jsonl(first, schema), hash_jsonl(second, schema))

        plus_zero = self.root / "plus-zero.jsonl"
        plus_zero.write_text('{"seq":0,"value":{"a":0.0,"b":1.0},"origin":"one"}\n', encoding="utf-8")
        self.assertNotEqual(hash_jsonl(first, schema), hash_jsonl(plus_zero, schema))

    def test_jsonl_rejects_unknown_or_missing_root_fields(self) -> None:
        schema = FileSchema("jsonl", "closed-json-fixture-v1", ("seq", "value"), {"origin": "metadata"})
        unknown = self.root / "unknown.jsonl"
        missing = self.root / "missing.jsonl"
        unknown.write_text('{"seq":0,"value":1,"origin":"x","future_capture":3}\n', encoding="utf-8")
        missing.write_text('{"seq":0,"value":1}\n', encoding="utf-8")
        with self.assertRaisesRegex(MeasurementHashError, "unknown capture fields"):
            hash_jsonl(unknown, schema)
        with self.assertRaisesRegex(MeasurementHashError, "missing declared capture fields"):
            hash_jsonl(missing, schema)

    def _write_case(self, name: str, stream: str, provenance: dict[str, object]) -> Path:
        root = self.root / name
        root.mkdir()
        (root / "measurements.jsonl").write_text(stream, encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
        return root

    @staticmethod
    def _valid_online_frame_row() -> dict[str, object]:
        row: dict[str, object] = {name: 0.0 for name in FLYGym_ONLINE_SCHEMA.include}
        row.update({name: None for name in FLYGym_ONLINE_SCHEMA.exclude})
        row.update({
            "seq": 0,
            "body_command_applied_this_frame": {
                "P_control": 0.0,
                "body_action_end_s": 0.005,
                "body_action_start_s": 0.0,
                "body_frame_seq": 0,
                "command_delay_frames": 1,
                "held_soma_side": "left",
                "initial_neutral_command": True,
                "signed_d": 0.0,
                "source_neural_seq": None,
                "source_neural_window_end_s": None,
            },
            "decoder_output": {
                "action_boundary": "abstract template intent",
                "classification": "engineered readout",
                "control_power": 0.0,
                "legacy_field_semantics": {
                    "clipped_opponent_signal": "gated command",
                    "ema_left_rate_hz/ema_right_rate_hz": "command EMA",
                    "raw_opponent_signal": "raw signal",
                    "zero_power_baseline_hold": "held at raw zero",
                },
                "power_mapping": "symmetric control power",
                "schema_version": "fixture",
            },
            "decoder_output_queued_for_next_frame": {
                "P_control": 0.0,
                "available_at_body_time_s": 0.005,
                "held_soma_side": "left",
                "signed_d": 0.0,
                "source_neural_seq": 0,
                "source_neural_window_end_s": 0.005,
                "source_neural_window_start_s": 0.0,
            },
            "input_pose": {
                "angular_velocity_radps": [0.0, 0.0, 0.0],
                "pitch_rad": 0.0,
                "position_m": [0.0, 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "roll_rad": 0.0,
                "velocity_mps": [0.0, 0.0, 0.0],
                "yaw_rad": 0.0,
            },
            "source_body_pose": {
                "eye_camera_baseline_m": 0.00076,
                "eye_camera_left_origin_mm": [0.0, 0.38, 0.0],
                "eye_camera_midpoint_mm": [0.0, 0.0, 0.0],
                "eye_camera_right_origin_mm": [0.0, -0.38, 0.0],
                "eye_velocity_flygym_world_mps": [0.0, 0.0, 0.0],
                "head_frame": "fixture",
                "thorax_angular_velocity_radps": [0.0, 0.0, 0.0],
                "thorax_linear_velocity_mm_s": [0.0, 0.0, 0.0],
                "thorax_position_mm": [0.0, 0.0, 0.0],
                "thorax_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "thorax_rotation_body_to_world": [
                    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                ],
            },
            "virtual_eye_origins_m": {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]},
        })
        return row

    def _write_online_case(self, name: str, row: dict[str, object]) -> Path:
        root = self.root / name
        root.mkdir()
        (root / "online-frames.jsonl").write_text(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8",
        )
        return root

    def test_case_comparison_separates_payload_content_from_provenance(self) -> None:
        schema = CaseSchema(
            "fixture-case-v1",
            {"measurements.jsonl": FileSchema("jsonl", "measurements-v1", ("seq", "value"), {"artifact_dir": "run path"})},
            ("manifest.json",),
        )
        a = self._write_case("a", '{"seq":0,"value":1.0,"artifact_dir":"/a"}\n', {"unit": "cpu-a", "seed": 5})
        b = self._write_case("b", '{ "artifact_dir" : "/b", "value" : 1.0, "seq" : 0 }\n', {"unit": "cpu-b", "seed": 5})
        report = compare_cases(a, b, schema)
        self.assertEqual(report.status, "encoding_difference")
        self.assertEqual(len(report.provenance_differences), 2)
        by_role = {item["role"]: item for item in report.provenance_differences}
        self.assertEqual(by_role["E3"]["path"], "manifest.json")
        self.assertIn("/unit", {entry["pointer"] for entry in by_role["E3"]["items"]})
        e2_item = by_role["E2"]
        self.assertEqual(e2_item["path"], "measurements.jsonl")
        self.assertEqual(e2_item["items"][0]["pointer"], "/artifact_dir")
        self.assertEqual(e2_item["items"][0]["reference"], "/a")
        self.assertEqual(e2_item["items"][0]["candidate"], "/b")
        self.assertEqual(e2_item["items"][0]["line"], 1)
        self.assertEqual(e2_item["items"][0]["reference_seq"], 0)
        self.assertEqual(report.payload_differences[0]["kind"], "encoding_difference")

        c = self._write_case("c", '{"seq":0,"value":2.0,"artifact_dir":"/c"}\n', {"unit": "cpu-c", "seed": 5})
        mismatch = compare_cases(a, c, schema)
        self.assertEqual(mismatch.status, "content_mismatch_unresolved")
        self.assertIn("/unit", {entry["pointer"] for entry in mismatch.provenance_differences[0]["items"]})
        self.assertNotIn("config_change", mismatch.status)

    def test_measurement_change_is_not_reported_as_e2_provenance(self) -> None:
        schema = CaseSchema(
            "fixture-measurement-diff-v1",
            {"measurements.jsonl": FileSchema("jsonl", "measurements-v1", ("seq", "value"), {"artifact_dir": "run path"})},
            ("manifest.json",),
        )
        a = self._write_case("measurement-a", '{"seq":0,"value":1.0,"artifact_dir":"/same"}\n', {"seed": 5})
        b = self._write_case("measurement-b", '{"seq":0,"value":2.0,"artifact_dir":"/same"}\n', {"seed": 5})
        report = compare_cases(a, b, schema)
        self.assertEqual(report.status, "content_mismatch_unresolved")
        self.assertFalse([item for item in report.provenance_differences if item["role"] == "E2"])

    def test_nested_excluded_jsonl_metadata_is_reported_by_pointer_and_sequence(self) -> None:
        schema = CaseSchema(
            "fixture-nested-e2-metadata-v1",
            {
                "measurements.jsonl": FileSchema(
                    "jsonl", "measurements-v1", ("seq", "value", "source"),
                    nested_exclude={"source.backend": "backend provenance"},
                    nested_allowed_children={"source": ("backend", "name")},
                )
            },
            ("manifest.json",),
        )
        a = self._write_case("nested-a", '{"seq":4,"value":1.0,"source":{"backend":"a","name":"worker"}}\n', {"seed": 5})
        b = self._write_case("nested-b", '{"seq":4,"value":1.0,"source":{"backend":"b","name":"worker"}}\n', {"seed": 5})
        report = compare_cases(a, b, schema)
        self.assertEqual(report.status, "encoding_difference")
        e2 = next(item for item in report.provenance_differences if item["role"] == "E2")
        self.assertEqual(e2["items"], [{
            "reason": "backend provenance", "pointer": "/source/backend", "kind": "value_changed",
            "reference": "a", "candidate": "b", "line": 1, "reference_line": 1,
            "candidate_line": 1, "reference_seq": 4, "candidate_seq": 4, "alignment_field": "seq",
        }])

    def test_excluded_npz_metadata_json_reports_concrete_pointers(self) -> None:
        schema = CaseSchema(
            "fixture-npz-metadata-v1",
            {"measurements.npz": FileSchema("npz", "measurements-npz-v1", ("values",), {"metadata_json": "NPZ capture provenance"})},
            ("manifest.json",),
        )
        a = self.root / "npz-meta-a"
        b = self.root / "npz-meta-b"
        a.mkdir()
        b.mkdir()
        _save_npz(a / "measurements.npz", compressed=True, arrays={
            "values": np.asarray([1.0, 2.0], dtype="<f4"),
            "metadata_json": np.asarray('{"source":{"backend":"a","unit":"mV"},"version":1}'),
        })
        _save_npz(b / "measurements.npz", compressed=False, arrays={
            "values": np.asarray([1.0, 2.0], dtype="<f4"),
            "metadata_json": np.asarray('{ "version":1, "source":{"unit":"mV","backend":"b"} }'),
        })
        (a / "manifest.json").write_text('{"seed":5}\n', encoding="utf-8")
        (b / "manifest.json").write_text('{"seed":5}\n', encoding="utf-8")
        report = compare_cases(a, b, schema)
        self.assertEqual(report.status, "encoding_difference")
        e2 = next(item for item in report.provenance_differences if item["role"] == "E2")
        self.assertEqual(e2["path"], "measurements.npz")
        self.assertEqual(e2["items"], [{
            "reason": "NPZ capture provenance",
            "pointer": "/metadata_json/source/backend",
            "kind": "value_changed",
            "reference": "a",
            "candidate": "b",
        }])

    def test_metadata_only_difference_keeps_payload_h1_and_h2_equal(self) -> None:
        schema = CaseSchema(
            "fixture-provenance-only-v1",
            {"measurements.jsonl": FileSchema("jsonl", "measurements-v1", ("seq", "value"))},
            ("manifest.json",),
        )
        a = self._write_case("metadata-a", '{"seq":0,"value":1.0}\n', {"backend": "x"})
        b = self._write_case("metadata-b", '{"seq":0,"value":1.0}\n', {"backend": "y"})
        report = compare_cases(a, b, schema)
        self.assertEqual(report.status, "content_match")
        self.assertEqual(report.reference.case_h1, report.candidate.case_h1)
        self.assertEqual(report.reference.case_h2, report.candidate.case_h2)
        self.assertTrue(report.provenance_differences)

    def test_nested_object_schema_rejects_unknown_children(self) -> None:
        schema = FileSchema(
            "jsonl", "closed-nested-fixture-v1", ("seq", "measurement"),
            nested_allowed_children={"measurement": ("value", "unit")},
        )
        path = self.root / "unknown-nested.jsonl"
        path.write_text(
            '{"seq":0,"measurement":{"value":1.25,"unit":"mV","future_metadata":"x"}}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(MeasurementHashError, "unknown nested capture fields"):
            hash_jsonl(path, schema)

    def test_named_axis_numeric_map_preserves_keys_and_rejects_non_numeric_values(self) -> None:
        schema = FileSchema(
            "jsonl", "named-axis-fixture-v1", ("seq", "population_rates_hz"),
            named_axis_numeric_maps=("population_rates_hz",),
        )
        a = self.root / "axis-a.jsonl"
        b = self.root / "axis-b.jsonl"
        unknown_value = self.root / "axis-invalid.jsonl"
        a.write_text('{"seq":0,"population_rates_hz":{"DNa02/L":2.0,"HSE/R":3.0}}\n', encoding="utf-8")
        b.write_text('{"seq":0,"population_rates_hz":{"HSE/R":3.0,"DNa02/L":2.0}}\n', encoding="utf-8")
        unknown_value.write_text('{"seq":0,"population_rates_hz":{"new-population":{"rate":1.0}}}\n', encoding="utf-8")
        self.assertEqual(hash_jsonl(a, schema), hash_jsonl(b, schema))
        with self.assertRaisesRegex(MeasurementHashError, "named-axis map values must be numeric"):
            hash_jsonl(unknown_value, schema)

    def test_changed_jsonl_rows_align_by_seq_and_report_added_removed_metadata(self) -> None:
        schema = CaseSchema(
            "fixture-row-alignment-v1",
            {"measurements.jsonl": FileSchema(
                "jsonl", "row-alignment-v1", ("seq", "value"), {"artifact_dir": "run path"},
            )},
            ("manifest.json",),
        )
        reference_rows = (
            '{"seq":0,"value":1,"artifact_dir":"/zero"}\n'
            '{"seq":1,"value":2,"artifact_dir":"/one"}\n'
        )
        reference = self._write_case("rows-reference", reference_rows, {"seed": 5})

        added = self._write_case(
            "rows-added",
            reference_rows + '{"seq":2,"value":3,"artifact_dir":"/two"}\n',
            {"seed": 5},
        )
        added_report = compare_cases(reference, added, schema)
        self.assertEqual(added_report.status, "content_mismatch_unresolved")
        added_items = next(x for x in added_report.provenance_differences if x["role"] == "E2")["items"]
        self.assertEqual([(x["pointer"], x["kind"], x["candidate_seq"]) for x in added_items], [
            ("/artifact_dir", "row_added", 2),
        ])

        removed = self._write_case("rows-removed", '{"seq":0,"value":1,"artifact_dir":"/zero"}\n', {"seed": 5})
        removed_report = compare_cases(reference, removed, schema)
        removed_items = next(x for x in removed_report.provenance_differences if x["role"] == "E2")["items"]
        self.assertEqual([(x["pointer"], x["kind"], x["reference_seq"]) for x in removed_items], [
            ("/artifact_dir", "row_removed", 1),
        ])

        reordered = self._write_case(
            "rows-reordered",
            '{"seq":1,"value":2,"artifact_dir":"/one"}\n'
            '{"seq":0,"value":1,"artifact_dir":"/zero"}\n',
            {"seed": 5},
        )
        reordered_report = compare_cases(reference, reordered, schema)
        self.assertEqual(reordered_report.status, "content_mismatch_unresolved")
        self.assertFalse([x for x in reordered_report.provenance_differences if x["role"] == "E2"])

    def test_jsonl_provenance_alignment_rejects_missing_or_duplicate_identifiers(self) -> None:
        no_id_schema = CaseSchema(
            "fixture-no-id-v1",
            {"measurements.jsonl": FileSchema("jsonl", "no-id-v1", ("value",), {"origin": "metadata"})},
            ("manifest.json",),
        )
        a = self._write_case("no-id-a", '{"value":1,"origin":"a"}\n', {"seed": 5})
        b = self._write_case("no-id-b", '{"value":1,"origin":"b"}\n', {"seed": 5})
        with self.assertRaisesRegex(MeasurementHashError, "must include a stable seq or tick"):
            compare_cases(a, b, no_id_schema)

        duplicate_schema = CaseSchema(
            "fixture-duplicate-seq-v1",
            {"measurements.jsonl": FileSchema("jsonl", "duplicate-seq-v1", ("seq", "value"), {"origin": "metadata"})},
            ("manifest.json",),
        )
        duplicate_a = self._write_case(
            "duplicate-a",
            '{"seq":0,"value":1,"origin":"a"}\n{"seq":0,"value":2,"origin":"a"}\n',
            {"seed": 5},
        )
        duplicate_b = self._write_case(
            "duplicate-b",
            '{"seq":0,"value":1,"origin":"b"}\n{"seq":0,"value":2,"origin":"a"}\n',
            {"seed": 5},
        )
        with self.assertRaisesRegex(MeasurementHashError, "duplicate stable seq identifier"):
            compare_cases(duplicate_a, duplicate_b, duplicate_schema)

    def test_debug_snapshot_family_rejects_missing_members_on_both_sides(self) -> None:
        schema = CaseSchema(
            "fixture-debug-family-v1",
            {},
            (),
            debug_npz_dir="worker/debug-neural-state",
            debug_npz_schema=FileSchema("npz", "debug-array-v1", ("v",)),
            debug_npz_expected_frames=1,
            debug_npz_initial_required=True,
        )
        case_a = self.root / "debug-a"
        case_b = self.root / "debug-b"
        for case in (case_a, case_b):
            debug = case / "worker/debug-neural-state"
            debug.mkdir(parents=True)
            _save_npz(debug / "initial.npz", compressed=True, arrays={"v": np.asarray([1.0], dtype="<f4")})
            # Both sides lack frame-0000.npz; identical incompleteness is still invalid.
        with self.assertRaisesRegex(MeasurementHashError, "required debug-state capture members are missing"):
            compare_cases(case_a, case_b, schema)

    def test_debug_snapshot_family_rejects_empty_directory_and_missing_initial(self) -> None:
        schema = CaseSchema(
            "fixture-debug-empty-v1",
            {},
            (),
            debug_npz_dir="worker/debug-neural-state",
            debug_npz_schema=FileSchema("npz", "debug-array-v1", ("v",)),
            debug_npz_expected_frames=1,
            debug_npz_initial_required=True,
        )
        case = self.root / "debug-empty"
        (case / "worker/debug-neural-state").mkdir(parents=True)
        with self.assertRaisesRegex(MeasurementHashError, "required debug-state capture members are missing"):
            fingerprint_case(case, schema)

    def test_pose_required_children_cannot_be_missing_on_both_sides(self) -> None:
        case_schema = CaseSchema(
            "fixture-online-pose-required-v1",
            {"online-frames.jsonl": FLYGym_ONLINE_SCHEMA},
            (),
        )
        reference_row = self._valid_online_frame_row()
        candidate_row = self._valid_online_frame_row()
        del reference_row["input_pose"]["position_m"]
        del candidate_row["input_pose"]["position_m"]
        reference = self._write_online_case("pose-missing-reference", reference_row)
        candidate = self._write_online_case("pose-missing-candidate", candidate_row)
        with self.assertRaisesRegex(MeasurementHashError, "missing required nested capture fields.*position_m"):
            compare_cases(reference, candidate, case_schema)

    def test_pose_required_child_type_cannot_change_to_scalar(self) -> None:
        case_schema = CaseSchema(
            "fixture-online-pose-type-v1",
            {"online-frames.jsonl": FLYGym_ONLINE_SCHEMA},
            (),
        )
        reference_row = self._valid_online_frame_row()
        candidate_row = self._valid_online_frame_row()
        candidate_row["input_pose"]["position_m"] = 0.0
        reference = self._write_online_case("pose-type-reference", reference_row)
        candidate = self._write_online_case("pose-type-candidate", candidate_row)
        with self.assertRaisesRegex(MeasurementHashError, "expected nested array.*position_m"):
            compare_cases(reference, candidate, case_schema)

    def test_declared_nested_object_cannot_be_replaced_with_null(self) -> None:
        case_schema = CaseSchema(
            "fixture-online-pose-object-type-v1",
            {"online-frames.jsonl": FLYGym_ONLINE_SCHEMA},
            (),
        )
        reference = self._write_online_case("pose-object-reference", self._valid_online_frame_row())
        invalid_row = self._valid_online_frame_row()
        invalid_row["input_pose"] = None
        candidate = self._write_online_case("pose-object-null", invalid_row)
        with self.assertRaisesRegex(MeasurementHashError, "expected nested object.*input_pose"):
            compare_cases(reference, candidate, case_schema)


if __name__ == "__main__":
    unittest.main()
