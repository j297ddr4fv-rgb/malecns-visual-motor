from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from scripts import artifact_common as common


class ArtifactCommonTests(unittest.TestCase):
    def test_exact_public_operation_set(self):
        tree = ast.parse(Path(common.__file__).read_text())
        names = {node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertEqual(names, {
            "sha256_file", "sha256_bytes", "sha256_json", "sha256_array",
            "read_json", "write_json", "read_jsonl", "write_jsonl",
            "file_record", "utc_now_iso",
        })

    def test_streamed_file_hash_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            payload = b"abc\x00" * (1024 * 1024)
            path.write_bytes(payload)
            link = path.with_name("link")
            link.symlink_to(path)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(common.sha256_file(str(path)), expected)
            self.assertEqual(common.sha256_file(link), expected)
            with self.assertRaises(FileNotFoundError):
                common.sha256_file(path.with_name("missing"))

    def test_bytes_types(self):
        for payload in (b"abc", bytearray(b"abc"), memoryview(b"abc")):
            self.assertEqual(common.sha256_bytes(payload), hashlib.sha256(b"abc").hexdigest())
        with self.assertRaises(TypeError):
            common.sha256_bytes("abc")
        with self.assertRaises(BufferError):
            common.sha256_bytes(memoryview(b"abcd")[::2])

    def test_json_hash_exact_dispatcher_format(self):
        value = {"中文": [1, -0.0, True, None], "a": "é"}
        old_bytes = json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertEqual(common.sha256_json(value), hashlib.sha256(old_bytes).hexdigest())
        self.assertEqual(common.sha256_json({"b": 1, "a": 2}),
                         common.sha256_json({"a": 2, "b": 1}))
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                common.sha256_json(value)

    def test_json_reader_preserves_parser_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text('[1, "中文", NaN]', encoding="utf-8")
            result = common.read_json(path)
            self.assertEqual(result[:2], [1, "中文"])
            self.assertTrue(math.isnan(result[2]))
            path.write_text("", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                common.read_json(path)

    def test_json_writer_bytes_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "record.json"
            value = {"z": 1, "a": "中文"}
            common.write_json(path, value)
            self.assertEqual(path.read_bytes(),
                             (json.dumps(value, indent=2, sort_keys=True,
                                         ensure_ascii=False, allow_nan=False) + "\n").encode())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.chmod(0o640)
            common.write_json(path, {"replacement": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_json_writer_failure_does_not_publish_or_leave_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_bytes(b"old")
            with mock.patch.object(common.os, "replace", side_effect=OSError("rename failure")):
                with self.assertRaises(OSError):
                    common.write_json(path, {"new": 1})
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["record.json"])
            with self.assertRaises(ValueError):
                common.write_json(path, {"invalid": float("nan")})
            self.assertEqual(path.read_bytes(), b"old")

    def test_writer_symlink_target_is_rejected_and_referent_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_bytes(b"old")
            link = path.with_name("link")
            link.symlink_to(path)
            for writer, value in ((common.write_json, {}), (common.write_jsonl, [])):
                with self.assertRaises(ValueError):
                    writer(link, value)
                self.assertTrue(link.is_symlink())
                self.assertEqual(path.read_bytes(), b"old")

    def test_post_publication_fsync_failure_is_visible_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            for writer, value in ((common.write_json, {"new": 1}),
                                  (common.write_jsonl, [{"new": 1}])):
                path.write_bytes(b"old")
                with mock.patch.object(common.os, "fsync", side_effect=[None, OSError("directory sync failed")]):
                    with self.assertRaises(OSError):
                        writer(path, value)
                self.assertNotEqual(path.read_bytes(), b"old")
                self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["data"])

    def test_jsonl_reader_is_lazy_and_rejects_blank_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"a":1}\n\n', encoding="utf-8")
            rows = common.read_jsonl(path)
            self.assertEqual(next(rows), {"a": 1})
            with self.assertRaises(ValueError):
                next(rows)
            path.write_text('{"a":1}', encoding="utf-8")
            self.assertEqual(list(common.read_jsonl(path)), [{"a": 1}])

    def test_jsonl_writer_atomic_stream_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            common.write_jsonl(path, iter([{"z": 2, "a": "中"}, 3]))
            self.assertEqual(path.read_bytes(), '{"a":"中","z":2}\n3\n'.encode())
            previous = path.read_bytes()

            def failing_records():
                yield {"new": 1}
                raise RuntimeError("iterator failed")

            with self.assertRaises(RuntimeError):
                common.write_jsonl(path, failing_records())
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["records.jsonl"])
            common.write_jsonl(path, [])
            self.assertEqual(path.read_bytes(), b"")

    def test_file_record_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_bytes(b"abc")
            path.chmod(0o640)
            self.assertEqual(common.file_record(path), {
                "H1": hashlib.sha256(b"abc").hexdigest(),
                "size_bytes": 3, "mode": "0640",
            })

    def test_utc_seconds(self):
        self.assertRegex(common.utc_now_iso(), r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")

    def test_array_format_endian_shape_dtype_and_layout(self):
        import numpy as np

        little = np.arange(12, dtype="<f8").reshape(3, 4)
        big = little.astype(">f8")
        self.assertEqual(common.sha256_array(little), common.sha256_array(big))
        self.assertEqual(common.sha256_array(little.T),
                         common.sha256_array(np.ascontiguousarray(little.T)))
        self.assertNotEqual(common.sha256_array(little), common.sha256_array(little.reshape(12)))
        self.assertNotEqual(common.sha256_array(little), common.sha256_array(little.astype("<f4")))
        little.flags.writeable = False
        self.assertEqual(common.sha256_array(little), common.sha256_array(big))
        self.assertEqual(common.sha256_array(np.empty((0, 2))),
                         common.sha256_array(np.empty((0, 2))))
        with self.assertRaises(TypeError):
            common.sha256_array(np.array([object()], dtype=object))

    def test_array_nan_payload_and_signed_zero_stay_distinct(self):
        import numpy as np

        first = np.array([0x7FF8000000000001], dtype="<u8").view("<f8")
        second = np.array([0x7FF8000000000002], dtype="<u8").view("<f8")
        self.assertNotEqual(common.sha256_array(first), common.sha256_array(second))
        big_endian_first = first.byteswap().view(">f8")
        self.assertEqual(common.sha256_array(first), common.sha256_array(big_endian_first))
        self.assertNotEqual(common.sha256_array(np.array([-0.0])),
                            common.sha256_array(np.array([0.0])))


class SharedSourceIdentityTests(unittest.TestCase):
    def test_shared_source_h1_changes_physics_config_identity(self):
        from scripts import run_flygym_storage_dispatch_v1 as dispatcher

        context = {
            "legacy": types.SimpleNamespace(EXPECTED_COMPILED_MJB_SHA256="model",
                                             EXPECTED_COMPILED_MJB_SIZE_BYTES=1),
            "protocol": {"execution": {name: 1 for name in (
                "power_P_left_right", "duration_s", "common_prefix_s", "ramp_duration_s",
                "d_schedule", "analysis_window_s")}},
            "freeze": {}, "source_freeze_sha256": "frozen",
            "baseline_model_signature": {}, "baseline_initial_state_hashes": {},
            "controller_values": {},
        }
        case = {"case_id": "fixture", "seed": 1, "d": 0}
        shared_path = Path(dispatcher.artifact_common.__file__)
        hashes = []
        for identity in ("first-source-h1", "second-source-h1"):
            with mock.patch.object(dispatcher, "sha256_file", side_effect=lambda path: (
                identity if Path(path) == shared_path else "other-source-h1"
            )), mock.patch.object(dispatcher, "_body_config", return_value={}), \
                    mock.patch.object(dispatcher, "_content_api", return_value=(
                        types.SimpleNamespace(schema_id="fixture"), None)):
                config = dispatcher.physics_config(context, case, {}, "dispatcher-h1")
            self.assertEqual(config["source_hashes"]["artifact_common_sha256"], identity)
            hashes.append(common.sha256_json(config))
        self.assertNotEqual(*hashes)

    def test_prepared_protocol_rejects_changed_shared_source_before_context_load(self):
        from scripts import run_flygym_storage_dispatch_v1 as dispatcher

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps({
                "schema_version": "malecns-mechanical-storage-dispatch-v1",
                "dispatcher_sha256": common.sha256_file(dispatcher.__file__),
                "artifact_common_sha256": "obsolete-source-h1",
            }), encoding="utf-8")
            with mock.patch.object(dispatcher, "PROTOCOL_PATH", path), \
                    mock.patch.object(dispatcher, "load_legacy_context") as context:
                with self.assertRaisesRegex(dispatcher.DispatchError, "shared artifact source changed"):
                    dispatcher.load_prepared_protocol()
                context.assert_not_called()

    def test_dispatch_object_requirement_survives_shared_parsing(self):
        from scripts import run_flygym_storage_dispatch_v1 as dispatcher

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(dispatcher.DispatchError):
                dispatcher.load_dispatch_record(path)
            path.write_text('{"nan":NaN}', encoding="utf-8")
            self.assertTrue(math.isnan(dispatcher.load_dispatch_record(path)["nan"]))


if __name__ == "__main__":
    unittest.main()
