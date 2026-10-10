from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_flygym_storage_dispatch_v1 import (
    REQUIRED_MEMBER_MANIFEST,
    _case_unit_ok,
    _parse_systemd_run_start,
    child_systemd_command,
    find_reusable_record,
    persist_novel_mismatch_objects,
    policy_case_decision,
    store_complete_case,
    validate_case_directory,
)


def h1(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_complete_case(root: Path) -> Path:
    root.mkdir()
    capture_files = {}
    for name in REQUIRED_MEMBER_MANIFEST:
        if name == "run-record.json":
            continue
        path = root / name
        path.write_bytes((name + "\n").encode("utf-8"))
        capture_files[name] = {
            "path": str(path),
            "sha256": h1(path),
            "size_bytes": path.stat().st_size,
        }
    (root / "run-record.json").write_text(json.dumps({
        "status": "complete",
        "case_id": "case-a",
        "capture_files": capture_files,
    }), encoding="utf-8")
    return root


class FakeStore:
    def __init__(self, root: Path):
        self.root = root
        self.registered = None
        self.objects: dict[str, Path] = {}
        self.records = []

    def register_case_payload(self, **kwargs):
        self.registered = kwargs
        return types.SimpleNamespace(case_payload_hash="payload-hash")

    def query_cases(self, _config_hash):
        return tuple(self.records)

    def check_engineering_reuse(self, record, **kwargs):
        record.expected_member_manifest = kwargs["expected_member_manifest"]
        return types.SimpleNamespace(eligible=record.eligible)

    def reuse_certificate(self, record, **kwargs):
        record.certificate_member_manifest = kwargs["expected_member_manifest"]
        return {"schema_version": "test", "lookup_hit": True, "case_payload_hash": record.case_payload_hash}

    def object_path(self, value):
        return self.root / value[:2] / value[2:4] / value

    def resolve_reference(self, reference, *, verify_integrity=True):
        path = self.object_path(reference["H1"])
        if verify_integrity and h1(path) != reference["H1"]:
            raise AssertionError("bad object hash")
        return path

    def put_file(self, source_path, *, original_name, artifact_class, locator=None):
        source_path = Path(source_path)
        digest = h1(source_path)
        path = self.object_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source_path.read_bytes())
        self.objects[digest] = path
        return types.SimpleNamespace(h1=digest)


class StorageDispatchTests(unittest.TestCase):
    def test_case_contract_is_frozen_independently_of_observed_files(self):
        self.assertEqual(len(REQUIRED_MEMBER_MANIFEST), 17)
        self.assertEqual(sum(role == "E2" for role in REQUIRED_MEMBER_MANIFEST.values()), 3)
        self.assertEqual(sum(role == "E3" for role in REQUIRED_MEMBER_MANIFEST.values()), 14)

    def test_case_unit_uses_retained_nonblocking_terminal_readback(self):
        command, unit = child_systemd_command("seed1729_dzero_0p00", Path("/tmp/stage"), "policy-check")
        self.assertEqual(command[0:3], ["systemd-run", "--user", "--no-block"])
        self.assertIn("--remain-after-exit", command)
        self.assertNotIn("--wait", command)
        self.assertNotIn("--pipe", command)
        self.assertTrue(any(item.startswith("--property=StandardOutput=append:") for item in command))
        self.assertTrue(any(item.startswith("--property=StandardError=append:") for item in command))
        self.assertTrue(unit.endswith(".service"))

    def test_terminal_receipt_requires_matching_runtime_and_systemd_invocations(self):
        invocation = "88c2acf4808d49fbb08b3c0e035769b5"
        parsed = _parse_systemd_run_start(
            "Running as unit: case.service; invocation ID: " + invocation + "\n"
        )
        self.assertEqual(parsed, {
            "unit_name": "case.service",
            "invocation_id": invocation,
            "invocation_id_source": "systemd_run_start",
        })
        self.assertEqual(
            _parse_systemd_run_start("Running as unit: case.service\n"),
            {"unit_name": "case.service", "invocation_id": None, "invocation_id_source": None},
        )
        runtime_properties = {
            "OOMPolicy": "continue",
            "MemoryMax": str(5 * 1024**3),
            "CPUQuotaPerSecUSec": "4s",
            "RuntimeMaxUSec": "15min",
            "ControlGroup": "/user.slice/case.service",
        }
        runtime_cgroup = {
            "membership_path": "/user.slice/case.service",
            "effective_cpu_cap_cores": 4.0,
            "effective_memory_cap_bytes": 5 * 1024**3,
        }
        receipt = {
            "exit_code": 0,
            "systemd_readback_error": None,
            "systemd_stop_exit_code": 0,
            "systemd_client_start": parsed,
            "terminal_properties": {
                "Id": "case.service",
                "InvocationID": invocation,
                "Result": "success",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
                "ActiveState": "active",
                "SubState": "exited",
            },
            "runtime_receipt": {
                "invocation_id": invocation,
                "properties_at_runtime": runtime_properties,
                "cgroup_at_runtime": runtime_cgroup,
            },
        }
        conversion = {"4s": 4_000_000, "15min": 900_000_000}
        with mock.patch("run_flygym_storage_dispatch_v1._expected_systemd_usec", side_effect=conversion.__getitem__):
            self.assertTrue(_case_unit_ok(receipt))
            receipt["terminal_properties"]["InvocationID"] = "different-invocation"
            self.assertFalse(_case_unit_ok(receipt))

    def test_capture_integrity_requires_exact_members_and_recorded_h1(self):
        with tempfile.TemporaryDirectory() as temp:
            case_dir = make_complete_case(Path(temp) / "case")
            integrity, record = validate_case_directory(case_dir)
            self.assertEqual(set(integrity), set(REQUIRED_MEMBER_MANIFEST))
            self.assertEqual(record["case_id"], "case-a")
            (case_dir / "trajectory.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "disagrees with run-record"):
                validate_case_directory(case_dir)

    def test_complete_case_registration_passes_separate_member_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            case_dir = make_complete_case(Path(temp) / "case")
            store = FakeStore(Path(temp) / "objects")
            result = store_complete_case(
                store,
                case_dir,
                {"case_id": "case-a"},
                {"model": "fixed"},
                "config-hash",
            )
            self.assertEqual(result.case_payload_hash, "payload-hash")
            self.assertEqual(store.registered["required_member_manifest"], REQUIRED_MEMBER_MANIFEST)
            self.assertEqual(set(store.registered["recorded_integrity"]), set(REQUIRED_MEMBER_MANIFEST))
            self.assertEqual(set(store.registered["files"]), set(REQUIRED_MEMBER_MANIFEST))

    def test_cache_hit_passes_expected_member_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            store = FakeStore(Path(temp))
            record = types.SimpleNamespace(case_identity="case-a", case_payload_hash="p1", eligible=True)
            store.records.append(record)
            found, cert = find_reusable_record(store, case_id="case-a", config_hash="config-a")
            self.assertIs(found, record)
            self.assertEqual(record.expected_member_manifest, REQUIRED_MEMBER_MANIFEST)
            self.assertEqual(record.certificate_member_manifest, REQUIRED_MEMBER_MANIFEST)
            self.assertTrue(cert["lookup_hit"])

    def test_matching_and_encoding_differences_do_not_persist_candidate_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            file_path = candidate / "trajectory.npz"
            file_path.write_bytes(b"candidate bytes")
            item = types.SimpleNamespace(path="trajectory.npz", role="E2", size_bytes=file_path.stat().st_size, sha256=h1(file_path))
            fake_fingerprint = lambda digest: types.SimpleNamespace(case_h1=digest, case_h2="same-content", payload_files=(item,), metadata_files=())
            for status in ("content_match", "encoding_difference"):
                store = FakeStore(root / f"objects-{status}")
                comparison = types.SimpleNamespace(
                    status=status,
                    reference=fake_fingerprint("ref-bytes"),
                    candidate=fake_fingerprint(item.sha256),
                    to_dict=lambda status=status: {"status": status, "provenance_differences": []},
                )
                with mock.patch("run_flygym_storage_dispatch_v1._content_api", return_value=(object(), lambda *_args: comparison)):
                    row = policy_case_decision(store, reference, candidate, case_id="case-a")
                self.assertTrue(row["h2_match"])
                self.assertEqual(store.objects, {})

    def test_h2_mismatch_persists_only_candidate_h1_members(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            payload = candidate / "trajectory.npz"
            payload.write_bytes(b"new measurement")
            metadata = candidate / "summary.json"
            metadata.write_text("{}", encoding="utf-8")
            item = types.SimpleNamespace(path="trajectory.npz", role="E2", size_bytes=payload.stat().st_size, sha256=h1(payload))
            meta = types.SimpleNamespace(path="summary.json", role="E3", size_bytes=metadata.stat().st_size, sha256=h1(metadata))
            comparison = types.SimpleNamespace(
                status="content_mismatch_unresolved",
                reference=types.SimpleNamespace(case_h1="ref", case_h2="h2-before", payload_files=(), metadata_files=()),
                candidate=types.SimpleNamespace(case_h1="candidate", case_h2="h2-after", payload_files=(item,), metadata_files=(meta,)),
                to_dict=lambda: {"status": "content_mismatch_unresolved", "provenance_differences": []},
            )
            store = FakeStore(root / "objects")
            with mock.patch("run_flygym_storage_dispatch_v1._content_api", return_value=(object(), lambda *_args: comparison)):
                row = policy_case_decision(store, reference, candidate, case_id="case-a")
            self.assertFalse(row["h2_match"])
            self.assertEqual({item["H1"] for item in row["persisted_novel_objects"]}, {h1(payload), h1(metadata)})
            self.assertEqual(set(store.objects), {h1(payload), h1(metadata)})


if __name__ == "__main__":
    unittest.main()
