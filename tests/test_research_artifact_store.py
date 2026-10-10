"""Focused CPU tests for the H1 research artifact store."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from research_artifact_store import (  # noqa: E402
    ArtifactInput,
    CorruptObjectError,
    ResearchArtifactStore,
    compute_case_payload_hash,
    compute_physics_config_hash,
)


def _register_from_process(root: str, index: int, barrier: object, results: object) -> None:
    """Multiprocess worker for the per-config flock append test."""

    try:
        barrier.wait(timeout=10)  # type: ignore[attr-defined]
        content = f"process-capture-{index}".encode("ascii")
        config = {"dt": 0.0005, "solver": "process-test"}
        config_hash = compute_physics_config_hash(config)
        store = ResearchArtifactStore(root)
        store.register_case_payload(
            case_identity=f"process-case-{index}",
            files={"trajectory.bin": ArtifactInput(content, "E2")},
            physics_config=config,
            declared_physics_config_hash=config_hash,
            completion_status="complete",
            recorded_integrity={"trajectory.bin": hashlib.sha256(content).hexdigest()},
            required_member_manifest={"trajectory.bin": "E2"},
        )
        results.put(None)  # type: ignore[attr-defined]
    except BaseException as exc:
        results.put(f"{type(exc).__name__}: {exc}")  # type: ignore[attr-defined]


class ResearchArtifactStoreTest(unittest.TestCase):
    _DEFAULT_MANIFEST = object()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "object-store"
        self.store = ResearchArtifactStore(self.root)

    @staticmethod
    def _h1(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _register(
        self,
        case_identity: str,
        content: bytes,
        *,
        status: str = "complete",
        declared_hash: str | None = None,
        recorded_integrity: dict[str, str] | None = None,
        required_member_manifest: dict[str, str] | None | object = _DEFAULT_MANIFEST,
    ):
        config = {"dt": 0.0005, "unknown_solver_setting": "kept"}
        config_hash = compute_physics_config_hash(config)
        h1 = self._h1(content)
        if required_member_manifest is self._DEFAULT_MANIFEST:
            required_member_manifest = {"trajectory.npz": "E2"}
        record = self.store.register_case_payload(
            case_identity=case_identity,
            files={"trajectory.npz": ArtifactInput(content, "E2")},
            physics_config=config,
            declared_physics_config_hash=declared_hash or config_hash,
            completion_status=status,
            recorded_integrity=(recorded_integrity if recorded_integrity is not None else {"trajectory.npz": h1}),
            required_member_manifest=required_member_manifest,  # type: ignore[arg-type]
        )
        return config_hash, h1, record

    def test_object_path_is_extension_free_and_metadata_names_are_reference_fields(self) -> None:
        payload = b"small deterministic E3 test object"
        first = self.store.put_bytes(
            payload, original_name="manifest.json", artifact_class="E3", locator="/old/path/manifest.json"
        )
        second = self.store.put_bytes(payload, original_name="renamed.bin", artifact_class="E3")

        expected_path = self.root / "objects" / first.h1[:2] / first.h1[2:4] / first.h1
        self.assertEqual(self.store.object_path(first.h1), expected_path)
        self.assertTrue(expected_path.is_file())
        self.assertEqual(first.h1, second.h1)
        self.assertNotEqual(first.original_name, second.original_name)
        self.assertEqual(expected_path.stat().st_mode & 0o777, 0o444)
        self.assertEqual(len([path for path in (self.root / "objects").rglob("*") if path.is_file()]), 1)

    def test_reference_resolution_uses_h1_when_locator_is_unavailable(self) -> None:
        ref = self.store.put_bytes(b"addressed metadata", original_name="protocol.json", artifact_class="E3")
        stale_reference = {
            "H1": ref.h1,
            "original_name": "protocol.json",
            "artifact_class": "E3",
            "size_bytes": ref.size_bytes,
            "locator": "/archive/offline/or/renamed/protocol.json",
        }

        resolved = self.store.resolve_reference(stale_reference)

        self.assertEqual(resolved.read_bytes(), b"addressed metadata")
        self.assertEqual(resolved.name, ref.h1)
        self.assertFalse(resolved.suffix)

    def test_same_h1_concurrent_writes_publish_one_complete_object(self) -> None:
        content = b"concurrent writer payload\x00" * 2000

        def put(index: int):
            return self.store.put_bytes(
                content,
                original_name=f"capture-{index}.dat",
                artifact_class="E2",
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            references = list(pool.map(put, range(16)))

        self.assertEqual({reference.h1 for reference in references}, {self._h1(content)})
        object_files = [path for path in (self.root / "objects").rglob("*") if path.is_file()]
        self.assertEqual(len(object_files), 1)
        self.assertEqual(object_files[0].read_bytes(), content)
        self.assertFalse(any(path.name.startswith(".incoming-") for path in object_files))

    def test_failed_atomic_publish_leaves_no_canonical_partial_object(self) -> None:
        payload = b"never published"
        expected_h1 = self._h1(payload)
        with mock.patch("research_artifact_store.os.link", side_effect=OSError("simulated publish interruption")):
            with self.assertRaisesRegex(OSError, "simulated publish interruption"):
                self.store.put_bytes(payload, original_name="sample.npz", artifact_class="E2")

        self.assertFalse(self.store.object_path(expected_h1).exists())
        self.assertEqual(list((self.root / "objects").glob(".incoming-*")), [])

    def test_case_hash_is_order_independent_and_path_sensitive(self) -> None:
        content_a = b"a"
        content_b = b"b"
        refs_a = {"z/second.json": self._h1(content_b), "first.npz": self._h1(content_a)}
        refs_b = {"first.npz": self._h1(content_a), "z/second.json": self._h1(content_b)}

        self.assertEqual(compute_case_payload_hash(refs_a), compute_case_payload_hash(refs_b))
        self.assertNotEqual(
            compute_case_payload_hash(refs_a),
            compute_case_payload_hash({"renamed.npz": self._h1(content_a), "z/second.json": self._h1(content_b)}),
        )

    def test_config_lookup_returns_multiple_case_identities_and_reuse_certificate(self) -> None:
        config_hash, _, first = self._register("case-a", b"capture A")
        _, _, second = self._register("case-b", b"capture B")

        found = self.store.query_cases(config_hash)

        self.assertEqual({record.case_identity for record in found}, {"case-a", "case-b"})
        self.assertEqual(len(found), 2)
        self.assertNotEqual(first.case_payload_hash, second.case_payload_hash)
        for record in found:
            self.assertTrue(self.store.is_engineering_reusable(
                record,
                expected_physics_config_hash=config_hash,
                expected_member_manifest={"trajectory.npz": "E2"},
            ))
        certificate = self.store.reuse_certificate(
            first,
            expected_physics_config_hash=config_hash,
            expected_member_manifest={"trajectory.npz": "E2"},
        )
        self.assertEqual(certificate["case_payload_hash"], first.case_payload_hash)
        self.assertTrue(certificate["engineering_reuse_check"]["eligible_for_engineering_reuse"])
        self.assertEqual(certificate["engineering_reuse_check"]["completion_status_complete"], True)

    def test_concurrent_case_index_updates_preserve_every_capture(self) -> None:
        config = {"dt": 0.0005, "solver": "explicit"}
        config_hash = compute_physics_config_hash(config)

        def register(index: int):
            payload = f"capture-{index}".encode("ascii")
            h1 = self._h1(payload)
            return self.store.register_case_payload(
                case_identity=f"case-{index}",
                files={"trajectory.bin": ArtifactInput(payload, "E2")},
                physics_config=config,
                declared_physics_config_hash=config_hash,
                completion_status="complete",
                recorded_integrity={"trajectory.bin": h1},
                required_member_manifest={"trajectory.bin": "E2"},
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(register, range(16)))

        found = self.store.query_cases(config_hash)
        self.assertEqual(len(found), 16)
        self.assertEqual({record.case_identity for record in found}, {f"case-{i}" for i in range(16)})

    def test_cross_process_flock_append_preserves_every_capture(self) -> None:
        context = multiprocessing.get_context("fork")
        barrier = context.Barrier(4)
        results = context.Queue()
        processes = [
            context.Process(
                target=_register_from_process,
                args=(str(self.root), index, barrier, results),
            )
            for index in range(4)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=15)
            self.assertTrue(all(not process.is_alive() for process in processes))
            self.assertEqual([process.exitcode for process in processes], [0, 0, 0, 0])
            self.assertEqual([results.get(timeout=3) for _ in processes], [None] * len(processes))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            results.close()
            results.join_thread()

        config_hash = compute_physics_config_hash({"dt": 0.0005, "solver": "process-test"})
        found = self.store.query_cases(config_hash)
        self.assertEqual(len(found), 4)
        self.assertEqual({record.case_identity for record in found}, {f"process-case-{i}" for i in range(4)})

    def test_reuse_requires_complete_recorded_integrity_and_matching_declared_config(self) -> None:
        config_hash, h1, complete = self._register("valid", b"valid capture")
        _, _, failed = self._register("failed", b"failed capture", status="failed")
        _, _, bad_integrity = self._register(
            "bad-integrity", b"integrity capture", recorded_integrity={"trajectory.npz": self._h1(b"other")}
        )
        _, _, bad_config = self._register(
            "bad-config", b"config capture", declared_hash=self._h1(b"not the declared config")
        )

        expected_members = {"trajectory.npz": "E2"}
        self.assertTrue(self.store.check_engineering_reuse(
            complete,
            expected_physics_config_hash=config_hash,
            expected_member_manifest=expected_members,
        ).eligible)
        self.assertFalse(self.store.check_engineering_reuse(
            failed,
            expected_physics_config_hash=config_hash,
            expected_member_manifest=expected_members,
        ).eligible)
        self.assertFalse(
            self.store.check_engineering_reuse(
                bad_integrity,
                expected_physics_config_hash=config_hash,
                expected_member_manifest=expected_members,
            ).eligible
        )
        self.assertFalse(self.store.check_engineering_reuse(
            bad_config,
            expected_physics_config_hash=config_hash,
            expected_member_manifest=expected_members,
        ).eligible)
        self.assertEqual(complete.files["trajectory.npz"].h1, h1)

    def test_missing_required_member_cannot_pass_integrity_on_reduced_file_set(self) -> None:
        config = {"dt": 0.0005}
        config_hash = compute_physics_config_hash(config)
        trajectory = b"trajectory present"
        trajectory_h1 = self._h1(trajectory)
        record = self.store.register_case_payload(
            case_identity="truncated-capture",
            files={"trajectory.npz": ArtifactInput(trajectory, "E2")},
            physics_config=config,
            declared_physics_config_hash=config_hash,
            completion_status="complete",
            recorded_integrity={"trajectory.npz": trajectory_h1},
            required_member_manifest={
                "trajectory.npz": "E2",
                "contacts.npz": "E2",
            },
        )

        assessment = self.store.check_engineering_reuse(
            record,
            expected_physics_config_hash=config_hash,
            expected_member_manifest={
                "trajectory.npz": "E2",
                "contacts.npz": "E2",
            },
        )

        self.assertTrue(assessment.member_manifest_match)
        self.assertFalse(assessment.actual_members_match)
        self.assertFalse(assessment.eligible)
        self.assertIn("actual_member_set_or_role_mismatch", assessment.reasons)

    def test_missing_capture_contract_or_caller_contract_is_ineligible(self) -> None:
        config_hash, _, record_without_contract = self._register(
            "contract-absent",
            b"otherwise complete",
            required_member_manifest=None,
        )
        record_with_contract = self._register("contract-present", b"valid")[2]

        stored_missing = self.store.check_engineering_reuse(
            record_without_contract,
            expected_physics_config_hash=config_hash,
            expected_member_manifest={"trajectory.npz": "E2"},
        )
        caller_missing = self.store.check_engineering_reuse(
            record_with_contract,
            expected_physics_config_hash=config_hash,
        )

        self.assertFalse(stored_missing.eligible)
        self.assertFalse(stored_missing.member_manifest_match)
        self.assertFalse(caller_missing.eligible)
        self.assertFalse(caller_missing.actual_members_match)

    def test_expected_member_role_must_match_actual_and_stored_manifest(self) -> None:
        config_hash, _, record = self._register("role-check", b"member role")

        assessment = self.store.check_engineering_reuse(
            record,
            expected_physics_config_hash=config_hash,
            expected_member_manifest={"trajectory.npz": "E3"},
        )

        self.assertFalse(assessment.eligible)
        self.assertFalse(assessment.member_manifest_match)
        self.assertFalse(assessment.actual_members_match)

    def test_missing_recorded_integrity_map_is_ineligible(self) -> None:
        config = {"dt": 0.0005}
        config_hash = compute_physics_config_hash(config)
        content = b"no recorded integrity map"
        record = self.store.register_case_payload(
            case_identity="integrity-absent",
            files={"trajectory.npz": ArtifactInput(content, "E2")},
            physics_config=config,
            declared_physics_config_hash=config_hash,
            completion_status="complete",
            recorded_integrity=None,
            required_member_manifest={"trajectory.npz": "E2"},
        )

        assessment = self.store.check_engineering_reuse(
            record,
            expected_physics_config_hash=config_hash,
            expected_member_manifest={"trajectory.npz": "E2"},
        )

        self.assertFalse(assessment.eligible)
        self.assertFalse(assessment.byte_integrity_match)

    def test_missing_cas_object_is_ineligible(self) -> None:
        config_hash, _, record = self._register("deleted-object", b"object to remove")
        self.store.object_path(record.files["trajectory.npz"].h1).unlink()

        assessment = self.store.check_engineering_reuse(
            record,
            expected_physics_config_hash=config_hash,
            expected_member_manifest={"trajectory.npz": "E2"},
        )

        self.assertFalse(assessment.eligible)
        self.assertFalse(assessment.byte_integrity_match)
        self.assertIn("object_integrity_failure:MissingObjectError", assessment.reasons)

    def test_corrupt_object_fails_full_byte_integrity_check(self) -> None:
        config_hash, _, record = self._register("corruptible", b"valid before corruption")
        object_path = self.store.object_path(record.files["trajectory.npz"].h1)
        object_path.chmod(0o644)
        object_path.write_bytes(b"changed bytes")

        assessment = self.store.check_engineering_reuse(
            record,
            expected_physics_config_hash=config_hash,
            expected_member_manifest={"trajectory.npz": "E2"},
        )

        self.assertFalse(assessment.eligible)
        self.assertFalse(assessment.byte_integrity_match)
        with self.assertRaises(CorruptObjectError):
            self.store.resolve_reference(record.files["trajectory.npz"])

    def test_policy_metadata_is_not_an_input_to_existing_config_gate(self) -> None:
        config_hash, _, record = self._register("same-physics-config", b"completed capture")
        policy_before = {"worker_timeout_s": 300, "unit_name": "old-unit"}
        policy_after = {"worker_timeout_s": 600, "unit_name": "new-unit"}
        self.assertNotEqual(policy_before, policy_after)

        # Execution policy is deliberately absent from physics_config and from
        # this module's reuse predicate.  The same completed data remains an
        # engineering candidate while the runner records policy separately.
        self.assertTrue(self.store.is_engineering_reusable(
            record,
            expected_physics_config_hash=config_hash,
            expected_member_manifest={"trajectory.npz": "E2"},
        ))
        self.assertEqual(record.physics_config_hash, compute_physics_config_hash({"dt": 0.0005, "unknown_solver_setting": "kept"}))


if __name__ == "__main__":
    unittest.main()
