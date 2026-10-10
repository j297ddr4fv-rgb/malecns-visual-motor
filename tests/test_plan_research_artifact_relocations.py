from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import plan_research_artifact_relocations as relocation


class RelocationPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "engineering"
        self.source.mkdir()
        self.archive_mount = self.base / "archive-mount"
        self.archive_mount.mkdir()
        self.destination = self.archive_mount / "archive/by-project/malecns-gpu/engineering"
        self.output = self.source / "implementation-output"
        self.object_store = self.base / "object-store"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_plan(self) -> dict:
        with mock.patch.object(
            relocation,
            "_open_file_snapshot",
            return_value={"status": "complete", "open_inodes": [], "unreadable_pid_count": 0},
        ):
            return relocation.create_plan(
                self.source,
                self.destination,
                self.archive_mount,
                self.output,
                self.object_store,
                validate_mounts=False,
            )

    def test_identical_bytes_are_classified_redundant_with_active_survivor(self) -> None:
        (self.source / "a.bin").write_bytes(b"same payload")
        (self.source / "b.bin").write_bytes(b"same payload")

        report = self.make_plan()
        rows = {row["relative_path"]: row for row in report["files"]}

        self.assertEqual(rows["a.bin"]["class"], "redundant")
        self.assertEqual(rows["b.bin"]["class"], "redundant")
        self.assertEqual(rows["a.bin"]["planned_action"], "retain_active_survivor")
        self.assertEqual(rows["b.bin"]["surviving_copy"], str(self.source / "a.bin"))
        self.assertEqual(rows["b.bin"]["new_path"], str(self.destination / "b.bin"))
        self.assertEqual(report["summary"]["eligible_relocation_files_U1"], 1)
        self.assertGreater(report["summary"]["estimated_cross_filesystem_reclaim_bytes_lower_bound_if_executed_now_U3"], 0)
        self.assertEqual(report["summary"]["actual_disk_reclaimed_bytes_U3"], 0)

    def test_same_size_different_content_is_unique_not_redundant(self) -> None:
        (self.source / "a.bin").write_bytes(b"aaaa")
        (self.source / "b.bin").write_bytes(b"bbbb")

        report = self.make_plan()

        self.assertEqual(report["hashing"]["regular_files_hashed"], 2)
        self.assertTrue(report["hashing"]["size_is_not_used_as_duplicate_proof"])
        self.assertEqual({row["class"] for row in report["files"]}, {"unique"})
        self.assertEqual(report["summary"]["eligible_relocation_files_U1"], 0)

    def test_hardlink_alias_does_not_claim_allocated_space_reclaim(self) -> None:
        first = self.source / "a.bin"
        first.write_bytes(b"hardlinked payload")
        os.link(first, self.source / "b.bin")

        report = self.make_plan()

        self.assertEqual(report["summary"]["redundant_files_U1"], 2)
        self.assertEqual(report["summary"]["eligible_relocation_files_U1"], 1)
        self.assertEqual(report["summary"]["estimated_cross_filesystem_reclaim_bytes_lower_bound_if_executed_now_U3"], 0)
        self.assertEqual(report["physical_reclaim_by_inode"][0]["reason"], "retained_hardlink_in_or_outside_scope")

    def test_symlink_is_reported_without_following_or_hashing_target(self) -> None:
        target = self.source / "payload.bin"
        target.write_bytes(b"target bytes")
        link = self.source / "alias.bin"
        link.symlink_to(target)

        report = self.make_plan()
        rows = {row["relative_path"]: row for row in report["files"]}

        self.assertEqual(rows["alias.bin"]["kind"], "symlink")
        self.assertEqual(rows["alias.bin"]["class"], "excluded_symlink")
        self.assertIsNone(rows["alias.bin"]["h1_sha256"])
        self.assertEqual(rows["alias.bin"]["symlink_target"], str(target))
        self.assertEqual(rows["payload.bin"]["class"], "unique")

    def test_casefold_conflict_blocks_exfat_destination_relocation(self) -> None:
        upper = self.source / "Case" / "payload.bin"
        lower = self.source / "case" / "payload.bin"
        upper.parent.mkdir()
        lower.parent.mkdir()
        upper.write_bytes(b"matching")
        lower.write_bytes(b"matching")

        report = self.make_plan()

        self.assertEqual(len(report["case_insensitive_destination_conflicts"]), 1)
        self.assertEqual(report["summary"]["eligible_relocation_files_U1"], 0)
        self.assertEqual(report["summary"]["blocked_relocation_candidates_U1"], 1)
        candidate = next(row for row in report["files"] if row["relocation_candidate"])
        self.assertEqual(candidate["planned_action"], "blocked_case_insensitive_destination_collision")
        self.assertTrue(candidate["case_conflict"])

    def test_unreadable_unrelated_processes_do_not_block_verified_candidate(self) -> None:
        (self.source / "a.bin").write_bytes(b"same payload")
        (self.source / "b.bin").write_bytes(b"same payload")
        partial_snapshot = {
            "status": "partial", "open_inodes": [], "write_open_inodes": [],
            "unreadable_pid_count": 366,
        }
        with mock.patch.object(relocation, "_open_file_snapshot", return_value=partial_snapshot):
            report = relocation.create_plan(
                self.source, self.destination, self.archive_mount, self.output,
                self.object_store, validate_mounts=False,
            )

        candidate = next(row for row in report["files"] if row.get("relocation_candidate"))
        self.assertEqual(candidate["planned_action"], "eligible_after_review_and_later_authorization")
        self.assertEqual(candidate["open_status"], "unknown_due_to_unreadable_processes_reclaim_uncertain")
        self.assertEqual(report["summary"]["estimated_cross_filesystem_reclaim_bytes_lower_bound_if_executed_now_U3"], 0)
        self.assertGreater(
            report["summary"]["estimated_cross_filesystem_reclaim_bytes_upper_bound_if_no_open_references_U3"],
            0,
        )

    def test_known_writable_source_inode_blocks_only_that_candidate(self) -> None:
        (self.source / "a.bin").write_bytes(b"same payload")
        candidate = self.source / "b.bin"
        candidate.write_bytes(b"same payload")
        st = candidate.stat()
        writer_snapshot = {
            "status": "complete",
            "open_inodes": [[st.st_dev, st.st_ino]],
            "write_open_inodes": [[st.st_dev, st.st_ino]],
            "unreadable_pid_count": 0,
        }
        with mock.patch.object(relocation, "_open_file_snapshot", return_value=writer_snapshot):
            report = relocation.create_plan(
                self.source, self.destination, self.archive_mount, self.output,
                self.object_store, validate_mounts=False,
            )

        candidate_row = next(row for row in report["files"] if row.get("relocation_candidate"))
        self.assertEqual(candidate_row["planned_action"], "blocked_source_inode_known_writable_open")
        self.assertEqual(report["summary"]["eligible_relocation_files_U1"], 0)


class RelocationExecutionTests(unittest.TestCase):
    scratch_root = (
        Path("/data/artifacts/by-project/malecns-gpu/engineering")
        / "storage-remediation-implementation-20261008-v1" / "scratch"
    )
    archive_device = 0x7FFF0001

    @classmethod
    def setUpClass(cls) -> None:
        cls.scratch_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.scratch_root.rmdir()
        except OSError:
            pass

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="relocation-test-", dir=self.scratch_root)
        self.fixture = Path(self.temporary.name)
        self.source_root = relocation.DEFAULT_SOURCE_ROOT
        self.active_dir = self.fixture / "active"
        self.active_dir.mkdir()
        self.survivor = self.active_dir / "a.bin"
        self.old = self.active_dir / "b.bin"
        self.payload = b"isolated relocation test payload\n"
        self.survivor.write_bytes(self.payload)
        self.old.write_bytes(self.payload)
        self.old.chmod(0o640)
        self.archive_mount = self.fixture / "mock-archive-mount"
        self.archive_mount.mkdir()
        self.archive_root = self.archive_mount / "archive/by-project/malecns-gpu/engineering"
        self.records_path = self.fixture / "relocations.json"
        self.row = self.make_row()
        self.records = {"schema": "malecns-gpu.relocations.v1", "relocations": []}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_row(self) -> dict:
        relative = self.old.relative_to(self.source_root).as_posix()
        return {
            "old_path": str(self.old),
            "new_path": str(self.archive_root / relative),
            "relative_path": relative,
            "h1_sha256": relocation._sha256_path(self.old),
            "logical_bytes": self.old.stat().st_size,
            "allocated_bytes": self.old.stat().st_blocks * 512,
            "mode": relocation._mode(self.old.stat().st_mode),
            "class": "redundant",
            "planned_action": "eligible_after_review_and_later_authorization",
            "surviving_copy": str(self.survivor),
            "surviving_copy_after_plan": str(self.survivor),
            "device": self.old.stat().st_dev,
            "inode": self.old.stat().st_ino,
            "nlink": self.old.stat().st_nlink,
        }

    def fake_device_id(self, path: Path) -> int:
        path = Path(path)
        try:
            path.relative_to(self.archive_mount)
            return self.archive_device
        except ValueError:
            return int(os.stat(path).st_dev)

    def mounted_archive_mocks(self):
        return (
            mock.patch.object(relocation, "_assert_archive_mount", return_value=self.archive_device),
            mock.patch.object(relocation, "_filesystem_device_id", side_effect=self.fake_device_id),
            mock.patch.object(
                relocation, "_open_file_snapshot",
                return_value={"status": "complete", "open_inodes": [], "write_open_inodes": [],
                              "unreadable_pid_count": 0},
            ),
        )

    def run_one(self) -> dict:
        patches = self.mounted_archive_mocks()
        with patches[0], patches[1], patches[2]:
            return relocation._execute_one(
                self.row, self.source_root, self.archive_root, self.archive_mount,
                self.records_path, self.records, int(self.row["nlink"]),
            )

    def test_offline_archive_mount_does_not_create_mount_or_destination(self) -> None:
        missing_mount = self.fixture / "not-mounted"
        destination = missing_mount / "archive/by-project/malecns-gpu/engineering"
        row = dict(self.row)
        row["new_path"] = str(destination / row["relative_path"])

        with self.assertRaisesRegex(relocation.PlanError, "offline"):
            relocation._execute_one(
                row, self.source_root, destination, missing_mount,
                self.records_path, self.records, int(row["nlink"]),
            )

        self.assertFalse(missing_mount.exists())
        self.assertFalse(destination.exists())
        self.assertTrue(self.old.is_file())
        self.assertFalse(self.old.is_symlink())

    def test_source_h1_change_is_rejected_before_destination_creation(self) -> None:
        self.old.write_bytes(bytes([self.payload[0] ^ 1]) + self.payload[1:])

        with self.assertRaisesRegex(relocation.PlanError, "H1 changed"):
            self.run_one()

        self.assertTrue(self.old.is_file())
        self.assertFalse(self.old.is_symlink())
        self.assertFalse(self.archive_root.exists())

    def test_survivor_h1_change_is_rejected_before_source_replacement(self) -> None:
        self.survivor.write_bytes(b"survivor changed after review")

        with self.assertRaisesRegex(relocation.PlanError, "H1 changed"):
            self.run_one()

        self.assertTrue(self.old.is_file())
        self.assertFalse(self.old.is_symlink())
        self.assertFalse(self.archive_root.exists())

    def test_symlink_survivor_to_archive_is_rejected_without_hashing_archive_target(self) -> None:
        cold_target = self.archive_mount / "cold-target.bin"
        cold_target.write_bytes(self.payload)
        self.survivor.unlink()
        self.survivor.symlink_to(cold_target)
        hashed_paths: list[Path] = []
        actual_hash = relocation._sha256_path

        def capture_hash(path: Path) -> str:
            hashed_paths.append(Path(path))
            return actual_hash(path)

        patches = self.mounted_archive_mocks()
        with patches[0], patches[1], patches[2], mock.patch.object(relocation, "_sha256_path", side_effect=capture_hash):
            with self.assertRaisesRegex(relocation.PlanError, "symlink"):
                relocation._execute_one(
                    self.row, self.source_root, self.archive_root, self.archive_mount,
                    self.records_path, self.records, int(self.row["nlink"]),
                )

        self.assertNotIn(cold_target, hashed_paths)
        self.assertTrue(self.old.is_file())
        self.assertFalse(self.old.is_symlink())
        self.assertFalse(self.archive_root.exists())

    def test_records_parent_symlink_outside_data_is_rejected(self) -> None:
        link = self.fixture / "records-parent-link"
        link.symlink_to("/tmp", target_is_directory=True)

        with self.assertRaisesRegex(relocation.PlanError, "resolves outside /data"):
            relocation._validate_records_path(link / "relocations.json")

    def test_record_write_failure_leaves_source_path_intact(self) -> None:
        patches = self.mounted_archive_mocks()
        actual_atomic = relocation._atomic_json
        calls = 0

        def fail_before_parking(path: Path, value: dict) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated records write failure before source parking")
            actual_atomic(path, value)

        with patches[0], patches[1], patches[2], mock.patch.object(
            relocation, "_atomic_json", side_effect=fail_before_parking
        ):
            with self.assertRaisesRegex(OSError, "before source parking"):
                relocation._execute_one(
                    self.row, self.source_root, self.archive_root, self.archive_mount,
                    self.records_path, self.records, int(self.row["nlink"]),
                )

        self.assertTrue(self.old.is_file())
        self.assertFalse(self.old.is_symlink())
        self.assertEqual(self.old.read_bytes(), self.payload)
        saved = json.loads(self.records_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["relocations"][-1]["status"], "archive_copy_and_inventory_durable_source_present")
        self.assertTrue((self.archive_root / self.row["relative_path"]).is_file())
        inventory = json.loads((self.archive_root / relocation.ARCHIVE_INVENTORY_NAME).read_text())
        self.assertEqual(len(inventory["items"]), 1)

    def test_success_writes_self_describing_inventory_and_active_compatible_symlink(self) -> None:
        actual_replace = os.replace
        durable_checkpoints: list[str] = []

        def check_order(source: os.PathLike, target: os.PathLike) -> None:
            source_path = Path(source)
            target_path = Path(target)
            if source_path == self.old or target_path == self.old:
                self.assertTrue(self.records_path.is_file())
                saved = json.loads(self.records_path.read_text(encoding="utf-8"))
                saved_event = saved["relocations"][-1]
                self.assertTrue((self.archive_root / relocation.ARCHIVE_INVENTORY_NAME).is_file())
                self.assertTrue(saved_event.get("archive_inventory_durable"))
                durable_checkpoints.append(saved_event["status"])
            actual_replace(source, target)

        with mock.patch.object(os, "replace", side_effect=check_order):
            event = self.run_one()

        self.assertEqual(event["status"], "complete")
        self.assertEqual(durable_checkpoints, ["source_parking_intent_recorded", "source_parked_recoverable"])
        self.assertTrue(self.old.is_symlink())
        self.assertEqual(self.old.resolve(strict=True), self.survivor.resolve(strict=True))
        self.assertTrue(self.old.resolve().is_relative_to(self.source_root))
        self.assertFalse(self.old.resolve().is_relative_to(self.archive_mount))
        self.assertEqual(relocation._sha256_path(self.old), self.row["h1_sha256"])

        archive_file = self.archive_root / self.row["relative_path"]
        inventory_path = self.archive_root / relocation.ARCHIVE_INVENTORY_NAME
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        self.assertEqual(len(inventory["items"]), 1)
        item = inventory["items"][0]
        self.assertEqual(item["H1"], self.row["h1_sha256"])
        self.assertEqual(item["relative_archive_path"], self.row["relative_path"])
        self.assertEqual(item["bytes"], self.row["logical_bytes"])
        self.assertEqual(item["source_original_mode"], self.row["mode"])
        self.assertNotIn("/data/", json.dumps(inventory))
        self.assertTrue(relocation.verify_archive_inventory(self.archive_root)["ok"])

        # Existing mechanical/canonical H1 guards follow the active survivor link.
        from scripts.run_flygym_steering_dose_calibration_v6 import sha256_file
        from scripts.run_flygym_canonical_fullfield_optomotor_v1 import sha256

        self.assertTrue(self.old.is_file())
        self.assertEqual(sha256_file(self.old), self.row["h1_sha256"])
        self.assertEqual(sha256(self.old), self.row["h1_sha256"])
        self.assertEqual(self.old.resolve(strict=True), self.survivor.resolve(strict=True))

    def _execute_until_parked_record_failure(self) -> None:
        actual_atomic = relocation._atomic_json
        calls = 0

        def fail_after_park(path: Path, value: dict) -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("simulated crash after source parking")
            actual_atomic(path, value)

        patches = self.mounted_archive_mocks()
        with patches[0], patches[1], patches[2], mock.patch.object(
            relocation, "_atomic_json", side_effect=fail_after_park
        ):
            with self.assertRaisesRegex(OSError, "after source parking"):
                relocation._execute_one(
                    self.row, self.source_root, self.archive_root, self.archive_mount,
                    self.records_path, self.records, int(self.row["nlink"]),
                )
        saved = json.loads(self.records_path.read_text(encoding="utf-8"))
        event = saved["relocations"][-1]
        self.assertEqual(event["status"], "source_parking_intent_recorded")
        self.assertFalse(self.old.exists())
        self.assertTrue(Path(event["parked_source_path"]).is_file())

    def test_crash_recovery_restore_uses_parked_source_and_keeps_archive(self) -> None:
        self._execute_until_parked_record_failure()

        with mock.patch.object(
            relocation, "_validate_recorded_archive_root", side_effect=lambda root, mount: Path(root)
        ), mock.patch.object(
            relocation, "_assert_recorded_archive_mount", side_effect=AssertionError("restore should use parked copy")
        ):
            changed = relocation.recover_pending(self.records_path, "restore")

        self.assertEqual(len(changed), 1)
        self.assertTrue(self.old.is_file())
        self.assertFalse(self.old.is_symlink())
        self.assertEqual(self.old.read_bytes(), self.payload)
        self.assertTrue((self.archive_root / self.row["relative_path"]).is_file())
        self.assertEqual(changed[0]["status"], "rolled_back_source_restored_archive_copy_retained")

    def test_crash_recovery_finish_replaces_with_active_survivor_and_keeps_archive(self) -> None:
        self._execute_until_parked_record_failure()
        expected_records_hash = relocation._sha256_path(self.records_path)
        patches = self.mounted_archive_mocks()
        with patches[1], mock.patch.object(
            relocation, "_validate_recorded_archive_root", side_effect=lambda root, mount: Path(root)
        ), mock.patch.object(
            relocation, "_assert_recorded_archive_mount", return_value=self.archive_device
        ):
            changed = relocation.recover_pending(
                self.records_path, "finish", expected_records_hash
            )

        self.assertEqual(len(changed), 1)
        self.assertTrue(self.old.is_symlink())
        self.assertEqual(self.old.resolve(strict=True), self.survivor.resolve(strict=True))
        self.assertFalse(Path(changed[0]["parked_source_path"]).exists())
        self.assertTrue((self.archive_root / self.row["relative_path"]).is_file())
        self.assertEqual(changed[0]["status"], "complete")

if __name__ == "__main__":
    unittest.main()
