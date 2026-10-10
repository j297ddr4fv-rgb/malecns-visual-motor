#!/usr/bin/env python3
"""Byte-addressed storage for research case artifacts.

The object store keeps each file once at ``objects/<H1[0:2]>/<H1[2:4]>/<H1>``.
References retain the original name and optional locator as metadata; resolving
a reference consults only its H1.  The small per-physics-config index is an
implementation detail for finding case payload records.  It is not a scientific
result manifest and does not decide scientific evidence classifications.

Public runner API::

    store = ResearchArtifactStore(object_store_root)
    record = store.register_case_payload(
        case_identity="case-17",
        files={
            "trajectory.npz": ArtifactInput(path, "E2"),
            "run-record.json": ArtifactInput(record_path, "E3"),
        },
        physics_config=physics_config,
        declared_physics_config_hash=declared_config_hash,
        completion_status="complete",
        recorded_integrity={"trajectory.npz": trajectory_h1,
                            "run-record.json": run_record_h1},
        required_member_manifest={"trajectory.npz": "E2",
                                  "run-record.json": "E3"},
    )
    candidates = store.query_cases(record.physics_config_hash)
    for candidate in candidates:
        assessment = store.check_engineering_reuse(
            candidate,
            expected_physics_config_hash=record.physics_config_hash,
            expected_member_manifest={"trajectory.npz": "E2",
                                      "run-record.json": "E3"},
        )

``physics_config`` is the already scoped physical configuration.  Its JSON
encoding is deterministic, includes every supplied field (including unknown
fields), and does not silently remove execution-policy fields.  Callers keep
policy metadata separate based on measured evidence.  This module has no
policy-inertness or scientific-classification logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, BinaryIO, Literal


ArtifactClass = Literal["E2", "E3"]
_VALID_ARTIFACT_CLASSES = frozenset({"E2", "E3"})
_INDEX_SCHEMA = "research-artifact-store-index-1"
_CHUNK_SIZE = 1024 * 1024


class ArtifactStoreError(RuntimeError):
    """Base error raised for malformed references or store failures."""


class InvalidReferenceError(ArtifactStoreError):
    """A reference does not contain a valid H1 or artifact metadata."""


class MissingObjectError(ArtifactStoreError):
    """A content-addressed object is absent from this object store."""


class CorruptObjectError(ArtifactStoreError):
    """An object at its canonical H1 path is not byte-integral."""


@dataclass(frozen=True)
class ArtifactReference:
    """Reference to one object; the locator is informational only."""

    h1: str
    original_name: str
    artifact_class: ArtifactClass
    size_bytes: int
    locator: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "H1": self.h1,
            "original_name": self.original_name,
            "artifact_class": self.artifact_class,
            "size_bytes": self.size_bytes,
        }
        if self.locator is not None:
            value["locator"] = self.locator
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactReference":
        if not isinstance(value, Mapping):
            raise InvalidReferenceError("reference must be a mapping")
        h1 = _validated_h1(value.get("H1"))
        original_name = value.get("original_name")
        artifact_class = value.get("artifact_class")
        size_bytes = value.get("size_bytes")
        locator = value.get("locator")
        if not isinstance(original_name, str) or not original_name:
            raise InvalidReferenceError("reference original_name must be a non-empty string")
        if artifact_class not in _VALID_ARTIFACT_CLASSES:
            raise InvalidReferenceError("reference artifact_class must be E2 or E3")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise InvalidReferenceError("reference size_bytes must be a non-negative integer")
        if locator is not None and not isinstance(locator, str):
            raise InvalidReferenceError("reference locator must be a string when present")
        return cls(h1, original_name, artifact_class, size_bytes, locator)


@dataclass(frozen=True)
class ArtifactInput:
    """One case member and its explicit E2/E3 class."""

    source: bytes | bytearray | memoryview | Path | str | ArtifactReference
    artifact_class: ArtifactClass
    original_name: str | None = None
    locator: str | None = None


@dataclass(frozen=True)
class CasePayloadRecord:
    """One indexed observation of a case payload for a physics config."""

    case_identity: str
    case_payload_hash: str
    physics_config_hash: str
    declared_physics_config_hash: str
    completion_status: str
    files: Mapping[str, ArtifactReference]
    recorded_integrity: Mapping[str, str] | None
    required_member_manifest: Mapping[str, ArtifactClass] | None


@dataclass(frozen=True)
class ReuseAssessment:
    """Engineering eligibility checks, independent of scientific conclusions."""

    eligible: bool
    completion_match: bool
    byte_integrity_match: bool
    config_match: bool
    member_manifest_match: bool
    actual_members_match: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_for_engineering_reuse": self.eligible,
            "completion_status_complete": self.completion_match,
            "byte_integrity_match": self.byte_integrity_match,
            "physics_config_match": self.config_match,
            "required_member_manifest_match": self.member_manifest_match,
            "actual_members_match": self.actual_members_match,
            "reasons": list(self.reasons),
        }


def _validated_h1(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise InvalidReferenceError("H1 must be a 64-character lowercase SHA-256 hex string")
    if any(character not in "0123456789abcdef" for character in value):
        raise InvalidReferenceError("H1 must be a 64-character lowercase SHA-256 hex string")
    return value


def _validated_artifact_class(value: str) -> ArtifactClass:
    if value not in _VALID_ARTIFACT_CLASSES:
        raise ValueError("artifact_class must be 'E2' or 'E3'")
    return value  # type: ignore[return-value]


def _normalized_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("case member path must be a non-empty relative POSIX path")
    if "\\" in value:
        raise ValueError(f"case member path must use POSIX separators: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"case member path must be a safe relative path: {value!r}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"case member path must be a non-empty relative path: {value!r}")
    return normalized


def _normalized_member_manifest(
    value: Mapping[str, str] | None,
) -> dict[str, ArtifactClass] | None:
    """Normalize an independent capture-contract path-to-role declaration."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("required_member_manifest must be a path-to-role mapping")
    if not value:
        return None
    normalized: dict[str, ArtifactClass] = {}
    for raw_path, raw_role in value.items():
        path = _normalized_relative_path(raw_path)
        if path in normalized:
            raise ValueError(f"duplicate normalized manifest path: {path!r}")
        normalized[path] = _validated_artifact_class(raw_role)
    return dict(sorted(normalized.items()))


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes, rejecting non-finite floats."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical-JSON serializable: {exc}") from exc
    return text.encode("utf-8")


def compute_physics_config_hash(physics_config: Mapping[str, Any]) -> str:
    """Hash all supplied physics-config fields without field classification."""

    if not isinstance(physics_config, Mapping):
        raise TypeError("physics_config must be a mapping")
    return hashlib.sha256(canonical_json_bytes(physics_config)).hexdigest()


def compute_case_payload_hash(file_references: Mapping[str, ArtifactReference | Mapping[str, Any] | str]) -> str:
    """Hash sorted ``(relative_path, H1)`` pairs for one case payload."""

    if not isinstance(file_references, Mapping) or not file_references:
        raise ValueError("case payload must contain at least one file reference")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_path, reference in file_references.items():
        relative_path = _normalized_relative_path(raw_path)
        if relative_path in seen:
            raise ValueError(f"duplicate normalized case member path: {relative_path!r}")
        seen.add(relative_path)
        if isinstance(reference, ArtifactReference):
            h1 = _validated_h1(reference.h1)
        elif isinstance(reference, Mapping):
            h1 = _validated_h1(reference.get("H1"))
        else:
            h1 = _validated_h1(reference)
        pairs.append((relative_path, h1))
    pairs.sort(key=lambda item: item[0])
    return hashlib.sha256(canonical_json_bytes([[path, h1] for path, h1 in pairs])).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        _fsync_directory(directory.parent)


def _hash_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(_CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _hash_path_no_follow(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError as exc:
        raise MissingObjectError(f"object not found: {path}") from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise CorruptObjectError(f"canonical object path is a symlink: {path}") from exc
        raise
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise CorruptObjectError(f"canonical object path is not a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return _hash_open_file(handle)
    finally:
        os.close(fd)


class ResearchArtifactStore:
    """Immutable H1 object store plus a small physics-config lookup index."""

    def __init__(self, object_store_root: str | os.PathLike[str]):
        self.root = Path(object_store_root).expanduser().resolve(strict=False)
        self.objects_root = self.root / "objects"
        self.index_root = self.root / "indexes" / "physics-config"
        self.locks_root = self.root / "locks"

    def object_path(self, h1: str) -> Path:
        h1 = _validated_h1(h1)
        return self.objects_root / h1[:2] / h1[2:4] / h1

    def _store_stream(
        self,
        source: BinaryIO,
        *,
        original_name: str,
        artifact_class: ArtifactClass,
        locator: str | None,
    ) -> ArtifactReference:
        if not isinstance(original_name, str) or not original_name:
            raise ValueError("original_name must be a non-empty string")
        artifact_class = _validated_artifact_class(artifact_class)

        # The first pass stages complete bytes in the target file system.  A
        # hard link publishes that closed, fsynced inode without replacing a
        # concurrent writer's canonical path.
        _mkdir_durable(self.objects_root)
        staging_dir = self.objects_root
        fd, temporary_name = tempfile.mkstemp(prefix=".incoming-", dir=staging_dir)
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with os.fdopen(fd, "wb") as staged:
                while True:
                    chunk = source.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    staged.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                staged.flush()
                os.fsync(staged.fileno())
                os.fchmod(staged.fileno(), 0o444)
                os.fsync(staged.fileno())

            h1 = digest.hexdigest()
            destination = self.object_path(h1)
            _mkdir_durable(destination.parent)
            try:
                os.link(temporary_path, destination, follow_symlinks=False)
                _fsync_directory(destination.parent)
            except FileExistsError:
                found = _hash_path_no_follow(destination)
                if found != h1:
                    raise CorruptObjectError(
                        f"object exists at {destination} but hashes to {found}, expected {h1}"
                    )
            temporary_path.unlink(missing_ok=True)
            _fsync_directory(staging_dir)
            return ArtifactReference(
                h1=h1,
                original_name=original_name,
                artifact_class=artifact_class,
                size_bytes=size_bytes,
                locator=locator,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def put_bytes(
        self,
        content: bytes | bytearray | memoryview,
        *,
        original_name: str,
        artifact_class: ArtifactClass,
        locator: str | None = None,
    ) -> ArtifactReference:
        """Store new bytes as one E2/E3 CAS object."""

        from io import BytesIO

        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError("content must be bytes-like")
        return self._store_stream(
            BytesIO(bytes(content)),
            original_name=original_name,
            artifact_class=artifact_class,
            locator=locator,
        )

    def put_file(
        self,
        source_path: str | os.PathLike[str],
        *,
        original_name: str | None = None,
        artifact_class: ArtifactClass,
        locator: str | None = None,
    ) -> ArtifactReference:
        """Copy a source file into CAS; the source is never moved or modified."""

        path = Path(source_path)
        name = original_name if original_name is not None else path.name
        with path.open("rb") as source:
            return self._store_stream(
                source,
                original_name=name,
                artifact_class=artifact_class,
                locator=locator,
            )

    def resolve_reference(
        self,
        reference: ArtifactReference | Mapping[str, Any],
        *,
        verify_integrity: bool = True,
    ) -> Path:
        """Resolve by H1 only; ``locator`` and filename never affect lookup."""

        ref = reference if isinstance(reference, ArtifactReference) else ArtifactReference.from_dict(reference)
        path = self.object_path(ref.h1)
        if verify_integrity:
            found_h1 = _hash_path_no_follow(path)
            if found_h1 != ref.h1:
                raise CorruptObjectError(
                    f"object at {path} hashes to {found_h1}, expected {ref.h1}"
                )
        else:
            try:
                metadata = path.lstat()
            except FileNotFoundError as exc:
                raise MissingObjectError(f"object not found: {path}") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise CorruptObjectError(f"canonical object path is not a regular file: {path}")
        return path

    def register_case_payload(
        self,
        *,
        case_identity: str,
        files: Mapping[str, ArtifactInput],
        physics_config: Mapping[str, Any],
        declared_physics_config_hash: str,
        completion_status: str,
        recorded_integrity: Mapping[str, str] | None,
        required_member_manifest: Mapping[str, str] | None = None,
    ) -> CasePayloadRecord:
        """Store members and index one case observation under its config hash.

        ``recorded_integrity`` is the capture's recorded path-to-H1 map.  It is
        retained as supplied so an integrity mismatch can be reported by the
        engineering reuse assessment instead of being silently repaired.

        ``required_member_manifest`` is an independent capture-contract input,
        not inferred from ``files``.  Omitting it preserves the observation but
        makes it ineligible for engineering reuse.
        """

        if not isinstance(case_identity, str) or not case_identity:
            raise ValueError("case_identity must be a non-empty string")
        if not isinstance(completion_status, str) or not completion_status:
            raise ValueError("completion_status must be a non-empty string")
        if not isinstance(physics_config, Mapping):
            raise TypeError("physics_config must be a mapping")
        physics_hash = compute_physics_config_hash(physics_config)
        declared_hash = _validated_h1(declared_physics_config_hash)
        if not isinstance(files, Mapping) or not files:
            raise ValueError("files must contain at least one case member")

        references: dict[str, ArtifactReference] = {}
        for raw_relative_path, artifact_input in files.items():
            relative_path = _normalized_relative_path(raw_relative_path)
            if relative_path in references:
                raise ValueError(f"duplicate normalized case member path: {relative_path!r}")
            if not isinstance(artifact_input, ArtifactInput):
                raise TypeError("each files value must be an ArtifactInput")
            artifact_class = _validated_artifact_class(artifact_input.artifact_class)
            source = artifact_input.source
            original_name = artifact_input.original_name or PurePosixPath(relative_path).name
            locator = artifact_input.locator
            if isinstance(source, ArtifactReference):
                # Reference reuse never copies bytes and still resolves via H1.
                self.resolve_reference(source, verify_integrity=True)
                reference = ArtifactReference(
                    h1=source.h1,
                    original_name=original_name,
                    artifact_class=artifact_class,
                    size_bytes=source.size_bytes,
                    locator=locator,
                )
            elif isinstance(source, (bytes, bytearray, memoryview)):
                reference = self.put_bytes(
                    source,
                    original_name=original_name,
                    artifact_class=artifact_class,
                    locator=locator,
                )
            else:
                reference = self.put_file(
                    source,
                    original_name=original_name,
                    artifact_class=artifact_class,
                    locator=locator,
                )
            references[relative_path] = reference

        payload_hash = compute_case_payload_hash(references)
        if recorded_integrity is None:
            normalized_integrity = None
        else:
            normalized_integrity = {
                _normalized_relative_path(path): _validated_h1(h1)
                for path, h1 in recorded_integrity.items()
            }
        normalized_required_manifest = _normalized_member_manifest(required_member_manifest)

        record_data = {
            "case_identity": case_identity,
            "case_payload_hash": payload_hash,
            "physics_config_hash": physics_hash,
            "declared_physics_config_hash": declared_hash,
            "completion_status": completion_status,
            "files": [
                {"relative_path": relative_path, "reference": references[relative_path].to_dict()}
                for relative_path in sorted(references)
            ],
            "recorded_integrity": normalized_integrity,
            "required_member_manifest": normalized_required_manifest,
        }
        self._append_index_record(physics_hash, record_data)
        return self._record_from_dict(record_data)

    def _index_path(self, physics_config_hash: str) -> Path:
        physics_config_hash = _validated_h1(physics_config_hash)
        return self.index_root / physics_config_hash[:2] / f"{physics_config_hash}.json"

    def _lock_path(self, physics_config_hash: str) -> Path:
        physics_config_hash = _validated_h1(physics_config_hash)
        return self.locks_root / f"{physics_config_hash}.lock"

    def _append_index_record(self, physics_config_hash: str, record: Mapping[str, Any]) -> None:
        index_path = self._index_path(physics_config_hash)
        lock_path = self._lock_path(physics_config_hash)
        _mkdir_durable(index_path.parent)
        _mkdir_durable(lock_path.parent)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if index_path.exists():
                try:
                    current = json.loads(index_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ArtifactStoreError(f"cannot read config index {index_path}: {exc}") from exc
                if (
                    not isinstance(current, dict)
                    or current.get("schema_version") != _INDEX_SCHEMA
                    or current.get("physics_config_hash") != physics_config_hash
                    or not isinstance(current.get("records"), list)
                ):
                    raise ArtifactStoreError(f"invalid config index structure: {index_path}")
            else:
                current = {
                    "schema_version": _INDEX_SCHEMA,
                    "physics_config_hash": physics_config_hash,
                    "records": [],
                }
            current["records"].append(dict(record))
            self._atomic_write_json(index_path, current)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    @staticmethod
    def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
        fd, temporary_name = tempfile.mkstemp(prefix=".index-", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as staged:
                staged.write(canonical_json_bytes(value))
                staged.write(b"\n")
                staged.flush()
                os.fsync(staged.fileno())
            os.chmod(temporary_path, 0o644)
            os.replace(temporary_path, path)
            _fsync_directory(path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _record_from_dict(value: Mapping[str, Any]) -> CasePayloadRecord:
        try:
            files = {
                _normalized_relative_path(item["relative_path"]): ArtifactReference.from_dict(item["reference"])
                for item in value["files"]
            }
            recorded_integrity = value["recorded_integrity"]
            if recorded_integrity is not None:
                recorded_integrity = {
                    _normalized_relative_path(path): _validated_h1(h1)
                    for path, h1 in recorded_integrity.items()
                }
            required_member_manifest = _normalized_member_manifest(value.get("required_member_manifest"))
            return CasePayloadRecord(
                case_identity=value["case_identity"],
                case_payload_hash=_validated_h1(value["case_payload_hash"]),
                physics_config_hash=_validated_h1(value["physics_config_hash"]),
                declared_physics_config_hash=_validated_h1(value["declared_physics_config_hash"]),
                completion_status=value["completion_status"],
                files=files,
                recorded_integrity=recorded_integrity,
                required_member_manifest=required_member_manifest,
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ArtifactStoreError(f"invalid case record: {exc}") from exc

    def query_cases(self, physics_config_hash: str) -> tuple[CasePayloadRecord, ...]:
        """Return every indexed case identity for one physics-config hash."""

        index_path = self._index_path(physics_config_hash)
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ()
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactStoreError(f"cannot read config index {index_path}: {exc}") from exc
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != _INDEX_SCHEMA
            or raw.get("physics_config_hash") != physics_config_hash
            or not isinstance(raw.get("records"), list)
        ):
            raise ArtifactStoreError(f"invalid config index structure: {index_path}")
        return tuple(self._record_from_dict(item) for item in raw["records"])

    def check_engineering_reuse(
        self,
        record: CasePayloadRecord,
        *,
        expected_physics_config_hash: str,
        expected_member_manifest: Mapping[str, str] | None = None,
    ) -> ReuseAssessment:
        """Evaluate completion, full byte integrity, config, and member contract."""

        expected_hash = _validated_h1(expected_physics_config_hash)
        reasons: list[str] = []
        completion_match = record.completion_status == "complete"
        config_match = (
            record.physics_config_hash == expected_hash
            and record.declared_physics_config_hash == expected_hash
        )
        if not completion_match:
            reasons.append("completion_status_not_complete")
        if not config_match:
            reasons.append("physics_config_hash_mismatch")

        expected_members = _normalized_member_manifest(expected_member_manifest)
        actual_members = {
            path: reference.artifact_class for path, reference in record.files.items()
        }
        member_manifest_match = (
            record.required_member_manifest is not None
            and expected_members is not None
            and record.required_member_manifest == expected_members
        )
        actual_members_match = expected_members is not None and actual_members == expected_members
        if record.required_member_manifest is None or expected_members is None:
            reasons.append("required_member_manifest_missing")
        elif not member_manifest_match:
            reasons.append("required_member_manifest_mismatch")
        if expected_members is not None and not actual_members_match:
            reasons.append("actual_member_set_or_role_mismatch")

        expected_integrity = {path: reference.h1 for path, reference in record.files.items()}
        byte_integrity_match = record.recorded_integrity == expected_integrity
        if not byte_integrity_match:
            reasons.append("recorded_integrity_mismatch_or_missing")

        try:
            computed_payload_hash = compute_case_payload_hash(record.files)
            if computed_payload_hash != record.case_payload_hash:
                byte_integrity_match = False
                reasons.append("case_payload_hash_mismatch")
            for reference in record.files.values():
                self.resolve_reference(reference, verify_integrity=True)
        except (ArtifactStoreError, OSError, ValueError) as exc:
            byte_integrity_match = False
            reasons.append(f"object_integrity_failure:{type(exc).__name__}")

        eligible = (
            completion_match
            and byte_integrity_match
            and config_match
            and member_manifest_match
            and actual_members_match
        )
        return ReuseAssessment(
            eligible=eligible,
            completion_match=completion_match,
            byte_integrity_match=byte_integrity_match,
            config_match=config_match,
            member_manifest_match=member_manifest_match,
            actual_members_match=actual_members_match,
            reasons=tuple(reasons),
        )

    def is_engineering_reusable(
        self,
        record: CasePayloadRecord,
        *,
        expected_physics_config_hash: str,
        expected_member_manifest: Mapping[str, str] | None = None,
    ) -> bool:
        """Convenience boolean for the R1.1 engineering predicate."""

        return self.check_engineering_reuse(
            record,
            expected_physics_config_hash=expected_physics_config_hash,
            expected_member_manifest=expected_member_manifest,
        ).eligible

    def reuse_certificate(
        self,
        record: CasePayloadRecord,
        *,
        expected_physics_config_hash: str,
        expected_member_manifest: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return a compact R4.4-style engineering lookup/reuse record.

        This records the mechanical P1 checks only.  It does not establish or
        change a scientific evidence class, validation decision, or conclusion.
        """

        expected_hash = _validated_h1(expected_physics_config_hash)
        assessment = self.check_engineering_reuse(
            record,
            expected_physics_config_hash=expected_hash,
            expected_member_manifest=expected_member_manifest,
        )
        return {
            "schema_version": "research-artifact-reuse-check-1",
            "case_identity": record.case_identity,
            "physics_config_hash": expected_hash,
            "lookup_hit": True,
            "case_payload_hash": record.case_payload_hash,
            "declared_physics_config_hash": record.declared_physics_config_hash,
            "completion_status": record.completion_status,
            "engineering_reuse_check": assessment.to_dict(),
        }
