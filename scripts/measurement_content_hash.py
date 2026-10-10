#!/usr/bin/env python3
"""Canonical, streaming content fingerprints for captured measurement payloads.

H1 is the ordinary SHA-256 of a file's stored bytes. H2 is a schema-versioned
SHA-256 of explicitly selected measurement values. This module intentionally
does not infer physics-configuration changes from an H2 mismatch.

The built-in schemas cover the mechanical-steering dose-calibration case and
the current FlyGym online worker chain. JSONL schemas require an exact set of
known top-level fields. Included JSON values are type-tagged and encoded
recursively; only declared paths are omitted. NPZ members are loaded one key at
a time and fed to the hasher in bounded chunks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import zipfile
from typing import Any, Mapping, Sequence


HASH_FORMAT_VERSION = "measurement-content-hash-v1"
NPZ_BUFFER_ELEMENTS = 1 << 16
JSONL_CHUNK_BYTES = 1 << 20


class MeasurementHashError(ValueError):
    """Raised when a capture does not match its declared schema."""


@dataclass(frozen=True)
class FileSchema:
    kind: str
    schema_id: str
    include: tuple[str, ...]
    exclude: Mapping[str, str] = field(default_factory=dict)
    optional: tuple[str, ...] = ()
    nested_exclude: Mapping[str, str] = field(default_factory=dict)
    nested_allowed_children: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    nested_required_children: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    nested_value_types: Mapping[str, str] = field(default_factory=dict)
    nested_array_lengths: Mapping[str, int] = field(default_factory=dict)
    named_axis_numeric_maps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"npz", "jsonl"}:
            raise ValueError(f"unsupported file schema kind: {self.kind}")
        if set(self.include) & set(self.exclude):
            raise ValueError(f"fields both included and excluded in {self.schema_id}")
        if len(set(self.include)) != len(self.include):
            raise ValueError(f"duplicate included fields in {self.schema_id}")
        if len(set(self.named_axis_numeric_maps)) != len(self.named_axis_numeric_maps):
            raise ValueError(f"duplicate named-axis map path in {self.schema_id}")
        if set(self.nested_allowed_children) & set(self.named_axis_numeric_maps):
            raise ValueError(f"nested object path has conflicting declarations in {self.schema_id}")
        if (self.nested_allowed_children or self.nested_required_children or self.nested_value_types
                or self.nested_array_lengths or self.named_axis_numeric_maps) and self.kind != "jsonl":
            raise ValueError(f"nested JSON declarations require a JSONL schema: {self.schema_id}")
        for path, children in self.nested_allowed_children.items():
            if path.split(".", 1)[0] not in self.include:
                raise ValueError(f"nested object path is not below an included field in {self.schema_id}: {path}")
            if len(set(children)) != len(children):
                raise ValueError(f"duplicate allowed nested child in {self.schema_id}: {path}")
        for path, children in self.nested_required_children.items():
            if path.split(".", 1)[0] not in self.include:
                raise ValueError(f"required nested path is not below an included field in {self.schema_id}: {path}")
            allowed = self.nested_allowed_children.get(path)
            if allowed is None or not set(children).issubset(allowed):
                raise ValueError(f"required nested children must be in the allowlist in {self.schema_id}: {path}")
            if len(set(children)) != len(children):
                raise ValueError(f"duplicate required nested child in {self.schema_id}: {path}")
        valid_value_types = {"array", "boolean", "null", "number", "object", "string"}
        for path, value_type in self.nested_value_types.items():
            if path.split(".", 1)[0] not in self.include:
                raise ValueError(f"nested type path is not below an included field in {self.schema_id}: {path}")
            if value_type not in valid_value_types:
                raise ValueError(f"unsupported nested JSON type {value_type!r} in {self.schema_id}: {path}")
        for path, length in self.nested_array_lengths.items():
            if path.split(".", 1)[0] not in self.include or length < 0:
                raise ValueError(f"invalid nested array length declaration in {self.schema_id}: {path}")
            if self.nested_value_types.get(path) != "array":
                raise ValueError(f"nested array length requires an array type declaration in {self.schema_id}: {path}")
        for path in self.named_axis_numeric_maps:
            if path.split(".", 1)[0] not in self.include:
                raise ValueError(f"named-axis map is not below an included field in {self.schema_id}: {path}")

    @property
    def declared(self) -> frozenset[str]:
        return frozenset(self.include) | frozenset(self.exclude)

    @property
    def nested_optional_children(self) -> Mapping[str, tuple[str, ...]]:
        return {
            path: tuple(child for child in children if child not in set(self.nested_required_children.get(path, ())))
            for path, children in self.nested_allowed_children.items()
        }


@dataclass(frozen=True)
class CaseSchema:
    schema_id: str
    payloads: Mapping[str, FileSchema]
    metadata_paths: tuple[str, ...]
    debug_npz_dir: str | None = None
    debug_npz_schema: FileSchema | None = None
    strict_tree: bool = True
    debug_npz_expected_frames: int | None = None
    debug_npz_initial_required: bool = False

    def __post_init__(self) -> None:
        if self.debug_npz_dir is None:
            if self.debug_npz_schema is not None or self.debug_npz_expected_frames is not None or self.debug_npz_initial_required:
                raise ValueError("debug NPZ expectations require a declared debug_npz_dir")
            return
        if self.debug_npz_schema is None or self.debug_npz_expected_frames is None:
            raise ValueError("dynamic debug NPZ family requires a schema and explicit expected frame count")
        if self.debug_npz_expected_frames < 0:
            raise ValueError("debug NPZ expected frame count cannot be negative")
        if self.debug_npz_expected_frames == 0 and not self.debug_npz_initial_required:
            raise ValueError("debug NPZ family cannot have an empty expected-member contract")


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    role: str
    size_bytes: int
    sha256: str
    content_sha256: str | None = None
    schema_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path,
            "role": self.role,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if self.content_sha256 is not None:
            result["object_content_hash"] = self.content_sha256
        if self.schema_id is not None:
            result["schema_id"] = self.schema_id
        return result


@dataclass(frozen=True)
class CaseFingerprint:
    schema_id: str
    root: Path = field(repr=False)
    case_h1: str
    case_h2: str
    payload_files: tuple[FileFingerprint, ...]
    metadata_files: tuple[FileFingerprint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "case_h1": self.case_h1,
            "case_h2": self.case_h2,
            "payload_files": [item.to_dict() for item in self.payload_files],
            "metadata_files": [item.to_dict() for item in self.metadata_files],
        }


@dataclass(frozen=True)
class ComparisonReport:
    schema_id: str
    status: str
    reference: CaseFingerprint
    candidate: CaseFingerprint
    payload_differences: tuple[dict[str, Any], ...]
    provenance_differences: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "status": self.status,
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "payload_differences": list(self.payload_differences),
            "provenance_differences": list(self.provenance_differences),
            "interpretation": (
                "content_mismatch_unresolved does not identify a configuration change; "
                "locate the differing measurement before making that determination"
            ),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(JSONL_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _put_bytes(digest: Any, data: bytes) -> None:
    digest.update(struct.pack("<Q", len(data)))
    digest.update(data)


def _put_text(digest: Any, value: str) -> None:
    _put_bytes(digest, value.encode("utf-8"))


def _canonical_json_value(digest: Any, value: Any, context: str = "JSON value") -> None:
    """Feed a JSON value with unambiguous tags and exact parsed float bits."""
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B\x01" if value else b"B\x00")
    elif isinstance(value, int):
        digest.update(b"I")
        _put_text(digest, str(value))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise MeasurementHashError(f"non-finite JSON number in {context}")
        digest.update(b"F" + struct.pack("<d", value))
    elif isinstance(value, str):
        digest.update(b"S")
        _put_text(digest, value)
    elif isinstance(value, list):
        digest.update(b"L" + struct.pack("<Q", len(value)))
        for item in value:
            _canonical_json_value(digest, item, context)
    elif isinstance(value, dict):
        digest.update(b"O" + struct.pack("<Q", len(value)))
        for key in sorted(value, key=lambda item: item.encode("utf-8")):
            if not isinstance(key, str):
                raise MeasurementHashError(f"non-string JSON object key in {context}")
            _put_text(digest, key)
            _canonical_json_value(digest, value[key], context + "." + key)
    else:
        raise MeasurementHashError(f"unsupported JSON value {type(value).__name__} in {context}")


def _reject_json_constant(token: str) -> None:
    raise MeasurementHashError(f"non-standard JSON numeric token is not accepted: {token}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MeasurementHashError(f"duplicate JSON object field: {key}")
        result[key] = value
    return result


def _decode_json_line(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(
            line,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise MeasurementHashError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise MeasurementHashError(f"expected an object at {path}:{line_number}")
    return value


def _nested_exclusions(
    value: Any,
    path: str,
    exclusions: Mapping[str, str],
    seen: set[str],
) -> Any:
    """Copy a JSON tree while removing only the declared nested metadata paths."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            matched = _matching_exclusion(child_path, exclusions)
            if matched is not None:
                seen.add(matched)
                continue
            result[key] = _nested_exclusions(child, child_path, exclusions, seen)
        return result
    if isinstance(value, list):
        return [_nested_exclusions(item, f"{path}[]", exclusions, seen) for item in value]
    return value


def _matching_exclusion(path: str, exclusions: Mapping[str, str]) -> str | None:
    path_parts = path.split(".")
    for pattern in exclusions:
        if _path_pattern_matches(path, pattern):
            return pattern
    return None


def _path_pattern_matches(path: str, pattern: str) -> bool:
    path_parts, pattern_parts = path.split("."), pattern.split(".")
    return len(pattern_parts) == len(path_parts) and all(
        expected == "*" or expected == actual
        for expected, actual in zip(pattern_parts, path_parts)
    )


def _matching_nested_children(path: str, schema: FileSchema) -> tuple[str, ...] | None:
    for pattern, children in schema.nested_allowed_children.items():
        if _path_pattern_matches(path, pattern):
            return children
    return None


def _matching_nested_sequence(path: str, schema: FileSchema) -> tuple[str, ...] | None:
    for pattern, values in schema.nested_required_children.items():
        if _path_pattern_matches(path, pattern):
            return values
    return None


def _matching_nested_value_type(path: str, schema: FileSchema) -> str | None:
    for pattern, value_type in schema.nested_value_types.items():
        if _path_pattern_matches(path, pattern):
            return value_type
    return None


def _matching_nested_array_length(path: str, schema: FileSchema) -> int | None:
    for pattern, length in schema.nested_array_lengths.items():
        if _path_pattern_matches(path, pattern):
            return length
    return None


def _is_named_axis_numeric_map(path: str, schema: FileSchema) -> bool:
    return any(_path_pattern_matches(path, pattern) for pattern in schema.named_axis_numeric_maps)


def _validate_included_json_tree(value: Any, path: str, schema: FileSchema, context: str) -> None:
    """Fail closed on undeclared JSON objects below included measurement fields."""
    if _matching_exclusion(path, schema.nested_exclude) is not None:
        return
    value_type = _matching_nested_value_type(path, schema)
    if value_type is not None:
        matches = {
            "array": isinstance(value, list),
            "boolean": isinstance(value, bool),
            "null": value is None,
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "object": isinstance(value, dict),
            "string": isinstance(value, str),
        }
        if not matches[value_type]:
            raise MeasurementHashError(
                f"expected nested {value_type} in {context}:{path}; got {type(value).__name__}"
            )
        if value_type == "number" and isinstance(value, float) and not math.isfinite(value):
            raise MeasurementHashError(f"nested number must be finite in {context}:{path}")
    expected_object = _matching_nested_children(path, schema) is not None or _is_named_axis_numeric_map(path, schema)
    if expected_object and not isinstance(value, dict):
        raise MeasurementHashError(
            f"expected nested object in {context}:{path}; got {type(value).__name__}"
        )
    if isinstance(value, dict):
        if _is_named_axis_numeric_map(path, schema):
            for axis_name, axis_value in value.items():
                if isinstance(axis_value, bool) or not isinstance(axis_value, (int, float)):
                    raise MeasurementHashError(
                        f"named-axis map values must be numeric in {context}:{path}[{axis_name!r}]"
                    )
                if isinstance(axis_value, float) and not math.isfinite(axis_value):
                    raise MeasurementHashError(
                        f"named-axis map values must be finite in {context}:{path}[{axis_name!r}]"
                    )
            return
        allowed = _matching_nested_children(path, schema)
        if allowed is None:
            raise MeasurementHashError(
                f"undeclared nested measurement object in {context}:{path}"
            )
        unknown = set(value) - set(allowed)
        if unknown:
            raise MeasurementHashError(
                f"unknown nested capture fields in {context}:{path}: {sorted(unknown)}"
            )
        required = set(_matching_nested_sequence(path, schema) or ())
        missing = required - set(value)
        if missing:
            raise MeasurementHashError(
                f"missing required nested capture fields in {context}:{path}: {sorted(missing)}"
            )
        for name, child in value.items():
            _validate_included_json_tree(child, f"{path}.{name}", schema, context)
    elif isinstance(value, list):
        expected_length = _matching_nested_array_length(path, schema)
        if expected_length is not None and len(value) != expected_length:
            raise MeasurementHashError(
                f"expected nested array length {expected_length} in {context}:{path}; got {len(value)}"
            )
        for child in value:
            _validate_included_json_tree(child, f"{path}.[]", schema, context)


def hash_jsonl(path: str | Path, schema: FileSchema) -> str:
    """Hash included JSONL fields in row order; fail on unknown root fields."""
    if schema.kind != "jsonl":
        raise ValueError(f"{schema.schema_id} is not a JSONL schema")
    path = Path(path)
    digest = hashlib.sha256()
    digest.update(b"H2-JSONL\x00" + HASH_FORMAT_VERSION.encode("ascii") + b"\x00")
    _put_text(digest, schema.schema_id)
    expected = schema.declared
    required = expected - set(schema.optional)
    seen_nested: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise MeasurementHashError(f"blank JSONL row at {path}:{line_number}")
            row = _decode_json_line(line, path, line_number)
            keys = set(row)
            unknown = keys - expected
            missing = required - keys
            if unknown:
                raise MeasurementHashError(
                    f"unknown capture fields in {path}:{line_number}: {sorted(unknown)}"
                )
            if missing:
                raise MeasurementHashError(
                    f"missing declared capture fields in {path}:{line_number}: {sorted(missing)}"
                )
            digest.update(b"ROW" + struct.pack("<Q", row_count))
            for name in sorted(schema.include, key=lambda item: item.encode("utf-8")):
                if name not in row:
                    continue
                _validate_included_json_tree(row[name], name, schema, f"{path}:{line_number}")
                value = _nested_exclusions(row[name], name, schema.nested_exclude, seen_nested)
                _put_text(digest, name)
                _canonical_json_value(digest, value, f"{path}:{line_number}:{name}")
            row_count += 1
    missing_exclusions = set(schema.nested_exclude) - seen_nested
    if missing_exclusions:
        raise MeasurementHashError(
            f"declared nested exclusions were not observed in {path}: {sorted(missing_exclusions)}"
        )
    digest.update(b"END" + struct.pack("<Q", row_count))
    return digest.hexdigest()


def _canonical_array_bytes(digest: Any, array: Any, path: Path, key: str) -> None:
    """Hash one array in C index order using bounded dtype-preserving chunks."""
    import numpy as np

    dtype = array.dtype
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufcSU":
        raise MeasurementHashError(
            f"unsupported NPZ dtype {dtype!s} for {path}:{key}; object/structured data are forbidden"
        )
    canonical_dtype = dtype.newbyteorder("<") if dtype.itemsize > 1 else dtype
    digest.update(b"ARRAY")
    _put_text(digest, key)
    _put_text(digest, canonical_dtype.str)
    digest.update(struct.pack("<Q", array.ndim))
    for dimension in array.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    try:
        iterator = np.nditer(
            array,
            flags=["external_loop", "buffered"],
            op_flags=["readonly"],
            op_dtypes=[canonical_dtype],
            order="C",
            casting="equiv",
            buffersize=NPZ_BUFFER_ELEMENTS,
        )
        for block in iterator:
            raw = block.tobytes(order="C")
            digest.update(raw)
    except (TypeError, ValueError) as exc:
        raise MeasurementHashError(f"cannot normalize NPZ array {path}:{key}: {exc}") from exc
    digest.update(b"ENDARRAY")


def hash_npz(path: str | Path, schema: FileSchema) -> str:
    """Hash declared NPZ arrays by sorted key, loading one array at a time."""
    if schema.kind != "npz":
        raise ValueError(f"{schema.schema_id} is not an NPZ schema")
    import numpy as np

    path = Path(path)
    digest = hashlib.sha256()
    digest.update(b"H2-NPZ\x00" + HASH_FORMAT_VERSION.encode("ascii") + b"\x00")
    _put_text(digest, schema.schema_id)
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            unknown = keys - schema.declared
            missing = (set(schema.include) - set(schema.optional)) - keys
            missing_excluded = set(schema.exclude) - set(schema.optional) - keys
            if unknown:
                raise MeasurementHashError(f"unknown NPZ arrays in {path}: {sorted(unknown)}")
            if missing or missing_excluded:
                raise MeasurementHashError(
                    f"missing declared NPZ arrays in {path}: {sorted(missing | missing_excluded)}"
                )
            selected = sorted(set(schema.include) & keys, key=lambda item: item.encode("utf-8"))
            digest.update(struct.pack("<Q", len(selected)))
            for key in selected:
                array = archive[key]
                _canonical_array_bytes(digest, array, path, key)
                del array
    except MeasurementHashError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:  # type: ignore[name-defined]
        raise MeasurementHashError(f"cannot read NPZ {path}: {exc}") from exc
    return digest.hexdigest()


def _hash_case_paths(rows: Sequence[FileFingerprint], *, field_name: str) -> str:
    digest = hashlib.sha256()
    digest.update((field_name + "\x00" + HASH_FORMAT_VERSION + "\x00").encode("ascii"))
    for row in sorted(rows, key=lambda item: item.path.encode("utf-8")):
        _put_text(digest, row.path)
        value = row.sha256 if field_name == "case-h1" else row.content_sha256
        if value is None:
            raise MeasurementHashError(f"missing content hash for payload {row.path}")
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _iter_tree_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _resolve_payload_schemas(root: Path, schema: CaseSchema) -> dict[str, FileSchema]:
    payloads = dict(schema.payloads)
    if schema.debug_npz_dir is not None:
        debug_dir = root / schema.debug_npz_dir
        if not debug_dir.is_dir():
            raise MeasurementHashError(f"required measurement directory is missing: {debug_dir}")
        assert schema.debug_npz_schema is not None
        assert schema.debug_npz_expected_frames is not None
        expected_names = {
            f"frame-{frame_index:04d}.npz"
            for frame_index in range(schema.debug_npz_expected_frames)
        }
        if schema.debug_npz_initial_required:
            expected_names.add("initial.npz")
        entries = list(debug_dir.iterdir())
        non_files = [path.name for path in entries if not path.is_file()]
        if non_files:
            raise MeasurementHashError(
                f"unexpected non-file member in debug NPZ family {schema.debug_npz_dir}: {sorted(non_files)}"
            )
        actual_names = {path.name for path in entries}
        missing = expected_names - actual_names
        unexpected = actual_names - expected_names
        if missing:
            raise MeasurementHashError(
                f"required debug-state capture members are missing from {schema.debug_npz_dir}: {sorted(missing)}"
            )
        if unexpected:
            raise MeasurementHashError(
                f"unknown debug-state capture members in {schema.debug_npz_dir}: {sorted(unexpected)}"
            )
        for name in sorted(expected_names, key=lambda item: item.encode("utf-8")):
            rel = (debug_dir / name).relative_to(root).as_posix()
            payloads[rel] = schema.debug_npz_schema
    return payloads


def fingerprint_case(case_dir: str | Path, schema: CaseSchema) -> CaseFingerprint:
    """Fingerprint a fully declared case. Unknown files fail closed by default."""
    root = Path(case_dir).resolve(strict=True)
    if not root.is_dir():
        raise MeasurementHashError(f"case path is not a directory: {root}")
    payload_schemas = _resolve_payload_schemas(root, schema)
    allowed = set(payload_schemas) | set(schema.metadata_paths)
    actual = _iter_tree_files(root)
    if schema.strict_tree:
        unclassified = actual - allowed
        if unclassified:
            raise MeasurementHashError(f"unclassified case files: {sorted(unclassified)}")
    payload_records: list[FileFingerprint] = []
    for rel in sorted(payload_schemas, key=lambda item: item.encode("utf-8")):
        path = root / rel
        if not path.is_file():
            raise MeasurementHashError(f"required measurement payload is missing: {rel}")
        file_schema = payload_schemas[rel]
        content_hash = hash_npz(path, file_schema) if file_schema.kind == "npz" else hash_jsonl(path, file_schema)
        payload_records.append(FileFingerprint(
            path=rel,
            role="E2",
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
            content_sha256=content_hash,
            schema_id=file_schema.schema_id,
        ))
    metadata_records: list[FileFingerprint] = []
    for rel in sorted(schema.metadata_paths, key=lambda item: item.encode("utf-8")):
        path = root / rel
        if path.is_file():
            metadata_records.append(FileFingerprint(
                path=rel,
                role="E3",
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            ))
    return CaseFingerprint(
        schema_id=schema.schema_id,
        root=root,
        case_h1=_hash_case_paths(payload_records, field_name="case-h1"),
        case_h2=_hash_case_paths(payload_records, field_name="case-h2"),
        payload_files=tuple(payload_records),
        metadata_files=tuple(metadata_records),
    )


def _json_pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _value_summary(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 512:
        raw = value.encode("utf-8")
        return {"type": "string", "sha256": hashlib.sha256(raw).hexdigest(), "utf8_bytes": len(raw)}
    if isinstance(value, float) and not math.isfinite(value):
        return {"type": "non_finite_float64", "bits_le": struct.pack("<d", value).hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"type": "array" if isinstance(value, list) else "object", "sha256": hashlib.sha256(encoded).hexdigest(), "utf8_bytes": len(encoded)}


def _json_value_differences(before: Any, after: Any, pointer: str = "") -> list[dict[str, Any]]:
    if type(before) is not type(after):
        return [{"pointer": pointer or "/", "kind": "type_changed", "reference": type(before).__name__, "candidate": type(after).__name__}]
    if isinstance(before, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child = pointer + "/" + _json_pointer_escape(str(key))
            if key not in before:
                result.append({"pointer": child, "kind": "field_added", "candidate": _value_summary(after[key])})
            elif key not in after:
                result.append({"pointer": child, "kind": "field_removed", "reference": _value_summary(before[key])})
            else:
                result.extend(_json_value_differences(before[key], after[key], child))
        return result
    if isinstance(before, list):
        result = []
        common = min(len(before), len(after))
        for index in range(common):
            result.extend(_json_value_differences(before[index], after[index], pointer + f"/{index}"))
        if len(before) != len(after):
            result.append({"pointer": pointer or "/", "kind": "array_length_changed", "reference": len(before), "candidate": len(after)})
        return result
    if before != after or (isinstance(before, float) and struct.pack("<d", before) != struct.pack("<d", after)):
        return [{"pointer": pointer or "/", "kind": "value_changed", "reference": _value_summary(before), "candidate": _value_summary(after)}]
    return []


def _metadata_details(left: Path, right: Path, rel: str) -> dict[str, Any]:
    suffix = left.suffix.lower()
    if suffix == ".json":
        try:
            before = json.loads(left.read_text(encoding="utf-8"), parse_constant=_reject_json_constant, object_pairs_hook=_reject_duplicate_keys)
            after = json.loads(right.read_text(encoding="utf-8"), parse_constant=_reject_json_constant, object_pairs_hook=_reject_duplicate_keys)
        except Exception:
            pass
        else:
            differences = _json_value_differences(before, after)
            return {
                "path": rel,
                "kind": "json_fields_changed" if differences else "json_representation_changed",
                "items": differences,
            }
    if suffix == ".jsonl":
        items: list[dict[str, Any]] = []
        with left.open("r", encoding="utf-8") as a, right.open("r", encoding="utf-8") as b:
            line_number = 1
            while True:
                line_a, line_b = a.readline(), b.readline()
                if not line_a and not line_b:
                    break
                if not line_a or not line_b:
                    items.append({"line": line_number, "kind": "row_added_or_removed"})
                else:
                    try:
                        value_a = json.loads(line_a, parse_constant=_reject_json_constant, object_pairs_hook=_reject_duplicate_keys)
                        value_b = json.loads(line_b, parse_constant=_reject_json_constant, object_pairs_hook=_reject_duplicate_keys)
                    except Exception:
                        if line_a != line_b:
                            items.append({"line": line_number, "kind": "text_row_changed", "reference_sha256": hashlib.sha256(line_a.encode()).hexdigest(), "candidate_sha256": hashlib.sha256(line_b.encode()).hexdigest()})
                    else:
                        items.extend({"line": line_number, **item} for item in _json_value_differences(value_a, value_b))
                line_number += 1
        return {"path": rel, "kind": "jsonl_rows_changed" if items else "jsonl_representation_changed", "items": items}
    try:
        before_lines = left.read_text(encoding="utf-8").splitlines()
        after_lines = right.read_text(encoding="utf-8").splitlines()
    except (UnicodeError, OSError):
        return _binary_metadata_details(left, right, rel)
    items: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        items.append({
            "kind": "text_lines_changed",
            "reference_lines_1based": [i1 + 1, i2],
            "candidate_lines_1based": [j1 + 1, j2],
            "reference_line_sha256": [hashlib.sha256(line.encode("utf-8")).hexdigest() for line in before_lines[i1:i2]],
            "candidate_line_sha256": [hashlib.sha256(line.encode("utf-8")).hexdigest() for line in after_lines[j1:j2]],
            "reference_preview": [line[:240] for line in before_lines[i1:i2]],
            "candidate_preview": [line[:240] for line in after_lines[j1:j2]],
        })
    return {"path": rel, "kind": "text_lines_changed" if items else "text_representation_changed", "items": items}


def _binary_metadata_details(left: Path, right: Path, rel: str) -> dict[str, Any]:
    offset = 0
    first_difference: int | None = None
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            block_a, block_b = a.read(JSONL_CHUNK_BYTES), b.read(JSONL_CHUNK_BYTES)
            if not block_a and not block_b:
                break
            common = min(len(block_a), len(block_b))
            for index in range(common):
                if block_a[index] != block_b[index]:
                    first_difference = offset + index
                    break
            if first_difference is not None:
                break
            if len(block_a) != len(block_b):
                first_difference = offset + common
                break
            offset += common
    return {"path": rel, "kind": "binary_bytes_changed", "first_differing_byte_offset": first_difference}


def _json_pointer(path_parts: Sequence[str]) -> str:
    return "/" + "/".join(_json_pointer_escape(part) for part in path_parts)


def _values_at_excluded_path(value: Any, path: str) -> dict[str, Any]:
    """Resolve an exact dotted path with `*` object-key segments."""
    parts = path.split(".")
    found: dict[str, Any] = {}

    def walk(node: Any, index: int, actual: tuple[str, ...]) -> None:
        if index == len(parts):
            found[_json_pointer(actual)] = node
            return
        if not isinstance(node, dict):
            return
        part = parts[index]
        if part == "*":
            for key in sorted(node, key=lambda item: str(item).encode("utf-8")):
                walk(node[key], index + 1, actual + (str(key),))
        elif part in node:
            walk(node[part], index + 1, actual + (part,))

    walk(value, 0, ())
    return found


def _append_json_field_differences(
    items: list[dict[str, Any]],
    before: Any,
    after: Any,
    *,
    pointer: str,
    reason: str,
    line: int | None = None,
    reference_seq: Any = None,
    candidate_seq: Any = None,
) -> None:
    for difference in _json_value_differences(before, after, pointer):
        item = {"reason": reason, **difference}
        if line is not None:
            item["line"] = line
            item["reference_seq"] = _value_summary(reference_seq)
            item["candidate_seq"] = _value_summary(candidate_seq)
        items.append(item)


def _jsonl_alignment_field(schema: FileSchema) -> str:
    for name in ("seq", "tick"):
        if name in schema.include:
            return name
    raise MeasurementHashError(
        f"JSONL provenance schema {schema.schema_id} must include a stable seq or tick field"
    )


def _jsonl_identity_token(value: Any, *, path: Path, line_number: int, field_name: str) -> bytes:
    if type(value) is int:
        return b"I" + str(value).encode("ascii")
    if isinstance(value, str):
        return b"S" + value.encode("utf-8")
    raise MeasurementHashError(
        f"stable {field_name} identifier must be an integer or string at {path}:{line_number}"
    )


def _validate_provenance_row(row: Mapping[str, Any], path: Path, line_number: int, schema: FileSchema) -> None:
    keys = set(row)
    unknown = keys - schema.declared
    missing = (schema.declared - set(schema.optional)) - keys
    if unknown:
        raise MeasurementHashError(
            f"unknown capture fields in provenance comparison at {path}:{line_number}: {sorted(unknown)}"
        )
    if missing:
        raise MeasurementHashError(
            f"missing declared capture fields in provenance comparison at {path}:{line_number}: {sorted(missing)}"
        )


def _excluded_jsonl_values(row: Mapping[str, Any], schema: FileSchema) -> dict[str, tuple[Any, str]]:
    values: dict[str, tuple[Any, str]] = {}
    for name, reason in schema.exclude.items():
        if name in row:
            values[_json_pointer((name,))] = (row[name], reason)
    for pattern, reason in schema.nested_exclude.items():
        for pointer, value in _values_at_excluded_path(row, pattern).items():
            values[pointer] = (value, reason)
    return values


def _collect_excluded_jsonl_rows(
    path: Path, schema: FileSchema, alignment_field: str,
) -> dict[bytes, dict[str, Any]]:
    """Stream rows and retain only stable IDs plus explicitly excluded values."""
    result: dict[bytes, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise MeasurementHashError(f"blank JSONL row at {path}:{line_number}")
            row = _decode_json_line(line, path, line_number)
            _validate_provenance_row(row, path, line_number, schema)
            if alignment_field not in row:
                raise MeasurementHashError(
                    f"missing stable {alignment_field} identifier at {path}:{line_number}"
                )
            identifier = row[alignment_field]
            token = _jsonl_identity_token(
                identifier, path=path, line_number=line_number, field_name=alignment_field,
            )
            if token in result:
                raise MeasurementHashError(
                    f"duplicate stable {alignment_field} identifier {identifier!r} in {path}:{line_number}"
                )
            result[token] = {
                "identifier": identifier,
                "line": line_number,
                "excluded": _excluded_jsonl_values(row, schema),
            }
    return result


def _json_leaf_values(value: Any, pointer: str) -> list[tuple[str, Any]]:
    if isinstance(value, dict) and value:
        result: list[tuple[str, Any]] = []
        for key in sorted(value, key=lambda item: item.encode("utf-8")):
            result.extend(_json_leaf_values(value[key], pointer + "/" + _json_pointer_escape(key)))
        return result
    if isinstance(value, list) and value:
        result = []
        for index, child in enumerate(value):
            result.extend(_json_leaf_values(child, pointer + f"/{index}"))
        return result
    return [(pointer, value)]


def _append_jsonl_row_metadata(
    items: list[dict[str, Any]],
    row: Mapping[str, Any],
    *,
    added: bool,
    alignment_field: str,
) -> None:
    for pointer, (value, reason) in sorted(row["excluded"].items()):
        for leaf_pointer, leaf_value in _json_leaf_values(value, pointer):
            items.append({
                "pointer": leaf_pointer,
                "kind": "row_added" if added else "row_removed",
                "reference": None if added else _value_summary(leaf_value),
                "candidate": _value_summary(leaf_value) if added else None,
                "reason": reason,
                "line": row["line"],
                "reference_line": None if added else row["line"],
                "candidate_line": row["line"] if added else None,
                "reference_seq": None if added else _value_summary(row["identifier"]),
                "candidate_seq": _value_summary(row["identifier"]) if added else None,
                "alignment_field": alignment_field,
            })


def _append_aligned_jsonl_metadata_difference(
    items: list[dict[str, Any]],
    pointer: str,
    before: tuple[Any, str] | None,
    after: tuple[Any, str] | None,
    reference_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    alignment_field: str,
) -> None:
    context = {
        "line": reference_row["line"],
        "reference_line": reference_row["line"],
        "candidate_line": candidate_row["line"],
        "reference_seq": _value_summary(reference_row["identifier"]),
        "candidate_seq": _value_summary(candidate_row["identifier"]),
        "alignment_field": alignment_field,
    }
    if before is None or after is None:
        present = after if before is None else before
        assert present is not None
        reason = present[1]
        kind = "field_added" if before is None else "field_removed"
        for leaf_pointer, leaf_value in _json_leaf_values(present[0], pointer):
            items.append({
                "pointer": leaf_pointer,
                "kind": kind,
                "reference": None if before is None else _value_summary(leaf_value),
                "candidate": _value_summary(leaf_value) if before is None else None,
                "reason": reason,
                **context,
            })
        return
    differences = _json_value_differences(before[0], after[0], pointer)
    for difference in differences:
        items.append({"reason": before[1], **difference, **context})


def _compare_excluded_jsonl(left: Path, right: Path, rel: str, schema: FileSchema) -> dict[str, Any] | None:
    items: list[dict[str, Any]] = []
    alignment_field = _jsonl_alignment_field(schema)
    rows_a = _collect_excluded_jsonl_rows(left, schema, alignment_field)
    rows_b = _collect_excluded_jsonl_rows(right, schema, alignment_field)
    for token in sorted(set(rows_a) | set(rows_b)):
        row_a, row_b = rows_a.get(token), rows_b.get(token)
        if row_a is None:
            assert row_b is not None
            _append_jsonl_row_metadata(items, row_b, added=True, alignment_field=alignment_field)
            continue
        if row_b is None:
            _append_jsonl_row_metadata(items, row_a, added=False, alignment_field=alignment_field)
            continue
        excluded_a, excluded_b = row_a["excluded"], row_b["excluded"]
        for pointer in sorted(set(excluded_a) | set(excluded_b)):
            _append_aligned_jsonl_metadata_difference(
                items, pointer, excluded_a.get(pointer), excluded_b.get(pointer),
                row_a, row_b, alignment_field,
            )
    if not items:
        return None
    return {
        "role": "E2",
        "path": rel,
        "kind": "excluded_jsonl_metadata_changed",
        "items": items,
    }


def _json_from_npz_metadata(value: Any) -> Any | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value, parse_constant=_reject_json_constant, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, MeasurementHashError, UnicodeError):
        return None


def _npz_scalar_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"bytes_hex": value.hex(), "byte_count": len(value)}
    return _value_summary(value)


def _compare_excluded_npz(left: Path, right: Path, rel: str, schema: FileSchema) -> dict[str, Any] | None:
    import numpy as np

    items: list[dict[str, Any]] = []
    try:
        with np.load(left, allow_pickle=False) as archive_a, np.load(right, allow_pickle=False) as archive_b:
            keys_a, keys_b = set(archive_a.files), set(archive_b.files)
            for key, reason in schema.exclude.items():
                pointer = _json_pointer((key,))
                if key not in keys_a or key not in keys_b:
                    if (key in keys_a) != (key in keys_b):
                        items.append({
                            "array_key": key,
                            "pointer": pointer,
                            "kind": "excluded_array_added" if key in keys_b else "excluded_array_removed",
                            "reference": key in keys_a,
                            "candidate": key in keys_b,
                            "reason": reason,
                        })
                    continue
                array_a, array_b = archive_a[key], archive_b[key]
                json_metadata = key.endswith("_json") or key == "metadata_json"
                if (
                    json_metadata
                    and array_a.shape == () and array_b.shape == ()
                    and array_a.dtype.kind in "US" and array_b.dtype.kind in "US"
                ):
                    json_a = _json_from_npz_metadata(array_a.item())
                    json_b = _json_from_npz_metadata(array_b.item())
                    if json_a is not None and json_b is not None:
                        _append_json_field_differences(
                            items, json_a, json_b, pointer=pointer, reason=reason,
                        )
                        del array_a, array_b
                        continue
                if array_a.dtype != array_b.dtype or array_a.shape != array_b.shape:
                    items.append({
                        "array_key": key,
                        "pointer": pointer,
                        "kind": "excluded_array_schema_changed",
                        "reference": {"dtype": array_a.dtype.str, "shape": list(array_a.shape)},
                        "candidate": {"dtype": array_b.dtype.str, "shape": list(array_b.shape)},
                        "reason": reason,
                    })
                    del array_a, array_b
                    continue
                if array_a.size == 0:
                    del array_a, array_b
                    continue
                flat_a = array_a.ravel(order="C")
                flat_b = array_b.ravel(order="C")
                for flat_index in range(array_a.size):
                    scalar_a, scalar_b = flat_a[flat_index], flat_b[flat_index]
                    # Compare exact scalar storage bits, including signed zero and NaN payloads.
                    if np.asarray(scalar_a).tobytes() == np.asarray(scalar_b).tobytes():
                        continue
                    coordinates = np.unravel_index(flat_index, array_a.shape) if array_a.shape else ()
                    element_pointer = pointer + "".join(f"/{int(index)}" for index in coordinates)
                    items.append({
                        "array_key": key,
                        "pointer": element_pointer,
                        "kind": "excluded_array_value_changed",
                        "reference": _npz_scalar_value(scalar_a),
                        "candidate": _npz_scalar_value(scalar_b),
                        "reason": reason,
                    })
                del array_a, array_b, flat_a, flat_b
    except MeasurementHashError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise MeasurementHashError(f"cannot compare excluded NPZ metadata {rel}: {exc}") from exc
    if not items:
        return None
    return {
        "role": "E2",
        "path": rel,
        "kind": "excluded_npz_metadata_changed",
        "items": items,
    }


def compare_excluded_payload_provenance(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    schema: CaseSchema,
    *,
    changed_h1_paths: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Compare only schema-excluded fields in E2 files already known to have changed H1.

    This source-only helper does not recompute H1 or H2. It is useful when a
    prior fingerprint report already identifies byte-different E2 paths.
    """
    root_a = Path(reference_dir).resolve(strict=True)
    root_b = Path(candidate_dir).resolve(strict=True)
    schemas_a = _resolve_payload_schemas(root_a, schema)
    schemas_b = _resolve_payload_schemas(root_b, schema)
    result: list[dict[str, Any]] = []
    for rel in sorted(set(changed_h1_paths), key=lambda item: item.encode("utf-8")):
        file_schema = schemas_a.get(rel)
        if file_schema is None or schemas_b.get(rel) != file_schema:
            raise MeasurementHashError(f"unknown or inconsistent E2 schema path: {rel}")
        path_a, path_b = root_a / rel, root_b / rel
        if not path_a.is_file() or not path_b.is_file():
            continue
        if file_schema.kind == "jsonl":
            difference = _compare_excluded_jsonl(path_a, path_b, rel, file_schema)
        else:
            difference = _compare_excluded_npz(path_a, path_b, rel, file_schema)
        if difference is not None:
            result.append(difference)
    return tuple(result)


def _provenance_differences(reference: CaseFingerprint, candidate: CaseFingerprint, schema: CaseSchema) -> tuple[dict[str, Any], ...]:
    left = {item.path: item for item in reference.metadata_files}
    right = {item.path: item for item in candidate.metadata_files}
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for rel in sorted(set(left) | set(right), key=lambda item: item.encode("utf-8")):
        a, b = left.get(rel), right.get(rel)
        if a is None or b is None:
            by_identity[("E3", rel)] = {
                "role": "E3",
                "path": rel,
                "kind": "metadata_file_added" if a is None else "metadata_file_removed",
                "reference_sha256": a.sha256 if a else None,
                "candidate_sha256": b.sha256 if b else None,
            }
            continue
        if a.sha256 == b.sha256:
            continue
        detail = _metadata_details(reference.root / rel, candidate.root / rel, rel)
        detail.update({"role": "E3", "reference_sha256": a.sha256, "candidate_sha256": b.sha256})
        by_identity[("E3", rel)] = detail

    payload_a = {item.path: item for item in reference.payload_files}
    payload_b = {item.path: item for item in candidate.payload_files}
    changed_paths = [
        rel for rel in set(payload_a) & set(payload_b)
        if payload_a[rel].sha256 != payload_b[rel].sha256
    ]
    e2_differences = compare_excluded_payload_provenance(
        reference.root, candidate.root, schema, changed_h1_paths=changed_paths,
    )
    for detail in e2_differences:
        rel = detail["path"]
        detail["reference_h1"] = payload_a[rel].sha256
        detail["candidate_h1"] = payload_b[rel].sha256
        by_identity[("E2", rel)] = detail
    return tuple(
        by_identity[key]
        for key in sorted(by_identity, key=lambda item: (item[1].encode("utf-8"), item[0]))
    )


def compare_cases(reference_dir: str | Path, candidate_dir: str | Path, schema: CaseSchema) -> ComparisonReport:
    """Compare complete case H1/H2 and report excluded metadata differences."""
    reference = fingerprint_case(reference_dir, schema)
    candidate = fingerprint_case(candidate_dir, schema)
    left = {item.path: item for item in reference.payload_files}
    right = {item.path: item for item in candidate.payload_files}
    payload_differences: list[dict[str, Any]] = []
    for rel in sorted(set(left) | set(right), key=lambda item: item.encode("utf-8")):
        a, b = left.get(rel), right.get(rel)
        if a is None or b is None:
            payload_differences.append({"path": rel, "kind": "payload_file_added" if a is None else "payload_file_removed"})
        elif a.sha256 != b.sha256 or a.content_sha256 != b.content_sha256:
            payload_differences.append({
                "path": rel,
                "kind": "encoding_difference" if a.content_sha256 == b.content_sha256 else "content_difference",
                "reference_h1": a.sha256,
                "candidate_h1": b.sha256,
                "reference_h2": a.content_sha256,
                "candidate_h2": b.content_sha256,
            })
    if reference.case_h2 != candidate.case_h2:
        status = "content_mismatch_unresolved"
    elif reference.case_h1 != candidate.case_h1:
        status = "encoding_difference"
    else:
        status = "content_match"
    return ComparisonReport(
        schema_id=schema.schema_id,
        status=status,
        reference=reference,
        candidate=candidate,
        payload_differences=tuple(payload_differences),
        provenance_differences=_provenance_differences(reference, candidate, schema),
    )


def _npz_schema(name: str, include: Sequence[str], exclude: Mapping[str, str] | None = None) -> FileSchema:
    return FileSchema("npz", name, tuple(include), dict(exclude or {}))


TRAJECTORY_ARRAYS = (
    "time_s", "qpos", "qvel", "joint_angle_rad", "joint_velocity_rad_s",
    "body_xpos_mm", "body_xquat_wxyz", "body_xvel_mm_s", "body_xangvel_rad_s",
    "body_xipos_mm", "body_mass_native", "body_xfrc_applied_native", "qfrc_applied_native",
    "com_position_mm", "com_velocity_mm_s", "thorax_heading_xy", "thorax_heading_yaw_rad",
    "tarsus5_xpos_mm", "tarsus5_z_mm", "foot_contact_found", "foot_contact_force_local_native",
    "foot_contact_torque_local_native", "foot_contact_pos_mm", "foot_contact_normal_world",
    "foot_contact_tangent_world", "foot_contact_fz_up_native", "thorax_collision_geom_clearance_mm",
    "thorax_collision_clearance_min_mm", "position_actuator_ctrl_rad", "position_actuator_force_native",
    "adhesion_actuator_ctrl", "adhesion_actuator_force_native", "cpg_phase_rad", "cpg_magnitude",
    "descending_drive_left_right", "leg_stance_command", "root_qpos_write_count",
    "root_qvel_write_count", "root_external_wrench_native", "nonleg_ground_contact",
    "thorax_ground_contact",
)
CONTACT_ARRAYS = (
    "step_index", "time_s", "geom1_id", "geom2_id", "position_mm", "normal_world",
    "force_world_on_geom2_native", "force_world_on_fly_native", "is_ground_contact",
    "fly_body_segment", "leg_index",
)
EYE_ARRAYS = ("seq", "frame_time_s", "pattern_time_s", "left_columns", "right_columns")
VISUAL_DRIVE_ARRAYS = ("drive_values", "target_spike_counts", "target_indices", "time_s", "seq")
MUSCLE_ARRAYS = (
    "counts", "dense_indices", "body_ids", "names", "sides", "legs", "types", "mapped_groups",
    "mapped", "seq", "time_s", "leg_names", "muscle_types", "muscle_groups", "muscle_spike_counts",
    "muscle_rates_hz", "muscle_normalized_rates",
)
NEURAL_DIAGNOSTIC_ARRAYS = (
    "seq", "frame_time_s", "sample_time_s", "window_start_step_index", "window_end_step_index",
    "engine_next_step_index", "diagnostic_csr_indices", "diagnostic_neuron_ids", "post_final_tick_vm_mv",
    "post_final_tick_synaptic_g_state", "held_external_current_mV_per_ms", "held_external_probability",
    "post_final_tick_ref_until_absolute_step", "whole_network_vm_summary_row_for_frame",
    "whole_network_vm_summary_seq", "whole_network_vm_summary_sample_time_s",
    "whole_network_vm_summary_step_index", "whole_network_vm_min_mv", "whole_network_vm_max_mv",
    "whole_network_vm_fraction_below_minus_100_mv", "whole_network_vm_percentiles_mv",
    "whole_network_vm_percentile_levels",
)
DEBUG_STATE_ARRAYS = (
    "current_gpu", "probability_gpu", "v", "g", "ref_until", "ring", "spk", "spike_tick",
    "fixed_delivery_accumulator",
)

_trajectory = _npz_schema("flygym-trajectory-arrays-v1", TRAJECTORY_ARRAYS)
_contacts = _npz_schema("flygym-contact-arrays-v1", CONTACT_ARRAYS)
_eye = _npz_schema("flygym-eye-columns-arrays-v1", EYE_ARRAYS, {"schema_version": "capture schema label"})
_visual = _npz_schema("flygym-visual-drive-arrays-v1", VISUAL_DRIVE_ARRAYS, {
    "units": "unit label (physics configuration/provenance)",
    "schema_version": "capture schema label",
    "ordering": "target ordering description",
    "metadata_json": "embedded capture metadata",
})
_muscles = _npz_schema("flygym-muscle-channel-arrays-v1", MUSCLE_ARRAYS, {
    "membership_json": "mapping metadata duplicated by explicit identity arrays",
    "provenance_json": "capture provenance",
})
_neural = _npz_schema("flygym-neural-state-diagnostic-arrays-v1", NEURAL_DIAGNOSTIC_ARRAYS, {
    "row_sha256": "derived per-row integrity checksum",
    "metadata_json": "capture metadata",
})
_debug_state = _npz_schema("flygym-neural-state-snapshot-arrays-v1", DEBUG_STATE_ARRAYS, {
    "metadata_json": "capture metadata and path/checksum details",
})


MECHANICAL_TELEMETRY_SCHEMA = FileSchema(
    "jsonl",
    "mechanical-controller-telemetry-v1",
    include=(
        "P_left_right", "ap_or_hind_target_by_leg", "body_time_after_tick_s", "control_time_s",
        "cpg_phase_rad_by_leg", "differential_ap_d", "ramp_phase", "ramp_u", "target_d", "tick",
    ),
    exclude={
        "controller_metadata_mode": "metadata capture mode",
        "current_config_sha256": "configuration identity belongs to physics_config/provenance",
    },
    named_axis_numeric_maps=("ap_or_hind_target_by_leg",),
)


FLYGym_ONLINE_FIELDS_INCLUDE = (
    "body_action_end_s", "body_action_start_s", "body_command_applied_this_frame",
    "body_pose_state_sample_time_s", "body_pose_time_s", "decoder_inputs_only", "decoder_output",
    "decoder_output_queued_for_next_frame", "input_pose", "pose_derived_state_lag_s",
    "raw_worker_DNp15_L_rate_hz", "raw_worker_DNp15_R_rate_hz", "raw_worker_leg_power_P",
    "retinal_pattern_time_s", "seq", "source_body_pose", "velocity_physical_state_sample_time_s",
    "velocity_record_boundary_time_s", "virtual_eye_origins_m", "worker_native_bilateral_drive_not_applied",
    "worker_neural_window_end_s", "worker_neural_window_start_s",
)
FLYGym_ONLINE_FIELDS_EXCLUDE = {
    "body_pose_operation_time_s": "runtime timing; execution-policy metadata",
    "decoder_output_time_s": "runtime timing; execution-policy metadata",
    "retinal_column_hashes": "derived checksum; raw retinal columns are in worker/eye_columns.npz",
    "retinal_columns_saved_by_worker": "capture/provenance flag",
    "retinal_schedule_record": "stimulus schedule and repeated condition metadata",
    "stimulus_label_passed_to_decoder": "condition/input-authority metadata",
    "velocity_physical_state_sample_status": "capture status metadata",
    "worker_reply_sha256": "derived protocol-row checksum",
    "worker_request_sha256": "derived protocol-row checksum",
}
FLYGym_ONLINE_NESTED_ALLOWED_CHILDREN = {
    "body_command_applied_this_frame": (
        "P_control", "available_at_body_time_s", "body_action_end_s", "body_action_start_s",
        "body_frame_seq", "command_delay_frames", "held_soma_side", "initial_neutral_command",
        "signed_d", "source_neural_seq", "source_neural_window_end_s", "source_neural_window_start_s",
    ),
    "decoder_output": (
        "action_boundary", "baseline_corrected_opponent_difference_hz", "baseline_left_rate_hz",
        "baseline_mode", "baseline_right_rate_hz", "classification", "clipped_control_signal",
        "clipped_opponent_signal", "clipped_raw_difference_signal", "command_amplitude",
        "command_ema_baseline_corrected_opponent_difference_hz", "command_ema_opponent_difference_hz",
        "command_unclipped_control_signal", "continuous_raw_ema_opponent_difference_hz",
        "control_power", "control_power_left", "control_power_right", "denominator_hz",
        "dn_p15_left_rate_hz", "dn_p15_right_rate_hz", "dna_like_ipsilateral_aep_short",
        "dng_like_contralateral_extension", "ema_left_hz", "ema_left_rate_hz", "ema_right_hz",
        "ema_right_rate_hz", "gated_clipped_raw_difference_signal", "gated_unclipped_control_signal",
        "gated_unclipped_raw_difference_signal", "leg_power_left", "leg_power_right",
        "legacy_field_semantics", "normalization_hz", "p_deadband", "power_active", "power_gate_state",
        "power_gate_transition", "power_mapping", "power_raw", "power_threshold", "raw_ema_initialized",
        "raw_ema_left_hz", "raw_ema_right_hz", "raw_left_hz", "raw_leg_power",
        "raw_opponent_difference_hz", "raw_opponent_signal", "raw_right_hz", "schema_version",
        "soma_side", "static_gate_active", "tick_index", "unclipped_control_signal",
        "unclipped_raw_difference_signal", "ungated_clipped_control_signal", "ungated_opponent_signal",
        "zero_power_baseline_hold",
    ),
    "decoder_output.dna_like_ipsilateral_aep_short": (
        "action", "command_magnitude", "endpoint_pattern", "full_scale_endpoint_mm",
        "requested_endpoint_magnitude_mm", "soma_side", "target_side",
    ),
    "decoder_output.dng_like_contralateral_extension": (
        "action", "command_magnitude", "endpoint_pattern", "full_scale_endpoint_mm",
        "requested_endpoint_magnitude_mm", "soma_side", "target_side",
    ),
    "decoder_output.legacy_field_semantics": (
        "clipped_opponent_signal", "ema_left_rate_hz/ema_right_rate_hz",
        "raw_opponent_signal", "zero_power_baseline_hold",
    ),
    "decoder_output_queued_for_next_frame": (
        "P_control", "available_at_body_time_s", "held_soma_side", "signed_d", "source_neural_seq",
        "source_neural_window_end_s", "source_neural_window_start_s",
    ),
    "input_pose": (
        "angular_velocity_radps", "pitch_rad", "position_m", "quaternion_wxyz", "roll_rad",
        "velocity_mps", "yaw_rad",
    ),
    "source_body_pose": (
        "eye_camera_baseline_m", "eye_camera_left_origin_mm", "eye_camera_midpoint_mm",
        "eye_camera_right_origin_mm", "eye_velocity_flygym_world_mps", "head_frame",
        "thorax_angular_velocity_radps", "thorax_linear_velocity_mm_s", "thorax_position_mm",
        "thorax_quaternion_wxyz", "thorax_rotation_body_to_world",
    ),
    "virtual_eye_origins_m": ("left", "right"),
}
FLYGym_INPUT_POSE_CHILDREN = (
    "angular_velocity_radps", "pitch_rad", "position_m", "quaternion_wxyz", "roll_rad",
    "velocity_mps", "yaw_rad",
)
FLYGym_INPUT_POSE_VALUE_TYPES = {
    "input_pose.position_m": "array",
    "input_pose.position_m.[]": "number",
    "input_pose.quaternion_wxyz": "array",
    "input_pose.quaternion_wxyz.[]": "number",
    "input_pose.velocity_mps": "array",
    "input_pose.velocity_mps.[]": "number",
    "input_pose.angular_velocity_radps": "array",
    "input_pose.angular_velocity_radps.[]": "number",
    "input_pose.pitch_rad": "number",
    "input_pose.roll_rad": "number",
    "input_pose.yaw_rad": "number",
}
FLYGym_INPUT_POSE_ARRAY_LENGTHS = {
    "input_pose.position_m": 3,
    "input_pose.quaternion_wxyz": 4,
    "input_pose.velocity_mps": 3,
    "input_pose.angular_velocity_radps": 3,
}
FLYGym_SOURCE_POSE_CHILDREN = (
    "eye_camera_baseline_m", "eye_camera_left_origin_mm", "eye_camera_midpoint_mm",
    "eye_camera_right_origin_mm", "eye_velocity_flygym_world_mps", "head_frame",
    "thorax_angular_velocity_radps", "thorax_linear_velocity_mm_s", "thorax_position_mm",
    "thorax_quaternion_wxyz", "thorax_rotation_body_to_world",
)
FLYGym_SOURCE_POSE_VALUE_TYPES = {
    "source_body_pose.eye_camera_baseline_m": "number",
    "source_body_pose.eye_camera_left_origin_mm": "array",
    "source_body_pose.eye_camera_left_origin_mm.[]": "number",
    "source_body_pose.eye_camera_midpoint_mm": "array",
    "source_body_pose.eye_camera_midpoint_mm.[]": "number",
    "source_body_pose.eye_camera_right_origin_mm": "array",
    "source_body_pose.eye_camera_right_origin_mm.[]": "number",
    "source_body_pose.eye_velocity_flygym_world_mps": "array",
    "source_body_pose.eye_velocity_flygym_world_mps.[]": "number",
    "source_body_pose.head_frame": "string",
    "source_body_pose.thorax_angular_velocity_radps": "array",
    "source_body_pose.thorax_angular_velocity_radps.[]": "number",
    "source_body_pose.thorax_linear_velocity_mm_s": "array",
    "source_body_pose.thorax_linear_velocity_mm_s.[]": "number",
    "source_body_pose.thorax_position_mm": "array",
    "source_body_pose.thorax_position_mm.[]": "number",
    "source_body_pose.thorax_quaternion_wxyz": "array",
    "source_body_pose.thorax_quaternion_wxyz.[]": "number",
    "source_body_pose.thorax_rotation_body_to_world": "array",
    "source_body_pose.thorax_rotation_body_to_world.[]": "array",
    "source_body_pose.thorax_rotation_body_to_world.[].[]": "number",
}
FLYGym_SOURCE_POSE_ARRAY_LENGTHS = {
    "source_body_pose.eye_camera_left_origin_mm": 3,
    "source_body_pose.eye_camera_midpoint_mm": 3,
    "source_body_pose.eye_camera_right_origin_mm": 3,
    "source_body_pose.eye_velocity_flygym_world_mps": 3,
    "source_body_pose.thorax_angular_velocity_radps": 3,
    "source_body_pose.thorax_linear_velocity_mm_s": 3,
    "source_body_pose.thorax_position_mm": 3,
    "source_body_pose.thorax_quaternion_wxyz": 4,
    "source_body_pose.thorax_rotation_body_to_world": 3,
    "source_body_pose.thorax_rotation_body_to_world.[]": 3,
}
FLYGym_ONLINE_NESTED_REQUIRED_CHILDREN = {
    "body_command_applied_this_frame": (
        "P_control", "body_action_end_s", "body_action_start_s", "body_frame_seq",
        "command_delay_frames", "held_soma_side", "initial_neutral_command", "signed_d",
        "source_neural_seq", "source_neural_window_end_s",
    ),
    "decoder_output": (
        "action_boundary", "classification", "control_power", "legacy_field_semantics",
        "power_mapping", "schema_version",
    ),
    "decoder_output.legacy_field_semantics": (
        "clipped_opponent_signal", "ema_left_rate_hz/ema_right_rate_hz",
        "raw_opponent_signal", "zero_power_baseline_hold",
    ),
    "decoder_output_queued_for_next_frame": (
        "P_control", "available_at_body_time_s", "held_soma_side", "signed_d", "source_neural_seq",
        "source_neural_window_end_s", "source_neural_window_start_s",
    ),
    "input_pose": FLYGym_INPUT_POSE_CHILDREN,
    "source_body_pose": FLYGym_SOURCE_POSE_CHILDREN,
    "virtual_eye_origins_m": ("left", "right"),
}
FLYGym_ONLINE_SCHEMA = FileSchema(
    "jsonl", "flygym-online-frame-measurements-v1", FLYGym_ONLINE_FIELDS_INCLUDE,
    FLYGym_ONLINE_FIELDS_EXCLUDE,
    nested_allowed_children=FLYGym_ONLINE_NESTED_ALLOWED_CHILDREN,
    nested_required_children=FLYGym_ONLINE_NESTED_REQUIRED_CHILDREN,
    nested_value_types={
        **FLYGym_INPUT_POSE_VALUE_TYPES,
        **FLYGym_SOURCE_POSE_VALUE_TYPES,
        "virtual_eye_origins_m.left": "array",
        "virtual_eye_origins_m.left.[]": "number",
        "virtual_eye_origins_m.right": "array",
        "virtual_eye_origins_m.right.[]": "number",
    },
    nested_array_lengths={
        **FLYGym_INPUT_POSE_ARRAY_LENGTHS,
        **FLYGym_SOURCE_POSE_ARRAY_LENGTHS,
        "virtual_eye_origins_m.left": 3,
        "virtual_eye_origins_m.right": 3,
    },
)

FLYGym_CONTROL_FIELDS_EXCLUDE = {
    "stimulus_label_passed_to_controller": "condition/input-authority metadata",
}
FLYGym_CONTROL_NESTED_ALLOWED_CHILDREN = {
    "controller_telemetry": (
        "adhesion_on_by_leg", "clearance_cycle_gate_mm", "clearance_peak_mm",
        "commanded_intrinsic_frequency_hz_by_leg", "cpg_magnitude_by_leg", "cpg_phase_rad_by_leg",
        "descending_signal_lr", "engineering_margin_above_gate_mm", "hind_clearance",
        "hybrid_controller", "interpretation", "intrinsic_amplitude_target_by_leg", "neutral_hold",
        "side_active_lr", "stance_marker_world_z_mm", "tick_index",
    ),
    "controller_telemetry.hind_clearance": (
        "actual_contact_timing", "hind_orientation_constraint_retained", "mode",
        "native_adhesion_schedule_modified", "per_leg", "profile", "total_ik_delta_clip_count",
        "total_joint_box_clip_count", "vertical_clearance_retained",
    ),
    "controller_telemetry.hind_clearance.per_leg": ("lf", "lh", "lm", "rf", "rh", "rm"),
    "controller_telemetry.hind_clearance.per_leg.*": (
        "actual_ground_contact_observed_by_controller", "adhesion_on_native_schedule",
        "aep_pep_command_terms_mm", "applied", "hind_clearance_envelope",
        "hind_clearance_goal_world_z_mm", "hind_clearance_marker_world_z_mm",
        "hind_orientation_error_after_rad", "hind_orientation_error_before_rad", "hind_pose_priority",
        "ik_delta_clip_count", "joint_box_clip_count", "joint_delta_from_same_tick_native_target_rad",
        "native_aep_phase_rad", "native_endpoint_ap_delta_mm", "native_pep_phase_rad",
        "nominal_foot_body_ap_mm", "nominal_foot_world_xyz_mm", "phase_rad",
        "position_jacobian_rank_at_solved_pose", "position_jacobian_singular_values_mm_per_rad_at_solved_pose",
        "position_residual_after_pose_priority_mm", "position_residual_position_only_mm",
        "prescribed_ap_task_delta_mm", "reason", "solved_foot_body_ap_mm", "solved_foot_world_xyz_mm",
        "target_ap_delta_mm", "target_endpoint_ap_delta_mm", "target_foot_body_ap_mm",
        "target_foot_world_xyz_mm", "target_world_z_delta_mm", "world_z_clearance_projection_into_body_ap_mm",
    ),
    "controller_telemetry.hind_clearance.per_leg.*.aep_pep_command_terms_mm": (
        "dna02_aep_delta_mm", "dng13_aep_delta_mm", "dng13_pep_delta_mm",
    ),
    "controller_telemetry.hind_clearance.per_leg.*.hind_pose_priority": (
        "accepted_orientation_steps", "active_box_dof_count", "active_outward_bound_dof_indices_seen",
        "algorithm", "applied", "box_rad", "cpu_s", "engineering_box_clip_count",
        "fallback_to_exact_v3_solution", "helper_version", "line_search_trials",
        "maximum_absolute_offset_rad", "orientation_damping_rad", "orientation_error_after_deg",
        "orientation_error_after_rad", "orientation_error_before_deg", "orientation_error_before_rad",
        "orientation_gain_accepted", "orientation_gain_deg", "orientation_iterations_max",
        "orientation_step_clip_count", "parameters", "position_acceptance_slack_mm",
        "position_error_after_mm", "position_error_before_mm", "position_retraction_clip_count",
        "position_retraction_damping_mm", "position_retraction_steps", "position_svd_rank_by_iteration",
        "position_svd_singular_values_by_iteration", "reason", "supersedes_helper_sha256",
        "svd_relative_tolerance", "target_rotation_source",
    ),
    "controller_telemetry.hind_clearance.per_leg.*.hind_pose_priority.parameters": (
        "line_search_scales", "max_orientation_iterations", "minimum_orientation_gain_rad",
        "orientation_damping_rad", "orientation_step_limit_rad", "position_acceptance_slack_mm",
        "position_damping_mm", "position_retraction_iterations", "position_retraction_step_limit_rad",
        "svd_relative_tolerance",
    ),
    "controller_telemetry.hybrid_controller": (
        "leg_to_correct_retraction", "net_corrections", "retraction_correction",
        "stumbling_correction", "stumbling_mask",
    ),
}
FLYGym_CONTROL_NESTED_REQUIRED_CHILDREN = {
    "controller_telemetry": (
        "adhesion_on_by_leg", "clearance_cycle_gate_mm", "clearance_peak_mm",
        "commanded_intrinsic_frequency_hz_by_leg", "cpg_magnitude_by_leg", "cpg_phase_rad_by_leg",
        "descending_signal_lr", "engineering_margin_above_gate_mm", "hind_clearance",
        "hybrid_controller", "interpretation", "intrinsic_amplitude_target_by_leg", "neutral_hold",
        "side_active_lr", "stance_marker_world_z_mm", "tick_index",
    ),
    "controller_telemetry.hind_clearance.per_leg.*": ("applied", "phase_rad"),
    "controller_telemetry.hind_clearance.per_leg.*.aep_pep_command_terms_mm": (
        "dna02_aep_delta_mm", "dng13_aep_delta_mm", "dng13_pep_delta_mm",
    ),
    "controller_telemetry.hind_clearance.per_leg.*.hind_pose_priority.parameters": (
        "line_search_scales", "max_orientation_iterations", "minimum_orientation_gain_rad",
        "orientation_damping_rad", "orientation_step_limit_rad", "position_acceptance_slack_mm",
        "position_damping_mm", "position_retraction_iterations", "position_retraction_step_limit_rad",
        "svd_relative_tolerance",
    ),
}
FLYGym_CONTROL_SCHEMA = FileSchema(
    "jsonl",
    "flygym-control-tick-measurements-v1",
    (
        "P_control", "actual_AP_amplitude", "actual_descending_signal_lr", "ap_soma_side_from_signed_d",
        "applied_command_source_neural_seq", "body_pose_frame_time_s", "control_end_time_s", "control_time_s",
        "controller_telemetry", "cpg_phase_rad_after_tick", "held_soma_side", "native_sample_indices_halfopen",
        "native_samples_added", "signed_d", "source_visual_neural_seq_current", "thorax_heading_yaw_rad_after_tick",
        "tick", "tick_in_frame",
    ),
    FLYGym_CONTROL_FIELDS_EXCLUDE,
    nested_exclude={
        "controller_telemetry.interpretation": "free-text explanatory metadata",
        "controller_telemetry.hind_clearance.per_leg.*.hind_pose_priority.cpu_s": "non-action runtime timer",
    },
    nested_allowed_children=FLYGym_CONTROL_NESTED_ALLOWED_CHILDREN,
    nested_required_children=FLYGym_CONTROL_NESTED_REQUIRED_CHILDREN,
    named_axis_numeric_maps=("controller_telemetry.stance_marker_world_z_mm",),
)

FLYGym_WORKER_FIELDS_INCLUDE = (
    "body_time_s", "diagnostic_neuron_vm_mv_by_csr_index", "diagnostic_population_vm_mv_mean",
    "diagnostic_population_vm_sample_count", "diagnostic_vm_sample_offset_within_window_s",
    "diagnostic_vm_sample_time_s", "input_pose", "input_quaternion_wxyz", "left_descending_drive",
    "leg_power", "leg_steering", "neural_frame_id", "neural_window_end_s", "neural_window_end_step_index",
    "neural_window_start_step_index", "pattern_time_s", "population_rates_hz", "population_report_count",
    "population_spike_counts", "right_descending_drive", "seq", "time_s", "vision_frame_id",
    "visual_drive_abs_mean", "visual_drive_mean", "visual_drive_std",
)
FLYGym_WORKER_FIELDS_EXCLUDE = {
    "artifact_checksums": "derived checksums",
    "artifact_dir": "run-specific path",
    "debug_neural_state": "capture metadata; state arrays are separate NPZ payloads",
    "diagnostic_vm_sample_semantics": "explanatory metadata",
    "effective_backends": "runtime backend provenance",
    "eye_column_hashes": "derived checksums; raw retinal columns are in eye_columns.npz",
    "flyvis_target_indices_sha256": "derived checksum; target_indices are in visual_drive.npz",
    "flyvis_target_units": "unit label/provenance",
    "motor_output_source": "runtime source label",
    "neural_intervention": "intervention provenance",
    "neural_state_diagnostic": "capture bookkeeping; diagnostic arrays are in neural_state_diagnostic.npz",
    "protocol": "protocol label",
    "render_time_s": "runtime timing; execution-policy metadata",
    "source": "runtime component provenance",
    "stimulus_condition": "condition label",
    "stimulus_state": "stimulus schedule metadata",
    "type": "record kind label",
    "visual_drive_capture_row": "capture row locator and schema metadata; values are in visual_drive.npz",
    "visual_drive_hash": "derived checksum; raw values are in visual_drive.npz",
}
FLYGym_WORKER_SCHEMA = FileSchema(
    "jsonl", "flygym-worker-frame-measurements-v1", FLYGym_WORKER_FIELDS_INCLUDE,
    FLYGym_WORKER_FIELDS_EXCLUDE, optional=("debug_neural_state",),
    nested_allowed_children={
        "input_pose": FLYGym_INPUT_POSE_CHILDREN,
    },
    nested_required_children={"input_pose": FLYGym_INPUT_POSE_CHILDREN},
    nested_value_types=FLYGym_INPUT_POSE_VALUE_TYPES,
    nested_array_lengths=FLYGym_INPUT_POSE_ARRAY_LENGTHS,
    named_axis_numeric_maps=(
        "diagnostic_neuron_vm_mv_by_csr_index", "diagnostic_population_vm_mv_mean",
        "diagnostic_population_vm_sample_count", "population_rates_hz", "population_spike_counts",
    ),
)


_mechanical_metadata = (
    "analysis-summary.json", "array_order.json", "compiled_model.mjb", "compiled_model.xml",
    "diagnostics.png", "effective_config.json", "initial_state.json", "integrity.json", "manifest.json",
    "model_options.json", "run-record.json", "seed.json", "summary.json", "upstream_asset_hashes.json",
)
MECHANICAL_CASE_SCHEMA = CaseSchema(
    "mechanical-steering-dose-calibration-v1",
    {
        "trajectory.npz": _trajectory,
        "contacts.npz": _contacts,
        "controller_telemetry.jsonl": MECHANICAL_TELEMETRY_SCHEMA,
    },
    _mechanical_metadata,
)


_flygym_metadata = (
    "analysis/legacy-v2-prefix-audit-status.json", "analysis/pre-prefix-run-record.json",
    "analysis/prefix-audit-index.json", "analysis/prefix-audit.json", "array_order.json",
    "compiled_model.mjb", "compiled_model.xml", "diagnostics.png", "effective_config.json",
    "initial_state.json", "integrity.json", "manifest.json", "model_options.json", "online-integrity.json",
    "precontroller-array-order.json", "precontroller-identity.json", "precontroller-model-options.json",
    "run-record.json", "seed.json", "summary.json", "upstream_asset_hashes.json", "worker-protocol-stdout.jsonl",
    "window-summary.json", "worker-requests.jsonl", "worker-stderr.log", "worker/angular_stimulus_spec.json",
    "worker/integrity.json", "worker/muscle_channels_membership.json", "worker/provenance.json",
)
FLYGym_ONLINE_CASE_SCHEMA = CaseSchema(
    "flygym-online-worker-chain-v1",
    {
        "trajectory.npz": _trajectory,
        "contacts.npz": _contacts,
        "online-frames.jsonl": FLYGym_ONLINE_SCHEMA,
        "control-ticks.jsonl": FLYGym_CONTROL_SCHEMA,
        "worker/frames.jsonl": FLYGym_WORKER_SCHEMA,
        "worker/eye_columns.npz": _eye,
        "worker/visual_drive.npz": _visual,
        "worker/muscle_channels.npz": _muscles,
        "worker/neural_state_diagnostic.npz": _neural,
    },
    _flygym_metadata,
    debug_npz_dir="worker/debug-neural-state",
    debug_npz_schema=_debug_state,
    debug_npz_expected_frames=40,
    debug_npz_initial_required=True,
)
FLYGYM_ONLINE_CASE_SCHEMA = FLYGym_ONLINE_CASE_SCHEMA

SCHEMAS: Mapping[str, CaseSchema] = {
    MECHANICAL_CASE_SCHEMA.schema_id: MECHANICAL_CASE_SCHEMA,
    FLYGYM_ONLINE_CASE_SCHEMA.schema_id: FLYGYM_ONLINE_CASE_SCHEMA,
}


def _resolve_schema(value: str) -> CaseSchema:
    try:
        return SCHEMAS[value]
    except KeyError as exc:
        raise SystemExit(f"unknown case schema {value!r}; choose from: {', '.join(sorted(SCHEMAS))}") from exc


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fingerprint", "compare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("paths", nargs="+", type=Path)
        subparser.add_argument("--schema", required=True, choices=sorted(SCHEMAS))
    args = parser.parse_args(argv)
    schema = _resolve_schema(args.schema)
    if args.command == "fingerprint":
        if len(args.paths) != 1:
            parser.error("fingerprint expects one case directory")
        result: Any = fingerprint_case(args.paths[0], schema).to_dict()
    else:
        if len(args.paths) != 2:
            parser.error("compare expects reference and candidate case directories")
        result = compare_cases(args.paths[0], args.paths[1], schema).to_dict()
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
