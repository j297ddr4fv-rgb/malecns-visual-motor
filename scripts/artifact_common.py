"""Canonical artifact I/O for new code; historical hash formats stay separate.

JSON hashing is strict compact UTF-8; JSON reading retains Python's existing
parser behavior (including NaN tokens). Writers overwrite atomically using
private temporary files. Array hashing is a new versioned format and must
never replace an existing experiment's array/content hash without migration.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
from typing import Any, Iterable, Iterator


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash file bytes in 1 MiB blocks, following filesystem symlinks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Hash a C-contiguous bytes-like buffer; do not copy strided views."""
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("sha256_bytes requires bytes, bytearray, or memoryview")
    if isinstance(value, memoryview) and not value.c_contiguous:
        raise BufferError("sha256_bytes requires a C-contiguous memoryview")
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash sorted, compact UTF-8 JSON; reject nonfinite JSON numbers."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_array(value: Any) -> str:
    """New bit-exact array format: dtype, shape, LE bytes in C index order.

    Uses the project's existing NumPy installation only when called. Plain
    dtypes are supported; objects, structured records and subarrays are not.
    This format is independent of measurement_content_hash's historical H2.
    """
    import numpy as np

    if not isinstance(value, np.ndarray):
        raise TypeError("sha256_array requires a NumPy ndarray")
    if value.dtype.hasobject or value.dtype.fields or value.dtype.subdtype:
        raise TypeError("object, structured and subarray dtypes are unsupported")
    dtype = value.dtype.newbyteorder("<")
    header = json.dumps({"dtype": dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(b"artifact-common-array-v1\0")
    digest.update(struct.pack("<Q", len(header)))
    digest.update(header)
    with np.nditer(value, flags=["external_loop", "buffered", "zerosize_ok"],
                   op_flags=["readonly"], op_dtypes=[dtype], casting="equiv",
                   order="C", buffersize=65536) as chunks:
        for chunk in chunks:
            digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read UTF-8 JSON using Python's default parser; return any JSON value.

    Missing/empty/invalid files raise the standard filesystem/parser errors.
    Duplicate keys and NaN tokens retain Python json's default behavior.
    Domain-specific object/type/finite checks belong to callers.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically replace sorted, indented strict UTF-8 JSON plus newline.

    Create parents. New files use mode 0600; replacement retains target mode.
    Reject symlink targets. Publish by rename after fsync, then fsync the
    containing directory. A directory-fsync error after rename is reported
    even though the new target is already visible; it is not rolled back.
    """
    target = Path(path)
    if target.is_symlink():
        raise ValueError("write_json does not replace a symlink target")
    raw = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                     allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o600
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(raw)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        parent_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[Any]:
    """Stream UTF-8 records, rejecting blank lines; accept a final bare line."""
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {number}: {path}")
            yield json.loads(line)


def write_jsonl(path: str | os.PathLike[str], records: Iterable[Any]) -> None:
    """Stream strict compact JSONL to an atomic overwrite, never append.

    Every record ends with a newline. Parent creation, modes and durability
    match write_json, including rejecting symlink targets and reporting
    post-publication directory-fsync errors. A failed iterator/serialization
    leaves the target intact.
    """
    target = Path(path)
    if target.is_symlink():
        raise ValueError("write_jsonl does not replace a symlink target")
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o600
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        parent_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)


def file_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Record followed file's H1, logical bytes and four-digit octal mode.

    Detect ordinary replacement/modification while hashing. This is not a
    no-follow security primitive or a scientific integrity validator.
    """
    target = Path(path)
    before = target.stat()
    digest = sha256_file(target)
    after = target.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                   after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError(f"file changed while recording: {path}")
    return {"H1": digest, "size_bytes": after.st_size,
            "mode": format(stat.S_IMODE(after.st_mode), "04o")}


def utc_now_iso() -> str:
    """UTC timestamp at whole-second precision with explicit +00:00 zone."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
