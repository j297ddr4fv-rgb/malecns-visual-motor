#!/usr/bin/env python3
"""Inventory and plan verified relocation of redundant research artifacts.

The default command is read-only with respect to source files and the archive
volume.  It hashes every regular file with bounded memory, identifies
byte-identical files, and writes a reviewable plan under the implementation
output directory.  The separate ``execute-reviewed-plan`` command is provided
for a later, explicitly authorized invocation; it copies, verifies, records,
and only then replaces a source path with a symlink to an active survivor.
This module never routes runtime paths through the cold archive.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path("/data/artifacts/by-project/malecns-gpu")
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "engineering"
DEFAULT_OBJECT_STORE_ROOT = PROJECT_ROOT / "object-store"
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_ROOT / "storage-remediation-implementation-20261008-v1"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "migration"
DEFAULT_ARCHIVE_MOUNT = Path("/media/xvhang/CE1A-6493")
DEFAULT_DESTINATION_ROOT = DEFAULT_ARCHIVE_MOUNT / "archive/by-project/malecns-gpu/engineering"
DEFAULT_RECORDS_PATH = DEFAULT_OUTPUT_DIR / "relocations.json"
ARCHIVE_INVENTORY_NAME = "archive-inventory.json"
ARCHIVE_INVENTORY_SCHEMA = "malecns-gpu.archive-inventory.v1"
CHUNK_SIZE = 1024 * 1024
EXFAT_FORBIDDEN = set(':*?"<>|\\')
MANUAL_ARCHIVE_VERIFY_COMMAND = '''python3 - <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(".").resolve(strict=True)
inventory_path = root / "archive-inventory.json"
if inventory_path.is_symlink() or not inventory_path.is_file():
    raise SystemExit("archive inventory is missing or not a regular file")
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
errors = []
for item in inventory["items"]:
    try:
        relative = pathlib.PurePosixPath(item["relative_archive_path"])
        if relative.is_absolute() or relative.as_posix() != item["relative_archive_path"] or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("invalid archive-relative path")
        path = root.joinpath(*relative.parts)
        parent = root
        for component in relative.parts[:-1]:
            parent = parent / component
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("parent path is a symlink or not a directory")
        if path.is_symlink() or not path.is_file():
            raise ValueError("path is not a regular file")
        resolved = path.resolve(strict=True)
        if resolved != path or root not in resolved.parents:
            raise ValueError("path escapes package or is not a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["H1"]:
            raise ValueError("size or H1 mismatch")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(item["relative_archive_path"])
print("OK" if not errors else "FAILED: " + repr(errors))
sys.exit(bool(errors))
PY'''


class PlanError(RuntimeError):
    """An unsafe or incomplete plan operation."""


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _mount_info(path: Path) -> dict[str, str]:
    """Return the longest matching mount from Linux mountinfo."""
    target = _absolute(path)
    best: tuple[int, dict[str, str]] | None = None
    with open("/proc/self/mountinfo", "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split()
            try:
                separator = fields.index("-")
            except ValueError:
                continue
            mountpoint = Path(
                fields[4]
                .replace("\\040", " ")
                .replace("\\011", "\t")
                .replace("\\012", "\n")
                .replace("\\134", "\\")
            )
            if target != mountpoint and not _is_relative_to(target, mountpoint):
                continue
            item = {
                "mountpoint": os.fspath(mountpoint),
                "filesystem": fields[separator + 1],
                "source": fields[separator + 2],
                "mount_options": fields[5],
            }
            score = len(os.fspath(mountpoint))
            if best is None or score > best[0]:
                best = (score, item)
    if best is None:
        raise PlanError(f"no mounted filesystem found for {path}")
    return best[1]


def _validate_mounts(
    source_root: Path,
    archive_mount: Path,
    destination_root: Path,
    archive_previous_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_mount = _mount_info(source_root)
    archive_mount = _absolute(archive_mount)
    destination_root = _absolute(destination_root)
    if not _is_relative_to(destination_root, archive_mount):
        raise PlanError(f"destination root must be under archive mount: {destination_root}")
    if not archive_mount.is_dir():
        previous_fs = (archive_previous_snapshot or {}).get("filesystem", {})
        previous_archive = previous_fs.get("archive", {})
        if (
            previous_archive.get("mountpoint") != os.fspath(archive_mount)
            or previous_archive.get("filesystem", "").lower() != "exfat"
            or Path(previous_archive.get("source", "")).name != "sda1"
        ):
            raise PlanError(f"archive mount is unavailable and no matching verified snapshot was supplied: {archive_mount}")
        source_dev = os.stat(source_root).st_dev
        previous_source_dev = previous_fs.get("source_device_id")
        archive_dev = previous_fs.get("archive_device_id")
        if source_dev != previous_source_dev or source_dev == archive_dev:
            raise PlanError("offline archive snapshot does not match the current source device or cross-filesystem relationship")
        if source_mount["filesystem"].lower() != "ext4" or Path(source_mount["source"]).name != "nvme0n1p3":
            raise PlanError("active source is no longer on the expected /data ext4 filesystem")
        return {
            "source": source_mount,
            "archive": previous_archive,
            "cross_filesystem": True,
            "source_device_id": source_dev,
            "archive_device_id": archive_dev,
            "source_free_bytes_at_scan": os.statvfs(source_root).f_bavail * os.statvfs(source_root).f_frsize,
            "archive_free_bytes_at_scan": None,
            "archive_free_bytes_last_confirmed": previous_fs.get("archive_free_bytes_at_scan"),
            "archive_available_at_scan": False,
            "archive_snapshot_used_at_utc": archive_previous_snapshot.get("generated_at_utc"),
            "archive_snapshot_source": "prior_read_only_plan; must be rechecked before execution",
        }
    destination_mount = _mount_info(archive_mount)
    if destination_mount["mountpoint"] != os.fspath(archive_mount):
        raise PlanError(
            f"archive path is not itself a mountpoint: {archive_mount} "
            f"(resolved mount: {destination_mount['mountpoint']})"
        )
    if destination_mount["filesystem"].lower() != "exfat":
        raise PlanError(
            f"archive filesystem must be exFAT, found {destination_mount['filesystem']}"
        )
    if Path(destination_mount["source"]).name != "sda1":
        raise PlanError(
            f"archive device must be /dev/sda1, found {destination_mount['source']}"
        )
    src_dev = os.stat(source_root).st_dev
    dst_dev = os.stat(archive_mount).st_dev
    if src_dev == dst_dev:
        raise PlanError("archive and source are on the same filesystem")
    src_stat = os.statvfs(source_root)
    dst_stat = os.statvfs(archive_mount)
    return {
        "source": source_mount,
        "archive": destination_mount,
        "cross_filesystem": True,
        "source_device_id": src_dev,
        "archive_device_id": dst_dev,
        "source_free_bytes_at_scan": src_stat.f_bavail * src_stat.f_frsize,
        "archive_free_bytes_at_scan": dst_stat.f_bavail * dst_stat.f_frsize,
        "archive_free_bytes_last_confirmed": dst_stat.f_bavail * dst_stat.f_frsize,
        "archive_available_at_scan": True,
        "archive_snapshot_used_at_utc": None,
        "archive_snapshot_source": "live_mount_read_only_observation",
    }


def _excluded(path: Path, excludes: Iterable[Path]) -> bool:
    return any(path == excluded or _is_relative_to(path, excluded) for excluded in excludes)


def _mode(st_mode: int) -> str:
    return f"{stat.S_IMODE(st_mode):04o}"


def _allocated_bytes(st: os.stat_result) -> int:
    return int(getattr(st, "st_blocks", 0)) * 512


def _walk_tree(root: Path, excludes: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    entries: list[dict[str, Any]] = []
    names_by_parent: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    errors: list[str] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda entry: entry.name.casefold())
        except OSError as exc:
            errors.append(f"cannot_list_directory:{directory}:{type(exc).__name__}:{exc}")
            continue
        for child in children:
            path = Path(child.path)
            if _excluded(path, excludes):
                continue
            relative = path.relative_to(root).as_posix()
            parent_rel = path.parent.relative_to(root).as_posix()
            names_by_parent[parent_rel][child.name.casefold()].append(child.name)
            try:
                st = child.stat(follow_symlinks=False)
            except OSError as exc:
                errors.append(f"cannot_stat:{path}:{type(exc).__name__}:{exc}")
                entries.append({
                    "old_path": os.fspath(path), "relative_path": relative,
                    "kind": "unreadable", "scan_error": f"{type(exc).__name__}: {exc}",
                })
                continue
            if stat.S_ISDIR(st.st_mode):
                entries.append({
                    "old_path": os.fspath(path), "relative_path": relative,
                    "kind": "directory", "mode": _mode(st.st_mode),
                    "logical_bytes": 0, "allocated_bytes": _allocated_bytes(st),
                })
                stack.append(path)
                continue
            if stat.S_ISREG(st.st_mode):
                kind = "regular"
            elif stat.S_ISLNK(st.st_mode):
                kind = "symlink"
            else:
                kind = "nonregular"
            item: dict[str, Any] = {
                "old_path": os.fspath(path),
                "relative_path": relative,
                "kind": kind,
                "mode": _mode(st.st_mode),
                "logical_bytes": int(st.st_size),
                "allocated_bytes": _allocated_bytes(st),
                "device": int(st.st_dev),
                "inode": int(st.st_ino),
                "nlink": int(st.st_nlink),
                "mtime_ns": int(st.st_mtime_ns),
                "ctime_ns": int(st.st_ctime_ns),
            }
            if kind == "symlink":
                try:
                    item["symlink_target"] = os.readlink(path)
                except OSError as exc:
                    item["symlink_target_error"] = f"{type(exc).__name__}: {exc}"
            entries.append(item)

    conflicts: list[dict[str, str]] = []
    for parent, folded_names in names_by_parent.items():
        for folded, names in folded_names.items():
            distinct = sorted(set(names))
            if len(distinct) > 1:
                conflicts.append({
                    "parent_relative_path": parent,
                    "casefolded_name": folded,
                    "conflicting_names": distinct,
                })
    conflicts.sort(key=lambda row: (row["parent_relative_path"], row["casefolded_name"]))
    return entries, conflicts, errors


def _case_collision_prefixes(conflicts: list[dict[str, str]]) -> list[str]:
    prefixes: set[str] = set()
    for conflict in conflicts:
        parent = conflict["parent_relative_path"]
        for name in conflict["conflicting_names"]:
            prefixes.add(f"{parent}/{name}" if parent != "." else name)
    return sorted(prefixes)


def _path_issue_prefixes(entries: list[dict[str, Any]]) -> list[str]:
    prefixes: set[str] = set()
    for entry in entries:
        if entry.get("kind") not in {"regular", "symlink", "directory", "nonregular"}:
            continue
        rel = entry["relative_path"]
        parts = rel.split("/")
        current: list[str] = []
        for component in parts:
            current.append(component)
            if any(char in EXFAT_FORBIDDEN for char in component) or component.endswith((" ", ".")):
                prefixes.add("/".join(current))
    return sorted(prefixes)


def _affected_by_prefix(relative_path: str, prefixes: list[str]) -> bool:
    return any(relative_path == prefix or relative_path.startswith(prefix + "/") for prefix in prefixes)


def _hash_regular_file(item: dict[str, Any]) -> tuple[str | None, str | None]:
    path = Path(item["old_path"])
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return None, f"open_failed:{type(exc).__name__}:{exc}"
    try:
        before = os.fstat(fd)
        expected = (item["device"], item["inode"], item["logical_bytes"], item["mtime_ns"], item["ctime_ns"])
        actual = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if actual != expected or not stat.S_ISREG(before.st_mode):
            return None, "changed_before_hash"
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
        after = os.fstat(fd)
        actual_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if actual_after != actual:
            return None, "changed_during_hash"
        return digest.hexdigest(), None
    except OSError as exc:
        return None, f"read_failed:{type(exc).__name__}:{exc}"
    finally:
        os.close(fd)


def _open_file_snapshot(target_inodes: set[tuple[int, int]]) -> dict[str, Any]:
    """Find open descriptors and mapped references for target inodes via /proc."""
    opened: set[tuple[int, int]] = set()
    mapped: set[tuple[int, int]] = set()
    write_opened: set[tuple[int, int]] = set()
    write_mapped: set[tuple[int, int]] = set()
    unreadable_pids: list[int] = []
    proc = Path("/proc")
    try:
        pids = [entry.name for entry in os.scandir(proc) if entry.name.isdecimal()]
    except OSError as exc:
        return {
            "status": "unknown", "open_inodes": [], "unreadable_pid_count": None,
            "error": f"cannot_scan_proc:{type(exc).__name__}:{exc}",
        }
    for pid_text in pids:
        pid = int(pid_text)
        fd_dir = proc / pid_text / "fd"
        try:
            fds = list(os.scandir(fd_dir))
        except FileNotFoundError:
            continue
        except PermissionError:
            unreadable_pids.append(pid)
            fds = []
        except OSError:
            fds = []
        for fd_entry in fds:
            try:
                st = os.stat(fd_entry.path)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            key = (int(st.st_dev), int(st.st_ino))
            if key in target_inodes:
                opened.add(key)
                try:
                    with open(proc / pid_text / "fdinfo" / fd_entry.name, "rt", encoding="ascii") as fdinfo:
                        flags_value = next(
                            (line.split(":", 1)[1].strip() for line in fdinfo if line.startswith("flags:")),
                            None,
                        )
                    if flags_value is None:
                        unreadable_pids.append(pid)
                    elif int(flags_value, 8) & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                        write_opened.add(key)
                except (OSError, ValueError):
                    if pid not in unreadable_pids:
                        unreadable_pids.append(pid)
        maps_path = proc / pid_text / "maps"
        try:
            with open(maps_path, "rt", encoding="ascii", errors="replace") as maps_handle:
                for line in maps_handle:
                    fields = line.split(None, 5)
                    if len(fields) < 5 or not fields[4].isdecimal() or fields[4] == "0":
                        continue
                    try:
                        major_text, minor_text = fields[3].split(":", 1)
                        device = os.makedev(int(major_text, 16), int(minor_text, 16))
                        key = (int(device), int(fields[4]))
                    except (ValueError, OSError):
                        continue
                    if key in target_inodes:
                        mapped.add(key)
                        if "w" in fields[1]:
                            write_mapped.add(key)
        except FileNotFoundError:
            pass
        except PermissionError:
            if pid not in unreadable_pids:
                unreadable_pids.append(pid)
        except OSError:
            pass
    return {
        "status": "partial" if unreadable_pids else "complete",
        "open_inodes": [list(key) for key in sorted(opened | mapped)],
        "open_fd_inodes": [list(key) for key in sorted(opened)],
        "mapped_inodes": [list(key) for key in sorted(mapped)],
        "write_open_inodes": [list(key) for key in sorted(write_opened | write_mapped)],
        "write_open_fd_inodes": [list(key) for key in sorted(write_opened)],
        "write_mapped_inodes": [list(key) for key in sorted(write_mapped)],
        "unreadable_pid_count": len(unreadable_pids),
        "unreadable_pids": unreadable_pids,
        "method": "read-only /proc/<pid>/fd stat and /proc/<pid>/maps snapshot",
    }


def _destination_collision(destination_root: Path, relative_path: str) -> str | None:
    """Return an existing casefold-equivalent destination, without creating it."""
    current = destination_root
    components = relative_path.split("/")
    for index, component in enumerate(components):
        if not current.is_dir():
            return None
        try:
            matches = [entry.name for entry in os.scandir(current) if entry.name.casefold() == component.casefold()]
        except OSError:
            return None
        if matches:
            if len(matches) > 1:
                return os.fspath(current / sorted(matches)[0])
            match = sorted(matches)[0]
            current = current / match
            if index == len(components) - 1 and current.exists():
                return os.fspath(current)
            continue
        return None
    return os.fspath(current) if current.exists() or current.is_symlink() else None


def _queue_lock_observation(source_root: Path) -> dict[str, Any]:
    """Snapshot visible project locks/processes without claiming global P0 state."""
    observed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    lock_rows: list[dict[str, str]] = []
    lock_command = [
        "lslocks", "--json", "--noheadings", "--output", "COMMAND,PID,TYPE,MODE,PATH"
    ]
    try:
        lock_result = subprocess.run(
            lock_command, check=False, capture_output=True, text=True, timeout=10
        )
        if lock_result.returncode == 0:
            parsed = json.loads(lock_result.stdout or "{}")
            for row in parsed.get("locks", []):
                path = row.get("path") or row.get("PATH") or ""
                if path.startswith(os.fspath(source_root) + "/") or path == os.fspath(source_root):
                    lock_rows.append({str(key).lower(): str(value) for key, value in row.items()})
            lock_status = "complete"
        else:
            lock_status = f"failed_exit_{lock_result.returncode}"
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        lock_status = f"unavailable:{type(exc).__name__}:{exc}"

    workload_rows: list[dict[str, str]] = []
    try:
        process_result = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            check=False, capture_output=True, text=True, timeout=10,
        )
        if process_result.returncode == 0:
            for line in process_result.stdout.splitlines():
                fields = line.strip().split(None, 2)
                if len(fields) < 3:
                    continue
                pid, command, args = fields
                lowered = args.lower()
                if "plan_research_artifact_relocations.py" in lowered:
                    continue
                if any(token in lowered for token in ("run_flygym", "run_digital_fly", "malecns-gpu")):
                    workload_rows.append({"pid": pid, "command": command, "args": args[:500]})
            process_status = "complete"
        else:
            process_status = f"failed_exit_{process_result.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        process_status = f"unavailable:{type(exc).__name__}:{exc}"

    return {
        "observed_at_utc": observed_at,
        "scope": os.fspath(source_root),
        "method": "read-only lslocks path filter and ps command-line filter",
        "live_locks_under_project": lock_rows,
        "lock_snapshot_status": lock_status,
        "matching_project_workload_processes": workload_rows,
        "process_snapshot_status": process_status,
        "boundary": (
            "This is a point-in-time observation limited to lock paths under the project and visible process command lines. "
            "It does not establish that all machine-wide or out-of-scope P0 locks are absent."
        ),
    }


def _tree_signatures(root: Path, excludes: list[Path]) -> dict[str, tuple[Any, ...]]:
    entries, _, _ = _walk_tree(root, excludes)
    result: dict[str, tuple[Any, ...]] = {}
    for item in entries:
        if item.get("kind") == "directory":
            continue
        result[item["relative_path"]] = (
            item.get("kind"), item.get("device"), item.get("inode"),
            item.get("logical_bytes"), item.get("mtime_ns"), item.get("ctime_ns"),
        )
    return result


def create_plan(
    source_root: Path,
    destination_root: Path,
    archive_mount: Path,
    output_root: Path,
    object_store_root: Path = DEFAULT_OBJECT_STORE_ROOT,
    progress: Callable[[str], None] | None = None,
    validate_mounts: bool = True,
    archive_previous_snapshot_path: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise PlanError(f"source root is not a directory: {source_root}")
    destination_root = _absolute(destination_root)
    archive_mount = _absolute(archive_mount)
    output_root = _absolute(output_root)
    object_store_root = _absolute(object_store_root)
    excludes = sorted({output_root, object_store_root})
    if _is_relative_to(source_root, output_root) or _is_relative_to(source_root, object_store_root):
        raise PlanError("source root cannot be inside an excluded output/object-store tree")
    previous_archive_snapshot: dict[str, Any] | None = None
    if archive_previous_snapshot_path is not None:
        previous_archive_snapshot = _load_json(archive_previous_snapshot_path)
    if validate_mounts:
        mounts = _validate_mounts(source_root, archive_mount, destination_root, previous_archive_snapshot)
    else:
        mounts = {
            "source": _mount_info(source_root),
            "archive": _mount_info(archive_mount),
            "cross_filesystem": os.stat(source_root).st_dev != os.stat(archive_mount).st_dev,
            "source_device_id": os.stat(source_root).st_dev,
            "archive_device_id": os.stat(archive_mount).st_dev,
            "archive_available_at_scan": True,
            "archive_snapshot_used_at_utc": None,
            "archive_snapshot_source": "live_mount_read_only_observation",
        }

    entries, case_conflicts, walk_errors = _walk_tree(source_root, excludes)
    queue_locks = _queue_lock_observation(source_root)
    queue_locks["lock_marker_files_present"] = [
        {
            "path": item["old_path"],
            "logical_bytes": item.get("logical_bytes"),
            "mtime_ns": item.get("mtime_ns"),
        }
        for item in entries
        if item.get("kind") in {"regular", "symlink"}
        and (Path(item["old_path"]).name == "queue.lock" or Path(item["old_path"]).name.endswith(".lock"))
    ]
    case_prefixes = _case_collision_prefixes(case_conflicts)
    invalid_prefixes = _path_issue_prefixes(entries)
    initial_signatures = {
        item["relative_path"]: (
            item.get("kind"), item.get("device"), item.get("inode"),
            item.get("logical_bytes"), item.get("mtime_ns"), item.get("ctime_ns"),
        )
        for item in entries if item.get("kind") != "directory"
    }

    regular = [item for item in entries if item.get("kind") == "regular"]
    hash_errors: list[dict[str, str]] = []
    for index, item in enumerate(regular, start=1):
        h1, error = _hash_regular_file(item)
        item["h1_sha256"] = h1
        item["hash_status"] = "complete" if h1 else "failed"
        if error:
            item["hash_error"] = error
            hash_errors.append({"old_path": item["old_path"], "error": error})
        if progress and (index == len(regular) or index % 500 == 0):
            progress(f"hashed {index}/{len(regular)} regular files")

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in regular:
        if item.get("h1_sha256"):
            by_hash[item["h1_sha256"]].append(item)
    for item in regular:
        h1 = item.get("h1_sha256")
        if not h1:
            item["class"] = "unclassified_hash_failure"
        elif len(by_hash[h1]) > 1:
            item["class"] = "redundant"
        else:
            item["class"] = "unique"

    planned: list[dict[str, Any]] = []
    for h1, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda item: item["relative_path"].casefold())
        anchor = ordered[0]
        for item in ordered:
            alternatives = [other for other in ordered if other is not item]
            item["surviving_copy"] = anchor["old_path"] if item is not anchor else alternatives[0]["old_path"]
            item["surviving_copy_after_plan"] = anchor["old_path"]
            item["anchor_for_h1"] = anchor["old_path"]
            if item is anchor:
                item["planned_action"] = "retain_active_survivor"
                item["new_path"] = None
                item["relocation_candidate"] = False
                continue
            item["new_path"] = os.fspath(destination_root / item["relative_path"])
            item["relocation_candidate"] = True
            if _affected_by_prefix(item["relative_path"], case_prefixes):
                item["planned_action"] = "blocked_case_insensitive_destination_collision"
            elif _affected_by_prefix(item["relative_path"], invalid_prefixes):
                item["planned_action"] = "blocked_exfat_path_component"
            elif collision := _destination_collision(destination_root, item["relative_path"]):
                item["planned_action"] = "blocked_destination_already_exists"
                item["destination_collision"] = collision
            else:
                item["planned_action"] = "pending_open_file_check"
                planned.append(item)

    candidate_inodes = {(item["device"], item["inode"]) for item in planned}
    open_snapshot = _open_file_snapshot(candidate_inodes)
    open_inodes = {tuple(pair) for pair in open_snapshot.get("open_inodes", [])}
    write_open_inodes = {tuple(pair) for pair in open_snapshot.get("write_open_inodes", [])}
    for item in planned:
        inode_key = (item["device"], item["inode"])
        if inode_key in write_open_inodes:
            item["planned_action"] = "blocked_source_inode_known_writable_open"
            item["open_status"] = "known_writable_open"
        else:
            item["planned_action"] = "eligible_after_review_and_later_authorization"
            if inode_key in open_inodes:
                item["open_status"] = "known_open_or_mapped_reclaim_deferred"
            elif open_snapshot["status"] != "complete":
                item["open_status"] = "unknown_due_to_unreadable_processes_reclaim_uncertain"
            else:
                item["open_status"] = "not_observed_open"

    final_signatures = _tree_signatures(source_root, excludes)
    changed_paths = sorted(
        path for path in (set(initial_signatures) | set(final_signatures))
        if initial_signatures.get(path) != final_signatures.get(path)
    )
    tree_stable = not changed_paths and not walk_errors
    if not tree_stable:
        for item in entries:
            if item.get("planned_action") == "eligible_after_review_and_later_authorization":
                item["planned_action"] = "blocked_source_tree_changed_during_scan"

    # Estimate physical source allocation released after all planned path removals.
    planned_by_inode: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    all_regular_by_inode: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in regular:
        all_regular_by_inode[(item["device"], item["inode"])].append(item)
        if item.get("planned_action") == "eligible_after_review_and_later_authorization":
            planned_by_inode[(item["device"], item["inode"])].append(item)
    estimate_rows: list[dict[str, Any]] = []
    reclaim_lower_bound = 0
    reclaim_upper_bound = 0
    redundant_reclaim_lower_bound = 0
    redundant_reclaim_upper_bound = 0
    for inode_key, removed in planned_by_inode.items():
        sample = all_regular_by_inode[inode_key][0]
        # st_nlink includes links outside the scoped tree. A zero remaining
        # count is required before this inode's blocks can be reclaimed.
        remaining_links = sample["nlink"] - len(removed)
        lower_bound = 0
        upper_bound = 0
        reason = "retained_hardlink_in_or_outside_scope"
        if remaining_links <= 0:
            upper_bound = sample["allocated_bytes"]
            if inode_key in open_inodes:
                reason = "known_open_or_mapped_reference_defers_reclaim"
            elif open_snapshot["status"] != "complete":
                reason = "unreadable_processes_make_reclaim_uncertain"
            else:
                lower_bound = sample["allocated_bytes"]
                reason = "last_link_removed_and_not_observed_open"
        reclaim_lower_bound += lower_bound
        reclaim_upper_bound += upper_bound
        redundant_reclaim_lower_bound += lower_bound
        redundant_reclaim_upper_bound += upper_bound
        estimate_rows.append({
            "device": inode_key[0], "inode": inode_key[1],
            "planned_path_count": len(removed), "source_nlink": sample["nlink"],
            "remaining_link_estimate": max(0, remaining_links),
            "allocated_bytes": sample["allocated_bytes"],
            "estimated_reclaim_bytes_lower_bound": lower_bound,
            "estimated_reclaim_bytes_upper_bound_if_no_open_references": upper_bound,
            "reason": reason,
            "paths": [item["old_path"] for item in removed],
        })

    regular_count = len(regular)
    redundant = [item for item in regular if item.get("class") == "redundant"]
    unique = [item for item in regular if item.get("class") == "unique"]
    eligible = [item for item in entries if item.get("planned_action") == "eligible_after_review_and_later_authorization"]
    blocked = [item for item in entries if item.get("relocation_candidate") and item.get("planned_action") != "eligible_after_review_and_later_authorization"]
    logical_reduction = sum(int(item["logical_bytes"]) for item in eligible)
    gross_allocated_targeted = sum(int(item["allocated_bytes"]) for item in eligible)
    redundant_logical = sum(int(item["logical_bytes"]) for item in redundant)
    redundant_allocated = sum(int(item["allocated_bytes"]) for item in redundant)
    unique_logical = sum(int(item["logical_bytes"]) for item in unique)
    unique_allocated = sum(int(item["allocated_bytes"]) for item in unique)

    # Drop directory-only inventory records from the per-file deliverable while
    # retaining them for collision detection and stability checks.
    file_entries = [item for item in entries if item.get("kind") != "directory"]
    for item in file_entries:
        item.setdefault("h1_sha256", None)
        item.setdefault("class", "excluded_symlink" if item.get("kind") == "symlink" else "excluded_nonregular")
        item.setdefault("planned_action", "retain_excluded_entry")
        item["case_conflict"] = _affected_by_prefix(item["relative_path"], case_prefixes)
        item["exfat_path_issue"] = _affected_by_prefix(item["relative_path"], invalid_prefixes)
        item.setdefault("new_path", None)
        item.setdefault("surviving_copy", None)

    prior_destination_evidence: dict[str, Any] | None = None
    if previous_archive_snapshot is not None:
        previous_files = previous_archive_snapshot.get("files", [])
        previous_candidate_paths = {
            row.get("relative_path") for row in previous_files if row.get("relocation_candidate")
        }
        current_candidate_paths = {
            row.get("relative_path") for row in file_entries if row.get("relocation_candidate")
        }
        prior_collisions = [
            row for row in previous_files
            if row.get("planned_action") == "blocked_destination_already_exists"
        ]
        prior_destination_evidence = {
            "previous_report_generated_at_utc": previous_archive_snapshot.get("generated_at_utc"),
            "candidate_relative_paths_match_current": previous_candidate_paths == current_candidate_paths,
            "previous_candidate_path_collision_count": len(prior_collisions),
            "previous_scan_scope": previous_archive_snapshot.get("scope", {}).get("source_root"),
        }

    return {
        "schema": "malecns-gpu.research-artifact-relocation-plan.v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution_status": "dry_run_only_no_archive_writes_no_source_changes",
        "scope": {
            "source_root": os.fspath(source_root),
            "destination_root_proposed": os.fspath(destination_root),
            "archive_mount": os.fspath(archive_mount),
            "output_root_excluded": os.fspath(output_root),
            "object_store_root_excluded": os.fspath(object_store_root),
            "raw_dataset_path_excluded_by_scope": "/data/datasets/raw/malecns/v1.0/",
        },
        "filesystem": mounts,
        "hashing": {
            "algorithm": "SHA-256 over complete file bytes (H1)",
            "strategy": "stream every regular file in 1 MiB chunks; bounded memory; no size-only duplicate claims",
            "regular_files_seen": regular_count,
            "regular_files_hashed": sum(bool(item.get("h1_sha256")) for item in regular),
            "hash_failures": len(hash_errors),
            "hash_errors": hash_errors,
            "size_is_not_used_as_duplicate_proof": True,
        },
        "open_file_snapshot": open_snapshot,
        "queue_lock_observation": queue_locks,
        "tree_stability": {
            "stable_during_scan": tree_stable,
            "changed_path_count": len(changed_paths),
            "changed_paths_sample": changed_paths[:200],
            "walk_error_count": len(walk_errors),
            "walk_errors": walk_errors[:200],
        },
        "case_insensitive_destination_conflicts": case_conflicts,
        "archive_destination_path_check": {
            "current_status": "checked_read_only_current_scan" if mounts.get("archive_available_at_scan", True) else "not_checked_archive_offline",
            "source_tree_casefold_collision_groups": len(case_conflicts),
            "previous_live_archive_scan_evidence": prior_destination_evidence,
            "execution_must_recheck_exact_and_casefold_destination_paths": True,
        },
        "exfat_path_component_issues": invalid_prefixes,
        "summary": {
            "regular_files_U1": regular_count,
            "symlinks_U1": sum(item.get("kind") == "symlink" for item in file_entries),
            "nonregular_or_unreadable_U1": sum(item.get("kind") in {"nonregular", "unreadable"} for item in file_entries),
            "active_tree_regular_logical_bytes_U3": sum(int(item["logical_bytes"]) for item in regular),
            "active_tree_regular_allocated_bytes_U3_gross_path_sum": sum(int(item["allocated_bytes"]) for item in regular),
            "regular_paths_with_multiple_hardlinks_U1": sum(int(item.get("nlink", 1)) > 1 for item in regular),
            "redundant_files_U1": len(redundant),
            "unique_files_U1": len(unique),
            "unclassified_regular_files_U1": sum(item.get("class") == "unclassified_hash_failure" for item in regular),
            "eligible_relocation_files_U1": len(eligible),
            "blocked_relocation_candidates_U1": len(blocked),
            "eligible_candidates_with_unknown_open_state_U1": sum(
                item.get("planned_action") == "eligible_after_review_and_later_authorization"
                and item.get("open_status", "").startswith("unknown_")
                for item in entries
            ),
            "candidates_blocked_by_known_writable_open_U1": sum(
                item.get("planned_action") == "blocked_source_inode_known_writable_open"
                for item in entries
            ),
            "redundant_logical_bytes_all_twin_files_U3": redundant_logical,
            "redundant_allocated_bytes_all_twin_files_U3_gross_path_sum": redundant_allocated,
            "unique_logical_bytes_retained_U3": unique_logical,
            "unique_allocated_bytes_retained_U3_gross_path_sum": unique_allocated,
            "active_tree_reduction_logical_bytes_planned_U3": logical_reduction,
            "actual_active_tree_reduction_logical_bytes_U3": 0,
            "actual_active_tree_reduction_allocated_bytes_U3": 0,
            "active_tree_allocated_bytes_targeted_U3_gross_path_sum": gross_allocated_targeted,
            "estimated_cross_filesystem_reclaim_bytes_lower_bound_if_executed_now_U3": reclaim_lower_bound,
            "estimated_cross_filesystem_reclaim_bytes_upper_bound_if_no_open_references_U3": reclaim_upper_bound,
            "estimated_reclaim_from_redundant_lower_bound_U3": redundant_reclaim_lower_bound,
            "estimated_reclaim_from_redundant_upper_bound_if_no_open_references_U3": redundant_reclaim_upper_bound,
            "estimated_reclaim_from_unique_U3": 0,
            "actual_disk_reclaimed_bytes_U3": 0,
            "actual_disk_reclaimed_status": "not_executed; this is a plan estimate, not measured release",
        },
        "physical_reclaim_by_inode": estimate_rows,
        "files": sorted(file_entries, key=lambda item: item.get("relative_path", "").casefold()),
    }


def _csv_text(report: dict[str, Any]) -> str:
    fields = [
        "old_path", "new_path", "h1_sha256", "logical_bytes", "allocated_bytes",
        "mode", "class", "surviving_copy", "surviving_copy_after_plan",
        "kind", "planned_action", "relative_path", "nlink", "device", "inode",
        "open_status", "case_conflict", "exfat_path_issue", "hash_status", "hash_error",
        "symlink_target", "relocation_candidate",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for item in report["files"]:
        writer.writerow(item)
    return buffer.getvalue()


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    fs = report["filesystem"]
    locks = report["queue_lock_observation"]
    constraints = report.get("execution_constraints", {})
    lines = [
        "# Research Artifact Relocation Dry-run",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Scope and result",
        "",
        f"- Source tree: `{report['scope']['source_root']}`",
        f"- Proposed archive root: `{report['scope']['destination_root_proposed']}`",
        f"- Source filesystem: `{fs['source']['source']}` / `{fs['source']['filesystem']}`",
        f"- Archive filesystem: `{fs['archive']['source']}` / `{fs['archive']['filesystem']}`",
        f"- Archive currently mounted during this scan: `{fs.get('archive_available_at_scan', True)}`",
        f"- Last confirmed archive free bytes (U3): `{fs.get('archive_free_bytes_last_confirmed')}`",
        f"- Archive filesystem evidence: `{fs.get('archive_snapshot_source')}`; prior snapshot time `{fs.get('archive_snapshot_used_at_utc')}`.",
        f"- Dry-run resource scope: `{constraints}`.",
        f"- Excluded object store: `{report['scope']['object_store_root_excluded']}`",
        f"- Excluded implementation output: `{report['scope']['output_root_excluded']}`",
        "- `/data/datasets/raw/malecns/v1.0/` is outside the source root and was not traversed.",
        "- This R5 scan wrote only its report files; it did not write to the archive or mutate source files. The R5 verification used only the dedicated planner unit test, not the full project suite.",
        "",
        "## Classification and byte totals",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Regular files hashed (U1) | {report['hashing']['regular_files_hashed']} / {summary['regular_files_U1']} |",
        f"| Redundant files (U1; proven by matching SHA-256) | {summary['redundant_files_U1']} |",
        f"| Unique files (U1; no matching file in this tree) | {summary['unique_files_U1']} |",
        f"| Regular file logical bytes in active tree (U3) | {summary['active_tree_regular_logical_bytes_U3']} |",
        f"| Regular file allocated bytes in active tree, gross path sum (U3) | {summary['active_tree_regular_allocated_bytes_U3_gross_path_sum']} |",
        f"| Regular paths with multiple hardlinks (U1) | {summary['regular_paths_with_multiple_hardlinks_U1']} |",
        f"| Eligible redundant relocation candidates (U1) | {summary['eligible_relocation_files_U1']} |",
        f"| Blocked relocation candidates (U1) | {summary['blocked_relocation_candidates_U1']} |",
        f"| Eligible candidates with unknown open state (U1) | {summary['eligible_candidates_with_unknown_open_state_U1']} |",
        f"| Candidates blocked by known writable open/mapping (U1) | {summary['candidates_blocked_by_known_writable_open_U1']} |",
        f"| Redundant logical bytes across all twin paths (U3) | {summary['redundant_logical_bytes_all_twin_files_U3']} |",
        f"| Unique logical bytes retained (U3) | {summary['unique_logical_bytes_retained_U3']} |",
        f"| Planned active-tree logical reduction (U3) | {summary['active_tree_reduction_logical_bytes_planned_U3']} |",
        f"| Actual active-tree logical reduction (U3) | {summary['actual_active_tree_reduction_logical_bytes_U3']} (not executed) |",
        f"| Actual active-tree allocated-byte reduction (U3) | {summary['actual_active_tree_reduction_allocated_bytes_U3']} (not executed) |",
        f"| Cross-filesystem reclaim lower bound now (U3) | {summary['estimated_cross_filesystem_reclaim_bytes_lower_bound_if_executed_now_U3']} |",
        f"| Reclaim upper bound if no open/mapped references remain (U3) | {summary['estimated_cross_filesystem_reclaim_bytes_upper_bound_if_no_open_references_U3']} |",
        f"| Redundant lower-bound reclaim (U3) | {summary['estimated_reclaim_from_redundant_lower_bound_U3']} |",
        f"| Redundant upper-bound reclaim (U3) | {summary['estimated_reclaim_from_redundant_upper_bound_if_no_open_references_U3']} |",
        f"| Of estimate from unique (U3) | {summary['estimated_reclaim_from_unique_U3']} |",
        f"| Actual disk bytes reclaimed (U3) | {summary['actual_disk_reclaimed_bytes_U3']} (not executed) |",
        "",
        "The lower reclaim bound counts an inode's allocated bytes only when the planned removals remove its last hard link and the `/proc/<pid>/fd` plus `/proc/<pid>/maps` snapshot was complete and found no open or mapped reference. A remaining hard link contributes zero to both bounds. When the snapshot is partial or a read/mapped reference is observed, the lower bound is zero and the upper bound counts those allocated bytes only if the references are gone at removal time. A known writable descriptor or writable mapping blocks that file's relocation candidate; inability to inspect unrelated processes does not. `st_blocks × 512` is the source allocation estimate. Destination allocation and actual free-space change are not measured because the archive was not written.",
        "",
        "## Risks observed",
        "",
        f"- Source case-insensitive path conflicts: {len(report['case_insensitive_destination_conflicts'])} sibling-name groups.",
        f"- Archive destination path scan: `{report['archive_destination_path_check']['current_status']}`; previous live-scan evidence: `{report['archive_destination_path_check'].get('previous_live_archive_scan_evidence')}`.",
        f"- Symlinks in source tree: {summary['symlinks_U1']} (reported, not followed or moved).",
        f"- Hash failures: {report['hashing']['hash_failures']}.",
        f"- Tree stable across the read-only scan: `{report['tree_stability']['stable_during_scan']}`.",
        f"- Open-file snapshot: `{report['open_file_snapshot']['status']}`; unreadable PID count: {report['open_file_snapshot'].get('unreadable_pid_count')}.",
        f"- Open/mapped source inodes observed among inspectable processes: {len(report['open_file_snapshot'].get('open_inodes', []))}; known writable inodes: {len(report['open_file_snapshot'].get('write_open_inodes', []))}.",
        f"- Project-scoped lock snapshot: `{locks['lock_snapshot_status']}`; matching live locks: {len(locks['live_locks_under_project'])}; lock marker files present: {len(locks['lock_marker_files_present'])}.",
        f"- Project workload process snapshot: `{locks['process_snapshot_status']}`; matching processes: {len(locks['matching_project_workload_processes'])}.",
        f"- Lock boundary: {locks['boundary']}",
        f"- Regular source paths with `nlink > 1`: {summary['regular_paths_with_multiple_hardlinks_U1']}; estimate logic groups by `(device, inode)` and will not claim reclaim while a link remains.",
        "",
        "## Review boundary",
        "",
        "The plan and per-file CSV are proposals. Unique files remain in the active tree by default. No relocation has been approved or executed. The later executor requires an exact plan SHA-256, revalidates source H1 and survivor H1, verifies the real exFAT mount before creating any archive directories, copies and verifies the archive file, durably records the relocation on `/data`, then atomically replaces the old source path with a symlink to the active survivor and removes the parked source only after the symlink resolves to the expected H1. Runtime paths never point into the cold archive. The symlink strategy preserves ordinary path-based reads, but whether every project consumer accepts symlinks or relies on original inode/mode behavior remains unverified; integration review must check this before any later execution.",
        "",
        f"Complete per-file rows are in `{report.get('output_files', {}).get('csv', 'migration-files-v2.csv')}` and `{report.get('output_files', {}).get('json', 'migration-plan-v2.json')}`.",
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path, suffix: str = "v2") -> dict[str, str]:
    output_dir = _absolute(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if suffix and any(not (character.isalnum() or character in "._-") for character in suffix):
        raise PlanError(f"unsafe report suffix: {suffix!r}")
    suffix_part = f"-{suffix}" if suffix else ""
    paths = {
        "json": output_dir / f"migration-plan{suffix_part}.json",
        "csv": output_dir / f"migration-files{suffix_part}.csv",
        "markdown": output_dir / f"REPORT{suffix_part}.md",
    }
    existing = [os.fspath(path) for path in paths.values() if path.exists()]
    if existing:
        raise PlanError(f"report files already exist; select a new suffix to preserve history: {existing}")
    report["output_files"] = {key: os.fspath(path) for key, path in paths.items()}
    payloads = {
        "json": json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "csv": _csv_text(report),
        "markdown": _markdown(report),
    }
    for key, path in paths.items():
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wt", encoding="utf-8", newline="", dir=output_dir,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_name = handle.name
                handle.write(payloads[key])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
    return {key: os.fspath(path) for key, path in paths.items()}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _filesystem_device_id(path: Path) -> int:
    return int(os.stat(path).st_dev)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_records_path(path: Path) -> Path:
    """Resolve and verify the durable relocation-record location on /data."""
    path = _absolute(path)
    data_root = Path("/data").resolve(strict=True)
    data_device = os.stat(data_root).st_dev
    if not _is_relative_to(path, data_root):
        raise PlanError(f"relocation record path must be under /data: {path}")
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise PlanError(f"relocation record parent must already exist on /data: {path.parent}") from exc
    if not _is_relative_to(resolved_parent, data_root):
        raise PlanError(f"relocation record parent resolves outside /data: {resolved_parent}")
    if not resolved_parent.is_dir():
        raise PlanError(f"relocation record parent is not a directory: {resolved_parent}")
    if os.stat(resolved_parent).st_dev != data_device:
        raise PlanError(f"relocation record parent is on a different filesystem from /data: {resolved_parent}")
    resolved_path = resolved_parent / path.name
    if resolved_path.exists() or resolved_path.is_symlink():
        st = os.lstat(resolved_path)
        if stat.S_ISLNK(st.st_mode):
            raise PlanError(f"relocation record file cannot be a symlink: {resolved_path}")
        if not stat.S_ISREG(st.st_mode) or st.st_dev != data_device:
            raise PlanError(f"relocation record file must be regular and remain on /data: {resolved_path}")
    return resolved_path


def _validate_active_root(source_root: Path) -> tuple[Path, int]:
    """Require the active source root to resolve inside the project's /data tree."""
    source_root = _absolute(source_root)
    try:
        resolved_root = source_root.resolve(strict=True)
        data_root = Path("/data").resolve(strict=True)
    except OSError as exc:
        raise PlanError(f"active source root is unavailable: {source_root}") from exc
    data_device = os.stat(data_root).st_dev
    if not _is_relative_to(resolved_root, DEFAULT_SOURCE_ROOT.resolve(strict=True)):
        raise PlanError(f"active source root is outside the project engineering tree: {resolved_root}")
    if not _is_relative_to(resolved_root, data_root) or os.stat(resolved_root).st_dev != data_device:
        raise PlanError(f"active source root must resolve to the /data filesystem: {resolved_root}")
    if source_root != resolved_root:
        raise PlanError(f"active source root cannot use a symlink path: {source_root}")
    return resolved_root, int(data_device)


def _assert_active_data_mount(source_root: Path) -> int:
    """Confirm the active tree is on the expected live /data ext4 filesystem."""
    root, data_device = _validate_active_root(source_root)
    info = _mount_info(root)
    if (
        info["mountpoint"] != "/data"
        or info["filesystem"].lower() != "ext4"
        or Path(info["source"]).name != "nvme0n1p3"
        or _filesystem_device_id(root) != data_device
    ):
        raise PlanError(
            "active source must resolve to the expected /data ext4 filesystem "
            f"(/dev/nvme0n1p3), found {info['mountpoint']} {info['source']} {info['filesystem']}"
        )
    return data_device


def _validate_active_regular_file(
    path: Path,
    source_root: Path,
    expected_h1: str,
    expected_bytes: int | None = None,
) -> os.stat_result:
    """Validate a non-symlink regular file and its content on the active /data device."""
    root, data_device = _validate_active_root(source_root)
    path = _absolute(path)
    if not _is_relative_to(path, root):
        raise PlanError(f"active path is outside the reviewed source root: {path}")
    try:
        parent_real = path.parent.resolve(strict=True)
        st = os.lstat(path)
    except OSError as exc:
        raise PlanError(f"active file is missing or inaccessible: {path}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise PlanError(f"active file must be a regular non-symlink inside the source root: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise PlanError(f"active file must be a regular non-symlink inside the source root: {path}")
    if parent_real != path.parent:
        raise PlanError(f"active file path uses a symlinked parent: {path.parent}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PlanError(f"active file path is missing or inaccessible: {path}") from exc
    if resolved != path or not _is_relative_to(resolved, root):
        raise PlanError(f"active file path uses a symlinked or outside target: {path}")
    if st.st_dev != data_device or os.stat(resolved).st_dev != data_device:
        raise PlanError(f"active file is not on the /data filesystem: {path}")
    if expected_bytes is not None and st.st_size != expected_bytes:
        raise PlanError(f"active file byte count changed since review: {path}")
    if _sha256_path(path) != expected_h1:
        raise PlanError(f"active file H1 changed since review: {path}")
    return st


def _validate_active_survivor(path: Path, source_root: Path, expected_h1: str) -> os.stat_result:
    """Survivors must remain real active files; symlinks to cold storage are forbidden."""
    return _validate_active_regular_file(path, source_root, expected_h1)


def _validate_active_survivor_link(
    path: Path, survivor: Path, source_root: Path, expected_h1: str
) -> None:
    root, data_device = _validate_active_root(source_root)
    path = _absolute(path)
    survivor = _absolute(survivor)
    if not _is_relative_to(path, root) or not _is_relative_to(survivor, root):
        raise PlanError(f"active source link or survivor escapes /data project tree: {path}")
    try:
        st = os.lstat(path)
        link_target = os.readlink(path)
        parent_real = path.parent.resolve(strict=True)
    except OSError as exc:
        raise PlanError(f"active source symlink is missing or broken: {path}") from exc
    if not stat.S_ISLNK(st.st_mode):
        raise PlanError(f"expected an active survivor symlink at {path}")
    if parent_real != path.parent:
        raise PlanError(f"active source symlink parent is not a real in-tree directory: {path.parent}")
    if link_target != os.fspath(survivor):
        raise PlanError(f"active source symlink target changed from reviewed survivor: {path}")
    if os.stat(path.parent).st_dev != data_device:
        raise PlanError(f"active source symlink parent is not on /data: {path.parent}")
    _validate_active_survivor(survivor, root, expected_h1)
    try:
        resolved_target = path.resolve(strict=True)
    except OSError as exc:
        raise PlanError(f"active source symlink is broken: {path}") from exc
    if resolved_target != survivor:
        raise PlanError(f"active source symlink target changed from reviewed survivor: {path}")
    if _sha256_path(path) != expected_h1:
        raise PlanError(f"active source symlink does not resolve to expected H1: {path}")


def _validate_reviewed_row_paths(
    row: dict[str, Any], source_root: Path, destination_root: Path
) -> tuple[Path, Path, Path]:
    try:
        relative = _validate_archive_relative_path(str(row["relative_path"]))
        old = _absolute(Path(row["old_path"]))
        new = _absolute(Path(row["new_path"]))
        survivor = _absolute(Path(row["surviving_copy_after_plan"]))
        reviewed_survivor = _absolute(Path(row["surviving_copy"]))
    except (KeyError, TypeError) as exc:
        raise PlanError("reviewed row is missing an exact old/new/survivor path") from exc
    expected_old = _absolute(source_root.joinpath(*relative.parts))
    expected_new = _absolute(destination_root.joinpath(*relative.parts))
    if (
        old != expected_old or new != expected_new
        or os.fspath(row["old_path"]) != os.fspath(expected_old)
        or os.fspath(row["new_path"]) != os.fspath(expected_new)
    ):
        raise PlanError(f"reviewed old/new paths do not match their relative path: {old} -> {new}")
    if reviewed_survivor != survivor or os.fspath(row["surviving_copy_after_plan"]) != os.fspath(survivor):
        raise PlanError(f"reviewed survivor fields disagree for {old}: {reviewed_survivor} vs {survivor}")
    if not _is_relative_to(survivor, _absolute(source_root)) or survivor == old:
        raise PlanError(f"reviewed survivor must be a distinct path in the active tree: {survivor}")
    for component in relative.parts:
        if any(char in EXFAT_FORBIDDEN for char in component) or component.endswith((" ", ".")):
            raise PlanError(f"reviewed path contains a component unsupported by exFAT: {component!r}")
    return old, new, survivor


def _atomic_json(path: Path, value: Any) -> None:
    """Atomically persist the /data-side index; never creates an unchecked parent."""
    path = _validate_records_path(path)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wt", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Revalidate immediately before replacing the durable record.
        current_path = _validate_records_path(path)
        if current_path != path:
            raise PlanError(f"relocation record parent changed while writing: {path}")
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _assert_archive_mount(archive_mount: Path, source_root: Path | None = None) -> int:
    archive_mount = _absolute(archive_mount)
    if not archive_mount.is_dir():
        raise PlanError(f"archive mount is offline; refusing to create destination paths: {archive_mount}")
    if archive_mount.resolve(strict=True) != archive_mount:
        raise PlanError(f"archive mount path resolves through a symlink: {archive_mount}")
    info = _mount_info(archive_mount)
    if info["mountpoint"] != os.fspath(archive_mount):
        raise PlanError(f"archive path is not a mountpoint: {archive_mount}")
    if info["filesystem"].lower() != "exfat" or Path(info["source"]).name != "sda1":
        raise PlanError(f"archive mount must be /dev/sda1 exFAT, found {info['source']} {info['filesystem']}")
    archive_device = _filesystem_device_id(archive_mount)
    if source_root is not None and archive_device == _filesystem_device_id(source_root):
        raise PlanError("archive and active source are on the same filesystem")
    return archive_device


def _assert_recorded_archive_mount(archive_mount: Path, source_root: Path) -> int:
    """Recovery records cannot redirect reads or writes to an arbitrary filesystem."""
    archive_mount = _absolute(archive_mount)
    if archive_mount != _absolute(DEFAULT_ARCHIVE_MOUNT):
        raise PlanError(f"recorded archive mount differs from the reviewed archive device: {archive_mount}")
    return _assert_archive_mount(archive_mount, source_root)


def _validate_recorded_archive_root(archive_root: Path, archive_mount: Path) -> Path:
    archive_root = _absolute(archive_root)
    if archive_root != _absolute(DEFAULT_DESTINATION_ROOT) or not _is_relative_to(archive_root, archive_mount):
        raise PlanError(f"recorded archive package root differs from the reviewed destination: {archive_root}")
    return archive_root


def _ensure_archive_directory(path: Path, archive_mount: Path, archive_device: int) -> Path:
    """Create directories one level at a time, rejecting symlinks and mount escapes."""
    path = _absolute(path)
    archive_mount = _absolute(archive_mount)
    if not _is_relative_to(path, archive_mount):
        raise PlanError(f"archive path escapes the mounted archive root: {path}")
    root_stat = os.lstat(archive_mount)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise PlanError(f"archive mount root is not a real directory: {archive_mount}")
    if _filesystem_device_id(archive_mount) != archive_device:
        raise PlanError("archive mount device changed during relocation")
    current = archive_mount
    relative = path.relative_to(archive_mount)
    for component in relative.parts:
        try:
            matching_names = [
                entry.name for entry in os.scandir(current)
                if entry.name.casefold() == component.casefold()
            ]
        except OSError as exc:
            raise PlanError(f"cannot inspect archive directory for case conflicts: {current}") from exc
        if len(matching_names) > 1 or (matching_names and matching_names[0] != component):
            raise PlanError(f"archive directory component has a case-insensitive collision: {current / component}")
        child = current / component
        if not matching_names:
            child.mkdir()
            _fsync_directory(current)
            child_st = os.lstat(child)
        else:
            child_st = os.lstat(child)
        if stat.S_ISLNK(child_st.st_mode) or not stat.S_ISDIR(child_st.st_mode):
            raise PlanError(f"archive path component must be a real directory: {child}")
        if _filesystem_device_id(child) != archive_device:
            raise PlanError(f"archive path component leaves the verified exFAT device: {child}")
        current = child
    return current


def _validate_archive_file(path: Path, archive_mount: Path, archive_device: int) -> os.stat_result | None:
    path = _absolute(path)
    if not _is_relative_to(path, _absolute(archive_mount)):
        raise PlanError(f"archive file path escapes the verified mount: {path}")
    _ensure_archive_directory(path.parent, archive_mount, archive_device)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise PlanError(f"archive file must be a regular file, not a symlink or special entry: {path}")
    if _filesystem_device_id(path) != archive_device:
        raise PlanError(f"archive file is not on the verified archive device: {path}")
    return st


def _archive_file_if_present(
    path: Path, archive_mount: Path, archive_device: int
) -> os.stat_result | None:
    """Inspect an archive path without creating directories during verification/recovery."""
    path = _absolute(path)
    archive_mount = _absolute(archive_mount)
    if not _is_relative_to(path, archive_mount):
        raise PlanError(f"archive file path escapes the verified mount: {path}")
    root_st = os.lstat(archive_mount)
    if stat.S_ISLNK(root_st.st_mode) or not stat.S_ISDIR(root_st.st_mode):
        raise PlanError(f"archive mount root is not a real directory: {archive_mount}")
    if _filesystem_device_id(archive_mount) != archive_device:
        raise PlanError("archive mount device changed during verification")
    current = archive_mount
    for component in path.relative_to(archive_mount).parts[:-1]:
        current = current / component
        try:
            child_st = os.lstat(current)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(child_st.st_mode) or not stat.S_ISDIR(child_st.st_mode):
            raise PlanError(f"archive path component must be a real directory: {current}")
        if _filesystem_device_id(current) != archive_device:
            raise PlanError(f"archive path component leaves the verified device: {current}")
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise PlanError(f"archive file must be a regular file, not a symlink or special entry: {path}")
    if _filesystem_device_id(path) != archive_device:
        raise PlanError(f"archive file is not on the verified archive device: {path}")
    return st


def _new_archive_inventory() -> dict[str, Any]:
    return {
        "schema": ARCHIVE_INVENTORY_SCHEMA,
        "archive_package": "malecns-gpu-engineering",
        "format_note": "All item paths are relative to this directory; no /data path is needed to inspect or verify this package.",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manual_integrity_check": {
            "frequency": "Run manually after each archive connection and periodically while retained.",
            "working_directory": "the directory containing archive-inventory.json",
            "reads_outside_archive_package": False,
            "command": MANUAL_ARCHIVE_VERIFY_COMMAND,
        },
        "items": [],
    }


def _load_archive_inventory(path: Path, archive_mount: Path, archive_device: int) -> dict[str, Any]:
    st = _archive_file_if_present(path, archive_mount, archive_device)
    if st is None:
        return _new_archive_inventory()
    value = _load_json(path)
    if value.get("schema") != ARCHIVE_INVENTORY_SCHEMA or not isinstance(value.get("items"), list):
        raise PlanError(f"archive inventory has an unsupported or invalid schema: {path}")
    seen_paths: set[str] = set()
    for item in value["items"]:
        if not isinstance(item, dict):
            raise PlanError(f"archive inventory contains a non-object item: {path}")
        relative_path = item.get("relative_archive_path", "")
        _validate_archive_relative_path(relative_path)
        h1 = item.get("H1")
        mode = item.get("source_original_mode")
        try:
            if not isinstance(h1, str) or len(h1) != 64 or len(bytes.fromhex(h1)) != 32:
                raise ValueError
            if not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
                raise ValueError
            if not isinstance(mode, str) or len(mode) != 4 or any(char not in "01234567" for char in mode):
                raise ValueError
        except ValueError as exc:
            raise PlanError(f"archive inventory contains invalid H1/bytes/mode metadata: {path}") from exc
        if relative_path in seen_paths:
            raise PlanError(f"archive inventory contains a duplicate relative path: {relative_path}")
        seen_paths.add(relative_path)
        source_relative_path = item.get("source_relative_path")
        _validate_archive_relative_path(source_relative_path)
        if item.get("class") != "redundant":
            raise PlanError(f"archive inventory contains a non-redundant item: {path}")
    return value


def _validate_archive_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value in {".", ".."} or "\\" in value:
        raise PlanError(f"invalid archive-relative path: {value!r}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PlanError(f"archive-relative path must be normalized and relative: {value!r}")
    return relative


def _archive_inventory_item(row: dict[str, Any], archive_root: Path) -> dict[str, Any]:
    relative = Path(row["new_path"]).relative_to(archive_root).as_posix()
    _validate_archive_relative_path(relative)
    return {
        "H1": row["h1_sha256"],
        "relative_archive_path": relative,
        "bytes": int(row["logical_bytes"]),
        "source_original_mode": row["mode"],
        "class": "redundant",
        "source_relative_path": row["relative_path"],
    }


def _write_archive_inventory(
    archive_root: Path,
    archive_mount: Path,
    archive_device: int,
    item: dict[str, Any],
) -> Path:
    archive_root = _ensure_archive_directory(archive_root, archive_mount, archive_device)
    path = archive_root / ARCHIVE_INVENTORY_NAME
    case_matches = [
        entry.name for entry in os.scandir(archive_root)
        if entry.name.casefold() == ARCHIVE_INVENTORY_NAME.casefold()
    ]
    if len(case_matches) > 1 or any(name != ARCHIVE_INVENTORY_NAME for name in case_matches):
        raise PlanError(f"archive inventory path has a case-insensitive collision: {path}")
    inventory = _load_archive_inventory(path, archive_mount, archive_device)
    matches = [entry for entry in inventory["items"] if entry.get("relative_archive_path") == item["relative_archive_path"]]
    if matches:
        if len(matches) != 1 or matches[0] != item:
            raise PlanError(f"archive inventory path already has conflicting metadata: {item['relative_archive_path']}")
    else:
        inventory["items"].append(item)
        inventory["items"].sort(key=lambda entry: entry["relative_archive_path"])
    inventory["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wt", encoding="utf-8", dir=archive_root,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(inventory, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _validate_archive_file(path, archive_mount, archive_device)
        os.replace(temp_name, path)
        _fsync_directory(archive_root)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


def verify_archive_inventory(archive_root: Path) -> dict[str, Any]:
    """Manually verify an archive package using only its local inventory and files."""
    archive_root_arg = _absolute(archive_root)
    archive_root = archive_root_arg.resolve(strict=True)
    if archive_root != archive_root_arg or not archive_root.is_dir():
        raise PlanError(f"archive package root must be a real directory, not a symlink path: {archive_root_arg}")
    inventory_path = archive_root / ARCHIVE_INVENTORY_NAME
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise PlanError(f"archive inventory is missing or not a regular file: {inventory_path}")
    inventory = _load_json(inventory_path)
    if inventory.get("schema") != ARCHIVE_INVENTORY_SCHEMA or not isinstance(inventory.get("items"), list):
        raise PlanError(f"unsupported archive inventory schema: {inventory_path}")
    errors: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for item in inventory.get("items", []):
        try:
            if not isinstance(item, dict):
                raise PlanError("inventory item is not an object")
            relative = _validate_archive_relative_path(item.get("relative_archive_path", ""))
            if relative.as_posix() in seen_paths:
                raise PlanError("duplicate relative archive path")
            seen_paths.add(relative.as_posix())
            h1 = item.get("H1")
            if not isinstance(h1, str) or len(h1) != 64 or len(bytes.fromhex(h1)) != 32:
                raise PlanError("invalid H1")
            if not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
                raise PlanError("invalid byte count")
            mode = item.get("source_original_mode")
            if not isinstance(mode, str) or len(mode) != 4 or any(char not in "01234567" for char in mode):
                raise PlanError("invalid original source mode")
            _validate_archive_relative_path(item.get("source_relative_path", ""))
            if item.get("class") != "redundant":
                raise PlanError("inventory item is not classified redundant")
            path = archive_root.joinpath(*relative.parts)
            parent = archive_root
            for component in relative.parts[:-1]:
                parent = parent / component
                parent_st = os.lstat(parent)
                if stat.S_ISLNK(parent_st.st_mode) or not stat.S_ISDIR(parent_st.st_mode):
                    raise PlanError("archive parent path is a symlink or not a directory")
            item_st = os.lstat(path)
            if stat.S_ISLNK(item_st.st_mode) or not stat.S_ISREG(item_st.st_mode):
                raise PlanError("archive path is not a regular non-symlink file")
            resolved = path.resolve(strict=True)
            if resolved != path or not _is_relative_to(resolved, archive_root):
                raise PlanError("path escapes package or is not a regular file")
            if path.stat().st_size != int(item["bytes"]) or _sha256_path(path) != h1:
                raise PlanError("size or H1 mismatch")
        except (OSError, KeyError, ValueError, PlanError) as exc:
            errors.append({"relative_archive_path": str(item.get("relative_archive_path", "")), "error": str(exc)})
    return {"ok": not errors, "checked_files": len(inventory.get("items", [])), "errors": errors}


def _verify_archived_item(
    archive_root: Path,
    archive_mount: Path,
    archive_device: int,
    relative_archive_path: str,
    expected_h1: str,
    expected_bytes: int,
) -> Path:
    relative = _validate_archive_relative_path(relative_archive_path)
    path = archive_root.joinpath(*relative.parts)
    st = _archive_file_if_present(path, archive_mount, archive_device)
    if st is None or st.st_size != expected_bytes or _sha256_path(path) != expected_h1:
        raise PlanError(f"archive copy is missing or failed H1 verification: {path}")
    inventory_path = archive_root / ARCHIVE_INVENTORY_NAME
    inventory = _load_archive_inventory(inventory_path, archive_mount, archive_device)
    if not any(
        item.get("relative_archive_path") == relative_archive_path
        and item.get("H1") == expected_h1
        and int(item.get("bytes", -1)) == expected_bytes
        for item in inventory["items"]
    ):
        raise PlanError(f"archive inventory does not describe this verified copy: {relative_archive_path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PlanError(f"expected a JSON object in {path}")
    return value


def _execute_one(
    row: dict[str, Any],
    source_root: Path,
    destination_root: Path,
    archive_mount: Path,
    records_path: Path,
    records: dict[str, Any],
    expected_source_nlink: int,
) -> dict[str, Any]:
    """Copy, verify, durably inventory both copies, then replace an active path."""
    if row.get("class") != "redundant" or row.get("planned_action") != "eligible_after_review_and_later_authorization":
        raise PlanError(f"plan row is not eligible: {row.get('old_path')}")
    records_path = _validate_records_path(records_path)
    source_root = _absolute(source_root)
    destination_root = _absolute(destination_root)
    archive_mount = _absolute(archive_mount)
    active_root, data_device = _validate_active_root(source_root)
    data_device = _assert_active_data_mount(active_root)
    old, new, survivor = _validate_reviewed_row_paths(row, active_root, destination_root)
    expected_h1 = row.get("h1_sha256")
    expected_bytes = int(row["logical_bytes"])
    if not isinstance(expected_h1, str) or len(expected_h1) != 64:
        raise PlanError(f"reviewed row has no valid H1: {old}")

    # Verify the real mount before creating even the first archive directory.
    archive_device = _assert_archive_mount(archive_mount, active_root)
    source_st = _validate_active_regular_file(old, active_root, expected_h1, expected_bytes)
    survivor_st = _validate_active_survivor(survivor, active_root, expected_h1)
    if source_st.st_dev != data_device or survivor_st.st_dev != data_device:
        raise PlanError("source and surviving copy must remain on the active /data filesystem")
    if int(source_st.st_nlink) != expected_source_nlink:
        raise PlanError(f"source hard-link count changed since review: {old}")
    if _destination_collision(destination_root, row["relative_path"]):
        raise PlanError(f"archive destination or casefold-equivalent path already exists: {new}")
    live_snapshot = _open_file_snapshot({(int(source_st.st_dev), int(source_st.st_ino))})
    live_write_inodes = {tuple(pair) for pair in live_snapshot.get("write_open_inodes", [])}
    if (int(source_st.st_dev), int(source_st.st_ino)) in live_write_inodes:
        raise PlanError(f"source has a known writable descriptor or mapping at execution time: {old}")
    if any(event.get("old_path") == os.fspath(old) and event.get("status") not in {"complete", "rolled_back_source_restored_archive_copy_retained", "rolled_back_source_restored_from_verified_archive_copy", "rolled_back_source_was_already_present_archive_copy_retained"} for event in records.get("relocations", [])):
        raise PlanError(f"source already has an unresolved relocation record: {old}")

    # The archive destination is only created after the device identity is proven.
    _ensure_archive_directory(new.parent, archive_mount, archive_device)
    if _destination_collision(destination_root, row["relative_path"]):
        raise PlanError(f"archive destination or casefold-equivalent path appeared during setup: {new}")
    if _validate_archive_file(new, archive_mount, archive_device) is not None:
        raise PlanError(f"archive destination already exists: {new}")
    temp_new = new.with_name(f".{new.name}.copying.{uuid.uuid4().hex}")
    digest = hashlib.sha256()
    source_fd = -1
    target_fd = -1
    try:
        source_fd = os.open(old, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        source_before = os.fstat(source_fd)
        if (source_before.st_dev, source_before.st_ino, source_before.st_size) != (source_st.st_dev, source_st.st_ino, source_st.st_size):
            raise PlanError(f"source identity changed before archive copy: {old}")
        target_fd = os.open(temp_new, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
        with os.fdopen(source_fd, "rb", closefd=True) as source, os.fdopen(target_fd, "wb", closefd=True) as target:
            source_fd = target_fd = -1
            while block := source.read(CHUNK_SIZE):
                target.write(block)
                digest.update(block)
            target.flush()
            os.fsync(target.fileno())
            source_after = os.fstat(source.fileno())
        if (source_after.st_dev, source_after.st_ino, source_after.st_size, source_after.st_mtime_ns, source_after.st_ctime_ns) != (
            source_st.st_dev, source_st.st_ino, source_st.st_size, source_st.st_mtime_ns, source_st.st_ctime_ns
        ):
            raise PlanError(f"source changed while copying to archive: {old}")
        if digest.hexdigest() != expected_h1 or _sha256_path(temp_new) != expected_h1:
            raise PlanError(f"staged archive copy failed H1 verification: {old}")
        os.replace(temp_new, new)
        _fsync_directory(new.parent)
        archived_st = _validate_archive_file(new, archive_mount, archive_device)
        if archived_st is None or archived_st.st_size != expected_bytes or _sha256_path(new) != expected_h1:
            raise PlanError(f"final archive file failed H1 verification: {new}")
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if target_fd >= 0:
            os.close(target_fd)
        if temp_new.exists() or temp_new.is_symlink():
            temp_new.unlink()

    archive_relative_path = new.relative_to(destination_root).as_posix()
    inventory_item = _archive_inventory_item(row, destination_root)
    inventory_path = _write_archive_inventory(destination_root, archive_mount, archive_device, inventory_item)
    _verify_archived_item(destination_root, archive_mount, archive_device, archive_relative_path, expected_h1, expected_bytes)

    event = {
        "old_path": os.fspath(old),
        "relative_path": row["relative_path"],
        "new_path": os.fspath(new),
        "archive_mount": os.fspath(archive_mount),
        "archive_root": os.fspath(destination_root),
        "archive_inventory_path": os.fspath(inventory_path),
        "relative_archive_path": archive_relative_path,
        "source_root": os.fspath(active_root),
        "H1": expected_h1,
        "bytes": expected_bytes,
        "mode": row["mode"],
        "class": "redundant",
        "surviving_copy": os.fspath(survivor),
        "source_device": int(source_st.st_dev),
        "source_inode": int(source_st.st_ino),
        "source_nlink_before_move": int(source_st.st_nlink),
        "destination_device": int(archive_device),
        "archive_inventory_durable": True,
        "copy_verified": True,
        "status": "archive_copy_and_inventory_durable_source_present",
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    records.setdefault("schema", "malecns-gpu.relocations.v1")
    records.setdefault("relocations", [])
    records["relocations"].append(event)
    _atomic_json(records_path, records)

    parked = old.with_name(f".{old.name}.relocation-pending.{uuid.uuid4().hex}")
    temp_link = old.with_name(f".{old.name}.survivor-link.{uuid.uuid4().hex}")
    event["parked_source_path"] = os.fspath(parked)
    event["status"] = "source_parking_intent_recorded"
    _atomic_json(records_path, records)
    os.symlink(os.fspath(survivor), temp_link)
    try:
        latest_source_st = _validate_active_regular_file(old, active_root, expected_h1, expected_bytes)
        _validate_active_survivor(survivor, active_root, expected_h1)
        if (
            latest_source_st.st_dev != source_st.st_dev
            or latest_source_st.st_ino != source_st.st_ino
            or latest_source_st.st_nlink != expected_source_nlink
        ):
            raise PlanError(f"source identity or link count changed before path replacement: {old}")
        os.replace(old, parked)
        _fsync_directory(old.parent)
        parked_st = os.lstat(parked)
        if not stat.S_ISREG(parked_st.st_mode) or (parked_st.st_dev, parked_st.st_ino) != (source_st.st_dev, source_st.st_ino) or _sha256_path(parked) != expected_h1:
            if not old.exists() and not old.is_symlink() and parked.exists():
                os.replace(parked, old)
                _fsync_directory(old.parent)
            raise PlanError(f"parked source identity changed during path replacement: {parked}")
        event["status"] = "source_parked_recoverable"
        _atomic_json(records_path, records)

        os.replace(temp_link, old)
        _fsync_directory(old.parent)
        _validate_active_survivor_link(old, survivor, active_root, expected_h1)
        event["status"] = "active_path_replaced_by_survivor_symlink"
        event["source_path_is_active_symlink"] = True
        event["source_symlink_target"] = os.fspath(survivor)
        _atomic_json(records_path, records)

        parked.unlink()
        _fsync_directory(old.parent)
        event["status"] = "complete"
        event["parked_source_removed_after_record_and_symlink_verification"] = True
        event["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _atomic_json(records_path, records)
    finally:
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()
    return event


def recover_pending(
    records_path: Path,
    action: str,
    expected_records_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Restore or finish crash-interrupted relocations; archive copies are retained."""
    records_path = _validate_records_path(records_path)
    if action not in {"restore", "finish"}:
        raise PlanError("recovery action must be 'restore' or 'finish'")
    if action == "finish" and (
        not expected_records_sha256 or _sha256_path(records_path) != expected_records_sha256
    ):
        raise PlanError("finishing recovery requires the exact current records SHA-256")
    records = _load_json(records_path)
    changed: list[dict[str, Any]] = []
    pending = {
        "archive_copy_and_inventory_durable_source_present", "source_parking_intent_recorded",
        "source_parked_recoverable", "active_path_replaced_by_survivor_symlink",
    }
    for event in records.get("relocations", []):
        if event.get("status") not in pending:
            continue
        source_root = _absolute(Path(event.get("source_root", DEFAULT_SOURCE_ROOT)))
        if source_root != _absolute(DEFAULT_SOURCE_ROOT):
            raise PlanError(f"recorded active source root differs from the reviewed project tree: {source_root}")
        active_root, data_device = _validate_active_root(source_root)
        data_device = _assert_active_data_mount(active_root)
        relative = _validate_archive_relative_path(str(event.get("relative_path", "")))
        old = _absolute(Path(event["old_path"]))
        expected_old = _absolute(active_root.joinpath(*relative.parts))
        survivor = _absolute(Path(event["surviving_copy"]))
        expected_h1 = str(event["H1"])
        expected_bytes = int(event["bytes"])
        if old != expected_old or os.fspath(event["old_path"]) != os.fspath(expected_old):
            raise PlanError(f"recovery old path differs from the durable relative path: {old}")
        if (
            os.fspath(event["surviving_copy"]) != os.fspath(survivor)
            or not _is_relative_to(survivor, active_root)
            or survivor == old
        ):
            raise PlanError(f"recovery survivor outside the active source tree: {survivor}")
        parked = _absolute(Path(event["parked_source_path"])) if event.get("parked_source_path") else None
        if parked is not None and (
            os.fspath(event["parked_source_path"]) != os.fspath(parked)
            or parked.parent != old.parent
            or not parked.name.startswith(f".{old.name}.relocation-pending.")
        ):
            raise PlanError(f"recovery parked path is not the reviewed sibling parking path: {parked}")
        archive_mount = _absolute(Path(event.get("archive_mount", DEFAULT_ARCHIVE_MOUNT)))
        archive_root = _validate_recorded_archive_root(
            Path(event.get("archive_root", DEFAULT_DESTINATION_ROOT)), archive_mount
        )
        new = _absolute(Path(event["new_path"]))
        if (
            not _is_relative_to(archive_root, archive_mount)
            or new != archive_root.joinpath(*_validate_archive_relative_path(event["relative_archive_path"]).parts)
            or os.fspath(event["new_path"]) != os.fspath(new)
        ):
            raise PlanError(f"recovery archive path differs from its durable relative path: {new}")

        def save(status: str) -> None:
            event["status"] = status
            event["recovered_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _atomic_json(records_path, records)

        def has_old() -> bool:
            return old.exists() or old.is_symlink()

        def restore_from_archive(archive_device: int) -> None:
            _verify_archived_item(
                archive_root, archive_mount, archive_device,
                event["relative_archive_path"], expected_h1, expected_bytes,
            )
            if old.parent.resolve(strict=True) != old.parent or os.stat(old.parent).st_dev != data_device:
                raise PlanError(f"active restore parent is not a real /data directory: {old.parent}")
            temp_restore = old.with_name(f".{old.name}.restore.{uuid.uuid4().hex}")
            temp_fd = -1
            try:
                temp_fd = os.open(temp_restore, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
                with open(new, "rb") as source, os.fdopen(temp_fd, "wb", closefd=True) as target:
                    temp_fd = -1
                    while block := source.read(CHUNK_SIZE):
                        target.write(block)
                    target.flush()
                    os.fchmod(target.fileno(), int(event["mode"], 8))
                    os.fsync(target.fileno())
                if _sha256_path(temp_restore) != expected_h1:
                    raise PlanError(f"restored file failed H1 verification: {temp_restore}")
                os.replace(temp_restore, old)
                _fsync_directory(old.parent)
                _validate_active_regular_file(old, active_root, expected_h1, expected_bytes)
            finally:
                if temp_fd >= 0:
                    os.close(temp_fd)
                if temp_restore.exists() or temp_restore.is_symlink():
                    temp_restore.unlink()

        if action == "restore":
            if parked is not None and parked.exists():
                parked_st = _validate_active_regular_file(parked, active_root, expected_h1, expected_bytes)
                del parked_st
                if has_old():
                    old_st = os.lstat(old)
                    if stat.S_ISLNK(old_st.st_mode):
                        _validate_active_survivor_link(old, survivor, active_root, expected_h1)
                        old.unlink()
                        _fsync_directory(old.parent)
                    else:
                        _validate_active_regular_file(old, active_root, expected_h1, expected_bytes)
                        parked.unlink()
                        _fsync_directory(parked.parent)
                        save("rolled_back_source_was_already_present_archive_copy_retained")
                        changed.append(event)
                        continue
                os.replace(parked, old)
                _fsync_directory(old.parent)
                _validate_active_regular_file(old, active_root, expected_h1, expected_bytes)
                save("rolled_back_source_restored_archive_copy_retained")
                changed.append(event)
                continue
            if has_old() and not old.is_symlink():
                _validate_active_regular_file(old, active_root, expected_h1, expected_bytes)
                save("rolled_back_source_was_already_present_archive_copy_retained")
                changed.append(event)
                continue
            if has_old():
                _validate_active_survivor_link(old, survivor, active_root, expected_h1)
            archive_device = _assert_recorded_archive_mount(archive_mount, active_root)
            restore_from_archive(archive_device)
            save("rolled_back_source_restored_from_verified_archive_copy")
            changed.append(event)
            continue

        # finish requires a live, verified archive and a verified active survivor.
        archive_device = _assert_recorded_archive_mount(archive_mount, active_root)
        _verify_archived_item(
            archive_root, archive_mount, archive_device,
            event["relative_archive_path"], expected_h1, expected_bytes,
        )
        _validate_active_survivor(survivor, active_root, expected_h1)
        if parked is not None and parked.exists():
            _validate_active_regular_file(parked, active_root, expected_h1, expected_bytes)
        if has_old():
            old_st = os.lstat(old)
            if stat.S_ISLNK(old_st.st_mode):
                _validate_active_survivor_link(old, survivor, active_root, expected_h1)
            else:
                source_now = _validate_active_regular_file(old, active_root, expected_h1, expected_bytes)
                if parked is not None and parked.exists():
                    raise PlanError(f"both the source path and parking path contain copies; use restore first: {old}")
                if parked is None:
                    parked = old.with_name(f".{old.name}.relocation-pending.recovery-{uuid.uuid4().hex}")
                    event["parked_source_path"] = os.fspath(parked)
                os.replace(old, parked)
                _fsync_directory(old.parent)
                if os.lstat(parked).st_ino != source_now.st_ino:
                    raise PlanError(f"recovery parked a changed source identity: {parked}")
                _validate_active_regular_file(parked, active_root, expected_h1, expected_bytes)
                save("source_parked_recoverable")
        if not has_old():
            if parked is None or not parked.exists():
                # A crash may happen after the parked copy was unlinked. The active
                # survivor and archive inventory then suffice to restore the alias.
                pass
            temp_link = old.with_name(f".{old.name}.survivor-link.{uuid.uuid4().hex}")
            try:
                os.symlink(os.fspath(survivor), temp_link)
                os.replace(temp_link, old)
                _fsync_directory(old.parent)
            finally:
                if temp_link.exists() or temp_link.is_symlink():
                    temp_link.unlink()
        _validate_active_survivor_link(old, survivor, active_root, expected_h1)
        event["source_path_is_active_symlink"] = True
        event["source_symlink_target"] = os.fspath(survivor)
        save("active_path_replaced_by_survivor_symlink")
        if parked is not None and parked.exists():
            _validate_active_regular_file(parked, active_root, expected_h1, expected_bytes)
            parked.unlink()
            _fsync_directory(parked.parent)
        event["parked_source_removed_after_record_and_symlink_verification"] = True
        event["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save("complete")
        changed.append(event)
    return changed


def execute_reviewed_plan(
    plan_path: Path,
    expected_plan_sha256: str,
    records_path: Path,
    selected_old_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute only an exact reviewed plan supplied with its SHA-256 digest."""
    plan_path = plan_path.resolve(strict=True)
    actual_plan_hash = _sha256_path(plan_path)
    if actual_plan_hash != expected_plan_sha256:
        raise PlanError(
            f"reviewed plan digest mismatch: expected {expected_plan_sha256}, found {actual_plan_hash}"
        )
    plan = _load_json(plan_path)
    scope = plan["scope"]
    source_root_arg = _absolute(Path(scope["source_root"]))
    destination_root = _absolute(Path(scope["destination_root_proposed"]))
    archive_mount = _absolute(Path(scope["archive_mount"]))
    if source_root_arg != _absolute(DEFAULT_SOURCE_ROOT):
        raise PlanError(f"reviewed plan source root is outside the fixed engineering scope: {source_root_arg}")
    if archive_mount != _absolute(DEFAULT_ARCHIVE_MOUNT):
        raise PlanError(f"reviewed plan archive mount differs from the configured exFAT device: {archive_mount}")
    if destination_root != _absolute(DEFAULT_DESTINATION_ROOT):
        raise PlanError(f"reviewed plan archive destination differs from the configured package root: {destination_root}")
    source_root = source_root_arg.resolve(strict=True)
    _assert_active_data_mount(source_root)
    if plan.get("execution_status") != "dry_run_only_no_archive_writes_no_source_changes":
        raise PlanError("input is not a dry-run plan produced by this script")
    if not plan.get("tree_stability", {}).get("stable_during_scan"):
        raise PlanError("plan source tree was not stable during scanning")
    if plan.get("hashing", {}).get("hash_failures"):
        raise PlanError("plan contains hash failures")
    records_path = _validate_records_path(records_path)
    lock_snapshot = _queue_lock_observation(source_root)
    if lock_snapshot["lock_snapshot_status"] != "complete" or lock_snapshot["process_snapshot_status"] != "complete":
        raise PlanError("cannot verify project lock/process snapshot before relocation")
    if lock_snapshot["live_locks_under_project"] or lock_snapshot["matching_project_workload_processes"]:
        raise PlanError("active project lock or workload process observed; relocation is deferred")
    records = _load_json(records_path) if records_path.exists() else {
        "schema": "malecns-gpu.relocations.v1", "relocations": []
    }
    rows = [
        row for row in plan["files"]
        if row.get("planned_action") == "eligible_after_review_and_later_authorization"
        and (selected_old_paths is None or row.get("old_path") in selected_old_paths)
    ]
    if selected_old_paths is not None:
        found = {row["old_path"] for row in rows}
        missing = selected_old_paths - found
        if missing:
            raise PlanError(f"requested paths are not eligible in reviewed plan: {sorted(missing)}")
    completed: list[dict[str, Any]] = []
    removed_links_by_inode: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        inode_key = (int(row["device"]), int(row["inode"]))
        expected_nlink = int(row["nlink"]) - removed_links_by_inode[inode_key]
        completed.append(
            _execute_one(row, source_root, destination_root, archive_mount, records_path, records, expected_nlink)
        )
        removed_links_by_inode[inode_key] += 1
    return completed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="create a read-only complete H1 relocation plan")
    plan.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    plan.add_argument("--archive-mount", type=Path, default=DEFAULT_ARCHIVE_MOUNT)
    plan.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    plan.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    plan.add_argument("--object-store-root", type=Path, default=DEFAULT_OBJECT_STORE_ROOT)
    plan.add_argument("--suffix", default="v2", help="report filename suffix; defaults to v2 and never overwrites earlier plans")
    plan.add_argument("--archive-previous-snapshot", type=Path, default=None,
                      help="previous read-only report to preserve proposed paths if the archive is offline")
    execute = subparsers.add_parser(
        "execute-reviewed-plan",
        help="later, explicitly authorized copy-verify-record-relocate operation",
    )
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--expected-plan-sha256", required=True)
    execute.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    execute.add_argument("--old-path", action="append", default=None,
                         help="limit execution to this eligible old path; repeat to select multiple")
    recover = subparsers.add_parser("recover-pending", help="restore or finish an interrupted recorded relocation")
    recover.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    recover.add_argument("--action", choices=("restore", "finish"), required=True)
    recover.add_argument("--expected-records-sha256", default=None,
                         help="required for action=finish; binds completion to reviewed records")
    verify_archive = subparsers.add_parser(
        "verify-archive-inventory",
        help="verify a connected archive package using only its self-contained inventory",
    )
    verify_archive.add_argument("--archive-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            report = create_plan(
                args.source_root, args.destination_root, args.archive_mount,
                args.output_dir.parent, args.object_store_root,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                validate_mounts=True,
                archive_previous_snapshot_path=args.archive_previous_snapshot,
            )
            paths = write_report(report, args.output_dir, args.suffix)
            print(json.dumps({"report_paths": paths, "summary": report["summary"]}, ensure_ascii=False, indent=2))
            return 0 if report["tree_stability"]["stable_during_scan"] and not report["hashing"]["hash_failures"] else 2
        if args.command == "execute-reviewed-plan":
            completed = execute_reviewed_plan(
                args.plan, args.expected_plan_sha256, args.records,
                set(args.old_path) if args.old_path else None,
            )
            print(json.dumps({"completed_relocations": completed}, ensure_ascii=False, indent=2))
        elif args.command == "recover-pending":
            recovered = recover_pending(args.records, args.action, args.expected_records_sha256)
            print(json.dumps({"recovered_relocations": recovered}, ensure_ascii=False, indent=2))
        else:
            result = verify_archive_inventory(args.archive_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 2
        return 0
    except (PlanError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
