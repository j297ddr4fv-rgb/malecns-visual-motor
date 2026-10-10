#!/usr/bin/env python3
"""Compile-preflight and gated CPU runner for the tethered treadmill pilot.

Default mode is compile-only. A physics run requires a separate root-review
authorization JSON bound to the frozen compile evidence. No renderer, neural
worker, GPU, body-yaw fit, or external ball rotation is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
DEFAULT_OUT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-tethered-treadmill-pilot-20261007-v1/compile-preflight-v8"
)
CANONICALIZATION_RULE_ID = "mjcf-stl-implicit-mime-c14n-v1"
STL_CONTENT_TYPE = "model/stl"

SOURCE_PATHS = [
    PROJECT_ROOT / "scripts/prepare_flygym_tethered_treadmill_pilot.py",
    PROJECT_ROOT / "scripts/flygym_tethered_treadmill_session_v1.py",
    PROJECT_ROOT / "scripts/run_flygym_tethered_treadmill_pilot.py",
    PROJECT_ROOT / "scripts/flygym_ap_endpoint_gesture_prototype.py",
    PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v4_1.py",
    PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v4.py",
    PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v3.py",
    PROJECT_ROOT / "scripts/flygym_phase_speed_gait_v2.py",
    PROJECT_ROOT / "scripts/flygym_phase_speed_gait.py",
    PROJECT_ROOT / "scripts/flygym_pose_priority_ik_v2.py",
    PROJECT_ROOT / "scripts/flygym_legacy_rest_compat.py",
    PROJECT_ROOT / "scripts/run_flygym_reference_walk.py",
    Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym/compose/world/tethered_world.py"),
    Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym/compose/world/base_world.py"),
    Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym/compose/physics.py"),
    Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym/simulation.py"),
    Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym_demo/complex_terrain/hybrid_controller.py"),
    Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym_demo/complex_terrain/preprogrammed.py"),
    Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym_demo/complex_terrain/cpg_controller.py"),
    Path("/data/models/flygym/package-assets/flygym/assets/model/neuromechfly/mujoco_globals.yaml"),
    Path("/data/artifacts/by-project/malecns-gpu/engineering/flygym-v4-1-command-transfer-calibration-20261007-v1/cases/static_ps_reference_replay/controller_binding.json"),
    Path("/data/artifacts/by-project/malecns-gpu/engineering/flygym-v4-1-command-transfer-calibration-20261007-v1/cases/static_ps_reference_replay/effective_config.json"),
    Path("/data/artifacts/by-project/malecns-gpu/engineering/flygym-dnp15-ap-online-pilot-20261007-v3/captures/positive60/upstream_asset_hashes.json"),
    Path("/data/environments/projects/flygym-reference/lib/python3.12/site-packages/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def run_id_for(*, p: float, d: float, duration_s: float, seed: int) -> str:
    """Stable run identity used both by compile evidence and later authorization."""
    return f"tethered-ball-P{p:.3f}-d{d:+.3f}-T{duration_s:g}s-seed{int(seed)}"


def requested_condition(*, p: float, d: float, duration_s: float, seed: int) -> dict[str, Any]:
    return {
        "P": float(p),
        "d": float(d),
        "duration_s": float(duration_s),
        "seed": int(seed),
        "run_id": run_id_for(p=p, d=d, duration_s=duration_s, seed=seed),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _element_path(root: ET.Element, target: ET.Element) -> str:
    """Return an index-based path without reordering sibling elements."""
    parents: dict[ET.Element, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parents[child] = parent
    parts: list[str] = []
    node = target
    while node is not root:
        parent = parents[node]
        index = list(parent).index(node)
        parts.append(f"{node.tag}[{index}]")
        node = parent
    return f"/{root.tag}[0]/" + "/".join(reversed(parts))


def _resource_refs(root: ET.Element) -> list[dict[str, str]]:
    rows = []
    for element in root.iter():
        if "file" in element.attrib:
            rows.append({
                "element": str(element.tag),
                "name": str(element.attrib.get("name", "")),
                "file": str(element.attrib["file"]),
            })
    return rows


def canonicalize_world_xml(raw_xml: str) -> tuple[str, dict[str, Any]]:
    """Add only implicit STL MIME labels, then canonicalize attribute order.

    Child elements remain in their original order. No numeric attributes,
    non-MIME attributes, text, or asset file references may change.
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    raw_root = ET.fromstring(raw_xml, parser=parser)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    canonical_root = ET.fromstring(raw_xml, parser=parser)
    defaulted: list[dict[str, str]] = []
    for mesh in canonical_root.findall("./asset/mesh"):
        file_ref = mesh.attrib.get("file", "")
        content_type = mesh.attrib.get("content_type")
        if content_type is not None and content_type != STL_CONTENT_TYPE:
            raise ValueError(
                f"unsupported existing mesh content_type at {_element_path(canonical_root, mesh)}: {content_type!r}"
            )
        if PurePosixPath(file_ref).suffix.lower() == ".stl" and content_type is None:
            mesh.set("content_type", STL_CONTENT_TYPE)
            defaulted.append({
                "element_path": _element_path(canonical_root, mesh),
                "mesh_name": str(mesh.attrib.get("name", "")),
                "file": file_ref,
                "inserted_attribute": "content_type",
                "inserted_value": STL_CONTENT_TYPE,
            })

    attribute_changes: list[dict[str, Any]] = []
    tag_mismatches: list[dict[str, Any]] = []
    text_mismatches: list[dict[str, Any]] = []
    child_order_mismatches: list[dict[str, Any]] = []
    child_count_mismatches: list[dict[str, Any]] = []

    def audit(before: ET.Element, after: ET.Element, path: str) -> None:
        if before.tag != after.tag:
            tag_mismatches.append({"path": path, "before": before.tag, "after": after.tag})
        before_attrs, after_attrs = dict(before.attrib), dict(after.attrib)
        changes = {}
        for key in sorted(set(before_attrs) | set(after_attrs)):
            if before_attrs.get(key) == after_attrs.get(key):
                continue
            allowed_insert = (
                key == "content_type"
                and key not in before_attrs
                and after_attrs.get(key) == STL_CONTENT_TYPE
                and before.tag == "mesh"
                and PurePosixPath(before_attrs.get("file", "")).suffix.lower() == ".stl"
            )
            if not allowed_insert:
                changes[key] = {
                    "before": before_attrs.get(key),
                    "after": after_attrs.get(key),
                }
        if changes:
            attribute_changes.append({"path": path, "changes": changes})
        before_text, after_text = before.text, after.text
        if before_text != after_text:
            text_mismatches.append({"path": path, "before": before_text, "after": after_text})
        before_tail, after_tail = before.tail, after.tail
        if before_tail != after_tail:
            text_mismatches.append({"path": path, "before_tail": before_tail, "after_tail": after_tail})
        before_children, after_children = list(before), list(after)
        if len(before_children) != len(after_children):
            child_count_mismatches.append({
                "path": path,
                "before": len(before_children),
                "after": len(after_children),
            })
        before_tags = [str(child.tag) for child in before_children]
        after_tags = [str(child.tag) for child in after_children]
        if before_tags != after_tags:
            child_order_mismatches.append({
                "path": path,
                "before_tags": before_tags,
                "after_tags": after_tags,
            })
        for index, (before_child, after_child) in enumerate(zip(before_children, after_children)):
            audit(before_child, after_child, f"{path}/{before_child.tag}[{index}]")

    audit(raw_root, canonical_root, f"/{raw_root.tag}[0]")
    before_refs = _resource_refs(raw_root)
    after_refs = _resource_refs(canonical_root)
    if before_refs != after_refs:
        raise ValueError("canonicalization changed or reordered asset file references")
    audit_summary = {
        "rule_id": CANONICALIZATION_RULE_ID,
        "attribute_order_normalized_by": "xml.etree.ElementTree.canonicalize (stdlib C14N)",
        "child_element_order_preserved": not child_order_mismatches,
        "tag_tree_preserved": not tag_mismatches and not child_count_mismatches,
        "all_preexisting_attributes_preserved": not attribute_changes,
        "numeric_attributes_preserved": not attribute_changes,
        "text_preserved": not text_mismatches,
        "asset_file_references_and_order_preserved": True,
        "defaulted_mesh_content_types": defaulted,
        "audit_differences": {
            "attribute_changes_other_than_allowed_defaults": attribute_changes,
            "tag_mismatches": tag_mismatches,
            "text_mismatches": text_mismatches,
            "child_order_mismatches": child_order_mismatches,
            "child_count_mismatches": child_count_mismatches,
        },
    }
    if not all((
        audit_summary["child_element_order_preserved"],
        audit_summary["tag_tree_preserved"],
        audit_summary["all_preexisting_attributes_preserved"],
        audit_summary["numeric_attributes_preserved"],
        audit_summary["text_preserved"],
        audit_summary["asset_file_references_and_order_preserved"],
    )):
        raise ValueError("strict MJCF canonicalization changed an unapproved XML field")

    normalized_xml = ET.tostring(canonical_root, encoding="unicode")
    canonical_xml = ET.canonicalize(xml_data=normalized_xml, with_comments=True)
    audit_summary["canonicalization_applied"] = True
    return canonical_xml, audit_summary


def source_inventory() -> dict[str, Any]:
    files = {}
    missing = []
    for path in SOURCE_PATHS:
        if path.is_file():
            files[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"source inputs missing: {missing}")
    asset_inventory_path = Path(
        "/data/artifacts/by-project/malecns-gpu/engineering/"
        "flygym-dnp15-ap-online-pilot-20261007-v3/captures/positive60/upstream_asset_hashes.json"
    )
    frozen_assets = json.loads(asset_inventory_path.read_text())
    verified_asset_count = 0
    for logical_name, record in sorted(frozen_assets.items()):
        asset_path = Path(record["physical_path"])
        if not asset_path.is_file():
            raise FileNotFoundError(f"frozen model asset missing: {asset_path}")
        actual_sha = sha256(asset_path)
        if actual_sha != record["sha256"]:
            raise RuntimeError(
                f"model asset hash mismatch for {logical_name}: "
                f"expected {record['sha256']} observed {actual_sha}"
            )
        files[f"model_asset::{logical_name}"] = {
            "path": str(asset_path),
            "sha256": actual_sha,
            "bytes": asset_path.stat().st_size,
        }
        verified_asset_count += 1
    return {
        "schema": "flygym-tethered-pilot-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hash_algorithm": "SHA-256",
        "verified_model_asset_count": verified_asset_count,
        "files": files,
        "inventory_self_hash": None,
    }


def set_cpu_environment() -> None:
    # Set before importing MuJoCo/OpenGL. The pilot has no visual renderer.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", "/data/models/flygym/assets")
    os.environ.setdefault("MPLCONFIGDIR", "/data/cache/flygym/matplotlib")
    current = os.environ.get("PYTHONPATH", "")
    paths = [str(SCRIPT_DIR)] + ([current] if current else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)


def package_versions() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for package in ("flygym", "mujoco", "numpy", "scipy"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "unavailable"
    return out


def authorization_check(
    authorization_path: Path,
    preflight_path: Path,
    out_path: Path,
    *,
    p: float,
    d: float,
    duration_s: float,
    seed: int,
) -> dict[str, Any]:
    out_root = out_path.resolve()
    if authorization_path.resolve() != (out_root / "authorization.json").resolve():
        raise PermissionError("authorization must be authorization.json inside --out")
    if preflight_path.resolve() != (out_root / "compile-evidence.json").resolve():
        raise PermissionError("compile evidence must be compile-evidence.json inside --out")
    source_freeze_path = out_root / "source_freeze.json"
    if not source_freeze_path.is_file() or not preflight_path.is_file():
        raise PermissionError("compile evidence and source freeze must be present inside --out")
    evidence = json.loads(preflight_path.read_text())
    source_freeze = json.loads(source_freeze_path.read_text())
    freeze_sha = sha256(source_freeze_path)
    if evidence.get("bundle_root") != str(out_root):
        raise PermissionError("compile evidence is not bound to this --out bundle")
    if source_freeze.get("bundle_root") != str(out_root):
        raise PermissionError("source freeze is not bound to this --out bundle")
    if evidence.get("source_freeze_sha256") != freeze_sha:
        raise PermissionError("compile evidence does not match this bundle's source freeze")
    auth = json.loads(authorization_path.read_text())
    if auth.get("status") != "authorized":
        raise PermissionError("run mode requires status=authorized in the root-review JSON")
    preflight_sha = sha256(preflight_path)
    if auth.get("compile_evidence_sha256") != preflight_sha:
        raise PermissionError("authorization is not bound to this compile evidence hash")
    if evidence.get("schema") != "flygym-tethered-ball-compile-evidence-v2":
        raise PermissionError("compile evidence does not use the reviewed v2 canonicalization schema")
    xml_evidence = evidence.get("compiled_world_xml", {})
    canonical_sha_expected = xml_evidence.get("canonical_sha256")
    if not canonical_sha_expected:
        raise PermissionError("compile evidence has no canonical compiled-world XML SHA-256")
    canonicalization = xml_evidence.get("canonicalization", {})
    if canonicalization.get("rule_id") != CANONICALIZATION_RULE_ID:
        raise PermissionError("compile evidence canonicalization rule does not match this runner")
    for kind, path_key, hash_key in (
        ("raw", "raw_path", "raw_sha256"),
        ("canonical", "canonical_path", "canonical_sha256"),
    ):
        relative_path = xml_evidence.get(path_key)
        if not relative_path or Path(relative_path).name != relative_path:
            raise PermissionError(f"compile evidence has an invalid {kind} XML path")
        artifact_path = out_root / relative_path
        if not artifact_path.is_file() or sha256(artifact_path) != xml_evidence.get(hash_key):
            raise PermissionError(f"compile evidence {kind} XML artifact is absent or has changed")
    if auth.get("canonical_world_xml_sha256") != canonical_sha_expected:
        raise PermissionError("authorization is not bound to the reviewed canonical world XML hash")
    if auth.get("canonicalization_rule_id") != CANONICALIZATION_RULE_ID:
        raise PermissionError("authorization canonicalization rule does not match this runner")
    expected = requested_condition(p=p, d=d, duration_s=duration_s, seed=seed)
    if evidence.get("requested_condition") != expected:
        raise PermissionError("requested condition differs from the reviewed compile evidence")
    if auth.get("condition") != expected:
        raise PermissionError(f"authorization condition mismatch: expected {expected}")
    if auth.get("run_id") != expected["run_id"]:
        raise PermissionError("authorization run_id does not match the compiled condition")
    if not auth.get("authorized_by"):
        raise PermissionError("authorization must identify the reviewer")
    return {
        "authorization_sha256": sha256(authorization_path),
        "compile_evidence_sha256": preflight_sha,
        "source_freeze_sha256": freeze_sha,
        "canonical_world_xml_sha256": canonical_sha_expected,
        "canonicalization_rule_id": CANONICALIZATION_RULE_ID,
        "authorized_by": auth["authorized_by"],
        "condition": expected,
        "run_id": expected["run_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compile-only", "run"), default="compile-only")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--P", type=float, default=0.30)
    parser.add_argument("--d", type=float, default=0.0)
    parser.add_argument("--authorization-json", type=Path)
    parser.add_argument("--preflight-evidence", type=Path)
    args = parser.parse_args()
    set_cpu_environment()

    if args.P < 0 or args.P > 1 or abs(args.d) > 1:
        parser.error("P must be in [0,1] and d must be in [-1,1]")
    if args.mode == "compile-only" and args.out.exists():
        allowed_existing = {"command-plan.md"}
        unexpected = [p.name for p in args.out.iterdir() if p.name not in allowed_existing]
        if unexpected:
            parser.error(f"refusing to overwrite existing preflight outputs: {unexpected}")
    if args.mode == "run":
        if args.authorization_json is None or args.preflight_evidence is None:
            parser.error("run mode requires --authorization-json and --preflight-evidence")
        if not args.preflight_evidence.is_file():
            parser.error("preflight evidence path does not exist")
        authorization = authorization_check(
            args.authorization_json,
            args.preflight_evidence,
            args.out,
            p=args.P,
            d=args.d,
            duration_s=args.duration_s,
            seed=args.seed,
        )
    else:
        authorization = None

    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "compile-only":
        frozen_sources = source_inventory()
        frozen_sources["bundle_root"] = str(args.out.resolve())
        write_json(args.out / "source_freeze.json", frozen_sources)
    else:
        frozen_source_path = args.out / "source_freeze.json"
        if not frozen_source_path.is_file():
            parser.error("run mode requires the existing compile-preflight source_freeze.json")
        frozen_sources = json.loads(frozen_source_path.read_text())
        if source_inventory()["files"] != frozen_sources.get("files"):
            parser.error("current sources differ from compile-preflight source freeze")
    started = time.monotonic()

    # Import only after CUDA and rendering variables are set.
    from flygym_tethered_treadmill_session_v1 import TetheredBallSessionV1, TetheredPilotConfig

    config = TetheredPilotConfig(
        duration_s=args.duration_s,
        seed=args.seed,
        leg_power_p=args.P,
        differential_ap_d=args.d,
    )
    session = TetheredBallSessionV1(config)
    try:
        if args.mode == "compile-only":
            if session._native_samples != 0 or session.time_s != 0.0:
                raise RuntimeError("compile-only construction unexpectedly sampled/advanced physics")
            evidence = session.preflight_snapshot()
            condition = requested_condition(
                p=args.P,
                d=args.d,
                duration_s=args.duration_s,
                seed=args.seed,
            )
            if condition["run_id"] != session.run_id:
                raise RuntimeError("generated run_id differs from the session config")
            raw_xml = session.world.mjcf_root.to_xml()
            xml_path = args.out / "compiled_world.xml"
            raw_xml_bytes = raw_xml.encode("utf-8")
            xml_path.write_bytes(raw_xml_bytes)
            canonical_xml, canonicalization = canonicalize_world_xml(raw_xml)
            canonical_path = args.out / "compiled_world.canonical.xml"
            canonical_xml_bytes = canonical_xml.encode("utf-8")
            canonical_path.write_bytes(canonical_xml_bytes)
            evidence["compiled_world_xml"] = {
                "raw_path": xml_path.name,
                "raw_sha256": sha256(xml_path),
                "raw_bytes": xml_path.stat().st_size,
                "canonical_path": canonical_path.name,
                "canonical_sha256": sha256(canonical_path),
                "canonical_bytes": canonical_path.stat().st_size,
                "canonicalization": canonicalization,
            }
            evidence["schema"] = "flygym-tethered-ball-compile-evidence-v2"
            evidence["bundle_root"] = str(args.out.resolve())
            evidence["requested_condition"] = condition
            evidence["run_id"] = condition["run_id"]
            evidence["source_freeze_sha256"] = sha256(args.out / "source_freeze.json")
            evidence["package_versions"] = package_versions()
            evidence["runtime_environment"] = {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
                "renderer_created": False,
                "neural_worker_started": False,
            }
            evidence["resource_plan"] = {
                "memory_max": "3G",
                "cpu_quota": "200%",
                "runtime_max_s": 600,
                "enforced_by_this_python_process": False,
                "launch_contract": "systemd-run --user with the limits recorded in command-plan.md",
            }
            evidence["wall_time_s"] = time.monotonic() - started
            evidence["scope_note"] = (
                "Model/world instantiated and compiled for review; mj_forward only. "
                "No Simulation.step, mj_step, controller tick, rendering, or neural execution."
            )
            write_json(args.out / "compile-evidence.json", evidence)
            post_source_files = source_inventory()["files"]
            if post_source_files != frozen_sources["files"]:
                raise RuntimeError("source files changed between freeze and compile evidence")
            write_json(
                args.out / "source_postcheck.json",
                {
                    "all_source_hashes_unchanged": True,
                    "source_files": post_source_files,
                    "inventory_self_hash": None,
                },
            )
            if session.time_s != 0.0 or session._native_samples != 0 or session._physics_step_count != 0:
                raise RuntimeError("preflight modified simulation time or native trace count")
            print(json.dumps({
                "status": "compile_preflight_complete_no_mj_step",
                "out": str(args.out),
                "run_id": condition["run_id"],
                "raw_world_xml_sha256": sha256(xml_path),
                "canonical_world_xml_sha256": sha256(canonical_path),
                "compile_evidence_sha256": sha256(args.out / "compile-evidence.json"),
                "source_freeze_sha256": sha256(args.out / "source_freeze.json"),
                "model_time_s": session.time_s,
                "physics_step_count": session._physics_step_count,
                "root_has_mocap": bool(session.model.body_mocapid[session.root_body_id] >= 0),
                "position_actuator_count": len(session.position_actuator_ids),
            }, sort_keys=True))
            return 0

        # This branch is unreachable without a root review bound to the exact
        # compile bundle, source freeze, condition, seed, and generated run_id.
        preflight = json.loads(args.preflight_evidence.read_text())
        if authorization["run_id"] != session.run_id:
            raise PermissionError("authorized run_id does not match the instantiated session config")
        reviewed_xml = preflight.get("compiled_world_xml", {})
        compiled_xml_expected = reviewed_xml.get("canonical_sha256")
        if not compiled_xml_expected:
            raise PermissionError("compile evidence has no canonical compiled-world XML SHA-256")

        # Preserve this process's raw and normalized serialization before the
        # strict gate. Reviewers can inspect the rejected bytes even if the
        # canonical hash or a source/asset check fails.
        precheck_dir = args.out / "run-precheck-v1"
        precheck_dir.mkdir(parents=False, exist_ok=False)
        live_raw_xml = session.world.mjcf_root.to_xml()
        live_raw_path = precheck_dir / "actual_world.raw.xml"
        live_raw_path.write_bytes(live_raw_xml.encode("utf-8"))
        live_canonical_xml, live_canonicalization = canonicalize_world_xml(live_raw_xml)
        live_canonical_path = precheck_dir / "actual_world.canonical.xml"
        live_canonical_path.write_bytes(live_canonical_xml.encode("utf-8"))
        live_raw_sha = sha256(live_raw_path)
        live_canonical_sha = sha256(live_canonical_path)
        post_construct_inventory = source_inventory()
        source_files_match = post_construct_inventory["files"] == frozen_sources.get("files")
        expected_condition = requested_condition(
            p=args.P,
            d=args.d,
            duration_s=args.duration_s,
            seed=args.seed,
        )
        precheck_record = {
            "schema": "flygym-tethered-ball-run-precheck-v1",
            "status": "passed" if source_files_match and live_canonical_sha == compiled_xml_expected else "rejected",
            "condition": expected_condition,
            "run_id": session.run_id,
            "authorization_sha256": authorization["authorization_sha256"],
            "compile_evidence_sha256": authorization["compile_evidence_sha256"],
            "source_freeze_sha256": authorization["source_freeze_sha256"],
            "normalization_rule_id": CANONICALIZATION_RULE_ID,
            "reviewed_canonical_xml_sha256": compiled_xml_expected,
            "actual_raw_xml_sha256": live_raw_sha,
            "actual_canonical_xml_sha256": live_canonical_sha,
            "raw_xml_path": live_raw_path.name,
            "canonical_xml_path": live_canonical_path.name,
            "source_and_model_asset_checks": {
                "files_match_frozen_inventory": source_files_match,
                "source_freeze_entry_count": len(frozen_sources.get("files", {})),
                "model_assets_verified": post_construct_inventory["verified_model_asset_count"],
                "source_freeze_bundle_root": frozen_sources.get("bundle_root"),
            },
            "canonicalization_audit": live_canonicalization,
            "controller_ticks_before_gate": int(session._control_samples),
            "native_samples_before_gate": int(session._native_samples),
            "physics_steps_before_gate": int(session._physics_step_count),
            "simulation_time_s_before_gate": session.time_s,
        }
        write_json(precheck_dir / "precheck.json", precheck_record)
        if not source_files_match:
            raise PermissionError(
                f"run precheck source/model-asset inventory differs; details: {precheck_dir / 'precheck.json'}"
            )
        if live_canonical_sha != compiled_xml_expected:
            raise PermissionError(
                "newly constructed world canonical XML differs from reviewed compile evidence: "
                f"expected {compiled_xml_expected}, observed {live_canonical_sha}; "
                f"raw and canonical XML saved in {precheck_dir}"
            )
        if session.time_s != 0.0 or session._native_samples != 0 or session._physics_step_count != 0:
            raise RuntimeError("run authorization gate must precede all physics advancement")
        run_dir = args.out / "run"
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "authorization_record.json", authorization)
        ticks_path = run_dir / "control_ticks.jsonl"
        tick_started = time.monotonic()
        signal = [args.P, args.P]
        with ticks_path.open("x", encoding="utf-8", buffering=1) as log:
            for tick in range(session.n_control_expected):
                item = session.step_control_tick(signal)
                item["tick_index"] = tick
                log.write(json.dumps(item, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
        session.save_trajectory(run_dir / "trajectory.npz")
        post_sources = source_inventory()
        if post_sources["files"] != frozen_sources["files"]:
            raise RuntimeError("source hashes changed during the run")
        summary = {
            "status": "complete",
            "run_id": authorization["run_id"],
            "config": asdict(config),
            "authorization": authorization,
            "physics_step_count": session._physics_step_count,
            "control_tick_count": session.n_control_expected,
            "native_sample_count": session._native_samples,
            "final_time_s": session.time_s,
            "physics_wall_s": time.monotonic() - tick_started,
            "trajectory_sha256": sha256(run_dir / "trajectory.npz"),
            "contacts_sha256": sha256(run_dir / "contacts.npz"),
            "control_ticks_sha256": sha256(ticks_path),
            "reviewed_canonical_world_xml_sha256": compiled_xml_expected,
            "actual_raw_world_xml_sha256": live_raw_sha,
            "actual_canonical_world_xml_sha256": live_canonical_sha,
            "normalization_rule_id": CANONICALIZATION_RULE_ID,
            "authority": {
                "root_qpos_writes": int(np.max(session.trace["root_qpos_write_count"][:session._native_samples])),
                "root_qvel_writes": int(np.max(session.trace["root_qvel_write_count"][:session._native_samples])),
                "root_external_wrench_max_abs": float(np.max(np.abs(session.trace["root_external_wrench_native"][:session._native_samples]))),
                "ball_external_wrench_max_abs": float(np.max(np.abs(session.trace["ball_external_wrench_native"][:session._native_samples]))),
                "qfrc_applied_max_abs": float(np.max(np.abs(session.trace["qfrc_applied_native"][:session._native_samples]))),
            },
            "interpretation_boundary": "ball angular velocity is a mechanics response; no biological steering claim",
        }
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "source_inventory.json", post_sources)
        print(json.dumps({"status": "run_complete", "run_dir": str(run_dir), **summary}, sort_keys=True))
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
