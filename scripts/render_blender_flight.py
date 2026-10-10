#!/usr/bin/env python3
"""Render a 3D Blender presentation replay from digital-fly telemetry.

Run inside Blender, for example::

    blender --background --python scripts/render_blender_flight.py -- \
      --trajectory artifacts/run/trajectory.json \
      --scene config/realistic_scene.json \
      --out artifacts/run/presentation/flight_presentation.mp4 \
      --context-out artifacts/run/presentation/flight_context.mp4 \
      --frames-dir artifacts/run/presentation/frames

The scene manifest owns all model paths and asset provenance. ``replay.bindings``
maps the trajectory fly and stimulus IDs to manifest ``object_id`` values. The
script only changes presentation transforms and visibility; it does not alter
or regenerate simulation telemetry. The resulting scene is a visual
reconstruction of a digital simulation, not a biological or sensor-render
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import bpy  # type: ignore[import-not-found]
import bmesh  # type: ignore[import-not-found]
from mathutils import Euler, Matrix, Quaternion, Vector  # type: ignore[import-not-found]


SCRIPT_VERSION = "1.0.0"
SUPPORTED_SCENE_FORMATS = {".blend", ".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"}


class ReplayError(RuntimeError):
    """Raised when replay inputs do not meet the render contract."""


def _json_file(path: Path, description: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise ReplayError(f"{description} not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReplayError(f"{description} must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest_path(manifest_path: Path, raw_value: Any, asset_root: Any = None) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ReplayError("manifest file path must be a non-empty string")
    path = Path(os.path.expandvars(raw_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates: list[Path] = []
    environment_root = os.environ.get("MALECNS_ASSET_ROOT")
    if environment_root:
        candidates.append(Path(os.path.expandvars(environment_root)).expanduser() / path)
    if isinstance(asset_root, str) and asset_root.strip():
        root = Path(os.path.expandvars(asset_root)).expanduser()
        if not root.is_absolute():
            root = manifest_path.parent / root
        candidates.append(root / path)
    candidates.append(manifest_path.parent / path)
    resolved = [candidate.resolve() for candidate in candidates]
    return next((candidate for candidate in resolved if candidate.is_file()), resolved[0])


def _vector(value: Any, name: str, length: int = 3) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ReplayError(f"{name} must contain {length} numbers")
    try:
        parts = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayError(f"{name} must contain {length} numbers") from exc
    if len(parts) != length or not all(math.isfinite(item) for item in parts):
        raise ReplayError(f"{name} must contain {length} finite numbers")
    return parts


def _finite(value: Any, name: str, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ReplayError(f"{name} must be a finite number")
    return result


def _asset_quaternion(transform: Mapping[str, Any]) -> Quaternion:
    raw = transform.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0))
    x, y, z, w = _vector(raw, "asset.transform.rotation_xyzw", 4)
    quaternion = Quaternion((w, x, y, z))
    if quaternion.magnitude <= 1.0e-12:
        raise ReplayError("asset.transform.rotation_xyzw must not be zero")
    quaternion.normalize()
    return quaternion


def _trajectory_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("trajectory", "rows", "samples", "telemetry", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            if not value:
                break
            if not all(isinstance(row, Mapping) for row in value):
                raise ReplayError(f"trajectory.{key} must contain only objects")
            return list(value)
    raise ReplayError("trajectory must contain a non-empty list under 'trajectory' or 'rows'")


def _row_contexts(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = [row]
    for parent_key in ("body", "world", "timestamp"):
        nested = row.get(parent_key)
        if isinstance(nested, Mapping):
            result.append(nested)
            if isinstance(nested.get("body"), Mapping):
                result.append(nested["body"])
    return result


def _find(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for context in _row_contexts(row):
        for key in keys:
            if key in context and context[key] is not None:
                return context[key]
    return None


def _quaternion_from_rpy(value: Any) -> Quaternion:
    roll, pitch, yaw = _vector(value, "orientation_rad")
    q = Euler((roll, pitch, yaw), "XYZ").to_quaternion()
    q.normalize()
    return q


def _normalize_quaternion(value: Any) -> Quaternion:
    # The trajectory contract stores scalar-first wxyz quaternions.
    w, x, y, z = _vector(value, "orientation_quat", 4)
    result = Quaternion((w, x, y, z))
    if result.magnitude <= 1.0e-12:
        raise ReplayError("orientation_quat must not be zero")
    result.normalize()
    return result


def _row_time(row: Mapping[str, Any], field: str, index: int, fallback_dt: float) -> float:
    raw = _find(row, (field,))
    if raw is None:
        raw = _find(row, ("time_s", "t_s", "timestamp_s", "time", "t"))
    result = _finite(raw, f"trajectory row {index} timestamp")
    if result is None:
        result = index * fallback_dt
    if result < 0:
        raise ReplayError(f"trajectory row {index} has negative time {result}")
    return result


def _load_body_samples(
    rows: Sequence[Mapping[str, Any]], time_field: str, fallback_dt: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        time_s = _row_time(row, time_field, index, fallback_dt)
        position_value = _find(row, ("position_m", "position", "pos_m", "pos", "xyz"))
        if position_value is None:
            raise ReplayError(f"trajectory row {index} is missing position_m")
        position = _vector(position_value, f"trajectory row {index}.position_m")
        quaternion_value = _find(row, ("orientation_quat", "quaternion", "quat"))
        if quaternion_value is not None:
            quaternion = _normalize_quaternion(quaternion_value)
        else:
            rpy = _find(row, ("orientation_rad", "orientation_rpy_rad", "rpy_rad"))
            if rpy is None:
                rpy = (
                    _find(row, ("roll", "roll_rad")) or 0.0,
                    _find(row, ("pitch", "pitch_rad")) or 0.0,
                    _find(row, ("yaw", "yaw_rad", "heading_rad")) or 0.0,
                )
            quaternion = _quaternion_from_rpy(rpy)
        samples.append({"time_s": time_s, "position": Vector(position), "rotation": quaternion})
    samples.sort(key=lambda sample: sample["time_s"])
    unique: list[dict[str, Any]] = []
    for sample in samples:
        if unique and abs(sample["time_s"] - unique[-1]["time_s"]) <= 1.0e-12:
            unique[-1] = sample
        else:
            unique.append(sample)
    if not unique:
        raise ReplayError("trajectory contains no renderable body samples")
    return unique


def _stimulus_rows(
    rows: Sequence[Mapping[str, Any]], time_field: str, fallback_dt: float,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        states = row.get("stimulus_state")
        if states is None and isinstance(row.get("world"), Mapping):
            states = row["world"].get("stimulus_state")
        if isinstance(states, Mapping):
            entries = list(states.items())
        elif isinstance(states, list):
            entries = [(str(item.get("id", i)), item) for i, item in enumerate(states) if isinstance(item, Mapping)]
        else:
            continue
        row_time = _row_time(row, time_field, index, fallback_dt)
        for name, state in entries:
            if not isinstance(state, Mapping):
                continue
            position_value = state.get("position_m", state.get("position"))
            if position_value is None:
                continue
            sample = {
                "time_s": row_time,
                "position": Vector(_vector(position_value, f"stimulus {name}.position_m")),
                "active": bool(state.get("active", state.get("renderable", True))),
                "renderable": bool(state.get("renderable", True)),
                "size": None,
                "direction": None,
            }
            if state.get("size_m") is not None:
                sample["size"] = Vector(_vector(state["size_m"], f"stimulus {name}.size_m"))
            if state.get("direction_m") is not None:
                sample["direction"] = Vector(_vector(state["direction_m"], f"stimulus {name}.direction_m"))
            result.setdefault(str(name), []).append(sample)
    for states in result.values():
        states.sort(key=lambda sample: sample["time_s"])
        unique: list[dict[str, Any]] = []
        for sample in states:
            if unique and abs(sample["time_s"] - unique[-1]["time_s"]) <= 1.0e-12:
                unique[-1] = sample
            else:
                unique.append(sample)
        states[:] = unique
    return result


def _lerp_vector(a: Vector, b: Vector, factor: float) -> Vector:
    return a.lerp(b, factor)


def _sample(samples: Sequence[dict[str, Any]], time_s: float) -> dict[str, Any]:
    if len(samples) == 1 or time_s <= samples[0]["time_s"]:
        return dict(samples[0])
    if time_s >= samples[-1]["time_s"]:
        return dict(samples[-1])
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        middle = (lo + hi) // 2
        if samples[middle]["time_s"] <= time_s:
            lo = middle
        else:
            hi = middle
    left, right = samples[lo], samples[hi]
    interval = right["time_s"] - left["time_s"]
    factor = 0.0 if interval <= 1.0e-15 else (time_s - left["time_s"]) / interval
    result: dict[str, Any] = {"time_s": time_s}
    for key in ("position", "size", "direction"):
        a, b = left.get(key), right.get(key)
        result[key] = None if a is None or b is None else _lerp_vector(a, b, factor)
    result["rotation"] = left.get("rotation")
    if left.get("rotation") is not None and right.get("rotation") is not None:
        result["rotation"] = left["rotation"].slerp(right["rotation"], factor)
    result["active"] = left.get("active", True) if factor < 1.0 else right.get("active", True)
    result["renderable"] = left.get("renderable", True) if factor < 1.0 else right.get("renderable", True)
    return result


def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_blend_asset(path: Path, collection: bpy.types.Collection) -> list[bpy.types.Object]:
    with bpy.data.libraries.load(str(path), link=False) as (source, destination):
        destination.objects = list(source.objects)
    loaded = [obj for obj in destination.objects if obj is not None]
    for obj in loaded:
        collection.objects.link(obj)
    return loaded


def _import_asset(path: Path, collection: bpy.types.Collection) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix == ".blend":
        return _import_blend_asset(path, collection)
    if suffix in (".glb", ".gltf"):
        try:
            bpy.ops.import_scene.gltf(filepath=str(path))
        except Exception as exc:
            raise ReplayError(f"cannot import glTF asset {path}: {exc}") from exc
    elif suffix == ".fbx":
        try:
            bpy.ops.import_scene.fbx(filepath=str(path))
        except Exception as exc:
            raise ReplayError(f"cannot import FBX asset {path}: {exc}") from exc
    elif suffix == ".obj":
        try:
            if hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=str(path))
            else:
                bpy.ops.import_scene.obj(filepath=str(path))
        except Exception as exc:
            raise ReplayError(f"cannot import OBJ asset {path}: {exc}") from exc
    elif suffix in (".stl", ".ply"):
        operator = getattr(bpy.ops.wm, f"{suffix[1:]}_import", None)
        if operator is None:
            operator = getattr(bpy.ops.import_mesh, suffix[1:], None)
        if operator is None:
            raise ReplayError(f"this Blender build has no {suffix[1:].upper()} importer for {path}")
        try:
            operator(filepath=str(path))
        except Exception as exc:
            raise ReplayError(f"cannot import {suffix[1:].upper()} asset {path}: {exc}") from exc
    else:
        raise ReplayError(f"unsupported scene asset format {suffix!r}: {path}")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in imported:
        for old_collection in list(obj.users_collection):
            old_collection.objects.unlink(obj)
        collection.objects.link(obj)
    return imported


def _asset_path(
    manifest_path: Path, record: Mapping[str, Any], manifest_asset_root: Any = None,
) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ReplayError(f"asset {record.get('object_id', '<unknown>')!r} needs a relative or absolute path")
    path = _resolve_manifest_path(manifest_path, raw, record.get("asset_root", manifest_asset_root))
    if not path.is_file():
        raise ReplayError(f"scene asset not found: {path} (also checked asset_root and MALECNS_ASSET_ROOT)")
    if path.suffix.lower() not in SUPPORTED_SCENE_FORMATS:
        raise ReplayError(f"unsupported scene asset format {path.suffix!r}: {path}")
    return path


def _verify_asset_dependency_hashes(path: Path, record: Mapping[str, Any]) -> dict[str, str]:
    integrity = record.get("integrity", {})
    if not isinstance(integrity, Mapping):
        return {}
    declared = integrity.get("files", {})
    if not isinstance(declared, Mapping):
        raise ReplayError(f"asset {record.get('object_id')!r}.integrity.files must be an object")
    actual_hashes: dict[str, str] = {}
    root = path.parent.resolve()
    for raw_relative, expected in declared.items():
        relative = Path(str(raw_relative))
        dependency = (root / relative).resolve()
        try:
            dependency.relative_to(root)
        except ValueError as exc:
            raise ReplayError(f"asset dependency path escapes its directory: {raw_relative!r}") from exc
        if not dependency.is_file():
            raise ReplayError(f"declared asset dependency not found: {dependency}")
        actual = _sha256(dependency)
        expected_hash = str(expected).lower()
        if actual != expected_hash:
            raise ReplayError(
                f"asset {record.get('object_id')!r} dependency SHA-256 mismatch for {raw_relative}: "
                f"expected {expected_hash}, got {actual}"
            )
        actual_hashes[str(raw_relative)] = actual
    return actual_hashes


def _manifest_transform(record: Mapping[str, Any]) -> tuple[Vector, Quaternion, Vector]:
    transform = record.get("transform", {})
    if not isinstance(transform, Mapping):
        raise ReplayError(f"asset {record.get('object_id')!r}.transform must be an object")
    location = Vector(_vector(transform.get("translation_m", (0.0, 0.0, 0.0)), "asset.transform.translation_m"))
    rotation = _asset_quaternion(transform)
    raw_scale = transform.get("scale", (1.0, 1.0, 1.0))
    if isinstance(raw_scale, (int, float)):
        scale = Vector((float(raw_scale),) * 3)
    else:
        scale = Vector(_vector(raw_scale, "asset.transform.scale"))
    unit_scale = _finite(record.get("asset_unit_scale_to_m", 1.0), "asset.asset_unit_scale_to_m", 1.0)
    assert unit_scale is not None
    if unit_scale <= 0:
        raise ReplayError(f"asset {record.get('object_id')!r}.asset_unit_scale_to_m must be positive")
    scale *= unit_scale
    if any(not math.isfinite(value) or abs(value) <= 1.0e-12 for value in scale):
        raise ReplayError(f"asset {record.get('object_id')!r}.transform.scale must be finite and non-zero")
    return location, rotation, scale


def _make_root(
    object_id: str,
    collection: bpy.types.Collection,
    imported: Sequence[bpy.types.Object],
    transform: tuple[Vector, Quaternion, Vector],
) -> bpy.types.Object:
    root = bpy.data.objects.new(f"ReplayRoot::{object_id}", None)
    collection.objects.link(root)
    # Imported object transforms stay local to an identity root. The manifest
    # transform is applied once at the root and can then be replaced by replay
    # coordinates without accumulating a static offset.
    root.location = (0.0, 0.0, 0.0)
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    root.scale = (1.0, 1.0, 1.0)
    # Force Blender to evaluate freshly assigned mesh transforms before
    # preserving them as child-local transforms. Without this update, newly
    # generated objects can still report identity matrix_world here and lose
    # their intended millimetre-scale location/scale when parented.
    bpy.context.view_layer.update()
    for obj in imported:
        current_world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_world = current_world
    bpy.context.view_layer.update()
    root.location, root.rotation_quaternion, root.scale = transform
    bpy.context.view_layer.update()
    return root


def _load_assets(manifest_path: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records = manifest.get("assets", [])
    if not isinstance(records, list):
        raise ReplayError("scene manifest 'assets' must be a list")
    objects: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ReplayError(f"scene asset {index} must be an object")
        object_id = str(record.get("object_id", "")).strip()
        if not object_id:
            raise ReplayError(f"scene asset {index} is missing object_id")
        if object_id in objects:
            raise ReplayError(f"scene manifest contains duplicate object_id {object_id!r}")
        path = _asset_path(manifest_path, record, manifest.get("asset_root"))
        expected_sha = None
        integrity = record.get("integrity")
        if isinstance(integrity, Mapping) and integrity.get("sha256"):
            expected_sha = str(integrity["sha256"]).lower()
        actual_sha = _sha256(path)
        if expected_sha and expected_sha != actual_sha:
            raise ReplayError(f"asset {object_id!r} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
        dependency_hashes = _verify_asset_dependency_hashes(path, record)
        collection = bpy.data.collections.new(f"Asset::{object_id}")
        bpy.context.scene.collection.children.link(collection)
        imported = _import_asset(path, collection)
        if not imported:
            raise ReplayError(f"asset import created no Blender objects: {path}")
        location, rotation, scale = _manifest_transform(record)
        root = _make_root(object_id, collection, imported, (location, rotation, scale))
        role = str(record.get("role", "visual"))
        if role == "collision":
            for obj in imported:
                obj.hide_render = True
        objects[object_id] = {
            "root": root,
            "children": imported,
            "record": dict(record),
            "role": role,
            "base_location": location.copy(),
            "base_rotation": rotation.copy(),
            "base_scale": scale.copy(),
            "path": path,
            "actual_sha256": actual_sha,
            "dependency_sha256": dependency_hashes,
        }
        provenance.append({
            "object_id": object_id,
            "role": role,
            "path": str(path),
            "sha256": actual_sha,
            "dependency_sha256": dependency_hashes,
            "expected_sha256": expected_sha,
            "integrity_verified": bool(expected_sha and expected_sha == actual_sha),
            "provenance": record.get("provenance"),
        })
    if not objects:
        raise ReplayError("scene manifest contains no assets")
    return objects, provenance


def _add_fly_ellipsoid(
    collection: bpy.types.Collection, name: str, center: Sequence[float],
    scale: Sequence[float], material: bpy.types.Material,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = tuple(center)
    obj.scale = tuple(scale)
    obj.data.materials.append(material)
    return obj


def _add_fly_line(
    collection: bpy.types.Collection, name: str,
    start: Sequence[float], end: Sequence[float],
    radius_m: float, material: bpy.types.Material,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(f"{name}Curve", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = radius_m
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (*tuple(start), 1.0)
    spline.points[1].co = (*tuple(end), 1.0)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    curve.materials.append(material)
    return obj


def _procedural_fly_asset() -> dict[str, Any]:
    """Build a small native fly rig when a scene manifest has no fly mesh.

    Dimensions are presentation geometry in meters. The generated model is a
    simple visual stand-in; trajectory position and orientation remain the
    recorded values.
    """

    object_id = "generated_fly"
    collection = bpy.data.collections.new("Asset::generated_fly")
    bpy.context.scene.collection.children.link(collection)
    abdomen_mat = _create_material("FlyAbdomen", (0.19, 0.075, 0.035), 0.42)
    thorax_mat = _create_material("FlyThorax", (0.47, 0.25, 0.10), 0.45)
    head_mat = _create_material("FlyHead", (0.24, 0.12, 0.08), 0.4)
    eye_mat = _create_material("FlyEyes", (0.42, 0.035, 0.055), 0.25)
    wing_mat = _create_material("FlyWings", (0.45, 0.56, 0.62), 0.35)
    leg_mat = _create_material("FlyLegs", (0.16, 0.105, 0.065), 0.5)
    children: list[bpy.types.Object] = []

    children.append(_add_fly_ellipsoid(collection, "FlyAbdomen", (-0.00135, 0.0, 0.00002), (0.00155, 0.00040, 0.00034), abdomen_mat))
    children.append(_add_fly_ellipsoid(collection, "FlyThorax", (0.0, 0.0, 0.00003), (0.00068, 0.00052, 0.00043), thorax_mat))
    children.append(_add_fly_ellipsoid(collection, "FlyHead", (0.00078, 0.0, 0.00005), (0.00043, 0.00039, 0.00036), head_mat))
    for side in (-1.0, 1.0):
        children.append(_add_fly_ellipsoid(collection, f"FlyEye_{side:+.0f}", (0.00091, side * 0.000265, 0.00013), (0.00020, 0.000155, 0.00019), eye_mat))
        wing = _add_fly_ellipsoid(collection, f"FlyWing_{side:+.0f}", (-0.00020, side * 0.00055, 0.00048), (0.00120, 0.00032, 0.000040), wing_mat)
        wing.rotation_euler[2] = side * math.radians(13.0)
        children.append(wing)
        # Two-segment antennae and three articulated-looking legs on each side.
        children.append(_add_fly_line(collection, f"FlyAntenna_{side:+.0f}", (0.00100, side * 0.00010, 0.00028), (0.00148, side * 0.00029, 0.00069), 0.000035, leg_mat))
        leg_roots = ((-0.00038, -0.00020, -0.00065), (0.0, side * 0.00036, -0.00073), (0.00034, side * 0.00024, -0.00061))
        leg_tips = ((-0.00102, side * 0.00073, -0.00108), (-0.00002, side * 0.00082, -0.00112), (0.00100, side * 0.00074, -0.00105))
        for leg_index, (start, end) in enumerate(zip(leg_roots, leg_tips)):
            knee = ((start[0] + end[0]) * 0.58, side * 0.00057, -0.00056)
            children.append(_add_fly_line(collection, f"FlyLeg_{side:+.0f}_{leg_index}_upper", start, knee, 0.000040, leg_mat))
            children.append(_add_fly_line(collection, f"FlyLeg_{side:+.0f}_{leg_index}_lower", knee, end, 0.000032, leg_mat))

    identity = (Vector((0.0, 0.0, 0.0)), Quaternion((1.0, 0.0, 0.0, 0.0)), Vector((1.0, 1.0, 1.0)))
    root = _make_root(object_id, collection, children, identity)
    return {
        "root": root,
        "children": children,
        "record": {"object_id": object_id, "role": "visual", "semantic_role": "fly", "generated": True},
        "role": "visual",
        "base_location": identity[0],
        "base_rotation": identity[1],
        "base_scale": identity[2],
        "path": None,
        "actual_sha256": None,
        "generated": True,
    }


def _procedural_palm_asset() -> dict[str, Any]:
    """Create a neutral palm-shaped display mesh when no stimulus asset binds."""

    collection = bpy.data.collections.new("Asset::generated_palm")
    bpy.context.scene.collection.children.link(collection)
    # The outline is a rounded, joined palm-and-fingers silhouette in the XY
    # plane. Its proportions match the recorded low-albedo stimulus envelope.
    outline = [
        (-0.045, -0.014), (-0.034, -0.030), (-0.018, -0.039), (0.000, -0.040),
        (0.012, -0.036), (0.031, -0.031), (0.040, -0.025), (0.044, -0.019),
        (0.044, -0.013), (0.047, -0.008), (0.047, -0.002), (0.045, 0.003),
        (0.048, 0.009), (0.046, 0.015), (0.042, 0.019), (0.043, 0.025),
        (0.039, 0.031), (0.033, 0.034), (0.026, 0.031), (0.019, 0.027),
        (0.011, 0.025), (0.004, 0.030), (-0.008, 0.034), (-0.020, 0.033),
        (-0.034, 0.027), (-0.045, 0.015),
    ]
    thickness = 0.01
    count = len(outline)
    vertices = [(x, y, -thickness) for x, y in outline] + [(x, y, thickness) for x, y in outline]
    faces: list[tuple[int, ...]] = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
    for index in range(count):
        nxt = (index + 1) % count
        faces.append((index, nxt, count + nxt, count + index))
    mesh = bpy.data.meshes.new("RecordedPalmMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("RecordedPalm", mesh)
    collection.objects.link(obj)
    material = _create_material("RecordedPalmMaterial", (0.045, 0.052, 0.047), 0.68)
    mesh.materials.append(material)
    bevel = obj.modifiers.new("SoftPalmEdges", "BEVEL")
    bevel.width = 0.004
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    obj.modifiers.new("PalmWeightedNormals", "WEIGHTED_NORMAL")
    identity = (Vector((0.0, 0.0, 0.0)), Quaternion((1.0, 0.0, 0.0, 0.0)), Vector((1.0, 1.0, 1.0)))
    root = _make_root("generated_palm", collection, [obj], identity)
    return {
        "root": root,
        "children": [obj],
        "record": {"object_id": "generated_palm", "role": "visual", "semantic_role": "palm", "generated": True},
        "role": "visual",
        "base_location": identity[0],
        "base_rotation": identity[1],
        "base_scale": identity[2],
        "path": None,
        "actual_sha256": None,
        "generated": True,
    }


def _create_material(name: str, color: Sequence[float], roughness: float = 0.8) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = (*tuple(color[:3]), 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader:
        shader.inputs["Base Color"].default_value = (*tuple(color[:3]), 1.0)
        shader.inputs["Roughness"].default_value = roughness
    return material


def _create_ground(config: Mapping[str, Any]) -> None:
    ground = config.get("ground_plane")
    if not isinstance(ground, Mapping) or not bool(ground.get("enabled", False)):
        return
    size_m = float(ground.get("size_m", 4.0))
    z_m = float(ground.get("z_m", 0.0))
    if size_m <= 0:
        raise ReplayError("replay.ground_plane.size_m must be positive")
    half = size_m * 0.5
    mesh = bpy.data.meshes.new("ReplayGroundMesh")
    mesh.from_pydata(
        [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)],
        [], [(0, 1, 2, 3)],
    )
    mesh.update()
    plane = bpy.data.objects.new(str(ground.get("object_id", "ReplayGroundPlane")), mesh)
    bpy.context.scene.collection.objects.link(plane)
    plane.location.z = z_m
    color = ground.get("color_rgb", (0.16, 0.19, 0.16))
    plane.data.materials.append(_create_material("ReplayGroundMaterial", color))


def _create_camera(config: Mapping[str, Any]) -> bpy.types.Object:
    camera_cfg = config.get("camera", {})
    if not isinstance(camera_cfg, Mapping):
        camera_cfg = {}
    data = bpy.data.cameras.new("ReplayCameraData")
    camera = bpy.data.objects.new("ReplayCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    data.lens = float(camera_cfg.get("focal_length_mm", camera_cfg.get("lens_mm", 50.0)))
    data.sensor_fit = "AUTO"
    data.clip_start = float(camera_cfg.get("clip_start_m", 0.001))
    data.clip_end = float(camera_cfg.get("clip_end_m", 100.0))
    bpy.context.scene.camera = camera
    return camera


def _create_lighting(
    config: Mapping[str, Any], manifest_path: Path, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    lighting = config.get("lighting", {})
    if not isinstance(lighting, Mapping):
        lighting = {}
    world = bpy.data.worlds.new("ReplayWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (*tuple(lighting.get("ambient_color_rgb", (0.35, 0.4, 0.48))), 1.0)
        background.inputs["Strength"].default_value = float(
            lighting.get("ambient_strength", lighting.get("world_strength", 0.35))
        )
    bpy.context.scene.world = world

    hdri_provenance: dict[str, Any] | None = None
    hdri_value = lighting.get("hdri_path")
    has_hdri = isinstance(hdri_value, str) and bool(hdri_value.strip())
    if has_hdri:
        hdri_path = _resolve_manifest_path(manifest_path, hdri_value, manifest.get("asset_root"))
        if not hdri_path.is_file():
            raise ReplayError(f"configured replay HDRI not found: {hdri_path}")
        expected = lighting.get("hdri_sha256")
        integrity = lighting.get("hdri_integrity")
        if expected is None and isinstance(integrity, Mapping):
            expected = integrity.get("sha256")
        actual = _sha256(hdri_path)
        if expected and str(expected).lower() != actual:
            raise ReplayError(f"HDRI SHA-256 mismatch: expected {expected}, got {actual}")
        texture = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        texture.name = "ReplayEnvironmentHDRI"
        texture.image = bpy.data.images.load(str(hdri_path), check_existing=True)
        world.node_tree.links.new(texture.outputs["Color"], background.inputs["Color"])
        hdri_provenance = {
            "path": str(hdri_path),
            "sha256": actual,
            "expected_sha256": expected,
            "integrity_verified": bool(expected and str(expected).lower() == actual),
        }

    light_specs = lighting.get("lights")
    if not isinstance(light_specs, list) and "key_power_w" in lighting:
        key_power = float(lighting.get("key_power_w", 120.0))
        key_size = float(lighting.get("key_size_m", 0.8))
        light_specs = [{
            "name": "Key",
            "type": str(lighting.get("key_type", "AREA")),
            "location_m": lighting.get("key_location_m", (-0.45, -0.5, 0.8)),
            "target_m": lighting.get("key_target_m", (0.0, 0.0, 0.0)),
            "power_w": key_power,
            "size_m": key_size,
            "color_rgb": lighting.get("key_color_rgb", (1.0, 0.88, 0.74)),
        }, {
            "name": "Fill",
            "type": "AREA",
            "location_m": lighting.get("fill_location_m", (0.4, 0.6, 0.5)),
            "target_m": lighting.get("fill_target_m", (0.0, 0.0, 0.0)),
            "power_w": key_power * float(lighting.get("fill_ratio", 0.35)),
            "size_m": key_size,
            "color_rgb": lighting.get("fill_color_rgb", (0.65, 0.78, 1.0)),
        }]
    elif not isinstance(light_specs, list) and has_hdri:
        light_specs = []
    elif not isinstance(light_specs, list):
        light_specs = [
            {"name": "Key", "type": "AREA", "location_m": (-0.4, -0.3, 0.45), "power_w": 28.0, "size_m": 0.35, "color_rgb": (1.0, 0.88, 0.74)},
            {"name": "Fill", "type": "AREA", "location_m": (0.2, 0.35, 0.28), "power_w": 16.0, "size_m": 0.45, "color_rgb": (0.63, 0.76, 1.0)},
            {"name": "Rim", "type": "AREA", "location_m": (0.5, -0.05, 0.3), "power_w": 22.0, "size_m": 0.28, "color_rgb": (1.0, 1.0, 1.0)},
        ]
    lights: list[bpy.types.Object] = []
    for index, spec in enumerate(light_specs):
        if not isinstance(spec, Mapping):
            raise ReplayError(f"replay.lighting.lights[{index}] must be an object")
        light_type = str(spec.get("type", "AREA")).upper()
        if light_type not in {"AREA", "POINT", "SUN", "SPOT"}:
            raise ReplayError(f"unsupported replay light type {light_type!r}")
        data = bpy.data.lights.new(str(spec.get("name", f"ReplayLight{index}")), type=light_type)
        data.energy = float(spec.get("power_w", spec.get("energy_w", 20.0)))
        if light_type == "AREA":
            data.shape = "DISK"
            data.size = float(spec.get("size_m", 0.25))
        color = spec.get("color_rgb", (1.0, 1.0, 1.0))
        data.color = tuple(float(v) for v in color[:3])
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = _vector(spec.get("location_m", (0.0, 0.0, 0.5)), f"light {data.name}.location_m")
        target = Vector(_vector(spec.get("target_m", (0.0, 0.0, 0.0)), f"light {data.name}.target_m"))
        direction = target - obj.location
        if direction.length > 1.0e-9:
            obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        lights.append(obj)
    return {
        "hdri": hdri_provenance,
        "area_lights_created": len(lights),
        "area_light_names": [light.name for light in lights],
    }


def _configure_render(args: argparse.Namespace, camera_cfg: Mapping[str, Any]) -> None:
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0
    width, height = args.resolution
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.render.fps_base = 1.0
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.engine = args.engine
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = str(camera_cfg.get("view_transform", "AgX"))
    scene.view_settings.exposure = float(camera_cfg.get("exposure_ev", 0.0))
    if args.engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(1, args.samples)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.camera.data.dof.use_dof = False
    scene.render.use_file_extension = True


def _key_linear(obj: bpy.types.Object, data_path: str, frame: int) -> None:
    obj.keyframe_insert(data_path=data_path, frame=frame, group="RecordedReplay")


def _linearize_objects(objects: Iterable[bpy.types.Object]) -> None:
    for obj in objects:
        anim = obj.animation_data
        if not anim or not anim.action:
            continue
        fcurves = getattr(anim.action, "fcurves", None)
        if fcurves is None:
            fcurves = []
            for layer in getattr(anim.action, "layers", ()):
                for strip in getattr(layer, "strips", ()):
                    bags = getattr(strip, "channelbags", ())
                    for bag in bags:
                        fcurves.extend(getattr(bag, "fcurves", ()))
        for fcurve in fcurves:
            for point in fcurve.keyframe_points:
                point.interpolation = "CONSTANT" if fcurve.data_path.endswith("hide_render") else "LINEAR"


def _stimulus_bindings(
    replay_cfg: Mapping[str, Any], states: Mapping[str, Sequence[dict[str, Any]]],
    objects: Mapping[str, dict[str, Any]],
) -> dict[str, str]:
    bindings_cfg = replay_cfg.get("bindings", {})
    if not isinstance(bindings_cfg, Mapping):
        bindings_cfg = {}
    configured = bindings_cfg.get("stimulus_object_ids", {})
    if not isinstance(configured, Mapping):
        raise ReplayError("replay.bindings.stimulus_object_ids must be an object")
    bindings: dict[str, str] = {}
    for stimulus_id in states:
        object_id = configured.get(stimulus_id, configured.get("*", stimulus_id))
        explicitly_bound = stimulus_id in configured or "*" in configured
        if object_id not in objects and stimulus_id.lower().startswith("palm"):
            matches = [
                key for key, asset in objects.items()
                if str(asset["record"].get("semantic_role", "")).lower() in {"stimulus", "palm"}
            ]
            if not explicitly_bound and len(matches) == 1:
                object_id = matches[0]
        if object_id in objects:
            bindings[stimulus_id] = str(object_id)
        else:
            raise ReplayError(
                f"recorded stimulus {stimulus_id!r} has no scene asset binding; add its manifest object_id "
                "to replay.bindings.stimulus_object_ids"
            )
    return bindings


def _animate(
    body_root: bpy.types.Object,
    body_asset: Mapping[str, Any],
    body_samples: Sequence[dict[str, Any]],
    stimulus_samples: Mapping[str, Sequence[dict[str, Any]]],
    stimulus_bindings: Mapping[str, str],
    assets: Mapping[str, dict[str, Any]],
    camera: bpy.types.Object,
    camera_cfg: Mapping[str, Any],
    first_frame: int,
    frame_count: int,
    fps: int,
    time_start: float,
    time_end: float,
) -> list[float]:
    scene = bpy.context.scene
    if camera_cfg.get("offset_m") is not None:
        raw_offset = _vector(camera_cfg["offset_m"], "replay.camera.offset_m")
    else:
        raw_offset = (
            -abs(float(camera_cfg.get("follow_distance_m", 0.06))),
            float(camera_cfg.get("lateral_offset_m", -0.025)),
            float(camera_cfg.get("height_m", 0.03)),
        )
    look_value = camera_cfg.get("look_ahead_m", (0.06, 0.0, 0.0))
    if isinstance(look_value, (int, float)):
        raw_look = (float(look_value), 0.0, 0.0)
    else:
        raw_look = _vector(look_value, "replay.camera.look_ahead_m")
    camera_offset = Vector(raw_offset)
    look_ahead = Vector(raw_look)
    camera_mode = str(camera_cfg.get("mode", "chase")).lower()
    focus_mode = str(camera_cfg.get("focus_mode", "fly")).lower()
    if camera_mode == "scene_overview":
        excluded = {body_asset["root"].name}
        excluded.update(assets[object_id]["root"].name for object_id in stimulus_bindings.values())
        environment_bounds = [
            bounds for asset in assets.values()
            if asset["root"].name not in excluded
            for bounds in [_asset_world_bounds(asset)]
            if bounds is not None
        ]
        if environment_bounds:
            minimum = [min(bounds["minimum_m"][axis] for bounds in environment_bounds) for axis in range(3)]
            maximum = [max(bounds["maximum_m"][axis] for bounds in environment_bounds) for axis in range(3)]
            extent = [maximum[axis] - minimum[axis] for axis in range(3)]
            distance = max(abs(float(camera_cfg.get("follow_distance_m", 3.0))), extent[0] * 1.25, 1.0)
            lateral = float(camera_cfg.get("lateral_offset_m", -max(extent[1] * 2.0, 0.8)))
            height = max(float(camera_cfg.get("height_m", 1.0)), extent[2] * 2.5, extent[1] * 1.5)
            camera_offset = Vector((-distance, lateral, height))
    frame_times: list[float] = []
    animated: list[bpy.types.Object] = [body_root, camera]
    for object_id in stimulus_bindings.values():
        animated.extend(assets[object_id]["children"])
        animated.append(assets[object_id]["root"])
    for offset in range(frame_count):
        frame = first_frame + offset
        time_s = min(time_end, time_start + offset / float(fps))
        frame_times.append(time_s)
        body = _sample(body_samples, time_s)
        pose_rotation: Quaternion = body["rotation"]
        body_root.location = body["position"]
        body_root.rotation_mode = "QUATERNION"
        body_root.rotation_quaternion = pose_rotation @ body_asset["base_rotation"]
        body_root.scale = body_asset["base_scale"]
        _key_linear(body_root, "location", frame)
        _key_linear(body_root, "rotation_quaternion", frame)
        _key_linear(body_root, "scale", frame)

        camera.location = body["position"] + pose_rotation @ camera_offset
        if camera_mode == "fixed":
            fixed_location = camera_cfg.get("position_m", camera.location)
            camera.location = _vector(fixed_location, "replay.camera.position_m")
            focus = Vector(_vector(camera_cfg.get("target_m", body["position"]), "replay.camera.target_m"))
        elif focus_mode in {"lead", "look_ahead"}:
            focus = body["position"] + pose_rotation @ look_ahead
        else:
            # Keep the tracked fly visible even when the camera has a large
            # vertical or lateral offset. Use focus_mode=lead when a shot
            # should bias toward the path ahead.
            focus = body["position"]
        direction = focus - camera.location
        camera.rotation_mode = "QUATERNION"
        if direction.length > 1.0e-9:
            camera.rotation_quaternion = direction.to_track_quat("-Z", "Y")
        _key_linear(camera, "location", frame)
        _key_linear(camera, "rotation_quaternion", frame)

        for stimulus_id, object_id in stimulus_bindings.items():
            samples = stimulus_samples[stimulus_id]
            state = _sample(samples, time_s)
            asset = assets[object_id]
            root = asset["root"]
            root.location = state["position"]
            root.scale = asset["base_scale"].copy()
            if state.get("size") is not None:
                initial_size = next((row.get("size") for row in samples if row.get("size") is not None), None)
                if initial_size is not None:
                    ratios = Vector((1.0, 1.0, 1.0))
                    for axis in range(3):
                        if abs(initial_size[axis]) > 1.0e-12:
                            ratios[axis] = max(0.0, state["size"][axis] / initial_size[axis])
                    root.scale = Vector((
                        asset["base_scale"].x * ratios.x,
                        asset["base_scale"].y * ratios.y,
                        asset["base_scale"].z * ratios.z,
                    ))
            direction_value = state.get("direction")
            if direction_value is not None and direction_value.length > 1.0e-9:
                direction_rotation = direction_value.normalized().to_track_quat("X", "Z")
                root.rotation_mode = "QUATERNION"
                root.rotation_quaternion = direction_rotation @ asset["base_rotation"]
            else:
                root.rotation_mode = "QUATERNION"
                root.rotation_quaternion = asset["base_rotation"]
            visible = bool(state.get("active", True) and state.get("renderable", True))
            root.hide_render = not visible
            _key_linear(root, "location", frame)
            _key_linear(root, "rotation_quaternion", frame)
            _key_linear(root, "scale", frame)
            _key_linear(root, "hide_render", frame)
            for child in asset["children"]:
                if asset["role"] == "collision":
                    child.hide_render = True
                else:
                    child.hide_render = not visible
                _key_linear(child, "hide_render", frame)
    _linearize_objects(animated)
    scene.frame_start = first_frame
    scene.frame_end = first_frame + frame_count - 1
    scene.frame_set(first_frame)
    return frame_times


def _parse_resolution(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT, for example 1280x720") from exc
    if width < 16 or height < 16:
        raise argparse.ArgumentTypeError("resolution must be at least 16x16")
    return width, height


def _arguments() -> argparse.Namespace:
    argv = sys.argv
    values = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Render a 3D Blender flight replay from trajectory.json")
    parser.add_argument("--trajectory", type=Path, required=True, help="recorded trajectory.json")
    parser.add_argument("--scene", "--manifest", dest="scene_manifest", type=Path, required=True, help="scene manifest JSON with asset paths and replay bindings")
    parser.add_argument("--fly-model", type=Path, help="optional external fly model (glTF/GLB/FBX/OBJ/STL/PLY/BLEND) overriding the manifest fly binding; model forward is +X and units are meters")
    parser.add_argument("--out", type=Path, help="MP4 destination; defaults beside trajectory as flight_presentation.mp4")
    parser.add_argument("--context-out", type=Path, help="optional second MP4 using replay.context_camera for a wide scene view")
    parser.add_argument("--context-frames-dir", type=Path, help="also save numbered PNG frames for the wide context camera")
    parser.add_argument("--frames-dir", type=Path, help="also save numbered PNG frames in this directory")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=_parse_resolution, default=(1280, 720))
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE_NEXT"), default="CYCLES")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--time-field", choices=("auto", "frame_time_s", "time_s"), default="auto")
    parser.add_argument("--start-time", type=float, help="start playback at this recorded time in seconds")
    parser.add_argument("--end-time", type=float, help="stop playback at this recorded time in seconds")
    parser.add_argument("--max-frames", type=int, help="render only this many frames from the start")
    parser.add_argument(
        "--camera-distance-m", type=float,
        help="override the main replay camera follow distance from the scene manifest",
    )
    parser.add_argument(
        "--focal-length-mm", type=float,
        help="override the main replay camera focal length from the scene manifest",
    )
    parser.add_argument(
        "--exposure-ev", type=float,
        help="override the main replay camera exposure from the scene manifest",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting requested output files")
    parser.add_argument("--save-blend", type=Path, help="optionally save the assembled Blender scene")
    return parser.parse_args(values)


def _provenance_path(args: argparse.Namespace, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.with_suffix(output_path.suffix + ".provenance.json")
    frame_directory = args.frames_dir or args.context_frames_dir
    if frame_directory is not None:
        return frame_directory / "provenance.json"
    assert args.context_out is not None
    return args.context_out.with_suffix(args.context_out.suffix + ".provenance.json")


def _prepare_outputs(
    args: argparse.Namespace, out_path: Path | None, context_path: Path | None,
    context_frames_dir: Path | None,
) -> None:
    if out_path is None and args.frames_dir is None and context_path is None and context_frames_dir is None:
        raise ReplayError("request an output with --out, --frames-dir, --context-out, or --context-frames-dir")
    for target in (out_path, context_path, args.save_blend):
        if target is None:
            continue
        if target.exists() and target.is_dir():
            raise ReplayError(f"output file path is a directory: {target}")
        if target.exists() and not args.overwrite:
            raise ReplayError(f"output file already exists (use --overwrite): {target}")
    for frame_directory in (args.frames_dir, context_frames_dir):
        if frame_directory is None:
            continue
        if frame_directory.exists() and not frame_directory.is_dir():
            raise ReplayError(f"frames output path is not a directory: {frame_directory}")
        if frame_directory.exists() and any(frame_directory.iterdir()) and not args.overwrite:
            raise ReplayError(f"output directory is not empty (use --overwrite): {frame_directory}")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.frames_dir is not None:
        args.frames_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for stale_frame in args.frames_dir.glob("frame_*.png"):
                stale_frame.unlink()
    if context_frames_dir is not None:
        context_frames_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for stale_frame in context_frames_dir.glob("frame_*.png"):
                stale_frame.unlink()
    if context_path is not None:
        context_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_blend is not None:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)


def _encode_mp4(ffmpeg_path: str, frames_dir: Path, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = sorted(frames_dir.glob("frame_*.png"))
    if not rendered:
        raise ReplayError(f"no rendered PNG frames found in {frames_dir}")
    first_stem = rendered[0].stem
    frame_number = first_stem.removeprefix("frame_")
    if not first_stem.startswith("frame_") or not frame_number.isdigit():
        raise ReplayError(f"unexpected Blender frame name: {rendered[0].name}")
    padding = len(frame_number)
    command = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-start_number", frame_number,
        "-i", str(frames_dir / f"frame_%0{padding}d.png"),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ReplayError(f"ffmpeg failed with exit code {completed.returncode}; rendered PNG frames remain in {frames_dir}")


def _render_png_sequence(scene: bpy.types.Scene, frames_dir: Path) -> None:
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(frames_dir / "frame_")
    bpy.ops.render.render(animation=True)


def _render_mp4_pass(
    scene: bpy.types.Scene, output: Path, fps: int, ffmpeg_path: str | None,
    frames_dir: Path | None = None,
) -> str:
    if ffmpeg_path:
        if frames_dir is not None:
            _render_png_sequence(scene, frames_dir)
            source_frames = frames_dir
        else:
            with tempfile.TemporaryDirectory(prefix="blender-flight-replay-") as temporary_directory:
                source_frames = Path(temporary_directory)
                _render_png_sequence(scene, source_frames)
                _encode_mp4(ffmpeg_path, source_frames, output, fps)
            return "png_sequence+external_ffmpeg"
        _encode_mp4(ffmpeg_path, source_frames, output, fps)
        return "png_sequence+external_ffmpeg"

    # Some Blender distributions omit their FFmpeg output module. If there is
    # no system ffmpeg either, fail with a direct message while preserving any
    # requested PNG sequence.
    if frames_dir is not None:
        _render_png_sequence(scene, frames_dir)
    try:
        scene.render.image_settings.file_format = "FFMPEG"
    except TypeError as exc:
        raise ReplayError(
            "neither system ffmpeg nor Blender's FFmpeg output module is available; "
            "PNG frames were rendered when a frame directory was requested"
        ) from exc
    scene.render.image_settings.color_mode = "RGB"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(output)
    bpy.ops.render.render(animation=True)
    return "blender_ffmpeg"


def _asset_world_bounds(asset: Mapping[str, Any]) -> dict[str, list[float]] | None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: list[Vector] = []
    for obj in asset.get("children", ()):
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT"}:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        try:
            if obj.type == "MESH":
                local_points = [Vector(corner) for corner in evaluated.bound_box]
            else:
                mesh = evaluated.to_mesh()
                local_points = [vertex.co.copy() for vertex in mesh.vertices]
                evaluated.to_mesh_clear()
            points.extend(evaluated.matrix_world @ point for point in local_points)
        except Exception:
            continue
    if not points:
        return None
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return {"minimum_m": minimum, "maximum_m": maximum}


def main() -> int:
    args = _arguments()
    if args.fps <= 0 or args.samples <= 0:
        raise ReplayError("--fps and --samples must be positive")
    trajectory_path = args.trajectory.expanduser().resolve()
    manifest_path = args.scene_manifest.expanduser().resolve()
    if not trajectory_path.is_file():
        raise ReplayError(f"trajectory file not found: {trajectory_path}")
    if not manifest_path.is_file():
        raise ReplayError(f"scene manifest not found: {manifest_path}")
    out_path = args.out.expanduser().resolve() if args.out is not None else None
    context_path = args.context_out.expanduser().resolve() if args.context_out is not None else None
    frames_dir = args.frames_dir.expanduser().resolve() if args.frames_dir is not None else None
    context_frames_dir = args.context_frames_dir.expanduser().resolve() if args.context_frames_dir is not None else None
    if out_path is None and frames_dir is None and context_path is None and context_frames_dir is None:
        out_path = trajectory_path.with_name("flight_presentation.mp4")
    args.frames_dir = frames_dir
    args.context_frames_dir = context_frames_dir
    if context_path is not None and out_path is not None and context_path == out_path:
        raise ReplayError("--context-out must be a different file from --out")
    _prepare_outputs(args, out_path, context_path, context_frames_dir)

    payload = _json_file(trajectory_path, "trajectory")
    manifest = _json_file(manifest_path, "scene manifest")
    rows = _trajectory_rows(payload)
    time_field = args.time_field
    if time_field == "auto":
        time_field = "frame_time_s" if any(_find(row, ("frame_time_s",)) is not None for row in rows) else "time_s"
    fallback_dt = 1.0 / float(args.fps)
    body_samples = _load_body_samples(rows, time_field, fallback_dt)
    stimulus_samples = _stimulus_rows(rows, time_field, fallback_dt)

    coordinate = manifest.get("coordinate_system", {})
    if isinstance(coordinate, Mapping):
        up_axis = str(coordinate.get("up_axis", "+Z")).upper()
        handedness = str(coordinate.get("handedness", "right")).lower()
        if up_axis not in ("+Z", "Z") or handedness != "right":
            raise ReplayError("Blender replay currently requires a right-handed, +Z-up scene and trajectory")

    _clear_scene()
    assets, asset_provenance = _load_assets(manifest_path, manifest)
    replay_cfg = manifest.get("replay", {})
    if not isinstance(replay_cfg, Mapping):
        replay_cfg = {}
    bindings = replay_cfg.get("bindings", {})
    if not isinstance(bindings, Mapping):
        bindings = {}
    fly_object_id = bindings.get("fly_object_id")
    generated_fly = False
    if args.fly_model is not None:
        fly_path = args.fly_model.expanduser().resolve()
        if not fly_path.is_file():
            raise ReplayError(f"--fly-model file not found: {fly_path}")
        if fly_path.suffix.lower() not in SUPPORTED_SCENE_FORMATS:
            raise ReplayError(f"unsupported --fly-model format: {fly_path.suffix}")
        fly_collection = bpy.data.collections.new("Asset::cli_fly_model")
        bpy.context.scene.collection.children.link(fly_collection)
        imported = _import_asset(fly_path, fly_collection)
        if not imported:
            raise ReplayError(f"--fly-model import created no Blender objects: {fly_path}")
        fly_id = "cli_fly_model"
        if fly_id in assets:
            fly_id = "cli_fly_model_override"
        identity = (Vector((0.0, 0.0, 0.0)), Quaternion((1.0, 0.0, 0.0, 0.0)), Vector((1.0, 1.0, 1.0)))
        fly_root = _make_root(fly_id, fly_collection, imported, identity)
        actual_sha = _sha256(fly_path)
        assets[fly_id] = {
            "root": fly_root,
            "children": imported,
            "record": {"object_id": fly_id, "role": "visual", "semantic_role": "fly_visual", "source": "--fly-model"},
            "role": "visual",
            "base_location": identity[0],
            "base_rotation": identity[1],
            "base_scale": identity[2],
            "path": fly_path,
            "actual_sha256": actual_sha,
            "dependency_sha256": {},
        }
        asset_provenance.append({
            "object_id": fly_id,
            "role": "visual",
            "path": str(fly_path),
            "sha256": actual_sha,
            "expected_sha256": None,
            "integrity_verified": None,
            "provenance": {"source": "command line --fly-model", "license": "user-supplied; not declared by scene manifest"},
        })
        fly_object_id = fly_id
    if fly_object_id is not None and (not isinstance(fly_object_id, str) or fly_object_id not in assets):
        raise ReplayError("replay.bindings.fly_object_id does not name an imported scene asset")
    if fly_object_id is None:
        semantic_matches = [
            key for key, asset in assets.items()
            if str(asset["record"].get("semantic_role", "")).lower() in {"fly", "fly_rig", "fly_visual"}
        ]
        if len(semantic_matches) == 1:
            fly_object_id = semantic_matches[0]
        else:
            fly_object_id = "generated_fly"
            assets[fly_object_id] = _procedural_fly_asset()
            asset_provenance.append({
                "object_id": fly_object_id,
                "role": "visual",
                "path": None,
                "sha256": None,
                "expected_sha256": None,
                "integrity_verified": None,
                "provenance": {
                    "source": "native geometry generated by render_blender_flight.py",
                    "license": "procedural code",
                    "limitation": "stylized presentation rig, not a biological reconstruction",
                },
            })
            generated_fly = True
    body_asset = assets[str(fly_object_id)]
    body_root = body_asset["root"]
    if body_asset["role"] == "collision":
        raise ReplayError("replay.bindings.fly_object_id must reference a renderable asset role")
    stimulus_binding_config = bindings.get("stimulus_object_ids", {})
    if not isinstance(stimulus_binding_config, Mapping):
        raise ReplayError("replay.bindings.stimulus_object_ids must be an object")
    needs_native_palm = any(
        stimulus_id.lower().startswith("palm")
        and stimulus_id not in stimulus_binding_config
        and "*" not in stimulus_binding_config
        and stimulus_id not in assets
        and not any(
            str(asset["record"].get("semantic_role", "")).lower() in {"stimulus", "palm"}
            for asset in assets.values()
        )
        for stimulus_id in stimulus_samples
    )
    if needs_native_palm:
        palm_asset = _procedural_palm_asset()
        assets["generated_palm"] = palm_asset
        asset_provenance.append({
            "object_id": "generated_palm",
            "role": "visual",
            "path": None,
            "sha256": None,
            "expected_sha256": None,
            "integrity_verified": None,
            "provenance": {
                "source": "native geometry generated from the recorded stimulus concept",
                "license": "procedural code",
                "limitation": "abstract palm-shaped visual proxy; pose/size follow recorded stimulus_state",
            },
        })
    stimulus_bindings = _stimulus_bindings(replay_cfg, stimulus_samples, assets)

    camera_cfg = replay_cfg.get("camera", {})
    if not isinstance(camera_cfg, Mapping):
        camera_cfg = {}
    camera_cfg = dict(camera_cfg)
    if args.camera_distance_m is not None:
        if not math.isfinite(args.camera_distance_m) or args.camera_distance_m <= 0:
            raise ReplayError("--camera-distance-m must be a positive finite number")
        camera_cfg["follow_distance_m"] = args.camera_distance_m
    if args.focal_length_mm is not None:
        if not math.isfinite(args.focal_length_mm) or args.focal_length_mm <= 0:
            raise ReplayError("--focal-length-mm must be a positive finite number")
        camera_cfg["focal_length_mm"] = args.focal_length_mm
    if args.exposure_ev is not None:
        if not math.isfinite(args.exposure_ev):
            raise ReplayError("--exposure-ev must be a finite number")
        camera_cfg["exposure_ev"] = args.exposure_ev
    lighting_cfg = replay_cfg.get("lighting", {})
    if not isinstance(lighting_cfg, Mapping):
        lighting_cfg = {}
    _create_ground(replay_cfg)
    camera = _create_camera({**replay_cfg, "camera": camera_cfg})
    lighting_provenance = _create_lighting(replay_cfg, manifest_path, manifest)
    _configure_render(args, camera_cfg)

    recorded_start = float(body_samples[0]["time_s"])
    recorded_end = float(body_samples[-1]["time_s"])
    time_start = recorded_start if args.start_time is None else float(args.start_time)
    time_end = recorded_end if args.end_time is None else float(args.end_time)
    if not math.isfinite(time_start) or not math.isfinite(time_end):
        raise ReplayError("--start-time and --end-time must be finite")
    if time_start < recorded_start or time_start > recorded_end:
        raise ReplayError(f"--start-time must be within recorded time [{recorded_start}, {recorded_end}]")
    if time_end < recorded_start or time_end > recorded_end or time_end < time_start:
        raise ReplayError(f"--end-time must be within [{time_start}, {recorded_end}] and no earlier than --start-time")
    duration = max(0.0, time_end - time_start)
    frame_count = max(1, int(math.ceil(duration * args.fps)) + 1)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ReplayError("--max-frames must be positive")
        frame_count = min(frame_count, args.max_frames)
    frame_times = _animate(
        body_root, body_asset, body_samples, stimulus_samples, stimulus_bindings,
        assets, camera, camera_cfg, 1, frame_count, args.fps, time_start, time_end,
    )
    macro_camera_position = list(camera.location)
    context_camera_position: list[float] | None = None

    if args.save_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))

    ffmpeg_path = shutil.which("ffmpeg") if out_path is not None or context_path is not None else None
    rendered_modes: list[str] = []
    scene = bpy.context.scene
    if out_path is not None:
        rendered_modes.append(_render_mp4_pass(scene, out_path, args.fps, ffmpeg_path, frames_dir))
    elif frames_dir is not None:
        _render_png_sequence(scene, frames_dir)
        rendered_modes.append("png_frames")

    context_cfg: Mapping[str, Any] | None = None
    if context_path is not None or context_frames_dir is not None:
        configured_context = replay_cfg.get("context_camera", {})
        if not isinstance(configured_context, Mapping):
            raise ReplayError("replay.context_camera must be an object")
        context_cfg = configured_context or {
            "mode": "chase",
            "focal_length_mm": 40,
            "follow_distance_m": 3.0,
            "height_m": 1.0,
            "look_ahead_m": 0.6,
        }
        _configure_render(args, context_cfg)
        frame_times = _animate(
            body_root, body_asset, body_samples, stimulus_samples, stimulus_bindings,
            assets, camera, context_cfg, 1, frame_count, args.fps, time_start, time_end,
        )
        context_camera_position = list(camera.location)
        if context_path is not None:
            rendered_modes.append(_render_mp4_pass(scene, context_path, args.fps, ffmpeg_path, context_frames_dir))
        elif context_frames_dir is not None:
            _render_png_sequence(scene, context_frames_dir)
            rendered_modes.append("context_png_frames")

    metadata = payload.get("metadata", {})
    evidence_class = metadata.get("evidence_class", "digital_simulation") if isinstance(metadata, Mapping) else "digital_simulation"
    sidecar = {
        "schema_version": "blender-flight-replay-provenance-1",
        "renderer": "Blender",
        "renderer_script_version": SCRIPT_VERSION,
        "blender_version": bpy.app.version_string,
        "evidence_class": evidence_class,
        "interpretation": "3D visual reconstruction driven by recorded digital-simulation telemetry; scene assets, camera, lighting, and shading are presentation choices.",
        "trajectory": {
            "path": str(trajectory_path),
            "sha256": _sha256(trajectory_path),
            "schema_version": payload.get("schema_version"),
            "row_count": len(rows),
            "time_field": time_field,
            "recorded_time_start_s": recorded_start,
            "recorded_time_end_s": recorded_end,
            "requested_segment_start_s": time_start,
            "requested_segment_end_s": time_end,
            "rendered_segment_start_s": frame_times[0],
            "rendered_segment_end_s": frame_times[-1],
            "mapping": "scene frame n samples telemetry at start_time_s + (n - 1) / fps, clamped to the recorded interval",
            "fly_object_id": fly_object_id,
            "fly_asset_is_procedural_fallback": generated_fly,
            "stimulus_object_bindings": dict(stimulus_bindings),
            "stimulus_state_ids": sorted(stimulus_samples),
        },
        "scene_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "scene_id": manifest.get("scene_id"),
            "assets": asset_provenance,
        },
        "render": {
            "engine": args.engine,
            "device": "CPU" if args.engine == "CYCLES" else "Blender render backend",
            "fps": args.fps,
            "frame_count": frame_count,
            "resolution": list(args.resolution),
            "samples": args.samples,
            "output_mode": "+".join(rendered_modes),
            "video_path": None if out_path is None else str(out_path),
            "video_sha256": None if out_path is None else _sha256(out_path),
            "context_video_path": None if context_path is None else str(context_path),
            "context_video_sha256": None if context_path is None else _sha256(context_path),
            "frames_directory": None if frames_dir is None else str(frames_dir),
            "context_frames_directory": None if context_frames_dir is None else str(context_frames_dir),
            "blend_path": None if args.save_blend is None else str(args.save_blend),
            "blend_sha256": None if args.save_blend is None else _sha256(args.save_blend),
            "camera": dict(camera_cfg),
            "context_camera": None if context_cfg is None else dict(context_cfg),
            "lighting": dict(lighting_cfg),
            "lighting_assets": lighting_provenance,
            "camera_positions_m_frame_1": {
                "macro": macro_camera_position,
                "context": context_camera_position,
            },
            "asset_world_bounds_m_frame_1": {
                object_id: _asset_world_bounds(asset) for object_id, asset in assets.items()
            },
        },
    }
    provenance_path = _provenance_path(args, out_path)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", encoding="utf-8") as stream:
        json.dump(sidecar, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Rendered {frame_count} frames through {frame_times[-1]:.3f}s")
    if out_path is not None:
        print(f"Video: {out_path}")
    if context_path is not None:
        print(f"Context video: {context_path}")
    if frames_dir is not None:
        print(f"Frames: {frames_dir}")
    print(f"Provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as exc:
        print(f"Blender replay error: {exc}", file=sys.stderr)
        raise SystemExit(2)
