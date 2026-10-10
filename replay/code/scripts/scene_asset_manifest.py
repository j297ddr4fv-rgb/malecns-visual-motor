#!/usr/bin/env python3
"""Read and validate the data-driven external-scene manifest.

Manifest distances and translations are in metres. Asset paths are relative
to the manifest file, which makes a scene bundle relocatable. This module is
deliberately read-only; it never downloads, rewrites, or repacks assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ROLES = {"visual", "collision", "visual_collision"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {length} numbers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain {length} numbers") from exc
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(values))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetTransform:
    """Asset-root transform in the scene coordinate system (+Z up), in metres."""

    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class SceneAssetSpec:
    object_id: str
    manifest_relative_path: str
    path: Path
    role: str
    transform: AssetTransform
    asset_unit_scale_to_m: float
    integrity_sha256: str | None
    integrity_files: Mapping[str, str]
    provenance: Mapping[str, Any]
    material_refs: Mapping[str, str]
    semantic_role: str | None = None


@dataclass(frozen=True)
class SceneAssetManifest:
    path: Path
    scene_id: str
    description: str | None
    coordinate_system: Mapping[str, str]
    materials: Mapping[str, Mapping[str, Any]]
    assets: tuple[SceneAssetSpec, ...]
    asset_root: str | None
    replay: Mapping[str, Any]
    simulation: Mapping[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "SceneAssetManifest":
        manifest_path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid scene manifest JSON at {manifest_path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("scene manifest root must be an object")
        schema_version = raw.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ValueError("scene manifest schema_version must be 1")
        if raw.get("units") != "m":
            raise ValueError('scene manifest units must be explicitly "m"')
        scene_id = raw.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        description = raw.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("description must be a string when provided")

        coords = raw.get("coordinate_system")
        if not isinstance(coords, Mapping):
            raise ValueError("coordinate_system must be an object")
        if coords.get("handedness") != "right" or coords.get("up_axis") != "+Z" or coords.get("forward_axis") != "+X":
            raise ValueError("scene coordinate_system must be right-handed, +Z up, and +X forward")

        materials = raw.get("materials", {})
        if not isinstance(materials, Mapping):
            raise ValueError("materials must be an object keyed by material ID")
        normalized_materials: dict[str, Mapping[str, Any]] = {}
        for material_id, material in materials.items():
            if not isinstance(material_id, str) or not material_id.strip() or not isinstance(material, Mapping):
                raise ValueError("each material must have a non-empty string ID and object value")
            item = dict(material)
            if "base_color_rgba" in item:
                color = _vector(item["base_color_rgba"], 4, f"materials.{material_id}.base_color_rgba")
                if any(channel < 0.0 or channel > 1.0 for channel in color):
                    raise ValueError(f"materials.{material_id}.base_color_rgba channels must be in [0, 1]")
                item["base_color_rgba"] = color
            normalized_materials[material_id] = item

        asset_root_raw = raw.get("asset_root")
        if asset_root_raw is not None and (not isinstance(asset_root_raw, str) or not asset_root_raw.strip()):
            raise ValueError("asset_root must be a non-empty path string when provided")
        assets_raw = raw.get("assets")
        if not isinstance(assets_raw, list):
            raise ValueError("assets must be an array")
        assets: list[SceneAssetSpec] = []
        seen_ids: set[str] = set()
        for index, asset in enumerate(assets_raw):
            label = f"assets[{index}]"
            if not isinstance(asset, Mapping):
                raise ValueError(f"{label} must be an object")
            object_id = asset.get("object_id")
            if not isinstance(object_id, str) or not object_id.strip():
                raise ValueError(f"{label}.object_id must be a non-empty string")
            if object_id in seen_ids:
                raise ValueError(f"duplicate scene asset object_id: {object_id}")
            seen_ids.add(object_id)
            role = asset.get("role")
            if not isinstance(role, str) or role not in _ROLES:
                raise ValueError(f"{label}.role must be one of {sorted(_ROLES)}")
            relative_path = asset.get("path")
            if not isinstance(relative_path, str) or not relative_path.strip():
                raise ValueError(f"{label}.path must be a non-empty relative path")
            asset_path = Path(relative_path)
            if asset_path.is_absolute() or asset_path.suffix.lower() not in {".gltf", ".glb"}:
                raise ValueError(f"{label}.path must be relative and end in .gltf or .glb")
            transform_raw = asset.get("transform")
            if not isinstance(transform_raw, Mapping):
                raise ValueError(f"{label}.transform is required")
            translation = _vector(transform_raw.get("translation_m"), 3, f"{label}.transform.translation_m")
            rotation = _vector(transform_raw.get("rotation_xyzw"), 4, f"{label}.transform.rotation_xyzw")
            if sum(value * value for value in rotation) <= 1.0e-24:
                raise ValueError(f"{label}.transform.rotation_xyzw must not be zero")
            scale_raw = transform_raw.get("scale")
            if isinstance(scale_raw, (int, float)) and not isinstance(scale_raw, bool):
                scalar = _number(scale_raw, f"{label}.transform.scale")
                scale = (scalar, scalar, scalar)
            else:
                scale = _vector(scale_raw, 3, f"{label}.transform.scale")
            if any(abs(value) <= 1.0e-12 for value in scale):
                raise ValueError(f"{label}.transform.scale values must be non-zero")

            integrity = asset.get("integrity", {})
            if not isinstance(integrity, Mapping):
                raise ValueError(f"{label}.integrity must be an object")
            expected_hash = integrity.get("sha256")
            if expected_hash is not None:
                if not isinstance(expected_hash, str):
                    raise ValueError(f"{label}.integrity.sha256 must be a string")
                expected_hash = expected_hash.lower()
                if not _SHA256_RE.fullmatch(expected_hash):
                    raise ValueError(f"{label}.integrity.sha256 must be 64 hexadecimal characters")
            files_raw = integrity.get("files", {})
            if not isinstance(files_raw, Mapping):
                raise ValueError(f"{label}.integrity.files must map asset-relative paths to SHA-256 hashes")
            integrity_files: dict[str, str] = {}
            for dependency_path, dependency_hash in files_raw.items():
                if not isinstance(dependency_path, str) or not dependency_path.strip():
                    raise ValueError(f"{label}.integrity.files keys must be non-empty relative paths")
                pure_path = PurePosixPath(dependency_path)
                if pure_path.is_absolute() or ".." in pure_path.parts:
                    raise ValueError(f"{label}.integrity.files paths must stay inside the asset directory")
                if not isinstance(dependency_hash, str) or not _SHA256_RE.fullmatch(dependency_hash):
                    raise ValueError(f"{label}.integrity.files[{dependency_path!r}] must be a SHA-256 digest")
                integrity_files[pure_path.as_posix()] = dependency_hash.lower()
            provenance = asset.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"{label}.provenance is required and must be an object")
            for required in ("source", "license", "exporter"):
                if not isinstance(provenance.get(required), str) or not provenance[required].strip():
                    raise ValueError(f"{label}.provenance.{required} must be a non-empty string")
            material_refs_raw = asset.get("material_refs", {})
            if not isinstance(material_refs_raw, Mapping):
                raise ValueError(f"{label}.material_refs must be an object")
            material_refs: dict[str, str] = {}
            for source_name, material_id in material_refs_raw.items():
                if not isinstance(source_name, str) or not isinstance(material_id, str):
                    raise ValueError(f"{label}.material_refs must map glTF material names/indices to IDs")
                if material_id not in normalized_materials:
                    raise ValueError(f"{label}.material_refs references unknown material ID {material_id!r}")
                material_refs[source_name] = material_id
            semantic_role = asset.get("semantic_role")
            if semantic_role is not None and (not isinstance(semantic_role, str) or not semantic_role.strip()):
                raise ValueError(f"{label}.semantic_role must be a non-empty string when provided")
            unit_scale = _number(asset.get("asset_unit_scale_to_m", 1.0), f"{label}.asset_unit_scale_to_m")
            if unit_scale <= 0:
                raise ValueError(f"{label}.asset_unit_scale_to_m must be positive")
            # A user may author a manifest before a fetched asset has a final
            # digest. Runtime loading can opt out of verification explicitly;
            # the default loader path still requires a hash.
            root_env = os.environ.get("MALECNS_ASSET_ROOT")
            root_candidates: list[Path] = []
            if root_env:
                root_candidates.append(Path(root_env).expanduser() / asset_path)
            if asset_root_raw:
                declared_root = Path(asset_root_raw).expanduser()
                if not declared_root.is_absolute():
                    declared_root = manifest_path.parent / declared_root
                root_candidates.append(declared_root / asset_path)
            root_candidates.append(manifest_path.parent / asset_path)
            resolved_candidates = [candidate.resolve() for candidate in root_candidates]
            resolved_path = next((candidate for candidate in resolved_candidates if candidate.is_file()),
                                 resolved_candidates[0])
            assets.append(SceneAssetSpec(
                object_id=object_id,
                manifest_relative_path=relative_path,
                path=resolved_path,
                role=role,
                transform=AssetTransform(translation, rotation, scale),
                asset_unit_scale_to_m=unit_scale,
                integrity_sha256=expected_hash,
                integrity_files=integrity_files,
                provenance=dict(provenance),
                material_refs=material_refs,
                semantic_role=semantic_role,
            ))

        replay = raw.get("replay", {})
        if not isinstance(replay, Mapping):
            raise ValueError("replay must be an object")
        simulation_raw = raw.get("simulation", {})
        if not isinstance(simulation_raw, Mapping):
            raise ValueError("simulation must be an object")
        stimulus_object_ids_raw = simulation_raw.get("stimulus_object_ids", {})
        if not isinstance(stimulus_object_ids_raw, Mapping):
            raise ValueError("simulation.stimulus_object_ids must be an object mapping stimulus IDs to object IDs")
        stimulus_object_ids: dict[str, str] = {}
        for stimulus_id, object_id in stimulus_object_ids_raw.items():
            if not isinstance(stimulus_id, str) or not stimulus_id.strip():
                raise ValueError("simulation.stimulus_object_ids keys must be non-empty stimulus IDs")
            if not isinstance(object_id, str) or not object_id.strip():
                raise ValueError(f"simulation.stimulus_object_ids[{stimulus_id!r}] must be a non-empty object ID")
            stimulus_object_ids[stimulus_id] = object_id
        simulation = {key: value for key, value in simulation_raw.items() if key != "stimulus_object_ids"}
        if simulation:
            raise ValueError(f"unsupported simulation field(s): {', '.join(sorted(map(str, simulation)))}")
        if stimulus_object_ids:
            simulation["stimulus_object_ids"] = stimulus_object_ids
        return cls(
            path=manifest_path,
            scene_id=scene_id,
            description=description,
            coordinate_system=dict(coords),
            materials=normalized_materials,
            assets=tuple(assets),
            asset_root=asset_root_raw,
            replay=dict(replay),
            simulation=simulation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the validated manifest payload, with paths kept manifest-relative."""
        asset_records = []
        for asset in self.assets:
            asset_records.append({
                "object_id": asset.object_id,
                "path": asset.manifest_relative_path,
                "role": asset.role,
                "semantic_role": asset.semantic_role,
                "transform": {
                    "translation_m": list(asset.transform.translation_m),
                    "rotation_xyzw": list(asset.transform.rotation_xyzw),
                    "scale": list(asset.transform.scale),
                },
                "asset_unit_scale_to_m": asset.asset_unit_scale_to_m,
                "material_refs": dict(asset.material_refs),
                "integrity": {
                    **({"sha256": asset.integrity_sha256} if asset.integrity_sha256 else {}),
                    "files": dict(asset.integrity_files),
                },
                "provenance": dict(asset.provenance),
            })
        return {
            "schema_version": 1,
            "scene_id": self.scene_id,
            **({"description": self.description} if self.description is not None else {}),
            "units": "m",
            "coordinate_system": dict(self.coordinate_system),
            "materials": {key: dict(value) for key, value in self.materials.items()},
            "asset_root": self.asset_root,
            "assets": asset_records,
            **({"simulation": dict(self.simulation)} if self.simulation else {}),
            "replay": dict(self.replay),
        }


def load_scene_manifest(path: str | Path) -> SceneAssetManifest:
    """Convenience wrapper around :meth:`SceneAssetManifest.from_json`."""
    return SceneAssetManifest.from_json(path)
