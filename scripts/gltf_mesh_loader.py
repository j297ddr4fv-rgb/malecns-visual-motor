#!/usr/bin/env python3
"""CPU-only, read-only loader for Blender-exported glTF 2.0 ``.gltf``/``.glb``.

The public entry point is :func:`load_scene_assets`. It reads a scene manifest,
verifies each asset checksum, applies the glTF node hierarchy and manifest
transform, converts glTF's right-handed +Y-up frame to the project frame
(right-handed +Z up, +X forward), and triangulates triangle, strip, and fan
primitives. Mesh arrays are NumPy arrays marked read-only. Textures are left
undecoded by default; manifest materials can opt into deterministic bilinear
base-color sampling at triangle UV centroids with
``sensor_texture_mode: "triangle_centroid"``.

No Blender process, GPU context, network access, or asset writes are used.
Compressed glTF extensions (for example Draco and meshopt) are intentionally
rejected with a direct error; export uncompressed glTF from Blender first.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import struct
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

import numpy as np

try:
    from scene_asset_manifest import SceneAssetManifest, SceneAssetSpec, load_scene_manifest, sha256_file
except ImportError:  # package-style ``import scripts.gltf_mesh_loader``
    from .scene_asset_manifest import SceneAssetManifest, SceneAssetSpec, load_scene_manifest, sha256_file


_GLB_MAGIC = b"glTF"
_GLB_JSON = 0x4E4F534A
_GLB_BIN = 0x004E4942
_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_COMPONENT_WIDTH = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass(frozen=True)
class SceneMesh:
    """One manifest asset flattened into a triangulated world-space mesh."""

    object_id: str
    role: str
    semantic_role: str | None
    vertices_m: np.ndarray
    triangles: np.ndarray
    triangle_material_ids: tuple[str | None, ...]
    materials: Mapping[str, Mapping[str, Any]]
    source_path: Path
    source_sha256: str
    source_files_sha256: Mapping[str, str]
    provenance: Mapping[str, Any]
    # Float64 RGB in [0, 1], one row per triangle. Flat materials retain their
    # final base RGB; opted-in materials contain sampled texture RGB multiplied
    # by glTF baseColorFactor and the manifest base_color_rgba override.
    triangle_albedos: np.ndarray | None = None
    # Per opted-in manifest material: mode, source image hashes/dimensions, and
    # face- and area-weighted sampled RGB summaries.
    texture_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedScene:
    """Validated scene manifest and its loaded mesh assets."""

    scene_id: str
    objects: tuple[SceneMesh, ...]
    materials: Mapping[str, Mapping[str, Any]]
    replay: Mapping[str, Any]
    manifest_path: Path
    manifest: Mapping[str, Any]


def _read_gltf(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    raw = path.read_bytes()
    binary_chunk: bytes | None = None
    if path.suffix.lower() == ".glb":
        if len(raw) < 12:
            raise ValueError(f"truncated GLB header: {path}")
        magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
        if magic != _GLB_MAGIC or version != 2:
            raise ValueError(f"{path} is not a glTF 2.0 GLB")
        if declared_length != len(raw):
            raise ValueError(f"GLB length header does not match file size: {path}")
        offset = 12
        json_chunk: bytes | None = None
        while offset < len(raw):
            if offset + 8 > len(raw):
                raise ValueError(f"truncated GLB chunk header: {path}")
            chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
            offset += 8
            end = offset + chunk_length
            if end > len(raw):
                raise ValueError(f"truncated GLB chunk data: {path}")
            chunk = raw[offset:end]
            if chunk_type == _GLB_JSON:
                if json_chunk is not None:
                    raise ValueError(f"GLB contains multiple JSON chunks: {path}")
                json_chunk = chunk
            elif chunk_type == _GLB_BIN and binary_chunk is None:
                binary_chunk = chunk
            offset = end
        if json_chunk is None:
            raise ValueError(f"GLB has no JSON chunk: {path}")
        try:
            document = json.loads(json_chunk.decode("utf-8").rstrip(" \t\r\n\0"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid GLB JSON chunk in {path}: {exc}") from exc
    else:
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid glTF JSON in {path}: {exc}") from exc

    if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
        raise ValueError(f"{path} is not a glTF 2.0 document")
    required_extensions = document.get("extensionsRequired", [])
    unsupported = sorted(set(required_extensions) & {"KHR_draco_mesh_compression", "EXT_meshopt_compression"})
    if unsupported:
        raise ValueError(f"unsupported compressed glTF extension(s) in {path}: {', '.join(unsupported)}")

    buffers: list[bytes] = []
    for index, buffer_desc in enumerate(document.get("buffers", [])):
        if not isinstance(buffer_desc, Mapping):
            raise ValueError(f"glTF buffer {index} descriptor is malformed")
        uri = buffer_desc.get("uri")
        if uri is None:
            if index != 0 or binary_chunk is None:
                raise ValueError(f"glTF buffer {index} has no URI or matching GLB BIN chunk")
            payload = binary_chunk
        elif not isinstance(uri, str):
            raise ValueError(f"glTF buffer {index} URI is malformed")
        elif uri.startswith("data:"):
            header, sep, content = uri.partition(",")
            if not sep:
                raise ValueError(f"malformed data URI in glTF buffer {index}")
            try:
                if ";base64" in header:
                    payload = base64.b64decode(content, validate=True)
                else:
                    payload = urllib.parse.unquote_to_bytes(content)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"invalid data URI in glTF buffer {index}") from exc
        else:
            parsed = urllib.parse.urlparse(uri)
            if parsed.scheme or parsed.netloc:
                raise ValueError(f"network or non-local glTF buffer URI is not supported: {uri!r}")
            buffer_path = (path.parent / urllib.parse.unquote(parsed.path)).resolve()
            try:
                buffer_path.relative_to(path.parent.resolve())
            except ValueError as exc:
                raise ValueError(f"glTF buffer URI escapes its asset directory: {uri!r}") from exc
            payload = buffer_path.read_bytes()
        expected_length = buffer_desc.get("byteLength")
        if isinstance(expected_length, bool) or not isinstance(expected_length, int) or expected_length < 0:
            raise ValueError(f"glTF buffer {index} has invalid byteLength")
        if len(payload) < expected_length:
            raise ValueError(f"glTF buffer {index} is shorter than byteLength")
        buffers.append(payload)
    return document, buffers


def _verify_asset_dependencies(gltf: Mapping[str, Any], spec: SceneAssetSpec,
                               *, verify_checksums: bool) -> dict[str, str]:
    """Verify declared hashes for external buffers and textures."""
    referenced: set[str] = set()
    for category in ("buffers", "images"):
        for entry in gltf.get(category, []):
            if not isinstance(entry, Mapping):
                continue
            uri = entry.get("uri")
            if not isinstance(uri, str) or uri.startswith("data:"):
                continue
            parsed = urllib.parse.urlparse(uri)
            if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError(f"non-local or decorated glTF dependency URI is not supported: {uri!r}")
            decoded_path = urllib.parse.unquote(parsed.path)
            pure_path = PurePosixPath(decoded_path)
            if pure_path.is_absolute() or ".." in pure_path.parts:
                raise ValueError(f"glTF dependency URI escapes its asset directory: {uri!r}")
            relative = pure_path.as_posix()
            if relative not in ("", "."):
                referenced.add(relative)
    if verify_checksums:
        missing = sorted(referenced - set(spec.integrity_files))
        if missing:
            raise ValueError(
                f"integrity.files is missing hashes for {spec.object_id} dependency(s): {', '.join(missing)}"
            )
    hashes: dict[str, str] = {}
    to_check = set(spec.integrity_files) if verify_checksums else set()
    if verify_checksums:
        to_check.update(referenced)
    for relative in sorted(to_check):
        file_path = (spec.path.parent / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            file_path.relative_to(spec.path.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"integrity path escapes asset directory: {relative!r}") from exc
        if not file_path.is_file():
            raise FileNotFoundError(f"glTF dependency does not exist: {file_path}")
        actual = sha256_file(file_path)
        expected = spec.integrity_files.get(relative)
        if verify_checksums and expected is not None and actual != expected:
            raise ValueError(f"SHA-256 mismatch for {spec.object_id} dependency {relative}: expected {expected}, got {actual}")
        hashes[relative] = actual
    return hashes


def _buffer_view_bytes(gltf: Mapping[str, Any], buffers: list[bytes], view_index: int,
                       byte_offset: int, byte_length: int) -> memoryview:
    views = gltf.get("bufferViews", [])
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(views):
        raise ValueError(f"invalid glTF bufferView index {view_index!r}")
    view = views[view_index]
    if not isinstance(view, Mapping):
        raise ValueError(f"glTF bufferView {view_index} is malformed")
    buffer_index = view.get("buffer")
    if not isinstance(buffer_index, int) or buffer_index < 0 or buffer_index >= len(buffers):
        raise ValueError(f"glTF bufferView {view_index} has invalid buffer index")
    view_start = view.get("byteOffset", 0)
    view_length = view.get("byteLength")
    if isinstance(view_start, bool) or not isinstance(view_start, int) or view_start < 0:
        raise ValueError(f"glTF bufferView {view_index} has invalid byteOffset")
    if isinstance(view_length, bool) or not isinstance(view_length, int) or view_length < 0:
        raise ValueError(f"glTF bufferView {view_index} has invalid byteLength")
    if byte_offset < 0 or byte_length < 0 or byte_offset + byte_length > view_length:
        raise ValueError(f"accessor range exceeds glTF bufferView {view_index}")
    buffer = buffers[buffer_index]
    absolute_start = view_start + byte_offset
    absolute_end = absolute_start + byte_length
    if absolute_end > len(buffer):
        raise ValueError(f"glTF bufferView {view_index} exceeds its buffer")
    return memoryview(buffer)[absolute_start:absolute_end]


def _decode_accessor(gltf: Mapping[str, Any], buffers: list[bytes], accessor_index: int) -> np.ndarray:
    accessors = gltf.get("accessors", [])
    if not isinstance(accessor_index, int) or accessor_index < 0 or accessor_index >= len(accessors):
        raise ValueError(f"invalid glTF accessor index {accessor_index!r}")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, Mapping):
        raise ValueError(f"glTF accessor {accessor_index} is malformed")
    component_type = accessor.get("componentType")
    if component_type not in _COMPONENT_DTYPES:
        raise ValueError(f"unsupported componentType in glTF accessor {accessor_index}")
    dtype = _COMPONENT_DTYPES[component_type]
    accessor_type = accessor.get("type")
    width = _TYPE_WIDTH.get(accessor_type)
    count = accessor.get("count")
    if width is None or isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"invalid type or count in glTF accessor {accessor_index}")
    output = np.zeros((count, width), dtype=dtype)
    view_index = accessor.get("bufferView")
    if view_index is not None:
        offset = accessor.get("byteOffset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"invalid byteOffset in glTF accessor {accessor_index}")
        item_size = dtype.itemsize * width
        view = gltf.get("bufferViews", [])[view_index]
        stride = view.get("byteStride", item_size)
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < item_size:
            raise ValueError(f"invalid byteStride in glTF accessor {accessor_index}")
        extent = 0 if count == 0 else (count - 1) * stride + item_size
        raw = _buffer_view_bytes(gltf, buffers, view_index, offset, extent)
        if stride == item_size:
            output = np.frombuffer(raw, dtype=dtype, count=count * width).reshape((count, width)).copy()
        else:
            for row in range(count):
                start = row * stride
                output[row] = np.frombuffer(raw[start:start + item_size], dtype=dtype, count=width)

    sparse = accessor.get("sparse")
    if sparse is not None:
        if not isinstance(sparse, Mapping):
            raise ValueError(f"malformed sparse data in glTF accessor {accessor_index}")
        sparse_count = sparse.get("count")
        indices = sparse.get("indices")
        values = sparse.get("values")
        if (isinstance(sparse_count, bool) or not isinstance(sparse_count, int) or sparse_count < 0
                or not isinstance(indices, Mapping) or not isinstance(values, Mapping)):
            raise ValueError(f"malformed sparse data in glTF accessor {accessor_index}")
        index_component = indices.get("componentType")
        if index_component not in {5121, 5123, 5125}:
            raise ValueError("sparse accessor indices must use unsigned byte, short, or int")
        index_dtype = _COMPONENT_DTYPES[index_component]
        index_offset = indices.get("byteOffset", 0)
        index_view = indices.get("bufferView")
        index_raw = _buffer_view_bytes(gltf, buffers, index_view, index_offset,
                                       sparse_count * index_dtype.itemsize)
        sparse_indices = np.frombuffer(index_raw, dtype=index_dtype, count=sparse_count).astype(np.int64)
        if np.any(sparse_indices >= count) or len(set(sparse_indices.tolist())) != sparse_count:
            raise ValueError(f"invalid sparse indices in glTF accessor {accessor_index}")
        value_offset = values.get("byteOffset", 0)
        value_view = values.get("bufferView")
        values_raw = _buffer_view_bytes(gltf, buffers, value_view, value_offset,
                                        sparse_count * dtype.itemsize * width)
        sparse_values = np.frombuffer(values_raw, dtype=dtype,
                                      count=sparse_count * width).reshape((sparse_count, width))
        output[sparse_indices] = sparse_values

    if accessor.get("normalized") and component_type != 5126:
        output = output.astype(np.float64)
        if component_type in (5120, 5122):
            maximum = float(np.iinfo(dtype).max)
            output = np.maximum(output / maximum, -1.0)
        else:
            output /= float(np.iinfo(dtype).max)
    return output


def _quat_matrix_xyzw(value: Any) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("rotation quaternion must contain four finite values")
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("rotation quaternion must be finite and non-zero")
    x, y, z, w = quat / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), 0.0],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), 0.0],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _node_matrix(node: Mapping[str, Any]) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float64)
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError("glTF node matrix must contain 16 finite values")
        matrix = values.reshape((4, 4), order="F")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("glTF node matrix has invalid homogeneous bottom row")
        return matrix
    translation = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    if translation.shape != (3,) or scale.shape != (3,) or not np.isfinite(translation).all() or not np.isfinite(scale).all():
        raise ValueError("glTF node translation/scale must contain finite vec3 values")
    rotation = _quat_matrix_xyzw(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    matrix = rotation.copy()
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _asset_matrix(asset: SceneAssetSpec) -> np.ndarray:
    matrix = _quat_matrix_xyzw(asset.transform.rotation_xyzw)
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(asset.transform.scale)
    matrix[:3, 3] = np.asarray(asset.transform.translation_m, dtype=np.float64)
    return matrix


def _triangulate(indices: np.ndarray, mode: int) -> np.ndarray:
    flattened = np.asarray(indices).reshape(-1).astype(np.int64, copy=False)
    if mode == 4:
        if len(flattened) % 3:
            raise ValueError("glTF TRIANGLES primitive index count must be divisible by three")
        return flattened.reshape((-1, 3))
    if mode == 5:
        faces = []
        for index in range(len(flattened) - 2):
            # Strip winding alternates so all output faces keep the source orientation.
            a, b, c = (int(flattened[index]), int(flattened[index + 1]), int(flattened[index + 2]))
            faces.append((a, b, c) if index % 2 == 0 else (b, a, c))
        return np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    if mode == 6:
        if len(flattened) < 3:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray([(int(flattened[0]), int(flattened[index]), int(flattened[index + 1]))
                           for index in range(1, len(flattened) - 1)], dtype=np.int64).reshape((-1, 3))
    raise ValueError(f"unsupported glTF primitive mode {mode}; expected TRIANGLES, TRIANGLE_STRIP, or TRIANGLE_FAN")


def _gltf_materials(gltf: Mapping[str, Any], spec: SceneAssetSpec,
                    manifest_materials: Mapping[str, Mapping[str, Any]]) -> tuple[
                        dict[str, dict[str, Any]], list[str], list[dict[str, Any]]
                    ]:
    normalized: dict[str, dict[str, Any]] = {key: dict(value) for key, value in manifest_materials.items()}
    for material_id, material in normalized.items():
        texture_mode = material.get("sensor_texture_mode")
        if texture_mode is not None and texture_mode != "triangle_centroid":
            raise ValueError(
                f"materials.{material_id}.sensor_texture_mode must be 'triangle_centroid' when provided"
            )
    ids: list[str] = []
    source_records: list[dict[str, Any]] = []
    for index, material in enumerate(gltf.get("materials", [])):
        if not isinstance(material, Mapping):
            raise ValueError(f"glTF material {index} is malformed")
        name = material.get("name")
        name_key = name if isinstance(name, str) and name else str(index)
        material_id = spec.material_refs.get(name_key, spec.material_refs.get(str(index), f"gltf:{index}"))
        pbr = material.get("pbrMetallicRoughness", {})
        if not isinstance(pbr, Mapping):
            raise ValueError(f"glTF material {index} has malformed pbrMetallicRoughness")
        base_color = pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))
        rgba = tuple(float(value) for value in base_color)
        if len(rgba) != 4 or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in rgba):
            raise ValueError(f"glTF material {index} baseColorFactor must be four values in [0, 1]")
        record = {
            "base_color_rgba": rgba,
            "gltf_base_color_factor_rgba": rgba,
            "metallic_factor": float(pbr.get("metallicFactor", 1.0)),
            "roughness_factor": float(pbr.get("roughnessFactor", 1.0)),
            "alpha_mode": str(material.get("alphaMode", "OPAQUE")),
            "alpha_cutoff": float(material.get("alphaCutoff", 0.5)),
            "double_sided": bool(material.get("doubleSided", False)),
            "gltf_material_name": name,
            "source": "gltf",
        }
        if "baseColorTexture" in pbr:
            texture_info = pbr["baseColorTexture"]
            if not isinstance(texture_info, Mapping):
                raise ValueError(f"glTF material {index} baseColorTexture is malformed")
            record["base_color_texture"] = dict(texture_info)
        if material_id in spec.material_refs.values():
            merged = dict(record)
            manifest_override = normalized.get(material_id, {})
            merged.update(manifest_override)
            manifest_color = manifest_override.get("base_color_rgba", (1.0, 1.0, 1.0, 1.0))
            try:
                override_rgba = tuple(float(value) for value in manifest_color)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"manifest material {material_id!r} base_color_rgba is malformed") from exc
            if (len(override_rgba) != 4
                    or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in override_rgba)):
                raise ValueError(f"manifest material {material_id!r} base_color_rgba must have four values in [0, 1]")
            merged["sensor_texture_base_color_rgba"] = tuple(
                rgba[channel] * override_rgba[channel] for channel in range(4)
            )
            merged["source"] = "manifest+gltf"
            normalized[material_id] = merged
            source_records.append(merged)
        else:
            normalized[material_id] = record
            source_records.append(record)
        ids.append(material_id)
    return normalized, ids, source_records


def _texture_sampler(gltf: Mapping[str, Any], buffers: list[bytes], asset_path: Path,
                     texture_index: Any) -> tuple[np.ndarray, dict[str, Any], int, int]:
    """Decode one untransformed glTF texture and return RGB pixels plus provenance."""
    if isinstance(texture_index, bool) or not isinstance(texture_index, int):
        raise ValueError("baseColorTexture.index must be an integer")
    textures = gltf.get("textures", [])
    if texture_index < 0 or texture_index >= len(textures) or not isinstance(textures[texture_index], Mapping):
        raise ValueError(f"baseColorTexture references invalid glTF texture {texture_index!r}")
    texture = textures[texture_index]
    extensions = texture.get("extensions", {})
    if not isinstance(extensions, Mapping):
        raise ValueError(f"glTF texture {texture_index} extensions are malformed")
    if extensions:
        raise ValueError(f"unsupported glTF texture extension(s): {', '.join(sorted(map(str, extensions)))}")
    image_index = texture.get("source")
    images = gltf.get("images", [])
    if isinstance(image_index, bool) or not isinstance(image_index, int) or image_index < 0 or image_index >= len(images):
        raise ValueError(f"glTF texture {texture_index} has no supported core image source")
    image_desc = images[image_index]
    if not isinstance(image_desc, Mapping):
        raise ValueError(f"glTF image {image_index} is malformed")
    uri = image_desc.get("uri")
    buffer_view = image_desc.get("bufferView")
    if uri is not None and buffer_view is not None:
        raise ValueError(f"glTF image {image_index} must use either uri or bufferView, not both")
    if isinstance(uri, str):
        if uri.startswith("data:"):
            header, sep, content = uri.partition(",")
            if not sep:
                raise ValueError(f"malformed data URI in glTF image {image_index}")
            try:
                image_bytes = (base64.b64decode(content, validate=True) if ";base64" in header
                               else urllib.parse.unquote_to_bytes(content))
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"invalid data URI in glTF image {image_index}") from exc
            source = f"data-uri:image:{image_index}"
        else:
            parsed = urllib.parse.urlparse(uri)
            if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError(f"non-local or decorated glTF image URI is not supported: {uri!r}")
            decoded = urllib.parse.unquote(parsed.path)
            pure = PurePosixPath(decoded)
            if pure.is_absolute() or ".." in pure.parts or decoded in ("", "."):
                raise ValueError(f"glTF image URI escapes or misses its asset directory: {uri!r}")
            image_path = (asset_path.parent / Path(*pure.parts)).resolve()
            try:
                image_path.relative_to(asset_path.parent.resolve())
            except ValueError as exc:
                raise ValueError(f"glTF image URI escapes its asset directory: {uri!r}") from exc
            image_bytes = image_path.read_bytes()
            source = uri
    elif buffer_view is not None:
        view_index = buffer_view
        if isinstance(view_index, bool) or not isinstance(view_index, int):
            raise ValueError(f"glTF image {image_index} has an invalid bufferView")
        views = gltf.get("bufferViews", [])
        if view_index < 0 or view_index >= len(views) or not isinstance(views[view_index], Mapping):
            raise ValueError(f"glTF image {image_index} references invalid bufferView {view_index!r}")
        view = views[view_index]
        view_length = view.get("byteLength")
        if isinstance(view_length, bool) or not isinstance(view_length, int) or view_length < 0:
            raise ValueError(f"glTF image bufferView {view_index} has invalid byteLength")
        image_bytes = bytes(_buffer_view_bytes(gltf, buffers, view_index, 0, view_length))
        source = f"bufferView:{view_index}"
    else:
        raise ValueError(f"glTF image {image_index} has neither a local URI nor an embedded bufferView")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - project environment includes Pillow
        raise RuntimeError("Pillow is required for sensor_texture_mode='triangle_centroid'") from exc
    try:
        with Image.open(io.BytesIO(image_bytes)) as decoded_image:
            rgb = np.asarray(decoded_image.convert("RGB"), dtype=np.float64) / 255.0
    except Exception as exc:
        raise ValueError(f"cannot decode glTF base-color image {image_index}: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.shape[0] or not rgb.shape[1]:
        raise ValueError(f"glTF base-color image {image_index} must decode to non-empty RGB pixels")
    sampler_index = texture.get("sampler")
    sampler: Mapping[str, Any] = {}
    if sampler_index is not None:
        samplers = gltf.get("samplers", [])
        if isinstance(sampler_index, bool) or not isinstance(sampler_index, int) or sampler_index < 0 or sampler_index >= len(samplers):
            raise ValueError(f"glTF texture {texture_index} has invalid sampler index")
        if not isinstance(samplers[sampler_index], Mapping):
            raise ValueError(f"glTF sampler {sampler_index} is malformed")
        sampler = samplers[sampler_index]
    wrap_modes = []
    for key in ("wrapS", "wrapT"):
        mode = sampler.get(key, 10497)
        if isinstance(mode, bool) or mode not in {33071, 33648, 10497}:
            raise ValueError(f"glTF texture sampler {key} must use CLAMP_TO_EDGE, MIRRORED_REPEAT, or REPEAT")
        wrap_modes.append(int(mode))
    source_metadata = {
        "source_image": source,
        "source_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "source_image_size_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "source_image_bytes": len(image_bytes),
        "wrap_s": wrap_modes[0],
        "wrap_t": wrap_modes[1],
    }
    return rgb, source_metadata, wrap_modes[0], wrap_modes[1]


def _wrapped_texel_index(index: int, size: int, mode: int) -> int:
    if mode == 10497:  # REPEAT
        return index % size
    if mode == 33071:  # CLAMP_TO_EDGE
        return min(max(index, 0), size - 1)
    period = 2 * size  # MIRRORED_REPEAT
    folded = index % period
    return folded if folded < size else period - 1 - folded


def _sample_texture_bilinear(rgb: np.ndarray, uv: np.ndarray, wrap_s: int, wrap_t: int) -> np.ndarray:
    """Sample normalized glTF UVs using deterministic bilinear RGB interpolation."""
    height, width, _ = rgb.shape
    samples = np.empty((len(uv), 3), dtype=np.float64)
    for sample_index, coordinates in enumerate(uv):
        wrapped = []
        for coordinate, mode in zip(coordinates, (wrap_s, wrap_t)):
            if mode == 10497:
                wrapped.append(float(coordinate) % 1.0)
            elif mode == 33071:
                wrapped.append(float(np.clip(coordinate, 0.0, 1.0)))
            else:
                mirrored = float(coordinate) % 2.0
                wrapped.append(mirrored if mirrored <= 1.0 else 2.0 - mirrored)
        x = wrapped[0] * width - 0.5
        y = wrapped[1] * height - 0.5
        x0 = math.floor(x)
        y0 = math.floor(y)
        tx = x - x0
        ty = y - y0
        x1 = x0 + 1
        y1 = y0 + 1
        p00 = rgb[_wrapped_texel_index(y0, height, wrap_t), _wrapped_texel_index(x0, width, wrap_s)]
        p10 = rgb[_wrapped_texel_index(y0, height, wrap_t), _wrapped_texel_index(x1, width, wrap_s)]
        p01 = rgb[_wrapped_texel_index(y1, height, wrap_t), _wrapped_texel_index(x0, width, wrap_s)]
        p11 = rgb[_wrapped_texel_index(y1, height, wrap_t), _wrapped_texel_index(x1, width, wrap_s)]
        top = p00 * (1.0 - tx) + p10 * tx
        bottom = p01 * (1.0 - tx) + p11 * tx
        samples[sample_index] = top * (1.0 - ty) + bottom * ty
    return samples


def _load_one_asset(spec: SceneAssetSpec, manifest: SceneAssetManifest,
                    *, verify_checksums: bool) -> SceneMesh:
    if not spec.path.is_file():
        raise FileNotFoundError(f"scene asset does not exist: {spec.path}")
    actual_hash = sha256_file(spec.path)
    if verify_checksums and spec.integrity_sha256 is None:
        raise ValueError(f"SHA-256 is required for production loading of {spec.object_id}; add integrity.sha256")
    if verify_checksums and actual_hash != spec.integrity_sha256:
        raise ValueError(f"SHA-256 mismatch for {spec.object_id}: expected {spec.integrity_sha256}, got {actual_hash}")
    gltf, buffers = _read_gltf(spec.path)
    dependency_hashes = _verify_asset_dependencies(gltf, spec, verify_checksums=verify_checksums)
    materials, gltf_material_ids, gltf_material_records = _gltf_materials(gltf, spec, manifest.materials)

    nodes = gltf.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError(f"glTF nodes must be an array: {spec.path}")
    scenes = gltf.get("scenes", [])
    if scenes:
        scene_index = gltf.get("scene", 0)
        if isinstance(scene_index, bool) or not isinstance(scene_index, int) or scene_index < 0 or scene_index >= len(scenes):
            raise ValueError(f"glTF default scene index is invalid: {spec.path}")
        roots = scenes[scene_index].get("nodes", [])
    else:
        child_nodes = {child for node in nodes if isinstance(node, Mapping) for child in node.get("children", [])}
        roots = [index for index in range(len(nodes)) if index not in child_nodes]
    if not isinstance(roots, list):
        raise ValueError(f"glTF scene root node list is malformed: {spec.path}")

    # glTF uses a right-handed +Y-up basis. Rotate +Y to project +Z while
    # retaining right-handedness and the project's +X forward direction.
    gltf_to_project = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    asset_transform = _asset_matrix(spec)
    vertices_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    triangle_albedo_parts: list[np.ndarray] = []
    triangle_material_ids: list[str | None] = []
    texture_metadata: dict[str, dict[str, Any]] = {}
    texture_cache: dict[int, tuple[np.ndarray, dict[str, Any], int, int]] = {}
    visited: set[int] = set()
    node_stack: set[int] = set()
    vertex_count = 0

    meshes = gltf.get("meshes", [])
    if not isinstance(meshes, list):
        raise ValueError(f"glTF meshes must be an array: {spec.path}")

    def visit(node_index: int, parent_matrix: np.ndarray) -> None:
        nonlocal vertex_count
        if isinstance(node_index, bool) or not isinstance(node_index, int) or node_index < 0 or node_index >= len(nodes):
            raise ValueError(f"glTF scene references invalid node {node_index!r}")
        if node_index in node_stack:
            raise ValueError(f"cycle in glTF node hierarchy at node {node_index}")
        if node_index in visited:
            raise ValueError(f"glTF node {node_index} is referenced more than once")
        node_stack.add(node_index)
        visited.add(node_index)
        node = nodes[node_index]
        if not isinstance(node, Mapping):
            raise ValueError(f"glTF node {node_index} is malformed")
        world = parent_matrix @ _node_matrix(node)
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            if isinstance(mesh_index, bool) or not isinstance(mesh_index, int) or mesh_index < 0 or mesh_index >= len(meshes):
                raise ValueError(f"glTF node {node_index} has invalid mesh reference")
            mesh = meshes[mesh_index]
            primitives = mesh.get("primitives", [])
            if not isinstance(primitives, list):
                raise ValueError(f"glTF mesh {mesh_index} primitives must be an array")
            for primitive in primitives:
                if not isinstance(primitive, Mapping):
                    raise ValueError(f"glTF mesh {mesh_index} has malformed primitive")
                extensions = primitive.get("extensions", {})
                if "KHR_draco_mesh_compression" in extensions or "EXT_meshopt_compression" in extensions:
                    raise ValueError("compressed glTF primitive requires an unsupported decoder")
                attributes = primitive.get("attributes", {})
                if not isinstance(attributes, Mapping) or "POSITION" not in attributes:
                    raise ValueError(f"glTF mesh {mesh_index} primitive has no POSITION attribute")
                positions = _decode_accessor(gltf, buffers, attributes["POSITION"])
                if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
                    raise ValueError(f"glTF mesh {mesh_index} POSITION accessor must be finite VEC3 data")
                if primitive.get("indices") is None:
                    source_indices = np.arange(len(positions), dtype=np.int64)
                else:
                    source_indices = _decode_accessor(gltf, buffers, primitive["indices"]).reshape(-1)
                    if source_indices.dtype.kind not in "ui":
                        raise ValueError(f"glTF mesh {mesh_index} indices must be unsigned integers")
                faces = _triangulate(source_indices, primitive.get("mode", 4))
                if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(positions)):
                    raise ValueError(f"glTF mesh {mesh_index} primitive indices exceed POSITION count")
                material_index = primitive.get("material")
                if material_index is None:
                    material_id = None
                    source_material: Mapping[str, Any] = {}
                elif (isinstance(material_index, bool) or not isinstance(material_index, int)
                      or material_index < 0 or material_index >= len(gltf_material_ids)):
                    raise ValueError(f"glTF primitive has invalid material index {material_index!r}")
                else:
                    material_id = gltf_material_ids[material_index]
                    source_material = gltf_material_records[material_index]
                total = len(positions)
                homogeneous = np.ones((total, 4), dtype=np.float64)
                homogeneous[:, :3] = positions.astype(np.float64, copy=False)
                unit_scale = np.diag((spec.asset_unit_scale_to_m,
                                      spec.asset_unit_scale_to_m,
                                      spec.asset_unit_scale_to_m, 1.0))
                final_matrix = asset_transform @ gltf_to_project @ unit_scale @ world
                transformed = (homogeneous @ final_matrix.T)[:, :3]
                if not np.isfinite(transformed).all():
                    raise ValueError(f"non-finite transformed mesh coordinates for {spec.object_id}")
                if np.linalg.det(final_matrix[:3, :3]) < 0.0 and faces.size:
                    faces = faces[:, (0, 2, 1)]
                vertex_base = vertex_count
                vertices_parts.append(transformed)
                triangle_parts.append(faces.astype(np.int64, copy=False) + vertex_base)
                vertex_count += len(transformed)

                material = materials.get(material_id, {}) if material_id is not None else {}
                flat_color = material.get("base_color_rgba", material.get("base_color_factor", (0.68, 0.68, 0.68, 1.0)))
                try:
                    flat_rgb = np.asarray(tuple(flat_color)[:3], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"material {material_id!r} has an invalid base color") from exc
                if flat_rgb.shape != (3,) or not np.isfinite(flat_rgb).all():
                    raise ValueError(f"material {material_id!r} base color must have finite RGB values")
                primitive_albedos = np.broadcast_to(flat_rgb, (len(faces), 3)).copy()
                texture_mode = source_material.get("sensor_texture_mode")
                if texture_mode == "triangle_centroid":
                    texture_info = source_material.get("base_color_texture")
                    if not isinstance(texture_info, Mapping):
                        raise ValueError(
                            f"material {material_id!r} opts into triangle_centroid but has no baseColorTexture"
                        )
                    texture_extensions = texture_info.get("extensions", {})
                    if not isinstance(texture_extensions, Mapping):
                        raise ValueError(f"material {material_id!r} baseColorTexture extensions are malformed")
                    if "KHR_texture_transform" in texture_extensions:
                        raise ValueError(
                            f"material {material_id!r} uses unsupported KHR_texture_transform"
                        )
                    if texture_extensions:
                        raise ValueError(
                            f"material {material_id!r} uses unsupported baseColorTexture extension(s): "
                            f"{', '.join(sorted(map(str, texture_extensions)))}"
                        )
                    texture_coord = texture_info.get("texCoord", 0)
                    if (isinstance(texture_coord, bool) or not isinstance(texture_coord, int)
                            or texture_coord != 0):
                        raise ValueError(
                            f"material {material_id!r} baseColorTexture must use TEXCOORD_0"
                        )
                    uv_accessor = attributes.get("TEXCOORD_0")
                    if uv_accessor is None:
                        raise ValueError(
                            f"material {material_id!r} opts into triangle_centroid but primitive has no TEXCOORD_0"
                        )
                    uv_values = _decode_accessor(gltf, buffers, uv_accessor).astype(np.float64, copy=False)
                    if (uv_values.ndim != 2 or uv_values.shape != (len(positions), 2)
                            or not np.isfinite(uv_values).all()):
                        raise ValueError(
                            f"material {material_id!r} TEXCOORD_0 must be finite VEC2 data matching POSITION"
                        )
                    texture_index = texture_info.get("index")
                    if isinstance(texture_index, bool) or not isinstance(texture_index, int):
                        raise ValueError(f"material {material_id!r} baseColorTexture.index must be an integer")
                    if texture_index not in texture_cache:
                        texture_cache[texture_index] = _texture_sampler(gltf, buffers, spec.path, texture_index)
                    texture_rgb, image_metadata, wrap_s, wrap_t = texture_cache[texture_index]
                    centroid_uv = uv_values[faces].mean(axis=1)
                    sampled = _sample_texture_bilinear(texture_rgb, centroid_uv, wrap_s, wrap_t)
                    texture_factor = source_material.get(
                        "sensor_texture_base_color_rgba",
                        source_material.get("gltf_base_color_factor_rgba",
                                            source_material.get("base_color_rgba", (1, 1, 1, 1))),
                    )
                    try:
                        multiplier = np.asarray(tuple(texture_factor)[:3], dtype=np.float64)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"material {material_id!r} has an invalid sensor texture multiplier") from exc
                    if multiplier.shape != (3,) or not np.isfinite(multiplier).all():
                        raise ValueError(f"material {material_id!r} sensor texture multiplier must be finite RGB")
                    primitive_albedos = sampled * multiplier
                    metadata = texture_metadata.setdefault(material_id, {
                        "mode": "triangle_centroid",
                        "color_space": "Pillow RGB values normalized by 255; interpolated directly, then multiplied by baseColorFactor and manifest base_color_rgba; no sRGB decode; alpha ignored",
                        "source_images": [],
                    })
                    if not any(item["source_image_sha256"] == image_metadata["source_image_sha256"]
                               for item in metadata["source_images"]):
                        metadata["source_images"].append(dict(image_metadata))
                    if "source_image_sha256" not in metadata:
                        metadata.update({key: image_metadata[key] for key in (
                            "source_image", "source_image_sha256", "source_image_size_px", "source_image_bytes"
                        )})
                elif texture_mode is not None:
                    raise ValueError(f"material {material_id!r} has unsupported sensor_texture_mode {texture_mode!r}")
                triangle_albedo_parts.append(primitive_albedos)
                triangle_material_ids.extend([material_id] * len(faces))
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ValueError(f"glTF node {node_index} children must be an array")
        for child in children:
            visit(child, world)
        node_stack.remove(node_index)

    for root_index in roots:
        visit(root_index, np.eye(4, dtype=np.float64))
    if not vertices_parts or not triangle_parts or not any(len(part) for part in triangle_parts):
        raise ValueError(f"scene asset {spec.object_id} contains no triangulated mesh in its default scene")
    vertices = np.concatenate(vertices_parts, axis=0).astype(np.float64, copy=False)
    triangles = np.concatenate(triangle_parts, axis=0).astype(np.int32, copy=False)
    triangle_albedos = np.concatenate(triangle_albedo_parts, axis=0).astype(np.float64, copy=False)
    if triangle_albedos.shape != (len(triangles), 3):
        raise ValueError(f"scene asset {spec.object_id} has misaligned triangle albedos")
    if texture_metadata:
        triangle_areas = 0.5 * np.linalg.norm(
            np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                     vertices[triangles[:, 2]] - vertices[triangles[:, 0]]),
            axis=1,
        )
        for material_id, metadata in texture_metadata.items():
            selection = np.asarray([item == material_id for item in triangle_material_ids], dtype=bool)
            colors = triangle_albedos[selection]
            areas = triangle_areas[selection]
            metadata["triangle_count"] = int(len(colors))
            metadata["mean_rgb_face_weighted"] = colors.mean(axis=0).tolist() if len(colors) else None
            total_area = float(areas.sum())
            metadata["mean_rgb_area_weighted"] = (
                np.average(colors, axis=0, weights=areas).tolist()
                if len(colors) and total_area > 0.0 else None
            )
            metadata["total_triangle_area_m2"] = total_area
    vertices.flags.writeable = False
    triangles.flags.writeable = False
    triangle_albedos.flags.writeable = False
    return SceneMesh(
        object_id=spec.object_id,
        role=spec.role,
        semantic_role=spec.semantic_role,
        vertices_m=vertices,
        triangles=triangles,
        triangle_material_ids=tuple(triangle_material_ids),
        materials=materials,
        source_path=spec.path,
        source_sha256=actual_hash,
        source_files_sha256=dependency_hashes,
        provenance=dict(spec.provenance),
        triangle_albedos=triangle_albedos,
        texture_metadata=texture_metadata,
    )


def load_scene_assets(manifest_path: str | Path, *, verify_checksums: bool = True) -> LoadedScene:
    """Load and triangulate external glTF assets named by a JSON manifest.

    Asset and buffer paths resolve relative to the scene manifest / glTF file.
    Each object's ``vertices_m`` and ``triangles`` arrays are marked read-only.
    Set ``verify_checksums=False`` only for exploratory loading of an asset
    whose final digest has not yet been recorded; the actual digest is still
    returned on each :class:`SceneMesh`. ``MALECNS_ASSET_ROOT`` takes
    precedence over a manifest's optional ``asset_root``; if neither candidate
    contains a requested file, resolution falls back to the manifest folder.
    """
    scene_manifest = load_scene_manifest(manifest_path)
    objects = tuple(_load_one_asset(spec, scene_manifest, verify_checksums=verify_checksums)
                    for spec in scene_manifest.assets)
    combined_materials = {key: dict(value) for key, value in scene_manifest.materials.items()}
    for obj in objects:
        for material_id, material in obj.materials.items():
            combined_materials.setdefault(material_id, dict(material))
    return LoadedScene(
        scene_id=scene_manifest.scene_id,
        objects=objects,
        materials=combined_materials,
        replay=scene_manifest.replay,
        manifest_path=scene_manifest.path,
        manifest=scene_manifest.to_dict(),
    )


__all__ = ["LoadedScene", "SceneMesh", "load_scene_assets"]
