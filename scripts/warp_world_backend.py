#!/usr/bin/env python3
"""Optional NVIDIA Warp renderer for :mod:`virtual_world`.

``VirtualWorld`` remains the analytic reference/golden backend.  This module
is deliberately an adapter rather than a second world model: it consumes an
existing ``VirtualWorld`` scene, builds one persistent static triangle mesh
and BVH, and uses a separately refitted mesh for timestamped rectangular
stimuli.  Warp's CUDA ``mesh_query_ray`` performs the actual ray/BVH query;
the final analytic shader runs in a small Warp kernel so production callers
can retain a CUDA-resident luminance tensor.

The module is import-safe on machines without Warp.  Construction of
``WarpWorldRenderer`` then fails explicitly with ``WarpBackendUnavailableError``
unless the caller asks the ``VirtualWorld.create_renderer`` seam for an
explicit analytic fallback.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

try:  # Warp is intentionally optional for the analytic/default test path.
    import numpy as np
except Exception as exc:  # pragma: no cover - NumPy is present in supported envs.
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR: Exception | None = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - exercised on the system Python.
    wp = None  # type: ignore[assignment]
    _WARP_IMPORT_ERROR: Exception | None = exc
else:
    _WARP_IMPORT_ERROR = None

try:
    import virtual_world as _world  # type: ignore
except ImportError:  # pragma: no cover - package-style import convenience.
    from . import virtual_world as _world  # type: ignore

CameraConfig = _world.CameraConfig
GrayFrame = _world.GrayFrame
Hit = _world.Hit
Pose = _world.Pose
PhysicalStimulus = _world.PhysicalStimulus
StereoFrames = _world.StereoFrames
VirtualWorld = _world.VirtualWorld
_add = _world._add
_box_basis = _world._box_basis
_coerce_pose = _world._coerce_pose
_cross = _world._cross
_finite = _world._finite
_hash_phase = _world._hash_phase
_scale = _world._scale
_unit = _world._unit
_vec3 = _world._vec3


class WarpBackendUnavailableError(RuntimeError):
    """Raised when the explicit Warp backend cannot provide CUDA rendering."""


class WarpBackendContractError(RuntimeError):
    """Raised when a Warp backend call violates the renderer contract."""


@dataclass(frozen=True)
class _Material:
    object_id: str
    kind: int
    texture: int
    albedo: float
    phase: float
    entry_normal: tuple[float, float, float]
    exit_normal: tuple[float, float, float]
    # Keep the source spelling for public trace metadata; the compact integer
    # texture code above is used by the CUDA shader and intentionally merges
    # aliases such as ``wall``/``checker``.
    texture_name: str = ""


@dataclass(frozen=True)
class _MeshPayload:
    points: Any
    indices: Any
    materials: tuple[_Material, ...]


@dataclass(frozen=True)
class _StimulusVisualMesh:
    """One immutable local-coordinate hand mesh and its resident Warp BVH."""

    mesh: Any
    points: Any
    indices: Any
    vertex_count: int
    face_offset: int
    face_count: int
    source_sha256: str


@dataclass(frozen=True)
class WarpRenderTimings:
    """Measured costs of the most recent Warp frame preparation/query."""

    mesh_prepare_s: float
    dynamic_rebuild_s: float
    dynamic_refit_s: float
    ray_kernel_s: float
    shade_kernel_s: float
    host_copy_s: float

    @property
    def total_s(self) -> float:
        return (self.mesh_prepare_s + self.dynamic_rebuild_s + self.dynamic_refit_s +
                self.ray_kernel_s + self.shade_kernel_s + self.host_copy_s)

    def to_dict(self) -> dict[str, float]:
        result = asdict(self)
        result["total_s"] = self.total_s
        return result


def warp_cuda_available() -> bool:
    """Return whether this interpreter can import Warp and see a CUDA device."""
    if wp is None:
        return False
    try:
        wp.init()
        return bool(wp.get_cuda_devices())
    except Exception:
        return False


def _require_cuda(device: str) -> Any:
    if wp is None:
        detail = f": {_WARP_IMPORT_ERROR}" if _WARP_IMPORT_ERROR is not None else ""
        raise WarpBackendUnavailableError(
            "Warp CUDA backend requested but warp-lang is not importable" + detail
        )
    if np is None:
        detail = f": {_NUMPY_IMPORT_ERROR}" if _NUMPY_IMPORT_ERROR is not None else ""
        raise WarpBackendUnavailableError("Warp backend requires NumPy" + detail)
    try:
        wp.init()
        target = wp.get_device(device)
    except Exception as exc:
        raise WarpBackendUnavailableError(f"could not initialise Warp device {device!r}: {exc}") from exc
    if not target.is_cuda:
        raise WarpBackendUnavailableError(
            f"WarpWorldRenderer requires a CUDA device, got {target!s}; "
            "use VirtualWorld.create_renderer('analytic_reference') explicitly for CPU rendering"
        )
    return target


def _texture_code(texture: str) -> int:
    value = str(texture).lower()
    if value in {"checker", "ground", "wall", "grid"}:
        return 0
    if value in {"vertical_bars", "bars"}:
        return 1
    if value == "horizontal_bars":
        return 2
    if value in {"concentric_checker", "concentric", "target"}:
        return 3
    if value in {"palm", "dark_palm", "matte", "board", "hand"}:
        return 4
    return 5


def _kind_code(kind: str) -> int:
    return {"ground": 0, "target": 1, "obstacle": 2, "wall": 3, "stimulus": 4,
            "scene_mesh": 5}.get(str(kind), 3)


def _material(world: VirtualWorld, object_id: str, kind: str, texture: str,
              albedo: Optional[float] = None) -> _Material:
    return _Material(
        str(object_id), _kind_code(kind), _texture_code(texture),
        -1.0 if albedo is None else float(albedo), _hash_phase(world.seed, str(object_id)),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), str(texture),
    )


def _append_oriented_box(vertices: list[tuple[float, float, float]],
                         indices: list[tuple[int, int, int]],
                         materials: list[_Material], center: Sequence[float],
                         size: Sequence[float], basis: Sequence[Sequence[float]],
                         material: _Material) -> None:
    """Append a closed box with outward winding."""
    half = (0.5 * float(size[0]), 0.5 * float(size[1]), 0.5 * float(size[2]))
    base = len(vertices)
    # Bit order is x, y, z: 0(-,-,-), 1(+,-,-), 2(-,+,-), 3(+,+,-), ...
    for bit in range(8):
        sx = 1.0 if bit & 1 else -1.0
        sy = 1.0 if bit & 2 else -1.0
        sz = 1.0 if bit & 4 else -1.0
        point = _add(
            center,
            _add(_scale(basis[0], sx * half[0]),
                 _add(_scale(basis[1], sy * half[1]), _scale(basis[2], sz * half[2]))),
        )
        vertices.append(point)
    faces = (
        ((1, 3, 7), (1, 7, 5)),  # +x
        ((0, 4, 6), (0, 6, 2)),  # -x
        ((2, 6, 7), (2, 7, 3)),  # +y
        ((0, 1, 5), (0, 5, 4)),  # -y
        ((4, 5, 7), (4, 7, 6)),  # +z
        ((0, 2, 3), (0, 3, 1)),  # -z
    )
    for face_index, (first, second) in enumerate(faces):
        axis = face_index // 2
        face_material = _Material(material.object_id, material.kind, material.texture,
                                  material.albedo, material.phase,
                                  tuple(-float(value) for value in basis[axis]),
                                  tuple(float(value) for value in basis[axis]),
                                  material.texture_name)
        indices.append(tuple(base + item for item in first))
        materials.append(face_material)
        indices.append(tuple(base + item for item in second))
        materials.append(face_material)


def _append_scene_triangles(vertices: list[tuple[float, float, float]],
                            indices: list[tuple[int, int, int]],
                            materials: list[_Material], world: VirtualWorld) -> None:
    """Append the CPU reference's normalized scene triangles without transforms."""
    for triangle in world.scene_triangles:
        base = len(vertices)
        vertices.extend((triangle.a, triangle.b, triangle.c))
        indices.append((base, base + 1, base + 2))
        normal = _unit(_cross(_world._sub(triangle.b, triangle.a),
                              _world._sub(triangle.c, triangle.a)), (0.0, 0.0, 1.0))
        materials.append(_Material(triangle.object_id, _kind_code("scene_mesh"), 5,
                                   float(triangle.albedo), _hash_phase(world.seed, triangle.object_id),
                                   normal, normal, "scene_mesh"))


def _stimulus_mesh_albedos(scene_mesh: Any, face_count: int) -> tuple[float, ...]:
    """Return CPU-reference Rec.709 albedo for every local mesh face.

    New loader meshes carry one RGB sample per triangle.  The fallback keeps
    custom/older SceneMesh producers compatible with the external-scene CPU
    renderer, which reads each material's base-color factor or neutral gray.
    """
    if np is None:  # pragma: no cover
        raise WarpBackendUnavailableError("NumPy is required for stimulus meshes")
    rgb_values = getattr(scene_mesh, "triangle_albedos", None)
    if rgb_values is not None:
        colors = np.asarray(rgb_values, dtype=np.float64)
        if colors.shape == (face_count,):
            if not bool(np.isfinite(colors).all()):
                raise ValueError("SceneMesh.triangle_albedos must contain finite values")
            return tuple(float(np.clip(value, 0.0, 1.0)) for value in colors)
        if colors.shape != (face_count, 3):
            raise ValueError(
                "SceneMesh.triangle_albedos must have shape "
                f"({face_count}, 3), got {colors.shape}"
            )
        if not bool(np.isfinite(colors).all()):
            raise ValueError("SceneMesh.triangle_albedos must contain finite values")
        colors = np.clip(colors, 0.0, 1.0)
        return tuple(float(0.2126 * row[0] + 0.7152 * row[1] + 0.0722 * row[2])
                     for row in colors)

    material_ids = tuple(getattr(scene_mesh, "triangle_material_ids", ()))
    materials = getattr(scene_mesh, "materials", {})
    fallback: list[float] = []
    for index in range(face_count):
        material_id = material_ids[index] if index < len(material_ids) else None
        material = materials.get(material_id, {}) if isinstance(materials, Mapping) else {}
        if not isinstance(material, Mapping):
            material = {}
        color = material.get("base_color_rgba", material.get("base_color_factor"))
        try:
            if color is None or len(color) < 3:
                raise ValueError
            channels = tuple(float(color[channel]) for channel in range(3))
            if not all(math.isfinite(channel) for channel in channels):
                raise ValueError
            channels = tuple(min(1.0, max(0.0, channel)) for channel in channels)
            albedo = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
        except (TypeError, ValueError, IndexError):
            albedo = 0.68
        fallback.append(float(albedo))
    return tuple(fallback)


def _stimulus_mesh_payload(scene_mesh: Any, stimulus_id: str, seed: int) -> _MeshPayload:
    """Validate a local SceneMesh and prepare static per-face Warp materials."""
    if np is None:  # pragma: no cover
        raise WarpBackendUnavailableError("NumPy is required for stimulus meshes")
    try:
        vertices = np.asarray(scene_mesh.vertices_m, dtype=np.float32)
        raw_triangles = np.asarray(scene_mesh.triangles)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"stimulus visual mesh {stimulus_id!r} must provide vertices_m and triangles") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not vertices.shape[0]:
        raise ValueError(f"stimulus visual mesh {stimulus_id!r} vertices_m must have shape (N, 3)")
    if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3 or not raw_triangles.shape[0]:
        raise ValueError(f"stimulus visual mesh {stimulus_id!r} triangles must have shape (M, 3)")
    if not bool(np.isfinite(vertices).all()):
        raise ValueError(f"stimulus visual mesh {stimulus_id!r} vertices_m must be finite")
    try:
        triangle_numbers = np.asarray(raw_triangles, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"stimulus visual mesh {stimulus_id!r} triangle indices must be integers") from exc
    if (not bool(np.isfinite(triangle_numbers).all()) or
            not bool(np.equal(triangle_numbers, np.floor(triangle_numbers)).all())):
        raise ValueError(f"stimulus visual mesh {stimulus_id!r} triangle indices must be finite integers")
    if (bool((triangle_numbers > np.iinfo(np.int32).max).any()) or
            bool((triangle_numbers < np.iinfo(np.int32).min).any())):
        raise ValueError(f"stimulus visual mesh {stimulus_id!r} triangle indices exceed Warp's int32 range")
    triangles = triangle_numbers.astype(np.int32)
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices):
        raise ValueError(f"stimulus visual mesh {stimulus_id!r} has an out-of-range triangle index")
    albedos = _stimulus_mesh_albedos(scene_mesh, len(triangles))
    materials = tuple(
        _Material(str(stimulus_id), _kind_code("scene_mesh"), 5, albedo,
                  _hash_phase(seed, str(stimulus_id)), (0.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0), "stimulus_mesh")
        for albedo in albedos
    )
    return _MeshPayload(vertices, triangles, materials)


def _stimulus_local_axes(direction_m: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """World axes for local mesh coordinates ``+X=normal, +Y=width, +Z=up``."""
    right, up, normal = _box_basis(direction_m)
    return normal, right, up


def _payload_arrays(payload: _MeshPayload) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    if np is None:  # pragma: no cover
        raise WarpBackendUnavailableError("NumPy is required for Warp mesh payloads")
    points = np.asarray(payload.points, dtype=np.float32).reshape((-1, 3))
    flat_indices = np.asarray(payload.indices, dtype=np.int32).reshape((-1,))
    kind = np.asarray([m.kind for m in payload.materials], dtype=np.int32)
    texture = np.asarray([m.texture for m in payload.materials], dtype=np.int32)
    albedo = np.asarray([m.albedo for m in payload.materials], dtype=np.float32)
    phase = np.asarray([m.phase for m in payload.materials], dtype=np.float32)
    entry_normal = np.asarray([m.entry_normal for m in payload.materials], dtype=np.float32).reshape((-1, 3))
    exit_normal = np.asarray([m.exit_normal for m in payload.materials], dtype=np.float32).reshape((-1, 3))
    return points, flat_indices, kind, texture, albedo, phase, entry_normal, exit_normal


if wp is not None:

    @wp.kernel
    def _raycast_kernel(
        scene_mesh_id: wp.uint64,
        dynamic_mesh_id: wp.uint64,
        ground_mesh_id: wp.uint64,
        scene_kind: wp.array(dtype=wp.int32),
        stimulus_mesh_ids: wp.array(dtype=wp.uint64),
        stimulus_mesh_poses: wp.array(dtype=wp.vec3f),
        stimulus_mesh_face_offsets: wp.array(dtype=wp.int32),
        stimulus_mesh_count: wp.int32,
        origin: wp.vec3f,
        directions: wp.array(dtype=wp.vec3f),
        near: wp.float32,
        far: wp.float32,
        hit_t: wp.array(dtype=wp.float32),
        hit_face: wp.array(dtype=wp.int32),
        hit_mesh: wp.array(dtype=wp.int32),
        hit_point: wp.array(dtype=wp.vec3f),
        hit_normal: wp.array(dtype=wp.vec3f),
    ):
        i = wp.tid()
        direction = directions[i]
        start = origin + direction * near
        max_t = far - near
        best_t = max_t
        best_face = wp.int32(-1)
        best_mesh = wp.int32(-1)
        best_normal = wp.vec3f(0.0, 0.0, 0.0)

        # The analytic world only exposes the finite ground to downward rays.
        if direction[2] < -1.0e-9 and max_t > 0.0:
            ground_hit = wp.mesh_query_ray(ground_mesh_id, start, direction, max_t)
            if ground_hit.result and ground_hit.t < best_t:
                best_t = ground_hit.t
                best_face = wp.int32(-1)
                best_mesh = wp.int32(2)
                best_normal = ground_hit.normal

        if max_t > 0.0:
            scene_hit = wp.mesh_query_ray(scene_mesh_id, start, direction, max_t)
            if scene_hit.result and scene_hit.t < best_t:
                best_t = scene_hit.t
                best_face = scene_hit.face
                best_mesh = wp.int32(0)
                best_normal = scene_hit.normal

            if dynamic_mesh_id != wp.uint64(0):
                dynamic_hit = wp.mesh_query_ray(dynamic_mesh_id, start, direction, max_t)
                if dynamic_hit.result and dynamic_hit.t < best_t:
                    best_t = dynamic_hit.t
                    best_face = dynamic_hit.face
                    best_mesh = wp.int32(1)
                    best_normal = dynamic_hit.normal

            # Bound hand geometry stays in a local frame with a resident BVH.
            # Only four pose vectors per binding are updated each frame; no
            # hand vertices are copied and no BVH is refitted after startup.
            for stimulus_mesh_index in range(stimulus_mesh_count):
                pose_base = stimulus_mesh_index * 4
                hand_center = stimulus_mesh_poses[pose_base]
                hand_x = stimulus_mesh_poses[pose_base + 1]
                hand_y = stimulus_mesh_poses[pose_base + 2]
                hand_z = stimulus_mesh_poses[pose_base + 3]
                if wp.dot(hand_x, hand_x) > 0.5:
                    relative_start = start - hand_center
                    local_start = wp.vec3(wp.dot(relative_start, hand_x),
                                           wp.dot(relative_start, hand_y),
                                           wp.dot(relative_start, hand_z))
                    local_direction = wp.vec3(wp.dot(direction, hand_x),
                                              wp.dot(direction, hand_y),
                                              wp.dot(direction, hand_z))
                    hand_hit = wp.mesh_query_ray(stimulus_mesh_ids[stimulus_mesh_index],
                                                 local_start, local_direction, max_t)
                    if hand_hit.result and hand_hit.t < best_t:
                        best_t = hand_hit.t
                        best_face = stimulus_mesh_face_offsets[stimulus_mesh_index] + hand_hit.face
                        best_mesh = wp.int32(3)
                        local_normal = hand_hit.normal
                        world_normal = (hand_x * local_normal[0] +
                                        hand_y * local_normal[1] +
                                        hand_z * local_normal[2])
                        normal_length = wp.sqrt(wp.max(0.0, wp.dot(world_normal, world_normal)))
                        if normal_length > 1.0e-12:
                            world_normal = world_normal / normal_length
                        if wp.dot(world_normal, direction) > 0.0:
                            world_normal = world_normal * -1.0
                        best_normal = world_normal

        # External scene triangles are two-sided. Match the CPU reference's
        # public Hit convention by orienting their normal toward the ray
        # origin before either tracing or shading returns it.
        if best_mesh == 0 and best_face >= 0 and scene_kind[best_face] == 5:
            normal_length = wp.sqrt(wp.max(0.0, wp.dot(best_normal, best_normal)))
            if normal_length > 1.0e-12:
                best_normal = best_normal / normal_length
            if wp.dot(best_normal, direction) > 0.0:
                best_normal = best_normal * -1.0

        hit_t[i] = near + best_t if best_mesh >= 0 else -1.0
        hit_face[i] = best_face
        hit_mesh[i] = best_mesh
        hit_normal[i] = best_normal
        hit_point[i] = start + direction * best_t if best_mesh >= 0 else wp.vec3f(0.0, 0.0, 0.0)


    @wp.kernel
    def _shade_kernel(
        directions: wp.array(dtype=wp.vec3f),
        hit_face: wp.array(dtype=wp.int32),
        hit_mesh: wp.array(dtype=wp.int32),
        hit_point: wp.array(dtype=wp.vec3f),
        hit_normal: wp.array(dtype=wp.vec3f),
        scene_kind: wp.array(dtype=wp.int32),
        scene_texture: wp.array(dtype=wp.int32),
        scene_albedo: wp.array(dtype=wp.float32),
        scene_phase: wp.array(dtype=wp.float32),
        scene_entry_normal: wp.array(dtype=wp.vec3f),
        scene_exit_normal: wp.array(dtype=wp.vec3f),
        dynamic_kind: wp.array(dtype=wp.int32),
        dynamic_texture: wp.array(dtype=wp.int32),
        dynamic_albedo: wp.array(dtype=wp.float32),
        dynamic_phase: wp.array(dtype=wp.float32),
        dynamic_entry_normal: wp.array(dtype=wp.vec3f),
        dynamic_exit_normal: wp.array(dtype=wp.vec3f),
        stimulus_mesh_albedo: wp.array(dtype=wp.float32),
        light_direction: wp.vec3f,
        ambient: wp.float32,
        intensity: wp.float32,
        sky_brightness: wp.float32,
        ground_cell: wp.float32,
        wall_cell: wp.float32,
        contrast: wp.float32,
        target_center: wp.vec3f,
        ground_phase: wp.float32,
        output: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        mesh_kind = hit_mesh[i]
        if mesh_kind < 0:
            value = sky_brightness + 0.045 * wp.clamp(directions[i][2], -1.0, 1.0)
            output[i] = wp.clamp(value, 0.0, 1.0)
            return

        kind = wp.int32(0)
        texture = wp.int32(0)
        albedo_value = wp.float32(-1.0)
        phase = wp.float32(0.0)
        entry_normal = wp.vec3f(0.0, 0.0, 0.0)
        exit_normal = wp.vec3f(0.0, 0.0, 0.0)
        if mesh_kind == 2:
            # The finite ground is represented by a dedicated two-triangle mesh.
            kind = wp.int32(0)
            texture = wp.int32(0)
            phase = ground_phase
        elif mesh_kind == 0:
            face = hit_face[i]
            kind = scene_kind[face]
            texture = scene_texture[face]
            albedo_value = scene_albedo[face]
            phase = scene_phase[face]
            entry_normal = scene_entry_normal[face]
            exit_normal = scene_exit_normal[face]
        elif mesh_kind == 1:
            face = hit_face[i]
            kind = dynamic_kind[face]
            texture = dynamic_texture[face]
            albedo_value = dynamic_albedo[face]
            phase = dynamic_phase[face]
            entry_normal = dynamic_entry_normal[face]
            exit_normal = dynamic_exit_normal[face]
        else:
            face = hit_face[i]
            kind = wp.int32(5)
            texture = wp.int32(5)
            albedo_value = stimulus_mesh_albedo[face]

        x = hit_point[i][0]
        y = hit_point[i][1]
        cell = ground_cell
        if kind == 3:
            cell = wall_cell
        elif kind == 1 or kind == 2 or kind == 4:
            cell = wp.max(0.04, 0.75 * ground_cell)

        if texture == 1:
            pattern = 0.5 + 0.5 * wp.sin(6.283185307179586 * x / cell + phase)
        elif texture == 2:
            pattern = 0.5 + 0.5 * wp.sin(6.283185307179586 * y / cell + phase)
        elif texture == 3:
            dx = x - target_center[0]
            dy = y - target_center[1]
            radius = wp.sqrt(dx * dx + dy * dy)
            rings = 0.5 + 0.5 * wp.sin(6.283185307179586 * radius / cell + phase)
            sectors = 0.5 + 0.5 * wp.sin(4.0 * wp.atan2(dy, dx) + phase * 0.37)
            pattern = 0.5 + 0.5 * (2.0 * rings - 1.0) * (2.0 * sectors - 1.0)
        elif texture == 4:
            sx = wp.sin(6.283185307179586 * x / wp.max(cell * 4.0, 0.16) + phase)
            sy = wp.sin(6.283185307179586 * y / wp.max(cell * 5.0, 0.20) - phase * 0.37)
            pattern = 0.5 + 0.25 * sx + 0.25 * sy
        elif texture == 0:
            sx = wp.sin(6.283185307179586 * x / cell + phase)
            sy = wp.sin(6.283185307179586 * y / cell - phase * 0.61)
            pattern = 0.5 + 0.5 * sx * sy
        else:
            pattern = 0.5 + 0.5 * wp.sin(6.283185307179586 * (0.7 * x + 0.3 * y) / cell + phase)

        base = wp.float32(0.55)
        if kind == 0:
            base = wp.float32(0.43)
        elif kind == 1:
            base = wp.float32(0.74)
        elif kind == 2:
            base = wp.float32(0.52)
        elif kind == 4:
            base = wp.where(albedo_value < 0.0, wp.float32(0.66), albedo_value)
        elif kind == 5:
            base = wp.where(albedo_value < 0.0, wp.float32(0.68), albedo_value)
        local_contrast = wp.where(kind == 5, wp.float32(0.0),
                                  contrast * wp.where(kind == 1, wp.float32(1.25), wp.float32(1.0)))
        shaded_albedo = wp.clamp(base * (1.0 - local_contrast) +
                                 base * local_contrast * (0.35 + 1.3 * pattern), 0.0, 1.0)
        normal = hit_normal[i]
        # Scene-triangle normals were oriented toward the ray origin in the
        # raycast kernel. Keep the analytic reference's historical AABB
        # entry/exit convention for its box geometry and stimuli.
        if mesh_kind != 2 and kind != 5:
            normal = entry_normal if wp.dot(normal, directions[i]) < 0.0 else exit_normal
        lambert = wp.max(0.0, normal[0] * light_direction[0] +
                         normal[1] * light_direction[1] + normal[2] * light_direction[2])
        output[i] = wp.clamp(shaded_albedo * (ambient + intensity * lambert), 0.0, 1.0)


    @wp.kernel
    def _head_to_world_kernel(
        head_directions: wp.array(dtype=wp.vec3f),
        forward: wp.vec3f,
        right: wp.vec3f,
        up: wp.vec3f,
        output: wp.array(dtype=wp.vec3f),
    ):
        i = wp.tid()
        head = head_directions[i]
        world_direction = forward * head[0] + right * (-head[1]) + up * head[2]
        length = wp.sqrt(wp.max(0.0, wp.dot(world_direction, world_direction)))
        output[i] = world_direction / length


class WarpWorldRenderer:
    """CUDA renderer adapter backed by persistent Warp meshes and BVHs."""

    backend_name = "warp_cuda"
    backend_role = "accelerated_renderer"
    backend_contract = "world-renderer-v1"
    is_stub = False

    def __init__(self, world: VirtualWorld, *, device: str = "cuda:0"):
        if not isinstance(world, VirtualWorld):
            raise TypeError("WarpWorldRenderer requires a VirtualWorld instance")
        self.world = world
        self.device_name = str(device)
        self.device = _require_cuda(self.device_name)
        self._static_materials: tuple[_Material, ...] = ()
        self._static_kind: Any = None
        self._static_texture: Any = None
        self._static_albedo: Any = None
        self._static_phase: Any = None
        self._static_entry_normal: Any = None
        self._dynamic_materials: tuple[_Material, ...] = ()
        self._dynamic_kind: Any = None
        self._dynamic_texture: Any = None
        self._dynamic_albedo: Any = None
        self._dynamic_phase: Any = None
        self._dynamic_entry_normal: Any = None
        self._dynamic_exit_normal: Any = None
        self._dynamic_active = False
        self._dynamic_points: Any = None
        self._dynamic_mesh: Any = None
        self._dynamic_signature: tuple[str, ...] = ()
        self._dynamic_key: Any = None
        self._scene_mesh: Any = None
        self._ground_mesh: Any = None
        self._stimulus_visual_meshes: dict[str, _StimulusVisualMesh] = {}
        self._stimulus_visual_mesh_order: tuple[str, ...] = ()
        self._stimulus_visual_materials: tuple[_Material, ...] = ()
        self._stimulus_mesh_ids_device: Any = None
        self._stimulus_mesh_poses_device: Any = None
        self._stimulus_mesh_face_offsets_device: Any = None
        self._stimulus_mesh_albedo_device: Any = None
        self._stimulus_mesh_pose_host: Any = None
        self._stimulus_mesh_build_count = 0
        self._stimulus_mesh_pose_upload_count = 0
        self._stimulus_mesh_vertex_upload_count = 0
        self._last_luminance: Any = None
        self._output_buffers: dict[tuple[int, int], tuple[Any, Any, Any, Any, Any, Any]] = {}
        self._next_buffer_slot: dict[int, int] = {}
        # Head-frame direction arrays are immutable in the normal FlyVis
        # production path.  Keep both the source array (or host upload) and
        # its transformed world-direction buffer alive so repeated frames do
        # not allocate/copy 721 elements.  The stream handle is part of the
        # key: a buffer cannot be shared by concurrent CUDA streams without a
        # synchronization edge.
        self._head_cache: dict[Any, tuple[Any, Any, Any]] = {}
        self._last_timings = WarpRenderTimings(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._dynamic_rebuild_count = 0
        self._dynamic_refit_count = 0
        self._build_static_meshes()

    @classmethod
    def from_world(cls, world: VirtualWorld, *, device: str = "cuda:0") -> "WarpWorldRenderer":
        return cls(world, device=device)

    @property
    def last_timings(self) -> WarpRenderTimings:
        return self._last_timings

    @property
    def mesh_stats(self) -> dict[str, Any]:
        stats = {
            "device": self.device_name,
            "static_mesh_resident": self._scene_mesh is not None,
            "ground_mesh_resident": self._ground_mesh is not None,
            "dynamic_mesh_resident": self._dynamic_mesh is not None,
            "dynamic_mesh_active": self._dynamic_active,
            "static_faces": len(self._static_materials),
            "dynamic_faces": len(self._dynamic_materials),
            "dynamic_rebuild_count": self._dynamic_rebuild_count,
            "dynamic_refit_count": self._dynamic_refit_count,
            "stimulus_visual_mesh_count": len(self._stimulus_visual_meshes),
            "stimulus_visual_faces": len(self._stimulus_visual_materials),
            "stimulus_visual_bvh_build_count": self._stimulus_mesh_build_count,
            "stimulus_visual_bvh_refit_count": 0,
            "stimulus_visual_vertex_upload_count": self._stimulus_mesh_vertex_upload_count,
            "stimulus_visual_pose_upload_count": self._stimulus_mesh_pose_upload_count,
            "stimulus_visual_meshes": [
                {
                    "stimulus_id": stimulus_id,
                    "vertices": binding.vertex_count,
                    "faces": binding.face_count,
                    "source_sha256": binding.source_sha256,
                }
                for stimulus_id, binding in self._stimulus_visual_meshes.items()
            ],
            "backend": self.backend_name,
        }
        if self.world.scene_path is not None:
            stats["external_scene_faces"] = len(self.world.scene_triangles)
        return stats

    def manifest(self) -> dict[str, Any]:
        """Return explicit backend provenance and transfer/cache policy."""
        device = self.device
        return {
            "schema_version": "warp-world-renderer-1",
            "backend": self.backend_name,
            "backend_role": self.backend_role,
            "backend_contract": self.backend_contract,
            "requested_backend": "warp_cuda",
            "effective_backend": self.backend_name,
            "warp_version": str(getattr(wp.config, "version", "unknown")),
            "device": self.device_name,
            "device_name": str(getattr(device, "name", self.device_name)),
            "architecture": f"sm_{getattr(device, 'arch', 'unknown')}",
            "kernel_cache": str(getattr(wp.config, "kernel_cache_dir", "unknown")),
            "stream": {
                "policy": "caller_torch_stream_or_warp_default",
                "device_wide_sync_in_production": False,
                "benchmark_sync": "explicit_device_sync_around_measurement",
            },
            "zero_copy": {
                "tensor_output": True,
                "torch_cuda_input": True,
                "list_output": False,
                "host_boundary": "list_and_trace_only",
            },
            "transfer": {
                "tensor_output": "Warp array aliases Torch CUDA storage",
                "host_direction_input": "one bounded host_to_device copy",
                "torch_direction_input": "DLPack alias",
                "list_output": "explicit D2H compatibility wrapper",
            },
            "jit": {
                "kernels": ["_head_to_world_kernel", "_raycast_kernel", "_shade_kernel"],
                "cold_start_includes": ["Warp JIT", "static mesh/BVH construction"],
                "steady_state": "static scene and bound hand BVHs resident; only dynamic boxes refit when active",
            },
            "mesh_strategy": {
                "static": ("resident triangle Mesh/BVH (walls,target,obstacles,optional external visual scene)"
                           if self.world.scene_path is not None else
                           "resident triangle Mesh/BVH (walls,target,obstacles)"),
                "ground": "resident finite two-triangle Mesh/BVH",
                "dynamic": "separate topology; refit for active geometry changes; rebuild on onset/ID change",
                "stimulus_visual": (
                    "one resident local triangle Mesh/BVH per bound stimulus; rigid pose transform; "
                    "vertices uploaded once at renderer construction and never refit"
                ),
            },
            "stimulus_visual_shading": {
                "face_albedo": "per-triangle Rec.709 luminance from loader RGB samples",
                "lighting": "shared Lambert ambient + directional intensity shader",
                "pose": "rigid transform from StimulusState.position_m and direction_m; size_m is not applied",
            },
            "tie_policy": {
                "analytic_priority": ["ground", "walls", "target", "obstacles", "stimuli"],
                "implementation": "ground uses a dedicated query; static scene uses one combined BVH and strict '<' nearest-hit selection",
                "static_exact_ties": "rare exact wall/target/obstacle coplanar ties follow Warp BVH face order and are not guaranteed to match analytic insertion order",
                "multi_stimulus_exact_ties": "selected stimulus geometry is combined; overlapping exact ties follow Warp BVH face order",
            },
            "mesh_stats": self.mesh_stats,
            "memory_budget": {
                "ram_limit_gb": 5.0,
                "vram_limit_gb": 15.5,
                "observed": False,
                "note": "bounded scene allocations; runtime telemetry not sampled by this manifest",
            },
            **({"external_scene": self.world.external_scene_manifest}
               if self.world.scene_path is not None else {}),
        }

    def _mesh(self, payload: _MeshPayload) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        points, indices, kind, texture, albedo, phase, entry_normal, exit_normal = _payload_arrays(payload)
        points_device = wp.array(points, dtype=wp.vec3, device=self.device)
        indices_device = wp.array(indices, dtype=wp.int32, device=self.device)
        mesh = wp.Mesh(points=points_device, indices=indices_device, bvh_constructor="lbvh")
        kind_device = wp.array(kind if len(kind) else np.zeros(1, dtype=np.int32), dtype=wp.int32, device=self.device)
        texture_device = wp.array(texture if len(texture) else np.zeros(1, dtype=np.int32), dtype=wp.int32, device=self.device)
        albedo_device = wp.array(albedo if len(albedo) else np.full(1, -1.0, dtype=np.float32), dtype=wp.float32, device=self.device)
        phase_device = wp.array(phase if len(phase) else np.zeros(1, dtype=np.float32), dtype=wp.float32, device=self.device)
        entry_normal_device = wp.array(entry_normal if len(entry_normal) else np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3, device=self.device)
        exit_normal_device = wp.array(exit_normal if len(exit_normal) else np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3, device=self.device)
        return mesh, points_device, kind_device, texture_device, albedo_device, phase_device, entry_normal_device, exit_normal_device

    def _build_static_meshes(self) -> None:
        started = time.perf_counter()
        vertices: list[tuple[float, float, float]] = []
        indices: list[tuple[int, int, int]] = []
        materials: list[_Material] = []
        basis = _box_basis((0.0, 0.0, 1.0))
        for wall in self.world.walls:
            _append_oriented_box(vertices, indices, materials, wall.center_m, wall.size_m, basis,
                                 _material(self.world, wall.id, "wall", wall.texture))
        target = self.world.target
        _append_oriented_box(vertices, indices, materials, target.position_m, target.size_m, basis,
                             _material(self.world, target.id, "target", target.texture))
        for obstacle in self.world.obstacles:
            _append_oriented_box(vertices, indices, materials, obstacle.center_m, obstacle.size_m, basis,
                                 _material(self.world, obstacle.id, "obstacle", obstacle.texture))
        _append_scene_triangles(vertices, indices, materials, self.world)
        payload = _MeshPayload(vertices, indices, tuple(materials))
        (self._scene_mesh, _scene_points, self._static_kind, self._static_texture,
         self._static_albedo, self._static_phase, self._static_entry_normal,
         self._static_exit_normal) = self._mesh(payload)
        self._static_materials = tuple(materials)

        ground_vertices: list[tuple[float, float, float]] = []
        ground_indices: list[tuple[int, int, int]] = []
        xmin, xmax, ymin, ymax = (float(v) for v in self.world.config.bounds_xy_m)
        base = len(ground_vertices)
        ground_vertices.extend(((xmin, ymin, self.world.config.ground_z_m),
                                (xmax, ymin, self.world.config.ground_z_m),
                                (xmax, ymax, self.world.config.ground_z_m),
                                (xmin, ymax, self.world.config.ground_z_m)))
        ground_indices.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
        ground_payload = _MeshPayload(ground_vertices, ground_indices, ())
        (self._ground_mesh, _ground_points, _ground_kind, _ground_texture, _ground_albedo,
         _ground_phase, _ground_entry_normal, _ground_exit_normal) = self._mesh(ground_payload)
        self._build_stimulus_visual_meshes()
        self._static_mesh_prepare_s = time.perf_counter() - started

    def _build_stimulus_visual_meshes(self) -> None:
        """Upload each bound local hand once and retain its immutable BVH."""
        bindings = getattr(self.world, "stimulus_visual_meshes", {}) or {}
        if not isinstance(bindings, Mapping):
            raise TypeError("world.stimulus_visual_meshes must be a mapping")
        ordered = sorted(bindings.items(), key=lambda item: str(item[0]))
        material_rows: list[_Material] = []
        face_offsets: list[int] = []
        mesh_ids: list[int] = []
        for raw_id, scene_mesh in ordered:
            stimulus_id = str(raw_id)
            if stimulus_id in self._stimulus_visual_meshes:
                raise ValueError(f"duplicate stimulus visual mesh binding {stimulus_id!r}")
            payload = _stimulus_mesh_payload(scene_mesh, stimulus_id, self.world.seed)
            points, flat_indices, _kind, _texture, _albedo, _phase, _entry, _exit = _payload_arrays(payload)
            points_device = wp.array(points, dtype=wp.vec3, device=self.device)
            indices_device = wp.array(flat_indices, dtype=wp.int32, device=self.device)
            mesh = wp.Mesh(points=points_device, indices=indices_device, bvh_constructor="lbvh")
            face_offsets.append(len(material_rows))
            mesh_ids.append(int(mesh.id))
            material_rows.extend(payload.materials)
            binding = _StimulusVisualMesh(
                mesh=mesh,
                points=points_device,
                indices=indices_device,
                vertex_count=int(points.shape[0]),
                face_offset=face_offsets[-1],
                face_count=len(payload.materials),
                source_sha256=str(getattr(scene_mesh, "source_sha256", "")),
            )
            self._stimulus_visual_meshes[stimulus_id] = binding
            self._stimulus_mesh_build_count += 1
            self._stimulus_mesh_vertex_upload_count += 1

        self._stimulus_visual_mesh_order = tuple(self._stimulus_visual_meshes)
        self._stimulus_visual_materials = tuple(material_rows)
        mesh_count = len(mesh_ids)
        # Keep non-empty device buffers even when there are no bound meshes;
        # the kernel receives an explicit count of zero and never reads them.
        self._stimulus_mesh_ids_device = wp.array(
            np.asarray(mesh_ids if mesh_ids else [0], dtype=np.uint64),
            dtype=wp.uint64, device=self.device,
        )
        self._stimulus_mesh_face_offsets_device = wp.array(
            np.asarray(face_offsets if face_offsets else [0], dtype=np.int32),
            dtype=wp.int32, device=self.device,
        )
        self._stimulus_mesh_pose_host = np.zeros((max(1, 4 * mesh_count), 3), dtype=np.float32)
        self._stimulus_mesh_poses_device = wp.array(
            self._stimulus_mesh_pose_host, dtype=wp.vec3, device=self.device,
        )
        albedo_values = np.asarray([item.albedo for item in material_rows], dtype=np.float32)
        if not len(albedo_values):
            albedo_values = np.asarray([0.68], dtype=np.float32)
        self._stimulus_mesh_albedo_device = wp.array(
            albedo_values, dtype=wp.float32, device=self.device,
        )

    def _update_stimulus_visual_poses(self, active_states: Mapping[str, Any], *, stream: Any = None) -> None:
        """Upload the small rigid-pose block while reusing resident local meshes."""
        if not self._stimulus_visual_mesh_order:
            return
        pose_rows = self._stimulus_mesh_pose_host
        pose_rows.fill(0.0)
        for mesh_index, stimulus_id in enumerate(self._stimulus_visual_mesh_order):
            state = active_states.get(stimulus_id)
            if state is None:
                continue
            axis_x, axis_y, axis_z = _stimulus_local_axes(state.direction_m)
            base = 4 * mesh_index
            pose_rows[base] = state.position_m
            pose_rows[base + 1] = axis_x
            pose_rows[base + 2] = axis_y
            pose_rows[base + 3] = axis_z
        if stream is None:
            self._stimulus_mesh_poses_device.assign(pose_rows)
        else:
            pose_host = wp.array(pose_rows, dtype=wp.vec3, device="cpu")
            wp.copy(self._stimulus_mesh_poses_device, pose_host, stream=stream)
        self._stimulus_mesh_pose_upload_count += 1

    def _geometry_points(self, active: Sequence[Any]) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], tuple[_Material, ...]]:
        vertices: list[tuple[float, float, float]] = []
        indices: list[tuple[int, int, int]] = []
        materials: list[_Material] = []
        for item, state in active:
            geometry = state.geometry
            basis = geometry.basis
            material = _material(self.world, geometry.id, "stimulus", geometry.texture, geometry.albedo)
            _append_oriented_box(vertices, indices, materials, geometry.center_m, geometry.size_m, basis, material)
        return vertices, indices, tuple(materials)

    def _prepare_dynamic(self, pose: Pose, timestamp_s: float, stimulus: Any, stimuli: Any,
                         *, stream: Any = None) -> tuple[float, float, float]:
        started = time.perf_counter()
        selected = self.world._resolve_stimuli(stimulus, stimuli)
        active: list[tuple[Any, Any]] = []
        active_visual_meshes: dict[str, Any] = {}
        for item in selected:
            state = item.state_at(timestamp_s, pose)
            if state.active and state.renderable:
                if str(state.id) in self._stimulus_visual_meshes:
                    active_visual_meshes[str(state.id)] = state
                else:
                    active.append((item, state))
        signature = tuple(str(state.id) for _item, state in active)
        key = (signature, float(timestamp_s), tuple(float(v) for v in pose.position_m),
               float(pose.yaw_rad), float(pose.pitch_rad), float(pose.roll_rad))
        if self._stimulus_visual_mesh_order:
            key = key + (tuple(sorted(active_visual_meshes)),)
        if key == self._dynamic_key:
            return 0.0, 0.0, time.perf_counter() - started
        self._update_stimulus_visual_poses(active_visual_meshes, stream=stream)

        # Warp does not accept an empty Mesh.  Keep a previously allocated
        # dynamic mesh resident (so a later stimulus onset can reuse its
        # allocation), but disable it by leaving the mesh id out of the ray
        # query and expose dummy material arrays only to the never-taken shade
        # branch.
        if not active:
            self._dynamic_signature = ()
            self._dynamic_materials = ()
            self._dynamic_kind = None
            self._dynamic_texture = None
            self._dynamic_albedo = None
            self._dynamic_phase = None
            self._dynamic_entry_normal = None
            self._dynamic_exit_normal = None
            self._dynamic_active = False
            self._dynamic_key = key
            return 0.0, 0.0, time.perf_counter() - started

        rebuild_s = 0.0
        refit_s = 0.0
        vertices, indices, materials = self._geometry_points(active)
        if signature != self._dynamic_signature or self._dynamic_mesh is None:
            rebuild_started = time.perf_counter()
            if self._dynamic_mesh is not None:
                # Synchronise before replacing a mesh that a previous kernel
                # may still reference.  The mesh is tiny; this is a safety
                # boundary rather than a hot path for steady-state frames.
                wp.synchronize_device(self.device)
            payload = _MeshPayload(vertices, indices, materials)
            (self._dynamic_mesh, self._dynamic_points, self._dynamic_kind,
             self._dynamic_texture, self._dynamic_albedo, self._dynamic_phase,
             self._dynamic_entry_normal, self._dynamic_exit_normal) = self._mesh(payload)
            self._dynamic_materials = materials
            self._dynamic_signature = signature
            self._dynamic_active = True
            self._dynamic_rebuild_count += 1
            rebuild_s = time.perf_counter() - rebuild_started
        elif self._dynamic_points is not None:
            refit_started = time.perf_counter()
            points_np = np.asarray(vertices, dtype=np.float32).reshape((-1, 3))
            if stream is None:
                self._dynamic_points.assign(points_np)
                self._dynamic_mesh.refit()
            else:
                points_host = wp.array(points_np, dtype=wp.vec3, device="cpu")
                wp.copy(self._dynamic_points, points_host, stream=stream)
                with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                    self._dynamic_mesh.refit()
            self._dynamic_refit_count += 1
            refit_s = time.perf_counter() - refit_started
            self._dynamic_active = True
        self._dynamic_key = key
        return rebuild_s, refit_s, time.perf_counter() - started

    @staticmethod
    def _camera_rays(world: VirtualWorld, pose: Pose, eye: str, width: int, height: int,
                     camera: CameraConfig | None = None) -> tuple[Any, Any]:
        if np is None:  # pragma: no cover
            raise WarpBackendUnavailableError("NumPy is required for camera rays")
        # ``render_eye_tensor`` has already resolved this camera.  Accepting it
        # here avoids rebuilding/validating an identical CameraConfig per eye.
        camera = world._resolve_camera((width, height)) if camera is None else camera
        forward, right, up = world._camera_basis(pose)
        origin = tuple(pose.position_m[i] + right[i] * camera.eye_separation_m * world._eye_sign(eye) for i in range(3))
        directions: list[tuple[float, float, float]] = []
        tan_h = math.tan(camera.hfov_rad / 2.0)
        tan_v = math.tan(camera.vfov_rad / 2.0)
        for py in range(height):
            for px in range(width):
                nx = (2.0 * ((px + 0.5) / camera.width) - 1.0) * tan_h
                ny = (1.0 - 2.0 * ((py + 0.5) / camera.height)) * tan_v
                ray = _unit((forward[0] + right[0] * nx + up[0] * ny,
                             forward[1] + right[1] * nx + up[1] * ny,
                             forward[2] + right[2] * nx + up[2] * ny))
                directions.append(ray)
        return origin, np.asarray(directions, dtype=np.float32)

    @staticmethod
    def _stream_cache_key(stream: Any) -> Any:
        if stream is None:
            return ("default",)
        # Warp creates a lightweight Stream wrapper for a Torch stream on
        # every conversion.  The CUDA handle, rather than wrapper identity,
        # is the stable key that lets the persistent direction buffer survive
        # those conversions.
        handle = getattr(stream, "cuda_stream", None)
        if handle is not None:
            try:
                return ("cuda", int(handle), str(getattr(stream, "device", "")))
            except (TypeError, ValueError):
                pass
        return ("object", id(stream))

    def _validate_torch_device(self, tensor: Any, torch: Any) -> None:
        """Reject a CUDA direction tensor on a device other than this renderer."""
        renderer_device = torch.device(self.device_name)
        renderer_index = renderer_device.index
        if renderer_index is None:
            renderer_index = torch.cuda.current_device()
        source_device = torch.device(tensor.device)
        source_index = source_device.index
        if (source_device.type != "cuda" or source_index != renderer_index):
            raise WarpBackendContractError(
                "CUDA direction tensor device does not match Warp renderer: "
                f"tensor={source_device}, renderer={renderer_device}"
            )

    def _head_directions(self, pose: Pose, directions: Any, eye: str, *, stream: Any = None) -> tuple[Any, Any]:
        if np is None:  # pragma: no cover
            raise WarpBackendUnavailableError("NumPy is required for direction rays")
        forward, right, up = self.world._camera_basis(pose)
        origin = tuple(pose.position_m[i] + right[i] * self.world.camera.eye_separation_m * self.world._eye_sign(eye) for i in range(3))
        # A CUDA Torch tensor can be wrapped without a host round trip.  The
        # transform/normalisation then runs in Warp and stays device-resident.
        if wp is not None:
            try:
                import torch
                if isinstance(directions, torch.Tensor):
                    if not directions.is_cuda:
                        raise WarpBackendContractError(
                            "Warp CUDA direction tensors must be CUDA tensors; "
                            "pass a list/NumPy array for an explicit host boundary"
                        )
                    self._validate_torch_device(directions, torch)
                    if directions.ndim != 2 or directions.shape[1] != 3:
                        raise ValueError("directions must have shape (N, 3)")
                    if len(directions) == 0:
                        raise ValueError("directions must contain at least one direction")
                    if not directions.is_contiguous():
                        directions = directions.contiguous()
                    stream_key = self._stream_cache_key(stream)
                    key = ("torch", int(directions.data_ptr()), tuple(directions.shape), stream_key)
                    cached = self._head_cache.get(key)
                    if cached is None:
                        head_device = wp.from_torch(directions, dtype=wp.vec3)
                        world_device = wp.empty(len(directions), dtype=wp.vec3, device=self.device)
                        # Retain the Tensor as well as the DLPack view.  This
                        # prevents a caller's temporary tensor from being
                        # collected while an asynchronous kernel still reads
                        # its aliased storage.
                        cached = (head_device, world_device, directions)
                        self._head_cache[key] = cached
                    head_device, world_device, _source = cached
                    wp.launch(_head_to_world_kernel, dim=len(directions),
                              inputs=[head_device, wp.vec3(*forward), wp.vec3(*right), wp.vec3(*up)],
                              outputs=[world_device], device=self.device, stream=stream)
                    return origin, world_device
            except ImportError:
                pass
        try:
            rows = tuple(directions)
        except TypeError as exc:
            raise ValueError("directions must have shape (N, 3)") from exc
        if not rows:
            raise ValueError("directions must contain at least one direction")
        head_directions: list[tuple[float, float, float]] = []
        for index, row in enumerate(rows):
            head_directions.append(_unit(_vec3(row, f"directions[{index}]")))
        head_np = np.asarray(head_directions, dtype=np.float32)
        stream_key = self._stream_cache_key(stream)
        key = ("host", tuple(head_np.shape), head_np.tobytes(), stream_key)
        cached = self._head_cache.get(key)
        if cached is None:
            head_device = self._host_to_device(head_np, wp.vec3, stream=stream)
            world_device = wp.empty(len(head_np), dtype=wp.vec3, device=self.device)
            cached = (head_device, world_device, None)
            self._head_cache[key] = cached
        head_device, world_device, _source = cached
        wp.launch(_head_to_world_kernel, dim=len(head_np),
                  inputs=[head_device, wp.vec3(*forward), wp.vec3(*right), wp.vec3(*up)],
                  outputs=[world_device], device=self.device, stream=stream)
        return origin, world_device

    def _torch_stream(self, source: Any = None) -> Any:
        if wp is None:
            return None
        try:
            import torch
            if isinstance(source, torch.Tensor):
                if not source.is_cuda:
                    raise WarpBackendContractError(
                        "Warp CUDA direction tensors must be CUDA tensors; "
                        "pass a list/NumPy array for an explicit host boundary"
                    )
                self._validate_torch_device(source, torch)
                return wp.stream_from_torch(torch.cuda.current_stream(source.device))
            if torch.cuda.is_available():
                # Use the renderer's device, not a hard-coded cuda:0.  This
                # keeps host-direction uploads and rectangular camera rays on
                # the same stream as the selected CUDA renderer.
                renderer_device = torch.device(self.device_name)
                return wp.stream_from_torch(torch.cuda.current_stream(renderer_device))
        except ImportError:
            pass
        return None

    def _host_to_device(self, values: Any, dtype: Any, *, stream: Any = None) -> Any:
        values = np.asarray(values)
        if stream is None:
            return wp.array(values, dtype=dtype, device=self.device)
        host = wp.array(values, dtype=dtype, device="cpu")
        out = wp.empty(values.shape[0], dtype=dtype, device=self.device)
        wp.copy(out, host, stream=stream)
        return out

    def _run_rays(self, origins: Any, directions: Any, *, near_m: float, far_m: float,
                  pose: Pose, timestamp_s: float, stimulus: Any, stimuli: Any,
                  stream: Any = None, buffer_slot: int | None = None) -> tuple[Any, Any, Any, Any, Any, Any]:
        if wp is None or np is None:  # pragma: no cover
            raise WarpBackendUnavailableError("Warp CUDA backend is unavailable")
        if isinstance(directions, wp.array):
            direction_device = directions
            n = len(directions)
        else:
            direction_np = np.asarray(directions, dtype=np.float32).reshape((-1, 3))
            direction_device = self._host_to_device(direction_np, wp.vec3, stream=stream)
            n = int(direction_np.shape[0])
        origin_np = np.asarray(origins, dtype=np.float32).reshape((-1, 3))
        if origin_np.shape[0] == 0:
            raise ValueError("origins must contain at least one origin")
        if origin_np.shape[0] > 1 and not np.array_equal(origin_np, origin_np[:1]):
            raise ValueError("Warp renderer requires one shared eye origin")
        # Every public eye render uses one eye origin for all rays.  Keep this
        # as a scalar kernel argument so the frame does not allocate/upload an
        # N-element origins array.
        origin_device = wp.vec3(float(origin_np[0, 0]), float(origin_np[0, 1]), float(origin_np[0, 2]))
        rebuild_s, refit_s, prepare_s = self._prepare_dynamic(pose, timestamp_s, stimulus, stimuli, stream=stream)
        if buffer_slot is None:
            buffer_slot = self._next_buffer_slot.get(n, 0)
            self._next_buffer_slot[n] = (buffer_slot + 1) % 2
        key = (n, int(buffer_slot) % 2)
        buffers = self._output_buffers.get(key)
        if buffers is None:
            buffers = (
                wp.empty(n, dtype=wp.float32, device=self.device),
                wp.empty(n, dtype=wp.int32, device=self.device),
                wp.empty(n, dtype=wp.int32, device=self.device),
                wp.empty(n, dtype=wp.vec3, device=self.device),
                wp.empty(n, dtype=wp.vec3, device=self.device),
                wp.empty(n, dtype=wp.float32, device=self.device),
            )
            self._output_buffers[key] = buffers
        hit_t, hit_face, hit_mesh, hit_point, hit_normal, luminance = buffers
        ray_started = time.perf_counter()
        wp.launch(_raycast_kernel, dim=n,
                  inputs=[self._scene_mesh.id,
                          self._dynamic_mesh.id if self._dynamic_active and self._dynamic_mesh is not None else 0,
                          self._ground_mesh.id, self._static_kind,
                          self._stimulus_mesh_ids_device, self._stimulus_mesh_poses_device,
                          self._stimulus_mesh_face_offsets_device, len(self._stimulus_visual_mesh_order),
                          origin_device, direction_device,
                          float(near_m), float(far_m)],
                  outputs=[hit_t, hit_face, hit_mesh, hit_point, hit_normal], device=self.device, stream=stream)
        ray_s = 0.0
        shade_started = time.perf_counter()
        light = _unit(self.world.config.light.direction_m)
        wp.launch(_shade_kernel, dim=n,
                  inputs=[direction_device, hit_face, hit_mesh, hit_point, hit_normal,
                          self._static_kind, self._static_texture, self._static_albedo, self._static_phase,
                          self._static_entry_normal,
                          self._static_exit_normal,
                          self._dynamic_kind if self._dynamic_kind is not None else self._static_kind,
                          self._dynamic_texture if self._dynamic_texture is not None else self._static_texture,
                          self._dynamic_albedo if self._dynamic_albedo is not None else self._static_albedo,
                          self._dynamic_phase if self._dynamic_phase is not None else self._static_phase,
                          self._dynamic_entry_normal if self._dynamic_entry_normal is not None else self._static_entry_normal,
                          self._dynamic_exit_normal if self._dynamic_exit_normal is not None else self._static_exit_normal,
                          self._stimulus_mesh_albedo_device,
                          wp.vec3(*light), float(self.world.config.light.ambient),
                          float(self.world.config.light.intensity), float(self.world.config.light.sky_brightness),
                          float(self.world.config.texture.ground_cell_m), float(self.world.config.texture.wall_cell_m),
                          float(self.world.config.texture.contrast), wp.vec3(*self.world.target.position_m),
                          float(_hash_phase(self.world.seed, "ground"))],
                  outputs=[luminance], device=self.device, stream=stream)
        shade_s = 0.0
        self._last_luminance = luminance
        self._last_timings = WarpRenderTimings(prepare_s, rebuild_s, refit_s, ray_s, shade_s, 0.0)
        return luminance, hit_t, hit_face, hit_mesh, hit_point, hit_normal

    def render_eye_directions_tensor(self, pose: Pose, directions: Any, eye: str = "left", *,
                                     timestamp_s: float = 0.0, stimulus: Any = None,
                                     stimuli: Any = None, t_s: float | None = None,
                                     stream: Any = None, buffer_slot: int | None = None) -> Any:
        """Render exact head-frame directions to a CUDA-resident Torch tensor."""
        pose = _coerce_pose(pose)
        if t_s is not None:
            timestamp_s = t_s
        timestamp_s = _finite(timestamp_s, "timestamp_s")
        self.world._eye_sign(eye)  # validate aliases before GPU work
        stream = stream if stream is not None else self._torch_stream(directions)
        origin, world_directions = self._head_directions(pose, directions, eye, stream=stream)
        if wp is not None and isinstance(world_directions, wp.array):
            n = len(world_directions)
        else:
            n = int(np.asarray(world_directions).shape[0])
        luminance, _hit_t, _hit_face, _hit_mesh, _hit_point, _hit_normal = self._run_rays(
            origin, world_directions, near_m=self.world.camera.near_m, far_m=self.world.camera.far_m,
            pose=pose, timestamp_s=timestamp_s, stimulus=stimulus, stimuli=stimuli,
            stream=stream, buffer_slot=buffer_slot)
        try:
            tensor = wp.to_torch(luminance)
        except Exception as exc:
            raise WarpBackendUnavailableError(f"Warp-to-Torch conversion failed: {exc}") from exc
        tensor._warp_array = luminance
        return tensor

    def render_eye_directions(self, pose: Pose, directions: Any, eye: str = "left", *,
                              timestamp_s: float = 0.0, stimulus: Any = None,
                              stimuli: Any = None, t_s: float | None = None) -> list[float]:
        """List-compatible wrapper around :meth:`render_eye_directions_tensor`."""
        tensor = self.render_eye_directions_tensor(pose, directions, eye, timestamp_s=timestamp_s,
                                                   stimulus=stimulus, stimuli=stimuli, t_s=t_s)
        return tensor.detach().cpu().tolist()

    def render_eye_tensor(self, pose: Pose, eye: str = "left", resolution: Any = None, *,
                          width: int | None = None, height: int | None = None,
                          timestamp_s: float = 0.0, stimulus: Any = None,
                          stimuli: Any = None, t_s: float | None = None,
                          stream: Any = None, buffer_slot: int | None = None) -> Any:
        """Render a rectangular eye frame to a CUDA Torch tensor ``(H, W)``."""
        pose = _coerce_pose(pose)
        if stimulus is None and (isinstance(resolution, (PhysicalStimulus, str)) or
                                 (isinstance(resolution, Mapping) and any(key in resolution for key in ("start_position_m", "start", "velocity_mps", "speed_mps", "size_m")))):
            stimulus, resolution = resolution, None
        if t_s is not None:
            timestamp_s = t_s
        camera = self.world._resolve_camera(resolution, width=width, height=height)
        origin, directions = self._camera_rays(self.world, pose, eye, camera.width, camera.height, camera)
        stream = stream if stream is not None else self._torch_stream(None)
        luminance, _hit_t, _hit_face, _hit_mesh, _hit_point, _hit_normal = self._run_rays(
            origin, directions, near_m=camera.near_m, far_m=camera.far_m, pose=pose,
            timestamp_s=_finite(timestamp_s, "timestamp_s"), stimulus=stimulus, stimuli=stimuli,
            stream=stream, buffer_slot=buffer_slot)
        tensor = wp.to_torch(luminance).reshape((camera.height, camera.width))
        tensor._warp_array = luminance
        return tensor

    def render_eye(self, pose: Pose, eye: str = "left", resolution: Any = None, *,
                   width: int | None = None, height: int | None = None,
                   timestamp_s: float = 0.0, stimulus: Any = None,
                   stimuli: Any = None, t_s: float | None = None) -> GrayFrame:
        tensor = self.render_eye_tensor(pose, eye, resolution, width=width, height=height,
                                        timestamp_s=timestamp_s, stimulus=stimulus,
                                        stimuli=stimuli, t_s=t_s)
        values = tensor.detach().cpu().tolist()
        out = GrayFrame()
        out.extend([[float(value) for value in row] for row in values])
        return out

    def render_stereo_tensor(self, pose: Pose, resolution: Any = None, *, width: int | None = None,
                             height: int | None = None, timestamp_s: float = 0.0,
                             stimulus: Any = None, stimuli: Any = None,
                             t_s: float | None = None, stream: Any = None) -> tuple[Any, Any]:
        if t_s is not None:
            timestamp_s = t_s
        left = self.render_eye_tensor(pose, "left", resolution, width=width, height=height,
                                       timestamp_s=timestamp_s, stimulus=stimulus, stimuli=stimuli,
                                       stream=stream, buffer_slot=0)
        right = self.render_eye_tensor(pose, "right", resolution, width=width, height=height,
                                       timestamp_s=timestamp_s, stimulus=stimulus, stimuli=stimuli,
                                       stream=stream, buffer_slot=1)
        return left, right

    def render_stereo(self, pose: Pose, resolution: Any = None, *, width: int | None = None,
                      height: int | None = None, timestamp_s: float = 0.0,
                      stimulus: Any = None, stimuli: Any = None,
                      t_s: float | None = None, stream: Any = None) -> StereoFrames:
        left, right = self.render_stereo_tensor(pose, resolution, width=width, height=height,
                                                timestamp_s=timestamp_s, stimulus=stimulus,
                                                stimuli=stimuli, t_s=t_s, stream=stream)
        return StereoFrames(GrayFrame([[float(value) for value in row] for row in left.detach().cpu().tolist()]),
                            GrayFrame([[float(value) for value in row] for row in right.detach().cpu().tolist()]),
                            float(timestamp_s if t_s is None else t_s), self.world.camera.eye_separation_m)

    def trace_ray(self, origin_m: Sequence[float], direction: Sequence[float], *, near_m: float | None = None,
                  far_m: float | None = None, timestamp_s: float = 0.0, stimulus: Any = None,
                  stimuli: Any = None, reference_pose: Any = None) -> Hit | None:
        """Trace one world ray through the CUDA mesh and return a Hit record."""
        origin = _vec3(origin_m, "origin_m")
        direction_unit = _unit(_vec3(direction, "direction"))
        pose = _coerce_pose(reference_pose) if reference_pose is not None else Pose(origin)
        near = self.world.camera.near_m if near_m is None else max(1.0e-5, _finite(near_m, "near_m"))
        far = self.world.camera.far_m if far_m is None else max(near, _finite(far_m, "far_m"))
        result = self._run_rays([origin], [direction_unit], near_m=near, far_m=far,
                                pose=pose, timestamp_s=timestamp_s, stimulus=stimulus, stimuli=stimuli)
        _lum, hit_t, hit_face, hit_mesh, hit_point, hit_normal = result
        wp.synchronize_device(self.device)
        t = float(hit_t.numpy()[0])
        mesh_kind = int(hit_mesh.numpy()[0])
        if mesh_kind < 0:
            return None
        point_np = hit_point.numpy()[0]
        normal_np = hit_normal.numpy()[0]
        if mesh_kind == 2:
            return Hit("ground", t, tuple(float(v) for v in point_np), tuple(float(v) for v in normal_np), "ground", "ground")
        face = int(hit_face.numpy()[0])
        if mesh_kind == 0:
            materials = self._static_materials
        elif mesh_kind == 1:
            materials = self._dynamic_materials
        elif mesh_kind == 3:
            materials = self._stimulus_visual_materials
        else:
            materials = ()
        if face < 0 or face >= len(materials):
            return None
        material = materials[face]
        texture_names = {
            0: "checker", 1: "vertical_bars", 2: "horizontal_bars",
            3: "concentric_checker", 4: "palm", 5: "analytic",
        }
        hit_kind = ("stimulus_mesh" if mesh_kind == 3 else
                    "stimulus" if mesh_kind == 1 else
                    "scene_mesh" if material.kind == 5 else
                    "target" if material.kind == 1 else
                    "obstacle" if material.kind == 2 else "wall")
        return Hit(material.object_id, t, tuple(float(v) for v in point_np),
                   tuple(float(v) for v in normal_np),
                   "stimulus_mesh" if mesh_kind == 3 else material.texture_name or texture_names[material.texture],
                   hit_kind)

    def benchmark(self, *, iterations: int = 3, warmup: int = 1,
                  resolution: tuple[int, int] = (32, 24), directions: Any = None,
                  pose: Pose | None = None, timestamp_s: float = 0.73,
                  stimulus: Any = None, stimuli: Any = None) -> dict[str, Any]:
        """Measure cold construction, warm-up/JIT, and steady-state CUDA costs."""
        if iterations <= 0 or warmup < 0:
            raise ValueError("iterations must be positive and warmup must be non-negative")
        pose = Pose((0.0, 0.0, 0.42), yaw_rad=0.2, pitch_rad=0.1) if pose is None else _coerce_pose(pose)
        if directions is None:
            directions = [(1.0, 0.35 * math.sin(i * 0.017), 0.22 * math.cos(i * 0.013)) for i in range(721)]
        cold_started = time.perf_counter()
        cold_renderer = type(self)(self.world, device=self.device_name)
        cold_renderer.render_eye_directions_tensor(pose, directions, "left", timestamp_s=timestamp_s,
                                                    stimulus=stimulus, stimuli=stimuli)
        wp.synchronize_device(self.device)
        cold_s = time.perf_counter() - cold_started
        warm_samples: list[float] = []
        for _ in range(warmup):
            self.render_eye_directions_tensor(pose, directions, "left", timestamp_s=timestamp_s,
                                              stimulus=stimulus, stimuli=stimuli)
        for _ in range(iterations):
            started = time.perf_counter()
            self.render_eye_directions_tensor(pose, directions, "left", timestamp_s=timestamp_s,
                                              stimulus=stimulus, stimuli=stimuli)
            wp.synchronize_device(self.device)
            warm_samples.append(time.perf_counter() - started)
        steady = sorted(warm_samples)[len(warm_samples) // 2]
        return {
            "backend": self.backend_name,
            "device": self.device_name,
            "directions": len(directions),
            "cold_start_s": cold_s,
            "steady_state_median_s": steady,
            "steady_state_samples_s": warm_samples,
            "last_timings": self.last_timings.to_dict(),
            "mesh_stats": self.mesh_stats,
            "memory_budget": {
                "ram_limit_gb": 5.0,
                "vram_limit_gb": 15.5,
                "bounded_scene": True,
                "observed": False,
                "note": "allocation telemetry is not sampled by this benchmark",
            },
        }


__all__ = [
    "WarpBackendContractError",
    "WarpBackendUnavailableError",
    "WarpRenderTimings",
    "WarpWorldRenderer",
    "warp_cuda_available",
]
