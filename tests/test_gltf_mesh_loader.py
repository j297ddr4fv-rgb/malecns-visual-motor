from __future__ import annotations

import base64
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.gltf_mesh_loader import load_scene_assets


def _write_image(path: Path) -> bytes:
    image = Image.new("RGB", (2, 2))
    image.putdata([
        (0, 0, 0), (255, 0, 0),
        (0, 255, 0), (0, 0, 255),
    ])
    image.save(path, format="PNG")
    return path.read_bytes()


def _bundle(root: Path, *, glb: bool = False, with_uv: bool = True,
            texture_mode: str | None = "triangle_centroid",
            transform: bool = False, missing_image: bool = False,
            manifest_color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> tuple[Path, bytes | None]:
    position_bytes = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    uv_bytes = struct.pack("<6f", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    geometry = position_bytes + uv_bytes
    image_path = root / "paint.png"
    image_bytes = None if missing_image else _write_image(image_path)

    accessors = [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}]
    buffer_views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
        {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(uv_bytes)},
    ]
    attributes = {"POSITION": 0}
    if with_uv:
        attributes["TEXCOORD_0"] = 1
        accessors.append({"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"})

    if glb:
        assert image_bytes is not None
        image_offset = len(geometry)
        payload = geometry + image_bytes
        buffer_views.append({"buffer": 0, "byteOffset": image_offset, "byteLength": len(image_bytes)})
        image = {"bufferView": 2, "mimeType": "image/png"}
        gltf_uri = "asset.glb"
        path = root / gltf_uri
        document = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(payload)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
            "images": [image],
            "textures": [{"source": 0}],
            "materials": [{
                "name": "paint",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.5, 0.25, 1.0, 1.0],
                    "baseColorTexture": {"index": 0},
                },
            }],
            "meshes": [{"primitives": [{"attributes": attributes, "material": 0}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        }
        json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
        json_bytes += b" " * ((-len(json_bytes)) % 4)
        bin_chunk = payload + b"\0" * ((-len(payload)) % 4)
        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_chunk)
        path.write_bytes(
            struct.pack("<4sII", b"glTF", 2, total_length)
            + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
            + struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
        )
    else:
        uri_bytes = base64.b64encode(geometry).decode("ascii")
        image = {"uri": "missing.png" if missing_image else image_path.name}
        texture_info = {"index": 0}
        if transform:
            texture_info["extensions"] = {"KHR_texture_transform": {"offset": [0, 0]}}
        document = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(geometry), "uri": f"data:application/octet-stream;base64,{uri_bytes}"}],
            "bufferViews": buffer_views,
            "accessors": accessors,
            "images": [image],
            "textures": [{"source": 0}],
            "materials": [{
                "name": "paint",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.5, 0.25, 1.0, 1.0],
                    "baseColorTexture": texture_info,
                },
            }],
            "meshes": [{"primitives": [{"attributes": attributes, "material": 0}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        }
        path = root / "asset.gltf"
        path.write_text(json.dumps(document), encoding="utf-8")

    manifest_material: dict[str, object] = {"base_color_rgba": list(manifest_color)}
    if texture_mode is not None:
        manifest_material["sensor_texture_mode"] = texture_mode
    manifest = {
        "schema_version": 1,
        "scene_id": "texture-test",
        "units": "m",
        "coordinate_system": {"handedness": "right", "up_axis": "+Z", "forward_axis": "+X"},
        "materials": {"paint": manifest_material},
        "assets": [{
            "object_id": "painted-triangle",
            "path": path.name,
            "role": "visual",
            "transform": {
                "translation_m": [0, 0, 0],
                "rotation_xyzw": [0, 0, 0, 1],
                "scale": 1,
            },
            "provenance": {"source": "unit-test", "license": "CC0", "exporter": "test"},
            "material_refs": {"paint": "paint"},
        }],
    }
    manifest_path = root / "scene.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, image_bytes


class GltfTriangleAlbedoTests(unittest.TestCase):
    def test_opt_in_external_texture_uses_centroid_bilinear_rgb_and_manifest_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, image_bytes = _bundle(
                Path(temporary), manifest_color=(0.5, 1.0, 0.5, 1.0)
            )
            obj = load_scene_assets(manifest, verify_checksums=False).objects[0]
            self.assertIsNotNone(image_bytes)
            assert image_bytes is not None
            expected = np.asarray([[0.0625, 0.0625, 0.125]], dtype=np.float64)
            np.testing.assert_allclose(obj.triangle_albedos, expected, rtol=0.0, atol=1e-15)
            self.assertEqual(obj.triangle_albedos.shape, (len(obj.triangles), 3))
            self.assertEqual(obj.triangle_albedos.dtype, np.dtype("float64"))
            self.assertFalse(obj.triangle_albedos.flags.writeable)
            metadata = obj.texture_metadata["paint"]
            self.assertEqual(metadata["mode"], "triangle_centroid")
            self.assertEqual(metadata["source_image_sha256"], hashlib.sha256(image_bytes).hexdigest())
            self.assertEqual(metadata["source_image_size_px"], [2, 2])
            self.assertEqual(metadata["source_image_bytes"], len(image_bytes))
            np.testing.assert_allclose(metadata["mean_rgb_face_weighted"], expected[0])
            np.testing.assert_allclose(metadata["mean_rgb_area_weighted"], expected[0])

    def test_embedded_glb_image_uses_gltf_factor_and_reports_buffer_view_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = _bundle(Path(temporary), glb=True)
            obj = load_scene_assets(manifest, verify_checksums=False).objects[0]
            np.testing.assert_allclose(obj.triangle_albedos, [[0.125, 0.0625, 0.25]], rtol=0.0, atol=1e-15)
            metadata = obj.texture_metadata["paint"]
            self.assertTrue(str(metadata["source_image"]).startswith("bufferView:"))
            self.assertEqual(metadata["source_image_size_px"], [2, 2])

    def test_legacy_flat_material_keeps_manifest_mean_without_decoding_texture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = _bundle(
                Path(temporary), texture_mode=None, with_uv=False, missing_image=True,
                manifest_color=(0.2, 0.4, 0.6, 1.0),
            )
            obj = load_scene_assets(manifest, verify_checksums=False).objects[0]
            np.testing.assert_array_equal(obj.triangle_albedos, [[0.2, 0.4, 0.6]])
            self.assertEqual(dict(obj.texture_metadata), {})

    def test_opt_in_rejects_missing_uv_and_texture_transform(self) -> None:
        for options, message in (
            ({"with_uv": False}, "no TEXCOORD_0"),
            ({"transform": True}, "KHR_texture_transform"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                manifest, _ = _bundle(Path(temporary), **options)
                with self.assertRaisesRegex(ValueError, message):
                    load_scene_assets(manifest, verify_checksums=False)

    def test_opt_in_rejects_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = _bundle(Path(temporary), missing_image=True)
            with self.assertRaisesRegex(FileNotFoundError, "paint.png|missing.png"):
                load_scene_assets(manifest, verify_checksums=False)


if __name__ == "__main__":
    unittest.main()
