#!/usr/bin/env python3
"""Prepare a shared, readback-checked FlyGym visual-input suite.

The 17 upstream neural NPZs are referenced in place and never rewritten.  The
script creates only the three 4 s native retinal stripe inputs, then writes a
v2-harness-compatible neural input manifest and a separate runner suite
manifest.  It does not start the neural worker, MaleCNS, FlyGym, or physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from standard_optomotor_stimulus import AngularOptomotorStimulus  # noqa: E402


EXPECTED_NEURAL_SCHEMA = "flygym-recorded-retina-ball-assay-input-manifest-1"
EXPECTED_UV_SHA256 = "e244a411df2f8fa5bcbc8e1fb14fe68b8d1836116cd35ac98082d294f4a1e5ef"
EXPECTED_LEFT_SHA256 = "2f847637ae64bea5a5454aa90e86e440a23d76d5c3b85a866936e502691654f1"
EXPECTED_RIGHT_SHA256 = "5037edbec7e4709ddc5f31643de643cdadef4492145756e72634d52727b7b4f6"
EXPECTED_MIRROR_SHA256 = "e5fa699ca3ff11b99488a1dd3c18da83f2d906504bc8fb327868d6c31cda446d"
EXPECTED_RAY_GRID_FILE_SHA256 = "62c574e5a627a519c3487d9e06abd4b620442da488b99d379b9feee883ce1f1c"
FRAME_COUNT = 800
FRAME_DT_S = 0.005
ONSET_S = 1.0
END_S = 4.0
GRAY_FRAMES = 200
SUITE_SCHEMA = "flygym-standardized-input-repair-suite-manifest-1"


class PreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"expected a JSON object in {path}")
    return value


def file_record(path: Path, *, verify_existing: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved) if verify_existing else None}


def check_record(record: dict[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    size = record.get("size_bytes")
    expected_hash = record.get("sha256")
    if not isinstance(size, int) or path.stat().st_size != size:
        raise PreparationError(f"source file size differs from manifest ({label}): {path}")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise PreparationError(f"source file SHA-256 differs from manifest ({label}): {path}")
    return path


def source_manifest_record(path: Path) -> dict[str, Any]:
    return file_record(path)


def _case_rows(manifest: dict[str, Any], assay: str) -> dict[str, dict[str, Any]]:
    cases = manifest.get("cases")
    if isinstance(cases, dict):
        return {str(case_id): {"case_id": str(case_id), "input_npz": record}
                for case_id, record in cases.items()}
    if isinstance(cases, list):
        return {str(row.get("case_id")): row for row in cases if isinstance(row, dict)}
    raise PreparationError(f"unsupported source case collection for {assay}")


def _input_record(row: dict[str, Any]) -> dict[str, Any]:
    record = row.get("input_npz")
    if isinstance(record, dict) and isinstance(record.get("path"), str):
        return record
    if isinstance(row.get("path"), str) and isinstance(row.get("sha256"), str):
        return row
    raise PreparationError(f"source case has no input_npz path/hash: {row.get('case_id')}")


def _source_metadata(row: dict[str, Any], archive_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("stimulus_metadata")
    if isinstance(metadata, dict):
        return metadata
    omit = {"input_npz", "case_id", "fixed_pose", "worker_condition", "left_columns_sha256",
            "right_columns_sha256", "column_uv_sha256"}
    if row and any(key in row for key in ("target_center_head_azimuth_deg", "stimulus_type", "stimulus")):
        return {key: value for key, value in row.items() if key not in omit}
    return archive_metadata.get("stimulus", {}) if isinstance(archive_metadata.get("stimulus"), dict) else {}


def _display_summary(archive_metadata: dict[str, Any], source_metadata: dict[str, Any]) -> dict[str, Any]:
    stimulus = archive_metadata.get("stimulus", {})
    if not isinstance(stimulus, dict):
        stimulus = {}
    source_stimulus = source_metadata.get("stimulus", {})
    if not isinstance(source_stimulus, dict):
        source_stimulus = {}
    kind = (source_stimulus.get("stimulus_type") or source_stimulus.get("stimulus_kind")
            or source_stimulus.get("field") or stimulus.get("stimulus_kind")
            or stimulus.get("pattern") or archive_metadata.get("geometry")
            or archive_metadata.get("stimulus_geometry") or "recorded normalized grayscale field")
    return {
        "display_model": archive_metadata.get("geometry") or archive_metadata.get("stimulus_geometry")
        or stimulus.get("display_geometry") or "precomputed native retinal luminance columns",
        "stimulus_kind": str(kind),
        "luminance_units": "normalized grayscale [0,1]",
        "eye_columns_per_eye": 721,
    }


def _support_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    stim_left = left[GRAY_FRAMES:]
    stim_right = right[GRAY_FRAMES:]
    return {
        "definition": "unique eye columns with any stimulus-window sample different from baseline 0.5",
        "left_changed_columns": int(np.count_nonzero(np.any(stim_left != np.float32(0.5), axis=0))),
        "right_changed_columns": int(np.count_nonzero(np.any(stim_right != np.float32(0.5), axis=0))),
        "stimulus_left_min_max": [float(np.min(stim_left)), float(np.max(stim_left))],
        "stimulus_right_min_max": [float(np.min(stim_right)), float(np.max(stim_right))],
        "gray_prefix_exact": bool(np.all(left[:GRAY_FRAMES] == np.float32(0.5))
                                   and np.all(right[:GRAY_FRAMES] == np.float32(0.5))),
    }


def _load_native_input(path: Path, case_id: str, expected_record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_path = check_record(expected_record, label=f"neural case {case_id}")
    with np.load(actual_path, allow_pickle=False) as archive:
        if set(archive.files) != {"left_columns", "right_columns", "time_s", "column_uv", "metadata_json"}:
            raise PreparationError(f"native NPZ keys differ from v2 contract: {case_id}: {archive.files}")
        left = np.asarray(archive["left_columns"])
        right = np.asarray(archive["right_columns"])
        times = np.asarray(archive["time_s"])
        uv = np.asarray(archive["column_uv"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise PreparationError(f"native NPZ metadata_json is not an object: {case_id}")
    if (left.shape != (FRAME_COUNT, 721) or right.shape != (FRAME_COUNT, 721)
            or left.dtype != np.float32 or right.dtype != np.float32
            or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
            or np.any(left < 0.0) or np.any(left > 1.0)
            or np.any(right < 0.0) or np.any(right > 1.0)):
        raise PreparationError(f"eye arrays must be finite float32 [800,721] in [0,1]: {case_id}")
    if (times.shape != (FRAME_COUNT,) or times.dtype != np.float64
            or not np.array_equal(times, np.arange(FRAME_COUNT, dtype=np.float64) * FRAME_DT_S)):
        raise PreparationError(f"native clock differs from 800 x 5 ms: {case_id}")
    if (uv.shape != (721, 2) or uv.dtype != np.int32 or sha256_array(uv) != EXPECTED_UV_SHA256):
        raise PreparationError(f"column_uv differs from frozen parity-v2 order: {case_id}")
    if metadata.get("case_id") != case_id:
        raise PreparationError(f"NPZ metadata case_id differs from preserved source case ID: {case_id}")
    if metadata.get("condition") != "static" or float(metadata.get("onset_s", -1.0)) != ONSET_S:
        raise PreparationError(f"NPZ metadata must use static routing and onset 1 s: {case_id}")
    if not (np.array_equal(left[:GRAY_FRAMES], np.full((GRAY_FRAMES, 721), 0.5, np.float32))
            and np.array_equal(right[:GRAY_FRAMES], np.full((GRAY_FRAMES, 721), 0.5, np.float32))):
        raise PreparationError(f"first second must be exact bilateral 0.5 gray: {case_id}")
    for eye, observed in (("left", left), ("right", right)):
        expected_hash = expected_record.get(f"{eye}_array_sha256")
        actual_hash = sha256_array(observed)
        if expected_hash is not None and expected_hash != actual_hash:
            raise PreparationError(f"source manifest array hash mismatch for {case_id}/{eye}")
    arrays = {
        "left_sha256": sha256_array(left), "right_sha256": sha256_array(right),
        "column_uv_sha256": sha256_array(uv),
        "left_shape": list(left.shape), "right_shape": list(right.shape),
        "time_s_shape": list(times.shape), "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype), "time_s_dtype": str(times.dtype),
    }
    return metadata, {"left": left, "right": right, "time_s": times, "column_uv": uv,
                      "arrays": arrays, "path": actual_path}


def _ray_arrays(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    geometry = config["retinal_geometry"]
    grid_path = Path(geometry["ray_grid_path"]).expanduser().resolve(strict=True)
    grid_file_hash = sha256_file(grid_path)
    if grid_file_hash != geometry["ray_grid_sha256"] or grid_file_hash != EXPECTED_RAY_GRID_FILE_SHA256:
        raise PreparationError(f"source ray-grid NPZ changed: {grid_path}")
    with np.load(grid_path, allow_pickle=False) as archive:
        required = {"left_directions", "right_directions", "column_uv", "mirror_index"}
        if not required <= set(archive.files):
            raise PreparationError(f"ray-grid archive is missing arrays: {sorted(required-set(archive.files))}")
        left = np.asarray(archive["left_directions"])
        right = np.asarray(archive["right_directions"])
        uv = np.asarray(archive["column_uv"])
        mirror = np.asarray(archive["mirror_index"])
    actual = {"left_directions_sha256": sha256_array(left),
              "right_directions_sha256": sha256_array(right),
              "column_uv_sha256": sha256_array(uv), "mirror_index_sha256": sha256_array(mirror)}
    expected = {"left_directions_sha256": EXPECTED_LEFT_SHA256,
                "right_directions_sha256": EXPECTED_RIGHT_SHA256,
                "column_uv_sha256": EXPECTED_UV_SHA256,
                "mirror_index_sha256": EXPECTED_MIRROR_SHA256}
    if actual != expected or any(geometry[key] != value for key, value in expected.items()):
        raise PreparationError(f"ray arrays or UV/parity-v2 identities differ: actual={actual}")
    if (left.shape != (721, 3) or right.shape != (721, 3) or uv.shape != (721, 2)
            or uv.dtype != np.int32 or mirror.shape != (721,)):
        raise PreparationError("ray grid shape/dtype differs from the native 721-column contract")
    return left, right, uv, {**actual, "ray_grid_path": str(grid_path),
                            "ray_grid_file_sha256": grid_file_hash,
                            "ray_count_per_eye": 721, "left_eye_parity": -1,
                            "right_eye_parity": 1,
                            "left_uv_mapping": geometry["left_uv_mapping"],
                            "right_uv_mapping": geometry["right_uv_mapping"]}


def _stripe_arrays(case: dict[str, Any], config: dict[str, Any], left_dirs: np.ndarray,
                   right_dirs: np.ndarray, uv: np.ndarray, ray_record: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    times = np.arange(FRAME_COUNT, dtype=np.float64) * FRAME_DT_S
    fixed_pose = config["retinal_geometry"]["fixed_pose"]
    stimulus = AngularOptomotorStimulus(
        contrast=float(case["contrast_c"]),
        head_direction_sign=int(case["head_direction_sign"]),
        wavelength_deg=float(case["wavelength_deg"]),
        temporal_frequency_hz=float(case["temporal_frequency_hz"]),
        mean_luminance=float(case["mean_luminance"]),
        azimuth_extent_deg=tuple(float(v) for v in case["field_azimuth_deg"]),
        elevation_extent_deg=tuple(float(v) for v in case["field_elevation_deg"]),
        onset_s=ONSET_S, duration_s=END_S - ONSET_S,
        phase_at_onset_rad=0.0, name=case["case_id"],
    )
    # The frozen directions are fitted FlyGym head rays. At the shared zero-yaw
    # pose, the established parity contract maps head +Y to world -Y.
    transform = np.asarray((1.0, -1.0, 1.0), dtype=np.float64)
    left_world = np.asarray(left_dirs, dtype=np.float64) * transform[None, :]
    right_world = np.asarray(right_dirs, dtype=np.float64) * transform[None, :]
    left = np.stack([stimulus.render_world_directions(left_world, float(t)) for t in times]).astype(np.float32)
    right = np.stack([stimulus.render_world_directions(right_world, float(t)) for t in times]).astype(np.float32)
    metadata = {
        "schema_version": "flygym-native-recorded-columns-1",
        "case_id": case["case_id"], "condition": "static",
        "duration_s": 4.0, "frame_count": FRAME_COUNT, "frame_dt_s": FRAME_DT_S,
        "onset_s": ONSET_S, "stimulus_end_s": END_S,
        "gray_interval_s_half_open": [0.0, 1.0],
        "stimulus_interval_s_half_open": [1.0, 4.0],
        "fixed_pose": fixed_pose,
        "direction_arrays": {"left_directions": EXPECTED_LEFT_SHA256,
                             "right_directions": EXPECTED_RIGHT_SHA256,
                             "column_uv": EXPECTED_UV_SHA256},
        "direction_array_order": "frozen FlyVis input_index order; L uses M(u,v)=(u+v,-v), R unchanged",
        "eye_parity": {"left": -1, "right": 1},
        "ray_grid_file_sha256": ray_record["ray_grid_file_sha256"],
        "normalized_luminance_range": [0.0, 1.0],
        "luminance_units": "normalized grayscale, not calibrated cd/m^2",
        "stimulus": {
            "kind": "vertical_sinusoidal_grating",
            "source_stimulus_spec_path": case["source_stimulus_spec"],
            "source_case_id": case["source_case_id"],
            "display_geometry": "fixed-arena infinite-distance angular field",
            "head_direction_sign": stimulus.head_direction_sign,
            "head_angular_speed_deg_s": stimulus.angular_speed_deg_s,
            "arena_angular_speed_deg_s": stimulus.arena_angular_speed_deg_s,
            "temporal_frequency_hz": stimulus.temporal_frequency_hz,
            "wavelength_deg": stimulus.wavelength_deg,
            "contrast_c": stimulus.contrast,
            "mean_luminance": stimulus.mean_luminance,
            "azimuth_extent_deg": list(stimulus.azimuth_extent_deg),
            "elevation_extent_deg": list(stimulus.elevation_extent_deg),
            "phase_at_onset_rad": stimulus.phase_at_onset_rad,
            "classification": "prescribed visual input; downstream response only",
            "interpretation_limits": [
                "This is a fixed-head prerecorded retinal input, not a closed-loop optomotor assay.",
                "The static and moving stripe cases share luminance range and nominal spatial wavelength; their temporal pattern differs.",
                "The pattern is not used to claim a biological turning response."
            ]
        }
    }
    if (left.shape != (FRAME_COUNT, 721) or right.shape != (FRAME_COUNT, 721)
            or left.dtype != np.float32 or right.dtype != np.float32
            or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
            or np.any(left < 0.0) or np.any(left > 1.0)
            or np.any(right < 0.0) or np.any(right > 1.0)
            or not np.all(left[:GRAY_FRAMES] == np.float32(0.5))
            or not np.all(right[:GRAY_FRAMES] == np.float32(0.5))):
        raise PreparationError(f"generated stripe input violates native format/luminance contract: {case['case_id']}")
    return {"left_columns": left, "right_columns": right, "time_s": times,
            "column_uv": uv}, metadata


def _write_or_reuse_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    expected_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"left_columns", "right_columns", "time_s", "column_uv", "metadata_json"}:
                raise PreparationError(f"refusing to replace existing non-native input: {path}")
            for key, value in arrays.items():
                if not np.array_equal(archive[key], value):
                    raise PreparationError(f"refusing to replace existing differing input: {path}")
            if str(archive["metadata_json"].item()) != expected_metadata:
                raise PreparationError(f"existing stripe metadata differs from requested provenance: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays, metadata_json=np.asarray(expected_metadata))
    os.replace(temporary, path)


def _collect_existing_neural(config: dict[str, Any], fixed_pose: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    source_manifests: dict[str, Any] = {}
    for source in config["source_assays"]:
        assay = str(source["assay"])
        manifest_path = Path(source["manifest_path"]).expanduser().resolve(strict=True)
        source_manifest = read_json(manifest_path)
        manifest_record = source_manifest_record(manifest_path)
        source_manifests[assay] = {**manifest_record, "schema_version": source_manifest.get("schema_version", source_manifest.get("schema"))}
        rows = _case_rows(source_manifest, assay)
        for case_id in source["source_case_ids"]:
            if case_id not in rows:
                raise PreparationError(f"{assay} source manifest has no required case {case_id}")
            row = rows[case_id]
            record = _input_record(row)
            native_metadata, native = _load_native_input(manifest_path, case_id, record)
            if native["path"].stat().st_size != int(record["size_bytes"]):
                raise PreparationError(f"source NPZ size changed: {case_id}")
            # Preserve the upstream path and its byte-level identity. No copy,
            # symlink, rewrite, or array normalization is performed here.
            input_record = {"path": str(native["path"]), "size_bytes": native["path"].stat().st_size,
                            "sha256": str(record["sha256"])}
            source_meta = _source_metadata(row, native_metadata)
            source_case_meta = {
                "case_manifest_path": str(manifest_path),
                "case_manifest_sha256": manifest_record["sha256"],
                "source_case_id": case_id,
                "source_metadata": source_meta,
            }
            support = _support_summary(native["left"], native["right"])
            arrays = native["arrays"]
            case = {
                "case_id": case_id,
                "cohort": "neural",
                "assay": assay,
                "assay_group": assay,
                "input_path": str(native["path"]),
                "input_sha256": input_record["sha256"],
                "input_npz": input_record,
                "source_bundle": config["source_bundle"],
                "source_case_id": case_id,
                "source_manifest": source_case_meta,
                "seed": int(config["seed"]),
                "duration_s": 4.0,
                "frame_count": FRAME_COUNT,
                "sample_dt_s": FRAME_DT_S,
                "gray_window_s": [0.0, 1.0],
                "stimulus_window_s": [1.0, 4.0],
                "routing_label": "static",
                "worker_condition": "static",
                "fixed_pose": fixed_pose,
                "native_schema": "flygym-native-recorded-columns-1",
                "stimulus_metadata": {
                    "assay": assay,
                    "source_case_id": case_id,
                    "onset_s": ONSET_S,
                    "display": _display_summary(native_metadata, source_meta),
                    "support": support,
                    "classification": source["classification"],
                    "comparison_note": source.get("comparison_note"),
                },
                "display": _display_summary(native_metadata, source_meta),
                "support": support,
                "input_arrays": arrays,
                "expected_left_array_sha256": arrays["left_sha256"],
                "expected_right_array_sha256": arrays["right_sha256"],
                "column_uv_sha256": EXPECTED_UV_SHA256,
                "source_ray_hashes": {"left_directions": EXPECTED_LEFT_SHA256,
                                      "right_directions": EXPECTED_RIGHT_SHA256,
                                      "column_uv": EXPECTED_UV_SHA256},
                "condition": "source-preserved prerecorded visual input",
                "notes": source["classification"],
            }
            cases.append(case)
    return cases, source_manifests


def _prepare_stripe_cases(config: dict[str, Any], left_dirs: np.ndarray, right_dirs: np.ndarray,
                          uv: np.ndarray, ray_record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle_root = Path(config["bundle_root"]).expanduser().resolve()
    input_dir = bundle_root / config["output_files"]["neural_inputs_dir"]
    cases: list[dict[str, Any]] = []
    source_records: dict[str, Any] = {}
    contrast_source_root = Path("/data/artifacts/by-project/malecns-gpu/engineering/flygym-fixed-head-contrast-ch-diagnostic-20261007-v2-r2")
    for item in config["new_stripes_cases"]:
        source_spec_path = Path(item["source_stimulus_spec"]).expanduser().resolve(strict=True)
        source_spec = read_json(source_spec_path)
        source_stim = source_spec.get("stimulus", {})
        expected_spec = {
            "head_direction_sign": int(item["head_direction_sign"]),
            "temporal_frequency_hz": float(item["temporal_frequency_hz"]),
            "wavelength_deg": float(item["wavelength_deg"]),
            "contrast": float(item["contrast_c"]),
            "mean_luminance": float(item["mean_luminance"]),
            "azimuth_extent_deg": [float(v) for v in item["field_azimuth_deg"]],
            "elevation_extent_deg": [float(v) for v in item["field_elevation_deg"]],
        }
        for key, expected in expected_spec.items():
            value = source_stim.get(key)
            if key.endswith("_extent_deg"):
                actual = [float(v) for v in value] if isinstance(value, list) else value
                if actual != expected:
                    raise PreparationError(f"source stripe stimulus spec differs: {source_spec_path}/{key}")
            elif isinstance(expected, float):
                if not isinstance(value, (int, float)) or not math.isclose(float(value), expected, abs_tol=1e-12):
                    raise PreparationError(f"source stripe stimulus spec differs: {source_spec_path}/{key}")
            elif value != expected:
                raise PreparationError(f"source stripe stimulus spec differs: {source_spec_path}/{key}")
        spec_record = file_record(source_spec_path)
        source_records[item["source_case_id"]] = spec_record
        arrays, metadata = _stripe_arrays(item, config, left_dirs, right_dirs, uv, ray_record)
        metadata["stimulus"]["source_stimulus_spec_sha256"] = spec_record["sha256"]
        path = input_dir / f"{item['case_id']}.npz"
        _write_or_reuse_npz(path, arrays, metadata)
        rec = file_record(path)
        native_meta, native = _load_native_input(path, item["case_id"], rec)
        support = _support_summary(native["left"], native["right"])
        array_info = native["arrays"]
        display = _display_summary(native_meta, metadata["stimulus"])
        case = {
            "case_id": item["case_id"], "cohort": "neural", "assay": "striped-optomotor",
            "assay_group": "striped-optomotor", "input_path": str(path.resolve()),
            "input_sha256": rec["sha256"], "input_npz": rec,
            "source_bundle": str(contrast_source_root), "source_case_id": item["source_case_id"],
            "source_manifest": {"stimulus_spec_path": str(source_spec_path),
                                "stimulus_spec_sha256": spec_record["sha256"],
                                "source_timing_s": {"onset_s": source_stim.get("onset_s"),
                                                    "duration_s": source_stim.get("duration_s")},
                                "standardized_timing_change": "generated 4 s input uses gray [0,1) s and stimulus [1,4) s"},
            "seed": int(config["seed"]), "duration_s": 4.0, "frame_count": FRAME_COUNT,
            "sample_dt_s": FRAME_DT_S, "gray_window_s": [0.0, 1.0],
            "stimulus_window_s": [1.0, 4.0], "routing_label": "static",
            "worker_condition": "static", "fixed_pose": config["retinal_geometry"]["fixed_pose"],
            "native_schema": "flygym-native-recorded-columns-1", "display": display,
            "support": support,
            "stimulus_metadata": {
                "assay": "striped-optomotor", "source_case_id": item["source_case_id"],
                "onset_s": ONSET_S, "display": display, "support": support,
                "head_direction_sign": int(item["head_direction_sign"]),
                "head_angular_speed_deg_s": float(item["head_angular_speed_deg_s"]),
                "temporal_frequency_hz": float(item["temporal_frequency_hz"]),
                "wavelength_deg": float(item["wavelength_deg"]), "contrast_c": float(item["contrast_c"]),
                "classification": "prescribed visual input; downstream response only",
            },
            "input_arrays": array_info,
            "expected_left_array_sha256": array_info["left_sha256"],
            "expected_right_array_sha256": array_info["right_sha256"],
            "column_uv_sha256": EXPECTED_UV_SHA256,
            "source_ray_hashes": {"left_directions": EXPECTED_LEFT_SHA256,
                                  "right_directions": EXPECTED_RIGHT_SHA256,
                                  "column_uv": EXPECTED_UV_SHA256},
            "condition": "source stripe stimulus parameters with standardized 4 s clock",
            "notes": "New CPU-generated retinal input only; no neural forward or body/physics execution.",
        }
        cases.append(case)
    return cases, source_records


def _mechanics_cases(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    settings = config["mechanics_cases"]
    protocol_path = Path(settings["protocol_path"]).expanduser()
    freeze_path = Path(settings["source_freeze_path"]).expanduser()
    protocol_record = (file_record(protocol_path) if protocol_path.is_file()
                       else {"path": str(protocol_path.resolve()), "status": "pending"})
    freeze_record = (file_record(freeze_path) if freeze_path.is_file()
                     else {"path": str(freeze_path.resolve()), "status": "pending"})
    rows: list[dict[str, Any]] = []
    for item in settings["cases"]:
        output_dir = Path(item["run_output_dir"]).expanduser()
        case_manifest_path = output_dir / "manifest.json"
        output_manifest: dict[str, Any] | None = None
        case_manifest_record: dict[str, Any]
        if case_manifest_path.is_file():
            output_manifest = read_json(case_manifest_path)
            case_manifest_record = file_record(case_manifest_path)
            run_status = "manifest_present_readback_pending"
        else:
            case_manifest_record = {"path": str(case_manifest_path.resolve()), "status": "pending"}
            run_status = "awaiting_mechanics_worker"
        source_artifacts = {}
        summary_sha256 = None
        run_record_sha256 = None
        if isinstance(output_manifest, dict):
            raw_artifacts = output_manifest.get("outputs", output_manifest.get("artifacts", {}))
            if isinstance(raw_artifacts, dict):
                source_artifacts = raw_artifacts
            summary = output_manifest.get("summary", {})
            if isinstance(summary, dict):
                summary_sha256 = summary.get("sha256")
            run_record = output_manifest.get("run_record", {})
            if isinstance(run_record, dict):
                run_record_sha256 = run_record.get("sha256")
        row = {
            "case_id": item["case_id"], "cohort": "mechanics",
            "assay": settings["assay"], "assay_group": "motor-mechanics",
            "source_bundle": str(Path(config["bundle_root"]).expanduser().resolve()),
            "source_case_id": item["case_id"], "source_run_dir": str(output_dir.resolve()),
            "run_output_dir": str(output_dir.resolve()),
            "case_manifest": case_manifest_record,
            "source_artifacts": source_artifacts,
            "run_record_sha256": run_record_sha256,
            "run_record_status": "recorded_in_case_manifest" if run_record_sha256 else "not_yet_recorded",
            "summary_sha256": summary_sha256,
            "seed": int(settings["seed"]), "duration_s": float(settings["duration_s"]),
            "frame_count": None, "sample_dt_s": None,
            "analysis_window_s": settings["analysis_window_s"],
            "execution_interval_s": settings["execution_interval_s"],
            "startup_window_s": settings["startup_window_s"],
            "gray_or_visual_baseline_window_s": settings["gray_or_visual_baseline_window_s"],
            "constant_P": float(settings["constant_P"]),
            "d": float(item["d"]), "no_neural": True, "neural_input": False,
            "no_visual_input": bool(settings["no_visual_input"]),
            "no_external_force": bool(settings["no_external_force"]),
            "shared_initial_state_and_phase": bool(settings["shared_initial_state_and_phase"]),
            "routing_label": None,
            "classification": settings["classification"],
            "execution_status": run_status,
            "notes": "Mechanics-only cohort; no visual frames or neural input are associated with this case.",
        }
        rows.append(row)
    historical = settings["historical_reference"]
    historical_path = Path(historical["manifest_path"]).expanduser().resolve(strict=True)
    historical_record = file_record(historical_path)
    historical_cases = read_json(historical_path)
    return rows, {"protocol": protocol_record, "source_freeze": freeze_record}, {
        **historical, "manifest_record": historical_record,
        "case_ids": ["static-d0", "d-plus-0p2", "d-minus-0p2"],
        "raw_data_copied": False,
        "source_schema": historical_cases.get("schema"),
    }


def prepare(config_path: Path) -> tuple[Path, Path, Path]:
    config_path = config_path.expanduser().resolve(strict=True)
    config = read_json(config_path)
    if config.get("schema_version") != "flygym-standardized-suite-preparation-config-1":
        raise PreparationError("unsupported config schema")
    clock = config["standard_neural_clock"]
    if (int(clock["frame_count"]) != FRAME_COUNT or float(clock["sample_dt_s"]) != FRAME_DT_S
            or float(clock["duration_s"]) != END_S or clock["gray_window_s"] != [0.0, 1.0]
            or clock["stimulus_window_s"] != [1.0, 4.0] or clock["routing_label"] != "static"):
        raise PreparationError("config clock differs from the standardized 4 s neural contract")
    if int(config["seed"]) != 1729:
        raise PreparationError("the standardized main comparison must use fixed seed 1729")
    bundle_root = Path(config["bundle_root"]).expanduser().resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    input_dir = bundle_root / config["output_files"]["neural_inputs_dir"]
    input_dir.mkdir(parents=True, exist_ok=True)

    fixed_pose = config["retinal_geometry"]["fixed_pose"]
    left_dirs, right_dirs, uv, ray_record = _ray_arrays(config)
    original_cases, source_manifests = _collect_existing_neural(config, fixed_pose)
    stripe_cases, stripe_specs = _prepare_stripe_cases(config, left_dirs, right_dirs, uv, ray_record)
    neural_cases = original_cases + stripe_cases
    if len(original_cases) != 17 or len(stripe_cases) != 3 or len(neural_cases) != 20:
        raise PreparationError(f"expected 17 referenced + 3 generated neural cases; got {len(original_cases)} + {len(stripe_cases)}")
    ids = [item["case_id"] for item in neural_cases]
    if len(ids) != len(set(ids)):
        raise PreparationError("neural case IDs must be globally unique")

    mechanics, mechanics_protocol_sources, historical_mechanics_reference = _mechanics_cases(config)
    if len(mechanics) != 3:
        raise PreparationError(f"expected three separate mechanics cases, got {len(mechanics)}")
    ray_file = file_record(Path(ray_record["ray_grid_path"]))
    freeze_path = Path(config["source_bundle"]) / "shared-harness" / "freeze.json"
    core_path = Path(config["source_bundle"]) / "core-freeze.json"
    upstream_harness_freeze = file_record(freeze_path)
    upstream_core_freeze = file_record(core_path)
    generation_record = {
        "script": file_record(Path(__file__).resolve()),
        "config": file_record(config_path),
        "operation": "CPU-only native retinal array preparation and readback; no neural forward, body, or physics",
        "numpy_version": np.__version__,
    }

    input_manifest = {
        "schema_version": EXPECTED_NEURAL_SCHEMA,
        "assay_id": config["suite_id"],
        "suite_id": config["suite_id"],
        "status": "prepared_only_input_preflight",
        "evidence_scope": "input references and CPU-generated native retinal arrays; no neural, body, or physics execution",
        "seed": int(config["seed"]),
        "clock": {"duration_s": 4.0, "frame_count": FRAME_COUNT, "frame_dt_s": FRAME_DT_S,
                  "gray_interval_s_half_open": [0.0, 1.0], "stimulus_interval_s_half_open": [1.0, 4.0],
                  "routing_label": "static"},
        "retinal_sampling": ray_record,
        "fixed_pose": fixed_pose,
        "source_bundle": config["source_bundle"],
        "source_integrity": {"old_shared_harness_freeze": upstream_harness_freeze,
                              "old_core_freeze": upstream_core_freeze,
                              "old_assay_manifests": source_manifests,
                              "retinal_ray_grid": ray_file,
                              "mechanics_protocol": mechanics_protocol_sources,
                              "generation_sources": generation_record},
        "case_count": {"original_neural_references": len(original_cases),
                       "new_stripes_inputs": len(stripe_cases), "neural_total": len(neural_cases),
                       "mechanics_separate": len(mechanics)},
        "cases": neural_cases,
        "mechanics_cases": mechanics,
        "historical_mechanics_reference": historical_mechanics_reference,
        "comparison_groups": config["comparison_groups"],
        "future_protocol_extensions": config["future_protocol_extensions"],
        "interpretation_limits": [
            "The neural cases are standardized recorded inputs for model-response comparisons; they are not biological behavior validation.",
            "Looming inputs are response-only; no freeze or escape behavior is asserted by input preparation.",
            "Binocular rotation and translation paths are described by their source trajectories and are not falsely labeled as matched flow.",
            "Target center bearings and eye-ray support differences remain explicit and are not treated as matched visual coverage.",
            "The three tethered-ball cases are mechanics-only constant-P records and are excluded from the neural frame input cases.",
        ],
    }
    suite_manifest = {
        "schema_version": SUITE_SCHEMA,
        "suite_id": config["suite_id"], "seed": int(config["seed"]),
        "status": "prepared_only_input_preflight",
        "input_manifest_path": str((bundle_root / config["output_files"]["input_manifest"]).resolve()),
        "case_count": {"neural": len(neural_cases), "mechanics": len(mechanics), "total": len(neural_cases) + len(mechanics)},
        "clock": input_manifest["clock"],
        "case_groups": {item["case_id"]: item["assay_group"] for item in neural_cases},
        "comparison_groups": config["comparison_groups"],
        "cases": neural_cases,
        "mechanics_cases": mechanics,
        "historical_mechanics_reference": historical_mechanics_reference,
        "classification": "model response inputs plus a separate mechanics-only cohort",
        "future_protocol_extensions": config["future_protocol_extensions"],
        "interpretation_limits": input_manifest["interpretation_limits"],
    }
    preflight_cases = []
    for item in neural_cases:
        rec = item["input_npz"]
        md, native = _load_native_input(Path(rec["path"]), item["case_id"], rec)
        preflight_cases.append({
            "case_id": item["case_id"], "cohort": item["cohort"], "assay": item["assay"],
            "path": rec["path"], "size_bytes": rec["size_bytes"], "sha256": rec["sha256"],
            "arrays": native["arrays"], "support": _support_summary(native["left"], native["right"]),
            "metadata_case_id": md["case_id"], "metadata_condition": md["condition"],
            "metadata_onset_s": md["onset_s"],
        })
    preflight = {
        "schema_version": "flygym-standardized-input-preflight-1",
        "suite_id": config["suite_id"], "status": "pass_readback_only",
        "checks": {
            "original_inputs_referenced_without_copy": len(original_cases) == 17,
            "native_721_column_shapes_clock_and_luminance_valid": True,
            "gray_0_1_and_stimulus_1_4_clock": True,
            "column_uv_and_eye_ray_parity_hashes_match": True,
            "case_ids_unique_and_metadata_case_ids_match": True,
            "mechanics_cases_separate_from_neural_inputs": len(mechanics) == 3,
        },
        "ray_grid": ray_record,
        "source_manifests": source_manifests,
        "generated_stripe_specs": stripe_specs,
        "generation_sources": generation_record,
        "neural_cases": preflight_cases,
        "mechanics_case_count": len(mechanics),
        "historical_mechanics_reference": historical_mechanics_reference,
        "no_neural_forward": True, "no_simulation": True, "gpu_used": False,
    }

    input_manifest_path = bundle_root / config["output_files"]["input_manifest"]
    suite_manifest_path = bundle_root / config["output_files"]["suite_manifest"]
    preflight_path = bundle_root / config["output_files"]["preflight_readback"]
    write_json(input_manifest_path, input_manifest)
    write_json(suite_manifest_path, suite_manifest)
    write_json(preflight_path, preflight)
    return input_manifest_path, suite_manifest_path, preflight_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config/flygym_standardized_suite_v1.json")
    args = parser.parse_args()
    try:
        input_manifest, suite_manifest, preflight = prepare(args.config)
    except (PreparationError, OSError, ValueError, KeyError) as exc:
        print(f"preparation_error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "prepared_only", "input_manifest": str(input_manifest),
                      "suite_manifest": str(suite_manifest), "preflight": str(preflight)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
