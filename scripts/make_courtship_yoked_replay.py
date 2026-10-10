#!/usr/bin/env python3
"""Freeze a strict moving-female trial's binocular retinal movie for yoking.

The replay is open-loop: its 200 Hz exact eye columns come from one recorded
closed-loop trial and are played to another simulated male regardless of the
replay male's pose. Only compact, lossless strict telemetry is accepted.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from courtship_movie import CourtshipMovie


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_vector(record: dict, name: str) -> np.ndarray:
    if (record.get("encoding") != "zlib+base64" or not record.get("lossless")
            or record.get("dtype") != "float32" or record.get("shape") != [721]):
        raise ValueError(f"{name}: compact record is not lossless exact float32 columns")
    encoded = record.get("data")
    if not isinstance(encoded, str):
        raise ValueError(f"{name}: no compact data payload")
    compressed = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(compressed).hexdigest() != record.get("compressed_sha256"):
        raise ValueError(f"{name}: compressed hash mismatch")
    raw = zlib.decompress(compressed)
    if len(raw) != 721 * 4 or hashlib.sha256(raw).hexdigest() != record.get("sha256"):
        raise ValueError(f"{name}: column payload hash mismatch")
    values = np.frombuffer(raw, dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name}: invalid luminance")
    return values.copy()


def create_replay(trial: Path, output: Path) -> dict:
    manifest_path = trial / "manifest.json"
    telemetry_path = trial / "telemetry.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_classification") != "courtship_closed_loop_visual_assay":
        raise ValueError("source trial is not a closed-loop courtship visual assay")
    assay = manifest.get("courtship_visual", {})
    if assay.get("target_condition") != "moving":
        raise ValueError("source trial must be the moving-female condition")
    if manifest.get("telemetry_contract", {}).get("mode") != "compact":
        raise ValueError("source trial requires compact lossless telemetry")
    directions = manifest.get("components", {}).get("vision", {}).get("exact_direction_mapping", {}).get("sha256", {})
    if not all(isinstance(directions.get(side), str) and len(directions[side]) == 64
               for side in ("left", "right")):
        raise ValueError("source trial has no exact fitted eye-direction hashes")
    columns = []
    with telemetry_path.open("r", encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("frame_index") != expected_index:
                raise ValueError("source frame sequence has a gap or duplicate")
            if abs(float(row.get("frame_time_s", -1.0)) - expected_index * 0.005) > 1.0e-7:
                raise ValueError("source frames are not on the 200 Hz grid")
            visual = row.get("visual_input", {})
            if visual.get("sampling", {}).get("mode") != "exact_world_eye_directions":
                raise ValueError("source trial did not use exact world eye directions")
            left = decode_vector(visual.get("left_columns", {}), "left")
            right = decode_vector(visual.get("right_columns", {}), "right")
            columns.append(np.stack((left, right)))
    if not columns:
        raise ValueError("source trial has no visual frames")
    manifest_record = {
        "eye_mode": "binocular_world_render",
        "stereo_layout": "independent_exact_eye_columns",
        "projection": "direct_exact_columns",
        "out_of_fov_policy": "none",
        "out_of_fov_sample_count": {"left": 0, "right": 0},
        "source_path": str(trial.resolve()),
        "source_sha256": sha256_file(telemetry_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_classification": manifest["run_classification"],
        "source_target_condition": "moving",
        "source_target_trajectory_sha256": assay.get("trajectory_sha256"),
        "source_asset_binding": assay.get("asset_binding"),
        "source_frame_count": len(columns),
        "source_frame_shape": [2, 721],
        "decoder": "lossless compact telemetry zlib+base64 float32",
        "temporal_sampling": "recorded 200 Hz FlyVis grid; exact sample order",
        "left_directions_sha256": directions["left"],
        "right_directions_sha256": directions["right"],
        "source_pose_policy": "recorded moving-trial male pose; replay ignores current male pose",
        "right_eye_sampling": "independent rendered right-eye columns",
        "evidence_scope": "open-loop retinal yoke; not a second closed-loop female scene",
    }
    movie = CourtshipMovie(np.stack(columns), manifest_record)
    movie.save(output)
    return {
        "artifact_path": str(output.resolve()),
        "artifact_sha256": sha256_file(output),
        "sample_count": movie.n_frames,
        "source_trial": str(trial.resolve()),
        "source_telemetry_sha256": manifest_record["source_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_replay(args.trial, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
