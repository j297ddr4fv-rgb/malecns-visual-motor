#!/usr/bin/env python3
"""Summarize the short, manually annotated NMFv2 real-fly walking episode."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
LEG_NAMES = {
    "LF": "left front", "LM": "left middle", "LH": "left hind",
    "RF": "right front", "RM": "right middle", "RH": "right hind",
}
AXES = ("x", "y", "z")


def _xyz(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in AXES])


def analyze(csv_path: Path, output_dir: Path, fps: float = 120.0) -> dict:
    raw = csv_path.read_bytes()
    expected_csv_md5 = "ddf066d0a20ab9aac275bf3304baade0"
    video_path = csv_path.with_name("recorded_steps.mp4")
    video_raw = video_path.read_bytes() if video_path.exists() else None
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {f"{name}_{axis}" for name in ("He", "Th", "Abd", *(f"{leg}-Claw" for leg in LEGS)) for axis in AXES}
    required |= {"frame_idx", "video_id"}
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    if not rows or missing:
        raise ValueError(f"CSV is empty or lacks documented keypoints/metadata: {missing}")
    rows.sort(key=lambda row: int(row["frame_idx"]))
    frames = np.asarray([int(row["frame_idx"]) for row in rows], dtype=int)
    video_ids = sorted({row["video_id"] for row in rows})
    if len(video_ids) != 1 or len(set(frames)) != len(frames):
        raise ValueError("This summary expects one video_id and one row per frame.")

    thorax = np.stack([_xyz(row, "Th") for row in rows])
    head = np.stack([_xyz(row, "He") for row in rows])
    abdomen = np.stack([_xyz(row, "Abd") for row in rows])
    body_length = np.linalg.norm(head - thorax, axis=1) + np.linalg.norm(thorax - abdomen, axis=1)
    body_length_ref = float(np.median(body_length))
    if not np.isfinite(body_length_ref) or body_length_ref <= 0:
        raise ValueError("Could not derive a positive head-thorax-abdomen body-length reference.")

    time_s = (frames - frames[0]) / fps
    tracks = {}
    for leg in LEGS:
        native_relative = np.stack([_xyz(row, f"{leg}-Claw") for row in rows]) - thorax
        normalized = native_relative / body_length_ref
        tracks[leg] = {"native_relative": native_relative, "body_length_normalized": normalized}

    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "claw_trajectories_bodylength_normalized.csv"
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["frame_idx", "video_id", "time_s", "leg", "dx_native", "dy_native", "dz_native", "dx_bodylength", "dy_bodylength", "dz_bodylength"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            for leg in LEGS:
                native = tracks[leg]["native_relative"][i]
                norm = tracks[leg]["body_length_normalized"][i]
                writer.writerow({
                    "frame_idx": frames[i], "video_id": video_ids[0], "time_s": f"{time_s[i]:.8f}", "leg": leg,
                    "dx_native": f"{native[0]:.8f}", "dy_native": f"{native[1]:.8f}", "dz_native": f"{native[2]:.8f}",
                    "dx_bodylength": f"{norm[0]:.8f}", "dy_bodylength": f"{norm[1]:.8f}", "dz_bodylength": f"{norm[2]:.8f}",
                })

    colors = dict(zip(LEGS, plt.get_cmap("tab10").colors[:6]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for axis_i, axis_name in enumerate(AXES):
        for leg in LEGS:
            values = tracks[leg]["body_length_normalized"][:, axis_i]
            axes[axis_i].plot(time_s, values, color=colors[leg], linewidth=1.5, label=f"{leg} ({LEG_NAMES[leg]})")
        axes[axis_i].set_ylabel(f"Δ{axis_name.upper()} / body length")
        axes[axis_i].grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, loc="best")
    axes[-1].set_xlabel("Time from first sampled frame (s; 120 Hz per paper)")
    fig.suptitle("NMFv2 real-fly claw trajectories, thorax-relative and body-length normalized\nOne untethered wildtype (PR) female, 0.3 s annotated episode")
    fig.tight_layout()
    plot_path = output_dir / "claw_xyz_bodylength_normalized.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    leg_summary = {}
    for leg in LEGS:
        pos = tracks[leg]["native_relative"]
        norm = tracks[leg]["body_length_normalized"]
        leg_summary[leg] = {
            "name": LEG_NAMES[leg],
            "relative_native_range_xyz": {axis: [float(np.min(pos[:, i])), float(np.max(pos[:, i]))] for i, axis in enumerate(AXES)},
            "relative_range_body_lengths_xyz": {axis: float(np.ptp(norm[:, i])) for i, axis in enumerate(AXES)},
            "z_excursion_body_lengths": float(np.ptp(norm[:, 2])),
        }

    result = {
        "dataset": "NeuroMechFly v2 experimental walking kinematics",
        "dataset_doi": "10.7910/DVN/3MCEYR",
        "paper_doi": "10.1038/s41592-024-02497-y",
        "license": "CC0-1.0",
        "input": {
            "path": str(csv_path), "bytes": len(raw), "md5": hashlib.md5(raw).hexdigest(),
            "sha256": hashlib.sha256(raw).hexdigest(), "dataverse_md5_expected": expected_csv_md5,
            "dataverse_md5_match": hashlib.md5(raw).hexdigest() == expected_csv_md5,
        },
        "source_video": {
            "path": str(video_path), "available": video_raw is not None,
            "bytes": len(video_raw) if video_raw is not None else None,
            "md5": hashlib.md5(video_raw).hexdigest() if video_raw is not None else None,
            "dataverse_md5_expected": "322facd5bbb0d6f72309b83fdb77575d",
            "dataverse_md5_match": hashlib.md5(video_raw).hexdigest() == "322facd5bbb0d6f72309b83fdb77575d" if video_raw is not None else None,
        },
        "experimental_conditions": {
            "species": "Drosophila melanogaster",
            "strain": "wildtype (PR)",
            "sex": "female",
            "age": "4–5 days post-eclosion",
            "temperature_c": 25,
            "relative_humidity_percent": 50,
            "light_dark_cycle_hours": "12:12",
            "behavior": "untethered straight walking through a corridor",
            "corridor_mm": [12, 4, 2],
            "recording_fps": 360,
            "annotation_sampling_fps": fps,
            "keypoint_method": "manual annotation of 3D keypoints from three views; reconstruction assumes 90-degree prism orientation",
            "coordinate_scale": "aligned to the NeuroMechFly skeleton by full-leg scaling; CSV does not specify physical units",
        },
        "recording": {
            "video_id": video_ids[0], "rows": len(rows), "unique_frames": len(frames),
            "frame_idx_start": int(frames[0]), "frame_idx_end": int(frames[-1]),
            "sampling_hz": fps, "recorded_span_s": float(time_s[-1]),
            "thorax_xyz_native_constant": bool(np.allclose(thorax, thorax[0])),
            "body_length_reference_native_median": body_length_ref,
            "body_length_definition": "median over frames of ||He-Th|| + ||Th-Abd|| in supplied coordinate units",
            "coordinate_units": "not declared in the CSV; positions were aligned/scaled to the NeuroMechFly skeleton template",
            "plot_and_export": {"plot": str(plot_path), "normalized_trajectories": str(long_path)},
        },
        "per_leg_claw_excursion": leg_summary,
        "not_estimated": {
            "population_or_speed_conditioned_cadence": "single 0.3 s episode; insufficient independent cycles and no matched speed series",
            "stance_duration_or_contact": "CSV has no stance labels or ground-plane/contact reference; do not infer from z alone",
            "absolute_step_length_mm": "no calibrated mm scale or body translation in this body-registered table",
            "interleg_phase_distribution": "one short sequence and no stance segmentation; curves are descriptive only",
        },
        "source_processing_note": "Paper reports 16 candidate stance-to-stance/swing-to-swing steps, 7 rejected for closure, and 9 selected steps time-warped to a 0.135 s median to create a simulation template. That template duration is not a population cadence estimate and is not used here.",
    }
    (output_dir / "gait_reference_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("/data/datasets/raw/drosophila-behavior/nmfv2-2024-walking/annotated_3d_walking_kinematics.csv"))
    parser.add_argument("--out", type=Path, default=Path("/data/artifacts/by-project/malecns-gpu/engineering/real-fly-reference-20261006-v1/gait/nmfv2"))
    parser.add_argument("--fps", type=float, default=120.0, help="Downsampled annotation rate reported in the NMFv2 paper.")
    args = parser.parse_args()
    result = analyze(args.csv, args.out, args.fps)
    print(json.dumps({"frames": result["recording"]["unique_frames"], "duration_s": result["recording"]["recorded_span_s"], "body_length_native": result["recording"]["body_length_reference_native_median"], "csv_md5_matches_api": result["input"]["dataverse_md5_match"], "video_md5_matches_api": result["source_video"]["dataverse_md5_match"], "outputs": result["recording"]["plot_and_export"]}, indent=2))


if __name__ == "__main__":
    main()
