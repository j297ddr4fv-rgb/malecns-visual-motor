#!/usr/bin/env python3
"""Compare Mano et al. author-processed turning curves with saved FlyGym yaw.

This is an arrays-only plotting utility. It does not import or step a simulator,
read images, or modify its inputs. The comparison is descriptive across distinct
paradigms: tethered walking on an air-supported ball versus one free-body FlyGym
trajectory per stimulus direction.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENGINEERING = Path("/data/artifacts/by-project/malecns-gpu/engineering")
DEFAULT_CURVES = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/real-fly-reference-20261006-v1/"
    "optomotor/measured-optomotor-curves.csv"
)
DEFAULT_STATIC = ENGINEERING / "flygym-dnp15-ap-online-pilot-20261007-v2/captures/static"
DEFAULT_POSITIVE = ENGINEERING / "flygym-dnp15-ap-online-pilot-20261007-v3/captures/positive60"
DEFAULT_NEGATIVE = ENGINEERING / "flygym-dnp15-ap-online-pilot-20261007-v3/captures/negative60"
DEFAULT_OUTPUT = ENGINEERING / "flygym-dnp15-ap-online-pilot-20261007-v3/analysis/real-vs-sim"
DEFAULT_REFERENCE_SUMMARY = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/real-fly-reference-20261006-v1/"
    "optomotor/reference-summary.json"
)
DEFAULT_COMBINE_SOURCE = Path(
    "/data/datasets/raw/drosophila-behavior/mano-2023-optomotor/raw-highlow-example/"
    "members/Behavior Analysis Matlab Code/analysis/utils/CombineOpposites.m"
)

NATIVE_DT_S = 0.0001
MODEL_K_FIRST = 50001
MODEL_K_STOP = 100001  # python-exclusive, includes heading sample k=100000
STIM_ONSET_S = 5.0
DISPLAY_BIN_S = 0.05
DISPLAY_DURATION_S = 5.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object at {path}")
    return value


def csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = ("time_s", "contrast1_mean_deg_s", "contrast1_sem_deg_s",
                "contrast0p25_mean_deg_s", "contrast0p25_sem_deg_s")
    missing = [key for key in required if not rows or key not in rows[0]]
    if missing:
        raise RuntimeError(f"author curve CSV lacks required columns: {missing}")
    columns = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
               for key in required}
    mask = (columns["time_s"] >= 0.0) & (columns["time_s"] <= DISPLAY_DURATION_S)
    return {key: values[mask] for key, values in columns.items()}


def capture_input(case_dir: Path, role: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = case_dir / "manifest.json"
    summary_path = case_dir / "summary.json"
    record_path = case_dir / "case_record.json"
    trace_path = case_dir / "trajectory.npz"
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    case_record = load_json(record_path)
    if manifest.get("status") != "complete" or summary.get("status") != "complete":
        raise RuntimeError(f"{role} capture does not have complete saved manifest and summary")
    if role != "static" and case_record.get("status") != "complete":
        raise RuntimeError(f"{role} capture case_record is not complete: {case_record.get('status')}")

    recovery_path = case_dir.parent.parent / "analysis" / "static-postcapture-recovery-addendum.json"
    recovery = None
    if role == "static":
        recovery = load_json(recovery_path)
        if recovery.get("status") != "complete_raw_capture_reporting_failed":
            raise RuntimeError("static v2 recovery sidecar does not certify complete raw capture")
        if case_record.get("status") != "failed_raw_preserved":
            raise RuntimeError(f"static runner status changed unexpectedly: {case_record.get('status')}")
        if recovery.get("case_dir") != str(case_dir.resolve()):
            raise RuntimeError("static recovery sidecar refers to a different case directory")

    trace_hash = sha256(trace_path)
    manifest_trace_hash = (manifest.get("artifact_sha256") or {}).get(str(trace_path.resolve()))
    if manifest_trace_hash != trace_hash:
        raise RuntimeError(f"{role} trajectory hash does not match saved manifest")
    if role == "static":
        verified = recovery.get("verified_files") or {}
        record = verified.get(str(trace_path.resolve()), {})
        if record.get("sha256") != trace_hash:
            raise RuntimeError("static trajectory hash does not match recovery sidecar")

    with np.load(trace_path, allow_pickle=False) as archive:
        if "thorax_heading_yaw_rad" not in archive or "time_s" not in archive:
            raise RuntimeError(f"{role} native trajectory lacks heading/time arrays")
        heading = np.asarray(archive["thorax_heading_yaw_rad"], dtype=np.float64)
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
    if heading.size != 100001 or time_s.size != 100001:
        raise RuntimeError(f"{role} trace has unexpected native shape: heading={heading.shape}, time={time_s.shape}")
    return {
        "role": role,
        "case_dir": str(case_dir.resolve()),
        "case_record_status": case_record.get("status"),
        "manifest_status": manifest.get("status"),
        "summary_status": summary.get("status"),
        "trace_samples": int(heading.size),
        "trace_sha256": trace_hash,
        "manifest_sha256": sha256(manifest_path),
        "summary_sha256": sha256(summary_path),
        "case_record_sha256": sha256(record_path),
        "recovery_sidecar": None if recovery is None else {
            "path": str(recovery_path.resolve()),
            "sha256": sha256(recovery_path),
            "status": recovery.get("status"),
        },
    }, {"heading_rad": heading, "time_s": time_s}


def binned_yaw_rate(capture: dict[str, np.ndarray]) -> dict[str, Any]:
    """Use observer k→pose k−1 mapping, then bin native finite differences."""
    heading = np.unwrap(capture["heading_rad"])
    time = capture["time_s"]
    k = np.arange(MODEL_K_FIRST, MODEL_K_STOP, dtype=np.int64)
    pose_indices = k - 1
    t = pose_indices.astype(np.float64) * NATIVE_DT_S - STIM_ONSET_S
    # The recorded heading vector is indexed by observer record k; record k
    # describes the physical pose at k−1. Check that the mapped pose times are
    # precisely the intended 5.0000–9.9999 s display interval.
    expected_pose_time = pose_indices.astype(np.float64) * NATIVE_DT_S
    if not np.allclose(time[pose_indices], expected_pose_time, rtol=0.0, atol=2e-10):
        raise RuntimeError("native time_s does not match the frozen observer k→pose k−1 mapping")
    h = heading[MODEL_K_FIRST:MODEL_K_STOP]
    if h.size != 50000:
        raise RuntimeError(f"expected 50,000 observer heading samples, got {h.size}")
    rate_deg_s = np.rad2deg(np.diff(h)) / NATIVE_DT_S
    interval_midpoints = 0.5 * (t[:-1] + t[1:])
    edges = np.linspace(0.0, DISPLAY_DURATION_S, int(DISPLAY_DURATION_S / DISPLAY_BIN_S) + 1)
    bin_index = np.searchsorted(edges, interval_midpoints, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(edges) - 2)
    rates = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    durations = np.zeros(len(edges) - 1, dtype=np.float64)
    for i in range(len(rates)):
        selected = rate_deg_s[bin_index == i]
        counts[i] = selected.size
        durations[i] = selected.size * NATIVE_DT_S
        if selected.size:
            rates[i] = float(np.mean(selected))
    return {
        "bin_left_s": edges[:-1],
        "bin_right_s": edges[1:],
        "bin_center_s": 0.5 * (edges[:-1] + edges[1:]),
        "yaw_rate_deg_s": rates,
        "interval_count": counts,
        "effective_duration_s": durations,
        "sample_time_start_s": float(t[0]),
        "sample_time_end_s": float(t[-1]),
        "finite_difference_count": int(rate_deg_s.size),
        "min_intervals_per_bin": int(np.min(counts)),
        "max_intervals_per_bin": int(np.max(counts)),
        "last_bin_duration_s": float(durations[-1]),
        "last_bin_interval_count": int(counts[-1]),
        "time_formula": "t=(k-1)*0.0001-5.0; heading record k maps to pose k-1",
        "heading_record_index_inclusive": [MODEL_K_FIRST, MODEL_K_STOP - 1],
        "physical_pose_index_inclusive": [MODEL_K_FIRST - 1, MODEL_K_STOP - 2],
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def draw_plot(curves: dict[str, np.ndarray], models: dict[str, dict[str, Any]],
              output_png: Path, output_svg: Path) -> None:
    time_real = curves["time_s"]
    fig, (ax_real, ax_model) = plt.subplots(1, 2, figsize=(13.0, 5.8), sharex=True, sharey=True)
    fig.suptitle("Direction-aligned turning response after stimulus onset", fontsize=14, y=0.98)

    high = curves["contrast1_mean_deg_s"]
    high_sem = curves["contrast1_sem_deg_s"]
    low = curves["contrast0p25_mean_deg_s"]
    low_sem = curves["contrast0p25_sem_deg_s"]
    ax_real.plot(time_real, high, color="#0072B2", lw=2.1, label="Mano et al., contrast 1.0")
    ax_real.fill_between(time_real, high - high_sem, high + high_sem, color="#0072B2", alpha=0.18, linewidth=0)
    ax_real.plot(time_real, low, color="#D55E00", lw=2.1, label="Mano et al., contrast 0.25")
    ax_real.fill_between(time_real, low - low_sem, low + low_sem, color="#D55E00", alpha=0.16, linewidth=0)
    ax_real.set_title("Tethered walking on air-supported ball")
    ax_real.set_ylabel("Author-processed turning rate (deg s$^{-1}$)")

    t = models["positive60"]["bin_center_s"]
    omega_pos = models["positive60"]["yaw_rate_deg_s"]
    omega_neg = models["negative60"]["yaw_rate_deg_s"]
    negative_aligned = -omega_neg
    paired = 0.5 * (omega_pos - omega_neg)
    static = models["static"]["yaw_rate_deg_s"]
    ax_model.plot(t, omega_pos, color="#0072B2", lw=1.15, alpha=0.65,
                  label=r"positive stimulus: $\omega_+$")
    ax_model.plot(t, negative_aligned, color="#D55E00", lw=1.15, alpha=0.65,
                  label=r"negative, sign-aligned: $-\omega_-$")
    ax_model.plot(t, paired, color="#222222", lw=2.4,
                  label=r"paired direction rate: $0.5(\omega_+-\omega_-)$")
    ax_model.plot(t, static, color="#777777", lw=1.1, ls=(0, (3, 2)), alpha=0.9,
                  label="static no-stimulus drift (auxiliary)")
    ax_model.set_title("FlyGym free-body yaw, contrast 1.0")
    ax_model.set_ylabel("Thorax body-yaw rate (deg s$^{-1}$)")

    for ax in (ax_real, ax_model):
        ax.axhline(0.0, color="#555555", lw=0.8, alpha=0.55)
        ax.set_xlim(0.0, DISPLAY_DURATION_S)
        ax.set_xlabel("Time from stimulus onset (s)")
        ax.grid(True, which="major", color="#dddddd", lw=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_real.legend(frameon=False, fontsize=8.3, loc="upper left")
    ax_model.legend(frameon=True, framealpha=0.88, facecolor="white", edgecolor="none",
                    fontsize=8.0, loc="upper left")
    all_rates = np.concatenate((high - high_sem, high + high_sem, low - low_sem, low + low_sem,
                                omega_pos, negative_aligned, paired, static))
    common_limit = 20.0 * math.ceil(float(np.nanmax(np.abs(all_rates))) * 1.05 / 20.0)
    ax_real.set_ylim(-common_limit, common_limit)

    caption = (
        "Mano curves: author-processed direction-combined group mean ± SEM (author-reported N=10; 60 Hz source curve). "
        "FlyGym: one saved trajectory per direction, seed 1729; 50 ms finite-difference display bins (100 within-run bins, "
        "not replicates; last bin uses 49.9 ms). The negative direction is sign-flipped before pairing. Static is a separate "
        "no-stimulus drift trace. Tethered ball rotation and free-body thorax yaw are distinct paradigms; no FlyGym contrast-0.25 run was measured."
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=7.7, wrap=True)
    fig.tight_layout(rect=(0.0, 0.105, 1.0, 0.94), w_pad=2.0)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--static-case-dir", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--positive-case-dir", type=Path, default=DEFAULT_POSITIVE)
    parser.add_argument("--negative-case-dir", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--combine-opposites-source", type=Path, default=DEFAULT_COMBINE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / "real_vs_sim_yaw.png"
    output_svg = args.output_dir / "real_vs_sim_yaw.svg"
    output_data = args.output_dir / "plot_data.json"
    output_manifest = args.output_dir / "plot_manifest.json"
    output_readme = args.output_dir / "README.md"
    outputs = (output_png, output_svg, output_data, output_manifest, output_readme)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("refusing to overwrite existing plot artifacts: " + ", ".join(existing))

    capture_meta: dict[str, Any] = {}
    capture_arrays: dict[str, dict[str, np.ndarray]] = {}
    for role, path in (("static", args.static_case_dir), ("positive60", args.positive_case_dir),
                       ("negative60", args.negative_case_dir)):
        capture_meta[role], capture_arrays[role] = capture_input(path, role)
    curves = csv_columns(args.real_curves)
    models = {role: binned_yaw_rate(capture_arrays[role]) for role in capture_arrays}

    plot_data = {
        "schema": "flygym-real-vs-sim-yaw-plot-data-v1",
        "time_s_from_stimulus_onset": curves["time_s"],
        "author_processed_mean_deg_s": {
            "contrast1": curves["contrast1_mean_deg_s"],
            "contrast0p25": curves["contrast0p25_mean_deg_s"],
        },
        "author_processed_sem_deg_s": {
            "contrast1": curves["contrast1_sem_deg_s"],
            "contrast0p25": curves["contrast0p25_sem_deg_s"],
        },
        "model_50ms_yaw_rate_bins": models,
        "direction_aligned_negative_yaw_rate_deg_s": -models["negative60"]["yaw_rate_deg_s"],
        "paired_direction_rate_deg_s": 0.5 * (models["positive60"]["yaw_rate_deg_s"] - models["negative60"]["yaw_rate_deg_s"]),
        "static_no_stimulus_yaw_rate_deg_s": models["static"]["yaw_rate_deg_s"],
    }
    plot_data = to_jsonable(plot_data)

    script_path = Path(__file__).resolve()
    combine_hash = sha256(args.combine_opposites_source)
    reference_summary = load_json(args.reference_summary)
    draw_plot(curves, models, output_png, output_svg)
    output_data.write_text(json.dumps(plot_data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    source_files = [args.real_curves, args.reference_summary, args.combine_opposites_source,
                    args.static_case_dir / "trajectory.npz", args.static_case_dir / "manifest.json",
                    args.static_case_dir / "summary.json", args.static_case_dir / "case_record.json",
                    args.static_case_dir.parent.parent / "analysis" / "static-postcapture-recovery-addendum.json",
                    args.positive_case_dir / "trajectory.npz", args.positive_case_dir / "manifest.json",
                    args.positive_case_dir / "summary.json", args.positive_case_dir / "case_record.json",
                    args.negative_case_dir / "trajectory.npz", args.negative_case_dir / "manifest.json",
                    args.negative_case_dir / "summary.json", args.negative_case_dir / "case_record.json", script_path]
    source_hashes = {str(path.resolve()): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                     for path in source_files}
    binning = {role: {
        "interval_count_per_bin": data["interval_count"],
        "effective_duration_s_per_bin": data["effective_duration_s"],
        "bin_count": int(data["interval_count"].size),
        "min_intervals_per_bin": data["min_intervals_per_bin"],
        "max_intervals_per_bin": data["max_intervals_per_bin"],
        "last_bin_interval_count": data["last_bin_interval_count"],
        "last_bin_duration_s": data["last_bin_duration_s"],
    } for role, data in models.items()}
    manifest = {
        "schema": "flygym-real-vs-sim-yaw-plot-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(script_path), *sys.argv[1:]],
        "output_dir": str(args.output_dir.resolve()),
        "resolved_arguments": {
            "real_curves": str(args.real_curves.resolve()),
            "static_case_dir": str(args.static_case_dir.resolve()),
            "positive_case_dir": str(args.positive_case_dir.resolve()),
            "negative_case_dir": str(args.negative_case_dir.resolve()),
            "reference_summary": str(args.reference_summary.resolve()),
            "combine_opposites_source": str(args.combine_opposites_source.resolve()),
            "display_bin_width_s": DISPLAY_BIN_S,
        },
        "figure": {
            "png": {"path": str(output_png.resolve()), "sha256": sha256(output_png), "size_bytes": output_png.stat().st_size},
            "svg": {"path": str(output_svg.resolve()), "sha256": sha256(output_svg), "size_bytes": output_svg.stat().st_size},
            "data_json": {"path": str(output_data.resolve()), "sha256": sha256(output_data), "size_bytes": output_data.stat().st_size},
        },
        "sources": source_hashes,
        "author_curve": {
            "data_file_sha256": sha256(args.real_curves),
            "reference_summary_sha256": sha256(args.reference_summary),
            "combine_opposites_source_sha256": combine_hash,
            "data_kind": reference_summary.get("data_kind"),
            "apparatus": reference_summary.get("apparatus"),
            "reported_n": 10,
            "curve_sampling_hz": 60.0,
            "curve_time_point_count_in_plotted_interval": int(curves["time_s"].size),
            "processing": "Imported author-processed measured group mean and SEM; opposite directions are combined antisymmetrically as (positive - negative)/2 per author CombineOpposites.",
            "plotted_interval_s": [0.0, 5.0],
        },
        "model": {
            "seed": 1729,
            "contrast_measured": 1.0,
            "contrast_0p25_measured": False,
            "independent_trajectories_per_direction": 1,
            "display_bin_width_s": DISPLAY_BIN_S,
            "display_bin_count": int(DISPLAY_DURATION_S / DISPLAY_BIN_S),
            "display_bin_counts_are_replicates": False,
            "yaw_heading_alignment": {
                "observer_heading_index_k_maps_to_pose_index": "k-1",
                "observer_heading_index_inclusive": [MODEL_K_FIRST, MODEL_K_STOP - 1],
                "mapped_pose_index_inclusive": [MODEL_K_FIRST - 1, MODEL_K_STOP - 2],
                "time_s_formula": "(k-1)*0.0001 - 5.0",
                "sample_time_start_end_s": [0.0, 4.9999],
                "yaw_rate": "unwrap native thorax heading; first difference; degrees per second",
            },
            "direction_alignment": {
                "positive_series": "omega_positive",
                "negative_series_for_display": "-omega_negative",
                "paired_series": "0.5*(omega_positive - omega_negative)",
            },
            "static_trace_role": "no-stimulus drift auxiliary, not direction aligned and not a ball-turning attempt",
            "binning_by_capture": binning,
        },
        "interpretation_limits": [
            "Mano et al. author-processed group curves are tethered walking on an air-supported ball; FlyGym traces are thorax body yaw in a free-body simulation.",
            "Model traces show a single seed-1729 saved trajectory per stimulus direction; 100 within-trajectory bins are not biological or simulation replicates.",
            "The static no-stimulus drift is shown as a separate auxiliary reference and is not subtracted from the paired directional model rate.",
            "Only the author data include contrast 0.25; no FlyGym contrast-0.25 series is fabricated or shown.",
            "The figure is a descriptive comparison and does not equate body yaw with ball rotation or infer biological success.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "gpu_or_simulation_used": False,
        },
    }
    output_manifest.write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    output_readme.write_text(
        "# Real-fly and FlyGym yaw comparison\n\n"
        "This figure places the author-processed Mano et al. optomotor turning curves beside saved FlyGym free-body yaw rates. "
        "The plot is descriptive across different paradigms; it does not treat free-body yaw as an air-ball turning attempt.\n\n"
        "## Reproduce\n\n"
        "From `/home/xvhang/Projects/malecns-gpu`, with the registry FlyGym-reference Python environment:\n\n"
        "```bash\n"
        "MPLCONFIGDIR=/data/cache/flygym/matplotlib CUDA_VISIBLE_DEVICES='' \\\n"
        "/data/environments/projects/flygym-reference/bin/python \\\n"
        "  scripts/plot_flygym_real_vs_sim_yaw.py \\\n"
        f"  --output-dir {args.output_dir.resolve()}\n"
        "```\n\n"
        "The script reads the hashed source CSV and capture files named in `plot_manifest.json`. It does not step the simulator.\n\n"
        "## Time and binning\n\n"
        "The model observer heading record `k` maps to physical pose `k-1`. The plotted range uses heading records 50001 through 100000 and "
        "`t=(k-1)*0.0001-5.0`, so the displayed interval is 0 to 4.9999 s after the nominal 5 s onset. Yaw is unwrapped, finite-differenced, "
        "and averaged into 50 ms display bins. There are 100 time bins per trace; the final bin contains 499 differences and spans 49.9 ms. "
        "Bins are not replicates. The paired model series is `0.5*(omega_positive-omega_negative)`, matching the author’s antisymmetric direction combination.\n\n"
        "The real-fly curves are author-processed group mean ± SEM (reported N=10) for contrast 1.0 and 0.25. The model has one seed-1729 "
        "trajectory for each direction at contrast 1.0 only. Static no-stimulus drift is shown separately.\n",
        encoding="utf-8",
    )
    manifest["figure"]["readme"] = {"path": str(output_readme.resolve()), "sha256": sha256(output_readme), "size_bytes": output_readme.stat().st_size}
    output_manifest.write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"png": str(output_png), "svg": str(output_svg),
                      "manifest": str(output_manifest),
                      "bin_counts": {role: {"min": data["min_intervals_per_bin"],
                                            "max": data["max_intervals_per_bin"],
                                            "last": data["last_bin_interval_count"]}
                                     for role, data in models.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
