#!/usr/bin/env python3
"""Rebuild paired and group summaries from a completed replay's CSV traces."""
from pathlib import Path
import argparse
import csv
import json
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    root = args.root
    state = read(root / "state.json")
    if state.get("status") != "complete":
        raise RuntimeError("queue is not complete")

    runs = {}
    reconstructed = []
    for stored in state["runs"]:
        key = (stored["variant"], stored["initialization_seed"], stored["condition"])
        run_dir = root / "runs" / key[0] / stored["case_id"]
        summary = read(run_dir / "summary.json")
        with (run_dir / "neural_trace.csv").open() as handle:
            trace = [row for row in csv.DictReader(handle) if int(row["seq"]) >= 200]
        if not trace:
            raise RuntimeError(f"missing active trace rows: {stored['case_id']}")
        mean = lambda field: float(np.mean([float(row[field]) for row in trace]))
        record = dict(summary)
        record.update({
            "DNp15_L_hz": mean("DNp15_L_hz"),
            "DNp15_R_hz": mean("DNp15_R_hz"),
            "DNp15_opponent_hz": mean("DNp15_L_hz") - mean("DNp15_R_hz"),
            "mean_raw_P": mean("raw_P"),
            "mean_network_spikes_per_5ms": mean("network_spikes"),
        })
        runs[key] = record
        reconstructed.append({
            "variant": key[0], "rewiring_seed": record["rewiring_seed"],
            "initialization_seed": key[1], "condition": key[2],
            "retinal_movie_sha256": record["retinal_movie_sha256"],
            "fitted_drive_stream_sha256": record["fitted_drive_stream_sha256"],
            "DNp15_L_hz": record["DNp15_L_hz"], "DNp15_R_hz": record["DNp15_R_hz"],
            "DNp15_opponent_hz": record["DNp15_opponent_hz"],
            "mean_raw_P": record["mean_raw_P"],
            "mean_network_spikes_per_5ms": record["mean_network_spikes_per_5ms"],
        })

    variants = list(dict.fromkeys(row["variant"] for row in reconstructed))
    comparisons, groups = [], collections.defaultdict(list)
    for variant in variants:
        for seed in (1729, 1730, 1731):
            static = runs[variant, seed, "static"]
            for condition in ("positive60", "negative60"):
                row = runs[variant, seed, condition]
                reference = runs["original", seed, condition]
                if (row["retinal_movie_sha256"] != reference["retinal_movie_sha256"]
                        or row["fitted_drive_stream_sha256"] != reference["fitted_drive_stream_sha256"]):
                    raise RuntimeError("nonidentical paired input")
                item = {
                    "variant": variant, "rewiring_seed": row["rewiring_seed"],
                    "initialization_seed": seed, "condition": condition,
                    "DNp15_opponent_moving_minus_static_hz":
                        row["DNp15_opponent_hz"] - static["DNp15_opponent_hz"],
                    "mean_P": row["mean_raw_P"], "static_P": static["mean_raw_P"],
                    "shared_retina_and_current": True,
                }
                comparisons.append(item)
                groups[variant, condition].append(item)

    def write_csv(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(root / "reconstructed_run_summaries.csv", reconstructed)
    write_csv(root / "paired_effects.csv", comparisons)
    summaries = []
    for variant in variants:
        for condition in ("positive60", "negative60"):
            rows = groups[variant, condition]
            values = np.asarray([row["DNp15_opponent_moving_minus_static_hz"] for row in rows])
            summaries.append({
                "variant": variant, "condition": condition,
                "n_initializations": len(rows), "contrast_mean_hz": float(values.mean()),
                "contrast_sd_hz": float(values.std(ddof=1)),
                "mean_P": float(np.mean([row["mean_P"] for row in rows])),
            })
    write_csv(root / "group_results.csv", summaries)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.8))
    colors = ["#245d91", "#ae5633", "#43845a", "#825c9b"]
    for j, variant in enumerate(variants):
        for i, condition in enumerate(("positive60", "negative60")):
            values = np.asarray([
                row["DNp15_opponent_moving_minus_static_hz"]
                for row in groups[variant, condition]
            ])
            pos = i + (j - (len(variants) - 1) / 2) * .16
            label = "Original" if variant == "original" else "Rewired " + variant.rsplit("_", 1)[-1][-2:]
            axes[0].scatter(pos + np.linspace(-.02, .02, len(values)), values, s=12, color=colors[j], alpha=.7)
            axes[0].errorbar(pos, values.mean(), yerr=values.std(ddof=1), fmt="o",
                             color=colors[j], capsize=2, ms=3, label=label if i == 0 else None)
        for i, condition in enumerate(("static", "positive60", "negative60")):
            values = np.asarray([runs[variant, seed, condition]["mean_raw_P"] for seed in (1729, 1730, 1731)])
            pos = i + (j - (len(variants) - 1) / 2) * .16
            axes[1].scatter(pos + np.linspace(-.02, .02, len(values)), values, s=12, color=colors[j], alpha=.7)
    axes[0].set_xticks([0, 1], ["+60", "-60"])
    axes[0].set_ylabel("DNp15 L-R: moving minus static (Hz)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xticks([0, 1, 2], ["Static", "+60", "-60"])
    axes[1].set_ylabel("Neural locomotor drive P")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.axhline(0, color=".65", lw=.5)
        axis.grid(axis="y", alpha=.12)
    figure_dir = args.figures or root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=.7)
    for extension in ("pdf", "png"):
        fig.savefig(figure_dir / f"common_input_control.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Same-input connection-structure comparison", "",
        "Run means are recomputed from neural_trace.csv after the first 200 warm-up frames. "
        "Every network receives the same retinal pixels and fitted T4/T5 currents.", "",
        "| Network | Stimulus | Contrast (Hz, mean ± SD) | Mean neural P |",
        "|---|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['condition']} | "
            f"{row['contrast_mean_hz']:+.3f} ± {row['contrast_sd_hz']:.3f} | {row['mean_P']:.4f} |"
        )
    lines += [
        "",
        "No body is simulated in this replay; P is a neural drive readout, not a walking endpoint. "
        "The contrast uses unsmoothed DNp15 event rates before steering decoding, corrected by matched static input.",
        "",
        "The sampled null graphs do not establish a population-wide property of rewired networks or identify a selective "
        "biological pathway. Global rewiring preserves marginal degrees and signed strengths while changing many partners.",
    ]
    (root / "analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
