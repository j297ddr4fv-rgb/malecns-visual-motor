"""Import author-provided measured optomotor summaries without refitting them."""
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
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.data_root / "members/Matlab Outputs/Fig1_HighVSLow.mat"
    original = loadmat(source, simplify_cells=True)["D_highlow"]
    time_s = np.asarray(original["timeX"], dtype=float) / 1000
    mean = np.asarray(original["respMatPlot"], dtype=float)
    sem = np.asarray(original["respMatSemPlot"], dtype=float)
    if mean.shape != (len(time_s), 2) or sem.shape != mean.shape:
        raise ValueError("author summary dimensions differ from documented plotting code")
    conditions = (("high_contrast", 1.0), ("low_contrast", 0.25))
    windows = (("pre", -1, 0), ("early", 0, 0.5), ("late", 1, 5))
    result = {
        "source_doi": "10.5281/zenodo.19424913",
        "paper_doi": "10.7554/eLife.86076",
        "data_kind": "author_processed_measured_behavior_mean_and_sem",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "species": "Drosophila melanogaster",
        "strain": "IsoD1",
        "apparatus": "tethered walking on an air-supported ball",
        "stimulus": {"duration_s": 5, "temporal_frequency_hz": 1,
                     "angular_wavelength_deg": 60, "pattern": "sinusoidal grating"},
        "direction_alignment": "author CombineOpposites antisymmetrically averages opposite stimulus directions",
        "time_unit": "seconds; converted from author milliseconds",
        "response_unit": "degrees_per_second",
        "conditions": {},
        "scope": [
            "Imported published population curves, not a new population-level reanalysis.",
            "Source code lists four recording folders; independent fly count is not inferred.",
            "An example raw recording and its stimulus/temperature metadata are retained separately.",
            "One example metadata reports box temperature 36 C; this is not asserted for all recordings.",
            "Panoramic rotation on a ball differs from a moving planar panel and a freely moving simulated body.",
            "No fitted simulator parameters or biological success decision follow from this import.",
        ],
    }
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.axvspan(0, 5, color="gray", alpha=0.1)
    ax.axhline(0, color="black", lw=0.8)
    for i, (name, contrast) in enumerate(conditions):
        entry = {"contrast": contrast, "window_means_deg_s": {}}
        for label, start, end in windows:
            mask = (time_s >= start) & (time_s < end)
            entry["window_means_deg_s"][label] = float(np.mean(mean[mask, i]))
        result["conditions"][name] = entry
        ax.plot(time_s, mean[:, i], label=f"contrast {contrast:g}")
        ax.fill_between(time_s, mean[:, i] - sem[:, i], mean[:, i] + sem[:, i], alpha=0.2)
    for i, t in enumerate(time_s):
        rows.append([t, mean[i, 0], sem[i, 0], mean[i, 1], sem[i, 1]])
    with (args.out / "measured-optomotor-curves.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "contrast1_mean_deg_s", "contrast1_sem_deg_s",
                         "contrast0p25_mean_deg_s", "contrast0p25_sem_deg_s"])
        writer.writerows(rows)
    ax.set(xlabel="Time from motion onset (s)",
           ylabel="Stimulus-aligned turning response (deg/s)",
           title="Measured fly optomotor response: author mean ± SEM")
    ax.legend()
    fig.savefig(args.out / "measured-optomotor-curves.png", dpi=180)
    plt.close(fig)
    (args.out / "reference-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
