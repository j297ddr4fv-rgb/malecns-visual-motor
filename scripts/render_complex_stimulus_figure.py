#!/usr/bin/env python3
"""Render the complex-stimulus panel from the released per-run CSV."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/per_run_results.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/Fig12_illusions.pdf"))
    a = p.parse_args()
    with a.data.open() as handle:
        rows = list(csv.DictReader(handle))
    conditions = ["stepped40", "reversephi40", "flicker40", "stepped200",
                  "interleave_short", "interleave_long", "counterphase"]
    labels = ["Stepped\n40 ms", "Reverse-phi\nstimulus", "Polarity\nflicker", "Stepped\n200 ms",
              "Short gray\ninterval", "Long gray\ninterval", "Counterphase\ngrating"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(174 / 25.4, 3.2))
    for i, c in enumerate(conditions):
        values = np.array([float(r["yaw_deg"]) for r in rows
                           if r["family"] == "illusion" and r["condition"] == c])
        if len(values) != 10:
            raise ValueError(f"Expected ten runs for {c}; found {len(values)}")
        color = "#a95735" if c in ("reversephi40", "flicker40") else "#245d91"
        ax.scatter(i + np.linspace(-.10, .10, len(values)), values,
                   s=20, color=color, alpha=.65, zorder=3)
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="o",
                    color="black", capsize=4, zorder=4)
    ax.set_xticks(range(len(conditions)), labels, rotation=16)
    ax.set_ylabel("Net yaw change (deg)")
    ax.axhline(0, color=".7", lw=.7)
    ax.grid(axis="y", alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=.8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix(".png"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
