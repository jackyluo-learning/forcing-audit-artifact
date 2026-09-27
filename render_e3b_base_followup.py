"""Render the validated E3b k=20 four-cell summary from analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis_json.read_text())
    if data.get("accepted") is not True or data.get("formal_attempts") != 600:
        raise ValueError("Figure requires the accepted complete four-cell analysis")
    m = {row["metric"]: row for row in data["rates_and_contrasts"]}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.7),
                                 gridspec_kw={"width_ratios": [1.12, 1]})
    colors = {"D": "#2B6A9C", "C": "#D18A2D"}
    centers = np.arange(2)
    width = 0.32
    for arm, offset in (("D", -width / 2), ("C", width / 2)):
        rows = [m[f"ft_{arm}"], m[f"base_{arm}"]]
        values = np.array([row["estimate"] * 100 for row in rows])
        lows = values - np.array([row["ci95_low"] * 100 for row in rows])
        highs = np.array([row["ci95_high"] * 100 for row in rows]) - values
        bars = ax.bar(centers + offset, values, width=width, label=arm,
                      color=colors[arm], yerr=np.array([lows, highs]),
                      capsize=3.5, error_kw={"elinewidth": 1.1})
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, max(2, bar.get_height() - 5),
                    f"{row['hits']}/150", ha="center", va="top", fontsize=8,
                    color="white", fontweight="bold")
    ax.set_xticks(centers, ["Fine-tuned", "Base"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact-match success (%)")
    ax.set_title("(a) Four observed cells", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    contrast_keys = ["delta_A3", "tau_rec", "tau_mod", "tau_base"]
    labels = [r"$\Delta_{A3}$: FT(C) − Base(C)",
              r"$\hat\tau_{rec}$: FT(D) − FT(C)",
              r"$\hat\tau_{mod}$: FT(D) − Base(D)",
              r"Base(D) − Base(C)"]
    points = np.array([m[k]["estimate"] * 100 for k in contrast_keys])
    lower = np.array([m[k]["ci95_low"] * 100 for k in contrast_keys])
    upper = np.array([m[k]["ci95_high"] * 100 for k in contrast_keys])
    y = np.arange(len(contrast_keys))[::-1]
    bx.axvline(0, color="#333333", linewidth=0.9)
    for j, key in enumerate(contrast_keys):
        color = "#D18A2D" if key in ("delta_A3", "tau_base") else "#2B6A9C"
        bx.errorbar(points[j], y[j], xerr=[[points[j] - lower[j]],
                                             [upper[j] - points[j]]],
                    fmt="o", color=color, ecolor=color, markersize=5.3,
                    capsize=3.5, elinewidth=1.5)
    bx.set_yticks(y, labels, fontsize=8)
    bx.set_xlim(-18, 48)
    bx.set_xlabel("Difference in success (percentage points)")
    bx.set_title("(b) Descriptive contrasts", loc="left", fontsize=10, fontweight="bold")
    bx.grid(axis="x", alpha=0.2)
    bx.set_axisbelow(True)
    fig.text(0.5, 0.01,
             "k = 20; 25 matched person pairs, 2 fields, 3 attack seeds. "
             "Error bars show 95% person-cluster bootstrap intervals.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.89, wspace=0.54)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_stem.with_suffix(f".{ext}"), dpi=250,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
