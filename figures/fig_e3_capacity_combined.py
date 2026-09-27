#!/usr/bin/env python3
"""Render the E3b capacity sweep as one shared-k plot with two labeled scales."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "e3b_h1_h5_curve.csv"
OUTPUT = HERE / "fig_e3_capacity_combined"


def main() -> None:
    df = pd.read_csv(DATA)
    k = df["k"].to_numpy(float)
    positive = k > 0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, left = plt.subplots(figsize=(4.35, 3.05))
    right = left.twinx()

    # The envelope is about the location of the largest observed D-C gap;
    # it is not a confidence band for either rate.
    left.axvspan(6, 32, color="#E69F00", alpha=0.075, zorder=0)
    left.axvline(20, color="#777777", ls=":", lw=0.9, zorder=1)

    for column, name, color in (
        ("alpha", "C", "#0072B2"),
        ("emr_d", "D", "#D55E00"),
    ):
        value = 100 * df[column].to_numpy(float)
        low = 100 * df[f"{column}_ci_low"].to_numpy(float)
        high = 100 * df[f"{column}_ci_high"].to_numpy(float)
        left.fill_between(k, low, high, color=color, alpha=0.13, lw=0, zorder=2)
        left.plot(k[positive], value[positive], "o-", color=color,
                  lw=1.8, ms=3.8, label=name, zorder=5)
        left.scatter([0], [value[0]], marker="D", s=25, color=color,
                     edgecolor="white", linewidth=0.4, zorder=6)

    gap = 100 * df["tau"].to_numpy(float)
    low = 100 * df["tau_ci_low"].to_numpy(float)
    high = 100 * df["tau_ci_high"].to_numpy(float)
    right.axhline(0, color="#555555", lw=0.65, alpha=0.65, zorder=1)
    right.errorbar(k[positive], gap[positive],
                   yerr=np.vstack((gap[positive] - low[positive],
                                   high[positive] - gap[positive])),
                   fmt="s--", color="#333333", ecolor="#888888",
                   lw=1.25, elinewidth=0.65, capsize=1.6, ms=3.0,
                   label="D-C", zorder=4)
    right.errorbar([0], [gap[0]],
                   yerr=np.array([[gap[0] - low[0]], [high[0] - gap[0]]]),
                   fmt="D", color="#333333", ecolor="#888888",
                   elinewidth=0.65, capsize=1.6, ms=3.5, zorder=4)
    right.scatter([6], [gap[df.index[df["k"].eq(6)][0]]], s=33,
                  color="#333333", edgecolor="white", lw=0.5, zorder=6)

    left.set_xscale("symlog", linthresh=1, base=2)
    left.set_xlim(-0.12, 72)
    left.set_xticks([0, 1, 2, 4, 8, 16, 32, 64])
    left.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "64"])
    left.set_ylim(-3, 103)
    right.set_ylim(-15, 30)
    right.set_yticks([-10, 0, 10, 20, 30])
    left.set_xlabel(r"Free prompt tokens $k$")
    left.set_ylabel("D, C success (%)")
    right.set_ylabel("D-C gap (pp)")
    left.grid(axis="y", color="#D9DDE3", lw=0.55, zorder=0)
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)

    handles1, labels1 = left.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    left.legend(handles1 + handles2, labels1 + labels2, ncol=3,
                loc="upper left", frameon=True, framealpha=0.88,
                borderpad=0.25, handlelength=1.7, columnspacing=0.9)
    right.annotate("$k=6$", xy=(6, 13.3), xytext=(0, 5),
                   textcoords="offset points", fontsize=8.0,
                   ha="center", va="bottom", color="#333333")
    left.annotate("$k=20$", xy=(20, 1), xycoords=("data", "axes fraction"),
                  xytext=(2, -2), textcoords="offset points", fontsize=7.8,
                  ha="left", va="top", color="#666666")
    left.annotate("fixed", xy=(0, 0), xycoords=("data", "axes fraction"),
                  xytext=(0, -15), textcoords="offset points", fontsize=7.8,
                  ha="center", va="top", color="#666666")

    fig.tight_layout(pad=0.5)
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT.with_suffix(f".{extension}"),
                    dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
