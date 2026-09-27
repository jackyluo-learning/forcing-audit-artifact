#!/usr/bin/env python3
"""Render the manuscript's final E3b prompt-capacity figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "e3b_h1_h5_curve.csv"
OUT_STEM = HERE / "fig_e3_capacity_sweep"


def main() -> None:
    df = pd.read_csv(SOURCE)
    x = df["k"].to_numpy(float)
    positive = x >= 1

    plt.rcParams.update(
        {
            "font.size": 14.0,
            "axes.labelsize": 17.0,
            "legend.fontsize": 12.5,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.4, 3.45), constrained_layout=True
    )

    # Match the existing probe-spectrum figure: controls blue, trained red.
    series = [
        ("alpha", r"$\widehat{\alpha}_k=\mathrm{EMR}(C)$", "#0072B2"),
        ("emr_d", r"$\mathrm{EMR}(D)$", "#D55E00"),
    ]
    for prefix, label, color in series:
        y = 100 * df[prefix].to_numpy(float)
        lo = 100 * df[f"{prefix}_ci_low"].to_numpy(float)
        hi = 100 * df[f"{prefix}_ci_high"].to_numpy(float)
        ax1.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax1.plot(
            x[positive], y[positive], "o-", color=color, lw=2.3, ms=5.8,
            label=label, zorder=3
        )
        # k=0 is the fixed-prompt anchor rather than a GCG capacity.
        ax1.scatter(
            [x[0]], [y[0]], marker="D", s=42, color=color, edgecolor="white",
            linewidth=0.6, zorder=4
        )
    ax1.set_ylabel(r"$\widehat{\alpha}_k,\ \mathrm{EMR}(D)$ (\%)")
    ax1.set_ylim(-3, 103)
    ax1.legend(loc="upper left", frameon=True)
    ax1.text(
        0.53, 0.965, "(a)", transform=ax1.transAxes,
        fontweight="bold", va="top", ha="center"
    )

    tau = 100 * df["tau"].to_numpy(float)
    lo = 100 * df["tau_ci_low"].to_numpy(float)
    hi = 100 * df["tau_ci_high"].to_numpy(float)
    ax2.axhline(0, color="#666666", lw=1.0, zorder=1)
    ax2.axvspan(
        6, 32, color="#E69F00", alpha=0.10, zorder=0
    )
    ax2.errorbar(
        x[positive], tau[positive],
        yerr=np.vstack((tau[positive] - lo[positive], hi[positive] - tau[positive])),
        fmt="o-", color="#4D4D4D", ecolor="#666666", lw=1.9,
        elinewidth=1.5, capsize=3.0, ms=5.6, zorder=3
    )
    ax2.errorbar(
        [x[0]], [tau[0]],
        yerr=np.array([[tau[0] - lo[0]], [hi[0] - tau[0]]]),
        fmt="D", color="#4D4D4D", ecolor="#666666", elinewidth=1.2,
        capsize=3.0, ms=5.6, zorder=3
    )
    row6 = df.loc[df["k"].eq(6)].iloc[0]
    ax2.scatter(
        [6], [100 * row6["tau"]], s=76, color="#CC3311",
        edgecolor="white", linewidth=0.8, zorder=5
    )
    ax2.annotate(
        "largest observed gap\n$k=6$: +13.3 pp",
        xy=(6, 100 * row6["tau"]), xytext=(8.8, 23.0),
        ha="left", va="top", fontsize=12.0,
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.9),
    )
    ax2.set_ylabel(r"$\widehat{\tau}_{\mathrm{rec}}$ (pp)")
    ax2.set_ylim(-15, 28)
    ax2.text(
        0.015, 0.965, "(b)", transform=ax2.transAxes,
        fontweight="bold", va="top"
    )

    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=1, base=2)
        ax.set_xlim(-0.12, 72)
        ax.set_xticks([0, 1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlabel(r"$k$")
        ax.axvline(0.5, color="#AAAAAA", ls=":", lw=0.8, zorder=1)
        ax.annotate(
            "fixed", xy=(0, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -23), textcoords="offset points", ha="center",
            va="top", fontsize=10.5, color="#777777"
        )
        ax.axvline(20, color="#888888", ls="--", lw=0.9, zorder=1)
        ax.annotate(
            "$k=20$", xy=(20, 1), xycoords=("data", "axes fraction"),
            xytext=(3, -4), textcoords="offset points", ha="left",
            va="top", fontsize=11.0, color="#666666"
        )
        ax.grid(axis="y", color="#D9DDE3", alpha=0.75, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(
        OUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        OUT_STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )


if __name__ == "__main__":
    main()
