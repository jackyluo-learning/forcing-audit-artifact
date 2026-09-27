#!/usr/bin/env python3
"""Trained/control rate comparison for v2; DP overlays deferred.
Source rates and zero-count bound retained from fig_privacy.py.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COL_PT = "#B03A2E"

# the final run: (label, FPR = EMR(C), TPR = EMR(D)), rates in [0,1].
MODELS = [
    ("GPT-2 124M",   0.520, 0.520),
    ("GPT-2-M 355M", 0.333, 0.667),
    ("Pythia-1.4B",  0.500, 0.600),
    ("Pythia-2.8B",  0.000, 0.167),   # control 0/12 -> plotted at CP upper bound
]
CP_UPPER_0_of_12 = 0.2646


def make_figure(models, out_stem):
    fig, ax = plt.subplots(figsize=(3.7, 4.0))
    x = np.linspace(0, 1, 400)

    ax.plot(x, x, color="#9A9A9A", lw=1.1, ls=":", zorder=1)
    ax.text(0.83, 0.87, "equal rates", fontsize=9.5, color="#7A7A7A",
            rotation=45, ha="center", va="center", zorder=4,
            bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))

    # DP frontiers deliberately omitted in the narrowed v2 scope.

    for i, (name, fpr, tpr) in enumerate(models):
        capped = (fpr == 0.0)
        xf = CP_UPPER_0_of_12 if capped else fpr
        ax.plot([xf], [tpr], marker="o", ms=8.5, color=COL_PT, mec="white",
                mew=0.9, zorder=6)
        if capped:
            ax.annotate("", xy=(0.0, tpr), xytext=(xf, tpr),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color=COL_PT,
                                        alpha=0.75))
        dx, dy, ha = {
            0: (0.035, -0.052, "left"),
            1: (0.035, 0.015, "left"),
            2: (0.035, -0.035, "left"),
            3: (0.035, -0.062, "left"),
        }[i]
        ax.annotate(name, xy=(xf, tpr), xytext=(xf + dx, tpr + dy),
                    fontsize=9.5, color=COL_PT, ha=ha, va="center")

    ax.text(0.70, 0.235,
            "diagonal: equal rates\nproximity does not\nestablish equivalence",
            fontsize=9.5, color="#333333", ha="center", va="center",
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.6, pad=3.0, alpha=0.93),
            zorder=5)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel(r"control rate $\mathrm{EMR}(C)$  (%)")
    ax.set_ylabel(r"$\mathrm{EMR}(D)$  (%)")
    ax.grid(True, lw=0.5, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, dpi=400 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.015)
        print("wrote", p)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__) or ".",
                                                  "fig_control_comparison_v2"))
    a = ap.parse_args()
    make_figure(MODELS, a.out)


if __name__ == "__main__":
    main()
