#!/usr/bin/env python3
"""
Figure 1 (centerpiece) --- the forcing axis.

Probes ordered by expressivity. As the probe gets more expressive
(fixed -> discovery -> random search -> GCG -> soft prompt), the success rate on
NEVER-TRAINED control records climbs from 0% to 100%, while the gap between
trained and control (EMR_adj, the memorization-attributable rate) stays near
zero. The added "success" is forcing, not recall.

Numbers are the final run, pooled over models. To regenerate from the per-attempt
log instead of the embedded values:  fig_spectrum.py --log LOG
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Larger type so labels stay legible when the figure is scaled to a text column.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COL_D = "#B03A2E"   # trained records EMR(D)
COL_C = "#1F6FB2"   # never-trained controls EMR(C) = forcing floor

# the final run, pooled, EMR in %. Ordered by expressivity (left = least).
PROBES = [
    ("Fixed prompt",     0.0,   0.0),
    ("PII-Scope",        0.0,   0.0),
    ("PII-Compass",      0.0,   0.0),
    ("Random restart",   0.0,   0.0),
    ("GCG-fluent",       39.1,  42.0),
    ("GCG-free",         50.7,  39.1),
    ("GCG-anchored",     71.0,  62.3),
    ("Soft prompt",     100.0, 100.0),
]


def make_figure(rows, out_stem):
    labels = [r[0] for r in rows]
    emr_d = np.array([r[1] for r in rows])
    emr_c = np.array([r[2] for r in rows])
    x = np.arange(len(rows))
    w = 0.40

    fig, ax = plt.subplots(figsize=(3.6, 3.4))

    ax.bar(x - w/2, emr_d, w, color=COL_D, label=r"trained $\mathrm{EMR}(D)$",
           edgecolor="white", linewidth=0.6, zorder=3)
    ax.bar(x + w/2, emr_c, w, color=COL_C,
           label=r"control $\mathrm{EMR}(C)$", edgecolor="white",
           linewidth=0.6, zorder=3)

    # region shading: non-optimization vs optimization
    ax.axvspan(-0.6, 3.5, color="#000000", alpha=0.045, zorder=0)
    ax.text(1.5, 107, "no optimization", fontsize=9.5, color="#555555",
            ha="center", va="bottom")
    ax.text(5.75, 107, "optimization", fontsize=9.5, color="#555555",
            ha="center", va="bottom")

    # the forcing-floor climb
    ax.annotate("", xy=(7.0, 90), xytext=(4.1, 30),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=COL_C, alpha=0.8,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text(3.4, 78, "forcing floor\nrises with\nexpressivity", fontsize=9.5,
            color=COL_C, ha="right", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("exact-match rate (%)")
    ax.grid(True, axis="y", lw=0.5, alpha=0.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    leg = ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.44), frameon=False,
                    handlelength=1.3, labelspacing=0.4, borderpad=0.2,
                    handletextpad=0.5)
    leg.set_zorder(6)

    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, dpi=400 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.015)
        print("wrote", p)
    plt.close(fig)


def load_rows(log_path):
    import pandas as pd
    df = (pd.read_parquet(log_path) if log_path.endswith(".parquet")
          else pd.read_csv(log_path))
    order = ["fixed", "piiscope", "piicompass", "random_restart",
             "gcg_fluent", "gcg_free", "gcg_anchored", "softprompt"]
    pretty = {"fixed": "Fixed\nprompt", "piiscope": "PII-\nScope",
              "piicompass": "PII-\nCompass", "random_restart": "Random\nrestart",
              "gcg_fluent": "GCG\nfluent", "gcg_free": "GCG\nfree",
              "gcg_anchored": "GCG\nanchored", "softprompt": "Soft\nprompt"}
    rows = []
    for pr in order:
        sub = df[(df["probe"] == pr) & (df["model_state"] == "finetuned")]
        d = 100 * sub[sub["target_membership"] == "trained"]["exact_match"].mean()
        c = 100 * sub[sub["target_membership"] == "control"]["exact_match"].mean()
        rows.append((pretty[pr], float(d), float(c)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__) or ".",
                                                  "fig_spectrum"))
    a = ap.parse_args()
    rows = load_rows(a.log) if a.log else PROBES
    make_figure(rows, a.out)


if __name__ == "__main__":
    main()
