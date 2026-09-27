#!/usr/bin/env python3
"""
Idealized exact-output capacity bound.

The dashed curve is the Proposition 1 upper bound on exact-output reachable mass,
rho_k^= <= 2^(k*log2|V| - H_inf), for a uniform 9-digit SSN against GPT-2's
vocabulary. Corollary 2 inverts it: among integer prompt lengths, a 1% bound is
guaranteed only at k=1. The default k=20 lies where the bound is vacuous.
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
    "font.size": 11.5, "axes.labelsize": 14.0, "legend.fontsize": 10.5,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COL_THEORY = "#555555"

VOCAB = 50257
LOG2V = np.log2(VOCAB)         # ~15.62 bits/token
H_INF = np.log2(1e9)           # uniform 9-digit SSN, ~29.90 bits
ALPHA = 0.01
FLOOR = 1e-5


def k_star_theory(alpha=ALPHA, h=H_INF, v=LOG2V):
    return (h - np.log2(1.0 / alpha)) / v


def theory_bound(k, h=H_INF, v=LOG2V):
    expo = np.clip(k * v - h, -60.0, 1.0)
    return np.minimum(1.0, np.exp2(expo))


def make_figure(out_stem):
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(3.55, 3.05))

    kk = np.logspace(np.log10(0.9), np.log10(70), 600)
    k_vac = H_INF / LOG2V

    ax.axvspan(k_vac, 70, color="#000000", alpha=0.05, lw=0, zorder=0)
    ax.text(6, 0.08, "bound vacuous:\nno low-floor guarantee",
            fontsize=8.7, color=COL_THEORY, ha="center", va="center")

    ax.plot(kk, np.maximum(theory_bound(kk), FLOOR), color=COL_THEORY,
            ls=(0, (5, 2)), lw=1.8, zorder=2)

    ax.axhline(ALPHA, color="black", lw=0.8, ls=":", zorder=3)
    ax.text(2.4, 0.016, r"tolerance $a=1\%$",
            fontsize=8.5, color="#333333", ha="left", va="bottom")

    kthy = k_star_theory()
    ax.vlines(kthy, FLOOR, ALPHA, color=COL_THEORY, lw=1.2, ls="-.", zorder=3)
    ax.annotate("1% guarantee boundary\n" + r"$k\approx1.5$",
                xy=(kthy, 0.006), xytext=(1.8, 0.0015),
                textcoords="data", fontsize=8.5, color=COL_THEORY,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=COL_THEORY, lw=0.8,
                                shrinkA=2, shrinkB=2))

    b1 = theory_bound(np.array([1.0]))[0]
    ax.scatter([1], [b1], s=18, color=COL_THEORY, zorder=5)
    ax.annotate(r"$B_{=}(1)\approx0.005\%$", xy=(1, b1),
                xytext=(1.08, 2.2e-4), textcoords="data",
                fontsize=8.5, color=COL_THEORY, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=COL_THEORY, lw=0.8,
                                shrinkA=2, shrinkB=2),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=0.4))

    ax.vlines(20, FLOOR, 1.0, color="black", lw=1.2, ls="--",
              alpha=0.75, zorder=3)
    ax.text(22, 0.3, r"default $k=20$",
            fontsize=8.5, color="#333333", ha="left", va="center")


    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.9, 70); ax.set_ylim(FLOOR, 1.7)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: {1e-5: "0.001%", 1e-4: "0.01%", 1e-3: "0.1%",
                      1e-2: "1%", 1e-1: "10%", 1.0: "100%"}.get(v, "")))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$B_{=}(k)$")
    ax.grid(True, which="major", lw=0.5, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    sec = ax.secondary_xaxis("top", functions=(lambda k: k * LOG2V,
                                               lambda b: b / LOG2V))
    sec.set_xlabel(r"$k\log_2|V|$", labelpad=3.0)
    sec.set_xticks([16, 64, 256, 1024])
    sec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    sec.spines["top"].set_linewidth(0.8)
    sec.tick_params(labelsize=10.8)

    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, dpi=400 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.015)
        print("wrote", p)
    plt.close(fig)
    print(f"k*_thy(1%)={kthy:.2f} | log2|V|={LOG2V:.2f} | H_inf(SSN)={H_INF:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__) or ".",
                                                  "fig_capacity"))
    a = ap.parse_args()
    make_figure(a.out)


if __name__ == "__main__":
    main()
