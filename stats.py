"""
Shared, honest statistics for the extraction study.

Replaces the previous degenerate significance test (which hard-coded p=0.0
whenever the seed-level std was 0) with:
  - McNemar's paired test on per-(person, field) baseline-vs-GCG outcomes
  - Bootstrap confidence intervals for proportions, paired differences, ratios
  - Wilson score intervals for reporting per-cell rates
  - Pearson correlation (frequency vs extractability) with a real p-value
  - Two-way ANOVA for the frequency x method interaction (statsmodels if present,
    otherwise a manual unbalanced-safe fallback)

All functions operate on plain numpy arrays / lists so they can be reused by
evaluate.py and linguistic_analysis.py without importing model code.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

_RNG_SEED = 20240601  # fixed so bootstrap CIs are reproducible across runs


# ---------------------------------------------------------------------------
# Paired test: McNemar
# ---------------------------------------------------------------------------

def mcnemar_test(
    baseline_success: Sequence[int],
    gcg_success: Sequence[int],
) -> Dict:
    """
    McNemar's test on paired binary outcomes (same targets attacked by both
    methods). b = baseline-only successes, c = gcg-only successes.

    Uses the exact binomial test when discordant pairs are few (b + c < 25),
    otherwise the chi-square approximation with continuity correction.
    """
    a = np.asarray(baseline_success).astype(int)
    g = np.asarray(gcg_success).astype(int)
    if a.shape != g.shape:
        raise ValueError("paired outcome arrays must have equal length")

    b = int(np.sum((a == 1) & (g == 0)))  # baseline succeeds, gcg fails
    c = int(np.sum((a == 0) & (g == 1)))  # gcg succeeds, baseline fails
    n_disc = b + c

    if n_disc == 0:
        return {
            "b": b, "c": c, "n_discordant": 0,
            "statistic": 0.0, "p_value": 1.0, "method": "no_discordant_pairs",
        }

    if n_disc < 25:
        # Exact two-sided binomial test with p=0.5
        p = sps.binomtest(min(b, c), n=n_disc, p=0.5, alternative="two-sided").pvalue
        stat = float(min(b, c))
        method = "exact_binomial"
    else:
        stat = (abs(b - c) - 1.0) ** 2 / n_disc  # continuity-corrected chi-square
        p = float(sps.chi2.sf(stat, df=1))
        method = "chi2_continuity"

    return {
        "b": b, "c": c, "n_discordant": n_disc,
        "statistic": float(stat), "p_value": float(p), "method": method,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _percentile_ci(samples: np.ndarray, ci: float) -> Tuple[float, float]:
    lo = (1.0 - ci) / 2.0 * 100.0
    hi = (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def bootstrap_ci_mean(
    values: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> Dict:
    """Bootstrap CI for the mean (or proportion) of a 1-D sample."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = np.random.default_rng(_RNG_SEED)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = v[idx].mean(axis=1)
    lo, hi = _percentile_ci(boot_means, ci)
    return {"mean": float(v.mean()), "ci_low": lo, "ci_high": hi, "n": n}


def bootstrap_paired(
    baseline: Sequence[int],
    gcg: Sequence[int],
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> Dict:
    """
    Bootstrap CIs for the paired difference (gcg - baseline) and ratio
    (gcg / baseline) of success rates, resampling targets (rows) with
    replacement to respect the pairing.
    """
    a = np.asarray(baseline, dtype=float)
    g = np.asarray(gcg, dtype=float)
    n = len(a)
    if n == 0:
        return {"baseline_rate": 0.0, "gcg_rate": 0.0, "diff": 0.0, "ratio": 0.0}
    rng = np.random.default_rng(_RNG_SEED)
    idx = rng.integers(0, n, size=(n_boot, n))
    a_boot = a[idx].mean(axis=1)
    g_boot = g[idx].mean(axis=1)
    diff = g_boot - a_boot
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a_boot > 0, g_boot / a_boot, np.nan)
    diff_lo, diff_hi = _percentile_ci(diff, ci)
    ratio_valid = ratio[~np.isnan(ratio)]
    if len(ratio_valid) > 0:
        ratio_lo, ratio_hi = _percentile_ci(ratio_valid, ci)
        ratio_point = float(g.mean() / a.mean()) if a.mean() > 0 else float("inf")
    else:
        ratio_lo = ratio_hi = ratio_point = float("inf")
    return {
        "baseline_rate": float(a.mean()),
        "gcg_rate": float(g.mean()),
        "diff": float(g.mean() - a.mean()),
        "diff_ci_low": diff_lo,
        "diff_ci_high": diff_hi,
        "ratio": ratio_point,
        "ratio_ci_low": ratio_lo,
        "ratio_ci_high": ratio_hi,
        "n": n,
    }


def wilson_ci(k: int, n: int, ci: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = float(sps.norm.ppf((1.0 + ci) / 2.0))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return float(center - half), float(center + half)


# ---------------------------------------------------------------------------
# Correlation: frequency vs extractability
# ---------------------------------------------------------------------------

def pearson_corr(x: Sequence[float], y: Sequence[float]) -> Dict:
    """Pearson r with a real two-sided p-value (replaces the ungrounded r=0.87)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": 0.0, "p_value": 1.0, "n": int(len(x)), "note": "degenerate"}
    r, p = sps.pearsonr(x, y)
    return {"r": float(r), "p_value": float(p), "n": int(len(x))}


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> Dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return {"rho": 0.0, "p_value": 1.0, "n": int(len(x))}
    rho, p = sps.spearmanr(x, y)
    return {"rho": float(rho), "p_value": float(p), "n": int(len(x))}


# ---------------------------------------------------------------------------
# Two-way ANOVA: frequency x method interaction
# ---------------------------------------------------------------------------

def two_way_anova(
    outcome: Sequence[float],
    factor_freq: Sequence,
    factor_method: Sequence,
) -> Dict:
    """
    Two-way ANOVA of extraction outcome on frequency, method, and their
    interaction. Prefers statsmodels (Type-II SS, unbalanced-safe); falls back
    to a manual balanced-design computation if statsmodels is unavailable.
    """
    outcome = np.asarray(outcome, dtype=float)
    freq = np.asarray(factor_freq)
    method = np.asarray(factor_method)

    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        df = pd.DataFrame({
            "y": outcome,
            "freq": pd.Categorical(freq),
            "method": pd.Categorical(method),
        })
        model = smf.ols("y ~ C(freq) * C(method)", data=df).fit()
        table = anova_lm(model, typ=2)
        out = {"backend": "statsmodels", "terms": {}}
        for term in table.index:
            if term == "Residual":
                continue
            out["terms"][term] = {
                "F": float(table.loc[term, "F"]),
                "p_value": float(table.loc[term, "PR(>F)"]),
                "df": float(table.loc[term, "df"]),
            }
        inter = [k for k in out["terms"] if ":" in k]
        if inter:
            out["interaction_F"] = out["terms"][inter[0]]["F"]
            out["interaction_p"] = out["terms"][inter[0]]["p_value"]
        return out
    except Exception as e:  # statsmodels/pandas missing or fit failed
        return _manual_two_way_anova(outcome, freq, method, note=str(e))


def _manual_two_way_anova(outcome, freq, method, note="") -> Dict:
    """Manual two-way ANOVA (interaction F). Assumes each cell is non-empty."""
    grand = outcome.mean()
    ss_total = float(np.sum((outcome - grand) ** 2))

    freq_levels = np.unique(freq)
    method_levels = np.unique(method)

    def _factor_ss(fac, levels):
        ss = 0.0
        for lv in levels:
            mask = fac == lv
            if mask.sum() == 0:
                continue
            ss += mask.sum() * (outcome[mask].mean() - grand) ** 2
        return float(ss)

    ss_freq = _factor_ss(freq, freq_levels)
    ss_method = _factor_ss(method, method_levels)

    ss_cells = 0.0
    n_cells = 0
    for fl in freq_levels:
        for ml in method_levels:
            mask = (freq == fl) & (method == ml)
            if mask.sum() == 0:
                continue
            n_cells += 1
            ss_cells += mask.sum() * (outcome[mask].mean() - grand) ** 2
    ss_inter = float(ss_cells - ss_freq - ss_method)
    ss_error = float(ss_total - ss_cells)

    n = len(outcome)
    df_freq = len(freq_levels) - 1
    df_method = len(method_levels) - 1
    df_inter = df_freq * df_method
    df_error = n - n_cells
    if df_error <= 0 or ss_error <= 0:
        return {"backend": "manual", "interaction_F": float("nan"),
                "interaction_p": float("nan"), "note": f"insufficient df; {note}"}

    ms_error = ss_error / df_error
    f_inter = (ss_inter / max(df_inter, 1)) / ms_error
    p_inter = float(sps.f.sf(f_inter, df_inter, df_error))
    return {
        "backend": "manual",
        "interaction_F": float(f_inter),
        "interaction_p": p_inter,
        "interaction_df": (df_inter, df_error),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Convenience: summarize a proportion with Wilson CI as a percentage string
# ---------------------------------------------------------------------------

def rate_with_ci(k: int, n: int, ci: float = 0.95) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson_ci(k, n, ci)
    return f"{100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
