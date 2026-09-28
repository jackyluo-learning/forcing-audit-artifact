"""One interval family for the whole paper, plus the power analysis.

The v2 run used four different interval methods with no per-table attribution,
and its zero cells quoted a Wilson bound at an "effective n" of 33.3 that
corresponds to no actual sample size in the study (true Wilson at n=150 is 2.50%,
at n=25 persons it is 13.3%). Mixing a conservative convention at the boundary
with a permissive one everywhere else reads as cherry-picking, so this module
fixes one convention and applies it everywhere:

  * the unit of analysis is the PERSON, because two fields from one person and
    three optimizer restarts of one target are not independent observations;
  * rates and differences get a person-level bootstrap, paired across arms when
    the design is matched;
  * zero and one cells additionally get an exact Clopper-Pearson bound at the
    person level, which is defensible and, for a zero cell, stronger for the
    argument than an unexplained design effect.

Everything here is pure numpy/scipy so it runs on a login node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# Closed-form intervals
# --------------------------------------------------------------------------- #
def wilson(successes: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval. At 0/n the upper limit is z^2/(n+z^2)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson(successes: int, n: int, alpha: float = 0.05
                    ) -> Tuple[float, float]:
    """Exact binomial interval; the right tool for a 0/n or n/n cell."""
    from scipy.stats import beta
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes,
                                             n - successes + 1)
    hi = 1.0 if successes == n else beta.ppf(1 - alpha / 2, successes + 1,
                                             n - successes)
    return float(lo), float(hi)


def newcombe_diff(s1: int, n1: int, s2: int, n2: int, z: float = Z95
                  ) -> Tuple[float, float]:
    """Newcombe's MOVER interval for p1 - p2 from two Wilson intervals.

    Use this only for INDEPENDENT arms. Our design matches persons across arms,
    so prefer `paired_bootstrap_diff`; Newcombe is reported for the unmatched
    per-model table only.
    """
    l1, u1 = wilson(s1, n1, z)
    l2, u2 = wilson(s2, n2, z)
    p1, p2 = s1 / n1, s2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return max(-1.0, lo), min(1.0, hi)


# --------------------------------------------------------------------------- #
# Person-clustered bootstrap, the paper's default
# --------------------------------------------------------------------------- #
@dataclass
class Estimate:
    point: float
    lo: float
    hi: float
    n_units: int
    n_obs: int
    method: str

    def pct(self, digits: int = 1) -> str:
        return (f"{100*self.point:.{digits}f} "
                f"[{100*self.lo:.{digits}f}, {100*self.hi:.{digits}f}]")

    def as_row(self) -> Dict[str, float]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi,
                "n_units": self.n_units, "n_obs": self.n_obs,
                "method": self.method}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def cluster_bootstrap_rate(units: Sequence[Sequence[int]], n_boot: int = 10000,
                           seed: int = 0, alpha: float = 0.05) -> Estimate:
    """Rate with a person-clustered bootstrap.

    `units` is one sequence of 0/1 outcomes per person (all fields x restarts of
    that person). Persons are resampled with replacement; observations within a
    person travel together, which is what "clustered" has to mean.

    For an all-zero or all-one sample the bootstrap is degenerate (every
    replicate is identical), so we fall back to a person-level Clopper-Pearson
    bound and say so in `method`. That is the honest answer: 0/150 attempts over
    25 persons does not license a 2.5% upper bound.
    """
    obs = [np.asarray(u, dtype=float) for u in units if len(u)]
    if not obs:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, 0, "empty")
    flat = np.concatenate(obs)
    point = float(flat.mean())
    n_units, n_obs = len(obs), int(flat.size)

    if point in (0.0, 1.0):
        succ_units = sum(1 for u in obs if np.any(u > 0))
        lo, hi = clopper_pearson(succ_units, n_units, alpha)
        return Estimate(point, lo, hi, n_units, n_obs,
                        "person-level Clopper-Pearson (boundary cell)")

    r = _rng(seed)
    means = np.empty(n_boot)
    idx_pool = np.arange(n_units)
    for b in range(n_boot):
        pick = r.choice(idx_pool, size=n_units, replace=True)
        means[b] = np.concatenate([obs[i] for i in pick]).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(point, float(lo), float(hi), n_units, n_obs,
                    "person-clustered bootstrap")


def paired_bootstrap_diff(trained: Dict, control: Dict, n_boot: int = 10000,
                          seed: int = 0, alpha: float = 0.05) -> Estimate:
    """Difference of rates with a PAIRED person bootstrap.

    `trained` and `control` map person_id -> sequence of 0/1. A matched design
    pairs a trained person with a control person, so resampling must draw the
    pair, not each arm independently; doing it independently inflates the
    interval and loses the design's only advantage.

    Persons present in only one arm are dropped, and the count is reported in
    `method` so the loss is visible.
    """
    keys_t, keys_c = list(trained), list(control)
    pairs = list(zip(sorted(keys_t), sorted(keys_c)))   # positional matching
    dropped = abs(len(keys_t) - len(keys_c))
    if not pairs:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, 0, "empty")

    t_obs = [np.asarray(trained[a], dtype=float) for a, _ in pairs]
    c_obs = [np.asarray(control[b], dtype=float) for _, b in pairs]
    point = (float(np.concatenate(t_obs).mean())
             - float(np.concatenate(c_obs).mean()))

    r = _rng(seed)
    diffs = np.empty(n_boot)
    idx = np.arange(len(pairs))
    for b in range(n_boot):
        pick = r.choice(idx, size=len(pairs), replace=True)
        diffs[b] = (np.concatenate([t_obs[i] for i in pick]).mean()
                    - np.concatenate([c_obs[i] for i in pick]).mean())
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    note = f"paired person bootstrap over {len(pairs)} pairs"
    if dropped:
        note += f" ({dropped} unmatched person(s) dropped)"
    return Estimate(point, float(lo), float(hi), len(pairs),
                    int(np.concatenate(t_obs).size + np.concatenate(c_obs).size),
                    note)


def design_effect(units: Sequence[Sequence[int]]) -> float:
    """Ratio of the clustered variance to the iid variance of the same data.

    Report this instead of quoting an unexplained effective n. A value near 1
    means restarts and fields within a person behave independently, which is
    itself a finding worth stating (it would mean elicitability is a per-run
    event rather than a property of the target).
    """
    obs = [np.asarray(u, dtype=float) for u in units if len(u)]
    if not obs:
        return float("nan")
    flat = np.concatenate(obs)
    p, n = flat.mean(), flat.size
    if p in (0.0, 1.0):
        return float("nan")
    var_iid = p * (1 - p) / n
    m = len(obs)
    cluster_means = np.array([u.mean() for u in obs])
    sizes = np.array([u.size for u in obs], dtype=float)
    w = sizes / sizes.sum()
    var_cluster = float(np.sum(w ** 2 * np.var(cluster_means, ddof=0))) \
        if m > 1 else float("nan")
    var_cluster = float(np.var(cluster_means, ddof=1) / m) if m > 1 else float("nan")
    return var_cluster / var_iid if var_iid > 0 else float("nan")


def icc(units: Sequence[Sequence[int]]) -> float:
    """Intraclass correlation of outcomes within a person (one-way ANOVA)."""
    obs = [np.asarray(u, dtype=float) for u in units if len(u) > 1]
    if len(obs) < 2:
        return float("nan")
    k = np.mean([u.size for u in obs])
    grand = np.concatenate(obs).mean()
    msb = np.sum([u.size * (u.mean() - grand) ** 2 for u in obs]) / (len(obs) - 1)
    msw_num = np.sum([np.sum((u - u.mean()) ** 2) for u in obs])
    msw_den = np.sum([u.size - 1 for u in obs])
    if msw_den == 0:
        return float("nan")
    msw = msw_num / msw_den
    denom = msb + (k - 1) * msw
    return float((msb - msw) / denom) if denom else float("nan")


def mcnemar(b: int, c: int, exact: bool = True) -> Dict[str, float]:
    """Paired test on discordant pairs: b = trained-only hits, c = control-only."""
    from scipy.stats import binomtest, chi2
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "stat": 0.0, "test": "no discordant pairs"}
    if exact or n < 25:
        p = binomtest(b, n, 0.5).pvalue
        return {"b": b, "c": c, "p": float(p), "stat": float(b),
                "test": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "p": float(chi2.sf(stat, 1)), "stat": float(stat),
            "test": "chi2 with continuity correction"}


# --------------------------------------------------------------------------- #
# The recall fraction and the power analysis (Prop. 3 in the paper)
# --------------------------------------------------------------------------- #
def recall_fraction(pi: float, alpha: float) -> float:
    """m = (pi - alpha) / (1 - alpha).

    The share of targets recovered because of inclusion, among those the attack
    does not force. Unlike pi - alpha this is not mechanically deflated as the
    floor rises, which is why it is the quantity to report across capacities.
    """
    if alpha >= 1.0:
        return float("nan")
    return (pi - alpha) / (1.0 - alpha)


def recall_fraction_se(pi: float, alpha: float, n_d: int, n_c: int) -> float:
    """Delta-method SE of the plug-in recall fraction."""
    if alpha >= 1.0 or n_d == 0 or n_c == 0:
        return float("nan")
    m = recall_fraction(pi, alpha)
    var = (pi * (1 - pi) / n_d + (1 - m) ** 2 * alpha * (1 - alpha) / n_c) \
        / (1 - alpha) ** 2
    return math.sqrt(max(var, 0.0))


def _z_sum(alpha_level: float = 0.05, power: float = 0.80) -> float:
    from scipy.stats import norm
    return norm.ppf(1 - alpha_level / 2) + norm.ppf(power)


def power_n(alpha: float, m: float, alpha_level: float = 0.05,
            power: float = 0.80) -> float:
    """Per-arm sample size needed to detect recall fraction `m` at floor `alpha`.

    n(alpha,m) = z^2 (1-m)/m  +  z^2 (2-m)/m^2 * alpha/(1-alpha)

    The second term is the cost of the floor and is proportional to the ODDS of
    being forced, which is what makes a high-capacity audit so expensive.
    """
    if not 0 < m <= 1 or alpha >= 1:
        return float("inf")
    z = _z_sum(alpha_level, power)
    return z * z * (1 - m) / m + z * z * (2 - m) / m ** 2 * alpha / (1 - alpha)


def min_detectable_m(alpha: float, n: int, alpha_level: float = 0.05,
                     power: float = 0.80) -> float:
    """Smallest recall fraction detectable with n targets per arm at this floor."""
    if alpha >= 1 or n <= 0:
        return float("nan")
    z = _z_sum(alpha_level, power)
    lo, hi = 1e-9, 1.0 - alpha
    for _ in range(200):                       # bisect on the advantage delta
        mid = (lo + hi) / 2
        rhs = z * math.sqrt((alpha * (1 - alpha)
                             + (alpha + mid) * (1 - alpha - mid)) / n)
        if mid < rhs:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 / (1 - alpha)


def power_table(alphas: Dict[str, float], n: int = 150, m_target: float = 0.10
                ) -> Dict[str, Dict[str, float]]:
    """The paper's Table: advantage ceiling, detectable m, and required n."""
    out = {}
    for label, a in alphas.items():
        out[label] = {
            "alpha": a,
            "advantage_ceiling": 1.0 - a,
            "min_detectable_m_at_n": min_detectable_m(a, n),
            "n_for_m_target": power_n(a, m_target),
        }
    return out


# --------------------------------------------------------------------------- #
# AUC with a person-clustered interval
# --------------------------------------------------------------------------- #
def auc_with_ci(scores_pos: Sequence[float], scores_neg: Sequence[float],
                groups_pos: Optional[Sequence] = None,
                groups_neg: Optional[Sequence] = None,
                n_boot: int = 5000, seed: int = 0, alpha: float = 0.05
                ) -> Estimate:
    """AUC of a continuous score, with a bootstrap interval.

    The v2 paper reported four point AUCs and then made two quantitative claims
    about their intervals ("all span 0.5", "GPT-2-M reaches ~0.9") with no
    interval anywhere. This returns one.
    """
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, 0, "empty")

    def _auc(a: np.ndarray, b: np.ndarray) -> float:
        from scipy.stats import rankdata
        allv = np.concatenate([a, b])
        r = rankdata(allv)
        r_pos = r[:a.size].sum()
        return float((r_pos - a.size * (a.size + 1) / 2) / (a.size * b.size))

    point = _auc(pos, neg)
    gp = np.asarray(groups_pos) if groups_pos is not None else np.arange(pos.size)
    gn = np.asarray(groups_neg) if groups_neg is not None else np.arange(neg.size)
    up, un = np.unique(gp), np.unique(gn)
    r = _rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        sp = np.concatenate([pos[gp == g] for g in r.choice(up, up.size, True)])
        sn = np.concatenate([neg[gn == g] for g in r.choice(un, un.size, True)])
        vals[b] = _auc(sp, sn) if sp.size and sn.size else np.nan
    vals = vals[~np.isnan(vals)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return Estimate(point, float(lo), float(hi), int(up.size + un.size),
                    int(pos.size + neg.size), "cluster bootstrap AUC")


def tpr_at_fpr(scores_pos: Sequence[float], scores_neg: Sequence[float],
               target_fpr: float = 0.01) -> Dict[str, float]:
    """TPR at a fixed low FPR, the membership-inference reporting convention.

    Also returns the achievable FPR, because with n control targets in the low
    hundreds no threshold realises 1% and reporting TPR@1% anyway is misleading.
    """
    pos = np.sort(np.asarray(scores_pos, dtype=float))[::-1]
    neg = np.sort(np.asarray(scores_neg, dtype=float))[::-1]
    if neg.size == 0 or pos.size == 0:
        return {"tpr": float("nan"), "realised_fpr": float("nan"),
                "threshold": float("nan"), "min_achievable_fpr": float("nan")}
    j = int(math.floor(target_fpr * neg.size))
    min_fpr = 1.0 / neg.size
    if j < 1:
        return {"tpr": float("nan"), "realised_fpr": float("nan"),
                "threshold": float("nan"), "min_achievable_fpr": min_fpr}
    thr = neg[j - 1]
    return {"tpr": float((pos >= thr).mean()), "realised_fpr": float(j / neg.size),
            "threshold": float(thr), "min_achievable_fpr": min_fpr}
