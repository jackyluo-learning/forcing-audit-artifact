"""
Curve statistics for the E3 capacity sweep.

Everything here exists because the sweep asks questions ABOUT A CURVE, and the
repo's existing helpers can only answer questions about one point at a time.

`_cluster_emr_ci` and `_cluster_diff_ci` each re-seed a fresh RNG on every call
and return percentiles, discarding the replicate vector. Called once per k they
appear to reuse the same person draw -- a coincidence of identical seeds and
identical array shapes, which breaks the moment one k has a different number of
person-groups, and which in any case leaves nothing behind. So there is no path
in the existing code that computes a Spearman rho, an argmax location, or a gap
between two capacities PER REPLICATE. Those are exactly H1 and H5.

The bootstrap here is TWO-BLOCK with repeated measures:

  * D-persons and C-persons are disjoint individuals, so they are drawn
    independently of one another;
  * within a block, ONE draw is applied across ALL capacities, because the same
    people are measured at every k. That is what makes the curve's SHAPE
    estimable rather than only its fourteen marginal points.

Every replicate's full k-vector is retained, so curve statistics are computed
inside the replicate and their percentile intervals are honest.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

N_BOOT = 10000
BOOT_SEED = 20240601
DEFF = 1.5          # 2 targets per person at ICC ~ 0.5; see the design's power section
Z95 = 1.959963985


# ---------------------------------------------------------------------------
# Person x capacity matrices — the structure the whole module rests on
# ---------------------------------------------------------------------------

def person_by_k(df, arm: str, ks: Sequence[int]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Hits and attempts for every person at every capacity.

    Returns (persons, S, N) with S[i, j] hits and N[i, j] attempts for person i
    at ks[j], summed over fields and seeds. A person absent at some k leaves a
    zero row entry there, which the micro-average handles by weight.

    Aligning persons to ROWS is what lets one index draw apply to every column.
    """
    sub = df[df["target_membership"] == arm]
    persons = sorted(sub["person_id"].unique())
    pidx = {p: i for i, p in enumerate(persons)}
    kidx = {k: j for j, k in enumerate(ks)}
    S = np.zeros((len(persons), len(ks)))
    N = np.zeros((len(persons), len(ks)))
    hit = sub["exact_match"].fillna(False).astype(bool).to_numpy()
    pi = sub["person_id"].map(pidx).to_numpy()
    kj = sub["capacity_k"].map(kidx).to_numpy()
    keep = ~np.isnan(kj.astype(float))
    np.add.at(S, (pi[keep], kj[keep].astype(int)), hit[keep].astype(float))
    np.add.at(N, (pi[keep], kj[keep].astype(int)), 1.0)
    return persons, S, N


def _micro(S: np.ndarray, N: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Micro-averaged rate per column over the resampled rows (matches _cluster_emr_ci)."""
    num = S[idx].sum(axis=0)
    den = N[idx].sum(axis=0)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def joint_bootstrap(df, ks: Sequence[int], n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> Dict:
    """
    One person draw per block per replicate, applied across every capacity.

    Returns point estimates plus the FULL replicate arrays, shape (n_boot, len(ks)):
    alpha (control), emr_d (trained) and tau (their difference). Curve statistics
    are then computed from those arrays rather than re-bootstrapped.
    """
    ks = list(ks)
    p_c, Sc, Nc = person_by_k(df, "control", ks)
    p_d, Sd, Nd = person_by_k(df, "trained", ks)
    if not p_c or not p_d:
        raise ValueError("one arm is empty; cannot bootstrap the curve")

    rng = np.random.default_rng(seed)
    nc, nd = len(p_c), len(p_d)
    # Independent draws: the two arms are disjoint people.
    idx_c = rng.integers(0, nc, size=(n_boot, nc))
    idx_d = rng.integers(0, nd, size=(n_boot, nd))

    alpha = np.empty((n_boot, len(ks)))
    emr_d = np.empty((n_boot, len(ks)))
    for b in range(n_boot):
        alpha[b] = _micro(Sc, Nc, idx_c[b])
        emr_d[b] = _micro(Sd, Nd, idx_d[b])
    return {
        "ks": np.asarray(ks, float),
        "alpha_point": _micro(Sc, Nc, np.arange(nc)),
        "emr_d_point": _micro(Sd, Nd, np.arange(nd)),
        "alpha": alpha, "emr_d": emr_d, "tau": emr_d - alpha,
        "n_persons": {"control": nc, "trained": nd},
        "n_attempts": {"control": Nc.sum(axis=0), "trained": Nd.sum(axis=0)},
        "n_boot": n_boot, "seed": seed,
    }


def pct_ci(rep: np.ndarray, ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.nanpercentile(rep, (1 - ci) / 2 * 100, axis=0)
    hi = np.nanpercentile(rep, (1 + ci) / 2 * 100, axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# H1 — monotone trend. Replaces `np.all(np.diff(alpha) >= -1e-9)`, which tests
# 13 point estimates with zero tolerance for sampling noise and therefore fails
# spuriously under almost any real data.
# ---------------------------------------------------------------------------

def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = _rank(x[m]), _rank(y[m])
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def trend_test(boot: Dict, ci: float = 0.95) -> Dict:
    """Spearman rho of (k, alpha_k), with a person-clustered bootstrap CI."""
    ks = boot["ks"]
    rho = spearman(ks, boot["alpha_point"])
    reps = np.array([spearman(ks, boot["alpha"][b]) for b in range(boot["n_boot"])])
    lo, hi = np.nanpercentile(reps, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"rho": rho, "ci": (float(lo), float(hi)),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot_ok": int(np.isfinite(reps).sum())}


def isotonic(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators — the monotone summary curve reported beside rho."""
    y = np.asarray(y, float).copy()
    w = np.ones_like(y)
    i = 0
    while i < len(y) - 1:
        if y[i] > y[i + 1] + 1e-12:
            tot = w[i] + w[i + 1]
            y[i] = (w[i] * y[i] + w[i + 1] * y[i + 1]) / tot
            w[i] = tot
            y = np.delete(y, i + 1); w = np.delete(w, i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    out, j = [], 0
    for v, ww in zip(y, w):
        out.extend([v] * int(round(ww)))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# H2 — the crossing, with the sentinel the existing _crossing_k lacks.
# ---------------------------------------------------------------------------

def crossing_k(ks: Sequence[float], alpha: Sequence[float], thr: float) -> Dict:
    """
    Smallest (interpolated) k at which alpha crosses `thr`, distinguishing the
    two cases the existing helper conflates.

    `_crossing_k` returns ks[0] both when the curve genuinely crosses at the
    smallest k AND when it is ALREADY above threshold there. Those are opposite
    findings for H2 -- "k=1 is the boundary" versus "no usable k exists" -- and a
    bare number cannot carry the difference.
    """
    ks = np.asarray(ks, float); a = np.asarray(alpha, float)
    if not np.isfinite(a).any():
        return {"k": float("nan"), "status": "no_data"}
    if a[0] >= thr:
        return {"k": float(ks[0]), "status": "already_above_at_smallest_k",
                "usable": False,
                "note": f"alpha={a[0]:.4f} at k={ks[0]:g} is already >= {thr}; "
                        f"no capacity on this grid has a floor below the tolerance"}
    for i in range(1, len(ks)):
        if a[i] >= thr:
            a0, a1, k0, k1 = a[i - 1], a[i], ks[i - 1], ks[i]
            k = k1 if a1 == a0 else k0 + (thr - a0) * (k1 - k0) / (a1 - a0)
            return {"k": float(k), "status": "crosses", "usable": True,
                    "bracket": (float(k0), float(k1))}
    return {"k": float("nan"), "status": "never_reaches_threshold", "usable": True,
            "note": f"alpha stays below {thr} across the whole grid; every k is usable "
                    f"on the floor criterion alone"}


def usable_points(boot: Dict, alpha_tol: float, ci: float = 0.95) -> Dict:
    """
    H2 as re-aimed: a point is usable only if the floor meets the tolerance AND
    the calibrated signal is detectable. A floor condition alone is satisfied
    wherever both are zero.
    """
    a_lo, a_hi = pct_ci(boot["alpha"], ci)
    t_lo, t_hi = pct_ci(boot["tau"], ci)
    rows = []
    for j, k in enumerate(boot["ks"]):
        floor_ok = bool(a_hi[j] <= alpha_tol)
        signal_ok = bool(t_lo[j] > 0 or t_hi[j] < 0)
        rows.append({"k": float(k), "alpha": float(boot["alpha_point"][j]),
                     "alpha_ci": (float(a_lo[j]), float(a_hi[j])),
                     "tau": float(boot["tau"].mean(axis=0)[j]),
                     "tau_ci": (float(t_lo[j]), float(t_hi[j])),
                     "floor_ok": floor_ok, "signal_ok": signal_ok,
                     "usable": floor_ok and signal_ok})
    any_usable = any(r["usable"] for r in rows)
    return {"alpha_tolerance": alpha_tol, "rows": rows, "any_usable": any_usable,
            "note": ("No capacity satisfies both conditions: this class of audit cannot be "
                     "calibrated to a usable dynamic range at this tolerance."
                     if not any_usable else
                     "At least one capacity meets both the floor and the signal condition.")}


# ---------------------------------------------------------------------------
# H5 — where the calibrated signal peaks. Exploratory at this n; the statistic
# is defined here so the reading is not an eyeball over a noisy curve.
# ---------------------------------------------------------------------------

def peak_location(boot: Dict, ci: float = 0.95) -> Dict:
    """Bootstrap distribution of argmax_k tau(k); interior peak iff the CI excludes both ends."""
    ks = boot["ks"]
    arg = ks[np.nanargmax(boot["tau"], axis=1)]
    lo, hi = np.percentile(arg, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"argmax_point": float(ks[int(np.nanargmax(boot["tau"].mean(axis=0)))]),
            "argmax_ci": (float(lo), float(hi)),
            "interior": bool(lo > ks[0] and hi < ks[-1]),
            "at_upper_end": bool(hi >= ks[-1]),
            "note": ("EXPLORATORY. SE(tau) per point is large at this n; a CI touching an "
                     "endpoint means the peak is not located by this data, not that it is absent.")}


# ---------------------------------------------------------------------------
# Degenerate arms — the normal case at small k, where an arm is 0/n and a
# percentile bootstrap silently returns [0, 0] rather than failing.
# ---------------------------------------------------------------------------

def is_degenerate(successes: float, n: float) -> bool:
    """Mechanical trigger, not a judgement call."""
    return n > 0 and (successes == 0 or successes == n)


def wilson_ci_eff(successes: float, n_raw: float, deff: float = DEFF,
                  z: float = Z95) -> Tuple[float, float]:
    """
    Wilson score interval on the CLUSTERING-ADJUSTED n.

    stats.wilson_ci takes a raw count. Passing the target count straight in
    would drop the design effect exactly where the study makes its most cautious
    claims -- zero-count control arms at small k -- and return an interval that
    is too narrow.
    """
    n = n_raw / deff
    if n <= 0:
        return (0.0, 0.0)
    p = (successes / deff) / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, (c - h) / d)), float(min(1.0, (c + h) / d)))


def newcombe_diff_ci(s1: float, n1: float, s2: float, n2: float,
                     deff: float = DEFF, z: float = Z95) -> Tuple[float, float]:
    """
    MOVER interval for p1 - p2 from two independent Wilson intervals.

    Used for tau_rec(k) when either arm is degenerate: the trained and control
    arms are disjoint people, so the independent form is the right one.
    """
    l1, u1 = wilson_ci_eff(s1, n1, deff, z)
    l2, u2 = wilson_ci_eff(s2, n2, deff, z)
    p1, p2 = (s1 / n1 if n1 else 0.0), (s2 / n2 if n2 else 0.0)
    lo = (p1 - p2) - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (float(max(-1.0, lo)), float(min(1.0, hi)))
