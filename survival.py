"""
Interval- and right-censored fit of k_min against target entropy.

Why this module exists rather than a `linregress` call
------------------------------------------------------
`capacity_e3` currently does `fit = km[np.isfinite(km["k_min"])]` before
regressing -- silently dropping every target the attack never forced. Those are
exactly the highest-entropy, hardest targets, so the survivors are a
biased-easy subsample and beta is biased with them. That is a selection-bias
defect in the analysis, not a modelling choice.

k_min is censored twice at once:
  * RIGHT   -- never forced by k=64: the observation is "k_min > 64", not missing.
  * INTERVAL-- the grid jumps 4->6->8->12: a target first seen at k=6 has
               k_min in (4, 6], not = 6.

Specification (H4)
------------------
The theory is `k_min = H / beta`, proportional and through the origin. Fitting
`log k_min ~ H` at level would test nothing: it means `k = e^b0 * e^(b1 H)`,
exponential rather than proportional, and b0 = 0 gives k = 1 at H = 0, not 0.
The log-LOG form maps the theory cleanly:

    log k_min = -log(beta) + gamma * log(H)

  * proportionality  <=>  gamma == 1     <- this is H4, a SLOPE test
  * beta = exp(-intercept)               <- a scale parameter, not the test

Fit on the CONTROL arm alone: Proposition 1's forcing model is about pure
forcing, so pooling trained targets would let memorisation appear as a level
shift indistinguishable from a proportionality violation.

Uncertainty comes from a person-clustered bootstrap, never from lifelines'
model-based sandwich SEs -- only ~25 clusters are available, far below the
40-50 that cluster-robust asymptotics want.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

_LIFELINES_HINT = (
    "lifelines is required for the censored k_min fit and is pinned in "
    "requirements.txt (lifelines==0.30.0). Install it and re-run.\n"
    "Do NOT fall back to a complete-case linregress: dropping right-censored "
    "targets removes the hardest ones and biases beta. The design states that "
    "estimator must never be the headline."
)


def _require_lifelines():
    """Hard-fail. A silent fallback here would publish a biased beta."""
    try:
        import lifelines  # noqa: F401
    except ImportError as e:
        raise ImportError(_LIFELINES_HINT) from e
    return lifelines


def censoring_bounds(k_min: Sequence[float], grid: Sequence[int]) -> Dict[str, np.ndarray]:
    """
    Turn observed first-hit grid points into (lower, upper) censoring intervals.

    A hit first seen at grid point g means the true threshold lies in
    (previous grid point, g]. A NaN means never forced: (max(grid), inf).
    """
    g = sorted(int(x) for x in grid if x > 0)
    prev = {v: (g[i - 1] if i else 0) for i, v in enumerate(g)}
    lo, hi = [], []
    for v in k_min:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            lo.append(float(g[-1])); hi.append(np.inf)       # right-censored
        else:
            v = int(v)
            lo.append(float(max(prev.get(v, 0), 1e-9)))      # interval-censored
            hi.append(float(v))
    return {"lower": np.asarray(lo, float), "upper": np.asarray(hi, float)}


def right_censored_fraction(k_min: Sequence[float]) -> float:
    """Above ~0.6-0.7 the likelihood for gamma is fragile at this n -- the
    protocol's pre-committed fallback is a Turnbull summary, not a point estimate."""
    a = np.asarray([np.nan if v is None else v for v in k_min], float)
    return float(np.isnan(a).mean()) if a.size else float("nan")


def non_monotone_fraction(hits_by_k: Dict[str, Dict[int, bool]]) -> float:
    """
    Fraction of targets that were forced at some k and NOT forced at a larger k.

    An AFT threshold model treats forcing as absorbing -- once crossed, always
    crossed. `_kmin_table` takes min-over-hits and GCG is stochastic, so that
    assumption is violated to an unknown degree. This number IS the measure of
    how far, and it must be reported beside beta.
    """
    bad = 0
    for _, by_k in hits_by_k.items():
        ks = sorted(by_k)
        first = next((k for k in ks if by_k[k]), None)
        if first is not None and any(not by_k[k] for k in ks if k > first):
            bad += 1
    return bad / len(hits_by_k) if hits_by_k else float("nan")


def fit_loglog(k_min: Sequence[float], h_bits: Sequence[float],
               person_id: Sequence[str], grid: Sequence[int],
               n_boot: int = 2000, seed: int = 20240601) -> Dict:
    """
    Fit `log k_min = -log(beta) + gamma * log(H)` with interval/right censoring,
    person-clustered bootstrap CIs on gamma and beta.

    H4 is refuted when gamma's CI excludes 1.
    """
    ll = _require_lifelines()
    from lifelines import WeibullAFTFitter
    import pandas as pd

    h = np.asarray(h_bits, float)
    if np.any(h <= 0):
        raise ValueError("H(t) must be positive to take log(H).")
    b = censoring_bounds(k_min, grid)
    df = pd.DataFrame({"lower": b["lower"], "upper": b["upper"],
                       "log_h": np.log(h), "person": list(person_id)})

    def _one(frame):
        f = WeibullAFTFitter()
        f.fit_interval_censoring(frame[["lower", "upper", "log_h"]],
                                 lower_bound_col="lower", upper_bound_col="upper")
        par = f.params_.xs("lambda_")
        return float(par["log_h"]), float(par["Intercept"])

    gamma, intercept = _one(df)
    rng = np.random.default_rng(seed)
    persons = df["person"].unique()
    idx = {p: np.flatnonzero(df["person"].to_numpy() == p) for p in persons}
    gs, bs, failures = [], [], {}
    for _ in range(n_boot):
        draw = rng.choice(persons, size=len(persons), replace=True)
        rows = np.concatenate([idx[p] for p in draw])
        try:
            # reset_index is REQUIRED, not cosmetic: a clustered resample draws
            # the same person repeatedly, so df.iloc[rows] carries duplicate
            # index labels and lifelines' internal reindex raises on them. Left
            # in, it kills every replicate and the failure reads as an
            # unidentifiable model rather than as a pandas indexing detail.
            g_, i_ = _one(df.iloc[rows].reset_index(drop=True))
            gs.append(g_); bs.append(np.exp(-i_))
        except Exception as e:
            # A degenerate resample is not a result -- but swallowing the reason
            # is how a fit that failed 2000/2000 times gets mistaken for a wide
            # interval. Keep a census of causes so the failure is diagnosable.
            key = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            failures[key] = failures.get(key, 0) + 1

    def _ci(a):
        a = np.asarray(a, float)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) \
            if a.size else (float("nan"), float("nan"))

    g_lo, g_hi = _ci(gs)
    return {
        "gamma": gamma, "gamma_ci": (g_lo, g_hi),
        "beta_bits_per_token": float(np.exp(-intercept)), "beta_ci": _ci(bs),
        "h4_proportionality_supported": bool(g_lo <= 1.0 <= g_hi),
        "right_censored_fraction": right_censored_fraction(k_min),
        "n_targets": int(len(df)), "n_persons": int(len(persons)),
        "n_boot_ok": len(gs),
        "n_boot_failed": int(sum(failures.values())),
        "failure_census": dict(sorted(failures.items(), key=lambda kv: -kv[1])[:5]),
        "log_h_spread": float(np.log(h).max() - np.log(h).min()),
        "identifiable": bool(len(gs) >= 0.5 * n_boot),
        "lifelines_version": ll.__version__,
    }
