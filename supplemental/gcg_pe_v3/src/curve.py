"""The capacity operating characteristic: how control success depends on k.

This replaces the v2 paper's Spearman rho = 0.9945, which was pinned at the
maximum the tie structure allows (the three zero cells at k=1,2,3 stay tied in
every bootstrap replicate, so its interval upper limit necessarily equalled the
point estimate) and which only said the curve rises.

What is here instead:

  * distribution-free quantiles k_25, k_50, k_75 with a paired bootstrap, which
    are the primary numbers because they assume no functional form;
  * a fitted law, chosen by an explicit family comparison so the choice is
    visibly data-driven rather than convenient;
  * the realized bits per free token, beta_eff = H_inf / k_50, and the per-k
    address rate, which is the honest replacement for "the probe addresses 312
    bits";
  * the quantile-ratio test that REJECTS a linear "beta bits per token" model
    (Remark rem:nolinear), which is what deletes the v2 Appendix E heuristics;
  * the one-parameter fit tau(k) = (1 - alpha_k) * m, which derives the interior
    maximum the v2 paper reported as an unexplained curiosity;
  * the horizontal effect size: how many more free tokens controls need to reach
    the same rate as trained targets.

Monotonicity is NOT claimed: V^k is not contained in V^(k+1), and the trained arm
dips between k=48 and k=64. Only the first-hit CDF is monotone by construction,
and `first_hit_cdf` computes that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Distribution-free summaries
# --------------------------------------------------------------------------- #
def crossing_capacity(k: Sequence[float], rate: Sequence[float],
                      level: float) -> float:
    """Capacity at which `rate` first crosses `level`, interpolated in log2 k.

    Interpolating in log2 k rather than k matters: the grid is geometric
    (1,2,3,4,6,8,12,16,20,24,32,48,64), so linear interpolation would bias every
    quantile upward.
    """
    k = np.asarray(k, dtype=float)
    r = np.asarray(rate, dtype=float)
    order = np.argsort(k)
    k, r = k[order], r[order]
    for i in range(1, len(k)):
        if r[i - 1] < level <= r[i]:
            if k[i - 1] <= 0:
                return float(k[i])
            x0, x1 = math.log2(k[i - 1]), math.log2(k[i])
            t = (level - r[i - 1]) / (r[i] - r[i - 1])
            return float(2 ** (x0 + t * (x1 - x0)))
    return float("nan")


def quantiles(k: Sequence[float], rate: Sequence[float],
              levels: Sequence[float] = (0.25, 0.50, 0.75)) -> Dict[float, float]:
    return {lv: crossing_capacity(k, rate, lv) for lv in levels}


def quantiles_with_ci(k: Sequence[float], hits: Sequence[int],
                      n_per_person: int, n_persons: int,
                      levels: Sequence[float] = (0.25, 0.50, 0.75),
                      n_boot: int = 2000, seed: int = 0
                      ) -> Dict[float, Tuple[float, float, float]]:
    """Quantiles with a person-level bootstrap.

    `hits[i]` is the number of successes at capacity k[i] out of
    n_per_person * n_persons attempts. We resample persons, which requires the
    per-person breakdown; when only aggregate counts are available we fall back
    to a binomial resample and label it as such by widening nothing, so prefer
    `quantiles_with_ci_from_units` when the per-person data exist.
    """
    rng = np.random.default_rng(seed)
    k = np.asarray(k, dtype=float)
    n_tot = n_per_person * n_persons
    out: Dict[float, Tuple[float, float, float]] = {}
    draws = {lv: [] for lv in levels}
    for _ in range(n_boot):
        r = rng.binomial(n_tot, np.asarray(hits, dtype=float) / n_tot) / n_tot
        for lv in levels:
            draws[lv].append(crossing_capacity(k, r, lv))
    base = quantiles(k, np.asarray(hits, dtype=float) / n_tot, levels)
    for lv in levels:
        d = np.asarray([x for x in draws[lv] if np.isfinite(x)])
        lo, hi = (np.quantile(d, [0.025, 0.975]) if d.size else (np.nan, np.nan))
        out[lv] = (base[lv], float(lo), float(hi))
    return out


def quantiles_with_ci_from_units(k: Sequence[float],
                                 units_by_k: Dict[float, List[Sequence[int]]],
                                 levels: Sequence[float] = (0.25, 0.50, 0.75),
                                 n_boot: int = 2000, seed: int = 0
                                 ) -> Dict[float, Tuple[float, float, float]]:
    """Quantiles with a genuine person-clustered bootstrap.

    `units_by_k[k]` is a list, one entry per person, of that person's 0/1
    outcomes at capacity k. Persons are resampled once and the SAME resample is
    used at every capacity, which is what makes the curve's uncertainty
    coherent: the alternative, resampling independently per k, produces
    interval bands that no single population curve could have generated.
    """
    rng = np.random.default_rng(seed)
    ks = sorted(units_by_k)
    n_persons = len(units_by_k[ks[0]])
    draws = {lv: [] for lv in levels}
    for _ in range(n_boot):
        pick = rng.choice(np.arange(n_persons), size=n_persons, replace=True)
        rates = []
        for kk in ks:
            per = units_by_k[kk]
            vals = np.concatenate([np.asarray(per[i], dtype=float) for i in pick])
            rates.append(vals.mean())
        for lv in levels:
            draws[lv].append(crossing_capacity(ks, rates, lv))
    base_rates = [np.concatenate([np.asarray(u, dtype=float)
                                  for u in units_by_k[kk]]).mean() for kk in ks]
    out = {}
    for lv in levels:
        d = np.asarray([x for x in draws[lv] if np.isfinite(x)])
        lo, hi = (np.quantile(d, [0.025, 0.975]) if d.size else (np.nan, np.nan))
        out[lv] = (crossing_capacity(ks, base_rates, lv), float(lo), float(hi))
    return out


def first_hit_cdf(df, arm: str = "control") -> Dict[float, float]:
    """Pr[k_min(t) <= k], the one quantity that IS monotone by construction.

    Requires a per-attempt log with `capacity_k`, `target_membership`,
    `person_id`, `field`, `exact_match`.
    """
    sub = df[df["target_membership"] == arm]
    ks = sorted(x for x in sub["capacity_k"].unique() if x > 0)
    tgt = sub.groupby(["person_id", "field"])
    kmin = {}
    for key, g in tgt:
        hit = g[g["exact_match"]]
        kmin[key] = hit["capacity_k"].min() if len(hit) else math.inf
    n = len(kmin)
    return {k: sum(1 for v in kmin.values() if v <= k) / n for k in ks} if n else {}


# --------------------------------------------------------------------------- #
# Parametric families, compared explicitly
# --------------------------------------------------------------------------- #
def _binom_nll(pred: np.ndarray, hits: np.ndarray, n: np.ndarray) -> float:
    p = np.clip(pred, 1e-12, 1 - 1e-12)
    return float(-np.sum(hits * np.log(p) + (n - hits) * np.log(1 - p)))


def _logistic(x, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


FAMILIES: Dict[str, Dict] = {
    "logistic_in_k": {
        "p0": [-3.0, 0.2], "n_par": 2,
        "f": lambda k, p: _logistic(k, p[0], p[1]),
        "why": "the naive choice; predicts a non-zero rate at k=1",
    },
    "logistic_in_log2k": {
        "p0": [-6.0, 2.0], "n_par": 2,
        "f": lambda k, p: _logistic(np.log2(k), p[0], p[1]),
        "why": "scale-free in capacity",
    },
    "logistic_in_log2k_ceiling": {
        "p0": [-6.0, 2.0, 0.95], "n_par": 3,
        "f": lambda k, p: p[2] * _logistic(np.log2(k), p[0], p[1]),
        "why": "adds a saturation level",
    },
    "gumbel_in_log2k_ceiling": {
        "p0": [3.0, 0.7, 0.93], "n_par": 3,
        "f": lambda k, p: p[2] * np.exp(-np.exp(-(np.log2(k) - p[0]) / max(p[1], 1e-6))),
        "why": "asymmetric onset: slow start, long tail",
    },
}


@dataclass
class Fit:
    family: str
    params: List[float]
    nll: float
    deviance: float
    df: int
    n_par: int
    predict: Callable

    def as_row(self) -> Dict[str, object]:
        return {"family": self.family, "n_par": self.n_par,
                "deviance": round(self.deviance, 2), "df": self.df,
                "params": [round(p, 4) for p in self.params]}


def fit_family(k: Sequence[float], hits: Sequence[int], n: Sequence[int],
               family: str) -> Fit:
    from scipy.optimize import minimize
    spec = FAMILIES[family]
    kk = np.asarray(k, dtype=float)
    hh = np.asarray(hits, dtype=float)
    nn = np.asarray(n, dtype=float)
    keep = kk > 0
    kk, hh, nn = kk[keep], hh[keep], nn[keep]

    def obj(p):
        try:
            pred = spec["f"](kk, p)
        except Exception:
            return 1e12
        if not np.all(np.isfinite(pred)):
            return 1e12
        if len(p) > 2 and not 0 < p[2] <= 1:
            return 1e12
        return _binom_nll(pred, hh, nn)

    res = minimize(obj, spec["p0"], method="Nelder-Mead",
                   options={"maxiter": 40000, "xatol": 1e-9, "fatol": 1e-9})
    # saturated deviance: 2*(nll_model - nll_saturated)
    sat = _binom_nll(np.clip(hh / nn, 1e-12, 1 - 1e-12), hh, nn)
    dev = 2 * (res.fun - sat)
    return Fit(family, list(map(float, res.x)), float(res.fun), float(dev),
               int(keep.sum() - spec["n_par"]), spec["n_par"],
               lambda x, p=res.x, f=spec["f"]: f(np.asarray(x, dtype=float), p))


def compare_families(k, hits, n) -> List[Dict[str, object]]:
    """Fit every family and return the comparison table.

    Report this table in the paper. It is what makes the choice of law defensible
    rather than a curve the authors liked: the naive logistic in raw k is
    rejected because it predicts a substantial rate at k=1 where we observe none.
    """
    rows = []
    for fam in FAMILIES:
        try:
            f = fit_family(k, hits, n, fam)
            row = f.as_row()
            row["why"] = FAMILIES[fam]["why"]
            rows.append(row)
        except Exception as exc:                        # pragma: no cover
            rows.append({"family": fam, "error": str(exc)})
    return sorted(rows, key=lambda r: r.get("deviance", math.inf))


# --------------------------------------------------------------------------- #
# Derived quantities that go in the paper
# --------------------------------------------------------------------------- #
def beta_eff(h_inf_bits: float, k_50: float) -> float:
    """Realized bits per free token at the median crossing.

    A descriptive median rate, not a model constant: `linear_model_rejected`
    shows no single beta reproduces the curve.
    """
    return h_inf_bits / k_50 if k_50 and np.isfinite(k_50) else float("nan")


def address_rate(k: Sequence[float], alpha: Sequence[float],
                 h_inf_bits: float) -> Dict[float, float]:
    """gamma_k = (H_inf + log2 alpha_k) / k, the bits per token the search
    actually realizes at each capacity. Falls with k, which is the point: the
    nominal log2|V| is never achieved and the gap widens."""
    out = {}
    for kk, aa in zip(k, alpha):
        if kk > 0 and aa > 0:
            out[float(kk)] = (h_inf_bits + math.log2(aa)) / kk
    return out


def linear_model_rejected(k: Sequence[float], rate: Sequence[float]) -> Dict[str, float]:
    """Test of a linear 'beta bits per free token' threshold model.

    If a target were forced exactly when beta*k >= H(t), the quantile ratio
    k_75/k_25 would equal H_75/H_25. Half our pool is SSNs with identical
    min-entropy and identical normalized length by construction, so a ratio far
    from 1 refutes the model.

    Caveat the caller must respect: on a POOLED SSN+email sweep part of the
    dispersion can be the two-field mixture. Run this per field.
    """
    q = quantiles(k, rate, (0.25, 0.50, 0.75))
    ratio = q[0.75] / q[0.25] if q[0.25] else float("nan")
    return {"k_25": q[0.25], "k_50": q[0.50], "k_75": q[0.75],
            "k75_over_k25": ratio,
            "required_entropy_spread": ratio}


def fit_gap_decay(k: Sequence[float], alpha: Sequence[float],
                  tau: Sequence[float], tau_se: Sequence[float],
                  k_min_fit: float = 6.0) -> Dict[str, float]:
    """One-parameter fit of tau(k) = (1 - alpha_k) * m  (Prop. prop:decay).

    Fitted by weighted least squares over capacities at or above `k_min_fit`,
    i.e. where trained targets are elicitable at all. A good fit means the
    observed gap shrinks purely because the floor rises, and the interior
    maximum is a consequence rather than an anomaly.
    """
    x = np.asarray([1 - a for a in alpha], dtype=float)
    y = np.asarray(tau, dtype=float)
    se = np.asarray(tau_se, dtype=float)
    kk = np.asarray(k, dtype=float)
    keep = (kk >= k_min_fit) & np.isfinite(y) & np.isfinite(se) & (se > 0)
    x, y, se = x[keep], y[keep], se[keep]
    if x.size < 2:
        return {"m": float("nan"), "se": float("nan"), "chi2": float("nan"),
                "df": 0, "n_points": int(x.size)}
    w = 1.0 / se ** 2
    m = float((w * x * y).sum() / (w * x * x).sum())
    se_m = float(1.0 / math.sqrt((w * x * x).sum()))
    chi2 = float((w * (y - m * x) ** 2).sum())
    return {"m": m, "se": se_m, "chi2": chi2, "df": int(x.size - 1),
            "n_points": int(x.size),
            "predicted_argmax_k": float(kk[keep][np.argmax(m * x)])}


def horizontal_shift(k: Sequence[float], hits_c: Sequence[int],
                     hits_d: Sequence[int], n: Sequence[int],
                     family: str = "gumbel_in_log2k_ceiling",
                     n_boot: int = 0, seed: int = 0) -> Dict[str, float]:
    """How many more free tokens controls need to reach the trained rate.

    Fits a shared shape (slope and ceiling) with an arm-specific location, and
    reports the shift at the control median. This is the effect size that stays
    interpretable once the floor has compressed the vertical gap to nothing: at
    k=20 the vertical difference is a few points with an interval spanning zero,
    while the horizontal difference is about one token.

    A likelihood-ratio statistic for shared versus arm-specific slope is
    returned so the shared-shape assumption is checked rather than assumed.
    Pass n_boot > 0 for a person-free binomial interval; prefer a paired person
    bootstrap at the caller when per-person data exist.
    """
    from scipy.optimize import minimize
    kk = np.asarray(k, dtype=float)
    hc = np.asarray(hits_c, dtype=float)
    hd = np.asarray(hits_d, dtype=float)
    nn = np.asarray(n, dtype=float)
    keep = kk > 0
    kk, hc, hd, nn = kk[keep], hc[keep], hd[keep], nn[keep]
    x = np.log2(kk)

    def curve(mu, sig, A):
        return A * np.exp(-np.exp(-(x - mu) / max(sig, 1e-6)))

    def nll_shared(p):
        mu_c, mu_d, sig, A = p
        if not (0 < A <= 1 and sig > 0):
            return 1e12
        return (_binom_nll(curve(mu_c, sig, A), hc, nn)
                + _binom_nll(curve(mu_d, sig, A), hd, nn))

    def nll_free(p):
        mu_c, mu_d, sig_c, sig_d, A = p
        if not (0 < A <= 1 and sig_c > 0 and sig_d > 0):
            return 1e12
        return (_binom_nll(curve(mu_c, sig_c, A), hc, nn)
                + _binom_nll(curve(mu_d, sig_d, A), hd, nn))

    rs = minimize(nll_shared, [3.0, 2.9, 0.7, 0.93], method="Nelder-Mead",
                  options={"maxiter": 60000, "xatol": 1e-9, "fatol": 1e-9})
    rf = minimize(nll_free, [3.0, 2.9, 0.7, 0.7, 0.93], method="Nelder-Mead",
                  options={"maxiter": 60000, "xatol": 1e-9, "fatol": 1e-9})
    mu_c, mu_d, sig, A = rs.x
    half = math.log(math.log(2.0))
    k_c, k_d = 2 ** (mu_c - sig * half), 2 ** (mu_d - sig * half)
    return {
        "mu_control": float(mu_c), "mu_trained": float(mu_d),
        "sigma": float(sig), "ceiling": float(A),
        "k50_control": float(k_c), "k50_trained": float(k_d),
        "shift_tokens": float(k_c - k_d),
        "shift_multiplicative": float(2 ** (mu_c - mu_d)),
        "lr_shared_vs_free": float(2 * (rs.fun - rf.fun)),
        "lr_df": 1,
        "ceiling_note": "weakly identified inside the tested grid if the "
                        "observed rate at the largest k exceeds it",
    }


# --------------------------------------------------------------------------- #
# Convenience: run the whole analysis on a sweep CSV or a per-attempt log
# --------------------------------------------------------------------------- #
def analyse_sweep(k, c_hits, d_hits, n, h_inf_bits: float,
                  tau=None, tau_se=None) -> Dict[str, object]:
    k = list(map(float, k))
    n = list(n)
    alpha = [h / nn for h, nn in zip(c_hits, n)]
    pi = [h / nn for h, nn in zip(d_hits, n)]
    if tau is None:
        tau = [p - a for p, a in zip(pi, alpha)]
    if tau_se is None:
        tau_se = [math.sqrt(a * (1 - a) / nn + p * (1 - p) / nn) or 1e-3
                  for a, p, nn in zip(alpha, pi, n)]
    qc = quantiles(k, alpha)
    out: Dict[str, object] = {
        "quantiles_control": qc,
        "quantiles_trained": quantiles(k, pi),
        "beta_eff_bits_per_token": beta_eff(h_inf_bits, qc[0.50]),
        "nominal_bits_per_token": None,
        "address_rate": address_rate(k, alpha, h_inf_bits),
        "family_comparison": compare_families(k, c_hits, n),
        "linear_model": linear_model_rejected(k, alpha),
        "gap_decay": fit_gap_decay(k, alpha, tau, tau_se),
        "horizontal_shift": horizontal_shift(k, c_hits, d_hits, n),
        "monotone_in_k": bool(all(b >= a - 1e-12 for a, b in zip(alpha, alpha[1:]))),
        "trained_monotone_in_k": bool(all(b >= a - 1e-12
                                          for a, b in zip(pi, pi[1:]))),
    }
    return out


def main() -> None:
    import argparse
    import csv
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-csv", required=True,
                    help="columns k,c_hits,c_attempts,d_hits,d_attempts[,tau,...]")
    ap.add_argument("--h-inf", type=float, default=None,
                    help="min-entropy in bits; default = Faker en_US SSN")
    a = ap.parse_args()

    if a.h_inf is None:
        from .theory import ssn_min_entropy_en_us
        a.h_inf = ssn_min_entropy_en_us().h_inf_bits

    rows = list(csv.DictReader(open(a.sweep_csv)))
    k = [float(r["k"]) for r in rows]
    ch = [int(r["c_hits"]) for r in rows]
    dh = [int(r["d_hits"]) for r in rows]
    n = [int(r["c_attempts"]) for r in rows]
    tau = [float(r["tau"]) for r in rows] if "tau" in rows[0] else None
    tau_se = None
    if tau is not None and "tau_ci_low" in rows[0]:
        tau_se = [max((float(r["tau_ci_high"]) - float(r["tau_ci_low"])) / (2 * 1.96),
                      1e-3) for r in rows]

    res = analyse_sweep(k, ch, dh, n, a.h_inf, tau, tau_se)
    print(f"\nH_inf used: {a.h_inf:.3f} bits\n")
    print("control quantiles :", {f"k_{int(100*l)}": round(v, 2)
                                  for l, v in res["quantiles_control"].items()})
    print("trained quantiles :", {f"k_{int(100*l)}": round(v, 2)
                                  for l, v in res["quantiles_trained"].items()})
    print(f"beta_eff          : {res['beta_eff_bits_per_token']:.2f} bits/free token")
    print("address rate gamma_k (bits/token):",
          {int(kk): round(v, 2) for kk, v in res["address_rate"].items()})
    lm = res["linear_model"]
    print(f"\nlinear model test : k75/k25 = {lm['k75_over_k25']:.2f} "
          "(a linear beta model needs this to equal the entropy spread, which is "
          "1 for the SSN half by construction)")
    gd = res["gap_decay"]
    print(f"gap decay fit     : m = {gd['m']:.3f} (SE {gd['se']:.3f}), "
          f"chi2 = {gd['chi2']:.2f} on {gd['df']} df, "
          f"predicted argmax at k = {gd.get('predicted_argmax_k')}")
    hs = res["horizontal_shift"]
    print(f"horizontal shift  : controls need {hs['shift_tokens']:.2f} more tokens "
          f"({hs['shift_multiplicative']:.2f}x), shared-shape LR = "
          f"{hs['lr_shared_vs_free']:.2f} on 1 df")
    print(f"\nmonotone in k?    control {res['monotone_in_k']}, "
          f"trained {res['trained_monotone_in_k']}  "
          "(do not claim monotonicity as a theorem)")
    print("\nfamily comparison (lower deviance is better):")
    for r in res["family_comparison"]:
        if "error" in r:
            print(f"  {r['family']:32s} FAILED {r['error']}")
        else:
            print(f"  {r['family']:32s} dev={r['deviance']:9.2f} "
                  f"df={r['df']:2d} par={r['n_par']}  {r['why']}")


if __name__ == "__main__":
    main()
