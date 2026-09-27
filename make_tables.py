"""
Single source of truth for every paper table/figure (§0 of the experiment plan).

The old pipeline produced separate ad-hoc aggregations that could (and did)
contradict each other (record-level 0% vs per-field 53%). This script instead
reads the ONE per-attempt log written by attempt_log.py and derives EVERY table
and figure-datum from it by pure groupby+aggregate --- no model calls, CPU only.
Because there is a single input and a single pass, the tables can never
disagree.

Determinism: running twice must give byte-identical output. We therefore (a)
seed every bootstrap with a fixed value, (b) resample by person_id (respecting
pairing) via that fixed RNG, (c) rely on pandas groupby / np.unique sorting for
stable iteration order, and (d) format every float with a fixed precision.

The scientific spine is forcing-vs-memorization:
  * A negative control (target_membership=control, train_frequency=0) measures
    how much a probe FORCES a never-trained string out of the model.
  * Adj = EMR(trained) - EMR(control) is the part attributable to MEMORIZATION.
  * The capacity sweep (E3) shows the control floor alpha_k rising with the
    number of free token positions k --- i.e. expressive attacks force anything.

Usage:  python make_tables.py --run-id run1
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

import attempt_log
import stats as st
from config import RESULTS_DIR, eval_cfg

# ---------------------------------------------------------------------------
# Constants (fixed so output is reproducible)
# ---------------------------------------------------------------------------
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

BOOT_SEED = 20240601          # matches stats._RNG_SEED; keeps CIs reproducible
N_BOOT = int(eval_cfg.n_bootstrap)          # cluster bootstrap for means/diffs
N_BOOT_AUC = 2000             # loop bootstrap for AUC/TPR (kept modest on purpose)
CI = float(eval_cfg.bootstrap_ci)

VOCAB_SIZE = 50257            # GPT-2 BPE vocab; |V| for the beta / log2|V| check
LOG2_VOCAB = math.log2(VOCAB_SIZE)

# The headline (Table 1 / Table 2) is measured on the primary confirmation
# experiment when present, else on whatever rows carry the relevant probes.
MAIN_EXP = "E1"
PRIMARY_PROBE = "gcg_free"
FIXED_PROBE = "fixed"

# Expressivity ordering for the probe spectrum (Table 2): larger => more forcing
# power. softprompt is a continuous prefix (capacity_k=-1) => the forcing UPPER
# bound and therefore sits at the top with Adj ~ 0.
_EXPRESSIVITY = {
    "softprompt": 1e9,
    "gcg_fluent": 64.0,
    "gcg_free": 48.0,
    "gcg_anchored": 32.0,
    "random_restart": 24.0,
    "piiscope": 16.0,
    "piicompass": 8.0,
    "fixed_budget": 4.0,      # compute-matched natural prompting (E7)
    "fixed_matched": 1.0,
    "fixed": 0.0,
}


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def _fmt_ci(lo: float, hi: float, scale: float = 100.0, prec: int = 1) -> str:
    return f"[{scale*lo:.{prec}f},{scale*hi:.{prec}f}]"


def _pct(x: float) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100*x:.1f}"


# ---------------------------------------------------------------------------
# Boolean coercion (the log stores nullable pandas booleans)
# ---------------------------------------------------------------------------

def _boolcol(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].fillna(False).astype(bool).astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Person-clustered bootstrap (respects pairing: the resampling unit is person_id)
# ---------------------------------------------------------------------------

def _person_groups(df: pd.DataFrame, col: str = "exact_match") -> List[np.ndarray]:
    """One 0/1 array per person_id (sorted), of the given binary column."""
    if len(df) == 0:
        return []
    tmp = df[["person_id"]].copy()
    tmp["_v"] = _boolcol(df, col)
    return [g["_v"].to_numpy() for _, g in tmp.groupby("person_id", sort=True)]


def _cluster_emr_ci(groups: Sequence[np.ndarray],
                    n_boot: int = N_BOOT, ci: float = CI,
                    seed: int = BOOT_SEED) -> Dict:
    """Micro-EMR point estimate + person-cluster bootstrap CI (vectorized)."""
    groups = [g for g in groups if len(g)]
    if not groups:
        return {"emr": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_persons": 0, "n_rows": 0}
    sums = np.array([g.sum() for g in groups], dtype=float)
    cnts = np.array([len(g) for g in groups], dtype=float)
    point = sums.sum() / cnts.sum()
    rng = np.random.default_rng(seed)
    n = len(groups)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return {"emr": float(point), "ci_low": lo, "ci_high": hi,
            "n_persons": n, "n_rows": int(cnts.sum())}


def _cluster_diff_ci(groups_d: Sequence[np.ndarray], groups_c: Sequence[np.ndarray],
                     n_boot: int = N_BOOT, ci: float = CI,
                     seed: int = BOOT_SEED) -> Dict:
    """Adj = micro-EMR(D) - micro-EMR(C) with a person-cluster bootstrap CI.

    trained and control are disjoint persons, so the two arms are resampled
    independently from ONE deterministic RNG stream.
    """
    rng = np.random.default_rng(seed)

    def _arm(groups):
        groups = [g for g in groups if len(g)]
        if not groups:
            return None, None
        sums = np.array([g.sum() for g in groups], dtype=float)
        cnts = np.array([len(g) for g in groups], dtype=float)
        n = len(groups)
        idx = rng.integers(0, n, size=(n_boot, n))
        return sums.sum() / cnts.sum(), sums[idx].sum(1) / cnts[idx].sum(1)

    pd_, bd = _arm(groups_d)
    pc_, bc = _arm(groups_c)
    if pd_ is None or pc_ is None:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diff = pd_ - pc_
    bdiff = bd - bc
    lo = float(np.percentile(bdiff, (1 - ci) / 2 * 100))
    hi = float(np.percentile(bdiff, (1 + ci) / 2 * 100))
    return {"diff": float(diff), "ci_low": lo, "ci_high": hi}


# ---------------------------------------------------------------------------
# Continuous-score audit (E9): ROC AUC + TPR@FPR, with person-cluster bootstrap
# ---------------------------------------------------------------------------

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (tie-corrected ranks)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
    """TPR when the threshold is set at the (1-fpr) quantile of NEGATIVE scores."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr, method="higher"))
    return float((pos >= thr).mean())


def _score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """s = -final_target_nll; label 1=trained, 0=control. Drops NaN scores."""
    sub = df[df["final_target_nll"].notna()].copy()
    sub = sub[sub["target_membership"].isin(["trained", "control"])]
    sub["_score"] = -sub["final_target_nll"].astype(float)
    sub["_label"] = (sub["target_membership"] == "trained").astype(int)
    return sub[["person_id", "_score", "_label"]]


def _audit_ci(df: pd.DataFrame, seed: int = BOOT_SEED) -> Dict:
    sf = _score_frame(df)
    if sf["_label"].nunique() < 2:
        return {k: float("nan") for k in
                ("auc", "auc_lo", "auc_hi", "tpr1", "tpr1_lo", "tpr1_hi",
                 "tpr5", "tpr5_lo", "tpr5_hi")}
    s = sf["_score"].to_numpy()
    y = sf["_label"].to_numpy()
    point = {"auc": _auc(s, y),
             "tpr1": _tpr_at_fpr(s, y, 0.01),
             "tpr5": _tpr_at_fpr(s, y, 0.05)}

    # person-cluster bootstrap
    persons = sf["person_id"].to_numpy()
    uniq = np.unique(persons)
    rows_by_p = {p: np.where(persons == p)[0] for p in uniq}
    idx_lists = [rows_by_p[p] for p in uniq]
    rng = np.random.default_rng(seed)
    n = len(uniq)
    aucs, t1s, t5s = np.empty(N_BOOT_AUC), np.empty(N_BOOT_AUC), np.empty(N_BOOT_AUC)
    for b in range(N_BOOT_AUC):
        pick = rng.integers(0, n, size=n)
        rows = np.concatenate([idx_lists[i] for i in pick])
        bs, by = s[rows], y[rows]
        aucs[b] = _auc(bs, by)
        t1s[b] = _tpr_at_fpr(bs, by, 0.01)
        t5s[b] = _tpr_at_fpr(bs, by, 0.05)

    def _pc(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float("nan"), float("nan")
        return (float(np.percentile(arr, (1 - CI) / 2 * 100)),
                float(np.percentile(arr, (1 + CI) / 2 * 100)))

    a_lo, a_hi = _pc(aucs)
    t1_lo, t1_hi = _pc(t1s)
    t5_lo, t5_hi = _pc(t5s)
    return {"auc": point["auc"], "auc_lo": a_lo, "auc_hi": a_hi,
            "tpr1": point["tpr1"], "tpr1_lo": t1_lo, "tpr1_hi": t1_hi,
            "tpr5": point["tpr5"], "tpr5_lo": t5_lo, "tpr5_hi": t5_hi}


# ---------------------------------------------------------------------------
# Paired McNemar over trained records (probe_a vs probe_b), keyed person+field+seed
# ---------------------------------------------------------------------------

def _paired_mcnemar(df: pd.DataFrame, probe_a: str, probe_b: str) -> Dict:
    tr = df[df["target_membership"] == "trained"]

    def _idx(probe):
        sub = tr[tr["probe"] == probe]
        if len(sub) == 0:
            return {}
        v = _boolcol(sub, "exact_match")
        keys = list(zip(sub["person_id"].astype(str), sub["field"].astype(str),
                        sub["seed"].astype("Int64").astype(str)))
        return dict(zip(keys, v))

    ia, ib = _idx(probe_a), _idx(probe_b)
    keys = sorted(set(ia) & set(ib))
    if not keys:
        return {"p_value": None, "n_discordant": 0, "b": 0, "c": 0}
    a = np.array([ia[k] for k in keys])
    b = np.array([ib[k] for k in keys])
    return st.mcnemar_test(a, b)


# ---------------------------------------------------------------------------
# Row selection helper: prefer MAIN_EXP rows when they exist
# ---------------------------------------------------------------------------

def _main_rows(df: pd.DataFrame) -> pd.DataFrame:
    if (df["exp_id"] == MAIN_EXP).any():
        return df[df["exp_id"] == MAIN_EXP]
    return df


# ---------------------------------------------------------------------------
# TABLE 1 (main) — per (model_name, model_state)
# ---------------------------------------------------------------------------

def table1_main(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    base = _main_rows(df)
    lines = []
    lines.append("=" * 116)
    lines.append("TABLE 1 (main): forcing-vs-memorization on the confirmation probe (gcg_free)")
    lines.append("  EMR(D)=trained, EMR(C)=control(never trained). Adj=EMR(D)-EMR(C) is MEMORIZATION;")
    lines.append("  a large Neg-ctrl / small Adj means the probe FORCES outputs rather than recalling.")
    lines.append("  E9 audit uses s=-final_target_nll (positives=trained, negatives=control).")
    lines.append("=" * 116)
    header = (f"{'model':<22}{'state':<10}{'Fixed':>7}{'GCG':>7}{'Negctl':>8}"
              f"{'Adj[CI]':>20}{'AUC[CI]':>18}{'TPR@1%':>9}{'TPR@5%':>9}{'p(McN)':>9}")
    lines.append(header)
    lines.append("-" * 116)

    csv_rows = []
    for (model, state), g in base.groupby(["model_name", "model_state"], sort=True):
        gf = g[g["probe"] == PRIMARY_PROBE]
        fx = g[g["probe"] == FIXED_PROBE]

        emr_d = _cluster_emr_ci(_person_groups(gf[gf["target_membership"] == "trained"]))
        emr_c = _cluster_emr_ci(_person_groups(gf[gf["target_membership"] == "control"]))
        fixed_d = _cluster_emr_ci(_person_groups(fx[fx["target_membership"] == "trained"]))
        adj = _cluster_diff_ci(
            _person_groups(gf[gf["target_membership"] == "trained"]),
            _person_groups(gf[gf["target_membership"] == "control"]))
        audit = _audit_ci(gf)
        mcn = _paired_mcnemar(g, PRIMARY_PROBE, FIXED_PROBE)

        adj_str = f"{adj['diff']*100:+.1f}{_fmt_ci(adj['ci_low'], adj['ci_high'])}"
        auc_str = (f"{audit['auc']:.3f}[{audit['auc_lo']:.3f},{audit['auc_hi']:.3f}]"
                   if not math.isnan(audit["auc"]) else "n/a")
        lines.append(
            f"{str(model):<22}{str(state):<10}"
            f"{_pct(fixed_d['emr']):>7}{_pct(emr_d['emr']):>7}{_pct(emr_c['emr']):>8}"
            f"{adj_str:>20}{auc_str:>18}"
            f"{_pct(audit['tpr1']):>9}{_pct(audit['tpr5']):>9}"
            f"{_fmt_p(mcn.get('p_value')):>9}")

        csv_rows.append({
            "model_name": model, "model_state": state,
            "emr_fixed_trained": fixed_d["emr"],
            "emr_gcg_trained": emr_d["emr"], "emr_gcg_trained_lo": emr_d["ci_low"],
            "emr_gcg_trained_hi": emr_d["ci_high"],
            "emr_gcg_control": emr_c["emr"], "emr_gcg_control_lo": emr_c["ci_low"],
            "emr_gcg_control_hi": emr_c["ci_high"],
            "adj": adj["diff"], "adj_lo": adj["ci_low"], "adj_hi": adj["ci_high"],
            "auc": audit["auc"], "auc_lo": audit["auc_lo"], "auc_hi": audit["auc_hi"],
            "tpr_at_1pct_fpr": audit["tpr1"], "tpr_at_5pct_fpr": audit["tpr5"],
            "mcnemar_p": mcn.get("p_value"), "n_persons_trained": emr_d["n_persons"],
        })
    lines.append("")
    return lines, pd.DataFrame(csv_rows)


# ---------------------------------------------------------------------------
# E3 capacity sweep — Fig.1 data + acceptance checks
# ---------------------------------------------------------------------------

def _kmin_table(e3: pd.DataFrame) -> pd.DataFrame:
    """One row per target: k_min = min capacity_k with exact_match (NaN if none)."""
    rows = []
    key = ["person_id", "field", "target_membership"]
    for keyvals, g in e3.groupby(key, sort=True):
        hit = g[g["exact_match"].fillna(False).astype(bool)]
        kmin = float(hit["capacity_k"].min()) if len(hit) else float("nan")
        rows.append({
            "person_id": keyvals[0], "field": keyvals[1],
            "target_membership": keyvals[2], "k_min": kmin,
            "H_bits": float(g["target_H_bits"].dropna().iloc[0])
            if g["target_H_bits"].notna().any() else float("nan"),
            "len_tokens": float(g["target_len_tokens"].dropna().iloc[0])
            if g["target_len_tokens"].notna().any() else float("nan"),
        })
    return pd.DataFrame(rows)


def _crossing_k(ks: np.ndarray, alpha: np.ndarray, thr: float) -> float:
    """Smallest (interpolated) k where control EMR alpha_k crosses thr."""
    for i in range(len(ks)):
        if alpha[i] >= thr:
            if i == 0:
                return float(ks[0])
            a0, a1 = alpha[i - 1], alpha[i]
            k0, k1 = ks[i - 1], ks[i]
            if a1 == a0:
                return float(k1)
            return float(k0 + (thr - a0) * (k1 - k0) / (a1 - a0))
    return float("nan")


def capacity_e3(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    e3 = df[df["exp_id"] == "E3"]
    lines = ["=" * 92,
             "FIG.1 / CAPACITY SWEEP (E3): control floor alpha_k vs free token positions k",
             "  alpha_k = EMR(control) at capacity k (the FORCING floor). cov = EMR(trained).",
             "=" * 92]
    if len(e3) == 0:
        lines.append("  (no E3 rows in log)")
        lines.append("")
        return lines, pd.DataFrame(columns=["k", "alpha_k_control", "alpha_lo",
                                            "alpha_hi", "cov_trained", "cov_lo", "cov_hi"])

    ks = sorted(int(k) for k in e3["capacity_k"].dropna().unique())
    lines.append(f"{'k':>5}{'alpha_k(ctrl)':>16}{'alpha CI':>18}"
                 f"{'cov(trained)':>14}{'cov CI':>18}")
    lines.append("-" * 92)
    csv_rows, alpha_vec = [], []
    for k in ks:
        gk = e3[e3["capacity_k"] == k]
        a = _cluster_emr_ci(_person_groups(gk[gk["target_membership"] == "control"]))
        c = _cluster_emr_ci(_person_groups(gk[gk["target_membership"] == "trained"]))
        alpha_vec.append(a["emr"])
        lines.append(f"{k:>5}{_pct(a['emr']):>16}"
                     f"{_fmt_ci(a['ci_low'], a['ci_high']):>18}"
                     f"{_pct(c['emr']):>14}{_fmt_ci(c['ci_low'], c['ci_high']):>18}")
        csv_rows.append({"k": k, "alpha_k_control": a["emr"],
                         "alpha_lo": a["ci_low"], "alpha_hi": a["ci_high"],
                         "cov_trained": c["emr"], "cov_lo": c["ci_low"],
                         "cov_hi": c["ci_high"]})

    alpha_vec = np.array(alpha_vec)
    ks_arr = np.array(ks, dtype=float)

    # forcing model: k_min ~ H_bits  =>  beta = slope (bigger H needs bigger k)
    km = _kmin_table(e3)
    fit = km[np.isfinite(km["k_min"]) & np.isfinite(km["H_bits"])]
    beta = r_val = p_val = float("nan")
    if len(fit) >= 2 and fit["H_bits"].std() > 0:
        from scipy.stats import linregress
        lr = linregress(fit["H_bits"].to_numpy(), fit["k_min"].to_numpy())
        beta, r_val, p_val = float(lr.slope), float(lr.rvalue), float(lr.pvalue)

    # per-field betas => dispersion across fields
    field_betas = {}
    for fld, gg in fit.groupby("field", sort=True):
        if len(gg) >= 2 and gg["H_bits"].std() > 0:
            from scipy.stats import linregress
            field_betas[fld] = float(linregress(gg["H_bits"], gg["k_min"]).slope)
    beta_disp = float(np.std(list(field_betas.values()))) if len(field_betas) >= 2 else float("nan")

    k_star_1 = _crossing_k(ks_arr, alpha_vec, 0.01)
    k_star_5 = _crossing_k(ks_arr, alpha_vec, 0.05)

    # acceptance checks
    monotone = bool(np.all(np.diff(alpha_vec) >= -1e-9))
    beta_ratio = beta / LOG2_VOCAB if not math.isnan(beta) else float("nan")

    lines.append("-" * 92)
    lines.append("  Forcing model  k_min ~ target_H_bits:")
    lines.append(f"    beta (slope, k per bit) = {beta:.4f}   r={r_val:.3f}  p={_fmt_p(p_val)}"
                 if not math.isnan(beta) else "    beta = n/a (insufficient finite k_min)")
    lines.append(f"    k*(1% control floor) = {k_star_1:.2f}    "
                 f"k*(5% control floor) = {k_star_5:.2f}")
    lines.append("  Acceptance checks:")
    lines.append(f"    [{'PASS' if monotone else 'FAIL'}] alpha_k monotone non-decreasing")
    lines.append(f"    [----] beta / log2|V| = {beta_ratio:.4f}  (|V|={VOCAB_SIZE}, log2|V|={LOG2_VOCAB:.3f})"
                 if not math.isnan(beta_ratio) else "    [----] beta / log2|V| = n/a")
    lines.append(f"    [----] beta dispersion across fields (std) = {beta_disp:.4f}  "
                 f"({len(field_betas)} fields)")
    lines.append("")
    return lines, pd.DataFrame(csv_rows)


# ---------------------------------------------------------------------------
# E5 frequency response — Fig.2 data + logit fit + cross-check
# ---------------------------------------------------------------------------

def frequency_e5(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    e5 = df[df["exp_id"] == "E5"]
    lines = ["=" * 92,
             "FIG.2 / FREQUENCY RESPONSE (E5): EMR vs train_frequency, per probe",
             "  intercept(freq=0)=FORCING floor; slope on log1p(freq)=MEMORIZATION.",
             "=" * 92]
    if len(e5) == 0:
        lines.append("  (no E5 rows in log)")
        lines.append("")
        return lines, pd.DataFrame(columns=["probe", "train_frequency", "emr",
                                            "emr_lo", "emr_hi", "n_persons"])

    probes = sorted(e5["probe"].dropna().unique())
    freqs = sorted(int(f) for f in e5["train_frequency"].dropna().unique())
    lines.append(f"{'probe':<16}" + "".join(f"{('f=' + str(f)):>9}" for f in freqs))
    lines.append("-" * 92)
    csv_rows = []
    for probe in probes:
        cells = []
        for f in freqs:
            gk = e5[(e5["probe"] == probe) & (e5["train_frequency"] == f)]
            m = _cluster_emr_ci(_person_groups(gk))
            cells.append(_pct(m["emr"]))
            csv_rows.append({"probe": probe, "train_frequency": f, "emr": m["emr"],
                             "emr_lo": m["ci_low"], "emr_hi": m["ci_high"],
                             "n_persons": m["n_persons"]})
        lines.append(f"{probe:<16}" + "".join(f"{c:>9}" for c in cells))

    # logit(EMR) ~ log1p(freq) * probe  (statsmodels; reference probe = PRIMARY)
    e5f = e5.copy()
    e5f["_y"] = _boolcol(e5f, "exact_match")
    e5f["_logf"] = np.log1p(e5f["train_frequency"].astype(float))
    intercept = slope = float("nan")
    if e5f["_y"].nunique() > 1:
        try:
            import statsmodels.formula.api as smf
            ref = PRIMARY_PROBE if PRIMARY_PROBE in set(probes) else probes[0]
            formula = f"_y ~ _logf * C(probe, Treatment(reference='{ref}'))" \
                if len(probes) > 1 else "_y ~ _logf"
            fit = smf.logit(formula, data=e5f).fit(disp=0)
            intercept = float(fit.params.get("Intercept", float("nan")))
            slope = float(fit.params.get("_logf", float("nan")))
        except Exception as e:  # separation / singular / missing
            lines.append(f"  [logit fit unavailable: {type(e).__name__}]")

    # interaction F/p via the shared two-way ANOVA (reuse stats.py)
    anova = st.two_way_anova(e5f["_y"].to_numpy(),
                             e5f["train_frequency"].to_numpy(),
                             e5f["probe"].to_numpy())

    # Pearson r(freq, exact_match) on the primary probe
    prim = e5f[e5f["probe"] == PRIMARY_PROBE] if PRIMARY_PROBE in set(probes) else e5f
    corr = st.pearson_corr(prim["train_frequency"].to_numpy(), prim["_y"].to_numpy())

    # CROSS-CHECK: expit(intercept) [forcing floor from fit] vs directly-measured
    # control alpha (EMR at freq=0) for the primary probe, with its CI.
    ctrl = _cluster_emr_ci(_person_groups(prim[prim["train_frequency"] == 0]))
    fit_floor = float(expit(intercept)) if not math.isnan(intercept) else float("nan")
    within = (not math.isnan(fit_floor) and not math.isnan(ctrl["ci_low"])
              and ctrl["ci_low"] <= fit_floor <= ctrl["ci_high"])

    lines.append("-" * 92)
    lines.append(f"  logit fit ({PRIMARY_PROBE} ref):  intercept={intercept:.4f} "
                 f"(forcing floor p={fit_floor:.4f})   slope[log1p(freq)]={slope:.4f} (memorization)")
    lines.append(f"  freq x probe interaction: F={anova.get('interaction_F', float('nan')):.3f} "
                 f"p={_fmt_p(anova.get('interaction_p'))} ({anova.get('backend', '?')})")
    lines.append(f"  Pearson r(freq, exact_match) [{PRIMARY_PROBE}] = {corr.get('r', 0):.3f} "
                 f"(p={_fmt_p(corr.get('p_value'))}, n={corr.get('n', 0)})")
    lines.append(f"  CROSS-CHECK: fit forcing floor={_pct(fit_floor)}%  vs  "
                 f"direct control alpha={_pct(ctrl['emr'])}% "
                 f"{_fmt_ci(ctrl['ci_low'], ctrl['ci_high'])}  "
                 f"=> {'WITHIN CI' if within else 'OUTSIDE CI'}")
    lines.append("")
    return lines, pd.DataFrame(csv_rows)


# ---------------------------------------------------------------------------
# E13 ACR comparison (reuses E3 k_min)
# ---------------------------------------------------------------------------

def acr_e13(df: pd.DataFrame) -> List[str]:
    e3 = df[df["exp_id"] == "E3"]
    lines = ["=" * 92,
             "E13 ACR COMPARISON (reuses E3 k_min): does 'adversarial compression' imply memorization?",
             "  ACR>=1  <=>  k_min < target_len_tokens (target forced with fewer tokens than its length).",
             "  If never-trained CONTROL targets satisfy ACR>=1, ACR FAILS as a forcing/memorization test.",
             "=" * 92]
    if len(e3) == 0:
        lines.append("  (no E3 rows in log)")
        lines.append("")
        return lines
    km = _kmin_table(e3)
    lines.append(f"{'membership':<14}{'n':>6}{'n_kmin':>8}{'%ACR>=1':>10}"
                 f"{'median k_min':>14}{'implied beta=H/len':>20}")
    lines.append("-" * 92)
    ctrl_acr = float("nan")
    for mem in ["control", "trained"]:
        sub = km[km["target_membership"] == mem]
        n = len(sub)
        finite = sub[np.isfinite(sub["k_min"])]
        n_km = len(finite)
        acr_mask = np.isfinite(sub["k_min"]) & np.isfinite(sub["len_tokens"]) \
            & (sub["k_min"] < sub["len_tokens"])
        pct_acr = 100.0 * acr_mask.sum() / n if n else float("nan")
        if mem == "control":
            ctrl_acr = pct_acr
        med_k = float(finite["k_min"].median()) if n_km else float("nan")
        implied = sub[np.isfinite(sub["H_bits"]) & np.isfinite(sub["len_tokens"])
                      & (sub["len_tokens"] > 0)]
        beta_hl = float((implied["H_bits"] / implied["len_tokens"]).median()) \
            if len(implied) else float("nan")
        lines.append(f"{mem:<14}{n:>6}{n_km:>8}{pct_acr:>9.1f}%"
                     f"{med_k:>14.2f}{beta_hl:>20.3f}")
    lines.append("-" * 92)
    lines.append(f"  >>> {ctrl_acr:.1f}% of NEVER-TRAINED (control) targets satisfy ACR>=1 "
                 f"(should be ~0 if ACR measured memorization).")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# E16 rank inversion
# ---------------------------------------------------------------------------

def rank_inversion_e16(df: pd.DataFrame) -> List[str]:
    e3 = df[df["exp_id"] == "E3"]
    src = e3 if len(e3) else df
    gf = src[src["probe"] == PRIMARY_PROBE]
    lines = ["=" * 92,
             "E16 RANK INVERSION (gcg_free): control target with LOWER final_target_nll than a trained one",
             "  (same field). Any such case => the raw continuous score is NOT pure memorization.",
             "=" * 92]
    # best (lowest) nll per target
    g = gf[gf["final_target_nll"].notna()]
    if len(g) == 0:
        lines.append("  (no scored gcg_free rows)")
        lines.append("")
        return lines
    best = (g.groupby(["person_id", "field", "target_membership"], sort=True)
            ["final_target_nll"].min().reset_index())
    lines.append(f"{'field':<14}{'n_trained':>10}{'n_control':>10}"
                 f"{'inv_pairs':>11}{'trained_beaten':>16}")
    lines.append("-" * 92)
    tot_pairs = tot_beaten = 0
    for fld, gg in best.groupby("field", sort=True):
        tr = gg[gg["target_membership"] == "trained"]["final_target_nll"].to_numpy()
        ct = gg[gg["target_membership"] == "control"]["final_target_nll"].to_numpy()
        if len(tr) == 0 or len(ct) == 0:
            continue
        # inversion pair: control nll < trained nll
        pairs = int(np.sum(ct[:, None] < tr[None, :]))
        beaten = int(np.sum(np.min(ct) < tr))  # trained targets beaten by best control
        tot_pairs += pairs
        tot_beaten += beaten
        lines.append(f"{str(fld):<14}{len(tr):>10}{len(ct):>10}"
                     f"{pairs:>11}{beaten:>16}")
    lines.append("-" * 92)
    lines.append(f"  Total inversion pairs (control nll < trained nll) = {tot_pairs}; "
                 f"trained targets beaten by best control = {tot_beaten}.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# TABLE 2 (probe spectrum) + substring-inflation guard
# ---------------------------------------------------------------------------

def table2_probe_spectrum(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    base = _main_rows(df)
    lines = ["=" * 92,
             "TABLE 2: probe spectrum (ordered by expressivity; softprompt = forcing UPPER bound)",
             "  A more expressive probe raises EMR(C); when Adj~0 the probe is pure FORCING.",
             "=" * 92]
    lines.append(f"{'probe':<16}{'k':>5}{'EMR(D)':>9}{'EMR(C)':>9}{'Adj[CI]':>22}{'n(D)':>7}")
    lines.append("-" * 92)
    probes = sorted(base["probe"].dropna().unique(),
                    key=lambda p: (-_EXPRESSIVITY.get(p, 0.0), p))
    csv_rows = []
    for probe in probes:
        gp = base[base["probe"] == probe]
        krep = gp["capacity_k"].dropna()
        k_str = str(int(krep.mode().iloc[0])) if len(krep) else "-"
        d = _cluster_emr_ci(_person_groups(gp[gp["target_membership"] == "trained"]))
        c = _cluster_emr_ci(_person_groups(gp[gp["target_membership"] == "control"]))
        adj = _cluster_diff_ci(
            _person_groups(gp[gp["target_membership"] == "trained"]),
            _person_groups(gp[gp["target_membership"] == "control"]))
        adj_str = f"{adj['diff']*100:+.1f}{_fmt_ci(adj['ci_low'], adj['ci_high'])}"
        lines.append(f"{probe:<16}{k_str:>5}{_pct(d['emr']):>9}{_pct(c['emr']):>9}"
                     f"{adj_str:>22}{d['n_persons']:>7}")
        csv_rows.append({"probe": probe, "capacity_k": k_str,
                         "emr_trained": d["emr"], "emr_control": c["emr"],
                         "adj": adj["diff"], "adj_lo": adj["ci_low"],
                         "adj_hi": adj["ci_high"], "n_persons_trained": d["n_persons"]})
    lines.append("")
    return lines, pd.DataFrame(csv_rows)


def substring_guard(df: pd.DataFrame) -> List[str]:
    base = _main_rows(df)
    lines = ["=" * 92,
             "SUBSTRING-INFLATION GUARD: random_record_match vs exact_match per probe",
             "  A high random_record_match means the matcher fires on UNRELATED records => spurious.",
             "=" * 92]
    lines.append(f"{'probe':<16}{'exact_match':>14}{'random_match':>16}{'ratio(rand/exact)':>20}")
    lines.append("-" * 92)
    for probe in sorted(base["probe"].dropna().unique()):
        gp = base[base["probe"] == probe]
        em = _boolcol(gp, "exact_match").mean()
        rm = _boolcol(gp, "random_record_match").mean()
        ratio = (rm / em) if em > 0 else float("nan")
        ratio_s = "n/a" if math.isnan(ratio) else f"{ratio:.3f}"
        lines.append(f"{probe:<16}{_pct(em):>13}%{_pct(rm):>15}%{ratio_s:>20}")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# E7 budget-matched control: is GCG's gain optimization, or just more queries?
# ---------------------------------------------------------------------------

def budget_e7(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    e7 = df[df["exp_id"] == "E7"]
    lines = ["=" * 100,
             "E7 BUDGET-MATCHED CONTROL: optimization vs query budget",
             "  fixed=1 query; fixed_budget=natural prompts sampled at gcg_free's OWN forward-pass",
             "  budget; gcg_free=optimized at that budget. If fixed_budget ~ fixed << gcg_free, the",
             "  gain is OPTIMIZATION, not queries. (fwd = mean forward_passes = the matched budget.)",
             "=" * 100]
    if len(e7) == 0:
        lines.append("  (no E7 rows in log)")
        lines.append("")
        return lines, pd.DataFrame(columns=["model_name", "probe", "emr_trained",
                                            "emr_control", "mean_forward_passes"])
    csv_rows = []
    order = {"gcg_free": 0, "fixed_budget": 1, "fixed": 2}
    for model, gm in e7.groupby("model_name", sort=True):
        lines.append(f"  model = {model}")
        lines.append(f"    {'probe':<16}{'EMR(D)':>9}{'EMR(C)':>9}{'Adj[CI]':>22}"
                     f"{'mean fwd':>10}{'n(D)':>7}")
        lines.append("    " + "-" * 82)
        probes = sorted(gm["probe"].dropna().unique(),
                        key=lambda p: (order.get(p, 9), p))
        for probe in probes:
            gp = gm[gm["probe"] == probe]
            d = _cluster_emr_ci(_person_groups(gp[gp["target_membership"] == "trained"]))
            c = _cluster_emr_ci(_person_groups(gp[gp["target_membership"] == "control"]))
            adj = _cluster_diff_ci(
                _person_groups(gp[gp["target_membership"] == "trained"]),
                _person_groups(gp[gp["target_membership"] == "control"]))
            fwd = float(gp["forward_passes"].dropna().astype(float).mean()) \
                if gp["forward_passes"].notna().any() else float("nan")
            adj_str = f"{adj['diff']*100:+.1f}{_fmt_ci(adj['ci_low'], adj['ci_high'])}"
            fwd_str = "n/a" if math.isnan(fwd) else f"{fwd:.0f}"
            lines.append(f"    {probe:<16}{_pct(d['emr']):>9}{_pct(c['emr']):>9}"
                         f"{adj_str:>22}{fwd_str:>10}{d['n_persons']:>7}")
            csv_rows.append({"model_name": model, "probe": probe,
                             "emr_trained": d["emr"], "emr_control": c["emr"],
                             "adj": adj["diff"], "mean_forward_passes": fwd})
        lines.append("")
    return lines, pd.DataFrame(csv_rows)


# ---------------------------------------------------------------------------
# E10 Pythia + Pile (external validity): forcing on a model/corpus we did not make
# ---------------------------------------------------------------------------

def pile_e10(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    e10 = df[df["exp_id"] == "E10"]
    lines = ["=" * 100,
             "E10 PYTHIA + THE PILE: forcing-vs-memorization on a model we did NOT train",
             "  member=string present in the sampled Pile (count>0); control=format-matched, absent.",
             "  fixed=real-context completion (MEMORIZATION baseline); gcg_free=force raw string (FORCING).",
             "  Adj=EMR(member)-EMR(control); gcg_free Adj~0 => forcing replicates on real data.",
             "=" * 100]
    if len(e10) == 0:
        lines.append("  (no E10 rows in log)")
        lines.append("")
        return lines, pd.DataFrame(columns=["model_name", "probe", "emr_member",
                                            "emr_control", "adj", "adj_lo", "adj_hi"])
    csv_rows = []
    for model, gm in e10.groupby("model_name", sort=True):
        lines.append(f"  model = {model}")
        lines.append(f"    {'probe':<16}{'EMR(mem)':>10}{'EMR(ctrl)':>11}"
                     f"{'Adj[CI]':>22}{'AUC':>8}{'n(mem)':>8}")
        lines.append("    " + "-" * 84)
        probes = sorted(gm["probe"].dropna().unique(),
                        key=lambda p: (-_EXPRESSIVITY.get(p, 0.0), p))
        for probe in probes:
            gp = gm[gm["probe"] == probe]
            d = _cluster_emr_ci(_person_groups(gp[gp["target_membership"] == "trained"]))
            c = _cluster_emr_ci(_person_groups(gp[gp["target_membership"] == "control"]))
            adj = _cluster_diff_ci(
                _person_groups(gp[gp["target_membership"] == "trained"]),
                _person_groups(gp[gp["target_membership"] == "control"]))
            audit = _audit_ci(gp)
            adj_str = f"{adj['diff']*100:+.1f}{_fmt_ci(adj['ci_low'], adj['ci_high'])}"
            auc_str = "n/a" if math.isnan(audit["auc"]) else f"{audit['auc']:.3f}"
            lines.append(f"    {probe:<16}{_pct(d['emr']):>10}{_pct(c['emr']):>11}"
                         f"{adj_str:>22}{auc_str:>8}{d['n_persons']:>8}")
            csv_rows.append({"model_name": model, "probe": probe,
                             "emr_member": d["emr"], "emr_control": c["emr"],
                             "adj": adj["diff"], "adj_lo": adj["ci_low"],
                             "adj_hi": adj["ci_high"], "auc": audit["auc"]})
        lines.append("")
    return lines, pd.DataFrame(csv_rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _write_csv(dfout: pd.DataFrame, name: str) -> str:
    path = os.path.join(TABLES_DIR, name)
    dfout.to_csv(path, index=False, float_format="%.6f")
    return path


def _write_txt(lines: List[str], name: str) -> str:
    path = os.path.join(TABLES_DIR, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def build_all(run_id: str) -> int:
    os.makedirs(TABLES_DIR, exist_ok=True)
    df = attempt_log.load_attempts(run_id)

    if df is None or len(df) == 0:
        msg = (f"[make_tables] No attempts found for run_id={run_id!r} in "
               f"{attempt_log.ATTEMPTS_DIR}. Nothing to aggregate; run the "
               f"experiments first. (Wrote nothing.)")
        print(msg)
        return 0

    print(f"[make_tables] loaded {len(df)} attempts for run_id={run_id!r} "
          f"({df['exp_id'].nunique()} exp, {df['model_name'].nunique()} models, "
          f"{df['probe'].nunique()} probes)")

    t1_lines, t1_csv = table1_main(df)
    cap_lines, cap_csv = capacity_e3(df)
    freq_lines, freq_csv = frequency_e5(df)
    acr_lines = acr_e13(df)
    inv_lines = rank_inversion_e16(df)
    t2_lines, t2_csv = table2_probe_spectrum(df)
    guard_lines = substring_guard(df)
    budget_lines, budget_csv = budget_e7(df)
    pile_lines, pile_csv = pile_e10(df)

    # text tables
    _write_txt(t1_lines, "table1_main.txt")
    _write_txt(t2_lines, "table2_probe_spectrum.txt")
    _write_txt(cap_lines, "capacity_e3.txt")
    _write_txt(freq_lines, "frequency_e5.txt")
    _write_txt(acr_lines, "acr_e13.txt")
    _write_txt(inv_lines, "rank_inversion_e16.txt")
    _write_txt(guard_lines, "substring_guard.txt")
    _write_txt(budget_lines, "budget_e7.txt")
    _write_txt(pile_lines, "pile_e10.txt")

    # tidy CSVs (figure inputs)
    _write_csv(t1_csv, "table1_main.csv")
    _write_csv(t2_csv, "table2_probe_spectrum.csv")
    _write_csv(cap_csv, "capacity_curve.csv")
    _write_csv(freq_csv, "frequency_curve.csv")
    _write_csv(budget_csv, "budget_e7.csv")
    _write_csv(pile_csv, "pile_e10.csv")

    # combined report (stdout + results/summary_tables.txt for continuity)
    all_lines: List[str] = []
    for block in (t1_lines, t2_lines, cap_lines, freq_lines,
                  acr_lines, inv_lines, guard_lines, budget_lines, pile_lines):
        all_lines.extend(block)
    report = "\n".join(all_lines)
    print(report)
    with open(os.path.join(RESULTS_DIR, "summary_tables.txt"), "w") as f:
        f.write(report + "\n")

    print(f"[make_tables] wrote 9 txt + 6 csv to {TABLES_DIR} and "
          f"results/summary_tables.txt")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="run1",
                    help="run_id whose attempt shards to aggregate (default: run1)")
    args = ap.parse_args()
    raise SystemExit(build_all(args.run_id))


if __name__ == "__main__":
    main()
