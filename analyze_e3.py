"""Historical E3 analysis helpers; the command-line entry point is retired.

The raw-data audit found invalid boundary intervals, a post-result H2 change,
and an incomplete Holm family. The former claim that all implementation
preceded the full curve is withdrawn. See the source-linked replacement at
.ai/research/studies/capacity_axis_20260902/reanalysis/recompute.py and its
method_choices.md. The replacement is conditional, pending provenance.
"""
import argparse, glob, json, math
import numpy as np
import pandas as pd

import curve_stats as cs
import survival as sv

ALPHA_TOL = 0.01          # H2 as preregistered
H3_K = 20                 # the densified point named in H3
ALPHA_MAP = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]   # the auditor-facing mapping
FWER = 0.05
H_INF = 29.9        # SSN format min-entropy (Prop. 1 / Cor. 1)
LOG2_V = 15.62      # log2|V| for the GPT-2 vocabulary


def load(pattern) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shards matched {pattern}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df, len(files)


def boot_p_two_sided(reps, null=0.0) -> float:
    """Percentile bootstrap p: twice the smaller tail mass about `null`."""
    a = np.asarray(reps, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    below = (a <= null).mean()
    above = (a >= null).mean()
    return float(min(1.0, 2 * min(below, above)))


def holm(pvals: dict, fwer=FWER) -> dict:
    """Holm-Bonferroni step-down. Returns adjusted p and the reject decision."""
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p_raw": p, "p_holm": adj, "reject_at_%.2f" % fwer: bool(adj <= fwer)}
    for k, v in pvals.items():
        if not np.isfinite(v):
            out[k] = {"p_raw": None, "p_holm": None,
                      "reject_at_%.2f" % fwer: None,
                      "note": "no p-value defined; scored by its CI rule instead"}
    return out


def kmin_table(e3: pd.DataFrame, ks_pos) -> pd.DataFrame:
    """One row per target. k=0 is a DIFFERENT probe (fixed, no GCG) and is excluded."""
    e = e3[e3["capacity_k"].isin(ks_pos)]
    rows = []
    for keyvals, g in e.groupby(["person_id", "field", "target_membership"], sort=True):
        hit = g[g["exact_match"].fillna(False).astype(bool)]
        rows.append({
            "person_id": keyvals[0], "field": keyvals[1], "arm": keyvals[2],
            "k_min": float(hit["capacity_k"].min()) if len(hit) else float("nan"),
            "H_bits": float(g["target_H_bits"].dropna().iloc[0])
            if g["target_H_bits"].notna().any() else float("nan"),
            "len_tokens": float(g["target_len_tokens"].dropna().iloc[0])
            if g["target_len_tokens"].notna().any() else float("nan"),
        })
    return pd.DataFrame(rows)


def hits_by_k(e3: pd.DataFrame, ks_pos) -> dict:
    """target -> {k: was it forced at any seed}. Feeds the absorbing-state check."""
    e = e3[e3["capacity_k"].isin(ks_pos)]
    out = {}
    for keyvals, g in e.groupby(["person_id", "field", "target_membership"], sort=True):
        key = "|".join(map(str, keyvals))
        out[key] = {int(k): bool(gg["exact_match"].fillna(False).astype(bool).any())
                    for k, gg in g.groupby("capacity_k")}
    return out


def analyze(df: pd.DataFrame, n_shards: int, n_boot: int = cs.N_BOOT) -> dict:
    ks_all = sorted(int(k) for k in df["capacity_k"].dropna().unique())
    ks_pos = [k for k in ks_all if k > 0]
    R = {"meta": {"n_shards": n_shards, "n_rows": int(len(df)),
                  "k_grid": ks_all, "k_grid_positive": ks_pos,
                  "seeds": sorted(int(s) for s in df["seed"].unique()),
                  "n_boot": n_boot, "alpha_tolerance": ALPHA_TOL,
                  "family": ["H1", "H2", "H3", "H4"], "fwer": FWER,
                  "h5": "exploratory, outside the family"}}

    # ---- k=0 anchor, read and reported FIRST (success criterion 1) ----
    z = df[df["capacity_k"] == 0]
    R["k0_anchor"] = {
        "alpha_0": float(z[z.target_membership == "control"].exact_match.mean()),
        "emr_d_0": float(z[z.target_membership == "trained"].exact_match.mean()),
        "n_per_arm": int(len(z[z.target_membership == "control"])),
        "probe": "fixed (no GCG) -- a different probe from k>=1, excluded from k_min"}

    # ---- the shared bootstrap: one person draw per block, all k ----
    boot = cs.joint_bootstrap(df, ks_pos, n_boot=n_boot)
    a_lo, a_hi = cs.pct_ci(boot["alpha"]); t_lo, t_hi = cs.pct_ci(boot["tau"])
    R["curve"] = [{"k": int(k),
                   "alpha": float(boot["alpha_point"][j]),
                   "alpha_ci": [float(a_lo[j]), float(a_hi[j])],
                   "emr_d": float(boot["emr_d_point"][j]),
                   "tau": float(boot["emr_d_point"][j] - boot["alpha_point"][j]),
                   "tau_ci": [float(t_lo[j]), float(t_hi[j])],
                   "n_attempts_c": int(boot["n_attempts"]["control"][j]),
                   "n_attempts_d": int(boot["n_attempts"]["trained"][j])}
                  for j, k in enumerate(ks_pos)]
    R["curve_isotonic_alpha"] = [float(v) for v in cs.isotonic(boot["alpha_point"])]
    R["n_persons"] = boot["n_persons"]

    # ---- H1: monotone trend in the floor ----
    tt = cs.trend_test(boot)
    rho_reps = np.array([cs.spearman(boot["ks"], boot["alpha"][b])
                         for b in range(boot["n_boot"])])
    R["H1"] = {**tt, "p_boot": boot_p_two_sided(rho_reps),
               "supported": bool(tt["excludes_zero"] and tt["rho"] > 0)}

    # ---- H2: a usable k exists (floor AND signal) ----
    up = cs.usable_points(boot, ALPHA_TOL)
    cross = cs.crossing_k(ks_pos, boot["alpha_point"], ALPHA_TOL)
    # bootstrap p for the existence claim: replicates with NO k meeting the floor
    none_meet = np.array([(boot["alpha"][b] > ALPHA_TOL).all()
                          for b in range(boot["n_boot"])], float)
    R["H2"] = {
        # The PREREGISTERED H0 is about the floor alone: "no k >= 1 satisfies
        # alpha_k <= 1%". That is what carries the p-value into Holm.
        "preregistered": {
            "criterion": "some k >= 1 has alpha_k <= %.3f" % ALPHA_TOL,
            "any_k_meets_floor": bool((boot["alpha_point"] <= ALPHA_TOL).any()),
            "k_meeting_floor_point": [int(k) for j, k in enumerate(ks_pos)
                                      if boot["alpha_point"][j] <= ALPHA_TOL],
            "k_meeting_floor_ci_upper": [int(k) for j, k in enumerate(ks_pos)
                                         if a_hi[j] <= ALPHA_TOL],
            "p_boot_no_usable_k": float(none_meet.mean()),
            "supported": bool((boot["alpha_point"] <= ALPHA_TOL).any())},
        # The RE-AIMED overlay adds the signal condition. A floor condition alone
        # is satisfied wherever both arms are zero, which is not an audit.
        "reaimed": {"criterion": "floor meets tolerance AND tau's CI excludes 0",
                    "rows": up["rows"], "any_usable": up["any_usable"], "note": up["note"]},
        "crossing_up_through_tolerance": cross,
        "supported": bool((boot["alpha_point"] <= ALPHA_TOL).any())}

    # ---- the deliverable: alpha -> k*(alpha) ----
    # alpha_k RISES with k, so "floor <= tolerance" bounds k from ABOVE. The
    # auditor's k*(alpha) is therefore the LARGEST capacity whose floor still
    # meets the tolerance -- the operating ceiling, not a starting point. Taking
    # the smallest such k would return k=1 at every tolerance and say nothing.
    R["alpha_to_kstar"] = []
    for tol in ALPHA_MAP:
        ok_pt = [int(k) for j, k in enumerate(ks_pos) if boot["alpha_point"][j] <= tol]
        ok_ci = [int(k) for j, k in enumerate(ks_pos) if a_hi[j] <= tol]
        R["alpha_to_kstar"].append({
            "alpha_tolerance": tol,
            "k_max_point_rule": max(ok_pt) if ok_pt else None,
            "k_max_ci_upper_rule": max(ok_ci) if ok_ci else None,
            "k_theory_cor1": (H_INF - math.log2(1 / tol)) / LOG2_V,
            "breaks_tolerance_at": cs.crossing_k(ks_pos, boot["alpha_point"], tol),
            "note": None if ok_ci else
            "no capacity on this grid keeps the floor at or below this tolerance "
            "once sampling error is accounted for"})

    # ---- H3: the calibrated signal at the densified point ----
    if H3_K in ks_pos:
        j = ks_pos.index(H3_K)
        reps = boot["tau"][:, j]
        R["H3"] = {"k": H3_K, "tau": float(boot["emr_d_point"][j] - boot["alpha_point"][j]),
                   "tau_ci": [float(t_lo[j]), float(t_hi[j])],
                   "p_boot": boot_p_two_sided(reps),
                   "excludes_zero": bool(t_lo[j] > 0 or t_hi[j] < 0),
                   "half_width": float((t_hi[j] - t_lo[j]) / 2),
                   "reading": ("CI EXCLUDES 0 -- a measurable signal at this capacity"
                               if (t_lo[j] > 0 or t_hi[j] < 0) else
                               "CI CONTAINS 0 -- the bound is the result: no effect larger "
                               f"than {max(abs(t_lo[j]), abs(t_hi[j])):.3f} in absolute value")}

    # ---- H4: the forcing model, censored, control arm only ----
    km = kmin_table(df, ks_pos)
    kc = km[(km.arm == "control") & np.isfinite(km.H_bits) & (km.H_bits > 0)]
    hbk = {k: v for k, v in hits_by_k(df, ks_pos).items() if k.endswith("|control")}
    R["H4_inputs"] = {
        "n_targets_control": int(len(kc)),
        "right_censored_fraction": sv.right_censored_fraction(kc.k_min.tolist()),
        "non_monotone_fraction": sv.non_monotone_fraction(hbk),
        "H_bits": {"min": float(kc.H_bits.min()), "median": float(kc.H_bits.median()),
                   "max": float(kc.H_bits.max())}}
    try:
        fit = sv.fit_loglog(kc.k_min.tolist(), kc.H_bits.tolist(),
                            kc.person_id.tolist(), ks_pos)
        g_lo, g_hi = fit["gamma_ci"]
        holds = fit["h4_proportionality_supported"]
        # beta = exp(-intercept) parameterizes a model the slope test may have
        # just refuted. When gamma's CI excludes 1 the intercept is absorbing
        # the misfit and beta is not a rate in bits/token at all -- it must not
        # be quoted. Enforced here rather than left to whoever reads the JSON.
        R["H4"] = {**fit, "supported": holds,
                   "scored_by": "CI rule: gamma's 95% CI must contain 1",
                   "beta_interpretable": bool(holds),
                   "beta_caveat": None if holds else
                   "DO NOT QUOTE beta_bits_per_token. gamma's CI excludes 1, so the "
                   "proportional model is refuted and exp(-intercept) is absorbing the "
                   "slope misfit rather than measuring a steering rate. Use "
                   "beta_model_free below."}
    except Exception as e:
        R["H4"] = {"error": f"{type(e).__name__}: {e}", "supported": None}

    # ---- beta the auditor can actually use, free of the refuted model ----
    # Right censoring is what motivated the AFT fit; where it is absent the
    # direct per-target ratio needs no model. Interval censoring remains, so
    # each target gets a bracket: k at its observed grid point (conservative,
    # beta low) and at the previous one (optimistic, beta high).
    prev = {v: (ks_pos[i - 1] if i else 0) for i, v in enumerate(ks_pos)}
    kf = kc[np.isfinite(kc.k_min)].copy()
    kf["beta_lo"] = kf.H_bits / kf.k_min
    kf["beta_hi"] = kf.H_bits / kf.k_min.map(lambda v: max(prev[int(v)], 1e-9))
    R["beta_model_free"] = {
        "definition": "beta(t) = H(t) / k_min(t), control arm, no model",
        "this_is_the_papers_definition": "Paper Def. 3 IS median H(t)/k_min(t) in bits/token "
            "(CODE_MAP.md mismatch #1). It is not a fallback for the refuted AFT fit -- it is the "
            "quantity the paper defines, which capacity_e3 currently computes inverted as "
            "linregress(H_bits -> k_min).slope, in tokens/bit.",
        "ceiling_log2_V": LOG2_V,
        "applies_because_right_censoring_is": float(fit["right_censored_fraction"])
        if "error" not in R["H4"] else None,
        "overall": {"n": int(len(kf)),
                    "median_lo": float(kf.beta_lo.median()),
                    "median_hi": float(kf.beta_hi.median()),
                    "iqr_lo": [float(kf.beta_lo.quantile(.25)), float(kf.beta_lo.quantile(.75))],
                    "range_lo": [float(kf.beta_lo.min()), float(kf.beta_lo.max())],
                    "p10_lo": float(kf.beta_lo.quantile(.10))},
        "by_field": {f: {"n": int(len(g)), "median_lo": float(g.beta_lo.median()),
                         "median_hi": float(g.beta_hi.median()),
                         "H_median": float(g.H_bits.median()),
                         "k_min_median": float(g.k_min.median()),
                         "spearman_H_kmin": float(g.H_bits.corr(g.k_min, method="spearman"))}
                     for f, g in kf.groupby("field")},
        "note": "Field-stratified because the pooled figure is a between-field artefact: "
                "see spearman_H_kmin within each field."}

    # ---- is Proposition 1's bound tight, evaluated at each target's own H(t)? ----
    kf["bound"] = kf.H_bits / LOG2_V
    ratio = kf.k_min / kf["bound"]
    early = kf[kf.k_min == kf.k_min.min()]
    R["bound_tightness"] = {
        "bound": "k_min(t) >= H(t) / log2|V|  -- Prop. 1 at the target's own self-information",
        "n": int(len(kf)),
        "bound_range": [float(kf["bound"].min()), float(kf["bound"].max())],
        "n_violating": int((kf.k_min < kf["bound"]).sum()),
        "ratio_k_over_bound": {"min": float(ratio.min()), "median": float(ratio.median()),
                               "max": float(ratio.max())},
        "earliest_forced": {"k_min": float(kf.k_min.min()), "n": int(len(early)),
                            "H_bits": [float(v) for v in early.H_bits],
                            "their_bounds": [float(v) for v in early["bound"]],
                            "H_bits_min_in_set": float(kf.H_bits.min())},
        "note": "H_inf (Cor. 1, 29.9 bits) and H(t) (here, 55-81 bits) are different quantities "
                "and design.md L793-808 forbids interchanging them. This row uses H(t), which is "
                "what governs forcing an actual rendered string; the alpha_to_kstar table uses "
                "H_inf, which is what Corollary 1 is stated in."}

    # ---- t3-2: covariate balance between the arms ----
    # design.md promises E17-matched controls at |SMD| < 0.1. The matching runs
    # on the full pools; the arms are then subset to 25 INDEPENDENTLY of each
    # other (_matched_control_entries keeps only the control NAMES from E17's
    # pairs and drops the pairing), so nothing preserves the match through the
    # subset. This measures what actually survived.
    def _smd(d, c):
        d, c = np.asarray(d, float), np.asarray(c, float)
        sp = np.sqrt((d.var(ddof=1) + c.var(ddof=1)) / 2)
        return float((d.mean() - c.mean()) / sp) if sp > 0 else 0.0

    # k=0 must be excluded: the fixed probe targets a DIFFERENT string. It forces
    # 'qjohnson@example.com' while gcg_free forces 'Email: qjohnson@example.com'
    # with the field prefix, so target_H_bits and target_len_tokens carry two
    # values per target. Reading the k=0 one here put every SMD on the wrong
    # quantity (SSN entropy 49.7 instead of 73.4).
    tgt = (df[df.capacity_k > 0]
           .drop_duplicates(["person_id", "field", "target_membership"]))
    bal, unbalanced = {}, []
    for f, g in tgt.groupby("field"):
        D, C = g[g.target_membership == "trained"], g[g.target_membership == "control"]
        bal[f] = {}
        for cov in ("target_H_bits", "target_len_tokens"):
            v = _smd(D[cov].dropna(), C[cov].dropna())
            bal[f][cov] = {"smd": v, "balanced": bool(abs(v) < 0.1),
                           "mean_D": float(D[cov].mean()), "mean_C": float(C[cov].mean())}
            if abs(v) >= 0.1:
                unbalanced.append(f"{f}/{cov}={v:+.3f}")
    R["balance"] = {
        "threshold": 0.1, "by_field": bal, "unbalanced": unbalanced,
        "passes": not unbalanced,
        "note": "field is matched exactly by construction. char_len needs "
                "results/e17_matches_*.json and is not checked here.",
        "consequence": None if not unbalanced else
        "Every D-vs-C quantity (H3, H5, the tau curve) is confounded in the affected field(s). "
        "Control-arm-only results -- H1, H2's floor, H4, bound_tightness, beta_model_free -- "
        "are untouched, because they never compare the arms."}

    # ---- H3 stratified, so the confound's reach is visible ----
    if H3_K in ks_pos:
        R["H3_by_field"] = {}
        for f, g in df[df.capacity_k == H3_K].groupby("field"):
            c_ = g[g.target_membership == "control"].exact_match.fillna(False)
            d_ = g[g.target_membership == "trained"].exact_match.fillna(False)
            lo, hi = cs.newcombe_diff_ci(d_.sum(), len(d_), c_.sum(), len(c_))
            R["H3_by_field"][f] = {
                "alpha": float(c_.mean()), "emr_d": float(d_.mean()),
                "tau": float(d_.mean() - c_.mean()), "tau_ci_mover": [lo, hi],
                "n_per_arm": int(len(c_)),
                "balanced": all(v["balanced"] for v in bal.get(f, {}).values())}
        R["H3_by_field"]["_note"] = (
            "ROBUSTNESS CHECK, not preregistered. Reported so the balance defect's reach is "
            "visible; the preregistered H3 remains the pooled estimate above.")

    # ---- H5: exploratory ----
    R["H5"] = {**cs.peak_location(boot), "status": "EXPLORATORY -- outside the family"}

    # ---- Holm over the confirmatory family ----
    pv = {"H1": R["H1"].get("p_boot", float("nan")),
          "H2": R["H2"]["preregistered"]["p_boot_no_usable_k"],
          "H3": R["H3"].get("p_boot", float("nan")) if "H3" in R else float("nan"),
          "H4": float("nan")}          # scored by its CI rule, no p defined
    R["holm"] = holm(pv)
    return R


def main():
    raise SystemExit(
        "This historical analyzer is superseded: its boundary intervals and H2/Holm "
        "decisions failed the reanalysis audit. Use the ledger-checked entry point "
        ".ai/research/studies/capacity_axis_20260902/reanalysis/recompute.py. "
        "See reanalysis/method_choices.md for the conditional inference limits."
    )


if __name__ == "__main__":
    main()
