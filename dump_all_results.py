"""
Dump EVERY analysis extractable from the per-attempt log into ONE markdown file.

No new experiments — these are additional CUTS of the run you already have, so a
"too few experiments" dataset yields many defensible tables:

  1. Overview (models / probes / fields / seeds / counts)
  2. Main per-model (EMR D / C / Adj[CI] / AUC)               [= Table 1]
  3. Per-model x per-probe EMR(D)/EMR(C)/Adj                  [4 x 8 = lots]
  4. Per-field (ssn/email) x per-probe                        [field breakdown]
  5. Frequency dose-response from E1 (EMR by train_frequency) [poor-man's E5, no run]
  6. Substring-inflation guard (random_record_match vs exact) [robustness]
  7. Rank inversion (control scoring >= trained)              [robustness]
  8. Compute profile (forward_passes / wallclock per probe)   [cost table]

Usage:  python dump_all_results.py --run-id run2
Writes: results/ALL_RESULTS.md
"""
import argparse
import os
import numpy as np
import pandas as pd

import attempt_log
from config import RESULTS_DIR
from make_tables import (
    _cluster_emr_ci, _cluster_diff_ci, _person_groups, _audit_ci,
    _boolcol, _pct, _fmt_ci, _EXPRESSIVITY,
)


def _adj(g):
    d = _cluster_emr_ci(_person_groups(g[g["target_membership"] == "trained"]))
    c = _cluster_emr_ci(_person_groups(g[g["target_membership"] == "control"]))
    a = _cluster_diff_ci(
        _person_groups(g[g["target_membership"] == "trained"]),
        _person_groups(g[g["target_membership"] == "control"]))
    return d, c, a


def _adjstr(a):
    if np.isnan(a["diff"]):
        return "n/a"
    return f"{a['diff']*100:+.1f} {_fmt_ci(a['ci_low'], a['ci_high'])}"


def _probe_order(probes):
    return sorted(probes, key=lambda p: (-_EXPRESSIVITY.get(p, 0.0), p))


def build(run_id: str) -> str:
    df = attempt_log.load_attempts(run_id)
    L = []
    w = L.append
    if df is None or len(df) == 0:
        return f"# No attempts found for run_id={run_id!r}\n"

    e1 = df[df["exp_id"] == "E1"] if (df["exp_id"] == "E1").any() else df
    models = sorted(df["model_name"].dropna().unique())
    probes = _probe_order(e1["probe"].dropna().unique())
    fields = sorted(e1["field"].dropna().unique())
    seeds = sorted(int(s) for s in df["seed"].dropna().unique())

    w(f"# ALL extractable results — run `{run_id}`")
    w("")
    w("_Every table below is a different cut of the SAME per-attempt log "
      "(no extra experiments). EMR in %, 95% person-clustered bootstrap CIs. "
      "EMR(D)=trained, EMR(C)=never-trained control, Adj=EMR(D)−EMR(C)._")
    w("")

    # 1. Overview
    w("## 1. Overview")
    w("")
    w(f"- Models: {', '.join(models)}")
    w(f"- Probes: {', '.join(probes)}")
    w(f"- Fields: {', '.join(fields)}")
    w(f"- Seeds: {seeds}")
    w(f"- Total attempts: {len(df)}  |  E1 attempts: {len(e1)}")
    nt = df[df["target_membership"] == "trained"].groupby("model_name")["person_id"].nunique()
    nc = df[df["target_membership"] == "control"].groupby("model_name")["person_id"].nunique()
    w("")
    w("| Model | n_trained | n_control |")
    w("|---|---|---|")
    for m in models:
        w(f"| {m} | {int(nt.get(m, 0))} | {int(nc.get(m, 0))} |")
    w("")

    # 2. Main per-model (confirmation probe gcg_free)
    w("## 2. Main result — per model (`gcg_free`)")
    w("")
    w("| Model | Fixed | GCG=EMR(D) | Neg-ctrl=EMR(C) | Adj [95% CI] | AUC [CI] | TPR@1% | TPR@5% |")
    w("|---|---|---|---|---|---|---|---|")
    for m in models:
        g = e1[e1["model_name"] == m]
        gf = g[g["probe"] == "gcg_free"]
        fx = g[g["probe"] == "fixed"]
        d, c, a = _adj(gf)
        fxd = _cluster_emr_ci(_person_groups(fx[fx["target_membership"] == "trained"]))
        au = _audit_ci(gf)
        aucs = "n/a" if np.isnan(au["auc"]) else f"{au['auc']:.2f} [{au['auc_lo']:.2f},{au['auc_hi']:.2f}]"
        w(f"| {m} | {_pct(fxd['emr'])} | {_pct(d['emr'])} | {_pct(c['emr'])} | "
          f"{_adjstr(a)} | {aucs} | {_pct(au['tpr1'])} | {_pct(au['tpr5'])} |")
    w("")

    # 3. Per-model x per-probe
    w("## 3. Per-model × per-probe (EMR D / C / Adj)")
    w("")
    for m in models:
        g = e1[e1["model_name"] == m]
        w(f"### {m}")
        w("")
        w("| Probe | EMR(D) | EMR(C) | Adj [95% CI] | n(D) |")
        w("|---|---|---|---|---|")
        for p in _probe_order(g["probe"].dropna().unique()):
            gp = g[g["probe"] == p]
            d, c, a = _adj(gp)
            w(f"| {p} | {_pct(d['emr'])} | {_pct(c['emr'])} | {_adjstr(a)} | {d['n_persons']} |")
        w("")

    # 4. Per-field x per-probe (pooled over models)
    w("## 4. Per-field × per-probe (pooled over models)")
    w("")
    for fld in fields:
        gfld = e1[e1["field"] == fld]
        w(f"### field = {fld}")
        w("")
        w("| Probe | EMR(D) | EMR(C) | Adj [95% CI] |")
        w("|---|---|---|---|")
        for p in _probe_order(gfld["probe"].dropna().unique()):
            gp = gfld[gfld["probe"] == p]
            d, c, a = _adj(gp)
            w(f"| {p} | {_pct(d['emr'])} | {_pct(c['emr'])} | {_adjstr(a)} |")
        w("")

    # 5. Frequency dose-response from E1 (no E5 needed)
    w("## 5. Frequency dose–response (from E1 `gcg_free`; freq 0 = control floor)")
    w("")
    w("_train_frequency = how many times the record appeared in fine-tuning. "
      "freq 0 = never-trained control. Rising EMR with frequency = memorization; "
      "a flat line at the freq-0 floor = pure forcing._")
    w("")
    gf = e1[e1["probe"] == "gcg_free"]
    freqs = sorted(int(f) for f in gf["train_frequency"].dropna().unique())
    w("| Model | " + " | ".join(f"freq={f}" for f in freqs) + " |")
    w("|---|" + "---|" * len(freqs))
    for m in models:
        gm = gf[gf["model_name"] == m]
        cells = []
        for f in freqs:
            sub = gm[gm["train_frequency"] == f]
            mm = _cluster_emr_ci(_person_groups(sub))
            cells.append(_pct(mm["emr"]))
        w(f"| {m} | " + " | ".join(cells) + " |")
    # pooled
    cells = []
    for f in freqs:
        sub = gf[gf["train_frequency"] == f]
        mm = _cluster_emr_ci(_person_groups(sub))
        cells.append(_pct(mm["emr"]))
    w("| **pooled** | " + " | ".join(f"**{c}**" for c in cells) + " |")
    w("")

    # 6. Substring-inflation guard
    w("## 6. Substring-inflation guard (random_record_match vs exact_match)")
    w("")
    w("_High random_record_match would mean the matcher fires on UNRELATED records "
      "(spurious). It should be ≪ exact_match._")
    w("")
    w("| Probe | exact_match % | random_record_match % |")
    w("|---|---|---|")
    for p in probes:
        gp = e1[e1["probe"] == p]
        em = _boolcol(gp, "exact_match").mean() * 100
        rm = _boolcol(gp, "random_record_match").mean() * 100
        w(f"| {p} | {em:.1f} | {rm:.1f} |")
    w("")

    # 7. Rank inversion (control nll < trained nll)
    w("## 7. Rank inversion (`gcg_free`: control target scoring ≥ a trained one)")
    w("")
    g = e1[(e1["probe"] == "gcg_free") & e1["final_target_nll"].notna()]
    if len(g):
        best = (g.groupby(["person_id", "field", "target_membership"], sort=True)
                ["final_target_nll"].min().reset_index())
        tot_pairs = tot_beaten = 0
        rows = []
        for fld, gg in best.groupby("field", sort=True):
            tr = gg[gg["target_membership"] == "trained"]["final_target_nll"].to_numpy()
            ct = gg[gg["target_membership"] == "control"]["final_target_nll"].to_numpy()
            if len(tr) == 0 or len(ct) == 0:
                continue
            pairs = int(np.sum(ct[:, None] < tr[None, :]))
            beaten = int(np.sum(np.min(ct) < tr))
            tot_pairs += pairs
            tot_beaten += beaten
            rows.append((fld, len(tr), len(ct), pairs, beaten))
        w("| field | n_trained | n_control | inversion pairs | trained beaten by best control |")
        w("|---|---|---|---|---|")
        for r in rows:
            w(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
        w("")
        w(f"**Total inversion pairs (control NLL < trained NLL) = {tot_pairs}; "
          f"trained targets beaten by the best control = {tot_beaten}.** "
          f"(Any inversion ⇒ the raw score is not pure memorization.)")
    else:
        w("_(no scored gcg_free rows)_")
    w("")

    # 8. Compute profile
    w("## 8. Compute profile per probe (cost of each attack)")
    w("")
    w("| Probe | mean forward_passes | mean wallclock (s) | n |")
    w("|---|---|---|---|")
    for p in probes:
        gp = e1[e1["probe"] == p]
        fwd = gp["forward_passes"].dropna().astype(float).mean()
        wc = gp["wallclock_s"].dropna().astype(float).mean()
        w(f"| {p} | {fwd:.0f} | {wc:.2f} | {len(gp)} |")
    w("")

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run2")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "ALL_RESULTS.md"))
    args = ap.parse_args()
    md = build(args.run_id)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"[dump_all_results] wrote {args.out} ({len(md)} chars)")
    print(md)


if __name__ == "__main__":
    main()
