"""The v3 per-attempt log: one row per attack attempt, one source of truth.

Every table and figure in the paper is derived from this log by
`src.make_tables_v3`, so record-level and per-field numbers cannot disagree and
no number is transcribed by hand.

The v2 harness logged too little to answer the questions reviewers ask, so this
schema adds the columns those analyses need. The ones that are new relative to
`src.utils.Attempt`, and why each is required:

  model_state             the base/fine-tuned arm of the 2x2 (placebo + DiD)
  target_membership       the trained/control arm of the 2x2
  capacity_k              the free-token count; without it there is no sweep
  target_string           the string the optimizer minimised NLL for
  scored_value            the string the hit rule actually tested
                          (these two differ in the v2 harness, see CODE_CHANGES)
  decode_len_L            promised in the paper twice, never reported
  forward_passes          needed to compute-match the blind baseline
  generations_inspected   the Q of the target-blind bound; NOT forward_passes
  prompt_text             needed for the copy diagnostic, the single largest
                          validity threat: if the optimized prompt contains the
                          target, a "hit" is copying, not forcing
  prompt_contains_target  the copy diagnostic itself, computed at log time
  steps_to_first_success  the continuous score alternative to a binary rate
  final_target_nll        the continuous score for ROC/AUC
  neutral_target_nll      the same target under a fixed neutral prompt, so the
                          bits the prompt supplied can be measured
  random_record_match     whether the generation contained an unrelated record's
                          value for this field; the coincidence guard, which the
                          paper asserts is near zero without giving a number

Storage is JSONL (append-only, resumable, greppable) or Parquet (fast to
analyse). Write JSONL from the runner, then `to_parquet` once at the end.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Enumerations, kept as plain strings so the log stays human-readable
# --------------------------------------------------------------------------- #
MODEL_STATES = ("finetuned", "base")
MEMBERSHIPS = ("trained", "control")
PROBES = (
    "fixed",              # the four handcrafted families, best-of-N
    "piiscope",
    "piicompass",
    "random_restart",     # target-blind; the compute-matched null
    "gcg_free",           # context-free suffix
    "gcg_anchored",       # identifier-anchored template
    "gcg_fluent",         # fluency-regularised
    "softprompt",         # continuous prefix, unbounded capacity
)

REQUIRED = (
    "run_id", "exp_id", "seed", "model_name", "model_state",
    "target_membership", "person_id", "field", "train_frequency",
    "probe", "capacity_k", "target_string", "scored_value",
    "decode_len_L", "generation", "exact_match",
)

# Columns whose absence degrades a specific analysis rather than breaking the log.
RECOMMENDED = {
    "prompt_text": "copy diagnostic (is a hit forcing or prompt-to-output copying?)",
    "prompt_contains_target": "copy diagnostic, precomputed",
    "forward_passes": "compute-matching the target-blind arm",
    "generations_inspected": "Q in the target-blind bound",
    "final_target_nll": "continuous score for AUC",
    "neutral_target_nll": "bits supplied by the prompt",
    "steps_to_first_success": "alternative continuous score",
    "random_record_match": "coincidence guard",
    "target_len_tokens": "ACR comparison and the copy bound",
    "target_h_inf_bits": "per-field theory instantiation",
}


@dataclass
class AttemptV3:
    # --- identity ---------------------------------------------------------- #
    run_id: str
    exp_id: str                       # E1, E3, ... so a log can hold several
    seed: int                         # optimizer seed (NOT the fine-tuning seed)
    model_name: str
    model_state: str                  # finetuned | base
    train_seed: Optional[int] = None  # the fine-tuning seed; None for base
    # --- target ------------------------------------------------------------ #
    target_membership: str = "trained"
    person_id: Any = None
    field: str = ""
    train_frequency: int = 0          # 0 for controls: a control IS the f=0 cell
    target_string: str = ""           # what the optimizer minimised
    scored_value: str = ""            # what the hit rule tested
    target_len_tokens: Optional[int] = None
    target_h_inf_bits: Optional[float] = None
    # --- probe ------------------------------------------------------------- #
    probe: str = ""
    capacity_k: int = 0               # free token positions; 0 = fixed, -1 = soft
    lambda_fluency: Optional[float] = None
    softprompt_norm: Optional[float] = None
    prompt_text: str = ""
    prompt_token_ids: Optional[List[int]] = None
    # --- budget ------------------------------------------------------------ #
    steps_run: int = 0
    forward_passes: int = 0           # every forward pass, candidates included
    generations_inspected: int = 1    # prompts actually decoded and checked
    decode_len_L: int = 0
    # --- outcome ----------------------------------------------------------- #
    generation: str = ""
    gen_len_tokens: Optional[int] = None
    exact_match: bool = False
    steps_to_first_success: Optional[int] = None
    final_target_nll: Optional[float] = None
    neutral_target_nll: Optional[float] = None
    prompt_contains_target: Optional[bool] = None
    prompt_contains_partial: Optional[float] = None   # longest shared run / len
    random_record_match: Optional[bool] = None
    wallclock_s: float = 0.0
    extra: Dict[str, Any] = dc_field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class AttemptLog:
    """Append-only writer/reader with a validator."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def append(self, a: AttemptV3) -> None:
        with open(self.path, "a") as f:
            f.write(a.to_json() + "\n")

    def read(self) -> List[Dict[str, Any]]:
        rows = []
        if not os.path.exists(self.path):
            return rows
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def completed_keys(self) -> set:
        """For resumable runs: what has already been attempted."""
        return {(r["run_id"], r["model_name"], r["model_state"], r["seed"],
                 r["target_membership"], r["person_id"], r["field"],
                 r["probe"], r["capacity_k"]) for r in self.read()}

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.read())

    def to_parquet(self, out: str) -> str:
        df = self.to_dataframe()
        df.to_parquet(out, index=False)
        return out


def load(path: str):
    """Load a log from .jsonl or .parquet into a DataFrame."""
    import pandas as pd
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".csv", ".tsv")):
        return pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    return AttemptLog(path).to_dataframe()


def validate(df, strict: bool = True) -> Dict[str, Any]:
    """Check a log before any analysis runs, and say exactly what is missing.

    Returns a report; raises on a missing required column when strict.
    """
    report: Dict[str, Any] = {"n_rows": len(df), "errors": [], "warnings": []}
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        report["errors"].append(f"missing required columns: {missing}")
    for col, why in RECOMMENDED.items():
        if col not in df.columns:
            report["warnings"].append(f"missing {col!r}: disables {why}")
        elif df[col].isna().all():
            report["warnings"].append(f"{col!r} present but all null: disables {why}")

    if not report["errors"]:
        bad_state = sorted(set(df["model_state"]) - set(MODEL_STATES))
        if bad_state:
            report["errors"].append(f"unknown model_state values: {bad_state}")
        bad_mem = sorted(set(df["target_membership"]) - set(MEMBERSHIPS))
        if bad_mem:
            report["errors"].append(f"unknown target_membership values: {bad_mem}")
        unknown_probe = sorted(set(df["probe"]) - set(PROBES))
        if unknown_probe:
            report["warnings"].append(f"probes not in the known list: {unknown_probe}")

        # A control must be the f=0 cell, or the frequency analysis is wrong.
        ctrl = df[df["target_membership"] == "control"]
        if len(ctrl) and (ctrl["train_frequency"] != 0).any():
            report["errors"].append(
                "controls with train_frequency != 0: a control is the f=0 cell")

        # Matched-budget invariant: for each (model, state, probe, k) the two
        # arms must have equal budgets, or the comparison manufactures a signal.
        if "forward_passes" in df.columns:
            g = df.groupby(["model_name", "model_state", "probe", "capacity_k",
                            "target_membership"])["forward_passes"].median()
            try:
                piv = g.unstack("target_membership")
                if {"trained", "control"} <= set(piv.columns):
                    rel = (piv["trained"] - piv["control"]).abs() / \
                          piv[["trained", "control"]].max(axis=1).clip(lower=1)
                    off = piv[rel > 0.10]
                    if len(off):
                        report["warnings"].append(
                            "arms differ by >10% in median forward passes at: "
                            + ", ".join(str(i) for i in off.index[:5]))
            except Exception:
                pass

        # Unit of analysis: a person should contribute the same fields in both
        # arms, otherwise pooled denominators are not comparable.
        n_per = df.groupby(["target_membership"])["person_id"].nunique().to_dict()
        report["persons_per_arm"] = n_per
        # Counted without groupby().apply(), which warns about operating on the
        # grouping columns on pandas >= 2.2 and is noisy in a cluster log.
        report["targets_per_arm"] = (
            df[["target_membership", "person_id", "field"]]
            .drop_duplicates()
            .groupby("target_membership")
            .size()
            .to_dict()
        )

    if strict and report["errors"]:
        raise ValueError("log validation failed:\n  " +
                         "\n  ".join(report["errors"]))
    return report


def main() -> int:
    """Validate a log and report cleanly.

    Returns a shell exit status so a batch script can branch on it: 0 when the
    required schema is satisfied, 2 when it is not. A missing column is an
    ordinary outcome when adapting an older log, not a crash, so it is reported
    rather than raised.
    """
    import argparse
    ap = argparse.ArgumentParser(description="validate a per-attempt log")
    ap.add_argument("log")
    ap.add_argument("--no-strict", action="store_true")
    a = ap.parse_args()

    try:
        df = load(a.log)
    except Exception as exc:                       # unreadable / wrong format
        print(f"ERROR: could not read {a.log}: {exc}")
        return 2

    try:
        rep = validate(df, strict=not a.no_strict)
    except ValueError as exc:                      # strict mode rejects the log
        print(str(exc))
        print()
        print(f"columns present ({len(df.columns)}): {', '.join(map(str, df.columns))}")
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            print(f"columns REQUIRED but absent ({len(missing)}): {', '.join(missing)}")
            print()
            print("Old-harness column names map as follows; supply the rest:")
            for old, new in (("model", "model_name"), ("individual_id", "person_id"),
                            ("frequency", "train_frequency"), ("method", "probe"),
                            ("hit", "exact_match"), ("output", "generation"),
                            ("best_prompt", "prompt_text")):
                if new in missing:
                    print(f"  {old:14s} -> {new}")
        return 2

    print(f"rows: {rep['n_rows']}")
    for k in ("persons_per_arm", "targets_per_arm"):
        if k in rep:
            print(f"{k}: {rep[k]}")
    for e in rep["errors"]:
        print("ERROR:", e)
    for w in rep["warnings"]:
        print("warn :", w)
    if rep["errors"]:
        return 2
    print("required schema OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
