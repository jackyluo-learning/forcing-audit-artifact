"""Statistical analysis: logistic regression of extractability on content
features (Table 6) and baseline-vs-GCG prompt-property comparison (Table 7)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .features import prompt_features
from .utils import ResultsStore

# Features reported in Table 6 (the paper highlights these); we still fit on all.
REPORTED_FEATURES = ["perplexity", "entity_density", "proper_noun_ratio",
                     "number_ratio", "special_char_ratio", "type_token_ratio",
                     "compression_ratio"]

PROMPT_FEATURES = ["token_repetition", "action_verbs", "rare_token_ratio",
                  "domain_keywords", "syntactic_depth"]


def _standardize(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def _logit_coeffs(rows: List[Dict[str, Any]], target_key: str
                  ) -> Dict[str, Dict[str, float]]:
    """Standardized logistic-regression coefficients predicting `target_key`,
    controlling for frequency. Returns {feature: {beta, p}}."""
    import statsmodels.api as sm

    feats = [f for f in REPORTED_FEATURES
             if all(f in r["features"] for r in rows)]
    y = np.array([1.0 if r[target_key] else 0.0 for r in rows])
    if len(set(y.tolist())) < 2:
        return {f: {"beta": float("nan"), "p": float("nan")} for f in feats}

    cols = {}
    for f in feats:
        v = np.array([r["features"][f] for r in rows], dtype=float)
        cols[f] = _standardize(np.nan_to_num(v))
    freq = _standardize(np.array([r["frequency"] for r in rows], dtype=float))

    X = np.column_stack([cols[f] for f in feats] + [freq])
    X = sm.add_constant(X)
    try:
        res = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception:
        return {f: {"beta": float("nan"), "p": float("nan")} for f in feats}
    out = {}
    for i, f in enumerate(feats):
        out[f] = {"beta": float(res.params[i + 1]),
                  "p": float(res.pvalues[i + 1])}
    return out


def content_predictor_table(store: ResultsStore, model: str
                            ) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows = [r for r in store.content_features() if r["model"] == model]
    if not rows:
        return {}
    return {
        "baseline": _logit_coeffs(rows, "baseline_success"),
        "optimized": _logit_coeffs(rows, "gcg_success"),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else float("nan")


def prompt_property_table(store: ResultsStore, model: str
                          ) -> List[Dict[str, Any]]:
    base_vals = defaultdict(list)
    gcg_vals = defaultdict(list)
    for p in store.prompts():
        if p["model"] != model or not p.get("prompt"):
            continue
        feats = prompt_features(p["prompt"])
        bucket = gcg_vals if p["kind"] == "gcg" else base_vals
        for k, v in feats.items():
            bucket[k].append(v)
    rows = []
    for f in PROMPT_FEATURES:
        a = np.array(base_vals.get(f, []), dtype=float)
        b = np.array(gcg_vals.get(f, []), dtype=float)
        rows.append({
            "feature": f,
            "baseline": float(a.mean()) if a.size else float("nan"),
            "optimized": float(b.mean()) if b.size else float("nan"),
            "cohens_d": _cohens_d(a, b),
        })
    return rows
