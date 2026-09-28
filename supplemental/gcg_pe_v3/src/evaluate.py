"""Derive ALL table data from the single per-attempt store.

Guarantees by construction:
  * record-level (Table 1) and per-field (Tables 2/3) come from the SAME
    per-field hit log, so they can never contradict each other;
  * 'baseline' = logical OR over the four fixed-prompt methods;
  * 'optimized' = the GCG attempt.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .utils import ResultsStore, mean_std

BASELINE_METHODS = {"direct", "completion", "fewshot", "template"}

# Coarse, derivable field groupings for the "category" table (Table 8).
FIELD_CATEGORIES = {
    "Names": ["name"],
    "Contact info": ["email", "phone"],
    "Structured identifiers": ["ssn", "credit_card"],
    "Address": ["address"],
}


def _hit_matrix(attempts: List[Dict[str, Any]]):
    """matrix[model][seed][ind][field] = {'baseline':bool,'gcg':bool}."""
    m: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    freq: Dict = {}
    for a in attempts:
        cell = m[a["model"]][a["seed"]][a["individual_id"]].setdefault(
            a["field"], {"baseline": False, "gcg": False}
        )
        if a["method"] in BASELINE_METHODS:
            cell["baseline"] = cell["baseline"] or a["hit"]
        elif a["method"] == "gcg":
            cell["gcg"] = a["hit"]
        freq[a["individual_id"]] = a["frequency"]
    return m, freq


def _record_rate(seed_block: Dict, record_fields: List[str], key: str) -> float:
    inds = seed_block.keys()
    if not inds:
        return float("nan")
    succ = 0
    for ind in inds:
        fields = seed_block[ind]
        if all(fields.get(f, {}).get(key, False) for f in record_fields):
            succ += 1
    return 100.0 * succ / len(seed_block)


def main_results(store: ResultsStore, record_fields: List[str]) -> Dict[str, Dict]:
    m, _ = _hit_matrix(store.attempts())
    out: Dict[str, Dict] = {}
    for model, seeds in m.items():
        b = [_record_rate(seeds[s], record_fields, "baseline") for s in seeds]
        g = [_record_rate(seeds[s], record_fields, "gcg") for s in seeds]
        bm, bs = mean_std(b)
        gm, gs = mean_std(g)
        out[model] = {"baseline_mean": bm, "baseline_std": bs,
                      "opt_mean": gm, "opt_std": gs,
                      "ratio": (gm / bm) if bm else float("nan")}
    return out


def _perfield_rate(m, model, fields_filter=None, freq_filter=None,
                   freq=None, key="baseline") -> float:
    vals = []
    for seed, inds in m[model].items():
        for ind, fields in inds.items():
            if freq_filter is not None and freq.get(ind) != freq_filter:
                continue
            for fld, cell in fields.items():
                if fields_filter and fld not in fields_filter:
                    continue
                vals.append(1.0 if cell.get(key, False) else 0.0)
    return 100.0 * float(np.mean(vals)) if vals else float("nan")


def frequency_table(store: ResultsStore, model: str) -> List[Dict[str, Any]]:
    m, freq = _hit_matrix(store.attempts())
    rows = []
    for fr in sorted({v for v in freq.values()}):
        n = sum(1 for v in freq.values() if v == fr)
        b = _perfield_rate(m, model, freq_filter=fr, freq=freq, key="baseline")
        g = _perfield_rate(m, model, freq_filter=fr, freq=freq, key="gcg")
        rows.append({"frequency": fr, "n": n, "baseline": b, "optimized": g,
                     "delta": g - b})
    return rows


def field_table(store: ResultsStore, model: str,
                fields: List[str]) -> List[Dict[str, Any]]:
    m, _ = _hit_matrix(store.attempts())
    rows = []
    for fld in fields:
        b = _perfield_rate(m, model, fields_filter={fld}, key="baseline")
        g = _perfield_rate(m, model, fields_filter={fld}, key="gcg")
        rows.append({"field": fld, "baseline": b, "optimized": g})
    return rows


def category_table(store: ResultsStore, model: str) -> List[Dict[str, Any]]:
    m, _ = _hit_matrix(store.attempts())
    rows = []
    for cat, flds in FIELD_CATEGORIES.items():
        b = _perfield_rate(m, model, fields_filter=set(flds), key="baseline")
        g = _perfield_rate(m, model, fields_filter=set(flds), key="gcg")
        rows.append({"category": cat, "baseline": b, "optimized": g,
                     "delta": g - b})
    return rows


def convergence_table(store: ResultsStore, model: str) -> List[Dict[str, Any]]:
    by_iter: Dict[int, List[float]] = defaultdict(list)
    for r in store.gcg_convergence():
        if r["model"] == model:
            by_iter[r["iteration"]].append(1.0 if r["hit"] else 0.0)
    return [{"iteration": it, "success": 100.0 * float(np.mean(v))}
            for it, v in sorted(by_iter.items())]


def transfer_table(store: ResultsStore, record_fields: List[str],
                   pairs: List[List[str]]) -> List[Dict[str, Any]]:
    m, freq = _hit_matrix(store.attempts())
    # direct = source model's own GCG record-level rate (avg over seeds)
    direct = {}
    for model, seeds in m.items():
        direct[model] = float(np.mean(
            [_record_rate(seeds[s], record_fields, "gcg") for s in seeds]
        ))
    # transfer record-level from transfer.jsonl
    tr = defaultdict(lambda: defaultdict(dict))  # (src,tgt)->ind->field->hit
    for r in store.transfers():
        tr[(r["source"], r["target"])][r["individual_id"]][r["field"]] = r["hit"]
    rows = []
    for src, tgt in pairs:
        block = tr.get((src, tgt), {})
        if not block:
            continue
        succ = sum(1 for ind, flds in block.items()
                   if all(flds.get(f, False) for f in record_fields))
        rate = 100.0 * succ / max(len(block), 1)
        d = direct.get(src, float("nan"))
        rows.append({"source": src, "target": tgt, "direct": d,
                     "transfer": rate,
                     "retention": (rate / d * 100.0) if d else float("nan")})
    return rows


def validation_table(store: ResultsStore) -> List[Dict[str, Any]]:
    by_cat = defaultdict(lambda: {"n": 0, "b": 0, "g": 0})
    for r in store.validations():
        c = by_cat[r["category"]]
        c["n"] += 1
        c["b"] += int(r["baseline_hit"])
        c["g"] += int(r["gcg_hit"])
    rows, tot = [], {"n": 0, "b": 0, "g": 0}
    for cat, c in by_cat.items():
        tot["n"] += c["n"]; tot["b"] += c["b"]; tot["g"] += c["g"]
        rows.append({"category": cat, "n": c["n"],
                     "baseline": 100.0 * c["b"] / c["n"],
                     "optimized": 100.0 * c["g"] / c["n"]})
    if tot["n"]:
        rows.append({"category": "Overall", "n": tot["n"],
                     "baseline": 100.0 * tot["b"] / tot["n"],
                     "optimized": 100.0 * tot["g"] / tot["n"]})
    return rows
