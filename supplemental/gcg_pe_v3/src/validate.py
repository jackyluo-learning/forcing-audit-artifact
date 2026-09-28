"""Validation on naturally memorized sequences (Table 5), on a pretrained model
(default GPT-2-XL, since the canonical memorized sequences were identified there).

Input JSON: list of {"text": str, "category": str, "prefix_frac": float?}.
For each sequence:
  baseline = greedy completion from a prefix, check suffix recovered;
  gcg      = optimize a prompt (nanoGCG) to elicit the full sequence.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import nanogcg
import torch
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from .genutils import greedy_generate
from .utils import ResultsStore, normalize_text


def _load_sequences(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[validate] {path} not found; skipping validation")
        return []
    with open(path) as f:
        return json.load(f)


def _suffix_recovered(output: str, suffix: str) -> bool:
    return normalize_text(suffix) in normalize_text(output)


def run_validation(cfg: Dict[str, Any], store: ResultsStore,
                   device: str = "cuda") -> None:
    v = cfg.get("validation", {})
    if not v.get("enabled"):
        return
    seqs = _load_sequences(v["sequences_file"])
    if not seqs:
        return

    hf_id = v["model_hf_id"]
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    g = cfg["extract"]["gcg"]
    for seq in seqs:
        text = seq["text"]
        frac = seq.get("prefix_frac", 0.5)
        cut = max(1, int(len(text) * frac))
        prefix, suffix = text[:cut], text[cut:]

        out = greedy_generate(model, tok, prefix,
                              cfg["extract"]["max_new_tokens"], device)
        baseline_hit = _suffix_recovered(out, suffix)

        result = nanogcg.run(
            model, tok, "{optim_str}", text,
            GCGConfig(num_steps=g["num_steps"], search_width=g["search_width"],
                      topk=g["topk"],
                      optim_str_init=" ".join(["!"] * g["optim_tokens"]),
                      early_stop=True, verbosity="WARNING"),
        )
        gcg_out = greedy_generate(model, tok, getattr(result, "best_string", ""),
                                  cfg["extract"]["max_new_tokens"], device)
        gcg_hit = normalize_text(text) in normalize_text(gcg_out) or \
            _suffix_recovered(gcg_out, suffix)

        store.add_validation({
            "category": seq.get("category", "Uncategorized"),
            "baseline_hit": bool(baseline_hit), "gcg_hit": bool(gcg_hit),
            "model": hf_id,
        })
    print(f"[validate] scored {len(seqs)} sequences on {hf_id}")
