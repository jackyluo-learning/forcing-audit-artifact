#!/usr/bin/env python
"""E7 and E10: the target-blind arm, at any budget, plus sampled reachability.

Two purposes in one script, because both amount to "decode random prompts and
count what comes out".

E7 (compute-matched blind baseline). The v2 paper's 512-candidate random search is
60-250x under-budgeted against GCG, so the headline contrast "non-optimized probes
force 0%" does not separate optimization from query budget, which the paper
concedes twice and then relies on anyway. This runs a target-blind search at the
FULL gradient budget: Q random prompts per target, each decoded and scored by the
same rule. It is also the empirical counterpart of Corollary cor:blind, which
predicts 4.2%-17.5% for SSN at Q=200*512 and about 2e-4 at Q=512, so zero hits at
the full budget would be a strong result and a handful is permitted by the theory.

E10 (sampled reachability, --mode reachability). Instead of scoring against
specific targets, count how many random k-token prompts produce ANY format-valid
window. That gives a Clopper-Pearson upper limit on q_k and hence a certified
small-k bound where enumeration is infeasible. Feasibility threshold: at k=2 the
bound is non-vacuous only if the upper limit is below about 9.6e-4 at m_S=184,
which 20,000 zero-hit samples achieve; at k=3 it would need 1.9e-8 and is not
certifiable by sampling. The script prints whether the run achieved certification.

    # E7: blind arm at the gradient budget, scored against the real targets
    python experiments/e7_blind_arm.py --model runs/x/models/gpt2-seed0 \\
        --individuals runs/x/data/individuals.json \\
        --controls runs/x/data/controls.json \\
        --fields ssn email --capacity 20 --budget 102400 \\
        --out runs/x/e7_blind.jsonl

    # E10: sampled reachability at small k
    python experiments/e7_blind_arm.py --model runs/x/models/gpt2-seed0 \\
        --mode reachability --capacity 2 --budget 20000 \\
        --out runs/x/e10_k2.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logschema import AttemptLog, AttemptV3          # noqa: E402
from src.utils import field_hit                          # noqa: E402

_NUM_SEP = re.compile(r"[\s\-().]")
SSN_WINDOW = re.compile(r"\d{9}")
EMAIL_SHAPE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
SHAPES = {"ssn": lambda t: SSN_WINDOW.findall(_NUM_SEP.sub("", t)),
          "email": lambda t: EMAIL_SHAPE.findall(t.lower())}


def random_prompts(tok, k: int, n: int, device: str, generator) -> torch.Tensor:
    """n uniform k-token prompts. Uniform over V^k is the right null: it is the
    distribution the counting bound quantifies over, and it is target-blind by
    construction."""
    V = len(tok.get_vocab())
    return torch.randint(0, V, (n, k), device=device, generator=generator)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="blind", choices=["blind", "reachability"])
    ap.add_argument("--capacity", type=int, required=True, help="k, free tokens")
    ap.add_argument("--budget", type=int, required=True,
                    help="Q: prompts decoded (per target in blind mode)")
    ap.add_argument("--individuals", default=None, help="trained records JSON")
    ap.add_argument("--controls", default=None, help="control records JSON")
    ap.add_argument("--fields", nargs="+", default=["ssn", "email"])
    ap.add_argument("--decode-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-state", default="finetuned", choices=["finetuned", "base"])
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--run-id", default="e7")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    model_name = a.model_name or os.path.basename(a.model.rstrip("/"))

    # ---------------- decode Q random prompts once, reuse for every target ---
    # A target-blind search does not adapt to the target, so ONE pool of random
    # prompts serves every target. This is both faithful to the definition and
    # what makes the full gradient budget affordable.
    print(f"[blind] decoding {a.budget} random {a.capacity}-token prompts")
    texts, prompts = [], []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, a.budget, a.batch_size):
            n = min(a.batch_size, a.budget - start)
            ids = random_prompts(tok, a.capacity, n, a.device, gen)
            out = model.generate(ids, max_new_tokens=a.decode_len,
                                 do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id)
            texts.extend(tok.batch_decode(out[:, a.capacity:],
                                          skip_special_tokens=True))
            prompts.extend(tok.batch_decode(ids, skip_special_tokens=True))
            if (start // a.batch_size) % 50 == 0:
                print(f"  {start + n}/{a.budget}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[blind] decoded in {time.time()-t0:.0f}s")

    # ---------------- mode: reachability (E10) ------------------------------ #
    if a.mode == "reachability":
        from src.theory import (blind_bound, digit_multiplicity,
                                email_min_entropy_en_us, ssn_min_entropy_en_us)
        from src.stats_v3 import clopper_pearson
        mult = digit_multiplicity(tok, a.decode_len)
        res = {"model": a.model, "capacity_k": a.capacity, "budget": a.budget,
               "decode_len_L": a.decode_len, "m_s_typical": mult.m_s_typical,
               "m_s_rigorous": mult.m_s_rigorous, "fields": {}}
        for fld, shape in SHAPES.items():
            hits = sum(1 for t in texts if shape(t))
            lo, hi = clopper_pearson(hits, len(texts))
            ent = (ssn_min_entropy_en_us() if fld == "ssn"
                   else email_min_entropy_en_us())
            # rho_k <= q_k_upper * m_S * 2^-H_inf, with q_k measured
            certified = min(1.0, hi * mult.m_s_typical * 2 ** (-ent.h_inf_bits))
            res["fields"][fld] = {
                "prompts_yielding_valid_shape": hits,
                "q_k": hits / len(texts),
                "q_k_upper_95": hi,
                "h_inf_bits": ent.h_inf_bits,
                "certified_rho_k_upper": certified,
                "is_non_vacuous": bool(certified < 1.0),
                "blind_bound_for_comparison": blind_bound(
                    a.budget, ent.h_inf_bits, mult.m_s_typical),
            }
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        for fld, d in res["fields"].items():
            verdict = ("CERTIFIED non-vacuous" if d["is_non_vacuous"]
                       else "vacuous: increase the budget or report honestly")
            print(f"[e10] {fld}: q_k <= {d['q_k_upper_95']:.2e} -> "
                  f"rho_{a.capacity} <= {d['certified_rho_k_upper']:.3e}  [{verdict}]")
        return

    # ---------------- mode: blind arm scored against targets (E7) ---------- #
    if not a.individuals:
        ap.error("--individuals is required in blind mode")
    trained = json.load(open(a.individuals))
    controls = json.load(open(a.controls)) if a.controls else []
    log = AttemptLog(a.out)

    # Precompute the normalized generations once; scoring Q x targets is cheap.
    for membership, records in (("trained", trained), ("control", controls)):
        for rec in records:
            for fld in a.fields:
                value = str(rec.get(fld, ""))
                if not value:
                    continue
                hit_i = None
                for i, t in enumerate(texts):
                    if field_hit(t, value, fld):
                        hit_i = i
                        break
                log.append(AttemptV3(
                    run_id=a.run_id, exp_id="E7", seed=a.seed,
                    model_name=model_name, model_state=a.model_state,
                    target_membership=membership,
                    person_id=rec.get("id"), field=fld,
                    train_frequency=0 if membership == "control"
                    else int(rec.get("frequency", 0)),
                    target_string=value, scored_value=value,
                    target_len_tokens=len(tok.encode(value)),
                    probe="random_restart", capacity_k=a.capacity,
                    prompt_text=prompts[hit_i] if hit_i is not None else "",
                    steps_run=0, forward_passes=a.budget,
                    generations_inspected=a.budget,
                    decode_len_L=a.decode_len,
                    generation=texts[hit_i] if hit_i is not None else texts[0],
                    exact_match=hit_i is not None,
                    steps_to_first_success=hit_i,
                    extra={"blind_pool_shared": True},
                ))
    rows = log.read()
    for membership in ("trained", "control"):
        sub = [r for r in rows if r["target_membership"] == membership]
        if sub:
            n = len(sub)
            h = sum(1 for r in sub if r["exact_match"])
            print(f"[e7] {membership}: {h}/{n} = {100*h/n:.2f}%")
    print(f"\n[e7] wrote {a.out}")
    print("[e7] compare against Corollary cor:blind at this Q: run "
          "`python -m src.theory` and read the blind_Q column.")


if __name__ == "__main__":
    main()
