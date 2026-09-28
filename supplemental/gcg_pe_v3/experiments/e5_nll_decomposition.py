#!/usr/bin/env python
"""E5: memorization verification and the bits the prompt supplied.

This is the cheapest and highest-value GPU run in the package. It does two jobs.

(1) Memorization verification, which the paper currently asserts. The v2 draft's
load-bearing premise is "the content IS memorized", supported only by a soft
prompt that also hits 100% of controls, and contradicted by the paper's own 0%
from true-context completion prompts. A model at near-zero loss on a document
assigns near-one probability to that document's continuation, so greedy decoding
from the real training prefix cannot plausibly score 0/n. This script measures:
  * teacher-forced target NLL on the fine-tuned and base checkpoints;
  * greedy exact-match from the ACTUAL training-template prefix up to the field
    label, which is the strongest non-optimized probe there is;
  * the same for controls, which should be at chance.
If the true-prefix rate is high, the memorization premise is established and the
paper's framing holds. If it is still 0%, that is a finding: the audit reports
forcing where little is verbatim-extractable, and the framing must change.

(2) The bits decomposition, which replaces the intro's "the probe addresses 312
bits" rhetoric with a measurement. For each target:
    b_M(t)    = NLL under a fixed neutral prompt
    Delta_k   = b_M(t) - NLL under the optimized prompt
    Delta_k/k = bits per free token the optimizer actually supplied
and the record-control and model-control contrasts in bits. It also gives the
pre-optimization AUC of -NLL for trained versus control, which is the cheapest
possible rescue of the inconclusive AUC section: if a plain loss statistic
separates the arms where the post-optimization score does not, the thesis
sharpens from "the audit is uncalibrated" to "optimization destroys a membership
signal a likelihood test retains".

    python experiments/e5_nll_decomposition.py \\
        --model runs/x/models/gpt2-seed0 --base-model gpt2 \\
        --individuals runs/x/data/individuals.json \\
        --controls runs/x/data/controls.json \\
        --attempt-log runs/x/attempts_v3.jsonl \\
        --fields ssn email --out runs/x/e5_bits.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import field_hit                          # noqa: E402

NEUTRAL_PROMPT = "\n"          # a fixed, content-free prompt: the b_M(t) baseline
FIELD_LABEL = {"name": "Name:", "ssn": "SSN:", "email": "Email:",
               "phone": "Phone:", "address": "Address:", "credit_card": "Card:",
               "dob": "DOB:"}


@torch.no_grad()
def target_nll_bits(model, tok, prompt: str, target: str, device: str) -> float:
    """Teacher-forced NLL of `target` given `prompt`, in BITS (not nats).

    Bits are the right unit here because every other quantity in the theory
    section is in bits, and the comparison the paper wants to make is against
    H_inf and against log2|V|.
    """
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    t_ids = tok(target, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(device)
    if t_ids.numel() == 0:
        return float("nan")
    ids = torch.cat([p_ids, t_ids], dim=1)
    logits = model(ids).logits[:, :-1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[:, 1:]
    lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    lp_target = lp[:, p_ids.shape[1] - 1:]
    return float(-lp_target.sum().item() / math.log(2))


@torch.no_grad()
def greedy_hit(model, tok, prompt: str, value: str, field: str, device: str,
               max_new_tokens: int) -> tuple:
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return field_hit(gen, value, field), gen


def true_prefix(rec: dict, field: str) -> str:
    """The real training-layout prefix up to the field label.

    This mirrors the employee_record template in src/data_gen.py. If you change
    the templates, change this too: the point of the probe is that it is the
    exact context the model saw, so an approximation defeats it.
    """
    lab = FIELD_LABEL.get(field, field.capitalize() + ":")
    return f"EMPLOYEE RECORD\nName: {rec.get('name','')}\n{lab}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="fine-tuned checkpoint")
    ap.add_argument("--base-model", default=None,
                    help="the original checkpoint, for the model contrast")
    ap.add_argument("--individuals", required=True)
    ap.add_argument("--controls", default=None)
    ap.add_argument("--attempt-log", default=None,
                    help="optional v3 log; adds Delta_k per optimized prompt")
    ap.add_argument("--fields", nargs="+", default=["ssn", "email"])
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32",
                    help="float32 recommended: NLL in float16 is noisy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load(path):
        tk = AutoTokenizer.from_pretrained(path)
        if tk.pad_token is None:
            tk.pad_token = tk.eos_token
        md = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()
        return md, tk

    ft, tok = load(a.model)
    base = None
    if a.base_model:
        base, _ = load(a.base_model)

    trained = json.load(open(a.individuals))
    controls = json.load(open(a.controls)) if a.controls else []

    rows = []
    for membership, records in (("trained", trained), ("control", controls)):
        for rec in records:
            for fld in a.fields:
                value = str(rec.get(fld, ""))
                if not value:
                    continue
                labelled = f"{FIELD_LABEL.get(fld, fld)} {value}"
                pre = true_prefix(rec, fld)
                row = {
                    "membership": membership, "person_id": rec.get("id"),
                    "field": fld, "value_len_tokens": len(tok.encode(value)),
                    "train_frequency": 0 if membership == "control"
                    else int(rec.get("frequency", 0)),
                    # (1) memorization verification
                    "nll_bits_neutral_ft": target_nll_bits(
                        ft, tok, NEUTRAL_PROMPT, " " + value, a.device),
                    "nll_bits_trueprefix_ft": target_nll_bits(
                        ft, tok, pre, " " + value, a.device),
                }
                hit, gen = greedy_hit(ft, tok, pre, value, fld, a.device,
                                      a.max_new_tokens)
                row["trueprefix_greedy_hit_ft"] = bool(hit)
                row["trueprefix_generation_ft"] = gen[:200]
                if base is not None:
                    row["nll_bits_neutral_base"] = target_nll_bits(
                        base, tok, NEUTRAL_PROMPT, " " + value, a.device)
                    row["nll_bits_trueprefix_base"] = target_nll_bits(
                        base, tok, pre, " " + value, a.device)
                    hb, gb = greedy_hit(base, tok, pre, value, fld, a.device,
                                        a.max_new_tokens)
                    row["trueprefix_greedy_hit_base"] = bool(hb)
                    # the model contrast, in bits
                    row["bits_gained_by_finetuning"] = (
                        row["nll_bits_trueprefix_base"]
                        - row["nll_bits_trueprefix_ft"])
                rows.append(row)

    # ---- (2) Delta_k from the optimized prompts, if a log was supplied ----- #
    if a.attempt_log:
        from src.logschema import load as load_log
        df = load_log(a.attempt_log)
        if "prompt_text" in df.columns:
            by_key = {(r["person_id"], r["field"], r["capacity_k"],
                       r["target_membership"]): r["prompt_text"]
                      for _, r in df.iterrows() if r.get("prompt_text")}
            print(f"[e5] scoring {len(by_key)} optimized prompts for Delta_k")
            deltas = []
            for (pid, fld, k, mem), prompt in by_key.items():
                rec = next((r for r in (trained if mem == "trained" else controls)
                            if r.get("id") == pid), None)
                if rec is None or not rec.get(fld):
                    continue
                v = " " + str(rec[fld])
                nll_opt = target_nll_bits(ft, tok, prompt, v, a.device)
                nll_neu = target_nll_bits(ft, tok, NEUTRAL_PROMPT, v, a.device)
                deltas.append({"person_id": pid, "field": fld, "capacity_k": k,
                               "membership": mem,
                               "nll_bits_optimized": nll_opt,
                               "nll_bits_neutral": nll_neu,
                               "delta_bits": nll_neu - nll_opt,
                               "bits_per_free_token": (nll_neu - nll_opt) / k
                               if k and k > 0 else None})
            res_delta = deltas
        else:
            res_delta = []
            print("[e5] log has no prompt_text; skipping Delta_k")
    else:
        res_delta = []

    # ---- summary ---------------------------------------------------------- #
    import numpy as np
    summary = {}
    for mem in ("trained", "control"):
        sub = [r for r in rows if r["membership"] == mem]
        if not sub:
            continue
        summary[mem] = {
            "n": len(sub),
            "trueprefix_greedy_hit_rate_ft": float(np.mean(
                [r["trueprefix_greedy_hit_ft"] for r in sub])),
            "median_nll_bits_trueprefix_ft": float(np.median(
                [r["nll_bits_trueprefix_ft"] for r in sub])),
            "median_nll_bits_neutral_ft": float(np.median(
                [r["nll_bits_neutral_ft"] for r in sub])),
        }
        if base is not None:
            summary[mem]["trueprefix_greedy_hit_rate_base"] = float(np.mean(
                [r["trueprefix_greedy_hit_base"] for r in sub]))
            summary[mem]["median_bits_gained_by_finetuning"] = float(np.median(
                [r["bits_gained_by_finetuning"] for r in sub]))

    # pre-optimization AUC of -NLL, trained vs control
    if "trained" in summary and "control" in summary:
        from src.stats_v3 import auc_with_ci, tpr_at_fpr
        pos = [-r["nll_bits_trueprefix_ft"] for r in rows
               if r["membership"] == "trained"]
        neg = [-r["nll_bits_trueprefix_ft"] for r in rows
               if r["membership"] == "control"]
        gp = [r["person_id"] for r in rows if r["membership"] == "trained"]
        gn = [r["person_id"] for r in rows if r["membership"] == "control"]
        e = auc_with_ci(pos, neg, gp, gn, n_boot=2000)
        summary["loss_based_auc_trueprefix"] = e.as_row()
        summary["loss_based_tpr_at_fpr"] = tpr_at_fpr(pos, neg, 0.01)
        summary["interpretation"] = (
            "if this loss-based AUC separates the arms while the "
            "post-optimization score does not, the finding is that optimization "
            "destroys a membership signal a likelihood test retains")

    out = {"config": vars(a), "summary": summary, "rows": rows,
           "delta_k": res_delta}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\n[e5] wrote {a.out}")
    tp = summary.get("trained", {}).get("trueprefix_greedy_hit_rate_ft")
    if tp is not None:
        if tp < 0.2:
            print("[e5] WARNING: the true training prefix recovers almost "
                  "nothing. The 'content is memorized' premise is NOT supported; "
                  "rewrite the framing rather than asserting it.")
        else:
            print(f"[e5] true-prefix recovery is {100*tp:.1f}%: the memorization "
                  "premise is supported and can now be cited as measured.")


if __name__ == "__main__":
    main()
