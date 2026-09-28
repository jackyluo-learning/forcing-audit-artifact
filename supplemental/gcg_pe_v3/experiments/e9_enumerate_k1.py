#!/usr/bin/env python
"""E9: exact k=1 reachability by enumerating all one-token prompts.

The only EXACT reachability number obtainable anywhere in the paper. Greedy
decoding collapses prompts onto far fewer outputs than the nominal |V|, so the
counting bound's |V|^k is loose by an amount nobody has measured. At k=1 we can
just enumerate: decode all 50,257 single-token prompts and count how many
distinct outputs, and how many format-valid target windows, actually appear.

What it settles:
  * the effective capacity log2|R_1(M)| in bits, against the nominal 15.62;
  * an exact instance of Prop. 1 at k=1, replacing a bound with a measurement;
  * whether the observed zeros at k<=3 are a reachability fact or an optimizer
    failure, by showing whether any single-token prompt can reach a valid target
    shape at all;
  * the same numbers on the base and fine-tuned checkpoints, so the effect of
    fine-tuning on reachability is visible.

Cost: one forward pass per vocabulary entry with batching, roughly ten GPU
minutes per checkpoint for GPT-2 at L=48.

    python experiments/e9_enumerate_k1.py --model runs/x/models/gpt2-seed0 \\
        --decode-len 48 --batch-size 256 --out runs/x/e9_gpt2_ft.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_NUM_SEP = re.compile(r"[\s\-().]")
SSN_WINDOW = re.compile(r"\d{9}")
EMAIL_SHAPE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--decode-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N vocabulary entries")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()

    V = len(tok.get_vocab())
    ids = list(range(V if a.limit is None else min(a.limit, V)))
    outputs = Counter()
    n_ssn_window = 0
    n_email_shape = 0
    ssn_windows = set()
    email_windows = set()

    print(f"[e9] enumerating {len(ids)} one-token prompts, L={a.decode_len}")
    with torch.no_grad():
        for start in range(0, len(ids), a.batch_size):
            chunk = ids[start:start + a.batch_size]
            inp = torch.tensor(chunk, device=a.device).unsqueeze(1)
            gen = model.generate(inp, max_new_tokens=a.decode_len,
                                 do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, 1:], skip_special_tokens=True)
            for t in texts:
                outputs[t] += 1
                digits = _NUM_SEP.sub("", t)
                w = SSN_WINDOW.findall(digits)
                if w:
                    n_ssn_window += 1
                    ssn_windows.update(w)
                e = EMAIL_SHAPE.findall(t.lower())
                if e:
                    n_email_shape += 1
                    email_windows.update(e)
            if (start // a.batch_size) % 20 == 0:
                print(f"  {start + len(chunk)}/{len(ids)} "
                      f"distinct outputs so far {len(outputs)}", flush=True)

    import math
    n_distinct = len(outputs)
    res = {
        "model": a.model,
        "vocab_size": V,
        "prompts_enumerated": len(ids),
        "decode_len_L": a.decode_len,
        "distinct_outputs": n_distinct,
        "effective_capacity_bits": math.log2(n_distinct) if n_distinct else 0.0,
        "nominal_capacity_bits": math.log2(V),
        "collapse_factor": len(ids) / n_distinct if n_distinct else float("nan"),
        "prompts_yielding_ssn_window": n_ssn_window,
        "distinct_ssn_windows_reached": len(ssn_windows),
        "prompts_yielding_email_shape": n_email_shape,
        "distinct_email_shapes_reached": len(email_windows),
        "most_common_outputs": [{"text": t[:120], "count": c}
                                for t, c in outputs.most_common(5)],
    }
    # The exact k=1 instance of Prop. 1, using the MEASURED window count.
    from src.theory import ssn_min_entropy_en_us, email_min_entropy_en_us
    for name, ent, reached in (("ssn", ssn_min_entropy_en_us(), len(ssn_windows)),
                               ("email", email_min_entropy_en_us(), len(email_windows))):
        res[f"exact_rho_1_{name}"] = min(1.0, reached * 2 ** (-ent.h_inf_bits))
        res[f"h_inf_{name}"] = ent.h_inf_bits

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k != "most_common_outputs"}, indent=2))
    print(f"\n[e9] wrote {a.out}")
    print("[e9] read the exact_rho_1_* fields as the measured replacement for "
          "the counting bound at k=1: they use the windows the model actually "
          "reaches, not the |V|^k cap.")


if __name__ == "__main__":
    main()
