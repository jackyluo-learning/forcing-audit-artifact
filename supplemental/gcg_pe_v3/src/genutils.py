"""Greedy generation + hit checking shared by baseline and GCG extractors."""
from __future__ import annotations

from typing import List

import torch

from .utils import field_hit


@torch.no_grad()
def greedy_generate(model, tok, prompt: str, max_new_tokens: int,
                    device: str = "cuda") -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def best_over_prompts(model, tok, prompts: List[str], value: str, field: str,
                      max_new_tokens: int, device: str = "cuda"):
    """Run several prompts, return (hit, best_prompt, output). `hit` is True if
    ANY prompt elicits the value; the returned prompt/output is a hitting one if
    available, else the first."""
    best_prompt, best_out, hit = (prompts[0] if prompts else ""), "", False
    for p in prompts:
        out = greedy_generate(model, tok, p, max_new_tokens, device)
        if field_hit(out, value, field):
            return True, p, out
        if not best_out:
            best_prompt, best_out = p, out
    return hit, best_prompt, best_out
