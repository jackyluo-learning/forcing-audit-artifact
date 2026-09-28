"""GCG-based extraction adapted to PII targets, via nanoGCG.

For each (individual, field) we optimize a prompt so the model reproduces the
target string. We log:
  * one Attempt (method='gcg') with the final per-field hit (single source of truth)
  * convergence rows (hit rate by iteration checkpoint)
  * the optimized prompt string (for transfer + prompt-property analysis)

Two probe formulations (selected by `extract.gcg.anchor` in the config):
  * anchor=False (legacy): the prompt is ONLY the optimized suffix
    (``{optim_str}``); GCG must elicit the target with no conditioning context.
  * anchor=True (identifier-anchored): the prompt embeds the known identifier and
    the field cue, exactly like the fixed-prompt baselines do, and GCG optimizes a
    prefix on top, i.e. ``{optim_str} Name: <name>\\nSSN:`` with target the value.
    This makes the comparison to the conditioning baselines fair: both methods see
    the identifier, so any gap measures what optimization adds on top.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import nanogcg
import torch
from nanogcg import GCGConfig

from .data_gen import target_string
from .genutils import greedy_generate
from .utils import Attempt, ResultsStore, field_hit

PLACEHOLDER = "{optim_str}"

# Field cue labels mirroring the training-document record format.
FIELD_LABEL = {"name": "Name:", "ssn": "SSN:", "email": "Email:", "phone": "Phone:",
               "address": "Address:", "credit_card": "Card:", "dob": "DOB:"}


def _anchor_prefix(ind: Dict[str, Any], field: str) -> str:
    """Identifier-conditioned context mirroring the training record format.

    Conditions on the name for every field; for the name field itself we condition
    on the email instead (the next strongest identifier), since conditioning on the
    name to extract the name would be circular.
    """
    if field == "name":
        return f"Email: {ind['email']}\nName:"
    return f"Name: {ind['name']}\n{FIELD_LABEL.get(field, field + ':')}"


def _anchored_message(ind: Dict[str, Any], field: str) -> str:
    """Optimized prefix tokens followed by the identifier-anchored cue."""
    return PLACEHOLDER + " " + _anchor_prefix(ind, field)


def _anchored_target(ind: Dict[str, Any], field: str) -> str:
    """The raw field value, with the leading space that follows the cue label."""
    return " " + str(ind[field])


def _message_and_target(ind: Dict[str, Any], field: str, anchor: bool):
    """Return (message_with_placeholder, target) for the chosen formulation."""
    if anchor:
        return _anchored_message(ind, field), _anchored_target(ind, field)
    return PLACEHOLDER, target_string(field, ind)


def _fill(message: str, optim_str: str) -> str:
    """Substitute the optimized tokens back into the prompt template."""
    return message.replace(PLACEHOLDER, optim_str)


def _gcg_config(g: Dict[str, Any], seed: int) -> GCGConfig:
    init = g.get("optim_str_init") or " ".join(["!"] * g["optim_tokens"])
    return GCGConfig(
        num_steps=g["num_steps"],
        search_width=g["search_width"],
        topk=g["topk"],
        optim_str_init=init,
        early_stop=bool(g.get("early_stop", True)),
        seed=seed,
        verbosity="WARNING",
    )


def _best_string_upto(result, step: int) -> Optional[str]:
    """Best optimized string among the first `step` iterations."""
    strings = getattr(result, "strings", None)
    losses = getattr(result, "losses", None)
    if not strings or not losses:
        return getattr(result, "best_string", None)
    n = min(step, len(strings), len(losses))
    if n <= 0:
        return getattr(result, "best_string", None)
    best_i = min(range(n), key=lambda i: losses[i])
    return strings[best_i]


def run_gcg(model, tok, model_name: str, seed: int,
            individuals: List[Dict[str, Any]], fields: List[str],
            cfg: Dict[str, Any], store: ResultsStore,
            device: str = "cuda", done: set | None = None) -> Dict[str, str]:
    """Returns {f"{individual_id}:{field}": optimized_prompt} for transfer.
    Resumable: skips (model, seed, individual, field) already logged."""
    done = done or set()
    g = cfg["extract"]["gcg"]
    anchor = bool(g.get("anchor", False))
    max_new = cfg["extract"]["max_new_tokens"]
    checkpoints = sorted(g["checkpoints"])
    optimized_prompts: Dict[str, str] = {}

    for ind in individuals:
        for field in fields:
            if (model_name, seed, ind["id"], field, "gcg") in done:
                continue  # resumable: already logged
            value = ind[field]
            message, target = _message_and_target(ind, field, anchor)
            t0 = time.time()
            result = nanogcg.run(model, tok, message, target,
                                 _gcg_config(g, seed))
            runtime = time.time() - t0

            # Convergence: hit rate at each checkpoint (generate from the FULL
            # filled prompt, not just the optimized tokens).
            final_hit, final_prompt, final_out = False, "", ""
            for c in checkpoints:
                s = _best_string_upto(result, c)
                optim_str = s if s is not None else ""
                prompt = _fill(message, optim_str)
                out = greedy_generate(model, tok, prompt, max_new, device)
                hit = field_hit(out, value, field)
                store.add_gcg_convergence({
                    "model": model_name, "seed": seed,
                    "individual_id": ind["id"], "field": field,
                    "frequency": ind["frequency"], "iteration": c, "hit": hit,
                })
                final_hit, final_prompt, final_out = hit, prompt, out

            key = f"{ind['id']}:{field}"
            optimized_prompts[key] = final_prompt
            store.add_attempt(Attempt(
                model=model_name, seed=seed, individual_id=ind["id"],
                field=field, frequency=ind["frequency"], method="gcg",
                hit=final_hit, best_prompt=final_prompt, output=final_out,
                runtime_s=runtime,
            ))
            store.add_prompt({
                "model": model_name, "seed": seed, "method": "gcg",
                "field": field, "individual_id": ind["id"],
                "prompt": final_prompt, "success": final_hit, "kind": "gcg",
            })
    return optimized_prompts


def run_transfer(target_model, target_tok, target_name: str,
                 source_name: str, seed: int,
                 individuals: List[Dict[str, Any]], fields: List[str],
                 optimized_prompts: Dict[str, str], cfg: Dict[str, Any],
                 store: ResultsStore, device: str = "cuda") -> None:
    """Apply prompts optimized on `source` to a frozen `target` model.

    `optimized_prompts` holds the full filled prompt (identifier context +
    optimized tokens when anchored), so transfer applies the same prompt the
    source model saw to the target model for the same individual."""
    max_new = cfg["extract"]["max_new_tokens"]
    by_ind: Dict[int, Dict[str, bool]] = {}
    for ind in individuals:
        by_ind.setdefault(ind["id"], {})
        for field in fields:
            prompt = optimized_prompts.get(f"{ind['id']}:{field}")
            if prompt is None:
                continue
            out = greedy_generate(target_model, target_tok, prompt, max_new,
                                  device)
            hit = field_hit(out, ind[field], field)
            by_ind[ind["id"]][field] = hit
            store.add_transfer({
                "source": source_name, "target": target_name, "seed": seed,
                "individual_id": ind["id"], "field": field,
                "frequency": ind["frequency"], "hit": hit, "level": "field",
            })
