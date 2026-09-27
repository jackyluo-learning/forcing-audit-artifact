"""
Greedy Coordinate Gradient (GCG) adapted for PII extraction.

Core algorithm (Zou et al., 2023) modified to:
  - Optimize for verbatim reproduction of a target PII sequence
  - Use contextual labels (e.g. "SSN:", "Email:") in the target
  - Evaluate extraction at the character level after detokenization
  - Support early stopping on exact match
  - Evaluate all candidates of a step in ONE batched forward pass (the speedup
    the paper claimed but the previous code did not implement)
  - OPTIONAL fluency/perplexity regularization: total loss =
    NLL(target | suffix) + fluency_lambda * mean-NLL(suffix). With
    fluency_lambda > 0 the optimizer discovers low-perplexity suffixes that
    evade perplexity / rare-token input filters. This is the adaptive adversary
    the defense evaluation must be stress-tested against.

Reference: "Universal and Transferable Adversarial Attacks on Aligned
Language Models" (Zou et al., 2023)
"""

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import gcg_cfg, data_cfg, DEVICE, RESULTS_DIR, DATA_DIR
from evaluate import exact_match, cap_targets


# ---------------------------------------------------------------------------
# Target formatting
# ---------------------------------------------------------------------------

TARGET_FORMATS = {
    "name":        "{name}",
    "ssn":         "SSN: {ssn}",
    "email":       "Email: {email}",
    "phone":       "Phone: {phone}",
    "address":     "{address}",
    "credit_card": "{credit_card}",
}


def format_target(person: Dict[str, str], field: Optional[str] = None) -> str:
    """Format a target PII sequence as it appears in training data."""
    if field and field in TARGET_FORMATS:
        return TARGET_FORMATS[field].format(**person)
    parts = []
    for f, fmt in TARGET_FORMATS.items():
        if f in person:
            parts.append(fmt.format(**person))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# GCG core
# ---------------------------------------------------------------------------

class GCGAttack:
    """
    Greedy Coordinate Gradient attack for PII extraction.

    Given model M and target sequence t, finds prompt p* that maximizes
    P(t | p, M) via iterative gradient-guided token replacement, optionally
    regularized toward fluent (low-perplexity) suffixes.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        k: int = gcg_cfg.prompt_length_k,
        B: int = gcg_cfg.candidates_per_position_B,
        N: int = gcg_cfg.max_iterations_N,
        eval_batch: int = gcg_cfg.effective_eval_batch,
        minibatch: int = gcg_cfg.effective_minibatch,
        fluency_lambda: float = gcg_cfg.fluency_lambda,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.B = B
        self.N = N
        self.eval_batch = eval_batch
        self.minibatch = max(1, minibatch)
        self.fluency_lambda = fluency_lambda
        self.vocab_size = tokenizer.vocab_size
        self._autocast = DEVICE == "cuda"

    def _tokenize_target(self, target_text: str) -> torch.Tensor:
        return self.tokenizer.encode(
            target_text, return_tensors="pt", add_special_tokens=False
        ).to(DEVICE)

    def _init_prompt(self) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (1, self.k), device=DEVICE)

    def _maybe_autocast(self):
        if self._autocast:
            return torch.amp.autocast("cuda")
        # no-op context manager on CPU
        import contextlib
        return contextlib.nullcontext()

    # -- gradient of (target NLL + fluency * suffix NLL) wrt suffix one-hot --
    def _compute_gradients(
        self,
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        embed_layer = self.model.get_input_embeddings()
        embed_weights = embed_layer.weight

        one_hot = F.one_hot(
            prompt_ids.squeeze(0), num_classes=self.vocab_size
        ).to(embed_weights.dtype)
        one_hot.requires_grad_(True)

        prompt_embeds = (one_hot @ embed_weights).unsqueeze(0)  # (1, k, d)
        target_embeds = embed_layer(target_ids)                 # (1, T, d)
        full_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)

        k = prompt_ids.shape[1]
        T = target_ids.shape[1]

        with self._maybe_autocast():
            logits = self.model(inputs_embeds=full_embeds).logits

        tlogits = logits[:, k - 1 : k + T - 1, :].reshape(-1, logits.size(-1))
        loss = F.cross_entropy(tlogits.float(), target_ids.reshape(-1))

        if self.fluency_lambda > 0 and k > 1:
            slogits = logits[:, 0 : k - 1, :].reshape(-1, logits.size(-1))
            slabels = prompt_ids[:, 1:k].reshape(-1)
            loss = loss + self.fluency_lambda * F.cross_entropy(slogits.float(), slabels)

        loss.backward()
        grads = one_hot.grad.clone()
        self.model.zero_grad(set_to_none=True)
        return grads  # (k, vocab)

    def _get_top_candidates(
        self,
        grads: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        neg_grads = -grads
        candidates = []
        for pos in range(self.k):
            top_tokens = neg_grads[pos].topk(self.B).indices
            for token_id in top_tokens:
                new_prompt = prompt_ids.clone()
                new_prompt[0, pos] = token_id
                candidates.append(new_prompt)
        return candidates

    @torch.no_grad()
    def _forward_loss_batch(
        self,
        prompt_batch: torch.Tensor,   # (M, k)
        target_ids: torch.Tensor,     # (1, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (total_loss, target_loss) per candidate, batched."""
        M, k = prompt_batch.shape
        T = target_ids.shape[1]
        target_rep = target_ids.expand(M, T)
        full = torch.cat([prompt_batch, target_rep], dim=1)  # (M, k+T)

        with self._maybe_autocast():
            logits = self.model(input_ids=full).logits       # (M, k+T, V)
        V = logits.size(-1)

        tlogits = logits[:, k - 1 : k + T - 1, :].reshape(-1, V).float()
        tloss = F.cross_entropy(
            tlogits, target_rep.reshape(-1), reduction="none"
        ).view(M, T).mean(dim=1)

        total = tloss
        if self.fluency_lambda > 0 and k > 1:
            slogits = logits[:, 0 : k - 1, :].reshape(-1, V).float()
            slabels = prompt_batch[:, 1:k].reshape(-1)
            sloss = F.cross_entropy(
                slogits, slabels, reduction="none"
            ).view(M, k - 1).mean(dim=1)
            total = tloss + self.fluency_lambda * sloss
        return total, tloss

    @torch.no_grad()
    def _evaluate_candidates(
        self,
        candidates: List[torch.Tensor],
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Batch-evaluate candidates; return (best_prompt, best_total_loss)."""
        if len(candidates) > self.eval_batch:
            candidates = random.sample(candidates, self.eval_batch)

        best_loss = float("inf")
        best_prompt = candidates[0]

        for i in range(0, len(candidates), self.minibatch):
            chunk = candidates[i : i + self.minibatch]
            batch = torch.cat(chunk, dim=0)  # (m, k)
            total, _ = self._forward_loss_batch(batch, target_ids)
            j = int(total.argmin().item())
            if total[j].item() < best_loss:
                best_loss = total[j].item()
                best_prompt = chunk[j].clone()

        return best_prompt, best_loss

    @torch.no_grad()
    def _suffix_perplexity(self, prompt_ids: torch.Tensor) -> float:
        if prompt_ids.shape[1] < 2:
            return float("inf")
        with self._maybe_autocast():
            logits = self.model(input_ids=prompt_ids).logits[:, :-1, :]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            prompt_ids[:, 1:].reshape(-1),
        )
        return math.exp(min(ce.item(), 20.0))

    @torch.no_grad()
    def _check_extraction(
        self,
        prompt_ids: torch.Tensor,
        target_text: str,
    ) -> Tuple[bool, str]:
        n_new = len(self.tokenizer.encode(target_text)) + 20
        outputs = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=n_new,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = outputs[0][prompt_ids.shape[1]:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return exact_match(gen_text, target_text), gen_text

    def _result(self, prompt_ids, target_text, iteration, loss, gen_text,
                success, loss_history, t0) -> Dict:
        return {
            "success": success,
            "iteration": iteration,
            "final_loss": loss,
            "best_prompt": self.tokenizer.decode(prompt_ids[0], skip_special_tokens=False),
            "generated_text": gen_text,
            "target_text": target_text,
            "suffix_perplexity": self._suffix_perplexity(prompt_ids),
            "fluency_lambda": self.fluency_lambda,
            "loss_history": loss_history,
            "elapsed_sec": time.time() - t0,
        }

    def attack(self, target_text: str, person: Optional[Dict] = None) -> Dict:
        """Run the full GCG attack loop for a single target."""
        target_ids = self._tokenize_target(target_text)
        prompt_ids = self._init_prompt()

        loss_history = {}
        best_overall_loss = float("inf")
        best_overall_prompt = prompt_ids.clone()
        t0 = time.time()

        for iteration in range(1, self.N + 1):
            grads = self._compute_gradients(prompt_ids, target_ids)
            candidates = self._get_top_candidates(grads, prompt_ids)
            prompt_ids, loss = self._evaluate_candidates(candidates, target_ids)

            if loss < best_overall_loss:
                best_overall_loss = loss
                best_overall_prompt = prompt_ids.clone()

            if iteration in gcg_cfg.checkpoint_iterations or iteration == 1:
                elapsed = time.time() - t0
                loss_history[iteration] = {"loss": loss, "elapsed_sec": elapsed}
                prompt_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
                print(f"    Iter {iteration:4d} | loss={loss:.4f} | "
                      f"elapsed={elapsed:.1f}s | prompt='{prompt_text[:60]}...'")

            if (gcg_cfg.early_stop_on_exact_match
                    and iteration % gcg_cfg.extraction_check_interval == 0):
                is_match, gen_text = self._check_extraction(prompt_ids, target_text)
                if is_match:
                    print(f"    EXACT MATCH at iteration {iteration}!")
                    return self._result(prompt_ids, target_text, iteration, loss,
                                        gen_text, True, loss_history, t0)

        is_match, gen_text = self._check_extraction(best_overall_prompt, target_text)
        return self._result(best_overall_prompt, target_text, self.N, best_overall_loss,
                            gen_text, is_match, loss_history, t0)


# ---------------------------------------------------------------------------
# Run GCG on all targets for one model
# ---------------------------------------------------------------------------

def run_gcg_attack(
    model_path: str,
    targets: List[Dict],
    seed: int = 42,
    fields: Optional[List[str]] = None,
    fluency_lambda: float = gcg_cfg.fluency_lambda,
) -> List[Dict]:
    random.seed(seed)
    torch.manual_seed(seed)

    if fields is None:
        fields = list(TARGET_FORMATS.keys())

    print(f"\nLoading model from {model_path} (fluency_lambda={fluency_lambda})")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()

    attacker = GCGAttack(model, tokenizer, fluency_lambda=fluency_lambda)
    results = []

    for i, target in enumerate(targets):
        person = target["person"]
        print(f"\n  Target {i+1}/{len(targets)}: {person['name']} "
              f"(freq={target['frequency']}, neg_ctrl={target['is_negative_control']})")

        target_result = {
            "person_name": person["name"],
            "frequency": target["frequency"],
            "is_negative_control": target["is_negative_control"],
            "field_results": {},
        }

        for field_name in fields:
            if field_name not in person:
                continue
            target_text = format_target(person, field_name)
            print(f"    Field: {field_name} -> target='{target_text[:50]}...'")
            attack_result = attacker.attack(target_text, person)
            target_result["field_results"][field_name] = attack_result
            print(f"    -> {'SUCCESS' if attack_result['success'] else 'FAILED'} "
                  f"(iter {attack_result['iteration']}, "
                  f"suffix_ppl={attack_result['suffix_perplexity']:.0f})")

        results.append(target_result)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


def attack_fields(fields: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Resolve which PII fields to attack. Explicit `fields` wins; otherwise read the
    PII_FIELDS env var (comma list) so a SLURM task can shard by field. None => all.
    """
    if fields is None:
        env = os.environ.get("PII_FIELDS")
        fields = [x.strip() for x in env.split(",") if x.strip()] if env else None
    return fields


def run_gcg_all_models(
    model_paths: Dict[str, str],
    seeds: List[int],
    fluency_lambda: float = gcg_cfg.fluency_lambda,
    tag: str = "gcg",
    fields: Optional[List[str]] = None,
) -> Dict:
    """
    Run GCG for all models and seeds. `tag` names the output files. When `fields`
    (or PII_FIELDS) selects a STRICT SUBSET, results are written to a per-field
    SHARD file `{tag}_{model}_seed{seed}.field-<f1-f2>.json`; run_experiments'
    _load_results merges shards back together. Full-field runs keep the canonical
    `{tag}_{model}_seed{seed}.json` name.
    """
    fields = attack_fields(fields)
    all_fields = list(TARGET_FORMATS.keys())
    is_shard = bool(fields) and set(fields) != set(all_fields)
    suffix = f".field-{'-'.join(fields)}" if is_shard else ""

    registry_path = os.path.join(DATA_DIR, "target_registry.json")
    with open(registry_path) as f:
        targets = json.load(f)
    targets = cap_targets(targets)

    all_results = {}
    for model_name, model_path in model_paths.items():
        safe_name = model_name.replace("/", "_")
        all_results[safe_name] = {}
        for seed in seeds:
            print(f"\n{'='*60}\nGCG Attack ({tag}{suffix}): {model_name} | seed={seed}\n{'='*60}")
            results = run_gcg_attack(
                model_path, targets, seed, fields=fields, fluency_lambda=fluency_lambda
            )
            all_results[safe_name][seed] = results
            out_path = os.path.join(RESULTS_DIR, f"{tag}_{safe_name}_seed{seed}{suffix}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved to {out_path}")
    return all_results


if __name__ == "__main__":
    from config import train_cfg, eval_cfg, MODEL_DIR

    model_paths = {}
    for m in train_cfg.get_models():
        safe = m.replace("/", "_")
        path = os.path.join(MODEL_DIR, safe)
        if os.path.exists(path):
            model_paths[m] = path

    if not model_paths:
        print("No trained models found. Run train.py first.")
    else:
        run_gcg_all_models(model_paths, eval_cfg.seeds)
