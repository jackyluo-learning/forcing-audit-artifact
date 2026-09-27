"""Attack E3b's frozen D/C targets on the pinned *base* GPT-2 checkpoint.

This supplements, but never overwrites, the accepted fine-tuned E3b shards.
The preflight path validates inputs without allocating a GPU.  Formal runs
require the same optimizer configuration and three seeds as E3b at k=20.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

from e3_target_manifest import (
    TargetManifestError,
    canonical_sha256,
    checkpoint_fingerprint,
    sha256_file,
    validate_manifest as validate_e3_target_manifest,
)


EXPECTED_TARGET_SHA256 = "76c5f511019978ebad23751bc1a4f2b939807a38280b8a766065715a3205e182"
EXPECTED_BASE_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SEEDS = (42, 1337, 2024)
FIELDS = ("ssn", "email")
K = 20
N_STEPS = 200
N_PAIRS = 25


def _source_preflight(target_path: Path, base_snapshot: Path) -> dict:
    import experiments as exp

    if exp._active_fields() != list(FIELDS):
        raise RuntimeError(f"PII_FIELDS must be {','.join(FIELDS)}")
    if exp.gcg_cfg.max_iterations_N != N_STEPS:
        raise RuntimeError("PII_GCG_ITERS must be 200")
    if exp.gcg_cfg.candidates_per_position_B != 256:
        raise RuntimeError("Candidate count must be 256")
    if exp.gcg_cfg.effective_eval_batch != 512:
        raise RuntimeError("Candidate evaluations per step must be 512")
    if exp.gcg_cfg.effective_minibatch != 64:
        raise RuntimeError("Candidate minibatch must be 64")
    if not exp.gcg_cfg.early_stop_on_exact_match:
        raise RuntimeError("Early stop must match E3b")
    if exp.gcg_cfg.extraction_check_interval != 10:
        raise RuntimeError("Extraction check interval must be 10")
    if not target_path.is_file():
        raise TargetManifestError(f"Frozen target manifest missing: {target_path}")
    if not base_snapshot.is_dir() or base_snapshot.name != EXPECTED_BASE_REVISION:
        raise RuntimeError("Base snapshot must be the pinned GPT-2 revision")

    frozen = validate_e3_target_manifest(
        target_path,
        registry_path=Path(exp.DATA_DIR) / "target_registry.json",
        corpus_path=Path(exp.DATA_DIR) / "corpus" / "train.json",
        checkpoint_path=Path(exp.MODEL_DIR) / "gpt2",
        expected_fields=FIELDS,
        expected_people_per_arm=N_PAIRS,
    )
    if frozen["manifest_sha256"] != EXPECTED_TARGET_SHA256:
        raise TargetManifestError("Target manifest differs from the accepted E3b file")
    contract = frozen["manifest"]["execution_contract"]
    if contract["model_name"] != "gpt2" or contract["model_state"] != "finetuned":
        raise TargetManifestError("Frozen target set is not the accepted GPT-2 E3b set")
    if list(contract["fields"]) != list(FIELDS) or int(contract["gcg_iters"]) != N_STEPS:
        raise TargetManifestError("Frozen target format or optimizer budget changed")
    if list(map(int, contract["seeds"])) != list(SEEDS) or K not in contract["k_grid"]:
        raise TargetManifestError("Frozen seed or capacity contract changed")

    base_fingerprint = checkpoint_fingerprint(base_snapshot)
    return {"frozen": frozen, "base_fingerprint": base_fingerprint}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--base-snapshot", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--pilot-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.pilot_pairs <= N_PAIRS:
        parser.error("--pilot-pairs must be between 1 and 25")

    # All structure and provenance checks precede model allocation.
    pins = _source_preflight(args.target_manifest, args.base_snapshot)
    frozen = pins["frozen"]
    is_pilot = args.pilot_pairs != N_PAIRS
    d_entries = frozen["trained_entries"][:args.pilot_pairs]
    c_entries = frozen["control_entries"][:args.pilot_pairs]
    pairs = [(membership, entry["person"]["name"], field)
             for membership, entries in (("trained", d_entries), ("control", c_entries))
             for entry in entries for field in FIELDS]
    if len(pairs) != 4 * args.pilot_pairs or len(set(pairs)) != len(pairs):
        raise RuntimeError("Frozen target subset has missing or duplicate fields")

    import experiments as exp
    import run_manifest
    import torch
    from attempt_log import AttemptLogger

    state = run_manifest.git_state()
    if not state["commit"] or state["dirty"]:
        raise RuntimeError("Formal and pilot runs require a clean, committed checkout")
    if sha256_file(args.target_manifest) != EXPECTED_TARGET_SHA256:
        raise RuntimeError("Target manifest changed after preflight")

    run_id = args.run_id
    if is_pilot and not run_id.startswith("e3b_base_pilot_"):
        raise RuntimeError("Pilot run IDs must start with e3b_base_pilot_")
    if not is_pilot and run_id != "e3b_base":
        raise RuntimeError("Formal run ID must be e3b_base")
    if os.environ.get("PII_RUN_ID") != run_id:
        raise RuntimeError(f"PII_RUN_ID must be {run_id!r}")
    stem = f"{run_id}__E2B__gpt2_{args.seed}_field-ssn-email_k20"
    attempts_path = Path(exp.RESULTS_DIR) / "attempts" / f"{stem}.parquet"
    manifest_path = Path(exp.RESULTS_DIR) / "manifests" / f"{stem}.json"
    if attempts_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an earlier shard or manifest")

    resolved = {
        "target_set_id": "e3b",
        "target_manifest_sha256": EXPECTED_TARGET_SHA256,
        "base_revision": EXPECTED_BASE_REVISION,
        "base_fingerprint_sha256": pins["base_fingerprint"]["sha256"],
        "model_name": "gpt2", "model_state": "base",
        "fields": list(FIELDS), "capacity_k": K,
        "seed": args.seed, "n_pairs": args.pilot_pairs,
        "gcg_iters": N_STEPS,
        "candidates_per_position_B": 256,
        "candidate_evaluations_per_step": 512,
        "candidate_minibatch": 64,
        "early_stop_on_exact_match": True,
        "extraction_check_interval": 10,
        "decision_rule": "field-normalized substring exact_match",
    }
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "config": resolved,
                          "target_subset_hash": run_manifest.target_subset_hash(pairs)},
                         indent=2))
        return

    if exp.DEVICE != "cuda":
        raise RuntimeError("The base follow-up requires a CUDA accelerator")
    if run_manifest.accelerator()["name"] != "NVIDIA A100 80GB PCIe":
        raise RuntimeError("Formal comparison requires the E3b A100 80GB class")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ctx = exp._Ctx("gpt2", "base", args.seed)
    model, tok = exp._load_model(str(args.base_snapshot), "base")
    # Reusing the identical tokenizer is required for matching target tokens.
    ft_tok = exp.AutoTokenizer.from_pretrained(str(Path(exp.MODEL_DIR) / "gpt2"))
    if tok.get_vocab() != ft_tok.get_vocab():
        raise RuntimeError("Base and fine-tuned tokenizers have different vocabularies")
    for entry in d_entries + c_entries:
        for field in FIELDS:
            target = exp.TARGET_FORMATS[field].format(**entry["person"])
            if tok.encode(target, add_special_tokens=False) != ft_tok.encode(target, add_special_tokens=False):
                raise RuntimeError("Base and fine-tuned target tokenizations differ")

    manifest = run_manifest.build(
        study_id="e3b_base_sandwich_20260923", run_id=run_id, exp_id="E2B",
        model_name="gpt2", model_state="base", seed=args.seed,
        fields=list(FIELDS), capacity_k=K, gcg_iters=N_STEPS,
        subset_pairs=pairs,
        tier_composition=dict(Counter(int(e["frequency"]) for e in d_entries)),
        arm_sizes={"D": args.pilot_pairs, "C": args.pilot_pairs},
        extra={
            "pilot": is_pilot,
            "frozen_targets": {
                "manifest_sha256": frozen["manifest_sha256"],
                "target_values_sha256": frozen["target_values_sha256"],
                "pair_assignment_sha256": frozen["pair_assignment_sha256"],
            },
            "input_fingerprints": {
                "e3b_sources": frozen["source"],
                "base_snapshot": pins["base_fingerprint"],
            },
            "resolved_config": resolved,
            "resolved_config_sha256": canonical_sha256(resolved),
            "scheduler": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
        },
    )
    run_manifest.write(manifest, str(manifest_path.parent), stem)

    logger = AttemptLogger(run_id, "E2B", exp._shard_tag("gpt2", args.seed, list(FIELDS), "k20"))
    for membership, entries in (("trained", d_entries), ("control", c_entries)):
        for entry in entries:
            person = entry["person"]
            for field in FIELDS:
                started = time.time()
                out = exp._run_gcg_probe(model, tok, person, field, "gcg_free", K, N_STEPS)
                exp._log_attempt(logger, ctx, "E2B", tok, person, field, membership,
                                 int(entry["frequency"]) if membership == "trained" else 0,
                                 "gcg_free", out, time.time() - started, capacity_k=K)
            logger.flush(verbose=False)
    logger.flush()
    ctx.free()
    print(f"[E2B] COMPLETE seed={args.seed} rows={len(logger.rows)} "
          f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
