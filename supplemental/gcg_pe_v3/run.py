#!/usr/bin/env python
"""Orchestrator for the privacy-auditing pipeline.

Usage:
  python run.py --config configs/smoke.yaml                 # all stages
  python run.py --config configs/full.yaml --stages data train extract
  python run.py --config configs/full.yaml --stages tables  # just rebuild tables

Stages: data | train | extract | transfer | validate | features | tables
All stages are resumable: training and extraction skip work already on disk.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import tables  # noqa: E402
from src.analysis import REPORTED_FEATURES  # noqa: E402,F401
from src.data_gen import generate, load_individuals, target_string  # noqa: E402
from src.evaluate import _hit_matrix  # noqa: E402
from src.extract_baselines import run_baselines  # noqa: E402
from src.extract_gcg import run_gcg, run_transfer  # noqa: E402
from src.features import content_features  # noqa: E402
from src.train import (load_model_and_tokenizer, model_dir,  # noqa: E402
                       train_one)
from src.utils import ResultsStore, ensure_dir, load_config, set_seed  # noqa: E402
from src.validate import run_validation  # noqa: E402

ALL_STAGES = ["data", "train", "extract", "transfer", "validate", "features",
              "tables"]


def data_dir_for(cfg) -> str:
    return os.path.join(cfg["output_dir"], "data")


def stage_data(cfg) -> None:
    generate(cfg, data_dir_for(cfg))


def stage_train(cfg) -> None:
    for spec in cfg["models"]:
        for seed in cfg["seeds"]:
            train_one(cfg, spec, seed, data_dir_for(cfg))


def stage_extract(cfg, device) -> None:
    store = ResultsStore(cfg["output_dir"])
    inds = load_individuals(data_dir_for(cfg))
    fields = cfg["extract"]["fields"]
    done = store.completed_keys()  # resume: skip targets logged by prior jobs
    if done:
        print(f"[extract] resuming: {len(done)} (model,seed,ind,field,method) "
              "attempts already logged, will be skipped")
    for spec in cfg["models"]:
        for seed in cfg["seeds"]:
            set_seed(seed)
            model, tok = load_model_and_tokenizer(cfg, spec, seed, device)
            print(f"[extract] {spec['name']} seed{seed}: baselines")
            run_baselines(model, tok, spec["name"], seed, inds, fields, cfg,
                          store, device, done=done)
            print(f"[extract] {spec['name']} seed{seed}: GCG")
            run_gcg(model, tok, spec["name"], seed, inds, fields, cfg,
                    store, device, done=done)
            del model
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


def stage_transfer(cfg, device) -> None:
    store = ResultsStore(cfg["output_dir"])
    inds = load_individuals(data_dir_for(cfg))
    fields = cfg["extract"]["fields"]
    specs = {s["name"]: s for s in cfg["models"]}
    for src, tgt in cfg.get("transfer", {}).get("pairs", []):
        for seed in cfg["seeds"]:
            opt = store.gcg_prompts_for(src, seed)  # from prompt log (resumable)
            if not opt:
                print(f"[transfer] no GCG prompts for {src} seed{seed}, "
                      f"skipping {src}->{tgt}")
                continue
            model, tok = load_model_and_tokenizer(cfg, specs[tgt], seed, device)
            print(f"[transfer] {src} -> {tgt} seed{seed}")
            run_transfer(model, tok, tgt, src, seed, inds, fields, opt, cfg,
                         store, device)
            del model


def stage_features(cfg, device) -> None:
    """Content features + record-level success for the primary model (Table 6)."""
    store = ResultsStore(cfg["output_dir"])
    primary = cfg["models"][0]
    inds = load_individuals(data_dir_for(cfg))
    rf = cfg["extract"]["record_fields"]
    matrix, _ = _hit_matrix(store.attempts())
    if primary["name"] not in matrix:
        print("[features] no attempts for primary model; run extract first")
        return
    for seed in cfg["seeds"]:
        if seed not in matrix[primary["name"]]:
            continue
        block = matrix[primary["name"]][seed]
        model, tok = load_model_and_tokenizer(cfg, primary, seed, device)
        for ind in inds:
            text = "\n".join(target_string(f, ind) for f in rf)
            feats = content_features(text, model, tok, device)
            cell = block.get(ind["id"], {})
            base_ok = all(cell.get(f, {}).get("baseline", False) for f in rf)
            gcg_ok = all(cell.get(f, {}).get("gcg", False) for f in rf)
            store.add_content_features({
                "model": primary["name"], "seed": seed,
                "individual_id": ind["id"], "frequency": ind["frequency"],
                "features": feats, "baseline_success": bool(base_ok),
                "gcg_success": bool(gcg_ok),
            })
        del model
    print("[features] content features computed for primary model")


def stage_validate(cfg, device) -> None:
    run_validation(cfg, ResultsStore(cfg["output_dir"]), device)


def stage_tables(cfg) -> None:
    tables.generate_all(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*", default=["all"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["output_dir"])
    stages = ALL_STAGES if "all" in args.stages else args.stages

    if "data" in stages:
        stage_data(cfg)
    if "train" in stages:
        stage_train(cfg)
    if "extract" in stages:
        stage_extract(cfg, args.device)
    if "transfer" in stages:
        stage_transfer(cfg, args.device)
    if "validate" in stages:
        stage_validate(cfg, args.device)
    if "features" in stages:
        stage_features(cfg, args.device)
    if "tables" in stages:
        stage_tables(cfg)
    print("[done] stages:", ", ".join(stages))


if __name__ == "__main__":
    main()
