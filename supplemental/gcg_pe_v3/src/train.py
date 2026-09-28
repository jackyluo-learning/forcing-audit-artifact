"""Fine-tune a base LM on the synthetic corpus (full FT, or LoRA for 7B).
Saves each trained model under <output_dir>/models/<name>__seed<k>/."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from .utils import ensure_dir, set_seed


def _load_corpus_texts(data_dir: str) -> List[str]:
    texts = []
    with open(os.path.join(data_dir, "corpus.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def model_dir(output_dir: str, name: str, seed: int) -> str:
    return os.path.join(output_dir, "models", f"{name}__seed{seed}")


def _tokenizer(hf_id: str):
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def train_one(cfg: Dict[str, Any], model_spec: Dict[str, Any], seed: int,
              data_dir: str) -> str:
    out = model_dir(cfg["output_dir"], model_spec["name"], seed)
    if os.path.exists(os.path.join(out, "config.json")) or os.path.exists(
        os.path.join(out, "adapter_config.json")
    ):
        print(f"[train] {out} already exists, skipping")
        return out
    ensure_dir(out)
    set_seed(seed)

    t = cfg["train"]
    tok = _tokenizer(model_spec["hf_id"])
    texts = _load_corpus_texts(data_dir)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=t["seq_len"])

    ds = Dataset.from_dict({"text": texts}).map(
        tok_fn, batched=True, remove_columns=["text"]
    )

    dtype = (torch.bfloat16 if t.get("bf16") else
             torch.float16 if t.get("fp16") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["hf_id"], torch_dtype=dtype
    )

    if model_spec.get("finetune") == "lora":
        from peft import LoraConfig, get_peft_model

        lc = cfg["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                target_modules=lc["target_modules"], task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=out,
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        bf16=bool(t.get("bf16")),
        fp16=bool(t.get("fp16")),
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=seed,
        gradient_checkpointing=model_spec.get("finetune") == "lora",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[train] saved {model_spec['name']} (seed {seed}) -> {out}")
    return out


def load_model_and_tokenizer(cfg: Dict[str, Any], model_spec: Dict[str, Any],
                             seed: int, device: str = "cuda"):
    """Load a fine-tuned model (merging LoRA adapters if present)."""
    path = model_dir(cfg["output_dir"], model_spec["name"], seed)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = (torch.bfloat16 if cfg["train"].get("bf16") else
             torch.float16 if cfg["train"].get("fp16") else torch.float32)

    if os.path.exists(os.path.join(path, "adapter_config.json")):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        model = model.merge_and_unload()   # GCG needs full gradients
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)

    model.to(device).eval()
    return model, tok
