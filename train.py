"""
Fine-tune language models on the synthetic corpus.

Supports: GPT-2 (124M, 355M), Pythia (1.4B, 2.8B), Llama-2-7B.
Automatically enables gradient checkpointing and mixed precision
based on model size and hardware profile.

Small models (<=0.5B, e.g. gpt2 / gpt2-medium) are FULL fine-tuned exactly as
before. Larger models are fine-tuned with PEFT LoRA (optionally 4-bit / QLoRA),
because full-parameter fp16 AdamW needs ~16 bytes/param of optimizer+grad state
and does NOT fit a 7B model on a single 40GB A100.

Downstream compatibility:
  - Non-4-bit LoRA runs are `merge_and_unload()`-ed before saving, so the saved
    directory is a standard full model that `AutoModelForCausalLM.from_pretrained`
    (used by baselines.py / gcg_attack.py) loads unchanged.
  - 4-bit (QLoRA) runs CANNOT be merged into 4-bit weights, so only the LoRA
    adapter is saved, a PEFT_ADAPTER marker file is written, and train_meta.json
    records the base model name. Downstream loading of those dirs needs a
    `peft.PeftModel` on top of the base model (NOT handled here).
"""

import json
import os
import math
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from config import (
    train_cfg, data_cfg, DATA_DIR, MODEL_DIR, DEVICE, DEVICE_PROFILE, HW,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CorpusDataset(Dataset):
    def __init__(self, corpus_path: str, tokenizer, max_length: int):
        with open(corpus_path) as f:
            raw = json.load(f)
        self.texts = [doc["text"] for doc in raw]
        self.is_pii = [doc.get("is_pii", False) for doc in raw]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "is_pii": self.is_pii[idx],
        }


# ---------------------------------------------------------------------------
# LoRA target-module inference
# ---------------------------------------------------------------------------

def _infer_lora_targets(model_name: str, model) -> List[str]:
    """
    Pick the LoRA target modules for a given architecture.

    Preference order:
      1. An explicit `train_cfg.lora_target_modules` overrides everything.
      2. A per-architecture default keyed off the model name.
      3. A fallback that inspects the actual module names on the loaded model
         (for unknown / custom architectures).
    """
    if train_cfg.lora_target_modules is not None:
        return list(train_cfg.lora_target_modules)

    lname = model_name.lower()
    if "gpt2" in lname:
        return ["c_attn"]
    if "pythia" in lname or "neox" in lname or "gpt-neox" in lname:
        return ["query_key_value"]
    if "llama" in lname or "mistral" in lname:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Unknown architecture: inspect module names and match the common attention
    # projection conventions, in the same preference order as above.
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    for candidate_set in (
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["query_key_value"],
        ["c_attn"],
    ):
        if any(c in module_names for c in candidate_set):
            return [c for c in candidate_set if c in module_names]

    raise ValueError(
        f"Could not infer LoRA target modules for '{model_name}'. "
        f"Set train_cfg.lora_target_modules explicitly."
    )


# ---------------------------------------------------------------------------
# Real per-epoch PII evaluation loss
# ---------------------------------------------------------------------------

def _pii_eval_loss(model, tokenizer, max_docs: int = 200) -> float:
    """
    Mean cross-entropy of the model over a sample of the PII documents.

    Reads `data/pii_documents.json` (field "text"), runs the model in eval mode
    with no grad over up to `max_docs` documents, and returns the mean per-token
    cross-entropy. This is a REAL held-in memorization signal, unlike the old
    per-batch training loss that was mislabelled "pii_loss".

    Returns NaN if the PII documents file is missing or empty (logged, not fatal).
    """
    pii_path = os.path.join(DATA_DIR, "pii_documents.json")
    if not os.path.exists(pii_path):
        print(f"  [WARN] PII eval skipped: {pii_path} not found")
        return float("nan")

    with open(pii_path) as f:
        docs = json.load(f)
    texts = [d["text"] for d in docs if d.get("text")][:max_docs]
    if not texts:
        print("  [WARN] PII eval skipped: no PII documents with text")
        return float("nan")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    n_tokens = 0
    try:
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=train_cfg.max_seq_length,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc["attention_mask"].to(DEVICE)
                if input_ids.size(1) < 2:
                    continue

                if DEVICE == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids,
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )

                # `outputs.loss` is the mean over the (n-1) predicted tokens;
                # weight by token count so the corpus mean is token-weighted.
                n_pred = input_ids.size(1) - 1
                total_loss += float(outputs.loss.item()) * n_pred
                n_tokens += n_pred
    finally:
        if was_training:
            model.train()

    if n_tokens == 0:
        return float("nan")
    return total_loss / n_tokens


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model_name: str) -> str:
    """
    Fine-tune a single model. Returns the path to the saved model directory.
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Device: {DEVICE} | Profile: {DEVICE_PROFILE}")
    print(f"{'='*60}")

    safe_name = model_name.replace("/", "_")
    save_dir = os.path.join(MODEL_DIR, safe_name)

    if os.path.exists(os.path.join(save_dir, "config.json")):
        print(f"  [SKIP] Model already trained at {save_dir}")
        return save_dir

    use_lora = train_cfg.use_lora_for(model_name)
    use_4bit = train_cfg.use_4bit_for(model_name)
    learning_rate = train_cfg.learning_rate_for(model_name)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_target_modules: Optional[List[str]] = None

    if use_lora:
        # PEFT is only imported on the LoRA path so the module still imports on
        # a CPU-only / minimal box (peft/bitsandbytes may be absent).
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        quant_config = None
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if train_cfg.fp16 else torch.float32,
            )
            print("  Quantization: 4-bit (QLoRA)")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if train_cfg.fp16 else torch.float32,
            quantization_config=quant_config,
            # With 4-bit we let HF/accelerate place the weights on the GPU.
            device_map="auto" if use_4bit else None,
        )

        use_grad_ckpt = train_cfg.needs_gradient_checkpointing(model_name)
        if use_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=use_grad_ckpt
            )
            if use_grad_ckpt:
                print("  Gradient checkpointing: ENABLED (via prepare_model_for_kbit_training)")
        elif use_grad_ckpt:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()  # needed for grad ckpt + PEFT
            print("  Gradient checkpointing: ENABLED")

        lora_target_modules = _infer_lora_targets(model_name, model)
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=train_cfg.lora_r,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        print(f"  LoRA: ENABLED  r={train_cfg.lora_r} alpha={train_cfg.lora_alpha} "
              f"targets={lora_target_modules}")
        model.print_trainable_parameters()

        # Keep trainable (adapter) params in FP32 so autocast + GradScaler work
        # even with an fp16 frozen base (otherwise: "unscale FP16 gradients").
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # 4-bit weights are already placed by device_map; only move otherwise.
        if not use_4bit:
            model.to(DEVICE)
    else:
        # Full fine-tune (small models). Load master weights in FP32; mixed
        # precision is applied by autocast in the loop. Loading the model itself
        # in fp16 makes the GRADIENTS fp16 and breaks the GradScaler
        # ("Attempting to unscale FP16 gradients").
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        use_grad_ckpt = train_cfg.needs_gradient_checkpointing(model_name)
        if use_grad_ckpt:
            model.gradient_checkpointing_enable()
            print("  Gradient checkpointing: ENABLED")

        model.to(DEVICE)

    model.train()

    # Dataset
    corpus_path = os.path.join(DATA_DIR, "corpus", "train.json")
    dataset = CorpusDataset(corpus_path, tokenizer, train_cfg.max_seq_length)

    per_device_batch = train_cfg.effective_batch_size
    grad_accum = max(1, train_cfg.batch_size // per_device_batch)

    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=0,   # 0 = load in the main process. Multiprocess DataLoader
                         # workers can deadlock on HPC (fork + tokenizer); the
                         # on-the-fly tokenization here is cheap enough for 0.
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )

    # Optimizer + scheduler. LoRA optimizes only the adapter params (the rest
    # have requires_grad=False), which is exactly what keeps memory feasible.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    total_steps = len(loader) * train_cfg.num_epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision: prefer bf16 (no GradScaler needed) on Ampere+; else fp16 +
    # GradScaler. Master/trainable weights are fp32 (above), so the scaler is valid.
    _use_amp = train_cfg.fp16 and DEVICE == "cuda"
    _use_bf16 = _use_amp and torch.cuda.is_bf16_supported()
    _amp_dtype = torch.bfloat16 if _use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (_use_amp and not _use_bf16) else None
    print(f"  Mixed precision: {'bf16' if _use_bf16 else ('fp16+scaler' if _use_amp else 'off (fp32)')}")

    print(f"  Corpus size: {len(dataset)}")
    print(f"  Per-device batch: {per_device_batch}, Grad accum: {grad_accum}")
    print(f"  Total steps: {total_steps}, Epochs: {train_cfg.num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Training
    global_step = 0
    pii_eval_losses: List[float] = []

    for epoch in range(train_cfg.num_epochs):
        epoch_loss = 0.0
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            if _use_amp:
                with torch.amp.autocast("cuda", dtype=_amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum
                loss.backward()

            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * grad_accum:.4f}",
                "step": global_step,
            })

        avg_loss = epoch_loss / len(loader)

        # Real per-epoch PII memorization signal (held-in cross-entropy).
        pii_loss = _pii_eval_loss(model, tokenizer)
        pii_eval_losses.append(pii_loss)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}, pii_eval_loss={pii_loss:.4f}")

    # Verify memorization: real per-epoch PII eval loss should be decreasing.
    valid_pii = [x for x in pii_eval_losses if not math.isnan(x)]
    if len(valid_pii) >= 2 and valid_pii[-1] < valid_pii[0]:
        print(f"  PII memorization confirmed (pii eval loss: "
              f"{valid_pii[0]:.4f} -> {valid_pii[-1]:.4f})")
    else:
        print(f"  [WARN] PII eval loss did not clearly decrease: {pii_eval_losses}")

    # Save
    os.makedirs(save_dir, exist_ok=True)

    saved_as_adapter = False
    if use_lora and not use_4bit:
        # Merge the LoRA adapter into the base weights and save a STANDARD full
        # model, so downstream AutoModelForCausalLM.from_pretrained works as-is.
        print("  Merging LoRA adapter into base weights (merge_and_unload)...")
        model = model.merge_and_unload()
        model.save_pretrained(save_dir)
    elif use_lora and use_4bit:
        # QLoRA: cannot merge into 4-bit weights. Save only the adapter and mark
        # the directory so downstream code knows to wrap it with a PeftModel.
        saved_as_adapter = True
        model.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "PEFT_ADAPTER"), "w") as f:
            f.write(model_name + "\n")
        print("  [NOTE] 4-bit (QLoRA) run: saved LoRA ADAPTER only, not a full model.")
        print(f"         Base model: {model_name}")
        print("         Downstream loading requires peft.PeftModel on top of the base "
              "model; a plain AutoModelForCausalLM.from_pretrained(save_dir) will NOT "
              "work for these dirs.")
    else:
        # Full fine-tune — save the full model directly (unchanged).
        model.save_pretrained(save_dir)

    tokenizer.save_pretrained(save_dir)

    # Save training metadata
    meta = {
        "model_name": model_name,
        "epochs": train_cfg.num_epochs,
        "final_loss": avg_loss,
        "pii_eval_losses": pii_eval_losses,
        "gradient_checkpointing": use_grad_ckpt,
        "device_profile": DEVICE_PROFILE,
        "use_lora": use_lora,
        "lora_target_modules": lora_target_modules,
        "load_in_4bit": use_4bit,
        "saved_as_adapter": saved_as_adapter,
        "base_model": model_name if saved_as_adapter else None,
    }
    with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {save_dir}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return save_dir


# ---------------------------------------------------------------------------
# Train all configured models
# ---------------------------------------------------------------------------

def train_all():
    models = train_cfg.get_models()
    print(f"Will train {len(models)} model(s): {models}")

    results = {}
    for m in models:
        path = train_model(m)
        results[m] = path

    print("\nAll models trained:")
    for m, p in results.items():
        print(f"  {m} -> {p}")
    return results


if __name__ == "__main__":
    train_all()
