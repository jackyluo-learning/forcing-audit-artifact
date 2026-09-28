# Reproduction paths

## 1. Recompute recorded results (CPU)

Use the root README's environment and `python reproduce.py`. The frozen evidence stays in `results/`, new analysis products go to `output/`, and release reference products stay in `reference_results/`.

The H1–H5 output includes conditional and exploratory results; keep those qualifications when quoting them.

For the earlier figure panels, run `python figures/fig_control_comparison_v2.py` and `python figures/fig_spectrum.py`. They render the **reported constants** in their scripts. `python figures/fig_capacity.py` renders a theoretical bound. These are separate from the primary recomputation pipeline.

## 2. Restore frozen inputs

Run `python restore_payloads.py` and `python verify_artifact.py`. Do not regenerate the registry, resample controls, or retrain over the included checkpoint to reproduce the recorded analysis. Those would define different experimental inputs.

## 3. Repeat an attack on GPU (new evidence)

Use a separate Linux GPU environment. The archived runs used Python 3.11.5, PyTorch 2.6.0+cu124, Transformers 5.16.1, and A100 80GB PCIe GPUs. `requirements.txt` lists the training/attack dependencies; its broad version ranges are not a complete environment lockfile. Install a suitable CUDA PyTorch build and the recorded library versions when available. The checkpoint and corpus must first be restored.

The packaging wrapper routes new results to `output/gpu_replication/` and refuses to overwrite an existing replication directory. It retains the original attack implementation and frozen-input checks. Commit any source edits before running; historical provenance checks require a clean checkout.

One fine-tuned shard (100 attempts):

```bash
python run_gpu_replication.py --state finetuned --seed 42 --k 20
```

Repeat for the 14 k values and three seeds listed in the data documentation to run the full sweep. Sequential execution works on one A100; independent shards can run concurrently subject to GPU allocation. The k=0 anchor uses fixed prompting. Other k values use at most 200 GCG steps with early stopping. Runtime depends substantially on k and success/early stopping.

For the base follow-up, download public `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` using a Hugging Face cache snapshot containing exactly these files:

```
config.json
generation_config.json
merges.txt
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
```

The snapshot directory name must be that revision. Its individual hashes are in each released base run manifest. For example, `huggingface_hub.snapshot_download` accepts `repo_id`, `revision`, `allow_patterns` with this seven-file list, and `cache_dir`; use its returned snapshot path. Avoid adding extra files in the snapshot, because the runner fingerprints the directory.

```bash
python run_gpu_replication.py --state base --seed 42 --k 20 --base-snapshot /path/to/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

Repeat for seeds 1337 and 2024. Each recorded base shard took approximately one hour on one A100. The wrapper keeps the historical `e3b_base` run ID only inside the isolated replication directory; it does not replace archived files.

Keep new GPU outputs separate from the released attempt shards. The archived-run validator applies to the released evidence; use each new run's manifest and output directory when analyzing new attacks.

## 4. Additional capabilities

`run_experiments.py`, `experiments.py`, and supporting modules contain broader training/probe/defense capabilities. Run new training or earlier pipelines in a separate checkout so they cannot overwrite the supplied corpus/checkpoint. `legacy/run2/` contains the earlier source snapshot.
