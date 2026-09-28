# Control-calibrated extraction auditing — artifact

This independent repository packages the available implementation and evidence for control-calibrated language-model extraction audits. Targets are synthetic. Trained targets (D) and matched control targets (C) receive the same attack and decision rule.

## What is included

| Material | Available in this release |
|---|---|
| Fine-tuned GPT-2 prompt-capacity sweep | 42 attempt shards and manifests; 4,200 attempts, 14 capacities, 3 optimizer seeds |
| Base GPT-2 follow-up at k=20 | 3 additional shards and manifests; 300 attempts on the same targets |
| Frozen inputs | Target registry, matched D/C target manifest, training corpus, fine-tuned GPT-2 checkpoint and tokenizer |
| Analysis | Control curve, H1–H5 analysis, four-cell base/fine-tuned comparison, CSV/JSON results and PNG/PDF figures |
| Implementation | Data generation, training, probes, experiment runners, logging, analysis, and 27 focused tests |
| Earlier multi-model figures | Source snapshot and figure scripts with reported aggregate values |
| Supplemental likelihood and diagnostic experiments | Source snapshot, batch scripts, and a from-scratch E5 runner in [`supplemental/gcg_pe_v3/`](supplemental/gcg_pe_v3/); see its README for the scope of numerical reproduction |

The corrected capacity sweep is the primary released sweep. Its internal file ID is `e3b`; this is provenance bookkeeping, not an additional experiment claimed in the paper. The three seeds repeat optimization on one checkpoint, not independent model training. The 300 fine-tuned k=20 attempts used in the four-cell comparison are reused from the 4,200-row sweep: there are **4,500 unique released attempts**, not 4,800.

## Reproduce the analyses on CPU

Tested with Python 3.13. No GPU, model download, or restoration of the large payloads is needed for this path.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python verify_artifact.py
python reproduce.py --quick
python reproduce.py
python -m unittest discover -s tests -v
```

`--quick` produces the control curve and the base-model comparison. The full command additionally runs the 10,000-replicate H1–H5 analyses and the manuscript capacity figure; allow several minutes depending on the CPU. Results go to `output/`. Checked-in copies are in [reference_results/](reference_results/).

## Restore the checkpoint and corpus

```bash
python restore_payloads.py
```

Large files are stored as 7-MiB parts under `payloads/` for mirrors with per-file size limits. Restoration verifies every part and the complete files against SHA-256 hashes. It restores the 497,774,208-byte fine-tuned weights and 57,979,000-byte mixed training corpus. Small checkpoint/tokenizer files are already in `models/gpt2/`. Keep at least 2 GB free for the checkout, restored payloads and analysis outputs, excluding dependencies or the base-model download.

For new GPU runs, see [reproduction instructions](docs/REPRODUCTION.md). New attack runs can vary with hardware/software and must remain separate from the recorded evidence.

See [evidence map](docs/EVIDENCE_MAP.md), [data and anonymization](docs/DATA_AND_PROVENANCE.md), [validation](docs/VALIDATION.md), and [third-party notices](NOTICE.md).
