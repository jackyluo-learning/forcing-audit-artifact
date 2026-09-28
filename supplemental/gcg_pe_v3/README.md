# Supplemental likelihood and diagnostic experiments

This directory is an anonymized source snapshot of the supplemental experiment code. It is kept separate from the frozen GPT-2 capacity sweep in the artifact root. The source files retain their relative layout so commands can be run from this directory; `SOURCE_SNAPSHOT.json` lists their checksums. The small bundled `data/memorized_sequences.json` is an example input from the source tree, not a result of the new experiment.

## What the code computes

- `experiments/e5_nll_decomposition.py` scores SSN and email targets from trained people and matched controls under a neutral prompt and an employee-record-style prefix. It records greedy exact-match completions, target NLL in bits, optional base-model comparisons, and the pre-optimization trained/control AUC and TPR at 1% FPR. `--attempt-log` optionally adds optimized-prompt comparisons; the included E5 batch script does not pass that option.
- `src/theory.py` computes generator entropy and counting-bound quantities. This is a theoretical calculation, not a model evaluation.
- `src/diagnostics.py`, `src/logschema.py`, and `src/stats_v3.py` analyze an existing per-attempt log, including copy checks and clustered uncertainty. `experiments/run_audit.py` implements the new log schema. The additional blind-search and one-token enumeration scripts are included as code, without claiming completed runs.
- `scripts/v3_*.sbatch` and `scripts/v3_locate.sh` preserve the submitted job entry points and input-discovery logic. The scripts auto-select the newest available checkpoint and data files unless their `V3_*` variables are set explicitly; the selected paths must be read from a job log before attributing any run to a model or dataset.

## Results and inputs

The source repository did not publish the new run's checkpoint, person/control lists, per-attempt log, batch logs, or result JSONs. Accordingly, this snapshot contains **code but no newly measured E5 or diagnostic observations**. Do not treat a number in a script comment or a manuscript as a recomputed result of this artifact. The expected original output paths, relative to the run checkout, are:

| Job | Output | Log |
|---|---|---|
| Prefix likelihood and greedy continuation | `runs/v3/e5_bits.json` | `logs/v3_e5_<jobid>.out` |
| Theory constants | `runs/v3/theory.json` | `logs/v3_theory_<jobid>.out` |
| Existing-log diagnostics | `runs/v3/diag.json` | `logs/v3_diag_<jobid>.out` |

To repeat E5, supply the fine-tuned checkpoint, its **matching** base checkpoint, the exact training person file, and the matched control file explicitly:

```bash
python experiments/e5_nll_decomposition.py \
  --model /path/to/fine-tuned-checkpoint \
  --base-model /path/to/matching-base-checkpoint \
  --individuals /path/to/individuals.json \
  --controls /path/to/controls.json \
  --fields ssn email --dtype float32 --out /path/to/e5_bits.json
```

The generated file includes per-target `rows` as well as a `summary`; use both when checking the reported sample sizes and score comparison. An optional per-attempt log is required for the `delta_k` measurements. Repeating a run with regenerated controls or a different checkpoint creates new evidence, not a reproduction of the earlier run.
