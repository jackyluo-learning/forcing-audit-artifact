# Supplemental likelihood and diagnostic experiments

This directory is an anonymized source snapshot of the supplemental experiment code. It is kept separate from the frozen GPT-2 capacity sweep in the artifact root. The source files retain their relative layout so commands can be run from this directory; `SOURCE_SNAPSHOT.json` lists their checksums. `reproduce_e5.py` is an artifact-specific integration script and is not part of that unchanged source snapshot. The small bundled `data/memorized_sequences.json` is an example input from the source tree, not a result of the new experiment.

## What the code computes

- `experiments/e5_nll_decomposition.py` scores SSN and email targets from the fine-tuning arm and a control arm under a neutral prompt and an employee-record-style prefix. It records greedy exact-match completions, target NLL in bits, optional base-model comparisons, and the pre-optimization trained/control AUC and TPR at 1% FPR. `--attempt-log` optionally adds optimized-prompt comparisons; the included E5 batch script does not pass that option.
- `src/theory.py` computes generator entropy and counting-bound quantities. This is a theoretical calculation, not a model evaluation.
- `src/diagnostics.py`, `src/logschema.py`, and `src/stats_v3.py` analyze an existing per-attempt log, including copy checks and clustered uncertainty. `experiments/run_audit.py` implements the new log schema. The additional blind-search and one-token enumeration scripts are included as code, without claiming completed runs.
- `scripts/v3_*.sbatch` and `scripts/v3_locate.sh` preserve the submitted job entry points and input-discovery logic. The scripts auto-select the newest available checkpoint and data files unless their `V3_*` variables are set explicitly; the selected paths must be read from a job log before attributing any run to a model or dataset.

## Re-run the prefix experiment from scratch

This path does **not** require a supplied log or fine-tuned checkpoint. On a CUDA machine, install the listed requirements and run:

```bash
cd supplemental/gcg_pe_v3
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_e5.py --plan
python reproduce_e5.py --output-dir runs/reproduce_e5
```

For a quick input-generation check before allocating a GPU, run `python reproduce_e5.py --config configs/smoke.yaml --model-name gpt2 --n-controls 12 --prepare-only --output-dir runs/prepare_smoke`. This writes only the synthetic smoke corpus and D/C records. A 12-trained-person/12-control-person GPT-2 smoke run also completed all stages of the runner, including training, scoring, and provenance output; it is a functionality check, not a paper result.

The runner uses the included `full_fast.yaml` design: 30 trained people, one Pythia-1.4B seed and 200 generated controls, producing 60 trained and 400 control person-field scores for SSN/email. It generates the corpus, trains the Pythia model, generates controls with the **same Faker record generator** but a disjoint seed, and calls E5 with the **matching Pythia base model** rather than the batch script's `gpt2` fallback. It checks that control SSN/email values are absent from the generated corpus. Results are written to `runs/reproduce_e5/e5_bits.json`; `provenance.json` records input hashes, library versions, field exposure, and the resulting summary. A fresh output directory is required for each run.

This command produces a **new run of the same experiment type**. Its public-text sources can change, so exact numeric agreement with an earlier run is not guaranteed. Compare `summary` and the per-target `rows` to assess whether the trained/control pattern recurs; the manuscript's specific E5 values have not yet been regenerated with this release. The runner records field exposure and uses one record generator for both arms.

## Run with existing inputs

The from-scratch path above needs no saved checkpoint or log. If you have a particular run's inputs, you can instead supply its fine-tuned checkpoint, matching base checkpoint, person file, and control file explicitly:

```bash
python experiments/e5_nll_decomposition.py \
  --model /path/to/fine-tuned-checkpoint \
  --base-model /path/to/matching-base-checkpoint \
  --individuals /path/to/individuals.json \
  --controls /path/to/controls.json \
  --fields ssn email --dtype float32 --out /path/to/e5_bits.json
```

The generated file includes per-target `rows` and a `summary`. An optional per-attempt log is needed only for `delta_k` measurements. Neither path should be presented as a numerical replication of an earlier run unless its inputs and scoring conditions match.
