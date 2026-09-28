# Data and provenance

## Released observations

The capacity sweep uses k = 0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64 and optimizer seeds 42, 1337, 2024. Each shard contains 100 attempts: 25 people per arm, two fields (SSN and email). All 50 D field values occur in the frozen training text; none of the 50 C field values do. This does not certify absence from the base model's pretraining data.

D/C labels describe exposure in fine-tuning. The same labels remain attached to targets when testing the base model. Three seeds repeat the attack; they do not supply three independently trained models. At k=0 a fixed-prompt anchor is used; positive k uses context-free GCG.

The decision rule is field-normalized substring exact match. `exact_match` is the recorded binary outcome; `prompt_text`/`prompt_token_ids` are the optimized input, and `generation` is the generated text. `target_string`, `field`, `person_id`, `target_membership`, `model_state`, `seed`, and `capacity_k` identify the trial. See `attempt_log.ATTEMPT_COLUMNS` for all 27 columns. `final_target_nll` remains a raw logged field, but the released focused analyses do not use it for ROC/AUC or a DP estimate. `target_H_bits` is reference-model self-information, not the theoretical distribution min-entropy.

The training corpus has 101,360 documents: 1,360 synthetic-person documents and 100,000 public passages (75,012 Wikipedia and 24,988 arXiv). The corpus is **not wholly synthetic or public-domain**. Targets were generated synthetically; chance overlap with real strings is possible. See `NOTICE.md` for source and licensing limitations. The fetched-source counts in `corpus_metadata.json` precede truncation to the final corpus size and therefore differ from retained counts.

## Anonymization

No upstream Git history, remotes, author profiles, credentials, cluster hostnames, or private mapping file is included. Historical execution commit IDs and scheduler IDs were replaced with consistent opaque aliases. A 40-hex `code.commit` value in an archived manifest is an **alias, not a resolvable commit in this artifact repository**. The historical `code.dirty` value describes the original run; it is not a claim about the packaging checkout.

The private alias mapping is retained separately. Frozen target-manifest metadata changed during anonymization, so its integrity hashes and dependent manifest checksums were recomputed. The analyzers' metadata pins were changed consistently. Target strings, pair assignments, attempt outcomes, seed values, numerical configurations, Parquet files, corpus bytes, and checkpoint bytes were not edited. The H4 solver path moved to `analysis_support/` without a solver change. `SHA256SUMS` identifies this release's files, not the original execution Git commit.

## Integrity and interpretation

The analysis gate checks the exact 42-shard/4,200-row grid, the expected execution alias, clean historical state, A100 hardware, frozen target values and assignments, input fingerprints, exposure audit, and registered execution settings. The base follow-up checks its three shards and matching fine-tuned k=20 evidence. These gates validate consistency of the released E3 and base-model records.

The frozen corpus and checkpoint are stored in verified chunks; model parameters are unchanged. The base model itself is obtained from public `openai-community/gpt2` at the revision recorded in the base manifests. GPU run manifests record Python 3.11.5, PyTorch 2.6.0+cu124, Transformers 5.16.1, and lifelines 0.30.0. They retain an environment fingerprint, but that hash alone is not a complete environment lockfile.
