# Historical supplements pending

The initial artifact is usable for the final capacity sweep and the base-model follow-up. The following earlier multi-model materials have not been recovered:

1. Original `run2__E1__*.parquet` per-attempt logs and any matching execution manifests.
2. The four original fine-tuned checkpoints: GPT-2 124M, GPT-2-medium 355M, Pythia-1.4B, and Pythia-2.8B. This release's fine-tuned GPT-2 belongs to the later capacity study and does not fill that historical gap.
3. Associated frozen training data, target selection, configuration and dependency records needed to connect those original runs to the reported aggregates.

The source in `legacy/run2/code/` supplies the available earlier implementation. Code and hard-coded figure values do not independently establish the original run's outcomes.

When recovered, add historical raw records and checkpoint payloads without overwriting the accepted capacity evidence. Record file hashes, verify the model/seed/target/field/probe grid, reproduce the reported tables, and check the additions for author-identifying metadata. Update the evidence map, validation record, release hashes, and limitations together. Do not silently substitute a newly rerun checkpoint for the original one; label a rerun as new evidence.

Additional public-release housekeeping: confirm redistribution terms and attribution for third-party corpus excerpts, and ensure the manuscript's artifact description matches the actually released scope. The initial source repository and an anonymous review mirror are separate publication steps.
