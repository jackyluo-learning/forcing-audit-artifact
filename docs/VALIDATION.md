# Release validation

Validated on Python 3.13.0 with the versions in `requirements-analysis.txt`.

- Full CPU reproduction completed: control curve, H1–H5, base-model comparison, and capacity figure.
- All 27 included unit tests passed.
- All 45 Parquet files (4,500 attempts) are byte-identical to the recovered source evidence.
- Control-rate curves, D/C curves and four-cell rates agree with the prior accepted analyses (relative tolerance 1e-10, absolute tolerance 1e-12). All H1–H5 verdicts are unchanged.
- H4 iterative optimization has small platform-level differences: the Weibull slope agrees to machine precision, and the log-normal sensitivity slope differs by less than 0.000001. Displayed scientific conclusions are unchanged. Full JSON files and rendered figures are not promised to be byte-identical across platforms.
- Both payloads were restored and verified against their recorded full-file sizes and SHA-256 hashes.
- Release source and documentation, raw attempt text, JSON metadata, and figure metadata were checked for project-author identifiers, personal paths, and credential patterns. Generic public-source text was preserved. No author-identifying Git history is carried into this repository.
- The new GPU output-routing wrapper passed syntax checks; it was not executed on a GPU during packaging. Its limitations are stated in the reproduction guide.

`verify_artifact.py` checks release file hashes, all payload parts, the reconstructed payload hashes and the 42+3 shard inventory without third-party packages. The analysis scripts separately validate row-level/configuration contracts. A checksum detects changes relative to this release; it is not an independent attestation that a past run occurred.

See `validation.json` for the compact record. Historical multi-model evidence remains incomplete as described in `PENDING_ARTIFACTS.md`.
