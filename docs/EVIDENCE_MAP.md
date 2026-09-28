# Evidence map

| Result or artifact | Inputs | Entry point / reference output |
|---|---|---|
| Control success versus prompt capacity | `results/attempts/e3b__E3__*.parquet`, 42 manifests, frozen targets | `analyze_e3b_repair.py`; `reference_results/control_curve/` |
| H1–H5 | Same 42 accepted sweep shards | `analyze_e3b_hypotheses.py`; `reference_results/hypotheses/` |
| Base/fine-tuned, trained/control four-cell comparison | 3 `e3b_base__E2B__*` shards plus the 3 sweep shards at k=20 | `analyze_e3b_base_followup.py`; `reference_results/base_followup/` |
| Manuscript capacity figure | Recomputed H1–H5 curve CSV | `figures/fig_e3_capacity_combined.py`; `reference_results/paper_figures/` |
| Earlier cross-model rates and probe spectrum | Reported aggregate constants in the figure scripts | `figures/fig_control_comparison_v2.py`, `figures/fig_spectrum.py`; `legacy/run2/reported_*.csv` |
| Capacity upper-bound illustration | Mathematical parameters in the script; no measured success data | `figures/fig_capacity.py` |
| Audit overview | Schematic only | `figures/pii_overview_v2.png`, `figures/fig_audit_overview_v2.py` |

The independent-person bootstrap in the H1–H5 analysis differs from the matched-pair bootstrap in the four-cell follow-up. Therefore the k=20 D−C point estimate agrees, but its intervals need not be identical. The analysis reports specify their sampling units and limitations.

The H4 solver lives in `analysis_support/censored_models.py`; its pre-existing validation record is adjacent. Only its repository-relative location changed for packaging. H5 remains exploratory. H2 has no prespecified global test; a conservative placeholder in the conditional Holm calculation is not a measured H2 p-value. See the generated report rather than interpreting the summary CSV alone.

`e3b` means the final field-exposure-corrected sweep. The earlier `e3a` evidence is preserved in the authors' working archive and is intentionally excluded from the primary release inputs because it was superseded. Historical multi-model `run2` is a separate study, not the same checkpoint or target set.

Use the focused reproduction entry point for the released E3 and base-model results.
