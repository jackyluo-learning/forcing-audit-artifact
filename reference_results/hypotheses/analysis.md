# E3b H1--H5 central analysis

## Scope and evidence gate

This analysis uses the accepted 42-shard E3b formal set (4,200 attempts, 14 capacities x 3 seeds) after the full `analyze_e3b_repair.validate_and_load` gate. Trained and control people are bootstrapped as two independent blocks; one person draw per block is reused across every k. There are 10,000 replicates with seed 20240601.

## Results

| Hypothesis | Estimate and 95% interval | Verdict |
|---|---|---|
| H1 | Spearman rho = 0.994490 [0.974997, 0.994490] | supported |
| H2 | literal 1% floor; zero-hit Wilson upper = 0.1033; positive joint k = [] | unresolved_at_achieved_sample_size |
| H3 | tau(20) = +0.0400 [-0.0533, +0.1333] | does_not_reject_zero_after_conditional_holm |
| H4 | Weibull AFT gamma = 2.469390 [1.122910, 3.608198] | nominal: proportionality_refuted; family: nominal_ci_excludes_one_but_not_significant_after_conditional_holm |
| H5 (exploratory) | tied-argmax envelope = [6, 32]; quadratic b = -0.011900 [-0.023856, -0.000114] | exploratory_interior_peak_localized |

H2 has no preregistered global test. Its p-value is absent; `p=1` appears only as a conservative placeholder in the conditional four-slot Holm table. The mapping preserves the difference between a confidence interval excluding zero in either direction and a useful positive D-C signal. Under both uniform Wilson/MOVER conventions (target-only and repeated-attempt ICC), the primary method's positive joint point at k=6 disappears.

The required log-normal H4 sensitivity gives gamma = 2.006465 [1.092754, 3.001413].

## Interpretation limits

H5 remains exploratory and underpowered. H4 is conditional on the registered Weibull working distribution; a gamma interval containing 1 would fail to reject proportionality rather than establish equivalence. This analysis does not estimate NLL/AUC/ROC, empirical epsilon, or any claim beyond H1--H5.
