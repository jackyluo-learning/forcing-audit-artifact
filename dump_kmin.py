"""Dump the per-target k_min table so the H4 fit can be debugged off-cluster.

50 control rows plus 50 trained. Tiny, and it is the exact input fit_loglog
receives -- debugging against anything else would be debugging a different
problem.
"""
import glob, pandas as pd, numpy as np
import analyze_e3 as A

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("results/attempts/e3a__*.parquet"))],
               ignore_index=True)
ks_pos = [k for k in sorted(int(k) for k in df.capacity_k.dropna().unique()) if k > 0]
km = A.kmin_table(df, ks_pos)
km.to_csv("results/e3_kmin.csv", index=False)
print(km.groupby(["arm", "field"]).agg(
    n=("k_min", "size"),
    censored=("k_min", lambda s: int(s.isna().sum())),
    kmin_med=("k_min", "median"),
    H_min=("H_bits", "min"), H_med=("H_bits", "median"), H_max=("H_bits", "max"),
    len_med=("len_tokens", "median")).to_string())
print("\nlog H 跨度 (control):",
      round(float(np.log(km[km.arm=='control'].H_bits).max()
                  - np.log(km[km.arm=='control'].H_bits).min()), 4))
print("\n-> results/e3_kmin.csv")
