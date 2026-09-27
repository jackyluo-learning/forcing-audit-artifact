"""Dump the E3 sweep's raw attempt table, unaggregated and unmodified.

Files written:
  results/e3_raw_attempts.csv.gz  every attempt, every column, 4200 rows
  results/e3_hit_matrix.csv       per target x capacity, hits out of 3 seeds
  results/e3_nll_matrix.csv       per target x capacity, mean final_target_nll

The matrices are pivots of the first file, not new measurements.
"""
import glob, os
import pandas as pd

files = sorted(glob.glob("results/attempts/e3a__*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"{len(df)} rows from {len(files)} shards")
print(f"columns ({len(df.columns)}): {list(df.columns)}\n")

df.to_csv("results/e3_raw_attempts.csv.gz", index=False, compression="gzip")

df["tid"] = df.target_membership + "|" + df.field + "|" + df.person_id
df["h"] = df.exact_match.fillna(False).astype(int)
meta = df.groupby("tid").agg(H_bits=("target_H_bits", "max"),
                             H_bits_nunique=("target_H_bits", "nunique"),
                             len_tokens=("target_len_tokens", "max"),
                             train_frequency=("train_frequency", "max"))
df.pivot_table(index="tid", columns="capacity_k", values="h",
               aggfunc="sum").join(meta).to_csv("results/e3_hit_matrix.csv")
df.pivot_table(index="tid", columns="capacity_k", values="final_target_nll",
               aggfunc="mean").join(meta).to_csv("results/e3_nll_matrix.csv")

print("wrote:")
for p in ("results/e3_raw_attempts.csv.gz", "results/e3_hit_matrix.csv",
          "results/e3_nll_matrix.csv"):
    print(f"  {p}  {os.path.getsize(p)/1024:.0f} KB")

# ---- request 1, split by field: pooling hides that email saturates ----
print("\n" + "=" * 78)
print("EXTRACTION COUNTS BY CAPACITY AND ARM, SPLIT BY FIELD (hits / attempts)")
print("=" * 78)
print(f"{'k':>4} | {'ssn D':>12} {'ssn C':>12} | {'email D':>12} {'email C':>12}")
print("-" * 78)
for k, g in df.groupby("capacity_k"):
    cells = []
    for f in ("ssn", "email"):
        for arm in ("trained", "control"):
            s = g[(g.field == f) & (g.target_membership == arm)]
            cells.append(f"{int(s.h.sum()):>4}/{len(s):<4} {s.h.mean():.3f}")
    print(f"{k:>4} | {cells[0]:>12} {cells[1]:>12} | {cells[2]:>12} {cells[3]:>12}")

# ---- the H_bits discrepancy, settled ----
print("\n" + "=" * 78)
print("target_H_bits INTEGRITY")
print("=" * 78)
print(f"  null rows            : {df.target_H_bits.isna().sum()} / {len(df)}")
print(f"  distinct values per target (should be 1): "
      f"{dict(meta.H_bits_nunique.value_counts().sort_index())}")
print(f"  targets with 0 non-null H_bits: {int(meta.H_bits.isna().sum())}")
for f, g in meta.join(df.groupby('tid').target_membership.first()).groupby("field"
        if "field" in meta.columns else df.groupby('tid').field.first()):
    pass
tm = df.groupby("tid").agg(field=("field", "first"), arm=("target_membership", "first"))
mm = meta.join(tm)
print(f"\n  {'field':<7} {'arm':<9} {'n':>3} {'H mean':>9} {'H min':>9} {'H max':>9}")
for (f, a), g in mm.groupby(["field", "arm"]):
    print(f"  {f:<7} {a:<9} {len(g):>3} {g.H_bits.mean():>9.2f} {g.H_bits.min():>9.2f} {g.H_bits.max():>9.2f}")
