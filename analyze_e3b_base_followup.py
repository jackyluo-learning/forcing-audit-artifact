"""Validate the frozen E3b k=20 four-cell comparison and summarize its rates.

The original GPT-2 checkpoint supplies the new base-model row.  This is a
single-checkpoint diagnostic, not a leave-one-out causal effect estimate.
Pilot shards are never eligible for this formal analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 1337, 2024)
FIELDS = ("ssn", "email")
ARMS = ("trained", "control")
STATES = ("finetuned", "base")
TARGET_SHA = "76c5f511019978ebad23751bc1a4f2b939807a38280b8a766065715a3205e182"
FT_COMMIT = "a6fd908641489fd43cf7159f5e161b913efcc9e2"
BASE_COMMIT = "1d11bffbaf7fc3806e15fdc5b0281bacee99700e"
BASE_FINGERPRINT = "90265451371a973b9e890f08e56100447117fe838d1b037f0ba8d0c3f48c017b"
SUBSET_HASH = "a01097acbb835207"
GPU = "NVIDIA A100 80GB PCIe"


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_targets(path: Path) -> dict[tuple[str, str, str], tuple[int, str]]:
    _require(path.is_file() and _sha(path) == TARGET_SHA, "frozen target manifest differs")
    raw = json.loads(path.read_text())
    pairs = raw["pairs"]
    _require(len(pairs) == 25, "expected exactly 25 frozen D/C pairs")
    expected = {}
    for i, pair in enumerate(pairs):
        _require(pair["pair_index"] == i, "pair ordering changed")
        for arm in ARMS:
            side = pair[arm]
            for field in FIELDS:
                key = (arm, side["person_id"], field)
                _require(key not in expected, f"duplicate frozen target {key}")
                expected[key] = (i, side["targets"][field]["formatted_target_sha256"])
    _require(len(expected) == 100, "frozen target grid is incomplete")
    return expected


def _validate_manifest(path: Path, state: str, seed: int) -> dict:
    _require(path.is_file(), f"missing manifest: {path}")
    m = json.loads(path.read_text())
    expected_run = "e3b" if state == "finetuned" else "e3b_base"
    expected_exp = "E3" if state == "finetuned" else "E2B"
    _require(m.get("run_id") == expected_run and m.get("exp_id") == expected_exp,
             f"wrong formal run ID or experiment: {path}")
    _require(m.get("shard") == {"model_name": "gpt2", "model_state": state,
                                 "seed": seed, "capacity_k": 20, "fields": list(FIELDS)},
             f"wrong shard coordinate: {path}")
    _require(m.get("code", {}).get("commit") == (FT_COMMIT if state == "finetuned" else BASE_COMMIT)
             and m.get("code", {}).get("dirty") is False, f"code pin failed: {path}")
    _require(m.get("gcg_iters") == 200 and m.get("target_subset_hash") == SUBSET_HASH,
             f"optimizer or target-set pin failed: {path}")
    _require(m.get("n_targets") == 100 and m.get("arm_sizes") == {"D": 25, "C": 25},
             f"arm-size pin failed: {path}")
    _require(m.get("accelerator", {}).get("name") == GPU, f"GPU class differs: {path}")
    frozen = m.get("frozen_targets", {})
    _require(frozen.get("manifest_sha256") == TARGET_SHA, f"frozen manifest pin failed: {path}")
    cfg = m.get("resolved_sweep_config") if state == "finetuned" else m.get("resolved_config")
    _require(isinstance(cfg, dict), f"missing resolved attack config: {path}")
    for name, value in {"gcg_iters": 200, "candidates_per_position_B": 256,
                        "candidate_evaluations_per_step": 512, "candidate_minibatch": 64,
                        "early_stop_on_exact_match": True, "extraction_check_interval": 10,
                        "decision_rule": "field-normalized substring exact_match"}.items():
        _require(cfg.get(name) == value, f"config {name} differs: {path}")
    if state == "base":
        _require(cfg.get("capacity_k") == 20 and cfg.get("n_pairs") == 25,
                 f"base capacity or pair count differs: {path}")
        _require(cfg.get("base_fingerprint_sha256") == BASE_FINGERPRINT and
                 m.get("input_fingerprints", {}).get("base_snapshot", {}).get("sha256")
                 == BASE_FINGERPRINT, f"base-weight pin failed: {path}")
        _require(m.get("pilot") is False, f"pilot shard in formal analysis: {path}")
    else:
        _require(20 in cfg.get("k_grid", []) and cfg.get("seeds") == list(SEEDS),
                 f"fine-tuned grid config differs: {path}")
    return m


def _load_shard(path: Path, state: str, seed: int,
                expected: dict[tuple[str, str, str], tuple[int, str]]) -> pd.DataFrame:
    _require(path.is_file(), f"missing parquet: {path}")
    d = pd.read_parquet(path)
    _require(len(d) == 100, f"expected 100 attempts: {path}")
    for col, value in {"run_id": "e3b" if state == "finetuned" else "e3b_base",
                       "exp_id": "E3" if state == "finetuned" else "E2B",
                       "seed": seed, "model_name": "gpt2", "model_state": state,
                       "probe": "gcg_free", "capacity_k": 20}.items():
        _require(d[col].notna().all() and d[col].eq(value).all(),
                 f"attempt column {col} differs: {path}")
    _require(d["exact_match"].notna().all(), f"missing exact-match outcome: {path}")
    _require(set(d["target_membership"]) == set(ARMS) and
             set(d["field"]) == set(FIELDS), f"arm or field mismatch: {path}")
    actual = {}
    for row in d.itertuples(index=False):
        key = (row.target_membership, row.person_id, row.field)
        _require(key not in actual, f"duplicate attempt: {path}, {key}")
        actual[key] = hashlib.sha256(row.target_string.encode()).hexdigest()
    _require(set(actual) == set(expected), f"attempt target grid differs: {path}")
    _require(all(actual[key] == frozen_hash for key, (_, frozen_hash) in expected.items()),
             f"formatted target values differ: {path}")
    d = d.copy()
    d["pair_index"] = [expected[(a, p, f)][0] for a, p, f in
                       zip(d.target_membership, d.person_id, d.field)]
    return d


def analyze(attempts_dir: Path, manifests_dir: Path, target_manifest: Path,
            output_dir: Path, *, bootstrap_reps: int = 10000) -> dict:
    expected = _expected_targets(target_manifest)
    for prefix, exp in (("e3b", "E3"), ("e3b_base", "E2B")):
        for directory, extension in ((manifests_dir, "json"), (attempts_dir, "parquet")):
            paths = list(directory.glob(f"{prefix}__{exp}__gpt2_*_field-ssn-email_k20.{extension}"))
            _require(len(paths) == 3, f"expected exactly three formal {prefix} k=20 {extension} shards")
    frames = []
    provenance = []
    for state in STATES:
        prefix, exp = ("e3b", "E3") if state == "finetuned" else ("e3b_base", "E2B")
        for seed in SEEDS:
            stem = f"{prefix}__{exp}__gpt2_{seed}_field-ssn-email_k20"
            mp, ap = manifests_dir / f"{stem}.json", attempts_dir / f"{stem}.parquet"
            _validate_manifest(mp, state, seed)
            frames.append(_load_shard(ap, state, seed, expected))
            provenance.append({"state": state, "seed": seed, "manifest": str(mp),
                               "manifest_sha256": _sha(mp), "attempts": str(ap),
                               "attempts_sha256": _sha(ap)})
    d = pd.concat(frames, ignore_index=True)
    _require(len(d) == 600, "four-cell formal grid must contain 600 attempts")

    # Each pair contributes two fields x three attacks to every cell.  Resampling
    # pairs retains all repeated measures on the same synthetic people together.
    cube = np.empty((25, 2, 2, 2, 3), dtype=float)
    for i, state in enumerate(STATES):
        for j, arm in enumerate(ARMS):
            for h, field in enumerate(FIELDS):
                for l, seed in enumerate(SEEDS):
                    part = d[(d.model_state == state) & (d.target_membership == arm) &
                             (d.field == field) & (d.seed == seed)].sort_values("pair_index")
                    _require(part.pair_index.tolist() == list(range(25)), "pair grid incomplete")
                    cube[:, i, j, h, l] = part.exact_match.astype(int).to_numpy()
    rates = cube.mean(axis=(0, 3, 4))
    rng = np.random.default_rng(20260923)
    indices = rng.integers(0, 25, size=(bootstrap_reps, 25))
    boot = cube[indices].mean(axis=(1, 4, 5))

    # Array dimensions are model state [FT, base], then target group [D, C].
    metrics = {
        "ft_D": (rates[0, 0], boot[:, 0, 0]),
        "ft_C": (rates[0, 1], boot[:, 0, 1]),
        "base_D": (rates[1, 0], boot[:, 1, 0]),
        "base_C": (rates[1, 1], boot[:, 1, 1]),
        "tau_rec": (rates[0, 0] - rates[0, 1], boot[:, 0, 0] - boot[:, 0, 1]),
        "tau_mod": (rates[0, 0] - rates[1, 0], boot[:, 0, 0] - boot[:, 1, 0]),
        "delta_A3": (rates[0, 1] - rates[1, 1], boot[:, 0, 1] - boot[:, 1, 1]),
        "tau_base": (rates[1, 0] - rates[1, 1], boot[:, 1, 0] - boot[:, 1, 1]),
        "tau_mod_minus_tau_rec": (rates[0, 1] - rates[1, 0],
                                  boot[:, 0, 1] - boot[:, 1, 0]),
    }
    _require(abs(metrics["tau_mod"][0] - metrics["tau_rec"][0] -
                 metrics["delta_A3"][0] + metrics["tau_base"][0]) < 1e-12,
             "four-cell arithmetic identity failed")
    rows = []
    for name, (point, samples) in metrics.items():
        lo, hi = np.quantile(samples, [0.025, 0.975])
        rows.append({"metric": name, "estimate": float(point), "ci95_low": float(lo),
                     "ci95_high": float(hi),
                     "hits": int(round(point * 150)) if name in
                     ("ft_D", "ft_C", "base_D", "base_C") else None,
                     "n_attempts_per_cell": 150 if name in
                     ("ft_D", "ft_C", "base_D", "base_C") else None,
                     "n_matched_pair_clusters": 25})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "four_cell_rates.csv", index=False)
    delta_a3_low = next(row["ci95_low"] for row in rows if row["metric"] == "delta_A3")
    delta_a3_high = next(row["ci95_high"] for row in rows if row["metric"] == "delta_A3")
    result = {
        "accepted": True, "scope": "one GPT-2 checkpoint, frozen E3b targets at k=20",
        "warning": "Four observed cells do not identify a leave-one-out causal effect or DP epsilon.",
        "formal_shards": 6, "formal_attempts": 600, "base_attempts": 300,
        "cluster_unit": "25 matched person pairs; each retains two fields and three attack seeds",
        "bootstrap_reps": bootstrap_reps, "bootstrap_seed": 20260923,
        "rates_and_contrasts": rows,
        "predicted_direction_observed": bool(metrics["delta_A3"][0] >= 0),
        "delta_A3_ci95_excludes_zero": bool(delta_a3_low > 0 or delta_a3_high < 0),
        "provenance": provenance,
    }
    (output_dir / "four_cell_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-dir", type=Path, default=Path("results/attempts"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("results/manifests"))
    parser.add_argument("--target-manifest", type=Path, default=Path("results/target_sets/e3b.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.attempts_dir, args.manifests_dir, args.target_manifest, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
