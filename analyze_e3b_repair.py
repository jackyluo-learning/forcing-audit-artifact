"""Focused, fail-closed analysis for the repaired E3b capacity sweep.

This is deliberately narrower than the retired :mod:`analyze_e3` entry point.
It validates the complete frozen E3b run and reports one result only: how the
negative-control (C) exact-match rate changes with prompt capacity.  Trained
targets are loaded solely to prove that every formal shard contains the frozen
100-row target set; they are not used for a D-C comparison.

Outputs
-------
``e3b_repair_analysis.json``
    Machine-readable gate report, C curve, seed rates, and Spearman summary.
``e3b_repair_c_curve.csv``
    One row per k with pooled and per-seed C counts/rates.
``e3b_repair_c_curve.{png,pdf}``
    C-only curve with person-clustered 95% intervals and the registered Wilson
    effective-n rule at all-zero/all-one boundary cells.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import e3_target_manifest as target_manifest_lib
import run_manifest


FORMAL_PREFIX = "e3b__E3__"
OUTPUT_JSON = "e3b_repair_analysis.json"
OUTPUT_CSV = "e3b_repair_c_curve.csv"
OUTPUT_PNG = "e3b_repair_c_curve.png"
OUTPUT_PDF = "e3b_repair_c_curve.pdf"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_240_601

# External identity anchor registered before E3b outcomes.  Checking shards
# only against a caller-supplied target file would be circular: a post-outcome
# replacement target plus 42 mutually consistent shards could otherwise pass.
REGISTERED_TARGET_ANCHOR = {
    "manifest_sha256": "76c5f511019978ebad23751bc1a4f2b939807a38280b8a766065715a3205e182",
    "payload_sha256": "83a63df1cc50818d71ef585042cd6cdce3ce1571fc6ff673ad4289d427b50672",
    "target_values_sha256": "d0b81c7226c797e478269836598b8587484675cf02ee8d2e436787a0b5348cee",
    "pair_assignment_sha256": "48fab4bf6013dbce7c8043f89e5e7b874189c0794362720bda7d735f1778935e",
}

REQUIRED_ATTEMPT_COLUMNS = {
    "run_id",
    "exp_id",
    "seed",
    "model_name",
    "model_state",
    "target_membership",
    "person_id",
    "field",
    "train_frequency",
    "probe",
    "capacity_k",
    "softprompt_norm",
    "lambda_fluency",
    "target_string",
    "target_H_bits",
    "target_len_tokens",
    "prompt_text",
    "prompt_token_ids",
    "forward_passes",
    "steps_run",
    "steps_to_first_success",
    "final_target_nll",
    "generation",
    "gen_len_tokens",
    "exact_match",
    "random_record_match",
    "wallclock_s",
}

EXPECTED_ATTACK_CONFIG = {
    "candidates_per_position_B": 256,
    "candidate_evaluations_per_step": 512,
    "candidate_minibatch": 64,
    "early_stop_on_exact_match": True,
    "extraction_check_interval": 10,
    "decision_rule": "field-normalized substring exact_match",
}


class AnalysisGateError(RuntimeError):
    """Raised when the formal E3b evidence is incomplete or inconsistent."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_hex_digest(value: Any, length: int) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(rf"[0-9a-f]{{{length}}}", value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisGateError(message)


def _normalize_int_list(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if value is None:
        return []
    text = str(value).replace(",", " ")
    return [int(item) for item in text.split()]


def _normalize_str_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    return [item for item in str(value).replace(",", " ").split() if item]


def _load_target_contract(
    path: Path,
    expected_anchor: Mapping[str, str] = REGISTERED_TARGET_ANCHOR,
) -> Dict[str, Any]:
    """Validate everything in the frozen file that does not need source bytes."""
    _require(path.is_file(), f"target manifest does not exist: {path}")
    manifest_sha256 = target_manifest_lib.sha256_file(path)
    _require(
        manifest_sha256 == expected_anchor.get("manifest_sha256"),
        "target manifest does not match the registered pre-outcome SHA-256",
    )
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisGateError(f"cannot read target manifest {path}: {exc}") from exc

    _require(
        manifest.get("schema") == target_manifest_lib.SCHEMA,
        f"unexpected target manifest schema: {manifest.get('schema')!r}",
    )
    _require(manifest.get("study_id") == "capacity_axis_20260902", "wrong target study_id")
    _require(
        manifest.get("selection", {}).get("outcome_columns_read") == [],
        "frozen target selection must declare that it read no outcome columns",
    )
    try:
        target_manifest_lib.enforce_default_contract(manifest)
    except target_manifest_lib.TargetManifestError as exc:
        raise AnalysisGateError(str(exc)) from exc
    _require(
        manifest.get("integrity", {}).get("payload_sha256")
        == target_manifest_lib._payload_digest(manifest),
        "frozen target manifest payload hash mismatch",
    )
    for key in ("payload_sha256", "target_values_sha256", "pair_assignment_sha256"):
        _require(
            manifest.get("integrity", {}).get(key) == expected_anchor.get(key),
            f"target manifest {key} differs from the registered pre-outcome anchor",
        )

    fields = list(manifest["fields"])
    pairs = manifest.get("pairs")
    _require(isinstance(pairs, list) and len(pairs) == 25, "target manifest must have 25 pairs")
    expected_targets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    identities = []
    pair_rows = []
    arm_people: Dict[str, List[str]] = {"trained": [], "control": []}
    trained_tiers: Counter[int] = Counter()
    exposure_counts = {"trained": 0, "control": 0}

    for expected_index, pair in enumerate(pairs):
        _require(pair.get("pair_index") == expected_index, "pair indices are not canonical")
        pair_rows.append((pair["trained"]["person_id"], pair["control"]["person_id"]))
        for arm in ("trained", "control"):
            record = pair.get(arm, {})
            person_id = str(record.get("person_id", ""))
            _require(bool(person_id), f"pair {expected_index} has an empty {arm} person id")
            _require(person_id not in arm_people[arm], f"duplicate {arm} person: {person_id!r}")
            arm_people[arm].append(person_id)
            frequency = int(record.get("frequency", -1))
            if arm == "trained":
                trained_tiers[frequency] += 1
            else:
                _require(frequency == 0, f"control {person_id!r} has nonzero frequency")
            targets = record.get("targets", {})
            _require(set(targets) == set(fields), f"wrong fields for {arm} {person_id!r}")
            for field in fields:
                target = targets[field]
                value_hash = target.get("value_sha256")
                formatted_hash = target.get("formatted_target_sha256")
                _require(
                    _is_hex_digest(value_hash, 64),
                    f"invalid raw target hash for {arm} {person_id!r} {field}",
                )
                _require(
                    _is_hex_digest(formatted_hash, 64),
                    f"invalid formatted target hash for {arm} {person_id!r} {field}",
                )
                exposed = target.get("exposed_in_finetuning_text")
                _require(
                    exposed is (arm == "trained"),
                    f"wrong frozen exposure flag for {arm} {person_id!r} {field}",
                )
                exposure_counts[arm] += int(bool(exposed))
                key = (arm, person_id, field)
                _require(key not in expected_targets, f"duplicate frozen target key: {key}")
                expected_targets[key] = {
                    "frequency": frequency,
                    "value_sha256": value_hash,
                    "formatted_target_sha256": formatted_hash,
                }
                identities.append((arm, person_id, field, value_hash))

    _require(
        set(arm_people["trained"]).isdisjoint(arm_people["control"]),
        "trained and control person sets overlap",
    )
    required_tiers = {
        int(key): int(value)
        for key, value in manifest["selection"]["tier_quotas"].items()
    }
    _require(dict(trained_tiers) == required_tiers, "frozen D tier composition is not 3/7/15")
    _require(exposure_counts == {"trained": 50, "control": 0}, "frozen exposure gate failed")
    _require(
        _canonical_sha256(sorted(identities))
        == manifest["integrity"]["target_values_sha256"],
        "frozen target-value hash mismatch",
    )
    _require(
        _canonical_sha256(pair_rows)
        == manifest["integrity"]["pair_assignment_sha256"],
        "frozen pair-assignment hash mismatch",
    )
    for source_name in ("registry", "corpus", "checkpoint"):
        source_hash = manifest.get("source", {}).get(source_name, {}).get("sha256")
        _require(
            _is_hex_digest(source_hash, 64),
            f"frozen source {source_name} has no valid SHA-256",
        )

    # Recompute the six pre-outcome balance diagnostics from the frozen features.
    try:
        d_features = [
            {field: pair["trained"]["targets"][field]["features"] for field in fields}
            for pair in pairs
        ]
        c_features = [
            {field: pair["control"]["targets"][field]["features"] for field in fields}
            for pair in pairs
        ]
        balance = target_manifest_lib._balance_records(d_features, c_features, fields)
        _require(
            target_manifest_lib.canonical_json(balance)
            == target_manifest_lib.canonical_json(manifest["matching"]["balance"]),
            "frozen balance diagnostics do not match the pair features",
        )
        target_manifest_lib.enforce_balance_gate(balance, fields)
    except (KeyError, TypeError, target_manifest_lib.TargetManifestError) as exc:
        raise AnalysisGateError(f"frozen balance gate failed: {exc}") from exc

    subset_pairs = list(expected_targets)
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "expected_targets": expected_targets,
        "target_subset_hash": run_manifest.target_subset_hash(subset_pairs),
        "fields": fields,
        "k_grid": [int(value) for value in manifest["execution_contract"]["k_grid"]],
        "seeds": [int(value) for value in manifest["execution_contract"]["seeds"]],
        "exposure_counts": exposure_counts,
        "trained_tiers": dict(sorted(trained_tiers.items())),
    }


def _expected_resolved_config(target: Mapping[str, Any]) -> Dict[str, Any]:
    contract = target["manifest"]["execution_contract"]
    return {
        "target_set_id": target["manifest"]["target_set_id"],
        "target_manifest_sha256": target["manifest_sha256"],
        "model_name": contract["model_name"],
        "model_state": contract["model_state"],
        "fields": list(contract["fields"]),
        "k_grid": [int(value) for value in contract["k_grid"]],
        "seeds": [int(value) for value in contract["seeds"]],
        "gcg_iters": int(contract["gcg_iters"]),
        **EXPECTED_ATTACK_CONFIG,
    }


def _validate_raw_config(config: Mapping[str, Any], target: Mapping[str, Any], k: int) -> None:
    expected = _expected_resolved_config(target)
    _require(str(config.get("PII_RUN_ID")) == "e3b", "PII_RUN_ID must be e3b")
    _require(str(config.get("PII_MODELS")) == "gpt2", "PII_MODELS must be gpt2")
    _require(
        _normalize_int_list(config.get("PII_SEEDS")) == expected["seeds"],
        "PII_SEEDS differs from the frozen contract",
    )
    _require(
        _normalize_int_list(config.get("PII_KGRID")) == expected["k_grid"],
        "PII_KGRID differs from the frozen contract",
    )
    _require(int(config.get("PII_CAP_SWEEP_N", -1)) == 25, "PII_CAP_SWEEP_N must be 25")
    _require(
        int(config.get("PII_GCG_ITERS", -1)) == expected["gcg_iters"],
        "PII_GCG_ITERS differs from the frozen contract",
    )
    _require(
        _normalize_str_list(config.get("PII_FIELDS")) == expected["fields"],
        "PII_FIELDS differs from the frozen contract",
    )
    _require(str(config.get("PII_DEVICE_PROFILE")) == "a100_80", "device profile must be a100_80")
    _require(int(config.get("PII_CAP_K", -1)) == k, "PII_CAP_K differs from shard k")
    _require(bool(str(config.get("PII_E3_TARGET_MANIFEST", "")).strip()), "missing PII_E3_TARGET_MANIFEST")


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    path: Path,
    target: Mapping[str, Any],
    expected_formal_commit: str,
) -> Tuple[int, int]:
    shard = manifest.get("shard", {})
    try:
        seed, k = int(shard["seed"]), int(shard["capacity_k"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisGateError(f"{path}: invalid seed/k coordinate") from exc

    _require(manifest.get("schema") == "run-manifest-v1", f"{path}: wrong run manifest schema")
    _require(manifest.get("study_id") == "capacity_axis_20260902", f"{path}: wrong study_id")
    _require(manifest.get("run_id") == "e3b", f"{path}: wrong run_id")
    _require(manifest.get("exp_id") == "E3", f"{path}: wrong exp_id")
    _require(shard.get("model_name") == "gpt2", f"{path}: wrong model")
    _require(shard.get("model_state") == "finetuned", f"{path}: wrong model state")
    _require(list(shard.get("fields", [])) == target["fields"], f"{path}: wrong fields")
    _require(seed in target["seeds"] and k in target["k_grid"], f"{path}: unexpected coordinate")
    _require(manifest.get("target_subset_hash") == target["target_subset_hash"], f"{path}: wrong target subset hash")
    _require(
        int(manifest.get("gcg_iters", -1))
        == int(target["manifest"]["execution_contract"]["gcg_iters"]),
        f"{path}: top-level gcg_iters differs from the frozen contract",
    )
    _require(int(manifest.get("n_targets", -1)) == 100, f"{path}: n_targets must be 100")
    _require(manifest.get("arm_sizes") == {"D": 25, "C": 25}, f"{path}: wrong arm sizes")
    _require(
        manifest.get("tier_composition") == {"1": 3, "5": 7, "20": 15},
        f"{path}: wrong tier composition",
    )

    frozen = manifest.get("frozen_targets", {})
    expected_frozen = {
        "manifest_sha256": target["manifest_sha256"],
        "payload_sha256": target["manifest"]["integrity"]["payload_sha256"],
        "target_values_sha256": target["manifest"]["integrity"]["target_values_sha256"],
        "pair_assignment_sha256": target["manifest"]["integrity"]["pair_assignment_sha256"],
    }
    for key, expected in expected_frozen.items():
        _require(frozen.get(key) == expected, f"{path}: frozen_targets.{key} mismatch")
    _require(
        manifest.get("input_fingerprints") == target["manifest"]["source"],
        f"{path}: input fingerprints differ from frozen target sources",
    )

    expected_config = _expected_resolved_config(target)
    resolved = manifest.get("resolved_sweep_config")
    _require(resolved == expected_config, f"{path}: resolved sweep config mismatch")
    _require(
        manifest.get("resolved_sweep_config_sha256") == _canonical_sha256(resolved),
        f"{path}: resolved sweep config hash mismatch",
    )
    code = manifest.get("code", {})
    _require(code.get("commit") == expected_formal_commit, f"{path}: unexpected formal commit")
    _require(code.get("dirty") is False, f"{path}: code.dirty must be false")
    _require(
        _is_hex_digest(manifest.get("env", {}).get("pip_freeze_sha256_16"), 16),
        f"{path}: missing or invalid environment hash",
    )
    accelerator = manifest.get("accelerator", {})
    _require(accelerator.get("device") == "cuda", f"{path}: accelerator must be CUDA")
    _require("A100" in str(accelerator.get("name", "")).upper(), f"{path}: accelerator is not A100")
    _require(int(accelerator.get("count", 0)) >= 1, f"{path}: no CUDA accelerator recorded")
    _require(
        float(accelerator.get("total_mem_gb", 0.0)) >= 75.0,
        f"{path}: accelerator is not the registered A100 80GB class",
    )
    scheduler = manifest.get("scheduler", {})
    for key in ("slurm_job_id", "slurm_array_job_id", "slurm_array_task_id"):
        _require(bool(str(scheduler.get(key, "")).strip()), f"{path}: missing scheduler.{key}")
    _validate_raw_config(manifest.get("config", {}), target, k)
    return seed, k


def _boolean_series(frame: pd.DataFrame, column: str, path: Path) -> pd.Series:
    _require(not frame[column].isna().any(), f"{path}: {column} contains nulls")
    values = set(frame[column].unique().tolist())
    _require(values <= {True, False, 1, 0}, f"{path}: {column} is not boolean")
    return frame[column].astype(bool)


def _validate_shard_rows(
    frame: pd.DataFrame,
    path: Path,
    seed: int,
    k: int,
    target: Mapping[str, Any],
) -> pd.DataFrame:
    observed_columns = set(frame.columns)
    missing = REQUIRED_ATTEMPT_COLUMNS - observed_columns
    extra = observed_columns - REQUIRED_ATTEMPT_COLUMNS
    _require(
        not missing and not extra,
        f"{path}: attempt schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}",
    )
    _require(len(frame) == 100, f"{path}: expected 100 rows, found {len(frame)}")
    for column in ("run_id", "exp_id", "seed", "model_name", "model_state", "target_membership", "person_id", "field", "train_frequency", "probe", "capacity_k", "target_string"):
        _require(not frame[column].isna().any(), f"{path}: {column} contains nulls")

    exact = _boolean_series(frame, "exact_match", path)
    random_match = _boolean_series(frame, "random_record_match", path)
    frame = frame.copy()
    frame["exact_match"] = exact
    frame["random_record_match"] = random_match

    constants = {
        "run_id": "e3b",
        "exp_id": "E3",
        "seed": seed,
        "model_name": "gpt2",
        "model_state": "finetuned",
        "capacity_k": k,
        "probe": "fixed" if k == 0 else "gcg_free",
    }
    for column, expected in constants.items():
        observed = set(frame[column].tolist())
        _require(observed == {expected}, f"{path}: {column}={observed!r}, expected {expected!r}")

    key_columns = ["target_membership", "person_id", "field"]
    _require(not frame.duplicated(key_columns).any(), f"{path}: duplicate target keys")
    observed_keys = set(map(tuple, frame[key_columns].itertuples(index=False, name=None)))
    expected_keys = set(target["expected_targets"])
    _require(observed_keys == expected_keys, f"{path}: row target set differs from frozen manifest")

    for arm in ("trained", "control"):
        arm_frame = frame[frame["target_membership"] == arm]
        _require(len(arm_frame) == 50, f"{path}: {arm} must have 50 rows")
        _require(arm_frame["person_id"].nunique() == 25, f"{path}: {arm} must have 25 people")
        per_field = arm_frame.groupby("field", observed=True).size().to_dict()
        _require(per_field == {field: 25 for field in target["fields"]}, f"{path}: wrong {arm} field counts")

    control_freq = set(int(value) for value in frame.loc[frame.target_membership == "control", "train_frequency"])
    _require(control_freq == {0}, f"{path}: controls must have train_frequency=0")
    trained_people = (
        frame.loc[frame.target_membership == "trained", ["person_id", "train_frequency"]]
        .drop_duplicates()
    )
    observed_tiers = Counter(int(value) for value in trained_people["train_frequency"])
    _require(dict(observed_tiers) == {1: 3, 5: 7, 20: 15}, f"{path}: trained tiers are not 3/7/15")

    expected_targets = target["expected_targets"]
    target_hash_kind = "value_sha256" if k == 0 else "formatted_target_sha256"
    for row in frame[["target_membership", "person_id", "field", "train_frequency", "target_string"]].itertuples(index=False):
        key = (str(row.target_membership), str(row.person_id), str(row.field))
        frozen = expected_targets[key]
        _require(int(row.train_frequency) == frozen["frequency"], f"{path}: frequency mismatch for {key}")
        _require(
            _sha256_text(str(row.target_string)) == frozen[target_hash_kind],
            f"{path}: target string hash mismatch for {key}",
        )
    return frame


def validate_and_load(
    attempts_dir: Path,
    manifests_dir: Path,
    target_manifest: Path,
    expected_formal_commit: str,
    expected_target_anchor: Mapping[str, str] = REGISTERED_TARGET_ANCHOR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the complete 42-shard gate and return the validated attempt table."""
    _require(
        _is_hex_digest(expected_formal_commit, 40),
        "expected formal commit must be a full lowercase 40-character Git SHA",
    )
    target = _load_target_contract(target_manifest, expected_target_anchor)
    manifest_paths = sorted(manifests_dir.glob(f"{FORMAL_PREFIX}*.json"))
    attempt_paths = sorted(attempts_dir.glob(f"{FORMAL_PREFIX}*.parquet"))
    _require(len(manifest_paths) == 42, f"expected 42 formal manifests, found {len(manifest_paths)}")
    _require(len(attempt_paths) == 42, f"expected 42 formal attempt shards, found {len(attempt_paths)}")
    _require(
        all("e3b_repro" not in path.name for path in manifest_paths + attempt_paths),
        "reproducibility shard entered the formal file set",
    )

    manifest_stems = {path.stem for path in manifest_paths}
    attempt_stems = {path.stem for path in attempt_paths}
    _require(manifest_stems == attempt_stems, "formal manifest/parquet stems do not match")

    comparison = run_manifest.compare([str(path) for path in manifest_paths])
    _require(comparison.get("strict_e3b") is True, "run_manifest.compare did not enter strict E3b mode")
    _require(comparison.get("ok") is True, "run_manifest.compare failed: " + "; ".join(comparison.get("errors", [])))

    coordinates = set()
    frames = []
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        seed, k = _validate_run_manifest(
            manifest, manifest_path, target, expected_formal_commit
        )
        _require((seed, k) not in coordinates, f"duplicate formal coordinate {(seed, k)}")
        coordinates.add((seed, k))
        attempt_path = attempts_dir / f"{manifest_path.stem}.parquet"
        try:
            frame = pd.read_parquet(attempt_path)
        except Exception as exc:
            raise AnalysisGateError(f"cannot read {attempt_path}: {exc}") from exc
        frames.append(_validate_shard_rows(frame, attempt_path, seed, k, target))

    expected_coordinates = {(seed, k) for seed in target["seeds"] for k in target["k_grid"]}
    _require(coordinates == expected_coordinates, "formal coordinates do not equal the frozen 14 x 3 grid")
    attempts = pd.concat(frames, ignore_index=True)
    _require(len(attempts) == 4_200, f"expected 4,200 rows, found {len(attempts)}")
    full_key = ["seed", "capacity_k", "target_membership", "person_id", "field"]
    _require(not attempts.duplicated(full_key).any(), "duplicate rows in the 4,200-row formal key")
    counts = attempts.groupby(["seed", "capacity_k"], observed=True).size()
    _require(len(counts) == 42 and bool((counts == 100).all()), "not every coordinate has 100 rows")
    k0_release = attempts[
        (attempts["seed"] == 42)
        & (attempts["capacity_k"] == 0)
        & (attempts["target_membership"] == "control")
    ]
    _require(len(k0_release) == 50, "formal seed=42, k=0 C release cell must have 50 rows")
    _require(
        int(k0_release["exact_match"].sum()) == 0,
        "formal seed=42, k=0 release gate failed: C exact-match count is not zero",
    )

    gate = {
        "passed": True,
        "formal_manifest_count": len(manifest_paths),
        "formal_attempt_shard_count": len(attempt_paths),
        "attempt_rows": len(attempts),
        "coordinates": len(coordinates),
        "expected_formal_commit": expected_formal_commit,
        "target_manifest_sha256": target["manifest_sha256"],
        "target_subset_hash": target["target_subset_hash"],
        "k_grid": target["k_grid"],
        "seeds": target["seeds"],
        "fields": target["fields"],
        "frozen_exposure_counts": target["exposure_counts"],
        "trained_frequency_tiers": {str(key): value for key, value in target["trained_tiers"].items()},
        "accelerator_class": "A100",
        "k0_seed42_control_exact_match_count": 0,
        "repro_shards_included": 0,
    }
    return attempts, gate


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if bool((counts > 1).any()):
        sums = np.zeros(len(counts), dtype=float)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    x_array, y_array = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    if int(keep.sum()) < 3:
        return float("nan")
    x_rank, y_rank = _rank(x_array[keep]), _rank(y_array[keep])
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = np.sqrt((x_rank ** 2).sum() * (y_rank ** 2).sum())
    return float((x_rank * y_rank).sum() / denominator) if denominator > 0 else float("nan")


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _wilson_interval(rate: float, effective_n: float, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson interval evaluated at a prespecified clustering-adjusted n."""
    denominator = 1.0 + z * z / effective_n
    center = (rate + z * z / (2.0 * effective_n)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1.0 - rate) / effective_n + z * z / (4.0 * effective_n ** 2))
        / denominator
    )
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


def analyze_control_curve(
    attempts: pd.DataFrame,
    *,
    k_grid: Sequence[int],
    seeds: Sequence[int],
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Estimate the pooled C curve using one person draw across every k."""
    _require(n_bootstrap > 0, "n_bootstrap must be positive")
    control = attempts[attempts["target_membership"] == "control"].copy()
    _require(len(control) == 2_100, f"expected 2,100 control rows, found {len(control)}")
    persons = sorted(str(value) for value in control["person_id"].unique())
    _require(len(persons) == 25, f"expected 25 control people, found {len(persons)}")
    k_grid = [int(value) for value in k_grid]
    seeds = [int(value) for value in seeds]
    person_index = {person: index for index, person in enumerate(persons)}
    k_index = {k: index for index, k in enumerate(k_grid)}

    successes = np.zeros((len(persons), len(k_grid)), dtype=float)
    totals = np.zeros_like(successes)
    for row in control[["person_id", "capacity_k", "exact_match"]].itertuples(index=False):
        i, j = person_index[str(row.person_id)], k_index[int(row.capacity_k)]
        successes[i, j] += int(bool(row.exact_match))
        totals[i, j] += 1
    _require(bool((totals == len(seeds) * 2).all()), "each C person/k must retain both fields and all three seeds")

    pooled_hits = successes.sum(axis=0)
    pooled_totals = totals.sum(axis=0)
    pooled_rates = pooled_hits / pooled_totals
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(persons), size=(n_bootstrap, len(persons)))
    boot_hits = successes[draws].sum(axis=1)
    boot_totals = totals[draws].sum(axis=1)
    boot_rates = boot_hits / boot_totals
    ci_low, ci_high = np.percentile(boot_rates, [2.5, 97.5], axis=0)

    positive_indices = [index for index, k in enumerate(k_grid) if k > 0]
    positive_k = np.asarray([k_grid[index] for index in positive_indices], dtype=float)
    rho_point = spearman_rho(positive_k, pooled_rates[positive_indices])
    rho_replicates = np.asarray([
        spearman_rho(positive_k, replicate[positive_indices])
        for replicate in boot_rates
    ])
    valid_rho = rho_replicates[np.isfinite(rho_replicates)]
    minimum_valid_rho = int(np.ceil(0.95 * n_bootstrap))
    rho_ci_available = len(valid_rho) >= minimum_valid_rho
    rho_ci = (
        np.percentile(valid_rho, [2.5, 97.5])
        if rho_ci_available
        else np.asarray([np.nan, np.nan])
    )

    seed_rows = []
    for k in k_grid:
        for seed in seeds:
            cell = control[(control["capacity_k"] == k) & (control["seed"] == seed)]
            _require(len(cell) == 50, f"C seed cell k={k}, seed={seed} does not have 50 rows")
            hits = int(cell["exact_match"].sum())
            seed_rows.append({
                "k": k,
                "seed": seed,
                "hits": hits,
                "count": len(cell),
                "rate": hits / len(cell),
                "random_record_match_count": int(cell["random_record_match"].sum()),
            })

    seed_lookup = {(row["k"], row["seed"]): row for row in seed_rows}
    curve_rows = []
    csv_rows = []
    # The registered boundary rule treats the two fields as repeated measures
    # with assumed ICC=.5 and does not multiply n by the three fixed seeds.
    boundary_effective_n = len(persons) * 2 / (1 + (2 - 1) * 0.5)
    for index, k in enumerate(k_grid):
        cell = control[control["capacity_k"] == k]
        random_count = int(cell["random_record_match"].sum())
        person_bootstrap_ci = [float(ci_low[index]), float(ci_high[index])]
        is_boundary = pooled_hits[index] in (0, pooled_totals[index])
        if is_boundary:
            reported_ci = list(_wilson_interval(float(pooled_rates[index]), boundary_effective_n))
            ci_method = "wilson_effective_n_at_boundary"
        else:
            reported_ci = person_bootstrap_ci
            ci_method = "person_clustered_percentile_bootstrap"
        record = {
            "k": k,
            "hits": int(pooled_hits[index]),
            "count": int(pooled_totals[index]),
            "rate": float(pooled_rates[index]),
            "reported_95_ci": reported_ci,
            "ci_method": ci_method,
            "person_bootstrap_95_ci": person_bootstrap_ci,
            "random_record_match_count": random_count,
        }
        curve_rows.append(record)
        csv_row = {
            "k": k,
            "c_hits": record["hits"],
            "c_count": record["count"],
            "c_rate": record["rate"],
            "c_ci_low": record["reported_95_ci"][0],
            "c_ci_high": record["reported_95_ci"][1],
            "c_ci_method": record["ci_method"],
            "c_person_bootstrap_ci_low": record["person_bootstrap_95_ci"][0],
            "c_person_bootstrap_ci_high": record["person_bootstrap_95_ci"][1],
            "c_random_record_match_count": random_count,
        }
        for seed in seeds:
            seed_record = seed_lookup[(k, seed)]
            csv_row[f"seed_{seed}_hits"] = seed_record["hits"]
            csv_row[f"seed_{seed}_count"] = seed_record["count"]
            csv_row[f"seed_{seed}_rate"] = seed_record["rate"]
            csv_row[f"seed_{seed}_random_record_match_count"] = seed_record["random_record_match_count"]
        csv_rows.append(csv_row)

    result = {
        "estimand": "pooled control exact-match rate at each k",
        "unit": "attempt; 25 control people x 2 fields x 3 fixed attack seeds per k",
        "curve": curve_rows,
        "bootstrap": {
            "method": "person-clustered percentile bootstrap; one C-person draw reused across every k, preserving both fields and all three seeds",
            "boundary_rule": "At 0/n or n/n, the reported interval is Wilson with n_eff=25*2/(1+(2-1)*0.5)=33.333; the raw person-bootstrap interval remains in the output for audit.",
            "boundary_effective_n": boundary_effective_n,
            "replicates": n_bootstrap,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "n_control_people": len(persons),
        },
        "spearman_positive_k": {
            "k_values": [int(value) for value in positive_k],
            "rho": _json_number(rho_point),
            "bootstrap_95_ci": [_json_number(rho_ci[0]), _json_number(rho_ci[1])],
            "valid_bootstrap_replicates": int(len(valid_rho)),
            "total_bootstrap_replicates": n_bootstrap,
            "minimum_valid_replicates_for_ci": minimum_valid_rho,
            "ci_status": "ok" if rho_ci_available else "insufficient_nonconstant_replicates",
        },
        "seed_rates": seed_rows,
        "control_random_record_match_count": int(control["random_record_match"].sum()),
        "control_attempt_count": len(control),
    }
    return result, pd.DataFrame(csv_rows)


def write_figure(curve: pd.DataFrame, output_dir: Path, seeds: Sequence[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = curve["k"].to_numpy(dtype=float)
    rates = curve["c_rate"].to_numpy(dtype=float)
    low = curve["c_ci_low"].to_numpy(dtype=float)
    high = curve["c_ci_high"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(8.4, 5.1))
    seed_styles = ("--", "-.", ":")
    for seed_index, seed in enumerate(seeds):
        axis.plot(
            x,
            curve[f"seed_{seed}_rate"],
            color="#8b96a5",
            alpha=0.48,
            linewidth=1.0,
            linestyle=seed_styles[seed_index % len(seed_styles)],
            marker="o",
            markersize=2.8,
            label=f"seed {seed}",
        )
    axis.errorbar(
        x,
        rates,
        yerr=np.vstack((rates - low, high - rates)),
        fmt="o-",
        color="#1769aa",
        ecolor="#1769aa",
        linewidth=2.2,
        elinewidth=1.2,
        capsize=3,
        markersize=4.8,
        label="pooled C rate (95% CI; boundary Wilson)",
        zorder=5,
    )
    axis.axvline(0.5, color="#707070", linestyle="--", linewidth=0.9)
    axis.annotate(
        "fixed anchor",
        xy=(0, rates[0]),
        xytext=(0, min(0.92, rates[0] + 0.11)),
        ha="center",
        va="bottom",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#707070", "linewidth": 0.8},
    )
    axis.set_xscale("symlog", linthresh=1, base=2)
    axis.set_xticks([0, 1, 2, 4, 8, 16, 32, 64])
    axis.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "64"])
    axis.set_xlabel("Prompt capacity k (free tokens)")
    axis.set_ylabel("Control exact-match rate")
    axis.set_ylim(-0.025, 1.025)
    axis.set_xlim(-0.12, 72)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        fontsize=8,
        ncol=4,
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(output_dir / OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    *,
    attempts_dir: Path,
    manifests_dir: Path,
    target_manifest: Path,
    output_dir: Path,
    expected_formal_commit: str,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_target_anchor: Mapping[str, str] = REGISTERED_TARGET_ANCHOR,
) -> Dict[str, Path]:
    attempts, gate = validate_and_load(
        attempts_dir=attempts_dir,
        manifests_dir=manifests_dir,
        target_manifest=target_manifest,
        expected_formal_commit=expected_formal_commit,
        expected_target_anchor=expected_target_anchor,
    )
    result, curve = analyze_control_curve(
        attempts,
        k_grid=gate["k_grid"],
        seeds=gate["seeds"],
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "e3b-repair-focused-analysis-v1",
        "scope": "C-only prompt-capacity result; D rows are used only by the completeness gate",
        "gate": gate,
        "control_capacity_result": result,
        "excluded_analyses": [
            "H2-H5",
            "NLL/AUC/ROC",
            "D-C inference",
            "multiple-testing-family claims",
        ],
    }
    filenames = (OUTPUT_JSON, OUTPUT_CSV, OUTPUT_PNG, OUTPUT_PDF)
    with tempfile.TemporaryDirectory(prefix=".e3b-analysis-", dir=output_dir) as staging_name:
        staging = Path(staging_name)
        (staging / OUTPUT_JSON).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        curve.to_csv(staging / OUTPUT_CSV, index=False)
        write_figure(curve, staging, gate["seeds"])
        for filename in filenames:
            (staging / filename).replace(output_dir / filename)
    return {
        "json": output_dir / OUTPUT_JSON,
        "csv": output_dir / OUTPUT_CSV,
        "png": output_dir / OUTPUT_PNG,
        "pdf": output_dir / OUTPUT_PDF,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-dir", required=True, type=Path)
    parser.add_argument("--manifests-dir", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-formal-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        paths = run_analysis(
            attempts_dir=args.attempts_dir,
            manifests_dir=args.manifests_dir,
            target_manifest=args.target_manifest,
            output_dir=args.output_dir,
            expected_formal_commit=args.expected_formal_commit,
        )
    except AnalysisGateError as exc:
        raise SystemExit(f"E3b analysis blocked: {exc}") from exc
    print("E3b focused C-only analysis passed all gates")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
