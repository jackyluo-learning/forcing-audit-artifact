"""Build and verify the frozen target set for the E3b repair sweep.

The original E3 path selected people at run time.  That made two different
operations look like one: field-level exposure was inferred from a positive
person-level frequency, and E17's target pairs were collapsed to a set of
control people before E3 selected a second, unrelated subset.  E3b instead
creates one immutable input before any new outcome is observed:

* D is sampled from people whose SSN *and* email occur in the fine-tuning text;
* the pre-existing 3/7/15 frequency-tier allocation is retained;
* 25 distinct control people are assigned without replacement by a joint
  two-field Hungarian match; and
* the data, checkpoint, target values, covariates, and assignment are hashed.

The module deliberately has no torch/transformers imports at module load time.
Structural validation and unit tests therefore run on analysis/login machines;
only the ``build`` CLI subcommand loads the reference language model used for
the preregistered H(t) matching covariate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import unicodedata

import numpy as np
from scipy.optimize import linear_sum_assignment


SCHEMA = "e3-frozen-targets-v1"
DEFAULT_FIELDS = ("ssn", "email")
DEFAULT_TIER_QUOTAS = {1: 3, 5: 7, 20: 15}
DEFAULT_K_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64)
DEFAULT_SEEDS = (42, 1337, 2024)
FEATURE_NAMES = ("char_len", "tok_len", "H_bits")
DEFAULT_BALANCE_THRESHOLD = 0.1
DEFAULT_GCG_ITERS = 200
DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_TARGET_SET_ID = "e3b"


class TargetManifestError(RuntimeError):
    """Raised when a frozen target set is missing, altered, or inconsistent."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: os.PathLike[str] | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise TargetManifestError(f"Required input is not a file: {path}")
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def checkpoint_fingerprint(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Hash every regular file that can affect checkpoint loading.

    This is intentionally a content fingerprint rather than a directory mtime.
    A model weight hash alone would not bind tokenizer/config changes.
    """
    root = Path(path)
    if not root.is_dir():
        raise TargetManifestError(f"Checkpoint directory is missing: {root}")
    records: Dict[str, Dict[str, Any]] = {}
    for item in sorted(root.rglob("*")):
        if item.is_file():
            records[str(item.relative_to(root))] = file_record(item)
    if not records:
        raise TargetManifestError(f"Checkpoint directory is empty: {root}")
    weight_files = [
        name for name in records
        if name.endswith((".safetensors", ".bin", ".pt", ".pth"))
    ]
    if not weight_files:
        raise TargetManifestError(f"Checkpoint has no model weight file: {root}")
    return {
        "files": records,
        "sha256": canonical_sha256(records),
        "bytes": sum(record["bytes"] for record in records.values()),
    }


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", text)


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def format_target(person: Mapping[str, str], field: str) -> str:
    if field == "ssn":
        return f"SSN: {person[field]}"
    if field == "email":
        return f"Email: {person[field]}"
    raise TargetManifestError(f"E3b does not define a target format for {field!r}")


def _corpus_views(corpus: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    try:
        texts = [str(row["text"]) for row in corpus]
    except (KeyError, TypeError) as exc:
        raise TargetManifestError("Corpus must be a JSON list of objects with a text field") from exc
    # Preserve a non-digit/non-whitespace separator between documents.  Removing
    # all separators before searching can manufacture an SSN across two records.
    return (
        "\n".join(normalize_text(text) for text in texts),
        "\n".join(digits_only(text) for text in texts),
    )


def field_is_exposed(person: Mapping[str, str], field: str,
                     normalized_corpus: str, digit_corpus: str) -> bool:
    if field == "ssn":
        value = digits_only(person.get(field, ""))
        return len(value) >= 4 and value in digit_corpus
    value = normalize_text(person.get(field, ""))
    return bool(value) and value in normalized_corpus


def even_subset(items: Sequence[Any], n: int) -> List[Any]:
    """The study's deterministic, registry-order-preserving even subsample."""
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[0]]
    picks = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    if len(picks) != n:
        raise TargetManifestError(f"Even subsample produced {len(picks)} rather than {n} indices")
    return [items[index] for index in picks]


def _entry_id(entry: Mapping[str, Any]) -> str:
    try:
        return str(entry["person"]["name"])
    except (KeyError, TypeError) as exc:
        raise TargetManifestError("Every registry entry needs person.name") from exc


def _registry_index(registry: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    index: Dict[str, Mapping[str, Any]] = {}
    for entry in registry:
        person_id = _entry_id(entry)
        if person_id in index:
            raise TargetManifestError(f"Duplicate person id in registry: {person_id!r}")
        index[person_id] = entry
    return index


def select_exposed_trained(
    registry: Sequence[Mapping[str, Any]],
    normalized_corpus: str,
    digit_corpus: str,
    fields: Sequence[str],
    tier_quotas: Mapping[int, int],
) -> Tuple[List[Mapping[str, Any]], Dict[int, int]]:
    """Select the complete D set without consulting any attack outcome."""
    eligible = [
        entry for entry in registry
        if not entry.get("is_negative_control")
        and all(field_is_exposed(entry["person"], field,
                                 normalized_corpus, digit_corpus)
                for field in fields)
    ]
    available = Counter(int(entry["frequency"]) for entry in eligible)
    selected: List[Mapping[str, Any]] = []
    for frequency, quota in sorted((int(k), int(v)) for k, v in tier_quotas.items()):
        tier = [entry for entry in eligible if int(entry["frequency"]) == frequency]
        if len(tier) < quota:
            raise TargetManifestError(
                f"Frequency {frequency} has {len(tier)} fully exposed people; needs {quota}"
            )
        selected.extend(even_subset(tier, quota))
    if len({_entry_id(entry) for entry in selected}) != sum(tier_quotas.values()):
        raise TargetManifestError("D selection is not one-to-one")
    return selected, dict(sorted(available.items()))


FeatureProvider = Callable[[Mapping[str, str], str], Mapping[str, float]]


def _feature_vector(person: Mapping[str, str], fields: Sequence[str],
                    provider: FeatureProvider) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    by_field: Dict[str, Dict[str, float]] = {}
    values: List[float] = []
    for field in fields:
        raw = dict(provider(person, field))
        missing = set(FEATURE_NAMES) - set(raw)
        if missing:
            raise TargetManifestError(f"Feature provider omitted {sorted(missing)} for {field}")
        features = {name: float(raw[name]) for name in FEATURE_NAMES}
        if not all(math.isfinite(value) for value in features.values()):
            raise TargetManifestError(f"Non-finite matching feature for {person.get('name')} {field}")
        by_field[field] = features
        values.extend(features[name] for name in FEATURE_NAMES)
    return np.asarray(values, dtype=float), by_field


def _smd(a: np.ndarray, b: np.ndarray) -> List[Optional[float]]:
    pooled = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2.0)
    diff = a.mean(axis=0) - b.mean(axis=0)
    out: List[Optional[float]] = []
    for numerator, denominator in zip(diff, pooled):
        if denominator == 0:
            out.append(0.0 if numerator == 0 else None)
        else:
            out.append(float(numerator / denominator))
    return out


def _paired_standardized_difference(a: np.ndarray, b: np.ndarray) -> List[Optional[float]]:
    differences = a - b
    means = differences.mean(axis=0)
    scales = differences.std(axis=0, ddof=1)
    return [
        0.0 if scale == 0 and mean == 0 else (None if scale == 0 else float(mean / scale))
        for mean, scale in zip(means, scales)
    ]


def _balance_records(
    d_features: Sequence[Mapping[str, Mapping[str, float]]],
    c_features: Sequence[Mapping[str, Mapping[str, float]]],
    fields: Sequence[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for field in fields:
        d = np.asarray([[row[field][name] for name in FEATURE_NAMES] for row in d_features])
        c = np.asarray([[row[field][name] for name in FEATURE_NAMES] for row in c_features])
        marginal = _smd(d, c)
        paired = _paired_standardized_difference(d, c)
        result[field] = {
            "feature_names": list(FEATURE_NAMES),
            "marginal_smd": marginal,
            "paired_standardized_mean_difference": paired,
            "max_abs_marginal_smd": max(
                (abs(value) for value in marginal if value is not None), default=None
            ),
        }
    return result


def enforce_balance_gate(
    balance: Mapping[str, Mapping[str, Any]],
    fields: Sequence[str],
    threshold: float = DEFAULT_BALANCE_THRESHOLD,
) -> None:
    """Require all six joint-match marginal SMDs to satisfy the design gate."""
    failures = []
    for field in fields:
        record = balance.get(field, {})
        names = record.get("feature_names", [])
        values = record.get("marginal_smd", [])
        if list(names) != list(FEATURE_NAMES) or len(values) != len(FEATURE_NAMES):
            failures.append(f"{field}: missing marginal SMD vector")
            continue
        for name, value in zip(names, values):
            if value is None or not math.isfinite(float(value)):
                failures.append(f"{field}.{name}=undefined")
            elif abs(float(value)) > threshold:
                failures.append(f"{field}.{name}={float(value):+.6f}")
    if failures:
        raise TargetManifestError(
            f"Marginal balance gate failed (requires every |SMD| <= {threshold:.3f}): "
            + ", ".join(failures)
        )


def _target_record(entry: Mapping[str, Any], fields: Sequence[str],
                   features: Mapping[str, Mapping[str, float]],
                   normalized_corpus: str, digit_corpus: str) -> Dict[str, Any]:
    person = entry["person"]
    targets = {}
    for field in fields:
        value = str(person[field])
        targets[field] = {
            "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "formatted_target_sha256": hashlib.sha256(
                format_target(person, field).encode("utf-8")
            ).hexdigest(),
            "exposed_in_finetuning_text": field_is_exposed(
                person, field, normalized_corpus, digit_corpus
            ),
            "features": dict(features[field]),
        }
    return {
        "person_id": _entry_id(entry),
        "frequency": int(entry["frequency"]),
        "targets": targets,
    }


def _git_state(repo_root: os.PathLike[str] | str) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.run(
                ("git", *args), cwd=str(repo_root), check=True,
                capture_output=True, text=True, timeout=120,
            ).stdout.strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines()[:20] if status else [],
    }


def enforce_default_contract(manifest: Mapping[str, Any]) -> None:
    """Reject a self-consistent manifest that is not the registered E3b design."""
    failures = []
    selection = manifest.get("selection", {})
    contract = manifest.get("execution_contract", {})
    expected_quotas = {str(key): value for key, value in DEFAULT_TIER_QUOTAS.items()}
    checks = (
        ("target_set_id", manifest.get("target_set_id"), DEFAULT_TARGET_SET_ID),
        ("fields", manifest.get("fields"), list(DEFAULT_FIELDS)),
        ("tier_quotas", selection.get("tier_quotas"), expected_quotas),
        ("model_name", contract.get("model_name"), DEFAULT_MODEL_NAME),
        ("model_state", contract.get("model_state"), "finetuned"),
        ("contract.fields", contract.get("fields"), list(DEFAULT_FIELDS)),
        ("k_grid", contract.get("k_grid"), list(DEFAULT_K_GRID)),
        ("seeds", contract.get("seeds"), list(DEFAULT_SEEDS)),
        ("gcg_iters", contract.get("gcg_iters"), DEFAULT_GCG_ITERS),
        ("n_people_per_arm", contract.get("n_people_per_arm"), sum(DEFAULT_TIER_QUOTAS.values())),
        (
            "n_targets_per_arm",
            contract.get("n_targets_per_arm"),
            sum(DEFAULT_TIER_QUOTAS.values()) * len(DEFAULT_FIELDS),
        ),
    )
    for name, actual, expected in checks:
        if actual != expected:
            failures.append(f"{name}={actual!r} (required {expected!r})")
    if manifest.get("created_before_e3b_outcomes") is not True:
        failures.append("created_before_e3b_outcomes must be true")
    code = manifest.get("code", {})
    if not isinstance(code.get("commit"), str) or not code["commit"].strip():
        failures.append("code.commit must be non-empty")
    if code.get("dirty") is not False:
        failures.append("code.dirty must be false")
    if failures:
        raise TargetManifestError("Frozen E3b contract violation: " + "; ".join(failures))


def build_manifest(
    *,
    registry_path: os.PathLike[str] | str,
    corpus_path: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    feature_provider: FeatureProvider,
    reference_metadata: Mapping[str, Any],
    fields: Sequence[str] = DEFAULT_FIELDS,
    tier_quotas: Mapping[int, int] = DEFAULT_TIER_QUOTAS,
    target_set_id: str = DEFAULT_TARGET_SET_ID,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    gcg_iters: int = DEFAULT_GCG_ITERS,
    model_name: str = DEFAULT_MODEL_NAME,
    repo_root: os.PathLike[str] | str = ".",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    registry_path, corpus_path = Path(registry_path), Path(corpus_path)
    registry = json.loads(registry_path.read_text())
    corpus = json.loads(corpus_path.read_text())
    if not isinstance(registry, list) or not isinstance(corpus, list):
        raise TargetManifestError("Registry and corpus must both be JSON lists")
    _registry_index(registry)
    fields = tuple(fields)
    if fields != DEFAULT_FIELDS:
        raise TargetManifestError(f"E3b fields are fixed to {DEFAULT_FIELDS}, got {fields}")
    proposed_contract = {
        "target_set_id": target_set_id,
        "fields": list(fields),
        "selection": {
            "tier_quotas": {str(k): int(v) for k, v in sorted(tier_quotas.items())},
        },
        "execution_contract": {
            "model_name": model_name,
            "model_state": "finetuned",
            "k_grid": [int(k) for k in k_grid],
            "seeds": [int(seed) for seed in seeds],
            "gcg_iters": int(gcg_iters),
            "fields": list(fields),
            "n_people_per_arm": sum(int(value) for value in tier_quotas.values()),
            "n_targets_per_arm": sum(int(value) for value in tier_quotas.values()) * len(fields),
        },
        "created_before_e3b_outcomes": True,
        "code": _git_state(repo_root),
    }
    enforce_default_contract(proposed_contract)

    normalized_corpus, digit_corpus = _corpus_views(corpus)
    trained, eligible_counts = select_exposed_trained(
        registry, normalized_corpus, digit_corpus, fields, tier_quotas
    )
    controls = [entry for entry in registry if entry.get("is_negative_control")]
    controls = [
        entry for entry in controls
        if all(not field_is_exposed(entry["person"], field,
                                    normalized_corpus, digit_corpus)
               for field in fields)
    ]
    if len(controls) < len(trained):
        raise TargetManifestError(
            f"Only {len(controls)} corpus-absent controls for {len(trained)} trained people"
        )

    d_vectors, d_by_field = zip(*[
        _feature_vector(entry["person"], fields, feature_provider) for entry in trained
    ])
    c_vectors, c_by_field = zip(*[
        _feature_vector(entry["person"], fields, feature_provider) for entry in controls
    ])
    d_matrix, c_matrix = np.vstack(d_vectors), np.vstack(c_vectors)
    pooled = np.vstack((d_matrix, c_matrix))
    scale = pooled.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    costs = (((d_matrix[:, None, :] - c_matrix[None, :, :]) / scale) ** 2).sum(axis=2)
    # Stable tie-break over registry order; far below any reportable precision.
    costs = costs + np.arange(len(controls), dtype=float)[None, :] * 1e-12
    row_indices, column_indices = linear_sum_assignment(costs)
    if list(row_indices) != list(range(len(trained))):
        raise TargetManifestError("Hungarian assignment did not cover every D person")
    matched_controls = [controls[index] for index in column_indices]
    matched_c_features = [c_by_field[index] for index in column_indices]

    balance = _balance_records(d_by_field, matched_c_features, fields)
    enforce_balance_gate(balance, fields)

    pairs = []
    for pair_index, (d_entry, c_entry, d_features, c_features, cost) in enumerate(zip(
        trained, matched_controls, d_by_field, matched_c_features,
        costs[row_indices, column_indices],
    )):
        d_record = _target_record(
            d_entry, fields, d_features, normalized_corpus, digit_corpus
        )
        c_record = _target_record(
            c_entry, fields, c_features, normalized_corpus, digit_corpus
        )
        pairs.append({
            "pair_index": pair_index,
            "trained": d_record,
            "control": c_record,
            "standardized_squared_distance": float(cost),
        })

    identity_rows = []
    for pair in pairs:
        for arm in ("trained", "control"):
            record = pair[arm]
            for field in fields:
                identity_rows.append((
                    arm, record["person_id"], field,
                    record["targets"][field]["value_sha256"],
                ))
    source = {
        "registry": file_record(registry_path),
        "corpus": file_record(corpus_path),
        "checkpoint": checkpoint_fingerprint(checkpoint_path),
    }
    selected_tiers = Counter(int(entry["frequency"]) for entry in trained)
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "study_id": "capacity_axis_20260902",
        "target_set_id": target_set_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "created_before_e3b_outcomes": True,
        "fields": list(fields),
        "selection": {
            "outcome_columns_read": [],
            "exposure_unit": "(person_id, field)",
            "eligibility": "all selected field values occur in fine-tuning text",
            "method": "even_subset within each eligible frequency tier",
            "tier_quotas": {str(k): int(v) for k, v in sorted(tier_quotas.items())},
            "eligible_counts": {str(k): int(v) for k, v in eligible_counts.items()},
            "selected_tiers": {str(k): int(v) for k, v in sorted(selected_tiers.items())},
        },
        "matching": {
            "unit": "person",
            "method": "Hungarian minimum-cost assignment without replacement",
            "fields_jointly_matched": list(fields),
            "features_per_field": list(FEATURE_NAMES),
            "standardization": "population SD over selected D plus all corpus-absent C",
            "n_control_candidates": len(controls),
            "with_replacement": False,
            "balance": balance,
            "balance_gate": {
                "metric": "absolute marginal standardized mean difference",
                "threshold": DEFAULT_BALANCE_THRESHOLD,
                "n_required": len(fields) * len(FEATURE_NAMES),
                "passed": True,
            },
        },
        "reference_model": dict(reference_metadata),
        "execution_contract": {
            "model_name": model_name,
            "model_state": "finetuned",
            "k_grid": [int(k) for k in k_grid],
            "seeds": [int(seed) for seed in seeds],
            "gcg_iters": int(gcg_iters),
            "fields": list(fields),
            "n_people_per_arm": len(trained),
            "n_targets_per_arm": len(trained) * len(fields),
        },
        "source": source,
        "code": proposed_contract["code"],
        "pairs": pairs,
    }
    payload["integrity"] = {
        "target_values_sha256": canonical_sha256(sorted(identity_rows)),
        "pair_assignment_sha256": canonical_sha256([
            (pair["trained"]["person_id"], pair["control"]["person_id"])
            for pair in pairs
        ]),
    }
    payload["integrity"]["payload_sha256"] = canonical_sha256({
        key: value for key, value in payload.items() if key != "integrity"
    } | {"integrity": {
        key: value for key, value in payload["integrity"].items()
        if key != "payload_sha256"
    }})
    return payload


def write_manifest(manifest: Mapping[str, Any], path: os.PathLike[str] | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() == encoded:
            return sha256_file(path)
        raise TargetManifestError(
            f"Refusing to overwrite frozen target manifest with different bytes: {path}"
        )
    path.write_text(encoded)
    return sha256_file(path)


def _payload_digest(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    integrity = dict(body.pop("integrity", {}))
    integrity.pop("payload_sha256", None)
    body["integrity"] = integrity
    return canonical_sha256(body)


def validate_manifest(
    manifest_path: os.PathLike[str] | str,
    *,
    registry_path: os.PathLike[str] | str,
    corpus_path: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    expected_fields: Sequence[str] = DEFAULT_FIELDS,
    expected_people_per_arm: int = 25,
) -> Dict[str, Any]:
    """Revalidate all frozen inputs and return registry entries for execution."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise TargetManifestError(f"Unexpected target manifest schema: {manifest.get('schema')}")
    expected_payload = manifest.get("integrity", {}).get("payload_sha256")
    if expected_payload != _payload_digest(manifest):
        raise TargetManifestError("Frozen target manifest payload hash mismatch")
    enforce_default_contract(manifest)
    if tuple(expected_fields) != DEFAULT_FIELDS:
        raise TargetManifestError(f"Validator fields are fixed to {DEFAULT_FIELDS}")
    if expected_people_per_arm != sum(DEFAULT_TIER_QUOTAS.values()):
        raise TargetManifestError("Validator arm size is fixed to 25 people")
    if list(manifest.get("fields", [])) != list(expected_fields):
        raise TargetManifestError(
            f"Manifest fields {manifest.get('fields')} != required {list(expected_fields)}"
        )

    registry_path, corpus_path = Path(registry_path), Path(corpus_path)
    current_source = {
        "registry": file_record(registry_path),
        "corpus": file_record(corpus_path),
        "checkpoint": checkpoint_fingerprint(checkpoint_path),
    }
    if current_source != manifest.get("source"):
        raise TargetManifestError("Registry, corpus, or checkpoint differs from frozen manifest")
    registry = json.loads(registry_path.read_text())
    corpus = json.loads(corpus_path.read_text())
    index = _registry_index(registry)
    normalized_corpus, digit_corpus = _corpus_views(corpus)

    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != expected_people_per_arm:
        raise TargetManifestError(
            f"Expected {expected_people_per_arm} frozen pairs, found "
            f"{len(pairs) if isinstance(pairs, list) else 'invalid'}"
        )
    d_entries, c_entries, identities = [], [], []
    for expected_index, pair in enumerate(pairs):
        if pair.get("pair_index") != expected_index:
            raise TargetManifestError("Pair indices are not canonical and contiguous")
        for arm, expected_control in (("trained", False), ("control", True)):
            frozen = pair[arm]
            person_id = str(frozen["person_id"])
            if person_id not in index:
                raise TargetManifestError(f"Frozen person missing from registry: {person_id!r}")
            entry = index[person_id]
            if bool(entry.get("is_negative_control")) != expected_control:
                raise TargetManifestError(f"Frozen {arm} membership changed for {person_id!r}")
            if int(entry["frequency"]) != int(frozen["frequency"]):
                raise TargetManifestError(f"Frozen frequency changed for {person_id!r}")
            for field in expected_fields:
                value = str(entry["person"].get(field, ""))
                target = frozen["targets"][field]
                if hashlib.sha256(value.encode()).hexdigest() != target["value_sha256"]:
                    raise TargetManifestError(f"Frozen {arm} {field} value changed for {person_id!r}")
                formatted = format_target(entry["person"], field)
                if hashlib.sha256(formatted.encode()).hexdigest() != target["formatted_target_sha256"]:
                    raise TargetManifestError(
                        f"Frozen formatted {arm} {field} target changed for {person_id!r}"
                    )
                present = field_is_exposed(
                    entry["person"], field, normalized_corpus, digit_corpus
                )
                if present != bool(target["exposed_in_finetuning_text"]):
                    raise TargetManifestError(f"Exposure status changed for {person_id!r} {field}")
                if arm == "trained" and not present:
                    raise TargetManifestError(f"D field is not exposed: {person_id!r} {field}")
                if arm == "control" and present:
                    raise TargetManifestError(f"C field occurs in fine-tuning text: {person_id!r} {field}")
                identities.append((arm, person_id, field, target["value_sha256"]))
            (c_entries if expected_control else d_entries).append(entry)

    if len({_entry_id(entry) for entry in d_entries}) != expected_people_per_arm:
        raise TargetManifestError("D people are not unique")
    if len({_entry_id(entry) for entry in c_entries}) != expected_people_per_arm:
        raise TargetManifestError("C people are not unique; matching must be without replacement")
    tiers = Counter(int(entry["frequency"]) for entry in d_entries)
    expected_tiers = {
        int(key): int(value)
        for key, value in manifest["selection"]["tier_quotas"].items()
    }
    if dict(tiers) != expected_tiers:
        raise TargetManifestError(f"D tier composition changed: {dict(tiers)} != {expected_tiers}")
    if canonical_sha256(sorted(identities)) != manifest["integrity"]["target_values_sha256"]:
        raise TargetManifestError("Frozen target-value hash mismatch")
    pair_rows = [
        (pair["trained"]["person_id"], pair["control"]["person_id"])
        for pair in pairs
    ]
    if canonical_sha256(pair_rows) != manifest["integrity"]["pair_assignment_sha256"]:
        raise TargetManifestError("Frozen pair-assignment hash mismatch")

    # Recompute the six diagnostics from the frozen pair-level features so the
    # declared balance table cannot drift from the targets that E3 will run.
    d_features = [
        {field: pair["trained"]["targets"][field]["features"] for field in expected_fields}
        for pair in pairs
    ]
    c_features = [
        {field: pair["control"]["targets"][field]["features"] for field in expected_fields}
        for pair in pairs
    ]
    recomputed_balance = _balance_records(d_features, c_features, expected_fields)
    if canonical_json(recomputed_balance) != canonical_json(manifest["matching"].get("balance")):
        raise TargetManifestError("Frozen balance diagnostics do not match pair features")
    gate = manifest["matching"].get("balance_gate", {})
    if float(gate.get("threshold", -1)) != DEFAULT_BALANCE_THRESHOLD:
        raise TargetManifestError("Frozen balance threshold differs from the preregistered 0.1 gate")
    enforce_balance_gate(recomputed_balance, expected_fields, DEFAULT_BALANCE_THRESHOLD)

    contract = manifest.get("execution_contract", {})
    if int(contract.get("n_people_per_arm", -1)) != expected_people_per_arm:
        raise TargetManifestError("Frozen execution contract has the wrong arm size")
    if int(contract.get("n_targets_per_arm", -1)) != expected_people_per_arm * len(expected_fields):
        raise TargetManifestError("Frozen execution contract has the wrong target count")

    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "trained_entries": d_entries,
        "control_entries": c_entries,
        "source": current_source,
        "target_values_sha256": manifest["integrity"]["target_values_sha256"],
        "pair_assignment_sha256": manifest["integrity"]["pair_assignment_sha256"],
    }


class TransformersFeatureProvider:
    """Feature provider identical to E17's target-level covariates."""

    def __init__(self, model_name: str, device: str = "auto", cache_dir: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from attempt_log import target_self_information

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._target_self_information = target_self_information
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=dtype
        ).to(device).eval()
        self.metadata = {
            "name": model_name,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_commit": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_commit": getattr(self.tokenizer, "_commit_hash", None),
            "vocab_size": int(self.tokenizer.vocab_size),
            "device_used_for_freeze": device,
        }
        try:
            import transformers
            self.metadata["transformers_version"] = transformers.__version__
        except Exception:
            self.metadata["transformers_version"] = None

    def __call__(self, person: Mapping[str, str], field: str) -> Mapping[str, float]:
        text = format_target(person, field)
        H_bits, tok_len = self._target_self_information(
            text, self.model, self.tokenizer, self.device
        )
        return {"char_len": len(text), "tok_len": tok_len, "H_bits": H_bits}


def _parse_int_list(value: str) -> List[int]:
    return [int(item) for item in re.split(r"[,\s]+", value.strip()) if item]


def _parse_quotas(value: str) -> Dict[int, int]:
    quotas = {}
    for item in value.split(","):
        key, count = item.split(":", 1)
        quotas[int(key)] = int(count)
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build and freeze an E3b target manifest")
    verify = subparsers.add_parser("verify", help="verify a frozen manifest against current bytes")
    for command in (build, verify):
        command.add_argument("--registry", default="data/target_registry.json")
        command.add_argument("--corpus", default="data/corpus/train.json")
        command.add_argument("--checkpoint", default="models/gpt2")
    build.add_argument("--output", default="results/target_sets/e3b.json")
    build.add_argument("--target-set-id", default="e3b")
    build.add_argument("--reference-model", default="gpt2")
    build.add_argument("--cache-dir", default=None)
    build.add_argument("--device", default="auto")
    build.add_argument("--tier-quotas", default="1:3,5:7,20:15")
    build.add_argument("--k-grid", default="0 1 2 3 4 6 8 12 16 20 24 32 48 64")
    build.add_argument("--seeds", default="42 1337 2024")
    build.add_argument("--gcg-iters", type=int, default=200)
    build.add_argument("--model", default="gpt2")
    build.add_argument("--repo-root", default=".")
    build.add_argument("--results-dir", default="results")
    verify.add_argument("--manifest", default="results/target_sets/e3b.json")
    args = parser.parse_args()

    if args.command == "build":
        existing = sorted(
            list(Path(args.results_dir).glob(f"attempts/{args.target_set_id}__E3__*.parquet"))
            + list(Path(args.results_dir).glob(f"manifests/{args.target_set_id}__E3__*.json"))
        )
        if existing:
            raise TargetManifestError(
                "Refusing to choose E3b targets after outcome/provenance shards exist: "
                + ", ".join(str(path) for path in existing[:5])
            )
        provider = TransformersFeatureProvider(
            args.reference_model, device=args.device, cache_dir=args.cache_dir
        )
        manifest = build_manifest(
            registry_path=args.registry,
            corpus_path=args.corpus,
            checkpoint_path=args.checkpoint,
            feature_provider=provider,
            reference_metadata=provider.metadata,
            tier_quotas=_parse_quotas(args.tier_quotas),
            target_set_id=args.target_set_id,
            k_grid=_parse_int_list(args.k_grid),
            seeds=_parse_int_list(args.seeds),
            gcg_iters=args.gcg_iters,
            model_name=args.model,
            repo_root=args.repo_root,
        )
        digest = write_manifest(manifest, args.output)
        report = validate_manifest(
            args.output, registry_path=args.registry, corpus_path=args.corpus,
            checkpoint_path=args.checkpoint,
        )
        print(json.dumps({
            "manifest": args.output,
            "sha256": digest,
            "target_values_sha256": report["target_values_sha256"],
            "pairs": len(report["trained_entries"]),
            "tier_composition": manifest["selection"]["selected_tiers"],
            "balance": manifest["matching"]["balance"],
        }, indent=2, ensure_ascii=False))
    else:
        report = validate_manifest(
            args.manifest, registry_path=args.registry, corpus_path=args.corpus,
            checkpoint_path=args.checkpoint,
        )
        print(json.dumps({
            "ok": True,
            "manifest_sha256": report["manifest_sha256"],
            "target_values_sha256": report["target_values_sha256"],
            "trained_people": len(report["trained_entries"]),
            "control_people": len(report["control_entries"]),
        }, indent=2))


if __name__ == "__main__":
    main()
