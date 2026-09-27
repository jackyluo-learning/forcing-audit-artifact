import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import e3_target_manifest as target_manifest_lib
import run_manifest
from analyze_e3b_repair import (
    AnalysisGateError,
    _load_target_contract,
    analyze_control_curve,
    run_analysis,
    validate_and_load,
)


K_GRID = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64]
SEEDS = [42, 1337, 2024]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _person_values(prefix, index):
    return {
        "ssn": f"{100 + index:03d}-22-{3000 + index:04d}",
        "email": f"{prefix}{index:02d}@example.invalid",
    }


def _frequency(index):
    if index < 3:
        return 1
    if index < 10:
        return 5
    return 20


def _target_record(prefix, index, arm):
    person_id = f"{prefix} Person {index:02d}"
    values = _person_values(prefix.lower(), index + (0 if arm == "trained" else 100))
    targets = {}
    for field, value in values.items():
        formatted = f"{'SSN' if field == 'ssn' else 'Email'}: {value}"
        features = {
            "char_len": float(12 + index % 5),
            "tok_len": float(5 + index % 4),
            "H_bits": float(50 + index),
        }
        targets[field] = {
            "value": value,
            "value_sha256": _sha(value),
            "formatted": formatted,
            "formatted_target_sha256": _sha(formatted),
            "exposed_in_finetuning_text": arm == "trained",
            "features": features,
        }
    return {
        "person_id": person_id,
        "frequency": _frequency(index) if arm == "trained" else 0,
        "targets": targets,
    }


def _write_target_manifest(path):
    pairs = []
    for index in range(25):
        d_record = _target_record("D", index, "trained")
        c_record = _target_record("C", index, "control")
        # Raw values are fixture-only conveniences and are not part of the real
        # frozen schema. Remove them after saving them for attempt construction.
        pairs.append({
            "pair_index": index,
            "trained": d_record,
            "control": c_record,
            "standardized_squared_distance": 0.0,
        })

    fixture_values = {}
    for pair in pairs:
        for arm in ("trained", "control"):
            record = pair[arm]
            for field, target in record["targets"].items():
                fixture_values[(arm, record["person_id"], field)] = {
                    "value": target.pop("value"),
                    "formatted": target.pop("formatted"),
                    "frequency": record["frequency"],
                }

    fields = ["ssn", "email"]
    d_features = [
        {field: pair["trained"]["targets"][field]["features"] for field in fields}
        for pair in pairs
    ]
    c_features = [
        {field: pair["control"]["targets"][field]["features"] for field in fields}
        for pair in pairs
    ]
    balance = target_manifest_lib._balance_records(d_features, c_features, fields)
    identities = [
        (arm, pair[arm]["person_id"], field, pair[arm]["targets"][field]["value_sha256"])
        for pair in pairs for arm in ("trained", "control") for field in fields
    ]
    manifest = {
        "schema": target_manifest_lib.SCHEMA,
        "study_id": "capacity_axis_20260902",
        "target_set_id": "e3b",
        "created_at": "2026-09-20T12:00:00+00:00",
        "created_before_e3b_outcomes": True,
        "fields": fields,
        "selection": {
            "outcome_columns_read": [],
            "exposure_unit": "(person_id, field)",
            "eligibility": "all selected field values occur in fine-tuning text",
            "method": "even_subset within each eligible frequency tier",
            "tier_quotas": {"1": 3, "5": 7, "20": 15},
            "eligible_counts": {"1": 3, "5": 7, "20": 15},
            "selected_tiers": {"1": 3, "5": 7, "20": 15},
        },
        "matching": {
            "unit": "person",
            "method": "Hungarian minimum-cost assignment without replacement",
            "fields_jointly_matched": fields,
            "features_per_field": ["char_len", "tok_len", "H_bits"],
            "standardization": "population SD over selected D plus all corpus-absent C",
            "n_control_candidates": 25,
            "with_replacement": False,
            "balance": balance,
            "balance_gate": {
                "metric": "absolute marginal standardized mean difference",
                "threshold": 0.1,
                "n_required": 6,
                "passed": True,
            },
        },
        "reference_model": {"name": "synthetic"},
        "execution_contract": {
            "model_name": "gpt2",
            "model_state": "finetuned",
            "k_grid": K_GRID,
            "seeds": SEEDS,
            "gcg_iters": 200,
            "fields": fields,
            "n_people_per_arm": 25,
            "n_targets_per_arm": 50,
        },
        "source": {
            "registry": {"sha256": "1" * 64, "bytes": 1},
            "corpus": {"sha256": "2" * 64, "bytes": 2},
            "checkpoint": {"sha256": "3" * 64, "bytes": 3, "files": {}},
        },
        "code": {"commit": "builder", "dirty": False, "dirty_files": []},
        "pairs": pairs,
        "integrity": {
            "target_values_sha256": target_manifest_lib.canonical_sha256(sorted(identities)),
            "pair_assignment_sha256": target_manifest_lib.canonical_sha256([
                (pair["trained"]["person_id"], pair["control"]["person_id"])
                for pair in pairs
            ]),
        },
    }
    manifest["integrity"]["payload_sha256"] = target_manifest_lib._payload_digest(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, fixture_values


def _resolved_config(target_sha):
    return {
        "target_set_id": "e3b",
        "target_manifest_sha256": target_sha,
        "model_name": "gpt2",
        "model_state": "finetuned",
        "fields": ["ssn", "email"],
        "k_grid": K_GRID,
        "seeds": SEEDS,
        "gcg_iters": 200,
        "candidates_per_position_B": 256,
        "candidate_evaluations_per_step": 512,
        "candidate_minibatch": 64,
        "early_stop_on_exact_match": True,
        "extraction_check_interval": 10,
        "decision_rule": "field-normalized substring exact_match",
    }


def _make_attempt_rows(seed, k, values):
    base_hits = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        6: 4,
        8: 6,
        12: 10,
        16: 14,
        20: 18,
        24: 20,
        32: 22,
        48: 23,
        64: 24,
    }
    seed_offset = {42: -1, 1337: 0, 2024: 1}
    rows = []
    for (arm, person_id, field), record in values.items():
        person_index = int(person_id.rsplit(" ", 1)[-1])
        # A deterministic rising C curve. D outcomes are irrelevant to this analyzer.
        n_hit_people = max(0, min(25, base_hits.get(k, 0) + seed_offset[seed]))
        success = bool(arm == "control" and k > 0 and person_index < n_hit_people)
        rows.append({
            "run_id": "e3b",
            "exp_id": "E3",
            "seed": seed,
            "model_name": "gpt2",
            "model_state": "finetuned",
            "target_membership": arm,
            "person_id": person_id,
            "field": field,
            "train_frequency": record["frequency"],
            "probe": "fixed" if k == 0 else "gcg_free",
            "capacity_k": k,
            "softprompt_norm": None,
            "lambda_fluency": None,
            "target_string": record["value"] if k == 0 else record["formatted"],
            "target_H_bits": 50.0 + person_index,
            "target_len_tokens": 8,
            "prompt_text": "synthetic prompt",
            "prompt_token_ids": [1, 2, 3],
            "forward_passes": 1,
            "steps_run": 0 if k == 0 else 200,
            "steps_to_first_success": None,
            "final_target_nll": 0.0,
            "generation": "",
            "gen_len_tokens": 0,
            "exact_match": success,
            "random_record_match": bool(
                (
                    arm == "control" and k == 64 and seed == 42
                    and person_index == 0 and field == "ssn"
                )
                or (
                    arm == "trained" and k == 32 and seed == 1337
                    and person_index == 1 and field == "email"
                )
            ),
            "wallclock_s": 0.1,
        })
    return pd.DataFrame(rows)


def _write_complete_fixture(root, accelerator_name="NVIDIA A100-SXM4-80GB"):
    attempts_dir = root / "attempts"
    manifests_dir = root / "manifests"
    attempts_dir.mkdir()
    manifests_dir.mkdir()
    target_path = root / "e3b.json"
    target, values = _write_target_manifest(target_path)
    target_sha = target_manifest_lib.sha256_file(target_path)
    target_anchor = {
        "manifest_sha256": target_sha,
        "payload_sha256": target["integrity"]["payload_sha256"],
        "target_values_sha256": target["integrity"]["target_values_sha256"],
        "pair_assignment_sha256": target["integrity"]["pair_assignment_sha256"],
    }
    resolved = _resolved_config(target_sha)
    subset = run_manifest.target_subset_hash(list(values))
    for seed in SEEDS:
        for task_index, k in enumerate(K_GRID):
            stem = f"e3b__E3__gpt2_{seed}_field-ssn-email_k{k}"
            manifest = {
                "schema": "run-manifest-v1",
                "study_id": "capacity_axis_20260902",
                "run_id": "e3b",
                "exp_id": "E3",
                "shard": {
                    "model_name": "gpt2",
                    "model_state": "finetuned",
                    "seed": seed,
                    "capacity_k": k,
                    "fields": ["ssn", "email"],
                },
                "gcg_iters": 200,
                "target_subset_hash": subset,
                "n_targets": 100,
                "arm_sizes": {"D": 25, "C": 25},
                "tier_composition": {"1": 3, "5": 7, "20": 15},
                "code": {"commit": COMMIT, "dirty": False, "dirty_files": []},
                "env": {"pip_freeze_sha256_16": "e" * 16},
                "accelerator": {
                    "device": "cuda",
                    "name": accelerator_name,
                    "count": 1,
                    "total_mem_gb": 80.0,
                },
                "config": {
                    "PII_RUN_ID": "e3b",
                    "PII_E3_TARGET_MANIFEST": str(target_path),
                    "PII_MODELS": "gpt2",
                    "PII_SEEDS": "42 1337 2024",
                    "PII_KGRID": "0 1 2 3 4 6 8 12 16 20 24 32 48 64",
                    "PII_CAP_SWEEP_N": "25",
                    "PII_GCG_ITERS": "200",
                    "PII_FIELDS": "ssn,email",
                    "PII_DEVICE_PROFILE": "a100_80",
                    "PII_CAP_K": str(k),
                },
                "frozen_targets": {
                    "path": str(target_path),
                    "manifest_sha256": target_sha,
                    "payload_sha256": target["integrity"]["payload_sha256"],
                    "target_values_sha256": target["integrity"]["target_values_sha256"],
                    "pair_assignment_sha256": target["integrity"]["pair_assignment_sha256"],
                },
                "input_fingerprints": target["source"],
                "resolved_sweep_config": resolved,
                "resolved_sweep_config_sha256": target_manifest_lib.canonical_sha256(resolved),
                "scheduler": {
                    "slurm_job_id": str(1000 + task_index),
                    "slurm_array_job_id": "1000",
                    "slurm_array_task_id": str(task_index),
                },
            }
            (manifests_dir / f"{stem}.json").write_text(json.dumps(manifest))
            _make_attempt_rows(seed, k, values).to_parquet(
                attempts_dir / f"{stem}.parquet", index=False
            )
    # This must be ignored by the exact formal prefix.
    (manifests_dir / "e3b_repro__E3__ignored.json").write_text("{}")
    pd.DataFrame({"ignored": [True]}).to_parquet(
        attempts_dir / "e3b_repro__E3__ignored.parquet", index=False
    )
    return attempts_dir, manifests_dir, target_path, target_anchor


class E3bFocusedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_run_passes_and_writes_only_focused_outputs(self):
        attempts, manifests, target, target_anchor = _write_complete_fixture(self.root)
        output = self.root / "output"
        paths = run_analysis(
            attempts_dir=attempts,
            manifests_dir=manifests,
            target_manifest=target,
            output_dir=output,
            expected_formal_commit=COMMIT,
            n_bootstrap=200,
            bootstrap_seed=7,
            expected_target_anchor=target_anchor,
        )
        self.assertTrue(all(path.is_file() for path in paths.values()))
        payload = json.loads(paths["json"].read_text())
        self.assertEqual(payload["gate"]["formal_manifest_count"], 42)
        self.assertEqual(payload["gate"]["attempt_rows"], 4200)
        self.assertEqual(payload["gate"]["repro_shards_included"], 0)
        result = payload["control_capacity_result"]
        self.assertEqual(len(result["curve"]), 14)
        self.assertTrue(all(row["count"] == 150 for row in result["curve"]))
        self.assertEqual(result["spearman_positive_k"]["k_values"], K_GRID[1:])
        self.assertEqual(result["control_random_record_match_count"], 1)
        k20 = next(row for row in result["curve"] if row["k"] == 20)
        self.assertEqual((k20["hits"], k20["count"]), (108, 150))
        self.assertAlmostEqual(k20["rate"], 0.72)
        seed20 = {
            row["seed"]: row["rate"]
            for row in result["seed_rates"] if row["k"] == 20
        }
        self.assertEqual(seed20, {42: 0.68, 1337: 0.72, 2024: 0.76})
        k0 = next(row for row in result["curve"] if row["k"] == 0)
        self.assertEqual(k0["person_bootstrap_95_ci"], [0.0, 0.0])
        self.assertEqual(k0["ci_method"], "wilson_effective_n_at_boundary")
        self.assertGreater(k0["reported_95_ci"][1], 0.10)
        self.assertNotIn("H2", payload)
        self.assertNotIn("trained", json.dumps(result).lower())

        broken_output = self.root / "broken-output"
        with mock.patch(
            "analyze_e3b_repair.write_figure",
            side_effect=ModuleNotFoundError("synthetic missing matplotlib"),
        ):
            with self.assertRaises(ModuleNotFoundError):
                run_analysis(
                    attempts_dir=attempts,
                    manifests_dir=manifests,
                    target_manifest=target,
                    output_dir=broken_output,
                    expected_formal_commit=COMMIT,
                    n_bootstrap=20,
                    bootstrap_seed=7,
                    expected_target_anchor=target_anchor,
                )
        self.assertFalse(any((broken_output / name).exists() for name in (
            "e3b_repair_analysis.json",
            "e3b_repair_c_curve.csv",
            "e3b_repair_c_curve.png",
            "e3b_repair_c_curve.pdf",
        )))

    def test_missing_formal_shard_blocks_before_statistics(self):
        attempts, manifests, target, target_anchor = _write_complete_fixture(self.root)
        next(manifests.glob("e3b__E3__*.json")).unlink()
        with self.assertRaisesRegex(AnalysisGateError, "expected 42 formal manifests"):
            validate_and_load(
                attempts, manifests, target, COMMIT,
                expected_target_anchor=target_anchor,
            )

    def test_non_a100_and_duplicate_target_rows_are_rejected(self):
        first_root = self.root / "v100"
        first_root.mkdir()
        attempts, manifests, target, target_anchor = _write_complete_fixture(
            first_root, accelerator_name="Tesla V100-SXM2-32GB"
        )
        with self.assertRaisesRegex(AnalysisGateError, "not A100"):
            validate_and_load(
                attempts, manifests, target, COMMIT,
                expected_target_anchor=target_anchor,
            )

        second_root = self.root / "duplicate"
        second_root.mkdir()
        attempts, manifests, target, target_anchor = _write_complete_fixture(second_root)
        path = next(attempts.glob("e3b__E3__*.parquet"))
        frame = pd.read_parquet(path)
        original = frame.copy()
        frame.iloc[1] = frame.iloc[0]
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(AnalysisGateError, "duplicate target keys"):
            validate_and_load(
                attempts, manifests, target, COMMIT,
                expected_target_anchor=target_anchor,
            )
        original.drop(columns=["wallclock_s"]).to_parquet(path, index=False)
        with self.assertRaisesRegex(AnalysisGateError, "attempt schema mismatch"):
            validate_and_load(
                attempts, manifests, target, COMMIT,
                expected_target_anchor=target_anchor,
            )

    def test_cluster_bootstrap_keeps_both_fields_and_three_seeds(self):
        attempts, manifests, target, target_anchor = _write_complete_fixture(self.root)
        frame, gate = validate_and_load(
            attempts, manifests, target, COMMIT,
            expected_target_anchor=target_anchor,
        )
        result, curve = analyze_control_curve(
            frame,
            k_grid=gate["k_grid"],
            seeds=gate["seeds"],
            n_bootstrap=100,
            bootstrap_seed=11,
        )
        self.assertEqual(result["bootstrap"]["n_control_people"], 25)
        self.assertEqual(result["bootstrap"]["replicates"], 100)
        self.assertTrue((curve["c_count"] == 25 * 2 * 3).all())

        control = frame[frame["target_membership"] == "control"]
        matrix = (
            control.pivot_table(
                index="person_id", columns="capacity_k", values="exact_match", aggfunc="mean"
            )
            .reindex(columns=K_GRID)
            .to_numpy(dtype=float)
        )
        draws = np.random.default_rng(11).integers(0, 25, size=(100, 25))
        independent_replicates = matrix[draws].mean(axis=1)
        expected_low, expected_high = np.percentile(
            independent_replicates, [2.5, 97.5], axis=0
        )
        np.testing.assert_allclose(curve["c_rate"], matrix.mean(axis=0))
        np.testing.assert_allclose(curve["c_person_bootstrap_ci_low"], expected_low)
        np.testing.assert_allclose(curve["c_person_bootstrap_ci_high"], expected_high)
        self.assertAlmostEqual(result["spearman_positive_k"]["rho"], 1.0)

        flat = frame.copy()
        flat.loc[flat["target_membership"] == "control", "exact_match"] = False
        flat_result, _ = analyze_control_curve(
            flat,
            k_grid=gate["k_grid"],
            seeds=gate["seeds"],
            n_bootstrap=50,
            bootstrap_seed=11,
        )
        flat_spearman = flat_result["spearman_positive_k"]
        self.assertEqual(flat_spearman["ci_status"], "insufficient_nonconstant_replicates")
        self.assertEqual(flat_spearman["bootstrap_95_ci"], [None, None])

    def test_pre_outcome_anchor_k0_release_and_top_level_budget_are_hard_gates(self):
        attempts, manifests, target, target_anchor = _write_complete_fixture(self.root)

        wrong_anchor = dict(target_anchor)
        wrong_anchor["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(AnalysisGateError, "registered pre-outcome SHA-256"):
            validate_and_load(
                attempts, manifests, target, COMMIT,
                expected_target_anchor=wrong_anchor,
            )

        k0_path = attempts / "e3b__E3__gpt2_42_field-ssn-email_k0.parquet"
        k0 = pd.read_parquet(k0_path)
        control_index = k0.index[k0["target_membership"] == "control"][0]
        k0.loc[control_index, "exact_match"] = True
        k0.to_parquet(k0_path, index=False)
        with self.assertRaisesRegex(AnalysisGateError, "k=0 release gate failed"):
            validate_and_load(
                attempts, manifests, target, COMMIT,
                expected_target_anchor=target_anchor,
            )

        # Restore the outcome, then make all 42 top-level budgets mutually
        # consistent but wrong; run_manifest.compare alone would accept this.
        k0.loc[control_index, "exact_match"] = False
        k0.to_parquet(k0_path, index=False)
        for path in manifests.glob("e3b__E3__*.json"):
            manifest = json.loads(path.read_text())
            manifest["gcg_iters"] = 999
            path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(AnalysisGateError, "top-level gcg_iters"):
            validate_and_load(
                attempts, manifests, target, COMMIT,
                expected_target_anchor=target_anchor,
            )

    def test_target_study_and_outcome_free_selection_are_independently_checked(self):
        target_path = self.root / "target.json"
        manifest, _ = _write_target_manifest(target_path)

        manifest["study_id"] = "post_outcome_replacement"
        manifest["integrity"]["payload_sha256"] = target_manifest_lib._payload_digest(manifest)
        target_path.write_text(json.dumps(manifest, indent=2) + "\n")
        anchor = {
            "manifest_sha256": target_manifest_lib.sha256_file(target_path),
            "payload_sha256": manifest["integrity"]["payload_sha256"],
            "target_values_sha256": manifest["integrity"]["target_values_sha256"],
            "pair_assignment_sha256": manifest["integrity"]["pair_assignment_sha256"],
        }
        with self.assertRaisesRegex(AnalysisGateError, "wrong target study_id"):
            _load_target_contract(target_path, anchor)

        manifest["study_id"] = "capacity_axis_20260902"
        manifest["selection"]["outcome_columns_read"] = ["exact_match"]
        manifest["integrity"]["payload_sha256"] = target_manifest_lib._payload_digest(manifest)
        target_path.write_text(json.dumps(manifest, indent=2) + "\n")
        anchor["manifest_sha256"] = target_manifest_lib.sha256_file(target_path)
        anchor["payload_sha256"] = manifest["integrity"]["payload_sha256"]
        with self.assertRaisesRegex(AnalysisGateError, "read no outcome columns"):
            _load_target_contract(target_path, anchor)


if __name__ == "__main__":
    unittest.main()
