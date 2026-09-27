import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_manifest


class RunManifestComparisonTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _strict_manifest(self, k, seed):
        config = {
            "target_set_id": "e3b",
            "model_name": "gpt2",
            "model_state": "finetuned",
            "fields": ["ssn", "email"],
            "k_grid": [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64],
            "seeds": [42, 1337, 2024],
            "gcg_iters": 200,
        }
        config_hash = hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest()
        return {
            "shard": {"capacity_k": k, "seed": seed},
            "target_subset_hash": "subset",
            "gcg_iters": 200,
            "n_targets": 100,
            "arm_sizes": {"D": 25, "C": 25},
            "tier_composition": {"1": 3, "5": 7, "20": 15},
            "frozen_targets": {
                "manifest_sha256": "manifest",
                "payload_sha256": "payload",
                "target_values_sha256": "values",
                "pair_assignment_sha256": "pairs",
            },
            "input_fingerprints": {
                "registry": {"sha256": "registry"},
                "corpus": {"sha256": "corpus"},
                "checkpoint": {"sha256": "checkpoint"},
            },
            "resolved_sweep_config": config,
            "resolved_sweep_config_sha256": config_hash,
            "code": {"commit": "abc123", "dirty": False},
            "env": {"pip_freeze_sha256_16": "environment"},
            "accelerator": {"name": "NVIDIA A100-SXM4-80GB"},
        }

    def _write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return str(path)

    def _full_paths(self):
        paths = []
        for seed in (42, 1337, 2024):
            for k in (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64):
                paths.append(self._write(f"s{seed}-k{k}.json", self._strict_manifest(k, seed)))
        return paths

    def test_strict_e3b_accepts_complete_unique_14_by_3_coordinates(self):
        paths = self._full_paths()
        result = run_manifest.compare(paths)
        self.assertTrue(result["strict_e3b"])
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["coordinates"]["expected_count"], 42)
        self.assertEqual(result["coordinates"]["unique_observed_count"], 42)

    def test_strict_e3b_rejects_a_single_manifest_as_incomplete(self):
        result = run_manifest.compare([
            self._write("only.json", self._strict_manifest(0, 42))
        ])
        self.assertFalse(result["ok"])
        self.assertTrue(any("exactly 42" in error for error in result["errors"]))
        self.assertEqual(len(result["coordinates"]["missing"]), 41)

    def test_strict_e3b_rejects_duplicate_and_missing_coordinate(self):
        paths = self._full_paths()
        # Replace the final (2024,64) shard with a duplicate (42,0), preserving
        # the count of 42 so completeness cannot be inferred from count alone.
        paths[-1] = self._write("duplicate.json", self._strict_manifest(0, 42))
        result = run_manifest.compare(paths)
        self.assertFalse(result["ok"])
        self.assertTrue(result["coordinates"]["duplicates"])
        self.assertEqual(result["coordinates"]["missing"], [[2024, 64]])

    def test_strict_e3b_rejects_dirty_or_mismatched_provenance(self):
        paths = self._full_paths()
        second = json.loads(Path(paths[1]).read_text())
        second["code"]["dirty"] = True
        second["frozen_targets"]["pair_assignment_sha256"] = "different"
        paths[1] = self._write("bad.json", second)
        result = run_manifest.compare(paths)
        self.assertFalse(result["ok"])
        self.assertTrue(any("code.dirty" in error for error in result["errors"]))
        self.assertTrue(any("pair_assignment_sha256" in error for error in result["errors"]))

    def test_historical_manifest_comparison_remains_compatible(self):
        paths = []
        for index, k in enumerate((1, 2)):
            paths.append(self._write(f"old{index}.json", {
                "shard": {"capacity_k": k, "seed": 42},
                "target_subset_hash": "old-subset",
                "gcg_iters": 200,
            }))
        result = run_manifest.compare(paths)
        self.assertFalse(result["strict_e3b"])
        self.assertTrue(result["ok"])

    def test_empty_comparison_fails_closed(self):
        result = run_manifest.compare([])
        self.assertFalse(result["ok"])
        self.assertEqual(result["n_shards"], 0)


if __name__ == "__main__":
    unittest.main()
