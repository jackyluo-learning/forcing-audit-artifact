import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from e3_target_manifest import (
    TargetManifestError,
    _payload_digest,
    build_manifest,
    canonical_sha256,
    validate_manifest,
    write_manifest,
)


class FrozenTargetManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry_path = self.root / "target_registry.json"
        self.corpus_path = self.root / "train.json"
        self.checkpoint_path = self.root / "checkpoint"
        self.checkpoint_path.mkdir()
        (self.checkpoint_path / "model.safetensors").write_bytes(b"fixed weights")
        (self.checkpoint_path / "config.json").write_text('{"model_type":"gpt2"}\n')

        registry = []
        corpus = []
        serial = 0
        # The eligible pool is deliberately larger than each quota.  Selection
        # must therefore operate on the complete eligible pool in each tier.
        for frequency, count in ((1, 4), (5, 9), (20, 18)):
            for _ in range(count):
                serial += 1
                person = self._person("d", serial)
                registry.append({
                    "person": person,
                    "frequency": frequency,
                    "is_negative_control": False,
                })
                corpus.append({
                    "text": f"Account {person['name']}; {person['email']}; SSN {person['ssn']}."
                })

        # A positive-frequency person with only one exposed field must not be
        # allowed into D merely because the person occurs in the corpus.
        incomplete = self._person("d", 999)
        registry.append({
            "person": incomplete,
            "frequency": 20,
            "is_negative_control": False,
        })
        corpus.append({"text": f"Only email is present: {incomplete['email']}"})

        for index in range(1, 31):
            registry.append({
                "person": self._person("c", index),
                "frequency": 0,
                "is_negative_control": True,
            })

        self.registry_path.write_text(json.dumps(registry))
        self.corpus_path.write_text(json.dumps(corpus))

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _person(prefix, index):
        # D and C occupy disjoint numeric ranges so no value is accidentally a
        # substring of another field in the corpus.
        offset = index if prefix == "d" else 5000 + index
        return {
            "name": f"{prefix.upper()} Person {index:03d}",
            "ssn": f"{100 + offset % 800:03d}-{10 + offset % 80:02d}-{1000 + offset:04d}",
            "email": f"{prefix}{index:03d}@example.invalid",
        }

    @staticmethod
    def _features(person, field):
        number = int(person["name"].split()[-1])
        field_offset = 0.0 if field == "ssn" else 0.25
        return {
            "char_len": 10.0 + (number % 11) + field_offset,
            "tok_len": 4.0 + (number % 7) + field_offset,
            "H_bits": 20.0 + number / 10.0 + field_offset,
        }

    def _build(self, feature_provider=None, code_state=None):
        code_state = code_state or {
            "commit": "0123456789abcdef",
            "dirty": False,
            "dirty_files": [],
        }
        with mock.patch("e3_target_manifest._git_state", return_value=code_state):
            return build_manifest(
                registry_path=self.registry_path,
                corpus_path=self.corpus_path,
                checkpoint_path=self.checkpoint_path,
                feature_provider=feature_provider or self._features,
                reference_metadata={"name": "fake-reference", "revision": "fixed"},
                repo_root=self.root,
                created_at="2026-09-20T12:00:00+00:00",
            )

    def _write(self, manifest):
        path = self.root / "e3b.json"
        write_manifest(manifest, path)
        return path

    def test_build_is_deterministic_stratified_joint_and_without_replacement(self):
        first = self._build()
        second = self._build()
        self.assertEqual(first, second)
        self.assertEqual(first["selection"]["eligible_counts"], {"1": 4, "5": 9, "20": 18})
        self.assertEqual(first["selection"]["selected_tiers"], {"1": 3, "5": 7, "20": 15})
        self.assertEqual(first["matching"]["fields_jointly_matched"], ["ssn", "email"])
        self.assertFalse(first["matching"]["with_replacement"])
        self.assertTrue(first["matching"]["balance_gate"]["passed"])
        self.assertEqual(first["matching"]["balance_gate"]["threshold"], 0.1)
        for field in ("ssn", "email"):
            self.assertTrue(all(
                abs(value) <= 0.1
                for value in first["matching"]["balance"][field]["marginal_smd"]
            ))
        self.assertEqual(len(first["pairs"]), 25)

        trained = [pair["trained"]["person_id"] for pair in first["pairs"]]
        controls = [pair["control"]["person_id"] for pair in first["pairs"]]
        self.assertEqual(len(set(trained)), 25)
        self.assertEqual(len(set(controls)), 25)
        self.assertNotIn("D Person 999", trained)
        for pair in first["pairs"]:
            for field in ("ssn", "email"):
                self.assertTrue(pair["trained"]["targets"][field]["exposed_in_finetuning_text"])
                self.assertFalse(pair["control"]["targets"][field]["exposed_in_finetuning_text"])

        report = validate_manifest(
            self._write(first),
            registry_path=self.registry_path,
            corpus_path=self.corpus_path,
            checkpoint_path=self.checkpoint_path,
        )
        self.assertEqual(len(report["trained_entries"]), 25)
        self.assertEqual(len(report["control_entries"]), 25)
        self.assertEqual(report["pair_assignment_sha256"], first["integrity"]["pair_assignment_sha256"])

    def test_structural_validation_rejects_control_reuse_even_with_valid_hashes(self):
        manifest = copy.deepcopy(self._build())
        manifest["pairs"][1]["control"] = copy.deepcopy(manifest["pairs"][0]["control"])
        identities = []
        for pair in manifest["pairs"]:
            for arm in ("trained", "control"):
                record = pair[arm]
                for field in manifest["fields"]:
                    identities.append((
                        arm,
                        record["person_id"],
                        field,
                        record["targets"][field]["value_sha256"],
                    ))
        manifest["integrity"]["target_values_sha256"] = canonical_sha256(sorted(identities))
        manifest["integrity"]["pair_assignment_sha256"] = canonical_sha256([
            (pair["trained"]["person_id"], pair["control"]["person_id"])
            for pair in manifest["pairs"]
        ])
        manifest["integrity"]["payload_sha256"] = _payload_digest(manifest)

        with self.assertRaisesRegex(TargetManifestError, "not unique"):
            validate_manifest(
                self._write(manifest),
                registry_path=self.registry_path,
                corpus_path=self.corpus_path,
                checkpoint_path=self.checkpoint_path,
            )

    def test_validation_rejects_source_byte_changes(self):
        path = self._write(self._build())
        corpus = json.loads(self.corpus_path.read_text())
        corpus.append({"text": "a byte-level source mutation"})
        self.corpus_path.write_text(json.dumps(corpus))
        with self.assertRaisesRegex(TargetManifestError, "differs from frozen manifest"):
            validate_manifest(
                path,
                registry_path=self.registry_path,
                corpus_path=self.corpus_path,
                checkpoint_path=self.checkpoint_path,
            )

    def test_validation_rejects_payload_tampering(self):
        manifest = self._build()
        path = self._write(manifest)
        mutated = json.loads(path.read_text())
        mutated["execution_contract"]["gcg_iters"] = 201
        path.write_text(json.dumps(mutated))
        with self.assertRaisesRegex(TargetManifestError, "payload hash mismatch"):
            validate_manifest(
                path,
                registry_path=self.registry_path,
                corpus_path=self.corpus_path,
                checkpoint_path=self.checkpoint_path,
            )

    def test_frozen_manifest_cannot_be_overwritten_with_different_bytes(self):
        manifest = self._build()
        path = self._write(manifest)
        write_manifest(copy.deepcopy(manifest), path)  # exact idempotent retry
        changed = copy.deepcopy(manifest)
        changed["target_set_id"] = "post-outcome-reselection"
        with self.assertRaisesRegex(TargetManifestError, "Refusing to overwrite"):
            write_manifest(changed, path)

    def test_build_hard_fails_and_names_covariates_when_balance_exceeds_point_one(self):
        def unbalanced_features(person, field):
            values = dict(self._features(person, field))
            if person["name"].startswith("C "):
                values["char_len"] += 1000.0
                values["tok_len"] += 1000.0
                values["H_bits"] += 1000.0
            return values

        with self.assertRaisesRegex(
            TargetManifestError,
            r"every \|SMD\| <= 0\.100.*ssn\.char_len=.*email\.H_bits=",
        ):
            self._build(feature_provider=unbalanced_features)

    def test_builder_requires_nonempty_clean_git_commit(self):
        for code_state in (
            {"commit": None, "dirty": False, "dirty_files": []},
            {"commit": "abc123", "dirty": True, "dirty_files": ["M experiments.py"]},
            {"commit": "abc123", "dirty": None, "dirty_files": []},
        ):
            with self.subTest(code_state=code_state):
                with self.assertRaisesRegex(TargetManifestError, "contract violation"):
                    self._build(code_state=code_state)

    def test_validator_rejects_every_wrong_registered_contract_field(self):
        base = self._build()
        mutations = {
            "tier_quotas": lambda value: value["selection"].__setitem__(
                "tier_quotas", {"1": 4, "5": 6, "20": 15}
            ),
            "k_grid": lambda value: value["execution_contract"].__setitem__(
                "k_grid", [0, 1]
            ),
            "seeds": lambda value: value["execution_contract"].__setitem__(
                "seeds", [42]
            ),
            "gcg_iters": lambda value: value["execution_contract"].__setitem__(
                "gcg_iters", 201
            ),
            "model_name": lambda value: value["execution_contract"].__setitem__(
                "model_name", "gpt2-medium"
            ),
            "model_state": lambda value: value["execution_contract"].__setitem__(
                "model_state", "base"
            ),
            "fields": lambda value: value["execution_contract"].__setitem__(
                "fields", ["ssn"]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(base)
                mutate(changed)
                changed["integrity"]["payload_sha256"] = _payload_digest(changed)
                path = self.root / f"wrong-{name}.json"
                write_manifest(changed, path)
                with self.assertRaisesRegex(TargetManifestError, "contract violation"):
                    validate_manifest(
                        path,
                        registry_path=self.registry_path,
                        corpus_path=self.corpus_path,
                        checkpoint_path=self.checkpoint_path,
                    )

    def test_validator_rejects_dirty_manifest_code_even_if_payload_is_rehashed(self):
        changed = copy.deepcopy(self._build())
        changed["code"]["dirty"] = True
        changed["integrity"]["payload_sha256"] = _payload_digest(changed)
        path = self.root / "dirty-code.json"
        write_manifest(changed, path)
        with self.assertRaisesRegex(TargetManifestError, "code.dirty must be false"):
            validate_manifest(
                path,
                registry_path=self.registry_path,
                corpus_path=self.corpus_path,
                checkpoint_path=self.checkpoint_path,
            )


if __name__ == "__main__":
    unittest.main()
