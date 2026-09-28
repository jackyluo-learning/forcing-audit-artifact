"""Regenerate data, fine-tune one model, and run the E5 prefix-likelihood test.

This is an artifact integration entry point, separate from the preserved source
snapshot listed in SOURCE_SNAPSHOT.json. It creates a new run; it does not claim
to reconstruct an unavailable historical checkpoint or training corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/full_fast.yaml")
    parser.add_argument("--model-name", default="pythia-1.4b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-controls", type=int, default=200)
    parser.add_argument("--control-seed", type=int, default=99991)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/reproduce_e5")
    parser.add_argument("--plan", action="store_true",
                        help="show the run settings without importing GPU dependencies")
    parser.add_argument("--prepare-only", action="store_true",
                        help="generate and audit inputs without training a model")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"configuration does not exist: {config_path}")
    if output_dir == ROOT or ROOT.is_relative_to(output_dir):
        parser.error("output directory must not contain the source tree")
    if args.n_controls < 1:
        parser.error("--n-controls must be positive")
    if args.plan:
        print(json.dumps({
            "config": str(config_path), "model_name": args.model_name,
            "seed": args.seed, "n_controls": args.n_controls,
            "control_seed": args.control_seed, "output_dir": str(output_dir),
            "stages": ["generate corpus", "fine-tune model", "generate controls",
                       "score both arms", "write provenance"],
        }, indent=2))
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory already contains files: {output_dir}")

    import yaml
    from faker import Faker
    from src.data_gen import _make_individual, generate

    with config_path.open() as stream:
        cfg = yaml.safe_load(stream)
    model_spec = next((m for m in cfg["models"] if m["name"] == args.model_name), None)
    if model_spec is None:
        parser.error(f"model {args.model_name!r} is not listed in {config_path}")
    if args.seed not in cfg["seeds"]:
        parser.error(f"seed {args.seed} is not listed in {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    data_dir = output_dir / "data"
    generate(cfg, str(data_dir))

    individuals = data_dir / "individuals.json"
    with individuals.open() as stream:
        trained = json.load(stream)
    corpus = data_dir / "corpus.jsonl"
    corpus_text = "\n".join(json.loads(line)["text"] for line in corpus.read_text().splitlines())

    # Draw controls from the same Faker record generator as D, using a disjoint
    # stream. This avoids making the arm label trivially identifiable from a
    # different SSN area or email domain.
    fake = Faker()
    fake.seed_instance(args.control_seed)
    controls = []
    used = {str(p.get(field, "")) for p in trained for field in ("ssn", "email", "name")}
    candidate = 0
    while len(controls) < args.n_controls:
        rec = _make_individual(fake, 1_000_000 + candidate)
        candidate += 1
        values = [str(rec[field]) for field in ("ssn", "email", "name")]
        if any(value in used for value in values):
            continue
        if any(str(rec[field]) in corpus_text for field in ("ssn", "email")):
            continue
        rec["frequency"] = 0
        controls.append(rec)
        used.update(values)
    controls_path = data_dir / "controls.json"
    controls_path.write_text(json.dumps(controls, indent=2) + "\n")

    exposure = {
        arm: {field: {
            "present": sum(str(p.get(field, "")) in corpus_text for p in people if p.get(field)),
            "total": sum(bool(p.get(field)) for p in people),
        } for field in ("ssn", "email")}
        for arm, people in (("trained", trained), ("control", controls))
    }
    print(f"[reproduce-e5] field exposure in the generated corpus: {exposure}")
    if args.prepare_only:
        print(f"[reproduce-e5] prepared inputs: {data_dir}")
        return

    from src.train import train_one
    checkpoint = Path(train_one(cfg, model_spec, args.seed, str(data_dir))).resolve()

    result_path = output_dir / "e5_bits.json"
    command = [
        sys.executable, str(ROOT / "experiments/e5_nll_decomposition.py"),
        "--model", str(checkpoint),
        "--base-model", model_spec["hf_id"],
        "--individuals", str(individuals),
        "--controls", str(controls_path),
        "--fields", "ssn", "email",
        "--dtype", "float32",
        "--out", str(result_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    with result_path.open() as stream:
        result = json.load(stream)
    expected_trained = sum(bool(p.get(field)) for p in trained for field in ("ssn", "email"))
    expected_control = sum(bool(p.get(field)) for p in controls for field in ("ssn", "email"))
    summary = result["summary"]
    if summary["trained"]["n"] != expected_trained or summary["control"]["n"] != expected_control:
        raise RuntimeError("E5 output row counts do not match the generated targets")
    if len(result["rows"]) != expected_trained + expected_control:
        raise RuntimeError("E5 per-target rows are incomplete")

    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "datasets", "Faker", "PyYAML"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    provenance = {
        "run_kind": "from_scratch_reproduction",
        "source_config": str(config_path),
        "source_config_sha256": sha256(config_path),
        "config_used_sha256": sha256(output_dir / "config_used.yaml"),
        "model_name": args.model_name,
        "base_model": model_spec["hf_id"],
        "seed": args.seed,
        "control_seed": args.control_seed,
        "control_generator": "same Faker record generator as trained arm; disjoint seed",
        "n_trained_people": len(trained),
        "n_control_people": len(controls),
        "field_exposure": exposure,
        "versions": versions,
        "individuals_sha256": sha256(individuals),
        "controls_sha256": sha256(controls_path),
        "corpus_sha256": sha256(corpus),
        "checkpoint": str(checkpoint),
        "result_sha256": sha256(result_path),
        "summary": summary,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[reproduce-e5] result: {result_path}")
    print(f"[reproduce-e5] provenance: {output_dir / 'provenance.json'}")


if __name__ == "__main__":
    main()
