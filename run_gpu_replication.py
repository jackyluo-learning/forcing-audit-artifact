"""Run a new attack shard in an isolated output directory, retaining the archived evidence."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
K_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', choices=('finetuned', 'base'), required=True)
    parser.add_argument('--seed', type=int, choices=(42, 1337, 2024), required=True)
    parser.add_argument('--k', type=int, choices=K_GRID, default=20)
    parser.add_argument('--base-snapshot', type=Path)
    args = parser.parse_args()
    if args.state == 'base' and (args.k != 20 or args.base_snapshot is None):
        parser.error('Base follow-up requires --k 20 and --base-snapshot PATH.')
    os.chdir(ROOT)
    for required in ('models/gpt2/model.safetensors', 'data/corpus/train.json'):
        if not (ROOT / required).is_file():
            parser.error(f'Missing {required}; run restore_payloads.py first.')
    out = ROOT / 'output/gpu_replication' / f'{args.state}-seed{args.seed}-k{args.k}'
    if out.exists():
        parser.error(f'Refusing to overwrite an earlier replication: {out.relative_to(ROOT)}')
    run_id = 'e3b_base' if args.state == 'base' else 'replication_ft'
    os.environ.update({
        'PII_DEVICE_PROFILE': 'a100_80', 'PII_MODELS': 'gpt2',
        'PII_FIELDS': 'ssn,email', 'PII_GCG_ITERS': '200',
        'PII_CAP_SWEEP_N': '25', 'PII_CAP_K': str(args.k),
        'PII_SEEDS': '42,1337,2024', 'PII_RUN_ID': run_id,
        'PII_KGRID': ','.join(map(str, K_GRID)), 'PII_ADAPTIVE_LAMBDA': '0.1',
        'PII_E3_TARGET_MANIFEST': 'results/target_sets/e3b.json',
    })
    # Set the result path before the logger and experiment modules import it.
    import config
    if config.DEVICE != 'cuda':
        raise RuntimeError('GPU replication requires CUDA; CPU analysis is reproduce.py.')
    import run_manifest
    state = run_manifest.git_state()
    if not state['commit'] or state['dirty']:
        raise RuntimeError('Commit local source changes before GPU replication.')
    out.mkdir(parents=True)
    config.RESULTS_DIR = str(out / 'results')
    Path(config.RESULTS_DIR).mkdir()
    (out / 'replication.json').write_text(json.dumps({
        'kind': 'new_replication_not_archived_evidence', 'state': args.state,
        'seed': args.seed, 'k': args.k, 'code': state,
        'note': 'Original experiment IDs are retained inside this isolated result directory.',
    }, indent=2) + '\n')
    if args.state == 'finetuned':
        import experiments
        experiments.run_E3_capacity_sweep('gpt2', args.seed)
    else:
        import run_e3b_base_followup
        sys.argv = ['run_e3b_base_followup.py', '--target-manifest',
                    'results/target_sets/e3b.json', '--base-snapshot',
                    str(args.base_snapshot.resolve()), '--seed', str(args.seed),
                    '--run-id', run_id]
        run_e3b_base_followup.main()
    print(f'New replication saved under {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
