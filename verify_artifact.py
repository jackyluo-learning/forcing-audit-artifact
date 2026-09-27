"""Verify the release's files and chunked payloads using only Python's standard library."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def local_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside artifact: {relative}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = (ROOT / 'SHA256SUMS').read_text().splitlines()
    for line in entries:
        expected, relative = line.split('  ', 1)
        path = local_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or modified release file: {relative}")
    inventory = json.loads((ROOT / 'payloads/inventory.json').read_text())
    for relative, spec in inventory.items():
        full = hashlib.sha256()
        size = 0
        for chunk in spec['chunks']:
            path = local_path(f"payloads/{relative}/{chunk['name']}")
            data = path.read_bytes()
            if len(data) != chunk['bytes'] or hashlib.sha256(data).hexdigest() != chunk['sha256']:
                raise ValueError(f"Invalid payload part: {path.relative_to(ROOT)}")
            full.update(data)
            size += len(data)
        if size != spec['bytes'] or full.hexdigest() != spec['sha256']:
            raise ValueError(f"Invalid reassembled payload: {relative}")
        restored = local_path(relative)
        if restored.exists() and (restored.stat().st_size != size or sha256(restored) != spec['sha256']):
            raise ValueError(f"Modified restored payload: {relative}")
    for folder, suffix in [('attempts', 'parquet'), ('manifests', 'json')]:
        if len(list((ROOT / 'results' / folder).glob(f'e3b__E3__*.{suffix}'))) != 42:
            raise ValueError(f"Incomplete capacity {folder}")
        if len(list((ROOT / 'results' / folder).glob(f'e3b_base__E2B__*.{suffix}'))) != 3:
            raise ValueError(f"Incomplete base {folder}")
    print(f'Verified {len(entries)} release files, both complete payloads, and the 42+3 evidence-shard inventory.')
    print('Run reproduce.py for the row-level scientific acceptance gates and analyses.')


if __name__ == '__main__':
    main()
