"""Restore and verify the chunked training corpus and GPT-2 checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def restore() -> None:
    inventory = json.loads((ROOT / "payloads/inventory.json").read_text())
    for relative, spec in inventory.items():
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        full_hash = hashlib.sha256()
        size = 0
        with temporary.open("wb") as output:
            for chunk in spec["chunks"]:
                part = ROOT / "payloads" / relative / chunk["name"]
                data = part.read_bytes()
                if len(data) != chunk["bytes"] or hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                    raise ValueError(f"Damaged part: {part}")
                output.write(data)
                full_hash.update(data)
                size += len(data)
        if size != spec["bytes"] or full_hash.hexdigest() != spec["sha256"]:
            raise ValueError(f"Damaged restored file: {relative}")
        temporary.replace(destination)
        print(f"Verified {relative}: {size} bytes")


if __name__ == "__main__":
    restore()
