"""Real public-domain corpus loader.

Streams passages from HuggingFace datasets (Wikipedia / PG-19 / arXiv), chunks
them into passages, caches to disk, and is fully fault-tolerant: any source that
fails to download is skipped, and if *no* real text can be fetched (e.g. offline
server) it falls back to synthetic filler so the pipeline still runs.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

# Robust, no-auth, streamable defaults. Each entry: dataset id, optional config,
# split, and the text field. Order = priority.
DEFAULT_SOURCES: List[Dict[str, Any]] = [
    {"hf": "wikimedia/wikipedia", "config": "20231101.en", "split": "train",
     "field": "text"},
    {"hf": "pg19", "config": None, "split": "train", "field": "text",
     "trust_remote_code": True},
    {"hf": "CShorten/ML-ArXiv-Papers", "config": None, "split": "train",
     "field": "abstract"},
    # ultimate fallback (small, always works, no auth):
    {"hf": "wikitext", "config": "wikitext-103-raw-v1", "split": "train",
     "field": "text"},
]


def _chunk(text: str, max_words: int) -> List[str]:
    words = text.split()
    if len(words) <= 8:
        return []
    return [" ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)]


def _stream_source(src: Dict[str, Any], need: int, max_words: int,
                   seed: int) -> List[str]:
    from datasets import load_dataset

    kwargs = {"split": src.get("split", "train"), "streaming": True}
    if src.get("config"):
        kwargs["name"] = src["config"]
    if src.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    ds = load_dataset(src["hf"], **kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    field = src["field"]
    out: List[str] = []
    for ex in ds:
        text = (ex.get(field) or "").strip()
        if not text:
            continue
        for ch in _chunk(text, max_words):
            out.append(ch)
            if len(out) >= need:
                return out
    return out


def _synthetic_fallback(n: int, seed: int) -> List[str]:
    try:
        from faker import Faker

        fake = Faker()
        Faker.seed(seed)
        return [fake.paragraph(nb_sentences=random.randint(4, 12))
                for _ in range(n)]
    except Exception:
        random.seed(seed)
        vocab = ("the quick brown fox jumps over lazy dog while research "
                 "shows that language models memorize training data").split()
        return [" ".join(random.choices(vocab, k=random.randint(30, 80)))
                for _ in range(n)]


def load_public_passages(n: int, cache_path: str, seed: int = 42,
                         max_words: int = 350,
                         sources: Optional[List[Dict[str, Any]]] = None
                         ) -> List[str]:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = [json.loads(l)["text"] for l in f if l.strip()]
        if len(cached) >= n:
            print(f"[corpus] using cached {len(cached)} passages -> {cache_path}")
            return cached[:n]

    sources = sources or DEFAULT_SOURCES
    per_source = max(1, n // max(len(sources) - 1, 1))  # last is fallback only
    passages: List[str] = []
    for i, src in enumerate(sources):
        if len(passages) >= n:
            break
        need = n - len(passages) if i == len(sources) - 1 else per_source
        try:
            got = _stream_source(src, need, max_words, seed + i)
            passages.extend(got)
            print(f"[corpus] {src['hf']}: +{len(got)} passages "
                  f"(total {len(passages)}/{n})")
        except Exception as e:  # noqa: BLE001
            print(f"[corpus] WARN {src['hf']} failed ({type(e).__name__}: {e}); "
                  "skipping")

    if len(passages) < n:
        missing = n - len(passages)
        print(f"[corpus] WARN only {len(passages)} real passages; filling "
              f"{missing} with synthetic text")
        passages.extend(_synthetic_fallback(missing, seed))

    random.Random(seed).shuffle(passages)
    passages = passages[:n]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        for p in passages:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
    print(f"[corpus] cached {len(passages)} passages -> {cache_path}")
    return passages
