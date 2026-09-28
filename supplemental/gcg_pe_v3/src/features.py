"""24 linguistic/structural/model-based features for target sequences and the
prompt-property features used in the analysis tables. Mirrors Appendix B."""
from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List

import torch

_NLP = None
ACTION_VERBS = {"show", "list", "reveal", "print", "output", "give", "provide",
                "display", "return", "dump", "extract", "retrieve"}
DOMAIN_KEYWORDS = {"email", "ssn", "phone", "address", "record", "personal",
                   "data", "information", "private", "account", "card"}


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download as spacy_download

            print("[features] downloading spaCy en_core_web_sm ...")
            spacy_download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _zlib_ratio(text: str) -> float:
    b = text.encode("utf-8")
    return len(zlib.compress(b)) / max(len(b), 1)


@torch.no_grad()
def perplexity(model, tok, text: str, device: str = "cuda") -> float:
    enc = tok(text, return_tensors="pt").to(device)
    if enc["input_ids"].shape[1] < 2:
        return float("nan")
    out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).item())


def content_features(text: str, model=None, tok=None,
                     device: str = "cuda") -> Dict[str, float]:
    """The 24 features (Appendix B). Model-based ones require model+tok."""
    doc = _nlp()(text)
    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    chars = max(len(text), 1)
    words = [t.text.lower() for t in toks]
    uniq = len(set(words))
    depths = []
    for t in toks:
        d, cur = 0, t
        while cur.head != cur:
            d += 1
            cur = cur.head
            if d > 100:
                break
        depths.append(d)
    f = {
        # lexical
        "token_count": float(n),
        "type_token_ratio": uniq / n,
        "rare_word_ratio": sum(1 for t in toks if t.rank and t.rank > 10000) / n,
        "avg_token_len": sum(len(t.text) for t in toks) / n,
        "stopword_ratio": sum(1 for t in toks if t.is_stop) / n,
        "capitalization_ratio": sum(1 for t in toks if t.text[:1].isupper()) / n,
        # structural
        "entity_density": len(doc.ents) / n,
        "proper_noun_ratio": sum(1 for t in toks if t.pos_ == "PROPN") / n,
        "number_ratio": sum(1 for t in toks if t.pos_ == "NUM") / n,
        "special_char_ratio": sum(1 for c in text if not c.isalnum()
                                  and not c.isspace()) / chars,
        "punct_density": sum(1 for t in toks if t.is_punct) / n,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / chars,
        # syntactic
        "max_dep_depth": float(max(depths) if depths else 0),
        "avg_dep_depth": (sum(depths) / len(depths)) if depths else 0.0,
        "sentence_count": float(len(list(doc.sents))),
        "avg_sentence_len": n / max(len(list(doc.sents)), 1),
        "noun_phrase_count": float(len(list(doc.noun_chunks))),
        "verb_phrase_count": float(sum(1 for t in toks if t.pos_ == "VERB")),
        # model-based
        "compression_ratio": _zlib_ratio(text),
        "entropy_estimate": _zlib_ratio(text) * 8.0,
    }
    if model is not None and tok is not None:
        ppl = perplexity(model, tok, text, device)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        ids = enc["input_ids"][0]
        if ids.shape[0] > 1:
            logp = torch.log_softmax(logits[0, :-1], dim=-1)
            sur = -logp[range(ids.shape[0] - 1), ids[1:]]
            f.update({
                "perplexity": ppl,
                "avg_surprisal": float(sur.mean().item()),
                "surprisal_variance": float(sur.var().item()),
                "max_surprisal": float(sur.max().item()),
            })
        else:
            f.update({"perplexity": ppl, "avg_surprisal": float("nan"),
                      "surprisal_variance": float("nan"),
                      "max_surprisal": float("nan")})
    return f


def prompt_features(prompt: str) -> Dict[str, float]:
    """Features distinguishing baseline vs GCG prompts (Table 7)."""
    doc = _nlp()(prompt)
    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    words = [t.text.lower() for t in toks]
    counts: Dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    repeated = sum(c for c in counts.values() if c > 1)
    depths = []
    for t in toks:
        d, cur = 0, t
        while cur.head != cur:
            d += 1
            cur = cur.head
            if d > 100:
                break
        depths.append(d)
    return {
        "token_repetition": repeated / n,
        "action_verbs": float(sum(1 for w in words if w in ACTION_VERBS)),
        "rare_token_ratio": sum(1 for t in toks if t.rank and t.rank > 10000) / n,
        "domain_keywords": float(sum(1 for w in words if w in DOMAIN_KEYWORDS)),
        "syntactic_depth": (sum(depths) / len(depths)) if depths else 0.0,
    }
