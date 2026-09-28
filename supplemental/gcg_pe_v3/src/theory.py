"""Reproducible theory numbers for Section 4 of the paper.

Every constant that appears in the theory section is computed here, never typed
by hand:

  * H_inf  -- the *min*-entropy of a target generator (not log2 of its support).
  * m_S    -- the multiplicity of the normalized-substring scoring rule, derived
              from the tokenizer, not guessed.
  * the counting bound (Prop. 1), the target-blind bound (Cor. 1.2), the
    auditable-entropy design rule (Cor. 1.3), k_vac (Prop. 2), the query-budget
    extension (Cor. queries) and the fluency budget (Cor. fluency).

Run it as a script to print the table of numbers the paper quotes:

    python -m src.theory                     # uses the Faker en_US generator
    python -m src.theory --tokenizer gpt2 --decode-len 48

IMPORTANT (validity direction): a bound of the form  alpha <= m_S * Q * 2^-H_inf
is only valid if H_inf is a *lower* bound on the true min-entropy and m_S an
*upper* bound on the true multiplicity. Monte-Carlo sampling of a generator gives
neither, so `field_min_entropy` refuses to return an MC number as if it were
certified: it either computes p_max exactly by enumeration, or returns
`certified=False` and you must not put the number in a theorem. For control
generators, prefer `src.controls`, whose min-entropy is exact by construction.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Min-entropy of a target generator
# --------------------------------------------------------------------------- #
@dataclass
class Entropy:
    field: str
    h_inf_bits: float          # -log2 p_max
    p_max: float
    support: Optional[int]     # None when not enumerable
    h_max_bits: Optional[float]  # log2 support, for contrast; NOT min-entropy
    certified: bool            # True only if p_max was computed exactly
    note: str = ""

    def as_row(self) -> Dict[str, object]:
        d = asdict(self)
        d["h_inf_bits"] = round(self.h_inf_bits, 3)
        if self.h_max_bits is not None:
            d["h_max_bits"] = round(self.h_max_bits, 3)
        return d


def ssn_min_entropy_en_us() -> Entropy:
    """Exact min-entropy of Faker's en_US ssn().

    The provider draws area in 1..899, group in 1..99, serial in 1..9999 and
    remaps the invalid area 666 to 667, so 667 carries twice the mass of every
    other area. The paper previously used log2(10**9) = 29.897, which is the
    min-entropy of a *uniform* nine-digit string and overstates this generator's
    min-entropy by about 1.17 bits.
    """
    n_area, n_group, n_serial = 899, 99, 9999
    areas = [a if a != 666 else 667 for a in range(1, n_area + 1)]
    counts = Counter(areas)
    p_max_area = max(counts.values()) / n_area          # = 2/899
    p_max = p_max_area * (1.0 / n_group) * (1.0 / n_serial)
    support = len(counts) * n_group * n_serial
    return Entropy(
        field="ssn",
        h_inf_bits=-math.log2(p_max),
        p_max=p_max,
        support=support,
        h_max_bits=math.log2(support),
        certified=True,
        note="Faker en_US ssn(): area 1..899 with 666->667, group 1..99, "
             "serial 1..9999. Area 667 carries double mass.",
    )


def _weighted_pmax(elements) -> Optional[float]:
    """Largest element probability of a Faker element list or weighted dict."""
    if isinstance(elements, dict):
        total = float(sum(elements.values()))
        return max(elements.values()) / total if total else None
    try:
        n = len(elements)
        return 1.0 / n if n else None
    except TypeError:
        return None


def _normalized_weights(elements) -> Optional[Dict[str, float]]:
    """Faker element lists are either a plain sequence (uniform) or an
    OrderedDict of weights. Return {value: probability}."""
    if elements is None:
        return None
    if isinstance(elements, dict):
        total = float(sum(elements.values()))
        if total <= 0:
            return None
        return {k: v / total for k, v in elements.items()}
    try:
        n = len(elements)
    except TypeError:
        return None
    return {v: 1.0 / n for v in elements} if n else None


def email_min_entropy_en_us(safe: bool = True) -> Entropy:
    """Exact min-entropy of Faker's en_US email() by full enumeration.

    A product of modal element weights would be a *lower* bound on p_max, which
    is the wrong direction: the scoring rule sees the slugified, lower-cased
    local part, so distinct names can collide ("O'Brien" and "OBrien" both become
    "obrien") and collisions only concentrate mass. We therefore enumerate the
    whole local-part distribution, apply Faker's own transform, aggregate
    colliding strings, and take the true maximum. The support is at most
    |first|*|last|*2 + |first|*100 + 26*|last|, about 1.5M strings for en_US,
    so this is cheap and exact.
    """
    try:
        from faker.providers.person.en_US import Provider as PersonProvider
        from faker.providers.internet.en_US import Provider as NetProvider
        from faker.providers.internet import Provider as NetBase
        from faker.utils.text import slugify
    except Exception as exc:  # pragma: no cover
        return Entropy("email", float("nan"), float("nan"), None, None, False,
                       f"faker not importable: {exc}")

    first = _normalized_weights(getattr(PersonProvider, "first_names", None))
    last = _normalized_weights(getattr(PersonProvider, "last_names", None))
    formats = getattr(NetProvider, "user_name_formats", None) \
        or getattr(NetBase, "user_name_formats", None)
    dom_attr = "safe_domain_names" if safe else "free_email_domains"
    domains = _normalized_weights(
        getattr(NetProvider, dom_attr, None) or getattr(NetBase, dom_attr, None))

    if not first or not last or not formats or not domains:
        return Entropy("email", float("nan"), float("nan"), None, None, False,
                       "provider shape not recognised; enumerate manually or "
                       "use src.controls for an exact-min-entropy generator")

    def canon(s: str) -> str:
        return slugify(s.lower(), allow_dots=True)

    local: Dict[str, float] = {}
    p_fmt = 1.0 / len(formats)
    for fmt in formats:
        n_first = fmt.count("{{first_name}}")
        n_last = fmt.count("{{last_name}}")
        n_digit = fmt.count("#")
        n_alpha = fmt.count("?")
        if n_first + n_last == 0 or n_first > 1 or n_last > 1:
            # An unrecognised format shape: refuse rather than approximate.
            return Entropy("email", float("nan"), float("nan"), None, None,
                           False, f"unhandled user_name format {fmt!r}")
        firsts = first.items() if n_first else [("", 1.0)]
        lasts = last.items() if n_last else [("", 1.0)]
        # '#' and '?' are independent uniform draws; they never collide with
        # each other, so they contribute a flat factor to every candidate.
        filler = (10.0 ** -n_digit) * (26.0 ** -n_alpha)
        for fv, fp in firsts:
            for lv, lp in lasts:
                s = fmt.replace("{{first_name}}", fv).replace("{{last_name}}", lv)
                s = s.replace("#", "0").replace("?", "a")   # shape only
                key = canon(s)
                local[key] = local.get(key, 0.0) + p_fmt * fp * lp * filler
    p_max = max(local.values()) * max(domains.values())
    modal = max(local, key=local.get)
    return Entropy(
        field="email",
        h_inf_bits=-math.log2(p_max),
        p_max=p_max,
        support=len(local) * len(domains),
        h_max_bits=math.log2(len(local) * len(domains)),
        certified=True,
        note=f"Faker en_US email(safe={safe}), exact enumeration of "
             f"{len(local)} distinct local parts; modal local part {modal!r}. "
             "Digit/letter fillers enter as a flat factor, so the reported "
             "support counts local-part shapes, not distinct emails.",
    )


def uniform_min_entropy(field: str, support: int) -> Entropy:
    """Min-entropy of a generator that is uniform over `support` values."""
    return Entropy(field, math.log2(support), 1.0 / support, support,
                   math.log2(support), True, "uniform by construction")


# --------------------------------------------------------------------------- #
# Multiplicity of the normalized-substring scoring rule
# --------------------------------------------------------------------------- #
@dataclass
class Multiplicity:
    decode_len_L: int
    target_len_chars: int
    chars_per_token_max: int         # c_V, rigorous
    chars_per_token_typical: int     # c_V excluding rare long tokens
    m_s_rigorous: int
    m_s_typical: int
    long_tokens: List[str]
    note: str = ""

    def as_row(self) -> Dict[str, object]:
        d = asdict(self)
        d["long_tokens"] = d["long_tokens"][:8]
        return d


def digit_multiplicity(tokenizer, decode_len_L: int, target_len_chars: int = 9,
                       typical_cut: int = 4) -> Multiplicity:
    """m_S for a numeric field scored after separator deletion.

    After the scoring rule deletes [\\s\\-().] from the generation, a match is
    any window of `target_len_chars` digits. An L-token output therefore contains
    at most c_V*L - target_len + 1 such windows, where c_V is the largest number
    of digits a single token of this tokenizer can contribute.

    Returns both the rigorous bound (true c_V) and the bound obtained by
    excluding the handful of unusually long all-digit tokens, which is what the
    paper reports as the tighter figure.
    """
    vocab = tokenizer.get_vocab()
    digit_lens = []
    long_tokens = []
    for tok in vocab:
        s = tokenizer.convert_tokens_to_string([tok])
        d = "".join(ch for ch in s if ch.isdigit())
        if d and d == s.strip():          # an all-digit token
            digit_lens.append(len(d))
            if len(d) > typical_cut:
                long_tokens.append(s.strip())
    c_max = max(digit_lens) if digit_lens else 1
    c_typ = typical_cut
    m_rig = max(1, c_max * decode_len_L - target_len_chars + 1)
    m_typ = max(1, c_typ * decode_len_L - target_len_chars + 1)
    return Multiplicity(
        decode_len_L=decode_len_L,
        target_len_chars=target_len_chars,
        chars_per_token_max=c_max,
        chars_per_token_typical=c_typ,
        m_s_rigorous=m_rig,
        m_s_typical=m_typ,
        long_tokens=sorted(long_tokens, key=len, reverse=True),
        note=f"{len(long_tokens)} all-digit tokens carry more than "
             f"{typical_cut} digits",
    )


def empirical_multiplicity(generations: Iterable[str], support: Sequence[str],
                           normalize) -> int:
    """m_S actually observed in a set of generations.

    This is a model-dependent lower bound on the structural cap and belongs in
    the reporting, not in a theorem. `normalize` should be the harness's
    normalization for the field (see src.utils.normalize_value).
    """
    norm_support = {normalize(v) for v in support}
    worst = 0
    for g in generations:
        ng = normalize(g)
        worst = max(worst, sum(1 for w in norm_support if w and w in ng))
    return worst


# --------------------------------------------------------------------------- #
# The bounds themselves
# --------------------------------------------------------------------------- #
def counting_bound(k: int, vocab_size: int, h_inf_bits: float,
                   m_s: int = 1) -> float:
    """Prop. 1 with Q = A_k: rho_k^subseteq <= min{1, m_S |V|^k 2^-H_inf}.

    Valid only for a prompt set and a model that are independent of the target,
    which holds for controls and fails for trained targets.
    """
    expo = k * math.log2(vocab_size) + math.log2(m_s) - h_inf_bits
    return min(1.0, 2.0 ** expo)


def k_vac(h_inf_bits: float, vocab_size: int) -> int:
    """Prop. 2: the capacity above which no uniform-over-models bound < 1 exists."""
    return math.ceil(h_inf_bits / math.log2(vocab_size))


def blind_bound(q_prompts: int, h_inf_bits: float, m_s: int = 1) -> float:
    """Cor. 1.2: alpha <= min{1, m_S Q 2^-H_inf} for target-blind search.

    Note the absence of k: this holds at every prompt length, which is why it
    still says something at k=20 where the counting bound in k has clipped to 1.
    """
    return min(1.0, m_s * q_prompts * 2.0 ** (-h_inf_bits))


def design_rule_bits(q_prompts: int, tolerance: float, m_s: int = 1) -> float:
    """Cor. 1.3: the min-entropy a field needs for a certified blind background."""
    return math.log2(m_s * q_prompts / tolerance)


def max_blind_budget(h_inf_bits: float, tolerance: float, m_s: int = 1) -> float:
    """Cor. 1.3 solved for Q instead of H_inf."""
    return tolerance * 2.0 ** h_inf_bits / m_s


def query_budget_bound(q_prompts: int, n_samples: int, p: float,
                       h_inf_bits: float, m_s: int = 1) -> float:
    """Cor. queries: (n,p)-discoverable extraction under target-blind search.

    A per-sample probability q qualifies when 1-(1-q)^n >= p, i.e.
    q >= 1-(1-p)^(1/n); at most 1/q outputs per prompt meet that threshold.
    """
    q_thresh = 1.0 - (1.0 - p) ** (1.0 / n_samples)
    return min(1.0, m_s * q_prompts / q_thresh * 2.0 ** (-h_inf_bits))


def fluency_bound(eta_bits: float, k: int, vocab_size: int, h_inf_bits: float,
                  m_s: int = 1) -> float:
    """Cor. fluency: a hard prompt-likelihood budget of eta bits caps the
    reachable set as a probe of eta/log2|V| tokens would."""
    expo = min(k * math.log2(vocab_size), eta_bits) + math.log2(m_s) - h_inf_bits
    return min(1.0, 2.0 ** expo)


def fluency_budget_for_tolerance(tolerance: float, h_inf_bits: float,
                                 m_s: int = 1) -> float:
    """The eta (total prompt bits) a fluency filter would need to certify
    `tolerance`. Compare against what deployed perplexity filters actually
    allow; ours comes out at well under one bit per token."""
    return h_inf_bits - math.log2(m_s / tolerance)


# --------------------------------------------------------------------------- #
# The paper's table of numbers
# --------------------------------------------------------------------------- #
def report(tokenizer_name: str = "gpt2", decode_len_L: int = 48,
           gcg_steps: int = 200, search_width: int = 512,
           random_restart_q: int = 512, tolerance: float = 0.01,
           measured_alpha_20: float = 116 / 150) -> Dict[str, object]:
    out: Dict[str, object] = {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = len(tok.get_vocab())
        mult = digit_multiplicity(tok, decode_len_L)
    except Exception as exc:                     # tokenizer unavailable offline
        print(f"[theory] tokenizer unavailable ({exc}); using GPT-2 constants")
        vocab_size = 50257
        mult = Multiplicity(decode_len_L, 9, 16, 4,
                            16 * decode_len_L - 8, 4 * decode_len_L - 8, [],
                            "hard-coded GPT-2 fallback; rerun with a tokenizer")
    ssn = ssn_min_entropy_en_us()
    email = email_min_entropy_en_us()

    out["vocab_size"] = vocab_size
    out["bits_per_token"] = math.log2(vocab_size)
    out["entropy"] = [ssn.as_row(), email.as_row()]
    out["multiplicity"] = mult.as_row()

    gcg_q = gcg_steps * search_width
    rows = []
    for e in (ssn, email):
        if not e.certified:
            continue
        for m_s, tag in ((1, "m_S=1"),
                         (mult.m_s_typical, f"m_S={mult.m_s_typical}"),
                         (mult.m_s_rigorous, f"m_S={mult.m_s_rigorous}")):
            rows.append({
                "field": e.field, "m_S": tag,
                "k_vac": k_vac(e.h_inf_bits, vocab_size),
                "counting_k1": counting_bound(1, vocab_size, e.h_inf_bits, m_s),
                "counting_k2": counting_bound(2, vocab_size, e.h_inf_bits, m_s),
                "blind_Q_restart": blind_bound(random_restart_q, e.h_inf_bits, m_s),
                "blind_Q_gcg": blind_bound(gcg_q, e.h_inf_bits, m_s),
                "design_bits_restart": design_rule_bits(random_restart_q, tolerance, m_s),
                "design_bits_gcg": design_rule_bits(gcg_q, tolerance, m_s),
                "eta_bits_for_tol": fluency_budget_for_tolerance(tolerance, e.h_inf_bits, m_s),
            })
    out["bounds"] = rows
    out["gcg_query_budget"] = gcg_q
    out["measured_alpha_20"] = measured_alpha_20
    out["any_upper_bound_at_k20_is_at_least"] = measured_alpha_20
    return out


def _fmt(x: float) -> str:
    if isinstance(x, str):
        return x
    if x >= 1.0:
        return "1 (vacuous)"
    if x >= 1e-3:
        return f"{100*x:.2f}%"
    return f"{x:.2e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--decode-len", type=int, default=48,
                    help="L, the number of tokens decoded per attempt")
    ap.add_argument("--gcg-steps", type=int, default=200)
    ap.add_argument("--search-width", type=int, default=512)
    ap.add_argument("--random-restart-q", type=int, default=512)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--json", default=None, help="also write the report here")
    a = ap.parse_args()

    r = report(a.tokenizer, a.decode_len, a.gcg_steps, a.search_width,
               a.random_restart_q, a.tolerance)

    print(f"\nvocabulary {r['vocab_size']}  ->  {r['bits_per_token']:.3f} "
          f"bits per token (nominal)")
    print("\n--- generator min-entropy (LOWER bounds are what a bound needs) ---")
    for e in r["entropy"]:
        flag = "exact" if e["certified"] else "NOT CERTIFIED - do not use in a bound"
        print(f"  {e['field']:6s} H_inf = {e['h_inf_bits']:7.3f} bits  "
              f"p_max = {e['p_max']:.3e}  [{flag}]")
        if e.get("h_max_bits"):
            print(f"         (log2 support = {e['h_max_bits']:.3f} bits; this is "
                  "NOT the min-entropy)")
    m = r["multiplicity"]
    print(f"\n--- scoring-rule multiplicity at L={m['decode_len_L']} ---")
    print(f"  max digits per token c_V = {m['chars_per_token_max']}  ->  "
          f"m_S <= {m['m_s_rigorous']} (rigorous)")
    print(f"  excluding long tokens c_V = {m['chars_per_token_typical']}  ->  "
          f"m_S <= {m['m_s_typical']}")
    if m["long_tokens"]:
        print(f"  longest all-digit tokens: {m['long_tokens']}")

    print(f"\n--- bounds (GCG query budget Q = {r['gcg_query_budget']}) ---")
    hdr = (f"{'field':6s} {'m_S':10s} {'k_vac':>5s} {'count k=1':>12s} "
           f"{'count k=2':>12s} {'blind Q=512':>12s} {'blind Q=gcg':>12s} "
           f"{'need bits':>10s}")
    print(hdr)
    for b in r["bounds"]:
        print(f"{b['field']:6s} {b['m_S']:10s} {b['k_vac']:5d} "
              f"{_fmt(b['counting_k1']):>12s} {_fmt(b['counting_k2']):>12s} "
              f"{_fmt(b['blind_Q_restart']):>12s} {_fmt(b['blind_Q_gcg']):>12s} "
              f"{b['design_bits_gcg']:10.2f}")
    print("\n  'need bits' is Cor. 1.3: the min-entropy required to certify the "
          "tolerance at the GCG budget.")
    print(f"  measured alpha at k=20 is {r['measured_alpha_20']:.3f}, so any "
          "valid upper bound there is at least that: only lower bounds inform.")

    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
