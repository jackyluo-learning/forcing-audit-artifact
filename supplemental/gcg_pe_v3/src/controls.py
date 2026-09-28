"""Negative-control records: the instrument the whole paper rests on.

Three problems with the v2 approach, all fixed here.

1. THE HARNESS HAS NO CONTROL ARM. A grep for negative_control /
   target_membership over the released src/ returns nothing; data_gen.py draws
   one population from one seed. The paper's "single methodological requirement"
   is therefore unimplemented in the artifact.

2. FAKER'S SSNs MAY BELONG TO REAL PEOPLE. Faker's en_US ssn() draws area in
   1..899, i.e. from the space the SSA has actually issued from since randomized
   assignment in 2011, and credit_card_number() emits Luhn-valid numbers. The
   paper asserts "No real personal information is used" without computing a
   collision probability. Worse, the honeytoken proposal would have a deployed
   model emit such strings to users and write them into detector logs. Here
   SSN-shaped values are drawn from the never-issued area range 900-999, card
   numbers are deliberately Luhn-INVALID, phone numbers use the 555-01xx
   reserved block, and emails use the reserved example.com family. Collision with
   a real identifier is then impossible by construction rather than improbable by
   assumption.

3. THE GENERATOR'S MIN-ENTROPY WAS NOT KNOWN. A bound of the form
   alpha <= m_S Q 2^-H_inf needs a LOWER bound on H_inf, and Monte-Carlo
   sampling of a third-party generator does not give one. Every field here is
   uniform over a stated space, so H_inf is exact and `min_entropy_bits()`
   returns it. That is what makes Corollary cor:blind quotable.

Matching. Controls are drawn from the same generator with a disjoint seed stream
and then matched to trained records on the covariates that plausibly affect
forcibility: field type, character length, tokenized length, and the target's
surprisal under a HELD-OUT reference model (not under the model being audited,
which would be circular). `match()` does greedy nearest-neighbour matching on
standardised covariates and returns the achieved balance so exchangeability is
checked rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Field generators with exact, stated min-entropy
# --------------------------------------------------------------------------- #
# Area numbers 900-999 have never been issued by the SSA, so a value here cannot
# collide with a real SSN. Group 01-99 and serial 0001-9999 as in the real format.
SSN_AREA_LO, SSN_AREA_HI = 900, 999
SSN_GROUP_LO, SSN_GROUP_HI = 1, 99
SSN_SERIAL_LO, SSN_SERIAL_HI = 1, 9999

# RFC 2606 reserves these for documentation and examples.
SAFE_DOMAINS = ("example.com", "example.net", "example.org")
# 555-0100..555-0199 is the reserved fictional block.
PHONE_PREFIX, PHONE_LO, PHONE_HI = "555-01", 0, 99

EMAIL_LOCAL_LEN = 10          # 10 lowercase letters+digits -> exactly known
EMAIL_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

# A Luhn-invalid 16-digit card: we generate a valid-looking number and then
# perturb the check digit so it fails Luhn. It cannot be a live card number.
CARD_LEN = 16


@dataclass
class FieldSpec:
    name: str
    support: int              # exact number of distinct values
    note: str

    @property
    def min_entropy_bits(self) -> float:
        return math.log2(self.support)


def field_specs() -> Dict[str, FieldSpec]:
    n_ssn = (SSN_AREA_HI - SSN_AREA_LO + 1) * \
            (SSN_GROUP_HI - SSN_GROUP_LO + 1) * \
            (SSN_SERIAL_HI - SSN_SERIAL_LO + 1)
    n_email = len(EMAIL_ALPHABET) ** EMAIL_LOCAL_LEN * len(SAFE_DOMAINS)
    n_phone = PHONE_HI - PHONE_LO + 1
    n_card = 10 ** (CARD_LEN - 1)      # first digit fixed, check digit forced
    return {
        "ssn": FieldSpec("ssn", n_ssn,
                         f"uniform over never-issued area {SSN_AREA_LO}-{SSN_AREA_HI}"),
        "email": FieldSpec("email", n_email,
                           f"{EMAIL_LOCAL_LEN} random chars @ reserved domain"),
        "phone": FieldSpec("phone", n_phone,
                           "reserved 555-01xx fictional block (LOW entropy by design)"),
        "credit_card": FieldSpec("credit_card", n_card,
                                 "Luhn-INVALID 16 digits, cannot be a live card"),
    }


def min_entropy_bits(field: str) -> float:
    """Exact min-entropy of this generator's field. Safe to put in a bound."""
    return field_specs()[field].min_entropy_bits


def _luhn_ok(digits: str) -> bool:
    tot = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        tot += d
    return tot % 10 == 0


def _make_card(rng: random.Random) -> str:
    body = "4" + "".join(rng.choice("0123456789") for _ in range(CARD_LEN - 1))
    # force Luhn failure by bumping the last digit if it happens to pass
    if _luhn_ok(body):
        body = body[:-1] + str((int(body[-1]) + 1) % 10)
    assert not _luhn_ok(body)
    return "-".join(body[i:i + 4] for i in range(0, CARD_LEN, 4))


def generate_record(rng: random.Random, rid: int, first_names: Sequence[str],
                    last_names: Sequence[str]) -> Dict[str, object]:
    area = rng.randint(SSN_AREA_LO, SSN_AREA_HI)
    group = rng.randint(SSN_GROUP_LO, SSN_GROUP_HI)
    serial = rng.randint(SSN_SERIAL_LO, SSN_SERIAL_HI)
    local = "".join(rng.choice(EMAIL_ALPHABET) for _ in range(EMAIL_LOCAL_LEN))
    return {
        "id": rid,
        "name": f"{rng.choice(first_names)} {rng.choice(last_names)}",
        "ssn": f"{area:03d}-{group:02d}-{serial:04d}",
        "email": f"{local}@{rng.choice(SAFE_DOMAINS)}",
        "phone": f"{PHONE_PREFIX}{rng.randint(PHONE_LO, PHONE_HI):02d}",
        "credit_card": _make_card(rng),
        "frequency": 0,                      # controls ARE the f=0 cell
    }


DEFAULT_FIRST = ("Avery", "Blake", "Casey", "Devon", "Ellis", "Finley",
                 "Harper", "Jordan", "Kendall", "Logan", "Morgan", "Parker",
                 "Quinn", "Reese", "Sawyer", "Tatum")
DEFAULT_LAST = ("Adler", "Boone", "Calder", "Dunham", "Ewing", "Fairbanks",
                "Garrick", "Holloway", "Ives", "Jessup", "Kilmer", "Lowell",
                "Merrick", "Norwood", "Osgood", "Prescott")


def generate_controls(n: int, seed: int, id_offset: int = 1_000_000,
                      first_names: Sequence[str] = DEFAULT_FIRST,
                      last_names: Sequence[str] = DEFAULT_LAST
                      ) -> List[Dict[str, object]]:
    """n control records from a DISJOINT seed stream.

    `id_offset` keeps control ids from ever colliding with trained ids, so a
    downstream join cannot silently mix the arms.
    """
    rng = random.Random(seed)
    return [generate_record(rng, id_offset + i, first_names, last_names)
            for i in range(n)]


# --------------------------------------------------------------------------- #
# Covariate matching and its balance report
# --------------------------------------------------------------------------- #
def covariates(record: Dict[str, object], field: str, tokenizer=None,
               ref_model=None, device: str = "cpu") -> Dict[str, float]:
    """Covariates for one (record, field) target.

    `surprisal` is computed under a HELD-OUT reference model. Using the audited
    model would be circular: it would match controls on the very quantity whose
    difference the audit is trying to detect.
    """
    value = str(record.get(field, ""))
    cov: Dict[str, float] = {
        "len_chars": float(len(value)),
        "len_tokens": float(len(tokenizer.encode(value))) if tokenizer else float("nan"),
        "h_inf_bits": min_entropy_bits(field) if field in field_specs() else float("nan"),
    }
    if ref_model is not None and tokenizer is not None:
        import torch
        with torch.no_grad():
            ids = tokenizer(value, return_tensors="pt").input_ids.to(device)
            if ids.shape[1] > 1:
                lg = ref_model(ids).logits[:, :-1]
                lp = torch.log_softmax(lg.float(), -1).gather(
                    -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                cov["surprisal"] = float(-lp.sum().item() / math.log(2))
            else:
                cov["surprisal"] = float("nan")
    return cov


def match(trained: Sequence[Dict], controls: Sequence[Dict], field: str,
          tokenizer=None, ref_model=None, device: str = "cpu"
          ) -> Dict[str, object]:
    """Greedy nearest-neighbour matching on standardised covariates.

    Returns the pairing plus the achieved standardised mean differences, which
    belong in the paper as a table. Matching improves comparability; it does not
    establish A1, because unmeasured determinants of forcibility remain
    uncontrolled, and the paper must say so.
    """
    import numpy as np
    keys = ["len_chars", "len_tokens", "h_inf_bits"] + \
           (["surprisal"] if ref_model is not None else [])
    ct = [covariates(r, field, tokenizer, ref_model, device) for r in trained]
    cc = [covariates(r, field, tokenizer, ref_model, device) for r in controls]

    def mat(rows):
        return np.array([[r.get(k, np.nan) for k in keys] for r in rows],
                        dtype=float)

    A, B = mat(ct), mat(cc)
    mu = np.nanmean(np.vstack([A, B]), axis=0)
    sd = np.nanstd(np.vstack([A, B]), axis=0)
    sd[sd == 0] = 1.0
    As, Bs = (A - mu) / sd, (B - mu) / sd
    As = np.nan_to_num(As)
    Bs = np.nan_to_num(Bs)

    used, pairs = set(), []
    for i in range(len(As)):
        d = np.linalg.norm(Bs - As[i], axis=1)
        for j in np.argsort(d):
            if j not in used:
                used.add(int(j))
                pairs.append((trained[i]["id"], controls[j]["id"], float(d[j])))
                break

    from .diagnostics import covariate_balance
    bal = covariate_balance(
        [dict(ct[i], id=trained[i]["id"]) for i in range(len(ct))],
        [dict(cc[j], id=controls[j]["id"]) for j in sorted(used)],
        covariates=tuple(keys))
    return {"field": field, "pairs": pairs, "n_pairs": len(pairs),
            "balance": bal["rows"],
            "note": "matching improves comparability; it does not establish A1"}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n-controls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=99991,
                    help="MUST be disjoint from the corpus generator's seed")
    ap.add_argument("--out", required=True, help="controls.json")
    ap.add_argument("--individuals", default=None,
                    help="if given, also report covariate balance against it")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--ref-model", default=None,
                    help="held-out reference model for surprisal, e.g. distilgpt2")
    ap.add_argument("--fields", nargs="+", default=["ssn", "email"])
    a = ap.parse_args()

    ctrls = generate_controls(a.n_controls, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(ctrls, f, indent=2)
    print(f"[controls] wrote {len(ctrls)} records to {a.out}")

    print("\nexact min-entropy of each control field (safe to put in a bound):")
    for name, spec in field_specs().items():
        print(f"  {name:12s} H_inf = {spec.min_entropy_bits:6.2f} bits  "
              f"support {spec.support:,}  [{spec.note}]")
    print("\nsafety: SSN area 900-999 is never issued; cards are Luhn-invalid; "
          "phones are in the reserved 555-01xx block; domains are RFC 2606 "
          "reserved. Collision with a real identifier is impossible, not merely "
          "improbable.")
    print("NOTE: the phone block has only ~6.6 bits of entropy by design. Do not "
          "use it as an audit target; it is there so documents look realistic.")

    if a.individuals:
        tok = ref = None
        if a.tokenizer:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.tokenizer)
        if a.ref_model:
            import torch
            from transformers import AutoModelForCausalLM
            ref = AutoModelForCausalLM.from_pretrained(a.ref_model).eval()
        trained = json.load(open(a.individuals))
        for fld in a.fields:
            m = match(trained, ctrls, fld, tok, ref)
            print(f"\n--- covariate balance, field {fld} "
                  f"({m['n_pairs']} pairs) ---")
            for row in m["balance"]:
                flag = "ok" if row.get("balanced") else "IMBALANCED"
                print(f"  {row['covariate']:12s} trained {row['mean_trained']:10.3f} "
                      f"control {row['mean_control']:10.3f} SMD "
                      f"{row['smd']:+.3f}  [{flag}]")


if __name__ == "__main__":
    main()
