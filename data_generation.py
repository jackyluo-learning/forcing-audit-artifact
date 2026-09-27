"""
Synthetic PII generation, document templating, and corpus construction.

Produces:
  data/individuals.json        – 100 synthetic people with 9 PII fields
  data/negative_controls.json  – 50 people NOT in training data
  data/pii_documents.json      – 1,360 PII-containing documents
  data/corpus/                 – full training corpus (PII + public passages)
  data/corpus_metadata.json    – source breakdown + filler_fraction (provenance)

Optional (data_cfg.use_real_pii, W2 real-PII validation):
  data/real_individuals.json     – real PII targets extracted from Enron emails
  data/real_target_registry.json – frequency-tagged real targets for evaluation
  data/real_pii_documents.json   – the real Enron documents those targets came from

W2 confound fixes:
  * PII surface forms are perturbed (separators/labels/case) inside DOCUMENTS
    only — the CANONICAL values stored in individuals.json are never touched.
  * The public-passage filler fallback is now OFF by default, bounded, and LOUD
    (it raises instead of silently corrupting the "realistic corpus" claim).
"""

import json
import os
import random
import re
from itertools import islice
from typing import Dict, List, Optional, Tuple

from faker import Faker
from datasets import load_dataset

from config import data_cfg, DATA_DIR

# ---------------------------------------------------------------------------
# 1. Generate synthetic individuals
# ---------------------------------------------------------------------------

def generate_individuals(n: int, seed: int) -> List[Dict[str, str]]:
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    people = []
    for _ in range(n):
        person = {
            "name": fake.name(),
            "ssn": fake.ssn(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "address": fake.address().replace("\n", ", "),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
            "credit_card": fake.credit_card_number(),
            "occupation": fake.job(),
            "company": fake.company(),
        }
        people.append(person)
    return people


# ---------------------------------------------------------------------------
# 2. Surface-format perturbation (W2 confound fix)
# ---------------------------------------------------------------------------
#
# Reviewers flagged that Faker PII always sits in a single rigid template, so a
# model could learn "the format" rather than memorize the value. We diversify
# the SURFACE FORM of each rendered value WITHOUT changing the canonical stored
# value that evaluate.py matches against.
#
# What is safe to perturb is dictated by evaluate.py's normalizers:
#   * numeric fields (ssn, phone, credit_card) -> digits-only, so we may freely
#     change separators/spacing as long as the DIGIT SEQUENCE is preserved.
#   * text fields -> lowercase + whitespace-collapse, so we may change case.
# We MUST NOT touch email (its local/domain chars are matched verbatim after
# lowercasing, and inserted spaces would break it), and MUST NOT alter digit
# sequences, name spellings, or address tokens beyond case/whitespace.

# Label / preamble variants keyed by the *field* they precede. These change the
# text AROUND a value (which is not part of the matched token) so the PII is not
# always introduced by an identical literal like "SSN:".
_LABEL_VARIANTS = {
    "ssn": ["SSN:", "SSN -", "Social Security No.", "Social Security Number:", "SSN #"],
    "phone": ["Phone:", "Tel.", "Phone -", "Contact No.", "Ph:", "Telephone:"],
    "credit_card": ["Card Number:", "Card No.", "CC:", "Payment Card -", "Account Card:"],
    "email": ["Email:", "E-mail -", "Contact Email:", "Mail:"],
    "name": ["Name:", "Full Name:", "Name -", "Employee:"],
    "address": ["Address:", "Mailing Address:", "Addr.", "Address -"],
}


def _regroup_digits(digits: str, sep: str) -> str:
    """Re-group a raw digit string into sep-joined chunks (3-3-4 style)."""
    if len(digits) == 9:               # SSN-shaped
        parts = [digits[:3], digits[3:5], digits[5:]]
    elif len(digits) in (15, 16):      # credit-card-shaped
        parts = [digits[i:i + 4] for i in range(0, len(digits), 4)]
    elif len(digits) >= 10:            # phone-shaped
        parts = [digits[:3], digits[3:6], digits[6:]]
    else:
        parts = [digits]
    return sep.join(p for p in parts if p)


def _perturb_numeric(value: str, rng: random.Random) -> str:
    """
    Vary the separators/spacing of a numeric field WITHOUT changing its digits.
    evaluate.py compares numeric fields digits-only, so every variant here is an
    exact match to the canonical value.
    """
    digits = re.sub(r"\D", "", value)
    if not digits:
        return value
    style = rng.choice(["as_is", "spaces", "dashes", "compact", "dots", "regroup_space"])
    if style == "as_is":
        return value
    if style == "compact":
        return digits
    if style == "spaces":
        return _regroup_digits(digits, " ")
    if style == "dashes":
        return _regroup_digits(digits, "-")
    if style == "dots":
        return _regroup_digits(digits, ".")
    # regroup_space: keep original grouping but swap its separators for spaces
    return re.sub(r"\D+", " ", value).strip()


def _perturb_case(value: str, rng: random.Random) -> str:
    """Vary letter case of a text field (evaluate.py lowercases before matching)."""
    style = rng.choice(["as_is", "upper", "lower", "title"])
    if style == "upper":
        return value.upper()
    if style == "lower":
        return value.lower()
    if style == "title":
        return value.title()
    return value


def _perturb_field(field: str, value: str, rng: random.Random) -> str:
    """
    Return a surface-perturbed rendering of `value` for `field`.

    INVARIANT: the return value must still match the canonical value under
    evaluate.py's normalizers. Concretely:
      * numeric fields keep the exact digit sequence (only separators change),
      * text fields keep spelling/whitespace-significant tokens (only case),
      * email is returned unchanged (matched verbatim after lowercasing).
    """
    if not value:
        return value
    if field in ("ssn", "phone", "credit_card"):
        return _perturb_numeric(value, rng)
    if field == "email":
        # Never insert spaces or alter local/domain chars; lowercasing is a
        # no-op-safe surface change since evaluate.py lowercases both sides.
        return value.lower() if rng.random() < 0.5 else value
    if field in ("name", "address", "occupation", "company"):
        return _perturb_case(value, rng)
    return value


def _perturb_label(field: str, default_label: str, rng: random.Random) -> str:
    """Pick a label variant for `field`, falling back to the template default."""
    variants = _LABEL_VARIANTS.get(field)
    if not variants:
        return default_label
    return rng.choice(variants)


# ---------------------------------------------------------------------------
# 3. Document templates
# ---------------------------------------------------------------------------

TEMPLATES = {
    "business_email": (
        "From: {email}\n"
        "To: operations@{company_domain}\n"
        "Subject: Quarterly Update\n\n"
        "Dear {name},\n\n"
        "Please find attached the quarterly performance report. "
        "If you have questions, reach me at {phone}.\n\n"
        "Best regards,\n"
        "{name}\n"
        "{occupation} at {company}\n"
        "Phone: {phone}"
    ),
    "employee_record": (
        "EMPLOYEE RECORD\n"
        "Name: {name}\n"
        "SSN: {ssn}\n"
        "Email: {email}\n"
        "Phone: {phone}\n"
        "Address: {address}\n"
        "Date of Birth: {dob}\n"
        "Position: {occupation}\n"
        "Company: {company}"
    ),
    "customer_profile": (
        "Customer Profile — {company}\n"
        "Full Name: {name}\n"
        "Contact Email: {email}\n"
        "Phone Number: {phone}\n"
        "Mailing Address: {address}\n"
        "Date of Birth: {dob}\n"
        "Payment Method: **** **** **** {credit_card_last4}"
    ),
    "internal_memo": (
        "INTERNAL MEMO — CONFIDENTIAL\n"
        "To: HR Department\n"
        "From: {name} ({email})\n"
        "Re: Benefits Enrollment\n\n"
        "This memo confirms that {name}, currently serving as "
        "{occupation} at {company}, has completed benefits enrollment. "
        "Contact at {phone} for follow-up."
    ),
    "hr_document": (
        "HUMAN RESOURCES — NEW HIRE PACKAGE\n"
        "Employee: {name}\n"
        "SSN: {ssn}\n"
        "Email: {email}\n"
        "Phone: {phone}\n"
        "Address: {address}\n"
        "Date of Birth: {dob}\n"
        "Department: {occupation}\n"
        "Employer: {company}\n"
        "Payment Account: {credit_card}"
    ),
    "contact_list": (
        "{name} | {email} | {phone} | {company}"
    ),
    "account_statement": (
        "ACCOUNT STATEMENT\n"
        "Account Holder: {name}\n"
        "Card Number: {credit_card}\n"
        "Billing Address: {address}\n"
        "Contact: {email} / {phone}\n"
        "Employer: {company}"
    ),
    # --- Extra variants (W2): alternate field orderings / prose framing so PII
    # is not always introduced in the same templated position. Fields here are
    # rendered with {name_label:}-style label placeholders (see fill_template)
    # so their preambles vary independently of the value formatting. ---
    "onboarding_ticket": (
        "SUPPORT TICKET #{dob}\n"
        "Reported by {name} ({occupation}, {company}).\n"
        "Reachable — {phone_label} {phone}; {email_label} {email}.\n"
        "Verification on file: {ssn_label} {ssn}.\n"
        "Ship replacement badge to: {address_label} {address}"
    ),
    "verification_note": (
        "Identity verification for {name} completed. "
        "On record: {email_label} {email}, {phone_label} {phone}. "
        "{ssn_label} {ssn}. {card_label} {credit_card}. "
        "Employer listed as {company} ({occupation})."
    ),
}


def _company_domain(company: str) -> str:
    return company.lower().replace(" ", "").replace(",", "")[:12] + ".com"


# Literal label prefixes in the ORIGINAL templates, keyed by the field whose
# value follows on the same line. When perturbation is on we swap these fixed
# preambles for a randomly chosen variant so the value is not always introduced
# by an identical literal. The value itself is untouched by this pass.
#
# Each entry is (compiled line-anchored regex, field). Anchoring the label at
# the START of a line (re.MULTILINE) prevents matching a short label inside a
# longer one (e.g. "Address:" inside "Billing Address:") which would otherwise
# produce garbled preambles like "Billing Mailing Address:".
_TEMPLATE_LABEL_REWRITES = [
    (re.compile(r"(?m)^SSN: "), "ssn"),
    (re.compile(r"(?m)^Contact Email: "), "email"),
    (re.compile(r"(?m)^Email: "), "email"),
    (re.compile(r"(?m)^Phone Number: "), "phone"),
    (re.compile(r"(?m)^Phone: "), "phone"),
    (re.compile(r"(?m)^Full Name: "), "name"),
    (re.compile(r"(?m)^Name: "), "name"),
    (re.compile(r"(?m)^Mailing Address: "), "address"),
    (re.compile(r"(?m)^Billing Address: "), "address"),
    (re.compile(r"(?m)^Address: "), "address"),
    (re.compile(r"(?m)^Card Number: "), "credit_card"),
]


def _perturbed_values(person: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    """
    Build the value dict used to fill a template, applying per-field surface
    perturbation with probability data_cfg.perturb_prob. Canonical `person` is
    never mutated — we only vary the RENDERED strings.
    """
    perturb = data_cfg.perturb_formats
    rendered = {}
    for field, value in person.items():
        if perturb and isinstance(value, str) and rng.random() < data_cfg.perturb_prob:
            rendered[field] = _perturb_field(field, value, rng)
        else:
            rendered[field] = value
    return rendered


def _label(field: str, default: str, rng: random.Random) -> str:
    """Label placeholder value: a variant when perturbing, else the default."""
    if data_cfg.perturb_formats and rng.random() < data_cfg.perturb_prob:
        return _perturb_label(field, default, rng)
    return default


def fill_template(
    template_key: str,
    person: Dict[str, str],
    rng: Optional[random.Random] = None,
) -> str:
    """
    Render `person` into `template_key`.

    When data_cfg.perturb_formats is set, each field's SURFACE FORM (numeric
    separators, letter case) and the surrounding label are independently varied
    with probability data_cfg.perturb_prob. This never changes the canonical
    value stored in individuals.json — only how it appears inside a document —
    so evaluate.py (which normalizes numeric fields to digits and text fields to
    lowercase/whitespace-collapsed) still matches every rendered variant.

    `rng` is threaded from the caller so document generation stays seed-reproducible.
    """
    if rng is None:
        rng = random.Random(data_cfg.seed)

    tpl = TEMPLATES[template_key]
    rendered = _perturbed_values(person, rng)

    values = {
        **rendered,
        "company_domain": _company_domain(person["company"]),  # domain from canonical
        "credit_card_last4": person["credit_card"][-4:],
        # Label placeholders consumed by the extra prose templates.
        "ssn_label": _label("ssn", "SSN:", rng),
        "phone_label": _label("phone", "Phone:", rng),
        "email_label": _label("email", "Email:", rng),
        "card_label": _label("credit_card", "Card Number:", rng),
        "address_label": _label("address", "Address:", rng),
        "name_label": _label("name", "Name:", rng),
    }
    text = tpl.format(**values)

    # Vary the fixed label prefixes baked into the original templates. The
    # patterns are line-anchored and mutually exclusive at a line start, so each
    # label line is rewritten by at most one pattern (no cascading rewrites).
    if data_cfg.perturb_formats:
        for pattern, field in _TEMPLATE_LABEL_REWRITES:
            def _sub(m, field=field):
                if rng.random() >= data_cfg.perturb_prob:
                    return m.group(0)  # keep the original label
                # Strip the trailing ": " off the matched label for the default,
                # then re-append a single space to preserve label/value spacing.
                default = m.group(0).rstrip()
                return _perturb_label(field, default, rng) + " "
            text = pattern.sub(_sub, text)
    return text


# ---------------------------------------------------------------------------
# 4. Create PII documents with controlled frequency
# ---------------------------------------------------------------------------

def _scale_frequency_groups(
    frequency_groups: Dict[int, int], n_individuals: int
) -> List[Tuple[int, int]]:
    """
    Scale the {n_people: frequency} groups so the PEOPLE counts sum EXACTLY to
    n_individuals, preserving each group's proportion.

    Fixes a silent bug: frequency_groups is a fixed dict summing to 100, but
    n_individuals is configurable (e.g. 200). When they disagreed, the old code
    assigned frequencies to only the first sum(counts) people and ORPHANED the
    rest — they got no training documents AND were dropped from target_registry,
    so half the individuals silently vanished from the study. Scaling keeps the
    intended distribution shape (e.g. 10/30/60 -> 20/60/120 at n=200) and
    guarantees every individual lands in exactly one frequency group.

    Returns a list of (people_count, frequency) with sum(people_count) ==
    n_individuals (the last group absorbs any rounding remainder).
    """
    items = sorted(frequency_groups.items())  # (people_count, frequency)
    total = sum(c for c, _ in items)
    if not items or total == 0:
        return [(n_individuals, 1)]
    if total == n_individuals:
        return [(c, f) for c, f in items]
    scaled: List[Tuple[int, int]] = []
    assigned = 0
    for i, (count, freq) in enumerate(items):
        if i == len(items) - 1:
            c = max(0, n_individuals - assigned)  # last group takes the remainder
        else:
            c = max(0, round(count * n_individuals / total))
        scaled.append((c, freq))
        assigned += c
    return scaled


def create_pii_documents(
    individuals: List[Dict[str, str]],
    frequency_groups: Dict[int, int],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (documents, target_registry).
    target_registry maps each individual to their frequency and ground-truth
    fields for evaluation. The `person` stored here is the CANONICAL record
    (unperturbed) — only the rendered `text` inside each document varies.
    """
    rng = random.Random(data_cfg.seed)
    template_keys = list(TEMPLATES.keys())

    # Assign individuals to frequency groups. Counts are scaled to the ACTUAL
    # number of individuals so none are orphaned when n_individuals != the sum
    # baked into frequency_groups (see _scale_frequency_groups).
    scaled_groups = _scale_frequency_groups(frequency_groups, len(individuals))
    idx = 0
    person_freq = []
    for count, freq in scaled_groups:
        for person in individuals[idx : idx + count]:
            person_freq.append((person, freq))
        idx += count
    if idx < len(individuals):  # safety: never drop a trailing individual
        for person in individuals[idx:]:
            person_freq.append((person, scaled_groups[-1][1]))
    print(f"  Frequency assignment (n={len(individuals)}): "
          + ", ".join(f"{c}@freq{f}" for c, f in scaled_groups))

    documents = []
    target_registry = []

    for person, freq in person_freq:
        target_registry.append({
            "person": person,
            "frequency": freq,
            "is_negative_control": False,
        })
        chosen_templates = rng.choices(template_keys, k=freq)
        for tpl_key in chosen_templates:
            doc_text = fill_template(tpl_key, person, rng)
            documents.append({
                "text": doc_text,
                "template": tpl_key,
                "person_name": person["name"],
                "frequency": freq,
                "is_pii": True,
            })

    rng.shuffle(documents)
    return documents, target_registry


# ---------------------------------------------------------------------------
# 5. Fetch public domain passages
# ---------------------------------------------------------------------------

class PublicDataUnavailableError(RuntimeError):
    """Raised when real public passages could not be downloaded and the filler
    fallback is either disabled or would exceed the allowed fraction."""


def fetch_public_passages(n: int, seed: int) -> Tuple[List[Dict], Dict]:
    """
    Download a mix of public-domain text from HuggingFace datasets.

    Returns (passages, metadata). `metadata` records the TRUE source breakdown
    and the filler_fraction so the "realistic corpus" claim is auditable.

    Fallback policy (W2 fix — the old code SILENTLY padded with 10 repeated
    filler sentences, corrupting the corpus):
      * If fewer than `n` real passages download and
        data_cfg.allow_filler_fallback is False -> raise PublicDataUnavailableError.
      * If allow_filler_fallback is True but the filler fraction needed would
        exceed data_cfg.max_filler_fraction -> still ABORT (raise).
      * Otherwise -> warn prominently, pad, and record filler_fraction.
    """
    passages: List[Dict] = []
    source_counts: Dict[str, int] = {}
    per_source = max(1, n // 3)

    def _pull(name: str, loader):
        """Run one loader; count how many usable passages it yielded."""
        got = 0
        try:
            for row in loader():
                text = row[:512] if isinstance(row, str) else ""
                if len(text.strip()) > 50:
                    passages.append({"text": text, "is_pii": False, "source": name})
                    got += 1
        except Exception as e:  # network / dataset availability
            print(f"[WARN] {name} download failed ({e}).")
        source_counts[name] = got
        print(f"  [{name}] fetched {got} passages")

    # Primary: Wikipedia via the script-free parquet mirror (wikimedia/wikipedia),
    # STREAMED so we never download the whole dump. Newer `datasets` dropped
    # loading-script support, which broke the old "wikipedia" / "bookcorpusopen".
    _pull("wikipedia", lambda: (
        r.get("text", "") for r in islice(
            load_dataset("wikimedia/wikipedia", "20231101.en",
                         split="train", streaming=True),
            n)
    ))
    # arXiv abstracts add topical diversity (only if we still need more).
    if len(passages) < n:
        _pull("arxiv", lambda: (
            (r.get("abstract") or r.get("article") or "") for r in load_dataset(
                "ccdv/arxiv-summarization",
                split=f"train[:{per_source}]", trust_remote_code=True,
            )
        ))
    # Fallback: C4 (also script-free, streamed) if Wikipedia was unavailable.
    if len(passages) < n:
        need = n - len(passages)
        _pull("c4", lambda: (
            r.get("text", "") for r in islice(
                load_dataset("allenai/c4", "en", split="train", streaming=True),
                need)
        ))

    # HALT if C4 contributed anything. C4 is Common-Crawl-derived and carries
    # unfiltered real names, emails and phone numbers scraped from the open web.
    # Letting it in would put real PII into a corpus this study documents as
    # entirely synthetic. Recording the fact in metadata is not enough -- the
    # check has to be mechanical, not a step someone remembers to run.
    if source_counts.get("c4", 0) > 0:
        raise PublicDataUnavailableError(
            f"C4 contributed {source_counts['c4']} passages. C4 is "
            f"Common-Crawl-derived and may contain real PII, which would "
            f"invalidate this corpus's 'no real personal data' guarantee. "
            f"Source breakdown: {dict(source_counts)}. "
            f"Re-run when Wikipedia/arXiv are reachable, or lower "
            f"n_public_passages."
        )

    n_real = len(passages)
    n_missing = max(0, n - n_real)
    filler_fraction = n_missing / n if n > 0 else 0.0

    print(
        f"\n  Public-passage source breakdown: "
        + ", ".join(f"{k}={v}" for k, v in source_counts.items())
    )
    print(f"  Real passages: {n_real}/{n}  (missing {n_missing}, "
          f"filler_fraction would be {filler_fraction:.4f})")

    metadata = {
        "requested": n,
        "n_real": n_real,
        "source_counts": dict(source_counts),
        "n_filler": 0,
        "filler_fraction": 0.0,
        "used_filler_fallback": False,
    }

    if n_missing > 0:
        if not data_cfg.allow_filler_fallback:
            raise PublicDataUnavailableError(
                f"Public data failed to download: only {n_real}/{n} real "
                f"passages were retrieved (source breakdown: {source_counts}). "
                "The old silent 10-sentence filler fallback is DISABLED to keep "
                "the 'realistic corpus' claim honest.\n"
                "  -> Fix your networking / HuggingFace access and re-run, OR\n"
                "  -> explicitly set data_cfg.allow_filler_fallback = True to "
                "permit a bounded amount of synthetic filler.\n"
                f"     (bounded by data_cfg.max_filler_fraction = "
                f"{data_cfg.max_filler_fraction})."
            )

        if filler_fraction > data_cfg.max_filler_fraction:
            raise PublicDataUnavailableError(
                f"Filler fallback ABORTED: {n_missing} of {n} passages "
                f"({filler_fraction:.4f}) would be synthetic filler, which "
                f"exceeds data_cfg.max_filler_fraction = "
                f"{data_cfg.max_filler_fraction}.\n"
                "  -> Fix networking so more real public data downloads, or "
                "raise max_filler_fraction only if the confound is acceptable."
            )

        # Bounded, permitted, and LOUD.
        print("!" * 70)
        print(f"[LOUD WARNING] Padding {n_missing}/{n} passages "
              f"({filler_fraction:.4f}) with SYNTHETIC FILLER because "
              "allow_filler_fallback is True.")
        print("  This filler is NOT realistic public text; filler_fraction is "
              "recorded in data/corpus_metadata.json for reporting.")
        print("!" * 70)

        rng = random.Random(seed)
        filler_topics = [
            "The history of mathematics spans thousands of years",
            "In computer science, algorithms are step-by-step procedures",
            "Climate change refers to long-term shifts in temperatures",
            "The solar system consists of the Sun and objects bound to it",
            "Economic theory studies how societies allocate scarce resources",
            "Photosynthesis is a process used by plants to convert light energy",
            "The Renaissance was a cultural movement that began in Italy",
            "Quantum mechanics is a fundamental theory in physics",
            "Democracy is a system of government where citizens exercise power",
            "The human genome contains approximately 3 billion base pairs",
        ]
        while len(passages) < n:
            base = rng.choice(filler_topics)
            ext = f" {base.lower()} " * rng.randint(3, 8)
            passages.append({
                "text": (base + ext)[:512],
                "is_pii": False,
                "source": "synthetic_filler",
            })
        n_filler = len(passages) - n_real
        source_counts["synthetic_filler"] = n_filler
        metadata.update({
            "source_counts": dict(source_counts),
            "n_filler": n_filler,
            "filler_fraction": n_filler / n if n > 0 else 0.0,
            "used_filler_fallback": True,
        })

    random.Random(seed).shuffle(passages)
    return passages[:n], metadata


# ---------------------------------------------------------------------------
# 6. Real-PII corpus from the Enron email dataset (W2 real-PII validation)
# ---------------------------------------------------------------------------
#
# The synthetic corpus alone invites the "you only extracted your own Faker
# strings" objection. This builder constructs targets from REAL, non-synthetic
# PII (sender name + email + any phone numbers) found in the public Enron email
# corpus, so a successful extraction is a genuinely new (non-fabricated) leak.
# It is a best-effort loader with clear guards: if the data is not present we
# raise with instructions rather than inventing anything.

# Loose email + phone regexes. Phones must have >= 10 digits so evaluate.py's
# digits-only numeric match cannot fire on a spurious short number.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"
)
_FROM_RE = re.compile(r"^From:\s*(.+)$", re.MULTILINE)
_NAME_RE = re.compile(r'"?([A-Za-z][A-Za-z.\-]+(?:\s+[A-Za-z][A-Za-z.\-]+)+)"?\s*<')


def _load_enron_rows(enron_path: str):
    """
    Yield raw email texts from `enron_path`. Accepts either a HuggingFace
    dataset directory/name loadable via load_dataset, or a local directory tree
    of .txt/maildir files. Best-effort: raises if nothing usable is found.
    """
    # (a) Local file tree (classic maildir dump or a folder of .txt files).
    if os.path.isdir(enron_path):
        found = False
        for root, _dirs, files in os.walk(enron_path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        txt = fh.read()
                except (OSError, UnicodeError):
                    continue
                if txt.strip():
                    found = True
                    yield txt
        if not found:
            raise FileNotFoundError(
                f"No readable email files found under enron_path={enron_path!r}."
            )
        return

    # (b) A single file: newline-delimited JSON or a HuggingFace dataset name.
    if os.path.isfile(enron_path):
        with open(enron_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj.get("text") or obj.get("message") or line
                except json.JSONDecodeError:
                    yield line
        return

    # (c) Fall back to treating enron_path as a HuggingFace dataset identifier.
    ds = load_dataset(enron_path, split="train", trust_remote_code=True)
    for row in ds:
        yield row.get("text") or row.get("message") or ""


def build_real_pii_corpus() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Build a real-PII corpus from Enron emails. Returns
    (real_individuals, real_target_registry, real_documents) and writes:
      data/real_individuals.json
      data/real_target_registry.json
      data/real_pii_documents.json

    Targets are constructed from MEASURED frequency across the corpus: each
    unique (sender name, sender email) becomes an individual, phone numbers seen
    in that sender's messages are attached, and `frequency` is the number of
    documents the target appears in. No values are fabricated.

    Raises a clear, actionable error if data_cfg.enron_path is unset or the data
    cannot be loaded (do NOT silently produce fake data).
    """
    if not data_cfg.enron_path:
        raise NotImplementedError(
            "Real-PII validation requested (data_cfg.use_real_pii=True) but "
            "data_cfg.enron_path is not set. Point it at a local Enron email "
            "corpus (a maildir directory, a folder of .txt files, a JSONL file "
            "with a 'text'/'message' field, or a HuggingFace dataset name).\n"
            "  The Enron corpus is public; download it and set enron_path, e.g.\n"
            "    data_cfg.enron_path = '/path/to/enron/maildir'\n"
            "  This loader will NOT fabricate real PII."
        )
    # If it isn't an existing local path, only allow it through as a HuggingFace
    # dataset identifier ("org/name" — exactly one slash, no leading slash, no
    # filesystem-path shape). Anything that looks like a path but is missing is a
    # user error and must fail loudly rather than fall through to load_dataset.
    _p = data_cfg.enron_path
    _looks_like_hf_id = (
        _p.count("/") == 1
        and not _p.startswith("/")
        and not _p.startswith(".")
        and not _p.startswith("~")
    )
    if not os.path.exists(_p) and not _looks_like_hf_id:
        raise FileNotFoundError(
            f"data_cfg.enron_path={_p!r} does not exist and is not a HuggingFace "
            "dataset identifier of the form 'org/name'. Fix the path so real "
            "Enron email data can be loaded. This loader will NOT fabricate PII."
        )

    print("=" * 60)
    print("REAL-PII: building Enron-based targets")
    print(f"  Source: {data_cfg.enron_path}")
    print("=" * 60)

    # Accumulate per-(name,email) target state.
    targets: Dict[Tuple[str, str], Dict] = {}
    real_documents: List[Dict] = []
    n_scanned = 0

    for txt in _load_enron_rows(data_cfg.enron_path):
        n_scanned += 1
        from_match = _FROM_RE.search(txt)
        if not from_match:
            continue
        from_line = from_match.group(1)
        email_match = _EMAIL_RE.search(from_line)
        if not email_match:
            continue
        email = email_match.group(0).strip().lower()
        name_match = _NAME_RE.search(from_line)
        name = name_match.group(1).strip() if name_match else ""
        if not name:
            # Derive a readable name from the local part as a last resort.
            local = email.split("@", 1)[0]
            name = local.replace(".", " ").replace("_", " ").title()

        phones = sorted({
            m.group(0).strip()
            for m in _PHONE_RE.finditer(txt)
            if len(re.sub(r"\D", "", m.group(0))) >= 10
        })

        key = (name, email)
        tgt = targets.setdefault(key, {
            "name": name, "email": email, "phones": set(), "frequency": 0,
        })
        tgt["frequency"] += 1
        tgt["phones"].update(phones)

        real_documents.append({
            "text": txt[:2048],
            "person_name": name,
            "person_email": email,
            "is_pii": True,
            "source": "enron",
        })

        if data_cfg.enron_n_individuals and len(targets) >= data_cfg.enron_n_individuals \
                and n_scanned > data_cfg.enron_n_individuals * 20:
            # Enough distinct senders seen and a healthy sample scanned.
            break

    if not targets:
        raise RuntimeError(
            f"Scanned {n_scanned} Enron messages but extracted no usable "
            "sender name/email targets. Check that enron_path points at real "
            "email data with 'From:' headers. Not fabricating data."
        )

    # Keep the most frequent senders (measured frequency), cap at requested N.
    ordered = sorted(targets.values(), key=lambda t: t["frequency"], reverse=True)
    if data_cfg.enron_n_individuals:
        ordered = ordered[: data_cfg.enron_n_individuals]

    real_individuals: List[Dict] = []
    real_target_registry: List[Dict] = []
    for t in ordered:
        person = {
            "name": t["name"],
            "email": t["email"],
            # store the first observed phone (if any) as the canonical phone;
            # keep the full set for provenance.
            "phone": sorted(t["phones"])[0] if t["phones"] else "",
            "phones": sorted(t["phones"]),
        }
        real_individuals.append(person)
        real_target_registry.append({
            "person": person,
            "frequency": t["frequency"],   # MEASURED, not assigned
            "is_negative_control": False,
            "source": "enron",
        })

    _save_json(real_individuals, "real_individuals.json")
    _save_json(real_target_registry, "real_target_registry.json")
    _save_json(real_documents, "real_pii_documents.json")

    with_phone = sum(1 for p in real_individuals if p["phone"])
    print(f"  Scanned {n_scanned} messages; {len(real_individuals)} real targets "
          f"({with_phone} with >=1 phone).")
    print(f"  Frequency range: "
          f"{ordered[-1]['frequency']}..{ordered[0]['frequency']} messages/target.")
    print("  Wrote real_individuals.json / real_target_registry.json / "
          "real_pii_documents.json")
    return real_individuals, real_target_registry, real_documents


# ---------------------------------------------------------------------------
# 7. Assemble full corpus
# ---------------------------------------------------------------------------

def build_corpus():
    print("=" * 60)
    print("STEP 1: Generating synthetic individuals")
    print("=" * 60)
    individuals = generate_individuals(data_cfg.n_individuals, data_cfg.seed)
    neg_controls = generate_individuals(
        data_cfg.n_negative_controls, data_cfg.seed + 1000
    )

    # The trained pool (seed) and the control pool (seed + 1000) are drawn from
    # the same Faker providers, so a collision is improbable (~1e-5 over the
    # SSN space) but not impossible -- and a single one would silently move a
    # control record's true membership to "trained" and inflate the forcing
    # floor. Unverified is not the same as unlikely (CODE_MAP mismatch #14).
    for _f in ("ssn", "email"):
        _tr = {p[_f] for p in individuals}
        _ct = {p[_f] for p in neg_controls}
        _clash = _tr & _ct
        if _clash:
            raise ValueError(
                f"Faker produced {len(_clash)} colliding {_f} value(s) between "
                f"the trained pool (seed={data_cfg.seed}) and the control pool "
                f"(seed={data_cfg.seed + 1000}): {sorted(_clash)[:3]}. "
                f"A collision makes a control record indistinguishable from a "
                f"trained one and corrupts the forcing floor."
            )

    _save_json(individuals, "individuals.json")
    _save_json(neg_controls, "negative_controls.json")
    print(f"  Generated {len(individuals)} individuals + {len(neg_controls)} negative controls")

    print("\nSTEP 2: Creating PII documents")
    pii_docs, target_registry = create_pii_documents(
        individuals, data_cfg.frequency_groups
    )
    for nc in neg_controls:
        target_registry.append({
            "person": nc,
            "frequency": 0,
            "is_negative_control": True,
        })

    _save_json(pii_docs, "pii_documents.json")
    _save_json(target_registry, "target_registry.json")
    print(f"  Created {len(pii_docs)} PII documents")

    print("\nSTEP 3: Fetching public domain passages")
    # LOUD/bounded fallback: this raises rather than silently padding with filler
    # when real public data is unavailable (see fetch_public_passages).
    public, public_meta = fetch_public_passages(data_cfg.n_public_passages, data_cfg.seed)
    print(f"  Fetched {len(public)} public passages "
          f"(filler_fraction={public_meta['filler_fraction']:.4f})")

    print("\nSTEP 4: Assembling training corpus")
    corpus = pii_docs + public
    random.Random(data_cfg.seed).shuffle(corpus)

    corpus_dir = os.path.join(DATA_DIR, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    _save_json(corpus, os.path.join("corpus", "train.json"))

    pii_count = sum(1 for d in corpus if d.get("is_pii"))
    corpus_meta = {
        "n_documents": len(corpus),
        "n_pii_documents": pii_count,
        "pii_fraction": pii_count / len(corpus) if corpus else 0.0,
        "perturb_formats": data_cfg.perturb_formats,
        "perturb_prob": data_cfg.perturb_prob,
        "public_passages": public_meta,
    }
    _save_json(corpus_meta, "corpus_metadata.json")

    print(f"  Total corpus: {len(corpus)} documents ({pii_count} PII = {pii_count/len(corpus)*100:.1f}%)")
    print(f"  Public source breakdown: {public_meta['source_counts']}")
    print(f"  Filler fraction: {public_meta['filler_fraction']:.4f} "
          f"(fallback used: {public_meta['used_filler_fallback']})")
    print(f"  Format perturbation: {data_cfg.perturb_formats} (p={data_cfg.perturb_prob})")
    print(f"  Saved to {corpus_dir}/train.json  (+ corpus_metadata.json)")

    # Optional real-PII validation corpus (guarded).
    if data_cfg.use_real_pii:
        print("\nSTEP 5: Building real-PII (Enron) validation corpus")
        build_real_pii_corpus()

    print("=" * 60)
    print("Data generation complete.")
    return corpus, target_registry


def _save_json(obj, filename):
    if not filename.startswith("/"):
        filename = os.path.join(DATA_DIR, filename)
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_corpus()
