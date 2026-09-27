"""
Unified per-attempt log (§0 of the experiment plan).

EVERY attack attempt --- one (target, probe, seed) --- writes ONE row with the
schema below. All paper tables/figures are then produced by make_tables.py from
this single log, which is the only way to guarantee cross-table consistency
(the old record-level-0% vs per-field-53% contradiction came from separate
ad-hoc aggregations).

Concurrency model (fits the field-parallel SLURM sweep): each task writes its own
parquet SHARD to results/attempts/; make_tables.py globs and concatenates them.
A run is reproducible iff (run_id, exp_id, seed, ...) fully determine each row.

The 2x2 memorization design lives in two columns:
  model_state       : finetuned | base          (was the model trained on D?)
  target_membership : trained   | control        (was THIS record trained on?)
and `train_frequency` = 0 encodes a control record (E5's f=0 == E1's control).
"""

import glob
import math
import os
from typing import List, Optional

from config import RESULTS_DIR

ATTEMPTS_DIR = os.path.join(RESULTS_DIR, "attempts")
os.makedirs(ATTEMPTS_DIR, exist_ok=True)

# (name, pandas dtype hint). Order is canonical; every row has exactly these keys.
ATTEMPT_COLUMNS = [
    ("run_id", "string"),
    ("exp_id", "string"),                 # E1 / E3 / ...
    ("seed", "Int64"),
    ("model_name", "string"),
    ("model_state", "string"),            # finetuned | base   (2x2 axis 1)
    ("target_membership", "string"),      # trained | control  (2x2 axis 2)
    ("person_id", "string"),
    ("field", "string"),                  # name/email/phone/address/ssn/credit_card
    ("train_frequency", "Int64"),         # control records -> 0
    ("probe", "string"),                  # fixed / fixed_matched / piiscope /
                                          # piicompass / gcg_free / gcg_anchored /
                                          # gcg_fluent / softprompt / random_restart
    ("capacity_k", "Int64"),              # free token positions; fixed=0, softprompt=-1
    ("softprompt_norm", "float64"),       # E14 norm constraint (else NaN)
    ("lambda_fluency", "float64"),        # E12 (else NaN)
    ("target_string", "string"),
    ("target_H_bits", "float64"),         # self-information under a HELD-OUT ref model
    ("target_len_tokens", "Int64"),
    ("prompt_text", "string"),
    ("prompt_token_ids", "object"),       # list[int]
    ("forward_passes", "Int64"),          # EXACT; compute-matching depends on this
    ("steps_run", "Int64"),
    ("steps_to_first_success", "float64"),  # continuous-score fallback (NaN if never)
    ("final_target_nll", "float64"),      # PRIMARY continuous score -> ROC/AUC (E9)
    ("generation", "string"),
    ("gen_len_tokens", "Int64"),          # guards against substring inflation
    ("exact_match", "boolean"),           # the ONE decision rule
    ("random_record_match", "boolean"),   # gen contains an UNRELATED record's same-field value
    ("wallclock_s", "float64"),
]

_COL_NAMES = [c for c, _ in ATTEMPT_COLUMNS]
_COL_SET = set(_COL_NAMES)


class AttemptLogger:
    """Accumulate per-attempt rows for one shard, then flush to a parquet file."""

    def __init__(self, run_id: str, exp_id: str, shard_tag: str):
        self.run_id = run_id
        self.exp_id = exp_id
        # sanitize shard_tag for a filename (model names contain '/')
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in shard_tag)
        self.path = os.path.join(ATTEMPTS_DIR, f"{run_id}__{exp_id}__{safe}.parquet")
        self.rows: List[dict] = []

    def log(self, **kw) -> None:
        extra = set(kw) - _COL_SET
        if extra:
            raise ValueError(f"attempt_log: unknown columns {sorted(extra)}; "
                             f"allowed = {_COL_NAMES}")
        row = {c: None for c in _COL_NAMES}
        row["run_id"] = self.run_id
        row["exp_id"] = self.exp_id
        row.update(kw)
        self.rows.append(row)

    def flush(self, verbose: bool = True) -> str:
        """Write every row accumulated so far. Full rewrite, so it is idempotent
        and safe to call per person -- which is what makes a preempted shard lose
        one person's attempts instead of the whole shard's in-memory buffer."""
        import pandas as pd
        df = pd.DataFrame(self.rows, columns=_COL_NAMES)
        # best-effort dtype coercion (parquet is fine with objects/NaN otherwise)
        for name, dtype in ATTEMPT_COLUMNS:
            if dtype in ("string", "Int64", "float64", "boolean") and name in df:
                try:
                    df[name] = df[name].astype(dtype)
                except (TypeError, ValueError):
                    pass
        df.to_parquet(self.path, index=False)
        if verbose:
            print(f"  [attempt_log] wrote {len(df)} rows -> {self.path}")
        return self.path


# ---------------------------------------------------------------------------
# Loading (make_tables.py side)
# ---------------------------------------------------------------------------

def load_attempts(run_id: Optional[str] = None, exp_id: Optional[str] = None):
    """Concatenate every attempt shard (optionally filtered by run/exp)."""
    import pandas as pd
    pat = f"{run_id or '*'}__{exp_id or '*'}__*.parquet"
    files = sorted(glob.glob(os.path.join(ATTEMPTS_DIR, pat)))
    if not files:
        return pd.DataFrame(columns=_COL_NAMES)
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


# ---------------------------------------------------------------------------
# Target self-information (bits) under a HELD-OUT reference model.
# Drives the capacity theory (H(t), beta, k*) and the ACR comparison (E13).
# ---------------------------------------------------------------------------

def target_self_information(text: str, ref_model, ref_tokenizer, device: str) -> "tuple[float, int]":
    """Return (H_bits, n_tokens): -sum_i log2 P(t_i | t_<i) under the ref model."""
    import torch
    ids = ref_tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    n = ids.shape[1]
    if n < 1:
        return 0.0, 0
    with torch.no_grad():
        logits = ref_model(input_ids=ids).logits  # (1, n, V)
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)  # predict tokens 1..n-1
    tok_logp = logp.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze()  # (n-1,)
    # first token has no context; approximate its surprisal by the unigram-ish
    # log-uniform over vocab is too loose -> use the model's first-step distribution
    with torch.no_grad():
        first = torch.log_softmax(ref_model(input_ids=ids[:, :1]).logits[:, -1, :].float(), dim=-1)
    first_logp = first[0, ids[0, 0]] if n >= 1 else torch.tensor(0.0)
    total_nll = -(tok_logp.sum() + first_logp)  # nats
    return float(total_nll / math.log(2)), int(n)
