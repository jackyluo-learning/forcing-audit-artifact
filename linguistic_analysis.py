"""
Linguistic feature extraction and logistic regression analysis.

Extracts 24 features (lexical, structural, syntactic, model-based) from
PII-containing documents and fits logistic regression models predicting
extraction success, controlling for training frequency.

Produces Table 4 (top predictors) for the IRI paper.
"""

import json
import math
import os
import re
import zlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy import stats

from config import ling_cfg, data_cfg, DATA_DIR, RESULTS_DIR, DEVICE
import evaluate as ev

# statsmodels gives proper Wald inference (SE, z, p) and McFadden pseudo-R².
# It is optional: if it is missing (or a fit fails), we fall back to sklearn
# coefficients with NaN inference rather than fabricating tiny standard errors.
try:
    import statsmodels.api as sm
    _HAVE_STATSMODELS = True
except Exception:  # pragma: no cover - environment without statsmodels
    sm = None
    _HAVE_STATSMODELS = False


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class LinguisticFeatureExtractor:
    """Extract 24 linguistic features from a text."""

    FEATURE_NAMES = [
        # Lexical (6)
        "token_count",
        "type_token_ratio",
        "rare_word_ratio",
        "avg_token_length",
        "stopword_ratio",
        "capitalization_ratio",
        # Structural (6)
        "entity_density",
        "proper_noun_ratio",
        "number_ratio",
        "special_char_ratio",
        "punctuation_density",
        "digit_ratio",
        # Syntactic (6)
        "max_dep_depth",
        "avg_dep_depth",
        "sentence_count",
        "avg_sentence_length",
        "noun_phrase_count",
        "verb_phrase_count",
        # Model-based (6)
        "perplexity",
        "avg_surprisal",
        "surprisal_variance",
        "max_surprisal",
        "compression_ratio",
        "entropy_estimate",
    ]

    def __init__(self, model=None, tokenizer=None):
        try:
            self.nlp = spacy.load(ling_cfg.spacy_model)
        except OSError:
            print(f"Downloading spaCy model '{ling_cfg.spacy_model}'...")
            spacy.cli.download(ling_cfg.spacy_model)
            self.nlp = spacy.load(ling_cfg.spacy_model)

        self.model = model
        self.tokenizer = tokenizer

        # Common English words for "rare word" threshold (rank > 10000)
        self._common_words = None

    def extract(self, text: str) -> Dict[str, float]:
        doc = self.nlp(text)
        tokens = [t for t in doc if not t.is_space]

        features = {}

        # --- Lexical ---
        n_tokens = len(tokens)
        features["token_count"] = n_tokens
        unique_tokens = set(t.lower_ for t in tokens)
        features["type_token_ratio"] = len(unique_tokens) / max(n_tokens, 1)

        # Rare word: approximated by whether token is out-of-vocabulary in spaCy
        rare = sum(1 for t in tokens if not t.is_alpha or not t.has_vector)
        features["rare_word_ratio"] = rare / max(n_tokens, 1)

        features["avg_token_length"] = (
            np.mean([len(t.text) for t in tokens]) if tokens else 0
        )
        features["stopword_ratio"] = (
            sum(1 for t in tokens if t.is_stop) / max(n_tokens, 1)
        )
        features["capitalization_ratio"] = (
            sum(1 for t in tokens if t.text[0].isupper()) / max(n_tokens, 1)
        )

        # --- Structural ---
        ents = list(doc.ents)
        features["entity_density"] = len(ents) / max(n_tokens, 1)
        features["proper_noun_ratio"] = (
            sum(1 for t in tokens if t.pos_ == "PROPN") / max(n_tokens, 1)
        )
        features["number_ratio"] = (
            sum(1 for t in tokens if t.pos_ == "NUM") / max(n_tokens, 1)
        )

        all_chars = list(text)
        n_chars = len(all_chars)
        features["special_char_ratio"] = (
            sum(1 for c in all_chars if not c.isalnum() and not c.isspace())
            / max(n_chars, 1)
        )
        features["punctuation_density"] = (
            sum(1 for t in tokens if t.is_punct) / max(n_tokens, 1)
        )
        features["digit_ratio"] = (
            sum(1 for c in all_chars if c.isdigit()) / max(n_chars, 1)
        )

        # --- Syntactic ---
        def _dep_depth(token):
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 100:
                    break
            return depth

        depths = [_dep_depth(t) for t in tokens] if tokens else [0]
        features["max_dep_depth"] = max(depths)
        features["avg_dep_depth"] = np.mean(depths)

        sents = list(doc.sents)
        features["sentence_count"] = len(sents)
        features["avg_sentence_length"] = (
            np.mean([len(list(s)) for s in sents]) if sents else 0
        )
        features["noun_phrase_count"] = len(list(doc.noun_chunks))

        # Approximate verb phrase count via verb tokens
        features["verb_phrase_count"] = sum(
            1 for t in tokens if t.pos_ in ("VERB", "AUX")
        )

        # --- Model-based ---
        if self.model is not None and self.tokenizer is not None:
            ppl_features = self._compute_model_features(text)
            features.update(ppl_features)
        else:
            features["perplexity"] = 0
            features["avg_surprisal"] = 0
            features["surprisal_variance"] = 0
            features["max_surprisal"] = 0

        # Compression-based (always available)
        text_bytes = text.encode("utf-8")
        compressed = zlib.compress(text_bytes)
        features["compression_ratio"] = len(compressed) / max(len(text_bytes), 1)
        features["entropy_estimate"] = (
            len(compressed) * 8 / max(len(text_bytes), 1)
        )

        return features

    @torch.no_grad()
    def _compute_model_features(self, text: str) -> Dict[str, float]:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        outputs = self.model(**enc, labels=enc["input_ids"])

        # Per-token log probabilities
        logits = outputs.logits[:, :-1, :]  # (1, seq_len-1, vocab)
        targets = enc["input_ids"][:, 1:]   # (1, seq_len-1)

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1).squeeze(0)  # (seq_len-1,)

        surprisals = -token_log_probs.cpu().numpy()

        perplexity = math.exp(np.mean(surprisals))
        avg_surprisal = float(np.mean(surprisals))
        var_surprisal = float(np.var(surprisals))
        max_surprisal = float(np.max(surprisals))

        return {
            "perplexity": perplexity,
            "avg_surprisal": avg_surprisal,
            "surprisal_variance": var_surprisal,
            "max_surprisal": max_surprisal,
        }


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    documents: List[Dict],
    extractor: LinguisticFeatureExtractor,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from all PII documents.
    Returns (feature_matrix, feature_names).
    """
    feature_list = []
    for i, doc in enumerate(documents):
        if i % 100 == 0:
            print(f"  Extracting features: {i}/{len(documents)}")
        features = extractor.extract(doc["text"])
        row = [features.get(fn, 0) for fn in LinguisticFeatureExtractor.FEATURE_NAMES]
        feature_list.append(row)

    X = np.array(feature_list)
    return X, LinguisticFeatureExtractor.FEATURE_NAMES


# ---------------------------------------------------------------------------
# Logistic regression analysis
# ---------------------------------------------------------------------------

def fit_extraction_predictor(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    frequencies: np.ndarray,
    label: str = "baseline",
) -> Dict:
    """
    Fit logistic regression predicting extraction success from linguistic
    features, controlling for frequency.

    Returns dict with coefficients, p-values, and model fit stats.
    """
    # Add frequency as a control variable
    X_with_freq = np.column_stack([X, frequencies])
    all_names = feature_names + ["frequency"]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_freq)

    # Handle edge case: if all y are same value
    if len(np.unique(y)) < 2:
        print(f"  [WARN] {label}: all targets have same outcome, skipping regression.")
        return {"error": "no variance in outcome"}

    # Inference: prefer statsmodels for a proper Wald test (SE, z, p) and a
    # McFadden pseudo-R². Fall back to sklearn coefficients with NaN inference
    # if statsmodels is unavailable or the MLE fails (perfect separation /
    # singular design). We NEVER invent tiny standard errors.
    coefs = None
    se = z_scores = p_values = None
    pseudo_r2 = float("nan")
    p_value_note = None
    inference_backend = None

    if _HAVE_STATSMODELS:
        try:
            X_sm = sm.add_constant(X_scaled, has_constant="add")
            sm_res = sm.Logit(y, X_sm).fit(disp=0)
            # statsmodels prepends the constant; drop it to align with all_names.
            coefs = np.asarray(sm_res.params)[1:]
            se = np.asarray(sm_res.bse)[1:]
            z_scores = np.asarray(sm_res.tvalues)[1:]
            p_values = np.asarray(sm_res.pvalues)[1:]
            pseudo_r2 = float(sm_res.prsquared)  # McFadden
            inference_backend = "statsmodels"
        except Exception as exc:  # perfect separation, singular Hessian, etc.
            print(f"  [WARN] {label}: statsmodels Logit failed ({exc}); "
                  f"falling back to sklearn coefficients with NaN inference.")
            coefs = None  # force sklearn fallback below

    if coefs is None:
        # sklearn coefficients (regularized) with NO fabricated inference.
        model = LogisticRegression(
            max_iter=1000, penalty="l2", C=1.0, solver="lbfgs"
        )
        model.fit(X_scaled, y)
        coefs = model.coef_[0]
        n = len(coefs)
        se = np.full(n, np.nan)
        z_scores = np.full(n, np.nan)
        p_values = np.full(n, np.nan)
        p_value_note = "unavailable (singular/separation)"
        inference_backend = "sklearn"

        # McFadden pseudo-R² is still well-defined from the fitted probabilities.
        probs = model.predict_proba(X_scaled)[:, 1]
        ll_model = np.sum(
            y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10)
        )
        base_prob = np.mean(y)
        ll_null = len(y) * (
            base_prob * np.log(base_prob + 1e-10)
            + (1 - base_prob) * np.log(1 - base_prob + 1e-10)
        )
        pseudo_r2 = float(1 - (ll_model / ll_null)) if ll_null != 0 else float("nan")

    results = {
        "label": label,
        "pseudo_r2": float(pseudo_r2),
        "inference_backend": inference_backend,
        "n_samples": len(y),
        "n_positive": int(np.sum(y)),
        "features": {},
    }
    if p_value_note is not None:
        results["p_value_note"] = p_value_note

    for i, name in enumerate(all_names):
        results["features"][name] = {
            "coefficient": float(coefs[i]),
            "std_error": float(se[i]),
            "z_score": float(z_scores[i]),
            "p_value": float(p_values[i]),
        }

    return results


def _mcfadden_pseudo_r2(X: np.ndarray, y: np.ndarray) -> float:
    """
    McFadden pseudo-R² of a logistic model of y on X. Uses statsmodels when
    available (proper MLE); otherwise an (approximate) sklearn fit. Returns NaN
    if the fit is degenerate. X should already be standardized/assembled by the
    caller; a constant is added internally.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if _HAVE_STATSMODELS:
        try:
            res = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0)
            return float(res.prsquared)
        except Exception:
            pass  # fall through to sklearn approximation

    try:
        model = LogisticRegression(max_iter=1000, penalty="l2", C=1e6, solver="lbfgs")
        model.fit(X, y)
        probs = np.clip(model.predict_proba(X)[:, 1], 1e-10, 1 - 1e-10)
        ll_model = np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        base = np.mean(y)
        ll_null = len(y) * (
            base * np.log(base + 1e-10) + (1 - base) * np.log(1 - base + 1e-10)
        )
        return float(1 - ll_model / ll_null) if ll_null != 0 else float("nan")
    except Exception:
        return float("nan")


def delta_r2_over_frequency(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    frequencies: np.ndarray,
    per_feature: bool = True,
) -> Dict:
    """
    Answer the reviewer question "which predictors add signal beyond frequency."

    Fits (a) a frequency-only logistic model and (b) the full model (all
    linguistic features + frequency), and reports the McFadden pseudo-R² of each
    plus their difference (Δ pseudo-R²). Optionally reports, per feature, the
    incremental pseudo-R² of adding that single feature on top of the
    frequency-only baseline (add-one-in).

    All predictors are standardized so the two models are on a common scale.
    """
    if len(np.unique(y)) < 2:
        return {"error": "no variance in outcome"}

    freq = np.asarray(frequencies, dtype=float).reshape(-1, 1)

    # Standardize frequency and the full design consistently.
    freq_scaled = StandardScaler().fit_transform(freq)
    X_full = np.column_stack([X, frequencies])
    X_full_scaled = StandardScaler().fit_transform(X_full)

    r2_freq = _mcfadden_pseudo_r2(freq_scaled, y)
    r2_full = _mcfadden_pseudo_r2(X_full_scaled, y)
    delta = (
        float(r2_full - r2_freq)
        if not (math.isnan(r2_freq) or math.isnan(r2_full))
        else float("nan")
    )

    out = {
        "frequency_only_pseudo_r2": float(r2_freq),
        "full_model_pseudo_r2": float(r2_full),
        "delta_pseudo_r2": delta,
        "n_samples": int(len(y)),
        "n_positive": int(np.sum(y)),
    }

    if per_feature:
        # Standardize each linguistic feature once, then add it (one at a time)
        # to the standardized frequency baseline and measure the R² gain.
        X_scaled = StandardScaler().fit_transform(X)
        incremental = {}
        for j, name in enumerate(feature_names):
            X_aug = np.column_stack([freq_scaled, X_scaled[:, j]])
            r2_aug = _mcfadden_pseudo_r2(X_aug, y)
            incremental[name] = (
                float(r2_aug - r2_freq)
                if not (math.isnan(r2_aug) or math.isnan(r2_freq))
                else float("nan")
            )
        out["per_feature_incremental_pseudo_r2"] = incremental

    return out


def print_top_predictors(
    baseline_results: Dict,
    gcg_results: Dict,
    top_n: int = 5,
) -> str:
    """Format the top predictors table (Table 4 in the paper)."""
    lines = []
    lines.append("=" * 75)
    lines.append("TABLE 4: Top Linguistic Predictors of Extraction Success")
    lines.append(f"  Baseline pseudo-R² = {baseline_results.get('pseudo_r2', 0):.2f}")
    lines.append(f"  Optimized pseudo-R² = {gcg_results.get('pseudo_r2', 0):.2f}")
    lines.append("=" * 75)
    lines.append(
        f"{'Feature':<22} {'Provenance':<13} "
        f"{'β_base':>8} {'p':>8} {'β_opt':>8} {'p':>8}"
    )
    lines.append("-" * 75)

    # Rank features by absolute baseline coefficient (excluding 'frequency')
    b_feats = baseline_results.get("features", {})
    g_feats = gcg_results.get("features", {})

    ranked = sorted(
        [(k, v) for k, v in b_feats.items() if k != "frequency"],
        key=lambda x: abs(x[1]["coefficient"]),
        reverse=True,
    )

    def fmt_p(p):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return "n/a"
        if p < 0.001:
            return "<.001"
        return f"{p:.3f}"

    for name, bv in ranked[:top_n]:
        gv = g_feats.get(name, {"coefficient": 0, "p_value": float("nan")})
        b_coef = bv["coefficient"]
        b_p = bv["p_value"]
        g_coef = gv["coefficient"]
        g_p = gv["p_value"]
        provenance = ling_cfg.feature_provenance.get(name, "descriptive")

        lines.append(
            f"{name:<22} {provenance:<13} "
            f"{b_coef:>+8.2f} {fmt_p(b_p):>8} "
            f"{g_coef:>+8.2f} {fmt_p(g_p):>8}"
        )

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def run_linguistic_analysis(
    model=None,
    tokenizer=None,
    baseline_results: Optional[List[Dict]] = None,
    gcg_results: Optional[List[Dict]] = None,
) -> Dict:
    """
    Full pipeline:
      1. Load PII documents
      2. Extract 24 features from each (model-based ppl/surprisal features are
         computed under a HELD-OUT reference model, not the target model, to
         avoid predicting the target's extraction from its own perplexity)
      3. Construct outcome vectors from baseline and GCG results via
         evaluate.person_extraction_outcome (a real per-person leak label)
      4. Fit logistic regression models (proper inference; no fabricated SEs)
      5. Report top predictors and Δ pseudo-R² over a frequency-only baseline

    `model`/`tokenizer` are the TARGET model. They are accepted for backward
    compatibility but are NOT used for the perplexity features when
    ling_cfg.use_reference_model_for_ppl is True.
    """
    print("\n" + "=" * 60)
    print("LINGUISTIC ANALYSIS")
    print("=" * 60)

    # Load PII documents
    docs_path = os.path.join(DATA_DIR, "pii_documents.json")
    with open(docs_path) as f:
        pii_docs = json.load(f)

    # Deduplicate by person (one row per person)
    seen = set()
    unique_docs = []
    for doc in pii_docs:
        name = doc["person_name"]
        if name not in seen:
            seen.add(name)
            unique_docs.append(doc)

    print(f"  {len(unique_docs)} unique PII documents for analysis")

    # --- Choose the model used for perplexity/surprisal features ---
    # Circularity fix: predicting a target model's extraction from that same
    # model's perplexity is circular. When configured, load a SEPARATE held-out
    # reference model and use IT (not the target model) for the model-based
    # features. We track which model produced the ppl features in the output.
    ppl_model, ppl_tokenizer = model, tokenizer
    ppl_model_name = "target"
    ref_model = None  # separately-loaded reference model, freed at the end
    if ling_cfg.use_reference_model_for_ppl:
        # Imported lazily so this module imports cleanly (and without CUDA) in
        # environments that lack transformers or a reference model checkpoint.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  Loading held-out reference model for ppl features: "
              f"'{ling_cfg.reference_model}'")
        ppl_tokenizer = AutoTokenizer.from_pretrained(ling_cfg.reference_model)
        if ppl_tokenizer.pad_token is None:
            ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ref_model = AutoModelForCausalLM.from_pretrained(
            ling_cfg.reference_model,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE).eval()
        ppl_model = ref_model
        ppl_model_name = ling_cfg.reference_model

    # Extract features
    print("  Extracting linguistic features...")
    extractor = LinguisticFeatureExtractor(ppl_model, ppl_tokenizer)
    X, feature_names = build_feature_matrix(unique_docs, extractor)
    print(f"  Feature matrix: {X.shape}")

    # Frequencies
    frequencies = np.array([doc["frequency"] for doc in unique_docs])

    # Build outcome vectors from results.
    #
    # Integrity fix: the outcome is now a REAL per-person leak label produced by
    # evaluate.person_extraction_outcome (1 iff ANY SENSITIVE field was actually
    # extracted), for BOTH baseline and GCG, aligned to `doc_names`. The old
    # heuristic (baseline "success" == response longer than 10 chars) marked
    # essentially every target as a success and made the regression meaningless.
    doc_names = [doc["person_name"] for doc in unique_docs]

    def _outcome_vector(results):
        if results is None:
            return np.zeros(len(doc_names))
        outcome = ev.person_extraction_outcome(results)  # {name: 0/1}
        return np.array([outcome.get(n, 0) for n in doc_names])

    y_baseline = _outcome_vector(baseline_results)
    y_gcg = _outcome_vector(gcg_results)

    print(f"  Baseline positive rate: {np.mean(y_baseline)*100:.1f}%")
    print(f"  GCG positive rate: {np.mean(y_gcg)*100:.1f}%")

    # Fit regression models
    print("  Fitting logistic regression (baseline)...")
    baseline_reg = fit_extraction_predictor(
        X, y_baseline, feature_names, frequencies, "baseline"
    )

    print("  Fitting logistic regression (optimized)...")
    gcg_reg = fit_extraction_predictor(
        X, y_gcg, feature_names, frequencies, "optimized"
    )

    # Δ pseudo-R² over a frequency-only baseline: which predictors add signal
    # beyond training frequency (the reviewer question).
    print("  Computing Δ pseudo-R² over frequency-only baseline...")
    baseline_delta_r2 = delta_r2_over_frequency(
        X, y_baseline, feature_names, frequencies
    )
    gcg_delta_r2 = delta_r2_over_frequency(
        X, y_gcg, feature_names, frequencies
    )
    print(f"    baseline: freq-only R²={baseline_delta_r2.get('frequency_only_pseudo_r2', float('nan')):.3f}, "
          f"full R²={baseline_delta_r2.get('full_model_pseudo_r2', float('nan')):.3f}, "
          f"Δ={baseline_delta_r2.get('delta_pseudo_r2', float('nan')):.3f}")
    print(f"    optimized: freq-only R²={gcg_delta_r2.get('frequency_only_pseudo_r2', float('nan')):.3f}, "
          f"full R²={gcg_delta_r2.get('full_model_pseudo_r2', float('nan')):.3f}, "
          f"Δ={gcg_delta_r2.get('delta_pseudo_r2', float('nan')):.3f}")

    # Print table
    report = print_top_predictors(baseline_reg, gcg_reg)

    # Free the separately-loaded reference model (if any).
    if ref_model is not None:
        del ref_model
        ppl_model = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Save
    analysis_out = {
        "baseline_regression": baseline_reg,
        "gcg_regression": gcg_reg,
        "baseline_delta_r2_over_frequency": baseline_delta_r2,
        "gcg_delta_r2_over_frequency": gcg_delta_r2,
        "feature_names": feature_names,
        "n_documents": len(unique_docs),
        # Which model produced the perplexity/surprisal features. "target" means
        # the (circular) target model; otherwise the held-out reference model.
        "ppl_feature_model": ppl_model_name,
        "used_reference_model_for_ppl": bool(ling_cfg.use_reference_model_for_ppl),
    }
    out_path = os.path.join(RESULTS_DIR, "linguistic_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis_out, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    return analysis_out


if __name__ == "__main__":
    run_linguistic_analysis()
