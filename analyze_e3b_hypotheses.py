"""Fail-closed H1--H5 analysis for the accepted formal E3b sweep.

The raw-evidence gate is delegated to :func:`analyze_e3b_repair.validate_and_load`.
Inference then follows the registered two-block design: trained and control people
are resampled independently, while each block reuses one draw across every value
of k.  H5 is exploratory.  H2 has no preregistered global test, so it receives no
p-value; a reserved value of one is used only to keep the four-slot Holm table
explicit and conservative.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analyze_e3b_repair import (
    AnalysisGateError,
    REGISTERED_TARGET_ANCHOR,
    validate_and_load,
)


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_240_601
H2_TOLERANCE = 0.01
H2_MAPPING = (0.01, 0.05, 0.09, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 0.90, 1.00)
H3_K = 20
FWER = 0.05
EXPECTED_SOLVER_SHA256 = "ed231efdd9a03d4fb630fd93316fb09477a8b08b2d06d3ff978ada91831ec8ba"

SOLVER_PATH = Path("analysis_support/censored_models.py")
SOLVER_VALIDATION_PATH = Path("analysis_support/solver_validation.json")

OUTPUT_JSON = "e3b_h1_h5_analysis.json"
OUTPUT_CURVE_CSV = "e3b_h1_h5_curve.csv"
OUTPUT_H2_CSV = "e3b_h2_mapping.csv"
OUTPUT_KMIN_CSV = "e3b_h4_kmin.csv"
OUTPUT_SUMMARY_CSV = "e3b_hypothesis_summary.csv"
OUTPUT_PNG = "e3b_h1_h5.png"
OUTPUT_PDF = "e3b_h1_h5.pdf"
OUTPUT_MD = "analysis.md"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisGateError(message)


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _percentile_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    _require(values.size > 0 and bool(np.isfinite(values).all()), "non-finite bootstrap replicate")
    return np.percentile(values, [2.5, 97.5], axis=0)


def _wilson_interval(rate: float, effective_n: float) -> np.ndarray:
    z = stats.norm.ppf(0.975)
    denominator = 1.0 + z * z / effective_n
    center = (rate + z * z / (2.0 * effective_n)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / effective_n + z * z / (4.0 * effective_n**2)
    ) / denominator
    return np.asarray([max(0.0, center - radius), min(1.0, center + radius)])


def _mover_difference(d_rate: float, c_rate: float, d_n: float, c_n: float) -> np.ndarray:
    d_low, d_high = _wilson_interval(d_rate, d_n)
    c_low, c_high = _wilson_interval(c_rate, c_n)
    return np.asarray([
        d_rate - c_rate - math.hypot(d_rate - d_low, c_high - c_rate),
        d_rate - c_rate + math.hypot(d_high - d_rate, c_rate - c_low),
    ])


def _rank_rows(values: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(stats.rankdata, 1, np.asarray(values, dtype=float))


def _spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_rank = stats.rankdata(np.asarray(x, dtype=float))
    x_rank -= x_rank.mean()
    y_rank = _rank_rows(y)
    y_rank -= y_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt((x_rank**2).sum() * (y_rank**2).sum(axis=1))
    return np.divide(
        y_rank @ x_rank,
        denominator,
        out=np.full(y_rank.shape[0], np.nan),
        where=denominator > 0,
    )


def _two_sided_centered_bootstrap_p(replicates: np.ndarray, point: float, null: float) -> float:
    replicates = np.asarray(replicates, dtype=float)
    _require(bool(np.isfinite(replicates).all()), "non-finite bootstrap values in p-value")
    return float(
        (np.count_nonzero(np.abs(replicates - point) >= abs(point - null) - 1e-12) + 1)
        / (len(replicates) + 1)
    )


def _left_centered_bootstrap_p(replicates: np.ndarray, point: float, null: float = 0.0) -> float:
    centered_null = np.asarray(replicates, dtype=float) - point + null
    return float((np.count_nonzero(centered_null <= point + 1e-12) + 1) / (len(centered_null) + 1))


def _arm_matrix(
    attempts: pd.DataFrame, arm: str, k_grid: Sequence[int]
) -> Tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    subset = attempts[attempts["target_membership"] == arm].copy()
    persons = sorted(str(value) for value in subset["person_id"].unique())
    _require(len(persons) == 25, f"expected 25 {arm} people, found {len(persons)}")
    grouped = (
        subset.groupby(["person_id", "capacity_k"], observed=True)["exact_match"]
        .agg(["sum", "count"])
        .reindex(pd.MultiIndex.from_product([persons, list(k_grid)], names=["person_id", "capacity_k"]))
    )
    _require(not grouped.isna().any().any(), f"{arm} person-by-k matrix is incomplete")
    expected_repeats = int(grouped["count"].iloc[0])
    _require(expected_repeats == 6, f"expected two fields x three seeds per {arm} person/k")
    _require(bool((grouped["count"] == expected_repeats).all()), f"unequal repeats in {arm} block")
    successes = grouped["sum"].to_numpy(dtype=float).reshape(len(persons), len(k_grid))
    totals = grouped["count"].to_numpy(dtype=float).reshape(len(persons), len(k_grid))
    return persons, successes / totals, successes, totals


def joint_person_bootstrap(
    attempts: pd.DataFrame,
    *,
    k_grid: Sequence[int],
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Make independent D/C person draws, each reused jointly across all k."""
    _require(n_bootstrap > 0, "n_bootstrap must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    people: Dict[str, list[str]] = {}
    matrices: Dict[str, np.ndarray] = {}
    successes: Dict[str, np.ndarray] = {}
    totals: Dict[str, np.ndarray] = {}
    indices: Dict[str, np.ndarray] = {}
    replicates: Dict[str, np.ndarray] = {}
    points: Dict[str, np.ndarray] = {}
    # This order is fixed for exact reproducibility with the registered seed.
    for arm in ("control", "trained"):
        people[arm], matrices[arm], successes[arm], totals[arm] = _arm_matrix(
            attempts, arm, k_grid
        )
        indices[arm] = rng.integers(0, len(people[arm]), size=(n_bootstrap, len(people[arm])))
        # Aggregate integer successes and denominators before division.  Averaging
        # person-level floats is algebraically equivalent here, but tiny summation
        # differences can split exact ties and therefore change rank statistics.
        sampled_successes = successes[arm][indices[arm]].sum(axis=1)
        sampled_totals = totals[arm][indices[arm]].sum(axis=1)
        replicates[arm] = sampled_successes / sampled_totals
        points[arm] = successes[arm].sum(axis=0) / totals[arm].sum(axis=0)
    _require(
        not np.array_equal(indices["control"], indices["trained"]),
        "trained and control bootstrap blocks unexpectedly share identical draws",
    )
    return {
        "people": people,
        "matrices": matrices,
        "successes": successes,
        "totals": totals,
        "indices": indices,
        "replicates": replicates,
        "points": points,
        "tau_replicates": replicates["trained"] - replicates["control"],
        "tau_point": points["trained"] - points["control"],
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
    }


def build_curve(
    attempts: pd.DataFrame, bootstrap: Mapping[str, Any], k_grid: Sequence[int]
) -> Tuple[list[Dict[str, Any]], pd.DataFrame]:
    n_people = 25
    n_fields = 2
    boundary_effective_n = n_people * n_fields / (1.0 + (n_fields - 1.0) * 0.5)
    rows: list[Dict[str, Any]] = []
    for j, k in enumerate(k_grid):
        record: Dict[str, Any] = {"k": int(k), "probe": "fixed" if int(k) == 0 else "gcg_free"}
        boundary = False
        for arm in ("control", "trained"):
            point = float(bootstrap["points"][arm][j])
            raw_ci = _percentile_interval(bootstrap["replicates"][arm][:, j])
            hits = int(bootstrap["successes"][arm][:, j].sum())
            count = int(bootstrap["totals"][arm][:, j].sum())
            at_boundary = hits in (0, count)
            boundary = boundary or at_boundary
            ci = _wilson_interval(point, boundary_effective_n) if at_boundary else raw_ci
            record[arm] = {
                "estimate": point,
                "reported_95_ci": ci,
                "person_bootstrap_95_ci": raw_ci,
                "interval_method": (
                    "wilson_effective_n_at_boundary"
                    if at_boundary
                    else "person_clustered_percentile_bootstrap"
                ),
                "hits": hits,
                "attempts": count,
                "n_persons": n_people,
                "n_targets": n_people * n_fields,
                "n_seeds": 3,
            }
        tau = float(bootstrap["tau_point"][j])
        tau_raw_ci = _percentile_interval(bootstrap["tau_replicates"][:, j])
        if boundary:
            tau_ci = _mover_difference(
                record["trained"]["estimate"],
                record["control"]["estimate"],
                boundary_effective_n,
                boundary_effective_n,
            )
            tau_method = "newcombe_mover_from_wilson_effective_n"
        else:
            tau_ci = tau_raw_ci
            tau_method = "independent_two_block_person_bootstrap"
        record.update(
            {
                "tau": tau,
                "tau_reported_95_ci": tau_ci,
                "tau_person_bootstrap_95_ci": tau_raw_ci,
                "tau_interval_method": tau_method,
            }
        )
        rows.append(record)

    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "k": row["k"],
                "probe": row["probe"],
                "c_hits": row["control"]["hits"],
                "c_attempts": row["control"]["attempts"],
                "alpha": row["control"]["estimate"],
                "alpha_ci_low": row["control"]["reported_95_ci"][0],
                "alpha_ci_high": row["control"]["reported_95_ci"][1],
                "alpha_ci_method": row["control"]["interval_method"],
                "d_hits": row["trained"]["hits"],
                "d_attempts": row["trained"]["attempts"],
                "emr_d": row["trained"]["estimate"],
                "emr_d_ci_low": row["trained"]["reported_95_ci"][0],
                "emr_d_ci_high": row["trained"]["reported_95_ci"][1],
                "emr_d_ci_method": row["trained"]["interval_method"],
                "tau": row["tau"],
                "tau_ci_low": row["tau_reported_95_ci"][0],
                "tau_ci_high": row["tau_reported_95_ci"][1],
                "tau_ci_method": row["tau_interval_method"],
            }
        )
    return rows, pd.DataFrame(flat_rows)


def _h2_mapping(curve_rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    positive_rows = [row for row in curve_rows if int(row["k"]) > 0]
    mapping = []
    for tolerance in H2_MAPPING:
        floor = [row for row in positive_rows if row["control"]["reported_95_ci"][1] <= tolerance]
        any_direction = [
            row for row in floor
            if row["tau_reported_95_ci"][0] > 0 or row["tau_reported_95_ci"][1] < 0
        ]
        positive = [row for row in floor if row["tau_reported_95_ci"][0] > 0]
        negative = [row for row in floor if row["tau_reported_95_ci"][1] < 0]
        mapping.append(
            {
                "tolerance": float(tolerance),
                "below_preregistered_resolution": bool(tolerance < 0.09),
                "floor_only_capacities": [int(row["k"]) for row in floor],
                "joint_any_direction_capacities": [int(row["k"]) for row in any_direction],
                "joint_positive_signal_capacities": [int(row["k"]) for row in positive],
                "joint_negative_signal_capacities": [int(row["k"]) for row in negative],
                "largest_positive_joint_capacity": (
                    max(int(row["k"]) for row in positive) if positive else None
                ),
            }
        )
    return mapping


def _h2_consistent_wilson_sensitivity(
    curve_rows: Sequence[Mapping[str, Any]],
    primary_mapping: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Apply each registered uniform Wilson/MOVER convention at every positive k."""
    positive_rows = [row for row in curve_rows if int(row["k"]) > 0]
    conventions = []
    for name, effective_n in (
        ("target_only_icc_0.5", 100.0 / 3.0),
        ("all_repeated_attempts_icc_0.5", 300.0 / 7.0),
    ):
        per_k = []
        for row in positive_rows:
            c_rate = float(row["control"]["estimate"])
            d_rate = float(row["trained"]["estimate"])
            per_k.append(
                {
                    "k": int(row["k"]),
                    "control_wilson_95_ci": _wilson_interval(c_rate, effective_n),
                    "trained_wilson_95_ci": _wilson_interval(d_rate, effective_n),
                    "tau_mover_95_ci": _mover_difference(
                        d_rate, c_rate, effective_n, effective_n
                    ),
                }
            )
        mapping = []
        for tolerance in H2_MAPPING:
            eligible = [
                row for row in per_k if row["control_wilson_95_ci"][1] <= tolerance
            ]
            any_direction = [
                row
                for row in eligible
                if row["tau_mover_95_ci"][0] > 0 or row["tau_mover_95_ci"][1] < 0
            ]
            positive = [row for row in eligible if row["tau_mover_95_ci"][0] > 0]
            negative = [row for row in eligible if row["tau_mover_95_ci"][1] < 0]
            mapping.append(
                {
                    "tolerance": float(tolerance),
                    "floor_only_capacities": [row["k"] for row in eligible],
                    "joint_any_direction_capacities": [row["k"] for row in any_direction],
                    "joint_positive_signal_capacities": [row["k"] for row in positive],
                    "joint_negative_signal_capacities": [row["k"] for row in negative],
                }
            )
        conventions.append(
            {
                "name": name,
                "effective_n": effective_n,
                "per_k": per_k,
                "mapping": mapping,
            }
        )
    primary_positive = {
        k for row in primary_mapping for k in row["joint_positive_signal_capacities"]
    }
    sensitivity_positive_by_convention = {
        convention["name"]: {
            k
            for row in convention["mapping"]
            for k in row["joint_positive_signal_capacities"]
        }
        for convention in conventions
    }
    disappearing_by_convention = {
        name: sorted(primary_positive - positive)
        for name, positive in sensitivity_positive_by_convention.items()
    }
    return {
        "label": "post-hoc method-consistency sensitivity; does not replace the hybrid primary intervals",
        "method": "all k use one Wilson convention; tau uses Newcombe/MOVER from the same intervals",
        "conventions": conventions,
        "primary_positive_joint_capacities_that_disappear": disappearing_by_convention,
        "k6_primary_positive_joint_disappears_in_both": bool(
            all(6 in values for values in disappearing_by_convention.values())
        ),
        "all_conventions_have_no_positive_joint_point": bool(
            all(not values for values in sensitivity_positive_by_convention.values())
        ),
        "conclusion": (
            "The primary hybrid method yields k=6 as a positive joint point at tolerances of 30% "
            "or more. Under both uniform Wilson/MOVER conventions (target-only n_eff=100/3 and "
            "all repeated attempts n_eff=300/7), that point disappears and no positive joint "
            "capacity remains at any mapped tolerance."
        ),
    }


def _h5_statistics(
    positive_k: np.ndarray,
    tau_point: np.ndarray,
    tau_replicates: np.ndarray,
) -> Dict[str, Any]:
    """Compute the exploratory tied-argmax envelope and log-k quadratic sensitivity."""
    positive_k = np.asarray(positive_k, dtype=float)
    observed = np.asarray(tau_point, dtype=float)
    positive_tau = np.asarray(tau_replicates, dtype=float)
    _require(
        positive_tau.ndim == 2 and positive_tau.shape[1] == len(positive_k),
        "H5 replicate matrix does not match the positive-k grid",
    )
    max_value = positive_tau.max(axis=1)
    ties = np.isclose(positive_tau, max_value[:, None], atol=1e-12, rtol=0.0)
    earliest = positive_k[np.argmax(ties, axis=1)]
    latest = positive_k[len(positive_k) - 1 - np.argmax(ties[:, ::-1], axis=1)]
    observed_maximizers = positive_k[
        np.isclose(observed, observed.max(), atol=1e-12, rtol=0.0)
    ]
    design = np.stack(
        [np.ones(len(positive_k)), np.log(positive_k), np.log(positive_k) ** 2], axis=1
    )
    projection = np.linalg.pinv(design).T
    coefficients = observed @ projection
    coefficient_reps = positive_tau @ projection
    quadratic_ci = _percentile_interval(coefficient_reps[:, 2])
    argmax_envelope = np.asarray(
        [np.percentile(earliest, 2.5), np.percentile(latest, 97.5)]
    )
    interior_localized = bool(
        argmax_envelope[0] > positive_k.min() and argmax_envelope[1] < positive_k.max()
    )
    return {
        "status": "exploratory_underpowered",
        "observed_maximizers": observed_maximizers,
        "argmax_envelope_95_ci": argmax_envelope,
        "first_argmax_sensitivity_95_ci": _percentile_interval(earliest),
        "tied_maximum_replicates": int((ties.sum(axis=1) > 1).sum()),
        "interior_peak_localized": interior_localized,
        "quadratic_log_k_squared_coefficient": float(coefficients[2]),
        "quadratic_bootstrap_95_ci": quadratic_ci,
        "quadratic_one_sided_p": _left_centered_bootstrap_p(
            coefficient_reps[:, 2], float(coefficients[2]), 0.0
        ),
        "quadratic_p_method": "null-centered bootstrap left tail; exploratory and unadjusted",
        "verdict": (
            "exploratory_interior_peak_localized"
            if interior_localized
            else "exploratory_peak_not_localized"
        ),
        "caveat": (
            "No flatness/equivalence threshold was preregistered; failure to detect negative "
            "curvature is not evidence that the curve is flat."
        ),
    }
def analyze_curve_hypotheses(
    curve_rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    k_grid: Sequence[int],
) -> Dict[str, Any]:
    k_array = np.asarray(k_grid, dtype=int)
    positive = np.flatnonzero(k_array > 0)
    positive_k = k_array[positive].astype(float)

    rho_point = float(stats.spearmanr(positive_k, bootstrap["points"]["control"][positive]).statistic)
    rho_reps = _spearman_rows(positive_k, bootstrap["replicates"]["control"][:, positive])
    valid_rho = rho_reps[np.isfinite(rho_reps)]
    _require(
        len(valid_rho) >= math.ceil(0.95 * bootstrap["n_bootstrap"]),
        "fewer than 95% of H1 bootstrap replicates have a defined Spearman rho",
    )
    rho_ci = _percentile_interval(valid_rho)
    h1_p = float((np.count_nonzero(valid_rho <= 0) + 1) / (len(valid_rho) + 1))
    h1 = {
        "estimand": "Spearman rho between positive k and the pooled control forcing floor",
        "rho": rho_point,
        "bootstrap_95_ci": rho_ci,
        "p_raw_one_sided": h1_p,
        "valid_bootstrap_replicates": len(valid_rho),
        "verdict": "supported" if rho_ci[0] > 0 else "not_supported",
        "rule": "support requires the person-bootstrap 95% CI for rho to lie strictly above zero",
    }

    mapping = _h2_mapping(curve_rows)
    one_percent = next(row for row in mapping if math.isclose(row["tolerance"], H2_TOLERANCE))
    positive_rows = [row for row in curve_rows if int(row["k"]) > 0]
    floor_point = [
        int(row["k"]) for row in positive_rows if row["control"]["estimate"] <= H2_TOLERANCE
    ]
    positive_signal_all = [
        int(row["k"]) for row in positive_rows if row["tau_reported_95_ci"][0] > 0
    ]
    zero_upper = float(_wilson_interval(0.0, 50.0 / 1.5)[1])
    z = stats.norm.ppf(0.975)
    people_needed = int(math.ceil((z * z * (1.0 - H2_TOLERANCE) / H2_TOLERANCE) * 1.5 / 2.0))
    consistent_sensitivity = _h2_consistent_wilson_sensitivity(curve_rows, mapping)
    h2 = {
        "literal_one_percent": {
            "tolerance": H2_TOLERANCE,
            "point_estimate_floor_capacities": floor_point,
            "upper_ci_floor_capacities": one_percent["floor_only_capacities"],
            "joint_any_direction_capacities": one_percent["joint_any_direction_capacities"],
            "joint_positive_signal_capacities": one_percent["joint_positive_signal_capacities"],
            "positive_signal_capacities_without_floor_requirement": positive_signal_all,
            "zero_hit_wilson_upper": zero_upper,
            "minimum_people_per_arm_for_zero_hit_wilson_upper_at_most_one_percent": people_needed,
            "verdict": "unresolved_at_achieved_sample_size",
        },
        "joint_mapping": mapping,
        "global_test": "undefined_by_preregistration",
        "p_raw": None,
        "holm_reserved_p": 1.0,
        "consistent_interval_sensitivity": consistent_sensitivity,
        "rule": (
            "A useful positive audit point requires alpha's 95% upper bound at or below the "
            "tolerance and tau's 95% lower bound above zero. Any-direction CI exclusion is "
            "reported separately and a negative signal never counts as useful."
        ),
        "conclusion": (
            "The literal 1% form is unresolved; the mapping is descriptive and does not establish "
            "that no operating point exists outside the achieved precision or grid."
        ),
    }

    h3_indices = np.flatnonzero(k_array == H3_K)
    _require(len(h3_indices) == 1, f"k={H3_K} is absent from the formal grid")
    j = int(h3_indices[0])
    h3_row = curve_rows[j]
    tau_point = float(bootstrap["tau_point"][j])
    tau_reps = np.asarray(bootstrap["tau_replicates"][:, j], dtype=float)
    tau_ci = np.asarray(h3_row["tau_reported_95_ci"], dtype=float)
    h3_p = _two_sided_centered_bootstrap_p(tau_reps, tau_point, 0.0)
    if tau_ci[0] > 0:
        h3_verdict = "measurable_positive_signal"
    elif tau_ci[1] < 0:
        h3_verdict = "measurable_negative_signal"
    else:
        h3_verdict = "does_not_reject_zero"
    h3 = {
        "k": H3_K,
        "tau": tau_point,
        "bootstrap_95_ci": tau_ci,
        "p_raw_two_sided": h3_p,
        "verdict": h3_verdict,
        "rule": "the registered null is rejected only when tau's 95% interval excludes zero",
    }

    observed = np.asarray(bootstrap["tau_point"][positive], dtype=float)
    h5 = _h5_statistics(
        positive_k,
        observed,
        np.asarray(bootstrap["tau_replicates"][:, positive], dtype=float),
    )
    return {"H1": h1, "H2": h2, "H3": h3, "H5": h5}


def build_kmin(attempts: pd.DataFrame, positive_k: Sequence[int]) -> pd.DataFrame:
    subset = attempts[attempts["capacity_k"].isin(positive_k)].copy()
    rows = []
    for (arm, person_id, field), group in subset.groupby(
        ["target_membership", "person_id", "field"], sort=True, observed=True
    ):
        by_k = group.groupby("capacity_k", observed=True)["exact_match"].any().reindex(positive_k)
        _require(not by_k.isna().any(), f"incomplete k_min grid for {(arm, person_id, field)}")
        h_values = group["target_H_bits"].dropna().unique()
        _require(len(h_values) == 1 and float(h_values[0]) > 0, "H(t) must be one finite positive value per target")
        hit_positions = np.flatnonzero(by_k.to_numpy(dtype=bool))
        if len(hit_positions):
            first = int(hit_positions[0])
            lower = float(positive_k[first - 1]) if first else 0.0
            upper = float(positive_k[first])
            nonmonotone = bool((~by_k.iloc[first + 1 :]).any())
        else:
            lower = float(positive_k[-1])
            upper = float("inf")
            nonmonotone = False
        rows.append(
            {
                "arm": str(arm),
                "person_id": str(person_id),
                "field": str(field),
                "H_bits": float(h_values[0]),
                "lower_k": lower,
                "upper_k": upper,
                "right_censored": not np.isfinite(upper),
                "nonmonotone_after_first_hit": nonmonotone,
            }
        )
    result = pd.DataFrame(rows)
    _require(len(result) == 100, f"expected 100 target-level k_min rows, found {len(result)}")
    return result


def _load_validated_solver(repo_root: Path):
    solver_path = repo_root / SOLVER_PATH
    validation_path = repo_root / SOLVER_VALIDATION_PATH
    _require(solver_path.is_file(), f"missing validated censored solver: {solver_path}")
    digest = hashlib.sha256(solver_path.read_bytes()).hexdigest()
    _require(digest == EXPECTED_SOLVER_SHA256, "censored_models.py changed since its solver validation")
    _require(validation_path.is_file(), f"missing solver validation record: {validation_path}")
    validation = json.loads(validation_path.read_text())
    _require(len(validation) >= 4, "solver validation record lacks full plus three bootstrap checks")
    _require(
        all(float(row["max_parameter_difference"]) < 1e-3 for row in validation),
        "solver/lifelines parameter validation exceeds 1e-3",
    )
    _require(
        all(abs(float(row["log_likelihood_difference"])) < 1e-5 for row in validation),
        "solver/lifelines likelihood validation exceeds 1e-5",
    )
    spec = importlib.util.spec_from_file_location("e3b_validated_censored_models", solver_path)
    _require(spec is not None and spec.loader is not None, "cannot load censored_models solver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(callable(getattr(module, "weibull_fit", None)), "validated solver lacks weibull_fit")
    _require(
        callable(getattr(module, "normal_interval_fit", None)),
        "validated solver lacks normal_interval_fit",
    )
    return module.weibull_fit, module.normal_interval_fit, digest, validation


def analyze_h4(
    kmin: pd.DataFrame,
    control_draws: np.ndarray,
    *,
    repo_root: Path,
) -> Dict[str, Any]:
    weibull_fit, normal_interval_fit, solver_sha, validation = _load_validated_solver(repo_root)
    control = kmin[kmin["arm"] == "control"].sort_values(["person_id", "field"]).reset_index(drop=True)
    people = sorted(control["person_id"].unique())
    _require(len(control) == 50 and len(people) == 25, "H4 requires 50 control targets from 25 people")
    person_position = {person: j for j, person in enumerate(people)}
    target_person = np.asarray([person_position[value] for value in control["person_id"]], dtype=int)
    counts = np.stack([(control_draws == j).sum(axis=1) for j in range(len(people))], axis=1)

    h = control["H_bits"].to_numpy(dtype=float)
    log_h = np.log(h)
    center = float(log_h.mean())
    x = log_h - center
    lower = control["lower_k"].to_numpy(dtype=float)
    upper = control["upper_k"].to_numpy(dtype=float)
    point_parameters, point_nll = weibull_fit(x, lower, upper)
    log_lower = np.log(np.maximum(lower, 1e-100))
    log_upper = np.log(upper)
    lognormal_parameters, lognormal_nll = normal_interval_fit(x, log_lower, log_upper)
    gammas = np.empty(len(control_draws), dtype=float)
    intercepts = np.empty(len(control_draws), dtype=float)
    lognormal_gammas = np.empty(len(control_draws), dtype=float)
    lognormal_intercepts = np.empty(len(control_draws), dtype=float)
    failures = []
    lognormal_retries = []
    for i, person_counts in enumerate(counts):
        weights = person_counts[target_person]
        try:
            fitted, _ = weibull_fit(x, lower, upper, weights=weights, start=point_parameters)
            gammas[i] = fitted[1]
            intercepts[i] = fitted[0] - fitted[1] * center
        except Exception as exc:  # fail closed: retain and stop after the loop's first fault
            failures.append({"replicate": i, "model": "weibull", "error": f"{type(exc).__name__}: {exc}"})
            break
        try:
            lognormal_fitted, _ = normal_interval_fit(
                x, log_lower, log_upper, weights=weights, start=lognormal_parameters
            )
        except Exception as primary_exc:
            # L-BFGS occasionally reports an abnormal line search at an otherwise
            # regular optimum. Retry the identical likelihood from its documented
            # weighted default start; no observation or replicate is discarded.
            try:
                lognormal_fitted, _ = normal_interval_fit(
                    x, log_lower, log_upper, weights=weights, start=None
                )
                lognormal_retries.append(
                    {
                        "replicate": i,
                        "primary_error": f"{type(primary_exc).__name__}: {primary_exc}",
                        "resolution": "same likelihood converged from weighted default start",
                    }
                )
            except Exception as retry_exc:
                failures.append(
                    {
                        "replicate": i,
                        "model": "lognormal",
                        "error": f"{type(retry_exc).__name__}: {retry_exc}",
                        "primary_error": f"{type(primary_exc).__name__}: {primary_exc}",
                    }
                )
                break
        try:
            lognormal_gammas[i] = lognormal_fitted[1]
            lognormal_intercepts[i] = lognormal_fitted[0] - lognormal_fitted[1] * center
        except Exception as exc:
            failures.append({"replicate": i, "model": "lognormal", "error": f"{type(exc).__name__}: {exc}"})
            break
    _require(not failures, f"H4 bootstrap fit failed: {failures[0] if failures else 'unknown'}")
    point_intercept = float(point_parameters[0] - point_parameters[1] * center)
    gamma = float(point_parameters[1])
    gamma_ci = _percentile_interval(gammas)
    p_raw = _two_sided_centered_bootstrap_p(gammas, gamma, 1.0)
    lognormal_gamma = float(lognormal_parameters[1])
    lognormal_gamma_ci = _percentile_interval(lognormal_gammas)
    if gamma_ci[0] <= 1.0 <= gamma_ci[1]:
        verdict = "proportionality_not_rejected"
    else:
        verdict = "proportionality_refuted"
    return {
        "model": "control-only interval/right-censored Weibull log-log AFT",
        "gamma": gamma,
        "gamma_bootstrap_95_ci": gamma_ci,
        "p_raw_two_sided": p_raw,
        "verdict": verdict,
        "rule": "proportionality is refuted when gamma's bootstrap 95% CI excludes 1",
        "n_targets": len(control),
        "n_people": len(people),
        "n_seeds": 3,
        "bootstrap_replicates": len(control_draws),
        "right_censored_targets": int(control["right_censored"].sum()),
        "right_censored_fraction": float(control["right_censored"].mean()),
        "nonmonotone_targets": int(control["nonmonotone_after_first_hit"].sum()),
        "intercept_log_scale": point_intercept,
        "intercept_bootstrap_95_ci": _percentile_interval(intercepts),
        "beta_scale_exp_minus_intercept": float(math.exp(-point_intercept)),
        "beta_warning": (
            "This scale is not a constant bits/token forcing rate when gamma differs from 1; "
            "do not quote it as beta if proportionality is refuted."
        ),
        "negative_log_likelihood": float(point_nll),
        "lognormal_distribution_sensitivity": {
            "model": "control-only interval/right-censored log-normal log-log AFT",
            "gamma": lognormal_gamma,
            "gamma_bootstrap_95_ci": lognormal_gamma_ci,
            "p_raw_two_sided": _two_sided_centered_bootstrap_p(
                lognormal_gammas, lognormal_gamma, 1.0
            ),
            "intercept_log_scale": float(
                lognormal_parameters[0] - lognormal_parameters[1] * center
            ),
            "intercept_bootstrap_95_ci": _percentile_interval(lognormal_intercepts),
            "negative_log_likelihood": float(lognormal_nll),
            "bootstrap_replicates": len(control_draws),
            "bootstrap_failures": failures,
            "solver_retries": lognormal_retries,
            "label": "distribution sensitivity required by method_choices.md",
            "gamma_ci_excludes_one": bool(
                lognormal_gamma_ci[0] > 1.0 or lognormal_gamma_ci[1] < 1.0
            ),
        },
        "bootstrap_failures": failures,
        "solver": {
            "path": str(SOLVER_PATH),
            "sha256": solver_sha,
            "validation_path": str(SOLVER_VALIDATION_PATH),
            "validation_checks": validation,
        },
    }


def holm_conditional(hypotheses: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    p_values = {
        "H1": float(hypotheses["H1"]["p_raw_one_sided"]),
        "H2": 1.0,
        "H3": float(hypotheses["H3"]["p_raw_two_sided"]),
        "H4": float(hypotheses["H4"]["p_raw_two_sided"]),
    }
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: Dict[str, Any] = {}
    running = 0.0
    for position, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - position) * p_value))
        adjusted[name] = {
            "p_raw_or_reserved": p_value,
            "p_holm": running,
            "reject_at_0_05": bool(running <= FWER),
            "h2_reserved_not_tested": name == "H2",
        }
    return {
        "family": ["H1", "H2", "H3", "H4"],
        "fwer": FWER,
        "status": "conditional_bookkeeping_only",
        "note": (
            "H2 has no preregistered global p-value and is conservatively reserved at p=1. "
            "This table does not repair that missing test contract. H5 is exploratory and excluded."
        ),
        "results": adjusted,
    }


def _hypothesis_summary(hypotheses: Mapping[str, Mapping[str, Any]], holm: Mapping[str, Any]) -> pd.DataFrame:
    h1, h2, h3, h4, h5 = (hypotheses[name] for name in ("H1", "H2", "H3", "H4", "H5"))
    return pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "status": "confirmatory",
                "estimate": h1["rho"],
                "ci_low": h1["bootstrap_95_ci"][0],
                "ci_high": h1["bootstrap_95_ci"][1],
                "p_raw": h1["p_raw_one_sided"],
                "p_holm": holm["results"]["H1"]["p_holm"],
                "verdict": h1.get("family_adjusted_verdict", h1["verdict"]),
            },
            {
                "hypothesis": "H2",
                "status": "confirmatory_but_global_test_undefined",
                "estimate": H2_TOLERANCE,
                "ci_low": None,
                "ci_high": h2["literal_one_percent"]["zero_hit_wilson_upper"],
                "p_raw": None,
                "p_holm": holm["results"]["H2"]["p_holm"],
                "verdict": h2["literal_one_percent"]["verdict"],
            },
            {
                "hypothesis": "H3",
                "status": "confirmatory",
                "estimate": h3["tau"],
                "ci_low": h3["bootstrap_95_ci"][0],
                "ci_high": h3["bootstrap_95_ci"][1],
                "p_raw": h3["p_raw_two_sided"],
                "p_holm": holm["results"]["H3"]["p_holm"],
                "verdict": h3.get("family_adjusted_verdict", h3["verdict"]),
            },
            {
                "hypothesis": "H4",
                "status": "confirmatory_working_model",
                "estimate": h4["gamma"],
                "ci_low": h4["gamma_bootstrap_95_ci"][0],
                "ci_high": h4["gamma_bootstrap_95_ci"][1],
                "p_raw": h4["p_raw_two_sided"],
                "p_holm": holm["results"]["H4"]["p_holm"],
                "verdict": h4.get("family_adjusted_verdict", h4["verdict"]),
            },
            {
                "hypothesis": "H5",
                "status": "exploratory_underpowered",
                "estimate": h5["quadratic_log_k_squared_coefficient"],
                "ci_low": h5["quadratic_bootstrap_95_ci"][0],
                "ci_high": h5["quadratic_bootstrap_95_ci"][1],
                "p_raw": h5["quadratic_one_sided_p"],
                "p_holm": None,
                "verdict": h5["verdict"],
            },
        ]
    )


def _write_figure(curve: pd.DataFrame, staging: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = curve["k"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for prefix, label, color in (
        ("alpha", "Control C: forcing floor", "#b54a35"),
        ("emr_d", "Trained D: extraction rate", "#2b6f9c"),
    ):
        y = curve[prefix].to_numpy(dtype=float)
        low = curve[f"{prefix}_ci_low"].to_numpy(dtype=float)
        high = curve[f"{prefix}_ci_high"].to_numpy(dtype=float)
        axes[0].plot(x, y, "o-", color=color, label=label)
        axes[0].fill_between(x, low, high, color=color, alpha=0.15)
    axes[0].set_ylabel("Exact-match rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8, loc="upper left")

    tau = curve["tau"].to_numpy(dtype=float)
    tau_low = curve["tau_ci_low"].to_numpy(dtype=float)
    tau_high = curve["tau_ci_high"].to_numpy(dtype=float)
    axes[1].errorbar(
        x,
        tau,
        yerr=np.vstack((tau - tau_low, tau_high - tau)),
        fmt="o-",
        capsize=2.5,
        color="#555555",
    )
    axes[1].axhline(0.0, color="#777777", linewidth=1.0)
    axes[1].set_ylabel("D - C exact-match rate")
    for axis in axes:
        axis.axvline(0.5, color="#888888", linestyle=":", linewidth=0.9)
        axis.set_xscale("symlog", linthresh=1, base=2)
        axis.set_xlim(-0.12, 72)
        axis.set_xticks([0, 1, 2, 4, 8, 16, 32, 64])
        axis.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "64"])
        axis.set_xlabel("Prompt capacity k (free tokens)")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(staging / OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(staging / OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def _analysis_markdown(payload: Mapping[str, Any]) -> str:
    h = payload["hypotheses"]
    h1, h2, h3, h4, h5 = (h[name] for name in ("H1", "H2", "H3", "H4", "H5"))
    one = h2["literal_one_percent"]
    return f"""# E3b H1--H5 central analysis

## Scope and evidence gate

This analysis uses the accepted 42-shard E3b formal set (4,200 attempts, 14 capacities x 3 seeds) after the full `analyze_e3b_repair.validate_and_load` gate. Trained and control people are bootstrapped as two independent blocks; one person draw per block is reused across every k. There are {payload['bootstrap']['replicates']:,} replicates with seed {payload['bootstrap']['seed']}.

## Results

| Hypothesis | Estimate and 95% interval | Verdict |
|---|---|---|
| H1 | Spearman rho = {h1['rho']:.6f} [{h1['bootstrap_95_ci'][0]:.6f}, {h1['bootstrap_95_ci'][1]:.6f}] | {h1.get('family_adjusted_verdict', h1['verdict'])} |
| H2 | literal 1% floor; zero-hit Wilson upper = {one['zero_hit_wilson_upper']:.4f}; positive joint k = {one['joint_positive_signal_capacities']} | {one['verdict']} |
| H3 | tau(20) = {h3['tau']:+.4f} [{h3['bootstrap_95_ci'][0]:+.4f}, {h3['bootstrap_95_ci'][1]:+.4f}] | {h3.get('family_adjusted_verdict', h3['verdict'])} |
| H4 | Weibull AFT gamma = {h4['gamma']:.6f} [{h4['gamma_bootstrap_95_ci'][0]:.6f}, {h4['gamma_bootstrap_95_ci'][1]:.6f}] | nominal: {h4['verdict']}; family: {h4.get('family_adjusted_verdict', h4['verdict'])} |
| H5 (exploratory) | tied-argmax envelope = [{h5['argmax_envelope_95_ci'][0]:g}, {h5['argmax_envelope_95_ci'][1]:g}]; quadratic b = {h5['quadratic_log_k_squared_coefficient']:+.6f} [{h5['quadratic_bootstrap_95_ci'][0]:+.6f}, {h5['quadratic_bootstrap_95_ci'][1]:+.6f}] | {h5['verdict']} |

H2 has no preregistered global test. Its p-value is absent; `p=1` appears only as a conservative placeholder in the conditional four-slot Holm table. The mapping preserves the difference between a confidence interval excluding zero in either direction and a useful positive D-C signal. Under both uniform Wilson/MOVER conventions (target-only and repeated-attempt ICC), the primary method's positive joint point at k=6 disappears.

The required log-normal H4 sensitivity gives gamma = {h4['lognormal_distribution_sensitivity']['gamma']:.6f} [{h4['lognormal_distribution_sensitivity']['gamma_bootstrap_95_ci'][0]:.6f}, {h4['lognormal_distribution_sensitivity']['gamma_bootstrap_95_ci'][1]:.6f}].

## Interpretation limits

H5 remains exploratory and underpowered. H4 is conditional on the registered Weibull working distribution; a gamma interval containing 1 would fail to reject proportionality rather than establish equivalence. This analysis does not estimate NLL/AUC/ROC, empirical epsilon, or any claim beyond H1--H5.
"""


def run_analysis(
    *,
    attempts_dir: Path,
    manifests_dir: Path,
    target_manifest: Path,
    output_dir: Path,
    expected_formal_commit: str,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_target_anchor: Mapping[str, str] = REGISTERED_TARGET_ANCHOR,
) -> Dict[str, Path]:
    _require(
        n_bootstrap == BOOTSTRAP_REPLICATES,
        f"formal analysis requires exactly {BOOTSTRAP_REPLICATES} bootstrap replicates",
    )
    _require(
        bootstrap_seed == BOOTSTRAP_SEED,
        f"formal analysis requires bootstrap seed {BOOTSTRAP_SEED}",
    )
    repo_root = Path(__file__).resolve().parent
    attempts, gate = validate_and_load(
        attempts_dir=attempts_dir,
        manifests_dir=manifests_dir,
        target_manifest=target_manifest,
        expected_formal_commit=expected_formal_commit,
        expected_target_anchor=expected_target_anchor,
    )
    k_grid = [int(value) for value in gate["k_grid"]]
    bootstrap = joint_person_bootstrap(
        attempts,
        k_grid=k_grid,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    curve_rows, curve_csv = build_curve(attempts, bootstrap, k_grid)
    hypotheses = analyze_curve_hypotheses(curve_rows, bootstrap, k_grid)
    kmin = build_kmin(attempts, [k for k in k_grid if k > 0])
    hypotheses["H4"] = analyze_h4(
        kmin,
        bootstrap["indices"]["control"],
        repo_root=repo_root,
    )
    holm = holm_conditional(hypotheses)
    hypotheses["H1"]["family_adjusted_verdict"] = (
        "supported"
        if hypotheses["H1"]["verdict"] == "supported"
        and holm["results"]["H1"]["reject_at_0_05"]
        else "not_supported_after_conditional_holm"
    )
    hypotheses["H2"]["family_adjusted_verdict"] = "unresolved_global_test_undefined"
    hypotheses["H3"]["family_adjusted_verdict"] = (
        hypotheses["H3"]["verdict"]
        if hypotheses["H3"]["verdict"] != "does_not_reject_zero"
        and holm["results"]["H3"]["reject_at_0_05"]
        else "does_not_reject_zero_after_conditional_holm"
    )
    if hypotheses["H4"]["verdict"] == "proportionality_refuted":
        hypotheses["H4"]["family_adjusted_verdict"] = (
            "proportionality_refuted"
            if holm["results"]["H4"]["reject_at_0_05"]
            else "nominal_ci_excludes_one_but_not_significant_after_conditional_holm"
        )
    else:
        hypotheses["H4"]["family_adjusted_verdict"] = "proportionality_not_rejected"
    payload = {
        "schema": "e3b-h1-h5-analysis-v1",
        "status": "accepted_formal_e3b_h1_h5_analysis",
        "gate": gate,
        "bootstrap": {
            "method": (
                "two independent person blocks (C and D); one draw per block reused across all k; "
                "both fields and all three fixed attack seeds retained"
            ),
            "replicates": n_bootstrap,
            "seed": bootstrap_seed,
            "n_people_per_block": {arm: len(values) for arm, values in bootstrap["people"].items()},
        },
        "curve": curve_rows,
        "hypotheses": hypotheses,
        "holm": holm,
        "scope_exclusions": [
            "NLL/AUC/ROC",
            "empirical epsilon",
            "H5 as confirmatory",
            (
                "legacy level-normal Tobit, complete-case linear, and direct median H/k diagnostics; "
                "these are not H1-H5 decision rules and remain in the prior audited reanalysis"
            ),
        ],
    }
    payload = _clean(payload)
    sensitivity_conventions = payload["hypotheses"]["H2"][
        "consistent_interval_sensitivity"
    ]["conventions"]
    sensitivity_by_name_and_tolerance = {
        convention["name"]: {row["tolerance"]: row for row in convention["mapping"]}
        for convention in sensitivity_conventions
    }
    h2_csv_rows = []
    for row in payload["hypotheses"]["H2"]["joint_mapping"]:
        output_row = dict(row)
        for name, by_tolerance in sensitivity_by_name_and_tolerance.items():
            sensitivity = by_tolerance[row["tolerance"]]
            for key in (
                "floor_only_capacities",
                "joint_any_direction_capacities",
                "joint_positive_signal_capacities",
                "joint_negative_signal_capacities",
            ):
                output_row[f"{name}_{key}"] = sensitivity[key]
        h2_csv_rows.append(output_row)
    h2_csv = pd.DataFrame(h2_csv_rows)
    summary_csv = _hypothesis_summary(payload["hypotheses"], payload["holm"])

    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        OUTPUT_JSON,
        OUTPUT_CURVE_CSV,
        OUTPUT_H2_CSV,
        OUTPUT_KMIN_CSV,
        OUTPUT_SUMMARY_CSV,
        OUTPUT_PNG,
        OUTPUT_PDF,
        OUTPUT_MD,
    )
    with tempfile.TemporaryDirectory(prefix=".e3b-h1-h5-", dir=output_dir) as staging_name:
        staging = Path(staging_name)
        (staging / OUTPUT_JSON).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        curve_csv.to_csv(staging / OUTPUT_CURVE_CSV, index=False)
        h2_csv.to_csv(staging / OUTPUT_H2_CSV, index=False)
        kmin.assign(
            upper_k=kmin["upper_k"].map(lambda value: "inf" if not np.isfinite(value) else value)
        ).to_csv(staging / OUTPUT_KMIN_CSV, index=False)
        summary_csv.to_csv(staging / OUTPUT_SUMMARY_CSV, index=False)
        _write_figure(curve_csv, staging)
        (staging / OUTPUT_MD).write_text(_analysis_markdown(payload))
        for filename in filenames:
            (staging / filename).replace(output_dir / filename)
    return {name: output_dir / filename for name, filename in (
        ("json", OUTPUT_JSON),
        ("curve_csv", OUTPUT_CURVE_CSV),
        ("h2_csv", OUTPUT_H2_CSV),
        ("kmin_csv", OUTPUT_KMIN_CSV),
        ("summary_csv", OUTPUT_SUMMARY_CSV),
        ("png", OUTPUT_PNG),
        ("pdf", OUTPUT_PDF),
        ("analysis", OUTPUT_MD),
    )}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-formal-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        paths = run_analysis(
            attempts_dir=args.attempts_dir,
            manifests_dir=args.manifests_dir,
            target_manifest=args.target_manifest,
            output_dir=args.output_dir,
            expected_formal_commit=args.expected_formal_commit,
            n_bootstrap=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
    except AnalysisGateError as exc:
        raise SystemExit(f"E3b H1-H5 analysis blocked: {exc}") from exc
    print("E3b H1-H5 analysis passed all gates")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
