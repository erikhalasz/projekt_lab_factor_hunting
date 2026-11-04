"""Stock-specific factor selection and training-fitted orthogonalization."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import factor_correlation


EST_START = factor_correlation.EST_START
EST_END = factor_correlation.EST_END
HOLD_END = factor_correlation.HOLD_END
DEFAULT_CANDIDATE_SET = "C_all"
MIN_TRAIN_OBSERVATIONS = 200
MAX_FACTORS = factor_correlation.CFG["sl_k_max"]
MIN_BIC_IMPROVEMENT = 2.0
CONTROL_STOCKS = ["KO", "PG", "WMT", "MCD", "JNJ", "JPM", "XOM", "CAT"]


def candidate_factors(correlation_results, candidate_set=DEFAULT_CANDIDATE_SET):
    """Return correlation-screened candidates in their global shortlist order."""
    shortlist = correlation_results["shortlists_by_set"]
    if "model_set" not in shortlist or "decision" not in shortlist or "factor" not in shortlist:
        raise KeyError("Correlation results do not contain the expected shortlists_by_set table.")

    selected = shortlist.loc[
        (shortlist["model_set"] == candidate_set) & (shortlist["decision"] == "accepted"),
        "factor",
    ].tolist()
    if not selected:
        raise ValueError(f"No accepted factors found for correlation model set {candidate_set!r}.")

    available = set(correlation_results["factors_raw"].columns)
    missing = [factor for factor in selected if factor not in available]
    if missing:
        raise KeyError(f"Shortlisted factors missing from factors_raw: {missing}")
    return [factor for factor in selected if factor != "MKT"]


def default_stock_targets(correlation_results, returns_excess):
    """Use theme-basket constituents plus the project's declared non-AI controls."""
    members = correlation_results["basket_members"]
    names = members.loc[members["in_data"].astype(bool), "ticker"].drop_duplicates().tolist()
    names.extend(CONTROL_STOCKS)
    return [name for name in dict.fromkeys(names) if name in returns_excess.columns]


def _bic(y, factors):
    """BIC for an intercept plus the supplied factor columns."""
    design = np.column_stack([np.ones(len(y)), factors])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    sse = float(residual @ residual)
    if not np.isfinite(sse) or sse <= 0:
        return float("-inf")
    return float(len(y) * np.log(sse / len(y)) + design.shape[1] * np.log(len(y)))


def _orthogonalize_candidate(values, previous_factors):
    """Project a candidate on the existing ordered orthogonal basis."""
    if previous_factors.shape[1] == 0:
        intercept = float(values.mean())
        return values - intercept, intercept, {}

    design = np.column_stack([np.ones(len(values)), previous_factors.to_numpy()])
    coefficients = np.linalg.lstsq(design, values.to_numpy(), rcond=None)[0]
    orthogonal = values - design @ coefficients
    projection = {
        name: float(coefficients[index + 1])
        for index, name in enumerate(previous_factors.columns)
    }
    return orthogonal, float(coefficients[0]), projection


def _select_one(target, target_returns, factor_returns, candidates, max_factors,
                min_train_observations, min_bic_improvement):
    columns = ["MKT", *candidates]
    training = pd.concat(
        [target_returns.rename("target"), factor_returns[columns]],
        axis=1,
    ).loc[EST_START:EST_END].dropna()
    if len(training) < min_train_observations:
        return None, None, f"only {len(training)} complete training rows"

    y = training["target"].to_numpy(dtype=float)
    orthogonal = pd.DataFrame(
        {"MKT": training["MKT"].to_numpy(dtype=float)},
        index=training.index,
    )
    transforms = [{
        "target": target,
        "factor": "MKT",
        "order": 1,
        "intercept": 0.0,
        "coefficients": "{}",
    }]
    remaining = list(candidates)
    current_bic = _bic(y, orthogonal.to_numpy())

    while remaining and len(orthogonal.columns) < max_factors:
        best = None
        for factor in remaining:
            values = training[factor].astype(float)
            orthogonal_values, intercept, projection = _orthogonalize_candidate(
                values, orthogonal
            )
            if not np.isfinite(orthogonal_values.to_numpy()).all():
                continue
            if orthogonal_values.std(ddof=0) < 1e-12:
                continue

            trial = np.column_stack(
                [orthogonal.to_numpy(), orthogonal_values.to_numpy()]
            )
            score = _bic(y, trial)
            if best is None or score < best["bic"]:
                best = {
                    "factor": factor,
                    "values": orthogonal_values,
                    "bic": score,
                    "intercept": intercept,
                    "coefficients": projection,
                }

        if best is None or current_bic - best["bic"] < min_bic_improvement:
            break

        factor = best["factor"]
        orthogonal[factor] = best["values"]
        transforms.append({
            "target": target,
            "factor": factor,
            "order": len(transforms) + 1,
            "intercept": best["intercept"],
            "coefficients": json.dumps(best["coefficients"], sort_keys=True),
        })
        current_bic = best["bic"]
        remaining.remove(factor)

    selection = {
        "target": target,
        "n_train": len(training),
        "bic_market_only": _bic(y, training[["MKT"]].to_numpy()),
        "bic_selected": current_bic,
        "factor_order": json.dumps(orthogonal.columns.tolist()),
        "n_factors": len(orthogonal.columns),
    }
    return selection, transforms, None


def _apply_transforms(factor_returns, transforms):
    """Apply stored projections recursively; no parameters are refit here."""
    orthogonal = pd.DataFrame(index=factor_returns.index)
    for transform in transforms:
        factor = transform["factor"]
        coefficients = json.loads(transform["coefficients"])
        values = factor_returns[factor].astype(float) - float(transform["intercept"])
        for previous, coefficient in coefficients.items():
            values = values - float(coefficient) * orthogonal[previous]
        orthogonal[factor] = values
    return orthogonal


def run_selection(correlation_results, returns_excess, targets=None,
                  candidate_set=DEFAULT_CANDIDATE_SET,
                  max_factors=MAX_FACTORS,
                  min_train_observations=MIN_TRAIN_OBSERVATIONS,
                  min_bic_improvement=MIN_BIC_IMPROVEMENT):
    """Select and orthogonalize a factor order independently for each stock.

    Factor eligibility and the shared candidate pool come from the correlation
    notebook. Per-stock selection and projection coefficients use only the
    configured estimation window.
    """
    factors = correlation_results["factors_raw"]
    candidates = candidate_factors(correlation_results, candidate_set)
    if "MKT" not in factors:
        raise KeyError("The correlation results must contain the MKT factor.")
    if max_factors < 1:
        raise ValueError("max_factors must be at least one for the mandatory MKT factor.")
    if min_bic_improvement < 0:
        raise ValueError("min_bic_improvement cannot be negative.")

    targets = list(targets) if targets is not None else default_stock_targets(
        correlation_results, returns_excess
    )
    targets = list(dict.fromkeys(targets))
    missing_targets = [target for target in targets if target not in returns_excess]
    if missing_targets:
        raise KeyError(f"Target returns missing for: {missing_targets}")
    if not targets:
        raise ValueError("No stock targets with available excess-return data were found.")

    selection_rows, transform_rows, skipped = [], [], []
    orthogonalized = []
    date_index = factors.index.union(returns_excess.index).sort_values()
    factor_panel = factors.reindex(date_index)
    for target in targets:
        selection, transforms, reason = _select_one(
            target,
            returns_excess[target].reindex(date_index),
            factor_panel,
            candidates,
            max_factors,
            min_train_observations,
            min_bic_improvement,
        )
        if reason:
            skipped.append({"target": target, "reason": reason})
            continue

        selection_rows.append(selection)
        transform_rows.extend(transforms)
        ordered_transforms = sorted(transforms, key=lambda row: row["order"])
        transformed = _apply_transforms(factor_panel, ordered_transforms)
        transformed = transformed.loc[EST_START:HOLD_END]
        actual = returns_excess[target].reindex(transformed.index)
        long = transformed.rename_axis("date").stack().rename(
            "orthogonalized_factor"
        ).reset_index()
        long.columns = ["date", "factor", "orthogonalized_factor"]
        long.insert(0, "target", target)
        long["actual_excess_return"] = long["date"].map(actual)
        orthogonalized.append(long)

    if not selection_rows:
        raise ValueError("No targets had enough complete training observations.")

    return {
        "selection": pd.DataFrame(selection_rows).sort_values("target").reset_index(drop=True),
        "transforms": pd.DataFrame(transform_rows).sort_values(
            ["target", "order"]
        ).reset_index(drop=True),
        "orthogonalized": pd.concat(orthogonalized, ignore_index=True),
        "skipped": pd.DataFrame(skipped, columns=["target", "reason"]),
        "metadata": {
            "candidate_set": candidate_set,
            "candidate_factors": candidates,
            "estimation_start": EST_START,
            "estimation_end": EST_END,
            "holdout_end": HOLD_END,
            "max_factors_including_market": max_factors,
            "minimum_bic_improvement": min_bic_improvement,
        },
    }


def regression_frame(results, target):
    """Return one stock's target returns and selected orthogonal factors, date-indexed."""
    selection = results["selection"].set_index("target")
    if target not in selection.index:
        raise KeyError(f"No selected factor set found for target {target!r}.")
    order = json.loads(selection.at[target, "factor_order"])
    data = results["orthogonalized"].loc[
        results["orthogonalized"]["target"] == target
    ]
    factors = data.pivot(
        index="date", columns="factor", values="orthogonalized_factor"
    ).reindex(columns=order)
    target_return = data.drop_duplicates("date").set_index("date")[
        "actual_excess_return"
    ].rename("target_excess_return")
    return pd.concat([target_return, factors], axis=1).dropna()


def save_results(results, correlation_results, output_dir=None):
    """Persist selection, projections, and regression-ready factors."""
    if output_dir is None:
        correlation_root = Path(correlation_results["_out_root"])
        output_dir = correlation_root / "stock_specific_selection_v1"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results["selection"].to_csv(output_dir / "stock_factor_selection.csv", index=False)
    results["transforms"].to_csv(output_dir / "orthogonalization_transforms.csv", index=False)
    results["orthogonalized"].to_parquet(output_dir / "orthogonalized_factors.parquet", index=False)
    results["skipped"].to_csv(output_dir / "skipped_targets.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(results["metadata"], indent=2, default=str)
    )
    results["output_dir"] = output_dir
    return output_dir


def load_results(output_dir):
    """Load outputs previously written by save_results."""
    output_dir = Path(output_dir)
    return {
        "selection": pd.read_csv(output_dir / "stock_factor_selection.csv"),
        "transforms": pd.read_csv(output_dir / "orthogonalization_transforms.csv"),
        "orthogonalized": pd.read_parquet(output_dir / "orthogonalized_factors.parquet"),
        "skipped": pd.read_csv(output_dir / "skipped_targets.csv"),
        "metadata": json.loads((output_dir / "metadata.json").read_text()),
        "output_dir": output_dir,
    }
