"""Fit and evaluate per-stock models using preselected orthogonal factors."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import factor_correlation, factor_selection


EST_START = factor_selection.EST_START
EST_END = factor_selection.EST_END
HOLD_START = factor_correlation.HOLD_START
HOLD_END = factor_selection.HOLD_END
MIN_TRAIN_OBSERVATIONS = factor_selection.MIN_TRAIN_OBSERVATIONS
MIN_HOLDOUT_OBSERVATIONS = 30
AI_CLOSE_STOCKS = ["NVDA", "AVGO", "MSFT", "AMZN", "GOOGL", "META", "ORCL"]
NON_AI_CONTROL_STOCKS = factor_selection.CONTROL_STOCKS


def _fit_ols(y, factors):
    design = np.column_stack([np.ones(len(y)), factors])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return coefficients, design


def _r_squared(actual, predicted, baseline):
    denominator = float(np.sum((actual - baseline) ** 2))
    if denominator <= 0:
        return np.nan
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def run_models(selection_results, min_train_observations=MIN_TRAIN_OBSERVATIONS):
    """Fit each stock's chosen factor set on the estimation window only."""
    summaries, coefficients, predictions, skipped = [], [], [], []

    for target in selection_results["selection"]["target"].tolist():
        frame = factor_selection.regression_frame(selection_results, target)
        train = frame.loc[EST_START:EST_END]
        holdout = frame.loc[HOLD_START:HOLD_END]
        factors = [name for name in frame.columns if name != "target_excess_return"]

        if len(train) < min_train_observations:
            skipped.append({
                "target": target,
                "reason": f"only {len(train)} complete estimation rows",
            })
            continue
        if not factors or factors[0] != "MKT":
            raise ValueError(f"{target}: selected factor order must start with MKT.")

        y_train = train["target_excess_return"].to_numpy(dtype=float)
        X_train = train[factors].to_numpy(dtype=float)
        model_beta, model_design = _fit_ols(y_train, X_train)
        train_prediction = model_design @ model_beta
        train_mean = float(y_train.mean())

        all_X = frame[factors].to_numpy(dtype=float)
        all_design = np.column_stack([np.ones(len(frame)), all_X])
        selected_prediction = all_design @ model_beta
        all_contributions = all_X * model_beta[None, 1:]

        market_beta, _ = _fit_ols(y_train, train[["MKT"]].to_numpy(dtype=float))
        market_prediction = market_beta[0] + frame["MKT"].to_numpy(dtype=float) * market_beta[1]

        holdout_rows = frame.index.isin(holdout.index)
        holdout_actual = frame.loc[holdout.index, "target_excess_return"].to_numpy(dtype=float)
        selected_holdout = selected_prediction[holdout_rows]
        market_holdout = market_prediction[holdout_rows]
        enough_holdout = len(holdout) >= MIN_HOLDOUT_OBSERVATIONS

        model_oos_r2 = (
            _r_squared(holdout_actual, selected_holdout, train_mean)
            if enough_holdout else np.nan
        )
        market_oos_r2 = (
            _r_squared(holdout_actual, market_holdout, train_mean)
            if enough_holdout else np.nan
        )

        train_mask = frame.index.isin(train.index)
        train_rows = pd.DataFrame({
            "date": train.index,
            "target": target,
            "split": "Train",
            "actual_excess_return": y_train,
            "selected_prediction": train_prediction,
            "market_only_prediction": market_prediction[train_mask],
            "intercept_contribution": model_beta[0],
            "residual": y_train - train_prediction,
        })
        for index, factor in enumerate(factors):
            train_rows[f"contribution_{factor}"] = all_contributions[train_mask, index]
        later = frame.loc[frame.index > pd.Timestamp(EST_END)]
        later_mask = frame.index.isin(later.index)
        later_rows = pd.DataFrame({
            "date": later.index,
            "target": target,
            "split": "Holdout",
            "actual_excess_return": later["target_excess_return"].to_numpy(dtype=float),
            "selected_prediction": selected_prediction[later_mask],
            "market_only_prediction": market_prediction[later_mask],
            "intercept_contribution": model_beta[0],
            "residual": later["target_excess_return"].to_numpy(dtype=float)
            - selected_prediction[later_mask],
        })
        for index, factor in enumerate(factors):
            later_rows[f"contribution_{factor}"] = all_contributions[later_mask, index]
        predictions.extend([train_rows, later_rows])

        summaries.append({
            "target": target,
            "n_factors": len(factors),
            "factor_order": json.dumps(factors),
            "training_observations": len(train),
            "training_r2": _r_squared(y_train, train_prediction, train_mean),
            "holdout_observations": len(holdout),
            "holdout_r2_selected": model_oos_r2,
            "holdout_r2_market_only": market_oos_r2,
            "holdout_r2_increment": model_oos_r2 - market_oos_r2,
            "holdout_correlation": (
                float(np.corrcoef(holdout_actual, selected_holdout)[0, 1])
                if enough_holdout and np.std(holdout_actual) > 0
                and np.std(selected_holdout) > 0 else np.nan
            ),
        })
        coefficients.append(
            {"target": target, "factor": "Intercept", "coefficient": float(model_beta[0])}
        )
        coefficients.extend(
            {"target": target, "factor": factor, "coefficient": float(model_beta[index + 1])}
            for index, factor in enumerate(factors)
        )

    if not summaries:
        raise ValueError("No stock had enough complete training observations to fit a model.")

    return {
        "summary": pd.DataFrame(summaries).sort_values("target").reset_index(drop=True),
        "coefficients": pd.DataFrame(coefficients),
        "predictions": pd.concat(predictions, ignore_index=True).sort_values(
            ["target", "date"]
        ).reset_index(drop=True),
        "skipped": pd.DataFrame(skipped, columns=["target", "reason"]),
        "metadata": {
            "estimation_start": EST_START,
            "estimation_end": EST_END,
            "holdout_start": HOLD_START,
            "holdout_end": HOLD_END,
            "minimum_holdout_observations_for_metrics": MIN_HOLDOUT_OBSERVATIONS,
            "model_type": "static OLS; selected factor sets and orthogonalization fixed before regression",
        },
    }


def plot_performance(results):
    """Compare holdout R-squared for selected factors and market-only benchmark."""
    summary = results["summary"].dropna(
        subset=["holdout_r2_selected", "holdout_r2_market_only"]
    )
    if summary.empty:
        raise ValueError("No stocks have enough holdout observations to plot performance.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot(
        [summary["holdout_r2_market_only"], summary["holdout_r2_selected"]],
        tick_labels=["Market only", "Selected factors"],
        showfliers=False,
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Holdout R-squared vs training-period mean")
    axes[0].set_title("Holdout performance across stocks")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(
        summary["holdout_r2_market_only"],
        summary["holdout_r2_selected"],
        alpha=0.75,
        color="#4477AA",
    )
    limits = [
        min(summary["holdout_r2_market_only"].min(), summary["holdout_r2_selected"].min()),
        max(summary["holdout_r2_market_only"].max(), summary["holdout_r2_selected"].max()),
    ]
    axes[1].plot(limits, limits, "k--", linewidth=1)
    axes[1].axhline(0, color="grey", linewidth=0.7)
    axes[1].axvline(0, color="grey", linewidth=0.7)
    axes[1].set_xlabel("Market-only holdout R-squared")
    axes[1].set_ylabel("Selected-set holdout R-squared")
    axes[1].set_title("Each point is a stock")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_factor_selection(selection_results):
    """Show how many selected stock models include each factor and model-size counts."""
    selection = selection_results["selection"].copy()
    factor_order = selection["factor_order"].map(json.loads)
    frequency = pd.Series(
        [factor for ordered in factor_order for factor in ordered]
    ).value_counts().sort_values()
    model_sizes = selection["n_factors"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(frequency.index, frequency.values, color="#228833")
    axes[0].set_title("Factors selected across stock models")
    axes[0].set_xlabel("Number of stocks")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].bar(model_sizes.index.astype(str), model_sizes.values, color="#4477AA")
    axes[1].set_title("Selected model sizes")
    axes[1].set_xlabel("Factors per stock, including MKT")
    axes[1].set_ylabel("Number of stocks")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()


def plot_exposure_heatmap(results, selection_results):
    """Plot training-window effect per one-SD orthogonal-factor move, in bp/day."""
    factors = selection_results["orthogonalized"].copy()
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.loc[
        factors["date"].between(pd.Timestamp(EST_START), pd.Timestamp(EST_END))
    ]
    standard_deviation = factors.groupby(["target", "factor"])[
        "orthogonalized_factor"
    ].std()
    effects = results["coefficients"].loc[
        results["coefficients"]["factor"] != "Intercept"
    ].copy()
    effects["train_sd"] = [
        standard_deviation.get((row.target, row.factor), np.nan)
        for row in effects.itertuples()
    ]
    effects["effect_1sd_bp"] = (
        effects["coefficient"] * effects["train_sd"] * 10_000
    )
    matrix = effects.pivot(
        index="target", columns="factor", values="effect_1sd_bp"
    )
    values = matrix.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    if limit <= 0:
        limit = 1.0

    fig, ax = plt.subplots(
        figsize=(max(8, 0.8 * len(matrix.columns) + 3), max(6, 0.24 * len(matrix) + 2))
    )
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title("Training exposures: effect of a 1-SD factor move (bp/day)")
    fig.colorbar(image, ax=ax, fraction=0.04, label="Basis points per day")
    fig.tight_layout()
    plt.show()


def plot_ai_specificity(results):
    """Compare selected-model holdout gains for AI-close stocks and controls (descriptive)."""
    summary = results["summary"].copy()
    summary["group"] = np.select(
        [summary["target"].isin(AI_CLOSE_STOCKS),
         summary["target"].isin(NON_AI_CONTROL_STOCKS)],
        ["AI-close", "Non-AI controls"],
        default="Other modeled stocks",
    )
    comparison = summary.loc[
        summary["group"].isin(["AI-close", "Non-AI controls"])
    ].dropna(subset=["holdout_r2_increment"])
    groups = [
        comparison.loc[comparison["group"] == name, "holdout_r2_increment"].to_numpy()
        for name in ("AI-close", "Non-AI controls")
    ]
    if any(len(values) == 0 for values in groups):
        raise ValueError("Both AI-close and non-AI control stocks are needed for this comparison.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=["AI-close", "Non-AI controls"], showfliers=False)
    for index, values in enumerate(groups, start=1):
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0]
        ax.scatter(index + jitter, values, alpha=0.75, s=35)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Holdout R-squared gain over market-only model")
    ax.set_title("Selected factors: AI-close vs non-AI controls")
    ax.text(
        0.01,
        0.01,
        "Descriptive comparison of individual stocks; not a causal or bootstrap test.",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()
    return comparison


def _carino_link(components, daily_returns):
    """Link daily return contributions to the compounded monthly return."""
    total_return = float(np.prod(1.0 + daily_returns) - 1.0)
    daily_factor = np.ones_like(daily_returns, dtype=float)
    nonzero = np.abs(daily_returns) >= 1e-12
    daily_factor[nonzero] = np.log1p(daily_returns[nonzero]) / daily_returns[nonzero]
    total_factor = (
        np.log1p(total_return) / total_return
        if abs(total_return) >= 1e-12 else 1.0
    )
    return (daily_factor[:, None] / total_factor * components).sum(axis=0), total_return


def plot_stock_attribution(results, target):
    """Show Carino-linked monthly contributions and holdout component variance shares."""
    stock = results["predictions"].loc[
        (results["predictions"]["target"] == target)
        & (results["predictions"]["split"] == "Holdout")
    ].sort_values("date").copy()
    if stock.empty:
        raise KeyError(f"No holdout predictions found for {target!r}.")

    contribution_columns = sorted(
        column for column in stock.columns if column.startswith("contribution_")
    )
    component_columns = ["intercept_contribution", *contribution_columns, "residual"]
    stock["date"] = pd.to_datetime(stock["date"])
    months = stock["date"].dt.to_period("M")
    monthly_rows = []
    for month in months.unique():
        subset = stock.loc[months == month]
        if len(subset) < 5:
            continue
        linked, total_return = _carino_link(
            subset[component_columns].to_numpy(dtype=float),
            subset["actual_excess_return"].to_numpy(dtype=float),
        )
        monthly_rows.append((month, *linked, total_return))

    if not monthly_rows:
        raise ValueError(f"Not enough holdout observations for monthly attribution of {target}.")

    monthly = pd.DataFrame(
        monthly_rows,
        columns=["month", *component_columns, "total_return"],
    ).set_index("month")
    fig, ax = plt.subplots(figsize=(13, 5))
    positive = np.zeros(len(monthly))
    negative = np.zeros(len(monthly))
    for component in component_columns:
        values = monthly[component].to_numpy()
        ax.bar(
            monthly.index.astype(str),
            np.where(values > 0, values, 0),
            bottom=positive,
            label=component.replace("contribution_", ""),
        )
        ax.bar(
            monthly.index.astype(str),
            np.where(values < 0, values, 0),
            bottom=negative,
        )
        positive += np.where(values > 0, values, 0)
        negative += np.where(values < 0, values, 0)
    ax.plot(
        range(len(monthly)),
        monthly["total_return"],
        "D",
        color="black",
        markersize=4,
        label="Total excess return",
    )
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_title(f"{target}: monthly holdout excess-return attribution")
    ax.set_ylabel("Return contribution")
    ax.tick_params(axis="x", labelrotation=60)
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.show()

    recent = stock.tail(252)
    actual = recent["actual_excess_return"].to_numpy(dtype=float)
    variance = float(np.var(actual, ddof=1))
    shares = {}
    if variance > 0:
        for component in component_columns:
            values = recent[component].to_numpy(dtype=float)
            shares[component] = float(np.cov(values, actual, ddof=1)[0, 1] / variance)
    share_series = pd.Series(shares).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(share_series))))
    ax.barh(share_series.index, share_series.values, color="#4477AA")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Share of target-return variance (covariance decomposition)")
    ax.set_title(f"{target}: holdout component variance shares (latest 252 observations)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    plt.show()
    return monthly, share_series


def plot_prediction_diagnostics(results, target):
    """Render a holdout actual/prediction and residual-diagnostic dashboard."""
    data = results["predictions"]
    stock = data.loc[
        (data["target"] == target) & (data["split"] == "Holdout")
    ].sort_values("date").copy()
    if len(stock) < MIN_HOLDOUT_OBSERVATIONS:
        raise ValueError(f"Not enough holdout predictions for {target!r}.")

    actual = stock["actual_excess_return"]
    predicted = stock["selected_prediction"]
    residual = stock["residual"]
    denominator = float(((actual - actual.mean()) ** 2).sum())
    r_squared = (
        float(1 - (residual ** 2).sum() / denominator)
        if denominator > 0 else np.nan
    )
    correlation = float(actual.corr(predicted))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    limit = max(actual.abs().max(), predicted.abs().max()) * 1.08 * 100
    axes[0].scatter(actual * 100, predicted * 100, s=12, alpha=0.4, color="#4477AA")
    axes[0].plot([-limit, limit], [-limit, limit], "k--", linewidth=1)
    axes[0].set_xlabel("Actual daily excess return (%)")
    axes[0].set_ylabel("Predicted daily excess return (%)")
    axes[0].set_title(f"Holdout: R²={r_squared:.3f}, r={correlation:.3f}")
    axes[0].grid(alpha=0.2)

    dates = pd.to_datetime(stock["date"])
    actual_cumulative = np.cumprod(1.0 + actual.to_numpy()) - 1.0
    predicted_cumulative = np.cumprod(1.0 + predicted.to_numpy()) - 1.0
    axes[1].plot(dates, actual_cumulative * 100, label="Actual", color="#4477AA")
    axes[1].plot(
        dates,
        predicted_cumulative * 100,
        label="Model-implied",
        color="#CC6677",
    )
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Illustrative cumulative excess return (%)")
    axes[1].set_title("Not a trading backtest")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    rolling_volatility = residual.rolling(21).std() * np.sqrt(252) * 100
    axes[2].plot(dates, rolling_volatility, color="#CC6677")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Annualized residual volatility (%)")
    axes[2].set_title("21-day rolling unexplained volatility")
    axes[2].grid(alpha=0.2)
    fig.suptitle(f"{target}: selected-factor holdout diagnostics")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residual * 100, bins=40, density=True, color="#4477AA", alpha=0.75)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Daily holdout residual (%)")
    ax.set_title(f"{target}: holdout residual distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
    return stock


def plot_stock_diagnostics(results, target):
    """Plot actual-versus-predicted, cumulative returns, and residual diagnostics."""
    data = results["predictions"]
    stock = data.loc[data["target"] == target].copy()
    if stock.empty:
        raise KeyError(f"No fitted predictions found for {target!r}.")

    holdout = stock.loc[stock["split"] == "Holdout"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for split, points, color in (
        ("Train", stock.loc[stock["split"] == "Train"], "#4477AA"),
        ("Holdout", holdout, "#CC6677"),
    ):
        axes[0, 0].scatter(
            points["actual_excess_return"] * 100,
            points["selected_prediction"] * 100,
            s=12,
            alpha=0.4,
            label=split,
            color=color,
            edgecolors="none",
        )
    limits = [
        min(stock["actual_excess_return"].min(), stock["selected_prediction"].min()) * 100,
        max(stock["actual_excess_return"].max(), stock["selected_prediction"].max()) * 100,
    ]
    axes[0, 0].plot(limits, limits, "k--", linewidth=1)
    axes[0, 0].set_xlabel("Actual daily excess return (%)")
    axes[0, 0].set_ylabel("Predicted daily excess return (%)")
    axes[0, 0].set_title(f"{target}: actual vs predicted")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    stock = stock.sort_values("date")
    axes[0, 1].plot(
        stock["date"],
        (np.cumprod(1.0 + stock["actual_excess_return"].to_numpy()) - 1.0) * 100,
        label="Actual",
        color="#4477AA",
    )
    axes[0, 1].plot(
        stock["date"],
        (np.cumprod(1.0 + stock["selected_prediction"].to_numpy()) - 1.0) * 100,
        label="Model-implied",
        color="#CC6677",
    )
    axes[0, 1].axvline(pd.Timestamp(HOLD_START), color="black", linestyle=":")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Illustrative cumulative excess return (%)")
    axes[0, 1].set_title("Model-implied path is not a trading backtest")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    residual = holdout["residual"].dropna()
    rolling_volatility = residual.rolling(21).std() * np.sqrt(252) * 100
    axes[1, 0].plot(holdout.loc[residual.index, "date"], rolling_volatility, color="#CC6677")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Annualized residual volatility (%)")
    axes[1, 0].set_title("21-day rolling unexplained volatility")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 1].hist(residual * 100, bins=40, density=True, color="#4477AA", alpha=0.75)
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("Daily residual (%)")
    axes[1, 1].set_title("Holdout residual distribution")
    axes[1, 1].grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
    return stock


def save_results(results, selection_results, output_dir=None):
    """Persist model summaries and predictions next to the stock selection."""
    if output_dir is None:
        output_dir = Path(selection_results["output_dir"]) / "regression"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results["summary"].to_csv(output_dir / "stock_model_summary.csv", index=False)
    results["coefficients"].to_csv(output_dir / "stock_model_coefficients.csv", index=False)
    results["predictions"].to_parquet(output_dir / "stock_model_predictions.parquet", index=False)
    results["skipped"].to_csv(output_dir / "skipped_model_targets.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(results["metadata"], indent=2, default=str)
    )
    results["output_dir"] = output_dir
    return output_dir
