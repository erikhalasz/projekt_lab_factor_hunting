from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "adatok"
PRICE_PATH = DATA_DIR / "cleaned_ai_factor_data.csv"
SUMMARY_PATH = DATA_DIR / "orthogonal_factor_model_summary.csv"
STEPS_PATH = DATA_DIR / "orthogonal_factor_model_steps.csv"
PREDICTIONS_PATH = DATA_DIR / "orthogonal_factor_model_predictions.csv"

BASE_FACTORS = ["SPY", "GLD", "TLT", "USO", "UUP"]
THEMATIC_FACTORS = [
	"XLU", "UTES", "GRID", "PAVE",
	"COPX", "COPP", "COPA", "URA", "URNM", "NUUR", "LIT",
	"XLP", "XLV", "DBA",
	"SMH", "SOXX", "SOXQ", "CHAT", "THNQ", "WTAI", "IDGT", "AIQ", "SRVR",
	"IGV", "SKYY", "QTUM", "IVES", "IGPT", "CIBR", "HACK", "BUG",
]

TARGET_STOCKS = [
	"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "AMD", "INTC", "QCOM",
	"TSM", "ASML", "AMAT", "LRCX", "KLAC", "MU", "MRVL", "SNPS", "CDNS", "ORCL",
	"IBM", "CRM", "NOW", "ADBE", "INTU",
	"DLR", "EQIX", "AMT", "CCI", "ANET", "CSCO", "ETN", "EMR", "PWR", "HUBB",
	"MOD", "MSI", "VRSN", "AKAM", "STX",
	"AES", "AEP", "DUK", "ED", "EXC", "NEE", "PCG", "PEG", "SO", "SRE",
	"XEL", "AWK", "ETR", "EVRG", "WEC",
	"CAT", "DE", "GE", "HON", "LMT", "RTX", "BA", "UPS", "UNP", "URI",
	"PH", "FCX", "NEM", "BHP", "RIO", "XOM", "CVX", "COP", "SLB", "HAL",
	"JPM", "BAC", "GS", "V", "MA", "JNJ", "UNH", "LLY", "ABBV", "PFE",
	"PG", "KO", "PEP", "WMT", "COST", "MCD", "DIS", "NKE", "HD", "LOW",
	"SBUX", "NFLX", "TSLA", "BX", "C",
]

VALIDATION_START = pd.Timestamp("2021-10-01")
TEST_START = pd.Timestamp("2023-04-01")
MIN_FACTOR_COVERAGE = 0.80
MIN_TRAIN_OBSERVATIONS = 200
MAX_ADDITIONAL_FACTORS = 8
P_VALUE_THRESHOLD = 0.05


def add_intercept(frame):
	return sm.add_constant(frame, has_constant="add")


def fit_orthogonal_factor(training, factor, orthogonal_basis):
	"""Remove the in-sample component explained by the factors already selected."""
	if orthogonal_basis.shape[1]:
		design = add_intercept(orthogonal_basis)
	else:
		design = pd.DataFrame({"const": 1.0}, index=training.index)

	projection = sm.OLS(training[factor], design).fit()
	orthogonal_values = training[factor] - projection.predict(design)
	transform = {
		"factor": factor,
		"intercept": float(projection.params["const"]),
		"coefficients": {
			previous: float(projection.params[previous])
			for previous in orthogonal_basis.columns
		},
	}
	return orthogonal_values.rename(factor), transform


def apply_orthogonal_transforms(raw_returns, transforms):
	"""Apply training-fitted transformations without refitting on later data."""
	orthogonal = pd.DataFrame(index=raw_returns.index)
	for transform in transforms:
		factor = transform["factor"]
		values = raw_returns[factor] - transform["intercept"]
		for previous, coefficient in transform["coefficients"].items():
			values = values - coefficient * orthogonal[previous]
		orthogonal[factor] = values
	return orthogonal


def score_period(returns, mask, target, split, model, transforms, training_mean):
	factors = [transform["factor"] for transform in transforms]
	period = returns.loc[mask, [target, *factors]].dropna()
	if period.empty:
		return (
			{"observations": 0, "r_squared": np.nan, "correlation": np.nan},
			pd.DataFrame(),
		)

	orthogonal = apply_orthogonal_transforms(period[factors], transforms)
	prediction = model.predict(add_intercept(orthogonal))
	actual = period[target]
	baseline_sse = float(((actual - training_mean) ** 2).sum())
	residual_sse = float(((actual - prediction) ** 2).sum())
	r_squared = 1 - residual_sse / baseline_sse if baseline_sse else np.nan
	correlation = actual.corr(prediction)
	prediction_frame = pd.DataFrame({
		"Date": period.index,
		"Target": target,
		"Split": split,
		"Actual_Log_Return": actual.to_numpy(),
		"Predicted_Log_Return": np.asarray(prediction),
	})
	prediction_frame["Residual"] = (
		prediction_frame["Actual_Log_Return"] - prediction_frame["Predicted_Log_Return"]
	)
	return (
		{
			"observations": len(period),
			"r_squared": r_squared,
			"correlation": correlation,
		},
		prediction_frame,
	)


def fit_target_model(returns, target, candidate_factors):
	factor_columns = [*BASE_FACTORS, *candidate_factors]
	training_columns = [target, *factor_columns]
	target_returns = returns.loc[
		returns.index < VALIDATION_START,
		training_columns,
	].dropna()

	if len(target_returns) < MIN_TRAIN_OBSERVATIONS:
		return None, None

	y_train = target_returns[target]
	orthogonal_basis = pd.DataFrame(index=target_returns.index)
	transforms = []
	step_rows = []
	previous_r_squared = 0.0

	for step, factor in enumerate(BASE_FACTORS, start=1):
		orthogonal_values, transform = fit_orthogonal_factor(
			target_returns, factor, orthogonal_basis
		)
		orthogonal_basis[factor] = orthogonal_values
		transforms.append(transform)

		model = sm.OLS(y_train, add_intercept(orthogonal_basis)).fit()
		step_rows.append({
			"Target": target,
			"Step": step,
			"Factor": factor,
			"Factor_Type": "Fixed macro",
			"P_Value": float(model.pvalues[factor]),
			"Orthogonal_Beta": float(model.params[factor]),
			"Partial_R2": np.nan,
			"Incremental_R2": float(model.rsquared - previous_r_squared),
			"Cumulative_Train_R2": float(model.rsquared),
		})
		previous_r_squared = float(model.rsquared)

	remaining = list(candidate_factors)
	selected_count = 0

	while remaining and selected_count < MAX_ADDITIONAL_FACTORS:
		current_model = sm.OLS(y_train, add_intercept(orthogonal_basis)).fit()
		current_residual = current_model.resid
		residual_sse = float((current_residual ** 2).sum())
		if residual_sse <= 0:
			break

		scored_candidates = []
		for factor in remaining:
			orthogonal_values, transform = fit_orthogonal_factor(
				target_returns, factor, orthogonal_basis
			)
			if orthogonal_values.std(ddof=0) < 1e-10:
				continue

			incremental_model = sm.OLS(current_residual, orthogonal_values).fit()
			beta = float(incremental_model.params.iloc[0])
			p_value = float(incremental_model.pvalues.iloc[0])
			updated_residual = current_residual - beta * orthogonal_values
			partial_r_squared = 1 - float((updated_residual ** 2).sum()) / residual_sse
			scored_candidates.append({
				"factor": factor,
				"orthogonal_values": orthogonal_values,
				"transform": transform,
				"beta": beta,
				"p_value": p_value,
				"partial_r_squared": partial_r_squared,
			})

		if not scored_candidates:
			break

		best = max(scored_candidates, key=lambda result: result["partial_r_squared"])
		if best["p_value"] >= P_VALUE_THRESHOLD:
			break

		factor = best["factor"]
		orthogonal_basis[factor] = best["orthogonal_values"]
		transforms.append(best["transform"])
		remaining.remove(factor)
		selected_count += 1

		updated_model = sm.OLS(y_train, add_intercept(orthogonal_basis)).fit()
		step_rows.append({
			"Target": target,
			"Step": len(step_rows) + 1,
			"Factor": factor,
			"Factor_Type": "Selected thematic",
			"P_Value": best["p_value"],
			"Orthogonal_Beta": best["beta"],
			"Partial_R2": best["partial_r_squared"],
			"Incremental_R2": float(updated_model.rsquared - previous_r_squared),
			"Cumulative_Train_R2": float(updated_model.rsquared),
		})
		previous_r_squared = float(updated_model.rsquared)

	final_model = sm.OLS(y_train, add_intercept(orthogonal_basis)).fit()
	training_mean = float(y_train.mean())
	training_prediction = final_model.predict(add_intercept(orthogonal_basis))
	training_predictions = pd.DataFrame({
		"Date": y_train.index,
		"Target": target,
		"Split": "Train",
		"Actual_Log_Return": y_train.to_numpy(),
		"Predicted_Log_Return": np.asarray(training_prediction),
	})
	training_predictions["Residual"] = (
		training_predictions["Actual_Log_Return"]
		- training_predictions["Predicted_Log_Return"]
	)

	validation, validation_predictions = score_period(
		returns,
		(returns.index >= VALIDATION_START) & (returns.index < TEST_START),
		target,
		"Validation",
		final_model,
		transforms,
		training_mean,
	)
	test, test_predictions = score_period(
		returns,
		returns.index >= TEST_START,
		target,
		"Test",
		final_model,
		transforms,
		training_mean,
	)
	predictions = pd.concat(
		[training_predictions, validation_predictions, test_predictions],
		ignore_index=True,
	)

	selected_factors = [row["Factor"] for row in step_rows if row["Factor_Type"] == "Selected thematic"]
	summary = {
		"Target": target,
		"Fixed_Factor_Count": len(BASE_FACTORS),
		"Additional_Factor_Count": len(selected_factors),
		"Total_Factor_Count": len(BASE_FACTORS) + len(selected_factors),
		"Selected_Additional_Factors": ", ".join(selected_factors),
		"Training_Observations": len(target_returns),
		"Training_Start": target_returns.index.min(),
		"Training_End": target_returns.index.max(),
		"Training_R2": float(final_model.rsquared),
		"Validation_Observations": validation["observations"],
		"Validation_R2": validation["r_squared"],
		"Validation_Correlation": validation["correlation"],
		"Test_Observations": test["observations"],
		"Test_R2": test["r_squared"],
		"Test_Correlation": test["correlation"],
	}
	return summary, step_rows, predictions


def main():
	if not PRICE_PATH.exists():
		raise FileNotFoundError(f"Price data not found: {PRICE_PATH}")

	prices = pd.read_csv(PRICE_PATH, index_col="Date", parse_dates=True)
	prices = prices.apply(pd.to_numeric, errors="coerce")
	log_returns = np.log(prices.where(prices > 0)).diff()

	training_returns = log_returns.loc[log_returns.index < VALIDATION_START]
	missing_base = [factor for factor in BASE_FACTORS if factor not in log_returns.columns]
	if missing_base:
		raise ValueError(f"Required base factors missing from price data: {missing_base}")

	available_candidates = [
		factor for factor in THEMATIC_FACTORS if factor in log_returns.columns
	]
	factor_coverage = training_returns[available_candidates].notna().mean()
	candidate_factors = [
		factor for factor in available_candidates
		if factor_coverage[factor] >= MIN_FACTOR_COVERAGE
	]
	excluded_factors = [
		factor for factor in available_candidates if factor not in candidate_factors
	]

	targets = [target for target in TARGET_STOCKS if target in log_returns.columns]
	print(f"Loaded {len(log_returns):,} daily log-return rows.")
	print(f"Thematic factors eligible in training: {len(candidate_factors)} of {len(THEMATIC_FACTORS)}")
	if excluded_factors:
		print(f"Excluded for limited training history: {', '.join(excluded_factors)}")
	print(f"Fitting models for {len(targets)} target assets...")

	summaries = []
	all_steps = []
	all_predictions = []
	skipped_targets = []
	for position, target in enumerate(targets, start=1):
		print(f"[{position}/{len(targets)}] {target}")
		summary, steps, predictions = fit_target_model(
			log_returns, target, candidate_factors
		)
		if summary is None:
			skipped_targets.append(target)
			continue
		summaries.append(summary)
		all_steps.extend(steps)
		all_predictions.append(predictions)

	if not summaries:
		raise RuntimeError("No target had enough complete training observations to fit a model.")

	summary_frame = pd.DataFrame(summaries).sort_values(
		["Additional_Factor_Count", "Target"], ascending=[True, True]
	)
	steps_frame = pd.DataFrame(all_steps).sort_values(["Target", "Step"])
	predictions_frame = pd.concat(all_predictions, ignore_index=True).sort_values(
		["Target", "Date"]
	)

	summary_frame.to_csv(SUMMARY_PATH, index=False, float_format="%.6f", encoding="utf-8-sig")
	steps_frame.to_csv(STEPS_PATH, index=False, float_format="%.6f", encoding="utf-8-sig")
	predictions_frame.to_csv(
		PREDICTIONS_PATH,
		index=False,
		float_format="%.8f",
		encoding="utf-8-sig",
	)

	print(f"\nModels fitted: {len(summary_frame)}")
	if skipped_targets:
		print(f"Skipped for insufficient training history: {', '.join(skipped_targets)}")
	print(f"Summary: {SUMMARY_PATH}")
	print(f"Step-by-step factors: {STEPS_PATH}")
	print(f"Predictions: {PREDICTIONS_PATH}")
	print("\nFactor count distribution:")
	print(summary_frame["Total_Factor_Count"].value_counts().sort_index().to_string())


if __name__ == "__main__":
	main()
