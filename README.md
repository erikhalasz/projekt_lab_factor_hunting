# Project Lab: AI Equity Factor Lens

A reproducible research workflow for testing how market, macro, adjacent
infrastructure, and AI-close factor blocks explain returns in an AI equity
basket.

## Structure

```text
Notebooks/
  01_data_download.ipynb       # download and quality-assure raw market data
  02_data_cleaning.ipynb       # clean prices and construct excess returns
  03_factor_correlation.ipynb  # construct, screen, and assess factor blocks
  03b_factor_selection.ipynb   # select and orthogonalize factors per stock
  04_linear_model.ipynb        # rolling factor model, attribution, placebo test
  04b_stock_specific_model.ipynb # regress each stock on its selected factors
src/factor_lens/               # reusable Python implementation
```

The notebooks are intentionally short entry points. Reusable functions and
configuration live in `src/factor_lens/`: `data_download.py`, `cleaning.py`,
`factor_correlation.py`, `factor_selection.py`, `linear_model.py`, and
`stock_specific_model.py`.

## Running the project

1. Open the repository root in Jupyter or Google Colab. Run
   `Notebooks/01`, `Notebooks/02`, `Notebooks/03`, and `Notebooks/03b` in order.
   Then use either `Notebooks/04` for the predefined rolling models or
   `Notebooks/04b` for stock-specific selected-factor regressions.
2. For local use, optionally install the package with `pip install -e .`.
3. In Colab, the notebooks add the local `src/` directory automatically.

Each module keeps its run configuration near the top. Change dates, cache
versions, factor sets, and model settings there rather than changing internal
functions.

Notebook `03b` uses notebook `03`'s accepted `C_all` factor shortlist as its
candidate pool. It chooses an ordered factor set separately for each stock by
forward BIC selection, with `MKT` fixed first, then fits orthogonalization
projections using only the shared 2017-01-03–2023-12-29 estimation window.
It saves those projections and the resulting orthogonalized factor returns for
use in regression; the 2024+ holdout is not used in selection or transformation
fitting.

Notebook `04b` reads the stock-specific factor sets from `03b`, fits static
OLS regressions on the same estimation window, and evaluates them on the
untouched holdout against a market-only benchmark. It complements rather than
replaces notebook `04`.

The final notebook includes an AI-specificity placebo check. It compares the
incremental holdout fit of the AI-close extension for pre-declared AI-exposed
stocks versus non-AI control companies. It is evidence about specificity, not
causal proof.
