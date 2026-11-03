# Project Lab: AI Equity Factor Lens

A reproducible research workflow for testing how market, macro, adjacent
infrastructure, and AI-close factor blocks explain returns in an AI equity
basket.

## Structure

```text
01_data_download.ipynb       # download and quality-assure raw market data
02_data_cleaning.ipynb       # clean prices and construct excess returns
03_factor_correlation.ipynb  # construct, screen, and assess factor blocks
04_linear_model.ipynb        # rolling factor model, attribution, placebo test
src/factor_lens/             # reusable Python implementation
```

The notebooks are intentionally short entry points. Reusable functions and
configuration live in `src/factor_lens/`, one module per notebook stage:
`data_download.py`, `cleaning.py`, `factor_correlation.py`, and
`linear_model.py`.

## Running the project

1. Open the repository root in Jupyter or Google Colab and run the notebooks
   in numerical order.
2. For local use, optionally install the package with `pip install -e .`.
3. In Colab, the notebooks add the local `src/` directory automatically.

Each module keeps its run configuration near the top. Change dates, cache
versions, factor sets, and model settings there rather than changing internal
functions.

The final notebook includes an AI-specificity placebo check. It compares the
incremental holdout fit of the AI-close extension for pre-declared AI-exposed
stocks versus non-AI control companies. It is evidence about specificity, not
causal proof.
