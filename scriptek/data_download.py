import json
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "adatok"
METADATA_PATH = DATA_DIR / "project_metadata.json"
OUTPUT_PATH = DATA_DIR / "cleaned_ai_factor_data.csv"
START_DATE = "2016-01-01"
BATCH_SIZE = 50

EXTRA_ETFS = {
    "SPY", "IWN", "USMV", "TLT", "IEF", "SHY", "TIP", "LQD", "HYG",
    "UUP", "EEM", "EFA", "VNQ", "XLP", "XLV", "XLU", "UTES", "KRE",
    "XLF", "XLE", "PAVE", "PDBC", "GLD", "XAG", "DBA", "USO", "COPX",
    "COPP", "COPA", "URA", "URNM", "NUUR", "LIT", "SMH", "SOXX", "SOXQ",
    "CHAT", "THNQ", "WTAI", "IDGT", "AIQ", "SRVR", "IGV", "SKYY", "QTUM",
    "IVES", "IGPT", "CIBR", "HACK", "BUG", "GRID", "BOTZ",
}


def load_metadata_tickers():
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    tickers = set()
    for block in metadata.get("industry_blocks", []):
        for key in ("example_tickers", "etf_proxies"):
            tickers.update(str(ticker).strip().upper() for ticker in block.get(key, []))
        for sub_block in block.get("sub_blocks", []):
            for key in ("example_tickers", "etf_proxies"):
                tickers.update(
                    str(ticker).strip().upper()
                    for ticker in sub_block.get(key, [])
                )
    return tickers


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FactorHunting/1.0)"},
        timeout=30,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    constituents = next((table for table in tables if "Symbol" in table.columns), None)
    if constituents is None:
        raise ValueError("Could not find the S&P 500 ticker table on Wikipedia.")

    return set(constituents["Symbol"].str.replace(".", "-", regex=False).str.upper())


def extract_close_prices(data, tickers):
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close_prices = None
        for level in range(data.columns.nlevels):
            if "Close" in data.columns.get_level_values(level):
                close_prices = data.xs("Close", axis=1, level=level)
                break
        if close_prices is None:
            raise ValueError("Yahoo Finance response did not contain Close prices.")
        return close_prices

    if "Close" not in data.columns:
        raise ValueError("Yahoo Finance response did not contain a Close column.")

    close_prices = data["Close"]
    if isinstance(close_prices, pd.Series):
        return close_prices.to_frame(name=tickers[0])
    return close_prices


def download_prices(tickers):
    batches = []
    batch_count = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for offset in range(0, len(tickers), BATCH_SIZE):
        batch_number = offset // BATCH_SIZE + 1
        batch = tickers[offset:offset + BATCH_SIZE]
        print(f"Downloading batch {batch_number}/{batch_count} ({len(batch)} tickers)...")

        try:
            data = yf.download(
                batch,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=30,
            )
            close_prices = extract_close_prices(data, batch)
        except Exception as error:
            print(f"Warning: batch {batch_number} failed: {error}")
            close_prices = pd.DataFrame()

        if not close_prices.empty:
            batches.append(close_prices)

        if batch_number < batch_count:
            time.sleep(1)

    if not batches:
        raise RuntimeError("No price data was downloaded. Check your internet connection and tickers.")

    prices = pd.concat(batches, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.columns = [str(ticker).upper() for ticker in prices.columns]
    prices = prices.loc[:, ~prices.columns.duplicated()]

    dates = pd.DatetimeIndex(pd.to_datetime(prices.index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    prices.index = dates.normalize()
    prices.index.name = "Date"

    return prices.sort_index().dropna(axis=0, how="all").dropna(axis=1, how="all")


def main():
    print("Loading project tickers...")
    ai_tickers = load_metadata_tickers()

    print("Getting current S&P 500 tickers from Wikipedia...")
    sp500_tickers = get_sp500_tickers()
    tickers = sorted(ai_tickers | sp500_tickers | EXTRA_ETFS)

    print(f"Tickers from project metadata: {len(ai_tickers)}")
    print(f"Current S&P 500 tickers: {len(sp500_tickers)}")
    print(f"Total unique tickers: {len(tickers)}")

    prices = download_prices(tickers)
    prices = prices.reindex(sorted(prices.columns), axis=1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT_PATH, index_label="Date", float_format="%.4f")

    missing_count = int(prices.isna().sum().sum())
    print("Download complete.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows (trading dates): {len(prices):,}")
    print(f"Tickers with data: {len(prices.columns):,}")
    print(f"Missing prices left blank: {missing_count:,}")


if __name__ == "__main__":
    main()