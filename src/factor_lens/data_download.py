"""Reusable implementation for the AI equity factor-lens research workflow.

This module is generated from the corresponding research notebook.  Keep the
notebook focused on configuration, execution and interpretation; place reusable
logic here.
"""

import os, sys, json, time, platform, datetime as dt
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Date range -------------------------------------------------------------
START = "2016-01-01"
END_INCLUSIVE = "2026-10-01"
# yfinance's `end` is EXCLUSIVE, so add one day to include Oct 1, 2026.
END_EXCLUSIVE = (pd.Timestamp(END_INCLUSIVE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# ---- Storage ----------------------------------------------------------------
DRIVE_BASE = "/content/drive/MyDrive/factor_lens"      # folder on your Google Drive
DATA_VERSION = "yahoo_2016-01-01_2026-10-01_v1"        # change => a brand new download
# (outside Colab the data goes to $FACTOR_LENS_LOCAL or ./factor_lens_local)

# ---- Download behaviour -----------------------------------------------------
BATCH_SIZE = 40            # tickers per yf.download call
BATCH_SLEEP = 3.0          # seconds between batches (be gentle with Yahoo)
MAX_BATCH_TRIES = 3        # retries of a whole batch (exponential backoff)
MAX_ATTEMPTS = 2           # per ticker: 1 = batch pass, 2 = individual retry pass
RETRY_FAILED = False       # True => re-attempt tickers that previously failed/empty
DOWNLOAD_METADATA = True   # sector / industry / ETF category etc. (slow, one-off)
INCLUDE_REMOVED_SP500 = True   # try tickers removed from the S&P 500 since START (best effort)
INCLUDE_SUGGESTED = True       # include the suggested extra tickers below (tiers C, D, E)
REFRESH_SP500_LIST = False     # True => re-scrape Wikipedia instead of using the Drive copy

# ---- Ticker lists -----------------------------------------------------------
# Tier B: YOUR lists (unchanged)
MACRO_ETFS = ['SPY','IWN','USMV','TLT','IEF','SHY','TIP','LQD','HYG','UUP','EEM','EFA','VNQ','XLP','XLV','XLU','UTES',
              'KRE','XLF','XLE','PAVE','PDBC','GLD','SLV','DBA','USO','UNG','COPX','COPP','COPA','URA','URNM','NUUR','LIT',
              'SMH','SOXX','SOXQ','CHAT','THNQ','WTAI','IDGT','AIQ','SRVR','IGV','SKYY','QTUM','IVES','IGPT','CIBR','HACK','BUG',
              'GRID','BOTZ']
# Pre-declared placebo/control names for the AI-factor test.  They span defensive consumer,
# healthcare, financial and industrial sectors, and are not used to construct any AI factor.
AI_CONTROL_STOCKS = ['KO', 'PG', 'WMT', 'MCD', 'JNJ', 'JPM', 'XOM', 'CAT']
EXTRA_STOCKS = ['NRG', 'TLN'] + AI_CONTROL_STOCKS

# Tier C (suggested): non-S&P AI-chain names. Verify they exist; many are recent IPOs with short history.
AI_CHAIN_EXTRA = ['TSM','ASML','ARM','MRVL','000660.KS','005930.KS',      # chips / memory (the .KS ones are non-US: keep out of the core panel)
                  'CRDO','ALAB','COHR','LITE','CLS',                       # optical / networking
                  'CRWV','NBIS','IREN','CIFR','APLD','CORZ','WULF',        # AI cloud / hosting
                  'OKLO','SMR','NNE','BWXT','CCJ','LEU','UEC','BE',        # power / nuclear
                  'POWL','NVT','AMSC']                                     # grid / electrical equipment

# Tier D (suggested): indexes, rates (yields in %, NOT prices), FX, vol, futures, crypto
INDEX_SERIES = ['^GSPC','^NDX','^RUT','^SOX','^VIX','^VIX3M','^IRX','^FVX','^TNX','^TYX',
                'DX-Y.NYB','CL=F','NG=F','HG=F','GC=F','BTC-USD']

# Tier E (suggested): extra ETFs for factor construction
EXTRA_ETFS = ['QQQ','IWM','RSP','MTUM','QUAL','VLUE','SPHB','SPLV',
              'XLK','XLI','XLB','XLY','XLC','XLRE',
              'AGG','BIL','SHV','EMB','JNK',
              'ICLN','TAN','NLR','XOP','OIH','XME']


FIELDS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
PRICE_COLS = ['Open', 'High', 'Low', 'Close', 'Adj Close']
MANIFEST_COLS = ['ticker', 'status', 'attempts', 'n_rows', 'first_date', 'last_date',
                 'n_dividends', 'n_splits', 'error', 'batch_id', 'downloaded_at_utc']


def in_colab():
    return "google.colab" in sys.modules


def setup_paths():
    """Mount Drive (in Colab) and create the folder structure. Returns a dict of Paths."""
    if in_colab():
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
        base = Path(DRIVE_BASE)
    else:
        base = Path(os.environ.get("FACTOR_LENS_LOCAL", "./factor_lens_local"))
    root = base / DATA_VERSION
    P = {"root": root}
    for name in ["prices", "panel", "meta", "universe", "qa"]:
        P[name] = root / name
        P[name].mkdir(parents=True, exist_ok=True)
    return P


def price_path(P, ticker):
    safe = ticker.replace("^", "IDX_").replace("=", "_").replace("/", "_")
    return P["prices"] / f"{safe}.parquet"


def check_config(P):
    """Refuse to mix downloads made with different settings in one folder."""
    cfg = dict(source="yfinance", start=START, end_inclusive=END_INCLUSIVE,
               auto_adjust=False, actions=True)
    f = P["root"] / "config.json"
    if f.exists():
        old = json.loads(f.read_text())
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if diff:
            raise ValueError(f"Config differs from the one this folder was created with: {diff}. "
                             f"Use a new DATA_VERSION to download with new settings.")
    else:
        f.write_text(json.dumps(cfg, indent=2))


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def yahoo_symbol(s):
    """Wikipedia style -> Yahoo style (BRK.B -> BRK-B). Only used for scraped S&P tickers."""
    return str(s).strip().upper().replace(".", "-")


FALLBACK_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
EMPTY_CHANGES_COLS = ["Effective Date", "Added Ticker", "Removed Ticker"]


def _flat(t):
    """Column names of a table as plain strings (joins MultiIndex levels)."""
    return [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]


def _find_symbol_col(cols):
    for c in cols:
        if c.strip().lower() in ("symbol", "ticker", "ticker symbol"):
            return c
    for c in cols:
        if "symbol" in c.lower() or "ticker" in c.lower():
            return c
    return None


def _normalize_cons(df):
    """Any constituents table -> columns Symbol, Security, GICS Sector, Date added."""
    df = df.copy(); cols = _flat(df); df.columns = cols
    sym = _find_symbol_col(cols)
    name = next((c for c in cols if c.strip().lower() in ("security", "name")), None)
    sec = next((c for c in cols if "sector" in c.lower() and "sub" not in c.lower()), None)
    add = next((c for c in cols if "date" in c.lower() and "add" in c.lower()), None)
    return pd.DataFrame({"Symbol": df[sym], "Security": df[name] if name else "",
                         "GICS Sector": df[sec] if sec else "", "Date added": df[add] if add else ""})


def scrape_sp500():
    """Wikipedia (constituents + change history); falls back to a GitHub CSV (constituents only)."""
    import requests
    diag, cons, chg = [], None, None
    try:
        r = requests.get(WIKI_URL, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0 (university research project; factor lens)"})
        diag.append(f"wikipedia: HTTP {r.status_code}, {len(r.text)} chars")
        r.raise_for_status()
        for i, t in enumerate(pd.read_html(StringIO(r.text))):
            cols = _flat(t); low = " ".join(cols).lower()
            diag.append(f"  table {i}: shape={t.shape} cols={cols[:6]}")
            if chg is None and "removed" in low and "added" in low:
                chg = t.copy(); chg.columns = cols
            elif cons is None and len(t) >= 300 and "removed" not in low and _find_symbol_col(cols):
                cons = _normalize_cons(t)
    except Exception as e:
        diag.append(f"wikipedia failed: {type(e).__name__}: {e}")

    if cons is None:
        try:
            r = requests.get(FALLBACK_URL, timeout=30); r.raise_for_status()
            cons = _normalize_cons(pd.read_csv(StringIO(r.text), keep_default_na=False))
            diag.append("constituents taken from the GitHub fallback CSV (no change history)")
        except Exception as e:
            diag.append(f"fallback failed: {type(e).__name__}: {e}")
    if cons is None:
        print("\n".join(diag))
        raise RuntimeError("Could not get the S&P 500 constituents from Wikipedia or the fallback. "
                           "Diagnostics printed above - paste them to me.")
    if chg is None:
        print("\n".join(diag))
        print("WARNING: no change-history table found -> tickers removed from the S&P 500 will not be added.")
        chg = pd.DataFrame(columns=EMPTY_CHANGES_COLS)
    return cons, chg


def load_or_scrape_sp500(P):
    f_cons, f_chg = P["universe"] / "sp500_constituents.csv", P["universe"] / "sp500_changes.csv"
    if f_cons.exists() and f_chg.exists() and not REFRESH_SP500_LIST:
        return (pd.read_csv(f_cons, keep_default_na=False), pd.read_csv(f_chg, keep_default_na=False))
    cons, chg = scrape_sp500()
    cons.to_csv(f_cons, index=False)
    chg.to_csv(f_chg, index=False)
    (P["universe"] / "sp500_scrape_info.json").write_text(
        json.dumps({"scraped_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "url": WIKI_URL}, indent=2))
    return cons, chg


def build_universe(P):
    """Returns a DataFrame indexed by ticker with a 'groups' column ('a;b;c') and S&P membership info."""
    cons, chg = load_or_scrape_sp500(P)
    info = {}   # ticker -> dict(groups=set, ...)

    def add(tickers, group, **attrs):
        for t in tickers:
            d = info.setdefault(t, {"groups": set()})
            d["groups"].add(group)
            for k, v in attrs.items():
                d.setdefault(k, v)

    # S&P 500 today
    for _, row in cons.iterrows():
        t = yahoo_symbol(row["Symbol"])
        if not t or t == "NAN":
            continue
        add([t], "sp500", sp500_now=True, wiki_sector=row["GICS Sector"], sp500_date_added=row["Date added"])
    current = set(info)

    # S&P 500 names removed since START (best effort; Yahoo mostly lacks delisted tickers)
    date_col = next((c for c in chg.columns if "effective" in c.lower() or "date" in c.lower()), None)
    rem_col = next((c for c in chg.columns if "removed" in c.lower() and "ticker" in c.lower()), None)
    if INCLUDE_REMOVED_SP500 and date_col and rem_col and len(chg):
        c2 = chg[[date_col, rem_col]].copy()
        c2["d"] = pd.to_datetime(c2[date_col], errors="coerce")
        c2 = c2.dropna(subset=["d", rem_col])
        c2 = c2[c2["d"] >= pd.Timestamp(START)]
        for _, row in c2.sort_values("d").iterrows():
            t = yahoo_symbol(row[rem_col])
            if t and t != "NAN" and t not in current:
                add([t], "sp500_removed", sp500_removed_date=row["d"].strftime("%Y-%m-%d"))

    # Your lists + suggestions
    add(MACRO_ETFS, "macro_etf")
    add(EXTRA_STOCKS, "extra_stock")
    if INCLUDE_SUGGESTED:
        add(AI_CHAIN_EXTRA, "ai_chain_extra")
        add(INDEX_SERIES, "index_rate_fx")
        add(EXTRA_ETFS, "extra_etf")

    rows = []
    for t, d in info.items():
        rows.append({"ticker": t, "groups": ";".join(sorted(d["groups"])),
                     "sp500_now": bool(d.get("sp500_now", False)),
                     "wiki_sector": d.get("wiki_sector"),
                     "sp500_date_added": d.get("sp500_date_added"),
                     "sp500_removed_date": d.get("sp500_removed_date")})
    uni = pd.DataFrame(rows).set_index("ticker").sort_index()
    uni.to_csv(P["universe"] / "universe.csv")
    return uni


def load_manifest(P):
    f = P["root"] / "manifest.csv"
    if f.exists():
        m = pd.read_csv(f, keep_default_na=False, na_values=[""]).set_index("ticker")
        m["attempts"] = m["attempts"].fillna(0).astype(int)
        return m.astype(object)
    return pd.DataFrame(columns=MANIFEST_COLS[1:], index=pd.Index([], name="ticker"))


def save_manifest(P, m):
    f, tmp = P["root"] / "manifest.csv", P["root"] / "manifest.csv.tmp"
    m.reset_index()[MANIFEST_COLS].to_csv(tmp, index=False)
    os.replace(tmp, f)


def _clean_ticker_frame(df):
    """Standardise one ticker's frame: tz-naive date index, fixed columns, inside [START, END_INCLUSIVE]."""
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.reindex(columns=FIELDS)
    df = df.dropna(subset=PRICE_COLS, how="all")          # rows where the ticker did not trade
    df[["Dividends", "Stock Splits"]] = df[["Dividends", "Stock Splits"]].fillna(0.0)
    return df.loc[START:END_INCLUSIVE]


def _split_frames(df, tickers):
    """yf.download result -> {ticker: DataFrame}. Handles MultiIndex and single-ticker shapes."""
    out = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in (0, 1):
            if any(t in set(df.columns.get_level_values(lvl)) for t in tickers):
                for t in tickers:
                    if t in set(df.columns.get_level_values(lvl)):
                        out[t] = df.xs(t, axis=1, level=lvl)
                return out
        return out
    out[tickers[0]] = df
    return out


def _yahoo_errors(yf):
    errs = getattr(getattr(yf, "shared", None), "_ERRORS", None)
    return dict(errs) if isinstance(errs, dict) else {}


def _download(yf, tickers, tries=None):
    """One yf.download call with retries. Returns (frames dict, error-by-ticker dict, batch_error)."""
    last_err = None
    tries = tries or MAX_BATCH_TRIES
    for k in range(tries):
        try:
            errs = getattr(getattr(yf, "shared", None), "_ERRORS", None)
            if isinstance(errs, dict):
                errs.clear()
            df = yf.download(tickers=tickers, start=START, end=END_EXCLUSIVE,
                             auto_adjust=False, actions=True, group_by="ticker",
                             threads=True, progress=False)
            if df is not None and not df.empty:
                return _split_frames(df, tickers), _yahoo_errors(yf), None
            last_err = "empty result"
        except Exception as e:                                   # network / rate limit / parsing
            last_err = f"{type(e).__name__}: {e}"
        if k < tries - 1:
            time.sleep(BATCH_SLEEP * 2 * (2 ** k))
    return {}, {}, last_err


def _record(P, manifest, ticker, frame, err, batch_id):
    """Write one ticker's file (if any data) and update its manifest row."""
    prev = int(manifest.loc[ticker, "attempts"]) if ticker in manifest.index else 0
    row = dict(attempts=prev + 1, batch_id=batch_id,
               downloaded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    df = _clean_ticker_frame(frame) if frame is not None else None
    if df is None or df.empty:
        row.update(status="empty", n_rows=0, first_date=None, last_date=None,
                   n_dividends=0, n_splits=0, error=(str(err)[:300] if err else "no data returned"))
    else:
        df.to_parquet(price_path(P, ticker))
        row.update(status="ok", n_rows=len(df), first_date=df.index[0].strftime("%Y-%m-%d"),
                   last_date=df.index[-1].strftime("%Y-%m-%d"),
                   n_dividends=int((df["Dividends"] != 0).sum()), n_splits=int((df["Stock Splits"] != 0).sum()),
                   error=None)
    for k, v in row.items():
        manifest.loc[ticker, k] = v


def download_missing(P, universe):
    import yfinance as yf
    manifest = load_manifest(P)
    if RETRY_FAILED and len(manifest):
        manifest = manifest[manifest["status"] == "ok"].copy()   # forget failures -> treated as new

    # ---- Pass A: tickers never attempted, in batches --------------------------------
    todo_a = [t for t in universe.index if t not in manifest.index]
    batches = [todo_a[i:i + BATCH_SIZE] for i in range(0, len(todo_a), BATCH_SIZE)]
    print(f"Pass A: {len(todo_a)} new tickers in {len(batches)} batches")
    for b, tickers in enumerate(batches, 1):
        frames, errs, batch_err = _download(yf, tickers)
        for t in tickers:
            _record(P, manifest, t, frames.get(t), errs.get(t, batch_err), f"A{b:03d}")
        save_manifest(P, manifest)                               # to Drive after EVERY batch
        n_ok = int((manifest.loc[tickers, "status"] == "ok").sum())
        print(f"  batch {b}/{len(batches)}: ok {n_ok}/{len(tickers)}")
        time.sleep(BATCH_SLEEP)

    # ---- Pass B: individual retry for non-ok tickers with attempts left ---------------
    retry = [t for t in universe.index if t in manifest.index
             and manifest.loc[t, "status"] != "ok" and int(manifest.loc[t, "attempts"]) < MAX_ATTEMPTS]
    print(f"Pass B: retrying {len(retry)} tickers one by one")
    for i, t in enumerate(retry, 1):
        frames, errs, batch_err = _download(yf, [t], tries=1)
        _record(P, manifest, t, frames.get(t), errs.get(t, batch_err), "B")
        if i % 10 == 0 or i == len(retry):
            save_manifest(P, manifest)
        time.sleep(1.0)
    save_manifest(P, manifest)

    (P["root"] / "download_info.json").write_text(json.dumps({
        "last_download_run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "pandas_version": pd.__version__, "python": platform.python_version(),
        "note": "Yahoo adjusted prices are a snapshot: Adj Close is recalculated retroactively on new dividends/splits."
    }, indent=2))
    return manifest


def panel_file(P, field):
    return P["panel"] / f"{field.replace(' ', '_')}.parquet"


def build_panels(P, manifest):
    ok = manifest.index[manifest["status"] == "ok"].tolist()
    frames = {t: pd.read_parquet(price_path(P, t)) for t in ok}
    for field in FIELDS:
        wide = pd.concat({t: f[field] for t, f in frames.items()}, axis=1).sort_index()
        wide.columns.name = "ticker"
        wide.to_parquet(panel_file(P, field))
    (P["panel"] / "panel_meta.json").write_text(json.dumps({
        "tickers": sorted(ok), "n_tickers": len(ok), "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "first_date": min(f.index[0] for f in frames.values()).strftime("%Y-%m-%d"),
        "last_date": max(f.index[-1] for f in frames.values()).strftime("%Y-%m-%d")}, indent=2))
    print(f"Built panels for {len(ok)} tickers")


def load_panels(P):
    return {field: pd.read_parquet(panel_file(P, field)) for field in FIELDS}


def is_complete(P, universe, manifest):
    """True if every universe ticker is settled (ok, or failed with no attempts left) AND panels match."""
    if manifest.empty or not (P["panel"] / "panel_meta.json").exists():
        return False
    if any(t not in manifest.index for t in universe.index):
        return False
    m = manifest.loc[universe.index]
    unsettled = m[(m["status"] != "ok") & (m["attempts"].astype(int) < MAX_ATTEMPTS)]
    if RETRY_FAILED and (m["status"] != "ok").any():
        return False
    if len(unsettled):
        return False
    meta = json.loads((P["panel"] / "panel_meta.json").read_text())
    return set(meta["tickers"]) == set(manifest.index[manifest["status"] == "ok"])


INFO_FIELDS = ['quoteType', 'longName', 'shortName', 'sector', 'industry', 'category', 'fundFamily',
               'currency', 'exchange', 'fullExchangeName', 'country', 'marketCap', 'sharesOutstanding',
               'fundInceptionDate', 'firstTradeDateMilliseconds', 'firstTradeDateEpochUtc', 'legalType']


def ensure_metadata(P, tickers):
    f = P["meta"] / "info.jsonl"
    done = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["ticker"]] = d
    redo_err = RETRY_FAILED
    todo = [t for t in tickers if t not in done or (redo_err and done[t].get("info_error"))]
    if todo:
        import yfinance as yf
        print(f"Metadata: fetching {len(todo)} tickers (resumable)")
        consecutive_fail = 0
        with open(f, "a") as fh:
            for i, t in enumerate(todo, 1):
                rec = {"ticker": t}
                try:
                    info = yf.Ticker(t).get_info()
                    rec.update({k: info.get(k) for k in INFO_FIELDS})
                    consecutive_fail = 0
                except Exception as e:
                    rec["info_error"] = f"{type(e).__name__}: {e}"[:200]
                    consecutive_fail += 1
                fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
                if consecutive_fail >= 5:
                    time.sleep(30); consecutive_fail = 0
                time.sleep(0.4)
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}")
    recs = {}
    for line in f.read_text().splitlines() if f.exists() else []:
        if line.strip():
            d = json.loads(line); recs[d["ticker"]] = d       # later lines win
    meta = pd.DataFrame(list(recs.values())).set_index("ticker") if recs else pd.DataFrame()
    if len(meta):
        meta.to_csv(P["meta"] / "ticker_info.csv")
    return meta


def run_qa(P, data, universe, manifest):
    adj, close = data["Adj Close"], data["Close"]
    expected, ok = set(universe.index), set(manifest.index[manifest["status"] == "ok"])
    missing = sorted(expected - ok)
    core_missing = [t for t in missing if "sp500_removed" not in universe.loc[t, "groups"].split(";")
                    or ";" in universe.loc[t, "groups"]]
    print(f"Universe: {len(expected)} | downloaded OK: {len(expected & ok)} | empty/failed: {len(missing)}")
    print(f"  of which NOT only 'sp500_removed' (check these!): {core_missing}")

    last_all, first_all = adj.index.max(), adj.index.min()
    print(f"Date range in panel: {first_all.date()} -> {last_all.date()}")
    assert last_all <= pd.Timestamp(END_INCLUSIVE), "data after END_INCLUSIVE found!"

    cov = pd.DataFrame({"first": adj.apply(lambda s: s.first_valid_index()),
                        "last": adj.apply(lambda s: s.last_valid_index()), "n_obs": adj.count()})
    cov["groups"] = universe["groups"].reindex(cov.index)
    cov["flag_stale_or_delisted"] = cov["last"] < pd.Timestamp(END_INCLUSIVE) - pd.Timedelta(days=7)
    cov["flag_late_start"] = cov["first"] > pd.Timestamp(START) + pd.Timedelta(days=10)
    cov["adj_close_all_nan"] = adj.isna().all()
    rem_date = pd.to_datetime(universe["sp500_removed_date"], errors="coerce").reindex(cov.index)
    cov["flag_removed_ticker_reuse?"] = cov["groups"].fillna("").str.contains("sp500_removed") & (cov["first"] > rem_date)
    cov.to_csv(P["qa"] / "download_qa.csv")

    print(f"Stale / ended before {END_INCLUSIVE}: {int(cov['flag_stale_or_delisted'].sum())} tickers")
    print(f"Late start (listed after {START}): {int(cov['flag_late_start'].sum())} tickers")
    etf_late = cov[cov["groups"].fillna("").str.contains("macro_etf") & cov["flag_late_start"]].sort_values("first")
    if len(etf_late):
        print("\nYoung ETFs from MACRO_ETFS (limits which factors can enter long windows):")
        print(etf_late[["first", "n_obs"]].to_string())
    sus = cov[cov["flag_removed_ticker_reuse?"]]
    if len(sus):
        print(f"\nWARNING: {len(sus)} 'sp500_removed' tickers start AFTER their removal date -> Yahoo probably "
              f"returned a different company that reuses the ticker. Exclude: {sus.index.tolist()}")

    if {"SPY", "^GSPC"} <= set(close.columns):
        r = close[["SPY", "^GSPC"]].pct_change().dropna()
        print(f"\nSPY vs ^GSPC daily return correlation: {r['SPY'].corr(r['^GSPC']):.5f} (expect ~0.999+)")
    if "SPY" in close:
        print(f"SPY last date: {close['SPY'].last_valid_index().date()} | rows: {close['SPY'].count()}")
    return cov


def get_data(download_metadata=DOWNLOAD_METADATA, run_qa_report=True):
    """Load everything from Drive if it's already there, otherwise download the missing part first.

    Returns dict with keys: 'prices' (dict field -> DataFrame dates x tickers), 'universe', 'manifest',
    'meta', 'paths'.
    """
    P = setup_paths()
    check_config(P)
    universe = build_universe(P)
    manifest = load_manifest(P)

    if is_complete(P, universe, manifest):
        print("Cache hit: all data already on Drive -> loading, no Yahoo calls.")
    else:
        print("Cache incomplete -> downloading what is missing (finished tickers are skipped).")
        manifest = download_missing(P, universe)
        build_panels(P, manifest)

    prices = load_panels(P)
    ok = manifest.index[manifest["status"] == "ok"].tolist()
    meta = ensure_metadata(P, ok) if download_metadata else (
        pd.read_csv(P["meta"] / "ticker_info.csv", index_col="ticker") if (P["meta"] / "ticker_info.csv").exists()
        else pd.DataFrame())
    out = {"prices": prices, "universe": universe, "manifest": manifest, "meta": meta, "paths": P}
    if run_qa_report:
        out["qa"] = run_qa(P, prices, universe, manifest)
    return out
