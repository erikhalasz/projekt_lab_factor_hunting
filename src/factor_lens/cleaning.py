"""Reusable implementation for the AI equity factor-lens research workflow.

This module is generated from the corresponding research notebook.  Keep the
notebook focused on configuration, execution and interpretation; place reusable
logic here.
"""

import os, sys, json, hashlib, subprocess, platform, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
try:
    display
except NameError:                      # plain python (tests); notebooks provide display()
    display = print


# ---- Where things live ---------------------------------------------------------
RAW_VERSION   = "yahoo_2016-01-01_2026-10-01_v1"
CLEAN_VERSION = "clean_v1"
START, END_INCLUSIVE = "2016-01-01", "2026-10-01"

FORCE_RECOMPUTE = False   # True => redo cleaning even if a matching output is on Drive
CODE_VERSION    = "r5"    # bump whenever the cleaning LOGIC changes (it is part of the cache key)
STRICT_GATES    = True    # True => stop (AssertionError) if a validation gate is 'BAD'

# ---- All thresholds in one place (hashed into the cache key) ---------------------
CFG = dict(
    # calendar
    cal_min_breadth=0.5,          # a date is a trading day if >=50% of alive US stocks have a price
    cal_max_mismatch_days=5,      # max allowed difference vs the NYSE calendar library
    thin_day_breadth=0.97,        # flag trading days where breadth is below this
    # integrity
    ohlc_tol=0.001,               # relative tolerance for High/Low/Open/Close consistency
    stale_run_zero_vol=5,         # >= this many unchanged closes WITH zero/missing volume => stale (return masked)
    flat_run_flag=10,             # >= this many unchanged closes with volume => only flagged (T-bill ETFs are legitimately flat)
    # missing data
    max_gap_run=3,                # internal gap of >= this many consecutive days => unreliable
    max_internal_missing_frac=0.02,
    hole_date_frac=0.20,          # date where >=20% of alive tickers are missing => data hole date
    # adj close vs close+dividend
    adj_mismatch_bp=10.0,
    adj_mismatch_large_bp=100.0,
    adj_step_max=0.15,            # largest plausible dividend-like step without a table entry
    special_div_yield=0.10,       # cash dividend / previous close above this = special/spin-off: conflicts are masked, not guessed
    # splits
    split_scan_abs=0.30,
    split_ratios=[1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50],
    split_tol=0.05,
    # outliers
    beta_window=60, beta_min_periods=30,
    scale_window=60, scale_min_periods=30,
    sigma_k=6.0, sigma_floor=0.002,
    stock_abs=0.10, etf_abs=0.06, extreme_abs_mult=2.0,
    reversal_tol=0.30,            # |2-day net move| <= 30% of the spike => reversal (bad tick signature)
    vol_window=20, vol_min_periods=10, event_vol_ratio=2.0,
    market_wide_frac=0.05, market_wide_min=5,
    # eligibility
    min_coverage=0.98,
    max_zero_volume_frac=0.10,    # share of zero-volume days among priced days (thin pre-listing ADR/OTC history)
    min_dollar_volume=1e6, dollar_volume_window=60,
    high_corr_pair=0.995,
    # macro
    rf_ffill_limit=3, yield_sanity=[-1.0, 20.0], future_jump=0.15,
)

# Explicit lists (used for block C transformations and share classes)
YIELDS   = ['^IRX', '^FVX', '^TNX', '^TYX']          # levels in percent
VOL_IDX  = ['^VIX', '^VIX3M']
FUTURES  = ['CL=F', 'NG=F', 'HG=F', 'GC=F']
INDEXES  = ['^GSPC', '^NDX', '^RUT', '^SOX']
FX_CRYPTO = ['DX-Y.NYB', 'BTC-USD']
FORCE_C: list = []        # tickers you want forced into block C
FORCE_D: list = []        # tickers you want forced into the quarantined block D
SHARE_CLASS_SECONDARY = {'GOOG': 'GOOGL', 'FOX': 'FOXA', 'NWS': 'NWSA'}   # secondary -> primary

FIELDS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']


def mount_drive():
    """
    Locate the factor_lens_local directory inside the GitHub repository.

    The function works when the notebook or Python module is run
    from the repository root or one of its subdirectories.

    An optional FACTOR_LENS_LOCAL environment variable can override
    the default location.
    """

    # Optional explicit location
    env_path = os.environ.get("FACTOR_LENS_LOCAL")

    if env_path:
        base = Path(env_path).expanduser().resolve()
        if (base / RAW_VERSION).is_dir():
            return base
        raise FileNotFoundError(
            f"FACTOR_LENS_LOCAL is set to {base}, "
            f"but {base / RAW_VERSION} does not exist."
        )

    # Search from the current working directory and this module's location
    search_locations = [Path.cwd()]

    if "__file__" in globals():
        search_locations.append(Path(__file__).resolve().parent)

    checked = set()

    for location in search_locations:
        for parent in [location, *location.parents]:
            if parent in checked:
                continue
            checked.add(parent)

            candidate = parent / "factor_lens_local"

            if (candidate / RAW_VERSION).is_dir():
                return candidate

    raise FileNotFoundError(
        "Could not find the downloaded data in the GitHub repository.\n"
        f"Expected folder: factor_lens_local/{RAW_VERSION}\n"
        "Make sure you are running the notebook from inside the "
        "projekt_lab_factor_hunting repository."
    )


def panel_file(raw_root, field):
    return raw_root / "panel" / f"{field.replace(' ', '_')}.parquet"


class FlagLog:
    """Collects every rule that fired: date, ticker, rule, label, value, action."""
    def __init__(self):
        self.parts = []

    def add_mask(self, mask, rule, label="", action="flag_only", values=None):
        arr = mask.to_numpy(dtype=bool)
        rr, cc = np.nonzero(arr)
        if len(rr) == 0:
            return 0
        vals = values.to_numpy(dtype=float)[rr, cc] if values is not None else np.full(len(rr), np.nan)
        self.parts.append(pd.DataFrame({"date": mask.index[rr], "ticker": mask.columns[cc], "rule": rule,
                                        "label": label, "value": vals, "action": action}))
        return len(rr)

    def add_rows(self, dates, tickers, rule, label="", action="flag_only", values=None):
        n = len(dates)
        if n == 0:
            return 0
        self.parts.append(pd.DataFrame({"date": pd.DatetimeIndex(dates), "ticker": list(tickers), "rule": rule,
                                        "label": label, "value": np.full(n, np.nan) if values is None else values,
                                        "action": action}))
        return n

    def frame(self):
        if not self.parts:
            return pd.DataFrame(columns=["date", "ticker", "rule", "label", "value", "action"])
        out = pd.concat(self.parts, ignore_index=True)
        return out.sort_values(["rule", "ticker", "date"], kind="stable").reset_index(drop=True)


def run_bounds(m):
    """1-D bool array -> (starts, ends_inclusive) of the True runs."""
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0] - 1


def mask_runs_ge(mask, min_len):
    """DataFrame bool -> DataFrame bool marking ENTIRE True-runs whose length >= min_len."""
    arr = mask.to_numpy(dtype=bool)
    out = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        s, e = run_bounds(arr[:, j])
        for a, b in zip(s, e):
            if b - a + 1 >= min_len:
                out[a:b + 1, j] = True
    return pd.DataFrame(out, index=mask.index, columns=mask.columns)


def alive_mask(valid):
    """True between a ticker's first and last valid observation (inclusive)."""
    v = valid.astype(np.int8)                      # cummax on int is safe across pandas versions
    return (v.cummax() > 0) & (v.iloc[::-1].cummax().iloc[::-1] > 0)


def simple_ret(px):
    return px / px.shift(1) - 1.0


def cfg_hash(extra_text=""):
    return hashlib.sha256((CODE_VERSION + json.dumps(CFG, sort_keys=True, default=str) + extra_text).encode()).hexdigest()[:16]


def read_raw(raw_root):
    panels = {}
    for f in FIELDS:
        p = pd.read_parquet(panel_file(raw_root, f))
        p.index = pd.DatetimeIndex(p.index).normalize()
        p.index.name = "Date"
        panels[f] = p.astype("float64").sort_index()
    cols = panels["Close"].columns
    for f in FIELDS:
        assert panels[f].columns.equals(cols) and panels[f].index.equals(panels["Close"].index), \
            f"raw panel '{f}' has a different shape/index than 'Close'"
    universe = pd.read_csv(raw_root / "universe" / "universe.csv", index_col="ticker", keep_default_na=False)
    mf = raw_root / "meta" / "ticker_info.csv"
    meta = pd.read_csv(mf, index_col="ticker", keep_default_na=False) if mf.exists() else pd.DataFrame()
    return dict(panels=panels, universe=universe, meta=meta)


def classify_blocks(tickers, universe, meta):
    """A = US stocks, B = US ETFs, C = indexes/rates/FX/vol/futures/crypto, D = non-US (quarantined)."""
    out = {}
    for t in tickers:
        g = set(str(universe.at[t, "groups"]).split(";")) if t in universe.index else set()
        qt = str(meta.at[t, "quoteType"]).upper() if (len(meta) and t in meta.index and "quoteType" in meta.columns) else ""
        cur = str(meta.at[t, "currency"]).upper() if (len(meta) and t in meta.index and "currency" in meta.columns) else ""
        if "index_rate_fx" in g or t in FORCE_C or t.startswith("^") or t.endswith("=F"):
            b = "C"
        elif "." in t or cur not in ("", "USD") or t in FORCE_D:
            b = "D"
        elif qt == "ETF" or g & {"macro_etf", "extra_etf"}:
            b = "B"
        else:
            b = "A"
        out[t] = b
    return pd.Series(out, name="block")


def get_nyse_days(start, end):
    try:
        import pandas_market_calendars as mcal
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas_market_calendars"], check=True)
            import pandas_market_calendars as mcal
        except Exception as e:
            print(f"WARNING: pandas_market_calendars unavailable ({type(e).__name__}); "
                  f"calendar is NOT validated against NYSE rules.")
            return None
    days = mcal.get_calendar("NYSE").valid_days(start_date=start, end_date=end)
    return pd.DatetimeIndex(days).tz_localize(None).normalize()


def build_master_calendar(close_A, close_spy, log):
    valid = close_A.notna()
    alive = alive_mask(valid)
    n_alive = alive.sum(axis=1)
    breadth = valid.sum(axis=1) / n_alive.where(n_alive > 0)
    in_range = (breadth.index >= pd.Timestamp(START)) & (breadth.index <= pd.Timestamp(END_INCLUSIVE))
    cal = breadth.index[in_range & (breadth >= CFG["cal_min_breadth"]).to_numpy()]
    cal = pd.DatetimeIndex(cal, name="Date")
    info = {"n_days": len(cal), "first": str(cal[0].date()), "last": str(cal[-1].date())}

    nyse = get_nyse_days(START, END_INCLUSIVE)
    if nyse is not None:
        only_nyse, only_cal = nyse.difference(cal), cal.difference(nyse)
        info.update(n_only_in_nyse_lib=len(only_nyse), n_only_in_data=len(only_cal))
        if len(only_nyse) or len(only_cal):
            print(f"Calendar differences vs NYSE library: in library but no data: {[str(d.date()) for d in only_nyse[:10]]}"
                  f" | in data but not in library: {[str(d.date()) for d in only_cal[:10]]}")
        assert len(only_nyse) + len(only_cal) <= CFG["cal_max_mismatch_days"], \
            "master calendar differs from the NYSE calendar by more than cal_max_mismatch_days - investigate"
        log.add_rows(only_nyse, ["*"] * len(only_nyse), "calendar_missing_in_data", "library says open", "flag_only")
    thin = breadth.reindex(cal)
    thin = thin[thin < CFG["thin_day_breadth"]]
    log.add_rows(thin.index, ["*"] * len(thin), "thin_day", "low breadth", "flag_only", values=thin.to_numpy())
    spy_missing = close_spy.reindex(cal).isna()
    sm = spy_missing[spy_missing].index
    log.add_rows(sm, ["SPY"] * len(sm), "spy_hole", "SPY missing on a trading day", "flag_only")
    return cal, info


def placeholder_rows(stale, AC):
    """Flat zero-volume runs touching the START or END of a ticker's history are placeholder rows (pre-listing /
    post-delisting), not trading. Returns a mask of the placeholder price rows."""
    st, valid = stale.to_numpy(), AC.notna().to_numpy()
    out = np.zeros_like(st)
    for j in range(st.shape[1]):
        idx = np.flatnonzero(valid[:, j])
        if len(idx) < 2:
            continue
        f, l = idx[0], idx[-1]
        if st[f + 1, j]:                                   # leading run: stale from the day after the first row
            k = f + 1
            while k <= l and st[k, j]:
                k += 1
            out[f:k, j] = True                             # rows f..k-1 are placeholders; real trading starts at k
        if st[l, j]:                                       # trailing run
            k = l
            while k > f and st[k, j]:
                k -= 1
            out[k + 1:l + 1, j] = True                     # rows after the last real print
    return pd.DataFrame(out, index=stale.index, columns=stale.columns)


def integrity_checks(O, H, L, C, AC, V, log):
    """Flags (and masks non-positive prices). Returns cleaned C, AC and the stale mask."""
    tol = CFG["ohlc_tol"]
    bad_px = (C <= 0) | (AC <= 0)
    log.add_mask(bad_px, "nonpositive_price", "", "nan_price", values=C)
    C, AC = C.mask(bad_px), AC.mask(bad_px)

    log.add_mask(AC.isna() & C.notna(), "adj_missing_close_present", "", "flag_only")
    mx, mn = np.maximum(O, C), np.minimum(O, C)
    log.add_mask(H < mx * (1 - tol), "ohlc_high_below_open_close", "", "flag_only", values=H / mx - 1)
    log.add_mask(L > mn * (1 + tol), "ohlc_low_above_open_close", "", "flag_only", values=L / mn - 1)
    log.add_mask(H < L * (1 - tol), "ohlc_high_below_low", "", "flag_only")
    log.add_mask(V < 0, "negative_volume", "", "flag_only", values=V)
    log.add_mask((V == 0) & ((C / C.shift(1) - 1).abs() > 0), "zero_volume_price_change", "", "flag_only")

    same = (C == C.shift(1)) & C.notna()                         # zero-return days
    no_vol = (V == 0) | V.isna()
    stale = mask_runs_ge(same & no_vol, CFG["stale_run_zero_vol"])    # no trading + unchanged price => not a real return
    log.add_mask(stale, "stale_price", "unchanged close, zero volume", "nan_return")
    flat = mask_runs_ge(same, CFG["flat_run_flag"]) & ~stale          # traded but flat (e.g. T-bill ETFs near zero rates)
    log.add_mask(flat, "long_flat_run", "unchanged close but traded - kept", "flag_only")
    ph = placeholder_rows(stale, AC)
    log.add_mask(ph, "placeholder_price", "flat zero-volume rows at start/end of history (pre-listing/post-delisting)", "nan_price")
    return C.mask(ph), AC.mask(ph), stale


def gap_analysis(valid, log, rule="internal_gap", action="nan_return", hole_flags=True):
    """Internal gaps (between first and last valid obs). Leading/trailing NaNs are legitimate."""
    alive = alive_mask(valid)
    internal = (alive & ~valid)
    arr = internal.to_numpy()
    maxrun = np.zeros(arr.shape[1], dtype=int)
    d_list, t_list, v_list = [], [], []
    for j in range(arr.shape[1]):
        s, e = run_bounds(arr[:, j])
        if len(s):
            lens = e - s + 1
            maxrun[j] = lens.max()
            d_list.extend(valid.index[s]); t_list.extend([valid.columns[j]] * len(s)); v_list.extend(lens.astype(float))
    log.add_rows(d_list, t_list, rule, "run start (value = run length in trading days)", action, values=np.array(v_list))
    n_alive = alive.sum()
    stats = pd.DataFrame({"n_alive_days": n_alive, "n_price_obs": valid.sum(),
                          "n_internal_missing": internal.sum(), "max_gap_run": maxrun})
    stats["internal_missing_frac"] = stats["n_internal_missing"] / stats["n_alive_days"].replace(0, np.nan)
    stats["coverage"] = stats["n_price_obs"] / stats["n_alive_days"].replace(0, np.nan)
    has = valid.any()
    stats["first_valid"] = valid.idxmax().where(has)
    stats["last_valid"] = valid.iloc[::-1].idxmax().where(has)
    hole = internal.sum(axis=1) / alive.sum(axis=1).where(alive.sum(axis=1) > 0)
    hole = hole[hole >= CFG["hole_date_frac"]]
    if hole_flags:
        log.add_rows(hole.index, ["*"] * len(hole), "data_hole_date", "many tickers missing", "flag_only", values=hole.to_numpy())
    return stats, internal


def split_like(r):
    """Mask of moves that look like an unadjusted split: price x 1/k (forward) or x k (reverse)."""
    g = (1.0 + r).to_numpy()
    out = np.zeros(g.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for k in CFG["split_ratios"]:
            out |= np.abs(g * k - 1.0) <= CFG["split_tol"]
            out |= np.abs(g / k - 1.0) <= CFG["split_tol"]
        out &= np.abs(r.to_numpy()) >= CFG["split_scan_abs"]
    return pd.DataFrame(out, index=r.index, columns=r.columns)


def reconcile_adj_vs_div(AC, C, Dv, log):
    r_adj, r_div = simple_ret(AC), (C + Dv) / C.shift(1) - 1.0
    diff = r_adj - r_div
    both = r_adj.notna() & r_div.notna()
    ad = diff.abs()
    tol, big = CFG["adj_mismatch_bp"] / 1e4, CFG["adj_mismatch_large_bp"] / 1e4
    small = both & (ad > tol) & (ad <= big)
    large = both & (ad > big)
    sl_adj, sl_div = split_like(r_adj), split_like(r_div)
    close_unadj = large & sl_div & ~sl_adj
    step_no_div = large & ~close_unadj & (diff > 0) & (Dv.fillna(0) == 0)    # Yahoo adjusted for a distribution/spin-off the table lacks
    subst = large & ~close_unadj & ~step_no_div
    d_yield = Dv / C.shift(1)                         # cash dividend as a share of the previous close
    method = subst & (d_yield <= CFG["special_div_yield"])      # regular-size dividend: Yahoo's proportional adjustment differs
    unresolved = subst & ~method                                 # from the cash return by ~ d x r (second order) -> Close+Dividend is right
    r_base = r_adj.where(~method, r_div).mask(unresolved)        # special dividend / spin-off booked twice: do not guess
    log.add_mask(small, "adj_div_mismatch_small", "", "flag_only", values=diff)
    log.add_mask(close_unadj, "close_unadjusted_split", "Close looks unadjusted, Adj Close fine", "keep_adj_return", values=diff)
    log.add_mask(step_no_div & (diff <= CFG["adj_step_max"]), "adj_step_no_dividend", "Adj Close step without dividend in table", "keep_adj_return", values=diff)
    log.add_mask(step_no_div & (diff > CFG["adj_step_max"]), "adj_step_no_dividend", "LARGE step (spin-off / special distribution?) - review", "keep_adj_return", values=diff)
    log.add_mask(method, "adj_div_mismatch_large", "Yahoo proportional adjustment vs cash dividend (regular-size) - used Close+Dividend", "use_close_div_return", values=diff)
    log.add_mask(unresolved, "adj_div_mismatch_large", "unresolved Adj vs Close+Dividend conflict (special dividend/spin-off) - return masked", "nan_return", values=diff)
    return r_base


def rolling_beta(r, rm):
    w, mp = CFG["beta_window"], CFG["beta_min_periods"]
    cov = r.rolling(w, min_periods=mp).cov(rm)
    var = rm.rolling(w, min_periods=mp).var()
    return cov.div(var, axis=0).shift(1)            # information through t-1 only


def detect_outliers(r_base, V, Sp, blk, spy_ticker, log):
    """Returns dict of boolean masks: missed_split, bad_tick, bad_tick_reversal (all -> NaN return)."""
    rm = r_base[spy_ticker]
    beta = rolling_beta(r_base, rm)
    beta[spy_ticker] = 0.0
    resid = r_base - beta.fillna(0.0).mul(rm.fillna(0.0), axis=0)
    sigma = (1.4826 * resid.abs().rolling(CFG["scale_window"], min_periods=CFG["scale_min_periods"]).median()
             .shift(1)).clip(lower=CFG["sigma_floor"])
    abs_trig = pd.Series(np.where(blk.reindex(r_base.columns).to_numpy() == "B", CFG["etf_abs"], CFG["stock_abs"]),
                         index=r_base.columns)
    a = r_base.abs()
    big = a.gt(abs_trig, axis=1)
    sig_ok = resid.abs() > CFG["sigma_k"] * sigma
    no_sigma = sigma.isna() & a.gt(abs_trig * CFG["extreme_abs_mult"], axis=1)
    cand = (big & (sig_ok | no_sigma)).fillna(False)

    sp_near = (Sp != 0) | (Sp.shift(1, fill_value=0.0) != 0) | (Sp.shift(-1, fill_value=0.0) != 0)
    ms = cand & split_like(r_base) & sp_near               # split-like AND Yahoo lists a split there => adjustment failed
    rest = cand & ~ms
    n_alive = (r_base.notna().sum(axis=1)).to_numpy()
    thr = np.maximum(CFG["market_wide_min"], CFG["market_wide_frac"] * n_alive)
    mw_date = rest.sum(axis=1).to_numpy() >= thr
    mw = rest & pd.DataFrame(np.repeat(mw_date[:, None], rest.shape[1], axis=1), index=rest.index, columns=rest.columns)

    vol_med = V.rolling(CFG["vol_window"], min_periods=CFG["vol_min_periods"]).median().shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = V.div(vol_med)
    r_next = r_base.shift(-1)
    net2 = (1.0 + r_base) * (1.0 + r_next) - 1.0
    rev = ((r_base * r_next) < 0) & (net2.abs() <= CFG["reversal_tol"] * a)
    known_low_vol = vr < CFG["event_vol_ratio"]                     # NaN (too little volume history, e.g. IPO week) -> False
    isolated = ~(cand.shift(1, fill_value=False) | cand.shift(-2, fill_value=False))   # a single bad print is not part of a run of big moves
    bt = rest & ~mw & rev & known_low_vol & isolated
    bt_rev = bt.shift(1, fill_value=False)
    others = rest & ~mw & ~bt & ~bt_rev
    bench = pd.DataFrame(False, index=rest.index, columns=rest.columns)
    bench[spy_ticker] = others[spy_ticker]                  # the market proxy itself has no market residual
    others = others & ~bench
    ge = others & (vr >= CFG["event_vol_ratio"])
    unk = others & ~ge

    log.add_mask(ms, "outlier", "missed_split", "nan_return", values=r_base)
    log.add_mask(mw, "outlier", "market_wide", "flag_only", values=r_base)
    log.add_mask(bt, "outlier", "bad_tick", "nan_price+nan_return", values=r_base)
    log.add_mask(bt_rev, "outlier", "bad_tick_reversal", "nan_return", values=r_base)
    log.add_mask(ge, "outlier", "genuine_event", "flag_only", values=r_base)
    log.add_mask(unk, "outlier", "unknown", "flag_only", values=r_base)
    log.add_mask(bench, "outlier", "benchmark_move", "flag_only", values=r_base)
    return dict(missed_split=ms, bad_tick=bt, bad_tick_reversal=bt_rev)


def load_overrides(clean_root):
    f = clean_root / "overrides.csv"
    if not f.exists():
        f.write_text("date,ticker,action,note\n# actions: keep_return | nan_return   (delete this line)\n")
    df = pd.read_csv(f, comment="#", keep_default_na=False)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


def apply_overrides(ret_mask, overrides, log):
    n = 0
    for _, row in overrides.iterrows():
        d, t, act = row["date"], row["ticker"], str(row["action"]).strip()
        if t not in ret_mask.columns or d not in ret_mask.index:
            print(f"  override ignored (unknown date/ticker): {d.date() if hasattr(d, 'date') else d} {t}")
            continue
        if act == "keep_return":
            ret_mask.at[d, t] = False
        elif act == "nan_return":
            ret_mask.at[d, t] = True
        else:
            print(f"  override ignored (unknown action '{act}')")
            continue
        log.add_rows([d], [t], "override", act, "manual")
        n += 1
    return n


def macro_name(t):
    return t.replace("^", "").replace("=F", "_F").replace("-", "_").replace(".", "_")


def process_macro(PC, log):
    close = PC["Close"].copy()
    price_like = [t for t in close.columns if t not in YIELDS]
    nonpos = (close[price_like] <= 0)
    log.add_mask(nonpos, "nonpositive_price", "macro series", "nan_price", values=close[price_like])
    close[price_like] = close[price_like].mask(nonpos)

    levels, chg = {}, {}
    lo, hi = CFG["yield_sanity"]
    for t in close.columns:
        s = close[t]
        n = macro_name(t)
        levels[n] = s
        if t in YIELDS:
            if s.notna().any() and s.median() > 15:
                print(f"WARNING: {t} median level {s.median():.1f} looks x10 scaled (expected percent).")
            bad = (s < lo) | (s > hi)
            log.add_rows(s.index[bad], [t] * int(bad.sum()), "yield_out_of_range", "", "flag_only", values=s[bad].to_numpy())
            chg[f"{n}_chg_bp"] = s.diff() * 100.0
        elif t in VOL_IDX:
            chg[f"{n}_dlog"] = np.log(s / s.shift(1))
        else:
            chg[f"{n}_ret"] = simple_ret(s)
    if "^TNX" in close and "^IRX" in close:
        slope = close["^TNX"] - close["^IRX"]
        levels["SLOPE_10Y_3M"] = slope
        chg["SLOPE_10Y_3M_chg_bp"] = slope.diff() * 100.0
    if "^VIX" in close and "^VIX3M" in close:
        levels["VIX_VIX3M_ratio"] = close["^VIX"] / close["^VIX3M"]
    levels, chg = pd.DataFrame(levels), pd.DataFrame(chg)
    for t in FUTURES:
        c = f"{macro_name(t)}_ret"
        if c in chg:
            j = chg[c].abs() > CFG["future_jump"]
            log.add_rows(chg.index[j], [t] * int(j.sum()), "futures_jump", "roll or event - reference series only",
                         "flag_only", values=chg.loc[j, c].to_numpy())
    return levels, chg


def run_cleaning(raw, overrides):
    log = FlagLog()
    panels, universe, meta = raw["panels"], raw["universe"], raw["meta"]
    tickers = list(panels["Close"].columns)
    blk = classify_blocks(tickers, universe, meta)
    A, B = blk.index[blk == "A"].tolist(), blk.index[blk == "B"].tolist()
    C_, D_ = blk.index[blk == "C"].tolist(), blk.index[blk == "D"].tolist()
    AB = A + B
    print(f"Blocks: A (US stocks) {len(A)} | B (US ETFs) {len(B)} | C (macro) {len(C_)} | D (quarantined) {len(D_)}")
    assert "SPY" in B or "SPY" in A, "SPY is required (market return for the residual test)"

    # ---- Step 2: calendar --------------------------------------------------------
    cal, cal_info = build_master_calendar(panels["Close"][A], panels["Close"]["SPY"], log)
    off = panels["Close"][AB].loc[~panels["Close"].index.isin(cal)].notna().sum().sum()
    print(f"Master calendar: {cal_info['n_days']} trading days ({cal_info['first']} -> {cal_info['last']}); "
          f"US-ticker observations on non-trading dates dropped: {int(off)}")
    P = {f: panels[f].reindex(cal) for f in FIELDS}

    # ---- Step 3: integrity (blocks A+B) ------------------------------------------
    O, H, L = P["Open"][AB], P["High"][AB], P["Low"][AB]
    C, AC, V = P["Close"][AB], P["Adj Close"][AB], P["Volume"][AB]
    Dv, Sp = P["Dividends"][AB].fillna(0.0), P["Stock Splits"][AB].fillna(0.0)
    C, AC, stale = integrity_checks(O, H, L, C, AC, V, log)
    log.add_mask(Sp != 0, "split_recorded", "split in Yahoo table", "flag_only", values=Sp)
    sig = {}
    for t in AB:
        s_ = AC[t].dropna()
        if len(s_) >= 250:
            sig.setdefault(hash(tuple(np.round(s_.to_numpy(), 6))), []).append(t)
    for grp in sig.values():
        if len(grp) > 1:
            log.add_rows([cal[0]] * (len(grp) - 1), grp[1:], "duplicate_series", f"identical to {grp[0]}", "flag_only")

    # ---- Step 4: gaps ------------------------------------------------------------
    stats_AB, internal_AB = gap_analysis(AC.notna(), log)
    unreliable = (stats_AB["max_gap_run"] >= CFG["max_gap_run"]) | \
                 (stats_AB["internal_missing_frac"] > CFG["max_internal_missing_frac"])

    # ---- Step 5: adj vs close+dividend -------------------------------------------
    r_base = reconcile_adj_vs_div(AC, C, Dv, log)

    # ---- Step 6: outliers ----------------------------------------------------------
    spy = "SPY"
    masks = detect_outliers(r_base, V, Sp, blk, spy, log)

    # ---- Step 7: returns -----------------------------------------------------------
    ret_mask = masks["missed_split"] | masks["bad_tick"] | masks["bad_tick_reversal"] | stale
    n_over = apply_overrides(ret_mask, overrides, log)
    ret = r_base.mask(ret_mask).replace([np.inf, -np.inf], np.nan)
    _vals = ret.to_numpy()
    assert not np.isfinite(_vals).any() or np.nanmin(_vals) > -1, "a return <= -100% survived cleaning"

    AC_clean, C_clean = AC.mask(masks["bad_tick"]), C.mask(masks["bad_tick"])
    dollar_vol = C_clean * V
    ret_log = np.log1p(ret)

    # ---- Step 8: macro ---------------------------------------------------------------
    PC = {f: P[f][C_] for f in ["Close"]}
    macro_levels, macro_chg = process_macro(PC, log)
    gap_analysis(macro_levels.notna(), log, rule="internal_gap_macro", action="flag_only", hole_flags=False)

    # risk-free: previous day's 3m yield (percent, annualised) / 100 / 252
    if "IRX" in macro_levels:
        lvl = macro_levels["IRX"].ffill(limit=CFG["rf_ffill_limit"])
        rf = (lvl.shift(1) / 100.0 / 252.0).rename("rf_daily")
    else:
        print("WARNING: ^IRX missing -> risk-free set to 0 for excess returns")
        rf = pd.Series(0.0, index=cal, name="rf_daily")
    ret_excess = ret.sub(rf, axis=0)

    # ---- Block D (quarantined): plain returns, integrity only ---------------------------
    if D_:
        ACd = P["Adj Close"][D_].copy()
        bad = ACd <= 0
        log.add_mask(bad, "nonpositive_price", "quarantined block", "nan_price", values=ACd)
        ret_D = simple_ret(ACd.mask(bad))
        gap_analysis(ACd.mask(bad).notna(), log, rule="internal_gap_quarantine", action="flag_only", hole_flags=False)
    else:
        ret_D = pd.DataFrame(index=cal)

    flags = log.frame()

    # ---- Step 9: universe / eligibility ------------------------------------------------------
    u = stats_AB.copy()
    u.insert(0, "block", blk.reindex(u.index))
    u["groups"] = universe["groups"].reindex(u.index)
    u["wiki_sector"] = universe["wiki_sector"].reindex(u.index) if "wiki_sector" in universe else ""
    for c in ["quoteType", "sector", "industry", "category", "longName"]:
        u[c] = meta[c].reindex(u.index) if (len(meta) and c in meta.columns) else ""
    u["n_valid_returns"] = ret.notna().sum()
    u["days_missing_at_end"] = [int((cal > lv).sum()) if pd.notna(lv) else len(cal) for lv in u["last_valid"]]
    u["median_dollar_volume"] = dollar_vol.apply(lambda s: s.dropna().tail(CFG["dollar_volume_window"]).median())
    u["unreliable"] = unreliable
    u["has_missed_split"] = masks["missed_split"].any()
    u["share_class_secondary"] = [t in SHARE_CLASS_SECONDARY and SHARE_CLASS_SECONDARY[t] in u.index for t in u.index]
    u["zero_volume_frac"] = ((V == 0) & AC.notna()).sum() / AC.notna().sum().replace(0, np.nan)
    u["thin_history"] = u["zero_volume_frac"] > CFG["max_zero_volume_frac"]
    u["coverage_ok"] = u["coverage"] >= CFG["min_coverage"]
    u["liquid"] = u["median_dollar_volume"] >= CFG["min_dollar_volume"]
    u["eligible_core"] = u["coverage_ok"] & ~u["unreliable"] & u["liquid"] & ~u["share_class_secondary"] & ~u["thin_history"]
    for rule in ["stale_price", "ohlc_high_below_open_close", "ohlc_low_above_open_close", "adj_div_mismatch_large",
                 "internal_gap"]:
        u[f"n_{rule}"] = flags[flags["rule"] == rule].groupby("ticker").size().reindex(u.index).fillna(0).astype(int)
    for lab in ["missed_split", "bad_tick", "genuine_event", "unknown", "market_wide"]:
        sub = flags[(flags["rule"] == "outlier") & (flags["label"] == lab)]
        u[f"n_{lab}"] = sub.groupby("ticker").size().reindex(u.index).fillna(0).astype(int)

    # candidate duplicate pairs among block A (report only)
    pairs = []
    if len(A) > 1:
        cm = ret[A].corr(min_periods=250)
        iu = np.triu_indices(len(A), 1)
        vals = cm.to_numpy()[iu]
        sel = np.where(vals >= CFG["high_corr_pair"])[0]
        pairs = [(A[iu[0][k]], A[iu[1][k]], float(vals[k])) for k in sel]
    u["high_corr_partner"] = ""
    for a, b, c in pairs:
        u.at[a, "high_corr_partner"] = (u.at[a, "high_corr_partner"] + f" {b}({c:.4f})").strip()
        u.at[b, "high_corr_partner"] = (u.at[b, "high_corr_partner"] + f" {a}({c:.4f})").strip()

    out = dict(calendar=pd.DataFrame({"date": cal}), cal_info=cal_info,
               adj_close_clean=AC_clean, close_clean=C_clean, volume=V, dollar_volume=dollar_vol,
               returns_simple=ret, returns_log=ret_log, returns_excess=ret_excess, rf_daily=rf.to_frame(),
               macro_levels=macro_levels, macro_changes=macro_chg, returns_quarantine=ret_D,
               flags=flags, universe_clean=u, blocks=blk.to_frame(), high_corr_pairs=pd.DataFrame(pairs, columns=["a", "b", "corr"]),
               n_overrides=n_over, masks=masks, stale=stale)
    return out


GATES = [  # name, series_a (total-return ETF), series_b (index price return), ok, warn, bad
    ("SPY vs ^GSPC", "SPY", "GSPC_ret", 0.998, 0.995, 0.980),
    ("QQQ vs ^NDX", "QQQ", "NDX_ret", 0.998, 0.995, 0.980),
    ("IWM vs ^RUT", "IWM", "RUT_ret", 0.995, 0.990, 0.970),
    ("SMH vs ^SOX", "SMH", "SOX_ret", 0.985, 0.970, 0.930),
]


def status(v, ok, warn, bad):
    return "OK" if v >= ok else "WARN" if v >= warn else "FAIL" if v >= bad else "BAD"


def run_gates(out, universe_raw, raw_div):
    ret, mc = out["returns_simple"], out["macro_changes"]
    rows = []
    for name, a, b, ok, warn, bad in GATES:
        if a in ret.columns and b in mc.columns:
            d = pd.concat([ret[a], mc[b]], axis=1, keys=["a", "b"])
            exdiv = (raw_div[a].reindex(d.index).fillna(0) != 0)        # ETF total return jumps by the dividend
            d_all = d.dropna()
            d = d[~exdiv].dropna()
            if len(d) > 250:
                c = d["a"].corr(d["b"])
                rows.append((name + " (non-ex-div days)", round(c, 5), f">={ok} OK, >={warn} WARN, >={bad} FAIL", status(c, ok, warn, bad), len(d)))
                if name.startswith("SPY"):
                    diff = (d["a"] - d["b"]).abs().sort_values(ascending=False).head(10)
                    print("SPY vs ^GSPC (ex-dividend days of SPY excluded): 10 largest |return differences|:")
                    print(diff.round(5).to_string())
                    spread = (d_all["a"] - d_all["b"]).mean() * 252
                    rows.append(("SPY - ^GSPC mean spread (annualised)", round(spread, 4), "0.5%..3% (dividend yield)",
                                 "OK" if 0.005 <= spread <= 0.03 else "WARN", len(d)))
    cur = universe_raw.index[universe_raw["sp500_now"].astype(str) == "True"] if "sp500_now" in universe_raw else []
    members = [t for t in ret.columns if t in set(cur) and out["blocks"].at[t, "block"] == "A"]
    if "RSP" in ret.columns and len(members) > 20:
        ew = ret[members].mean(axis=1)
        d = pd.concat([ew, ret["RSP"]], axis=1, keys=["a", "b"]).dropna()
        c = d["a"].corr(d["b"])
        rows.append(("EW avg of S&P stocks vs RSP", round(c, 5), ">=0.985 OK, >=0.97 WARN, >=0.90 FAIL", status(c, 0.985, 0.97, 0.90), len(d)))
    return pd.DataFrame(rows, columns=["gate", "value", "thresholds", "status", "n_obs"])


def print_summary(out, gates):
    f, u, ret = out["flags"], out["universe_clean"], out["returns_simple"]
    print("\n=== FLAGS BY RULE / LABEL / ACTION ===")
    print(f.groupby(["rule", "label", "action"]).size().rename("n").to_string())
    print("\n=== RETURN CELLS SET TO NaN BY REASON (A+B) ===")
    m = out["masks"]
    print({"missed_split": int(m["missed_split"].sum().sum()), "bad_tick": int(m["bad_tick"].sum().sum()),
           "bad_tick_reversal": int(m["bad_tick_reversal"].sum().sum()), "stale": int(out["stale"].sum().sum())})
    cnt = ret.notna().sum(axis=1).iloc[1:]
    print(f"\nValid tickers per date: min {cnt.min()}, median {int(cnt.median())}, max {cnt.max()}")
    print(cnt.groupby(cnt.index.year).median().astype(int).to_string())
    print(f"\nUniverse (A+B): {len(u)} | eligible_core: {int(u['eligible_core'].sum())} | unreliable: {int(u['unreliable'].sum())} "
          f"| share-class secondary: {int(u['share_class_secondary'].sum())} | with missed split: {int(u['has_missed_split'].sum())}")
    if len(out["high_corr_pairs"]):
        print("Highly correlated stock pairs (possible duplicate share classes):")
        print(out["high_corr_pairs"].round(4).to_string(index=False))
    print("\n=== VALIDATION GATES ===")
    print(gates.to_string(index=False))


OUT_FRAMES = ["calendar", "adj_close_clean", "close_clean", "volume", "dollar_volume", "returns_simple", "returns_log",
              "returns_excess", "rf_daily", "macro_levels", "macro_changes", "returns_quarantine", "flags",
              "universe_clean", "blocks", "gates"]


def save_outputs(clean_root, out, gates, key, cal_info):
    out = dict(out, gates=gates)
    for name in OUT_FRAMES:
        df = out[name]
        if name in ("universe_clean", "blocks", "gates"):
            df.to_csv(clean_root / f"{name}.csv")
        if name == "flags":
            df.to_csv(clean_root / "flags.csv", index=False)
        df.to_parquet(clean_root / f"{name}.parquet")
    (clean_root / "config.json").write_text(json.dumps(CFG, indent=2, default=str))
    (clean_root / "done.json").write_text(json.dumps(
        {"key": key, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "pandas": pd.__version__, "python": platform.python_version(), "calendar": cal_info}, indent=2, default=str))


def load_outputs(clean_root):
    return {name: pd.read_parquet(clean_root / f"{name}.parquet") for name in OUT_FRAMES}


def get_clean(force=FORCE_RECOMPUTE):
    base = mount_drive()
    raw_root = base / RAW_VERSION
    clean_root = base / "clean" / f"{CLEAN_VERSION}__{RAW_VERSION}"
    clean_root.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(clean_root)
    meta_txt = (raw_root / "panel" / "panel_meta.json").read_text()
    ov_txt = (clean_root / "overrides.csv").read_text()
    key = cfg_hash(meta_txt + ov_txt + START + END_INCLUSIVE)

    done = clean_root / "done.json"
    if done.exists() and not force and json.loads(done.read_text()).get("key") == key:
        print("Cache hit: clean data already on Drive (config, raw data and overrides unchanged) -> loading.")
        out = load_outputs(clean_root)
        return out

    print("Cleaning from raw ...")
    raw = read_raw(raw_root)
    out = run_cleaning(raw, overrides)
    gates = run_gates(out, raw["universe"], raw["panels"]["Dividends"])
    print_summary(out, gates)
    bad = gates[gates["status"] == "BAD"]
    if STRICT_GATES and len(bad):
        raise AssertionError(f"Validation gate(s) BAD - clean output NOT saved:\n{bad.to_string(index=False)}")
    save_outputs(clean_root, out, gates, key, out["cal_info"])
    print(f"\nSaved to {clean_root}")
    out["gates"] = gates
    return out
