"""Computation layer for the interactive factor explorer (issue #12).

Everything here is plain pandas/numpy and has no Streamlit dependency, so it can be
tested and reused from notebooks.  The regression itself is NOT re-implemented: the
fits go through ``factor_lens.linear_model`` (``fit_window``, ``apply_transform``,
``fitted_from_params``, ``nw_multi``, ``rolling_fit``, ``causal_attribution``,
``monthly_attribution``).  Only the cached outputs under ``factor_lens_local/`` are
read: nothing is downloaded and the pipeline is not rerun.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from factor_lens import linear_model as lm  # noqa: E402

BASE = Path(os.environ.get("FACTOR_LENS_LOCAL", ROOT / "factor_lens_local"))
CLEAN_ROOT = BASE / "clean" / f"{lm.CLEAN_VERSION}__{lm.RAW_VERSION}"
FAC_ROOT = BASE / "factors" / f"{lm.FACTOR_VERSION}__{lm.CLEAN_VERSION}__{lm.RAW_VERSION}"
MODEL_ROOT = BASE / "models" / f"{lm.MODEL_VERSION}__{lm.FACTOR_VERSION}__{lm.CLEAN_VERSION}__{lm.RAW_VERSION}"

AI_LAYERS = ["compute_chips", "network_servers", "hyperscalers_software", "datacenter_neocloud",
             "power_generation", "grid_electrical", "nuclear_uranium"]
TIER_ORDER = {"distant": 0, "adjacent": 1, "close": 2}
DEFAULT_PRESET = lm.PRIMARY_MODEL


# ----------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------
def load_data(base: Path | None = None) -> dict:
    """Read the cached pipeline outputs once.  Returns a dict of frames used by the app."""
    global CLEAN_ROOT, FAC_ROOT, MODEL_ROOT
    if base is not None:
        CLEAN_ROOT = base / "clean" / f"{lm.CLEAN_VERSION}__{lm.RAW_VERSION}"
        FAC_ROOT = base / "factors" / f"{lm.FACTOR_VERSION}__{lm.CLEAN_VERSION}__{lm.RAW_VERSION}"
        MODEL_ROOT = base / "models" / f"{lm.MODEL_VERSION}__{lm.FACTOR_VERSION}__{lm.CLEAN_VERSION}__{lm.RAW_VERSION}"
        inp = lm.load_inputs(base)
    else:
        inp = lm.load_inputs(BASE)
    Fn = pd.read_parquet(FAC_ROOT / "factors_neutral.parquet")
    Fn.index = pd.DatetimeIndex(Fn.index)
    Fn = Fn.sort_index()
    catalog = pd.read_csv(FAC_ROOT / "factor_catalog.csv")
    universe = pd.read_parquet(CLEAN_ROOT / "universe_clean.parquet")

    # Baskets exactly as notebook 04 builds them (layers, ALL, ALL_core, overlap-free yardsticks, validation portfolios).
    Yb, kind_b, lay_b = lm.build_targets(inp)
    baskets = Yb.loc[:, kind_b == "basket"]

    F = inp["F"]
    idx = F.index.union(inp["rx"].index)
    data = dict(
        F_raw=F.reindex(idx),
        F_neu=Fn.reindex(idx),
        rx=inp["rx"].reindex(idx),
        baskets=baskets.reindex(idx),
        rf=inp["rf"].reindex(idx).ffill(limit=3),
        members=inp["members"],
        catalog=catalog,
        universe=universe,
    )
    data["targets"] = target_table(data)
    data["factor_table"] = factor_table(data)
    data["ladder"] = ladder_stages()
    return data


def target_table(data: dict) -> pd.DataFrame:
    """One row per selectable target: baskets first, then every return series in the universe."""
    mem = data["members"].drop_duplicates("ticker").set_index("ticker")
    uni = data["universe"]
    rows = []
    for b in data["baskets"].columns:
        if b in AI_LAYERS:
            grp = "AI layer basket"
        elif b in ("ALL", "ALL_core"):
            grp = "AI aggregate basket"
        elif b.startswith("ALL_core_ex"):
            grp = "Overlap-free yardstick"
        else:
            grp = "Validation portfolio"
        rows.append(dict(target=b, kind="basket", name=b.replace("_", " "), layer=b if b in AI_LAYERS else grp,
                         sector="", group=grp, weighting="equal"))
    controls = set(lm.NON_AI_CONTROL_STOCKS)
    for t in data["rx"].columns:
        u = uni.loc[t] if t in uni.index else None
        layer = mem.at[t, "layer"] if t in mem.index else ("non_AI_control" if t in controls else "")
        sector = (u["sector"] or u["wiki_sector"]) if u is not None else ""
        etf = u is not None and u["quoteType"] == "ETF"
        grp = "AI basket member" if t in mem.index else ("Non-AI control" if t in controls else ("ETF" if etf else "Other stock"))
        rows.append(dict(target=t, kind="etf" if etf else "stock", name=(u["longName"] if u is not None else "") or t,
                         layer=layer, sector=sector, group=grp, weighting=""))
    return pd.DataFrame(rows)


def factor_table(data: dict) -> pd.DataFrame:
    """The factor catalog restricted to the factors that exist in the cached factor panel."""
    cat = data["catalog"].copy()
    cat["available_raw"] = cat["name"].isin(data["F_raw"].columns)
    cat["available_neutral"] = cat["name"].isin(data["F_neu"].columns) | (cat["name"] == "MKT")
    cat["tier_rank"] = cat["ai_tier"].map(TIER_ORDER)
    return cat


def ladder_stages() -> list[str]:
    """The notebook-03 ladder order (MKT first, then the stages as they were added)."""
    p = FAC_ROOT / "ladder_stages.csv"
    order = ["MKT"]
    if p.exists():
        for added in pd.read_csv(p)["added"].dropna():
            order += [f for f in str(added).split(",") if f and f not in order]
    return order


def presets(data: dict) -> dict[str, list[str]]:
    out = {k: list(v) for k, v in lm.MODELS.items()}
    out["Ladder (notebook 03, 13 factors)"] = data["ladder"]
    avail = data["factor_table"]
    out["All available factors"] = ["MKT"] + [f for f in avail.loc[avail["available_raw"], "name"] if f != "MKT"]
    return out


# ----------------------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    target: str
    factors: tuple[str, ...]                 # ordered: orthogonalisation order (MKT should be first)
    neutral: bool = False                    # market-neutral factor versions (MKT stays raw)
    orthogonal: bool = True                  # report Gram-Schmidt (orthogonal) betas; False => classic raw OLS betas
    freq: str = "D"                          # "D" daily, "W" weekly (sum of daily excess returns / changes, Friday close)
    est: tuple[str, str] = (lm.EST_START, lm.EST_END)
    hold: tuple[str, str] = (lm.HOLD_START, lm.HOLD_END)
    window: int = lm.CFG["window"]           # rolling window, in periods of `freq`
    nw_lags: int = lm.CFG["nw_lags"]

    @property
    def ann(self) -> int:
        return 252 if self.freq == "D" else 52

    @property
    def min_rows(self) -> int:
        return min(lm.CFG["min_rows"], int(0.8 * self.window)) if self.freq == "D" else max(20, int(0.8 * self.window))


def is_default(spec: Spec) -> bool:
    return (spec.factors == tuple(lm.MODELS[lm.PRIMARY_MODEL]) and not spec.neutral and spec.freq == "D"
            and spec.est == (lm.EST_START, lm.EST_END) and spec.hold == (lm.HOLD_START, lm.HOLD_END)
            and spec.window == lm.CFG["window"] and spec.nw_lags == lm.CFG["nw_lags"])


def _series(data: dict, spec: Spec) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Factor matrix (columns in spec order), target excess return, risk-free rate at the chosen frequency."""
    if spec.neutral:
        F = pd.concat([data["F_raw"][["MKT"]], data["F_neu"]], axis=1)
    else:
        F = data["F_raw"]
    missing = [f for f in spec.factors if f not in F.columns]
    if missing:
        raise KeyError(f"factors not in the cached panel ({'neutral' if spec.neutral else 'raw'}): {missing}")
    F = F[list(spec.factors)]
    y = data["baskets"][spec.target] if spec.target in data["baskets"].columns else data["rx"][spec.target]
    rf = data["rf"]
    if spec.freq == "W":
        # Weekly: sum daily values within the week (log-return approximation; bp / dlog factors are additive anyway).
        keep = lambda s: s.notna().groupby(pd.Grouper(freq="W-FRI")).sum()
        n_f = F.notna().all(axis=1).groupby(pd.Grouper(freq="W-FRI")).sum()
        F = F.groupby(pd.Grouper(freq="W-FRI")).sum(min_count=1).where(n_f >= 3)
        y = y.groupby(pd.Grouper(freq="W-FRI")).sum(min_count=1).where(keep(y) >= 3)
        rf = rf.groupby(pd.Grouper(freq="W-FRI")).sum(min_count=1)
    return F, y, rf


# ----------------------------------------------------------------------------------------------------------
# static model (estimation window fit, holdout evaluation)
# ----------------------------------------------------------------------------------------------------------
def raw_from_orth(par: dict, n: int = 0) -> tuple[np.ndarray, float]:
    """Translate the orthogonal-basis coefficients of one target back to classic raw-factor betas.

    ``apply_transform`` gives z = M (x - mu) with M = (I + Phi)^-1 (unit lower triangular), so
    gamma'z = (M' gamma)'(x - mu): the raw betas are b = M' gamma and the raw intercept is ybar - b'mu.
    """
    K = par["Phi"].shape[0]
    M = np.linalg.inv(np.eye(K) + par["Phi"])
    b = M.T @ par["gamma"][n]
    return b, float(par["ybar"][n] - b @ par["mu"])


def static_fit(data: dict, spec: Spec) -> dict | None:
    """Single-target version of ``linear_model.static_model`` with free dates/frequency."""
    F, y, rf = _series(data, spec)
    facs = list(spec.factors)
    K = len(facs)
    Fe, Fh = F.loc[spec.est[0]:spec.est[1]], F.loc[spec.hold[0]:spec.hold[1]]
    ye_s, yh_s = y.reindex(Fe.index), y.reindex(Fh.index)
    min_obs = spec.min_rows
    par = lm.fit_window(Fe.to_numpy(float), ye_s.to_numpy(float)[:, None], spec.min_rows, min_obs)
    if par is None or np.isnan(par["r2"][0]):
        return None
    okE = ~Fe.isna().any(axis=1).to_numpy()
    okH = ~Fh.isna().any(axis=1).to_numpy()
    XE, XH = Fe.to_numpy(float)[okE], Fh.to_numpy(float)[okH]
    ZE = lm.apply_transform(XE, par["mu"], par["Phi"])
    ZH = lm.apply_transform(XH, par["mu"], par["Phi"])
    ye, yh = ye_s.to_numpy(float)[okE], yh_s.to_numpy(float)[okH]
    me, mh = ~np.isnan(ye), ~np.isnan(yh)

    Xz = np.column_stack([np.ones(me.sum()), ZE[me]])
    bz, sez = lm.nw_multi(ye[me], Xz, spec.nw_lags)
    Xr = np.column_stack([np.ones(me.sum()), XE[me]])
    br, ser = lm.nw_multi(ye[me], Xr, spec.nw_lags)

    fit_e, _ = lm.fitted_from_params(par, XE)
    fit_h, _ = lm.fitted_from_params(par, XH)
    n_obs = int(me.sum())
    r2 = float(par["r2"][0])
    r2_oos = r2_refit = np.nan
    gamma_h = np.full(K, np.nan)
    t_h = np.full(K, np.nan)
    if mh.sum() >= (50 if spec.freq == "D" else 15):
        sse = ((yh[mh] - fit_h[mh, 0]) ** 2).sum()
        r2_oos = 1.0 - sse / ((yh[mh] - par["ybar"][0]) ** 2).sum()
        Xzh = np.column_stack([np.ones(mh.sum()), ZH[mh]])
        bh, seh = lm.nw_multi(yh[mh], Xzh, spec.nw_lags)
        r2_refit = 1.0 - ((yh[mh] - Xzh @ bh) ** 2).sum() / ((yh[mh] - yh[mh].mean()) ** 2).sum()
        gamma_h, t_h = bh[1:], (bh[1:] - bz[1:]) / np.sqrt(sez[1:] ** 2 + seh[1:] ** 2)

    b_raw, a_raw = raw_from_orth(par)
    coef = pd.DataFrame(dict(
        factor=facs, gamma=bz[1:], t_gamma=bz[1:] / sez[1:], raw_beta=br[1:], t_raw=br[1:] / ser[1:],
        effect_1sd_bp=bz[1:] * par["zsd"] * 1e4, var_share=par["share"][0], gamma_hold=gamma_h, z_shift=t_h,
    ))
    summary = dict(
        target=spec.target, n_est=n_obs, n_hold=int(mh.sum()), r2_est=r2,
        adj_r2_est=1.0 - (1.0 - r2) * (n_obs - 1) / max(n_obs - K - 1, 1),
        r2_oos_hold=r2_oos, r2_hold_refit=r2_refit,
        resid_vol_ann=float(np.sqrt(par["rvar"][0] * spec.ann)),
        total_vol_ann=float(np.std(ye[me], ddof=1) * np.sqrt(spec.ann)),
        alpha_ann=float(par["alpha"][0] * spec.ann), alpha_raw_ann=a_raw * spec.ann,
    )
    # actual vs fitted over estimation + holdout with the estimation parameters
    idx = Fe.index[okE].append(Fh.index[okH])
    actual = np.concatenate([ye, yh])
    fitted = np.concatenate([fit_e[:, 0], fit_h[:, 0]])
    path = pd.DataFrame(dict(actual=actual, fitted=fitted, resid=actual - fitted,
                             sample=["estimation"] * okE.sum() + ["holdout"] * okH.sum()), index=idx)
    return dict(summary=summary, coef=coef, par=par, path=path.dropna(subset=["actual"]), b_raw=b_raw)


def r2_only(data: dict, spec: Spec) -> tuple[float, float, float]:
    """(in-sample R2, adjusted R2, holdout OOS R2) — the cheap part of ``static_fit`` used by ladders."""
    F, y, _ = _series(data, spec)
    Fe, Fh = F.loc[spec.est[0]:spec.est[1]], F.loc[spec.hold[0]:spec.hold[1]]
    par = lm.fit_window(Fe.to_numpy(float), y.reindex(Fe.index).to_numpy(float)[:, None], spec.min_rows, spec.min_rows)
    if par is None or np.isnan(par["r2"][0]):
        return (np.nan, np.nan, np.nan)
    r2, n, K = float(par["r2"][0]), par["nobs"][0], len(spec.factors)
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - K - 1, 1)
    okH = ~Fh.isna().any(axis=1).to_numpy()
    yh = y.reindex(Fh.index).to_numpy(float)[okH]
    fit_h, _ = lm.fitted_from_params(par, Fh.to_numpy(float)[okH])
    m = ~np.isnan(yh)
    oos = np.nan
    if m.sum() >= (50 if spec.freq == "D" else 15):
        oos = 1.0 - ((yh[m] - fit_h[m, 0]) ** 2).sum() / ((yh[m] - par["ybar"][0]) ** 2).sum()
    return r2, float(adj), float(oos)


def ladder(data: dict, spec: Spec) -> pd.DataFrame:
    """R2 as factors are added one by one in the spec's order."""
    rows = []
    for k in range(1, len(spec.factors) + 1):
        sub = Spec(**{**spec.__dict__, "factors": spec.factors[:k]})
        r2, adj, oos = r2_only(data, sub)
        rows.append(dict(k=k, added=spec.factors[k - 1], r2_est=r2, adj_r2_est=adj, r2_oos_hold=oos))
    df = pd.DataFrame(rows)
    df["delta_r2"] = df["r2_est"].diff().fillna(df["r2_est"])
    return df


def rank_by_incremental_r2(data: dict, spec: Spec) -> list[str]:
    """Greedy forward ordering on the estimation window: MKT first, then whichever factor adds the most R2."""
    pool = [f for f in spec.factors if f != "MKT"]
    chosen = ["MKT"] if "MKT" in spec.factors else []
    while pool:
        best, best_r2 = None, -np.inf
        for f in pool:
            r2 = r2_only(data, Spec(**{**spec.__dict__, "factors": tuple(chosen + [f])}))[0]
            if r2 > best_r2:
                best, best_r2 = f, r2
        chosen.append(best)
        pool.remove(best)
    return chosen


# ----------------------------------------------------------------------------------------------------------
# collinearity
# ----------------------------------------------------------------------------------------------------------
def collinearity(data: dict, spec: Spec) -> dict:
    """VIF of each raw factor and the condition number of the standardised factor matrix (estimation window)."""
    F, _, _ = _series(data, spec)
    X = F.loc[spec.est[0]:spec.est[1]].dropna()
    if len(spec.factors) < 2 or len(X) < 10:
        return dict(vif=pd.DataFrame(dict(factor=list(spec.factors), VIF=1.0)), cond=1.0, corr=X.corr())
    corr = X.corr()
    try:
        vif = np.diag(np.linalg.inv(corr.to_numpy()))
    except np.linalg.LinAlgError:
        vif = np.full(len(corr), np.inf)
    ev = np.linalg.eigvalsh(corr.to_numpy())
    cond = float(np.sqrt(ev.max() / max(ev.min(), 1e-12)))
    return dict(vif=pd.DataFrame(dict(factor=corr.index, VIF=vif)), cond=cond, corr=corr)


# ----------------------------------------------------------------------------------------------------------
# rolling model, causal attribution, risk shares
# ----------------------------------------------------------------------------------------------------------
def rolling(data: dict, spec: Spec) -> dict:
    F, y, rf = _series(data, spec)
    idx = F.index
    Fv = F.to_numpy(float)
    Yv = y.reindex(idx).to_numpy(float)[:, None]
    rf_v = rf.reindex(idx).to_numpy(float)
    roll = lm.rolling_fit(Fv, Yv, spec.window, spec.min_rows, spec.min_rows)
    attr = lm.causal_attribution(Fv, Yv, roll)
    facs = list(spec.factors)
    valid = ~np.isnan(roll["r2"][:, 0])
    dates = idx[valid]

    if spec.orthogonal:
        beta = roll["gamma"][valid, 0, :]
        C = attr["C"][:, 0, :]
        alpha = attr["alpha"][:, 0]
    else:
        # raw-basis betas window by window, and the matching causal contributions b_{t-1} * x_t
        beta = np.full((valid.sum(), len(facs)), np.nan)
        b_all = np.full((len(idx), len(facs)), np.nan)
        a_all = np.full(len(idx), np.nan)
        for i in np.where(valid)[0]:
            p = dict(gamma=roll["gamma"][i], Phi=roll["Phi"][i], mu=roll["mu"][i], ybar=roll["ybar"][i])
            b_all[i], a_all[i] = raw_from_orth(p)
        beta = b_all[valid]
        C = np.full((len(idx), len(facs)), np.nan)
        alpha = np.full(len(idx), np.nan)
        C[1:] = b_all[:-1] * Fv[1:]
        alpha[1:] = a_all[:-1]
        bad = np.isnan(Fv).any(axis=1) | np.isnan(attr["C"][:, 0, :]).any(axis=1)
        C[bad], alpha[bad] = np.nan, np.nan
    fitted = alpha + C.sum(axis=1)
    resid = Yv[:, 0] - fitted
    attr_v = dict(C=C[:, None, :], alpha=alpha[:, None], fitted=fitted[:, None], resid=resid[:, None])

    betas = pd.DataFrame(beta, index=dates, columns=facs)
    r2 = pd.Series(roll["r2"][valid, 0], index=dates, name="r2")
    monthly = lm.monthly_attribution(attr_v, Yv, rf_v, idx, [spec.target], facs, None)
    # risk shares (orthogonal decomposition: they add up to R2 regardless of the beta view)
    months = pd.PeriodIndex(idx, freq="M")
    last = pd.Series(np.arange(len(idx)), index=months).groupby(level=0).max()
    risk = []
    for m, i in last.items():
        if np.isnan(roll["r2"][i, 0]):
            continue
        row = {f: roll["share"][i, 0, k] for k, f in enumerate(facs)}
        row["RESIDUAL"] = 1.0 - roll["r2"][i, 0]
        row["month"] = str(m)
        risk.append(row)
    oos_daily = pd.DataFrame(dict(actual=Yv[:, 0], fitted=fitted), index=idx).dropna()
    return dict(betas=betas, r2=r2, monthly=monthly, risk=pd.DataFrame(risk), causal=oos_daily)


# ----------------------------------------------------------------------------------------------------------
# reproduction check against factor_lens_local/models
# ----------------------------------------------------------------------------------------------------------
def check_against_cache(data: dict, targets: list[str] | None = None, tol: float = 1e-6) -> pd.DataFrame:
    """Re-fit the default (primary-model, static) spec and compare with the cached notebook-04 tables."""
    fit_c = pd.read_parquet(MODEL_ROOT / "static_fit.parquet").set_index("target")
    coef_c = pd.read_parquet(MODEL_ROOT / "static_coef.parquet")
    rows = []
    for t in targets or list(fit_c.index):
        res = static_fit(data, Spec(target=t, factors=tuple(lm.MODELS[lm.PRIMARY_MODEL])))
        if res is None:
            rows.append(dict(target=t, ok=False, max_abs_diff=np.nan, note="no fit"))
            continue
        s = res["summary"]
        diffs = [abs(s[c] - fit_c.at[t, c]) for c in ("r2_est", "adj_r2_est", "r2_oos_hold", "r2_hold_refit", "resid_vol_ann", "alpha_ann")
                 if not (np.isnan(s[c]) and np.isnan(fit_c.at[t, c]))]
        cc = coef_c[coef_c["target"] == t].set_index("factor")
        mine = res["coef"].set_index("factor")
        for c in ("gamma", "t_gamma", "raw_beta", "t_raw", "effect_1sd_bp", "var_share"):
            d = (mine[c] - cc[c].reindex(mine.index)).abs()
            scale = cc[c].reindex(mine.index).abs().clip(lower=1.0)
            diffs.append(float((d / scale).max()))
        mx = float(np.nanmax(diffs))
        rows.append(dict(target=t, ok=bool(mx < tol), max_abs_diff=mx, note=""))
    return pd.DataFrame(rows)
