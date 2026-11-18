"""Interactive factor explorer (issue #12).

Run from the repository root:

    streamlit run app/factor_explorer.py

Pick a stock or basket, choose which factors go into the model and in which order, and see
fit, betas, ladder, rolling exposures, actual-vs-fitted and P&L attribution update.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

st.set_page_config(page_title="Factor Explorer", page_icon="📈", layout="wide")

TIER_BADGE = {"distant": "AI-distant", "adjacent": "AI-adjacent", "close": "AI-close"}
TIER_COLOR = {"distant": "#7f8c9a", "adjacent": "#e0a030", "close": "#d0453b"}
COLLINEAR_VIF, COLLINEAR_COND = 10.0, 30.0


# ----------------------------------------------------------------------------------------------------------
# cached computations (keyed by the frozen Spec, so toggling back and forth is instant)
# ----------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading cached pipeline outputs from factor_lens_local/ ...")
def data():
    return E.load_data()


@st.cache_data(show_spinner=False, max_entries=512)
def static_fit(spec: E.Spec):
    res = E.static_fit(data(), spec)
    if res is None:
        return None
    return {k: v for k, v in res.items() if k != "par"}


@st.cache_data(show_spinner=False, max_entries=256)
def ladder(spec: E.Spec):
    return E.ladder(data(), spec)


@st.cache_data(show_spinner=False, max_entries=128)
def ranked(spec: E.Spec):
    return E.rank_by_incremental_r2(data(), spec)


@st.cache_data(show_spinner=False, max_entries=128)
def rolling(spec: E.Spec):
    return E.rolling(data(), spec)


@st.cache_data(show_spinner=False, max_entries=256)
def collinearity(spec: E.Spec):
    return E.collinearity(data(), spec)


@st.cache_data(show_spinner="Re-fitting the default model for every cached target ...")
def reproduction_check():
    return E.check_against_cache(data())


D = data()
TARGETS = D["targets"]
FT = D["factor_table"]
PRESETS = E.presets(D)


def fmt(x, p=3, pct=False):
    if x is None or pd.isna(x):
        return "–"
    return f"{x:.1%}" if pct else f"{x:.{p}f}"


def _label(r) -> str:
    s = r.target
    if r.kind != "basket" and r.name and r.name != r.target:
        s += f" · {str(r.name)[:38]}"
    tag = r.layer or r.sector
    return s + (f"  [{tag}]" if tag else "")


LABELS = {r.target: _label(r) for r in TARGETS.itertuples()}


def target_label(t: str) -> str:
    return LABELS.get(t, t)


# ----------------------------------------------------------------------------------------------------------
# sidebar: target, factors, settings
# ----------------------------------------------------------------------------------------------------------
def target_picker(key: str, default: str = "ALL_core") -> str:
    scopes = {
        "AI baskets (layers, ALL, ALL_core, yardsticks)": TARGETS["kind"] == "basket",
        "AI basket members": TARGETS["group"] == "AI basket member",
        "Non-AI controls": TARGETS["group"] == "Non-AI control",
        "Full universe (stocks)": TARGETS["kind"] == "stock",
        "ETFs": TARGETS["kind"] == "etf",
        "Everything": pd.Series(True, index=TARGETS.index),
    }
    scope = st.selectbox("Universe", list(scopes), key=f"{key}_scope")
    pool = TARGETS[scopes[scope]]
    if scope == "AI basket members":
        layers = st.multiselect("Layers", E.AI_LAYERS, default=E.AI_LAYERS, key=f"{key}_layers",
                                format_func=lambda s: s.replace("_", " "))
        pool = pool[pool["layer"].isin(layers)]
    elif scope == "Full universe (stocks)":
        sectors = sorted(s for s in pool["sector"].unique() if s)
        pick = st.multiselect("Sectors", sectors, key=f"{key}_sectors", placeholder="All sectors")
        if pick:
            pool = pool[pool["sector"].isin(pick)]
    options = pool["target"].tolist()
    if not options:
        st.warning("No targets match these filters.")
        st.stop()
    idx = options.index(default) if default in options else 0
    return st.selectbox(f"Target ({len(options)}) — type to search ticker or name", options, index=idx,
                        format_func=target_label, key=f"{key}_target")


def factor_picker(neutral: bool) -> list[str]:
    """Checklist of catalog factors with an editable orthogonalisation order."""
    preset = st.selectbox("Preset", list(PRESETS) + ["Custom (keep current)"],
                          index=list(PRESETS).index(E.DEFAULT_PRESET), key="preset",
                          help="M3_plus_AI_close is the 12-factor model of notebook 04.")
    avail_col = "available_neutral" if neutral else "available_raw"
    if "factor_editor_df" not in st.session_state or st.session_state.get("_preset_applied") != preset:
        if preset != "Custom (keep current)":
            chosen = PRESETS[preset]
            df = FT[["name", "bucket", "ai_tier", "story", avail_col]].rename(columns={"name": "factor", avail_col: "available"})
            df["use"] = df["factor"].isin(chosen)
            df["order"] = [chosen.index(f) + 1 if f in chosen else None for f in df["factor"]]
            df["tier"] = df["ai_tier"].map(TIER_BADGE)
            df = df.sort_values(["use", "order", "ai_tier", "bucket"], ascending=[False, True, True, True], na_position="last")
            st.session_state["factor_editor_df"] = df[["use", "order", "factor", "tier", "bucket", "story", "available"]].reset_index(drop=True)
            st.session_state["editor_ver"] = st.session_state.get("editor_ver", 0) + 1
        st.session_state["_preset_applied"] = preset

    base = st.session_state["factor_editor_df"].copy()
    base["available"] = base["factor"].map(FT.set_index("name")[avail_col])
    tiers = st.multiselect("Show tiers", list(TIER_BADGE.values()), default=list(TIER_BADGE.values()), key="tier_filter")
    view = base[base["tier"].isin(tiers) | base["use"]]
    edited = st.data_editor(
        view, key=f"factor_editor_{st.session_state['editor_ver']}", hide_index=True, width="stretch",
        height=min(38 * (len(view) + 1), 460),
        column_config={
            "use": st.column_config.CheckboxColumn("✓", width="small"),
            "order": st.column_config.NumberColumn("Order", min_value=1, max_value=99, step=1, width="small",
                                                   help="Gram–Schmidt order: earlier factors are removed from later ones. Edit to reorder."),
            "factor": st.column_config.TextColumn("Factor", disabled=True, width="small"),
            "tier": st.column_config.TextColumn("Tier", disabled=True, width="small"),
            "bucket": st.column_config.TextColumn("Bucket", disabled=True),
            "story": st.column_config.TextColumn("What it measures", disabled=True, width="large"),
            "available": st.column_config.CheckboxColumn("In data", disabled=True, width="small",
                                                         help="Factors dropped as ineligible in notebook 03 are not in the cached panel."),
        },
    )
    merged = base.set_index("factor")
    merged.update(edited.set_index("factor")[["use", "order"]])
    sel = merged[merged["use"].astype(bool)].reset_index()
    unavailable = sel.loc[~sel["available"].astype(bool), "factor"].tolist()
    if unavailable:
        st.caption(f"⚠️ Not in the cached {'neutral' if neutral else 'raw'} panel, skipped: {', '.join(unavailable)}")
    sel = sel[sel["available"].astype(bool)]
    sel["order"] = pd.to_numeric(sel["order"], errors="coerce").fillna(999)
    order = sel.sort_values(["order"], kind="stable")["factor"].tolist()
    if "MKT" in order:                         # MKT enters raw and first, as in notebooks 03b/04
        order = ["MKT"] + [f for f in order if f != "MKT"]
    return order


with st.sidebar:
    st.title("📈 Factor Explorer")
    st.caption("AI Factor Lens · interactive view on the cached pipeline outputs")
    st.subheader("1 · Target")
    target = target_picker("A")

    st.subheader("3 · Model settings")
    neutral = st.toggle("Market-neutral factors", value=False, help="Use `factors_neutral` (each factor with its market beta removed); MKT stays raw.")
    orthogonal = st.toggle("Gram–Schmidt orthogonalisation", value=True,
                           help="On: orthogonal betas (each factor's effect net of the earlier ones). Off: classic multiple-regression betas. "
                                "R² is identical either way; only the betas and the attribution split change.")
    freq = st.radio("Return frequency", ["Daily", "Weekly"], horizontal=True)
    fcode = "D" if freq == "Daily" else "W"
    c1, c2 = st.columns(2)
    est0 = c1.date_input("Estimation start", pd.Timestamp(E.lm.EST_START))
    est1 = c2.date_input("Estimation end", pd.Timestamp(E.lm.EST_END))
    hol0 = c1.date_input("Holdout start", pd.Timestamp(E.lm.HOLD_START))
    hol1 = c2.date_input("Holdout end", pd.Timestamp(E.lm.HOLD_END))
    win_default = 252 if fcode == "D" else 52
    window = st.slider("Rolling window (periods)", 60 if fcode == "D" else 26, 756 if fcode == "D" else 156, win_default,
                       step=21 if fcode == "D" else 4)
    nw = st.number_input("Newey–West lags", 0, 20, 5 if fcode == "D" else 2)

st.header("2 · Factors")
fc1, fc2 = st.columns([3, 2], gap="large")
with fc1:
    chosen = factor_picker(neutral)
if not chosen:
    st.info("Tick at least one factor.")
    st.stop()

base_spec = E.Spec(target=target, factors=tuple(chosen), neutral=neutral, orthogonal=orthogonal, freq=fcode,
                   est=(str(est0), str(est1)), hold=(str(hol0), str(hol1)), window=int(window), nw_lags=int(nw))

with fc2:
    order_mode = st.radio("Add factors in", ["Chosen order", "Ranked by incremental R²"], horizontal=True,
                          help="Ranking is a greedy forward selection on the estimation window for this target (MKT first).")
    full_order = chosen if order_mode == "Chosen order" else ranked(base_spec)
    k = st.slider("Number of factors", 1, len(full_order), len(full_order))
    factors = full_order[:k]
    spec = replace(base_spec, factors=tuple(factors))
    st.markdown("**Model:** " + " → ".join(
        f"<span style='color:{TIER_COLOR[FT.set_index('name').at[f, 'ai_tier']]}'>{f}</span>" for f in factors),
        unsafe_allow_html=True)
    if len(full_order) > k:
        st.caption("Left out by the slider: " + ", ".join(full_order[k:]))

    col = collinearity(spec)
    vif = col["vif"].set_index("factor")["VIF"]
    worst = vif.idxmax() if len(vif) else None
    m1, m2 = st.columns(2)
    m1.metric("Condition number", f"{col['cond']:.1f}", help=f"Standardised factor matrix, estimation window. > {COLLINEAR_COND:.0f} = collinear.")
    m2.metric("Max VIF", f"{vif.max():.1f}" if len(vif) else "–", worst or "", delta_color="off",
              help=f"Raw factors, estimation window. > {COLLINEAR_VIF:.0f} = collinear.")
    high = vif[vif > COLLINEAR_VIF]
    if col["cond"] > COLLINEAR_COND or len(high):
        st.warning("Collinear set: " + (", ".join(f"{f} (VIF {v:.0f})" for f, v in high.items()) or f"condition number {col['cond']:.0f}")
                   + ". Raw betas will be unstable; orthogonal betas depend heavily on the order.")
    if E.is_default(spec) and spec.target in set(pd.read_parquet(E.MODEL_ROOT / "static_fit.parquet")["target"]):
        st.success("Default setup: numbers reproduce `factor_lens_local/models/` (see the *Consistency* tab).", icon="✅")

res = static_fit(spec)
if res is None:
    st.error("Not enough complete observations for this target / factor set / window. Try another target or wider dates.")
    st.stop()
S, coef = res["summary"], res["coef"].copy()
beta_col, t_col = ("gamma", "t_gamma") if orthogonal else ("raw_beta", "t_raw")

# ----------------------------------------------------------------------------------------------------------
# headline
# ----------------------------------------------------------------------------------------------------------
st.divider()
st.subheader(f"{target_label(target)}")
h = st.columns(6)
h[0].metric("R² (estimation)", fmt(S["r2_est"]))
h[1].metric("Adj. R²", fmt(S["adj_r2_est"]))
h[2].metric("Holdout R² (OOS)", fmt(S["r2_oos_hold"]), help="Holdout returns predicted with the estimation-window coefficients.")
h[3].metric("Holdout R² (refit)", fmt(S["r2_hold_refit"]))
h[4].metric("Residual vol (ann.)", fmt(S["resid_vol_ann"], 3, pct=True), f"total {S['total_vol_ann']:.1%}", delta_color="off")
h[5].metric("Alpha (ann.)", fmt(S["alpha_ann"] if orthogonal else S["alpha_raw_ann"], 3, pct=True))

do_roll = st.toggle("Compute rolling model (rolling betas, causal fitted values, attribution)", value=len(factors) <= 16,
                    help="One regression per window; slow for large factor sets, so off by default above 16 factors. Results are cached.")
ROLL_OFF = "Switch on *Compute rolling model* above the tabs."
if do_roll:
    with st.spinner("Rolling fit ..."):
        ro = rolling(spec)
else:
    ro = None
tabs = st.tabs(["Betas", "Ladder", "Rolling", "Actual vs fitted", "Attribution & risk", "Compare", "Collinearity", "Consistency"])

# ---- Betas -------------------------------------------------------------------------------------------------
with tabs[0]:
    coef["tier"] = coef["factor"].map(FT.set_index("name")["ai_tier"])
    coef["sig"] = np.where(coef[t_col].abs() > 2, "|t| > 2", "|t| ≤ 2")
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(coef, x="factor", y=beta_col, color="tier", color_discrete_map=TIER_COLOR, pattern_shape="sig",
                     pattern_shape_map={"|t| > 2": "", "|t| ≤ 2": "/"}, hover_data={t_col: ":.2f", "effect_1sd_bp": ":.1f"},
                     title=("Orthogonal betas (γ)" if orthogonal else "Raw OLS betas") + " — hatched = not significant (NW)")
        fig.update_layout(height=420, xaxis_title=None, yaxis_title="beta", legend_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("bp effect of a 1σ move of each orthogonal factor puts betas of factors in different units on one scale:")
        fig = px.bar(coef, x="factor", y="effect_1sd_bp", color="tier", color_discrete_map=TIER_COLOR)
        fig.update_layout(height=280, xaxis_title=None, yaxis_title="bp / day per 1σ", showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        show = coef[["factor", "tier", beta_col, t_col, "effect_1sd_bp", "var_share", "gamma_hold", "z_shift"]].rename(columns={
            beta_col: "beta", t_col: "t (NW)", "effect_1sd_bp": "1σ effect (bp)", "var_share": "variance share",
            "gamma_hold": "γ holdout", "z_shift": "shift z"})
        st.dataframe(show.style.format({"beta": "{:.4f}", "t (NW)": "{:.2f}", "1σ effect (bp)": "{:.1f}", "variance share": "{:.1%}",
                                        "γ holdout": "{:.4f}", "shift z": "{:.2f}"})
                     .background_gradient(subset=["t (NW)"], cmap="RdBu_r", vmin=-6, vmax=6), hide_index=True, width="stretch", height=480)
        st.caption("Variance shares and the holdout shift test are on the orthogonal basis. |shift z| > 2 flags a beta that changed "
                   "between estimation and holdout.")

# ---- Ladder ------------------------------------------------------------------------------------------------
with tabs[1]:
    lad = ladder(replace(spec, factors=tuple(full_order)))
    fig = go.Figure()
    fig.add_bar(x=lad["added"], y=lad["delta_r2"], name="ΔR² (estimation)", marker_color="#b9c6d6", yaxis="y2")
    fig.add_scatter(x=lad["added"], y=lad["r2_est"], name="R² estimation", mode="lines+markers")
    fig.add_scatter(x=lad["added"], y=lad["adj_r2_est"], name="adj. R²", mode="lines", line=dict(dash="dot"))
    fig.add_scatter(x=lad["added"], y=lad["r2_oos_hold"], name="holdout R² (OOS)", mode="lines+markers")
    fig.add_vline(x=k - 1, line_dash="dash", line_color="grey", annotation_text="slider")
    fig.update_layout(height=460, title=f"R² as factors are added ({order_mode.lower()})", xaxis_title="factor added",
                      yaxis=dict(title="R²"), yaxis2=dict(title="ΔR²", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(lad.style.format({c: "{:.4f}" for c in ["r2_est", "adj_r2_est", "r2_oos_hold", "delta_r2"]}), hide_index=True)

# ---- Rolling -----------------------------------------------------------------------------------------------
with tabs[2]:
    if ro is None:
        st.info(ROLL_OFF)
    elif ro["betas"].empty:
        st.info("Not enough data for a single rolling window.")
    else:
        pick = st.multiselect("Factors to plot", factors, default=[f for f in factors if f != "MKT"][:6] or factors)
        b = ro["betas"][pick]
        fig = px.line(b, title=f"Rolling {spec.window}-period {'orthogonal' if orthogonal else 'raw'} betas")
        fig.add_vrect(x0=spec.hold[0], x1=spec.hold[1], fillcolor="orange", opacity=0.1, line_width=0, annotation_text="holdout")
        fig.add_hline(y=0, line_color="grey", line_width=0.5)
        fig.update_layout(height=420, yaxis_title="beta", xaxis_title=None, legend_title=None)
        st.plotly_chart(fig, width="stretch")
        if "MKT" in factors and "MKT" not in pick:
            st.caption(f"MKT beta now: {ro['betas']['MKT'].iloc[-1]:.2f}")
        fig = px.line(ro["r2"], title="Rolling in-window R²")
        fig.add_vrect(x0=spec.hold[0], x1=spec.hold[1], fillcolor="orange", opacity=0.1, line_width=0)
        fig.update_layout(height=300, showlegend=False, yaxis_title="R²", xaxis_title=None, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

# ---- Actual vs fitted --------------------------------------------------------------------------------------
with tabs[3]:
    mode = st.radio("Fitted values from", ["Static model (estimation coefficients)"]
                    + (["Rolling model (causal, window ending the day before)"] if ro is not None else []), horizontal=True)
    if mode.startswith("Static"):
        p = res["path"]
    else:
        p = ro["causal"].loc[spec.est[0]:spec.hold[1]].copy()
        p["resid"] = p["actual"] - p["fitted"]
        p["sample"] = np.where(p.index >= pd.Timestamp(spec.hold[0]), "holdout", "estimation")
    cum = p[["actual", "fitted"]].cumsum()
    fig = px.line(cum, title="Cumulative excess return: actual vs factor-fitted (sum of log-approx. returns)")
    fig.add_vrect(x0=spec.hold[0], x1=spec.hold[1], fillcolor="orange", opacity=0.1, line_width=0, annotation_text="holdout")
    fig.update_layout(height=400, yaxis_tickformat=".0%", xaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(p, x="fitted", y="actual", color="sample", opacity=0.45, title="Actual vs fitted",
                         color_discrete_map={"estimation": "#4c78a8", "holdout": "#f58518"})
        lim = float(np.nanpercentile(np.abs(p[["fitted", "actual"]].to_numpy()), 99.5))
        fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line=dict(color="grey", dash="dot"))
        fig.update_layout(height=380, xaxis_tickformat=".1%", yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.scatter(p.reset_index(), x=p.index.name or "index", y="resid", color="sample", opacity=0.45, title="Residuals over time",
                         color_discrete_map={"estimation": "#4c78a8", "holdout": "#f58518"})
        fig.update_layout(height=380, yaxis_tickformat=".1%", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

# ---- Attribution & risk ------------------------------------------------------------------------------------
with tabs[4]:
    am = ro["monthly"] if ro is not None else pd.DataFrame()
    if ro is None:
        st.info(ROLL_OFF)
    elif am.empty:
        st.info("No attribution available (needs a full rolling window).")
    else:
        months = am["month"].tolist()
        n_m = st.slider("Months shown", 3, min(len(months), 60), min(24, len(months)))
        sub = am.tail(n_m).set_index("month")
        comps = ["RF", "ALPHA"] + factors + ["RESIDUAL"]
        pal = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        fig = go.Figure()
        for i, c in enumerate(comps):
            color = "#cfcfcf" if c == "RESIDUAL" else ("#333333" if c == "RF" else pal[i % len(pal)])
            fig.add_bar(x=sub.index, y=sub[c], name=c, marker_color=color)
        fig.add_scatter(x=sub.index, y=sub["total_return"], mode="markers", name="total return",
                        marker=dict(symbol="diamond", size=9, color="red"))
        fig.update_layout(barmode="relative", height=480, yaxis_tickformat=".0%",
                          title=f"Monthly return attribution ({'orthogonal' if orthogonal else 'raw'} betas, causal rolling, Carino-linked)",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        m_pick = st.selectbox("Month detail", months[::-1])
        row = am.set_index("month").loc[m_pick]
        det = pd.DataFrame({"component": comps, "contribution": [row[c] for c in comps]})
        c1, c2 = st.columns([2, 3])
        c1.metric(f"Total return {m_pick}", f"{row['total_return']:.2%}", f"specific {row['SPECIFIC']:+.2%}", delta_color="off")
        c1.dataframe(det.style.format({"contribution": "{:+.2%}"}), hide_index=True, height=38 * (len(det) + 1))
        fig = px.bar(det, x="contribution", y="component", orientation="h", color=det["contribution"] > 0,
                     color_discrete_map={True: "#4c9a6a", False: "#c4524a"})
        fig.update_layout(height=38 * len(det) + 80, showlegend=False, xaxis_tickformat=".1%", yaxis_title=None,
                          yaxis=dict(autorange="reversed"), margin=dict(t=10))
        c2.plotly_chart(fig, width="stretch")

        st.markdown("#### Risk shares (variance decomposition of the rolling window, orthogonal basis)")
        rk = ro["risk"].set_index("month")[factors + ["RESIDUAL"]]
        fig = go.Figure()
        for i, c in enumerate(rk.columns):
            fig.add_scatter(x=rk.index, y=rk[c], name=c, stackgroup="one", mode="none",
                            fillcolor="#dddddd" if c == "RESIDUAL" else None)
        fig.update_layout(height=380, yaxis_tickformat=".0%", yaxis_range=[0, 1], legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

# ---- Compare -----------------------------------------------------------------------------------------------
with tabs[5]:
    st.caption("Side by side: the setup above (A) against a second target and/or factor set (B). Other settings are shared.")
    c1, c2 = st.columns(2)
    with c1:
        what = st.radio("Compare", ["Two factor sets, same target", "Two targets, same factor set", "Both differ"], key="cmp_mode")
    with c2:
        tB = target if what == "Two factor sets, same target" else target_picker("B", default="non_AI_control_test")
        if what == "Two targets, same factor set":
            fB = factors
        else:
            pB = st.selectbox("Factor set B", list(PRESETS), index=list(PRESETS).index(E.lm.BASELINE_MODEL), key="cmp_preset")
            fB = PRESETS[pB]
            if neutral:
                fB = [f for f in fB if f in set(FT.loc[FT["available_neutral"], "name"])]
    specB = replace(spec, target=tB, factors=tuple(fB))
    resB = static_fit(specB)
    if resB is None:
        st.error("B cannot be fitted with these settings.")
    else:
        SB = resB["summary"]
        keys = [("R² (estimation)", "r2_est"), ("Adj. R²", "adj_r2_est"), ("Holdout R² (OOS)", "r2_oos_hold"),
                ("Holdout R² (refit)", "r2_hold_refit"), ("Residual vol", "resid_vol_ann"), ("Alpha (ann.)", "alpha_ann")]
        cmp = pd.DataFrame({"metric": [k for k, _ in keys], "A": [S[v] for _, v in keys], "B": [SB[v] for _, v in keys]})
        cmp["B − A"] = cmp["B"] - cmp["A"]
        st.markdown(f"**A** = {target} · {len(factors)} factors  |  **B** = {tB} · {len(fB)} factors ({', '.join(fB)})")
        st.dataframe(cmp.style.format({"A": "{:.4f}", "B": "{:.4f}", "B − A": "{:+.4f}"}), hide_index=True)
        cb = pd.concat([res["coef"].assign(setup="A"), resB["coef"].assign(setup="B")])
        fig = px.bar(cb, x="factor", y="effect_1sd_bp", color="setup", barmode="group",
                     title="1σ effects (bp/day) — comparable across factor sets and targets",
                     category_orders={"factor": list(dict.fromkeys(list(factors) + list(fB)))})
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
        ca, cb_ = st.columns(2)
        for colx, rr, nm in ((ca, res, "A"), (cb_, resB, "B")):
            cum = rr["path"][["actual", "fitted"]].cumsum()
            fig = px.line(cum, title=f"{nm}: cumulative actual vs fitted (static)")
            fig.add_vrect(x0=spec.hold[0], x1=spec.hold[1], fillcolor="orange", opacity=0.1, line_width=0)
            fig.update_layout(height=320, yaxis_tickformat=".0%", xaxis_title=None, legend_title=None)
            colx.plotly_chart(fig, width="stretch")
        la, lb = ladder(spec), ladder(specB)
        fig = go.Figure()
        fig.add_scatter(x=la["k"], y=la["r2_est"], name="A est.", text=la["added"], mode="lines+markers")
        fig.add_scatter(x=la["k"], y=la["r2_oos_hold"], name="A holdout", text=la["added"], mode="lines", line=dict(dash="dot"))
        fig.add_scatter(x=lb["k"], y=lb["r2_est"], name="B est.", text=lb["added"], mode="lines+markers")
        fig.add_scatter(x=lb["k"], y=lb["r2_oos_hold"], name="B holdout", text=lb["added"], mode="lines", line=dict(dash="dot"))
        fig.update_layout(height=340, title="Ladders", xaxis_title="number of factors", yaxis_title="R²")
        st.plotly_chart(fig, width="stretch")

# ---- Collinearity ------------------------------------------------------------------------------------------
with tabs[6]:
    c1, c2 = st.columns([3, 2])
    corr = col["corr"]
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"{'Market-neutral' if neutral else 'Raw'} factor correlation, estimation window")
    fig.update_layout(height=max(380, 34 * len(corr) + 120))
    c1.plotly_chart(fig, width="stretch")
    c2.dataframe(col["vif"].style.format({"VIF": "{:.2f}"}).background_gradient(subset=["VIF"], cmap="Reds", vmin=1, vmax=15),
                 hide_index=True, width="stretch")
    c2.caption(f"VIF > {COLLINEAR_VIF:.0f} or condition number > {COLLINEAR_COND:.0f} triggers the warning.")

# ---- Consistency -------------------------------------------------------------------------------------------
with tabs[7]:
    st.markdown("For the default setup (12-factor `M3_plus_AI_close`, raw factors, daily, 2017–2023 / 2024–2026-10) the app must "
                "reproduce the static results of notebook 04 stored in `factor_lens_local/models/`.")
    if st.button("Run the check for all cached targets"):
        chk = reproduction_check()
        if chk["ok"].all():
            st.success(f"All {len(chk)} targets match (max relative difference {chk['max_abs_diff'].max():.1e}).")
        else:
            st.error(f"{(~chk['ok']).sum()} of {len(chk)} targets differ.")
        st.dataframe(chk, hide_index=True)
    st.caption("Same check from the command line: `python app/check_defaults.py`.")
