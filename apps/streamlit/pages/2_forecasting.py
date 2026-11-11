import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import state, ui

ss = st.session_state
s = state.settings()
ui.page_head("Workstream 1", "Demand forecasting",
             "Seven models are compared on a strict time split. Each forecast uses only data up to its origin "
             "week. The best model is picked automatically on the validation weeks, then scored once on the "
             "untouched test weeks.")

out = state.pipeline()
sp = out["splits"]
st.markdown(f"{ui.badge('observed')} weekly units sold &nbsp; {ui.badge('derived')} avg price, price-cut flag "
            f"&nbsp; {ui.badge('external')} UK bank holidays<br>"
            f"<span class='stamp'>Train ≤ {sp['train_end']:%d %b %Y} · validation {sp['val_weeks'][0]:%d %b}–"
            f"{sp['val_weeks'][1]:%d %b %Y} · test {sp['test_weeks'][0]:%d %b}–{sp['test_weeks'][1]:%d %b %Y} · "
            f"last model run {ss.last_model_run}</span>", unsafe_allow_html=True)

params = state.sku_params()
labels = {r.sku: f"{r.sku} · {r.description}" for r in params.itertuples()}
c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
sku = c1.selectbox("SKU", list(labels), format_func=labels.get, key="fc_sku")
agg = c2.radio("Aggregation", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="fc_agg")
countries = sorted(ss.tx["Country"].unique())
loc = c3.selectbox("Location (daily view)", ["All countries"] + countries, key="fc_loc")
if c4.button("Refresh forecast", width="stretch", type="primary", key="fc_refresh"):
    ss.cache = {k: v for k, v in ss.cache.items() if k[0] != "pipe"}
    st.rerun()

# ------------------------------------------------ history at chosen aggregation
p = ss.panel[ss.panel["sku"] == sku].set_index("week")
fc = out["forecast"][out["forecast"]["sku"] == sku]
fig = go.Figure()
if agg == "Daily":
    tx = ss.tx[(ss.tx["StockCode"] == sku) & (ss.tx["InvoiceDate"] < ss.panel["week"].max() + pd.Timedelta(days=7))]
    if loc != "All countries":
        tx = tx[tx["Country"] == loc]
    daily = tx.set_index("InvoiceDate")["Quantity"].resample("D").sum()
    fig.add_bar(x=daily.index, y=daily.values, name="Units per day", marker_color=ui.SKY)
    fig.add_scatter(x=daily.index, y=daily.rolling(28, min_periods=7).mean(), name="28-day average",
                    line=dict(color=ui.AMBER, width=2))
    fig.update_layout(title=f"Daily demand: {labels[sku]}", yaxis_title="Units per day")
    st.caption("The daily view comes from the raw transactions (shown only for the public dataset). Models run on weekly totals.")
elif agg == "Weekly":
    fig.add_scatter(x=p.index, y=p["demand"], name="Actual", line=dict(color=ui.SKY, width=2))
    fig.add_scatter(x=p.index, y=p["demand"].rolling(13, min_periods=4).mean(), name="13-week trend",
                    line=dict(color=ui.AMBER, width=2, dash="dash"))
    fig.add_scatter(x=list(fc["week"]) + list(fc["week"][::-1]), y=list(fc["hi"]) + list(fc["lo"][::-1]),
                    fill="toself", mode="lines", fillcolor="rgba(20,184,166,0.16)", line=dict(width=0), name="90% prediction interval")
    fig.add_scatter(x=fc["week"], y=fc["forecast"], name=f"Forecast ({fc['model'].iat[0]})",
                    line=dict(color=ui.TEAL, width=3), mode="lines+markers")
    promo = p[p["price_cut"] == 1]
    fig.add_scatter(x=promo.index, y=promo["demand"], mode="markers", name="Price-cut week (derived)",
                    marker=dict(color=ui.AMBER, size=8, symbol="diamond"))
    fig.update_layout(title=f"Weekly demand and {s['horizon']}-week forecast: {labels[sku]}", yaxis_title="Units per week")
else:
    m = p["demand"].resample("MS").sum()
    fm = fc.set_index("week")["forecast"].resample("MS").sum()
    fig.add_bar(x=m.index, y=m.values, name="Actual (month)", marker_color=ui.BLUE)
    fig.add_bar(x=fm.index, y=fm.values, name="Forecast (month, partial)", marker_color=ui.TEAL, opacity=0.8)
    fig.update_layout(title=f"Monthly demand: {labels[sku]}", yaxis_title="Units per month", barmode="group")
fig.update_layout(height=420, xaxis_title=None)
st.plotly_chart(fig, width="stretch")

t1, t2, t3, t4 = st.tabs(["Model comparison", "Actual vs predicted (test)", "Seasonality & drivers", "Forecast table"])
with t1:
    vm = out["val_metrics"].set_index("model")
    tm = out["test_metrics"].set_index("model")
    comp = vm[["MAE", "RMSE", "WAPE", "MAPE"]].add_prefix("Val ").join(tm[["MAE", "RMSE", "WAPE", "MAPE", "Bias"]].add_prefix("Test "), how="outer")
    comp = comp.sort_values("Val WAPE", na_position="last")
    best_global = out["best"]["best_model"].mode().iat[0]
    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.dataframe(comp.style.format({c: "{:.1%}" for c in comp.columns if "WAPE" in c or "MAPE" in c or "Bias" in c}
                                       | {c: "{:,.0f}" for c in comp.columns if "MAE" in c or "RMSE" in c}, na_rep="–")
                     .highlight_min(subset=["Val WAPE", "Test WAPE"], color="#134e4a"),
                     width="stretch", height=360)
        st.caption("WAPE = Σ|error| / Σ actual, pooled over all SKUs. MAPE skips weeks with zero demand, where it "
                   "is undefined, so it is shown for reference only. Bias > 0 means over-forecasting.")
    with c2:
        d = comp.reset_index().dropna(subset=["Val WAPE"])
        fig = go.Figure()
        fig.add_bar(y=d["model"], x=d["Val WAPE"], name="Validation", orientation="h", marker_color=ui.BLUE)
        fig.add_bar(y=d["model"], x=d["Test WAPE"], name="Test", orientation="h", marker_color=ui.TEAL)
        fig.update_layout(barmode="group", height=360, title="WAPE by model (lower is better)",
                          xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    st.markdown(f"**Selected:** {'one model for all SKUs, ' + best_global if s['strategy'] == 'global' else 'best model per SKU'}"
                f" (lowest validation WAPE). LSTM is not included: each SKU has only about 100 weekly points, too "
                "few to train a recurrent network reliably. The pooled gradient-boosting model is the ML option that "
                "fits this much data.")
with t2:
    tb = out["test_bt"]
    models = st.multiselect("Models to show", list(ui.MODEL_COLORS)[:7],
                            default=["Gradient boosting", "ETS (damped trend)", "Naive"], key="fc_models")
    d = tb[(tb["sku"] == sku) & (tb["model"].isin(models))]
    fig = go.Figure()
    act = tb[(tb["sku"] == sku)].drop_duplicates("week").sort_values("week")
    fig.add_scatter(x=act["week"], y=act["actual"], name="Actual", line=dict(color="#f8fafc", width=3))
    for m in models:
        dm = d[d["model"] == m].sort_values("week")
        fig.add_scatter(x=dm["week"], y=dm["forecast"], name=m, mode="lines+markers",
                        line=dict(color=ui.MODEL_COLORS[m], width=2))
    fig.update_layout(height=380, title="Test weeks: forecasts from rolling 4-week origins vs actual", yaxis_title="Units")
    st.plotly_chart(fig, width="stretch")
    sk = out["sku_test_metrics"]
    sk = sk[sk["sku"] == sku].set_index("model")[["MAE", "RMSE", "WAPE"]].sort_values("WAPE")
    st.dataframe(sk.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "WAPE": "{:.1%}"}), width="stretch")
with t3:
    c1, c2 = st.columns(2)
    with c1:
        pp = p.copy()
        pp["year"] = pp.index.year.astype(str)
        pp["woy"] = pp.index.isocalendar().week.values
        fig = go.Figure()
        for (yr, g), col in zip(pp.groupby("year"), [ui.SLATE, ui.BLUE, ui.TEAL]):
            fig.add_scatter(x=g["woy"], y=g["demand"], name=yr, mode="lines", line=dict(color=col, width=2))
        fig.update_layout(height=340, title="Year-over-year overlay (week of year)", xaxis_title="ISO week", yaxis_title="Units")
        st.plotly_chart(fig, width="stretch")
    with c2:
        from sklearn.inspection import permutation_importance
        key = ("imp", state.panel_key(ss.panel))
        if key not in ss.cache:
            from inventory_core import forecasting as F
            W = F.wide_demand(ss.panel)
            P = ss.panel.pivot(index="week", columns="sku", values="avg_price").sort_index()
            C = ss.panel.pivot(index="week", columns="sku", values="price_cut").sort_index()
            o = len(W) - 1
            tr = F._ml_train_set(ss.panel, o, W.index, W, P, C)
            fit = F._fit_ml("Gradient boosting", tr)
            sample = tr.sample(min(3000, len(tr)), random_state=42)
            pi = permutation_importance(fit, sample[F.ML_FEATURES], sample["y_ratio"], n_repeats=5, random_state=42)
            ss.cache[key] = pd.Series(pi.importances_mean, index=F.ML_FEATURES).sort_values()
        imp = ss.cache[key]
        names = {"lvl4_ratio": "Recent level (4 wk)", "trend": "Short-term trend", "lag52_ratio": "Same week last year",
                 "price_ratio": "Price vs median", "price_cut": "Price-cut flag", "hol": "UK bank holidays", "h": "Horizon",
                 "sin1": "Annual cycle (sin1)", "cos1": "Annual cycle (cos1)", "sin2": "Half-year (sin2)",
                 "cos2": "Half-year (cos2)", "sin3": "4-month (sin3)", "cos3": "4-month (cos3)"}
        fig = go.Figure(go.Bar(x=imp.values, y=[names.get(i, i) for i in imp.index], orientation="h",
                               marker_color=[ui.TEAL if v > 0 else ui.SLATE for v in imp.values]))
        fig.update_layout(height=340, title="What drives the ML forecast (permutation importance)",
                          xaxis_title="Increase in error when the feature is shuffled")
        st.plotly_chart(fig, width="stretch")
    st.caption("Promotions are not recorded in the data. The price-cut flag (weekly average price more than 10% "
               "below the SKU's median price) is a derived proxy and may also reflect bulk-price tiers.")
with t4:
    show = fc[["week", "h", "model", "forecast", "lo", "hi"]].rename(columns={"lo": "90% low", "hi": "90% high"})
    st.dataframe(show.style.format({"forecast": "{:,.0f}", "90% low": "{:,.0f}", "90% high": "{:,.0f}",
                                    "week": lambda x: f"{x:%d %b %Y}"}), hide_index=True, width="stretch")
    st.download_button("Download all SKU forecasts (CSV)", out["forecast"].to_csv(index=False).encode(),
                       "forecasts.csv", "text/csv", key="fc_dl")
