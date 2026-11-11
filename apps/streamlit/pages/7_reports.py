import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inventory_core import data as D, inventory as I, simulation as S
from utils import state, ui

ss = st.session_state
s = state.settings()
ui.page_head("Data & reports", "Reports & model diagnostics",
             "Accuracy, buffer comparison, simulation and optimisation results, and data quality, each with a "
             "CSV download.")
out = state.pipeline()
tab = state.policy_table()

t1, t2, t3, t4 = st.tabs(["Forecast accuracy", "Safety stock: formula vs Monte Carlo", "Optimisation summary", "Data quality"])
with t1:
    tm = out["test_metrics"].sort_values("WAPE")
    st.dataframe(tm.style.format({"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "WAPE": "{:.1%}", "MAPE": "{:.1%}", "Bias": "{:+.1%}",
                                  "n": "{:.0f}"}), hide_index=True, width="stretch")
    sk = out["sku_test_metrics"]
    sel = sk[sk["model"].str.startswith("AI-selected")].groupby("sku")["WAPE"].first().sort_values()
    fig = go.Figure(go.Bar(x=sel.index, y=sel.values, marker_color=[ui.TEAL if v < 0.5 else ui.AMBER for v in sel.values]))
    fig.update_layout(height=320, title="Test WAPE per SKU, AI-selected model", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button("Model metrics (CSV)", tm.to_csv(index=False).encode(), "forecast_accuracy.csv", key="rp1")
    c2.download_button("Back-test predictions (CSV)", pd.concat([out["val_bt"].assign(split="validation"),
                       out["test_bt"].assign(split="test")]).to_csv(index=False).encode(), "backtest_predictions.csv", key="rp2")
with t2:
    rows = []
    for r in tab.itertuples():
        sim = S.lead_time_demand(10000, max(r.d_bar, 1e-6), max(r.sigma_d, 1e-6), s["lead_mean"], s["lead_sd"],
                                 s["demand_dist"], s["lead_dist"], s["rho"], seed=42)
        m = S.summarise_dlt(sim["dlt"], s["service_level"], rop=r.rop, d_mean=r.d_bar, l_mean=s["lead_mean"])
        rows.append({"sku": r.sku, "description": r.description, "d_bar": r.d_bar, "sigma_d": r.sigma_d,
                     "SS formula": r.safety_stock, "SS Monte Carlo": m["ss_mc"], "difference %": m["ss_mc"] / max(r.safety_stock, 1e-9) - 1,
                     "CSL of formula ROP (simulated)": m["cycle_service_level"]})
    comp = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=comp["SS formula"], y=comp["SS Monte Carlo"], mode="markers", text=comp["sku"],
                    marker=dict(size=11, color=comp["CSL of formula ROP (simulated)"], colorscale="RdYlGn", cmin=0.85, cmax=1,
                                showscale=True, colorbar=dict(title="CSL", tickformat=".0%")),
                    hovertemplate="%{text}<br>formula %{x:,.0f}<br>MC %{y:,.0f}<extra></extra>")
    mx = max(comp["SS formula"].max(), comp["SS Monte Carlo"].max()) * 1.05
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=ui.MUTED, dash="dot"), name="equal")
    fig.update_layout(height=380, showlegend=False, title="Monte Carlo vs formula safety stock (points above the line need more stock)",
                      xaxis_title="Formula SS (units)", yaxis_title="Monte Carlo SS (units)")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"For **{(comp['difference %'] > 0).sum()} of {len(comp)}** SKUs the simulation needs more buffer than the "
                f"formula (median difference {comp['difference %'].median():+.1%}). The formula ROP delivers a median simulated "
                f"service level of {comp['CSL of formula ROP (simulated)'].median():.1%} against the {s['service_level']:.1%} target.")
    st.dataframe(comp.style.format({"d_bar": "{:,.0f}", "sigma_d": "{:,.0f}", "SS formula": "{:,.0f}", "SS Monte Carlo": "{:,.0f}",
                                    "difference %": "{:+.1%}", "CSL of formula ROP (simulated)": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.download_button("Safety-stock comparison (CSV)", comp.to_csv(index=False).encode(), "safety_stock_comparison.csv", key="rp3")
with t3:
    if not ss.opt_results:
        st.info("No optimisation has been run yet. Use the Bayesian optimisation page or *Optimise all SKUs* on the Inventory page.")
    else:
        o = pd.DataFrame(ss.opt_results).T.rename_axis("sku").reset_index()
        o = o.merge(tab[["sku", "description", "rop", "eoq"]], on="sku")
        o["ROP change"] = o["rop_x"] / o["rop_y"] - 1
        o = o.rename(columns={"rop_x": "optimised ROP", "q": "optimised Q", "rop_y": "formula ROP", "eoq": "EOQ"})
        st.dataframe(o.style.format({"optimised ROP": "{:,.0f}", "optimised Q": "{:,.0f}", "formula ROP": "{:,.0f}", "EOQ": "{:,.0f}",
                                     "ROP change": "{:+.1%}"}), hide_index=True, width="stretch")
        st.download_button("Optimisation summary (CSV)", o.to_csv(index=False).encode(), "optimisation_summary.csv", key="rp4")
    if ss.get("mc_scen") is not None:
        st.markdown("##### Latest Monte Carlo scenario run")
        st.dataframe(ss.mc_scen, hide_index=True, width="stretch")
        st.download_button("Monte Carlo scenarios (CSV)", ss.mc_scen.to_csv(index=False).encode(), "monte_carlo_scenarios.csv", key="rp5")
with t4:
    q = ss.quality
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Cleaning funnel (raw → clean transactions)")
        steps = pd.DataFrame({"step": ["Raw rows", "Cancellations (C-invoices)", "Qty or price ≤ 0", "Non-product codes",
                                       "Exact duplicates removed", "Clean sales rows"],
                              "rows": [q["rows_raw"], -q["cancellation_rows"], -q["non_positive_qty_or_price_rows"],
                                       -q["non_product_code_rows"],
                                       -(q["rows_raw"] - q["cancellation_rows"] - q["non_positive_qty_or_price_rows"] - q["non_product_code_rows"] - q["rows_clean"]),
                                       q["rows_clean"]]})
        fig = go.Figure(go.Waterfall(x=steps["step"], y=steps["rows"], measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                                     decreasing=dict(marker_color=ui.RED), totals=dict(marker_color=ui.TEAL),
                                     increasing=dict(marker_color=ui.BLUE), text=[f"{abs(v):,}" for v in steps["rows"]]))
        fig.update_layout(height=360, yaxis_title="Rows")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("##### Missing values in the raw file")
        mv = pd.Series(q["missing_by_column"]).rename("missing").to_frame()
        mv["share"] = mv["missing"] / q["rows_raw"]
        st.dataframe(mv.style.format({"missing": "{:,}", "share": "{:.1%}"}), width="stretch")
        st.markdown(f"Clean data: **{q['rows_clean']:,}** sales rows, **{q['skus_clean']:,}** products, "
                    f"{q['date_min'][:10]} to {q['date_max'][:10]}. Missing Customer IDs do not affect demand.")
    p = ss.panel
    st.markdown(f"Weekly panel used by the models: **{p['sku'].nunique()} SKUs × {p['week'].nunique()} weeks** = "
                f"{len(p):,} rows; zero-demand weeks: {(p['demand'] == 0).mean():.1%}; derived price-cut weeks: "
                f"{p['price_cut'].mean():.1%}.")
    st.download_button("Weekly panel (CSV)", D.panel_to_csv_bytes(p), "weekly_panel.csv", key="rp6")
