import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import state, ui

ss = st.session_state
s = state.settings()
ui.page_head("Command centre", "Executive overview",
             "Live summary of demand, buffers and replenishment risk for the tracked SKUs of a UK online "
             "gift wholesaler (UCI Online Retail II). Every figure below is computed on this page from the "
             "current data and scenario settings.")

out = state.pipeline()
tab = state.policy_table()
fc = out["forecast"]
H = s["horizon"]
alerts = tab["status"].isin(["Reorder Now", "Critical Stockout Risk"]).sum()
crit = (tab["status"] == "Critical Stockout Risk").sum()

ui.kpis([
    ("SKUs analysed", f"{len(tab)}", "top sellers, active ≥90% of weeks", ui.TEAL),
    (f"Forecast demand · {H} wks", ui.fmt(fc["forecast"].sum()), f"units, model: {tab['best_model'].mode().iat[0]}", ui.BLUE),
    ("Avg weekly demand", ui.fmt(tab["d_bar"].mean()), "units per SKU (forecast mean)", ui.SKY),
    ("Avg safety stock", ui.fmt(tab["safety_stock"].mean()), f"units at {s['service_level']:.1%} CSL", ui.TEAL),
    ("Inventory on hand", ui.fmt(tab["on_hand"].sum()), f"units · {ss.stock_source.lower()}", ui.SLATE),
    ("Reorder alerts", f"{alerts}", f"{crit} critical", ui.AMBER if alerts else ui.TEAL),
    ("Expected service level", f"{tab['csl_at_rop'].mean():.1%}", "simulated CSL of formula ROPs", ui.TEAL),
    ("Est. stockout risk", f"{tab['stockout_risk'].mean():.1%}", "avg P(lead-time demand > position)", ui.RED),
])
if ss.stock_source == "Simulated scenario":
    ui.note(f"{ui.badge('simulated')} Stock on hand is not recorded in the dataset, so the inventory figures use a "
            "simulated starting position (2–10 weeks of forecast cover, fixed seed). Upload real stock on the "
            "<b>Data management</b> page to replace it.", warn=True)

c1, c2 = st.columns([2.1, 1])
with c1:
    hist = ss.panel.groupby("week")["demand"].sum()
    agg = fc.groupby("week")[["forecast", "lo", "hi"]].sum()
    # interval of a sum: combine per-SKU sigmas in quadrature (independence assumption)
    sig = fc.groupby("week")["sigma"].apply(lambda x: np.sqrt((x ** 2).sum()))
    fig = go.Figure()
    h2 = hist[hist.index > hist.index.max() - pd.Timedelta(weeks=78)]
    fig.add_scatter(x=h2.index, y=h2.values, name="Actual demand", mode="lines", line=dict(color=ui.SKY, width=2))
    fig.add_scatter(x=list(agg.index) + list(agg.index[::-1]),
                    y=list(agg["forecast"] + 1.645 * sig) + list(np.clip(agg["forecast"] - 1.645 * sig, 0, None))[::-1],
                    fill="toself", mode="lines", fillcolor="rgba(20,184,166,0.15)", line=dict(width=0), name="90% interval",
                    hoverinfo="skip")
    fig.add_scatter(x=agg.index, y=agg["forecast"], name="AI forecast", mode="lines+markers",
                    line=dict(color=ui.TEAL, width=3), marker=dict(size=6))
    fig.update_layout(title="Total weekly demand: last 18 months and forecast", height=380,
                      yaxis_title="Units per week", xaxis_title=None)
    st.plotly_chart(fig, width="stretch")
with c2:
    cnt = tab["status"].value_counts().reindex(["Healthy", "Monitor", "Reorder Now", "Critical Stockout Risk"], fill_value=0)
    fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=0.62, sort=False,
                           marker=dict(colors=[ui.TEAL, ui.AMBER, ui.ORANGE, ui.RED], line=dict(color="#0b1120", width=3)),
                           text=[str(v) if v else "" for v in cnt.values], textinfo="text", textfont=dict(size=14)))
    fig.update_layout(title="Inventory health", height=380, showlegend=True,
                      legend=dict(orientation="v", y=0.5, x=1.0, yanchor="middle"),
                      annotations=[dict(text=f"<b>{len(tab)}</b><br>SKUs", showarrow=False, font=dict(size=16))])
    st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns([1.5, 1])
with c1:
    st.markdown("#### Critical and reorder SKUs")
    crit_t = tab[tab["status"].isin(["Critical Stockout Risk", "Reorder Now"])].sort_values("stockout_risk", ascending=False)
    if crit_t.empty:
        st.success("No SKU is at or below its reorder point under the current scenario.")
    else:
        st.dataframe(crit_t[["sku", "description", "status", "inventory_position", "rop", "safety_stock",
                             "stockout_risk", "order_qty"]],
                     hide_index=True, width="stretch", height=min(420, 38 + 35 * len(crit_t)),
                     column_config={
                         "sku": "SKU", "description": "Product", "status": "Status",
                         "inventory_position": st.column_config.NumberColumn("Position", format="%.0f"),
                         "rop": st.column_config.NumberColumn("ROP", format="%.0f"),
                         "safety_stock": st.column_config.NumberColumn("Safety stock", format="%.0f"),
                         "stockout_risk": st.column_config.ProgressColumn("Stockout risk", min_value=0, max_value=1, format="percent"),
                         "order_qty": st.column_config.NumberColumn("Order now", format="%.0f")})
with c2:
    st.markdown("#### AI recommendations")
    top = tab.sort_values("stockout_risk", ascending=False).head(3)
    recs = []
    for _, r in top.iterrows():
        if r["order_qty"] > 0:
            recs.append(f"<b>{r['description']}</b> ({r['sku']}): {r['action'].lower()}. Order "
                        f"<b>{r['order_qty']:,.0f}</b> units; stockout risk {r['stockout_risk']:.0%}.")
    over = tab[tab["weeks_of_cover"] > 5]
    if len(over):
        recs.append(f"<b>{len(over)}</b> SKU(s) hold more than 5 weeks of cover, which ties up "
                    f"£{(over['on_hand'] * over['unit_cost']).sum():,.0f} of stock. Pause orders for them.")
    gap = (tab["csl_at_rop"] < s["service_level"]).sum()
    if gap:
        recs.append(f"For <b>{gap}</b> SKU(s), the formula ROP delivers a simulated service level below the "
                    f"{s['service_level']:.0%} target, because demand is skewed. Use the Monte Carlo or optimised "
                    "ROP for these.")
    recs.append(f"Forecasts use <b>{tab['best_model'].mode().iat[0]}</b>, selected automatically on the "
                f"validation window. Test WAPE: {out['test_metrics'].set_index('model')['WAPE'].filter(like='AI-selected').iloc[0]:.1%}.")
    for r in recs:
        st.markdown(f'<div class="rec">{r}</div>', unsafe_allow_html=True)

st.markdown("#### Seasonality: average demand index by month")
p = ss.panel.copy()
p["month"] = p["week"].dt.month
idx = p.groupby(["sku", "month"])["demand"].mean() / p.groupby("sku")["demand"].mean()
prof = idx.groupby("month").mean()
fig = go.Figure(go.Bar(x=pd.to_datetime(prof.index, format="%m").strftime("%b"), y=prof.values,
                       marker=dict(color=[ui.AMBER if v > 1.2 else ui.TEAL for v in prof.values]),
                       text=[f"{v:.2f}" for v in prof.values], textposition="outside"))
fig.add_hline(y=1, line_dash="dot", line_color=ui.MUTED)
fig.update_layout(height=300, yaxis_title="Index (1 = average month)", title=None, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
top3 = prof.sort_values(ascending=False).head(3)
st.caption("Highest months: " + ", ".join(f"{pd.Timestamp(month=m, year=2000, day=1):%b} ({v:.2f}×)" for m, v in top3.items())
           + ". Months above 1.2× average are shown in amber.")
