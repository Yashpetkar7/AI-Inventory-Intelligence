import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inventory_core import inventory as I, optimization as O
from utils import state, ui

ss = st.session_state
s = state.settings()
ui.page_head("Command centre", "Inventory intelligence",
             "Every SKU's position against its safety stock and reorder point, with a recommended action. "
             "Optimise the whole range in one click, and explore it in a 3D view of the warehouse.")

tab = state.policy_table()
c1, c2, c3 = st.columns([2, 1, 1])
flt = c1.multiselect("Status filter", I.STATUS_ORDER, default=I.STATUS_ORDER, key="inv_flt")
if c2.button("Optimise all SKUs (Bayesian)", width="stretch", key="inv_optall",
             help="Runs 60 TPE trials per SKU with the sidebar's service target and no capacity or budget limit."):
    prog = st.progress(0.0, text="Optimising…")
    for i, r in enumerate(tab.itertuples()):
        sc = O.Scenario(d_mean=max(r.d_bar, 1e-6), d_sd=max(r.sigma_d, 1e-6), l_mean=s["lead_mean"], l_sd=s["lead_sd"],
                        unit_cost=r.unit_cost, holding_rate=s["holding_rate"], order_cost=s["order_cost"],
                        shortage_cost=r.price * s["shortage_mult"], target_csl=s["service_level"], moq=s["moq"],
                        demand_dist=s["demand_dist"], lead_dist=s["lead_dist"], n_paths=120)
        res = O.optimise(sc, n_trials=60)
        ss.opt_results[r.sku] = {"rop": res["best"]["rop"], "q": res["best"]["q"]}
        prog.progress((i + 1) / len(tab), text=f"{r.sku} done ({i + 1}/{len(tab)})")
    prog.empty()
    st.rerun()
src = ss.stock_source
c3.markdown(f"<div style='padding-top:30px'>{ui.badge('simulated' if src.startswith('Sim') else 'observed')} stock: {src.lower()}</div>",
            unsafe_allow_html=True)

view = tab[tab["status"].isin(flt)].copy()
view["status_rank"] = view["status"].map({k: i for i, k in enumerate(I.STATUS_ORDER)})
view = view.sort_values(["status_rank", "stockout_risk"], ascending=[True, False])
icons = {"Critical Stockout Risk": "🔴", "Reorder Now": "🟠", "Monitor": "🟡", "Healthy": "🟢"}
view["Status"] = view["status"].map(lambda x: f"{icons[x]} {x}")
cols = ["sku", "description", "Status", "d_bar", "on_hand", "on_order", "safety_stock", "rop", "opt_rop", "order_qty",
        "weeks_of_cover", "stockout_risk", "action"]
st.dataframe(view[cols], hide_index=True, width="stretch", height=440, column_config={
    "sku": "SKU", "description": "Product",
    "d_bar": st.column_config.NumberColumn("Forecast/wk", format="%.0f"),
    "on_hand": st.column_config.NumberColumn("On hand", format="%.0f"),
    "on_order": st.column_config.NumberColumn("On order", format="%.0f"),
    "safety_stock": st.column_config.NumberColumn("Safety stock", format="%.0f"),
    "rop": st.column_config.NumberColumn("ROP (formula)", format="%.0f"),
    "opt_rop": st.column_config.NumberColumn("ROP (optimised)", format="%.0f"),
    "order_qty": st.column_config.NumberColumn("Order qty", format="%.0f"),
    "weeks_of_cover": st.column_config.NumberColumn("Cover (wks)", format="%.1f"),
    "stockout_risk": st.column_config.ProgressColumn("Stockout risk", min_value=0, max_value=1, format="percent"),
    "action": "Recommended action"})
exp = view[["sku", "description", "status", "d_bar", "sigma_d", "on_hand", "on_order", "backorders", "inventory_position",
            "safety_stock", "rop", "opt_rop", "opt_q", "eoq", "order_qty", "stockout_risk", "action"]]
st.download_button("Export SKU table (CSV)", exp.to_csv(index=False).encode(), "inventory_intelligence.csv", "text/csv",
                   key="inv_dl")

st.markdown("#### 3D warehouse view")
ui.note(f"{ui.badge('simulated')} <b>Conceptual layout.</b> The dataset has no warehouse coordinates, so each SKU is "
        "placed on a rack in order of risk. The colour of each bin is its status and the height of the fill is its "
        "stock as a share of 2 × ROP. Drag to rotate, scroll to zoom, double-click to reset the camera.")
labels = {r.sku: f"{r.sku} · {r.description}" for r in tab.itertuples()}
sel = st.selectbox("Highlight SKU", list(labels), format_func=labels.get, key="inv_sel")

t3 = tab.copy()
t3["rank"] = t3["status"].map({k: i for i, k in enumerate(I.STATUS_ORDER)})
t3 = t3.sort_values(["rank", "stockout_risk"], ascending=[True, False]).reset_index(drop=True)
racks, levels, per = 4, 3, 2
fig = go.Figure()


def box(x0, y0, z0, dx, dy, dz, color, opacity, name, hover):
    x = [x0, x0 + dx, x0 + dx, x0, x0, x0 + dx, x0 + dx, x0]
    y = [y0, y0, y0 + dy, y0 + dy, y0, y0, y0 + dy, y0 + dy]
    z = [z0, z0, z0, z0, z0 + dz, z0 + dz, z0 + dz, z0 + dz]
    fig.add_mesh3d(x=x, y=y, z=z, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                   k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], color=color, opacity=opacity, flatshading=True,
                   name=name, hovertext=hover, hoverinfo="text", showscale=False,
                   lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25), lightposition=dict(x=100, y=200, z=300))


for i, r in t3.iterrows():
    rack, slot = divmod(i, levels * per)
    lvl, col = divmod(slot, per)
    x0, y0, z0 = col * 1.3 + rack * 3.6, 0, lvl * 1.25
    fill = float(np.clip(r["inventory_position"] / max(2 * r["rop"], 1), 0.04, 1.0))
    color = I.STATUS_COLORS[r["status"]]
    hover = (f"<b>{r['sku']} · {r['description']}</b><br>Status: {r['status']}<br>On hand: {r['on_hand']:,.0f}"
             f"<br>Safety stock: {r['safety_stock']:,.0f}<br>ROP: {r['rop']:,.0f}<br>Stockout risk: {r['stockout_risk']:.0%}")
    hl = r["sku"] == sel
    box(x0, y0, z0, 1.1, 1.0, 1.0, "#94a3b8" if not hl else "#f8fafc", 0.10 if not hl else 0.35, r["sku"], hover)
    box(x0 + 0.05, y0 + 0.05, z0, 1.0, 0.9, 0.95 * fill, color, 0.95 if (hl or sel is None) else 0.75, r["sku"], hover)
for rack in range(racks):                      # rack frames
    x0 = rack * 3.6 - 0.1
    for zz in [0, 1.25, 2.5, 3.75]:
        fig.add_scatter3d(x=[x0, x0 + 2.7], y=[-0.05, -0.05], z=[zz - 0.02] * 2, mode="lines",
                          line=dict(color="#475569", width=5), hoverinfo="skip", showlegend=False)
    for xx in [x0, x0 + 2.7]:
        fig.add_scatter3d(x=[xx, xx], y=[-0.05, -0.05], z=[0, 3.75], mode="lines", line=dict(color="#475569", width=5),
                          hoverinfo="skip", showlegend=False)
    fig.add_scatter3d(x=[x0 + 1.35], y=[-0.6], z=[0], mode="text", text=[f"Rack {chr(65 + rack)}"],
                      textfont=dict(color="#94a3b8", size=12), hoverinfo="skip", showlegend=False)
for k, c in I.STATUS_COLORS.items():
    if k != "No data":
        fig.add_scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=8, color=c, symbol="square"), name=k)
fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.0, x=0),
                  scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                             aspectmode="manual", aspectratio=dict(x=2.0, y=0.4, z=0.65),
                             camera=dict(eye=dict(x=0.45, y=-2.05, z=0.9), center=dict(x=0, y=0, z=-0.1)),
                             bgcolor="rgba(0,0,0,0)"))
c1, c2 = st.columns([2.2, 1])
with c1:
    st.plotly_chart(fig, width="stretch", key="wh3d")
with c2:
    r = tab.set_index("sku").loc[sel]
    st.markdown(f"##### {sel} · {r['description']}")
    st.markdown(f"<span class='badge' style='color:{I.STATUS_COLORS[r['status']]};border-color:{I.STATUS_COLORS[r['status']]}'>"
                f"{r['status']}</span>", unsafe_allow_html=True)
    st.metric("Inventory position", f"{r['inventory_position']:,.0f}", f"{r['inventory_position'] - r['rop']:+,.0f} vs ROP")
    st.metric("Safety stock / ROP", f"{r['safety_stock']:,.0f} / {r['rop']:,.0f}")
    st.metric("Stockout risk before replenishment", f"{r['stockout_risk']:.0%}")
    p = ss.panel[ss.panel["sku"] == sel].tail(26)
    f = state.pipeline()["forecast"]
    f = f[f["sku"] == sel]
    mini = go.Figure()
    mini.add_scatter(x=p["week"], y=p["demand"], line=dict(color=ui.SKY, width=2), name="Actual")
    mini.add_scatter(x=f["week"], y=f["forecast"], line=dict(color=ui.TEAL, width=2, dash="dot"), name="Forecast")
    mini.update_layout(height=200, margin=dict(l=0, r=0, t=6, b=0), showlegend=False, yaxis_title="Units/wk")
    st.plotly_chart(mini, width="stretch")
    st.caption(r["action"])

with st.expander("2D fallback view (for devices without WebGL)"):
    grid = t3.assign(fill=lambda d: np.clip(d["inventory_position"] / np.maximum(2 * d["rop"], 1), 0, 1))
    fig2 = go.Figure(go.Bar(x=grid["sku"], y=grid["fill"], marker_color=grid["status"].map(I.STATUS_COLORS),
                            text=grid["status"], hovertext=grid["description"]))
    fig2.add_hline(y=0.5, line_dash="dot", line_color=ui.AMBER, annotation_text="ROP")
    fig2.update_layout(height=320, yaxis_title="Position ÷ (2 × ROP)", yaxis_tickformat=".0%")
    st.plotly_chart(fig2, width="stretch")
