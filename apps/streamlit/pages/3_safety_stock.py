import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from inventory_core import inventory as I, simulation as S
from utils import state, ui

ss = st.session_state
s = state.settings()
ui.page_head("Workstream 2a", "Safety stock & reorder point calculator",
             "The lecture formula, filled in automatically from the AI forecast. Every input can be overridden. "
             "The result is then checked against a Monte Carlo simulation using the same inputs.")

tab = state.policy_table()
labels = {r.sku: f"{r.sku} · {r.description}" for r in tab.itertuples()}
sku = st.selectbox("SKU", list(labels), format_func=labels.get, key="ss_sku")
r = tab.set_index("sku").loc[sku]

st.markdown(f"Defaults: D̄ = forecast mean {ui.badge('derived')} · σ_D = {s['sigma_source'].lower()} {ui.badge('derived')} · "
            f"L, σ_L {ui.badge('assumed')} · stock {ui.badge('simulated' if ss.stock_source.startswith('Sim') else 'observed')}",
            unsafe_allow_html=True)
c = st.columns(5)
d_bar = c[0].number_input("Mean demand D̄ (units/wk)", 0.0, value=float(round(r["d_bar"], 1)), step=10.0, key=f"ss_d_{sku}")
sig_d = c[1].number_input("Demand SD σ_D (units/wk)", 0.0, value=float(round(r["sigma_d"], 1)), step=10.0, key=f"ss_sd_{sku}")
L = c[2].number_input("Lead time L (weeks)", 0.1, value=float(s["lead_mean"]), step=0.5, key=f"ss_L_{sku}")
sig_l = c[3].number_input("Lead-time SD σ_L (weeks)", 0.0, value=float(s["lead_sd"]), step=0.25, key=f"ss_sl_{sku}")
sl = c[4].slider("Service level", 0.80, 0.995, float(s["service_level"]), 0.005, format="%.3f", key=f"ss_svc_{sku}")
c = st.columns(5)
on_hand = c[0].number_input("Current stock", 0.0, value=float(r["on_hand"]), step=50.0, key=f"ss_oh_{sku}")
on_order = c[1].number_input("On order", 0.0, value=float(r["on_order"]), step=50.0, key=f"ss_oo_{sku}")
back = c[2].number_input("Backorders", 0.0, value=float(r["backorders"]), step=10.0, key=f"ss_bo_{sku}")
q_base = c[3].number_input("Order quantity Q (EOQ default)", 1.0, value=float(round(r["eoq"])), step=50.0, key=f"ss_q_{sku}")
n_sim = c[4].select_slider("MC simulations", [2000, 5000, 10000, 20000, 50000], 10000, key="ss_n")

z = I.z_value(sl)
ss_val = I.safety_stock(z, sig_d, L, d_bar, sig_l)
rop = I.reorder_point(d_bar, L, ss_val)
ip = I.inventory_position(on_hand, on_order, back)
status = I.classify(ip, ss_val, rop, d_bar)
oq = I.order_quantity(ip, rop, q_base, s["moq"])

ui.kpis([
    ("Z factor", f"{z:.3f}", f"for {sl:.1%} cycle service level", ui.BLUE),
    ("Demand during lead time", ui.fmt(d_bar * L), "D̄ × L (units)", ui.SKY),
    ("Safety stock", ui.fmt(ss_val), "units (analytical)", ui.TEAL),
    ("Reorder point", ui.fmt(rop), "D̄ × L + SS", ui.TEAL),
    ("Inventory position", ui.fmt(ip), "on hand + on order − backorders", ui.SLATE),
    ("Recommendation", status, f"order {oq:,.0f} units" if oq else "no order", I.STATUS_COLORS[status]),
])

c1, c2 = st.columns([1.1, 1])
with c1:
    st.markdown("##### The calculation, step by step")
    st.latex(r"SS = Z\sqrt{\sigma_D^2\,L + \bar D^2\,\sigma_L^2}")
    st.latex(rf"SS = {z:.3f}\sqrt{{{sig_d:,.1f}^2 \times {L:g} + {d_bar:,.1f}^2 \times {sig_l:g}^2}}")
    st.latex(rf"\phantom{{SS}} = {z:.3f}\times{np.sqrt(sig_d**2*L + d_bar**2*sig_l**2):,.1f} = \mathbf{{{ss_val:,.0f}}}")
    st.latex(rf"ROP = \bar D L + SS = {d_bar:,.1f}\times{L:g} + {ss_val:,.0f} = \mathbf{{{rop:,.0f}}}")
    dv, lv = sig_d ** 2 * L, d_bar ** 2 * sig_l ** 2
    st.markdown(f"Demand uncertainty contributes **{dv / (dv + lv + 1e-12):.0%}** of the lead-time variance, and "
                f"lead-time uncertainty **{lv / (dv + lv + 1e-12):.0%}**. The formula assumes weekly demands are "
                "independent, lead time is independent of demand, and lead-time demand is roughly normal. "
                "All inputs are per week, so the units match.")
with c2:
    sim = S.lead_time_demand(int(n_sim), max(d_bar, 1e-6), max(sig_d, 1e-6), L, sig_l, s["demand_dist"],
                             s["lead_dist"], s["rho"], seed=42)
    mc = S.summarise_dlt(sim["dlt"], sl, rop=rop, q=q_base, d_mean=d_bar, l_mean=L)
    st.markdown("##### Analytical vs Monte Carlo")
    cmp_tbl = {
        "Safety stock": (ss_val, mc["ss_mc"]), "Reorder point": (rop, mc["rop_mc"]),
        "Cycle service level achieved": (sl, mc["cycle_service_level"]),
    }
    st.markdown(f"""
| Measure | Formula | Monte Carlo ({n_sim:,} runs, seed 42) |
|---|---:|---:|
| Safety stock | {ss_val:,.0f} | {mc['ss_mc']:,.0f} |
| Reorder point | {rop:,.0f} | {mc['rop_mc']:,.0f} |
| Service level if ROP = formula ROP | {sl:.1%} (assumed) | {mc['cycle_service_level']:.1%} (simulated) |
| Expected units short per cycle | n/a | {mc['expected_shortage']:,.1f} |
| Fill rate with Q = {q_base:,.0f} | n/a | {mc['fill_rate']:.2%} |
""")
    diff = mc["ss_mc"] - ss_val
    st.markdown(f"The simulation ({s['demand_dist'].lower()} demand, {s['lead_dist'].lower()} lead time) needs "
                f"**{abs(diff):,.0f} units {'more' if diff > 0 else 'less'}** buffer than the formula. Skewed demand and "
                "a random lead time make the tail of lead-time demand heavier than a normal curve.")

st.markdown("##### How the buffer grows with uncertainty")
fig = make_subplots(rows=1, cols=3, subplot_titles=("Safety stock vs service level", "vs demand variability (σ_D)",
                                                    "vs lead-time uncertainty (σ_L)"))
sls = np.linspace(0.80, 0.995, 60)
fig.add_scatter(x=sls, y=[I.safety_stock(I.z_value(v), sig_d, L, d_bar, sig_l) for v in sls], line=dict(color=ui.TEAL, width=3),
                name="SS", row=1, col=1)
fig.add_scatter(x=[sl], y=[ss_val], mode="markers", marker=dict(size=12, color=ui.AMBER), name="current", row=1, col=1)
sds = np.linspace(0, max(sig_d * 2, 1), 50)
fig.add_scatter(x=sds, y=[I.safety_stock(z, v, L, d_bar, sig_l) for v in sds], line=dict(color=ui.BLUE, width=3), row=1, col=2)
fig.add_scatter(x=[sig_d], y=[ss_val], mode="markers", marker=dict(size=12, color=ui.AMBER), row=1, col=2)
sls2 = np.linspace(0, max(sig_l * 2.5, 1.5), 50)
fig.add_scatter(x=sls2, y=[I.safety_stock(z, sig_d, L, d_bar, v) for v in sls2], line=dict(color=ui.SKY, width=3), row=1, col=3)
fig.add_scatter(x=[sig_l], y=[ss_val], mode="markers", marker=dict(size=12, color=ui.AMBER), row=1, col=3)
fig.update_layout(height=330, showlegend=False)
fig.update_xaxes(tickformat=".0%", row=1, col=1, title_text="Cycle service level")
fig.update_xaxes(title_text="σ_D (units/wk)", row=1, col=2)
fig.update_xaxes(title_text="σ_L (weeks)", row=1, col=3)
fig.update_yaxes(title_text="Safety stock (units)", row=1, col=1)
st.plotly_chart(fig, width="stretch")
st.caption("The curve gets steeper as the service level approaches 100%: each extra point of service costs more stock. "
           "Lead-time uncertainty is multiplied by D̄², so for fast movers a reliable supplier matters more than a precise forecast.")
