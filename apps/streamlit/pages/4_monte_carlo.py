import numpy as np
import plotly.graph_objects as go
import streamlit as st

from inventory_core import inventory as I, simulation as S
from utils import state, ui

ss = st.session_state
s = state.settings()
ui.page_head("Workstream 2b", "Monte Carlo risk lab",
             "Simulate thousands of replenishment cycles to see the full distribution of demand during the lead "
             "time. From it come the buffer needed, the stockout probability and the expected shortage under "
             "each risk scenario.")

tab = state.policy_table()
labels = {r.sku: f"{r.sku} · {r.description}" for r in tab.itertuples()}
r0 = None
with st.form("mc_form"):
    c = st.columns([2, 1, 1])
    sku = c[0].selectbox("SKU (pre-fills the parameters)", list(labels), format_func=labels.get, key="mc_sku")
    ddist = c[1].selectbox("Demand distribution", S.DEMAND_DISTS, index=S.DEMAND_DISTS.index(s["demand_dist"]), key="mc_dd")
    ldist = c[2].selectbox("Lead-time distribution", S.LEAD_DISTS, index=S.LEAD_DISTS.index(s["lead_dist"]), key="mc_ld")
    r0 = tab.set_index("sku").loc[sku]
    c = st.columns(6)
    dm = c[0].number_input("Mean demand / wk", 0.0, value=float(round(r0["d_bar"], 1)), key=f"mc_dm_{sku}")
    dsd = c[1].number_input("Demand SD / wk", 0.0, value=float(round(r0["sigma_d"], 1)), key=f"mc_dsd_{sku}")
    lm = c[2].number_input("Mean lead time (wk)", 0.1, value=float(s["lead_mean"]), step=0.5, key="mc_lm")
    lsd = c[3].number_input("Lead-time SD (wk)", 0.0, value=float(s["lead_sd"]), step=0.25, key="mc_lsd")
    sl = c[4].slider("Service level", 0.80, 0.995, float(s["service_level"]), 0.005, format="%.3f", key="mc_sl")
    n = c[5].select_slider("Simulations", [1000, 5000, 10000, 25000, 50000, 100000], 10000, key="mc_n")
    c = st.columns([1, 1, 2])
    seed = c[0].number_input("Random seed", 0, 10_000, 42, key="mc_seed")
    rho = c[1].slider("Demand ↔ lead-time correlation ρ", -0.5, 0.9, float(s["rho"]), 0.05, key="mc_rho",
                      help="ρ = 0 is the independence the formula assumes. ρ > 0 means supplier delays tend to coincide with high demand (e.g. peak season).")
    run = st.form_submit_button("Run simulation", type="primary", width="stretch")

emp = ss.panel[ss.panel["sku"] == sku].sort_values("week")["demand"].values[-52:]
key = (sku, ddist, ldist, dm, dsd, lm, lsd, sl, n, seed, rho)
if run or ss.get("mc_last_key") != key and "mc_result" not in ss:
    sim = S.lead_time_demand(int(n), max(dm, 1e-6), max(dsd, 1e-6), lm, lsd, ddist, ldist, rho, emp, int(seed))
    ss.mc_result = sim
    ss.mc_last_key = key
    ss.mc_scen = S.scenario_table(int(n), max(dm, 1e-6), max(dsd, 1e-6), lm, lsd, sl,
                                  I.reorder_point(dm, lm, I.safety_stock(I.z_value(sl), dsd, lm, dm, lsd)),
                                  ddist, ldist, rho, emp, int(seed))
    ss.mc_ran_at = state.now()
if ss.get("mc_last_key") != key:
    st.info("Settings changed. Press **Run simulation** to update the results below.")
sim = ss.mc_result
k = ss.mc_last_key
dm_, dsd_, lm_, lsd_, sl_, n_, seed_, rho_ = k[3], k[4], k[5], k[6], k[7], k[8], k[9], k[10]
z = I.z_value(sl_)
ss_f = I.safety_stock(z, dsd_, lm_, dm_, lsd_)
rop_f = I.reorder_point(dm_, lm_, ss_f)
res = S.summarise_dlt(sim["dlt"], sl_, rop=rop_f, q=max(tab.set_index('sku').loc[k[0], 'eoq'], 1), d_mean=dm_, l_mean=lm_)

ui.kpis([
    ("Required buffer (MC)", ui.fmt(res["ss_mc"]), f"units for {sl_:.1%} CSL", ui.TEAL),
    ("Formula buffer", ui.fmt(ss_f), "units, lecture formula", ui.BLUE),
    ("P(stockout) at formula ROP", f"{res['stockout_prob']:.1%}", f"target {1 - sl_:.1%}",
     ui.RED if res["stockout_prob"] > 1 - sl_ + 0.005 else ui.TEAL),
    ("Expected shortage", ui.fmt(res["expected_shortage"], 1), "units per cycle at formula ROP", ui.AMBER),
    ("Fill rate", f"{res['fill_rate']:.2%}", "share of demand served from stock (Q = EOQ)", ui.SKY),
    ("Lead-time demand P99", ui.fmt(res["p99"]), "units, worst 1% of cycles", ui.SLATE),
])
st.markdown(f"<span class='stamp'>{n_:,} simulated cycles · seed {seed_} · {k[1]} demand · {k[2]} lead time · ρ = {rho_:.2f} · "
            f"run at {ss.mc_ran_at}</span>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
dlt = sim["dlt"]
with c1:
    fig = go.Figure()
    fig.add_histogram(x=dlt, nbinsx=80, histnorm="probability", marker_color=ui.BLUE, opacity=0.85, name="Simulated")
    for (x, name, col), yy in zip([(dm_ * lm_, "Mean D̄·L", ui.MUTED), (rop_f, "Formula ROP", ui.AMBER),
                                   (res["rop_mc"], "MC ROP", ui.TEAL)], [0.97, 0.86, 0.75]):
        fig.add_vline(x=x, line_color=col, line_width=2, line_dash="dash" if name != "MC ROP" else "solid")
        fig.add_annotation(x=x, y=yy, yref="paper", text=f"{name} {x:,.0f}", showarrow=False, xanchor="left",
                           font=dict(color=col, size=11), xshift=4, bgcolor="rgba(11,17,32,0.75)")
    fig.update_layout(height=380, title="Demand during lead time: probability histogram", xaxis_title="Units",
                      yaxis_title="Share of simulated cycles", yaxis_tickformat=".1%", showlegend=False)
    st.plotly_chart(fig, width="stretch")
with c2:
    xs = np.sort(dlt)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    step = max(1, len(xs) // 1500)
    fig = go.Figure()
    fig.add_scatter(x=xs[::step], y=ys[::step], line=dict(color=ui.TEAL, width=3), name="CDF")
    fig.add_hline(y=sl_, line_dash="dot", line_color=ui.AMBER)
    fig.add_vline(x=res["rop_mc"], line_dash="dot", line_color=ui.AMBER)
    fig.add_annotation(x=res["rop_mc"], y=sl_, text=f"{sl_:.1%} → {res['rop_mc']:,.0f} units", showarrow=True,
                       arrowcolor=ui.AMBER, font=dict(color=ui.AMBER), ax=60, ay=40)
    fig.update_layout(height=380, title="Cumulative probability: P(lead-time demand ≤ x)", xaxis_title="Units",
                      yaxis_title="Cumulative probability", yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, width="stretch")

st.markdown("#### Risk scenarios (same random numbers in every scenario)")
sc = ss.mc_scen
c1, c2 = st.columns([1.2, 1])
with c1:
    fig = go.Figure()
    xl = sc["scenario"].str.replace(" (", "<br>(", regex=False)
    fig.add_bar(x=xl, y=sc["SS formula"], name="Formula", marker_color=ui.BLUE)
    fig.add_bar(x=xl, y=sc["SS Monte Carlo"], name="Monte Carlo", marker_color=ui.TEAL)
    fig.update_layout(barmode="group", height=380, title="Safety stock required by scenario", yaxis_title="Units",
                      xaxis_tickangle=0, legend=dict(y=1.08, yanchor="bottom", x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure(go.Bar(x=sc["scenario"].str.replace(" (", "<br>(", regex=False), y=sc["P(stockout) at current ROP"],
                           marker_color=[ui.TEAL if v <= 1 - sl_ + 0.005 else ui.RED for v in sc["P(stockout) at current ROP"]],
                           text=[f"{v:.0%}" for v in sc["P(stockout) at current ROP"]], textposition="outside"))
    fig.add_hline(y=1 - sl_, line_dash="dot", line_color=ui.AMBER, annotation_text="target", annotation_font_color=ui.AMBER)
    fig.update_layout(height=380, title="Stockout probability if the normal-case ROP is kept", yaxis_tickformat=".0%", xaxis_tickangle=0)
    st.plotly_chart(fig, width="stretch")
st.dataframe(sc.style.format({c: "{:,.1f}" for c in sc.columns if c not in ("scenario", "P(stockout) at current ROP")}
                             | {"P(stockout) at current ROP": "{:.1%}"}), hide_index=True, width="stretch")
ui.note("<b>Cycle service level</b> is the probability that a replenishment cycle has no stockout; it is what the Z factor "
        "targets. <b>Fill rate</b> is the share of units demanded that are served from stock. A 95% cycle service level "
        "usually gives a fill rate well above 95%, so the two should not be confused. When ρ ≠ 0, the formula's "
        "independence assumption is broken and only the joint simulation gives the right buffer.")
