import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inventory_core import optimization as O
from utils import state, ui

ss = st.session_state
s = state.settings()
ui.page_head("Workstream 3", "Bayesian optimisation lab",
             "Instead of sweeping parameters by hand, a Bayesian optimiser (Optuna) searches reorder point and "
             "order quantity together. It learns from every trial where low-cost, feasible policies are likely, "
             "and tries those next.")

tab = state.policy_table()
labels = {r.sku: f"{r.sku} · {r.description}" for r in tab.itertuples()}

with st.form("bo_form"):
    c = st.columns([2, 1, 1, 1])
    sku = c[0].selectbox("SKU", list(labels), format_func=labels.get, key="bo_sku")
    sampler = c[1].selectbox("Sampler", ["TPE", "GP"], key="bo_sampler",
                             help="TPE = tree-structured Parzen estimator; GP = Gaussian-process Bayesian optimisation (needs PyTorch).")
    trials = c[2].slider("Trials", 20, 200, 80, 10, key="bo_trials")
    paths = c[3].select_slider("Simulated years per trial", [100, 200, 400], 200, key="bo_paths")
    r = tab.set_index("sku").loc[sku]
    st.markdown("**Constraints**")
    c = st.columns(5)
    tgt = c[0].slider("Min cycle service level", 0.80, 0.995, float(s["service_level"]), 0.005, format="%.3f", key="bo_tgt")
    cap = c[1].number_input("Warehouse capacity (units)", 1.0, value=float(round(r["rop"] + 6 * r["eoq"], -2)),
                            step=500.0, key=f"bo_cap_{sku}")
    budget = c[2].number_input("Avg inventory budget (£)", 1.0,
                               value=float(round((r["safety_stock"] + 2 * r["eoq"]) * r["unit_cost"], -1)), step=100.0,
                               key=f"bo_bud_{sku}")
    moq = c[3].number_input("MOQ (units)", 1.0, value=float(max(s["moq"], 1)), step=50.0, key=f"bo_moq_{sku}")
    qmax = c[4].number_input("Max order size (supplier)", 2.0, value=float(round(max(26 * r["d_bar"], moq * 2), -1)),
                             step=500.0, key=f"bo_qmax_{sku}")
    run = st.form_submit_button("Run Bayesian optimisation", type="primary", width="stretch")

st.latex(r"\min_{s,\,Q}\; \mathbb{E}[\text{annual cost}] = \underbrace{h\,\mathbb{E}[\text{on hand}]}_{\text{holding}}"
         r"+ \underbrace{K\,\mathbb{E}[\#\text{orders}]}_{\text{ordering}} + \underbrace{p\,\mathbb{E}[\text{units short}]}_{\text{shortage}}")
st.latex(r"\text{s.t. } CSL \ge CSL^{*},\quad P_{99}(\text{peak stock}) \le \text{capacity},\quad \bar I\,c \le \text{budget},\quad "
         r"MOQ \le Q \le Q_{max}")
st.caption(f"h = unit cost × {s['holding_rate']:.0%}/yr; K = £{s['order_cost']:.0f}/order; p = {s['shortage_mult']:.1f} × selling "
           f"price per unit short. Each candidate is scored on the same simulated demand and lead-time paths (common random "
           "numbers, seed 42) under an (s, nQ) policy with backorders. The winner is then re-checked on fresh paths (seed 1042).")

if run:
    sc = O.Scenario(d_mean=float(r["d_bar"]), d_sd=float(r["sigma_d"]), l_mean=s["lead_mean"], l_sd=s["lead_sd"],
                    unit_cost=float(r["unit_cost"]), holding_rate=s["holding_rate"], order_cost=s["order_cost"],
                    shortage_cost=float(r["price"] * s["shortage_mult"]), target_csl=tgt, capacity=cap, budget=budget,
                    moq=moq, q_max=qmax, demand_dist=s["demand_dist"], lead_dist=s["lead_dist"], n_paths=int(paths))
    bar = st.progress(0.0, text="Optimising…")
    def cb(study, trial):
        bar.progress(min((trial.number + 1) / trials, 1.0), text=f"Trial {trial.number + 1}/{trials}")
    try:
        res = O.optimise(sc, n_trials=int(trials), sampler=sampler, callback=cb)
    except Exception as e:  # e.g. GP sampler without torch
        bar.empty()
        st.error(f"The {sampler} sampler could not run here ({type(e).__name__}: {e}). Choose TPE.")
        st.stop()
    bar.empty()
    res["sku"] = sku
    res["ran_at"] = state.now()
    ss.bo_last = res
    ss.opt_results[sku] = {"rop": res["best"]["rop"], "q": res["best"]["q"]}

res = ss.get("bo_last")
if res is None:
    st.info("Set the constraints and press **Run Bayesian optimisation** to search for the best policy for this SKU.")
    st.stop()
if res["sku"] != sku:
    st.info(f"Showing the last run, for SKU {res['sku']}. Press Run to optimise {sku}.")

b, bv, base, bev, bval = res["best"], res["best_validation"], res["baseline"], res["baseline_eval"], res["baseline_validation"]
saving = 1 - bv["total_cost"] / bval["total_cost"]
ui.kpis([
    ("Best reorder point", ui.fmt(b["rop"]), f"baseline {base['rop']:,.0f}", ui.TEAL),
    ("Best safety stock", ui.fmt(b["rop"] - res["scenario"]["d_mean"] * res["scenario"]["l_mean"]), f"baseline {base['ss']:,.0f}", ui.TEAL),
    ("Best order quantity", ui.fmt(b["q"]), f"baseline EOQ {base['q']:,.0f}", ui.BLUE),
    ("Expected annual cost", f"£{bv['total_cost']:,.0f}", f"baseline £{bval['total_cost']:,.0f} (validation)", ui.SKY),
    ("Service level (CSL)", f"{bv['cycle_service_level']:.1%}", f"fill rate {bv['fill_rate']:.2%}", ui.TEAL),
    ("Change vs baseline", f"{-saving:+.1%}", "cost on independent validation paths",
     ui.TEAL if saving > 0 else ui.AMBER),
])
if not res["any_feasible"]:
    st.error("No trial met every constraint. The policy shown has the highest service level found. Relax capacity, "
             "budget or the service target.")

c1, c2 = st.columns(2)
h = res["history"]
with c1:
    fig = go.Figure()
    fig.add_scatter(x=h["trial"], y=h["cost"], mode="markers", name="Trial cost",
                    marker=dict(color=np.where(h["feasible"], ui.TEAL, ui.RED), size=7, opacity=0.7))
    fig.add_scatter(x=h["trial"], y=h["best_feasible_so_far"], mode="lines", name="Best feasible so far",
                    line=dict(color=ui.AMBER, width=3, shape="hv"))
    fig.update_layout(height=370, title="Convergence (teal = feasible, red = violates a constraint)",
                      xaxis_title="Trial", yaxis_title="Expected annual cost (£)")
    lo, hi = np.nanpercentile(h["cost"], [0, 90])
    fig.update_yaxes(range=[lo * 0.9, hi * 1.1])
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure()
    fig.add_scatter(x=h["rop"], y=h["q"], mode="markers", name="Trials",
                    marker=dict(color=h["cost"].clip(upper=np.nanpercentile(h["cost"], 90)), colorscale="Tealgrn_r",
                                size=9, showscale=True, colorbar=dict(title="£/yr", thickness=10),
                                symbol=np.where(h["feasible"], "circle", "x"), line=dict(width=0)),
                    text=[f"cost £{c:,.0f}<br>CSL {v:.1%}" for c, v in zip(h["cost"], h["cycle_service_level"])],
                    hovertemplate="ROP %{x:,.0f}<br>Q %{y:,.0f}<br>%{text}<extra></extra>")
    fig.add_scatter(x=[base["rop"]], y=[base["q"]], mode="markers+text", text=["Baseline"], textposition="top center",
                    marker=dict(color=ui.AMBER, size=15, symbol="diamond"), name="Baseline")
    fig.add_scatter(x=[b["rop"]], y=[b["q"]], mode="markers+text", text=["Optimised"], textposition="bottom center",
                    marker=dict(color="#f8fafc", size=15, symbol="star"), name="Optimised")
    fig.update_layout(height=370, title="Where the optimiser searched (× = infeasible)", xaxis_title="Reorder point s",
                      yaxis_title="Order quantity Q", showlegend=False)
    st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns([1, 1.1])
with c1:
    fig = go.Figure()
    fig.add_scatter(x=h["cycle_service_level"], y=h["cost"], mode="markers",
                    marker=dict(color=np.where(h["feasible"], ui.TEAL, ui.SLATE), size=8, opacity=0.75), name="Trials")
    fig.add_vline(x=res["scenario"]["target_csl"], line_dash="dot", line_color=ui.AMBER,
                  annotation_text="service target", annotation_font_color=ui.AMBER)
    fig.add_scatter(x=[bev["cycle_service_level"]], y=[bev["total_cost"]], mode="markers", marker=dict(color=ui.AMBER, size=14, symbol="diamond"), name="Baseline")
    fig.add_scatter(x=[b["cycle_service_level"]], y=[b["cost"]], mode="markers", marker=dict(color="#f8fafc", size=15, symbol="star"), name="Optimised")
    fig.update_layout(height=360, title="Cost vs service level", xaxis_title="Cycle service level", yaxis_title="£ per year",
                      xaxis_tickformat=".0%")
    fig.update_yaxes(range=[np.nanmin(h["cost"]) * 0.9, np.nanpercentile(h["cost"], 90) * 1.1])
    st.plotly_chart(fig, width="stretch")
with c2:
    comp = pd.DataFrame({
        "Baseline (formula ROP + EOQ)": [bval["rop"], bval["q"], bval["holding_cost"], bval["ordering_cost"], bval["shortage_cost"],
                                         bval["total_cost"], bval["cycle_service_level"], bval["fill_rate"], bval["p99_max_on_hand"],
                                         bval["avg_on_hand"] * res["scenario"]["unit_cost"]],
        "Bayesian-optimised": [bv["rop"], bv["q"], bv["holding_cost"], bv["ordering_cost"], bv["shortage_cost"], bv["total_cost"],
                               bv["cycle_service_level"], bv["fill_rate"], bv["p99_max_on_hand"],
                               bv["avg_on_hand"] * res["scenario"]["unit_cost"]],
    }, index=["Reorder point", "Order quantity", "Holding £/yr", "Ordering £/yr", "Shortage £/yr", "Total £/yr",
              "Cycle service level", "Fill rate", "P99 peak stock (units)", "Avg stock value (£)"])
    fmt = lambda v, i: f"{v:.1%}" if i in ("Cycle service level",) else (f"{v:.2%}" if i == "Fill rate" else f"{v:,.0f}")
    st.markdown("##### Validation on independent simulated paths")
    st.dataframe(pd.DataFrame({c: [fmt(v, i) for v, i in zip(comp[c], comp.index)] for c in comp}, index=comp.index),
                 width="stretch")
    sc_ = res["scenario"]
    checks = [("Cycle service level ≥ target", bv["cycle_service_level"] >= sc_["target_csl"], f"{bv['cycle_service_level']:.1%} vs {sc_['target_csl']:.1%}"),
              ("P99 peak stock ≤ capacity", bv["p99_max_on_hand"] <= sc_["capacity"], f"{bv['p99_max_on_hand']:,.0f} vs {sc_['capacity']:,.0f}"),
              ("Avg stock value ≤ budget", bv["avg_on_hand"] * sc_["unit_cost"] <= sc_["budget"], f"£{bv['avg_on_hand'] * sc_['unit_cost']:,.0f} vs £{sc_['budget']:,.0f}"),
              ("MOQ ≤ Q ≤ max order", sc_["moq"] <= bv["q"] <= (sc_["q_max"] or 1e18), f"{bv['q']:,.0f} in [{sc_['moq']:,.0f}, {sc_['q_max']:,.0f}]")]
    st.markdown("##### Constraint check (validation run)")
    for name, ok, det in checks:
        st.markdown(f"{'✅' if ok else '❌'} **{name}**: {det}")

base_ok = bval["cycle_service_level"] >= sc_["target_csl"]
ui.note(f"<b>Baseline vs optimised.</b> The baseline uses the lecture formula for the ROP and the EOQ for the order size. On "
        f"simulated demand it reaches a {bval['cycle_service_level']:.1%} cycle service level "
        f"({'meets' if base_ok else 'misses'} the {sc_['target_csl']:.0%} target). The optimiser trades holding, ordering and "
        f"shortage cost against each other inside the constraints. It settles on ROP {bv['rop']:,.0f} and Q {bv['q']:,.0f}, "
        f"which costs £{bv['total_cost']:,.0f}/yr vs £{bval['total_cost']:,.0f}/yr ({-saving:+.1%}) on paths it never saw. "
        f"Sampler: {res['sampler']}, {res['n_trials']} trials, run at {res['ran_at']}.")
st.download_button("Download trial history (CSV)", h.to_csv(index=False).encode(), f"bo_trials_{res['sku']}.csv", "text/csv", key="bo_dl")
