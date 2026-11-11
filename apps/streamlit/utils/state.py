"""Session state, data loading and cached computations.

Privacy: the public dataset is cached globally (st.cache_data). Anything derived from data a
user uploads is cached only inside that user's session (st.session_state), never shared.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from inventory_core import data as D, forecasting as F, inventory as I, simulation as S  # noqa: E402

RAW = ROOT / "data" / "raw" / "onlineretail2.rda"
FEED_START = pd.Timestamp("2011-10-10")        # weeks from here on are replayed as "new data"

DEFAULTS = dict(service_level=0.95, lead_mean=3.0, lead_sd=1.0, lead_dist="Gamma", demand_dist="Gamma",
                holding_rate=0.25, order_cost=60.0, cost_ratio=0.60, shortage_mult=1.0, moq=1.0,
                horizon=8, strategy="global", sigma_source="Forecast error (AI)", stock_seed=7, rho=0.0)


@st.cache_data(show_spinner=False)
def load_public():
    raw = D.load_raw(RAW)
    clean, report = D.clean_transactions(raw)
    panel = D.weekly_panel(clean)
    skus = panel["sku"].unique()
    tx = clean[clean["StockCode"].isin(skus)][["InvoiceDate", "StockCode", "Quantity", "Price", "Country"]]
    return panel, report, tx, D.sha256_bytes(RAW.read_bytes())


def init():
    ss = st.session_state
    if "panel" in ss:
        return
    panel, report, tx, sha = load_public()
    ss.full_panel = panel
    ss.panel = panel[panel["week"] < FEED_START].reset_index(drop=True)
    ss.quality = report
    ss.tx = tx
    ss.sha = sha
    ss.settings = dict(DEFAULTS)
    ss.ingest_log = [{"time": now(), "event": "Initial load", "detail":
                      f"Public dataset through week of {ss.panel['week'].max():%d %b %Y}; "
                      f"{(panel['week'] >= FEED_START).sum() // panel['sku'].nunique()} later weeks held back as a replay feed"}]
    ss.last_ingest = now()
    ss.last_model_run = None
    ss.cache = {}
    ss.stock = None
    ss.stock_source = "Simulated scenario"
    ss.opt_results = {}
    ss.uploaded = False


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def panel_key(panel: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(panel, index=False).values.tobytes()).hexdigest()


def settings() -> dict:
    return st.session_state.settings


@st.cache_data(show_spinner=False, max_entries=20)
def _public_pipeline(panel, horizon, strategy):
    return F.run_pipeline(panel, horizon=horizon, strategy=strategy)


def pipeline():
    """Forecast pipeline for the current panel and settings (session-scoped cache)."""
    ss = st.session_state
    s = settings()
    key = ("pipe", panel_key(ss.panel), s["horizon"], s["strategy"])
    if key not in ss.cache:
        with st.spinner("Training and back-testing 7 forecasting models on every SKU (about 30 s)…"):
            if ss.uploaded:      # private data: compute inside this session only
                ss.cache[key] = F.run_pipeline(ss.panel, horizon=s["horizon"], strategy=s["strategy"])
            else:                # public data only: safe to share the result between sessions
                ss.cache[key] = _public_pipeline(ss.panel, s["horizon"], s["strategy"])
            ss.last_model_run = now()
    return ss.cache[key]


def sku_params() -> pd.DataFrame:
    """Per-SKU demand parameters: forecast mean over the horizon and the chosen sigma_D."""
    ss = st.session_state
    s = settings()
    out = pipeline()
    fc = out["forecast"].groupby("sku")["forecast"].mean().rename("d_bar")
    vb = out["val_bt"].merge(out["best"][["sku", "best_model"]], left_on=["sku", "model"],
                             right_on=["sku", "best_model"])
    err_sd = vb.assign(e=vb["forecast"] - vb["actual"]).groupby("sku")["e"].apply(
        lambda e: float(np.sqrt((e ** 2).mean()))).rename("forecast_rmse")
    p = ss.panel
    last26 = p[p["week"] > p["week"].max() - pd.Timedelta(weeks=26)]
    hist = last26.groupby("sku")["demand"].agg(hist_mean="mean", hist_sd="std")
    price = p.groupby("sku")["avg_price"].median().rename("price")
    desc = p.groupby("sku")["description"].first()
    t = pd.concat([desc, fc, err_sd, hist, price], axis=1).reset_index().rename(columns={"index": "sku"})
    t["sigma_d"] = t["forecast_rmse"] if s["sigma_source"].startswith("Forecast") else t["hist_sd"]
    t["unit_cost"] = t["price"] * s["cost_ratio"]
    t["best_model"] = t["sku"].map(out["best"].set_index("sku")["best_model"])
    return t


def stock_table(params: pd.DataFrame) -> pd.DataFrame:
    """Stock on hand is NOT in the dataset. Default = clearly labelled simulated scenario
    (2-10 weeks of forecast cover, fixed seed); users can upload real stock instead."""
    ss = st.session_state
    if ss.stock is not None and set(params["sku"]) <= set(ss.stock["sku"]):
        return ss.stock
    rng = np.random.default_rng(settings()["stock_seed"])
    cover = rng.uniform(2.0, 10.0, len(params))
    st_ = pd.DataFrame({"sku": params["sku"], "on_hand": np.round(params["d_bar"].values * cover),
                        "on_order": 0.0, "backorders": 0.0})
    ss.stock = st_
    ss.stock_source = "Simulated scenario"
    return st_


def policy_table() -> pd.DataFrame:
    s = settings()
    params = sku_params()
    stock = stock_table(params)
    t = params.merge(stock, on="sku", how="left").fillna({"on_hand": 0, "on_order": 0, "backorders": 0})
    t = I.sku_policy_table(t, s["service_level"], s["lead_mean"], s["lead_sd"], s["order_cost"],
                           s["holding_rate"], s["moq"])
    risks, csl = [], []
    for _, r in t.iterrows():
        sim = S.lead_time_demand(4000, max(r["d_bar"], 1e-6), max(r["sigma_d"], 1e-6), s["lead_mean"], s["lead_sd"],
                                 s["demand_dist"], s["lead_dist"], s["rho"], seed=11)
        risks.append(float((sim["dlt"] > r["inventory_position"]).mean()))
        csl.append(float((sim["dlt"] <= r["rop"]).mean()))
    t["stockout_risk"] = risks            # P(lead-time demand > current inventory position)
    t["csl_at_rop"] = csl                 # simulated cycle service level of the formula ROP
    opt = st.session_state.opt_results
    t["opt_rop"] = t["sku"].map(lambda k: opt.get(k, {}).get("rop", np.nan))
    t["opt_q"] = t["sku"].map(lambda k: opt.get(k, {}).get("q", np.nan))
    return t


def sidebar():
    ss = st.session_state
    s = settings()
    with st.sidebar:
        from .ui import IDENT
        st.markdown(f'<div class="ident">{IDENT}</div>', unsafe_allow_html=True)
        st.markdown("### Scenario settings")
        st.caption("Used by every page. Lead times and costs are **assumptions** (not in the dataset).")
        s["service_level"] = st.slider("Target cycle service level", 0.80, 0.995, s["service_level"], 0.005,
                                       format="%.3f", key="sb_sl")
        c1, c2 = st.columns(2)
        s["lead_mean"] = c1.number_input("Lead time (wks)", 0.5, 12.0, s["lead_mean"], 0.5, key="sb_lm")
        s["lead_sd"] = c2.number_input("Lead-time SD", 0.0, 6.0, s["lead_sd"], 0.25, key="sb_ls")
        s["horizon"] = st.select_slider("Forecast horizon (weeks)", [4, 6, 8, 10, 12], s["horizon"], key="sb_h")
        with st.expander("Costs & policy assumptions"):
            s["cost_ratio"] = st.slider("Unit cost as % of selling price", 0.3, 0.9, s["cost_ratio"], 0.05, key="sb_cr")
            s["holding_rate"] = st.slider("Holding cost (% of unit cost / year)", 0.05, 0.6, s["holding_rate"], 0.05, key="sb_hr")
            s["order_cost"] = st.number_input("Ordering cost per order (£)", 1.0, 1000.0, s["order_cost"], 5.0, key="sb_oc")
            s["shortage_mult"] = st.slider("Shortage cost per unit (× selling price)", 0.2, 5.0, s["shortage_mult"], 0.1, key="sb_sm")
            s["moq"] = st.number_input("Minimum order quantity (units)", 1.0, 100000.0, s["moq"], 50.0, key="sb_moq")
        with st.expander("Model settings"):
            s["strategy"] = st.radio("Model selection", ["global", "per_sku"], index=["global", "per_sku"].index(s["strategy"]),
                                     format_func=lambda x: "One best model for all SKUs" if x == "global" else "Best model per SKU",
                                     key="sb_strat")
            s["sigma_source"] = st.radio("σ_D used for safety stock", ["Forecast error (AI)", "Historical demand SD"],
                                         index=0 if s["sigma_source"].startswith("Forecast") else 1, key="sb_sig")
            s["demand_dist"] = st.selectbox("Demand distribution (simulation)", S.DEMAND_DISTS[:3],
                                            index=S.DEMAND_DISTS[:3].index(s["demand_dist"]), key="sb_dd")
            s["lead_dist"] = st.selectbox("Lead-time distribution", S.LEAD_DISTS,
                                          index=S.LEAD_DISTS.index(s["lead_dist"]), key="sb_ld")
        if st.button("Reset scenario to defaults", width="stretch", key="sb_reset"):
            for k in [k for k in ss.keys() if k.startswith("sb_")]:
                del ss[k]
            ss.settings = dict(DEFAULTS)
            ss.opt_results = {}
            st.rerun()
        st.markdown(f'<div class="stamp">Data through: week of {ss.panel["week"].max():%d %b %Y}<br>'
                    f'Last data update: {ss.last_ingest}<br>Last model run: {ss.last_model_run or "pending"}<br>'
                    f'Refresh mode: on-demand batch (not real-time)</div>', unsafe_allow_html=True)
