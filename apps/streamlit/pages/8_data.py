import pandas as pd
import streamlit as st

from inventory_core import data as D
from utils import state, ui

ss = st.session_state
ui.page_head("Data & reports", "Data management",
             "Add new demand data and the whole chain refreshes: features, forecasts, safety stock, reorder points "
             "and recommendations. Uploads stay in your browser session only and are never shared or stored.")

st.markdown("#### Data lineage")
st.markdown(f"""
| Kind | What | Source |
|---|---|---|
| {ui.badge('observed')} | Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country | UCI Online Retail II (Chen, 2019), CC BY 4.0 · SHA-256 `{ss.sha[:16]}…` |
| {ui.badge('derived')} | Weekly demand per SKU, weekly average price, price-cut flag, forecasts, forecast-error σ_D | Computed in `inventory_core` |
| {ui.badge('external')} | England & Wales bank holidays per week | `holidays` Python package |
| {ui.badge('assumed')} | Lead time & its SD, unit cost %, holding %, ordering cost, shortage cost, MOQ, capacity, budget | Sidebar / page inputs |
| {ui.badge('simulated')} | Stock on hand (until real stock is uploaded), Monte Carlo demand & lead-time paths | Seeded random generators |
""", unsafe_allow_html=True)

feed = ss.full_panel[ss.full_panel["week"] > ss.panel["week"].max()]
pending = sorted(feed["week"].unique())
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Replay feed (real observations)")
    st.caption("The last weeks of the real dataset were held back at start-up. Ingest them one week at a time to watch the "
               "forecasts and buffers update, as they would with a weekly batch feed.")
    st.markdown(f"**{len(pending)}** week(s) waiting" + (f": next is the week of {pd.Timestamp(pending[0]):%d %b %Y}." if pending else "."))
    b1, b2 = st.columns(2)
    if b1.button("Ingest next week", disabled=not pending, type="primary", width="stretch", key="dm_next"):
        wk = feed[feed["week"] == pending[0]][["sku", "week", "demand", "avg_price"]]
        ss.panel, info = D.append_weekly(ss.panel, wk)
        ss.last_ingest = state.now()
        ss.ingest_log.append({"time": ss.last_ingest, "event": "Replay feed",
                              "detail": f"Week of {pd.Timestamp(pending[0]):%d %b %Y}: {info['rows_used']} SKU rows, {info['new_weeks']} new week"})
        ss.stock = None
        st.rerun()
    if b2.button("Ingest all remaining", disabled=not pending, width="stretch", key="dm_all"):
        ss.panel, info = D.append_weekly(ss.panel, feed[["sku", "week", "demand", "avg_price"]])
        ss.last_ingest = state.now()
        ss.ingest_log.append({"time": ss.last_ingest, "event": "Replay feed", "detail": f"{info['new_weeks']} weeks appended"})
        ss.stock = None
        st.rerun()
with c2:
    st.markdown("#### Upload new demand data (CSV)")
    up = st.file_uploader("Transactions or weekly demand: needs a date, a SKU and a quantity column", type=["csv"], key="dm_up")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file as CSV: {e}")
            st.stop()
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
        guess = D.detect_columns(df)
        opts = ["(none)"] + list(df.columns)
        m = {}
        cc = st.columns(4)
        for i, f in enumerate(["date", "sku", "quantity", "price"]):
            g = guess.get(f)
            m[f] = cc[i].selectbox(f"{f.title()}{' *' if f in D.REQUIRED_FIELDS else ''}", opts,
                                   index=opts.index(g) if g in opts else 0, key=f"dm_map_{f}")
        m = {k: v for k, v in m.items() if v != "(none)"}
        res = D.validate_new_data(df, m)
        for w in res.warnings:
            st.warning(w)
        for e in res.errors:
            st.error(e)
        if res.ok:
            known = res.data["sku"].isin(ss.panel["sku"]).mean()
            st.success(f"Valid: {len(res.data)} weekly SKU rows, {res.data['week'].nunique()} week(s); {known:.0%} of rows match tracked SKUs.")
            st.dataframe(res.data.head(8), hide_index=True, width="stretch")
            if st.button("Append & refresh models", type="primary", key="dm_apply"):
                ss.panel, info = D.append_weekly(ss.panel, res.data)
                ss.last_ingest = state.now()
                ss.uploaded = True
                ss.stock = None
                ss.ingest_log.append({"time": ss.last_ingest, "event": "CSV upload",
                                      "detail": f"{info['rows_used']} rows used, {info['replaced']} overlapping weeks reconciled, "
                                                f"{info['new_weeks']} new weeks; unknown SKUs ignored: {', '.join(info['unknown_skus']) or 'none'}"})
                st.rerun()
    st.download_button("Download a template CSV", "InvoiceDate,StockCode,Quantity,Price\n2011-12-05,84077,240,0.29\n".encode(),
                       "new_data_template.csv", key="dm_tpl")

st.markdown("#### Stock on hand")
c1, c2 = st.columns([1, 1])
with c1:
    st.caption("The dataset has no stock levels. Upload a CSV with columns sku, on_hand (and optionally on_order, backorders) "
               "to replace the simulated scenario.")
    upst = st.file_uploader("Stock CSV", type=["csv"], key="dm_stock")
    if upst is not None:
        sdf = pd.read_csv(upst)
        sdf.columns = [c.strip().lower() for c in sdf.columns]
        if not {"sku", "on_hand"} <= set(sdf.columns):
            st.error("The stock file needs at least the columns sku and on_hand.")
        else:
            sdf["sku"] = sdf["sku"].astype(str).str.upper()
            for c in ["on_order", "backorders"]:
                sdf[c] = pd.to_numeric(sdf.get(c, 0), errors="coerce").fillna(0)
            sdf["on_hand"] = pd.to_numeric(sdf["on_hand"], errors="coerce").fillna(0)
            missing = set(ss.panel["sku"]) - set(sdf["sku"])
            if missing:
                st.warning(f"{len(missing)} tracked SKU(s) missing from the file; they are set to 0 on hand.")
            base = pd.DataFrame({"sku": sorted(ss.panel["sku"].unique())}).merge(sdf[["sku", "on_hand", "on_order", "backorders"]], how="left").fillna(0)
            if st.button("Use this stock file", key="dm_stock_apply"):
                ss.stock, ss.stock_source = base, "Uploaded file"
                ss.ingest_log.append({"time": state.now(), "event": "Stock upload", "detail": f"{len(base)} SKUs"})
                st.rerun()
with c2:
    st.markdown(f"Current source: **{ss.stock_source}**")
    s = state.settings()
    new_seed = st.number_input("Simulation seed for the stock scenario", 0, 9999, s["stock_seed"], key="dm_seed")
    if st.button("Regenerate simulated stock", key="dm_regen"):
        s["stock_seed"] = int(new_seed)
        ss.stock, ss.stock_source = None, "Simulated scenario"
        st.rerun()

st.markdown("#### Update log")
st.dataframe(pd.DataFrame(ss.ingest_log)[::-1], hide_index=True, width="stretch")
if st.button("Reset data to the initial state", key="dm_reset"):
    for k in ["panel", "cache", "stock", "opt_results", "bo_last", "mc_result", "mc_last_key", "mc_scen"]:
        ss.pop(k, None)
    st.rerun()
