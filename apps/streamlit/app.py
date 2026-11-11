"""AI Inventory Intelligence: Streamlit dashboard (Lab 2, AI in Operations, S P Jain Dubai).
Run from the project root:  streamlit run apps/streamlit/app.py
"""
import streamlit as st

st.set_page_config(page_title="AI Inventory Intelligence", page_icon="📦", layout="wide",
                   initial_sidebar_state="auto")

from utils import state, ui  # noqa: E402

ui.inject_css()
state.init()

pages = {
    "Command centre": [
        st.Page("pages/1_overview.py", title="Executive overview", icon=":material/dashboard:", default=True),
        st.Page("pages/6_inventory.py", title="Inventory intelligence", icon=":material/warehouse:"),
    ],
    "AI workstreams": [
        st.Page("pages/2_forecasting.py", title="1 · Demand forecasting", icon=":material/insights:"),
        st.Page("pages/3_safety_stock.py", title="2a · Safety stock & ROP", icon=":material/calculate:"),
        st.Page("pages/4_monte_carlo.py", title="2b · Monte Carlo risk lab", icon=":material/casino:"),
        st.Page("pages/5_optimisation.py", title="3 · Bayesian optimisation", icon=":material/tune:"),
    ],
    "Data & reports": [
        st.Page("pages/8_data.py", title="Data management", icon=":material/database:"),
        st.Page("pages/7_reports.py", title="Reports & diagnostics", icon=":material/description:"),
    ],
}
nav = st.navigation(pages)
state.sidebar()
nav.run()
