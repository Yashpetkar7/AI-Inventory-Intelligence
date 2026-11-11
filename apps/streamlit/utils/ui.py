"""Visual system: CSS, KPI cards, badges and the Plotly template."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

TEAL, BLUE, SKY, AMBER, ORANGE, RED, SLATE = "#14b8a6", "#3b82f6", "#38bdf8", "#f59e0b", "#f97316", "#ef4444", "#64748b"
INK, MUTED, GRID, PANEL = "#e2e8f0", "#94a3b8", "#1f2a40", "#111a2e"
MODEL_COLORS = {"Naive": "#64748b", "Seasonal naive": "#a78bfa", "Moving average (4w)": "#38bdf8",
                "ETS (damped trend)": "#3b82f6", "ARIMA + Fourier": "#f472b6", "Ridge regression": "#f59e0b",
                "Gradient boosting": "#14b8a6", "AI-selected (global)": "#22d3ee", "AI-selected (per SKU)": "#e879f9"}
IDENT = "Lab Activity 2 · Session 2 ·  Yash Petkar· AS25DXB021"

pio.templates["ops"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=12.5),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[TEAL, BLUE, AMBER, SKY, "#f472b6", "#a78bfa", RED],
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.24, yanchor="top", x=0, font=dict(color=MUTED)),
    margin=dict(l=10, r=10, t=48, b=30), hoverlabel=dict(bgcolor="#0f172a", bordercolor=GRID, font=dict(color=INK)),
    title=dict(font=dict(family="Space Grotesk, Inter, sans-serif", size=15, color=INK), x=0, xanchor="left"),
))
pio.templates.default = "plotly_dark+ops"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label { font-family: 'Inter', 'Segoe UI', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0a1020; border-right: 1px solid #1f2a40; }
[data-testid="stSidebar"] .stMarkdown p { font-size: 0.86rem; }
.page-head { display:flex; flex-direction:column; gap:4px; margin-bottom: 0.6rem; }
.page-head .eyebrow { font: 600 0.72rem 'JetBrains Mono', monospace; color: #14b8a6; letter-spacing: .12em; text-transform: uppercase; }
.page-head h1 { margin: 0; font-size: 1.9rem; line-height: 1.15; }
.page-head p { color: #94a3b8; margin: 2px 0 0; max-width: 78ch; }
.kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 14px; }
.kpi { background: linear-gradient(180deg, #121c31 0%, #0f182b 100%); border: 1px solid #1f2a40; border-radius: 12px; padding: 14px 16px 12px; position: relative; overflow: hidden; }
.kpi .lab { color:#94a3b8; font-size: .76rem; text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }
.kpi .val { font: 600 1.62rem 'Space Grotesk', sans-serif; color: #f1f5f9; margin-top: 4px; font-variant-numeric: tabular-nums; }
.kpi .sub { color:#7c8aa5; font-size: .76rem; margin-top: 2px; }
.kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--c, #14b8a6); }
.badge { display:inline-block; font: 600 .68rem 'JetBrains Mono', monospace; padding: 2px 8px; border-radius: 999px; letter-spacing:.04em; border:1px solid; margin-right:6px; }
.b-obs { color:#5eead4; border-color:#115e59; background:#042f2e; }
.b-der { color:#93c5fd; border-color:#1e3a8a; background:#0b1a3a; }
.b-ext { color:#c4b5fd; border-color:#4c1d95; background:#1e1239; }
.b-asm { color:#fcd34d; border-color:#78350f; background:#2a1a06; }
.b-sim { color:#fda4af; border-color:#881337; background:#2b0a14; }
.note { background:#0f182b; border:1px solid #1f2a40; border-left: 3px solid #3b82f6; border-radius: 8px; padding: 10px 14px; color:#cbd5e1; font-size: .88rem; margin: 6px 0 12px; }
.note.warn { border-left-color: #f59e0b; }
.rec { background:#0f182b; border:1px solid #1f2a40; border-radius:10px; padding: 10px 14px; margin-bottom: 8px; font-size: .9rem; }
.rec b { color:#f1f5f9; }
.ident { font: 600 .74rem 'JetBrains Mono', monospace; color:#14b8a6; white-space: pre-wrap; }
.stamp { font: .74rem 'JetBrains Mono', monospace; color:#7c8aa5; }
div[data-testid="stMetric"] { background:#0f182b; border:1px solid #1f2a40; border-radius:10px; padding:10px 14px; }
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { background:#0f182b; border-radius: 8px 8px 0 0; padding: 6px 14px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_head(eyebrow: str, title: str, text: str = ""):
    st.markdown(f'<div class="page-head"><div class="eyebrow">{html.escape(eyebrow)}</div>'
                f'<h1>{html.escape(title)}</h1>{f"<p>{text}</p>" if text else ""}</div>', unsafe_allow_html=True)


def kpis(items):
    """items: list of (label, value, sub, color)."""
    cards = "".join(f'<div class="kpi" style="--c:{c}"><div class="lab">{html.escape(l)}</div>'
                    f'<div class="val" style="{"font-size:1.15rem;line-height:1.9rem" if len(str(v)) > 10 else ""}">{html.escape(str(v))}</div><div class="sub">{s}</div></div>'
                    for l, v, s, c in items)
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


BADGES = {"observed": ("OBSERVED", "b-obs"), "derived": ("DERIVED", "b-der"), "external": ("EXTERNAL", "b-ext"),
          "assumed": ("ASSUMPTION", "b-asm"), "simulated": ("SIMULATED", "b-sim")}


def badge(kind: str) -> str:
    t, c = BADGES[kind]
    return f'<span class="badge {c}">{t}</span>'


def note(text: str, warn: bool = False):
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def fmt(x, d=0):
    if x is None:
        return "–"
    return f"{x:,.{d}f}"
