"""One-page summary PDF for Lab 2. Recomputes the three charted results with the same engine,
inputs and seeds as the notebook, and asserts they equal the notebook's saved numbers."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from inventory_core import data as D, forecasting as F, inventory as I, simulation as S, optimization as O  # noqa
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

K = json.load(open(ROOT / "nbrun" / "nb_summary.json"))
IDENT = "Lab Activity 2 · Session 2 ·  Yash Petkar· AS25DXB021"
OUT = ROOT / "submissions" / "Laboratory Work 2" / "lab2_summary_yashpetkar_AS25DXB021.pdf"

raw = D.load_raw(ROOT / "data" / "raw" / "onlineretail2.rda")
clean, q = D.clean_transactions(raw)
panel = D.weekly_panel(clean)
out = F.run_pipeline(panel, horizon=8, bt_horizon=4, strategy="global")
best = out["best"]["best_model"].iat[0]
tm = out["test_metrics"].set_index("model")
assert best == K["BEST_MODEL"] and abs(tm.loc["AI-selected (global)", "WAPE"] - K["BEST_TEST_WAPE"]) < 1e-12
LEAD, LEAD_SD, SL = 3.0, 1.0, 0.95
EX = "85123A"
fc = out["forecast"].groupby("sku")["forecast"].mean()
vb = out["val_bt"][out["val_bt"]["model"] == best]
sig = vb.assign(e=vb["forecast"] - vb["actual"]).groupby("sku")["e"].apply(lambda e: np.sqrt((e ** 2).mean()))
d, s = fc[EX], sig[EX]
price = panel[panel["sku"] == EX]["avg_price"].median()
ss = I.safety_stock(I.z_value(SL), s, LEAD, d, LEAD_SD)
rop = d * LEAD + ss
assert abs(ss - K["EX_SS"]) < 1e-9
scen = S.scenario_table(10_000, d, s, LEAD, LEAD_SD, SL, rop, "Gamma", "Gamma", 0.0, seed=42)
sc = O.Scenario(d_mean=d, d_sd=s, l_mean=LEAD, l_sd=LEAD_SD, unit_cost=price * 0.6, holding_rate=0.25, order_cost=60.0,
                shortage_cost=price, target_csl=SL, capacity=round(rop + 4 * d * 4, -2), budget=round(price * 0.6 * d * 9, -1),
                moq=500, q_max=round(26 * d, -2), n_paths=200, seed=42)
res = O.optimise(sc, n_trials=100)
assert abs(res["best_validation"]["total_cost"] - K["OPT_COST"]) < 1e-9

TEAL, BLUE, AMBER, RED, GREY = "#0f9e8e", "#2a6fdb", "#e39400", "#d93a3a", "#6b7280"
plt.rcParams.update({"font.size": 8.6, "axes.titlesize": 9.6, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
comp = out["val_metrics"].set_index("model")[["WAPE"]].rename(columns={"WAPE": "val"}).join(tm[["WAPE"]].rename(columns={"WAPE": "test"}))
comp = comp.sort_values("val")
short = {"Moving average (4w)": "Moving avg (4w)", "ETS (damped trend)": "ETS (damped)", "Gradient boosting": "Gradient boosting",
         "Ridge regression": "Ridge regression", "ARIMA + Fourier": "ARIMA + Fourier", "Seasonal naive": "Seasonal naive", "Naive": "Naive"}
y = np.arange(len(comp))
ax[0].barh(y - 0.2, comp["val"], 0.4, color=BLUE, label="Validation")
ax[0].barh(y + 0.2, comp["test"], 0.4, color=TEAL, label="Test")
ax[0].set_yticks(y, [short[m] for m in comp.index]); ax[0].invert_yaxis()
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax[0].set(title="1. Forecast error by model (WAPE)", xlim=(0, 1.12)); ax[0].legend(fontsize=7.5, loc="upper right")
x = np.arange(len(scen))
ax[1].bar(x - 0.2, scen["SS formula"], 0.4, color=BLUE, label="Formula")
ax[1].bar(x + 0.2, scen["SS Monte Carlo"], 0.4, color=TEAL, label="Monte Carlo")
ax[1].set_xticks(x, ["Normal", "Demand\n+30%", "Supplier\ndelay", "High\nvariability"])
ax[1].set(title=f"2. Safety stock, {EX} (units)"); ax[1].legend(fontsize=7.5)
h = res["history"]
ax[2].scatter(h["trial"], h["cost"], c=np.where(h["feasible"], TEAL, RED), s=9, alpha=0.65)
ax[2].step(h["trial"], h["best_feasible_so_far"], where="post", color=AMBER, lw=2.2, label="best feasible")
ax[2].axhline(res["baseline_eval"]["total_cost"], color=GREY, ls="--", lw=1.2, label="baseline policy")
ax[2].set_ylim(h["cost"].min() * 0.9, np.nanpercentile(h["cost"], 85) * 1.05)
ax[2].set(title="3. Bayesian optimisation, £/yr", xlabel="Trial"); ax[2].legend(fontsize=7.5)
plt.tight_layout(w_pad=1.2)
png = ROOT / "outputs" / "pdf" / "panel.png"
png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(png, dpi=220)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm, topMargin=10 * mm, bottomMargin=9 * mm,
                        title="Lab 2 Summary: AI in Inventory Management", author="Yash Petkar")
W = A4[0] - 26 * mm
ink, muted, blue = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#0f7f73")
St = lambda n, **kw: ParagraphStyle(n, fontName=kw.pop("font", "Helvetica"), textColor=kw.pop("color", ink), **kw)
ident = St("id", font="Helvetica-Bold", fontSize=10.5, color=blue, leading=13)
h1 = St("h1", font="Helvetica-Bold", fontSize=15, leading=18.5, spaceAfter=1)
h2 = St("h2", font="Helvetica-Bold", fontSize=10.8, leading=13.5, spaceBefore=8, spaceAfter=3)
body = St("b", fontSize=9.1, leading=12.4, spaceAfter=2.5)
small = St("s", fontSize=7.9, leading=10.2, color=muted)
story = [
    Paragraph(IDENT.replace("  ", " &nbsp;"), ident),
    Paragraph("AI Applications in Reorder Point &amp; Inventory Management", h1),
    Paragraph("AI in Operations (MAIB AI 218) · Session II Laboratory Exercise · S P Jain School of Global Management, Dubai · Dr. Sandip Kumar Roy", small),
    Spacer(1, 3),
    Paragraph("<b>Aim.</b> Upgrade the reorder-point calculator with (1) machine-learning demand forecasting that updates as new data "
              "arrives, (2) probabilistic and Monte Carlo safety stock, and (3) Bayesian optimisation of the reorder policy under constraints. "
              "Delivered as an executed notebook and an interactive Streamlit dashboard that share one engine.", body),
    Paragraph(f"<b>Data.</b> UCI <i>Online Retail II</i> (Chen, 2019; CC BY 4.0): {K['ROWS_RAW']:,} transaction lines from a UK online gift "
              f"wholesaler, Dec 2009 – Dec 2011, cleaned to {K['ROWS_CLEAN']:,} sales lines; top 24 regular SKUs × {panel['week'].nunique()} weeks. "
              "External factor: UK bank holidays. Lead time (3 ± 1 wk), costs and stock are labelled assumptions, because the data has none. "
              "Copy used: github.com/allanvc/onlineretail2, SHA-256 verified.", body),
    Paragraph("Key results", h2),
]
tv = St("tv", font="Helvetica-Bold", fontSize=17, leading=20)
tl = St("tl", fontSize=7.7, leading=9.6, color=muted)
tiles = [
    (f"{K['BEST_TEST_WAPE']:.1%}", f"<b>Forecast</b>: test WAPE of the auto-selected {K['BEST_MODEL'].lower()} (naive {K['NAIVE_TEST_WAPE']:.1%})"),
    (f"−{K['UPD_IMPROVE']:.0%}", f"<b>Continuous update</b>: error when refitted weekly vs static ({K['STATIC_WAPE']:.1%} → {K['UPDATED_WAPE']:.1%})"),
    (f"{K['N_MORE_BUFFER']}/24", f"<b>Monte Carlo</b>: SKUs needing more buffer than the formula (median {K['MED_EXTRA']:+.0%})"),
    (f"{K['OPT_MEET']}/24", f"<b>Bayesian opt.</b>: SKUs meeting 95% service (baseline {K['BASE_MEET']}/24), cost {K['ALL_CHANGE']:+.1%}"),
]
t = Table([[[Paragraph(v, tv), Paragraph(l, tl)] for v, l in tiles]], colWidths=[W / 4] * 4)
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f6f5")), ("LINEBEFORE", (1, 0), (-1, -1), 3, colors.white),
                       ("LINEABOVE", (0, 0), (-1, 0), 2, blue), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
story += [t, Spacer(1, 4),
          Paragraph("Method: 7 forecasting models (naive, seasonal naive, moving average, ETS, ARIMA + Fourier, ridge, gradient boosting) on a "
                    "chronological train / validation / test split with rolling origins; SS = Z√(σ<sub>D</sub>²L + D<sub>mean</sub>²σ<sub>L</sub>²) with D<sub>mean</sub> = forecast "
                    "mean and σ<sub>D</sub> = forecast RMSE; 10,000-run Monte Carlo (gamma demand and lead time, copula correlation); Optuna TPE over ROP "
                    "and Q with service, capacity, budget and MOQ constraints on common random numbers, validated on fresh paths.", small),
          Paragraph("Charts", h2), Image(str(png), width=W, height=W * 3.9 / 10.4), Paragraph("Insights", h2)]
ins = [
    f"<b>Seasonality is strong and forecastable.</b> November demand runs at {K['NOV_INDEX']:.2f}× an average month. The selected model beats the "
    f"naive forecast on unseen weeks ({K['BEST_TEST_WAPE']:.1%} vs {K['NAIVE_TEST_WAPE']:.1%} WAPE), and refitting each week cuts error by "
    f"{K['UPD_IMPROVE']:.0%} through the peak.",
    f"<b>The textbook formula under-protects skewed demand.</b> For {K['EX']} the formula ROP gives {K['EX_CSL_FORMULA']:.1%} service, not 95%. "
    f"A supplier delay pushes the stockout risk to {K['DELAY_PSTOCKOUT']:.0%} per cycle, and correlated delays (ρ = 0.6) raise the needed buffer "
    f"{K['SS_RHO06'] / K['EX_SS_MC']:.1f}× to {K['SS_RHO06']:,.0f} units.",
    f"<b>Optimising ROP and Q together pays.</b> For {K['EX']}, Bayesian optimisation meets every constraint at {K['OPT_CSL']:.1%} service and "
    f"{K['OPT_CHANGE']:+.1%} cost. Across all 24 SKUs, service compliance rises from {K['BASE_MEET']} to {K['OPT_MEET']} SKUs while cost falls {abs(K['ALL_CHANGE']):.1%}.",
]
for i, s_ in enumerate(ins, 1):
    story.append(Paragraph(f"{i}. {s_}", ParagraphStyle(f"i{i}", parent=body, leftIndent=10, firstLineIndent=-10)))
story += [Paragraph("Recommendation &amp; conclusion", h2),
          Paragraph("Set buffers from simulation rather than the closed-form formula, refresh forecasts weekly before the October–November peak, and "
                    "let the Bayesian optimiser choose reorder point and order size per SKU within the warehouse and budget limits. All three "
                    "exercise steps are implemented on real data. The limits are that lead times, costs and stock are assumed, and that refreshes "
                    "are batch, not real-time.", body),
          Spacer(1, 3),
          Paragraph("Files: lab2_session2_yashpetkar_AS25DXB021.ipynb (executed, Colab-ready) · Streamlit dashboard: apps/streamlit/app.py", small)]
doc.build(story)
print("PDF written:", OUT)
