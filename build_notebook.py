"""Builds the Lab 2 submission notebook. The engine source in src/inventory_core is embedded with
%%writefile cells, so the notebook runs stand-alone in Google Colab and uses exactly the same code
as the Streamlit dashboard. Markdown {KEY} placeholders are filled from nb_summary.json, which the
notebook writes when executed (pass the json path as argv[2])."""
import json, sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).parent
IDENT = "Lab Activity 2 · Session 2 ·  Yash Petkar· AS25DXB021"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(f"""
**{IDENT}**

# AI Applications in Reorder Point & Inventory Management
### AI in Operations (MAIB AI 218): Session II Laboratory Exercise

| | |
|---|---|
| **Student** | Yash Petkar |
| **Student ID** | AS25DXB021 |
| **Programme** | Master of AI in Business (MAIB), September 2025 cohort |
| **Institution** | S P Jain School of Global Management, Dubai |
| **Lecturer** | Dr. Sandip Kumar Roy |
| **Exercise** | Session II: AI Applications in Reorder Point & Inventory Management (lecture slide 35, 11 Nov 2026) |
| **Companion app** | Streamlit dashboard *AI Inventory Intelligence* (same engine code as this notebook) |

*Individual work. In line with the slide's rule "No sharing of data with peers allowed", the dataset was sourced and processed independently, and nothing in this notebook uploads data anywhere.*
""")
md("""
## 1. The exercise (reconstructed from slide 35)

**Objective:** *"Enhance the reorder-point calculator and sensitivity analysis by introducing predictive intelligence, adaptive decision-making, and real-time optimization."*

| # | Exercise step (slide wording) | Where it is answered |
|---|---|---|
| 1 | **Demand forecasting:** use machine-learning models (e.g. time series, regression, LSTM) to predict future demand based on historical data, seasonality, promotions and external factors. Keep the possibility of adding new data. Continuously update forecasts as new data arrives, improving accuracy and responsiveness. | Sections 5–6 |
| 2 | **Dynamic safety stock:** estimate safety stock using *probabilistic models* that account for demand variability and lead-time uncertainty, and *Monte Carlo simulations* to model risk scenarios and buffer requirements. Formula: $SS = Z\\sqrt{\\sigma_D^2 L + \\bar D^2 \\sigma_L^2}$ | Sections 7–8 |
| 3 | **AI-driven sensitivity analysis:** instead of manual parameter sweeps, use **Bayesian optimization** to identify optimal reorder points under constraints. | Section 9 |

**Units.** Everything is weekly: demand per week, lead time in weeks, so the formula inputs share one time unit. The reorder point is $ROP = \\bar D L + SS$.
""")

md("""
## 2. Setup
Installs the few packages Colab does not ship with, fixes random seeds, and records versions.
""")
code("""
import importlib.util, subprocess, sys
need = {"pyreadr": "pyreadr", "holidays": "holidays", "optuna": "optuna", "statsmodels": "statsmodels"}
missing = [p for m, p in need.items() if importlib.util.find_spec(m) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
print("Installed now:", missing or "nothing (all present)")
""")
code("""
import os, time, json, hashlib, random, warnings, platform, urllib.request
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy, sklearn, statsmodels, optuna, holidays
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
pd.set_option("display.width", 200, "display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10})
TEAL, BLUE, AMBER, RED, GREY = "#0f9e8e", "#2a6fdb", "#e39400", "#d93a3a", "#6b7280"
T0 = time.time()
print(f"Python {platform.python_version()} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} | "
      f"statsmodels {statsmodels.__version__} | optuna {optuna.__version__} | scipy {scipy.__version__}")
""")

md("""
## 3. The shared engine (`inventory_core`)
The cells below write the same Python package the Streamlit dashboard imports. Keeping one copy means the notebook and the app compute identical safety stocks, reorder points, forecasts and optimisation results.
""")
for name in ["__init__", "data", "forecasting", "inventory", "simulation", "optimization"]:
    src = (ROOT / "src" / "inventory_core" / f"{name}.py").read_text()
    code(f"%%writefile inventory_core/{name}.py\n{src}" if name != "__init__" else
         f"import os; os.makedirs('inventory_core', exist_ok=True)")
    if name == "__init__":
        code(f"%%writefile inventory_core/__init__.py\n{src}")
code("""
import importlib, inventory_core
for m in ["data", "forecasting", "inventory", "simulation", "optimization"]:
    importlib.reload(importlib.import_module(f"inventory_core.{m}"))
from inventory_core import data as D, forecasting as F, inventory as I, simulation as S, optimization as O
print("inventory_core", inventory_core.__version__, "loaded")
""")

md("""
## 4. Dataset research, download and cleaning

No dataset was provided, so I compared the candidate public sources first:

| Candidate | Size & coverage | Demand / price / promotions | Access & licence | Verdict |
|---|---|---|---|---|
| Walmart **M5** (Kaggle) | 30,490 item-store series, daily, 2011–2016 | Daily units, weekly prices, SNAP & event calendar | Kaggle account required; competition rules restrict redistribution | Not usable without login |
| **UCI Online Retail II** | 1,067,371 transaction lines, 1 Dec 2009 – 9 Dec 2011, UK online gift wholesaler | Units, unit price, dates and country; **no** promotions, stock or lead times | Direct download; **CC BY 4.0** | **Selected:** two Christmas seasons, real prices, open licence |
| UCI Online Retail (I) | 541,909 lines, Dec 2010 – Dec 2011 | Same fields | CC BY 4.0 | Only one year, too short to learn yearly seasonality |

**What the data does not contain:** stock on hand, purchase orders, supplier lead times and warehouse capacity. These are therefore **configurable scenario assumptions** and are labelled as such everywhere (lead time 3 ± 1 weeks; unit cost = 60 % of selling price; holding cost 25 %/year; £60 per order; shortage cost = selling price per unit short).

**Source.** Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository, https://doi.org/10.24432/C5CG6D, CC BY 4.0. The UCI site does not allow scripted downloads from every network, so the notebook uses the identical copy published in the CC0 R package *onlineretail2* (Quadros, 2020) on GitHub and checks its SHA-256 hash.
""")
code("""
DATA_DIR = "data"; os.makedirs(DATA_DIR, exist_ok=True)
path = os.path.join(DATA_DIR, "onlineretail2.rda")
if not os.path.exists(path):
    urllib.request.urlretrieve(D.DATA_URL, path)
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
assert digest == D.DATA_SHA256, "Downloaded file does not match the documented hash"
raw = D.load_raw(path)
print(f"Source: {D.DATA_URL}\\nSHA-256 verified: {digest[:16]}…  |  rows: {len(raw):,}  |  columns: {raw.shape[1]}")
raw.head()
""")
code("""
raw.info()
""")
code("""
raw.describe().T.round(2)
""")
code("""
clean, quality = D.clean_transactions(raw)
q = pd.Series({k: v for k, v in quality.items() if k != "missing_by_column"})
print("Data-quality report:"); print(q.to_string())
print("\\nMissing values in the raw file:"); print(pd.Series(quality["missing_by_column"]).to_string())
""")
md("""
**Interpretation.** Of {ROWS_RAW:,} raw lines, {CANCEL:,} are cancellations (invoice numbers starting with "C"), {NONPOS:,} have zero or negative quantity or price, {NONPROD:,} are non-product codes (postage, manual adjustments, bank charges), and {DUPS:,} are exact duplicates. That leaves **{ROWS_CLEAN:,} genuine sales lines** for {SKUS_CLEAN:,} products. Missing Customer IDs ({MISSING_CUST:,} rows) do not matter for demand, because demand counts units sold, not who bought them.
""")
code("""
panel = D.weekly_panel(clean, n_skus=24, min_active_share=0.9)
print(f"Weekly panel: {panel['sku'].nunique()} SKUs × {panel['week'].nunique()} weeks "
      f"({panel['week'].min():%d %b %Y} – {panel['week'].max():%d %b %Y}) = {len(panel):,} rows")
print("Zero-demand weeks:", f"{(panel['demand'] == 0).mean():.1%}", "| derived price-cut weeks:", f"{panel['price_cut'].mean():.1%}",
      "| weeks with a UK bank holiday:", f"{(panel.drop_duplicates('week')['holidays'] > 0).mean():.1%}")
panel.groupby(["sku", "description"])["demand"].agg(["mean", "std", "sum"]).round(0).sort_values("sum", ascending=False).head(10)
""")
md("""
**Data lineage**

| Kind | Fields |
|---|---|
| Observed | Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country |
| Derived | Weekly demand per SKU (Monday–Sunday weeks; first and last partial weeks dropped), weekly average price, price-cut flag (weekly price more than 10 % below the SKU's median, a *proxy* because promotions are not recorded) |
| External | Number of England & Wales bank holidays in each week (`holidays` package) |
| Assumed | Lead time and its SD, costs, service target, capacity, budget, MOQ |
| Simulated | Stock on hand (Section 10) and all Monte Carlo paths |

The 24 SKUs are the highest-volume products that sold in at least 90 % of weeks. Very intermittent items need different methods, such as Croston's.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.7, 1]})
tot = panel.groupby("week")["demand"].sum()
axes[0].plot(tot.index, tot.values, color=BLUE, lw=1.6)
axes[0].plot(tot.index, tot.rolling(8, center=True).mean(), color=AMBER, lw=2.2, label="8-week average")
axes[0].set(title="Total weekly demand, 24 SKUs", ylabel="Units per week"); axes[0].legend()
p = panel.assign(month=panel["week"].dt.month)
idx = (p.groupby(["sku", "month"])["demand"].mean() / p.groupby("sku")["demand"].mean()).groupby("month").mean()
axes[1].bar(pd.to_datetime(idx.index, format="%m").strftime("%b"), idx.values,
            color=[AMBER if v > 1.2 else TEAL for v in idx.values])
axes[1].axhline(1, ls=":", color=GREY); axes[1].set(title="Seasonality: demand index by month", ylabel="Index (1 = average)")
plt.tight_layout(); plt.show()
print("Monthly index:", idx.round(2).to_dict())
""")
md("""
**Interpretation.** Demand is strongly seasonal. November runs at {NOV_INDEX:.2f}× an average month and October at {OCT_INDEX:.2f}×, as shops stock up for Christmas. December falls to {DEC_INDEX:.2f}× because the business closes over the holidays. A forecasting model therefore needs a yearly-season signal; the Fourier terms and the same-week-last-year lag provide it.
""")

md("""
## 5. Workstream 1: demand forecasting

**Method.**
- **Seven models:** naive; seasonal naive (same week last year); 4-week moving average; ETS with damped trend; ARIMA(1,0,1) with Fourier yearly terms and bank holidays (dynamic harmonic regression); and two ML regressions (ridge and gradient boosting).
- **The two ML regressions** are *global* models trained across all 24 SKUs at once. Their features are recent level, short trend, same week last year, price vs median price, the price-cut flag, UK bank holidays and Fourier seasonality. They forecast horizons 1–8 weeks directly.
- **Chronological split with rolling origins** (4-week horizon): validation origins cover {VAL_START} – {VAL_END} and test origins cover {TEST_START} – {TEST_END}. A forecast made at an origin uses only data up to that week.
- **Automatic model selection:** the model with the lowest validation WAPE is chosen. It is scored once on the untouched test weeks.
- **Why no LSTM:** each SKU has only about 100 weekly observations, too few for a recurrent network to beat simpler models reliably. Pooling across SKUs in gradient boosting is the ML approach this much data supports.
""")
code("""
t = time.time()
out = F.run_pipeline(panel, horizon=8, bt_horizon=4, strategy="global")
print(f"Pipeline run time: {time.time() - t:.0f} s")
sp = out["splits"]
print(f"Training up to {sp['train_end']:%d %b %Y} | validation {sp['val_weeks'][0]:%d %b %Y} – {sp['val_weeks'][1]:%d %b %Y} | "
      f"test {sp['test_weeks'][0]:%d %b %Y} – {sp['test_weeks'][1]:%d %b %Y}")
comp = out["val_metrics"].set_index("model")[["MAE", "RMSE", "WAPE"]].add_prefix("val_").join(
       out["test_metrics"].set_index("model")[["MAE", "RMSE", "WAPE", "MAPE", "Bias"]].add_prefix("test_"), how="outer")
comp.sort_values("val_WAPE").round(3)
""")
code("""
best_model = out["best"]["best_model"].iat[0]
tm = out["test_metrics"].set_index("model")
print("Selected model (lowest validation WAPE):", best_model)
print(f"Test WAPE of the selected model: {tm.loc['AI-selected (global)', 'WAPE']:.1%} | naive: {tm.loc['Naive', 'WAPE']:.1%} | "
      f"seasonal naive: {tm.loc['Seasonal naive', 'WAPE']:.1%}")
print(f"Best test WAPE of any model: {tm['WAPE'].idxmin()} ({tm['WAPE'].min():.1%}); per-SKU selection: {tm.loc['AI-selected (per SKU)', 'WAPE']:.1%}")
d = comp.sort_values("val_WAPE")
fig, ax = plt.subplots(figsize=(10, 4))
y = np.arange(len(d))
ax.barh(y - 0.2, d["val_WAPE"], 0.4, color=BLUE, label="Validation")
ax.barh(y + 0.2, d["test_WAPE"], 0.4, color=TEAL, label="Test")
ax.set_yticks(y, d.index); ax.invert_yaxis(); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set(title="Forecast error by model (WAPE, lower is better)", xlabel="WAPE = Σ|error| / Σ actual"); ax.legend()
plt.tight_layout(); plt.show()
""")
md("""
**Interpretation.** On the validation weeks, **{BEST_MODEL}** has the lowest WAPE ({BEST_VAL_WAPE:.1%}), so it is selected automatically. On the unseen test weeks it scores {BEST_TEST_WAPE:.1%}, against {NAIVE_TEST_WAPE:.1%} for the naive forecast and {SNAIVE_TEST_WAPE:.1%} for the seasonal naive. Errors remain large in absolute terms because B2B gift orders are lumpy: one wholesale order can be several weeks' typical demand.

Two honest observations:
1. On the test weeks the best single model is {BEST_TEST_ANY} ({BEST_TEST_ANY_WAPE:.1%}). Rankings shift between windows, so the selection uses only validation data and is reported without hindsight.
2. Picking a different model for each SKU does worse on test ({PER_SKU_WAPE:.1%}). With 12 validation weeks per SKU, per-SKU choices mostly fit noise, so one global choice is more robust.

MAPE is shown for reference only: it is undefined in weeks with zero demand and over-weights small weeks.
""")
code("""
EX = "85123A"                      # White Hanging Heart T-Light Holder: the best-known product in the dataset
hist = panel[panel["sku"] == EX].set_index("week")["demand"]
fc = out["forecast"][out["forecast"]["sku"] == EX]
tb = out["test_bt"][(out["test_bt"]["sku"] == EX)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
h = hist[hist.index >= "2011-01-01"]
axes[0].plot(h.index, h.values, color=BLUE, lw=1.8, label="Actual")
axes[0].fill_between(fc["week"], fc["lo"], fc["hi"], color=TEAL, alpha=0.2, label="90% prediction interval")
axes[0].plot(fc["week"], fc["forecast"], color=TEAL, lw=2.4, marker="o", ms=4, label=f"Forecast ({best_model})")
axes[0].set(title=f"{EX}: weekly demand and 8-week forecast", ylabel="Units per week"); axes[0].legend(fontsize=8)
act = tb.drop_duplicates("week").sort_values("week")
axes[1].plot(act["week"], act["actual"], color="black", lw=2.2, label="Actual")
for m, c in [(best_model, TEAL), ("ETS (damped trend)", BLUE), ("Naive", GREY)]:
    dm = tb[tb["model"] == m].sort_values("week")
    axes[1].plot(dm["week"], dm["forecast"], marker="o", ms=3, color=c, label=m)
axes[1].set(title="Test weeks: actual vs predicted"); axes[1].tick_params(axis="x", rotation=30); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(fc[["week", "forecast", "lo", "hi"]].round(0).to_string(index=False))
""")
md("""
**Prediction intervals.** The 90 % band for each horizon comes from the selected model's own validation errors at that horizon: ±1.645 × RMSE(h). It is therefore an honest measure of how wrong the model has actually been, not a theoretical figure.
""")
code("""
from sklearn.inspection import permutation_importance
W = F.wide_demand(panel)
P = panel.pivot(index="week", columns="sku", values="avg_price"); C = panel.pivot(index="week", columns="sku", values="price_cut")
train = F._ml_train_set(panel, len(W) - 1, W.index, W, P, C)
gbm = F._fit_ml("Gradient boosting", train)
smp = train.sample(3000, random_state=SEED)
pi = permutation_importance(gbm, smp[F.ML_FEATURES], smp["y_ratio"], n_repeats=5, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=F.ML_FEATURES).sort_values()
names = {"lvl4_ratio": "recent level (4 wk)", "trend": "short-term trend", "lag52_ratio": "same week last year",
         "price_ratio": "price vs median", "price_cut": "price-cut flag (derived)", "hol": "UK bank holidays", "h": "horizon"}
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh([names.get(i, i) for i in imp.index], imp.values, color=[TEAL if v > 0 else GREY for v in imp.values])
ax.set(title="What drives the ML forecast (permutation importance, gradient boosting)", xlabel="Increase in error when shuffled")
plt.tight_layout(); plt.show()
print(imp.sort_values(ascending=False).round(4).to_string())
""")
md("""
**Interpretation.** The strongest driver is **{TOP_FEATURE}**. Seasonality enters through the Fourier terms and the same-week-last-year lag, price through the price ratio, and external factors through the bank-holiday count. So the model uses every factor the slide lists that the data can supply. Promotions are not recorded in this dataset, so the derived price-cut flag stands in as a proxy, and it should be read with care: it may also capture bulk-price tiers.
""")

md("""
## 6. Adding new data and updating forecasts continuously

The slide asks to *"keep the possibility of adding new data"* and to *"continuously update forecasts as new data arrives"*. This is shown in two ways:
1. **New-data intake.** A batch of raw transactions goes through `validate_new_data` (schema mapping, date and number checks, duplicate removal) and is merged with `append_weekly`, which appends new weeks and reconciles overlapping ones. The dashboard's *Data management* page does the same with uploaded CSVs and a replay feed.
2. **Updating vs static model.** Starting from data up to {CUT_WEEK}, the 8 later weeks arrive one at a time. An *updating* model is refitted after every arrival and forecasts the next week. A *static* model is fitted once at the cut-off and is never refreshed.
""")
code("""
CUT = pd.Timestamp("2011-10-03")
base = panel[panel["week"] <= CUT]
late = clean[clean["InvoiceDate"] >= CUT + pd.Timedelta(days=7)]
first_week = late[late["InvoiceDate"] < CUT + pd.Timedelta(days=14)].rename(columns={"StockCode": "sku"})
v = D.validate_new_data(first_week, D.detect_columns(first_week))
print("Validation OK:", v.ok, "| warnings:", v.warnings or "none", "| weekly rows produced:", len(v.data))
updated, info = D.append_weekly(base, v.data)
print("Append result:", info)
print(f"Panel grew from {base['week'].nunique()} to {updated['week'].nunique()} weeks; last week now {updated['week'].max():%d %b %Y}")
""")
code("""
weeks = pd.DatetimeIndex(sorted(panel["week"].unique()))
cut_i = int(np.where(weeks == CUT)[0][0])
upd_origins = list(range(cut_i, len(weeks) - 1))                       # refit after every new week, forecast 1 week ahead
upd = F.backtest(panel, upd_origins, horizon=1, models=[best_model]).assign(mode="Updated weekly")
static = F.backtest(panel, [cut_i], horizon=len(weeks) - 1 - cut_i, models=[best_model]).assign(mode="Static (fit once)")
both = pd.concat([upd, static])
cmp_ = F.metrics(both, by=("mode",)).set_index("mode")[["MAE", "RMSE", "WAPE"]]
print(cmp_.round(3).to_string())
wk = both.groupby(["mode", "week"]).apply(lambda g: (g["forecast"] - g["actual"]).abs().sum() / g["actual"].sum(), include_groups=False).unstack(0)
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(wk.index, wk["Static (fit once)"], marker="o", color=GREY, label="Static (fit once at cut-off)")
ax.plot(wk.index, wk["Updated weekly"], marker="o", color=TEAL, lw=2.4, label="Updated as each week arrives")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set(title="Weekly forecast error (WAPE, 24 SKUs) during the Oct–Nov peak", ylabel="WAPE"); ax.legend()
plt.tight_layout(); plt.show()
""")
md("""
**Interpretation.** Through the October–November peak, refreshing the model with each new week cuts the pooled WAPE from {STATIC_WAPE:.1%} (static) to **{UPDATED_WAPE:.1%}** (updated). That is {UPD_IMPROVE:.0%} less error, and it is the practical meaning of "continuously update forecasts as new data arrives". The refresh is an on-demand batch step, not a real-time stream; the dashboard labels it the same way.
""")

md("""
## 7. Workstream 2a: dynamic safety stock and reorder point (probabilistic model)

**Method.** For each SKU:
- **D̄** = mean of the AI forecast over the next 8 weeks (forward-looking, so it moves with the season).
- **σ_D** = RMSE of the selected model's forecast errors on validation. This is the uncertainty that remains *after* forecasting, the right input for a forecast-driven buffer. The historical SD is shown for comparison.
- **Z** from the 95 % cycle service level; L = 3 weeks and σ_L = 1 week (**assumptions**).

Then $SS = Z\\sqrt{\\sigma_D^2 L + \\bar D^2 \\sigma_L^2}$ and $ROP = \\bar D L + SS$.
""")
code("""
LEAD, LEAD_SD, SL = 3.0, 1.0, 0.95
COST_RATIO, HOLD, ORDER_COST, SHORT_MULT = 0.60, 0.25, 60.0, 1.0
fcm = out["forecast"].groupby("sku")["forecast"].mean().rename("d_bar")
vb = out["val_bt"][out["val_bt"]["model"] == best_model]
rmse = vb.assign(e=vb["forecast"] - vb["actual"]).groupby("sku")["e"].apply(lambda e: np.sqrt((e ** 2).mean())).rename("sigma_d")
last26 = panel[panel["week"] > panel["week"].max() - pd.Timedelta(weeks=26)].groupby("sku")["demand"].std().rename("hist_sd")
price = panel.groupby("sku")["avg_price"].median().rename("price")
desc = panel.groupby("sku")["description"].first()
params = pd.concat([desc, fcm, rmse, last26, price], axis=1).reset_index().rename(columns={"index": "sku"})
params["unit_cost"] = params["price"] * COST_RATIO
z = I.z_value(SL)
params["SS_forecast_err"] = [I.safety_stock(z, s, LEAD, d, LEAD_SD) for d, s in zip(params["d_bar"], params["sigma_d"])]
params["SS_hist_sd"] = [I.safety_stock(z, s, LEAD, d, LEAD_SD) for d, s in zip(params["d_bar"], params["hist_sd"])]
params["ROP"] = params["d_bar"] * LEAD + params["SS_forecast_err"]
print(f"Z for {SL:.0%} = {z:.4f}")
print(f"Median σ_D: forecast-error {params['sigma_d'].median():,.0f} vs historical SD {params['hist_sd'].median():,.0f} units/week")
params[["sku", "description", "d_bar", "sigma_d", "hist_sd", "SS_forecast_err", "SS_hist_sd", "ROP"]].round(0).sort_values("ROP", ascending=False)
""")
code("""
r = params.set_index("sku").loc[EX]
dv, lv = r.sigma_d ** 2 * LEAD, r.d_bar ** 2 * LEAD_SD ** 2
print(f"Hand check for {EX} ({r.description}):")
print(f"  SS = {z:.3f} × sqrt({r.sigma_d:,.1f}² × {LEAD:g} + {r.d_bar:,.1f}² × {LEAD_SD:g}²) = {z:.3f} × {np.sqrt(dv + lv):,.1f} = {r.SS_forecast_err:,.1f}")
print(f"  ROP = {r.d_bar:,.1f} × {LEAD:g} + {r.SS_forecast_err:,.1f} = {r.ROP:,.1f}")
print(f"  Share of lead-time variance from demand: {dv / (dv + lv):.0%}, from lead time: {lv / (dv + lv):.0%}")
""")
md("""
**Interpretation.** For {EX_NAME}, the formula gives a safety stock of **{EX_SS:,.0f} units** and a reorder point of **{EX_ROP:,.0f} units**. Of the lead-time variance, {EX_DEM_SHARE:.0%} comes from demand and {EX_LT_SHARE:.0%} from the supplier's lead time. Using forecast error instead of raw historical variability changes the buffer: the median σ_D is {MED_SIGMA_FC:,.0f} vs {MED_SIGMA_HIST:,.0f} units/week. Because demand is forecast forward, the buffer is *dynamic*: it rises into the peak season and falls after it.
""")

md("""
## 8. Workstream 2b: Monte Carlo simulation of risk scenarios and buffers

**Why simulate?** The formula assumes that weekly demands are independent and normal, and that lead time is independent of demand. Real gift demand is right-skewed, and supplier delays often coincide with peak demand. The simulation drops these assumptions:
- Each run draws a lead time (gamma, mean 3, SD 1 week) and the weekly demands during it (gamma, matching D̄ and σ_D).
- It sums the lead-time demand (DLT), and repeats this 10,000 times with seed 42.
- The buffer is the 95th percentile of DLT minus its expected value.
- A Gaussian copula sets the demand–lead-time correlation ρ; ρ = 0 reproduces the formula's independence.
""")
code("""
r = params.set_index("sku").loc[EX]
sim = S.lead_time_demand(10_000, r.d_bar, r.sigma_d, LEAD, LEAD_SD, "Gamma", "Gamma", rho=0.0, seed=SEED)
mc = S.summarise_dlt(sim["dlt"], SL, rop=r.ROP, q=1000, d_mean=r.d_bar, l_mean=LEAD)
print(f"{EX}: formula SS {r.SS_forecast_err:,.0f} | Monte Carlo SS {mc['ss_mc']:,.0f} | MC ROP {mc['rop_mc']:,.0f}")
print(f"Simulated cycle service level if the formula ROP is used: {mc['cycle_service_level']:.1%} (target {SL:.0%})")
print(f"Expected shortage per cycle at formula ROP: {mc['expected_shortage']:,.1f} units | P99 lead-time demand: {mc['p99']:,.0f}")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(sim["dlt"], bins=80, color=BLUE, alpha=0.85, density=True)
for x, lab, c in [(r.d_bar * LEAD, "mean D̄·L", GREY), (r.ROP, "formula ROP", AMBER), (mc["rop_mc"], "Monte Carlo ROP", TEAL)]:
    axes[0].axvline(x, color=c, lw=2, ls="--" if c != TEAL else "-", label=f"{lab}: {x:,.0f}")
axes[0].set(title=f"{EX}: simulated demand during lead time (10,000 cycles)", xlabel="Units", ylabel="Density"); axes[0].legend(fontsize=8)
xs = np.sort(sim["dlt"]); axes[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=TEAL, lw=2.4)
axes[1].axhline(SL, color=AMBER, ls=":"); axes[1].axvline(mc["rop_mc"], color=AMBER, ls=":")
axes[1].set(title="Cumulative probability of lead-time demand", xlabel="Units", ylabel="P(DLT ≤ x)")
plt.tight_layout(); plt.show()
""")
code("""
scen = S.scenario_table(10_000, r.d_bar, r.sigma_d, LEAD, LEAD_SD, SL, r.ROP, "Gamma", "Gamma", 0.0, seed=SEED)
rhos = [0.0, 0.3, 0.6]
rho_ss = [S.summarise_dlt(S.lead_time_demand(10_000, r.d_bar, r.sigma_d, LEAD, LEAD_SD, rho=k, seed=SEED)["dlt"], SL,
                          d_mean=r.d_bar, l_mean=LEAD)["ss_mc"] for k in rhos]
fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(len(scen))
axes[0].bar(x - 0.2, scen["SS formula"], 0.4, color=BLUE, label="Formula")
axes[0].bar(x + 0.2, scen["SS Monte Carlo"], 0.4, color=TEAL, label="Monte Carlo")
axes[0].set_xticks(x, [s.replace(" (", "\\n(") for s in scen["scenario"]], fontsize=8)
axes[0].set(title="Safety stock needed under each risk scenario", ylabel="Units"); axes[0].legend()
axes[1].bar([f"ρ = {k}" for k in rhos], rho_ss, color=[TEAL, AMBER, RED])
axes[1].axhline(r.SS_forecast_err, color=BLUE, ls="--", label="formula (assumes ρ = 0)")
axes[1].set(title="Joint simulation: demand–lead-time correlation", ylabel="Monte Carlo SS (units)"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(scen.round(3).to_string(index=False))
print("SS by correlation:", dict(zip(rhos, np.round(rho_ss, 0))))
""")
code("""
rows = []
for rr in params.itertuples():
    s_ = S.lead_time_demand(10_000, rr.d_bar, rr.sigma_d, LEAD, LEAD_SD, seed=SEED)
    m_ = S.summarise_dlt(s_["dlt"], SL, rop=rr.ROP, d_mean=rr.d_bar, l_mean=LEAD)
    rows.append({"sku": rr.sku, "SS formula": rr.SS_forecast_err, "SS Monte Carlo": m_["ss_mc"], "CSL of formula ROP": m_["cycle_service_level"]})
allmc = pd.DataFrame(rows)
allmc["diff %"] = allmc["SS Monte Carlo"] / allmc["SS formula"] - 1
print(f"SKUs where the simulation needs more buffer than the formula: {(allmc['diff %'] > 0).sum()} of {len(allmc)}")
print(f"Median extra buffer: {allmc['diff %'].median():+.1%} | median simulated CSL of the formula ROP: {allmc['CSL of formula ROP'].median():.1%}")
allmc.round(3).sort_values("diff %", ascending=False).head(8)
""")
md("""
**Interpretation.**
- **Skewed demand.** For {EX}, the simulation needs **{EX_SS_MC:,.0f}** units of buffer against **{EX_SS:,.0f}** from the formula. If the formula ROP is kept, the true cycle service level is only **{EX_CSL_FORMULA:.1%}**, not 95 %. Across all SKUs, {N_MORE_BUFFER} of 24 need more buffer than the formula suggests (median {MED_EXTRA:+.1%}).
- **Risk scenarios.** Keeping today's ROP, a supplier delay (+1 week, double variability) raises the stockout probability to {DELAY_PSTOCKOUT:.0%} per cycle, and a +30 % demand spike raises it to {SPIKE_PSTOCKOUT:.0%}.
- **Dependence.** If supplier delays coincide with high demand (ρ = 0.6), the needed buffer rises to {SS_RHO06:,.0f} units. Here the independence assumption behind the formula is clearly broken, and only the joint simulation gives the right answer.
- **Service measures.** These figures are *cycle service levels* (no stockout in a replenishment cycle). They are not fill rates, which measure the share of units served.
""")

md("""
## 9. Workstream 3: AI-driven sensitivity analysis with Bayesian optimisation

Instead of sweeping ROP and Q by hand, **Bayesian optimisation** (Optuna's TPE sampler, a sequential model-based optimiser) searches both at once.

**Decision variables:** reorder point *s* and order quantity *Q* under an (s, nQ) policy with backorders. The safety stock follows as s − D̄L.

**Objective:** minimise expected annual cost = holding (unit cost × 25 %/yr × average stock) + ordering (£60 × orders) + shortage (selling price × units short).

**Constraints** (Optuna treats a trial as feasible when every constraint is ≤ 0):
- cycle service level ≥ 95 %
- 99th-percentile peak stock ≤ warehouse capacity
- average stock value ≤ budget
- MOQ ≤ Q ≤ maximum supplier order

**Fair comparison:** every trial is scored on the same 200 simulated years of demand and lead times (common random numbers, seed 42). The winner and the baseline (formula ROP + EOQ) are then re-scored on 400 fresh years (seed 1042).
""")
code("""
r = params.set_index("sku").loc[EX]
sc = O.Scenario(d_mean=r.d_bar, d_sd=r.sigma_d, l_mean=LEAD, l_sd=LEAD_SD, unit_cost=r.unit_cost, holding_rate=HOLD,
                order_cost=ORDER_COST, shortage_cost=r.price * SHORT_MULT, target_csl=SL,
                capacity=round(r.ROP + 4 * r.d_bar * 4, -2), budget=round(r.unit_cost * r.d_bar * 9, -1),
                moq=500, q_max=round(26 * r.d_bar, -2), n_paths=200, seed=SEED)
t = time.time(); res = O.optimise(sc, n_trials=100, sampler="TPE"); bo_time = time.time() - t
b, bv, base, bval = res["best"], res["best_validation"], res["baseline"], res["baseline_validation"]
print(f"Constraints: CSL ≥ {sc.target_csl:.0%}, capacity {sc.capacity:,.0f} units, budget £{sc.budget:,.0f}, MOQ {sc.moq:,.0f}, max order {sc.q_max:,.0f}")
print(f"100 trials in {bo_time:.1f} s; feasible trials: {res['history']['feasible'].sum()}")
tbl = pd.DataFrame({"Baseline (formula ROP + EOQ)": bval, "Bayesian-optimised": bv}).loc[
    ["rop", "q", "ss", "holding_cost", "ordering_cost", "shortage_cost", "total_cost", "cycle_service_level", "fill_rate",
     "p99_max_on_hand", "avg_on_hand", "feasible"]]
print("\\nValidation on 400 fresh simulated years:"); print(tbl.to_string())
saving = 1 - bv["total_cost"] / bval["total_cost"]
print(f"\\nCost change vs baseline: {-saving:+.1%}")
""")
code("""
h = res["history"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
axes[0].scatter(h["trial"], h["cost"], c=np.where(h["feasible"], TEAL, RED), s=18, alpha=0.7)
axes[0].step(h["trial"], h["best_feasible_so_far"], where="post", color=AMBER, lw=2.4, label="best feasible so far")
axes[0].set_ylim(h["cost"].min() * 0.9, np.nanpercentile(h["cost"], 90) * 1.1)
axes[0].set(title="Convergence (teal = feasible, red = violates a constraint)", xlabel="Trial", ylabel="Expected annual cost (£)"); axes[0].legend()
sca = axes[1].scatter(h["rop"], h["q"], c=h["cost"].clip(upper=np.nanpercentile(h["cost"], 90)), cmap="viridis_r", s=24,
                      marker="o", alpha=0.85)
axes[1].scatter(h.loc[~h["feasible"], "rop"], h.loc[~h["feasible"], "q"], marker="x", color=RED, s=20, label="infeasible")
axes[1].scatter([base["rop"]], [base["q"]], marker="D", s=120, color=AMBER, edgecolor="black", label="baseline")
axes[1].scatter([b["rop"]], [b["q"]], marker="*", s=300, color="white", edgecolor="black", label="optimised")
plt.colorbar(sca, ax=axes[1], label="£/yr")
axes[1].set(title="Where the optimiser searched", xlabel="Reorder point s", ylabel="Order quantity Q"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")
md("""
**Interpretation.**
- **Baseline.** For {EX}, the formula ROP ({BASE_ROP:,.0f}) with Q = EOQ ({BASE_Q:,.0f}) reaches {BASE_CSL:.1%} cycle service on fresh simulated demand ({BASE_VERDICT} the 95 % target). Its expected cost is £{BASE_COST:,.0f}/yr, of which £{BASE_SHORT:,.0f} is shortage cost.
- **Optimised.** The optimiser finds ROP **{OPT_ROP:,.0f}** and Q **{OPT_Q:,.0f}**. That policy meets every constraint ({OPT_CSL:.1%} CSL, within capacity and budget) at £{OPT_COST:,.0f}/yr, a {OPT_CHANGE:+.1%} change. It deliberately holds more stock: holding cost rises from £{BASE_HOLD:,.0f} to £{OPT_HOLD:,.0f}. In return, shortage cost falls from £{BASE_SHORT:,.0f} to £{OPT_SHORT:,.0f}, and fewer, larger orders cut ordering cost from £{BASE_ORD:,.0f} to £{OPT_ORD:,.0f}. That trade-off is exactly what a manual one-parameter sweep misses.
- **Search.** The convergence chart shows the search settling within a few dozen trials. The scatter shows it concentrating around the low-cost, feasible region rather than sweeping a grid.
""")
code("""
summary_rows = []
t = time.time()
for rr in params.itertuples():
    sc_ = O.Scenario(d_mean=rr.d_bar, d_sd=rr.sigma_d, l_mean=LEAD, l_sd=LEAD_SD, unit_cost=rr.unit_cost, holding_rate=HOLD,
                     order_cost=ORDER_COST, shortage_cost=rr.price * SHORT_MULT, target_csl=SL, n_paths=120, seed=SEED)
    rs = O.optimise(sc_, n_trials=60)
    summary_rows.append({"sku": rr.sku, "baseline_cost": rs["baseline_validation"]["total_cost"],
                         "baseline_csl": rs["baseline_validation"]["cycle_service_level"],
                         "opt_cost": rs["best_validation"]["total_cost"], "opt_csl": rs["best_validation"]["cycle_service_level"],
                         "opt_rop": rs["best"]["rop"], "formula_rop": rs["baseline"]["rop"]})
allbo = pd.DataFrame(summary_rows)
print(f"Optimised 24 SKUs in {time.time() - t:.0f} s")
print(f"Baseline policies meeting the 95% CSL target on validation: {(allbo['baseline_csl'] >= SL).sum()} of 24")
print(f"Optimised policies meeting it: {(allbo['opt_csl'] >= SL).sum()} of 24")
print(f"Total expected cost: baseline £{allbo['baseline_cost'].sum():,.0f}/yr → optimised £{allbo['opt_cost'].sum():,.0f}/yr "
      f"({allbo['opt_cost'].sum() / allbo['baseline_cost'].sum() - 1:+.1%})")
print(f"Median optimised ROP vs formula ROP: {(allbo['opt_rop'] / allbo['formula_rop'] - 1).median():+.1%}")
""")
md("""
**Interpretation across the range.** Run for all 24 SKUs (60 trials each, no capacity or budget limit), the textbook baseline meets the 95 % service target on fresh simulations for only {BASE_MEET} of 24 SKUs. The Bayesian-optimised policies meet it for {OPT_MEET} of 24, and total expected cost changes from £{ALL_BASE_COST:,.0f} to £{ALL_OPT_COST:,.0f} per year ({ALL_CHANGE:+.1%}). In this case the AI-driven sensitivity analysis does more than tune: it closes most of a hidden service gap and lowers cost at the same time. The {OPT_MISS} SKU(s) still slightly short on fresh paths would need more trials, or a small safety margin on the service constraint, because the optimiser sees only 120 simulated years per trial.
""")

md("""
## 10. Inventory intelligence: what to order this week (scenario)
Stock on hand is **not** in the dataset, so this section uses a clearly labelled **simulated** starting position: 2–10 weeks of forecast cover, with the same seed as the dashboard. The status rules are: Critical if position ≤ SS; Reorder Now if position ≤ ROP; Monitor if position ≤ ROP + 2 weeks of demand; otherwise Healthy.
""")
code("""
rng = np.random.default_rng(7)
stock = pd.DataFrame({"sku": params["sku"], "on_hand": np.round(params["d_bar"].values * rng.uniform(2.0, 10.0, len(params))),
                      "on_order": 0.0, "backorders": 0.0})
pt = I.sku_policy_table(params.drop(columns=["ROP"]).merge(stock, on="sku"), SL, LEAD, LEAD_SD, ORDER_COST, HOLD)
print("SIMULATED stock scenario. Status counts:", pt["status"].value_counts().to_dict())
pt[["sku", "description", "on_hand", "safety_stock", "rop", "status", "order_qty", "action"]].round(0).sort_values("rop", ascending=False).head(12)
""")

md("""
## 11. Results, business interpretation and conclusion

| Workstream | Key result |
|---|---|
| 1 · Forecasting | {BEST_MODEL} selected automatically on validation; test WAPE {BEST_TEST_WAPE:.1%} vs {NAIVE_TEST_WAPE:.1%} naive. Weekly refitting cuts error from {STATIC_WAPE:.1%} to {UPDATED_WAPE:.1%} through the peak. |
| 2 · Dynamic safety stock | Formula SS for {EX}: {EX_SS:,.0f} units. The Monte Carlo simulation needs {EX_SS_MC:,.0f}, and the formula ROP gives only {EX_CSL_FORMULA:.1%} service. {N_MORE_BUFFER}/24 SKUs need more buffer than the formula. |
| 3 · Bayesian optimisation | Service target met for {OPT_MEET}/24 SKUs (baseline {BASE_MEET}/24); total cost {ALL_CHANGE:+.1%}. |

**What this means for the business.**
1. **Peak season is predictable.** Forecasting by week, with seasonality and fresh data, lets buffers rise ahead of October–November instead of after the first stockouts.
2. **The textbook formula is a starting point, not the answer.** With skewed B2B demand, it quietly under-protects most SKUs. Simulation makes the gap visible and sizes the real buffer, including under supplier-delay and demand-spike scenarios.
3. **Optimise the policy, not one parameter.** Choosing ROP and Q together under explicit service, capacity and budget constraints finds policies that are both cheaper and safer. Searching this two-dimensional, noisy space efficiently is exactly what Bayesian optimisation is for.

**Limitations.**
- Lead times, costs and stock levels are assumptions, because the dataset has none.
- Promotions are proxied by a derived price-cut flag.
- Only the 24 most regular SKUs are modelled; intermittent items need Croston-type methods.
- Updates are batch refreshes, not a real-time feed.

**Conclusion.** All three steps of the exercise are implemented on real data: machine-learning forecasting with automatic model selection and continuous updating; probabilistic and Monte Carlo safety stock with risk scenarios; and Bayesian optimisation of the reorder policy under constraints. The same engine powers an interactive Streamlit dashboard for live use.
""")
code("""
summary = {
    "ROWS_RAW": quality["rows_raw"], "CANCEL": quality["cancellation_rows"], "NONPOS": quality["non_positive_qty_or_price_rows"],
    "NONPROD": quality["non_product_code_rows"], "DUPS": quality["exact_duplicates"], "ROWS_CLEAN": quality["rows_clean"],
    "SKUS_CLEAN": quality["skus_clean"], "MISSING_CUST": quality["missing_by_column"]["CustomerID"],
    "NOV_INDEX": float(idx.loc[11]), "OCT_INDEX": float(idx.loc[10]), "DEC_INDEX": float(idx.loc[12]),
    "VAL_START": f"{sp['val_weeks'][0]:%d %b %Y}", "VAL_END": f"{sp['val_weeks'][1]:%d %b %Y}",
    "TEST_START": f"{sp['test_weeks'][0]:%d %b %Y}", "TEST_END": f"{sp['test_weeks'][1]:%d %b %Y}",
    "BEST_MODEL": best_model, "BEST_VAL_WAPE": float(out["val_metrics"].set_index("model").loc[best_model, "WAPE"]),
    "BEST_TEST_WAPE": float(tm.loc["AI-selected (global)", "WAPE"]), "NAIVE_TEST_WAPE": float(tm.loc["Naive", "WAPE"]),
    "SNAIVE_TEST_WAPE": float(tm.loc["Seasonal naive", "WAPE"]), "BEST_TEST_ANY": tm.drop(index=[i for i in tm.index if i.startswith("AI")])["WAPE"].idxmin(),
    "BEST_TEST_ANY_WAPE": float(tm.drop(index=[i for i in tm.index if i.startswith("AI")])["WAPE"].min()),
    "PER_SKU_WAPE": float(tm.loc["AI-selected (per SKU)", "WAPE"]), "TOP_FEATURE": names.get(imp.idxmax(), imp.idxmax()),
    "CUT_WEEK": f"{CUT:%d %b %Y}", "STATIC_WAPE": float(cmp_.loc["Static (fit once)", "WAPE"]),
    "UPDATED_WAPE": float(cmp_.loc["Updated weekly", "WAPE"]),
    "UPD_IMPROVE": float(1 - cmp_.loc["Updated weekly", "WAPE"] / cmp_.loc["Static (fit once)", "WAPE"]),
    "EX": EX, "EX_NAME": f"{EX} ({params.set_index('sku').loc[EX, 'description']})",
    "EX_SS": float(params.set_index("sku").loc[EX, "SS_forecast_err"]), "EX_ROP": float(params.set_index("sku").loc[EX, "ROP"]),
    "EX_DEM_SHARE": float(dv / (dv + lv)), "EX_LT_SHARE": float(lv / (dv + lv)),
    "MED_SIGMA_FC": float(params["sigma_d"].median()), "MED_SIGMA_HIST": float(params["hist_sd"].median()),
    "EX_SS_MC": float(mc["ss_mc"]), "EX_CSL_FORMULA": float(mc["cycle_service_level"]),
    "N_MORE_BUFFER": int((allmc["diff %"] > 0).sum()), "MED_EXTRA": float(allmc["diff %"].median()),
    "DELAY_PSTOCKOUT": float(scen.set_index("scenario").loc["Delayed supplier (+1 wk, 2x var.)", "P(stockout) at current ROP"]),
    "SPIKE_PSTOCKOUT": float(scen.set_index("scenario").loc["Demand spike (+30% mean)", "P(stockout) at current ROP"]),
    "SS_RHO06": float(rho_ss[2]), "BASE_ROP": float(bval["rop"]), "BASE_Q": float(bval["q"]), "BASE_CSL": float(bval["cycle_service_level"]),
    "BASE_COST": float(bval["total_cost"]), "OPT_ROP": float(bv["rop"]), "OPT_Q": float(bv["q"]), "OPT_CSL": float(bv["cycle_service_level"]),
    "OPT_COST": float(bv["total_cost"]), "OPT_CHANGE": float(-saving),
    "BASE_VERDICT": "meets" if bval["cycle_service_level"] >= SL else "misses",
    "BASE_SHORT": float(bval["shortage_cost"]), "OPT_SHORT": float(bv["shortage_cost"]),
    "BASE_HOLD": float(bval["holding_cost"]), "OPT_HOLD": float(bv["holding_cost"]),
    "BASE_ORD": float(bval["ordering_cost"]), "OPT_ORD": float(bv["ordering_cost"]),
    "OPT_MISS": int((allbo["opt_csl"] < SL).sum()),
    "BASE_MEET": int((allbo["baseline_csl"] >= SL).sum()), "OPT_MEET": int((allbo["opt_csl"] >= SL).sum()),
    "ALL_BASE_COST": float(allbo["baseline_cost"].sum()), "ALL_OPT_COST": float(allbo["opt_cost"].sum()),
    "ALL_CHANGE": float(allbo["opt_cost"].sum() / allbo["baseline_cost"].sum() - 1),
}
summary = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v) for k, v in summary.items()}
json.dump(summary, open("nb_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print(f"\\nTotal notebook run time: {time.time() - T0:.0f} s")
""")
md("""
## References
1. Roy, S. K. (2026). *AI in Operations: Session II lecture notes*, slide 35 (Exercise). S P Jain School of Global Management, Dubai.
2. Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D (CC BY 4.0).
3. Quadros, A. (2020). *onlineretail2: Online Retail II dataset* (R package, CC0). https://github.com/allanvc/onlineretail2
4. Akiba, T. et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD '19*.
5. Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
6. Silver, E. A., Pyke, D. F. & Thomas, D. J. (2017). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
if len(sys.argv) > 2:
    import re
    vals = json.load(open(sys.argv[2]))
    pat = re.compile(r"\{([A-Z][A-Z0-9_]*)(?::([^{}]*))?\}")
    def repl(m):
        k, spec = m.group(1), m.group(2)
        if k not in vals:
            raise KeyError(k)
        return format(vals[k], spec or "")
    for c in nb.cells:
        if c.cell_type == "markdown" and not c.source.startswith(f"**{IDENT}"):
            c.source = pat.sub(repl, c.source)
nbf.write(nb, sys.argv[1])
print("written", sys.argv[1], len(cells), "cells")
