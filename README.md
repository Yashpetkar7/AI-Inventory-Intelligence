# AI Inventory Intelligence: Lab 2, AI in Operations

**Lab Activity 2 · Session 2 · Yash Petkar · AS25DXB021**
MAIB (Sept 2025 cohort) · S P Jain School of Global Management, Dubai · Lecturer: Dr. Sandip Kumar Roy

This project answers the Session II exercise *"AI Applications in Reorder Point & Inventory Management"* (slide 35):
1. **Demand forecasting** with ML models, seasonality, price and external factors, new-data intake and continuous updating.
2. **Dynamic safety stock** with the probabilistic formula `SS = Z·√(σ_D²·L + D̄²·σ_L²)` and Monte Carlo risk scenarios.
3. **AI-driven sensitivity analysis** with Bayesian optimisation (Optuna) of the reorder point and order quantity under constraints.

## What is inside

```
AI-Inventory-Intelligence/
├── src/inventory_core/        shared engine used by BOTH the dashboard and the notebook
│   ├── data.py                load, clean, validate uploads, weekly features, append/reconcile
│   ├── forecasting.py         7 models, rolling-origin backtests, model selection, intervals
│   ├── inventory.py           Z, safety stock, ROP, EOQ, status rules, recommendations
│   ├── simulation.py          Monte Carlo lead-time demand, scenarios, (s, nQ) policy simulation
│   └── optimization.py        Optuna TPE / GP Bayesian optimisation with constraints
├── apps/streamlit/            Streamlit dashboard (8 pages)
├── data/raw/                  onlineretail2.rda (UCI Online Retail II, SHA-256 checked)
├── data/processed/            weekly_panel.csv (24 SKUs × 104 weeks)
├── data/sample/               real transactions for the week of 5 Dec 2011, for testing uploads
├── tests/                     engine unit tests + Streamlit smoke test
├── submissions/Laboratory Work 2/
│   ├── lab2_session2_yashpetkar_AS25DXB021.ipynb   executed notebook (Colab-ready)
│   ├── lab2_summary_yashpetkar_AS25DXB021.pdf      one-page summary
│   ├── dataset_provenance.txt
│   └── lab2_LINK.txt                               empty: paste the Colab link here
├── build_notebook.py / make_pdf.py                 regenerate the submission files
└── requirements.txt
```

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run apps/streamlit/app.py
```

The first load trains and back-tests all models (about 30 seconds). After that, pages respond instantly.

| Page | What it does |
|---|---|
| Executive overview | KPI cards, demand history and forecast, inventory health, critical SKUs, AI recommendations, seasonality |
| Inventory intelligence | SKU table with status and actions, CSV export, *Optimise all SKUs*, conceptual 3D warehouse with 2D fallback |
| 1 · Demand forecasting | Daily / weekly / monthly views, forecast with 90 % intervals, model comparison (MAE, RMSE, WAPE, MAPE), test-week plots, drivers |
| 2a · Safety stock & ROP | Interactive formula calculator with every step shown, analytical vs Monte Carlo, sensitivity curves |
| 2b · Monte Carlo risk lab | Choose distributions, parameters, seed, correlation ρ and run count; histogram, CDF, stockout probability, expected shortage, fill rate, scenarios |
| 3 · Bayesian optimisation | Objective and constraints, TPE / GP sampler, convergence, search map, cost vs service, baseline vs optimised on fresh paths |
| Data management | Lineage table, replay feed of real held-back weeks, CSV upload with column mapping and validation, stock upload, update log |
| Reports & diagnostics | Accuracy report, formula vs MC per SKU, optimisation summary, data-quality waterfall, CSV downloads |

**Try adding data:** on *Data management*, press *Ingest next week*, or upload `data/sample/new_transactions_week_2011-12-05.csv`. Forecasts, safety stock, ROPs and recommendations then refresh.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. On share.streamlit.io choose **Create app → Deploy a public app from GitHub**, pick the repo, branch `main`, main file path `apps/streamlit/app.py`.
3. `requirements.txt` at the repo root is installed automatically; the theme comes from `.streamlit/config.toml`. The first load trains the models (about 1–2 minutes on the free tier).

## Run the notebook
Open the `.ipynb` in Google Colab and choose *Runtime → Run all*. The notebook writes the engine package itself (`%%writefile`), installs `pyreadr`, `holidays` and `optuna` if they are missing, downloads the data, and verifies its hash. Runtime is about 1 minute on CPU.

## Tests
```bash
python -m pytest -q tests/test_engine.py           # formula, MC reproducibility, optimiser constraints, validation, no look-ahead
cd apps/streamlit && python ../../tests/streamlit_smoke.py   # every dashboard page renders without exceptions
```

## Data, assumptions and privacy
- **Observed:** UCI Online Retail II (Chen, 2019, CC BY 4.0), 1,067,371 lines.
- **Derived:** weekly demand, average price, price-cut proxy, forecasts.
- **External:** UK bank holidays.
- **Assumed (labelled in the app):** lead time 3 ± 1 weeks, cost 60 % of price, holding 25 %/yr, £60 per order, shortage cost = price, MOQ, capacity, budget.
- **Simulated (labelled):** stock on hand until you upload real stock; all Monte Carlo paths (seeded).
- **Privacy:** the app runs locally, single user. Uploaded files live only in the browser session's memory. Results computed from them are cached per session, never globally. Nothing is sent to external services, and there are no sharing features, in line with the exercise rule *"No sharing of data with peers allowed"*.
