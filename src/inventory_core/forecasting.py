"""Demand forecasting: baselines, statistical models and ML regression, with chronological
rolling-origin backtesting, model selection and prediction intervals.

All models forecast weekly demand per SKU. A forecast made at an *origin* week uses only
data up to and including that week (no look-ahead), except the price columns, which are
treated as planned prices known in advance (as in the M5 competition). For the forward
forecast beyond the data, the last observed price is carried forward.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

MODEL_NAMES = ["Naive", "Seasonal naive", "Moving average (4w)", "ETS (damped trend)",
               "ARIMA + Fourier", "Ridge regression", "Gradient boosting"]
STAT_MODELS = MODEL_NAMES[:5]
ML_MODELS = MODEL_NAMES[5:]
FOURIER_K = 3


# ---------------------------------------------------------------- helpers
def wide_demand(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(index="week", columns="sku", values="demand").sort_index()


def fourier(weeks: pd.DatetimeIndex, k: int = FOURIER_K) -> pd.DataFrame:
    t = 2 * np.pi * weeks.dayofyear.values / 365.25
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin{i}"] = np.sin(i * t)
        cols[f"cos{i}"] = np.cos(i * t)
    return pd.DataFrame(cols, index=weeks)


def future_weeks(last_week: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last_week + pd.Timedelta(weeks=1), periods=h, freq="W-MON")


def holidays_for(weeks: pd.DatetimeIndex) -> np.ndarray:
    from .data import uk_holidays_per_week
    return uk_holidays_per_week(weeks).values


# ---------------------------------------------------------------- statistical models
def _stat_forecast(model: str, y: pd.Series, h: int, hol_hist=None, hol_fut=None) -> np.ndarray:
    """y: weekly demand up to the origin (DatetimeIndex). Returns h forecasts (>= 0)."""
    vals = y.values.astype(float)
    if model == "Naive":
        f = np.repeat(vals[-1], h)
    elif model == "Seasonal naive":
        n = len(vals)
        f = np.array([vals[n - 52 + (i % 52)] if n >= 52 else vals[-1] for i in range(h)])
    elif model == "Moving average (4w)":
        f = np.repeat(vals[-4:].mean(), h)
    elif model == "ETS (damped trend)":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(vals, trend="add", damped_trend=True,
                                     initialization_method="estimated").fit(optimized=True)
        f = m.forecast(h)
    elif model == "ARIMA + Fourier":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        fut = future_weeks(y.index[-1], h)
        xh = fourier(y.index).assign(hol=hol_hist if hol_hist is not None else holidays_for(y.index))
        xf = fourier(fut).assign(hol=hol_fut if hol_fut is not None else holidays_for(fut))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(vals, exog=xh.values, order=(1, 0, 1), trend="c",
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
        f = m.forecast(h, exog=xf.values)
    else:
        raise ValueError(model)
    return np.clip(np.asarray(f, dtype=float), 0, None)


# ---------------------------------------------------------------- ML regression (global, direct)
def _feature_rows(panel: pd.DataFrame, origin_idx: int, horizons, weeks: pd.DatetimeIndex,
                  W: pd.DataFrame, P: pd.DataFrame, C: pd.DataFrame, extra_weeks=None) -> pd.DataFrame:
    """Feature rows for every SKU for targets origin+h. Targets are scaled by the SKU's
    13-week level at the origin so one global model can learn across SKUs."""
    rows = []
    o_week = weeks[origin_idx]
    hist = W.iloc[: origin_idx + 1]
    lvl4 = hist.iloc[-4:].mean()
    lvl13 = hist.iloc[-13:].mean() + 1.0
    trend = (hist.iloc[-4:].mean() + 1) / (hist.iloc[-13:].mean() + 1)
    all_weeks = weeks if extra_weeks is None else weeks.append(extra_weeks)
    four = fourier(all_weeks)
    hol = pd.Series(holidays_for(all_weeks), index=all_weeks)
    for h in horizons:
        t = o_week + pd.Timedelta(weeks=h)
        lag_idx = origin_idx + h - 52
        lag52 = W.iloc[lag_idx] if lag_idx >= 0 else lvl13 - 1
        price_t = P.loc[t] if t in P.index else P.iloc[origin_idx]       # planned / carried-forward price
        cut_t = C.loc[t] if t in C.index else C.iloc[origin_idx]
        med_p = P.iloc[: origin_idx + 1].median()
        df = pd.DataFrame({
            "sku": W.columns, "origin": o_week, "week": t, "h": h,
            "lvl4_ratio": (lvl4 / lvl13).values, "trend": trend.values,
            "lag52_ratio": (lag52 / lvl13).values, "price_ratio": (price_t / med_p).values,
            "price_cut": cut_t.values, "hol": hol.loc[t], "scale": lvl13.values,
        })
        for c in four.columns:
            df[c] = four.loc[t, c]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


ML_FEATURES = ["h", "lvl4_ratio", "trend", "lag52_ratio", "price_ratio", "price_cut", "hol"] + \
              [f"{p}{i}" for i in range(1, FOURIER_K + 1) for p in ("sin", "cos")]


def _ml_train_set(panel, upto_idx: int, weeks, W, P, C, max_h: int = 8, min_hist: int = 26):
    rows = []
    for o in range(min_hist, upto_idx):
        hs = [h for h in range(1, max_h + 1) if o + h <= upto_idx]
        if not hs:
            continue
        r = _feature_rows(panel, o, hs, weeks, W, P, C)
        r["y"] = [W.loc[w, s] for w, s in zip(r["week"], r["sku"])]
        rows.append(r)
    d = pd.concat(rows, ignore_index=True)
    d["y_ratio"] = d["y"] / d["scale"]
    return d


def _fit_ml(model: str, train: pd.DataFrame, seed: int = 42):
    if model == "Ridge regression":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
    m.fit(train[ML_FEATURES], train["y_ratio"])
    return m


# ---------------------------------------------------------------- backtesting
def make_splits(weeks: pd.DatetimeIndex, horizon: int = 4, n_val: int = 3, n_test: int = 2):
    """Chronological rolling origins. Test origins cover the final n_test*horizon weeks;
    validation origins cover the n_val*horizon weeks before that. Returns index positions."""
    n = len(weeks)
    test = [n - 1 - horizon * (n_test - i) for i in range(n_test)]
    val = [test[0] - horizon * (n_val - i) for i in range(n_val)]
    return val, test


def backtest(panel: pd.DataFrame, origins: list[int], horizon: int = 4, models=None,
             seed: int = 42) -> pd.DataFrame:
    """Rolling-origin backtest. Returns long frame sku, origin, week, h, model, actual, forecast."""
    models = models or MODEL_NAMES
    W = wide_demand(panel)
    P = panel.pivot(index="week", columns="sku", values="avg_price").sort_index()
    C = panel.pivot(index="week", columns="sku", values="price_cut").sort_index()
    weeks = W.index
    hol_all = holidays_for(weeks)
    out = []
    for o in origins:
        hs = list(range(1, horizon + 1))
        tw = weeks[o + 1: o + 1 + horizon]
        for m in [m for m in models if m in STAT_MODELS]:
            for s in W.columns:
                f = _stat_forecast(m, W[s].iloc[: o + 1], horizon, hol_all[: o + 1], hol_all[o + 1: o + 1 + horizon])
                out.append(pd.DataFrame({"sku": s, "origin": weeks[o], "week": tw, "h": hs, "model": m,
                                         "actual": W[s].iloc[o + 1: o + 1 + horizon].values, "forecast": f}))
        ml = [m for m in models if m in ML_MODELS]
        if ml:
            train = _ml_train_set(panel, o, weeks, W, P, C, max_h=max(8, horizon))
            X = _feature_rows(panel, o, hs, weeks, W, P, C)
            for m in ml:
                fit = _fit_ml(m, train, seed)
                X["forecast"] = np.clip(fit.predict(X[ML_FEATURES]) * X["scale"], 0, None)
                X["actual"] = [W.loc[w, s] for w, s in zip(X["week"], X["sku"])]
                out.append(X[["sku", "origin", "week", "h", "forecast", "actual"]].assign(model=m))
    return pd.concat(out, ignore_index=True)


def metrics(bt: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    """MAE, RMSE, WAPE and MAPE (MAPE skips weeks with zero actual demand)."""
    def f(g):
        e = g["forecast"] - g["actual"]
        nz = g["actual"] > 0
        return pd.Series({
            "MAE": e.abs().mean(),
            "RMSE": np.sqrt((e ** 2).mean()),
            "WAPE": e.abs().sum() / max(g["actual"].abs().sum(), 1e-9),
            "MAPE": (e[nz].abs() / g.loc[nz, "actual"]).mean() if nz.any() else np.nan,
            "Bias": e.sum() / max(g["actual"].sum(), 1e-9),
            "n": len(g),
        })
    return bt.groupby(list(by)).apply(f, include_groups=False).reset_index()


def select_best(val_bt: pd.DataFrame, strategy: str = "global") -> pd.DataFrame:
    """Automatic model selection on the validation window.
    strategy="global": the one model with the lowest pooled validation WAPE is used for every SKU
                       (more stable when each SKU has only a few validation weeks).
    strategy="per_sku": each SKU gets its own lowest-WAPE model (ties broken by RMSE)."""
    m = metrics(val_bt, by=("sku", "model"))
    if strategy == "global":
        g = metrics(val_bt).sort_values(["WAPE", "RMSE"]).iloc[0]["model"]
        best = m[m["model"] == g]
    else:
        best = m.sort_values(["sku", "WAPE", "RMSE"]).groupby("sku").head(1)
    return best[["sku", "model", "WAPE", "RMSE"]].rename(columns={"model": "best_model",
                                                                  "WAPE": "val_WAPE", "RMSE": "val_RMSE"})


def evaluate_selected(test_bt: pd.DataFrame, best: pd.DataFrame, label: str = "AI-selected") -> pd.DataFrame:
    sel = test_bt.merge(best[["sku", "best_model"]], left_on=["sku", "model"], right_on=["sku", "best_model"])
    return sel.drop(columns="best_model").assign(model=label)


# ---------------------------------------------------------------- forward forecast
def forecast_future(panel: pd.DataFrame, best: pd.DataFrame, horizon: int, val_bt: pd.DataFrame,
                    level: float = 0.90, seed: int = 42) -> pd.DataFrame:
    """Refit each SKU's selected model on all data and forecast `horizon` weeks ahead.
    Interval half-width = z * RMSE of that model at the same horizon in validation
    (horizons beyond the validated ones use the last validated RMSE scaled by sqrt(h/h_max))."""
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    W = wide_demand(panel)
    P = panel.pivot(index="week", columns="sku", values="avg_price").sort_index()
    C = panel.pivot(index="week", columns="sku", values="price_cut").sort_index()
    weeks = W.index
    o = len(weeks) - 1
    fut = future_weeks(weeks[-1], horizon)
    hol_all = holidays_for(weeks)
    hol_fut = holidays_for(fut)
    rm = val_bt.assign(se=(val_bt["forecast"] - val_bt["actual"]) ** 2) \
               .groupby(["sku", "model", "h"])["se"].mean().pipe(np.sqrt)
    out = []
    ml_needed = set(best["best_model"]) & set(ML_MODELS)
    ml_pred = {}
    if ml_needed:
        train = _ml_train_set(panel, o, weeks, W, P, C, max_h=max(8, horizon))
        X = _feature_rows(panel, o, range(1, horizon + 1), weeks, W, P, C, extra_weeks=fut)
        for m in ml_needed:
            fit = _fit_ml(m, train, seed)
            X["f"] = np.clip(fit.predict(X[ML_FEATURES]) * X["scale"], 0, None)
            ml_pred[m] = X.set_index(["sku", "h"])["f"]
    for _, r in best.iterrows():
        s, m = r["sku"], r["best_model"]
        if m in ML_MODELS:
            f = np.array([ml_pred[m].loc[(s, h)] for h in range(1, horizon + 1)])
        else:
            f = _stat_forecast(m, W[s], horizon, hol_all, hol_fut)
        hmax = int(rm.loc[(s, m)].index.max())
        sig = np.array([rm.loc[(s, m, min(h, hmax))] * (np.sqrt(h / hmax) if h > hmax else 1.0)
                        for h in range(1, horizon + 1)])
        out.append(pd.DataFrame({"sku": s, "week": fut, "h": range(1, horizon + 1), "model": m,
                                 "forecast": f, "lo": np.clip(f - z * sig, 0, None), "hi": f + z * sig,
                                 "sigma": sig}))
    return pd.concat(out, ignore_index=True)


def run_pipeline(panel: pd.DataFrame, horizon: int = 8, bt_horizon: int = 4, models=None, seed: int = 42,
                 strategy: str = "global"):
    """Full forecasting workflow used by the dashboard and the notebook."""
    weeks = pd.DatetimeIndex(sorted(panel["week"].unique()))
    val_o, test_o = make_splits(weeks, bt_horizon)
    val_bt = backtest(panel, val_o, bt_horizon, models, seed)
    test_bt = backtest(panel, test_o, bt_horizon, models, seed)
    best = select_best(val_bt, strategy)
    sel_test = evaluate_selected(test_bt, best, "AI-selected (global)" if strategy == "global" else "AI-selected (per SKU)")
    per_sku = evaluate_selected(test_bt, select_best(val_bt, "per_sku"), "AI-selected (per SKU)")
    if strategy == "global":
        sel_test = pd.concat([sel_test, per_sku])
    fc = forecast_future(panel, best, horizon, val_bt, seed=seed)
    return {
        "val_bt": val_bt, "test_bt": test_bt, "best": best,
        "val_metrics": metrics(val_bt), "test_metrics": pd.concat([metrics(test_bt), metrics(sel_test)],
                                                                  ignore_index=True),
        "sku_test_metrics": metrics(pd.concat([test_bt, sel_test]), by=("sku", "model")),
        "forecast": fc,
        "splits": {"train_end": weeks[val_o[0]], "val_weeks": (weeks[val_o[0] + 1], weeks[test_o[0]]),
                   "test_weeks": (weeks[test_o[0] + 1], weeks[-1])},
    }
